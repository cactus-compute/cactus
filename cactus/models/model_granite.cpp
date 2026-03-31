#include "model.h"
#include "../graph/graph.h"
#include "../npu/npu.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <set>

namespace cactus {
namespace engine {

GraniteModel::GraniteModel() : Model() {}

GraniteModel::GraniteModel(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
    residual_multiplier_ = config.residual_multiplier;
    logits_scaling_ = config.logits_scaling;
    embedding_multiplier_ = config.embedding_multiplier;
}

void GraniteModel::load_weights_to_graph(CactusGraph* gb) {
    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    weight_nodes_.output_norm_weight = gb->mmap_weights(model_folder_path_ + "/output_norm.weights");

    if (config_.tie_word_embeddings) {
        weight_nodes_.output_weight = embedding_node_id_;
        output_weight_node_id_ = embedding_node_id_;
    } else {
        weight_nodes_.output_weight = gb->mmap_weights(model_folder_path_ + "/output_weight.weights");
        output_weight_node_id_ = weight_nodes_.output_weight;
    }

    for (uint32_t i = 0; i < config_.num_layers; i++) {
        auto& layer = weight_nodes_.layers[i];
        std::string layer_prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";
        layer.attn_q_weight = gb->mmap_weights(layer_prefix + "attn_q.weights");
        layer.attn_k_weight = gb->mmap_weights(layer_prefix + "attn_k.weights");
        layer.attn_v_weight = gb->mmap_weights(layer_prefix + "attn_v.weights");
        layer.attn_output_weight = gb->mmap_weights(layer_prefix + "attn_output.weights");
        layer.input_layernorm_weight = gb->mmap_weights(layer_prefix + "input_norm.weights");
        layer.ffn_gate_weight = gb->mmap_weights(layer_prefix + "ffn_gate.weights");
        layer.ffn_up_weight = gb->mmap_weights(layer_prefix + "ffn_up.weights");
        layer.ffn_down_weight = gb->mmap_weights(layer_prefix + "ffn_down.weights");
        layer.post_attention_layernorm_weight = gb->mmap_weights(layer_prefix + "post_attn_norm.weights");
    }

    if (npu::is_npu_available()) {
        std::string npu_prefill_path = model_folder_path_ + "/model.mlpackage";
        load_npu_prefill(npu_prefill_path);
    }
}

size_t GraniteModel::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                     ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];

    auto q_proj = gb->matmul(normalized_input, layer.attn_q_weight, true, backend);
    auto k_proj = gb->matmul(normalized_input, layer.attn_k_weight, true, backend);
    auto v_proj = gb->matmul(normalized_input, layer.attn_v_weight, true, backend);

    const auto& q_shape = gb->get_output_buffer(q_proj).shape;
    size_t batch_seq = q_shape[0];
    size_t seq_len = batch_seq;
    size_t num_kv_heads = config_.attention_kv_heads;
    size_t head_dim = config_.attention_head_dim;

    auto q_proj_4d = gb->reshape(q_proj, {1, seq_len, config_.attention_heads, config_.attention_head_dim});
    auto k_proj_4d = gb->reshape(k_proj, {1, seq_len, config_.attention_kv_heads, config_.attention_head_dim});
    auto v_proj_4d = gb->reshape(v_proj, {1, seq_len, config_.attention_kv_heads, config_.attention_head_dim});

    if (config_.rope_theta > 0) {
        q_proj_4d = gb->rope(q_proj_4d, config_.rope_theta, position_offset);
        k_proj_4d = gb->rope(k_proj_4d, config_.rope_theta, position_offset);
    }

    size_t attn_output_4d;

    if (use_cache) {
        cache_k_output_nodes_[layer_idx] = k_proj_4d;
        cache_v_output_nodes_[layer_idx] = v_proj_4d;
    }

    if (use_cache && !kv_cache_.is_empty()) {
        attn_output_4d = gb->attention_int8_hybrid(
            q_proj_4d, k_proj_4d, v_proj_4d,
            attention_scale_, position_offset,
            kv_cache_.get_keys_int8(layer_idx),
            kv_cache_.get_values_int8(layer_idx),
            kv_cache_.get_key_scales(layer_idx),
            kv_cache_.get_value_scales(layer_idx),
            kv_cache_.current_seq_len, num_kv_heads, head_dim
        );
    } else {
        attn_output_4d = gb->attention(q_proj_4d, k_proj_4d, v_proj_4d, attention_scale_, position_offset);
    }

    auto attn_output = gb->reshape(attn_output_4d, {seq_len, config_.attention_head_dim * config_.attention_heads});
    return gb->matmul(attn_output, layer.attn_output_weight, true, backend);
}


size_t GraniteModel::build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                               ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];
    size_t gate_output = gb->matmul(normalized_h, layer.ffn_gate_weight, true, backend);
    size_t up_output = gb->matmul(normalized_h, layer.ffn_up_weight, true, backend);
    size_t gate_silu = gb->silu(gate_output);
    size_t gated = gb->multiply(gate_silu, up_output);
    return gb->matmul(gated, layer.ffn_down_weight, true, backend);
}


size_t GraniteModel::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                             ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];

    auto normalized_input = gb->rms_norm(hidden, layer.input_layernorm_weight, config_.layer_norm_eps);
    auto attn_output = build_attention(gb, normalized_input, layer_idx, backend, use_cache, position_offset);

    auto scaled_attn = gb->scalar_multiply(attn_output, residual_multiplier_);
    auto after_attention = gb->add(hidden, scaled_attn);

    auto normalized_for_mlp = gb->rms_norm(after_attention, layer.post_attention_layernorm_weight, config_.layer_norm_eps);
    auto mlp_output = build_mlp(gb, normalized_for_mlp, layer_idx, backend);

    auto scaled_mlp = gb->scalar_multiply(mlp_output, residual_multiplier_);
    return gb->add(after_attention, scaled_mlp);
}


size_t GraniteModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }

    if (tokens.empty()) {
        throw std::runtime_error("Token sequence cannot be empty");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    auto seq_len = static_cast<size_t>(tokens.size());

    size_t position_offset = use_cache ? kv_cache_.get_total_seq_len() : 0;

    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    auto input_node_id = gb->input({seq_len}, Precision::FP32);
    auto hidden = gb->embedding(embedding_node_id_, input_node_id);

    hidden = gb->scalar_multiply(hidden, embedding_multiplier_);

    std::vector<float> input_data(seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        input_data[i] = static_cast<float>(tokens[i]);
    }
    gb->set_input(input_node_id, input_data.data(), Precision::FP32);

    static std::set<uint32_t> skip_layers = {};
    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; layer_idx++) {
        if (skip_layers.count(layer_idx)) {
            continue;
        }
        hidden = build_transformer_block(gb, hidden, layer_idx, backend, use_cache, position_offset);
    }

    auto final_hidden = gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);

    return final_hidden;
}

uint32_t GraniteModel::decode(const std::vector<uint32_t>& tokens, float temperature, float top_p,
                               size_t top_k, const std::string& profile_file, float* out_entropy) {
    if (temperature < 0) {
        temperature = config_.default_temperature;
    }
    if (top_p < 0) {
        top_p = config_.default_top_p;
    }
    if (top_k == 0) {
        top_k = config_.default_top_k;
    }

    auto final_hidden = forward(tokens, true);

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    auto last_hidden = gb->index(final_hidden, tokens.size() - 1, 0);
    const auto& last_hidden_buf = gb->get_output_buffer(last_hidden);
    size_t hidden_dim = last_hidden_buf.shape[0];
    last_hidden = gb->reshape(last_hidden, {1, hidden_dim});

    auto logits_node_id = gb->matmul(last_hidden, output_weight_node_id_, true, backend);

    if (logits_scaling_ != 1.0f) {
        logits_node_id = gb->scalar_divide(logits_node_id, logits_scaling_);
    }

    auto sampled_token_id = gb->sample(logits_node_id, temperature, top_p, top_k, tool_constrainer_.get_bias());

    gb->execute(profile_file);

    if (out_entropy) {
        const auto& logits_buf = gb->get_output_buffer(logits_node_id);
        void* logits_ptr = gb->get_output(logits_node_id);
        size_t vocab_size = logits_buf.shape.back();
        size_t seq_len = 1;
        if (logits_buf.shape.size() >= 2) {
            seq_len = logits_buf.shape[logits_buf.shape.size() - 2];
        }
        size_t row_offset = (seq_len > 0 ? (seq_len - 1) * vocab_size : 0);

        std::vector<float> logits(vocab_size);
        if (logits_buf.precision == Precision::FP32) {
            float* src = static_cast<float*>(logits_ptr) + row_offset;
            std::copy(src, src + vocab_size, logits.begin());
        } else if (logits_buf.precision == Precision::FP16) {
            __fp16* src = static_cast<__fp16*>(logits_ptr) + row_offset;
            Quantization::fp16_to_fp32(src, logits.data(), vocab_size);
        } else {
            int8_t* src = static_cast<int8_t*>(logits_ptr) + row_offset;
            Quantization::int8_to_fp32(src, logits.data(), vocab_size, 1.0f);
        }

        float max_logit = *std::max_element(logits.begin(), logits.end());
        double sum_exp = 0.0;
        for (size_t i = 0; i < vocab_size; ++i) {
            sum_exp += std::exp(static_cast<double>(logits[i] - max_logit));
        }
        double log_sum_exp = static_cast<double>(max_logit) + std::log(sum_exp);

        double entropy = 0.0;
        for (size_t i = 0; i < vocab_size; ++i) {
            double log_prob = static_cast<double>(logits[i]) - log_sum_exp;
            double prob = std::exp(log_prob);
            if (prob > 1e-10) {
                entropy -= prob * log_prob;
            }
        }

        double max_entropy = std::log(static_cast<double>(vocab_size));
        *out_entropy = static_cast<float>(entropy / max_entropy);
    }

    post_execute_updates(gb, tokens.size());
    update_kv_cache(gb, tokens.size());

    auto* output_ptr = gb->get_output(sampled_token_id);
    return *static_cast<uint32_t*>(output_ptr);
}

}
}
