#include "model.h"
#include "../graph/graph.h"
#include "../npu/npu.h"
#include <cmath>
#include <stdexcept>
#include <set>

namespace cactus {
namespace engine {

GemmaModel::GemmaModel() : Model() {}

GemmaModel::GemmaModel(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
}

void GemmaModel::post_init() {
    kv_cache_.set_window_size(0, 0); 
}

void GemmaModel::load_weights_to_graph(CactusGraph* gb) {
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
        layer.attn_q_norm_weight = gb->mmap_weights(layer_prefix + "attn_q_norm.weights");
        layer.attn_k_norm_weight = gb->mmap_weights(layer_prefix + "attn_k_norm.weights");
        layer.ffn_gate_weight = gb->mmap_weights(layer_prefix + "ffn_gate.weights");
        layer.ffn_up_weight = gb->mmap_weights(layer_prefix + "ffn_up.weights");
        layer.ffn_down_weight = gb->mmap_weights(layer_prefix + "ffn_down.weights");
        layer.post_attention_layernorm_weight = gb->mmap_weights(layer_prefix + "post_attn_norm.weights");
        layer.pre_feedforward_layernorm_weight = gb->mmap_weights(layer_prefix + "pre_ffn_norm.weights");
        layer.post_feedforward_layernorm_weight = gb->mmap_weights(layer_prefix + "post_ffn_norm.weights");
    }

    if (npu::is_npu_available()) {
        std::string npu_prefill_path = model_folder_path_ + "/model.mlpackage";
        load_npu_prefill(npu_prefill_path);
    }
}

size_t GemmaModel::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                 ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];

    auto q_proj = gb->matmul(normalized_input, layer.attn_q_weight, true, backend);
    auto k_proj = gb->matmul(normalized_input, layer.attn_k_weight, true, backend);
    auto v_proj = gb->matmul(normalized_input, layer.attn_v_weight, true, backend);

    const auto& q_shape = gb->get_output_buffer(q_proj).shape;
    size_t batch_seq = q_shape[0];
    size_t num_heads = config_.attention_heads;
    size_t head_dim = config_.attention_head_dim;
    q_proj = gb->reshape(q_proj, {batch_seq * num_heads, head_dim});
    q_proj = gb->rms_norm(q_proj, layer.attn_q_norm_weight, config_.layer_norm_eps);
    q_proj = gb->reshape(q_proj, {batch_seq, num_heads * head_dim});

    size_t num_kv_heads = config_.attention_kv_heads;
    k_proj = gb->reshape(k_proj, {batch_seq * num_kv_heads, head_dim});
    k_proj = gb->rms_norm(k_proj, layer.attn_k_norm_weight, config_.layer_norm_eps);
    k_proj = gb->reshape(k_proj, {batch_seq, num_kv_heads * head_dim});

    size_t seq_len = batch_seq;

    auto q_proj_4d = gb->reshape(q_proj, {1, seq_len, config_.attention_heads, config_.attention_head_dim});
    auto k_proj_4d = gb->reshape(k_proj, {1, seq_len, config_.attention_kv_heads, config_.attention_head_dim});
    auto v_proj_4d = gb->reshape(v_proj, {1, seq_len, config_.attention_kv_heads, config_.attention_head_dim});

    bool is_global_attention = ((layer_idx + 1) % 6) == 0;

    if (config_.rope_theta > 0) {
        float rope_freq = is_global_attention ? 1000000.0f : 10000.0f;

        q_proj_4d = gb->rope(q_proj_4d, rope_freq, position_offset);
        k_proj_4d = gb->rope(k_proj_4d, rope_freq, position_offset);
    }

    if (use_cache) {
        cache_k_output_nodes_[layer_idx] = k_proj_4d;
        cache_v_output_nodes_[layer_idx] = v_proj_4d;
    }

    size_t attn_output_4d;
    size_t window_size = is_global_attention ? 0 : 512;

    if (use_cache && !kv_cache_.is_empty()) {
        attn_output_4d = gb->attention_int8_hybrid(
            q_proj_4d, k_proj_4d, v_proj_4d,
            attention_scale_, position_offset,
            kv_cache_.get_keys_int8(layer_idx),
            kv_cache_.get_values_int8(layer_idx),
            kv_cache_.get_key_scales(layer_idx),
            kv_cache_.get_value_scales(layer_idx),
            kv_cache_.current_seq_len, num_kv_heads, head_dim, window_size
        );
    } else {
        attn_output_4d = gb->attention(q_proj_4d, k_proj_4d, v_proj_4d, attention_scale_, position_offset, window_size);
    }
    auto attn_output = gb->reshape(attn_output_4d, {seq_len, config_.attention_head_dim * config_.attention_heads});
    return gb->matmul(attn_output, layer.attn_output_weight, true, backend);
}


size_t GemmaModel::build_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx,
                           ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];

    size_t gate_output = gb->matmul(input, layer.ffn_gate_weight, true, backend);
    size_t up_output = gb->matmul(input, layer.ffn_up_weight, true, backend);
    size_t gate_gelu = gb->gelu(gate_output);
    size_t gated = gb->multiply(gate_gelu, up_output);
    return gb->matmul(gated, layer.ffn_down_weight, true, backend);
}

size_t GemmaModel::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                         ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];

    auto normalized_input = gb->rms_norm(hidden, layer.input_layernorm_weight, config_.layer_norm_eps);
    auto attn_output = build_attention(gb, normalized_input, layer_idx, backend, use_cache, position_offset);

    auto normalized_attn = gb->rms_norm(attn_output, layer.post_attention_layernorm_weight, config_.layer_norm_eps);

    auto after_attention = gb->add_clipped(hidden, normalized_attn);

    auto pre_mlp_norm = gb->rms_norm(after_attention, layer.pre_feedforward_layernorm_weight, config_.layer_norm_eps);
    auto mlp_output = build_mlp(gb, pre_mlp_norm, layer_idx, backend);

    auto normalized_mlp = gb->rms_norm(mlp_output, layer.post_feedforward_layernorm_weight, config_.layer_norm_eps);

    return gb->add_clipped(after_attention, normalized_mlp);
}


size_t GemmaModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
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

    float embed_scale = std::sqrt(static_cast<float>(config_.hidden_dim));
    hidden = gb->scalar_multiply(hidden, embed_scale);

    static std::set<uint32_t> skip_layers = {};
    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; layer_idx++) {
        if (skip_layers.count(layer_idx)) {
            continue;
        }
        hidden = build_transformer_block(gb, hidden, layer_idx, backend, use_cache, position_offset);
    }

    auto final_hidden = gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);

    std::vector<float> input_data(seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        input_data[i] = static_cast<float>(tokens[i]);
    }
    gb->set_input(input_node_id, input_data.data(), Precision::FP32);

    return final_hidden;
}

}
}