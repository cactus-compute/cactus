#include "model.h"
#include "../graph/graph.h"
#include "../npu/npu.h"
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <set>
#include <stdexcept>

namespace cactus {
namespace engine {

namespace {

bool qwen_theoretical_1bit_enabled() {
    const char* env = std::getenv("CACTUS_QWEN_V_THEORETICAL_1BIT");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    return env[0] != '0';
}

uint64_t qwen_theoretical_1bit_seed() {
    const char* env = std::getenv("CACTUS_QWEN_V_THEORETICAL_1BIT_SEED");
    if (env == nullptr || env[0] == '\0') {
        return 1234;
    }
    return static_cast<uint64_t>(std::strtoull(env, nullptr, 10));
}

} // namespace

QwenModel::QwenModel() : Model() {}

QwenModel::QwenModel(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
}

void QwenModel::post_init() {
    if (kv_cache_.precision != Precision::INT8 || !qwen_theoretical_1bit_enabled()) {
        return;
    }

    kv_cache_.set_value_quantization(KVValueQuantization::THEORETICAL_1BIT, qwen_theoretical_1bit_seed());
    kv_cache_.set_window_size(kv_cache_.window_size, kv_cache_.sink_size);
}

void QwenModel::load_weights_to_graph(CactusGraph* gb) {
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
    }

    if (npu::is_npu_available()) {
        std::string npu_prefill_path = model_folder_path_ + "/model.mlpackage";
        load_npu_prefill(npu_prefill_path);
    }
}

size_t QwenModel::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                 ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];

    size_t q_proj = 0;
    size_t k_proj = 0;
    size_t v_proj = 0;
    const size_t batch_seq = gb->get_output_buffer(normalized_input).shape[0];
    const bool use_fused_qkv = (backend == ComputeBackend::CPU && batch_seq == 1);
    if (use_fused_qkv) {
        const size_t q_dim = config_.attention_heads * config_.attention_head_dim;
        const size_t kv_dim = config_.attention_kv_heads * config_.attention_head_dim;
        const size_t qkv = gb->matmul_concat(normalized_input,
                                             {layer.attn_q_weight, layer.attn_k_weight, layer.attn_v_weight},
                                             true, backend);
        q_proj = gb->slice(qkv, 1, 0, q_dim);
        k_proj = gb->slice(qkv, 1, q_dim, kv_dim);
        v_proj = gb->slice(qkv, 1, q_dim + kv_dim, kv_dim);
    } else {
        q_proj = gb->matmul(normalized_input, layer.attn_q_weight, true, backend);
        k_proj = gb->matmul(normalized_input, layer.attn_k_weight, true, backend);
        v_proj = gb->matmul(normalized_input, layer.attn_v_weight, true, backend);
    }

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
            kv_cache_.current_seq_len, num_kv_heads, head_dim, 0, 0,
            kv_cache_.get_value_inverse_rotation_scaled(layer_idx),
            kv_cache_.get_value_quantization() == KVValueQuantization::THEORETICAL_1BIT
        );
    } else {
        attn_output_4d = gb->attention(q_proj_4d, k_proj_4d, v_proj_4d, attention_scale_, position_offset);
    }

    auto attn_output = gb->reshape(attn_output_4d, {seq_len, config_.attention_head_dim * config_.attention_heads});
    return gb->matmul(attn_output, layer.attn_output_weight, true, backend);
}


size_t QwenModel::build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                           ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];
    size_t gate_output = 0;
    size_t up_output = 0;
    const size_t batch_seq = gb->get_output_buffer(normalized_h).shape[0];
    const bool use_fused_gate_up = (backend == ComputeBackend::CPU && batch_seq == 1);
    if (use_fused_gate_up) {
        const size_t inter_dim = config_.ffn_intermediate_dim;
        const size_t gate_up = gb->matmul_concat(normalized_h,
                                                 {layer.ffn_gate_weight, layer.ffn_up_weight},
                                                 true, backend);
        gate_output = gb->slice(gate_up, 1, 0, inter_dim);
        up_output = gb->slice(gate_up, 1, inter_dim, inter_dim);
    } else {
        gate_output = gb->matmul(normalized_h, layer.ffn_gate_weight, true, backend);
        up_output = gb->matmul(normalized_h, layer.ffn_up_weight, true, backend);
    }
    size_t gate_silu = gb->silu(gate_output);
    size_t gated = gb->multiply(gate_silu, up_output);
    return gb->matmul(gated, layer.ffn_down_weight, true, backend);
}


size_t QwenModel::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                         ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto normalized_input = gb->rms_norm(hidden, layer.input_layernorm_weight, config_.layer_norm_eps);
    auto attn_output = build_attention(gb, normalized_input, layer_idx, backend, use_cache, position_offset);
    auto after_attention = gb->add(hidden, attn_output);
    auto normalized_after_attention = gb->rms_norm(after_attention, layer.post_attention_layernorm_weight, config_.layer_norm_eps);
    auto mlp_output = build_mlp(gb, normalized_after_attention, layer_idx, backend);
    return gb->add(after_attention, mlp_output);
}


size_t QwenModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
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

}
}
