#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <stdexcept>

namespace cactus {
namespace engine {

YoutuModel::YoutuModel() : Model() {}

YoutuModel::YoutuModel(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
}

std::vector<size_t> YoutuModel::get_kv_layer_dims() const {
    return std::vector<size_t>(config_.num_layers, static_cast<size_t>(config_.qk_head_dim));
}

void YoutuModel::post_init() {
    std::vector<size_t> v_dims(config_.num_layers, static_cast<size_t>(config_.v_head_dim));
    std::vector<size_t> v_kv_heads(config_.num_layers, static_cast<size_t>(config_.attention_kv_heads));
    v_cache_.init(config_.num_layers, kv_cache_.max_seq_len,
                  v_dims, v_kv_heads, Precision::INT8);
    v_cache_.set_window_size(kv_cache_.window_size, kv_cache_.sink_size);
    cache_v_nodes_.resize(config_.num_layers, 0);
}

void YoutuModel::post_execute_updates(CactusGraph* gb, size_t seq_len) {
    v_cache_.update_from_graph(gb, cache_v_nodes_, cache_v_nodes_, seq_len,
                               config_.num_layers);
}

void YoutuModel::reset_cache() {
    kv_cache_.reset();
    v_cache_.reset();
}

void YoutuModel::load_weights_to_graph(CactusGraph* gb) {
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
        if (config_.q_lora_rank == 0) {
            layer.attn_q_weight = gb->mmap_weights(layer_prefix + "attn_q.weights");
        } else {
            layer.attn_q_a_weight = gb->mmap_weights(layer_prefix + "attn_q_a.weights");
            layer.attn_q_a_norm_weight = gb->mmap_weights(layer_prefix + "attn_q_a_norm.weights");
            layer.attn_q_b_weight = gb->mmap_weights(layer_prefix + "attn_q_b.weights");
        }
        layer.attn_kv_a_weight = gb->mmap_weights(layer_prefix + "attn_kv_a.weights");
        layer.attn_kv_a_norm_weight = gb->mmap_weights(layer_prefix + "attn_kv_a_norm.weights");
        layer.attn_kv_b_weight = gb->mmap_weights(layer_prefix + "attn_kv_b.weights");
        layer.attn_output_weight = gb->mmap_weights(layer_prefix + "attn_output.weights");
        if (config_.attention_bias) {
            layer.attn_q_a_bias = gb->mmap_weights(layer_prefix + "attn_q_a_bias.weights");
            layer.attn_kv_a_bias = gb->mmap_weights(layer_prefix + "attn_kv_a_bias.weights");
            layer.attn_output_bias = gb->mmap_weights(layer_prefix + "attn_output_bias.weights");
        }
        layer.input_layernorm_weight = gb->mmap_weights(layer_prefix + "input_norm.weights");
        layer.post_attention_layernorm_weight = gb->mmap_weights(layer_prefix + "post_attn_norm.weights");
        layer.ffn_gate_weight = gb->mmap_weights(layer_prefix + "ffn_gate.weights");
        layer.ffn_up_weight = gb->mmap_weights(layer_prefix + "ffn_up.weights");
        layer.ffn_down_weight = gb->mmap_weights(layer_prefix + "ffn_down.weights");
    }
}

static float yarn_get_mscale(float scale, float mscale) {
    if (scale <= 1.0f) return 1.0f;
    return 0.1f * mscale * logf(scale) + 1.0f;
}

size_t YoutuModel::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                   ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];

    const size_t seq_len = gb->get_output_buffer(normalized_input).shape[0];
    const size_t num_heads = config_.attention_heads;
    const size_t num_kv_heads = config_.attention_kv_heads;
    const size_t kv_lora = config_.kv_lora_rank;
    const size_t qk_head = config_.qk_head_dim;
    const size_t qk_nope = config_.qk_nope_head_dim;
    const size_t qk_rope = config_.qk_rope_head_dim;
    const size_t v_dim = config_.v_head_dim;
    const float eps = config_.layer_norm_eps;

    size_t q_full;
    if (config_.q_lora_rank == 0) {
        q_full = gb->matmul(normalized_input, layer.attn_q_weight, true, backend);
    } else {
        auto q_latent = gb->matmul(normalized_input, layer.attn_q_a_weight, true, backend);
        if (config_.attention_bias) {
            q_latent = gb->add(q_latent, layer.attn_q_a_bias);
        }
        q_latent = gb->rms_norm(q_latent, layer.attn_q_a_norm_weight, eps);
        q_full = gb->matmul(q_latent, layer.attn_q_b_weight, true, backend);
    }

    q_full = gb->reshape(q_full, {seq_len * num_heads, qk_head});
    auto q_nope = gb->slice(q_full, 1, 0, qk_nope);
    auto q_rope_raw = gb->slice(q_full, 1, qk_nope, qk_rope);

    q_rope_raw = gb->reshape(q_rope_raw, {1, seq_len, num_heads, qk_rope});
    size_t q_rope_rotated;
    if (config_.rope_interleave) {
        q_rope_rotated = gb->rope_gptj(q_rope_raw, config_.rope_theta, position_offset, qk_rope);
    } else {
        q_rope_rotated = gb->rope(q_rope_raw, config_.rope_theta, position_offset);
    }
    q_rope_rotated = gb->reshape(q_rope_rotated, {seq_len * num_heads, qk_rope});

    auto q_combined = gb->concat(q_nope, q_rope_rotated, -1);
    auto q_4d = gb->reshape(q_combined, {1, seq_len, num_heads, qk_head});

    auto kv_combined = gb->matmul(normalized_input, layer.attn_kv_a_weight, true, backend);
    if (config_.attention_bias) {
        kv_combined = gb->add(kv_combined, layer.attn_kv_a_bias);
    }
    auto kv_latent = gb->slice(kv_combined, 1, 0, kv_lora);
    auto k_rope_raw = gb->slice(kv_combined, 1, kv_lora, qk_rope);

    kv_latent = gb->rms_norm(kv_latent, layer.attn_kv_a_norm_weight, eps);

    auto kv_decoded = gb->matmul(kv_latent, layer.attn_kv_b_weight, true, backend);
    kv_decoded = gb->reshape(kv_decoded, {seq_len * num_kv_heads, qk_nope + v_dim});
    auto k_nope = gb->slice(kv_decoded, 1, 0, qk_nope);
    auto v_flat = gb->slice(kv_decoded, 1, qk_nope, v_dim);

    k_rope_raw = gb->reshape(k_rope_raw, {1, seq_len, 1, qk_rope});
    size_t k_rope_rotated;
    if (config_.rope_interleave) {
        k_rope_rotated = gb->rope_gptj(k_rope_raw, config_.rope_theta, position_offset, qk_rope);
    } else {
        k_rope_rotated = gb->rope(k_rope_raw, config_.rope_theta, position_offset);
    }
    std::vector<size_t> k_rope_copies(num_kv_heads, k_rope_rotated);
    auto k_rope_4d = gb->cat(k_rope_copies, 2);
    auto k_rope_flat = gb->reshape(k_rope_4d, {seq_len * num_kv_heads, qk_rope});

    auto k_combined = gb->concat(k_nope, k_rope_flat, -1);
    auto k_4d = gb->reshape(k_combined, {1, seq_len, num_kv_heads, qk_head});
    auto v_4d = gb->reshape(v_flat, {1, seq_len, num_kv_heads, v_dim});

    if (use_cache && layer_idx < cache_k_output_nodes_.size()) {
        cache_k_output_nodes_[layer_idx] = gb->reshape(k_4d, {seq_len * num_kv_heads, qk_head});
        cache_v_output_nodes_[layer_idx] = cache_k_output_nodes_[layer_idx];
        cache_v_nodes_[layer_idx] = gb->reshape(v_4d, {seq_len * num_kv_heads, v_dim});
    }

    float scale = 1.0f / sqrtf(static_cast<float>(qk_head));
    if (config_.rope_mscale_all_dim != 0.0f && config_.rope_scaling_factor > 1.0f) {
        float mscale = yarn_get_mscale(config_.rope_scaling_factor, config_.rope_mscale_all_dim);
        scale *= mscale * mscale;
    }

    size_t attn_4d;
    if (use_cache && !kv_cache_.is_empty()) {
        attn_4d = gb->attention_int8_hybrid(
            q_4d, k_4d, v_4d, scale, position_offset,
            kv_cache_.get_keys_int8(layer_idx),
            v_cache_.get_values_int8(layer_idx),
            kv_cache_.get_key_scales(layer_idx),
            v_cache_.get_value_scales(layer_idx),
            kv_cache_.current_seq_len, num_kv_heads, qk_head,
            kv_cache_.window_size, v_dim);
    } else {
        attn_4d = gb->attention(q_4d, k_4d, v_4d, scale, position_offset);
    }

    auto attn_out = gb->reshape(attn_4d, {seq_len, num_heads * v_dim});
    size_t out = gb->matmul(attn_out, layer.attn_output_weight, true, backend);
    if (config_.attention_bias) {
        out = gb->add(out, layer.attn_output_bias);
    }
    return out;
}


size_t YoutuModel::build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                              ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];
    size_t gate_output = gb->matmul(normalized_h, layer.ffn_gate_weight, true, backend);
    size_t up_output = gb->matmul(normalized_h, layer.ffn_up_weight, true, backend);
    size_t gate_silu = gb->silu(gate_output);
    size_t gated = gb->multiply(gate_silu, up_output);
    return gb->matmul(gated, layer.ffn_down_weight, true, backend);
}


size_t YoutuModel::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                           ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto normalized_input = gb->rms_norm(hidden, layer.input_layernorm_weight, config_.layer_norm_eps);
    auto attn_output = build_attention(gb, normalized_input, layer_idx, backend, use_cache, position_offset);
    auto after_attention = gb->add_clipped(hidden, attn_output);
    auto normalized_after_attention = gb->rms_norm(after_attention, layer.post_attention_layernorm_weight, config_.layer_norm_eps);
    auto mlp_output = build_mlp(gb, normalized_after_attention, layer_idx, backend);
    return gb->add_clipped(after_attention, mlp_output);
}


size_t YoutuModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }
    if (tokens.empty()) {
        throw std::runtime_error("Token sequence cannot be empty");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    const size_t seq_len = tokens.size();
    const size_t position_offset = use_cache ? kv_cache_.get_total_seq_len() : 0;

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

    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; layer_idx++) {
        hidden = build_transformer_block(gb, hidden, layer_idx, backend, use_cache, position_offset);
    }

    return gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);
}

}
}
