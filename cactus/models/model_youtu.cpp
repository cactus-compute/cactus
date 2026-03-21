#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <stdexcept>

namespace cactus {
namespace engine {

YoutuModel::YoutuModel() : Model() {}

YoutuModel::YoutuModel(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
    layer_kv_nodes_.resize(config.num_layers);
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
        if (config_.attention_bias)
            q_latent = gb->add(q_latent, layer.attn_q_a_bias);
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
    if (config_.attention_bias)
        kv_combined = gb->add(kv_combined, layer.attn_kv_a_bias);
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

    if (layer_idx < layer_kv_nodes_.size()) {
        layer_kv_nodes_[layer_idx] = {
            gb->reshape(k_4d, {seq_len * num_kv_heads, qk_head}),
            gb->reshape(v_4d, {seq_len * num_kv_heads, v_dim})
        };
    }

    size_t k_attn = k_4d;
    size_t v_attn = v_4d;

    if (use_cache && kv_cache_valid_ && layer_idx < kv_cache_.size() && kv_cache_[layer_idx].len > 0) {
        const size_t cache_len = kv_cache_[layer_idx].len;

        auto ck = gb->input({cache_len * num_kv_heads, qk_head}, Precision::FP16);
        gb->set_input(ck, kv_cache_[layer_idx].k.data(), Precision::FP16);
        auto ck_4d = gb->reshape(ck, {1, cache_len, num_kv_heads, qk_head});
        k_attn = gb->concat(ck_4d, k_4d, 1);

        auto cv = gb->input({cache_len * num_kv_heads, v_dim}, Precision::FP16);
        gb->set_input(cv, kv_cache_[layer_idx].v.data(), Precision::FP16);
        auto cv_4d = gb->reshape(cv, {1, cache_len, num_kv_heads, v_dim});
        v_attn = gb->concat(cv_4d, v_4d, 1);
    }

    float scale = 1.0f / sqrtf(static_cast<float>(qk_head));
    if (config_.rope_mscale_all_dim != 0.0f && config_.rope_scaling_factor > 1.0f) {
        float mscale = yarn_get_mscale(config_.rope_scaling_factor, config_.rope_mscale_all_dim);
        scale *= mscale * mscale;
    }
    auto attn_4d = gb->attention(q_4d, k_attn, v_attn, scale, position_offset);

    auto attn_out = gb->reshape(attn_4d, {seq_len, num_heads * v_dim});
    size_t out = gb->matmul(attn_out, layer.attn_output_weight, true, backend);
    if (config_.attention_bias)
        out = gb->add(out, layer.attn_output_bias);
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

    auto seq_len = static_cast<size_t>(tokens.size());

    size_t position_offset = 0;
    if (use_cache && kv_cache_valid_ && !kv_cache_.empty()) {
        position_offset = kv_cache_[0].len;
    }

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


void YoutuModel::prefill(const std::vector<uint32_t>& tokens, size_t /*chunk_size*/, const std::string& profile_file) {
    if (!initialized_ || !graph_handle_) return;
    if (tokens.empty()) return;

    const bool warm = kv_cache_valid_;
    layer_kv_nodes_.resize(config_.num_layers);

    if (!warm) {
        token_history_ = tokens;
        kv_cache_.clear();
        kv_cache_.resize(config_.num_layers);
        kv_cache_valid_ = false;
        forward(token_history_, false);
    } else {
        for (auto t : tokens) token_history_.push_back(t);
        forward(tokens, true);
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->execute(profile_file);

    const size_t Hkv = config_.attention_kv_heads;
    const size_t qk_head = config_.qk_head_dim;
    const size_t v_dim = config_.v_head_dim;
    const size_t new_len = tokens.size();

    if (!warm) {
        for (uint32_t i = 0; i < config_.num_layers; i++) {
            const uint16_t* k_ptr = reinterpret_cast<const uint16_t*>(gb->get_output(layer_kv_nodes_[i].first));
            const uint16_t* v_ptr = reinterpret_cast<const uint16_t*>(gb->get_output(layer_kv_nodes_[i].second));
            kv_cache_[i].k.assign(k_ptr, k_ptr + new_len * Hkv * qk_head);
            kv_cache_[i].v.assign(v_ptr, v_ptr + new_len * Hkv * v_dim);
            kv_cache_[i].len = new_len;
        }
        kv_cache_valid_ = true;
    } else {
        for (uint32_t i = 0; i < config_.num_layers; i++) {
            const uint16_t* k_ptr = reinterpret_cast<const uint16_t*>(gb->get_output(layer_kv_nodes_[i].first));
            const uint16_t* v_ptr = reinterpret_cast<const uint16_t*>(gb->get_output(layer_kv_nodes_[i].second));
            kv_cache_[i].k.insert(kv_cache_[i].k.end(), k_ptr, k_ptr + new_len * Hkv * qk_head);
            kv_cache_[i].v.insert(kv_cache_[i].v.end(), v_ptr, v_ptr + new_len * Hkv * v_dim);
            kv_cache_[i].len += new_len;
        }
    }
}


uint32_t YoutuModel::decode(const std::vector<uint32_t>& tokens, float temperature,
                             float top_p, size_t top_k,
                             const std::string& profile_file, float* out_entropy) {
    if (!initialized_ || !graph_handle_) return 0;

    if (temperature < 0) temperature = config_.default_temperature;
    if (top_p < 0) top_p = config_.default_top_p;
    if (top_k == 0) top_k = config_.default_top_k;

    for (auto t : tokens) token_history_.push_back(t);

    layer_kv_nodes_.resize(config_.num_layers);

    const bool using_cache = kv_cache_valid_;

    size_t final_hidden;
    if (using_cache) {
        final_hidden = forward(tokens, true);
    } else {
        kv_cache_.resize(config_.num_layers);
        final_hidden = forward(token_history_, false);
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    const size_t last_idx = using_cache ? (tokens.size() - 1) : (token_history_.size() - 1);
    auto last_hidden = gb->index(final_hidden, last_idx, 0);
    const auto& buf = gb->get_output_buffer(last_hidden);
    last_hidden = gb->reshape(last_hidden, {1, buf.shape[0]});

    auto logits = gb->matmul(last_hidden, output_weight_node_id_, true, backend);
    auto sampled = sample_token(gb, logits, temperature, top_p, top_k);

    gb->execute(profile_file);

    compute_entropy(gb, logits, out_entropy);

    const size_t Hkv = config_.attention_kv_heads;
    const size_t qk_head = config_.qk_head_dim;
    const size_t v_dim = config_.v_head_dim;
    const size_t new_len = tokens.size();

    if (using_cache) {
        for (uint32_t i = 0; i < config_.num_layers; i++) {
            const size_t k_node = layer_kv_nodes_[i].first;
            const size_t v_node = layer_kv_nodes_[i].second;
            const uint16_t* k_ptr = reinterpret_cast<const uint16_t*>(gb->get_output(k_node));
            const uint16_t* v_ptr = reinterpret_cast<const uint16_t*>(gb->get_output(v_node));
            const size_t k_chunk = new_len * Hkv * qk_head;
            const size_t v_chunk = new_len * Hkv * v_dim;
            kv_cache_[i].k.insert(kv_cache_[i].k.end(), k_ptr, k_ptr + k_chunk);
            kv_cache_[i].v.insert(kv_cache_[i].v.end(), v_ptr, v_ptr + v_chunk);
            kv_cache_[i].len += new_len;
        }
    } else {
        const size_t full_len = token_history_.size();
        for (uint32_t i = 0; i < config_.num_layers; i++) {
            const size_t k_node = layer_kv_nodes_[i].first;
            const size_t v_node = layer_kv_nodes_[i].second;
            const uint16_t* k_ptr = reinterpret_cast<const uint16_t*>(gb->get_output(k_node));
            const uint16_t* v_ptr = reinterpret_cast<const uint16_t*>(gb->get_output(v_node));
            const size_t k_total = full_len * Hkv * qk_head;
            const size_t v_total = full_len * Hkv * v_dim;
            kv_cache_[i].k.assign(k_ptr, k_ptr + k_total);
            kv_cache_[i].v.assign(v_ptr, v_ptr + v_total);
            kv_cache_[i].len = full_len;
        }
        kv_cache_valid_ = true;
    }

    auto* ptr = gb->get_output(sampled);
    if (!ptr) return 0;
    return *reinterpret_cast<const uint32_t*>(ptr);
}


void YoutuModel::reset_cache() {
    token_history_.clear();
    kv_cache_.clear();
    kv_cache_valid_ = false;
    layer_kv_nodes_.clear();
}

}
}
