#include "model_gemma4.h"
#include "../../graph/graph.h"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace cactus {
namespace engine {

Gemma4Model::Gemma4Model() : Model() {}

Gemma4Model::Gemma4Model(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
}

bool Gemma4Model::is_global_layer(uint32_t idx) const {
    return config_.layer_types[idx] == "global";
}

std::vector<size_t> Gemma4Model::get_kv_layer_dims() const {
    uint32_t n = config_.num_layers;

    std::vector<size_t> dims(n);
    for (uint32_t i = 0; i < n; i++) {
        if (i >= first_shared_layer_) {
            dims[i] = 0;
        } else if (is_global_layer(i)) {
            dims[i] = config_.global_head_dim > 0 ? config_.global_head_dim : config_.attention_head_dim * 2;
        } else {
            dims[i] = config_.attention_head_dim;
        }
    }
    return dims;
}

std::vector<size_t> Gemma4Model::get_kv_layer_heads() const {
    uint32_t n = config_.num_layers;

    std::vector<size_t> heads(n);
    for (uint32_t i = 0; i < n; i++) {
        if (i >= first_shared_layer_) {
            heads[i] = 0;
        } else if (is_global_layer(i) && config_.num_global_kv_heads > 0) {
            heads[i] = config_.num_global_kv_heads;
        } else {
            heads[i] = config_.attention_kv_heads;
        }
    }
    return heads;
}

void Gemma4Model::compact_kv_cache() {
    uint32_t n = config_.num_layers;

    std::vector<size_t> target_windows(n, 0);
    for (uint32_t i = 0; i < n; i++) {
        if (i >= first_shared_layer_) continue;
        if (!is_global_layer(i))
            target_windows[i] = config_.sliding_window;
    }
    kv_cache_.compact_to_windows(target_windows);
}

void Gemma4Model::post_init() {
    uint32_t n = config_.num_layers;

    kv_cache_.set_window_size(0, 0);

    kv_share_map_.resize(n, -1);
    shared_k_nodes_.resize(n, 0);
    shared_v_nodes_.resize(n, 0);

    for (uint32_t i = first_shared_layer_; i < n; i++) {
        bool is_global = is_global_layer(i);
        for (int j = static_cast<int>(first_shared_layer_) - 1; j >= 0; j--) {
            if (is_global_layer(j) == is_global) {
                kv_share_map_[i] = j;
                break;
            }
        }
    }
}

void Gemma4Model::load_weights_to_graph(CactusGraph* gb) {
    uint32_t n = config_.num_layers;
    uint32_t num_shared = config_.num_kv_shared_layers;
    first_shared_layer_ = (n > num_shared) ? n - num_shared : n;

    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    weight_nodes_.output_norm_weight = gb->mmap_weights(model_folder_path_ + "/output_norm.weights");
    if (config_.tie_word_embeddings) {
        weight_nodes_.output_weight = embedding_node_id_;
        output_weight_node_id_ = embedding_node_id_;
    } else {
        weight_nodes_.output_weight = gb->mmap_weights(model_folder_path_ + "/output_weight.weights");
        output_weight_node_id_ = weight_nodes_.output_weight;
    }

    bool has_pli = config_.hidden_size_per_layer_input > 0;
    if (has_pli) {
        weight_nodes_.embed_tokens_per_layer = gb->mmap_embeddings(model_folder_path_ + "/embed_tokens_per_layer.weights");
        weight_nodes_.per_layer_model_proj = gb->mmap_weights(model_folder_path_ + "/per_layer_model_proj.weights");
        weight_nodes_.per_layer_proj_norm = gb->mmap_weights(model_folder_path_ + "/per_layer_proj_norm.weights");
    }

    bool has_moe = config_.enable_moe_block;

    for (uint32_t i = 0; i < config_.num_layers; i++) {
        auto& layer = weight_nodes_.layers[i];
        std::string prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";
        bool is_shared = (i >= first_shared_layer_);

        layer.attn_q_weight                    = gb->mmap_weights(prefix + "attn_q.weights");
        layer.attn_k_weight                    = is_shared ? 0 : gb->mmap_weights(prefix + "attn_k.weights");
        bool k_eq_v = config_.attention_k_eq_v && is_global_layer(i);
        layer.attn_v_weight                    = is_shared ? 0 : (k_eq_v ? layer.attn_k_weight : gb->mmap_weights(prefix + "attn_v.weights"));
        layer.attn_output_weight               = gb->mmap_weights(prefix + "attn_output.weights");
        layer.input_layernorm_weight           = gb->mmap_weights(prefix + "input_norm.weights");
        layer.attn_q_norm_weight               = gb->mmap_weights(prefix + "attn_q_norm.weights");
        layer.attn_k_norm_weight               = is_shared ? 0 : gb->mmap_weights(prefix + "attn_k_norm.weights");
        layer.ffn_gate_weight                  = gb->mmap_weights(prefix + "ffn_gate.weights");
        layer.ffn_up_weight                    = gb->mmap_weights(prefix + "ffn_up.weights");
        layer.ffn_down_weight                  = gb->mmap_weights(prefix + "ffn_down.weights");
        layer.post_attention_layernorm_weight   = gb->mmap_weights(prefix + "post_attn_norm.weights");
        layer.pre_feedforward_layernorm_weight  = gb->mmap_weights(prefix + "pre_ffn_norm.weights");
        layer.post_feedforward_layernorm_weight = gb->mmap_weights(prefix + "post_ffn_norm.weights");
        if (has_pli) {
            layer.per_layer_gate               = gb->mmap_weights(prefix + "per_layer_gate.weights");
            layer.per_layer_proj               = gb->mmap_weights(prefix + "per_layer_proj.weights");
            layer.post_per_layer_norm          = gb->mmap_weights(prefix + "post_per_layer_norm.weights");
        }
        layer.layer_scalar                     = std::filesystem::exists(prefix + "layer_scalar.weights") ? gb->mmap_weights(prefix + "layer_scalar.weights") : 0;

        if (has_moe) {
            uint32_t num_experts = config_.num_experts;
            uint32_t hidden = config_.hidden_dim;
            uint32_t expert_dim = config_.expert_intermediate_size > 0 ? config_.expert_intermediate_size : config_.ffn_intermediate_dim;
            layer.moe_w1_experts.resize(num_experts);
            layer.moe_w3_experts.resize(num_experts);
            layer.moe_w2_experts.resize(num_experts);

            auto w1_packed = gb->mmap_weights(prefix + "moe_gate_proj.weights");
            auto w3_packed = gb->mmap_weights(prefix + "moe_up_proj.weights");
            auto w2_packed = gb->mmap_weights(prefix + "moe_down_proj.weights");

            const auto& w1_buf = gb->get_output_buffer(w1_packed);
            const auto& w3_buf = gb->get_output_buffer(w3_packed);
            const auto& w2_buf = gb->get_output_buffer(w2_packed);

            auto setup_experts = [&](const BufferDesc& buf,
                                     std::vector<size_t>& expert_nodes,
                                     size_t out_dim, size_t in_dim) {
                auto* base = static_cast<char*>(const_cast<void*>(buf.get_data()));
                Precision prec = buf.precision;
                bool is_quantized = PrecisionTraits::is_integer(prec) && buf.group_size > 0;
                size_t K = in_dim;
                if (is_quantized && K % buf.group_size != 0)
                    K = ((K + buf.group_size - 1) / buf.group_size) * buf.group_size;
                size_t expert_data_bytes = PrecisionTraits::packed_size_of(prec, out_dim * K);
                size_t groups_per_expert = is_quantized ? (out_dim * K + buf.group_size - 1) / buf.group_size : 0;
                size_t expert_scales_bytes = groups_per_expert * sizeof(__fp16);

                char* scales_base = is_quantized ? static_cast<char*>(const_cast<void*>(buf.scales_data)) : nullptr;

                for (uint32_t e = 0; e < num_experts; e++) {
                    expert_nodes[e] = gb->input({out_dim, K}, prec);
                    gb->set_external_input(expert_nodes[e], base + e * expert_data_bytes, prec);
                    if (is_quantized) {
                        gb->set_grouped_scales(expert_nodes[e], buf.group_size, groups_per_expert,
                                               scales_base + e * expert_scales_bytes);
                        if (buf.is_interleaved)
                            gb->set_interleaved(expert_nodes[e], true, out_dim);
                    }
                }
            };

            setup_experts(w1_buf, layer.moe_w1_experts, expert_dim, hidden);
            setup_experts(w3_buf, layer.moe_w3_experts, expert_dim, hidden);
            setup_experts(w2_buf, layer.moe_w2_experts, hidden, expert_dim);

            layer.router_proj = gb->mmap_weights(prefix + "router_proj.weights");
            layer.router_scale = gb->mmap_weights(prefix + "router_scale.weights");
            layer.moe_per_expert_scale = gb->mmap_weights(prefix + "moe_per_expert_scale.weights");
            layer.post_ffn_norm_1 = gb->mmap_weights(prefix + "post_ffn_norm_1.weights");
            layer.pre_ffn_norm_2 = gb->mmap_weights(prefix + "pre_ffn_norm_2.weights");
            layer.post_ffn_norm_2 = gb->mmap_weights(prefix + "post_ffn_norm_2.weights");
        }
    }

    size_t sliding_head_dim = config_.attention_head_dim;
    size_t global_head_dim = config_.global_head_dim > 0 ? config_.global_head_dim : config_.attention_head_dim * 2;
    v_norm_ones_weight_.assign(std::max(sliding_head_dim, global_head_dim), static_cast<__fp16>(1.0f));
    v_norm_ones_node_ = gb->input({sliding_head_dim}, Precision::FP16);
    gb->set_external_input(v_norm_ones_node_, v_norm_ones_weight_.data(), Precision::FP16);
    v_norm_ones_global_node_ = gb->input({global_head_dim}, Precision::FP16);
    gb->set_external_input(v_norm_ones_global_node_, v_norm_ones_weight_.data(), Precision::FP16);

    if (npu::is_npu_available()) {
        std::string npu_prefill_path = model_folder_path_ + "/model.mlpackage";
        if (std::filesystem::exists(npu_prefill_path)) {
            if (!load_npu_prefill(npu_prefill_path) || !has_npu_prefill()) {
                CACTUS_LOG_WARN("npu", "[gemma4] found model.mlpackage but failed to enable NPU prefill; using CPU prefill");
            }
        } else {
            CACTUS_LOG_WARN("npu", "[gemma4] model.mlpackage not found; using CPU prefill");
        }
    } else {
        CACTUS_LOG_WARN("npu", "[gemma4] NPU backend unavailable on this device; using CPU prefill");
    }
}

size_t Gemma4Model::build_per_layer_input(CactusGraph* gb, size_t hidden, size_t pli_combined, uint32_t layer_idx,
                                              ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];
    uint32_t pli_dim = config_.hidden_size_per_layer_input;

    auto gate = gb->gelu(gb->matmul(hidden, layer.per_layer_gate, true, backend));
    auto pli_slice = gb->slice(pli_combined, 1, layer_idx * pli_dim, pli_dim);
    auto gated = gb->multiply(gate, pli_slice);
    auto pli_proj = gb->matmul(gated, layer.per_layer_proj, true, backend);
    auto pli_normed = gb->rms_norm(pli_proj, layer.post_per_layer_norm, config_.layer_norm_eps);

    return gb->add(hidden, pli_normed);
}

size_t Gemma4Model::apply_partial_rope(CactusGraph* gb, size_t tensor, size_t head_dim, size_t rot_dim,
                                           float rope_freq, size_t position_offset) {
    if (rot_dim < head_dim) {
        size_t half_dim = head_dim / 2;
        size_t half_rot = rot_dim / 2;
        size_t pass_len = half_dim - half_rot;
        float adjusted_theta = std::pow(rope_freq, static_cast<float>(rot_dim) / static_cast<float>(head_dim));

        auto left_rot   = gb->slice(tensor, 3, 0,                   half_rot);
        auto left_pass  = gb->slice(tensor, 3, half_rot,            pass_len);
        auto right_rot  = gb->slice(tensor, 3, half_dim,            half_rot);
        auto right_pass = gb->slice(tensor, 3, half_dim + half_rot, pass_len);

        auto rotated = gb->rope(gb->concat(left_rot, right_rot, 3), adjusted_theta, position_offset);
        auto rotated_left  = gb->slice(rotated, 3, 0,        half_rot);
        auto rotated_right = gb->slice(rotated, 3, half_rot, half_rot);

        auto new_left  = gb->concat(rotated_left,  left_pass,  3);
        auto new_right = gb->concat(rotated_right, right_pass, 3);
        return gb->concat(new_left, new_right, 3);
    }
    return gb->rope(tensor, rope_freq, position_offset);
}

size_t Gemma4Model::build_attention(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                       ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];
    size_t seq_len = gb->get_output_buffer(input).shape[0];
    int share_src = (layer_idx < kv_share_map_.size()) ? kv_share_map_[layer_idx] : -1;

    bool is_global = is_global_layer(layer_idx);

    size_t head_dim  = is_global ? (config_.global_head_dim > 0 ? config_.global_head_dim : config_.attention_head_dim * 2) : config_.attention_head_dim;
    size_t num_heads = config_.attention_heads;
    size_t kv_heads  = is_global && config_.num_global_kv_heads > 0 ? config_.num_global_kv_heads : config_.attention_kv_heads;
    float rope_freq  = is_global ? config_.rope_theta : config_.rope_local_base_freq;
    size_t window    = is_global ? 0 : config_.sliding_window;
    size_t rot_dim   = static_cast<size_t>(head_dim * (is_global ? config_.global_partial_rotary_factor : 1.0f));

    auto q = gb->matmul(input, layer.attn_q_weight, true, backend);
    q = gb->reshape(q, {seq_len * num_heads, head_dim});
    q = gb->rms_norm(q, layer.attn_q_norm_weight, config_.layer_norm_eps);
    q = gb->reshape(q, {1, seq_len, num_heads, head_dim});

    size_t q4 = apply_partial_rope(gb, q, head_dim, rot_dim, rope_freq, position_offset);

    size_t k4, v4;
    if (share_src >= 0 && shared_k_nodes_[share_src] != 0) {
        k4 = shared_k_nodes_[share_src];
        v4 = shared_v_nodes_[share_src];
    } else {
        auto k = gb->matmul(input, layer.attn_k_weight, true, backend);
        k = gb->reshape(k, {seq_len * kv_heads, head_dim});
        k = gb->rms_norm(k, layer.attn_k_norm_weight, config_.layer_norm_eps);
        k = gb->reshape(k, {1, seq_len, kv_heads, head_dim});

        k4 = apply_partial_rope(gb, k, head_dim, rot_dim, rope_freq, position_offset);

        auto v_proj = gb->matmul(input, layer.attn_v_weight, true, backend);
        size_t v_ones = is_global ? v_norm_ones_global_node_ : v_norm_ones_node_;
        auto v = gb->rms_norm(gb->reshape(v_proj, {seq_len * kv_heads, head_dim}), v_ones, config_.layer_norm_eps);
        v4 = gb->reshape(v, {1, seq_len, kv_heads, head_dim});

        shared_k_nodes_[layer_idx] = k4;
        shared_v_nodes_[layer_idx] = v4;
    }

    if (use_cache && share_src < 0) {
        cache_k_output_nodes_[layer_idx] = k4;
        cache_v_output_nodes_[layer_idx] = v4;
    }

    size_t cache_src = (share_src >= 0) ? static_cast<size_t>(share_src) : layer_idx;
    size_t attn;
    if (use_cache && !kv_cache_.is_empty()) {
        attn = gb->attention_int8_hybrid(q4, k4, v4, attention_scale_, position_offset,
            kv_cache_.get_keys_int8(cache_src), kv_cache_.get_values_int8(cache_src),
            kv_cache_.get_key_scales(cache_src), kv_cache_.get_value_scales(cache_src),
            kv_cache_.current_seq_len, kv_heads, head_dim, window);
    } else {
        attn = gb->attention(q4, k4, v4, attention_scale_, position_offset, window);
    }

    return gb->matmul(gb->reshape(attn, {seq_len, num_heads * head_dim}), layer.attn_output_weight, true, backend);
}

size_t Gemma4Model::build_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                  ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto gate = gb->gelu(gb->matmul(input, layer.ffn_gate_weight, true, backend));
    auto up = gb->matmul(input, layer.ffn_up_weight, true, backend);
    return gb->matmul(gb->multiply(gate, up), layer.ffn_down_weight, true, backend);
}

size_t Gemma4Model::build_moe(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                  ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];

    auto router_logits = gb->matmul(input, layer.router_proj, true, backend);
    auto topk_result = gb->topk(router_logits, config_.num_experts_per_tok);
    auto topk_indices = gb->index(topk_result, 0, 0);
    auto routing_probs = gb->softmax(router_logits);

    return gb->moe_layer(input, routing_probs, topk_indices,
                         layer.moe_w1_experts, layer.moe_w3_experts, layer.moe_w2_experts,
                         config_.num_experts, config_.num_experts_per_tok,
                         true, 1e-6f, 1.0f, Activation::GELU,
                         layer.moe_per_expert_scale);
}

size_t Gemma4Model::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                                ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];

    auto normed = gb->rms_norm(hidden, layer.input_layernorm_weight, config_.layer_norm_eps);
    auto attn_raw = build_attention(gb, normed, layer_idx, backend, use_cache, position_offset);
    auto attn = gb->rms_norm(attn_raw, layer.post_attention_layernorm_weight, config_.layer_norm_eps);
    auto residual = gb->add(hidden, attn);

    if (config_.enable_moe_block) {
        auto h1 = gb->rms_norm(residual, layer.pre_feedforward_layernorm_weight, config_.layer_norm_eps);
        h1 = build_mlp(gb, h1, layer_idx, backend);
        h1 = gb->rms_norm(h1, layer.post_ffn_norm_1, config_.layer_norm_eps);

        auto h2 = gb->multiply(residual, layer.router_scale);
        h2 = gb->rms_norm(h2, layer.pre_ffn_norm_2, config_.layer_norm_eps);
        h2 = build_moe(gb, h2, layer_idx, backend);
        h2 = gb->rms_norm(h2, layer.post_ffn_norm_2, config_.layer_norm_eps);

        auto combined = gb->rms_norm(gb->add(h1, h2), layer.post_feedforward_layernorm_weight, config_.layer_norm_eps);
        return gb->add(residual, combined);
    }

    auto pre_mlp = gb->rms_norm(residual, layer.pre_feedforward_layernorm_weight, config_.layer_norm_eps);
    auto mlp_raw = build_mlp(gb, pre_mlp, layer_idx, backend);
    auto mlp = gb->rms_norm(mlp_raw, layer.post_feedforward_layernorm_weight, config_.layer_norm_eps);

    return gb->add(residual, mlp);
}

size_t Gemma4Model::apply_transformer_layer(CactusGraph* gb, size_t hidden, size_t pli, uint32_t layer_idx,
                                                ComputeBackend backend, bool use_cache, size_t pos_offset) {
    hidden = build_transformer_block(gb, hidden, layer_idx, backend, use_cache, pos_offset);
    if (config_.hidden_size_per_layer_input > 0)
        hidden = build_per_layer_input(gb, hidden, pli, layer_idx, backend);
    if (weight_nodes_.layers[layer_idx].layer_scalar != 0)
        hidden = gb->multiply(hidden, weight_nodes_.layers[layer_idx].layer_scalar);
    return hidden;
}

size_t Gemma4Model::build_pli_combined(CactusGraph* gb, size_t hidden, size_t pli_embed,
                                           size_t seq_len, ComputeBackend backend) {
    uint32_t num_layers = config_.num_layers;
    uint32_t pli_dim = config_.hidden_size_per_layer_input;

    auto pli_proj = gb->scalar_multiply(gb->matmul(hidden, weight_nodes_.per_layer_model_proj, true, backend),
                                        1.0f / std::sqrt(static_cast<float>(config_.hidden_dim)));
    pli_proj = gb->reshape(pli_proj, {seq_len * num_layers, pli_dim});
    pli_proj = gb->rms_norm(pli_proj, weight_nodes_.per_layer_proj_norm, config_.layer_norm_eps);
    pli_proj = gb->reshape(pli_proj, {seq_len, num_layers * pli_dim});
    return gb->scalar_multiply(gb->add(pli_proj, pli_embed), 1.0f / std::sqrt(2.0f));
}

std::pair<size_t, size_t> Gemma4Model::build_preamble_and_embed(CactusGraph* gb, size_t seq_len, ComputeBackend backend,
                                                                    size_t& token_input, size_t& pli_input) {
    uint32_t pli_dim = config_.hidden_size_per_layer_input;

    token_input = gb->input({seq_len}, Precision::FP32);
    auto hidden = gb->scalar_multiply(gb->embedding(embedding_node_id_, token_input),
                                      std::sqrt(static_cast<float>(config_.hidden_dim)));

    pli_input = gb->input({seq_len}, Precision::FP32);
    auto pli_embed = gb->scalar_multiply(gb->embedding(weight_nodes_.embed_tokens_per_layer, pli_input),
                                         std::sqrt(static_cast<float>(pli_dim)));
    auto pli_combined = build_pli_combined(gb, hidden, pli_embed, seq_len, backend);

    return {hidden, pli_combined};
}

void Gemma4Model::set_token_inputs(CactusGraph* gb, size_t token_input, size_t pli_input,
                                       const std::vector<uint32_t>& tokens) {
    std::vector<float> input_data(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++)
        input_data[i] = static_cast<float>(tokens[i]);
    gb->set_input(token_input, input_data.data(), Precision::FP32);
    gb->set_input(pli_input, input_data.data(), Precision::FP32);
}

size_t Gemma4Model::forward_from_embeddings(CactusGraph* gb, size_t hidden, const std::vector<uint32_t>& pli_tokens,
                                                size_t seq_len, ComputeBackend backend, bool use_cache) {
    return forward_from_embeddings(gb, hidden, hidden, pli_tokens, seq_len, backend, use_cache);
}

size_t Gemma4Model::build_pli_combined_from_tokens(CactusGraph* gb, size_t hidden,
                                                       const std::vector<uint32_t>& pli_tokens,
                                                       size_t seq_len, ComputeBackend backend) {
    if (config_.hidden_size_per_layer_input == 0)
        return 0;

    uint32_t pli_dim = config_.hidden_size_per_layer_input;

    auto pli_input = gb->input({seq_len}, Precision::FP32);
    auto pli_embed = gb->scalar_multiply(gb->embedding(weight_nodes_.embed_tokens_per_layer, pli_input),
                                         std::sqrt(static_cast<float>(pli_dim)));
    auto pli_combined = build_pli_combined(gb, hidden, pli_embed, seq_len, backend);

    std::vector<float> pli_data(pli_tokens.size());
    for (size_t i = 0; i < pli_tokens.size(); i++)
        pli_data[i] = static_cast<float>(pli_tokens[i]);
    gb->set_input(pli_input, pli_data.data(), Precision::FP32);

    return pli_combined;
}

size_t Gemma4Model::forward_from_embeddings(CactusGraph* gb, size_t hidden, size_t pli_hidden_source,
                                                const std::vector<uint32_t>& pli_tokens, size_t seq_len,
                                                ComputeBackend backend, bool use_cache) {
    size_t pos_offset = use_cache ? kv_cache_.get_total_seq_len() : 0;

    std::fill(shared_k_nodes_.begin(), shared_k_nodes_.end(), 0);
    std::fill(shared_v_nodes_.begin(), shared_v_nodes_.end(), 0);

    if (config_.hidden_size_per_layer_input == 0) {
        for (uint32_t i = 0; i < config_.num_layers; i++)
            hidden = apply_transformer_layer(gb, hidden, 0, i, backend, use_cache, pos_offset);
        return gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);
    }

    auto pli_combined = build_pli_combined_from_tokens(gb, pli_hidden_source, pli_tokens, seq_len, backend);

    for (uint32_t i = 0; i < config_.num_layers; i++)
        hidden = apply_transformer_layer(gb, hidden, pli_combined, i, backend, use_cache, pos_offset);

    return gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);
}

size_t Gemma4Model::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (!initialized_ || !graph_handle_)
        throw std::runtime_error("Model not initialized - call init() first");
    if (tokens.empty())
        throw std::runtime_error("Token sequence cannot be empty");

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    std::fill(shared_k_nodes_.begin(), shared_k_nodes_.end(), 0);
    std::fill(shared_v_nodes_.begin(), shared_v_nodes_.end(), 0);

    size_t pos_offset = use_cache ? kv_cache_.get_total_seq_len() : 0;
    auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;

    if (config_.hidden_size_per_layer_input == 0) {
        auto token_input = gb->input({tokens.size()}, Precision::FP32);
        auto hidden = gb->scalar_multiply(gb->embedding(embedding_node_id_, token_input),
                                          std::sqrt(static_cast<float>(config_.hidden_dim)));

        for (uint32_t i = 0; i < config_.num_layers; i++)
            hidden = apply_transformer_layer(gb, hidden, 0, i, backend, use_cache, pos_offset);

        std::vector<float> input_data(tokens.size());
        for (size_t i = 0; i < tokens.size(); i++)
            input_data[i] = static_cast<float>(tokens[i]);
        gb->set_input(token_input, input_data.data(), Precision::FP32);
        return gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);
    }

    size_t token_input, pli_input;
    auto hidden_pli = build_preamble_and_embed(gb, tokens.size(), backend, token_input, pli_input);
    size_t hidden = hidden_pli.first, pli = hidden_pli.second;

    for (uint32_t i = 0; i < config_.num_layers; i++)
        hidden = apply_transformer_layer(gb, hidden, pli, i, backend, use_cache, pos_offset);

    set_token_inputs(gb, token_input, pli_input, tokens);
    return gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);
}

size_t Gemma4Model::forward_split(const std::vector<uint32_t>& tokens, bool use_cache) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    std::fill(shared_k_nodes_.begin(), shared_k_nodes_.end(), 0);
    std::fill(shared_v_nodes_.begin(), shared_v_nodes_.end(), 0);

    size_t seq_len = tokens.size();
    size_t pos_offset = use_cache ? kv_cache_.get_total_seq_len() : 0;
    auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;

    if (config_.hidden_size_per_layer_input == 0) {
        auto token_input = gb->input({seq_len}, Precision::FP32);
        auto hidden = gb->scalar_multiply(gb->embedding(embedding_node_id_, token_input),
                                          std::sqrt(static_cast<float>(config_.hidden_dim)));

        for (uint32_t i = 0; i < first_shared_layer_; i++)
            hidden = apply_transformer_layer(gb, hidden, 0, i, backend, use_cache, pos_offset);

        hidden = gb->index(hidden, seq_len - 1, 0);
        hidden = gb->reshape(hidden, {1, config_.hidden_dim});

        size_t shared_pos_offset = pos_offset + seq_len - 1;
        for (uint32_t i = first_shared_layer_; i < config_.num_layers; i++)
            hidden = apply_transformer_layer(gb, hidden, 0, i, backend, use_cache, shared_pos_offset);

        std::vector<float> input_data(tokens.size());
        for (size_t i = 0; i < tokens.size(); i++)
            input_data[i] = static_cast<float>(tokens[i]);
        gb->set_input(token_input, input_data.data(), Precision::FP32);
        return gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);
    }

    size_t token_input, pli_input;
    auto hidden_pli = build_preamble_and_embed(gb, seq_len, backend, token_input, pli_input);
    size_t hidden = hidden_pli.first, pli = hidden_pli.second;

    for (uint32_t i = 0; i < first_shared_layer_; i++)
        hidden = apply_transformer_layer(gb, hidden, pli, i, backend, use_cache, pos_offset);

    hidden = gb->index(hidden, seq_len - 1, 0);
    hidden = gb->reshape(hidden, {1, config_.hidden_dim});
    auto pli_last = gb->index(pli, seq_len - 1, 0);
    pli_last = gb->reshape(pli_last, {1, config_.num_layers * config_.hidden_size_per_layer_input});

    size_t shared_pos_offset = pos_offset + seq_len - 1;
    for (uint32_t i = first_shared_layer_; i < config_.num_layers; i++)
        hidden = apply_transformer_layer(gb, hidden, pli_last, i, backend, use_cache, shared_pos_offset);

    set_token_inputs(gb, token_input, pli_input, tokens);
    return gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);
}


void Gemma4Model::prefill(const std::vector<uint32_t>& tokens, size_t chunk_size, const std::string& profile_file) {
    if (tokens.empty())
        return;

    if (has_npu_prefill()) {
        size_t npu_chunk_size = static_cast<size_t>(npu_prefill_->get_chunk_size());
        if (tokens.size() > npu_chunk_size) {
            Model::prefill(tokens, chunk_size, profile_file);
            return;
        }
    }

    static constexpr size_t SPLIT_PREFILL_MIN_TOKENS = 32;
    bool use_split = config_.num_kv_shared_layers > 0
                  && tokens.size() >= SPLIT_PREFILL_MIN_TOKENS
                  && !std::getenv("CACTUS_DISABLE_SPLIT_PREFILL");

    if (!use_split) {
        Model::prefill(tokens, chunk_size, profile_file);
        return;
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    auto process_chunk = [&](const std::vector<uint32_t>& chunk) {
        forward_split(chunk, true);
        gb->execute(profile_file);
        update_kv_cache(gb, chunk.size());
    };

    if (tokens.size() <= chunk_size) {
        process_chunk(tokens);
        return;
    }

    size_t num_full_chunks = (tokens.size() - 1) / chunk_size;
    for (size_t i = 0; i < num_full_chunks; ++i) {
        size_t start = i * chunk_size;
        std::vector<uint32_t> chunk(tokens.begin() + start, tokens.begin() + start + chunk_size);
        if (i == 1)
            gb->set_prefill_mode(true);
        process_chunk(chunk);
    }

    gb->set_prefill_mode(false);
    size_t final_start = num_full_chunks * chunk_size;
    process_chunk(std::vector<uint32_t>(tokens.begin() + final_start, tokens.end()));
}

}
}
