#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <stdexcept>

namespace cactus {
namespace engine {

static const float RSQRT2 = 1.0f / std::sqrt(2.0f);

GemmaModel3n::GemmaModel3n() : Model() {}

GemmaModel3n::GemmaModel3n(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
}

void GemmaModel3n::post_init() {
    kv_cache_.set_window_size(0, 0);

    uint32_t n = config_.num_layers;
    uint32_t num_shared = 10;
    uint32_t first_shared = (n > num_shared) ? n - num_shared : n;

    kv_share_map_.resize(n, -1);
    shared_k_nodes_.resize(n, 0);
    shared_v_nodes_.resize(n, 0);

    auto is_global_layer = [&](uint32_t idx) -> bool {
        if (!config_.layer_types.empty() && idx < config_.layer_types.size()) {
            const auto& lt = config_.layer_types[idx];
            return (lt == "global" || lt == "full" || lt == "full_attention");
        }
        return (idx % 5) == 4;
    };

    for (uint32_t i = first_shared; i < n; i++) {
        bool is_global = is_global_layer(i);
        for (int j = static_cast<int>(first_shared) - 1; j >= 0; j--) {
            if (is_global_layer(j) == is_global) {
                kv_share_map_[i] = j;
                break;
            }
        }
    }
}

void GemmaModel3n::load_weights_to_graph(CactusGraph* gb) {
    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    weight_nodes_.output_norm_weight = gb->mmap_weights(model_folder_path_ + "/output_norm.weights");
    weight_nodes_.output_weight = gb->mmap_weights(model_folder_path_ + "/output_weight.weights");
    output_weight_node_id_ = weight_nodes_.output_weight;

    for (int i = 0; i < 3; i++) {
        auto idx = std::to_string(i);
        weight_nodes_.altup_proj_weights[i] = gb->mmap_weights(model_folder_path_ + "/altup_proj_" + idx + ".weights");
        weight_nodes_.altup_unembed_proj_weights[i] = gb->mmap_weights(model_folder_path_ + "/altup_unembed_proj_" + idx + ".weights");
    }

    weight_nodes_.embed_tokens_per_layer = gb->mmap_embeddings(model_folder_path_ + "/embed_tokens_per_layer.weights");
    weight_nodes_.per_layer_model_proj = gb->mmap_weights(model_folder_path_ + "/per_layer_model_proj.weights");
    weight_nodes_.per_layer_proj_norm = gb->mmap_weights(model_folder_path_ + "/per_layer_proj_norm.weights");

    for (uint32_t i = 0; i < config_.num_layers; i++) {
        auto& layer = weight_nodes_.layers[i];
        std::string prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";

        layer.attn_q_weight       = gb->mmap_weights(prefix + "attn_q.weights");
        layer.attn_k_weight       = gb->mmap_weights(prefix + "attn_k.weights");
        layer.attn_v_weight       = gb->mmap_weights(prefix + "attn_v.weights");
        layer.attn_output_weight  = gb->mmap_weights(prefix + "attn_output.weights");
        layer.input_layernorm_weight           = gb->mmap_weights(prefix + "input_norm.weights");
        layer.attn_q_norm_weight               = gb->mmap_weights(prefix + "attn_q_norm.weights");
        layer.attn_k_norm_weight               = gb->mmap_weights(prefix + "attn_k_norm.weights");
        layer.ffn_gate_weight                  = gb->mmap_weights(prefix + "ffn_gate.weights");
        layer.ffn_up_weight                    = gb->mmap_weights(prefix + "ffn_up.weights");
        layer.ffn_down_weight                  = gb->mmap_weights(prefix + "ffn_down.weights");
        layer.post_attention_layernorm_weight   = gb->mmap_weights(prefix + "post_attn_norm.weights");
        layer.pre_feedforward_layernorm_weight  = gb->mmap_weights(prefix + "pre_ffn_norm.weights");
        layer.post_feedforward_layernorm_weight = gb->mmap_weights(prefix + "post_ffn_norm.weights");

        layer.altup_router_norm          = gb->mmap_weights(prefix + "altup_router_norm.weights");
        layer.altup_prediction_coefs     = gb->mmap_weights(prefix + "altup_prediction_coefs.weights");
        layer.altup_correction_coefs     = gb->mmap_weights(prefix + "altup_correction_coefs.weights");
        layer.altup_correct_output_scale = gb->mmap_weights(prefix + "altup_correct_output_scale.weights");
        layer.altup_modality_router      = gb->mmap_weights(prefix + "altup_modality_router.weights");
        layer.laurel_left                = gb->mmap_weights(prefix + "laurel_left.weights");
        layer.laurel_right               = gb->mmap_weights(prefix + "laurel_right.weights");
        layer.laurel_norm                = gb->mmap_weights(prefix + "laurel_norm.weights");
        layer.per_layer_gate             = gb->mmap_weights(prefix + "per_layer_gate.weights");
        layer.per_layer_proj             = gb->mmap_weights(prefix + "per_layer_proj.weights");
        layer.post_per_layer_norm        = gb->mmap_weights(prefix + "post_per_layer_norm.weights");
    }
}


size_t GemmaModel3n::build_rms_norm_no_weight(CactusGraph* gb, size_t input, size_t num_rows, size_t row_dim) const {
    auto flat = gb->reshape(input, {num_rows, row_dim});
    auto variance = gb->mean(gb->multiply(flat, flat), -1);
    auto rms = gb->scalar_sqrt(gb->scalar_add(variance, config_.layer_norm_eps));
    return gb->divide(flat, rms);
}

size_t GemmaModel3n::build_magnitude_normalize(CactusGraph* gb, size_t reference, size_t target) const {
    auto ref_sq = gb->mean(gb->multiply(reference, reference), -1);
    auto tgt_sq = gb->mean(gb->multiply(target, target), -1);
    auto ref_mag = gb->scalar_sqrt(ref_sq);
    auto tgt_mag = gb->scalar_add(gb->scalar_sqrt(tgt_sq), 1e-5f);
    auto ratio = gb->divide(ref_mag, tgt_mag);
    return gb->multiply(target, ratio);
}

size_t GemmaModel3n::build_gaussian_topk(CactusGraph* gb, size_t input, float ppf) const {
    auto mu    = gb->mean(input, -1);
    auto diff  = gb->subtract(input, mu);
    auto var   = gb->mean(gb->multiply(diff, diff), -1);
    auto sigma = gb->scalar_sqrt(var);
    auto cutoff = gb->add(mu, gb->scalar_multiply(sigma, ppf));
    return gb->relu(gb->subtract(input, cutoff));
}

size_t GemmaModel3n::build_laurel(CactusGraph* gb, size_t normed_input, uint32_t layer_idx,
                                  ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto x = gb->matmul(normed_input, layer.laurel_left, true, backend);
    x = gb->matmul(x, layer.laurel_right, true, backend);
    return gb->add(normed_input, gb->rms_norm(x, layer.laurel_norm, config_.layer_norm_eps));
}

size_t GemmaModel3n::build_altup_router_modalities(CactusGraph* gb, size_t stream0, uint32_t layer_idx,
                                                    ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto x = gb->rms_norm(stream0, layer.altup_router_norm, config_.layer_norm_eps);
    x = gb->scalar_multiply(x, 1.0f / static_cast<float>(config_.hidden_dim));
    return gb->tanh(gb->matmul(x, layer.altup_modality_router, true, backend));
}

void GemmaModel3n::build_altup_predict(CactusGraph* gb, size_t modalities, uint32_t layer_idx,
                                        const size_t* streams, size_t* predictions) const {
    uint32_t n = config_.altup_num_inputs;
    auto coefs = gb->matmul(modalities, weight_nodes_.layers[layer_idx].altup_prediction_coefs, true, ComputeBackend::CPU);

    for (uint32_t i = 0; i < n; i++) {
        predictions[i] = streams[i];
        for (uint32_t j = 0; j < n; j++) {
            auto c = gb->slice(coefs, 1, i * n + j, 1);
            predictions[i] = gb->add(predictions[i], gb->multiply(c, streams[j]));
        }
    }
}

void GemmaModel3n::build_altup_correct(CactusGraph* gb, size_t activated, size_t modalities, uint32_t layer_idx,
                                        ComputeBackend backend, const size_t* predictions, size_t* corrected) const {
    uint32_t n = config_.altup_num_inputs;
    auto coefs = gb->scalar_add(gb->matmul(modalities, weight_nodes_.layers[layer_idx].altup_correction_coefs, true, backend), 1.0f);
    auto innovation = gb->subtract(activated, predictions[0]);

    for (uint32_t i = 0; i < n; i++) {
        auto c = gb->slice(coefs, 1, i, 1);
        corrected[i] = gb->add(predictions[i], gb->multiply(c, innovation));
    }
}

void GemmaModel3n::build_per_layer_input(CactusGraph* gb, size_t pli_combined, uint32_t layer_idx,
                                          ComputeBackend backend, size_t* streams) const {
    const auto& layer = weight_nodes_.layers[layer_idx];
    uint32_t pli_dim = config_.hidden_size_per_layer_input;

    auto gated = gb->multiply(streams[0], layer.altup_correct_output_scale);
    gated = gb->gelu(gb->matmul(gated, layer.per_layer_gate, true, backend));

    auto pli = gb->multiply(gated, gb->slice(pli_combined, 1, layer_idx * pli_dim, pli_dim));
    pli = gb->rms_norm(gb->matmul(pli, layer.per_layer_proj, true, backend), layer.post_per_layer_norm, config_.layer_norm_eps);

    for (uint32_t i = 1; i < config_.altup_num_inputs; i++)
        streams[i] = gb->add(streams[i], pli);
}


size_t GemmaModel3n::build_attention(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                     ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];
    size_t seq_len    = gb->get_output_buffer(input).shape[0];
    size_t num_heads  = config_.attention_heads;
    size_t kv_heads   = config_.attention_kv_heads;
    size_t head_dim   = config_.attention_head_dim;
    int share_src     = (layer_idx < kv_share_map_.size()) ? kv_share_map_[layer_idx] : -1;

    bool is_global = false;
    if (!config_.layer_types.empty() && layer_idx < config_.layer_types.size()) {
        const auto& lt = config_.layer_types[layer_idx];
        is_global = (lt == "global" || lt == "full" || lt == "full_attention");
    } else {
        is_global = (layer_idx % 5) == 4;
    }
    float rope_freq   = is_global ? config_.rope_theta : config_.rope_local_base_freq;
    size_t window     = is_global ? 0 : 512;

    auto q = gb->matmul(input, layer.attn_q_weight, true, backend);
    q = gb->reshape(q, {seq_len * num_heads, head_dim});
    q = gb->rms_norm(q, layer.attn_q_norm_weight, config_.layer_norm_eps);
    q = gb->reshape(q, {1, seq_len, num_heads, head_dim});
    auto q4 = gb->rope(q, rope_freq, position_offset);

    size_t k4, v4;
    if (share_src >= 0 && shared_k_nodes_[share_src] != 0) {
        k4 = shared_k_nodes_[share_src];
        v4 = shared_v_nodes_[share_src];
    } else {
        auto k = gb->matmul(input, layer.attn_k_weight, true, backend);
        k = gb->reshape(k, {seq_len * kv_heads, head_dim});
        k = gb->rms_norm(k, layer.attn_k_norm_weight, config_.layer_norm_eps);
        k = gb->reshape(k, {1, seq_len, kv_heads, head_dim});
        k4 = gb->rope(k, rope_freq, position_offset);

        auto v = build_rms_norm_no_weight(gb, gb->matmul(input, layer.attn_v_weight, true, backend), seq_len * kv_heads, head_dim);
        v4 = gb->reshape(v, {1, seq_len, kv_heads, head_dim});

        shared_k_nodes_[layer_idx] = k4;
        shared_v_nodes_[layer_idx] = v4;
    }

    if (use_cache) {
        cache_k_output_nodes_[layer_idx] = k4;
        cache_v_output_nodes_[layer_idx] = v4;
    }

    size_t attn;
    if (use_cache && !kv_cache_.is_empty()) {
        attn = gb->attention_int8_hybrid(
            q4, k4, v4, attention_scale_, position_offset,
            kv_cache_.get_keys_int8(layer_idx), kv_cache_.get_values_int8(layer_idx),
            kv_cache_.get_key_scales(layer_idx), kv_cache_.get_value_scales(layer_idx),
            kv_cache_.current_seq_len, kv_heads, head_dim, window);
    } else {
        attn = gb->attention(q4, k4, v4, attention_scale_, position_offset, window);
    }

    return gb->matmul(gb->reshape(attn, {seq_len, num_heads * head_dim}), layer.attn_output_weight, true, backend);
}


size_t GemmaModel3n::build_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx,
                               ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];

    auto gate = gb->matmul(input, layer.ffn_gate_weight, true, backend);
    auto up   = gb->matmul(input, layer.ffn_up_weight, true, backend);

    if (layer_idx < config_.activation_sparsity_ppf.size() && config_.activation_sparsity_ppf[layer_idx] > 0.0f)
        gate = build_gaussian_topk(gb, gate, config_.activation_sparsity_ppf[layer_idx]);

    auto activated = gb->multiply(gb->gelu(gate), up);
    return gb->matmul(activated, layer.ffn_down_weight, true, backend);
}


size_t GemmaModel3n::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                             ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto normed = gb->rms_norm(hidden, layer.input_layernorm_weight, config_.layer_norm_eps);

    auto laurel   = build_laurel(gb, normed, layer_idx, backend);
    auto attn     = gb->rms_norm(build_attention(gb, normed, layer_idx, backend, use_cache, position_offset),
                                 layer.post_attention_layernorm_weight, config_.layer_norm_eps);
    auto combined = gb->add(gb->add(hidden, attn), laurel);
    auto residual = gb->scalar_multiply(combined, RSQRT2);

    auto pre_mlp = gb->rms_norm(residual, layer.pre_feedforward_layernorm_weight, config_.layer_norm_eps);
    auto mlp = build_mlp(gb, pre_mlp, layer_idx, backend);
    mlp = gb->rms_norm(mlp, layer.post_feedforward_layernorm_weight, config_.layer_norm_eps);

    return gb->add(residual, mlp);
}


size_t GemmaModel3n::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (!initialized_ || !graph_handle_)
        throw std::runtime_error("Model not initialized - call init() first");
    if (tokens.empty())
        throw std::runtime_error("Token sequence cannot be empty");

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    size_t seq_len      = tokens.size();
    size_t pos_offset   = use_cache ? kv_cache_.get_total_seq_len() : 0;
    auto backend        = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;
    uint32_t num_layers = config_.num_layers;
    uint32_t pli_dim    = config_.hidden_size_per_layer_input;
    uint32_t num_altup  = config_.altup_num_inputs;

    auto token_input = gb->input({seq_len}, Precision::FP32);
    auto x = gb->scalar_multiply(gb->embedding(embedding_node_id_, token_input),
                                 std::sqrt(static_cast<float>(config_.hidden_dim)));

    auto pli_input = gb->input({seq_len}, Precision::FP32);
    auto pli_embed = gb->scalar_multiply(gb->embedding(weight_nodes_.embed_tokens_per_layer, pli_input),
                                         std::sqrt(static_cast<float>(pli_dim)));
    auto pli_proj = gb->scalar_multiply(gb->matmul(x, weight_nodes_.per_layer_model_proj, true, backend),
                                        1.0f / static_cast<float>(config_.hidden_dim));
    pli_proj = gb->reshape(pli_proj, {seq_len * num_layers, pli_dim});
    pli_proj = gb->rms_norm(pli_proj, weight_nodes_.per_layer_proj_norm, config_.layer_norm_eps);
    pli_proj = gb->reshape(pli_proj, {seq_len, num_layers * pli_dim});
    auto pli_combined = gb->scalar_multiply(gb->add(pli_proj, pli_embed), RSQRT2);

    size_t streams[4];
    streams[0] = x;
    for (uint32_t i = 1; i < num_altup; i++)
        streams[i] = build_magnitude_normalize(gb, x, gb->matmul(x, weight_nodes_.altup_proj_weights[i - 1], true, backend));

    for (uint32_t layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        auto modalities = build_altup_router_modalities(gb, streams[0], layer_idx, backend);

        size_t predictions[4];
        build_altup_predict(gb, modalities, layer_idx, streams, predictions);

        auto activated = build_transformer_block(gb, predictions[0], layer_idx, backend, use_cache, pos_offset);

        build_altup_correct(gb, activated, build_altup_router_modalities(gb, activated, layer_idx, backend),
                            layer_idx, backend, predictions, streams);

        if (pli_dim > 0)
            build_per_layer_input(gb, pli_combined, layer_idx, backend, streams);
    }

    for (uint32_t i = 1; i < num_altup; i++) {
        streams[i] = gb->matmul(streams[i], weight_nodes_.altup_unembed_proj_weights[i - 1], true, backend);
        streams[i] = build_magnitude_normalize(gb, streams[0], streams[i]);
    }

    auto hidden = streams[0];
    for (uint32_t i = 1; i < num_altup; i++)
        hidden = gb->add(hidden, streams[i]);
    hidden = gb->scalar_multiply(hidden, 1.0f / static_cast<float>(num_altup));

    std::vector<float> input_data(seq_len);
    for (size_t i = 0; i < seq_len; i++)
        input_data[i] = static_cast<float>(tokens[i]);
    gb->set_input(token_input, input_data.data(), Precision::FP32);
    gb->set_input(pli_input, input_data.data(), Precision::FP32);

    return gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);
}

}
}
