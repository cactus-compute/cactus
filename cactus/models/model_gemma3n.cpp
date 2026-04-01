#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <stdexcept>

namespace cactus {
namespace engine {

static const float RSQRT2 = 1.0f / std::sqrt(2.0f);

GemmaModel3n::GemmaModel3n() : Model() {}

GemmaModel3n::GemmaModel3n(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
}

std::vector<size_t> GemmaModel3n::get_kv_layer_dims() const {
    uint32_t n = config_.num_layers;
    uint32_t num_shared = config_.num_kv_shared_layers;
    uint32_t first_shared = (n > num_shared) ? n - num_shared : n;

    std::vector<size_t> dims(n);
    for (uint32_t i = 0; i < n; i++) {
        dims[i] = (i >= first_shared) ? 0 : config_.attention_head_dim;
    }
    return dims;
}

void GemmaModel3n::post_init() {
    kv_cache_.set_window_size(0, 0);

    uint32_t n = config_.num_layers;
    uint32_t num_shared = config_.num_kv_shared_layers;
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
    assert(config_.altup_num_inputs == 4 && "WeightNodeIDs altup arrays assume exactly 4 AltUp streams");
    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    weight_nodes_.output_norm_weight = gb->mmap_weights(model_folder_path_ + "/output_norm.weights");
    if (config_.tie_word_embeddings) {
        weight_nodes_.output_weight = embedding_node_id_;
        output_weight_node_id_ = embedding_node_id_;
    } else {
        weight_nodes_.output_weight = gb->mmap_weights(model_folder_path_ + "/output_weight.weights");
        output_weight_node_id_ = weight_nodes_.output_weight;
    }

    for (int i = 0; i < 3; i++) {
        auto idx = std::to_string(i);
        weight_nodes_.altup_proj_weights[i] = gb->mmap_weights(model_folder_path_ + "/altup_proj_" + idx + ".weights");
        weight_nodes_.altup_unembed_proj_weights[i] = gb->mmap_weights(model_folder_path_ + "/altup_unembed_proj_" + idx + ".weights");
    }

    weight_nodes_.embed_tokens_per_layer = gb->mmap_embeddings(model_folder_path_ + "/embed_tokens_per_layer.weights");
    weight_nodes_.per_layer_model_proj = gb->mmap_weights(model_folder_path_ + "/per_layer_model_proj.weights");
    weight_nodes_.per_layer_proj_norm = gb->mmap_weights(model_folder_path_ + "/per_layer_proj_norm.weights");

    uint32_t num_shared = config_.num_kv_shared_layers;
    uint32_t first_shared = (config_.num_layers > num_shared) ? config_.num_layers - num_shared : config_.num_layers;

    for (uint32_t i = 0; i < config_.num_layers; i++) {
        auto& layer = weight_nodes_.layers[i];
        std::string prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";
        bool is_shared = (i >= first_shared);

        layer.attn_q_weight                    = gb->mmap_weights(prefix + "attn_q.weights");
        layer.attn_k_weight                    = is_shared ? 0 : gb->mmap_weights(prefix + "attn_k.weights");
        layer.attn_v_weight                    = is_shared ? 0 : gb->mmap_weights(prefix + "attn_v.weights");
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

    size_t head_dim = config_.attention_head_dim;
    v_norm_ones_weight_.assign(head_dim, static_cast<__fp16>(1.0f));
    v_norm_ones_node_ = gb->input({head_dim}, Precision::FP16);
    gb->set_external_input(v_norm_ones_node_, v_norm_ones_weight_.data(), Precision::FP16);
}


size_t GemmaModel3n::build_rms_norm_no_weight(CactusGraph* gb, size_t input, size_t num_rows, size_t row_dim) const {
    auto flat = gb->reshape(input, {num_rows, row_dim});
    return gb->rms_norm(flat, v_norm_ones_node_, config_.layer_norm_eps);
}

size_t GemmaModel3n::build_magnitude_normalize(CactusGraph* gb, size_t reference, size_t target) const {
    size_t rows = gb->get_output_buffer(reference).shape[0];
    auto ref_sq = gb->reshape(gb->mean(gb->multiply(reference, reference), 1), {rows, 1});
    auto tgt_sq = gb->reshape(gb->mean(gb->multiply(target, target), 1), {rows, 1});
    auto ref_mag = gb->scalar_sqrt(ref_sq);
    auto tgt_mag = gb->scalar_sqrt(gb->scalar_add(tgt_sq, 1e-5f));
    auto ratio = gb->divide(ref_mag, tgt_mag);
    return gb->multiply(target, ratio);
}

size_t GemmaModel3n::build_gaussian_topk(CactusGraph* gb, size_t input, float ppf) const {
    return gb->gaussian_topk(input, ppf);
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

    size_t seq_len = gb->get_output_buffer(streams[0]).shape[0];
    auto fused = gb->altup_predict(coefs, streams, n);
    for (uint32_t i = 0; i < n; i++) {
        predictions[i] = gb->slice(fused, 0, i * seq_len, seq_len);
    }
}

void GemmaModel3n::build_altup_correct(CactusGraph* gb, size_t activated, size_t modalities, uint32_t layer_idx,
                                        ComputeBackend backend, const size_t* predictions, size_t* corrected) const {
    uint32_t n = config_.altup_num_inputs;
    auto coefs = gb->scalar_add(gb->matmul(modalities, weight_nodes_.layers[layer_idx].altup_correction_coefs, true, backend), 1.0f);
    auto innovation = gb->subtract(activated, predictions[0]);

    size_t seq_len = gb->get_output_buffer(predictions[0]).shape[0];
    auto fused = gb->altup_correct(coefs, innovation, predictions, n);
    for (uint32_t i = 0; i < n; i++) {
        corrected[i] = gb->slice(fused, 0, i * seq_len, seq_len);
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
    size_t window     = is_global ? 0 : config_.sliding_window;

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

        auto v_proj = gb->matmul(input, layer.attn_v_weight, true, backend);
        auto v = build_rms_norm_no_weight(gb, v_proj, seq_len * kv_heads, head_dim);
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
        attn = gb->attention_int8_hybrid(
            q4, k4, v4, attention_scale_, position_offset,
            kv_cache_.get_keys_int8(cache_src), kv_cache_.get_values_int8(cache_src),
            kv_cache_.get_key_scales(cache_src), kv_cache_.get_value_scales(cache_src),
            kv_cache_.current_seq_len, kv_heads, head_dim, window);
    } else {
        attn = gb->attention(q4, k4, v4, attention_scale_, position_offset, window);
    }

    auto o_proj = gb->matmul(gb->reshape(attn, {seq_len, num_heads * head_dim}), layer.attn_output_weight, true, backend);
    return o_proj;
}


size_t GemmaModel3n::build_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx,
                               ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];

    auto gate = gb->matmul(input, layer.ffn_gate_weight, true, backend);
    auto up   = gb->matmul(input, layer.ffn_up_weight, true, backend);

    if (layer_idx < config_.activation_sparsity_ppf.size() && config_.activation_sparsity_ppf[layer_idx] > 0.0f)
        gate = build_gaussian_topk(gb, gate, config_.activation_sparsity_ppf[layer_idx]);

    auto gate_activated = gb->gelu(gate);
    auto activated = gb->multiply(gate_activated, up);
    return gb->matmul(activated, layer.ffn_down_weight, true, backend);
}


size_t GemmaModel3n::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                             ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto normed = gb->rms_norm(hidden, layer.input_layernorm_weight, config_.layer_norm_eps);

    auto laurel   = build_laurel(gb, normed, layer_idx, backend);
    auto attn_raw = build_attention(gb, normed, layer_idx, backend, use_cache, position_offset);
    auto attn     = gb->rms_norm(attn_raw, layer.post_attention_layernorm_weight, config_.layer_norm_eps);
    auto combined = gb->add(gb->add(hidden, attn), laurel);
    auto residual = gb->scalar_multiply(combined, RSQRT2);

    auto pre_mlp = gb->rms_norm(residual, layer.pre_feedforward_layernorm_weight, config_.layer_norm_eps);
    auto mlp_raw = build_mlp(gb, pre_mlp, layer_idx, backend);
    auto mlp = gb->rms_norm(mlp_raw, layer.post_feedforward_layernorm_weight, config_.layer_norm_eps);

    auto block_out = gb->add(residual, mlp);
    return block_out;
}


size_t GemmaModel3n::build_preamble(CactusGraph* gb, size_t seq_len, ComputeBackend backend,
                                     size_t& token_input, size_t& pli_input, size_t* streams) {
    uint32_t num_layers = config_.num_layers;
    uint32_t pli_dim    = config_.hidden_size_per_layer_input;
    uint32_t num_altup  = config_.altup_num_inputs;

    token_input = gb->input({seq_len}, Precision::FP32);
    auto x = gb->scalar_multiply(gb->embedding(embedding_node_id_, token_input),
                                 std::sqrt(static_cast<float>(config_.hidden_dim)));

    pli_input = gb->input({seq_len}, Precision::FP32);
    auto pli_embed = gb->scalar_multiply(gb->embedding(weight_nodes_.embed_tokens_per_layer, pli_input),
                                         std::sqrt(static_cast<float>(pli_dim)));
    auto pli_proj = gb->scalar_multiply(gb->matmul(x, weight_nodes_.per_layer_model_proj, true, backend),
                                        1.0f / std::sqrt(static_cast<float>(config_.hidden_dim)));
    pli_proj = gb->reshape(pli_proj, {seq_len * num_layers, pli_dim});
    pli_proj = gb->rms_norm(pli_proj, weight_nodes_.per_layer_proj_norm, config_.layer_norm_eps);
    pli_proj = gb->reshape(pli_proj, {seq_len, num_layers * pli_dim});
    auto pli_combined = gb->scalar_multiply(gb->add(pli_proj, pli_embed), RSQRT2);

    streams[0] = x;
    for (uint32_t i = 1; i < num_altup; i++) {
        streams[i] = build_magnitude_normalize(gb, x, gb->matmul(x, weight_nodes_.altup_proj_weights[i - 1], true, backend));
    }

    return pli_combined;
}

void GemmaModel3n::build_layer(CactusGraph* gb, uint32_t layer_idx, ComputeBackend backend,
                                      bool use_cache, size_t pos_offset, size_t pli, size_t* streams) {
    auto modalities = build_altup_router_modalities(gb, streams[0], layer_idx, backend);

    size_t predictions[4];
    build_altup_predict(gb, modalities, layer_idx, streams, predictions);

    auto activated = build_transformer_block(gb, predictions[0], layer_idx, backend, use_cache, pos_offset);

    auto modalities_post = build_altup_router_modalities(gb, activated, layer_idx, backend);

    build_altup_correct(gb, activated, modalities_post, layer_idx, backend, predictions, streams);

    if (config_.hidden_size_per_layer_input > 0) {
        build_per_layer_input(gb, pli, layer_idx, backend, streams);
    }
}

size_t GemmaModel3n::build_output_head(CactusGraph* gb, size_t* streams, ComputeBackend backend) {
    uint32_t num_altup = config_.altup_num_inputs;

    for (uint32_t i = 1; i < num_altup; i++) {
        streams[i] = gb->matmul(streams[i], weight_nodes_.altup_unembed_proj_weights[i - 1], true, backend);
        streams[i] = build_magnitude_normalize(gb, streams[0], streams[i]);
    }

    auto hidden = streams[0];
    for (uint32_t i = 1; i < num_altup; i++)
        hidden = gb->add(hidden, streams[i]);
    hidden = gb->scalar_multiply(hidden, 1.0f / static_cast<float>(num_altup));

    auto final_normed = gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);
    return final_normed;
}

void GemmaModel3n::set_token_inputs(CactusGraph* gb, size_t token_input, size_t pli_input,
                                     const std::vector<uint32_t>& tokens) {
    std::vector<float> input_data(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++)
        input_data[i] = static_cast<float>(tokens[i]);
    gb->set_input(token_input, input_data.data(), Precision::FP32);
    gb->set_input(pli_input, input_data.data(), Precision::FP32);
}


size_t GemmaModel3n::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (!initialized_ || !graph_handle_)
        throw std::runtime_error("Model not initialized - call init() first");
    if (tokens.empty())
        throw std::runtime_error("Token sequence cannot be empty");

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    size_t pos_offset = use_cache ? kv_cache_.get_total_seq_len() : 0;
    auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;

    size_t token_input, pli_input, streams[4];
    auto pli = build_preamble(gb, tokens.size(), backend, token_input, pli_input, streams);

    for (uint32_t i = 0; i < config_.num_layers; i++)
        build_layer(gb, i, backend, use_cache, pos_offset, pli, streams);

    set_token_inputs(gb, token_input, pli_input, tokens);
    return build_output_head(gb, streams, backend);
}


size_t GemmaModel3n::forward_split(const std::vector<uint32_t>& tokens, bool use_cache) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    size_t seq_len = tokens.size();
    size_t pos_offset = use_cache ? kv_cache_.get_total_seq_len() : 0;
    auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;
    uint32_t num_layers = config_.num_layers;
    uint32_t first_shared = num_layers - config_.num_kv_shared_layers;

    size_t token_input, pli_input, streams[4];
    auto pli = build_preamble(gb, seq_len, backend, token_input, pli_input, streams);

    for (uint32_t i = 0; i < first_shared; i++)
        build_layer(gb, i, backend, use_cache, pos_offset, pli, streams);

    size_t hidden_dim = config_.hidden_dim;
    for (uint32_t i = 0; i < config_.altup_num_inputs; i++) {
        streams[i] = gb->index(streams[i], seq_len - 1, 0);
        streams[i] = gb->reshape(streams[i], {1, hidden_dim});
    }

    auto pli_last = gb->index(pli, seq_len - 1, 0);
    pli_last = gb->reshape(pli_last, {1, num_layers * config_.hidden_size_per_layer_input});

    for (uint32_t i = first_shared; i < num_layers; i++)
        build_layer(gb, i, backend, use_cache, pos_offset, pli_last, streams);

    set_token_inputs(gb, token_input, pli_input, tokens);
    return build_output_head(gb, streams, backend);
}


void GemmaModel3n::prefill(const std::vector<uint32_t>& tokens, size_t chunk_size, const std::string& profile_file) {
    if (tokens.empty())
        return;

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
    std::vector<uint32_t> final_chunk(tokens.begin() + final_start, tokens.end());
    process_chunk(final_chunk);
}

}
}
