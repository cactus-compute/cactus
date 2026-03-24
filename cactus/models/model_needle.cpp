#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <stdexcept>

namespace cactus {
namespace engine {

namespace {

size_t delta_rms_norm(CactusGraph* gb, size_t input, size_t weight, float epsilon) {
    return gb->rms_norm(input, gb->scalar_add(weight, 1.0f), epsilon);
}

size_t normalize_qk_proj(CactusGraph* gb,
                         size_t proj,
                         size_t norm_weight,
                         size_t seq_len,
                         size_t num_heads,
                         size_t head_dim,
                         float epsilon) {
    proj = gb->reshape(proj, {seq_len * num_heads, head_dim});
    proj = delta_rms_norm(gb, proj, norm_weight, epsilon);
    return gb->reshape(proj, {seq_len, num_heads * head_dim});
}

} // namespace

NeedleModel::NeedleModel() : Model() {}

NeedleModel::NeedleModel(const Config& config) : Model(config) {
    weight_nodes_.encoder_layers.resize(config.num_encoder_layers);
    weight_nodes_.decoder_layers.resize(config.num_decoder_layers);
    float hd = static_cast<float>(config.attention_head_dim);
    if (hd <= 0.0f) {
        hd = 64.0f;
    }
    attention_scale_ = 1.0f / std::sqrt(hd);
    encoder_k_persistent_.assign(config.num_decoder_layers, 0);
    encoder_v_persistent_.assign(config.num_decoder_layers, 0);
}

void NeedleModel::load_weights_to_graph(CactusGraph* gb) {
    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    weight_nodes_.encoder_norm_weight = gb->mmap_weights(model_folder_path_ + "/encoder_layer_norm_weight.weights");
    weight_nodes_.decoder_norm_weight = gb->mmap_weights(model_folder_path_ + "/output_norm.weights");

    if (config_.tie_word_embeddings) {
        weight_nodes_.output_weight = embedding_node_id_;
        output_weight_node_id_ = embedding_node_id_;
    } else {
        weight_nodes_.output_weight = gb->mmap_weights(model_folder_path_ + "/output_weight.weights");
        output_weight_node_id_ = weight_nodes_.output_weight;
    }

    for (uint32_t i = 0; i < config_.num_encoder_layers; ++i) {
        auto& layer = weight_nodes_.encoder_layers[i];
        std::string prefix = model_folder_path_ + "/encoder_layer_" + std::to_string(i) + "_";
        layer.input_norm_weight = gb->mmap_weights(prefix + "input_norm.weights");
        layer.post_attn_norm_weight = gb->mmap_weights(prefix + "post_attn_norm.weights");
        layer.attn_q_weight = gb->mmap_weights(prefix + "attn_q.weights");
        layer.attn_k_weight = gb->mmap_weights(prefix + "attn_k.weights");
        layer.attn_v_weight = gb->mmap_weights(prefix + "attn_v.weights");
        layer.attn_output_weight = gb->mmap_weights(prefix + "attn_output.weights");
        layer.attn_q_norm_weight = gb->mmap_weights(prefix + "attn_q_norm.weights");
        layer.attn_k_norm_weight = gb->mmap_weights(prefix + "attn_k_norm.weights");
        layer.ffn_gate_weight = gb->mmap_weights(prefix + "ffn_gate.weights");
        layer.ffn_up_weight = gb->mmap_weights(prefix + "ffn_up.weights");
        layer.ffn_down_weight = gb->mmap_weights(prefix + "mlp_fc2.weights");
    }

    for (uint32_t i = 0; i < config_.num_decoder_layers; ++i) {
        auto& layer = weight_nodes_.decoder_layers[i];
        std::string prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";
        layer.input_norm_weight = gb->mmap_weights(prefix + "input_norm.weights");
        layer.post_attn_norm_weight = gb->mmap_weights(prefix + "post_attn_norm.weights");
        layer.final_norm_weight = gb->mmap_weights(prefix + "final_norm.weights");

        layer.self_attn_q_weight = gb->mmap_weights(prefix + "attn_q.weights");
        layer.self_attn_k_weight = gb->mmap_weights(prefix + "attn_k.weights");
        layer.self_attn_v_weight = gb->mmap_weights(prefix + "attn_v.weights");
        layer.self_attn_output_weight = gb->mmap_weights(prefix + "attn_output.weights");
        layer.self_attn_q_norm_weight = gb->mmap_weights(prefix + "attn_q_norm.weights");
        layer.self_attn_k_norm_weight = gb->mmap_weights(prefix + "attn_k_norm.weights");

        layer.encoder_attn_q_weight = gb->mmap_weights(prefix + "encoder_attn_q.weights");
        layer.encoder_attn_k_weight = gb->mmap_weights(prefix + "encoder_attn_k.weights");
        layer.encoder_attn_v_weight = gb->mmap_weights(prefix + "encoder_attn_v.weights");
        layer.encoder_attn_output_weight = gb->mmap_weights(prefix + "encoder_attn_output.weights");
        layer.encoder_attn_q_norm_weight = gb->mmap_weights(prefix + "encoder_attn_q_norm.weights");
        layer.encoder_attn_k_norm_weight = gb->mmap_weights(prefix + "encoder_attn_k_norm.weights");

        layer.ffn_gate_weight = gb->mmap_weights(prefix + "ffn_gate.weights");
        layer.ffn_up_weight = gb->mmap_weights(prefix + "ffn_up.weights");
        layer.ffn_down_weight = gb->mmap_weights(prefix + "mlp_fc2.weights");
    }
}

void NeedleModel::reset_graph_side_cache_nodes() {
    cache_k_output_nodes_.assign(config_.num_decoder_layers, 0);
    cache_v_output_nodes_.assign(config_.num_decoder_layers, 0);
}

void NeedleModel::reset_cache() {
    Model::reset_cache();
    encoder_ready_ = false;

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    if (gb) {
        if (last_encoder_post_norm_node_ != 0) {
            gb->invalidate_persistent(last_encoder_post_norm_node_);
            last_encoder_post_norm_node_ = 0;
        }
        for (size_t i = 0; i < encoder_k_persistent_.size(); ++i) {
            if (encoder_k_persistent_[i] != 0) {
                gb->invalidate_persistent(encoder_k_persistent_[i]);
                encoder_k_persistent_[i] = 0;
            }
            if (encoder_v_persistent_[i] != 0) {
                gb->invalidate_persistent(encoder_v_persistent_[i]);
                encoder_v_persistent_[i] = 0;
            }
        }
    }

    reset_graph_side_cache_nodes();
}

size_t NeedleModel::build_encoder_self_attention(CactusGraph* gb,
                                                 size_t input,
                                                 uint32_t layer_idx,
                                                 ComputeBackend backend,
                                                 bool use_cache,
                                                 size_t /*position_offset*/) {
    if (use_cache) {
        throw std::runtime_error("Needle encoder attention does not support KV caching");
    }

    const auto& layer = weight_nodes_.encoder_layers[layer_idx];
    auto q_proj = gb->matmul(input, layer.attn_q_weight, true, backend);
    auto k_proj = gb->matmul(input, layer.attn_k_weight, true, backend);
    auto v_proj = gb->matmul(input, layer.attn_v_weight, true, backend);

    const auto& q_shape = gb->get_output_buffer(q_proj).shape;
    if (q_shape.size() != 2) {
        throw std::runtime_error("Needle encoder self-attn expects [T, D] input");
    }

    size_t seq_len = q_shape[0];
    size_t num_heads = config_.attention_heads;
    size_t num_kv_heads = config_.attention_kv_heads;
    size_t head_dim = config_.attention_head_dim;

    q_proj = normalize_qk_proj(
        gb, q_proj, layer.attn_q_norm_weight, seq_len, num_heads, head_dim, config_.layer_norm_eps);
    k_proj = normalize_qk_proj(
        gb, k_proj, layer.attn_k_norm_weight, seq_len, num_kv_heads, head_dim, config_.layer_norm_eps);

    auto q_4d = gb->reshape(q_proj, {1, seq_len, num_heads, head_dim});
    auto k_4d = gb->reshape(k_proj, {1, seq_len, num_kv_heads, head_dim});
    auto v_4d = gb->reshape(v_proj, {1, seq_len, num_kv_heads, head_dim});

    if (config_.rope_theta > 0) {
        q_4d = gb->rope(q_4d, config_.rope_theta, 0);
        k_4d = gb->rope(k_4d, config_.rope_theta, 0);
    }

    auto attn = gb->attention(q_4d, k_4d, v_4d, attention_scale_, false);
    attn = gb->reshape(attn, {seq_len, num_heads * head_dim});
    return gb->matmul(attn, layer.attn_output_weight, true, backend);
}

size_t NeedleModel::build_decoder_self_attention(CactusGraph* gb,
                                                 size_t input,
                                                 uint32_t layer_idx,
                                                 ComputeBackend backend,
                                                 bool use_cache,
                                                 size_t position_offset) {
    const auto& layer = weight_nodes_.decoder_layers[layer_idx];
    auto q_proj = gb->matmul(input, layer.self_attn_q_weight, true, backend);
    auto k_proj = gb->matmul(input, layer.self_attn_k_weight, true, backend);
    auto v_proj = gb->matmul(input, layer.self_attn_v_weight, true, backend);

    const auto& q_shape = gb->get_output_buffer(q_proj).shape;
    if (q_shape.size() != 2) {
        throw std::runtime_error("Needle decoder self-attn expects [T, D] input");
    }

    size_t seq_new = q_shape[0];
    size_t num_heads = config_.attention_heads;
    size_t num_kv_heads = config_.attention_kv_heads;
    size_t head_dim = config_.attention_head_dim;

    q_proj = normalize_qk_proj(
        gb, q_proj, layer.self_attn_q_norm_weight, seq_new, num_heads, head_dim, config_.layer_norm_eps);
    k_proj = normalize_qk_proj(
        gb, k_proj, layer.self_attn_k_norm_weight, seq_new, num_kv_heads, head_dim, config_.layer_norm_eps);

    auto q_4d = gb->reshape(q_proj, {1, seq_new, num_heads, head_dim});
    auto k_4d = gb->reshape(k_proj, {1, seq_new, num_kv_heads, head_dim});
    auto v_4d = gb->reshape(v_proj, {1, seq_new, num_kv_heads, head_dim});

    if (config_.rope_theta > 0) {
        q_4d = gb->rope(q_4d, config_.rope_theta, position_offset);
        k_4d = gb->rope(k_4d, config_.rope_theta, position_offset);
    }

    size_t final_k = k_4d;
    size_t final_v = v_4d;

    if (use_cache && !kv_cache_.is_empty()) {
        auto k_view = kv_cache_.get_key_view(layer_idx);
        auto v_view = kv_cache_.get_value_view(layer_idx);

        if (!k_view.ptr1 || !v_view.ptr1) {
            throw std::runtime_error("Needle KV cache view missing");
        }

        size_t cache_len = kv_cache_.current_seq_len;
        size_t cache_k_node = gb->input({1, cache_len, num_kv_heads, head_dim}, kv_cache_.precision);
        size_t cache_v_node = gb->input({1, cache_len, num_kv_heads, head_dim}, kv_cache_.precision);

        if (k_view.ptr2 == nullptr && v_view.ptr2 == nullptr) {
            gb->set_external_input(cache_k_node, const_cast<void*>(k_view.ptr1), kv_cache_.precision);
            gb->set_external_input(cache_v_node, const_cast<void*>(v_view.ptr1), kv_cache_.precision);
        } else {
            gb->set_external_input(cache_k_node, kv_cache_.get_key_ptr(layer_idx), kv_cache_.precision);
            gb->set_external_input(cache_v_node, kv_cache_.get_value_ptr(layer_idx), kv_cache_.precision);
        }

        final_k = gb->concat(cache_k_node, k_4d, 1);
        final_v = gb->concat(cache_v_node, v_4d, 1);
    }

    if (use_cache) {
        cache_k_output_nodes_[layer_idx] = final_k;
        cache_v_output_nodes_[layer_idx] = final_v;
    } else {
        cache_k_output_nodes_[layer_idx] = k_4d;
        cache_v_output_nodes_[layer_idx] = v_4d;
    }

    auto attn = gb->attention(q_4d, final_k, final_v, attention_scale_, position_offset);
    attn = gb->reshape(attn, {seq_new, num_heads * head_dim});
    return gb->matmul(attn, layer.self_attn_output_weight, true, backend);
}

size_t NeedleModel::build_decoder_cross_attention(CactusGraph* gb,
                                                  size_t input,
                                                  uint32_t layer_idx,
                                                  ComputeBackend backend,
                                                  bool /*use_cache*/,
                                                  size_t /*position_offset*/) {
    if (last_encoder_post_norm_node_ == 0) {
        throw std::runtime_error("Needle encoder outputs are not available for cross-attention");
    }

    const auto& layer = weight_nodes_.decoder_layers[layer_idx];
    auto q_proj = gb->matmul(input, layer.encoder_attn_q_weight, true, backend);

    const auto& q_shape = gb->get_output_buffer(q_proj).shape;
    if (q_shape.size() != 2) {
        throw std::runtime_error("Needle decoder cross-attn expects [T_dec, D] query input");
    }

    size_t seq_dec = q_shape[0];
    size_t num_heads = config_.attention_heads;
    size_t num_kv_heads = config_.attention_kv_heads;
    size_t head_dim = config_.attention_head_dim;

    q_proj = normalize_qk_proj(
        gb, q_proj, layer.encoder_attn_q_norm_weight, seq_dec, num_heads, head_dim, config_.layer_norm_eps);
    auto q_4d = gb->reshape(q_proj, {1, seq_dec, num_heads, head_dim});

    size_t k_4d = 0;
    size_t v_4d = 0;
    bool has_persistent = encoder_k_persistent_[layer_idx] != 0 &&
                          gb->is_populated(encoder_k_persistent_[layer_idx]);

    if (has_persistent) {
        k_4d = encoder_k_persistent_[layer_idx];
        v_4d = encoder_v_persistent_[layer_idx];
    } else {
        auto k_proj = gb->matmul(last_encoder_post_norm_node_, layer.encoder_attn_k_weight, true, backend);
        auto v_proj = gb->matmul(last_encoder_post_norm_node_, layer.encoder_attn_v_weight, true, backend);

        const auto& k_shape = gb->get_output_buffer(k_proj).shape;
        if (k_shape.size() != 2) {
            throw std::runtime_error("Needle decoder cross-attn expects [T_enc, D] encoder input");
        }

        size_t seq_enc = k_shape[0];
        k_proj = normalize_qk_proj(
            gb, k_proj, layer.encoder_attn_k_norm_weight, seq_enc, num_kv_heads, head_dim, config_.layer_norm_eps);

        k_4d = gb->reshape(k_proj, {1, seq_enc, num_kv_heads, head_dim});
        v_4d = gb->reshape(v_proj, {1, seq_enc, num_kv_heads, head_dim});

        if (encoder_k_persistent_[layer_idx] == 0) {
            encoder_k_persistent_[layer_idx] = gb->persistent(k_4d);
            encoder_v_persistent_[layer_idx] = gb->persistent(v_4d);
        }

        k_4d = encoder_k_persistent_[layer_idx];
        v_4d = encoder_v_persistent_[layer_idx];
    }

    auto attn = gb->attention(q_4d, k_4d, v_4d, attention_scale_, false);
    attn = gb->reshape(attn, {seq_dec, num_heads * head_dim});
    return gb->matmul(attn, layer.encoder_attn_output_weight, true, backend);
}

size_t NeedleModel::build_encoder_mlp(CactusGraph* gb,
                                      size_t input,
                                      uint32_t layer_idx,
                                      ComputeBackend backend) const {
    const auto& layer = weight_nodes_.encoder_layers[layer_idx];
    auto gate = gb->matmul(input, layer.ffn_gate_weight, true, backend);
    auto up = gb->matmul(input, layer.ffn_up_weight, true, backend);
    gate = config_.encoder_act_gelu ? gb->gelu(gate) : gb->silu(gate);
    auto hidden = gb->multiply(gate, up);
    return gb->matmul(hidden, layer.ffn_down_weight, true, backend);
}

size_t NeedleModel::build_decoder_mlp(CactusGraph* gb,
                                      size_t input,
                                      uint32_t layer_idx,
                                      ComputeBackend backend) const {
    const auto& layer = weight_nodes_.decoder_layers[layer_idx];
    auto gate = gb->matmul(input, layer.ffn_gate_weight, true, backend);
    auto up = gb->matmul(input, layer.ffn_up_weight, true, backend);
    gate = config_.decoder_act_gelu ? gb->gelu(gate) : gb->silu(gate);
    auto hidden = gb->multiply(gate, up);
    return gb->matmul(hidden, layer.ffn_down_weight, true, backend);
}

size_t NeedleModel::build_encoder_transformer_block(CactusGraph* gb,
                                                    size_t hidden,
                                                    uint32_t layer_idx,
                                                    ComputeBackend backend,
                                                    bool use_cache,
                                                    size_t position_offset) {
    const auto& layer = weight_nodes_.encoder_layers[layer_idx];
    auto input_norm = delta_rms_norm(gb, hidden, layer.input_norm_weight, config_.layer_norm_eps);
    auto attn = build_encoder_self_attention(gb, input_norm, layer_idx, backend, use_cache, position_offset);
    auto after_attn = gb->add_clipped(hidden, attn);
    auto post_attn_norm = delta_rms_norm(gb, after_attn, layer.post_attn_norm_weight, config_.layer_norm_eps);
    auto mlp = build_encoder_mlp(gb, post_attn_norm, layer_idx, backend);
    return gb->add_clipped(after_attn, mlp);
}

size_t NeedleModel::build_decoder_transformer_block(CactusGraph* gb,
                                                    size_t hidden,
                                                    uint32_t layer_idx,
                                                    ComputeBackend backend,
                                                    bool use_cache,
                                                    size_t position_offset) {
    const auto& layer = weight_nodes_.decoder_layers[layer_idx];
    auto input_norm = delta_rms_norm(gb, hidden, layer.input_norm_weight, config_.layer_norm_eps);
    auto self_attn = build_decoder_self_attention(gb, input_norm, layer_idx, backend, use_cache, position_offset);
    auto after_self_attn = gb->add_clipped(hidden, self_attn);
    auto post_attn_norm = delta_rms_norm(gb, after_self_attn, layer.post_attn_norm_weight, config_.layer_norm_eps);
    auto cross_attn = build_decoder_cross_attention(gb, post_attn_norm, layer_idx, backend, use_cache, position_offset);
    auto after_cross_attn = gb->add_clipped(after_self_attn, cross_attn);
    auto final_norm = delta_rms_norm(gb, after_cross_attn, layer.final_norm_weight, config_.layer_norm_eps);
    auto mlp = build_decoder_mlp(gb, final_norm, layer_idx, backend);
    return gb->add_clipped(after_cross_attn, mlp);
}

void NeedleModel::run_encoder(const std::vector<uint32_t>& encoder_tokens) {
    if (encoder_tokens.empty()) {
        throw std::runtime_error("Needle encoder token sequence cannot be empty");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    size_t seq_len = encoder_tokens.size();
    size_t input_node = gb->input({seq_len}, Precision::FP32);
    auto hidden = gb->embedding(embedding_node_id_, input_node);
    hidden = gb->scalar_multiply(hidden, std::sqrt(static_cast<float>(config_.hidden_dim)));

    std::vector<float> input_data(seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        input_data[i] = static_cast<float>(encoder_tokens[i]);
    }
    gb->set_input(input_node, input_data.data(), Precision::FP32);

    for (uint32_t layer_idx = 0; layer_idx < config_.num_encoder_layers; ++layer_idx) {
        hidden = build_encoder_transformer_block(gb, hidden, layer_idx, backend, false, 0);
    }

    auto encoder_norm = delta_rms_norm(gb, hidden, weight_nodes_.encoder_norm_weight, config_.layer_norm_eps);
    last_encoder_post_norm_node_ = gb->persistent(encoder_norm);
}

size_t NeedleModel::run_decoder_step(const std::vector<uint32_t>& tokens, bool use_cache, bool last_token_only) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    if (tokens.empty()) {
        throw std::runtime_error("Needle decoder token sequence cannot be empty");
    }

    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    size_t new_tokens = tokens.size();
    size_t position_offset = use_cache ? kv_cache_.current_seq_len : 0;

    size_t tok_input = gb->input({new_tokens}, Precision::FP32);
    std::vector<float> tok_f(new_tokens);
    for (size_t i = 0; i < new_tokens; ++i) {
        tok_f[i] = static_cast<float>(tokens[i]);
    }
    gb->set_input(tok_input, tok_f.data(), Precision::FP32);

    auto hidden = gb->embedding(embedding_node_id_, tok_input);
    hidden = gb->scalar_multiply(hidden, std::sqrt(static_cast<float>(config_.hidden_dim)));

    for (uint32_t layer_idx = 0; layer_idx < config_.num_decoder_layers; ++layer_idx) {
        hidden = build_decoder_transformer_block(gb, hidden, layer_idx, backend, use_cache, position_offset);
    }

    auto decoder_norm = delta_rms_norm(gb, hidden, weight_nodes_.decoder_norm_weight, config_.layer_norm_eps);
    size_t logits_input = decoder_norm;
    if (last_token_only) {
        logits_input = gb->slice(logits_input, 0, new_tokens - 1, 1);
    }

    auto logits = gb->matmul(logits_input, output_weight_node_id_, true, backend);
    return logits;
}

size_t NeedleModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Needle model not initialized");
    }
    if (tokens.empty()) {
        throw std::runtime_error("Needle token sequence cannot be empty");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    reset_graph_side_cache_nodes();

    if (!encoder_ready_ || !use_cache) {
        if (tokens.size() < 2) {
            throw std::runtime_error("Needle forward requires encoder tokens plus decoder seed");
        }
        std::vector<uint32_t> encoder_tokens(tokens.begin(), tokens.end() - 1);
        std::vector<uint32_t> decoder_seed(tokens.end() - 1, tokens.end());

        if (!use_cache) {
            reset_cache();
            gb->soft_reset();
        }

        run_encoder(encoder_tokens);
        encoder_ready_ = true;
        return run_decoder_step(decoder_seed, false, false);
    }

    return run_decoder_step(tokens, true, tokens.size() == 1);
}

void NeedleModel::prefill(const std::vector<uint32_t>& tokens, size_t /*chunk_size*/, const std::string& profile_file) {
    if (tokens.empty()) {
        return;
    }
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Needle model not initialized");
    }

    reset_cache();
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    run_encoder(tokens);

    if (!profile_file.empty()) {
        gb->execute(profile_file);
    } else {
        gb->execute();
    }

    encoder_ready_ = true;
}

uint32_t NeedleModel::decode(const std::vector<uint32_t>& tokens,
                             float temperature,
                             float top_p,
                             size_t top_k,
                             const std::string& profile_file,
                             float* out_entropy) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Needle model not initialized");
    }
    if (tokens.empty()) {
        throw std::runtime_error("Needle decode requires at least one decoder token");
    }

    if (temperature < 0) {
        temperature = config_.default_temperature;
    }
    if (top_p < 0) {
        top_p = config_.default_top_p;
    }
    if (top_k == 0) {
        top_k = config_.default_top_k;
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    reset_graph_side_cache_nodes();

    bool cold_start = kv_cache_.is_empty();
    std::vector<uint32_t> cold_start_tokens;
    const std::vector<uint32_t>* decode_tokens = &tokens;
    if (!encoder_ready_) {
        if (tokens.size() < 2) {
            throw std::runtime_error("Needle decode requires prefill or combined encoder+decoder tokens");
        }

        reset_cache();
        gb->soft_reset();

        std::vector<uint32_t> encoder_tokens(tokens.begin(), tokens.end() - 1);
        cold_start_tokens.assign(tokens.end() - 1, tokens.end());
        run_encoder(encoder_tokens);
        encoder_ready_ = true;
        decode_tokens = &cold_start_tokens;
        cold_start = true;
    }

    size_t logits_node = cold_start
        ? run_decoder_step(*decode_tokens, false, false)
        : run_decoder_step(tokens, true, tokens.size() == 1);

    if (config_.final_logit_softcapping > 0.0f) {
        float inv_cap = 1.0f / config_.final_logit_softcapping;
        logits_node = gb->scalar_multiply(logits_node, inv_cap);
        logits_node = gb->tanh(logits_node);
        logits_node = gb->scalar_multiply(logits_node, config_.final_logit_softcapping);
    }

    auto sampled_token_id = sample_token(gb, logits_node, temperature, top_p, top_k);

    if (!profile_file.empty()) {
        gb->execute(profile_file);
    } else {
        gb->execute();
    }

    compute_entropy(gb, logits_node, out_entropy);
    post_execute_updates(gb, tokens.size());
    update_kv_cache(gb, decode_tokens->size());

    auto* output_ptr = gb->get_output(sampled_token_id);
    return *static_cast<uint32_t*>(output_ptr);
}

} // namespace engine
} // namespace cactus
