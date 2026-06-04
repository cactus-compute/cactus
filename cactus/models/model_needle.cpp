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

size_t normalize_qk_proj(CactusGraph* gb, size_t proj, size_t norm_weight,
                         size_t seq_len, size_t num_heads, size_t head_dim, float epsilon) {
    proj = gb->reshape(proj, {seq_len * num_heads, head_dim});
    proj = delta_rms_norm(gb, proj, norm_weight, epsilon);
    return gb->reshape(proj, {seq_len, num_heads * head_dim});
}

size_t apply_residual_gate(CactusGraph* gb, size_t residual, size_t update, size_t gate_weight) {
    return gb->add_clipped(residual, gb->multiply(update, gb->sigmoid(gate_weight)));
}

size_t embed_tokens(CactusGraph* gb, size_t embedding_id, const std::vector<uint32_t>& tokens, size_t hidden_dim) {
    size_t n = tokens.size();
    size_t input_node = gb->input({n}, Precision::FP32);
    std::vector<float> data(n);
    for (size_t i = 0; i < n; ++i) data[i] = static_cast<float>(tokens[i]);
    gb->set_input(input_node, data.data(), Precision::FP32);
    return gb->scalar_multiply(gb->embedding(embedding_id, input_node), std::sqrt(static_cast<float>(hidden_dim)));
}

} // namespace

NeedleModel::NeedleModel() : Model() {}

NeedleModel::NeedleModel(const Config& config) : Model(config) {
    weight_nodes_.encoder_layers.resize(config.num_encoder_layers);
    weight_nodes_.decoder_layers.resize(config.num_decoder_layers);
    float hd = static_cast<float>(config.attention_head_dim);
    attention_scale_ = 1.0f / std::sqrt(hd > 0.0f ? hd : 64.0f);
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
        auto& l = weight_nodes_.encoder_layers[i];
        std::string p = model_folder_path_ + "/encoder_layer_" + std::to_string(i) + "_";
        l.attn_gate_weight   = gb->mmap_weights(p + "attn_gate.weights");
        l.input_norm_weight  = gb->mmap_weights(p + "input_norm.weights");
        l.attn_q_weight      = gb->mmap_weights(p + "attn_q.weights");
        l.attn_k_weight      = gb->mmap_weights(p + "attn_k.weights");
        l.attn_v_weight      = gb->mmap_weights(p + "attn_v.weights");
        l.attn_output_weight = gb->mmap_weights(p + "attn_output.weights");
        l.attn_q_norm_weight = gb->mmap_weights(p + "attn_q_norm.weights");
        l.attn_k_norm_weight = gb->mmap_weights(p + "attn_k_norm.weights");
    }

    for (uint32_t i = 0; i < config_.num_decoder_layers; ++i) {
        auto& l = weight_nodes_.decoder_layers[i];
        std::string p = model_folder_path_ + "/layer_" + std::to_string(i) + "_";
        l.self_attn_gate_weight  = gb->mmap_weights(p + "self_attn_gate.weights");
        l.cross_attn_gate_weight = gb->mmap_weights(p + "cross_attn_gate.weights");
        l.input_norm_weight      = gb->mmap_weights(p + "input_norm.weights");
        l.post_attn_norm_weight  = gb->mmap_weights(p + "post_attn_norm.weights");
        l.self_attn_q_weight      = gb->mmap_weights(p + "attn_q.weights");
        l.self_attn_k_weight      = gb->mmap_weights(p + "attn_k.weights");
        l.self_attn_v_weight      = gb->mmap_weights(p + "attn_v.weights");
        l.self_attn_output_weight = gb->mmap_weights(p + "attn_output.weights");
        l.self_attn_q_norm_weight = gb->mmap_weights(p + "attn_q_norm.weights");
        l.self_attn_k_norm_weight = gb->mmap_weights(p + "attn_k_norm.weights");
        l.encoder_attn_q_weight      = gb->mmap_weights(p + "encoder_attn_q.weights");
        l.encoder_attn_k_weight      = gb->mmap_weights(p + "encoder_attn_k.weights");
        l.encoder_attn_v_weight      = gb->mmap_weights(p + "encoder_attn_v.weights");
        l.encoder_attn_output_weight = gb->mmap_weights(p + "encoder_attn_output.weights");
        l.encoder_attn_q_norm_weight = gb->mmap_weights(p + "encoder_attn_q_norm.weights");
        l.encoder_attn_k_norm_weight = gb->mmap_weights(p + "encoder_attn_k_norm.weights");
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
    if (!gb) { reset_graph_side_cache_nodes(); return; }

    if (last_encoder_post_norm_node_ != 0) {
        gb->invalidate_persistent(last_encoder_post_norm_node_);
        last_encoder_post_norm_node_ = 0;
    }
    for (size_t i = 0; i < encoder_k_persistent_.size(); ++i) {
        if (encoder_k_persistent_[i] != 0) { gb->invalidate_persistent(encoder_k_persistent_[i]); encoder_k_persistent_[i] = 0; }
        if (encoder_v_persistent_[i] != 0) { gb->invalidate_persistent(encoder_v_persistent_[i]); encoder_v_persistent_[i] = 0; }
    }
    reset_graph_side_cache_nodes();
}

size_t NeedleModel::build_encoder_self_attention(CactusGraph* gb, size_t input,
                                                 uint32_t layer_idx, ComputeBackend backend,
                                                 bool /*use_cache*/, size_t /*position_offset*/) {
    const auto& l = weight_nodes_.encoder_layers[layer_idx];
    auto q = gb->matmul(input, l.attn_q_weight, true, backend);
    auto k = gb->matmul(input, l.attn_k_weight, true, backend);
    auto v = gb->matmul(input, l.attn_v_weight, true, backend);

    size_t seq = gb->get_output_buffer(q).shape[0];
    size_t nh = config_.attention_heads, nkv = config_.attention_kv_heads, hd = config_.attention_head_dim;

    q = normalize_qk_proj(gb, q, l.attn_q_norm_weight, seq, nh, hd, config_.layer_norm_eps);
    k = normalize_qk_proj(gb, k, l.attn_k_norm_weight, seq, nkv, hd, config_.layer_norm_eps);

    auto q4 = gb->reshape(q, {1, seq, nh, hd});
    auto k4 = gb->reshape(k, {1, seq, nkv, hd});
    auto v4 = gb->reshape(v, {1, seq, nkv, hd});
    if (config_.rope_theta > 0) { q4 = gb->rope(q4, config_.rope_theta, 0); k4 = gb->rope(k4, config_.rope_theta, 0); }

    auto attn = gb->attention(q4, k4, v4, attention_scale_, false);
    return gb->matmul(gb->reshape(attn, {seq, nh * hd}), l.attn_output_weight, true, backend);
}

size_t NeedleModel::build_decoder_self_attention(CactusGraph* gb, size_t input,
                                                 uint32_t layer_idx, ComputeBackend backend,
                                                 bool use_cache, size_t position_offset) {
    const auto& l = weight_nodes_.decoder_layers[layer_idx];
    auto q = gb->matmul(input, l.self_attn_q_weight, true, backend);
    auto k = gb->matmul(input, l.self_attn_k_weight, true, backend);
    auto v = gb->matmul(input, l.self_attn_v_weight, true, backend);

    size_t seq = gb->get_output_buffer(q).shape[0];
    size_t nh = config_.attention_heads, nkv = config_.attention_kv_heads, hd = config_.attention_head_dim;

    q = normalize_qk_proj(gb, q, l.self_attn_q_norm_weight, seq, nh, hd, config_.layer_norm_eps);
    k = normalize_qk_proj(gb, k, l.self_attn_k_norm_weight, seq, nkv, hd, config_.layer_norm_eps);

    auto q4 = gb->reshape(q, {1, seq, nh, hd});
    auto k4 = gb->reshape(k, {1, seq, nkv, hd});
    auto v4 = gb->reshape(v, {1, seq, nkv, hd});
    if (config_.rope_theta > 0) { q4 = gb->rope(q4, config_.rope_theta, position_offset); k4 = gb->rope(k4, config_.rope_theta, position_offset); }

    size_t final_k = k4, final_v = v4;
    if (use_cache && !kv_cache_.is_empty()) {
        size_t cache_len = kv_cache_.current_seq_len;
        size_t ck = gb->input({1, cache_len, nkv, hd}, kv_cache_.precision);
        size_t cv = gb->input({1, cache_len, nkv, hd}, kv_cache_.precision);
        auto kv = kv_cache_.get_key_view(layer_idx);
        auto vv = kv_cache_.get_value_view(layer_idx);
        if (!kv.ptr2 && !vv.ptr2) {
            gb->set_external_input(ck, const_cast<void*>(kv.ptr1), kv_cache_.precision);
            gb->set_external_input(cv, const_cast<void*>(vv.ptr1), kv_cache_.precision);
        } else {
            gb->set_external_input(ck, kv_cache_.get_key_ptr(layer_idx), kv_cache_.precision);
            gb->set_external_input(cv, kv_cache_.get_value_ptr(layer_idx), kv_cache_.precision);
        }
        final_k = gb->concat(ck, k4, 1);
        final_v = gb->concat(cv, v4, 1);
    }

    cache_k_output_nodes_[layer_idx] = use_cache ? final_k : k4;
    cache_v_output_nodes_[layer_idx] = use_cache ? final_v : v4;

    auto attn = gb->attention(q4, final_k, final_v, attention_scale_, position_offset);
    return gb->matmul(gb->reshape(attn, {seq, nh * hd}), l.self_attn_output_weight, true, backend);
}

size_t NeedleModel::build_decoder_cross_attention(CactusGraph* gb, size_t input,
                                                  uint32_t layer_idx, ComputeBackend backend,
                                                  bool /*use_cache*/, size_t /*position_offset*/) {
    const auto& l = weight_nodes_.decoder_layers[layer_idx];
    size_t nh = config_.attention_heads, nkv = config_.attention_kv_heads, hd = config_.attention_head_dim;

    auto q = gb->matmul(input, l.encoder_attn_q_weight, true, backend);
    size_t seq_dec = gb->get_output_buffer(q).shape[0];
    q = normalize_qk_proj(gb, q, l.encoder_attn_q_norm_weight, seq_dec, nh, hd, config_.layer_norm_eps);
    auto q4 = gb->reshape(q, {1, seq_dec, nh, hd});

    size_t k4 = 0, v4 = 0;
    bool cached = encoder_k_persistent_[layer_idx] != 0 && gb->is_populated(encoder_k_persistent_[layer_idx]);
    if (cached) {
        k4 = encoder_k_persistent_[layer_idx];
        v4 = encoder_v_persistent_[layer_idx];
    } else {
        auto kp = gb->matmul(last_encoder_post_norm_node_, l.encoder_attn_k_weight, true, backend);
        auto vp = gb->matmul(last_encoder_post_norm_node_, l.encoder_attn_v_weight, true, backend);
        size_t seq_enc = gb->get_output_buffer(kp).shape[0];
        kp = normalize_qk_proj(gb, kp, l.encoder_attn_k_norm_weight, seq_enc, nkv, hd, config_.layer_norm_eps);
        k4 = gb->reshape(kp, {1, seq_enc, nkv, hd});
        v4 = gb->reshape(vp, {1, seq_enc, nkv, hd});
        if (encoder_k_persistent_[layer_idx] == 0) {
            encoder_k_persistent_[layer_idx] = gb->persistent(k4);
            encoder_v_persistent_[layer_idx] = gb->persistent(v4);
        }
        k4 = encoder_k_persistent_[layer_idx];
        v4 = encoder_v_persistent_[layer_idx];
    }

    auto attn = gb->attention(q4, k4, v4, attention_scale_, false);
    return gb->matmul(gb->reshape(attn, {seq_dec, nh * hd}), l.encoder_attn_output_weight, true, backend);
}

size_t NeedleModel::build_encoder_transformer_block(CactusGraph* gb, size_t hidden,
                                                    uint32_t layer_idx, ComputeBackend backend,
                                                    bool use_cache, size_t position_offset) {
    const auto& l = weight_nodes_.encoder_layers[layer_idx];
    auto normed = delta_rms_norm(gb, hidden, l.input_norm_weight, config_.layer_norm_eps);
    auto attn = build_encoder_self_attention(gb, normed, layer_idx, backend, use_cache, position_offset);
    return apply_residual_gate(gb, hidden, attn, l.attn_gate_weight);
}

size_t NeedleModel::build_decoder_transformer_block(CactusGraph* gb, size_t hidden,
                                                    uint32_t layer_idx, ComputeBackend backend,
                                                    bool use_cache, size_t position_offset) {
    const auto& l = weight_nodes_.decoder_layers[layer_idx];
    auto normed = delta_rms_norm(gb, hidden, l.input_norm_weight, config_.layer_norm_eps);
    auto self_out = build_decoder_self_attention(gb, normed, layer_idx, backend, use_cache, position_offset);
    auto after_self = apply_residual_gate(gb, hidden, self_out, l.self_attn_gate_weight);
    auto cross_in = delta_rms_norm(gb, after_self, l.post_attn_norm_weight, config_.layer_norm_eps);
    auto cross_out = build_decoder_cross_attention(gb, cross_in, layer_idx, backend, use_cache, position_offset);
    return apply_residual_gate(gb, after_self, cross_out, l.cross_attn_gate_weight);
}

void NeedleModel::run_encoder(const std::vector<uint32_t>& encoder_tokens) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;
    auto hidden = embed_tokens(gb, embedding_node_id_, encoder_tokens, config_.hidden_dim);

    for (uint32_t i = 0; i < config_.num_encoder_layers; ++i)
        hidden = build_encoder_transformer_block(gb, hidden, i, backend, false, 0);

    last_encoder_post_norm_node_ = gb->persistent(
        delta_rms_norm(gb, hidden, weight_nodes_.encoder_norm_weight, config_.layer_norm_eps));
}

size_t NeedleModel::run_decoder_step(const std::vector<uint32_t>& tokens, bool use_cache, bool last_token_only) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;
    size_t pos_offset = use_cache ? kv_cache_.current_seq_len : 0;

    auto hidden = embed_tokens(gb, embedding_node_id_, tokens, config_.hidden_dim);
    for (uint32_t i = 0; i < config_.num_decoder_layers; ++i)
        hidden = build_decoder_transformer_block(gb, hidden, i, backend, use_cache, pos_offset);

    auto norm = delta_rms_norm(gb, hidden, weight_nodes_.decoder_norm_weight, config_.layer_norm_eps);
    if (last_token_only) norm = gb->slice(norm, 0, tokens.size() - 1, 1);
    return gb->matmul(norm, output_weight_node_id_, true, backend);
}

size_t NeedleModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    reset_graph_side_cache_nodes();

    if (!encoder_ready_ || !use_cache) {
        std::vector<uint32_t> enc(tokens.begin(), tokens.end() - 1);
        std::vector<uint32_t> seed(tokens.end() - 1, tokens.end());
        if (!use_cache) { reset_cache(); gb->soft_reset(); }
        run_encoder(enc);
        encoder_ready_ = true;
        return run_decoder_step(seed, false, false);
    }
    return run_decoder_step(tokens, true, tokens.size() == 1);
}

void NeedleModel::prefill(const std::vector<uint32_t>& tokens, size_t /*chunk_size*/, const std::string& profile_file) {
    if (tokens.empty()) return;
    reset_cache();
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    run_encoder(tokens);
    if (!profile_file.empty()) gb->execute(profile_file); else gb->execute();
    encoder_ready_ = true;
}

uint32_t NeedleModel::decode(const std::vector<uint32_t>& tokens,
                             float temperature, float top_p, size_t top_k,
                             const std::string& profile_file, float* out_entropy,
                             float /*min_p*/, float /*repetition_penalty*/) {
    if (temperature < 0) temperature = config_.default_temperature;
    if (top_p < 0) top_p = config_.default_top_p;
    if (top_k == 0) top_k = config_.default_top_k;

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    reset_graph_side_cache_nodes();

    bool cold_start = kv_cache_.is_empty();
    std::vector<uint32_t> cold_tokens;
    const std::vector<uint32_t>* dec_tokens = &tokens;
    if (!encoder_ready_) {
        reset_cache(); gb->soft_reset();
        std::vector<uint32_t> enc(tokens.begin(), tokens.end() - 1);
        cold_tokens.assign(tokens.end() - 1, tokens.end());
        run_encoder(enc);
        encoder_ready_ = true;
        dec_tokens = &cold_tokens;
        cold_start = true;
    }

    size_t logits = cold_start
        ? run_decoder_step(*dec_tokens, false, false)
        : run_decoder_step(tokens, true, tokens.size() == 1);

    if (config_.final_logit_softcapping > 0.0f) {
        float inv = 1.0f / config_.final_logit_softcapping;
        logits = gb->scalar_multiply(gb->tanh(gb->scalar_multiply(logits, inv)), config_.final_logit_softcapping);
    }

    auto sampled = sample_token(gb, logits, temperature, top_p, top_k, 0.0f, 1.0f);
    if (!profile_file.empty()) gb->execute(profile_file); else gb->execute();

    compute_entropy(gb, logits, out_entropy);
    post_execute_updates(gb, tokens.size());
    update_kv_cache(gb, dec_tokens->size());
    return *static_cast<uint32_t*>(gb->get_output(sampled));
}

} // namespace engine
} // namespace cactus
