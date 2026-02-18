#include "model.h"
#include "../graph/graph.h"
#include "../npu/npu.h"
#include "../kernel/kernel.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace cactus {
namespace engine {

TrOCRModel::TrOCRModel() : Model() {}

TrOCRModel::TrOCRModel(const Config& config) : Model(config) {
    weight_nodes_.encoder_layers.resize(config.encoder_num_layers);
    weight_nodes_.decoder_layers.resize(config.decoder_num_layers);

    float hd = static_cast<float>(config.attention_head_dim);
    if (hd <= 0.0f) {
        hd = static_cast<float>(config.encoder_hidden_dim / config.encoder_attention_heads);
    }
    attention_scale_ = 1.0f / std::sqrt(hd);

    encoder_k_persistent_.assign(config.decoder_num_layers, 0);
    encoder_v_persistent_.assign(config.decoder_num_layers, 0);
}

void TrOCRModel::load_weights_to_graph(CactusGraph* gb) {
    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);

    weight_nodes_.encoder_patch_embedding_weight = gb->mmap_weights(model_folder_path_ + "/encoder_patch_embedding.weights");
    weight_nodes_.encoder_patch_embedding_bias = gb->mmap_weights(model_folder_path_ + "/encoder_patch_embedding.bias");
    weight_nodes_.encoder_position_embedding = gb->mmap_weights(model_folder_path_ + "/encoder_position_embeddings.weights");

    std::string cls_token_path = model_folder_path_ + "/encoder_cls_token.weights";
    if (std::ifstream(cls_token_path).good()) {
        weight_nodes_.encoder_cls_token = gb->mmap_weights(cls_token_path);
    }

    weight_nodes_.encoder_layernorm_weight = gb->mmap_weights(model_folder_path_ + "/encoder_layernorm.weights");
    weight_nodes_.encoder_layernorm_bias = gb->mmap_weights(model_folder_path_ + "/encoder_layernorm.bias");

    weight_nodes_.decoder_embed_tokens = gb->mmap_weights(model_folder_path_ + "/decoder_token_embeddings.weights");
    weight_nodes_.decoder_position_embedding = gb->mmap_weights(model_folder_path_ + "/decoder_position_embeddings.weights");

    std::string embed_ln_path = model_folder_path_ + "/decoder_embed_layernorm.weights";
    if (std::ifstream(embed_ln_path).good()) {
        weight_nodes_.decoder_embed_layernorm_weight = gb->mmap_weights(embed_ln_path);
        weight_nodes_.decoder_embed_layernorm_bias = gb->mmap_weights(model_folder_path_ + "/decoder_embed_layernorm.bias");
    }

    if (config_.tie_word_embeddings) {
        weight_nodes_.output_weight = weight_nodes_.decoder_embed_tokens;
        output_weight_node_id_ = weight_nodes_.decoder_embed_tokens;
    } else {
        weight_nodes_.output_weight = gb->mmap_weights(model_folder_path_ + "/output_weight.weights");
        output_weight_node_id_ = weight_nodes_.output_weight;
    }

    for (uint32_t i = 0; i < config_.encoder_num_layers; i++) {
        auto& layer = weight_nodes_.encoder_layers[i];
        std::string prefix = model_folder_path_ + "/encoder.layer_" + std::to_string(i) + "_";

        layer.encoder_self_attn_q_weight = gb->mmap_weights(prefix + "self_attn_q.weights");
        layer.encoder_self_attn_k_weight = gb->mmap_weights(prefix + "self_attn_k.weights");
        layer.encoder_self_attn_v_weight = gb->mmap_weights(prefix + "self_attn_v.weights");
        layer.encoder_self_attn_output_weight = gb->mmap_weights(prefix + "self_attn_output.weights");

        layer.encoder_self_attn_q_bias = gb->mmap_weights(prefix + "self_attn_q.bias");
        layer.encoder_self_attn_k_bias = gb->mmap_weights(prefix + "self_attn_k.bias");
        layer.encoder_self_attn_v_bias = gb->mmap_weights(prefix + "self_attn_v.bias");
        layer.encoder_self_attn_output_bias = gb->mmap_weights(prefix + "self_attn_output.bias");

        layer.encoder_layernorm1_weight = gb->mmap_weights(prefix + "layernorm1.weights");
        layer.encoder_layernorm1_bias = gb->mmap_weights(prefix + "layernorm1.bias");
        layer.encoder_layernorm2_weight = gb->mmap_weights(prefix + "layernorm2.weights");
        layer.encoder_layernorm2_bias = gb->mmap_weights(prefix + "layernorm2.bias");

        layer.encoder_mlp_fc1_weight = gb->mmap_weights(prefix + "mlp_fc1.weights");
        layer.encoder_mlp_fc1_bias = gb->mmap_weights(prefix + "mlp_fc1.bias");
        layer.encoder_mlp_fc2_weight = gb->mmap_weights(prefix + "mlp_fc2.weights");
        layer.encoder_mlp_fc2_bias = gb->mmap_weights(prefix + "mlp_fc2.bias");
    }

    for (uint32_t i = 0; i < config_.decoder_num_layers; i++) {
        auto& layer = weight_nodes_.decoder_layers[i];
        std::string prefix = model_folder_path_ + "/decoder.layer_" + std::to_string(i) + "_";

        layer.decoder_self_attn_q_weight = gb->mmap_weights(prefix + "self_attn_q.weights");
        layer.decoder_self_attn_k_weight = gb->mmap_weights(prefix + "self_attn_k.weights");
        layer.decoder_self_attn_v_weight = gb->mmap_weights(prefix + "self_attn_v.weights");
        layer.decoder_self_attn_output_weight = gb->mmap_weights(prefix + "self_attn_output.weights");

        layer.decoder_self_attn_q_bias = gb->mmap_weights(prefix + "self_attn_q.bias");
        layer.decoder_self_attn_k_bias = gb->mmap_weights(prefix + "self_attn_k.bias");
        layer.decoder_self_attn_v_bias = gb->mmap_weights(prefix + "self_attn_v.bias");
        layer.decoder_self_attn_output_bias = gb->mmap_weights(prefix + "self_attn_output.bias");

        layer.decoder_self_attn_layernorm_weight = gb->mmap_weights(prefix + "self_attn_norm.weights");
        layer.decoder_self_attn_layernorm_bias = gb->mmap_weights(prefix + "self_attn_norm.bias");

        layer.decoder_cross_attn_q_weight = gb->mmap_weights(prefix + "cross_attn_q.weights");
        layer.decoder_cross_attn_k_weight = gb->mmap_weights(prefix + "cross_attn_k.weights");
        layer.decoder_cross_attn_v_weight = gb->mmap_weights(prefix + "cross_attn_v.weights");
        layer.decoder_cross_attn_output_weight = gb->mmap_weights(prefix + "cross_attn_output.weights");

        layer.decoder_cross_attn_q_bias = gb->mmap_weights(prefix + "cross_attn_q.bias");
        layer.decoder_cross_attn_k_bias = gb->mmap_weights(prefix + "cross_attn_k.bias");
        layer.decoder_cross_attn_v_bias = gb->mmap_weights(prefix + "cross_attn_v.bias");
        layer.decoder_cross_attn_output_bias = gb->mmap_weights(prefix + "cross_attn_output.bias");

        layer.decoder_cross_attn_layernorm_weight = gb->mmap_weights(prefix + "cross_attn_norm.weights");
        layer.decoder_cross_attn_layernorm_bias = gb->mmap_weights(prefix + "cross_attn_norm.bias");

        layer.decoder_mlp_fc1_weight = gb->mmap_weights(prefix + "mlp_fc1.weights");
        layer.decoder_mlp_fc1_bias = gb->mmap_weights(prefix + "mlp_fc1.bias");
        layer.decoder_mlp_fc2_weight = gb->mmap_weights(prefix + "mlp_fc2.weights");
        layer.decoder_mlp_fc2_bias = gb->mmap_weights(prefix + "mlp_fc2.bias");

        layer.decoder_final_layernorm_weight = gb->mmap_weights(prefix + "final_norm.weights");
        layer.decoder_final_layernorm_bias = gb->mmap_weights(prefix + "final_norm.bias");
    }
}


size_t TrOCRModel::build_patch_embedding(CactusGraph* gb, size_t image_input, size_t height, size_t width) {
    size_t patch_size = config_.trocr_patch_size;
    size_t num_patches_h = height / patch_size;
    size_t num_patches_w = width / patch_size;
    size_t num_patches = num_patches_h * num_patches_w;
    size_t channels = config_.vision_num_channels > 0 ? config_.vision_num_channels : 3;
    size_t patch_dim = patch_size * patch_size * channels;

    size_t patches = gb->matmul(image_input, weight_nodes_.encoder_patch_embedding_weight, true, ComputeBackend::CPU);
    patches = gb->add(patches, weight_nodes_.encoder_patch_embedding_bias);

    size_t pos_embed = gb->slice(weight_nodes_.encoder_position_embedding, 0, 0, num_patches + 1);  // +1 for CLS token

    const auto& patches_buf = gb->get_output_buffer(patches);
    const auto& pos_buf = gb->get_output_buffer(pos_embed);

    size_t pos_node = pos_embed;
    if (pos_buf.precision != patches_buf.precision) {
        pos_node = gb->precision_cast(pos_embed, patches_buf.precision);
    }

    size_t embedded = gb->add(patches, pos_node);
    return embedded;
}

size_t TrOCRModel::build_encoder_self_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = weight_nodes_.encoder_layers[layer_idx];
    size_t q = gb->matmul(input, layer.encoder_self_attn_q_weight, true, backend);
    q = gb->add(q, layer.encoder_self_attn_q_bias);

    size_t k = gb->matmul(input, layer.encoder_self_attn_k_weight, true, backend);
    k = gb->add(k, layer.encoder_self_attn_k_bias);

    size_t v = gb->matmul(input, layer.encoder_self_attn_v_weight, true, backend);
    v = gb->add(v, layer.encoder_self_attn_v_bias);

    const auto& q_buf = gb->get_output_buffer(q);
    size_t seq_len = q_buf.shape[0];
    size_t num_heads = config_.encoder_attention_heads;
    size_t head_dim = config_.encoder_hidden_dim / num_heads;
    q = gb->reshape(q, {1, seq_len, num_heads, head_dim});
    k = gb->reshape(k, {1, seq_len, num_heads, head_dim});
    v = gb->reshape(v, {1, seq_len, num_heads, head_dim});
    size_t attn = gb->attention(q, k, v, attention_scale_, false);
    attn = gb->reshape(attn, {seq_len, num_heads * head_dim});
    size_t output = gb->matmul(attn, layer.encoder_self_attn_output_weight, true, backend);
    output = gb->add(output, layer.encoder_self_attn_output_bias);

    return output;
}

size_t TrOCRModel::build_encoder_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = weight_nodes_.encoder_layers[layer_idx];
    size_t fc1 = gb->matmul(input, layer.encoder_mlp_fc1_weight, true, backend);
    fc1 = gb->add(fc1, layer.encoder_mlp_fc1_bias);
    size_t act = gb->gelu_erf(fc1);
    size_t fc2 = gb->matmul(act, layer.encoder_mlp_fc2_weight, true, backend);
    fc2 = gb->add(fc2, layer.encoder_mlp_fc2_bias);

    return fc2;
}

size_t TrOCRModel::build_encoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = weight_nodes_.encoder_layers[layer_idx];
    size_t ln1 = gb->layernorm(hidden, layer.encoder_layernorm1_weight, layer.encoder_layernorm1_bias);
    size_t attn = build_encoder_self_attention(gb, ln1, layer_idx, backend);
    size_t x = gb->add(hidden, attn);
    size_t ln2 = gb->layernorm(x, layer.encoder_layernorm2_weight, layer.encoder_layernorm2_bias);
    size_t mlp = build_encoder_mlp(gb, ln2, layer_idx, backend);
    size_t output = gb->add(x, mlp);

    return output;
}

void TrOCRModel::run_encoder(const std::vector<float>& image_pixels, size_t height, size_t width) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    if (!gb) {
        throw std::runtime_error("Graph handle is null in run_encoder");
    }

    auto backend = (config_.default_backend == Config::Backend::CPU) ? ComputeBackend::CPU : ComputeBackend::NPU;

    size_t channels = config_.vision_num_channels > 0 ? config_.vision_num_channels : 3;
    size_t patch_size = config_.trocr_patch_size;
    size_t num_patches_h = height / patch_size;
    size_t num_patches_w = width / patch_size;
    size_t num_patches = num_patches_h * num_patches_w;
    size_t patch_dim = patch_size * patch_size * channels;

    std::vector<__fp16> image_f16(image_pixels.size());
    cactus_fp32_to_fp16(image_pixels.data(), image_f16.data(), image_pixels.size());

    size_t image_input = gb->input({num_patches, patch_dim}, Precision::FP16);
    gb->set_input(image_input, image_f16.data(), Precision::FP16);

    size_t hidden = build_patch_embedding(gb, image_input, height, width);

    for (uint32_t i = 0; i < config_.encoder_num_layers; ++i) {
        hidden = build_encoder_transformer_block(gb, hidden, i, backend);
    }

    size_t encoder_output = gb->layernorm(hidden, weight_nodes_.encoder_layernorm_weight, weight_nodes_.encoder_layernorm_bias);

    weight_nodes_.encoder_output = encoder_output;
    last_encoder_post_norm_node_ = encoder_output;
}

// ============ Text Decoder Methods ============

size_t TrOCRModel::build_decoder_self_attention(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                                 ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.decoder_layers[layer_idx];

    size_t q = gb->matmul(input, layer.decoder_self_attn_q_weight, true, backend);
    q = gb->add(q, layer.decoder_self_attn_q_bias);

    size_t k = gb->matmul(input, layer.decoder_self_attn_k_weight, true, backend);
    k = gb->add(k, layer.decoder_self_attn_k_bias);

    size_t v = gb->matmul(input, layer.decoder_self_attn_v_weight, true, backend);
    v = gb->add(v, layer.decoder_self_attn_v_bias);

    const auto& q_buf = gb->get_output_buffer(q);
    size_t seq_new = q_buf.shape[0];
    size_t num_heads = config_.decoder_attention_heads;
    size_t head_dim = config_.decoder_hidden_dim / num_heads;

    size_t q_4d = gb->reshape(q, {1, seq_new, num_heads, head_dim});
    size_t k_4d = gb->reshape(k, {1, seq_new, num_heads, head_dim});
    size_t v_4d = gb->reshape(v, {1, seq_new, num_heads, head_dim});

    size_t final_k = k_4d;
    size_t final_v = v_4d;

    if (use_cache && !kv_cache_.is_empty()) {
        auto k_view = kv_cache_.get_key_view(layer_idx);
        auto v_view = kv_cache_.get_value_view(layer_idx);

        if (!k_view.ptr1 || !v_view.ptr1) {
            throw std::runtime_error("KV cache view is empty but kv_cache_.is_empty()==false");
        }

        size_t cache_len = kv_cache_.current_seq_len;
        size_t cache_k_node = gb->input({1, cache_len, num_heads, head_dim}, kv_cache_.precision);
        size_t cache_v_node = gb->input({1, cache_len, num_heads, head_dim}, kv_cache_.precision);

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

    size_t attn = gb->attention(q_4d, final_k, final_v, attention_scale_, position_offset);
    attn = gb->reshape(attn, {seq_new, num_heads * head_dim});

    size_t output = gb->matmul(attn, layer.decoder_self_attn_output_weight, true, backend);
    output = gb->add(output, layer.decoder_self_attn_output_bias);

    return output;
}

size_t TrOCRModel::build_decoder_cross_attention(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                                 ComputeBackend backend, bool /*use_cache*/) {
    const auto& layer = weight_nodes_.decoder_layers[layer_idx];

    size_t q = gb->matmul(input, layer.decoder_cross_attn_q_weight, true, backend);
    q = gb->add(q, layer.decoder_cross_attn_q_bias);

    const auto& q_buf = gb->get_output_buffer(q);
    if (q_buf.shape.size() != 2) {
        throw std::runtime_error("encoder cross-attn: q must be [T_dec, D]");
    }

    size_t seq_dec = q_buf.shape[0];
    size_t num_heads = config_.decoder_attention_heads;
    size_t head_dim = config_.decoder_hidden_dim / num_heads;

    q = gb->reshape(q, {1, seq_dec, num_heads, head_dim});

    size_t k_4d = 0;
    size_t v_4d = 0;

    bool is_populated = (encoder_k_persistent_[layer_idx] != 0 && gb->is_populated(encoder_k_persistent_[layer_idx]));
    if (is_populated) {
        k_4d = encoder_k_persistent_[layer_idx];
        v_4d = encoder_v_persistent_[layer_idx];
    } else {
        size_t enc_norm = last_encoder_post_norm_node_;

        size_t k = gb->matmul(enc_norm, layer.decoder_cross_attn_k_weight, true, backend);
        k = gb->add(k, layer.decoder_cross_attn_k_bias);

        size_t v = gb->matmul(enc_norm, layer.decoder_cross_attn_v_weight, true, backend);
        v = gb->add(v, layer.decoder_cross_attn_v_bias);

        const auto& k_buf = gb->get_output_buffer(k);
        if (k_buf.shape.size() != 2) {
            throw std::runtime_error("encoder cross-attn: k must be [T_enc, D]");
        }
        size_t seq_enc = k_buf.shape[0];

        k_4d = gb->reshape(k, {1, seq_enc, num_heads, head_dim});
        v_4d = gb->reshape(v, {1, seq_enc, num_heads, head_dim});

        if (encoder_k_persistent_[layer_idx] == 0) {
            encoder_k_persistent_[layer_idx] = gb->persistent(k_4d);
            encoder_v_persistent_[layer_idx] = gb->persistent(v_4d);
        }
        k_4d = encoder_k_persistent_[layer_idx];
        v_4d = encoder_v_persistent_[layer_idx];
    }

    size_t attn = gb->attention(q, k_4d, v_4d, attention_scale_, false);
    attn = gb->reshape(attn, {seq_dec, num_heads * head_dim});

    size_t output = gb->matmul(attn, layer.decoder_cross_attn_output_weight, true, backend);
    output = gb->add(output, layer.decoder_cross_attn_output_bias);

    return output;
}

size_t TrOCRModel::build_decoder_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend) const {
    const auto& layer = weight_nodes_.decoder_layers[layer_idx];

    // FC1
    size_t fc1 = gb->matmul(input, layer.decoder_mlp_fc1_weight, true, backend);
    fc1 = gb->add(fc1, layer.decoder_mlp_fc1_bias);

    // GELU activation
    size_t act = gb->gelu_erf(fc1);

    // FC2
    size_t fc2 = gb->matmul(act, layer.decoder_mlp_fc2_weight, true, backend);
    fc2 = gb->add(fc2, layer.decoder_mlp_fc2_bias);

    return fc2;
}

size_t TrOCRModel::build_decoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                                    ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.decoder_layers[layer_idx];

    // Self-attention block
    size_t ln1 = gb->layernorm(hidden, layer.decoder_self_attn_layernorm_weight, layer.decoder_self_attn_layernorm_bias);
    size_t sa = build_decoder_self_attention(gb, ln1, layer_idx, backend, use_cache, position_offset);
    size_t x = gb->add(hidden, sa);

    // Cross-attention block
    size_t ln2 = gb->layernorm(x, layer.decoder_cross_attn_layernorm_weight, layer.decoder_cross_attn_layernorm_bias);
    size_t ca = build_decoder_cross_attention(gb, ln2, layer_idx, backend, use_cache);
    x = gb->add(x, ca);

    // MLP block
    size_t ln3 = gb->layernorm(x, layer.decoder_final_layernorm_weight, layer.decoder_final_layernorm_bias);
    size_t mlp = build_decoder_mlp(gb, ln3, layer_idx, backend);
    x = gb->add(x, mlp);

    return x;
}

size_t TrOCRModel::run_decoder_step(const std::vector<uint32_t>& tokens, bool use_cache, bool last_token_only) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);

    const size_t full_len = tokens.size();
    if (full_len == 0) {
        throw std::runtime_error("Decoder token list cannot be empty");
    }

    auto backend = (config_.default_backend == Config::Backend::CPU) ? ComputeBackend::CPU : ComputeBackend::NPU;

    size_t start_idx = (use_cache && kv_cache_.current_seq_len > 0) ? full_len - 1 : 0;
    size_t new_tokens = full_len - start_idx;

    // Token input
    size_t tok_input = gb->input({new_tokens}, Precision::FP32);
    std::vector<float> tok_f(new_tokens);
    for (size_t i = 0; i < new_tokens; i++) {
        tok_f[i] = static_cast<float>(tokens[start_idx + i]);
    }
    gb->set_input(tok_input, tok_f.data(), Precision::FP32);

    // Token embedding
    size_t hidden = gb->embedding(weight_nodes_.decoder_embed_tokens, tok_input);

    // Position embedding
    size_t position_offset = kv_cache_.current_seq_len;
    size_t pos = gb->slice(weight_nodes_.decoder_position_embedding, 0, position_offset, new_tokens);

    const auto& h_buf = gb->get_output_buffer(hidden);
    const auto& pos_buf = gb->get_output_buffer(pos);

    size_t pos_node = pos;
    if (pos_buf.precision != h_buf.precision) {
        pos_node = gb->precision_cast(pos, h_buf.precision);
    }

    hidden = gb->add(hidden, pos_node);

    // Decoder transformer layers
    for (uint32_t layer_idx = 0; layer_idx < config_.decoder_num_layers; ++layer_idx) {
        hidden = build_decoder_transformer_block(gb, hidden, layer_idx, backend, use_cache, position_offset);
    }

    // Get last token logits if needed
    size_t logits_input = hidden;
    if (last_token_only) {
        size_t row_index = new_tokens - 1;
        logits_input = gb->slice(logits_input, 0, row_index, 1);
    }

    // Output projection
    size_t logits = gb->matmul(logits_input, output_weight_node_id_, true, backend);

    last_new_tokens_ = new_tokens;
    return logits;
}

void TrOCRModel::reset_graph_side_cache_nodes() {
    cache_k_output_nodes_.assign(config_.decoder_num_layers, 0);
    cache_v_output_nodes_.assign(config_.decoder_num_layers, 0);
}

void TrOCRModel::reset_cache() {
    Model::reset_cache();
    encoder_ready_ = false;
    first_decode_step_ = true;

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    if (gb) {
        if (encoder_output_persistent_ != 0) {
            gb->invalidate_persistent(encoder_output_persistent_);
            encoder_output_persistent_ = 0;
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
}

uint32_t TrOCRModel::decode_with_image(
    const std::vector<uint32_t>& tokens,
    const std::vector<float>& image_pixels,
    size_t image_height,
    size_t image_width,
    float temperature,
    float top_p,
    size_t top_k,
    const std::string& profile_file,
    float* out_entropy) {

    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }
    if (tokens.empty()) {
        throw std::runtime_error("Token sequence cannot be empty");
    }
    if (image_pixels.empty()) {
        throw std::runtime_error("Image pixels cannot be empty");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);

    bool cold_start = !encoder_ready_;
    size_t logits_node = 0;

    // Prepend BOS token
    uint32_t bos = static_cast<uint32_t>(get_tokenizer()->get_bos_token());
    std::vector<uint32_t> full_tokens;
    full_tokens.reserve(tokens.size() + 1);
    full_tokens.push_back(bos);
    full_tokens.insert(full_tokens.end(), tokens.begin(), tokens.end());

    if (cold_start) {
        gb->soft_reset();
        kv_cache_.reset();
        kv_cache_.current_seq_len = 0;
        reset_graph_side_cache_nodes();

        first_decode_step_ = true;

        run_encoder(image_pixels, image_height, image_width);

        logits_node = run_decoder_step(full_tokens, false, false);
    } else {
        gb->soft_reset();
        reset_graph_side_cache_nodes();

        std::vector<uint32_t> last_token_vec = {tokens.back()};
        logits_node = run_decoder_step(last_token_vec, true, true);
    }

    // Sample from logits
    size_t sampled_token_id = gb->sample(logits_node, temperature, top_p, top_k);

    if (!profile_file.empty()) {
        gb->execute(profile_file);
    } else {
        gb->execute();
    }

    if (cold_start) {
        if (encoder_output_persistent_ == 0) {
            encoder_output_persistent_ = gb->persistent(last_encoder_post_norm_node_);
        }
        last_encoder_post_norm_node_ = encoder_output_persistent_;
        encoder_ready_ = true;
    }

    // Compute entropy if requested
    if (out_entropy) {
        const auto& logits_buf = gb->get_output_buffer(logits_node);
        void* logits_ptr = gb->get_output(logits_node);
        size_t vocab_size = logits_buf.shape.back();

        std::vector<float> logits(vocab_size);
        if (logits_buf.precision == Precision::FP32) {
            float* src = static_cast<float*>(logits_ptr);
            std::copy(src, src + vocab_size, logits.begin());
        } else if (logits_buf.precision == Precision::FP16) {
            __fp16* src = static_cast<__fp16*>(logits_ptr);
            Quantization::fp16_to_fp32(src, logits.data(), vocab_size);
        } else {
            int8_t* src = static_cast<int8_t*>(logits_ptr);
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

    // Update KV cache
    post_execute_updates(gb, full_tokens.size());
    update_kv_cache(gb, last_new_tokens_);

    // Get sampled token
    auto* out_ptr = gb->get_output(sampled_token_id);
    uint32_t sampled = *reinterpret_cast<uint32_t*>(out_ptr);

    return sampled;
}

std::vector<float> TrOCRModel::get_image_embeddings(const std::string& image_path) {
    int width, height, channels;
    unsigned char* img_data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (!img_data) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    size_t target_h = config_.trocr_image_size;
    size_t target_w = config_.trocr_image_size;
    size_t patch_size = config_.trocr_patch_size;
    size_t num_patches_h = target_h / patch_size;
    size_t num_patches_w = target_w / patch_size;
    size_t num_patches = num_patches_h * num_patches_w;
    size_t patch_dim = patch_size * patch_size * 3;
    std::vector<float> resized(target_h * target_w * 3);
    stbir_resize_uint8_linear(img_data, width, height, 0,
                              reinterpret_cast<unsigned char*>(resized.data()),
                              static_cast<int>(target_w), static_cast<int>(target_h), 0,
                              STBIR_RGB);

    stbi_image_free(img_data);
    for (auto& v : resized) {
        v = (v / 255.0f - 0.5f) / 0.5f;
    }
    std::vector<float> patches(num_patches * patch_dim);
    for (size_t ph = 0; ph < num_patches_h; ++ph) {
        for (size_t pw = 0; pw < num_patches_w; ++pw) {
            size_t patch_idx = ph * num_patches_w + pw;
            for (size_t py = 0; py < patch_size; ++py) {
                for (size_t px = 0; px < patch_size; ++px) {
                    size_t img_y = ph * patch_size + py;
                    size_t img_x = pw * patch_size + px;
                    for (size_t c = 0; c < 3; ++c) {
                        size_t patch_offset = patch_idx * patch_dim + (py * patch_size + px) * 3 + c;
                        size_t img_offset = (img_y * target_w + img_x) * 3 + c;
                        patches[patch_offset] = resized[img_offset];
                    }
                }
            }
        }
    }
    run_encoder(patches, target_h, target_w);

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    size_t pooled = gb->mean(weight_nodes_.encoder_output, 0);
    gb->execute();

    const auto& output_buf = gb->get_output_buffer(pooled);
    size_t hidden_dim = output_buf.total_size;

    std::vector<float> embedding(hidden_dim);
    void* output_data = gb->get_output(pooled);
    const float* output_ptr = static_cast<const float*>(output_data);
    std::copy(output_ptr, output_ptr + hidden_dim, embedding.begin());

    reset_cache();
    return embedding;
}

} // namespace engine
} // namespace cactus
