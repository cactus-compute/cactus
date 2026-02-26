#include "model.h"
#include "../graph/graph.h"
#include "../npu/npu.h"
#include "../kernel/kernel.h"
#include <cmath>
#include <stdexcept>
#include <set>
#include <iostream>
#include <algorithm>
#include <cstdlib>
namespace cactus {
namespace engine {

static size_t moonshine_downsampled_len(size_t audio_len) {
    if (audio_len < 127) {
        return 0;
    }
    size_t l1 = (audio_len - 127) / 64 + 1;
    if (l1 < 7) {
        return 0;
    }
    size_t l2 = (l1 - 7) / 3 + 1;
    if (l2 < 3) {
        return 0;
    }
    size_t l3 = (l2 - 1) / 2 + 1;
    return l3;
}

MoonshineModel::MoonshineModel() : Model() {}
MoonshineModel::MoonshineModel(const Config& config) : Model(config) {
    weight_nodes_.encoder_layers.resize(config.num_encoder_layers);
    weight_nodes_.decoder_layers.resize(config.num_decoder_layers);
    float hd = static_cast<float>(config.attention_head_dim);
    if (hd <= 0.0f) {
        hd = 64.0f;
    }
    attention_scale_ = 1.0f / std::sqrt(hd);
    encoder_block_out_nodes_.resize(config.num_encoder_layers, 0);
    encoder_k_persistent_.assign(config.num_decoder_layers, 0);
    encoder_v_persistent_.assign(config.num_decoder_layers, 0);
}
void MoonshineModel::load_weights_to_graph(CactusGraph* gb) {
    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    weight_nodes_.decoder_norm_weight = gb->mmap_weights(model_folder_path_ + "/output_norm.weights");
    output_weight_node_id_ = gb->mmap_weights(model_folder_path_ + "/output_weight.weights");
    if (output_weight_node_id_ == 0) {
        output_weight_node_id_ = gb->mmap_weights(model_folder_path_ + "/output_layer.weights");
    }
    weight_nodes_.output_weight = output_weight_node_id_;
    if (npu::is_npu_available()) {
        std::string npu_encoder_path = model_folder_path_ + "/model.mlpackage";
        npu_encoder_ = npu::create_encoder();
        if (npu_encoder_ && npu_encoder_->load(npu_encoder_path)) {
            use_npu_encoder_ = true;
            std::vector<int> input_shape = npu_encoder_->get_input_shape();
            if (!input_shape.empty()) {
                npu_encoder_->preallocate(input_shape, "x", "");
            }
        }
        else {
            use_npu_encoder_ = false;
            npu_encoder_.reset();
        }
    }
    if (!use_npu_encoder_) {
        weight_nodes_.encoder_conv1_weight = gb->mmap_weights(model_folder_path_ + "/encoder_conv1_weight.weights");
        weight_nodes_.encoder_conv2_weight = gb->mmap_weights(model_folder_path_ + "/encoder_conv2_weight.weights");
        weight_nodes_.encoder_conv2_bias = gb->mmap_weights(model_folder_path_ + "/encoder_conv2_bias.bias");
        weight_nodes_.encoder_conv3_weight = gb->mmap_weights(model_folder_path_ + "/encoder_conv3_weight.weights");
        weight_nodes_.encoder_conv3_bias = gb->mmap_weights(model_folder_path_ + "/encoder_conv3_bias.bias");
        weight_nodes_.encoder_norm_weight = gb->mmap_weights(model_folder_path_ + "/encoder_norm_weight.weights"); 
        weight_nodes_.encoder_norm_bias = gb->mmap_weights(model_folder_path_ + "/encoder_norm_bias.bias");
        weight_nodes_.encoder_layer_norm_weight = gb->mmap_weights(model_folder_path_ + "/encoder_layer_norm_weight.weights"); 
    }
    for (uint32_t i = 0; i < config_.num_decoder_layers; i++) {
        auto& layer = weight_nodes_.decoder_layers[i];
        std::string layer_prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";
        layer.decoder_encoder_attn_k_weight = gb->mmap_weights(layer_prefix + "encoder_attn_k.weights");
        layer.decoder_encoder_attn_q_weight = gb->mmap_weights(layer_prefix + "encoder_attn_q.weights");
        layer.decoder_encoder_attn_v_weight = gb->mmap_weights(layer_prefix + "encoder_attn_v.weights");
        layer.decoder_encoder_attn_output_weight = gb->mmap_weights(layer_prefix + "encoder_attn_output.weights");
        layer.decoder_post_encoder_layernorm_weight = gb->mmap_weights(layer_prefix + "post_attn_norm.weights");
        if (config_.decoder_act_gelu) {
            layer.decoder_ffn1_weight = gb->mmap_weights(layer_prefix + "mlp_fc1.weights");
            layer.decoder_ffn1_bias = gb->mmap_weights(layer_prefix + "mlp_fc1.bias");
        }
        else {
            layer.decoder_ffn_gate_weight = gb->mmap_weights(layer_prefix + "ffn_gate.weights");
            layer.decoder_ffn_gate_bias = gb->mmap_weights(layer_prefix + "ffn_gate.bias");
            layer.decoder_ffn_up_weight = gb->mmap_weights(layer_prefix + "ffn_up.weights");
            layer.decoder_ffn_up_bias = gb->mmap_weights(layer_prefix + "ffn_up.bias");
        }
        layer.decoder_ffn2_weight = gb->mmap_weights(layer_prefix + "mlp_fc2.weights");
        layer.decoder_ffn2_bias = gb->mmap_weights(layer_prefix + "mlp_fc2.bias");
        layer.decoder_post_ffn_layernorm_weight = gb->mmap_weights(layer_prefix + "final_norm.weights");
        layer.decoder_self_attn_k_weight = gb->mmap_weights(layer_prefix + "attn_k.weights");
        layer.decoder_self_attn_q_weight = gb->mmap_weights(layer_prefix + "attn_q.weights");
        layer.decoder_self_attn_v_weight = gb->mmap_weights(layer_prefix + "attn_v.weights");
        layer.decoder_self_attn_output_weight = gb->mmap_weights(layer_prefix + "attn_output.weights");
        layer.decoder_post_attn_layernorm_weight = gb->mmap_weights(layer_prefix + "input_norm.weights");
    }
    if (!use_npu_encoder_) {
        for (uint32_t i = 0; i < config_.num_encoder_layers; i++) {
            auto& layer = weight_nodes_.encoder_layers[i];
            std::string layer_prefix = model_folder_path_ + "/encoder_layer_" + std::to_string(i) + "_";
            if (config_.encoder_act_gelu) {
                layer.encoder_ffn1_weight = gb->mmap_weights(layer_prefix + "mlp_fc1.weights");
                layer.encoder_ffn1_bias = gb->mmap_weights(layer_prefix + "mlp_fc1.bias");
            }
            else {
                layer.encoder_ffn_gate_weight = gb->mmap_weights(layer_prefix + "ffn_gate.weights");
                layer.encoder_ffn_gate_bias = gb->mmap_weights(layer_prefix + "ffn_gate.bias");
                layer.encoder_ffn_up_weight = gb->mmap_weights(layer_prefix + "ffn_up.weights");
                layer.encoder_ffn_up_bias = gb->mmap_weights(layer_prefix + "ffn_up.bias");
            }
            layer.encoder_ffn2_weight = gb->mmap_weights(layer_prefix + "mlp_fc2.weights");
            layer.encoder_ffn2_bias = gb->mmap_weights(layer_prefix + "mlp_fc2.bias");
            layer.encoder_post_ffn_layernorm_weight = gb->mmap_weights(layer_prefix + "post_attn_norm.weights");
            layer.encoder_self_attn_k_weight = gb->mmap_weights(layer_prefix + "attn_k.weights");
            layer.encoder_self_attn_q_weight = gb->mmap_weights(layer_prefix + "attn_q.weights");
            layer.encoder_self_attn_v_weight = gb->mmap_weights(layer_prefix + "attn_v.weights");
            layer.encoder_self_attn_output_weight = gb->mmap_weights(layer_prefix + "attn_output.weights");
            layer.encoder_post_attn_layernorm_weight = gb->mmap_weights(layer_prefix + "input_norm.weights");
        }
    }
}
static size_t build_encoder_mlp_gelu(CactusGraph* gb, size_t input, size_t w1, size_t b1, size_t w2, size_t b2, ComputeBackend backend, uint32_t /*layer_idx*/) {
    auto ffn1_weight = gb->matmul(input, w1, true, backend);
    auto ffn1_bias = gb->add(ffn1_weight, b1);
    auto ffn1_act = gb->gelu(ffn1_bias);
    auto ffn2_weight = gb->matmul(ffn1_act, w2, true, backend);
    auto ffn2_bias = gb->add(ffn2_weight, b2);
    return ffn2_bias;
}
static size_t build_decoder_mlp_silu(CactusGraph* gb, size_t input, size_t w_gate, size_t b_gate, size_t w_up, size_t b_up, size_t w2, size_t b2, ComputeBackend backend, uint32_t /*layer_idx*/) {
    auto gate_weight = gb->matmul(input, w_gate, true, backend);
    auto gate_bias = gb->add(gate_weight, b_gate);
    auto gate_act = gb->silu(gate_bias);
    auto up_weight = gb->matmul(input, w_up, true, backend);
    auto up_bias = gb->add(up_weight, b_up);
    auto intermediate = gb->multiply(gate_act, up_bias);
    auto output_weight = gb->matmul(intermediate, w2, true, backend);
    auto output_bias = gb->add(output_weight, b2);
    return output_bias;
}
size_t MoonshineModel::build_decoder_cross_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend, bool /*use_cache*/, size_t /*position_offset*/){
    const auto& layer = weight_nodes_.decoder_layers[layer_idx];
    size_t q = gb->matmul(input, layer.decoder_encoder_attn_q_weight, true, backend);
    const auto& q_buf   = gb->get_output_buffer(q);
    if (q_buf.shape.size() != 2) {
        throw std::runtime_error("encoder cross-attn: q must be [T_dec, D]");
    }
    size_t T_dec   = q_buf.shape[0];
    size_t q_heads = config_.attention_heads;
    size_t kv_heads = config_.attention_kv_heads;
    size_t head_dim = config_.attention_head_dim;
    q = gb->reshape(q, {1, T_dec, q_heads, head_dim});
    size_t k_4d = 0;
    size_t v_4d = 0;
    bool is_pop = (encoder_k_persistent_[layer_idx] != 0 && gb->is_populated(encoder_k_persistent_[layer_idx]));
    std::string prefix = "model.decoder.layers." + std::to_string(layer_idx) + ".encoder_attn.";
    if (is_pop) {
        k_4d = encoder_k_persistent_[layer_idx];
        v_4d = encoder_v_persistent_[layer_idx];
    }
   	else {
        size_t enc_norm = last_encoder_post_norm_node_;
        size_t k = gb->matmul(enc_norm, layer.decoder_encoder_attn_k_weight, true, backend);
        size_t v = gb->matmul(enc_norm, layer.decoder_encoder_attn_v_weight, true, backend);
        const auto& k_buf = gb->get_output_buffer(k);
        if (k_buf.shape.size() != 2) {
            throw std::runtime_error("encoder cross-attn: k must be [T_enc, D]");
        }
        size_t T_enc = k_buf.shape[0];
        k_4d = gb->reshape(k, {1, T_enc, kv_heads, head_dim});
        v_4d = gb->reshape(v, {1, T_enc, kv_heads, head_dim});
        if (encoder_k_persistent_[layer_idx] == 0) {
            encoder_k_persistent_[layer_idx] = gb->persistent(k_4d);
            encoder_v_persistent_[layer_idx] = gb->persistent(v_4d);
        }
        k_4d = encoder_k_persistent_[layer_idx];
        v_4d = encoder_v_persistent_[layer_idx];
    }
    size_t attn = gb->attention(q, k_4d, v_4d, attention_scale_, false);
    attn = gb->reshape(attn, {T_dec, q_heads * head_dim});
    size_t output = gb->matmul(attn, layer.decoder_encoder_attn_output_weight, true, backend);
    return output;
}
void MoonshineModel::reset_graph_side_cache_nodes() {
    cache_k_output_nodes_.assign(config_.num_decoder_layers, 0);
    cache_v_output_nodes_.assign(config_.num_decoder_layers, 0);
}
void MoonshineModel::reset_cache() {
    Model::reset_cache();
    encoder_ready_ = false;
    encoder_kv_ready_ = false;
    first_decode_step_ = true;
    encoder_output_host_.clear();
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
}
size_t MoonshineModel::build_decoder_self_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t position_offset){
    const auto& layer = weight_nodes_.decoder_layers[layer_idx];
    size_t q_proj = gb->matmul(input, layer.decoder_self_attn_q_weight, true, backend);
    size_t k_proj = gb->matmul(input, layer.decoder_self_attn_k_weight, true, backend);
    size_t v_proj = gb->matmul(input, layer.decoder_self_attn_v_weight, true, backend);
    std::string prefix = "model.decoder.layers." + std::to_string(layer_idx) + ".self_attn.";
    const auto& q_shape = gb->get_output_buffer(q_proj).shape;
    if (q_shape.size() != 2) {
        throw std::runtime_error("decoder self-attn: q must be [T_new, D]");
    }
    size_t seq_new = q_shape[0];
    size_t num_heads = config_.attention_heads;
    size_t head_dim = config_.attention_head_dim;
    size_t num_kv_heads = config_.attention_kv_heads;
    size_t q_4d = gb->reshape(q_proj, {1, seq_new, num_heads, head_dim});
    size_t k_4d = gb->reshape(k_proj, {1, seq_new, num_kv_heads, head_dim});
    size_t v_4d = gb->reshape(v_proj, {1, seq_new, num_kv_heads, head_dim});
    size_t rot_dim = std::max(head_dim / 2, (size_t)32);
    if (config_.rope_theta > 0) {
        q_4d = gb->rope_gptj(q_4d, config_.rope_theta, position_offset, rot_dim);
        k_4d = gb->rope_gptj(k_4d, config_.rope_theta, position_offset, rot_dim);
    }
    size_t final_k = k_4d;
    size_t final_v = v_4d;
    if (use_cache && !kv_cache_.is_empty()) {
        auto k_view = kv_cache_.get_key_view(layer_idx);
        auto v_view = kv_cache_.get_value_view(layer_idx);
        if (!k_view.ptr1 || !v_view.ptr1) {
            throw std::runtime_error("KV cache view is empty but kv_cache_.is_empty()==false");
        }
        size_t cache_len = kv_cache_.current_seq_len;
        size_t cache_k_node = gb->input(
            {1, cache_len, num_kv_heads, head_dim}
,
            kv_cache_.precision
        );
        size_t cache_v_node = gb->input(
            {1, cache_len, num_kv_heads, head_dim}
,
            kv_cache_.precision
        );
        if (k_view.ptr2 == nullptr && v_view.ptr2 == nullptr) {
            gb->set_external_input(cache_k_node, const_cast<void*>(k_view.ptr1), kv_cache_.precision);
            gb->set_external_input(cache_v_node, const_cast<void*>(v_view.ptr1), kv_cache_.precision);
        }
        else {
            gb->set_external_input(cache_k_node, kv_cache_.get_key_ptr(layer_idx), kv_cache_.precision);
            gb->set_external_input(cache_v_node, kv_cache_.get_value_ptr(layer_idx), kv_cache_.precision);
        }
        final_k = gb->concat(cache_k_node, k_4d, 1);
        final_v = gb->concat(cache_v_node, v_4d, 1);
    }
    if (use_cache) {
        cache_k_output_nodes_[layer_idx] = final_k;
        cache_v_output_nodes_[layer_idx] = final_v;
    }
    else {
        cache_k_output_nodes_[layer_idx] = k_4d;
        cache_v_output_nodes_[layer_idx] = v_4d;
    }
    auto attn_out_4d = gb->attention(q_4d, final_k, final_v, attention_scale_, position_offset);
    auto attn_out    = gb->reshape(attn_out_4d, {seq_new, num_heads * head_dim});
    auto output = gb->matmul(attn_out, layer.decoder_self_attn_output_weight, true, backend);
    return output;
}
size_t MoonshineModel::build_encoder_self_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t /*position_offset*/){
    const auto& layer = weight_nodes_.encoder_layers[layer_idx];
    if(use_cache)
        throw std::runtime_error("The encoder attention layers are not auto-regressive, and thus don't use KV caching!");
    auto q_proj = gb->matmul(input, layer.encoder_self_attn_q_weight, true, backend);
    auto v_proj = gb->matmul(input, layer.encoder_self_attn_v_weight, true, backend);
    auto k_proj = gb->matmul(input, layer.encoder_self_attn_k_weight, true, backend);
    std::string prefix = "model.encoder.layers." + std::to_string(layer_idx) + ".self_attn.";
    size_t seq_len = gb->get_output_buffer(q_proj).shape[0];
    size_t num_heads = config_.attention_heads;
    size_t head_dim  = config_.attention_head_dim;
    auto q = gb->reshape(q_proj, {1, seq_len, num_heads, head_dim});
    auto k = gb->reshape(k_proj, {1, seq_len, num_heads, head_dim});
    auto v = gb->reshape(v_proj, {1, seq_len, num_heads, head_dim});
    size_t rot_dim = static_cast<size_t>(head_dim * config_.partial_rotary_factor); 
    if (rot_dim % 2 != 0) throw std::runtime_error("rot_dim must be even");
    if (config_.rope_theta > 0) {
        q = gb->rope_gptj(q, config_.rope_theta, 0, rot_dim);
        k = gb->rope_gptj(k, config_.rope_theta, 0, rot_dim);
    }
    auto attn = gb->attention(q, k, v, attention_scale_, false);
    attn = gb->reshape(attn, {seq_len, num_heads * head_dim});
    auto output = gb->matmul(attn, layer.encoder_self_attn_output_weight, true, backend);
    return output;
}
size_t MoonshineModel::build_audio_preprocessor(CactusGraph* gb, size_t input)
{
    size_t conv_input = input;
    const auto& xbuf = gb->get_output_buffer(input);
    if (xbuf.precision == Precision::INT8) { 
        conv_input = gb->precision_cast(input, Precision::FP16);
    }
    size_t conv1 = gb->conv1d(conv_input, weight_nodes_.encoder_conv1_weight, 64);
    last_conv1_node_ = conv1;
    size_t conv1_act = gb->tanh(conv1); 
    size_t group_norm = gb->groupnorm(conv1_act, weight_nodes_.encoder_norm_weight, weight_nodes_.encoder_norm_bias, 1);

    size_t conv2_with_bias = gb->conv1d_k7s3(group_norm, weight_nodes_.encoder_conv2_weight, weight_nodes_.encoder_conv2_bias);
    size_t conv2_act = gb->gelu(conv2_with_bias);
    size_t conv3 = gb->conv1d_k3(conv2_act, weight_nodes_.encoder_conv3_weight, 2);
    auto bias3_shape = gb->get_output_buffer(weight_nodes_.encoder_conv3_bias).shape;
    size_t C3 = bias3_shape[0];
    size_t bias3 = gb->reshape(weight_nodes_.encoder_conv3_bias, {1, C3, 1});
    size_t conv3_with_bias = gb->add(conv3, bias3);
    size_t conv3_act = gb->gelu(conv3_with_bias);
    const auto& buf = gb->get_output_buffer(conv3_act);
    size_t conv_out_transposed;
    if (buf.precision == Precision::FP16) {
        conv_out_transposed = gb->transpose(conv3_act, ComputeBackend::CPU);
    }
    else {
        size_t conv3_f16 = gb->precision_cast(conv3_act, Precision::FP16);
        conv_out_transposed = gb->transpose(conv3_f16, ComputeBackend::CPU);
    }
    return conv_out_transposed;
}
size_t MoonshineModel::build_encoder_transformer_block(
    CactusGraph* gb,
    size_t hidden,
    uint32_t layer_idx,
    ComputeBackend backend,
    bool use_cache,
    size_t position_offset)
{
    const auto& layer = weight_nodes_.encoder_layers[layer_idx];
    size_t input_layernorm = gb->layernorm(
        hidden,
        layer.encoder_post_attn_layernorm_weight
    );
    size_t self_attn_out = build_encoder_self_attention(
        gb, input_layernorm, layer_idx, backend, use_cache, position_offset
    );
    size_t post_attention_layernorm_input = gb->add(hidden, self_attn_out);
    size_t post_attention_layernorm = gb->layernorm(
        post_attention_layernorm_input,
        layer.encoder_post_ffn_layernorm_weight
    );
    std::string prefix = "model.encoder.layers." + std::to_string(layer_idx) + ".";
    size_t mlp_out = build_encoder_mlp_gelu(
        gb, post_attention_layernorm, layer.encoder_ffn1_weight, layer.encoder_ffn1_bias,
        layer.encoder_ffn2_weight, layer.encoder_ffn2_bias, backend, layer_idx
    );
    size_t out = gb->add(post_attention_layernorm_input, mlp_out);
    if (layer_idx < encoder_block_out_nodes_.size()) {
        encoder_block_out_nodes_[layer_idx] = out;
    }
    return out;
}
size_t MoonshineModel::build_decoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t position_offset){
    const auto& layer = weight_nodes_.decoder_layers[layer_idx];
    size_t input_layernorm = gb->layernorm(hidden, layer.decoder_post_attn_layernorm_weight);
    size_t sa = build_decoder_self_attention(gb, input_layernorm, layer_idx, backend, use_cache, position_offset);
    size_t x_post_sa = gb->add(hidden, sa);
    size_t post_attention_layernorm = gb->layernorm(x_post_sa, layer.decoder_post_encoder_layernorm_weight);
    size_t ca = build_decoder_cross_attention(gb, post_attention_layernorm, layer_idx, backend, use_cache, position_offset);
    size_t x_post_ca = gb->add(x_post_sa, ca);
    size_t final_layernorm = gb->layernorm(x_post_ca, layer.decoder_post_ffn_layernorm_weight);
    size_t mlp_out = build_decoder_mlp_silu(
            gb, final_layernorm, layer.decoder_ffn_gate_weight, layer.decoder_ffn_gate_bias,
            layer.decoder_ffn_up_weight, layer.decoder_ffn_up_bias,
            layer.decoder_ffn2_weight, layer.decoder_ffn2_bias, backend, layer_idx
            );
    size_t x_post_ffn = gb->add(x_post_ca, mlp_out);
    std::string prefix = "model.decoder.layers." + std::to_string(layer_idx) + ".";
    return x_post_ffn;
}
size_t MoonshineModel::build_encoder(CactusGraph* gb, const std::vector<float>& audio_features)
{
    if (!audio_features.empty()) {
        float min_val = audio_features[0];
        float max_val = audio_features[0];
        double sum_val = 0.0;
        for (float val : audio_features) {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum_val += val;
        }
        (void)sum_val;
    }
    if (use_npu_encoder_ && npu_encoder_ && npu_encoder_->is_available()) {
        std::vector<int> input_shape = npu_encoder_->get_input_shape();
        if (input_shape.size() != 3) {
            input_shape = {1, 1, static_cast<int>(audio_features.size())};
        }
        if (input_shape[1] != 1) {
            throw std::runtime_error("Moonshine NPU encoder expects mono audio input");
        }
        size_t expected_samples = static_cast<size_t>(input_shape[2]);
        std::vector<__fp16> audio_f16(expected_samples, 0);
        size_t copy_samples = std::min(audio_features.size(), expected_samples);
        if (copy_samples > 0) {
            cactus_fp32_to_fp16(audio_features.data(), audio_f16.data(), copy_samples);
        }
        size_t T_enc = moonshine_downsampled_len(expected_samples);
        size_t D_enc = static_cast<size_t>(config_.hidden_dim);
        if (T_enc == 0 || D_enc == 0) {
            std::cout << "NPU encoder output has unexpected shape for input size "
                      << expected_samples << std::endl;
            std::cout << "Falling back to CPU encoder path." << std::endl;
            goto encoder_cpu_fallback;
        }
        
        size_t total_elements = T_enc * D_enc;
        std::vector<__fp16> npu_output(total_elements);
        size_t elements_written = npu_encoder_->encode(
            audio_f16.data(),
            npu_output.data(),
            input_shape,
            "x",
            ""
        );
        if (elements_written > 0) {
            size_t enc_output_node = gb->input({T_enc, D_enc}, Precision::FP16);
            gb->set_input(enc_output_node, npu_output.data(), Precision::FP16);
            size_t enc_output_persistent = gb->persistent(enc_output_node);
            last_encoder_post_norm_node_ = enc_output_persistent;
            return enc_output_persistent;
        }
    }
encoder_cpu_fallback:
    auto backend =
        (config_.default_backend == Config::Backend::CPU)
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;
    size_t audio_input = 0;
    const std::vector<float>* cpu_audio = &audio_features;
    std::vector<__fp16> audio_f16(cpu_audio->size());
    if (!cpu_audio->empty()) {
        cactus_fp32_to_fp16(cpu_audio->data(), audio_f16.data(), cpu_audio->size());
    }
    size_t audio_length = cpu_audio->size();  
    audio_input = gb->input({1, 1, audio_length}, Precision::FP16);
    gb->set_input(audio_input, audio_f16.data(), Precision::FP16);
    size_t conv2_transposed = build_audio_preprocessor(gb, audio_input);
    const auto& conv_shape = gb->get_output_buffer(conv2_transposed).shape;
    if (conv_shape.size() != 3 || conv_shape[0] != 1)
        throw std::runtime_error("Conv2 transpose should be [1, T_enc, D].");
    size_t T_enc = conv_shape[1];
    size_t D_enc = conv_shape[2];
    size_t h = gb->reshape(conv2_transposed, {T_enc, D_enc}); 
    for (uint32_t i = 0; i < config_.num_encoder_layers; ++i){
        h = build_encoder_transformer_block(gb, h, i, backend, false, 0);
    }
    size_t h_norm = gb->layernorm(
        h,
        weight_nodes_.encoder_layer_norm_weight
    );
    size_t h_norm_persistent = gb->persistent(h_norm);
    last_encoder_post_norm_node_ = h_norm_persistent;
    return h_norm_persistent;
}
size_t MoonshineModel::build_decoder(const std::vector<uint32_t>& tokens, bool use_cache, bool last_token_only) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    const size_t full_len = tokens.size();
    if (full_len == 0) {
        throw std::runtime_error("Decoder token list cannot be empty.");
    }
    auto backend = config_.default_backend == Config::Backend::CPU
                                                ? ComputeBackend::CPU
                                                : ComputeBackend::NPU;
    size_t start_idx = (use_cache && kv_cache_.current_seq_len > 0) ? full_len - 1 : 0;
    size_t new_tokens = full_len - start_idx;
    size_t position_offset = use_cache ? kv_cache_.current_seq_len : 0;
    size_t tok_input = gb->input({new_tokens}, Precision::FP32);
    std::vector<float> tok_f(new_tokens);
    for (size_t i = 0; i < new_tokens; i++) {
        tok_f[i] = static_cast<float>(tokens[start_idx + i]);
    }
    gb->set_input(tok_input, tok_f.data(), Precision::FP32);
    size_t dec_hidden = gb->embedding(embedding_node_id_, tok_input);
    for (uint32_t layer_idx = 0; layer_idx < config_.num_decoder_layers; ++layer_idx) {
        dec_hidden = build_decoder_transformer_block(
            gb,
            dec_hidden,
            layer_idx,
            backend,
            use_cache,
            position_offset
        );
    }
    size_t dec_norm = gb->layernorm(
        dec_hidden,
        weight_nodes_.decoder_norm_weight
    );
    size_t logits_input = dec_norm;
    if (last_token_only) {
        size_t row_index = new_tokens - 1;
        logits_input = gb->slice(logits_input, 0, row_index, 1); 
    }
    auto w_shape = gb->get_output_buffer(output_weight_node_id_).shape;
    size_t logits = gb->matmul(logits_input, output_weight_node_id_, true, backend);
    last_new_tokens_ = new_tokens;
    return logits;
}
size_t MoonshineModel::forward(const std::vector<float>& audio_features, const std::vector<uint32_t>& tokens, bool use_cache)
{
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->clear_debug_nodes();
    build_encoder(gb, audio_features);
    size_t logits = build_decoder(tokens, use_cache, true);
    return logits;
}
std::vector<float> MoonshineModel::get_audio_embeddings(const std::vector<float>& audio_features) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    build_encoder(gb, audio_features); 
    size_t pooled = gb->mean(last_encoder_post_norm_node_, 0);
    gb->execute();
    const auto& output_buf = gb->get_output_buffer(pooled);
    size_t hidden_dim = output_buf.total_size;
    std::vector<float> embedding(hidden_dim);
    void* output_data = gb->get_output(pooled);
    if (output_buf.precision == Precision::FP32) {
        const float* output_ptr = static_cast<const float*>(output_data);
        std::copy(output_ptr, output_ptr + hidden_dim, embedding.begin());
    } else if (output_buf.precision == Precision::FP16) {
        const __fp16* output_ptr = static_cast<const __fp16*>(output_data);
        Quantization::fp16_to_fp32(output_ptr, embedding.data(), hidden_dim);
    } else if (output_buf.precision == Precision::INT8) {
        const int8_t* output_ptr = static_cast<const int8_t*>(output_data);
        Quantization::int8_to_fp32(output_ptr, embedding.data(), hidden_dim, 1.0f);
    } else {
        throw std::runtime_error("Unsupported encoder embedding precision");
    }
    reset_cache();
    return embedding;
}
uint32_t MoonshineModel::decode_with_audio(
    const std::vector<uint32_t>& tokens,
    const std::vector<float>& audio_features,
    float temperature,
    float top_p,
    size_t top_k,
    const std::string& profile_file,
    float* out_entropy)
{
    if (!initialized_ || !graph_handle_)
        throw std::runtime_error("Model not initialized - call init() first");
    if (audio_features.empty())
        throw std::runtime_error("Audio features cannot be empty in Moonshine decode_with_audio");
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->clear_debug_nodes();
    bool cold_start = !encoder_ready_;
    size_t logits_node = 0;
    uint32_t bos = static_cast<uint32_t>(get_tokenizer()->get_bos_token());
    std::vector<uint32_t> full_tokens;
    full_tokens.reserve(tokens.size() + 1);
    full_tokens.push_back(bos);
    full_tokens.insert(full_tokens.end(), tokens.begin(), tokens.end());
    if (cold_start)
    {
        gb->soft_reset();
        kv_cache_.reset();
        kv_cache_.current_seq_len = 0;
        reset_graph_side_cache_nodes();
        encoder_kv_ready_ = false;
        first_decode_step_ = true;
        build_encoder(gb, audio_features);
        logits_node = build_decoder(full_tokens, false, false);
        encoder_ready_ = true;
    }
   	else
    {
        gb->soft_reset();
        reset_graph_side_cache_nodes();
        std::vector<uint32_t> last_token_vec = { tokens.back() };
        logits_node = build_decoder(last_token_vec, true, true);
    }
    size_t sampled_token_id = gb->sample(logits_node, temperature, top_p, top_k);
    if (!profile_file.empty()) gb->execute(profile_file);
   	else gb->execute();
    if (out_entropy) {
        const auto& logits_buf = gb->get_output_buffer(logits_node);
        void* logits_ptr = gb->get_output(logits_node);
        size_t vocab_size = logits_buf.shape.back();
        std::vector<float> logits(vocab_size);
        if (logits_buf.precision == Precision::FP32) {
            float* src = static_cast<float*>(logits_ptr);
            std::copy(src, src + vocab_size, logits.begin());
        }
        else if (logits_buf.precision == Precision::FP16) {
            __fp16* src = static_cast<__fp16*>(logits_ptr);
            Quantization::fp16_to_fp32(src, logits.data(), vocab_size);
        }
        else {
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
    post_execute_updates(gb, full_tokens.size());
    update_kv_cache(gb, last_new_tokens_);
    auto* out_ptr = gb->get_output(sampled_token_id);
    uint32_t sampled = *reinterpret_cast<uint32_t*>(out_ptr);
    return sampled;
}
}
}
