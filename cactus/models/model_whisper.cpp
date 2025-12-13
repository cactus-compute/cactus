#include "model.h"
#include "../graph/graph.h"
#include "../npu/npu.h"
#include "../kernel/kernel.h"
#include <cmath>
#include <stdexcept>
#include <set>
#include <iostream>

namespace cactus {
namespace engine {


struct ConvDebugNodes {
    size_t conv1;
    size_t conv2;
    size_t conv2_transposed;
    size_t output;
};


WhisperModel::WhisperModel() : Model() {}

WhisperModel::WhisperModel(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);

    float hd = static_cast<float>(config.attention_head_dim);
    if (hd <= 0.0f) {
        hd = 64.0f;
    }

    attention_scale_ = 1.0f / std::sqrt(hd);

    encoder_block_out_nodes_.resize(config.num_layers, 0);
    encoder_k_nodes_.assign(config.num_layers, 0);
    encoder_v_nodes_.assign(config.num_layers, 0);

}

void WhisperModel::load_weights_to_graph(CactusGraph* gb) {

    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    
    weight_nodes_.decoder_norm_weight = gb->mmap_weights(model_folder_path_ + "/decoder_norm.weights");
    weight_nodes_.decoder_norm_bias = gb->mmap_weights(model_folder_path_ + "/decoder_norm.bias");
    weight_nodes_.decoder_position_embeddings_weight = gb->mmap_weights(model_folder_path_ + "/decoder_position_embeddings.weights");

    if (config_.tie_word_embeddings) {
        weight_nodes_.output_weight = embedding_node_id_;
        output_weight_node_id_ = embedding_node_id_;
    } else {
        weight_nodes_.output_weight = gb->mmap_weights(model_folder_path_ + "/output_weight.weights");
        output_weight_node_id_ = weight_nodes_.output_weight;
    }

    if (npu::is_npu_available()) {
        std::string npu_encoder_path = model_folder_path_ + "/model.mlpackage";
        npu_encoder_ = npu::create_encoder();
        if (npu_encoder_ && npu_encoder_->load(npu_encoder_path)) {
            use_npu_encoder_ = true;

            std::vector<int> typical_input_shape = {1, 80, 3000};
            npu_encoder_->preallocate(typical_input_shape, "x", "");
        } else {
            use_npu_encoder_ = false;
            npu_encoder_.reset();
        }
    }

    if (!use_npu_encoder_) {
        weight_nodes_.encoder_position_embeddings = gb->mmap_weights(model_folder_path_ + "/encoder_position_embeddings.weights");
        weight_nodes_.encoder_conv1_weight = gb->mmap_weights(model_folder_path_ + "/encoder_conv1_weight.weights");
        weight_nodes_.encoder_conv1_bias = gb->mmap_weights(model_folder_path_ + "/encoder_conv1_bias.bias");
        weight_nodes_.encoder_conv2_weight = gb->mmap_weights(model_folder_path_ + "/encoder_conv2_weight.weights");
        weight_nodes_.encoder_conv2_bias = gb->mmap_weights(model_folder_path_ + "/encoder_conv2_bias.bias");
        weight_nodes_.encoder_norm_weight = gb->mmap_weights(model_folder_path_ + "/encoder_norm_weight.weights");
        weight_nodes_.encoder_norm_bias = gb->mmap_weights(model_folder_path_ + "/encoder_norm_bias.bias");
    }

    for (uint32_t i = 0; i < config_.num_layers; i++) {
        auto& layer = weight_nodes_.layers[i];

        // Decoder Layers (always needed)
        std::string layer_prefix = model_folder_path_ + "/decoder.layer_" + std::to_string(i) + "_";

        layer.decoder_encoder_attn_k_weight = gb->mmap_weights(layer_prefix + "encoder_attn_k.weights");
        layer.decoder_encoder_attn_q_weight = gb->mmap_weights(layer_prefix + "encoder_attn_q.weights");
        layer.decoder_encoder_attn_v_weight = gb->mmap_weights(layer_prefix + "encoder_attn_v.weights");
        layer.decoder_encoder_attn_output_weight = gb->mmap_weights(layer_prefix + "encoder_attn_output.weights");
        layer.decoder_encoder_attn_q_bias = gb->mmap_weights(layer_prefix + "encoder_attn_q.bias");
        layer.decoder_encoder_attn_v_bias = gb->mmap_weights(layer_prefix + "encoder_attn_v.bias");
        layer.decoder_encoder_attn_output_bias = gb->mmap_weights(layer_prefix + "encoder_attn_output.bias");

        layer.decoder_post_encoder_layernorm_weight = gb->mmap_weights(layer_prefix + "encoder_attn_norm.weights");
        layer.decoder_post_encoder_layernorm_bias = gb->mmap_weights(layer_prefix + "encoder_attn_norm.bias");

        layer.decoder_ffn1_weight = gb->mmap_weights(layer_prefix + "mlp_fc1.weights");
        layer.decoder_ffn1_bias = gb->mmap_weights(layer_prefix + "mlp_fc1.bias");
        layer.decoder_ffn2_weight = gb->mmap_weights(layer_prefix + "mlp_fc2.weights");
        layer.decoder_ffn2_bias = gb->mmap_weights(layer_prefix + "mlp_fc2.bias");

        layer.decoder_post_ffn_layernorm_weight = gb->mmap_weights(layer_prefix + "final_norm.weights");
        layer.decoder_post_ffn_layernorm_bias = gb->mmap_weights(layer_prefix + "final_norm.bias");

        layer.decoder_self_attn_k_weight = gb->mmap_weights(layer_prefix + "self_attn_k.weights");
        layer.decoder_self_attn_q_weight = gb->mmap_weights(layer_prefix + "self_attn_q.weights");
        layer.decoder_self_attn_v_weight = gb->mmap_weights(layer_prefix + "self_attn_v.weights");
        layer.decoder_self_attn_output_weight = gb->mmap_weights(layer_prefix + "self_attn_output.weights");
        layer.decoder_self_attn_q_bias = gb->mmap_weights(layer_prefix + "self_attn_q.bias");
        layer.decoder_self_attn_v_bias = gb->mmap_weights(layer_prefix + "self_attn_v.bias");
        layer.decoder_self_attn_output_bias = gb->mmap_weights(layer_prefix + "self_attn_output.bias");

        layer.decoder_post_attn_layernorm_weight = gb->mmap_weights(layer_prefix + "self_attn_norm.weights");
        layer.decoder_post_attn_layernorm_bias = gb->mmap_weights(layer_prefix + "self_attn_norm.bias");

        if (!use_npu_encoder_) {
            layer_prefix = model_folder_path_ + "/encoder.layer_" + std::to_string(i) + "_";

            layer.encoder_ffn1_weight = gb->mmap_weights(layer_prefix + "mlp_fc1.weights");
            layer.encoder_ffn1_bias = gb->mmap_weights(layer_prefix + "mlp_fc1.bias");
            layer.encoder_ffn2_weight = gb->mmap_weights(layer_prefix + "mlp_fc2.weights");
            layer.encoder_ffn2_bias = gb->mmap_weights(layer_prefix + "mlp_fc2.bias");

            layer.encoder_post_ffn_layernorm_weight = gb->mmap_weights(layer_prefix + "final_norm.weights");
            layer.encoder_post_ffn_layernorm_bias = gb->mmap_weights(layer_prefix + "final_norm.bias");

            layer.encoder_self_attn_k_weight = gb->mmap_weights(layer_prefix + "self_attn_k.weights");
            layer.encoder_self_attn_q_weight = gb->mmap_weights(layer_prefix + "self_attn_q.weights");
            layer.encoder_self_attn_v_weight = gb->mmap_weights(layer_prefix + "self_attn_v.weights");
            layer.encoder_self_attn_output_weight = gb->mmap_weights(layer_prefix + "self_attn_output.weights");
            layer.encoder_self_attn_q_bias = gb->mmap_weights(layer_prefix + "self_attn_q.bias");
            layer.encoder_self_attn_v_bias = gb->mmap_weights(layer_prefix + "self_attn_v.bias");
            layer.encoder_self_attn_output_bias = gb->mmap_weights(layer_prefix + "self_attn_output.bias");

            layer.encoder_post_attn_layernorm_weight = gb->mmap_weights(layer_prefix + "self_attn_norm.weights");
            layer.encoder_post_attn_layernorm_bias = gb->mmap_weights(layer_prefix + "self_attn_norm.bias");
        }
    }
}   

size_t WhisperModel::build_encoder_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = weight_nodes_.layers[layer_idx];

    auto ffn1_weight = gb->matmul(input, layer.encoder_ffn1_weight, true, backend);
    auto ffn1_bias = gb->add(ffn1_weight, layer.encoder_ffn1_bias);

    encoder_pre_gelu = ffn1_bias;

    auto ffn1_act = gb->gelu_erf(ffn1_bias);

    encoder_post_gelu = ffn1_act;

    auto ffn2_weight = gb->matmul(ffn1_act, layer.encoder_ffn2_weight, true, backend);
    auto ffn2_bias = gb->add(ffn2_weight, layer.encoder_ffn2_bias);
    return ffn2_bias;
}

size_t WhisperModel::build_decoder_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto ffn1_weight = gb->matmul(input, layer.decoder_ffn1_weight, true, backend);
    auto ffn1_bias = gb->add(ffn1_weight, layer.decoder_ffn1_bias);
    auto ffn1_act = gb->gelu_erf(ffn1_bias);
    auto ffn2_weight = gb->matmul(ffn1_act, layer.decoder_ffn2_weight, true, backend);
    auto ffn2_bias = gb->add(ffn2_weight, layer.decoder_ffn2_bias);
    return ffn2_bias;
}



size_t WhisperModel::build_encoder_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t /*position_offset*/){

    const auto& layer = weight_nodes_.layers[layer_idx];

    size_t q = gb->matmul(input, layer.decoder_encoder_attn_q_weight, true, backend);
    q = gb->add(q, layer.decoder_encoder_attn_q_bias);

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

    if (use_cache && encoder_kv_ready_) {
        const auto& k_shape = encoder_k_shape_[layer_idx];
        const auto& v_shape = encoder_v_shape_[layer_idx];

        size_t cache_k_node = gb->input(k_shape, encoder_kv_precision_);
        size_t cache_v_node = gb->input(v_shape, encoder_kv_precision_);

        gb->set_input(cache_k_node, encoder_k_host_[layer_idx].data(), encoder_kv_precision_);
        gb->set_input(cache_v_node, encoder_v_host_[layer_idx].data(), encoder_kv_precision_);

        k_4d = cache_k_node;
        v_4d = cache_v_node;
    } else {
        size_t enc_norm = weight_nodes_.encoder_output;

        size_t k = gb->matmul(enc_norm, layer.decoder_encoder_attn_k_weight, true, backend);
        size_t v = gb->matmul(enc_norm, layer.decoder_encoder_attn_v_weight, true, backend);
        v = gb->add(v, layer.decoder_encoder_attn_v_bias);

        const auto& k_buf = gb->get_output_buffer(k);
        if (k_buf.shape.size() != 2) {
            throw std::runtime_error("encoder cross-attn: k must be [T_enc, D]");
        }
        size_t T_enc = k_buf.shape[0];

        k_4d = gb->reshape(k, {1, T_enc, kv_heads, head_dim});
        v_4d = gb->reshape(v, {1, T_enc, kv_heads, head_dim});

        if (!encoder_kv_ready_) {
            encoder_k_nodes_[layer_idx] = k_4d;
            encoder_v_nodes_[layer_idx] = v_4d;
        }
    }

    size_t attn = gb->attention(q, k_4d, v_4d, attention_scale_, false);

    attn = gb->reshape(attn, {T_dec, q_heads * head_dim});
    size_t out = gb->matmul(attn, layer.decoder_encoder_attn_output_weight, true, backend);
    out = gb->add(out, layer.decoder_encoder_attn_output_bias);

    return out;
}

void WhisperModel::reset_graph_side_cache_nodes() {
    cache_k_output_nodes_.assign(config_.num_layers, 0);
    cache_v_output_nodes_.assign(config_.num_layers, 0);
}

void WhisperModel::reset_cache() {
    Model::reset_cache();
    encoder_ready_ = false;
    encoder_kv_ready_ = false;
    first_decode_step_ = true;
    encoder_output_host_.clear();
    encoder_k_host_.clear();
    encoder_v_host_.clear();
    encoder_k_shape_.clear();
    encoder_v_shape_.clear();
}

size_t WhisperModel::build_decoder_self_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t position_offset){
    const auto& layer = weight_nodes_.layers[layer_idx];

    auto q = gb->matmul(input, layer.decoder_self_attn_q_weight, true, backend);
    q = gb->add(q, layer.decoder_self_attn_q_bias);

    auto k = gb->matmul(input, layer.decoder_self_attn_k_weight, true, backend);
    auto v = gb->matmul(input, layer.decoder_self_attn_v_weight, true, backend);
    v = gb->add(v, layer.decoder_self_attn_v_bias);

    const auto& q_shape = gb->get_output_buffer(q).shape;
    if (q_shape.size() != 2) {
        throw std::runtime_error("decoder self-attn: q must be [T_new, D]");
    }

    size_t seq_new = q_shape[0];
    size_t num_heads = config_.attention_heads;
    size_t head_dim = config_.attention_head_dim;
    size_t num_kv_heads = config_.attention_kv_heads;
 
    auto q_4d = gb->reshape(q, {1, seq_new, num_heads, head_dim});
    auto k_4d = gb->reshape(k, {1, seq_new, num_kv_heads, head_dim});
    auto v_4d = gb->reshape(v, {1, seq_new, num_kv_heads, head_dim});

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
            {1, cache_len, num_kv_heads, head_dim},
            kv_cache_.precision
        );
        size_t cache_v_node = gb->input(
            {1, cache_len, num_kv_heads, head_dim},
            kv_cache_.precision
        );
 
        if (k_view.ptr2 == nullptr && v_view.ptr2 == nullptr) {
            gb->set_input(cache_k_node, k_view.ptr1, kv_cache_.precision);
            gb->set_input(cache_v_node, v_view.ptr1, kv_cache_.precision);
        } else {
            gb->set_input(cache_k_node, kv_cache_.get_key_ptr(layer_idx), kv_cache_.precision);
            gb->set_input(cache_v_node, kv_cache_.get_value_ptr(layer_idx), kv_cache_.precision);
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
 
    auto attn_out_4d = gb->attention(q_4d, final_k, final_v, attention_scale_, position_offset);
    auto attn_out    = gb->reshape(attn_out_4d, {seq_new, num_heads * head_dim});

    auto output = gb->matmul(attn_out, layer.decoder_self_attn_output_weight, true, backend);
    output = gb->add(output, layer.decoder_self_attn_output_bias);
    return output;
}

size_t WhisperModel::build_encoder_self_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t /*position_offset*/){
    const auto& layer = weight_nodes_.layers[layer_idx];

    if(use_cache)
        throw std::runtime_error("The encoder attention layers are not auto-regressive, and thus don't use KV caching!");

    auto q = gb->matmul(input, layer.encoder_self_attn_q_weight, true, backend);
    q = gb->add(q, layer.encoder_self_attn_q_bias);
    auto v = gb->matmul(input, layer.encoder_self_attn_v_weight, true, backend);
    v = gb->add(v, layer.encoder_self_attn_v_bias);
    auto k = gb->matmul(input, layer.encoder_self_attn_k_weight, true, backend);

    size_t seq_len = gb->get_output_buffer(q).shape[0];
    size_t num_heads = config_.attention_heads;
    size_t head_dim  = config_.attention_head_dim;

    q = gb->reshape(q, {1, seq_len, num_heads, head_dim});
    k = gb->reshape(k, {1, seq_len, num_heads, head_dim});
    v = gb->reshape(v, {1, seq_len, num_heads, head_dim});

    auto attn = gb->attention(q, k, v, attention_scale_, false);

    attn = gb->reshape(attn, {seq_len, num_heads * head_dim});

    auto output = gb->matmul(attn, layer.encoder_self_attn_output_weight, true, backend);
    output = gb->add(output, layer.encoder_self_attn_output_bias);

    return output;
}

size_t WhisperModel::build_conv1d(CactusGraph* gb, size_t input)
{
    size_t conv_input = input;
    const auto& xbuf = gb->get_output_buffer(input);

    if (xbuf.precision == Precision::INT8) { 
        conv_input = gb->precision_cast(input, Precision::FP16);
    }

    size_t conv1 = gb->conv1d_k3(conv_input, weight_nodes_.encoder_conv1_weight, 1);

    auto bias1_shape = gb->get_output_buffer(weight_nodes_.encoder_conv1_bias).shape;
    size_t C1 = bias1_shape[0];
    size_t bias1 = gb->reshape(weight_nodes_.encoder_conv1_bias, {1, C1, 1});
    conv1 = gb->add(conv1, bias1);

    last_conv1_node_ = conv1;

    conv1 = gb->gelu_erf(conv1);

    size_t conv2 = gb->conv1d_k3(conv1, weight_nodes_.encoder_conv2_weight, 2);
    auto bias2_shape = gb->get_output_buffer(weight_nodes_.encoder_conv2_bias).shape;
    size_t C2 = bias2_shape[0];
    size_t bias2 = gb->reshape(weight_nodes_.encoder_conv2_bias, {1, C2, 1});
    conv2 = gb->add(conv2, bias2);

    last_conv2_node_ = conv2;

    conv2 = gb->gelu_erf(conv2);

    const auto& buf = gb->get_output_buffer(conv2);

    size_t conv2_transposed;
    if (buf.precision == Precision::FP16) {
        size_t conv2_f32 = gb->precision_cast(conv2, Precision::FP32);
        size_t conv2_T_f32 = gb->transpose(conv2_f32, ComputeBackend::CPU);
        conv2_transposed = gb->precision_cast(conv2_T_f32, Precision::FP16);
    } else {
        conv2_transposed = gb->transpose(conv2, ComputeBackend::CPU);
    }

    return conv2_transposed;
}

size_t WhisperModel::build_encoder_transformer_block(
    CactusGraph* gb,
    size_t hidden,
    uint32_t layer_idx,
    ComputeBackend backend,
    bool use_cache,
    size_t position_offset)
{
    const auto& layer = weight_nodes_.layers[layer_idx];

    size_t ln1 = gb->layernorm(
        hidden,
        layer.encoder_post_attn_layernorm_weight,
        layer.encoder_post_attn_layernorm_bias
    );

    size_t sa = build_encoder_self_attention(
        gb, ln1, layer_idx, backend, use_cache, position_offset
    );

    size_t x_post_sa = gb->add(hidden, sa);

    size_t ln2 = gb->layernorm(
        x_post_sa,
        layer.encoder_post_ffn_layernorm_weight,
        layer.encoder_post_ffn_layernorm_bias
    );

    size_t ffn_out = build_encoder_mlp(
        gb, ln2, layer_idx, backend
    );

    size_t out = gb->add(x_post_sa, ffn_out);

    if (layer_idx < encoder_block_out_nodes_.size()) {
        encoder_block_out_nodes_[layer_idx] = out;
    }

    return out;
}

size_t WhisperModel::build_decoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t position_offset){
    const auto& layer = weight_nodes_.layers[layer_idx];

    size_t ln1 = gb->layernorm(hidden, layer.decoder_post_attn_layernorm_weight, layer.decoder_post_attn_layernorm_bias);
    size_t sa = build_decoder_self_attention(gb, ln1, layer_idx, backend, use_cache, position_offset);
    size_t x_post_sa = gb->add(hidden, sa);

    size_t ln2 = gb->layernorm(x_post_sa, layer.decoder_post_encoder_layernorm_weight, layer.decoder_post_encoder_layernorm_bias);
    size_t ca = build_encoder_attention(gb, ln2, layer_idx, backend, use_cache, position_offset);
    size_t x_post_ca = gb->add(x_post_sa, ca);

    size_t ln3 = gb->layernorm(x_post_ca,layer.decoder_post_ffn_layernorm_weight,layer.decoder_post_ffn_layernorm_bias);
    size_t ffn_out = build_decoder_mlp(gb, ln3, layer_idx, backend);
    size_t x_post_ffn = gb->add(x_post_ca, ffn_out);

    return x_post_ffn;

}

void WhisperModel::run_encoder(const std::vector<float>& mel_bins)
{
    if (mel_bins.size() % 80 != 0)
        throw std::runtime_error("Mel bins length must be divisible by 80.");

    size_t T_mel = mel_bins.size() / 80;
    if (T_mel == 0)
        throw std::runtime_error("Mel bins has zero frames.");

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    if (!gb)
        throw std::runtime_error("Graph handle is null in run_encoder.");

    if (use_npu_encoder_ && npu_encoder_ && npu_encoder_->is_available()) {
        std::vector<int> out_shape = npu_encoder_->get_output_shape();
        size_t T_enc, D_enc;
        if (out_shape.size() == 3) {
            T_enc = static_cast<size_t>(out_shape[1]);
            D_enc = static_cast<size_t>(out_shape[2]);
        } else if (out_shape.size() == 2) {
            T_enc = static_cast<size_t>(out_shape[0]);
            D_enc = static_cast<size_t>(out_shape[1]);
        } else {
            throw std::runtime_error("NPU encoder output has unexpected shape");
        }

        std::vector<__fp16> mel_bins_f16(mel_bins.size());
        cactus_fp32_to_fp16(mel_bins.data(), mel_bins_f16.data(), mel_bins.size());

        std::vector<int> input_shape = {1, 80, static_cast<int>(T_mel)};

        __fp16* output_buffer = npu_encoder_->get_output_buffer();
        if (output_buffer) {
            size_t elements_written = npu_encoder_->encode(
                mel_bins_f16.data(),
                output_buffer,  
                input_shape,
                "x",
                ""
            );

            if (elements_written > 0) {
                size_t enc_output_node = gb->input({T_enc, D_enc}, Precision::FP16);
                gb->set_input(enc_output_node, output_buffer, Precision::FP16);

                weight_nodes_.encoder_output = enc_output_node;
                return;
            }
        } else {
            std::vector<__fp16> npu_output(T_enc * D_enc);
            size_t elements_written = npu_encoder_->encode(
                mel_bins_f16.data(),
                npu_output.data(),
                input_shape,
                "x",
                ""
            );

            if (elements_written > 0) {
                size_t enc_output_node = gb->input({T_enc, D_enc}, Precision::FP16);
                gb->set_input(enc_output_node, npu_output.data(), Precision::FP16);

                weight_nodes_.encoder_output = enc_output_node;
                return;
            }
        }
    }

    auto backend =
        (config_.default_backend == Config::Backend::CPU)
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    Precision act_precision;
    switch (config_.precision) {
        case Config::Precision::INT8:  act_precision = Precision::INT8;  break;
        case Config::Precision::FP16:  act_precision = Precision::FP16;  break;
        case Config::Precision::FP32:
        default:                       act_precision = Precision::FP32;  break;
    }

    size_t mel_input = 0;

    if (act_precision == Precision::FP16) {
        std::vector<__fp16> mel_bins_f16(mel_bins.size());
        cactus_fp32_to_fp16(mel_bins.data(), mel_bins_f16.data(), mel_bins.size());

        mel_input = gb->input({1, 80, T_mel}, Precision::FP16);
        gb->set_input(mel_input, mel_bins_f16.data(), Precision::FP16);
    }

    else if (act_precision == Precision::INT8) {
        std::vector<int8_t> mel_bins_i8(mel_bins.size());
        for (size_t i = 0; i < mel_bins.size(); ++i) {
            mel_bins_i8[i] = static_cast<int8_t>(mel_bins[i]);
        }

        mel_input = gb->input({1, 80, T_mel}, Precision::INT8);
        gb->set_input(mel_input, mel_bins_i8.data(), Precision::INT8);
    }

    else {
        mel_input = gb->input({1, 80, T_mel}, Precision::FP32);
        gb->set_input(mel_input, mel_bins.data(), Precision::FP32);
    }

    size_t conv2_transposed = build_conv1d(gb, mel_input);

    const auto& conv_shape = gb->get_output_buffer(conv2_transposed).shape;
    if (conv_shape.size() != 3 || conv_shape[0] != 1)
        throw std::runtime_error("Conv2 transpose should be [1, T_enc, D].");

    size_t T_enc = conv_shape[1];
    size_t D_enc = conv_shape[2];

    size_t pos_slice = gb->slice(weight_nodes_.encoder_position_embeddings, 0, 0, T_enc);

    size_t h2d = gb->reshape(conv2_transposed, {T_enc, D_enc});

    auto& h2d_buf = gb->get_output_buffer(h2d);
    auto& pos_buf = gb->get_output_buffer(pos_slice);

    if (pos_buf.precision != h2d_buf.precision) {
        pos_slice = gb->precision_cast(pos_slice, h2d_buf.precision);
    }

    size_t h_pos = gb->add(h2d, pos_slice);
    last_enc_plus_pos_node_ = h_pos;

    size_t h = h_pos;
    for (uint32_t i = 0; i < config_.num_layers; ++i){
        h = build_encoder_transformer_block(gb, h, i, backend, false, 0);
        if (i == 0) {
            encoder_transformer_block_0 = h;
        }
    }

    size_t h_norm = gb->layernorm(
        h,
        weight_nodes_.encoder_norm_weight,
        weight_nodes_.encoder_norm_bias
    );
    last_encoder_post_norm_node_ = h_norm;


    weight_nodes_.encoder_output = h_norm;
}



size_t WhisperModel::run_decoder_step(const std::vector<uint32_t>& tokens, bool use_cache, bool last_token_only) {
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

    size_t tok_input = gb->input({new_tokens}, Precision::FP32);
    std::vector<float> tok_f(new_tokens);
    for (size_t i = 0; i < new_tokens; i++) {
        tok_f[i] = static_cast<float>(tokens[start_idx + i]);
    }
    gb->set_input(tok_input, tok_f.data(), Precision::FP32);

    size_t dec_hidden = gb->embedding(embedding_node_id_, tok_input);

    size_t position_offset = kv_cache_.current_seq_len;
    size_t dec_pos = gb->slice(weight_nodes_.decoder_position_embeddings_weight,0,position_offset,new_tokens);

    {
        const auto& h_buf   = gb->get_output_buffer(dec_hidden);
        const auto& pos_buf = gb->get_output_buffer(dec_pos);

        size_t pos_node_for_add = dec_pos;
        if (pos_buf.precision != h_buf.precision) {
            pos_node_for_add = gb->precision_cast(dec_pos, h_buf.precision);
        }

        dec_hidden = gb->add(dec_hidden, pos_node_for_add);
    }

    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
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
        weight_nodes_.decoder_norm_weight,
        weight_nodes_.decoder_norm_bias
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


size_t WhisperModel::forward(const std::vector<float>& mel_bins, const std::vector<uint32_t>& tokens, bool use_cache)
{

    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized");
    }

    if (!use_cache) {
        kv_cache_.reset();
        kv_cache_.current_seq_len = 0;
        reset_graph_side_cache_nodes();
        run_encoder(mel_bins);
    }

    return run_decoder_step(tokens, use_cache, false);
}

std::vector<float> WhisperModel::get_audio_embeddings(const std::vector<float>& mel_bins) {
    run_encoder(mel_bins);

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

uint32_t WhisperModel::decode_with_audio(
    const std::vector<uint32_t>& tokens,
    const std::vector<float>& mel_bins,
    float temperature,
    float top_p,
    size_t top_k,
    const std::string& profile_file)
{
    if (!initialized_ || !graph_handle_)
        throw std::runtime_error("Model not initialized - call init() first");
    if (tokens.empty())
        throw std::runtime_error("Token sequence cannot be empty");
    if (mel_bins.empty())
        throw std::runtime_error("Mel bins cannot be empty in Whisper decode_with_audio");

    auto* gb = static_cast<CactusGraph*>(graph_handle_);

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
        encoder_k_nodes_.assign(config_.num_layers, 0);
        encoder_v_nodes_.assign(config_.num_layers, 0);
        encoder_k_host_.clear();
        encoder_v_host_.clear();
        encoder_k_shape_.clear();
        encoder_v_shape_.clear();

        first_decode_step_ = true;
        run_encoder(mel_bins);
        logits_node = run_decoder_step(full_tokens, false, false);
    }

    else
    {
        gb->soft_reset();
        reset_graph_side_cache_nodes();

        if (encoder_output_host_.empty())
            throw std::runtime_error("Missing encoder_output_host_ in warm step!");

        size_t enc_node = gb->input(encoder_output_shape_, encoder_output_precision_);
        gb->set_input(enc_node, encoder_output_host_.data(), encoder_output_precision_);
        weight_nodes_.encoder_output = enc_node;

        std::vector<uint32_t> last_token_vec = { tokens.back() };
        logits_node = run_decoder_step(last_token_vec, true, true);
    }
    
    size_t sampled_token_id = gb->sample(logits_node, temperature, top_p, top_k);
    if (!profile_file.empty()) gb->execute(profile_file);
    else gb->execute();

    if (cold_start)
    {
        auto& out_buf = gb->get_output_buffer(weight_nodes_.encoder_output);

        encoder_output_shape_      = out_buf.shape;
        encoder_output_precision_  = out_buf.precision;

        size_t total_elems = 1;
        for (auto s : out_buf.shape)
            total_elems *= s;

        size_t elem_size = 0;
        switch (out_buf.precision) {
            case Precision::FP32: elem_size = sizeof(float);    break;
            case Precision::FP16: elem_size = sizeof(uint16_t); break;
            case Precision::INT8: elem_size = sizeof(int8_t);   break;
            default:
                throw std::runtime_error("Unsupported encoder_output precision in WhisperModel");
        }

        const size_t total_bytes = total_elems * elem_size;

        encoder_output_host_.resize(total_bytes);
        std::memcpy(
            encoder_output_host_.data(),
            gb->get_output(weight_nodes_.encoder_output),
            total_bytes
        );

        {
            if (config_.num_layers == 0) {
                throw std::runtime_error("WhisperModel: num_layers is zero?");
            }

            auto& k0_buf = gb->get_output_buffer(encoder_k_nodes_[0]);
            encoder_kv_precision_ = k0_buf.precision;

            encoder_k_host_.resize(config_.num_layers);
            encoder_v_host_.resize(config_.num_layers);
            encoder_k_shape_.resize(config_.num_layers);
            encoder_v_shape_.resize(config_.num_layers);

            size_t kv_elem_size = 0;
            switch (encoder_kv_precision_) {
                case Precision::FP32: kv_elem_size = sizeof(float);    break;
                case Precision::FP16: kv_elem_size = sizeof(uint16_t); break;
                case Precision::INT8: kv_elem_size = sizeof(int8_t);   break;
                default:
                    throw std::runtime_error("Unsupported encoder K/V precision in WhisperModel");
            }

            for (uint32_t i = 0; i < config_.num_layers; ++i) {
                size_t k_node = encoder_k_nodes_[i];
                size_t v_node = encoder_v_nodes_[i];

                auto& k_buf = gb->get_output_buffer(k_node);
                auto& v_buf = gb->get_output_buffer(v_node);

                encoder_k_shape_[i] = k_buf.shape;
                encoder_v_shape_[i] = v_buf.shape;

                size_t k_elems = 1;
                for (auto s : k_buf.shape) k_elems *= s;
                size_t v_elems = 1;
                for (auto s : v_buf.shape) v_elems *= s;

                encoder_k_host_[i].resize(k_elems * kv_elem_size);
                encoder_v_host_[i].resize(v_elems * kv_elem_size);

                std::memcpy(
                    encoder_k_host_[i].data(),
                    gb->get_output(k_node),
                    k_elems * kv_elem_size
                );
                std::memcpy(
                    encoder_v_host_[i].data(),
                    gb->get_output(v_node),
                    v_elems * kv_elem_size
                );
            }

            encoder_kv_ready_ = true;
        }

        encoder_ready_ = true;
    }


    post_execute_updates(gb, full_tokens.size());
    update_kv_cache(gb, last_new_tokens_);

    auto* out_ptr = gb->get_output(sampled_token_id);
    uint32_t sampled = *reinterpret_cast<uint32_t*>(out_ptr);

    return sampled;
}


}
}
