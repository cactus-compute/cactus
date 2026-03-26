#include "model.h"
#include "../graph/graph.h"
#include "../kernel/kernel.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace cactus {
namespace engine {

namespace {

float sortformer_attention_scale(const Config& config) {
    float hd = static_cast<float>(config.attention_head_dim);
    if (hd <= 0.0f) {
        hd = 64.0f;
    }
    return 1.0f / std::sqrt(hd);
}

}  // namespace

SortformerDiarModel::SortformerDiarModel() : Model() {}

SortformerDiarModel::SortformerDiarModel(const Config& config) : Model(config) {
    weight_nodes_.fc_layers.resize(config.num_layers);
    weight_nodes_.tf_layers.resize(config.diar_tf_num_layers);
    attention_scale_ = sortformer_attention_scale(config);
}

bool SortformerDiarModel::init(const std::string& model_folder, size_t context_size,
                               const std::string& system_prompt, bool do_warmup) {
    (void)context_size;
    (void)system_prompt;
    (void)do_warmup;

    if (initialized_) {
        return true;
    }

    auto* gb = new CactusGraph();
    graph_handle_ = gb;
    owns_graph_ = true;
    return init_internal_sortformer(gb, model_folder);
}

bool SortformerDiarModel::init(CactusGraph* external_graph, const std::string& model_folder, size_t context_size,
                               const std::string& system_prompt, bool do_warmup) {
    (void)context_size;
    (void)system_prompt;
    (void)do_warmup;

    if (!external_graph) {
        throw std::invalid_argument("External graph pointer must not be null");
    }

    if (initialized_) {
        graph_handle_ = external_graph;
        owns_graph_ = false;
        return true;
    }

    graph_handle_ = external_graph;
    owns_graph_ = false;
    return init_internal_sortformer(external_graph, model_folder);
}

bool SortformerDiarModel::init_internal_sortformer(CactusGraph* gb, const std::string& model_folder) {
    model_folder_path_ = model_folder;
    if (!config_.from_json(model_folder + "/config.txt")) {
        return false;
    }

    weight_nodes_.fc_layers.resize(config_.num_layers);
    weight_nodes_.tf_layers.resize(config_.diar_tf_num_layers);
    attention_scale_ = sortformer_attention_scale(config_);
    load_weights_to_graph(gb);

    tokenizer_.reset();
    initialized_ = true;
    reset_cache();
    return true;
}

void SortformerDiarModel::load_weights_to_graph(CactusGraph* gb) {
    auto mmap_optional = [&](const std::string& path) -> size_t {
        return std::filesystem::exists(path) ? gb->mmap_weights(path) : 0;
    };

    weight_nodes_.subsampling_conv0_weight = gb->mmap_weights(model_folder_path_ + "/subsampling_conv0_weight.weights");
    weight_nodes_.subsampling_conv0_bias = gb->mmap_weights(model_folder_path_ + "/subsampling_conv0_bias.bias");
    weight_nodes_.subsampling_depthwise1_weight = gb->mmap_weights(model_folder_path_ + "/subsampling_depthwise1_weight.weights");
    weight_nodes_.subsampling_depthwise1_bias = gb->mmap_weights(model_folder_path_ + "/subsampling_depthwise1_bias.bias");
    weight_nodes_.subsampling_pointwise1_weight = gb->mmap_weights(model_folder_path_ + "/subsampling_pointwise1_weight.weights");
    weight_nodes_.subsampling_pointwise1_bias = gb->mmap_weights(model_folder_path_ + "/subsampling_pointwise1_bias.bias");
    weight_nodes_.subsampling_depthwise2_weight = gb->mmap_weights(model_folder_path_ + "/subsampling_depthwise2_weight.weights");
    weight_nodes_.subsampling_depthwise2_bias = gb->mmap_weights(model_folder_path_ + "/subsampling_depthwise2_bias.bias");
    weight_nodes_.subsampling_pointwise2_weight = gb->mmap_weights(model_folder_path_ + "/subsampling_pointwise2_weight.weights");
    weight_nodes_.subsampling_pointwise2_bias = gb->mmap_weights(model_folder_path_ + "/subsampling_pointwise2_bias.bias");
    weight_nodes_.subsampling_linear_weight = gb->mmap_weights(model_folder_path_ + "/subsampling_linear_weight.weights");
    weight_nodes_.subsampling_linear_bias = gb->mmap_weights(model_folder_path_ + "/subsampling_linear_bias.bias");

    weight_nodes_.encoder_proj_weight = mmap_optional(model_folder_path_ + "/encoder_proj_weight.weights");
    weight_nodes_.encoder_proj_bias = mmap_optional(model_folder_path_ + "/encoder_proj_bias.bias");
    weight_nodes_.tf_embed_positions = mmap_optional(model_folder_path_ + "/tf_embed_positions.weights");

    weight_nodes_.head_hidden_weight = gb->mmap_weights(model_folder_path_ + "/head_hidden_weight.weights");
    weight_nodes_.head_hidden_bias = gb->mmap_weights(model_folder_path_ + "/head_hidden_bias.bias");
    weight_nodes_.head_single_spk_weight = gb->mmap_weights(model_folder_path_ + "/head_single_spk_weight.weights");
    weight_nodes_.head_single_spk_bias = gb->mmap_weights(model_folder_path_ + "/head_single_spk_bias.bias");
    weight_nodes_.head_pair_spk_weight = mmap_optional(model_folder_path_ + "/head_pair_spk_weight.weights");
    weight_nodes_.head_pair_spk_bias = mmap_optional(model_folder_path_ + "/head_pair_spk_bias.bias");

    for (uint32_t i = 0; i < config_.num_layers; ++i) {
        auto& layer = weight_nodes_.fc_layers[i];
        std::string layer_prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";

        layer.ff1_linear1_weight = gb->mmap_weights(layer_prefix + "ff1_linear1.weights");
        layer.ff1_linear1_bias = gb->mmap_weights(layer_prefix + "ff1_linear1.bias");
        layer.ff1_linear2_weight = gb->mmap_weights(layer_prefix + "ff1_linear2.weights");
        layer.ff1_linear2_bias = gb->mmap_weights(layer_prefix + "ff1_linear2.bias");
        layer.ff2_linear1_weight = gb->mmap_weights(layer_prefix + "ff2_linear1.weights");
        layer.ff2_linear1_bias = gb->mmap_weights(layer_prefix + "ff2_linear1.bias");
        layer.ff2_linear2_weight = gb->mmap_weights(layer_prefix + "ff2_linear2.weights");
        layer.ff2_linear2_bias = gb->mmap_weights(layer_prefix + "ff2_linear2.bias");

        layer.self_attn_q_weight = gb->mmap_weights(layer_prefix + "self_attn_q.weights");
        layer.self_attn_q_bias = gb->mmap_weights(layer_prefix + "self_attn_q.bias");
        layer.self_attn_k_weight = gb->mmap_weights(layer_prefix + "self_attn_k.weights");
        layer.self_attn_k_bias = gb->mmap_weights(layer_prefix + "self_attn_k.bias");
        layer.self_attn_v_weight = gb->mmap_weights(layer_prefix + "self_attn_v.weights");
        layer.self_attn_v_bias = gb->mmap_weights(layer_prefix + "self_attn_v.bias");
        layer.self_attn_output_weight = gb->mmap_weights(layer_prefix + "self_attn_output.weights");
        layer.self_attn_output_bias = gb->mmap_weights(layer_prefix + "self_attn_output.bias");
        layer.self_attn_relative_k_weight = gb->mmap_weights(layer_prefix + "self_attn_relative_k.weights");
        layer.self_attn_bias_u = gb->mmap_weights(layer_prefix + "self_attn_bias_u.weights");
        layer.self_attn_bias_v = gb->mmap_weights(layer_prefix + "self_attn_bias_v.weights");

        layer.norm_ff1_weight = gb->mmap_weights(layer_prefix + "norm_ff1.weights");
        layer.norm_ff1_bias = gb->mmap_weights(layer_prefix + "norm_ff1.bias");
        layer.norm_self_attn_weight = gb->mmap_weights(layer_prefix + "norm_self_attn.weights");
        layer.norm_self_attn_bias = gb->mmap_weights(layer_prefix + "norm_self_attn.bias");
        layer.norm_conv_weight = gb->mmap_weights(layer_prefix + "norm_conv.weights");
        layer.norm_conv_bias = gb->mmap_weights(layer_prefix + "norm_conv.bias");
        layer.norm_ff2_weight = gb->mmap_weights(layer_prefix + "norm_ff2.weights");
        layer.norm_ff2_bias = gb->mmap_weights(layer_prefix + "norm_ff2.bias");
        layer.norm_out_weight = gb->mmap_weights(layer_prefix + "norm_out.weights");
        layer.norm_out_bias = gb->mmap_weights(layer_prefix + "norm_out.bias");

        layer.conv_pointwise1_weight = gb->mmap_weights(layer_prefix + "conv_pointwise1.weights");
        layer.conv_pointwise1_bias = gb->mmap_weights(layer_prefix + "conv_pointwise1.bias");
        layer.conv_depthwise_weight = gb->mmap_weights(layer_prefix + "conv_depthwise.weights");
        layer.conv_depthwise_bias = gb->mmap_weights(layer_prefix + "conv_depthwise.bias");
        layer.conv_pointwise2_weight = gb->mmap_weights(layer_prefix + "conv_pointwise2.weights");
        layer.conv_pointwise2_bias = gb->mmap_weights(layer_prefix + "conv_pointwise2.bias");
        layer.conv_batchnorm_weight = gb->mmap_weights(layer_prefix + "conv_batchnorm_weight.weights");
        layer.conv_batchnorm_bias = gb->mmap_weights(layer_prefix + "conv_batchnorm_bias.bias");
        layer.conv_batchnorm_running_mean = gb->mmap_weights(layer_prefix + "conv_batchnorm_running_mean.weights");
        layer.conv_batchnorm_running_var = gb->mmap_weights(layer_prefix + "conv_batchnorm_running_var.weights");
    }

    for (uint32_t i = 0; i < config_.diar_tf_num_layers; ++i) {
        auto& layer = weight_nodes_.tf_layers[i];
        std::string prefix = model_folder_path_ + "/tf_layer_" + std::to_string(i) + "_";

        layer.self_attn_q_weight = gb->mmap_weights(prefix + "self_attn_q.weights");
        layer.self_attn_q_bias = gb->mmap_weights(prefix + "self_attn_q.bias");
        layer.self_attn_k_weight = gb->mmap_weights(prefix + "self_attn_k.weights");
        layer.self_attn_k_bias = mmap_optional(prefix + "self_attn_k.bias");
        layer.self_attn_v_weight = gb->mmap_weights(prefix + "self_attn_v.weights");
        layer.self_attn_v_bias = gb->mmap_weights(prefix + "self_attn_v.bias");
        layer.self_attn_output_weight = gb->mmap_weights(prefix + "self_attn_output.weights");
        layer.self_attn_output_bias = gb->mmap_weights(prefix + "self_attn_output.bias");
        layer.self_attn_layernorm_weight = gb->mmap_weights(prefix + "self_attn_layernorm.weights");
        layer.self_attn_layernorm_bias = gb->mmap_weights(prefix + "self_attn_layernorm.bias");
        layer.ff1_weight = gb->mmap_weights(prefix + "ff1.weights");
        layer.ff1_bias = gb->mmap_weights(prefix + "ff1.bias");
        layer.ff2_weight = gb->mmap_weights(prefix + "ff2.weights");
        layer.ff2_bias = gb->mmap_weights(prefix + "ff2.bias");
        layer.final_layernorm_weight = gb->mmap_weights(prefix + "final_layernorm.weights");
        layer.final_layernorm_bias = gb->mmap_weights(prefix + "final_layernorm.bias");
    }
}

size_t SortformerDiarModel::build_subsampling(CactusGraph* gb, const std::vector<float>& audio_features) {
    const size_t num_mels = std::max<size_t>(1, static_cast<size_t>(config_.num_mel_bins));
    if (audio_features.empty() || (audio_features.size() % num_mels) != 0) {
        throw std::runtime_error("Sortformer expects audio_features with shape [num_mels, num_frames]");
    }

    const size_t frames = audio_features.size() / num_mels;
    std::vector<float> time_major(frames * num_mels);
    for (size_t m = 0; m < num_mels; ++m) {
        const float* src = &audio_features[m * frames];
        for (size_t t = 0; t < frames; ++t) {
            time_major[t * num_mels + m] = src[t];
        }
    }

    std::vector<__fp16> features_f16(time_major.size());
    cactus_fp32_to_fp16(time_major.data(), features_f16.data(), time_major.size());

    size_t x = gb->input({1, 1, frames, num_mels}, Precision::FP16);
    gb->set_input(x, features_f16.data(), Precision::FP16);

    x = gb->conv2d_k3s2p1(x, weight_nodes_.subsampling_conv0_weight, weight_nodes_.subsampling_conv0_bias);
    x = gb->relu(x);
    x = gb->conv2d_depthwise_k3s2p1(x, weight_nodes_.subsampling_depthwise1_weight, weight_nodes_.subsampling_depthwise1_bias);
    x = gb->conv2d_pointwise_1x1(x, weight_nodes_.subsampling_pointwise1_weight, weight_nodes_.subsampling_pointwise1_bias);
    x = gb->relu(x);
    x = gb->conv2d_depthwise_k3s2p1(x, weight_nodes_.subsampling_depthwise2_weight, weight_nodes_.subsampling_depthwise2_bias);
    x = gb->conv2d_pointwise_1x1(x, weight_nodes_.subsampling_pointwise2_weight, weight_nodes_.subsampling_pointwise2_bias);
    x = gb->relu(x);

    const auto& conv_shape = gb->get_output_buffer(x).shape;
    if (conv_shape.size() != 4 || conv_shape[0] != 1) {
        throw std::runtime_error("Sortformer subsampling produced invalid shape");
    }

    const size_t c = conv_shape[1];
    const size_t t = conv_shape[2];
    const size_t w = conv_shape[3];

    size_t t_major = gb->transposeN(x, {0, 2, 1, 3}, ComputeBackend::CPU);
    size_t flattened = gb->reshape(t_major, {t, c * w});
    size_t projected = gb->matmul(flattened, weight_nodes_.subsampling_linear_weight, true, ComputeBackend::CPU);
    projected = gb->add(projected, weight_nodes_.subsampling_linear_bias);
    return projected;
}

size_t SortformerDiarModel::build_relative_position_embeddings(CactusGraph* gb, size_t seq_len) {
    const size_t hidden_dim = std::max<size_t>(1, static_cast<size_t>(config_.hidden_dim));
    const size_t half_dim = hidden_dim / 2;
    const size_t rel_len = 2 * seq_len - 1;

    std::vector<float> pos_embed(rel_len * hidden_dim, 0.0f);
    for (size_t p = 0; p < rel_len; ++p) {
        const int rel_pos = static_cast<int>(seq_len - 1) - static_cast<int>(p);
        for (size_t i = 0; i < half_dim; ++i) {
            const float exponent = static_cast<float>(2 * i) / static_cast<float>(hidden_dim);
            const float inv_freq = 1.0f / std::pow(10000.0f, exponent);
            const float angle = static_cast<float>(rel_pos) * inv_freq;
            pos_embed[p * hidden_dim + 2 * i] = std::sin(angle);
            if (2 * i + 1 < hidden_dim) {
                pos_embed[p * hidden_dim + 2 * i + 1] = std::cos(angle);
            }
        }
    }

    std::vector<__fp16> pos_embed_f16(pos_embed.size());
    cactus_fp32_to_fp16(pos_embed.data(), pos_embed_f16.data(), pos_embed.size());

    size_t pos_node = gb->input({rel_len, hidden_dim}, Precision::FP16);
    gb->set_input(pos_node, pos_embed_f16.data(), Precision::FP16);
    return pos_node;
}

size_t SortformerDiarModel::build_fc_self_attention(CactusGraph* gb, size_t hidden, size_t position_embeddings,
                                                    uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = weight_nodes_.fc_layers[layer_idx];

    size_t q = gb->matmul(hidden, layer.self_attn_q_weight, true, backend);
    q = gb->add(q, layer.self_attn_q_bias);
    size_t k = gb->matmul(hidden, layer.self_attn_k_weight, true, backend);
    k = gb->add(k, layer.self_attn_k_bias);
    size_t v = gb->matmul(hidden, layer.self_attn_v_weight, true, backend);
    v = gb->add(v, layer.self_attn_v_bias);

    const auto& q_shape = gb->get_output_buffer(q).shape;
    if (q_shape.size() != 2) {
        throw std::runtime_error("Sortformer FC self-attention expects [T, D]");
    }

    const size_t t = q_shape[0];
    const size_t q_heads = std::max<size_t>(1, static_cast<size_t>(config_.attention_heads));
    const size_t kv_heads = std::max<size_t>(1, static_cast<size_t>(config_.attention_kv_heads));
    const size_t head_dim = std::max<size_t>(1, static_cast<size_t>(config_.attention_head_dim));

    size_t q4 = gb->reshape(q, {1, t, q_heads, head_dim});
    size_t k4 = gb->reshape(k, {1, t, kv_heads, head_dim});
    size_t v4 = gb->reshape(v, {1, t, kv_heads, head_dim});

    size_t bias_u = gb->reshape(layer.self_attn_bias_u, {1, static_cast<size_t>(1), q_heads, head_dim});
    size_t bias_v = gb->reshape(layer.self_attn_bias_v, {1, static_cast<size_t>(1), q_heads, head_dim});
    size_t q_u4 = gb->add(q4, bias_u);
    size_t q_v4 = gb->add(q4, bias_v);

    size_t rel_k_flat = gb->matmul(position_embeddings, layer.self_attn_relative_k_weight, true, backend);
    size_t rel_k4 = gb->reshape(rel_k_flat, {1, 2 * t - 1, q_heads, head_dim});
    size_t rel_bias = gb->rel_pos_bias(q_v4, rel_k4, attention_scale_);

    size_t attn = gb->attention_masked(q_u4, k4, v4, rel_bias, attention_scale_, false, backend, true);
    attn = gb->reshape(attn, {t, q_heads * head_dim});

    size_t out = gb->matmul(attn, layer.self_attn_output_weight, true, backend);
    out = gb->add(out, layer.self_attn_output_bias);
    return out;
}

size_t SortformerDiarModel::build_fc_feed_forward(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                                  bool second_ff, ComputeBackend backend) {
    const auto& layer = weight_nodes_.fc_layers[layer_idx];
    const size_t w1 = second_ff ? layer.ff2_linear1_weight : layer.ff1_linear1_weight;
    const size_t b1 = second_ff ? layer.ff2_linear1_bias : layer.ff1_linear1_bias;
    const size_t w2 = second_ff ? layer.ff2_linear2_weight : layer.ff1_linear2_weight;
    const size_t b2 = second_ff ? layer.ff2_linear2_bias : layer.ff1_linear2_bias;

    size_t x = gb->matmul(hidden, w1, true, backend);
    x = gb->add(x, b1);

    std::string act = config_.encoder_hidden_act;
    std::transform(act.begin(), act.end(), act.begin(), ::tolower);
    if (act.find("gelu") != std::string::npos) {
        x = gb->gelu(x);
    } else if (act == "relu") {
        x = gb->relu(x);
    } else {
        x = gb->silu(x);
    }

    x = gb->matmul(x, w2, true, backend);
    x = gb->add(x, b2);
    return x;
}

size_t SortformerDiarModel::build_fc_convolution_module(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                                        ComputeBackend backend) {
    (void)backend;
    const auto& layer = weight_nodes_.fc_layers[layer_idx];
    const auto& hidden_shape = gb->get_output_buffer(hidden).shape;
    if (hidden_shape.size() != 2) {
        throw std::runtime_error("Sortformer FC convolution module expects [T, D]");
    }

    const size_t t = hidden_shape[0];
    const size_t d = hidden_shape[1];

    size_t x = gb->reshape(hidden, {1, t, d});
    x = gb->conv1d_pointwise(x, layer.conv_pointwise1_weight, layer.conv_pointwise1_bias);
    x = gb->glu(x, -1);
    x = gb->conv1d_same_depthwise_k9(x, layer.conv_depthwise_weight, layer.conv_depthwise_bias);
    x = gb->batchnorm(
        x,
        layer.conv_batchnorm_weight,
        layer.conv_batchnorm_bias,
        layer.conv_batchnorm_running_mean,
        layer.conv_batchnorm_running_var,
        2,
        1e-5f
    );

    std::string act = config_.encoder_hidden_act;
    std::transform(act.begin(), act.end(), act.begin(), ::tolower);
    if (act.find("gelu") != std::string::npos) {
        x = gb->gelu(x);
    } else if (act == "relu") {
        x = gb->relu(x);
    } else {
        x = gb->silu(x);
    }

    x = gb->conv1d_pointwise(x, layer.conv_pointwise2_weight, layer.conv_pointwise2_bias);
    x = gb->reshape(x, {t, d});
    return x;
}

size_t SortformerDiarModel::build_fc_encoder_block(CactusGraph* gb, size_t hidden, size_t position_embeddings,
                                                   uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = weight_nodes_.fc_layers[layer_idx];

    size_t ff1_in = gb->layernorm(hidden, layer.norm_ff1_weight, layer.norm_ff1_bias);
    size_t ff1 = build_fc_feed_forward(gb, ff1_in, layer_idx, false, backend);
    ff1 = gb->scalar_multiply(ff1, 0.5f);
    size_t x = gb->add(hidden, ff1);

    size_t attn_in = gb->layernorm(x, layer.norm_self_attn_weight, layer.norm_self_attn_bias);
    size_t attn = build_fc_self_attention(gb, attn_in, position_embeddings, layer_idx, backend);
    x = gb->add(x, attn);

    size_t conv_in = gb->layernorm(x, layer.norm_conv_weight, layer.norm_conv_bias);
    size_t conv = build_fc_convolution_module(gb, conv_in, layer_idx, backend);
    x = gb->add(x, conv);

    size_t ff2_in = gb->layernorm(x, layer.norm_ff2_weight, layer.norm_ff2_bias);
    size_t ff2 = build_fc_feed_forward(gb, ff2_in, layer_idx, true, backend);
    ff2 = gb->scalar_multiply(ff2, 0.5f);
    x = gb->add(x, ff2);

    x = gb->layernorm(x, layer.norm_out_weight, layer.norm_out_bias);
    return x;
}

size_t SortformerDiarModel::build_fc_encoder(CactusGraph* gb, const std::vector<float>& audio_features) {
    size_t hidden = build_subsampling(gb, audio_features);
    const auto& shape = gb->get_output_buffer(hidden).shape;
    if (shape.size() != 2) {
        throw std::runtime_error("Sortformer FC encoder expects subsampling output [T, D]");
    }

    size_t position_embeddings = build_relative_position_embeddings(gb, shape[0]);
    for (uint32_t i = 0; i < config_.num_layers; ++i) {
        hidden = build_fc_encoder_block(gb, hidden, position_embeddings, i, ComputeBackend::CPU);
    }
    return hidden;
}

size_t SortformerDiarModel::build_tf_encoder_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                                   ComputeBackend backend) {
    const auto& layer = weight_nodes_.tf_layers[layer_idx];
    const auto& shape = gb->get_output_buffer(hidden).shape;
    if (shape.size() != 2) {
        throw std::runtime_error("Sortformer TF encoder block expects [T, D]");
    }

    const size_t t = shape[0];
    const size_t d = shape[1];
    const size_t heads = std::max<size_t>(1, static_cast<size_t>(config_.diar_tf_attention_heads));
    if (d % heads != 0) {
        throw std::runtime_error("Sortformer TF hidden dim must be divisible by attention heads");
    }
    const size_t head_dim = d / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t q = gb->matmul(hidden, layer.self_attn_q_weight, true, backend);
    q = gb->add(q, layer.self_attn_q_bias);
    size_t k = gb->matmul(hidden, layer.self_attn_k_weight, true, backend);
    if (layer.self_attn_k_bias != 0) {
        k = gb->add(k, layer.self_attn_k_bias);
    }
    size_t v = gb->matmul(hidden, layer.self_attn_v_weight, true, backend);
    v = gb->add(v, layer.self_attn_v_bias);

    size_t q4 = gb->reshape(q, {1, t, heads, head_dim});
    size_t k4 = gb->reshape(k, {1, t, heads, head_dim});
    size_t v4 = gb->reshape(v, {1, t, heads, head_dim});
    size_t attn = gb->attention(q4, k4, v4, scale, false, backend);
    attn = gb->reshape(attn, {t, d});
    attn = gb->matmul(attn, layer.self_attn_output_weight, true, backend);
    attn = gb->add(attn, layer.self_attn_output_bias);

    size_t x = gb->add(hidden, attn);
    x = gb->layernorm(x, layer.self_attn_layernorm_weight, layer.self_attn_layernorm_bias);

    size_t ff = gb->matmul(x, layer.ff1_weight, true, backend);
    ff = gb->add(ff, layer.ff1_bias);
    ff = gb->relu(ff);
    ff = gb->matmul(ff, layer.ff2_weight, true, backend);
    ff = gb->add(ff, layer.ff2_bias);

    x = gb->add(x, ff);
    x = gb->layernorm(x, layer.final_layernorm_weight, layer.final_layernorm_bias);
    return x;
}

size_t SortformerDiarModel::build_tf_encoder(CactusGraph* gb, size_t fc_hidden) {
    size_t hidden = fc_hidden;
    const auto& fc_shape = gb->get_output_buffer(fc_hidden).shape;
    if (fc_shape.size() != 2) {
        throw std::runtime_error("Sortformer TF encoder expects FC encoder output [T, D]");
    }
    const size_t t = fc_shape[0];

    if (weight_nodes_.encoder_proj_weight != 0) {
        hidden = gb->matmul(hidden, weight_nodes_.encoder_proj_weight, true, ComputeBackend::CPU);
        if (weight_nodes_.encoder_proj_bias != 0) {
            hidden = gb->add(hidden, weight_nodes_.encoder_proj_bias);
        }
    }

    const auto& tf_shape = gb->get_output_buffer(hidden).shape;
    if (tf_shape.size() != 2) {
        throw std::runtime_error("Sortformer TF projection must produce rank-2 [T, D]");
    }

    if (weight_nodes_.tf_embed_positions != 0) {
        const auto& pe_shape = gb->get_output_buffer(weight_nodes_.tf_embed_positions).shape;
        if (pe_shape.size() == 2 && pe_shape[0] >= t && pe_shape[1] == tf_shape[1]) {
            size_t pos = gb->slice(weight_nodes_.tf_embed_positions, 0, 0, t);
            hidden = gb->add(hidden, pos);
        }
    }

    for (uint32_t i = 0; i < config_.diar_tf_num_layers; ++i) {
        hidden = build_tf_encoder_block(gb, hidden, i, ComputeBackend::CPU);
    }
    return hidden;
}

size_t SortformerDiarModel::build_speaker_probs(CactusGraph* gb, size_t hidden) {
    size_t x = gb->relu(hidden);
    x = gb->matmul(x, weight_nodes_.head_hidden_weight, true, ComputeBackend::CPU);
    x = gb->add(x, weight_nodes_.head_hidden_bias);
    x = gb->relu(x);

    size_t logits = gb->matmul(x, weight_nodes_.head_single_spk_weight, true, ComputeBackend::CPU);
    logits = gb->add(logits, weight_nodes_.head_single_spk_bias);
    return gb->sigmoid(logits);
}

size_t SortformerDiarModel::forward(const std::vector<float>& audio_features, const std::vector<uint32_t>& tokens,
                                    bool use_cache) {
    (void)tokens;
    (void)use_cache;
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->clear_debug_nodes();
    size_t fc_hidden = build_fc_encoder(gb, audio_features);
    size_t tf_hidden = build_tf_encoder(gb, fc_hidden);
    return build_speaker_probs(gb, tf_hidden);
}

std::vector<float> SortformerDiarModel::extract_probabilities(
    CactusGraph* gb,
    size_t probs_node,
    size_t* out_frames,
    size_t* out_speakers) const {
    const auto& probs_buf = gb->get_output_buffer(probs_node);
    if (probs_buf.shape.size() != 2) {
        throw std::runtime_error("Sortformer speaker probabilities must be [T, S]");
    }

    const size_t t = probs_buf.shape[0];
    const size_t s = probs_buf.shape[1];
    if (out_frames) {
        *out_frames = t;
    }
    if (out_speakers) {
        *out_speakers = s;
    }

    std::vector<float> probs(t * s, 0.0f);
    if (t == 0 || s == 0) {
        return probs;
    }

    if (probs_buf.precision == Precision::FP32) {
        const float* src = probs_buf.data_as<float>();
        std::copy(src, src + probs.size(), probs.begin());
    } else if (probs_buf.precision == Precision::FP16) {
        const __fp16* src = probs_buf.data_as<__fp16>();
        Quantization::fp16_to_fp32(src, probs.data(), probs.size());
    } else {
        const int8_t* src = probs_buf.data_as<int8_t>();
        Quantization::int8_to_fp32(src, probs.data(), probs.size(), 1.0f);
    }
    return probs;
}

std::vector<float> SortformerDiarModel::get_speaker_activity(const std::vector<float>& audio_features,
                                                             size_t* out_frames,
                                                             size_t* out_speakers) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Sortformer model not initialized - call init() first");
    }
    if (audio_features.empty()) {
        throw std::runtime_error("Audio features cannot be empty in Sortformer get_speaker_activity");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    size_t probs_node = forward(audio_features, {}, false);
    gb->execute();
    std::vector<float> probs = extract_probabilities(gb, probs_node, out_frames, out_speakers);
    reset_cache();
    return probs;
}

std::vector<SortformerDiarModel::SpeakerTimestamp> SortformerDiarModel::get_speaker_timestamps(
    const std::vector<float>& audio_features) {
    return get_speaker_timestamps(audio_features, SpeakerTimestampsOptions{});
}

std::vector<SortformerDiarModel::SpeakerTimestamp> SortformerDiarModel::get_speaker_timestamps(
    const std::vector<float>& audio_features,
    const SpeakerTimestampsOptions& options) {
    size_t t = 0;
    size_t s = 0;
    std::vector<float> probs = get_speaker_activity(audio_features, &t, &s);
    std::vector<SpeakerTimestamp> timestamps;
    if (t == 0 || s == 0) {
        return timestamps;
    }

    const float frame_step = config_.diar_frame_step_seconds > 0.0f ? config_.diar_frame_step_seconds : 0.08f;
    const float threshold = std::max(0.0f, options.threshold);
    const size_t min_speech_frames = std::max<size_t>(
        1, static_cast<size_t>(std::round(static_cast<float>(options.min_speech_duration_ms) / (frame_step * 1000.0f))));
    const size_t min_silence_frames = std::max<size_t>(
        0, static_cast<size_t>(std::round(static_cast<float>(options.min_silence_duration_ms) / (frame_step * 1000.0f))));

    timestamps.reserve(t);
    for (size_t spk = 0; spk < s; ++spk) {
        std::vector<std::pair<size_t, size_t>> raw;
        raw.reserve(t / 2 + 1);

        bool active = false;
        size_t active_start = 0;
        for (size_t ti = 0; ti < t; ++ti) {
            const bool on = probs[ti * s + spk] >= threshold;
            if (on && !active) {
                active = true;
                active_start = ti;
            } else if (!on && active) {
                raw.emplace_back(active_start, ti);
                active = false;
            }
        }
        if (active) {
            raw.emplace_back(active_start, t);
        }

        if (raw.empty()) {
            continue;
        }

        std::vector<std::pair<size_t, size_t>> merged;
        merged.reserve(raw.size());
        for (const auto& seg : raw) {
            if (merged.empty()) {
                merged.push_back(seg);
                continue;
            }
            auto& last = merged.back();
            if (seg.first <= last.second + min_silence_frames) {
                if (seg.second > last.second) {
                    last.second = seg.second;
                }
            } else {
                merged.push_back(seg);
            }
        }

        for (const auto& seg : merged) {
            if (seg.second <= seg.first) {
                continue;
            }
            const size_t dur_frames = seg.second - seg.first;
            if (dur_frames < min_speech_frames) {
                continue;
            }
            timestamps.push_back(SpeakerTimestamp{
                static_cast<float>(seg.first) * frame_step,
                static_cast<float>(seg.second) * frame_step,
                static_cast<uint32_t>(spk),
            });
        }
    }

    std::sort(timestamps.begin(), timestamps.end(), [](const SpeakerTimestamp& a, const SpeakerTimestamp& b) {
        if (a.start == b.start) {
            if (a.end == b.end) {
                return a.speaker < b.speaker;
            }
            return a.end < b.end;
        }
        return a.start < b.start;
    });
    return timestamps;
}

void SortformerDiarModel::set_diarization_threshold(float threshold) {
    diarization_threshold_override_ = std::max(0.0f, threshold);
}

void SortformerDiarModel::clear_diarization_threshold_override() {
    diarization_threshold_override_ = -1.0f;
}

void SortformerDiarModel::extract_speaker_segments(CactusGraph* gb, size_t probs_node) {
    speaker_tokens_.clear();
    speaker_token_starts_.clear();
    speaker_token_ends_.clear();

    size_t t = 0;
    size_t s = 0;
    std::vector<float> probs = extract_probabilities(gb, probs_node, &t, &s);
    if (t == 0 || s == 0) {
        return;
    }

    const float frame_step = config_.diar_frame_step_seconds > 0.0f ? config_.diar_frame_step_seconds : 0.08f;
    const float sil_th = std::max(
        0.0f,
        diarization_threshold_override_ >= 0.0f ? diarization_threshold_override_ : config_.diar_sil_threshold
    );

    struct Seg {
        uint32_t speaker;
        float start_sec;
        float end_sec;
    };
    std::vector<Seg> segs;
    segs.reserve(t);

    for (size_t spk = 0; spk < s; ++spk) {
        bool active = false;
        size_t active_start = 0;
        for (size_t ti = 0; ti < t; ++ti) {
            const float p = probs[ti * s + spk];
            const bool on = p >= sil_th;
            if (on && !active) {
                active = true;
                active_start = ti;
            } else if (!on && active) {
                segs.push_back(Seg{
                    static_cast<uint32_t>(spk),
                    static_cast<float>(active_start) * frame_step,
                    static_cast<float>(ti) * frame_step
                });
                active = false;
            }
        }
        if (active) {
            segs.push_back(Seg{
                static_cast<uint32_t>(spk),
                static_cast<float>(active_start) * frame_step,
                static_cast<float>(t) * frame_step
            });
        }
    }

    std::sort(segs.begin(), segs.end(), [](const Seg& a, const Seg& b) {
        if (a.start_sec == b.start_sec) {
            if (a.end_sec == b.end_sec) {
                return a.speaker < b.speaker;
            }
            return a.end_sec < b.end_sec;
        }
        return a.start_sec < b.start_sec;
    });

    for (const auto& seg : segs) {
        if (seg.end_sec <= seg.start_sec) {
            continue;
        }
        speaker_tokens_.push_back(seg.speaker);
        speaker_token_starts_.push_back(seg.start_sec);
        speaker_token_ends_.push_back(seg.end_sec);
    }
}

uint32_t SortformerDiarModel::decode_with_audio(const std::vector<uint32_t>& tokens,
                                                const std::vector<float>& audio_features,
                                                float temperature,
                                                float top_p,
                                                size_t top_k,
                                                const std::string& profile_file,
                                                float* out_entropy,
                                                float* out_token_time_start,
                                                float* out_token_time_end) {
    (void)temperature;
    (void)top_p;
    (void)top_k;

    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Sortformer model not initialized - call init() first");
    }
    if (audio_features.empty()) {
        throw std::runtime_error("Audio features cannot be empty in Sortformer decode_with_audio");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    const bool new_request = !speaker_segments_ready_ || tokens.size() < last_input_token_count_;
    if (new_request) {
        gb->soft_reset();
        size_t probs_node = forward(audio_features, tokens, false);
        gb->execute(profile_file);
        extract_speaker_segments(gb, probs_node);
        speaker_emit_index_ = 0;
        speaker_segments_ready_ = true;
    }

    last_input_token_count_ = tokens.size();
    if (out_entropy) {
        *out_entropy = 0.0f;
    }

    if (speaker_emit_index_ < speaker_tokens_.size()) {
        if (out_token_time_start) {
            *out_token_time_start = speaker_token_starts_[speaker_emit_index_];
        }
        if (out_token_time_end) {
            *out_token_time_end = speaker_token_ends_[speaker_emit_index_];
        }
        return speaker_tokens_[speaker_emit_index_++];
    }

    if (out_token_time_start) {
        *out_token_time_start = 0.0f;
    }
    if (out_token_time_end) {
        *out_token_time_end = 0.0f;
    }
    if (get_tokenizer()) {
        return get_tokenizer()->get_eos_token();
    }
    return 0;
}

std::vector<float> SortformerDiarModel::get_audio_embeddings(const std::vector<float>& audio_features) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    size_t fc_hidden = build_fc_encoder(gb, audio_features);
    size_t tf_hidden = build_tf_encoder(gb, fc_hidden);
    size_t pooled = gb->mean(tf_hidden, 0);
    gb->execute();

    const auto& output_buf = gb->get_output_buffer(pooled);
    const size_t dim = output_buf.total_size;
    std::vector<float> embedding(dim, 0.0f);

    if (output_buf.precision == Precision::FP32) {
        const float* src = output_buf.data_as<float>();
        std::copy(src, src + dim, embedding.begin());
    } else if (output_buf.precision == Precision::FP16) {
        const __fp16* src = output_buf.data_as<__fp16>();
        Quantization::fp16_to_fp32(src, embedding.data(), dim);
    } else {
        const int8_t* src = output_buf.data_as<int8_t>();
        Quantization::int8_to_fp32(src, embedding.data(), dim, 1.0f);
    }

    reset_cache();
    return embedding;
}

void SortformerDiarModel::reset_cache() {
    Model::reset_cache();
    speaker_segments_ready_ = false;
    speaker_emit_index_ = 0;
    speaker_tokens_.clear();
    speaker_token_starts_.clear();
    speaker_token_ends_.clear();
    last_input_token_count_ = 0;
}

}  // namespace engine
}  // namespace cactus
