#include "model.h"
#include "../graph/graph.h"
#include "../npu/npu.h"
#include "../kernel/kernel.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

size_t shape_elements(const std::vector<int>& shape) {
    if (shape.empty()) return 0;
    size_t total = 1;
    for (int d : shape) {
        if (d <= 0) return 0;
        total *= static_cast<size_t>(d);
    }
    return total;
}

bool pack_parakeet_features_for_npu(
    const std::vector<__fp16>& time_major_f16,
    size_t frames,
    size_t num_mels,
    const std::vector<int>& input_shape,
    std::vector<__fp16>& packed)
{
    if (input_shape.empty()) return false;
    const size_t total = shape_elements(input_shape);
    if (total == 0) return false;
    packed.assign(total, static_cast<__fp16>(0.0f));

    auto tm = [&](size_t t, size_t m) -> __fp16 {
        return time_major_f16[t * num_mels + m];
    };

    if (input_shape.size() == 4) {
        const size_t s0 = static_cast<size_t>(input_shape[0]);
        const size_t s1 = static_cast<size_t>(input_shape[1]);
        const size_t s2 = static_cast<size_t>(input_shape[2]);
        const size_t s3 = static_cast<size_t>(input_shape[3]);
        if (s0 != 1) return false;

        if (s1 == 1 && s2 >= frames && s3 == num_mels) {
            for (size_t t = 0; t < frames; ++t) {
                for (size_t m = 0; m < num_mels; ++m) {
                    packed[(t * s3) + m] = tm(t, m);
                }
            }
            return true;
        }
        if (s1 >= frames && s2 == num_mels && s3 == 1) {
            for (size_t t = 0; t < frames; ++t) {
                for (size_t m = 0; m < num_mels; ++m) {
                    packed[((t * s2 + m) * s3)] = tm(t, m);
                }
            }
            return true;
        }
        if (s1 == num_mels && s2 >= frames && s3 == 1) {
            for (size_t t = 0; t < frames; ++t) {
                for (size_t m = 0; m < num_mels; ++m) {
                    packed[((m * s2 + t) * s3)] = tm(t, m);
                }
            }
            return true;
        }
        if (s1 == 1 && s2 == num_mels && s3 >= frames) {
            for (size_t t = 0; t < frames; ++t) {
                for (size_t m = 0; m < num_mels; ++m) {
                    packed[(m * s3) + t] = tm(t, m);
                }
            }
            return true;
        }
        return false;
    }

    if (input_shape.size() == 3) {
        const size_t s0 = static_cast<size_t>(input_shape[0]);
        const size_t s1 = static_cast<size_t>(input_shape[1]);
        const size_t s2 = static_cast<size_t>(input_shape[2]);

        if (s0 == 1 && s1 >= frames && s2 == num_mels) {
            for (size_t t = 0; t < frames; ++t) {
                for (size_t m = 0; m < num_mels; ++m) {
                    packed[t * s2 + m] = tm(t, m);
                }
            }
            return true;
        }
        if (s0 == 1 && s1 == num_mels && s2 >= frames) {
            for (size_t t = 0; t < frames; ++t) {
                for (size_t m = 0; m < num_mels; ++m) {
                    packed[m * s2 + t] = tm(t, m);
                }
            }
            return true;
        }
        if (s0 >= frames && s1 == num_mels && s2 == 1) {
            for (size_t t = 0; t < frames; ++t) {
                for (size_t m = 0; m < num_mels; ++m) {
                    packed[(t * s1 + m) * s2] = tm(t, m);
                }
            }
            return true;
        }
        if (s0 == num_mels && s1 >= frames && s2 == 1) {
            for (size_t t = 0; t < frames; ++t) {
                for (size_t m = 0; m < num_mels; ++m) {
                    packed[(m * s1 + t) * s2] = tm(t, m);
                }
            }
            return true;
        }
        return false;
    }

    if (input_shape.size() == 2) {
        const size_t s0 = static_cast<size_t>(input_shape[0]);
        const size_t s1 = static_cast<size_t>(input_shape[1]);

        if (s0 >= frames && s1 == num_mels) {
            for (size_t t = 0; t < frames; ++t) {
                for (size_t m = 0; m < num_mels; ++m) {
                    packed[t * s1 + m] = tm(t, m);
                }
            }
            return true;
        }
        if (s0 == num_mels && s1 >= frames) {
            for (size_t t = 0; t < frames; ++t) {
                for (size_t m = 0; m < num_mels; ++m) {
                    packed[m * s1 + t] = tm(t, m);
                }
            }
            return true;
        }
        return false;
    }

    return false;
}

bool infer_npu_encoder_output_shape(
    const std::vector<int>& output_shape,
    size_t elements_written,
    size_t fallback_hidden_dim,
    size_t& time_steps,
    size_t& hidden_dim)
{
    if (elements_written == 0) return false;

    std::vector<size_t> dims;
    dims.reserve(output_shape.size());
    for (int d : output_shape) {
        if (d > 0) dims.push_back(static_cast<size_t>(d));
    }

    if (dims.size() >= 2) {
        time_steps = dims[dims.size() - 2];
        hidden_dim = dims[dims.size() - 1];
    } else if (dims.size() == 1) {
        hidden_dim = dims[0];
        if (hidden_dim == 0 || (elements_written % hidden_dim) != 0) return false;
        time_steps = elements_written / hidden_dim;
    } else {
        hidden_dim = fallback_hidden_dim;
        if (hidden_dim == 0 || (elements_written % hidden_dim) != 0) return false;
        time_steps = elements_written / hidden_dim;
    }

    if (time_steps == 0 || hidden_dim == 0) return false;
    if (time_steps * hidden_dim > elements_written) return false;
    return true;
}

} // namespace

namespace cactus {
namespace engine {

ParakeetModel::ParakeetModel() : Model() {}

ParakeetModel::ParakeetModel(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);

    float hd = static_cast<float>(config.attention_head_dim);
    if (hd <= 0.0f) {
        hd = 64.0f;
    }
    attention_scale_ = 1.0f / std::sqrt(hd);

    // Legacy converted Parakeet configs inherited generic rope_theta defaults.
    // Keep RoPE disabled unless explicitly set to a non-default value.
    if (std::fabs(config_.rope_theta - 10000.0f) < 1e-3f ||
        std::fabs(config_.rope_theta - 1000000.0f) < 1e-3f) {
        config_.rope_theta = 0.0f;
    }
}

void ParakeetModel::load_weights_to_graph(CactusGraph* gb) {
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

    weight_nodes_.ctc_head_weight = gb->mmap_weights(model_folder_path_ + "/ctc_head_weight.weights");
    weight_nodes_.ctc_head_bias = gb->mmap_weights(model_folder_path_ + "/ctc_head_bias.bias");

    if (npu::is_npu_available()) {
        std::string npu_encoder_path = model_folder_path_ + "/model.mlpackage";
        npu_encoder_ = npu::create_encoder();
        if (npu_encoder_ && npu_encoder_->load(npu_encoder_path)) {
            use_npu_encoder_ = true;
            std::vector<int> input_shape = npu_encoder_->get_input_shape();
            if (!input_shape.empty()) {
                npu_encoder_->preallocate(input_shape, "x", "");
            }
        } else {
            use_npu_encoder_ = false;
            npu_encoder_.reset();
        }
    }

    for (uint32_t i = 0; i < config_.num_layers; ++i) {
        auto& layer = weight_nodes_.layers[i];
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
}

size_t ParakeetModel::build_subsampling(CactusGraph* gb, const std::vector<float>& audio_features) {
    const size_t num_mels = std::max<size_t>(1, static_cast<size_t>(config_.num_mel_bins));
    if (audio_features.empty() || (audio_features.size() % num_mels) != 0) {
        throw std::runtime_error("Parakeet expects audio_features with shape [num_mels, num_frames]");
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
        throw std::runtime_error("Parakeet subsampling produced invalid shape");
    }

    const size_t C = conv_shape[1];
    const size_t T = conv_shape[2];
    const size_t W = conv_shape[3];

    size_t t_major = gb->transposeN(x, {0, 2, 1, 3}, ComputeBackend::CPU);
    size_t flattened = gb->reshape(t_major, {T, C * W});
    size_t projected = gb->matmul(flattened, weight_nodes_.subsampling_linear_weight, true, ComputeBackend::CPU);
    projected = gb->add(projected, weight_nodes_.subsampling_linear_bias);
    return projected;
}

size_t ParakeetModel::build_relative_position_embeddings(CactusGraph* gb, size_t seq_len) {
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

size_t ParakeetModel::build_self_attention(CactusGraph* gb, size_t hidden, size_t position_embeddings,
                                           uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = weight_nodes_.layers[layer_idx];

    size_t q = gb->matmul(hidden, layer.self_attn_q_weight, true, backend);
    q = gb->add(q, layer.self_attn_q_bias);
    size_t k = gb->matmul(hidden, layer.self_attn_k_weight, true, backend);
    k = gb->add(k, layer.self_attn_k_bias);
    size_t v = gb->matmul(hidden, layer.self_attn_v_weight, true, backend);
    v = gb->add(v, layer.self_attn_v_bias);

    const auto& q_shape = gb->get_output_buffer(q).shape;
    if (q_shape.size() != 2) {
        throw std::runtime_error("Parakeet self-attention expects [T, D]");
    }

    const size_t T = q_shape[0];
    const size_t q_heads = std::max<size_t>(1, static_cast<size_t>(config_.attention_heads));
    const size_t kv_heads = std::max<size_t>(1, static_cast<size_t>(config_.attention_kv_heads));
    const size_t head_dim = std::max<size_t>(1, static_cast<size_t>(config_.attention_head_dim));

    size_t q4 = gb->reshape(q, {1, T, q_heads, head_dim});
    size_t k4 = gb->reshape(k, {1, T, kv_heads, head_dim});
    size_t v4 = gb->reshape(v, {1, T, kv_heads, head_dim});

    size_t bias_u = gb->reshape(layer.self_attn_bias_u, {1, static_cast<size_t>(1), q_heads, head_dim});
    size_t bias_v = gb->reshape(layer.self_attn_bias_v, {1, static_cast<size_t>(1), q_heads, head_dim});
    size_t q_u4 = gb->add(q4, bias_u);
    size_t q_v4 = gb->add(q4, bias_v);

    size_t rel_k_flat = gb->matmul(position_embeddings, layer.self_attn_relative_k_weight, true, backend);
    size_t rel_k4 = gb->reshape(rel_k_flat, {1, 2 * T - 1, q_heads, head_dim});
    size_t rel_bias = gb->rel_pos_bias(q_v4, rel_k4, attention_scale_);

    size_t attn = gb->attention_masked(q_u4, k4, v4, rel_bias, attention_scale_, false, backend, true);
    attn = gb->reshape(attn, {T, q_heads * head_dim});

    size_t out = gb->matmul(attn, layer.self_attn_output_weight, true, backend);
    out = gb->add(out, layer.self_attn_output_bias);
    return out;
}

size_t ParakeetModel::build_feed_forward(CactusGraph* gb, size_t hidden, uint32_t layer_idx, bool second_ff, ComputeBackend backend) {
    const auto& layer = weight_nodes_.layers[layer_idx];

    size_t w1 = second_ff ? layer.ff2_linear1_weight : layer.ff1_linear1_weight;
    size_t b1 = second_ff ? layer.ff2_linear1_bias : layer.ff1_linear1_bias;
    size_t w2 = second_ff ? layer.ff2_linear2_weight : layer.ff1_linear2_weight;
    size_t b2 = second_ff ? layer.ff2_linear2_bias : layer.ff1_linear2_bias;

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

size_t ParakeetModel::build_convolution_module(CactusGraph* gb, size_t hidden, uint32_t layer_idx, ComputeBackend backend) {
    (void)backend;
    const auto& layer = weight_nodes_.layers[layer_idx];
    const auto& hidden_shape = gb->get_output_buffer(hidden).shape;
    if (hidden_shape.size() != 2) {
        throw std::runtime_error("Parakeet convolution module expects [T, D]");
    }

    const size_t T = hidden_shape[0];
    const size_t D = hidden_shape[1];

    size_t x = gb->reshape(hidden, {1, T, D});
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

    x = gb->reshape(x, {T, D});
    return x;
}

size_t ParakeetModel::build_encoder_block(CactusGraph* gb, size_t hidden, size_t position_embeddings,
                                          uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = weight_nodes_.layers[layer_idx];

    size_t ff1_in = gb->layernorm(hidden, layer.norm_ff1_weight, layer.norm_ff1_bias);
    size_t ff1 = build_feed_forward(gb, ff1_in, layer_idx, false, backend);
    ff1 = gb->scalar_multiply(ff1, 0.5f);
    size_t x = gb->add(hidden, ff1);

    size_t attn_in = gb->layernorm(x, layer.norm_self_attn_weight, layer.norm_self_attn_bias);
    size_t attn = build_self_attention(gb, attn_in, position_embeddings, layer_idx, backend);
    x = gb->add(x, attn);

    size_t conv_in = gb->layernorm(x, layer.norm_conv_weight, layer.norm_conv_bias);
    size_t conv = build_convolution_module(gb, conv_in, layer_idx, backend);
    x = gb->add(x, conv);

    size_t ff2_in = gb->layernorm(x, layer.norm_ff2_weight, layer.norm_ff2_bias);
    size_t ff2 = build_feed_forward(gb, ff2_in, layer_idx, true, backend);
    ff2 = gb->scalar_multiply(ff2, 0.5f);
    x = gb->add(x, ff2);

    x = gb->layernorm(x, layer.norm_out_weight, layer.norm_out_bias);
    return x;
}

size_t ParakeetModel::build_encoder(CactusGraph* gb, const std::vector<float>& audio_features) {
    const size_t num_mels = std::max<size_t>(1, static_cast<size_t>(config_.num_mel_bins));
    if (audio_features.empty() || (audio_features.size() % num_mels) != 0) {
        throw std::runtime_error("Parakeet expects audio_features with shape [num_mels, num_frames]");
    }
    const size_t frames = audio_features.size() / num_mels;
    size_t expected_hidden_dim = std::max<size_t>(1, static_cast<size_t>(config_.hidden_dim));
    size_t expected_vocab_size = std::max<size_t>(1, static_cast<size_t>(config_.vocab_size));
    {
        const auto& ctc_w_shape = gb->get_output_buffer(weight_nodes_.ctc_head_weight).shape;
        if (ctc_w_shape.size() == 2) {
            expected_vocab_size = ctc_w_shape[0];
            expected_hidden_dim = ctc_w_shape[1];
        } else if (ctc_w_shape.size() == 3) {
            expected_vocab_size = ctc_w_shape[0];
            expected_hidden_dim = ctc_w_shape[1];
        }
    }

    if (use_npu_encoder_ && npu_encoder_ && npu_encoder_->is_available()) {
        std::vector<float> time_major(frames * num_mels);
        for (size_t m = 0; m < num_mels; ++m) {
            const float* src = &audio_features[m * frames];
            for (size_t t = 0; t < frames; ++t) {
                time_major[t * num_mels + m] = src[t];
            }
        }

        std::vector<__fp16> time_major_f16(time_major.size());
        cactus_fp32_to_fp16(time_major.data(), time_major_f16.data(), time_major.size());

        std::vector<int> input_shape = npu_encoder_->get_input_shape();
        std::vector<__fp16> npu_input;
        if (pack_parakeet_features_for_npu(time_major_f16, frames, num_mels, input_shape, npu_input)) {
            npu_encoder_->preallocate(input_shape, "x", "");

            std::vector<int> output_shape = npu_encoder_->get_output_shape();
            size_t output_capacity = npu_encoder_->get_output_buffer_size();
            if (output_capacity == 0) {
                output_capacity = shape_elements(output_shape);
            }
            if (output_capacity == 0) {
                output_capacity = std::max<size_t>(
                    1, frames * std::max<size_t>(1, static_cast<size_t>(config_.hidden_dim)));
            }

            std::vector<__fp16> npu_output(output_capacity);
            const size_t elements_written = npu_encoder_->encode(
                npu_input.data(),
                npu_output.data(),
                input_shape,
                "x",
                ""
            );

            size_t T_enc = 0;
            size_t D_enc = 0;
            if (elements_written > 0 &&
                infer_npu_encoder_output_shape(output_shape, elements_written,
                                               static_cast<size_t>(config_.hidden_dim), T_enc, D_enc)) {
                const __fp16* src = npu_output.data();
                __fp16* cached_output = npu_encoder_->get_output_buffer();
                const size_t cached_count = npu_encoder_->get_output_buffer_size();
                const size_t required = T_enc * D_enc;
                if (cached_output != nullptr && cached_count >= required) {
                    src = cached_output;
                }

                // Accept encoder hidden output [T, hidden] or direct logits [T, vocab].
                if (D_enc == expected_hidden_dim || D_enc == expected_vocab_size) {
                    size_t enc_output = gb->input({T_enc, D_enc}, Precision::FP16);
                    gb->set_input(enc_output, src, Precision::FP16);
                    return enc_output;
                }
            }
        }
    }
    ComputeBackend backend = ComputeBackend::CPU;
    size_t hidden = build_subsampling(gb, audio_features);
    const auto& hidden_shape = gb->get_output_buffer(hidden).shape;
    if (hidden_shape.size() != 2) {
        throw std::runtime_error("Parakeet encoder expects subsampling output [T, D]");
    }
    size_t position_embeddings = build_relative_position_embeddings(gb, hidden_shape[0]);
    for (uint32_t i = 0; i < config_.num_layers; ++i) {
        hidden = build_encoder_block(gb, hidden, position_embeddings, i, backend);
    }
    return hidden;
}

size_t ParakeetModel::build_ctc_logits(CactusGraph* gb, size_t hidden_states) {
    const auto& hidden_shape = gb->get_output_buffer(hidden_states).shape;
    if (hidden_shape.size() != 2) {
        throw std::runtime_error("Parakeet CTC head expects hidden states [T, D]");
    }
    const size_t T = hidden_shape[0];
    const size_t D = hidden_shape[1];

    size_t hidden_nlc = gb->reshape(hidden_states, {1, T, D});
    const auto& ctc_w_shape = gb->get_output_buffer(weight_nodes_.ctc_head_weight).shape;
    size_t vocab_size = ctc_w_shape.empty() ? config_.vocab_size : ctc_w_shape[0];
    size_t logits_nlc = gb->conv1d_pointwise(hidden_nlc, weight_nodes_.ctc_head_weight, weight_nodes_.ctc_head_bias);

    return gb->reshape(logits_nlc, {T, vocab_size});
}

size_t ParakeetModel::forward(const std::vector<float>& audio_features, const std::vector<uint32_t>& tokens, bool use_cache) {
    (void)tokens;
    (void)use_cache;
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->clear_debug_nodes();
    size_t encoder_out = build_encoder(gb, audio_features);

    const auto& enc_shape = gb->get_output_buffer(encoder_out).shape;
    if (enc_shape.size() != 2) {
        throw std::runtime_error("Parakeet encoder output must be rank-2 [T, D]");
    }

    size_t ctc_hidden_dim = std::max<size_t>(1, static_cast<size_t>(config_.hidden_dim));
    size_t ctc_vocab_size = std::max<size_t>(1, static_cast<size_t>(config_.vocab_size));
    const auto& ctc_w_shape = gb->get_output_buffer(weight_nodes_.ctc_head_weight).shape;
    if (ctc_w_shape.size() == 2) {
        ctc_vocab_size = ctc_w_shape[0];
        ctc_hidden_dim = ctc_w_shape[1];
    } else if (ctc_w_shape.size() == 3) {
        ctc_vocab_size = ctc_w_shape[0];
        ctc_hidden_dim = ctc_w_shape[1];
    }

    // If the NPU graph already emits [T, vocab] logits, skip CPU CTC head.
    if (enc_shape[1] == ctc_vocab_size && ctc_vocab_size != ctc_hidden_dim) {
        return encoder_out;
    }

    if (enc_shape[1] != ctc_hidden_dim) {
        throw std::runtime_error(
            "Parakeet encoder output dim mismatch: expected hidden dim " +
            std::to_string(ctc_hidden_dim) + ", got " + std::to_string(enc_shape[1]));
    }

    return build_ctc_logits(gb, encoder_out);
}

std::vector<uint32_t> ParakeetModel::greedy_decode_tokens(CactusGraph* gb, size_t logits_node) const {
    const auto& logits_buf = gb->get_output_buffer(logits_node);
    if (logits_buf.shape.size() != 2) {
        throw std::runtime_error("Parakeet logits must be rank-2 [T, vocab]");
    }

    const size_t T = logits_buf.shape[0];
    const size_t vocab_size = logits_buf.shape[1];
    std::vector<uint32_t> frame_ids(T, 0);

    if (logits_buf.precision == Precision::FP32) {
        const float* src = logits_buf.data_as<float>();
        for (size_t t = 0; t < T; ++t) {
            const float* row = src + t * vocab_size;
            size_t best_idx = 0;
            float best_val = row[0];
            for (size_t v = 1; v < vocab_size; ++v) {
                if (row[v] > best_val) {
                    best_val = row[v];
                    best_idx = v;
                }
            }
            frame_ids[t] = static_cast<uint32_t>(best_idx);
        }
    } else if (logits_buf.precision == Precision::FP16) {
        const __fp16* src = logits_buf.data_as<__fp16>();
        for (size_t t = 0; t < T; ++t) {
            const __fp16* row = src + t * vocab_size;
            size_t best_idx = 0;
            float best_val = static_cast<float>(row[0]);
            for (size_t v = 1; v < vocab_size; ++v) {
                const float val = static_cast<float>(row[v]);
                if (val > best_val) {
                    best_val = val;
                    best_idx = v;
                }
            }
            frame_ids[t] = static_cast<uint32_t>(best_idx);
        }
    } else {
        const int8_t* src = logits_buf.data_as<int8_t>();
        for (size_t t = 0; t < T; ++t) {
            const int8_t* row = src + t * vocab_size;
            size_t best_idx = 0;
            int best_val = static_cast<int>(row[0]);
            for (size_t v = 1; v < vocab_size; ++v) {
                const int val = static_cast<int>(row[v]);
                if (val > best_val) {
                    best_val = val;
                    best_idx = v;
                }
            }
            frame_ids[t] = static_cast<uint32_t>(best_idx);
        }
    }

    const uint32_t blank_id = config_.pad_token_id > 0 ? config_.pad_token_id : static_cast<uint32_t>(vocab_size - 1);
    std::vector<uint32_t> decoded;
    decoded.reserve(frame_ids.size());

    uint32_t prev = blank_id;
    for (uint32_t id : frame_ids) {
        if (id == blank_id) {
            prev = blank_id;
            continue;
        }
        if (id == prev) {
            continue;
        }
        decoded.push_back(id);
        prev = id;
    }

    return decoded;
}

uint32_t ParakeetModel::decode_with_audio(
    const std::vector<uint32_t>& tokens,
    const std::vector<float>& audio_features,
    float temperature,
    float top_p,
    size_t top_k,
    const std::string& profile_file,
    float* out_entropy)
{
    (void)temperature;
    (void)top_p;
    (void)top_k;

    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }
    if (audio_features.empty()) {
        throw std::runtime_error("Audio features cannot be empty in Parakeet decode_with_audio");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    const bool new_request = !ctc_tokens_ready_ || tokens.empty() || tokens.size() < last_input_token_count_;
    if (new_request) {
        gb->soft_reset();
        size_t logits_node = forward(audio_features, tokens, false);

        if (!profile_file.empty()) {
            gb->execute(profile_file);
        } else {
            gb->execute();
        }

        ctc_tokens_ = greedy_decode_tokens(gb, logits_node);
        ctc_emit_index_ = 0;
        ctc_tokens_ready_ = true;
    }

    last_input_token_count_ = tokens.size();
    if (out_entropy) {
        *out_entropy = 0.0f;
    }

    if (ctc_emit_index_ < ctc_tokens_.size()) {
        return ctc_tokens_[ctc_emit_index_++];
    }
    return get_tokenizer()->get_eos_token();
}

std::vector<float> ParakeetModel::get_audio_embeddings(const std::vector<float>& audio_features) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    size_t hidden = build_encoder(gb, audio_features);

    const auto& hidden_shape = gb->get_output_buffer(hidden).shape;
    size_t ctc_hidden_dim = std::max<size_t>(1, static_cast<size_t>(config_.hidden_dim));
    size_t ctc_vocab_size = std::max<size_t>(1, static_cast<size_t>(config_.vocab_size));
    const auto& ctc_w_shape = gb->get_output_buffer(weight_nodes_.ctc_head_weight).shape;
    if (ctc_w_shape.size() == 2) {
        ctc_vocab_size = ctc_w_shape[0];
        ctc_hidden_dim = ctc_w_shape[1];
    } else if (ctc_w_shape.size() == 3) {
        ctc_vocab_size = ctc_w_shape[0];
        ctc_hidden_dim = ctc_w_shape[1];
    }

    // Embeddings require hidden states; if NPU returned logits, rerun CPU encoder.
    if (hidden_shape.size() == 2 &&
        hidden_shape[1] == ctc_vocab_size &&
        ctc_vocab_size != ctc_hidden_dim &&
        use_npu_encoder_) {
        const bool prev_use_npu = use_npu_encoder_;
        use_npu_encoder_ = false;
        gb->soft_reset();
        hidden = build_encoder(gb, audio_features);
        use_npu_encoder_ = prev_use_npu;
    }

    size_t pooled = gb->mean(hidden, 0);
    gb->execute();

    const auto& output_buf = gb->get_output_buffer(pooled);
    const size_t hidden_dim = output_buf.total_size;
    std::vector<float> embedding(hidden_dim);

    if (output_buf.precision == Precision::FP32) {
        const float* src = output_buf.data_as<float>();
        std::copy(src, src + hidden_dim, embedding.begin());
    } else if (output_buf.precision == Precision::FP16) {
        const __fp16* src = output_buf.data_as<__fp16>();
        Quantization::fp16_to_fp32(src, embedding.data(), hidden_dim);
    } else {
        const int8_t* src = output_buf.data_as<int8_t>();
        Quantization::int8_to_fp32(src, embedding.data(), hidden_dim, 1.0f);
    }

    reset_cache();
    return embedding;
}

void ParakeetModel::reset_cache() {
    Model::reset_cache();
    ctc_tokens_ready_ = false;
    ctc_emit_index_ = 0;
    ctc_tokens_.clear();
    last_input_token_count_ = 0;
}

} // namespace engine
} // namespace cactus