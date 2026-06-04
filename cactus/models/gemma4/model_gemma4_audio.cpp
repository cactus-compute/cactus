#include "model_gemma4.h"
#include "../../graph/graph.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace cactus {
namespace engine {

static const float INV_LN2 = 1.0f / std::log(2.0f);
static const float GEMMA4_AUDIO_K_SCALE = std::log(1.0f + std::exp(1.0f)) / std::log(2.0f);
static const char* GEMMA4_AUDIO_NPU_INPUT_NAME = "x";

static size_t shape_elements(const std::vector<int>& shape) {
    if (shape.empty()) return 0;
    size_t total = 1;
    for (int d : shape) {
        if (d <= 0) return 0;
        total *= static_cast<size_t>(d);
    }
    return total;
}

static size_t infer_npu_audio_max_frames(const std::vector<int>& shape, size_t mel_bins) {
    if (shape.size() == 2) {
        size_t s0 = static_cast<size_t>(shape[0]);
        size_t s1 = static_cast<size_t>(shape[1]);
        if (s1 == mel_bins) return s0;
        if (s0 == mel_bins) return s1;
    } else if (shape.size() == 3) {
        size_t s0 = static_cast<size_t>(shape[0]);
        size_t s1 = static_cast<size_t>(shape[1]);
        size_t s2 = static_cast<size_t>(shape[2]);
        if (s0 == 1 && s2 == mel_bins) return s1; // [1, T, F]
        if (s0 == 1 && s1 == mel_bins) return s2; // [1, F, T]
        if (s1 == mel_bins && s2 == 1) return s0; // [T, F, 1]
        if (s0 == mel_bins && s2 == 1) return s1; // [F, T, 1]
    } else if (shape.size() == 4) {
        size_t s0 = static_cast<size_t>(shape[0]);
        size_t s1 = static_cast<size_t>(shape[1]);
        size_t s2 = static_cast<size_t>(shape[2]);
        size_t s3 = static_cast<size_t>(shape[3]);
        if (s0 == 1 && s1 == 1 && s3 == mel_bins) return s2; // [1, 1, T, F]
        if (s0 == 1 && s1 == 1 && s2 == mel_bins) return s3; // [1, 1, F, T]
        if (s0 == 1 && s1 == mel_bins && s3 == 1) return s2; // [1, F, T, 1]
        if (s0 == 1 && s2 == mel_bins && s3 == 1) return s1; // [1, T, F, 1]
    }
    return 0;
}

static bool pack_gemma4_audio_for_npu(const std::vector<float>& mel_features,
                                         size_t frames,
                                         size_t mel_bins,
                                         const std::vector<int>& input_shape,
                                         std::vector<__fp16>& packed) {
    if (input_shape.empty()) return false;
    const size_t total = shape_elements(input_shape);
    if (total == 0) return false;
    packed.resize(total);
    std::fill(packed.begin(), packed.end(), static_cast<__fp16>(0.0f));

    auto tm = [&](size_t t, size_t m) -> __fp16 {
        return static_cast<__fp16>(mel_features[t * mel_bins + m]);
    };

    if (input_shape.size() == 4) {
        const size_t s0 = static_cast<size_t>(input_shape[0]);
        const size_t s1 = static_cast<size_t>(input_shape[1]);
        const size_t s2 = static_cast<size_t>(input_shape[2]);
        const size_t s3 = static_cast<size_t>(input_shape[3]);
        if (s0 != 1) return false;

        if (s1 == 1 && s2 >= frames && s3 == mel_bins) {
            for (size_t t = 0; t < frames; ++t)
                for (size_t m = 0; m < mel_bins; ++m)
                    packed[(t * s3) + m] = tm(t, m);
            return true;
        }
        if (s1 == 1 && s2 == mel_bins && s3 >= frames) {
            for (size_t t = 0; t < frames; ++t)
                for (size_t m = 0; m < mel_bins; ++m)
                    packed[(m * s3) + t] = tm(t, m);
            return true;
        }
        if (s1 >= frames && s2 == mel_bins && s3 == 1) {
            for (size_t t = 0; t < frames; ++t)
                for (size_t m = 0; m < mel_bins; ++m)
                    packed[((t * s2 + m) * s3)] = tm(t, m);
            return true;
        }
        if (s1 == mel_bins && s2 >= frames && s3 == 1) {
            for (size_t t = 0; t < frames; ++t)
                for (size_t m = 0; m < mel_bins; ++m)
                    packed[((m * s2 + t) * s3)] = tm(t, m);
            return true;
        }
        return false;
    }

    if (input_shape.size() == 3) {
        const size_t s0 = static_cast<size_t>(input_shape[0]);
        const size_t s1 = static_cast<size_t>(input_shape[1]);
        const size_t s2 = static_cast<size_t>(input_shape[2]);

        if (s0 == 1 && s1 >= frames && s2 == mel_bins) {
            for (size_t t = 0; t < frames; ++t)
                for (size_t m = 0; m < mel_bins; ++m)
                    packed[t * s2 + m] = tm(t, m);
            return true;
        }
        if (s0 == 1 && s1 == mel_bins && s2 >= frames) {
            for (size_t t = 0; t < frames; ++t)
                for (size_t m = 0; m < mel_bins; ++m)
                    packed[m * s2 + t] = tm(t, m);
            return true;
        }
        if (s0 >= frames && s1 == mel_bins && s2 == 1) {
            for (size_t t = 0; t < frames; ++t)
                for (size_t m = 0; m < mel_bins; ++m)
                    packed[(t * s1 + m) * s2] = tm(t, m);
            return true;
        }
        if (s0 == mel_bins && s1 >= frames && s2 == 1) {
            for (size_t t = 0; t < frames; ++t)
                for (size_t m = 0; m < mel_bins; ++m)
                    packed[(m * s1 + t) * s2] = tm(t, m);
            return true;
        }
        return false;
    }

    if (input_shape.size() == 2) {
        const size_t s0 = static_cast<size_t>(input_shape[0]);
        const size_t s1 = static_cast<size_t>(input_shape[1]);

        if (s0 >= frames && s1 == mel_bins) {
            for (size_t t = 0; t < frames; ++t)
                for (size_t m = 0; m < mel_bins; ++m)
                    packed[t * s1 + m] = tm(t, m);
            return true;
        }
        if (s0 == mel_bins && s1 >= frames) {
            for (size_t t = 0; t < frames; ++t)
                for (size_t m = 0; m < mel_bins; ++m)
                    packed[m * s1 + t] = tm(t, m);
            return true;
        }
        return false;
    }

    return false;
}

struct NPUAudioOutputLayout {
    std::vector<size_t> dims;
    size_t hidden_axis = SIZE_MAX;
    size_t hidden_dim = 0;
    size_t time_steps = 0;
};

static bool infer_npu_audio_output_layout(const std::vector<int>& output_shape,
                                          size_t elements_written,
                                          size_t expected_hidden_dim,
                                          NPUAudioOutputLayout& layout) {
    if (elements_written == 0) return false;

    layout = NPUAudioOutputLayout{};
    for (int d : output_shape) {
        if (d > 0) layout.dims.push_back(static_cast<size_t>(d));
    }

    if (layout.dims.empty()) {
        if (expected_hidden_dim == 0 || (elements_written % expected_hidden_dim) != 0) {
            return false;
        }
        layout.dims = {elements_written / expected_hidden_dim, expected_hidden_dim};
    }

    const size_t rank = layout.dims.size();
    size_t hidden_axis = SIZE_MAX;
    if (expected_hidden_dim > 0) {
        if (layout.dims[rank - 1] == expected_hidden_dim) {
            hidden_axis = rank - 1;
        } else if (rank >= 2 && layout.dims[rank - 2] == expected_hidden_dim) {
            hidden_axis = rank - 2;
        } else {
            for (size_t i = 0; i < rank; ++i) {
                if (layout.dims[i] == expected_hidden_dim) {
                    hidden_axis = i;
                    break;
                }
            }
        }
    }

    if (hidden_axis == SIZE_MAX) {
        hidden_axis = rank - 1;
    }

    const size_t hidden_dim = layout.dims[hidden_axis];
    if (hidden_dim == 0 || (elements_written % hidden_dim) != 0) {
        return false;
    }

    layout.hidden_axis = hidden_axis;
    layout.hidden_dim = hidden_dim;
    layout.time_steps = elements_written / hidden_dim;
    return layout.time_steps > 0;
}

static bool materialize_npu_audio_time_major(const __fp16* src,
                                             const NPUAudioOutputLayout& layout,
                                             size_t time_steps,
                                             std::vector<__fp16>& dst) {
    if (!src || layout.dims.empty() || layout.hidden_dim == 0 || layout.time_steps == 0) {
        return false;
    }
    if (layout.hidden_axis >= layout.dims.size() || time_steps == 0 || time_steps > layout.time_steps) {
        return false;
    }

    const size_t rank = layout.dims.size();
    if (layout.hidden_axis == rank - 1) {
        // Already [time, hidden] in row-major flattening.
        return false;
    }

    std::vector<size_t> strides(rank, 1);
    for (size_t i = rank; i-- > 1;) {
        strides[i - 1] = strides[i] * layout.dims[i];
    }

    std::vector<size_t> non_hidden_axes;
    non_hidden_axes.reserve(rank - 1);
    for (size_t i = 0; i < rank; ++i) {
        if (i != layout.hidden_axis) non_hidden_axes.push_back(i);
    }

    dst.assign(time_steps * layout.hidden_dim, static_cast<__fp16>(0.0f));

    for (size_t t = 0; t < time_steps; ++t) {
        size_t rem = t;
        size_t base_offset = 0;
        for (size_t i = non_hidden_axes.size(); i-- > 0;) {
            size_t axis = non_hidden_axes[i];
            size_t dim = layout.dims[axis];
            size_t coord = rem % dim;
            rem /= dim;
            base_offset += coord * strides[axis];
        }

        for (size_t h = 0; h < layout.hidden_dim; ++h) {
            size_t src_index = base_offset + h * strides[layout.hidden_axis];
            dst[t * layout.hidden_dim + h] = src[src_index];
        }
    }

    return true;
}

static size_t graph_clamp(CactusGraph* gb, size_t x, float lo, float hi) {
    if (lo >= hi || std::isinf(lo) || std::isinf(hi)) return x;
    x = gb->scalar_add(x, -lo);
    x = gb->relu(x);
    x = gb->scalar_add(x, lo);
    x = gb->scalar_multiply(gb->scalar_add(x, -hi), -1.0f);
    x = gb->relu(x);
    x = gb->scalar_add(gb->scalar_multiply(x, -1.0f), hi);
    return x;
}

static size_t clipped_matmul(CactusGraph* gb, size_t input, size_t weight,
                              float in_min, float in_max, float out_min, float out_max,
                              ComputeBackend backend) {
    size_t clamped_in = graph_clamp(gb, input, in_min, in_max);
    size_t result = gb->matmul(clamped_in, weight, true, backend);
    return graph_clamp(gb, result, out_min, out_max);
}

static float read_scalar_weight(CactusGraph* gb, const std::string& path) {
    size_t node = gb->mmap_weights(path);
    const auto& buf = gb->get_output_buffer(node);
    if (buf.precision == Precision::FP16)
        return static_cast<float>(*buf.data_as<__fp16>());
    if (buf.precision == Precision::FP32)
        return *buf.data_as<float>();
    return 0.0f;
}

static Gemma4AudioModel::AudioWeightNodes::ClipBounds load_clip_bounds(CactusGraph* gb, const std::string& prefix) {
    Gemma4AudioModel::AudioWeightNodes::ClipBounds cb{0, 0, 0, 0};
    std::string in_min_path = prefix + "_input_min.weights";
    if (!std::filesystem::exists(in_min_path))
        return cb;
    cb.in_min = read_scalar_weight(gb, in_min_path);
    cb.in_max = read_scalar_weight(gb, prefix + "_input_max.weights");
    cb.out_min = read_scalar_weight(gb, prefix + "_output_min.weights");
    cb.out_max = read_scalar_weight(gb, prefix + "_output_max.weights");
    return cb;
}

static void get_conv2d_output_hw(const CactusGraph* gb, size_t node, size_t& h, size_t& w) {
    const auto& buf = gb->get_output_buffer(node);
    h = buf.shape[2];
    w = buf.shape[3];
}

Gemma4AudioModel::Gemma4AudioModel() : Model() {}
Gemma4AudioModel::Gemma4AudioModel(const Config& config) : Model(config) {}

void Gemma4AudioModel::load_weights_to_graph(CactusGraph* gb) {
    auto resolve = [&](const std::string& name) -> std::string {
        return model_folder_path_ + "/" + name;
    };
    auto resolve_existing = [&](const std::string& preferred,
                                const std::string& legacy = std::string()) -> std::string {
        std::string preferred_path = resolve(preferred);
        if (std::filesystem::exists(preferred_path)) {
            return preferred_path;
        }
        if (!legacy.empty()) {
            std::string legacy_path = resolve(legacy);
            if (std::filesystem::exists(legacy_path)) {
                return legacy_path;
            }
        }
        return preferred_path;
    };
    auto load_weight = [&](const std::string& preferred,
                           const std::string& legacy = std::string()) -> size_t {
        return gb->mmap_weights(resolve_existing(preferred, legacy));
    };
    auto load_clip_from_weight = [&](const std::string& preferred,
                                     const std::string& legacy = std::string())
                                     -> Gemma4AudioModel::AudioWeightNodes::ClipBounds {
        std::string weight_path = resolve_existing(preferred, legacy);
        static const std::string suffix = ".weights";
        if (weight_path.size() > suffix.size() &&
            weight_path.compare(weight_path.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return load_clip_bounds(gb, weight_path.substr(0, weight_path.size() - suffix.size()));
        }
        return Gemma4AudioModel::AudioWeightNodes::ClipBounds{0, 0, 0, 0};
    };

    if (!disable_npu_ && npu::is_npu_available()) {
        std::string npu_path = model_folder_path_ + "/audio_encoder.mlpackage";
        if (std::filesystem::exists(npu_path)) {
            npu_encoder_ = npu::create_encoder();
            if (npu_encoder_ && npu_encoder_->load(npu_path)) {
                use_npu_encoder_ = true;
                std::vector<int> input_shape = npu_encoder_->get_input_shape();
                npu_encoder_->preallocate(input_shape, GEMMA4_AUDIO_NPU_INPUT_NAME, "");
            } else {
                use_npu_encoder_ = false;
                npu_encoder_.reset();
                CACTUS_LOG_WARN("npu", "[gemma4-audio] found audio_encoder.mlpackage but failed to enable NPU audio encoder; using CPU");
            }
        } else {
            CACTUS_LOG_WARN("npu", "[gemma4-audio] audio_encoder.mlpackage not found; using CPU audio encoder");
        }
    } else if (!disable_npu_) {
        CACTUS_LOG_WARN("npu", "[gemma4-audio] NPU backend unavailable on this device; using CPU audio encoder");
    }

    // Keep CPU audio weights available even when the NPU encoder is enabled.
    // The runtime may still fall back to the graph path for unsupported inputs.
    {
        audio_weights_.sscp_conv0_weight = load_weight(
            "audio_subsample_conv_projection_conv_0_conv.weights",
            "audio_subsample_conv_projection_layer0_conv.weights");
        audio_weights_.sscp_conv0_norm = load_weight(
            "audio_subsample_conv_projection_conv_0_norm.weights",
            "audio_subsample_conv_projection_layer0_norm.weights");
        audio_weights_.sscp_conv1_weight = load_weight(
            "audio_subsample_conv_projection_conv_1_conv.weights",
            "audio_subsample_conv_projection_layer1_conv.weights");
        audio_weights_.sscp_conv1_norm = load_weight(
            "audio_subsample_conv_projection_conv_1_norm.weights",
            "audio_subsample_conv_projection_layer1_norm.weights");
        audio_weights_.sscp_input_proj = load_weight(
            "audio_subsample_conv_projection_input_proj.weights",
            "audio_subsample_conv_projection_input_proj_linear.weights");

        audio_weights_.layers.resize(config_.audio_num_layers);
        for (uint32_t i = 0; i < config_.audio_num_layers; i++) {
            auto& layer = audio_weights_.layers[i];

            auto layer_weight = [&](const std::string& preferred_suffix,
                                    const std::string& legacy_suffix = std::string()) -> size_t {
                std::string legacy = legacy_suffix.empty() ? preferred_suffix : legacy_suffix;
                return load_weight(
                    "audio_conformer_" + std::to_string(i) + "_" + preferred_suffix,
                    "audio_layers_" + std::to_string(i) + "_" + legacy);
            };
            auto layer_clip = [&](const std::string& preferred_suffix,
                                  const std::string& legacy_suffix = std::string())
                                  -> Gemma4AudioModel::AudioWeightNodes::ClipBounds {
                std::string legacy = legacy_suffix.empty() ? preferred_suffix : legacy_suffix;
                return load_clip_from_weight(
                    "audio_conformer_" + std::to_string(i) + "_" + preferred_suffix,
                    "audio_layers_" + std::to_string(i) + "_" + legacy);
            };

            layer.ffw_start_1 = layer_weight("ffw_layer_start_ffw_layer_1.weights", "feed_forward1_ffw_layer_1.weights");
            layer.ffw_start_2 = layer_weight("ffw_layer_start_ffw_layer_2.weights", "feed_forward1_ffw_layer_2.weights");
            layer.ffw_start_1_clip = layer_clip("ffw_layer_start_ffw_layer_1.weights", "feed_forward1_ffw_layer_1.weights");
            layer.ffw_start_2_clip = layer_clip("ffw_layer_start_ffw_layer_2.weights", "feed_forward1_ffw_layer_2.weights");
            layer.ffw_start_pre_norm = layer_weight("ffw_layer_start_pre_layer_norm.weights", "feed_forward1_pre_layer_norm.weights");
            layer.ffw_start_post_norm = layer_weight("ffw_layer_start_post_layer_norm.weights", "feed_forward1_post_layer_norm.weights");

            layer.attn_q = layer_weight("attention_attn_q_proj.weights", "self_attn_q_proj.weights");
            layer.attn_k = layer_weight("attention_attn_k_proj.weights", "self_attn_k_proj.weights");
            layer.attn_v = layer_weight("attention_attn_v_proj.weights", "self_attn_v_proj.weights");
            layer.attn_q_clip = layer_clip("attention_attn_q_proj.weights", "self_attn_q_proj.weights");
            layer.attn_k_clip = layer_clip("attention_attn_k_proj.weights", "self_attn_k_proj.weights");
            layer.attn_v_clip = layer_clip("attention_attn_v_proj.weights", "self_attn_v_proj.weights");
            layer.attn_per_dim_scale = layer_weight("attention_attn_per_dim_scale.weights", "self_attn_per_dim_scale.weights");
            layer.attn_rel_pos_proj = layer_weight(
                "attention_attn_relative_position_embedding_pos_proj.weights",
                "self_attn_relative_k_proj.weights");
            layer.attn_post = layer_weight("attention_post.weights", "self_attn_post.weights");
            layer.attn_post_clip = layer_clip("attention_post.weights", "self_attn_post.weights");
            layer.attn_pre_norm = layer_weight("attention_pre_attn_norm.weights", "norm_pre_attn.weights");
            layer.attn_post_norm = layer_weight("attention_post_norm.weights", "norm_post_attn.weights");

            layer.lconv_start = layer_weight("lconv1d_linear_start.weights");
            layer.lconv_depthwise = layer_weight("lconv1d_depthwise_conv1d.weights");
            layer.lconv_end = layer_weight("lconv1d_linear_end.weights");
            layer.lconv_start_clip = layer_clip("lconv1d_linear_start.weights");
            layer.lconv_end_clip = layer_clip("lconv1d_linear_end.weights");
            layer.lconv_pre_norm = layer_weight("lconv1d_pre_layer_norm.weights");
            layer.lconv_conv_norm = layer_weight("lconv1d_conv_norm.weights");

            layer.ffw_end_1 = layer_weight("ffw_layer_end_ffw_layer_1.weights", "feed_forward2_ffw_layer_1.weights");
            layer.ffw_end_2 = layer_weight("ffw_layer_end_ffw_layer_2.weights", "feed_forward2_ffw_layer_2.weights");
            layer.ffw_end_1_clip = layer_clip("ffw_layer_end_ffw_layer_1.weights", "feed_forward2_ffw_layer_1.weights");
            layer.ffw_end_2_clip = layer_clip("ffw_layer_end_ffw_layer_2.weights", "feed_forward2_ffw_layer_2.weights");
            layer.ffw_end_pre_norm = layer_weight("ffw_layer_end_pre_layer_norm.weights", "feed_forward2_pre_layer_norm.weights");
            layer.ffw_end_post_norm = layer_weight("ffw_layer_end_post_layer_norm.weights", "feed_forward2_post_layer_norm.weights");

            layer.block_norm = layer_weight("norm.weights", "norm_out.weights");
        }
        
        audio_weights_.output_proj = load_weight("audio_output_proj.weights");
        audio_weights_.output_proj_bias = load_weight("audio_output_proj.bias");
    }
    audio_weights_.embed_audio_proj = load_weight("embed_audio_proj.weights");

    size_t proj_out_dim = config_.audio_output_proj_dims > 0
        ? static_cast<size_t>(config_.audio_output_proj_dims)
        : static_cast<size_t>(config_.audio_hidden_dim);
    audio_proj_norm_ones_.assign(proj_out_dim, static_cast<__fp16>(1.0f));
    audio_proj_norm_ones_node_ = gb->input({proj_out_dim}, Precision::FP16);
    gb->set_external_input(audio_proj_norm_ones_node_, audio_proj_norm_ones_.data(), Precision::FP16);

    output_weight_node_id_ = 0;
}

size_t Gemma4AudioModel::build_sscp(CactusGraph* gb, const std::vector<float>& mel_features,
                                        size_t num_frames, ComputeBackend backend) {
    size_t mel_bins = config_.audio_input_feat_size;
    size_t conv0_ch = config_.audio_sscp_conv0_channels;
    size_t conv1_ch = config_.audio_sscp_conv1_channels;

    size_t mel_input = gb->input({num_frames, mel_bins}, Precision::FP32);
    gb->set_input(mel_input, mel_features.data(), Precision::FP32);
    size_t mel_fp16 = gb->precision_cast(mel_input, Precision::FP16);

    size_t x = gb->reshape(mel_fp16, {1, 1, num_frames, mel_bins});

    x = gb->conv2d_k3s2p1(x, audio_weights_.sscp_conv0_weight);
    size_t h1, w1;
    get_conv2d_output_hw(gb, x, h1, w1);

    x = gb->reshape(x, {conv0_ch, h1, w1});
    x = gb->transposeN(x, {1, 2, 0});
    x = gb->reshape(x, {h1 * w1, conv0_ch});
    x = gb->layernorm(x, audio_weights_.sscp_conv0_norm, config_.audio_sscp_conv_eps);
    x = gb->reshape(x, {h1, w1, conv0_ch});
    x = gb->transposeN(x, {2, 0, 1});
    x = gb->relu(x);
    x = gb->reshape(x, {1, conv0_ch, h1, w1});

    x = gb->conv2d_k3s2p1(x, audio_weights_.sscp_conv1_weight);
    size_t h2, w2;
    get_conv2d_output_hw(gb, x, h2, w2);

    x = gb->reshape(x, {conv1_ch, h2, w2});
    x = gb->transposeN(x, {1, 2, 0});
    x = gb->reshape(x, {h2 * w2, conv1_ch});
    x = gb->layernorm(x, audio_weights_.sscp_conv1_norm, config_.audio_sscp_conv_eps);
    x = gb->reshape(x, {h2, w2, conv1_ch});
    x = gb->transposeN(x, {2, 0, 1});
    x = gb->relu(x);

    x = gb->transposeN(x, {1, 2, 0});
    x = gb->reshape(x, {h2, w2 * conv1_ch});

    return gb->matmul(x, audio_weights_.sscp_input_proj, true, backend);
}

static std::vector<float> compute_timing_signal(size_t num_positions, size_t hidden_dim, size_t max_past) {
    size_t num_timescales = hidden_dim / 2;
    float log_ts_inc = std::log(1.0e4f) / std::max(static_cast<int>(num_timescales) - 1, 1);

    std::vector<float> signal(num_positions * hidden_dim);
    for (size_t p = 0; p < num_positions; p++) {
        float pos = static_cast<float>(max_past) - static_cast<float>(p);
        for (size_t i = 0; i < num_timescales; i++) {
            float scaled = pos * std::exp(-static_cast<float>(i) * log_ts_inc);
            signal[p * hidden_dim + i] = std::sin(scaled);
            signal[p * hidden_dim + num_timescales + i] = std::cos(scaled);
        }
    }
    return signal;
}

static size_t create_zero_pad_node(CactusGraph* gb, size_t time_len, size_t num_heads, size_t head_dim) {
    if (time_len == 0) return 0;
    std::vector<__fp16> zeros(time_len * num_heads * head_dim, static_cast<__fp16>(0.0f));
    size_t node = gb->input({time_len, num_heads, head_dim}, Precision::FP16);
    gb->set_input(node, zeros.data(), Precision::FP16);
    return node;
}

Gemma4AudioModel::ConformerContext Gemma4AudioModel::build_conformer_context(
    CactusGraph* gb, size_t sscp_output) {

    size_t num_heads = config_.audio_num_heads;
    size_t head_dim = config_.audio_head_dim;
    size_t hidden_dim = num_heads * head_dim;
    size_t max_past = config_.audio_context_left > 0 ? config_.audio_context_left - 1 : 0;
    size_t num_positions = max_past + config_.audio_context_right + 1;
    size_t seq_len = gb->get_output_buffer(sscp_output).shape[0];

    auto timing_signal = compute_timing_signal(num_positions, hidden_dim, max_past);
    size_t timing_input = gb->input({num_positions, hidden_dim}, Precision::FP32);
    gb->set_input(timing_input, timing_signal.data(), Precision::FP32);
    size_t timing_fp16 = gb->precision_cast(timing_input, Precision::FP16);

    size_t front_pad_len = (seq_len >= num_positions) ? seq_len - num_positions : 0;
    size_t back_pad_len = seq_len > 0 ? seq_len - 1 : 0;

    return {
        timing_fp16,
        create_zero_pad_node(gb, front_pad_len, num_heads, head_dim),
        create_zero_pad_node(gb, back_pad_len, num_heads, head_dim),
        seq_len
    };
}

size_t Gemma4AudioModel::build_conformer_ffw(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                                 bool is_end, ComputeBackend backend) {
    const auto& layer = audio_weights_.layers[layer_idx];
    float residual_weight = config_.audio_residual_weight;
    float eps = config_.audio_rms_norm_eps;

    size_t residual = input;

    size_t pre_norm = is_end ? layer.ffw_end_pre_norm : layer.ffw_start_pre_norm;
    size_t post_norm = is_end ? layer.ffw_end_post_norm : layer.ffw_start_post_norm;
    size_t w1 = is_end ? layer.ffw_end_1 : layer.ffw_start_1;
    size_t w2 = is_end ? layer.ffw_end_2 : layer.ffw_start_2;

    auto& c1 = is_end ? layer.ffw_end_1_clip : layer.ffw_start_1_clip;
    auto& c2 = is_end ? layer.ffw_end_2_clip : layer.ffw_start_2_clip;

    size_t x = gb->rms_norm(input, pre_norm, eps);

    x = clipped_matmul(gb, x, w1, c1.in_min, c1.in_max, c1.out_min, c1.out_max, backend);
    x = gb->silu(x);
    x = clipped_matmul(gb, x, w2, c2.in_min, c2.in_max, c2.out_min, c2.out_max, backend);

    x = gb->rms_norm(x, post_norm, eps);

    x = gb->scalar_multiply(x, residual_weight);
    return gb->add(residual, x);
}

size_t Gemma4AudioModel::build_conformer_attention(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                                       const ConformerContext& ctx, ComputeBackend backend) {
    const auto& layer = audio_weights_.layers[layer_idx];
    float eps = config_.audio_rms_norm_eps;
    size_t num_heads = config_.audio_num_heads;
    size_t head_dim = config_.audio_head_dim;
    size_t hidden_dim = num_heads * head_dim;
    float logit_cap = config_.audio_logit_cap;
    size_t max_past = config_.audio_context_left > 0 ? config_.audio_context_left - 1 : 0;
    size_t num_positions = max_past + config_.audio_context_right + 1;
    size_t seq_len = ctx.seq_len;

    size_t residual = input;

    size_t x = gb->rms_norm(input, layer.attn_pre_norm, eps);

    size_t q = clipped_matmul(gb, x, layer.attn_q, layer.attn_q_clip.in_min, layer.attn_q_clip.in_max, layer.attn_q_clip.out_min, layer.attn_q_clip.out_max, backend);
    size_t k = clipped_matmul(gb, x, layer.attn_k, layer.attn_k_clip.in_min, layer.attn_k_clip.in_max, layer.attn_k_clip.out_min, layer.attn_k_clip.out_max, backend);
    size_t v = clipped_matmul(gb, x, layer.attn_v, layer.attn_v_clip.in_min, layer.attn_v_clip.in_max, layer.attn_v_clip.out_min, layer.attn_v_clip.out_max, backend);

    float q_scale = (1.0f / std::sqrt(static_cast<float>(head_dim))) * INV_LN2;
    size_t q_dim_scale_fp16 = gb->precision_cast(layer.attn_per_dim_scale, Precision::FP16);
    size_t q_dim_scale = gb->scalar_exp(q_dim_scale_fp16);
    q_dim_scale = gb->scalar_add(q_dim_scale, 1.0f);
    q_dim_scale = gb->scalar_log(q_dim_scale);
    q_dim_scale = gb->scalar_multiply(q_dim_scale, q_scale);

    size_t q_flat = gb->reshape(q, {seq_len * num_heads, head_dim});
    q_flat = gb->multiply(q_flat, q_dim_scale);
    q = gb->reshape(q_flat, {seq_len, hidden_dim});

    k = gb->scalar_multiply(k, GEMMA4_AUDIO_K_SCALE);

    size_t sin_emb = gb->matmul(ctx.timing_fp16, layer.attn_rel_pos_proj, true, backend);
    size_t sin_emb_3d = gb->reshape(sin_emb, {num_positions, num_heads, head_dim});
    size_t R = 2 * seq_len - 1;

    size_t rel_key;
    if (seq_len >= num_positions) {
        rel_key = (ctx.front_pad != 0) ? gb->concat(ctx.front_pad, sin_emb_3d, 0) : sin_emb_3d;
        if (ctx.back_pad != 0)
            rel_key = gb->concat(rel_key, ctx.back_pad, 0);
    } else {
        size_t start_idx = num_positions - seq_len;
        size_t sin_emb_sliced = gb->slice(sin_emb_3d, 0, start_idx, seq_len);
        rel_key = (ctx.back_pad != 0) ? gb->concat(sin_emb_sliced, ctx.back_pad, 0) : sin_emb_sliced;
    }

    size_t rel_key_4d = gb->reshape(rel_key, {1, R, num_heads, head_dim});
    size_t q4 = gb->reshape(q, {1, seq_len, num_heads, head_dim});
    size_t rel_bias = gb->rel_pos_bias(q4, rel_key_4d, 1.0f);

    size_t k4 = gb->reshape(k, {1, seq_len, num_heads, head_dim});
    size_t v4 = gb->reshape(v, {1, seq_len, num_heads, head_dim});

    size_t attn = gb->attention_masked(q4, k4, v4, rel_bias, 1.0f,
                                        true, backend, true, 0, max_past, logit_cap);

    size_t attn_out = gb->reshape(attn, {seq_len, hidden_dim});

    size_t out = clipped_matmul(gb, attn_out, layer.attn_post, layer.attn_post_clip.in_min, layer.attn_post_clip.in_max, layer.attn_post_clip.out_min, layer.attn_post_clip.out_max, backend);
    out = gb->rms_norm(out, layer.attn_post_norm, eps);

    return gb->add(residual, out);
}

size_t Gemma4AudioModel::build_conformer_lconv1d(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                                      ComputeBackend backend) {
    const auto& layer = audio_weights_.layers[layer_idx];
    float eps = config_.audio_rms_norm_eps;
    size_t kernel_size = config_.audio_conf_conv_kernel_size;
    size_t hidden_dim = config_.audio_hidden_dim;

    size_t residual = input;

    size_t x = gb->rms_norm(input, layer.lconv_pre_norm, eps);

    x = clipped_matmul(gb, x, layer.lconv_start, layer.lconv_start_clip.in_min, layer.lconv_start_clip.in_max, layer.lconv_start_clip.out_min, layer.lconv_start_clip.out_max, backend);

    x = gb->glu(x, -1);

    const auto& x_buf = gb->get_output_buffer(x);
    size_t seq_len = x_buf.shape[0];
    x = gb->reshape(x, {1, seq_len, hidden_dim});

    x = gb->conv1d_causal(x, layer.lconv_depthwise, kernel_size);

    x = gb->reshape(x, {seq_len, hidden_dim});

    x = gb->rms_norm(x, layer.lconv_conv_norm, eps);
    x = gb->silu(x);

    x = clipped_matmul(gb, x, layer.lconv_end, layer.lconv_end_clip.in_min, layer.lconv_end_clip.in_max, layer.lconv_end_clip.out_min, layer.lconv_end_clip.out_max, backend);

    return gb->add(x, residual);
}

size_t Gemma4AudioModel::build_conformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                                    const ConformerContext& ctx, ComputeBackend backend) {
    hidden = build_conformer_ffw(gb, hidden, layer_idx, false, backend);
    hidden = build_conformer_attention(gb, hidden, layer_idx, ctx, backend);
    hidden = build_conformer_lconv1d(gb, hidden, layer_idx, backend);
    hidden = build_conformer_ffw(gb, hidden, layer_idx, true, backend);
    return gb->rms_norm(hidden, audio_weights_.layers[layer_idx].block_norm, config_.audio_rms_norm_eps);
}

size_t Gemma4AudioModel::forward_audio(CactusGraph* gb, const std::vector<float>& mel_features,
                                           size_t num_frames, ComputeBackend backend) {
    if (use_npu_encoder_ && npu_encoder_ && npu_encoder_->is_available()) {
        std::vector<int> npu_input_shape = npu_encoder_->get_input_shape();
        size_t mel_bins = static_cast<size_t>(config_.audio_input_feat_size);
        size_t max_npu_frames = infer_npu_audio_max_frames(npu_input_shape, mel_bins);
        size_t copy_frames = max_npu_frames > 0 ? std::min(num_frames, max_npu_frames) : num_frames;
        size_t input_values = copy_frames * mel_bins;

        if (mel_features.size() < input_values) {
            CACTUS_LOG_WARN("npu", "[gemma4-audio] insufficient mel input values; falling back to CPU audio encoder");
        } else if (!pack_gemma4_audio_for_npu(mel_features, copy_frames, mel_bins, npu_input_shape,
                                                 npu_audio_input_scratch_)) {
            CACTUS_LOG_WARN("npu", "[gemma4-audio] unsupported NPU input shape; falling back to CPU audio encoder");
        } else {
            size_t t1 = (copy_frames + 1) / 2;
            size_t t2 = (t1 + 1) / 2;
            size_t expected_out_dim = config_.audio_output_proj_dims > 0
                ? static_cast<size_t>(config_.audio_output_proj_dims)
                : static_cast<size_t>(config_.audio_hidden_dim);

            std::vector<int> out_shape = npu_encoder_->get_output_shape();
            size_t out_elements = npu_encoder_->get_output_buffer_size();
            if (out_elements == 0) out_elements = shape_elements(out_shape);
            if (out_elements == 0) out_elements = std::max<size_t>(1, t2 * expected_out_dim);

            if (npu_audio_output_scratch_.size() < out_elements)
                npu_audio_output_scratch_.resize(out_elements);

            size_t written = npu_encoder_->encode(
                npu_audio_input_scratch_.data(), npu_audio_output_scratch_.data(), npu_input_shape,
                GEMMA4_AUDIO_NPU_INPUT_NAME, "");

            NPUAudioOutputLayout out_layout;
            if (written > 0 &&
                infer_npu_audio_output_layout(out_shape, written, expected_out_dim, out_layout) &&
                out_layout.hidden_dim == expected_out_dim) {
                const __fp16* src = npu_audio_output_scratch_.data();
                __fp16* cached_output = npu_encoder_->get_output_buffer();
                size_t cached_count = npu_encoder_->get_output_buffer_size();
                size_t required = out_layout.time_steps * out_layout.hidden_dim;
                if (cached_output != nullptr && cached_count >= required) {
                    src = cached_output;
                }

                size_t actual_time = std::min(t2, out_layout.time_steps);
                if (actual_time > 0) {
                    const __fp16* final_src = src;
                    if (out_layout.hidden_axis != (out_layout.dims.size() - 1)) {
                        if (!materialize_npu_audio_time_major(src, out_layout, actual_time,
                                                              npu_audio_reorder_scratch_)) {
                            CACTUS_LOG_WARN("npu", "[gemma4-audio] failed to reorder NPU output layout; falling back to CPU audio encoder");
                        } else {
                            final_src = npu_audio_reorder_scratch_.data();
                        }
                    }

                    if (final_src != nullptr) {
                        size_t hidden = gb->input({actual_time, out_layout.hidden_dim}, Precision::FP16);
                        gb->set_input(hidden, final_src, Precision::FP16);
                        return hidden;
                    }
                }
            }

            if (written > 0 && out_layout.hidden_dim != expected_out_dim) {
                CACTUS_LOG_WARN("npu", "[gemma4-audio] NPU output dim mismatch; falling back to CPU audio encoder");
            } else {
                CACTUS_LOG_WARN("npu", "audio encode failed, falling back to CPU");
            }
        }
    }

    size_t hidden = build_sscp(gb, mel_features, num_frames, backend);
    ConformerContext ctx = build_conformer_context(gb, hidden);
    for (uint32_t i = 0; i < config_.audio_num_layers; i++)
        hidden = build_conformer_block(gb, hidden, i, ctx, backend);

    if (config_.audio_output_proj_dims > 0) {
        hidden = gb->matmul(hidden, audio_weights_.output_proj, true, backend);
        hidden = gb->add(hidden, audio_weights_.output_proj_bias);
    }

    return hidden;
}

size_t Gemma4AudioModel::build_audio_projector(CactusGraph* gb, size_t audio_features, ComputeBackend backend) {
    size_t normed = gb->rms_norm(audio_features, audio_proj_norm_ones_node_, config_.audio_rms_norm_eps);
    size_t projected = gb->matmul(normed, audio_weights_.embed_audio_proj, true, backend);
    return gb->scalar_multiply(projected, 1.0f / 16.0f);
}

}
}
