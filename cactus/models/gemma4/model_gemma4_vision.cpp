#include "model_gemma4.h"
#include "../../graph/graph.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

extern "C" {
    #include "../../../libs/stb/stb_image.h"
    #include "../../../libs/stb/stb_image_resize2.h"
}

namespace cactus {
namespace engine {

static std::pair<std::vector<float>, std::vector<float>> compute_2d_rope_tables(
    const Gemma4VisionModel::PreprocessedImage& img, size_t max_patches, size_t head_dim, float theta) {

    size_t half_dim = head_dim / 2;
    size_t freq_per_dim = half_dim / 2;

    std::vector<float> cos_table(max_patches * head_dim, 1.0f);
    std::vector<float> sin_table(max_patches * head_dim, 0.0f);

    for (size_t py = 0; py < img.patch_height; py++) {
        for (size_t px = 0; px < img.patch_width; px++) {
            size_t patch_idx = py * img.patch_width + px;

            for (size_t i = 0; i < freq_per_dim; i++) {
                float inv_freq = 1.0f / std::pow(theta, 2.0f * i / static_cast<float>(half_dim));

                float x_angle = static_cast<float>(px) * inv_freq;
                cos_table[patch_idx * head_dim + i] = std::cos(x_angle);
                cos_table[patch_idx * head_dim + freq_per_dim + i] = std::cos(x_angle);
                sin_table[patch_idx * head_dim + i] = std::sin(x_angle);
                sin_table[patch_idx * head_dim + freq_per_dim + i] = std::sin(x_angle);

                float y_angle = static_cast<float>(py) * inv_freq;
                cos_table[patch_idx * head_dim + half_dim + i] = std::cos(y_angle);
                cos_table[patch_idx * head_dim + half_dim + freq_per_dim + i] = std::cos(y_angle);
                sin_table[patch_idx * head_dim + half_dim + i] = std::sin(y_angle);
                sin_table[patch_idx * head_dim + half_dim + freq_per_dim + i] = std::sin(y_angle);
            }
        }
    }

    return {cos_table, sin_table};
}

static std::pair<std::vector<__fp16>, std::vector<__fp16>> compute_npu_rope_tables(
    const Gemma4VisionModel::PreprocessedImage& img, size_t max_patches, size_t head_dim, float theta) {

    auto [cos_f32, sin_f32] = compute_2d_rope_tables(img, max_patches, head_dim, theta);

    std::vector<__fp16> cos_full(max_patches * head_dim);
    std::vector<__fp16> sin_signed(max_patches * head_dim);

    for (size_t p = 0; p < max_patches; p++) {
        size_t base = p * head_dim;
        for (size_t d = 0; d < head_dim; d++) {
            cos_full[base + d] = static_cast<__fp16>(cos_f32[base + d]);
            sin_signed[base + d] = static_cast<__fp16>(sin_f32[base + d]);
        }
    }

    return {cos_full, sin_signed};
}

Gemma4VisionModel::Gemma4VisionModel() : Model() {}
Gemma4VisionModel::Gemma4VisionModel(const Config& config) : Model(config) {}

void Gemma4VisionModel::load_weights_to_graph(CactusGraph* gb) {
    auto resolve = [&](const std::string& name) -> std::string {
        return model_folder_path_ + "/" + name;
    };

    if (!disable_npu_ && npu::is_npu_available()) {
        std::string npu_path = model_folder_path_ + "/vision_encoder.mlpackage";
        if (std::filesystem::exists(npu_path)) {
            npu_encoder_ = npu::create_encoder();
            if (npu_encoder_ && npu_encoder_->load(npu_path)) {
                use_npu_encoder_ = true;
                std::vector<int> input_shape = npu_encoder_->get_input_shape();
                if (input_shape.empty()) {
                    input_shape = {static_cast<int>(config_.vision_default_output_length *
                        config_.vision_pooling_kernel_size * config_.vision_pooling_kernel_size),
                        static_cast<int>(config_.vision_embed_dim)};
                }
                npu_encoder_->preallocate(input_shape, "hidden_states", "output");
            } else {
                use_npu_encoder_ = false;
                npu_encoder_.reset();
                CACTUS_LOG_WARN("npu", "[gemma4-vision] found vision_encoder.mlpackage but failed to enable NPU vision encoder; using CPU");
            }
        } else {
            CACTUS_LOG_WARN("npu", "[gemma4-vision] vision_encoder.mlpackage not found; using CPU vision encoder");
        }
    } else if (!disable_npu_) {
        CACTUS_LOG_WARN("npu", "[gemma4-vision] NPU backend unavailable on this device; using CPU vision encoder");
    }

    vision_weights_.patch_input_proj = gb->mmap_weights(resolve("vision_patch_embedder_input_proj.weights"));
    vision_weights_.position_table = gb->mmap_weights(resolve("vision_patch_embedder_position_embedding_table.weights"));

    if (!use_npu_encoder_) {
        vision_weights_.layers.resize(config_.vision_num_layers);
        for (uint32_t i = 0; i < config_.vision_num_layers; i++) {
            auto& layer = vision_weights_.layers[i];
            std::string prefix = resolve("vision_encoder_layers_" + std::to_string(i) + "_");

            layer.attn_q_weight = gb->mmap_weights(prefix + "self_attn_q_proj.weights");
            layer.attn_k_weight = gb->mmap_weights(prefix + "self_attn_k_proj.weights");
            layer.attn_v_weight = gb->mmap_weights(prefix + "self_attn_v_proj.weights");
            layer.attn_output_weight = gb->mmap_weights(prefix + "self_attn_o_proj.weights");
            layer.attn_q_norm = gb->mmap_weights(prefix + "self_attn_q_norm.weights");
            layer.attn_k_norm = gb->mmap_weights(prefix + "self_attn_k_norm.weights");
            layer.input_layernorm = gb->mmap_weights(prefix + "input_layernorm.weights");
            layer.post_attention_layernorm = gb->mmap_weights(prefix + "post_attention_layernorm.weights");
            layer.pre_feedforward_layernorm = gb->mmap_weights(prefix + "pre_feedforward_layernorm.weights");
            layer.post_feedforward_layernorm = gb->mmap_weights(prefix + "post_feedforward_layernorm.weights");
            layer.mlp_gate_proj = gb->mmap_weights(prefix + "mlp_gate_proj.weights");
            layer.mlp_up_proj = gb->mmap_weights(prefix + "mlp_up_proj.weights");
            layer.mlp_down_proj = gb->mmap_weights(prefix + "mlp_down_proj.weights");
            layer.layer_scalar = std::filesystem::exists(prefix + "layer_scalar.weights") ? gb->mmap_weights(prefix + "layer_scalar.weights") : 0;
        }
    }

    vision_weights_.embed_vision_proj = gb->mmap_weights(resolve("embed_vision_proj.weights"));

    size_t vision_head_dim = config_.vision_head_dim;
    vision_v_norm_ones_.assign(vision_head_dim, static_cast<__fp16>(1.0f));
    vision_v_norm_ones_node_ = gb->input({vision_head_dim}, Precision::FP16);
    gb->set_external_input(vision_v_norm_ones_node_, vision_v_norm_ones_.data(), Precision::FP16);

    vision_weights_.post_proj_norm = gb->mmap_weights(resolve("embed_vision_post_proj_norm.weights"));

    output_weight_node_id_ = 0;
}

Gemma4VisionModel::PreprocessedImage Gemma4VisionModel::preprocess_image(const std::string& image_path) {
    int w, h, channels;
    unsigned char* data = stbi_load(image_path.c_str(), &w, &h, &channels, 3);
    if (!data)
        throw std::runtime_error("Failed to load image: " + image_path);

    uint32_t patch_size = config_.vision_patch_size;
    uint32_t pooling_k = config_.vision_pooling_kernel_size;
    uint32_t max_patches = config_.vision_default_output_length * pooling_k * pooling_k;
    uint32_t side_mult = pooling_k * patch_size;

    float total_px = static_cast<float>(h) * static_cast<float>(w);
    float target_px = static_cast<float>(max_patches) * static_cast<float>(patch_size * patch_size);
    float factor = std::sqrt(target_px / total_px);

    int target_h = static_cast<int>(std::floor(factor * h / side_mult)) * side_mult;
    int target_w = static_cast<int>(std::floor(factor * w / side_mult)) * side_mult;

    if (target_h == 0) target_h = side_mult;
    if (target_w == 0) target_w = side_mult;

    std::vector<unsigned char> resized_data;
    unsigned char* src = data;
    if (target_h != h || target_w != w) {
        resized_data.resize(target_h * target_w * 3);
        stbir_resize_uint8_linear(data, w, h, 0, resized_data.data(), target_w, target_h, 0,
                                   static_cast<stbir_pixel_layout>(3));
        src = resized_data.data();
    }

    size_t ph = target_h / patch_size;
    size_t pw = target_w / patch_size;
    size_t num_patches = ph * pw;

    std::vector<float> pixels(3 * target_h * target_w);
    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            size_t src_idx = (y * target_w + x) * 3;
            for (int c = 0; c < 3; c++) {
                pixels[c * target_h * target_w + y * target_w + x] =
                    static_cast<float>(src[src_idx + c]) * config_.rescale_factor;
            }
        }
    }

    stbi_image_free(data);

    return PreprocessedImage{
        std::move(pixels),
        static_cast<size_t>(target_h),
        static_cast<size_t>(target_w),
        ph, pw, num_patches
    };
}

size_t Gemma4VisionModel::build_vision_patch_embedding(CactusGraph* gb, const PreprocessedImage& img,
                                                           ComputeBackend backend) {
    uint32_t patch_size = config_.vision_patch_size;
    size_t patch_dim = 3 * patch_size * patch_size;
    size_t num_patches = img.num_patches;
    size_t hidden_size = config_.vision_embed_dim;
    size_t pos_embed_size = config_.vision_position_embedding_size;

    size_t pixel_input = gb->input({3, img.height, img.width}, Precision::FP32);
    gb->set_input(pixel_input, img.pixel_values.data(), Precision::FP32);

    size_t pixel_fp16 = gb->precision_cast(pixel_input, Precision::FP16);
    size_t reshaped = gb->reshape(pixel_fp16, {3, img.patch_height, patch_size, img.patch_width, patch_size});
    size_t permuted = gb->transposeN(reshaped, {1, 3, 2, 4, 0});
    size_t patches = gb->reshape(permuted, {num_patches, patch_dim});

    float mean = config_.image_mean;
    float norm_std = config_.image_std;
    patches = gb->scalar_multiply(gb->scalar_add(patches, -mean), 1.0f / norm_std);

    size_t projected = gb->matmul(patches, vision_weights_.patch_input_proj, true, backend);

    size_t pos_table = gb->reshape(vision_weights_.position_table, {2, pos_embed_size, hidden_size});

    std::vector<float> x_positions(num_patches);
    std::vector<float> y_positions(num_patches);
    for (size_t py = 0; py < img.patch_height; py++) {
        for (size_t px = 0; px < img.patch_width; px++) {
            size_t idx = py * img.patch_width + px;
            x_positions[idx] = static_cast<float>(px);
            y_positions[idx] = static_cast<float>(py);
        }
    }

    size_t x_pos_input = gb->input({num_patches}, Precision::FP32);
    gb->set_input(x_pos_input, x_positions.data(), Precision::FP32);
    size_t y_pos_input = gb->input({num_patches}, Precision::FP32);
    gb->set_input(y_pos_input, y_positions.data(), Precision::FP32);

    size_t x_table = gb->slice(pos_table, 0, 0, 1);
    x_table = gb->reshape(x_table, {pos_embed_size, hidden_size});
    size_t y_table = gb->slice(pos_table, 0, 1, 1);
    y_table = gb->reshape(y_table, {pos_embed_size, hidden_size});

    size_t x_embed = gb->embedding(x_table, x_pos_input);
    size_t y_embed = gb->embedding(y_table, y_pos_input);
    size_t pos_embed = gb->add(x_embed, y_embed);

    size_t pos_embed_fp16 = gb->precision_cast(pos_embed, Precision::FP16);

    return gb->add(projected, pos_embed_fp16);
}

std::pair<size_t, size_t> Gemma4VisionModel::build_2d_rope_nodes(
    CactusGraph* gb, const PreprocessedImage& img, size_t max_patches) {

    size_t head_dim = config_.vision_head_dim;
    auto rope_tables = compute_2d_rope_tables(img, max_patches, head_dim, config_.vision_rope_theta);
    auto& cos_table = rope_tables.first, sin_table = rope_tables.second;

    size_t cos_input = gb->input({max_patches, head_dim}, Precision::FP32);
    gb->set_input(cos_input, cos_table.data(), Precision::FP32);
    size_t cos_node = gb->precision_cast(cos_input, Precision::FP16);
    cos_node = gb->reshape(cos_node, {1, max_patches, 1, head_dim});

    size_t sin_input = gb->input({max_patches, head_dim}, Precision::FP32);
    gb->set_input(sin_input, sin_table.data(), Precision::FP32);
    size_t sin_node = gb->precision_cast(sin_input, Precision::FP16);
    sin_node = gb->reshape(sin_node, {1, max_patches, 1, head_dim});

    return {cos_node, sin_node};
}

size_t Gemma4VisionModel::build_padding_mask(CactusGraph* gb, size_t num_real, size_t max_patches) {
    if (num_real >= max_patches)
        return 0;

    std::vector<float> mask(max_patches * max_patches, 0.0f);
    for (size_t q = 0; q < num_real; q++)
        for (size_t k = 0; k < num_real; k++)
            mask[q * max_patches + k] = 1.0f;
    size_t mask_input = gb->input({max_patches, max_patches}, Precision::FP32);
    gb->set_input(mask_input, mask.data(), Precision::FP32);
    size_t mask_node = gb->precision_cast(mask_input, Precision::FP16);
    return gb->reshape(mask_node, {1, max_patches, max_patches});
}

size_t Gemma4VisionModel::build_vision_attention(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                                     size_t cos_node, size_t sin_node,
                                                     size_t attn_mask_node, ComputeBackend backend) {
    const auto& layer = vision_weights_.layers[layer_idx];
    size_t head_dim = config_.vision_head_dim;
    size_t num_heads = config_.vision_attention_heads;
    size_t kv_heads = config_.vision_kv_heads;

    const auto& input_buf = gb->get_output_buffer(input);
    size_t seq_len = input_buf.shape[0];

    auto q = gb->matmul(input, layer.attn_q_weight, true, backend);
    q = gb->reshape(q, {seq_len * num_heads, head_dim});
    q = gb->rms_norm(q, layer.attn_q_norm, config_.layer_norm_eps);
    q = gb->reshape(q, {1, seq_len, num_heads, head_dim});

    auto k = gb->matmul(input, layer.attn_k_weight, true, backend);
    k = gb->reshape(k, {seq_len * kv_heads, head_dim});
    k = gb->rms_norm(k, layer.attn_k_norm, config_.layer_norm_eps);
    k = gb->reshape(k, {1, seq_len, kv_heads, head_dim});

    auto v_proj = gb->matmul(input, layer.attn_v_weight, true, backend);
    auto v = gb->rms_norm(gb->reshape(v_proj, {seq_len * kv_heads, head_dim}),
                          vision_v_norm_ones_node_, config_.layer_norm_eps);
    v = gb->reshape(v, {1, seq_len, kv_heads, head_dim});

    size_t half_dim = head_dim / 2;

    auto apply_2d_rope = [&](size_t tensor) -> size_t {
        auto first_half = gb->slice(tensor, 3, 0, half_dim);
        auto second_half = gb->slice(tensor, 3, half_dim, half_dim);

        auto rot_first_a = gb->slice(first_half, 3, 0, half_dim / 2);
        auto rot_first_b = gb->slice(first_half, 3, half_dim / 2, half_dim / 2);
        auto neg_b = gb->scalar_multiply(rot_first_b, -1.0f);

        size_t cos_x = gb->slice(cos_node, 3, 0, half_dim);
        size_t sin_x = gb->slice(sin_node, 3, 0, half_dim);

        auto first_rotated = gb->add(
            gb->multiply(first_half, cos_x),
            gb->multiply(gb->concat(neg_b, rot_first_a, 3), sin_x));

        auto rot_second_a = gb->slice(second_half, 3, 0, half_dim / 2);
        auto rot_second_b = gb->slice(second_half, 3, half_dim / 2, half_dim / 2);
        auto neg_sb = gb->scalar_multiply(rot_second_b, -1.0f);

        size_t cos_y = gb->slice(cos_node, 3, half_dim, half_dim);
        size_t sin_y = gb->slice(sin_node, 3, half_dim, half_dim);

        auto second_rotated = gb->add(
            gb->multiply(second_half, cos_y),
            gb->multiply(gb->concat(neg_sb, rot_second_a, 3), sin_y));

        return gb->concat(first_rotated, second_rotated, 3);
    };

    auto q_rot = apply_2d_rope(q);
    auto k_rot = apply_2d_rope(k);

    size_t attn;
    if (attn_mask_node != 0)
        attn = gb->attention_masked(q_rot, k_rot, v, attn_mask_node, 1.f, false, backend);
    else
        attn = gb->attention(q_rot, k_rot, v, 1.f, false, backend);

    return gb->matmul(gb->reshape(attn, {seq_len, num_heads * head_dim}),
                      layer.attn_output_weight, true, backend);
}

size_t Gemma4VisionModel::build_vision_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx,
                                               ComputeBackend backend) {
    const auto& layer = vision_weights_.layers[layer_idx];
    auto gate = gb->gelu(gb->matmul(input, layer.mlp_gate_proj, true, backend));
    auto up = gb->matmul(input, layer.mlp_up_proj, true, backend);
    return gb->matmul(gb->multiply(gate, up), layer.mlp_down_proj, true, backend);
}

size_t Gemma4VisionModel::build_vision_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                                             size_t cos_node, size_t sin_node,
                                                             size_t attn_mask_node, ComputeBackend backend) {
    const auto& layer = vision_weights_.layers[layer_idx];

    auto normed = gb->rms_norm(hidden, layer.input_layernorm, config_.layer_norm_eps);
    auto attn_raw = build_vision_attention(gb, normed, layer_idx, cos_node, sin_node, attn_mask_node, backend);
    auto attn = gb->rms_norm(attn_raw, layer.post_attention_layernorm, config_.layer_norm_eps);
    auto residual = gb->add(hidden, attn);

    auto pre_mlp = gb->rms_norm(residual, layer.pre_feedforward_layernorm, config_.layer_norm_eps);
    auto mlp_raw = build_vision_mlp(gb, pre_mlp, layer_idx, backend);
    auto mlp = gb->rms_norm(mlp_raw, layer.post_feedforward_layernorm, config_.layer_norm_eps);

    auto out = gb->add(residual, mlp);

    if (layer.layer_scalar != 0)
        out = gb->multiply(out, layer.layer_scalar);

    return out;
}

size_t Gemma4VisionModel::build_vision_pooler(CactusGraph* gb, size_t hidden, const PreprocessedImage& img,
                                                  ComputeBackend backend) {
    size_t k = config_.vision_pooling_kernel_size;
    size_t output_length = config_.vision_default_output_length;
    size_t num_patches = img.num_patches;
    size_t k_squared = k * k;

    if (num_patches == output_length)
        return hidden;

    size_t max_x = img.patch_width;
    size_t max_y = img.patch_height;

    std::vector<float> pool_weights(num_patches * output_length, 0.0f);
    std::vector<bool> valid_bins(output_length, false);
    for (size_t py = 0; py < max_y; py++) {
        for (size_t px = 0; px < max_x; px++) {
            size_t patch_idx = py * max_x + px;
            size_t kx = px / k;
            size_t ky = py / k;
            size_t kernel_idx = kx + (max_x / k) * ky;
            if (kernel_idx < output_length) {
                pool_weights[kernel_idx * num_patches + patch_idx] = 1.0f / static_cast<float>(k_squared);
                valid_bins[kernel_idx] = true;
            }
        }
    }

    size_t weights_node = gb->input({output_length, num_patches}, Precision::FP32);
    gb->set_input(weights_node, pool_weights.data(), Precision::FP32);
    size_t weights_fp16 = gb->precision_cast(weights_node, Precision::FP16);

    size_t pooled = gb->matmul(weights_fp16, hidden, false, backend);

    size_t valid_count = 0;
    for (bool v : valid_bins) if (v) valid_count++;

    if (valid_count == output_length)
        return pooled;

    std::vector<size_t> valid_slices;
    for (size_t i = 0; i < output_length; i++) {
        if (valid_bins[i]) {
            auto slice = gb->slice(pooled, 0, i, 1);
            valid_slices.push_back(slice);
        }
    }

    size_t stripped = valid_slices[0];
    for (size_t i = 1; i < valid_slices.size(); i++)
        stripped = gb->concat(stripped, valid_slices[i], 0);

    return stripped;
}

size_t Gemma4VisionModel::forward_vision(CactusGraph* gb, const PreprocessedImage& img, ComputeBackend backend) {
    size_t hidden = build_vision_patch_embedding(gb, img, backend);

    size_t num_real = img.num_patches;
    size_t hidden_size = config_.vision_embed_dim;
    uint32_t pooling_k = config_.vision_pooling_kernel_size;
    size_t max_patches = config_.vision_default_output_length * pooling_k * pooling_k;
    bool can_use_npu_path = use_npu_encoder_ && npu_encoder_ && npu_encoder_->is_available();
    if (can_use_npu_path) {
        std::vector<int> npu_input_shape = npu_encoder_->get_input_shape();
        if (npu_input_shape.size() >= 2) {
            size_t a = static_cast<size_t>(npu_input_shape[0]);
            size_t b = static_cast<size_t>(npu_input_shape[1]);
            if (b == hidden_size) {
                max_patches = a;
            } else if (a == hidden_size) {
                max_patches = b;
            } else if (a > 0) {
                max_patches = a;
            }
        }
    }
    if (num_real > max_patches) {
        if (can_use_npu_path) {
            CACTUS_LOG_WARN("npu", "[gemma4-vision] image has more patches than NPU encoder supports; falling back to CPU vision encoder");
            can_use_npu_path = false;
            max_patches = config_.vision_default_output_length * pooling_k * pooling_k;
        }
    }
    size_t num_padding = max_patches - num_real;

    if (num_padding > 0) {
        std::vector<__fp16> zeros(num_padding * hidden_size, static_cast<__fp16>(0.0f));
        size_t pad_node = gb->input({num_padding, hidden_size}, Precision::FP16);
        gb->set_input(pad_node, zeros.data(), Precision::FP16);
        hidden = gb->concat(hidden, pad_node, 0);
    }

    auto [cos_node, sin_node] = build_2d_rope_nodes(gb, img, max_patches);
    size_t attn_mask_node = build_padding_mask(gb, num_real, max_patches);

    if (can_use_npu_path) {
        gb->execute();

        const auto& h_buf = gb->get_output_buffer(hidden);
        if (h_buf.precision != Precision::FP16)
            throw std::runtime_error("[gemma4-vision] expected FP16 hidden output for NPU path");
        std::vector<__fp16> hidden_fp16(max_patches * hidden_size);
        memcpy(hidden_fp16.data(), h_buf.data_as<__fp16>(), max_patches * hidden_size * sizeof(__fp16));

        size_t head_dim = config_.vision_head_dim;
        auto [cos_fp16, sin_fp16] = compute_npu_rope_tables(img, max_patches, head_dim, config_.vision_rope_theta);

        std::vector<__fp16> additive_mask(max_patches * max_patches, static_cast<__fp16>(0.0f));
        if (num_padding > 0) {
            for (size_t q = 0; q < max_patches; q++)
                for (size_t k = 0; k < max_patches; k++)
                    if (q >= num_real || k >= num_real)
                        additive_mask[q * max_patches + k] = static_cast<__fp16>(-65504.0f);
        }

        std::vector<npu::NPUNamedInput> inputs = {
            {"hidden_states", hidden_fp16.data(), {static_cast<int>(max_patches), static_cast<int>(hidden_size)}},
            {"cos_full", cos_fp16.data(), {static_cast<int>(max_patches), 1, static_cast<int>(head_dim)}},
            {"sin_signed", sin_fp16.data(), {static_cast<int>(max_patches), 1, static_cast<int>(head_dim)}},
            {"attention_mask", additive_mask.data(), {1, static_cast<int>(max_patches), static_cast<int>(max_patches)}},
        };

        std::vector<int> out_shape = npu_encoder_->get_output_shape();
        size_t out_elements = 1;
        for (int d : out_shape) out_elements *= d;
        size_t required = max_patches * hidden_size;
        if (out_elements < required)
            out_elements = required;

        std::vector<__fp16> npu_output(out_elements);
        size_t written = npu_encoder_->encode_multimodal_input(inputs, npu_output.data(), "output");

        if (written >= required) {
            hidden = gb->input({max_patches, hidden_size}, Precision::FP16);
            gb->set_input(hidden, npu_output.data(), Precision::FP16);
        } else {
            CACTUS_LOG_WARN("npu", "[gemma4-vision] encode_multimodal_input returned insufficient output, falling back to CPU");
            for (uint32_t i = 0; i < config_.vision_num_layers; i++)
                hidden = build_vision_transformer_block(gb, hidden, i, cos_node, sin_node, attn_mask_node, backend);
        }
    } else {
        for (uint32_t i = 0; i < config_.vision_num_layers; i++)
            hidden = build_vision_transformer_block(gb, hidden, i, cos_node, sin_node, attn_mask_node, backend);
    }

    if (num_padding > 0)
        hidden = gb->slice(hidden, 0, 0, num_real);

    return build_vision_pooler(gb, hidden, img, backend);
}

size_t Gemma4VisionModel::build_vision_projector(CactusGraph* gb, size_t vision_features, ComputeBackend backend) {
    size_t projected = gb->matmul(vision_features, vision_weights_.embed_vision_proj, true, backend);
    return gb->rms_norm(projected, vision_weights_.post_proj_norm, config_.layer_norm_eps);
}

}
}
