#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <stdexcept>
#include <utility>
#include <iostream>

namespace cactus {
namespace engine {

Siglip2VisionModel::Siglip2VisionModel() : Model() {
    config_.model_type = Config::ModelType::SIGLIP2;
    
}

Siglip2VisionModel::Siglip2VisionModel(const Config& cfg) : Model(cfg) {
    Siglip2Preprocessor::Config preprocessor_config;
    preprocessor_config.patch_size = static_cast<int>(config_.vision_patch_size);
    preprocessor_config.downsample_factor = static_cast<int>(config_.downsample_factor);
    preprocessor_config.min_tiles = static_cast<int>(config_.min_tiles);
    preprocessor_config.max_tiles = static_cast<int>(config_.max_tiles);
    preprocessor_config.use_thumbnail = config_.use_thumbnail;
    preprocessor_config.min_image_tokens = static_cast<int>(config_.min_image_tokens);
    preprocessor_config.max_image_tokens = static_cast<int>(config_.max_image_tokens);
    preprocessor_config.max_num_patches = static_cast<int>(config_.max_num_patches);
    preprocessor_config.tile_size = static_cast<int>(config_.tile_size);
    preprocessor_config.max_pixels_tolerance = config_.max_pixels_tolerance;
    preprocessor_config.do_resize = true;
    preprocessor_config.do_rescale = true;
    preprocessor_config.do_normalize = true;
    preprocessor_config.do_convert_rgb = true;
    preprocessor_config.do_image_splitting = config_.do_image_splitting;
    preprocessor_config.rescale_factor = config_.rescale_factor;
    preprocessor_config.image_mean[0] = config_.image_mean;
    preprocessor_config.image_mean[1] = config_.image_mean;
    preprocessor_config.image_mean[2] = config_.image_mean;
    preprocessor_config.image_std[0] = config_.image_std;
    preprocessor_config.image_std[1] = config_.image_std;
    preprocessor_config.image_std[2] = config_.image_std;
    
    preprocessor_ = Siglip2Preprocessor(preprocessor_config);
    
}

void Siglip2VisionModel::load_weights_to_graph(CactusGraph* gb) {
    vision_weight_nodes_.vision_layers.resize(config_.vision_num_layers);
    std::string base = model_folder_path_ + "/";

    vision_weight_nodes_.patch_embedding_weight = gb->mmap_weights(base + "vision_patch_embedding.weights");
    vision_weight_nodes_.patch_embedding_bias = gb->mmap_weights(base + "vision_patch_embedding.bias.weights");

    vision_weight_nodes_.position_embedding = gb->mmap_weights(base + "vision_position_embedding.weights");
    
    vision_weight_nodes_.post_layernorm_weight = gb->mmap_weights(base + "vision_post_layernorm.weights");
    vision_weight_nodes_.post_layernorm_bias = gb->mmap_weights(base + "vision_post_layernorm.bias.weights");
    
    for (uint32_t i = 0; i < vision_weight_nodes_.vision_layers.size(); ++i) {
        auto& layer = vision_weight_nodes_.vision_layers[i];
        std::string prefix = base + "vision_layer_" + std::to_string(i) + "_";
    
        layer.attn_q_weight = gb->mmap_weights(prefix + "self_attn_q.weights");
        layer.attn_q_bias = gb->mmap_weights(prefix + "self_attn_q.bias.weights");
        layer.attn_k_weight = gb->mmap_weights(prefix + "self_attn_k.weights");
        layer.attn_k_bias = gb->mmap_weights(prefix + "self_attn_k.bias.weights");
        layer.attn_v_weight = gb->mmap_weights(prefix + "self_attn_v.weights");
        layer.attn_v_bias = gb->mmap_weights(prefix + "self_attn_v.bias.weights");
        layer.attn_output_weight = gb->mmap_weights(prefix + "self_attn_out.weights");
        layer.attn_output_bias = gb->mmap_weights(prefix + "self_attn_out.bias.weights");
    
        layer.layer_norm1_weight = gb->mmap_weights(prefix + "layer_norm1.weights");
        layer.layer_norm1_bias = gb->mmap_weights(prefix + "layer_norm1.bias.weights");
        layer.layer_norm2_weight = gb->mmap_weights(prefix + "layer_norm2.weights");
        layer.layer_norm2_bias = gb->mmap_weights(prefix + "layer_norm2.bias.weights");
    
        layer.mlp_fc1_weight = gb->mmap_weights(prefix + "ffn_fc1.weights");
        layer.mlp_fc1_bias = gb->mmap_weights(prefix + "ffn_fc1.bias.weights");
        layer.mlp_fc2_weight = gb->mmap_weights(prefix + "ffn_fc2.weights");
        layer.mlp_fc2_bias = gb->mmap_weights(prefix + "ffn_fc2.bias.weights");
    }
}

Siglip2VisionModel::VisionEmbeddingResult Siglip2VisionModel::build_vision_embeddings(
    CactusGraph* gb,
    const Siglip2Preprocessor::PreprocessedImage& preprocessed_image,
    ComputeBackend backend) {
    const int num_tiles = preprocessed_image.num_tiles;
    const int max_patches = preprocessed_image.max_patches_per_tile;
    const int patch_dim = preprocessed_image.patch_dim;

    const size_t expected_size = static_cast<size_t>(num_tiles) * static_cast<size_t>(max_patches) *
                                 static_cast<size_t>(patch_dim);
    if (preprocessed_image.pixel_values.size() != expected_size) {
        throw std::runtime_error(
            "Pixel values size mismatch: expected " + std::to_string(expected_size) +
            " (tiles=" + std::to_string(num_tiles) + " * max_patches=" + std::to_string(max_patches) +
            " * patch_dim=" + std::to_string(patch_dim) + ") but got " +
            std::to_string(preprocessed_image.pixel_values.size()));
    }
    for (size_t i = 0; i < std::min<size_t>(100, preprocessed_image.pixel_values.size()); ++i) {
        float val = preprocessed_image.pixel_values[i];
        if (std::isnan(val) || std::isinf(val)) {
            throw std::runtime_error(
                "Invalid value in pixel_values at index " + std::to_string(i) + ": " + std::to_string(val));
        }
    }

    const BufferDesc& weight_buffer = gb->get_output_buffer(vision_weight_nodes_.patch_embedding_weight);
    float original_weight_buffer_quantization_scale = weight_buffer.quantization_scale;

    size_t reshaped_weight = gb->reshape(
        vision_weight_nodes_.patch_embedding_weight,
        {static_cast<size_t>(config_.vision_embed_dim), static_cast<size_t>(patch_dim)});
    gb->set_quantization_scale(reshaped_weight, original_weight_buffer_quantization_scale);

    size_t patch_bias = vision_weight_nodes_.patch_embedding_bias;
    std::vector<size_t> tile_embeddings;
    tile_embeddings.reserve(static_cast<size_t>(num_tiles));

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const auto& shape = preprocessed_image.spatial_shapes[tile_idx];
        const int tile_h = shape.first;
        const int tile_w = shape.second;
        const int actual_patches = tile_h * tile_w;
        if (actual_patches <= 0) {
            
            continue;
        }

        const float* tile_data = preprocessed_image.pixel_values.data() +
                                 static_cast<size_t>(tile_idx) * static_cast<size_t>(max_patches) *
                                     static_cast<size_t>(patch_dim);

        size_t tile_input_fp32 = gb->input(
            {static_cast<size_t>(actual_patches), static_cast<size_t>(patch_dim)}, Precision::FP32);
        gb->set_input(tile_input_fp32, tile_data, Precision::FP32);
        size_t tile_input = gb->precision_cast(tile_input_fp32, Precision::FP16);
        size_t tile_patch = gb->matmul(tile_input, reshaped_weight, true, backend);
        size_t tile_bias = gb->add(tile_patch, patch_bias);
        size_t tile_pos = gb->bilinear_interpolation(
            vision_weight_nodes_.position_embedding,
            static_cast<size_t>(tile_h),
            static_cast<size_t>(tile_w));
        size_t tile_pos_cast = gb->precision_cast(tile_pos, Precision::FP16);
        size_t tile_embed = gb->add(tile_bias, tile_pos_cast);
        tile_embeddings.push_back(tile_embed);
    }

    if (tile_embeddings.empty()) {
        throw std::runtime_error("No valid tiles produced embeddings in build_vision_embeddings");
    }
    auto concat_nodes = [&](const std::vector<size_t>& nodes) {
        if (nodes.empty()) {
            throw std::runtime_error("Attempted to concatenate an empty node list");
        }
        size_t combined = nodes.front();
        for (size_t i = 1; i < nodes.size(); ++i) {
            combined = gb->concat(combined, nodes[i], /*axis=*/0);
        }
        return combined;
    };

    size_t embeddings = concat_nodes(tile_embeddings);
    return VisionEmbeddingResult{embeddings, std::move(tile_embeddings)};
}

size_t Siglip2VisionModel::build_vision_attention(CactusGraph* gb, size_t hidden_states, 
                                                  uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = vision_weight_nodes_.vision_layers[layer_idx];
    
    size_t q = gb->matmul(hidden_states, layer.attn_q_weight, true, backend);
    q = gb->add(q, layer.attn_q_bias);
    size_t k = gb->matmul(hidden_states, layer.attn_k_weight, true, backend);
    k = gb->add(k, layer.attn_k_bias);
    size_t v = gb->matmul(hidden_states, layer.attn_v_weight, true, backend);
    v = gb->add(v, layer.attn_v_bias);
    const size_t num_heads = static_cast<size_t>(config_.vision_attention_heads);
    const size_t head_dim = static_cast<size_t>(config_.vision_embed_dim / config_.vision_attention_heads);
    const auto& q_buf = gb->get_output_buffer(q);
    size_t seq_len = q_buf.shape[0];
    
    size_t q_4d = gb->reshape(q, {1, seq_len, num_heads, head_dim});
    size_t k_4d = gb->reshape(k, {1, seq_len, num_heads, head_dim});
    size_t v_4d = gb->reshape(v, {1, seq_len, num_heads, head_dim});
    
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    size_t attn_output = gb->attention(q_4d, k_4d, v_4d, scale, false, backend);
    size_t attn_2d = gb->reshape(attn_output, {seq_len, num_heads * head_dim});
    size_t output = gb->matmul(attn_2d, layer.attn_output_weight, true, backend);
    output = gb->add(output, layer.attn_output_bias);
    return output;
}

size_t Siglip2VisionModel::build_vision_mlp(CactusGraph* gb, size_t hidden_states, 
                                           uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = vision_weight_nodes_.vision_layers[layer_idx];
    
    size_t fc1_output = gb->matmul(hidden_states, layer.mlp_fc1_weight, true, backend);
    fc1_output = gb->add(fc1_output, layer.mlp_fc1_bias);
    size_t activated = gb->gelu(fc1_output);
    size_t fc2_output = gb->matmul(activated, layer.mlp_fc2_weight, true, backend);
    fc2_output = gb->add(fc2_output, layer.mlp_fc2_bias);
    return fc2_output;
}

size_t Siglip2VisionModel::build_vision_transformer_layer(CactusGraph* gb, size_t hidden_states, 
                                                          uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = vision_weight_nodes_.vision_layers[layer_idx];
    
    size_t residual = hidden_states;
    size_t normalized = gb->layernorm(hidden_states, layer.layer_norm1_weight, 
                                      layer.layer_norm1_bias, config_.layer_norm_eps);
    size_t attn_output = build_vision_attention(gb, normalized, layer_idx, backend);
    hidden_states = gb->add(residual, attn_output);
    residual = hidden_states;
    normalized = gb->layernorm(hidden_states, layer.layer_norm2_weight, 
                               layer.layer_norm2_bias, config_.layer_norm_eps);
    size_t mlp_output = build_vision_mlp(gb, normalized, layer_idx, backend);
    hidden_states = gb->add(residual, mlp_output);
    return hidden_states;
}

size_t Siglip2VisionModel::forward_vision(
    CactusGraph* gb,
    const Siglip2Preprocessor::PreprocessedImage& preprocessed_image,
    ComputeBackend backend) {
    auto embedding_result = build_vision_embeddings(gb, preprocessed_image, backend);
    auto concat_nodes = [&](const std::vector<size_t>& nodes) {
        if (nodes.empty()) {
            throw std::runtime_error("Attempted to concatenate an empty node list in forward_vision");
        }
        size_t combined = nodes.front();
        for (size_t i = 1; i < nodes.size(); ++i) {
            combined = gb->concat(combined, nodes[i], /*axis=*/0);
        }
        return combined;
    };

    std::vector<size_t> tile_outputs;
    tile_outputs.reserve(embedding_result.tile_embeddings.size());

    for (size_t tile_idx = 0; tile_idx < embedding_result.tile_embeddings.size(); ++tile_idx) {
        size_t hidden_states = embedding_result.tile_embeddings[tile_idx];
        
        for (uint32_t layer_idx = 0; layer_idx < config_.vision_num_layers; ++layer_idx) {
            hidden_states = build_vision_transformer_layer(gb, hidden_states, layer_idx, backend);
            
        }

        hidden_states = gb->layernorm(hidden_states,
                                       vision_weight_nodes_.post_layernorm_weight,
                                       vision_weight_nodes_.post_layernorm_bias,
                                       config_.layer_norm_eps);
        tile_outputs.push_back(hidden_states);
        
    }

    if (tile_outputs.empty()) {
        throw std::runtime_error("No tile outputs generated in forward_vision");
    }

    size_t combined_output = concat_nodes(tile_outputs);
    return combined_output;
}

size_t Siglip2VisionModel::forward_vision(const Siglip2Preprocessor::PreprocessedImage& preprocessed_image) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;
    return forward_vision(gb, preprocessed_image, backend);
}

std::vector<float> Siglip2VisionModel::get_image_features(const std::string& image_path) {
    auto preprocessed = preprocessor_.preprocess_from_file(image_path);
    
    return get_image_features(preprocessed);
}

size_t Siglip2VisionModel::get_image_features_node(const Siglip2Preprocessor::PreprocessedImage& preprocessed_image) {
    return forward_vision(preprocessed_image);
}

std::vector<float> Siglip2VisionModel::get_image_features(const Siglip2Preprocessor::PreprocessedImage& preprocessed_image) {
    size_t last_hidden_state = forward_vision(preprocessed_image);
    
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->execute();
    
    const auto& output_buf = gb->get_output_buffer(last_hidden_state);
    size_t total_elements = 1;
    for (auto dim : output_buf.shape) {
        total_elements *= dim;
    }
    
    std::vector<float> features(total_elements);
    void* output_data = gb->get_output(last_hidden_state);
    const float* output_ptr = static_cast<const float*>(output_data);
    std::copy(output_ptr, output_ptr + total_elements, features.begin());
    return features;
}

size_t Siglip2VisionModel::forward(const std::vector<uint32_t>&, bool) {return 0;}

size_t Siglip2VisionModel::build_attention(CactusGraph*, size_t, uint32_t,
                                           ComputeBackend, bool, size_t) {
    return 0;
}

size_t Siglip2VisionModel::build_mlp(CactusGraph*, size_t, uint32_t,
                                     ComputeBackend) const {
    return 0;
}

size_t Siglip2VisionModel::build_transformer_block(CactusGraph*, size_t, uint32_t,
                                                   ComputeBackend, bool, size_t) {
    return 0;
}

}
}
