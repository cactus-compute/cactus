#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <filesystem>
#include <iostream>

namespace cactus {
namespace engine {

Lfm2VlModel::Lfm2VlModel() : Model() {
    config_.model_type = Config::ModelType::LFM2;
}

Lfm2VlModel::Lfm2VlModel(const Config& config)
        : Model(config),
            vision_tower_(config),
            language_model_(config) {
    Siglip2Preprocessor::Config preprocessor_config;
    preprocessor_config.patch_size = static_cast<int>(config.vision_patch_size);
    preprocessor_config.downsample_factor = static_cast<int>(config.downsample_factor);
    preprocessor_config.min_tiles = static_cast<int>(config.min_tiles);
    preprocessor_config.max_tiles = static_cast<int>(config.max_tiles);
    preprocessor_config.use_thumbnail = config.use_thumbnail;
    preprocessor_config.min_image_tokens = static_cast<int>(config.min_image_tokens);
    preprocessor_config.max_image_tokens = static_cast<int>(config.max_image_tokens);
    preprocessor_config.max_num_patches = static_cast<int>(config.max_num_patches);
    preprocessor_config.tile_size = static_cast<int>(config.tile_size);
    preprocessor_config.max_pixels_tolerance = config.max_pixels_tolerance;
    preprocessor_config.do_resize = true;
    preprocessor_config.do_rescale = true;
    preprocessor_config.do_normalize = true;
    preprocessor_config.do_convert_rgb = true;
    preprocessor_config.do_image_splitting = config.do_image_splitting;
    preprocessor_config.rescale_factor = config.rescale_factor;
    preprocessor_config.image_mean[0] = config.image_mean;
    preprocessor_config.image_mean[1] = config.image_mean;
    preprocessor_config.image_mean[2] = config.image_mean;
    preprocessor_config.image_std[0] = config.image_std;
    preprocessor_config.image_std[1] = config.image_std;
    preprocessor_config.image_std[2] = config.image_std;
    
    preprocessor_ = Siglip2Preprocessor(preprocessor_config);
}

bool Lfm2VlModel::init(const std::string& model_folder, size_t context_size, const std::string& system_prompt, bool do_warmup) {

    if (!Model::init(model_folder, context_size, system_prompt, false)) {
        return false;
    }
    auto* shared_graph = static_cast<CactusGraph*>(graph_handle_);
    if (!shared_graph) {
        throw std::runtime_error("Shared graph was not initialized for Lfm2VlModel");
    }
    std::string vision_folder = model_folder;
    if (!vision_tower_.init(shared_graph, vision_folder, context_size, "", false)) {
        throw std::runtime_error("Failed to initialize vision tower");
    }

    vision_weights_loaded_ = true;
    if (!language_model_.init(shared_graph, model_folder, context_size, system_prompt, false)) {
        throw std::runtime_error("Failed to initialize language model");
    }

    language_model_.output_weight_node_id_ = output_weight_node_id_;

    language_weights_loaded_ = true;

    if (do_warmup) {
        std::string warmup_text = system_prompt.empty() ? "Hello" : system_prompt;
        auto warmup_tokens = tokenizer_->encode(warmup_text);
        language_model_.decode(warmup_tokens, config_.default_temperature, config_.default_top_p, config_.default_top_k, "");
        language_model_.reset_cache();
    }

    return true;
}

void Lfm2VlModel::reset_cache() {
    Model::reset_cache();
    language_model_.reset_cache();
    image_prefill_completed_ = false;
    last_token_count_ = 0;
}

void Lfm2VlModel::load_weights_to_graph(CactusGraph* gb) {
    namespace fs = std::filesystem;
    fs::path base(model_folder_path_);

    auto resolve_weight = [&](const std::string& primary, const std::string& fallback = "") -> std::string {
        fs::path primary_path = base / primary;
        if (fs::exists(primary_path)) {
            return primary_path.string();
        }
        if (!fallback.empty()) {
            fs::path fallback_path = base / fallback;
            if (fs::exists(fallback_path)) {
                return fallback_path.string();
            }
        }
        return primary_path.string();
    };

    projector_weights_.layer_norm_weight = gb->mmap_weights(resolve_weight("projector_layer_norm.weights"));
    
    projector_weights_.layer_norm_bias = gb->mmap_weights(resolve_weight("projector_layer_norm.bias.weights"));
    
    projector_weights_.linear_1_weight = gb->mmap_weights(resolve_weight("projector_linear_1.weights", "projector_linear1.weights"));
    projector_weights_.linear_1_bias = gb->mmap_weights(resolve_weight("projector_linear_1.bias.weights", "projector_linear1.bias.weights"));
    projector_weights_.linear_2_weight = gb->mmap_weights(resolve_weight("projector_linear_2.weights", "projector_linear2.weights"));
    projector_weights_.linear_2_bias = gb->mmap_weights(resolve_weight("projector_linear_2.bias.weights", "projector_linear2.bias.weights"));
    output_weight_node_id_ = gb->mmap_weights(resolve_weight("output_weight.weights"));
}

size_t Lfm2VlModel::pixel_unshuffle(CactusGraph* gb, size_t hidden_states, 
                                     size_t height, size_t width, size_t channels) {
    
    const size_t factor = config_.downsample_factor;
    const size_t new_height = height / factor;
    const size_t new_width = width / factor;
    size_t step1 = gb->reshape(hidden_states, {1, height, new_width, channels * factor});
    step1 = gb->transposeN(step1, {0, 2, 1, 3});
    size_t step2 = gb->reshape(step1, {1, new_width, new_height, channels * factor * factor});
    size_t result = gb->transposeN(step2, {0, 2, 1, 3});
    return result;
}

size_t Lfm2VlModel::build_multimodal_projector(CactusGraph* gb, size_t image_features,
                                               size_t tile_h, size_t tile_w, ComputeBackend backend) {
    const size_t vision_hidden = config_.vision_embed_dim;
    
    size_t image_features_fp32 = gb->precision_cast(image_features, Precision::FP32);
    size_t unshuffled = pixel_unshuffle(gb, image_features_fp32, tile_h, tile_w, vision_hidden);
    const size_t factor = config_.downsample_factor;
    const size_t new_h = tile_h / factor;
    const size_t new_w = tile_w / factor;
    const size_t in_channels = vision_hidden * factor * factor;
    const size_t seq_len = new_h * new_w;
    size_t flattened = gb->reshape(unshuffled, {seq_len, in_channels});
    size_t flattened_fp16 = gb->precision_cast(flattened, Precision::FP16);
    size_t normalized = gb->layernorm(flattened_fp16, projector_weights_.layer_norm_weight,
                                      projector_weights_.layer_norm_bias, config_.layer_norm_eps);
    size_t hidden = gb->matmul(normalized, projector_weights_.linear_1_weight, true, backend);
    hidden = gb->add(hidden, projector_weights_.linear_1_bias);
    hidden = gb->gelu(hidden);
    size_t output = gb->matmul(hidden, projector_weights_.linear_2_weight, true, backend);
    output = gb->add(output, projector_weights_.linear_2_bias);
    return output;
}

std::vector<float> Lfm2VlModel::get_image_embeddings(const std::string& image_path) {
    return vision_tower_.get_image_embedding(image_path);
}

std::vector<Lfm2VlModel::ProjectedTileFeature> Lfm2VlModel::get_image_features(
    CactusGraph* gb,
    const Siglip2Preprocessor::PreprocessedImage& preprocessed_image,
    ComputeBackend backend) {
    
    size_t vision_output = vision_tower_.forward_vision(gb, preprocessed_image, backend);
    std::vector<ProjectedTileFeature> projected_features;
    projected_features.reserve(preprocessed_image.spatial_shapes.size());
    
    size_t offset = 0;
    for (size_t tile_idx = 0; tile_idx < preprocessed_image.spatial_shapes.size(); ++tile_idx) {
        const auto& shape = preprocessed_image.spatial_shapes[tile_idx];
        const size_t tile_h = shape.first;
        const size_t tile_w = shape.second;
        const size_t tile_tokens = tile_h * tile_w;
        const size_t factor = config_.downsample_factor;
        if (factor == 0) {
            throw std::runtime_error("Downsample factor must be greater than zero");
        }
        if (tile_h % factor != 0 || tile_w % factor != 0) {
            throw std::runtime_error("Tile dimensions must be divisible by downsample factor");
        }
        const size_t new_h = tile_h / factor;
        const size_t new_w = tile_w / factor;
        const size_t projected_tokens = new_h * new_w;
        
        size_t tile_features = gb->slice(vision_output, 0, offset, tile_tokens);
        offset += tile_tokens;
        
        size_t reshaped = gb->reshape(tile_features, {1, tile_h, tile_w, config_.vision_embed_dim});
        size_t projected = build_multimodal_projector(gb, reshaped, tile_h, tile_w, backend);
        ProjectedTileFeature feature{};
        feature.node_id = projected;
        feature.token_count = projected_tokens;
        projected_features.push_back(feature);
        
    }
    
    return projected_features;
}

Lfm2VlModel::MergedEmbeddingResult Lfm2VlModel::merge_image_text_embeddings(
    CactusGraph* gb,
    const std::vector<uint32_t>& tokens,
    const std::vector<std::vector<ProjectedTileFeature>>& image_embedding_nodes,
    std::vector<TextEmbeddingInput>& text_embedding_inputs) {

    text_embedding_inputs.clear();

    Tokenizer* tokenizer = language_model_.get_tokenizer();
    if (!tokenizer) {
        throw std::runtime_error("Tokenizer must be initialized before merging embeddings");
    }

    const uint32_t image_token_id = tokenizer->get_image_token_id();

    auto get_token_id = [tokenizer](const std::string& token_text) -> uint32_t {
        auto encoded = tokenizer->encode(token_text);
        if (encoded.size() != 1) {
            
            throw std::runtime_error("Expected single token encoding for " + token_text);
        }
        return encoded[0];
    };

    const uint32_t image_start_id = get_token_id("<|image_start|>");
    const uint32_t image_end_id = get_token_id("<|image_end|>");

    std::vector<size_t> sequence_nodes;
    sequence_nodes.reserve(tokens.size() + image_embedding_nodes.size());

    std::vector<uint32_t> current_segment;
    current_segment.reserve(tokens.size());

    size_t total_seq_len = 0;
    
    auto flush_segment = [&](void) {
        if (current_segment.empty()) {
            
            return;
        }

        const size_t segment_len = current_segment.size();
        TextEmbeddingInput segment;
        segment.tokens.swap(current_segment);
    segment.input_node = gb->input({segment.tokens.size()}, Precision::FP32);

        const auto& embedding_buffer = gb->get_output_buffer(language_model_.embedding_node_id_);

        for (size_t i = 0; i < embedding_buffer.shape.size(); ++i) {
            (void)i; 
        }

    size_t embedding_node = gb->embedding(language_model_.embedding_node_id_, segment.input_node);

        text_embedding_inputs.push_back(std::move(segment));

        sequence_nodes.push_back(embedding_node);

        total_seq_len += segment_len;

        current_segment.clear();

    };

    size_t token_index = 0;
    size_t image_index = 0;

    while (token_index < tokens.size()) {
        uint32_t token_id = tokens[token_index];
        
        if (token_id == image_start_id) {
            flush_segment();

            if (image_index >= image_embedding_nodes.size()) {
                
                throw std::runtime_error("Encountered <|image_start|> without corresponding image features");
            }

            current_segment.push_back(token_id);
            ++token_index;
            
            const auto& tiles = image_embedding_nodes[image_index];
            size_t tile_index = 0;
            while (token_index < tokens.size()) {
                uint32_t inner_token = tokens[token_index];
                if (inner_token == image_token_id) {
                    flush_segment();
                    
                    if (tile_index >= tiles.size()) {
                        
                        throw std::runtime_error("More <image> placeholders than projected tile features");
                    }

                    const auto& tile = tiles[tile_index++];
                    sequence_nodes.push_back(tile.node_id);
                    
                    total_seq_len += tile.token_count;
                    
                    for (size_t count = 0; count < tile.token_count; ++count) {
                        if (token_index >= tokens.size()) {
                            throw std::runtime_error("Insufficient <image> tokens for projected features");
                        }
                        if (tokens[token_index] != image_token_id) {
                            throw std::runtime_error("Unexpected token encountered within image feature span");
                        }
                        ++token_index;
                        
                    }
                    continue;
                }

                current_segment.push_back(inner_token);
                
                ++token_index;

                if (inner_token == image_end_id) {
                    flush_segment();
                    break;
                }
            }

            if (tile_index != tiles.size()) {
                if (tile_index < tiles.size()) {
                    for (size_t remaining = tile_index; remaining < tiles.size(); ++remaining) {
                        (void)tiles[remaining];
                    }
                }
                throw std::runtime_error("Unused projected tile features remain after processing image block");
            }

            ++image_index;
            
        } else {
            current_segment.push_back(token_id);
            
            ++token_index;
            
        }
    }

    flush_segment();
    if (image_index != image_embedding_nodes.size()) {
        throw std::runtime_error("Not all image features were consumed while merging embeddings");
    }

    if (sequence_nodes.empty()) {
        throw std::runtime_error("Failed to build embedding sequence from provided tokens");
    }

    size_t merged = sequence_nodes[0];
    for (size_t idx = 1; idx < sequence_nodes.size(); ++idx) {
        merged = gb->concat(merged, sequence_nodes[idx], 0);
        
    }

    return MergedEmbeddingResult{merged, total_seq_len};
}

size_t Lfm2VlModel::build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) {
    throw std::runtime_error("build_attention should not be called directly on Lfm2VlModel");
}

size_t Lfm2VlModel::build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const {
    throw std::runtime_error("build_mlp should not be called directly on Lfm2VlModel");
}

size_t Lfm2VlModel::build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) {
    throw std::runtime_error("build_transformer_block should not be called directly on Lfm2VlModel");
}

size_t Lfm2VlModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }
    
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    
    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;
    
    size_t final_hidden = language_model_.forward(gb, tokens, backend, use_cache);
    
    return final_hidden;
}

uint32_t Lfm2VlModel::decode(const std::vector<uint32_t>& tokens,
                               float temperature,
                               float top_p,
                               size_t top_k,
                               const std::string& profile_file) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
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

    image_prefill_completed_ = false;
    last_token_count_ = tokens.size();

    return language_model_.decode(tokens, temperature, top_p, top_k, profile_file);
}

void Lfm2VlModel::prefill(const std::vector<uint32_t>& tokens, size_t chunk_size, const std::string& profile_file) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }

    image_prefill_completed_ = false;
    last_token_count_ = tokens.size();

    language_model_.prefill(tokens, chunk_size, profile_file);
}

Lfm2VlModel::ForwardImageResult Lfm2VlModel::forward_images(
    CactusGraph* gb,
    const std::vector<uint32_t>& tokens,
    const std::vector<std::string>& image_paths,
    ComputeBackend backend,
    bool use_cache) {
    if (!gb) {
        throw std::runtime_error("Graph must be initialized before forwarding");
    }
    if (tokens.empty()) {
        throw std::runtime_error("Token sequence cannot be empty");
    }

    std::vector<std::vector<ProjectedTileFeature>> all_image_embeddings;
    all_image_embeddings.reserve(image_paths.size());
    for (const auto& image_path : image_paths) {
        auto preprocessed = preprocessor_.preprocess_from_file(image_path);
        
        auto image_features = get_image_features(gb, preprocessed, backend);
        
        all_image_embeddings.push_back(std::move(image_features));
    }

    std::vector<TextEmbeddingInput> text_embedding_inputs;
    text_embedding_inputs.reserve(tokens.size() / 4 + 1);
    
    auto merged_embeddings = merge_image_text_embeddings(gb, tokens, all_image_embeddings, text_embedding_inputs);
    if (merged_embeddings.seq_len == 0) {
        throw std::runtime_error("Merged embedding sequence length cannot be zero");
    }

    for (const auto& embedding_input : text_embedding_inputs) {
        if (embedding_input.tokens.empty()) {
            continue;
        }

        std::vector<float> segment_data(embedding_input.tokens.size());
        for (size_t i = 0; i < embedding_input.tokens.size(); ++i) {
            segment_data[i] = static_cast<float>(embedding_input.tokens[i]);
        }
        gb->set_input(embedding_input.input_node, segment_data.data(), Precision::FP32);
    }
    size_t final_hidden = language_model_.forward(gb, merged_embeddings.node_id, merged_embeddings.seq_len, backend, use_cache);
    return ForwardImageResult{final_hidden, merged_embeddings.seq_len};
}

uint32_t Lfm2VlModel::decode_with_images(
    const std::vector<uint32_t>& tokens,
    const std::vector<std::string>& image_paths,
    float temperature,
    float top_p,
    size_t top_k,
    const std::string& profile_file) {

    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }

    if (image_paths.empty()) {

        image_prefill_completed_ = false;
        last_token_count_ = tokens.size();
        return language_model_.decode(tokens, temperature, top_p, top_k, profile_file);
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
    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;
    bool cache_empty = language_model_.is_cache_empty();
    bool need_prefill = cache_empty || !image_prefill_completed_;

    if (!need_prefill && tokens.size() <= last_token_count_) {
        
        reset_cache();
        need_prefill = true;
    }

    size_t seq_len_for_updates = 0;
    size_t final_hidden_node = 0;

    if (need_prefill) {
        auto forward_result = forward_images(gb, tokens, image_paths, backend, true);
        
        final_hidden_node = forward_result.final_hidden_node;
        seq_len_for_updates = forward_result.seq_len;
        image_prefill_completed_ = true;
        last_token_count_ = tokens.size();
    } else {
        size_t delta = tokens.size() - last_token_count_;
        if (delta > tokens.size()) {
            delta = tokens.size();
        }
        if (delta == 0) {
            if (tokens.empty()) {
                throw std::runtime_error("Token sequence cannot be empty for cached decode step");
            }
            delta = 1;
            
        }
        std::vector<uint32_t> incremental_tokens(tokens.end() - delta, tokens.end());
        
        final_hidden_node = language_model_.forward(gb, incremental_tokens, backend, true);
        
        seq_len_for_updates = incremental_tokens.size();
        last_token_count_ = tokens.size();
    }

    auto logits_node_id = gb->matmul(final_hidden_node, language_model_.output_weight_node_id_, true, backend);
    auto sampled_token_id = gb->sample(logits_node_id, temperature, top_p, top_k);
    if (!profile_file.empty()) {
        gb->execute(profile_file);
        
    } else {
        gb->execute();
        
    }

    language_model_.post_execute_updates(gb, seq_len_for_updates);
    language_model_.update_kv_cache(gb, seq_len_for_updates);

    auto* output_ptr = gb->get_output(sampled_token_id);
    return *static_cast<uint32_t*>(output_ptr);
}

}
}