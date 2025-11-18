#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#define STBI_NO_BMP
#define STBI_NO_PSD
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_TGA

#include "engine.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace cactus {
namespace engine {

Siglip2Preprocessor::PreprocessedImage::~PreprocessedImage() {
    pixel_values.clear();
    pixel_attention_mask.clear();
}

Siglip2Preprocessor::Siglip2Preprocessor(const Config& config)
    : config_(config) {}

Siglip2Preprocessor::Siglip2Preprocessor() : config_() {}

Siglip2Preprocessor::~Siglip2Preprocessor() = default;

int Siglip2Preprocessor::round_by_factor(int number, int factor) {
    if (factor == 0) {
        return number;
    }
    double scaled = static_cast<double>(number) / static_cast<double>(factor);
    long long rounded = std::llround(scaled);
    
    return static_cast<int>(rounded * factor);
}

std::pair<int,int> Siglip2Preprocessor::smart_resize(int height, int width) {
    const int total_factor = config_.patch_size * config_.downsample_factor;
    const int64_t min_pixels = static_cast<int64_t>(config_.min_image_tokens) *
                               config_.patch_size * config_.patch_size *
                               config_.downsample_factor * config_.downsample_factor;
    const int64_t max_pixels = static_cast<int64_t>(config_.max_image_tokens) *
                               config_.patch_size * config_.patch_size *
                               config_.downsample_factor * config_.downsample_factor;

    int h_bar = std::max(total_factor, round_by_factor(height, total_factor));
    int w_bar = std::max(total_factor, round_by_factor(width, total_factor));
    

    if (static_cast<int64_t>(h_bar) * static_cast<int64_t>(w_bar) > max_pixels) {
        double beta = std::sqrt((static_cast<double>(height) * static_cast<double>(width)) /
                                static_cast<double>(max_pixels));
        h_bar = std::max(total_factor,
                         static_cast<int>(std::floor(height / beta / total_factor)) * total_factor);
        w_bar = std::max(total_factor,
                         static_cast<int>(std::floor(width / beta / total_factor)) * total_factor);
    } else if (static_cast<int64_t>(h_bar) * static_cast<int64_t>(w_bar) < min_pixels) {
        double beta = std::sqrt(static_cast<double>(min_pixels) /
                                (static_cast<double>(height) * static_cast<double>(width)));
        h_bar = std::max(total_factor,
                         static_cast<int>(std::ceil(height * beta / total_factor)) * total_factor);
        w_bar = std::max(total_factor,
                         static_cast<int>(std::ceil(width * beta / total_factor)) * total_factor);
    }

    return {w_bar, h_bar};
}


bool Siglip2Preprocessor::is_image_too_large(int height, int width) {
    const int total_factor = config_.patch_size * config_.downsample_factor;
    const int h_bar = std::max(config_.patch_size, round_by_factor(height, total_factor));
    const int w_bar = std::max(config_.patch_size, round_by_factor(width, total_factor));

    const int64_t pixels = static_cast<int64_t>(h_bar) * static_cast<int64_t>(w_bar);
    const double max_pixels = static_cast<double>(config_.max_image_tokens) *
                              config_.patch_size * config_.patch_size *
                              config_.downsample_factor * config_.downsample_factor *
                              config_.max_pixels_tolerance;

    bool result = static_cast<double>(pixels) > max_pixels;
    
    return result;
}


std::pair<int, int> Siglip2Preprocessor::find_closest_aspect_ratio(float aspect_ratio, int width, int height) {
    float best_ratio_diff = std::numeric_limits<float>::infinity();
    std::pair<int, int> best_ratio = {1, 1};
    int area = width * height;

    std::vector<std::pair<int, int>> target_ratios;
    for (int n = config_.min_tiles; n <= config_.max_tiles; ++n) {
        for (int w = 1; w <= n; ++w) {
            for (int h = 1; h <= n; ++h) {
                int total_tiles = w * h;
                if (total_tiles >= config_.min_tiles && total_tiles <= config_.max_tiles) {
                    target_ratios.push_back({w, h});
                }
            }
        }
    }

    std::sort(target_ratios.begin(), target_ratios.end());
    target_ratios.erase(std::unique(target_ratios.begin(), target_ratios.end()), target_ratios.end());
    std::sort(target_ratios.begin(), target_ratios.end(), [](const auto& a, const auto& b) {
        return (a.first * a.second) < (b.first * b.second);
    });

    for (const auto& ratio : target_ratios) {
        float target_aspect_ratio = static_cast<float>(ratio.first) / ratio.second;
        float ratio_diff = std::abs(aspect_ratio - target_aspect_ratio);

        if (ratio_diff < best_ratio_diff) {
            best_ratio_diff = ratio_diff;
            best_ratio = ratio;
        } else if (ratio_diff == best_ratio_diff) {
            int target_area = config_.tile_size * config_.tile_size * ratio.first * ratio.second;
            if (area > 0.5f * target_area) {
                best_ratio = ratio;
            }
        }
    }

    
    return best_ratio;
}

std::pair<int,int> Siglip2Preprocessor::get_grid_layout(int height, int width) {
    float aspect_ratio = (float)width / (float)height;
    auto [grid_width, grid_height] = find_closest_aspect_ratio(aspect_ratio, width, height);

    int target_width  = config_.tile_size * grid_width;
    int target_height = config_.tile_size * grid_height;
    
    return {target_width, target_height};
}

Siglip2Preprocessor::SpatialShapeResult Siglip2Preprocessor::compute_spatial_shapes(int height, int width) {
    if (height <= 0 || width <= 0) {
        throw std::runtime_error("Image dimensions must be positive");
    }

    if (config_.patch_size <= 0) {
        throw std::runtime_error("Patch size must be positive");
    }

    const int patch = config_.patch_size;
    auto [resized_width, resized_height] = smart_resize(height, width);
    const bool should_split = config_.do_image_splitting && is_image_too_large(height, width);
    

    SpatialShapeResult result;
    result.grid_rows = 1;
    result.grid_cols = 1;

    if (should_split) {
        if (config_.tile_size % patch != 0) {
            throw std::runtime_error("Tile size must be divisible by patch size");
        }

        auto [grid_target_width, grid_target_height] = get_grid_layout(height, width);
        result.grid_cols = grid_target_width / config_.tile_size;
        result.grid_rows = grid_target_height / config_.tile_size;
    

        const int patches_per_tile_side = config_.tile_size / patch;
        const auto tile_shape = std::make_pair(patches_per_tile_side, patches_per_tile_side);

        result.shapes.reserve(static_cast<size_t>(result.grid_rows) * result.grid_cols + 1);
        for (int row = 0; row < result.grid_rows; ++row) {
            for (int col = 0; col < result.grid_cols; ++col) {
                result.shapes.push_back(tile_shape);
                
            }
        }

        if (config_.use_thumbnail && result.grid_rows * result.grid_cols != 1) {
            if (resized_height % patch != 0 || resized_width % patch != 0) {
                throw std::runtime_error("Resized thumbnail dimensions must be divisible by patch size");
            }
            result.shapes.emplace_back(resized_height / patch, resized_width / patch);
            
        }
    } else {
        int target_width = resized_width;
        int target_height = resized_height;
        if (!config_.do_resize) {
            target_width = width;
            target_height = height;
        }

        if (target_height % patch != 0 || target_width % patch != 0) {
            throw std::runtime_error("Target dimensions must be divisible by patch size");
        }

        result.shapes.emplace_back(target_height / patch, target_width / patch);
        
    }

    return result;
}


Siglip2Preprocessor::PreprocessedImage Siglip2Preprocessor::preprocess_from_file(const std::string& image_path) {
    int width, height, channels;
    unsigned char* img_data = stbi_load(image_path.c_str(), &width, &height, &channels, 0);
    
    if (!img_data) {
        throw std::runtime_error("Failed to load image: " + image_path + " - " + std::string(stbi_failure_reason()));
    }

    PreprocessedImage result;
    result = preprocess_from_memory(img_data, width, height, channels);
    
    stbi_image_free(img_data);
    return result;
}

Siglip2Preprocessor::PreprocessedImage Siglip2Preprocessor::preprocess_from_memory(
    const unsigned char* img_data, int width, int height, int channels) {
    if (!img_data) {
        throw std::runtime_error("Invalid image data pointer");
    }

    const int expected_channels = 3;
    std::vector<unsigned char> rgb_data;
    const unsigned char* source_data = img_data;
    int source_channels = channels;

    if (config_.do_convert_rgb && channels != expected_channels) {
        rgb_data = convert_to_rgb(img_data, width, height, channels);
        source_data = rgb_data.data();
        source_channels = expected_channels;
        
    }

    if (source_channels != expected_channels) {
        throw std::runtime_error("Image must have 3 channels (RGB)");
    }

    const int patch = config_.patch_size;
    const int downsample = config_.downsample_factor;
    const int patch_dim = patch * patch * expected_channels;

    if (patch <= 0) {
        throw std::runtime_error("Patch size must be positive");
    }
    if (config_.tile_size % patch != 0) {
        throw std::runtime_error("Tile size must be divisible by patch size");
    }

    auto [resized_width, resized_height] = smart_resize(height, width);
    

    const int patches_per_tile_side = config_.tile_size / patch;
    const int tile_patch_count = patches_per_tile_side * patches_per_tile_side;
    const int thumbnail_patch_cap = config_.max_image_tokens * downsample * downsample;
    int max_patches_per_tile = config_.max_num_patches;
    max_patches_per_tile = std::max(max_patches_per_tile, tile_patch_count);
    max_patches_per_tile = std::max(max_patches_per_tile, thumbnail_patch_cap);

    const bool allow_splitting = config_.do_image_splitting;
    const bool should_split = allow_splitting && is_image_too_large(height, width);
    

    size_t expected_tiles = should_split ? static_cast<size_t>(config_.max_tiles) : 1;
    if (should_split && config_.use_thumbnail) {
        expected_tiles += 1;
    }

    std::vector<std::vector<float>> tile_patches;
    tile_patches.reserve(expected_tiles);
    std::vector<std::pair<int, int>> spatial_shapes;
    spatial_shapes.reserve(expected_tiles);

    auto normalize_and_patchify = [&](const float* data_ptr, int img_width, int img_height) {
        if (img_height % patch != 0 || img_width % patch != 0) {
            throw std::runtime_error("Image dimensions must be divisible by patch size");
        }
        std::vector<float> normalized = normalize_image(data_ptr, img_width, img_height, expected_channels);
        auto patches = convert_image_to_patches(normalized, img_width, img_height, expected_channels, patch);
        std::vector<float> flattened(patches.size() * patch_dim);
        for (size_t idx = 0; idx < patches.size(); ++idx) {
            std::copy(patches[idx].begin(), patches[idx].end(), flattened.begin() + idx * patch_dim);
        }
        tile_patches.push_back(std::move(flattened));
        spatial_shapes.emplace_back(img_height / patch, img_width / patch);
    };

    int grid_rows = 1;
    int grid_cols = 1;
    bool thumbnail_added = false;

    if (should_split) {
        auto [grid_target_width, grid_target_height] = get_grid_layout(height, width);
        grid_cols = grid_target_width / config_.tile_size;
        grid_rows = grid_target_height / config_.tile_size;
        printf("[debug] grid_target_width=%d grid_target_height=%d grid_cols=%d grid_rows=%d\n",
               grid_target_width, grid_target_height, grid_cols, grid_rows);
    

        std::vector<float> resized_grid = resize_image(
            source_data, width, height, grid_target_width, grid_target_height, expected_channels);

        std::vector<float> tile_buffer(
            static_cast<size_t>(config_.tile_size) * config_.tile_size * expected_channels);

        for (int row = 0; row < grid_rows; ++row) {
            for (int col = 0; col < grid_cols; ++col) {
                for (int y = 0; y < config_.tile_size; ++y) {
                    const float* src_row = resized_grid.data() +
                        ((row * config_.tile_size + y) * grid_target_width + col * config_.tile_size) *
                        expected_channels;
                    float* dst_row = tile_buffer.data() + (y * config_.tile_size) * expected_channels;
                    std::copy_n(src_row,
                                static_cast<size_t>(config_.tile_size) * expected_channels,
                                dst_row);
                }
                normalize_and_patchify(tile_buffer.data(), config_.tile_size, config_.tile_size);
                
            }
        }

        if (config_.use_thumbnail && grid_rows * grid_cols != 1) {
            std::vector<float> thumbnail_bytes = resize_image(
                source_data, width, height, resized_width, resized_height, expected_channels);
            normalize_and_patchify(thumbnail_bytes.data(), resized_width, resized_height);
            thumbnail_added = true;
            
        }
    } else {
        int target_width = resized_width;
        int target_height = resized_height;

        const bool needs_resize = config_.do_resize && (width != target_width || height != target_height);

        std::vector<float> resized_image;
        if (needs_resize) {
            resized_image = resize_image(source_data, width, height, target_width, target_height, expected_channels);
            normalize_and_patchify(resized_image.data(), target_width, target_height);
            
        } else {
            resized_image.resize(static_cast<size_t>(width) * height * expected_channels);
            for (size_t idx = 0; idx < resized_image.size(); ++idx) {
                resized_image[idx] = static_cast<float>(source_data[idx]);
            }
            normalize_and_patchify(resized_image.data(), width, height);
            target_width = width;
            target_height = height;
            resized_width = target_width;
            resized_height = target_height;
            
        }

        grid_rows = 1;
        grid_cols = 1;
    }

    PreprocessedImage result = pad_patches(tile_patches, spatial_shapes, patch_dim, max_patches_per_tile);
    

    result.image_rows = grid_rows;
    result.image_cols = grid_cols;
    result.image_width = resized_width;
    result.image_height = resized_height;

    auto compute_tokens = [&](int patches_h, int patches_w) -> int {
        int tokens_h = (patches_h + downsample - 1) / downsample;
        int tokens_w = (patches_w + downsample - 1) / downsample;
        return tokens_h * tokens_w;
    };

    result.tokens_per_tile = spatial_shapes.empty()
                                 ? 0
                                 : compute_tokens(spatial_shapes.front().first, spatial_shapes.front().second);
    

    if (thumbnail_added && !spatial_shapes.empty()) {
        const auto& thumb_shape = spatial_shapes.back();
        result.thumbnail_tokens = compute_tokens(thumb_shape.first, thumb_shape.second);
        
    } else {
        result.thumbnail_tokens = 0;
    }

    return result;
}

std::vector<unsigned char> Siglip2Preprocessor::convert_to_rgb(
    const unsigned char* img_data, int width, int height, int channels) {
    
    std::vector<unsigned char> rgb_data(width * height * 3);
    
    if (channels == 1) {
        for (int i = 0; i < width * height; ++i) {
            rgb_data[i * 3 + 0] = img_data[i];
            rgb_data[i * 3 + 1] = img_data[i];
            rgb_data[i * 3 + 2] = img_data[i];
        }
    } else if (channels == 4) {
        for (int i = 0; i < width * height; ++i) {
            rgb_data[i * 3 + 0] = img_data[i * 4 + 0];
            rgb_data[i * 3 + 1] = img_data[i * 4 + 1];
            rgb_data[i * 3 + 2] = img_data[i * 4 + 2];
        }
    } else if (channels == 2) {
        for (int i = 0; i < width * height; ++i) {
            rgb_data[i * 3 + 0] = img_data[i * 2 + 0];
            rgb_data[i * 3 + 1] = img_data[i * 2 + 0];
            rgb_data[i * 3 + 2] = img_data[i * 2 + 0];
        }
    } else {
        throw std::runtime_error("Unsupported number of channels: " + std::to_string(channels));
    }
    
    return rgb_data;
}

std::vector<float> Siglip2Preprocessor::resize_image(
    const unsigned char* img_data, int src_width, int src_height,
    int dst_width, int dst_height, int channels) {
    
    const size_t src_elements = static_cast<size_t>(src_width) * src_height * channels;
    std::vector<float> src_float(src_elements);
    for (size_t idx = 0; idx < src_elements; ++idx) {
        src_float[idx] = static_cast<float>(img_data[idx]);
    }
    

    std::vector<float> resized_data(static_cast<size_t>(dst_width) * dst_height * channels);
    
    stbir_pixel_layout layout = (channels == 1) ? STBIR_1CHANNEL : 
                                (channels == 3) ? STBIR_RGB : STBIR_RGBA;
    
    float* result = stbir_resize_float_linear(
        src_float.data(), src_width, src_height, 0,
        resized_data.data(), dst_width, dst_height, 0,
        layout
    );

    if (!result) {
        throw std::runtime_error("Failed to resize image");
    }

    
    return resized_data;
}

std::vector<float> Siglip2Preprocessor::normalize_image(
    const float* img_data, int width, int height, int channels) {
    
    size_t total_pixels = width * height * channels;
    std::vector<float> normalized(total_pixels);

    for (size_t i = 0; i < width * height; ++i) {
        for (int c = 0; c < channels; ++c) {
            size_t idx = i * channels + c;
            float pixel = img_data[idx];
            
            if (config_.do_rescale) {
                pixel *= config_.rescale_factor;
            }
            
            if (config_.do_normalize) {
                pixel = (pixel - config_.image_mean[c]) / config_.image_std[c];
            }
            
            normalized[idx] = pixel;
        }
    }
    

    return normalized;
}

std::vector<std::vector<float>> Siglip2Preprocessor::convert_image_to_patches(
    const std::vector<float>& image, int width, int height, int channels, int patch_size) {
    
    int num_patches_height = height / patch_size;
    int num_patches_width = width / patch_size;
    int num_patches = num_patches_height * num_patches_width;
    int patch_elements = patch_size * patch_size * channels;

    std::vector<std::vector<float>> patches(num_patches, std::vector<float>(patch_elements));

    for (int ph = 0; ph < num_patches_height; ++ph) {
        for (int pw = 0; pw < num_patches_width; ++pw) {
            int patch_idx = ph * num_patches_width + pw;
            
            for (int y = 0; y < patch_size; ++y) {
                for (int x = 0; x < patch_size; ++x) {
                    int img_y = ph * patch_size + y;
                    int img_x = pw * patch_size + x;
                    int img_idx = (img_y * width + img_x) * channels;
                    int patch_offset = (y * patch_size + x) * channels;
                    
                    for (int c = 0; c < channels; ++c) {
                        patches[patch_idx][patch_offset + c] = image[img_idx + c];
                    }
                }
            }
        }
    }

    return patches;
}

Siglip2Preprocessor::PreprocessedImage Siglip2Preprocessor::pad_patches(
    const std::vector<std::vector<float>>& tile_patches,
    const std::vector<std::pair<int,int>>& spatial_shapes,
    int patch_dim,
    int max_patches_per_tile) {

    if (tile_patches.size() != spatial_shapes.size()) {
        throw std::runtime_error("Mismatch between tile data and spatial shapes");
    }

    PreprocessedImage result;

    const int num_tiles = static_cast<int>(tile_patches.size());
    result.num_tiles = num_tiles;
    result.patch_dim = patch_dim;
    result.max_patches_per_tile = max_patches_per_tile;
    result.spatial_shapes = spatial_shapes;
    result.actual_num_patches = 0;

    const size_t total_values = static_cast<size_t>(num_tiles) * max_patches_per_tile * patch_dim;
    result.pixel_values.assign(total_values, 0.0f);
    result.pixel_attention_mask.assign(static_cast<size_t>(num_tiles) * max_patches_per_tile, 0);
    

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const auto& [patches_h, patches_w] = spatial_shapes[tile_idx];
        const int actual_patches = patches_h * patches_w;

        if (actual_patches > max_patches_per_tile) {
            throw std::runtime_error("Actual patches exceed max_patches_per_tile");
        }

        const auto& flattened = tile_patches[tile_idx];
        const size_t expected_size = static_cast<size_t>(actual_patches) * patch_dim;
        if (flattened.size() != expected_size) {
            throw std::runtime_error("Tile patch data has unexpected size");
        }

        float* destination = result.pixel_values.data() +
                             static_cast<size_t>(tile_idx) * max_patches_per_tile * patch_dim;
        std::memcpy(destination, flattened.data(), expected_size * sizeof(float));
    

        int mask_offset = tile_idx * max_patches_per_tile;
        for (int p = 0; p < actual_patches; ++p) {
            result.pixel_attention_mask[mask_offset + p] = 1;
        }

        result.actual_num_patches += actual_patches;
    }

    if (!spatial_shapes.empty()) {
        result.num_patches_height = spatial_shapes.front().first;
        result.num_patches_width = spatial_shapes.front().second;
    } else {
        result.num_patches_height = 0;
        result.num_patches_width = 0;
    }

    result.pixel_values_shape = {
        static_cast<size_t>(num_tiles),
        static_cast<size_t>(max_patches_per_tile),
        static_cast<size_t>(patch_dim)
    };
    

    result.pixel_attention_mask_shape = {
        static_cast<size_t>(num_tiles),
        static_cast<size_t>(max_patches_per_tile)
    };

    result.spatial_shapes_shape = {
        static_cast<size_t>(num_tiles),
        static_cast<size_t>(2)
    };
    

    return result;
}

} // namespace engine
} // namespace cactus


