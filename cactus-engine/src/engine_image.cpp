#define STBI_NO_BMP
#define STBI_NO_PSD
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_TGA

#include "engine.h"
#include "stb_image.h"
#include "stb_image_resize2.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace cactus {
namespace engine {

namespace {

constexpr int kPillowResizePrecisionBits = 22;

static unsigned char pillow_clip8(int64_t value) {
    int shifted = static_cast<int>(value >> kPillowResizePrecisionBits);
    if (shifted < 0) return 0;
    if (shifted > 255) return 255;
    return static_cast<unsigned char>(shifted);
}

static void precompute_pillow_bilinear_coeffs(int in_size, int out_size,
                                              std::vector<int>& bounds,
                                              std::vector<int32_t>& coeffs,
                                              int& ksize) {
    const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    const double filterscale = std::max(scale, 1.0);
    const double support = filterscale;
    const double inv_filterscale = 1.0 / filterscale;
    ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    bounds.assign(static_cast<size_t>(out_size) * 2, 0);
    coeffs.assign(static_cast<size_t>(out_size) * ksize, 0);

    for (int out = 0; out < out_size; ++out) {
        const double center = (static_cast<double>(out) + 0.5) * scale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;

        bounds[static_cast<size_t>(out) * 2] = xmin;
        bounds[static_cast<size_t>(out) * 2 + 1] = xmax;

        std::vector<double> weights(static_cast<size_t>(ksize), 0.0);
        double total = 0.0;
        for (int i = 0; i < xmax; ++i) {
            double x = (static_cast<double>(i + xmin) - center + 0.5) * inv_filterscale;
            if (x < 0.0) x = -x;
            double weight = x < 1.0 ? 1.0 - x : 0.0;
            weights[static_cast<size_t>(i)] = weight;
            total += weight;
        }
        for (int i = 0; i < ksize; ++i) {
            double weight = total != 0.0 ? weights[static_cast<size_t>(i)] / total : 0.0;
            coeffs[static_cast<size_t>(out) * ksize + i] = static_cast<int32_t>(
                weight < 0.0
                    ? -0.5 + weight * static_cast<double>(1 << kPillowResizePrecisionBits)
                    : 0.5 + weight * static_cast<double>(1 << kPillowResizePrecisionBits));
        }
    }
}

static std::vector<unsigned char> resize_rgb_uint8_pillow_bilinear(const unsigned char* src,
                                                                   int src_w,
                                                                   int src_h,
                                                                   int dst_w,
                                                                   int dst_h) {
    std::vector<int> x_bounds;
    std::vector<int> y_bounds;
    std::vector<int32_t> x_coeffs;
    std::vector<int32_t> y_coeffs;
    int x_ksize = 0;
    int y_ksize = 0;
    precompute_pillow_bilinear_coeffs(src_w, dst_w, x_bounds, x_coeffs, x_ksize);
    precompute_pillow_bilinear_coeffs(src_h, dst_h, y_bounds, y_coeffs, y_ksize);

    const int y_first = y_bounds[0];
    const int y_last = y_bounds[static_cast<size_t>(dst_h - 1) * 2] +
                       y_bounds[static_cast<size_t>(dst_h - 1) * 2 + 1];
    const int temp_h = y_last - y_first;
    std::vector<unsigned char> temp(static_cast<size_t>(dst_w) * temp_h * 3);

    for (int ty = 0; ty < temp_h; ++ty) {
        const int sy = y_first + ty;
        for (int dx = 0; dx < dst_w; ++dx) {
            const int xmin = x_bounds[static_cast<size_t>(dx) * 2];
            const int count = x_bounds[static_cast<size_t>(dx) * 2 + 1];
            const int32_t* k = x_coeffs.data() + static_cast<size_t>(dx) * x_ksize;
            for (int c = 0; c < 3; ++c) {
                int64_t sum = static_cast<int64_t>(1) << (kPillowResizePrecisionBits - 1);
                for (int x = 0; x < count; ++x) {
                    sum += static_cast<int64_t>(src[(static_cast<size_t>(sy) * src_w + xmin + x) * 3 + c]) * k[x];
                }
                temp[(static_cast<size_t>(ty) * dst_w + dx) * 3 + c] = pillow_clip8(sum);
            }
        }
    }

    std::vector<unsigned char> dst(static_cast<size_t>(dst_w) * dst_h * 3);
    for (int dy = 0; dy < dst_h; ++dy) {
        const int ymin = y_bounds[static_cast<size_t>(dy) * 2] - y_first;
        const int count = y_bounds[static_cast<size_t>(dy) * 2 + 1];
        const int32_t* k = y_coeffs.data() + static_cast<size_t>(dy) * y_ksize;
        for (int dx = 0; dx < dst_w; ++dx) {
            for (int c = 0; c < 3; ++c) {
                int64_t sum = static_cast<int64_t>(1) << (kPillowResizePrecisionBits - 1);
                for (int y = 0; y < count; ++y) {
                    sum += static_cast<int64_t>(temp[(static_cast<size_t>(ymin + y) * dst_w + dx) * 3 + c]) * k[y];
                }
                dst[(static_cast<size_t>(dy) * dst_w + dx) * 3 + c] = pillow_clip8(sum);
            }
        }
    }
    return dst;
}

} // namespace

Siglip2Preprocessor::PreprocessedImage::~PreprocessedImage() {
    pixel_values.clear();
    pixel_attention_mask.clear();
}

Siglip2Preprocessor::Siglip2Preprocessor(const Config& config)
    : config_(config) {}

Siglip2Preprocessor::Siglip2Preprocessor() : config_() {}

Siglip2Preprocessor::~Siglip2Preprocessor() = default;

std::pair<int64_t, int64_t> Siglip2Preprocessor::compute_pixel_limits() const {
    int64_t min_pixels = static_cast<int64_t>(config_.min_image_tokens) *
                         config_.patch_size * config_.patch_size *
                         config_.downsample_factor * config_.downsample_factor;
    int64_t max_pixels = static_cast<int64_t>(config_.max_image_tokens) *
                         config_.patch_size * config_.patch_size *
                         config_.downsample_factor * config_.downsample_factor;
    return {min_pixels, max_pixels};
}

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
    auto [min_pixels, max_pixels] = compute_pixel_limits();

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
    auto [min_pixels, max_pixels] = compute_pixel_limits();
    const double max_pixels_with_tolerance = static_cast<double>(max_pixels) * config_.max_pixels_tolerance;

    bool result = static_cast<double>(pixels) > max_pixels_with_tolerance;

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
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Image dimensions must be positive");
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

    for (size_t i = 0; i < static_cast<size_t>(width * height); ++i) {
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

Gemma4ImagePreprocessed preprocess_gemma4_image(const std::string& image_path, const Config& config) {
    Gemma4ImagePreprocessed result;

    uint32_t patch_size_u = config.vision_patch_size ? config.vision_patch_size : 16;
    uint32_t pooling_k_u = config.vision_pooling_kernel_size ? config.vision_pooling_kernel_size : 3;
    uint32_t max_soft_tokens_u = config.vision_default_output_length ? config.vision_default_output_length : 280;
    if (patch_size_u == 0 || pooling_k_u == 0 || max_soft_tokens_u == 0) {
        throw std::runtime_error("Gemma4 image config has invalid patch/pooling/soft-token values");
    }

    const int patch_size = static_cast<int>(patch_size_u);
    const int pooling_k = static_cast<int>(pooling_k_u);
    const size_t max_patches = static_cast<size_t>(max_soft_tokens_u) * pooling_k * pooling_k;
    const int side_multiple = pooling_k * patch_size;
    const size_t patch_dim = static_cast<size_t>(3) * patch_size * patch_size;

    int width = 0, height = 0, channels = 0;
    unsigned char* raw = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (!raw) {
        throw std::runtime_error("Failed to load image: " + image_path + " - " + std::string(stbi_failure_reason()));
    }
    if (width <= 0 || height <= 0) {
        stbi_image_free(raw);
        throw std::runtime_error("Loaded image has invalid dimensions: " + image_path);
    }

    const double target_pixels = static_cast<double>(max_patches) * patch_size * patch_size;
    const double pixel_count = std::max(1.0, static_cast<double>(width) * static_cast<double>(height));
    const double factor = std::sqrt(target_pixels / pixel_count);
    int target_h = static_cast<int>(std::floor(factor * height / side_multiple)) * side_multiple;
    int target_w = static_cast<int>(std::floor(factor * width / side_multiple)) * side_multiple;
    if (target_h == 0) target_h = side_multiple;
    if (target_w == 0) target_w = side_multiple;

    std::vector<float> resized(static_cast<size_t>(target_w) * target_h * 3);

    if (target_w == width && target_h == height) {
        for (size_t i = 0; i < resized.size(); ++i) {
            resized[i] = static_cast<float>(raw[i]);
        }
    } else {
        std::vector<unsigned char> resized_u8 = resize_rgb_uint8_pillow_bilinear(raw, width, height, target_w, target_h);
        for (size_t i = 0; i < resized.size(); ++i) {
            resized[i] = static_cast<float>(resized_u8[i]);
        }
    }
    stbi_image_free(raw);

    const bool do_rescale = true;
    const float rescale_factor = config.rescale_factor > 0.0f ? config.rescale_factor : (1.0f / 255.0f);
    const bool do_normalize = false;
    const float mean_v = do_normalize ? config.image_mean : 0.0f;
    const float std_v = (do_normalize && config.image_std != 0.0f) ? config.image_std : 1.0f;

    if (do_rescale || do_normalize) {
        const float inv_std = 1.0f / std_v;
        for (size_t i = 0; i < resized.size(); ++i) {
            float v = resized[i];
            if (do_rescale) v *= rescale_factor;
            if (do_normalize) v = (v - mean_v) * inv_std;
            resized[i] = v;
        }
    }

    const int patch_h = target_h / patch_size;
    const int patch_w = target_w / patch_size;
    const size_t num_patches = static_cast<size_t>(patch_h) * static_cast<size_t>(patch_w);
    if (num_patches > max_patches) {
        throw std::runtime_error("Gemma4 native image preprocessing produced too many patches");
    }

    result.pixel_values.assign(max_patches * patch_dim, 0.0f);
    result.pixel_position_ids.assign(max_patches * 2, -1);

    for (int py = 0; py < patch_h; ++py) {
        for (int px = 0; px < patch_w; ++px) {
            const size_t patch_idx = static_cast<size_t>(py) * patch_w + px;
            float* dst = result.pixel_values.data() + patch_idx * patch_dim;
            for (int y = 0; y < patch_size; ++y) {
                const int img_y = py * patch_size + y;
                for (int x = 0; x < patch_size; ++x) {
                    const int img_x = px * patch_size + x;
                    const size_t src_off = (static_cast<size_t>(img_y) * target_w + img_x) * 3;
                    const size_t dst_off = (static_cast<size_t>(y) * patch_size + x) * 3;
                    dst[dst_off + 0] = resized[src_off + 0];
                    dst[dst_off + 1] = resized[src_off + 1];
                    dst[dst_off + 2] = resized[src_off + 2];
                }
            }
            result.pixel_position_ids[patch_idx * 2 + 0] = static_cast<int64_t>(px);
            result.pixel_position_ids[patch_idx * 2 + 1] = static_cast<int64_t>(py);
        }
    }

    result.num_patches = num_patches;
    result.max_patches = max_patches;
    result.patch_dim = patch_dim;
    return result;
}

Qwen3VlImagePreprocessed preprocess_qwen3_vl_image(const std::string& image_path, const Config& config) {
    Qwen3VlImagePreprocessed result;
    const int patch_size = static_cast<int>(config.vision_patch_size ? config.vision_patch_size : 16);
    const int temporal_patch_size = 2;
    const int merge_size = 2;
    const size_t image_tokens = config.image_seq_len ? static_cast<size_t>(config.image_seq_len) : 64;
    const size_t merged_patches = image_tokens * static_cast<size_t>(merge_size * merge_size);
    const int grid_side = static_cast<int>(std::sqrt(static_cast<double>(merged_patches)));
    if (patch_size <= 0 || grid_side <= 0 || static_cast<size_t>(grid_side * grid_side) != merged_patches) {
        throw std::runtime_error("Qwen3-VL native image preprocessing requires a square static image grid");
    }

    int width = 0, height = 0, channels = 0;
    unsigned char* raw = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (!raw) {
        throw std::runtime_error("Failed to load image: " + image_path + " - " + std::string(stbi_failure_reason()));
    }
    if (width <= 0 || height <= 0) {
        stbi_image_free(raw);
        throw std::runtime_error("Loaded image has invalid dimensions: " + image_path);
    }

    const int target_h = grid_side * patch_size;
    const int target_w = grid_side * patch_size;
    std::vector<float> resized(static_cast<size_t>(target_w) * target_h * 3);
    if (target_w == width && target_h == height) {
        for (size_t i = 0; i < resized.size(); ++i) {
            resized[i] = static_cast<float>(raw[i]);
        }
    } else {
        std::vector<float> src_float(static_cast<size_t>(width) * height * 3);
        for (size_t i = 0; i < src_float.size(); ++i) {
            src_float[i] = static_cast<float>(raw[i]);
        }
        if (!stbir_resize_float_linear(
                src_float.data(), width, height, 0,
                resized.data(), target_w, target_h, 0,
                STBIR_RGB)) {
            stbi_image_free(raw);
            throw std::runtime_error("Failed to resize image: " + image_path);
        }
    }
    stbi_image_free(raw);

    const float rescale_factor = config.rescale_factor > 0.0f ? config.rescale_factor : (1.0f / 255.0f);
    const float mean_v = config.image_mean;
    const float std_v = config.image_std != 0.0f ? config.image_std : 1.0f;
    for (float& v : resized) {
        v = (v * rescale_factor - mean_v) / std_v;
    }

    const size_t channel = 3;
    const size_t grid_t = 1;
    const size_t grid_h = static_cast<size_t>(grid_side);
    const size_t grid_w = static_cast<size_t>(grid_side);
    const size_t patch_dim = channel * static_cast<size_t>(temporal_patch_size) * patch_size * patch_size;
    result.pixel_values.assign(grid_t * grid_h * grid_w * patch_dim, 0.0f);

    for (size_t gh_major = 0; gh_major < grid_h / merge_size; ++gh_major) {
        for (size_t gw_major = 0; gw_major < grid_w / merge_size; ++gw_major) {
            for (size_t mh = 0; mh < static_cast<size_t>(merge_size); ++mh) {
                for (size_t mw = 0; mw < static_cast<size_t>(merge_size); ++mw) {
                    const size_t patch_y = gh_major * merge_size + mh;
                    const size_t patch_x = gw_major * merge_size + mw;
                    const size_t patch_index =
                        (((gh_major * (grid_w / merge_size) + gw_major) * merge_size + mh) * merge_size + mw);
                    float* dst = result.pixel_values.data() + patch_index * patch_dim;
                    size_t dst_i = 0;
                    for (size_t c = 0; c < channel; ++c) {
                        for (size_t t = 0; t < static_cast<size_t>(temporal_patch_size); ++t) {
                            (void)t;
                            for (int y = 0; y < patch_size; ++y) {
                                const size_t img_y = patch_y * patch_size + static_cast<size_t>(y);
                                for (int x = 0; x < patch_size; ++x) {
                                    const size_t img_x = patch_x * patch_size + static_cast<size_t>(x);
                                    const size_t src_off = (img_y * static_cast<size_t>(target_w) + img_x) * channel + c;
                                    dst[dst_i++] = resized[src_off];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    result.grid_t = grid_t;
    result.grid_h = grid_h;
    result.grid_w = grid_w;
    result.patch_dim = patch_dim;
    return result;
}

Lfm2VlImagePreprocessed preprocess_lfm2_vl_image(const std::string& image_path, const Config& config) {
    Siglip2Preprocessor::Config sp_cfg;
    sp_cfg.patch_size = config.vision_patch_size ? static_cast<int>(config.vision_patch_size) : 16;
    sp_cfg.downsample_factor = config.downsample_factor ? static_cast<int>(config.downsample_factor) : 2;
    sp_cfg.min_tiles = config.min_tiles ? static_cast<int>(config.min_tiles) : 2;
    sp_cfg.max_tiles = config.max_tiles ? static_cast<int>(config.max_tiles) : 10;
    sp_cfg.use_thumbnail = config.use_thumbnail;
    sp_cfg.min_image_tokens = config.min_image_tokens ? static_cast<int>(config.min_image_tokens) : 64;
    sp_cfg.max_image_tokens = config.max_image_tokens ? static_cast<int>(config.max_image_tokens) : 256;
    sp_cfg.max_num_patches = config.max_num_patches ? static_cast<int>(config.max_num_patches) : 1024;
    sp_cfg.tile_size = config.tile_size ? static_cast<int>(config.tile_size) : 512;
    sp_cfg.max_pixels_tolerance = config.max_pixels_tolerance > 0.0f ? config.max_pixels_tolerance : 2.0f;
    sp_cfg.rescale_factor = config.rescale_factor > 0.0f ? config.rescale_factor : (1.0f / 255.0f);
    sp_cfg.image_mean[0] = sp_cfg.image_mean[1] = sp_cfg.image_mean[2] = config.image_mean;
    sp_cfg.image_std[0]  = sp_cfg.image_std[1]  = sp_cfg.image_std[2]  = config.image_std;
    sp_cfg.do_image_splitting = false;
    sp_cfg.do_resize = true;
    sp_cfg.do_rescale = true;
    sp_cfg.do_normalize = true;
    sp_cfg.do_convert_rgb = true;

    Siglip2Preprocessor pre(sp_cfg);
    auto sp_out = pre.preprocess_from_file(image_path);

    Lfm2VlImagePreprocessed result;
    result.pixel_values = std::move(sp_out.pixel_values);
    result.pixel_attention_mask.assign(sp_out.pixel_attention_mask.begin(),
                                       sp_out.pixel_attention_mask.end());
    if (!sp_out.spatial_shapes.empty()) {
        result.spatial_shape = sp_out.spatial_shapes.front();
    }
    result.num_patches = static_cast<size_t>(sp_out.actual_num_patches);
    result.patch_dim = static_cast<size_t>(sp_out.patch_dim);
    result.max_num_patches = static_cast<size_t>(sp_cfg.max_num_patches);
    return result;
}

} // namespace engine
} // namespace cactus
