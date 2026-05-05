#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#define STBI_NO_BMP
#define STBI_NO_PSD
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_TGA

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "stb_image.h"
#include "stb_image_resize2.h"

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "cactus_kernels.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

unsigned char* cactus_image_load(const char* path, int* width, int* height, int* channels, int desired_channels) {
    return stbi_load(path, width, height, channels, desired_channels);
}

void cactus_image_free(unsigned char* data) {
    stbi_image_free(data);
}

const char* cactus_image_failure_reason() {
    return stbi_failure_reason();
}

void cactus_image_resize_uint8(
    const unsigned char* input, int src_w, int src_h,
    unsigned char* output, int dst_w, int dst_h, int channels) {
    stbir_resize_uint8_linear(input, src_w, src_h, 0,
                               output, dst_w, dst_h, 0,
                               static_cast<stbir_pixel_layout>(channels));
}

void cactus_image_resize_float(
    const float* input, int src_w, int src_h,
    float* output, int dst_w, int dst_h, int channels) {
    stbir_pixel_layout layout = (channels == 1) ? STBIR_1CHANNEL :
                                (channels == 3) ? STBIR_RGB : STBIR_RGBA;
    stbir_resize_float_linear(input, src_w, src_h, 0,
                               output, dst_w, dst_h, 0, layout);
}

void cactus_image_normalize(
    const float* input, float* output,
    int width, int height, int channels,
    float rescale_factor, const float* mean, const float* std_dev) {
    size_t total = static_cast<size_t>(width) * height;
    for (size_t i = 0; i < total; ++i) {
        for (int c = 0; c < channels; ++c) {
            size_t idx = i * channels + c;
            float pixel = input[idx] * rescale_factor;
            output[idx] = (pixel - mean[c]) / std_dev[c];
        }
    }
}

void cactus_image_to_patches(
    const float* image, float* patches,
    int width, int height, int channels, int patch_size) {
    int ph = height / patch_size;
    int pw = width / patch_size;
    int patch_elements = patch_size * patch_size * channels;

    for (int py = 0; py < ph; ++py) {
        for (int px = 0; px < pw; ++px) {
            int patch_idx = py * pw + px;
            for (int y = 0; y < patch_size; ++y) {
                for (int x = 0; x < patch_size; ++x) {
                    int img_y = py * patch_size + y;
                    int img_x = px * patch_size + x;
                    int img_idx = (img_y * width + img_x) * channels;
                    int patch_offset = (y * patch_size + x) * channels;
                    for (int c = 0; c < channels; ++c) {
                        patches[patch_idx * patch_elements + patch_offset + c] = image[img_idx + c];
                    }
                }
            }
        }
    }
}

void cactus_image_convert_to_rgb(
    const unsigned char* input, unsigned char* output,
    int width, int height, int channels) {
    int total = width * height;
    if (channels == 1) {
        for (int i = 0; i < total; ++i) {
            output[i * 3 + 0] = input[i];
            output[i * 3 + 1] = input[i];
            output[i * 3 + 2] = input[i];
        }
    } else if (channels == 4) {
        for (int i = 0; i < total; ++i) {
            output[i * 3 + 0] = input[i * 4 + 0];
            output[i * 3 + 1] = input[i * 4 + 1];
            output[i * 3 + 2] = input[i * 4 + 2];
        }
    } else if (channels == 2) {
        for (int i = 0; i < total; ++i) {
            output[i * 3 + 0] = input[i * 2 + 0];
            output[i * 3 + 1] = input[i * 2 + 0];
            output[i * 3 + 2] = input[i * 2 + 0];
        }
    }
}
