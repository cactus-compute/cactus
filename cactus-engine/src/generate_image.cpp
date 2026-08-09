#include "../cactus_engine.h"
#include "utils.h"
#include <cstring>
#include <mutex>

using namespace cactus::engine;
using namespace cactus::ffi;

extern "C" {

int cactus_generate_image(
    cactus_model_t model,
    const char* prompt,
    uint8_t* rgb_buffer,
    size_t buffer_size,
    unsigned int* image_width,
    unsigned int* image_height,
    int steps,
    float guidance_scale,
    unsigned long long seed
) {
    if (!model || !prompt || !rgb_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("generate_image", "Invalid parameters for image generation");
        return -1;
    }

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);
        std::lock_guard<std::mutex> lock(handle->model_mutex);
        uint32_t width = 0;
        uint32_t height = 0;
        const int result = handle->model->generate_image(
            prompt, rgb_buffer, buffer_size, &width, &height,
            steps, guidance_scale, static_cast<uint64_t>(seed));
        if (result > 0) {
            if (image_width) *image_width = width;
            if (image_height) *image_height = height;
        }
        return result;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        CACTUS_LOG_ERROR("generate_image", "Exception: " << e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during image generation";
        CACTUS_LOG_ERROR("generate_image", last_error_message);
        return -1;
    }
}

}
