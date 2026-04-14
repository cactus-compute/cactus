#include "cactus_ffi.h"
#include "cactus_utils.h"
#include <string>

using namespace cactus::engine;
using namespace cactus::ffi;

static constexpr size_t DEFAULT_CONTEXT_SIZE = 512;

extern "C" {

cactus_context_t cactus_context_create(cactus_model_t model) {
    if (!model) {
        last_error_message = "Cannot create context: model is null";
        CACTUS_LOG_ERROR("context", last_error_message);
        return nullptr;
    }

    auto* parent = static_cast<CactusModelHandle*>(model);

    try {
        auto ctx = std::make_unique<CactusContextHandle>();
        ctx->parent_model = parent;
        ctx->model = create_model(parent->model_path);

        if (!ctx->model) {
            last_error_message = "Failed to create model for context - check config.txt at: " + parent->model_path;
            CACTUS_LOG_ERROR("context", last_error_message);
            return nullptr;
        }

        if (!ctx->model->init(parent->model_path, DEFAULT_CONTEXT_SIZE)) {
            last_error_message = "Failed to initialize model for context - check weight files at: " + parent->model_path;
            CACTUS_LOG_ERROR("context", last_error_message);
            return nullptr;
        }

        CACTUS_LOG_INFO("context", "Created new context for model: " << parent->model_name);
        return ctx.release();
    } catch (const std::exception& e) {
        last_error_message = "Exception creating context: " + std::string(e.what());
        CACTUS_LOG_ERROR("context", last_error_message);
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown exception creating context";
        CACTUS_LOG_ERROR("context", last_error_message);
        return nullptr;
    }
}

void cactus_context_destroy(cactus_context_t ctx) {
    if (ctx) delete static_cast<CactusContextHandle*>(ctx);
}

void cactus_context_reset(cactus_context_t ctx) {
    if (!ctx) return;
    auto* handle = static_cast<CactusContextHandle*>(ctx);
    handle->model->reset_cache();
    handle->processed_tokens.clear();
    handle->processed_images.clear();
}

void cactus_context_stop(cactus_context_t ctx) {
    if (!ctx) return;
    auto* handle = static_cast<CactusContextHandle*>(ctx);
    handle->should_stop = true;
}

int cactus_context_complete(
    cactus_context_t ctx,
    const char* messages_json,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    const char* tools_json,
    cactus_token_callback callback,
    void* user_data,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size
) {
    if (!ctx) {
        handle_error_response("Context is null", response_buffer, buffer_size);
        return -1;
    }
    auto* handle = static_cast<CactusContextHandle*>(ctx);
    const char* model_name = handle->parent_model ? handle->parent_model->model_name.c_str() : "unknown";
    return cactus_complete_impl(handle, model_name,
        messages_json, response_buffer, buffer_size, options_json, tools_json,
        callback, user_data, pcm_buffer, pcm_buffer_size);
}

int cactus_context_prefill(
    cactus_context_t ctx,
    const char* messages_json,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    const char* tools_json,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size
) {
    if (!ctx) {
        handle_error_response("Context is null", response_buffer, buffer_size);
        return -1;
    }
    auto* handle = static_cast<CactusContextHandle*>(ctx);
    const char* model_name = handle->parent_model ? handle->parent_model->model_name.c_str() : "unknown";
    return cactus_prefill_impl(handle, model_name,
        messages_json, response_buffer, buffer_size, options_json, tools_json,
        pcm_buffer, pcm_buffer_size);
}

int cactus_context_transcribe(
    cactus_context_t ctx,
    const char* audio_file_path,
    const char* prompt,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    cactus_token_callback callback,
    void* user_data,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size
) {
    if (!ctx) {
        handle_error_response("Context is null", response_buffer, buffer_size);
        return -1;
    }
    auto* handle = static_cast<CactusContextHandle*>(ctx);
    const char* model_name = handle->parent_model ? handle->parent_model->model_name.c_str() : "unknown";
    return cactus_transcribe_impl(handle, handle->parent_model, model_name,
        audio_file_path, prompt, response_buffer, buffer_size, options_json,
        callback, user_data, pcm_buffer, pcm_buffer_size);
}

}
