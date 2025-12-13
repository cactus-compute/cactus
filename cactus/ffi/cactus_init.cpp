#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "cactus_telemetry.h"
#include <string>
#include <algorithm>

using namespace cactus::engine;
using namespace cactus::ffi;

std::string last_error_message;

bool matches_stop_sequence(const std::vector<uint32_t>& generated_tokens,
                           const std::vector<std::vector<uint32_t>>& stop_sequences) {
    for (const auto& stop_seq : stop_sequences) {
        if (stop_seq.empty()) continue;
        if (generated_tokens.size() >= stop_seq.size()) {
            if (std::equal(stop_seq.rbegin(), stop_seq.rend(), generated_tokens.rbegin()))
                return true;
        }
    }
    return false;
}

extern "C" {

const char* cactus_get_last_error() {
    return last_error_message.c_str();
}

cactus_model_t cactus_init(const char* model_path, size_t context_size, const char* corpus_dir) {
    std::string model_path_str = model_path ? std::string(model_path) : "unknown";

    std::string model_name = model_path_str;
    size_t last_slash = model_path_str.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        model_name = model_path_str.substr(last_slash + 1);
    }

    CACTUS_LOG_INFO("init", "Loading model: " << model_name << " from " << model_path_str);

    try {
        auto* handle = new CactusModelHandle();
        handle->model = create_model(model_path);
        handle->model_name = model_name;

        if (!handle->model) {
            last_error_message = "Failed to create model - check config.txt exists at: " + model_path_str;
            CACTUS_LOG_ERROR("init", last_error_message);

            CactusTelemetry::getInstance().recordInit(
                model_name, false, last_error_message
            );

            delete handle;
            return nullptr;
        }

        if (!handle->model->init(model_path, context_size)) {
            last_error_message = "Failed to initialize model - check weight files at: " + model_path_str;
            CACTUS_LOG_ERROR("init", last_error_message);

            CactusTelemetry::getInstance().recordInit(
                model_name, false, last_error_message
            );

            delete handle;
            return nullptr;
        }

        if (corpus_dir != nullptr) {
            Tokenizer* tok = handle->model->get_tokenizer();
            if (tok) {
                try {
                    tok->set_corpus_dir(std::string(corpus_dir));
                    CACTUS_LOG_INFO("init", "Corpus directory set: " << corpus_dir);
                } catch (const std::exception& e) {
                    CACTUS_LOG_WARN("init", "Failed to set corpus directory: " << e.what());
                }
            }
        }

        CACTUS_LOG_INFO("init", "Model loaded successfully: " << model_name);

        CactusTelemetry::getInstance().recordInit(
            model_name, true, "Model initialized successfully"
        );

        return handle;
    } catch (const std::exception& e) {
        last_error_message = "Exception during init: " + std::string(e.what());
        CACTUS_LOG_ERROR("init", last_error_message);

        CactusTelemetry::getInstance().recordInit(
            model_name, false, last_error_message
        );

        return nullptr;
    } catch (...) {
        last_error_message = "Unknown exception during model initialization";
        CACTUS_LOG_ERROR("init", last_error_message);

        CactusTelemetry::getInstance().recordInit(
            model_name, false, last_error_message
        );

        return nullptr;
    }
}

void cactus_destroy(cactus_model_t model) {
    if (model) delete static_cast<CactusModelHandle*>(model);
}

void cactus_reset(cactus_model_t model) {
    if (!model) return;
    auto* handle = static_cast<CactusModelHandle*>(model);
    handle->model->reset_cache();
    handle->processed_tokens.clear();
}

void cactus_stop(cactus_model_t model) {
    if (!model) return;
    auto* handle = static_cast<CactusModelHandle*>(model);
    handle->should_stop = true;
}

void cactus_set_log_level(int level) {
    if (level < 0) level = 0;
    if (level > 4) level = 4;
    cactus::Logger::instance().set_level(static_cast<cactus::LogLevel>(level));
}

static cactus_log_callback g_log_callback = nullptr;
static void* g_log_user_data = nullptr;

void cactus_set_log_callback(cactus_log_callback callback, void* user_data) {
    g_log_callback = callback;
    g_log_user_data = user_data;

    if (callback) {
        cactus::Logger::instance().set_callback(
            [](cactus::LogLevel level, const std::string& component, const std::string& message) {
                if (g_log_callback) {
                    g_log_callback(static_cast<int>(level), component.c_str(), message.c_str(), g_log_user_data);
                }
            }
        );
    } else {
        cactus::Logger::instance().set_callback(nullptr);
    }
}

}
