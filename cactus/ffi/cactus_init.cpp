#include "cactus_ffi.h"
#include "cactus_utils.h"
#include <string>
#include <algorithm>

using namespace cactus::engine;

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
    try {
        auto* handle = new CactusModelHandle();
        handle->model = create_model(model_path);

        if (!handle->model) {
            last_error_message = "Failed to create model from: " + std::string(model_path);
            delete handle;
            return nullptr;
        }

        if (!handle->model->init(model_path, context_size)) {
            last_error_message = "Failed to initialize model from: " + std::string(model_path);
            delete handle;
            return nullptr;
        }

        if (corpus_dir != nullptr) {
            Tokenizer* tok = handle->model->get_tokenizer();
            if (tok) {
                try {
                    tok->set_corpus_dir(std::string(corpus_dir));
                } catch (...) {}
            }
        }

        return handle;
    } catch (const std::exception& e) {
        last_error_message = std::string(e.what());
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown error during model initialization";
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

}
