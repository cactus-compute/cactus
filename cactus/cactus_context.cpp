#include "ggml.h"  
#include "cactus.h"
#include "common.h" 
#include "mtmd.h"
namespace cactus {

/**
 * @brief Destructor for cactus_context
 * 
 * Cleans up resources, including the sampling context
 */
cactus_context::~cactus_context() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
        ctx_sampling = nullptr; 
    }
    if (ctx_mtmd != nullptr) { 
        mtmd_free(ctx_mtmd);
        ctx_mtmd = nullptr;
    }

    if (vocoder_ctx != nullptr) {
        llama_free(vocoder_ctx);
        vocoder_ctx = nullptr;
    }
    if (vocoder_model != nullptr) {
        llama_model_free(vocoder_model);
        vocoder_model = nullptr;
    }
}


/**
 * @brief Rewinds the context to start a new generation
 * 
 * Resets internal state to prepare for a new generation task
 */
void cactus_context::rewind() {
    is_interrupted = false;
    is_predicting = false; 
    params.antiprompt.clear();
    params.sampling.grammar.clear();
    num_prompt_tokens = 0;
    num_tokens_predicted = 0;
    generated_text = "";
    generated_text.reserve(params.n_ctx); 
    generated_token_probs.clear();
    truncated = false;
    stopped_eos = false;
    stopped_word = false;
    stopped_limit = false;
    stopping_word = "";
    incomplete = false;
    n_remain = 0;
    n_past = 0;
    embd.clear();

    if (ctx) {
        llama_kv_self_clear(ctx);
        LOG_VERBOSE("KV cache (self) cleared in rewind.");
    }

    if (ctx_mtmd != nullptr) {
        LOG_VERBOSE("Freeing existing mtmd_context in rewind.");
        mtmd_free(ctx_mtmd);
        ctx_mtmd = nullptr;
    }

    if (!this->params.mmproj.path.empty() && this->model != nullptr) {
        LOG_VERBOSE("Re-initializing mtmd_context in rewind for mmproj: %s", this->params.mmproj.path.c_str());
        struct mtmd_context_params mtmd_params_rewind = mtmd_context_params_default();
        mtmd_params_rewind.use_gpu = this->params.mmproj_use_gpu;
        mtmd_params_rewind.n_threads = this->params.cpuparams.n_threads; 
        mtmd_params_rewind.verbosity = this->params.verbosity > 0 ? LM_GGML_LOG_LEVEL_INFO : LM_GGML_LOG_LEVEL_ERROR;
        
        ctx_mtmd = mtmd_init_from_file(this->params.mmproj.path.c_str(), this->model, mtmd_params_rewind);

        if (ctx_mtmd == nullptr) {
            LOG_ERROR("Failed to re-initialize mtmd_context in rewind with mmproj: %s", this->params.mmproj.path.c_str());
        } else {
            LOG_INFO("mtmd_context re-initialized successfully in rewind.");
        }
    } else if (ctx_mtmd != nullptr) {
        LOG_WARNING("mtmd_context was non-null in rewind but no mmproj path found in params to re-initialize. Ensuring it is null.");
        mtmd_free(ctx_mtmd);
        ctx_mtmd = nullptr;
    }

    if (ctx_sampling) {
        common_sampler_reset(ctx_sampling);
    }
}


/**
 * @brief Initializes the sampling context
 * 
 * @return true if initialization succeeded, false otherwise
 */
bool cactus_context::initSampling() {
    if (this->ctx_sampling != nullptr) {
        common_sampler_free(this->ctx_sampling);
        this->ctx_sampling = nullptr;
    }
    if (!this->model) { 
        LOG_ERROR("Cannot initialize sampler: model is not loaded.");
        return false;
    }

    this->ctx_sampling = common_sampler_init(this->model, this->params.sampling);
    
    if (!this->ctx_sampling) {
        LOG_ERROR("Failed to initialize common_sampler.");
        return false;
    }
    // If common_sampler_init was successful, often n_prev is set based on context.
    // This logic might already be in common_sampler_init or needs to be here if it was in original cactus.
    // params.sampling.n_prev = n_ctx; // Example, check if needed after common_sampler_init
    return true;
}


} // namespace cactus 