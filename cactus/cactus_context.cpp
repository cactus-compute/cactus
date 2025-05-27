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

    if (ctx) { // Ensure ctx is not null before clearing
        llama_kv_self_clear(ctx);
        LOG_VERBOSE("KV cache (self) cleared in rewind.");
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