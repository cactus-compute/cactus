#include "cactus.h"
#include "common.h"
#include "json.hpp"

using json = nlohmann::ordered_json;

namespace cactus {

cactus_context::~cactus_context() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
        ctx_sampling = nullptr;
    }
    releaseMultimodal();
    releaseVocoder();
}

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
    context_full = false;
    stopped_eos = false;
    stopped_word = false;
    stopped_limit = false;
    stopping_word = "";
    incomplete = false;
    n_remain = 0;
    n_past = 0;
    embd.clear();
    next_token_uses_guide_token = true;
    guide_tokens.clear();
    mtmd_bitmap_past_hashes.clear();
    audio_tokens.clear();
    if (ctx_sampling) {
    }
    if (ctx) {
        llama_kv_self_clear(ctx);
    }
}

bool cactus_context::initSampling() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
        ctx_sampling = nullptr;
    }
    if (model) {
        ctx_sampling = common_sampler_init(model, params.sampling);
        if (ctx_sampling) {
             params.sampling.n_prev = n_ctx;
        }
    } else {
        LOG_ERROR("Cannot initialize sampling context: model is not loaded.");
        return false;
    }
    return ctx_sampling != nullptr;
}

void cactus_context::setGuideTokens(const std::vector<llama_token> &tokens) {
    guide_tokens = tokens;
}

void cactus_context::endCompletion() {
    is_predicting = false;
}

} // namespace cactus