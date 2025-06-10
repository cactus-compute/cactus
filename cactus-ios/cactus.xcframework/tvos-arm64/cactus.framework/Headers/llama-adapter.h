#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// llama_adapter_cvec
//

struct llama_adapter_cvec {
    lm_ggml_tensor * tensor_for(int il) const;

    lm_ggml_tensor * apply_to(lm_ggml_context * ctx, lm_ggml_tensor * cur, int  il) const;

    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const llama_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<lm_ggml_context_ptr> ctxs;
    std::vector<lm_ggml_backend_buffer_ptr> bufs;

    std::vector<lm_ggml_tensor *> tensors; // per layer
};

//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    lm_ggml_tensor * a = nullptr;
    lm_ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(lm_ggml_tensor * a, lm_ggml_tensor * b) : a(a), b(b) {}
};

struct llama_adapter_lora {
    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    std::vector<lm_ggml_context_ptr> ctxs;
    std::vector<lm_ggml_backend_buffer_ptr> bufs;

    float alpha;

    llama_adapter_lora() = default;
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(lm_ggml_tensor * w);
};

using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
