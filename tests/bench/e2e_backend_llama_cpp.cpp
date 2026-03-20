#include "e2e_driver.h"

#ifdef WITH_LLAMA_CPP

#include <llama.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cstring>

namespace {

struct LlamaCppHandle {
    llama_model* model;
    llama_context* ctx;
    llama_sampler* sampler;
};

bool available() {
    return true;
}

void* load(const char* model_path, int threads) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU only — no Metal/GPU offload

    llama_model* model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        std::cerr << "[llama_cpp] Failed to load model: " << model_path << "\n";
        return nullptr;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    if (threads > 0) {
        ctx_params.n_threads = threads;
        ctx_params.n_threads_batch = threads;
    }

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "[llama_cpp] Failed to create context\n";
        llama_model_free(model);
        return nullptr;
    }

    // Create greedy sampler
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    auto* h = new LlamaCppHandle{model, ctx, sampler};
    return h;
}

e2e::E2EResult generate(void* handle, const char* prompt, int max_tokens) {
    auto* h = static_cast<LlamaCppHandle*>(handle);
    e2e::E2EResult result = {};

    // Clear KV cache
    llama_memory_clear(llama_get_memory(h->ctx), true);

    // Tokenize
    const llama_vocab* vocab = llama_model_get_vocab(h->model);
    int n_prompt_max = static_cast<int>(strlen(prompt)) + 128;
    std::vector<llama_token> tokens(n_prompt_max);
    int n_tokens = llama_tokenize(vocab, prompt, static_cast<int>(strlen(prompt)),
                                   tokens.data(), n_prompt_max, true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt, static_cast<int>(strlen(prompt)),
                                   tokens.data(), static_cast<int>(tokens.size()), true, true);
    }
    tokens.resize(n_tokens);
    result.prefill_tokens = n_tokens;

    auto wall_start = std::chrono::steady_clock::now();

    // Prefill
    auto prefill_start = std::chrono::steady_clock::now();
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    llama_decode(h->ctx, batch);
    auto prefill_end = std::chrono::steady_clock::now();

    double prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
    result.ttft_ms = prefill_ms;

    // Decode loop
    auto decode_start = std::chrono::steady_clock::now();
    int generated = 0;
    llama_token eos = llama_vocab_eos(vocab);

    for (int i = 0; i < max_tokens; i++) {
        llama_token token = llama_sampler_sample(h->sampler, h->ctx, -1);
        generated++;
        if (token == eos) break;

        llama_batch single = llama_batch_get_one(&token, 1);
        llama_decode(h->ctx, single);
    }
    auto decode_end = std::chrono::steady_clock::now();

    double decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    auto wall_end = std::chrono::steady_clock::now();

    result.decode_tokens = generated;
    result.prefill_tps = (prefill_ms > 0.0) ? (n_tokens * 1000.0 / prefill_ms) : 0.0;
    result.decode_tps = (decode_ms > 0.0) ? (generated * 1000.0 / decode_ms) : 0.0;
    result.total_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    return result;
}

void unload(void* handle) {
    auto* h = static_cast<LlamaCppHandle*>(handle);
    llama_sampler_free(h->sampler);
    llama_free(h->ctx);
    llama_model_free(h->model);
    delete h;
}

static int reg = [] {
    e2e::register_e2e_backend({
        "llama_cpp", "llama_cpp",
        available, load, generate, unload
    });
    return 0;
}();

} // namespace

#endif // WITH_LLAMA_CPP
