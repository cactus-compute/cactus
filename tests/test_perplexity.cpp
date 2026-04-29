#include "test_utils.h"
#include "../cactus/cactus.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <memory>

using cactus::engine::Model;
using cactus::engine::create_model;

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");

// Short prose excerpt (public domain: opening of Pride and Prejudice + a short
// passage from Frankenstein). Kept deliberately in-distribution for a small
// instruction model so perplexity stays in a meaningful range.
static const char* PERPLEXITY_CORPUS =
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife. However little known the "
    "feelings or views of such a man may be on his first entering a "
    "neighbourhood, this truth is so well fixed in the minds of the "
    "surrounding families, that he is considered the rightful property of "
    "some one or other of their daughters. The day of my departure at length "
    "arrived. Clerval spent the last evening with us. He had endeavoured to "
    "persuade his father to permit him to accompany me and to become my "
    "fellow student, but in vain. His father was a narrow-minded trader, and "
    "saw idleness and ruin in the aspirations and ambition of his son.";

struct Result {
    double total_logprob = 0.0;
    size_t tokens_scored = 0;
    double perplexity = 0.0;
    double ms = 0.0;
};

static Result score_corpus(const std::string& model_path,
                           const std::vector<uint32_t>& tokens,
                           size_t context) {
    auto model = create_model(model_path);
    if (!model) {
        throw std::runtime_error("Failed to create model for " + model_path);
    }
    if (!model->init(model_path, /*context_size=*/2048, "", /*do_warmup=*/false)) {
        throw std::runtime_error("Failed to init model");
    }

    Result r;
    auto t0 = std::chrono::high_resolution_clock::now();
    r.total_logprob = model->score_tokens_cached_logprob(
        tokens, /*start=*/1, /*end=*/tokens.size(), context, &r.tokens_scored);
    auto t1 = std::chrono::high_resolution_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (r.tokens_scored > 0) {
        const double nll = -r.total_logprob / static_cast<double>(r.tokens_scored);
        r.perplexity = std::exp(nll);
    }
    return r;
}

static Result score_corpus_window(const std::string& model_path,
                                  const std::vector<uint32_t>& tokens,
                                  size_t context) {
    auto model = create_model(model_path);
    if (!model) throw std::runtime_error("Failed to create model");
    if (!model->init(model_path, /*context_size=*/2048, "", /*do_warmup=*/false)) {
        throw std::runtime_error("Failed to init model");
    }

    Result r;
    auto t0 = std::chrono::high_resolution_clock::now();
    r.total_logprob = model->score_tokens_window_logprob(
        tokens, /*start=*/1, /*end=*/tokens.size(), context, &r.tokens_scored);
    auto t1 = std::chrono::high_resolution_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (r.tokens_scored > 0) {
        r.perplexity = std::exp(-r.total_logprob / r.tokens_scored);
    }
    return r;
}

static void print_result(const char* label, const Result& r) {
    std::cout << "  " << std::setw(28) << std::left << label
              << "  ppl = " << std::fixed << std::setprecision(3) << r.perplexity
              << "   NLL = " << std::setprecision(4) << (-r.total_logprob / std::max<size_t>(1, r.tokens_scored))
              << "   tokens = " << r.tokens_scored
              << "   " << std::setprecision(1) << r.ms << " ms"
              << std::endl;
}

bool test_perplexity_comparison() {
    if (!g_model_path) {
        std::cerr << "CACTUS_TEST_MODEL not set; skipping" << std::endl;
        return true;
    }

    std::cout << "\n  Model: " << g_model_path << std::endl;

    auto tokenizer_model = create_model(g_model_path);
    if (!tokenizer_model ||
        !tokenizer_model->init(g_model_path, 2048, "", /*do_warmup=*/false)) {
        std::cerr << "Failed to initialize tokenizer model" << std::endl;
        return false;
    }
    auto tokens = tokenizer_model->get_tokenizer()->encode(PERPLEXITY_CORPUS);
    tokenizer_model.reset();

    std::cout << "  Corpus: " << tokens.size() << " tokens, "
              << std::strlen(PERPLEXITY_CORPUS) << " chars" << std::endl;

    if (tokens.size() < 8) {
        std::cerr << "Corpus too short" << std::endl;
        return false;
    }

    const size_t context = std::min<size_t>(512, tokens.size());

    std::cout << "\n  [No KV cache — full-precision reference]" << std::endl;
    unsetenv("CACTUS_KV_QUANT_METHOD");
    Result ref = score_corpus_window(g_model_path, tokens, context);
    print_result("FP16 recompute (no cache)", ref);

    std::cout << "\n  [Cache-aware scoring — uses quantized KV]" << std::endl;

    setenv("CACTUS_KV_QUANT_METHOD", "int8", 1);
    Result int8_r = score_corpus(g_model_path, tokens, context);
    print_result("INT8 group (default)", int8_r);

    setenv("CACTUS_KV_QUANT_METHOD", "turboquant", 1);
    Result tq_r = score_corpus(g_model_path, tokens, context);
    print_result("TurboQuant (K2V2)", tq_r);

    unsetenv("CACTUS_KV_QUANT_METHOD");

    std::cout << "\n  TQ/INT8 ppl ratio: "
              << std::fixed << std::setprecision(4)
              << (tq_r.perplexity / std::max(1e-9, int8_r.perplexity))
              << std::endl;
    std::cout << "  INT8/FP16 ppl ratio: "
              << (int8_r.perplexity / std::max(1e-9, ref.perplexity))
              << std::endl;
    std::cout << "  TQ/FP16   ppl ratio: "
              << (tq_r.perplexity / std::max(1e-9, ref.perplexity))
              << std::endl;

    // Sanity: perplexities should be finite and positive.
    bool ok = std::isfinite(int8_r.perplexity) && int8_r.perplexity > 0.0
           && std::isfinite(tq_r.perplexity)  && tq_r.perplexity  > 0.0
           && std::isfinite(ref.perplexity)   && ref.perplexity   > 0.0;
    return ok;
}

int main() {
    TestUtils::TestRunner runner("Perplexity");
    runner.run_test("KV-quant perplexity (INT8 vs TurboQuant)", test_perplexity_comparison());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
