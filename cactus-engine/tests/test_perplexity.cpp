#include "test_utils.h"
#include "src/engine.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <memory>

using cactus::engine::Model;
using cactus::engine::create_model;

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");

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

static Result score_corpus_cached(const std::string& model_path,
                                  const std::vector<uint32_t>& tokens,
                                  size_t context) {
    auto model = create_model(model_path);
    if (!model) throw std::runtime_error("create_model failed");
    if (!model->init(model_path, 2048, "", true)) throw std::runtime_error("init failed");

    Result r;
    auto t0 = std::chrono::high_resolution_clock::now();
    r.total_logprob = model->score_tokens_cached_logprob(
        tokens, 1, tokens.size(), context, &r.tokens_scored);
    auto t1 = std::chrono::high_resolution_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (r.tokens_scored > 0) r.perplexity = std::exp(-r.total_logprob / r.tokens_scored);
    return r;
}

static Result score_corpus_window(const std::string& model_path,
                                  const std::vector<uint32_t>& tokens,
                                  size_t context) {
    auto model = create_model(model_path);
    if (!model) throw std::runtime_error("create_model failed");
    if (!model->init(model_path, 2048, "", true)) throw std::runtime_error("init failed");

    Result r;
    auto t0 = std::chrono::high_resolution_clock::now();
    r.total_logprob = model->score_tokens_window_logprob(
        tokens, 1, tokens.size(), context, &r.tokens_scored);
    auto t1 = std::chrono::high_resolution_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (r.tokens_scored > 0) r.perplexity = std::exp(-r.total_logprob / r.tokens_scored);
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
    if (!tokenizer_model || !tokenizer_model->init(g_model_path, 2048, "", false)) {
        std::cerr << "Failed to initialize tokenizer model" << std::endl;
        return false;
    }
    auto tokens = tokenizer_model->get_tokenizer()->encode(PERPLEXITY_CORPUS);
    tokenizer_model.reset();

    std::cout << "  Corpus: " << tokens.size() << " tokens, "
              << std::strlen(PERPLEXITY_CORPUS) << " chars" << std::endl;

    if (tokens.size() < 8) return false;

    const size_t context = std::min<size_t>(512, tokens.size());

    Result ref;
    bool have_ref = false;
    if (std::getenv("CACTUS_PPL_SKIP_FP16") == nullptr) {
        std::cout << "\n  [No KV cache - full-precision reference]" << std::endl;
        unsetenv("CACTUS_KV_TQ_BITS");
        try {
            ref = score_corpus_window(g_model_path, tokens, context);
            have_ref = true;
            print_result("FP16 recompute (no cache)", ref);
        } catch (const std::exception& e) {
            std::cerr << "  FP16 reference failed: " << e.what() << std::endl;
            std::cerr << "  Set CACTUS_PPL_SKIP_FP16=1 to skip silently." << std::endl;
        }
    }

    std::cout << "\n  [Cache-aware scoring]" << std::endl;

    unsetenv("CACTUS_KV_TQ_BITS");
    Result int8_r = score_corpus_cached(g_model_path, tokens, context);
    print_result("INT8 group (default)", int8_r);

    setenv("CACTUS_KV_TQ_BITS", "4", 1);
    Result tq4_r = score_corpus_cached(g_model_path, tokens, context);
    print_result("TurboQuant K=4", tq4_r);

    setenv("CACTUS_KV_TQ_BITS", "6", 1);
    Result tq6_r = score_corpus_cached(g_model_path, tokens, context);
    print_result("TurboQuant K=6", tq6_r);

    unsetenv("CACTUS_KV_TQ_BITS");

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  TQ-K4/INT8 ppl ratio:   "
              << (tq4_r.perplexity / std::max(1e-9, int8_r.perplexity)) << std::endl;
    std::cout << "  TQ-K6/INT8 ppl ratio:   "
              << (tq6_r.perplexity / std::max(1e-9, int8_r.perplexity)) << std::endl;
    if (have_ref) {
        std::cout << "  INT8/FP16 ppl ratio:    "
                  << (int8_r.perplexity / std::max(1e-9, ref.perplexity)) << std::endl;
        std::cout << "  TQ-K4/FP16 ppl ratio:   "
                  << (tq4_r.perplexity / std::max(1e-9, ref.perplexity)) << std::endl;
        std::cout << "  TQ-K6/FP16 ppl ratio:   "
                  << (tq6_r.perplexity / std::max(1e-9, ref.perplexity)) << std::endl;
    }

    bool ok = std::isfinite(int8_r.perplexity) && int8_r.perplexity > 0.0
           && std::isfinite(tq4_r.perplexity)  && tq4_r.perplexity  > 0.0
           && std::isfinite(tq6_r.perplexity)  && tq6_r.perplexity  > 0.0;
    return ok;
}

int main() {
    TestUtils::TestRunner runner("Perplexity");
    runner.run_test("KV-quant perplexity (FP16 vs INT8 vs TQ-K4 vs TQ-K6)",
                    test_perplexity_comparison());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
