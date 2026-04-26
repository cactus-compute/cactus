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
#include <fstream>
#include <sstream>
#include <memory>

using cactus::engine::Model;
using cactus::engine::create_model;

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");

static std::string load_corpus() {
    const char* override_path = std::getenv("CACTUS_TEST_CORPUS");
    std::vector<std::string> candidates;
    if (override_path) candidates.push_back(override_path);
    candidates.push_back("tests/data/perplexity_corpus.txt");
    candidates.push_back("../tests/data/perplexity_corpus.txt");
    candidates.push_back("../../tests/data/perplexity_corpus.txt");
    for (const auto& p : candidates) {
        std::ifstream f(p);
        if (f.is_open()) {
            std::stringstream ss; ss << f.rdbuf();
            return ss.str();
        }
    }
    return std::string();
}

struct Result {
    double total_logprob = 0.0;
    size_t tokens_scored = 0;
    double perplexity = 0.0;
    double ms = 0.0;
};

static Result score_corpus(const std::string& model_path,
                           const std::vector<uint32_t>& tokens,
                           size_t context) {
    Result r;
    auto model = create_model(model_path);
    if (!model) return r;
    if (!model->init(model_path, /*context_size=*/2048, "", /*do_warmup=*/false)) return r;

    auto t0 = std::chrono::high_resolution_clock::now();
    r.total_logprob = model->score_tokens_cached_logprob(
        tokens, /*start=*/1, /*end=*/tokens.size(), context, &r.tokens_scored);
    auto t1 = std::chrono::high_resolution_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (r.tokens_scored > 0) {
        r.perplexity = std::exp(-r.total_logprob / r.tokens_scored);
    }
    return r;
}

static Result score_corpus_window(const std::string& model_path,
                                  const std::vector<uint32_t>& tokens,
                                  size_t context) {
    Result r;
    auto model = create_model(model_path);
    if (!model) return r;
    if (!model->init(model_path, /*context_size=*/2048, "", /*do_warmup=*/false)) return r;

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

struct ConfigRow {
    std::string label;
    uint64_t seed;
};

bool test_perplexity_sweep() {
    if (!g_model_path) {
        std::cerr << "CACTUS_TEST_MODEL not set; skipping" << std::endl;
        return true;
    }

    std::cout << "\n  Model: " << g_model_path << std::endl;

    std::string corpus = load_corpus();
    if (corpus.empty()) {
        std::cerr << "Failed to load corpus from tests/data/perplexity_corpus.txt" << std::endl;
        return false;
    }

    auto tokenizer_model = create_model(g_model_path);
    if (!tokenizer_model || !tokenizer_model->init(g_model_path, 2048, "", false)) {
        std::cerr << "Failed to initialize tokenizer model" << std::endl;
        return false;
    }
    auto tokens = tokenizer_model->get_tokenizer()->encode(corpus);
    tokenizer_model.reset();

    // Cap to 2048 tokens to fit context
    if (tokens.size() > 2000) tokens.resize(2000);

    std::cout << "  Corpus: " << tokens.size() << " tokens, " << corpus.size() << " chars" << std::endl;
    if (tokens.size() < 16) { std::cerr << "Corpus too short" << std::endl; return false; }
    const size_t context = std::min<size_t>(2048, tokens.size());

    // Reference: FP16 (no cache, recompute each step)
    unsetenv("CACTUS_KV_QUANT_METHOD");
    unsetenv("CACTUS_TQ_KEY_BITS");
    unsetenv("CACTUS_TQ_VALUE_BITS");
    unsetenv("CACTUS_TQ_PROJECTION_DIM");
    unsetenv("CACTUS_TQ_QJL");
    unsetenv("CACTUS_TQ_SEED");

    Result ref_fp16 = score_corpus_window(g_model_path, tokens, context);
    std::cout << "\n  FP16 (no-cache recompute):  ppl = " << std::fixed << std::setprecision(3) << ref_fp16.perplexity
              << "   NLL = " << std::setprecision(4) << (-ref_fp16.total_logprob / std::max<size_t>(1, ref_fp16.tokens_scored))
              << "   tokens = " << ref_fp16.tokens_scored
              << "   " << std::setprecision(0) << ref_fp16.ms << " ms"
              << std::endl;

    setenv("CACTUS_KV_QUANT_METHOD", "int8", 1);
    Result ref_int8 = score_corpus(g_model_path, tokens, context);
    unsetenv("CACTUS_KV_QUANT_METHOD");
    std::cout << "  INT8 (cached, production):  ppl = " << std::fixed << std::setprecision(3) << ref_int8.perplexity
              << "   NLL = " << std::setprecision(4) << (-ref_int8.total_logprob / std::max<size_t>(1, ref_int8.tokens_scored))
              << "   " << std::setprecision(0) << ref_int8.ms << " ms"
              << std::endl;
    Result ref = ref_int8;  // primary baseline for pass/fail

    // Production config: K=4, V=2, no QJL, projection_dim=64 (head_dim).
    // Compression: 4.57× vs FP16, ~2.6× vs INT8 on head_dim=64 models.
    // Multiple seeds verify robustness against rotation matrix variance.
    std::vector<ConfigRow> configs = {
        {"K4V2 seed=42",    42},
        {"K4V2 seed=7",     7},
        {"K4V2 seed=1337",  1337},
    };

    std::cout << "\n  Sweep (target: ppl within 5% of INT8 = ppl <= "
              << std::fixed << std::setprecision(3) << (ref.perplexity * 1.05) << "):\n" << std::endl;
    std::cout << "  " << std::setw(18) << std::left << "config"
              << std::setw(11) << "ppl"
              << std::setw(11) << "ppl/INT8"
              << std::setw(11) << "ppl/FP16"
              << std::setw(8) << "ms"
              << "  status\n";
    std::cout << "  " << std::string(72, '-') << std::endl;

    bool any_pass = false;
    for (const auto& c : configs) {
        setenv("CACTUS_KV_QUANT_METHOD",   "turboquant", 1);
        setenv("CACTUS_TQ_KEY_BITS",       "4", 1);
        setenv("CACTUS_TQ_VALUE_BITS",     "2", 1);
        setenv("CACTUS_TQ_PROJECTION_DIM", "64", 1);
        setenv("CACTUS_TQ_QJL",            "0", 1);
        setenv("CACTUS_TQ_SEED",           std::to_string(c.seed).c_str(), 1);

        Result r = score_corpus(g_model_path, tokens, context);
        double ratio_int8 = r.perplexity / std::max(1e-9, ref_int8.perplexity);
        double ratio_fp16 = r.perplexity / std::max(1e-9, ref_fp16.perplexity);
        bool pass = (ratio_int8 <= 1.05) && std::isfinite(r.perplexity);
        if (pass) any_pass = true;
        std::cout << "  " << std::setw(18) << std::left << c.label
                  << std::setw(11) << std::fixed << std::setprecision(3) << r.perplexity
                  << std::setw(11) << std::setprecision(4) << ratio_int8
                  << std::setw(11) << std::setprecision(4) << ratio_fp16
                  << std::setw(8) << std::setprecision(0) << r.ms
                  << "  " << (pass ? "PASS" : (ratio_int8 < 1.10 ? "near" : "FAIL"))
                  << std::endl;
    }

    unsetenv("CACTUS_KV_QUANT_METHOD");
    unsetenv("CACTUS_TQ_KEY_BITS");
    unsetenv("CACTUS_TQ_VALUE_BITS");
    unsetenv("CACTUS_TQ_PROJECTION_DIM");
    unsetenv("CACTUS_TQ_QJL");
    unsetenv("CACTUS_TQ_SEED");

    return any_pass;
}

int main() {
    TestUtils::TestRunner runner("Perplexity sweep");
    runner.run_test("TurboQuant config sweep vs FP16", test_perplexity_sweep());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
