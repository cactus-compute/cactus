// Sweeps TurboQuant K/V configs and seeds across multiple context lengths
// against a long text corpus. INT8 KV cache no longer exists — TQ is the only
// cached path. The FP16-recompute reference (no cache) is skipped above
// CACTUS_FP16_MAX_CTX (default 1024) because it is O(N^2). Note: the
// FP16-recompute path uses different kernels than the cached-decode path;
// the resulting TQ/FP16 ratio bundles quantization error with cached-vs-
// recompute pipeline drift.
//
// Usage:
//   CACTUS_TEST_MODEL=weights_v2/gemma-4-e2b-it-fp16 ./test_perplexity_context_sweep
//
// Optional:
//   CACTUS_PPL_CORPUS=path/to/corpus.txt    default: tests/data/perplexity_corpus_long.txt
//   CACTUS_CTX_SWEEP=256,512,1024,2000      comma-separated token caps
//   CACTUS_TQ_SEEDS=7,1337                  comma-separated seeds
//   CACTUS_SWEEP_CONFIGS=K4V2,K4V4,K6V2,K6V4  K and V each one of {2,4,6,8}
//   CACTUS_FP16_MAX_CTX=1024                skip FP16 reference above this ctx

#include "test_utils.h"
#include "src/engine.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
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
    const char* override_path = std::getenv("CACTUS_PPL_CORPUS");
    std::vector<std::string> candidates;
    if (override_path) candidates.push_back(override_path);
    candidates.push_back("tests/data/perplexity_corpus_long.txt");
    candidates.push_back("../tests/data/perplexity_corpus_long.txt");
    candidates.push_back("../../tests/data/perplexity_corpus_long.txt");
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

static size_t pick_context_size(size_t target_tokens) {
    size_t c = 2048;
    while (c < target_tokens + 64) c *= 2;
    return c;
}

static Result score_corpus_cached(const std::string& model_path,
                                  const std::vector<uint32_t>& tokens,
                                  size_t context) {
    Result r;
    auto model = create_model(model_path);
    if (!model) return r;
    size_t ctx_size = pick_context_size(tokens.size());
    if (!model->init(model_path, ctx_size, "", /*do_warmup=*/false)) return r;

    auto t0 = std::chrono::high_resolution_clock::now();
    size_t start = std::min<size_t>(64, tokens.size() / 4);
    if (start < 1) start = 1;
    r.total_logprob = model->score_tokens_cached_logprob(
        tokens, start, tokens.size(), context, &r.tokens_scored);
    auto t1 = std::chrono::high_resolution_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (r.tokens_scored > 0) {
        r.perplexity = std::exp(-r.total_logprob / r.tokens_scored);
    }
    return r;
}

static Result score_corpus_fp16(const std::string& model_path,
                                const std::vector<uint32_t>& tokens) {
    Result r;
    auto model = create_model(model_path);
    if (!model) return r;
    size_t ctx_size = pick_context_size(tokens.size());
    if (!model->init(model_path, ctx_size, "", /*do_warmup=*/false)) return r;

    auto t0 = std::chrono::high_resolution_clock::now();
    size_t start = std::min<size_t>(64, tokens.size() / 4);
    if (start < 1) start = 1;
    r.total_logprob = model->score_tokens_window_logprob(
        tokens, start, tokens.size(), tokens.size(), &r.tokens_scored);
    auto t1 = std::chrono::high_resolution_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (r.tokens_scored > 0) {
        r.perplexity = std::exp(-r.total_logprob / r.tokens_scored);
    }
    return r;
}

static std::vector<std::string> parse_csv(const char* env, const std::string& def) {
    std::vector<std::string> v;
    std::string s = env ? env : def;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(tok);
    return v;
}

static std::vector<size_t> parse_context_sweep() {
    std::vector<size_t> v;
    for (const auto& tok : parse_csv(std::getenv("CACTUS_CTX_SWEEP"), "256,512,1024,2000")) {
        v.push_back(static_cast<size_t>(std::stoul(tok)));
    }
    return v;
}

static void clear_tq_env() {
    unsetenv("CACTUS_KV_TQ_BITS");
    unsetenv("CACTUS_KV_TQ_K_BITS");
    unsetenv("CACTUS_KV_TQ_V_BITS");
    unsetenv("CACTUS_KV_TQ_SEED");
    unsetenv("CACTUS_KV_REF_INT8");
}

bool test_perplexity_context_sweep() {
    if (!g_model_path) {
        std::cerr << "CACTUS_TEST_MODEL not set; skipping" << std::endl;
        return true;
    }

    std::cout << "\n  Model: " << g_model_path << std::endl;

    std::string corpus = load_corpus();
    if (corpus.empty()) {
        std::cerr << "Failed to load corpus (set CACTUS_PPL_CORPUS or place at tests/data/perplexity_corpus_long.txt)" << std::endl;
        return false;
    }

    auto tokenizer_model = create_model(g_model_path);
    if (!tokenizer_model || !tokenizer_model->init(g_model_path, 2048, "", false)) {
        std::cerr << "Failed to initialize tokenizer model" << std::endl;
        return false;
    }
    auto full_tokens = tokenizer_model->get_tokenizer()->encode(corpus);
    tokenizer_model.reset();

    std::cout << "  Corpus tokenized: " << full_tokens.size() << " tokens" << std::endl;

    std::vector<size_t> ctx_lengths = parse_context_sweep();
    std::cout << "  Context sweep:";
    for (auto c : ctx_lengths) std::cout << " " << c;
    std::cout << std::endl;

    std::vector<std::string> seeds = parse_csv(std::getenv("CACTUS_TQ_SEEDS"), "7,1337");
    std::vector<std::string> cfgs  = parse_csv(std::getenv("CACTUS_SWEEP_CONFIGS"), "K4V2,K4V4,K6V2,K6V4");

    auto parse_kv = [](const std::string& cfg, std::string& kbits, std::string& vbits) {
        kbits = std::string(1, cfg[1]);
        vbits = std::string(1, cfg[3]);
    };

    std::cout << "  Seeds:"; for (auto& s : seeds) std::cout << " " << s;
    std::cout << "\n  Configs:"; for (auto& c : cfgs) std::cout << " " << c;
    std::cout << "\n";

    size_t fp16_max_ctx = 1024;
    if (const char* env = std::getenv("CACTUS_FP16_MAX_CTX")) {
        fp16_max_ctx = static_cast<size_t>(std::stoul(env));
    }

    std::cout << "\n  " << std::setw(8) << std::left << "ctx"
              << std::setw(8) << "config"
              << std::setw(7) << "seed"
              << std::setw(11) << "FP16 ppl"
              << std::setw(11) << "TQ ppl"
              << std::setw(11) << "TQ/FP16"
              << std::setw(10) << "ms"
              << "\n";
    std::cout << "  " << std::string(72, '-') << std::endl;

    bool any_ok = false;
    for (size_t ctx : ctx_lengths) {
        if (ctx > full_tokens.size()) {
            std::cout << "  ctx=" << ctx << " skipped (corpus only "
                      << full_tokens.size() << " tokens)" << std::endl;
            continue;
        }
        std::vector<uint32_t> tokens(full_tokens.begin(), full_tokens.begin() + ctx);

        Result fp16_r;
        if (ctx <= fp16_max_ctx) {
            clear_tq_env();
            fp16_r = score_corpus_fp16(g_model_path, tokens);
        }

        for (const auto& cfg : cfgs) {
            std::string kbits, vbits;
            parse_kv(cfg, kbits, vbits);
            for (const auto& seed : seeds) {
                clear_tq_env();
                setenv("CACTUS_KV_TQ_K_BITS", kbits.c_str(), 1);
                setenv("CACTUS_KV_TQ_V_BITS", vbits.c_str(), 1);
                setenv("CACTUS_KV_TQ_SEED", seed.c_str(), 1);
                Result tq = score_corpus_cached(g_model_path, tokens, ctx);

                double ratio_fp16 = (fp16_r.perplexity > 0)
                                  ? tq.perplexity / fp16_r.perplexity : 0.0;

                std::cout << "  " << std::setw(8) << std::left << ctx
                          << std::setw(8) << cfg
                          << std::setw(7) << seed;
                if (fp16_r.perplexity > 0) {
                    std::cout << std::setw(11) << std::fixed << std::setprecision(3) << fp16_r.perplexity;
                } else {
                    std::cout << std::setw(11) << "skip";
                }
                std::cout << std::setw(11) << std::fixed << std::setprecision(3) << tq.perplexity;
                if (ratio_fp16 > 0) {
                    std::cout << std::setw(11) << std::setprecision(4) << ratio_fp16;
                } else {
                    std::cout << std::setw(11) << "-";
                }
                std::cout << std::setw(10) << std::fixed << std::setprecision(0) << tq.ms
                          << std::endl;

                if (std::isfinite(tq.perplexity) && tq.perplexity > 0) any_ok = true;
            }
        }
        clear_tq_env();
    }

    return any_ok;
}

int main() {
    TestUtils::TestRunner runner("Perplexity context sweep");
    runner.run_test("TQ K/V configs across context lengths vs FP16",
                    test_perplexity_context_sweep());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
