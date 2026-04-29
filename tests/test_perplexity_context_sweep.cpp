// Sweeps TurboQuant K4V2 / K4V4 perplexity vs INT8 baseline across multiple
// context lengths for one model. Skips FP16 reference (O(N^2)) — INT8 is the
// production baseline that matters in practice.
//
// Usage:
//   CACTUS_TEST_MODEL=weights/<model>      ./test_perplexity_context_sweep
//
// Optional:
//   CACTUS_TEST_CORPUS=path/to/corpus.txt  (default: tests/data/perplexity_corpus.txt)
//   CACTUS_CTX_SWEEP=256,512,1024,2000     comma-separated token caps
//   CACTUS_TQ_SEED=42                      seed used for both K4V2 and K4V4

#include "test_utils.h"
#include "../cactus/cactus.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <memory>

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

static size_t pick_context_size(size_t target_tokens) {
    size_t c = 2048;
    while (c < target_tokens + 64) c *= 2;
    return c;
}

static Result score_corpus(const std::string& model_path,
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
        tokens, /*start=*/start, /*end=*/tokens.size(), context, &r.tokens_scored);
    auto t1 = std::chrono::high_resolution_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (r.tokens_scored > 0) {
        r.perplexity = std::exp(-r.total_logprob / r.tokens_scored);
    }
    return r;
}

static std::vector<size_t> parse_context_sweep() {
    std::vector<size_t> v;
    const char* env = std::getenv("CACTUS_CTX_SWEEP");
    std::string s = env ? env : "256,512,1024,2000";
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) v.push_back(static_cast<size_t>(std::stoul(tok)));
    }
    return v;
}

static void clear_tq_env() {
    unsetenv("CACTUS_KV_QUANT_METHOD");
    unsetenv("CACTUS_TQ_KEY_BITS");
    unsetenv("CACTUS_TQ_VALUE_BITS");
    unsetenv("CACTUS_TQ_SEED");
}

bool test_perplexity_context_sweep() {
    if (!g_model_path) {
        std::cerr << "CACTUS_TEST_MODEL not set; skipping" << std::endl;
        return true;
    }

    std::cout << "\n  Model: " << g_model_path << std::endl;

    std::string corpus = load_corpus();
    if (corpus.empty()) {
        std::cerr << "Failed to load corpus" << std::endl;
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

    // Seeds and configs swept simultaneously. Override:
    //   CACTUS_TQ_SEEDS=7,1337         (default 7,1337)
    //   CACTUS_SWEEP_CONFIGS=K4V2,K4V4,K6V2,K6V4 (default all four)
    auto parse_csv = [](const char* env, const std::string& def) {
        std::vector<std::string> v;
        std::string s = env ? env : def;
        std::stringstream ss(s); std::string tok;
        while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(tok);
        return v;
    };
    std::vector<std::string> seeds = parse_csv(std::getenv("CACTUS_TQ_SEEDS"), "7,1337");
    std::vector<std::string> cfgs  = parse_csv(std::getenv("CACTUS_SWEEP_CONFIGS"), "K4V2,K4V4,K6V2,K6V4");

    auto parse_kv = [](const std::string& cfg, std::string& kbits, std::string& vbits) {
        // Format: K<digit>V<digit> e.g. K4V2, K6V4
        kbits = std::string(1, cfg[1]);
        vbits = std::string(1, cfg[3]);
    };

    std::cout << "\n  Seeds: "; for (auto& s : seeds) std::cout << s << " ";
    std::cout << "\n  Configs: "; for (auto& c : cfgs) std::cout << c << " ";
    std::cout << "\n";

    std::cout << "\n  " << std::setw(8) << std::left << "ctx"
              << std::setw(8) << "config"
              << std::setw(7) << "seed"
              << std::setw(11) << "INT8 ppl"
              << std::setw(11) << "TQ ppl"
              << std::setw(13) << "TQ/INT8"
              << std::setw(10) << "INT8 ms"
              << std::setw(10) << "TQ ms"
              << "\n";
    std::cout << "  " << std::string(78, '-') << std::endl;

    bool any_ok = false;
    for (size_t ctx : ctx_lengths) {
        if (ctx > full_tokens.size()) {
            std::cout << "  ctx=" << ctx << " skipped (corpus only "
                      << full_tokens.size() << " tokens)" << std::endl;
            continue;
        }
        std::vector<uint32_t> tokens(full_tokens.begin(), full_tokens.begin() + ctx);

        clear_tq_env();
        setenv("CACTUS_KV_QUANT_METHOD", "int8", 1);
        Result int8_r = score_corpus(g_model_path, tokens, ctx);

        for (const auto& cfg : cfgs) {
            std::string kbits, vbits;
            parse_kv(cfg, kbits, vbits);
            for (const auto& seed : seeds) {
                clear_tq_env();
                setenv("CACTUS_KV_QUANT_METHOD", "turboquant", 1);
                setenv("CACTUS_TQ_KEY_BITS", kbits.c_str(), 1);
                setenv("CACTUS_TQ_VALUE_BITS", vbits.c_str(), 1);
                setenv("CACTUS_TQ_SEED", seed.c_str(), 1);
                Result tq = score_corpus(g_model_path, tokens, ctx);

                double ratio = tq.perplexity / std::max(1e-9, int8_r.perplexity);
                std::cout << "  " << std::setw(8) << std::left << ctx
                          << std::setw(8) << cfg
                          << std::setw(7) << seed
                          << std::setw(11) << std::fixed << std::setprecision(3) << int8_r.perplexity
                          << std::setw(11) << std::setprecision(3) << tq.perplexity
                          << std::setw(13) << std::setprecision(4) << ratio
                          << std::setw(10) << std::setprecision(0) << int8_r.ms
                          << std::setw(10) << std::setprecision(0) << tq.ms
                          << std::endl;

                if (std::isfinite(tq.perplexity)) any_ok = true;
            }
        }
        clear_tq_env();
    }

    return any_ok;
}

int main() {
    TestUtils::TestRunner runner("Perplexity context sweep");
    runner.run_test("INT8 vs K4V2/K4V4 across context lengths", test_perplexity_context_sweep());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
