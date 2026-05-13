#include "test_utils.h"
#include "src/engine.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <memory>
#include <algorithm>

using cactus::engine::Model;
using cactus::engine::create_model;

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");

static const char* DEFAULT_CORPUS =
    "<start_of_turn>user\nWrite a brief explanation of how compilers work."
    "<end_of_turn>\n<start_of_turn>model\n"
    "A compiler translates source code into machine code through five phases: "
    "lexical analysis tokenizes the source, parsing builds an AST, semantic "
    "analysis checks types and scopes, optimization transforms the IR, and "
    "code generation emits target instructions."
    "<end_of_turn>";

struct Result {
    double total_logprob = 0.0;
    size_t tokens_scored = 0;
    double perplexity = 0.0;
    double ms = 0.0;
    std::vector<float> per_pos_logprob;
    std::vector<uint32_t> per_pos_argmax;
    size_t start_pos = 1;  // position offset of per_pos[0] in the original token sequence
};

static Result score_cached(const std::string& model_path,
                            const std::vector<uint32_t>& tokens,
                            size_t context) {
    auto model = create_model(model_path);
    if (!model) throw std::runtime_error("create_model failed");
    size_t init_ctx = std::max<size_t>(2048, std::max<size_t>(context, tokens.size()) + 64);
    if (!model->init(model_path, init_ctx, "", true)) throw std::runtime_error("init failed");
    Result r;
    r.start_pos = 1;
    auto t0 = std::chrono::high_resolution_clock::now();
    r.total_logprob = model->score_tokens_cached_logprob(
        tokens, 1, tokens.size(), context, &r.tokens_scored,
        &r.per_pos_logprob, &r.per_pos_argmax);
    auto t1 = std::chrono::high_resolution_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (r.tokens_scored > 0) r.perplexity = std::exp(-r.total_logprob / r.tokens_scored);
    return r;
}

static std::vector<size_t> parse_int_list(const std::string& s) {
    std::vector<size_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) {
            try { out.push_back(std::stoul(tok)); } catch (...) {}
        }
    }
    return out;
}

// Position buckets (absolute token index of the *target* position being scored).
// Positions outside the last bucket fall into the open-ended last range.
static const std::vector<size_t> BUCKET_EDGES = {256, 512, 1024, 2048};

static std::string bucket_label(size_t bucket_idx) {
    if (bucket_idx == 0) return "[1," + std::to_string(BUCKET_EDGES[0]) + ")";
    if (bucket_idx < BUCKET_EDGES.size()) {
        return "[" + std::to_string(BUCKET_EDGES[bucket_idx - 1]) + "," + std::to_string(BUCKET_EDGES[bucket_idx]) + ")";
    }
    return "[" + std::to_string(BUCKET_EDGES.back()) + ",+)";
}

static size_t bucket_for(size_t pos) {
    for (size_t i = 0; i < BUCKET_EDGES.size(); ++i) {
        if (pos < BUCKET_EDGES[i]) return i;
    }
    return BUCKET_EDGES.size();
}

struct BucketStats {
    double sum_lp = 0.0;
    size_t n = 0;
    double ppl() const { return n > 0 ? std::exp(-sum_lp / n) : 0.0; }
};

static std::vector<BucketStats> bucket_logprobs(const Result& r) {
    std::vector<BucketStats> buckets(BUCKET_EDGES.size() + 1);
    for (size_t k = 0; k < r.per_pos_logprob.size(); ++k) {
        size_t pos = r.start_pos + k;
        size_t b = bucket_for(pos);
        buckets[b].sum_lp += r.per_pos_logprob[k];
        buckets[b].n += 1;
    }
    return buckets;
}

static double top1_agreement(const Result& a, const Result& b) {
    size_t n = std::min(a.per_pos_argmax.size(), b.per_pos_argmax.size());
    if (n == 0) return 0.0;
    size_t agree = 0;
    for (size_t k = 0; k < n; ++k) {
        if (a.per_pos_argmax[k] == b.per_pos_argmax[k]) ++agree;
    }
    return double(agree) / double(n);
}

bool test_seed_sweep() {
    if (!g_model_path) {
        std::cerr << "CACTUS_TEST_MODEL not set; skipping" << std::endl;
        return true;
    }

    std::vector<size_t> seeds = {7, 42, 137, 1234, 1337, 2024};
    if (const char* s = std::getenv("CACTUS_TQ_SEED_LIST")) {
        seeds = parse_int_list(s);
    }
    if (seeds.empty()) {
        std::cerr << "Empty seed list" << std::endl;
        return false;
    }

    std::string k_bits_env = std::getenv("CACTUS_KV_TQ_K_BITS") ? std::getenv("CACTUS_KV_TQ_K_BITS") : "6";
    std::string v_bits_env = std::getenv("CACTUS_KV_TQ_V_BITS") ? std::getenv("CACTUS_KV_TQ_V_BITS") : "2";
    size_t k_bits = std::stoul(k_bits_env);
    size_t v_bits = std::stoul(v_bits_env);

    std::cout << "\n  Model: " << g_model_path << std::endl;
    std::cout << "  K bits: " << k_bits << "  V bits: " << v_bits << std::endl;

    std::string corpus_buf;
    const char* selected = DEFAULT_CORPUS;
    if (const char* corpus_path = std::getenv("CACTUS_TEST_CORPUS")) {
        std::ifstream f(corpus_path);
        if (f.is_open()) {
            std::stringstream ss; ss << f.rdbuf();
            corpus_buf = ss.str();
            selected = corpus_buf.c_str();
        }
    }

    auto tokenizer_model = create_model(g_model_path);
    if (!tokenizer_model || !tokenizer_model->init(g_model_path, 2048, "", false)) {
        std::cerr << "Failed to initialize tokenizer model" << std::endl;
        return false;
    }
    auto tokens = tokenizer_model->get_tokenizer()->encode(selected);
    tokenizer_model.reset();

    if (tokens.size() < 8) return false;

    size_t ctx_cap = 1024;
    if (const char* c = std::getenv("CACTUS_PPL_CTX")) ctx_cap = std::stoul(c);
    const size_t context = std::min<size_t>(ctx_cap, tokens.size());
    std::cout << "  Corpus: " << tokens.size() << " tokens, context: " << context << std::endl;

    // INT8 baseline
    std::cout << "\n  Running INT8 baseline..." << std::endl;
    unsetenv("CACTUS_KV_TQ_BITS");
    unsetenv("CACTUS_KV_TQ_K_BITS");
    unsetenv("CACTUS_KV_TQ_V_BITS");
    Result int8_r = score_cached(g_model_path, tokens, context);
    std::cout << "  INT8 ppl: " << std::fixed << std::setprecision(3) << int8_r.perplexity
              << "   " << std::setprecision(0) << int8_r.ms << " ms" << std::endl;

    // Print INT8 bucket breakdown
    auto int8_buckets = bucket_logprobs(int8_r);
    std::cout << "\n  INT8 ppl by position bucket:\n";
    for (size_t b = 0; b < int8_buckets.size(); ++b) {
        if (int8_buckets[b].n == 0) continue;
        std::cout << "    " << std::setw(14) << std::left << bucket_label(b)
                  << " n=" << std::setw(5) << int8_buckets[b].n
                  << " ppl=" << std::fixed << std::setprecision(3) << int8_buckets[b].ppl()
                  << std::endl;
    }

    // Sweep seeds
    std::cout << "\n  Sweeping seeds for K=" << k_bits << " V=" << v_bits << ":\n" << std::endl;

    setenv("CACTUS_KV_TQ_K_BITS", std::to_string(k_bits).c_str(), 1);
    setenv("CACTUS_KV_TQ_V_BITS", std::to_string(v_bits).c_str(), 1);

    struct SeedResult {
        size_t seed;
        double ppl;
        double ratio;
        double ms;
        double top1_agree;  // fraction of positions where TQ argmax matches INT8 argmax
        std::vector<BucketStats> buckets;
    };
    std::vector<SeedResult> results;

    // Header
    std::cout << "  " << std::setw(8) << std::left << "seed"
              << std::setw(12) << "TQ ppl"
              << std::setw(10) << "ratio"
              << std::setw(10) << "top1%"
              << std::setw(8) << "ms" << std::endl;
    std::cout << "  " << std::string(48, '-') << std::endl;

    for (size_t seed : seeds) {
        setenv("CACTUS_KV_TQ_SEED", std::to_string(seed).c_str(), 1);
        Result r = score_cached(g_model_path, tokens, context);
        double ratio = r.perplexity / std::max(1e-9, int8_r.perplexity);
        double t1 = top1_agreement(int8_r, r);
        SeedResult sr;
        sr.seed = seed;
        sr.ppl = r.perplexity;
        sr.ratio = ratio;
        sr.ms = r.ms;
        sr.top1_agree = t1;
        sr.buckets = bucket_logprobs(r);
        results.push_back(sr);

        std::cout << "  " << std::setw(8) << std::left << seed
                  << std::setw(12) << std::fixed << std::setprecision(3) << r.perplexity
                  << std::setw(10) << std::setprecision(4) << ratio
                  << std::setw(10) << std::setprecision(2) << (100.0 * t1)
                  << std::setw(8) << std::setprecision(0) << r.ms << std::endl;
    }

    // Per-bucket ratio table per seed
    std::cout << "\n  Position-bucketed TQ/INT8 ratio:\n";
    std::cout << "  " << std::setw(8) << std::left << "seed";
    for (size_t b = 0; b < int8_buckets.size(); ++b) {
        if (int8_buckets[b].n == 0) continue;
        std::cout << std::setw(14) << bucket_label(b);
    }
    std::cout << std::endl;
    std::cout << "  " << std::string(8 + 14 * int8_buckets.size(), '-') << std::endl;
    for (const auto& sr : results) {
        std::cout << "  " << std::setw(8) << std::left << sr.seed;
        for (size_t b = 0; b < sr.buckets.size(); ++b) {
            if (b >= int8_buckets.size() || int8_buckets[b].n == 0) continue;
            double base = int8_buckets[b].ppl();
            double tqp  = sr.buckets[b].ppl();
            double r    = (base > 0 && sr.buckets[b].n > 0) ? tqp / base : 0.0;
            std::cout << std::setw(14) << std::fixed << std::setprecision(3) << r;
        }
        std::cout << std::endl;
    }

    // Best/worst by ratio (overall)
    auto sorted = results;
    std::sort(sorted.begin(), sorted.end(),
              [](const SeedResult& a, const SeedResult& b) { return a.ratio < b.ratio; });

    std::cout << "\n  Best seed (by overall ppl ratio): " << sorted[0].seed
              << " (ratio " << std::fixed << std::setprecision(4) << sorted[0].ratio
              << ", top1 " << std::setprecision(2) << (100.0 * sorted[0].top1_agree) << "%)"
              << std::endl;
    std::cout << "  Worst seed: " << sorted.back().seed
              << " (ratio " << std::setprecision(4) << sorted.back().ratio
              << ", top1 " << std::setprecision(2) << (100.0 * sorted.back().top1_agree) << "%)"
              << std::endl;
    std::cout << "  Spread: " << std::setprecision(2)
              << (100.0 * (sorted.back().ratio - sorted[0].ratio) / std::max(0.01, sorted[0].ratio))
              << "%" << std::endl;

    return true;
}

int main() {
    TestUtils::TestRunner runner("Seed sweep");
    runner.run_test("TQ seed sweep", test_seed_sweep());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
