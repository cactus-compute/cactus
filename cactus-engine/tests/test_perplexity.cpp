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

using cactus::engine::Model;
using cactus::engine::create_model;

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");

static const char* PERPLEXITY_CORPUS_PROSE =
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

static const char* PERPLEXITY_CORPUS =
    "<start_of_turn>user\nWrite a detailed explanation of how compilers work, "
    "covering lexical analysis, parsing, semantic analysis, optimization, and "
    "code generation. Use specific examples from production compilers."
    "<end_of_turn>\n<start_of_turn>model\n"
    "Compilers translate human-readable source code into machine instructions "
    "that processors can execute. The translation pipeline has five canonical "
    "phases. First, lexical analysis (also called tokenization or scanning) "
    "reads the source as a stream of characters and groups them into tokens "
    "such as identifiers, keywords, operators, and literals. Tools like flex "
    "and re2c automate this phase using regular expressions. Second, parsing "
    "consumes the token stream and builds an abstract syntax tree (AST) "
    "according to the language's grammar. Most modern compilers use either "
    "recursive descent (Clang, V8) or generated LR parsers (yacc, bison). "
    "Third, semantic analysis traverses the AST to enforce typing rules, "
    "resolve names and scopes, check for undefined variables, and attach "
    "type information to expressions. Fourth, optimization transforms an "
    "intermediate representation to reduce the number of instructions or "
    "memory accesses while preserving semantics. LLVM's IR is the canonical "
    "example: SSA form makes data-flow analysis tractable, enabling passes "
    "like dead code elimination, loop-invariant code motion, common "
    "subexpression elimination, and inlining. Finally, code generation lowers "
    "the optimized IR into target machine code, handling register allocation, "
    "instruction selection, and scheduling. GCC and LLVM both use modular "
    "back-ends that emit code for x86, ARM, RISC-V, and other architectures "
    "from a shared front-end. Just-in-time compilers like V8's TurboFan and "
    "Java's HotSpot interleave these phases at runtime, recompiling hot code "
    "with profile-guided optimization."
    "<end_of_turn>";

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
    const char* corpus_sel = std::getenv("CACTUS_PPL_CORPUS");
    std::string corpus_buf;
    const char* selected = nullptr;
    if (corpus_sel && std::string(corpus_sel) == "prose") {
        selected = PERPLEXITY_CORPUS_PROSE;
    } else if (corpus_sel && std::ifstream(corpus_sel).good()) {
        std::ifstream f(corpus_sel);
        std::stringstream ss; ss << f.rdbuf();
        corpus_buf = ss.str();
        selected = corpus_buf.c_str();
    } else {
        selected = PERPLEXITY_CORPUS;
    }
    auto tokens = tokenizer_model->get_tokenizer()->encode(selected);
    tokenizer_model.reset();

    std::cout << "  Corpus: " << tokens.size() << " tokens, "
              << std::strlen(PERPLEXITY_CORPUS) << " chars" << std::endl;

    if (tokens.size() < 8) return false;

    size_t ctx_cap = 512;
    if (const char* c = std::getenv("CACTUS_PPL_CTX")) ctx_cap = std::stoul(c);
    const size_t context = std::min<size_t>(ctx_cap, tokens.size());
    std::cout << "  Context: " << context << " tokens" << std::endl;

    Result ref;
    bool have_ref = false;
    if (std::getenv("CACTUS_PPL_SKIP_FP16") == nullptr) {
        std::cout << "\n  [No KV cache - full-precision reference]" << std::endl;
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

    const char* only_mode = std::getenv("CACTUS_PPL_ONLY");
    Result tq4_r{}, tq6_r{}, tq8_r{};
    bool run_k4 = !only_mode || std::string(only_mode) == "k4";
    bool run_k6 = !only_mode || std::string(only_mode) == "k6";
    bool run_k8 = !only_mode || std::string(only_mode) == "k8";

    if (run_k4) {
        setenv("CACTUS_KV_TQ_BITS", "4", 1);
        tq4_r = score_corpus_cached(g_model_path, tokens, context);
        print_result("TurboQuant K=4", tq4_r);
    }

    if (run_k6) {
        setenv("CACTUS_KV_TQ_BITS", "6", 1);
        tq6_r = score_corpus_cached(g_model_path, tokens, context);
        print_result("TurboQuant K=6", tq6_r);
    }

    if (run_k8) {
        setenv("CACTUS_KV_TQ_BITS", "8", 1);
        tq8_r = score_corpus_cached(g_model_path, tokens, context);
        print_result("TurboQuant K=8", tq8_r);
    }

    unsetenv("CACTUS_KV_TQ_BITS");

    if (only_mode) return true;

    if (have_ref) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  TQ-K4/FP16 ppl ratio:   "
                  << (tq4_r.perplexity / std::max(1e-9, ref.perplexity)) << std::endl;
        std::cout << "  TQ-K6/FP16 ppl ratio:   "
                  << (tq6_r.perplexity / std::max(1e-9, ref.perplexity)) << std::endl;
        std::cout << "  TQ-K8/FP16 ppl ratio:   "
                  << (tq8_r.perplexity / std::max(1e-9, ref.perplexity)) << std::endl;
    }

    bool ok = std::isfinite(tq4_r.perplexity) && tq4_r.perplexity > 0.0
           && std::isfinite(tq6_r.perplexity) && tq6_r.perplexity > 0.0
           && std::isfinite(tq8_r.perplexity) && tq8_r.perplexity > 0.0;
    return ok;
}

int main() {
    TestUtils::TestRunner runner("Perplexity");
    runner.run_test("KV-quant perplexity (FP16 vs TQ-K4 vs TQ-K6 vs TQ-K8)",
                    test_perplexity_comparison());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
