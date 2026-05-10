// test_decode_latency: real decode benchmark.
// Prefills a prompt, then calls model->decode() in a tight loop generating
// N tokens, measuring wall-clock tokens/sec. This is the actual production
// generation path.
//
// Env:
//   CACTUS_TEST_MODEL          — required, path to model
//   CACTUS_PPL_CORPUS          — optional, prompt source (default short stub)
//   CACTUS_DECODE_PROMPT_LEN   — prompt token count (default 512)
//   CACTUS_DECODE_GEN_TOKENS   — number of tokens to generate (default 64)
//   CACTUS_DECODE_WARMUP       — warmup tokens before timing (default 8)
//   CACTUS_KV_TQ_BITS / CACTUS_KV_TQ_K_BITS / CACTUS_KV_TQ_V_BITS / CACTUS_KV_TQ_SEED

#include "src/engine.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using cactus::engine::Model;
using cactus::engine::create_model;

static std::string load_corpus() {
    const char* p = std::getenv("CACTUS_PPL_CORPUS");
    if (!p) return "The quick brown fox jumps over the lazy dog. The rain in Spain falls mainly on the plain. To be or not to be, that is the question. ";
    std::ifstream f(p);
    if (!f.is_open()) return "The quick brown fox jumps over the lazy dog. ";
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

static size_t pick_context_size(size_t target_tokens) {
    size_t c = 2048;
    while (c < target_tokens + 256) c *= 2;
    return c;
}

int main() {
    const char* model_path = std::getenv("CACTUS_TEST_MODEL");
    if (!model_path) {
        std::cerr << "CACTUS_TEST_MODEL not set" << std::endl;
        return 1;
    }

    size_t prompt_len  = 512;
    size_t gen_tokens  = 64;
    size_t warmup      = 8;
    if (const char* e = std::getenv("CACTUS_DECODE_PROMPT_LEN"))   prompt_len = std::stoul(e);
    if (const char* e = std::getenv("CACTUS_DECODE_GEN_TOKENS"))   gen_tokens = std::stoul(e);
    if (const char* e = std::getenv("CACTUS_DECODE_WARMUP"))       warmup     = std::stoul(e);

    std::string corpus = load_corpus();

    auto model = create_model(model_path);
    if (!model) { std::cerr << "create_model failed" << std::endl; return 1; }
    size_t ctx_size = pick_context_size(prompt_len + warmup + gen_tokens);
    if (!model->init(model_path, ctx_size, "", /*do_warmup=*/false)) {
        std::cerr << "init failed" << std::endl;
        return 1;
    }

    auto prompt_tokens = model->get_tokenizer()->encode(corpus);
    if (prompt_tokens.size() > prompt_len) prompt_tokens.resize(prompt_len);
    if (prompt_tokens.size() < 16) {
        std::cerr << "prompt too short (got " << prompt_tokens.size() << " tokens)" << std::endl;
        return 1;
    }
    prompt_len = prompt_tokens.size();

    const char* tq_bits  = std::getenv("CACTUS_KV_TQ_BITS");
    const char* tq_kbits = std::getenv("CACTUS_KV_TQ_K_BITS");
    const char* tq_vbits = std::getenv("CACTUS_KV_TQ_V_BITS");
    std::string mode = "TQ default";
    if (tq_kbits || tq_vbits) {
        mode = "TQ K=" + std::string(tq_kbits ? tq_kbits : (tq_bits ? tq_bits : "8"))
             + " V=" + std::string(tq_vbits ? tq_vbits : (tq_bits ? tq_bits : "8"));
    } else if (tq_bits) {
        mode = "TQ K=V=" + std::string(tq_bits);
    }

    std::cout << "Model:        " << model_path << std::endl;
    std::cout << "Mode:         " << mode << std::endl;
    std::cout << "Prompt tok:   " << prompt_len << std::endl;
    std::cout << "Gen tok:      " << gen_tokens << " (after " << warmup << " warmup)" << std::endl;

    // Prefill
    auto t0_pre = std::chrono::high_resolution_clock::now();
    model->prefill(prompt_tokens);
    auto t1_pre = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1_pre - t0_pre).count();
    std::cout << "Prefill:      " << std::fixed << std::setprecision(1) << prefill_ms
              << " ms (" << std::setprecision(2) << (prefill_ms / prompt_len) << " ms/tok)" << std::endl;

    // Greedy decode (temperature=0). Feed the last prompt token, then each generated token.
    std::vector<uint32_t> dec_input = { prompt_tokens.back() };

    // Warmup decodes — not timed.
    for (size_t i = 0; i < warmup; ++i) {
        uint32_t t = model->decode(dec_input, /*temp*/0.0f, /*top_p*/1.0f, /*top_k*/1);
        dec_input = { t };
    }

    // Timed decodes
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < gen_tokens; ++i) {
        uint32_t t = model->decode(dec_input, /*temp*/0.0f, /*top_p*/1.0f, /*top_k*/1);
        dec_input = { t };
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_tok = total_ms / gen_tokens;
    double tok_per_sec = 1000.0 / per_tok;

    std::cout << "Decode total: " << std::fixed << std::setprecision(1) << total_ms << " ms" << std::endl;
    std::cout << "Decode/tok:   " << std::fixed << std::setprecision(2) << per_tok << " ms/tok"
              << "  (" << std::setprecision(1) << tok_per_sec << " tok/sec)" << std::endl;
    return 0;
}
