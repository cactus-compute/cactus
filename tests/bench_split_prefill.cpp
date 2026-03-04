#include "../cactus/cactus.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>

using namespace cactus::engine;

struct BenchResult {
    size_t num_tokens;
    double prefill_ms;
    double tps;
};

BenchResult run_prefill(Model* model, const std::vector<uint32_t>& tokens) {
    BenchResult result;
    result.num_tokens = tokens.size();
    model->reset_cache();
    auto start = std::chrono::high_resolution_clock::now();
    model->prefill(tokens, 256);
    auto end = std::chrono::high_resolution_clock::now();
    result.prefill_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    result.tps = (result.num_tokens * 1000.0) / result.prefill_ms;
    return result;
}

int main(int argc, char* argv[]) {
    std::string model_path = "weights/gemma-3n-e2b-it";
    if (argc > 1) model_path = argv[1];
    std::cout << "Loading model from: " << model_path << std::endl;

    auto model = create_model(model_path);
    if (!model || !model->init(model_path, 2048, "", false)) {
        std::cerr << "Failed to create/init model" << std::endl;
        return 1;
    }

    auto* tokenizer = model->get_tokenizer();
    std::vector<size_t> token_counts = {32, 64, 128, 256, 512};

    std::vector<std::vector<uint32_t>> token_sets;
    for (size_t count : token_counts) {
        std::string text;
        for (size_t i = 0; i < count * 2; i++)
            text += "The quick brown fox jumps over the lazy dog. ";
        auto tokens = tokenizer->encode(text);
        if (tokens.size() > count)
            tokens.resize(count);
        token_sets.push_back(tokens);
    }

    // Warmup both paths
    setenv("CACTUS_DISABLE_SPLIT_PREFILL", "1", 1);
    run_prefill(model.get(), token_sets[0]);
    unsetenv("CACTUS_DISABLE_SPLIT_PREFILL");
    run_prefill(model.get(), token_sets[0]);

    // Interleave baseline and split for each token count to reduce ordering bias
    std::vector<BenchResult> baseline(token_sets.size()), optimized(token_sets.size());

    for (size_t i = 0; i < token_sets.size(); i++) {
        setenv("CACTUS_DISABLE_SPLIT_PREFILL", "1", 1);
        baseline[i] = run_prefill(model.get(), token_sets[i]);

        unsetenv("CACTUS_DISABLE_SPLIT_PREFILL");
        optimized[i] = run_prefill(model.get(), token_sets[i]);
    }

    std::cout << "\n=== Split-Prefill Benchmark ===" << std::endl;
    std::cout << std::setw(10) << "Tokens"
              << std::setw(16) << "Baseline (ms)"
              << std::setw(14) << "Split (ms)"
              << std::setw(10) << "Speedup"
              << std::setw(16) << "Base Tok/s"
              << std::setw(16) << "Split Tok/s" << std::endl;
    std::cout << std::string(82, '-') << std::endl;
    for (size_t i = 0; i < baseline.size(); i++) {
        double speedup = baseline[i].prefill_ms / optimized[i].prefill_ms;
        std::cout << std::setw(10) << baseline[i].num_tokens
                  << std::setw(16) << std::fixed << std::setprecision(1) << baseline[i].prefill_ms
                  << std::setw(14) << std::setprecision(1) << optimized[i].prefill_ms
                  << std::setw(9) << std::setprecision(2) << speedup << "x"
                  << std::setw(16) << std::setprecision(1) << baseline[i].tps
                  << std::setw(16) << std::setprecision(1) << optimized[i].tps << std::endl;
    }

    return 0;
}
