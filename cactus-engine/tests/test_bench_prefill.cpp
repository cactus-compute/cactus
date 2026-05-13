#include "test_utils.h"
#include "../cactus_engine.h"
#ifdef __APPLE__
#include "cactus_kernels.h"
#endif
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace EngineTestUtils;

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");

static const char* g_options = R"({
    "max_tokens": 32,
    "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
    "telemetry_enabled": false
})";

static const char* g_long_prompt = R"([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short summary of the following passage. The history of computing is a long and complex story. The earliest mechanical aids to calculation were the abacus and slide rule. In the 19th century, Charles Babbage designed the Analytical Engine, often considered the first general-purpose mechanical computer, though it was never fully built in his lifetime. Ada Lovelace wrote the first algorithm intended for processing on a machine, making her widely regarded as the first computer programmer. The 20th century brought rapid advances: vacuum tubes, transistors, integrated circuits, and microprocessors. The advent of personal computers in the 1970s and 1980s, followed by the rise of networked computing and the internet in the 1990s, transformed society. Today computing is ubiquitous, embedded in nearly every device we touch. Machine learning and artificial intelligence have emerged as some of the most active areas of research, with implications across science, medicine, and industry. The story is far from complete and continues to evolve at a remarkable pace."}
])";

struct RunResult {
    double ttft_ms = 0;
    double prefill_tps = 0;
    double decode_tps = 0;
    double prefill_tokens = 0;
};

static RunResult run_one(cactus_model_t model, const char* prompt) {
    StreamingData data;
    data.model = model;
    char response[8192];
    int result = cactus_complete(model, prompt, response, sizeof(response),
                                 g_options, nullptr, stream_callback, &data, nullptr, 0);
    RunResult r;
    if (result <= 0) return r;
    Metrics m;
    m.parse(response);
    r.ttft_ms = m.ttft;
    r.prefill_tps = m.prefill_tps;
    r.decode_tps = m.decode_tps;
    r.prefill_tokens = m.prefill_tokens;
    return r;
}

int main() {
    if (!g_model_path) {
        std::cerr << "CACTUS_TEST_MODEL not set\n";
        return 1;
    }
    std::cout << "Model: " << g_model_path << "\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "Failed to init model\n";
        return 1;
    }

#ifdef __APPLE__
    bool mps_avail = cactus_mps_available();
    std::cout << "MPS available: " << (mps_avail ? "yes" : "no") << "\n\n";
#endif

    std::cout << std::left << std::setw(8) << "mode"
              << std::right << std::setw(12) << "prefill_tok"
              << std::setw(14) << "ttft (ms)"
              << std::setw(16) << "prefill_tps"
              << std::setw(14) << "decode_tps"
              << "\n";
    std::cout << std::string(64, '-') << "\n";

    auto print = [&](const char* mode, const RunResult& r) {
        std::cout << std::left << std::setw(8) << mode
                  << std::right << std::setw(12) << std::fixed << std::setprecision(0) << r.prefill_tokens
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.ttft_ms
                  << std::setw(16) << std::fixed << std::setprecision(1) << r.prefill_tps
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.decode_tps
                  << "\n";
    };

    // Warmup
#ifdef __APPLE__
    cactus_mps_set_enabled(false);
#endif
    run_one(model, g_long_prompt);

    const int RUNS = 3;
    for (int i = 0; i < RUNS; ++i) {
#ifdef __APPLE__
        cactus_mps_set_enabled(false);
#endif
        RunResult r_cpu = run_one(model, g_long_prompt);
        print("CPU", r_cpu);

#ifdef __APPLE__
        if (mps_avail) {
            cactus_mps_set_enabled(true);
            RunResult r_mps = run_one(model, g_long_prompt);
            print("MPS", r_mps);
            double speedup = r_cpu.prefill_tps > 0 ? r_mps.prefill_tps / r_cpu.prefill_tps : 0.0;
            std::cout << "    speedup: " << std::fixed << std::setprecision(2) << speedup << "x prefill\n";
        }
#endif
    }

    cactus_destroy(model);
    return 0;
}
