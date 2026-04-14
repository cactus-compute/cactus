#include "../cactus/npu/npu.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <parakeet_weights_dir> [n_runs]\n", argv[0]);
        return 1;
    }
    const std::string weights = argv[1];
    const int n_runs = argc >= 3 ? std::stoi(argv[2]) : 20;

    if (!cactus::npu::is_npu_available()) {
        fprintf(stderr, "NPU not available\n"); return 1;
    }

    auto encoder = cactus::npu::create_encoder();
    if (!encoder) { fprintf(stderr, "create_encoder failed\n"); return 1; }

    fprintf(stderr, "Loading encoder...\n");
    int64_t t0 = now_ms();
    if (!encoder->load(weights)) { fprintf(stderr, "load failed\n"); return 1; }
    fprintf(stderr, "Load: %lldms\n", (long long)(now_ms() - t0));

    auto in_shape  = encoder->get_input_shape();
    auto out_shape = encoder->get_output_shape();
    int T = in_shape[0], M = in_shape[1];
    int T_out = out_shape[0], D = out_shape[1];
    fprintf(stderr, "Input: [%d, %d]  Output: [%d, %d]\n", T, M, T_out, D);

    std::vector<__fp16> input((size_t)T * M, (__fp16)0.0f);
    std::vector<__fp16> output((size_t)T_out * D);

    // Warmup
    fprintf(stderr, "Warmup...\n");
    encoder->encode(input.data(), output.data(), in_shape);

    fprintf(stderr, "Running %d encode calls...\n", n_runs);
    std::vector<double> times;
    times.reserve(n_runs);
    for (int i = 0; i < n_runs; i++) {
        int64_t t = now_ms();
        encoder->encode(input.data(), output.data(), in_shape);
        double ms = (double)(now_ms() - t);
        times.push_back(ms);
        fprintf(stderr, "  [%2d] %.0fms\n", i+1, ms);
    }

    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double mn  = *std::min_element(times.begin(), times.end());
    double mx  = *std::max_element(times.begin(), times.end());
    // p50
    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    double p50 = sorted[sorted.size()/2];
    double p90 = sorted[(size_t)(sorted.size()*0.9)];

    fprintf(stderr, "\n=== Results (%d runs, T=%d) ===\n", n_runs, T);
    fprintf(stderr, "  avg=%.0fms  min=%.0fms  max=%.0fms  p50=%.0fms  p90=%.0fms\n",
            avg, mn, mx, p50, p90);
    return 0;
}
