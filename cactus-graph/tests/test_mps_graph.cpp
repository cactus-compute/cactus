#ifdef __APPLE__

#include "test_utils.h"
#include "../cactus_graph.h"
#include "cactus_kernels.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

using namespace TestUtils;
using namespace cactus;

namespace {

// MLP-style chain: hidden -> intermediate -> hidden -> intermediate -> ...
// Both shapes hit MPS_F16 (N >= 1024 for hidden=1152, N=6912 for intermediate).
double bench_mlp_chain_fp16(int num_blocks, size_t M, size_t hidden, size_t intermediate) {
    CactusGraph graph;
    size_t input_id = graph.input({M, hidden}, Precision::FP16);
    size_t cur = input_id;

    std::vector<std::vector<__fp16>> ws;
    std::vector<size_t> w_ids;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.3f, 0.3f);

    auto add_w = [&](size_t out_dim, size_t in_dim) -> size_t {
        ws.emplace_back(out_dim * in_dim);
        for (auto& v : ws.back()) v = static_cast<__fp16>(dis(gen));
        size_t id = graph.input({out_dim, in_dim}, Precision::FP16);
        w_ids.push_back(id);
        return id;
    };

    for (int b = 0; b < num_blocks; ++b) {
        size_t w_up = add_w(intermediate, hidden);
        cur = graph.matmul(cur, w_up, /*pretransposed_rhs=*/true);
        size_t w_down = add_w(hidden, intermediate);
        cur = graph.matmul(cur, w_down, /*pretransposed_rhs=*/true);
    }

    std::vector<__fp16> input_data(M * hidden);
    for (auto& v : input_data) v = static_cast<__fp16>(dis(gen));
    graph.set_input(input_id, input_data.data(), Precision::FP16);
    for (size_t i = 0; i < w_ids.size(); ++i) {
        graph.set_input(w_ids[i], ws[i].data(), Precision::FP16);
    }

    graph.execute();

    Timer t;
    graph.execute();
    return t.elapsed_ms();
}

bool bench_mps_graph_fp16() {
    if (!cactus_mps_available()) {
        std::cout << "  MPS not available, skipping\n";
        return true;
    }

    std::cout << std::left << std::setw(20) << "config"
              << std::right << std::setw(14) << "cpu (ms)"
              << std::setw(14) << "mps (ms)"
              << std::setw(14) << "speedup"
              << "\n";
    std::cout << std::string(62, '-') << "\n";

    struct Cfg { const char* name; size_t M, hidden, inter; int blocks; };
    std::vector<Cfg> cfgs = {
        {"M=322  H=1152 I=6912 b=1", 322, 1152, 6912, 1},
        {"M=322  H=1152 I=6912 b=4", 322, 1152, 6912, 4},
        {"M=322  H=1152 I=6912 b=13", 322, 1152, 6912, 13},
        {"M=64   H=2048 I=8192 b=4", 64, 2048, 8192, 4},
    };

    for (const auto& c : cfgs) {
        cactus_mps_set_enabled(false);
        double cpu_ms = bench_mlp_chain_fp16(c.blocks, c.M, c.hidden, c.inter);

        cactus_mps_set_enabled(true);
        double mps_ms = bench_mlp_chain_fp16(c.blocks, c.M, c.hidden, c.inter);

        std::cout << std::left << std::setw(28) << c.name
                  << std::right << std::setw(14) << std::fixed << std::setprecision(2) << cpu_ms
                  << std::setw(14) << std::fixed << std::setprecision(2) << mps_ms
                  << std::setw(13) << std::fixed << std::setprecision(2) << (cpu_ms / mps_ms) << "x"
                  << "\n";
    }
    return true;
}

}

bool test_dispatch_fp16_correctness() {
    if (!cactus_mps_available()) return true;
    const size_t M = 322, K = 1152, N = 6912;
    CactusGraph graph;
    size_t in = graph.input({M, K}, Precision::FP16);
    size_t w  = graph.input({N, K}, Precision::FP16);
    size_t out = graph.matmul(in, w, true);

    std::mt19937 gen(99);
    std::uniform_real_distribution<float> dis(-0.3f, 0.3f);
    std::vector<__fp16> in_d(M*K), w_d(N*K);
    for (auto& v : in_d) v = static_cast<__fp16>(dis(gen));
    for (auto& v : w_d)  v = static_cast<__fp16>(dis(gen));

    graph.set_input(in, in_d.data(), Precision::FP16);
    graph.set_input(w,  w_d.data(),  Precision::FP16);

    cactus_mps_set_enabled(false);
    graph.execute();
    std::vector<__fp16> c_cpu(M*N);
    std::memcpy(c_cpu.data(), graph.get_output(out), M*N*sizeof(__fp16));

    cactus_mps_set_enabled(true);
    graph.execute();
    std::vector<__fp16> c_mps(M*N);
    std::memcpy(c_mps.data(), graph.get_output(out), M*N*sizeof(__fp16));

    float max_err = 0.0f;
    for (size_t i = 0; i < M*N; ++i) {
        float e = std::abs(static_cast<float>(c_cpu[i]) - static_cast<float>(c_mps[i]));
        if (e > max_err) max_err = e;
    }
    std::cout << "  graph fp16 dispatch max abs err: " << max_err << "\n";
    return max_err < 5.0f;
}

int main() {
    TestRunner runner("MPS Graph-Level Prefill");
    runner.run_test("graph fp16 dispatch correctness", test_dispatch_fp16_correctness());
    runner.print_benchmarks_header();
    bench_mps_graph_fp16();
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}

#else

int main() { return 0; }

#endif
