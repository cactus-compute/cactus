#ifdef __APPLE__

#include "test_utils.h"
#include "cactus_kernels.h"
#include <vector>
#include <cmath>
#include <random>

using namespace TestUtils;

bool test_mps_available() {
    return cactus_mps_available();
}

bool test_mps_matmul_f16_correctness() {
    if (!cactus_mps_available()) return true;

    const size_t M = 64, K = 1024, N = 256;
    std::vector<__fp16> A(M * K);
    std::vector<__fp16> BT(N * K);
    std::vector<__fp16> C_cpu(M * N);
    std::vector<__fp16> C_mps(M * N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    for (auto& v : A) v = static_cast<__fp16>(dis(gen));
    for (auto& v : BT) v = static_cast<__fp16>(dis(gen));

    cactus_matmul_f16(A.data(), BT.data(), C_cpu.data(), M, K, N);
    cactus_matmul_f16_mps(A.data(), BT.data(), C_mps.data(), M, K, N);

    float max_err = 0.0f;
    for (size_t i = 0; i < M * N; ++i) {
        float e = std::abs(static_cast<float>(C_mps[i]) - static_cast<float>(C_cpu[i]));
        if (e > max_err) max_err = e;
    }
    std::cout << "  MPS fp16 matmul max abs err: " << max_err << "\n";
    return max_err < 1.0f;
}

bool test_mps_matmul_f16_shapes() {
    if (!cactus_mps_available()) return true;

    struct Shape { size_t M, K, N; };
    std::vector<Shape> shapes = {
        {1, 1024, 1024},
        {64, 1152, 1024},
        {128, 1152, 6912},
        {256, 6912, 1152},
        {322, 1152, 256},
    };

    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    for (auto& s : shapes) {
        std::vector<__fp16> A(s.M * s.K);
        std::vector<__fp16> BT(s.N * s.K);
        std::vector<__fp16> C_cpu(s.M * s.N);
        std::vector<__fp16> C_mps(s.M * s.N);
        for (auto& v : A) v = static_cast<__fp16>(dis(gen));
        for (auto& v : BT) v = static_cast<__fp16>(dis(gen));

        cactus_matmul_f16(A.data(), BT.data(), C_cpu.data(), s.M, s.K, s.N);
        cactus_matmul_f16_mps(A.data(), BT.data(), C_mps.data(), s.M, s.K, s.N);
    
        float max_err = 0.0f;
        for (size_t i = 0; i < s.M * s.N; ++i) {
            float e = std::abs(static_cast<float>(C_mps[i]) - static_cast<float>(C_cpu[i]));
            if (e > max_err) max_err = e;
        }
        std::cout << "  MPS fp16 (" << s.M << "x" << s.K << "x" << s.N << ") max err: " << max_err << "\n";
        if (max_err > 2.0f) return false;
    }
    return true;
}

bool bench_mps_vs_cpu_f16() {
    if (!cactus_mps_available()) return true;

    struct Shape { const char* name; size_t M, K, N; };
    std::vector<Shape> shapes = {
        {"f16  322x1152x1024", 322, 1152, 1024},
        {"f16  322x1152x6912", 322, 1152, 6912},
        {"f16  322x6912x1152", 322, 6912, 1152},
        {"f16 1024x4096x4096", 1024, 4096, 4096},
    };

    std::mt19937 gen(11);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    for (auto& s : shapes) {
        std::vector<__fp16> A(s.M * s.K), BT(s.N * s.K), C(s.M * s.N);
        for (auto& v : A) v = static_cast<__fp16>(dis(gen));
        for (auto& v : BT) v = static_cast<__fp16>(dis(gen));

        cactus_matmul_f16(A.data(), BT.data(), C.data(), s.M, s.K, s.N);
        cactus_matmul_f16_mps(A.data(), BT.data(), C.data(), s.M, s.K, s.N);

        const int N_RUNS = 5;
        Timer cpu_t;
        for (int i = 0; i < N_RUNS; ++i) {
            cactus_matmul_f16(A.data(), BT.data(), C.data(), s.M, s.K, s.N);
        }
        double cpu_ms = cpu_t.elapsed_ms() / N_RUNS;

        Timer mps_t;
        for (int i = 0; i < N_RUNS; ++i) {
            cactus_matmul_f16_mps(A.data(), BT.data(), C.data(), s.M, s.K, s.N);
        }
        double mps_ms = mps_t.elapsed_ms() / N_RUNS;

        double gflops = 2.0 * s.M * s.K * s.N / 1e9;
        double cpu_tflops = gflops / (cpu_ms / 1000.0) / 1000.0;
        double mps_tflops = gflops / (mps_ms / 1000.0) / 1000.0;

        std::cout << "  " << s.name
                  << "  cpu=" << std::fixed << std::setprecision(2) << cpu_ms << "ms (" << cpu_tflops << "TF)"
                  << "  mps=" << mps_ms << "ms (" << mps_tflops << "TF)"
                  << "  speedup=" << std::setprecision(2) << (cpu_ms / mps_ms) << "x\n";
    }
    return true;
}

struct MinimalCQ4 {
    uint32_t K, N, group_size, num_groups;
    std::vector<__fp16> codebook;
    std::vector<__fp16> input_scale;
    std::vector<__fp16> input_scale_recip;
    std::vector<__fp16> norms;
    std::vector<int8_t> left_signs;
    std::vector<int8_t> right_signs;
    std::vector<uint32_t> permutation;
    std::vector<uint8_t> packed;

    MinimalCQ4(uint32_t k, uint32_t n, uint32_t gs, uint32_t seed = 42)
        : K(k), N(n), group_size(gs), num_groups(k / gs) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        codebook.resize(16);
        for (auto& v : codebook) v = static_cast<__fp16>(dist(gen));
        input_scale.resize(K);
        input_scale_recip.resize(K);
        for (uint32_t i = 0; i < K; i++) {
            float s = 0.5f + std::abs(dist(gen));
            input_scale[i] = static_cast<__fp16>(s);
            input_scale_recip[i] = static_cast<__fp16>(1.f / s);
        }
        norms.resize(static_cast<size_t>(N) * num_groups);
        for (auto& v : norms) v = static_cast<__fp16>(dist(gen) * 0.1f);
        left_signs.resize(group_size);
        right_signs.resize(group_size);
        for (auto& v : left_signs) v = (gen() & 1) ? 1 : -1;
        for (auto& v : right_signs) v = (gen() & 1) ? 1 : -1;
        permutation.resize(group_size);
        for (uint32_t i = 0; i < group_size; i++) permutation[i] = i;
        size_t pgb = cactus_quant_packed_group_bytes(4, group_size);
        packed.resize(static_cast<size_t>(N) * num_groups * pgb);
        for (auto& v : packed) v = static_cast<uint8_t>(gen() & 0xFF);
    }

    CactusQuantMatrix matrix() const {
        return CactusQuantMatrix{
            .bits = 4, .K = K, .N = N,
            .group_size = group_size, .num_groups = num_groups,
            .flags = CACTUS_QUANT_FLAG_CODE_ORDERED_INDICES,
            .codebook = codebook.data(),
            .input_scale = input_scale.data(),
            .input_scale_recip = input_scale_recip.data(),
            .norms = norms.data(),
            .packed_indices = packed.data(),
            .left_signs = left_signs.data(),
            .right_signs = right_signs.data(),
            .permutation = permutation.data(),
            .expanded = nullptr,
            .norm_f32 = nullptr,
        };
    }
};

bool test_mps_cq4_matmul_correctness() {
    if (!cactus_mps_available()) return true;

    const uint32_t K = 1152, N = 1024, gs = 128;
    const uint32_t M = 64;
    MinimalCQ4 cq(K, N, gs, 123);
    CactusQuantMatrix mat = cq.matrix();

    std::mt19937 gen(91);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<__fp16> A(static_cast<size_t>(M) * K);
    for (auto& v : A) v = static_cast<__fp16>(dist(gen));

    std::vector<__fp16> C_cpu(static_cast<size_t>(M) * N, static_cast<__fp16>(0));
    std::vector<__fp16> C_mps(static_cast<size_t>(M) * N, static_cast<__fp16>(0));

    cactus_quant_4bit_gemm(&mat, A.data(), M, C_cpu.data());
    cactus_quant_4bit_matmul_mps(&mat, A.data(), M, C_mps.data());

    float max_err = 0.0f;
    double sse = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(M) * N; ++i) {
        float e = std::abs(static_cast<float>(C_mps[i]) - static_cast<float>(C_cpu[i]));
        sse += static_cast<double>(e) * e;
        if (e > max_err) max_err = e;
    }
    double rmse = std::sqrt(sse / (static_cast<size_t>(M) * N));
    std::cout << "  MPS CQ4 matmul max abs err: " << max_err << "  rmse: " << rmse << "\n";
    return max_err < 0.5f;
}

bool bench_mps_vs_cpu_cq4() {
    if (!cactus_mps_available()) return true;

    struct Shape { const char* name; uint32_t M, K, N; uint32_t gs; };
    std::vector<Shape> shapes = {
        {"cq4 322x1152x1024", 322, 1152, 1024, 128},
        {"cq4 322x1152x6912", 322, 1152, 6912, 128},
        {"cq4 322x6912x1152", 322, 6912, 1152, 128},
    };

    std::mt19937 gen(13);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (auto& s : shapes) {
        MinimalCQ4 cq(s.K, s.N, s.gs, 17);
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> A(static_cast<size_t>(s.M) * s.K);
        for (auto& v : A) v = static_cast<__fp16>(dist(gen));
        std::vector<__fp16> C(static_cast<size_t>(s.M) * s.N);

        cactus_quant_4bit_gemm(&mat, A.data(), s.M, C.data());
        cactus_quant_4bit_matmul_mps(&mat, A.data(), s.M, C.data());

        const int N_RUNS = 5;
        Timer cpu_t;
        for (int i = 0; i < N_RUNS; ++i) {
            cactus_quant_4bit_gemm(&mat, A.data(), s.M, C.data());
        }
        double cpu_ms = cpu_t.elapsed_ms() / N_RUNS;

        Timer mps_t;
        for (int i = 0; i < N_RUNS; ++i) {
            cactus_quant_4bit_matmul_mps(&mat, A.data(), s.M, C.data());
        }
        double mps_ms = mps_t.elapsed_ms() / N_RUNS;

        std::cout << "  " << s.name
                  << "  cpu=" << std::fixed << std::setprecision(2) << cpu_ms << "ms"
                  << "  mps=" << mps_ms << "ms"
                  << "  speedup=" << std::setprecision(2) << (cpu_ms / mps_ms) << "x\n";
    }
    return true;
}

int main() {
    TestRunner runner("MPS Acceleration");
    runner.run_test("MPS available", test_mps_available());
    runner.run_test("MPS fp16 matmul correctness", test_mps_matmul_f16_correctness());
    runner.run_test("MPS fp16 matmul (model shapes)", test_mps_matmul_f16_shapes());
    runner.run_test("MPS CQ4 matmul correctness", test_mps_cq4_matmul_correctness());

    runner.print_benchmarks_header();
    bench_mps_vs_cpu_f16();
    bench_mps_vs_cpu_cq4();

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}

#else

int main() {
    return 0;
}

#endif
