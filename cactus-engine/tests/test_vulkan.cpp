#include "test_utils.h"
#include "vulkan_backend.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace EngineTestUtils;

bool test_elementwise() {
    const size_t n = 1000;
    auto a = rand_halfs(n), b = rand_halfs(n);
    std::vector<__fp16> got(n);
    std::vector<float> want(n);

    if (!cactus_vulkan_binary_f16(0, got.data(), a.data(), b.data(), n)) return false;
    for (size_t i = 0; i < n; ++i) want[i] = (float)a[i] + (float)b[i];
    if (!close_all(got, want, 5e-3f, "binary add")) return false;

    if (!cactus_vulkan_binary_f16(3, got.data(), a.data(), b.data(), n)) return false;
    for (size_t i = 0; i < n; ++i) want[i] = (float)a[i] * (float)b[i];
    if (!close_all(got, want, 5e-3f, "binary mul")) return false;

    if (!cactus_vulkan_scalar_f16(2, got.data(), a.data(), n, 1.7f)) return false;
    for (size_t i = 0; i < n; ++i) want[i] = (float)a[i] * 1.7f;
    if (!close_all(got, want, 5e-3f, "scalar mul")) return false;

    if (!cactus_vulkan_unary_f16(0, got.data(), a.data(), n)) return false;
    for (size_t i = 0; i < n; ++i) want[i] = gelu_ref((float)a[i]);
    if (!close_all(got, want, 5e-3f, "gelu")) return false;

    if (!cactus_vulkan_unary_f16(2, got.data(), a.data(), n)) return false;
    for (size_t i = 0; i < n; ++i) want[i] = (float)a[i] / (1.0f + std::exp(-(float)a[i]));
    if (!close_all(got, want, 5e-3f, "silu")) return false;

    return true;
}

bool test_swiglu() {
    const size_t n = 777;
    const float scale = 0.6f;
    auto gate = rand_halfs(n), up = rand_halfs(n);
    std::vector<__fp16> got(n);
    std::vector<float> want(n);
    if (!cactus_vulkan_swiglu_f16(got.data(), gate.data(), up.data(), n, scale)) return false;
    for (size_t i = 0; i < n; ++i) {
        __fp16 g1 = (__fp16)gelu_ref((float)gate[i]);
        __fp16 g2 = (__fp16)((float)g1 * scale);
        want[i] = (float)g2 * (float)up[i];
    }
    return close_all(got, want, 5e-3f, "swiglu");
}

bool test_rms_norm() {
    const size_t rows = 7, dim = 320;
    const float eps = 1e-6f;
    auto x = rand_halfs(rows * dim);
    auto w = rand_halfs(dim, 0.5f, 1.5f);
    std::vector<__fp16> got(rows * dim);
    std::vector<float> want(rows * dim);
    if (!cactus_vulkan_rms_norm_f16(got.data(), x.data(), w.data(), rows, dim, eps)) return false;
    for (size_t r = 0; r < rows; ++r) {
        double ss = 0;
        for (size_t i = 0; i < dim; ++i) { double v = (float)x[r * dim + i]; ss += v * v; }
        float inv = 1.0f / std::sqrt((float)(ss / dim) + eps);
        for (size_t i = 0; i < dim; ++i)
            want[r * dim + i] = (float)x[r * dim + i] * inv * (float)w[i];
    }
    return close_all(got, want, 2e-2f, "rms_norm");
}

static bool run_cq_gemv(bool interleaved, int iters, const char* what, uint32_t gs = 128) {
    CQ4Fixture f(interleaved, 512, 64, gs);
    auto x = rand_halfs(f.K);
    std::vector<__fp16> got(f.N);
    if (!cactus_vulkan_cq_gemv(got.data(), x.data(), &f.W, iters)) {
        std::cerr << "  [✗] " << what << " encode failed\n";
        return false;
    }
    auto want = f.oracle(x);
    float maxy = 0;
    for (float v : want) maxy = std::max(maxy, std::fabs(v));
    return close_all(got, want, std::max(5e-2f, 0.02f * maxy), what);
}

bool test_cq4_gemv() { return run_cq_gemv(false, 1, "cq4_gemv"); }
bool test_cq4_gemv_interleaved() { return run_cq_gemv(true, 1, "cq4_gemv interleaved"); }
bool test_cq4_gemv_iter_loop() { return run_cq_gemv(false, 8, "cq4_gemv iters=8"); }
bool test_cq4_gemv_gs64() { return run_cq_gemv(false, 1, "cq4_gemv gs=64", 64); }

static bool oracle_matches_cpu(bool interleaved, const char* what) {
    CQ4Fixture f(interleaved);
    auto x = rand_halfs(f.K);
    std::vector<__fp16> cpu(f.N);
    cactus_quant_matmul(&f.W, x.data(), 1, cpu.data());
    auto want = f.oracle(x);
    float maxy = 0;
    for (float v : want) maxy = std::max(maxy, std::fabs(v));
    return close_all(cpu, want, std::max(5e-2f, 0.02f * maxy), what);
}

bool test_oracle_vs_cpu() {
    return oracle_matches_cpu(false, "oracle vs cpu flat")
        && oracle_matches_cpu(true, "oracle vs cpu interleaved");
}

static bool gpu_matches_cpu(bool interleaved, const char* what) {
    CQ4Fixture f(interleaved);
    auto x = rand_halfs(f.K);
    std::vector<__fp16> cpu(f.N), gpu(f.N);
    cactus_quant_matmul(&f.W, x.data(), 1, cpu.data());
    if (!cactus_vulkan_cq_gemv(gpu.data(), x.data(), &f.W)) {
        std::cerr << "  [✗] " << what << " encode failed\n";
        return false;
    }
    std::vector<float> want(f.N);
    for (uint32_t i = 0; i < f.N; ++i) want[i] = (float)cpu[i];
    float maxy = 0;
    for (float v : want) maxy = std::max(maxy, std::fabs(v));
    return close_all(gpu, want, std::max(5e-2f, 0.02f * maxy), what);
}

bool test_cq4_gemv_vs_cpu() {
    return gpu_matches_cpu(false, "gpu vs cpu flat")
        && gpu_matches_cpu(true, "gpu vs cpu interleaved");
}

bool test_session_chain() {
    const size_t n = 512;
    CQ4Fixture f(false);
    auto a = rand_halfs(n), b = rand_halfs(n);
    auto w = rand_halfs(n, 0.5f, 1.5f);

    cactus_vulkan_session_begin();
    auto* pa = (__fp16*)cactus_vulkan_alloc_shared(n * 2);
    auto* pb = (__fp16*)cactus_vulkan_alloc_shared(n * 2);
    auto* pw = (__fp16*)cactus_vulkan_alloc_shared(n * 2);
    auto* t1 = (__fp16*)cactus_vulkan_alloc_pooled(n * 2);
    auto* t2 = (__fp16*)cactus_vulkan_alloc_pooled(n * 2);
    auto* t3 = (__fp16*)cactus_vulkan_alloc_pooled(n * 2);
    auto* t4 = (__fp16*)cactus_vulkan_alloc_pooled(n * 2);
    auto* py = (__fp16*)cactus_vulkan_alloc_shared(f.N * 2);
    if (!pa || !pb || !pw || !t1 || !t2 || !t3 || !t4 || !py) return false;
    std::memcpy(pa, a.data(), n * 2);
    std::memcpy(pb, b.data(), n * 2);
    std::memcpy(pw, w.data(), n * 2);

    bool ok = cactus_vulkan_encode_binary_f16(0, t1, pa, pb, n);
    ok = ok && cactus_vulkan_encode_scalar_f16(2, t2, t1, n, 0.5f);
    cactus_vulkan_session_flush();
    ok = ok && cactus_vulkan_encode_unary_f16(2, t3, t2, n);
    ok = ok && cactus_vulkan_encode_rms_norm_f16(t4, t3, pw, 1, n, 1e-6f);
    ok = ok && cactus_vulkan_encode_cq_gemv(py, t4, &f.W);
    cactus_vulkan_session_sync();

    std::vector<__fp16> got(f.N);
    if (ok) std::memcpy(got.data(), py, f.N * 2);
    for (void* p : {(void*)pa, (void*)pb, (void*)pw, (void*)t1, (void*)t2, (void*)t3, (void*)t4, (void*)py})
        cactus_vulkan_free_shared(p);
    cactus_vulkan_session_end();
    if (!ok) {
        std::cerr << "  [✗] session chain encode failed\n";
        return false;
    }

    std::vector<__fp16> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = (__fp16)((float)a[i] + (float)b[i]);
    for (size_t i = 0; i < n; ++i) h[i] = (__fp16)((float)h[i] * 0.5f);
    for (size_t i = 0; i < n; ++i) {
        float v = (float)h[i];
        h[i] = (__fp16)(v / (1.0f + std::exp(-v)));
    }
    double ss = 0;
    for (size_t i = 0; i < n; ++i) { double v = (float)h[i]; ss += v * v; }
    float inv = 1.0f / std::sqrt((float)(ss / n) + 1e-6f);
    std::vector<__fp16> hn(n);
    for (size_t i = 0; i < n; ++i) hn[i] = (__fp16)((float)h[i] * inv * (float)w[i]);
    auto want = f.oracle(hn);
    float maxy = 0;
    for (float v : want) maxy = std::max(maxy, std::fabs(v));
    return close_all(got, want, std::max(5e-2f, 0.03f * maxy), "session chain");
}

bool test_argmax() {
    const uint32_t n = 50000;
    auto logits = rand_halfs(n, -8.0f, 8.0f);
    std::mt19937 rng(21);
    std::uniform_int_distribution<uint32_t> pick(0, n - 1);
    uint32_t planted = pick(rng);
    logits[planted] = (__fp16)20.0f;
    uint32_t idx = 0;
    float best = 0;
    if (!cactus_vulkan_argmax_f16(logits.data(), n, &idx, &best)) return false;
    if (idx != planted || std::fabs(best - 20.0f) > 0.1f) {
        std::cerr << "  [✗] argmax got idx " << idx << " best " << best
                  << " want idx " << planted << "\n";
        return false;
    }
    return true;
}

int main() {
    TestUtils::TestRunner runner("Vulkan Tests");
    std::cout << "Device: " << cactus_vulkan_device_info() << "\n";
    runner.run_test("oracle_vs_cpu", test_oracle_vs_cpu());
    if (!cactus_vulkan_available()) {
        runner.log_skip("vulkan", "no usable Vulkan GPU on this host");
        runner.print_summary();
        return runner.all_passed() ? 0 : 1;
    }
    runner.run_test("elementwise", test_elementwise());
    runner.run_test("swiglu", test_swiglu());
    runner.run_test("rms_norm", test_rms_norm());
    runner.run_test("cq4_gemv", test_cq4_gemv());
    runner.run_test("cq4_gemv_interleaved", test_cq4_gemv_interleaved());
    runner.run_test("cq4_gemv_iter_loop", test_cq4_gemv_iter_loop());
    runner.run_test("cq4_gemv_gs64", test_cq4_gemv_gs64());
    runner.run_test("cq4_gemv_vs_cpu", test_cq4_gemv_vs_cpu());
    runner.run_test("session_chain", test_session_chain());
    runner.run_test("argmax", test_argmax());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
