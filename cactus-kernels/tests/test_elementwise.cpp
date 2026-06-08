#include "test_utils.h"
#include <vector>
#include <cmath>

using namespace TestUtils;

bool test_add_f16() {
    const size_t n = 1024;
    std::vector<__fp16> a(n), b(n), out(n), expected(n);
    fill_random_fp16(a); fill_random_fp16(b);
    for (size_t i = 0; i < n; i++) expected[i] = a[i] + b[i];
    cactus_add_f16(a.data(), b.data(), out.data(), n);
    return compare_arrays(out.data(), expected.data(), n);
}

bool test_subtract_f16() {
    const size_t n = 1024;
    std::vector<__fp16> a(n), b(n), out(n), expected(n);
    fill_random_fp16(a); fill_random_fp16(b);
    for (size_t i = 0; i < n; i++) expected[i] = a[i] - b[i];
    cactus_subtract_f16(a.data(), b.data(), out.data(), n);
    return compare_arrays(out.data(), expected.data(), n);
}

bool test_multiply_f16() {
    const size_t n = 1024;
    std::vector<__fp16> a(n), b(n), out(n), expected(n);
    fill_random_fp16(a); fill_random_fp16(b);
    for (size_t i = 0; i < n; i++) expected[i] = a[i] * b[i];
    cactus_multiply_f16(a.data(), b.data(), out.data(), n);
    return compare_arrays(out.data(), expected.data(), n);
}

bool test_divide_f16() {
    const size_t n = 1024;
    std::vector<__fp16> a(n), b(n), out(n), expected(n);
    fill_random_fp16(a);
    fill_random_fp16(b, 0.5f, 2.0f);
    for (size_t i = 0; i < n; i++) expected[i] = a[i] / b[i];
    cactus_divide_f16(a.data(), b.data(), out.data(), n);
    return compare_arrays(out.data(), expected.data(), n, 5e-2f);
}

bool test_add_clipped_f16() {
    const size_t n = 256;
    std::vector<__fp16> a(n), b(n), out(n);
    fill_random_fp16(a, -100.0f, 100.0f);
    fill_random_fp16(b, -100.0f, 100.0f);
    cactus_add_f16_clipped(a.data(), b.data(), out.data(), n);
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(static_cast<float>(out[i]))) return false;
    }
    return true;
}

bool test_scalar_ops_f16() {
    const size_t n = 256;
    std::vector<__fp16> in(n), out(n);

    fill_random_fp16(in, 0.1f, 2.0f);
    cactus_scalar_op_f16(in.data(), out.data(), n, 3.0f, ScalarOpType::ADD);
    for (size_t i = 0; i < n; i++) {
        float diff = std::abs(static_cast<float>(out[i]) - (static_cast<float>(in[i]) + 3.0f));
        if (diff > 0.05f) return false;
    }

    fill_random_fp16(in, 0.1f, 2.0f);
    cactus_scalar_op_f16(in.data(), out.data(), n, 2.5f, ScalarOpType::MULTIPLY);
    for (size_t i = 0; i < n; i++) {
        float diff = std::abs(static_cast<float>(out[i]) - (static_cast<float>(in[i]) * 2.5f));
        if (diff > 0.05f) return false;
    }

    fill_random_fp16(in, -2.0f, 2.0f);
    cactus_scalar_op_f16(in.data(), out.data(), n, 0.0f, ScalarOpType::EXP);
    for (size_t i = 0; i < n; i++) {
        float actual = static_cast<float>(out[i]);
        float ref = std::exp(static_cast<float>(in[i]));
        float rel_err = std::abs(actual - ref) / std::max(std::abs(ref), 1e-6f);
        if (rel_err > 0.01f) return false;
    }
    return true;
}

bool test_transpose_2d_f16() {
    const size_t rows = 64, cols = 128;
    std::vector<__fp16> src(rows * cols), dst(rows * cols);
    fill_random_fp16(src);
    cactus_transpose_2d_f16(src.data(), dst.data(), rows, cols, 0, rows);
    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < cols; c++)
            if (std::abs(static_cast<float>(src[r * cols + c]) - static_cast<float>(dst[c * rows + r])) > 1e-4f)
                return false;
    return true;
}

bool test_fast_tanh_f32x4() {
    constexpr float TOL = 1e-5f;
    for (int i = -10000; i <= 10000; ++i) {
        float x = 0.001f * static_cast<float>(i);
        float32x4_t r = fast_tanh_f32x4(vdupq_n_f32(x));
        float lanes[4];
        vst1q_f32(lanes, r);
        float want = std::tanh(x);
        for (int k = 0; k < 4; ++k) {
            if (std::fabs(lanes[k] - want) > TOL) return false;
        }
    }
    return true;
}

bool run_benchmarks() {
    auto bench_binary = [](const char* label, void(*fn)(const __fp16*, const __fp16*, __fp16*, size_t)) {
        const size_t n = 1024 * 1024;
        std::vector<__fp16> a(n), b(n), out(n);
        fill_random_fp16(a); fill_random_fp16(b);
        fn(a.data(), b.data(), out.data(), n);
        Timer t;
        for (int i = 0; i < 100; i++) fn(a.data(), b.data(), out.data(), n);
        double ms = t.elapsed_ms() / 100.0;
        double gb_s = (3.0 * n * sizeof(__fp16)) / (ms * 1e6);
        std::cout << "  ⚡ " << std::left << std::setw(28) << label
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gb_s << " GB/s\n";
    };

    bench_binary("add 1M", cactus_add_f16);
    bench_binary("subtract 1M", cactus_subtract_f16);
    bench_binary("multiply 1M", cactus_multiply_f16);
    bench_binary("divide 1M", cactus_divide_f16);

    {
        const size_t n = 1024 * 1024;
        std::vector<__fp16> in(n), out(n);
        fill_random_fp16(in, 0.1f, 2.0f);
        cactus_scalar_op_f16(in.data(), out.data(), n, 2.0f, ScalarOpType::MULTIPLY);
        Timer t;
        for (int i = 0; i < 100; i++) cactus_scalar_op_f16(in.data(), out.data(), n, 2.0f, ScalarOpType::MULTIPLY);
        double ms = t.elapsed_ms() / 100.0;
        double gb_s = (2.0 * n * sizeof(__fp16)) / (ms * 1e6);
        std::cout << "  ⚡ " << std::left << std::setw(28) << "scalar_multiply 1M"
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gb_s << " GB/s\n";
    }

    {
        const size_t n = 1024 * 1024;
        std::vector<__fp16> in(n), out(n);
        fill_random_fp16(in, -2.0f, 2.0f);
        cactus_scalar_op_f16(in.data(), out.data(), n, 0.0f, ScalarOpType::EXP);
        Timer t;
        for (int i = 0; i < 100; i++) cactus_scalar_op_f16(in.data(), out.data(), n, 0.0f, ScalarOpType::EXP);
        double ms = t.elapsed_ms() / 100.0;
        double gb_s = (2.0 * n * sizeof(__fp16)) / (ms * 1e6);
        std::cout << "  ⚡ " << std::left << std::setw(28) << "scalar_exp 1M"
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gb_s << " GB/s\n";
    }

    {
        const size_t rows = 1024, cols = 1024;
        std::vector<__fp16> src(rows * cols), dst(rows * cols);
        fill_random_fp16(src);
        cactus_transpose_2d_f16(src.data(), dst.data(), rows, cols, 0, rows);
        Timer t;
        for (int i = 0; i < 100; i++) cactus_transpose_2d_f16(src.data(), dst.data(), rows, cols, 0, rows);
        double ms = t.elapsed_ms() / 100.0;
        double gb_s = (2.0 * rows * cols * sizeof(__fp16)) / (ms * 1e6);
        std::cout << "  ⚡ " << std::left << std::setw(28) << "transpose 1024x1024"
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gb_s << " GB/s\n";
    }

    return true;
}

int main() {
    TestRunner runner("Elementwise & Scalar Ops");
    runner.run_test("add_f16", test_add_f16());
    runner.run_test("subtract_f16", test_subtract_f16());
    runner.run_test("multiply_f16", test_multiply_f16());
    runner.run_test("divide_f16", test_divide_f16());
    runner.run_test("add_clipped_f16", test_add_clipped_f16());
    runner.run_test("scalar_ops_f16", test_scalar_ops_f16());
    runner.run_test("transpose_2d_f16", test_transpose_2d_f16());
    runner.run_test("fast_tanh_f32x4", test_fast_tanh_f32x4());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
