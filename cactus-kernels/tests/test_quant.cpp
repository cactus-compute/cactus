#include "test_utils.h"
#include <vector>
#include <cmath>

using namespace TestUtils;

bool test_fp16_fp32_roundtrip() {
    const size_t n = 256;
    std::vector<__fp16> src(n);
    std::vector<float> fp32(n);
    std::vector<__fp16> dst(n);
    fill_random_fp16(src, -10.0f, 10.0f);

    cactus_fp16_to_fp32(src.data(), fp32.data(), n);
    cactus_fp32_to_fp16(fp32.data(), dst.data(), n);

    return compare_arrays(src.data(), dst.data(), n, 1e-3f);
}

bool test_int8_fp32_roundtrip() {
    const size_t n = 256;
    std::vector<float> src(n), dst(n);
    std::vector<int8_t> quantized(n);
    for (size_t i = 0; i < n; i++) src[i] = static_cast<float>(i % 127) - 63.0f;

    float scale = 1.0f;
    cactus_fp32_to_int8(src.data(), quantized.data(), n, scale);
    cactus_int8_to_fp32(quantized.data(), dst.data(), n, scale);

    for (size_t i = 0; i < n; i++) {
        float expected = std::max(-127.0f, std::min(127.0f, std::round(src[i])));
        if (std::abs(dst[i] - expected) > 1.0f) {
            std::cerr << "  int8 roundtrip mismatch at " << i << ": " << dst[i] << " vs " << expected << "\n";
            return false;
        }
    }
    return true;
}

bool test_int8_fp16_conversion() {
    const size_t n = 256;
    std::vector<int8_t> src(n);
    std::vector<__fp16> dst(n);
    for (size_t i = 0; i < n; i++) src[i] = static_cast<int8_t>(i % 127 - 63);

    cactus_int8_to_fp16(src.data(), dst.data(), n, 0.1f);

    for (size_t i = 0; i < n; i++) {
        float expected = static_cast<float>(src[i]) * 0.1f;
        float actual = static_cast<float>(dst[i]);
        if (std::abs(actual - expected) > 0.05f) {
            std::cerr << "  int8_to_fp16 mismatch at " << i << ": " << actual << " vs " << expected << "\n";
            return false;
        }
    }
    return true;
}

bool test_fp16_max_abs() {
    const size_t n = 256;
    std::vector<__fp16> data(n);
    fill_random_fp16(data, -5.0f, 5.0f);
    data[42] = static_cast<__fp16>(-99.0f);

    float result = cactus_fp16_max_abs(data.data(), n);
    if (std::abs(result - 99.0f) > 0.5f) {
        std::cerr << "  fp16_max_abs: " << result << " (expected ~99.0)\n";
        return false;
    }
    return true;
}

bool test_kv_quantize_int8() {
    const size_t seq = 4, kv_heads = 2, head_dim = 64;
    const size_t group_size = 32;
    const size_t num_groups = head_dim / group_size;
    std::vector<__fp16> src(seq * kv_heads * head_dim);
    std::vector<int8_t> dst(seq * kv_heads * head_dim);
    std::vector<float> scales(seq * kv_heads * num_groups);
    fill_random_fp16(src, -2.0f, 2.0f);

    cactus_quantize_kv_fp16_to_int8(src.data(), dst.data(), scales.data(),
                                     seq, kv_heads, head_dim, group_size);

    for (size_t i = 0; i < scales.size(); i++) {
        if (scales[i] <= 0.0f || !std::isfinite(scales[i])) {
            std::cerr << "  kv_quantize: invalid scale at " << i << ": " << scales[i] << "\n";
            return false;
        }
    }

    for (size_t s = 0; s < seq; s++) {
        for (size_t h = 0; h < kv_heads; h++) {
            for (size_t g = 0; g < num_groups; g++) {
                float scale = scales[(s * kv_heads + h) * num_groups + g];
                for (size_t k = 0; k < group_size; k++) {
                    size_t idx = (s * kv_heads + h) * head_dim + g * group_size + k;
                    float original = static_cast<float>(src[idx]);
                    float dequant = static_cast<float>(dst[idx]) * scale;
                    if (std::abs(original - dequant) > 0.1f) {
                        std::cerr << "  kv_quantize: dequant error at " << idx << ": "
                                  << dequant << " vs " << original << "\n";
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool run_benchmarks() {
    const size_t n = 1024 * 1024;

    {
        std::vector<__fp16> src(n);
        std::vector<float> dst(n);
        fill_random_fp16(src);
        cactus_fp16_to_fp32(src.data(), dst.data(), n);
        Timer t;
        for (int i = 0; i < 100; i++) cactus_fp16_to_fp32(src.data(), dst.data(), n);
        double ms = t.elapsed_ms() / 100.0;
        double gb_s = (static_cast<double>(n) * (sizeof(__fp16) + sizeof(float))) / (ms * 1e6);
        std::cout << "  ⚡ " << std::left << std::setw(28) << "fp16_to_fp32 1M"
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gb_s << " GB/s\n";
    }
    {
        std::vector<float> src(n);
        std::vector<__fp16> dst(n);
        for (size_t i = 0; i < n; i++) src[i] = static_cast<float>(i % 1000) * 0.001f;
        cactus_fp32_to_fp16(src.data(), dst.data(), n);
        Timer t;
        for (int i = 0; i < 100; i++) cactus_fp32_to_fp16(src.data(), dst.data(), n);
        double ms = t.elapsed_ms() / 100.0;
        double gb_s = (static_cast<double>(n) * (sizeof(float) + sizeof(__fp16))) / (ms * 1e6);
        std::cout << "  ⚡ " << std::left << std::setw(28) << "fp32_to_fp16 1M"
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gb_s << " GB/s\n";
    }
    {
        std::vector<int8_t> src(n);
        std::vector<__fp16> dst(n);
        for (size_t i = 0; i < n; i++) src[i] = static_cast<int8_t>(i % 127 - 63);
        cactus_int8_to_fp16(src.data(), dst.data(), n, 0.01f);
        Timer t;
        for (int i = 0; i < 100; i++) cactus_int8_to_fp16(src.data(), dst.data(), n, 0.01f);
        double ms = t.elapsed_ms() / 100.0;
        double gb_s = (static_cast<double>(n) * (sizeof(int8_t) + sizeof(__fp16))) / (ms * 1e6);
        std::cout << "  ⚡ " << std::left << std::setw(28) << "int8_to_fp16 1M"
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gb_s << " GB/s\n";
    }
    {
        std::vector<__fp16> src(n);
        std::vector<int8_t> dst(n);
        fill_random_fp16(src, -1.0f, 1.0f);
        cactus_fp16_to_int8(src.data(), dst.data(), n, 0.01f);
        Timer t;
        for (int i = 0; i < 100; i++) cactus_fp16_to_int8(src.data(), dst.data(), n, 0.01f);
        double ms = t.elapsed_ms() / 100.0;
        double gb_s = (static_cast<double>(n) * (sizeof(__fp16) + sizeof(int8_t))) / (ms * 1e6);
        std::cout << "  ⚡ " << std::left << std::setw(28) << "fp16_to_int8 1M"
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gb_s << " GB/s\n";
    }
    {
        const size_t seq = 4, kv_heads = 8, head_dim = 64, group_size = 32;
        const size_t num_groups = head_dim / group_size;
        std::vector<__fp16> src(seq * kv_heads * head_dim);
        std::vector<int8_t> dst(seq * kv_heads * head_dim);
        std::vector<float> scales(seq * kv_heads * num_groups);
        fill_random_fp16(src, -2.0f, 2.0f);
        cactus_quantize_kv_fp16_to_int8(src.data(), dst.data(), scales.data(), seq, kv_heads, head_dim, group_size);
        Timer t;
        for (int i = 0; i < 100; i++)
            cactus_quantize_kv_fp16_to_int8(src.data(), dst.data(), scales.data(), seq, kv_heads, head_dim, group_size);
        double ms = t.elapsed_ms() / 100.0;
        size_t nbytes = seq * kv_heads * head_dim;
        double gb_s = (static_cast<double>(nbytes) * (sizeof(__fp16) + sizeof(int8_t))) / (ms * 1e6);
        std::cout << "  ⚡ " << std::left << std::setw(28) << "kv_quantize 4x8x64"
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gb_s << " GB/s\n";
    }
    return true;
}

int main() {
    TestRunner runner("Quantization & Conversion");
    runner.run_test("fp16_fp32_roundtrip", test_fp16_fp32_roundtrip());
    runner.run_test("int8_fp32_roundtrip", test_int8_fp32_roundtrip());
    runner.run_test("int8_fp16_conversion", test_int8_fp16_conversion());
    runner.run_test("fp16_max_abs", test_fp16_max_abs());
    runner.run_test("kv_quantize_int8", test_kv_quantize_int8());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
