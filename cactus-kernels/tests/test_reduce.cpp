#include "test_utils.h"
#include <vector>
#include <cmath>

using namespace TestUtils;

bool test_sum_all() {
    const size_t n = 256;
    std::vector<__fp16> data(n);
    fill_random_fp16(data, -1.0f, 1.0f);

    double result = cactus_sum_all_f16(data.data(), n);
    double expected = 0.0;
    for (size_t i = 0; i < n; i++) expected += static_cast<double>(static_cast<float>(data[i]));

    if (std::abs(result - expected) > 0.5) {
        std::cerr << "  sum_all: " << result << " vs " << expected << "\n";
        return false;
    }
    return true;
}

bool test_mean_all() {
    const size_t n = 256;
    std::vector<__fp16> data(n);
    fill_random_fp16(data, -1.0f, 1.0f);

    double result = cactus_mean_all_f16(data.data(), n);
    double expected = 0.0;
    for (size_t i = 0; i < n; i++) expected += static_cast<double>(static_cast<float>(data[i]));
    expected /= n;

    if (std::abs(result - expected) > 0.01) {
        std::cerr << "  mean_all: " << result << " vs " << expected << "\n";
        return false;
    }
    return true;
}

bool test_variance_all() {
    const size_t n = 256;
    std::vector<__fp16> data(n);
    fill_random_fp16(data, -1.0f, 1.0f);

    double result = cactus_variance_all_f16(data.data(), n);

    double mean = 0.0;
    for (size_t i = 0; i < n; i++) mean += static_cast<float>(data[i]);
    mean /= n;
    double var = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = static_cast<float>(data[i]) - mean;
        var += d * d;
    }
    var /= n;

    if (std::abs(result - var) > 0.02) {
        std::cerr << "  variance_all: " << result << " vs " << var << "\n";
        return false;
    }
    return true;
}

bool test_min_max_all() {
    const size_t n = 256;
    std::vector<__fp16> data(n);
    fill_random_fp16(data, -10.0f, 10.0f);

    __fp16 result_min = cactus_min_all_f16(data.data(), n);
    __fp16 result_max = cactus_max_all_f16(data.data(), n);

    __fp16 expected_min = data[0], expected_max = data[0];
    for (size_t i = 1; i < n; i++) {
        if (data[i] < expected_min) expected_min = data[i];
        if (data[i] > expected_max) expected_max = data[i];
    }

    if (result_min != expected_min || result_max != expected_max) {
        std::cerr << "  min/max: got [" << static_cast<float>(result_min) << ", "
                  << static_cast<float>(result_max) << "] expected ["
                  << static_cast<float>(expected_min) << ", "
                  << static_cast<float>(expected_max) << "]\n";
        return false;
    }
    return true;
}

bool test_sum_axis() {
    const size_t outer = 4, axis = 8, inner = 16;
    std::vector<__fp16> input(outer * axis * inner), output(outer * inner);
    fill_random_fp16(input, -1.0f, 1.0f);

    cactus_sum_axis_f16(input.data(), output.data(), outer, axis, inner);

    for (size_t o = 0; o < outer; o++) {
        for (size_t i = 0; i < inner; i++) {
            float ref = 0.0f;
            for (size_t a = 0; a < axis; a++) {
                ref += static_cast<float>(input[(o * axis + a) * inner + i]);
            }
            float actual = static_cast<float>(output[o * inner + i]);
            if (std::abs(actual - ref) > 0.1f) {
                std::cerr << "  sum_axis mismatch [" << o << "," << i << "]: "
                          << actual << " vs " << ref << "\n";
                return false;
            }
        }
    }
    return true;
}

bool test_neon_sum_axis_inner1_correctness() {
    const size_t outer_size = 2;
    const size_t axis_size = 9;
    const size_t inner_size = 1;

    std::vector<__fp16> input = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
        2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f
    };
    std::vector<__fp16> output(outer_size * inner_size);
    std::vector<__fp16> expected = {45.0f, 90.0f};

    cactus_sum_axis_f16(input.data(), output.data(), outer_size, axis_size, inner_size);
    return TestUtils::compare_arrays(output.data(), expected.data(), expected.size(), 1e-2f);
}

bool test_neon_mean_axis_inner1_correctness() {
    const size_t outer_size = 2;
    const size_t axis_size = 9;
    const size_t inner_size = 1;

    std::vector<__fp16> input = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
        2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f
    };
    std::vector<__fp16> output(outer_size * inner_size);
    std::vector<__fp16> expected = {5.0f, 10.0f};

    cactus_mean_axis_f16(input.data(), output.data(), outer_size, axis_size, inner_size);
    return TestUtils::compare_arrays(output.data(), expected.data(), expected.size(), 1e-2f);
}

bool test_neon_variance_axis_inner1_correctness() {
    const size_t outer_size = 2;
    const size_t axis_size = 9;
    const size_t inner_size = 1;

    std::vector<__fp16> input = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
        2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f
    };
    std::vector<__fp16> output(outer_size * inner_size);
    std::vector<__fp16> expected = {6.6667f, 26.6667f};

    cactus_variance_axis_f16(input.data(), output.data(), outer_size, axis_size, inner_size);
    return TestUtils::compare_arrays(output.data(), expected.data(), expected.size(), 0.05f);
}

bool test_neon_variance_axis_non_inner1_correctness() {
    const size_t outer_size = 2;
    const size_t axis_size = 9;
    const size_t inner_size = 3;

    std::vector<__fp16> input(outer_size * axis_size * inner_size);
    auto idx = [&](size_t outer, size_t axis, size_t inner) {
        return outer * axis_size * inner_size + axis * inner_size + inner;
    };

    for (size_t a = 0; a < axis_size; ++a) {
        input[idx(0, a, 0)] = static_cast<__fp16>(1.0f + static_cast<float>(a));
        input[idx(0, a, 1)] = static_cast<__fp16>(2.0f + static_cast<float>(a));
        input[idx(0, a, 2)] = static_cast<__fp16>(-4.0f + static_cast<float>(a));

        input[idx(1, a, 0)] = static_cast<__fp16>(2.0f + 2.0f * static_cast<float>(a));
        input[idx(1, a, 1)] = static_cast<__fp16>(3.0f + 2.0f * static_cast<float>(a));
        input[idx(1, a, 2)] = static_cast<__fp16>(-8.0f + 2.0f * static_cast<float>(a));
    }

    std::vector<__fp16> output(outer_size * inner_size);
    std::vector<__fp16> expected = {
        6.6667f, 6.6667f, 6.6667f,
        26.6667f, 26.6667f, 26.6667f
    };

    cactus_variance_axis_f16(input.data(), output.data(), outer_size, axis_size, inner_size);
    return TestUtils::compare_arrays(output.data(), expected.data(), expected.size(), 0.05f);
}

bool test_neon_min_axis_inner1_correctness() {
    const size_t outer_size = 2;
    const size_t axis_size = 9;
    const size_t inner_size = 1;

    std::vector<__fp16> input = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
        2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f
    };
    std::vector<__fp16> output(outer_size * inner_size);
    std::vector<__fp16> expected = {1.0f, 2.0f};

    cactus_min_axis_f16(input.data(), output.data(), outer_size, axis_size, inner_size);
    return TestUtils::compare_arrays(output.data(), expected.data(), expected.size(), 1e-2f);
}

bool test_neon_max_axis_inner1_correctness() {
    const size_t outer_size = 2;
    const size_t axis_size = 9;
    const size_t inner_size = 1;

    std::vector<__fp16> input = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
        2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f
    };
    std::vector<__fp16> output(outer_size * inner_size);
    std::vector<__fp16> expected = {9.0f, 18.0f};

    cactus_max_axis_f16(input.data(), output.data(), outer_size, axis_size, inner_size);
    return TestUtils::compare_arrays(output.data(), expected.data(), expected.size(), 1e-2f);
}

bool run_benchmarks(TestRunner& runner) {
    (void)runner;
    auto benchmark = [&](const std::string& label, size_t n, auto fn) {
        fn();
        Timer t;
        for (int i = 0; i < 100; i++) fn();
        double ms = t.elapsed_ms() / 100.0;
        double gb_s = (static_cast<double>(n) * sizeof(__fp16)) / (ms * 1e6);
        std::cout << "  ⚡ " << std::left << std::setw(28) << label
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gb_s << " GB/s\n";
    };

    const size_t n = 1024 * 1024;
    std::vector<__fp16> data(n);
    fill_random_fp16(data, -1.0f, 1.0f);

    volatile double sink_d = 0.0;
    volatile __fp16 sink_h = static_cast<__fp16>(0.0f);

    benchmark("sum_all 1M", n, [&]{ sink_d = cactus_sum_all_f16(data.data(), n); });
    benchmark("mean_all 1M", n, [&]{ sink_d = cactus_mean_all_f16(data.data(), n); });
    benchmark("variance_all 1M", n, [&]{ sink_d = cactus_variance_all_f16(data.data(), n); });
    benchmark("min_all 1M", n, [&]{ sink_h = cactus_min_all_f16(data.data(), n); });
    benchmark("max_all 1M", n, [&]{ sink_h = cactus_max_all_f16(data.data(), n); });

    const size_t outer = 1024;
    const size_t axis = 1024;
    const size_t inner1 = 1;
    std::vector<__fp16> axis_input(outer * axis * inner1);
    std::vector<__fp16> axis_output(outer * inner1);
    fill_random_fp16(axis_input, -1.0f, 1.0f);

    benchmark("sum_axis 1024x1024", axis_input.size(), [&]{
        cactus_sum_axis_f16(axis_input.data(), axis_output.data(), outer, axis, inner1);
    });
    benchmark("mean_axis 1024x1024", axis_input.size(), [&]{
        cactus_mean_axis_f16(axis_input.data(), axis_output.data(), outer, axis, inner1);
    });
    benchmark("variance_axis 1024x1024", axis_input.size(), [&]{
        cactus_variance_axis_f16(axis_input.data(), axis_output.data(), outer, axis, inner1);
    });
    benchmark("min_axis 1024x1024", axis_input.size(), [&]{
        cactus_min_axis_f16(axis_input.data(), axis_output.data(), outer, axis, inner1);
    });
    benchmark("max_axis 1024x1024", axis_input.size(), [&]{
        cactus_max_axis_f16(axis_input.data(), axis_output.data(), outer, axis, inner1);
    });

    const size_t channels = 4;
    std::vector<__fp16> variance_input_non_inner1(outer * axis * channels);
    std::vector<__fp16> variance_output_non_inner1(outer * channels);
    fill_random_fp16(variance_input_non_inner1, -1.0f, 1.0f);

    benchmark("variance_axis 1024x1024x4", variance_input_non_inner1.size(), [&]{
        cactus_variance_axis_f16(
            variance_input_non_inner1.data(),
            variance_output_non_inner1.data(),
            outer,
            axis,
            channels
        );
    });

    (void)sink_d; (void)sink_h;
    return true;
}

int main() {
    TestRunner runner("Reduction Operations");
    runner.run_test("sum_all", test_sum_all());
    runner.run_test("mean_all", test_mean_all());
    runner.run_test("variance_all", test_variance_all());
    runner.run_test("min_max_all", test_min_max_all());
    runner.run_test("sum_axis", test_sum_axis());
    runner.run_test("Kernel Sum Axis Inner1 FP16 Correctness", test_neon_sum_axis_inner1_correctness());
    runner.run_test("Kernel Mean Axis Inner1 FP16 Correctness", test_neon_mean_axis_inner1_correctness());
    runner.run_test("Kernel Variance Axis Inner1 FP16 Correctness", test_neon_variance_axis_inner1_correctness());
    runner.run_test("Kernel Variance Axis Non-Inner1 FP16 Correctness", test_neon_variance_axis_non_inner1_correctness());
    runner.run_test("Kernel Min Axis Inner1 FP16 Correctness", test_neon_min_axis_inner1_correctness());
    runner.run_test("Kernel Max Axis Inner1 FP16 Correctness", test_neon_max_axis_inner1_correctness());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks(runner));
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
