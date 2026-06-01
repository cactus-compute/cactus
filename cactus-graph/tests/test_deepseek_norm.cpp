#include "test_utils.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace TestUtils;

namespace {

std::vector<float> reference_rms(const std::vector<float>& input, size_t rows, size_t cols, float eps) {
    std::vector<float> out(input.size());
    for (size_t row = 0; row < rows; ++row) {
        double mean_square = 0.0;
        for (size_t col = 0; col < cols; ++col) {
            float value = input[row * cols + col];
            mean_square += static_cast<double>(value) * value;
        }
        float rsqrt = 1.0f / std::sqrt(static_cast<float>(mean_square / cols) + eps);
        for (size_t col = 0; col < cols; ++col) out[row * cols + col] = input[row * cols + col] * rsqrt;
    }
    return out;
}

bool compare_output(const float* actual, const std::vector<float>& expected, float tol) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > tol) {
            std::cerr << "mismatch at " << i << ": actual=" << actual[i]
                      << " expected=" << expected[i] << "\n";
            return false;
        }
    }
    return true;
}

} // namespace

bool test_dsv4_rms_norm_last_dim() {
    constexpr size_t B = 2;
    constexpr size_t S = 3;
    constexpr size_t H = 2;
    constexpr size_t D = 5;
    constexpr float eps = 1e-6f;
    std::vector<float> input(B * S * H * D);
    for (size_t i = 0; i < input.size(); ++i) input[i] = -0.7f + 0.03f * static_cast<float>(i);

    CactusGraph graph;
    size_t input_id = graph.input({B, S, H, D}, Precision::FP32);
    size_t out_id = graph.dsv4_rms_norm(input_id, eps);
    graph.set_input(input_id, input.data(), Precision::FP32);
    graph.execute();

    auto expected = reference_rms(input, B * S * H, D, eps);
    return compare_output(static_cast<const float*>(graph.get_output(out_id)), expected, 1e-6f);
}

int main() {
    TestUtils::TestRunner runner("DeepSeek V4 Norm Tests");
    runner.run_test("unweighted RMS", test_dsv4_rms_norm_last_dim());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
