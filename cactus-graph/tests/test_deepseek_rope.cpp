#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace TestUtils;

namespace {

constexpr float PI_F = 3.14159265358979323846f;

float correction_dim(float rotations, size_t dim, float base, size_t max_seq_len) {
    return static_cast<float>(dim) *
           std::log(static_cast<float>(max_seq_len) / (rotations * 2.0f * PI_F)) /
           (2.0f * std::log(base));
}

float ramp(float low, float high, size_t idx) {
    if (low == high) high += 0.001f;
    return std::clamp((static_cast<float>(idx) - low) / (high - low), 0.0f, 1.0f);
}

std::vector<float> reference_rope(const std::vector<float>& input,
                                  size_t batch,
                                  size_t seq,
                                  size_t heads,
                                  size_t head_dim,
                                  size_t rope_dim,
                                  float theta,
                                  size_t position_offset,
                                  bool use_yarn,
                                  float factor,
                                  size_t original_max,
                                  float beta_fast,
                                  float beta_slow,
                                  bool inverse) {
    std::vector<float> out(input);
    const size_t nope_dim = head_dim - rope_dim;
    std::vector<float> freqs(rope_dim / 2);
    float low = 0.0f;
    float high = 0.0f;
    if (use_yarn && factor != 1.0f) {
        low = std::floor(correction_dim(beta_fast, rope_dim, theta, original_max));
        high = std::ceil(correction_dim(beta_slow, rope_dim, theta, original_max));
        low = std::max(0.0f, low);
        high = std::min(static_cast<float>(rope_dim - 1), high);
    }
    for (size_t pair = 0; pair < rope_dim / 2; ++pair) {
        float freq = 1.0f / std::pow(theta, static_cast<float>(2 * pair) / static_cast<float>(rope_dim));
        if (use_yarn && factor != 1.0f) {
            float smooth = 1.0f - ramp(low, high, pair);
            freq = freq / factor * (1.0f - smooth) + freq * smooth;
        }
        freqs[pair] = freq;
    }

    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq; ++t) {
            float pos = static_cast<float>(position_offset + t);
            for (size_t h = 0; h < heads; ++h) {
                size_t base = ((b * seq + t) * heads + h) * head_dim;
                for (size_t pair = 0; pair < rope_dim / 2; ++pair) {
                    size_t d0 = nope_dim + 2 * pair;
                    size_t d1 = d0 + 1;
                    float x0 = input[base + d0];
                    float x1 = input[base + d1];
                    float angle = pos * freqs[pair];
                    float c = std::cos(angle);
                    float s = std::sin(angle);
                    if (inverse) s = -s;
                    out[base + d0] = x0 * c - x1 * s;
                    out[base + d1] = x0 * s + x1 * c;
                }
            }
        }
    }
    return out;
}

bool compare_output(CactusGraph& graph, size_t node, const std::vector<float>& expected, float tol) {
    const auto* actual = static_cast<const float*>(graph.get_output(node));
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > tol) {
            std::cerr << "mismatch at " << i << ": actual=" << actual[i] << " expected=" << expected[i] << "\n";
            return false;
        }
    }
    return true;
}

} // namespace

bool test_dsv4_rope_main_matches_reference() {
    constexpr size_t B = 1, S = 4, H = 2, D = 10, RD = 4;
    std::vector<float> input(B * S * H * D);
    for (size_t i = 0; i < input.size(); ++i) input[i] = 0.01f * static_cast<float>(i + 1);

    CactusGraph graph;
    size_t input_id = graph.input({B, S, H, D}, Precision::FP32);
    size_t out_id = graph.dsv4_rope(input_id, RD, 10000.0f, 3, false, 1.0f, 65536, 32.0f, 1.0f, false);
    graph.set_input(input_id, input.data(), Precision::FP32);
    graph.execute();

    auto expected = reference_rope(input, B, S, H, D, RD, 10000.0f, 3, false, 1.0f, 65536, 32.0f, 1.0f, false);
    return compare_output(graph, out_id, expected, 1e-6f);
}

bool test_dsv4_rope_compress_yarn_matches_reference() {
    constexpr size_t B = 1, S = 6, H = 1, D = 12, RD = 8;
    std::vector<float> input(B * S * H * D);
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(0.05f * static_cast<float>(i + 1));

    CactusGraph graph;
    size_t input_id = graph.input({B, S, H, D}, Precision::FP32);
    size_t out_id = graph.dsv4_rope(input_id, RD, 160000.0f, 124, true, 16.0f, 65536, 32.0f, 1.0f, false);
    graph.set_input(input_id, input.data(), Precision::FP32);
    graph.execute();

    auto expected = reference_rope(input, B, S, H, D, RD, 160000.0f, 124, true, 16.0f, 65536, 32.0f, 1.0f, false);
    const auto* actual = static_cast<const float*>(graph.get_output(out_id));
    for (size_t i = 0; i < input.size(); ++i) {
        size_t d = i % D;
        if (d < D - RD && actual[i] != input[i]) return false;
    }
    return compare_output(graph, out_id, expected, 1e-6f);
}

bool test_dsv4_rope_inverse_roundtrip() {
    constexpr size_t B = 1, S = 3, H = 1, D = 8, RD = 6;
    std::vector<float> input(B * S * H * D);
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(0.09f * static_cast<float>(i + 1));

    CactusGraph graph;
    size_t input_id = graph.input({B, S, H, D}, Precision::FP32);
    size_t rotated = graph.dsv4_rope(input_id, RD, 160000.0f, 127, true, 16.0f, 65536, 32.0f, 1.0f, false);
    size_t restored = graph.dsv4_rope(rotated, RD, 160000.0f, 127, true, 16.0f, 65536, 32.0f, 1.0f, true);
    graph.set_input(input_id, input.data(), Precision::FP32);
    graph.execute();
    return compare_output(graph, restored, input, 2e-6f);
}

int main() {
    TestUtils::TestRunner runner("DeepSeek V4 RoPE Tests");
    runner.run_test("main interleaved", test_dsv4_rope_main_matches_reference());
    runner.run_test("compress yarn", test_dsv4_rope_compress_yarn_matches_reference());
    runner.run_test("inverse roundtrip", test_dsv4_rope_inverse_roundtrip());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
