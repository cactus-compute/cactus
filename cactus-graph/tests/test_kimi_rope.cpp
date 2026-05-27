#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

float yarn_find_correction_dim(float rotations, size_t dim, float base, size_t max_position_embeddings) {
    return (static_cast<float>(dim) *
            std::log(static_cast<float>(max_position_embeddings) /
                     (rotations * 2.0f * static_cast<float>(M_PI)))) /
           (2.0f * std::log(base));
}

float yarn_get_mscale(float scale, float mscale) {
    if (scale <= 1.0f) return 1.0f;
    return 0.1f * mscale * std::log(scale) + 1.0f;
}

float yarn_ramp(float idx, float low, float high) {
    if (low == high) high += 0.001f;
    float value = (idx - low) / (high - low);
    return std::clamp(value, 0.0f, 1.0f);
}

std::vector<__fp16> reference_kimi_yarn_rope(const std::vector<__fp16>& input,
                                             size_t batch,
                                             size_t seq,
                                             size_t heads,
                                             size_t dim,
                                             size_t position_offset,
                                             float theta,
                                             float scaling_factor,
                                             size_t original_max_position_embeddings,
                                             float beta_fast,
                                             float beta_slow,
                                             float mscale,
                                             float mscale_all_dim) {
    const size_t half = dim / 2;
    float low = std::floor(yarn_find_correction_dim(beta_fast, dim, theta, original_max_position_embeddings));
    float high = std::ceil(yarn_find_correction_dim(beta_slow, dim, theta, original_max_position_embeddings));
    low = std::max(0.0f, low);
    high = std::min(static_cast<float>(dim - 1), high);
    const float rotary_scale = yarn_get_mscale(scaling_factor, mscale) /
                               yarn_get_mscale(scaling_factor, mscale_all_dim);

    std::vector<float> inv_freq(half);
    for (size_t i = 0; i < half; ++i) {
        float base_freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(dim));
        float inv_freq_mask = 1.0f - yarn_ramp(static_cast<float>(i), low, high);
        inv_freq[i] = (base_freq / scaling_factor) * (1.0f - inv_freq_mask) + base_freq * inv_freq_mask;
    }

    std::vector<__fp16> output(input.size());
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq; ++t) {
            float pos = static_cast<float>(position_offset + t);
            for (size_t h = 0; h < heads; ++h) {
                size_t offset = ((b * seq + t) * heads + h) * dim;
                for (size_t i = 0; i < half; ++i) {
                    float angle = pos * inv_freq[i];
                    float c = std::cos(angle) * rotary_scale;
                    float s = std::sin(angle) * rotary_scale;
                    float even = static_cast<float>(input[offset + 2 * i]);
                    float odd = static_cast<float>(input[offset + 2 * i + 1]);
                    output[offset + i] = static_cast<__fp16>(even * c - odd * s);
                    output[offset + half + i] = static_cast<__fp16>(odd * c + even * s);
                }
            }
        }
    }
    return output;
}

bool test_kimi_yarn_rope_matches_reference() {
    const size_t batch = 1;
    const size_t seq = 3;
    const size_t heads = 2;
    const size_t dim = 8;
    const size_t count = batch * seq * heads * dim;

    std::vector<__fp16> input(count);
    for (size_t i = 0; i < count; ++i) {
        input[i] = static_cast<__fp16>((static_cast<int>(i % 17) - 8) * 0.125f);
    }

    CactusGraph graph;
    size_t input_node = graph.input({batch, seq, heads, dim}, Precision::FP16);
    size_t output_node = graph.kimi_yarn_rope(input_node,
                                              10000.0f,
                                              5,
                                              2.5f,
                                              16,
                                              32.0f,
                                              1.0f,
                                              1.0f,
                                              1.0f);
    graph.set_input(input_node, input.data(), Precision::FP16);
    graph.execute();

    auto expected = reference_kimi_yarn_rope(input, batch, seq, heads, dim,
                                            5, 10000.0f, 2.5f, 16, 32.0f, 1.0f, 1.0f, 1.0f);
    const auto* actual = static_cast<const __fp16*>(graph.get_output(output_node));
    return TestUtils::compare_arrays(actual, expected.data(), count, 1e-3f);
}

bool test_topk_indices_are_fp32_for_moe() {
    CactusGraph graph;
    size_t input = graph.input({2, 4}, Precision::FP16);
    size_t topk = graph.topk(input, 2);
    size_t topk_indices = graph.index(topk, 0, 0);
    return graph.get_output_buffer(topk).precision == Precision::FP32 &&
           graph.get_output_buffer(topk_indices).precision == Precision::FP32;
}

} // namespace

int main() {
    TestUtils::TestRunner runner("Kimi RoPE Tests");
    runner.run_test("YaRN RoPE", test_kimi_yarn_rope_matches_reference());
    runner.run_test("MoE TopK Precision", test_topk_indices_are_fp32_for_moe());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
