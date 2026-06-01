#include "test_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using namespace TestUtils;

namespace {

constexpr size_t HC = 4;
constexpr size_t MIX = 24;

float sigmoid_ref(float x) {
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    float z = std::exp(x);
    return z / (1.0f + z);
}

void sinkhorn_ref(float comb[HC][HC], size_t iters, float eps) {
    for (size_t r = 0; r < HC; ++r) {
        float max_v = comb[r][0];
        for (size_t c = 1; c < HC; ++c) max_v = std::max(max_v, comb[r][c]);
        float sum = 0.0f;
        for (size_t c = 0; c < HC; ++c) {
            comb[r][c] = std::exp(comb[r][c] - max_v);
            sum += comb[r][c];
        }
        for (size_t c = 0; c < HC; ++c) comb[r][c] = comb[r][c] / sum + eps;
    }
    float col_sum[HC] = {};
    for (size_t c = 0; c < HC; ++c) {
        for (size_t r = 0; r < HC; ++r) col_sum[c] += comb[r][c];
    }
    for (size_t r = 0; r < HC; ++r) {
        for (size_t c = 0; c < HC; ++c) comb[r][c] /= (col_sum[c] + eps);
    }
    for (size_t iter = 1; iter < iters; ++iter) {
        for (size_t r = 0; r < HC; ++r) {
            float row_sum = 0.0f;
            for (size_t c = 0; c < HC; ++c) row_sum += comb[r][c];
            for (size_t c = 0; c < HC; ++c) comb[r][c] /= (row_sum + eps);
        }
        std::fill(std::begin(col_sum), std::end(col_sum), 0.0f);
        for (size_t c = 0; c < HC; ++c) {
            for (size_t r = 0; r < HC; ++r) col_sum[c] += comb[r][c];
        }
        for (size_t r = 0; r < HC; ++r) {
            for (size_t c = 0; c < HC; ++c) comb[r][c] /= (col_sum[c] + eps);
        }
    }
}

std::vector<float> reference_mix(const std::vector<float>& streams,
                                 const std::vector<float>& fn,
                                 const std::vector<float>& base,
                                 const std::vector<float>& scale,
                                 size_t tokens,
                                 size_t hidden,
                                 float eps,
                                 size_t iters) {
    const size_t flat = HC * hidden;
    std::vector<float> out(tokens * MIX);
    std::vector<float> normed(flat);
    for (size_t t = 0; t < tokens; ++t) {
        double mean_square = 0.0;
        for (size_t i = 0; i < flat; ++i) {
            float v = streams[t * flat + i];
            mean_square += static_cast<double>(v) * v;
        }
        float rsqrt = 1.0f / std::sqrt(static_cast<float>(mean_square / flat) + eps);
        for (size_t i = 0; i < flat; ++i) normed[i] = streams[t * flat + i] * rsqrt;

        float mixes[MIX] = {};
        for (size_t m = 0; m < MIX; ++m) {
            for (size_t i = 0; i < flat; ++i) mixes[m] += fn[m * flat + i] * normed[i];
        }
        for (size_t i = 0; i < HC; ++i) {
            out[t * MIX + i] = sigmoid_ref(mixes[i] * scale[0] + base[i]) + eps;
            out[t * MIX + HC + i] = 2.0f * sigmoid_ref(mixes[HC + i] * scale[1] + base[HC + i]);
        }
        float comb[HC][HC] = {};
        for (size_t r = 0; r < HC; ++r) {
            for (size_t c = 0; c < HC; ++c) {
                size_t idx = 2 * HC + r * HC + c;
                comb[r][c] = mixes[idx] * scale[2] + base[idx];
            }
        }
        sinkhorn_ref(comb, iters, eps);
        for (size_t r = 0; r < HC; ++r) {
            for (size_t c = 0; c < HC; ++c) {
                out[t * MIX + 2 * HC + r * HC + c] = comb[r][c];
            }
        }
    }
    return out;
}

std::vector<float> reference_collapse(const std::vector<float>& streams,
                                      const std::vector<float>& mix,
                                      size_t tokens,
                                      size_t hidden) {
    std::vector<float> out(tokens * hidden);
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t d = 0; d < hidden; ++d) {
            float acc = 0.0f;
            for (size_t h = 0; h < HC; ++h) {
                acc += mix[t * MIX + h] * streams[(t * HC + h) * hidden + d];
            }
            out[t * hidden + d] = acc;
        }
    }
    return out;
}

std::vector<float> reference_post(const std::vector<float>& sublayer,
                                  const std::vector<float>& residual,
                                  const std::vector<float>& mix,
                                  size_t tokens,
                                  size_t hidden) {
    std::vector<float> out(tokens * HC * hidden);
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t k = 0; k < HC; ++k) {
            float post = mix[t * MIX + HC + k];
            for (size_t d = 0; d < hidden; ++d) {
                float acc = post * sublayer[t * hidden + d];
                for (size_t j = 0; j < HC; ++j) {
                    acc += mix[t * MIX + 2 * HC + j * HC + k] * residual[(t * HC + j) * hidden + d];
                }
                out[(t * HC + k) * hidden + d] = acc;
            }
        }
    }
    return out;
}

std::vector<float> reference_head(const std::vector<float>& streams,
                                  const std::vector<float>& fn,
                                  const std::vector<float>& base,
                                  const std::vector<float>& scale,
                                  size_t tokens,
                                  size_t hidden,
                                  float eps) {
    const size_t flat = HC * hidden;
    std::vector<float> out(tokens * hidden);
    std::vector<float> normed(flat);
    for (size_t t = 0; t < tokens; ++t) {
        double mean_square = 0.0;
        for (size_t i = 0; i < flat; ++i) {
            float v = streams[t * flat + i];
            mean_square += static_cast<double>(v) * v;
        }
        float rsqrt = 1.0f / std::sqrt(static_cast<float>(mean_square / flat) + eps);
        for (size_t i = 0; i < flat; ++i) normed[i] = streams[t * flat + i] * rsqrt;

        float pre[HC] = {};
        for (size_t h = 0; h < HC; ++h) {
            float acc = 0.0f;
            for (size_t i = 0; i < flat; ++i) acc += fn[h * flat + i] * normed[i];
            pre[h] = sigmoid_ref(acc * scale[0] + base[h]) + eps;
        }
        for (size_t d = 0; d < hidden; ++d) {
            float acc = 0.0f;
            for (size_t h = 0; h < HC; ++h) acc += pre[h] * streams[(t * HC + h) * hidden + d];
            out[t * hidden + d] = acc;
        }
    }
    return out;
}

bool compare_float_output(CactusGraph& graph, size_t node, const std::vector<float>& expected, float tol) {
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

bool test_hc_mix_matches_reference() {
    constexpr size_t T = 3;
    constexpr size_t D = 5;
    constexpr float eps = 1e-6f;
    constexpr size_t iters = 20;
    std::vector<float> streams(T * HC * D);
    std::vector<float> fn(MIX * HC * D);
    std::vector<float> base(MIX);
    std::vector<float> scale = {0.75f, -0.5f, 0.25f};

    for (size_t i = 0; i < streams.size(); ++i) streams[i] = std::sin(0.17f * static_cast<float>(i + 1));
    for (size_t i = 0; i < fn.size(); ++i) fn[i] = 0.05f * std::cos(0.11f * static_cast<float>(i + 3));
    for (size_t i = 0; i < base.size(); ++i) base[i] = 0.03f * std::sin(0.13f * static_cast<float>(i + 5));

    CactusGraph graph;
    size_t streams_id = graph.input({T, HC, D}, Precision::FP32);
    size_t fn_id = graph.input({MIX, HC * D}, Precision::FP32);
    size_t base_id = graph.input({MIX}, Precision::FP32);
    size_t scale_id = graph.input({3}, Precision::FP32);
    size_t mix_id = graph.dsv4_hc_mix(streams_id, fn_id, base_id, scale_id, eps, iters);

    graph.set_input(streams_id, streams.data(), Precision::FP32);
    graph.set_input(fn_id, fn.data(), Precision::FP32);
    graph.set_input(base_id, base.data(), Precision::FP32);
    graph.set_input(scale_id, scale.data(), Precision::FP32);
    graph.execute();

    auto expected = reference_mix(streams, fn, base, scale, T, D, eps, iters);
    if (!compare_float_output(graph, mix_id, expected, 1e-5f)) return false;

    const auto* mix = static_cast<const float*>(graph.get_output(mix_id));
    for (size_t t = 0; t < T; ++t) {
        for (size_t r = 0; r < HC; ++r) {
            float row_sum = 0.0f;
            float col_sum = 0.0f;
            for (size_t c = 0; c < HC; ++c) {
                row_sum += mix[t * MIX + 2 * HC + r * HC + c];
                col_sum += mix[t * MIX + 2 * HC + c * HC + r];
            }
            if (std::abs(row_sum - 1.0f) > 5e-5f || std::abs(col_sum - 1.0f) > 5e-5f) return false;
        }
    }
    return true;
}

bool test_hc_collapse_post_matches_reference() {
    constexpr size_t T = 2;
    constexpr size_t D = 3;
    std::vector<float> streams(T * HC * D);
    std::vector<float> sublayer(T * D);
    std::vector<float> mix(T * MIX, 0.0f);
    for (size_t i = 0; i < streams.size(); ++i) streams[i] = 0.1f * static_cast<float>(i + 1);
    for (size_t i = 0; i < sublayer.size(); ++i) sublayer[i] = -0.2f + 0.07f * static_cast<float>(i);
    for (size_t t = 0; t < T; ++t) {
        for (size_t h = 0; h < HC; ++h) {
            mix[t * MIX + h] = 0.1f * static_cast<float>(h + 1);
            mix[t * MIX + HC + h] = 0.2f * static_cast<float>(h + 1);
        }
        float comb[HC][HC] = {
            {0.10f, 0.20f, 0.30f, 0.40f},
            {0.50f, 0.10f, 0.20f, 0.20f},
            {0.15f, 0.35f, 0.25f, 0.25f},
            {0.25f, 0.25f, 0.15f, 0.35f},
        };
        for (size_t r = 0; r < HC; ++r) {
            for (size_t c = 0; c < HC; ++c) {
                mix[t * MIX + 2 * HC + r * HC + c] = comb[r][c] + 0.01f * static_cast<float>(t);
            }
        }
    }

    CactusGraph graph;
    size_t streams_id = graph.input({T, HC, D}, Precision::FP32);
    size_t sublayer_id = graph.input({T, D}, Precision::FP32);
    size_t mix_id = graph.input({T, MIX}, Precision::FP32);
    size_t collapse_id = graph.dsv4_hc_collapse(streams_id, mix_id);
    size_t post_id = graph.dsv4_hc_post(sublayer_id, streams_id, mix_id);
    graph.set_input(streams_id, streams.data(), Precision::FP32);
    graph.set_input(sublayer_id, sublayer.data(), Precision::FP32);
    graph.set_input(mix_id, mix.data(), Precision::FP32);
    graph.execute();

    return compare_float_output(graph, collapse_id, reference_collapse(streams, mix, T, D), 1e-6f) &&
           compare_float_output(graph, post_id, reference_post(sublayer, streams, mix, T, D), 1e-6f);
}

bool test_hc_head_matches_reference() {
    constexpr size_t T = 3;
    constexpr size_t D = 4;
    constexpr float eps = 1e-6f;
    std::vector<float> streams(T * HC * D);
    std::vector<float> fn(HC * HC * D);
    std::vector<float> base(HC);
    std::vector<float> scale = {0.6f};
    for (size_t i = 0; i < streams.size(); ++i) streams[i] = std::cos(0.07f * static_cast<float>(i + 1));
    for (size_t i = 0; i < fn.size(); ++i) fn[i] = 0.04f * std::sin(0.19f * static_cast<float>(i + 2));
    for (size_t i = 0; i < base.size(); ++i) base[i] = -0.08f + 0.03f * static_cast<float>(i);

    CactusGraph graph;
    size_t streams_id = graph.input({T, HC, D}, Precision::FP32);
    size_t fn_id = graph.input({HC, HC * D}, Precision::FP32);
    size_t base_id = graph.input({HC}, Precision::FP32);
    size_t scale_id = graph.input({1}, Precision::FP32);
    size_t head_id = graph.dsv4_hc_head(streams_id, fn_id, base_id, scale_id, eps);
    graph.set_input(streams_id, streams.data(), Precision::FP32);
    graph.set_input(fn_id, fn.data(), Precision::FP32);
    graph.set_input(base_id, base.data(), Precision::FP32);
    graph.set_input(scale_id, scale.data(), Precision::FP32);
    graph.execute();

    return compare_float_output(graph, head_id, reference_head(streams, fn, base, scale, T, D, eps), 1e-5f);
}

int main() {
    TestUtils::TestRunner runner("DeepSeek V4 mHC Tests");
    runner.run_test("HC mix", test_hc_mix_matches_reference());
    runner.run_test("HC collapse/post", test_hc_collapse_post_matches_reference());
    runner.run_test("HC head", test_hc_head_matches_reference());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
