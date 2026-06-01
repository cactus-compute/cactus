#include "test_utils.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace TestUtils;

namespace {

std::vector<float> reference_sparse_attention(const std::vector<float>& query,
                                              const std::vector<float>& kv,
                                              const std::vector<float>& sink,
                                              const std::vector<float>& indices,
                                              size_t batch,
                                              size_t seq,
                                              size_t heads,
                                              size_t head_dim,
                                              size_t kv_len,
                                              size_t top_k,
                                              float scale) {
    std::vector<float> out(batch * seq * heads * head_dim, 0.0f);
    std::vector<float> logits(top_k);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq; ++t) {
            for (size_t h = 0; h < heads; ++h) {
                float max_logit = sink[h];
                for (size_t k = 0; k < top_k; ++k) {
                    float raw_idx = indices[(b * seq + t) * top_k + k];
                    if (raw_idx < 0.0f) {
                        logits[k] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    size_t idx = static_cast<size_t>(raw_idx + 0.5f);
                    float logit = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        logit += query[((b * seq + t) * heads + h) * head_dim + d] *
                                 kv[(b * kv_len + idx) * head_dim + d];
                    }
                    logit *= scale;
                    logits[k] = logit;
                    max_logit = std::max(max_logit, logit);
                }
                float denom = std::exp(sink[h] - max_logit);
                for (float logit : logits) {
                    if (std::isfinite(logit)) denom += std::exp(logit - max_logit);
                }
                for (size_t d = 0; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (size_t k = 0; k < top_k; ++k) {
                        if (!std::isfinite(logits[k])) continue;
                        size_t idx = static_cast<size_t>(indices[(b * seq + t) * top_k + k] + 0.5f);
                        float prob = std::exp(logits[k] - max_logit) / denom;
                        acc += prob * kv[(b * kv_len + idx) * head_dim + d];
                    }
                    out[((b * seq + t) * heads + h) * head_dim + d] = acc;
                }
            }
        }
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

bool test_sparse_attention_matches_reference() {
    constexpr size_t B = 1;
    constexpr size_t S = 2;
    constexpr size_t H = 2;
    constexpr size_t D = 4;
    constexpr size_t N = 5;
    constexpr size_t K = 4;
    constexpr float scale = 0.5f;

    std::vector<float> query(B * S * H * D);
    std::vector<float> kv(B * N * D);
    std::vector<float> sink = {0.1f, -0.2f};
    std::vector<float> indices = {
        0.0f, 1.0f, -1.0f, 3.0f,
        2.0f, -1.0f, 4.0f, 2.0f,
    };
    for (size_t i = 0; i < query.size(); ++i) query[i] = 0.2f * std::sin(0.17f * static_cast<float>(i + 1));
    for (size_t i = 0; i < kv.size(); ++i) kv[i] = 0.3f * std::cos(0.11f * static_cast<float>(i + 2));

    CactusGraph graph;
    size_t q_id = graph.input({B, S, H, D}, Precision::FP32);
    size_t kv_id = graph.input({B, N, D}, Precision::FP32);
    size_t sink_id = graph.input({H}, Precision::FP32);
    size_t idx_id = graph.input({B, S, K}, Precision::FP32);
    size_t out_id = graph.dsv4_sparse_attention(q_id, kv_id, sink_id, idx_id, scale);
    graph.set_input(q_id, query.data(), Precision::FP32);
    graph.set_input(kv_id, kv.data(), Precision::FP32);
    graph.set_input(sink_id, sink.data(), Precision::FP32);
    graph.set_input(idx_id, indices.data(), Precision::FP32);
    graph.execute();

    auto expected = reference_sparse_attention(query, kv, sink, indices, B, S, H, D, N, K, scale);
    return compare_output(static_cast<const float*>(graph.get_output(out_id)), expected, 1e-6f);
}

bool test_sparse_attention_sink_contributes_no_value() {
    constexpr size_t B = 1;
    constexpr size_t S = 1;
    constexpr size_t H = 1;
    constexpr size_t D = 2;
    constexpr size_t N = 2;
    constexpr size_t K = 2;
    constexpr float scale = 1.0f;

    std::vector<float> query = {0.0f, 0.0f};
    std::vector<float> kv = {2.0f, 0.0f, 0.0f, 2.0f};
    std::vector<float> indices = {0.0f, 1.0f};
    std::vector<float> low_sink = {-5.0f};
    std::vector<float> high_sink = {5.0f};

    CactusGraph graph_low;
    size_t q_low = graph_low.input({B, S, H, D}, Precision::FP32);
    size_t kv_low = graph_low.input({B, N, D}, Precision::FP32);
    size_t sink_low = graph_low.input({H}, Precision::FP32);
    size_t idx_low = graph_low.input({B, S, K}, Precision::FP32);
    size_t out_low = graph_low.dsv4_sparse_attention(q_low, kv_low, sink_low, idx_low, scale);
    graph_low.set_input(q_low, query.data(), Precision::FP32);
    graph_low.set_input(kv_low, kv.data(), Precision::FP32);
    graph_low.set_input(sink_low, low_sink.data(), Precision::FP32);
    graph_low.set_input(idx_low, indices.data(), Precision::FP32);
    graph_low.execute();

    CactusGraph graph_high;
    size_t q_high = graph_high.input({B, S, H, D}, Precision::FP32);
    size_t kv_high = graph_high.input({B, N, D}, Precision::FP32);
    size_t sink_high = graph_high.input({H}, Precision::FP32);
    size_t idx_high = graph_high.input({B, S, K}, Precision::FP32);
    size_t out_high = graph_high.dsv4_sparse_attention(q_high, kv_high, sink_high, idx_high, scale);
    graph_high.set_input(q_high, query.data(), Precision::FP32);
    graph_high.set_input(kv_high, kv.data(), Precision::FP32);
    graph_high.set_input(sink_high, high_sink.data(), Precision::FP32);
    graph_high.set_input(idx_high, indices.data(), Precision::FP32);
    graph_high.execute();

    auto low_expected = reference_sparse_attention(query, kv, low_sink, indices, B, S, H, D, N, K, scale);
    auto high_expected = reference_sparse_attention(query, kv, high_sink, indices, B, S, H, D, N, K, scale);
    const auto* low_actual = static_cast<const float*>(graph_low.get_output(out_low));
    const auto* high_actual = static_cast<const float*>(graph_high.get_output(out_high));
    if (!compare_output(low_actual, low_expected, 1e-6f)) return false;
    if (!compare_output(high_actual, high_expected, 1e-6f)) return false;
    float low_norm = std::abs(low_actual[0]) + std::abs(low_actual[1]);
    float high_norm = std::abs(high_actual[0]) + std::abs(high_actual[1]);
    return high_norm < low_norm;
}

int main() {
    TestUtils::TestRunner runner("DeepSeek V4 Sparse Attention Tests");
    runner.run_test("sparse attention", test_sparse_attention_matches_reference());
    runner.run_test("sink no value", test_sparse_attention_sink_contributes_no_value());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
