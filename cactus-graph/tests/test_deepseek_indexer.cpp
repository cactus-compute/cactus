#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace TestUtils;

namespace {

std::vector<float> reference_indexer(const std::vector<float>& query,
                                     const std::vector<float>& compressed_kv,
                                     const std::vector<float>& weights,
                                     const std::vector<float>& position_ids,
                                     size_t batch,
                                     size_t seq,
                                     size_t heads,
                                     size_t head_dim,
                                     size_t compressed_len,
                                     size_t top_k,
                                     size_t ratio,
                                     size_t offset,
                                     float scale) {
    std::vector<float> out(batch * seq * top_k);
    std::vector<std::pair<float, size_t>> scores(compressed_len);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq; ++t) {
            size_t position = static_cast<size_t>(position_ids[b * seq + t] + 0.5f);
            size_t threshold = (position + 1) / ratio;
            for (size_t n = 0; n < compressed_len; ++n) {
                if (n >= threshold) {
                    scores[n] = {-std::numeric_limits<float>::infinity(), n};
                    continue;
                }
                float score = 0.0f;
                for (size_t h = 0; h < heads; ++h) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        dot += query[((b * seq + t) * heads + h) * head_dim + d] *
                               compressed_kv[(b * compressed_len + n) * head_dim + d];
                    }
                    score += std::max(dot, 0.0f) * weights[(b * seq + t) * heads + h] * scale;
                }
                scores[n] = {score, n};
            }
            std::partial_sort(scores.begin(),
                              scores.begin() + static_cast<std::ptrdiff_t>(top_k),
                              scores.end(),
                              [](const auto& a, const auto& b) {
                                  if (a.first != b.first) return a.first > b.first;
                                  return a.second < b.second;
                              });
            for (size_t k = 0; k < top_k; ++k) {
                size_t idx = scores[k].second;
                out[(b * seq + t) * top_k + k] = idx >= threshold ? -1.0f : static_cast<float>(idx + offset);
            }
        }
    }
    return out;
}

bool compare_output(const float* actual, const std::vector<float>& expected) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << "mismatch at " << i << ": actual=" << actual[i]
                      << " expected=" << expected[i] << "\n";
            return false;
        }
    }
    return true;
}

} // namespace

bool test_indexer_topk_and_sentinels() {
    constexpr size_t B = 1;
    constexpr size_t S = 12;
    constexpr size_t H = 3;
    constexpr size_t D = 4;
    constexpr size_t N = 3;
    constexpr size_t K = 2;
    constexpr size_t R = 4;
    constexpr size_t offset = 11;
    constexpr float scale = 0.25f;

    std::vector<float> query(B * S * H * D);
    std::vector<float> compressed(B * N * D);
    std::vector<float> weights(B * S * H);
    std::vector<float> positions(B * S);
    for (size_t i = 0; i < query.size(); ++i) query[i] = 0.2f * std::sin(0.11f * static_cast<float>(i + 1));
    for (size_t i = 0; i < compressed.size(); ++i) compressed[i] = 0.3f * std::cos(0.17f * static_cast<float>(i + 2));
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = -0.4f + 0.09f * static_cast<float>((i * 7) % 11);
    for (size_t i = 0; i < positions.size(); ++i) positions[i] = static_cast<float>(i);

    CactusGraph graph;
    size_t q_id = graph.input({B, S, H, D}, Precision::FP32);
    size_t kv_id = graph.input({B, N, D}, Precision::FP32);
    size_t weights_id = graph.input({B, S, H}, Precision::FP32);
    size_t pos_id = graph.input({B, S}, Precision::FP32);
    size_t out_id = graph.dsv4_indexer_topk(q_id, kv_id, weights_id, pos_id, K, R, offset, scale);
    graph.set_input(q_id, query.data(), Precision::FP32);
    graph.set_input(kv_id, compressed.data(), Precision::FP32);
    graph.set_input(weights_id, weights.data(), Precision::FP32);
    graph.set_input(pos_id, positions.data(), Precision::FP32);
    graph.execute();

    auto expected = reference_indexer(query, compressed, weights, positions, B, S, H, D, N, K, R, offset, scale);
    return compare_output(static_cast<const float*>(graph.get_output(out_id)), expected);
}

int main() {
    TestUtils::TestRunner runner("DeepSeek V4 Indexer Tests");
    runner.run_test("topk sentinels", test_indexer_topk_and_sentinels());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
