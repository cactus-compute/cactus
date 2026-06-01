#include "test_utils.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace TestUtils;

namespace {

std::vector<float> reference_hca(const std::vector<float>& kv,
                                 const std::vector<float>& score,
                                 const std::vector<float>& ape,
                                 const std::vector<float>& norm_weight,
                                 size_t batch,
                                 size_t seq,
                                 size_t head_dim,
                                 size_t ratio,
                                 float eps) {
    const size_t chunks = seq / ratio;
    std::vector<float> out(batch * chunks * head_dim);
    std::vector<float> compressed(head_dim);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < chunks; ++c) {
            for (size_t d = 0; d < head_dim; ++d) {
                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t r = 0; r < ratio; ++r) {
                    float s = score[(b * seq + c * ratio + r) * head_dim + d] + ape[r * head_dim + d];
                    max_score = std::max(max_score, s);
                }
                float denom = 0.0f;
                float acc = 0.0f;
                for (size_t r = 0; r < ratio; ++r) {
                    size_t src = (b * seq + c * ratio + r) * head_dim + d;
                    float w = std::exp(score[src] + ape[r * head_dim + d] - max_score);
                    denom += w;
                    acc += w * kv[src];
                }
                compressed[d] = acc / denom;
            }
            double mean_square = 0.0;
            for (float value : compressed) mean_square += static_cast<double>(value) * value;
            float rsqrt = 1.0f / std::sqrt(static_cast<float>(mean_square / head_dim) + eps);
            for (size_t d = 0; d < head_dim; ++d) {
                out[(b * chunks + c) * head_dim + d] = compressed[d] * rsqrt * norm_weight[d];
            }
        }
    }
    return out;
}

std::vector<float> reference_csa(const std::vector<float>& kv,
                                 const std::vector<float>& score,
                                 const std::vector<float>& ape,
                                 const std::vector<float>& norm_weight,
                                 size_t batch,
                                 size_t seq,
                                 size_t head_dim,
                                 size_t ratio,
                                 float eps) {
    const size_t chunks = seq / ratio;
    std::vector<float> out(batch * chunks * head_dim);
    std::vector<float> compressed(head_dim);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < chunks; ++c) {
            for (size_t d = 0; d < head_dim; ++d) {
                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t slot = 0; slot < 2 * ratio; ++slot) {
                    if (slot < ratio && c == 0) continue;
                    size_t src_chunk = slot < ratio ? c - 1 : c;
                    size_t r = slot % ratio;
                    size_t dim = slot < ratio ? d : head_dim + d;
                    float s = score[(b * seq + src_chunk * ratio + r) * (2 * head_dim) + dim] +
                              ape[r * (2 * head_dim) + dim];
                    max_score = std::max(max_score, s);
                }
                float denom = 0.0f;
                float acc = 0.0f;
                for (size_t slot = 0; slot < 2 * ratio; ++slot) {
                    if (slot < ratio && c == 0) continue;
                    size_t src_chunk = slot < ratio ? c - 1 : c;
                    size_t r = slot % ratio;
                    size_t dim = slot < ratio ? d : head_dim + d;
                    size_t src = (b * seq + src_chunk * ratio + r) * (2 * head_dim) + dim;
                    float w = std::exp(score[src] + ape[r * (2 * head_dim) + dim] - max_score);
                    denom += w;
                    acc += w * kv[src];
                }
                compressed[d] = acc / denom;
            }
            double mean_square = 0.0;
            for (float value : compressed) mean_square += static_cast<double>(value) * value;
            float rsqrt = 1.0f / std::sqrt(static_cast<float>(mean_square / head_dim) + eps);
            for (size_t d = 0; d < head_dim; ++d) {
                out[(b * chunks + c) * head_dim + d] = compressed[d] * rsqrt * norm_weight[d];
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

bool run_hca_case(size_t seq) {
    constexpr size_t B = 1;
    constexpr size_t D = 6;
    constexpr size_t R = 128;
    constexpr float eps = 1e-6f;
    std::vector<float> kv(B * seq * D);
    std::vector<float> score(B * seq * D);
    std::vector<float> ape(R * D);
    std::vector<float> norm(D);
    for (size_t i = 0; i < kv.size(); ++i) {
        kv[i] = 0.2f * std::sin(0.013f * static_cast<float>(i + 1));
        score[i] = 0.15f * std::cos(0.017f * static_cast<float>(i + 3));
    }
    for (size_t i = 0; i < ape.size(); ++i) ape[i] = 0.03f * std::sin(0.021f * static_cast<float>(i + 5));
    for (size_t i = 0; i < norm.size(); ++i) norm[i] = 0.7f + 0.05f * static_cast<float>(i);

    CactusGraph graph;
    size_t kv_id = graph.input({B, seq, D}, Precision::FP32);
    size_t score_id = graph.input({B, seq, D}, Precision::FP32);
    size_t ape_id = graph.input({R, D}, Precision::FP32);
    size_t norm_id = graph.input({D}, Precision::FP32);
    size_t out_id = graph.dsv4_compress_hca(kv_id, score_id, ape_id, norm_id, eps, R);
    graph.set_input(kv_id, kv.data(), Precision::FP32);
    graph.set_input(score_id, score.data(), Precision::FP32);
    graph.set_input(ape_id, ape.data(), Precision::FP32);
    graph.set_input(norm_id, norm.data(), Precision::FP32);
    graph.execute();

    auto expected = reference_hca(kv, score, ape, norm, B, seq, D, R, eps);
    return compare_output(static_cast<const float*>(graph.get_output(out_id)), expected, 1e-6f);
}

bool run_csa_case(size_t seq) {
    constexpr size_t B = 1;
    constexpr size_t D = 5;
    constexpr size_t R = 4;
    constexpr float eps = 1e-6f;
    std::vector<float> kv(B * seq * 2 * D);
    std::vector<float> score(B * seq * 2 * D);
    std::vector<float> ape(R * 2 * D);
    std::vector<float> norm(D);
    for (size_t i = 0; i < kv.size(); ++i) {
        kv[i] = 0.18f * std::sin(0.09f * static_cast<float>(i + 2));
        score[i] = 0.12f * std::cos(0.07f * static_cast<float>(i + 4));
    }
    for (size_t i = 0; i < ape.size(); ++i) ape[i] = 0.025f * std::sin(0.05f * static_cast<float>(i + 1));
    for (size_t i = 0; i < norm.size(); ++i) norm[i] = 0.8f + 0.03f * static_cast<float>(i);

    CactusGraph graph;
    size_t kv_id = graph.input({B, seq, 2 * D}, Precision::FP32);
    size_t score_id = graph.input({B, seq, 2 * D}, Precision::FP32);
    size_t ape_id = graph.input({R, 2 * D}, Precision::FP32);
    size_t norm_id = graph.input({D}, Precision::FP32);
    size_t out_id = graph.dsv4_compress_csa(kv_id, score_id, ape_id, norm_id, eps, R);
    graph.set_input(kv_id, kv.data(), Precision::FP32);
    graph.set_input(score_id, score.data(), Precision::FP32);
    graph.set_input(ape_id, ape.data(), Precision::FP32);
    graph.set_input(norm_id, norm.data(), Precision::FP32);
    graph.execute();

    auto expected = reference_csa(kv, score, ape, norm, B, seq, D, R, eps);
    return compare_output(static_cast<const float*>(graph.get_output(out_id)), expected, 1e-6f);
}

} // namespace

int main() {
    TestUtils::TestRunner runner("DeepSeek V4 Compressor Tests");
    runner.run_test("HCA 128", run_hca_case(128));
    runner.run_test("HCA 129", run_hca_case(129));
    runner.run_test("HCA 256", run_hca_case(256));
    runner.run_test("CSA 4", run_csa_case(4));
    runner.run_test("CSA 5", run_csa_case(5));
    runner.run_test("CSA 8", run_csa_case(8));
    runner.run_test("CSA 9", run_csa_case(9));
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
