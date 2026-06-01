#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace TestUtils;

namespace {

float sigmoid_ref(float x) {
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    float z = std::exp(x);
    return z / (1.0f + z);
}

float silu_ref(float x) {
    return x * sigmoid_ref(x);
}

float softplus_ref(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

float sqrt_softplus_ref(float x) {
    return std::sqrt(softplus_ref(x));
}

float dot_router(const std::vector<float>& hidden,
                 const std::vector<float>& weight,
                 size_t token,
                 size_t expert,
                 size_t hidden_dim) {
    float acc = 0.0f;
    for (size_t d = 0; d < hidden_dim; ++d) {
        acc += hidden[token * hidden_dim + d] * weight[expert * hidden_dim + d];
    }
    return acc;
}

std::vector<float> reference_routed_router(const std::vector<float>& hidden,
                                           const std::vector<float>& weight,
                                           const std::vector<float>& bias,
                                           size_t tokens,
                                           size_t hidden_dim,
                                           size_t experts,
                                           size_t top_k,
                                           float route_scale,
                                           float eps) {
    std::vector<float> out(2 * tokens * top_k);
    std::vector<float> original_scores(experts);
    std::vector<std::pair<float, size_t>> selection(experts);
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t e = 0; e < experts; ++e) {
            float score = sqrt_softplus_ref(dot_router(hidden, weight, t, e, hidden_dim));
            original_scores[e] = score;
            selection[e] = {score + bias[e], e};
        }
        std::partial_sort(selection.begin(),
                          selection.begin() + static_cast<std::ptrdiff_t>(top_k),
                          selection.end(),
                          [](const auto& a, const auto& b) {
                              if (a.first != b.first) return a.first > b.first;
                              return a.second < b.second;
                          });
        float denom = eps;
        for (size_t k = 0; k < top_k; ++k) denom += original_scores[selection[k].second];
        for (size_t k = 0; k < top_k; ++k) {
            size_t expert = selection[k].second;
            out[t * top_k + k] = static_cast<float>(expert);
            out[tokens * top_k + t * top_k + k] = original_scores[expert] / denom * route_scale;
        }
    }
    return out;
}

std::vector<float> reference_hash_router(const std::vector<float>& hidden,
                                         const std::vector<float>& input_ids,
                                         const std::vector<float>& weight,
                                         const std::vector<float>& tid2eid,
                                         size_t tokens,
                                         size_t hidden_dim,
                                         size_t experts,
                                         size_t top_k,
                                         float route_scale,
                                         float eps) {
    std::vector<float> out(2 * tokens * top_k);
    std::vector<float> original_scores(experts);
    for (size_t t = 0; t < tokens; ++t) {
        size_t token_id = static_cast<size_t>(input_ids[t] + 0.5f);
        for (size_t e = 0; e < experts; ++e) {
            original_scores[e] = sqrt_softplus_ref(dot_router(hidden, weight, t, e, hidden_dim));
        }
        float denom = eps;
        for (size_t k = 0; k < top_k; ++k) {
            size_t expert = static_cast<size_t>(tid2eid[token_id * top_k + k] + 0.5f);
            out[t * top_k + k] = static_cast<float>(expert);
            denom += original_scores[expert];
        }
        for (size_t k = 0; k < top_k; ++k) {
            size_t expert = static_cast<size_t>(out[t * top_k + k] + 0.5f);
            out[tokens * top_k + t * top_k + k] = original_scores[expert] / denom * route_scale;
        }
    }
    return out;
}

std::vector<float> reference_moe(const std::vector<float>& hidden,
                                 const std::vector<float>& route,
                                 const std::vector<std::vector<float>>& gate_weights,
                                 const std::vector<std::vector<float>>& up_weights,
                                 const std::vector<std::vector<float>>& down_weights,
                                 size_t tokens,
                                 size_t hidden_dim,
                                 size_t inter_dim,
                                 size_t top_k,
                                 float limit) {
    std::vector<float> out(tokens * hidden_dim, 0.0f);
    std::vector<float> product(inter_dim);
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t k = 0; k < top_k; ++k) {
            size_t expert = static_cast<size_t>(route[t * top_k + k] + 0.5f);
            float route_weight = route[tokens * top_k + t * top_k + k];
            for (size_t i = 0; i < inter_dim; ++i) {
                float gate = 0.0f;
                float up = 0.0f;
                for (size_t d = 0; d < hidden_dim; ++d) {
                    float x = hidden[t * hidden_dim + d];
                    gate += x * gate_weights[expert][i * hidden_dim + d];
                    up += x * up_weights[expert][i * hidden_dim + d];
                }
                if (limit > 0.0f) {
                    gate = std::min(gate, limit);
                    up = std::clamp(up, -limit, limit);
                }
                product[i] = silu_ref(gate) * up;
            }
            for (size_t d = 0; d < hidden_dim; ++d) {
                float expert_out = 0.0f;
                for (size_t i = 0; i < inter_dim; ++i) {
                    expert_out += product[i] * down_weights[expert][d * inter_dim + i];
                }
                out[t * hidden_dim + d] += route_weight * expert_out;
            }
        }
    }
    return out;
}

std::vector<float> reference_shared(const std::vector<float>& hidden,
                                    const std::vector<float>& gate_weight,
                                    const std::vector<float>& up_weight,
                                    const std::vector<float>& down_weight,
                                    size_t tokens,
                                    size_t hidden_dim,
                                    size_t inter_dim,
                                    float limit) {
    std::vector<float> out(tokens * hidden_dim);
    std::vector<float> product(inter_dim);
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t i = 0; i < inter_dim; ++i) {
            float gate = 0.0f;
            float up = 0.0f;
            for (size_t d = 0; d < hidden_dim; ++d) {
                float x = hidden[t * hidden_dim + d];
                gate += x * gate_weight[i * hidden_dim + d];
                up += x * up_weight[i * hidden_dim + d];
            }
            if (limit > 0.0f) {
                gate = std::min(gate, limit);
                up = std::clamp(up, -limit, limit);
            }
            product[i] = silu_ref(gate) * up;
        }
        for (size_t d = 0; d < hidden_dim; ++d) {
            float acc = 0.0f;
            for (size_t i = 0; i < inter_dim; ++i) {
                acc += product[i] * down_weight[d * inter_dim + i];
            }
            out[t * hidden_dim + d] = acc;
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

bool test_routed_router_matches_reference() {
    constexpr size_t T = 3;
    constexpr size_t D = 4;
    constexpr size_t E = 5;
    constexpr size_t K = 2;
    constexpr float route_scale = 1.7f;
    constexpr float eps = 1e-20f;

    std::vector<float> hidden(T * D);
    std::vector<float> weight(E * D);
    std::vector<float> bias = {-0.01f, 0.03f, -0.02f, 0.05f, 0.00f};
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = std::sin(0.23f * static_cast<float>(i + 1));
    for (size_t i = 0; i < weight.size(); ++i) weight[i] = 0.17f * std::cos(0.19f * static_cast<float>(i + 2));

    CactusGraph graph;
    size_t hidden_id = graph.input({T, D}, Precision::FP32);
    size_t weight_id = graph.input({E, D}, Precision::FP32);
    size_t bias_id = graph.input({E}, Precision::FP32);
    size_t route_id = graph.dsv4_router_topk(hidden_id, weight_id, bias_id, E, K, route_scale, eps);
    graph.set_input(hidden_id, hidden.data(), Precision::FP32);
    graph.set_input(weight_id, weight.data(), Precision::FP32);
    graph.set_input(bias_id, bias.data(), Precision::FP32);
    graph.execute();

    auto expected = reference_routed_router(hidden, weight, bias, T, D, E, K, route_scale, eps);
    return compare_output(static_cast<const float*>(graph.get_output(route_id)), expected, 1e-6f);
}

bool test_hash_router_matches_reference() {
    constexpr size_t T = 4;
    constexpr size_t D = 3;
    constexpr size_t E = 6;
    constexpr size_t K = 3;
    constexpr size_t V = 7;
    constexpr float route_scale = 0.85f;
    constexpr float eps = 1e-20f;

    std::vector<float> hidden(T * D);
    std::vector<float> input_ids = {0.0f, 2.0f, 4.0f, 6.0f};
    std::vector<float> weight(E * D);
    std::vector<float> tid2eid(V * K);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = -0.4f + 0.11f * static_cast<float>(i);
    for (size_t i = 0; i < weight.size(); ++i) weight[i] = 0.13f * std::sin(0.31f * static_cast<float>(i + 1));
    for (size_t token = 0; token < V; ++token) {
        for (size_t k = 0; k < K; ++k) {
            tid2eid[token * K + k] = static_cast<float>((token + 2 * k + 1) % E);
        }
    }

    CactusGraph graph;
    size_t hidden_id = graph.input({T, D}, Precision::FP32);
    size_t ids_id = graph.input({T}, Precision::FP32);
    size_t weight_id = graph.input({E, D}, Precision::FP32);
    size_t tid_id = graph.input({V, K}, Precision::FP32);
    size_t route_id = graph.dsv4_hash_router(hidden_id, ids_id, weight_id, tid_id, E, K, route_scale, eps);
    graph.set_input(hidden_id, hidden.data(), Precision::FP32);
    graph.set_input(ids_id, input_ids.data(), Precision::FP32);
    graph.set_input(weight_id, weight.data(), Precision::FP32);
    graph.set_input(tid_id, tid2eid.data(), Precision::FP32);
    graph.execute();

    auto expected = reference_hash_router(hidden, input_ids, weight, tid2eid, T, D, E, K, route_scale, eps);
    return compare_output(static_cast<const float*>(graph.get_output(route_id)), expected, 1e-6f);
}

bool test_moe_layer_matches_reference() {
    constexpr size_t T = 3;
    constexpr size_t D = 4;
    constexpr size_t I = 5;
    constexpr size_t E = 4;
    constexpr size_t K = 2;
    constexpr float limit = 0.45f;

    std::vector<float> hidden(T * D);
    std::vector<float> route(2 * T * K);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = std::cos(0.17f * static_cast<float>(i + 2));
    route = {
        1.0f, 2.0f,
        2.0f, 0.0f,
        1.0f, 1.0f,
        0.65f, 0.35f,
        0.40f, 0.60f,
        0.25f, 0.75f,
    };

    std::vector<std::vector<float>> gate(E, std::vector<float>(I * D));
    std::vector<std::vector<float>> up(E, std::vector<float>(I * D));
    std::vector<std::vector<float>> down(E, std::vector<float>(D * I));
    for (size_t e = 0; e < E; ++e) {
        for (size_t i = 0; i < I * D; ++i) {
            gate[e][i] = 0.2f * std::sin(0.13f * static_cast<float>(1 + i + 7 * e));
            up[e][i] = 0.25f * std::cos(0.11f * static_cast<float>(3 + i + 5 * e));
        }
        for (size_t i = 0; i < D * I; ++i) {
            down[e][i] = 0.18f * std::sin(0.07f * static_cast<float>(2 + i + 11 * e));
        }
    }

    CactusGraph graph;
    size_t hidden_id = graph.input({T, D}, Precision::FP32);
    size_t route_id = graph.input({2, T, K}, Precision::FP32);
    std::vector<size_t> gate_ids;
    std::vector<size_t> up_ids;
    std::vector<size_t> down_ids;
    for (size_t e = 0; e < E; ++e) gate_ids.push_back(graph.input({I, D}, Precision::FP32));
    for (size_t e = 0; e < E; ++e) up_ids.push_back(graph.input({I, D}, Precision::FP32));
    for (size_t e = 0; e < E; ++e) down_ids.push_back(graph.input({D, I}, Precision::FP32));
    size_t out_id = graph.dsv4_moe_layer(hidden_id, route_id, gate_ids, up_ids, down_ids, E, K, limit);

    graph.set_input(hidden_id, hidden.data(), Precision::FP32);
    graph.set_input(route_id, route.data(), Precision::FP32);
    for (size_t e = 0; e < E; ++e) graph.set_input(gate_ids[e], gate[e].data(), Precision::FP32);
    for (size_t e = 0; e < E; ++e) graph.set_input(up_ids[e], up[e].data(), Precision::FP32);
    for (size_t e = 0; e < E; ++e) graph.set_input(down_ids[e], down[e].data(), Precision::FP32);
    graph.execute();

    auto expected = reference_moe(hidden, route, gate, up, down, T, D, I, K, limit);
    return compare_output(static_cast<const float*>(graph.get_output(out_id)), expected, 1e-6f);
}

bool test_shared_expert_matches_reference() {
    constexpr size_t T = 2;
    constexpr size_t D = 4;
    constexpr size_t I = 5;
    constexpr float limit = 0.35f;
    std::vector<float> hidden(T * D);
    std::vector<float> gate(I * D);
    std::vector<float> up(I * D);
    std::vector<float> down(D * I);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = 0.15f * std::sin(0.19f * static_cast<float>(i + 1));
    for (size_t i = 0; i < gate.size(); ++i) {
        gate[i] = 0.3f * std::cos(0.23f * static_cast<float>(i + 2));
        up[i] = 0.25f * std::sin(0.17f * static_cast<float>(i + 3));
    }
    for (size_t i = 0; i < down.size(); ++i) down[i] = 0.2f * std::cos(0.13f * static_cast<float>(i + 4));

    CactusGraph graph;
    size_t hidden_id = graph.input({T, D}, Precision::FP32);
    size_t gate_id = graph.input({I, D}, Precision::FP32);
    size_t up_id = graph.input({I, D}, Precision::FP32);
    size_t down_id = graph.input({D, I}, Precision::FP32);
    size_t out_id = graph.dsv4_shared_expert(hidden_id, gate_id, up_id, down_id, limit);
    graph.set_input(hidden_id, hidden.data(), Precision::FP32);
    graph.set_input(gate_id, gate.data(), Precision::FP32);
    graph.set_input(up_id, up.data(), Precision::FP32);
    graph.set_input(down_id, down.data(), Precision::FP32);
    graph.execute();

    auto expected = reference_shared(hidden, gate, up, down, T, D, I, limit);
    return compare_output(static_cast<const float*>(graph.get_output(out_id)), expected, 1e-6f);
}

int main() {
    TestUtils::TestRunner runner("DeepSeek V4 MoE Tests");
    runner.run_test("routed router", test_routed_router_matches_reference());
    runner.run_test("hash router", test_hash_router_matches_reference());
    runner.run_test("moe selected experts", test_moe_layer_matches_reference());
    runner.run_test("shared expert", test_shared_expert_matches_reference());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
