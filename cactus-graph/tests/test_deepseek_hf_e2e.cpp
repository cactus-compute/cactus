#include "test_utils.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

using namespace TestUtils;

namespace {

constexpr size_t T = 3;
constexpr size_t V = 9;
constexpr size_t D = 8;
constexpr size_t HC = 4;
constexpr size_t MIX = 24;
constexpr size_t H = 2;
constexpr size_t HD = 4;
constexpr size_t RD = 2;
constexpr size_t QA = 6;
constexpr size_t OG = 2;
constexpr size_t OR = 3;
constexpr size_t E = 3;
constexpr size_t I = 5;
constexpr size_t K = 2;
constexpr float EPS = 1.0e-6f;
constexpr float SWIGLU_LIMIT = 10.0f;

float seed_value(size_t i, float scale, float phase) {
    return scale * std::sin(phase + 0.173f * static_cast<float>(i + 1));
}

void fill_half(std::vector<__fp16>& v, float scale, float phase) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<__fp16>(seed_value(i, scale, phase));
}

void fill_float(std::vector<float>& v, float scale, float phase) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = seed_value(i, scale, phase);
}

struct HFWeights {
    std::vector<__fp16> embed;
    std::vector<float> hc_attn_fn, hc_attn_base, hc_attn_scale;
    std::vector<float> hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    std::vector<float> hc_head_fn, hc_head_base, hc_head_scale;
    std::vector<__fp16> input_norm, post_norm;
    std::vector<__fp16> q_a_proj, q_a_norm, q_b_proj;
    std::vector<__fp16> kv_proj, kv_norm, o_a_proj, o_b_proj;
    std::vector<float> attn_sink, topk_indices;
    std::vector<__fp16> router, router_bias;
    std::vector<std::vector<__fp16>> expert_gate, expert_up, expert_down;
    std::vector<__fp16> shared_gate, shared_up, shared_down;
    std::vector<__fp16> final_norm, lm_head;
};

HFWeights make_hf_weights() {
    HFWeights w;
    w.embed.resize(V * D);
    w.hc_attn_fn.resize(MIX * HC * D);
    w.hc_attn_base.resize(MIX);
    w.hc_attn_scale = {0.7f, -0.35f, 0.2f};
    w.hc_ffn_fn.resize(MIX * HC * D);
    w.hc_ffn_base.resize(MIX);
    w.hc_ffn_scale = {-0.4f, 0.55f, 0.25f};
    w.hc_head_fn.resize(HC * HC * D);
    w.hc_head_base.resize(HC);
    w.hc_head_scale = {0.6f};
    w.input_norm.resize(D);
    w.post_norm.resize(D);
    w.q_a_proj.resize(QA * D);
    w.q_a_norm.resize(QA);
    w.q_b_proj.resize(H * HD * QA);
    w.kv_proj.resize(HD * D);
    w.kv_norm.resize(HD);
    w.o_a_proj.resize(OG * OR * (H * HD / OG));
    w.o_b_proj.resize(D * OG * OR);
    w.attn_sink.resize(H);
    w.topk_indices = {
        0.0f, -1.0f, -1.0f,
        0.0f, 1.0f, -1.0f,
        0.0f, 1.0f, 2.0f,
    };
    w.router.resize(E * D);
    w.router_bias.resize(E);
    w.expert_gate.assign(E, std::vector<__fp16>(I * D));
    w.expert_up.assign(E, std::vector<__fp16>(I * D));
    w.expert_down.assign(E, std::vector<__fp16>(D * I));
    w.shared_gate.resize(I * D);
    w.shared_up.resize(I * D);
    w.shared_down.resize(D * I);
    w.final_norm.resize(D);
    w.lm_head.resize(V * D);

    fill_half(w.embed, 0.12f, 0.1f);
    fill_float(w.hc_attn_fn, 0.025f, 0.2f);
    fill_float(w.hc_attn_base, 0.015f, 0.3f);
    fill_float(w.hc_ffn_fn, 0.022f, 0.4f);
    fill_float(w.hc_ffn_base, 0.014f, 0.5f);
    fill_float(w.hc_head_fn, 0.02f, 0.6f);
    fill_float(w.hc_head_base, 0.015f, 0.7f);
    for (size_t i = 0; i < D; ++i) {
        w.input_norm[i] = static_cast<__fp16>(0.86f + 0.011f * static_cast<float>(i));
        w.post_norm[i] = static_cast<__fp16>(0.91f - 0.007f * static_cast<float>(i));
        w.final_norm[i] = static_cast<__fp16>(0.82f + 0.014f * static_cast<float>(i));
    }
    for (size_t i = 0; i < QA; ++i) w.q_a_norm[i] = static_cast<__fp16>(0.88f + 0.009f * static_cast<float>(i));
    for (size_t i = 0; i < HD; ++i) w.kv_norm[i] = static_cast<__fp16>(0.9f - 0.012f * static_cast<float>(i));
    fill_half(w.q_a_proj, 0.055f, 0.8f);
    fill_half(w.q_b_proj, 0.05f, 0.9f);
    fill_half(w.kv_proj, 0.052f, 1.0f);
    fill_half(w.o_a_proj, 0.048f, 1.1f);
    fill_half(w.o_b_proj, 0.046f, 1.2f);
    fill_float(w.attn_sink, 0.025f, 1.3f);
    fill_half(w.router, 0.05f, 1.4f);
    fill_half(w.router_bias, 0.01f, 1.5f);
    for (size_t e = 0; e < E; ++e) {
        fill_half(w.expert_gate[e], 0.045f, 1.6f + static_cast<float>(e));
        fill_half(w.expert_up[e], 0.043f, 2.0f + static_cast<float>(e));
        fill_half(w.expert_down[e], 0.041f, 2.4f + static_cast<float>(e));
    }
    fill_half(w.shared_gate, 0.04f, 3.1f);
    fill_half(w.shared_up, 0.038f, 3.4f);
    fill_half(w.shared_down, 0.036f, 3.7f);
    fill_half(w.lm_head, 0.06f, 4.0f);
    return w;
}

size_t grouped_o_a(CactusGraph& graph, size_t attn_flat, size_t o_a_proj) {
    constexpr size_t in_per_group = H * HD / OG;
    std::vector<size_t> group_outputs;
    for (size_t g = 0; g < OG; ++g) {
        size_t xg = graph.slice(attn_flat, 1, g * in_per_group, in_per_group);
        size_t wg = graph.slice(o_a_proj, 0, g * OR, OR);
        group_outputs.push_back(graph.matmul(xg, wg, true, ComputeBackend::CPU));
    }
    return graph.cat(group_outputs, 1);
}

size_t build_and_run_graph(const HFWeights& w) {
    static CactusGraph graph;
    graph.hard_reset();

    size_t embed_id = graph.input({V, D}, Precision::FP16);
    size_t ids_id = graph.input({T}, Precision::FP32);
    size_t hc_attn_fn = graph.input({MIX, HC * D}, Precision::FP32);
    size_t hc_attn_base = graph.input({MIX}, Precision::FP32);
    size_t hc_attn_scale = graph.input({3}, Precision::FP32);
    size_t hc_ffn_fn = graph.input({MIX, HC * D}, Precision::FP32);
    size_t hc_ffn_base = graph.input({MIX}, Precision::FP32);
    size_t hc_ffn_scale = graph.input({3}, Precision::FP32);
    size_t hc_head_fn = graph.input({HC, HC * D}, Precision::FP32);
    size_t hc_head_base = graph.input({HC}, Precision::FP32);
    size_t hc_head_scale = graph.input({1}, Precision::FP32);
    size_t input_norm = graph.input({D}, Precision::FP16);
    size_t post_norm = graph.input({D}, Precision::FP16);
    size_t q_a_proj = graph.input({QA, D}, Precision::FP16);
    size_t q_a_norm = graph.input({QA}, Precision::FP16);
    size_t q_b_proj = graph.input({H * HD, QA}, Precision::FP16);
    size_t kv_proj = graph.input({HD, D}, Precision::FP16);
    size_t kv_norm = graph.input({HD}, Precision::FP16);
    size_t o_a_proj = graph.input({OG * OR, H * HD / OG}, Precision::FP16);
    size_t o_b_proj = graph.input({D, OG * OR}, Precision::FP16);
    size_t sink = graph.input({H}, Precision::FP32);
    size_t idx = graph.input({1, T, T}, Precision::FP32);
    size_t router = graph.input({E, D}, Precision::FP16);
    size_t router_bias = graph.input({E}, Precision::FP16);
    std::vector<size_t> eg, eu, ed;
    for (size_t e = 0; e < E; ++e) eg.push_back(graph.input({I, D}, Precision::FP16));
    for (size_t e = 0; e < E; ++e) eu.push_back(graph.input({I, D}, Precision::FP16));
    for (size_t e = 0; e < E; ++e) ed.push_back(graph.input({D, I}, Precision::FP16));
    size_t sg = graph.input({I, D}, Precision::FP16);
    size_t su = graph.input({I, D}, Precision::FP16);
    size_t sd = graph.input({D, I}, Precision::FP16);
    size_t final_norm = graph.input({D}, Precision::FP16);
    size_t lm_head = graph.input({V, D}, Precision::FP16);

    std::vector<float> token_ids = {1.0f, 3.0f, 5.0f};
    size_t hidden = graph.embedding(embed_id, ids_id);
    size_t one_stream = graph.reshape(hidden, {T, 1, D});
    size_t streams = graph.cat({one_stream, one_stream, one_stream, one_stream}, 1);

    size_t mix_a = graph.dsv4_hc_mix(streams, hc_attn_fn, hc_attn_base, hc_attn_scale, EPS, 20);
    size_t attn_in = graph.rms_norm(graph.dsv4_hc_collapse(streams, mix_a), input_norm, EPS);
    size_t q_residual = graph.rms_norm(graph.matmul(attn_in, q_a_proj, true, ComputeBackend::CPU), q_a_norm, EPS);
    size_t q = graph.matmul(q_residual, q_b_proj, true, ComputeBackend::CPU);
    q = graph.reshape(q, {1, T, H, HD});
    q = graph.dsv4_rms_norm(q, EPS);
    q = graph.dsv4_rope(q, RD, 10000.0f, 0, false, 1.0f, 65536, 32.0f, 1.0f, false);
    size_t kv = graph.rms_norm(graph.matmul(attn_in, kv_proj, true, ComputeBackend::CPU), kv_norm, EPS);
    kv = graph.reshape(kv, {1, T, 1, HD});
    kv = graph.dsv4_rope(kv, RD, 10000.0f, 0, false, 1.0f, 65536, 32.0f, 1.0f, false);
    kv = graph.reshape(kv, {1, T, HD});
    size_t attn = graph.dsv4_sparse_attention(q, kv, sink, idx, 1.0f / std::sqrt(static_cast<float>(HD)));
    attn = graph.dsv4_rope(attn, RD, 10000.0f, 0, false, 1.0f, 65536, 32.0f, 1.0f, true);
    attn = graph.reshape(attn, {T, H * HD});
    size_t grouped = grouped_o_a(graph, attn, o_a_proj);
    size_t attn_out = graph.matmul(grouped, o_b_proj, true, ComputeBackend::CPU);
    streams = graph.dsv4_hc_post(attn_out, streams, mix_a);

    size_t mix_f = graph.dsv4_hc_mix(streams, hc_ffn_fn, hc_ffn_base, hc_ffn_scale, EPS, 20);
    size_t ffn_in = graph.rms_norm(graph.dsv4_hc_collapse(streams, mix_f), post_norm, EPS);
    size_t route = graph.dsv4_router_topk(ffn_in, router, router_bias, E, K, 1.0f, 1e-20f);
    size_t routed = graph.dsv4_moe_layer(ffn_in, route, eg, eu, ed, E, K, SWIGLU_LIMIT);
    size_t shared = graph.dsv4_shared_expert(ffn_in, sg, su, sd, SWIGLU_LIMIT);
    size_t ffn_out = graph.add(routed, shared);
    streams = graph.dsv4_hc_post(ffn_out, streams, mix_f);

    size_t collapsed = graph.dsv4_hc_head(streams, hc_head_fn, hc_head_base, hc_head_scale, EPS);
    size_t normed = graph.rms_norm(collapsed, final_norm, EPS);
    size_t logits = graph.matmul(normed, lm_head, true, ComputeBackend::CPU);

    graph.set_input(embed_id, w.embed.data(), Precision::FP16);
    graph.set_input(ids_id, token_ids.data(), Precision::FP32);
    graph.set_input(hc_attn_fn, w.hc_attn_fn.data(), Precision::FP32);
    graph.set_input(hc_attn_base, w.hc_attn_base.data(), Precision::FP32);
    graph.set_input(hc_attn_scale, w.hc_attn_scale.data(), Precision::FP32);
    graph.set_input(hc_ffn_fn, w.hc_ffn_fn.data(), Precision::FP32);
    graph.set_input(hc_ffn_base, w.hc_ffn_base.data(), Precision::FP32);
    graph.set_input(hc_ffn_scale, w.hc_ffn_scale.data(), Precision::FP32);
    graph.set_input(hc_head_fn, w.hc_head_fn.data(), Precision::FP32);
    graph.set_input(hc_head_base, w.hc_head_base.data(), Precision::FP32);
    graph.set_input(hc_head_scale, w.hc_head_scale.data(), Precision::FP32);
    graph.set_input(input_norm, w.input_norm.data(), Precision::FP16);
    graph.set_input(post_norm, w.post_norm.data(), Precision::FP16);
    graph.set_input(q_a_proj, w.q_a_proj.data(), Precision::FP16);
    graph.set_input(q_a_norm, w.q_a_norm.data(), Precision::FP16);
    graph.set_input(q_b_proj, w.q_b_proj.data(), Precision::FP16);
    graph.set_input(kv_proj, w.kv_proj.data(), Precision::FP16);
    graph.set_input(kv_norm, w.kv_norm.data(), Precision::FP16);
    graph.set_input(o_a_proj, w.o_a_proj.data(), Precision::FP16);
    graph.set_input(o_b_proj, w.o_b_proj.data(), Precision::FP16);
    graph.set_input(sink, w.attn_sink.data(), Precision::FP32);
    graph.set_input(idx, w.topk_indices.data(), Precision::FP32);
    graph.set_input(router, w.router.data(), Precision::FP16);
    graph.set_input(router_bias, w.router_bias.data(), Precision::FP16);
    for (size_t e = 0; e < E; ++e) graph.set_input(eg[e], w.expert_gate[e].data(), Precision::FP16);
    for (size_t e = 0; e < E; ++e) graph.set_input(eu[e], w.expert_up[e].data(), Precision::FP16);
    for (size_t e = 0; e < E; ++e) graph.set_input(ed[e], w.expert_down[e].data(), Precision::FP16);
    graph.set_input(sg, w.shared_gate.data(), Precision::FP16);
    graph.set_input(su, w.shared_up.data(), Precision::FP16);
    graph.set_input(sd, w.shared_down.data(), Precision::FP16);
    graph.set_input(final_norm, w.final_norm.data(), Precision::FP16);
    graph.set_input(lm_head, w.lm_head.data(), Precision::FP16);
    graph.execute();

    const auto* out = static_cast<const __fp16*>(graph.get_output(logits));
    std::ostringstream os;
    os << "CACTUS_DSV4_HF_LOGITS [";
    for (size_t i = 0; i < T * V; ++i) {
        if (i) os << ",";
        os << static_cast<float>(out[i]);
    }
    os << "]\n";
    std::cout << os.str();
    return logits;
}

bool test_hf_parity_runner_smoke() {
    HFWeights w = make_hf_weights();
    build_and_run_graph(w);
    return true;
}

} // namespace

int main() {
    TestRunner runner("DeepSeek V4 HF E2E Runner");
    runner.run_test("cactus logits dump", test_hf_parity_runner_smoke());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
