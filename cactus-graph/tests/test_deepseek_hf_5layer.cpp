#include "test_utils.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

using namespace TestUtils;

namespace {

constexpr size_t T = 1060;
constexpr size_t V = 32;
constexpr size_t D = 8;
constexpr size_t HC = 4;
constexpr size_t MIX = 24;
constexpr size_t H = 2;
constexpr size_t HD = 4;
constexpr size_t RD = 2;
constexpr size_t QA = 6;
constexpr size_t OG = 2;
constexpr size_t OR = 3;
constexpr size_t E = 16;
constexpr size_t I = 5;
constexpr size_t K = 4;
constexpr size_t IH = 2;
constexpr size_t ID = 4;
constexpr size_t NLAYERS = 5;
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

std::vector<float> token_ids() {
    std::vector<float> ids(T);
    for (size_t t = 0; t < T; ++t) ids[t] = static_cast<float>((t * 7 + 3) % V);
    const char* override = std::getenv("CACTUS_DSV4_5L_TOKENS");
    if (override != nullptr) {
        std::stringstream ss(override);
        std::string item;
        size_t i = 0;
        while (std::getline(ss, item, ',')) {
            if (i >= T) throw std::runtime_error("CACTUS_DSV4_5L_TOKENS has too many entries");
            int value = std::stoi(item);
            if (value < 0 || value >= static_cast<int>(V)) {
                throw std::runtime_error("CACTUS_DSV4_5L_TOKENS contains out-of-range token id");
            }
            ids[i++] = static_cast<float>(value);
        }
        if (i != T) throw std::runtime_error("CACTUS_DSV4_5L_TOKENS must contain exactly 1060 entries");
    }
    return ids;
}

std::vector<float> position_ids() {
    std::vector<float> pos(T);
    for (size_t t = 0; t < T; ++t) pos[t] = static_cast<float>(t);
    return pos;
}

std::vector<float> sliding_indices(size_t extra = 0, size_t ratio = 1, bool causal_extra = false) {
    std::vector<float> idx(T * (T + extra), -1.0f);
    for (size_t t = 0; t < T; ++t) {
        for (size_t j = 0; j <= t; ++j) idx[t * (T + extra) + j] = static_cast<float>(j);
        for (size_t e = 0; e < extra; ++e) {
            if (!causal_extra || e < (t + 1) / ratio) {
                idx[t * (T + extra) + T + e] = static_cast<float>(T + e);
            }
        }
    }
    return idx;
}

struct LayerWeights {
    std::vector<float> hc_attn_fn, hc_attn_base, hc_attn_scale;
    std::vector<float> hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    std::vector<__fp16> input_norm, post_norm;
    std::vector<__fp16> q_a_proj, q_a_norm, q_b_proj;
    std::vector<__fp16> kv_proj, kv_norm, o_a_proj, o_b_proj;
    std::vector<float> attn_sink;
    std::vector<__fp16> hca_kv, hca_gate, hca_norm;
    std::vector<float> hca_pos;
    std::vector<__fp16> csa_kv, csa_gate, csa_norm;
    std::vector<float> csa_pos;
    std::vector<__fp16> idx_kv, idx_gate, idx_norm, idx_q_b, idx_weights;
    std::vector<float> idx_pos;
    std::vector<__fp16> router, router_bias, tid2eid;
    std::vector<std::vector<__fp16>> expert_gate, expert_up, expert_down;
    std::vector<__fp16> shared_gate, shared_up, shared_down;
};

struct Weights {
    std::vector<__fp16> embed, final_norm, lm_head;
    std::vector<float> hc_head_fn, hc_head_base, hc_head_scale;
    std::vector<LayerWeights> layers;
};

LayerWeights make_layer(size_t layer) {
    LayerWeights w;
    const float p = 0.35f + static_cast<float>(layer) * 4.0f;
    w.hc_attn_fn.resize(MIX * HC * D);
    w.hc_attn_base.resize(MIX);
    w.hc_attn_scale = {0.55f + 0.03f * layer, -0.28f + 0.02f * layer, 0.17f};
    w.hc_ffn_fn.resize(MIX * HC * D);
    w.hc_ffn_base.resize(MIX);
    w.hc_ffn_scale = {-0.31f, 0.42f + 0.02f * layer, 0.19f};
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
    w.hca_kv.resize(HD * D);
    w.hca_gate.resize(HD * D);
    w.hca_norm.resize(HD);
    w.hca_pos.resize(128 * HD);
    w.csa_kv.resize(2 * HD * D);
    w.csa_gate.resize(2 * HD * D);
    w.csa_norm.resize(HD);
    w.csa_pos.resize(4 * 2 * HD);
    w.idx_kv.resize(2 * ID * D);
    w.idx_gate.resize(2 * ID * D);
    w.idx_norm.resize(ID);
    w.idx_q_b.resize(IH * ID * QA);
    w.idx_weights.resize(IH * D);
    w.idx_pos.resize(4 * 2 * ID);
    w.router.resize(E * D);
    w.router_bias.resize(E);
    w.tid2eid.resize(V * K);
    w.expert_gate.assign(E, std::vector<__fp16>(I * D));
    w.expert_up.assign(E, std::vector<__fp16>(I * D));
    w.expert_down.assign(E, std::vector<__fp16>(D * I));
    w.shared_gate.resize(I * D);
    w.shared_up.resize(I * D);
    w.shared_down.resize(D * I);

    fill_float(w.hc_attn_fn, 0.013f, p + 0.1f);
    fill_float(w.hc_attn_base, 0.008f, p + 0.2f);
    fill_float(w.hc_ffn_fn, 0.012f, p + 0.3f);
    fill_float(w.hc_ffn_base, 0.007f, p + 0.4f);
    for (size_t i = 0; i < D; ++i) {
        w.input_norm[i] = static_cast<__fp16>(0.9f + 0.003f * static_cast<float>((i + layer) % 5));
        w.post_norm[i] = static_cast<__fp16>(0.88f + 0.004f * static_cast<float>((i + 2 * layer) % 5));
    }
    for (size_t i = 0; i < QA; ++i) w.q_a_norm[i] = static_cast<__fp16>(0.91f + 0.002f * static_cast<float>((i + layer) % 4));
    for (size_t i = 0; i < HD; ++i) {
        w.kv_norm[i] = static_cast<__fp16>(0.89f + 0.004f * static_cast<float>((i + layer) % 3));
        w.hca_norm[i] = static_cast<__fp16>(0.87f + 0.004f * static_cast<float>((i + layer) % 3));
        w.csa_norm[i] = static_cast<__fp16>(0.86f + 0.004f * static_cast<float>((i + layer) % 3));
    }
    for (size_t i = 0; i < ID; ++i) w.idx_norm[i] = static_cast<__fp16>(0.85f + 0.003f * static_cast<float>((i + layer) % 4));
    fill_half(w.q_a_proj, 0.024f, p + 0.5f);
    fill_half(w.q_b_proj, 0.022f, p + 0.6f);
    fill_half(w.kv_proj, 0.023f, p + 0.7f);
    fill_half(w.o_a_proj, 0.021f, p + 0.8f);
    fill_half(w.o_b_proj, 0.020f, p + 0.9f);
    fill_float(w.attn_sink, 0.012f, p + 1.0f);
    fill_half(w.hca_kv, 0.018f, p + 1.1f);
    fill_half(w.hca_gate, 0.017f, p + 1.2f);
    fill_float(w.hca_pos, 0.010f, p + 1.3f);
    fill_half(w.csa_kv, 0.018f, p + 1.4f);
    fill_half(w.csa_gate, 0.017f, p + 1.5f);
    fill_float(w.csa_pos, 0.010f, p + 1.6f);
    fill_half(w.idx_kv, 0.019f, p + 1.7f);
    fill_half(w.idx_gate, 0.018f, p + 1.8f);
    fill_half(w.idx_q_b, 0.020f, p + 1.9f);
    fill_half(w.idx_weights, 0.021f, p + 2.0f);
    fill_float(w.idx_pos, 0.010f, p + 2.1f);
    fill_half(w.router, 0.021f, p + 2.2f);
    fill_half(w.router_bias, 0.006f, p + 2.3f);
    for (size_t tok = 0; tok < V; ++tok) {
        for (size_t k = 0; k < K; ++k) w.tid2eid[tok * K + k] = static_cast<__fp16>((tok + k * 3 + layer) % E);
    }
    for (size_t e = 0; e < E; ++e) {
        fill_half(w.expert_gate[e], 0.019f, p + 2.4f + 0.13f * static_cast<float>(e));
        fill_half(w.expert_up[e], 0.018f, p + 2.8f + 0.13f * static_cast<float>(e));
        fill_half(w.expert_down[e], 0.017f, p + 3.2f + 0.13f * static_cast<float>(e));
    }
    fill_half(w.shared_gate, 0.018f, p + 3.6f);
    fill_half(w.shared_up, 0.017f, p + 3.7f);
    fill_half(w.shared_down, 0.016f, p + 3.8f);
    return w;
}

Weights make_weights() {
    Weights w;
    w.embed.resize(V * D);
    w.final_norm.resize(D);
    w.lm_head.resize(V * D);
    w.hc_head_fn.resize(HC * HC * D);
    w.hc_head_base.resize(HC);
    w.hc_head_scale = {0.5f};
    fill_half(w.embed, 0.055f, 0.1f);
    fill_half(w.lm_head, 0.030f, 0.2f);
    fill_float(w.hc_head_fn, 0.012f, 0.3f);
    fill_float(w.hc_head_base, 0.006f, 0.4f);
    for (size_t i = 0; i < D; ++i) w.final_norm[i] = static_cast<__fp16>(0.92f + 0.003f * static_cast<float>(i));
    for (size_t l = 0; l < NLAYERS; ++l) w.layers.push_back(make_layer(l));
    return w;
}

size_t grouped_o_a(CactusGraph& graph, size_t attn_flat, size_t o_a_proj) {
    constexpr size_t in_per_group = H * HD / OG;
    std::vector<size_t> outs;
    for (size_t g = 0; g < OG; ++g) {
        size_t xg = graph.slice(attn_flat, 1, g * in_per_group, in_per_group);
        size_t wg = graph.slice(o_a_proj, 0, g * OR, OR);
        outs.push_back(graph.matmul(xg, wg, true, ComputeBackend::CPU));
    }
    return graph.cat(outs, 1);
}

struct LayerInputs {
    size_t hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    size_t input_norm, post_norm, q_a_proj, q_a_norm, q_b_proj, kv_proj, kv_norm, o_a_proj, o_b_proj, sink;
    size_t hca_kv, hca_gate, hca_norm, hca_pos, csa_kv, csa_gate, csa_norm, csa_pos;
    size_t idx_kv, idx_gate, idx_norm, idx_q_b, idx_weights, idx_pos;
    size_t router, router_bias, tid2eid, static_idx;
    std::vector<size_t> eg, eu, ed;
    size_t sg, su, sd;
};

LayerInputs add_layer_inputs(CactusGraph& graph, size_t layer_type) {
    LayerInputs in;
    in.hc_attn_fn = graph.input({MIX, HC * D}, Precision::FP32);
    in.hc_attn_base = graph.input({MIX}, Precision::FP32);
    in.hc_attn_scale = graph.input({3}, Precision::FP32);
    in.hc_ffn_fn = graph.input({MIX, HC * D}, Precision::FP32);
    in.hc_ffn_base = graph.input({MIX}, Precision::FP32);
    in.hc_ffn_scale = graph.input({3}, Precision::FP32);
    in.input_norm = graph.input({D}, Precision::FP16);
    in.post_norm = graph.input({D}, Precision::FP16);
    in.q_a_proj = graph.input({QA, D}, Precision::FP16);
    in.q_a_norm = graph.input({QA}, Precision::FP16);
    in.q_b_proj = graph.input({H * HD, QA}, Precision::FP16);
    in.kv_proj = graph.input({HD, D}, Precision::FP16);
    in.kv_norm = graph.input({HD}, Precision::FP16);
    in.o_a_proj = graph.input({OG * OR, H * HD / OG}, Precision::FP16);
    in.o_b_proj = graph.input({D, OG * OR}, Precision::FP16);
    in.sink = graph.input({H}, Precision::FP32);
    in.hca_kv = graph.input({HD, D}, Precision::FP16);
    in.hca_gate = graph.input({HD, D}, Precision::FP16);
    in.hca_norm = graph.input({HD}, Precision::FP16);
    in.hca_pos = graph.input({128, HD}, Precision::FP32);
    in.csa_kv = graph.input({2 * HD, D}, Precision::FP16);
    in.csa_gate = graph.input({2 * HD, D}, Precision::FP16);
    in.csa_norm = graph.input({HD}, Precision::FP16);
    in.csa_pos = graph.input({4, 2 * HD}, Precision::FP32);
    in.idx_kv = graph.input({2 * ID, D}, Precision::FP16);
    in.idx_gate = graph.input({2 * ID, D}, Precision::FP16);
    in.idx_norm = graph.input({ID}, Precision::FP16);
    in.idx_q_b = graph.input({IH * ID, QA}, Precision::FP16);
    in.idx_weights = graph.input({IH, D}, Precision::FP16);
    in.idx_pos = graph.input({4, 2 * ID}, Precision::FP32);
    in.router = graph.input({E, D}, Precision::FP16);
    in.router_bias = graph.input({E}, Precision::FP16);
    in.tid2eid = graph.input({V, K}, Precision::FP16);
    size_t idx_width = T;
    if (layer_type == 128) idx_width = T + T / 128;
    in.static_idx = graph.input({1, T, idx_width}, Precision::FP32);
    for (size_t e = 0; e < E; ++e) in.eg.push_back(graph.input({I, D}, Precision::FP16));
    for (size_t e = 0; e < E; ++e) in.eu.push_back(graph.input({I, D}, Precision::FP16));
    for (size_t e = 0; e < E; ++e) in.ed.push_back(graph.input({D, I}, Precision::FP16));
    in.sg = graph.input({I, D}, Precision::FP16);
    in.su = graph.input({I, D}, Precision::FP16);
    in.sd = graph.input({D, I}, Precision::FP16);
    return in;
}

size_t csa_compressed(CactusGraph& graph, size_t attn_in, const LayerInputs& in) {
    size_t kv = graph.matmul(attn_in, in.csa_kv, true, ComputeBackend::CPU);
    size_t gate = graph.matmul(attn_in, in.csa_gate, true, ComputeBackend::CPU);
    kv = graph.reshape(kv, {1, T, 2 * HD});
    gate = graph.reshape(gate, {1, T, 2 * HD});
    size_t comp = graph.dsv4_compress_csa(kv, gate, in.csa_pos, in.csa_norm, EPS, 4);
    comp = graph.reshape(comp, {1, T / 4, 1, HD});
    comp = graph.dsv4_rope(comp, RD, 160000.0f, 0, true, 16.0f, 65536, 32.0f, 1.0f, false, 4);
    return graph.reshape(comp, {1, T / 4, HD});
}

size_t hca_compressed(CactusGraph& graph, size_t attn_in, const LayerInputs& in) {
    size_t kv = graph.matmul(attn_in, in.hca_kv, true, ComputeBackend::CPU);
    size_t gate = graph.matmul(attn_in, in.hca_gate, true, ComputeBackend::CPU);
    kv = graph.reshape(kv, {1, T, HD});
    gate = graph.reshape(gate, {1, T, HD});
    size_t comp = graph.dsv4_compress_hca(kv, gate, in.hca_pos, in.hca_norm, EPS, 128);
    comp = graph.reshape(comp, {1, T / 128, 1, HD});
    comp = graph.dsv4_rope(comp, RD, 160000.0f, 0, true, 16.0f, 65536, 32.0f, 1.0f, false, 128);
    return graph.reshape(comp, {1, T / 128, HD});
}

size_t csa_index(CactusGraph& graph, size_t attn_in, size_t q_residual, size_t pos_id, const LayerInputs& in) {
    size_t kv = graph.matmul(attn_in, in.idx_kv, true, ComputeBackend::CPU);
    size_t gate = graph.matmul(attn_in, in.idx_gate, true, ComputeBackend::CPU);
    kv = graph.reshape(kv, {1, T, 2 * ID});
    gate = graph.reshape(gate, {1, T, 2 * ID});
    size_t comp = graph.dsv4_compress_csa(kv, gate, in.idx_pos, in.idx_norm, EPS, 4);
    comp = graph.reshape(comp, {1, T / 4, 1, ID});
    comp = graph.dsv4_rope(comp, RD, 160000.0f, 0, true, 16.0f, 65536, 32.0f, 1.0f, false, 4);
    comp = graph.reshape(comp, {1, T / 4, ID});
    size_t q = graph.matmul(q_residual, in.idx_q_b, true, ComputeBackend::CPU);
    q = graph.reshape(q, {1, T, IH, ID});
    q = graph.dsv4_rope(q, RD, 160000.0f, 0, true, 16.0f, 65536, 32.0f, 1.0f, false);
    size_t weights = graph.matmul(attn_in, in.idx_weights, true, ComputeBackend::CPU);
    weights = graph.reshape(weights, {1, T, IH});
    return graph.dsv4_indexer_topk(q, comp, weights, pos_id, K, 4, T, 1.0f / std::sqrt(static_cast<float>(ID * IH)));
}

size_t add_layer(CactusGraph& graph, size_t streams, size_t ids, size_t pos, const LayerInputs& in,
                 size_t layer_type, bool hash_moe) {
    const bool compressed = layer_type != 0;
    size_t mix_a = graph.dsv4_hc_mix(streams, in.hc_attn_fn, in.hc_attn_base, in.hc_attn_scale, EPS, 20);
    size_t attn_in = graph.rms_norm(graph.dsv4_hc_collapse(streams, mix_a), in.input_norm, EPS);
    size_t q_residual = graph.rms_norm(graph.matmul(attn_in, in.q_a_proj, true, ComputeBackend::CPU), in.q_a_norm, EPS);
    size_t q = graph.matmul(q_residual, in.q_b_proj, true, ComputeBackend::CPU);
    q = graph.reshape(q, {1, T, H, HD});
    q = graph.dsv4_rms_norm(q, EPS);
    q = graph.dsv4_rope(q, RD, compressed ? 160000.0f : 10000.0f, 0, compressed, compressed ? 16.0f : 1.0f,
                        65536, 32.0f, 1.0f, false);
    size_t kv = graph.rms_norm(graph.matmul(attn_in, in.kv_proj, true, ComputeBackend::CPU), in.kv_norm, EPS);
    kv = graph.reshape(kv, {1, T, 1, HD});
    kv = graph.dsv4_rope(kv, RD, compressed ? 160000.0f : 10000.0f, 0, compressed, compressed ? 16.0f : 1.0f,
                         65536, 32.0f, 1.0f, false);
    kv = graph.reshape(kv, {1, T, HD});
    size_t attn_idx = in.static_idx;
    if (layer_type == 4) {
        size_t comp = csa_compressed(graph, attn_in, in);
        kv = graph.cat({kv, comp}, 1);
        size_t dyn_idx = csa_index(graph, attn_in, q_residual, pos, in);
        attn_idx = graph.cat({in.static_idx, dyn_idx}, 2);
    } else if (layer_type == 128) {
        size_t comp = hca_compressed(graph, attn_in, in);
        kv = graph.cat({kv, comp}, 1);
    }
    size_t attn = graph.dsv4_sparse_attention(q, kv, in.sink, attn_idx, 1.0f / std::sqrt(static_cast<float>(HD)));
    attn = graph.dsv4_rope(attn, RD, compressed ? 160000.0f : 10000.0f, 0, compressed, compressed ? 16.0f : 1.0f,
                           65536, 32.0f, 1.0f, true);
    attn = graph.reshape(attn, {T, H * HD});
    size_t grouped = grouped_o_a(graph, attn, in.o_a_proj);
    size_t attn_out = graph.matmul(grouped, in.o_b_proj, true, ComputeBackend::CPU);
    streams = graph.dsv4_hc_post(attn_out, streams, mix_a);

    size_t mix_f = graph.dsv4_hc_mix(streams, in.hc_ffn_fn, in.hc_ffn_base, in.hc_ffn_scale, EPS, 20);
    size_t ffn_in = graph.rms_norm(graph.dsv4_hc_collapse(streams, mix_f), in.post_norm, EPS);
    size_t route = hash_moe
        ? graph.dsv4_hash_router(ffn_in, ids, in.router, in.tid2eid, E, K, 1.0f, 1e-20f)
        : graph.dsv4_router_topk(ffn_in, in.router, in.router_bias, E, K, 1.0f, 1e-20f);
    size_t routed = graph.dsv4_moe_layer(ffn_in, route, in.eg, in.eu, in.ed, E, K, SWIGLU_LIMIT);
    size_t shared = graph.dsv4_shared_expert(ffn_in, in.sg, in.su, in.sd, SWIGLU_LIMIT);
    size_t ffn_out = graph.add(routed, shared);
    return graph.dsv4_hc_post(ffn_out, streams, mix_f);
}

void set_layer_inputs(CactusGraph& graph, const LayerInputs& in, const LayerWeights& w, const std::vector<float>& static_idx) {
    graph.set_input(in.hc_attn_fn, w.hc_attn_fn.data(), Precision::FP32);
    graph.set_input(in.hc_attn_base, w.hc_attn_base.data(), Precision::FP32);
    graph.set_input(in.hc_attn_scale, w.hc_attn_scale.data(), Precision::FP32);
    graph.set_input(in.hc_ffn_fn, w.hc_ffn_fn.data(), Precision::FP32);
    graph.set_input(in.hc_ffn_base, w.hc_ffn_base.data(), Precision::FP32);
    graph.set_input(in.hc_ffn_scale, w.hc_ffn_scale.data(), Precision::FP32);
    graph.set_input(in.input_norm, w.input_norm.data(), Precision::FP16);
    graph.set_input(in.post_norm, w.post_norm.data(), Precision::FP16);
    graph.set_input(in.q_a_proj, w.q_a_proj.data(), Precision::FP16);
    graph.set_input(in.q_a_norm, w.q_a_norm.data(), Precision::FP16);
    graph.set_input(in.q_b_proj, w.q_b_proj.data(), Precision::FP16);
    graph.set_input(in.kv_proj, w.kv_proj.data(), Precision::FP16);
    graph.set_input(in.kv_norm, w.kv_norm.data(), Precision::FP16);
    graph.set_input(in.o_a_proj, w.o_a_proj.data(), Precision::FP16);
    graph.set_input(in.o_b_proj, w.o_b_proj.data(), Precision::FP16);
    graph.set_input(in.sink, w.attn_sink.data(), Precision::FP32);
    graph.set_input(in.hca_kv, w.hca_kv.data(), Precision::FP16);
    graph.set_input(in.hca_gate, w.hca_gate.data(), Precision::FP16);
    graph.set_input(in.hca_norm, w.hca_norm.data(), Precision::FP16);
    graph.set_input(in.hca_pos, w.hca_pos.data(), Precision::FP32);
    graph.set_input(in.csa_kv, w.csa_kv.data(), Precision::FP16);
    graph.set_input(in.csa_gate, w.csa_gate.data(), Precision::FP16);
    graph.set_input(in.csa_norm, w.csa_norm.data(), Precision::FP16);
    graph.set_input(in.csa_pos, w.csa_pos.data(), Precision::FP32);
    graph.set_input(in.idx_kv, w.idx_kv.data(), Precision::FP16);
    graph.set_input(in.idx_gate, w.idx_gate.data(), Precision::FP16);
    graph.set_input(in.idx_norm, w.idx_norm.data(), Precision::FP16);
    graph.set_input(in.idx_q_b, w.idx_q_b.data(), Precision::FP16);
    graph.set_input(in.idx_weights, w.idx_weights.data(), Precision::FP16);
    graph.set_input(in.idx_pos, w.idx_pos.data(), Precision::FP32);
    graph.set_input(in.router, w.router.data(), Precision::FP16);
    graph.set_input(in.router_bias, w.router_bias.data(), Precision::FP16);
    graph.set_input(in.tid2eid, w.tid2eid.data(), Precision::FP16);
    graph.set_input(in.static_idx, static_idx.data(), Precision::FP32);
    for (size_t e = 0; e < E; ++e) graph.set_input(in.eg[e], w.expert_gate[e].data(), Precision::FP16);
    for (size_t e = 0; e < E; ++e) graph.set_input(in.eu[e], w.expert_up[e].data(), Precision::FP16);
    for (size_t e = 0; e < E; ++e) graph.set_input(in.ed[e], w.expert_down[e].data(), Precision::FP16);
    graph.set_input(in.sg, w.shared_gate.data(), Precision::FP16);
    graph.set_input(in.su, w.shared_up.data(), Precision::FP16);
    graph.set_input(in.sd, w.shared_down.data(), Precision::FP16);
}

bool run_five_layer() {
    Weights w = make_weights();
    CactusGraph graph;
    size_t embed = graph.input({V, D}, Precision::FP16);
    size_t ids = graph.input({T}, Precision::FP32);
    size_t pos = graph.input({1, T}, Precision::FP32);
    size_t hc_head_fn = graph.input({HC, HC * D}, Precision::FP32);
    size_t hc_head_base = graph.input({HC}, Precision::FP32);
    size_t hc_head_scale = graph.input({1}, Precision::FP32);
    size_t final_norm = graph.input({D}, Precision::FP16);
    size_t lm_head = graph.input({V, D}, Precision::FP16);

    const std::vector<size_t> layer_types = {0, 4, 128, 128, 4};
    std::vector<LayerInputs> layer_inputs;
    for (size_t l = 0; l < NLAYERS; ++l) layer_inputs.push_back(add_layer_inputs(graph, layer_types[l]));

    size_t hidden = graph.embedding(embed, ids);
    size_t one_stream = graph.reshape(hidden, {T, 1, D});
    size_t streams = graph.cat({one_stream, one_stream, one_stream, one_stream}, 1);
    for (size_t l = 0; l < NLAYERS; ++l) {
        streams = add_layer(graph, streams, ids, pos, layer_inputs[l], layer_types[l], l < 2);
    }
    size_t collapsed = graph.dsv4_hc_head(streams, hc_head_fn, hc_head_base, hc_head_scale, EPS);
    size_t normed = graph.rms_norm(collapsed, final_norm, EPS);
    size_t logits = graph.matmul(normed, lm_head, true, ComputeBackend::CPU);

    auto ids_v = token_ids();
    auto pos_v = position_ids();
    graph.set_input(embed, w.embed.data(), Precision::FP16);
    graph.set_input(ids, ids_v.data(), Precision::FP32);
    graph.set_input(pos, pos_v.data(), Precision::FP32);
    graph.set_input(hc_head_fn, w.hc_head_fn.data(), Precision::FP32);
    graph.set_input(hc_head_base, w.hc_head_base.data(), Precision::FP32);
    graph.set_input(hc_head_scale, w.hc_head_scale.data(), Precision::FP32);
    graph.set_input(final_norm, w.final_norm.data(), Precision::FP16);
    graph.set_input(lm_head, w.lm_head.data(), Precision::FP16);
    for (size_t l = 0; l < NLAYERS; ++l) {
        std::vector<float> static_idx;
        if (layer_types[l] == 0 || layer_types[l] == 4) {
            static_idx = sliding_indices();
        } else {
            static_idx = sliding_indices(T / 128, 128, true);
        }
        set_layer_inputs(graph, layer_inputs[l], w.layers[l], static_idx);
    }

    graph.execute();
    if (std::getenv("CACTUS_DSV4_DUMP_5L_LOGITS") != nullptr) {
        const auto* out = static_cast<const __fp16*>(graph.get_output(logits));
        std::ostringstream os;
        os << "CACTUS_DSV4_HF_5L_LOGITS [";
        for (size_t i = 0; i < T * V; ++i) {
            if (i) os << ",";
            os << static_cast<float>(out[i]);
        }
        os << "]\n";
        std::cout << os.str();
    }
    return true;
}

} // namespace

int main() {
    TestRunner runner("DeepSeek V4 HF 5-layer Runner");
    runner.run_test("cactus 5-layer logits dump", run_five_layer());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
