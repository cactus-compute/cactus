#pragma once

#include "../../graph/graph.h"
#include <cstdint>
#include <memory>
#include <vector>

struct Gemma4MLXState {
    struct LayerKV {
        std::vector<uint16_t> k;
        std::vector<uint16_t> v;
        size_t cached_len = 0;
        size_t kv_heads   = 0;
        size_t head_dim   = 0;
    };
    std::vector<LayerKV> layer_kv;
    size_t cached_seq_len = 0;

    void init(uint32_t num_layers) { layer_kv.resize(num_layers); }
    void reset() {
        for (auto& kv : layer_kv) { kv.k.clear(); kv.v.clear(); kv.cached_len = 0; }
        cached_seq_len = 0;
    }
};

struct Gemma4MLXLayerNodes {
    size_t q_weight = 0, k_weight = 0, v_weight = 0, o_weight = 0;
    size_t q_norm = 0, k_norm = 0;
    size_t input_ln = 0, post_attn_ln = 0;
    size_t gate_weight = 0, up_weight = 0, down_weight = 0;
    size_t pre_ffn_ln = 0, post_ffn_ln = 0;
    size_t pli_gate = 0, pli_proj = 0, pli_norm = 0;
    size_t layer_scalar = 0;
    bool   is_global = false;
    size_t num_heads = 0, kv_heads = 0, head_dim = 0;
    size_t rot_dim = 0;
    float  rope_freq = 10000.f;
    size_t window = 0;
    int32_t share_src = -1;
};

struct Gemma4MLXForwardParams {
    CactusGraph*gb = nullptr;
    std::vector<Gemma4MLXLayerNodes> layers;
    size_t per_layer_model_proj_node = 0;
    size_t per_layer_proj_norm_node = 0;
    uint32_t num_layers = 0;
    uint32_t hidden_dim = 0;
    uint32_t pli_dim = 0;
    bool has_pli = false;
    float norm_eps = 1e-6f;
    float attention_scale = 1.0f;
};

MLXFusedFn make_gemma4_full_forward_fn(
    Gemma4MLXState*                          state,
    std::shared_ptr<Gemma4MLXForwardParams>  params);
