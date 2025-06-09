#pragma once

#include "llama.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-hparams.h"
#include "llama-memory.h"
#include "llama-vocab.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_ubatch;
struct llama_model_loader;

// available models
enum llm_type {
    LLM_TYPE_UNKNOWN,
    LLM_TYPE_14M,
    LLM_TYPE_17M,
    LLM_TYPE_22M,
    LLM_TYPE_33M,
    LLM_TYPE_60M,
    LLM_TYPE_70M,
    LLM_TYPE_80M,
    LLM_TYPE_109M,
    LLM_TYPE_137M,
    LLM_TYPE_160M,
    LLM_TYPE_190M,
    LLM_TYPE_220M,
    LLM_TYPE_250M,
    LLM_TYPE_270M,
    LLM_TYPE_335M,
    LLM_TYPE_410M,
    LLM_TYPE_450M,
    LLM_TYPE_475M,
    LLM_TYPE_770M,
    LLM_TYPE_780M,
    LLM_TYPE_0_5B,
    LLM_TYPE_0_6B,
    LLM_TYPE_1B,
    LLM_TYPE_1_3B,
    LLM_TYPE_1_4B,
    LLM_TYPE_1_5B,
    LLM_TYPE_1_6B,
    LLM_TYPE_1_7B,
    LLM_TYPE_1_8B,
    LLM_TYPE_2B,
    LLM_TYPE_2_8B,
    LLM_TYPE_2_9B,
    LLM_TYPE_3B,
    LLM_TYPE_4B,
    LLM_TYPE_6B,
    LLM_TYPE_6_9B,
    LLM_TYPE_7B,
    LLM_TYPE_8B,
    LLM_TYPE_9B,
    LLM_TYPE_11B,
    LLM_TYPE_12B,
    LLM_TYPE_13B,
    LLM_TYPE_14B,
    LLM_TYPE_15B,
    LLM_TYPE_16B,
    LLM_TYPE_20B,
    LLM_TYPE_27B,
    LLM_TYPE_30B,
    LLM_TYPE_32B,
    LLM_TYPE_34B,
    LLM_TYPE_35B,
    LLM_TYPE_40B,
    LLM_TYPE_65B,
    LLM_TYPE_70B,
    LLM_TYPE_236B,
    LLM_TYPE_290B,
    LLM_TYPE_314B,
    LLM_TYPE_405B,
    LLM_TYPE_671B,
    LLM_TYPE_SMALL,
    LLM_TYPE_MEDIUM,
    LLM_TYPE_LARGE,
    LLM_TYPE_XL,
    LLM_TYPE_A1_7B,
    LLM_TYPE_A2_7B,
    LLM_TYPE_8x7B,
    LLM_TYPE_8x22B,
    LLM_TYPE_16x12B,
    LLM_TYPE_16x3_8B,
    LLM_TYPE_10B_128x3_66B,
    LLM_TYPE_57B_A14B,
    LLM_TYPE_17B_16E, // llama4 Scout
    LLM_TYPE_17B_128E, // llama4 Maverick
    LLM_TYPE_30B_A3B,
    LLM_TYPE_235B_A22B,
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type);

struct llama_layer_posnet {
    // resnet
    struct lm_ggml_tensor * norm1   = nullptr;
    struct lm_ggml_tensor * norm1_b = nullptr;

    struct lm_ggml_tensor * conv1   = nullptr;
    struct lm_ggml_tensor * conv1_b = nullptr;

    struct lm_ggml_tensor * norm2   = nullptr;
    struct lm_ggml_tensor * norm2_b = nullptr;

    struct lm_ggml_tensor * conv2   = nullptr;
    struct lm_ggml_tensor * conv2_b = nullptr;

    // attention
    struct lm_ggml_tensor * attn_norm   = nullptr;
    struct lm_ggml_tensor * attn_norm_b = nullptr;

    struct lm_ggml_tensor * attn_q   = nullptr;
    struct lm_ggml_tensor * attn_q_b = nullptr;

    struct lm_ggml_tensor * attn_k   = nullptr;
    struct lm_ggml_tensor * attn_k_b = nullptr;

    struct lm_ggml_tensor * attn_v   = nullptr;
    struct lm_ggml_tensor * attn_v_b = nullptr;

    struct lm_ggml_tensor * attn_o   = nullptr;
    struct lm_ggml_tensor * attn_o_b = nullptr;

    // normalize
    struct lm_ggml_tensor * norm   = nullptr;
    struct lm_ggml_tensor * norm_b = nullptr;
};

struct llama_layer_convnext {
    struct lm_ggml_tensor * dw   = nullptr;
    struct lm_ggml_tensor * dw_b = nullptr;

    struct lm_ggml_tensor * norm   = nullptr;
    struct lm_ggml_tensor * norm_b = nullptr;

    struct lm_ggml_tensor * pw1   = nullptr;
    struct lm_ggml_tensor * pw1_b = nullptr;

    struct lm_ggml_tensor * pw2   = nullptr;
    struct lm_ggml_tensor * pw2_b = nullptr;

    struct lm_ggml_tensor * gamma = nullptr;
};

struct llama_layer {
    // normalization
    struct lm_ggml_tensor * attn_norm       = nullptr;
    struct lm_ggml_tensor * attn_norm_b     = nullptr;
    struct lm_ggml_tensor * attn_norm_2     = nullptr;
    struct lm_ggml_tensor * attn_norm_2_b   = nullptr;
    struct lm_ggml_tensor * attn_q_norm     = nullptr;
    struct lm_ggml_tensor * attn_q_norm_b   = nullptr;
    struct lm_ggml_tensor * attn_k_norm     = nullptr;
    struct lm_ggml_tensor * attn_k_norm_b   = nullptr;
    struct lm_ggml_tensor * attn_out_norm   = nullptr;
    struct lm_ggml_tensor * attn_out_norm_b = nullptr;
    struct lm_ggml_tensor * attn_q_a_norm   = nullptr;
    struct lm_ggml_tensor * attn_kv_a_norm  = nullptr;
    struct lm_ggml_tensor * attn_sub_norm   = nullptr;
    struct lm_ggml_tensor * attn_post_norm  = nullptr;
    struct lm_ggml_tensor * ffn_sub_norm    = nullptr;
    struct lm_ggml_tensor * attn_norm_cross = nullptr;
    struct lm_ggml_tensor * attn_norm_enc   = nullptr;

    // attention
    struct lm_ggml_tensor * wq        = nullptr;
    struct lm_ggml_tensor * wk        = nullptr;
    struct lm_ggml_tensor * wv        = nullptr;
    struct lm_ggml_tensor * wo        = nullptr;
    struct lm_ggml_tensor * wqkv      = nullptr;
    struct lm_ggml_tensor * wq_a      = nullptr;
    struct lm_ggml_tensor * wq_b      = nullptr;
    struct lm_ggml_tensor * wkv_a_mqa = nullptr;
    struct lm_ggml_tensor * wkv_b     = nullptr;
    struct lm_ggml_tensor * wk_b      = nullptr;
    struct lm_ggml_tensor * wv_b      = nullptr;
    struct lm_ggml_tensor * wq_cross  = nullptr;
    struct lm_ggml_tensor * wk_cross  = nullptr;
    struct lm_ggml_tensor * wv_cross  = nullptr;
    struct lm_ggml_tensor * wo_cross  = nullptr;
    struct lm_ggml_tensor * wq_enc    = nullptr;
    struct lm_ggml_tensor * wk_enc    = nullptr;
    struct lm_ggml_tensor * wv_enc    = nullptr;
    struct lm_ggml_tensor * wo_enc    = nullptr;

    // attention bias
    struct lm_ggml_tensor * bq   = nullptr;
    struct lm_ggml_tensor * bk   = nullptr;
    struct lm_ggml_tensor * bv   = nullptr;
    struct lm_ggml_tensor * bo   = nullptr;
    struct lm_ggml_tensor * bqkv = nullptr;

    // relative position bias
    struct lm_ggml_tensor * attn_rel_b       = nullptr;
    struct lm_ggml_tensor * attn_rel_b_enc   = nullptr;
    struct lm_ggml_tensor * attn_rel_b_cross = nullptr;

    // normalization
    struct lm_ggml_tensor * ffn_norm         = nullptr;
    struct lm_ggml_tensor * ffn_norm_b       = nullptr;
    struct lm_ggml_tensor * ffn_post_norm    = nullptr;
    struct lm_ggml_tensor * layer_out_norm   = nullptr;
    struct lm_ggml_tensor * layer_out_norm_b = nullptr;
    struct lm_ggml_tensor * ffn_norm_exps    = nullptr;
    struct lm_ggml_tensor * ffn_norm_enc     = nullptr;

    // ff
    struct lm_ggml_tensor * ffn_gate     = nullptr; // w1
    struct lm_ggml_tensor * ffn_down     = nullptr; // w2
    struct lm_ggml_tensor * ffn_up       = nullptr; // w3
    struct lm_ggml_tensor * ffn_gate_enc = nullptr;
    struct lm_ggml_tensor * ffn_down_enc = nullptr;
    struct lm_ggml_tensor * ffn_up_enc   = nullptr;

    // ff MoE
    struct lm_ggml_tensor * ffn_gate_inp  = nullptr;
    struct lm_ggml_tensor * ffn_gate_exps = nullptr;
    struct lm_ggml_tensor * ffn_down_exps = nullptr;
    struct lm_ggml_tensor * ffn_up_exps   = nullptr;

    // ff shared expert (shexp)
    struct lm_ggml_tensor * ffn_gate_inp_shexp = nullptr;
    struct lm_ggml_tensor * ffn_gate_shexp     = nullptr;
    struct lm_ggml_tensor * ffn_down_shexp     = nullptr;
    struct lm_ggml_tensor * ffn_up_shexp       = nullptr;

    // ff bias
    struct lm_ggml_tensor * ffn_gate_b = nullptr;
    struct lm_ggml_tensor * ffn_down_b = nullptr; // b2
    struct lm_ggml_tensor * ffn_up_b   = nullptr; // b3
    struct lm_ggml_tensor * ffn_act    = nullptr;
    struct lm_ggml_tensor * ffn_exp_probs_b = nullptr;

    // mamba proj
    struct lm_ggml_tensor * ssm_in  = nullptr;
    struct lm_ggml_tensor * ssm_x   = nullptr;
    struct lm_ggml_tensor * ssm_dt  = nullptr;
    struct lm_ggml_tensor * ssm_out = nullptr;

    // mamba
    struct lm_ggml_tensor * ssm_conv1d = nullptr;
    struct lm_ggml_tensor * ssm_a      = nullptr;
    struct lm_ggml_tensor * ssm_d      = nullptr;

    // mamba bias
    struct lm_ggml_tensor * ssm_conv1d_b = nullptr;
    struct lm_ggml_tensor * ssm_dt_b     = nullptr;

    // rwkv
    struct lm_ggml_tensor * time_mix_w1         = nullptr;
    struct lm_ggml_tensor * time_mix_w2         = nullptr;
    struct lm_ggml_tensor * time_mix_lerp_x     = nullptr;
    struct lm_ggml_tensor * time_mix_lerp_w     = nullptr;
    struct lm_ggml_tensor * time_mix_lerp_k     = nullptr;
    struct lm_ggml_tensor * time_mix_lerp_v     = nullptr;
    struct lm_ggml_tensor * time_mix_lerp_r     = nullptr;
    struct lm_ggml_tensor * time_mix_lerp_g     = nullptr;
    struct lm_ggml_tensor * time_mix_lerp_fused = nullptr;

    struct lm_ggml_tensor * time_mix_first        = nullptr;
    struct lm_ggml_tensor * time_mix_decay        = nullptr;
    struct lm_ggml_tensor * time_mix_decay_w1     = nullptr;
    struct lm_ggml_tensor * time_mix_decay_w2     = nullptr;
    struct lm_ggml_tensor * time_mix_key          = nullptr;
    struct lm_ggml_tensor * time_mix_key_b        = nullptr;
    struct lm_ggml_tensor * time_mix_value        = nullptr;
    struct lm_ggml_tensor * time_mix_value_b      = nullptr;
    struct lm_ggml_tensor * time_mix_receptance   = nullptr;
    struct lm_ggml_tensor * time_mix_receptance_b = nullptr;
    struct lm_ggml_tensor * time_mix_gate         = nullptr;

    // rwkv7
    struct lm_ggml_tensor * time_mix_w0         = nullptr;
    struct lm_ggml_tensor * time_mix_a0         = nullptr;
    struct lm_ggml_tensor * time_mix_a1         = nullptr;
    struct lm_ggml_tensor * time_mix_a2         = nullptr;
    struct lm_ggml_tensor * time_mix_v0         = nullptr;
    struct lm_ggml_tensor * time_mix_v1         = nullptr;
    struct lm_ggml_tensor * time_mix_v2         = nullptr;
    struct lm_ggml_tensor * time_mix_g1         = nullptr;
    struct lm_ggml_tensor * time_mix_g2         = nullptr;
    struct lm_ggml_tensor * time_mix_k_k        = nullptr;
    struct lm_ggml_tensor * time_mix_k_a        = nullptr;
    struct lm_ggml_tensor * time_mix_r_k        = nullptr;

    struct lm_ggml_tensor * time_mix_ln     = nullptr;
    struct lm_ggml_tensor * time_mix_ln_b   = nullptr;
    struct lm_ggml_tensor * time_mix_output = nullptr;

    struct lm_ggml_tensor * channel_mix_lerp_k = nullptr;
    struct lm_ggml_tensor * channel_mix_lerp_r = nullptr;

    struct lm_ggml_tensor * channel_mix_key        = nullptr;
    struct lm_ggml_tensor * channel_mix_receptance = nullptr;
    struct lm_ggml_tensor * channel_mix_value      = nullptr;

    // long rope factors
    struct lm_ggml_tensor * rope_long  = nullptr;
    struct lm_ggml_tensor * rope_short = nullptr;
    struct lm_ggml_tensor * rope_freqs = nullptr;

    // bitnet scale
    struct lm_ggml_tensor * wq_scale       = nullptr;
    struct lm_ggml_tensor * wk_scale       = nullptr;
    struct lm_ggml_tensor * wv_scale       = nullptr;
    struct lm_ggml_tensor * wo_scale       = nullptr;
    struct lm_ggml_tensor * ffn_gate_scale = nullptr;
    struct lm_ggml_tensor * ffn_up_scale   = nullptr;
    struct lm_ggml_tensor * ffn_down_scale = nullptr;

    struct llama_layer_posnet posnet;

    struct llama_layer_convnext convnext;
};

struct llama_model {
    llm_type type = LLM_TYPE_UNKNOWN;
    llm_arch arch = LLM_ARCH_UNKNOWN;

    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab   vocab;

    struct lm_ggml_tensor * tok_embd   = nullptr;
    struct lm_ggml_tensor * type_embd  = nullptr;
    struct lm_ggml_tensor * pos_embd   = nullptr;
    struct lm_ggml_tensor * tok_norm   = nullptr;
    struct lm_ggml_tensor * tok_norm_b = nullptr;

    struct lm_ggml_tensor * output_norm     = nullptr;
    struct lm_ggml_tensor * output_norm_b   = nullptr;
    struct lm_ggml_tensor * output          = nullptr;
    struct lm_ggml_tensor * output_b        = nullptr;
    struct lm_ggml_tensor * output_norm_enc = nullptr;

    // classifier
    struct lm_ggml_tensor * cls       = nullptr;
    struct lm_ggml_tensor * cls_b     = nullptr;
    struct lm_ggml_tensor * cls_out   = nullptr;
    struct lm_ggml_tensor * cls_out_b = nullptr;

    struct lm_ggml_tensor * conv1d   = nullptr;
    struct lm_ggml_tensor * conv1d_b = nullptr;

    std::vector<llama_layer> layers;

    llama_model_params params;

    // gguf metadata
    std::unordered_map<std::string, std::string> lm_gguf_kv;

    // list of devices used in this model
    std::vector<lm_ggml_backend_dev_t> devices;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct lm_ggml_tensor *>> tensors_by_name;

    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    explicit llama_model(const struct llama_model_params & params);
    ~llama_model();

    void load_stats  (llama_model_loader & ml);
    void load_arch   (llama_model_loader & ml);
    void load_hparams(llama_model_loader & ml);
    void load_vocab  (llama_model_loader & ml);
    bool load_tensors(llama_model_loader & ml); // returns false if cancelled by progress_callback

    std::string arch_name() const;
    std::string type_name() const;

    std::string desc() const;

    size_t size() const;
    size_t n_tensors() const;
    size_t n_devices() const;

    // total number of parameters in the model
    uint64_t n_elements() const;

    void print_info() const;

    lm_ggml_backend_dev_t dev_layer(int il) const;
    lm_ggml_backend_dev_t dev_output() const;

    lm_ggml_backend_buffer_type_t select_buft(int il) const;

    bool has_tensor_overrides() const;

    const struct lm_ggml_tensor * get_tensor(const char * name) const;

    float get_rope_freq_base (const llama_cparams & cparams, int il) const;
    float get_rope_freq_scale(const llama_cparams & cparams, int il) const;

    lm_ggml_tensor * get_rope_factors(const llama_cparams & cparams, int il) const;

    // note: can mutate `cparams`
    // TODO: move this to new llm_arch_model_i interface
    llama_memory_i * create_memory(const llama_memory_params & params, llama_cparams & cparams) const;

    // TODO: move this to new llm_arch_model_i interface
    llm_graph_result_ptr build_graph(
            const llm_graph_params & params,
                       lm_ggml_cgraph * gf,
                    llm_graph_type   type) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

const char * llm_type_name(llm_type type);

// For internal test use
// TODO: remove
const std::vector<std::pair<std::string, lm_ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);
