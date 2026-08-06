#ifndef CACTUS_METAL_BACKEND_H
#define CACTUS_METAL_BACKEND_H

#if defined(__APPLE__)
#define CACTUS_HAS_METAL 1
#else
#define CACTUS_HAS_METAL 0
#endif

#include "cactus_kernels.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif
bool cactus_metal_available();
#ifdef __cplusplus
}
#endif

void cactus_metal_set_active(bool active);
bool cactus_metal_active_mode();

void cactus_metal_quant_matmul(const CactusQuantMatrix* W, const __fp16* A,
                               uint32_t M, __fp16* C);

void  cactus_metal_session_begin();
void  cactus_metal_session_sync();
void  cactus_metal_session_flush();
void  cactus_metal_invalidate_host_wraps();
void  cactus_metal_trim_prefill_cache();
void  cactus_metal_session_end();

void* cactus_metal_alloc_shared(size_t bytes);
void* cactus_metal_alloc_pooled(size_t bytes);
void  cactus_metal_free_shared(void* contents);

bool cactus_metal_encode_copy(void* out, const void* in, size_t bytes);
bool cactus_metal_encode_binary(int op_type, void* out, const void* a, const void* b, size_t n);
bool cactus_metal_encode_scalar(int op_type, void* out, const void* in, size_t n, float param);
bool cactus_metal_encode_unary(int op_type, void* out, const void* in, size_t n);
bool cactus_metal_encode_swiglu(void* out, const void* gate, const void* up, size_t n, float scale);
bool cactus_metal_encode_rms_norm(void* out, const void* in, const void* weight,
                                  size_t rows, size_t dim, float eps);
bool cactus_metal_encode_rms_norm_add(void* out, const void* in, const void* weight, const void* res,
                                      size_t rows, size_t dim, float eps);
bool cactus_metal_encode_rms_norm_add_rms(void* h_out, void* xn_out, const void* in, const void* w1,
                                          const void* res, const void* w2,
                                          size_t rows, size_t dim, float eps, float out_scale);
bool cactus_metal_encode_rms_norm_add_scale(void* out, const void* in, const void* weight, const void* res,
                                            size_t rows, size_t dim, float eps, float out_scale);
bool cactus_metal_encode_argmax(const void* logits, uint32_t vocab, void* out3, const void* bias);
bool cactus_metal_encode_softcap(void* out, const void* in, size_t n, float cap);
bool cactus_metal_encode_adjust_logits(void* logits, size_t vocab,
                                       const uint32_t* recent, uint32_t n_recent,
                                       int64_t suppressed, float penalty);
bool cactus_metal_encode_cast(void* out, int out_prec, const void* in, int in_prec, size_t n);
bool cactus_metal_encode_quant_matmul(void* out, const void* lhs, const CactusQuantMatrix* W);
bool cactus_metal_encode_quant_matmul_m(void* out, const void* lhs, const CactusQuantMatrix* W, uint32_t M);
bool cactus_metal_encode_transform_batch(const void* x, const CactusQuantMatrix* const* Ws, int B, void* const* codes);
bool cactus_metal_encode_gemv_precoded(void* out, const void* code, const CactusQuantMatrix* W);
bool cactus_metal_encode_transform_gemv(void* out, const void* x, const CactusQuantMatrix* W, const void* osw);
bool cactus_metal_transform_gemv_fits(uint32_t K);
bool cactus_metal_encode_gemv_cat(void* const* outs, const void* const* codes,
                                  const CactusQuantMatrix* const* Ws, int B);
bool cactus_metal_encode_swiglu_transform(void* code, const void* gate, const void* up,
                                          const CactusQuantMatrix* W, float scale);
bool cactus_metal_prewarm_quant(const CactusQuantMatrix* W);
bool cactus_metal_encode_rope_pair(void* out, const void* x, const void* c, const void* s,
                                   uint32_t H, uint32_t D);
bool cactus_metal_encode_rope_pair_rms(void* out, const void* x, const void* w,
                                       const void* c, const void* s,
                                       uint32_t H, uint32_t D, float eps);
bool cactus_metal_encode_deltanet_decode(void* out, const void* q, const void* k, const void* v,
                                         const void* g, const void* b, const void* s,
                                         uint32_t B, uint32_t Hq, uint32_t Hv,
                                         uint32_t K, uint32_t V, float scale);
bool cactus_metal_encode_deltanet_prefill(void* out, const void* q, const void* k, const void* v,
                                          const void* g, const void* b, const void* s,
                                          uint32_t B, uint32_t T, uint32_t Hq, uint32_t Hv,
                                          uint32_t K, uint32_t V, float scale);
bool cactus_metal_encode_rms2_add_clip(void* out, const void* a, const void* wa,
                                       const void* b, const void* wb, size_t dim,
                                       float eps_a, float eps_b);
bool cactus_metal_encode_rms_norm_scale(void* out, const void* in, const void* weight,
                                        size_t rows, size_t dim, float eps, float oscale);
bool cactus_metal_encode_softmax_topk(void* probs, void* topk, const void* in,
                                      size_t rows, size_t cols, size_t k, float scale);
bool cactus_metal_encode_topk_rows(void* out, const void* in, size_t rows, size_t cols, size_t k);
bool cactus_metal_moe_cq4_ready(const CactusQuantMatrix* w1_0);
bool cactus_metal_moe_cq4_build(const CactusQuantMatrix* w1s, const CactusQuantMatrix* w3s,
                                const CactusQuantMatrix* w2s, uint32_t num_experts);
bool cactus_metal_encode_moe_gated_cq4(void* out, const void* hidden, const void* probs,
                                       const void* topk, const CactusQuantMatrix* w1_0,
                                       uint32_t num_experts, uint32_t top_k, uint32_t tokens,
                                       uint32_t act, uint32_t normalize, float eps, float scaling);
bool cactus_metal_encode_quant_matmul_ortho(void* out, const void* act, void* code,
                                            const CactusQuantMatrix* W);
bool cactus_metal_encode_embedding_ortho(void* out, uint32_t row, const CactusQuantMatrix* W, float scale);
bool cactus_metal_encode_embedding_hadamard(void* out, uint32_t row, const CactusQuantMatrix* W);
bool cactus_metal_encode_embedding_ortho_m(void* out, const CactusQuantMatrix* W, const uint32_t* rows, uint32_t M);
bool cactus_metal_encode_embedding_hadamard_m(void* out, const CactusQuantMatrix* W, const uint32_t* rows, uint32_t M);
bool cactus_metal_encode_gather_f16(void* out, const void* table, size_t table_bytes, const uint32_t* rows, uint32_t M, uint32_t D);

bool cactus_metal_encode_attention_i8(
    void* out, const void* q, const void* knew, const void* vnew,
    const void* kc, const void* vc, const void* ks, const void* vs,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t v_hdim,
    uint32_t history_len, uint32_t total_keys, uint32_t kv_start, uint32_t kv_end,
    float scale, size_t kc_bytes, size_t vc_bytes, size_t ks_bytes, size_t vs_bytes);

bool cactus_metal_encode_attention_fused_i8(
    void* out, const void* q, const void* kraw, const void* vraw,
    void* kc, void* vc, void* ks, void* vs,
    const void* qw, const void* kw, const void* vw, const void* cs, const void* sn,
    uint32_t nqh, uint32_t hd, uint32_t vhd,
    uint32_t kv_start, uint32_t kv_end, uint32_t slot, uint32_t has_new,
    float eps, float scale,
    size_t kc_bytes, size_t vc_bytes, size_t ks_bytes, size_t vs_bytes);

bool cactus_metal_encode_attention_i8_prefill(
    void* out, const void* q, const void* knew, const void* vnew,
    const void* kc, const void* vc, const void* ks, const void* vs,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t v_hdim,
    uint32_t history_len, uint32_t new_len, uint32_t q_pos0, uint32_t window, uint32_t is_causal, uint32_t M,
    float scale, size_t kc_bytes, size_t vc_bytes, size_t ks_bytes, size_t vs_bytes,
    uint32_t sink, uint32_t ring);

bool cactus_metal_encode_binary_f32(int op, void* out, const void* a, const void* b, size_t n);
bool cactus_metal_encode_scalar_f32(int op, void* out, const void* in, size_t n, float p);
bool cactus_metal_encode_unary_f32(int op, void* out, const void* in, size_t n);
bool cactus_metal_encode_clamp(void* out, const void* in, size_t n, float lo, float hi, int f32);
bool cactus_metal_encode_glu(void* out, const void* in, size_t split, size_t inner, size_t n_out);
bool cactus_metal_encode_layer_norm(void* out, const void* in, const void* w, const void* b,
                                    size_t rows, size_t dim, float eps);
bool cactus_metal_encode_softmax_rows(void* out, const void* in, size_t rows, size_t cols);
bool cactus_metal_encode_conv1d_k3(void* out, const void* x, const void* w, int w_int8,
                                   const void* w_scales, uint32_t w_gs,
                                   uint32_t Cin, uint32_t L, uint32_t Cout, uint32_t Lout, uint32_t stride);
bool cactus_metal_encode_gemm_f16(void* out, const void* lhs, const void* rhs,
                                  uint32_t M, uint32_t K, uint32_t N, int pretransposed);
bool cactus_metal_encode_attention_f16(void* out, const void* q, const void* k, const void* v, const void* mask,
    uint32_t B, uint32_t T, uint32_t S, uint32_t HQ, uint32_t HKV, uint32_t D, uint32_t DV,
    float scale, uint32_t causal, uint32_t pos_off, uint32_t window, float logit_cap, uint32_t mask_mode);
bool cactus_metal_encode_reduce_axis(int op, void* out, const void* in, uint32_t outer,
                                     uint32_t axis_size, uint32_t inner, int f32);
bool cactus_metal_encode_cumsum(void* out, const void* in, uint32_t outer,
                                uint32_t axis_size, uint32_t inner, int f32);
bool cactus_metal_encode_concat2(void* out, const void* a, const void* b,
                                 uint32_t a_outer, uint32_t b_outer,
                                 uint32_t a_axis, uint32_t b_axis, uint32_t inner);
bool cactus_metal_encode_gather_f32idx(void* out, const void* table, const void* idx,
                                       uint32_t rows, uint32_t D, size_t table_bytes);
bool cactus_metal_encode_rope_full(void* out, const void* in, uint32_t tokens, uint32_t S,
                                   uint32_t H, uint32_t D, uint32_t rot, uint32_t pos0,
                                   float theta, int gptj);
bool cactus_metal_encode_maxpool1d(void* out, const void* in, uint32_t NC, uint32_t L,
                                   uint32_t Lout, uint32_t K, uint32_t stride);
bool cactus_metal_encode_bilinear(void* out, const void* in, uint32_t sh, uint32_t sw,
                                  uint32_t dh, uint32_t dw, uint32_t E, int align);
bool cactus_metal_encode_conv1d_gen(void* out, const void* x, const void* w, const void* bias,
                                    uint32_t N, uint32_t Cin, uint32_t L, uint32_t Cout,
                                    uint32_t Lout, uint32_t K, uint32_t stride, int w_ck_co);
bool cactus_metal_encode_conv1d_nlc_dw(void* out, const void* x, const void* w, const void* bias,
                                       uint32_t N, uint32_t L, uint32_t C, uint32_t K,
                                       uint32_t dil, uint32_t pad);
bool cactus_metal_encode_conv2d(void* out, const void* x, const void* w, const void* bias,
                                uint32_t N, uint32_t Cin, uint32_t H, uint32_t W, uint32_t Cout,
                                uint32_t Ho, uint32_t Wo, uint32_t K, uint32_t stride,
                                uint32_t pad, int dw);
bool cactus_metal_encode_batchnorm(void* out, const void* x, const void* w, const void* b,
                                   const void* rm, const void* rv, uint32_t C, uint32_t inner,
                                   uint32_t total, float eps);
bool cactus_metal_encode_groupnorm(void* out, const void* x, const void* w, const void* b,
                                   uint32_t N, uint32_t C, uint32_t S, uint32_t groups, float eps);
bool cactus_metal_encode_bias_add_rows(void* y, const void* bias, uint32_t C, uint32_t total);
bool cactus_metal_encode_elemwise_chain(void* out, const void* in, const float* steps,
                                        uint32_t nsteps, const void* side0, const void* side1,
                                        const void* side2, const size_t* side_elems,
                                        size_t n, uint32_t flags, uint32_t inner);
bool cactus_metal_encode_rms_norm_add_rows(void* ysum, void* ynorm, const void* x, const void* res,
                                           const void* w, uint32_t rows, uint32_t dim, float eps,
                                           int clipped);
bool cactus_metal_encode_gemm_batch(void* out, const void* a, const void* b,
                                    uint32_t M, uint32_t K, uint32_t N, uint32_t batch, int f32out, int f32a);
bool cactus_metal_encode_conv1d_dw(void* out, const void* x, const void* w,
                                   uint32_t C, uint32_t L, uint32_t Lout, uint32_t K, uint32_t stride);
bool cactus_metal_encode_transpose2d(void* out, const void* in, uint32_t batch, uint32_t R, uint32_t C);
bool cactus_metal_encode_strided_copy(void* out, const void* in, const uint32_t* oshape,
    const uint32_t* sstride, uint32_t ndim, uint32_t total, uint32_t base, size_t in_bytes, size_t out_bytes);

bool cactus_metal_encode_bcast_binary(int op, void* out, const void* a, const void* b,
    const uint32_t* oshape, const uint32_t* astride, const uint32_t* bstride, uint32_t ndim, uint32_t total,
    size_t a_bytes, size_t b_bytes, size_t out_bytes);

bool cactus_metal_encode_strided_scatter(void* out, const void* in, const uint32_t* ishape,
    const uint32_t* ostride, uint32_t ndim, uint32_t total, uint32_t base, size_t in_bytes, size_t out_bytes);

bool cactus_metal_encode_kv_append_i8(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t current_len, uint32_t group_size,
    size_t src_bytes, size_t int8_bytes, size_t scale_bytes);
bool cactus_metal_encode_kv_append_sliding_i8(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t keep_sink, uint32_t remaining, uint32_t shift_src,
    uint32_t group_size, size_t src_bytes, size_t int8_bytes, size_t scale_bytes);
bool cactus_metal_encode_kv_append_sliding_i8_m(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t keep_sink, uint32_t remaining, uint32_t shift_src,
    uint32_t group_size, uint32_t M, size_t src_bytes, size_t int8_bytes, size_t scale_bytes);

bool cactus_metal_encode_kv_append_i8_m(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t current_len, uint32_t group_size, uint32_t M,
    size_t src_bytes, size_t int8_bytes, size_t scale_bytes);

bool cactus_metal_encode_kv_append_ring_i8_m(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t current_len, uint32_t group_size, uint32_t M,
    uint32_t sink, uint32_t W, size_t src_bytes, size_t int8_bytes, size_t scale_bytes);

bool cactus_metal_encode_conv_cache_append(void* out, const void* src, void* ring,
    uint32_t hd, uint32_t ws, uint32_t nnew, uint32_t head0, uint32_t count_new,
    uint32_t num_rows, int src_f32);

bool cactus_metal_encode_rel_pos_bias(void* y, const void* q, const void* r,
    uint32_t B, uint32_t T, uint32_t H, uint32_t D, uint32_t R, int r_batched, float scale);

bool cactus_metal_encode_gemv_bias(void* out, const void* x, const void* w, const void* bias,
    uint32_t K, uint32_t N, int pretransposed);

#endif
