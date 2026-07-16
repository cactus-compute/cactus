#ifndef CACTUS_GPU_H
#define CACTUS_GPU_H

#include "cactus_kernels.h"
#include <cstdint>

#if !defined(__ANDROID__)

#include "metal_backend.h"

inline bool cactus_gpu_supports_plans() { return true; }
inline bool cactus_gpu_fold_ready() { return true; }
inline void cactus_gpu_fold_buffers(void*, size_t, void*, size_t) {}
inline bool cactus_gpu_owns(const void*) { return false; }
inline void cactus_gpu_note_mmap_range(const void*, size_t) {}
inline const char* cactus_gpu_default_rules() { return nullptr; }
inline bool cactus_gpu_auto_available() { return cactus_metal_available(); }

#define cactus_gpu_available cactus_metal_available
#define cactus_gpu_set_active cactus_metal_set_active
#define cactus_gpu_active_mode cactus_metal_active_mode
#define cactus_gpu_quant_matmul cactus_metal_quant_matmul
#define cactus_gpu_session_begin cactus_metal_session_begin
#define cactus_gpu_session_sync cactus_metal_session_sync
#define cactus_gpu_session_flush cactus_metal_session_flush
#define cactus_gpu_invalidate_host_wraps cactus_metal_invalidate_host_wraps
#define cactus_gpu_trim_prefill_cache cactus_metal_trim_prefill_cache
#define cactus_gpu_session_end cactus_metal_session_end
#define cactus_gpu_alloc_shared cactus_metal_alloc_shared
#define cactus_gpu_alloc_pooled cactus_metal_alloc_pooled
#define cactus_gpu_free_shared cactus_metal_free_shared
#define cactus_gpu_encode_copy cactus_metal_encode_copy
#define cactus_gpu_encode_binary cactus_metal_encode_binary
#define cactus_gpu_encode_scalar cactus_metal_encode_scalar
#define cactus_gpu_encode_unary cactus_metal_encode_unary
#define cactus_gpu_encode_swiglu cactus_metal_encode_swiglu
#define cactus_gpu_encode_rms_norm cactus_metal_encode_rms_norm
#define cactus_gpu_encode_rms_norm_add cactus_metal_encode_rms_norm_add
#define cactus_gpu_encode_rms_norm_add_rms cactus_metal_encode_rms_norm_add_rms
#define cactus_gpu_encode_rms_norm_add_scale cactus_metal_encode_rms_norm_add_scale
#define cactus_gpu_encode_argmax cactus_metal_encode_argmax
#define cactus_gpu_encode_softcap cactus_metal_encode_softcap
#define cactus_gpu_encode_adjust_logits cactus_metal_encode_adjust_logits
#define cactus_gpu_encode_cast cactus_metal_encode_cast
#define cactus_gpu_encode_quant_matmul cactus_metal_encode_quant_matmul
#define cactus_gpu_encode_quant_matmul_m cactus_metal_encode_quant_matmul_m
#define cactus_gpu_encode_transform_batch cactus_metal_encode_transform_batch
#define cactus_gpu_encode_gemv_precoded cactus_metal_encode_gemv_precoded
#define cactus_gpu_encode_transform_gemv cactus_metal_encode_transform_gemv
#define cactus_gpu_transform_gemv_fits cactus_metal_transform_gemv_fits
#define cactus_gpu_encode_gemv_cat cactus_metal_encode_gemv_cat
#define cactus_gpu_encode_swiglu_transform cactus_metal_encode_swiglu_transform
#define cactus_gpu_prewarm_quant cactus_metal_prewarm_quant
#define cactus_gpu_encode_rope_pair cactus_metal_encode_rope_pair
#define cactus_gpu_encode_rope_pair_rms cactus_metal_encode_rope_pair_rms
#define cactus_gpu_encode_deltanet_decode cactus_metal_encode_deltanet_decode
#define cactus_gpu_encode_deltanet_prefill cactus_metal_encode_deltanet_prefill
#define cactus_gpu_encode_rms2_add_clip cactus_metal_encode_rms2_add_clip
#define cactus_gpu_encode_rms_norm_scale cactus_metal_encode_rms_norm_scale
#define cactus_gpu_encode_softmax_topk cactus_metal_encode_softmax_topk
#define cactus_gpu_encode_topk_rows cactus_metal_encode_topk_rows
#define cactus_gpu_moe_cq4_ready cactus_metal_moe_cq4_ready
#define cactus_gpu_moe_cq4_build cactus_metal_moe_cq4_build
#define cactus_gpu_encode_moe_gated_cq4 cactus_metal_encode_moe_gated_cq4
#define cactus_gpu_encode_quant_matmul_ortho cactus_metal_encode_quant_matmul_ortho
#define cactus_gpu_encode_embedding_ortho cactus_metal_encode_embedding_ortho
#define cactus_gpu_encode_embedding_hadamard cactus_metal_encode_embedding_hadamard
#define cactus_gpu_encode_embedding_ortho_m cactus_metal_encode_embedding_ortho_m
#define cactus_gpu_encode_embedding_hadamard_m cactus_metal_encode_embedding_hadamard_m
#define cactus_gpu_encode_gather_f16 cactus_metal_encode_gather_f16
#define cactus_gpu_encode_attention_i8 cactus_metal_encode_attention_i8
#define cactus_gpu_encode_attention_fused_i8 cactus_metal_encode_attention_fused_i8
#define cactus_gpu_encode_attention_i8_prefill cactus_metal_encode_attention_i8_prefill
#define cactus_gpu_encode_binary_f32 cactus_metal_encode_binary_f32
#define cactus_gpu_encode_scalar_f32 cactus_metal_encode_scalar_f32
#define cactus_gpu_encode_unary_f32 cactus_metal_encode_unary_f32
#define cactus_gpu_encode_clamp cactus_metal_encode_clamp
#define cactus_gpu_encode_glu cactus_metal_encode_glu
#define cactus_gpu_encode_layer_norm cactus_metal_encode_layer_norm
#define cactus_gpu_encode_softmax_rows cactus_metal_encode_softmax_rows
#define cactus_gpu_encode_conv1d_k3 cactus_metal_encode_conv1d_k3
#define cactus_gpu_encode_gemm_f16 cactus_metal_encode_gemm_f16
#define cactus_gpu_encode_attention_f16 cactus_metal_encode_attention_f16
#define cactus_gpu_encode_reduce_axis cactus_metal_encode_reduce_axis
#define cactus_gpu_encode_cumsum cactus_metal_encode_cumsum
#define cactus_gpu_encode_concat2 cactus_metal_encode_concat2
#define cactus_gpu_encode_gather_f32idx cactus_metal_encode_gather_f32idx
#define cactus_gpu_encode_rope_full cactus_metal_encode_rope_full
#define cactus_gpu_encode_maxpool1d cactus_metal_encode_maxpool1d
#define cactus_gpu_encode_bilinear cactus_metal_encode_bilinear
#define cactus_gpu_encode_conv1d_gen cactus_metal_encode_conv1d_gen
#define cactus_gpu_encode_conv1d_nlc_dw cactus_metal_encode_conv1d_nlc_dw
#define cactus_gpu_encode_conv2d cactus_metal_encode_conv2d
#define cactus_gpu_encode_batchnorm cactus_metal_encode_batchnorm
#define cactus_gpu_encode_groupnorm cactus_metal_encode_groupnorm
#define cactus_gpu_encode_bias_add_rows cactus_metal_encode_bias_add_rows
#define cactus_gpu_encode_elemwise_chain cactus_metal_encode_elemwise_chain
#define cactus_gpu_encode_rms_norm_add_rows cactus_metal_encode_rms_norm_add_rows
#define cactus_gpu_encode_gemm_batch cactus_metal_encode_gemm_batch
#define cactus_gpu_encode_conv1d_dw cactus_metal_encode_conv1d_dw
#define cactus_gpu_encode_transpose2d cactus_metal_encode_transpose2d
#define cactus_gpu_encode_strided_copy cactus_metal_encode_strided_copy
#define cactus_gpu_encode_bcast_binary cactus_metal_encode_bcast_binary
#define cactus_gpu_encode_strided_scatter cactus_metal_encode_strided_scatter
#define cactus_gpu_encode_kv_append_i8 cactus_metal_encode_kv_append_i8
#define cactus_gpu_encode_kv_append_sliding_i8 cactus_metal_encode_kv_append_sliding_i8
#define cactus_gpu_encode_kv_append_sliding_i8_m cactus_metal_encode_kv_append_sliding_i8_m
#define cactus_gpu_encode_kv_append_i8_m cactus_metal_encode_kv_append_i8_m
#define cactus_gpu_encode_kv_append_ring_i8_m cactus_metal_encode_kv_append_ring_i8_m
#define cactus_gpu_encode_conv_cache_append cactus_metal_encode_conv_cache_append
#define cactus_gpu_encode_rel_pos_bias cactus_metal_encode_rel_pos_bias
#define cactus_gpu_encode_gemv_bias cactus_metal_encode_gemv_bias

#else

#include "vulkan_backend.h"

inline bool cactus_gpu_supports_plans() { return cactus_vulkan_available(); }
inline bool cactus_gpu_fold_ready() { return cactus_vulkan_available(); }
inline void cactus_gpu_fold_buffers(void* h, size_t hb, void* p, size_t pb) { cactus_vulkan_fold_buffers(h, hb, p, pb); }
inline bool cactus_gpu_owns(const void* p) { return cactus_vulkan_owns(p); }
inline void cactus_gpu_note_mmap_range(const void* b, size_t n) { cactus_vulkan_note_mmap_range(b, n); }
inline const char* cactus_gpu_default_rules() { return "1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18"; }
inline bool cactus_gpu_auto_available() { return cactus_vulkan_available(); }

inline bool cactus_gpu_available() { return cactus_vulkan_available(); }
inline bool& cactus_gpu_active_flag_ref() { static bool f = false; return f; }
inline void cactus_gpu_set_active(bool on) { cactus_gpu_active_flag_ref() = on; }
inline bool cactus_gpu_active_mode() { return cactus_vulkan_available() && cactus_gpu_active_flag_ref(); }
inline void cactus_gpu_session_begin() { cactus_vulkan_session_begin(); }
inline void cactus_gpu_session_sync() { cactus_vulkan_session_sync(); }
inline void cactus_gpu_session_flush() { cactus_vulkan_session_flush(); }
inline void cactus_gpu_session_end() { cactus_vulkan_session_end(); }
inline void cactus_gpu_invalidate_host_wraps() { cactus_vulkan_invalidate_host_wraps(); }
inline void cactus_gpu_trim_prefill_cache() { cactus_vulkan_trim_prefill_cache(); }
inline void* cactus_gpu_alloc_shared(size_t bytes) { return cactus_vulkan_alloc_shared(bytes); }
inline void* cactus_gpu_alloc_pooled(size_t bytes) { return cactus_vulkan_alloc_pooled(bytes); }
inline void cactus_gpu_free_shared(void* p) { cactus_vulkan_free_shared(p); }

inline void cactus_gpu_quant_matmul(const CactusQuantMatrix* W, const __fp16* A, uint32_t M, __fp16* C) {
    cactus_quant_matmul(W, A, M, C);
}

inline bool cactus_gpu_encode_binary(int op, void* out, const void* a, const void* b, size_t n) {
    return (cactus_vulkan_op_enabled("ew") || cactus_vulkan_op_enabled("binary"))
        && cactus_vulkan_encode_binary_f16(op, out, a, b, n);
}
inline bool cactus_gpu_encode_scalar(int op, void* out, const void* in, size_t n, float p) {
    return (cactus_vulkan_op_enabled("ew") || cactus_vulkan_op_enabled("scalar"))
        && cactus_vulkan_encode_scalar_f16(op, out, in, n, p);
}
inline bool cactus_gpu_encode_unary(int op, void* out, const void* in, size_t n) {
    return (cactus_vulkan_op_enabled("ew") || cactus_vulkan_op_enabled("unary"))
        && cactus_vulkan_encode_unary_f16(op, out, in, n);
}
inline bool cactus_gpu_encode_swiglu(void* out, const void* gate, const void* up, size_t n, float scale) {
    return cactus_vulkan_op_enabled("swiglu") && cactus_vulkan_encode_swiglu_f16(out, gate, up, n, scale);
}
inline bool cactus_gpu_encode_rms_norm(void* out, const void* in, const void* weight,
                                       size_t rows, size_t dim, float eps) {
    return cactus_vulkan_op_enabled("rms") && cactus_vulkan_encode_rms_norm_f16(out, in, weight, rows, dim, eps);
}
inline bool cactus_gpu_encode_quant_matmul(void* out, const void* lhs, const CactusQuantMatrix* W) {
    return cactus_vulkan_op_enabled("matmul") && cactus_vulkan_encode_cq_gemv(out, lhs, W);
}

inline bool cactus_gpu_encode_copy(void* out, const void* in, size_t bytes) {
    return cactus_vulkan_op_enabled("copy") && cactus_vulkan_encode_copy(out, in, bytes);
}
inline bool cactus_gpu_encode_rms_norm_add(void* out, const void* in, const void* w, const void* res,
        size_t rows, size_t dim, float eps) {
    return cactus_vulkan_op_enabled("rms")
        && cactus_vulkan_encode_rms_norm_add(out, in, w, res, rows, dim, eps, 1.0f);
}
inline bool cactus_gpu_encode_rms_norm_add_rms(void* h_out, void* xn_out, const void* in, const void* w1,
        const void* res, const void* w2, size_t rows, size_t dim, float eps, float out_scale) {
    return cactus_vulkan_op_enabled("rms")
        && cactus_vulkan_encode_rms_norm_add_rms(h_out, xn_out, in, w1, res, w2, rows, dim, eps, out_scale);
}
inline bool cactus_gpu_encode_rms_norm_add_scale(void* out, const void* in, const void* w, const void* res,
        size_t rows, size_t dim, float eps, float out_scale) {
    return cactus_vulkan_op_enabled("rms")
        && cactus_vulkan_encode_rms_norm_add(out, in, w, res, rows, dim, eps, out_scale);
}
inline bool cactus_gpu_encode_argmax(const void* logits, uint32_t vocab, void* out3, const void* bias) {
    return cactus_vulkan_op_enabled("argmax") && cactus_vulkan_encode_argmax(logits, vocab, out3, bias);
}
inline bool cactus_gpu_encode_softcap(void* out, const void* in, size_t n, float cap) {
    return cactus_vulkan_op_enabled("softcap") && cactus_vulkan_encode_softcap(out, in, n, cap);
}
inline bool cactus_gpu_encode_adjust_logits(void* logits, size_t vocab, const uint32_t* recent,
        uint32_t n_recent, int64_t suppressed, float penalty) {
    return cactus_vulkan_op_enabled("argmax")
        && cactus_vulkan_encode_adjust_logits(logits, vocab, recent, n_recent, suppressed, penalty);
}
inline bool cactus_gpu_encode_cast(void* out, int out_prec, const void* in, int in_prec, size_t n) {
    return cactus_vulkan_op_enabled("cast") && cactus_vulkan_encode_cast(out, out_prec, in, in_prec, n);
}
inline bool cactus_gpu_encode_quant_matmul_m(void* out, const void* lhs, const CactusQuantMatrix* W, uint32_t M) {
    return cactus_vulkan_op_enabled("matmul") && cactus_vulkan_encode_quant_matmul_m(out, lhs, W, M);
}
inline bool cactus_gpu_encode_transform_batch(const void* x, const CactusQuantMatrix* const* Ws, int B, void* const* codes) {
    return cactus_vulkan_op_enabled("matmul") && cactus_vulkan_encode_transform_batch(x, Ws, B, codes);
}
inline bool cactus_gpu_encode_gemv_precoded(void* out, const void* code, const CactusQuantMatrix* W) {
    return cactus_vulkan_op_enabled("matmul") && cactus_vulkan_encode_gemv_precoded(out, code, W);
}
inline bool cactus_gpu_encode_transform_gemv(void* out, const void* x, const CactusQuantMatrix* W, const void* osw) {
    return cactus_vulkan_op_enabled("matmul") && cactus_vulkan_encode_transform_gemv(out, x, W, osw);
}
inline bool cactus_gpu_transform_gemv_fits(uint32_t K) { return cactus_vulkan_transform_gemv_fits(K); }
inline bool cactus_gpu_encode_gemv_cat(void* const* o, const void* const* c, const CactusQuantMatrix* const* w, int b) { return cactus_vulkan_encode_gemv_cat(o, c, w, b); }
inline bool cactus_gpu_encode_swiglu_transform(void* code, const void* gate, const void* up,
        const CactusQuantMatrix* W, float scale) {
    return cactus_vulkan_op_enabled("matmul")
        && cactus_vulkan_encode_swiglu_transform(code, gate, up, W, scale);
}
inline bool cactus_gpu_prewarm_quant(const CactusQuantMatrix* W) {
    return cactus_vulkan_prewarm_quant(W);
}
inline bool cactus_gpu_encode_rope_pair(void* out, const void* x, const void* c, const void* s,
        uint32_t H, uint32_t D) {
    return cactus_vulkan_op_enabled("rope") && cactus_vulkan_encode_rope_pair(out, x, c, s, H, D);
}
inline bool cactus_gpu_encode_rope_pair_rms(void* out, const void* x, const void* w, const void* c,
        const void* s, uint32_t H, uint32_t D, float eps) {
    return cactus_vulkan_op_enabled("rope")
        && cactus_vulkan_encode_rope_pair_rms(out, x, w, c, s, H, D, eps);
}
inline bool cactus_gpu_encode_deltanet_decode(void* out, const void* q, const void* k, const void* v,
        const void* g, const void* b, const void* s, uint32_t B, uint32_t Hq, uint32_t Hv,
        uint32_t K, uint32_t V, float scale) {
    return cactus_vulkan_op_enabled("deltanet")
        && cactus_vulkan_encode_deltanet_decode(out, q, k, v, g, b, s, B, Hq, Hv, K, V, scale);
}
inline bool cactus_gpu_encode_deltanet_prefill(void* out, const void* q, const void* k, const void* v,
        const void* g, const void* b, const void* s, uint32_t B, uint32_t T, uint32_t Hq, uint32_t Hv,
        uint32_t K, uint32_t V, float scale) {
    return cactus_vulkan_op_enabled("deltanet")
        && cactus_vulkan_encode_deltanet_prefill(out, q, k, v, g, b, s, B, T, Hq, Hv, K, V, scale);
}
inline bool cactus_gpu_encode_rms2_add_clip(void* out, const void* a, const void* wa,
        const void* b, const void* wb, size_t dim, float eps_a, float eps_b) {
    return cactus_vulkan_op_enabled("rms")
        && cactus_vulkan_encode_rms2_add_clip(out, a, wa, b, wb, dim, eps_a, eps_b);
}
inline bool cactus_gpu_encode_rms_norm_scale(void* out, const void* in, const void* w,
        size_t rows, size_t dim, float eps, float oscale) {
    return cactus_vulkan_op_enabled("rms")
        && cactus_vulkan_encode_rms_norm_scale(out, in, w, rows, dim, eps, oscale);
}
inline bool cactus_gpu_encode_softmax_topk(void* probs, void* topk, const void* in,
        size_t rows, size_t cols, size_t k, float scale) {
    return cactus_vulkan_op_enabled("topk")
        && cactus_vulkan_encode_softmax_topk(probs, topk, in, rows, cols, k, scale);
}
inline bool cactus_gpu_encode_topk_rows(void* out, const void* in, size_t rows, size_t cols, size_t k) {
    return cactus_vulkan_op_enabled("topk")
        && cactus_vulkan_encode_topk_rows(out, in, rows, cols, k);
}
inline bool cactus_gpu_moe_cq4_ready(const CactusQuantMatrix* w) {
    return cactus_vulkan_moe_cq4_ready(w);
}
inline bool cactus_gpu_moe_cq4_build(const CactusQuantMatrix* w1s, const CactusQuantMatrix* w3s,
        const CactusQuantMatrix* w2s, uint32_t E) {
    return cactus_vulkan_moe_cq4_build(w1s, w3s, w2s, E);
}
inline bool cactus_gpu_encode_moe_gated_cq4(void* out, const void* hidden, const void* probs,
        const void* topk, const CactusQuantMatrix* w1_0, uint32_t E, uint32_t top_k, uint32_t tokens,
        uint32_t act, uint32_t normalize, float eps, float scaling) {
    return cactus_vulkan_op_enabled("moe")
        && cactus_vulkan_encode_moe_gated_cq4(out, hidden, probs, topk, w1_0, E, top_k, tokens,
               act, normalize, eps, scaling);
}
inline bool cactus_gpu_encode_quant_matmul_ortho(void* out, const void* act, void* code,
        const CactusQuantMatrix* W) {
    (void)code;
    return cactus_vulkan_op_enabled("matmul") && cactus_vulkan_encode_cq_gemv(out, act, W);
}
inline bool cactus_gpu_encode_embedding_ortho(void* out, uint32_t row, const CactusQuantMatrix* W, float scale) {
    return cactus_vulkan_op_enabled("emb") && cactus_vulkan_encode_embedding_ortho(out, row, W, scale);
}
inline bool cactus_gpu_encode_embedding_hadamard(void* out, uint32_t row, const CactusQuantMatrix* W) {
    return cactus_vulkan_op_enabled("emb") && cactus_vulkan_encode_embedding_hadamard(out, row, W);
}
inline bool cactus_gpu_encode_embedding_ortho_m(void* out, const CactusQuantMatrix* W, const uint32_t* rows, uint32_t M) {
    return cactus_vulkan_op_enabled("emb") && cactus_vulkan_encode_embedding_ortho_m(out, W, rows, M);
}
inline bool cactus_gpu_encode_embedding_hadamard_m(void* out, const CactusQuantMatrix* W, const uint32_t* rows, uint32_t M) {
    return cactus_vulkan_op_enabled("emb") && cactus_vulkan_encode_embedding_hadamard_m(out, W, rows, M);
}
inline bool cactus_gpu_encode_gather_f16(void* out, const void* table, size_t table_bytes,
        const uint32_t* rows, uint32_t M, uint32_t D) {
    return cactus_vulkan_op_enabled("gather")
        && cactus_vulkan_encode_gather_f16(out, table, table_bytes, rows, M, D);
}
inline bool cactus_gpu_encode_attention_i8(void* out, const void* q, const void* knew, const void* vnew,
        const void* kc, const void* vc, const void* ks, const void* vs,
        uint32_t nqh, uint32_t nkvh, uint32_t hd, uint32_t vhd,
        uint32_t hist, uint32_t total_keys, uint32_t kv_start, uint32_t kv_end,
        float scale, size_t kcb, size_t vcb, size_t ksb, size_t vsb) {
    return cactus_vulkan_op_enabled("attn")
        && cactus_vulkan_encode_attention_i8(out, q, knew, vnew, kc, vc, ks, vs,
               nqh, nkvh, hd, vhd, hist, total_keys, kv_start, kv_end, scale, kcb, vcb, ksb, vsb);
}
inline bool cactus_gpu_encode_attention_fused_i8(void* out, const void* q, const void* kraw, const void* vraw,
        void* kc, void* vc, void* ks, void* vs,
        const void* qw, const void* kw, const void* vw, const void* cs, const void* sn,
        uint32_t nqh, uint32_t hd, uint32_t vhd,
        uint32_t kv_start, uint32_t kv_end, uint32_t slot, uint32_t has_new,
        float eps, float scale, size_t kcb, size_t vcb, size_t ksb, size_t vsb) {
    return cactus_vulkan_op_enabled("attn")
        && cactus_vulkan_encode_attention_fused_i8(out, q, kraw, vraw, kc, vc, ks, vs,
               qw, kw, vw, cs, sn, nqh, hd, vhd, kv_start, kv_end, slot, has_new,
               eps, scale, kcb, vcb, ksb, vsb);
}
inline bool cactus_gpu_encode_attention_i8_prefill(void* out, const void* q, const void* knew, const void* vnew,
        const void* kc, const void* vc, const void* ks, const void* vs,
        uint32_t nqh, uint32_t nkvh, uint32_t hd, uint32_t vhd,
        uint32_t hist, uint32_t new_len, uint32_t q_pos0, uint32_t window, uint32_t is_causal,
        uint32_t M, float scale, size_t kcb, size_t vcb, size_t ksb, size_t vsb,
        uint32_t sink, uint32_t ring) {
    return cactus_vulkan_op_enabled("attn")
        && cactus_vulkan_encode_attention_i8_prefill(out, q, knew, vnew, kc, vc, ks, vs,
               nqh, nkvh, hd, vhd, hist, new_len, q_pos0, window, is_causal, M, scale,
               kcb, vcb, ksb, vsb, sink, ring);
}
inline bool cactus_gpu_encode_binary_f32(int op, void* y, const void* a, const void* b, size_t n) {
    return cactus_vulkan_op_enabled("ew") && cactus_vulkan_encode_binary_f32(op, y, a, b, n);
}
inline bool cactus_gpu_encode_scalar_f32(int op, void* y, const void* in, size_t n, float p) {
    return cactus_vulkan_op_enabled("ew") && cactus_vulkan_encode_scalar_f32(op, y, in, n, p);
}
inline bool cactus_gpu_encode_unary_f32(int op, void* y, const void* in, size_t n) {
    return cactus_vulkan_op_enabled("ew") && cactus_vulkan_encode_unary_f32(op, y, in, n);
}
inline bool cactus_gpu_encode_clamp(void* out, const void* in, size_t n, float lo, float hi, int f32) {
    return cactus_vulkan_op_enabled("ew") && cactus_vulkan_encode_clamp(out, in, n, lo, hi, f32);
}
inline bool cactus_gpu_encode_glu(void* out, const void* in, size_t split, size_t inner, size_t n_out) {
    return cactus_vulkan_op_enabled("glu") && cactus_vulkan_encode_glu(out, in, split, inner, n_out);
}
inline bool cactus_gpu_encode_layer_norm(void* out, const void* in, const void* w, const void* b,
        size_t rows, size_t dim, float eps) {
    return cactus_vulkan_op_enabled("norm") && cactus_vulkan_encode_layer_norm(out, in, w, b, rows, dim, eps);
}
inline bool cactus_gpu_encode_softmax_rows(void* out, const void* in, size_t rows, size_t cols) {
    return cactus_vulkan_op_enabled("softmax") && cactus_vulkan_encode_softmax_rows(out, in, rows, cols);
}
inline bool cactus_gpu_encode_conv1d_k3(void* out, const void* x, const void* w, int w_int8,
        const void* w_scales, uint32_t w_gs, uint32_t Cin, uint32_t L, uint32_t Cout, uint32_t Lout, uint32_t stride) {
    return cactus_vulkan_op_enabled("conv")
        && cactus_vulkan_encode_conv1d_k3(out, x, w, w_int8, w_scales, w_gs, Cin, L, Cout, Lout, stride);
}
inline bool cactus_gpu_encode_gemm_f16(void* out, const void* lhs, const void* rhs,
        uint32_t M, uint32_t K, uint32_t N, int pretransposed) {
    return cactus_vulkan_op_enabled("gemm")
        && cactus_vulkan_encode_gemm_f16(out, lhs, rhs, M, K, N, pretransposed);
}
inline bool cactus_gpu_encode_attention_f16(void* out, const void* q, const void* k, const void* v,
        const void* mask, uint32_t B, uint32_t T, uint32_t S, uint32_t HQ, uint32_t HKV,
        uint32_t D, uint32_t DV, float scale, uint32_t causal, uint32_t pos_off,
        uint32_t window, float logit_cap, uint32_t mask_mode) {
    return cactus_vulkan_op_enabled("attn")
        && cactus_vulkan_encode_attention_f16(out, q, k, v, mask, B, T, S, HQ, HKV, D, DV,
               scale, causal, pos_off, window, logit_cap, mask_mode);
}
inline bool cactus_gpu_encode_reduce_axis(int op, void* out, const void* in, uint32_t outer,
        uint32_t axis_size, uint32_t inner, int f32) {
    return cactus_vulkan_op_enabled("reduce")
        && cactus_vulkan_encode_reduce_axis(op, out, in, outer, axis_size, inner, f32);
}
inline bool cactus_gpu_encode_cumsum(void* out, const void* in, uint32_t outer,
        uint32_t axis_size, uint32_t inner, int f32) {
    return cactus_vulkan_op_enabled("reduce")
        && cactus_vulkan_encode_cumsum(out, in, outer, axis_size, inner, f32);
}
inline bool cactus_gpu_encode_concat2(void* out, const void* a, const void* b,
        uint32_t a_outer, uint32_t b_outer, uint32_t a_axis, uint32_t b_axis, uint32_t inner) {
    return cactus_vulkan_op_enabled("concat")
        && cactus_vulkan_encode_concat2(out, a, b, a_outer, b_outer, a_axis, b_axis, inner);
}
inline bool cactus_gpu_encode_gather_f32idx(void* out, const void* table, const void* idx,
        uint32_t rows, uint32_t D, size_t table_bytes) {
    return cactus_vulkan_op_enabled("gather")
        && cactus_vulkan_encode_gather_f32idx(out, table, idx, rows, D, table_bytes);
}
inline bool cactus_gpu_encode_rope_full(void* out, const void* in, uint32_t tokens, uint32_t S,
        uint32_t H, uint32_t D, uint32_t rot, uint32_t pos0, float theta, int gptj) {
    return cactus_vulkan_op_enabled("rope")
        && cactus_vulkan_encode_rope_full(out, in, tokens, S, H, D, rot, pos0, theta, gptj);
}
inline bool cactus_gpu_encode_maxpool1d(void* out, const void* in, uint32_t NC, uint32_t L,
        uint32_t Lout, uint32_t K, uint32_t stride) {
    return cactus_vulkan_op_enabled("pool") && cactus_vulkan_encode_maxpool1d(out, in, NC, L, Lout, K, stride);
}
inline bool cactus_gpu_encode_bilinear(void* out, const void* in, uint32_t sh, uint32_t sw,
        uint32_t dh, uint32_t dw, uint32_t E, int align) {
    return cactus_vulkan_op_enabled("image") && cactus_vulkan_encode_bilinear(out, in, sh, sw, dh, dw, E, align);
}
inline bool cactus_gpu_encode_conv1d_gen(void* out, const void* x, const void* w, const void* bias,
        uint32_t N, uint32_t Cin, uint32_t L, uint32_t Cout, uint32_t Lout, uint32_t K, uint32_t stride, int w_ck_co) {
    return cactus_vulkan_op_enabled("conv")
        && cactus_vulkan_encode_conv1d_gen(out, x, w, bias, N, Cin, L, Cout, Lout, K, stride, w_ck_co);
}
inline bool cactus_gpu_encode_conv1d_nlc_dw(void* out, const void* x, const void* w, const void* bias,
        uint32_t N, uint32_t L, uint32_t C, uint32_t K, uint32_t dil, uint32_t pad) {
    return cactus_vulkan_op_enabled("conv")
        && cactus_vulkan_encode_conv1d_nlc_dw(out, x, w, bias, N, L, C, K, dil, pad);
}
inline bool cactus_gpu_encode_conv2d(void* out, const void* x, const void* w, const void* bias,
        uint32_t N, uint32_t Cin, uint32_t H, uint32_t W, uint32_t Cout, uint32_t Ho, uint32_t Wo,
        uint32_t K, uint32_t stride, uint32_t pad, int dw) {
    return cactus_vulkan_op_enabled("conv")
        && cactus_vulkan_encode_conv2d(out, x, w, bias, N, Cin, H, W, Cout, Ho, Wo, K, stride, pad, dw);
}
inline bool cactus_gpu_encode_batchnorm(void* out, const void* x, const void* w, const void* b,
        const void* rm, const void* rv, uint32_t C, uint32_t inner, uint32_t total, float eps) {
    return cactus_vulkan_op_enabled("norm")
        && cactus_vulkan_encode_batchnorm(out, x, w, b, rm, rv, C, inner, total, eps);
}
inline bool cactus_gpu_encode_groupnorm(void* out, const void* x, const void* w, const void* b,
        uint32_t N, uint32_t C, uint32_t S, uint32_t groups, float eps) {
    return cactus_vulkan_op_enabled("norm")
        && cactus_vulkan_encode_groupnorm(out, x, w, b, N, C, S, groups, eps);
}
inline bool cactus_gpu_encode_bias_add_rows(void* y, const void* bias, uint32_t C, uint32_t total) {
    return cactus_vulkan_op_enabled("bias") && cactus_vulkan_encode_bias_add_rows(y, bias, C, total);
}
inline bool cactus_gpu_encode_elemwise_chain(void* out, const void* in, const float* steps,
        uint32_t nsteps, const void* side0, const void* side1, const void* side2,
        const size_t* side_elems, size_t n, uint32_t flags, uint32_t inner) {
    return cactus_vulkan_op_enabled("chain")
        && cactus_vulkan_encode_elemwise_chain(out, in, steps, nsteps, side0, side1, side2,
               side_elems, n, flags, inner);
}
inline bool cactus_gpu_encode_rms_norm_add_rows(void* ysum, void* ynorm, const void* x, const void* res,
        const void* w, uint32_t rows, uint32_t dim, float eps, int clipped) {
    return cactus_vulkan_op_enabled("rms")
        && cactus_vulkan_encode_rms_norm_add_rows(ysum, ynorm, x, res, w, rows, dim, eps, clipped);
}
inline bool cactus_gpu_encode_gemm_batch(void* out, const void* a, const void* b,
        uint32_t M, uint32_t K, uint32_t N, uint32_t batch, int f32out, int f32a) {
    return cactus_vulkan_op_enabled("gemm")
        && cactus_vulkan_encode_gemm_batch(out, a, b, M, K, N, batch, f32out, f32a);
}
inline bool cactus_gpu_encode_conv1d_dw(void* out, const void* x, const void* w,
        uint32_t C, uint32_t L, uint32_t Lout, uint32_t K, uint32_t stride) {
    return cactus_vulkan_op_enabled("conv") && cactus_vulkan_encode_conv1d_dw(out, x, w, C, L, Lout, K, stride);
}
inline bool cactus_gpu_encode_transpose2d(void* out, const void* in, uint32_t batch, uint32_t R, uint32_t C) {
    uint32_t oshape[3] = {batch, C, R}, sstride[3] = {R * C, 1, C};
    return cactus_vulkan_op_enabled("strided")
        && cactus_vulkan_encode_strided_copy(out, in, oshape, sstride, 3, batch * R * C, 0,
               (size_t)batch * R * C * 2, (size_t)batch * R * C * 2);
}
inline bool cactus_gpu_encode_strided_copy(void* out, const void* in, const uint32_t* oshape,
        const uint32_t* sstride, uint32_t ndim, uint32_t total, uint32_t base, size_t ib, size_t ob) {
    return cactus_vulkan_op_enabled("strided")
        && cactus_vulkan_encode_strided_copy(out, in, oshape, sstride, ndim, total, base, ib, ob);
}
inline bool cactus_gpu_encode_bcast_binary(int op, void* out, const void* a, const void* b,
        const uint32_t* oshape, const uint32_t* astride, const uint32_t* bstride, uint32_t ndim, uint32_t total,
        size_t ab, size_t bb, size_t ob) {
    return cactus_vulkan_op_enabled("bcast")
        && cactus_vulkan_encode_bcast_binary(op, out, a, b, oshape, astride, bstride, ndim, total, ab, bb, ob);
}
inline bool cactus_gpu_encode_strided_scatter(void* out, const void* in, const uint32_t* ishape,
        const uint32_t* ostride, uint32_t ndim, uint32_t total, uint32_t base, size_t ib, size_t ob) {
    return cactus_vulkan_op_enabled("strided")
        && cactus_vulkan_encode_strided_scatter(out, in, ishape, ostride, ndim, total, base, ib, ob);
}
inline bool cactus_gpu_encode_kv_append_i8(const void* src, void* i8b, void* scb,
        uint32_t kvh, uint32_t hdim, uint32_t cur, uint32_t gs, size_t sb, size_t ib, size_t scb_sz) {
    return cactus_vulkan_op_enabled("kv")
        && cactus_vulkan_encode_kv_append_i8(src, i8b, scb, kvh, hdim, cur, gs, 1, 0, 0, sb, ib, scb_sz);
}
inline bool cactus_gpu_encode_kv_append_sliding_i8(const void* src, void* i8b, void* scb,
        uint32_t kvh, uint32_t hdim, uint32_t keep_sink, uint32_t remaining, uint32_t shift_src,
        uint32_t gs, size_t sb, size_t ib, size_t scb_sz) {
    return cactus_vulkan_op_enabled("kv")
        && cactus_vulkan_encode_kv_append_sliding_i8(src, i8b, scb, kvh, hdim, keep_sink, remaining,
               shift_src, gs, 1, sb, ib, scb_sz);
}
inline bool cactus_gpu_encode_kv_append_sliding_i8_m(const void* src, void* i8b, void* scb,
        uint32_t kvh, uint32_t hdim, uint32_t keep_sink, uint32_t remaining, uint32_t shift_src,
        uint32_t gs, uint32_t M, size_t sb, size_t ib, size_t scb_sz) {
    return cactus_vulkan_op_enabled("kv")
        && cactus_vulkan_encode_kv_append_sliding_i8(src, i8b, scb, kvh, hdim, keep_sink, remaining,
               shift_src, gs, M, sb, ib, scb_sz);
}
inline bool cactus_gpu_encode_kv_append_i8_m(const void* src, void* i8b, void* scb,
        uint32_t kvh, uint32_t hdim, uint32_t cur, uint32_t gs, uint32_t M, size_t sb, size_t ib, size_t scb_sz) {
    return cactus_vulkan_op_enabled("kv")
        && cactus_vulkan_encode_kv_append_i8(src, i8b, scb, kvh, hdim, cur, gs, M, 0, 0, sb, ib, scb_sz);
}
inline bool cactus_gpu_encode_kv_append_ring_i8_m(const void* src, void* i8b, void* scb,
        uint32_t kvh, uint32_t hdim, uint32_t cur, uint32_t gs, uint32_t M, uint32_t sink, uint32_t W,
        size_t sb, size_t ib, size_t scb_sz) {
    return cactus_vulkan_op_enabled("kv")
        && cactus_vulkan_encode_kv_append_i8(src, i8b, scb, kvh, hdim, cur, gs, M, sink, W, sb, ib, scb_sz);
}
inline bool cactus_gpu_encode_conv_cache_append(void* out, const void* src, void* ring,
        uint32_t hd, uint32_t ws, uint32_t nnew, uint32_t head0, uint32_t count_new,
        uint32_t num_rows, int src_f32) {
    return cactus_vulkan_op_enabled("conv")
        && cactus_vulkan_encode_conv_cache_append(out, src, ring, hd, ws, nnew, head0, count_new, num_rows, src_f32);
}
inline bool cactus_gpu_encode_rel_pos_bias(void* y, const void* q, const void* r,
        uint32_t B, uint32_t T, uint32_t H, uint32_t D, uint32_t R, int r_batched, float scale) {
    return cactus_vulkan_op_enabled("attn")
        && cactus_vulkan_encode_rel_pos_bias(y, q, r, B, T, H, D, R, r_batched, scale);
}
inline bool cactus_gpu_encode_gemv_bias(void* out, const void* x, const void* w, const void* bias,
        uint32_t K, uint32_t N, int tr) {
    return cactus_vulkan_op_enabled("matmul")
        && cactus_vulkan_encode_gemv_bias(out, x, w, bias, K, N, tr);
}

#endif

#endif
