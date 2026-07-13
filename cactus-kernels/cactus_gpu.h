#ifndef CACTUS_GPU_H
#define CACTUS_GPU_H

#include "cactus_kernels.h"
#include <cstdint>

#if !defined(__ANDROID__)

#include "metal_backend.h"

inline bool cactus_gpu_supports_plans() { return true; }
inline bool cactus_gpu_fold_ready() { return true; }
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

inline bool cactus_gpu_supports_plans() {
    static const bool on = [] {
        const char* v = getenv("CACTUS_VK_PLANS");
        return !(v && v[0] == '0');
    }();
    return on && cactus_vulkan_available();
}
inline bool cactus_gpu_fold_ready() { return false; }
inline const char* cactus_gpu_default_rules() { return "2,3,4,12,13,14,15,17"; }
inline bool cactus_gpu_auto_available() { return false; }

inline bool cactus_gpu_available() { return cactus_vulkan_available(); }
inline void cactus_gpu_set_active(bool) {}
inline bool cactus_gpu_active_mode() { return cactus_vulkan_available(); }
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
inline bool cactus_gpu_encode_transform_gemv(void*, const void*, const CactusQuantMatrix*, const void*) { return false; }
inline bool cactus_gpu_transform_gemv_fits(uint32_t) { return false; }
inline bool cactus_gpu_encode_gemv_cat(void* const*, const void* const*, const CactusQuantMatrix* const*, int) { return false; }
inline bool cactus_gpu_encode_swiglu_transform(void*, const void*, const void*, const CactusQuantMatrix*, float) { return false; }
inline bool cactus_gpu_prewarm_quant(const CactusQuantMatrix* W) {
    return cactus_vulkan_prewarm_quant(W);
}
inline bool cactus_gpu_encode_rope_pair(void* out, const void* x, const void* c, const void* s,
        uint32_t H, uint32_t D) {
    return cactus_vulkan_op_enabled("rope") && cactus_vulkan_encode_rope_pair(out, x, c, s, H, D);
}
inline bool cactus_gpu_encode_rope_pair_rms(void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, float) { return false; }
inline bool cactus_gpu_encode_deltanet_decode(void*, const void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float) { return false; }
inline bool cactus_gpu_encode_deltanet_prefill(void*, const void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float) { return false; }
inline bool cactus_gpu_encode_rms2_add_clip(void*, const void*, const void*, const void*, const void*, size_t, float, float) { return false; }
inline bool cactus_gpu_encode_rms_norm_scale(void* out, const void* in, const void* w,
        size_t rows, size_t dim, float eps, float oscale) {
    return cactus_vulkan_op_enabled("rms")
        && cactus_vulkan_encode_rms_norm_scale(out, in, w, rows, dim, eps, oscale);
}
inline bool cactus_gpu_encode_softmax_topk(void*, void*, const void*, size_t, size_t, size_t, float) { return false; }
inline bool cactus_gpu_encode_topk_rows(void*, const void*, size_t, size_t, size_t) { return false; }
inline bool cactus_gpu_moe_cq4_ready(const CactusQuantMatrix*) { return false; }
inline bool cactus_gpu_moe_cq4_build(const CactusQuantMatrix*, const CactusQuantMatrix*, const CactusQuantMatrix*, uint32_t) { return false; }
inline bool cactus_gpu_encode_moe_gated_cq4(void*, const void*, const void*, const void*, const CactusQuantMatrix*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float) { return false; }
inline bool cactus_gpu_encode_quant_matmul_ortho(void*, const void*, void*, const CactusQuantMatrix*) { return false; }
inline bool cactus_gpu_encode_embedding_ortho(void*, uint32_t, const CactusQuantMatrix*, float) { return false; }
inline bool cactus_gpu_encode_embedding_hadamard(void*, uint32_t, const CactusQuantMatrix*) { return false; }
inline bool cactus_gpu_encode_embedding_ortho_m(void*, const CactusQuantMatrix*, const uint32_t*, uint32_t) { return false; }
inline bool cactus_gpu_encode_embedding_hadamard_m(void*, const CactusQuantMatrix*, const uint32_t*, uint32_t) { return false; }
inline bool cactus_gpu_encode_gather_f16(void*, const void*, size_t, const uint32_t*, uint32_t, uint32_t) { return false; }
inline bool cactus_gpu_encode_attention_i8(void* out, const void* q, const void* knew, const void* vnew,
        const void* kc, const void* vc, const void* ks, const void* vs,
        uint32_t nqh, uint32_t nkvh, uint32_t hd, uint32_t vhd,
        uint32_t hist, uint32_t total_keys, uint32_t kv_start, uint32_t kv_end,
        float scale, size_t kcb, size_t vcb, size_t ksb, size_t vsb) {
    return cactus_vulkan_op_enabled("attn")
        && cactus_vulkan_encode_attention_i8(out, q, knew, vnew, kc, vc, ks, vs,
               nqh, nkvh, hd, vhd, hist, total_keys, kv_start, kv_end, scale, kcb, vcb, ksb, vsb);
}
inline bool cactus_gpu_encode_attention_fused_i8(void*, const void*, const void*, const void*, void*, void*, void*, void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, size_t, size_t, size_t, size_t) { return false; }
inline bool cactus_gpu_encode_attention_i8_prefill(void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, size_t, size_t, size_t, size_t, uint32_t, uint32_t) { return false; }
inline bool cactus_gpu_encode_binary_f32(int, void*, const void*, const void*, size_t) { return false; }
inline bool cactus_gpu_encode_scalar_f32(int, void*, const void*, size_t, float) { return false; }
inline bool cactus_gpu_encode_unary_f32(int, void*, const void*, size_t) { return false; }
inline bool cactus_gpu_encode_clamp(void*, const void*, size_t, float, float, int) { return false; }
inline bool cactus_gpu_encode_glu(void*, const void*, size_t, size_t, size_t) { return false; }
inline bool cactus_gpu_encode_layer_norm(void*, const void*, const void*, const void*, size_t, size_t, float) { return false; }
inline bool cactus_gpu_encode_softmax_rows(void*, const void*, size_t, size_t) { return false; }
inline bool cactus_gpu_encode_conv1d_k3(void*, const void*, const void*, int, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
inline bool cactus_gpu_encode_gemm_f16(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, int) { return false; }
inline bool cactus_gpu_encode_attention_f16(void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t, uint32_t, uint32_t, float, uint32_t) { return false; }
inline bool cactus_gpu_encode_reduce_axis(int, void*, const void*, uint32_t, uint32_t, uint32_t, int) { return false; }
inline bool cactus_gpu_encode_cumsum(void*, const void*, uint32_t, uint32_t, uint32_t, int) { return false; }
inline bool cactus_gpu_encode_concat2(void* out, const void* a, const void* b,
        uint32_t a_outer, uint32_t b_outer, uint32_t a_axis, uint32_t b_axis, uint32_t inner) {
    return cactus_vulkan_op_enabled("concat")
        && cactus_vulkan_encode_concat2(out, a, b, a_outer, b_outer, a_axis, b_axis, inner);
}
inline bool cactus_gpu_encode_gather_f32idx(void*, const void*, const void*, uint32_t, uint32_t, size_t) { return false; }
inline bool cactus_gpu_encode_rope_full(void* out, const void* in, uint32_t tokens, uint32_t S,
        uint32_t H, uint32_t D, uint32_t rot, uint32_t pos0, float theta, int gptj) {
    return cactus_vulkan_op_enabled("rope")
        && cactus_vulkan_encode_rope_full(out, in, tokens, S, H, D, rot, pos0, theta, gptj);
}
inline bool cactus_gpu_encode_maxpool1d(void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
inline bool cactus_gpu_encode_bilinear(void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int) { return false; }
inline bool cactus_gpu_encode_conv1d_gen(void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int) { return false; }
inline bool cactus_gpu_encode_conv1d_nlc_dw(void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
inline bool cactus_gpu_encode_conv2d(void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int) { return false; }
inline bool cactus_gpu_encode_batchnorm(void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, float) { return false; }
inline bool cactus_gpu_encode_groupnorm(void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, float) { return false; }
inline bool cactus_gpu_encode_bias_add_rows(void*, const void*, uint32_t, uint32_t) { return false; }
inline bool cactus_gpu_encode_elemwise_chain(void*, const void*, const float*, uint32_t, const void*, const void*, const void*, const size_t*, size_t, uint32_t, uint32_t) { return false; }
inline bool cactus_gpu_encode_rms_norm_add_rows(void*, void*, const void*, const void*, const void*, uint32_t, uint32_t, float, int) { return false; }
inline bool cactus_gpu_encode_gemm_batch(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, int, int) { return false; }
inline bool cactus_gpu_encode_conv1d_dw(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
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
inline bool cactus_gpu_encode_conv_cache_append(void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int) { return false; }
inline bool cactus_gpu_encode_rel_pos_bias(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int, float) { return false; }
inline bool cactus_gpu_encode_gemv_bias(void*, const void*, const void*, const void*, uint32_t, uint32_t, int) { return false; }

#endif

#endif
