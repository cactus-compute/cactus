#ifndef CACTUS_GPU_H
#define CACTUS_GPU_H

#include "cactus_kernels.h"
#include <cstdint>

#if !defined(__ANDROID__)

#include "metal_backend.h"

inline bool cactus_gpu_supports_plans() { return true; }
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

inline bool cactus_gpu_supports_plans() { return false; }
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
    return cactus_vulkan_op_enabled("ew") && cactus_vulkan_encode_binary_f16(op, out, a, b, n);
}
inline bool cactus_gpu_encode_scalar(int op, void* out, const void* in, size_t n, float p) {
    return cactus_vulkan_op_enabled("ew") && cactus_vulkan_encode_scalar_f16(op, out, in, n, p);
}
inline bool cactus_gpu_encode_unary(int op, void* out, const void* in, size_t n) {
    return cactus_vulkan_op_enabled("ew") && cactus_vulkan_encode_unary_f16(op, out, in, n);
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

inline bool cactus_gpu_encode_copy(void*, const void*, size_t) { return false; }
inline bool cactus_gpu_encode_rms_norm_add(void*, const void*, const void*, const void*, size_t, size_t, float) { return false; }
inline bool cactus_gpu_encode_rms_norm_add_rms(void*, void*, const void*, const void*, const void*, const void*, size_t, size_t, float, float) { return false; }
inline bool cactus_gpu_encode_rms_norm_add_scale(void*, const void*, const void*, const void*, size_t, size_t, float, float) { return false; }
inline bool cactus_gpu_encode_argmax(const void*, uint32_t, void*, const void*) { return false; }
inline bool cactus_gpu_encode_softcap(void*, const void*, size_t, float) { return false; }
inline bool cactus_gpu_encode_adjust_logits(void*, size_t, const uint32_t*, uint32_t, int64_t, float) { return false; }
inline bool cactus_gpu_encode_cast(void*, int, const void*, int, size_t) { return false; }
inline bool cactus_gpu_encode_quant_matmul_m(void*, const void*, const CactusQuantMatrix*, uint32_t) { return false; }
inline bool cactus_gpu_encode_transform_batch(const void*, const CactusQuantMatrix* const*, int, void* const*) { return false; }
inline bool cactus_gpu_encode_gemv_precoded(void*, const void*, const CactusQuantMatrix*) { return false; }
inline bool cactus_gpu_encode_transform_gemv(void*, const void*, const CactusQuantMatrix*, const void*) { return false; }
inline bool cactus_gpu_transform_gemv_fits(uint32_t) { return false; }
inline bool cactus_gpu_encode_gemv_cat(void* const*, const void* const*, const CactusQuantMatrix* const*, int) { return false; }
inline bool cactus_gpu_encode_swiglu_transform(void*, const void*, const void*, const CactusQuantMatrix*, float) { return false; }
inline bool cactus_gpu_prewarm_quant(const CactusQuantMatrix*) { return false; }
inline bool cactus_gpu_encode_rope_pair(void*, const void*, const void*, const void*, uint32_t, uint32_t) { return false; }
inline bool cactus_gpu_encode_rope_pair_rms(void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, float) { return false; }
inline bool cactus_gpu_encode_deltanet_decode(void*, const void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float) { return false; }
inline bool cactus_gpu_encode_deltanet_prefill(void*, const void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float) { return false; }
inline bool cactus_gpu_encode_rms2_add_clip(void*, const void*, const void*, const void*, const void*, size_t, float, float) { return false; }
inline bool cactus_gpu_encode_rms_norm_scale(void*, const void*, const void*, size_t, size_t, float, float) { return false; }
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
inline bool cactus_gpu_encode_attention_i8(void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, size_t, size_t, size_t, size_t) { return false; }
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
inline bool cactus_gpu_encode_concat2(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
inline bool cactus_gpu_encode_gather_f32idx(void*, const void*, const void*, uint32_t, uint32_t, size_t) { return false; }
inline bool cactus_gpu_encode_rope_full(void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, int) { return false; }
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
inline bool cactus_gpu_encode_transpose2d(void*, const void*, uint32_t, uint32_t, uint32_t) { return false; }
inline bool cactus_gpu_encode_strided_copy(void*, const void*, const uint32_t*, const uint32_t*, uint32_t, uint32_t, uint32_t, size_t, size_t) { return false; }
inline bool cactus_gpu_encode_bcast_binary(int, void*, const void*, const void*, const uint32_t*, const uint32_t*, const uint32_t*, uint32_t, uint32_t, size_t, size_t, size_t) { return false; }
inline bool cactus_gpu_encode_strided_scatter(void*, const void*, const uint32_t*, const uint32_t*, uint32_t, uint32_t, uint32_t, size_t, size_t) { return false; }
inline bool cactus_gpu_encode_kv_append_i8(const void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, size_t, size_t, size_t) { return false; }
inline bool cactus_gpu_encode_kv_append_sliding_i8(const void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, size_t, size_t) { return false; }
inline bool cactus_gpu_encode_kv_append_sliding_i8_m(const void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, size_t, size_t) { return false; }
inline bool cactus_gpu_encode_kv_append_i8_m(const void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, size_t, size_t) { return false; }
inline bool cactus_gpu_encode_kv_append_ring_i8_m(const void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, size_t, size_t) { return false; }
inline bool cactus_gpu_encode_conv_cache_append(void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int) { return false; }
inline bool cactus_gpu_encode_rel_pos_bias(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int, float) { return false; }
inline bool cactus_gpu_encode_gemv_bias(void*, const void*, const void*, const void*, uint32_t, uint32_t, int) { return false; }

#endif

#endif
