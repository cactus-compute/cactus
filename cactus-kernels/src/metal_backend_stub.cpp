
#include "metal_backend.h"

#if !CACTUS_HAS_METAL
bool cactus_metal_available() { return false; }
void cactus_metal_set_active(bool) {}
bool cactus_metal_active_mode() { return false; }
bool cactus_metal_encode_softcap(void*, const void*, size_t, float) { return false; }
bool cactus_metal_encode_adjust_logits(void*, size_t, const uint32_t*, uint32_t,
                                       int64_t, float) { return false; }
bool cactus_metal_encode_transform_gemv(void*, const void*, const CactusQuantMatrix*, const void*) { return false; }
bool cactus_metal_transform_gemv_fits(uint32_t) { return false; }
bool cactus_metal_encode_gemv_cat(void* const*, const void* const*,
                                  const CactusQuantMatrix* const*, int) { return false; }
bool cactus_metal_encode_swiglu_transform(void*, const void*, const void*,
                                          const CactusQuantMatrix*, float) { return false; }
void cactus_metal_quant_matmul(const CactusQuantMatrix* W, const __fp16* A,
                               uint32_t M, __fp16* C) {
    cactus_quant_matmul(W, A, M, C);
}
void  cactus_metal_session_begin() {}
void  cactus_metal_session_sync() {}
void  cactus_metal_session_flush() {}
void  cactus_metal_invalidate_host_wraps() {}
void  cactus_metal_trim_prefill_cache() {}
void  cactus_metal_session_end() {}
void* cactus_metal_alloc_shared(size_t) { return nullptr; }
void* cactus_metal_alloc_pooled(size_t) { return nullptr; }
void  cactus_metal_free_shared(void*) {}
bool cactus_metal_encode_copy(void*, const void*, size_t) { return false; }
bool cactus_metal_encode_binary(int, void*, const void*, const void*, size_t) { return false; }
bool cactus_metal_encode_scalar(int, void*, const void*, size_t, float) { return false; }
bool cactus_metal_encode_unary(int, void*, const void*, size_t) { return false; }
bool cactus_metal_encode_swiglu(void*, const void*, const void*, size_t, float) { return false; }
bool cactus_metal_encode_rms_norm(void*, const void*, const void*, size_t, size_t, float) { return false; }
bool cactus_metal_encode_rms_norm_add(void*, const void*, const void*, const void*, size_t, size_t, float) { return false; }
bool cactus_metal_encode_rms_norm_add_rms(void*, void*, const void*, const void*, const void*, const void*,
                                          size_t, size_t, float, float) { return false; }
bool cactus_metal_encode_rms_norm_add_scale(void*, const void*, const void*, const void*, size_t, size_t, float, float) { return false; }
bool cactus_metal_encode_argmax(const void*, uint32_t, void*, const void*) { return false; }
bool cactus_metal_encode_cast(void*, int, const void*, int, size_t) { return false; }
bool cactus_metal_encode_quant_matmul(void*, const void*, const CactusQuantMatrix*) { return false; }
bool cactus_metal_encode_quant_matmul_m(void*, const void*, const CactusQuantMatrix*, uint32_t) { return false; }
bool cactus_metal_encode_transform_batch(const void*, const CactusQuantMatrix* const*, int, void* const*) { return false; }
bool cactus_metal_encode_gemv_precoded(void*, const void*, const CactusQuantMatrix*) { return false; }
bool cactus_metal_prewarm_quant(const CactusQuantMatrix*) { return false; }
bool cactus_metal_encode_quant_matmul_ortho(void*, const void*, void*, const CactusQuantMatrix*) { return false; }
bool cactus_metal_encode_embedding_ortho(void*, uint32_t, const CactusQuantMatrix*, float) { return false; }
bool cactus_metal_encode_embedding_hadamard(void*, uint32_t, const CactusQuantMatrix*) { return false; }
bool cactus_metal_encode_embedding_ortho_m(void*, const CactusQuantMatrix*, const uint32_t*, uint32_t) { return false; }
bool cactus_metal_encode_embedding_hadamard_m(void*, const CactusQuantMatrix*, const uint32_t*, uint32_t) { return false; }
bool cactus_metal_encode_gather_f16(void*, const void*, size_t, const uint32_t*, uint32_t, uint32_t) { return false; }
bool cactus_metal_encode_attention_i8(void*, const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    float, size_t, size_t, size_t, size_t) { return false; }
bool cactus_metal_encode_attention_fused_i8(void*, const void*, const void*, const void*,
    void*, void*, void*, void*, const void*, const void*, const void*, const void*, const void*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    float, float, size_t, size_t, size_t, size_t) { return false; }
bool cactus_metal_encode_attention_i8_prefill(void*, const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, float, size_t, size_t, size_t, size_t, uint32_t, uint32_t) { return false; }
bool cactus_metal_encode_binary_f32(int, void*, const void*, const void*, size_t) { return false; }
bool cactus_metal_encode_scalar_f32(int, void*, const void*, size_t, float) { return false; }
bool cactus_metal_encode_unary_f32(int, void*, const void*, size_t) { return false; }
bool cactus_metal_encode_clamp(void*, const void*, size_t, float, float, int) { return false; }
bool cactus_metal_encode_glu(void*, const void*, size_t, size_t, size_t) { return false; }
bool cactus_metal_encode_layer_norm(void*, const void*, const void*, const void*, size_t, size_t, float) { return false; }
bool cactus_metal_encode_softmax_rows(void*, const void*, size_t, size_t) { return false; }
bool cactus_metal_encode_conv1d_k3(void*, const void*, const void*, int, const void*, uint32_t,
                                   uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
bool cactus_metal_encode_gemm_f16(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, int) { return false; }
bool cactus_metal_encode_attention_f16(void*, const void*, const void*, const void*, const void*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    float, uint32_t, uint32_t, uint32_t, float, uint32_t) { return false; }
bool cactus_metal_encode_reduce_axis(int, void*, const void*, uint32_t, uint32_t, uint32_t, int) { return false; }
bool cactus_metal_encode_cumsum(void*, const void*, uint32_t, uint32_t, uint32_t, int) { return false; }
bool cactus_metal_encode_concat2(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
bool cactus_metal_encode_gather_f32idx(void*, const void*, const void*, uint32_t, uint32_t, size_t) { return false; }
bool cactus_metal_encode_rope_full(void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, int) { return false; }
bool cactus_metal_encode_maxpool1d(void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
bool cactus_metal_encode_bilinear(void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int) { return false; }
bool cactus_metal_encode_conv1d_gen(void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int) { return false; }
bool cactus_metal_encode_conv1d_nlc_dw(void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
bool cactus_metal_encode_conv2d(void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int) { return false; }
bool cactus_metal_encode_batchnorm(void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, float) { return false; }
bool cactus_metal_encode_groupnorm(void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, float) { return false; }
bool cactus_metal_encode_bias_add_rows(void*, const void*, uint32_t, uint32_t) { return false; }
bool cactus_metal_encode_elemwise_chain(void*, const void*, const float*, uint32_t, const void*, const void*, const void*, const size_t*, size_t, uint32_t, uint32_t) { return false; }
bool cactus_metal_encode_rms_norm_add_rows(void*, void*, const void*, const void*, const void*, uint32_t, uint32_t, float, int) { return false; }
bool cactus_metal_encode_gemm_batch(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, int, int) { return false; }
bool cactus_metal_encode_conv1d_dw(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
bool cactus_metal_encode_transpose2d(void*, const void*, uint32_t, uint32_t, uint32_t) { return false; }
bool cactus_metal_encode_strided_copy(void*, const void*, const uint32_t*, const uint32_t*,
    uint32_t, uint32_t, uint32_t, size_t, size_t) { return false; }
bool cactus_metal_encode_bcast_binary(int, void*, const void*, const void*, const uint32_t*,
    const uint32_t*, const uint32_t*, uint32_t, uint32_t, size_t, size_t, size_t) { return false; }
bool cactus_metal_encode_strided_scatter(void*, const void*, const uint32_t*, const uint32_t*,
    uint32_t, uint32_t, uint32_t, size_t, size_t) { return false; }
bool cactus_metal_encode_kv_append_i8(const void*, void*, void*, uint32_t, uint32_t, uint32_t,
    uint32_t, size_t, size_t, size_t) { return false; }
bool cactus_metal_encode_kv_append_sliding_i8(const void*, void*, void*, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, size_t, size_t, size_t) { return false; }
bool cactus_metal_encode_kv_append_sliding_i8_m(const void*, void*, void*, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, size_t, size_t, size_t) { return false; }
bool cactus_metal_encode_kv_append_i8_m(const void*, void*, void*, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, size_t, size_t, size_t) { return false; }
bool cactus_metal_encode_kv_append_ring_i8_m(const void*, void*, void*, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, size_t, size_t, size_t) { return false; }
bool cactus_metal_encode_conv_cache_append(void*, const void*, void*, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, int) { return false; }
bool cactus_metal_encode_rel_pos_bias(void*, const void*, const void*, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, int, float) { return false; }
bool cactus_metal_encode_gemv_bias(void*, const void*, const void*, const void*,
    uint32_t, uint32_t, int) { return false; }
bool cactus_metal_encode_rope_pair(void*, const void*, const void*, const void*,
    uint32_t, uint32_t) { return false; }
bool cactus_metal_encode_rope_pair_rms(void*, const void*, const void*, const void*,
    const void*, uint32_t, uint32_t, float) { return false; }
bool cactus_metal_encode_deltanet_decode(void*, const void*, const void*, const void*,
    const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float) { return false; }
bool cactus_metal_encode_deltanet_prefill(void*, const void*, const void*, const void*,
    const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float) { return false; }
bool cactus_metal_encode_rms2_add_clip(void*, const void*, const void*,
    const void*, const void*, size_t, float, float) { return false; }
bool cactus_metal_encode_rms_norm_scale(void*, const void*, const void*,
    size_t, size_t, float, float) { return false; }
bool cactus_metal_encode_softmax_topk(void*, void*, const void*,
    size_t, size_t, size_t, float) { return false; }
bool cactus_metal_encode_topk_rows(void*, const void*, size_t, size_t, size_t) { return false; }
bool cactus_metal_moe_cq4_ready(const CactusQuantMatrix*) { return false; }
bool cactus_metal_moe_cq4_build(const CactusQuantMatrix*, const CactusQuantMatrix*,
    const CactusQuantMatrix*, uint32_t) { return false; }
bool cactus_metal_encode_moe_gated_cq4(void*, const void*, const void*, const void*, const CactusQuantMatrix*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float) { return false; }
#endif
