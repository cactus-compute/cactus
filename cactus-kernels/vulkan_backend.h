#ifndef CACTUS_VULKAN_BACKEND_H
#define CACTUS_VULKAN_BACKEND_H

#include "cactus_kernels.h"
#include <cstddef>
#include <cstdint>

bool cactus_vulkan_available();
const char* cactus_vulkan_device_info();
bool cactus_vulkan_op_enabled(const char* name);

void cactus_vulkan_session_begin();
void cactus_vulkan_session_flush();
void cactus_vulkan_session_sync();
void cactus_vulkan_session_end();
void cactus_vulkan_invalidate_host_wraps();
void cactus_vulkan_trim_prefill_cache();

void* cactus_vulkan_alloc_shared(size_t bytes);
void* cactus_vulkan_alloc_pooled(size_t bytes);
void cactus_vulkan_free_shared(void* p);

bool cactus_vulkan_encode_binary_f16(int op, void* y, const void* a, const void* b, size_t n);
bool cactus_vulkan_encode_scalar_f16(int op, void* y, const void* in, size_t n, float p);
bool cactus_vulkan_encode_unary_f16(int op, void* y, const void* in, size_t n);
bool cactus_vulkan_encode_swiglu_f16(void* y, const void* gate, const void* up, size_t n, float scale);
bool cactus_vulkan_encode_rms_norm_f16(void* y, const void* in, const void* w,
                                       size_t rows, size_t dim, float eps);
bool cactus_vulkan_encode_cq_gemv(void* y, const void* x, const CactusQuantMatrix* W);
bool cactus_vulkan_encode_copy(void* out, const void* in, size_t bytes);
bool cactus_vulkan_encode_cast(void* out, int out_prec, const void* in, int in_prec, size_t n);
bool cactus_vulkan_encode_strided_copy(void* out, const void* in, const uint32_t* oshape,
    const uint32_t* sstride, uint32_t ndim, uint32_t total, uint32_t base, size_t in_bytes, size_t out_bytes);
bool cactus_vulkan_encode_strided_scatter(void* out, const void* in, const uint32_t* ishape,
    const uint32_t* ostride, uint32_t ndim, uint32_t total, uint32_t base, size_t in_bytes, size_t out_bytes);
bool cactus_vulkan_encode_bcast_binary(int op, void* out, const void* a, const void* b,
    const uint32_t* oshape, const uint32_t* astride, const uint32_t* bstride, uint32_t ndim, uint32_t total,
    size_t a_bytes, size_t b_bytes, size_t out_bytes);
bool cactus_vulkan_encode_concat2(void* out, const void* a, const void* b,
    uint32_t a_outer, uint32_t b_outer, uint32_t a_axis, uint32_t b_axis, uint32_t inner);
bool cactus_vulkan_encode_rms_norm_add(void* out, const void* in, const void* w, const void* res,
    size_t rows, size_t dim, float eps, float out_scale);
bool cactus_vulkan_encode_rms_norm_add_rms(void* h_out, void* xn_out, const void* in, const void* w1,
    const void* res, const void* w2, size_t rows, size_t dim, float eps, float out_scale);
bool cactus_vulkan_encode_rms_norm_scale(void* out, const void* in, const void* w,
    size_t rows, size_t dim, float eps, float oscale);
bool cactus_vulkan_encode_rope_full(void* out, const void* in, uint32_t tokens, uint32_t S,
    uint32_t H, uint32_t D, uint32_t rot, uint32_t pos0, float theta, int gptj);
bool cactus_vulkan_encode_softcap(void* out, const void* in, size_t n, float cap);
bool cactus_vulkan_encode_adjust_logits(void* logits, size_t vocab, const uint32_t* recent,
    uint32_t n_recent, int64_t suppressed, float penalty);
bool cactus_vulkan_encode_argmax(const void* logits, uint32_t vocab, void* out3, const void* bias);
bool cactus_vulkan_encode_kv_append_i8(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t current_len, uint32_t group_size, uint32_t M,
    uint32_t sink, uint32_t W, size_t src_bytes, size_t int8_bytes, size_t scale_bytes);
bool cactus_vulkan_encode_attention_i8(
    void* out, const void* q, const void* knew, const void* vnew,
    const void* kc, const void* vc, const void* ks, const void* vs,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t v_hdim,
    uint32_t history_len, uint32_t total_keys, uint32_t kv_start, uint32_t kv_end,
    float scale, size_t kc_bytes, size_t vc_bytes, size_t ks_bytes, size_t vs_bytes);

bool cactus_vulkan_binary_f16(int op, __fp16* out, const __fp16* a, const __fp16* b, size_t n);
bool cactus_vulkan_scalar_f16(int op, __fp16* out, const __fp16* in, size_t n, float p);
bool cactus_vulkan_unary_f16(int op, __fp16* out, const __fp16* in, size_t n);
bool cactus_vulkan_swiglu_f16(__fp16* out, const __fp16* gate, const __fp16* up, size_t n, float scale);
bool cactus_vulkan_rms_norm_f16(__fp16* out, const __fp16* in, const __fp16* weight,
                                size_t rows, size_t dim, float eps);
bool cactus_vulkan_cq_gemv(__fp16* y, const __fp16* x, const CactusQuantMatrix* W,
                           int iters = 1, double* kernel_ms = nullptr);
bool cactus_vulkan_argmax_f16(const __fp16* logits, uint32_t n, uint32_t* idx, float* best);

#endif
