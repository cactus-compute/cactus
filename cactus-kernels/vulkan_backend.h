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
