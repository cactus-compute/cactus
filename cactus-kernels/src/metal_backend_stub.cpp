
#include "metal_backend.h"

#if !CACTUS_HAS_METAL
bool cactus_metal_available() { return false; }
void cactus_metal_set_active(bool) {}
bool cactus_metal_active_mode() { return false; }
bool cactus_metal_concurrent() { return false; }
void cactus_metal_barrier() {}
void cactus_metal_quant_matmul(const CactusQuantMatrix* W, const __fp16* A,
                               uint32_t M, __fp16* C) {
    cactus_quant_matmul(W, A, M, C);
}
void  cactus_metal_session_begin() {}
void  cactus_metal_session_sync() {}
void  cactus_metal_session_end() {}
void* cactus_metal_alloc_shared(size_t) { return nullptr; }
void  cactus_metal_free_shared(void*) {}
bool cactus_metal_encode_copy(void*, const void*, size_t) { return false; }
bool cactus_metal_encode_binary(int, void*, const void*, const void*, size_t) { return false; }
bool cactus_metal_encode_scalar(int, void*, const void*, size_t, float) { return false; }
bool cactus_metal_encode_unary(int, void*, const void*, size_t) { return false; }
bool cactus_metal_encode_swiglu(void*, const void*, const void*, size_t, float) { return false; }
bool cactus_metal_encode_rope(void*, const void*, const void*, const void*, uint32_t, uint32_t) { return false; }
bool cactus_metal_encode_rms_norm(void*, const void*, const void*, size_t, size_t, float) { return false; }
bool cactus_metal_encode_rms_norm_add(void*, const void*, const void*, const void*, size_t, size_t, float) { return false; }
bool cactus_metal_encode_argmax(const void*, uint32_t, void*) { return false; }
bool cactus_metal_encode_cast(void*, int, const void*, int, size_t) { return false; }
bool cactus_metal_encode_quant_matmul(void*, const void*, const CactusQuantMatrix*) { return false; }
bool cactus_metal_encode_quant_matmul_m(void*, const void*, const CactusQuantMatrix*, uint32_t) { return false; }
bool cactus_metal_prewarm_quant(const CactusQuantMatrix*) { return false; }
bool cactus_metal_encode_quant_matmul_ortho(void*, const void*, void*, const CactusQuantMatrix*) { return false; }
bool cactus_metal_encode_embedding_ortho(void*, uint32_t, const CactusQuantMatrix*) { return false; }
bool cactus_metal_encode_embedding_hadamard(void*, uint32_t, const CactusQuantMatrix*) { return false; }
bool cactus_metal_encode_embedding_ortho_m(void*, const CactusQuantMatrix*, const uint32_t*, uint32_t) { return false; }
bool cactus_metal_encode_embedding_hadamard_m(void*, const CactusQuantMatrix*, const uint32_t*, uint32_t) { return false; }
bool cactus_metal_encode_gather_f16(void*, const void*, size_t, const uint32_t*, uint32_t, uint32_t) { return false; }
bool cactus_metal_encode_attention_i8(void*, const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    float, size_t, size_t, size_t, size_t) { return false; }
bool cactus_metal_encode_attention_i8_prefill(void*, const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, float, size_t, size_t, size_t, size_t, uint32_t, uint32_t) { return false; }
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
#endif
