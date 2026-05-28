/* High-level pipeline factories: build the right Metal pipeline for each
 * cactus GPU operation, mapping logical params (K, N, head_dim, etc.) to
 * the correct kernel name + function constant set.
 *
 * Function constant index assignments are kept in lockstep with the
 * `[[function_constant(N)]]` declarations in the .metal sources. */
#import <Metal/Metal.h>
#include "cactus_gpu.h"
#include "internal.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace cactus {
namespace gpu {

#if !CACTUS_HAS_METAL

Pipeline* pipeline_mul_mv_int4_fp16(Context*, uint32_t, uint32_t) { return nullptr; }
Pipeline* pipeline_mul_mm_int4_fp16(Context*, uint32_t, uint32_t, uint32_t) { return nullptr; }
Pipeline* pipeline_mul_mm_fp16(Context*, uint32_t, uint32_t) { return nullptr; }
Pipeline* pipeline_rms_norm_fp16(Context*, uint32_t) { return nullptr; }
Pipeline* pipeline_flash_attn(Context*, uint32_t, uint32_t, uint32_t, bool, bool) { return nullptr; }
Pipeline* pipeline_rope_apply(Context*, uint32_t, bool, float) { return nullptr; }
Pipeline* pipeline_swiglu(Context*, uint32_t) { return nullptr; }
Pipeline* pipeline_kv_cache_append(Context*, uint32_t, uint32_t) { return nullptr; }
Pipeline* pipeline_sample_argmax(Context*, uint32_t) { return nullptr; }
Pipeline* pipeline_sample_top_k_top_p(Context*, uint32_t) { return nullptr; }
Pipeline* pipeline_embed_lookup(Context*, uint32_t) { return nullptr; }
Pipeline* pipeline_residual_add(Context*, uint32_t) { return nullptr; }
Pipeline* pipeline_mul_mv_fp16(Context*, uint32_t, uint32_t) { return nullptr; }

#else

// ============================================================================
// Function constant index assignments — KEEP IN SYNC WITH .metal sources
// ============================================================================
namespace fc {
    // Matmul family
    constexpr uint32_t MM_K           = 10;
    constexpr uint32_t MM_N           = 11;
    constexpr uint32_t MM_M_TILE      = 12;
    constexpr uint32_t MM_HAS_BIAS    = 13;

    // RMSNorm
    constexpr uint32_t RMS_AXIS_SIZE  = 20;
    constexpr uint32_t RMS_EPS        = 21;

    // Flash attention
    constexpr uint32_t FA_HEAD_DIM_Q  = 30;
    constexpr uint32_t FA_HEAD_DIM_V  = 31;
    constexpr uint32_t FA_NUM_GROUPS  = 32;
    constexpr uint32_t FA_CAUSAL      = 33;
    constexpr uint32_t FA_HAS_SOFTCAP = 34;

    // RoPE
    constexpr uint32_t ROPE_HEAD_DIM  = 40;
    constexpr uint32_t ROPE_IS_NEOX   = 41;
    constexpr uint32_t ROPE_THETA     = 42;

    // KV cache
    constexpr uint32_t KV_NUM_HEADS   = 50;
    constexpr uint32_t KV_HEAD_DIM    = 51;

    // SwiGLU
    constexpr uint32_t SW_HIDDEN_DIM  = 60;

    // Sampling / lookup
    constexpr uint32_t SAMPLE_VOCAB   = 70;
    constexpr uint32_t EMBED_HIDDEN   = 71;

    // Residual add
    constexpr uint32_t RESIDUAL_AXIS  = 80;

    // fp16 mat-vec (LM head)
    constexpr uint32_t MM_K_FP        = 90;
    constexpr uint32_t MM_N_FP        = 91;
}

// ----------------------------------------------------------------------------
// Matmul: int4 × fp16, decode (mat-vec)
// ----------------------------------------------------------------------------
Pipeline* pipeline_mul_mv_int4_fp16(Context* ctx, uint32_t K, uint32_t N) {
    FunctionConstant fcs[3] = {
        {"K",        FCType::UINT32, fc::MM_K,        {.u32 = K}},
        {"N",        FCType::UINT32, fc::MM_N,        {.u32 = N}},
        {"HAS_BIAS", FCType::BOOL,   fc::MM_HAS_BIAS, {.b   = false}},
    };
    return pipeline_create(ctx, "mul_mv_int4_fp16", fcs, 3);
}

// ----------------------------------------------------------------------------
// Matmul: int4 × fp16, prefill (mat-mat)
// ----------------------------------------------------------------------------
Pipeline* pipeline_mul_mm_int4_fp16(Context* ctx, uint32_t K, uint32_t N, uint32_t M_tile) {
    FunctionConstant fcs[4] = {
        {"K",        FCType::UINT32, fc::MM_K,        {.u32 = K}},
        {"N",        FCType::UINT32, fc::MM_N,        {.u32 = N}},
        {"M_TILE",   FCType::UINT32, fc::MM_M_TILE,   {.u32 = M_tile}},
        {"HAS_BIAS", FCType::BOOL,   fc::MM_HAS_BIAS, {.b   = false}},
    };
    return pipeline_create(ctx, "mul_mm_int4_fp16", fcs, 4);
}

// ----------------------------------------------------------------------------
// Matmul: fp16 × fp16 (e.g., LM head)
// ----------------------------------------------------------------------------
Pipeline* pipeline_mul_mm_fp16(Context* ctx, uint32_t K, uint32_t N) {
    FunctionConstant fcs[2] = {
        {"K", FCType::UINT32, fc::MM_K, {.u32 = K}},
        {"N", FCType::UINT32, fc::MM_N, {.u32 = N}},
    };
    return pipeline_create(ctx, "mul_mm_fp16", fcs, 2);
}

// ----------------------------------------------------------------------------
// RMSNorm with fused weight scale
// ----------------------------------------------------------------------------
Pipeline* pipeline_rms_norm_fp16(Context* ctx, uint32_t axis_size) {
    FunctionConstant fcs[1] = {
        {"AXIS_SIZE", FCType::UINT32, fc::RMS_AXIS_SIZE, {.u32 = axis_size}},
    };
    return pipeline_create(ctx, "rms_norm_fp16", fcs, 1);
}

// ----------------------------------------------------------------------------
// Flash attention — kernel name encodes head_dim (specialized per arch)
// ----------------------------------------------------------------------------
Pipeline* pipeline_flash_attn(Context* ctx,
                              uint32_t head_dim_q,
                              uint32_t head_dim_v,
                              uint32_t num_query_groups,
                              bool causal,
                              bool has_softcap) {
    char kname[128];
    std::snprintf(kname, sizeof(kname),
                  "flash_attn_dk%u_dv%u",
                  head_dim_q, head_dim_v);
    FunctionConstant fcs[3] = {
        {"NUM_GROUPS",  FCType::UINT32, fc::FA_NUM_GROUPS,  {.u32 = num_query_groups}},
        {"CAUSAL",      FCType::BOOL,   fc::FA_CAUSAL,      {.b   = causal}},
        {"HAS_SOFTCAP", FCType::BOOL,   fc::FA_HAS_SOFTCAP, {.b   = has_softcap}},
    };
    return pipeline_create(ctx, kname, fcs, 3);
}

// ----------------------------------------------------------------------------
// RoPE — applied in-place to Q and K
// ----------------------------------------------------------------------------
Pipeline* pipeline_rope_apply(Context* ctx,
                              uint32_t head_dim,
                              bool is_neox,
                              float theta_base) {
    FunctionConstant fcs[3] = {
        {"HEAD_DIM", FCType::UINT32, fc::ROPE_HEAD_DIM, {.u32 = head_dim}},
        {"IS_NEOX",  FCType::BOOL,   fc::ROPE_IS_NEOX,  {.b   = is_neox}},
        {"THETA",    FCType::FP32,   fc::ROPE_THETA,    {.f32 = theta_base}},
    };
    return pipeline_create(ctx, "rope_apply", fcs, 3);
}

// ----------------------------------------------------------------------------
// SwiGLU
// ----------------------------------------------------------------------------
Pipeline* pipeline_swiglu(Context* ctx, uint32_t hidden_dim) {
    FunctionConstant fcs[1] = {
        {"HIDDEN_DIM", FCType::UINT32, fc::SW_HIDDEN_DIM, {.u32 = hidden_dim}},
    };
    return pipeline_create(ctx, "swiglu_fwd", fcs, 1);
}

// ----------------------------------------------------------------------------
// KV cache append
// ----------------------------------------------------------------------------
Pipeline* pipeline_kv_cache_append(Context* ctx, uint32_t num_kv_heads, uint32_t head_dim) {
    FunctionConstant fcs[2] = {
        {"NUM_KV_HEADS", FCType::UINT32, fc::KV_NUM_HEADS, {.u32 = num_kv_heads}},
        {"HEAD_DIM",     FCType::UINT32, fc::KV_HEAD_DIM,  {.u32 = head_dim}},
    };
    return pipeline_create(ctx, "kv_cache_append", fcs, 2);
}

// ----------------------------------------------------------------------------
// Sampling (final stage of decode)
// ----------------------------------------------------------------------------
Pipeline* pipeline_sample_argmax(Context* ctx, uint32_t vocab_size) {
    FunctionConstant fcs[1] = {
        {"VOCAB_SIZE", FCType::UINT32, fc::SAMPLE_VOCAB, {.u32 = vocab_size}},
    };
    return pipeline_create(ctx, "sample_argmax", fcs, 1);
}

Pipeline* pipeline_sample_top_k_top_p(Context* ctx, uint32_t vocab_size) {
    FunctionConstant fcs[1] = {
        {"VOCAB_SIZE", FCType::UINT32, fc::SAMPLE_VOCAB, {.u32 = vocab_size}},
    };
    return pipeline_create(ctx, "sample_top_k_top_p", fcs, 1);
}

// ----------------------------------------------------------------------------
// Embedding lookup
// ----------------------------------------------------------------------------
Pipeline* pipeline_embed_lookup(Context* ctx, uint32_t hidden_dim) {
    FunctionConstant fcs[1] = {
        {"HIDDEN_DIM", FCType::UINT32, fc::EMBED_HIDDEN, {.u32 = hidden_dim}},
    };
    return pipeline_create(ctx, "embed_lookup", fcs, 1);
}

// ----------------------------------------------------------------------------
// Residual add: y[:] += x[:]
// ----------------------------------------------------------------------------
Pipeline* pipeline_residual_add(Context* ctx, uint32_t axis_size) {
    FunctionConstant fcs[1] = {
        {"AXIS_SIZE_RES", FCType::UINT32, fc::RESIDUAL_AXIS, {.u32 = axis_size}},
    };
    return pipeline_create(ctx, "residual_add", fcs, 1);
}

// ----------------------------------------------------------------------------
// fp16 mat-vec (LM head)
// ----------------------------------------------------------------------------
Pipeline* pipeline_mul_mv_fp16(Context* ctx, uint32_t K, uint32_t N) {
    FunctionConstant fcs[2] = {
        {"MM_K_FP", FCType::UINT32, fc::MM_K_FP, {.u32 = K}},
        {"MM_N_FP", FCType::UINT32, fc::MM_N_FP, {.u32 = N}},
    };
    return pipeline_create(ctx, "mul_mv_fp16", fcs, 2);
}

#endif  // CACTUS_HAS_METAL

} // namespace gpu
} // namespace cactus
