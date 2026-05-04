#include "../cactus_kernels.h"
#include "threading.h"
#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

static constexpr float TQ_LLOYD_SLOPE    = 0.9966f;
static constexpr float TQ_LLOYD_OFFSET   = -1.4949f;
static constexpr float TQ_LLOYD_BOUNDARY = 0.9816f;

static constexpr float TQ_4BIT_CENTROIDS[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9424f, -0.6568f, -0.3882f, -0.1284f,
     0.1284f,  0.3882f,  0.6568f,  0.9424f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
};

static constexpr float TQ_4BIT_BOUNDARIES[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f, -0.7996f, -0.5224f, -0.2582f,
     0.0f,
     0.2582f,  0.5224f,  0.7996f,  1.0993f,  1.4371f,  1.8435f,  2.4008f
};

struct Xoshiro256 {
    uint64_t s[4];
    explicit Xoshiro256(uint64_t seed) {
        for (int i = 0; i < 4; i++) {
            seed += 0x9e3779b97f4a7c15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }

    static uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t next() {
        uint64_t result = rotl(s[1] * 5, 7) * 9;
        uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }
};

void hadamard_inplace(float* __restrict data, size_t dim) {
    if (dim < 4 || (dim & (dim - 1)) != 0) return;

    size_t i = 0;
    for (; i + 15 < dim; i += 16) {
        float32x4_t v0 = vld1q_f32(data + i);
        float32x4_t v1 = vld1q_f32(data + i + 4);
        float32x4_t v2 = vld1q_f32(data + i + 8);
        float32x4_t v3 = vld1q_f32(data + i + 12);

        float32x4_t r0 = vrev64q_f32(v0), r1 = vrev64q_f32(v1);
        float32x4_t r2 = vrev64q_f32(v2), r3 = vrev64q_f32(v3);

        float32x4_t p0 = vtrn1q_f32(vaddq_f32(v0, r0), vsubq_f32(v0, r0));
        float32x4_t p1 = vtrn1q_f32(vaddq_f32(v1, r1), vsubq_f32(v1, r1));
        float32x4_t p2 = vtrn1q_f32(vaddq_f32(v2, r2), vsubq_f32(v2, r2));
        float32x4_t p3 = vtrn1q_f32(vaddq_f32(v3, r3), vsubq_f32(v3, r3));

        float32x4_t s0 = vcombine_f32(vget_high_f32(p0), vget_low_f32(p0));
        float32x4_t s1 = vcombine_f32(vget_high_f32(p1), vget_low_f32(p1));
        float32x4_t s2 = vcombine_f32(vget_high_f32(p2), vget_low_f32(p2));
        float32x4_t s3 = vcombine_f32(vget_high_f32(p3), vget_low_f32(p3));

        vst1q_f32(data + i,      vcombine_f32(vget_low_f32(vaddq_f32(p0, s0)), vget_low_f32(vsubq_f32(p0, s0))));
        vst1q_f32(data + i + 4,  vcombine_f32(vget_low_f32(vaddq_f32(p1, s1)), vget_low_f32(vsubq_f32(p1, s1))));
        vst1q_f32(data + i + 8,  vcombine_f32(vget_low_f32(vaddq_f32(p2, s2)), vget_low_f32(vsubq_f32(p2, s2))));
        vst1q_f32(data + i + 12, vcombine_f32(vget_low_f32(vaddq_f32(p3, s3)), vget_low_f32(vsubq_f32(p3, s3))));
    }
    for (; i < dim; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        float32x4_t r = vrev64q_f32(v);
        float32x4_t p = vtrn1q_f32(vaddq_f32(v, r), vsubq_f32(v, r));
        float32x4_t s = vcombine_f32(vget_high_f32(p), vget_low_f32(p));
        vst1q_f32(data + i, vcombine_f32(vget_low_f32(vaddq_f32(p, s)), vget_low_f32(vsubq_f32(p, s))));
    }

    if (dim >= 16) {
        for (size_t k = 0; k < dim; k += 8) {
            float32x4_t a = vld1q_f32(data + k);
            float32x4_t b = vld1q_f32(data + k + 4);
            vst1q_f32(data + k,     vaddq_f32(a, b));
            vst1q_f32(data + k + 4, vsubq_f32(a, b));
        }
    }

    if (dim >= 32) {
        for (size_t k = 0; k < dim; k += 16) {
            float32x4_t a0 = vld1q_f32(data + k);
            float32x4_t a1 = vld1q_f32(data + k + 4);
            float32x4_t b0 = vld1q_f32(data + k + 8);
            float32x4_t b1 = vld1q_f32(data + k + 12);

            vst1q_f32(data + k,      vaddq_f32(a0, b0));
            vst1q_f32(data + k + 8,  vsubq_f32(a0, b0));
            vst1q_f32(data + k + 4,  vaddq_f32(a1, b1));
            vst1q_f32(data + k + 12, vsubq_f32(a1, b1));
        }
    }

    for (size_t half = 16; half * 2 < dim; half <<= 1) {
        for (size_t g = 0; g < dim; g += half * 2) {
            size_t j = g;
            for (; j + 15 < g + half; j += 16) {
                float32x4_t a0 = vld1q_f32(data + j);
                float32x4_t a1 = vld1q_f32(data + j + 4);
                float32x4_t a2 = vld1q_f32(data + j + 8);
                float32x4_t a3 = vld1q_f32(data + j + 12);
                float32x4_t b0 = vld1q_f32(data + j + half);
                float32x4_t b1 = vld1q_f32(data + j + half + 4);
                float32x4_t b2 = vld1q_f32(data + j + half + 8);
                float32x4_t b3 = vld1q_f32(data + j + half + 12);

                vst1q_f32(data + j,             vaddq_f32(a0, b0));
                vst1q_f32(data + j + half,      vsubq_f32(a0, b0));
                vst1q_f32(data + j + 4,         vaddq_f32(a1, b1));
                vst1q_f32(data + j + half + 4,  vsubq_f32(a1, b1));
                vst1q_f32(data + j + 8,         vaddq_f32(a2, b2));
                vst1q_f32(data + j + half + 8,  vsubq_f32(a2, b2));
                vst1q_f32(data + j + 12,        vaddq_f32(a3, b3));
                vst1q_f32(data + j + half + 12, vsubq_f32(a3, b3));
            }
            for (; j < g + half; j += 4) {
                float32x4_t a = vld1q_f32(data + j);
                float32x4_t b = vld1q_f32(data + j + half);
                vst1q_f32(data + j,        vaddq_f32(a, b));
                vst1q_f32(data + j + half, vsubq_f32(a, b));
            }
        }
    }

    float32x4_t norm_vec = vdupq_n_f32(1.0f / sqrtf(static_cast<float>(dim)));
    if (dim >= 8) {
        size_t half = dim / 2;
        size_t j = 0;
        for (; j + 15 < half; j += 16) {
            float32x4_t a0 = vld1q_f32(data + j);
            float32x4_t a1 = vld1q_f32(data + j + 4);
            float32x4_t a2 = vld1q_f32(data + j + 8);
            float32x4_t a3 = vld1q_f32(data + j + 12);

            float32x4_t b0 = vld1q_f32(data + j + half);
            float32x4_t b1 = vld1q_f32(data + j + half + 4);
            float32x4_t b2 = vld1q_f32(data + j + half + 8);
            float32x4_t b3 = vld1q_f32(data + j + half + 12);

            vst1q_f32(data + j,             vmulq_f32(vaddq_f32(a0, b0), norm_vec));
            vst1q_f32(data + j + half,      vmulq_f32(vsubq_f32(a0, b0), norm_vec));
            vst1q_f32(data + j + 4,         vmulq_f32(vaddq_f32(a1, b1), norm_vec));
            vst1q_f32(data + j + half + 4,  vmulq_f32(vsubq_f32(a1, b1), norm_vec));
            vst1q_f32(data + j + 8,         vmulq_f32(vaddq_f32(a2, b2), norm_vec));
            vst1q_f32(data + j + half + 8,  vmulq_f32(vsubq_f32(a2, b2), norm_vec));
            vst1q_f32(data + j + 12,        vmulq_f32(vaddq_f32(a3, b3), norm_vec));
            vst1q_f32(data + j + half + 12, vmulq_f32(vsubq_f32(a3, b3), norm_vec));
        }
        for (; j < half; j += 4) {
            float32x4_t a = vld1q_f32(data + j);
            float32x4_t b = vld1q_f32(data + j + half);

            vst1q_f32(data + j,        vmulq_f32(vaddq_f32(a, b), norm_vec));
            vst1q_f32(data + j + half, vmulq_f32(vsubq_f32(a, b), norm_vec));
        }
    } else {
        vst1q_f32(data, vmulq_f32(vld1q_f32(data), norm_vec));
    }
}

void apply_signs(float* __restrict data, const uint8_t* __restrict signs_packed, size_t dim) {
    const uint8x8_t kBitMasks = vcreate_u8(0x8040201008040201ULL);
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        int8x8_t sm0 = vreinterpret_s8_u8(vtst_u8(vdup_n_u8(signs_packed[i / 8]),     kBitMasks));
        int8x8_t sm1 = vreinterpret_s8_u8(vtst_u8(vdup_n_u8(signs_packed[i / 8 + 1]), kBitMasks));
        int16x8_t sm16_0 = vmovl_s8(sm0);
        int16x8_t sm16_1 = vmovl_s8(sm1);
        int32x4_t f0 = vshlq_n_s32(vmovl_s16(vget_low_s16(sm16_0)),  31);
        int32x4_t f1 = vshlq_n_s32(vmovl_high_s16(sm16_0),            31);
        int32x4_t f2 = vshlq_n_s32(vmovl_s16(vget_low_s16(sm16_1)),  31);
        int32x4_t f3 = vshlq_n_s32(vmovl_high_s16(sm16_1),            31);
        vst1q_f32(data + i,      vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(vld1q_f32(data + i)),      f0)));
        vst1q_f32(data + i + 4,  vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(vld1q_f32(data + i + 4)),  f1)));
        vst1q_f32(data + i + 8,  vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(vld1q_f32(data + i + 8)),  f2)));
        vst1q_f32(data + i + 12, vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(vld1q_f32(data + i + 12)), f3)));
    }
    for (; i < dim; i++)
        if ((signs_packed[i / 8] >> (i % 8)) & 1) data[i] = -data[i];
}

void rotate_forward(float* data, const uint8_t* signs_packed, size_t dim) {
    const size_t row_bytes = (dim + 7) / 8;
    for (size_t layer = 0; layer < TURBOQUANT_ROTATION_LAYERS; ++layer) {
        apply_signs(data, signs_packed + layer * row_bytes, dim);
        hadamard_inplace(data, dim);
    }
}

void rotate_inverse(float* data, const uint8_t* signs_packed, size_t dim) {
    const size_t row_bytes = (dim + 7) / 8;
    for (size_t layer = TURBOQUANT_ROTATION_LAYERS; layer > 0; --layer) {
        hadamard_inplace(data, dim);
        apply_signs(data, signs_packed + (layer - 1) * row_bytes, dim);
    }
}

float dot_2bit_f32(
    const float* __restrict q,
    float q_sum,
    const uint8_t* __restrict packed,
    size_t dim
) {
    const uint32x4_t mask2 = vdupq_n_u32(3);
    const int32x4_t bit_shifts = {0, -2, -4, -6};

    float32x4_t code_dot0 = vdupq_n_f32(0.0f);
    float32x4_t code_dot1 = vdupq_n_f32(0.0f);
    size_t p_idx = 0;
    size_t i = 0;

    for (; i + 16 <= dim; i += 16) {
        uint32_t word;
        memcpy(&word, packed + p_idx, 4); p_idx += 4;

        uint32x4_t bv0 = vandq_u32(vshlq_u32(vdupq_n_u32( word        & 0xFF), bit_shifts), mask2);
        uint32x4_t bv1 = vandq_u32(vshlq_u32(vdupq_n_u32((word >>  8) & 0xFF), bit_shifts), mask2);
        uint32x4_t bv2 = vandq_u32(vshlq_u32(vdupq_n_u32((word >> 16) & 0xFF), bit_shifts), mask2);
        uint32x4_t bv3 = vandq_u32(vshlq_u32(vdupq_n_u32( word >> 24),         bit_shifts), mask2);

        code_dot0 = vfmaq_f32(code_dot0, vld1q_f32(q + i),      vcvtq_f32_u32(bv0));
        code_dot1 = vfmaq_f32(code_dot1, vld1q_f32(q + i + 4),  vcvtq_f32_u32(bv1));
        code_dot0 = vfmaq_f32(code_dot0, vld1q_f32(q + i + 8),  vcvtq_f32_u32(bv2));
        code_dot1 = vfmaq_f32(code_dot1, vld1q_f32(q + i + 12), vcvtq_f32_u32(bv3));
    }
    for (; i + 4 <= dim; i += 4) {
        uint32x4_t bv = vandq_u32(vshlq_u32(vdupq_n_u32(packed[p_idx++]), bit_shifts), mask2);
        code_dot0 = vfmaq_f32(code_dot0, vld1q_f32(q + i), vcvtq_f32_u32(bv));
    }

    float code_dot = vaddvq_f32(vaddq_f32(code_dot0, code_dot1));
    for (; i < dim; i++) {
        uint8_t code = (packed[i / 4] >> ((i % 4) * 2)) & 0x3;
        code_dot += q[i] * (float)code;
    }

    const float inv_sqrt_dim = 1.0f / sqrtf(static_cast<float>(dim));
    return (TQ_LLOYD_SLOPE * code_dot + TQ_LLOYD_OFFSET * q_sum) * inv_sqrt_dim;
}

void accumulate_2bit_f32(
    const uint8_t* __restrict packed,
    float weight_radius,
    float* __restrict accum,
    size_t dim
) {
    const float inv_sqrt_dim = 1.0f / sqrtf(static_cast<float>(dim));
    const uint32x4_t mask2 = vdupq_n_u32(3);
    const int32x4_t bit_shifts = {0, -2, -4, -6};
    const float32x4_t wr_half = vdupq_n_f32(TQ_LLOYD_SLOPE * inv_sqrt_dim * weight_radius);
    const float32x4_t wr_off  = vdupq_n_f32(TQ_LLOYD_OFFSET * inv_sqrt_dim * weight_radius);

    size_t p_idx = 0;
    size_t i = 0;

    for (; i + 16 <= dim; i += 16) {
        uint32_t word;
        memcpy(&word, packed + p_idx, 4); p_idx += 4;

        float32x4_t c0 = vcvtq_f32_u32(vandq_u32(vshlq_u32(vdupq_n_u32( word        & 0xFF), bit_shifts), mask2));
        float32x4_t c1 = vcvtq_f32_u32(vandq_u32(vshlq_u32(vdupq_n_u32((word >>  8) & 0xFF), bit_shifts), mask2));
        float32x4_t c2 = vcvtq_f32_u32(vandq_u32(vshlq_u32(vdupq_n_u32((word >> 16) & 0xFF), bit_shifts), mask2));
        float32x4_t c3 = vcvtq_f32_u32(vandq_u32(vshlq_u32(vdupq_n_u32( word >> 24),         bit_shifts), mask2));

        vst1q_f32(accum + i,      vfmaq_f32(vaddq_f32(vld1q_f32(accum + i),      wr_off), c0, wr_half));
        vst1q_f32(accum + i + 4,  vfmaq_f32(vaddq_f32(vld1q_f32(accum + i + 4),  wr_off), c1, wr_half));
        vst1q_f32(accum + i + 8,  vfmaq_f32(vaddq_f32(vld1q_f32(accum + i + 8),  wr_off), c2, wr_half));
        vst1q_f32(accum + i + 12, vfmaq_f32(vaddq_f32(vld1q_f32(accum + i + 12), wr_off), c3, wr_half));
    }
    for (; i + 4 <= dim; i += 4) {
        float32x4_t c = vcvtq_f32_u32(vandq_u32(vshlq_u32(vdupq_n_u32(packed[p_idx++]), bit_shifts), mask2));
        vst1q_f32(accum + i, vfmaq_f32(vaddq_f32(vld1q_f32(accum + i), wr_off), c, wr_half));
    }
    for (; i < dim; i++) {
        uint8_t code = (packed[i / 4] >> ((i % 4) * 2)) & 0x3;
        accum[i] += (TQ_LLOYD_SLOPE * (float)code + TQ_LLOYD_OFFSET) * inv_sqrt_dim * weight_radius;
    }
}

static void quantize_4bit(const float* src, uint8_t* dst, size_t dim) {
    const float sqrt_dim = sqrtf(static_cast<float>(dim));
    const float32x4_t sd = vdupq_n_f32(sqrt_dim);

    float32x4_t bv[15];
    for (int b = 0; b < 15; b++)
        bv[b] = vdupq_n_f32(TQ_4BIT_BOUNDARIES[b]);

    size_t out = 0;
    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t vals = vmulq_f32(vld1q_f32(src + i), sd);

        int32x4_t code = vdupq_n_s32(0);
        for (int b = 0; b < 15; b++)
            code = vsubq_s32(code, vreinterpretq_s32_u32(vcgtq_f32(vals, bv[b])));

        code = vmaxq_s32(vminq_s32(code, vdupq_n_s32(15)), vdupq_n_s32(0));

        int32_t c[4];
        vst1q_s32(c, code);
        dst[out++] = (uint8_t)((c[1] << 4) | c[0]);
        dst[out++] = (uint8_t)((c[3] << 4) | c[2]);
    }
    for (; i + 2 <= dim; i += 2) {
        float v0 = src[i] * sqrt_dim, v1 = src[i + 1] * sqrt_dim;
        int c0 = 0, c1 = 0;
        for (int b = 0; b < 15; b++) {
            if (v0 > TQ_4BIT_BOUNDARIES[b]) c0 = b + 1;
            if (v1 > TQ_4BIT_BOUNDARIES[b]) c1 = b + 1;
        }
        dst[out++] = (uint8_t)((c1 << 4) | c0);
    }
    if (i < dim) {
        float v0 = src[i] * sqrt_dim;
        int c0 = 0;
        for (int b = 0; b < 15; b++)
            if (v0 > TQ_4BIT_BOUNDARIES[b]) c0 = b + 1;
        dst[out++] = (uint8_t)c0;
    }
}

void dequantize_4bit(const uint8_t* src, float* dst, size_t dim) {
    const float inv_sqrt_dim = 1.0f / sqrtf(static_cast<float>(dim));
    float lut[16];
    for (int i = 0; i < 16; i++)
        lut[i] = TQ_4BIT_CENTROIDS[i] * inv_sqrt_dim;

    size_t p = 0;
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        float tmp[8];
        tmp[0] = lut[src[p]   & 0xF]; tmp[1] = lut[src[p]   >> 4];
        tmp[2] = lut[src[p+1] & 0xF]; tmp[3] = lut[src[p+1] >> 4];
        tmp[4] = lut[src[p+2] & 0xF]; tmp[5] = lut[src[p+2] >> 4];
        tmp[6] = lut[src[p+3] & 0xF]; tmp[7] = lut[src[p+3] >> 4];
        p += 4;
        vst1q_f32(dst + i,     vld1q_f32(tmp));
        vst1q_f32(dst + i + 4, vld1q_f32(tmp + 4));
    }
    for (; i + 2 <= dim; i += 2) {
        dst[i]     = lut[src[p] & 0xF];
        dst[i + 1] = lut[src[p] >> 4];
        p++;
    }
    if (i < dim)
        dst[i] = lut[src[p] & 0xF];
}

float dot_4bit_f32(
    const float* __restrict q,
    const uint8_t* __restrict packed,
    size_t dim
) {
    const float inv_sqrt_dim = 1.0f / sqrtf(static_cast<float>(dim));

    thread_local size_t cached_d4_dim = 0;
    thread_local __fp16 cached_d4_lut[16] __attribute__((aligned(16)));
    thread_local uint8x16_t cached_d4_lo;
    thread_local uint8x16_t cached_d4_hi;
    if (cached_d4_dim != dim) {
        for (int i = 0; i < 16; i++)
            cached_d4_lut[i] = static_cast<__fp16>(TQ_4BIT_CENTROIDS[i] * inv_sqrt_dim);
        uint8x16x2_t split = vld2q_u8(reinterpret_cast<const uint8_t*>(cached_d4_lut));
        cached_d4_lo = split.val[0];
        cached_d4_hi = split.val[1];
        cached_d4_dim = dim;
    }
    const uint8x16_t v_lo_lut = cached_d4_lo;
    const uint8x16_t v_hi_lut = cached_d4_hi;

    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    size_t p = 0;
    size_t i = 0;

    const uint8x8_t mask_lo = vdup_n_u8(0x0F);
    for (; i + 16 <= dim; i += 16) {
        uint8x8_t packed8 = vld1_u8(packed + p);
        p += 8;
        uint8x8_t lo_nib = vand_u8(packed8, mask_lo);
        uint8x8_t hi_nib = vshr_n_u8(packed8, 4);

        uint8x8_t idx_lo = vzip1_u8(lo_nib, hi_nib);
        uint8x8_t idx_hi = vzip2_u8(lo_nib, hi_nib);
        uint8x16_t indices = vcombine_u8(idx_lo, idx_hi);

        uint8x16_t lo_bytes = vqtbl1q_u8(v_lo_lut, indices);
        uint8x16_t hi_bytes = vqtbl1q_u8(v_hi_lut, indices);

        uint8x16_t fp16_pairs_lo = vzip1q_u8(lo_bytes, hi_bytes);
        uint8x16_t fp16_pairs_hi = vzip2q_u8(lo_bytes, hi_bytes);

        float16x8_t f16_lo = vreinterpretq_f16_u8(fp16_pairs_lo);
        float16x8_t f16_hi = vreinterpretq_f16_u8(fp16_pairs_hi);

        float32x4_t f32_0 = vcvt_f32_f16(vget_low_f16(f16_lo));
        float32x4_t f32_1 = vcvt_f32_f16(vget_high_f16(f16_lo));
        float32x4_t f32_2 = vcvt_f32_f16(vget_low_f16(f16_hi));
        float32x4_t f32_3 = vcvt_f32_f16(vget_high_f16(f16_hi));

        acc0 = vfmaq_f32(acc0, vld1q_f32(q + i),      f32_0);
        acc1 = vfmaq_f32(acc1, vld1q_f32(q + i + 4),  f32_1);
        acc2 = vfmaq_f32(acc2, vld1q_f32(q + i + 8),  f32_2);
        acc3 = vfmaq_f32(acc3, vld1q_f32(q + i + 12), f32_3);
    }

    float result = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1),
                                        vaddq_f32(acc2, acc3)));

    if (i < dim) {
        float lut[16];
        for (int k = 0; k < 16; k++)
            lut[k] = TQ_4BIT_CENTROIDS[k] * inv_sqrt_dim;
        for (; i + 2 <= dim; i += 2) {
            result += q[i]     * lut[packed[p] & 0xF];
            result += q[i + 1] * lut[packed[p] >> 4];
            p++;
        }
        if (i < dim)
            result += q[i] * lut[packed[p] & 0xF];
    }
    return result;
}

void accumulate_4bit_f32(
    const uint8_t* __restrict packed,
    float weight_radius,
    float* __restrict accum,
    size_t dim
) {
    const float inv_sqrt_dim = 1.0f / sqrtf(static_cast<float>(dim));
    const float wr = inv_sqrt_dim * weight_radius;

    __fp16 fp16_lut[16] __attribute__((aligned(16)));
    for (int i = 0; i < 16; i++)
        fp16_lut[i] = static_cast<__fp16>(TQ_4BIT_CENTROIDS[i] * wr);
    uint8x16x2_t split = vld2q_u8(reinterpret_cast<const uint8_t*>(fp16_lut));
    const uint8x16_t v_lo_lut = split.val[0];
    const uint8x16_t v_hi_lut = split.val[1];

    size_t p = 0;
    size_t i = 0;

    const uint8x8_t mask_lo = vdup_n_u8(0x0F);
    for (; i + 16 <= dim; i += 16) {
        uint8x8_t packed8 = vld1_u8(packed + p);
        p += 8;
        uint8x8_t lo_nib = vand_u8(packed8, mask_lo);
        uint8x8_t hi_nib = vshr_n_u8(packed8, 4);
        uint8x8_t idx_lo = vzip1_u8(lo_nib, hi_nib);
        uint8x8_t idx_hi = vzip2_u8(lo_nib, hi_nib);
        uint8x16_t indices = vcombine_u8(idx_lo, idx_hi);

        uint8x16_t lo_bytes = vqtbl1q_u8(v_lo_lut, indices);
        uint8x16_t hi_bytes = vqtbl1q_u8(v_hi_lut, indices);

        uint8x16_t fp16_pairs_lo = vzip1q_u8(lo_bytes, hi_bytes);
        uint8x16_t fp16_pairs_hi = vzip2q_u8(lo_bytes, hi_bytes);

        float16x8_t f16_lo = vreinterpretq_f16_u8(fp16_pairs_lo);
        float16x8_t f16_hi = vreinterpretq_f16_u8(fp16_pairs_hi);

        float32x4_t v0 = vcvt_f32_f16(vget_low_f16(f16_lo));
        float32x4_t v1 = vcvt_f32_f16(vget_high_f16(f16_lo));
        float32x4_t v2 = vcvt_f32_f16(vget_low_f16(f16_hi));
        float32x4_t v3 = vcvt_f32_f16(vget_high_f16(f16_hi));

        vst1q_f32(accum + i,      vaddq_f32(vld1q_f32(accum + i),      v0));
        vst1q_f32(accum + i + 4,  vaddq_f32(vld1q_f32(accum + i + 4),  v1));
        vst1q_f32(accum + i + 8,  vaddq_f32(vld1q_f32(accum + i + 8),  v2));
        vst1q_f32(accum + i + 12, vaddq_f32(vld1q_f32(accum + i + 12), v3));
    }

    if (i < dim) {
        float lut[16];
        for (int k = 0; k < 16; k++) lut[k] = TQ_4BIT_CENTROIDS[k] * wr;
        for (; i + 2 <= dim; i += 2) {
            accum[i]     += lut[packed[p] & 0xF];
            accum[i + 1] += lut[packed[p] >> 4];
            p++;
        }
        if (i < dim)
            accum[i] += lut[packed[p] & 0xF];
    }
}

static void quantize_2bit(const float* src, uint8_t* dst, size_t dim) {
    const float quant_scale = sqrtf(static_cast<float>(dim)) / TQ_LLOYD_BOUNDARY;
    const float32x4_t qs_vec = vdupq_n_f32(quant_scale);
    const float32x4_t two = vdupq_n_f32(2.0f);
    const int32x4_t zero_i = vdupq_n_s32(0);
    const int32x4_t three_i = vdupq_n_s32(3);
    static const int32_t shift_arr[4] = {0, 2, 4, 6};
    const int32x4_t shift4 = vld1q_s32(shift_arr);
    size_t packed = 0;
    size_t i = 0;

    for (; i + 16 <= dim; i += 16) {
        int32x4_t c0 = vcvtmq_s32_f32(vfmaq_f32(two, vld1q_f32(&src[i]),      qs_vec));
        int32x4_t c1 = vcvtmq_s32_f32(vfmaq_f32(two, vld1q_f32(&src[i + 4]),  qs_vec));
        int32x4_t c2 = vcvtmq_s32_f32(vfmaq_f32(two, vld1q_f32(&src[i + 8]),  qs_vec));
        int32x4_t c3 = vcvtmq_s32_f32(vfmaq_f32(two, vld1q_f32(&src[i + 12]), qs_vec));
        c0 = vmaxq_s32(vminq_s32(c0, three_i), zero_i);
        c1 = vmaxq_s32(vminq_s32(c1, three_i), zero_i);
        c2 = vmaxq_s32(vminq_s32(c2, three_i), zero_i);
        c3 = vmaxq_s32(vminq_s32(c3, three_i), zero_i);

        uint32_t b0 = (uint32_t)vaddvq_s32(vshlq_s32(c0, shift4));
        uint32_t b1 = (uint32_t)vaddvq_s32(vshlq_s32(c1, shift4));
        uint32_t b2 = (uint32_t)vaddvq_s32(vshlq_s32(c2, shift4));
        uint32_t b3 = (uint32_t)vaddvq_s32(vshlq_s32(c3, shift4));
        uint32_t word = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | (b3 << 24);
        memcpy(dst + packed, &word, 4);
        packed += 4;
    }
    for (; i + 4 <= dim; i += 4) {
        int32x4_t c = vcvtmq_s32_f32(vfmaq_f32(two, vld1q_f32(&src[i]), qs_vec));
        c = vmaxq_s32(vminq_s32(c, three_i), zero_i);
        dst[packed++] = (uint8_t)vaddvq_s32(vshlq_s32(c, shift4));
    }
    if (i < dim) {
        uint8_t byte = 0;
        for (size_t j = 0; j < dim - i; j++) {
            int code = (int)floorf(src[i + j] * quant_scale + 2.0f);
            code = std::max(0, std::min(3, code));
            byte |= (uint8_t)(code << (j * 2));
        }
        dst[packed] = byte;
    }
}

void dequantize_2bit(const uint8_t* src, float* dst, size_t dim) {
    const float inv_sqrt_dim = 1.0f / sqrtf(static_cast<float>(dim));
    const uint32x4_t mask = vdupq_n_u32(3);
    const float32x4_t slope_v  = vdupq_n_f32(TQ_LLOYD_SLOPE * inv_sqrt_dim);
    const float32x4_t offset_v = vdupq_n_f32(TQ_LLOYD_OFFSET * inv_sqrt_dim);
    static const int32_t shift_arr[4] = {0, -2, -4, -6};
    const int32x4_t bit_shifts = vld1q_s32(shift_arr);
    size_t packed = 0;
    size_t i = 0;

    for (; i + 16 <= dim; i += 16) {
        uint32_t word;
        memcpy(&word, src + packed, 4); packed += 4;
        uint32x4_t bv0 = vdupq_n_u32( word        & 0xFF);
        uint32x4_t bv1 = vdupq_n_u32((word >>  8) & 0xFF);
        uint32x4_t bv2 = vdupq_n_u32((word >> 16) & 0xFF);
        uint32x4_t bv3 = vdupq_n_u32( word >> 24);
        vst1q_f32(&dst[i],      vfmaq_f32(offset_v, vcvtq_f32_u32(vandq_u32(vshlq_u32(bv0, bit_shifts), mask)), slope_v));
        vst1q_f32(&dst[i + 4],  vfmaq_f32(offset_v, vcvtq_f32_u32(vandq_u32(vshlq_u32(bv1, bit_shifts), mask)), slope_v));
        vst1q_f32(&dst[i + 8],  vfmaq_f32(offset_v, vcvtq_f32_u32(vandq_u32(vshlq_u32(bv2, bit_shifts), mask)), slope_v));
        vst1q_f32(&dst[i + 12], vfmaq_f32(offset_v, vcvtq_f32_u32(vandq_u32(vshlq_u32(bv3, bit_shifts), mask)), slope_v));
    }
    for (; i + 4 <= dim; i += 4) {
        uint32x4_t bv = vdupq_n_u32(src[packed++]);
        vst1q_f32(&dst[i], vfmaq_f32(offset_v, vcvtq_f32_u32(vandq_u32(vshlq_u32(bv, bit_shifts), mask)), slope_v));
    }
    if (i < dim) {
        uint8_t byte = src[packed];
        for (size_t j = 0; j < dim - i; j++) {
            uint8_t code = (byte >> (j * 2)) & 0x03;
            dst[i + j] = (TQ_LLOYD_SLOPE * (float)code + TQ_LLOYD_OFFSET) * inv_sqrt_dim;
        }
    }
}

static const float* tq_6bit_centroids() {
    static const std::array<float, 64> values = []() {
        constexpr int N = 64;
        constexpr int ITERS = 60;
        constexpr float kInvSqrt2 = 0.70710678118f;
        constexpr float kInvSqrt2pi = 0.39894228040f;

        std::array<float, N> c{};
        for (int i = 0; i < N; i++)
            c[i] = -4.0f + 8.0f * (float(i) + 0.5f) / float(N);

        std::array<float, N - 1> b{};
        for (int iter = 0; iter < ITERS; iter++) {
            for (int i = 0; i < N - 1; i++) b[i] = 0.5f * (c[i] + c[i+1]);
            for (int i = 0; i < N; i++) {
                float lo = (i == 0) ? -1e30f : b[i - 1];
                float hi = (i == N - 1) ? 1e30f : b[i];
                auto phi = [&](float x){ return kInvSqrt2pi * std::exp(-0.5f * x * x); };
                auto Phi = [&](float x){ return 0.5f * (1.0f + std::erf(x * kInvSqrt2)); };
                float num = phi(lo) - phi(hi);
                float den = Phi(hi) - Phi(lo);
                if (den > 1e-30f) c[i] = num / den;
            }
        }
        return c;
    }();
    return values.data();
}

static void quantize_6bit(const float* src, uint8_t* dst, size_t dim) {
    const float* centroids = tq_6bit_centroids();
    const float sqrt_dim = sqrtf(static_cast<float>(dim));

    auto encode = [&](float x) -> uint8_t {
        const float v = x * sqrt_dim;
        int lo = 0, hi = 63;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            float boundary = 0.5f * (centroids[mid] + centroids[mid + 1]);
            if (v < boundary) hi = mid;
            else              lo = mid + 1;
        }
        return static_cast<uint8_t>(lo);
    };

    size_t i = 0;
    size_t p = 0;
    for (; i + 4 <= dim; i += 4) {
        uint8_t c0 = encode(src[i]);
        uint8_t c1 = encode(src[i + 1]);
        uint8_t c2 = encode(src[i + 2]);
        uint8_t c3 = encode(src[i + 3]);
        dst[p]     = c0 | uint8_t((c1 & 0x03) << 6);
        dst[p + 1] = uint8_t(c1 >> 2) | uint8_t((c2 & 0x0F) << 4);
        dst[p + 2] = uint8_t(c2 >> 4) | uint8_t(c3 << 2);
        p += 3;
    }
}

void dequantize_6bit(const uint8_t* src, float* dst, size_t dim) {
    const float* centroids = tq_6bit_centroids();
    const float inv_sqrt_dim = 1.0f / sqrtf(static_cast<float>(dim));

    size_t i = 0;
    size_t p = 0;
    for (; i + 4 <= dim; i += 4) {
        uint8_t b0 = src[p];
        uint8_t b1 = src[p + 1];
        uint8_t b2 = src[p + 2];
        uint8_t c0 = b0 & 0x3F;
        uint8_t c1 = uint8_t((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F;
        uint8_t c2 = uint8_t((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F;
        uint8_t c3 = uint8_t(b2 >> 2) & 0x3F;
        dst[i]     = centroids[c0] * inv_sqrt_dim;
        dst[i + 1] = centroids[c1] * inv_sqrt_dim;
        dst[i + 2] = centroids[c2] * inv_sqrt_dim;
        dst[i + 3] = centroids[c3] * inv_sqrt_dim;
        p += 3;
    }
}

float dot_6bit_f32(const float* __restrict q, const uint8_t* __restrict packed, size_t dim) {
    const float* centroids = tq_6bit_centroids();

    thread_local size_t cached_dim = 0;
    thread_local uint8_t cached_lut_lo[64] __attribute__((aligned(16)));
    thread_local uint8_t cached_lut_hi[64] __attribute__((aligned(16)));
    if (cached_dim != dim) {
        const float inv_sqrt_dim = 1.0f / sqrtf(static_cast<float>(dim));
        __fp16 lut_fp16[64] __attribute__((aligned(16)));
        for (int i = 0; i < 64; i++) lut_fp16[i] = static_cast<__fp16>(centroids[i] * inv_sqrt_dim);
        const uint8_t* src = reinterpret_cast<const uint8_t*>(lut_fp16);
        for (int i = 0; i < 64; i++) {
            cached_lut_lo[i] = src[2 * i];
            cached_lut_hi[i] = src[2 * i + 1];
        }
        cached_dim = dim;
    }
    uint8x16x4_t v_lut_lo = {{vld1q_u8(cached_lut_lo),     vld1q_u8(cached_lut_lo + 16),
                              vld1q_u8(cached_lut_lo + 32), vld1q_u8(cached_lut_lo + 48)}};
    uint8x16x4_t v_lut_hi = {{vld1q_u8(cached_lut_hi),     vld1q_u8(cached_lut_hi + 16),
                              vld1q_u8(cached_lut_hi + 32), vld1q_u8(cached_lut_hi + 48)}};
    const float inv_sqrt_dim = 1.0f / sqrtf(static_cast<float>(dim));

    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    size_t i = 0;
    size_t p = 0;
    for (; i + 16 <= dim; i += 16) {
        uint8_t idx[16] __attribute__((aligned(16)));
        for (int g = 0; g < 4; g++) {
            uint8_t b0 = packed[p + 3*g];
            uint8_t b1 = packed[p + 3*g + 1];
            uint8_t b2 = packed[p + 3*g + 2];
            idx[4*g]     = b0 & 0x3F;
            idx[4*g + 1] = static_cast<uint8_t>((b0 >> 6) | ((b1 & 0x0F) << 2));
            idx[4*g + 2] = static_cast<uint8_t>((b1 >> 4) | ((b2 & 0x03) << 4));
            idx[4*g + 3] = static_cast<uint8_t>(b2 >> 2);
        }
        p += 12;

        uint8x16_t idx_vec = vld1q_u8(idx);
        uint8x16_t lo_bytes = vqtbl4q_u8(v_lut_lo, idx_vec);
        uint8x16_t hi_bytes = vqtbl4q_u8(v_lut_hi, idx_vec);

        uint8x16_t fp16_pairs_lo = vzip1q_u8(lo_bytes, hi_bytes);
        uint8x16_t fp16_pairs_hi = vzip2q_u8(lo_bytes, hi_bytes);
        float16x8_t f16_lo = vreinterpretq_f16_u8(fp16_pairs_lo);
        float16x8_t f16_hi = vreinterpretq_f16_u8(fp16_pairs_hi);

        float32x4_t f32_0 = vcvt_f32_f16(vget_low_f16(f16_lo));
        float32x4_t f32_1 = vcvt_f32_f16(vget_high_f16(f16_lo));
        float32x4_t f32_2 = vcvt_f32_f16(vget_low_f16(f16_hi));
        float32x4_t f32_3 = vcvt_f32_f16(vget_high_f16(f16_hi));

        acc0 = vfmaq_f32(acc0, vld1q_f32(q + i),      f32_0);
        acc1 = vfmaq_f32(acc1, vld1q_f32(q + i + 4),  f32_1);
        acc2 = vfmaq_f32(acc2, vld1q_f32(q + i + 8),  f32_2);
        acc3 = vfmaq_f32(acc3, vld1q_f32(q + i + 12), f32_3);
    }

    float result = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1),
                                        vaddq_f32(acc2, acc3)));

    for (; i + 4 <= dim; i += 4) {
        uint8_t b0 = packed[p];
        uint8_t b1 = packed[p + 1];
        uint8_t b2 = packed[p + 2];
        uint8_t c0 = b0 & 0x3F;
        uint8_t c1 = uint8_t((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F;
        uint8_t c2 = uint8_t((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F;
        uint8_t c3 = uint8_t(b2 >> 2) & 0x3F;
        result += q[i]     * centroids[c0] * inv_sqrt_dim;
        result += q[i + 1] * centroids[c1] * inv_sqrt_dim;
        result += q[i + 2] * centroids[c2] * inv_sqrt_dim;
        result += q[i + 3] * centroids[c3] * inv_sqrt_dim;
        p += 3;
    }

    return result;
}

void accumulate_6bit_f32(const uint8_t* __restrict packed, float weight_radius,
                         float* __restrict accum, size_t dim) {
    const float* centroids = tq_6bit_centroids();
    const float inv_sqrt_dim = 1.0f / sqrtf(static_cast<float>(dim));
    const float wr = inv_sqrt_dim * weight_radius;

    __fp16 lut_fp16[64] __attribute__((aligned(16)));
    for (int i = 0; i < 64; i++) lut_fp16[i] = static_cast<__fp16>(centroids[i] * wr);

    uint8_t lut_lo[64] __attribute__((aligned(16)));
    uint8_t lut_hi[64] __attribute__((aligned(16)));
    {
        const uint8_t* src = reinterpret_cast<const uint8_t*>(lut_fp16);
        for (int i = 0; i < 64; i++) {
            lut_lo[i] = src[2 * i];
            lut_hi[i] = src[2 * i + 1];
        }
    }
    uint8x16x4_t v_lut_lo = {{vld1q_u8(lut_lo),     vld1q_u8(lut_lo + 16),
                              vld1q_u8(lut_lo + 32), vld1q_u8(lut_lo + 48)}};
    uint8x16x4_t v_lut_hi = {{vld1q_u8(lut_hi),     vld1q_u8(lut_hi + 16),
                              vld1q_u8(lut_hi + 32), vld1q_u8(lut_hi + 48)}};

    size_t i = 0;
    size_t p = 0;
    for (; i + 16 <= dim; i += 16) {
        uint8_t idx[16] __attribute__((aligned(16)));
        for (int g = 0; g < 4; g++) {
            uint8_t b0 = packed[p + 3*g];
            uint8_t b1 = packed[p + 3*g + 1];
            uint8_t b2 = packed[p + 3*g + 2];
            idx[4*g]     = b0 & 0x3F;
            idx[4*g + 1] = static_cast<uint8_t>((b0 >> 6) | ((b1 & 0x0F) << 2));
            idx[4*g + 2] = static_cast<uint8_t>((b1 >> 4) | ((b2 & 0x03) << 4));
            idx[4*g + 3] = static_cast<uint8_t>(b2 >> 2);
        }
        p += 12;

        uint8x16_t idx_vec = vld1q_u8(idx);
        uint8x16_t lo_bytes = vqtbl4q_u8(v_lut_lo, idx_vec);
        uint8x16_t hi_bytes = vqtbl4q_u8(v_lut_hi, idx_vec);

        uint8x16_t fp16_pairs_lo = vzip1q_u8(lo_bytes, hi_bytes);
        uint8x16_t fp16_pairs_hi = vzip2q_u8(lo_bytes, hi_bytes);
        float16x8_t f16_lo = vreinterpretq_f16_u8(fp16_pairs_lo);
        float16x8_t f16_hi = vreinterpretq_f16_u8(fp16_pairs_hi);

        float32x4_t f32_0 = vcvt_f32_f16(vget_low_f16(f16_lo));
        float32x4_t f32_1 = vcvt_f32_f16(vget_high_f16(f16_lo));
        float32x4_t f32_2 = vcvt_f32_f16(vget_low_f16(f16_hi));
        float32x4_t f32_3 = vcvt_f32_f16(vget_high_f16(f16_hi));

        float32x4_t a0 = vaddq_f32(vld1q_f32(accum + i),      f32_0);
        float32x4_t a1 = vaddq_f32(vld1q_f32(accum + i + 4),  f32_1);
        float32x4_t a2 = vaddq_f32(vld1q_f32(accum + i + 8),  f32_2);
        float32x4_t a3 = vaddq_f32(vld1q_f32(accum + i + 12), f32_3);
        vst1q_f32(accum + i,      a0);
        vst1q_f32(accum + i + 4,  a1);
        vst1q_f32(accum + i + 8,  a2);
        vst1q_f32(accum + i + 12, a3);
    }

    for (; i + 4 <= dim; i += 4) {
        uint8_t b0 = packed[p];
        uint8_t b1 = packed[p + 1];
        uint8_t b2 = packed[p + 2];
        uint8_t c0 = b0 & 0x3F;
        uint8_t c1 = uint8_t((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F;
        uint8_t c2 = uint8_t((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F;
        uint8_t c3 = uint8_t(b2 >> 2) & 0x3F;
        accum[i]     += centroids[c0] * wr;
        accum[i + 1] += centroids[c1] * wr;
        accum[i + 2] += centroids[c2] * wr;
        accum[i + 3] += centroids[c3] * wr;
        p += 3;
    }
}

static const float* tq_8bit_centroids() {
    static const std::array<float, 256> values = []() {
        constexpr int N = 256;
        constexpr int ITERS = 80;
        constexpr float kInvSqrt2 = 0.70710678118f;
        constexpr float kInvSqrt2pi = 0.39894228040f;

        std::array<float, N> c{};
        for (int i = 0; i < N; i++)
            c[i] = -4.0f + 8.0f * (float(i) + 0.5f) / float(N);

        std::array<float, N - 1> b{};
        for (int iter = 0; iter < ITERS; iter++) {
            for (int i = 0; i < N - 1; i++) b[i] = 0.5f * (c[i] + c[i+1]);
            for (int i = 0; i < N; i++) {
                float lo = (i == 0) ? -1e30f : b[i - 1];
                float hi = (i == N - 1) ? 1e30f : b[i];
                auto phi = [&](float x){ return kInvSqrt2pi * std::exp(-0.5f * x * x); };
                auto Phi = [&](float x){ return 0.5f * (1.0f + std::erf(x * kInvSqrt2)); };
                float num = phi(lo) - phi(hi);
                float den = Phi(hi) - Phi(lo);
                if (den > 1e-30f) c[i] = num / den;
            }
        }
        return c;
    }();
    return values.data();
}

static void quantize_8bit(const float* src, uint8_t* dst, size_t dim) {
    const float* centroids = tq_8bit_centroids();
    const float sqrt_dim = sqrtf(static_cast<float>(dim));

    auto encode = [&](float x) -> uint8_t {
        const float v = x * sqrt_dim;
        int lo = 0, hi = 255;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            float boundary = 0.5f * (centroids[mid] + centroids[mid + 1]);
            if (v < boundary) hi = mid;
            else              lo = mid + 1;
        }
        return static_cast<uint8_t>(lo);
    };

    for (size_t i = 0; i < dim; i++) dst[i] = encode(src[i]);
}

void dequantize_8bit(const uint8_t* src, float* dst, size_t dim) {
    const float* centroids = tq_8bit_centroids();
    const float inv_sqrt_dim = 1.0f / sqrtf(static_cast<float>(dim));
    for (size_t i = 0; i < dim; i++) dst[i] = centroids[src[i]] * inv_sqrt_dim;
}

float dot_8bit_f32(const float* __restrict q, const uint8_t* __restrict packed, size_t dim) {
    const float* centroids = tq_8bit_centroids();
    const float inv_sqrt_dim = 1.0f / sqrtf(static_cast<float>(dim));
    float result = 0.0f;
    for (size_t i = 0; i < dim; i++) result += q[i] * centroids[packed[i]];
    return result * inv_sqrt_dim;
}

void accumulate_8bit_f32(const uint8_t* __restrict packed, float weight_radius,
                         float* __restrict accum, size_t dim) {
    const float* centroids = tq_8bit_centroids();
    const float inv_sqrt_dim = 1.0f / sqrtf(static_cast<float>(dim));
    const float wr = inv_sqrt_dim * weight_radius;
    for (size_t i = 0; i < dim; i++) accum[i] += centroids[packed[i]] * wr;
}

static void quantize_nbit(const float* src, uint8_t* dst, size_t dim, size_t angle_bits) {
    if (angle_bits == 2)      quantize_2bit(src, dst, dim);
    else if (angle_bits == 6) quantize_6bit(src, dst, dim);
    else if (angle_bits == 8) quantize_8bit(src, dst, dim);
    else                      quantize_4bit(src, dst, dim);
}

void cactus_turboquant_init(
    uint8_t* rotation_signs,
    size_t head_dim, uint64_t seed
) {
    assert((head_dim & (head_dim - 1)) == 0 && head_dim >= 8);
    Xoshiro256 rng(seed);

    const size_t total_sign_bytes = turboquant_rotation_signs_bytes(head_dim);
    for (size_t i = 0; i < total_sign_bytes; ) {
        uint64_t w = rng.next();
        size_t n = std::min(size_t(8), total_sign_bytes - i);
        std::memcpy(rotation_signs + i, &w, n);
        i += n;
    }
}


void cactus_turboquant_encode_kv_fp16(
    const __fp16* src, float* dst_radii, uint8_t* dst_angles,
    const uint8_t* rotation_signs,
    size_t seq_len, size_t kv_heads, size_t head_dim,
    size_t angle_bits
) {
    assert((head_dim & (head_dim - 1)) == 0);
    assert(head_dim <= 512);

    const size_t angles_bytes = turboquant_angles_bytes_per_head(head_dim, angle_bits);

    CactusThreading::parallel_for_static(
        seq_len * kv_heads, CactusThreading::Thresholds::ELEMENT_WISE,
        [=](size_t start, size_t end) {
            std::vector<float> _buf(head_dim);
            float* buf = _buf.data();

            for (size_t idx = start; idx < end; idx++) {
                const __fp16* in = src + idx * head_dim;

                float32x4_t norm_acc0 = vdupq_n_f32(0.0f);
                float32x4_t norm_acc1 = vdupq_n_f32(0.0f);
                size_t d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    float16x8_t h0 = vld1q_f16(in + d);
                    float16x8_t h1 = vld1q_f16(in + d + 8);
                    float32x4_t a = vcvt_f32_f16(vget_low_f16(h0));
                    float32x4_t b = vcvt_f32_f16(vget_high_f16(h0));
                    float32x4_t c = vcvt_f32_f16(vget_low_f16(h1));
                    float32x4_t e = vcvt_f32_f16(vget_high_f16(h1));
                    vst1q_f32(buf + d,      a);
                    vst1q_f32(buf + d + 4,  b);
                    vst1q_f32(buf + d + 8,  c);
                    vst1q_f32(buf + d + 12, e);
                    norm_acc0 = vfmaq_f32(vfmaq_f32(norm_acc0, a, a), b, b);
                    norm_acc1 = vfmaq_f32(vfmaq_f32(norm_acc1, c, c), e, e);
                }
                for (; d + 8 <= head_dim; d += 8) {
                    float16x8_t h = vld1q_f16(in + d);
                    float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
                    float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
                    vst1q_f32(buf + d,     lo);
                    vst1q_f32(buf + d + 4, hi);
                    norm_acc0 = vfmaq_f32(vfmaq_f32(norm_acc0, lo, lo), hi, hi);
                }
                float norm_sq = vaddvq_f32(vaddq_f32(norm_acc0, norm_acc1));
                for (; d < head_dim; d++) {
                    float v = static_cast<float>(in[d]);
                    buf[d] = v;
                    norm_sq += v * v;
                }

                float radius = sqrtf(norm_sq);
                dst_radii[idx] = radius;

                if (radius > 1e-10f) {
                    const float inv_r = 1.0f / radius;
                    float32x4_t inv_r_vec = vdupq_n_f32(inv_r);
                    d = 0;
                    for (; d + 16 <= head_dim; d += 16) {
                        vst1q_f32(buf + d,      vmulq_f32(vld1q_f32(buf + d),      inv_r_vec));
                        vst1q_f32(buf + d + 4,  vmulq_f32(vld1q_f32(buf + d + 4),  inv_r_vec));
                        vst1q_f32(buf + d + 8,  vmulq_f32(vld1q_f32(buf + d + 8),  inv_r_vec));
                        vst1q_f32(buf + d + 12, vmulq_f32(vld1q_f32(buf + d + 12), inv_r_vec));
                    }
                    for (; d < head_dim; d++) buf[d] *= inv_r;
                }

                rotate_forward(buf, rotation_signs, head_dim);

                uint8_t* q_angles = dst_angles + idx * angles_bytes;
                quantize_nbit(buf, q_angles, head_dim, angle_bits);
            }
        });

    static std::atomic<size_t> tq_encode_call_counter{0};
    if (const char* dump_dir = std::getenv("CACTUS_CACHE_DIFF_DUMP")) {
        const size_t call_idx = tq_encode_call_counter.fetch_add(1, std::memory_order_relaxed);
        if (call_idx < 100) {
            char path[1024];
            std::snprintf(path, sizeof(path), "%s/cache_%05zu.bin", dump_dir, call_idx);
            FILE* f = std::fopen(path, "wb");
            if (f) {
                size_t hdr[5] = {call_idx, seq_len, kv_heads, head_dim, angle_bits};
                std::fwrite(hdr, sizeof(size_t), 5, f);
                std::fwrite(dst_radii, sizeof(float), seq_len * kv_heads, f);
                std::fwrite(dst_angles, 1, seq_len * kv_heads * angles_bytes, f);
                std::fclose(f);
            }
        }
    }
}

void cactus_turboquant_decode_kv_fp16(
    const float* radii, const uint8_t* angles, const uint8_t* rotation_signs,
    __fp16* dst, size_t seq_len, size_t kv_heads, size_t head_dim, size_t angle_bits
) {
    const size_t angles_bytes = turboquant_angles_bytes_per_head(head_dim, angle_bits);
    CactusThreading::parallel_for_static(
        seq_len * kv_heads, CactusThreading::Thresholds::ELEMENT_WISE,
        [=](size_t start, size_t end) {
            std::vector<float> _buf(head_dim);
            float* buf = _buf.data();
            for (size_t idx = start; idx < end; idx++) {
                float radius = radii[idx];
                if (angle_bits == 2)      dequantize_2bit(angles + idx * angles_bytes, buf, head_dim);
                else if (angle_bits == 6) dequantize_6bit(angles + idx * angles_bytes, buf, head_dim);
                else if (angle_bits == 8) dequantize_8bit(angles + idx * angles_bytes, buf, head_dim);
                else                      dequantize_4bit(angles + idx * angles_bytes, buf, head_dim);
                float32x4_t r_vec = vdupq_n_f32(radius);
                size_t d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    vst1q_f32(&buf[d],      vmulq_f32(vld1q_f32(&buf[d]),      r_vec));
                    vst1q_f32(&buf[d + 4],  vmulq_f32(vld1q_f32(&buf[d + 4]),  r_vec));
                    vst1q_f32(&buf[d + 8],  vmulq_f32(vld1q_f32(&buf[d + 8]),  r_vec));
                    vst1q_f32(&buf[d + 12], vmulq_f32(vld1q_f32(&buf[d + 12]), r_vec));
                }
                for (; d + 4 <= head_dim; d += 4) {
                    vst1q_f32(&buf[d], vmulq_f32(vld1q_f32(&buf[d]), r_vec));
                }
                for (; d < head_dim; d++) buf[d] *= radius;

                rotate_inverse(buf, rotation_signs, head_dim);
                __fp16* out = dst + idx * head_dim;
                d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    vst1q_f16(&out[d],     vcombine_f16(vcvt_f16_f32(vld1q_f32(&buf[d])),      vcvt_f16_f32(vld1q_f32(&buf[d + 4]))));
                    vst1q_f16(&out[d + 8], vcombine_f16(vcvt_f16_f32(vld1q_f32(&buf[d + 8])), vcvt_f16_f32(vld1q_f32(&buf[d + 12]))));
                }
                for (; d + 8 <= head_dim; d += 8) {
                    vst1q_f16(&out[d], vcombine_f16(vcvt_f16_f32(vld1q_f32(&buf[d])), vcvt_f16_f32(vld1q_f32(&buf[d + 4]))));
                }
                for (; d < head_dim; d++) out[d] = static_cast<__fp16>(buf[d]);
            }
        });
}

