#include "kernel.h"

#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

constexpr uint8_t kMagic[4] = {'C', 'A', 'C', 'T'};
constexpr uint32_t kPrecisionTQ2 = 10;

inline uint32_t rd_u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t rd_u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

inline float16x8_t signs_to_fp16_mul(const int8_t* signs) {
    return vcvtq_f16_s16(vmovl_s8(vld1_s8(signs)));
}

inline void fwht128_f16(__fp16* x) {
    float16x8_t v[16];
    for (int i = 0; i < 16; ++i) v[i] = vld1q_f16(x + i * 8);

    for (int i = 0; i < 16; ++i) {
        float16x8_t r = vreinterpretq_f16_u16(
            vrev32q_u16(vreinterpretq_u16_f16(v[i])));
        float16x8_t s = vaddq_f16(v[i], r);
        float16x8_t d = vsubq_f16(v[i], r);
        v[i] = vreinterpretq_f16_u16(vtrn1q_u16(
            vreinterpretq_u16_f16(s), vreinterpretq_u16_f16(d)));
    }
    for (int i = 0; i < 16; ++i) {
        float32x4_t f32 = vreinterpretq_f32_f16(v[i]);
        float16x8_t a = vreinterpretq_f16_f32(vtrn1q_f32(f32, f32));
        float16x8_t b = vreinterpretq_f16_f32(vtrn2q_f32(f32, f32));
        float16x8_t s = vaddq_f16(a, b);
        float16x8_t d = vsubq_f16(a, b);
        v[i] = vreinterpretq_f16_f32(vtrn1q_f32(
            vreinterpretq_f32_f16(s), vreinterpretq_f32_f16(d)));
    }
    for (int i = 0; i < 16; ++i) {
        float16x4_t lo = vget_low_f16(v[i]);
        float16x4_t hi = vget_high_f16(v[i]);
        v[i] = vcombine_f16(vadd_f16(lo, hi), vsub_f16(lo, hi));
    }

    auto pass = [&](int s) {
        for (int base = 0; base < 16; base += (s << 1)) {
            for (int j = 0; j < s; ++j) {
                float16x8_t a = v[base + j];
                float16x8_t b = v[base + j + s];
                v[base + j]     = vaddq_f16(a, b);
                v[base + j + s] = vsubq_f16(a, b);
            }
        }
    };
    pass(1);
    pass(2);
    pass(4);
    pass(8);

    float16x8_t iv = vdupq_n_f16((__fp16)(1.0f / std::sqrt(128.0f)));
    for (int i = 0; i < 16; ++i) vst1q_f16(x + i * 8, vmulq_f16(v[i], iv));
}

inline void fwht_scalar_f16(__fp16* x, size_t n) {
    for (size_t h = 1; h < n; h <<= 1) {
        for (size_t i = 0; i < n; i += (h << 1)) {
            for (size_t j = i; j < i + h; ++j) {
                __fp16 a = x[j];
                __fp16 b = x[j + h];
                x[j]     = (__fp16)(a + b);
                x[j + h] = (__fp16)(a - b);
            }
        }
    }
    const __fp16 inv = (__fp16)(1.0f / std::sqrt((float)n));
    for (size_t i = 0; i < n; ++i) x[i] = (__fp16)(x[i] * inv);
}

void dequant_group(const CactusTQ2* tq2, uint32_t token_id, uint32_t group_idx, __fp16* out) {
    const uint32_t gs = tq2->group_size;

    __fp16 cb[4] = {
        (__fp16)tq2->codebook[0], (__fp16)tq2->codebook[1],
        (__fp16)tq2->codebook[2], (__fp16)tq2->codebook[3],
    };

    __fp16 tmp[256];
    const uint8_t* packed_row =
        tq2->packed + ((size_t)token_id * tq2->num_groups + group_idx) * tq2->per_group_bytes;
    for (uint32_t k = 0; k < gs; k += 4) {
        uint8_t byte = packed_row[k >> 2];
        tmp[k    ] = cb[(byte     ) & 0x3];
        tmp[k + 1] = cb[(byte >> 2) & 0x3];
        tmp[k + 2] = cb[(byte >> 4) & 0x3];
        tmp[k + 3] = cb[(byte >> 6) & 0x3];
    }

    __fp16 y[256];
    for (uint32_t k = 0; k < gs; ++k) y[k] = tmp[tq2->inv_permutation[k]];

    for (uint32_t k = 0; k < gs; k += 8) {
        vst1q_f16(y + k, vmulq_f16(vld1q_f16(y + k), signs_to_fp16_mul(tq2->right_signs + k)));
    }

    if (gs == 128) fwht128_f16(y);
    else           fwht_scalar_f16(y, gs);

    for (uint32_t j = 0; j < gs; j += 8) {
        vst1q_f16(y + j, vmulq_f16(vld1q_f16(y + j), signs_to_fp16_mul(tq2->left_signs + j)));
    }

    const __fp16 rn = tq2->scales[(size_t)token_id * tq2->num_groups + group_idx];
    float16x8_t rn_v = vdupq_n_f16(rn);
    for (uint32_t j = 0; j < gs; j += 8) {
        vst1q_f16(y + j, vmulq_f16(vld1q_f16(y + j), rn_v));
    }

    const __fp16* is = tq2->input_scale + group_idx * gs;
    for (uint32_t j = 0; j < gs; j += 8) {
        vst1q_f16(out + j, vdivq_f16(vld1q_f16(y + j), vld1q_f16(is + j)));
    }
}

}  // namespace


int cactus_tq2_load(CactusTQ2* out, const void* blob_ptr, size_t blob_size) {
    if (out == nullptr || blob_ptr == nullptr) return 0;
    if (blob_size < 128) return 0;

    const uint8_t* blob = reinterpret_cast<const uint8_t*>(blob_ptr);
    if (std::memcmp(blob, kMagic, 4) != 0) return 0;

    uint32_t ndim           = rd_u32(blob + 12);
    uint64_t dim0           = rd_u64(blob + 16);
    uint64_t dim1           = rd_u64(blob + 24);
    uint32_t precision      = rd_u32(blob + 48);
    uint64_t indices_bytes  = rd_u64(blob + 52);
    uint64_t scales_bytes   = rd_u64(blob + 60);
    uint32_t group_size     = rd_u32(blob + 68);
    uint32_t num_groups     = rd_u32(blob + 72);
    uint32_t bits_per_index = rd_u32(blob + 76);

    if (ndim != 2)                                 return 0;
    if (precision != kPrecisionTQ2)                return 0;
    if (bits_per_index != 2)                       return 0;
    if (group_size == 0 || group_size > 256)       return 0;
    if ((group_size & (group_size - 1)) != 0)      return 0;
    if ((group_size & 7) != 0)                     return 0;
    if ((uint64_t)num_groups * group_size != dim1) return 0;

    uint64_t off_cb  = rd_u64(blob + 80);
    uint64_t off_is  = rd_u64(blob + 88);
    uint64_t off_rot = rd_u64(blob + 96);
    uint64_t off_sc  = rd_u64(blob + 104);
    uint64_t off_ix  = rd_u64(blob + 112);
    uint64_t total   = rd_u64(blob + 120);
    if (total != blob_size) return 0;

    out->vocab           = (uint32_t)dim0;
    out->group_size      = group_size;
    out->num_groups      = num_groups;
    out->per_group_bytes = group_size / 4;

    out->codebook    = reinterpret_cast<const float*>(blob + off_cb);
    out->input_scale = reinterpret_cast<const __fp16*>(blob + off_is);
    out->left_signs  = reinterpret_cast<const int8_t*>(blob + off_rot);
    out->right_signs = reinterpret_cast<const int8_t*>(blob + off_rot + group_size);
    out->permutation = reinterpret_cast<const uint32_t*>(blob + off_rot + 2u * group_size);
    out->scales      = reinterpret_cast<const __fp16*>(blob + off_sc);
    out->packed      = reinterpret_cast<const uint8_t*>(blob + off_ix);

    if (scales_bytes  != (uint64_t)out->vocab * num_groups * sizeof(__fp16))     return 0;
    if (indices_bytes != (uint64_t)out->vocab * num_groups * out->per_group_bytes) return 0;

    for (uint32_t i = 0; i < group_size; ++i) {
        uint32_t p = out->permutation[i];
        if (p >= group_size) return 0;
        out->inv_permutation[p] = i;
    }
    return 1;
}

void cactus_tq2_dequant_row(const CactusTQ2* tq2, uint32_t token_id, __fp16* out) {
    const uint32_t gs = tq2->group_size;
    for (uint32_t g = 0; g < tq2->num_groups; ++g) {
        dequant_group(tq2, token_id, g, out + g * gs);
    }
}
