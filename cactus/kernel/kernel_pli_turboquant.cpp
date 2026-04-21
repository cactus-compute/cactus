// Scalar-free NEON fp16 implementation of Precision::INT2_HADAMARD
// (PLI TurboQuant) dequantization. Public ABI in kernel.h.
//
// On-disk layout (little-endian, 32-byte aligned sections):
//   [128-byte header]: magic "CACT" | flags | alignment | ndim | shape[4]
//                      precision=10 | indices_bytes | scales_bytes | group_size
//                      | num_groups | bits_per_index | offset_codebook
//                      | offset_input_scale | offset_rotation | offset_scales
//                      | offset_indices | total_bytes
//   [codebook]        : 2^bits fp32 values
//   [input_scale]     : pli_dim fp16 values (per-dim TurboQuant pre-scale)
//   [rotation]        : group_size int8 left_signs,
//                       group_size int8 right_signs,
//                       group_size uint32 permutation
//   [scales]          : vocab * num_groups fp16 per-(row, group) norms
//   [indices]         : vocab * num_groups * group_size * bits / 8 packed bytes
//
// Dequant per (token, group), all ops in fp16:
//   1. unpack 2-bit indices → fp16 codebook values
//   2. y[k] = tmp[inv_perm[k]]         (undo column permutation)
//   3. y   *= right_signs              (undo diag(R_s))
//   4. y   = FWHT(y) / sqrt(gs)        (apply base Hadamard; self-inverse)
//   5. y  *= left_signs                (undo diag(L))
//   6. y  *= row_norm[token, group]    (restore unit-vector norm)
//   7. out = y / input_scale[group_slice]

#include "kernel.h"

#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

constexpr uint8_t kMagic[4] = {'C', 'A', 'C', 'T'};
constexpr uint32_t kPrecisionInt2Hadamard = 10;

inline uint32_t rd_u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t rd_u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// Broadcast ±1 int8 signs into a fp16x8 ±1.0 multiplier vector.
inline float16x8_t signs_to_fp16_mul(const int8_t* signs) {
    int8x8_t s8  = vld1_s8(signs);
    int16x8_t s16 = vmovl_s8(s8);
    return vcvtq_f16_s16(s16);
}

// 128-point in-place FWHT on fp16, normalized by 1/sqrt(128).
// Assumes 16-byte alignment (every mmap'd section is 32-byte aligned).
// Layout: 16 float16x8_t lanes -> v[0..15].
//
// 7 butterfly passes: h = 1, 2, 4 (in-register) and 8, 16, 32, 64 (across lanes).
// The in-register butterflies fuse add/sub + lane interleave into a single
// vtrn1q_f{16,32}(sum, diff) so we never touch scalar lanes.
inline void fwht128_f16(__fp16* x) {
    float16x8_t v[16];
    for (int i = 0; i < 16; ++i) v[i] = vld1q_f16(x + i * 8);

    // h=1: pair (0,1), (2,3), (4,5), (6,7) inside each lane.
    //   vrev32q_u16 swaps adjacent 16-bit words within each 32-bit lane
    //   (no fp16-typed intrinsic exists; reinterpret for the shuffle).
    for (int i = 0; i < 16; ++i) {
        float16x8_t r = vreinterpretq_f16_u16(
            vrev32q_u16(vreinterpretq_u16_f16(v[i])));
        float16x8_t s = vaddq_f16(v[i], r);
        float16x8_t d = vsubq_f16(v[i], r);
        v[i] = vreinterpretq_f16_u16(vtrn1q_u16(
            vreinterpretq_u16_f16(s), vreinterpretq_u16_f16(d)));
    }
    // h=2: pair (0,2),(1,3),(4,6),(5,7) inside each lane (stride 2).
    //   Treat each lane as 4 × fp32 and use vtrnq_f32 to gather/butterfly.
    for (int i = 0; i < 16; ++i) {
        float32x4_t f32 = vreinterpretq_f32_f16(v[i]);
        float32x4_t a = vtrn1q_f32(f32, f32);   // [X0 X0 X2 X2]  (Xk = pair (2k,2k+1))
        float32x4_t b = vtrn2q_f32(f32, f32);   // [X1 X1 X3 X3]
        float16x8_t af = vreinterpretq_f16_f32(a);
        float16x8_t bf = vreinterpretq_f16_f32(b);
        float16x8_t s = vaddq_f16(af, bf);
        float16x8_t d = vsubq_f16(af, bf);
        v[i] = vreinterpretq_f16_f32(vtrn1q_f32(
            vreinterpretq_f32_f16(s), vreinterpretq_f32_f16(d)));
    }
    // h=4: butterfly across the two 4-lane halves of each 8-lane vector.
    for (int i = 0; i < 16; ++i) {
        float16x4_t lo = vget_low_f16(v[i]);
        float16x4_t hi = vget_high_f16(v[i]);
        v[i] = vcombine_f16(vadd_f16(lo, hi), vsub_f16(lo, hi));
    }

    // Cross-lane butterflies. At stride s vectors: pair (v[j], v[j+s]).
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
    pass(1);   // h = 8
    pass(2);   // h = 16
    pass(4);   // h = 32
    pass(8);   // h = 64

    const __fp16 inv = (__fp16)(1.0f / std::sqrt(128.0f));
    float16x8_t iv = vdupq_n_f16(inv);
    for (int i = 0; i < 16; ++i) vst1q_f16(x + i * 8, vmulq_f16(v[i], iv));
}

// Scalar fp16 FWHT for group_size values other than 128. Real files ship
// group_size=128; this path is here for safety only.
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

void dequant_one_group(const CactusPliTurboquant* pli,
                       uint32_t token_id, uint32_t group_idx,
                       __fp16* out) {
    const uint32_t gs = pli->group_size;

    // Codebook: file stores fp32 (2^bits = 4 entries here). Convert once.
    __fp16 cb[4];
    for (uint32_t i = 0; i < (1u << pli->bits_per_index); ++i) {
        cb[i] = (__fp16)pli->codebook[i];
    }

    // 1. Unpack 2-bit indices → fp16 codebook values. Gs is always a multiple
    //    of 4 (indices pack 4/byte), so one byte yields four output lanes.
    __fp16 tmp[256];
    const uint8_t* packed_row =
        pli->packed + ((size_t)token_id * pli->num_groups + group_idx) * pli->per_group_bytes;
    for (uint32_t k = 0; k < gs; k += 4) {
        uint8_t byte = packed_row[k >> 2];
        tmp[k    ] = cb[(byte     ) & 0x3];
        tmp[k + 1] = cb[(byte >> 2) & 0x3];
        tmp[k + 2] = cb[(byte >> 4) & 0x3];
        tmp[k + 3] = cb[(byte >> 6) & 0x3];
    }

    // 2. Undo column permutation.
    __fp16 y[256];
    for (uint32_t k = 0; k < gs; ++k) y[k] = tmp[pli->inv_permutation[k]];

    // 3. Undo diag(right_signs).
    for (uint32_t k = 0; k < gs; k += 8) {
        float16x8_t yv = vld1q_f16(y + k);
        float16x8_t sv = signs_to_fp16_mul(pli->right_signs + k);
        vst1q_f16(y + k, vmulq_f16(yv, sv));
    }

    // 4. Apply base Hadamard (self-inverse up to the 1/sqrt(n) factor).
    if (gs == 128) {
        fwht128_f16(y);
    } else {
        fwht_scalar_f16(y, gs);
    }

    // 5. Undo diag(left_signs).
    for (uint32_t j = 0; j < gs; j += 8) {
        float16x8_t yv = vld1q_f16(y + j);
        float16x8_t sv = signs_to_fp16_mul(pli->left_signs + j);
        vst1q_f16(y + j, vmulq_f16(yv, sv));
    }

    // 6. Multiply by fp16 row-norm.
    const __fp16* row_norms = reinterpret_cast<const __fp16*>(pli->scales);
    const __fp16 rn = row_norms[(size_t)token_id * pli->num_groups + group_idx];
    float16x8_t rn_v = vdupq_n_f16(rn);
    for (uint32_t j = 0; j < gs; j += 8) {
        vst1q_f16(y + j, vmulq_f16(vld1q_f16(y + j), rn_v));
    }

    // 7. Divide by fp16 per-dim input_scale for this group's slice.
    const __fp16* is = reinterpret_cast<const __fp16*>(pli->input_scale) + group_idx * gs;
    for (uint32_t j = 0; j < gs; j += 8) {
        float16x8_t yv = vld1q_f16(y + j);
        float16x8_t sv = vld1q_f16(is + j);
        vst1q_f16(out + j, vdivq_f16(yv, sv));
    }
}

}  // namespace


int cactus_pli_tq_load(CactusPliTurboquant* out,
                       const void* blob_ptr,
                       size_t blob_size) {
    if (out == nullptr || blob_ptr == nullptr) return 0;
    if (blob_size < 128) return 0;

    const uint8_t* blob = reinterpret_cast<const uint8_t*>(blob_ptr);
    if (std::memcmp(blob, kMagic, 4) != 0) return 0;

    uint32_t ndim             = rd_u32(blob + 12);
    uint64_t dim0             = rd_u64(blob + 16);
    uint64_t dim1             = rd_u64(blob + 24);
    uint32_t precision        = rd_u32(blob + 48);
    uint64_t indices_bytes    = rd_u64(blob + 52);
    uint64_t scales_bytes     = rd_u64(blob + 60);
    uint32_t group_size       = rd_u32(blob + 68);
    uint32_t num_groups       = rd_u32(blob + 72);
    uint32_t bits_per_index   = rd_u32(blob + 76);

    if (ndim != 2)                                 return 0;
    if (precision != kPrecisionInt2Hadamard)       return 0;
    if (bits_per_index != 2)                       return 0;
    if (group_size == 0 || group_size > 256)       return 0;
    if ((group_size & (group_size - 1)) != 0)      return 0;   // power of two
    if ((group_size & 7) != 0)                     return 0;   // 8-lane NEON
    if ((uint64_t)num_groups * group_size != dim1) return 0;

    uint64_t off_cb  = rd_u64(blob + 80);
    uint64_t off_is  = rd_u64(blob + 88);
    uint64_t off_rot = rd_u64(blob + 96);
    uint64_t off_sc  = rd_u64(blob + 104);
    uint64_t off_ix  = rd_u64(blob + 112);
    uint64_t total   = rd_u64(blob + 120);
    if (total != blob_size) return 0;

    out->vocab            = (uint32_t)dim0;
    out->pli_dim          = (uint32_t)dim1;
    out->group_size       = group_size;
    out->num_groups       = num_groups;
    out->bits_per_index   = bits_per_index;
    out->groups_per_layer = 0;  // caller sets from model config, or num_groups for full row
    out->per_group_bytes  = group_size * bits_per_index / 8;

    out->codebook    = reinterpret_cast<const float*>(blob + off_cb);
    out->input_scale = reinterpret_cast<const uint16_t*>(blob + off_is);
    out->left_signs  = reinterpret_cast<const int8_t*>(blob + off_rot);
    out->right_signs = reinterpret_cast<const int8_t*>(blob + off_rot + group_size);
    out->permutation = reinterpret_cast<const uint32_t*>(blob + off_rot + 2u * group_size);
    out->scales      = reinterpret_cast<const uint16_t*>(blob + off_sc);
    out->packed      = reinterpret_cast<const uint8_t*>(blob + off_ix);

    uint64_t expected_sc = (uint64_t)out->vocab * num_groups * sizeof(uint16_t);
    uint64_t expected_ix = (uint64_t)out->vocab * num_groups * out->per_group_bytes;
    if (scales_bytes != expected_sc) return 0;
    if (indices_bytes != expected_ix) return 0;

    for (uint32_t i = 0; i < group_size; ++i) {
        uint32_t p = out->permutation[i];
        if (p >= group_size) return 0;
        out->inv_permutation[p] = i;
    }
    return 1;
}


void cactus_pli_tq_dequant_layer_slice(const CactusPliTurboquant* pli,
                                       uint32_t token_id,
                                       uint32_t layer_idx,
                                       __fp16* out) {
    const uint32_t gs  = pli->group_size;
    const uint32_t gpl = pli->groups_per_layer;
    const uint32_t g0  = layer_idx * gpl;
    for (uint32_t g = 0; g < gpl; ++g) {
        dequant_one_group(pli, token_id, g0 + g, out + g * gs);
    }
}
