// Generic TurboQuant-Hadamard kernel for Precision::TQ1/TQ3/TQ4 (1/3/4-bit).
// Single dequant chain (inv_perm -> right_signs -> FWHT -> left_signs -> row_norm
// -> /input_scale) parameterized by bit-width via the unpack step. Orthogonal
// full-width rotation is supported for the embed-table case (used by the LM
// token embedding in the gemma-4-e2b TQH-p3 build).

#include "kernel.h"
#include "kernel_utils.h"

#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace {

constexpr uint8_t kMagic[4] = {'C', 'A', 'C', 'T'};
constexpr uint32_t kPrecisionTQ1 = 9;
constexpr uint32_t kPrecisionTQ3 = 11;
constexpr uint32_t kPrecisionTQ4 = 12;
constexpr uint32_t kRotationHadamard = 0;
constexpr uint32_t kRotationOrthFull = 1;
constexpr size_t kHeaderSize = 136;  // ext header — see python/src/tqh_pack.py

inline uint32_t rd_u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t rd_u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

inline float16x8_t signs_to_fp16_mul(const int8_t* signs) {
    return vcvtq_f16_s16(vmovl_s8(vld1_s8(signs)));
}

inline float16x8_t lookup_tq4_codebook8(uint8x8_t nibbles, uint8x16x2_t cb_bytes) {
    uint8x8_t byte_offsets = vshl_n_u8(nibbles, 1);
    uint8x8_t byte_offsets_hi = vadd_u8(byte_offsets, vdup_n_u8(1));
    uint8x8x2_t zipped = vzip_u8(byte_offsets, byte_offsets_hi);
    uint8x16_t byte_idx = vcombine_u8(zipped.val[0], zipped.val[1]);
    return vreinterpretq_f16_u8(vqtbl2q_u8(cb_bytes, byte_idx));
}

// Same FWHT as kernel_tq2.cpp::fwht128_f16 / fwht_scalar_f16. Duplicated rather
// than shared via header to keep each kernel translation unit self-contained.
inline void fwht128_f16(__fp16* x) {
    float16x8_t v[16];
    for (int i = 0; i < 16; ++i) v[i] = vld1q_f16(x + i * 8);
    for (int i = 0; i < 16; ++i) {
        float16x8_t r = vreinterpretq_f16_u16(vrev32q_u16(vreinterpretq_u16_f16(v[i])));
        float16x8_t s = vaddq_f16(v[i], r);
        float16x8_t d = vsubq_f16(v[i], r);
        v[i] = vreinterpretq_f16_u16(vtrn1q_u16(vreinterpretq_u16_f16(s), vreinterpretq_u16_f16(d)));
    }
    for (int i = 0; i < 16; ++i) {
        float32x4_t f32 = vreinterpretq_f32_f16(v[i]);
        float16x8_t a = vreinterpretq_f16_f32(vtrn1q_f32(f32, f32));
        float16x8_t b = vreinterpretq_f16_f32(vtrn2q_f32(f32, f32));
        float16x8_t s = vaddq_f16(a, b);
        float16x8_t d = vsubq_f16(a, b);
        v[i] = vreinterpretq_f16_f32(vtrn1q_f32(vreinterpretq_f32_f16(s), vreinterpretq_f32_f16(d)));
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
    pass(1); pass(2); pass(4); pass(8);
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

// Unpack 8 indices from a 24-bit LE word (3 bytes), LSB-first. (bits == 3)
inline void unpack_8x3bit_le(const uint8_t* p, uint8_t out[8]) {
    uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
    out[0] = (uint8_t)((w      ) & 0x7);
    out[1] = (uint8_t)((w >>  3) & 0x7);
    out[2] = (uint8_t)((w >>  6) & 0x7);
    out[3] = (uint8_t)((w >>  9) & 0x7);
    out[4] = (uint8_t)((w >> 12) & 0x7);
    out[5] = (uint8_t)((w >> 15) & 0x7);
    out[6] = (uint8_t)((w >> 18) & 0x7);
    out[7] = (uint8_t)((w >> 21) & 0x7);
}

// Codebook lookup into tmp[gs] for a single group, by bit-width. LSB-first
// within byte for 1-bit (idx[i] = (byte >> i) & 1).
inline void codebook_lookup_group(const CactusTQN* tqn, const uint8_t* packed_row,
                                   const __fp16* cb, __fp16* tmp) {
    const uint32_t gs = tqn->group_size;
    switch (tqn->bits) {
        case 1:
            for (uint32_t k = 0; k < gs; k += 8) {
                uint8_t b = packed_row[k >> 3];
                tmp[k    ] = cb[(b     ) & 1];
                tmp[k + 1] = cb[(b >> 1) & 1];
                tmp[k + 2] = cb[(b >> 2) & 1];
                tmp[k + 3] = cb[(b >> 3) & 1];
                tmp[k + 4] = cb[(b >> 4) & 1];
                tmp[k + 5] = cb[(b >> 5) & 1];
                tmp[k + 6] = cb[(b >> 6) & 1];
                tmp[k + 7] = cb[(b >> 7) & 1];
            }
            break;
        case 3:
            for (uint32_t k = 0; k < gs; k += 8) {
                uint8_t idx[8];
                unpack_8x3bit_le(packed_row + (k >> 3) * 3, idx);
                for (int i = 0; i < 8; ++i) tmp[k + i] = cb[idx[i]];
            }
            break;
        case 4:
        default:
        {
            uint8x16x2_t cb_bytes;
            const uint8_t* cb_u8 = reinterpret_cast<const uint8_t*>(cb);
            cb_bytes.val[0] = vld1q_u8(cb_u8);
            cb_bytes.val[1] = vld1q_u8(cb_u8 + 16);
            for (uint32_t k = 0; k < gs; k += 16) {
                uint8x8_t bytes = vld1_u8(packed_row + (k >> 1));
                uint8x8_t lo = vand_u8(bytes, vdup_n_u8(0x0F));
                uint8x8_t hi = vshr_n_u8(bytes, 4);
                vst1q_f16(tmp + k, lookup_tq4_codebook8(vzip1_u8(lo, hi), cb_bytes));
                vst1q_f16(tmp + k + 8, lookup_tq4_codebook8(vzip2_u8(lo, hi), cb_bytes));
            }
            break;
        }
    }
}

// Hadamard-family group dequant. Mirrors kernel_tq2.cpp::dequant_group with
// bit-width-parameterized unpack and a 2^bits-entry codebook.
void dequant_group_hadamard(const CactusTQN* tqn, uint32_t row_id, uint32_t group_idx, __fp16* out) {
    const uint32_t gs = tqn->group_size;
    const uint32_t cb_size = 1u << tqn->bits;

    __fp16 cb[16];
    for (uint32_t i = 0; i < cb_size; ++i) cb[i] = (__fp16)tqn->codebook[i];

    __fp16 tmp[256];
    const uint8_t* packed_row =
        tqn->packed + ((size_t)row_id * tqn->num_groups + group_idx) * tqn->per_group_bytes;
    codebook_lookup_group(tqn, packed_row, cb, tmp);

    __fp16 y[256];
    for (uint32_t k = 0; k < gs; ++k) y[k] = tmp[tqn->inv_permutation[k]];

    for (uint32_t k = 0; k < gs; k += 8) {
        vst1q_f16(y + k, vmulq_f16(vld1q_f16(y + k), signs_to_fp16_mul(tqn->right_signs + k)));
    }

    if (gs == 128) fwht128_f16(y);
    else           fwht_scalar_f16(y, gs);

    for (uint32_t j = 0; j < gs; j += 8) {
        vst1q_f16(y + j, vmulq_f16(vld1q_f16(y + j), signs_to_fp16_mul(tqn->left_signs + j)));
    }

    const __fp16 rn = tqn->scales[(size_t)row_id * tqn->num_groups + group_idx];
    float16x8_t rn_v = vdupq_n_f16(rn);
    for (uint32_t j = 0; j < gs; j += 8) {
        vst1q_f16(y + j, vmulq_f16(vld1q_f16(y + j), rn_v));
    }

    if (tqn->has_input_scale) {
        const __fp16* is = tqn->input_scale + group_idx * gs;
        const __fp16* isr = tqn->input_scale_recip
            ? tqn->input_scale_recip + group_idx * gs
            : nullptr;
        for (uint32_t j = 0; j < gs; j += 8) {
            float16x8_t sv = isr ? vld1q_f16(isr + j)
                                  : vdivq_f16(vdupq_n_f16(1), vld1q_f16(is + j));
            vst1q_f16(out + j, vmulq_f16(vld1q_f16(y + j), sv));
        }
    } else {
        for (uint32_t j = 0; j < gs; j += 8) {
            vst1q_f16(out + j, vld1q_f16(y + j));
        }
    }
}

// Orthogonal full-width row dequant: tmp = cb[idx[0..K]]; out = tmp @ R^T;
// out *= row_norm; out /= input_scale[i]. R is row-major fp16[K*K]; tmp @ R^T
// == R @ tmp (matvec).
void dequant_row_orth(const CactusTQN* tqn, uint32_t row_id, __fp16* out) {
    const uint32_t K = tqn->dim1;
    const uint32_t cb_size = 1u << tqn->bits;

    __fp16 cb[16];
    for (uint32_t i = 0; i < cb_size; ++i) cb[i] = (__fp16)tqn->codebook[i];

    std::vector<float> tmp(K);
    const uint8_t* row = tqn->packed + (size_t)row_id * tqn->per_group_bytes;
    switch (tqn->bits) {
        case 1:
            for (uint32_t k = 0; k < K; k += 8) {
                uint8_t b = row[k >> 3];
                for (int i = 0; i < 8; ++i) tmp[k + i] = (float)cb[(b >> i) & 1];
            }
            break;
        case 3:
            for (uint32_t k = 0; k < K; k += 8) {
                uint8_t idx[8];
                unpack_8x3bit_le(row + (k >> 3) * 3, idx);
                for (int i = 0; i < 8; ++i) tmp[k + i] = (float)cb[idx[i]];
            }
            break;
        default:  // bits == 4
            for (uint32_t k = 0; k < K; k += 2) {
                uint8_t b = row[k >> 1];
                tmp[k    ] = (float)cb[b & 0xF];
                tmp[k + 1] = (float)cb[(b >> 4) & 0xF];
            }
            break;
    }

    const float rn = (float)tqn->scales[row_id];  // num_groups == 1
    for (uint32_t i = 0; i < K; ++i) {
        const __fp16* rrow = tqn->orth_R + (size_t)i * K;
        float32x4_t acc0 = vdupq_n_f32(0.f);
        float32x4_t acc1 = vdupq_n_f32(0.f);
        uint32_t j = 0;
        for (; j + 8 <= K; j += 8) {
            float16x8_t r = vld1q_f16(rrow + j);
            float16x8_t t = vcombine_f16(
                vcvt_f16_f32(vld1q_f32(tmp.data() + j)),
                vcvt_f16_f32(vld1q_f32(tmp.data() + j + 4)));
            float16x8_t prod = vmulq_f16(r, t);
            acc0 = vaddq_f32(acc0, vcvt_f32_f16(vget_low_f16(prod)));
            acc1 = vaddq_f32(acc1, vcvt_f32_f16(vget_high_f16(prod)));
        }
        float acc = vaddvq_f32(acc0) + vaddvq_f32(acc1);
        for (; j < K; ++j) acc += (float)rrow[j] * tmp[j];

        float v = acc * rn;
        if (tqn->has_input_scale) {
            v *= tqn->input_scale_recip ? (float)tqn->input_scale_recip[i]
                                        : 1.0f / (float)tqn->input_scale[i];
        }
        out[i] = (__fp16)v;
    }
}

}  // namespace


int cactus_tqn_load(CactusTQN* out, const void* blob_ptr, size_t blob_size) {
    if (out == nullptr || blob_ptr == nullptr) return 0;
    if (blob_size < kHeaderSize) return 0;

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

    if (ndim != 2) return 0;
    if (precision == kPrecisionTQ1) {
        if (bits_per_index != 1) return 0;
    } else if (precision == kPrecisionTQ3) {
        if (bits_per_index != 3) return 0;
    } else if (precision == kPrecisionTQ4) {
        if (bits_per_index != 4) return 0;
    } else {
        return 0;
    }
    if (group_size == 0)                            return 0;
    if ((uint64_t)num_groups * group_size != dim1)  return 0;
    if (bits_per_index == 1 && (group_size & 7))    return 0;
    if (bits_per_index == 3 && (group_size & 7))    return 0;
    if (bits_per_index == 4 && (group_size & 1))    return 0;

    uint64_t off_cb  = rd_u64(blob + 80);
    uint64_t off_is  = rd_u64(blob + 88);
    uint64_t off_rot = rd_u64(blob + 96);
    uint64_t off_sc  = rd_u64(blob + 104);
    uint64_t off_ix  = rd_u64(blob + 112);
    uint64_t total   = rd_u64(blob + 120);
    if (total != blob_size) return 0;

    uint32_t rotation_family = rd_u32(blob + 128);
    uint32_t has_input_scale = rd_u32(blob + 132);

    out->dim0            = (uint32_t)dim0;
    out->dim1            = (uint32_t)dim1;
    out->group_size      = group_size;
    out->num_groups      = num_groups;
    out->bits            = bits_per_index;
    out->rotation_family = rotation_family;
    out->per_group_bytes = (group_size * bits_per_index) / 8;
    out->has_input_scale = has_input_scale;

    out->codebook    = reinterpret_cast<const float*>(blob + off_cb);
    out->input_scale = has_input_scale
                       ? reinterpret_cast<const __fp16*>(blob + off_is)
                       : nullptr;
    out->input_scale_recip = nullptr;
    out->input_scale_recip_storage.clear();
    if (out->input_scale != nullptr) {
        out->input_scale_recip_storage.resize((size_t)dim1);
        for (size_t i = 0; i < out->input_scale_recip_storage.size(); ++i) {
            out->input_scale_recip_storage[i] = (__fp16)(1.0f / (float)out->input_scale[i]);
        }
        out->input_scale_recip = out->input_scale_recip_storage.data();
    }
    out->scales      = reinterpret_cast<const __fp16*>(blob + off_sc);
    out->packed      = reinterpret_cast<const uint8_t*>(blob + off_ix);

    if (rotation_family == kRotationHadamard) {
        if (group_size > 256)                       return 0;
        if ((group_size & (group_size - 1)) != 0)   return 0;
        out->left_signs  = reinterpret_cast<const int8_t*>(blob + off_rot);
        out->right_signs = reinterpret_cast<const int8_t*>(blob + off_rot + group_size);
        out->permutation = reinterpret_cast<const uint32_t*>(blob + off_rot + 2u * group_size);
        for (uint32_t i = 0; i < group_size; ++i) {
            uint32_t p = out->permutation[i];
            if (p >= group_size) return 0;
            out->inv_permutation[p] = i;
        }
        out->orth_R = nullptr;
    } else if (rotation_family == kRotationOrthFull) {
        if (num_groups != 1)              return 0;
        if (group_size != dim1)           return 0;
        out->orth_R = reinterpret_cast<const __fp16*>(blob + off_rot);
        out->left_signs = nullptr;
        out->right_signs = nullptr;
        out->permutation = nullptr;
    } else {
        return 0;
    }

    if (scales_bytes  != (uint64_t)dim0 * num_groups * sizeof(__fp16))            return 0;
    if (indices_bytes != (uint64_t)dim0 * num_groups * out->per_group_bytes)      return 0;

    return 1;
}


void cactus_tqn_dequant_row(const CactusTQN* tqn, uint32_t row_id, __fp16* out) {
    if (tqn->rotation_family == kRotationOrthFull) {
        dequant_row_orth(tqn, row_id, out);
    } else {
        const uint32_t gs = tqn->group_size;
        for (uint32_t g = 0; g < tqn->num_groups; ++g) {
            dequant_group_hadamard(tqn, row_id, g, out + g * gs);
        }
    }
}


void cactus_tqn_dequant_layer(const CactusTQN* tqn, __fp16* out) {
    const size_t K = tqn->dim1;
    for (uint32_t row = 0; row < tqn->dim0; ++row) {
        cactus_tqn_dequant_row(tqn, row, out + (size_t)row * K);
    }
}


// Companion bring-up helper: pre-dequant TQ2 weights to fp16 (lifts the existing
// embeddings-only restriction). Same per-group math as cactus_tq2_dequant_row.
void cactus_tq2_dequant_layer(const CactusTQ2* tq2, __fp16* out) {
    const size_t K = (size_t)tq2->num_groups * tq2->group_size;
    for (uint32_t row = 0; row < tq2->vocab; ++row) {
        cactus_tq2_dequant_row(tq2, row, out + row * K);
    }
}


// ── Fused TQ4 gemv (decode hot path) ─────────────────────────────────────────
//
// Math:
//   W[n, gs*g + k] = (FWHT(P^T x))[k] * row_norm[n, g] / input_scale[gs*g + k]
// where x = right_signs ⊙ codebook[indices[n, g]] (after inv_perm) and
// FWHT is the normalized Walsh-Hadamard transform; left_signs is applied after
// FWHT before the row-norm scale.
//
// y[n] = sum_k W[n, k] * x_act[k]
//      = sum_g row_norm[n, g] * sum_k (FWHT(...)[k] * (x_act[gs*g+k] / input_scale[gs*g+k]))
//
// We pre-divide x_act by input_scale once (K elements) and then per row
// dequant-then-dot per group. The codebook, signs, and inv_permutation are
// loop-invariants shared across all rows.

namespace {

// Codebook symmetry: Lloyd-Max on the Beta(group_dim) coordinate distribution
// produces a point-symmetric codebook (sorted ascending, cb[i] = -cb[15-i] for
// 4-bit). We split into 8 magnitudes + sign bit to enable a 16-byte NEON LUT
// for the magnitudes; the sign comes from the high bit of the index (bit 3).
struct Tq4DecodeTables {
    __fp16 cb[16];               // Full fp16 codebook (kept for fallback)
    uint8_t inv_perm_lo[128];    // First 128 entries of inv_permutation, as bytes
};

inline void prepare_decode_tables(const CactusTQN* tqn, Tq4DecodeTables& tbl) {
    for (int i = 0; i < 16; ++i) tbl.cb[i] = (__fp16)tqn->codebook[i];
    if (tqn->group_size <= 128) {
        for (uint32_t k = 0; k < tqn->group_size; ++k) {
            tbl.inv_perm_lo[k] = (uint8_t)tqn->inv_permutation[k];
        }
    }
}

// Dequant one row's group into y_grp[gs] (fp16). Mirrors dequant_group_hadamard
// without the row_norm and input_scale steps (those are folded outside).
inline void tq4_dequant_group_pre_norm(
    const CactusTQN* tqn, const Tq4DecodeTables& tbl,
    uint32_t row_id, uint32_t group_idx, __fp16* y_grp)
{
    const uint32_t gs = tqn->group_size;
    const uint8_t* packed_row =
        tqn->packed + ((size_t)row_id * tqn->num_groups + group_idx) * tqn->per_group_bytes;

    // Step 1: 4-bit unpack + codebook lookup into tmp[gs] (LSB-first nibble pair per byte).
    __fp16 tmp[256];
    for (uint32_t k = 0; k < gs; k += 2) {
        uint8_t b = packed_row[k >> 1];
        tmp[k    ] = tbl.cb[b & 0xF];
        tmp[k + 1] = tbl.cb[(b >> 4) & 0xF];
    }

    // Step 2: inverse permutation into y_grp[gs].
    if (gs == 128) {
        for (uint32_t k = 0; k < 128; ++k) y_grp[k] = tmp[tbl.inv_perm_lo[k]];
    } else {
        for (uint32_t k = 0; k < gs; ++k) y_grp[k] = tmp[tqn->inv_permutation[k]];
    }

    // Step 3: right_signs.
    for (uint32_t k = 0; k < gs; k += 8) {
        vst1q_f16(y_grp + k, vmulq_f16(vld1q_f16(y_grp + k),
                                        signs_to_fp16_mul(tqn->right_signs + k)));
    }

    // Step 4: FWHT.
    if (gs == 128) fwht128_f16(y_grp);
    else           fwht_scalar_f16(y_grp, gs);

    // Step 5: left_signs.
    for (uint32_t k = 0; k < gs; k += 8) {
        vst1q_f16(y_grp + k, vmulq_f16(vld1q_f16(y_grp + k),
                                        signs_to_fp16_mul(tqn->left_signs + k)));
    }
    // Note: row_norm and 1/input_scale are folded outside this function.
}

}  // namespace


void cactus_gemv_tq4_hadamard_f16(const CactusTQN* tqn, const __fp16* x,
                                   __fp16* y, size_t N) {
    cactus_matmul_tqn_f16(tqn, x, y, 1, tqn ? tqn->dim1 : 0, N);
    return;

    const size_t K = tqn->dim1;
    const uint32_t G = tqn->num_groups;
    const uint32_t gs = tqn->group_size;

    Tq4DecodeTables tbl;
    prepare_decode_tables(tqn, tbl);

    // Pre-divide activation by input_scale (per-channel, fp16 / fp16). This is
    // K fp16 ops, paid once and amortized across all N output rows.
    std::vector<__fp16> x_scaled(K);
    if (tqn->has_input_scale) {
        for (size_t k = 0; k < K; k += 8) {
            float16x8_t xv = vld1q_f16(x + k);
            float16x8_t sv = vld1q_f16(tqn->input_scale + k);
            vst1q_f16(x_scaled.data() + k, vdivq_f16(xv, sv));
        }
    } else {
        std::memcpy(x_scaled.data(), x, K * sizeof(__fp16));
    }

    // Per-row gemv: dequant one group at a time, accumulate dot with x_scaled.
    for (size_t n = 0; n < N; ++n) {
        const __fp16* row_norms = tqn->scales + n * G;  // [G] fp16
        float acc_total = 0.f;

        for (uint32_t g = 0; g < G; ++g) {
            __fp16 y_grp[256];
            tq4_dequant_group_pre_norm(tqn, tbl, (uint32_t)n, g, y_grp);

            // Dot y_grp[gs] · x_scaled[g*gs..(g+1)*gs] in fp32 accumulator.
            const __fp16* xg = x_scaled.data() + (size_t)g * gs;
            float32x4_t accv = vdupq_n_f32(0.f);
            for (uint32_t k = 0; k < gs; k += 8) {
                float16x8_t yv = vld1q_f16(y_grp + k);
                float16x8_t xv = vld1q_f16(xg + k);
                float16x8_t pv = vmulq_f16(yv, xv);
                accv = vaddq_f32(accv, vcvt_f32_f16(vget_low_f16(pv)));
                accv = vaddq_f32(accv, vcvt_f32_f16(vget_high_f16(pv)));
            }
            float dot_g = vaddvq_f32(accv);

            // Fold row_norm[g].
            acc_total += dot_g * (float)row_norms[g];
        }

        y[n] = (__fp16)acc_total;
    }
}

namespace {

inline float dot_f16_f32(const __fp16* a, const __fp16* b, size_t len) {
    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    size_t k = 0;
    for (; k + 8 <= len; k += 8) {
        float16x8_t av = vld1q_f16(a + k);
        float16x8_t bv = vld1q_f16(b + k);
        float16x8_t pv = vmulq_f16(av, bv);
        acc0 = vaddq_f32(acc0, vcvt_f32_f16(vget_low_f16(pv)));
        acc1 = vaddq_f32(acc1, vcvt_f32_f16(vget_high_f16(pv)));
    }
    float acc = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    for (; k < len; ++k) acc += (float)a[k] * (float)b[k];
    return acc;
}

inline __fp16 hsum_f16x8_local(float16x8_t v) {
    float16x4_t lo = vget_low_f16(v);
    float16x4_t hi = vget_high_f16(v);
    float16x4_t sum4 = vadd_f16(lo, hi);
    float16x4_t sum2 = vadd_f16(sum4, vext_f16(sum4, sum4, 2));
    float16x4_t sum1 = vadd_f16(sum2, vext_f16(sum2, sum2, 1));
    return vget_lane_f16(sum1, 0);
}

void tqn_dequant_group_pre_norm(const CactusTQN* tqn, uint32_t row_id,
                                uint32_t group_idx, __fp16* y);
void tq2_dequant_group_pre_norm(const CactusTQ2* tq2, uint32_t row_id,
                                uint32_t group_idx, __fp16* y);

inline void accumulate_group_dot_mblocked(const __fp16* x_base, size_t K,
                                          const __fp16* group, size_t gs,
                                          size_t M, float scale, float* acc) {
    size_t m = 0;
    for (; m + 4 <= M; m += 4) {
        float32x4_t a0 = vdupq_n_f32(0.f);
        float32x4_t a1 = vdupq_n_f32(0.f);
        float32x4_t a2 = vdupq_n_f32(0.f);
        float32x4_t a3 = vdupq_n_f32(0.f);
        const __fp16* x0 = x_base + (m + 0) * K;
        const __fp16* x1 = x_base + (m + 1) * K;
        const __fp16* x2 = x_base + (m + 2) * K;
        const __fp16* x3 = x_base + (m + 3) * K;
        for (size_t k = 0; k < gs; k += 8) {
            float16x8_t gv = vld1q_f16(group + k);
            float16x8_t p0 = vmulq_f16(vld1q_f16(x0 + k), gv);
            float16x8_t p1 = vmulq_f16(vld1q_f16(x1 + k), gv);
            float16x8_t p2 = vmulq_f16(vld1q_f16(x2 + k), gv);
            float16x8_t p3 = vmulq_f16(vld1q_f16(x3 + k), gv);
            a0 = vaddq_f32(a0, vcvt_f32_f16(vget_low_f16(p0)));
            a0 = vaddq_f32(a0, vcvt_f32_f16(vget_high_f16(p0)));
            a1 = vaddq_f32(a1, vcvt_f32_f16(vget_low_f16(p1)));
            a1 = vaddq_f32(a1, vcvt_f32_f16(vget_high_f16(p1)));
            a2 = vaddq_f32(a2, vcvt_f32_f16(vget_low_f16(p2)));
            a2 = vaddq_f32(a2, vcvt_f32_f16(vget_high_f16(p2)));
            a3 = vaddq_f32(a3, vcvt_f32_f16(vget_low_f16(p3)));
            a3 = vaddq_f32(a3, vcvt_f32_f16(vget_high_f16(p3)));
        }
        acc[m + 0] += scale * vaddvq_f32(a0);
        acc[m + 1] += scale * vaddvq_f32(a1);
        acc[m + 2] += scale * vaddvq_f32(a2);
        acc[m + 3] += scale * vaddvq_f32(a3);
    }
    for (; m < M; ++m) {
        acc[m] += scale * dot_f16_f32(x_base + m * K, group, gs);
    }
}

void matmul_f16_ntile4(const __fp16* A, const __fp16* B_tile, __fp16* C,
                       size_t M, size_t K, size_t N, size_t n_start, size_t actual_n) {
    constexpr size_t TILE_M = 4;
    const size_t K16 = (K / 16) * 16;
    const size_t K8 = (K / 8) * 8;

    for (size_t m_start = 0; m_start < M; m_start += TILE_M) {
        const size_t actual_m = std::min(TILE_M, M - m_start);
        float16x8_t acc[TILE_M][4];
        for (size_t mi = 0; mi < TILE_M; ++mi)
            for (size_t ni = 0; ni < 4; ++ni)
                acc[mi][ni] = vdupq_n_f16(0);

        for (size_t k = 0; k < K16; k += 16) {
            float16x8_t a_lo[TILE_M], a_hi[TILE_M];
            for (size_t mi = 0; mi < TILE_M; ++mi) {
                if (mi < actual_m) {
                    const __fp16* ap = A + (m_start + mi) * K + k;
                    a_lo[mi] = vld1q_f16(ap);
                    a_hi[mi] = vld1q_f16(ap + 8);
                } else {
                    a_lo[mi] = vdupq_n_f16(0);
                    a_hi[mi] = vdupq_n_f16(0);
                }
            }
            for (size_t ni = 0; ni < actual_n; ++ni) {
                const __fp16* bp = B_tile + ni * K + k;
                float16x8_t b_lo = vld1q_f16(bp);
                float16x8_t b_hi = vld1q_f16(bp + 8);
                for (size_t mi = 0; mi < actual_m; ++mi) {
                    acc[mi][ni] = vfmaq_f16(acc[mi][ni], a_lo[mi], b_lo);
                    acc[mi][ni] = vfmaq_f16(acc[mi][ni], a_hi[mi], b_hi);
                }
            }
        }
        for (size_t k = K16; k < K8; k += 8) {
            float16x8_t a_v[TILE_M];
            for (size_t mi = 0; mi < TILE_M; ++mi) {
                a_v[mi] = mi < actual_m ? vld1q_f16(A + (m_start + mi) * K + k)
                                        : vdupq_n_f16(0);
            }
            for (size_t ni = 0; ni < actual_n; ++ni) {
                float16x8_t b_v = vld1q_f16(B_tile + ni * K + k);
                for (size_t mi = 0; mi < actual_m; ++mi) {
                    acc[mi][ni] = vfmaq_f16(acc[mi][ni], a_v[mi], b_v);
                }
            }
        }
        for (size_t k = K8; k < K; ++k) {
            for (size_t mi = 0; mi < actual_m; ++mi) {
                __fp16 av = A[(m_start + mi) * K + k];
                for (size_t ni = 0; ni < actual_n; ++ni) {
                    acc[mi][ni] = vsetq_lane_f16(
                        vgetq_lane_f16(acc[mi][ni], 0) + av * B_tile[ni * K + k],
                        acc[mi][ni], 0);
                }
            }
        }
        for (size_t mi = 0; mi < actual_m; ++mi) {
            for (size_t ni = 0; ni < actual_n; ++ni) {
                C[(m_start + mi) * N + n_start + ni] = hsum_f16x8_local(acc[mi][ni]);
            }
        }
    }
}

void matmul_f16_ntile6(const __fp16* A, const __fp16* B_tile, __fp16* C,
                       size_t M, size_t K, size_t N, size_t n_start, size_t actual_n) {
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 14;
    const size_t K16 = (K / 16) * 16;
    const size_t K8 = (K / 8) * 8;

    for (size_t m_start = 0; m_start < M; m_start += TILE_M) {
        const size_t actual_m = std::min(TILE_M, M - m_start);
        float16x8_t acc[TILE_M][TILE_N];
        for (size_t mi = 0; mi < TILE_M; ++mi)
            for (size_t ni = 0; ni < TILE_N; ++ni)
                acc[mi][ni] = vdupq_n_f16(0);

        for (size_t k = 0; k < K16; k += 16) {
            float16x8_t a_lo[TILE_M], a_hi[TILE_M];
            for (size_t mi = 0; mi < TILE_M; ++mi) {
                if (mi < actual_m) {
                    const __fp16* ap = A + (m_start + mi) * K + k;
                    a_lo[mi] = vld1q_f16(ap);
                    a_hi[mi] = vld1q_f16(ap + 8);
                } else {
                    a_lo[mi] = vdupq_n_f16(0);
                    a_hi[mi] = vdupq_n_f16(0);
                }
            }
            for (size_t ni = 0; ni < actual_n; ++ni) {
                const __fp16* bp = B_tile + ni * K + k;
                float16x8_t b_lo = vld1q_f16(bp);
                float16x8_t b_hi = vld1q_f16(bp + 8);
                for (size_t mi = 0; mi < actual_m; ++mi) {
                    acc[mi][ni] = vfmaq_f16(acc[mi][ni], a_lo[mi], b_lo);
                    acc[mi][ni] = vfmaq_f16(acc[mi][ni], a_hi[mi], b_hi);
                }
            }
        }
        for (size_t k = K16; k < K8; k += 8) {
            float16x8_t a_v[TILE_M];
            for (size_t mi = 0; mi < TILE_M; ++mi) {
                a_v[mi] = mi < actual_m ? vld1q_f16(A + (m_start + mi) * K + k)
                                        : vdupq_n_f16(0);
            }
            for (size_t ni = 0; ni < actual_n; ++ni) {
                float16x8_t b_v = vld1q_f16(B_tile + ni * K + k);
                for (size_t mi = 0; mi < actual_m; ++mi) {
                    acc[mi][ni] = vfmaq_f16(acc[mi][ni], a_v[mi], b_v);
                }
            }
        }
        for (size_t k = K8; k < K; ++k) {
            for (size_t mi = 0; mi < actual_m; ++mi) {
                __fp16 av = A[(m_start + mi) * K + k];
                for (size_t ni = 0; ni < actual_n; ++ni) {
                    acc[mi][ni] = vsetq_lane_f16(
                        vgetq_lane_f16(acc[mi][ni], 0) + av * B_tile[ni * K + k],
                        acc[mi][ni], 0);
                }
            }
        }
        for (size_t mi = 0; mi < actual_m; ++mi) {
            for (size_t ni = 0; ni < actual_n; ++ni) {
                C[(m_start + mi) * N + n_start + ni] = hsum_f16x8_local(acc[mi][ni]);
            }
        }
    }
}

void matmul_f16_ntile12_segment_accum(const __fp16* A, size_t a_stride,
                                      const __fp16* B_tile, __fp16* C,
                                      size_t M, size_t Kseg, size_t N,
                                      size_t n_start, size_t actual_n) {
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 24;
    const size_t K16 = (Kseg / 16) * 16;
    const size_t K8 = (Kseg / 8) * 8;

    for (size_t m_start = 0; m_start < M; m_start += TILE_M) {
        const size_t actual_m = std::min(TILE_M, M - m_start);
        float16x8_t acc[TILE_M][TILE_N];
        for (size_t mi = 0; mi < TILE_M; ++mi)
            for (size_t ni = 0; ni < TILE_N; ++ni)
                acc[mi][ni] = vdupq_n_f16(0);

        for (size_t k = 0; k < K16; k += 16) {
            float16x8_t a_lo[TILE_M], a_hi[TILE_M];
            for (size_t mi = 0; mi < TILE_M; ++mi) {
                if (mi < actual_m) {
                    const __fp16* ap = A + (m_start + mi) * a_stride + k;
                    a_lo[mi] = vld1q_f16(ap);
                    a_hi[mi] = vld1q_f16(ap + 8);
                } else {
                    a_lo[mi] = vdupq_n_f16(0);
                    a_hi[mi] = vdupq_n_f16(0);
                }
            }
            for (size_t ni = 0; ni < actual_n; ++ni) {
                const __fp16* bp = B_tile + ni * Kseg + k;
                float16x8_t b_lo = vld1q_f16(bp);
                float16x8_t b_hi = vld1q_f16(bp + 8);
                for (size_t mi = 0; mi < actual_m; ++mi) {
                    acc[mi][ni] = vfmaq_f16(acc[mi][ni], a_lo[mi], b_lo);
                    acc[mi][ni] = vfmaq_f16(acc[mi][ni], a_hi[mi], b_hi);
                }
            }
        }
        for (size_t k = K16; k < K8; k += 8) {
            float16x8_t a_v[TILE_M];
            for (size_t mi = 0; mi < TILE_M; ++mi) {
                a_v[mi] = mi < actual_m ? vld1q_f16(A + (m_start + mi) * a_stride + k)
                                        : vdupq_n_f16(0);
            }
            for (size_t ni = 0; ni < actual_n; ++ni) {
                float16x8_t b_v = vld1q_f16(B_tile + ni * Kseg + k);
                for (size_t mi = 0; mi < actual_m; ++mi) {
                    acc[mi][ni] = vfmaq_f16(acc[mi][ni], a_v[mi], b_v);
                }
            }
        }
        for (size_t k = K8; k < Kseg; ++k) {
            for (size_t mi = 0; mi < actual_m; ++mi) {
                __fp16 av = A[(m_start + mi) * a_stride + k];
                for (size_t ni = 0; ni < actual_n; ++ni) {
                    acc[mi][ni] = vsetq_lane_f16(
                        vgetq_lane_f16(acc[mi][ni], 0) + av * B_tile[ni * Kseg + k],
                        acc[mi][ni], 0);
                }
            }
        }
        for (size_t mi = 0; mi < actual_m; ++mi) {
            for (size_t ni = 0; ni < actual_n; ++ni) {
                __fp16* dst = C + (m_start + mi) * N + n_start + ni;
                *dst = (__fp16)((float)*dst + (float)hsum_f16x8_local(acc[mi][ni]));
            }
        }
    }
}

void matmul_f16_ntile8_m2(const __fp16* A, const __fp16* B_tile, __fp16* C,
                          size_t M, size_t K, size_t N, size_t n_start, size_t actual_n) {
    constexpr size_t TILE_M = 2;
    constexpr size_t TILE_N = 16;
    const size_t K16 = (K / 16) * 16;
    const size_t K8 = (K / 8) * 8;

    for (size_t m_start = 0; m_start < M; m_start += TILE_M) {
        const size_t actual_m = std::min(TILE_M, M - m_start);
        float16x8_t acc[TILE_M][TILE_N];
        for (size_t mi = 0; mi < TILE_M; ++mi)
            for (size_t ni = 0; ni < TILE_N; ++ni)
                acc[mi][ni] = vdupq_n_f16(0);

        for (size_t k = 0; k < K16; k += 16) {
            float16x8_t a_lo[TILE_M], a_hi[TILE_M];
            for (size_t mi = 0; mi < TILE_M; ++mi) {
                if (mi < actual_m) {
                    const __fp16* ap = A + (m_start + mi) * K + k;
                    a_lo[mi] = vld1q_f16(ap);
                    a_hi[mi] = vld1q_f16(ap + 8);
                } else {
                    a_lo[mi] = vdupq_n_f16(0);
                    a_hi[mi] = vdupq_n_f16(0);
                }
            }
            for (size_t ni = 0; ni < actual_n; ++ni) {
                const __fp16* bp = B_tile + ni * K + k;
                float16x8_t b_lo = vld1q_f16(bp);
                float16x8_t b_hi = vld1q_f16(bp + 8);
                for (size_t mi = 0; mi < actual_m; ++mi) {
                    acc[mi][ni] = vfmaq_f16(acc[mi][ni], a_lo[mi], b_lo);
                    acc[mi][ni] = vfmaq_f16(acc[mi][ni], a_hi[mi], b_hi);
                }
            }
        }
        for (size_t k = K16; k < K8; k += 8) {
            float16x8_t a_v[TILE_M];
            for (size_t mi = 0; mi < TILE_M; ++mi) {
                a_v[mi] = mi < actual_m ? vld1q_f16(A + (m_start + mi) * K + k)
                                        : vdupq_n_f16(0);
            }
            for (size_t ni = 0; ni < actual_n; ++ni) {
                float16x8_t b_v = vld1q_f16(B_tile + ni * K + k);
                for (size_t mi = 0; mi < actual_m; ++mi) {
                    acc[mi][ni] = vfmaq_f16(acc[mi][ni], a_v[mi], b_v);
                }
            }
        }
        for (size_t k = K8; k < K; ++k) {
            for (size_t mi = 0; mi < actual_m; ++mi) {
                __fp16 av = A[(m_start + mi) * K + k];
                for (size_t ni = 0; ni < actual_n; ++ni) {
                    acc[mi][ni] = vsetq_lane_f16(
                        vgetq_lane_f16(acc[mi][ni], 0) + av * B_tile[ni * K + k],
                        acc[mi][ni], 0);
                }
            }
        }
        for (size_t mi = 0; mi < actual_m; ++mi) {
            for (size_t ni = 0; ni < actual_n; ++ni) {
                C[(m_start + mi) * N + n_start + ni] = hsum_f16x8_local(acc[mi][ni]);
            }
        }
    }
}

void tq2_dequant_row_scaled_no_input_scale(const CactusTQ2* tq2, uint32_t row, __fp16* out) {
    const uint32_t G = tq2->num_groups;
    const uint32_t gs = tq2->group_size;
    for (uint32_t g = 0; g < G; ++g) {
        __fp16* dst = out + (size_t)g * gs;
        tq2_dequant_group_pre_norm(tq2, row, g, dst);
        float16x8_t rn = vdupq_n_f16(tq2->scales[(size_t)row * G + g]);
        for (uint32_t k = 0; k < gs; k += 8) {
            vst1q_f16(dst + k, vmulq_f16(vld1q_f16(dst + k), rn));
        }
    }
}

void tqn_dequant_row_scaled_no_input_scale(const CactusTQN* tqn, uint32_t row, __fp16* out) {
    const uint32_t G = tqn->num_groups;
    const uint32_t gs = tqn->group_size;
    for (uint32_t g = 0; g < G; ++g) {
        __fp16* dst = out + (size_t)g * gs;
        tqn_dequant_group_pre_norm(tqn, row, g, dst);
        float16x8_t rn = vdupq_n_f16(tqn->scales[(size_t)row * G + g]);
        for (uint32_t k = 0; k < gs; k += 8) {
            vst1q_f16(dst + k, vmulq_f16(vld1q_f16(dst + k), rn));
        }
    }
}

template<typename WorkFunc>
void tq_parallel_ranges(size_t total_work, size_t work_per_thread, WorkFunc work_func) {
    if (total_work == 0) return;
    auto& pool = CactusThreading::get_thread_pool();
    size_t num_threads = std::min(pool.num_workers(),
                                  (total_work + work_per_thread - 1) / work_per_thread);
    num_threads = std::min(num_threads, total_work);
    if (num_threads <= 1) {
        work_func(0, total_work);
        return;
    }
    pool.enqueue_n_threads(total_work, num_threads, work_func);
    pool.wait_all();
}

#ifdef __APPLE__
template<typename DequantRow>
bool tq_prefill_blas_tiles(const __fp16* x_scaled, __fp16* C,
                           size_t M, size_t K, size_t N, size_t rows,
                           DequantRow dequant_row) {
    if (true || M < 16 || K < 256 || rows < 16) return false;

    constexpr size_t TILE_N = 64;
    thread_local std::vector<float> a_f32;
    thread_local std::vector<float> b_f32;
    thread_local std::vector<float> c_f32;
    thread_local std::vector<__fp16> row_f16;

    if (a_f32.size() < M * K) a_f32.resize(M * K);
    if (b_f32.size() < TILE_N * K) b_f32.resize(TILE_N * K);
    if (c_f32.size() < M * TILE_N) c_f32.resize(M * TILE_N);
    if (row_f16.size() < K) row_f16.resize(K);

    for (size_t i = 0; i < M * K; ++i) a_f32[i] = (float)x_scaled[i];

    for (size_t n_start = 0; n_start < rows; n_start += TILE_N) {
        const size_t actual_n = std::min(TILE_N, rows - n_start);
        for (size_t ni = 0; ni < actual_n; ++ni) {
            dequant_row((uint32_t)(n_start + ni), row_f16.data());
            float* dst = b_f32.data() + ni * K;
            for (size_t k = 0; k < K; ++k) dst[k] = (float)row_f16[k];
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)M, (int)actual_n, (int)K,
                    1.0f, a_f32.data(), (int)K,
                    b_f32.data(), (int)K,
                    0.0f, c_f32.data(), (int)actual_n);

        for (size_t m = 0; m < M; ++m) {
            for (size_t ni = 0; ni < actual_n; ++ni) {
                float v = c_f32[m * actual_n + ni];
                if (v > 65504.f) v = 65504.f;
                else if (v < -65504.f) v = -65504.f;
                C[m * N + n_start + ni] = (__fp16)v;
            }
        }
    }
    return true;
}
#endif

void scale_activations_by_input_scale(const __fp16* A, const __fp16* input_scale,
                                      const __fp16* input_scale_recip,
                                      bool has_input_scale, __fp16* out,
                                      size_t M, size_t K) {
    if (!has_input_scale || input_scale == nullptr) {
        std::memcpy(out, A, M * K * sizeof(__fp16));
        return;
    }

    if (M < 4) {
        for (size_t m = 0; m < M; ++m) {
            const __fp16* src = A + m * K;
            __fp16* dst = out + m * K;
            size_t k = 0;
            for (; k + 8 <= K; k += 8) {
                float16x8_t sv = input_scale_recip != nullptr
                    ? vld1q_f16(input_scale_recip + k)
                    : vdivq_f16(vdupq_n_f16(1), vld1q_f16(input_scale + k));
                vst1q_f16(dst + k, vmulq_f16(vld1q_f16(src + k), sv));
            }
            for (; k < K; ++k) {
                float sv = input_scale_recip != nullptr
                    ? (float)input_scale_recip[k]
                    : 1.0f / (float)input_scale[k];
                dst[k] = (__fp16)((float)src[k] * sv);
            }
        }
        return;
    }

    CactusThreading::parallel_for(M, CactusThreading::ParallelConfig{2, 1},
        [A, input_scale, input_scale_recip, out, K](size_t m_start, size_t m_end) {
            for (size_t m = m_start; m < m_end; ++m) {
                const __fp16* src = A + m * K;
                __fp16* dst = out + m * K;
                size_t k = 0;
                for (; k + 8 <= K; k += 8) {
                    float16x8_t sv = input_scale_recip != nullptr
                        ? vld1q_f16(input_scale_recip + k)
                        : vdivq_f16(vdupq_n_f16(1), vld1q_f16(input_scale + k));
                    vst1q_f16(dst + k, vmulq_f16(vld1q_f16(src + k), sv));
                }
                for (; k < K; ++k) {
                    float sv = input_scale_recip != nullptr
                        ? (float)input_scale_recip[k]
                        : 1.0f / (float)input_scale[k];
                    dst[k] = (__fp16)((float)src[k] * sv);
                }
            }
        });
}

void tqn_dequant_group_pre_norm(const CactusTQN* tqn, uint32_t row_id,
                                uint32_t group_idx, __fp16* y) {
    const uint32_t gs = tqn->group_size;
    const uint32_t cb_size = 1u << tqn->bits;

    __fp16 cb[16];
    for (uint32_t i = 0; i < cb_size; ++i) cb[i] = (__fp16)tqn->codebook[i];

    __fp16 tmp[256];
    const uint8_t* packed_row =
        tqn->packed + ((size_t)row_id * tqn->num_groups + group_idx) * tqn->per_group_bytes;
    codebook_lookup_group(tqn, packed_row, cb, tmp);

    for (uint32_t k = 0; k < gs; ++k) y[k] = tmp[tqn->inv_permutation[k]];

    for (uint32_t k = 0; k < gs; k += 8) {
        vst1q_f16(y + k, vmulq_f16(vld1q_f16(y + k), signs_to_fp16_mul(tqn->right_signs + k)));
    }

    if (gs == 128) fwht128_f16(y);
    else           fwht_scalar_f16(y, gs);

    for (uint32_t k = 0; k < gs; k += 8) {
        vst1q_f16(y + k, vmulq_f16(vld1q_f16(y + k), signs_to_fp16_mul(tqn->left_signs + k)));
    }
}

void tq2_dequant_group_pre_norm(const CactusTQ2* tq2, uint32_t row_id,
                                uint32_t group_idx, __fp16* y) {
    const uint32_t gs = tq2->group_size;
    const __fp16 cb[4] = {
        (__fp16)tq2->codebook[0], (__fp16)tq2->codebook[1],
        (__fp16)tq2->codebook[2], (__fp16)tq2->codebook[3],
    };

    __fp16 tmp[256];
    const uint8_t* packed_row =
        tq2->packed + ((size_t)row_id * tq2->num_groups + group_idx) * tq2->per_group_bytes;
    for (uint32_t k = 0; k < gs; k += 4) {
        uint8_t byte = packed_row[k >> 2];
        tmp[k    ] = cb[(byte     ) & 0x3];
        tmp[k + 1] = cb[(byte >> 2) & 0x3];
        tmp[k + 2] = cb[(byte >> 4) & 0x3];
        tmp[k + 3] = cb[(byte >> 6) & 0x3];
    }

    for (uint32_t k = 0; k < gs; ++k) y[k] = tmp[tq2->inv_permutation[k]];

    for (uint32_t k = 0; k < gs; k += 8) {
        vst1q_f16(y + k, vmulq_f16(vld1q_f16(y + k), signs_to_fp16_mul(tq2->right_signs + k)));
    }

    if (gs == 128) fwht128_f16(y);
    else           fwht_scalar_f16(y, gs);

    for (uint32_t k = 0; k < gs; k += 8) {
        vst1q_f16(y + k, vmulq_f16(vld1q_f16(y + k), signs_to_fp16_mul(tq2->left_signs + k)));
    }
}

void tqn_transform_hadamard_activation(const CactusTQN* tqn, const __fp16* x_group,
                                       __fp16* code_basis) {
    const uint32_t gs = tqn->group_size;
    __fp16 tmp[256];

    for (uint32_t k = 0; k < gs; k += 8) {
        vst1q_f16(tmp + k, vmulq_f16(vld1q_f16(x_group + k),
                                      signs_to_fp16_mul(tqn->left_signs + k)));
    }
    if (gs == 128) fwht128_f16(tmp);
    else           fwht_scalar_f16(tmp, gs);
    for (uint32_t k = 0; k < gs; k += 8) {
        vst1q_f16(tmp + k, vmulq_f16(vld1q_f16(tmp + k),
                                      signs_to_fp16_mul(tqn->right_signs + k)));
    }
    for (uint32_t j = 0; j < gs; ++j) {
        code_basis[j] = tmp[tqn->permutation[j]];
    }
}

void tq2_transform_hadamard_activation(const CactusTQ2* tq2, const __fp16* x_group,
                                       __fp16* code_basis) {
    const uint32_t gs = tq2->group_size;
    __fp16 tmp[256];

    for (uint32_t k = 0; k < gs; k += 8) {
        vst1q_f16(tmp + k, vmulq_f16(vld1q_f16(x_group + k),
                                      signs_to_fp16_mul(tq2->left_signs + k)));
    }
    if (gs == 128) fwht128_f16(tmp);
    else           fwht_scalar_f16(tmp, gs);
    for (uint32_t k = 0; k < gs; k += 8) {
        vst1q_f16(tmp + k, vmulq_f16(vld1q_f16(tmp + k),
                                      signs_to_fp16_mul(tq2->right_signs + k)));
    }
    for (uint32_t j = 0; j < gs; ++j) {
        code_basis[j] = tmp[tq2->permutation[j]];
    }
}

float tq2_codebook_dot_group(const CactusTQ2* tq2, uint32_t row_id,
                             uint32_t group_idx, const __fp16* code_basis) {
    const uint32_t gs = tq2->group_size;
    const __fp16 cb[4] = {
        (__fp16)tq2->codebook[0], (__fp16)tq2->codebook[1],
        (__fp16)tq2->codebook[2], (__fp16)tq2->codebook[3],
    };
    const uint8_t* packed_row =
        tq2->packed + ((size_t)row_id * tq2->num_groups + group_idx) * tq2->per_group_bytes;

    float16x8_t acc_h = vdupq_n_f16(0);
    for (uint32_t k = 0; k < gs; k += 8) {
        uint8_t byte = packed_row[k >> 2];
        uint8_t byte1 = packed_row[(k >> 2) + 1];
        __fp16 vals[8] = {
            cb[(byte     ) & 0x3], cb[(byte >> 2) & 0x3],
            cb[(byte >> 4) & 0x3], cb[(byte >> 6) & 0x3],
            cb[(byte1     ) & 0x3], cb[(byte1 >> 2) & 0x3],
            cb[(byte1 >> 4) & 0x3], cb[(byte1 >> 6) & 0x3],
        };
        float16x8_t cv = vld1q_f16(vals);
        float16x8_t zv = vld1q_f16(code_basis + k);
        acc_h = vfmaq_f16(acc_h, cv, zv);
    }
    return (float)hsum_f16x8_local(acc_h);
}

float tqn_codebook_dot_group(const CactusTQN* tqn, uint32_t row_id,
                             uint32_t group_idx, const __fp16* code_basis) {
    const uint32_t gs = tqn->group_size;
    __fp16 cb[16];
    const uint32_t cb_size = 1u << tqn->bits;
    for (uint32_t i = 0; i < cb_size; ++i) cb[i] = (__fp16)tqn->codebook[i];

    const uint8_t* packed_row =
        tqn->packed + ((size_t)row_id * tqn->num_groups + group_idx) * tqn->per_group_bytes;
    float16x8_t acc_h = vdupq_n_f16(0);

    switch (tqn->bits) {
        case 1:
            for (uint32_t k = 0; k < gs; k += 8) {
                uint8_t b = packed_row[k >> 3];
                __fp16 vals[8] = {
                    cb[(b     ) & 1], cb[(b >> 1) & 1],
                    cb[(b >> 2) & 1], cb[(b >> 3) & 1],
                    cb[(b >> 4) & 1], cb[(b >> 5) & 1],
                    cb[(b >> 6) & 1], cb[(b >> 7) & 1],
                };
                acc_h = vfmaq_f16(acc_h, vld1q_f16(vals), vld1q_f16(code_basis + k));
            }
            break;
        case 3:
            for (uint32_t k = 0; k < gs; k += 8) {
                uint8_t idx[8];
                unpack_8x3bit_le(packed_row + (k >> 3) * 3, idx);
                __fp16 vals[8] = {
                    cb[idx[0]], cb[idx[1]], cb[idx[2]], cb[idx[3]],
                    cb[idx[4]], cb[idx[5]], cb[idx[6]], cb[idx[7]],
                };
                acc_h = vfmaq_f16(acc_h, vld1q_f16(vals), vld1q_f16(code_basis + k));
            }
            break;
        default:
        {
            uint8x16x2_t cb_bytes;
            const uint8_t* cb_u8 = reinterpret_cast<const uint8_t*>(cb);
            cb_bytes.val[0] = vld1q_u8(cb_u8);
            cb_bytes.val[1] = vld1q_u8(cb_u8 + 16);
            for (uint32_t k = 0; k < gs; k += 16) {
                uint8x8_t packed = vld1_u8(packed_row + (k >> 1));
                uint8x8_t lo = vand_u8(packed, vdup_n_u8(0x0F));
                uint8x8_t hi = vshr_n_u8(packed, 4);

                float16x8_t cv0 = lookup_tq4_codebook8(vzip1_u8(lo, hi), cb_bytes);
                acc_h = vfmaq_f16(acc_h, cv0, vld1q_f16(code_basis + k));

                float16x8_t cv1 = lookup_tq4_codebook8(vzip2_u8(lo, hi), cb_bytes);
                acc_h = vfmaq_f16(acc_h, cv1, vld1q_f16(code_basis + k + 8));
            }
            break;
        }
    }

    return (float)hsum_f16x8_local(acc_h);
}

void tqn_matmul_tq4_codebasis_tiled(const CactusTQN* tqn, const __fp16* code_basis,
                                    __fp16* C, size_t M, size_t K, size_t N, size_t rows) {
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 4;
    const size_t m_blocks = (M + TILE_M - 1) / TILE_M;
    const size_t n_blocks = (rows + TILE_N - 1) / TILE_N;
    const size_t total_tiles = m_blocks * n_blocks;

    __fp16 cb[16];
    for (int i = 0; i < 16; ++i) cb[i] = (__fp16)tqn->codebook[i];

    tq_parallel_ranges(total_tiles, 8,
        [=](size_t tile_start, size_t tile_end) {
            for (size_t tile = tile_start; tile < tile_end; ++tile) {
                const size_t mb = tile / n_blocks;
                const size_t nb = tile - mb * n_blocks;
                const size_t m_start = mb * TILE_M;
                const size_t n_start = nb * TILE_N;
                const size_t actual_m = std::min(TILE_M, M - m_start);
                const size_t actual_n = std::min(TILE_N, rows - n_start);

                uint8x16x2_t cb_bytes;
                const uint8_t* cb_u8 = reinterpret_cast<const uint8_t*>(cb);
                cb_bytes.val[0] = vld1q_u8(cb_u8);
                cb_bytes.val[1] = vld1q_u8(cb_u8 + 16);
                float acc[TILE_M][TILE_N] = {};
                for (uint32_t g = 0; g < tqn->num_groups; ++g) {
                    const size_t g_base = (size_t)g * tqn->group_size;
                    const __fp16* z[TILE_M] = {};
                    for (size_t mi = 0; mi < actual_m; ++mi) {
                        z[mi] = code_basis + (m_start + mi) * K + g_base;
                    }
                    const uint8_t* packed[TILE_N] = {};
                    float rn[TILE_N] = {};
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        const size_t row = n_start + ni;
                        packed[ni] = tqn->packed + ((row * tqn->num_groups + g) * tqn->per_group_bytes);
                        rn[ni] = (float)tqn->scales[row * tqn->num_groups + g];
                    }

                    float32x4_t acc_lo[TILE_M][TILE_N];
                    float32x4_t acc_hi[TILE_M][TILE_N];
                    for (size_t mi = 0; mi < TILE_M; ++mi) {
                        for (size_t ni = 0; ni < TILE_N; ++ni) {
                            acc_lo[mi][ni] = vdupq_n_f32(0.f);
                            acc_hi[mi][ni] = vdupq_n_f32(0.f);
                        }
                    }

                    for (uint32_t k = 0; k < tqn->group_size; k += 16) {
                        float16x8_t z0[TILE_M];
                        float16x8_t z1[TILE_M];
                        for (size_t mi = 0; mi < actual_m; ++mi) {
                            z0[mi] = vld1q_f16(z[mi] + k);
                            z1[mi] = vld1q_f16(z[mi] + k + 8);
                        }
                        for (size_t ni = 0; ni < actual_n; ++ni) {
                            const uint8_t* p = packed[ni] + (k >> 1);
                            uint8x8_t bytes = vld1_u8(p);
                            uint8x8_t lo = vand_u8(bytes, vdup_n_u8(0x0F));
                            uint8x8_t hi = vshr_n_u8(bytes, 4);
                            float16x8_t cv0 = lookup_tq4_codebook8(vzip1_u8(lo, hi), cb_bytes);
                            float16x8_t cv1 = lookup_tq4_codebook8(vzip2_u8(lo, hi), cb_bytes);
                            for (size_t mi = 0; mi < actual_m; ++mi) {
                                float16x8_t prod = vmulq_f16(z0[mi], cv0);
                                acc_lo[mi][ni] = vaddq_f32(acc_lo[mi][ni],
                                                           vcvt_f32_f16(vget_low_f16(prod)));
                                acc_hi[mi][ni] = vaddq_f32(acc_hi[mi][ni],
                                                           vcvt_f32_f16(vget_high_f16(prod)));
                                prod = vmulq_f16(z1[mi], cv1);
                                acc_lo[mi][ni] = vaddq_f32(acc_lo[mi][ni],
                                                           vcvt_f32_f16(vget_low_f16(prod)));
                                acc_hi[mi][ni] = vaddq_f32(acc_hi[mi][ni],
                                                           vcvt_f32_f16(vget_high_f16(prod)));
                            }
                        }
                    }
                    for (size_t mi = 0; mi < actual_m; ++mi) {
                        for (size_t ni = 0; ni < actual_n; ++ni) {
                            acc[mi][ni] += rn[ni] *
                                (vaddvq_f32(acc_lo[mi][ni]) + vaddvq_f32(acc_hi[mi][ni]));
                        }
                    }
                }

                for (size_t mi = 0; mi < actual_m; ++mi) {
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        C[(m_start + mi) * N + n_start + ni] = (__fp16)acc[mi][ni];
                    }
                }
            }
        });
}

void tqn_gemv_tq4_codebasis_tiled(const CactusTQN* tqn, const __fp16* code_basis,
                                  __fp16* C, size_t K, size_t N, size_t rows) {
    constexpr size_t TILE_N = 12;
    const size_t n_blocks = (rows + TILE_N - 1) / TILE_N;

    __fp16 cb[16];
    for (int i = 0; i < 16; ++i) cb[i] = (__fp16)tqn->codebook[i];
    uint8x16x2_t cb_bytes;
    const uint8_t* cb_u8 = reinterpret_cast<const uint8_t*>(cb);
    cb_bytes.val[0] = vld1q_u8(cb_u8);
    cb_bytes.val[1] = vld1q_u8(cb_u8 + 16);

    tq_parallel_ranges(n_blocks, 16,
        [=](size_t block_start, size_t block_end) {
            for (size_t block = block_start; block < block_end; ++block) {
                const size_t n_start = block * TILE_N;
                const size_t actual_n = std::min(TILE_N, rows - n_start);
                float acc[TILE_N] = {};

                for (uint32_t g = 0; g < tqn->num_groups; ++g) {
                    const size_t g_base = (size_t)g * tqn->group_size;
                    const __fp16* z = code_basis + g_base;

                    const uint8_t* packed[TILE_N] = {};
                    float rn[TILE_N] = {};
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        const size_t row = n_start + ni;
                        packed[ni] = tqn->packed + ((row * tqn->num_groups + g) * tqn->per_group_bytes);
                        rn[ni] = (float)tqn->scales[row * tqn->num_groups + g];
                    }

                    float16x8_t acc0[TILE_N];
                    float16x8_t acc1[TILE_N];
                    for (size_t ni = 0; ni < TILE_N; ++ni) {
                        acc0[ni] = vdupq_n_f16(0);
                        acc1[ni] = vdupq_n_f16(0);
                    }

                    for (uint32_t k = 0; k < tqn->group_size; k += 16) {
                        float16x8_t z0 = vld1q_f16(z + k);
                        float16x8_t z1 = vld1q_f16(z + k + 8);
                        for (size_t ni = 0; ni < actual_n; ++ni) {
                            uint8x8_t bytes = vld1_u8(packed[ni] + (k >> 1));
                            uint8x8_t lo = vand_u8(bytes, vdup_n_u8(0x0F));
                            uint8x8_t hi = vshr_n_u8(bytes, 4);
                            float16x8_t cv0 = lookup_tq4_codebook8(vzip1_u8(lo, hi), cb_bytes);
                            acc0[ni] = vfmaq_f16(acc0[ni], z0, cv0);

                            float16x8_t cv1 = lookup_tq4_codebook8(vzip2_u8(lo, hi), cb_bytes);
                            acc1[ni] = vfmaq_f16(acc1[ni], z1, cv1);
                        }
                    }

                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        acc[ni] += rn[ni] *
                            ((float)hsum_f16x8_local(acc0[ni]) + (float)hsum_f16x8_local(acc1[ni]));
                    }
                }

                for (size_t ni = 0; ni < actual_n; ++ni) {
                    C[n_start + ni] = (__fp16)acc[ni];
                }
            }
        });
}

void tqn_prefill_tq4_codebasis_group_gemm(const CactusTQN* tqn, const __fp16* code_basis,
                                          __fp16* C, size_t M, size_t K, size_t N,
                                          size_t rows) {
    constexpr size_t TILE_N = 12;
    const size_t n_blocks = (rows + TILE_N - 1) / TILE_N;
    const uint32_t G = tqn->num_groups;
    const uint32_t gs = tqn->group_size;

    __fp16 cb[16];
    for (int i = 0; i < 16; ++i) cb[i] = (__fp16)tqn->codebook[i];
    uint8x16x2_t cb_bytes;
    const uint8_t* cb_u8 = reinterpret_cast<const uint8_t*>(cb);
    cb_bytes.val[0] = vld1q_u8(cb_u8);
    cb_bytes.val[1] = vld1q_u8(cb_u8 + 16);

    tq_parallel_ranges(n_blocks, 4,
        [=](size_t block_start, size_t block_end) {
            thread_local std::vector<__fp16> b_tile;
            if (b_tile.size() < TILE_N * gs) b_tile.resize(TILE_N * gs);

            for (size_t block = block_start; block < block_end; ++block) {
                const size_t n_start = block * TILE_N;
                const size_t actual_n = std::min(TILE_N, rows - n_start);

                for (size_t m = 0; m < M; ++m) {
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        C[m * N + n_start + ni] = (__fp16)0;
                    }
                }

                for (uint32_t g = 0; g < G; ++g) {
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        const size_t row = n_start + ni;
                        const uint8_t* packed =
                            tqn->packed + ((row * G + g) * tqn->per_group_bytes);
                        __fp16* dst = b_tile.data() + ni * gs;
                        const __fp16 rn = tqn->scales[row * G + g];
                        float16x8_t rn_v = vdupq_n_f16(rn);
                        for (uint32_t k = 0; k < gs; k += 16) {
                            uint8x8_t bytes = vld1_u8(packed + (k >> 1));
                            uint8x8_t lo = vand_u8(bytes, vdup_n_u8(0x0F));
                            uint8x8_t hi = vshr_n_u8(bytes, 4);
                            vst1q_f16(dst + k,
                                      vmulq_f16(lookup_tq4_codebook8(vzip1_u8(lo, hi), cb_bytes), rn_v));
                            vst1q_f16(dst + k + 8,
                                      vmulq_f16(lookup_tq4_codebook8(vzip2_u8(lo, hi), cb_bytes), rn_v));
                        }
                    }

                    matmul_f16_ntile12_segment_accum(code_basis + (size_t)g * gs, K,
                                                     b_tile.data(), C, M, gs, N,
                                                     n_start, actual_n);
                }
            }
        });
}

void transform_orth_activation(const CactusTQN* tqn, const __fp16* x_scaled,
                               __fp16* out) {
    const uint32_t K = tqn->dim1;
    for (uint32_t j = 0; j < K; ++j) {
        float32x4_t acc0 = vdupq_n_f32(0.f);
        float32x4_t acc1 = vdupq_n_f32(0.f);
        uint32_t i = 0;
        for (; i + 8 <= K; i += 8) {
            __fp16 rvals[8] = {
                tqn->orth_R[(size_t)(i + 0) * K + j],
                tqn->orth_R[(size_t)(i + 1) * K + j],
                tqn->orth_R[(size_t)(i + 2) * K + j],
                tqn->orth_R[(size_t)(i + 3) * K + j],
                tqn->orth_R[(size_t)(i + 4) * K + j],
                tqn->orth_R[(size_t)(i + 5) * K + j],
                tqn->orth_R[(size_t)(i + 6) * K + j],
                tqn->orth_R[(size_t)(i + 7) * K + j],
            };
            float16x8_t pv = vmulq_f16(vld1q_f16(x_scaled + i), vld1q_f16(rvals));
            acc0 = vaddq_f32(acc0, vcvt_f32_f16(vget_low_f16(pv)));
            acc1 = vaddq_f32(acc1, vcvt_f32_f16(vget_high_f16(pv)));
        }
        float acc = vaddvq_f32(acc0) + vaddvq_f32(acc1);
        for (; i < K; ++i) acc += (float)x_scaled[i] * (float)tqn->orth_R[(size_t)i * K + j];
        out[j] = (__fp16)acc;
    }
}

}  // namespace

void cactus_matmul_tq2_f16(const CactusTQ2* tq2, const __fp16* A, __fp16* C,
                           size_t M, size_t K, size_t N) {
    if (tq2 == nullptr || A == nullptr || C == nullptr || M == 0 || K == 0 || N == 0) return;
    const size_t expected_K = (size_t)tq2->num_groups * tq2->group_size;
    const size_t rows = std::min<size_t>(N, tq2->vocab);
    if (K != expected_K) return;

    thread_local std::vector<__fp16> x_scaled_buf;
    if (x_scaled_buf.size() < M * K) x_scaled_buf.resize(M * K);
    scale_activations_by_input_scale(A, tq2->input_scale, tq2->input_scale_recip,
                                     true, x_scaled_buf.data(), M, K);
    const __fp16* x_scaled = x_scaled_buf.data();

    if (M != 1) {
#ifdef __APPLE__
        if (tq_prefill_blas_tiles(x_scaled, C, M, K, N, rows,
                                  [tq2](uint32_t row, __fp16* out) {
                                      tq2_dequant_row_scaled_no_input_scale(tq2, row, out);
                                  })) {
            return;
        }
#endif
        constexpr size_t TILE_N = 14;
        const size_t n_blocks = (rows + TILE_N - 1) / TILE_N;
        tq_parallel_ranges(n_blocks, 8,
            [tq2, x_scaled, C, M, K, N, rows](size_t block_start, size_t block_end) {
                thread_local std::vector<__fp16> b_tile;
                if (b_tile.size() < 14 * K) b_tile.resize(14 * K);
                for (size_t block = block_start; block < block_end; ++block) {
                    size_t n_start = block * 14;
                    size_t actual_n = std::min<size_t>(14, rows - n_start);
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        tq2_dequant_row_scaled_no_input_scale(
                            tq2, (uint32_t)(n_start + ni), b_tile.data() + ni * K);
                    }
                    matmul_f16_ntile6(x_scaled, b_tile.data(), C, M, K, N, n_start, actual_n);
                }
            });
        return;
    }

    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < M * K) code_basis_buf.resize(M * K);
    __fp16* code_basis_all = code_basis_buf.data();
    const uint32_t G = tq2->num_groups;
    const uint32_t gs = tq2->group_size;
    for (size_t idx = 0; idx < M * G; ++idx) {
        size_t m = idx / G;
        size_t g = idx - m * G;
        tq2_transform_hadamard_activation(tq2,
            x_scaled + m * K + g * gs,
            code_basis_all + m * K + g * gs);
    }

    tq_parallel_ranges(rows, 32,
        [tq2, code_basis_all, C, M, K, N](size_t n_start, size_t n_end) {
            thread_local std::vector<float> acc_buf;
            if (acc_buf.size() < M) acc_buf.resize(M);

            const uint32_t G = tq2->num_groups;
            const uint32_t gs = tq2->group_size;
            for (size_t n = n_start; n < n_end; ++n) {
                std::fill(acc_buf.begin(), acc_buf.begin() + M, 0.f);
                const __fp16* row_norms = tq2->scales + n * G;

                for (uint32_t g = 0; g < G; ++g) {
                    const float rn = (float)row_norms[g];
                    for (size_t m = 0; m < M; ++m) {
                        acc_buf[m] += rn * tq2_codebook_dot_group(
                            tq2, (uint32_t)n, g, code_basis_all + m * K + (size_t)g * gs);
                    }
                }
                for (size_t m = 0; m < M; ++m) C[m * N + n] = (__fp16)acc_buf[m];
            }
        });
}

void cactus_matmul_tqn_f16(const CactusTQN* tqn, const __fp16* A, __fp16* C,
                           size_t M, size_t K, size_t N) {
    if (tqn == nullptr || A == nullptr || C == nullptr || M == 0 || K == 0 || N == 0) return;
    const size_t rows = std::min<size_t>(N, tqn->dim0);
    if (K != tqn->dim1) return;

    if (tqn->rotation_family == kRotationOrthFull) {
        if (M != 1) {
            tq_parallel_ranges(rows, 16,
                [tqn, A, C, M, K, N](size_t n_start, size_t n_end) {
                    thread_local std::vector<__fp16> row_buf;
                    if (row_buf.size() < K) row_buf.resize(K);
                    for (size_t n = n_start; n < n_end; ++n) {
                        cactus_tqn_dequant_row(tqn, (uint32_t)n, row_buf.data());
                        for (size_t m = 0; m < M; ++m) {
                            C[m * N + n] = (__fp16)dot_f16_f32(A + m * K, row_buf.data(), K);
                        }
                    }
                });
            return;
        }

        thread_local std::vector<__fp16> x_scaled_buf;
        if (x_scaled_buf.size() < M * K) x_scaled_buf.resize(M * K);
        scale_activations_by_input_scale(A, tqn->input_scale, tqn->input_scale_recip,
                                         tqn->has_input_scale != 0,
                                         x_scaled_buf.data(), M, K);

        thread_local std::vector<__fp16> code_basis_buf;
        if (code_basis_buf.size() < M * K) code_basis_buf.resize(M * K);
        for (size_t m = 0; m < M; ++m) {
            transform_orth_activation(tqn, x_scaled_buf.data() + m * K,
                                      code_basis_buf.data() + m * K);
        }

        tq_parallel_ranges(rows, 16,
            [tqn, code_basis = code_basis_buf.data(), C, M, K, N](size_t n_start, size_t n_end) {
                for (size_t n = n_start; n < n_end; ++n) {
                    const float rn = (float)tqn->scales[n];
                    for (size_t m = 0; m < M; ++m) {
                        C[m * N + n] = (__fp16)(rn * tqn_codebook_dot_group(
                            tqn, (uint32_t)n, 0, code_basis + m * K));
                    }
                }
            });
        return;
    }

    thread_local std::vector<__fp16> x_scaled_buf;
    if (x_scaled_buf.size() < M * K) x_scaled_buf.resize(M * K);
    scale_activations_by_input_scale(A, tqn->input_scale, tqn->input_scale_recip,
                                     tqn->has_input_scale != 0,
                                     x_scaled_buf.data(), M, K);
    const __fp16* x_scaled = x_scaled_buf.data();

    if (M != 1) {
        if (false && tqn->bits == 4) {
            thread_local std::vector<__fp16> code_basis_buf;
            if (code_basis_buf.size() < M * K) code_basis_buf.resize(M * K);
            __fp16* code_basis_all = code_basis_buf.data();
            const uint32_t G = tqn->num_groups;
            const uint32_t gs = tqn->group_size;
            for (size_t idx = 0; idx < M * G; ++idx) {
                size_t m = idx / G;
                size_t g = idx - m * G;
                tqn_transform_hadamard_activation(tqn,
                    x_scaled + m * K + g * gs,
                    code_basis_all + m * K + g * gs);
            }
            tqn_prefill_tq4_codebasis_group_gemm(tqn, code_basis_all, C, M, K, N, rows);
            return;
        }
#ifdef __APPLE__
        if (tq_prefill_blas_tiles(x_scaled, C, M, K, N, rows,
                                  [tqn](uint32_t row, __fp16* out) {
                                      tqn_dequant_row_scaled_no_input_scale(tqn, row, out);
                                  })) {
            return;
        }
#endif
        constexpr size_t TILE_N = 14;
        const size_t n_blocks = (rows + TILE_N - 1) / TILE_N;
        tq_parallel_ranges(n_blocks, 8,
            [tqn, x_scaled, C, M, K, N, rows](size_t block_start, size_t block_end) {
                thread_local std::vector<__fp16> b_tile;
                if (b_tile.size() < 14 * K) b_tile.resize(14 * K);
                for (size_t block = block_start; block < block_end; ++block) {
                    size_t n_start = block * 14;
                    size_t actual_n = std::min<size_t>(14, rows - n_start);
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        tqn_dequant_row_scaled_no_input_scale(
                            tqn, (uint32_t)(n_start + ni), b_tile.data() + ni * K);
                    }
                    matmul_f16_ntile6(x_scaled, b_tile.data(), C, M, K, N, n_start, actual_n);
                }
            });
        return;
    }

    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < M * K) code_basis_buf.resize(M * K);
    __fp16* code_basis_all = code_basis_buf.data();
    const uint32_t G = tqn->num_groups;
    const uint32_t gs = tqn->group_size;
    for (size_t idx = 0; idx < M * G; ++idx) {
        size_t m = idx / G;
        size_t g = idx - m * G;
        tqn_transform_hadamard_activation(tqn,
            x_scaled + m * K + g * gs,
            code_basis_all + m * K + g * gs);
    }

    if (tqn->bits == 4) {
        tqn_gemv_tq4_codebasis_tiled(tqn, code_basis_all, C, K, N, rows);
        return;
    }

    tq_parallel_ranges(rows, 32,
        [tqn, code_basis_all, C, M, K, N](size_t n_start, size_t n_end) {
            thread_local std::vector<float> acc_buf;
            if (acc_buf.size() < M) acc_buf.resize(M);

            const uint32_t G = tqn->num_groups;
            const uint32_t gs = tqn->group_size;
            for (size_t n = n_start; n < n_end; ++n) {
                std::fill(acc_buf.begin(), acc_buf.begin() + M, 0.f);
                const __fp16* row_norms = tqn->scales + n * G;

                for (uint32_t g = 0; g < G; ++g) {
                    const float rn = (float)row_norms[g];
                    for (size_t m = 0; m < M; ++m) {
                        acc_buf[m] += rn * tqn_codebook_dot_group(
                            tqn, (uint32_t)n, g, code_basis_all + m * K + (size_t)g * gs);
                    }
                }
                for (size_t m = 0; m < M; ++m) C[m * N + n] = (__fp16)acc_buf[m];
            }
        });
}
