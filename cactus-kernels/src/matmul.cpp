#include "../cactus_kernels.h"
#include "threading.h"
#include <arm_neon.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
constexpr size_t ACCELERATE_M_THRESHOLD = 4;
constexpr size_t ACCELERATE_K_THRESHOLD = 256;
#endif

// Do NOT Remove: Uncomment for testing on various paths
// -----
// TEMPORARY: Force fallback path for testing on DOTPROD devices
// #undef __ARM_FEATURE_DOTPROD

#if defined(__ARM_FEATURE_DOTPROD)
    #define CACTUS_DOTQ_LANE(acc, b, a, lane) vdotq_laneq_s32(acc, b, a, lane)
#else
    static inline int32x4_t cactus_dotq_with_pattern(int32x4_t acc, int8x16_t b, int8x8_t a_pattern) {
        int8x8_t b_lo = vget_low_s8(b);
        int8x8_t b_hi = vget_high_s8(b);

        int16x8_t prod_lo = vmull_s8(b_lo, a_pattern);
        int16x8_t prod_hi = vmull_s8(b_hi, a_pattern);

        int32x4_t sum_lo = vpaddlq_s16(prod_lo);
        int32x4_t sum_hi = vpaddlq_s16(prod_hi);

        int32x2_t final_lo = vpadd_s32(vget_low_s32(sum_lo), vget_high_s32(sum_lo));
        int32x2_t final_hi = vpadd_s32(vget_low_s32(sum_hi), vget_high_s32(sum_hi));

        return vaddq_s32(acc, vcombine_s32(final_lo, final_hi));
    }

    static inline int32x4_t cactus_dotq_lane0(int32x4_t acc, int8x16_t b, int8x16_t a) {
        int8x8_t a_lo = vget_low_s8(a);
        int8x8_t a_pattern = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(a_lo), 0));
        return cactus_dotq_with_pattern(acc, b, a_pattern);
    }

    static inline int32x4_t cactus_dotq_lane1(int32x4_t acc, int8x16_t b, int8x16_t a) {
        int8x8_t a_lo = vget_low_s8(a);
        int8x8_t a_pattern = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(a_lo), 1));
        return cactus_dotq_with_pattern(acc, b, a_pattern);
    }

    static inline int32x4_t cactus_dotq_lane2(int32x4_t acc, int8x16_t b, int8x16_t a) {
        int8x8_t a_hi = vget_high_s8(a);
        int8x8_t a_pattern = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(a_hi), 0));
        return cactus_dotq_with_pattern(acc, b, a_pattern);
    }

    static inline int32x4_t cactus_dotq_lane3(int32x4_t acc, int8x16_t b, int8x16_t a) {
        int8x8_t a_hi = vget_high_s8(a);
        int8x8_t a_pattern = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(a_hi), 1));
        return cactus_dotq_with_pattern(acc, b, a_pattern);
    }

    #define CACTUS_DOTQ_LANE(acc, b, a, lane) cactus_dotq_lane##lane(acc, b, a)
#endif

static inline __fp16 hsum_f16x8(float16x8_t v) {
    float16x4_t lo = vget_low_f16(v);
    float16x4_t hi = vget_high_f16(v);
    float16x4_t sum4 = vadd_f16(lo, hi);
    float16x4_t sum2 = vadd_f16(sum4, vext_f16(sum4, sum4, 2));
    float16x4_t sum1 = vadd_f16(sum2, vext_f16(sum2, sum2, 1));
    return vget_lane_f16(sum1, 0);
}

namespace {

constexpr uint32_t kTQPanelN = 4;
constexpr uint32_t kTQPanelKChunk = 16;

static inline bool cactus_tq_panel_major(const CactusTQMatrix& W) {
    return (W.flags & CACTUS_TQ_FLAG_PANEL_MAJOR) != 0;
}

static inline bool cactus_tq_code_ordered(const CactusTQMatrix& W) {
    return (W.flags & CACTUS_TQ_FLAG_CODE_ORDERED_INDICES) != 0;
}

static inline float16x8_t cactus_tq_signs_to_f16(const int8_t* signs, uint32_t offset) {
    if (signs == nullptr) return vdupq_n_f16(1);
    return vcvtq_f16_s16(vmovl_s8(vld1_s8(signs + offset)));
}

static inline float16x8_t cactus_tq_input_scale_recip8(const CactusTQMatrix& W, uint32_t offset) {
    if (W.input_scale_recip != nullptr) {
        return vld1q_f16(W.input_scale_recip + offset);
    }
    if (W.input_scale != nullptr) {
        return vdivq_f16(vdupq_n_f16(1), vld1q_f16(W.input_scale + offset));
    }
    return vdupq_n_f16(1);
}

static inline __fp16 cactus_tq_input_scale_recip1(const CactusTQMatrix& W, uint32_t offset) {
    if (W.input_scale_recip != nullptr) return W.input_scale_recip[offset];
    if (W.input_scale != nullptr) return static_cast<__fp16>(1.0f / static_cast<float>(W.input_scale[offset]));
    return static_cast<__fp16>(1);
}

static inline uint32_t cactus_tq_panel_chunks(const CactusTQMatrix& W) {
    return W.group_size / kTQPanelKChunk;
}

static inline uint32_t cactus_tq_panel_chunk_bytes(uint32_t bits) {
    return (kTQPanelKChunk * bits) / 8;
}

static inline const __fp16* cactus_tq_scale_ptr(const CactusTQMatrix& W, uint32_t row, uint32_t group) {
    if (!cactus_tq_panel_major(W)) {
        return W.norms + static_cast<size_t>(row) * W.num_groups + group;
    }

    const uint32_t lane = row & (kTQPanelN - 1);
    const uint32_t block = row / kTQPanelN;
    return W.norms + (((static_cast<size_t>(block) * W.num_groups + group) * kTQPanelN) + lane);
}

static inline const uint8_t* cactus_tq_packed_chunk_ptr(
    const CactusTQMatrix& W,
    uint32_t row,
    uint32_t group,
    uint32_t k) {
    if (!cactus_tq_panel_major(W)) {
        return W.packed_indices
            + (((static_cast<size_t>(row) * W.num_groups + group)
                * cactus_tq_packed_group_bytes(W.bits, W.group_size))
               + (static_cast<size_t>(k) * W.bits) / 8);
    }

    const uint32_t lane = row & (kTQPanelN - 1);
    const uint32_t block = row / kTQPanelN;
    const uint32_t chunk = k / kTQPanelKChunk;
    const uint32_t intra = ((k % kTQPanelKChunk) * W.bits) / 8;
    const uint32_t chunk_bytes = cactus_tq_panel_chunk_bytes(W.bits);
    return W.packed_indices
        + ((((static_cast<size_t>(block) * W.num_groups + group)
             * cactus_tq_panel_chunks(W) + chunk)
            * kTQPanelN + lane)
           * chunk_bytes)
        + intra;
}

static inline float16x8_t cactus_tq4_lookup_codebook8(uint8x8_t nibbles, uint8x16x2_t cb_bytes) {
    uint8x8_t byte_offsets = vshl_n_u8(nibbles, 1);
    uint8x8_t byte_offsets_hi = vadd_u8(byte_offsets, vdup_n_u8(1));
    uint8x8x2_t zipped = vzip_u8(byte_offsets, byte_offsets_hi);
    uint8x16_t byte_idx = vcombine_u8(zipped.val[0], zipped.val[1]);
    return vreinterpretq_f16_u8(vqtbl2q_u8(cb_bytes, byte_idx));
}

static inline uint8x8_t cactus_tq2_unpack_8x2bit_le(uint8_t b0, uint8_t b1) {
    uint64_t idx_word =
        ((uint64_t)((b0     ) & 0x3)      ) |
        ((uint64_t)((b0 >> 2) & 0x3) <<  8) |
        ((uint64_t)((b0 >> 4) & 0x3) << 16) |
        ((uint64_t)((b0 >> 6) & 0x3) << 24) |
        ((uint64_t)((b1     ) & 0x3) << 32) |
        ((uint64_t)((b1 >> 2) & 0x3) << 40) |
        ((uint64_t)((b1 >> 4) & 0x3) << 48) |
        ((uint64_t)((b1 >> 6) & 0x3) << 56);
    return vcreate_u8(idx_word);
}

static inline float16x8_t cactus_tq2_lookup_codebook8(uint8x8_t indices, uint8x8_t cb_bytes) {
    uint8x8_t off_lo = vshl_n_u8(indices, 1);
    uint8x8_t off_hi = vadd_u8(off_lo, vdup_n_u8(1));
    uint8x8x2_t zipped = vzip_u8(off_lo, off_hi);
    uint8x16_t byte_idx = vcombine_u8(zipped.val[0], zipped.val[1]);
    uint8x16_t lut = vcombine_u8(cb_bytes, cb_bytes);
    return vreinterpretq_f16_u8(vqtbl1q_u8(lut, byte_idx));
}

static void cactus_tq_fwht128_f16(__fp16* x) {
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
                v[base + j] = vaddq_f16(a, b);
                v[base + j + s] = vsubq_f16(a, b);
            }
        }
    };
    pass(1);
    pass(2);
    pass(4);
    pass(8);

    float16x8_t inv = vdupq_n_f16(static_cast<__fp16>(1.0f / std::sqrt(128.0f)));
    for (int i = 0; i < 16; ++i) {
        vst1q_f16(x + i * 8, vmulq_f16(v[i], inv));
    }
}

static void cactus_tq_fwht_f16(__fp16* x, uint32_t n) {
    for (uint32_t h = 1; h < n; h <<= 1) {
        for (uint32_t i = 0; i < n; i += (h << 1)) {
            for (uint32_t j = i; j < i + h; j += 8) {
                if (j + 8 <= i + h) {
                    float16x8_t a = vld1q_f16(x + j);
                    float16x8_t b = vld1q_f16(x + j + h);
                    vst1q_f16(x + j, vaddq_f16(a, b));
                    vst1q_f16(x + j + h, vsubq_f16(a, b));
                } else {
                    for (uint32_t k = j; k < i + h; ++k) {
                        __fp16 a = x[k];
                        __fp16 b = x[k + h];
                        x[k] = static_cast<__fp16>(a + b);
                        x[k + h] = static_cast<__fp16>(a - b);
                    }
                }
            }
        }
    }

    const float16x8_t inv_v = vdupq_n_f16(static_cast<__fp16>(1.0f / std::sqrt(static_cast<float>(n))));
    uint32_t k = 0;
    for (; k + 8 <= n; k += 8) {
        vst1q_f16(x + k, vmulq_f16(vld1q_f16(x + k), inv_v));
    }
    const __fp16 inv = static_cast<__fp16>(1.0f / std::sqrt(static_cast<float>(n)));
    for (; k < n; ++k) {
        x[k] = static_cast<__fp16>(x[k] * inv);
    }
}

static void cactus_tq_transform_hadamard_group(
    const CactusTQMatrix& W,
    const __fp16* x_group,
    uint32_t group,
    __fp16* code_basis) {
    const uint32_t gs = W.group_size;
    __fp16 tmp[256];
    __fp16* work = (cactus_tq_code_ordered(W) || W.permutation == nullptr) ? code_basis : tmp;

    uint32_t k = 0;
    for (; k + 8 <= gs; k += 8) {
        const uint32_t offset = group * gs + k;
        float16x8_t x_v = vld1q_f16(x_group + k);
        x_v = vmulq_f16(x_v, cactus_tq_input_scale_recip8(W, offset));
        float16x8_t s_v = cactus_tq_signs_to_f16(W.left_signs, k);
        vst1q_f16(work + k, vmulq_f16(x_v, s_v));
    }
    for (; k < gs; ++k) {
        const uint32_t offset = group * gs + k;
        const float sign = W.left_signs ? static_cast<float>(W.left_signs[k]) : 1.0f;
        const float scale = static_cast<float>(cactus_tq_input_scale_recip1(W, offset));
        work[k] = static_cast<__fp16>(static_cast<float>(x_group[k]) * scale * sign);
    }

    if (gs == 128) {
        cactus_tq_fwht128_f16(work);
    } else {
        cactus_tq_fwht_f16(work, gs);
    }

    k = 0;
    for (; k + 8 <= gs; k += 8) {
        float16x8_t w_v = vld1q_f16(work + k);
        float16x8_t s_v = cactus_tq_signs_to_f16(W.right_signs, k);
        vst1q_f16(work + k, vmulq_f16(w_v, s_v));
    }
    for (; k < gs; ++k) {
        const float sign = W.right_signs ? static_cast<float>(W.right_signs[k]) : 1.0f;
        work[k] = static_cast<__fp16>(static_cast<float>(work[k]) * sign);
    }

    if (work != code_basis) {
        for (uint32_t j = 0; j < gs; ++j) {
            code_basis[j] = work[W.permutation[j]];
        }
    }
}

static void cactus_tq_transform_hadamard_activations(
    const CactusTQMatrix& W,
    const __fp16* A,
    uint32_t M,
    __fp16* code_basis) {
    const size_t work_items = static_cast<size_t>(M) * W.num_groups;
    CactusThreading::parallel_for(
        work_items,
        CactusThreading::ParallelConfig{16, 1},
        [&](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                const size_t m = idx / W.num_groups;
                const size_t g = idx - m * W.num_groups;
                cactus_tq_transform_hadamard_group(
                    W,
                    A + m * W.K + g * W.group_size,
                    static_cast<uint32_t>(g),
                    code_basis + m * W.K + g * W.group_size);
            }
        });
}

template<typename WorkFunc>
static void cactus_tq_parallel_ranges(size_t total_work, size_t work_per_thread, WorkFunc work_func) {
    if (total_work == 0) return;
    if (work_per_thread == 0) work_per_thread = 1;

    auto& pool = CactusThreading::get_thread_pool();
    size_t num_threads = std::min(pool.num_workers(), (total_work + work_per_thread - 1) / work_per_thread);
    num_threads = std::min(num_threads, total_work);
    if (num_threads <= 1) {
        work_func(0, total_work);
        return;
    }

    pool.enqueue_n_threads(total_work, num_threads, work_func);
    pool.wait_all();
}

static void cactus_tq_matmul_f16_segment_accum(
    const __fp16* A,
    size_t a_stride,
    const __fp16* B_tile,
    __fp16* C,
    size_t M,
    size_t Kseg,
    size_t N,
    size_t n_start,
    size_t actual_n) {
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 24;
    const size_t K16 = (Kseg / 16) * 16;
    const size_t K8 = (Kseg / 8) * 8;

    for (size_t m_start = 0; m_start < M; m_start += TILE_M) {
        const size_t actual_m = std::min(TILE_M, M - m_start);
        float16x8_t acc[TILE_M][TILE_N];
        for (size_t mi = 0; mi < TILE_M; ++mi) {
            for (size_t ni = 0; ni < TILE_N; ++ni) {
                acc[mi][ni] = vdupq_n_f16(0);
            }
        }

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
                a_v[mi] = mi < actual_m
                    ? vld1q_f16(A + (m_start + mi) * a_stride + k)
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
                        acc[mi][ni],
                        0);
                }
            }
        }

        for (size_t mi = 0; mi < actual_m; ++mi) {
            for (size_t ni = 0; ni < actual_n; ++ni) {
                __fp16* dst = C + (m_start + mi) * N + n_start + ni;
                *dst = static_cast<__fp16>(static_cast<float>(*dst) + static_cast<float>(hsum_f16x8(acc[mi][ni])));
            }
        }
    }
}

struct CactusTQ4ScaledDecoder {
    uint8x16x2_t cb_bytes;

    explicit CactusTQ4ScaledDecoder(const CactusTQMatrix& W) {
        cb_bytes.val[0] = vld1q_u8(reinterpret_cast<const uint8_t*>(W.codebook));
        cb_bytes.val[1] = vld1q_u8(reinterpret_cast<const uint8_t*>(W.codebook) + 16);
    }

    void operator()(const CactusTQMatrix& W, uint32_t row, uint32_t group, __fp16* dst) const {
        const __fp16 rn = *cactus_tq_scale_ptr(W, row, group);
        const float16x8_t rn_v = vdupq_n_f16(rn);
        for (uint32_t k = 0; k < W.group_size; k += 16) {
            const uint8_t* packed = cactus_tq_packed_chunk_ptr(W, row, group, k);
            uint8x8_t bytes = vld1_u8(packed);
            uint8x8_t lo = vand_u8(bytes, vdup_n_u8(0x0F));
            uint8x8_t hi = vshr_n_u8(bytes, 4);
            vst1q_f16(dst + k,
                      vmulq_f16(cactus_tq4_lookup_codebook8(vzip1_u8(lo, hi), cb_bytes), rn_v));
            vst1q_f16(dst + k + 8,
                      vmulq_f16(cactus_tq4_lookup_codebook8(vzip2_u8(lo, hi), cb_bytes), rn_v));
        }
    }
};

struct CactusTQ2ScaledDecoder {
    uint8x8_t cb_bytes;

    explicit CactusTQ2ScaledDecoder(const CactusTQMatrix& W)
        : cb_bytes(vld1_u8(reinterpret_cast<const uint8_t*>(W.codebook))) {}

    void operator()(const CactusTQMatrix& W, uint32_t row, uint32_t group, __fp16* dst) const {
        const __fp16 rn = *cactus_tq_scale_ptr(W, row, group);
        const float16x8_t rn_v = vdupq_n_f16(rn);
        for (uint32_t k = 0; k < W.group_size; k += 8) {
            const uint8_t* packed = cactus_tq_packed_chunk_ptr(W, row, group, k);
            uint8x8_t indices = cactus_tq2_unpack_8x2bit_le(packed[0], packed[1]);
            vst1q_f16(dst + k,
                      vmulq_f16(cactus_tq2_lookup_codebook8(indices, cb_bytes), rn_v));
        }
    }
};

template<uint32_t Bits, typename DecodeGroup>
static void cactus_tq_group_gemm(
    const CactusTQMatrix& W,
    const __fp16* A,
    uint32_t M,
    __fp16* C,
    DecodeGroup decode_group) {
    static_assert(Bits == 2 || Bits == 4);
    if (W.bits != Bits) return;

    constexpr size_t TILE_N = 16;
    const size_t n_blocks = (W.N + TILE_N - 1) / TILE_N;

    cactus_tq_parallel_ranges(n_blocks, 4, [&, decode_group](size_t block_start, size_t block_end) {
        thread_local std::vector<__fp16> b_tile;
        if (b_tile.size() < TILE_N * W.group_size) {
            b_tile.resize(TILE_N * W.group_size);
        }

        for (size_t block = block_start; block < block_end; ++block) {
            const size_t n_start = block * TILE_N;
            const size_t actual_n = std::min(TILE_N, static_cast<size_t>(W.N) - n_start);

            for (size_t m = 0; m < M; ++m) {
                for (size_t ni = 0; ni < actual_n; ++ni) {
                    C[m * W.N + n_start + ni] = 0;
                }
            }

            for (uint32_t g = 0; g < W.num_groups; ++g) {
                for (size_t ni = 0; ni < actual_n; ++ni) {
                    decode_group(W, static_cast<uint32_t>(n_start + ni), g,
                                 b_tile.data() + ni * W.group_size);
                }
                cactus_tq_matmul_f16_segment_accum(
                    A + static_cast<size_t>(g) * W.group_size,
                    W.K,
                    b_tile.data(),
                    C,
                    M,
                    W.group_size,
                    W.N,
                    n_start,
                    actual_n);
            }
        }
    });
}

static bool cactus_tq_valid_common(const CactusTQMatrix* W, const void* A, void* C) {
    if (W == nullptr || A == nullptr || C == nullptr) return false;
    if (W->K == 0 || W->N == 0 || W->group_size == 0 || W->num_groups == 0) return false;
    if (W->group_size > 256) return false;
    if ((W->group_size & (W->group_size - 1)) != 0) return false;
    if (W->K != W->group_size * W->num_groups) return false;
    if (W->codebook == nullptr || W->norms == nullptr || W->packed_indices == nullptr) return false;
    return true;
}

}  // namespace

static void cactus_matmul_f16_worker(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t /*M*/,
    size_t K,
    size_t N,
    size_t start_row,
    size_t end_row
) {
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 4;
    const size_t K16 = (K / 16) * 16;
    const size_t K8 = (K / 8) * 8;

    for (size_t row_block = start_row; row_block < end_row; row_block += TILE_M) {
        const size_t m_end = std::min(row_block + TILE_M, end_row);

        for (size_t col_block = 0; col_block < N; col_block += TILE_N) {
            const size_t n_end = std::min(col_block + TILE_N, N);

            float16x8_t acc[TILE_M][TILE_N];
            for (size_t m = 0; m < TILE_M; ++m)
                for (size_t n = 0; n < TILE_N; ++n)
                    acc[m][n] = vdupq_n_f16(0);

            for (size_t k = 0; k < K16; k += 16) {
                float16x8_t a0_lo = (row_block < m_end) ? vld1q_f16(a + row_block * K + k) : vdupq_n_f16(0);
                float16x8_t a0_hi = (row_block < m_end) ? vld1q_f16(a + row_block * K + k + 8) : vdupq_n_f16(0);
                float16x8_t a1_lo = (row_block + 1 < m_end) ? vld1q_f16(a + (row_block + 1) * K + k) : vdupq_n_f16(0);
                float16x8_t a1_hi = (row_block + 1 < m_end) ? vld1q_f16(a + (row_block + 1) * K + k + 8) : vdupq_n_f16(0);
                float16x8_t a2_lo = (row_block + 2 < m_end) ? vld1q_f16(a + (row_block + 2) * K + k) : vdupq_n_f16(0);
                float16x8_t a2_hi = (row_block + 2 < m_end) ? vld1q_f16(a + (row_block + 2) * K + k + 8) : vdupq_n_f16(0);
                float16x8_t a3_lo = (row_block + 3 < m_end) ? vld1q_f16(a + (row_block + 3) * K + k) : vdupq_n_f16(0);
                float16x8_t a3_hi = (row_block + 3 < m_end) ? vld1q_f16(a + (row_block + 3) * K + k + 8) : vdupq_n_f16(0);

                for (size_t ni = 0; ni < TILE_N && col_block + ni < n_end; ++ni) {
                    float16x8_t b_lo = vld1q_f16(b_transposed + (col_block + ni) * K + k);
                    float16x8_t b_hi = vld1q_f16(b_transposed + (col_block + ni) * K + k + 8);

                    acc[0][ni] = vfmaq_f16(acc[0][ni], a0_lo, b_lo);
                    acc[0][ni] = vfmaq_f16(acc[0][ni], a0_hi, b_hi);
                    acc[1][ni] = vfmaq_f16(acc[1][ni], a1_lo, b_lo);
                    acc[1][ni] = vfmaq_f16(acc[1][ni], a1_hi, b_hi);
                    acc[2][ni] = vfmaq_f16(acc[2][ni], a2_lo, b_lo);
                    acc[2][ni] = vfmaq_f16(acc[2][ni], a2_hi, b_hi);
                    acc[3][ni] = vfmaq_f16(acc[3][ni], a3_lo, b_lo);
                    acc[3][ni] = vfmaq_f16(acc[3][ni], a3_hi, b_hi);
                }
            }

            for (size_t k = K16; k < K8; k += 8) {
                float16x8_t a0_v = (row_block < m_end) ? vld1q_f16(a + row_block * K + k) : vdupq_n_f16(0);
                float16x8_t a1_v = (row_block + 1 < m_end) ? vld1q_f16(a + (row_block + 1) * K + k) : vdupq_n_f16(0);
                float16x8_t a2_v = (row_block + 2 < m_end) ? vld1q_f16(a + (row_block + 2) * K + k) : vdupq_n_f16(0);
                float16x8_t a3_v = (row_block + 3 < m_end) ? vld1q_f16(a + (row_block + 3) * K + k) : vdupq_n_f16(0);

                for (size_t ni = 0; ni < TILE_N && col_block + ni < n_end; ++ni) {
                    float16x8_t b_v = vld1q_f16(b_transposed + (col_block + ni) * K + k);
                    acc[0][ni] = vfmaq_f16(acc[0][ni], a0_v, b_v);
                    acc[1][ni] = vfmaq_f16(acc[1][ni], a1_v, b_v);
                    acc[2][ni] = vfmaq_f16(acc[2][ni], a2_v, b_v);
                    acc[3][ni] = vfmaq_f16(acc[3][ni], a3_v, b_v);
                }
            }

            for (size_t k = K8; k < K; ++k) {
                for (size_t mi = 0; mi < TILE_M && row_block + mi < m_end; ++mi) {
                    __fp16 av = a[(row_block + mi) * K + k];
                    for (size_t ni = 0; ni < TILE_N && col_block + ni < n_end; ++ni) {
                        __fp16 bv = b_transposed[(col_block + ni) * K + k];
                        acc[mi][ni] = vsetq_lane_f16(vgetq_lane_f16(acc[mi][ni], 0) + av * bv, acc[mi][ni], 0);
                    }
                }
            }

            for (size_t mi = 0; mi < TILE_M && row_block + mi < m_end; ++mi) {
                for (size_t ni = 0; ni < TILE_N && col_block + ni < n_end; ++ni) {
                    c[(row_block + mi) * N + col_block + ni] = hsum_f16x8(acc[mi][ni]);
                }
            }
        }
    }
}

void cactus_matmul_f16(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N
) {

#ifdef __APPLE__
    if (K >= ACCELERATE_K_THRESHOLD && M >= ACCELERATE_M_THRESHOLD) {
        const size_t a_len = M * K;
        const size_t b_len = N * K;
        const size_t c_len = M * N;

        std::vector<float> A_f32(a_len);
        std::vector<float> BT_f32(b_len);
        std::vector<float> C_f32(c_len);

        for (size_t i = 0; i < a_len; i++) A_f32[i] = (float)a[i];
        for (size_t i = 0; i < b_len; i++) BT_f32[i] = (float)b_transposed[i];

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)M, (int)N, (int)K,
                    1.0f, A_f32.data(), (int)K,
                    BT_f32.data(), (int)K,
                    0.0f, C_f32.data(), (int)N);

        for (size_t i = 0; i < c_len; i++) {
            float v = C_f32[i];
            if (v > 65504.f) v = 65504.f;
            else if (v < -65504.f) v = -65504.f;
            c[i] = (__fp16)v;
        }
        return;
    }
#endif

    constexpr size_t TILE_M = 4;
    const size_t num_row_blocks = (M + TILE_M - 1) / TILE_M;

    CactusThreading::parallel_for(num_row_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start_block, size_t end_block) {
            for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
                size_t start_row = block_idx * TILE_M;
                size_t end_row = std::min(start_row + TILE_M, M);

                cactus_matmul_f16_worker(
                    a, b_transposed, c,
                    M, K, N,
                    start_row, end_row
                );

            }
        });
}

uint32_t cactus_tq_packed_group_bytes(uint32_t bits, uint32_t group_size) {
    if (bits != 2 && bits != 4) return 0;
    const uint32_t values_per_byte = 8 / bits;
    return (group_size + values_per_byte - 1) / values_per_byte;
}

void cactus_tq4_gemv(
    const CactusTQMatrix* W,
    const __fp16* x,
    __fp16* y) {
    if (!cactus_tq_valid_common(W, x, y)) return;
    if (W->bits != 4 || (W->group_size % 16) != 0) return;

    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < W->K) code_basis_buf.resize(W->K);
    cactus_tq_transform_hadamard_activations(*W, x, 1, code_basis_buf.data());
    const __fp16* code_basis = code_basis_buf.data();

    constexpr size_t TILE_N = 12;
    const size_t n_blocks = (W->N + TILE_N - 1) / TILE_N;

    uint8x16x2_t cb_bytes;
    cb_bytes.val[0] = vld1q_u8(reinterpret_cast<const uint8_t*>(W->codebook));
    cb_bytes.val[1] = vld1q_u8(reinterpret_cast<const uint8_t*>(W->codebook) + 16);

    const size_t k_stride = cactus_tq_panel_major(*W)
        ? static_cast<size_t>(kTQPanelN) * cactus_tq_panel_chunk_bytes(W->bits)
        : static_cast<size_t>(kTQPanelKChunk) * W->bits / 8;

    cactus_tq_parallel_ranges(n_blocks, 16, [&](size_t block_start, size_t block_end) {
        for (size_t block = block_start; block < block_end; ++block) {
            const size_t n_start = block * TILE_N;
            const size_t actual_n = std::min(TILE_N, static_cast<size_t>(W->N) - n_start);
            float acc[TILE_N] = {};

            for (uint32_t g = 0; g < W->num_groups; ++g) {
                const __fp16* z = code_basis + static_cast<size_t>(g) * W->group_size;

                const uint8_t* packed[TILE_N] = {};
                float rn[TILE_N] = {};
                for (size_t ni = 0; ni < actual_n; ++ni) {
                    const uint32_t row = static_cast<uint32_t>(n_start + ni);
                    packed[ni] = cactus_tq_packed_chunk_ptr(*W, row, g, 0);
                    rn[ni] = static_cast<float>(*cactus_tq_scale_ptr(*W, row, g));
                }
                if (g + 1 < W->num_groups) {
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        __builtin_prefetch(cactus_tq_packed_chunk_ptr(
                            *W, static_cast<uint32_t>(n_start + ni), g + 1, 0));
                    }
                }

                float16x8_t acc0[TILE_N];
                float16x8_t acc1[TILE_N];
                for (size_t ni = 0; ni < TILE_N; ++ni) {
                    acc0[ni] = vdupq_n_f16(0);
                    acc1[ni] = vdupq_n_f16(0);
                }

                size_t byte_off = 0;
                for (uint32_t k = 0; k < W->group_size; k += 16) {
                    float16x8_t z0 = vld1q_f16(z + k);
                    float16x8_t z1 = vld1q_f16(z + k + 8);
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        const uint8_t* p = packed[ni] + byte_off;
                        uint8x8_t bytes = vld1_u8(p);
                        uint8x8_t lo = vand_u8(bytes, vdup_n_u8(0x0F));
                        uint8x8_t hi = vshr_n_u8(bytes, 4);

                        float16x8_t cv0 = cactus_tq4_lookup_codebook8(vzip1_u8(lo, hi), cb_bytes);
                        float16x8_t cv1 = cactus_tq4_lookup_codebook8(vzip2_u8(lo, hi), cb_bytes);
                        acc0[ni] = vfmaq_f16(acc0[ni], z0, cv0);
                        acc1[ni] = vfmaq_f16(acc1[ni], z1, cv1);
                    }
                    byte_off += k_stride;
                }

                for (size_t ni = 0; ni < actual_n; ++ni) {
                    acc[ni] += rn[ni] *
                        (static_cast<float>(hsum_f16x8(acc0[ni])) +
                         static_cast<float>(hsum_f16x8(acc1[ni])));
                }
            }

            for (size_t ni = 0; ni < actual_n; ++ni) {
                y[n_start + ni] = static_cast<__fp16>(acc[ni]);
            }
        }
    });
}

void cactus_tq2_gemv(
    const CactusTQMatrix* W,
    const __fp16* x,
    __fp16* y) {
    if (!cactus_tq_valid_common(W, x, y)) return;
    if (W->bits != 2 || (W->group_size % 8) != 0) return;

    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < W->K) code_basis_buf.resize(W->K);
    cactus_tq_transform_hadamard_activations(*W, x, 1, code_basis_buf.data());
    const __fp16* code_basis = code_basis_buf.data();

    constexpr size_t TILE_N = 12;
    const size_t n_blocks = (W->N + TILE_N - 1) / TILE_N;

    uint8x8_t cb_bytes = vld1_u8(reinterpret_cast<const uint8_t*>(W->codebook));

    cactus_tq_parallel_ranges(n_blocks, 16, [&](size_t block_start, size_t block_end) {
        for (size_t block = block_start; block < block_end; ++block) {
            const size_t n_start = block * TILE_N;
            const size_t actual_n = std::min(TILE_N, static_cast<size_t>(W->N) - n_start);
            float acc[TILE_N] = {};

            for (uint32_t g = 0; g < W->num_groups; ++g) {
                const __fp16* z = code_basis + static_cast<size_t>(g) * W->group_size;

                const uint8_t* packed[TILE_N] = {};
                float rn[TILE_N] = {};
                for (size_t ni = 0; ni < actual_n; ++ni) {
                    const uint32_t row = static_cast<uint32_t>(n_start + ni);
                    packed[ni] = cactus_tq_packed_chunk_ptr(*W, row, g, 0);
                    rn[ni] = static_cast<float>(*cactus_tq_scale_ptr(*W, row, g));
                }

                float16x8_t accv[TILE_N];
                for (size_t ni = 0; ni < TILE_N; ++ni) {
                    accv[ni] = vdupq_n_f16(0);
                }

                for (uint32_t k = 0; k < W->group_size; k += 8) {
                    float16x8_t z_v = vld1q_f16(z + k);
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        const uint8_t* p = cactus_tq_packed_chunk_ptr(
                            *W, static_cast<uint32_t>(n_start + ni), g, k);
                        uint8x8_t indices = cactus_tq2_unpack_8x2bit_le(p[0], p[1]);
                        float16x8_t cv = cactus_tq2_lookup_codebook8(indices, cb_bytes);
                        accv[ni] = vfmaq_f16(accv[ni], z_v, cv);
                    }
                }

                for (size_t ni = 0; ni < actual_n; ++ni) {
                    acc[ni] += rn[ni] * static_cast<float>(hsum_f16x8(accv[ni]));
                }
            }

            for (size_t ni = 0; ni < actual_n; ++ni) {
                y[n_start + ni] = static_cast<__fp16>(acc[ni]);
            }
        }
    });
}

void cactus_tq4_gemm(
    const CactusTQMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C) {
    if (!cactus_tq_valid_common(W, A, C) || M == 0) return;
    if (W->bits != 4 || (W->group_size % 16) != 0) return;
    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < static_cast<size_t>(M) * W->K) {
        code_basis_buf.resize(static_cast<size_t>(M) * W->K);
    }
    cactus_tq_transform_hadamard_activations(*W, A, M, code_basis_buf.data());
    cactus_tq_group_gemm<4>(*W, code_basis_buf.data(), M, C, CactusTQ4ScaledDecoder(*W));
}

void cactus_tq2_gemm(
    const CactusTQMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C) {
    if (!cactus_tq_valid_common(W, A, C) || M == 0) return;
    if (W->bits != 2 || (W->group_size % 8) != 0) return;
    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < static_cast<size_t>(M) * W->K) {
        code_basis_buf.resize(static_cast<size_t>(M) * W->K);
    }
    cactus_tq_transform_hadamard_activations(*W, A, M, code_basis_buf.data());
    cactus_tq_group_gemm<2>(*W, code_basis_buf.data(), M, C, CactusTQ2ScaledDecoder(*W));
}

void cactus_gemv_int8(
    const int8_t* A,
    const float A_scale,
    const int8_t* B,
    const __fp16* B_scales,
    __fp16* C,
    size_t K, size_t N,
    size_t group_size
) {
    if (K == 0 || N == 0) return;

    const size_t num_groups = K / group_size;
    const size_t N_blocks = (N + 3) / 4;

    auto process_blocks = [=](size_t block_start, size_t block_end) {
        for (size_t n_block = block_start; n_block < block_end; ++n_block) {
            const size_t n_start = n_block * 4;
            const size_t actual_n = std::min(size_t(4), N - n_start);

            float32x4_t running_sum = vdupq_n_f32(0.0f);

            size_t g = 0;
            for (; g + 1 < num_groups; g += 2) {
                const size_t k_base0 = g * group_size;
                const size_t k_base1 = (g + 1) * group_size;

                const int8_t* a_ptr0 = A + k_base0;
                const int8_t* a_ptr1 = A + k_base1;
                const int8_t* b_base0 = B + (n_block * K + k_base0) * 4;
                const int8_t* b_base1 = B + (n_block * K + k_base1) * 4;

                __builtin_prefetch(b_base0 + group_size * 8, 0, 3);

                int32x4_t acc0 = vdupq_n_s32(0);
                int32x4_t acc1 = vdupq_n_s32(0);

                {
                    int8x16_t a_vec = vld1q_s8(a_ptr0);
                    int8x16_t b0 = vld1q_s8(b_base0);
                    int8x16_t b1 = vld1q_s8(b_base0 + 16);
                    int8x16_t b2 = vld1q_s8(b_base0 + 32);
                    int8x16_t b3 = vld1q_s8(b_base0 + 48);

                    acc0 = CACTUS_DOTQ_LANE(acc0, b0, a_vec, 0);
                    acc0 = CACTUS_DOTQ_LANE(acc0, b1, a_vec, 1);
                    acc0 = CACTUS_DOTQ_LANE(acc0, b2, a_vec, 2);
                    acc0 = CACTUS_DOTQ_LANE(acc0, b3, a_vec, 3);

                    a_vec = vld1q_s8(a_ptr0 + 16);
                    b0 = vld1q_s8(b_base0 + 64);
                    b1 = vld1q_s8(b_base0 + 80);
                    b2 = vld1q_s8(b_base0 + 96);
                    b3 = vld1q_s8(b_base0 + 112);

                    acc0 = CACTUS_DOTQ_LANE(acc0, b0, a_vec, 0);
                    acc0 = CACTUS_DOTQ_LANE(acc0, b1, a_vec, 1);
                    acc0 = CACTUS_DOTQ_LANE(acc0, b2, a_vec, 2);
                    acc0 = CACTUS_DOTQ_LANE(acc0, b3, a_vec, 3);
                }

                {
                    int8x16_t a_vec = vld1q_s8(a_ptr1);
                    int8x16_t b0 = vld1q_s8(b_base1);
                    int8x16_t b1 = vld1q_s8(b_base1 + 16);
                    int8x16_t b2 = vld1q_s8(b_base1 + 32);
                    int8x16_t b3 = vld1q_s8(b_base1 + 48);

                    acc1 = CACTUS_DOTQ_LANE(acc1, b0, a_vec, 0);
                    acc1 = CACTUS_DOTQ_LANE(acc1, b1, a_vec, 1);
                    acc1 = CACTUS_DOTQ_LANE(acc1, b2, a_vec, 2);
                    acc1 = CACTUS_DOTQ_LANE(acc1, b3, a_vec, 3);

                    a_vec = vld1q_s8(a_ptr1 + 16);
                    b0 = vld1q_s8(b_base1 + 64);
                    b1 = vld1q_s8(b_base1 + 80);
                    b2 = vld1q_s8(b_base1 + 96);
                    b3 = vld1q_s8(b_base1 + 112);

                    acc1 = CACTUS_DOTQ_LANE(acc1, b0, a_vec, 0);
                    acc1 = CACTUS_DOTQ_LANE(acc1, b1, a_vec, 1);
                    acc1 = CACTUS_DOTQ_LANE(acc1, b2, a_vec, 2);
                    acc1 = CACTUS_DOTQ_LANE(acc1, b3, a_vec, 3);
                }

                const __fp16* scale_ptr0 = B_scales + (n_block * num_groups + g) * 4;
                const __fp16* scale_ptr1 = B_scales + (n_block * num_groups + g + 1) * 4;

                float16x4_t scales0_f16 = vld1_f16(scale_ptr0);
                float16x4_t scales1_f16 = vld1_f16(scale_ptr1);
                float32x4_t scales0 = vcvt_f32_f16(scales0_f16);
                float32x4_t scales1 = vcvt_f32_f16(scales1_f16);

                running_sum = vmlaq_f32(running_sum, vcvtq_f32_s32(acc0), scales0);
                running_sum = vmlaq_f32(running_sum, vcvtq_f32_s32(acc1), scales1);
            }

            for (; g < num_groups; g++) {
                const size_t k_base = g * group_size;
                const int8_t* a_ptr = A + k_base;
                const int8_t* b_base = B + (n_block * K + k_base) * 4;

                int32x4_t acc = vdupq_n_s32(0);

                int8x16_t a_vec = vld1q_s8(a_ptr);
                int8x16_t b0 = vld1q_s8(b_base);
                int8x16_t b1 = vld1q_s8(b_base + 16);
                int8x16_t b2 = vld1q_s8(b_base + 32);
                int8x16_t b3 = vld1q_s8(b_base + 48);

                acc = CACTUS_DOTQ_LANE(acc, b0, a_vec, 0);
                acc = CACTUS_DOTQ_LANE(acc, b1, a_vec, 1);
                acc = CACTUS_DOTQ_LANE(acc, b2, a_vec, 2);
                acc = CACTUS_DOTQ_LANE(acc, b3, a_vec, 3);

                a_vec = vld1q_s8(a_ptr + 16);
                b0 = vld1q_s8(b_base + 64);
                b1 = vld1q_s8(b_base + 80);
                b2 = vld1q_s8(b_base + 96);
                b3 = vld1q_s8(b_base + 112);

                acc = CACTUS_DOTQ_LANE(acc, b0, a_vec, 0);
                acc = CACTUS_DOTQ_LANE(acc, b1, a_vec, 1);
                acc = CACTUS_DOTQ_LANE(acc, b2, a_vec, 2);
                acc = CACTUS_DOTQ_LANE(acc, b3, a_vec, 3);

                const __fp16* scale_ptr = B_scales + (n_block * num_groups + g) * 4;
                float16x4_t scales_f16 = vld1_f16(scale_ptr);
                float32x4_t scales = vcvt_f32_f16(scales_f16);

                running_sum = vmlaq_f32(running_sum, vcvtq_f32_s32(acc), scales);
            }

            float32x4_t result = vmulq_n_f32(running_sum, A_scale);
            float16x4_t result_f16 = vcvt_f16_f32(result);

            if (actual_n == 4) {
                vst1_f16(C + n_start, result_f16);
            } else {
                for (size_t ni = 0; ni < actual_n; ni++) {
                    C[n_start + ni] = vget_lane_f16(result_f16, 0);
                    result_f16 = vext_f16(result_f16, result_f16, 1);
                }
            }
        }
    };

    auto& pool = CactusThreading::get_thread_pool();
    size_t num_threads = CactusThreading::GemmThreading::get_gemv_threads(N_blocks, pool.num_workers());
    num_threads = std::min(num_threads, N_blocks);

    if (num_threads <= 1) {
        process_blocks(0, N_blocks);
    } else {
        pool.enqueue_n_threads(N_blocks, num_threads, process_blocks);
        pool.wait_all();
    }
}

void cactus_gemm_int8(
    const int8_t* A,
    const float* A_scales,
    const int8_t* B,
    const __fp16* B_scales,
    __fp16* C,
    size_t M, size_t K, size_t N,
    size_t group_size
) {
    if (M == 0 || K == 0 || N == 0) return;

    constexpr size_t TILE_M = 8;
    constexpr size_t TILE_N = 4;

    const size_t num_groups = K / group_size;
    const size_t N_blocks = (N + TILE_N - 1) / TILE_N;
    const size_t num_row_tiles = (M + TILE_M - 1) / TILE_M;
    const size_t total_tiles = num_row_tiles * N_blocks;

    CactusThreading::parallel_gemm_tiles(M, total_tiles,
        [=](size_t tile_start, size_t tile_end) {
            for (size_t tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
                const size_t tile_row = tile_idx / N_blocks;
                const size_t n_block = tile_idx % N_blocks;
                const size_t m_start = tile_row * TILE_M;
                const size_t m_end = std::min(m_start + TILE_M, M);
                const size_t n_start = n_block * TILE_N;
                const size_t n_end = std::min(n_start + TILE_N, N);
                const size_t actual_m = m_end - m_start;
                const size_t actual_n = n_end - n_start;

                const int8_t* a_rows[TILE_M];
                for (size_t mi = 0; mi < TILE_M; mi++) {
                    size_t row = m_start + (mi < actual_m ? mi : actual_m - 1);
                    a_rows[mi] = A + row * K;
                }

                float32x4_t running_sum[TILE_M];
                for (size_t mi = 0; mi < TILE_M; mi++) {
                    running_sum[mi] = vdupq_n_f32(0.0f);
                }

                for (size_t g = 0; g < num_groups; g++) {
                    const size_t k_base = g * group_size;
                    const int8_t* b_base = B + (n_block * K + k_base) * 4;

                    __builtin_prefetch(b_base + group_size * 4, 0, 3);

                    int8x16_t b00 = vld1q_s8(b_base);
                    int8x16_t b01 = vld1q_s8(b_base + 16);
                    int8x16_t b02 = vld1q_s8(b_base + 32);
                    int8x16_t b03 = vld1q_s8(b_base + 48);

                    int8x16_t b10 = vld1q_s8(b_base + 64);
                    int8x16_t b11 = vld1q_s8(b_base + 80);
                    int8x16_t b12 = vld1q_s8(b_base + 96);
                    int8x16_t b13 = vld1q_s8(b_base + 112);

                    const __fp16* scale_ptr = B_scales + (n_block * num_groups + g) * 4;
                    float16x4_t scales_f16 = vld1_f16(scale_ptr);
                    float32x4_t scales = vcvt_f32_f16(scales_f16);

                    #define CACTUS_GEMM_ROW(ROW) do { \
                        const int8_t* a_ptr_##ROW = a_rows[ROW] + k_base; \
                        int32x4_t acc_##ROW = vdupq_n_s32(0); \
                        int8x16_t a_lo_##ROW = vld1q_s8(a_ptr_##ROW); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b00, a_lo_##ROW, 0); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b01, a_lo_##ROW, 1); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b02, a_lo_##ROW, 2); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b03, a_lo_##ROW, 3); \
                        int8x16_t a_hi_##ROW = vld1q_s8(a_ptr_##ROW + 16); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b10, a_hi_##ROW, 0); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b11, a_hi_##ROW, 1); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b12, a_hi_##ROW, 2); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b13, a_hi_##ROW, 3); \
                        running_sum[ROW] = vmlaq_f32(running_sum[ROW], vcvtq_f32_s32(acc_##ROW), scales); \
                    } while(0)

                    CACTUS_GEMM_ROW(0);
                    CACTUS_GEMM_ROW(1);
                    CACTUS_GEMM_ROW(2);
                    CACTUS_GEMM_ROW(3);
                    CACTUS_GEMM_ROW(4);
                    CACTUS_GEMM_ROW(5);
                    CACTUS_GEMM_ROW(6);
                    CACTUS_GEMM_ROW(7);
                    #undef CACTUS_GEMM_ROW
                }

                for (size_t mi = 0; mi < actual_m; mi++) {
                    const float a_scale = A_scales[m_start + mi];
                    float32x4_t result = vmulq_n_f32(running_sum[mi], a_scale);
                    float16x4_t result_f16 = vcvt_f16_f32(result);

                    if (actual_n == 4) {
                        vst1_f16(C + (m_start + mi) * N + n_start, result_f16);
                    } else {
                        for (size_t ni = 0; ni < actual_n; ni++) {
                            C[(m_start + mi) * N + n_start + ni] = vget_lane_f16(result_f16, 0);
                            result_f16 = vext_f16(result_f16, result_f16, 1);
                        }
                    }
                }
            }
        });
}

void cactus_matmul_int8(
    const int8_t* A,
    const float* A_scales,
    const int8_t* B,
    const __fp16* B_scales,
    __fp16* C,
    size_t M, size_t K, size_t N,
    size_t group_size
) {
    if (M == 0 || K == 0 || N == 0) return;

#if defined(CACTUS_COMPILE_I8MM)
    if (cpu_has_i8mm()) {
        if (M == 1) {
            cactus_gemv_int8_i8mm(A, A_scales[0], B, B_scales, C, K, N, group_size);
        } else {
            cactus_gemm_int8_i8mm(A, A_scales, B, B_scales, C, M, K, N, group_size);
        }
        return;
    }
#endif

    if (M == 1) {
        cactus_gemv_int8(A, A_scales[0], B, B_scales, C, K, N, group_size);
    } else {
        cactus_gemm_int8(A, A_scales, B, B_scales, C, M, K, N, group_size);
    }
}

void cactus_gemv_int4(
    const int8_t* A,
    const float A_scale,
    const int8_t* B_packed_raw,
    const __fp16* B_scales,
    __fp16* C,
    size_t K, size_t N,
    size_t group_size
) {
    const uint8_t* B_packed = reinterpret_cast<const uint8_t*>(B_packed_raw);
    if (K == 0 || N == 0) return;

    const size_t num_groups = K / group_size;
    const size_t N_blocks = (N + 3) / 4;

    auto process_blocks = [=](size_t block_start, size_t block_end) {
        size_t n_block = block_start;

        for (; n_block + 1 < block_end; n_block += 2) {
            float32x4_t sum_a = vdupq_n_f32(0.0f);
            float32x4_t sum_b = vdupq_n_f32(0.0f);

            for (size_t g = 0; g < num_groups; g++) {
                const size_t k_base = g * group_size;
                const int8_t* a_ptr = A + k_base;
                const uint8_t* ba = B_packed + (n_block * K + k_base) * 2;
                const uint8_t* bb = B_packed + ((n_block + 1) * K + k_base) * 2;

                int32x4_t acc_a = vdupq_n_s32(0);
                int32x4_t acc_b = vdupq_n_s32(0);

                int8x16_t a_lo = vld1q_s8(a_ptr);
                int8x16_t a_hi = vld1q_s8(a_ptr + 16);

                {
                    int8x16_t b0, b1, b2, b3;
                    unpack_int4_as_int8x16x2(ba, b1, b0);
                    unpack_int4_as_int8x16x2(ba + 16, b3, b2);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b0, a_lo, 0);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b1, a_lo, 1);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b2, a_lo, 2);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b3, a_lo, 3);
                    unpack_int4_as_int8x16x2(ba + 32, b1, b0);
                    unpack_int4_as_int8x16x2(ba + 48, b3, b2);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b0, a_hi, 0);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b1, a_hi, 1);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b2, a_hi, 2);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b3, a_hi, 3);
                }
                {
                    int8x16_t b0, b1, b2, b3;
                    unpack_int4_as_int8x16x2(bb, b1, b0);
                    unpack_int4_as_int8x16x2(bb + 16, b3, b2);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b0, a_lo, 0);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b1, a_lo, 1);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b2, a_lo, 2);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b3, a_lo, 3);
                    unpack_int4_as_int8x16x2(bb + 32, b1, b0);
                    unpack_int4_as_int8x16x2(bb + 48, b3, b2);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b0, a_hi, 0);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b1, a_hi, 1);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b2, a_hi, 2);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b3, a_hi, 3);
                }

                const __fp16* spa = B_scales + (n_block * num_groups + g) * 4;
                const __fp16* spb = B_scales + ((n_block + 1) * num_groups + g) * 4;
                float32x4_t sa = vcvt_f32_f16(vld1_f16(spa));
                float32x4_t sb = vcvt_f32_f16(vld1_f16(spb));
                sum_a = vmlaq_f32(sum_a, vcvtq_f32_s32(acc_a), sa);
                sum_b = vmlaq_f32(sum_b, vcvtq_f32_s32(acc_b), sb);
            }

            vst1_f16(C + n_block * 4, vcvt_f16_f32(vmulq_n_f32(sum_a, A_scale)));
            vst1_f16(C + (n_block + 1) * 4, vcvt_f16_f32(vmulq_n_f32(sum_b, A_scale)));
        }

        for (; n_block < block_end; ++n_block) {
            const size_t n_start = n_block * 4;
            const size_t actual_n = std::min(size_t(4), N - n_start);
            float32x4_t running_sum = vdupq_n_f32(0.0f);

            for (size_t g = 0; g < num_groups; g++) {
                const size_t k_base = g * group_size;
                const int8_t* a_ptr = A + k_base;
                const uint8_t* b_base = B_packed + (n_block * K + k_base) * 2;

                int32x4_t acc = vdupq_n_s32(0);
                int8x16_t a_lo = vld1q_s8(a_ptr);
                int8x16_t a_hi = vld1q_s8(a_ptr + 16);

                int8x16_t b0, b1, b2, b3;
                unpack_int4_as_int8x16x2(b_base, b1, b0);
                unpack_int4_as_int8x16x2(b_base + 16, b3, b2);
                acc = CACTUS_DOTQ_LANE(acc, b0, a_lo, 0);
                acc = CACTUS_DOTQ_LANE(acc, b1, a_lo, 1);
                acc = CACTUS_DOTQ_LANE(acc, b2, a_lo, 2);
                acc = CACTUS_DOTQ_LANE(acc, b3, a_lo, 3);
                unpack_int4_as_int8x16x2(b_base + 32, b1, b0);
                unpack_int4_as_int8x16x2(b_base + 48, b3, b2);
                acc = CACTUS_DOTQ_LANE(acc, b0, a_hi, 0);
                acc = CACTUS_DOTQ_LANE(acc, b1, a_hi, 1);
                acc = CACTUS_DOTQ_LANE(acc, b2, a_hi, 2);
                acc = CACTUS_DOTQ_LANE(acc, b3, a_hi, 3);

                float32x4_t scales = vcvt_f32_f16(vld1_f16(B_scales + (n_block * num_groups + g) * 4));
                running_sum = vmlaq_f32(running_sum, vcvtq_f32_s32(acc), scales);
            }

            float32x4_t result = vmulq_n_f32(running_sum, A_scale);
            float16x4_t result_f16 = vcvt_f16_f32(result);
            if (actual_n == 4) {
                vst1_f16(C + n_start, result_f16);
            } else {
                for (size_t ni = 0; ni < actual_n; ni++) {
                    C[n_start + ni] = vget_lane_f16(result_f16, 0);
                    result_f16 = vext_f16(result_f16, result_f16, 1);
                }
            }
        }
    };

    auto& pool = CactusThreading::get_thread_pool();
    size_t num_threads = CactusThreading::GemmThreading::get_gemv_threads(N_blocks, pool.num_workers());
    num_threads = std::min(num_threads, N_blocks);

    if (num_threads <= 1) {
        process_blocks(0, N_blocks);
    } else {
        pool.enqueue_n_threads(N_blocks, num_threads, process_blocks);
        pool.wait_all();
    }
}


void cactus_gemm_int4(
    const int8_t* A,
    const float* A_scales,
    const int8_t* B_packed_raw,
    const __fp16* B_scales,
    __fp16* C,
    size_t M, size_t K, size_t N,
    size_t group_size
) {
    if (M == 0 || K == 0 || N == 0) return;

    const uint8_t* B_packed = reinterpret_cast<const uint8_t*>(B_packed_raw);

    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 4;

    const size_t num_groups = K / group_size;
    const size_t N_blocks = (N + TILE_N - 1) / TILE_N;
    const size_t num_row_tiles = (M + TILE_M - 1) / TILE_M;
    const size_t total_tiles = num_row_tiles * N_blocks;

    CactusThreading::parallel_gemm_tiles(M, total_tiles,
        [=](size_t tile_start, size_t tile_end) {
            for (size_t tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
                const size_t tile_row = tile_idx / N_blocks;
                const size_t n_block = tile_idx % N_blocks;
                const size_t m_start = tile_row * TILE_M;
                const size_t m_end = std::min(m_start + TILE_M, M);
                const size_t n_start = n_block * TILE_N;
                const size_t n_end = std::min(n_start + TILE_N, N);
                const size_t actual_m = m_end - m_start;
                const size_t actual_n = n_end - n_start;

                float32x4_t running_sum[TILE_M] = {
                    vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                    vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)
                };

                for (size_t g = 0; g < num_groups; g++) {
                    const size_t k_base = g * group_size;
                    const uint8_t* b_base = B_packed + (n_block * K + k_base) * 2;

                    __builtin_prefetch(b_base + group_size * 2, 0, 3);

                    int8x16_t b00, b01, b02, b03;
                    int8x16_t b10, b11, b12, b13;

                    unpack_int4_as_int8x16x2(b_base, b01, b00);
                    unpack_int4_as_int8x16x2(b_base + 16, b03, b02);
                    unpack_int4_as_int8x16x2(b_base + 32, b11, b10);
                    unpack_int4_as_int8x16x2(b_base + 48, b13, b12);

                    const __fp16* scale_ptr = B_scales + (n_block * num_groups + g) * 4;
                    float16x4_t scales_f16 = vld1_f16(scale_ptr);
                    float32x4_t scales = vcvt_f32_f16(scales_f16);

                    for (size_t mi = 0; mi < actual_m; mi++) {
                        const int8_t* a_ptr = A + (m_start + mi) * K + k_base;

                        int32x4_t acc = vdupq_n_s32(0);

                        int8x16_t a_vec = vld1q_s8(a_ptr);
                        acc = CACTUS_DOTQ_LANE(acc, b00, a_vec, 0);
                        acc = CACTUS_DOTQ_LANE(acc, b01, a_vec, 1);
                        acc = CACTUS_DOTQ_LANE(acc, b02, a_vec, 2);
                        acc = CACTUS_DOTQ_LANE(acc, b03, a_vec, 3);

                        a_vec = vld1q_s8(a_ptr + 16);
                        acc = CACTUS_DOTQ_LANE(acc, b10, a_vec, 0);
                        acc = CACTUS_DOTQ_LANE(acc, b11, a_vec, 1);
                        acc = CACTUS_DOTQ_LANE(acc, b12, a_vec, 2);
                        acc = CACTUS_DOTQ_LANE(acc, b13, a_vec, 3);

                        running_sum[mi] = vmlaq_f32(running_sum[mi], vcvtq_f32_s32(acc), scales);
                    }
                }

                for (size_t mi = 0; mi < actual_m; mi++) {
                    const float a_scale = A_scales[m_start + mi];
                    float32x4_t result = vmulq_n_f32(running_sum[mi], a_scale);
                    float16x4_t result_f16 = vcvt_f16_f32(result);

                    if (actual_n == 4) {
                        vst1_f16(C + (m_start + mi) * N + n_start, result_f16);
                    } else {
                        for (size_t ni = 0; ni < actual_n; ni++) {
                            C[(m_start + mi) * N + n_start + ni] = vget_lane_f16(result_f16, 0);
                            result_f16 = vext_f16(result_f16, result_f16, 1);
                        }
                    }
                }
            }
        });
}

void cactus_matmul_int4(const int8_t* A, const float* A_scales,
                        const int8_t* B_packed, const __fp16* B_scales,
                        __fp16* C, size_t M, size_t K, size_t N, size_t group_size) {
    if (M == 0 || K == 0 || N == 0) return;

    if (M == 1) {
        cactus_gemv_int4(A, A_scales[0], B_packed, B_scales, C, K, N, group_size);
    } else {
        cactus_gemm_int4(A, A_scales, B_packed, B_scales, C, M, K, N, group_size);
    }
}

void cactus_matmul_integer(Precision precision,
                            const int8_t* A, const float* A_scales,
                            const int8_t* B, const __fp16* B_scales,
                            __fp16* C, size_t M, size_t K, size_t N, size_t group_size) {
    if (precision == Precision::INT4) {
        cactus_matmul_int4(A, A_scales, B, B_scales, C, M, K, N, group_size);
    } else {
        cactus_matmul_int8(A, A_scales, B, B_scales, C, M, K, N, group_size);
    }
}
