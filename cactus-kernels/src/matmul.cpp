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

static inline bool cactus_quant_panel_major(const CactusQuantMatrix& W) {
    return (W.flags & CACTUS_QUANT_FLAG_PANEL_MAJOR) != 0;
}

static inline bool cactus_quant_code_ordered(const CactusQuantMatrix& W) {
    return (W.flags & CACTUS_QUANT_FLAG_CODE_ORDERED_INDICES) != 0;
}

static inline float16x8_t cactus_quant_signs_to_f16(const int8_t* signs, uint32_t offset) {
    if (signs == nullptr) return vdupq_n_f16(1);
    return vcvtq_f16_s16(vmovl_s8(vld1_s8(signs + offset)));
}

static inline float16x8_t cactus_quant_input_scale_recip8(const CactusQuantMatrix& W, uint32_t offset) {
    if (W.input_scale_recip != nullptr) {
        return vld1q_f16(W.input_scale_recip + offset);
    }
    if (W.input_scale != nullptr) {
        return vdivq_f16(vdupq_n_f16(1), vld1q_f16(W.input_scale + offset));
    }
    return vdupq_n_f16(1);
}

static inline __fp16 cactus_quant_input_scale_recip1(const CactusQuantMatrix& W, uint32_t offset) {
    if (W.input_scale_recip != nullptr) return W.input_scale_recip[offset];
    if (W.input_scale != nullptr) return static_cast<__fp16>(1.0f / static_cast<float>(W.input_scale[offset]));
    return static_cast<__fp16>(1);
}

static inline uint32_t cactus_quant_panel_chunks(const CactusQuantMatrix& W) {
    return W.group_size / kTQPanelKChunk;
}

static inline uint32_t cactus_quant_panel_chunk_bytes(uint32_t bits) {
    return (kTQPanelKChunk * bits) / 8;
}

static inline const __fp16* cactus_quant_scale_ptr(const CactusQuantMatrix& W, uint32_t row, uint32_t group) {
    if (!cactus_quant_panel_major(W)) {
        return W.norms + static_cast<size_t>(row) * W.num_groups + group;
    }

    const uint32_t lane = row & (kTQPanelN - 1);
    const uint32_t block = row / kTQPanelN;
    return W.norms + (((static_cast<size_t>(block) * W.num_groups + group) * kTQPanelN) + lane);
}

static inline const uint8_t* cactus_quant_packed_chunk_ptr(
    const CactusQuantMatrix& W,
    uint32_t row,
    uint32_t group,
    uint32_t k) {
    if (!cactus_quant_panel_major(W)) {
        return W.packed_indices
            + (((static_cast<size_t>(row) * W.num_groups + group)
                * cactus_quant_packed_group_bytes(W.bits, W.group_size))
               + (static_cast<size_t>(k) * W.bits) / 8);
    }

    const uint32_t lane = row & (kTQPanelN - 1);
    const uint32_t block = row / kTQPanelN;
    const uint32_t chunk = k / kTQPanelKChunk;
    const uint32_t intra = ((k % kTQPanelKChunk) * W.bits) / 8;
    const uint32_t chunk_bytes = cactus_quant_panel_chunk_bytes(W.bits);
    return W.packed_indices
        + ((((static_cast<size_t>(block) * W.num_groups + group)
             * cactus_quant_panel_chunks(W) + chunk)
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

static void cactus_quant_fwht128_f16(__fp16* x) {
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

static void cactus_quant_fwht_f16(__fp16* x, uint32_t n) {
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

static void cactus_quant_transform_hadamard_group(
    const CactusQuantMatrix& W,
    const __fp16* x_group,
    uint32_t group,
    __fp16* code_basis) {
    const uint32_t gs = W.group_size;
    __fp16 tmp[256];
    __fp16* work = (cactus_quant_code_ordered(W) || W.permutation == nullptr) ? code_basis : tmp;

    uint32_t k = 0;
    for (; k + 8 <= gs; k += 8) {
        const uint32_t offset = group * gs + k;
        float16x8_t x_v = vld1q_f16(x_group + k);
        x_v = vmulq_f16(x_v, cactus_quant_input_scale_recip8(W, offset));
        float16x8_t s_v = cactus_quant_signs_to_f16(W.left_signs, k);
        vst1q_f16(work + k, vmulq_f16(x_v, s_v));
    }
    for (; k < gs; ++k) {
        const uint32_t offset = group * gs + k;
        const float sign = W.left_signs ? static_cast<float>(W.left_signs[k]) : 1.0f;
        const float scale = static_cast<float>(cactus_quant_input_scale_recip1(W, offset));
        work[k] = static_cast<__fp16>(static_cast<float>(x_group[k]) * scale * sign);
    }

    if (gs == 128) {
        cactus_quant_fwht128_f16(work);
    } else {
        cactus_quant_fwht_f16(work, gs);
    }

    k = 0;
    for (; k + 8 <= gs; k += 8) {
        float16x8_t w_v = vld1q_f16(work + k);
        float16x8_t s_v = cactus_quant_signs_to_f16(W.right_signs, k);
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

static void cactus_quant_transform_hadamard_activations(
    const CactusQuantMatrix& W,
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
                cactus_quant_transform_hadamard_group(
                    W,
                    A + m * W.K + g * W.group_size,
                    static_cast<uint32_t>(g),
                    code_basis + m * W.K + g * W.group_size);
            }
        });
}

template<typename WorkFunc>
static void cactus_quant_parallel_ranges(size_t total_work, size_t work_per_thread, WorkFunc work_func) {
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

static void cactus_quant_matmul_f16_segment_accum(
    const __fp16* __restrict__ A,
    size_t a_stride,
    const __fp16* __restrict__ B_tile,
    __fp16* __restrict__ C,
    size_t M,
    size_t Kseg,
    size_t N,
    size_t n_start,
    size_t actual_n) {
    constexpr size_t TILE_M = 4;

    for (size_t m_start = 0; m_start < M; m_start += TILE_M) {
        const size_t actual_m = std::min(TILE_M, M - m_start);

        float16x8_t acc[TILE_M][16];
        for (size_t mi = 0; mi < actual_m; ++mi) {
            for (size_t ni = 0; ni < actual_n; ++ni) {
                acc[mi][ni] = vdupq_n_f16(0);
            }
        }

        for (size_t k = 0; k < Kseg; k += 16) {
            float16x8_t a_lo[TILE_M], a_hi[TILE_M];
            for (size_t mi = 0; mi < actual_m; ++mi) {
                const __fp16* ap = A + (m_start + mi) * a_stride + k;
                a_lo[mi] = vld1q_f16(ap);
                a_hi[mi] = vld1q_f16(ap + 8);
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

    explicit CactusTQ4ScaledDecoder(const CactusQuantMatrix& W) {
        cb_bytes.val[0] = vld1q_u8(reinterpret_cast<const uint8_t*>(W.codebook));
        cb_bytes.val[1] = vld1q_u8(reinterpret_cast<const uint8_t*>(W.codebook) + 16);
    }

    void operator()(const CactusQuantMatrix& W, uint32_t row, uint32_t group, __fp16* dst) const {
        const __fp16 rn = *cactus_quant_scale_ptr(W, row, group);
        const float16x8_t rn_v = vdupq_n_f16(rn);
        for (uint32_t k = 0; k < W.group_size; k += 16) {
            const uint8_t* packed = cactus_quant_packed_chunk_ptr(W, row, group, k);
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

    explicit CactusTQ2ScaledDecoder(const CactusQuantMatrix& W)
        : cb_bytes(vld1_u8(reinterpret_cast<const uint8_t*>(W.codebook))) {}

    void operator()(const CactusQuantMatrix& W, uint32_t row, uint32_t group, __fp16* dst) const {
        const __fp16 rn = *cactus_quant_scale_ptr(W, row, group);
        const float16x8_t rn_v = vdupq_n_f16(rn);
        for (uint32_t k = 0; k < W.group_size; k += 8) {
            const uint8_t* packed = cactus_quant_packed_chunk_ptr(W, row, group, k);
            uint8x8_t indices = cactus_tq2_unpack_8x2bit_le(packed[0], packed[1]);
            vst1q_f16(dst + k,
                      vmulq_f16(cactus_tq2_lookup_codebook8(indices, cb_bytes), rn_v));
        }
    }
};

template<uint32_t Bits, typename DecodeGroup>
static void cactus_quant_group_gemm(
    const CactusQuantMatrix& W,
    const __fp16* A,
    uint32_t M,
    __fp16* C,
    DecodeGroup decode_group) {
    static_assert(Bits >= 1 && Bits <= 4);
    if (W.bits != Bits) return;

    constexpr size_t TILE_N = 16;
    const size_t n_blocks = (W.N + TILE_N - 1) / TILE_N;


    CactusThreading::parallel_gemm_tiles(M, n_blocks,
        [&, decode_group](size_t block_start, size_t block_end) {
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
                    cactus_quant_matmul_f16_segment_accum(
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

static bool cactus_quant_valid_common(const CactusQuantMatrix* W, const void* A, void* C) {
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

uint32_t cactus_quant_packed_group_bytes(uint32_t bits, uint32_t group_size) {
    if (bits == 0 || bits > 4) return 0;
    return (group_size * bits + 7) / 8;
}

void cactus_quant_4bit_gemv(
    const CactusQuantMatrix* W,
    const __fp16* x,
    __fp16* y) {
    if (!cactus_quant_valid_common(W, x, y)) return;
    if (W->bits != 4 || (W->group_size % 16) != 0) return;

    constexpr size_t TILE_N = 12;
    const size_t n_blocks = (W->N + TILE_N - 1) / TILE_N;
    const uint32_t gs = W->group_size;

    uint8x16x2_t cb_bytes;
    cb_bytes.val[0] = vld1q_u8(reinterpret_cast<const uint8_t*>(W->codebook));
    cb_bytes.val[1] = vld1q_u8(reinterpret_cast<const uint8_t*>(W->codebook) + 16);

    const uint32_t pgb = cactus_quant_packed_group_bytes(4, gs);

    cactus_quant_parallel_ranges(n_blocks, 16, [&](size_t block_start, size_t block_end) {
        __fp16 z_buf[256];

        for (size_t block = block_start; block < block_end; ++block) {
            const size_t n_start = block * TILE_N;
            const size_t actual_n = std::min(TILE_N, static_cast<size_t>(W->N) - n_start);
            float acc[TILE_N] = {};

            for (uint32_t g = 0; g < W->num_groups; ++g) {

                cactus_quant_transform_hadamard_group(*W, x + g * gs, g, z_buf);
                const __fp16* z = z_buf;

                float16x8_t acc0[TILE_N];
                float16x8_t acc1[TILE_N];
                for (size_t ni = 0; ni < TILE_N; ++ni) {
                    acc0[ni] = vdupq_n_f16(0);
                    acc1[ni] = vdupq_n_f16(0);
                }

                for (uint32_t k = 0; k < gs; k += 16) {
                    float16x8_t z0 = vld1q_f16(z + k);
                    float16x8_t z1 = vld1q_f16(z + k + 8);
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        const uint8_t* p = W->packed_indices
                            + (static_cast<size_t>(n_start + ni) * W->num_groups + g) * pgb
                            + k / 2;
                        uint8x8_t bytes = vld1_u8(p);
                        uint8x8_t lo = vand_u8(bytes, vdup_n_u8(0x0F));
                        uint8x8_t hi = vshr_n_u8(bytes, 4);

                        float16x8_t cv0 = cactus_tq4_lookup_codebook8(vzip1_u8(lo, hi), cb_bytes);
                        float16x8_t cv1 = cactus_tq4_lookup_codebook8(vzip2_u8(lo, hi), cb_bytes);
                        acc0[ni] = vfmaq_f16(acc0[ni], z0, cv0);
                        acc1[ni] = vfmaq_f16(acc1[ni], z1, cv1);
                    }
                }

                for (size_t ni = 0; ni < actual_n; ++ni) {
                    float rn = static_cast<float>(W->norms[(n_start + ni) * W->num_groups + g]);
                    acc[ni] += rn *
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

void cactus_quant_2bit_gemv(
    const CactusQuantMatrix* W,
    const __fp16* x,
    __fp16* y) {
    if (!cactus_quant_valid_common(W, x, y)) return;
    if (W->bits != 2 || (W->group_size % 8) != 0) return;

    constexpr size_t TILE_N = 12;
    const size_t n_blocks = (W->N + TILE_N - 1) / TILE_N;
    const uint32_t gs = W->group_size;

    uint8x8_t cb_bytes = vld1_u8(reinterpret_cast<const uint8_t*>(W->codebook));

    const uint32_t pgb = cactus_quant_packed_group_bytes(2, gs);

    cactus_quant_parallel_ranges(n_blocks, 16, [&](size_t block_start, size_t block_end) {
        __fp16 z_buf[256];

        for (size_t block = block_start; block < block_end; ++block) {
            const size_t n_start = block * TILE_N;
            const size_t actual_n = std::min(TILE_N, static_cast<size_t>(W->N) - n_start);
            float acc[TILE_N] = {};

            for (uint32_t g = 0; g < W->num_groups; ++g) {
                cactus_quant_transform_hadamard_group(*W, x + g * gs, g, z_buf);
                const __fp16* z = z_buf;

                float16x8_t accv[TILE_N];
                for (size_t ni = 0; ni < TILE_N; ++ni) {
                    accv[ni] = vdupq_n_f16(0);
                }

                for (uint32_t k = 0; k < gs; k += 8) {
                    float16x8_t z_v = vld1q_f16(z + k);
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        const uint8_t* p = W->packed_indices
                            + (static_cast<size_t>(n_start + ni) * W->num_groups + g) * pgb
                            + k / 4;
                        uint8x8_t indices = cactus_tq2_unpack_8x2bit_le(p[0], p[1]);
                        float16x8_t cv = cactus_tq2_lookup_codebook8(indices, cb_bytes);
                        accv[ni] = vfmaq_f16(accv[ni], z_v, cv);
                    }
                }

                for (size_t ni = 0; ni < actual_n; ++ni) {
                    float rn = static_cast<float>(W->norms[(n_start + ni) * W->num_groups + g]);
                    acc[ni] += rn * static_cast<float>(hsum_f16x8(accv[ni]));
                }
            }

            for (size_t ni = 0; ni < actual_n; ++ni) {
                y[n_start + ni] = static_cast<__fp16>(acc[ni]);
            }
        }
    });
}

void cactus_quant_4bit_gemm(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C) {
    if (!cactus_quant_valid_common(W, A, C) || M == 0) return;
    if (W->bits != 4 || (W->group_size % 16) != 0) return;
    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < static_cast<size_t>(M) * W->K) {
        code_basis_buf.resize(static_cast<size_t>(M) * W->K);
    }
    cactus_quant_transform_hadamard_activations(*W, A, M, code_basis_buf.data());
    cactus_quant_group_gemm<4>(*W, code_basis_buf.data(), M, C, CactusTQ4ScaledDecoder(*W));
}

void cactus_quant_2bit_gemm(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C) {
    if (!cactus_quant_valid_common(W, A, C) || M == 0) return;
    if (W->bits != 2 || (W->group_size % 8) != 0) return;
    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < static_cast<size_t>(M) * W->K) {
        code_basis_buf.resize(static_cast<size_t>(M) * W->K);
    }
    cactus_quant_transform_hadamard_activations(*W, A, M, code_basis_buf.data());
    cactus_quant_group_gemm<2>(*W, code_basis_buf.data(), M, C, CactusTQ2ScaledDecoder(*W));
}



struct CactusTQ1ScaledDecoder {
    uint8x8_t cb_bytes;

    explicit CactusTQ1ScaledDecoder(const CactusQuantMatrix& W)
        : cb_bytes(vld1_u8(reinterpret_cast<const uint8_t*>(W.codebook))) {}

    void operator()(const CactusQuantMatrix& W, uint32_t row, uint32_t group, __fp16* dst) const {
        const __fp16 rn = *cactus_quant_scale_ptr(W, row, group);
        const float16x8_t rn_v = vdupq_n_f16(rn);
        for (uint32_t k = 0; k < W.group_size; k += 8) {
            const uint8_t* packed = cactus_quant_packed_chunk_ptr(W, row, group, k);

            uint8_t byte = packed[0];
            uint64_t idx_word =
                ((uint64_t)((byte >> 0) & 1)      ) |
                ((uint64_t)((byte >> 1) & 1) <<  8) |
                ((uint64_t)((byte >> 2) & 1) << 16) |
                ((uint64_t)((byte >> 3) & 1) << 24) |
                ((uint64_t)((byte >> 4) & 1) << 32) |
                ((uint64_t)((byte >> 5) & 1) << 40) |
                ((uint64_t)((byte >> 6) & 1) << 48) |
                ((uint64_t)((byte >> 7) & 1) << 56);
            uint8x8_t indices = vcreate_u8(idx_word);

            uint8x8_t off_lo = vshl_n_u8(indices, 1);
            uint8x8_t off_hi = vadd_u8(off_lo, vdup_n_u8(1));
            uint8x8x2_t zipped = vzip_u8(off_lo, off_hi);
            uint8x16_t byte_idx = vcombine_u8(zipped.val[0], zipped.val[1]);
            uint8x16_t lut = vcombine_u8(cb_bytes, cb_bytes);
            float16x8_t cv = vreinterpretq_f16_u8(vqtbl1q_u8(lut, byte_idx));
            vst1q_f16(dst + k, vmulq_f16(cv, rn_v));
        }
    }
};

void cactus_quant_1bit_gemv(
    const CactusQuantMatrix* W,
    const __fp16* x,
    __fp16* y) {
    if (!cactus_quant_valid_common(W, x, y)) return;
    if (W->bits != 1 || (W->group_size % 8) != 0) return;

    constexpr size_t TILE_N = 12;
    const size_t n_blocks = (W->N + TILE_N - 1) / TILE_N;
    const uint32_t gs = W->group_size;

    uint8x8_t cb_bytes = vld1_u8(reinterpret_cast<const uint8_t*>(W->codebook));
    uint8x16_t cb_lut = vcombine_u8(cb_bytes, cb_bytes);

    const uint32_t pgb = cactus_quant_packed_group_bytes(1, gs);

    cactus_quant_parallel_ranges(n_blocks, 16, [&](size_t block_start, size_t block_end) {
        __fp16 z_buf[256];

        for (size_t block = block_start; block < block_end; ++block) {
            const size_t n_start = block * TILE_N;
            const size_t actual_n = std::min(TILE_N, static_cast<size_t>(W->N) - n_start);
            float acc[TILE_N] = {};

            for (uint32_t g = 0; g < W->num_groups; ++g) {
                cactus_quant_transform_hadamard_group(*W, x + g * gs, g, z_buf);
                const __fp16* z = z_buf;

                float16x8_t accv[TILE_N];
                for (size_t ni = 0; ni < TILE_N; ++ni) accv[ni] = vdupq_n_f16(0);

                for (uint32_t k = 0; k < gs; k += 8) {
                    float16x8_t z_v = vld1q_f16(z + k);
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        const uint8_t* p = W->packed_indices
                            + (static_cast<size_t>(n_start + ni) * W->num_groups + g) * pgb
                            + k / 8;
                        uint8_t byte = p[0];
                        uint64_t idx_word =
                            ((uint64_t)((byte >> 0) & 1)      ) |
                            ((uint64_t)((byte >> 1) & 1) <<  8) |
                            ((uint64_t)((byte >> 2) & 1) << 16) |
                            ((uint64_t)((byte >> 3) & 1) << 24) |
                            ((uint64_t)((byte >> 4) & 1) << 32) |
                            ((uint64_t)((byte >> 5) & 1) << 40) |
                            ((uint64_t)((byte >> 6) & 1) << 48) |
                            ((uint64_t)((byte >> 7) & 1) << 56);
                        uint8x8_t indices = vcreate_u8(idx_word);
                        uint8x8_t off_lo = vshl_n_u8(indices, 1);
                        uint8x8_t off_hi = vadd_u8(off_lo, vdup_n_u8(1));
                        uint8x8x2_t zipped = vzip_u8(off_lo, off_hi);
                        uint8x16_t byte_idx = vcombine_u8(zipped.val[0], zipped.val[1]);
                        float16x8_t cv = vreinterpretq_f16_u8(vqtbl1q_u8(cb_lut, byte_idx));
                        accv[ni] = vfmaq_f16(accv[ni], z_v, cv);
                    }
                }

                for (size_t ni = 0; ni < actual_n; ++ni) {
                    float rn = static_cast<float>(W->norms[(n_start + ni) * W->num_groups + g]);
                    acc[ni] += rn * static_cast<float>(hsum_f16x8(accv[ni]));
                }
            }

            for (size_t ni = 0; ni < actual_n; ++ni) {
                y[n_start + ni] = static_cast<__fp16>(acc[ni]);
            }
        }
    });
}

void cactus_quant_1bit_gemm(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C) {
    if (!cactus_quant_valid_common(W, A, C) || M == 0) return;
    if (W->bits != 1 || (W->group_size % 8) != 0) return;
    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < static_cast<size_t>(M) * W->K) {
        code_basis_buf.resize(static_cast<size_t>(M) * W->K);
    }
    cactus_quant_transform_hadamard_activations(*W, A, M, code_basis_buf.data());
    cactus_quant_group_gemm<1>(*W, code_basis_buf.data(), M, C, CactusTQ1ScaledDecoder(*W));
}



static inline uint8x8_t cactus_tq3_unpack_8x3bit(const uint8_t* packed) {

    uint32_t b0 = packed[0], b1 = packed[1], b2 = packed[2];
    uint32_t word = b0 | (b1 << 8) | (b2 << 16);
    uint64_t idx_word =
        ((uint64_t)((word >> 0)  & 0x7)      ) |
        ((uint64_t)((word >> 3)  & 0x7) <<  8) |
        ((uint64_t)((word >> 6)  & 0x7) << 16) |
        ((uint64_t)((word >> 9)  & 0x7) << 24) |
        ((uint64_t)((word >> 12) & 0x7) << 32) |
        ((uint64_t)((word >> 15) & 0x7) << 40) |
        ((uint64_t)((word >> 18) & 0x7) << 48) |
        ((uint64_t)((word >> 21) & 0x7) << 56);
    return vcreate_u8(idx_word);
}

static inline float16x8_t cactus_tq3_lookup_codebook8(uint8x8_t indices, uint8x16_t cb_bytes) {
    uint8x8_t off_lo = vshl_n_u8(indices, 1);
    uint8x8_t off_hi = vadd_u8(off_lo, vdup_n_u8(1));
    uint8x8x2_t zipped = vzip_u8(off_lo, off_hi);
    uint8x16_t byte_idx = vcombine_u8(zipped.val[0], zipped.val[1]);
    return vreinterpretq_f16_u8(vqtbl1q_u8(cb_bytes, byte_idx));
}

struct CactusTQ3ScaledDecoder {
    uint8x16_t cb_bytes;

    explicit CactusTQ3ScaledDecoder(const CactusQuantMatrix& W)
        : cb_bytes(vld1q_u8(reinterpret_cast<const uint8_t*>(W.codebook))) {}

    void operator()(const CactusQuantMatrix& W, uint32_t row, uint32_t group, __fp16* dst) const {
        const __fp16 rn = *cactus_quant_scale_ptr(W, row, group);
        const float16x8_t rn_v = vdupq_n_f16(rn);
        for (uint32_t k = 0; k < W.group_size; k += 8) {
            const uint8_t* packed = cactus_quant_packed_chunk_ptr(W, row, group, k);
            uint8x8_t indices = cactus_tq3_unpack_8x3bit(packed);
            float16x8_t cv = cactus_tq3_lookup_codebook8(indices, cb_bytes);
            vst1q_f16(dst + k, vmulq_f16(cv, rn_v));
        }
    }
};

void cactus_quant_3bit_gemv(
    const CactusQuantMatrix* W,
    const __fp16* x,
    __fp16* y) {
    if (!cactus_quant_valid_common(W, x, y)) return;
    if (W->bits != 3 || (W->group_size % 8) != 0) return;

    constexpr size_t TILE_N = 12;
    const size_t n_blocks = (W->N + TILE_N - 1) / TILE_N;
    const uint32_t gs = W->group_size;

    uint8x16_t cb_bytes = vld1q_u8(reinterpret_cast<const uint8_t*>(W->codebook));

    const uint32_t pgb = cactus_quant_packed_group_bytes(3, gs);

    cactus_quant_parallel_ranges(n_blocks, 16, [&](size_t block_start, size_t block_end) {
        __fp16 z_buf[256];

        for (size_t block = block_start; block < block_end; ++block) {
            const size_t n_start = block * TILE_N;
            const size_t actual_n = std::min(TILE_N, static_cast<size_t>(W->N) - n_start);
            float acc[TILE_N] = {};

            for (uint32_t g = 0; g < W->num_groups; ++g) {
                cactus_quant_transform_hadamard_group(*W, x + g * gs, g, z_buf);
                const __fp16* z = z_buf;

                float16x8_t accv[TILE_N];
                for (size_t ni = 0; ni < TILE_N; ++ni) accv[ni] = vdupq_n_f16(0);

                for (uint32_t k = 0; k < gs; k += 8) {
                    float16x8_t z_v = vld1q_f16(z + k);
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        const uint8_t* p = W->packed_indices
                            + (static_cast<size_t>(n_start + ni) * W->num_groups + g) * pgb
                            + (k * 3) / 8;
                        uint8x8_t indices = cactus_tq3_unpack_8x3bit(p);
                        float16x8_t cv = cactus_tq3_lookup_codebook8(indices, cb_bytes);
                        accv[ni] = vfmaq_f16(accv[ni], z_v, cv);
                    }
                }

                for (size_t ni = 0; ni < actual_n; ++ni) {
                    float rn = static_cast<float>(W->norms[(n_start + ni) * W->num_groups + g]);
                    acc[ni] += rn * static_cast<float>(hsum_f16x8(accv[ni]));
                }
            }

            for (size_t ni = 0; ni < actual_n; ++ni) {
                y[n_start + ni] = static_cast<__fp16>(acc[ni]);
            }
        }
    });
}

void cactus_quant_3bit_gemm(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C) {
    if (!cactus_quant_valid_common(W, A, C) || M == 0) return;
    if (W->bits != 3 || (W->group_size % 8) != 0) return;
    thread_local std::vector<__fp16> code_basis_buf;
    if (code_basis_buf.size() < static_cast<size_t>(M) * W->K) {
        code_basis_buf.resize(static_cast<size_t>(M) * W->K);
    }
    cactus_quant_transform_hadamard_activations(*W, A, M, code_basis_buf.data());
    cactus_quant_group_gemm<3>(*W, code_basis_buf.data(), M, C, CactusTQ3ScaledDecoder(*W));
}

static inline float tq_quantize_codebook_i8(const __fp16* codebook, int8_t* cb_i8, uint32_t cb_size) {
    float max_abs = 0.f;
    for (uint32_t i = 0; i < cb_size; i++) {
        float v = std::abs(static_cast<float>(codebook[i]));
        if (v > max_abs) max_abs = v;
    }
    float scale = max_abs / 127.f;
    if (scale < 1e-10f) scale = 1e-10f;
    float inv = 1.f / scale;
    for (uint32_t i = 0; i < cb_size; i++)
        cb_i8[i] = static_cast<int8_t>(std::round(static_cast<float>(codebook[i]) * inv));
    return scale;
}

static inline float tq_quantize_group_i8(const __fp16* src, int8_t* dst, uint32_t gs) {
    float32x4_t max_vec = vdupq_n_f32(0.f);
    uint32_t k = 0;
    for (; k + 8 <= gs; k += 8) {
        float16x8_t v = vld1q_f16(src + k);
        max_vec = vmaxq_f32(max_vec, vabsq_f32(vcvt_f32_f16(vget_low_f16(v))));
        max_vec = vmaxq_f32(max_vec, vabsq_f32(vcvt_f32_f16(vget_high_f16(v))));
    }
    float max_abs = vmaxvq_f32(max_vec);
    for (; k < gs; k++) {
        float v = std::abs(static_cast<float>(src[k]));
        if (v > max_abs) max_abs = v;
    }
    float scale = max_abs / 127.f;
    if (scale < 1e-10f) scale = 1e-10f;
    float32x4_t inv_vec = vdupq_n_f32(1.f / scale);
    k = 0;
    for (; k + 8 <= gs; k += 8) {
        float16x8_t v = vld1q_f16(src + k);
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(v)), inv_vec));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(v)), inv_vec));
        vst1_s8(dst + k, vqmovn_s16(vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1))));
    }
    for (; k < gs; k++)
        dst[k] = static_cast<int8_t>(std::round(static_cast<float>(src[k]) / scale));
    return scale;
}


static inline int8x16_t tq_expand_i8_16(const uint8_t* packed, uint32_t bits, int8x16_t cb_lut) {
    if (bits == 4) {
        uint8x8_t bytes = vld1_u8(packed);
        uint8x8_t lo = vand_u8(bytes, vdup_n_u8(0x0F));
        uint8x8_t hi = vshr_n_u8(bytes, 4);
        uint8x16_t idx = vcombine_u8(vzip1_u8(lo, hi), vzip2_u8(lo, hi));
        return vqtbl1q_s8(cb_lut, idx);
    } else if (bits == 2) {
        uint8_t b0 = packed[0], b1 = packed[1], b2 = packed[2], b3 = packed[3];
        uint64_t lo_w =
            ((uint64_t)(b0&3)) | ((uint64_t)((b0>>2)&3)<<8) |
            ((uint64_t)((b0>>4)&3)<<16) | ((uint64_t)((b0>>6)&3)<<24) |
            ((uint64_t)(b1&3)<<32) | ((uint64_t)((b1>>2)&3)<<40) |
            ((uint64_t)((b1>>4)&3)<<48) | ((uint64_t)((b1>>6)&3)<<56);
        uint64_t hi_w =
            ((uint64_t)(b2&3)) | ((uint64_t)((b2>>2)&3)<<8) |
            ((uint64_t)((b2>>4)&3)<<16) | ((uint64_t)((b2>>6)&3)<<24) |
            ((uint64_t)(b3&3)<<32) | ((uint64_t)((b3>>2)&3)<<40) |
            ((uint64_t)((b3>>4)&3)<<48) | ((uint64_t)((b3>>6)&3)<<56);
        return vqtbl1q_s8(cb_lut, vcombine_u8(vcreate_u8(lo_w), vcreate_u8(hi_w)));
    } else if (bits == 1) {
        uint8_t b0 = packed[0], b1 = packed[1];
        uint64_t lo_w =
            ((uint64_t)((b0>>0)&1)) | ((uint64_t)((b0>>1)&1)<<8) |
            ((uint64_t)((b0>>2)&1)<<16) | ((uint64_t)((b0>>3)&1)<<24) |
            ((uint64_t)((b0>>4)&1)<<32) | ((uint64_t)((b0>>5)&1)<<40) |
            ((uint64_t)((b0>>6)&1)<<48) | ((uint64_t)((b0>>7)&1)<<56);
        uint64_t hi_w =
            ((uint64_t)((b1>>0)&1)) | ((uint64_t)((b1>>1)&1)<<8) |
            ((uint64_t)((b1>>2)&1)<<16) | ((uint64_t)((b1>>3)&1)<<24) |
            ((uint64_t)((b1>>4)&1)<<32) | ((uint64_t)((b1>>5)&1)<<40) |
            ((uint64_t)((b1>>6)&1)<<48) | ((uint64_t)((b1>>7)&1)<<56);
        return vqtbl1q_s8(cb_lut, vcombine_u8(vcreate_u8(lo_w), vcreate_u8(hi_w)));
    } else { // bits == 3
        uint64_t raw = 0;
        std::memcpy(&raw, packed, 6);
        uint64_t lo_w = 0, hi_w = 0;
        for (int i = 0; i < 8; i++) lo_w |= ((raw >> (i*3)) & 7ULL) << (i*8);
        for (int i = 0; i < 8; i++) hi_w |= ((raw >> ((i+8)*3)) & 7ULL) << (i*8);
        return vqtbl1q_s8(cb_lut, vcombine_u8(vcreate_u8(lo_w), vcreate_u8(hi_w)));
    }
}

void cactus_quant_matmul(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C) {
    if (!cactus_quant_valid_common(W, A, C) || M == 0) return;

    const uint32_t gs = W->group_size;
    const uint32_t bits = W->bits;
    const uint32_t num_groups = W->num_groups;
    const uint32_t pgb = cactus_quant_packed_group_bytes(bits, gs);

    int8_t cb_i8[16] = {};
    float cb_scale = tq_quantize_codebook_i8(W->codebook, cb_i8, 1u << bits);
    int8x16_t cb_lut = vld1q_s8(cb_i8);

    const size_t act_size = static_cast<size_t>(M) * W->K;
    const size_t scales_size = static_cast<size_t>(M) * num_groups;

    static thread_local std::vector<__fp16> tl_code_basis;
    static thread_local std::vector<int8_t> tl_act_i8;
    static thread_local std::vector<float> tl_act_scales;

    __fp16* code_basis_ptr;
    int8_t* act_i8_ptr;
    float* act_scales_ptr;

    std::vector<__fp16> heap_code_basis;
    std::vector<int8_t> heap_act_i8;
    std::vector<float> heap_act_scales;

    if (M == 1) {
        if (tl_code_basis.size() < act_size) tl_code_basis.resize(act_size);
        if (tl_act_i8.size() < act_size) tl_act_i8.resize(act_size);
        if (tl_act_scales.size() < scales_size) tl_act_scales.resize(scales_size);
        code_basis_ptr = tl_code_basis.data();
        act_i8_ptr = tl_act_i8.data();
        act_scales_ptr = tl_act_scales.data();
    } else {
        heap_code_basis.resize(act_size);
        heap_act_i8.resize(act_size);
        heap_act_scales.resize(scales_size);
        code_basis_ptr = heap_code_basis.data();
        act_i8_ptr = heap_act_i8.data();
        act_scales_ptr = heap_act_scales.data();
    }

    cactus_quant_transform_hadamard_activations(*W, A, M, code_basis_ptr);

    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t g = 0; g < num_groups; g++) {
            act_scales_ptr[m * num_groups + g] = tq_quantize_group_i8(
                code_basis_ptr + m * W->K + g * gs,
                act_i8_ptr + m * W->K + g * gs, gs);
        }
    }

    const size_t N_blocks = (W->N + 3) / 4;

    const int8_t* w_il = W->expanded;
    const float* n_f32 = W->norm_f32;

    std::vector<int8_t> w_il_buf;
    std::vector<float> n_f32_buf;
    if (!w_il) {
        w_il_buf.resize(N_blocks * num_groups * gs * 4);
        n_f32_buf.resize(N_blocks * num_groups * 4);
        for (size_t nb = 0; nb < N_blocks; ++nb) {
            size_t n_start = nb * 4;
            size_t valid_n = std::min(size_t(4), static_cast<size_t>(W->N) - n_start);
            for (uint32_t g = 0; g < num_groups; ++g) {
                int8x16_t exp4[4][16];
                uint32_t n_vecs = gs / 16;
                for (size_t ni = 0; ni < valid_n; ++ni) {
                    const uint8_t* p = W->packed_indices + (static_cast<size_t>(n_start+ni)*num_groups+g)*pgb;
                    for (uint32_t v = 0; v < n_vecs; ++v)
                        exp4[ni][v] = tq_expand_i8_16(p + (v*16*bits)/8, bits, cb_lut);
                }
                for (size_t ni = valid_n; ni < 4; ++ni)
                    for (uint32_t v = 0; v < n_vecs; ++v) exp4[ni][v] = vdupq_n_s8(0);

                int8_t* dst = w_il_buf.data() + (nb*num_groups+g)*gs*4;
                for (uint32_t v = 0; v < n_vecs; ++v) {
                    int32x4_t r0=vreinterpretq_s32_s8(exp4[0][v]), r1=vreinterpretq_s32_s8(exp4[1][v]);
                    int32x4_t r2=vreinterpretq_s32_s8(exp4[2][v]), r3=vreinterpretq_s32_s8(exp4[3][v]);
                    int32x4_t t01l=vzip1q_s32(r0,r1), t01h=vzip2q_s32(r0,r1);
                    int32x4_t t23l=vzip1q_s32(r2,r3), t23h=vzip2q_s32(r2,r3);
                    vst1q_s8(dst+v*64,    vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s32(t01l),vreinterpretq_s64_s32(t23l))));
                    vst1q_s8(dst+v*64+16, vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s32(t01l),vreinterpretq_s64_s32(t23l))));
                    vst1q_s8(dst+v*64+32, vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s32(t01h),vreinterpretq_s64_s32(t23h))));
                    vst1q_s8(dst+v*64+48, vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s32(t01h),vreinterpretq_s64_s32(t23h))));
                }
                float* nd = n_f32_buf.data() + (nb*num_groups+g)*4;
                for (size_t ni = 0; ni < 4; ++ni)
                    nd[ni] = (n_start+ni < W->N) ? static_cast<float>(W->norms[(n_start+ni)*num_groups+g]) * cb_scale : 0.f;
            }
        }
        w_il = w_il_buf.data();
        n_f32 = n_f32_buf.data();
    }

    if (M == 1) {
        const int8_t* a_ptr = act_i8_ptr;
        const float* a_sc_ptr = act_scales_ptr;

        auto& pool = CactusThreading::get_thread_pool();
        size_t num_threads = CactusThreading::GemmThreading::get_gemv_threads(N_blocks, pool.num_workers());
        num_threads = std::min(num_threads, N_blocks);

        auto process_blocks = [&](size_t block_start, size_t block_end) {
            for (size_t n_block = block_start; n_block < block_end; ++n_block) {
                float32x4_t running_sum = vdupq_n_f32(0.f);
                const size_t nb_off = n_block * num_groups;

                uint32_t g = 0;
                for (; g + 1 < num_groups; g += 2) {
                    const int8_t* a0 = a_ptr + g * gs;
                    const int8_t* a1 = a_ptr + (g + 1) * gs;
                    const int8_t* b0 = w_il + (nb_off + g) * gs * 4;
                    const int8_t* b1 = w_il + (nb_off + g + 1) * gs * 4;

                    __builtin_prefetch(b1 + gs * 4, 0, 3);

                    int32x4_t acc0 = vdupq_n_s32(0);
                    int32x4_t acc1 = vdupq_n_s32(0);

                    for (uint32_t k = 0; k < gs; k += 32) {
                        int8x16_t av = vld1q_s8(a0 + k);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4),      av, 0);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4 + 16), av, 1);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4 + 32), av, 2);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4 + 48), av, 3);
                        av = vld1q_s8(a0 + k + 16);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4 + 64), av, 0);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4 + 80), av, 1);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4 + 96), av, 2);
                        acc0 = CACTUS_DOTQ_LANE(acc0, vld1q_s8(b0 + k*4 + 112),av, 3);
                    }


                    for (uint32_t k = 0; k < gs; k += 32) {
                        int8x16_t av = vld1q_s8(a1 + k);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4),      av, 0);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4 + 16), av, 1);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4 + 32), av, 2);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4 + 48), av, 3);
                        av = vld1q_s8(a1 + k + 16);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4 + 64), av, 0);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4 + 80), av, 1);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4 + 96), av, 2);
                        acc1 = CACTUS_DOTQ_LANE(acc1, vld1q_s8(b1 + k*4 + 112),av, 3);
                    }

                    float32x4_t s0 = vmulq_n_f32(vld1q_f32(n_f32 + (nb_off + g) * 4), a_sc_ptr[g]);
                    float32x4_t s1 = vmulq_n_f32(vld1q_f32(n_f32 + (nb_off + g + 1) * 4), a_sc_ptr[g + 1]);
                    running_sum = vmlaq_f32(running_sum, vcvtq_f32_s32(acc0), s0);
                    running_sum = vmlaq_f32(running_sum, vcvtq_f32_s32(acc1), s1);
                }

                for (; g < num_groups; ++g) {
                    const int8_t* a_group = a_ptr + g * gs;
                    const int8_t* b_base = w_il + (nb_off + g) * gs * 4;

                    int32x4_t acc = vdupq_n_s32(0);
                    for (uint32_t k = 0; k < gs; k += 32) {
                        int8x16_t av = vld1q_s8(a_group + k);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4),      av, 0);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4 + 16), av, 1);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4 + 32), av, 2);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4 + 48), av, 3);
                        av = vld1q_s8(a_group + k + 16);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4 + 64), av, 0);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4 + 80), av, 1);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4 + 96), av, 2);
                        acc = CACTUS_DOTQ_LANE(acc, vld1q_s8(b_base + k*4 + 112),av, 3);
                    }

                    float32x4_t scale = vmulq_n_f32(vld1q_f32(n_f32 + (nb_off + g) * 4), a_sc_ptr[g]);
                    running_sum = vmlaq_f32(running_sum, vcvtq_f32_s32(acc), scale);
                }

                const size_t n_start = n_block * 4;
                const size_t actual_n = std::min(size_t(4), static_cast<size_t>(W->N) - n_start);
                float16x4_t result = vcvt_f16_f32(running_sum);
                if (actual_n == 4) {
                    vst1_f16(C + n_start, result);
                } else {
                    for (size_t ni = 0; ni < actual_n; ni++) {
                        C[n_start + ni] = vget_lane_f16(result, 0);
                        result = vext_f16(result, result, 1);
                    }
                }
            }
        };

        if (num_threads <= 1) {
            process_blocks(0, N_blocks);
        } else {
            pool.enqueue_n_threads(N_blocks, num_threads, process_blocks);
            pool.wait_all();
        }

    } else {

        constexpr size_t TILE_M = 8;
        const size_t M_blocks = (M + TILE_M - 1) / TILE_M;
        const size_t total_tiles = M_blocks * N_blocks;

        CactusThreading::parallel_gemm_tiles(M, total_tiles,
            [&](size_t tile_start, size_t tile_end) {
                for (size_t tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
                    const size_t m_block = tile_idx / N_blocks;
                    const size_t n_block = tile_idx % N_blocks;
                    const size_t m_start = m_block * TILE_M;
                    const size_t m_end = std::min(m_start + TILE_M, static_cast<size_t>(M));
                    const size_t actual_m = m_end - m_start;
                    const size_t n_start = n_block * 4;
                    const size_t actual_n = std::min(size_t(4), static_cast<size_t>(W->N) - n_start);

                    const int8_t* a_rows[TILE_M];
                    for (size_t mi = 0; mi < TILE_M; mi++) {
                        size_t row = m_start + (mi < actual_m ? mi : actual_m - 1);
                        a_rows[mi] = act_i8_ptr + row * W->K;
                    }

                    float32x4_t running_sum[TILE_M];
                    for (size_t mi = 0; mi < TILE_M; mi++)
                        running_sum[mi] = vdupq_n_f32(0.f);

                    for (uint32_t g = 0; g < num_groups; ++g) {
                        const int8_t* b_base = w_il + (n_block * num_groups + g) * gs * 4;
                        float32x4_t norms_v = vld1q_f32(n_f32 + (n_block * num_groups + g) * 4);

                        __builtin_prefetch(b_base + gs * 4, 0, 3);

                        int32x4_t row_acc[TILE_M];
                        for (size_t mi = 0; mi < TILE_M; mi++) row_acc[mi] = vdupq_n_s32(0);

                        for (uint32_t k = 0; k < gs; k += 32) {
                            const int8_t* bk = b_base + k * 4;
                            int8x16_t b00 = vld1q_s8(bk);
                            int8x16_t b01 = vld1q_s8(bk + 16);
                            int8x16_t b02 = vld1q_s8(bk + 32);
                            int8x16_t b03 = vld1q_s8(bk + 48);
                            int8x16_t b10 = vld1q_s8(bk + 64);
                            int8x16_t b11 = vld1q_s8(bk + 80);
                            int8x16_t b12 = vld1q_s8(bk + 96);
                            int8x16_t b13 = vld1q_s8(bk + 112);

                            #define TQ_GEMM_ROW(ROW) do { \
                                const int8_t* ap = a_rows[ROW] + g * gs + k; \
                                int8x16_t a_lo = vld1q_s8(ap); \
                                row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b00, a_lo, 0); \
                                row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b01, a_lo, 1); \
                                row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b02, a_lo, 2); \
                                row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b03, a_lo, 3); \
                                int8x16_t a_hi = vld1q_s8(ap + 16); \
                                row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b10, a_hi, 0); \
                                row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b11, a_hi, 1); \
                                row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b12, a_hi, 2); \
                                row_acc[ROW] = CACTUS_DOTQ_LANE(row_acc[ROW], b13, a_hi, 3); \
                            } while(0)

                            TQ_GEMM_ROW(0);
                            TQ_GEMM_ROW(1);
                            TQ_GEMM_ROW(2);
                            TQ_GEMM_ROW(3);
                            TQ_GEMM_ROW(4);
                            TQ_GEMM_ROW(5);
                            TQ_GEMM_ROW(6);
                            TQ_GEMM_ROW(7);
                            #undef TQ_GEMM_ROW
                        }

                        for (size_t mi = 0; mi < actual_m; mi++) {
                            float a_sc = act_scales_ptr[(m_start + mi) * num_groups + g];
                            running_sum[mi] = vmlaq_f32(running_sum[mi],
                                vcvtq_f32_s32(row_acc[mi]), vmulq_n_f32(norms_v, a_sc));
                        }
                    }

                    for (size_t mi = 0; mi < actual_m; ++mi) {
                        float16x4_t r = vcvt_f16_f32(running_sum[mi]);
                        if (actual_n == 4) {
                            vst1_f16(C + (m_start + mi) * W->N + n_start, r);
                        } else {
                            for (size_t ni = 0; ni < actual_n; ni++) {
                                C[(m_start + mi) * W->N + n_start + ni] = vget_lane_f16(r, 0);
                                r = vext_f16(r, r, 1);
                            }
                        }
                    }
                }
            });
    }
}

