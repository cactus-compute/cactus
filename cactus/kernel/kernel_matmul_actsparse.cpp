#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

// Activation-sparse INT4 × INT8 GEMV kernels.
//
// Contract
// --------
// Callers zero dropped lanes of A *before* calling any of these kernels —
// so A_masked[k] == 0 for every lane k that the mask drops. The kernels
// therefore only need to decide which whole K-groups to skip (to save
// weight bandwidth); lanes inside a live K-group still participate via
// the already-zeroed entries in A_masked (they contribute zero to the
// integer dot product). This gives bit-identical integer accumulation to
// the corresponding dense call on A_masked; any FP divergence from the
// dense baseline is only from dropping-per-group FP scale accumulations
// (one mul+add per dropped group instead of a no-op).
//
// Baseline-layout kernels (azero, bitmask, livelist, livelist_pf, mask_2nb)
// accept B in the exact layout of `cactus_gemv_int4`: per (n_block, g),
// 64 bytes of packed int4 plus a separate `B_scales` array.
//
// K-major-layout kernels (kmajor, kmi, kmi2, kmi4, kmi4_fast, kmi4_v2)
// accept a one-time repack of B that does NOT depend on S. The repack
// `cactus_repack_int4_kmajor[_inline]` runs at matrix load time and turns
// the baseline layout into a (K-group, n_block)-major stream so that each
// live group's bytes are contiguous — the access pattern act-sparsity
// actually wants. This is permitted by the spec because the repacked
// layout is a function of B alone, not of the per-token score vector S.

#if defined(__ARM_FEATURE_DOTPROD)
    #define AS_DOTQ_LANE(acc, b, a, lane) vdotq_laneq_s32(acc, b, a, lane)
#else
    static inline int32x4_t as_dotq_with_pattern(int32x4_t acc, int8x16_t b, int8x8_t a_pattern) {
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
    static inline int32x4_t as_dotq_lane0(int32x4_t a, int8x16_t b, int8x16_t v) {
        return as_dotq_with_pattern(a, b, vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vget_low_s8(v)), 0)));
    }
    static inline int32x4_t as_dotq_lane1(int32x4_t a, int8x16_t b, int8x16_t v) {
        return as_dotq_with_pattern(a, b, vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vget_low_s8(v)), 1)));
    }
    static inline int32x4_t as_dotq_lane2(int32x4_t a, int8x16_t b, int8x16_t v) {
        return as_dotq_with_pattern(a, b, vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vget_high_s8(v)), 0)));
    }
    static inline int32x4_t as_dotq_lane3(int32x4_t a, int8x16_t b, int8x16_t v) {
        return as_dotq_with_pattern(a, b, vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vget_high_s8(v)), 1)));
    }
    #define AS_DOTQ_LANE(acc, b, a, lane) as_dotq_lane##lane(acc, b, a)
#endif

static inline void as_unpack_nibbles(const uint8_t* ptr, int8x16_t& high, int8x16_t& low) {
    int8x16_t packed = vreinterpretq_s8_u8(vld1q_u8(ptr));
    high = vshrq_n_s8(packed, 4);
    low = vshrq_n_s8(vshlq_n_s8(packed, 4), 4);
}

// Dot a 64-byte packed tile at b_base against (a_lo, a_hi) into a single
// int32x4 accumulator (one entry per row of the 4-row n_block).
static inline int32x4_t as_dot_group_one_row(const uint8_t* b_base, int8x16_t a_lo, int8x16_t a_hi) {
    int32x4_t acc = vdupq_n_s32(0);
    int8x16_t b0, b1, b2, b3;
    as_unpack_nibbles(b_base,      b1, b0);
    as_unpack_nibbles(b_base + 16, b3, b2);
    acc = AS_DOTQ_LANE(acc, b0, a_lo, 0);
    acc = AS_DOTQ_LANE(acc, b1, a_lo, 1);
    acc = AS_DOTQ_LANE(acc, b2, a_lo, 2);
    acc = AS_DOTQ_LANE(acc, b3, a_lo, 3);
    as_unpack_nibbles(b_base + 32, b1, b0);
    as_unpack_nibbles(b_base + 48, b3, b2);
    acc = AS_DOTQ_LANE(acc, b0, a_hi, 0);
    acc = AS_DOTQ_LANE(acc, b1, a_hi, 1);
    acc = AS_DOTQ_LANE(acc, b2, a_hi, 2);
    acc = AS_DOTQ_LANE(acc, b3, a_hi, 3);
    return acc;
}

// ---------------------------------------------------------------------------
// Kernel I1 — A-zero + dense (baseline with pre-zeroed A).
// Reads every group of B; the only saving is that zero A lanes contribute
// nothing to the integer dot product. Purpose: establish the "compute
// savings floor" when B bandwidth cannot be avoided.
// ---------------------------------------------------------------------------
void cactus_gemv_int4_actsparse_azero(
    const int8_t* A_masked, float A_scale,
    const int8_t* B_packed_raw, const __fp16* B_scales,
    __fp16* C, size_t K, size_t N, size_t group_size)
{
    cactus_gemv_int4(A_masked, A_scale, B_packed_raw, B_scales, C, K, N, group_size);
}

// ---------------------------------------------------------------------------
// Kernel I2 — bitmask over K-groups.
// Given a uint64_t bitmask (one bit per K-group), skip any group whose bit
// is zero. A_masked already has dropped lanes zeroed so live groups use the
// unchanged dense micro-kernel.
// ---------------------------------------------------------------------------
void cactus_gemv_int4_actsparse_bitmask(
    const int8_t* A_masked, float A_scale,
    const int8_t* B_packed_raw, const __fp16* B_scales,
    const uint64_t* live_group_bitmask,
    __fp16* C, size_t K, size_t N, size_t group_size)
{
    const uint8_t* B_packed = reinterpret_cast<const uint8_t*>(B_packed_raw);
    if (K == 0 || N == 0) return;

    const size_t num_groups = K / group_size;
    const size_t N_blocks = (N + 3) / 4;

    auto process_blocks = [=](size_t block_start, size_t block_end) {
        for (size_t n_block = block_start; n_block < block_end; ++n_block) {
            const size_t n_start = n_block * 4;
            const size_t actual_n = std::min(size_t(4), N - n_start);
            float32x4_t running_sum = vdupq_n_f32(0.0f);

            const uint8_t* b_row = B_packed + n_block * K * 2;
            const __fp16* scale_row = B_scales + n_block * num_groups * 4;

            // Walk groups by set bits of the mask: O(popcount) not O(num_groups).
            for (size_t w = 0; w * 64 < num_groups; ++w) {
                uint64_t bits = live_group_bitmask[w];
                const size_t base_g = w * 64;
                const size_t end_g  = std::min(base_g + 64, num_groups);
                const size_t local_end = end_g - base_g;
                if (local_end < 64) {
                    bits &= (local_end == 64) ? ~uint64_t(0) : ((uint64_t(1) << local_end) - 1);
                }
                while (bits) {
                    const unsigned b_idx = __builtin_ctzll(bits);
                    bits &= bits - 1;
                    const size_t g = base_g + b_idx;
                    const size_t k_base = g * group_size;
                    const uint8_t* b_base = b_row + k_base * 2;
                    __builtin_prefetch(b_base + 256, 0, 0);

                    int8x16_t a_lo = vld1q_s8(A_masked + k_base);
                    int8x16_t a_hi = vld1q_s8(A_masked + k_base + 16);
                    int32x4_t acc = as_dot_group_one_row(b_base, a_lo, a_hi);
                    float32x4_t scales = vcvt_f32_f16(vld1_f16(scale_row + g * 4));
                    running_sum = vmlaq_f32(running_sum, vcvtq_f32_s32(acc), scales);
                }
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

// ---------------------------------------------------------------------------
// Kernel I3 — live-group index list.
// A compact uint16_t[] of live group indices, shared across all rows.
// Same inner kernel as I2; driver differs — no bitmask walking, just a
// tight loop over the short index list. Expect small constant-factor diffs
// vs I2 depending on pattern density.
// ---------------------------------------------------------------------------
void cactus_gemv_int4_actsparse_livelist(
    const int8_t* A_masked, float A_scale,
    const int8_t* B_packed_raw, const __fp16* B_scales,
    const uint16_t* live_groups, size_t num_live,
    __fp16* C, size_t K, size_t N, size_t group_size)
{
    const uint8_t* B_packed = reinterpret_cast<const uint8_t*>(B_packed_raw);
    if (K == 0 || N == 0 || num_live == 0) {
        if (N && C) std::memset(C, 0, N * sizeof(__fp16));
        return;
    }

    const size_t num_groups = K / group_size;
    const size_t N_blocks = (N + 3) / 4;

    auto process_blocks = [=](size_t block_start, size_t block_end) {
        for (size_t n_block = block_start; n_block < block_end; ++n_block) {
            const size_t n_start = n_block * 4;
            const size_t actual_n = std::min(size_t(4), N - n_start);
            float32x4_t running_sum = vdupq_n_f32(0.0f);

            const uint8_t* b_row = B_packed + n_block * K * 2;
            const __fp16* scale_row = B_scales + n_block * num_groups * 4;

            for (size_t i = 0; i < num_live; ++i) {
                const size_t g = live_groups[i];
                const size_t k_base = g * group_size;
                const uint8_t* b_base = b_row + k_base * 2;

                if (i + 1 < num_live) {
                    const size_t g_next = live_groups[i + 1];
                    __builtin_prefetch(b_row + g_next * group_size * 2, 0, 0);
                }

                int8x16_t a_lo = vld1q_s8(A_masked + k_base);
                int8x16_t a_hi = vld1q_s8(A_masked + k_base + 16);
                int32x4_t acc = as_dot_group_one_row(b_base, a_lo, a_hi);
                float32x4_t scales = vcvt_f32_f16(vld1_f16(scale_row + g * 4));
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

// ---------------------------------------------------------------------------
// Mask-building helpers (operate on scores, produce A_masked + metadata).
// These live on the critical path per token; kept simple for round 1.
// ---------------------------------------------------------------------------
namespace {
template <typename ScoreT>
size_t build_actsparse_mask_impl(const ScoreT* S, const int8_t* A, size_t K,
                                 float sparsity, size_t group_size,
                                 int8_t* A_masked, uint64_t* bitmask,
                                 uint16_t* live_groups) {
    const size_t num_groups = K / group_size;
    const size_t num_mask_qwords = (num_groups + 63) / 64;
    std::memset(bitmask, 0, num_mask_qwords * sizeof(uint64_t));

    // Derive threshold = (sparsity)-quantile of |S|.
    size_t num_drop = static_cast<size_t>(sparsity * static_cast<float>(K));
    if (num_drop >= K) num_drop = K - 1;
    std::vector<float> abs_scores(K);
    for (size_t k = 0; k < K; ++k) {
        float v = static_cast<float>(S[k]);
        abs_scores[k] = v < 0 ? -v : v;
    }
    if (num_drop == 0) {
        std::memcpy(A_masked, A, K);
        for (size_t q = 0; q < num_mask_qwords; ++q) bitmask[q] = ~uint64_t(0);
        if (num_groups % 64) {
            size_t used = num_groups % 64;
            bitmask[num_mask_qwords - 1] = (uint64_t(1) << used) - 1;
        }
        for (size_t g = 0; g < num_groups; ++g) live_groups[g] = static_cast<uint16_t>(g);
        return num_groups;
    }
    std::vector<float> sorted = abs_scores;
    std::nth_element(sorted.begin(), sorted.begin() + num_drop, sorted.end());
    float threshold = sorted[num_drop];

    // Apply lane mask to A and set group bits.
    for (size_t g = 0; g < num_groups; ++g) {
        const size_t k0 = g * group_size;
        bool any_live = false;
        for (size_t j = 0; j < group_size; ++j) {
            bool live = abs_scores[k0 + j] > threshold;
            A_masked[k0 + j] = live ? A[k0 + j] : int8_t(0);
            any_live |= live;
        }
        if (any_live) bitmask[g >> 6] |= uint64_t(1) << (g & 63);
    }
    // Handle tail (K not a multiple of group_size): pass-through any tail lanes.
    for (size_t k = num_groups * group_size; k < K; ++k) {
        A_masked[k] = A[k];
    }
    // Compact to live-group list.
    size_t num_live = 0;
    for (size_t q = 0; q < num_mask_qwords; ++q) {
        uint64_t bits = bitmask[q];
        const size_t base = q * 64;
        while (bits) {
            const unsigned idx = __builtin_ctzll(bits);
            bits &= bits - 1;
            live_groups[num_live++] = static_cast<uint16_t>(base + idx);
        }
    }
    return num_live;
}
} // namespace

size_t cactus_build_actsparse_mask_f16(const __fp16* S, const int8_t* A, size_t K,
                                       float sparsity, size_t group_size,
                                       int8_t* A_masked, uint64_t* bitmask,
                                       uint16_t* live_groups) {
    return build_actsparse_mask_impl<__fp16>(S, A, K, sparsity, group_size,
                                             A_masked, bitmask, live_groups);
}

size_t cactus_build_actsparse_mask_f32(const float* S, const int8_t* A, size_t K,
                                       float sparsity, size_t group_size,
                                       int8_t* A_masked, uint64_t* bitmask,
                                       uint16_t* live_groups) {
    return build_actsparse_mask_impl<float>(S, A, K, sparsity, group_size,
                                            A_masked, bitmask, live_groups);
}

size_t cactus_build_actsparse_mask_threshold_f32(
    const float* S, const int8_t* A, size_t K,
    float threshold, size_t group_size,
    int8_t* A_masked, uint64_t* bitmask, uint16_t* live_groups)
{
    const size_t num_groups = K / group_size;
    const size_t num_mask_qwords = (num_groups + 63) / 64;
    std::memset(bitmask, 0, num_mask_qwords * sizeof(uint64_t));

    size_t num_live = 0;
    for (size_t g = 0; g < num_groups; ++g) {
        const size_t k0 = g * group_size;
        bool any_live = false;
        for (size_t j = 0; j < group_size; ++j) {
            float v = S[k0 + j];
            bool live = (v < 0 ? -v : v) > threshold;
            A_masked[k0 + j] = live ? A[k0 + j] : int8_t(0);
            any_live |= live;
        }
        if (any_live) {
            bitmask[g >> 6] |= uint64_t(1) << (g & 63);
            live_groups[num_live++] = static_cast<uint16_t>(g);
        }
    }
    for (size_t k = num_groups * group_size; k < K; ++k) A_masked[k] = A[k];
    return num_live;
}

// NEON-vectorized fused GELU(gate) * up -> int8 quantize in one pass.
// Uses the tanh-based GELU approximation matching `cactus_gelu_f16`:
//   gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
void cactus_gelu_mul_quant_fp16_to_int8(
    const __fp16* gate, const __fp16* up, int8_t* out,
    size_t n, float quant_scale)
{
    const float32x4_t half   = vdupq_n_f32(0.5f);
    const float32x4_t one    = vdupq_n_f32(1.0f);
    const float32x4_t sqrt2p = vdupq_n_f32(0.7978845608028654f);
    const float32x4_t coeff  = vdupq_n_f32(0.044715f);
    const float32x4_t qsv    = vdupq_n_f32(quant_scale);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t g16 = vld1q_f16(gate + i);
        float16x8_t u16 = vld1q_f16(up + i);
        float32x4_t g_lo = vcvt_f32_f16(vget_low_f16(g16));
        float32x4_t g_hi = vcvt_f32_f16(vget_high_f16(g16));
        float32x4_t u_lo = vcvt_f32_f16(vget_low_f16(u16));
        float32x4_t u_hi = vcvt_f32_f16(vget_high_f16(u16));

        float32x4_t g3_lo = vmulq_f32(vmulq_f32(g_lo, g_lo), g_lo);
        float32x4_t g3_hi = vmulq_f32(vmulq_f32(g_hi, g_hi), g_hi);
        float32x4_t inner_lo = vfmaq_f32(g_lo, coeff, g3_lo);
        float32x4_t inner_hi = vfmaq_f32(g_hi, coeff, g3_hi);
        inner_lo = vmulq_f32(sqrt2p, inner_lo);
        inner_hi = vmulq_f32(sqrt2p, inner_hi);
        float32x4_t t_lo = fast_tanh_f32x4(inner_lo);
        float32x4_t t_hi = fast_tanh_f32x4(inner_hi);
        float32x4_t gelu_lo = vmulq_f32(vmulq_f32(half, g_lo), vaddq_f32(one, t_lo));
        float32x4_t gelu_hi = vmulq_f32(vmulq_f32(half, g_hi), vaddq_f32(one, t_hi));

        float32x4_t h_lo = vmulq_f32(vmulq_f32(gelu_lo, u_lo), qsv);
        float32x4_t h_hi = vmulq_f32(vmulq_f32(gelu_hi, u_hi), qsv);

        int32x4_t q_lo = vcvtnq_s32_f32(h_lo);
        int32x4_t q_hi = vcvtnq_s32_f32(h_hi);
        int16x4_t s_lo = vqmovn_s32(q_lo);
        int16x4_t s_hi = vqmovn_s32(q_hi);
        int8x8_t s8 = vqmovn_s16(vcombine_s16(s_lo, s_hi));
        vst1_s8(out + i, s8);
    }
    for (; i < n; ++i) {
        float gv = static_cast<float>(gate[i]);
        float uv = static_cast<float>(up[i]);
        float gelu = 0.5f * gv * (1.0f + std::tanh(0.7978845608028654f * (gv + 0.044715f * gv * gv * gv)));
        float h = gelu * uv * quant_scale;
        int32_t qv = static_cast<int32_t>(std::round(h));
        qv = std::max(-128, std::min(127, qv));
        out[i] = static_cast<int8_t>(qv);
    }
}

// NEON-vectorized bitmask application. Assumes group_size == 32 (all the
// kernels in this file assume that). For each group the 32-byte tile of A
// is either copied to A_masked (live) or written as zeros (dead); live
// groups are simultaneously emitted to the live_groups list.
size_t cactus_apply_actsparse_bitmask(
    const uint64_t* bitmask, const int8_t* A, size_t K, size_t group_size,
    int8_t* A_masked, uint16_t* live_groups)
{
    const size_t num_groups = K / group_size;
    size_t num_live = 0;
    const int8x16_t zero16 = vdupq_n_s8(0);
    for (size_t qw = 0; qw * 64 < num_groups; ++qw) {
        uint64_t bits = bitmask[qw];
        const size_t g_base = qw * 64;
        const size_t g_end = std::min(g_base + 64, num_groups);
        for (size_t g = g_base; g < g_end; ++g) {
            const size_t off = g * group_size;
            bool live = (bits >> (g - g_base)) & 1ull;
            if (live) {
                int8x16_t lo = vld1q_s8(A + off);
                int8x16_t hi = vld1q_s8(A + off + 16);
                vst1q_s8(A_masked + off, lo);
                vst1q_s8(A_masked + off + 16, hi);
                live_groups[num_live++] = static_cast<uint16_t>(g);
            } else {
                vst1q_s8(A_masked + off, zero16);
                vst1q_s8(A_masked + off + 16, zero16);
            }
        }
    }
    for (size_t k = num_groups * group_size; k < K; ++k) A_masked[k] = A[k];
    return num_live;
}

// ---------------------------------------------------------------------------
// Kernel R2.A — bitmask, 2 n_blocks per iteration.
// Mirrors the dense baseline's 2-n_block pattern: per live group load
// A[k..k+32) once and issue TWO B tile loads (one per n_block) to keep the
// DOTQ pipeline and the DRAM request queue fuller.
// ---------------------------------------------------------------------------
void cactus_gemv_int4_actsparse_bitmask_2nb(
    const int8_t* A_masked, float A_scale,
    const int8_t* B_packed_raw, const __fp16* B_scales,
    const uint64_t* live_group_bitmask,
    __fp16* C, size_t K, size_t N, size_t group_size)
{
    const uint8_t* B_packed = reinterpret_cast<const uint8_t*>(B_packed_raw);
    if (K == 0 || N == 0) return;

    const size_t num_groups = K / group_size;
    const size_t N_blocks = (N + 3) / 4;

    auto process_blocks = [=](size_t block_start, size_t block_end) {
        size_t n_block = block_start;
        for (; n_block + 1 < block_end; n_block += 2) {
            float32x4_t sum_a = vdupq_n_f32(0.0f);
            float32x4_t sum_b = vdupq_n_f32(0.0f);
            const uint8_t* ba_row = B_packed + n_block * K * 2;
            const uint8_t* bb_row = B_packed + (n_block + 1) * K * 2;
            const __fp16* sa_row = B_scales + n_block * num_groups * 4;
            const __fp16* sb_row = B_scales + (n_block + 1) * num_groups * 4;

            for (size_t w = 0; w * 64 < num_groups; ++w) {
                uint64_t bits = live_group_bitmask[w];
                const size_t base_g = w * 64;
                const size_t end_g = std::min(base_g + 64, num_groups);
                const size_t local_end = end_g - base_g;
                if (local_end < 64)
                    bits &= (uint64_t(1) << local_end) - 1;
                while (bits) {
                    const unsigned b_idx = __builtin_ctzll(bits);
                    bits &= bits - 1;
                    const size_t g = base_g + b_idx;
                    const size_t k_base = g * group_size;
                    const uint8_t* ba = ba_row + k_base * 2;
                    const uint8_t* bb = bb_row + k_base * 2;
                    __builtin_prefetch(ba + 512, 0, 0);
                    __builtin_prefetch(bb + 512, 0, 0);

                    int8x16_t a_lo = vld1q_s8(A_masked + k_base);
                    int8x16_t a_hi = vld1q_s8(A_masked + k_base + 16);
                    int32x4_t acc_a = as_dot_group_one_row(ba, a_lo, a_hi);
                    int32x4_t acc_b = as_dot_group_one_row(bb, a_lo, a_hi);

                    float32x4_t sa = vcvt_f32_f16(vld1_f16(sa_row + g * 4));
                    float32x4_t sb = vcvt_f32_f16(vld1_f16(sb_row + g * 4));
                    sum_a = vmlaq_f32(sum_a, vcvtq_f32_s32(acc_a), sa);
                    sum_b = vmlaq_f32(sum_b, vcvtq_f32_s32(acc_b), sb);
                }
            }

            vst1_f16(C + n_block * 4, vcvt_f16_f32(vmulq_n_f32(sum_a, A_scale)));
            vst1_f16(C + (n_block + 1) * 4, vcvt_f16_f32(vmulq_n_f32(sum_b, A_scale)));
        }
        // Tail n_block (if N_blocks odd).
        for (; n_block < block_end; ++n_block) {
            const size_t n_start = n_block * 4;
            const size_t actual_n = std::min(size_t(4), N - n_start);
            float32x4_t running_sum = vdupq_n_f32(0.0f);
            const uint8_t* b_row = B_packed + n_block * K * 2;
            const __fp16* scale_row = B_scales + n_block * num_groups * 4;

            for (size_t w = 0; w * 64 < num_groups; ++w) {
                uint64_t bits = live_group_bitmask[w];
                const size_t base_g = w * 64;
                const size_t end_g = std::min(base_g + 64, num_groups);
                const size_t local_end = end_g - base_g;
                if (local_end < 64)
                    bits &= (uint64_t(1) << local_end) - 1;
                while (bits) {
                    const unsigned b_idx = __builtin_ctzll(bits);
                    bits &= bits - 1;
                    const size_t g = base_g + b_idx;
                    const size_t k_base = g * group_size;
                    const uint8_t* b_base = b_row + k_base * 2;

                    int8x16_t a_lo = vld1q_s8(A_masked + k_base);
                    int8x16_t a_hi = vld1q_s8(A_masked + k_base + 16);
                    int32x4_t acc = as_dot_group_one_row(b_base, a_lo, a_hi);
                    float32x4_t scales = vcvt_f32_f16(vld1_f16(scale_row + g * 4));
                    running_sum = vmlaq_f32(running_sum, vcvtq_f32_s32(acc), scales);
                }
            }
            float32x4_t result = vmulq_n_f32(running_sum, A_scale);
            float16x4_t result_f16 = vcvt_f16_f32(result);
            if (actual_n == 4) vst1_f16(C + n_start, result_f16);
            else for (size_t ni = 0; ni < actual_n; ni++) {
                C[n_start + ni] = vget_lane_f16(result_f16, 0);
                result_f16 = vext_f16(result_f16, result_f16, 1);
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

// ---------------------------------------------------------------------------
// Kernel R2.D — livelist with multi-step prefetch on baseline layout.
// ---------------------------------------------------------------------------
void cactus_gemv_int4_actsparse_livelist_pf(
    const int8_t* A_masked, float A_scale,
    const int8_t* B_packed_raw, const __fp16* B_scales,
    const uint16_t* live_groups, size_t num_live,
    __fp16* C, size_t K, size_t N, size_t group_size)
{
    const uint8_t* B_packed = reinterpret_cast<const uint8_t*>(B_packed_raw);
    if (K == 0 || N == 0 || num_live == 0) {
        if (N && C) std::memset(C, 0, N * sizeof(__fp16));
        return;
    }
    constexpr size_t PF_STEPS = 4;

    const size_t num_groups = K / group_size;
    const size_t N_blocks = (N + 3) / 4;

    auto process_blocks = [=](size_t block_start, size_t block_end) {
        size_t n_block = block_start;
        for (; n_block + 1 < block_end; n_block += 2) {
            float32x4_t sum_a = vdupq_n_f32(0.0f);
            float32x4_t sum_b = vdupq_n_f32(0.0f);
            const uint8_t* ba_row = B_packed + n_block * K * 2;
            const uint8_t* bb_row = B_packed + (n_block + 1) * K * 2;
            const __fp16* sa_row = B_scales + n_block * num_groups * 4;
            const __fp16* sb_row = B_scales + (n_block + 1) * num_groups * 4;

            // Prime prefetches.
            for (size_t p = 0; p < std::min(PF_STEPS, num_live); ++p) {
                size_t gp = live_groups[p];
                __builtin_prefetch(ba_row + gp * group_size * 2, 0, 0);
                __builtin_prefetch(bb_row + gp * group_size * 2, 0, 0);
            }

            for (size_t i = 0; i < num_live; ++i) {
                const size_t g = live_groups[i];
                const size_t k_base = g * group_size;
                const uint8_t* ba = ba_row + k_base * 2;
                const uint8_t* bb = bb_row + k_base * 2;
                if (i + PF_STEPS < num_live) {
                    size_t gp = live_groups[i + PF_STEPS];
                    __builtin_prefetch(ba_row + gp * group_size * 2, 0, 0);
                    __builtin_prefetch(bb_row + gp * group_size * 2, 0, 0);
                }
                int8x16_t a_lo = vld1q_s8(A_masked + k_base);
                int8x16_t a_hi = vld1q_s8(A_masked + k_base + 16);
                int32x4_t acc_a = as_dot_group_one_row(ba, a_lo, a_hi);
                int32x4_t acc_b = as_dot_group_one_row(bb, a_lo, a_hi);
                float32x4_t sa = vcvt_f32_f16(vld1_f16(sa_row + g * 4));
                float32x4_t sb = vcvt_f32_f16(vld1_f16(sb_row + g * 4));
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
            const uint8_t* b_row = B_packed + n_block * K * 2;
            const __fp16* scale_row = B_scales + n_block * num_groups * 4;
            for (size_t i = 0; i < num_live; ++i) {
                const size_t g = live_groups[i];
                const size_t k_base = g * group_size;
                const uint8_t* b_base = b_row + k_base * 2;
                int8x16_t a_lo = vld1q_s8(A_masked + k_base);
                int8x16_t a_hi = vld1q_s8(A_masked + k_base + 16);
                int32x4_t acc = as_dot_group_one_row(b_base, a_lo, a_hi);
                float32x4_t scales = vcvt_f32_f16(vld1_f16(scale_row + g * 4));
                running_sum = vmlaq_f32(running_sum, vcvtq_f32_s32(acc), scales);
            }
            float32x4_t result = vmulq_n_f32(running_sum, A_scale);
            float16x4_t result_f16 = vcvt_f16_f32(result);
            if (actual_n == 4) vst1_f16(C + n_start, result_f16);
            else for (size_t ni = 0; ni < actual_n; ni++) {
                C[n_start + ni] = vget_lane_f16(result_f16, 0);
                result_f16 = vext_f16(result_f16, result_f16, 1);
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

// ---------------------------------------------------------------------------
// Kernel R2.B — K-major repack + K-outer GEMV.
// Once-per-matrix repack (legal: doesn't depend on S): baseline layout
// [n_block][k-group][64B] becomes [k-group][n_block][64B]. Then per live
// group the 64 B × N_blocks weight stream is contiguous in memory — one
// cacheline-perfect streaming read per group. Scales are also repacked
// into [k-group][n_block][4 fp16] layout so the scale/byte ratio is kept
// in-cache.
// ---------------------------------------------------------------------------
void cactus_repack_int4_kmajor(
    const int8_t* B_packed_raw, const __fp16* B_scales,
    uint8_t* B_packed_km, __fp16* B_scales_km,
    size_t K, size_t N, size_t group_size)
{
    const uint8_t* B_packed = reinterpret_cast<const uint8_t*>(B_packed_raw);
    const size_t num_groups = K / group_size;
    const size_t N_blocks = (N + 3) / 4;
    for (size_t g = 0; g < num_groups; ++g) {
        const size_t k_base = g * group_size;
        uint8_t* dst_pack = B_packed_km + g * N_blocks * 64;
        __fp16* dst_scale = B_scales_km + g * N_blocks * 4;
        for (size_t nb = 0; nb < N_blocks; ++nb) {
            const uint8_t* src = B_packed + (nb * K + k_base) * 2;
            std::memcpy(dst_pack + nb * 64, src, 64);
            const __fp16* ssrc = B_scales + (nb * num_groups + g) * 4;
            std::memcpy(dst_scale + nb * 4, ssrc, 4 * sizeof(__fp16));
        }
    }
}

void cactus_gemv_int4_actsparse_kmajor(
    const int8_t* A_masked, float A_scale,
    const uint8_t* B_packed_km, const __fp16* B_scales_km,
    const uint16_t* live_groups, size_t num_live,
    __fp16* C, size_t K, size_t N, size_t group_size)
{
    if (K == 0 || N == 0) return;
    const size_t N_blocks = (N + 3) / 4;

    // Parallelise over N_blocks (each thread produces a disjoint slice of C).
    auto process = [=](size_t nb_start, size_t nb_end) {
        // Zero the slice of C this worker owns so we accumulate cleanly.
        const size_t n_start = nb_start * 4;
        const size_t n_end = std::min(N, nb_end * 4);
        // Per-n_block accumulators are kept in a small scratch vector so
        // per-group updates touch only contiguous memory.
        const size_t slice_nb = nb_end - nb_start;
        if (slice_nb == 0) return;
        std::vector<float32x4_t> acc(slice_nb, vdupq_n_f32(0.0f));

        if (num_live == 0) {
            for (size_t nb = nb_start; nb < nb_end; ++nb) {
                const size_t ns = nb * 4;
                const size_t actual_n = std::min(size_t(4), N - ns);
                float16x4_t z = vcvt_f16_f32(vdupq_n_f32(0.0f));
                if (actual_n == 4) vst1_f16(C + ns, z);
                else for (size_t ni = 0; ni < actual_n; ni++) C[ns + ni] = __fp16(0);
            }
            return;
        }

        // K-outer: for each live group g, stream 64-byte tiles × N_blocks.
        for (size_t i = 0; i < num_live; ++i) {
            const size_t g = live_groups[i];
            const uint8_t* gp = B_packed_km + g * N_blocks * 64 + nb_start * 64;
            const __fp16* sp = B_scales_km + g * N_blocks * 4 + nb_start * 4;
            int8x16_t a_lo = vld1q_s8(A_masked + g * group_size);
            int8x16_t a_hi = vld1q_s8(A_masked + g * group_size + 16);

            // Prefetch a few tiles ahead of the nb-inner loop.
            __builtin_prefetch(gp + 512, 0, 0);
            __builtin_prefetch(sp + 64, 0, 0);

            // If next group exists, prefetch the head of its stream.
            if (i + 1 < num_live) {
                size_t g_next = live_groups[i + 1];
                __builtin_prefetch(B_packed_km + g_next * N_blocks * 64 + nb_start * 64, 0, 0);
            }

            for (size_t j = 0; j < slice_nb; ++j) {
                __builtin_prefetch(gp + j * 64 + 512, 0, 0);
                int32x4_t acc_i = as_dot_group_one_row(gp + j * 64, a_lo, a_hi);
                float32x4_t scales = vcvt_f32_f16(vld1_f16(sp + j * 4));
                acc[j] = vmlaq_f32(acc[j], vcvtq_f32_s32(acc_i), scales);
            }
        }

        // Flush accumulators to C.
        for (size_t j = 0; j < slice_nb; ++j) {
            size_t nb = nb_start + j;
            size_t ns = nb * 4;
            size_t actual_n = std::min(size_t(4), N - ns);
            float32x4_t result = vmulq_n_f32(acc[j], A_scale);
            float16x4_t result_f16 = vcvt_f16_f32(result);
            if (actual_n == 4) vst1_f16(C + ns, result_f16);
            else for (size_t ni = 0; ni < actual_n; ni++) {
                C[ns + ni] = vget_lane_f16(result_f16, 0);
                result_f16 = vext_f16(result_f16, result_f16, 1);
            }
        }
        (void)n_start; (void)n_end;
    };

    auto& pool = CactusThreading::get_thread_pool();
    size_t num_threads = CactusThreading::GemmThreading::get_gemv_threads(N_blocks, pool.num_workers());
    num_threads = std::min(num_threads, N_blocks);
    if (num_threads <= 1) {
        process(0, N_blocks);
    } else {
        pool.enqueue_n_threads(N_blocks, num_threads, process);
        pool.wait_all();
    }
}

// ---------------------------------------------------------------------------
// Round 3 — K-major inline layout.
// Per (g, nb): 64 packed bytes + 4 fp16 scales = 72 bytes, tightly packed.
// One contiguous stream per live group covers packed + scales.
// ---------------------------------------------------------------------------
static constexpr size_t KMI_TILE = 72;

void cactus_repack_int4_kmajor_inline(
    const int8_t* B_packed_raw, const __fp16* B_scales,
    uint8_t* B_km_inline,
    size_t K, size_t N, size_t group_size)
{
    const uint8_t* B_packed = reinterpret_cast<const uint8_t*>(B_packed_raw);
    const size_t num_groups = K / group_size;
    const size_t N_blocks = (N + 3) / 4;
    for (size_t g = 0; g < num_groups; ++g) {
        const size_t k_base = g * group_size;
        uint8_t* gdst = B_km_inline + g * N_blocks * KMI_TILE;
        for (size_t nb = 0; nb < N_blocks; ++nb) {
            const uint8_t* src = B_packed + (nb * K + k_base) * 2;
            const __fp16* ssrc = B_scales + (nb * num_groups + g) * 4;
            uint8_t* tile = gdst + nb * KMI_TILE;
            std::memcpy(tile, src, 64);
            std::memcpy(tile + 64, ssrc, 4 * sizeof(__fp16));
        }
    }
}

namespace {

// Helper: stream-kernel for one live group on inline layout.
// Accumulates into slice of acc[] (size = slice_nb). A_group = (a_lo, a_hi).
inline void kmi_accumulate_group(
    const uint8_t* group_start, size_t slice_nb, size_t nb_start,
    int8x16_t a_lo, int8x16_t a_hi,
    float32x4_t* acc)
{
    const uint8_t* p = group_start + nb_start * KMI_TILE;
    for (size_t j = 0; j < slice_nb; ++j) {
        __builtin_prefetch(p + 4 * KMI_TILE, 0, 0);
        int32x4_t acc_i = as_dot_group_one_row(p, a_lo, a_hi);
        float32x4_t scales = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p + 64)));
        acc[j] = vmlaq_f32(acc[j], vcvtq_f32_s32(acc_i), scales);
        p += KMI_TILE;
    }
}

} // namespace

void cactus_gemv_int4_actsparse_kmi(
    const int8_t* A_masked, float A_scale,
    const uint8_t* B_km_inline,
    const uint16_t* live_groups, size_t num_live,
    __fp16* C, size_t K, size_t N, size_t group_size)
{
    if (K == 0 || N == 0) return;
    const size_t N_blocks = (N + 3) / 4;

    auto process = [=](size_t nb_start, size_t nb_end) {
        const size_t slice_nb = nb_end - nb_start;
        if (slice_nb == 0) return;
        std::vector<float32x4_t> acc(slice_nb, vdupq_n_f32(0.0f));

        for (size_t i = 0; i < num_live; ++i) {
            const size_t g = live_groups[i];
            const uint8_t* gp = B_km_inline + g * N_blocks * KMI_TILE;
            int8x16_t a_lo = vld1q_s8(A_masked + g * group_size);
            int8x16_t a_hi = vld1q_s8(A_masked + g * group_size + 16);
            if (i + 1 < num_live) {
                size_t g_next = live_groups[i + 1];
                __builtin_prefetch(B_km_inline + g_next * N_blocks * KMI_TILE + nb_start * KMI_TILE, 0, 0);
            }
            kmi_accumulate_group(gp, slice_nb, nb_start, a_lo, a_hi, acc.data());
        }

        for (size_t j = 0; j < slice_nb; ++j) {
            size_t nb = nb_start + j;
            size_t ns = nb * 4;
            size_t actual_n = std::min(size_t(4), N - ns);
            float32x4_t result = vmulq_n_f32(acc[j], A_scale);
            float16x4_t result_f16 = vcvt_f16_f32(result);
            if (actual_n == 4) vst1_f16(C + ns, result_f16);
            else for (size_t ni = 0; ni < actual_n; ni++) {
                C[ns + ni] = vget_lane_f16(result_f16, 0);
                result_f16 = vext_f16(result_f16, result_f16, 1);
            }
        }
    };

    auto& pool = CactusThreading::get_thread_pool();
    size_t num_threads = CactusThreading::GemmThreading::get_gemv_threads(N_blocks, pool.num_workers());
    num_threads = std::min(num_threads, N_blocks);
    if (num_threads <= 1) {
        process(0, N_blocks);
    } else {
        pool.enqueue_n_threads(N_blocks, num_threads, process);
        pool.wait_all();
    }
}

// ---------------------------------------------------------------------------
// R3.B — process 2 live groups per outer pass.
// Amortizes acc[j] R+W across 2 group-streams.
// ---------------------------------------------------------------------------
void cactus_gemv_int4_actsparse_kmi2(
    const int8_t* A_masked, float A_scale,
    const uint8_t* B_km_inline,
    const uint16_t* live_groups, size_t num_live,
    __fp16* C, size_t K, size_t N, size_t group_size)
{
    if (K == 0 || N == 0) return;
    const size_t N_blocks = (N + 3) / 4;

    auto process = [=](size_t nb_start, size_t nb_end) {
        const size_t slice_nb = nb_end - nb_start;
        if (slice_nb == 0) return;
        std::vector<float32x4_t> acc(slice_nb, vdupq_n_f32(0.0f));

        size_t i = 0;
        for (; i + 1 < num_live; i += 2) {
            const size_t g0 = live_groups[i];
            const size_t g1 = live_groups[i + 1];
            const uint8_t* p0 = B_km_inline + g0 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
            const uint8_t* p1 = B_km_inline + g1 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
            int8x16_t a0_lo = vld1q_s8(A_masked + g0 * group_size);
            int8x16_t a0_hi = vld1q_s8(A_masked + g0 * group_size + 16);
            int8x16_t a1_lo = vld1q_s8(A_masked + g1 * group_size);
            int8x16_t a1_hi = vld1q_s8(A_masked + g1 * group_size + 16);
            if (i + 2 < num_live) {
                size_t g2 = live_groups[i + 2];
                __builtin_prefetch(B_km_inline + g2 * N_blocks * KMI_TILE + nb_start * KMI_TILE, 0, 0);
            }
            for (size_t j = 0; j < slice_nb; ++j) {
                __builtin_prefetch(p0 + 4 * KMI_TILE, 0, 0);
                __builtin_prefetch(p1 + 4 * KMI_TILE, 0, 0);
                int32x4_t acc_0 = as_dot_group_one_row(p0, a0_lo, a0_hi);
                int32x4_t acc_1 = as_dot_group_one_row(p1, a1_lo, a1_hi);
                float32x4_t s0 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p0 + 64)));
                float32x4_t s1 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p1 + 64)));
                float32x4_t r = acc[j];
                r = vmlaq_f32(r, vcvtq_f32_s32(acc_0), s0);
                r = vmlaq_f32(r, vcvtq_f32_s32(acc_1), s1);
                acc[j] = r;
                p0 += KMI_TILE; p1 += KMI_TILE;
            }
        }
        for (; i < num_live; ++i) {
            const size_t g = live_groups[i];
            const uint8_t* gp = B_km_inline + g * N_blocks * KMI_TILE;
            int8x16_t a_lo = vld1q_s8(A_masked + g * group_size);
            int8x16_t a_hi = vld1q_s8(A_masked + g * group_size + 16);
            kmi_accumulate_group(gp, slice_nb, nb_start, a_lo, a_hi, acc.data());
        }

        for (size_t j = 0; j < slice_nb; ++j) {
            size_t nb = nb_start + j;
            size_t ns = nb * 4;
            size_t actual_n = std::min(size_t(4), N - ns);
            float32x4_t result = vmulq_n_f32(acc[j], A_scale);
            float16x4_t result_f16 = vcvt_f16_f32(result);
            if (actual_n == 4) vst1_f16(C + ns, result_f16);
            else for (size_t ni = 0; ni < actual_n; ni++) {
                C[ns + ni] = vget_lane_f16(result_f16, 0);
                result_f16 = vext_f16(result_f16, result_f16, 1);
            }
        }
    };

    auto& pool = CactusThreading::get_thread_pool();
    size_t num_threads = CactusThreading::GemmThreading::get_gemv_threads(N_blocks, pool.num_workers());
    num_threads = std::min(num_threads, N_blocks);
    if (num_threads <= 1) {
        process(0, N_blocks);
    } else {
        pool.enqueue_n_threads(N_blocks, num_threads, process);
        pool.wait_all();
    }
}

namespace {
// Reusable thread-local scratch for per-worker accumulators.
// Each worker sizes the buffer up once then reuses it across calls.
struct AccScratch {
    std::vector<float32x4_t> buf;
};
inline AccScratch& kmi_scratch() {
    thread_local AccScratch s;
    return s;
}
} // namespace

// ---------------------------------------------------------------------------
// R3.C — process 4 live groups per outer pass.
// ---------------------------------------------------------------------------
void cactus_gemv_int4_actsparse_kmi4(
    const int8_t* A_masked, float A_scale,
    const uint8_t* B_km_inline,
    const uint16_t* live_groups, size_t num_live,
    __fp16* C, size_t K, size_t N, size_t group_size)
{
    if (K == 0 || N == 0) return;
    const size_t N_blocks = (N + 3) / 4;

    auto process = [=](size_t nb_start, size_t nb_end) {
        const size_t slice_nb = nb_end - nb_start;
        if (slice_nb == 0) return;
        std::vector<float32x4_t> acc(slice_nb, vdupq_n_f32(0.0f));

        size_t i = 0;
        for (; i + 3 < num_live; i += 4) {
            const size_t g0 = live_groups[i];
            const size_t g1 = live_groups[i + 1];
            const size_t g2 = live_groups[i + 2];
            const size_t g3 = live_groups[i + 3];
            const uint8_t* p0 = B_km_inline + g0 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
            const uint8_t* p1 = B_km_inline + g1 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
            const uint8_t* p2 = B_km_inline + g2 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
            const uint8_t* p3 = B_km_inline + g3 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
            int8x16_t a0_lo = vld1q_s8(A_masked + g0 * group_size);
            int8x16_t a0_hi = vld1q_s8(A_masked + g0 * group_size + 16);
            int8x16_t a1_lo = vld1q_s8(A_masked + g1 * group_size);
            int8x16_t a1_hi = vld1q_s8(A_masked + g1 * group_size + 16);
            int8x16_t a2_lo = vld1q_s8(A_masked + g2 * group_size);
            int8x16_t a2_hi = vld1q_s8(A_masked + g2 * group_size + 16);
            int8x16_t a3_lo = vld1q_s8(A_masked + g3 * group_size);
            int8x16_t a3_hi = vld1q_s8(A_masked + g3 * group_size + 16);
            if (i + 4 < num_live) {
                size_t g4 = live_groups[i + 4];
                __builtin_prefetch(B_km_inline + g4 * N_blocks * KMI_TILE + nb_start * KMI_TILE, 0, 0);
            }
            for (size_t j = 0; j < slice_nb; ++j) {
                __builtin_prefetch(p0 + 8 * KMI_TILE, 0, 0);
                __builtin_prefetch(p1 + 8 * KMI_TILE, 0, 0);
                __builtin_prefetch(p2 + 8 * KMI_TILE, 0, 0);
                __builtin_prefetch(p3 + 8 * KMI_TILE, 0, 0);
                int32x4_t acc_0 = as_dot_group_one_row(p0, a0_lo, a0_hi);
                int32x4_t acc_1 = as_dot_group_one_row(p1, a1_lo, a1_hi);
                int32x4_t acc_2 = as_dot_group_one_row(p2, a2_lo, a2_hi);
                int32x4_t acc_3 = as_dot_group_one_row(p3, a3_lo, a3_hi);
                float32x4_t s0 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p0 + 64)));
                float32x4_t s1 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p1 + 64)));
                float32x4_t s2 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p2 + 64)));
                float32x4_t s3 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p3 + 64)));
                float32x4_t r = acc[j];
                r = vmlaq_f32(r, vcvtq_f32_s32(acc_0), s0);
                r = vmlaq_f32(r, vcvtq_f32_s32(acc_1), s1);
                r = vmlaq_f32(r, vcvtq_f32_s32(acc_2), s2);
                r = vmlaq_f32(r, vcvtq_f32_s32(acc_3), s3);
                acc[j] = r;
                p0 += KMI_TILE; p1 += KMI_TILE; p2 += KMI_TILE; p3 += KMI_TILE;
            }
        }
        // Remainder as a 2-batch + single.
        for (; i + 1 < num_live; i += 2) {
            const size_t g0 = live_groups[i];
            const size_t g1 = live_groups[i + 1];
            const uint8_t* p0 = B_km_inline + g0 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
            const uint8_t* p1 = B_km_inline + g1 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
            int8x16_t a0_lo = vld1q_s8(A_masked + g0 * group_size);
            int8x16_t a0_hi = vld1q_s8(A_masked + g0 * group_size + 16);
            int8x16_t a1_lo = vld1q_s8(A_masked + g1 * group_size);
            int8x16_t a1_hi = vld1q_s8(A_masked + g1 * group_size + 16);
            for (size_t j = 0; j < slice_nb; ++j) {
                int32x4_t acc_0 = as_dot_group_one_row(p0, a0_lo, a0_hi);
                int32x4_t acc_1 = as_dot_group_one_row(p1, a1_lo, a1_hi);
                float32x4_t s0 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p0 + 64)));
                float32x4_t s1 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p1 + 64)));
                float32x4_t r = acc[j];
                r = vmlaq_f32(r, vcvtq_f32_s32(acc_0), s0);
                r = vmlaq_f32(r, vcvtq_f32_s32(acc_1), s1);
                acc[j] = r;
                p0 += KMI_TILE; p1 += KMI_TILE;
            }
        }
        for (; i < num_live; ++i) {
            const size_t g = live_groups[i];
            const uint8_t* gp = B_km_inline + g * N_blocks * KMI_TILE;
            int8x16_t a_lo = vld1q_s8(A_masked + g * group_size);
            int8x16_t a_hi = vld1q_s8(A_masked + g * group_size + 16);
            kmi_accumulate_group(gp, slice_nb, nb_start, a_lo, a_hi, acc.data());
        }

        for (size_t j = 0; j < slice_nb; ++j) {
            size_t nb = nb_start + j;
            size_t ns = nb * 4;
            size_t actual_n = std::min(size_t(4), N - ns);
            float32x4_t result = vmulq_n_f32(acc[j], A_scale);
            float16x4_t result_f16 = vcvt_f16_f32(result);
            if (actual_n == 4) vst1_f16(C + ns, result_f16);
            else for (size_t ni = 0; ni < actual_n; ni++) {
                C[ns + ni] = vget_lane_f16(result_f16, 0);
                result_f16 = vext_f16(result_f16, result_f16, 1);
            }
        }
    };

    auto& pool = CactusThreading::get_thread_pool();
    size_t num_threads = CactusThreading::GemmThreading::get_gemv_threads(N_blocks, pool.num_workers());
    num_threads = std::min(num_threads, N_blocks);
    if (num_threads <= 1) {
        process(0, N_blocks);
    } else {
        pool.enqueue_n_threads(N_blocks, num_threads, process);
        pool.wait_all();
    }
}

void cactus_gemv_int4_actsparse_kmi4_chain(
    const int8_t* A_masked, float A_scale,
    const uint8_t* B_km_inline,
    const uint16_t* live_groups, size_t num_live,
    __fp16* C, size_t K, size_t N, size_t group_size)
{
    if (K == 0 || N == 0) return;
    const size_t N_blocks = (N + 3) / 4;

    auto process = [=](size_t nb_start, size_t nb_end) {
        const size_t slice_nb = nb_end - nb_start;
        if (slice_nb == 0) return;
        std::vector<float32x4_t> acc(slice_nb, vdupq_n_f32(0.0f));

        size_t i = 0;
        for (; i + 3 < num_live; i += 4) {
            const size_t g0 = live_groups[i];
            const size_t g1 = live_groups[i + 1];
            const size_t g2 = live_groups[i + 2];
            const size_t g3 = live_groups[i + 3];
            const uint8_t* p0 = B_km_inline + g0 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
            const uint8_t* p1 = B_km_inline + g1 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
            const uint8_t* p2 = B_km_inline + g2 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
            const uint8_t* p3 = B_km_inline + g3 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
            int8x16_t a0_lo = vld1q_s8(A_masked + g0 * group_size);
            int8x16_t a0_hi = vld1q_s8(A_masked + g0 * group_size + 16);
            int8x16_t a1_lo = vld1q_s8(A_masked + g1 * group_size);
            int8x16_t a1_hi = vld1q_s8(A_masked + g1 * group_size + 16);
            int8x16_t a2_lo = vld1q_s8(A_masked + g2 * group_size);
            int8x16_t a2_hi = vld1q_s8(A_masked + g2 * group_size + 16);
            int8x16_t a3_lo = vld1q_s8(A_masked + g3 * group_size);
            int8x16_t a3_hi = vld1q_s8(A_masked + g3 * group_size + 16);
            if (i + 4 < num_live) {
                size_t g4 = live_groups[i + 4];
                __builtin_prefetch(B_km_inline + g4 * N_blocks * KMI_TILE + nb_start * KMI_TILE, 0, 0);
            }
            for (size_t j = 0; j < slice_nb; ++j) {
                __builtin_prefetch(p0 + 8 * KMI_TILE, 0, 0);
                __builtin_prefetch(p1 + 8 * KMI_TILE, 0, 0);
                __builtin_prefetch(p2 + 8 * KMI_TILE, 0, 0);
                __builtin_prefetch(p3 + 8 * KMI_TILE, 0, 0);
                float32x4_t r = acc[j];
                int32x4_t acc_0 = as_dot_group_one_row(p0, a0_lo, a0_hi);
                float32x4_t s0 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p0 + 64)));
                r = vmlaq_f32(r, vcvtq_f32_s32(acc_0), s0);
                int32x4_t acc_1 = as_dot_group_one_row(p1, a1_lo, a1_hi);
                float32x4_t s1 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p1 + 64)));
                r = vmlaq_f32(r, vcvtq_f32_s32(acc_1), s1);
                int32x4_t acc_2 = as_dot_group_one_row(p2, a2_lo, a2_hi);
                float32x4_t s2 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p2 + 64)));
                r = vmlaq_f32(r, vcvtq_f32_s32(acc_2), s2);
                int32x4_t acc_3 = as_dot_group_one_row(p3, a3_lo, a3_hi);
                float32x4_t s3 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p3 + 64)));
                acc[j] = vmlaq_f32(r, vcvtq_f32_s32(acc_3), s3);
                p0 += KMI_TILE; p1 += KMI_TILE; p2 += KMI_TILE; p3 += KMI_TILE;
            }
        }
        for (; i + 1 < num_live; i += 2) {
            const size_t g0 = live_groups[i];
            const size_t g1 = live_groups[i + 1];
            const uint8_t* p0 = B_km_inline + g0 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
            const uint8_t* p1 = B_km_inline + g1 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
            int8x16_t a0_lo = vld1q_s8(A_masked + g0 * group_size);
            int8x16_t a0_hi = vld1q_s8(A_masked + g0 * group_size + 16);
            int8x16_t a1_lo = vld1q_s8(A_masked + g1 * group_size);
            int8x16_t a1_hi = vld1q_s8(A_masked + g1 * group_size + 16);
            for (size_t j = 0; j < slice_nb; ++j) {
                float32x4_t r = acc[j];
                int32x4_t acc_0 = as_dot_group_one_row(p0, a0_lo, a0_hi);
                float32x4_t s0 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p0 + 64)));
                r = vmlaq_f32(r, vcvtq_f32_s32(acc_0), s0);
                int32x4_t acc_1 = as_dot_group_one_row(p1, a1_lo, a1_hi);
                float32x4_t s1 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p1 + 64)));
                acc[j] = vmlaq_f32(r, vcvtq_f32_s32(acc_1), s1);
                p0 += KMI_TILE; p1 += KMI_TILE;
            }
        }
        for (; i < num_live; ++i) {
            const size_t g = live_groups[i];
            const uint8_t* gp = B_km_inline + g * N_blocks * KMI_TILE;
            int8x16_t a_lo = vld1q_s8(A_masked + g * group_size);
            int8x16_t a_hi = vld1q_s8(A_masked + g * group_size + 16);
            kmi_accumulate_group(gp, slice_nb, nb_start, a_lo, a_hi, acc.data());
        }

        for (size_t j = 0; j < slice_nb; ++j) {
            size_t nb = nb_start + j;
            size_t ns = nb * 4;
            size_t actual_n = std::min(size_t(4), N - ns);
            float32x4_t result = vmulq_n_f32(acc[j], A_scale);
            float16x4_t result_f16 = vcvt_f16_f32(result);
            if (actual_n == 4) vst1_f16(C + ns, result_f16);
            else for (size_t ni = 0; ni < actual_n; ni++) {
                C[ns + ni] = vget_lane_f16(result_f16, 0);
                result_f16 = vext_f16(result_f16, result_f16, 1);
            }
        }
    };

    auto& pool = CactusThreading::get_thread_pool();
    size_t num_threads = CactusThreading::GemmThreading::get_gemv_threads(N_blocks, pool.num_workers());
    num_threads = std::min(num_threads, N_blocks);
    if (num_threads <= 1) {
        process(0, N_blocks);
    } else {
        pool.enqueue_n_threads(N_blocks, num_threads, process);
        pool.wait_all();
    }
}

// ---------------------------------------------------------------------------
// R4 — kmi4_fast: KMI4 + cold-start prefetch + thread_local acc scratch +
// single-threaded fast path for small N.
//
// Below ~384 n_blocks we skip the thread pool entirely; the dispatch cost
// alone is ~5–10 µs which is significant when the whole call should be
// ~30 µs. Between groups we prefetch the first 8 tiles of the next group
// before entering the inner loop to prime L1.
// ---------------------------------------------------------------------------
namespace {
inline void kmi4_fast_body(
    const int8_t* A_masked, float A_scale,
    const uint8_t* B_km_inline,
    const uint16_t* live_groups, size_t num_live,
    __fp16* C, size_t N, size_t N_blocks,
    size_t group_size, size_t nb_start, size_t nb_end)
{
    const size_t slice_nb = nb_end - nb_start;
    if (slice_nb == 0) return;

    auto& s = kmi_scratch();
    if (s.buf.size() < slice_nb) s.buf.resize(slice_nb);
    float32x4_t* acc = s.buf.data();
    for (size_t j = 0; j < slice_nb; ++j) acc[j] = vdupq_n_f32(0.0f);

    auto prefetch_group_head = [&](size_t g) {
        const uint8_t* gp = B_km_inline + g * N_blocks * KMI_TILE + nb_start * KMI_TILE;
        __builtin_prefetch(gp + 0,   0, 0);
        __builtin_prefetch(gp + 128, 0, 0);
        __builtin_prefetch(gp + 256, 0, 0);
        __builtin_prefetch(gp + 384, 0, 0);
        __builtin_prefetch(gp + 512, 0, 0);
    };
    for (size_t p = 0; p < std::min<size_t>(4, num_live); ++p)
        prefetch_group_head(live_groups[p]);

    size_t i = 0;
    for (; i + 3 < num_live; i += 4) {
        const size_t g0 = live_groups[i];
        const size_t g1 = live_groups[i + 1];
        const size_t g2 = live_groups[i + 2];
        const size_t g3 = live_groups[i + 3];
        const uint8_t* p0 = B_km_inline + g0 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
        const uint8_t* p1 = B_km_inline + g1 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
        const uint8_t* p2 = B_km_inline + g2 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
        const uint8_t* p3 = B_km_inline + g3 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
        int8x16_t a0_lo = vld1q_s8(A_masked + g0 * group_size);
        int8x16_t a0_hi = vld1q_s8(A_masked + g0 * group_size + 16);
        int8x16_t a1_lo = vld1q_s8(A_masked + g1 * group_size);
        int8x16_t a1_hi = vld1q_s8(A_masked + g1 * group_size + 16);
        int8x16_t a2_lo = vld1q_s8(A_masked + g2 * group_size);
        int8x16_t a2_hi = vld1q_s8(A_masked + g2 * group_size + 16);
        int8x16_t a3_lo = vld1q_s8(A_masked + g3 * group_size);
        int8x16_t a3_hi = vld1q_s8(A_masked + g3 * group_size + 16);
        for (size_t p = 0; p < 4 && i + 4 + p < num_live; ++p)
            prefetch_group_head(live_groups[i + 4 + p]);

        for (size_t j = 0; j < slice_nb; ++j) {
            __builtin_prefetch(p0 + 8 * KMI_TILE, 0, 0);
            __builtin_prefetch(p1 + 8 * KMI_TILE, 0, 0);
            __builtin_prefetch(p2 + 8 * KMI_TILE, 0, 0);
            __builtin_prefetch(p3 + 8 * KMI_TILE, 0, 0);
            int32x4_t acc_0 = as_dot_group_one_row(p0, a0_lo, a0_hi);
            int32x4_t acc_1 = as_dot_group_one_row(p1, a1_lo, a1_hi);
            int32x4_t acc_2 = as_dot_group_one_row(p2, a2_lo, a2_hi);
            int32x4_t acc_3 = as_dot_group_one_row(p3, a3_lo, a3_hi);
            float32x4_t s0 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p0 + 64)));
            float32x4_t s1 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p1 + 64)));
            float32x4_t s2 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p2 + 64)));
            float32x4_t s3 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p3 + 64)));
            float32x4_t r = acc[j];
            r = vmlaq_f32(r, vcvtq_f32_s32(acc_0), s0);
            r = vmlaq_f32(r, vcvtq_f32_s32(acc_1), s1);
            r = vmlaq_f32(r, vcvtq_f32_s32(acc_2), s2);
            r = vmlaq_f32(r, vcvtq_f32_s32(acc_3), s3);
            acc[j] = r;
            p0 += KMI_TILE; p1 += KMI_TILE; p2 += KMI_TILE; p3 += KMI_TILE;
        }
    }
    for (; i + 1 < num_live; i += 2) {
        const size_t g0 = live_groups[i];
        const size_t g1 = live_groups[i + 1];
        const uint8_t* p0 = B_km_inline + g0 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
        const uint8_t* p1 = B_km_inline + g1 * N_blocks * KMI_TILE + nb_start * KMI_TILE;
        int8x16_t a0_lo = vld1q_s8(A_masked + g0 * group_size);
        int8x16_t a0_hi = vld1q_s8(A_masked + g0 * group_size + 16);
        int8x16_t a1_lo = vld1q_s8(A_masked + g1 * group_size);
        int8x16_t a1_hi = vld1q_s8(A_masked + g1 * group_size + 16);
        for (size_t j = 0; j < slice_nb; ++j) {
            int32x4_t acc_0 = as_dot_group_one_row(p0, a0_lo, a0_hi);
            int32x4_t acc_1 = as_dot_group_one_row(p1, a1_lo, a1_hi);
            float32x4_t s0 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p0 + 64)));
            float32x4_t s1 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p1 + 64)));
            float32x4_t r = acc[j];
            r = vmlaq_f32(r, vcvtq_f32_s32(acc_0), s0);
            r = vmlaq_f32(r, vcvtq_f32_s32(acc_1), s1);
            acc[j] = r;
            p0 += KMI_TILE; p1 += KMI_TILE;
        }
    }
    for (; i < num_live; ++i) {
        const size_t g = live_groups[i];
        const uint8_t* gp = B_km_inline + g * N_blocks * KMI_TILE;
        int8x16_t a_lo = vld1q_s8(A_masked + g * group_size);
        int8x16_t a_hi = vld1q_s8(A_masked + g * group_size + 16);
        kmi_accumulate_group(gp, slice_nb, nb_start, a_lo, a_hi, acc);
    }

    for (size_t j = 0; j < slice_nb; ++j) {
        size_t nb = nb_start + j;
        size_t ns = nb * 4;
        size_t actual_n = std::min(size_t(4), N - ns);
        float32x4_t result = vmulq_n_f32(acc[j], A_scale);
        float16x4_t result_f16 = vcvt_f16_f32(result);
        if (actual_n == 4) vst1_f16(C + ns, result_f16);
        else for (size_t ni = 0; ni < actual_n; ni++) {
            C[ns + ni] = vget_lane_f16(result_f16, 0);
            result_f16 = vext_f16(result_f16, result_f16, 1);
        }
    }
}
} // namespace

void cactus_gemv_int4_actsparse_kmi4_fast(
    const int8_t* A_masked, float A_scale,
    const uint8_t* B_km_inline,
    const uint16_t* live_groups, size_t num_live,
    __fp16* C, size_t K, size_t N, size_t group_size)
{
    if (K == 0 || N == 0) return;
    const size_t N_blocks = (N + 3) / 4;
    constexpr size_t ST_GATE = 384;
    if (N_blocks <= ST_GATE) {
        kmi4_fast_body(A_masked, A_scale, B_km_inline, live_groups, num_live,
                       C, N, N_blocks, group_size, 0, N_blocks);
        return;
    }
    auto& pool = CactusThreading::get_thread_pool();
    size_t num_threads = CactusThreading::GemmThreading::get_gemv_threads(N_blocks, pool.num_workers());
    num_threads = std::min(num_threads, N_blocks);
    if (num_threads <= 1) {
        kmi4_fast_body(A_masked, A_scale, B_km_inline, live_groups, num_live,
                       C, N, N_blocks, group_size, 0, N_blocks);
        return;
    }
    pool.enqueue_n_threads(N_blocks, num_threads, [=](size_t nb_start, size_t nb_end) {
        kmi4_fast_body(A_masked, A_scale, B_km_inline, live_groups, num_live,
                       C, N, N_blocks, group_size, nb_start, nb_end);
    });
    pool.wait_all();
}

// ---------------------------------------------------------------------------
// R5 — kmi4_v2: deeper prefetch + precomputed group pointers + cached A loads.
// ---------------------------------------------------------------------------
namespace {
struct KmiV2Scratch {
    std::vector<float32x4_t> acc;
    std::vector<const uint8_t*> group_ptrs;
    std::vector<int8x16_t> a_los;
    std::vector<int8x16_t> a_his;
};
inline KmiV2Scratch& kmi_v2_scratch() {
    thread_local KmiV2Scratch s;
    return s;
}

// Depth of prefetch in tiles (72 B each). 24 tiles ≈ 1.7 KB — enough to
// cover an L2-miss round trip (~300 cycles × 72 B/cycle peak bus ≈ 1.2 KB).
constexpr size_t KMI_V2_PF = 24;

inline void kmi4_v2_body(
    const int8_t* A_masked, float A_scale,
    const uint8_t* B_km_inline,
    const uint16_t* live_groups, size_t num_live,
    __fp16* C, size_t N, size_t N_blocks,
    size_t group_size, size_t nb_start, size_t nb_end)
{
    const size_t slice_nb = nb_end - nb_start;
    if (slice_nb == 0) return;

    auto& s = kmi_v2_scratch();
    if (s.acc.size() < slice_nb) s.acc.resize(slice_nb);
    if (s.group_ptrs.size() < num_live) s.group_ptrs.resize(num_live);
    if (s.a_los.size() < num_live) s.a_los.resize(num_live);
    if (s.a_his.size() < num_live) s.a_his.resize(num_live);

    float32x4_t* acc = s.acc.data();
    const uint8_t** gptr = s.group_ptrs.data();
    int8x16_t* a_los = s.a_los.data();
    int8x16_t* a_his = s.a_his.data();

    for (size_t j = 0; j < slice_nb; ++j) acc[j] = vdupq_n_f32(0.0f);

    // Precompute per-live-group pointers and A loads.
    for (size_t i = 0; i < num_live; ++i) {
        const size_t g = live_groups[i];
        gptr[i] = B_km_inline + g * N_blocks * KMI_TILE + nb_start * KMI_TILE;
        a_los[i] = vld1q_s8(A_masked + g * group_size);
        a_his[i] = vld1q_s8(A_masked + g * group_size + 16);
    }

    // Warm up the streams for the first 4 live groups.
    for (size_t i = 0; i < std::min<size_t>(4, num_live); ++i) {
        const uint8_t* gp = gptr[i];
        for (size_t o = 0; o < 8; ++o)
            __builtin_prefetch(gp + o * 128, 0, 0);
    }

    size_t i = 0;
    for (; i + 3 < num_live; i += 4) {
        const uint8_t* p0 = gptr[i];
        const uint8_t* p1 = gptr[i + 1];
        const uint8_t* p2 = gptr[i + 2];
        const uint8_t* p3 = gptr[i + 3];
        int8x16_t a0_lo = a_los[i];
        int8x16_t a0_hi = a_his[i];
        int8x16_t a1_lo = a_los[i + 1];
        int8x16_t a1_hi = a_his[i + 1];
        int8x16_t a2_lo = a_los[i + 2];
        int8x16_t a2_hi = a_his[i + 2];
        int8x16_t a3_lo = a_los[i + 3];
        int8x16_t a3_hi = a_his[i + 3];

        // Prefetch the next 4-group batch's heads.
        for (size_t k = 0; k < 4 && i + 4 + k < num_live; ++k) {
            const uint8_t* q = gptr[i + 4 + k];
            __builtin_prefetch(q + 0,   0, 0);
            __builtin_prefetch(q + 128, 0, 0);
            __builtin_prefetch(q + 256, 0, 0);
            __builtin_prefetch(q + 384, 0, 0);
        }

        for (size_t j = 0; j < slice_nb; ++j) {
            __builtin_prefetch(p0 + KMI_V2_PF * KMI_TILE, 0, 0);
            __builtin_prefetch(p1 + KMI_V2_PF * KMI_TILE, 0, 0);
            __builtin_prefetch(p2 + KMI_V2_PF * KMI_TILE, 0, 0);
            __builtin_prefetch(p3 + KMI_V2_PF * KMI_TILE, 0, 0);
            int32x4_t acc_0 = as_dot_group_one_row(p0, a0_lo, a0_hi);
            int32x4_t acc_1 = as_dot_group_one_row(p1, a1_lo, a1_hi);
            int32x4_t acc_2 = as_dot_group_one_row(p2, a2_lo, a2_hi);
            int32x4_t acc_3 = as_dot_group_one_row(p3, a3_lo, a3_hi);
            float32x4_t s0 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p0 + 64)));
            float32x4_t s1 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p1 + 64)));
            float32x4_t s2 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p2 + 64)));
            float32x4_t s3 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p3 + 64)));
            float32x4_t r = acc[j];
            r = vmlaq_f32(r, vcvtq_f32_s32(acc_0), s0);
            r = vmlaq_f32(r, vcvtq_f32_s32(acc_1), s1);
            r = vmlaq_f32(r, vcvtq_f32_s32(acc_2), s2);
            r = vmlaq_f32(r, vcvtq_f32_s32(acc_3), s3);
            acc[j] = r;
            p0 += KMI_TILE; p1 += KMI_TILE; p2 += KMI_TILE; p3 += KMI_TILE;
        }
    }
    for (; i + 1 < num_live; i += 2) {
        const uint8_t* p0 = gptr[i];
        const uint8_t* p1 = gptr[i + 1];
        int8x16_t a0_lo = a_los[i];
        int8x16_t a0_hi = a_his[i];
        int8x16_t a1_lo = a_los[i + 1];
        int8x16_t a1_hi = a_his[i + 1];
        for (size_t j = 0; j < slice_nb; ++j) {
            int32x4_t acc_0 = as_dot_group_one_row(p0, a0_lo, a0_hi);
            int32x4_t acc_1 = as_dot_group_one_row(p1, a1_lo, a1_hi);
            float32x4_t s0 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p0 + 64)));
            float32x4_t s1 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p1 + 64)));
            float32x4_t r = acc[j];
            r = vmlaq_f32(r, vcvtq_f32_s32(acc_0), s0);
            r = vmlaq_f32(r, vcvtq_f32_s32(acc_1), s1);
            acc[j] = r;
            p0 += KMI_TILE; p1 += KMI_TILE;
        }
    }
    for (; i < num_live; ++i) {
        const uint8_t* p = gptr[i];
        int8x16_t a_lo = a_los[i];
        int8x16_t a_hi = a_his[i];
        for (size_t j = 0; j < slice_nb; ++j) {
            int32x4_t acc_0 = as_dot_group_one_row(p, a_lo, a_hi);
            float32x4_t s0 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(p + 64)));
            acc[j] = vmlaq_f32(acc[j], vcvtq_f32_s32(acc_0), s0);
            p += KMI_TILE;
        }
    }

    for (size_t j = 0; j < slice_nb; ++j) {
        size_t nb = nb_start + j;
        size_t ns = nb * 4;
        size_t actual_n = std::min(size_t(4), N - ns);
        float32x4_t result = vmulq_n_f32(acc[j], A_scale);
        float16x4_t result_f16 = vcvt_f16_f32(result);
        if (actual_n == 4) vst1_f16(C + ns, result_f16);
        else for (size_t ni = 0; ni < actual_n; ni++) {
            C[ns + ni] = vget_lane_f16(result_f16, 0);
            result_f16 = vext_f16(result_f16, result_f16, 1);
        }
    }
}
} // namespace

void cactus_gemv_int4_actsparse_kmi4_v2(
    const int8_t* A_masked, float A_scale,
    const uint8_t* B_km_inline,
    const uint16_t* live_groups, size_t num_live,
    __fp16* C, size_t K, size_t N, size_t group_size)
{
    if (K == 0 || N == 0) return;
    const size_t N_blocks = (N + 3) / 4;
    constexpr size_t ST_GATE = 384;
    if (N_blocks <= ST_GATE) {
        kmi4_v2_body(A_masked, A_scale, B_km_inline, live_groups, num_live,
                     C, N, N_blocks, group_size, 0, N_blocks);
        return;
    }
    auto& pool = CactusThreading::get_thread_pool();
    size_t num_threads = CactusThreading::GemmThreading::get_gemv_threads(N_blocks, pool.num_workers());
    num_threads = std::min(num_threads, N_blocks);
    if (num_threads <= 1) {
        kmi4_v2_body(A_masked, A_scale, B_km_inline, live_groups, num_live,
                     C, N, N_blocks, group_size, 0, N_blocks);
        return;
    }
    pool.enqueue_n_threads(N_blocks, num_threads, [=](size_t nb_start, size_t nb_end) {
        kmi4_v2_body(A_masked, A_scale, B_km_inline, live_groups, num_live,
                     C, N, N_blocks, group_size, nb_start, nb_end);
    });
    pool.wait_all();
}

// ---------------------------------------------------------------------------
// N-sparse up-proj (Round 1: simple per-n_block loop over live N-groups).
// ---------------------------------------------------------------------------
static inline void nsparse_up_compute_nblock(
    const int8_t* A,
    const uint8_t* B_packed,
    const __fp16*  B_scales_row,     // scales for this n_block: [num_k_groups * 4]
    size_t         k_base_offset,    // always 0 here
    size_t         n_block,
    size_t         num_k_groups,
    size_t         K_group_size,
    __fp16*        C,
    size_t         n_start,
    size_t         actual_n,
    float          A_scale)
{
    (void)k_base_offset;
    float32x4_t running_sum = vdupq_n_f32(0.0f);
    for (size_t g = 0; g < num_k_groups; ++g) {
        const size_t k_base = g * K_group_size;
        const int8_t* a_ptr = A + k_base;
        const uint8_t* b_base = B_packed + (n_block * num_k_groups * K_group_size + k_base) * 2;
        __builtin_prefetch(b_base + K_group_size * 2, 0, 3);

        int32x4_t acc = vdupq_n_s32(0);
        int8x16_t a_lo = vld1q_s8(a_ptr);
        int8x16_t a_hi = vld1q_s8(a_ptr + 16);
        int8x16_t b0, b1, b2, b3;
        as_unpack_nibbles(b_base,      b1, b0);
        as_unpack_nibbles(b_base + 16, b3, b2);
        acc = AS_DOTQ_LANE(acc, b0, a_lo, 0);
        acc = AS_DOTQ_LANE(acc, b1, a_lo, 1);
        acc = AS_DOTQ_LANE(acc, b2, a_lo, 2);
        acc = AS_DOTQ_LANE(acc, b3, a_lo, 3);
        as_unpack_nibbles(b_base + 32, b1, b0);
        as_unpack_nibbles(b_base + 48, b3, b2);
        acc = AS_DOTQ_LANE(acc, b0, a_hi, 0);
        acc = AS_DOTQ_LANE(acc, b1, a_hi, 1);
        acc = AS_DOTQ_LANE(acc, b2, a_hi, 2);
        acc = AS_DOTQ_LANE(acc, b3, a_hi, 3);

        float32x4_t scales = vcvt_f32_f16(vld1_f16(B_scales_row + g * 4));
        running_sum = vmlaq_f32(running_sum, vcvtq_f32_s32(acc), scales);
    }
    float32x4_t result = vmulq_n_f32(running_sum, A_scale);
    float16x4_t result_f16 = vcvt_f16_f32(result);
    if (actual_n == 4) {
        vst1_f16(C + n_start, result_f16);
    } else {
        for (size_t ni = 0; ni < actual_n; ++ni) {
            C[n_start + ni] = vget_lane_f16(result_f16, 0);
            result_f16 = vext_f16(result_f16, result_f16, 1);
        }
    }
}

void cactus_gemv_int4_nsparse_up(
    const int8_t* A, float A_scale,
    const int8_t* B_packed_raw, const __fp16* B_scales,
    const uint16_t* live_N_groups, size_t num_live,
    __fp16* C, size_t K, size_t N,
    size_t K_group_size, size_t N_group_size)
{
    const uint8_t* B_packed = reinterpret_cast<const uint8_t*>(B_packed_raw);
    if (K == 0 || N == 0 || num_live == 0) return;
    if (N_group_size % 4 != 0) return;  // must pack into whole n_blocks
    const size_t num_k_groups     = K / K_group_size;
    const size_t nblocks_per_ngrp = N_group_size / 4;
    const size_t N_blocks_total   = (N + 3) / 4;

    auto process_live_range = [=](size_t ig_start, size_t ig_end) {
        for (size_t ig = ig_start; ig < ig_end; ++ig) {
            const size_t Ng = live_N_groups[ig];
            const size_t nb_start = Ng * nblocks_per_ngrp;
            const size_t nb_end   = std::min(nb_start + nblocks_per_ngrp, N_blocks_total);
            for (size_t nb = nb_start; nb < nb_end; ++nb) {
                const size_t n_start  = nb * 4;
                const size_t actual_n = std::min(size_t(4), N - n_start);
                const __fp16* scale_row = B_scales + nb * num_k_groups * 4;
                nsparse_up_compute_nblock(
                    A, B_packed, scale_row, 0, nb,
                    num_k_groups, K_group_size,
                    C, n_start, actual_n, A_scale);
            }
        }
    };

    auto& pool = CactusThreading::get_thread_pool();
    const size_t work_total = num_live * nblocks_per_ngrp;
    size_t num_threads = CactusThreading::GemmThreading::get_gemv_threads(work_total, pool.num_workers());
    num_threads = std::min(num_threads, num_live);
    if (num_threads <= 1) {
        process_live_range(0, num_live);
    } else {
        pool.enqueue_n_threads(num_live, num_threads, process_live_range);
        pool.wait_all();
    }
}
