#include "kernel.h"
#include "kernel_utils.h"
#include "../graph/graph.h"
#include <arm_neon.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>
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

static const std::array<int8_t, 16> kTernaryPacked2DecodeNibbleLo = []() {
    std::array<int8_t, 16> lut{};
    for (size_t nibble = 0; nibble < lut.size(); ++nibble) {
        const uint8_t code = static_cast<uint8_t>(nibble & 0x3u);
        lut[nibble] = code == 0x1u ? static_cast<int8_t>(1)
                     : code == 0x2u ? static_cast<int8_t>(-1)
                                    : static_cast<int8_t>(0);
    }
    return lut;
}();

static const std::array<int8_t, 16> kTernaryPacked2DecodeNibbleHi = []() {
    std::array<int8_t, 16> lut{};
    for (size_t nibble = 0; nibble < lut.size(); ++nibble) {
        const uint8_t code = static_cast<uint8_t>((nibble >> 2) & 0x3u);
        lut[nibble] = code == 0x1u ? static_cast<int8_t>(1)
                     : code == 0x2u ? static_cast<int8_t>(-1)
                                    : static_cast<int8_t>(0);
    }
    return lut;
}();

static inline void unpack_ternary_packed2_as_int8x16x4(
    const uint8_t* src,
    int8x16_t& out0,
    int8x16_t& out1,
    int8x16_t& out2,
    int8x16_t& out3
) {
    const uint8x16_t packed = vld1q_u8(src);
    const uint8x16_t low_nibbles = vandq_u8(packed, vdupq_n_u8(0x0Fu));
    const uint8x16_t high_nibbles = vshrq_n_u8(packed, 4);
    const int8x16_t lut_lo = vld1q_s8(kTernaryPacked2DecodeNibbleLo.data());
    const int8x16_t lut_hi = vld1q_s8(kTernaryPacked2DecodeNibbleHi.data());

    const int8x16_t d0 = vqtbl1q_s8(lut_lo, low_nibbles);
    const int8x16_t d1 = vqtbl1q_s8(lut_hi, low_nibbles);
    const int8x16_t d2 = vqtbl1q_s8(lut_lo, high_nibbles);
    const int8x16_t d3 = vqtbl1q_s8(lut_hi, high_nibbles);

    const int16x8_t pair01_lo = vreinterpretq_s16_s8(vzip1q_s8(d0, d1));
    const int16x8_t pair01_hi = vreinterpretq_s16_s8(vzip2q_s8(d0, d1));
    const int16x8_t pair23_lo = vreinterpretq_s16_s8(vzip1q_s8(d2, d3));
    const int16x8_t pair23_hi = vreinterpretq_s16_s8(vzip2q_s8(d2, d3));

    out0 = vreinterpretq_s8_s16(vzip1q_s16(pair01_lo, pair23_lo));
    out1 = vreinterpretq_s8_s16(vzip2q_s16(pair01_lo, pair23_lo));
    out2 = vreinterpretq_s8_s16(vzip1q_s16(pair01_hi, pair23_hi));
    out3 = vreinterpretq_s8_s16(vzip2q_s16(pair01_hi, pair23_hi));
}

static inline float ternary_output_scale_scalar(
    TernaryScaleMode scale_mode,
    const __fp16* scales,
    size_t n_idx
) {
    switch (scale_mode) {
        case TernaryScaleMode::COLUMN:
            return 1.0f;
        case TernaryScaleMode::TENSOR:
            return static_cast<float>(scales[0]);
        case TernaryScaleMode::ROW:
            return static_cast<float>(scales[n_idx]);
        case TernaryScaleMode::NONE:
            throw std::runtime_error("Packed ternary matmul requires ternary scale mode metadata");
    }
    return 1.0f;
}

static inline float32x4_t ternary_output_scale_vector(
    TernaryScaleMode scale_mode,
    const __fp16* scales,
    size_t n_start
) {
    switch (scale_mode) {
        case TernaryScaleMode::COLUMN:
            return vdupq_n_f32(1.0f);
        case TernaryScaleMode::TENSOR:
            return vdupq_n_f32(static_cast<float>(scales[0]));
        case TernaryScaleMode::ROW:
            return vcvt_f32_f16(vld1_f16(scales + n_start));
        case TernaryScaleMode::NONE:
            throw std::runtime_error("Packed ternary matmul requires ternary scale mode metadata");
    }
    return vdupq_n_f32(1.0f);
}

struct TernaryLutBytes {
    uint8x16_t lo;
    uint8x16_t hi;
};

struct TernaryLut4 {
    TernaryLutBytes pair01;
    TernaryLutBytes pair23;
};

static inline int16_t ternary_pair_sum(uint8_t packed_codes, int8_t a0, int8_t a1) {
    const uint8_t code0 = packed_codes & 0x3u;
    const uint8_t code1 = (packed_codes >> 2) & 0x3u;
    const int16_t s0 = code0 == 0x1u ? static_cast<int16_t>(a0)
                     : code0 == 0x2u ? static_cast<int16_t>(-a0)
                                     : static_cast<int16_t>(0);
    const int16_t s1 = code1 == 0x1u ? static_cast<int16_t>(a1)
                     : code1 == 0x2u ? static_cast<int16_t>(-a1)
                                     : static_cast<int16_t>(0);
    return static_cast<int16_t>(s0 + s1);
}

static inline TernaryLutBytes build_ternary_pair_lut(int8_t a0, int8_t a1) {
    alignas(16) uint8_t lo[16];
    alignas(16) uint8_t hi[16];

    for (size_t code = 0; code < 16; ++code) {
        const uint16_t value = static_cast<uint16_t>(ternary_pair_sum(
            static_cast<uint8_t>(code), a0, a1));
        lo[code] = static_cast<uint8_t>(value & 0xFFu);
        hi[code] = static_cast<uint8_t>((value >> 8) & 0xFFu);
    }

    return {vld1q_u8(lo), vld1q_u8(hi)};
}

static inline TernaryLut4 build_ternary_lut4(const int8_t* a_ptr) {
    return {
        build_ternary_pair_lut(a_ptr[0], a_ptr[1]),
        build_ternary_pair_lut(a_ptr[2], a_ptr[3])
    };
}

static inline void accumulate_ternary_lut_tile16(
    uint8x16_t codes,
    const TernaryLut4& lut,
    int32x4_t& acc0,
    int32x4_t& acc1,
    int32x4_t& acc2,
    int32x4_t& acc3
) {
    const uint8x16_t idx01 = vandq_u8(codes, vdupq_n_u8(0x0Fu));
    const uint8x16_t idx23 = vshrq_n_u8(codes, 4);

    const uint8x16_t sum01_lo_bytes = vqtbl1q_u8(lut.pair01.lo, idx01);
    const uint8x16_t sum01_hi_bytes = vqtbl1q_u8(lut.pair01.hi, idx01);
    const uint8x16_t sum23_lo_bytes = vqtbl1q_u8(lut.pair23.lo, idx23);
    const uint8x16_t sum23_hi_bytes = vqtbl1q_u8(lut.pair23.hi, idx23);

    const int16x8_t sum01_lo = vreinterpretq_s16_u8(vzip1q_u8(sum01_lo_bytes, sum01_hi_bytes));
    const int16x8_t sum01_hi = vreinterpretq_s16_u8(vzip2q_u8(sum01_lo_bytes, sum01_hi_bytes));
    const int16x8_t sum23_lo = vreinterpretq_s16_u8(vzip1q_u8(sum23_lo_bytes, sum23_hi_bytes));
    const int16x8_t sum23_hi = vreinterpretq_s16_u8(vzip2q_u8(sum23_lo_bytes, sum23_hi_bytes));

    const int16x8_t total_lo = vaddq_s16(sum01_lo, sum23_lo);
    const int16x8_t total_hi = vaddq_s16(sum01_hi, sum23_hi);

    acc0 = vaddw_s16(acc0, vget_low_s16(total_lo));
    acc1 = vaddw_s16(acc1, vget_high_s16(total_lo));
    acc2 = vaddw_s16(acc2, vget_low_s16(total_hi));
    acc3 = vaddw_s16(acc3, vget_high_s16(total_hi));
}

template<size_t K_QUADS>
static inline void accumulate_ternary_lut_kquads_fixed(
    const TernaryLut4* luts,
    const uint8_t* B_codes,
    size_t N_padded,
    size_t n_start,
    int32x4_t& acc0,
    int32x4_t& acc1,
    int32x4_t& acc2,
    int32x4_t& acc3
) {
    for (size_t q = 0; q < K_QUADS; ++q) {
        const uint8x16_t codes = vld1q_u8(B_codes + q * N_padded + n_start);
        accumulate_ternary_lut_tile16(codes, luts[q], acc0, acc1, acc2, acc3);
    }
}

static inline void accumulate_ternary_lut_kquads_dynamic(
    const TernaryLut4* luts,
    size_t k_quads,
    const uint8_t* B_codes,
    size_t N_padded,
    size_t n_start,
    int32x4_t& acc0,
    int32x4_t& acc1,
    int32x4_t& acc2,
    int32x4_t& acc3
) {
    for (size_t q = 0; q < k_quads; ++q) {
        const uint8x16_t codes = vld1q_u8(B_codes + q * N_padded + n_start);
        accumulate_ternary_lut_tile16(codes, luts[q], acc0, acc1, acc2, acc3);
    }
}

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

#if defined(CACTUS_COMPILE_SME2)
constexpr size_t SME2_M_THRESHOLD = 4;

void cactus_matmul_f16_sme2_caller(
	const __fp16* a,
	const __fp16* b_transposed,
	__fp16* c,
	size_t M,
	size_t K,
	size_t N
);
#endif

void cactus_matmul_f16(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N
) {

#if defined(CACTUS_COMPILE_SME2)
	if (cpu_has_sme2() && M >= SME2_M_THRESHOLD) {
		cactus_matmul_f16_sme2_caller(
			a, b_transposed, c,
			M, K, N
		);
		return;
	}
#endif

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

void cactus_gemv_ternary_packed2(
    const int8_t* A,
    float A_scale,
    const uint8_t* B_packed,
    const __fp16* B_scales,
    TernaryScaleMode scale_mode,
    __fp16* C,
    size_t K, size_t N,
    size_t group_size
) {
    if (K == 0 || N == 0) return;
    if ((group_size % 32) != 0) {
        throw std::runtime_error("Packed ternary matmul requires group_size to be a multiple of 32");
    }
    if (scale_mode != TernaryScaleMode::COLUMN && B_scales == nullptr) {
        throw std::runtime_error("Packed ternary matmul requires raw tensor/row scales");
    }

    const size_t num_groups = K / group_size;
    const size_t N_blocks = (N + 3) / 4;
    auto process_blocks = [=](size_t block_start, size_t block_end) {
        size_t n_block = block_start;

        for (; n_block + 1 < block_end; n_block += 2) {
            int32x4_t sum_a = vdupq_n_s32(0);
            int32x4_t sum_b = vdupq_n_s32(0);

            for (size_t g = 0; g < num_groups; ++g) {
                const size_t k_base = g * group_size;
                const int8_t* a_ptr = A + k_base;
                const uint8_t* ba = B_packed + (n_block * K + k_base);
                const uint8_t* bb = B_packed + ((n_block + 1) * K + k_base);

                int32x4_t acc_a = vdupq_n_s32(0);
                int32x4_t acc_b = vdupq_n_s32(0);

                for (size_t chunk = 0; chunk < group_size; chunk += 32) {
                    const int8_t* a_chunk = a_ptr + chunk;
                    const uint8_t* ba_chunk = ba + chunk;
                    const uint8_t* bb_chunk = bb + chunk;

                    const int8x16_t a_lo = vld1q_s8(a_chunk);
                    const int8x16_t a_hi = vld1q_s8(a_chunk + 16);

                    int8x16_t b0, b1, b2, b3;
                    unpack_ternary_packed2_as_int8x16x4(ba_chunk, b0, b1, b2, b3);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b0, a_lo, 0);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b1, a_lo, 1);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b2, a_lo, 2);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b3, a_lo, 3);
                    unpack_ternary_packed2_as_int8x16x4(ba_chunk + 16, b0, b1, b2, b3);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b0, a_hi, 0);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b1, a_hi, 1);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b2, a_hi, 2);
                    acc_a = CACTUS_DOTQ_LANE(acc_a, b3, a_hi, 3);

                    unpack_ternary_packed2_as_int8x16x4(bb_chunk, b0, b1, b2, b3);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b0, a_lo, 0);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b1, a_lo, 1);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b2, a_lo, 2);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b3, a_lo, 3);
                    unpack_ternary_packed2_as_int8x16x4(bb_chunk + 16, b0, b1, b2, b3);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b0, a_hi, 0);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b1, a_hi, 1);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b2, a_hi, 2);
                    acc_b = CACTUS_DOTQ_LANE(acc_b, b3, a_hi, 3);
                }

                sum_a = vaddq_s32(sum_a, acc_a);
                sum_b = vaddq_s32(sum_b, acc_b);
            }

            const size_t n_start_a = n_block * 4;
            const size_t n_start_b = (n_block + 1) * 4;
            const float32x4_t out_scale_a = ternary_output_scale_vector(scale_mode, B_scales, n_start_a);
            const float32x4_t result_a = vmulq_f32(vmulq_n_f32(vcvtq_f32_s32(sum_a), A_scale), out_scale_a);
            vst1_f16(C + n_start_a, vcvt_f16_f32(result_a));

            const size_t actual_n_b = std::min(size_t(4), N - n_start_b);
            if (actual_n_b == 4) {
                const float32x4_t out_scale_b = ternary_output_scale_vector(scale_mode, B_scales, n_start_b);
                const float32x4_t result_b = vmulq_f32(vmulq_n_f32(vcvtq_f32_s32(sum_b), A_scale), out_scale_b);
                vst1_f16(C + n_start_b, vcvt_f16_f32(result_b));
            } else {
                alignas(16) int32_t sum_b_lanes[4];
                vst1q_s32(sum_b_lanes, sum_b);
                for (size_t ni = 0; ni < actual_n_b; ++ni) {
                    const float scale = A_scale * ternary_output_scale_scalar(scale_mode, B_scales, n_start_b + ni);
                    C[n_start_b + ni] = static_cast<__fp16>(
                        static_cast<float>(sum_b_lanes[ni]) * scale);
                }
            }
        }

        for (; n_block < block_end; ++n_block) {
            const size_t n_start = n_block * 4;
            const size_t actual_n = std::min(size_t(4), N - n_start);
            int32x4_t running_sum = vdupq_n_s32(0);

            for (size_t g = 0; g < num_groups; ++g) {
                const size_t k_base = g * group_size;
                const int8_t* a_ptr = A + k_base;
                const uint8_t* b_base = B_packed + (n_block * K + k_base);

                __builtin_prefetch(b_base + group_size, 0, 3);

                int32x4_t acc = vdupq_n_s32(0);

                for (size_t chunk = 0; chunk < group_size; chunk += 32) {
                    const int8_t* a_chunk = a_ptr + chunk;
                    const uint8_t* b_chunk = b_base + chunk;

                    int8x16_t a_lo = vld1q_s8(a_chunk);
                    int8x16_t b0, b1, b2, b3;
                    unpack_ternary_packed2_as_int8x16x4(b_chunk, b0, b1, b2, b3);

                    acc = CACTUS_DOTQ_LANE(acc, b0, a_lo, 0);
                    acc = CACTUS_DOTQ_LANE(acc, b1, a_lo, 1);
                    acc = CACTUS_DOTQ_LANE(acc, b2, a_lo, 2);
                    acc = CACTUS_DOTQ_LANE(acc, b3, a_lo, 3);

                    int8x16_t a_hi = vld1q_s8(a_chunk + 16);
                    unpack_ternary_packed2_as_int8x16x4(b_chunk + 16, b0, b1, b2, b3);

                    acc = CACTUS_DOTQ_LANE(acc, b0, a_hi, 0);
                    acc = CACTUS_DOTQ_LANE(acc, b1, a_hi, 1);
                    acc = CACTUS_DOTQ_LANE(acc, b2, a_hi, 2);
                    acc = CACTUS_DOTQ_LANE(acc, b3, a_hi, 3);
                }

                running_sum = vaddq_s32(running_sum, acc);
            }

            if (actual_n == 4) {
                const float32x4_t out_scale = ternary_output_scale_vector(scale_mode, B_scales, n_start);
                float16x4_t result_f16 = vcvt_f16_f32(
                    vmulq_f32(vmulq_n_f32(vcvtq_f32_s32(running_sum), A_scale), out_scale));
                vst1_f16(C + n_start, result_f16);
            } else {
                alignas(16) int32_t running_sum_lanes[4];
                vst1q_s32(running_sum_lanes, running_sum);
                for (size_t ni = 0; ni < actual_n; ++ni) {
                    const float scale = A_scale * ternary_output_scale_scalar(scale_mode, B_scales, n_start + ni);
                    C[n_start + ni] = static_cast<__fp16>(
                        static_cast<float>(running_sum_lanes[ni]) * scale);
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

void cactus_gemv_ternary_lut16_benchmark(
    const int8_t* A,
    float A_scale,
    const uint8_t* B_codes,
    const __fp16* row_scales,
    float tensor_scale,
    __fp16* C,
    size_t K,
    size_t N,
    size_t N_padded
) {
    if (K == 0 || N == 0) return;
    if ((K % 32) != 0) {
        throw std::runtime_error("Ternary LUT GEMV requires K padded to a multiple of 32");
    }
    if ((N_padded % 16) != 0 || N_padded < N) {
        throw std::runtime_error("Ternary LUT GEMV requires N padded to a multiple of 16");
    }

    constexpr size_t TILE_N = 16;
    const size_t k_quads = K / 4;
    const size_t num_tiles = N_padded / TILE_N;
    std::vector<TernaryLut4> luts(k_quads);
    for (size_t q = 0; q < k_quads; ++q) {
        luts[q] = build_ternary_lut4(A + q * 4);
    }
    const TernaryLut4* luts_ptr = luts.data();

    auto process_tiles = [=](size_t tile_start, size_t tile_end) {
        std::vector<int32_t> accum((tile_end - tile_start) * TILE_N, 0);

        const float base_scale = A_scale * tensor_scale;
        for (size_t tile = tile_start; tile < tile_end; ++tile) {
            const size_t n_start = tile * TILE_N;
            int32_t* tile_acc = accum.data() + (tile - tile_start) * TILE_N;
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            switch (k_quads) {
                case 256:
                    accumulate_ternary_lut_kquads_fixed<256>(
                        luts_ptr, B_codes, N_padded, n_start, acc0, acc1, acc2, acc3);
                    break;
                case 512:
                    accumulate_ternary_lut_kquads_fixed<512>(
                        luts_ptr, B_codes, N_padded, n_start, acc0, acc1, acc2, acc3);
                    break;
                case 1536:
                    accumulate_ternary_lut_kquads_fixed<1536>(
                        luts_ptr, B_codes, N_padded, n_start, acc0, acc1, acc2, acc3);
                    break;
                default:
                    accumulate_ternary_lut_kquads_dynamic(
                        luts_ptr, k_quads, B_codes, N_padded, n_start, acc0, acc1, acc2, acc3);
                    break;
            }

            vst1q_s32(tile_acc + 0, acc0);
            vst1q_s32(tile_acc + 4, acc1);
            vst1q_s32(tile_acc + 8, acc2);
            vst1q_s32(tile_acc + 12, acc3);

            const size_t actual_n = std::min(TILE_N, N - n_start);

            if (actual_n == TILE_N) {
                float32x4_t out0 = vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(tile_acc + 0)), base_scale);
                float32x4_t out1 = vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(tile_acc + 4)), base_scale);
                float32x4_t out2 = vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(tile_acc + 8)), base_scale);
                float32x4_t out3 = vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(tile_acc + 12)), base_scale);

                if (row_scales != nullptr) {
                    out0 = vmulq_f32(out0, vcvt_f32_f16(vld1_f16(row_scales + n_start + 0)));
                    out1 = vmulq_f32(out1, vcvt_f32_f16(vld1_f16(row_scales + n_start + 4)));
                    out2 = vmulq_f32(out2, vcvt_f32_f16(vld1_f16(row_scales + n_start + 8)));
                    out3 = vmulq_f32(out3, vcvt_f32_f16(vld1_f16(row_scales + n_start + 12)));
                }

                vst1_f16(C + n_start + 0, vcvt_f16_f32(out0));
                vst1_f16(C + n_start + 4, vcvt_f16_f32(out1));
                vst1_f16(C + n_start + 8, vcvt_f16_f32(out2));
                vst1_f16(C + n_start + 12, vcvt_f16_f32(out3));
                continue;
            }

            for (size_t ni = 0; ni < actual_n; ++ni) {
                float scale = base_scale;
                if (row_scales != nullptr) {
                    scale *= static_cast<float>(row_scales[n_start + ni]);
                }
                C[n_start + ni] = static_cast<__fp16>(static_cast<float>(tile_acc[ni]) * scale);
            }
        }
    };

    auto& pool = CactusThreading::get_thread_pool();
    size_t num_threads = CactusThreading::GemmThreading::get_gemv_threads(num_tiles, pool.num_workers());
    num_threads = std::min(num_threads, num_tiles);

    if (num_threads <= 1) {
        process_tiles(0, num_tiles);
    } else {
        pool.enqueue_n_threads(num_tiles, num_threads, process_tiles);
        pool.wait_all();
    }
}

void cactus_gemm_ternary_packed2(
    const int8_t* A,
    const float* A_scales,
    const uint8_t* B_packed,
    const __fp16* B_scales,
    TernaryScaleMode scale_mode,
    __fp16* C,
    size_t M, size_t K, size_t N,
    size_t group_size
) {
    if (M == 0 || K == 0 || N == 0) return;
    if ((group_size % 32) != 0) {
        throw std::runtime_error("Packed ternary matmul requires group_size to be a multiple of 32");
    }
    if (scale_mode != TernaryScaleMode::COLUMN && B_scales == nullptr) {
        throw std::runtime_error("Packed ternary matmul requires raw tensor/row scales");
    }
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 4;

    const size_t num_groups = K / group_size;
    const size_t N_blocks = (N + TILE_N - 1) / TILE_N;
    const size_t num_row_tiles = (M + TILE_M - 1) / TILE_M;
    const size_t total_tiles = num_row_tiles * N_blocks;

    auto process_tiles = [=](size_t tile_start, size_t tile_end) {
        for (size_t tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
            const size_t tile_row = tile_idx / N_blocks;
            const size_t n_block = tile_idx % N_blocks;
            const size_t m_start = tile_row * TILE_M;
            const size_t m_end = std::min(m_start + TILE_M, M);
            const size_t n_start = n_block * TILE_N;
            const size_t actual_m = m_end - m_start;
            const size_t actual_n = std::min(size_t(4), N - n_start);

            const int8_t* a_rows[TILE_M];
            for (size_t mi = 0; mi < TILE_M; ++mi) {
                const size_t row = m_start + (mi < actual_m ? mi : actual_m - 1);
                a_rows[mi] = A + row * K;
            }

            int32x4_t running_sum[TILE_M] = {
                vdupq_n_s32(0), vdupq_n_s32(0),
                vdupq_n_s32(0), vdupq_n_s32(0)
            };

            for (size_t g = 0; g < num_groups; ++g) {
                const size_t k_base = g * group_size;
                const uint8_t* b_base = B_packed + (n_block * K + k_base);
                __builtin_prefetch(b_base + group_size, 0, 3);

                for (size_t chunk = 0; chunk < group_size; chunk += 32) {
                    const uint8_t* b_chunk = b_base + chunk;
                    int8x16_t b0, b1, b2, b3;
                    int8x16_t b4, b5, b6, b7;
                    unpack_ternary_packed2_as_int8x16x4(b_chunk, b0, b1, b2, b3);
                    unpack_ternary_packed2_as_int8x16x4(b_chunk + 16, b4, b5, b6, b7);

                    #define CACTUS_TERNARY_GEMM_ROW(ROW) do { \
                        const int8_t* a_ptr_##ROW = a_rows[ROW] + k_base + chunk; \
                        int32x4_t acc_##ROW = vdupq_n_s32(0); \
                        const int8x16_t a_lo_##ROW = vld1q_s8(a_ptr_##ROW); \
                        const int8x16_t a_hi_##ROW = vld1q_s8(a_ptr_##ROW + 16); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b0, a_lo_##ROW, 0); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b1, a_lo_##ROW, 1); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b2, a_lo_##ROW, 2); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b3, a_lo_##ROW, 3); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b4, a_hi_##ROW, 0); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b5, a_hi_##ROW, 1); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b6, a_hi_##ROW, 2); \
                        acc_##ROW = CACTUS_DOTQ_LANE(acc_##ROW, b7, a_hi_##ROW, 3); \
                        running_sum[ROW] = vaddq_s32(running_sum[ROW], acc_##ROW); \
                    } while (0)

                    CACTUS_TERNARY_GEMM_ROW(0);
                    CACTUS_TERNARY_GEMM_ROW(1);
                    CACTUS_TERNARY_GEMM_ROW(2);
                    CACTUS_TERNARY_GEMM_ROW(3);
                    #undef CACTUS_TERNARY_GEMM_ROW
                }
            }

            for (size_t mi = 0; mi < actual_m; ++mi) {
                if (actual_n == 4) {
                    const float32x4_t out_scale = ternary_output_scale_vector(scale_mode, B_scales, n_start);
                    const float32x4_t result = vmulq_f32(
                        vmulq_n_f32(vcvtq_f32_s32(running_sum[mi]), A_scales[m_start + mi]),
                        out_scale);
                    float16x4_t result_f16 = vcvt_f16_f32(result);
                    vst1_f16(C + (m_start + mi) * N + n_start, result_f16);
                } else {
                    alignas(16) int32_t running_sum_lanes[4];
                    vst1q_s32(running_sum_lanes, running_sum[mi]);
                    for (size_t ni = 0; ni < actual_n; ++ni) {
                        const float scale = A_scales[m_start + mi] *
                            ternary_output_scale_scalar(scale_mode, B_scales, n_start + ni);
                        C[(m_start + mi) * N + n_start + ni] = static_cast<__fp16>(
                            static_cast<float>(running_sum_lanes[ni]) * scale);
                    }
                }
            }
        }
    };

    CactusThreading::parallel_gemm_tiles(M, total_tiles, process_tiles);
}

void cactus_matmul_ternary_packed2(const int8_t* A, const float* A_scales,
                                   const uint8_t* B_packed, const __fp16* B_scales,
                                   TernaryScaleMode scale_mode,
                                   __fp16* C, size_t M, size_t K, size_t N, size_t group_size) {
    if (M == 0 || K == 0 || N == 0) return;

    if (M == 1) {
        cactus_gemv_ternary_packed2(A, A_scales[0], B_packed, B_scales, scale_mode, C, K, N, group_size);
    } else {
        cactus_gemm_ternary_packed2(A, A_scales, B_packed, B_scales, scale_mode, C, M, K, N, group_size);
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
