#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cstring>
#include <algorithm>
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

        for (size_t i = 0; i < c_len; i++) c[i] = (__fp16)C_f32[i];
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
