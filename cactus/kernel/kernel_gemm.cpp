#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <cmath>

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

                    acc[0][ni] = vfmaq_f16_compat(acc[0][ni], a0_lo, b_lo);
                    acc[0][ni] = vfmaq_f16_compat(acc[0][ni], a0_hi, b_hi);
                    acc[1][ni] = vfmaq_f16_compat(acc[1][ni], a1_lo, b_lo);
                    acc[1][ni] = vfmaq_f16_compat(acc[1][ni], a1_hi, b_hi);
                    acc[2][ni] = vfmaq_f16_compat(acc[2][ni], a2_lo, b_lo);
                    acc[2][ni] = vfmaq_f16_compat(acc[2][ni], a2_hi, b_hi);
                    acc[3][ni] = vfmaq_f16_compat(acc[3][ni], a3_lo, b_lo);
                    acc[3][ni] = vfmaq_f16_compat(acc[3][ni], a3_hi, b_hi);
                }
            }

            for (size_t k = K16; k < K8; k += 8) {
                float16x8_t a0_v = (row_block < m_end) ? vld1q_f16(a + row_block * K + k) : vdupq_n_f16(0);
                float16x8_t a1_v = (row_block + 1 < m_end) ? vld1q_f16(a + (row_block + 1) * K + k) : vdupq_n_f16(0);
                float16x8_t a2_v = (row_block + 2 < m_end) ? vld1q_f16(a + (row_block + 2) * K + k) : vdupq_n_f16(0);
                float16x8_t a3_v = (row_block + 3 < m_end) ? vld1q_f16(a + (row_block + 3) * K + k) : vdupq_n_f16(0);

                for (size_t ni = 0; ni < TILE_N && col_block + ni < n_end; ++ni) {
                    float16x8_t b_v = vld1q_f16(b_transposed + (col_block + ni) * K + k);
                    acc[0][ni] = vfmaq_f16_compat(acc[0][ni], a0_v, b_v);
                    acc[1][ni] = vfmaq_f16_compat(acc[1][ni], a1_v, b_v);
                    acc[2][ni] = vfmaq_f16_compat(acc[2][ni], a2_v, b_v);
                    acc[3][ni] = vfmaq_f16_compat(acc[3][ni], a3_v, b_v);
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

    #if defined(__APPLE__) && defined(__arm64__)
      constexpr size_t TILE_M = 4;
      constexpr size_t TILE_N = 8;
    #else
      constexpr size_t TILE_M = 4;
      constexpr size_t TILE_N = 4;
    #endif

    const size_t num_groups = K / group_size;
    const size_t num_row_tiles = (M + TILE_M - 1) / TILE_M;
    const size_t num_col_tiles = (N + TILE_N - 1) / TILE_N;
    const size_t total_tiles = num_row_tiles * num_col_tiles;

    CactusThreading::parallel_gemm_tiles(M, total_tiles,
        [=](size_t tile_start, size_t tile_end) {
            for (size_t tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
            const size_t tile_row = tile_idx / num_col_tiles;
            const size_t tile_col = tile_idx % num_col_tiles;
            const size_t m_start = tile_row * TILE_M;
            const size_t m_end = std::min(m_start + TILE_M, M);
            const size_t n_start = tile_col * TILE_N;
            const size_t n_end = std::min(n_start + TILE_N, N);
            const size_t actual_m = m_end - m_start;
            const size_t actual_n = n_end - n_start;

            float running_sum[TILE_M][TILE_N] = {{0.0f}};

            for (size_t ni = 0; ni < actual_n; ni++) {
                __builtin_prefetch(&B_scales[(n_start + ni) * num_groups], 0, 3);
            }

            for (size_t g = 0; g < num_groups; g++) {
                const size_t k_base = g * group_size;

                int32_t group_acc[TILE_M][TILE_N] = {{0}};

                if (cactus_has_i8mm()) {
                    size_t mi = 0;
                    for (; mi + 1 < actual_m; mi += 2) {
                        size_t ni = 0;
                        for (; ni + 3 < actual_n; ni += 4) {
                            int32x4_t acc01 = vdupq_n_s32(0);
                            int32x4_t acc23 = vdupq_n_s32(0);

                            const int8_t* a_base0 = A + (m_start + mi) * K + k_base;
                            const int8_t* a_base1 = A + (m_start + mi + 1) * K + k_base;
                            const int8_t* b_base0 = B + (n_start + ni) * K + k_base;
                            const int8_t* b_base1 = B + (n_start + ni + 1) * K + k_base;
                            const int8_t* b_base2 = B + (n_start + ni + 2) * K + k_base;
                            const int8_t* b_base3 = B + (n_start + ni + 3) * K + k_base;

                            for (size_t k_offset = 0; k_offset < group_size; k_offset += 64) {
                                #pragma unroll
                                for (int kk = 0; kk < 64; kk += 16) {
                                    int8x16_t a_lo = vcombine_s8(vld1_s8(a_base0 + k_offset + kk),
                                                                vld1_s8(a_base1 + k_offset + kk));
                                    int8x16_t a_hi = vcombine_s8(vld1_s8(a_base0 + k_offset + kk + 8),
                                                                vld1_s8(a_base1 + k_offset + kk + 8));

                                    int8x16_t b01_lo = vcombine_s8(vld1_s8(b_base0 + k_offset + kk),
                                                                    vld1_s8(b_base1 + k_offset + kk));
                                    int8x16_t b01_hi = vcombine_s8(vld1_s8(b_base0 + k_offset + kk + 8),
                                                                    vld1_s8(b_base1 + k_offset + kk + 8));
                                    acc01 = accum_matmul(acc01, a_lo, b01_lo);
                                    acc01 = accum_matmul(acc01, a_hi, b01_hi);

                                    int8x16_t b23_lo = vcombine_s8(vld1_s8(b_base2 + k_offset + kk),
                                                                    vld1_s8(b_base3 + k_offset + kk));
                                    int8x16_t b23_hi = vcombine_s8(vld1_s8(b_base2 + k_offset + kk + 8),
                                                                    vld1_s8(b_base3 + k_offset + kk + 8));
                                    acc23 = accum_matmul(acc23, a_lo, b23_lo);
                                    acc23 = accum_matmul(acc23, a_hi, b23_hi);
                                }
                            }

                            group_acc[mi][ni] += vgetq_lane_s32(acc01, 0);
                            group_acc[mi][ni + 1] += vgetq_lane_s32(acc01, 1);
                            group_acc[mi + 1][ni] += vgetq_lane_s32(acc01, 2);
                            group_acc[mi + 1][ni + 1] += vgetq_lane_s32(acc01, 3);
                            group_acc[mi][ni + 2] += vgetq_lane_s32(acc23, 0);
                            group_acc[mi][ni + 3] += vgetq_lane_s32(acc23, 1);
                            group_acc[mi + 1][ni + 2] += vgetq_lane_s32(acc23, 2);
                            group_acc[mi + 1][ni + 3] += vgetq_lane_s32(acc23, 3);
                        }

                        for (; ni + 1 < actual_n; ni += 2) {
                            int32x4_t acc0 = vdupq_n_s32(0);

                            for (size_t k_offset = 0; k_offset < group_size; k_offset += 64) {
                                const int8_t* a_ptr0 = A + (m_start + mi) * K + k_base + k_offset;
                                const int8_t* a_ptr1 = A + (m_start + mi + 1) * K + k_base + k_offset;
                                const int8_t* b_ptr0 = B + (n_start + ni) * K + k_base + k_offset;
                                const int8_t* b_ptr1 = B + (n_start + ni + 1) * K + k_base + k_offset;

                                for (int kk = 0; kk < 64; kk += 8) {
                                    int8x16_t a_vec = vcombine_s8(vld1_s8(a_ptr0 + kk), vld1_s8(a_ptr1 + kk));
                                    int8x16_t b_vec = vcombine_s8(vld1_s8(b_ptr0 + kk), vld1_s8(b_ptr1 + kk));
                                    acc0 = accum_matmul(acc0, a_vec, b_vec);
                                }
                            }

                            group_acc[mi][ni] += vgetq_lane_s32(acc0, 0);
                            group_acc[mi][ni + 1] += vgetq_lane_s32(acc0, 1);
                            group_acc[mi + 1][ni] += vgetq_lane_s32(acc0, 2);
                            group_acc[mi + 1][ni + 1] += vgetq_lane_s32(acc0, 3);
                        }

                        for (; ni < actual_n; ni++) {
                            for (size_t mii = mi; mii < mi + 2; mii++) {
                                for (size_t k_offset = 0; k_offset < group_size; k_offset += 64) {
                                    const int8_t* a_ptr = A + (m_start + mii) * K + k_base + k_offset;
                                    const int8_t* b_ptr = B + (n_start + ni) * K + k_base + k_offset;
                                    int32x4_t sum = vdupq_n_s32(0);
                                    sum = accum_dot(sum, vld1q_s8(a_ptr), vld1q_s8(b_ptr));
                                    sum = accum_dot(sum, vld1q_s8(a_ptr + 16), vld1q_s8(b_ptr + 16));
                                    sum = accum_dot(sum, vld1q_s8(a_ptr + 32), vld1q_s8(b_ptr + 32));
                                    sum = accum_dot(sum, vld1q_s8(a_ptr + 48), vld1q_s8(b_ptr + 48));
                                    group_acc[mii][ni] += vaddvq_s32(sum);
                                }
                            }
                        }
                    }

                    for (; mi < actual_m; mi++) {
                        for (size_t k_offset = 0; k_offset < group_size; k_offset += 64) {
                            const int8_t* a_ptr = A + (m_start + mi) * K + k_base + k_offset;
                            int8x16_t a_vec0 = vld1q_s8(a_ptr);
                            int8x16_t a_vec1 = vld1q_s8(a_ptr + 16);
                            int8x16_t a_vec2 = vld1q_s8(a_ptr + 32);
                            int8x16_t a_vec3 = vld1q_s8(a_ptr + 48);

                            for (size_t ni = 0; ni < actual_n; ni++) {
                                const int8_t* b_ptr = B + (n_start + ni) * K + k_base + k_offset;
                                int32x4_t sum = vdupq_n_s32(0);
                                sum = accum_dot(sum, a_vec0, vld1q_s8(b_ptr));
                                sum = accum_dot(sum, a_vec1, vld1q_s8(b_ptr + 16));
                                sum = accum_dot(sum, a_vec2, vld1q_s8(b_ptr + 32));
                                sum = accum_dot(sum, a_vec3, vld1q_s8(b_ptr + 48));
                                group_acc[mi][ni] += vaddvq_s32(sum);
                            }
                        }
                    }
                } else {
                    for (size_t k_offset = 0; k_offset < group_size; k_offset += 64) {
                        int8x16_t b_vec0[TILE_N], b_vec1[TILE_N], b_vec2[TILE_N], b_vec3[TILE_N];
                        for (size_t ni = 0; ni < actual_n; ni++) {
                            const int8_t* b_ptr = B + (n_start + ni) * K + k_base + k_offset;
                            b_vec0[ni] = vld1q_s8(b_ptr);
                            b_vec1[ni] = vld1q_s8(b_ptr + 16);
                            b_vec2[ni] = vld1q_s8(b_ptr + 32);
                            b_vec3[ni] = vld1q_s8(b_ptr + 48);
                        }

                        for (size_t mi = 0; mi < actual_m; mi++) {
                            const int8_t* a_ptr = A + (m_start + mi) * K + k_base + k_offset;
                            int8x16_t a_vec0 = vld1q_s8(a_ptr);
                            int8x16_t a_vec1 = vld1q_s8(a_ptr + 16);
                            int8x16_t a_vec2 = vld1q_s8(a_ptr + 32);
                            int8x16_t a_vec3 = vld1q_s8(a_ptr + 48);

                            for (size_t ni = 0; ni < actual_n; ni++) {
                                int32x4_t sum = vdupq_n_s32(0);
                                sum = accum_dot(sum, a_vec0, b_vec0[ni]);
                                sum = accum_dot(sum, a_vec1, b_vec1[ni]);
                                sum = accum_dot(sum, a_vec2, b_vec2[ni]);
                                sum = accum_dot(sum, a_vec3, b_vec3[ni]);
                                group_acc[mi][ni] += vaddvq_s32(sum);
                            }
                        }
                    }
                }

                for (size_t ni = 0; ni < actual_n; ni++) {
                    const float b_scale = (float)B_scales[(n_start + ni) * num_groups + g];
                    for (size_t mi = 0; mi < actual_m; mi++) {
                        running_sum[mi][ni] += (float)group_acc[mi][ni] * b_scale;
                    }
                }
            }

            for (size_t mi = 0; mi < actual_m; mi++) {
                const float a_scale = A_scales[m_start + mi];
                for (size_t ni = 0; ni < actual_n; ni++) {
                    C[(m_start + mi) * N + (n_start + ni)] = (__fp16)(running_sum[mi][ni] * a_scale);
                }
            }
            } // tile_idx
        });
}

void cactus_matmul_int4(
    const int8_t* A,
    const float* A_scales,
    const uint8_t* B_packed,
    const __fp16* B_scales,
    __fp16* C,
    size_t M, size_t K, size_t N,
    size_t group_size
) {
    if (M == 0 || K == 0 || N == 0) return;

    #if defined(__APPLE__) && defined(__arm64__)
      constexpr size_t TILE_M = 4;
      constexpr size_t TILE_N = 8;
    #else
      constexpr size_t TILE_M = 4;
      constexpr size_t TILE_N = 4;
    #endif

    const size_t num_groups = K / group_size;
    const size_t K_packed = K / 2;

    const size_t num_row_tiles = (M + TILE_M - 1) / TILE_M;
    const size_t num_col_tiles = (N + TILE_N - 1) / TILE_N;
    const size_t total_tiles = num_row_tiles * num_col_tiles;

    if (M == 1) {
        const int8_t* a_row = A;
        const float a_scale = A_scales[0];

        CactusThreading::parallel_for(N, CactusThreading::Thresholds::ELEMENT_WISE,
            [=](size_t n_start, size_t n_end) {
                for (size_t n = n_start; n < n_end; n++) {
                    const uint8_t* b_col = B_packed + n * K_packed;
                    const __fp16* col_scales = &B_scales[n * num_groups];

                    float sum = 0.0f;
                    for (size_t g = 0; g < num_groups; g++) {
                        int32_t group_sum = int4_dot_m1_asm(
                            a_row + g * group_size,
                            b_col + (g * group_size) / 2,
                            group_size
                        );
                        sum += (float)group_sum * (float)col_scales[g];
                    }
                    C[n] = (__fp16)(sum * a_scale);
                }
            });
        return;
    }

    CactusThreading::parallel_gemm_tiles(M, total_tiles,
        [=](size_t tile_start, size_t tile_end) {
            alignas(64) int8_t B_unpacked[TILE_N][128];

            for (size_t tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
            const size_t tile_row = tile_idx / num_col_tiles;
            const size_t tile_col = tile_idx % num_col_tiles;
            const size_t m_start = tile_row * TILE_M;
            const size_t m_end = std::min(m_start + TILE_M, M);
            const size_t n_start = tile_col * TILE_N;
            const size_t n_end = std::min(n_start + TILE_N, N);
            const size_t actual_m = m_end - m_start;
            const size_t actual_n = n_end - n_start;

            float running_sum[TILE_M][TILE_N] = {{0.0f}};

            for (size_t ni = 0; ni < actual_n; ni++) {
                __builtin_prefetch(&B_scales[(n_start + ni) * num_groups], 0, 3);
            }

            for (size_t g = 0; g < num_groups; g++) {
                const size_t k_base = g * group_size;
                const size_t k_base_packed = k_base / 2;

                int32_t group_acc[TILE_M][TILE_N] = {{0}};

                for (size_t ni = 0; ni < actual_n; ni++) {
                    const uint8_t* b_ptr = B_packed + (n_start + ni) * K_packed + k_base_packed;
                    int8_t* dst = B_unpacked[ni];

                    for (size_t k = 0; k < group_size; k += 32) {
                        uint8x16_t packed = vld1q_u8(b_ptr + k / 2);
                        int8x16_t lo, hi;
                        unpack_int4_to_int8x32(packed, lo, hi);
                        vst1q_s8(dst + k, lo);
                        vst1q_s8(dst + k + 16, hi);
                    }
                }

                if (cactus_has_i8mm()) {
                    size_t mi = 0;
                    for (; mi + 1 < actual_m; mi += 2) {
                        const int8_t* a_base0 = A + (m_start + mi) * K + k_base;
                        const int8_t* a_base1 = A + (m_start + mi + 1) * K + k_base;

                        size_t ni = 0;
                        for (; ni + 3 < actual_n; ni += 4) {
                            int32x4_t acc01 = vdupq_n_s32(0);
                            int32x4_t acc23 = vdupq_n_s32(0);

                            const int8_t* b0 = B_unpacked[ni];
                            const int8_t* b1 = B_unpacked[ni + 1];
                            const int8_t* b2 = B_unpacked[ni + 2];
                            const int8_t* b3 = B_unpacked[ni + 3];

                            for (size_t k_offset = 0; k_offset < group_size; k_offset += 8) {
                                int8x8_t a0_8 = vld1_s8(a_base0 + k_offset);
                                int8x8_t a1_8 = vld1_s8(a_base1 + k_offset);
                                int8x16_t a_combined = vcombine_s8(a0_8, a1_8);

                                int8x8_t b0_8 = vld1_s8(b0 + k_offset);
                                int8x8_t b1_8 = vld1_s8(b1 + k_offset);
                                int8x8_t b2_8 = vld1_s8(b2 + k_offset);
                                int8x8_t b3_8 = vld1_s8(b3 + k_offset);

                                acc01 = accum_matmul(acc01, a_combined, vcombine_s8(b0_8, b1_8));
                                acc23 = accum_matmul(acc23, a_combined, vcombine_s8(b2_8, b3_8));
                            }

                            group_acc[mi][ni] += vgetq_lane_s32(acc01, 0);
                            group_acc[mi][ni + 1] += vgetq_lane_s32(acc01, 1);
                            group_acc[mi + 1][ni] += vgetq_lane_s32(acc01, 2);
                            group_acc[mi + 1][ni + 1] += vgetq_lane_s32(acc01, 3);
                            group_acc[mi][ni + 2] += vgetq_lane_s32(acc23, 0);
                            group_acc[mi][ni + 3] += vgetq_lane_s32(acc23, 1);
                            group_acc[mi + 1][ni + 2] += vgetq_lane_s32(acc23, 2);
                            group_acc[mi + 1][ni + 3] += vgetq_lane_s32(acc23, 3);
                        }

                        // Handle remaining columns
                        for (; ni < actual_n; ni++) {
                            const int8_t* b_col = B_unpacked[ni];
                            for (size_t mii = mi; mii < mi + 2 && mii < actual_m; mii++) {
                                const int8_t* a_ptr = A + (m_start + mii) * K + k_base;
                                int32x4_t sum = vdupq_n_s32(0);
                                for (size_t k_offset = 0; k_offset < group_size; k_offset += 16) {
                                    sum = accum_dot(sum, vld1q_s8(a_ptr + k_offset), vld1q_s8(b_col + k_offset));
                                }
                                group_acc[mii][ni] += vaddvq_s32(sum);
                            }
                        }
                    }

                    for (; mi < actual_m; mi++) {
                        const int8_t* a_base = A + (m_start + mi) * K + k_base;
                        for (size_t ni = 0; ni < actual_n; ni++) {
                            const int8_t* b_col = B_unpacked[ni];
                            int32x4_t sum = vdupq_n_s32(0);
                            for (size_t k_offset = 0; k_offset < group_size; k_offset += 16) {
                                sum = accum_dot(sum, vld1q_s8(a_base + k_offset), vld1q_s8(b_col + k_offset));
                            }
                            group_acc[mi][ni] += vaddvq_s32(sum);
                        }
                    }
                } else {
                    for (size_t mi = 0; mi < actual_m; mi++) {
                        const int8_t* a_ptr = A + (m_start + mi) * K + k_base;

                        for (size_t ni = 0; ni < actual_n; ni++) {
                            const int8_t* b_col = B_unpacked[ni];
                            int32x4_t sum = vdupq_n_s32(0);

                            for (size_t k_offset = 0; k_offset < group_size; k_offset += 64) {
                                int8x16_t a0 = vld1q_s8(a_ptr + k_offset);
                                int8x16_t a1 = vld1q_s8(a_ptr + k_offset + 16);
                                int8x16_t a2 = vld1q_s8(a_ptr + k_offset + 32);
                                int8x16_t a3 = vld1q_s8(a_ptr + k_offset + 48);

                                int8x16_t b0 = vld1q_s8(b_col + k_offset);
                                int8x16_t b1 = vld1q_s8(b_col + k_offset + 16);
                                int8x16_t b2 = vld1q_s8(b_col + k_offset + 32);
                                int8x16_t b3 = vld1q_s8(b_col + k_offset + 48);

                                sum = accum_dot(sum, a0, b0);
                                sum = accum_dot(sum, a1, b1);
                                sum = accum_dot(sum, a2, b2);
                                sum = accum_dot(sum, a3, b3);
                            }
                            group_acc[mi][ni] += vaddvq_s32(sum);
                        }
                    }
                }

                for (size_t ni = 0; ni < actual_n; ni++) {
                    const float b_scale = (float)B_scales[(n_start + ni) * num_groups + g];
                    for (size_t mi = 0; mi < actual_m; mi++) {
                        running_sum[mi][ni] += (float)group_acc[mi][ni] * b_scale;
                    }
                }
            }

            for (size_t mi = 0; mi < actual_m; mi++) {
                const float a_scale = A_scales[m_start + mi];
                for (size_t ni = 0; ni < actual_n; ni++) {
                    C[(m_start + mi) * N + (n_start + ni)] = (__fp16)(running_sum[mi][ni] * a_scale);
                }
            }
            } // tile_idx
        });
}
