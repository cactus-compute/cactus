#include <arm_neon.h>

#ifndef __ARM_FEATURE_MATMUL_INT8
#error "kernel_i8mm.cpp must be compiled with I8MM enabled (e.g. -march=armv8.2-a+fp16+simd+dotprod+i8mm)"
#endif

#include "kernel.h"
#include "kernel_utils.h"
#include <algorithm>
#include <cstdint>

void cactus_gemv_int8_i8mm(
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

            for (size_t g = 0; g < num_groups; g++) {
                const size_t k_base = g * group_size;
                const int8_t* a_ptr = A + k_base;
                const int8_t* b_ptr = B + (n_block * K + k_base) * 4;

                __builtin_prefetch(b_ptr + group_size * 8, 0, 3);

                int32x4_t acc_01 = vdupq_n_s32(0);
                int32x4_t acc_23 = vdupq_n_s32(0);

                for (size_t k_off = 0; k_off < group_size; k_off += 8) {
                    int32x4_t chunk0 = vreinterpretq_s32_s8(vld1q_s8(b_ptr + k_off * 4));
                    int32x4_t chunk1 = vreinterpretq_s32_s8(vld1q_s8(b_ptr + k_off * 4 + 16));

                    int8x16_t b01 = vreinterpretq_s8_s32(vzip1q_s32(chunk0, chunk1));
                    int8x16_t b23 = vreinterpretq_s8_s32(vzip2q_s32(chunk0, chunk1));

                    int8x8_t a8 = vld1_s8(a_ptr + k_off);
                    int8x16_t a_dup = vcombine_s8(a8, a8);

                    acc_01 = vmmlaq_s32(acc_01, a_dup, b01);
                    acc_23 = vmmlaq_s32(acc_23, a_dup, b23);
                }

                float32x2_t r01 = vcvt_f32_s32(vget_low_s32(acc_01));
                float32x2_t r23 = vcvt_f32_s32(vget_low_s32(acc_23));
                float32x4_t group_result = vcombine_f32(r01, r23);

                const __fp16* scale_ptr = B_scales + (n_block * num_groups + g) * 4;
                float32x4_t scales = vcvt_f32_f16(vld1_f16(scale_ptr));
                running_sum = vmlaq_f32(running_sum, group_result, scales);
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

void cactus_gemm_int8_i8mm(
    const int8_t* A,
    const float* A_scales,
    const int8_t* B,
    const __fp16* B_scales,
    __fp16* C,
    size_t M, size_t K, size_t N,
    size_t group_size
) {
    if (M == 0 || K == 0 || N == 0) return;

    const size_t num_groups = K / group_size;
    const size_t N_blocks = (N + 3) / 4;
    const size_t M_pairs = (M + 1) / 2;
    const size_t total_tiles = M_pairs * N_blocks;

    CactusThreading::parallel_gemm_tiles(M, total_tiles,
        [=](size_t tile_start, size_t tile_end) {
            for (size_t tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
                const size_t m_pair = tile_idx / N_blocks;
                const size_t n_block = tile_idx % N_blocks;
                const size_t m0 = m_pair * 2;
                const size_t m1 = std::min(m0 + 1, M - 1);
                const size_t n_start = n_block * 4;
                const size_t actual_n = std::min(size_t(4), N - n_start);

                const int8_t* a_row0 = A + m0 * K;
                const int8_t* a_row1 = A + m1 * K;

                float32x4_t sum_row0 = vdupq_n_f32(0.0f);
                float32x4_t sum_row1 = vdupq_n_f32(0.0f);

                for (size_t g = 0; g < num_groups; g++) {
                    const size_t k_base = g * group_size;
                    const int8_t* b_ptr = B + (n_block * K + k_base) * 4;

                    __builtin_prefetch(b_ptr + group_size * 8, 0, 3);

                    int32x4_t acc_01 = vdupq_n_s32(0);
                    int32x4_t acc_23 = vdupq_n_s32(0);

                    for (size_t k_off = 0; k_off < group_size; k_off += 8) {
                        int32x4_t chunk0 = vreinterpretq_s32_s8(vld1q_s8(b_ptr + k_off * 4));
                        int32x4_t chunk1 = vreinterpretq_s32_s8(vld1q_s8(b_ptr + k_off * 4 + 16));

                        int8x16_t b01 = vreinterpretq_s8_s32(vzip1q_s32(chunk0, chunk1));
                        int8x16_t b23 = vreinterpretq_s8_s32(vzip2q_s32(chunk0, chunk1));

                        int8x8_t a0_8 = vld1_s8(a_row0 + k_base + k_off);
                        int8x8_t a1_8 = vld1_s8(a_row1 + k_base + k_off);
                        int8x16_t a_2rows = vcombine_s8(a0_8, a1_8);

                        acc_01 = vmmlaq_s32(acc_01, a_2rows, b01);
                        acc_23 = vmmlaq_s32(acc_23, a_2rows, b23);
                    }

                    const __fp16* scale_ptr = B_scales + (n_block * num_groups + g) * 4;
                    float32x4_t scales = vcvt_f32_f16(vld1_f16(scale_ptr));

                    float32x2_t r0_01 = vcvt_f32_s32(vget_low_s32(acc_01));
                    float32x2_t r0_23 = vcvt_f32_s32(vget_low_s32(acc_23));
                    float32x4_t group_row0 = vcombine_f32(r0_01, r0_23);
                    sum_row0 = vmlaq_f32(sum_row0, group_row0, scales);

                    float32x2_t r1_01 = vcvt_f32_s32(vget_high_s32(acc_01));
                    float32x2_t r1_23 = vcvt_f32_s32(vget_high_s32(acc_23));
                    float32x4_t group_row1 = vcombine_f32(r1_01, r1_23);
                    sum_row1 = vmlaq_f32(sum_row1, group_row1, scales);
                }

                {
                    float32x4_t result = vmulq_n_f32(sum_row0, A_scales[m0]);
                    float16x4_t result_f16 = vcvt_f16_f32(result);
                    if (actual_n == 4) {
                        vst1_f16(C + m0 * N + n_start, result_f16);
                    } else {
                        for (size_t ni = 0; ni < actual_n; ni++) {
                            C[m0 * N + n_start + ni] = vget_lane_f16(result_f16, 0);
                            result_f16 = vext_f16(result_f16, result_f16, 1);
                        }
                    }
                }

                if (m0 + 1 < M) {
                    float32x4_t result = vmulq_n_f32(sum_row1, A_scales[m1]);
                    float16x4_t result_f16 = vcvt_f16_f32(result);
                    if (actual_n == 4) {
                        vst1_f16(C + m1 * N + n_start, result_f16);
                    } else {
                        for (size_t ni = 0; ni < actual_n; ni++) {
                            C[m1 * N + n_start + ni] = vget_lane_f16(result_f16, 0);
                            result_f16 = vext_f16(result_f16, result_f16, 1);
                        }
                    }
                }
            }
        });
}
