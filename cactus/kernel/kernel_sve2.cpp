#include <arm_sve.h>
#include <cstdint>
#ifndef __ARM_FEATURE_SVE2
#error "kernel_sve2.cpp must be compiled with SVE2 enabled (e.g. -march=armv9-a+sve2)"
#endif

#include "kernel.h"
#include "kernel_utils.h"
#include <algorithm>

static void cactus_matmul_f16_sve2_worker(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t K,
    size_t N,
    size_t start_row,
    size_t end_row
) {
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 4;
    const size_t vl = svcnth();
    const svbool_t pg_all = svptrue_b16();

    for (size_t row_block = start_row; row_block < end_row; row_block += TILE_M) {
        const size_t m_end = std::min(row_block + TILE_M, end_row);

        for (size_t col_block = 0; col_block < N; col_block += TILE_N) {
            const size_t n_end = std::min(col_block + TILE_N, N);

            svfloat16_t acc00 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc01 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc02 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc03 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc10 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc11 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc12 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc13 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc20 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc21 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc22 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc23 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc30 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc31 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc32 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc33 = svdup_n_f16(static_cast<__fp16>(0));

            for (size_t k = 0; k < K; k += vl) {
                const svbool_t pg = svwhilelt_b16(static_cast<uint64_t>(k), static_cast<uint64_t>(K));

                const svfloat16_t a0 = (row_block < m_end)
                    ? svld1_f16(pg, a + row_block * K + k)
                    : svdup_n_f16(static_cast<__fp16>(0));
                const svfloat16_t a1 = (row_block + 1 < m_end)
                    ? svld1_f16(pg, a + (row_block + 1) * K + k)
                    : svdup_n_f16(static_cast<__fp16>(0));
                const svfloat16_t a2 = (row_block + 2 < m_end)
                    ? svld1_f16(pg, a + (row_block + 2) * K + k)
                    : svdup_n_f16(static_cast<__fp16>(0));
                const svfloat16_t a3 = (row_block + 3 < m_end)
                    ? svld1_f16(pg, a + (row_block + 3) * K + k)
                    : svdup_n_f16(static_cast<__fp16>(0));

                if (col_block < n_end) {
                    const svfloat16_t b0 = svld1_f16(pg, b_transposed + col_block * K + k);
                    acc00 = svmla_f16_m(pg, acc00, a0, b0);
                    acc10 = svmla_f16_m(pg, acc10, a1, b0);
                    acc20 = svmla_f16_m(pg, acc20, a2, b0);
                    acc30 = svmla_f16_m(pg, acc30, a3, b0);
                }
                if (col_block + 1 < n_end) {
                    const svfloat16_t b1 = svld1_f16(pg, b_transposed + (col_block + 1) * K + k);
                    acc01 = svmla_f16_m(pg, acc01, a0, b1);
                    acc11 = svmla_f16_m(pg, acc11, a1, b1);
                    acc21 = svmla_f16_m(pg, acc21, a2, b1);
                    acc31 = svmla_f16_m(pg, acc31, a3, b1);
                }
                if (col_block + 2 < n_end) {
                    const svfloat16_t b2 = svld1_f16(pg, b_transposed + (col_block + 2) * K + k);
                    acc02 = svmla_f16_m(pg, acc02, a0, b2);
                    acc12 = svmla_f16_m(pg, acc12, a1, b2);
                    acc22 = svmla_f16_m(pg, acc22, a2, b2);
                    acc32 = svmla_f16_m(pg, acc32, a3, b2);
                }
                if (col_block + 3 < n_end) {
                    const svfloat16_t b3 = svld1_f16(pg, b_transposed + (col_block + 3) * K + k);
                    acc03 = svmla_f16_m(pg, acc03, a0, b3);
                    acc13 = svmla_f16_m(pg, acc13, a1, b3);
                    acc23 = svmla_f16_m(pg, acc23, a2, b3);
                    acc33 = svmla_f16_m(pg, acc33, a3, b3);
                }
            }

            if (row_block < m_end) {
                if (col_block < n_end) c[row_block * N + col_block] = svaddv_f16(pg_all, acc00);
                if (col_block + 1 < n_end) c[row_block * N + col_block + 1] = svaddv_f16(pg_all, acc01);
                if (col_block + 2 < n_end) c[row_block * N + col_block + 2] = svaddv_f16(pg_all, acc02);
                if (col_block + 3 < n_end) c[row_block * N + col_block + 3] = svaddv_f16(pg_all, acc03);
            }
            if (row_block + 1 < m_end) {
                if (col_block < n_end) c[(row_block + 1) * N + col_block] = svaddv_f16(pg_all, acc10);
                if (col_block + 1 < n_end) c[(row_block + 1) * N + col_block + 1] = svaddv_f16(pg_all, acc11);
                if (col_block + 2 < n_end) c[(row_block + 1) * N + col_block + 2] = svaddv_f16(pg_all, acc12);
                if (col_block + 3 < n_end) c[(row_block + 1) * N + col_block + 3] = svaddv_f16(pg_all, acc13);
            }
            if (row_block + 2 < m_end) {
                if (col_block < n_end) c[(row_block + 2) * N + col_block] = svaddv_f16(pg_all, acc20);
                if (col_block + 1 < n_end) c[(row_block + 2) * N + col_block + 1] = svaddv_f16(pg_all, acc21);
                if (col_block + 2 < n_end) c[(row_block + 2) * N + col_block + 2] = svaddv_f16(pg_all, acc22);
                if (col_block + 3 < n_end) c[(row_block + 2) * N + col_block + 3] = svaddv_f16(pg_all, acc23);
            }
            if (row_block + 3 < m_end) {
                if (col_block < n_end) c[(row_block + 3) * N + col_block] = svaddv_f16(pg_all, acc30);
                if (col_block + 1 < n_end) c[(row_block + 3) * N + col_block + 1] = svaddv_f16(pg_all, acc31);
                if (col_block + 2 < n_end) c[(row_block + 3) * N + col_block + 2] = svaddv_f16(pg_all, acc32);
                if (col_block + 3 < n_end) c[(row_block + 3) * N + col_block + 3] = svaddv_f16(pg_all, acc33);
            }
        }
    }
}

void cactus_matmul_f16_sve2_caller(
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
                const size_t start_row = block_idx * TILE_M;
                const size_t end_row = std::min(start_row + TILE_M, M);

                cactus_matmul_f16_sve2_worker(
                    a, b_transposed, c,
                    K, N,
                    start_row, end_row
                );
            }
        });
}
