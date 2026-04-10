#include <arm_sve.h>
#include <cstdint>
#ifndef __ARM_FEATURE_SVE2
#error "kernel_sve2.cpp must be compiled with SVE2 enabled (e.g. -march=armv9-a+sve2)"
#endif

#include "kernel.h"
#include "kernel_utils.h"
#include <algorithm>
#include <memory>

namespace {

constexpr size_t TILE_M = 4;
constexpr size_t TILE_N_VECTORS = 2;

void cactus_pack_b_f16_sve_from_bt(
    const __fp16* __restrict b_transposed,
    __fp16* __restrict b_packed,
    size_t K,
    size_t N,
    size_t nr
) {
    const size_t num_panels = (N + nr - 1) / nr;

    CactusThreading::parallel_for(num_panels * K, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                const size_t panel = idx / K;
                const size_t k = idx % K;
                const size_t col0 = panel * nr;
                const size_t active_cols = std::min(nr, N - col0);

                __fp16* dst = b_packed + (panel * K + k) * nr;
                for (size_t lane = 0; lane < active_cols; ++lane) {
                    dst[lane] = b_transposed[(col0 + lane) * K + k];
                }
                for (size_t lane = active_cols; lane < nr; ++lane) {
                    dst[lane] = static_cast<__fp16>(0);
                }
            }
        });
}

void cactus_matmul_f16_sve2_worker(
    const __fp16* a,
    const __fp16* b_packed,
    __fp16* c,
    size_t K,
    size_t N,
    size_t nr,
    size_t start_row,
    size_t end_row
) {
    const size_t vl = svcnth();
    const size_t num_panels = (N + nr - 1) / nr;
    const svbool_t pg_all = svptrue_b16();

    for (size_t row_block = start_row; row_block < end_row; row_block += TILE_M) {
        const size_t m_end = std::min(row_block + TILE_M, end_row);

        const __fp16* a0_ptr = (row_block < m_end) ? a + row_block * K : nullptr;
        const __fp16* a1_ptr = (row_block + 1 < m_end) ? a + (row_block + 1) * K : nullptr;
        const __fp16* a2_ptr = (row_block + 2 < m_end) ? a + (row_block + 2) * K : nullptr;
        const __fp16* a3_ptr = (row_block + 3 < m_end) ? a + (row_block + 3) * K : nullptr;

        for (size_t panel = 0; panel < num_panels; ++panel) {
            const size_t col0 = panel * nr;
            const size_t col1 = col0 + vl;
            const svbool_t pg0 = svwhilelt_b16(static_cast<uint64_t>(col0), static_cast<uint64_t>(N));
            const svbool_t pg1 = svwhilelt_b16(static_cast<uint64_t>(col1), static_cast<uint64_t>(N));

            svfloat16_t acc00 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc01 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc10 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc11 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc20 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc21 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc30 = svdup_n_f16(static_cast<__fp16>(0));
            svfloat16_t acc31 = svdup_n_f16(static_cast<__fp16>(0));

            const __fp16* panel_ptr = b_packed + panel * K * nr;
            for (size_t k = 0; k < K; ++k) {
                const __fp16* b_row = panel_ptr + k * nr;
                const svfloat16_t b0 = svld1_f16(pg_all, b_row);
                const svfloat16_t b1 = svld1_f16(pg_all, b_row + vl);

                if (a0_ptr) {
                    const svfloat16_t a0 = svdup_n_f16(a0_ptr[k]);
                    acc00 = svmla_f16_m(pg_all, acc00, a0, b0);
                    acc01 = svmla_f16_m(pg_all, acc01, a0, b1);
                }
                if (a1_ptr) {
                    const svfloat16_t a1 = svdup_n_f16(a1_ptr[k]);
                    acc10 = svmla_f16_m(pg_all, acc10, a1, b0);
                    acc11 = svmla_f16_m(pg_all, acc11, a1, b1);
                }
                if (a2_ptr) {
                    const svfloat16_t a2 = svdup_n_f16(a2_ptr[k]);
                    acc20 = svmla_f16_m(pg_all, acc20, a2, b0);
                    acc21 = svmla_f16_m(pg_all, acc21, a2, b1);
                }
                if (a3_ptr) {
                    const svfloat16_t a3 = svdup_n_f16(a3_ptr[k]);
                    acc30 = svmla_f16_m(pg_all, acc30, a3, b0);
                    acc31 = svmla_f16_m(pg_all, acc31, a3, b1);
                }
            }

            if (a0_ptr) {
                svst1(pg0, c + row_block * N + col0, acc00);
                if (col1 < N) svst1(pg1, c + row_block * N + col1, acc01);
            }
            if (a1_ptr) {
                svst1(pg0, c + (row_block + 1) * N + col0, acc10);
                if (col1 < N) svst1(pg1, c + (row_block + 1) * N + col1, acc11);
            }
            if (a2_ptr) {
                svst1(pg0, c + (row_block + 2) * N + col0, acc20);
                if (col1 < N) svst1(pg1, c + (row_block + 2) * N + col1, acc21);
            }
            if (a3_ptr) {
                svst1(pg0, c + (row_block + 3) * N + col0, acc30);
                if (col1 < N) svst1(pg1, c + (row_block + 3) * N + col1, acc31);
            }
        }
    }
}

}  // namespace

void cactus_matmul_f16_sve2_caller(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N
) {
    const size_t vl = svcnth();
    const size_t nr = vl * TILE_N_VECTORS;
    const size_t num_panels = (N + nr - 1) / nr;
    const size_t packed_b_elems = num_panels * K * nr;
    const size_t num_row_blocks = (M + TILE_M - 1) / TILE_M;

    std::unique_ptr<__fp16[]> b_packed(new __fp16[packed_b_elems]);
    cactus_pack_b_f16_sve_from_bt(b_transposed, b_packed.get(), K, N, nr);
    const __fp16* b_packed_ptr = b_packed.get();

    CactusThreading::parallel_for(num_row_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start_block, size_t end_block) {
            for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
                const size_t start_row = block_idx * TILE_M;
                const size_t end_row = std::min(start_row + TILE_M, M);

                cactus_matmul_f16_sve2_worker(
                    a, b_packed_ptr, c,
                    K, N, nr,
                    start_row, end_row
                );
            }
        });
}
