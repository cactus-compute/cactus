#include <arm_sve.h>
#include <cstdint>
#ifndef __ARM_FEATURE_SME2
#error "kernel_sme2.cpp must be compiled with SME2 enabled (e.g. -march=armv9.2-a+sme2)"
#endif

#include "kernel.h"
#include "kernel_utils.h"
#include <arm_sme.h>

static void cactus_matmul_f16_sme2_worker(
	const __fp16* a_transposed, const __fp16* b, __fp16* c,
	size_t M, size_t K, size_t N,
	size_t start_row, size_t end_row,
	size_t TILE_M, size_t TILE_N,
	float* tmp
) __arm_streaming __arm_inout("za") {
	(void) M;
	if (start_row >= end_row) return;

	for (size_t row = start_row; row < end_row; row += TILE_M) {
		const size_t active_r = std::min(TILE_M, end_row - row);
		const svbool_t pMh = svwhilelt_b16(0, active_r * 2);
		const svbool_t pM16 = svwhilelt_b16(0, active_r);
		const svbool_t pMDim = svwhilelt_b32((uint64_t) row, (uint64_t) end_row);

		for (size_t col0 = 0; col0 < N; col0 += 2 * TILE_N) {
			const size_t col1 = col0 + TILE_N;	

			const size_t active_c0 = std::min(TILE_N, N - col0);
			const svbool_t pNh0 = svwhilelt_b16(0, active_c0 * 2);
			const svbool_t pN16_0 = svwhilelt_b16(0, active_c0);
			const svbool_t pN32_0 = svwhilelt_b32((uint64_t) 0, (uint64_t) active_c0);
			const svbool_t pNDim0 = svwhilelt_b32((uint64_t) col0, (uint64_t) N);

			const size_t active_c1 = (col1 < N) ? std::min(TILE_N, N - col1) : 0;
			const svbool_t pNh1 = svwhilelt_b16(0, active_c1 * 2);
			const svbool_t pN16_1 = svwhilelt_b16(0, active_c1);
			const svbool_t pN32_1 = svwhilelt_b32((uint64_t) 0, (uint64_t) active_c1);
			const svbool_t pNDim1 = svwhilelt_b32((uint64_t) col1, (uint64_t) N);

			svzero_za();
			for (size_t k = 0; k < K; k += 2) {
				svfloat16_t a0 = svld1(pM16, &a_transposed[(k + 0) * M + row]);
				svfloat16_t a1 = (k + 1 < K) ? svld1(pM16, &a_transposed[(k + 1) * M + row]) : svdup_n_f16(0);
				svfloat16_t zA = svzip1_f16(a0, a1);

				svfloat16_t b0_0 = svld1(pN16_0, &b[(k + 0) * N + col0]);
				svfloat16_t b1_0 = (k + 1 < K) ? svld1(pN16_0, &b[(k + 1) * N + col0]) : svdup_n_f16(0);
				svfloat16_t zB0 = svzip1_f16(b0_0, b1_0);

				svmopa_za32_f16_m(0, pMh, pNh0, zA, zB0);

				if (active_c1) {
					svfloat16_t b0_1 = svld1(pN16_1, &b[(k + 0) * N + col1]);
					svfloat16_t b1_1 = (k + 1 < K) ? svld1(pN16_1, &b[(k + 1) * N + col1]) : svdup_n_f16(0);
					svfloat16_t zB1 = svzip1_f16(b0_1, b1_1);

					svmopa_za32_f16_m(1, pMh, pNh1, zA, zB1);
				}
			}

			for (size_t m = 0; m < active_r; ++m) {
				svbool_t p_lane0 = svpsel_lane_b32(pNDim0, pMDim, row + m);
				svst1_hor_za32(0, m, p_lane0, tmp);
				
				svfloat32_t out32 = svld1(pN32_0, tmp);
				svfloat16_t out16 = svcvt_f16_f32_z(pN32_0, out32);
				out16 = svuzp1_f16(out16, out16);
				svst1(pN16_0, &c[(row + m) * N + col0], out16);

				if (active_c1) {
					svbool_t p_lane1 = svpsel_lane_b32(pNDim1, pMDim, row + m);
					svst1_hor_za32(1, m, p_lane1, tmp);
					
					svfloat32_t out32 = svld1(pN32_1, tmp);
					svfloat16_t out16 = svcvt_f16_f32_z(pN32_1, out32);
					out16 = svuzp1_f16(out16, out16);
					svst1(pN16_1, &c[(row + m) * N + col1], out16);
				}
			}
		}
	}
}

__arm_new("za") __arm_locally_streaming
static void cactus_matmul_f16_sme2_thread_entry(
	const __fp16* a_transposed, const __fp16* b, __fp16* c,
	size_t M, size_t K, size_t N,
	size_t ROW_BLOCK_SIZE, size_t start_block, size_t end_block
) {
	const size_t TILE = svcntsw();
	std::vector<float> tmp(TILE);		
	
	for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
        size_t start_row = block_idx * ROW_BLOCK_SIZE;
		size_t end_row = std::min(start_row + ROW_BLOCK_SIZE, M);

		cactus_matmul_f16_sme2_worker(
			a_transposed, b, c,
			M, K, N,
			start_row, end_row,
			TILE, TILE,
			tmp.data()
		);
    }
}

static inline void cactus_transpose_2d_f16_parallel(
	const __fp16* source,
	__fp16* destination,
	size_t num_rows,
	size_t num_cols
) {
	constexpr size_t TILE_ROWS = 32;
	const size_t num_row_blocks = (num_rows + TILE_ROWS - 1) / TILE_ROWS;

	CactusThreading::parallel_for(num_row_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
		[=](size_t start_block, size_t end_block) {
			for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
				const size_t start_row = block_idx * TILE_ROWS;
				const size_t end_row = std::min(start_row + TILE_ROWS, num_rows);
				cactus_transpose_2d_f16(source, destination, num_rows, num_cols, start_row, end_row);
			}
		});
}

__arm_new("za") __arm_locally_streaming
void cactus_matmul_f16_sme2_caller(
	const __fp16* a,
	const __fp16* b_transposed,
	__fp16* c,
	size_t M,
	size_t K,
	size_t N
) {
	std::vector<__fp16> aT_storage(K * M);
	std::vector<__fp16> b_storage(K * N);

	__fp16* a_T = aT_storage.data();
	__fp16* b = b_storage.data();

	cactus_transpose_2d_f16_parallel(a, a_T, M, K);
	cactus_transpose_2d_f16_parallel(b_transposed, b, N, K);
	
	constexpr size_t TILES_PER_THREAD = 2;
	const size_t ROW_BLOCK_SIZE = TILES_PER_THREAD * svcntsw();
    const size_t num_row_blocks = (M + ROW_BLOCK_SIZE - 1) / ROW_BLOCK_SIZE;

    CactusThreading::parallel_for(num_row_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start_block, size_t end_block) {
			cactus_matmul_f16_sme2_thread_entry(
				a_T, b, c,
				M, K, N,
				ROW_BLOCK_SIZE, start_block, end_block
			);
        });
}
