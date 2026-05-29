/* fp16 × fp16 mat-vec for the LM head (and any other fp16-weight layer).
 *
 * Simpler than int4: no dequant. One threadgroup per output row block,
 * 32 threads in a simdgroup, each accumulates a partial dot, simd_sum
 * at the end.
 *
 * Output:  y[N] = W[N, K] @ x[K]
 * Layout:  W is row-major [N, K]  (HF Linear convention)
 *
 * Function constants:
 *   MM_K_FP : K
 *   MM_N_FP : N
 */
#include "common.metal"

constant uint MM_K_FP [[function_constant(90)]];
constant uint MM_N_FP [[function_constant(91)]];

#define MV_FP16_ROWS 4u
#define MV_FP16_THREADS 32u

kernel void mul_mv_fp16(
    device const half * W       [[buffer(0)]],   // [N, K]
    device const half * x       [[buffer(1)]],   // [K]
    device       half * y       [[buffer(2)]],   // [N]
    uint3 tgid              [[threadgroup_position_in_grid]],
    uint3 tid_v             [[thread_position_in_threadgroup]])
{
    const uint tid = tid_v.x;
    const uint row_base = tgid.x * MV_FP16_ROWS;
    if (row_base >= MM_N_FP) return;

    float acc[MV_FP16_ROWS] = {0.f, 0.f, 0.f, 0.f};

    // Each thread handles K / 32 consecutive elements per row.
    const uint per_thread = MM_K_FP / MV_FP16_THREADS;
    const uint k_start    = tid * per_thread;

    #pragma unroll
    for (uint r = 0; r < MV_FP16_ROWS; ++r) {
        device const half* w_row = W + (row_base + r) * MM_K_FP;
        for (uint kk = 0; kk < per_thread; ++kk) {
            acc[r] += float(w_row[k_start + kk]) * float(x[k_start + kk]);
        }
    }

    #pragma unroll
    for (uint r = 0; r < MV_FP16_ROWS; ++r) {
        acc[r] = simd_sum(acc[r]);
    }
    if (tid == 0u) {
        for (uint r = 0; r < MV_FP16_ROWS; ++r) {
            y[row_base + r] = half(acc[r]);
        }
    }
}
