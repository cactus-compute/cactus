/* INT4 (CQ4) × FP16 matmul kernels.
 *
 * Two kernels:
 *   - mul_mv_int4_fp16:  matrix × vector  (decode hot path, M = 1)
 *   - mul_mm_int4_fp16:  matrix × matrix  (prefill, M > 1)
 *
 * Weight layout (per cactus CQ4 group=128, kernel-side):
 *   - qs:     uint16[K/4 × N]  (4 nibbles per uint16, N columns of K/4 uint16s)
 *   - scales: half[K/128 × N]  (one fp16 scale per group of 128 weight values)
 *
 * Notes:
 *   - The hadamard / orthogonal rotation that CQ4 applies is *baked into the
 *     weights at convert time*. From the kernel's perspective, dequant is
 *     just `scale * (nibble - 8)` (CQ4 stores 0..15 representing -8..7).
 *   - The Python transpiler is responsible for applying the *inverse*
 *     rotation to the activation X before this matmul fires. The kernel
 *     sees pre-rotated X.
 */
#include "common.metal"

// Function constants — must match indices in pipeline_factory.mm::fc::*
constant uint K            [[function_constant(10)]];
constant uint N            [[function_constant(11)]];
constant uint M_TILE       [[function_constant(12)]];
constant bool HAS_BIAS     [[function_constant(13)]];

// ============================================================================
// mul_mv_int4_fp16 — INT4 weights × fp16 vector, decode hot path
// ============================================================================
// Each threadgroup computes one tile of N rows. Each thread handles 4 rows
// (matches llama.cpp's Q4_0 mat-vec pattern). K loops 128 at a time (one CQ4
// group per iteration).
//
// Grid:  (N / 4, 1, 1) groups of 32 threads
// In:    qs[K/4 × N] uint16, scales[K/128 × N] half, x[K] half, bias[N] half? (optional)
// Out:   y[N] half
//
// Inside each thread: 4 fp32 accumulators (one per row), 4 fp32 inputs with
// pre-shift scaling (x[i] * {1, 1/16, 1/256, 1/4096}). Each iter consumes one
// uint16 (4 weights) per row, reads in parallel across the 4 rows.

#define MV_THREADS_PER_GROUP 32u
#define MV_ROWS_PER_GROUP    4u   // each thread processes 4 rows

kernel void mul_mv_int4_fp16(
    device const uint16_t * qs        [[buffer(0)]],  // K/4 × N
    device const half     * scales    [[buffer(1)]],  // K/128 × N
    device const half     * x         [[buffer(2)]],  // K
    device       half     * y         [[buffer(3)]],  // N
    device const half     * bias      [[buffer(4), function_constant(HAS_BIAS)]],
    uint3   tgid              [[threadgroup_position_in_grid]],
    uint3   tg_threads        [[threads_per_threadgroup]],
    uint3   tid_v             [[thread_position_in_threadgroup]])
{
    const uint tid = tid_v.x;
    // Row block this threadgroup owns: 4 consecutive rows.
    const uint row_base = tgid.x * MV_ROWS_PER_GROUP;
    if (row_base >= N) return;

    // 32 threads × 4 weights/thread = 128 weights = 1 CQ4 group per iteration.
    // Each thread accumulates into 4 per-row registers; simd_sum reduces
    // across the simdgroup at the end. (sum-of-x for the -8 zero-point fix
    // will be reintroduced when we wire the cactus CQ4 zero-point semantics.)
    float acc[MV_ROWS_PER_GROUP] = {0.f, 0.f, 0.f, 0.f};

    const uint group_count = K / CQ4_GROUP_SIZE;
    // One scale per group of 128, so 1 scale per 32 uint16s of qs.
    // Loop K in chunks of 32 uint16s = 128 nibbles = 1 CQ4 group.
    for (uint g = 0; g < group_count; ++g) {
        // Scale for this group, per row (4 rows).
        float s[MV_ROWS_PER_GROUP];
        #pragma unroll
        for (uint r = 0; r < MV_ROWS_PER_GROUP; ++r) {
            s[r] = float(scales[(g * N) + (row_base + r)]);
        }

        // Per-iteration: 1 thread handles 1 uint16 = 4 nibbles per row.
        // The 32 threads cover all 32 uint16s in the group.
        const uint w_off_thread = (g * CQ4_UINT16_PER_GROUP) + tid;          // uint16 index
        const uint x_off_thread = (g * CQ4_GROUP_SIZE) + (tid * 4u);         // 4 fp16 inputs

        // Load 4 inputs and pre-shift.
        float xv[4];
        xv[0] = float(x[x_off_thread + 0]);
        xv[1] = float(x[x_off_thread + 1]) * (1.0f / 16.0f);
        xv[2] = float(x[x_off_thread + 2]) * (1.0f / 256.0f);
        xv[3] = float(x[x_off_thread + 3]) * (1.0f / 4096.0f);
        // Inner reduce: for each of 4 rows, fetch one uint16 of nibbles,
        // multiply by pre-shifted x, accumulate.
        #pragma unroll
        for (uint r = 0; r < MV_ROWS_PER_GROUP; ++r) {
            uint16_t q = qs[(w_off_thread * N) + (row_base + r)];
            acc[r] += s[r] * cq4_nibble_dot(q, xv);
        }
    }

    // Reduce across simdgroup: each thread holds partial per-row sums.
    #pragma unroll
    for (uint r = 0; r < MV_ROWS_PER_GROUP; ++r) {
        acc[r] = simd_sum(acc[r]);
    }

    if (tid == 0u) {
        for (uint r = 0; r < MV_ROWS_PER_GROUP; ++r) {
            // Per-group fixup placeholder: we hold the cactus codebook-centered-
            // at-zero assumption. The -8 zero-point correction is folded
            // (sum_x_minus8 unused for now; reinstated when we add the
            // proper per-group fixup as part of the CQ4 quant integration).
            float out_val = acc[r];
            if (HAS_BIAS) out_val += float(bias[row_base + r]);
            y[row_base + r] = half(out_val);
        }
    }
}

// ============================================================================
// mul_mm_int4_fp16 — INT4 weights × fp16 multi-token activations (prefill)
// ============================================================================
// Uses simdgroup_matrix<half, 8, 8> for the dense inner FMA, with INT4
// weights dequantized into the simdgroup matrix fragment.
//
// Tile shape per threadgroup:
//   BM = M_TILE  (tokens, e.g. 32 or 64)
//   BN = 32      (output cols)
//   BK = 32      (reduction over K)
// 4 simdgroups per threadgroup (128 threads), each owning an 8×8 output tile.
//
// Implementation strategy (matches MLX Steel):
//   1. Load A (X activations) tile into threadgroup memory, fp16.
//   2. Load B (INT4 weights) tile, dequantize into threadgroup memory as fp16.
//   3. For each K chunk, load A and B fragments into simdgroup_matrix, MMA.
//   4. Accumulate in a simdgroup_matrix<float, 8, 8>.
//   5. Write tile of C back to global memory.
//
// This is a *substantial* kernel; the prefill matmul drives a third of
// total runtime. M1 deliverable is mat-vec (above) + numerical correctness;
// optimized mat-mat lands at M2.

kernel void mul_mm_int4_fp16(
    device const uint16_t * qs       [[buffer(0)]],
    device const half     * scales   [[buffer(1)]],
    device const half     * x        [[buffer(2)]],  // M_TILE × K
    device       half     * y        [[buffer(3)]],  // M_TILE × N
    device const half     * bias     [[buffer(4), function_constant(HAS_BIAS)]],
    uint3   tgid             [[threadgroup_position_in_grid]],
    uint3   tg_threads       [[threads_per_threadgroup]],
    uint3   tid_v            [[thread_position_in_threadgroup]])
{
    const uint tid = tid_v.x;
    // M1 placeholder: route to mat-vec per row. Correctness > speed for now.
    // Replaced at M2 with the simdgroup_matrix tiled version.
    const uint row_base = tgid.x * 4u;
    const uint m        = tgid.y;
    if (row_base >= N || m >= M_TILE) return;

    device const half* xm = x + m * K;
    device       half* ym = y + m * N;

    const uint group_count = K / CQ4_GROUP_SIZE;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint g = 0; g < group_count; ++g) {
        float s[4];
        #pragma unroll
        for (uint r = 0; r < 4u; ++r) s[r] = float(scales[(g * N) + (row_base + r)]);
        const uint w_off = (g * CQ4_UINT16_PER_GROUP) + tid;
        const uint x_off = (g * CQ4_GROUP_SIZE) + (tid * 4u);
        float xv[4];
        xv[0] = float(xm[x_off + 0]);
        xv[1] = float(xm[x_off + 1]) * (1.0f / 16.0f);
        xv[2] = float(xm[x_off + 2]) * (1.0f / 256.0f);
        xv[3] = float(xm[x_off + 3]) * (1.0f / 4096.0f);
        #pragma unroll
        for (uint r = 0; r < 4u; ++r) {
            uint16_t q = qs[(w_off * N) + (row_base + r)];
            acc[r] += s[r] * cq4_nibble_dot(q, xv);
        }
    }
    #pragma unroll
    for (uint r = 0; r < 4u; ++r) {
        acc[r] = simd_sum(acc[r]);
    }
    if (simd_is_first()) {
        for (uint r = 0; r < 4u; ++r) {
            float out_val = acc[r];
            if (HAS_BIAS) out_val += float(bias[row_base + r]);
            ym[row_base + r] = half(out_val);
        }
    }
}
