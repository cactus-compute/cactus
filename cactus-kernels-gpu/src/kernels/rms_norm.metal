/* RMSNorm with fused weight scaling.
 *
 * Pattern stolen from MLX rms_norm.metal: simd_sum for warp reduction
 * (free in hardware), threadgroup memory only for the final cross-warp
 * reduction. 2 barriers total.
 *
 *   out[i] = w[i] * x[i] * rsqrt( mean(x^2) + eps )
 *
 * One threadgroup per row. AXIS_SIZE is the last-dim (hidden_dim).
 * Each thread processes N_READS contiguous elements; total threads per
 * group = AXIS_SIZE / N_READS.
 */
#include "common.metal"

constant uint  AXIS_SIZE [[function_constant(20)]];

#define N_READS 4
#define MAX_TG_SIMDGROUPS 32  // 32 * 32 threads = 1024 max threadgroup size

kernel void rms_norm_fp16(
    device const half  * x         [[buffer(0)]],
    device const half  * w         [[buffer(1)]],     // gain (gamma)
    device       half  * out       [[buffer(2)]],
    constant     float & eps       [[buffer(3)]],
    uint3   tgid              [[threadgroup_position_in_grid]],
    uint3   tid_v             [[thread_position_in_threadgroup]],
    uint3   tg_size_v         [[threads_per_threadgroup]],
    uint    simd_lane_id      [[thread_index_in_simdgroup]],
    uint    simd_group_id     [[simdgroup_index_in_threadgroup]])
{
    const uint tid = tid_v.x;
    const uint tg_size = tg_size_v.x;
    const uint row = tgid.x;

    threadgroup float local_sums[MAX_TG_SIMDGROUPS];
    threadgroup float local_inv_mean[1];

    device const half* x_row   = x   + row * AXIS_SIZE;
    device       half* out_row = out + row * AXIS_SIZE;

    // Step 1: each thread reads N_READS elements, sum-of-squares.
    const uint base = tid * N_READS;
    float acc = 0.0f;
    if (base < AXIS_SIZE) {
        #pragma unroll
        for (uint i = 0; i < N_READS; ++i) {
            if (base + i < AXIS_SIZE) {
                float v = float(x_row[base + i]);
                acc += v * v;
            }
        }
    }

    // Step 2: warp reduce — free (1 instruction).
    acc = simd_sum(acc);

    // Step 3: cross-warp reduce via threadgroup memory.
    const uint num_simdgroups = (tg_size + 31u) / 32u;
    if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float x = (simd_lane_id < num_simdgroups) ? local_sums[simd_lane_id] : 0.0f;
        x = simd_sum(x);
        if (simd_lane_id == 0) {
            local_inv_mean[0] = precise::rsqrt(x / float(AXIS_SIZE) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_mean = local_inv_mean[0];

    // Step 4: fused write with weight gain.
    if (base < AXIS_SIZE) {
        #pragma unroll
        for (uint i = 0; i < N_READS; ++i) {
            if (base + i < AXIS_SIZE) {
                float gain = float(w[base + i]);
                float v    = float(x_row[base + i]) * inv_mean * gain;
                out_row[base + i] = half(v);
            }
        }
    }
}
