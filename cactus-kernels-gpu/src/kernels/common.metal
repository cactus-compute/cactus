/* Cactus GPU kernels — shared utilities.
 *
 * Conventions:
 *   - We target Metal 2.4+ (Apple Silicon M1+ / A14+).
 *   - All hot-path kernels use fp16 for activations and weights where possible;
 *     accumulators are fp32 to avoid loss of precision in long K dims.
 *   - Threadgroup size is always a multiple of 32 (one Apple simdgroup).
 *   - `simd_sum`, `simd_max`, and `simdgroup_matrix` are the three Apple-
 *     specific primitives we lean on hardest. NVIDIA / AMD do not have these.
 */
#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

// ----------------------------------------------------------------------------
// CQ4 (Cactus quantized 4-bit) constants.
// ----------------------------------------------------------------------------
// One CQ4 group covers 128 source values, packed into 64 bytes (4 bits per
// value), stored as 32 × uint16_t to match the bitwise-mask dequant trick:
// (qs & 0x000f), (qs & 0x00f0), (qs & 0x0f00), (qs & 0xf000) extract the four
// nibbles of one uint16 in one pass.  Per-group scale is fp16.
#define CQ4_GROUP_SIZE 128u
#define CQ4_NIBBLES_PER_GROUP 128u
#define CQ4_UINT16_PER_GROUP  (CQ4_NIBBLES_PER_GROUP / 4u)  // 32
#define CQ4_BYTES_PER_GROUP   (CQ4_NIBBLES_PER_GROUP / 2u)  // 64

// ----------------------------------------------------------------------------
// Reductions
// ----------------------------------------------------------------------------
// Sum all lanes of a 32-thread simdgroup. Used for partial dot-products,
// RMS-norm sum-of-squares, softmax denominators. Frees us from needing
// threadgroup-memory reductions for warp-local sums.
inline float simd_sum_f32(float v) { return simd_sum(v); }
inline float simd_max_f32(float v) { return simd_max(v); }

// Two-stage reduce: first within each simdgroup (free via simd_sum), then
// across simdgroups via threadgroup memory. Used when the threadgroup has
// multiple simdgroups (e.g., 128 threads = 4 simdgroups).
//
// `slot` must point to a threadgroup-memory array of at least num_simdgroups
// floats. `simd_group_id` and `simd_lane_id` are the per-thread builtins.
inline float threadgroup_sum_f32(
    float          v,
    threadgroup float* slot,
    uint           simd_group_id,
    uint           simd_lane_id,
    uint           num_simdgroups)
{
    v = simd_sum(v);
    if (simd_lane_id == 0) slot[simd_group_id] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    if (simd_group_id == 0) {
        float x = (simd_lane_id < num_simdgroups) ? slot[simd_lane_id] : 0.0f;
        x = simd_sum(x);
        if (simd_lane_id == 0) slot[0] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc = slot[0];
    return acc;
}

inline float threadgroup_max_f32(
    float          v,
    threadgroup float* slot,
    uint           simd_group_id,
    uint           simd_lane_id,
    uint           num_simdgroups)
{
    v = simd_max(v);
    if (simd_lane_id == 0) slot[simd_group_id] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float x = (simd_lane_id < num_simdgroups) ? slot[simd_lane_id] : -INFINITY;
        x = simd_max(x);
        if (simd_lane_id == 0) slot[0] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return slot[0];
}

// ----------------------------------------------------------------------------
// CQ4 nibble dequant — the llama.cpp / MLX trick.
//
// `qs` is one uint16 holding 4 nibbles (4 packed weight values).
// `x` is a thread-local float[4] of pre-shift-scaled inputs:
//     x[0] = input * 1
//     x[1] = input * (1.0f / 16.0f)   (compensates for nibble 1 being shifted 4 bits)
//     x[2] = input * (1.0f / 256.0f)  (nibble 2: shifted 8 bits)
//     x[3] = input * (1.0f / 4096.0f) (nibble 3: shifted 12 bits)
// We then mask each nibble *in its native position* and multiply by the
// corresponding pre-shifted x. Result: no per-nibble shift needed in the
// hot loop. The "-8" zero-point correction (CQ4 quants are stored in 0..15
// but represent -8..7) is folded into a separate `sum_x` term:
//   final = scale * (cq4_nibble_dot(qs, x) - 8 * sum_x)
// where sum_x is the sum of the 4 inputs (no shifts).
//
// Returns the partial dot product for these 4 weight values.
// ----------------------------------------------------------------------------
inline float cq4_nibble_dot(uint16_t qs, thread const float* x) {
    return (float(qs & 0x000fu) * x[0]) +
           (float(qs & 0x00f0u) * x[1]) +
           (float(qs & 0x0f00u) * x[2]) +
           (float(qs & 0xf000u) * x[3]);
}
