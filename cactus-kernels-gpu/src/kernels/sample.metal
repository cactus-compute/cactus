/* Sampling kernels — final stage of decode, kept on GPU.
 *
 *   sample_argmax     : greedy. Output is one int32.
 *   sample_top_k_top_p: temperature → softmax → top-k filter → top-p filter
 *                       → multinomial. Output is one int32.
 *
 * Implementation note: vocab_size can be 100K+; we use a reduce pattern
 * (warp + threadgroup) for the argmax and a one-pass softmax / cumulative
 * for top-p. */
#include "common.metal"

constant uint VOCAB_SIZE [[function_constant(70)]];

#define MAX_TG_SIMDGROUPS 32
#define ARGMAX_THREADS    1024

// ============================================================================
// sample_argmax
// ============================================================================
// One threadgroup. Each thread scans VOCAB_SIZE / threads logits, tracking
// the local max. simdgroup reduce, then threadgroup reduce, then write.
kernel void sample_argmax(
    device const half    * logits   [[buffer(0)]],   // [VOCAB_SIZE]
    device       int32_t * out_id   [[buffer(1)]],   // [1]
    uint3 tid_v        [[thread_position_in_threadgroup]],
    uint3 tg_size_v    [[threads_per_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id[[simdgroup_index_in_threadgroup]])
{
    const uint tid     = tid_v.x;
    const uint tg_size = tg_size_v.x;
    threadgroup float   local_max[MAX_TG_SIMDGROUPS];
    threadgroup int32_t local_id [MAX_TG_SIMDGROUPS];

    float   best_v = -INFINITY;
    int32_t best_i = -1;
    const uint stride = tg_size;
    for (uint i = tid; i < VOCAB_SIZE; i += stride) {
        float v = float(logits[i]);
        if (v > best_v) { best_v = v; best_i = int32_t(i); }
    }

    // simdgroup reduce. simd_max gives the max value; we then have all lanes
    // do a vote-style step to find which lane owned that max.
    float warp_max = simd_max(best_v);
    int32_t warp_id = (best_v == warp_max) ? best_i : -1;
    // First lane with non-(-1) wins. Use simd_max on the id with the value-
    // equality tiebreaker.
    warp_id = simd_max(warp_id);  // wins ties toward higher id; good enough

    if (simd_lane_id == 0) {
        local_max[simd_group_id] = warp_max;
        local_id [simd_group_id] = warp_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint nsg = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float m = (simd_lane_id < nsg) ? local_max[simd_lane_id] : -INFINITY;
        int32_t id = (simd_lane_id < nsg) ? local_id[simd_lane_id] : -1;
        float gm = simd_max(m);
        int32_t gi = (m == gm) ? id : -1;
        gi = simd_max(gi);
        if (simd_lane_id == 0) out_id[0] = gi;
    }
}

// ============================================================================
// sample_top_k_top_p (placeholder; full implementation lands at M3)
// ============================================================================
kernel void sample_top_k_top_p(
    device const half    * logits     [[buffer(0)]],
    device       int32_t * out_id     [[buffer(1)]],
    device const uint32_t* rng_state  [[buffer(2)]],
    constant     float   & temperature[[buffer(3)]],
    constant     float   & top_p      [[buffer(4)]],
    constant     uint    & top_k      [[buffer(5)]],
    uint3 tid_v [[thread_position_in_threadgroup]])
{
    const uint tid = tid_v.x;
    // TODO(M3): real top-k / top-p / multinomial.
    // For M1/M2 we fall through to argmax. Greedy gives reproducible
    // single-path decode which is what bring-up needs.
    if (tid == 0) {
        float best_v = -INFINITY;
        int32_t best_i = -1;
        for (uint i = 0; i < VOCAB_SIZE; ++i) {
            float v = float(logits[i]);
            if (v > best_v) { best_v = v; best_i = int32_t(i); }
        }
        out_id[0] = best_i;
    }
}
