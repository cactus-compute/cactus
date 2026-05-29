/* Flash Attention — single-pass online softmax with KV cache walk.
 *
 * For each Q tile (BQ rows), iterate over K-tiles in the KV cache. For each
 * K-tile:
 *   1. Compute QK^T (BQ × BK matmul, fp16 in, fp32 accumulate).
 *   2. Apply mask + softcap.
 *   3. Online softmax: track running max m and running sum l; rescale
 *      previously-accumulated O when m changes.
 *   4. P = exp(QK - m); O += P @ V; l += sum(P).
 * Final write: O / l with optional weight scale.
 *
 * Kernel name encodes head_dim_q (dk) and head_dim_v (dv) so the compiler
 * sees the head_dim as a compile-time constant in the inner loop. We
 * instantiate one variant per common (dk, dv) pair via Metal templates.
 *
 * Decode hot path: BQ = 1. Prefill: BQ = up to 64.
 *
 * Function constants:
 *   NUM_GROUPS  : num_query_heads / num_kv_heads (GQA factor)
 *   CAUSAL      : true for causal LMs
 *   HAS_SOFTCAP : true for Gemma-style attention soft-capping
 */
#include "common.metal"

constant uint NUM_GROUPS  [[function_constant(32)]];
constant bool CAUSAL      [[function_constant(33)]];
constant bool HAS_SOFTCAP [[function_constant(34)]];
constant bool HAS_MASK    [[function_constant(35)]];

// ----------------------------------------------------------------------------
// Single-token / per-query-head decode flash attention.
// Specialized per (DK, DV). 32-thread simdgroup.
// One threadgroup per query head. K, V are read from the cache; the cache
// length is passed in.
//
// In:    q[1, HEADS_Q, DK]
//        cache_k[num_kv_heads, max_seq, DK]
//        cache_v[num_kv_heads, max_seq, DV]
// Out:   o[1, HEADS_Q, DV]
// ----------------------------------------------------------------------------
template <uint DK, uint DV>
void flash_attn_decode_impl(
    device const half  * q          ,    // [HEADS_Q × DK]
    device const half  * cache_k    ,    // [HEADS_KV, MAX_SEQ, DK]
    device const half  * cache_v    ,    // [HEADS_KV, MAX_SEQ, DV]
    device const half  * mask_or_null,   // optional, [seq_k] or nullptr
    device       half  * o          ,    // [HEADS_Q × DV]
    constant     float & scale      ,    // 1/sqrt(DK)
    constant     uint  & seq_k      ,    // current KV cache length
    constant     uint  & max_seq    ,    // KV cache stride
    uint qh        ,                     // query head index (= tgid.x)
    uint tid       ,                     // thread in simdgroup, 0..31
    uint num_threads,
    constant float & softcap         )
{
    // KV head this query head reads from (GQA).
    const uint kv_head = qh / NUM_GROUPS;

    device const half* qrow = q + qh * DK;
    device const half* k_base = cache_k + kv_head * max_seq * DK;
    device const half* v_base = cache_v + kv_head * max_seq * DV;
    device       half* orow = o + qh * DV;

    // Each thread holds a slice of Q in registers. DK/32 elements per thread.
    float q_reg[DK / 32];
    #pragma unroll
    for (uint i = 0; i < DK / 32; ++i) {
        q_reg[i] = float(qrow[tid + i * 32]) * scale;
    }

    // Per-thread partial accumulator for O. We process DV/32 elements
    // (DV must be a multiple of 32). On reduce, each thread owns its slice.
    float o_reg[DV / 32];
    #pragma unroll
    for (uint i = 0; i < DV / 32; ++i) o_reg[i] = 0.0f;

    float m = -INFINITY;   // running max
    float l = 0.0f;        // running sum of exp(qk - m)

    // Walk all KV positions. Each position: compute one qk dot product,
    // online-softmax update, multiply by v.
    for (uint kpos = 0; kpos < seq_k; ++kpos) {
        if (CAUSAL && kpos > seq_k - 1) break;  // single-token decode: causal trivially OK

        // qk = dot(q, k_row). Each thread holds DK/32 q entries; same for k.
        device const half* krow = k_base + kpos * DK;
        float partial = 0.0f;
        #pragma unroll
        for (uint i = 0; i < DK / 32; ++i) {
            partial += q_reg[i] * float(krow[tid + i * 32]);
        }
        float qk = simd_sum(partial);

        if (HAS_SOFTCAP) {
            qk = softcap * precise::tanh(qk / softcap);
        }
        // HAS_MASK is a compile-time constant; when false the branch (and
        // its load) is eliminated entirely, so a nullptr mask binding is safe.
        if (HAS_MASK) {
            qk += float(mask_or_null[kpos]);
        }

        // Online softmax update.
        float m_new = max(m, qk);
        float alpha = precise::exp(m - m_new);   // rescale prev O
        float p     = precise::exp(qk - m_new);
        l = l * alpha + p;
        m = m_new;

        // O_reg = alpha * O_reg + p * V_row
        device const half* vrow = v_base + kpos * DV;
        #pragma unroll
        for (uint i = 0; i < DV / 32; ++i) {
            o_reg[i] = alpha * o_reg[i] + p * float(vrow[tid + i * 32]);
        }
    }

    // Normalize. Each thread writes its slice of O.
    const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    #pragma unroll
    for (uint i = 0; i < DV / 32; ++i) {
        orow[tid + i * 32] = half(o_reg[i] * inv_l);
    }
}

// ----------------------------------------------------------------------------
// Instantiations: name encodes (dk, dv).
// ----------------------------------------------------------------------------
#define INSTANTIATE_FLASH_ATTN_DECODE(DK, DV)                                       \
kernel void flash_attn_dk##DK##_dv##DV(                                             \
    device const half  * q          [[buffer(0)]],                                  \
    device const half  * cache_k    [[buffer(1)]],                                  \
    device const half  * cache_v    [[buffer(2)]],                                  \
    device const half  * mask       [[buffer(3)]],                                  \
    device       half  * o          [[buffer(4)]],                                  \
    constant     float & scale      [[buffer(5)]],                                  \
    constant     uint  & seq_k      [[buffer(6)]],                                  \
    constant     uint  & max_seq    [[buffer(7)]],                                  \
    constant     float & softcap    [[buffer(8)]],                                  \
    uint3 tgid    [[threadgroup_position_in_grid]],                                 \
    uint3 tid_v   [[thread_position_in_threadgroup]],                               \
    uint3 ntid_v  [[threads_per_threadgroup]])                                      \
{                                                                                   \
    flash_attn_decode_impl<DK, DV>(q, cache_k, cache_v, mask, o, scale,             \
                                    seq_k, max_seq, tgid.x, tid_v.x, ntid_v.x,     \
                                    softcap);                                       \
}

// Common head-dim combinations. Add more as needed.
INSTANTIATE_FLASH_ATTN_DECODE(64,  64)
INSTANTIATE_FLASH_ATTN_DECODE(128, 128)
INSTANTIATE_FLASH_ATTN_DECODE(256, 256)

// TODO(M2): tiled multi-Q (BQ > 1) variant for prefill. The decode kernel
// above runs once per query head per token; for prefill we'd encode one
// dispatch per Q-tile of 64 tokens and reuse the K/V loads across them.
