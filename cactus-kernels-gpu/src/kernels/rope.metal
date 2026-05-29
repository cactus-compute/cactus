/* RoPE — Rotary Position Embedding, in-place on Q and K.
 *
 * For each head, each (pair of) positions in head_dim:
 *   theta = position * theta_base^(-2i/head_dim)
 *   [q0, q1] = [q0*cos - q1*sin, q0*sin + q1*cos]
 *
 * Two layouts:
 *   - IS_NEOX = false  ("GPT-J interleaved"):  pairs are (q[2i], q[2i+1])
 *   - IS_NEOX = true   ("LLaMA / Gemma"):      pairs are (q[i], q[i + head_dim/2])
 *
 * One threadgroup per (token, head). Threads process head_dim / 2 pairs.
 */
#include "common.metal"

constant uint  ROPE_HEAD_DIM [[function_constant(40)]];
constant bool  IS_NEOX       [[function_constant(41)]];
constant float THETA         [[function_constant(42)]];

kernel void rope_apply(
    device       half  * q          [[buffer(0)]],   // [tokens, heads, head_dim]
    device       half  * k          [[buffer(1)]],   // [tokens, num_kv_heads, head_dim]
    device const int32_t* positions [[buffer(2)]],   // [tokens]
    constant     uint  & num_q_heads  [[buffer(3)]],
    constant     uint  & num_kv_heads [[buffer(4)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid_v [[thread_position_in_threadgroup]])
{
    const uint tid   = tid_v.x;
    const uint token = tgid.x;
    const uint head  = tgid.y;                       // joint over q and k heads
    const uint pair  = tid;                          // 0..head_dim/2 - 1
    if (pair >= ROPE_HEAD_DIM / 2u) return;

    const int32_t pos = positions[token];
    // theta_i = pos * THETA^(-2i / ROPE_HEAD_DIM)
    const float exponent = (-2.0f * float(pair)) / float(ROPE_HEAD_DIM);
    const float freq = precise::pow(THETA, exponent);
    const float angle = float(pos) * freq;
    const float c = precise::cos(angle);
    const float s = precise::sin(angle);

    // Apply to Q if head < num_q_heads
    if (head < num_q_heads) {
        device half* qrow = q + (token * num_q_heads + head) * ROPE_HEAD_DIM;
        uint i0, i1;
        if (IS_NEOX) { i0 = pair; i1 = pair + ROPE_HEAD_DIM / 2u; }
        else         { i0 = pair * 2u; i1 = pair * 2u + 1u; }
        float a = float(qrow[i0]);
        float b = float(qrow[i1]);
        qrow[i0] = half(a * c - b * s);
        qrow[i1] = half(a * s + b * c);
    }
    // Apply to K if head < num_kv_heads
    if (head < num_kv_heads) {
        device half* krow = k + (token * num_kv_heads + head) * ROPE_HEAD_DIM;
        uint i0, i1;
        if (IS_NEOX) { i0 = pair; i1 = pair + ROPE_HEAD_DIM / 2u; }
        else         { i0 = pair * 2u; i1 = pair * 2u + 1u; }
        float a = float(krow[i0]);
        float b = float(krow[i1]);
        krow[i0] = half(a * c - b * s);
        krow[i1] = half(a * s + b * c);
    }
}
