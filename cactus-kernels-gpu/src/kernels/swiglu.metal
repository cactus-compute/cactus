/* SwiGLU: out = silu(gate) * up
 *
 *   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Used in LLaMA / Gemma / Qwen MLP. Single-pass, fully fused, one thread
 * per element.
 */
#include "common.metal"

constant uint SW_HIDDEN_DIM [[function_constant(60)]];

kernel void swiglu_fwd(
    device const half * gate   [[buffer(0)]],   // [tokens, hidden_dim]
    device const half * up     [[buffer(1)]],   // [tokens, hidden_dim]
    device       half * out    [[buffer(2)]],   // [tokens, hidden_dim]
    uint3 gid_v [[thread_position_in_grid]])
{
    const uint gid = gid_v.x;
    if (gid >= SW_HIDDEN_DIM) return;  // multi-token: caller dispatches per token
    const float g = float(gate[gid]);
    const float u = float(up[gid]);
    const float silu = g / (1.0f + precise::exp(-g));
    out[gid] = half(silu * u);
}
