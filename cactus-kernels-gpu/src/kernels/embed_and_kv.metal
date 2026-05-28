/* Embedding lookup + KV cache append.
 *
 * Both kernels touch tiny amounts of compute relative to the data they
 * move. The win is staying resident on GPU so we don't sync to CPU. */
#include "common.metal"

constant uint EMBED_HIDDEN_DIM [[function_constant(71)]];
constant uint NUM_KV_HEADS     [[function_constant(50)]];
constant uint KV_HEAD_DIM      [[function_constant(51)]];

// ============================================================================
// embed_lookup — gather embedding row by token id
// ============================================================================
// Grid: (EMBED_HIDDEN_DIM, num_tokens, 1), one thread per (dim, token).
kernel void embed_lookup(
    device const half    * embed_table [[buffer(0)]],   // [vocab, hidden]
    device const int32_t * token_ids   [[buffer(1)]],   // [num_tokens]
    device       half    * out         [[buffer(2)]],   // [num_tokens, hidden]
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint t = gid.y;
    if (d >= EMBED_HIDDEN_DIM) return;
    const int32_t token = token_ids[t];
    out[t * EMBED_HIDDEN_DIM + d] = embed_table[uint(token) * EMBED_HIDDEN_DIM + d];
}

// ============================================================================
// kv_cache_append — write new K, V at the right offset
// ============================================================================
// Grid: (KV_HEAD_DIM, NUM_KV_HEADS, num_new_tokens), one thread per element.
// New tokens are appended at cache_offset; cache layout is
// [num_kv_heads, max_seq_len, head_dim].
kernel void kv_cache_append(
    device const half  * new_k        [[buffer(0)]],   // [num_new, num_kv_heads, head_dim]
    device const half  * new_v        [[buffer(1)]],
    device       half  * cache_k      [[buffer(2)]],   // [num_kv_heads, max_seq, head_dim]
    device       half  * cache_v      [[buffer(3)]],
    constant     uint  & cache_offset [[buffer(4)]],
    constant     uint  & max_seq      [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    if (d >= KV_HEAD_DIM || h >= NUM_KV_HEADS) return;

    const uint cache_pos = cache_offset + t;
    if (cache_pos >= max_seq) return;

    const uint src = (t * NUM_KV_HEADS + h) * KV_HEAD_DIM + d;
    const uint dst = (h * max_seq + cache_pos) * KV_HEAD_DIM + d;
    cache_k[dst] = new_k[src];
    cache_v[dst] = new_v[src];
}
