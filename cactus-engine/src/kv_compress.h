#ifndef CACTUS_KV_COMPRESS_H
#define CACTUS_KV_COMPRESS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cactus {
namespace kvcompress {

struct Params {
    float  recent_frac = 0.30f;
    size_t sink        = 4;
    int    abs_budget  = 0;       // keep budget B = min(max(1, abs_budget), n) per (layer, kv-head)
};

// 64-byte KV cache header at the front of each K/V buffer (mirrors cactus-graph CacheMetadata).
struct CacheHeader {
    uint64_t current_seq_len;
    uint64_t max_seq_len;
    uint64_t num_kv_heads;
    uint64_t head_dim;
    uint64_t sink_size;
    uint64_t reserved[3];
};
static_assert(sizeof(CacheHeader) == 64, "CacheHeader must be 64 bytes");

// KeyDiff per-token score for one (layer, kv-head): s_i = -cos(k_i, mean(k)).
// keys: row-major [n][head_dim] (pre-RoPE). out: [n]. Mirrors keydiff_score().
void keydiff_score(const float* keys, size_t n, size_t head_dim, float* out);

// KeyDiff keep-set for ONE cell. Returns EXACTLY B = min(max(1, abs_budget), n) sorted unique
// indices: sink + recent + top-score middle. Mirrors keepset_for_head() EXACTLY,
// including the |reserved|>B fallback (sink-first then most-recent). Top-B uses a stable
// descending-score sort, ties broken by ascending index, to match np.argsort(-scores).
std::vector<int> keepset_for_head(const float* scores, size_t n, const Params& p);

// --------------------------------------------------------------------------- //
// Route-B RoPE un-rotation / renumber (rotate_half / NeoX convention)          //
// --------------------------------------------------------------------------- //
// The cache stores POST-RoPE keys. RoPE here is the rotate_half layout used by
// cactus-kernels/src/norms_rope.cpp: for pair (i, i+half), angle = pos * theta^(-2i/d),
//   out[i]      = x[i]*cos - x[i+half]*sin
//   out[i+half] = x[i+half]*cos + x[i]*sin
// Rotating a post-RoPE row by angle for position `delta_pos` gives the row as if RoPE'd at
// (orig_pos + delta_pos). So un-RoPE = rotate by -orig_pos; renumber = rotate by (rank-orig).

// In-place delta rotation of one key row of length head_dim by `delta_pos`.
void rope_rotate_row(float* row, size_t head_dim, double rope_theta, double delta_pos);

// Recover the pre-RoPE keys for one kv-head from post-RoPE rows. `post_rope` is
// [n][head_dim], rows at absolute positions 0..n-1. Writes pre_rope [n][head_dim].
void unrope_head(const float* post_rope, size_t n, size_t head_dim, double rope_theta,
                 float* pre_rope);

// --------------------------------------------------------------------------- //
// Per-head physical compaction + renumber on a live cache buffer (post-RoPE).  //
// Mirrors run_niah_renumber.compact_renumber_kv: gather survivors in rank order,
// renumber K to contiguous positions 0..B-1, gather V unchanged (no RoPE).
// kept_per_head[h] is the sorted survivor index list for kv-head h (length B, equal
// across heads). header.current_seq_len is set to B by the caller.
// --------------------------------------------------------------------------- //

// FP16 cache. key_rows/val_rows point past the 64-byte header at row 0.
// Layout: [max_seq][kv_heads][head_dim] fp16. K is renumbered, V is not.
void compact_fp16(uint16_t* key_rows, uint16_t* val_rows, size_t kv_heads, size_t head_dim,
                  const std::vector<std::vector<int>>& kept_per_head, double rope_theta);

// INT8 buffer (one of K or V). int8 region [max_seq][kv_heads*head_dim]; scales
// [max_seq][kv_heads*groups] float, groups = ceil(head_dim/group_size). Dequant =
// int8 * scale[group]. When `renumber` is true (the K buffer) each gathered row is
// dequantized, renumbered (rotate by rank-abs), and re-quantized per group; when false
// (the V buffer) rows are gathered and re-quantized as-is (no rotation).
void compact_int8(int8_t* int8_rows, float* scale_rows, size_t kv_heads,
                  size_t head_dim, size_t group_size,
                  const std::vector<std::vector<int>>& kept_per_head, double rope_theta,
                  bool renumber);

// Build per-head keep-sets for one layer from POST-RoPE FP16 key rows: un-RoPE -> score ->
// keepset_for_head. Returns n_kv_heads lists, each length B.
std::vector<std::vector<int>> keepsets_from_fp16(const uint16_t* key_rows, size_t n,
                                                 size_t kv_heads, size_t head_dim,
                                                 double rope_theta, const Params& p);

// Same as keepsets_from_fp16 but for the INT8 K buffer: dequantize each row
// (int8 * scale[group]) -> un-RoPE -> score -> keepset_for_head. group_size is the KV
// quantization group size; scale_rows is [max_seq][kv_heads*groups], groups = ceil(head_dim/group_size).
std::vector<std::vector<int>> keepsets_from_int8(const int8_t* int8_rows, const float* scale_rows,
                                                 size_t n, size_t kv_heads, size_t head_dim,
                                                 size_t group_size, double rope_theta,
                                                 const Params& p);

// Physically-compressible layer indices, mirroring MLX compressible_layers(physical=True).
// `layer_types` token contains "sliding" for local layers; everything else is full/global.
// Empty layer_types -> all layers full (Qwen). num_kv_shared drops the shared consumer
// range and the per-type source layers (Gemma -> {4,9}).
std::vector<size_t> physical_compressible_layers(const std::vector<std::string>& layer_types,
                                                 size_t num_layers, size_t num_kv_shared);

}  // namespace kvcompress
}  // namespace cactus

#endif  // CACTUS_KV_COMPRESS_H
