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

// 64-byte KV cache header mirroring cactus-graph CacheMetadata.
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
// keys: row-major [n][head_dim] (pre-RoPE). out: [n].
void keydiff_score(const float* keys, size_t n, size_t head_dim, float* out);

// Keep-set for one cell: exactly B = min(max(1, abs_budget), n) sorted unique indices
// (sink + recent + top-score middle). Top-B sort is stable descending-score, ties broken
// by ascending index, to match np.argsort(-scores).
std::vector<int> keepset_for_head(const float* scores, size_t n, const Params& p);

// The cache stores POST-RoPE keys (rotate_half layout, see cactus-kernels/src/norms_rope.cpp):
// for pair (i, i+half), angle = pos * theta^(-2i/d),
//   out[i]      = x[i]*cos - x[i+half]*sin
//   out[i+half] = x[i+half]*cos + x[i]*sin
// Rotating a post-RoPE row by `delta_pos` yields the row as if RoPE'd at (orig_pos + delta_pos),
// so un-RoPE = rotate by -orig_pos and renumber = rotate by (rank - orig).
void rope_rotate_row(float* row, size_t head_dim, double rope_theta, double delta_pos);

// INT8 analog of rope_rotate_row: dequant (int8 * per-group scale), rotate, re-quantize in place.
// `scale` is the row's [groups] scales, groups = ceil(head_dim/group_size).
void rotate_int8_row(int8_t* int8, float* scale, size_t head_dim, size_t group_size,
                     double rope_theta, double delta_pos);

void unrope_head(const float* post_rope, size_t n, size_t head_dim, double rope_theta,
                 float* pre_rope);

// Per-head physical compaction: gather survivors in rank order, renumber K to contiguous
// positions 0..B-1, gather V unchanged. kept_per_head[h] is the survivor index list for
// kv-head h (length B, equal across heads); the caller sets header.current_seq_len to B.
// key_rows/val_rows point past the 64-byte header; layout [max_seq][kv_heads][head_dim].
void compact_fp16(uint16_t* key_rows, uint16_t* val_rows, size_t kv_heads, size_t head_dim,
                  const std::vector<std::vector<int>>& kept_per_head, double rope_theta);

// INT8 buffer (one of K or V). int8 region [max_seq][kv_heads*head_dim]; scales
// [max_seq][kv_heads*groups], groups = ceil(head_dim/group_size). `renumber` true (K buffer)
// rotates each gathered row by (rank - abs); false (V buffer) gathers as-is.
void compact_int8(int8_t* int8_rows, float* scale_rows, size_t kv_heads,
                  size_t head_dim, size_t group_size,
                  const std::vector<std::vector<int>>& kept_per_head, double rope_theta,
                  bool renumber);

// Rotate recent K rows [lo, hi) by `delta_pos` so a windowed cache tracks the renumbered global
// frontier after compaction. Rows [0, lo) (the sink) stay fixed; V is never rotated. `rope_theta`
// is the LOCAL theta for sliding layers. No-op when delta_pos == 0 or hi <= lo.
void rerope_recent_fp16(uint16_t* key_rows, size_t kv_heads, size_t head_dim,
                        size_t lo, size_t hi, double rope_theta, double delta_pos);
void rerope_recent_int8(int8_t* int8_rows, float* scale_rows, size_t kv_heads, size_t head_dim,
                        size_t group_size, size_t lo, size_t hi, double rope_theta, double delta_pos);

// Per-head keep-sets for one layer from POST-RoPE FP16 key rows: un-RoPE -> score -> keepset.
std::vector<std::vector<int>> keepsets_from_fp16(const uint16_t* key_rows, size_t n,
                                                 size_t kv_heads, size_t head_dim,
                                                 double rope_theta, const Params& p);

// As keepsets_from_fp16 but dequantizing the INT8 K buffer first. scale_rows is
// [max_seq][kv_heads*groups], groups = ceil(head_dim/group_size).
std::vector<std::vector<int>> keepsets_from_int8(const int8_t* int8_rows, const float* scale_rows,
                                                 size_t n, size_t kv_heads, size_t head_dim,
                                                 size_t group_size, double rope_theta,
                                                 const Params& p);

// Sliding (local) layers re-rope with the LOCAL theta during compaction; full-attention layers --
// including KV-shared global *source* layers that are excluded from compaction yet still ride the
// renumbered frame -- return false and re-rope with the GLOBAL theta. Empty layer_types -> global.
bool is_sliding_layer(const std::vector<std::string>& layer_types, size_t li);

// Physically-compressible layer indices. Empty layer_types -> all layers (Qwen). num_kv_shared
// drops the shared consumer range and the per-type source layers (Gemma -> {4,9}).
std::vector<size_t> physical_compressible_layers(const std::vector<std::string>& layer_types,
                                                 size_t num_layers, size_t num_kv_shared);

}  // namespace kvcompress
}  // namespace cactus

#endif  // CACTUS_KV_COMPRESS_H
