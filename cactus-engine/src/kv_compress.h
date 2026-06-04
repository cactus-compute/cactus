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

// Test hook: force the scalar fallback to compare against the NEON path on a NEON build.
void kv_set_simd(bool on);

// KeyDiff per-token score for one (layer, kv-head): s_i = -cos(k_i, mean(k)).
// keys: row-major [n][head_dim] (pre-RoPE). out: [n].
void keydiff_score(const float* keys, size_t n, size_t head_dim, float* out);

// Keep-set for one cell: B = min(max(1, abs_budget), n) sorted unique indices (sink + recent +
// top-score middle). Top-B is stable by descending score, ties by ascending index (np.argsort(-s)).
std::vector<int> keepset_for_head(const float* scores, size_t n, const Params& p);

// Keys are stored POST-RoPE (rotate_half, see cactus-kernels/src/norms_rope.cpp). Rotating a row by
// delta_pos re-RoPEs it to (orig + delta_pos): un-RoPE = -orig, renumber = (rank - orig).
void rope_rotate_row(float* row, size_t head_dim, double rope_theta, double delta_pos);

// INT8 analog of rope_rotate_row: dequant, rotate, re-quantize in place. scale is [groups] per row.
void rotate_int8_row(int8_t* int8, float* scale, size_t head_dim, size_t group_size,
                     double rope_theta, double delta_pos);

// Precomputed cos/sin per dim-pair for a fixed RoPE delta.
struct RopeRotation { std::vector<double> cos, sin; };

// Un-rope rotations for positions [0, n) at (head_dim, theta): row t rotates by -t. Built once and
// shared across a compaction's keep-set scoring -- all compressible (global) layers share theta.
std::vector<RopeRotation> unrope_table(size_t n, size_t head_dim, double rope_theta);

// Per-head compaction: gather survivors in rank order, renumber K to 0..B-1, gather V unchanged.
// kept_per_head[h] is head h's survivor indices (length B); rows point past the header, [max_seq][kv_heads][head_dim].
void compact_fp16(uint16_t* key_rows, uint16_t* val_rows, size_t kv_heads, size_t head_dim,
                  const std::vector<std::vector<int>>& kept_per_head, double rope_theta);

// INT8 analog of compact_fp16 (K or V). renumber=true (K) rotates each gathered row by (rank - abs);
// false (V) gathers as-is. scales are [max_seq][kv_heads*groups], groups = ceil(head_dim/group_size).
void compact_int8(int8_t* int8_rows, float* scale_rows, size_t kv_heads,
                  size_t head_dim, size_t group_size,
                  const std::vector<std::vector<int>>& kept_per_head, double rope_theta,
                  bool renumber);

// Rotate recent K rows [lo, hi) by delta_pos so a windowed cache tracks the renumbered frontier;
// sink rows [0, lo) and V stay fixed. No-op when delta_pos == 0 or hi <= lo.
void rerope_recent_fp16(uint16_t* key_rows, size_t kv_heads, size_t head_dim,
                        size_t lo, size_t hi, double rope_theta, double delta_pos);
void rerope_recent_int8(int8_t* int8_rows, float* scale_rows, size_t kv_heads, size_t head_dim,
                        size_t group_size, size_t lo, size_t hi, double rope_theta, double delta_pos);

// Per-head keep-sets for one layer from POST-RoPE FP16 key rows: un-RoPE -> score -> keepset. The
// theta overload builds the un-rope table itself; pass a shared table to reuse it across layers.
std::vector<std::vector<int>> keepsets_from_fp16(const uint16_t* key_rows, size_t n,
                                                 size_t kv_heads, size_t head_dim,
                                                 const std::vector<RopeRotation>& unrope,
                                                 const Params& p);
std::vector<std::vector<int>> keepsets_from_fp16(const uint16_t* key_rows, size_t n,
                                                 size_t kv_heads, size_t head_dim,
                                                 double rope_theta, const Params& p);

// As keepsets_from_fp16 but dequantizing the INT8 K buffer first; scale_rows [max_seq][kv_heads*groups].
std::vector<std::vector<int>> keepsets_from_int8(const int8_t* int8_rows, const float* scale_rows,
                                                 size_t n, size_t kv_heads, size_t head_dim,
                                                 size_t group_size,
                                                 const std::vector<RopeRotation>& unrope,
                                                 const Params& p);
std::vector<std::vector<int>> keepsets_from_int8(const int8_t* int8_rows, const float* scale_rows,
                                                 size_t n, size_t kv_heads, size_t head_dim,
                                                 size_t group_size, double rope_theta,
                                                 const Params& p);

// True for sliding (local) layers. False for full-attention layers -- including KV-shared global
// *source* layers excluded from compaction -- so the re-rope picks local vs global theta correctly.
bool is_sliding_layer(const std::vector<std::string>& layer_types, size_t li);

// Physically-compressible layer indices. Empty layer_types -> all layers (Qwen). num_kv_shared
// drops the shared consumer range and the per-type source layers (Gemma -> {4,9}).
std::vector<size_t> physical_compressible_layers(const std::vector<std::string>& layer_types,
                                                 size_t num_layers, size_t num_kv_shared);

}  // namespace kvcompress
}  // namespace cactus

#endif  // CACTUS_KV_COMPRESS_H
