#ifndef CACTUS_KV_COMPRESS_H
#define CACTUS_KV_COMPRESS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace cactus {
namespace kvcompress {

struct Params {
    float  recent_frac = 0.30f;
    size_t sink        = 4;
    int    abs_budget  = 0;       // per (layer, kv-head) keep budget, clamped to [1, n]
    std::vector<int> protect;     // positions always kept (special tokens)
};

// Mirrors cactus-graph CacheMetadata.
struct CacheHeader {
    uint64_t current_seq_len;
    uint64_t max_seq_len;
    uint64_t num_kv_heads;
    uint64_t head_dim;
    uint64_t sink_size;
    uint64_t reserved[3];
};
static_assert(sizeof(CacheHeader) == 64, "CacheHeader must be 64 bytes");

void kv_set_simd(bool on);

// score s_i = -cos(k_i, mean(k)); keys pre-RoPE.
void keydiff_score(const float* keys, size_t n, size_t head_dim, float* out);

// Keep-set: sink + recent + top-score middle; ties by ascending index.
std::vector<int> keepset_for_head(const float* scores, size_t n, const Params& p);

// Keys stored POST-RoPE; rotating by delta_pos re-RoPEs to orig+delta_pos.
void rope_rotate_row(float* row, size_t head_dim, double rope_theta, double delta_pos);

void rotate_int8_row(int8_t* int8, float* scale, size_t head_dim, size_t group_size,
                     double rope_theta, double delta_pos);

struct RopeRotation { std::vector<double> cos, sin; };

// Un-rope rotations for [0, n): row t rotates by -t.
std::vector<RopeRotation> unrope_table(size_t n, size_t head_dim, double rope_theta);

// Gather survivors in rank order; re-rope K to 0..B-1, gather V unchanged.
void compact_fp16(uint16_t* key_rows, uint16_t* val_rows, size_t kv_heads, size_t head_dim,
                  const std::vector<std::vector<int>>& kept_per_head,
                  const std::vector<RopeRotation>& unrope);
void compact_fp16(uint16_t* key_rows, uint16_t* val_rows, size_t kv_heads, size_t head_dim,
                  const std::vector<std::vector<int>>& kept_per_head, double rope_theta);

// renumber=true (K) re-ropes each row to its new rank; false (V) gathers as-is.
void compact_int8(int8_t* int8_rows, float* scale_rows, size_t kv_heads,
                  size_t head_dim, size_t group_size,
                  const std::vector<std::vector<int>>& kept_per_head,
                  const std::vector<RopeRotation>& unrope, bool renumber);
void compact_int8(int8_t* int8_rows, float* scale_rows, size_t kv_heads,
                  size_t head_dim, size_t group_size,
                  const std::vector<std::vector<int>>& kept_per_head, double rope_theta,
                  bool renumber);

// Rotate recent K rows [lo, hi) by delta_pos; sink and V stay fixed.
void rerope_recent_fp16(uint16_t* key_rows, size_t kv_heads, size_t head_dim,
                        size_t lo, size_t hi, double rope_theta, double delta_pos);
void rerope_recent_int8(int8_t* int8_rows, float* scale_rows, size_t kv_heads, size_t head_dim,
                        size_t group_size, size_t lo, size_t hi, double rope_theta, double delta_pos);

// Per-head keep-sets from POST-RoPE keys. protect_per_head overrides Params::protect; empty -> shared.
std::vector<std::vector<int>> keepsets_from_fp16(const uint16_t* key_rows, size_t n,
                                                 size_t kv_heads, size_t head_dim,
                                                 const std::vector<RopeRotation>& unrope,
                                                 const Params& p,
                                                 const std::vector<std::vector<int>>& protect_per_head = {});
std::vector<std::vector<int>> keepsets_from_fp16(const uint16_t* key_rows, size_t n,
                                                 size_t kv_heads, size_t head_dim,
                                                 double rope_theta, const Params& p);

std::vector<std::vector<int>> keepsets_from_int8(const int8_t* int8_rows, const float* scale_rows,
                                                 size_t n, size_t kv_heads, size_t head_dim,
                                                 size_t group_size,
                                                 const std::vector<RopeRotation>& unrope,
                                                 const Params& p,
                                                 const std::vector<std::vector<int>>& protect_per_head = {});
std::vector<std::vector<int>> keepsets_from_int8(const int8_t* int8_rows, const float* scale_rows,
                                                 size_t n, size_t kv_heads, size_t head_dim,
                                                 size_t group_size, double rope_theta,
                                                 const Params& p);

std::vector<int> remap_rows_through_kept(const std::vector<int>& rows, const std::vector<int>& kept);

// Per-(compressible layer, head) special rows; lets compaction re-protect each head's own specials.
class SpecialRowTracker {
public:
    void clear() { tracked_len_ = 0; layer_rows_.clear(); valid_ = true; }
    size_t tracked_len() const { return tracked_len_; }
    void set_tracked_len(size_t len) { tracked_len_ = len; }
    bool valid() const { return valid_; }
    void invalidate() { valid_ = false; layer_rows_.clear(); }

    // appended_rows must be >= tracked_len (the still head-aligned region).
    void add_appended(size_t layer, size_t kv_heads, const std::vector<int>& appended_rows);
    const std::vector<std::vector<int>>& protect(size_t layer) const;
    size_t max_reserved(size_t layer, size_t sink, const std::vector<int>& appended) const;
    void remap(size_t layer, const std::vector<std::vector<int>>& kept_per_head);

private:
    size_t tracked_len_ = 0;
    bool valid_ = true;
    std::vector<std::vector<std::vector<int>>> layer_rows_;  // [layer][head] -> sorted rows
};

// Sliding (local) vs full-attention; selects local vs global theta.
bool is_sliding_layer(const std::vector<std::string>& layer_types, size_t li);

// Compressible layers: empty -> all (Qwen); num_kv_shared drops shared consumers+sources (Gemma -> {4,9}).
std::vector<size_t> physical_compressible_layers(const std::vector<std::string>& layer_types,
                                                 size_t num_layers, size_t num_kv_shared);

}  // namespace kvcompress
}  // namespace cactus

#endif  // CACTUS_KV_COMPRESS_H
