#include "kv_compress.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace cactus {
namespace kvcompress {

namespace {

// Round half-to-even like numpy. recent_frac widens from float to a slightly-off double, so snap
// near-halves first to avoid flipping an exact .5 boundary the Python reference keeps.
long py_round(double x) {
    double twice = x * 2.0;
    double twice_rounded = std::nearbyint(twice);
    if (std::fabs(twice - twice_rounded) < 1e-6) x = twice_rounded * 0.5;
    double r = std::nearbyint(x);  // FE_TONEAREST default == round-half-to-even
    return static_cast<long>(r);
}

}  // namespace

void keydiff_score(const float* keys, size_t n, size_t head_dim, float* out) {
    // double accumulation matches the float64 reference.
    std::vector<double> mu(head_dim, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const float* k = keys + i * head_dim;
        for (size_t d = 0; d < head_dim; ++d) mu[d] += static_cast<double>(k[d]);
    }
    double mu_norm = 0.0;
    for (size_t d = 0; d < head_dim; ++d) {
        mu[d] /= static_cast<double>(n);
        mu_norm += mu[d] * mu[d];
    }
    mu_norm = std::sqrt(mu_norm) + 1e-8;
    for (size_t d = 0; d < head_dim; ++d) mu[d] /= mu_norm;

    for (size_t i = 0; i < n; ++i) {
        const float* k = keys + i * head_dim;
        double knorm = 0.0, dot = 0.0;
        for (size_t d = 0; d < head_dim; ++d) {
            double v = static_cast<double>(k[d]);
            knorm += v * v;
            dot += v * mu[d];
        }
        knorm = std::sqrt(knorm) + 1e-8;
        out[i] = static_cast<float>(-(dot / knorm));
    }
}

std::vector<int> keepset_for_head(const float* scores, size_t n, const Params& p) {
    long budget = std::max<long>(1, static_cast<long>(p.abs_budget));
    long B = std::min<long>(budget, static_cast<long>(n));
    long sink = std::min<long>(std::max<long>(static_cast<long>(p.sink), 0), static_cast<long>(n));
    long n_recent = std::min<long>(py_round(static_cast<double>(p.recent_frac) * static_cast<double>(B)),
                                   static_cast<long>(n));

    std::set<long> reserved;
    for (long i = 0; i < sink; ++i) reserved.insert(i);
    for (long i = static_cast<long>(n) - n_recent; i < static_cast<long>(n); ++i)
        if (i >= 0) reserved.insert(i);

    // Reserved can exceed B on a tight budget: keep sink first, then most-recent, until B.
    if (static_cast<long>(reserved.size()) > B) {
        std::vector<long> ordered;
        for (long i = 0; i < sink; ++i) ordered.push_back(i);
        for (long i = static_cast<long>(n) - 1; i >= sink; --i) ordered.push_back(i);
        reserved.clear();
        for (long i : ordered) {
            if (i >= 0 && i < static_cast<long>(n)) reserved.insert(i);
            if (static_cast<long>(reserved.size()) == B) break;
        }
    }

    std::set<long> keep(reserved);
    long remaining = B - static_cast<long>(keep.size());
    if (remaining > 0) {
        std::vector<long> order(n);
        std::iota(order.begin(), order.end(), 0L);
        std::stable_sort(order.begin(), order.end(), [&](long a, long b) {
            return scores[a] > scores[b];
        });
        for (long i : order) {
            if (keep.count(i)) continue;
            keep.insert(i);
            if (--remaining == 0) break;
        }
    }

    std::vector<int> result;
    result.reserve(keep.size());
    for (long i : keep) result.push_back(static_cast<int>(i));
    return result;
}

namespace {

// Precomputed cos/sin per dim-pair, built once so row loops don't recompute pow/trig.
struct RopeRot { std::vector<double> cos, sin; };

std::vector<double> rope_inv_freq(size_t head_dim, double rope_theta) {
    std::vector<double> inv(head_dim / 2);
    for (size_t i = 0; i < inv.size(); ++i)
        inv[i] = std::pow(rope_theta, -(2.0 * static_cast<double>(i)) / static_cast<double>(head_dim));
    return inv;
}

RopeRot rope_rot(const std::vector<double>& inv_freq, double delta_pos) {
    RopeRot r;
    r.cos.resize(inv_freq.size());
    r.sin.resize(inv_freq.size());
    for (size_t i = 0; i < inv_freq.size(); ++i) {
        double a = delta_pos * inv_freq[i];
        r.cos[i] = std::cos(a);
        r.sin[i] = std::sin(a);
    }
    return r;
}

void rope_apply(float* row, const RopeRot& r) {
    size_t half = r.cos.size();
    for (size_t i = 0; i < half; ++i) {
        double x1 = row[i], x2 = row[i + half];
        row[i] = static_cast<float>(x1 * r.cos[i] - x2 * r.sin[i]);
        row[i + half] = static_cast<float>(x2 * r.cos[i] + x1 * r.sin[i]);
    }
}

}  // namespace

void rope_rotate_row(float* row, size_t head_dim, double rope_theta, double delta_pos) {
    rope_apply(row, rope_rot(rope_inv_freq(head_dim, rope_theta), delta_pos));
}

void unrope_head(const float* post_rope, size_t n, size_t head_dim, double rope_theta,
                 float* pre_rope) {
    auto inv = rope_inv_freq(head_dim, rope_theta);
    for (size_t t = 0; t < n; ++t) {
        const float* src = post_rope + t * head_dim;
        float* dst = pre_rope + t * head_dim;
        for (size_t d = 0; d < head_dim; ++d) dst[d] = src[d];
        rope_apply(dst, rope_rot(inv, -static_cast<double>(t)));
    }
}

namespace {

inline double renumber_delta(int abs_pos, size_t rank) {
    return static_cast<double>(rank) - static_cast<double>(abs_pos);
}

// Matches the engine's KV quant convention (quantize_group_fp16_to_int8 in
// cactus-kernels/src/quants.cpp): scale floored at 1e-10, roundf, clamp to [-128, 127].
void requant_row(const float* row, int8_t* dst, float* dsc, size_t head_dim, size_t group_size) {
    size_t groups = (head_dim + group_size - 1) / group_size;
    for (size_t g = 0; g < groups; ++g) {
        size_t lo = g * group_size, hi = std::min(head_dim, lo + group_size);
        float amax = 0.0f;
        for (size_t d = lo; d < hi; ++d) amax = std::max(amax, std::fabs(row[d]));
        float scale = amax / 127.0f;
        if (scale < 1e-10f) scale = 1e-10f;
        dsc[g] = scale;
        float inv = 1.0f / scale;
        for (size_t d = lo; d < hi; ++d) {
            int32_t q = static_cast<int32_t>(std::roundf(row[d] * inv));
            q = std::max(-128, std::min(127, q));
            dst[d] = static_cast<int8_t>(q);
        }
    }
}

void rotate_int8_row_rot(int8_t* int8, float* scale, size_t head_dim, size_t group_size,
                         const RopeRot& rot) {
    std::vector<float> row(head_dim);
    for (size_t d = 0; d < head_dim; ++d) row[d] = static_cast<float>(int8[d]) * scale[d / group_size];
    rope_apply(row.data(), rot);
    requant_row(row.data(), int8, scale, head_dim, group_size);
}

// fill_post(h, post) gathers head h's post-RoPE rows; scoring is shared across fp16/int8.
template <typename FillPost>
std::vector<std::vector<int>> keepsets_per_head(size_t n, size_t kv_heads, size_t head_dim,
                                                double rope_theta, const Params& p, FillPost fill_post) {
    std::vector<float> post(n * head_dim), pre(n * head_dim), scores(n);
    std::vector<std::vector<int>> out;
    out.reserve(kv_heads);
    for (size_t h = 0; h < kv_heads; ++h) {
        fill_post(h, post.data());
        unrope_head(post.data(), n, head_dim, rope_theta, pre.data());
        keydiff_score(pre.data(), n, head_dim, scores.data());
        out.push_back(keepset_for_head(scores.data(), n, p));
    }
    return out;
}

}  // namespace

void compact_fp16(uint16_t* key_rows_u, uint16_t* val_rows_u, size_t kv_heads, size_t head_dim,
                  const std::vector<std::vector<int>>& kept_per_head, double rope_theta) {
    __fp16* key_rows = reinterpret_cast<__fp16*>(key_rows_u);
    __fp16* val_rows = reinterpret_cast<__fp16*>(val_rows_u);
    auto inv = rope_inv_freq(head_dim, rope_theta);
    std::vector<float> krow(head_dim), vrow(head_dim);
    for (size_t h = 0; h < kv_heads; ++h) {
        const std::vector<int>& kept = kept_per_head[h];
        for (size_t rank = 0; rank < kept.size(); ++rank) {
            int abs_pos = kept[rank];
            const __fp16* ksrc = key_rows + (static_cast<size_t>(abs_pos) * kv_heads + h) * head_dim;
            const __fp16* vsrc = val_rows + (static_cast<size_t>(abs_pos) * kv_heads + h) * head_dim;
            for (size_t d = 0; d < head_dim; ++d) { krow[d] = ksrc[d]; vrow[d] = vsrc[d]; }
            rope_apply(krow.data(), rope_rot(inv, renumber_delta(abs_pos, rank)));
            __fp16* kdst = key_rows + (rank * kv_heads + h) * head_dim;
            __fp16* vdst = val_rows + (rank * kv_heads + h) * head_dim;
            for (size_t d = 0; d < head_dim; ++d) {
                kdst[d] = static_cast<__fp16>(krow[d]);
                vdst[d] = static_cast<__fp16>(vrow[d]);
            }
        }
    }
}

void compact_int8(int8_t* int8_rows, float* scale_rows, size_t kv_heads,
                  size_t head_dim, size_t group_size,
                  const std::vector<std::vector<int>>& kept_per_head, double rope_theta,
                  bool renumber) {
    size_t groups = (head_dim + group_size - 1) / group_size;
    size_t int8_stride = kv_heads * head_dim;
    size_t scale_stride = kv_heads * groups;
    auto inv = rope_inv_freq(head_dim, rope_theta);
    std::vector<float> row(head_dim);
    for (size_t h = 0; h < kv_heads; ++h) {
        const std::vector<int>& kept = kept_per_head[h];
        for (size_t rank = 0; rank < kept.size(); ++rank) {
            int abs_pos = kept[rank];
            size_t src_t = static_cast<size_t>(abs_pos);
            const int8_t* src = int8_rows + src_t * int8_stride + h * head_dim;
            const float* ssc = scale_rows + src_t * scale_stride + h * groups;
            for (size_t d = 0; d < head_dim; ++d) row[d] = static_cast<float>(src[d]) * ssc[d / group_size];
            if (renumber)
                rope_apply(row.data(), rope_rot(inv, renumber_delta(abs_pos, rank)));
            int8_t* dst = int8_rows + rank * int8_stride + h * head_dim;
            float* dsc = scale_rows + rank * scale_stride + h * groups;
            requant_row(row.data(), dst, dsc, head_dim, group_size);
        }
    }
}

void rotate_int8_row(int8_t* int8, float* scale, size_t head_dim, size_t group_size,
                     double rope_theta, double delta_pos) {
    rotate_int8_row_rot(int8, scale, head_dim, group_size,
                        rope_rot(rope_inv_freq(head_dim, rope_theta), delta_pos));
}

void rerope_recent_fp16(uint16_t* key_rows_u, size_t kv_heads, size_t head_dim,
                        size_t lo, size_t hi, double rope_theta, double delta_pos) {
    if (delta_pos == 0.0 || hi <= lo) return;
    __fp16* key_rows = reinterpret_cast<__fp16*>(key_rows_u);
    RopeRot rot = rope_rot(rope_inv_freq(head_dim, rope_theta), delta_pos);
    std::vector<float> row(head_dim);
    for (size_t t = lo; t < hi; ++t)
        for (size_t h = 0; h < kv_heads; ++h) {
            __fp16* r = key_rows + (t * kv_heads + h) * head_dim;
            for (size_t d = 0; d < head_dim; ++d) row[d] = r[d];
            rope_apply(row.data(), rot);
            for (size_t d = 0; d < head_dim; ++d) r[d] = static_cast<__fp16>(row[d]);
        }
}

void rerope_recent_int8(int8_t* int8_rows, float* scale_rows, size_t kv_heads, size_t head_dim,
                        size_t group_size, size_t lo, size_t hi, double rope_theta, double delta_pos) {
    if (delta_pos == 0.0 || hi <= lo) return;
    size_t groups = (head_dim + group_size - 1) / group_size;
    size_t int8_stride = kv_heads * head_dim, scale_stride = kv_heads * groups;
    RopeRot rot = rope_rot(rope_inv_freq(head_dim, rope_theta), delta_pos);
    for (size_t t = lo; t < hi; ++t)
        for (size_t h = 0; h < kv_heads; ++h)
            rotate_int8_row_rot(int8_rows + t * int8_stride + h * head_dim,
                                scale_rows + t * scale_stride + h * groups,
                                head_dim, group_size, rot);
}

std::vector<std::vector<int>> keepsets_from_fp16(const uint16_t* key_rows_u, size_t n,
                                                 size_t kv_heads, size_t head_dim,
                                                 double rope_theta, const Params& p) {
    const __fp16* key_rows = reinterpret_cast<const __fp16*>(key_rows_u);
    return keepsets_per_head(n, kv_heads, head_dim, rope_theta, p, [&](size_t h, float* post) {
        for (size_t t = 0; t < n; ++t) {
            const __fp16* src = key_rows + (t * kv_heads + h) * head_dim;
            for (size_t d = 0; d < head_dim; ++d) post[t * head_dim + d] = src[d];
        }
    });
}

std::vector<std::vector<int>> keepsets_from_int8(const int8_t* int8_rows, const float* scale_rows,
                                                 size_t n, size_t kv_heads, size_t head_dim,
                                                 size_t group_size, double rope_theta,
                                                 const Params& p) {
    size_t groups = (head_dim + group_size - 1) / group_size;
    size_t int8_stride = kv_heads * head_dim;
    size_t scale_stride = kv_heads * groups;
    return keepsets_per_head(n, kv_heads, head_dim, rope_theta, p, [&](size_t h, float* post) {
        for (size_t t = 0; t < n; ++t) {
            const int8_t* src = int8_rows + t * int8_stride + h * head_dim;
            const float* ssc = scale_rows + t * scale_stride + h * groups;
            for (size_t d = 0; d < head_dim; ++d)
                post[t * head_dim + d] = static_cast<float>(src[d]) * ssc[d / group_size];
        }
    });
}

bool is_sliding_layer(const std::vector<std::string>& layer_types, size_t li) {
    return li < layer_types.size() && layer_types[li].find("sliding") != std::string::npos;
}

std::vector<size_t> physical_compressible_layers(const std::vector<std::string>& layer_types,
                                                 size_t num_layers, size_t num_kv_shared) {
    auto is_full = [&](size_t i) { return !is_sliding_layer(layer_types, i); };
    std::vector<size_t> full;
    for (size_t i = 0; i < num_layers; ++i) if (is_full(i)) full.push_back(i);

    if (num_kv_shared == 0 || num_kv_shared >= num_layers) return full;
    size_t first_shared = num_layers - num_kv_shared;

    std::set<size_t> sources;
    std::vector<std::pair<std::string, size_t>> last_of_type;
    for (size_t i = 0; i < first_shared && i < layer_types.size(); ++i) {
        bool found = false;
        for (auto& e : last_of_type) if (e.first == layer_types[i]) { e.second = i; found = true; break; }
        if (!found) last_of_type.emplace_back(layer_types[i], i);
    }
    for (auto& e : last_of_type) sources.insert(e.second);

    std::vector<size_t> out;
    for (size_t i : full) if (i < first_shared && !sources.count(i)) out.push_back(i);
    return out;
}

}  // namespace kvcompress
}  // namespace cactus
