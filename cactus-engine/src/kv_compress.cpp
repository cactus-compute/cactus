#include "kv_compress.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace cactus {
namespace kvcompress {

namespace {

// Python's round()/numpy round half-to-even. round(recent_frac*B) must match the reference
// exactly, so reproduce banker's rounding rather than std::round (which rounds half away from
// zero). recent_frac arrives as float and widens to a slightly-off double (e.g. 0.30f ->
// 0.30000001...), which can flip the rounding of an exact half boundary (0.3*15 = 4.5 but
// 0.30000001*15 = 4.50000018). Snap values within a tiny epsilon of a half-integer back onto it
// before banker's rounding so the result matches the Python double reference exactly.
long py_round(double x) {
    double twice = x * 2.0;
    double twice_rounded = std::nearbyint(twice);
    if (std::fabs(twice - twice_rounded) < 1e-6) x = twice_rounded * 0.5;
    double r = std::nearbyint(x);  // honors the default FE_TONEAREST (round-half-to-even)
    return static_cast<long>(r);
}

}  // namespace

void keydiff_score(const float* keys, size_t n, size_t head_dim, float* out) {
    // mu = mean key; s_i = -cos(k_i, mu). Computed in double to match the float64 reference.
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

    // Reserved may exceed B (large recent_frac + sink on a tight budget): keep sink first,
    // then the most-recent tokens, until B.
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
        // argsort(-scores, kind="stable"): descending score, ties broken by ascending index.
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

// --------------------------------------------------------------------------- //
// RoPE (rotate_half) helpers                                                   //
// --------------------------------------------------------------------------- //
void rope_rotate_row(float* row, size_t head_dim, double rope_theta, double delta_pos) {
    size_t half = head_dim / 2;
    for (size_t i = 0; i < half; ++i) {
        double inv = std::pow(rope_theta, -(2.0 * static_cast<double>(i)) / static_cast<double>(head_dim));
        double ang = delta_pos * inv;
        double c = std::cos(ang), s = std::sin(ang);
        double x1 = row[i], x2 = row[i + half];
        row[i] = static_cast<float>(x1 * c - x2 * s);
        row[i + half] = static_cast<float>(x2 * c + x1 * s);
    }
}

void unrope_head(const float* post_rope, size_t n, size_t head_dim, double rope_theta,
                 float* pre_rope) {
    for (size_t t = 0; t < n; ++t) {
        const float* src = post_rope + t * head_dim;
        float* dst = pre_rope + t * head_dim;
        for (size_t d = 0; d < head_dim; ++d) dst[d] = src[d];
        rope_rotate_row(dst, head_dim, rope_theta, -static_cast<double>(t));
    }
}

namespace {

// Net renumber rotation for a survivor at absolute position `abs_pos` -> rank: rotate the
// stored post-RoPE row by (rank - abs_pos). Equivalent to un-RoPE then re-RoPE at rank.
inline double renumber_delta(int abs_pos, size_t rank) {
    return static_cast<double>(rank) - static_cast<double>(abs_pos);
}

// Per-head KeyDiff keep-set pipeline shared by the fp16 and int8 entry points. `fill_post(h, post)`
// gathers head h's post-RoPE rows into `post` ([n][head_dim]); the rest (un-RoPE -> score -> keepset)
// is identical regardless of storage precision.
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
    std::vector<float> krow(head_dim), vrow(head_dim);
    for (size_t h = 0; h < kv_heads; ++h) {
        const std::vector<int>& kept = kept_per_head[h];
        for (size_t rank = 0; rank < kept.size(); ++rank) {
            int abs_pos = kept[rank];
            const __fp16* ksrc = key_rows + (static_cast<size_t>(abs_pos) * kv_heads + h) * head_dim;
            const __fp16* vsrc = val_rows + (static_cast<size_t>(abs_pos) * kv_heads + h) * head_dim;
            for (size_t d = 0; d < head_dim; ++d) { krow[d] = ksrc[d]; vrow[d] = vsrc[d]; }
            // Renumber K: rotate post-RoPE row by (rank - abs_pos). V is not rotated.
            rope_rotate_row(krow.data(), head_dim, rope_theta, renumber_delta(abs_pos, rank));
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
                rope_rotate_row(row.data(), head_dim, rope_theta, renumber_delta(abs_pos, rank));
            int8_t* dst = int8_rows + rank * int8_stride + h * head_dim;
            float* dsc = scale_rows + rank * scale_stride + h * groups;
            for (size_t g = 0; g < groups; ++g) {
                size_t lo = g * group_size, hi = std::min(head_dim, lo + group_size);
                // Match the engine's KV quantization convention (quantize_group_fp16_to_int8 in
                // cactus-kernels/src/quants.cpp): scale floored at 1e-10, round-to-nearest via
                // roundf, clamp to [-128, 127]. Keeps re-quantized rows consistent with how the
                // rest of the engine writes the int8 cache.
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
    }
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

std::vector<size_t> physical_compressible_layers(const std::vector<std::string>& layer_types,
                                                 size_t num_layers, size_t num_kv_shared) {
    auto is_full = [&](size_t i) {
        if (i >= layer_types.size()) return true;
        return layer_types[i].find("sliding") == std::string::npos;
    };
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
