#include "test_utils.h"
#include "../src/kv_compress.h"
#include "../src/engine.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

using namespace TestUtils;
using namespace cactus::kvcompress;

namespace {

using Header = cactus::kvcompress::CacheHeader;
constexpr size_t kHeaderBytes = sizeof(Header);
constexpr size_t kGroupSize = 32;  // KV_QUANT_GROUP_SIZE

float f16_to_f32(uint16_t u) { __fp16 v; std::memcpy(&v, &u, 2); return static_cast<float>(v); }
uint16_t f32_to_f16(float f) { __fp16 v = static_cast<__fp16>(f); uint16_t u; std::memcpy(&u, &v, 2); return u; }

using EngineTestUtils::rope_reference;

// Build an FP16 cache buffer: 64B header + [max_seq][kv_heads][head_dim] fp16. Returns the
// raw buffer; `pre_rope[h][t][d]` is the planted pre-RoPE key, stored post-RoPE at position t.
std::vector<char> make_fp16_cache(size_t n, size_t max_seq, size_t kv_heads, size_t head_dim,
                                  double theta, std::vector<std::vector<std::vector<float>>>& pre_rope,
                                  std::vector<std::vector<std::vector<float>>>& values) {
    std::vector<char> buf(kHeaderBytes + max_seq * kv_heads * head_dim * sizeof(uint16_t), 0);
    auto* h = reinterpret_cast<Header*>(buf.data());
    h->current_seq_len = n; h->max_seq_len = max_seq; h->num_kv_heads = kv_heads;
    h->head_dim = head_dim; h->sink_size = 4;
    auto* rows = reinterpret_cast<uint16_t*>(buf.data() + kHeaderBytes);
    pre_rope.assign(kv_heads, std::vector<std::vector<float>>(n));
    values.assign(kv_heads, std::vector<std::vector<float>>(n));
    for (size_t hh = 0; hh < kv_heads; ++hh)
        for (size_t t = 0; t < n; ++t) {
            std::vector<float> k(head_dim), v(head_dim);
            for (size_t d = 0; d < head_dim; ++d) {
                k[d] = std::sin(0.1f * (hh + 1) * (t + 1) + 0.3f * d) + 0.05f * d;
                v[d] = std::cos(0.07f * (hh + 1) * (t + 2) + 0.2f * d);
            }
            pre_rope[hh][t] = k; values[hh][t] = v;
            std::vector<float> kr = rope_reference(k, (double)t, theta);
            for (size_t d = 0; d < head_dim; ++d)
                rows[(t * kv_heads + hh) * head_dim + d] = f32_to_f16(kr[d]);
        }
    return buf;
}

std::vector<char> make_fp16_value_cache(size_t n, size_t max_seq, size_t kv_heads, size_t head_dim,
                                        const std::vector<std::vector<std::vector<float>>>& values) {
    std::vector<char> buf(kHeaderBytes + max_seq * kv_heads * head_dim * sizeof(uint16_t), 0);
    auto* h = reinterpret_cast<Header*>(buf.data());
    h->current_seq_len = n; h->max_seq_len = max_seq; h->num_kv_heads = kv_heads;
    h->head_dim = head_dim; h->sink_size = 4;
    auto* rows = reinterpret_cast<uint16_t*>(buf.data() + kHeaderBytes);
    for (size_t hh = 0; hh < kv_heads; ++hh)
        for (size_t t = 0; t < n; ++t)
            for (size_t d = 0; d < head_dim; ++d)
                rows[(t * kv_heads + hh) * head_dim + d] = f32_to_f16(values[hh][t][d]);
    return buf;
}

}  // namespace

bool test_compact_fp16_cache() {
    const size_t n = 60, max_seq = 128, kv_heads = 3, head_dim = 16;
    const double theta = 1000000.0;
    std::vector<std::vector<std::vector<float>>> pre, vals;
    auto kbuf = make_fp16_cache(n, max_seq, kv_heads, head_dim, theta, pre, vals);
    auto vbuf = make_fp16_value_cache(n, max_seq, kv_heads, head_dim, vals);
    auto* krows = reinterpret_cast<uint16_t*>(kbuf.data() + kHeaderBytes);
    auto* vrows = reinterpret_cast<uint16_t*>(vbuf.data() + kHeaderBytes);

    // Heterogeneous per-head keep-sets (different indices, same length B).
    std::vector<std::vector<int>> kept = {
        {0, 5, 10, 20, 30, 59}, {1, 4, 12, 25, 40, 58}, {0, 2, 15, 33, 50, 55}};
    size_t B = kept[0].size();
    compact_fp16(krows, vrows, kv_heads, head_dim, kept, theta);

    bool ok = true;
    for (size_t h = 0; h < kv_heads; ++h) {
        for (size_t rank = 0; rank < B; ++rank) {
            int abs_pos = kept[h][rank];
            // Expected K at slot `rank` = rope(pre_rope[h][abs_pos], rank).
            std::vector<float> expK = rope_reference(pre[h][abs_pos], (double)rank, theta);
            const uint16_t* kdst = krows + (rank * kv_heads + h) * head_dim;
            const uint16_t* vdst = vrows + (rank * kv_heads + h) * head_dim;
            for (size_t d = 0; d < head_dim; ++d) {
                if (std::abs(f16_to_f32(kdst[d]) - expK[d]) > 5e-2f) ok = false;
                if (std::abs(f16_to_f32(vdst[d]) - vals[h][abs_pos][d]) > 5e-2f) ok = false;
            }
        }
    }
    return ok && B == 6;
}

bool test_dense_check_full_budget() {
    // abs_budget == n -> keep all indices in order; renumber maps rank==abs_pos so RoPE delta
    // is 0 -> the K rows are byte-identical and V unchanged.
    const size_t n = 40, max_seq = 64, kv_heads = 2, head_dim = 16;
    const double theta = 1000000.0;
    std::vector<std::vector<std::vector<float>>> pre, vals;
    auto kbuf = make_fp16_cache(n, max_seq, kv_heads, head_dim, theta, pre, vals);
    auto vbuf = make_fp16_value_cache(n, max_seq, kv_heads, head_dim, vals);
    std::vector<char> kbuf0 = kbuf, vbuf0 = vbuf;
    auto* krows = reinterpret_cast<uint16_t*>(kbuf.data() + kHeaderBytes);
    auto* vrows = reinterpret_cast<uint16_t*>(vbuf.data() + kHeaderBytes);

    Params p; p.recent_frac = 0.3f; p.sink = 4; p.abs_budget = (int)n;
    auto kept = keepsets_from_fp16(krows, n, kv_heads, head_dim, theta, p);
    for (auto& k : kept) if (k.size() != n) return false;  // B == n
    compact_fp16(krows, vrows, kv_heads, head_dim, kept, theta);

    // Byte-identical over the live region.
    size_t live = n * kv_heads * head_dim * sizeof(uint16_t);
    if (std::memcmp(kbuf.data() + kHeaderBytes, kbuf0.data() + kHeaderBytes, live) != 0) return false;
    if (std::memcmp(vbuf.data() + kHeaderBytes, vbuf0.data() + kHeaderBytes, live) != 0) return false;
    return true;
}

bool test_rope_renumber_contiguous() {
    // Route-B delta rotation: rope(stored_post_rope_at_abs, rank - abs) == rope(pre, rank).
    const size_t head_dim = 16; const double theta = 1000000.0;
    std::vector<float> pre(head_dim);
    for (size_t d = 0; d < head_dim; ++d) pre[d] = std::sin(0.4 * d) + 0.1 * d;
    int abs_pos = 137; size_t rank = 5;
    std::vector<float> at_abs = rope_reference(pre, (double)abs_pos, theta);
    std::vector<float> at_rank = rope_reference(pre, (double)rank, theta);
    std::vector<float> delta = at_abs;
    rope_rotate_row(delta.data(), head_dim, theta, (double)rank - (double)abs_pos);
    for (size_t d = 0; d < head_dim; ++d)
        if (std::abs(delta[d] - at_rank[d]) > 1e-3f) return false;
    return true;
}

bool test_fp16_storage_round_trip() {
    // Write pre-RoPE -> post-RoPE into an fp16 cache, compact+renumber, read back, and compare
    // each survivor against the reference rope(pre, rank) within fp16 tolerance. Exercises the
    // full fp16 storage path (f32->f16 store, gather, delta-rotate, f16->f16 write-back).
    const size_t n = 64, max_seq = 128, kv_heads = 2, head_dim = 16;
    const double theta = 1000000.0;
    std::vector<std::vector<std::vector<float>>> pre, vals;
    auto kbuf = make_fp16_cache(n, max_seq, kv_heads, head_dim, theta, pre, vals);
    auto vbuf = make_fp16_value_cache(n, max_seq, kv_heads, head_dim, vals);
    auto* krows = reinterpret_cast<uint16_t*>(kbuf.data() + kHeaderBytes);
    auto* vrows = reinterpret_cast<uint16_t*>(vbuf.data() + kHeaderBytes);

    Params p; p.recent_frac = 0.3f; p.sink = 4; p.abs_budget = 16;
    auto kept = keepsets_from_fp16(krows, n, kv_heads, head_dim, theta, p);
    compact_fp16(krows, vrows, kv_heads, head_dim, kept, theta);

    bool ok = true;
    for (size_t h = 0; h < kv_heads; ++h) {
        for (size_t rank = 0; rank < kept[h].size(); ++rank) {
            int abs_pos = kept[h][rank];
            std::vector<float> expK = rope_reference(pre[h][abs_pos], (double)rank, theta);
            const uint16_t* kdst = krows + (rank * kv_heads + h) * head_dim;
            const uint16_t* vdst = vrows + (rank * kv_heads + h) * head_dim;
            for (size_t d = 0; d < head_dim; ++d) {
                if (std::abs(f16_to_f32(kdst[d]) - expK[d]) > 5e-2f) ok = false;
                if (std::abs(f16_to_f32(vdst[d]) - vals[h][abs_pos][d]) > 5e-2f) ok = false;
            }
        }
    }
    return ok;
}

bool test_gemma_layer_selection() {
    // Qwen: all 28 layers (full_attention), no KV sharing.
    std::vector<std::string> qwen(28, "full_attention");
    auto q = physical_compressible_layers(qwen, 28, 0);
    if (q.size() != 28) return false;
    for (size_t i = 0; i < 28; ++i) if (q[i] != i) return false;

    // Gemma4-e2b: layer_types sliding/global, num_layers=35, num_kv_shared=20 -> {4,9}.
    std::vector<std::string> g;
    for (int i = 0; i < 35; ++i) g.push_back(((i + 1) % 5 == 0) ? "global" : "sliding");
    auto gg = physical_compressible_layers(g, 35, 20);
    std::vector<size_t> expect = {4, 9};
    return gg == expect;
}

bool test_compact_int8_cache() {
    // INT8: gather + renumber K, gather V (no rotation); dequant of compacted survivor ~ the
    // expected renumbered/gathered value.
    const size_t n = 50, max_seq = 96, kv_heads = 2, head_dim = 32;
    const double theta = 1000000.0;
    size_t groups = (head_dim + kGroupSize - 1) / kGroupSize;
    size_t int8_stride = kv_heads * head_dim, scale_stride = kv_heads * groups;

    // Build pre-RoPE keys + values, then store post-RoPE K quantized per group.
    std::vector<std::vector<std::vector<float>>> pre(kv_heads, std::vector<std::vector<float>>(n));
    std::vector<std::vector<std::vector<float>>> vals(kv_heads, std::vector<std::vector<float>>(n));
    std::vector<int8_t> k_i8(max_seq * int8_stride, 0), v_i8(max_seq * int8_stride, 0);
    std::vector<float> k_sc(max_seq * scale_stride, 0.f), v_sc(max_seq * scale_stride, 0.f);

    // Match the engine KV quant convention (quants.cpp): scale floored at 1e-10, roundf,
    // clamp [-128,127] -- same as compact_int8's re-quant path.
    auto quant_row = [&](const std::vector<float>& row, int8_t* dst, float* sc) {
        for (size_t g = 0; g < groups; ++g) {
            size_t lo = g * kGroupSize, hi = std::min(head_dim, lo + kGroupSize);
            float amax = 0.f; for (size_t d = lo; d < hi; ++d) amax = std::max(amax, std::fabs(row[d]));
            float scale = amax / 127.f; if (scale < 1e-10f) scale = 1e-10f; sc[g] = scale;
            float inv = 1.f / scale;
            for (size_t d = lo; d < hi; ++d) { int32_t q = (int32_t)std::roundf(row[d] * inv); q = std::max(-128, std::min(127, q)); dst[d] = (int8_t)q; }
        }
    };
    for (size_t h = 0; h < kv_heads; ++h)
        for (size_t t = 0; t < n; ++t) {
            std::vector<float> k(head_dim), v(head_dim);
            for (size_t d = 0; d < head_dim; ++d) { k[d] = std::sin(0.1 * (h + 1) * (t + 1) + 0.2 * d); v[d] = std::cos(0.05 * (t + 1) + 0.1 * d); }
            pre[h][t] = k; vals[h][t] = v;
            std::vector<float> kr = rope_reference(k, (double)t, theta);
            quant_row(kr, k_i8.data() + t * int8_stride + h * head_dim, k_sc.data() + t * scale_stride + h * groups);
            quant_row(v, v_i8.data() + t * int8_stride + h * head_dim, v_sc.data() + t * scale_stride + h * groups);
        }

    std::vector<std::vector<int>> kept = {{0, 5, 11, 22, 33, 49}, {1, 4, 13, 26, 40, 48}};
    size_t B = kept[0].size();
    compact_int8(k_i8.data(), k_sc.data(), kv_heads, head_dim, kGroupSize, kept, theta, true);
    compact_int8(v_i8.data(), v_sc.data(), kv_heads, head_dim, kGroupSize, kept, theta, false);

    bool ok = true;
    for (size_t h = 0; h < kv_heads; ++h)
        for (size_t rank = 0; rank < B; ++rank) {
            int abs_pos = kept[h][rank];
            std::vector<float> expK = rope_reference(pre[h][abs_pos], (double)rank, theta);
            const int8_t* kd = k_i8.data() + rank * int8_stride + h * head_dim;
            const float* ks = k_sc.data() + rank * scale_stride + h * groups;
            const int8_t* vd = v_i8.data() + rank * int8_stride + h * head_dim;
            const float* vs = v_sc.data() + rank * scale_stride + h * groups;
            for (size_t d = 0; d < head_dim; ++d) {
                float dq = (float)kd[d] * ks[d / kGroupSize];
                if (std::abs(dq - expK[d]) > 0.05f) ok = false;
                float dqv = (float)vd[d] * vs[d / kGroupSize];
                if (std::abs(dqv - vals[h][abs_pos][d]) > 0.05f) ok = false;
            }
        }
    return ok;
}

// Mirror Model::maybe_roll_compact: when the live seq-len reaches trigger_len, KeyDiff-keep the
// best target_len tokens (absolute budget), renumber survivors to 0..B-1, set header len = B.
// Returns B (the new bounded length). Operates on an FP16 K/V cache pair in place.
size_t roll_compact_once(uint16_t* krows, uint16_t* vrows, Header* khdr, Header* vhdr,
                         size_t kv_heads, size_t head_dim, double theta, int target_len) {
    size_t n = khdr->current_seq_len;
    Params p; p.recent_frac = 0.30f; p.sink = 4;
    p.abs_budget = target_len;  // absolute budget -> keep exactly min(target_len, n) per head
    auto kept = keepsets_from_fp16(krows, n, kv_heads, head_dim, theta, p);
    compact_fp16(krows, vrows, kv_heads, head_dim, kept, theta);
    size_t B = kept.empty() ? 0 : kept[0].size();
    khdr->current_seq_len = B;
    vhdr->current_seq_len = B;
    return B;
}

bool test_rolling_bounded_compaction() {
    // Drive the compactor across repeated thresholds on a synthetic cache, exactly as the rolling
    // bounded path does: grow to trigger_len -> compact to target_len -> grow again -> compact.
    // Asserts the cache stays bounded <= trigger_len, positions renumber to 0..B-1 each cycle, and
    // a planted distinctive mid-token survives every compaction.
    const int trigger_len = 4096, target_len = 2048;
    const size_t kv_heads = 2, head_dim = 16, max_seq = trigger_len + 8;
    const double theta = 1000000.0;

    std::vector<char> kbuf(kHeaderBytes + max_seq * kv_heads * head_dim * sizeof(uint16_t), 0);
    std::vector<char> vbuf(kbuf.size(), 0);
    auto* khdr = reinterpret_cast<Header*>(kbuf.data());
    auto* vhdr = reinterpret_cast<Header*>(vbuf.data());
    *khdr = Header{0, max_seq, kv_heads, head_dim, 4, {0, 0, 0}};
    *vhdr = *khdr;
    auto* krows = reinterpret_cast<uint16_t*>(kbuf.data() + kHeaderBytes);
    auto* vrows = reinterpret_cast<uint16_t*>(vbuf.data() + kHeaderBytes);

    // A strongly distinctive pre-RoPE key direction (centroid-opposite) that KeyDiff must keep.
    std::vector<float> distinctive(head_dim);
    for (size_t d = 0; d < head_dim; ++d) distinctive[d] = (d % 2 == 0 ? -1.0f : 1.0f) * 4.0f;

    // Append fresh post-RoPE rows for positions [from, to) for every head. The token at
    // `plant_abs` (if in range) is the distinctive one; the rest cluster on a common centroid.
    auto append = [&](size_t from, size_t to, long plant_abs) {
        for (size_t t = from; t < to; ++t)
            for (size_t h = 0; h < kv_heads; ++h) {
                std::vector<float> k(head_dim);
                if ((long)t == plant_abs) {
                    k = distinctive;
                } else {
                    for (size_t d = 0; d < head_dim; ++d)
                        k[d] = std::sin(0.05 * (h + 1) + 0.3 * d) + 0.01f * std::sin(0.01 * t);
                }
                std::vector<float> kr = rope_reference(k, (double)t, theta);
                std::vector<float> vr(head_dim);
                for (size_t d = 0; d < head_dim; ++d) vr[d] = std::cos(0.07 * (t + 1) + 0.2 * d);
                for (size_t d = 0; d < head_dim; ++d) {
                    krows[(t * kv_heads + h) * head_dim + d] = f32_to_f16(kr[d]);
                    vrows[(t * kv_heads + h) * head_dim + d] = f32_to_f16(vr[d]);
                }
            }
    };

    // Does any survivor row (un-RoPE'd at its rank) match the planted distinctive direction?
    auto distinctive_survives = [&](size_t B) -> bool {
        for (size_t h = 0; h < kv_heads; ++h)
            for (size_t rank = 0; rank < B; ++rank) {
                std::vector<float> pre(head_dim);
                const uint16_t* src = krows + (rank * kv_heads + h) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) pre[d] = f16_to_f32(src[d]);
                // The row is stored as rope(pre, rank); un-RoPE by -rank to recover pre.
                rope_rotate_row(pre.data(), head_dim, theta, -(double)rank);
                double dot = 0, na = 0, nb = 0;
                for (size_t d = 0; d < head_dim; ++d) {
                    dot += pre[d] * distinctive[d]; na += pre[d] * pre[d]; nb += distinctive[d] * distinctive[d];
                }
                if (dot / (std::sqrt(na) * std::sqrt(nb) + 1e-9) > 0.99) return true;
            }
        return false;
    };

    bool ok = true;
    const int cycles = 3;
    for (int c = 0; c < cycles; ++c) {
        size_t start = khdr->current_seq_len;            // survivors carried over from last cycle
        long plant_abs = (c == 0) ? (trigger_len / 2) : -1;  // plant once; must persist after that
        append(start, (size_t)trigger_len, plant_abs);
        khdr->current_seq_len = trigger_len;
        vhdr->current_seq_len = trigger_len;

        // The distinctive token must be present BEFORE the first compaction renumbers it.
        if (c == 0 && !distinctive_survives(trigger_len)) ok = false;

        size_t B = roll_compact_once(krows, vrows, khdr, vhdr, kv_heads, head_dim, theta, target_len);
        if (B != (size_t)target_len) ok = false;                 // absolute budget honored
        if (khdr->current_seq_len > (size_t)trigger_len) ok = false;  // bounded
        if (khdr->current_seq_len != (size_t)target_len) ok = false;  // header set to B

        // Renumbering: survivor K rows must read back as rope(pre, rank). We verify via the
        // un-RoPE-at-rank == direction check for the planted token, and that rows are dense
        // (positions 0..B-1 populated, no gaps) by re-reading the live region length.
        if (!distinctive_survives(B)) ok = false;                // distinctive persists each cycle
    }
    // After all cycles the cache is still bounded at target_len, never exceeding trigger_len.
    if (khdr->current_seq_len != (size_t)target_len) ok = false;
    return ok;
}

bool test_config_parse_rolling_fields() {
    // Config::from_json parses a key=value file. Rolling is the default (4096 -> 2048).
    cactus::engine::Config def;
    if (!def.kv_compress) return false;
    if (def.kv_compress_trigger_len != 4096 || def.kv_compress_target_len != 2048) return false;

    char tmpl[] = "/tmp/cactus_kvcfg_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) return false;
    ::close(fd);
    {
        std::ofstream f(tmpl);
        // model_type=qwen avoids the Gemma4-only required-field validation; we only assert the
        // new kv_compress rolling fields round-trip through the key=value parser.
        f << "model_type=qwen\n"
          << "kv_compress=true\n"
          << "kv_compress_trigger_len=4096\n"
          << "kv_compress_target_len=2048\n";
    }
    cactus::engine::Config cfg;
    bool parsed = cfg.from_json(tmpl);
    std::remove(tmpl);
    if (!parsed) return false;
    return cfg.kv_compress && cfg.kv_compress_trigger_len == 4096 && cfg.kv_compress_target_len == 2048;
}

bool test_trigger_zero_gates_rolling() {
    // The rolling gate (Model::maybe_roll_compact): fire iff kv_compress && trigger_len > 0 &&
    // current_seq_len >= trigger_len. Drive the SAME compactor under both arms on identical caches:
    // trigger_len == 0 must skip (cache byte-identical); a reached trigger must compact (len -> target,
    // bytes change). The skip assertion is meaningful only against this proven mutation.
    const size_t n = 5000, kv_heads = 2, head_dim = 16, max_seq = 5008;
    const double theta = 1000000.0;

    auto fresh_cache = [&]() {
        std::vector<char> buf(kHeaderBytes + max_seq * kv_heads * head_dim * sizeof(uint16_t), 0);
        *reinterpret_cast<Header*>(buf.data()) = Header{n, max_seq, kv_heads, head_dim, 4, {0, 0, 0}};
        auto* rows = reinterpret_cast<uint16_t*>(buf.data() + kHeaderBytes);
        for (size_t i = 0; i < n * kv_heads * head_dim; ++i) rows[i] = f32_to_f16(0.01f * (i % 97));
        return buf;
    };
    auto roll_if_gated = [&](cactus::engine::Config& cfg, std::vector<char>& kbuf, std::vector<char>& vbuf) {
        auto* khdr = reinterpret_cast<Header*>(kbuf.data());
        bool fire = cfg.kv_compress && cfg.kv_compress_trigger_len > 0 &&
                    khdr->current_seq_len >= (size_t)cfg.kv_compress_trigger_len;
        if (fire)
            roll_compact_once(reinterpret_cast<uint16_t*>(kbuf.data() + kHeaderBytes),
                              reinterpret_cast<uint16_t*>(vbuf.data() + kHeaderBytes), khdr,
                              reinterpret_cast<Header*>(vbuf.data()), kv_heads, head_dim, theta,
                              cfg.kv_compress_target_len);
    };

    // Gate closed (trigger_len == 0): caller skips => cache byte-identical, length unchanged.
    cactus::engine::Config off; off.kv_compress_trigger_len = 0; off.kv_compress_target_len = 0;
    std::vector<char> k_off = fresh_cache(), v_off(k_off.size(), 0), k_off0 = k_off, v_off0 = v_off;
    roll_if_gated(off, k_off, v_off);
    if (reinterpret_cast<Header*>(k_off.data())->current_seq_len != n) return false;
    if (std::memcmp(k_off.data(), k_off0.data(), k_off.size()) != 0) return false;
    if (std::memcmp(v_off.data(), v_off0.data(), v_off.size()) != 0) return false;

    // Gate open (trigger reached): the same compactor compacts to target_len and rewrites bytes.
    cactus::engine::Config on; on.kv_compress_trigger_len = 4096; on.kv_compress_target_len = 2048;
    std::vector<char> k_on = fresh_cache(), v_on(k_on.size(), 0), k_on0 = k_on;
    roll_if_gated(on, k_on, v_on);
    if (reinterpret_cast<Header*>(k_on.data())->current_seq_len != 2048) return false;
    if (std::memcmp(k_on.data(), k_on0.data(), k_on.size()) == 0) return false;
    return true;
}

bool test_degenerate_rolling_config_disabled() {
    // validate_kv_compress(): when trigger_len > 0, require 0 < target_len < trigger_len.
    // Bad configs must disable rolling (trigger_len/target_len reset to 0); good configs survive.
    auto disabled = [](int trig, int targ) {
        cactus::engine::Config c;
        c.kv_compress = true;
        c.kv_compress_trigger_len = trig;
        c.kv_compress_target_len = targ;
        c.validate_kv_compress();
        return c.kv_compress_trigger_len == 0 && c.kv_compress_target_len == 0;
    };
    if (!disabled(4096, 0)) return false;        // target_len <= 0
    if (!disabled(4096, 4096)) return false;     // target_len == trigger_len
    if (!disabled(2048, 4096)) return false;     // target_len > trigger_len
    if (!disabled(4096, -1)) return false;       // negative target_len
    // Valid config is untouched.
    cactus::engine::Config ok;
    ok.kv_compress = true;
    ok.kv_compress_trigger_len = 4096;
    ok.kv_compress_target_len = 2048;
    ok.validate_kv_compress();
    return ok.kv_compress_trigger_len == 4096 && ok.kv_compress_target_len == 2048;
}

bool test_env_override_parse() {
    // parse_kv_compress_override maps CACTUS_KV_COMPRESS_AT / CACTUS_KV_COMPRESS_TO onto Config.
    // Unset => no override; the rolling default (4096 -> 2048) is preserved.
    cactus::engine::Config off;
    if (off.parse_kv_compress_override(nullptr, nullptr)) return false;
    if (!off.kv_compress || off.kv_compress_trigger_len != 4096 || off.kv_compress_target_len != 2048) return false;

    // Both vars override.
    cactus::engine::Config both;
    if (!both.parse_kv_compress_override("3000", "1000")) return false;
    if (!both.kv_compress || both.kv_compress_trigger_len != 3000 || both.kv_compress_target_len != 1000) return false;

    // Setting only one keeps the other's default.
    cactus::engine::Config one;
    if (!one.parse_kv_compress_override(nullptr, "1500")) return false;
    if (one.kv_compress_trigger_len != 4096 || one.kv_compress_target_len != 1500) return false;

    // CACTUS_KV_COMPRESS_AT=0 disables.
    cactus::engine::Config disabled;
    if (!disabled.parse_kv_compress_override("0", nullptr)) return false;
    if (disabled.kv_compress) return false;

    // Degenerate (target >= trigger) disabled by the validate guard.
    cactus::engine::Config bad;
    if (!bad.parse_kv_compress_override("2048", "4096")) return false;
    if (bad.kv_compress_trigger_len != 0 || bad.kv_compress_target_len != 0) return false;
    return true;
}

int main() {
    TestUtils::TestRunner runner("KV Compress Free-Function Tests");
    runner.run_test("compact_fp16_cache", test_compact_fp16_cache());
    runner.run_test("dense_check_full_budget", test_dense_check_full_budget());
    runner.run_test("rope_renumber_contiguous", test_rope_renumber_contiguous());
    runner.run_test("fp16_storage_round_trip", test_fp16_storage_round_trip());
    runner.run_test("gemma_layer_selection", test_gemma_layer_selection());
    runner.run_test("compact_int8_cache", test_compact_int8_cache());
    runner.run_test("rolling_bounded_compaction", test_rolling_bounded_compaction());
    runner.run_test("config_parse_rolling_fields", test_config_parse_rolling_fields());
    runner.run_test("trigger_zero_gates_rolling", test_trigger_zero_gates_rolling());
    runner.run_test("degenerate_rolling_config_disabled", test_degenerate_rolling_config_disabled());
    runner.run_test("env_override_parse", test_env_override_parse());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
