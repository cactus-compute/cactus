#include "test_utils.h"
#include "../src/kv_compress.h"
#include "../src/engine.h"
#include "cactus_graph.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
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

// `pre_rope[h][t][d]` is the planted pre-RoPE key, stored post-RoPE at position t.
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

// Independent oracle quantizer, kept separate from production requant so tests don't self-validate.
void quantize_row(const std::vector<float>& row, int8_t* dst, float* sc,
                  size_t head_dim, size_t group_size) {
    size_t groups = (head_dim + group_size - 1) / group_size;
    for (size_t g = 0; g < groups; ++g) {
        size_t lo = g * group_size, hi = std::min(head_dim, lo + group_size);
        float amax = 0.f; for (size_t d = lo; d < hi; ++d) amax = std::max(amax, std::fabs(row[d]));
        float scale = amax / 127.f; if (scale < 1e-10f) scale = 1e-10f; sc[g] = scale;
        float inv = 1.f / scale;
        for (size_t d = lo; d < hi; ++d) { int32_t q = (int32_t)std::roundf(row[d] * inv); q = std::max(-128, std::min(127, q)); dst[d] = (int8_t)q; }
    }
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

    std::vector<std::vector<int>> kept = {
        {0, 5, 10, 20, 30, 59}, {1, 4, 12, 25, 40, 58}, {0, 2, 15, 33, 50, 55}};
    size_t B = kept[0].size();
    compact_fp16(krows, vrows, kv_heads, head_dim, kept, theta);

    bool ok = true;
    for (size_t h = 0; h < kv_heads; ++h) {
        for (size_t rank = 0; rank < B; ++rank) {
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
    return ok && B == 6;
}

bool test_dense_check_full_budget() {
    // abs_budget == n keeps every index in order, so renumber delta is 0 and the buffer is unchanged.
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
    for (auto& k : kept) if (k.size() != n) return false;
    compact_fp16(krows, vrows, kv_heads, head_dim, kept, theta);

    size_t live = n * kv_heads * head_dim * sizeof(uint16_t);
    if (std::memcmp(kbuf.data() + kHeaderBytes, kbuf0.data() + kHeaderBytes, live) != 0) return false;
    if (std::memcmp(vbuf.data() + kHeaderBytes, vbuf0.data() + kHeaderBytes, live) != 0) return false;
    return true;
}

bool test_rope_renumber_contiguous() {
    // rope(stored_post_rope_at_abs, rank - abs) == rope(pre, rank).
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
    // Each survivor must read back as rope(pre, rank) within fp16 tolerance.
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
    std::vector<std::string> qwen(28, "full_attention");
    auto q = physical_compressible_layers(qwen, 28, 0);
    if (q.size() != 28) return false;
    for (size_t i = 0; i < 28; ++i) if (q[i] != i) return false;

    std::vector<std::string> g;
    for (int i = 0; i < 35; ++i) g.push_back(((i + 1) % 5 == 0) ? "global" : "sliding");
    auto gg = physical_compressible_layers(g, 35, 20);
    std::vector<size_t> expect = {4, 9};
    if (gg != expect) return false;

    // Pass-2 theta follows is_sliding_layer, NOT compressibility: layer 14 is a global KV-shared
    // *source* excluded from compaction (not in {4,9}) yet must still re-rope with the global theta.
    if (is_sliding_layer(g, 14)) return false;
    if (is_sliding_layer(g, 4)) return false;
    if (!is_sliding_layer(g, 3)) return false;
    for (size_t i = 0; i < 28; ++i) if (is_sliding_layer(qwen, i)) return false;
    return true;
}

bool test_compact_int8_cache() {
    const size_t n = 50, max_seq = 96, kv_heads = 2, head_dim = 32;
    const double theta = 1000000.0;
    size_t groups = (head_dim + kGroupSize - 1) / kGroupSize;
    size_t int8_stride = kv_heads * head_dim, scale_stride = kv_heads * groups;

    std::vector<std::vector<std::vector<float>>> pre(kv_heads, std::vector<std::vector<float>>(n));
    std::vector<std::vector<std::vector<float>>> vals(kv_heads, std::vector<std::vector<float>>(n));
    std::vector<int8_t> k_i8(max_seq * int8_stride, 0), v_i8(max_seq * int8_stride, 0);
    std::vector<float> k_sc(max_seq * scale_stride, 0.f), v_sc(max_seq * scale_stride, 0.f);

    auto quant_row = [&](const std::vector<float>& row, int8_t* dst, float* sc) {
        quantize_row(row, dst, sc, head_dim, kGroupSize);
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

// Mirrors Model::maybe_roll_compact on an in-place FP16 K/V pair; returns the new bounded length B.
size_t roll_compact_once(uint16_t* krows, uint16_t* vrows, Header* khdr, Header* vhdr,
                         size_t kv_heads, size_t head_dim, double theta, int target_len) {
    size_t n = khdr->current_seq_len;
    Params p; p.recent_frac = 0.30f; p.sink = 4;
    p.abs_budget = target_len;
    auto kept = keepsets_from_fp16(krows, n, kv_heads, head_dim, theta, p);
    compact_fp16(krows, vrows, kv_heads, head_dim, kept, theta);
    size_t B = kept.empty() ? 0 : kept[0].size();
    khdr->current_seq_len = B;
    vhdr->current_seq_len = B;
    return B;
}

bool test_rolling_bounded_compaction() {
    // Grow to trigger_len -> compact to target_len -> repeat, asserting the cache stays bounded and
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

    // A centroid-opposite direction that KeyDiff must keep.
    std::vector<float> distinctive(head_dim);
    for (size_t d = 0; d < head_dim; ++d) distinctive[d] = (d % 2 == 0 ? -1.0f : 1.0f) * 4.0f;

    // The token at `plant_abs` is the distinctive one; the rest cluster on a common centroid.
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

    auto distinctive_survives = [&](size_t B) -> bool {
        for (size_t h = 0; h < kv_heads; ++h)
            for (size_t rank = 0; rank < B; ++rank) {
                std::vector<float> pre(head_dim);
                const uint16_t* src = krows + (rank * kv_heads + h) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) pre[d] = f16_to_f32(src[d]);
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
        size_t start = khdr->current_seq_len;
        long plant_abs = (c == 0) ? (trigger_len / 2) : -1;
        append(start, (size_t)trigger_len, plant_abs);
        khdr->current_seq_len = trigger_len;
        vhdr->current_seq_len = trigger_len;

        if (c == 0 && !distinctive_survives(trigger_len)) ok = false;

        size_t B = roll_compact_once(krows, vrows, khdr, vhdr, kv_heads, head_dim, theta, target_len);
        if (B != (size_t)target_len) ok = false;
        if (khdr->current_seq_len > (size_t)trigger_len) ok = false;
        if (khdr->current_seq_len != (size_t)target_len) ok = false;
        if (!distinctive_survives(B)) ok = false;
    }
    if (khdr->current_seq_len != (size_t)target_len) ok = false;
    return ok;
}

bool test_config_parse_rolling_fields() {
    cactus::engine::Config def;
    if (!def.kv_compress) return false;
    if (def.kv_compress_trigger_len != 4096 || def.kv_compress_target_len != 2048) return false;

    std::string tmpl_str = (std::filesystem::temp_directory_path() / "cactus_kvcfg_XXXXXX").string();
    std::vector<char> tmpl(tmpl_str.begin(), tmpl_str.end());
    tmpl.push_back('\0');
    int fd = mkstemp(tmpl.data());
    if (fd < 0) return false;
    ::close(fd);
    {
        std::ofstream f(tmpl.data());
        // model_type=qwen avoids the Gemma4-only required-field validation.
        f << "model_type=qwen\n"
          << "kv_compress=true\n"
          << "kv_compress_trigger_len=4096\n"
          << "kv_compress_target_len=2048\n";
    }
    cactus::engine::Config cfg;
    bool parsed = cfg.from_json(tmpl.data());
    std::remove(tmpl.data());
    if (!parsed) return false;
    return cfg.kv_compress && cfg.kv_compress_trigger_len == 4096 && cfg.kv_compress_target_len == 2048;
}

bool test_trigger_zero_gates_rolling() {
    // The maybe_roll_compact gate fires iff kv_compress && trigger_len > 0 && seq_len >= trigger_len.
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

    // Gate closed: cache byte-identical, length unchanged.
    cactus::engine::Config off; off.kv_compress_trigger_len = 0; off.kv_compress_target_len = 0;
    std::vector<char> k_off = fresh_cache(), v_off(k_off.size(), 0), k_off0 = k_off, v_off0 = v_off;
    roll_if_gated(off, k_off, v_off);
    if (reinterpret_cast<Header*>(k_off.data())->current_seq_len != n) return false;
    if (std::memcmp(k_off.data(), k_off0.data(), k_off.size()) != 0) return false;
    if (std::memcmp(v_off.data(), v_off0.data(), v_off.size()) != 0) return false;

    // Gate open: compacts to target_len and rewrites bytes.
    cactus::engine::Config on; on.kv_compress_trigger_len = 4096; on.kv_compress_target_len = 2048;
    std::vector<char> k_on = fresh_cache(), v_on(k_on.size(), 0), k_on0 = k_on;
    roll_if_gated(on, k_on, v_on);
    if (reinterpret_cast<Header*>(k_on.data())->current_seq_len != 2048) return false;
    if (std::memcmp(k_on.data(), k_on0.data(), k_on.size()) == 0) return false;
    return true;
}

bool test_degenerate_rolling_config_disabled() {
    // validate_kv_compress requires 0 < target_len < trigger_len; bad configs reset both to 0.
    auto disabled = [](int trig, int targ) {
        cactus::engine::Config c;
        c.kv_compress = true;
        c.kv_compress_trigger_len = trig;
        c.kv_compress_target_len = targ;
        c.validate_kv_compress();
        return c.kv_compress_trigger_len == 0 && c.kv_compress_target_len == 0;
    };
    if (!disabled(4096, 0)) return false;
    if (!disabled(4096, 4096)) return false;
    if (!disabled(2048, 4096)) return false;
    if (!disabled(4096, -1)) return false;
    cactus::engine::Config ok;
    ok.kv_compress = true;
    ok.kv_compress_trigger_len = 4096;
    ok.kv_compress_target_len = 2048;
    ok.validate_kv_compress();
    return ok.kv_compress_trigger_len == 4096 && ok.kv_compress_target_len == 2048;
}

bool test_env_override_parse() {
    cactus::engine::Config off;
    if (off.parse_kv_compress_override(nullptr, nullptr)) return false;
    if (!off.kv_compress || off.kv_compress_trigger_len != 4096 || off.kv_compress_target_len != 2048) return false;

    cactus::engine::Config both;
    if (!both.parse_kv_compress_override("3000", "1000")) return false;
    if (!both.kv_compress || both.kv_compress_trigger_len != 3000 || both.kv_compress_target_len != 1000) return false;

    // Setting only one keeps the other's default.
    cactus::engine::Config one;
    if (!one.parse_kv_compress_override(nullptr, "1500")) return false;
    if (one.kv_compress_trigger_len != 4096 || one.kv_compress_target_len != 1500) return false;

    cactus::engine::Config disabled;
    if (!disabled.parse_kv_compress_override("0", nullptr)) return false;
    if (disabled.kv_compress) return false;

    cactus::engine::Config bad;
    if (!bad.parse_kv_compress_override("2048", "4096")) return false;
    if (bad.kv_compress_trigger_len != 0 || bad.kv_compress_target_len != 0) return false;
    return true;
}

bool test_rerope_recent_fp16_uniform() {
    // After a delta shift each recent row reads as rope(pre, t+delta); sink rows stay byte-identical.
    const size_t n = 40, max_seq = 64, kv_heads = 2, head_dim = 16, sink = 4;
    const double theta = 10000.0, delta = -7.0;
    std::vector<std::vector<std::vector<float>>> pre, vals;
    auto kbuf = make_fp16_cache(n, max_seq, kv_heads, head_dim, theta, pre, vals);
    auto* krows = reinterpret_cast<uint16_t*>(kbuf.data() + kHeaderBytes);
    std::vector<uint16_t> sink_before(krows, krows + sink * kv_heads * head_dim);

    rerope_recent_fp16(krows, kv_heads, head_dim, sink, n, theta, delta);

    if (std::memcmp(krows, sink_before.data(), sink_before.size() * sizeof(uint16_t)) != 0) return false;
    for (size_t t = sink; t < n; ++t)
        for (size_t h = 0; h < kv_heads; ++h) {
            std::vector<float> exp = rope_reference(pre[h][t], (double)t + delta, theta);
            const uint16_t* r = krows + (t * kv_heads + h) * head_dim;
            for (size_t d = 0; d < head_dim; ++d)
                if (std::abs(f16_to_f32(r[d]) - exp[d]) > 5e-2f) return false;
        }
    return true;
}

namespace {
void build_int8_key_cache(size_t n, size_t kv_heads, size_t head_dim, double theta,
                          std::vector<std::vector<std::vector<float>>>& pre,
                          std::vector<int8_t>& k_i8, std::vector<float>& k_sc) {
    size_t groups = (head_dim + kGroupSize - 1) / kGroupSize;
    size_t int8_stride = kv_heads * head_dim, scale_stride = kv_heads * groups;
    pre.assign(kv_heads, std::vector<std::vector<float>>(n));
    k_i8.assign(n * int8_stride, 0); k_sc.assign(n * scale_stride, 0.f);
    for (size_t h = 0; h < kv_heads; ++h)
        for (size_t t = 0; t < n; ++t) {
            std::vector<float> k(head_dim);
            for (size_t d = 0; d < head_dim; ++d) k[d] = std::sin(0.1 * (h + 1) * (t + 1) + 0.2 * d);
            pre[h][t] = k;
            std::vector<float> kr = rope_reference(k, (double)t, theta);
            quantize_row(kr, k_i8.data() + t * int8_stride + h * head_dim,
                         k_sc.data() + t * scale_stride + h * groups, head_dim, kGroupSize);
        }
}
}  // namespace

bool test_rerope_recent_int8() {
    const size_t n = 40, kv_heads = 2, head_dim = 32, sink = 4;
    const double theta = 10000.0, delta = -9.0;
    size_t groups = (head_dim + kGroupSize - 1) / kGroupSize;
    size_t int8_stride = kv_heads * head_dim, scale_stride = kv_heads * groups;
    std::vector<std::vector<std::vector<float>>> pre;
    std::vector<int8_t> k_i8; std::vector<float> k_sc;
    build_int8_key_cache(n, kv_heads, head_dim, theta, pre, k_i8, k_sc);

    std::vector<int8_t> sink_i8(k_i8.begin(), k_i8.begin() + sink * int8_stride);
    std::vector<float> sink_sc(k_sc.begin(), k_sc.begin() + sink * scale_stride);
    rerope_recent_int8(k_i8.data(), k_sc.data(), kv_heads, head_dim, kGroupSize, sink, n, theta, delta);

    if (std::memcmp(k_i8.data(), sink_i8.data(), sink_i8.size()) != 0) return false;
    if (std::memcmp(k_sc.data(), sink_sc.data(), sink_sc.size() * sizeof(float)) != 0) return false;
    for (size_t h = 0; h < kv_heads; ++h)
        for (size_t t = sink; t < n; ++t) {
            std::vector<float> exp = rope_reference(pre[h][t], (double)t + delta, theta);
            const int8_t* kd = k_i8.data() + t * int8_stride + h * head_dim;
            const float* ks = k_sc.data() + t * scale_stride + h * groups;
            for (size_t d = 0; d < head_dim; ++d)
                if (std::abs((float)kd[d] * ks[d / kGroupSize] - exp[d]) > 0.05f) return false;
        }
    return true;
}

bool test_rotate_int8_row_matches_inline() {
    // rotate_int8_row must be bitwise-identical to the inline dequant -> rotate -> requant sequence.
    const size_t head_dim = 32; const double theta = 10000.0, delta = -9.0;
    size_t groups = (head_dim + kGroupSize - 1) / kGroupSize;
    std::vector<float> pre(head_dim);
    for (size_t d = 0; d < head_dim; ++d) pre[d] = std::sin(0.3 * d) + 0.1 * d;
    std::vector<float> stored = rope_reference(pre, 21.0, theta);
    std::vector<int8_t> a_i8(head_dim), b_i8(head_dim);
    std::vector<float> a_sc(groups), b_sc(groups);
    quantize_row(stored, a_i8.data(), a_sc.data(), head_dim, kGroupSize);
    b_i8 = a_i8; b_sc = a_sc;

    std::vector<float> row(head_dim);
    for (size_t d = 0; d < head_dim; ++d) row[d] = (float)a_i8[d] * a_sc[d / kGroupSize];
    rope_rotate_row(row.data(), head_dim, theta, delta);
    quantize_row(row, a_i8.data(), a_sc.data(), head_dim, kGroupSize);
    rotate_int8_row(b_i8.data(), b_sc.data(), head_dim, kGroupSize, theta, delta);

    return std::memcmp(a_i8.data(), b_i8.data(), head_dim) == 0
        && std::memcmp(a_sc.data(), b_sc.data(), groups * sizeof(float)) == 0;
}

bool test_sliding_plus_global_rolling() {
    // A global cache renumbers survivors to 0..B-1 while a sliding cache re-ropes its recent rows by
    // -Δ (Δ = old_total - B), keeping both in the same compressed frame across 3 cycles.
    const size_t kv_heads = 2, head_dim = 16, sink = 4;
    const double gtheta = 1000000.0, ltheta = 10000.0;

    {
        const size_t n = 60, max_seq = 96;
        std::vector<std::vector<std::vector<float>>> pre, vals;
        auto kbuf = make_fp16_cache(n, max_seq, kv_heads, head_dim, gtheta, pre, vals);
        auto vbuf = make_fp16_value_cache(n, max_seq, kv_heads, head_dim, vals);
        auto* krows = reinterpret_cast<uint16_t*>(kbuf.data() + kHeaderBytes);
        auto* vrows = reinterpret_cast<uint16_t*>(vbuf.data() + kHeaderBytes);
        Params p; p.recent_frac = 0.3f; p.sink = sink; p.abs_budget = 16;
        auto kept = keepsets_from_fp16(krows, n, kv_heads, head_dim, gtheta, p);
        size_t B = kept[0].size();
        compact_fp16(krows, vrows, kv_heads, head_dim, kept, gtheta);
        for (auto& k : kept) if (k.size() != B) return false;
        for (size_t h = 0; h < kv_heads; ++h)
            for (size_t rank = 0; rank < B; ++rank) {
                std::vector<float> exp = rope_reference(pre[h][kept[h][rank]], (double)rank, gtheta);
                const uint16_t* r = krows + (rank * kv_heads + h) * head_dim;
                for (size_t d = 0; d < head_dim; ++d)
                    if (std::abs(f16_to_f32(r[d]) - exp[d]) > 5e-2f) return false;
            }
    }

    const size_t n = 16, max_seq = 32;
    std::vector<std::vector<std::vector<float>>> pre, vals;
    auto kbuf = make_fp16_cache(n, max_seq, kv_heads, head_dim, ltheta, pre, vals);
    auto* krows = reinterpret_cast<uint16_t*>(kbuf.data() + kHeaderBytes);
    std::vector<uint16_t> sink_before(krows, krows + sink * kv_heads * head_dim);

    const size_t old_total = 30, B = 12;
    const double Delta = (double)(old_total - B);
    double accum = 0.0;
    for (int cycle = 0; cycle < 3; ++cycle) {
        rerope_recent_fp16(krows, kv_heads, head_dim, sink, n, ltheta, -Delta);
        accum -= Delta;
    }
    if (std::memcmp(krows, sink_before.data(), sink_before.size() * sizeof(uint16_t)) != 0) return false;
    for (size_t t = sink; t < n; ++t)
        for (size_t h = 0; h < kv_heads; ++h) {
            std::vector<float> exp = rope_reference(pre[h][t], (double)t + accum, ltheta);
            const uint16_t* r = krows + (t * kv_heads + h) * head_dim;
            for (size_t d = 0; d < head_dim; ++d)
                if (std::abs(f16_to_f32(r[d]) - exp[d]) > 8e-2f) return false;   // looser for 3-cycle drift
        }
    return true;
}

bool test_rerope_local_theta_used() {
    // The sliding re-rope must use the LOCAL theta; the wrong (global) theta misses the position.
    const size_t n = 24, max_seq = 32, kv_heads = 1, head_dim = 16, sink = 4;
    const double ltheta = 10000.0, gtheta = 1000000.0, delta = -5.0;
    std::vector<std::vector<std::vector<float>>> pre, vals;
    auto kbuf = make_fp16_cache(n, max_seq, kv_heads, head_dim, ltheta, pre, vals);
    std::vector<char> kbuf2 = kbuf;
    auto* a = reinterpret_cast<uint16_t*>(kbuf.data() + kHeaderBytes);
    auto* b = reinterpret_cast<uint16_t*>(kbuf2.data() + kHeaderBytes);

    rerope_recent_fp16(a, kv_heads, head_dim, sink, n, ltheta, delta);   // correct theta
    rerope_recent_fp16(b, kv_heads, head_dim, sink, n, gtheta, delta);   // wrong theta

    bool correct_ok = true, wrong_differs = false;
    for (size_t t = sink; t < n; ++t) {
        std::vector<float> exp = rope_reference(pre[0][t], (double)t + delta, ltheta);
        const uint16_t* ra = a + (t * kv_heads) * head_dim;
        const uint16_t* rb = b + (t * kv_heads) * head_dim;
        for (size_t d = 0; d < head_dim; ++d) {
            if (std::abs(f16_to_f32(ra[d]) - exp[d]) > 5e-2f) correct_ok = false;
            if (std::abs(f16_to_f32(rb[d]) - exp[d]) > 5e-2f) wrong_differs = true;
        }
    }
    return correct_ok && wrong_differs;
}

bool test_rerope_zero_delta_noop() {
    const size_t n = 20, max_seq = 32, kv_heads = 2, head_dim = 16, sink = 4;
    const double theta = 10000.0;
    std::vector<std::vector<std::vector<float>>> pre, vals;
    auto kbuf = make_fp16_cache(n, max_seq, kv_heads, head_dim, theta, pre, vals);
    std::vector<char> before = kbuf;
    auto* krows = reinterpret_cast<uint16_t*>(kbuf.data() + kHeaderBytes);
    rerope_recent_fp16(krows, kv_heads, head_dim, sink, n, theta, 0.0);   // zero delta
    rerope_recent_fp16(krows, kv_heads, head_dim, n, n, theta, -3.0);     // empty range hi<=lo
    return kbuf == before;
}

// dequant_row's NEON path must match the scalar fallback, via int8 compact and the int8 keep-set fill.
bool test_dequant_simd_matches_scalar() {
    const size_t n = 64, max_seq = 96, kv_heads = 4, head_dim = 128;
    const double theta = 1000000.0;
    size_t groups = (head_dim + kGroupSize - 1) / kGroupSize;
    size_t int8_stride = kv_heads * head_dim, scale_stride = kv_heads * groups;
    Params p; p.recent_frac = 0.30f; p.sink = 4; p.abs_budget = 32;

    std::vector<int8_t> k8(max_seq * int8_stride, 0), v8(max_seq * int8_stride, 0);
    std::vector<float> ks(max_seq * scale_stride, 0.f), vs(max_seq * scale_stride, 0.f);
    for (size_t h = 0; h < kv_heads; ++h)
        for (size_t t = 0; t < n; ++t) {
            std::vector<float> k(head_dim), v(head_dim);
            for (size_t d = 0; d < head_dim; ++d) { k[d] = std::sin(0.1 * (h + 1) * (t + 1) + 0.2 * d); v[d] = std::cos(0.05 * (t + 1) + 0.1 * d); }
            std::vector<float> kr = rope_reference(k, (double)t, theta);
            quantize_row(kr, k8.data() + t * int8_stride + h * head_dim, ks.data() + t * scale_stride + h * groups, head_dim, kGroupSize);
            quantize_row(v, v8.data() + t * int8_stride + h * head_dim, vs.data() + t * scale_stride + h * groups, head_dim, kGroupSize);
        }
    std::vector<std::vector<int>> kept = {{0, 5, 11, 22}, {1, 4, 13, 26}, {2, 8, 17, 30}, {3, 9, 19, 40}};

    auto compact = [&](bool simd) {
        kv_set_simd(simd);
        std::vector<int8_t> a = k8;
        std::vector<float> b = ks;
        compact_int8(a.data(), b.data(), kv_heads, head_dim, kGroupSize, kept, theta, true);
        a.insert(a.end(), reinterpret_cast<int8_t*>(b.data()), reinterpret_cast<int8_t*>(b.data() + b.size()));
        return a;
    };
    auto keepsets = [&](bool simd) {
        kv_set_simd(simd);
        return keepsets_from_int8(k8.data(), ks.data(), n, kv_heads, head_dim, kGroupSize, theta, p);
    };
    bool ok = compact(true) == compact(false) && keepsets(true) == keepsets(false);
    kv_set_simd(true);
    return ok;
}

bool test_keepset_protect_union() {
    // A low-score position KeyDiff would evict is force-kept via Params::protect.
    const size_t n = 64;
    std::vector<float> scores(n);
    for (size_t i = 0; i < n; ++i) scores[i] = 0.01f * static_cast<float>(i);
    Params p; p.sink = 4; p.recent_frac = 0.25f; p.abs_budget = 24;
    const int special = 9;

    auto base = keepset_for_head(scores.data(), n, p);
    if (std::find(base.begin(), base.end(), special) != base.end()) return false;

    p.protect = {special};
    auto prot = keepset_for_head(scores.data(), n, p);
    if (std::find(prot.begin(), prot.end(), special) == prot.end()) return false;
    if (prot.size() != 24) return false;
    for (int i = 0; i < 4; ++i)
        if (std::find(prot.begin(), prot.end(), i) == prot.end()) return false;
    return true;
}

bool test_rolling_protect_survives() {
    // A token KeyDiff would evict survives each compaction cycle when re-protected.
    const int trigger_len = 256, target_len = 128;
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

    std::vector<float> special_v(head_dim);
    for (size_t d = 0; d < head_dim; ++d) special_v[d] = (d % 2 ? -3.0f : 3.0f);

    auto append = [&](size_t from, size_t to, long special_abs) {
        for (size_t t = from; t < to; ++t)
            for (size_t h = 0; h < kv_heads; ++h) {
                std::vector<float> k(head_dim);
                for (size_t d = 0; d < head_dim; ++d) k[d] = std::sin(0.05 * (h + 1) + 0.3 * d);
                std::vector<float> kr = rope_reference(k, (double)t, theta);
                std::vector<float> vr(head_dim);
                if ((long)t == special_abs) vr = special_v;
                else for (size_t d = 0; d < head_dim; ++d) vr[d] = std::cos(0.07 * (t + 1) + 0.2 * d);
                for (size_t d = 0; d < head_dim; ++d) {
                    krows[(t * kv_heads + h) * head_dim + d] = f32_to_f16(kr[d]);
                    vrows[(t * kv_heads + h) * head_dim + d] = f32_to_f16(vr[d]);
                }
            }
    };

    auto special_rank = [&]() -> long {
        for (size_t r = 0; r < khdr->current_seq_len; ++r) {
            const uint16_t* v = vrows + (r * kv_heads + 0) * head_dim;
            double dot = 0, na = 0, nb = 0;
            for (size_t d = 0; d < head_dim; ++d) {
                float x = f16_to_f32(v[d]); dot += x * special_v[d]; na += x * x; nb += special_v[d] * special_v[d];
            }
            if (dot / (std::sqrt(na) * std::sqrt(nb) + 1e-9) > 0.99) return (long)r;
        }
        return -1;
    };

    long special_pos = trigger_len / 2;
    for (int c = 0; c < 3; ++c) {
        size_t start = khdr->current_seq_len;
        append(start, (size_t)trigger_len, c == 0 ? special_pos : -1);
        khdr->current_seq_len = trigger_len;
        vhdr->current_seq_len = trigger_len;

        Params p; p.sink = 4; p.recent_frac = 0.30f; p.abs_budget = target_len;
        p.protect = {(int)special_pos};
        auto kept = keepsets_from_fp16(krows, (size_t)trigger_len, kv_heads, head_dim, theta, p);
        compact_fp16(krows, vrows, kv_heads, head_dim, kept, theta);
        khdr->current_seq_len = kept[0].size();
        vhdr->current_seq_len = kept[0].size();

        special_pos = special_rank();
        if (special_pos < 0) return false;
    }
    return true;
}

bool test_cache_starts_small_and_grows() {
    CactusGraph gb;
    const size_t kv_heads = 2, head_dim = 64, ceiling = 100000, chunk = 100;
    const size_t groups = (head_dim + kGroupSize - 1) / kGroupSize;
    const size_t int8_stride = kv_heads * head_dim, scale_stride = kv_heads * groups;
    size_t new_kv = gb.input({chunk, kv_heads, head_dim}, Precision::FP16);
    size_t state = gb.kv_cache_state(ceiling, kv_heads, head_dim, /*window*/0, /*sink*/4);
    size_t append = gb.kv_cache_append(new_kv, state, /*window*/0, /*sink*/4);
    gb.retain_outputs({static_cast<int>(state), static_cast<int>(append)});

    bool ok = true;
    std::vector<float> expect;
    auto val = [](size_t row, size_t c) { return std::sin(0.013 * row + 0.07 * c); };
    for (size_t step = 0; step < 3; ++step) {
        std::vector<uint16_t> in(chunk * int8_stride);
        for (size_t r = 0; r < chunk; ++r)
            for (size_t c = 0; c < int8_stride; ++c) {
                float v = static_cast<float>(val(step * chunk + r, c));
                in[r * int8_stride + c] = f32_to_f16(v);
                expect.push_back(v);
            }
        gb.set_input(new_kv, in.data(), Precision::FP16);
        gb.execute();
        if (step == 0 && reinterpret_cast<Header*>(gb.get_output(state))->max_seq_len != 256) ok = false;
    }

    const auto* hdr = reinterpret_cast<const Header*>(gb.get_output(state));
    if (hdr->max_seq_len != 512) ok = false;
    if (hdr->current_seq_len != 300) ok = false;

    const auto* base = static_cast<const char*>(gb.get_output(state));
    const int8_t* i8 = reinterpret_cast<const int8_t*>(base + kHeaderBytes);
    const float* sc = reinterpret_cast<const float*>(base + kHeaderBytes + 512 * int8_stride);
    for (size_t r = 0; r < 300 && ok; ++r)
        for (size_t h = 0; h < kv_heads; ++h)
            for (size_t d = 0; d < head_dim; ++d) {
                float dq = static_cast<float>(i8[r * int8_stride + h * head_dim + d]) *
                           sc[r * scale_stride + h * groups + d / kGroupSize];
                if (std::abs(dq - expect[r * int8_stride + h * head_dim + d]) > 0.05f) ok = false;
            }
    return ok;
}

bool test_sliding_layer_fixed_capacity() {
    CactusGraph gb;
    const size_t kv_heads = 2, head_dim = 64, ceiling = 100000, window = 512, sink = 4;
    size_t state = gb.kv_cache_state(ceiling, kv_heads, head_dim, window, sink);
    gb.retain_outputs({static_cast<int>(state)});
    gb.execute();
    auto* hdr = reinterpret_cast<Header*>(gb.get_output(state));
    return hdr->max_seq_len == window + sink + 1;
}

bool test_special_rows_remap_through_kept() {
    std::vector<int> rows = {3, 7, 10};
    std::vector<int> kept = {0, 3, 5, 7, 9, 10, 12};
    return remap_rows_through_kept(rows, kept) == std::vector<int>{1, 3, 5};
}

bool test_per_head_protect_keeps_specials() {
    const double theta = 1000000.0;
    const size_t n = 64, kv_heads = 3, head_dim = 16;
    std::vector<std::vector<std::vector<float>>> pre, val;
    auto kbuf = make_fp16_cache(n, n, kv_heads, head_dim, theta, pre, val);
    auto* krows = reinterpret_cast<uint16_t*>(kbuf.data() + kHeaderBytes);
    Params p; p.sink = 4; p.recent_frac = 0.30f; p.abs_budget = 24;
    auto unrope = unrope_table(n, head_dim, theta);
    std::vector<std::vector<int>> protect(kv_heads);
    for (size_t h = 0; h < kv_heads; ++h) protect[h] = {static_cast<int>(20 + h)};
    auto kept = keepsets_from_fp16(krows, n, kv_heads, head_dim, unrope, p, protect);
    if (kept.size() != kv_heads) return false;
    for (size_t h = 0; h < kv_heads; ++h)
        if (std::find(kept[h].begin(), kept[h].end(), static_cast<int>(20 + h)) == kept[h].end()) return false;
    return true;
}

bool test_preflight_specials_exceed_budget() {
    SpecialRowTracker tracked;
    std::vector<int> rows;
    for (int r = 0; r < 50; ++r) rows.push_back(r);
    tracked.add_appended(0, 2, rows);
    SpecialRowTracker untracked;
    return tracked.max_reserved(0, 4, {}) == 50 && tracked.max_reserved(0, 4, {}) > 40 &&
           untracked.max_reserved(0, 4, rows) == 50 && untracked.max_reserved(0, 4, rows) > 40;
}

bool test_empty_protect_per_head_uses_params_fallback() {
    const double theta = 1000000.0;
    const size_t n = 64, kv_heads = 3, head_dim = 16;
    std::vector<std::vector<std::vector<float>>> pre, val;
    auto kbuf = make_fp16_cache(n, n, kv_heads, head_dim, theta, pre, val);
    auto* krows = reinterpret_cast<uint16_t*>(kbuf.data() + kHeaderBytes);
    Params p; p.sink = 4; p.recent_frac = 0.30f; p.abs_budget = 24; p.protect = {25};
    auto unrope = unrope_table(n, head_dim, theta);
    auto kept = keepsets_from_fp16(krows, n, kv_heads, head_dim, unrope, p, /*protect_per_head=*/{});
    if (kept.size() != kv_heads) return false;
    for (size_t h = 0; h < kv_heads; ++h)
        if (std::find(kept[h].begin(), kept[h].end(), 25) == kept[h].end()) return false;
    return true;
}

bool test_all_heads_keep_special_across_cycles() {
    const int trigger_len = 256, target_len = 128;
    const size_t kv_heads = 4, head_dim = 16, max_seq = trigger_len + 8;
    const double theta = 1000000.0;
    std::vector<char> kbuf(kHeaderBytes + max_seq * kv_heads * head_dim * sizeof(uint16_t), 0);
    std::vector<char> vbuf(kbuf.size(), 0);
    auto* khdr = reinterpret_cast<Header*>(kbuf.data());
    auto* vhdr = reinterpret_cast<Header*>(vbuf.data());
    *khdr = Header{0, max_seq, kv_heads, head_dim, 4, {0, 0, 0}};
    *vhdr = *khdr;
    auto* krows = reinterpret_cast<uint16_t*>(kbuf.data() + kHeaderBytes);
    auto* vrows = reinterpret_cast<uint16_t*>(vbuf.data() + kHeaderBytes);

    auto special_val = [&](size_t h) {
        std::vector<float> v(head_dim);
        for (size_t d = 0; d < head_dim; ++d) v[d] = (d % 2 ? -2.0f : 2.0f) * static_cast<float>(h + 1);
        return v;
    };
    const long special_abs = trigger_len / 2;  // a MIDDLE position (not sink, not recent)
    auto append = [&](size_t from, size_t to) {
        for (size_t t = from; t < to; ++t)
            for (size_t h = 0; h < kv_heads; ++h) {
                std::vector<float> k(head_dim);
                // Special key is non-distinctive (KeyDiff would evict it); other rows differ per (h,t).
                for (size_t d = 0; d < head_dim; ++d)
                    k[d] = (static_cast<long>(t) == special_abs) ? 0.01f
                                                                 : std::sin(0.11 * (h + 1) * (t + 1) + 0.3 * d) + 0.05 * d;
                std::vector<float> kr = rope_reference(k, static_cast<double>(t), theta);
                std::vector<float> vr(head_dim);
                if (static_cast<long>(t) == special_abs) vr = special_val(h);
                else for (size_t d = 0; d < head_dim; ++d) vr[d] = std::cos(0.07 * (t + 1) + 0.2 * d);
                for (size_t d = 0; d < head_dim; ++d) {
                    krows[(t * kv_heads + h) * head_dim + d] = f32_to_f16(kr[d]);
                    vrows[(t * kv_heads + h) * head_dim + d] = f32_to_f16(vr[d]);
                }
            }
    };
    auto special_rank = [&](size_t h) -> long {
        std::vector<float> sv = special_val(h);
        for (size_t r = 0; r < khdr->current_seq_len; ++r) {
            const uint16_t* v = vrows + (r * kv_heads + h) * head_dim;
            double dot = 0, na = 0, nb = 0;
            for (size_t d = 0; d < head_dim; ++d) { float x = f16_to_f32(v[d]); dot += x * sv[d]; na += x * x; nb += sv[d] * sv[d]; }
            if (dot / (std::sqrt(na) * std::sqrt(nb) + 1e-9) > 0.99) return static_cast<long>(r);
        }
        return -1;
    };

    SpecialRowTracker tracker;
    Params p; p.sink = 4; p.recent_frac = 0.30f; p.abs_budget = target_len;
    auto unrope = unrope_table(trigger_len, head_dim, theta);
    for (int c = 0; c < 3; ++c) {
        size_t start = khdr->current_seq_len;
        append(start, static_cast<size_t>(trigger_len));
        khdr->current_seq_len = trigger_len; vhdr->current_seq_len = trigger_len;
        std::vector<int> appended;
        if (c == 0) appended.push_back(static_cast<int>(special_abs));
        tracker.add_appended(0, kv_heads, appended);
        auto kept = keepsets_from_fp16(krows, static_cast<size_t>(trigger_len), kv_heads, head_dim,
                                       unrope, p, tracker.protect(0));
        compact_fp16(krows, vrows, kv_heads, head_dim, kept, theta);
        tracker.remap(0, kept);
        size_t B = kept.empty() ? 0 : kept[0].size();
        khdr->current_seq_len = B; vhdr->current_seq_len = B;
        tracker.set_tracked_len(B);
    }
    long r0 = special_rank(0);
    if (r0 < 0) return false;
    bool diverged = false;
    for (size_t h = 0; h < kv_heads; ++h) {
        long r = special_rank(h);
        if (r < 0) return false;
        if (r != r0) diverged = true;  // ranks differ across heads -> a real per-head scenario, not trivially aligned
    }
    return diverged;
}

bool test_shrink_cache_buffer_preserves_rows() {
    CactusGraph gb;
    const size_t kv_heads = 2, head_dim = 64, ceiling = 100000, chunk = 300;
    const size_t groups = (head_dim + kGroupSize - 1) / kGroupSize;
    const size_t int8_stride = kv_heads * head_dim, scale_stride = kv_heads * groups;
    size_t new_kv = gb.input({chunk, kv_heads, head_dim}, Precision::FP16);
    size_t state = gb.kv_cache_state(ceiling, kv_heads, head_dim, /*window*/0, /*sink*/4);
    size_t append = gb.kv_cache_append(new_kv, state, /*window*/0, /*sink*/4);
    gb.retain_outputs({static_cast<int>(state), static_cast<int>(append)});

    std::vector<float> expect;
    auto val = [](size_t row, size_t c) { return std::sin(0.013 * row + 0.07 * c); };
    std::vector<uint16_t> in(chunk * int8_stride);
    for (size_t r = 0; r < chunk; ++r)
        for (size_t c = 0; c < int8_stride; ++c) {
            float v = static_cast<float>(val(r, c));
            in[r * int8_stride + c] = f32_to_f16(v);
            expect.push_back(v);
        }
    gb.set_input(new_kv, in.data(), Precision::FP16);
    gb.execute();  // appends 300 -> grows 256->512, occupancy 300

    auto* hdr = reinterpret_cast<Header*>(gb.get_output(state));
    if (hdr->max_seq_len != 512 || hdr->current_seq_len != 300) return false;

    gb.shrink_cache_buffer(state, 256);  // clamped up to occupancy 300
    hdr = reinterpret_cast<Header*>(gb.get_output(state));
    if (hdr->max_seq_len != 300 || hdr->current_seq_len != 300) return false;

    const auto* base = static_cast<const char*>(gb.get_output(state));
    const int8_t* i8 = reinterpret_cast<const int8_t*>(base + kHeaderBytes);
    const float* sc = reinterpret_cast<const float*>(base + kHeaderBytes + 300 * int8_stride);
    bool ok = true;
    for (size_t r = 0; r < 300 && ok; ++r)
        for (size_t h = 0; h < kv_heads; ++h)
            for (size_t d = 0; d < head_dim; ++d) {
                float dq = static_cast<float>(i8[r * int8_stride + h * head_dim + d]) *
                           sc[r * scale_stride + h * groups + d / kGroupSize];
                if (std::abs(dq - expect[r * int8_stride + h * head_dim + d]) > 0.05f) ok = false;
            }
    return ok;
}

int main() {
    TestUtils::TestRunner runner("KV Compress Free-Function Tests");
    runner.run_test("cache_starts_small_and_grows", test_cache_starts_small_and_grows());
    runner.run_test("sliding_layer_fixed_capacity", test_sliding_layer_fixed_capacity());
    runner.run_test("compact_fp16_cache", test_compact_fp16_cache());
    runner.run_test("dense_check_full_budget", test_dense_check_full_budget());
    runner.run_test("rope_renumber_contiguous", test_rope_renumber_contiguous());
    runner.run_test("fp16_storage_round_trip", test_fp16_storage_round_trip());
    runner.run_test("gemma_layer_selection", test_gemma_layer_selection());
    runner.run_test("compact_int8_cache", test_compact_int8_cache());
    runner.run_test("rerope_recent_fp16_uniform", test_rerope_recent_fp16_uniform());
    runner.run_test("rerope_recent_int8", test_rerope_recent_int8());
    runner.run_test("rotate_int8_row_matches_inline", test_rotate_int8_row_matches_inline());
    runner.run_test("sliding_plus_global_rolling", test_sliding_plus_global_rolling());
    runner.run_test("rerope_local_theta_used", test_rerope_local_theta_used());
    runner.run_test("rerope_zero_delta_noop", test_rerope_zero_delta_noop());
    runner.run_test("rolling_bounded_compaction", test_rolling_bounded_compaction());
    runner.run_test("config_parse_rolling_fields", test_config_parse_rolling_fields());
    runner.run_test("trigger_zero_gates_rolling", test_trigger_zero_gates_rolling());
    runner.run_test("degenerate_rolling_config_disabled", test_degenerate_rolling_config_disabled());
    runner.run_test("env_override_parse", test_env_override_parse());
    runner.run_test("dequant_simd_matches_scalar", test_dequant_simd_matches_scalar());
    runner.run_test("keepset_protect_union", test_keepset_protect_union());
    runner.run_test("rolling_protect_survives", test_rolling_protect_survives());
    runner.run_test("special_rows_remap_through_kept", test_special_rows_remap_through_kept());
    runner.run_test("per_head_protect_keeps_specials", test_per_head_protect_keeps_specials());
    runner.run_test("preflight_specials_exceed_budget", test_preflight_specials_exceed_budget());
    runner.run_test("empty_protect_per_head_uses_params_fallback", test_empty_protect_per_head_uses_params_fallback());
    runner.run_test("all_heads_keep_special_across_cycles", test_all_heads_keep_special_across_cycles());
    runner.run_test("shrink_cache_buffer_preserves_rows", test_shrink_cache_buffer_preserves_rows());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
