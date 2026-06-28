// LaRoSA vs CLAWS — same synthetic harness as
// test_bench_gemma4_mlp_actsparse.cpp (identical weight build, cold-cache
// 35-layer cycling, Gemma-4 E2B/E4B size mix, INT4 weights + INT8 acts).
//
// Gives LaRoSA's input-side (column) sparsity its fairest shot on ARM by
// reusing the SAME K-major repack + K-outer kmi4_v2 kernel that makes CLAWS's
// down-proj fast. Three variants:
//
//   dense  : gate+up+down all dense
//   claws  : gate dense + up real N-sparse + down K-major sparse (= CLAWS)
//   larosa : LaRoSA (Liu et al., 2025) — static layer-wise rotation + dynamic
//            per-token magnitude-threshold mask. Faithful to the paper: the
//            dynamic mask forces a FULL rotation Q^T x every token and a
//            K-major scatter on gate/up (can't pre-slice); down stays dense.
//
// All sparse matmuls use the best kernel (kmi4_v2) so no variant is
// handicapped by kernel choice. Router/mask precompute is done OUTSIDE the
// timed region for every variant (same convention as the original bench).
//
// MLP shapes: gate/up = [N=INTER x K=HIDDEN]; down = [N=HIDDEN x K=INTER].

#include "../cactus/kernel/kernel.h"

#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace {

constexpr size_t GROUP_SIZE = 32;
constexpr size_t INTERLEAVE = 4;
constexpr size_t HIDDEN     = 1536;
constexpr size_t INTER_S    = 6144;
constexpr size_t INTER_L    = 12288;
constexpr size_t INTER_MAX  = INTER_L;
constexpr size_t NUM_LAYERS = 35;
constexpr size_t SMALL_IN_CYCLE = 3;
constexpr size_t CYCLE_LEN      = 7;
constexpr size_t CACHE_TRASH_BYTES = 64ull * 1024 * 1024;

void pack_int4_32(const int8_t* src32, uint8_t* dst16) {
    for (size_t i = 0; i < 16; ++i) {
        uint8_t lo = static_cast<uint8_t>(src32[i]) & 0x0F;
        uint8_t hi = static_cast<uint8_t>(src32[16 + i]) & 0x0F;
        dst16[i] = lo | (hi << 4);
    }
}

struct DenseW {
    std::vector<uint8_t> packed;
    std::vector<__fp16> scales;
};

// Build a dense INT4 weight [N x K] in the 4-row-interleaved pack + fp16 scales.
DenseW build_weights(size_t N, size_t K, unsigned seed) {
    DenseW w;
    const size_t num_groups = K / GROUP_SIZE;
    const size_t N_blocks = (N + INTERLEAVE - 1) / INTERLEAVE;
    const size_t N_padded = N_blocks * INTERLEAVE;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    std::vector<int8_t> q(N * K);
    std::vector<float> scales(N * num_groups);
    for (size_t n = 0; n < N; ++n)
        for (size_t g = 0; g < num_groups; ++g) {
            float max_abs = 1e-5f, buf[GROUP_SIZE];
            for (size_t i = 0; i < GROUP_SIZE; ++i) {
                float v = ud(rng); buf[i] = v;
                max_abs = std::max(max_abs, std::fabs(v));
            }
            float s = max_abs / 7.0f;
            scales[n * num_groups + g] = s;
            for (size_t i = 0; i < GROUP_SIZE; ++i) {
                int32_t qv = static_cast<int32_t>(std::round(buf[i] / s));
                qv = std::max(-8, std::min(7, qv));
                q[n * K + g * GROUP_SIZE + i] = static_cast<int8_t>(qv);
            }
        }
    std::vector<int8_t> interleaved(N_padded * K, 0);
    for (size_t nb = 0; nb < N_blocks; ++nb)
        for (size_t kg = 0; kg < K / 4; ++kg)
            for (size_t bi = 0; bi < INTERLEAVE; ++bi) {
                size_t row = nb * INTERLEAVE + bi;
                if (row >= N) continue;
                for (size_t ki = 0; ki < 4; ++ki) {
                    size_t dst = ((nb * (K / 4) + kg) * INTERLEAVE + bi) * 4 + ki;
                    interleaved[dst] = q[row * K + kg * 4 + ki];
                }
            }
    w.packed.resize(N_padded * K / 2);
    for (size_t i = 0; i < N_padded * K; i += 32)
        pack_int4_32(interleaved.data() + i, w.packed.data() + i / 2);
    w.scales.assign(N_blocks * num_groups * INTERLEAVE, __fp16(1e-6f));
    for (size_t nb = 0; nb < N_blocks; ++nb)
        for (size_t bi = 0; bi < INTERLEAVE; ++bi) {
            size_t row = nb * INTERLEAVE + bi;
            if (row >= N) continue;
            for (size_t g = 0; g < num_groups; ++g)
                w.scales[(nb * num_groups + g) * INTERLEAVE + bi] =
                    static_cast<__fp16>(scales[row * num_groups + g]);
        }
    return w;
}

struct LayerWeights {
    size_t intermediate;
    DenseW gate, up, down, rot;       // rot = Q (HIDDEN x HIDDEN) int4 rotation
    std::vector<uint8_t> down_km_inline;   // K-major: K=INTER  (CLAWS down)
    std::vector<uint8_t> gate_km_inline;   // K-major: K=HIDDEN (LaRoSA gate)
    std::vector<uint8_t> up_km_inline;     // K-major: K=HIDDEN (LaRoSA up)
};

LayerWeights build_layer(size_t intermediate, size_t seed_base) {
    LayerWeights l;
    l.intermediate = intermediate;
    l.gate = build_weights(intermediate, HIDDEN, static_cast<unsigned>(seed_base + 1));
    l.up   = build_weights(intermediate, HIDDEN, static_cast<unsigned>(seed_base + 2));
    l.down = build_weights(HIDDEN, intermediate, static_cast<unsigned>(seed_base + 3));
    l.rot  = build_weights(HIDDEN, HIDDEN,       static_cast<unsigned>(seed_base + 4));

    const size_t dn_groups = intermediate / GROUP_SIZE;
    const size_t dn_nblk   = (HIDDEN + INTERLEAVE - 1) / INTERLEAVE;
    l.down_km_inline.resize(dn_groups * dn_nblk * 72);
    cactus_repack_int4_kmajor_inline(
        reinterpret_cast<const int8_t*>(l.down.packed.data()),
        l.down.scales.data(), l.down_km_inline.data(),
        intermediate, HIDDEN, GROUP_SIZE);

    const size_t gu_groups = HIDDEN / GROUP_SIZE;
    const size_t gu_nblk   = (intermediate + INTERLEAVE - 1) / INTERLEAVE;
    l.gate_km_inline.resize(gu_groups * gu_nblk * 72);
    l.up_km_inline.resize(gu_groups * gu_nblk * 72);
    cactus_repack_int4_kmajor_inline(
        reinterpret_cast<const int8_t*>(l.gate.packed.data()),
        l.gate.scales.data(), l.gate_km_inline.data(),
        HIDDEN, intermediate, GROUP_SIZE);
    cactus_repack_int4_kmajor_inline(
        reinterpret_cast<const int8_t*>(l.up.packed.data()),
        l.up.scales.data(), l.up_km_inline.data(),
        HIDDEN, intermediate, GROUP_SIZE);
    return l;
}

void make_blocky_scores(std::vector<float>& S, size_t K, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    S.assign(K, 0.f);
    const size_t nb = (K + GROUP_SIZE - 1) / GROUP_SIZE;
    for (size_t b = 0; b < nb; ++b) {
        float coarse = -std::log(std::max(1e-6f, u(rng)));
        size_t k0 = b * GROUP_SIZE, k1 = std::min(k0 + GROUP_SIZE, K);
        for (size_t k = k0; k < k1; ++k) S[k] = coarse + 0.1f * u(rng);
    }
}

void build_group_bitmask_from_scores(const std::vector<float>& S, size_t K,
                                     float sparsity, std::vector<uint64_t>& bm) {
    const size_t ng = K / GROUP_SIZE;
    std::vector<float> gmax(ng, 0.f);
    for (size_t g = 0; g < ng; ++g) {
        float m = 0.f;
        for (size_t j = 0; j < GROUP_SIZE; ++j)
            m = std::max(m, std::fabs(S[g * GROUP_SIZE + j]));
        gmax[g] = m;
    }
    size_t drop = static_cast<size_t>(sparsity * static_cast<float>(ng));
    if (drop >= ng) drop = ng - 1;
    std::vector<float> sorted = gmax;
    std::nth_element(sorted.begin(), sorted.begin() + drop, sorted.end());
    float thr = sorted[drop];
    bm.assign((ng + 63) / 64, 0);
    for (size_t g = 0; g < ng; ++g)
        if (gmax[g] > thr) bm[g >> 6] |= uint64_t(1) << (g & 63);
}

void trash_cache(std::vector<uint8_t>& t) {
    for (size_t i = 0; i < t.size(); i += 64) t[i] = static_cast<uint8_t>(i);
    asm volatile ("" : : "r"(t.data()) : "memory");
}

float quantize_f16_to_int8(const __fp16* src, int8_t* dst, size_t n) {
    float max_abs = 1e-5f;
    for (size_t i = 0; i < n; ++i)
        max_abs = std::max(max_abs, std::fabs(static_cast<float>(src[i])));
    float scale = max_abs / 127.0f, inv = 1.0f / scale;
    for (size_t i = 0; i < n; ++i) {
        int v = static_cast<int>(std::lround(static_cast<float>(src[i]) * inv));
        dst[i] = static_cast<int8_t>(std::max(-127, std::min(127, v)));
    }
    return scale;
}

// Extract a live-group index list from a bitmask (router precompute; untimed).
size_t extract_live_groups(const uint64_t* bm, size_t ng, uint16_t* live) {
    size_t n = 0;
    for (size_t qw = 0; qw * 64 < ng; ++qw) {
        uint64_t bits = bm[qw];
        size_t g0 = qw * 64, g1 = std::min(g0 + 64, ng);
        for (size_t g = g0; g < g1; ++g)
            if ((bits >> (g - g0)) & 1ull) live[n++] = static_cast<uint16_t>(g);
    }
    return n;
}

double ns_to_us(std::chrono::steady_clock::duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
}

struct MlpScratch {
    std::vector<int8_t> x_q, h_q, h_masked, x_rot_q, x_rot_masked;
    std::vector<__fp16> gate_out, up_out, x_rot, out;
    std::vector<uint16_t> live_groups, up_live_groups, rot_live_groups;
};

void init_scratch(MlpScratch& s, unsigned seed) {
    s.x_q.resize(HIDDEN);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> ad(-120, 120);
    for (auto& v : s.x_q) v = static_cast<int8_t>(ad(rng));
    s.h_q.resize(INTER_MAX);  s.h_masked.resize(INTER_MAX);
    s.x_rot_q.resize(HIDDEN); s.x_rot_masked.resize(HIDDEN);
    s.gate_out.resize(INTER_MAX); s.up_out.resize(INTER_MAX);
    s.x_rot.resize(HIDDEN); s.out.resize(HIDDEN);
    s.live_groups.resize(INTER_MAX / GROUP_SIZE);
    s.up_live_groups.resize(INTER_MAX / GROUP_SIZE);
    s.rot_live_groups.resize(HIDDEN / GROUP_SIZE);
}

// Per-layer precomputed masks (router output; built outside timing).
struct LayerMask {
    size_t num_up_live;                      // CLAWS: live N-groups of W_up
    size_t num_down_live;                    // CLAWS: live K-groups of W_down
    size_t num_rot_live;                     // LaRoSA: live rotated-input groups
    std::vector<uint16_t> up_live, down_live, rot_live;
    std::vector<int8_t>   h_masked;          // CLAWS down input (masked h)
    std::vector<int8_t>   x_rot_masked;      // LaRoSA gate/up input (masked rot x)
    float rot_scale;
};

// ----------------------------- variants -----------------------------------
// Each returns µs for the full gate+up+down MLP of one layer (cold cache).

double mlp_dense(const LayerWeights& L, MlpScratch& s) {
    const size_t INTER = L.intermediate;
    auto t0 = std::chrono::steady_clock::now();
    cactus_gemv_int4(s.x_q.data(), 0.05f,
        reinterpret_cast<const int8_t*>(L.gate.packed.data()),
        L.gate.scales.data(), s.gate_out.data(), HIDDEN, INTER, GROUP_SIZE);
    cactus_gemv_int4(s.x_q.data(), 0.05f,
        reinterpret_cast<const int8_t*>(L.up.packed.data()),
        L.up.scales.data(), s.up_out.data(), HIDDEN, INTER, GROUP_SIZE);
    cactus_gelu_mul_quant_fp16_to_int8(s.gate_out.data(), s.up_out.data(),
                                       s.h_q.data(), INTER, 8.0f);
    cactus_gemv_int4(s.h_q.data(), 0.125f,
        reinterpret_cast<const int8_t*>(L.down.packed.data()),
        L.down.scales.data(), s.out.data(), INTER, HIDDEN, GROUP_SIZE);
    return ns_to_us(std::chrono::steady_clock::now() - t0);
}

// CLAWS: gate dense + up real N-sparse + down K-major sparse (kmi4_v2).
double mlp_claws(const LayerWeights& L, MlpScratch& s, const LayerMask& m) {
    const size_t INTER = L.intermediate;
    auto t0 = std::chrono::steady_clock::now();
    cactus_gemv_int4(s.x_q.data(), 0.05f,
        reinterpret_cast<const int8_t*>(L.gate.packed.data()),
        L.gate.scales.data(), s.gate_out.data(), HIDDEN, INTER, GROUP_SIZE);
    std::memset(s.up_out.data(), 0, INTER * sizeof(__fp16));
    cactus_gemv_int4_nsparse_up(s.x_q.data(), 0.05f,
        reinterpret_cast<const int8_t*>(L.up.packed.data()),
        L.up.scales.data(), m.up_live.data(), m.num_up_live,
        s.up_out.data(), HIDDEN, INTER, GROUP_SIZE, GROUP_SIZE);
    cactus_gelu_mul_quant_fp16_to_int8(s.gate_out.data(), s.up_out.data(),
                                       s.h_q.data(), INTER, 8.0f);
    cactus_gemv_int4_actsparse_kmi4_v2(m.h_masked.data(), 0.125f,
        L.down_km_inline.data(), m.down_live.data(), m.num_down_live,
        s.out.data(), INTER, HIDDEN, GROUP_SIZE);
    return ns_to_us(std::chrono::steady_clock::now() - t0);
}

// LaRoSA (Liu et al., 2025): static layer-wise rotation + DYNAMIC per-token
// magnitude-threshold mask on the rotated input. The dynamic mask forces both
// costs modeled here: (1) the FULL rotation must run every token (you can't
// know which rotated dims clear the threshold without computing all of them),
// and (2) the kept set changes per token, so gate/up can't be pre-sliced and
// must use the K-major scatter kernel. Down stays dense (input-side paradigm).
double mlp_larosa(const LayerWeights& L, MlpScratch& s, const LayerMask& m) {
    const size_t INTER = L.intermediate;
    auto t0 = std::chrono::steady_clock::now();
    // 1. full rotation Q^T x (int4, all HIDDEN dims — needed to threshold).
    cactus_gemv_int4(s.x_q.data(), 0.05f,
        reinterpret_cast<const int8_t*>(L.rot.packed.data()),
        L.rot.scales.data(), s.x_rot.data(), HIDDEN, HIDDEN, GROUP_SIZE);
    quantize_f16_to_int8(s.x_rot.data(), s.x_rot_q.data(), HIDDEN);
    // 2. K-major sparse gate/up over live rotated-input groups; dense down.
    cactus_gemv_int4_actsparse_kmi4_v2(m.x_rot_masked.data(), m.rot_scale,
        L.gate_km_inline.data(), m.rot_live.data(), m.num_rot_live,
        s.gate_out.data(), HIDDEN, INTER, GROUP_SIZE);
    cactus_gemv_int4_actsparse_kmi4_v2(m.x_rot_masked.data(), m.rot_scale,
        L.up_km_inline.data(), m.rot_live.data(), m.num_rot_live,
        s.up_out.data(), HIDDEN, INTER, GROUP_SIZE);
    cactus_gelu_mul_quant_fp16_to_int8(s.gate_out.data(), s.up_out.data(),
                                       s.h_q.data(), INTER, 8.0f);
    cactus_gemv_int4(s.h_q.data(), 0.125f,
        reinterpret_cast<const int8_t*>(L.down.packed.data()),
        L.down.scales.data(), s.out.data(), INTER, HIDDEN, GROUP_SIZE);
    return ns_to_us(std::chrono::steady_clock::now() - t0);
}

// ----------------------------- harness -------------------------------------

struct Pool {
    std::vector<LayerWeights> layers;
    std::vector<std::vector<float>> ffn_scores, rot_scores;
    // [sparsity][layer]
    std::vector<std::vector<LayerMask>> masks;
};

Pool build_pool(const std::vector<float>& sparsities) {
    Pool p;
    for (size_t l = 0; l < NUM_LAYERS; ++l) {
        size_t INTER = (l % CYCLE_LEN < SMALL_IN_CYCLE) ? INTER_S : INTER_L;
        p.layers.push_back(build_layer(INTER, 0xE2B0 + l * 101));
    }
    p.ffn_scores.resize(NUM_LAYERS);
    p.rot_scores.resize(NUM_LAYERS);
    for (size_t l = 0; l < NUM_LAYERS; ++l) {
        make_blocky_scores(p.ffn_scores[l], p.layers[l].intermediate, 0xABCD + l * 37);
        make_blocky_scores(p.rot_scores[l], HIDDEN, 0x5A5A + l * 53);
    }
    // Precompute all masks (router output) once, outside any timing.
    MlpScratch tmp; init_scratch(tmp, 0x1234);
    p.masks.resize(sparsities.size());
    for (size_t si = 0; si < sparsities.size(); ++si) {
        p.masks[si].resize(NUM_LAYERS);
        for (size_t l = 0; l < NUM_LAYERS; ++l) {
            size_t INTER = p.layers[l].intermediate;
            LayerMask& m = p.masks[si][l];
            // FFN-axis bitmask -> CLAWS up (N-groups) + down (K-groups, masked h).
            std::vector<uint64_t> ffn_bm;
            build_group_bitmask_from_scores(p.ffn_scores[l], INTER, sparsities[si], ffn_bm);
            m.up_live.resize(INTER / GROUP_SIZE);
            m.num_up_live = extract_live_groups(ffn_bm.data(), INTER / GROUP_SIZE,
                                                m.up_live.data());
            m.h_masked.resize(INTER);
            m.down_live.resize(INTER / GROUP_SIZE);
            m.num_down_live = cactus_apply_actsparse_bitmask(
                ffn_bm.data(), tmp.h_q.data(), INTER, GROUP_SIZE,
                m.h_masked.data(), m.down_live.data());
            // Rotated-input bitmask -> LaRoSA gate/up (K-groups over HIDDEN).
            std::vector<uint64_t> rot_bm;
            build_group_bitmask_from_scores(p.rot_scores[l], HIDDEN, sparsities[si], rot_bm);
            cactus_gemv_int4(tmp.x_q.data(), 0.05f,
                reinterpret_cast<const int8_t*>(p.layers[l].rot.packed.data()),
                p.layers[l].rot.scales.data(), tmp.x_rot.data(),
                HIDDEN, HIDDEN, GROUP_SIZE);
            m.rot_scale = quantize_f16_to_int8(tmp.x_rot.data(), tmp.x_rot_q.data(), HIDDEN);
            m.x_rot_masked.resize(HIDDEN);
            m.rot_live.resize(HIDDEN / GROUP_SIZE);
            m.num_rot_live = cactus_apply_actsparse_bitmask(
                rot_bm.data(), tmp.x_rot_q.data(), HIDDEN, GROUP_SIZE,
                m.x_rot_masked.data(), m.rot_live.data());
        }
    }
    return p;
}

struct Stat { double stack_us; double small_us; double large_us; };

template <class Fn>
Stat run_variant(Pool& pool, MlpScratch& s, std::vector<uint8_t>& trash,
                 size_t warmup, size_t bench, Fn fn) {
    std::vector<double> stack(bench, 0.0);
    double small_sum = 0, large_sum = 0; size_t small_n = 0, large_n = 0;
    for (size_t pass = 0; pass < warmup + bench; ++pass) {
        double sum = 0;
        for (size_t l = 0; l < pool.layers.size(); ++l) {
            trash_cache(trash);
            double t = fn(pool.layers[l], s, l);
            if (pass >= warmup) {
                sum += t;
                bool small = pool.layers[l].intermediate == INTER_S;
                if (small) { small_sum += t; if (pass == warmup) ++small_n; }
                else       { large_sum += t; if (pass == warmup) ++large_n; }
            }
        }
        if (pass >= warmup) stack[pass - warmup] = sum;
    }
    std::sort(stack.begin(), stack.end());
    double sc = static_cast<double>(small_n * bench), lc = static_cast<double>(large_n * bench);
    return { stack[stack.size() / 2], sc ? small_sum / sc : 0, lc ? large_sum / lc : 0 };
}

} // namespace

int main() {
    std::printf("LaRoSA vs CLAWS — same synthetic harness, cold cache\n");
    std::printf("  35 layers (3 small INTER=%zu : 4 large INTER=%zu per 7), HIDDEN=%zu\n",
                INTER_S, INTER_L, HIDDEN);
    std::printf("  hw.concurrency=%u\n", std::thread::hardware_concurrency());

    const std::vector<float> sparsities = {0.30f, 0.50f, 0.60f, 0.70f, 0.80f, 0.90f};
    std::printf("\nBuilding pool + precomputing masks...\n");
    Pool pool = build_pool(sparsities);

    MlpScratch s; init_scratch(s, 0x1234);
    std::vector<uint8_t> trash(CACHE_TRASH_BYTES, 0);
    constexpr size_t WARMUP = 1, BENCH = 8;

    std::printf("\n===== FULL 35-LAYER STACK (median µs; ×dense, higher=faster) =====\n");
    std::printf("  LaRoSA = static rotation + dynamic per-token mask (full rotation +\n");
    std::printf("  K-major sparse gate/up + dense down). CLAWS/LaRoSA > 1 => CLAWS faster.\n");
    std::printf("%-5s %10s %14s %14s   %12s\n",
                "sp%", "dense", "CLAWS(x)", "LaRoSA(x)", "CLAWS/LaRoSA");

    FILE* csv = std::fopen("larosa_vs_claws_bench.csv", "w");
    if (csv) std::fprintf(csv, "sparsity,dense_us,claws_us,larosa_us\n");

    for (size_t si = 0; si < sparsities.size(); ++si) {
        float sp = sparsities[si];
        auto& M = pool.masks[si];
        Stat d  = run_variant(pool, s, trash, WARMUP, BENCH,
            [&](const LayerWeights& L, MlpScratch& sc, size_t){ return mlp_dense(L, sc); });
        Stat cl = run_variant(pool, s, trash, WARMUP, BENCH,
            [&](const LayerWeights& L, MlpScratch& sc, size_t l){ return mlp_claws(L, sc, M[l]); });
        Stat lr = run_variant(pool, s, trash, WARMUP, BENCH,
            [&](const LayerWeights& L, MlpScratch& sc, size_t l){ return mlp_larosa(L, sc, M[l]); });

        double D = d.stack_us;
        std::printf("%4.0f%% %10.1f %10.1f(%4.2f) %10.1f(%4.2f)   %10.2fx\n",
                    sp * 100.0, D,
                    cl.stack_us, D / cl.stack_us,
                    lr.stack_us, D / lr.stack_us,
                    lr.stack_us / cl.stack_us);
        if (csv) std::fprintf(csv, "%.2f,%.2f,%.2f,%.2f\n",
                              sp, D, cl.stack_us, lr.stack_us);
    }
    if (csv) std::fclose(csv);
    return 0;
}
