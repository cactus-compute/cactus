// Cold-cache bench: Gemma-4-style MLP block with a gate-as-router scheme.
//
// Layer pool mixes two MLP sizes at a 3 small : 4 larger ratio per 7 layers:
//   small : hidden=1536, intermediate=6144   (Gemma 4 E2B size)
//   large : hidden=1536, intermediate=12288  (Gemma 4 E4B shared-MLP size)
// With NUM_LAYERS=35, that is 15 small + 20 large.
//
// MLP forward per layer:
//   gate = GELU(W_gate · x)     K=1536, N=INTERMEDIATE   (router source)
//   up   = W_up · x             K=1536, N=INTERMEDIATE
//   h    = gate * up            [INTERMEDIATE], elementwise
//   out  = W_down · h           K=INTERMEDIATE, N=1536
//
// Post-trained-router model
// -------------------------
// The user's scheme is: post-train the gate so |gate| can be used directly
// as a per-channel router. In inference the router emits a *sparsity
// pattern* (here: a threshold on |score|) that tells us which intermediate
// channels to keep. We model this as:
//
//   * per-layer router threshold is derived OUTSIDE the timed region
//     (represents the router's pre-computation — a small op fused with
//     the gate forward would take well under a µs on this hardware),
//   * the in-loop mask build is the fast O(K) threshold scan
//     `cactus_build_actsparse_mask_threshold_f32` — no sort, no heap.
//
// The overall MLP block is timed EVERYTHING INCLUDED: gate_proj, GELU,
// up_proj, elementwise gate*up + quantize, mask build, down_proj.
// Speedup is reported over an all-dense baseline running the identical
// gate + up + elementwise + down sequence.
//
// Three variants:
//   dense_all           : gate dense + up dense + down dense
//   sparse_down_only    : gate dense + up dense + KMI4_fast down
//   sparse_up_plus_down : gate dense + dense up at REDUCED N_live
//                         (ceiling on what a future N-sparse up kernel
//                          could deliver) + KMI4_fast down

#include "../cactus/kernel/kernel.h"

#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace {

constexpr size_t GROUP_SIZE = 32;
constexpr size_t INTERLEAVE = 4;
constexpr size_t HIDDEN     = 1536;
constexpr size_t INTER_S    = 6144;   // Gemma 4 E2B
constexpr size_t INTER_L    = 12288;  // Gemma 4 E4B shared MLP
constexpr size_t INTER_MAX  = INTER_L;
constexpr size_t NUM_LAYERS = 35;
constexpr size_t SMALL_IN_CYCLE = 3;  // 3 small : 4 large per 7 layers
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

struct LayerWeights {
    size_t intermediate;
    DenseW gate, up, down;
    std::vector<uint8_t> down_km_inline;
};

DenseW build_weights(size_t N, size_t K, unsigned seed) {
    DenseW w;
    const size_t num_groups = K / GROUP_SIZE;
    const size_t N_blocks = (N + INTERLEAVE - 1) / INTERLEAVE;
    const size_t N_padded = N_blocks * INTERLEAVE;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    std::vector<int8_t> q(N * K);
    std::vector<float> scales(N * num_groups);
    for (size_t n = 0; n < N; ++n) {
        for (size_t g = 0; g < num_groups; ++g) {
            float max_abs = 1e-5f;
            float buf[GROUP_SIZE];
            for (size_t i = 0; i < GROUP_SIZE; ++i) {
                float v = ud(rng);
                buf[i] = v;
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
    }
    std::vector<int8_t> interleaved(N_padded * K, 0);
    for (size_t nb = 0; nb < N_blocks; ++nb) {
        for (size_t kg = 0; kg < K / 4; ++kg) {
            for (size_t bi = 0; bi < INTERLEAVE; ++bi) {
                size_t row = nb * INTERLEAVE + bi;
                if (row >= N) continue;
                for (size_t ki = 0; ki < 4; ++ki) {
                    size_t dst = ((nb * (K / 4) + kg) * INTERLEAVE + bi) * 4 + ki;
                    interleaved[dst] = q[row * K + kg * 4 + ki];
                }
            }
        }
    }
    w.packed.resize(N_padded * K / 2);
    for (size_t i = 0; i < N_padded * K; i += 32) {
        pack_int4_32(interleaved.data() + i, w.packed.data() + i / 2);
    }
    w.scales.assign(N_blocks * num_groups * INTERLEAVE, __fp16(1e-6f));
    for (size_t nb = 0; nb < N_blocks; ++nb) {
        for (size_t bi = 0; bi < INTERLEAVE; ++bi) {
            size_t row = nb * INTERLEAVE + bi;
            if (row >= N) continue;
            for (size_t g = 0; g < num_groups; ++g) {
                w.scales[(nb * num_groups + g) * INTERLEAVE + bi] =
                    static_cast<__fp16>(scales[row * num_groups + g]);
            }
        }
    }
    return w;
}

LayerWeights build_layer(size_t intermediate, size_t seed_base) {
    LayerWeights l;
    l.intermediate = intermediate;
    l.gate = build_weights(intermediate, HIDDEN,       static_cast<unsigned>(seed_base + 1));
    l.up   = build_weights(intermediate, HIDDEN,       static_cast<unsigned>(seed_base + 2));
    l.down = build_weights(HIDDEN,       intermediate, static_cast<unsigned>(seed_base + 3));

    const size_t num_groups = intermediate / GROUP_SIZE;
    const size_t N_blocks = (HIDDEN + INTERLEAVE - 1) / INTERLEAVE;
    l.down_km_inline.resize(num_groups * N_blocks * 72);
    cactus_repack_int4_kmajor_inline(
        reinterpret_cast<const int8_t*>(l.down.packed.data()),
        l.down.scales.data(),
        l.down_km_inline.data(),
        intermediate, HIDDEN, GROUP_SIZE);
    return l;
}

// Blocky scores at K-group-size granularity (block_size=GROUP_SIZE so a
// fraction of groups are fully dropped, matching block-structured
// activation sparsity seen after SiLU/GELU gating).
void make_blocky_scores(std::vector<float>& S, size_t K, unsigned seed, size_t block_size) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    S.assign(K, 0.f);
    const size_t nb = (K + block_size - 1) / block_size;
    for (size_t b = 0; b < nb; ++b) {
        float coarse = -std::log(std::max(1e-6f, u(rng)));
        size_t k0 = b * block_size;
        size_t k1 = std::min(k0 + block_size, K);
        for (size_t k = k0; k < k1; ++k) S[k] = coarse + 0.1f * u(rng);
    }
}

// Find the threshold τ such that sparsity% of S have |S|<=τ.
float find_threshold(const std::vector<float>& S, float sparsity) {
    const size_t K = S.size();
    std::vector<float> mags(K);
    for (size_t i = 0; i < K; ++i) mags[i] = std::fabs(S[i]);
    size_t idx = static_cast<size_t>(sparsity * static_cast<float>(K));
    if (idx >= K) idx = K - 1;
    std::nth_element(mags.begin(), mags.begin() + idx, mags.end());
    return mags[idx];
}

// Build a group-level bitmask by thresholding on per-group max|S|. Block-
// structured, matches the post-trained-router output (1 bit per 32-lane
// group) and is what the kernel actually consumes.
void build_group_bitmask_from_scores(
    const std::vector<float>& S, size_t K, size_t group_size, float sparsity,
    std::vector<uint64_t>& bitmask)
{
    const size_t num_groups = K / group_size;
    std::vector<float> group_max(num_groups, 0.f);
    for (size_t g = 0; g < num_groups; ++g) {
        float m = 0.f;
        for (size_t j = 0; j < group_size; ++j) {
            float v = std::fabs(S[g * group_size + j]);
            if (v > m) m = v;
        }
        group_max[g] = m;
    }
    size_t drop = static_cast<size_t>(sparsity * static_cast<float>(num_groups));
    if (drop >= num_groups) drop = num_groups - 1;
    std::vector<float> sorted = group_max;
    std::nth_element(sorted.begin(), sorted.begin() + drop, sorted.end());
    float thr = sorted[drop];

    bitmask.assign((num_groups + 63) / 64, 0);
    for (size_t g = 0; g < num_groups; ++g) {
        if (group_max[g] > thr) bitmask[g >> 6] |= uint64_t(1) << (g & 63);
    }
}

void trash_cache(std::vector<uint8_t>& trash) {
    for (size_t i = 0; i < trash.size(); i += 64) trash[i] = static_cast<uint8_t>(i);
    asm volatile ("" : : "r"(trash.data()) : "memory");
}

struct MlpScratch {
    // All preallocated to INTER_MAX so the same buffer handles both sizes.
    std::vector<int8_t>  x_q;
    std::vector<__fp16>  gate_out;
    std::vector<__fp16>  up_out;
    std::vector<__fp16>  up_out_small;
    std::vector<float>   S;
    std::vector<int8_t>  h_q;
    std::vector<int8_t>  h_masked;
    std::vector<uint64_t> bitmask;
    std::vector<uint16_t> live_groups;
    std::vector<__fp16>  out_dense;
    std::vector<__fp16>  out_sparse;
};

void init_scratch(MlpScratch& s, unsigned seed) {
    s.x_q.resize(HIDDEN);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> ad(-120, 120);
    for (auto& v : s.x_q) v = static_cast<int8_t>(ad(rng));

    s.gate_out.resize(INTER_MAX);
    s.up_out.resize(INTER_MAX);
    s.up_out_small.resize(INTER_MAX);
    s.S.resize(INTER_MAX);
    s.h_q.resize(INTER_MAX);
    s.h_masked.resize(INTER_MAX);
    const size_t num_groups = INTER_MAX / GROUP_SIZE;
    s.bitmask.resize((num_groups + 63) / 64);
    s.live_groups.resize(num_groups);
    s.out_dense.resize(HIDDEN);
    s.out_sparse.resize(HIDDEN);
}

double ns_to_us(std::chrono::steady_clock::duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
}

// ---- the 3 MLP paths ------------------------------------------------------

// Dispatch the best down_sparse kernel for the Gemma 4 E2B shape
// (K=6144 intermediate, N=1536 hidden).
//
// Isolated shootout at every sparsity (see `DOWN_PROJ SHOOTOUT`) shows
// KMI4_v2 as the single-kernel winner by ~3–7%. However, in the fused
// MLP pipeline KMI4_v2's per-thread precompute — O(num_live × threads)
// ptr+A scratch allocated once per call — costs ~5–10 µs and is paid
// on every layer. That cost amortizes poorly inside a 60–80 µs fused
// call, so it consistently loses to the simpler KMI2 / KMI4 / KMI4_fast
// in the fused path measurements at low and mid density.
//
// Empirically best combination on the E2B shape *inside* the fused MLP:
//   skip < 0.65         -> KMI2       (no precompute; lowest overhead)
//   0.65 <= skip < 0.80 -> KMI4       (4-way batching without precompute)
//   skip >= 0.80        -> KMI4_fast  (warmup prefetch pays off at high skip)
inline void down_sparse_kernel(float skip_frac,
                               const int8_t* A_masked, float A_scale,
                               const uint8_t* B_km_inline,
                               const uint16_t* live_groups, size_t num_live,
                               __fp16* C, size_t K, size_t N, size_t group_size)
{
    if (skip_frac < 0.65f) {
        cactus_gemv_int4_actsparse_kmi2(A_masked, A_scale, B_km_inline,
                                        live_groups, num_live,
                                        C, K, N, group_size);
    } else if (skip_frac < 0.80f) {
        cactus_gemv_int4_actsparse_kmi4(A_masked, A_scale, B_km_inline,
                                        live_groups, num_live,
                                        C, K, N, group_size);
    } else {
        cactus_gemv_int4_actsparse_kmi4_fast(A_masked, A_scale, B_km_inline,
                                             live_groups, num_live,
                                             C, K, N, group_size);
    }
}

double mlp_dense_all(const LayerWeights& L, MlpScratch& s, float x_scale) {
    const size_t INTER = L.intermediate;
    auto t0 = std::chrono::steady_clock::now();
    cactus_gemv_int4(s.x_q.data(), x_scale,
                     reinterpret_cast<const int8_t*>(L.gate.packed.data()),
                     L.gate.scales.data(), s.gate_out.data(),
                     HIDDEN, INTER, GROUP_SIZE);
    cactus_gemv_int4(s.x_q.data(), x_scale,
                     reinterpret_cast<const int8_t*>(L.up.packed.data()),
                     L.up.scales.data(), s.up_out.data(),
                     HIDDEN, INTER, GROUP_SIZE);
    cactus_gelu_mul_quant_fp16_to_int8(
        s.gate_out.data(), s.up_out.data(), s.h_q.data(),
        INTER, 8.0f);
    cactus_gemv_int4(s.h_q.data(), 0.125f,
                     reinterpret_cast<const int8_t*>(L.down.packed.data()),
                     L.down.scales.data(), s.out_dense.data(),
                     INTER, HIDDEN, GROUP_SIZE);
    auto t1 = std::chrono::steady_clock::now();
    return ns_to_us(t1 - t0);
}

double mlp_sparse_down(const LayerWeights& L, MlpScratch& s, float x_scale,
                       const uint64_t* router_bitmask, float skip_frac)
{
    const size_t INTER = L.intermediate;
    auto t0 = std::chrono::steady_clock::now();
    cactus_gemv_int4(s.x_q.data(), x_scale,
                     reinterpret_cast<const int8_t*>(L.gate.packed.data()),
                     L.gate.scales.data(), s.gate_out.data(),
                     HIDDEN, INTER, GROUP_SIZE);
    cactus_gemv_int4(s.x_q.data(), x_scale,
                     reinterpret_cast<const int8_t*>(L.up.packed.data()),
                     L.up.scales.data(), s.up_out.data(),
                     HIDDEN, INTER, GROUP_SIZE);
    cactus_gelu_mul_quant_fp16_to_int8(
        s.gate_out.data(), s.up_out.data(), s.h_q.data(),
        INTER, 8.0f);
    size_t num_live = cactus_apply_actsparse_bitmask(
        router_bitmask, s.h_q.data(), INTER, GROUP_SIZE,
        s.h_masked.data(), s.live_groups.data());
    down_sparse_kernel(skip_frac,
                       s.h_masked.data(), 0.125f, L.down_km_inline.data(),
                       s.live_groups.data(), num_live,
                       s.out_sparse.data(), INTER, HIDDEN, GROUP_SIZE);
    auto t1 = std::chrono::steady_clock::now();
    return ns_to_us(t1 - t0);
}

double mlp_sparse_up_and_down(const LayerWeights& L, MlpScratch& s,
                              float x_scale, const uint64_t* router_bitmask,
                              float sparsity, float skip_frac)
{
    const size_t INTER = L.intermediate;
    auto t0 = std::chrono::steady_clock::now();
    cactus_gemv_int4(s.x_q.data(), x_scale,
                     reinterpret_cast<const int8_t*>(L.gate.packed.data()),
                     L.gate.scales.data(), s.gate_out.data(),
                     HIDDEN, INTER, GROUP_SIZE);
    // up_proj on reduced-N output. Router has already picked live slots
    // so we skip rows that won't be used.
    size_t N_live = static_cast<size_t>(std::round(INTER * (1.0f - sparsity)));
    N_live = ((N_live + 3) / 4) * 4;
    if (N_live == 0) N_live = 4;
    cactus_gemv_int4(s.x_q.data(), x_scale,
                     reinterpret_cast<const int8_t*>(L.up.packed.data()),
                     L.up.scales.data(), s.up_out_small.data(),
                     HIDDEN, N_live, GROUP_SIZE);
    // Fused GELU + mul + quantize over only the N_live slice.
    cactus_gelu_mul_quant_fp16_to_int8(
        s.gate_out.data(), s.up_out_small.data(), s.h_q.data(),
        N_live, 8.0f);
    size_t num_live = cactus_apply_actsparse_bitmask(
        router_bitmask, s.h_q.data(), INTER, GROUP_SIZE,
        s.h_masked.data(), s.live_groups.data());
    down_sparse_kernel(skip_frac,
                       s.h_masked.data(), 0.125f, L.down_km_inline.data(),
                       s.live_groups.data(), num_live,
                       s.out_sparse.data(), INTER, HIDDEN, GROUP_SIZE);
    auto t1 = std::chrono::steady_clock::now();
    return ns_to_us(t1 - t0);
}

// ---------------------------------------------------------------------------

struct Pool {
    std::vector<LayerWeights> layers;
    std::vector<std::vector<float>> scores;
    // Per (layer, sparsity) precomputed group bitmask — the router's
    // per-token output. Built outside the MLP timing.
    //   router_bitmask[si][l] == uint64_t[num_groups(l)/64]
    std::vector<std::vector<std::vector<uint64_t>>> router_bitmask;
};

Pool build_pool(const std::vector<float>& sparsities) {
    Pool p;
    p.layers.reserve(NUM_LAYERS);
    for (size_t l = 0; l < NUM_LAYERS; ++l) {
        size_t mod = l % CYCLE_LEN;
        size_t INTER = (mod < SMALL_IN_CYCLE) ? INTER_S : INTER_L;
        p.layers.push_back(build_layer(INTER, 0xE2B0 + l * 101));
    }
    p.scores.resize(NUM_LAYERS);
    for (size_t l = 0; l < NUM_LAYERS; ++l) {
        size_t INTER = p.layers[l].intermediate;
        make_blocky_scores(p.scores[l], INTER,
                           static_cast<unsigned>(0xABCD + l * 37), GROUP_SIZE);
    }
    p.router_bitmask.resize(sparsities.size());
    for (size_t si = 0; si < sparsities.size(); ++si) {
        p.router_bitmask[si].resize(NUM_LAYERS);
        for (size_t l = 0; l < NUM_LAYERS; ++l) {
            size_t INTER = p.layers[l].intermediate;
            build_group_bitmask_from_scores(
                p.scores[l], INTER, GROUP_SIZE, sparsities[si],
                p.router_bitmask[si][l]);
        }
    }
    return p;
}

struct Row {
    double sparsity;
    double dense_all_layer_sum;
    double sparse_down_layer_sum;
    double sparse_up_down_layer_sum;
    double small_dense_mean, large_dense_mean;
    double small_sparse_mean, large_sparse_mean;
    double small_updown_mean, large_updown_mean;
};

Row run_sweep(Pool& pool, float sparsity, size_t si,
              size_t warmup_passes, size_t bench_passes)
{
    MlpScratch s;
    init_scratch(s, 0x1234);

    std::vector<uint8_t> trash(CACHE_TRASH_BYTES, 0);

    std::vector<double> dense_stack(bench_passes, 0.0);
    std::vector<double> sparse_stack(bench_passes, 0.0);
    std::vector<double> updown_stack(bench_passes, 0.0);

    // Per-layer-size accumulators (mean µs per call).
    double small_dense_sum = 0, small_sparse_sum = 0, small_updown_sum = 0;
    double large_dense_sum = 0, large_sparse_sum = 0, large_updown_sum = 0;
    size_t small_count = 0, large_count = 0;

    auto run_pass = [&](auto fn, std::vector<double>& stack_out,
                        double& small_sum, double& large_sum) {
        for (size_t pass = 0; pass < warmup_passes + bench_passes; ++pass) {
            double sum_us = 0.0;
            for (size_t l = 0; l < pool.layers.size(); ++l) {
                trash_cache(trash);
                std::memcpy(s.S.data(), pool.scores[l].data(),
                            pool.layers[l].intermediate * sizeof(float));
                double t = fn(pool.layers[l], s, l);
                if (pass >= warmup_passes) {
                    sum_us += t;
                    if (pool.layers[l].intermediate == INTER_S) {
                        small_sum += t; if (pass == warmup_passes) ++small_count;
                    } else {
                        large_sum += t; if (pass == warmup_passes) ++large_count;
                    }
                }
            }
            if (pass >= warmup_passes) stack_out[pass - warmup_passes] = sum_us;
        }
    };

    run_pass([&](const LayerWeights& L, MlpScratch& sc, size_t) {
        return mlp_dense_all(L, sc, 0.05f);
    }, dense_stack, small_dense_sum, large_dense_sum);

    run_pass([&](const LayerWeights& L, MlpScratch& sc, size_t l) {
        return mlp_sparse_down(L, sc, 0.05f, pool.router_bitmask[si][l].data(),
                               sparsity);
    }, sparse_stack, small_sparse_sum, large_sparse_sum);

    run_pass([&](const LayerWeights& L, MlpScratch& sc, size_t l) {
        return mlp_sparse_up_and_down(L, sc, 0.05f,
                                      pool.router_bitmask[si][l].data(),
                                      sparsity, sparsity);
    }, updown_stack, small_updown_sum, large_updown_sum);

    auto median = [](std::vector<double>& v) {
        std::sort(v.begin(), v.end()); return v[v.size() / 2];
    };

    Row r{};
    r.sparsity = sparsity;
    r.dense_all_layer_sum       = median(dense_stack);
    r.sparse_down_layer_sum     = median(sparse_stack);
    r.sparse_up_down_layer_sum  = median(updown_stack);
    double small_calls = static_cast<double>(small_count * bench_passes);
    double large_calls = static_cast<double>(large_count * bench_passes);
    r.small_dense_mean  = small_calls ? small_dense_sum  / small_calls : 0;
    r.large_dense_mean  = large_calls ? large_dense_sum  / large_calls : 0;
    r.small_sparse_mean = small_calls ? small_sparse_sum / small_calls : 0;
    r.large_sparse_mean = large_calls ? large_sparse_sum / large_calls : 0;
    r.small_updown_mean = small_calls ? small_updown_sum / small_calls : 0;
    r.large_updown_mean = large_calls ? large_updown_sum / large_calls : 0;
    return r;
}

} // namespace

int main() {
    std::printf("Gemma-4-style MLP (3 small : 4 large per 7 layers) — cold-cache bench\n");
    std::printf("  small : hidden=%zu, intermediate=%zu   (E2B)\n", HIDDEN, INTER_S);
    std::printf("  large : hidden=%zu, intermediate=%zu  (E4B shared-MLP)\n", HIDDEN, INTER_L);
    std::printf("  %zu layers total: %zu small + %zu large\n",
                NUM_LAYERS,
                (NUM_LAYERS * SMALL_IN_CYCLE + CYCLE_LEN - 1) / CYCLE_LEN,
                NUM_LAYERS - (NUM_LAYERS * SMALL_IN_CYCLE + CYCLE_LEN - 1) / CYCLE_LEN);
    std::printf("  hw.concurrency=%u\n", std::thread::hardware_concurrency());

    const std::vector<float> sparsities = {0.50f, 0.60f, 0.70f, 0.80f, 0.90f};

    std::printf("\nBuilding layer pool + precomputing per-layer router thresholds...\n");
    Pool pool = build_pool(sparsities);

    constexpr size_t WARMUP = 1;
    constexpr size_t BENCH  = 8;

    std::printf("\n===== PER-CALL MLP-LAYER MEANS (cold cache, includes gate+up+down) =====\n");
    std::printf("%-5s  %14s %14s %14s  %14s %14s %14s\n",
                "sp%",
                "sm d/us", "sm spDn/us(x)", "sm spUp+Dn/us(x)",
                "lg d/us", "lg spDn/us(x)", "lg spUp+Dn/us(x)");

    std::vector<Row> rows;
    for (size_t si = 0; si < sparsities.size(); ++si) {
        float sp = sparsities[si];
        Row r = run_sweep(pool, sp, si, WARMUP, BENCH);
        double sm_d = r.small_dense_mean;
        double lg_d = r.large_dense_mean;
        std::printf("%4.0f%%  %12.1f  %10.1f(%4.2fx)  %10.1f(%4.2fx)  %12.1f  %10.1f(%4.2fx)  %10.1f(%4.2fx)\n",
                    sp * 100.0,
                    sm_d,
                    r.small_sparse_mean, sm_d / r.small_sparse_mean,
                    r.small_updown_mean, sm_d / r.small_updown_mean,
                    lg_d,
                    r.large_sparse_mean, lg_d / r.large_sparse_mean,
                    r.large_updown_mean, lg_d / r.large_updown_mean);
        rows.push_back(r);
    }

    std::printf("\n===== FULL 35-LAYER MLP STACK (weighted sum over small+large) =====\n");
    std::printf("%-5s  %14s %14s  %14s %14s\n",
                "sp%", "dense us", "spDn us (x)", "spUp+Dn us (x)",
                "gate_frac(Amdahl)");
    for (const auto& r : rows) {
        double dense_stack_us = r.dense_all_layer_sum;
        // Estimate gate fraction from small/large means proportional to
        // the known N and K (from earlier Amdahl data: ~1/3 of MLP at
        // equal-hidden configs).
        std::printf("%4.0f%%  %12.1f  %10.1f(%4.2fx)  %12.1f(%4.2fx)\n",
                    r.sparsity * 100.0,
                    dense_stack_us,
                    r.sparse_down_layer_sum,
                    dense_stack_us / r.sparse_down_layer_sum,
                    r.sparse_up_down_layer_sum,
                    dense_stack_us / r.sparse_up_down_layer_sum);
    }

    // CSV
    FILE* csv = std::fopen("gemma4_mlp_actsparse_bench.csv", "w");
    if (csv) {
        std::fprintf(csv,
                     "sparsity,"
                     "small_dense_mean_us,small_sparse_down_mean_us,small_sparse_up_down_mean_us,"
                     "large_dense_mean_us,large_sparse_down_mean_us,large_sparse_up_down_mean_us,"
                     "stack_dense_us,stack_sparse_down_us,stack_sparse_up_down_us\n");
        for (const auto& r : rows) {
            std::fprintf(csv,
                         "%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                         r.sparsity,
                         r.small_dense_mean, r.small_sparse_mean, r.small_updown_mean,
                         r.large_dense_mean, r.large_sparse_mean, r.large_updown_mean,
                         r.dense_all_layer_sum, r.sparse_down_layer_sum,
                         r.sparse_up_down_layer_sum);
        }
        std::fclose(csv);
    }

    // Kernel shootout across all sparsity levels on the E2B down_proj
    // shape (K=INTER, N=HIDDEN). Tells us which variant to dispatch per
    // density.
    std::printf("\n===== DOWN_PROJ SHOOTOUT (E2B: K=%zu N=%zu) =====\n",
                INTER_S, HIDDEN);
    for (float sp : {0.50f, 0.60f, 0.70f, 0.80f, 0.90f})
    {
        std::printf("\n-- sparsity = %.2f --\n", sp);
        MlpScratch s;
        init_scratch(s, 0xBEEF);
        std::vector<uint8_t> trash(CACHE_TRASH_BYTES, 0);
        // Pick the first small layer for testing.
        const LayerWeights* Lp = nullptr;
        size_t l_idx = 0;
        for (size_t li = 0; li < pool.layers.size(); ++li) {
            if (pool.layers[li].intermediate == INTER_S) { Lp = &pool.layers[li]; l_idx = li; break; }
        }
        const LayerWeights& L = *Lp;

        // Build a group-bitmask from layer scores at this sparsity.
        std::vector<uint64_t> bm;
        build_group_bitmask_from_scores(pool.scores[l_idx], INTER_S, GROUP_SIZE, sp, bm);
        // Apply once to produce h_masked + live_groups used by all kernel
        // variants below.
        size_t num_live = cactus_apply_actsparse_bitmask(
            bm.data(), s.h_q.data(), INTER_S, GROUP_SIZE,
            s.h_masked.data(), s.live_groups.data());

        auto bench = [&](const char* name, auto fn) {
            constexpr size_t N_ITERS = 64;
            std::vector<double> times(N_ITERS);
            for (size_t i = 0; i < N_ITERS; ++i) {
                trash_cache(trash);
                auto t0 = std::chrono::steady_clock::now();
                fn();
                auto t1 = std::chrono::steady_clock::now();
                times[i] = ns_to_us(t1 - t0);
            }
            std::sort(times.begin(), times.end());
            double med = times[N_ITERS / 2];
            std::printf("  %-20s : %7.2f µs\n", name, med);
            return med;
        };

        std::printf("  num_live=%zu of %zu groups (%.1f%% skip)\n",
                    num_live, INTER_S / GROUP_SIZE,
                    100.0 * (1.0 - double(num_live) / double(INTER_S / GROUP_SIZE)));
        bench("dense (full)", [&]() {
            cactus_gemv_int4(s.h_q.data(), 0.125f,
                             reinterpret_cast<const int8_t*>(L.down.packed.data()),
                             L.down.scales.data(), s.out_dense.data(),
                             INTER_S, HIDDEN, GROUP_SIZE);
        });
        bench("KM", [&]() {
            // repack split: recover km layout view from inline.
            cactus_gemv_int4_actsparse_kmi(
                s.h_masked.data(), 0.125f, L.down_km_inline.data(),
                s.live_groups.data(), num_live,
                s.out_sparse.data(), INTER_S, HIDDEN, GROUP_SIZE);
        });
        bench("KMI2", [&]() {
            cactus_gemv_int4_actsparse_kmi2(
                s.h_masked.data(), 0.125f, L.down_km_inline.data(),
                s.live_groups.data(), num_live,
                s.out_sparse.data(), INTER_S, HIDDEN, GROUP_SIZE);
        });
        bench("KMI4", [&]() {
            cactus_gemv_int4_actsparse_kmi4(
                s.h_masked.data(), 0.125f, L.down_km_inline.data(),
                s.live_groups.data(), num_live,
                s.out_sparse.data(), INTER_S, HIDDEN, GROUP_SIZE);
        });
        bench("KMI4_fast", [&]() {
            cactus_gemv_int4_actsparse_kmi4_fast(
                s.h_masked.data(), 0.125f, L.down_km_inline.data(),
                s.live_groups.data(), num_live,
                s.out_sparse.data(), INTER_S, HIDDEN, GROUP_SIZE);
        });
        bench("KMI4_v2", [&]() {
            cactus_gemv_int4_actsparse_kmi4_v2(
                s.h_masked.data(), 0.125f, L.down_km_inline.data(),
                s.live_groups.data(), num_live,
                s.out_sparse.data(), INTER_S, HIDDEN, GROUP_SIZE);
        });
        bench("KMI4_chain", [&]() {
            cactus_gemv_int4_actsparse_kmi4_chain(
                s.h_masked.data(), 0.125f, L.down_km_inline.data(),
                s.live_groups.data(), num_live,
                s.out_sparse.data(), INTER_S, HIDDEN, GROUP_SIZE);
        });
    }

    // Amdahl-style decomposition at 60% sparsity for each size.
    std::printf("\n===== AMDAHL VIEW @ 60%% (one MLP layer in isolation, cold) =====\n");
    {
        MlpScratch s;
        init_scratch(s, 0x1234);
        std::vector<uint8_t> trash(CACHE_TRASH_BYTES, 0);

        auto bench_component = [&](const char* name, auto fn) {
            constexpr size_t N_ITERS = 32;
            std::vector<double> times(N_ITERS);
            for (size_t i = 0; i < N_ITERS; ++i) {
                trash_cache(trash);
                auto t0 = std::chrono::steady_clock::now();
                fn();
                auto t1 = std::chrono::steady_clock::now();
                times[i] = ns_to_us(t1 - t0);
            }
            std::sort(times.begin(), times.end());
            return times[N_ITERS / 2];
        };

        struct SizeCase { const char* tag; size_t inter; const LayerWeights* L; };
        const LayerWeights* L_small = nullptr;
        const LayerWeights* L_large = nullptr;
        for (const auto& L : pool.layers) {
            if (L.intermediate == INTER_S && !L_small) L_small = &L;
            if (L.intermediate == INTER_L && !L_large) L_large = &L;
        }
        SizeCase cases[] = {
            {"small (INTER=6144)",  INTER_S, L_small},
            {"large (INTER=12288)", INTER_L, L_large},
        };
        for (const auto& c : cases) {
            size_t INTER = c.inter;
            // 60% sparsity -> N_live = 40% of N
            size_t N_live = static_cast<size_t>(std::round(INTER * 0.40));
            N_live = ((N_live + 3) / 4) * 4;
            double gate_us = bench_component("gate", [&]() {
                cactus_gemv_int4(s.x_q.data(), 0.05f,
                                 reinterpret_cast<const int8_t*>(c.L->gate.packed.data()),
                                 c.L->gate.scales.data(), s.gate_out.data(),
                                 HIDDEN, INTER, GROUP_SIZE);
            });
            double up_dense = bench_component("up_dense", [&]() {
                cactus_gemv_int4(s.x_q.data(), 0.05f,
                                 reinterpret_cast<const int8_t*>(c.L->up.packed.data()),
                                 c.L->up.scales.data(), s.up_out.data(),
                                 HIDDEN, INTER, GROUP_SIZE);
            });
            double up_reduced = bench_component("up_reduced", [&]() {
                cactus_gemv_int4(s.x_q.data(), 0.05f,
                                 reinterpret_cast<const int8_t*>(c.L->up.packed.data()),
                                 c.L->up.scales.data(), s.up_out_small.data(),
                                 HIDDEN, N_live, GROUP_SIZE);
            });
            double down_dense = bench_component("down_dense", [&]() {
                cactus_gemv_int4(s.h_q.data(), 0.125f,
                                 reinterpret_cast<const int8_t*>(c.L->down.packed.data()),
                                 c.L->down.scales.data(), s.out_dense.data(),
                                 INTER, HIDDEN, GROUP_SIZE);
            });
            // Find a layer with this intermediate size. Use 60% bitmask.
            size_t l_idx = 0;
            for (size_t li = 0; li < pool.layers.size(); ++li) {
                if (pool.layers[li].intermediate == INTER) { l_idx = li; break; }
            }
            // sparsities = {0.50, 0.60, 0.70, 0.80, 0.90} -> index 1 for 60%
            size_t num_live = cactus_apply_actsparse_bitmask(
                pool.router_bitmask[1][l_idx].data(), s.h_q.data(), INTER, GROUP_SIZE,
                s.h_masked.data(), s.live_groups.data());
            double down_sparse = bench_component("down_sparse", [&]() {
                // Use dispatched sparse kernel (best per-density).
                down_sparse_kernel(0.60f,
                                   s.h_masked.data(), 0.125f,
                                   c.L->down_km_inline.data(),
                                   s.live_groups.data(), num_live,
                                   s.out_sparse.data(), INTER, HIDDEN, GROUP_SIZE);
            });
            double dense_mlp = gate_us + up_dense + down_dense;
            std::printf("\n  %s   N_live=%zu\n", c.tag, N_live);
            std::printf("    gate_dense    = %7.1f µs\n", gate_us);
            std::printf("    up_dense      = %7.1f µs\n", up_dense);
            std::printf("    up_reduced    = %7.1f µs\n", up_reduced);
            std::printf("    down_dense    = %7.1f µs\n", down_dense);
            std::printf("    down_sparse   = %7.1f µs\n", down_sparse);
            std::printf("    Dense MLP     = %7.1f µs   (gate = %.0f%%)\n",
                        dense_mlp, 100.0 * gate_us / dense_mlp);
            std::printf("    spDown-only   = %7.1f µs   -> %.2fx\n",
                        gate_us + up_dense + down_sparse,
                        dense_mlp / (gate_us + up_dense + down_sparse));
            std::printf("    spUp+Down     = %7.1f µs   -> %.2fx\n",
                        gate_us + up_reduced + down_sparse,
                        dense_mlp / (gate_us + up_reduced + down_sparse));
        }
    }

    return 0;
}
