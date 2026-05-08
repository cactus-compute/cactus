// bench_sparse_actsparse.cpp — Cold-cache 35-layer cycling GEMV benchmark
//
// Replicates the structure of test_bench_gemma4_mlp_actsparse.cpp (sparse-matmul
// branch) for GGML Q4_0 and PowerInfer-2 scatter so results are directly
// comparable: 35 distinct random weight matrices are allocated (one per
// Gemma-4 E2B decoder layer) and each timed iteration cycles through all 35
// sequentially.  By the time a layer's weights are revisited they have been
// evicted from L2/L3, giving cold-cache bandwidth measurements that match a
// real single-token decode pass.
//
// Dimensions: down-projection only — Gemma-4 E2B
//   D_FFN=6144  (K — input channels from intermediate activation)
//   D_MODEL=1536 (N — output channels / hidden dim)
//   batch=1 token
//
// Backends:
//   (1) ggml_q4_dense       — vec_dot on all D_FFN rows (no sparsity; baseline)
//   (2) powerinfer_scatter  — vec_dot called once per live row, scatter into output
//                             (PowerInfer-2 fused_sparse_ffn.cpp strategy)
//
// Reference: Cactus KMI4_fast values are read from the previously measured
//   /Users/noahcylich/Documents/Desert/matryoshka-distil/logs/bench_cactus_sparse_up_real.txt
//   (DOWN_PROJ SHOOTOUT section, E2B K=6144 N=1536)
//
// Timing protocol (mirrors test_bench_gemma4_mlp_actsparse):
//   Warmup  : 5 full passes × 35 layers = 175 GEMVs
//   Timed   : 50 full passes × 35 layers = 1750 GEMVs
//   Per-layer latency = total_elapsed_us / (50 × 35)
//   Median over the 50 per-pass observations is also reported.
//
// Sparsity levels: 0.30, 0.50, 0.60
// Each layer gets its own randomly generated livelist (same seed → reproducible).

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// ── Gemma-4 E2B down-projection dimensions ─────────────────────────────────
static constexpr size_t D_FFN   = 6144;   // K — rows in W_down (intermediate dim)
static constexpr size_t D_MODEL = 1536;   // N — cols in W_down (hidden dim)

// ── Benchmark parameters (matching Cactus cold-cache bench) ─────────────────
static constexpr size_t N_LAYERS = 35;
static constexpr int    N_WARMUP = 5;    // full 35-layer passes
static constexpr int    N_PASSES = 50;   // timed full passes

// ── Block granularity for sparsity mask (matches Cactus GROUP_SIZE=32) ──────
static constexpr size_t BLOCK = 32;

// ── Timer ───────────────────────────────────────────────────────────────────
static double now_us() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double median_val(std::vector<double> v) {  // by value so we can sort
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ── Weight storage for one layer ─────────────────────────────────────────────
// Q4_0 row-major: D_FFN rows, each D_MODEL elements.
// The activation is Q8_0, D_MODEL elements.
struct LayerWeights {
    size_t row_stride_q40;  // bytes per Q4_0 row (D_MODEL elements)
    size_t row_stride_q80;  // bytes per Q8_0 row (D_MODEL elements)
    std::vector<uint8_t> q40;     // Q4_0 weight matrix [D_FFN × D_MODEL]
    std::vector<uint8_t> q80_act; // Q8_0 activation [1 × D_MODEL]
};

static LayerWeights build_layer(size_t seed) {
    LayerWeights lw;
    lw.row_stride_q40 = ggml_row_size(GGML_TYPE_Q4_0, static_cast<int64_t>(D_MODEL));
    lw.row_stride_q80 = ggml_row_size(GGML_TYPE_Q8_0, static_cast<int64_t>(D_MODEL));

    const auto* q40_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_0);
    const auto* q80_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);

    ggml_from_float_t q40_from_float = q40_traits->from_float;
    if (!q40_from_float)
        q40_from_float = ggml_get_type_traits(GGML_TYPE_Q4_0)->from_float_ref;
    ggml_from_float_t q80_from_float = q80_traits->from_float;
    if (!q80_from_float)
        q80_from_float = ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref;

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    // Weights: D_FFN rows
    lw.q40.resize(lw.row_stride_q40 * D_FFN);
    std::vector<float> row_fp32(D_MODEL);
    for (size_t r = 0; r < D_FFN; ++r) {
        for (auto& v : row_fp32) v = ud(rng);
        q40_from_float(row_fp32.data(),
                       lw.q40.data() + r * lw.row_stride_q40,
                       static_cast<int64_t>(D_MODEL));
    }

    // Activation: 1 row (same vector reused for all layers — residual stream)
    lw.q80_act.resize(lw.row_stride_q80);
    std::vector<float> act_fp32(D_MODEL);
    for (auto& v : act_fp32) v = ud(rng);
    q80_from_float(act_fp32.data(), lw.q80_act.data(), static_cast<int64_t>(D_MODEL));

    return lw;
}

// ── Livelist for one layer ───────────────────────────────────────────────────
// Block-structured: whole BLOCK=32-element groups are either all live or all dead.
// Returns sorted vector of live row indices.
static std::vector<uint32_t> build_livelist(float sparsity, size_t seed) {
    const size_t num_blocks = D_FFN / BLOCK;
    const size_t drop_blocks =
        static_cast<size_t>(std::round(sparsity * static_cast<float>(num_blocks)));

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::vector<size_t> order(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);

    std::vector<bool> block_dead(num_blocks, false);
    for (size_t b = 0; b < drop_blocks; ++b) block_dead[order[b]] = true;

    std::vector<uint32_t> live;
    live.reserve(D_FFN);
    for (size_t blk = 0; blk < num_blocks; ++blk) {
        if (!block_dead[blk]) {
            for (size_t j = 0; j < BLOCK; ++j)
                live.push_back(static_cast<uint32_t>(blk * BLOCK + j));
        }
    }
    return live;
}

// ── Reusable scratch buffers ─────────────────────────────────────────────────
struct Scratch {
    std::vector<float> out_dense;   // [D_FFN] for dense output
    std::vector<float> out_scatter; // [D_FFN] for scatter output

    Scratch() : out_dense(D_FFN, 0.f), out_scatter(D_FFN, 0.f) {}
};

// ── Benchmark: ggml_q4_dense ─────────────────────────────────────────────────
// One full GEMV over all D_FFN rows of a single layer.
// Returns elapsed µs.
static inline double run_dense_one_layer(const LayerWeights& lw,
                                          ggml_vec_dot_t vec_dot,
                                          Scratch& sc) {
    const int K = static_cast<int>(D_MODEL);
    double t0 = now_us();
    for (size_t r = 0; r < D_FFN; ++r) {
        vec_dot(K, &sc.out_dense[r], 0,
                lw.q40.data() + r * lw.row_stride_q40, 0,
                lw.q80_act.data(), 0, 1);
    }
    return now_us() - t0;
}

// ── Benchmark: powerinfer_scatter ────────────────────────────────────────────
// One vec_dot call per live row; scatter into output buffer at live index.
// Returns elapsed µs.
static inline double run_scatter_one_layer(const LayerWeights& lw,
                                            const std::vector<uint32_t>& live,
                                            ggml_vec_dot_t vec_dot,
                                            Scratch& sc) {
    const int K = static_cast<int>(D_MODEL);
    double t0 = now_us();
    for (uint32_t r : live) {
        vec_dot(K, &sc.out_scatter[r], 0,
                lw.q40.data() + r * lw.row_stride_q40, 0,
                lw.q80_act.data(), 0, 1);
    }
    return now_us() - t0;
}

// ── Full 35-layer cycling pass ───────────────────────────────────────────────
// Returns per-layer µs for this pass (total_elapsed / N_LAYERS).
static double run_one_pass_dense(
    const std::vector<LayerWeights>& layers,
    ggml_vec_dot_t vec_dot,
    Scratch& sc)
{
    double total = 0.0;
    for (size_t l = 0; l < N_LAYERS; ++l)
        total += run_dense_one_layer(layers[l], vec_dot, sc);
    return total / static_cast<double>(N_LAYERS);
}

static double run_one_pass_scatter(
    const std::vector<LayerWeights>& layers,
    const std::vector<std::vector<uint32_t>>& livelists,
    ggml_vec_dot_t vec_dot,
    Scratch& sc)
{
    double total = 0.0;
    for (size_t l = 0; l < N_LAYERS; ++l)
        total += run_scatter_one_layer(layers[l], livelists[l], vec_dot, sc);
    return total / static_cast<double>(N_LAYERS);
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main() {
    std::printf(
        "Cold-cache 35-layer cycling bench — Gemma-4 E2B down_proj\n"
        "  D_FFN=%zu (K), D_MODEL=%zu (N), batch=1\n"
        "  GGML Q4_0 weights + Q8_0 activation\n"
        "  %zu layers × %d warmup passes + %d timed passes (= %d GEMVs timed)\n"
        "  Per-layer µs = median of per-pass averages\n\n",
        D_FFN, D_MODEL, N_LAYERS, N_WARMUP, N_PASSES,
        N_PASSES * static_cast<int>(N_LAYERS));

    // ── Allocate 35 distinct weight matrices ──────────────────────────────
    std::printf("Allocating %zu distinct Q4_0 weight matrices "
                "(%.1f MB each, %.1f MB total)...\n",
                N_LAYERS,
                static_cast<double>(ggml_row_size(GGML_TYPE_Q4_0, D_MODEL) * D_FFN)
                    / (1024.0 * 1024.0),
                static_cast<double>(ggml_row_size(GGML_TYPE_Q4_0, D_MODEL) * D_FFN
                    * N_LAYERS) / (1024.0 * 1024.0));

    std::vector<LayerWeights> layers;
    layers.reserve(N_LAYERS);
    for (size_t l = 0; l < N_LAYERS; ++l)
        layers.push_back(build_layer(0xDEAD0000 + l * 1337));

    std::printf("Done.\n\n");

    // ── Grab vec_dot function pointer ─────────────────────────────────────
    const auto* q40_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_0);
    ggml_vec_dot_t vec_dot = q40_traits->vec_dot;

    // ── Reference values from measured Cactus log ─────────────────────────
    // Source: logs/bench_cactus_sparse_up_real.txt — DOWN_PROJ SHOOTOUT section
    //   E2B K=6144, N=1536, cold-cache 35-layer cycling bench
    //   KMI4_fast µs: sp=0.30 → 93.58, sp=0.50 → 66.17, sp=0.60 → 53.04
    //   KMI4_v2  µs: sp=0.30 → 83.67, sp=0.50 → 61.62, sp=0.60 → 54.21
    //   dense    µs: sp=0.30 → 112.38 (all-dense baseline from shootout)
    // Using KMI4_fast as the conservative Cactus reference.
    const std::vector<float> sparsities     = {0.30f, 0.50f, 0.60f};
    const std::vector<double> cactus_dense_ref  = {112.38, 113.50, 109.38};
    const std::vector<double> cactus_kmi4f_ref  = {93.58, 66.17, 53.04};
    const std::vector<double> cactus_kmi4v2_ref = {83.67, 61.62, 54.21};

    Scratch sc;

    // ── One sparsity level at a time ──────────────────────────────────────
    std::printf("%-6s  %-8s  %-22s  %-24s  %-22s  %-22s\n",
                "sp%", "K_live",
                "ggml_q4_dense(µs)",
                "powerinfer_scatter(µs)",
                "cactus_dense(µs)[ref]",
                "cactus_KMI4f(µs)[ref]");
    std::printf("%s\n", std::string(120, '-').c_str());

    // Collect all results to print the final formatted table
    struct Result {
        float sparsity;
        size_t k_live;
        double dense_med, scatter_med;
    };
    std::vector<Result> results;

    for (size_t si = 0; si < sparsities.size(); ++si) {
        float sp = sparsities[si];

        // Build per-layer livelists (different seed per layer, same per run)
        std::vector<std::vector<uint32_t>> livelists;
        livelists.reserve(N_LAYERS);
        for (size_t l = 0; l < N_LAYERS; ++l)
            livelists.push_back(build_livelist(sp, 0xCAFE0000 + si * 10000 + l * 31));

        // K_live is approximately the same for all layers (block-aligned)
        size_t k_live = livelists[0].size();

        // ── Warmup: 5 full 35-layer passes ─────────────────────────────
        for (int w = 0; w < N_WARMUP; ++w) {
            run_one_pass_dense(layers, vec_dot, sc);
            run_one_pass_scatter(layers, livelists, vec_dot, sc);
        }

        // ── Timed: 50 full 35-layer passes ─────────────────────────────
        std::vector<double> dense_pass_us(N_PASSES);
        std::vector<double> scatter_pass_us(N_PASSES);

        for (int p = 0; p < N_PASSES; ++p) {
            dense_pass_us[p]   = run_one_pass_dense(layers, vec_dot, sc);
            scatter_pass_us[p] = run_one_pass_scatter(layers, livelists, vec_dot, sc);
        }

        double dense_med   = median_val(dense_pass_us);
        double scatter_med = median_val(scatter_pass_us);

        results.push_back({sp, k_live, dense_med, scatter_med});

        double cactus_dense = cactus_dense_ref[si];
        double cactus_kmi4f = cactus_kmi4f_ref[si];

        std::printf("  %3.0f%%  %6zu/%6zu  %16.1f (1.00x)  %18.1f (%4.2fx)  "
                    "%15.2f [ref]  %15.2f [ref]\n",
                    sp * 100.0f, k_live, D_FFN,
                    dense_med,
                    scatter_med, dense_med / scatter_med,
                    cactus_dense,
                    cactus_kmi4f);
    }

    // ── Detailed formatted table matching goal spec ───────────────────────
    std::printf("\n\n");
    std::printf("Cold-cache 35-layer cycling bench — Gemma-4 E2B down_proj "
                "(D_FFN=6144, D_model=1536)\n");
    std::printf("35 layers × synthetic random Q4_0 weights, "
                "%d timed cycles, per-layer median µs\n\n", N_PASSES);

    for (size_t si = 0; si < sparsities.size(); ++si) {
        const auto& r = results[si];
        double dense_med   = r.dense_med;
        double scatter_med = r.scatter_med;
        double cactus_kmi4f = cactus_kmi4f_ref[si];

        std::printf("sp=%.2f (K_live=%zu):\n", r.sparsity, r.k_live);
        std::printf("  ggml_q4_dense        = %6.1f µs/layer   (1.00x)\n",
                    dense_med);
        std::printf("  powerinfer_scatter   = %6.1f µs/layer   (%.2fx)\n",
                    scatter_med, dense_med / scatter_med);
        std::printf("  cactus_KMI4_fast     = ~%5.1f µs/layer   (%.2fx) [ref from bench log]\n",
                    cactus_kmi4f, dense_med / cactus_kmi4f);
        std::printf("  cactus_KMI4_v2       = ~%5.1f µs/layer   (%.2fx) [ref from bench log]\n",
                    cactus_kmi4v2_ref[si], dense_med / cactus_kmi4v2_ref[si]);
        std::printf("\n");
    }

    // ── Interpretation notes ──────────────────────────────────────────────
    std::printf("--- Notes ---\n");
    std::printf(
        "  Cold-cache design: 35 distinct weight matrices (%zu rows × %zu cols each).\n"
        "  Total weight data per pass: 35 × %.1f MB = %.1f MB — exceeds typical L3.\n"
        "  Each layer's weights must be fetched from DRAM per pass.\n\n",
        D_FFN, D_MODEL,
        static_cast<double>(ggml_row_size(GGML_TYPE_Q4_0, D_MODEL) * D_FFN)
            / (1024.0 * 1024.0),
        static_cast<double>(ggml_row_size(GGML_TYPE_Q4_0, D_MODEL) * D_FFN * N_LAYERS)
            / (1024.0 * 1024.0));

    std::printf(
        "  powerinfer_scatter vs ggml_q4_dense:\n"
        "    Both touch only the live rows of the weight matrix (K_live rows).\n"
        "    Scatter still has per-row function call overhead vs dense loop unroll.\n"
        "    Cache-miss pressure is proportional to K_live for both.\n\n");

    std::printf(
        "  Cactus KMI4_fast advantage (from bench log):\n"
        "    - K-major interleaved INT4 weight layout (4-row interleave, 64-byte aligned)\n"
        "    - Prefetch-optimized inner loop with NEON sdot\n"
        "    - Group-level bitmask → contiguous live_group list → no per-row scatter\n"
        "    - At sp=0.50: Cactus %.1f µs vs GGML dense %.1f µs vs scatter %.1f µs\n",
        cactus_kmi4f_ref[1], results[1].dense_med, results[1].scatter_med);

    std::printf(
        "\n  [PowerInfer-2 note] aarch64/gemv.cpp uses the same interleaved Q4_0x4\n"
        "  dense GEMV as GGML. The sparse path (fused_sparse_ffn.cpp) is neuron-level\n"
        "  scatter-gather: vec_dot_q4_0_q8_0 called once per live row — no standalone\n"
        "  sparse NEON kernel. powerinfer_scatter above replicates that exactly.\n");

    return 0;
}
