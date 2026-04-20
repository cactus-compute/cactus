// Sanity speed test: runs the 60%-sparsity activation-sparse MLP pipeline
// over an actual Gemma 4 E2B layer's mmapped INT4 weights from
// weights/gemma-4-e2b-it-it4/ and reports per-call speedup over the
// all-dense baseline on the same layer.
//
// Purpose: confirm that the measured 60% speedup isn't an artifact of the
// synthetic-weight bench — the speedup should survive real trained
// weights too. The router is simulated as a cheap O(num_groups) threshold
// on |silu(gate_out)|, done outside the timed region.
//
// Pass criterion: at 60% sparsity the sparse-down MLP (gate + up dense +
// gelu_mul_quant + mask_apply + KMI2 down) must beat the all-dense MLP
// by >= 1.15x. The target is ~1.25-1.43x per our bench; the 1.15x gate
// is to survive run-to-run noise on any single-process test invocation.

#include "../cactus/kernel/kernel.h"
#include "../cactus/graph/graph.h"

#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr size_t GROUP_SIZE = 32;
constexpr size_t INTERLEAVE = 4;
constexpr size_t HIDDEN     = 1536;
constexpr size_t INTER      = 6144;  // Gemma 4 E2B ffn_intermediate_dim

// Cache trasher so each MLP call starts cold. M4 Pro SLC ≈ 24 MB; 64 MB
// keeps every byte we care about out of cache.
constexpr size_t CACHE_TRASH_BYTES = 64ull * 1024 * 1024;

void trash_cache(std::vector<uint8_t>& trash) {
    for (size_t i = 0; i < trash.size(); i += 64) trash[i] = static_cast<uint8_t>(i);
    asm volatile ("" : : "r"(trash.data()) : "memory");
}

double ns_to_us(std::chrono::steady_clock::duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
}

// Build a per-group bitmask from per-channel |scores|: drop the bottom
// `sparsity` fraction of groups by max|S| per group. Produces the same
// output a post-trained router would.
void build_group_bitmask(const std::vector<float>& S, float sparsity,
                         std::vector<uint64_t>& bitmask)
{
    const size_t num_groups = INTER / GROUP_SIZE;
    std::vector<float> group_max(num_groups, 0.f);
    for (size_t g = 0; g < num_groups; ++g) {
        float m = 0.f;
        for (size_t j = 0; j < GROUP_SIZE; ++j) {
            float v = std::fabs(S[g * GROUP_SIZE + j]);
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

struct Weights {
    std::unique_ptr<GraphFile::MappedFile> gate;
    std::unique_ptr<GraphFile::MappedFile> up;
    std::unique_ptr<GraphFile::MappedFile> down;
    std::vector<uint8_t> down_km_inline;  // one-time repack for KMI kernels.
};

// Best-effort describe a weight file (helps if the layout doesn't match
// what the GEMV kernel expects).
void describe(const char* tag, const GraphFile::MappedFile& m) {
    std::printf("  %-8s  precision=%d  shape=[", tag, static_cast<int>(m.precision()));
    for (size_t i = 0; i < m.shape().size(); ++i)
        std::printf("%zu%s", m.shape()[i], i + 1 < m.shape().size() ? "," : "");
    std::printf("]  group_size=%zu  num_groups=%zu  interleaved=%d  byte_size=%zu\n",
                m.group_size(), m.num_groups(),
                static_cast<int>(m.is_interleaved()), m.byte_size());
}

} // namespace

int main(int argc, char** argv) {
    std::string model_dir = "weights/gemma-4-e2b-it-int4";
    if (const char* env = std::getenv("CACTUS_TEST_GEMMA4_MODEL")) model_dir = env;
    if (argc > 1) model_dir = argv[1];

    // Pick layer 0 for the sanity check. Any non-shared layer in E2B has
    // the same MLP shape so the result is representative.
    std::string gate_path = model_dir + "/layer_0_ffn_gate.weights";
    std::string up_path   = model_dir + "/layer_0_ffn_up.weights";
    std::string down_path = model_dir + "/layer_0_ffn_down.weights";

    std::printf("Gemma 4 E2B MLP 60%%-sparsity sanity speed test\n");
    std::printf("  model dir: %s\n", model_dir.c_str());

    Weights W;
    try {
        W.gate = std::make_unique<GraphFile::MappedFile>(gate_path);
        W.up   = std::make_unique<GraphFile::MappedFile>(up_path);
        W.down = std::make_unique<GraphFile::MappedFile>(down_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  ERROR loading weights: %s\n", e.what());
        std::fprintf(stderr, "  (expected %s/layer_0_ffn_{gate,up,down}.weights)\n",
                     model_dir.c_str());
        return 1;
    }

    std::printf("\nWeight file metadata:\n");
    describe("gate", *W.gate);
    describe("up",   *W.up);
    describe("down", *W.down);

    // Verify the layout matches what cactus_gemv_int4 expects. The shipped
    // int4 weight files are the 4-row-interleaved format the baseline
    // kernel consumes directly.
    if (W.gate->precision() != Precision::INT4 ||
        W.up->precision()   != Precision::INT4 ||
        W.down->precision() != Precision::INT4 ||
        W.gate->group_size() != GROUP_SIZE ||
        W.up->group_size()   != GROUP_SIZE ||
        W.down->group_size() != GROUP_SIZE)
    {
        std::fprintf(stderr, "  ERROR: unexpected precision or group_size\n");
        return 1;
    }
    if (!W.gate->is_interleaved() || !W.up->is_interleaved() || !W.down->is_interleaved()) {
        std::fprintf(stderr, "  ERROR: weights are not in the 4-row interleaved "
                             "layout cactus_gemv_int4 expects\n");
        return 1;
    }

    const int8_t* W_gate = reinterpret_cast<const int8_t*>(W.gate->data());
    const int8_t* W_up   = reinterpret_cast<const int8_t*>(W.up->data());
    const int8_t* W_down = reinterpret_cast<const int8_t*>(W.down->data());
    const __fp16* S_gate = reinterpret_cast<const __fp16*>(W.gate->scales_data());
    const __fp16* S_up   = reinterpret_cast<const __fp16*>(W.up->scales_data());
    const __fp16* S_down = reinterpret_cast<const __fp16*>(W.down->scales_data());

    // One-time K-major inline repack of W_down (does NOT depend on the
    // per-token router; pure data re-layout).
    const size_t down_num_groups = INTER / GROUP_SIZE;
    const size_t down_N_blocks   = (HIDDEN + INTERLEAVE - 1) / INTERLEAVE;
    W.down_km_inline.resize(down_num_groups * down_N_blocks * 72);
    cactus_repack_int4_kmajor_inline(
        W_down, S_down, W.down_km_inline.data(),
        /*K=*/INTER, /*N=*/HIDDEN, GROUP_SIZE);

    // Simulated quantized input x. Matches what a real residual stream
    // looks like after RMSNorm + quantize.
    std::vector<int8_t> x_q(HIDDEN);
    std::mt19937 rng(0xE2B0);
    std::uniform_int_distribution<int> ad(-120, 120);
    for (auto& v : x_q) v = static_cast<int8_t>(ad(rng));
    float x_scale = 0.05f;

    // Output buffers.
    std::vector<__fp16> gate_out(INTER);
    std::vector<__fp16> up_out(INTER);
    std::vector<__fp16> up_out_small(INTER);
    std::vector<int8_t> h_q(INTER);
    std::vector<int8_t> h_masked(INTER);
    std::vector<__fp16> out_dense(HIDDEN);
    std::vector<__fp16> out_sparse(HIDDEN);
    std::vector<uint16_t> live_groups(down_num_groups);
    std::vector<uint64_t> router_bitmask((down_num_groups + 63) / 64);

    // Build the router bitmask ONCE outside the timed region. In a real
    // post-trained router this comes from a tiny head on the gate
    // output; we get a representative pattern by running gate once on x
    // and thresholding |gelu(gate)|.
    cactus_gemv_int4(x_q.data(), x_scale,
                     W_gate, S_gate, gate_out.data(),
                     HIDDEN, INTER, GROUP_SIZE);
    std::vector<float> score(INTER);
    for (size_t i = 0; i < INTER; ++i) {
        float g = static_cast<float>(gate_out[i]);
        float gelu = 0.5f * g * (1.0f + std::tanh(0.7978845608f * (g + 0.044715f * g * g * g)));
        score[i] = std::fabs(gelu);
    }
    constexpr float SPARSITY = 0.60f;
    build_group_bitmask(score, SPARSITY, router_bitmask);

    const size_t N_live = [&]() {
        size_t n = static_cast<size_t>(std::round(INTER * (1.0f - SPARSITY)));
        n = ((n + 3) / 4) * 4;
        return std::max<size_t>(n, 4);
    }();

    auto dense_mlp = [&]() {
        cactus_gemv_int4(x_q.data(), x_scale,
                         W_gate, S_gate, gate_out.data(),
                         HIDDEN, INTER, GROUP_SIZE);
        cactus_gemv_int4(x_q.data(), x_scale,
                         W_up, S_up, up_out.data(),
                         HIDDEN, INTER, GROUP_SIZE);
        cactus_gelu_mul_quant_fp16_to_int8(
            gate_out.data(), up_out.data(), h_q.data(), INTER, 8.0f);
        cactus_gemv_int4(h_q.data(), 0.125f,
                         W_down, S_down, out_dense.data(),
                         INTER, HIDDEN, GROUP_SIZE);
    };

    auto sparse_down_mlp = [&]() {
        cactus_gemv_int4(x_q.data(), x_scale,
                         W_gate, S_gate, gate_out.data(),
                         HIDDEN, INTER, GROUP_SIZE);
        cactus_gemv_int4(x_q.data(), x_scale,
                         W_up, S_up, up_out.data(),
                         HIDDEN, INTER, GROUP_SIZE);
        cactus_gelu_mul_quant_fp16_to_int8(
            gate_out.data(), up_out.data(), h_q.data(), INTER, 8.0f);
        size_t num_live = cactus_apply_actsparse_bitmask(
            router_bitmask.data(), h_q.data(), INTER, GROUP_SIZE,
            h_masked.data(), live_groups.data());
        cactus_gemv_int4_actsparse_kmi2(
            h_masked.data(), 0.125f, W.down_km_inline.data(),
            live_groups.data(), num_live,
            out_sparse.data(), INTER, HIDDEN, GROUP_SIZE);
    };

    auto sparse_updown_mlp = [&]() {
        cactus_gemv_int4(x_q.data(), x_scale,
                         W_gate, S_gate, gate_out.data(),
                         HIDDEN, INTER, GROUP_SIZE);
        cactus_gemv_int4(x_q.data(), x_scale,
                         W_up, S_up, up_out_small.data(),
                         HIDDEN, N_live, GROUP_SIZE);
        cactus_gelu_mul_quant_fp16_to_int8(
            gate_out.data(), up_out_small.data(), h_q.data(), N_live, 8.0f);
        size_t num_live = cactus_apply_actsparse_bitmask(
            router_bitmask.data(), h_q.data(), INTER, GROUP_SIZE,
            h_masked.data(), live_groups.data());
        cactus_gemv_int4_actsparse_kmi2(
            h_masked.data(), 0.125f, W.down_km_inline.data(),
            live_groups.data(), num_live,
            out_sparse.data(), INTER, HIDDEN, GROUP_SIZE);
    };

    std::vector<uint8_t> trash(CACHE_TRASH_BYTES, 0);

    auto bench = [&](const char* name, auto fn) {
        constexpr size_t WARMUP = 2;
        constexpr size_t ITERS  = 64;
        for (size_t i = 0; i < WARMUP; ++i) { trash_cache(trash); fn(); }
        std::vector<double> t(ITERS);
        for (size_t i = 0; i < ITERS; ++i) {
            trash_cache(trash);
            auto t0 = std::chrono::steady_clock::now();
            fn();
            auto t1 = std::chrono::steady_clock::now();
            t[i] = ns_to_us(t1 - t0);
        }
        std::sort(t.begin(), t.end());
        double med = t[ITERS / 2];
        double p95 = t[(ITERS * 95) / 100];
        std::printf("  %-16s median=%7.1f µs   p95=%7.1f µs\n", name, med, p95);
        return med;
    };

    // Count live groups for reporting.
    size_t skipped = 0;
    for (size_t q = 0; q < router_bitmask.size(); ++q)
        skipped += __builtin_popcountll(~router_bitmask[q]);
    skipped = std::min(skipped, down_num_groups);
    const size_t kept = down_num_groups - skipped;

    std::printf("\nRouter bitmask built from layer 0 gate output:\n");
    std::printf("  target sparsity = %.0f%%   num_groups = %zu  "
                "live groups = %zu (%.1f%% group-skip)   up N_live = %zu\n",
                SPARSITY * 100.0, down_num_groups, kept,
                100.0 * double(skipped) / double(down_num_groups), N_live);

    std::printf("\nCold-cache MLP timings (layer 0 of %s):\n", model_dir.c_str());
    double us_dense        = bench("dense_all",     dense_mlp);
    double us_sparse_down  = bench("sparse_down",   sparse_down_mlp);
    double us_sparse_updn  = bench("sparse_up+down",sparse_updown_mlp);

    double sp_down = us_dense / us_sparse_down;
    double sp_updn = us_dense / us_sparse_updn;

    std::printf("\n=== RESULTS ===\n");
    std::printf("  sparse_down  speedup = %5.2fx\n", sp_down);
    std::printf("  sparse_up+dn speedup = %5.2fx\n", sp_updn);

    constexpr double PASS_THRESH = 1.15;
    bool ok_down = sp_down >= PASS_THRESH;
    bool ok_updn = sp_updn >= PASS_THRESH;

    if (ok_down && ok_updn) {
        std::printf("  PASS: both paths >= %.2fx faster than all-dense.\n", PASS_THRESH);
        return 0;
    } else {
        std::printf("  FAIL: speedup below %.2fx threshold.\n", PASS_THRESH);
        if (!ok_down) std::printf("    sparse_down     %5.2fx\n", sp_down);
        if (!ok_updn) std::printf("    sparse_up+down  %5.2fx\n", sp_updn);
        return 1;
    }
}
