// Correctness gate for activation-sparse INT4 × INT8 GEMV kernels.
//
// Compares each candidate kernel against a scalar reference that does:
//   for k in [0, K): A_masked[k] = live[k] ? A[k] : 0
//   for n in [0, N): for each group g: use int4 weights + fp16 scales like
//                   the dense kernel does, with A_masked in place of A.
//
// The candidates should be bit-identical to the dense-baseline call on
// A_masked (since they only skip all-zero groups), so we also diff against
// cactus_gemv_int4(A_masked, ...). Failures on either check reject the
// kernel from the benchmark.

#include "../cactus/kernel/kernel.h"

#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr size_t GROUP_SIZE = 32;
constexpr size_t INTERLEAVE = 4;

struct DenseWeights {
    std::vector<uint8_t> packed;
    std::vector<__fp16> scales;
    std::vector<int8_t> unpacked; // row-major NxK, for reference
};

void pack_int4_32(const int8_t* src32, uint8_t* dst16) {
    for (size_t i = 0; i < 16; ++i) {
        uint8_t lo = static_cast<uint8_t>(src32[i]) & 0x0F;
        uint8_t hi = static_cast<uint8_t>(src32[16 + i]) & 0x0F;
        dst16[i] = lo | (hi << 4);
    }
}

DenseWeights build_weights(size_t N, size_t K, unsigned seed) {
    DenseWeights w;
    const size_t num_groups = K / GROUP_SIZE;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    w.unpacked.resize(N * K);
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
                w.unpacked[n * K + g * GROUP_SIZE + i] = static_cast<int8_t>(qv);
            }
        }
    }

    const size_t N_blocks = (N + INTERLEAVE - 1) / INTERLEAVE;
    const size_t N_padded = N_blocks * INTERLEAVE;
    const size_t groups_per_row = num_groups;

    w.packed.assign(N_padded * K / 2, 0);
    w.scales.assign(N_blocks * groups_per_row * INTERLEAVE,
                    static_cast<__fp16>(1e-6f));

    // Interleaved buffer for the whole N_padded×K space, then pack 32-int-chunks.
    // Per 4 K-lanes (kg), interleave across 4 rows (bi), 4 K-lanes (ki):
    //   interleaved[((nb * K/4 + kg) * INTERLEAVE + bi) * 4 + ki] = W(nb*INTERLEAVE+bi, kg*4+ki)
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
    const size_t total = N_padded * K;
    for (size_t i = 0; i < total; i += 32) {
        pack_int4_32(interleaved.data() + i, w.packed.data() + i / 2);
    }

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

void scalar_reference(
    const int8_t* A_masked, float A_scale,
    const int8_t* B_unpacked, // N×K row major
    const float* B_scales_row_major, // N × num_groups
    __fp16* C,
    size_t K, size_t N)
{
    const size_t num_groups = K / GROUP_SIZE;
    for (size_t n = 0; n < N; ++n) {
        double sum = 0.0;
        for (size_t g = 0; g < num_groups; ++g) {
            int32_t acc = 0;
            for (size_t i = 0; i < GROUP_SIZE; ++i) {
                acc += int32_t(A_masked[g * GROUP_SIZE + i]) *
                       int32_t(B_unpacked[n * K + g * GROUP_SIZE + i]);
            }
            sum += double(acc) * double(B_scales_row_major[n * num_groups + g]);
        }
        C[n] = static_cast<__fp16>(sum * double(A_scale));
    }
}

bool check_close(const char* tag, const __fp16* a, const __fp16* b, size_t N,
                 float abs_tol, float rel_tol)
{
    float max_abs = 0.f, max_rel = 0.f;
    size_t bad = 0;
    for (size_t i = 0; i < N; ++i) {
        float va = static_cast<float>(a[i]);
        float vb = static_cast<float>(b[i]);
        float ad = std::fabs(va - vb);
        float rd = ad / std::max(1e-6f, std::max(std::fabs(va), std::fabs(vb)));
        max_abs = std::max(max_abs, ad);
        max_rel = std::max(max_rel, rd);
        if (ad > abs_tol && rd > rel_tol) {
            if (bad < 4)
                std::fprintf(stderr,
                             "    [%s] mismatch idx=%zu: got=%.6f exp=%.6f\n",
                             tag, i, va, vb);
            ++bad;
        }
    }
    std::fprintf(stderr,
                 "    [%s] max_abs=%.4g max_rel=%.4g mismatches=%zu\n",
                 tag, max_abs, max_rel, bad);
    return bad == 0;
}

void make_iid_scores(std::vector<float>& S, size_t K, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    S.resize(K);
    for (auto& v : S) v = std::fabs(nd(rng));
}

void make_blocky_scores(std::vector<float>& S, size_t K, unsigned seed, size_t block_size) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    S.assign(K, 0.f);
    // Each block shares a coarse magnitude drawn from a heavy-tailed dist.
    const size_t nb = (K + block_size - 1) / block_size;
    for (size_t b = 0; b < nb; ++b) {
        float coarse = -std::log(std::max(1e-6f, u(rng)));  // Exp(1)
        size_t k0 = b * block_size;
        size_t k1 = std::min(k0 + block_size, K);
        for (size_t k = k0; k < k1; ++k) {
            float jitter = 0.1f * u(rng);
            S[k] = coarse + jitter;
        }
    }
}

bool run_one(size_t K, size_t N, float sparsity, const char* pattern, bool blocky) {
    std::fprintf(stderr, "K=%zu N=%zu sparsity=%.2f pattern=%s\n",
                 K, N, sparsity, pattern);
    auto w = build_weights(N, K, /*seed*/42 + static_cast<unsigned>(K + N * 7));
    std::mt19937 rng(1234 + static_cast<unsigned>(K * 31 + N));
    std::uniform_int_distribution<int> ad(-120, 120);
    std::vector<int8_t> A(K);
    for (auto& v : A) v = static_cast<int8_t>(ad(rng));
    float A_scale = 0.05f;

    std::vector<float> S;
    if (blocky) make_blocky_scores(S, K, 77 + static_cast<unsigned>(K), /*block*/GROUP_SIZE * 4);
    else make_iid_scores(S, K, 77 + static_cast<unsigned>(K));

    // Build mask via the kernel helper to exercise it too.
    std::vector<int8_t> A_masked(K);
    const size_t num_groups = K / GROUP_SIZE;
    std::vector<uint64_t> bitmask((num_groups + 63) / 64);
    std::vector<uint16_t> live_groups(num_groups);
    size_t num_live = cactus_build_actsparse_mask_f32(
        S.data(), A.data(), K, sparsity, GROUP_SIZE,
        A_masked.data(), bitmask.data(), live_groups.data());

    // Reference: scalar matmul with A_masked against row-major unpacked B.
    std::vector<float> B_scales_rm(N * num_groups);
    for (size_t n = 0; n < N; ++n) {
        size_t nb = n / INTERLEAVE, bi = n % INTERLEAVE;
        for (size_t g = 0; g < num_groups; ++g) {
            B_scales_rm[n * num_groups + g] =
                static_cast<float>(w.scales[(nb * num_groups + g) * INTERLEAVE + bi]);
        }
    }
    std::vector<__fp16> C_ref(N);
    scalar_reference(A_masked.data(), A_scale,
                     w.unpacked.data(), B_scales_rm.data(),
                     C_ref.data(), K, N);

    std::vector<__fp16> C_dense(N, __fp16(0));
    cactus_gemv_int4(A_masked.data(), A_scale,
                     reinterpret_cast<const int8_t*>(w.packed.data()),
                     w.scales.data(), C_dense.data(), K, N, GROUP_SIZE);

    bool ok_dense = check_close("dense-vs-ref", C_dense.data(), C_ref.data(),
                                N, /*abs*/5e-2f, /*rel*/1e-2f);

    std::vector<__fp16> C_azero(N), C_mask(N), C_live(N);
    std::vector<__fp16> C_mask2(N), C_livepf(N), C_km(N);
    cactus_gemv_int4_actsparse_azero(
        A_masked.data(), A_scale,
        reinterpret_cast<const int8_t*>(w.packed.data()),
        w.scales.data(), C_azero.data(), K, N, GROUP_SIZE);
    cactus_gemv_int4_actsparse_bitmask(
        A_masked.data(), A_scale,
        reinterpret_cast<const int8_t*>(w.packed.data()),
        w.scales.data(), bitmask.data(),
        C_mask.data(), K, N, GROUP_SIZE);
    cactus_gemv_int4_actsparse_livelist(
        A_masked.data(), A_scale,
        reinterpret_cast<const int8_t*>(w.packed.data()),
        w.scales.data(), live_groups.data(), num_live,
        C_live.data(), K, N, GROUP_SIZE);
    cactus_gemv_int4_actsparse_bitmask_2nb(
        A_masked.data(), A_scale,
        reinterpret_cast<const int8_t*>(w.packed.data()),
        w.scales.data(), bitmask.data(),
        C_mask2.data(), K, N, GROUP_SIZE);
    cactus_gemv_int4_actsparse_livelist_pf(
        A_masked.data(), A_scale,
        reinterpret_cast<const int8_t*>(w.packed.data()),
        w.scales.data(), live_groups.data(), num_live,
        C_livepf.data(), K, N, GROUP_SIZE);

    // K-major repack path.
    const size_t N_blocks = (N + INTERLEAVE - 1) / INTERLEAVE;
    std::vector<uint8_t> pack_km(num_groups * N_blocks * 64);
    std::vector<__fp16> scales_km(num_groups * N_blocks * 4);
    cactus_repack_int4_kmajor(
        reinterpret_cast<const int8_t*>(w.packed.data()),
        w.scales.data(),
        pack_km.data(), scales_km.data(),
        K, N, GROUP_SIZE);
    cactus_gemv_int4_actsparse_kmajor(
        A_masked.data(), A_scale,
        pack_km.data(), scales_km.data(),
        live_groups.data(), num_live,
        C_km.data(), K, N, GROUP_SIZE);

    // K-major inline path (R3)
    std::vector<uint8_t> km_inline(num_groups * N_blocks * 72);
    std::vector<__fp16> C_kmi(N), C_kmi2(N), C_kmi4(N), C_kmi4c(N), C_kmi4f(N), C_kmi4v2(N);
    cactus_repack_int4_kmajor_inline(
        reinterpret_cast<const int8_t*>(w.packed.data()),
        w.scales.data(), km_inline.data(), K, N, GROUP_SIZE);
    cactus_gemv_int4_actsparse_kmi(
        A_masked.data(), A_scale, km_inline.data(),
        live_groups.data(), num_live, C_kmi.data(), K, N, GROUP_SIZE);
    cactus_gemv_int4_actsparse_kmi2(
        A_masked.data(), A_scale, km_inline.data(),
        live_groups.data(), num_live, C_kmi2.data(), K, N, GROUP_SIZE);
    cactus_gemv_int4_actsparse_kmi4(
        A_masked.data(), A_scale, km_inline.data(),
        live_groups.data(), num_live, C_kmi4.data(), K, N, GROUP_SIZE);
    cactus_gemv_int4_actsparse_kmi4_chain(
        A_masked.data(), A_scale, km_inline.data(),
        live_groups.data(), num_live, C_kmi4c.data(), K, N, GROUP_SIZE);
    cactus_gemv_int4_actsparse_kmi4_fast(
        A_masked.data(), A_scale, km_inline.data(),
        live_groups.data(), num_live, C_kmi4f.data(), K, N, GROUP_SIZE);
    cactus_gemv_int4_actsparse_kmi4_v2(
        A_masked.data(), A_scale, km_inline.data(),
        live_groups.data(), num_live, C_kmi4v2.data(), K, N, GROUP_SIZE);

    bool ok_azero = check_close("azero-vs-dense", C_azero.data(), C_dense.data(),
                                N, 1e-3f, 1e-3f);
    bool ok_mask  = check_close("bitmask-vs-dense", C_mask.data(), C_dense.data(),
                                N, 1e-3f, 1e-3f);
    bool ok_live  = check_close("livelist-vs-dense", C_live.data(), C_dense.data(),
                                N, 1e-3f, 1e-3f);
    bool ok_mask2 = check_close("mask2nb-vs-dense", C_mask2.data(), C_dense.data(),
                                N, 1e-3f, 1e-3f);
    bool ok_livepf = check_close("livepf-vs-dense", C_livepf.data(), C_dense.data(),
                                 N, 1e-3f, 1e-3f);
    bool ok_km   = check_close("kmajor-vs-dense", C_km.data(), C_dense.data(),
                                N, 1e-3f, 1e-3f);
    bool ok_kmi  = check_close("kmi-vs-dense",   C_kmi.data(),  C_dense.data(),
                                N, 1e-3f, 1e-3f);
    bool ok_kmi2 = check_close("kmi2-vs-dense",  C_kmi2.data(), C_dense.data(),
                                N, 1e-3f, 1e-3f);
    bool ok_kmi4 = check_close("kmi4-vs-dense",  C_kmi4.data(), C_dense.data(),
                                N, 1e-3f, 1e-3f);
    bool ok_kmi4c = check_close("kmi4chain-vs-dense", C_kmi4c.data(), C_dense.data(),
                                 N, 1e-3f, 1e-3f);
    bool ok_kmi4f = check_close("kmi4fast-vs-dense", C_kmi4f.data(), C_dense.data(),
                                 N, 1e-3f, 1e-3f);
    bool ok_kmi4v2 = check_close("kmi4v2-vs-dense", C_kmi4v2.data(), C_dense.data(),
                                 N, 1e-3f, 1e-3f);

    std::fprintf(stderr, "  groups=%zu live=%zu (%.2f%% skipped)\n",
                 num_groups, num_live,
                 100.0 * double(num_groups - num_live) / double(std::max<size_t>(1, num_groups)));

    return ok_dense && ok_azero && ok_mask && ok_live && ok_mask2 &&
           ok_livepf && ok_km && ok_kmi && ok_kmi2 && ok_kmi4 &&
           ok_kmi4c && ok_kmi4f && ok_kmi4v2;
}

} // namespace

int main() {
    struct Case { size_t K, N; float s; };
    Case cases[] = {
        {2048, 2048, 0.00f},
        {2048, 2048, 0.50f},
        {2048, 2048, 0.70f},
        {2048, 2048, 0.80f},
        {4096, 4096, 0.70f},
        {4096, 8192, 0.80f},
        {3072, 4096, 0.50f},
        {8192, 2048, 0.80f},
        {2048, 16384, 0.70f},
    };
    int rc = 0;
    for (auto c : cases) {
        if (!run_one(c.K, c.N, c.s, "iid", /*blocky*/false)) rc = 1;
        if (!run_one(c.K, c.N, c.s, "blocky", /*blocky*/true)) rc = 1;
    }
    std::fprintf(stderr, "%s\n", rc ? "FAIL" : "PASS");
    return rc;
}
