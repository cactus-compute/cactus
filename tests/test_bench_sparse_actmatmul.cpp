// Cold-cache benchmark for activation-sparse INT4 × INT8 GEMV.
//
// Rotates through a large pool of independent weight matrices (total
// footprint ≫ last-level cache) so every call sees cold B. Fresh
// A + S every iteration. Reports median, P95 µs, and achieved weight
// bandwidth (GB/s). Dense baseline is cactus_gemv_int4.
//
// Primary pattern: "blocky" — realistic block-structured S where a
// lane-drop threshold translates into whole-group drops (i.e.
// activation sparsity as seen in MLP hidden states after gating).
// Also reports the "iid" pathological case (little group-skip) for
// honesty.

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
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr size_t GROUP_SIZE = 32;
constexpr size_t INTERLEAVE = 4;

// Require each weight pool be at least this large so ≥ 4× SLC is hit.
// M4 Pro SLC ≈ 24 MB; 4× = 96 MB. Use 256 MB to be safe.
constexpr size_t POOL_TARGET_BYTES = 256ull * 1024 * 1024;
constexpr size_t CACHE_TRASH_BYTES = 64ull * 1024 * 1024;

void pack_int4_32(const int8_t* src32, uint8_t* dst16) {
    for (size_t i = 0; i < 16; ++i) {
        uint8_t lo = static_cast<uint8_t>(src32[i]) & 0x0F;
        uint8_t hi = static_cast<uint8_t>(src32[16 + i]) & 0x0F;
        dst16[i] = lo | (hi << 4);
    }
}

struct DenseWeights {
    std::vector<uint8_t> packed;
    std::vector<__fp16> scales;
};

DenseWeights build_weights(size_t N, size_t K, unsigned seed) {
    DenseWeights w;
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

struct KMajorWeights {
    std::vector<uint8_t> packed;
    std::vector<__fp16> scales;
};

struct KMInlineWeights {
    std::vector<uint8_t> buf;
};

struct Pool {
    std::vector<DenseWeights> mats;
    std::vector<KMajorWeights> km;
    std::vector<KMInlineWeights> kmi;
    size_t bytes_per_matrix;
};

Pool build_pool(size_t N, size_t K) {
    Pool p;
    const size_t N_padded = ((N + INTERLEAVE - 1) / INTERLEAVE) * INTERLEAVE;
    p.bytes_per_matrix = N_padded * K / 2
                       + (N_padded / INTERLEAVE) * (K / GROUP_SIZE) * INTERLEAVE * sizeof(__fp16);
    size_t count = (POOL_TARGET_BYTES + p.bytes_per_matrix - 1) / p.bytes_per_matrix;
    count = std::max<size_t>(count, 8);
    p.mats.reserve(count);
    p.km.reserve(count);
    const size_t num_groups = K / GROUP_SIZE;
    const size_t N_blocks = N_padded / INTERLEAVE;
    p.kmi.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto w = build_weights(N, K, static_cast<unsigned>(0xC0DE + i * 1013));
        KMajorWeights km;
        km.packed.resize(num_groups * N_blocks * 64);
        km.scales.resize(num_groups * N_blocks * 4);
        cactus_repack_int4_kmajor(
            reinterpret_cast<const int8_t*>(w.packed.data()),
            w.scales.data(),
            km.packed.data(), km.scales.data(),
            K, N, GROUP_SIZE);
        KMInlineWeights kmi;
        kmi.buf.resize(num_groups * N_blocks * 72);
        cactus_repack_int4_kmajor_inline(
            reinterpret_cast<const int8_t*>(w.packed.data()),
            w.scales.data(), kmi.buf.data(), K, N, GROUP_SIZE);
        p.mats.push_back(std::move(w));
        p.km.push_back(std::move(km));
        p.kmi.push_back(std::move(kmi));
    }
    return p;
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
    const size_t nb = (K + block_size - 1) / block_size;
    for (size_t b = 0; b < nb; ++b) {
        float coarse = -std::log(std::max(1e-6f, u(rng)));
        size_t k0 = b * block_size;
        size_t k1 = std::min(k0 + block_size, K);
        for (size_t k = k0; k < k1; ++k) {
            S[k] = coarse + 0.1f * u(rng);
        }
    }
}

void fill_A(std::vector<int8_t>& A, size_t K, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> ad(-120, 120);
    A.resize(K);
    for (auto& v : A) v = static_cast<int8_t>(ad(rng));
}

// Flush caches by streaming through a large buffer.
void trash_cache(std::vector<uint8_t>& trash) {
    for (size_t i = 0; i < trash.size(); i += 64) {
        trash[i] = static_cast<uint8_t>(i);
    }
    asm volatile ("" : : "r"(trash.data()) : "memory");
}

template <typename F>
std::pair<double, double> bench_stats(F&& fn, size_t iters) {
    std::vector<double> samples(iters);
    for (size_t i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn(i);
        auto t1 = std::chrono::high_resolution_clock::now();
        samples[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    std::sort(samples.begin(), samples.end());
    double med = samples[iters / 2];
    double p95 = samples[std::min(iters - 1, (iters * 95) / 100)];
    return {med, p95};
}

struct Row {
    std::string pattern;
    size_t K, N;
    float sparsity;
    double skip_pct;
    double us_dense, us_winner;
    double winner_bw, dense_bw;
    double ideal_bw;
    const char* winner_name;
    double speedup;
};

// Weight bytes actually read by a kernel, for bandwidth accounting.
// For baseline and azero: always reads all groups.
// For bitmask/livelist: reads live-group bytes only.
double weight_bytes(size_t N, size_t K, double live_fraction) {
    const size_t N_blocks = (N + INTERLEAVE - 1) / INTERLEAVE;
    const size_t num_groups = K / GROUP_SIZE;
    const double bytes_per_group = 64.0 + 4.0 * sizeof(__fp16); // packed + scales
    return double(N_blocks) * double(num_groups) * bytes_per_group * live_fraction;
}

void bench_one(const std::string& pattern_name, bool blocky,
               size_t K, size_t N, float sparsity,
               std::vector<Row>& out, FILE* csv)
{
    Pool pool = build_pool(N, K);

    // Build one A / S / mask set per matrix so each iteration sees fresh
    // inputs as well.
    const size_t matrices = pool.mats.size();
    std::vector<std::vector<int8_t>> As(matrices);
    std::vector<std::vector<int8_t>> A_masked(matrices);
    std::vector<std::vector<uint64_t>> bitmasks(matrices);
    std::vector<std::vector<uint16_t>> livelists(matrices);
    std::vector<size_t>                live_counts(matrices);
    const size_t num_groups = K / GROUP_SIZE;
    size_t skipped_total = 0;

    for (size_t i = 0; i < matrices; ++i) {
        fill_A(As[i], K, static_cast<unsigned>(0xA11C + i * 97));
        A_masked[i].assign(K, 0);
        bitmasks[i].assign((num_groups + 63) / 64, 0);
        livelists[i].assign(num_groups, 0);
        std::vector<float> S;
        if (blocky)
            make_blocky_scores(S, K, static_cast<unsigned>(0x5C0 + i * 37), GROUP_SIZE * 4);
        else
            make_iid_scores(S, K, static_cast<unsigned>(0x5C0 + i * 37));

        live_counts[i] = cactus_build_actsparse_mask_f32(
            S.data(), As[i].data(), K, sparsity, GROUP_SIZE,
            A_masked[i].data(), bitmasks[i].data(), livelists[i].data());
        skipped_total += (num_groups - live_counts[i]);
    }
    (void)skipped_total;

    std::vector<__fp16> C(N);
    std::vector<uint8_t> trash(CACHE_TRASH_BYTES, 0);

    // Run a large number of iterations (at least ~3 full pool cycles).
    size_t iters = std::max<size_t>(matrices * 6, 64);

    auto run_idx = [&](size_t i) { return i % matrices; };

    // warm-up (exercise each matrix once so we hit TLB etc.)
    for (size_t i = 0; i < matrices; ++i) {
        cactus_gemv_int4(As[i].data(), 0.05f,
                         reinterpret_cast<const int8_t*>(pool.mats[i].packed.data()),
                         pool.mats[i].scales.data(), C.data(), K, N, GROUP_SIZE);
    }

    auto bench_kernel = [&](auto run) {
        std::vector<double> samples; samples.reserve(iters);
        for (size_t i = 0; i < iters; ++i) {
            trash_cache(trash);
            size_t idx = run_idx(i);
            auto t0 = std::chrono::high_resolution_clock::now();
            run(idx);
            auto t1 = std::chrono::high_resolution_clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        std::sort(samples.begin(), samples.end());
        double med = samples[iters / 2];
        double p95 = samples[(iters * 95) / 100];
        return std::pair<double,double>{med, p95};
    };

    auto [us_dense, p95_dense] = bench_kernel([&](size_t idx) {
        cactus_gemv_int4(As[idx].data(), 0.05f,
                         reinterpret_cast<const int8_t*>(pool.mats[idx].packed.data()),
                         pool.mats[idx].scales.data(), C.data(), K, N, GROUP_SIZE);
    });
    auto [us_azero, p95_azero] = bench_kernel([&](size_t idx) {
        cactus_gemv_int4_actsparse_azero(
            A_masked[idx].data(), 0.05f,
            reinterpret_cast<const int8_t*>(pool.mats[idx].packed.data()),
            pool.mats[idx].scales.data(), C.data(), K, N, GROUP_SIZE);
    });
    auto [us_mask, p95_mask] = bench_kernel([&](size_t idx) {
        cactus_gemv_int4_actsparse_bitmask(
            A_masked[idx].data(), 0.05f,
            reinterpret_cast<const int8_t*>(pool.mats[idx].packed.data()),
            pool.mats[idx].scales.data(), bitmasks[idx].data(),
            C.data(), K, N, GROUP_SIZE);
    });
    auto [us_live, p95_live] = bench_kernel([&](size_t idx) {
        cactus_gemv_int4_actsparse_livelist(
            A_masked[idx].data(), 0.05f,
            reinterpret_cast<const int8_t*>(pool.mats[idx].packed.data()),
            pool.mats[idx].scales.data(), livelists[idx].data(), live_counts[idx],
            C.data(), K, N, GROUP_SIZE);
    });
    auto [us_mask2, p95_mask2] = bench_kernel([&](size_t idx) {
        cactus_gemv_int4_actsparse_bitmask_2nb(
            A_masked[idx].data(), 0.05f,
            reinterpret_cast<const int8_t*>(pool.mats[idx].packed.data()),
            pool.mats[idx].scales.data(), bitmasks[idx].data(),
            C.data(), K, N, GROUP_SIZE);
    });
    auto [us_livepf, p95_livepf] = bench_kernel([&](size_t idx) {
        cactus_gemv_int4_actsparse_livelist_pf(
            A_masked[idx].data(), 0.05f,
            reinterpret_cast<const int8_t*>(pool.mats[idx].packed.data()),
            pool.mats[idx].scales.data(), livelists[idx].data(), live_counts[idx],
            C.data(), K, N, GROUP_SIZE);
    });
    auto [us_km, p95_km] = bench_kernel([&](size_t idx) {
        cactus_gemv_int4_actsparse_kmajor(
            A_masked[idx].data(), 0.05f,
            pool.km[idx].packed.data(), pool.km[idx].scales.data(),
            livelists[idx].data(), live_counts[idx],
            C.data(), K, N, GROUP_SIZE);
    });
    auto [us_kmi, p95_kmi] = bench_kernel([&](size_t idx) {
        cactus_gemv_int4_actsparse_kmi(
            A_masked[idx].data(), 0.05f,
            pool.kmi[idx].buf.data(),
            livelists[idx].data(), live_counts[idx],
            C.data(), K, N, GROUP_SIZE);
    });
    auto [us_kmi2, p95_kmi2] = bench_kernel([&](size_t idx) {
        cactus_gemv_int4_actsparse_kmi2(
            A_masked[idx].data(), 0.05f,
            pool.kmi[idx].buf.data(),
            livelists[idx].data(), live_counts[idx],
            C.data(), K, N, GROUP_SIZE);
    });
    auto [us_kmi4, p95_kmi4] = bench_kernel([&](size_t idx) {
        cactus_gemv_int4_actsparse_kmi4(
            A_masked[idx].data(), 0.05f,
            pool.kmi[idx].buf.data(),
            livelists[idx].data(), live_counts[idx],
            C.data(), K, N, GROUP_SIZE);
    });
    auto [us_kmi4c, p95_kmi4c] = bench_kernel([&](size_t idx) {
        cactus_gemv_int4_actsparse_kmi4_chain(
            A_masked[idx].data(), 0.05f,
            pool.kmi[idx].buf.data(),
            livelists[idx].data(), live_counts[idx],
            C.data(), K, N, GROUP_SIZE);
    });
    auto [us_kmi4f, p95_kmi4f] = bench_kernel([&](size_t idx) {
        cactus_gemv_int4_actsparse_kmi4_fast(
            A_masked[idx].data(), 0.05f,
            pool.kmi[idx].buf.data(),
            livelists[idx].data(), live_counts[idx],
            C.data(), K, N, GROUP_SIZE);
    });
    auto [us_kmi4v2, p95_kmi4v2] = bench_kernel([&](size_t idx) {
        cactus_gemv_int4_actsparse_kmi4_v2(
            A_masked[idx].data(), 0.05f,
            pool.kmi[idx].buf.data(),
            livelists[idx].data(), live_counts[idx],
            C.data(), K, N, GROUP_SIZE);
    });

    double live_frac = num_groups > 0
        ? 1.0 - double(skipped_total) / (double(matrices) * double(num_groups))
        : 1.0;
    double bw_dense  = weight_bytes(N, K, 1.0) / (us_dense * 1e3);
    double bw_mask   = weight_bytes(N, K, live_frac) / (us_mask * 1e3);
    double bw_live   = weight_bytes(N, K, live_frac) / (us_live * 1e3);
    double bw_mask2  = weight_bytes(N, K, live_frac) / (us_mask2 * 1e3);
    double bw_livepf = weight_bytes(N, K, live_frac) / (us_livepf * 1e3);
    double bw_km     = weight_bytes(N, K, live_frac) / (us_km * 1e3);
    // For KMI, the "byte budget" is 72 per tile instead of 72; keep the same BW model.
    double bw_kmi   = weight_bytes(N, K, live_frac) / (us_kmi * 1e3);
    double bw_kmi2  = weight_bytes(N, K, live_frac) / (us_kmi2 * 1e3);
    double bw_kmi4  = weight_bytes(N, K, live_frac) / (us_kmi4 * 1e3);
    double bw_kmi4c = weight_bytes(N, K, live_frac) / (us_kmi4c * 1e3);
    double bw_kmi4f = weight_bytes(N, K, live_frac) / (us_kmi4f * 1e3);
    double bw_kmi4v2 = weight_bytes(N, K, live_frac) / (us_kmi4v2 * 1e3);
    double sp_azero  = us_dense / us_azero;
    double sp_mask   = us_dense / us_mask;
    double sp_live   = us_dense / us_live;
    double sp_mask2  = us_dense / us_mask2;
    double sp_livepf = us_dense / us_livepf;
    double sp_km     = us_dense / us_km;
    double sp_kmi    = us_dense / us_kmi;
    double sp_kmi2   = us_dense / us_kmi2;
    double sp_kmi4   = us_dense / us_kmi4;
    double sp_kmi4c  = us_dense / us_kmi4c;
    double sp_kmi4f  = us_dense / us_kmi4f;
    double sp_kmi4v2 = us_dense / us_kmi4v2;

    std::printf(
        "[%s] K=%-5zu N=%-5zu sp=%2.0f%% skip=%4.1f%% d=%6.1f(%5.1fGBs) lpf=%4.2fx KM=%4.2fx KMI2=%4.2fx KMI4=%4.2fx(%5.1f) KMI4c=%4.2fx(%5.1f) KMI4f=%4.2fx(%5.1f) KMI4v2=%4.2fx(%5.1f)\n",
        pattern_name.c_str(), K, N, sparsity * 100.0,
        100.0 - 100.0 * live_frac,
        us_dense, bw_dense,
        sp_livepf,
        sp_km,
        sp_kmi2,
        sp_kmi4, bw_kmi4,
        sp_kmi4c, bw_kmi4c,
        sp_kmi4f, bw_kmi4f,
        sp_kmi4v2, bw_kmi4v2);

    (void)sp_azero;

    (void)sp_mask2; (void)sp_kmi; (void)bw_kmi;

    // Pick winner among the sparse kernels.
    struct { const char* name; double sp; double bw; double us; } cands[] = {
        {"livepf", sp_livepf, 0.0, us_livepf},
        {"KM",     sp_km,     bw_km,   us_km},
        {"KMI2",   sp_kmi2,   bw_kmi2, us_kmi2},
        {"KMI4",   sp_kmi4,   bw_kmi4, us_kmi4},
        {"KMI4c",  sp_kmi4c,  bw_kmi4c,us_kmi4c},
        {"KMI4f",  sp_kmi4f,  bw_kmi4f,us_kmi4f},
        {"KMI4v2", sp_kmi4v2, bw_kmi4v2,us_kmi4v2},
    };
    size_t best = 0;
    for (size_t i = 1; i < sizeof(cands)/sizeof(cands[0]); ++i)
        if (cands[i].sp > cands[best].sp) best = i;

    Row r;
    r.pattern = pattern_name;
    r.K = K; r.N = N; r.sparsity = sparsity;
    r.skip_pct = 100.0 - 100.0 * live_frac;
    r.us_dense = us_dense;
    r.us_winner = cands[best].us;
    r.speedup = cands[best].sp;
    r.winner_bw = cands[best].bw;
    r.dense_bw = bw_dense;
    r.ideal_bw = bw_dense;  // theoretical: same per-byte throughput × fewer bytes
    r.winner_name = cands[best].name;
    out.push_back(r);

    (void)sp_mask; (void)sp_live;

    if (csv) {
        std::fprintf(csv,
                     "%s,%zu,%zu,%.2f,%.4f,"
                     "%.4f,%.4f,%.4f,"       // dense
                     "%.4f,%.4f,"            // azero
                     "%.4f,%.4f,%.4f,"       // mask
                     "%.4f,%.4f,%.4f,"       // live
                     "%.4f,%.4f,%.4f,"       // mask2
                     "%.4f,%.4f,%.4f,"       // livepf
                     "%.4f,%.4f,%.4f,"       // km
                     "%.4f,%.4f,%.4f,"       // kmi
                     "%.4f,%.4f,%.4f,"       // kmi2
                     "%.4f,%.4f,%.4f,"       // kmi4
                     "%.4f,%.4f,%.4f,"       // kmi4_chain
                     "%.4f,%.4f,%.4f,"       // kmi4_fast
                     "%.4f,%.4f,%.4f\n",     // kmi4_v2
                     pattern_name.c_str(), K, N, sparsity,
                     100.0 - 100.0 * live_frac,
                     us_dense, p95_dense, bw_dense,
                     us_azero, p95_azero,
                     us_mask, p95_mask, bw_mask,
                     us_live, p95_live, bw_live,
                     us_mask2, p95_mask2, bw_mask2,
                     us_livepf, p95_livepf, bw_livepf,
                     us_km, p95_km, bw_km,
                     us_kmi, p95_kmi, bw_kmi,
                     us_kmi2, p95_kmi2, bw_kmi2,
                     us_kmi4, p95_kmi4, bw_kmi4,
                     us_kmi4c, p95_kmi4c, bw_kmi4c,
                     us_kmi4f, p95_kmi4f, bw_kmi4f,
                     us_kmi4v2, p95_kmi4v2, bw_kmi4v2);
    }

    (void)out;
}

} // namespace

int main(int argc, char** argv) {
    std::printf("Cold-cache Sparse-activation INT4 GEMV bench\n");
    std::printf("hw.concurrency=%u  pool_target=%zu MB\n",
                std::thread::hardware_concurrency(),
                POOL_TARGET_BYTES / (1024 * 1024));

    const size_t Ks[] = {2048, 3072, 4096, 8192};
    const size_t Ns[] = {2048, 4096, 8192, 16384};
    const float Sp[] = {0.50f, 0.70f, 0.80f};

    std::vector<std::string> patterns = {"blocky"};
    if (argc > 1 && std::string(argv[1]) == "--iid-also") patterns.push_back("iid");

    FILE* csv = std::fopen("sparse_actmatmul_bench.csv", "w");
    if (csv) {
        std::fprintf(csv,
                     "pattern,K,N,sparsity,grp_skip_pct,"
                     "dense_us,dense_p95,dense_bw,"
                     "azero_us,azero_p95,"
                     "mask_us,mask_p95,mask_bw,"
                     "live_us,live_p95,live_bw,"
                     "mask2_us,mask2_p95,mask2_bw,"
                     "livepf_us,livepf_p95,livepf_bw,"
                     "km_us,km_p95,km_bw,"
                     "kmi_us,kmi_p95,kmi_bw,"
                     "kmi2_us,kmi2_p95,kmi2_bw,"
                     "kmi4_us,kmi4_p95,kmi4_bw,"
                     "kmi4c_us,kmi4c_p95,kmi4c_bw,"
                     "kmi4f_us,kmi4f_p95,kmi4f_bw,"
                     "kmi4v2_us,kmi4v2_p95,kmi4v2_bw\n");
    }

    std::vector<Row> rows;
    for (const auto& pat : patterns) {
        bool blocky = (pat == "blocky");
        for (size_t K : Ks) {
            for (size_t N : Ns) {
                for (float sp : Sp) {
                    bench_one(pat, blocky, K, N, sp, rows, csv);
                }
            }
        }
    }
    if (csv) std::fclose(csv);

    // Final summary table (spec §7 deliverable).
    std::printf("\n===== FINAL SUMMARY (best kernel per shape) =====\n");
    std::printf("%-6s %5s %5s %5s %7s  %8s %8s %7s  %8s %8s  %-8s\n",
                "patt", "K", "N", "sp%", "skip%",
                "dense_us", "win_us", "speedup",
                "win_GBs", "dBW_GBs", "winner");
    auto print_section = [&](const std::string& pat) {
        for (const auto& r : rows) {
            if (r.pattern != pat) continue;
            std::printf("%-6s %5zu %5zu %4.0f%% %6.1f%%  %8.1f %8.1f %6.2fx  %7.1f %7.1f  %s\n",
                        r.pattern.c_str(), r.K, r.N, r.sparsity * 100.0, r.skip_pct,
                        r.us_dense, r.us_winner, r.speedup,
                        r.winner_bw, r.dense_bw, r.winner_name);
        }
    };
    print_section("blocky");
    print_section("iid");

    // Criterion check against spec §3.
    std::printf("\n===== SUCCESS-CRITERION CHECK (blocky only) =====\n");
    std::printf("target: sp=0.80 ≥3.0x, sp=0.70 ≥2.0x, sp=0.50 ≥1.4x\n");
    auto check = [&](float sp_target, double thresh) {
        int total = 0, pass = 0;
        for (const auto& r : rows) {
            if (r.pattern != "blocky") continue;
            if (std::fabs(r.sparsity - sp_target) > 0.01) continue;
            ++total;
            if (r.speedup >= thresh) ++pass;
        }
        std::printf("  sp=%.2f  threshold=%.2fx  passed %d/%d shapes\n",
                    sp_target, thresh, pass, total);
    };
    check(0.80f, 3.0);
    check(0.70f, 2.0);
    check(0.50f, 1.4);
    return 0;
}
