#include "bench_driver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

namespace bench {

// Working-set target — exceed Apple Silicon SLC (~32-48MB) so weight reads
// miss to RAM, matching real inference where every layer has unique weights.
static constexpr size_t kCacheBypassBytes = 64u * 1024u * 1024u;

static constexpr size_t kMaxMatrixCount = 4096;
static constexpr size_t kMaxAttnStateCount = 512;

// Slow backends are excluded from the interleaved loop (otherwise the
// round-robin is bounded by the slowest per iteration), then re-timed in a
// solo warmed pass so their number still reflects sustained performance.
static constexpr int    kProbeIters     = 3;
static constexpr double kSlowMultiplier = 25.0;
static constexpr int    kSoloWarmup     = 5;
static constexpr int    kSoloIters      = 10;

static double median_ms(std::vector<double> samples) {
    if (samples.empty()) return 0.0;
    const size_t mid = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + mid, samples.end());
    double hi = samples[mid];
    if (samples.size() % 2) return hi;
    std::nth_element(samples.begin(), samples.begin() + mid - 1, samples.end());
    return 0.5 * (samples[mid - 1] + hi);
}

static std::vector<MatmulBackendVariant>& matmul_registry() {
    static std::vector<MatmulBackendVariant> r;
    return r;
}
void register_matmul_backend(MatmulBackendVariant v) { matmul_registry().push_back(v); }
const std::vector<MatmulBackendVariant>& get_matmul_backends() { return matmul_registry(); }

static std::vector<AttnBackendVariant>& attn_registry() {
    static std::vector<AttnBackendVariant> r;
    return r;
}
void register_attn_backend(AttnBackendVariant v) { attn_registry().push_back(v); }
const std::vector<AttnBackendVariant>& get_attn_backends() { return attn_registry(); }

static std::string thread_label(int n) {
    if (n == 0) return "default";
    if (n == static_cast<int>(std::thread::hardware_concurrency())) {
        std::ostringstream os; os << "max(" << n << ")"; return os.str();
    }
    std::ostringstream os; os << n; return os.str();
}

bool run_matmul_benchmark(const MatmulBenchOptions& opt) {
    const auto& all = get_matmul_backends();

    std::vector<const MatmulBackendVariant*> active;
    for (const auto& b : all) {
        if (!b.run_kernel) continue;
        if (framework_matches_filter(b.framework, opt.backends_filter))
            active.push_back(&b);
    }
    if (active.empty()) {
        std::cerr << "[matmul] no backends matched filter\n";
        return false;
    }

    set_thread_override(opt.num_threads);

    std::cout << "# matmul-bench: warmup=" << opt.warmup
              << " iter=" << opt.iterations
              << " cache_target=" << (kCacheBypassBytes >> 20) << "MB"
              << " threads=" << thread_label(opt.num_threads)
              << " backends=";
    for (size_t i = 0; i < active.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << active[i]->name;
    }
    std::cout << "\n";

    std::ofstream csv;
    if (!opt.csv_path.empty()) {
        csv.open(opt.csv_path);
        csv << "graph,sweep_dim,M,K,N,backend,framework,time_us,p50_us,gops,nrmse,max_err\n";
    }

    auto configs = build_matmul_configs(opt);
    std::mt19937 gen(270270u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Round-robin across backends per iteration averages thermal/frequency
    // state — without it, whichever backend runs first cools the CPU for
    // whichever runs second.
    struct Entry { void* w = nullptr; void* a = nullptr; };
    struct BackendSlot {
        const MatmulBackendVariant* backend = nullptr;
        std::vector<Entry> entries;
        AccuracyResult acc;
        bool prepared = false;
        bool skipped_slow = false;
        double probe_avg_ms = 0.0;
        double total_ms = 0.0;
        std::vector<double> samples_ms;
    };

    for (const auto& cfg : configs) {
        const size_t M = cfg.M, K = cfg.K, N = cfg.N;

        // NM cycles enough distinct quantized matrices to push the weight
        // working set past kCacheBypassBytes. Floor=2 so a single matrix
        // doesn't sit in registers across iterations.
        const size_t weight_bytes = N * K;
        const size_t NM_for_bypass = (kCacheBypassBytes + weight_bytes - 1)
                                      / std::max(weight_bytes, size_t(1));
        const size_t NM = std::min(kMaxMatrixCount, std::max(size_t(2), NM_for_bypass));

        std::cout << "\n── " << matmul_graph_name(cfg.graph)
                  << " sweep=" << cfg.sweep_dim
                  << " (" << M << "x" << K << "x" << N << ")"
                  << " NM=" << NM << "\n";

        std::vector<std::vector<float>> fp32_w(NM);
        for (size_t i = 0; i < NM; ++i) {
            fp32_w[i].resize(N * K);
            for (auto& v : fp32_w[i]) v = dist(gen);
        }

        std::mt19937 agen(static_cast<uint32_t>(42u + cfg.sweep_dim + (size_t)cfg.graph));
        auto act = prepare_cactus_activations(M, K, agen);

        std::vector<float> reference(M * N);
        reference_matmul_fp32(act.fp32.data(), fp32_w[0].data(),
                              reference.data(), M, K, N);

        std::vector<BackendSlot> slots(active.size());
        const size_t out_count = M * N;

        for (size_t bi = 0; bi < active.size(); ++bi) {
            auto& slot = slots[bi];
            slot.backend = active[bi];
            slot.entries.resize(NM);
            slot.prepared = true;

            for (size_t i = 0; i < NM; ++i) {
                slot.entries[i].w = slot.backend->prepare_weights(fp32_w[i].data(), N, K);
                if (!slot.entries[i].w) { slot.prepared = false; break; }
                if (slot.backend->prepare_activations)
                    slot.entries[i].a = slot.backend->prepare_activations(
                        act.fp32.data(), M, K, slot.entries[i].w);
            }

            if (!slot.prepared) {
                std::cout << "  " << slot.backend->name << "  prepare=FAIL\n";
                for (auto& e : slot.entries)
                    if (slot.backend->cleanup) slot.backend->cleanup(e.w, e.a);
                slot.entries.clear();
                continue;
            }

            std::vector<float> captured(out_count, 0.0f);
            std::vector<float> backend_ref(out_count, 0.0f);
            slot.backend->run_kernel(M, K, N, slot.entries[0].w, slot.entries[0].a,
                                     act.int8.data(), act.scales.data(),
                                     captured.data(), backend_ref.data());

            bool has_backend_ref = false;
            for (size_t i = 0; i < out_count && !has_backend_ref; ++i)
                if (backend_ref[i] != 0.0f) has_backend_ref = true;
            const float* ref = has_backend_ref ? backend_ref.data() : reference.data();
            float tol = has_backend_ref ? 0.01f : 0.10f;
            slot.acc = check_accuracy(ref, captured.data(), out_count, tol);
        }

        // Drop FP32 source weights so we don't pay 4× memory at high NM.
        std::vector<std::vector<float>>().swap(fp32_w);

        for (auto& slot : slots) {
            if (!slot.prepared) continue;
            double total = 0.0;
            for (int p = 0; p < kProbeIters; ++p) {
                size_t idx = static_cast<size_t>(p) % NM;
                double t0 = now_ms();
                slot.backend->run_kernel(M, K, N,
                                         slot.entries[idx].w, slot.entries[idx].a,
                                         act.int8.data(), act.scales.data(),
                                         nullptr, nullptr);
                total += now_ms() - t0;
            }
            slot.probe_avg_ms = total / kProbeIters;
        }
        double cactus_probe_ms = -1.0;
        for (const auto& slot : slots) {
            if (slot.prepared && std::string(slot.backend->framework) == "cactus") {
                cactus_probe_ms = slot.probe_avg_ms;
                break;
            }
        }
        if (cactus_probe_ms > 0.0) {
            for (auto& slot : slots) {
                if (!slot.prepared) continue;
                if (slot.probe_avg_ms > kSlowMultiplier * cactus_probe_ms)
                    slot.skipped_slow = true;
            }
        } else {
            static bool warned = false;
            if (!warned) {
                std::cerr << "[bench] cactus not in active backends — slow-skip "
                             "disabled. Run will not auto-skip outlier-slow "
                             "backends; expect long wall time.\n";
                warned = true;
            }
        }

        for (auto& slot : slots) {
            if (!slot.prepared || !slot.skipped_slow) continue;
            for (int w = 0; w < kSoloWarmup; ++w) {
                size_t idx = static_cast<size_t>(w) % NM;
                slot.backend->run_kernel(M, K, N,
                                         slot.entries[idx].w, slot.entries[idx].a,
                                         act.int8.data(), act.scales.data(),
                                         nullptr, nullptr);
            }
            slot.total_ms = 0.0;
            slot.samples_ms.clear();
            for (int it = 0; it < kSoloIters; ++it) {
                size_t idx = static_cast<size_t>(it) % NM;
                double t0 = now_ms();
                slot.backend->run_kernel(M, K, N,
                                         slot.entries[idx].w, slot.entries[idx].a,
                                         act.int8.data(), act.scales.data(),
                                         nullptr, nullptr);
                double dt = now_ms() - t0;
                slot.total_ms += dt;
                slot.samples_ms.push_back(dt);
            }
        }

        for (int w = 0; w < opt.warmup; ++w) {
            size_t idx = static_cast<size_t>(w) % NM;
            for (size_t n = 0; n < slots.size(); ++n) {
                auto& slot = slots[(static_cast<size_t>(w) + n) % slots.size()];
                if (!slot.prepared || slot.skipped_slow) continue;
                slot.backend->run_kernel(M, K, N,
                                         slot.entries[idx].w, slot.entries[idx].a,
                                         act.int8.data(), act.scales.data(),
                                         nullptr, nullptr);
            }
        }

        for (int it = 0; it < opt.iterations; ++it) {
            size_t idx = static_cast<size_t>(it) % NM;
            for (size_t n = 0; n < slots.size(); ++n) {
                auto& slot = slots[(static_cast<size_t>(it) + n) % slots.size()];
                if (!slot.prepared || slot.skipped_slow) continue;
                double t0 = now_ms();
                slot.backend->run_kernel(M, K, N,
                                         slot.entries[idx].w, slot.entries[idx].a,
                                         act.int8.data(), act.scales.data(),
                                         nullptr, nullptr);
                double dt = now_ms() - t0;
                slot.total_ms += dt;
                slot.samples_ms.push_back(dt);
            }
        }

        for (auto& slot : slots) {
            if (!slot.prepared) continue;
            int iters = slot.samples_ms.empty() ? 1 : static_cast<int>(slot.samples_ms.size());
            double avg_ms = slot.total_ms / iters;
            double avg_us = avg_ms * 1000.0;
            double p50_us = median_ms(slot.samples_ms) * 1000.0;
            double gops = compute_matmul_gops(M, K, N, iters, slot.total_ms);

            std::cout << "  " << std::left << std::setw(28) << slot.backend->name
                      << " mean=" << std::fixed << std::setprecision(3) << avg_ms << " ms"
                      << " p50=" << std::setprecision(3) << (p50_us / 1000.0) << " ms"
                      << " (" << std::setprecision(2) << gops << " GOPS)"
                      << "  acc=" << (slot.acc.passed ? "PASS" : "FAIL")
                      << " nrmse=" << std::setprecision(4) << slot.acc.nrmse;
            if (slot.skipped_slow)
                std::cout << "  [slow — solo " << kSoloIters << "-iter pass]";
            std::cout << "\n";

            if (csv.is_open()) {
                csv << matmul_graph_name(cfg.graph) << ","
                    << cfg.sweep_dim << "," << M << "," << K << "," << N << ","
                    << slot.backend->name << "," << slot.backend->framework << ","
                    << std::fixed << std::setprecision(3) << avg_us << ","
                    << std::setprecision(3) << p50_us << ","
                    << std::setprecision(3) << gops << ","
                    << std::setprecision(6) << slot.acc.nrmse << ","
                    << std::setprecision(6) << slot.acc.max_abs_error << "\n";
                csv.flush();
            }

            for (auto& e : slot.entries)
                if (slot.backend->cleanup) slot.backend->cleanup(e.w, e.a);
        }
    }

    set_thread_override(0);
    return true;
}

static float attn_tolerance(const char* name) {
    std::string n(name);
    if (n.find("q4") != std::string::npos) return 0.20f;
    if (n.find("int8") != std::string::npos || n.find("hybrid") != std::string::npos
        || n.find("q8") != std::string::npos)
        return 0.10f;
    return 0.05f;
}

bool run_attn_benchmark(const AttnBenchOptions& opt) {
    const auto& all = get_attn_backends();

    std::vector<const AttnBackendVariant*> active;
    for (const auto& b : all) {
        if (!b.run) continue;
        if (framework_matches_filter(b.framework, opt.backends_filter))
            active.push_back(&b);
    }
    if (active.empty()) {
        std::cerr << "[attn] no backends matched filter\n";
        return false;
    }

    set_thread_override(opt.num_threads);

    std::cout << "# attn-bench: warmup=" << opt.warmup
              << " iter=" << opt.iterations
              << " threads=" << thread_label(opt.num_threads)
              << " model_dim=" << opt.model_dim
              << " heads=" << opt.num_heads
              << " backends=";
    for (size_t i = 0; i < active.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << active[i]->name;
    }
    std::cout << "\n";

    std::ofstream csv;
    if (!opt.csv_path.empty()) {
        csv.open(opt.csv_path);
        csv << "graph,sweep_dim,seq_len,cache_len,head_dim,q_heads,kv_heads,backend,framework,time_us,p50_us,gflops,nrmse,max_err\n";
    }

    auto configs = build_attn_configs(opt);
    std::mt19937 gen(270270u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    struct AttnSlot {
        const AttnBackendVariant* backend = nullptr;
        std::vector<void*> states;
        AccuracyResult acc;
        bool prepared = false;
        bool skipped_slow = false;
        double probe_avg_ms = 0.0;
        double total_ms = 0.0;
        std::vector<double> samples_ms;
    };

    for (const auto& cfg : configs) {
        const auto& dims = cfg.dims;
        const size_t sl = cfg.seq_len;
        const size_t kvl = (cfg.mode == AttnMode::PREFILL) ? sl : (cfg.cache_len + 1);
        const float scale = 1.0f / std::sqrt(static_cast<float>(dims.head_dim));

        // NS cycles enough Q/K/V states to push the per-state KV footprint
        // past kCacheBypassBytes. We size against the *smallest* backend storage
        // (INT8 K+V plus fp32 group scales) so quantized backends don't get
        // unfair L2/SLC locality on small cache_lens.
        const size_t groups_per_token = std::max<size_t>(1, dims.head_dim / 32);
        const size_t per_state_kv_bytes =
            2 * dims.num_kv_heads * kvl * dims.head_dim * sizeof(int8_t) +
            2 * dims.num_kv_heads * kvl * groups_per_token * sizeof(float);
        const size_t NS_for_bypass = (kCacheBypassBytes + per_state_kv_bytes - 1)
                                      / std::max(per_state_kv_bytes, size_t(1));
        const size_t NS = std::min(kMaxAttnStateCount, std::max(static_cast<size_t>(2), NS_for_bypass));

        std::cout << "\n── " << attn_graph_name(cfg.graph)
                  << " sweep=" << cfg.sweep_dim
                  << " (sl=" << sl << " kvl=" << kvl
                  << " hd=" << dims.head_dim
                  << " qh=" << dims.num_q_heads
                  << " kvh=" << dims.num_kv_heads << ")"
                  << " NS=" << NS << "\n";

        size_t q_count = dims.num_q_heads * sl * dims.head_dim;
        size_t kv_count = dims.num_kv_heads * kvl * dims.head_dim;

        std::vector<std::vector<float>> q_set(NS), k_set(NS), v_set(NS);
        for (size_t s = 0; s < NS; ++s) {
            q_set[s].resize(q_count);
            k_set[s].resize(kv_count);
            v_set[s].resize(kv_count);
            for (auto& x : q_set[s]) x = dist(gen);
            for (auto& x : k_set[s]) x = dist(gen);
            for (auto& x : v_set[s]) x = dist(gen);
        }

        std::vector<float> reference(q_count);
        reference_attention_fp32(q_set[0].data(), k_set[0].data(), v_set[0].data(),
                                 reference.data(),
                                 dims.num_q_heads, dims.num_kv_heads,
                                 sl, kvl, dims.head_dim, scale);

        std::vector<AttnSlot> slots;
        for (const auto* backend : active) {
            if (backend->mode != cfg.mode) continue;
            slots.emplace_back();
            auto& slot = slots.back();
            slot.backend = backend;
            slot.states.reserve(NS);
            slot.prepared = true;

            for (size_t s = 0; s < NS; ++s) {
                void* st = backend->prepare(dims, sl, cfg.cache_len,
                                             q_set[s].data(), k_set[s].data(), v_set[s].data());
                if (!st) { slot.prepared = false; break; }
                slot.states.push_back(st);
            }

            if (!slot.prepared) {
                std::cout << "  " << backend->name << "  prepare=FAIL\n";
                for (auto* st : slot.states) backend->cleanup(st);
                slot.states.clear();
                continue;
            }

            std::vector<float> captured(q_count, 0.0f);
            backend->run(slot.states[0], captured.data());
            slot.acc = check_accuracy(reference.data(), captured.data(), q_count,
                                       attn_tolerance(backend->name));
        }

        std::vector<std::vector<float>>().swap(q_set);
        std::vector<std::vector<float>>().swap(k_set);
        std::vector<std::vector<float>>().swap(v_set);

        for (auto& slot : slots) {
            if (!slot.prepared) continue;
            double total = 0.0;
            for (int p = 0; p < kProbeIters; ++p) {
                size_t idx = static_cast<size_t>(p) % NS;
                double t0 = now_ms();
                slot.backend->run(slot.states[idx], nullptr);
                total += now_ms() - t0;
            }
            slot.probe_avg_ms = total / kProbeIters;
        }
        double cactus_probe_ms = -1.0;
        for (const auto& slot : slots) {
            if (slot.prepared && std::string(slot.backend->framework) == "cactus") {
                cactus_probe_ms = slot.probe_avg_ms;
                break;
            }
        }
        if (cactus_probe_ms > 0.0) {
            for (auto& slot : slots) {
                if (!slot.prepared) continue;
                if (slot.probe_avg_ms > kSlowMultiplier * cactus_probe_ms)
                    slot.skipped_slow = true;
            }
        } else {
            static bool warned = false;
            if (!warned) {
                std::cerr << "[bench] cactus not in active backends — slow-skip "
                             "disabled. Run will not auto-skip outlier-slow "
                             "backends; expect long wall time.\n";
                warned = true;
            }
        }

        for (auto& slot : slots) {
            if (!slot.prepared || !slot.skipped_slow) continue;
            for (int w = 0; w < kSoloWarmup; ++w) {
                size_t idx = static_cast<size_t>(w) % NS;
                slot.backend->run(slot.states[idx], nullptr);
            }
            slot.total_ms = 0.0;
            slot.samples_ms.clear();
            for (int it = 0; it < kSoloIters; ++it) {
                size_t idx = static_cast<size_t>(it) % NS;
                double t0 = now_ms();
                slot.backend->run(slot.states[idx], nullptr);
                double dt = now_ms() - t0;
                slot.total_ms += dt;
                slot.samples_ms.push_back(dt);
            }
        }

        for (int w = 0; w < opt.warmup; ++w) {
            size_t idx = static_cast<size_t>(w) % NS;
            for (size_t n = 0; n < slots.size(); ++n) {
                auto& slot = slots[(static_cast<size_t>(w) + n) % slots.size()];
                if (!slot.prepared || slot.skipped_slow) continue;
                slot.backend->run(slot.states[idx], nullptr);
            }
        }

        for (int it = 0; it < opt.iterations; ++it) {
            size_t idx = static_cast<size_t>(it) % NS;
            for (size_t n = 0; n < slots.size(); ++n) {
                auto& slot = slots[(static_cast<size_t>(it) + n) % slots.size()];
                if (!slot.prepared || slot.skipped_slow) continue;
                double t0 = now_ms();
                slot.backend->run(slot.states[idx], nullptr);
                double dt = now_ms() - t0;
                slot.total_ms += dt;
                slot.samples_ms.push_back(dt);
            }
        }

        for (auto& slot : slots) {
            if (!slot.prepared) continue;
            int iters = slot.samples_ms.empty() ? 1 : static_cast<int>(slot.samples_ms.size());
            double avg_ms = slot.total_ms / iters;
            double avg_us = avg_ms * 1000.0;
            double p50_us = median_ms(slot.samples_ms) * 1000.0;
            double gflops = compute_attention_gflops(dims.num_q_heads, sl, kvl,
                                                      dims.head_dim, iters, slot.total_ms);

            std::cout << "  " << std::left << std::setw(28) << slot.backend->name
                      << " mean=" << std::fixed << std::setprecision(3) << avg_ms << " ms"
                      << " p50=" << std::setprecision(3) << (p50_us / 1000.0) << " ms"
                      << " (" << std::setprecision(2) << gflops << " GFLOPS)"
                      << "  acc=" << (slot.acc.passed ? "PASS" : "FAIL")
                      << " nrmse=" << std::setprecision(4) << slot.acc.nrmse;
            if (slot.skipped_slow)
                std::cout << "  [slow — solo " << kSoloIters << "-iter pass]";
            std::cout << "\n";

            if (csv.is_open()) {
                csv << attn_graph_name(cfg.graph) << ","
                    << cfg.sweep_dim << "," << sl << "," << cfg.cache_len << ","
                    << dims.head_dim << "," << dims.num_q_heads << "," << dims.num_kv_heads << ","
                    << slot.backend->name << "," << slot.backend->framework << ","
                    << std::fixed << std::setprecision(3) << avg_us << ","
                    << std::setprecision(3) << p50_us << ","
                    << std::setprecision(3) << gflops << ","
                    << std::setprecision(6) << slot.acc.nrmse << ","
                    << std::setprecision(6) << slot.acc.max_abs_error << "\n";
                csv.flush();
            }

            for (auto* st : slot.states) slot.backend->cleanup(st);
        }
    }

    set_thread_override(0);
    return true;
}

} // namespace bench
