#include "test_utils.h"
#include "vulkan_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using namespace EngineTestUtils;

bool test_elementwise() {
    const size_t n = 1000;
    auto a = rand_halfs(n), b = rand_halfs(n);
    std::vector<__fp16> got(n);
    std::vector<float> want(n);

    if (!cactus_vulkan_binary_f16(0, got.data(), a.data(), b.data(), n)) return false;
    for (size_t i = 0; i < n; ++i) want[i] = (float)a[i] + (float)b[i];
    if (!close_all(got, want, 5e-3f, "binary add")) return false;

    if (!cactus_vulkan_binary_f16(3, got.data(), a.data(), b.data(), n)) return false;
    for (size_t i = 0; i < n; ++i) want[i] = (float)a[i] * (float)b[i];
    if (!close_all(got, want, 5e-3f, "binary mul")) return false;

    if (!cactus_vulkan_scalar_f16(2, got.data(), a.data(), n, 1.7f)) return false;
    for (size_t i = 0; i < n; ++i) want[i] = (float)a[i] * 1.7f;
    if (!close_all(got, want, 5e-3f, "scalar mul")) return false;

    if (!cactus_vulkan_unary_f16(0, got.data(), a.data(), n)) return false;
    for (size_t i = 0; i < n; ++i) want[i] = gelu_ref((float)a[i]);
    if (!close_all(got, want, 5e-3f, "gelu")) return false;

    if (!cactus_vulkan_unary_f16(2, got.data(), a.data(), n)) return false;
    for (size_t i = 0; i < n; ++i) want[i] = (float)a[i] / (1.0f + std::exp(-(float)a[i]));
    if (!close_all(got, want, 5e-3f, "silu")) return false;

    auto big = rand_halfs(n, -200.0f, 200.0f);
    if (!cactus_vulkan_unary_f16(0, got.data(), big.data(), n)) return false;
    for (size_t i = 0; i < n; ++i) want[i] = gelu_ref((float)big[i]);
    if (!close_all(got, want, 2e-1f, "gelu large")) return false;

    if (!cactus_vulkan_unary_f16(1, got.data(), big.data(), n)) return false;
    for (size_t i = 0; i < n; ++i) want[i] = std::tanh((float)big[i]);
    if (!close_all(got, want, 5e-3f, "tanh large")) return false;

    if (!cactus_vulkan_unary_f16(2, got.data(), big.data(), n)) return false;
    for (size_t i = 0; i < n; ++i) { float v = (float)big[i]; want[i] = v / (1.0f + std::exp(-v)); }
    if (!close_all(got, want, 2e-1f, "silu large")) return false;

    return true;
}

bool test_swiglu() {
    const size_t n = 777;
    const float scale = 0.6f;
    auto gate = rand_halfs(n), up = rand_halfs(n);
    std::vector<__fp16> got(n);
    std::vector<float> want(n);
    if (!cactus_vulkan_swiglu_f16(got.data(), gate.data(), up.data(), n, scale)) return false;
    for (size_t i = 0; i < n; ++i) {
        __fp16 g1 = (__fp16)gelu_ref((float)gate[i]);
        __fp16 g2 = (__fp16)((float)g1 * scale);
        want[i] = (float)g2 * (float)up[i];
    }
    if (!close_all(got, want, 5e-3f, "swiglu")) return false;

    auto bg = rand_halfs(n, -200.0f, 200.0f);
    if (!cactus_vulkan_swiglu_f16(got.data(), bg.data(), up.data(), n, scale)) return false;
    for (size_t i = 0; i < n; ++i) {
        __fp16 g1 = (__fp16)gelu_ref((float)bg[i]);
        __fp16 g2 = (__fp16)((float)g1 * scale);
        want[i] = (float)g2 * (float)up[i];
    }
    return close_all(got, want, 2e-1f, "swiglu large");
}

bool test_rms_norm() {
    const size_t rows = 7, dim = 320;
    const float eps = 1e-6f;
    auto x = rand_halfs(rows * dim);
    auto w = rand_halfs(dim, 0.5f, 1.5f);
    std::vector<__fp16> got(rows * dim);
    std::vector<float> want(rows * dim);
    if (!cactus_vulkan_rms_norm_f16(got.data(), x.data(), w.data(), rows, dim, eps)) return false;
    for (size_t r = 0; r < rows; ++r) {
        double ss = 0;
        for (size_t i = 0; i < dim; ++i) { double v = (float)x[r * dim + i]; ss += v * v; }
        float inv = 1.0f / std::sqrt((float)(ss / dim) + eps);
        for (size_t i = 0; i < dim; ++i)
            want[r * dim + i] = (float)x[r * dim + i] * inv * (float)w[i];
    }
    return close_all(got, want, 2e-2f, "rms_norm");
}

static bool run_cq_gemv(bool interleaved, int iters, const char* what, uint32_t gs = 128) {
    CQ4Fixture f(interleaved, 512, 64, gs);
    auto x = rand_halfs(f.K);
    std::vector<__fp16> got(f.N);
    if (!cactus_vulkan_cq_gemv(got.data(), x.data(), &f.W, iters)) {
        std::cerr << "  [✗] " << what << " encode failed\n";
        return false;
    }
    auto want = f.oracle(x);
    float maxy = 0;
    for (float v : want) maxy = std::max(maxy, std::fabs(v));
    return close_all(got, want, std::max(5e-2f, 0.02f * maxy), what);
}

bool test_cq4_gemv() { return run_cq_gemv(false, 1, "cq4_gemv"); }
bool test_cq4_gemv_interleaved() { return run_cq_gemv(true, 1, "cq4_gemv interleaved"); }
bool test_cq4_gemv_iter_loop() { return run_cq_gemv(false, 8, "cq4_gemv iters=8"); }
bool test_cq4_gemv_gs64() { return run_cq_gemv(false, 1, "cq4_gemv gs=64", 64); }

static bool oracle_matches_cpu(bool interleaved, const char* what) {
    CQ4Fixture f(interleaved);
    auto x = rand_halfs(f.K);
    std::vector<__fp16> cpu(f.N);
    cactus_quant_matmul(&f.W, x.data(), 1, cpu.data());
    auto want = f.oracle(x);
    float maxy = 0;
    for (float v : want) maxy = std::max(maxy, std::fabs(v));
    return close_all(cpu, want, std::max(5e-2f, 0.02f * maxy), what);
}

bool test_oracle_vs_cpu() {
    return oracle_matches_cpu(false, "oracle vs cpu flat")
        && oracle_matches_cpu(true, "oracle vs cpu interleaved");
}

static bool gpu_matches_cpu(bool interleaved, const char* what) {
    CQ4Fixture f(interleaved);
    auto x = rand_halfs(f.K);
    std::vector<__fp16> cpu(f.N), gpu(f.N);
    cactus_quant_matmul(&f.W, x.data(), 1, cpu.data());
    if (!cactus_vulkan_cq_gemv(gpu.data(), x.data(), &f.W)) {
        std::cerr << "  [✗] " << what << " encode failed\n";
        return false;
    }
    std::vector<float> want(f.N);
    for (uint32_t i = 0; i < f.N; ++i) want[i] = (float)cpu[i];
    float maxy = 0;
    for (float v : want) maxy = std::max(maxy, std::fabs(v));
    return close_all(gpu, want, std::max(5e-2f, 0.02f * maxy), what);
}

bool test_cq4_gemv_vs_cpu() {
    return gpu_matches_cpu(false, "gpu vs cpu flat")
        && gpu_matches_cpu(true, "gpu vs cpu interleaved");
}

bool test_real_weights_gemv() {
    const char* bundle = std::getenv("CACTUS_TEST_MODEL");
    if (!bundle || !*bundle) {
        std::cout << "  [-] CACTUS_TEST_MODEL not set, skipping\n";
        return true;
    }
    const char* names[] = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"};
    const int layers[] = {0, 1, 20, 34};
    bool all = true;
    size_t tested = 0;
    std::vector<std::string> paths;
    for (int l : layers)
        for (const char* n : names)
            paths.push_back(std::string(bundle) + "/layer_" + std::to_string(l) + "_" + n + ".weights");
    paths.push_back(std::string(bundle) + "/token_embeddings.weights");
    for (const std::string& path : paths) {
        {
            CactusGraph g;
            size_t id;
            try {
                id = g.mmap_weights(path);
            } catch (const std::exception&) {
                continue;
            }
            const BufferDesc& b = g.get_output_buffer(id);
            if (!PrecisionTraits::is_cq(b.precision) || b.group_size == 0) continue;
            CactusQuantMatrix W = b.to_cq_matrix();
            std::string base = path.substr(path.find_last_of('/') + 1);
            auto x = rand_halfs(W.K);
            std::vector<__fp16> cpu(W.N), gpu(W.N);
            cactus_quant_matmul(&W, x.data(), 1, cpu.data());
            if (!cactus_vulkan_cq_gemv(gpu.data(), x.data(), &W)) {
                std::cout << "  [-] " << base << " encode refused (bits=" << W.bits
                          << " K=" << W.K << " N=" << W.N << " gs=" << W.group_size
                          << " flags=" << W.flags << ")\n";
                continue;
            }
            std::vector<float> want(W.N);
            float maxy = 0;
            for (uint32_t i = 0; i < W.N; ++i) {
                want[i] = (float)cpu[i];
                maxy = std::max(maxy, std::fabs(want[i]));
            }
            char what[160];
            std::snprintf(what, sizeof(what), "%s K=%u N=%u gs=%u fl=%u",
                          base.c_str(), W.K, W.N, W.group_size, W.flags);
            if (!close_all(gpu, want, std::max(5e-2f, 0.02f * maxy), what)) all = false;
            ++tested;
        }
    }
    if (tested == 0) {
        std::cerr << "  [✗] no CQ weight files found under " << bundle << "\n";
        return false;
    }
    std::cout << "  [i] " << tested << " real matrices compared\n";
    return all;
}

bool test_graph_ew_parity() {
    const size_t dim = 2048;
    auto x = rand_halfs(dim);
    auto run = [&](const char* backend, std::vector<__fp16>& out) {
        cactus_backend_select(backend);
        CactusGraph g;
        size_t in = g.input({1, dim}, Precision::FP16);
        g.set_external_input(in, (void*)x.data(), Precision::FP16);
        size_t h = in;
        for (int i = 0; i < 60; ++i) {
            size_t a = g.scalar_multiply(h, 1.007f);
            size_t b = g.gelu(a);
            size_t c = g.multiply(b, h);
            h = g.add(h, c);
        }
        g.execute();
        std::memcpy(out.data(), g.get_output(h), dim * 2);
    };
    std::vector<__fp16> cpu(dim), gpu(dim);
    run("cpu", cpu);
    run("vulkan", gpu);
    cactus_backend_select("auto");
    std::vector<float> want(dim);
    float maxy = 0;
    for (size_t i = 0; i < dim; ++i) {
        want[i] = (float)cpu[i];
        maxy = std::max(maxy, std::fabs(want[i]));
    }
    return close_all(gpu, want, std::max(8e-2f, 0.02f * maxy), "graph ew parity");
}

bool test_qkv_chain() {
    CQ4Fixture fq(false), fk(false, 512, 96, 128), fv(true, 512, 64, 128);
    auto x = rand_halfs(fq.K);
    cactus_vulkan_session_begin();
    auto* px = (__fp16*)cactus_vulkan_alloc_shared(fq.K * 2);
    auto* pq = (__fp16*)cactus_vulkan_alloc_shared(fq.N * 2);
    auto* pk = (__fp16*)cactus_vulkan_alloc_shared(fk.N * 2);
    auto* pv = (__fp16*)cactus_vulkan_alloc_shared(fv.N * 2);
    if (!px || !pq || !pk || !pv) return false;
    std::memcpy(px, x.data(), fq.K * 2);
    bool ok = cactus_vulkan_encode_cq_gemv(pq, px, &fq.W)
           && cactus_vulkan_encode_cq_gemv(pk, px, &fk.W)
           && cactus_vulkan_encode_cq_gemv(pv, px, &fv.W);
    cactus_vulkan_session_sync();
    std::vector<__fp16> gq(fq.N), gk(fk.N), gv(fv.N);
    if (ok) {
        std::memcpy(gq.data(), pq, fq.N * 2);
        std::memcpy(gk.data(), pk, fk.N * 2);
        std::memcpy(gv.data(), pv, fv.N * 2);
    }
    for (void* p : {(void*)px, (void*)pq, (void*)pk, (void*)pv})
        cactus_vulkan_free_shared(p);
    cactus_vulkan_session_end();
    if (!ok) {
        std::cerr << "  [✗] qkv chain encode failed\n";
        return false;
    }
    auto check = [&](const std::vector<__fp16>& got, const CQ4Fixture& f, const char* what) {
        auto want = f.oracle(x);
        float maxy = 0;
        for (float v : want) maxy = std::max(maxy, std::fabs(v));
        return close_all(got, want, std::max(5e-2f, 0.02f * maxy), what);
    };
    return check(gq, fq, "qkv chain q") && check(gk, fk, "qkv chain k") && check(gv, fv, "qkv chain v");
}

bool test_gemv_large_n() {
    CQ4Fixture f(false, 128, 262146, 128);
    auto x = rand_halfs(f.K);
    std::vector<__fp16> got(f.N);
    if (!cactus_vulkan_cq_gemv(got.data(), x.data(), &f.W)) {
        std::cerr << "  [✗] large-N gemv encode failed\n";
        return false;
    }
    auto want = f.oracle(x);
    float maxy = 0;
    for (float v : want) maxy = std::max(maxy, std::fabs(v));
    return close_all(got, want, std::max(5e-2f, 0.02f * maxy), "cq4_gemv N=262146");
}

bool test_session_chain() {
    const size_t n = 512;
    CQ4Fixture f(false);
    auto a = rand_halfs(n), b = rand_halfs(n);
    auto w = rand_halfs(n, 0.5f, 1.5f);

    cactus_vulkan_session_begin();
    auto* pa = (__fp16*)cactus_vulkan_alloc_shared(n * 2);
    auto* pb = (__fp16*)cactus_vulkan_alloc_shared(n * 2);
    auto* pw = (__fp16*)cactus_vulkan_alloc_shared(n * 2);
    auto* t1 = (__fp16*)cactus_vulkan_alloc_pooled(n * 2);
    auto* t2 = (__fp16*)cactus_vulkan_alloc_pooled(n * 2);
    auto* t3 = (__fp16*)cactus_vulkan_alloc_pooled(n * 2);
    auto* t4 = (__fp16*)cactus_vulkan_alloc_pooled(n * 2);
    auto* py = (__fp16*)cactus_vulkan_alloc_shared(f.N * 2);
    if (!pa || !pb || !pw || !t1 || !t2 || !t3 || !t4 || !py) return false;
    std::memcpy(pa, a.data(), n * 2);
    std::memcpy(pb, b.data(), n * 2);
    std::memcpy(pw, w.data(), n * 2);

    bool ok = cactus_vulkan_encode_binary_f16(0, t1, pa, pb, n);
    ok = ok && cactus_vulkan_encode_scalar_f16(2, t2, t1, n, 0.5f);
    cactus_vulkan_session_flush();
    ok = ok && cactus_vulkan_encode_unary_f16(2, t3, t2, n);
    ok = ok && cactus_vulkan_encode_rms_norm_f16(t4, t3, pw, 1, n, 1e-6f);
    ok = ok && cactus_vulkan_encode_cq_gemv(py, t4, &f.W);
    cactus_vulkan_session_sync();

    std::vector<__fp16> got(f.N);
    if (ok) std::memcpy(got.data(), py, f.N * 2);
    for (void* p : {(void*)pa, (void*)pb, (void*)pw, (void*)t1, (void*)t2, (void*)t3, (void*)t4, (void*)py})
        cactus_vulkan_free_shared(p);
    cactus_vulkan_session_end();
    if (!ok) {
        std::cerr << "  [✗] session chain encode failed\n";
        return false;
    }

    std::vector<__fp16> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = (__fp16)((float)a[i] + (float)b[i]);
    for (size_t i = 0; i < n; ++i) h[i] = (__fp16)((float)h[i] * 0.5f);
    for (size_t i = 0; i < n; ++i) {
        float v = (float)h[i];
        h[i] = (__fp16)(v / (1.0f + std::exp(-v)));
    }
    double ss = 0;
    for (size_t i = 0; i < n; ++i) { double v = (float)h[i]; ss += v * v; }
    float inv = 1.0f / std::sqrt((float)(ss / n) + 1e-6f);
    std::vector<__fp16> hn(n);
    for (size_t i = 0; i < n; ++i) hn[i] = (__fp16)((float)h[i] * inv * (float)w[i]);
    auto want = f.oracle(hn);
    float maxy = 0;
    for (float v : want) maxy = std::max(maxy, std::fabs(v));
    return close_all(got, want, std::max(5e-2f, 0.03f * maxy), "session chain");
}

struct VkBuf {
    void* p = nullptr;
    VkBuf(size_t bytes, const void* src = nullptr) {
        p = cactus_vulkan_alloc_shared(bytes);
        if (p && src) std::memcpy(p, src, bytes);
    }
    ~VkBuf() { if (p) cactus_vulkan_free_shared(p); }
};

bool test_phase1_simple_ops() {
    const size_t n = 4096;
    auto x = rand_halfs(n, -4.0f, 4.0f);
    VkBuf bx(n * 2, x.data()), by(n * 2), bf(n * 4), bh(n * 2);
    if (!bx.p || !by.p || !bf.p || !bh.p) return false;

    if (!cactus_vulkan_encode_copy(by.p, bx.p, n * 2)) return false;
    cactus_vulkan_session_sync();
    if (std::memcmp(by.p, bx.p, n * 2) != 0) { std::cerr << "  [✗] copy mismatch\n"; return false; }

    if (!cactus_vulkan_encode_cast(bf.p, 2, bx.p, 1, n)) return false;
    if (!cactus_vulkan_encode_cast(bh.p, 1, bf.p, 2, n)) return false;
    cactus_vulkan_session_sync();
    if (std::memcmp(bh.p, bx.p, n * 2) != 0) { std::cerr << "  [✗] cast roundtrip mismatch\n"; return false; }

    if (!cactus_vulkan_encode_softcap(by.p, bx.p, n, 30.0f)) return false;
    cactus_vulkan_session_sync();
    std::vector<float> want(n);
    for (size_t i = 0; i < n; ++i) {
        float v1 = (float)(__fp16)((float)x[i] / 30.0f);
        float v2 = (float)(__fp16)std::tanh(v1);
        want[i] = v2 * 30.0f;
    }
    std::vector<__fp16> got(n);
    std::memcpy(got.data(), by.p, n * 2);
    if (!close_all(got, want, 5e-2f, "softcap")) return false;

    const uint32_t R = 32, C = 128;
    uint32_t oshape[2] = {C, R}, sstride[2] = {1, C};
    if (!cactus_vulkan_encode_strided_copy(by.p, bx.p, oshape, sstride, 2, R * C, 0, R * C * 2, R * C * 2)) return false;
    cactus_vulkan_session_sync();
    std::memcpy(got.data(), by.p, R * C * 2);
    bool tok = true;
    for (uint32_t r = 0; r < R && tok; ++r)
        for (uint32_t c = 0; c < C; ++c)
            if ((float)got[c * R + r] != (float)x[r * C + c]) { tok = false; break; }
    if (!tok) { std::cerr << "  [✗] strided_copy transpose mismatch\n"; return false; }

    const uint32_t outer = 4, aa = 3, ba = 5, inner = 32;
    auto a2 = rand_halfs(outer * aa * inner), b2 = rand_halfs(outer * ba * inner);
    VkBuf ba2(a2.size() * 2, a2.data()), bb2(b2.size() * 2, b2.data()), bo2(outer * (aa + ba) * inner * 2);
    if (!ba2.p || !bb2.p || !bo2.p) return false;
    if (!cactus_vulkan_encode_concat2(bo2.p, ba2.p, bb2.p, outer, outer, aa, ba, inner)) return false;
    cactus_vulkan_session_sync();
    const __fp16* oc = (const __fp16*)bo2.p;
    for (uint32_t u = 0; u < outer; ++u)
        for (uint32_t ax = 0; ax < aa + ba; ++ax)
            for (uint32_t e = 0; e < inner; ++e) {
                float wv = ax < aa ? (float)a2[(u * aa + ax) * inner + e]
                                   : (float)b2[(u * ba + (ax - aa)) * inner + e];
                if ((float)oc[(u * (aa + ba) + ax) * inner + e] != wv) {
                    std::cerr << "  [✗] concat2 mismatch\n";
                    return false;
                }
            }
    return true;
}

bool test_rms_fused() {
    const size_t rows = 3, dim = 2048;
    const float eps = 1e-6f;
    auto x = rand_halfs(rows * dim), r = rand_halfs(rows * dim);
    auto w1 = rand_halfs(dim, 0.5f, 1.5f), w2 = rand_halfs(dim, 0.5f, 1.5f);
    VkBuf bx(rows * dim * 2, x.data()), br(rows * dim * 2, r.data());
    VkBuf bw1(dim * 2, w1.data()), bw2(dim * 2, w2.data());
    VkBuf bh(rows * dim * 2), bxn(rows * dim * 2);
    if (!bx.p || !br.p || !bw1.p || !bw2.p || !bh.p || !bxn.p) return false;
    const float oscale = 0.75f;
    if (!cactus_vulkan_encode_rms_norm_add_rms(bh.p, bxn.p, bx.p, bw1.p, br.p, bw2.p, rows, dim, eps, oscale))
        return false;
    cactus_vulkan_session_sync();
    std::vector<float> wanth(rows * dim), wantxn(rows * dim);
    for (size_t row = 0; row < rows; ++row) {
        double ss = 0;
        for (size_t i = 0; i < dim; ++i) { double v = (float)x[row * dim + i]; ss += v * v; }
        float inv = 1.0f / std::sqrt((float)(ss / dim) + eps);
        double ss2 = 0;
        for (size_t i = 0; i < dim; ++i) {
            float rr = ((float)r[row * dim + i]
                        + (float)(__fp16)((float)x[row * dim + i] * inv * (float)w1[i])) * oscale;
            float hv = (float)(__fp16)std::max(-65500.0f, std::min(65500.0f, rr));
            wanth[row * dim + i] = hv;
            ss2 += hv * hv;
        }
        float inv2 = 1.0f / std::sqrt((float)(ss2 / dim) + eps);
        for (size_t i = 0; i < dim; ++i)
            wantxn[row * dim + i] = wanth[row * dim + i] * inv2 * (float)w2[i];
    }
    std::vector<__fp16> goth(rows * dim), gotxn(rows * dim);
    std::memcpy(goth.data(), bh.p, rows * dim * 2);
    std::memcpy(gotxn.data(), bxn.p, rows * dim * 2);
    if (!close_all(goth, wanth, 3e-2f, "rms_norm_add_rms h")) return false;
    if (!close_all(gotxn, wantxn, 3e-2f, "rms_norm_add_rms xn")) return false;

    if (!cactus_vulkan_encode_rms_norm_add(bh.p, bx.p, bw1.p, br.p, rows, dim, eps, 1.0f)) return false;
    cactus_vulkan_session_sync();
    std::memcpy(goth.data(), bh.p, rows * dim * 2);
    for (size_t row = 0; row < rows; ++row) {
        double ss = 0;
        for (size_t i = 0; i < dim; ++i) { double v = (float)x[row * dim + i]; ss += v * v; }
        float inv = 1.0f / std::sqrt((float)(ss / dim) + eps);
        for (size_t i = 0; i < dim; ++i)
            wanth[row * dim + i] = (float)r[row * dim + i]
                + (float)(__fp16)((float)x[row * dim + i] * inv * (float)w1[i]);
    }
    return close_all(goth, wanth, 3e-2f, "rms_norm_add");
}

bool test_rope_full_gpu() {
    const uint32_t S = 4, H = 2, D = 64, tokens = S * H;
    const float theta = 10000.0f;
    const uint32_t pos0 = 37;
    auto x = rand_halfs(tokens * D);
    VkBuf bx(tokens * D * 2, x.data()), by(tokens * D * 2);
    if (!bx.p || !by.p) return false;
    if (!cactus_vulkan_encode_rope_full(by.p, bx.p, tokens, S, H, D, D, pos0, theta, 0)) return false;
    cactus_vulkan_session_sync();
    std::vector<__fp16> got(tokens * D);
    std::memcpy(got.data(), by.p, tokens * D * 2);
    std::vector<float> want(tokens * D);
    for (uint32_t tok = 0; tok < tokens; ++tok) {
        uint32_t seq = (tok / H) % S;
        std::vector<float> row(D);
        for (uint32_t d = 0; d < D; ++d) row[d] = (float)x[tok * D + d];
        auto o = rope_reference(row, (double)(pos0 + seq), theta);
        for (uint32_t d = 0; d < D; ++d) want[tok * D + d] = o[d];
    }
    return close_all(got, want, 3e-2f, "rope_full");
}

bool test_adjust_argmax_gpu() {
    const uint32_t V = 50000;
    auto logits = rand_halfs(V, -8.0f, 8.0f);
    logits[31337] = (__fp16)19.0f;
    logits[777] = (__fp16)18.0f;
    uint32_t recent[3] = {31337, 5, 9};
    VkBuf bl(V * 2, logits.data()), bo(12);
    if (!bl.p || !bo.p) return false;
    if (!cactus_vulkan_encode_adjust_logits(bl.p, V, recent, 3, 42, 1.3f)) return false;
    if (!cactus_vulkan_encode_argmax(bl.p, V, bo.p, nullptr)) return false;
    cactus_vulkan_session_sync();
    std::vector<float> ref(V);
    for (uint32_t i = 0; i < V; ++i) ref[i] = (float)logits[i];
    for (uint32_t id : recent) ref[id] = ref[id] > 0 ? ref[id] / 1.3f : ref[id] * 1.3f;
    ref[42] = -65504.0f;
    float b = -1e30f, s = -1e30f;
    uint32_t bi = 0;
    for (uint32_t i = 0; i < V; ++i) {
        float v = (float)(__fp16)ref[i];
        if (v > b) { s = b; b = v; bi = i; }
        else if (v > s) s = v;
    }
    const float* o3 = (const float*)bo.p;
    if ((uint32_t)o3[2] != bi || std::fabs(o3[0] - b) > 1e-2f || std::fabs(o3[1] - s) > 1e-2f) {
        std::cerr << "  [✗] adjust+argmax got idx " << (uint32_t)o3[2] << " best " << o3[0]
                  << " second " << o3[1] << " want " << bi << "/" << b << "/" << s << "\n";
        return false;
    }
    return true;
}

static bool run_attn_case(uint32_t T, const char* what) {
    const uint32_t nqh = 8, nkvh = 2, hd = 64, vhd = 64, gs = 32;
    const uint32_t ngK = hd / gs;
    const float scale = 1.0f / std::sqrt((float)hd);
    auto kdata = rand_halfs((size_t)T * nkvh * hd), vdata = rand_halfs((size_t)T * nkvh * vhd);
    auto q = rand_halfs(nqh * hd);
    auto knew = rand_halfs(nkvh * hd), vnew = rand_halfs(nkvh * vhd);
    size_t i8b = (size_t)(T + 1) * nkvh * hd, scb = (size_t)(T + 1) * nkvh * ngK * 4;
    VkBuf cache(64 + i8b + scb);
    VkBuf bkn(nkvh * hd * 2, knew.data()), bvn(nkvh * vhd * 2, vnew.data());
    VkBuf bq(nqh * hd * 2, q.data()), bo(nqh * vhd * 2);
    VkBuf bksrc((size_t)T * nkvh * hd * 2, kdata.data());
    VkBuf vcache(64 + i8b + scb);
    VkBuf bvsrc((size_t)T * nkvh * vhd * 2, vdata.data());
    if (!cache.p || !vcache.p || !bkn.p || !bvn.p || !bq.p || !bo.p || !bksrc.p || !bvsrc.p) return false;
    char* kb = (char*)cache.p;
    char* vb = (char*)vcache.p;
    if (!cactus_vulkan_encode_kv_append_i8(bksrc.p, kb + 64, kb + 64 + i8b, nkvh, hd, 0, gs, T, 0, 0,
            (size_t)T * nkvh * hd * 2, i8b, scb)) return false;
    if (!cactus_vulkan_encode_kv_append_i8(bvsrc.p, vb + 64, vb + 64 + i8b, nkvh, vhd, 0, gs, T, 0, 0,
            (size_t)T * nkvh * vhd * 2, i8b, scb)) return false;
    if (!cactus_vulkan_encode_attention_i8(bo.p, bq.p, bkn.p, bvn.p,
            kb + 64, vb + 64, kb + 64 + i8b, vb + 64 + i8b,
            nqh, nkvh, hd, vhd, T, T + 1, 0, T + 1, scale, i8b, i8b, scb, scb)) return false;
    cactus_vulkan_session_sync();

    auto quant = [&](const std::vector<__fp16>& src, uint32_t width, std::vector<float>& deq) {
        deq.resize((size_t)T * nkvh * width);
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t h = 0; h < nkvh; ++h)
                for (uint32_t g = 0; g < width / gs; ++g) {
                    float ma = 0;
                    size_t base = ((size_t)t * nkvh + h) * width + g * gs;
                    for (uint32_t k = 0; k < gs; ++k) ma = std::max(ma, std::fabs((float)src[base + k]));
                    float sc = std::max(ma / 127.0f, 1e-10f);
                    for (uint32_t k = 0; k < gs; ++k) {
                        float qv = std::max(-128.0f, std::min(127.0f, std::round((float)src[base + k] / sc)));
                        deq[base + k] = qv * sc;
                    }
                }
    };
    std::vector<float> kd, vd;
    quant(kdata, hd, kd);
    quant(vdata, vhd, vd);
    std::vector<float> want(nqh * vhd);
    for (uint32_t h = 0; h < nqh; ++h) {
        uint32_t kvh = h / (nqh / nkvh);
        std::vector<float> sc(T + 1);
        float m = -1e30f;
        for (uint32_t k = 0; k <= T; ++k) {
            float dot = 0;
            for (uint32_t d = 0; d < hd; ++d) {
                float kvv = k < T ? kd[((size_t)k * nkvh + kvh) * hd + d]
                                  : (float)knew[kvh * hd + d];
                dot += (float)q[h * hd + d] * kvv;
            }
            sc[k] = dot * scale;
            m = std::max(m, sc[k]);
        }
        float l = 0;
        for (uint32_t k = 0; k <= T; ++k) { sc[k] = std::exp(sc[k] - m); l += sc[k]; }
        for (uint32_t d = 0; d < vhd; ++d) {
            float acc = 0;
            for (uint32_t k = 0; k <= T; ++k) {
                float vv = k < T ? vd[((size_t)k * nkvh + kvh) * vhd + d]
                                 : (float)vnew[kvh * vhd + d];
                acc += sc[k] * vv;
            }
            want[h * vhd + d] = acc / l;
        }
    }
    std::vector<__fp16> got(nqh * vhd);
    std::memcpy(got.data(), bo.p, nqh * vhd * 2);
    float maxy = 0;
    for (float v : want) maxy = std::max(maxy, std::fabs(v));
    return close_all(got, want, std::max(3e-2f, 0.03f * maxy), what);
}

bool test_kv_attn_gpu() {
    return run_attn_case(40, "attn decode T=40 nwg=1")
        && run_attn_case(900, "attn decode T=900 combine");
}

bool test_kv_ring_slots() {
    const uint32_t kvh = 1, hd = 32, gs = 32, sink = 4, W = 16;
    const uint32_t cur = 20;
    auto src = rand_halfs(kvh * hd, 0.5f, 1.0f);
    size_t i8b = (size_t)(W + 8) * kvh * hd, scb = (size_t)(W + 8) * kvh * 4;
    VkBuf cache(64 + i8b + scb);
    VkBuf bsrc(kvh * hd * 2, src.data());
    if (!cache.p || !bsrc.p) return false;
    std::memset(cache.p, 0, 64 + i8b + scb);
    char* kb = (char*)cache.p;
    if (!cactus_vulkan_encode_kv_append_i8(bsrc.p, kb + 64, kb + 64 + i8b, kvh, hd, cur, gs, 1, sink, W,
            kvh * hd * 2, i8b, scb)) return false;
    cactus_vulkan_session_sync();
    uint32_t R = W - sink;
    uint32_t slot = sink + ((cur - sink) % R);
    const float* scales = (const float*)(kb + 64 + i8b);
    for (uint32_t s = 0; s < W + 8; ++s) {
        bool wrote = scales[s] != 0.0f;
        if (wrote != (s == slot)) {
            std::cerr << "  [✗] ring slot " << s << (wrote ? " unexpectedly written" : " missing write")
                      << " (want slot " << slot << ")\n";
            return false;
        }
    }
    return true;
}

bool test_kv_sliding() {
    const uint32_t kvh = 2, hd = 64, gs = 32, T = 8;
    const uint32_t keep_sink = 2, remaining = 4, shift_src = 4;
    auto tok = rand_halfs((size_t)T * kvh * hd, 0.25f, 1.0f);
    auto newtok = rand_halfs(kvh * hd, 0.25f, 1.0f);
    size_t i8s = (size_t)kvh * hd, scs = (size_t)kvh * (hd / gs);
    size_t i8b = (size_t)(T + 2) * i8s, scb = (size_t)(T + 2) * scs * 4;
    VkBuf cache(64 + i8b + scb), bsrc(tok.size() * 2, tok.data()), bnew(newtok.size() * 2, newtok.data());
    if (!cache.p || !bsrc.p || !bnew.p) return false;
    char* kb = (char*)cache.p;
    if (!cactus_vulkan_encode_kv_append_i8(bsrc.p, kb + 64, kb + 64 + i8b, kvh, hd, 0, gs, T, 0, 0,
            tok.size() * 2, i8b, scb)) return false;
    cactus_vulkan_session_sync();
    std::vector<char> before(i8b);
    std::memcpy(before.data(), kb + 64, i8b);
    std::vector<float> sbefore((T + 2) * scs);
    std::memcpy(sbefore.data(), kb + 64 + i8b, (T + 2) * scs * 4);
    if (!cactus_vulkan_encode_kv_append_sliding_i8(bnew.p, kb + 64, kb + 64 + i8b, kvh, hd,
            keep_sink, remaining, shift_src, gs, 1, newtok.size() * 2, i8b, scb)) return false;
    cactus_vulkan_session_sync();
    const char* after = kb + 64;
    const float* safter = (const float*)(kb + 64 + i8b);
    for (uint32_t s = 0; s < keep_sink; ++s)
        if (std::memcmp(after + s * i8s, before.data() + s * i8s, i8s) != 0) {
            std::cerr << "  [✗] sliding sink slot " << s << " changed\n";
            return false;
        }
    for (uint32_t s = 0; s < remaining; ++s) {
        if (std::memcmp(after + (keep_sink + s) * i8s, before.data() + (shift_src + s) * i8s, i8s) != 0) {
            std::cerr << "  [✗] sliding shift slot " << keep_sink + s << " mismatch\n";
            return false;
        }
        for (uint32_t g = 0; g < scs; ++g)
            if (safter[(keep_sink + s) * scs + g] != sbefore[(shift_src + s) * scs + g]) {
                std::cerr << "  [✗] sliding scale slot " << keep_sink + s << " mismatch\n";
                return false;
            }
    }
    if (safter[(keep_sink + remaining) * scs] == 0.0f) {
        std::cerr << "  [✗] sliding append slot missing\n";
        return false;
    }
    return true;
}

bool test_argmax() {
    const uint32_t n = 50000;
    auto logits = rand_halfs(n, -8.0f, 8.0f);
    std::mt19937 rng(21);
    std::uniform_int_distribution<uint32_t> pick(0, n - 1);
    uint32_t planted = pick(rng);
    logits[planted] = (__fp16)20.0f;
    uint32_t idx = 0;
    float best = 0;
    if (!cactus_vulkan_argmax_f16(logits.data(), n, &idx, &best)) return false;
    if (idx != planted || std::fabs(best - 20.0f) > 0.1f) {
        std::cerr << "  [✗] argmax got idx " << idx << " best " << best
                  << " want idx " << planted << "\n";
        return false;
    }
    return true;
}

int main() {
    TestUtils::TestRunner runner("Vulkan Tests");
    std::cout << "Device: " << cactus_vulkan_device_info() << "\n";
    runner.run_test("oracle_vs_cpu", test_oracle_vs_cpu());
    if (!cactus_vulkan_available()) {
        runner.log_skip("vulkan", "no usable Vulkan GPU on this host");
        runner.print_summary();
        return runner.all_passed() ? 0 : 1;
    }
    runner.run_test("elementwise", test_elementwise());
    runner.run_test("swiglu", test_swiglu());
    runner.run_test("rms_norm", test_rms_norm());
    runner.run_test("cq4_gemv", test_cq4_gemv());
    runner.run_test("cq4_gemv_interleaved", test_cq4_gemv_interleaved());
    runner.run_test("cq4_gemv_iter_loop", test_cq4_gemv_iter_loop());
    runner.run_test("cq4_gemv_gs64", test_cq4_gemv_gs64());
    runner.run_test("cq4_gemv_vs_cpu", test_cq4_gemv_vs_cpu());
    runner.run_test("real_weights_gemv", test_real_weights_gemv());
    runner.run_test("graph_ew_parity", test_graph_ew_parity());
    runner.run_test("qkv_chain", test_qkv_chain());
    runner.run_test("gemv_large_n", test_gemv_large_n());
    runner.run_test("session_chain", test_session_chain());
    runner.run_test("argmax", test_argmax());
    runner.run_test("phase1_simple_ops", test_phase1_simple_ops());
    runner.run_test("rms_fused", test_rms_fused());
    runner.run_test("rope_full", test_rope_full_gpu());
    runner.run_test("adjust_argmax", test_adjust_argmax_gpu());
    runner.run_test("kv_attn", test_kv_attn_gpu());
    runner.run_test("kv_ring_slots", test_kv_ring_slots());
    runner.run_test("kv_sliding", test_kv_sliding());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
