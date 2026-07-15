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

static uint8_t real_weight_idx(const CactusQuantMatrix& W, uint32_t n, uint32_t g, uint32_t e) {
    const uint32_t pgb = W.group_size / 2;
    size_t byte;
    uint32_t shift;
    if (W.flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) {
        uint32_t blk = e >> 4, j = e & 15u, sub = j >> 2, b = j & 3u;
        byte = ((size_t)(n >> 2) * W.num_groups + g) * 4u * pgb
             + (2u * blk + (sub >> 1)) * 16u + (n & 3u) * 4u + b;
        shift = (sub & 1u) * 4u;
    } else {
        byte = ((size_t)n * W.num_groups + g) * pgb + (e >> 1);
        shift = (e & 1u) * 4u;
    }
    return (W.packed_indices[byte] >> shift) & 0xFu;
}

// The CPU kernel's Hadamard transform runs in fp16 and drifts on outlier-heavy
// matrices, so real weights are checked against this double-precision oracle
// (transform in double, codes rounded to fp16 as both backends do) instead.
static std::vector<float> real_weight_oracle(const CactusQuantMatrix& W, const std::vector<__fp16>& x) {
    const uint32_t K = W.K, N = W.N, gs = W.group_size, ng = W.num_groups;
    std::vector<__fp16> code(K);
    std::vector<double> z(gs);
    for (uint32_t g = 0; g < ng; ++g) {
        for (uint32_t k = 0; k < gs; ++k) {
            uint32_t gk = g * gs + k;
            double recip = W.input_scale_recip ? (double)W.input_scale_recip[gk]
                         : W.input_scale ? 1.0 / (double)W.input_scale[gk] : 1.0;
            double ls = W.left_signs ? (double)W.left_signs[k] : 1.0;
            z[k] = (double)x[gk] * recip * ls;
        }
        for (uint32_t h = 1; h < gs; h <<= 1)
            for (uint32_t k = 0; k < gs; ++k)
                if ((k & h) == 0) { double a = z[k], b = z[k + h]; z[k] = a + b; z[k + h] = a - b; }
        double s = 1.0 / std::sqrt((double)gs);
        for (uint32_t k = 0; k < gs; ++k) {
            uint32_t src = W.permutation ? W.permutation[k] : k;
            double rs = W.right_signs ? (double)W.right_signs[src] : 1.0;
            code[g * gs + k] = (__fp16)(z[src] * s * rs);
        }
    }
    std::vector<float> y(N);
    for (uint32_t n = 0; n < N; ++n) {
        double acc = 0;
        for (uint32_t g = 0; g < ng; ++g) {
            double nm = (W.flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW)
                      ? (double)W.norms[(((size_t)(n >> 2) * ng + g) << 2) + (n & 3u)]
                      : (double)W.norms[(size_t)n * ng + g];
            double p = 0;
            for (uint32_t e = 0; e < gs; ++e)
                p += (double)code[g * gs + e] * (double)W.codebook[real_weight_idx(W, n, g, e)];
            acc += nm * p;
        }
        y[n] = (float)acc;
    }
    return y;
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
            std::vector<__fp16> gpu(W.N);
            if (!cactus_vulkan_cq_gemv(gpu.data(), x.data(), &W)) {
                std::cout << "  [-] " << base << " encode refused (bits=" << W.bits
                          << " K=" << W.K << " N=" << W.N << " gs=" << W.group_size
                          << " flags=" << W.flags << ")\n";
                continue;
            }
            std::vector<float> want;
            if (W.flags & CACTUS_QUANT_FLAG_ORTHOGONAL) {
                std::vector<__fp16> cpu(W.N);
                cactus_quant_matmul(&W, x.data(), 1, cpu.data());
                want.resize(W.N);
                for (uint32_t i = 0; i < W.N; ++i) want[i] = (float)cpu[i];
            } else {
                want = real_weight_oracle(W, x);
            }
            float maxy = 0;
            for (uint32_t i = 0; i < W.N; ++i) maxy = std::max(maxy, std::fabs(want[i]));
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

bool test_gemm_f16() {
    bool all = true;
    for (int tr = 0; tr < 2; ++tr) {
        for (uint32_t M : {1u, 7u}) {
            const uint32_t K = 96, N = 160;
            auto a = rand_halfs((size_t)M * K);
            auto w = rand_halfs((size_t)K * N);
            auto* pa = (__fp16*)cactus_vulkan_alloc_shared((size_t)M * K * 2);
            auto* pw = (__fp16*)cactus_vulkan_alloc_shared((size_t)K * N * 2);
            auto* py = (__fp16*)cactus_vulkan_alloc_shared((size_t)M * N * 2);
            if (!pa || !pw || !py) return false;
            std::memcpy(pa, a.data(), (size_t)M * K * 2);
            std::memcpy(pw, w.data(), (size_t)K * N * 2);
            bool ok = cactus_vulkan_encode_gemm_f16(py, pa, pw, M, K, N, tr);
            cactus_vulkan_session_sync();
            std::vector<__fp16> got((size_t)M * N);
            if (ok) std::memcpy(got.data(), py, (size_t)M * N * 2);
            for (void* p : {(void*)pa, (void*)pw, (void*)py}) cactus_vulkan_free_shared(p);
            if (!ok) { std::cerr << "  [-] gemm_f16 encode refused tr=" << tr << " M=" << M << "\n"; return false; }
            std::vector<float> want((size_t)M * N);
            float maxy = 0;
            for (uint32_t m = 0; m < M; ++m)
                for (uint32_t nn = 0; nn < N; ++nn) {
                    double acc = 0;
                    for (uint32_t k = 0; k < K; ++k) {
                        float bv = tr ? (float)w[(size_t)nn * K + k] : (float)w[(size_t)k * N + nn];
                        acc += (float)a[(size_t)m * K + k] * bv;
                    }
                    want[(size_t)m * N + nn] = (float)acc;
                    maxy = std::max(maxy, std::fabs((float)acc));
                }
            char what[64];
            std::snprintf(what, sizeof(what), "gemm_f16 tr=%d M=%u", tr, M);
            if (!close_all(got, want, std::max(5e-2f, 0.02f * maxy), what)) all = false;
        }
    }
    return all;
}

bool test_conv_cache_append() {
    const uint32_t hd = 96, ws = 8;
    bool all = true;
    for (uint32_t nnew : {1u, 3u}) {
        for (int f32 : {0, 1}) {
            uint32_t num_rows = nnew;
            std::vector<__fp16> ring_init = rand_halfs((size_t)ws * hd);
            auto src_h = rand_halfs((size_t)num_rows * hd);
            std::vector<float> src_f(src_h.size());
            for (size_t i = 0; i < src_h.size(); ++i) src_f[i] = (float)src_h[i];
            uint32_t head0 = 5, count = 6;
            uint32_t count_new = std::min(ws, count + num_rows);

            char* cache = (char*)cactus_vulkan_alloc_shared(64 + (size_t)ws * hd * 2);
            auto* psrc = (char*)cactus_vulkan_alloc_shared(src_h.size() * (f32 ? 4 : 2));
            auto* pout = (__fp16*)cactus_vulkan_alloc_shared((size_t)ws * hd * 2);
            if (!cache || !psrc || !pout) return false;
            std::memcpy(cache + 64, ring_init.data(), (size_t)ws * hd * 2);
            if (f32) std::memcpy(psrc, src_f.data(), src_f.size() * 4);
            else std::memcpy(psrc, src_h.data(), src_h.size() * 2);

            bool ok = cactus_vulkan_encode_conv_cache_append(pout, psrc, cache + 64,
                          hd, ws, nnew, head0, count_new, num_rows, f32);
            cactus_vulkan_session_sync();
            std::vector<__fp16> got_out((size_t)ws * hd), got_ring((size_t)ws * hd);
            if (ok) {
                std::memcpy(got_out.data(), pout, got_out.size() * 2);
                std::memcpy(got_ring.data(), cache + 64, got_ring.size() * 2);
            }
            for (void* p : {(void*)cache, (void*)psrc, (void*)pout}) cactus_vulkan_free_shared(p);
            if (!ok) { std::cerr << "  [-] conv_cache_append refused f32=" << f32 << "\n"; return false; }

            std::vector<float> want_ring(ring_init.size());
            for (size_t i = 0; i < ring_init.size(); ++i) want_ring[i] = (float)ring_init[i];
            uint32_t start_row = num_rows - nnew;
            for (uint32_t i = 0; i < nnew; ++i)
                for (uint32_t x = 0; x < hd; ++x)
                    want_ring[((head0 + i) % ws) * hd + x] = (float)src_h[(start_row + i) * hd + x];
            std::vector<float> want_out((size_t)ws * hd, 0.0f);
            uint32_t pad = ws - count_new;
            for (uint32_t y = pad; y < ws; ++y) {
                uint32_t a = count_new - 1 - (y - pad);
                for (uint32_t x = 0; x < hd; ++x) {
                    float v;
                    if (a < nnew) v = (float)src_h[(num_rows - 1 - a) * hd + x];
                    else v = (float)ring_init[((head0 + 2 * ws - 1 - (a - nnew)) % ws) * hd + x];
                    want_out[y * hd + x] = v;
                }
            }
            char what[64];
            std::snprintf(what, sizeof(what), "conv_cache out n=%u f32=%d", nnew, f32);
            if (!close_all(got_out, want_out, 5e-3f, what)) all = false;
            std::snprintf(what, sizeof(what), "conv_cache ring n=%u f32=%d", nnew, f32);
            if (!close_all(got_ring, want_ring, 5e-3f, what)) all = false;
        }
    }
    return all;
}

bool test_w1_kernels() {
    bool all = true;
    auto devbuf = [](size_t bytes) { return cactus_vulkan_alloc_shared(bytes); };

    {   // f32 elementwise: binary add, scalar mul, unary silu, clamp f16+f32
        const size_t n = 517;
        std::vector<float> a(n), b(n);
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> d(-2.0f, 2.0f);
        for (size_t i = 0; i < n; ++i) { a[i] = d(rng); b[i] = d(rng); }
        auto* pa = (float*)devbuf(n * 4); auto* pb = (float*)devbuf(n * 4); auto* py = (float*)devbuf(n * 4);
        std::memcpy(pa, a.data(), n * 4); std::memcpy(pb, b.data(), n * 4);
        bool ok = cactus_vulkan_encode_binary_f32(0, py, pa, pb, n);
        ok = ok && cactus_vulkan_encode_scalar_f32(2, py, py, n, 1.5f);
        ok = ok && cactus_vulkan_encode_unary_f32(2, py, py, n);
        ok = ok && cactus_vulkan_encode_clamp(py, py, n, -0.9f, 0.9f, 1);
        cactus_vulkan_session_sync();
        std::vector<float> got(n);
        if (ok) std::memcpy(got.data(), py, n * 4);
        for (void* p : {(void*)pa, (void*)pb, (void*)py}) cactus_vulkan_free_shared(p);
        if (!ok) { std::cerr << "  [-] f32 ew refused\n"; return false; }
        std::vector<__fp16> goth(n);
        std::vector<float> want(n);
        for (size_t i = 0; i < n; ++i) {
            float v = (a[i] + b[i]) * 1.5f;
            v = v / (1.0f + std::exp(-v));
            want[i] = std::min(0.9f, std::max(-0.9f, v));
            goth[i] = (__fp16)got[i];
        }
        if (!close_all(goth, want, 5e-3f, "f32 ew chain")) all = false;
    }

    {   // reduce_axis (sum/mean/var/min/max) + cumsum, f16
        const uint32_t outer = 3, axis = 17, inner = 5;
        auto x = rand_halfs((size_t)outer * axis * inner);
        auto* px = (__fp16*)devbuf(x.size() * 2);
        auto* py = (__fp16*)devbuf((size_t)outer * inner * 2);
        auto* pc = (__fp16*)devbuf(x.size() * 2);
        std::memcpy(px, x.data(), x.size() * 2);
        for (int op = 0; op <= 4; ++op) {
            if (!cactus_vulkan_encode_reduce_axis(op, py, px, outer, axis, inner, 0)) { all = false; break; }
            cactus_vulkan_session_sync();
            std::vector<__fp16> got(outer * inner);
            std::memcpy(got.data(), py, got.size() * 2);
            std::vector<float> want(outer * inner);
            for (uint32_t o = 0; o < outer; ++o)
                for (uint32_t i = 0; i < inner; ++i) {
                    size_t base = (size_t)o * axis * inner + i;
                    float sum = 0, mn = 1e30f, mx = -1e30f;
                    for (uint32_t aI = 0; aI < axis; ++aI) {
                        float v = (float)x[base + aI * inner];
                        sum += v; mn = std::min(mn, v); mx = std::max(mx, v);
                    }
                    float mean = sum / axis, var = 0;
                    for (uint32_t aI = 0; aI < axis; ++aI) {
                        float dv = (float)x[base + aI * inner] - mean; var += dv * dv;
                    }
                    var /= axis;
                    want[o * inner + i] = op == 0 ? sum : op == 1 ? mean : op == 2 ? var : op == 3 ? mn : mx;
                }
            char what[32]; std::snprintf(what, sizeof(what), "reduce op=%d", op);
            if (!close_all(got, want, 5e-2f, what)) all = false;
        }
        if (cactus_vulkan_encode_cumsum(pc, px, outer, axis, inner, 0)) {
            cactus_vulkan_session_sync();
            std::vector<__fp16> got(x.size());
            std::memcpy(got.data(), pc, x.size() * 2);
            std::vector<float> want(x.size());
            for (uint32_t o = 0; o < outer; ++o)
                for (uint32_t i = 0; i < inner; ++i) {
                    float run = 0;
                    for (uint32_t aI = 0; aI < axis; ++aI) {
                        size_t idx = (size_t)o * axis * inner + aI * inner + i;
                        run += (float)x[idx];
                        want[idx] = run;
                    }
                }
            if (!close_all(got, want, 5e-2f, "cumsum")) all = false;
        } else { std::cerr << "  [-] cumsum refused\n"; all = false; }
        for (void* p : {(void*)px, (void*)py, (void*)pc}) cactus_vulkan_free_shared(p);
    }

    {   // gather_f32idx
        const uint32_t R = 40, D = 33, M = 7;
        auto table = rand_halfs((size_t)R * D);
        std::vector<float> idxf = {3, 0, 39, 12, 12, 5, 21};
        auto* pi = (float*)devbuf(M * 4);
        auto* po = (__fp16*)devbuf((size_t)M * D * 2);
        std::memcpy(pi, idxf.data(), M * 4);
        bool ok = cactus_vulkan_encode_gather_f32idx(po, table.data(), pi, M, D, table.size() * 2);
        cactus_vulkan_session_sync();
        std::vector<__fp16> got((size_t)M * D);
        if (ok) std::memcpy(got.data(), po, got.size() * 2);
        cactus_vulkan_free_shared(pi); cactus_vulkan_free_shared(po);
        if (!ok) { std::cerr << "  [-] gather_f32idx refused\n"; all = false; }
        else {
            std::vector<float> want((size_t)M * D);
            for (uint32_t m = 0; m < M; ++m)
                for (uint32_t dd = 0; dd < D; ++dd)
                    want[m * D + dd] = (float)table[(size_t)(uint32_t)idxf[m] * D + dd];
            if (!close_all(got, want, 1e-3f, "gather_f32idx")) all = false;
        }
    }

    {   // maxpool1d + bilinear + groupnorm
        const uint32_t NC = 6, L = 50, K = 3, ST = 2, Lout = (L - K) / ST + 1;
        auto x = rand_halfs((size_t)NC * L);
        auto* px = (__fp16*)devbuf(x.size() * 2);
        auto* py = (__fp16*)devbuf((size_t)NC * Lout * 2);
        std::memcpy(px, x.data(), x.size() * 2);
        bool ok = cactus_vulkan_encode_maxpool1d(py, px, NC, L, Lout, K, ST);
        cactus_vulkan_session_sync();
        std::vector<__fp16> got((size_t)NC * Lout);
        if (ok) std::memcpy(got.data(), py, got.size() * 2);
        cactus_vulkan_free_shared(px); cactus_vulkan_free_shared(py);
        if (!ok) { std::cerr << "  [-] maxpool1d refused\n"; all = false; }
        else {
            std::vector<float> want((size_t)NC * Lout);
            for (uint32_t c = 0; c < NC; ++c)
                for (uint32_t lo = 0; lo < Lout; ++lo) {
                    float r = -1e30f;
                    for (uint32_t k = 0; k < K && lo * ST + k < L; ++k)
                        r = std::max(r, (float)x[(size_t)c * L + lo * ST + k]);
                    want[c * Lout + lo] = r;
                }
            if (!close_all(got, want, 1e-3f, "maxpool1d")) all = false;
        }

        const uint32_t sh = 8, sw = 6, dh = 5, dw = 9, E = 12;
        auto img = rand_halfs((size_t)sh * sw * E);
        auto* pim = (__fp16*)devbuf(img.size() * 2);
        auto* pob = (__fp16*)devbuf((size_t)dh * dw * E * 2);
        std::memcpy(pim, img.data(), img.size() * 2);
        ok = cactus_vulkan_encode_bilinear(pob, pim, sh, sw, dh, dw, E, 0);
        cactus_vulkan_session_sync();
        std::vector<__fp16> gotb((size_t)dh * dw * E);
        if (ok) std::memcpy(gotb.data(), pob, gotb.size() * 2);
        cactus_vulkan_free_shared(pim); cactus_vulkan_free_shared(pob);
        if (!ok) { std::cerr << "  [-] bilinear refused\n"; all = false; }
        else {
            std::vector<float> want((size_t)dh * dw * E);
            for (uint32_t pix = 0; pix < dh * dw; ++pix)
                for (uint32_t e = 0; e < E; ++e) {
                    uint32_t dy = pix / dw, dx = pix % dw;
                    float syf = std::min(std::max((dy + 0.5f) * (float)sh / dh - 0.5f, 0.0f), (float)sh - 1);
                    float sxf = std::min(std::max((dx + 0.5f) * (float)sw / dw - 0.5f, 0.0f), (float)sw - 1);
                    int y0 = (int)std::floor(syf), x0 = (int)std::floor(sxf);
                    int y1 = std::min(y0 + 1, (int)sh - 1), x1 = std::min(x0 + 1, (int)sw - 1);
                    float fy = syf - y0, fx = sxf - x0;
                    auto at = [&](int yy, int xx) { return (float)img[((size_t)yy * sw + xx) * E + e]; };
                    want[(size_t)pix * E + e] = at(y0,x0)*(1-fx)*(1-fy) + at(y0,x1)*fx*(1-fy)
                                              + at(y1,x0)*(1-fx)*fy + at(y1,x1)*fx*fy;
                }
            if (!close_all(gotb, want, 5e-3f, "bilinear")) all = false;
        }

        const uint32_t N = 2, C = 8, S = 10, G = 4;
        auto gx = rand_halfs((size_t)N * C * S);
        auto gw = rand_halfs(C, 0.5f, 1.5f);
        auto gb = rand_halfs(C, -0.5f, 0.5f);
        auto* pgx = (__fp16*)devbuf(gx.size() * 2);
        auto* pgo = (__fp16*)devbuf(gx.size() * 2);
        std::memcpy(pgx, gx.data(), gx.size() * 2);
        ok = cactus_vulkan_encode_groupnorm(pgo, pgx, gw.data(), gb.data(), N, C, S, G, 1e-5f);
        cactus_vulkan_session_sync();
        std::vector<__fp16> gotg(gx.size());
        if (ok) std::memcpy(gotg.data(), pgo, gotg.size() * 2);
        cactus_vulkan_free_shared(pgx); cactus_vulkan_free_shared(pgo);
        if (!ok) { std::cerr << "  [-] groupnorm refused\n"; all = false; }
        else {
            std::vector<float> want(gx.size());
            uint32_t cpg = C / G;
            for (uint32_t nb = 0; nb < N; ++nb)
                for (uint32_t g = 0; g < G; ++g) {
                    size_t base = ((size_t)nb * C + g * cpg) * S;
                    uint32_t cnt = cpg * S;
                    double sum = 0, sq = 0;
                    for (uint32_t i = 0; i < cnt; ++i) { double v = (float)gx[base + i]; sum += v; sq += v * v; }
                    float mean = (float)(sum / cnt);
                    float inv = 1.0f / std::sqrt((float)(sq / cnt) - mean * mean + 1e-5f);
                    for (uint32_t i = 0; i < cnt; ++i) {
                        uint32_t c = g * cpg + i / S;
                        want[base + i] = ((float)gx[base + i] - mean) * inv * (float)gw[c] + (float)gb[c];
                    }
                }
            if (!close_all(gotg, want, 2e-2f, "groupnorm")) all = false;
        }
    }
    return all;
}

bool test_conv2d() {
    const uint32_t N = 1, Cin = 3, H = 9, W = 11, Cout = 5, K = 3, ST = 2, PAD = 1;
    const uint32_t Ho = (H + 2 * PAD - K) / ST + 1, Wo = (W + 2 * PAD - K) / ST + 1;
    auto x = rand_halfs((size_t)N * Cin * H * W);
    auto w = rand_halfs((size_t)Cout * Cin * K * K);
    auto b = rand_halfs(Cout);
    auto* px = (__fp16*)cactus_vulkan_alloc_shared(x.size() * 2);
    auto* py = (__fp16*)cactus_vulkan_alloc_shared((size_t)N * Cout * Ho * Wo * 2);
    std::memcpy(px, x.data(), x.size() * 2);
    bool ok = cactus_vulkan_encode_conv2d(py, px, w.data(), b.data(), N, Cin, H, W, Cout, Ho, Wo, K, ST, PAD, 0);
    cactus_vulkan_session_sync();
    std::vector<__fp16> got((size_t)N * Cout * Ho * Wo);
    if (ok) std::memcpy(got.data(), py, got.size() * 2);
    cactus_vulkan_free_shared(px); cactus_vulkan_free_shared(py);
    if (!ok) { std::cerr << "  [-] conv2d refused\n"; return false; }
    std::vector<float> want(got.size());
    for (uint32_t co = 0; co < Cout; ++co)
        for (uint32_t ho = 0; ho < Ho; ++ho)
            for (uint32_t wo = 0; wo < Wo; ++wo) {
                float acc = (float)b[co];
                for (uint32_t ci = 0; ci < Cin; ++ci)
                    for (uint32_t kh = 0; kh < K; ++kh)
                        for (uint32_t kw = 0; kw < K; ++kw) {
                            int h = (int)(ho * ST) - (int)PAD + (int)kh;
                            int ww2 = (int)(wo * ST) - (int)PAD + (int)kw;
                            if (h < 0 || h >= (int)H || ww2 < 0 || ww2 >= (int)W) continue;
                            acc += (float)x[((size_t)ci * H + h) * W + ww2]
                                 * (float)w[(((size_t)co * Cin + ci) * K + kh) * K + kw];
                        }
                want[((size_t)co * Ho + ho) * Wo + wo] = acc;
            }
    return close_all(got, want, 5e-2f, "conv2d");
}

bool test_embeddings() {
    bool all = true;
    {   // hadamard embedding vs reference
        CQ4Fixture f(false, 256, 12, 64);
        uint32_t rows[3] = {0, 7, 11};
        auto* po = (__fp16*)cactus_vulkan_alloc_shared((size_t)3 * f.K * 2);
        bool ok = cactus_vulkan_encode_embedding_hadamard_m(po, &f.W, rows, 3);
        cactus_vulkan_session_sync();
        std::vector<__fp16> got((size_t)3 * f.K);
        if (ok) std::memcpy(got.data(), po, got.size() * 2);
        cactus_vulkan_free_shared(po);
        if (!ok) { std::cerr << "  [-] emb_hadamard refused\n"; return false; }
        std::vector<float> want((size_t)3 * f.K);
        for (int m = 0; m < 3; ++m) {
            uint32_t r = rows[m];
            for (uint32_t g = 0; g < f.ng; ++g) {
                std::vector<float> z(f.gs, 0.0f);
                for (uint32_t k = 0; k < f.gs; ++k) {
                    uint32_t dst = f.perm[k];
                    z[dst] = (float)f.codebook[f.idx[(size_t)r * f.K + g * f.gs + k]] * (float)f.rs[dst];
                }
                for (uint32_t h = 1; h < f.gs; h <<= 1)
                    for (uint32_t k = 0; k < f.gs; ++k)
                        if ((k & h) == 0) { float a = z[k], b = z[k + h]; z[k] = a + b; z[k + h] = a - b; }
                float nrm = (float)f.norms[(size_t)r * f.ng + g];
                float inv = 1.0f / std::sqrt((float)f.gs);
                for (uint32_t k = 0; k < f.gs; ++k) {
                    uint32_t col = g * f.gs + k;
                    want[(size_t)m * f.K + col] = z[k] * inv * (float)f.ls[k] * nrm * (float)f.recip[col];
                }
            }
        }
        if (!close_all(got, want, 5e-2f, "emb_hadamard_m")) all = false;
    }
    {   // ortho embedding vs reference
        const uint32_t K = 128, NR = 10;
        CQ4Fixture f(false, K, NR, K);   // ng=1, gs=K
        auto rot = rand_halfs((size_t)K * K, -0.2f, 0.2f);
        f.W.rotation = rot.data();
        f.W.flags |= CACTUS_QUANT_FLAG_ORTHOGONAL;
        uint32_t rows[2] = {2, 9};
        auto* po = (__fp16*)cactus_vulkan_alloc_shared((size_t)2 * K * 2);
        bool ok = cactus_vulkan_encode_embedding_ortho_m(po, &f.W, rows, 2);
        cactus_vulkan_session_sync();
        std::vector<__fp16> got((size_t)2 * K);
        if (ok) std::memcpy(got.data(), po, got.size() * 2);
        cactus_vulkan_free_shared(po);
        if (!ok) { std::cerr << "  [-] emb_ortho refused\n"; return false; }
        std::vector<float> want((size_t)2 * K);
        for (int m = 0; m < 2; ++m) {
            uint32_t r = rows[m];
            float nrm = (float)f.norms[r];
            for (uint32_t j = 0; j < K; ++j) {
                double acc = 0;
                for (uint32_t k = 0; k < K; ++k)
                    acc += (float)f.codebook[f.idx[(size_t)r * K + k]] * (float)rot[(size_t)j * K + k];
                want[(size_t)m * K + j] = (float)acc * nrm * (float)f.recip[j];
            }
        }
        if (!close_all(got, want, 5e-2f, "emb_ortho_m")) all = false;
    }
    return all;
}

bool test_attn_fused() {
    // fused(norm+rope+append+attend) must match the composed primitives
    const uint32_t nqh = 8, hd = 64, vhd = 64, max_seq = 32, hist = 9;
    const uint32_t ngK = (hd + 31) / 32, ngV = (vhd + 31) / 32;
    const float eps = 1e-6f, scale = 0.125f;
    auto q = rand_halfs((size_t)nqh * hd);
    auto kraw = rand_halfs(hd), vraw = rand_halfs(vhd);
    auto qw = rand_halfs(hd, 0.5f, 1.5f), kw = rand_halfs(hd, 0.5f, 1.5f), vw = rand_halfs(vhd, 0.5f, 1.5f);
    std::vector<__fp16> cs(hd), sn(hd);
    for (uint32_t i = 0; i < hd; ++i) { cs[i] = (__fp16)std::cos(0.01f * i); sn[i] = (__fp16)std::sin(0.01f * i); }

    size_t kcb = (size_t)max_seq * hd, vcb = (size_t)max_seq * vhd;
    size_t ksb = (size_t)max_seq * ngK * 4, vsb = (size_t)max_seq * ngV * 4;
    auto* pq = (__fp16*)cactus_vulkan_alloc_shared(q.size() * 2);
    auto* pk = (__fp16*)cactus_vulkan_alloc_shared(hd * 2);
    auto* pv = (__fp16*)cactus_vulkan_alloc_shared(vhd * 2);
    auto* pcs = (__fp16*)cactus_vulkan_alloc_shared(hd * 2);
    auto* psn = (__fp16*)cactus_vulkan_alloc_shared(hd * 2);
    char* kc1 = (char*)cactus_vulkan_alloc_shared(kcb); char* vc1 = (char*)cactus_vulkan_alloc_shared(vcb);
    auto* ks1 = (float*)cactus_vulkan_alloc_shared(ksb); auto* vs1 = (float*)cactus_vulkan_alloc_shared(vsb);
    char* kc2 = (char*)cactus_vulkan_alloc_shared(kcb); char* vc2 = (char*)cactus_vulkan_alloc_shared(vcb);
    auto* ks2 = (float*)cactus_vulkan_alloc_shared(ksb); auto* vs2 = (float*)cactus_vulkan_alloc_shared(vsb);
    auto* o1 = (__fp16*)cactus_vulkan_alloc_shared((size_t)nqh * vhd * 2);
    auto* o2 = (__fp16*)cactus_vulkan_alloc_shared((size_t)nqh * vhd * 2);
    auto* qn = (__fp16*)cactus_vulkan_alloc_shared(q.size() * 2);
    auto* kn = (__fp16*)cactus_vulkan_alloc_shared(hd * 2);
    auto* vn = (__fp16*)cactus_vulkan_alloc_shared(vhd * 2);
    if (!pq || !kc1 || !kc2 || !o1 || !o2 || !qn || !kn || !vn) return false;
    std::memcpy(pq, q.data(), q.size() * 2);
    std::memcpy(pk, kraw.data(), hd * 2);
    std::memcpy(pv, vraw.data(), vhd * 2);
    std::memcpy(pcs, cs.data(), hd * 2);
    std::memcpy(psn, sn.data(), hd * 2);
    // seed identical random history in both cache pairs
    std::mt19937 rng(3);
    std::vector<char> hk(kcb), hv(vcb);
    std::vector<float> hks(ksb / 4), hvs(vsb / 4);
    for (auto& v : hk) v = (char)(int)(rng() % 255) - 127;
    for (auto& v : hv) v = (char)(int)(rng() % 255) - 127;
    for (auto& v : hks) v = 0.001f + 0.001f * (rng() % 100);
    for (auto& v : hvs) v = 0.001f + 0.001f * (rng() % 100);
    std::memcpy(kc1, hk.data(), kcb); std::memcpy(kc2, hk.data(), kcb);
    std::memcpy(vc1, hv.data(), vcb); std::memcpy(vc2, hv.data(), vcb);
    std::memcpy(ks1, hks.data(), ksb); std::memcpy(ks2, hks.data(), ksb);
    std::memcpy(vs1, hvs.data(), vsb); std::memcpy(vs2, hvs.data(), vsb);

    bool ok = cactus_vulkan_encode_attention_fused_i8(o1, pq, pk, pv, kc1, vc1, ks1, vs1,
                  qw.data(), kw.data(), vw.data(), pcs, psn,
                  nqh, hd, vhd, 0, hist + 1, hist, 1, eps, scale, kcb, vcb, ksb, vsb);
    if (!ok) { std::cerr << "  [-] fused attn refused\n"; return false; }

    // composed reference on GPU: rope_pair_rms(q), rope_pair_rms(k), rms(v), append, attend
    ok = cactus_vulkan_encode_rope_pair_rms(qn, pq, qw.data(), pcs, psn, nqh, hd, eps);
    ok = ok && cactus_vulkan_encode_rope_pair_rms(kn, pk, kw.data(), pcs, psn, 1, hd, eps);
    ok = ok && cactus_vulkan_encode_rms_norm_f16(vn, pv, vw.data(), 1, vhd, eps);
    ok = ok && cactus_vulkan_encode_kv_append_i8(kn, kc2, ks2, 1, hd, hist, 32, 1, 0, 0, hd * 2, kcb, ksb);
    ok = ok && cactus_vulkan_encode_kv_append_i8(vn, vc2, vs2, 1, vhd, hist, 32, 1, 0, 0, vhd * 2, vcb, vsb);
    ok = ok && cactus_vulkan_encode_attention_i8(o2, qn, nullptr, nullptr, kc2, vc2, ks2, vs2,
                   nqh, 1, hd, vhd, hist + 1, hist + 1, 0, hist + 1, scale, kcb, vcb, ksb, vsb);
    cactus_vulkan_session_sync();
    if (!ok) { std::cerr << "  [-] composed ref refused\n"; return false; }
    std::vector<__fp16> g1((size_t)nqh * vhd);
    std::vector<float> g2v((size_t)nqh * vhd);
    std::memcpy(g1.data(), o1, g1.size() * 2);
    for (size_t i = 0; i < g2v.size(); ++i) g2v[i] = (float)o2[i];
    bool pass = close_all(g1, g2v, 5e-2f, "fused vs composed");
    for (void* p : {(void*)pq,(void*)pk,(void*)pv,(void*)pcs,(void*)psn,(void*)kc1,(void*)vc1,(void*)ks1,(void*)vs1,
                    (void*)kc2,(void*)vc2,(void*)ks2,(void*)vs2,(void*)o1,(void*)o2,(void*)qn,(void*)kn,(void*)vn})
        cactus_vulkan_free_shared(p);
    return pass;
}

bool test_elemwise_chain() {
    // interpreter kernel vs CPU reference: gelu -> *side0 -> +0.5 -> clamp
    const size_t n = 300;
    auto x = rand_halfs(n);
    auto s0 = rand_halfs(n);
    float steps[4 * 4] = {};
    auto set_step = [&](int i, int kind, int code, float p0, float p1) {
        uint32_t* u = reinterpret_cast<uint32_t*>(steps + i * 4);
        u[0] = (uint32_t)kind; u[1] = (uint32_t)code;
        steps[i * 4 + 2] = p0; steps[i * 4 + 3] = p1;
    };
    set_step(0, 0, 0, 0, 0);          // unary gelu
    set_step(1, 2, 3, 0, 0);          // side op: slot0, op mul, lhs
    set_step(2, 1, 0, 0.5f, 0);       // scalar add 0.5
    set_step(3, 3, 0, -0.8f, 0.9f);   // clamp
    auto* px = (__fp16*)cactus_vulkan_alloc_shared(n * 2);
    auto* ps = (__fp16*)cactus_vulkan_alloc_shared(n * 2);
    auto* py = (__fp16*)cactus_vulkan_alloc_shared(n * 2);
    std::memcpy(px, x.data(), n * 2);
    std::memcpy(ps, s0.data(), n * 2);
    size_t side_elems[3] = {n, 0, 0};
    bool ok = cactus_vulkan_encode_elemwise_chain(py, px, steps, 4, ps, nullptr, nullptr,
                  side_elems, n, 0, 1);
    cactus_vulkan_session_sync();
    std::vector<__fp16> got(n);
    if (ok) std::memcpy(got.data(), py, n * 2);
    for (void* p : {(void*)px, (void*)ps, (void*)py}) cactus_vulkan_free_shared(p);
    if (!ok) { std::cerr << "  [-] elemwise_chain refused\n"; return false; }
    std::vector<float> want(n);
    for (size_t i = 0; i < n; ++i) {
        float v = (float)x[i];
        v = gelu_ref(v);            v = (float)(__fp16)v;
        v = v * (float)s0[i];       v = (float)(__fp16)v;
        v = v + 0.5f;               v = (float)(__fp16)v;
        v = std::min(0.9f, std::max(-0.8f, v)); v = (float)(__fp16)v;
        want[i] = v;
    }
    if (!close_all(got, want, 1e-2f, "elemwise_chain")) return false;

    {   // broadcast side (bmode=1: outer index) + f32 output
        const uint32_t outer = 20, inner = 16;
        const size_t nn = (size_t)outer * inner;
        auto xx = rand_halfs(nn);
        auto sd = rand_halfs(outer);
        float st2[2 * 4] = {};
        auto set2 = [&](int i, int kind, int code, float p0) {
            uint32_t* u = reinterpret_cast<uint32_t*>(st2 + i * 4);
            u[0] = (uint32_t)kind; u[1] = (uint32_t)code; st2[i * 4 + 2] = p0;
        };
        set2(0, 2, 3 | (1 << 8), 0);   // mul side0, bmode=1 (i/inner)
        set2(1, 4, 1, 0);              // switch to f32 precision
        auto* px2 = (__fp16*)cactus_vulkan_alloc_shared(nn * 2);
        auto* ps2 = (__fp16*)cactus_vulkan_alloc_shared(outer * 2);
        auto* py2 = (float*)cactus_vulkan_alloc_shared(nn * 4);
        std::memcpy(px2, xx.data(), nn * 2);
        std::memcpy(ps2, sd.data(), outer * 2);
        size_t se2[3] = {outer, 0, 0};
        bool ok2 = cactus_vulkan_encode_elemwise_chain(py2, px2, st2, 2, ps2, nullptr, nullptr,
                       se2, nn, 2u /*out f32*/, inner);
        cactus_vulkan_session_sync();
        std::vector<float> gotf(nn);
        if (ok2) std::memcpy(gotf.data(), py2, nn * 4);
        for (void* p : {(void*)px2, (void*)ps2, (void*)py2}) cactus_vulkan_free_shared(p);
        if (!ok2) { std::cerr << "  [-] chain bmode refused\n"; return false; }
        std::vector<__fp16> goth(nn);
        std::vector<float> wantf(nn);
        for (size_t i = 0; i < nn; ++i) {
            float v = (float)xx[i] * (float)sd[i / inner];
            v = (float)(__fp16)v;
            wantf[i] = v;
            goth[i] = (__fp16)gotf[i];
        }
        if (!close_all(goth, wantf, 1e-2f, "chain bmode+f32out")) return false;
    }
    return true;
}

bool test_lowbit_gemv() {
    // bits=2 flat: reuse CQ4Fixture transform fields, repack 2-bit indices
    CQ4Fixture f(false, 256, 24, 64);
    const uint32_t K = f.K, N = f.N, gs = f.gs, ng = f.ng;
    std::vector<__fp16> cb2 = {(__fp16)-1.0f, (__fp16)-0.33f, (__fp16)0.33f, (__fp16)1.0f};
    std::vector<uint8_t> idx2((size_t)N * K);
    std::mt19937 rng(21);
    for (auto& v : idx2) v = (uint8_t)(rng() % 4);
    uint32_t pgb2 = gs / 4;
    std::vector<uint8_t> packed2((size_t)N * ng * pgb2, 0);
    for (uint32_t n = 0; n < N; ++n)
        for (uint32_t g = 0; g < ng; ++g)
            for (uint32_t e = 0; e < gs; ++e) {
                uint8_t v = idx2[(size_t)n * K + g * gs + e];
                packed2[((size_t)n * ng + g) * pgb2 + (e >> 2)] |= (uint8_t)(v << ((e & 3u) * 2u));
            }
    CactusQuantMatrix W = f.W;
    W.bits = 2;
    W.codebook = cb2.data();
    W.packed_indices = packed2.data();
    auto x = rand_halfs(K);
    std::vector<__fp16> got(N);
    if (!cactus_vulkan_cq_gemv(got.data(), x.data(), &W)) {
        std::cerr << "  [-] lowbit gemv refused\n";
        return false;
    }
    // oracle: same transform as fixture, dot with 2-bit codebook
    std::vector<__fp16> code(K);
    for (uint32_t g = 0; g < ng; ++g) {
        std::vector<float> z(gs);
        for (uint32_t k = 0; k < gs; ++k) {
            uint32_t gk = g * gs + k;
            z[k] = (float)x[gk] * (float)f.recip[gk] * (float)f.ls[k];
        }
        for (uint32_t h = 1; h < gs; h <<= 1)
            for (uint32_t k = 0; k < gs; ++k)
                if ((k & h) == 0) { float a2 = z[k], b2 = z[k + h]; z[k] = a2 + b2; z[k + h] = a2 - b2; }
        float sc = 1.0f / std::sqrt((float)gs);
        for (uint32_t k = 0; k < gs; ++k) z[k] *= sc * (float)f.rs[k];
        for (uint32_t k = 0; k < gs; ++k) code[g * gs + k] = (__fp16)z[f.perm[k]];
    }
    std::vector<float> want(N);
    float maxy = 0;
    for (uint32_t n = 0; n < N; ++n) {
        double acc = 0;
        for (uint32_t g = 0; g < ng; ++g) {
            double p = 0;
            for (uint32_t e = 0; e < gs; ++e)
                p += (float)code[g * gs + e] * (float)cb2[idx2[(size_t)n * K + g * gs + e]];
            acc += (float)f.norms[(size_t)n * ng + g] * p;
        }
        want[n] = (float)acc;
        maxy = std::max(maxy, std::fabs(want[n]));
    }
    return close_all(got, want, std::max(5e-2f, 0.02f * maxy), "lowbit gemv bits=2");
}

bool test_deltanet() {
    const uint32_t B = 1, T = 3, Hq = 2, Hv = 4, K = 32, V = 16;
    const float scale = 0.18f;
    auto q = rand_halfs((size_t)B * T * Hq * K), k = rand_halfs((size_t)B * T * Hq * K);
    auto v = rand_halfs((size_t)B * T * Hv * V);
    auto g = rand_halfs((size_t)B * T * Hv, -2.0f, 0.0f), bt = rand_halfs((size_t)B * T * Hv, 0.1f, 0.9f);
    auto s0 = rand_halfs((size_t)B * K * Hv * V);
    size_t ysz = (size_t)B * (T + K) * Hv * V;
    auto* pq = (__fp16*)cactus_vulkan_alloc_shared(q.size() * 2);
    auto* pk = (__fp16*)cactus_vulkan_alloc_shared(k.size() * 2);
    auto* pv = (__fp16*)cactus_vulkan_alloc_shared(v.size() * 2);
    auto* pg = (__fp16*)cactus_vulkan_alloc_shared(g.size() * 2);
    auto* pb = (__fp16*)cactus_vulkan_alloc_shared(bt.size() * 2);
    auto* ps = (__fp16*)cactus_vulkan_alloc_shared(s0.size() * 2);
    auto* py = (__fp16*)cactus_vulkan_alloc_shared(ysz * 2);
    std::memcpy(pq, q.data(), q.size() * 2); std::memcpy(pk, k.data(), k.size() * 2);
    std::memcpy(pv, v.data(), v.size() * 2); std::memcpy(pg, g.data(), g.size() * 2);
    std::memcpy(pb, bt.data(), bt.size() * 2); std::memcpy(ps, s0.data(), s0.size() * 2);
    bool ok = cactus_vulkan_encode_deltanet_prefill(py, pq, pk, pv, pg, pb, ps, B, T, Hq, Hv, K, V, scale);
    cactus_vulkan_session_sync();
    std::vector<__fp16> got(ysz);
    if (ok) std::memcpy(got.data(), py, ysz * 2);
    for (void* p : {(void*)pq,(void*)pk,(void*)pv,(void*)pg,(void*)pb,(void*)ps,(void*)py})
        cactus_vulkan_free_shared(p);
    if (!ok) { std::cerr << "  [-] deltanet prefill refused\n"; return false; }
    // CPU reference recurrence
    size_t hv_stride = (size_t)Hv * V;
    std::vector<float> st((size_t)Hv * K * V);
    for (uint32_t h = 0; h < Hv; ++h)
        for (uint32_t kd = 0; kd < K; ++kd)
            for (uint32_t tv = 0; tv < V; ++tv)
                st[((size_t)h * K + kd) * V + tv] = (float)s0[(size_t)kd * hv_stride + h * V + tv];
    std::vector<float> want(ysz, 0.0f);
    for (uint32_t step = 0; step < T; ++step)
        for (uint32_t h = 0; h < Hv; ++h) {
            uint32_t qk_head = h / (Hv / Hq);
            float gl = (float)g[(size_t)step * Hv + h];
            float beta = std::min(1.0f, std::max(0.0f, (float)bt[(size_t)step * Hv + h]));
            float alpha = std::exp(std::min(6.0f, std::max(-20.0f, gl)));
            for (uint32_t tv = 0; tv < V; ++tv) {
                float proj = 0;
                for (uint32_t kd = 0; kd < K; ++kd)
                    proj += st[((size_t)h * K + kd) * V + tv] * (float)k[((size_t)step * Hq + qk_head) * K + kd];
                float delta = ((float)v[((size_t)step * Hv + h) * V + tv] - alpha * proj) * beta;
                float acc = 0;
                for (uint32_t kd = 0; kd < K; ++kd) {
                    float sn = st[((size_t)h * K + kd) * V + tv] * alpha
                             + (float)k[((size_t)step * Hq + qk_head) * K + kd] * delta;
                    st[((size_t)h * K + kd) * V + tv] = sn;
                    acc += sn * (float)q[((size_t)step * Hq + qk_head) * K + kd];
                }
                want[(size_t)step * hv_stride + h * V + tv] = acc * scale;
            }
        }
    for (uint32_t kd = 0; kd < K; ++kd)
        for (uint32_t h = 0; h < Hv; ++h)
            for (uint32_t tv = 0; tv < V; ++tv)
                want[(size_t)(T + kd) * hv_stride + h * V + tv] = st[((size_t)h * K + kd) * V + tv];
    return close_all(got, want, 5e-2f, "deltanet prefill");
}

bool test_moe() {
    const uint32_t E = 4, TOPK = 2, TOK = 2, K1 = 256, N1 = 128, K2 = 128, N2 = 256;
    std::vector<CQ4Fixture> w1s, w3s, w2s;
    for (uint32_t e = 0; e < E; ++e) {
        w1s.emplace_back(true, K1, N1, 128);
        w3s.emplace_back(true, K1, N1, 128);
        w2s.emplace_back(true, K2, N2, 128);
    }
    // Lay each expert's sections into a uniform slot so the arena builder accepts them
    auto build_slab = [&](std::vector<CQ4Fixture>& fs, std::vector<uint8_t>& slab,
                          std::vector<CactusQuantMatrix>& Ws) {
        auto& f0 = fs[0];
        size_t pk = f0.packed.size(), nm = f0.norms.size() * 2, cb = 32,
               rc = f0.recip.size() * 2, sg = f0.ls.size(), pm = f0.perm.size() * 4;
        size_t o_pk = 0, o_nm = (o_pk + pk + 15) & ~15UL, o_cb = (o_nm + nm + 15) & ~15UL,
               o_rc = (o_cb + cb + 15) & ~15UL, o_ls = (o_rc + rc + 15) & ~15UL,
               o_rs = (o_ls + sg + 15) & ~15UL, o_pm = (o_rs + sg + 15) & ~15UL;
        size_t slot = (o_pm + pm + 15) & ~15UL;
        slab.assign(slot * E, 0);
        Ws.resize(E);
        for (uint32_t e = 0; e < E; ++e) {
            uint8_t* base = slab.data() + slot * e;
            auto& f = fs[e];
            std::memcpy(base + o_pk, f.packed.data(), pk);
            std::memcpy(base + o_nm, f.norms.data(), nm);
            std::memcpy(base + o_cb, f.codebook.data(), cb);
            std::memcpy(base + o_rc, f.recip.data(), rc);
            std::memcpy(base + o_ls, f.ls.data(), sg);
            std::memcpy(base + o_rs, f.rs.data(), sg);
            std::memcpy(base + o_pm, f.perm.data(), pm);
            Ws[e] = f.W;
            Ws[e].packed_indices = base + o_pk;
            Ws[e].norms = (const __fp16*)(base + o_nm);
            Ws[e].codebook = (const __fp16*)(base + o_cb);
            Ws[e].input_scale_recip = (const __fp16*)(base + o_rc);
            Ws[e].left_signs = (const int8_t*)(base + o_ls);
            Ws[e].right_signs = (const int8_t*)(base + o_rs);
            Ws[e].permutation = (const uint32_t*)(base + o_pm);
        }
    };
    std::vector<uint8_t> slab1, slab3, slab2;
    std::vector<CactusQuantMatrix> W1, W3, W2;
    build_slab(w1s, slab1, W1);
    build_slab(w3s, slab3, W3);
    build_slab(w2s, slab2, W2);
    if (!cactus_vulkan_moe_cq4_build(W1.data(), W3.data(), W2.data(), E)) {
        std::cerr << "  [-] moe build refused\n";
        return false;
    }
    auto hidden = rand_halfs((size_t)TOK * K1);
    std::vector<float> tk = {1, 3, 2, 0};
    auto pr = rand_halfs((size_t)TOK * E, 0.1f, 0.9f);
    auto* ph = (__fp16*)cactus_vulkan_alloc_shared(hidden.size() * 2);
    auto* pt = (float*)cactus_vulkan_alloc_shared(tk.size() * 4);
    auto* pp = (__fp16*)cactus_vulkan_alloc_shared(pr.size() * 2);
    auto* po = (__fp16*)cactus_vulkan_alloc_shared((size_t)TOK * N2 * 2);
    std::memcpy(ph, hidden.data(), hidden.size() * 2);
    std::memcpy(pt, tk.data(), tk.size() * 4);
    std::memcpy(pp, pr.data(), pr.size() * 2);
    bool ok = cactus_vulkan_encode_moe_gated_cq4(po, ph, pp, pt, &W1[0], E, TOPK, TOK, 1, 1, 1e-6f, 1.0f);
    cactus_vulkan_session_sync();
    std::vector<__fp16> got((size_t)TOK * N2);
    if (ok) std::memcpy(got.data(), po, got.size() * 2);
    for (void* p : {(void*)ph, (void*)pt, (void*)pp, (void*)po}) cactus_vulkan_free_shared(p);
    if (!ok) { std::cerr << "  [-] moe encode refused\n"; return false; }
    std::vector<float> want((size_t)TOK * N2, 0.0f);
    float maxy = 0;
    for (uint32_t t = 0; t < TOK; ++t) {
        std::vector<__fp16> hx(hidden.begin() + (size_t)t * K1, hidden.begin() + (size_t)(t + 1) * K1);
        float denom = 1e-6f;
        for (uint32_t j = 0; j < TOPK; ++j) denom += (float)pr[(size_t)t * E + (uint32_t)tk[t * TOPK + j]];
        for (uint32_t j = 0; j < TOPK; ++j) {
            uint32_t e = (uint32_t)tk[t * TOPK + j];
            float w = (float)pr[(size_t)t * E + e] / denom;
            auto gvec = w1s[e].oracle(hx);
            auto uvec = w3s[e].oracle(hx);
            std::vector<__fp16> gu(K2, (__fp16)0.0f);
            for (uint32_t i = 0; i < N1; ++i) {
                float gh = (float)(__fp16)gvec[i];
                float a = gelu_ref(gh);   // metal act==1 -> gelu
                gu[i] = (__fp16)((float)(__fp16)a * (float)(__fp16)uvec[i]);
            }
            auto dvec = w2s[e].oracle(gu);
            for (uint32_t i = 0; i < N2; ++i) want[(size_t)t * N2 + i] += w * dvec[i];
        }
        for (uint32_t i = 0; i < N2; ++i) maxy = std::max(maxy, std::fabs(want[(size_t)t * N2 + i]));
    }
    bool pass = close_all(got, want, std::max(8e-2f, 0.03f * maxy), "moe gated cq4");
    if (!pass) {
        for (int i = 0; i < 4; ++i)
            std::cerr << "  [moe dbg] i=" << i << " got=" << (float)got[i]
                      << " want=" << want[i] << "\n";
    }
    return pass;
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

static bool run_gemm_case(bool interleaved, uint32_t M, uint32_t K, const char* what) {
    CQ4Fixture f(interleaved, K, 64, 128);
    auto x = rand_halfs((size_t)M * f.K);
    std::vector<__fp16> cpu((size_t)M * f.N);
    cactus_quant_matmul(&f.W, x.data(), M, cpu.data());
    VkBuf bx(x.size() * 2, x.data()), by(cpu.size() * 2);
    if (!bx.p || !by.p) return false;
    if (!cactus_vulkan_encode_quant_matmul_m(by.p, bx.p, &f.W, M)) {
        std::cout << "  [-] " << what << " encode refused\n";
        return true;
    }
    cactus_vulkan_session_sync();
    std::vector<__fp16> gpu(cpu.size());
    std::memcpy(gpu.data(), by.p, cpu.size() * 2);
    std::vector<float> want(cpu.size());
    float maxy = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        want[i] = (float)cpu[i];
        maxy = std::max(maxy, std::fabs(want[i]));
    }
    return close_all(gpu, want, std::max(5e-2f, 0.02f * maxy), what);
}

bool test_gemm_m() {
    return run_gemm_case(false, 8, 512, "gemm M=8 flat")
        && run_gemm_case(true, 8, 512, "gemm M=8 il")
        && run_gemm_case(true, 5, 512, "gemm M=5 ragged")
        && run_gemm_case(true, 12, 2048, "gemm M=12 K=2048 chunked");
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
    runner.run_test("gemm_m", test_gemm_m());
    runner.run_test("gemm_f16", test_gemm_f16());
    runner.run_test("conv_cache_append", test_conv_cache_append());
    runner.run_test("w1_kernels", test_w1_kernels());
    runner.run_test("conv2d", test_conv2d());
    runner.run_test("embeddings", test_embeddings());
    runner.run_test("attn_fused", test_attn_fused());
    runner.run_test("elemwise_chain", test_elemwise_chain());
    runner.run_test("lowbit_gemv", test_lowbit_gemv());
    runner.run_test("deltanet", test_deltanet());
    runner.run_test("moe", test_moe());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
