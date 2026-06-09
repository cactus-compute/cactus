#include "test_utils.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace TestUtils;

bool test_rms_norm() {
    const size_t batch = 4, dim = 128;
    std::vector<__fp16> input(batch * dim), weight(dim), output(batch * dim);
    fill_random_fp16(input, -1.0f, 1.0f);
    for (size_t i = 0; i < dim; i++) weight[i] = static_cast<__fp16>(1.0f);
    cactus_rms_norm_f16(input.data(), weight.data(), output.data(), batch, dim, 1e-6f);
    for (size_t b = 0; b < batch; b++) {
        float sum_sq = 0.0f;
        for (size_t d = 0; d < dim; d++) { float v = static_cast<float>(output[b * dim + d]); sum_sq += v * v; }
        if (std::abs(std::sqrt(sum_sq / dim) - 1.0f) > 0.05f) return false;
    }
    return true;
}

bool test_layer_norm() {
    const size_t batch = 4, dim = 128;
    std::vector<__fp16> input(batch * dim), weight(dim), bias(dim), output(batch * dim);
    fill_random_fp16(input, -1.0f, 1.0f);
    for (size_t i = 0; i < dim; i++) { weight[i] = static_cast<__fp16>(1.0f); bias[i] = static_cast<__fp16>(0.0f); }
    cactus_layer_norm_f16(input.data(), weight.data(), bias.data(), output.data(), batch, dim, 1e-5f);
    for (size_t b = 0; b < batch; b++) {
        float mean = 0.0f;
        for (size_t d = 0; d < dim; d++) mean += static_cast<float>(output[b * dim + d]);
        mean /= dim;
        if (std::abs(mean) > 0.05f) return false;
    }
    return true;
}

bool test_softmax() {
    const size_t rows = 8, vocab = 128;
    std::vector<__fp16> input(rows * vocab), output(rows * vocab);
    fill_random_fp16(input, -5.0f, 5.0f);
    cactus_softmax_f16(input.data(), output.data(), rows, 1, vocab);
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t j = 0; j < vocab; j++) {
            float v = static_cast<float>(output[r * vocab + j]);
            if (v < 0.0f || v > 1.0f) return false;
            sum += v;
        }
        if (std::abs(sum - 1.0f) > 0.01f) return false;
    }
    return true;
}

bool test_rope() {
    const size_t batch = 1, seq = 4, heads = 2, dim = 16;
    std::vector<__fp16> input(batch * seq * heads * dim), output(batch * seq * heads * dim);
    for (auto& v : input) v = static_cast<__fp16>(1.0f);
    cactus_rope_f16(input.data(), output.data(), batch, seq, heads, dim, 0, 10000.0f);
    for (size_t i = 0; i < heads * dim; i++)
        if (std::abs(static_cast<float>(output[i]) - 1.0f) > 0.01f) return false;
    bool changed = false;
    for (size_t i = heads * dim; i < output.size(); i++)
        if (std::abs(static_cast<float>(output[i]) - 1.0f) > 0.001f) { changed = true; break; }
    return changed;
}

bool test_attention_f16() {
    const size_t batch = 1, seq = 8, heads = 2, kv_heads = 2, dim = 16;
    std::vector<__fp16> q(batch * seq * heads * dim), k(batch * seq * kv_heads * dim);
    std::vector<__fp16> v(batch * seq * kv_heads * dim), out(batch * seq * heads * dim);
    fill_random_fp16(q, -0.5f, 0.5f); fill_random_fp16(k, -0.5f, 0.5f); fill_random_fp16(v, -0.5f, 0.5f);
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    cactus_attention_f16(q.data(), k.data(), v.data(), out.data(), batch, seq, seq, heads, kv_heads, dim, scale,
                         nullptr, 0, 0, true, false, false, 0, 0.0f);
    for (size_t i = 0; i < out.size(); i++)
        if (!std::isfinite(static_cast<float>(out[i]))) return false;
    return true;
}

// Double-precision reference for cactus_attention_hybrid_int8_fp16, replicating the incumbent's
// mask semantics exactly (causal kv_end clamp, sliding window, sink exemption on rolled caches).
static void ref_hybrid_attention(
    const __fp16* Q, const int8_t* Kc, const int8_t* Vc,
    const float* ks, const float* vs, const __fp16* Kn, const __fp16* Vn,
    double* out, size_t B, size_t seq, size_t cache_len, size_t new_len,
    size_t qh, size_t kvh, size_t hd, size_t pos_off,
    bool causal, size_t window, size_t qg)
{
    const size_t kv_seq = cache_len + new_len, gqa = qh / kvh, ngk = hd / qg;
    const size_t cache_abs_offset = pos_off >= cache_len ? pos_off - cache_len : 0;
    const double scale = 1.0 / static_cast<double>(sqrtf(static_cast<float>(hd)));
    std::vector<double> sc(kv_seq);
    for (size_t b = 0; b < B; b++)
    for (size_t h = 0; h < qh; h++)
    for (size_t q = 0; q < seq; q++) {
        const __fp16* qv = Q + ((b * seq + q) * qh + h) * hd;
        double* ov = out + ((b * seq + q) * qh + h) * hd;
        const size_t kv_head = h / gqa;
        const size_t abs_q = pos_off + q;
        const size_t kv_start = (window > 0 && abs_q > window) ? abs_q - window : 0;
        const size_t kv_end = causal ? std::min(kv_seq, cache_len + q + 1) : kv_seq;
        std::fill(sc.begin(), sc.end(), -1e300);
        for (size_t kv = 0; kv < kv_end; kv++) {
            bool wm = false;
            if (window > 0 && kv_start > 0) {
                if (kv < cache_len) {
                    if (cache_abs_offset == 0 || kv >= 4) wm = (cache_abs_offset + kv < kv_start);
                } else {
                    wm = (kv + cache_abs_offset < kv_start);
                }
            }
            if ((causal && kv > abs_q) || wm) continue;
            double s = 0.0;
            if (kv < cache_len) {
                const int8_t* kr = Kc + ((b * cache_len + kv) * kvh + kv_head) * hd;
                const float* kss = ks + (kv * kvh + kv_head) * ngk;   // scales: no batch stride (matches kernel)
                for (size_t g = 0; g < ngk; g++)
                    for (size_t d = g * qg; d < (g + 1) * qg; d++)
                        s += static_cast<double>(static_cast<float>(qv[d])) * kr[d] * kss[g];
            } else {
                const __fp16* kr = Kn + ((b * new_len + (kv - cache_len)) * kvh + kv_head) * hd;
                for (size_t d = 0; d < hd; d++)
                    s += static_cast<double>(static_cast<float>(qv[d])) * static_cast<float>(kr[d]);
            }
            sc[kv] = s * scale;
        }
        double m = -1e300;
        for (size_t kv = 0; kv < kv_end; kv++) m = std::max(m, sc[kv]);
        for (size_t d = 0; d < hd; d++) ov[d] = 0.0;
        if (m <= -1e300) continue;
        double den = 0.0;
        for (size_t kv = 0; kv < kv_end; kv++) {
            if (sc[kv] <= -1e300) { sc[kv] = 0.0; continue; }
            sc[kv] = std::exp(sc[kv] - m);
            den += sc[kv];
        }
        if (den <= 0.0) continue;
        for (size_t kv = 0; kv < kv_end; kv++) {
            if (sc[kv] == 0.0) continue;
            const double p = sc[kv];
            if (kv < cache_len) {
                const int8_t* vr = Vc + ((b * cache_len + kv) * kvh + kv_head) * hd;
                const float* vss = vs + (kv * kvh + kv_head) * ngk;
                for (size_t g = 0; g < ngk; g++)
                    for (size_t d = g * qg; d < (g + 1) * qg; d++)
                        ov[d] += p * vr[d] * vss[g];
            } else {
                const __fp16* vr = Vn + ((b * new_len + (kv - cache_len)) * kvh + kv_head) * hd;
                for (size_t d = 0; d < hd; d++)
                    ov[d] += p * static_cast<double>(static_cast<float>(vr[d]));
            }
        }
        for (size_t d = 0; d < hd; d++) ov[d] /= den;
    }
}

// Differential + oracle test for the SME2 prefill attention path (cached-int8 segment via SMOPA
// tiles) against the incumbent NEON flash kernel. Cases cover: global layer mid-prefill, sliding
// window with rolled cache (sink exemption live), sliding window at cache_abs_offset==0, and
// partial q-tile + partial 64-kv block + head_dim 128 + batch 2.
bool test_attention_hybrid_sme() {
    struct Case { size_t B, cache_len, seq, window, pos_off, hd, qh, kvh; };
    const Case cases[] = {
        {1, 640, 128, 0,   640, 256, 8, 4},
        {1, 603, 128, 512, 768, 256, 8, 4},
        {1, 600, 128, 512, 600, 256, 8, 4},
        {2, 137, 40,  0,   137, 128, 4, 2},
    };
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> i8d(-127, 127);
    std::uniform_real_distribution<float> scd(0.005f, 0.02f);
    bool all_ok = true;
    for (const auto& cs : cases) {
        const size_t qg = 32, ngk = cs.hd / qg, new_len = cs.seq;
        std::vector<__fp16> Q(cs.B * cs.seq * cs.qh * cs.hd);
        std::vector<__fp16> Kn(cs.B * new_len * cs.kvh * cs.hd), Vn(cs.B * new_len * cs.kvh * cs.hd);
        std::vector<int8_t> Kc(cs.B * cs.cache_len * cs.kvh * cs.hd), Vc(cs.B * cs.cache_len * cs.kvh * cs.hd);
        std::vector<float> ks(cs.cache_len * cs.kvh * ngk), vs(cs.cache_len * cs.kvh * ngk);
        fill_random_fp16(Q, -1.0f, 1.0f);
        fill_random_fp16(Kn, -1.0f, 1.0f);
        fill_random_fp16(Vn, -1.0f, 1.0f);
        for (auto& x : Kc) x = static_cast<int8_t>(i8d(gen));
        for (auto& x : Vc) x = static_cast<int8_t>(i8d(gen));
        for (auto& x : ks) x = scd(gen);
        for (auto& x : vs) x = scd(gen);

        const size_t osz = cs.B * cs.seq * cs.qh * cs.hd;
        std::vector<__fp16> out_neon(osz), out_sme(osz);
        cactus_quant_set_backend(1);
        cactus_attention_hybrid_int8_fp16(Q.data(), Kc.data(), Vc.data(), ks.data(), vs.data(),
            Kn.data(), Vn.data(), out_neon.data(), cs.B, cs.seq, cs.cache_len, new_len,
            cs.qh, cs.kvh, cs.hd, 0.0f, cs.pos_off, true, cs.window, qg, cs.hd);
        cactus_quant_set_backend(0);
        cactus_attention_hybrid_int8_fp16(Q.data(), Kc.data(), Vc.data(), ks.data(), vs.data(),
            Kn.data(), Vn.data(), out_sme.data(), cs.B, cs.seq, cs.cache_len, new_len,
            cs.qh, cs.kvh, cs.hd, 0.0f, cs.pos_off, true, cs.window, qg, cs.hd);

        std::vector<double> ref(osz);
        ref_hybrid_attention(Q.data(), Kc.data(), Vc.data(), ks.data(), vs.data(),
            Kn.data(), Vn.data(), ref.data(), cs.B, cs.seq, cs.cache_len, new_len,
            cs.qh, cs.kvh, cs.hd, cs.pos_off, true, cs.window, qg);

        double ref_amax = 0.0, e_neon = 0.0, e_sme = 0.0, d_ns = 0.0;
        for (size_t i = 0; i < osz; i++) {
            ref_amax = std::max(ref_amax, std::fabs(ref[i]));
            e_neon = std::max(e_neon, std::fabs(static_cast<float>(out_neon[i]) - ref[i]));
            e_sme = std::max(e_sme, std::fabs(static_cast<float>(out_sme[i]) - ref[i]));
            d_ns = std::max(d_ns, static_cast<double>(std::fabs(
                static_cast<float>(out_sme[i]) - static_cast<float>(out_neon[i]))));
        }
        const double rel_neon = e_neon / std::max(ref_amax, 1e-12);
        const double rel_sme = e_sme / std::max(ref_amax, 1e-12);
        const bool engaged_expected = cactus_quant_sme_available() != 0;
        bool ok = rel_sme < 0.05 && rel_sme < 5.0 * std::max(rel_neon, 0.002);
        if (engaged_expected && d_ns == 0.0) ok = false;   // SME path silently not taken
        std::cout << "    hybrid_sme cache=" << cs.cache_len << " seq=" << cs.seq
                  << " win=" << cs.window << " off=" << cs.pos_off << " hd=" << cs.hd
                  << " B=" << cs.B << ": rel_neon=" << rel_neon << " rel_sme=" << rel_sme
                  << " maxdiff=" << d_ns << (ok ? "" : "  <-- FAIL") << "\n";
        all_ok = all_ok && ok;
    }
    cactus_quant_set_backend(0);
    return all_ok;
}

bool run_benchmarks() {
    auto bench = [](const char* label, auto fn) {
        fn();
        Timer t;
        for (int i = 0; i < 100; i++) fn();
        double ms = t.elapsed_ms() / 100.0;
        std::cout << "  ⚡ " << std::left << std::setw(30) << label
                  << std::fixed << std::setprecision(3) << ms << " ms\n";
    };

    {
        const size_t b = 1024, d = 1024;
        std::vector<__fp16> in(b * d), w(d), out(b * d);
        fill_random_fp16(in); for (size_t i = 0; i < d; i++) w[i] = static_cast<__fp16>(1.0f);
        bench("rms_norm 1024x1024", [&]{ cactus_rms_norm_f16(in.data(), w.data(), out.data(), b, d, 1e-6f); });
    }
    {
        const size_t b = 1024, d = 1024;
        std::vector<__fp16> in(b * d), w(d), bias(d), out(b * d);
        fill_random_fp16(in);
        for (size_t i = 0; i < d; i++) { w[i] = static_cast<__fp16>(1.0f); bias[i] = static_cast<__fp16>(0.0f); }
        bench("layer_norm 1024x1024", [&]{ cactus_layer_norm_f16(in.data(), w.data(), bias.data(), out.data(), b, d, 1e-5f); });
    }
    {
        const size_t rows = 1024, cols = 1024;
        std::vector<__fp16> in(rows * cols), out(rows * cols);
        fill_random_fp16(in, -5.0f, 5.0f);
        bench("softmax 1024x1024", [&]{ cactus_softmax_f16(in.data(), out.data(), rows, 1, cols); });
    }
    {
        const size_t b = 1, s = 256, h = 16, d = 128;
        std::vector<__fp16> in(b * s * h * d), out(b * s * h * d);
        fill_random_fp16(in, -0.5f, 0.5f);
        bench("rope 256x16x128", [&]{ cactus_rope_f16(in.data(), out.data(), b, s, h, d, 0, 10000.0f); });
    }
    {
        const size_t b = 1, s = 256, h = 16, kv = 8, d = 128;
        std::vector<__fp16> q(b * s * h * d), k(b * s * kv * d), v(b * s * kv * d), out(b * s * h * d);
        fill_random_fp16(q, -0.3f, 0.3f); fill_random_fp16(k, -0.3f, 0.3f); fill_random_fp16(v, -0.3f, 0.3f);
        float sc = 1.0f / std::sqrt(static_cast<float>(d));
        bench("attention seq=256 h=16 d=128", [&]{
            cactus_attention_f16(q.data(), k.data(), v.data(), out.data(), b, s, s, h, kv, d, sc,
                                 nullptr, 0, 0, true, false, false, 0, 0.0f);
        });
    }
    // Hybrid int8 prefill attention at Gemma 4 E2B chunked-prefill shapes: NEON vs SME backend
    for (size_t window : {size_t(0), size_t(512)}) {
        const size_t B = 1, cache_len = 896, seq = 128, qh = 8, kvh = 4, hd = 256, qg = 32;
        const size_t ngk = hd / qg, pos_off = window ? cache_len + 128 : cache_len;
        std::mt19937 g2(7);
        std::uniform_int_distribution<int> i8d(-127, 127);
        std::uniform_real_distribution<float> scd(0.005f, 0.02f);
        std::vector<__fp16> Q(B * seq * qh * hd), Kn(B * seq * kvh * hd), Vn(B * seq * kvh * hd);
        std::vector<int8_t> Kc(B * cache_len * kvh * hd), Vc(B * cache_len * kvh * hd);
        std::vector<float> ks(cache_len * kvh * ngk), vs(cache_len * kvh * ngk);
        fill_random_fp16(Q, -1.0f, 1.0f); fill_random_fp16(Kn, -1.0f, 1.0f); fill_random_fp16(Vn, -1.0f, 1.0f);
        for (auto& x : Kc) x = static_cast<int8_t>(i8d(g2));
        for (auto& x : Vc) x = static_cast<int8_t>(i8d(g2));
        for (auto& x : ks) x = scd(g2);
        for (auto& x : vs) x = scd(g2);
        std::vector<__fp16> out(B * seq * qh * hd);
        auto run = [&]{
            cactus_attention_hybrid_int8_fp16(Q.data(), Kc.data(), Vc.data(), ks.data(), vs.data(),
                Kn.data(), Vn.data(), out.data(), B, seq, cache_len, seq,
                qh, kvh, hd, 0.0f, pos_off, true, window, qg, hd);
        };
        char label[96];
        cactus_quant_set_backend(1);
        snprintf(label, sizeof label, "hybrid prefill c=896 win=%zu NEON", window);
        bench(label, run);
        cactus_quant_set_backend(0);
        snprintf(label, sizeof label, "hybrid prefill c=896 win=%zu SME", window);
        bench(label, run);
        cactus_quant_set_backend(0);
    }
    return true;
}

int main() {
    TestRunner runner("Attention, RoPE & Normalization");
    runner.run_test("rms_norm", test_rms_norm());
    runner.run_test("layer_norm", test_layer_norm());
    runner.run_test("softmax", test_softmax());
    runner.run_test("rope", test_rope());
    runner.run_test("attention_f16", test_attention_f16());
    runner.run_test("attention_hybrid_sme", test_attention_hybrid_sme());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
