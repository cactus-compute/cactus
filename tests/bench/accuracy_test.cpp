// Multi-seed FP32 reference accuracy test for the optimized hybrid INT8/FP16
// decode kernel. Runs N seeds across multiple cache lengths and reports
// max nrmse / max-abs vs an FP32 reference.

#include "bench_common.h"
#include "../../cactus-kernels/cactus_kernels.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace bench;

namespace {

struct AccuracyStats {
    float max_nrmse = 0.0f;
    float mean_nrmse = 0.0f;
    float max_abs = 0.0f;
    int n = 0;
};

static AccuracyStats run_one(size_t cache_len, int seeds, AttnDims dims) {
    AccuracyStats out;
    const size_t kvl = cache_len + 1;
    const size_t hd = dims.head_dim;
    const size_t qh = dims.num_q_heads;
    const size_t kvh = dims.num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    for (int s = 0; s < seeds; ++s) {
        std::mt19937 gen(0xC0FFEEu + cache_len * 7919u + s * 13u);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        // Reference layout: [head, seq, head_dim].
        std::vector<float> Q_full(qh * 1 * hd);
        std::vector<float> K_full(kvh * kvl * hd);
        std::vector<float> V_full(kvh * kvl * hd);
        for (auto& x : Q_full) x = dist(gen);
        for (auto& x : K_full) x = dist(gen);
        for (auto& x : V_full) x = dist(gen);

        // FP32 reference: standard scaled dot-product attention (causal).
        std::vector<float> ref(qh * hd);
        reference_attention_fp32(Q_full.data(), K_full.data(), V_full.data(),
                                 ref.data(), qh, kvh, 1, kvl, hd, scale);

        // Cactus kernel layout: [seq, head, head_dim] — transpose from reference.
        std::vector<__fp16> q_fp16(qh * hd);
        for (size_t h = 0; h < qh; ++h)
            for (size_t d = 0; d < hd; ++d)
                q_fp16[h * hd + d] = static_cast<__fp16>(Q_full[h * 1 * hd + d]);

        // K/V cache as int8 (group=32) over the first cache_len positions; the
        // last position (kvl-1) is the new fp16 token. Cactus layout: [kv, head, dim].
        const size_t groups_per_token = hd / kGroupSize;
        std::vector<int8_t> k_cached(kvh * cache_len * hd);
        std::vector<int8_t> v_cached(kvh * cache_len * hd);
        std::vector<float>  k_scales(kvh * cache_len * groups_per_token);
        std::vector<float>  v_scales(kvh * cache_len * groups_per_token);

        for (size_t kv = 0; kv < cache_len; ++kv) {
            for (size_t h = 0; h < kvh; ++h) {
                // Reference K/V layout: [head, seq, dim] → flat = h * kvl * hd + kv * hd
                const float* kr = K_full.data() + h * kvl * hd + kv * hd;
                const float* vr = V_full.data() + h * kvl * hd + kv * hd;
                // Cactus storage layout: [kv, head, dim].
                int8_t* kdst = k_cached.data() + (kv * kvh + h) * hd;
                int8_t* vdst = v_cached.data() + (kv * kvh + h) * hd;
                float* ksdst = k_scales.data() + (kv * kvh + h) * groups_per_token;
                float* vsdst = v_scales.data() + (kv * kvh + h) * groups_per_token;
                for (size_t g = 0; g < groups_per_token; ++g) {
                    float kmax = 0.0f, vmax = 0.0f;
                    for (size_t d = 0; d < kGroupSize; ++d) {
                        kmax = std::max(kmax, std::abs(kr[g * kGroupSize + d]));
                        vmax = std::max(vmax, std::abs(vr[g * kGroupSize + d]));
                    }
                    float ksc = kmax / 127.0f;
                    float vsc = vmax / 127.0f;
                    ksdst[g] = ksc;
                    vsdst[g] = vsc;
                    float kinv = ksc > 0.0f ? 127.0f / kmax : 0.0f;
                    float vinv = vsc > 0.0f ? 127.0f / vmax : 0.0f;
                    for (size_t d = 0; d < kGroupSize; ++d) {
                        kdst[g * kGroupSize + d] = static_cast<int8_t>(std::lrintf(
                            std::max(-127.0f, std::min(127.0f, kr[g * kGroupSize + d] * kinv))));
                        vdst[g * kGroupSize + d] = static_cast<int8_t>(std::lrintf(
                            std::max(-127.0f, std::min(127.0f, vr[g * kGroupSize + d] * vinv))));
                    }
                }
            }
        }

        // Last KV position is fp16 new token. Reference layout for that
        // position is [head, kv=cache_len, dim]; cactus expects [new_pos, head, dim].
        std::vector<__fp16> k_new(kvh * hd), v_new(kvh * hd);
        for (size_t h = 0; h < kvh; ++h) {
            const float* kr = K_full.data() + h * kvl * hd + cache_len * hd;
            const float* vr = V_full.data() + h * kvl * hd + cache_len * hd;
            for (size_t d = 0; d < hd; ++d) {
                k_new[h * hd + d] = static_cast<__fp16>(kr[d]);
                v_new[h * hd + d] = static_cast<__fp16>(vr[d]);
            }
        }

        std::vector<__fp16> out_fp16(qh * hd);
        cactus_attention_hybrid_int8_fp16(
            q_fp16.data(),
            k_cached.data(), v_cached.data(),
            k_scales.data(), v_scales.data(),
            k_new.data(), v_new.data(),
            out_fp16.data(),
            1, 1, cache_len, 1,
            qh, kvh, hd,
            scale, cache_len, true, 0, kGroupSize);

        // Cactus output layout: [seq, head, dim]. Reference layout: [head, seq, dim].
        std::vector<float> got(qh * hd);
        for (size_t h = 0; h < qh; ++h)
            for (size_t d = 0; d < hd; ++d)
                got[h * 1 * hd + d] = static_cast<float>(out_fp16[h * hd + d]);

        AccuracyResult ar = check_accuracy(ref.data(), got.data(), got.size(), 1.0f);
        out.max_nrmse = std::max(out.max_nrmse, ar.nrmse);
        out.mean_nrmse += ar.nrmse;
        out.max_abs = std::max(out.max_abs, ar.max_abs_error);
        out.n++;
    }
    if (out.n) out.mean_nrmse /= out.n;
    return out;
}

} // namespace

int main() {
    AttnDims dims;
    dims.head_dim = 128;
    dims.num_q_heads = 8;
    dims.num_kv_heads = 8;
    const std::vector<size_t> cache_lens = {32, 128, 512, 1024, 2048, 4096};
    const int seeds = 8;

    std::cout << "Hybrid INT8/FP16 decode — accuracy vs FP32 reference\n";
    std::cout << "head_dim=128 q_heads=8 kv_heads=8 group=" << kGroupSize
              << " seeds=" << seeds << "\n\n";
    std::cout << std::left << std::setw(12) << "cache_len"
              << std::setw(14) << "max_nrmse"
              << std::setw(14) << "mean_nrmse"
              << std::setw(14) << "max_abs_err" << "\n";
    std::cout << std::string(54, '-') << "\n";

    bool all_ok = true;
    constexpr float kNoiseFloor = 0.005f;  // Empirically int8 group=32 noise floor.

    for (size_t cl : cache_lens) {
        AccuracyStats st = run_one(cl, seeds, dims);
        std::cout << std::left << std::setw(12) << cl
                  << std::fixed << std::setprecision(6)
                  << std::setw(14) << st.max_nrmse
                  << std::setw(14) << st.mean_nrmse
                  << std::setw(14) << st.max_abs << "\n";
        if (st.max_nrmse > kNoiseFloor) all_ok = false;
    }

    std::cout << "\nNoise floor threshold: " << kNoiseFloor << "\n";
    std::cout << (all_ok ? "PASS — within noise floor at all cache lengths\n"
                         : "FAIL — exceeded noise floor\n");
    return all_ok ? 0 : 1;
}
