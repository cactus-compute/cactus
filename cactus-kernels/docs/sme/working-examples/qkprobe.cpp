// Probe: SMOPA QK^T scores tile — 16 q-rows x 64 kv-positions, head_dim=256, K int8 with per-32
// group scales, Q int8-quantized per-32-group. Oracle: fp32 dequant dot. Layout reuses the proven
// GEMM conventions: zn = [16 q x 4 dim] act-pack, zm = [64 kv...]: per dim-group of 4, four
// 16-kv weight vectors -> za0..3 (= kv 0-15,16-31,32-47,48-63).
#include <arm_sme.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

__arm_locally_streaming __arm_new("za")
static void qk_tile(const int8_t* qpack, const int8_t* kpack, int32_t* out,
                    uint32_t dim_groups /*per quant-group: 32/4=8*/, uint32_t n_qgroups,
                    const float* qs, const float* ks, float* scores /*16x64 fp32*/) {
    const svbool_t pg = svptrue_b8();
    const svbool_t pg32 = svptrue_b32();
    for (uint32_t qg = 0; qg < n_qgroups; ++qg) {
        svzero_za();
        const int8_t* qb = qpack + (size_t)qg * dim_groups * 64;
        const int8_t* kb = kpack + (size_t)qg * dim_groups * 256;
        for (uint32_t dg = 0; dg < dim_groups; ++dg) {
            svint8_t zn = svld1_s8(pg, qb + (size_t)dg * 64);
            const int8_t* p = kb + (size_t)dg * 256;
            svmopa_za32_s8_m(0, pg, pg, zn, svld1_s8(pg, p));
            svmopa_za32_s8_m(1, pg, pg, zn, svld1_s8(pg, p + 64));
            svmopa_za32_s8_m(2, pg, pg, zn, svld1_s8(pg, p + 128));
            svmopa_za32_s8_m(3, pg, pg, zn, svld1_s8(pg, p + 192));
        }
        for (uint32_t r = 0; r < 16; ++r) {
            int32_t buf[64];
            svst1_hor_za32(0, r, pg32, buf);
            svst1_hor_za32(1, r, pg32, buf + 16);
            svst1_hor_za32(2, r, pg32, buf + 32);
            svst1_hor_za32(3, r, pg32, buf + 48);
            for (uint32_t c = 0; c < 64; ++c)
                scores[r * 64 + c] += buf[c] * qs[r * n_qgroups + qg] * ks[c * n_qgroups + qg];
        }
    }
    (void)out;
}

int main() {
    const uint32_t HD = 256, QG = 32, NQG = HD / QG, DG = QG / 4;
    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> Q(16 * HD);
    std::vector<int8_t> K(64 * HD);
    std::vector<float> ks(64 * NQG);
    for (auto& v : Q) v = dist(gen);
    for (auto& v : K) v = (int8_t)(gen() % 250 - 125);
    for (auto& v : ks) v = 0.01f + 0.005f * std::fabs(dist(gen));

    // quantize Q per row per 32-group
    std::vector<int8_t> qi(16 * HD);
    std::vector<float> qs(16 * NQG);
    for (int r = 0; r < 16; r++)
        for (uint32_t g = 0; g < NQG; g++) {
            float mx = 0;
            for (uint32_t d = 0; d < QG; d++) mx = std::max(mx, std::fabs(Q[r * HD + g * QG + d]));
            float s = mx / 127.f + 1e-12f;
            qs[r * NQG + g] = s;
            for (uint32_t d = 0; d < QG; d++)
                qi[r * HD + g * QG + d] = (int8_t)lrintf(Q[r * HD + g * QG + d] / s);
        }
    // oracle: dequant fp32 dot of the QUANTIZED q (isolates SMOPA correctness from q-quant error)
    std::vector<float> ref(16 * 64, 0.f);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 64; c++)
            for (uint32_t g = 0; g < NQG; g++) {
                float acc = 0;
                for (uint32_t d = 0; d < QG; d++)
                    acc += (float)qi[r * HD + g * QG + d] * (float)K[c * HD + g * QG + d];
                ref[r * 64 + c] += acc * qs[r * NQG + g] * ks[c * NQG + g];
            }
    // pack: qpack[qg][dg][16r][4d]; kpack[qg][dg][4 vec][16kv][4d]
    std::vector<int8_t> qpack(16 * HD), kpack(64 * HD);
    for (uint32_t qg = 0; qg < NQG; qg++)
        for (uint32_t dg = 0; dg < DG; dg++) {
            for (int r = 0; r < 16; r++)
                for (int d = 0; d < 4; d++)
                    qpack[(qg * DG + dg) * 64 + r * 4 + d] = qi[r * HD + qg * QG + dg * 4 + d];
            for (int c = 0; c < 64; c++)
                for (int d = 0; d < 4; d++)
                    kpack[(qg * DG + dg) * 256 + (c / 16) * 64 + (c % 16) * 4 + d] =
                        K[c * HD + qg * QG + dg * 4 + d];
        }
    std::vector<float> scores(16 * 64, 0.f);
    qk_tile(qpack.data(), kpack.data(), nullptr, DG, NQG, qs.data(), ks.data(), scores.data());
    double mx = 0;
    for (int i = 0; i < 16 * 64; i++) mx = std::max(mx, (double)std::fabs(scores[i] - ref[i]));
    printf("QK SMOPA tile vs oracle: max_err=%.6f -> %s\n", mx, mx < 1e-3 ? "PASS" : "FAIL");
    return mx >= 1e-3;
}
