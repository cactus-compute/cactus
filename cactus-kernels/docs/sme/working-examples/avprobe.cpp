// Probe: SMOPA AV tile — out[16q][head_dim] += P[16q][64kv] x V[64kv][head_dim], P int8-quantized
// per q-row (softmax probs in [0,1] -> scale = rowmax/127), V int8 with per-32-group scales over
// the HEAD-DIM axis. zn = [16q x 4kv] P-pack, zm = [64-dim chunk... AV contracts over kv: SMOPA
// contraction dim = kv (4 per instr). zm = V[4kv][16 dim-cols] panels per dim-block of 16.
#include <arm_sme.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>

__arm_locally_streaming __arm_new("za")
static void av_tile(const int8_t* ppack /*[16kvg][16q][4kv]*/, const int8_t* vpack,
                    /*[16kvg][4 dimblk][16dim... see main]*/ int32_t* out /*16q x 64dim*/,
                    uint32_t kv_groups /*64/4*/) {
    const svbool_t pg = svptrue_b8();
    const svbool_t pg32 = svptrue_b32();
    svzero_za();
    for (uint32_t kg = 0; kg < kv_groups; ++kg) {
        svint8_t zn = svld1_s8(pg, ppack + (size_t)kg * 64);
        const int8_t* p = vpack + (size_t)kg * 256;
        svmopa_za32_s8_m(0, pg, pg, zn, svld1_s8(pg, p));
        svmopa_za32_s8_m(1, pg, pg, zn, svld1_s8(pg, p + 64));
        svmopa_za32_s8_m(2, pg, pg, zn, svld1_s8(pg, p + 128));
        svmopa_za32_s8_m(3, pg, pg, zn, svld1_s8(pg, p + 192));
    }
    for (uint32_t r = 0; r < 16; ++r) {
        svst1_hor_za32(0, r, pg32, out + (size_t)r * 64);
        svst1_hor_za32(1, r, pg32, out + (size_t)r * 64 + 16);
        svst1_hor_za32(2, r, pg32, out + (size_t)r * 64 + 32);
        svst1_hor_za32(3, r, pg32, out + (size_t)r * 64 + 48);
    }
}

int main() {
    const uint32_t KV = 64, D = 64;   // one 64-dim slice of head_dim (per-32-group scales -> 2 groups)
    std::mt19937 gen(9);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    std::vector<float> P(16 * KV);
    std::vector<int8_t> V(KV * D);
    std::vector<float> vs(KV * 2);    // per kv, per 32-dim group
    for (auto& v : P) v = dist(gen);
    for (auto& v : V) v = (int8_t)(gen() % 250 - 125);
    for (auto& v : vs) v = 0.01f + 0.005f * dist(gen);
    // quantize P per row
    std::vector<int8_t> pi(16 * KV);
    std::vector<float> ps(16);
    for (int r = 0; r < 16; r++) {
        float mx = 0; for (uint32_t c = 0; c < KV; c++) mx = std::max(mx, P[r * KV + c]);
        ps[r] = mx / 127.f + 1e-12f;
        for (uint32_t c = 0; c < KV; c++) pi[r * KV + c] = (int8_t)lrintf(P[r * KV + c] / ps[r]);
    }
    // oracle on quantized P (isolate SMOPA correctness): out[r][d] = sum_kv pi*V * ps[r]*vs[kv][d/32]
    std::vector<float> ref(16 * D, 0.f);
    for (int r = 0; r < 16; r++)
        for (uint32_t d = 0; d < D; d++)
            for (uint32_t c = 0; c < KV; c++)
                ref[r * D + d] += (float)pi[r * KV + c] * (float)V[c * D + d] * ps[r] * vs[c * 2 + d / 32];
    // CAUTION: per-(kv,dimgroup) scales can't factor out of the kv-contraction! vs depends on c ->
    // must FOLD vs into V before quantize... can't (V already int8). => per-kv scales break the
    // single-SMOPA contraction. Test the FOLDED variant: rescale per kv inside by splitting the
    // contraction per kv?? That defeats SMOPA. INSTEAD: common production trick — V groups are over
    // DIM (d/32), NOT kv... vs[c][g] varies per kv c. SMOPA sum_kv pi*V needs vs constant over kv.
    // => the AV SMOPA requires per-dim-group-only V scales. Check: if engine's v_scales are
    // per (kv, group) this probe documents the blocker. Compute the error of assuming mean scale:
    std::vector<int8_t> ppack(16 * KV), vpack(KV * D);
    for (uint32_t kg = 0; kg < KV / 4; kg++) {
        for (int r = 0; r < 16; r++)
            for (int c4 = 0; c4 < 4; c4++)
                ppack[kg * 64 + r * 4 + c4] = pi[r * KV + kg * 4 + c4];
        for (uint32_t d = 0; d < D; d++)
            for (int c4 = 0; c4 < 4; c4++)
                vpack[kg * 256 + (d / 16) * 64 + (d % 16) * 4 + c4] = V[(kg * 4 + c4) * D + d];
    }
    std::vector<int32_t> out(16 * 64);
    av_tile(ppack.data(), vpack.data(), out.data(), KV / 4);
    // exact check possible when vs constant per kv: set vs constant and re-derive both
    double mx = 0;
    std::vector<float> ref2(16 * D, 0.f);
    for (int r = 0; r < 16; r++)
        for (uint32_t d = 0; d < D; d++) {
            for (uint32_t c = 0; c < KV; c++)
                ref2[r * D + d] += (float)pi[r * KV + c] * (float)V[c * D + d];
            float got = out[r * 64 + d];
            mx = std::max(mx, (double)std::fabs(got - ref2[r * D + d]));
        }
    printf("AV SMOPA tile (unit scales) vs oracle: max_err=%.6f -> %s\n", mx, mx < 0.5 ? "PASS" : "FAIL");
    printf("NOTE: per-(kv,group) V scales cannot factor out of a kv-contraction SMOPA — integration\n"
           "must rescale per 4-kv subgroup or restructure (documented).\n");
    return mx >= 0.5;
}
