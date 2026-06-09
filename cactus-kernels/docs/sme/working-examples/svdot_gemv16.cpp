// Probe: is svdot_s32 streaming-legal on Apple M4, and does the broadcast-activation GEMV work?
// Compile: clang++ -O2 -std=c++20 -march=armv8.2-a+fp16+simd+dotprod+i8mm+sme2 svdot.cpp -o svdot && ./svdot
#include <arm_sme.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>

// Streaming GEMV over 16 channels via svdot (all 16 lanes useful, no ZA).
// w packed [K/4][16][4]: w[kg*64 + n*4 + c] = W[n][kg*4 + c].  a: K int8.  y: 16 int32.
__arm_locally_streaming
static void sme_svdot_gemv16(const int8_t* a, const int8_t* w, int32_t* y, uint32_t K) {
    svbool_t pg = svptrue_b8();
    svint32_t acc = svdup_n_s32(0);
    for (uint32_t kg = 0; kg < K / 4; ++kg) {
        uint32_t a4; std::memcpy(&a4, a + kg * 4, 4);
        svint8_t xb = svreinterpret_s8_u32(svdup_n_u32(a4));  // a[kg*4..+3] replicated to all lanes
        svint8_t wv = svld1_s8(pg, w + (size_t)kg * 64);      // 16 channels x 4 K
        acc = svdot_s32(acc, xb, wv);                          // acc[n] += sum_c a[kg*4+c]*W[n][kg*4+c]
    }
    svst1_s32(svptrue_b32(), y, acc);
}

int main() {
    const uint32_t K = 256, Nc = 16;
    std::vector<int8_t> a(K), W((size_t)Nc * K);
    std::mt19937 g(7);
    std::uniform_int_distribution<int> d(-100, 100);
    for (auto& v : a) v = (int8_t)d(g);
    for (auto& v : W) v = (int8_t)d(g);

    std::vector<int32_t> ref(Nc, 0);
    for (uint32_t n = 0; n < Nc; n++) { int32_t s = 0; for (uint32_t k = 0; k < K; k++) s += (int32_t)a[k]*(int32_t)W[n*K+k]; ref[n] = s; }

    std::vector<int8_t> wpack((size_t)Nc * K);
    for (uint32_t kg = 0; kg < K/4; kg++) for (uint32_t n = 0; n < Nc; n++) for (uint32_t c = 0; c < 4; c++)
        wpack[(size_t)kg*64 + n*4 + c] = W[n*K + kg*4 + c];

    std::vector<int32_t> y(Nc, -123);
    sme_svdot_gemv16(a.data(), wpack.data(), y.data(), K);

    int64_t maxerr = 0; for (uint32_t n = 0; n < Nc; n++) maxerr = std::max<int64_t>(maxerr, std::llabs((int64_t)y[n]-ref[n]));
    printf("svdot streaming GEMV16 K=%u: maxerr=%lld -> %s\n", K, (long long)maxerr, maxerr==0?"PASS":"FAIL");
    return maxerr==0?0:1;
}
