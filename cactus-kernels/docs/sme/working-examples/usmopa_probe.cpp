// Probe: svusmopa_za32_u8_m semantics on M4 — za32[i][j] += sum_{c<4} u8(zn[4i+c]) * s8(zm[4j+c])?
#include <arm_sme.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

__arm_locally_streaming __arm_new("za")
static void usmopa_tile(const uint8_t* zn_b, const int8_t* zm_b, int32_t* out) {
    const svbool_t pg = svptrue_b8();
    const svbool_t pg32 = svptrue_b32();
    svzero_za();
    svuint8_t zn = svld1_u8(pg, zn_b);
    svint8_t zm = svld1_s8(pg, zm_b);
    svusmopa_za32_u8_m(0, pg, pg, zn, zm);
    for (uint32_t r = 0; r < 16; ++r)
        svst1_hor_za32(0, r, pg32, out + (size_t)r * 16);
}

int main() {
    std::mt19937 gen(3);
    uint8_t zn[64]; int8_t zm[64];
    for (auto& v : zn) v = (uint8_t)(gen() % 256);          // full unsigned range incl. >127
    for (auto& v : zm) v = (int8_t)((int)(gen() % 255) - 127);
    int32_t out[256];
    usmopa_tile(zn, zm, out);
    int64_t maxerr = 0;
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) {
            int32_t ref = 0;
            for (int c = 0; c < 4; c++) ref += (int32_t)zn[4*i+c] * (int32_t)zm[4*j+c];
            int64_t e = (int64_t)out[i*16+j] - ref;
            if (e < 0) e = -e;
            if (e > maxerr) maxerr = e;
        }
    printf("USMOPA u8xs8 probe: max_err=%lld -> %s\n", (long long)maxerr, maxerr == 0 ? "PASS" : "FAIL");
    return maxerr != 0;
}
