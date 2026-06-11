// LUTI4/ZT0 leaves emitting raw int32 partials to memory: ZA->core reads drain the shared
// in-order SME queue (stores do not). -O1 TU only: clang miscompiles vg1x4 dots at -O2+ (SIGILL).
#include "matmul_simd.h"

#if defined(__ARM_FEATURE_SME2)
#include <arm_sme.h>

__arm_locally_streaming __arm_new("za") __arm_new("zt0")
void cactus_sme_cq_gemv_luti4_s8(const int8_t* act_i8, const uint8_t* esme_packed,
                                 const uint8_t* zt_table, int32_t* partials,
                                 uint32_t num_groups, uint32_t gs,
                                 uint32_t sb_start, uint32_t sb_count) {
    svldr_zt(0, zt_table);
    const svbool_t pg = svptrue_b8();
    const svbool_t pg32 = svptrue_b32();
    const uint32_t nkg = gs >> 2;
    const size_t g_stride = (size_t)nkg * 128;          // packed bytes per group panel
    const size_t sb_stride = (size_t)num_groups * g_stride;

    #define CACTUS_LUTI4_DOT(SL, P, OFF, AV, IDX) do { \
        svint8x2_t e0_ = svluti4_lane_zt_s8_x2(0, svld1_u8(pg, (P) + (OFF)), 0); \
        svint8x2_t e1_ = svluti4_lane_zt_s8_x2(0, svld1_u8(pg, (P) + (OFF) + 64), 0); \
        svdot_lane_za32_s8_vg1x4(SL, \
            svcreate4_s8(svget2_s8(e0_, 0), svget2_s8(e0_, 1), \
                         svget2_s8(e1_, 0), svget2_s8(e1_, 1)), AV, IDX); \
    } while (0)
    // vg1 ZA vector v stores at hor(tile 0, slice (v&3)*4 + (v>>2)); fire-and-forget.
    #define CACTUS_STORE_QUAD(S, DST) do { \
        svst1_hor_za32(0, (((S) + 0) & 3u) * 4u + (((S) + 0) >> 2), pg32, (DST)); \
        svst1_hor_za32(0, (((S) + 1) & 3u) * 4u + (((S) + 1) >> 2), pg32, (DST) + 16); \
        svst1_hor_za32(0, (((S) + 2) & 3u) * 4u + (((S) + 2) >> 2), pg32, (DST) + 32); \
        svst1_hor_za32(0, (((S) + 3) & 3u) * 4u + (((S) + 3) >> 2), pg32, (DST) + 48); \
    } while (0)

    for (uint32_t s = 0; s < sb_count; ++s) {
        const uint8_t* wsb = esme_packed + (size_t)(sb_start + s) * sb_stride;
        int32_t* psb = partials + (size_t)s * num_groups * 64;
        svzero_za();
        uint32_t g = 0;
        for (; g + 3 < num_groups; g += 4) {        // 4 groups on slice-quads 0/4/8/12 = 4 chains
            const int8_t* aA = act_i8 + (size_t)g * gs;
            const uint8_t* wA = wsb + (size_t)g * g_stride;
            for (uint32_t kk = 0; kk < gs; kk += 16) {
                svint8_t avA = svld1rq_s8(pg, aA + kk);
                svint8_t avB = svld1rq_s8(pg, aA + gs + kk);
                svint8_t avC = svld1rq_s8(pg, aA + 2 * gs + kk);
                svint8_t avD = svld1rq_s8(pg, aA + 3 * gs + kk);
                const uint8_t* pA = wA + (size_t)(kk >> 2) * 128;
                const uint8_t* pB = pA + g_stride;
                const uint8_t* pC = pB + g_stride;
                const uint8_t* pD = pC + g_stride;
                CACTUS_LUTI4_DOT(0, pA, 0, avA, 0);
                CACTUS_LUTI4_DOT(4, pB, 0, avB, 0);
                CACTUS_LUTI4_DOT(8, pC, 0, avC, 0);
                CACTUS_LUTI4_DOT(12, pD, 0, avD, 0);
                CACTUS_LUTI4_DOT(0, pA, 128, avA, 1);
                CACTUS_LUTI4_DOT(4, pB, 128, avB, 1);
                CACTUS_LUTI4_DOT(8, pC, 128, avC, 1);
                CACTUS_LUTI4_DOT(12, pD, 128, avD, 1);
                CACTUS_LUTI4_DOT(0, pA, 256, avA, 2);
                CACTUS_LUTI4_DOT(4, pB, 256, avB, 2);
                CACTUS_LUTI4_DOT(8, pC, 256, avC, 2);
                CACTUS_LUTI4_DOT(12, pD, 256, avD, 2);
                CACTUS_LUTI4_DOT(0, pA, 384, avA, 3);
                CACTUS_LUTI4_DOT(4, pB, 384, avB, 3);
                CACTUS_LUTI4_DOT(8, pC, 384, avC, 3);
                CACTUS_LUTI4_DOT(12, pD, 384, avD, 3);
            }
            int32_t* pd = psb + (size_t)g * 64;
            CACTUS_STORE_QUAD(0, pd);
            CACTUS_STORE_QUAD(4, pd + 64);
            CACTUS_STORE_QUAD(8, pd + 128);
            CACTUS_STORE_QUAD(12, pd + 192);
            if (g + 4 < num_groups) svzero_za();    // in-order queue: runs after the stores drain
        }
        for (uint32_t t = 0; g < num_groups; ++g, ++t) {   // <=3 tail groups; slices zeroed above
            const int8_t* a = act_i8 + (size_t)g * gs;
            const uint8_t* w = wsb + (size_t)g * g_stride;
            const uint32_t sl = t * 4u;
            for (uint32_t kk = 0; kk < gs; kk += 16) {
                svint8_t av = svld1rq_s8(pg, a + kk);
                const uint8_t* p = w + (size_t)(kk >> 2) * 128;
                CACTUS_LUTI4_DOT(sl, p, 0, av, 0);
                CACTUS_LUTI4_DOT(sl, p, 128, av, 1);
                CACTUS_LUTI4_DOT(sl, p, 256, av, 2);
                CACTUS_LUTI4_DOT(sl, p, 384, av, 3);
            }
            CACTUS_STORE_QUAD(sl, psb + (size_t)g * 64);
        }
    }
    #undef CACTUS_LUTI4_DOT
    #undef CACTUS_STORE_QUAD
}

// GEMM leaf: per (group, kg) one 64B act load, two luti4_x2, four SMOPA (one per ZA tile);
// act_packed [num_groups][gs/4][16][4], partials [num_groups][16][64], NEON rescales after.
__arm_locally_streaming __arm_new("za") __arm_new("zt0")
void cactus_sme_cq_gemm_luti4_s8(const int8_t* act_packed, const uint8_t* esme_sb,
                                 const uint8_t* zt_table, int32_t* partials,
                                 uint32_t num_groups, uint32_t gs) {
    svldr_zt(0, zt_table);
    const svbool_t pg = svptrue_b8();
    const svbool_t pg32 = svptrue_b32();
    const uint32_t nkg = gs >> 2;
    for (uint32_t g = 0; g < num_groups; ++g) {
        const int8_t* ag = act_packed + (size_t)g * gs * 16;
        const uint8_t* wg = esme_sb + (size_t)g * nkg * 128;
        svzero_za();
        for (uint32_t kg = 0; kg < nkg; ++kg) {
            svint8_t zn = svld1_s8(pg, ag + (size_t)kg * 64);
            const uint8_t* p = wg + (size_t)kg * 128;
            svint8x2_t e0 = svluti4_lane_zt_s8_x2(0, svld1_u8(pg, p), 0);
            svint8x2_t e1 = svluti4_lane_zt_s8_x2(0, svld1_u8(pg, p + 64), 0);
            svmopa_za32_s8_m(0, pg, pg, zn, svget2_s8(e0, 0));
            svmopa_za32_s8_m(1, pg, pg, zn, svget2_s8(e0, 1));
            svmopa_za32_s8_m(2, pg, pg, zn, svget2_s8(e1, 0));
            svmopa_za32_s8_m(3, pg, pg, zn, svget2_s8(e1, 1));
        }
        int32_t* pgrp = partials + (size_t)g * 16 * 64;
        for (uint32_t r = 0; r < 16; ++r) {
            int32_t* dst = pgrp + (size_t)r * 64;
            svst1_hor_za32(0, r, pg32, dst);
            svst1_hor_za32(1, r, pg32, dst + 16);
            svst1_hor_za32(2, r, pg32, dst + 32);
            svst1_hor_za32(3, r, pg32, dst + 48);
        }
    }
}

#else  // ---- compile-time fallback ----
void cactus_sme_cq_gemv_luti4_s8(const int8_t*, const uint8_t*, const uint8_t*, int32_t* partials,
                                 uint32_t num_groups, uint32_t, uint32_t, uint32_t sb_count) {
    for (uint32_t i = 0; i < sb_count * num_groups * 64u; ++i) partials[i] = 0;
}
void cactus_sme_cq_gemm_luti4_s8(const int8_t*, const uint8_t*, const uint8_t*, int32_t* partials,
                                 uint32_t num_groups, uint32_t) {
    for (uint32_t i = 0; i < num_groups * 16u * 64u; ++i) partials[i] = 0;
}
#endif
