// SME2 streaming leaves: integer SMOPA accumulate only — transform, quantize, and fp rescale
// stay in NEON, so partials are bit-identical to the SDOT path. Scalar stubs without SME2.
#include "matmul_simd.h"

#if defined(__ARM_FEATURE_SME2)
#include <arm_sme.h>

// Dispatch gates on SVL == 64; other SVLs fall back to NEON.
__arm_locally_streaming
uint32_t cactus_sme2_svl_bytes(void) {
    return (uint32_t)svcntb();
}

// M=1: one act row vs up to 4 channels per group, ZA row 0 only.
__arm_locally_streaming __arm_new("za")
void cactus_sme_cq_gemv_4col_s8(const int8_t* act_i8, const int8_t* w_block,
                                int32_t* partials_out, uint32_t num_groups, uint32_t gs) {
    const svbool_t pn  = svwhilelt_b8_u32(0u, 4u);    // activation: row 0, 4 K lanes
    const svbool_t pm  = svwhilelt_b8_u32(0u, 16u);   // weights: 4 channels x 4 K
    const svbool_t pst = svwhilelt_b32_u32(0u, 4u);   // store 4 int32 (the 4 channels)
    const uint32_t nkg = gs >> 2;
    for (uint32_t g = 0; g < num_groups; ++g) {
        const int8_t* aptr = act_i8 + (size_t)g * gs;
        const int8_t* wptr = w_block + (size_t)g * gs * 4;
        svzero_za();
        for (uint32_t kg = 0; kg < nkg; ++kg) {
            svint8_t zn = svld1_s8(pn, aptr + (size_t)kg * 4);
            svint8_t zm = svld1_s8(pm, wptr + (size_t)kg * 16);
            svmopa_za32_s8_m(0, pn, pm, zn, zm);
        }
        svst1_hor_za32(0, 0, pst, partials_out + (size_t)g * 4);
    }
}

// M>1: full 16x16 ZA tile over pre-packed [num_groups][gs/4][16][4].
__arm_locally_streaming __arm_new("za")
void cactus_sme_cq_gemm_16x16_s8(const int8_t* act_packed, const int8_t* w_sme,
                                 int32_t* partials_out, uint32_t num_groups, uint32_t gs,
                                 uint32_t m_rows) {
    const svbool_t pn  = svwhilelt_b8_u32(0u, m_rows * 4u);  // m_rows x 4 K
    const svbool_t pm  = svptrue_b8();                       // all 16 channels x 4 K
    const svbool_t pst = svptrue_b32();                      // store 16 int32 columns
    const uint32_t nkg = gs >> 2;
    for (uint32_t g = 0; g < num_groups; ++g) {
        const int8_t* a = act_packed + (size_t)g * gs * 16;
        const int8_t* w = w_sme + (size_t)g * gs * 16;
        svzero_za();
        for (uint32_t kg = 0; kg < nkg; ++kg) {
            svint8_t zn = svld1_s8(pn, a + (size_t)kg * 64);
            svint8_t zm = svld1_s8(pm, w + (size_t)kg * 64);
            svmopa_za32_s8_m(0, pn, pm, zn, zm);
        }
        for (uint32_t i = 0; i < m_rows; ++i)
            svst1_hor_za32(0, i, pst, partials_out + ((size_t)g * 16 + i) * 16);
    }
}

// QK: ZA accumulates the full head_dim per 64-kv block, single readout per block; caller
// rescales by qs[r]*ksflat[c] in NEON.
__arm_locally_streaming __arm_new("za")
void cactus_sme_attn_qk_seg(const int8_t* qpack, const int8_t* kpack, int32_t* partials,
                            uint32_t dim_quads, uint32_t num_blocks) {
    const svbool_t pg   = svptrue_b8();
    const svbool_t pg32 = svptrue_b32();
    for (uint32_t b = 0; b < num_blocks; ++b) {
        svzero_za();
        const int8_t* kb = kpack + (size_t)b * dim_quads * 256;
        for (uint32_t w = 0; w < dim_quads; ++w) {
            svint8_t zn = svld1_s8(pg, qpack + (size_t)w * 64);
            const int8_t* p = kb + (size_t)w * 256;
            svmopa_za32_s8_m(0, pg, pg, zn, svld1_s8(pg, p));
            svmopa_za32_s8_m(1, pg, pg, zn, svld1_s8(pg, p + 64));
            svmopa_za32_s8_m(2, pg, pg, zn, svld1_s8(pg, p + 128));
            svmopa_za32_s8_m(3, pg, pg, zn, svld1_s8(pg, p + 192));
        }
        int32_t* ob = partials + (size_t)b * 1024;
        for (uint32_t r = 0; r < 16; ++r) {
            svst1_hor_za32(0, r, pg32, ob + (size_t)r * 64);
            svst1_hor_za32(1, r, pg32, ob + (size_t)r * 64 + 16);
            svst1_hor_za32(2, r, pg32, ob + (size_t)r * 64 + 32);
            svst1_hor_za32(3, r, pg32, ob + (size_t)r * 64 + 48);
        }
    }
}

// AV: one 64-dim slice, per-block readout (P scale is per row/v-group/block); P is u8 via
// USMOPA. zna feeds dim tiles 0-1, znb tiles 2-3.
__arm_locally_streaming __arm_new("za")
void cactus_sme_attn_av_pass(const uint8_t* ppack, const int8_t* vpack, int32_t* out,
                             uint32_t num_blocks) {
    const svbool_t pg   = svptrue_b8();
    const svbool_t pg32 = svptrue_b32();
    for (uint32_t b = 0; b < num_blocks; ++b) {
        svzero_za();
        const uint8_t* pp = ppack + (size_t)b * 2048;
        const int8_t* vp = vpack + (size_t)b * 4096;
        for (uint32_t kg = 0; kg < 16; ++kg) {
            svuint8_t zna = svld1_u8(pg, pp + (size_t)kg * 64);
            svuint8_t znb = svld1_u8(pg, pp + 1024 + (size_t)kg * 64);
            const int8_t* v = vp + (size_t)kg * 256;
            svusmopa_za32_u8_m(0, pg, pg, zna, svld1_s8(pg, v));
            svusmopa_za32_u8_m(1, pg, pg, zna, svld1_s8(pg, v + 64));
            svusmopa_za32_u8_m(2, pg, pg, znb, svld1_s8(pg, v + 128));
            svusmopa_za32_u8_m(3, pg, pg, znb, svld1_s8(pg, v + 192));
        }
        int32_t* ob = out + (size_t)b * 1024;
        for (uint32_t r = 0; r < 16; ++r) {
            svst1_hor_za32(0, r, pg32, ob + (size_t)r * 64);
            svst1_hor_za32(1, r, pg32, ob + (size_t)r * 64 + 16);
            svst1_hor_za32(2, r, pg32, ob + (size_t)r * 64 + 32);
            svst1_hor_za32(3, r, pg32, ob + (size_t)r * 64 + 48);
        }
    }
}

#else  // ---- compile-time fallback (toolchain/target without SME2) ----
uint32_t cactus_sme2_svl_bytes(void) { return 0; }
void cactus_sme_cq_gemv_4col_s8(const int8_t*, const int8_t*, int32_t* p,
                                uint32_t num_groups, uint32_t) {
    for (uint32_t i = 0; i < num_groups * 4u; ++i) p[i] = 0;
}
void cactus_sme_attn_qk_seg(const int8_t*, const int8_t*, int32_t* p,
                            uint32_t, uint32_t num_blocks) {
    for (uint32_t i = 0; i < num_blocks * 16u * 64u; ++i) p[i] = 0;
}
void cactus_sme_attn_av_pass(const uint8_t*, const int8_t*, int32_t* out, uint32_t num_blocks) {
    for (uint32_t i = 0; i < num_blocks * 16u * 64u; ++i) out[i] = 0;
}
void cactus_sme_cq_gemm_16x16_s8(const int8_t*, const int8_t*, int32_t* p,
                                 uint32_t num_groups, uint32_t, uint32_t) {
    for (uint32_t i = 0; i < num_groups * 16u * 16u; ++i) p[i] = 0;
}
#endif
