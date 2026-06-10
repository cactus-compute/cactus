// SME2 streaming-mode leaf kernels for the Cactus CQ quantized matmul.
// Compiled with `-march=...+sme2` (see CMakeLists.txt); falls back to scalar stubs when the toolchain
// lacks SME2 so the symbols always exist and the base library links on any arm64 target.
//
// These functions do ONLY the integer SMOPA accumulate. The Hadamard transform, per-group INT8
// activation quantization, codebook expansion, and the final FP rescale all stay in the non-streaming
// NEON code (matmul.cpp). The integer partial sums produced here are bit-identical to the NEON SDOT
// inner loop, so the existing FP rescale reproduces the FP32 reference oracle exactly.
//
// SMOPA semantics (verified, see docs/sme/working-examples/int8_smopa_matmul.cpp):
//   za32[i][j] += sum_{c=0..3} Zn[i*4+c] * Zm[j*4+c]   (each instruction consumes 4 K-values)
// SVL on Apple M4 = 64 bytes => za32 tile is 16x16 int32, Z vectors hold 16*4 = 64 int8.
#include "matmul_simd.h"

#if defined(__ARM_FEATURE_SME2)
#include <arm_sme.h>

// Streaming vector length probe for the runtime dispatch gate (cpu_has_sme2 requires SVL == 64 B
// because every shipped layout is pinned to it; other SVLs fall back to NEON).
__arm_locally_streaming
uint32_t cactus_sme2_svl_bytes(void) {
    return (uint32_t)svcntb();
}

// CQ GEMV (M=1): single activation row vs up to 4 weight channels, accumulated per group.
// Uses only ZA row 0 (predicated) and columns 0..3 — low utilization but correct; this is the
// M=1 path (perf-deferred). w_block kg-th 16 bytes = 4 channels x 4 K (Cactus 'expanded' layout).
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

// CQ GEMM full-tile: up to 16 act-rows x 16 output channels per call — full ZA tile width
// (pm=ptrue_b8). act_packed and w_sme are both pre-packed [num_groups][gs/4][16][4]; w_sme is
// gathered from four 4-channel 'expanded' panels.
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

// Prefill attention QK with flat per-row/per-kv scales: ZA accumulates over the FULL head_dim per
// 64-kv block (256 SMOPAs for hd=256), single 4 KB readout per block, one streaming entry per
// tile. 409 MACs/queue-op — the fused-GEMM regime. The fp rescale (qs[r]*ksflat[c]) happens in
// NEON after return. Layout verified by qkprobe.cpp (same byte layout, w = qg*8+dg).
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

// Prefill attention AV: one 64-dim slice over all cached 64-kv blocks, PER-BLOCK readout (the P
// quantization scale is per (row, v-group, block) for accuracy — a global scale wastes resolution
// on blocks far below the row max). P is UNSIGNED u8 via USMOPA (softmax probs are >= 0; the sign
// bit would be wasted — doubles resolution, probe-verified exact in usmopa_probe). zna feeds dim
// tiles 0-1 (P folded with v-scale group 0 of the slice), znb feeds tiles 2-3 (group 1). Stores
// are fire-and-forget; MACs/queue-op stays ~512.
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
