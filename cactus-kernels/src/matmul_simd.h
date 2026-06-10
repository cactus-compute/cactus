#pragma once
// Declarations bridging the NEON TUs and the SME2 TUs; leaves are __arm_locally_streaming
// (self-transition + own ZA), so they call as ordinary functions. Stubbed without SME2.
#include <cstdint>

extern "C" {

// Streaming svcntb (0 without SME2); every layout is pinned to SVL 64 and dispatch gates on it.
uint32_t cactus_sme2_svl_bytes(void);

// M=1 per-group SMOPA over one 4-channel block; raw int32 partials, caller rescales in NEON.
void cactus_sme_cq_gemv_4col_s8(const int8_t* act_i8, const int8_t* w_block,
                                int32_t* partials_out, uint32_t num_groups, uint32_t gs);

// M>1 full 16x16 ZA tile; act_packed and w_sme both [num_groups][gs/4][16][4] int8.
void cactus_sme_cq_gemm_16x16_s8(const int8_t* act_packed, const int8_t* w_sme,
                                 int32_t* partials_out, uint32_t num_groups, uint32_t gs,
                                 uint32_t m_rows);

// LUTI4/ZT0 GEMV over packed panels: raw int32 partials [sb][group][64], zt_table byte 4*i
// = cb_i8[i]. Defined in matmul_sme2_gemv.cpp (-O1: clang miscompiles vg1x4 dots at -O2+).
void cactus_sme_cq_gemv_luti4_s8(const int8_t* act_i8, const uint8_t* esme_packed,
                                 const uint8_t* zt_table, int32_t* partials,
                                 uint32_t num_groups, uint32_t gs,
                                 uint32_t sb_start, uint32_t sb_count);

// SMOPA GEMM over one (16-row M-tile, 64-channel super-block) pair of packed panels;
// act_packed [num_groups][gs/4][16][4], partials [num_groups][16][64].
void cactus_sme_cq_gemm_luti4_s8(const int8_t* act_packed, const uint8_t* esme_sb,
                                 const uint8_t* zt_table, int32_t* partials,
                                 uint32_t num_groups, uint32_t gs);

// Prefill QK: 16 q-rows x whole cached segment with FLAT scales, one ZA readout per 64-kv
// block; qpack [hd/4][16][4], kpack [hd/4][4][16][4] per block, partials [block][16][64].
void cactus_sme_attn_qk_seg(const int8_t* qpack, const int8_t* kpack, int32_t* partials,
                            uint32_t dim_quads, uint32_t num_blocks);

// Prefill AV: one 64-dim slice, u8 P x s8 V (USMOPA — softmax >= 0), per-block readout;
// ppack [block][2][16][16][4] u8, vpack [block][16][4][16][4] s8, out [block][16][64].
void cactus_sme_attn_av_pass(const uint8_t* ppack, const int8_t* vpack, int32_t* out,
                             uint32_t num_blocks);

} // extern "C"
