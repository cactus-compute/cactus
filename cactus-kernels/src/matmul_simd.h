#pragma once
// Shared declarations bridging the base (NEON, armv8.2-a+...) translation units and the
// SME2 translation unit (matmul_sme2.cpp, compiled with +sme2). The SME leaf kernels below are
// __arm_locally_streaming in their definition (they self-transition to streaming mode + own ZA),
// so they are callable as ordinary functions from non-streaming code — the declaration here is
// plain. Guarded behind runtime cpu_has_sme2() dispatch by the caller; a compile-time stub is
// provided when the TU is built without SME2 so the symbol always exists.
#include <cstdint>

extern "C" {

// Streaming vector length in bytes (svcntb in streaming mode); 0 when built without SME2.
// cpu_has_sme2() gates ALL SME dispatch on this returning 64 — every shipped layout is pinned
// to 512-bit SVL; other implementations fall back to NEON until SVL-parametric variants exist.
uint32_t cactus_sme2_svl_bytes(void);

// CQ GEMV (M=1) per-group SMOPA accumulate, 4 output channels per call-block.
//   act_i8       : [num_groups * gs] int8, per-group quantized activations (group g at +g*gs).
//   w_block      : Cactus 'expanded' INT8 weights for ONE 4-channel N-block; group g panel at
//                  w_block + (size_t)g*gs*4, with each kg-th 16 bytes = 4 channels x 4 K
//                  (identical layout to the NEON SDOT path). Up to `valid_cols` (1..4) channels valid.
//   partials_out : [num_groups * 4] int32 raw dot products; partials_out[g*4 + c] =
//                  sum_k act[g][k] * weight[channel c][k]. Caller applies act_scale * norm rescale.
// Numerically identical (integer) to the NEON SDOT inner loop, so the existing FP rescale reproduces
// the reference exactly.
void cactus_sme_cq_gemv_4col_s8(const int8_t* act_i8, const int8_t* w_block,
                                int32_t* partials_out, uint32_t num_groups, uint32_t gs);

// CQ GEMM full-tile (M>1): up to 16 act-rows x 16 output channels per call — full ZA-tile width.
//   act_packed : [num_groups][gs/4][16][4] (rows). w_sme: [num_groups][gs/4][16][4] (16 channels,
//                gathered from four 4-channel 'expanded' panels). Both contiguous, 64 B per kg.
//   partials_out: [num_groups * 16 rows * 16 cols] int32, [g][i][c] layout.
void cactus_sme_cq_gemm_16x16_s8(const int8_t* act_packed, const int8_t* w_sme,
                                 int32_t* partials_out, uint32_t num_groups, uint32_t gs,
                                 uint32_t m_rows);

// CQ GEMV (M=1): packed 4-bit nibbles + in-engine LUTI4/ZT0 codebook expansion + indexed
// multi-vector dots, emitting RAW INT32 PARTIALS (caller rescales in NEON). Streams half the
// weight bytes of the int8 path and never reads ZA back to the core — both required to compete
// with NEON on Apple's shared in-order SME command queue. Defined in matmul_sme2_gemv.cpp,
// compiled at -O1 (clang -O2 mis-compiles vg1x4 ZA dots -> runtime SIGILL).
//   act_i8 : [num_groups*gs] int8 acts. esme_packed: packed cache (CactusQuantMatrix::expanded_sme).
//   zt_table : 64B ZT0 image, byte 4*i = cb_i8[i].
//   partials : [sb_count][num_groups][64] int32 (s relative to sb_start);
//              partials[s][g][c] = sum_k act[g][k] * cb_i8[idx(n,k)] for channel n = (sb+s)*64+c.
void cactus_sme_cq_gemv_luti4_s8(const int8_t* act_i8, const uint8_t* esme_packed,
                                 const uint8_t* zt_table, int32_t* partials,
                                 uint32_t num_groups, uint32_t gs,
                                 uint32_t sb_start, uint32_t sb_count);

// CQ GEMM: one (16-row M-tile, 64-channel super-block) pair via SMOPA over LUTI4-expanded packed
// nibbles — 4 ZA tiles in flight, no per-call weight gather, partials to memory (NEON rescale).
// Defined in matmul_sme2_gemv.cpp (-O1 TU).
//   act_packed: [num_groups][gs/4][16 rows][4 K] int8 (rows beyond m_rows zero-padded).
//   esme_sb   : expanded_sme + sb*(num_groups*(gs/4)*128).
//   partials  : [num_groups][16 rows][64 ch] int32.
void cactus_sme_cq_gemm_luti4_s8(const int8_t* act_packed, const uint8_t* esme_sb,
                                 const uint8_t* zt_table, int32_t* partials,
                                 uint32_t num_groups, uint32_t gs);

// Prefill attention QK: 16 q-rows x the WHOLE cached segment, FLAT scales — Q is quantized with
// one scale per row and cached K is REQUANTIZED at pre-pack to one scale per kv (per-group scale
// ratios folded into the int8 values), so ZA accumulates across all of head_dim and reads out
// ONCE per 64-kv block (16x64 int32 = 4 KB). The earlier per-qgroup-readout variant was
// store-dominated (512 ZA stores vs 256 SMOPAs per block; profile: 57% of busy time stalled in
// the leaf). Caller rescales scores by qs[r]*ksflat[c] in NEON. Layout = the proven GEMM
// conventions (docs/sme/working-examples/qkprobe.cpp, max_err=0).
//   qpack    : [head_dim/4 dim-quads][16 rows][4 dims] int8 (head_dim*16 B).
//   kpack    : per block [head_dim/4][4 vec][16 kv][4 dims] int8 (head_dim*64 B per block),
//              vec v = kv 16v..16v+15 (za tile v); blocks contiguous.
//   partials : [num_blocks][16 rows][64 kv] int32 raw dots.
void cactus_sme_attn_qk_seg(const int8_t* qpack, const int8_t* kpack, int32_t* partials,
                            uint32_t dim_quads, uint32_t num_blocks);

// Prefill attention AV: ONE 64-dim slice of head_dim over the whole cached segment, PER-BLOCK
// ZA readout. P comes in u8-quantized (USMOPA u8 x s8 — softmax probs are non-negative, so
// unsigned doubles the resolution; probe-verified) with the per-(kv,32-dim-group) V scale folded
// in (two pack variants per slice) and a per-(row, v-group, BLOCK) scale — the caller rescales
// each block's int32 tile by that scale and accumulates in fp32. Verified layout:
// docs/sme/working-examples/avprobe.cpp (max_err=0) + usmopa probe.
//   ppack : [num_blocks][2][16 kv-grp][16 rows][4 kv] u8 (num_blocks*2048 B); variant 0 feeds
//           dim tiles 0-1 (slice dims 0-31), variant 1 feeds tiles 2-3 (dims 32-63).
//   vpack : [num_blocks][16 kv-grp][4 tiles][16 dims][4 kv] int8 (num_blocks*4096 B).
//   out   : [num_blocks][16 rows][64 dims] int32.
void cactus_sme_attn_av_pass(const uint8_t* ppack, const int8_t* vpack, int32_t* out,
                             uint32_t num_blocks);

} // extern "C"
