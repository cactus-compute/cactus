# KleidiAI SME2 INT8×INT4 mopa matmul — extraction + mapping onto Cactus CQ4

Source: local clone `/tmp/sme-refs/kleidiai` (ARM-software/kleidiai).
Target Cactus file: `cactus-kernels/src/matmul.cpp`.

The primary micro-kernel studied is the **symmetric** dynamic-INT8 LHS × 4-bit-32-block RHS f32-output SME2 mopa kernel, because it is the closest analogue to Cactus CQ4 (both LHS and RHS are *symmetric* per-block scaled int with no zero-point / row-sum correction in the hot path):

```
kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa
```

Its paired packers (confirmed from `test/tests/matmul_clamp_f32_qsi8d32p_qsi4c32p_test.cpp:93-94, 118-119`):

| operand | packer |
|---|---|
| LHS (dyn-quant + pack) | `kai_lhs_quant_pack_qsi8d32p_f32_neon` |
| RHS (pack pre-quant int4) | `kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon` |

A second, *asymmetric* variant `kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa` exists (qai8dxp = dynamic int8 with a per-row **zero-point + offset**, RHS carries per-column **row-sum + bias**). Its hot path is an external asm symbol (`kai_kernel_...`, not inlined). It is *less* aligned to Cactus (Cactus is symmetric, no zero-point) so I treat qsi8d32p as the reference and only note qai8dxp's extra correction terms where relevant.

---

## 0. Tile geometry (mr/nr/kr/sr, block depth, SVL)

From the qsi8d32p kernel `.c` (lines 20-37) and headers:

```
kai_m_step = 1   (× SVLs)      kai_n_step = 4   (× SVLs)
kai_mr     = 1   (× SVLs)      kai_nr     = 4   (× SVLs)
kai_kr     = 4                 kai_sr     = 2
kai_bl     = 32                (quant block length, fixed multiple of 32)
```

`SVLs = kai_get_sme_vector_length_u32() = SVL_bytes/4` = number of **fp32 lanes** per Z/ZA tile row (`kai_common.h:235`). On a 512-bit SME implementation SVLs = 16. Therefore one mopa macro-tile is:

* **M rows  = mr = 1·SVLs**   (e.g. 16 output rows)
* **N cols  = nr = 4·SVLs**   (e.g. 64 output cols), produced as **4 ZA tiles** `za0..za3`, each SVLs wide.
* The integer SMOPA consumes **int8** operands; the int4 RHS is widened to int8 *inside* the kernel via `luti4`/`ZT0` LUT before each mopa, so the accumulation tile depth per `smopa` is `kr = 4` along K (one 32-bit word of 4×int8 per K-step). `sr = 2` means a kr=4 block is sub-split into 2 nibble groups during packing (the s1s0 nibble interleave, see §1).

`bl=32` (one quant block) ⇒ each K-block contributes `bl/kr = 32/4 = 8` mopa iterations per ZA tile.

### Packed strides
LHS (`.c:42-44, 59-62`): per block = `bl·1 (int8) + 2 (fp16 scale)` bytes; `lhs_packed_stride = mr · num_blocks · (bl+2)`. **Scales live in a trailing region** after all the int8 values: `lhs_scales = lhs_packed + stride − mr·num_blocks·2` (`.c:175-176`).

RHS (`.c:46-50, 64-75`): per block = `bl/2 (4-bit) + 2 (fp16 scale)` bytes; `rhs_packed_stride = nr · num_blocks · (bl/2 + 2)`. Scales trailing: `rhs_scales = rhs_packed + stride − nr·num_blocks·2` (`.c:177-178`). (The symmetric kernel carries **no** row-sum / bias; only the asymmetric qai8dxp variant appends `nr·4` rsum + `nr·4` bias per row, `qai8dxp...c:114-118`.)

---

## 1. RHS 4-bit packing layout (`kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon`)

Input RHS is already quantized to **qsi4c32** in *source* (`s16s0`) order, NxK with the layout *per row, per block*: `[uint16 fp16 scale][bl/2 bytes of packed nibbles]`, where the source nibble order is `s16s0` ("stride-16, start-0"): within a 32-element block the byte at offset `b` holds element `b` in the low nibble and element `b+16` in the high nibble.

The packer does two things:

**(a) Nibble re-order `s16s0 → s1s0`** (`convert_*` helpers, `.c:34-152`).  Output `s1s0` means *sequential* nibble order packed two-per-byte: output uint16 holds elements `{k, k+1, k+2, k+3}` as nibbles. The bl==32 fast path (`.c:110-120`):
```c
const uint8x16_t v0 = vld1q_u8(src_blk);          // 16 src bytes = 32 nibbles, s16s0
const uint8x16_t even_dup = vuzp1q_u8(v0, v0);
const uint8x16_t odd_dup  = vuzp2q_u8(v0, v0);
const uint8x16_t lo_pairs_dup = vsliq_n_u8(even_dup, odd_dup, 4);  // low half: seq pairs
const uint8x16_t hi_pairs_dup = vsriq_n_u8(odd_dup, even_dup, 4);  // high half: seq pairs
const uint8x16_t packed = vcombine_u8(vget_low_u8(lo_pairs_dup), vget_low_u8(hi_pairs_dup));
vst1q_u8(dst_bytes, packed);                       // 16 bytes = 32 nibbles, s1s0
```

**(b) Interleave 4 N-rows at uint16 granularity** (`interleave_rows4_u16`, `.c:276-311`). The packed-row destination stores, **column-major within a block over nr**, one uint16 (=4 nibbles along K) from each of the 4 consecutive N-rows side by side:
```c
const uint64_t packed0 = (uint64_t)row0_ptr[0] | (row1_ptr[0]<<16) | (row2_ptr[0]<<32) | (row3_ptr[0]<<48);
*((uint64_t*)dst_ptr) = packed0;          // row0..row3 nibble-quad #0
dst_ptr += 2*nr;                          // next K-quad column, stride nr (in uint16)
```
So the in-block layout is: for each of the `bl/4` K-quad columns, `nr` uint16 values (one per N-row in the nr group), i.e. **`[k-quad][n-row]` with n-row contiguous**. Block stride = `bl4 · nr` uint16. After all `num_blocks` blocks, the per-(nr,block) **fp16 scales** are written contiguously in a trailing region (`.c:469-476`), packed 4-rows-per-uint64.

### Why this matches the mopa loads
In the kernel inner loop (`.c:270`) the RHS is read `ld1w {z2.s-z3.s}` = 2 vectors of 32-bit words. Each 32-bit word = 8 nibbles for one N-column laid along K; the `nr`-contiguous interleave means consecutive lanes are consecutive N-columns. `luti4 {z4.b-z5.b}, zt0, z2[0]` (`.c:277`) expands those nibbles to int8 via the ZT0 LUT (`lut[16] = {-8..7}`, `.c:40`), producing the int8 RHS operand for `smopa`.

**Nibble value convention:** packer XORs each output nibble with `0x8` (`s1s0scalef16` stores raw signed nibble; the generic `qsu4c32s1s0` variant does `dst_q ^ 0x8888`, `.c:241`, converting signed→unsigned-offset-8). The kernel's LUT then maps the stored nibble index back to the signed int8 value. **Low nibble = lower K index, high nibble = next K index** (sequential s1s0).

---

## 2. LHS dynamic INT8 quant+pack (`kai_lhs_quant_pack_qsi8d32p_f32_neon`)

Per M-row, per 32-block (`.c:101-126`):
```c
float amax = max|lhs|;                 // per block (32 elems)
float sf   = amax / 127;               // symmetric scale
float sf_inv = sf ? 1/sf : 0;
for (bl_idx = 0; bl_idx < bl; bl_idx += kr)          // kr = 4
  for (kr_idx = 0; kr_idx < kr; ++kr_idx)
    lhs_packed_ptr[kr_idx] = (int8)round(lhs[kr_idx] * sf_inv);   // symmetric int8
  lhs_packed_ptr += mr * kr;                          // interleave mr rows, kr-quad stride
lhs_packed_scales[0] = kai_cast_f16_f32(sf);          // fp16 scale, trailing region
lhs_packed_scales += mr;
```
Layout: **kr=4 int8 values per M-row interleaved across `mr` rows** (`ptr += mr*kr`), so the packed LHS is `[k-quad][m-row]`-interleaved exactly mirroring the RHS `[k-quad][n-row]` interleave. fp16 per-block scales sit in a trailing region (`lhs_packed + stride − mr·num_blocks·2`). The kernel reads LHS as `ld1h {z8.h}` (`.c:273`) — i.e. it reads **pairs of int8 as 16-bit lanes** and feeds them directly to `smopa za, z8.b, z4.b`.

---

## 3. SMOPA accumulation loop (the hot path)

Full structure from `.c:184-441`. Loop nest: **N (n_step=4·SVLs) → M (m_step=1·SVLs) → K-block (bl=32) → in-block (kr=4)**.

Setup once:
```asm
smstart                       ; enter streaming SVE + enable ZA
cntw x14                      ; x14 = SVLs (fp32 lanes)
ptrue p0.b, all
ldr  zt0, [x6]                ; ZT0 = int4->int8 LUT  (lut = {-8..7})
ld1rw z15.s = scalar_min ; ld1rw z17.s = scalar_max
```

Per K-block (`.c:252-289`):
```asm
3:                                       ; .LOOP_K_START
  zero {za}                              ; clear all 4 ZA.s tiles
  ld1w {z0.s-z1.s}, pn8/z, [x17], x21    ; RHS fp16 block-scales (2 vec)
  zip  {z0.h-z1.h}, z0.h, z1.h           ; interleave to per-lane fp16 scale pairs
  mov  x11, bl                           ; in-block counter = 32
4:                                       ; .LOOP_BL_START  (8 iters: 32/4)
  ld1w {z2.s-z3.s}, pn8/z, [x16], x20    ; RHS packed int4 (2 vec = 64 nibbles/col grp)
  ld1h {z8.h}, p0/z, [x22, x20, lsl #1]  ; LHS int8 (read as 16-bit lanes)
  inch x20, all
  luti4 {z4.b-z5.b}, zt0, z2[0]          ; int4 -> int8 (ZT0 LUT) for za0,za1 cols
  luti4 {z6.b-z7.b}, zt0, z3[0]          ; int4 -> int8 for za2,za3 cols
  smopa za0.s, p0/m, p0/m, z8.b, z4.b    ; INT8 outer-product accumulate, signed
  smopa za1.s, p0/m, p0/m, z8.b, z5.b
  smopa za2.s, p0/m, p0/m, z8.b, z6.b
  smopa za3.s, p0/m, p0/m, z8.b, z7.b
  subs x11, x11, #4                      ; advance kr=4 along K
  b.gt 4b
```
`smopa za.s, Zn.b, Zm.b` is the **signed 8-bit→32-bit MOPA**: it computes `za[i][j] += sum_{p=0..3} Zn.b[i*4+p] * Zm.b[j*4+p]` — i.e. each ZA fp32 lane is a 4-deep int8 dot-product outer product. Over the 8 in-block iterations the K=32 block is fully reduced into the int32 ZA tile. **No scaling is applied during integer accumulation** — ZA holds raw int32 dot products.

### Post-mopa scaling (per K-block, in ZA→Z drain, `.c:294-357`)
```asm
ld1b {z16.b}, p4/z, [x23, x21]           ; LHS fp16 block-scales (this block, mr rows)
inch x21, all
5:                                       ; .LOOP_ZA  (over mr output rows)
  pnext p3.h ; clastb z19.h, p3, z19.h, z16.h   ; select this row's LHS fp16 scale, broadcast
  mova {z28.b-z31.b}, za0h.b[w12, 0:3]   ; read 4 ZA fp32 lanes (int32 results) for row
  scvtf {z28.s-z31.s}, {z28.s-z31.s}     ; int32 -> fp32
  ; combined scale = LHS_scale(fp16) * RHS_scale(fp16), widened fp16->fp32:
  movprfx z8,z18 ; fmlalb z8.s,  z19.h, z0.h    ; z8  = lhs_s * rhs_s (lane lo, za0)
  movprfx z9,z18 ; fmlalb z9.s,  z19.h, z1.h    ;          ... za1
  movprfx z10,z18; fmlalt z10.s, z19.h, z0.h    ;          ... za2 (top fp16 lane)
  movprfx z11,z18; fmlalt z11.s, z19.h, z1.h    ;          ... za3
  cmp x10, K ; b.ne 6f                    ; first K-block? init : accumulate
  fmul z24.s,  z8.s, z28.s                ; first block: result = scale * int_result
  fmul z25.s,  z9.s, z29.s
  fmul z26.s, z10.s, z30.s
  fmul z27.s, z11.s, z31.s
  b 7f
6:                                        ; .ACCUMULATE (subsequent K-blocks)
  ld1w {z24.s-z27.s}, pn9/z, [x25]        ; load partial f32 result
  fmla z24.s, p0/m, z8.s,  z28.s          ; result += scale * int_result
  fmla z25.s, p0/m, z9.s,  z29.s
  fmla z26.s, p0/m, z10.s, z30.s
  fmla z27.s, p0/m, z11.s, z31.s
7: st1w {z24.s-z27.s}, pn9, [x25]         ; store f32 partial to DST
   add x25, x25, dst_stride_row
   cmp x12, x15 ; blt 5b
```
Key points:
* `za0h.b[w12,0:3]` drains **4 fp32 lanes (one row of the 1·SVLs output, 4 ZA tiles wide)** at a time.
* The **per-block** combined fp32 scale is `lhs_scale[m-row] * rhs_scale[n-col]`, both fp16, multiplied via `fmlalb`/`fmlalt` (fp16-widening). `fmlalb` uses even/low fp16 lanes (za0/za1), `fmlalt` the odd/high lanes (za2/za3) — matching the `zip` of the two RHS scale vectors.
* The f32 *partial* result is **accumulated in DST memory across K-blocks** (`fmla` into a reloaded `ld1w`), not in registers. This is how per-block scales are applied **after** each integer mopa but **before** summing blocks: each block's `int32 · (s_lhs·s_rhs)` is added into the running f32 output.

### Rescale to f32 + clamp (`.c:333-336, 365-388`)
The int32→f32 conversion is `scvtf`, the f32 scale is `fmul`/`fmla` as above (so DST already holds correctly-scaled f32). After the K loop, an optional clamp pass:
```asm
ld1w {z24.s-z27.s}, pn9/z, [x24]
fclamp {z24.s-z27.s}, z15.s, z17.s        ; SME2 fclamp to [scalar_min, scalar_max]
st1w {z24.s-z27.s}, pn9, [x24]
```
`is_clamp_valid` is 0 when min=-FLT_MAX & max=+FLT_MAX, skipping the pass.

### qai8dxp (asymmetric) deltas, for completeness
The asymmetric variant additionally: (a) LHS pack stores a per-row **fp32 scale + int32 zero-point/offset** (`kai_num_bytes_multiplier_lhs=4`, `..._offset_lhs=4`); (b) RHS pack stores a per-column **row-sum (fp32)** and **bias (fp32)** (`qai8dxp...c:75-76, 114-118`); the int mopa result is corrected by `lhs_offset · rhs_rowsum` before fp32 scaling and `+bias`. Cactus is symmetric, so these are **not** needed — qsi8d32p is the right template.

---

## 4. Mapping onto Cactus CQ4

### 4.1 What Cactus has today (the "expanded" INT8 weight + INT8 acts dot-product GEMM)

`src/matmul.cpp` M>1 path (`cactus_quant_matmul`, lines ~1672-1801) and `tq_preexpand_weights` (~line 1440):

* **Weights** are dequantized from k-bit codebook indices to **int8 via the int8-quantized codebook** (`tq_quantize_codebook_i8`, line 1233: `cb_i8[i]=round(codebook[i]/cb_scale)`, `cb_scale=max|codebook|/127`). `tq_expand_i8_16` (line 1290) gathers codebook int8 through `vqtbl1q_s8(cb_lut, idx)`.
* The expanded int8 weights are **4 N-rows interleaved at 32-bit granularity** by `tq_interleave_4x_s8` (line 115): given row0..row3 int8x16, it produces an output where each 16-byte store holds, for one 4-wide K position, `{row0[w], row1[w], row2[w], row3[w]}` 32-bit words. Layout written to `w_il` at `(nb·num_groups+g)·gs·4`: **`[K][4 N-rows]` with the 4 rows contiguous at 32-bit stride**, group-major, `gs*4` bytes per (Nblk,group). This is *exactly* `nr=4`-style N-row interleave but at **NEON 4-lane** granularity, not SME `nr=4·SVLs`.
* **Per-(Nblk,group) norms** are `n_f32[(nb·num_groups+g)·4 + ni] = norms[...] · cb_scale` (line 1429-1432). So the codebook int8 scale (`cb_scale`) is **folded into the weight norm** — the weight's effective fp32 scale is `norm · cb_scale`.
* **Activations** are Hadamard-transformed (`cactus_quant_transform_hadamard_activations`), then per-group dynamically quantized to int8 (`tq_quantize_group_i8`, line 1247: `scale=max|x|/127`, symmetric) producing `act_i8` + per-group fp32 `act_scales`. This is **identical in spirit to qsi8d32p**: symmetric int8, per-group (=per-block) fp32 scale.
* **The GEMM core** (lines 1745-1786): for each group, loads 8 interleaved weight vectors `b00..b13` (covering K=32 × 4 N-rows) and per-M-row activation `a_lo/a_hi`, and does `vdotq_laneq_s32` (NEON SDOT) into `row_acc` int32. Then drains: `running_sum += f32(row_acc) · (norms_v · act_scale)`. **This is the NEON-dot analogue of the SME2 mopa**: int8·int8 → int32 dot, then `int32 · (weight_norm·cb_scale) · act_scale` → f32. The block depth per SDOT lane is 4 (= `kr`), exactly matching SMOPA's 4-deep int8 accumulation.

### 4.2 Operand correspondence (Cactus ↔ KleidiAI mopa)

| concept | Cactus CQ4 | KleidiAI qsi8d32p mopa |
|---|---|---|
| LHS = activations | int8, symmetric, per-group (`gs`) fp32 scale `act_scales` | int8, symmetric, per-block (`bl=32`) **fp16** scale |
| LHS pack layout | `[K][m-row]` not pre-packed; rows read with `a_rows[mi]+g·gs+k` | `[k-quad][m-row]` interleaved, `ptr += mr·kr`; fp16 scales trailing |
| RHS = weights | **already int8** ("expanded"), `[K][4 N-rows]` @32-bit, per-(Nblk,grp) fp32 `norm·cb_scale` | int4 packed `[k-quad][n-row]` @uint16, fp16 scale; widened to int8 in-kernel by `luti4` |
| K-reduce unit | `vdotq_laneq_s32` 4-deep int8 dot | `smopa` 4-deep int8 outer-product |
| N-row tile | 4 (NEON lanes) | nr = 4·SVLs |
| M-row tile | TILE_M = 8 | mr = 1·SVLs |
| post-int scaling | `f32(acc)·(norm·cb_scale)·act_scale` | `scvtf(za)·(s_lhs_fp16·s_rhs_fp16)`, accumulated per block |
| group/block size | `gs` (≥32, ≤256) | `bl` (multiple of 32; kernel reduces 32 at a time) |

### 4.3 How a SMOPA path would consume re-tiled Cactus operands

Cactus' "expanded" buffer is **already int8** — so unlike KleidiAI it does **not** need `luti4` (the int4→int8 LUT) at all; the SMOPA can consume Cactus int8 weights directly with `smopa za, z_act.b, z_wt.b`, skipping the LUT widening. The codebook int8 scale (`cb_scale`) is already folded into `n_f32`, so the f32 drain math `scvtf(za)·(norm·cb_scale)·act_scale` maps 1:1 onto KleidiAI's `scvtf(za)·(s_lhs·s_rhs)` with `s_rhs = norm·cb_scale`, `s_lhs = act_scale`.

**Re-tiling required to feed SMOPA:**

1. **N-row interleave widening: 4 → nr = 4·SVLs.** Cactus interleaves exactly 4 N-rows (`tq_interleave_4x_s8`) because NEON SDOT processes 4 output columns (4×int32 lanes). SMOPA's ZA tile is `nr = 4·SVLs` wide across 4 ZA tiles. The expanded weights must be re-laid as `[k-quad][n-col]` with **n-col contiguous over the full nr=4·SVLs**, split into 4 ZA-tile groups of SVLs columns each (mirroring KleidiAI's `interleave_rows4_u16` but at int8 and 4·SVLs width). Concretely: for each K-quad, store `nr` int8 columns contiguously; the per-K-quad stride feeds one `ld1w {z2.s-z3.s}` per 2 ZA tiles.

2. **K interleave at kr=4 (32-bit words).** Both Cactus and SMOPA use a 4-deep int8 reduction. Cactus' `[K][4 N-rows]@32-bit` already groups K into 4-lane SDOT words; the SMOPA layout wants `[k-quad][n-col]` so that one `ld1w` lane = one 32-bit word = 4 K-values for one N-column. The transform is a transpose of Cactus' current "4-N-row-major within a K-word" to "N-col-major across the nr group, K-quad-major" — i.e. **swap the inner (4 N-row) grouping for an (nr N-col) grouping**, keeping the 4-deep K word intact.

3. **LHS (activation) re-pack to qsi8d32p form.** Cactus reads activations row-wise unpacked. SMOPA wants `[k-quad][m-row]` interleaved over `mr=1·SVLs` rows (`lhs_pack ... ptr += mr·kr`) with **fp16** per-block scales in a trailing region. So: (a) pack act_i8 into `mr`-row-interleaved kr=4 words, (b) convert `act_scales` (fp32) → fp16 and store trailing. Block length must be 32 (or feed gs in 32-chunks like the kernel's `bl` inner loop).

4. **Scale dtype.** KleidiAI uses **fp16** for both LHS and RHS per-block scales and combines via `fmlalb`/`fmlalt`. Cactus keeps fp32 `n_f32` and `act_scales`. To reuse the KleidiAI mopa verbatim, Cactus would down-convert both to fp16 (precision loss; acceptable since these are per-block amax/127 scales). Alternatively a Cactus-custom drain can keep fp32 scales (`fmul`/`fmla` with `ld1rw`-broadcast fp32) and skip the fp16 widen — preferable given Cactus already has fp32 `norm·cb_scale`.

5. **No int4 stage / no `luti4`.** Because Cactus pre-expands to int8, the SMOPA path **omits** the `luti4 {z4.b-z5.b}, zt0, z2[0]` widening and the ZT0 LUT setup; weights load directly as int8 via `ld1w`/`ld1b`. This is the chief structural simplification vs. KleidiAI's int4 kernel (at the cost of 2× weight memory traffic — Cactus already pays this for the SDOT path). If memory traffic matters, Cactus could instead keep weights as int4 and adopt KleidiAI's `luti4` widening verbatim, packing with `kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon` and folding `cb_scale` into the fp16 RHS scale.

6. **Block-wise f32 accumulation in DST.** KleidiAI accumulates per-block scaled results into DST memory across K-blocks (`fmla` into reloaded `ld1w`). Cactus accumulates in registers (`running_sum[mi]`) across groups. For SMOPA with `gs > 32`, Cactus must either (a) follow KleidiAI and re-load/accumulate DST per 32-block, or (b) keep a register f32 accumulator per (m-row, n-tile) and add each 32-block's `scvtf(za)·s_lhs·s_rhs`. Option (b) matches Cactus' current register-accumulate style and avoids DST round-trips when the `nr×mr` tile fits ZA + Z registers.

### 4.4 Summary of the minimal SMOPA bring-up for Cactus

* Reuse Cactus' int8 expanded weights and int8 acts as-is (symmetric, per-group fp32 scale, `cb_scale` folded into norm).
* Re-tile weights from `[K][4 N-rows]@32-bit` → `[k-quad][nr=4·SVLs N-cols]` across 4 ZA tiles (transpose inner grouping, keep kr=4 K-words).
* Re-pack acts to `[k-quad][mr=1·SVLs M-rows]` interleaved + fp16 (or fp32) trailing scales.
* Inner loop: `ld1w` weights (int8, no `luti4`) + `ld1h` acts → `smopa za0..3` over 8 iters per 32-block → `mova`/`scvtf` drain → `fmul/fmla` by `act_scale · (norm·cb_scale)` → optional `fclamp` → store f32 (or convert to fp16 for Cactus' `__fp16* C`).
* This is a drop-in structural match to `kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa` minus the int4 widening stage.

---

## File/line index (for re-verification)
* mopa hot loop: `/tmp/sme-refs/kleidiai/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.c:184-441` (tile args 20-37, scales 175-178)
* RHS pack: `/tmp/sme-refs/kleidiai/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.c` (nibble convert 34-152, row4 interleave 276-311, driver 398-503)
* LHS quant+pack: `/tmp/sme-refs/kleidiai/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.c:72-131`
* qai8dxp asymmetric variant: `/tmp/sme-refs/kleidiai/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.c` (rsum/bias/zero-point 42-118)
* test pairing: `/tmp/sme-refs/kleidiai/test/tests/matmul_clamp_f32_qsi8d32p_qsi4c32p_test.cpp:93-94,118-119`
* Cactus expand/interleave: `src/matmul.cpp` `tq_interleave_4x_s8`:115, `tq_quantize_codebook_i8`:1233, `tq_quantize_group_i8`:1247, `tq_expand_i8_16`:1290, `tq_preexpand_weights`:1440, M>1 GEMM:1672-1801, dot core:1745-1786
