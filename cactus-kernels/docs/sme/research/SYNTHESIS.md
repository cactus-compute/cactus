# SME2 CQ4 Matmul — Implementation Synthesis (single source of truth)

> Consolidates `research/*.md` + `api-notes.md` + `gotchas.md` + `working-examples/*` into one
> actionable plan for `src/matmul_sme2.cpp`. **Verified** = compiled+run (or asm-checked) on *this*
> M4 Pro / Apple clang 17.0.0; **Unverified** = from docs/KleidiAI source, not yet exercised on HW.
>
> Target box: Apple M4 Pro, Apple clang 17.0.0, SVL = 512 bit = 64 B (16 FP32 / 64 INT8 lanes),
> `FEAT_SME2=1`, `FEAT_SME_F16F16=0` ⇒ **all MOPA accumulates into FP32/INT32 `za32`**.

---

## 1. Verified intrinsics + `-march` (the load-bearing facts)

### `-march` string

| String | Status | Note |
|---|---|---|
| `armv8.2-a+fp16+simd+dotprod+i8mm+sme2` | **VERIFIED — USE THIS** | `[ran-correct]` both example matmuls PASS, max_err=0. The TU flag for `matmul_sme2.cpp`. |
| `armv8-a+sme` / `-mcpu=apple-m4` | **VERIFIED runs** | also run-correct; less explicit about i8mm/fp16. |
| `armv9-a+sme2` | **VERIFIED BROKEN at runtime** | compiles + emits *identical* asm but **SIGILLs (exit 132)** on M4 — `armv9-a` implies non-streaming SVE2 that Apple does not implement. ASM-inspection only; NEVER ship. |

Build wiring: per-TU `set_source_files_properties(src/matmul_sme2.cpp PROPERTIES COMPILE_OPTIONS
"-march=armv8.2-a+fp16+simd+dotprod+i8mm+sme2")`. Library-wide flag (CMakeLists line 34) stays
`armv8.2-a+fp16+simd+dotprod+i8mm`. Double-guard the TU with `#if defined(__ARM_FEATURE_SME2)`.
Header: `#include <arm_sme.h>` (pulls in `arm_sve.h`).

### Intrinsics (names/signatures all VERIFIED against this box's `arm_sme.h` + LLVM CodeGen tests)

| Purpose | Intrinsic (exact) | Emits | Status |
|---|---|---|---|
| FP32 FMOPA | `void svmopa_za32_f32_m(uint64_t tile, svbool_t pn, svbool_t pm, svfloat32_t zn, svfloat32_t zm)` | `fmopa za<t>.s,p/m,p/m,z.s,z.s` | **VERIFIED ran** |
| INT8 SMOPA | `void svmopa_za32_s8_m(uint64_t tile, svbool_t pn, svbool_t pm, svint8_t zn, svint8_t zm)` | `smopa za<t>.s,p/m,p/m,z.b,z.b` | **VERIFIED ran** |
| FP16→FP32 MOPA | `svmopa_za32_f16_m(...)` | `fmopa ...,z.h,z.h` (2-way widen, za32 form; F16F16=0) | **VERIFIED asm** |
| BF16→FP32 MOPA | `svmopa_za32_bf16_m(...)` | `bfmopa ...,z.h,z.h` | **VERIFIED asm** |
| ZA zero | `void svzero_za(void)` (or `svzero_mask_za(255)`) | `zero {za}` | **VERIFIED asm** |
| ZA readout (USE THIS) | `void svst1_hor_za32(uint64_t tile, uint32_t slice, svbool_t pg, void* ptr)` | `st1w {za0h.s[wN,0]},p,[ptr]` | **VERIFIED ran** |
| ZA→Z readout (for in-kernel rescale) | `svfloat32_t svread_hor_za32_f32_m(svfloat32_t inactive, svbool_t pg, uint64_t tile, uint32_t slice)` / `..._s32_m` | `mov z.s,p/m,za0h.s[wN,0]` | **VERIFIED asm** |
| Vertical add-accumulate (GEMV reduce option) | `void svaddva_za32_s32_m(uint64_t tile, svbool_t pn, svbool_t pm, svint32_t)` | `addva` | present (Unverified use) |
| Streaming-VL counts | `svcntw()`=16 (FP32 lanes), `svcntb()`=64 (INT8 lanes), `svcntsw()`/`svcntsb()` same in-stream | — | **VERIFIED ran** |
| Predicates | `svptrue_b8()`, `svptrue_b32()`; `svwhilelt_b32_u64(i,N)`, `svwhilelt_b8(i,N)` | `ptrue`/`whilelt` | **VERIFIED ran** |

> ⚠ Arg order pitfall (verified): `svst1_hor_za32` is `(tile, slice, pg, ptr)` — tile first, NOT pg
> first. `svread_hor_za32_*_m` is `(inactive, pg, tile, slice)`.

### Streaming / ZA attributes (VERIFIED, position matters)

- **Leaf kernel callable straight from NEON code:** `__arm_new("za") __arm_locally_streaming` in the
  **PREFIX** position (these are *declaration* attrs). Auto-brackets `smstart sm`/`smstart za` … `smstop`.
  Both working examples use exactly this; it's the recommended Cactus leaf-kernel form.
- `__arm_streaming`, `__arm_inout("za")`, `__arm_in/out/preserves("za")` are **type** attrs — go
  **AFTER** the param list. Putting `__arm_streaming` in prefix = hard compile error.
- Runtime gate: `sysctlbyname("hw.optional.arm.FEAT_SME2")==1` (or `__arm_has_sme()`) before dispatch,
  else SIGILL on non-SME Macs.

---

## 2. Recommended SMOPA tiling for SVL=64B on M4

Geometry (verified): ZA = 64×64 B. For `.s` (32-bit) accumulators there are **4 ZA tiles**
`za0.s..za3.s`, each a **16×16 INT32** accumulator. INT8 SMOPA fills the same `.s` tiles; each Z holds
**64 INT8 lanes** = 16 outer × 4 inner-K, so one `smopa` does a 16×16 INT32 update reducing **K=4**.

**Recommended macro-tile** (drawn from KleidiAI `mr=1·SVLs, nr=4·SVLs` + scalable-analyses' all-4-tile
microkernel):

| Param | Value (M4, SVLs=16) | Rationale |
|---|---|---|
| Output tile rows (M) | `mr = 1·SVLs = 16` | one M-strip; matches KleidiAI `m_step=1` |
| Output tile cols (N) | `nr = 4·SVLs = 64` | **4 ZA tiles** `za0..za3`, 16 cols each |
| ZA tiles used | **all 4** (`za0.s..za3.s`) | required for peak throughput — single-tile serializes on ZA write-back (scalable-analyses probe) |
| K-step per `smopa` | **4** (inner widening) | a K=32 quant block = 8 smopa iters per ZA tile |
| Z-reg staging | LHS → 1 Z (`z8`, shared across 4 tiles); RHS → 4 Z (`z4..z7`, one per ZA tile) | one `ld1h` LHS + `ld1w` RHS pairs per K-quad; **stage via Z then MOPA** (direct ZA loads ~2.6× slower, "Hello SME!") |

Inner loop per K-block (the verified SMOPA contract — `ZA[i][j] += Σ_{c=0..3} zA[4i+c]·zB[4j+c]`):
```
zero {za}                         // clear 4 tiles
for kq in 0..(blk/4):             // 8 iters for blk=32
    zA = ld LHS  (16 rows × 4 K interleaved → 64 bytes)
    zB0..zB3 = ld RHS (4 tiles × 16 cols × 4 K)
    smopa za0.s, pB,pB, zA.b, zB0.b   // 4 byte-predicated MOPAs
    smopa za1.s, ...; za2; za3
```
Keep `smstart`/`smstop` **outside** the whole K-loop (and ideally the whole GEMM); they are expensive
and zero Z/P. Pin to P-cluster via high QoS (no hard affinity on macOS).

> **CRITICAL predicate trap (verified, cost a 90351-err bug):** SMOPA predicates `pn`/`pm` must be
> **byte** predicates (`svptrue_b8()`). A `b32` predicate marks only 1 byte in 4 → silently drops
> 3 of every 4 inner-K products → 100% wrong. Use `b32` ONLY for the `svst1_hor_za32` readout. Handle
> M/N tails by **zero-padding the packed Z panels**, never by narrowing the byte predicate.

---

## 3. Re-tiling Cactus "expanded" INT8 → SMOPA + where the rescale lives

### What Cactus already has (verified against `src/matmul.cpp`)
- **Weights `expanded` (int8):** built by `tq_preexpand_weights` → `tq_interleave_4x_s8` (matmul.cpp
  :115). For each `(Nblock nb, group g)` a `gs*4`-byte int8 panel at offset `(nb*num_groups+g)*gs*4`,
  holding **4 N-rows interleaved at 32-bit granularity** (`vzip` of 4 dequant int8 rows). Layout =
  `[K][4 N-rows]@32-bit`, group-major. The codebook int8 scale `cb_scale` is **folded into the norm**:
  `n_f32[(nb*num_groups+g)*4 + ni] = norms[...]·cb_scale` (matmul.cpp:1429). So weight effective fp32
  scale = `norm·cb_scale`, already in `n_f32`.
- **Activations (int8):** Hadamard-transformed then per-group symmetric int8 quant (`tq_quantize_group_i8`,
  scale = max|x|/127) → `act_i8` + per-group fp32 `act_scales`.
- This is structurally KleidiAI `qsi8d32p` (symmetric int8 LHS) × `qsi4c32p` (RHS) **minus the int4
  stage** — Cactus pre-expands to int8, so the SMOPA path **omits `luti4`/ZT0** entirely and loads
  weights directly as int8. (Chief simplification vs KleidiAI; costs 2× weight traffic, already paid.)

### Re-tiling needed (Unverified — must validate against working-examples pack math)
Cactus interleaves exactly **4** N-rows (NEON SDOT width). SMOPA wants **nr = 4·SVLs = 64** N-cols
across 4 ZA tiles. Two re-tile transforms:

1. **Weights `[K][4 N-rows]@32-bit` → `[k-quad][nr=64 N-cols]` split into 4 ZA-tile groups of 16 cols.**
   Keep the kr=4 K-word intact (it already matches SMOPA's 4-deep group); **swap the inner 4-N-row
   grouping for a 64-N-col grouping**. Per K-quad store 64 int8 cols contiguously; each `ld1w {z2-z3}`
   feeds 2 ZA tiles. Concretely the byte pack must satisfy the verified contract:
   `bPanel[4*j + c] = B[(4g+c)*N + (col+j)]` (j = N-col 0..63, c = inner-K 0..3).
2. **Activations `[K][m-row]` → `[k-quad][mr=16 M-rows]` interleaved.** Pack `act_i8` so
   `aPanel[4*i + c] = A[(row+i)*K + 4g+c]` (i = M-row 0..15, c = inner-K). For M=1 (decode) this is a
   single broadcastable row; for M>1 interleave up to 16 rows.

Start with **on-the-fly re-tiling** in the kernel (stage into stack `aPanel[64]/bPanel[64]` exactly
like `int8_smopa_matmul.cpp`); add a dedicated `expanded_sme` pre-pack only once correctness is proven
and the re-tile shows up in the profile.

### Where `norm_f32` rescale applies (verified mapping)
**After** the integer SMOPA, **before** the fp16 store. Per output `(m-row i, n-col j)`:
```
acc_i32 = ZA[i][j]                         // raw int8·int8 dot, NO scaling in ZA
f       = (float)acc_i32 · act_scale[group] · n_f32[ncol]    // n_f32 already = norm·cb_scale
C[...]  = (__fp16) f                        // Cactus C is __fp16*
```
This is KleidiAI's `scvtf(za)·(s_lhs·s_rhs)` with `s_lhs = act_scale`, `s_rhs = norm·cb_scale` — a 1:1
map. Two ways to drain:
- **Drain to int32 in memory, rescale in NON-streaming NEON** (simplest M3): `svst1_hor_za32` the raw
  int32, then `vcvt`/`vmul` by `act_scale·n_f32` in NEON, store fp16. Keeps all float math out of the
  streaming region.
- **In-kernel drain** (KleidiAI style, fewer round-trips for M>1): `svread_hor_za32_s32_m` → SVE
  `svcvt_f32_s32_x` → `svmul`/`svmla` by broadcast fp32 scales → store. Keep Cactus's **fp32** scales
  (`ld1rw` broadcast); do NOT down-convert to fp16 — Cactus already has fp32 `n_f32`, avoid KleidiAI's
  fp16 widen and its precision loss.

For `gs > 32`: accumulate per-32-block scaled results either in a register fp32 accumulator per
(m-row, n-tile) (matches Cactus's current `running_sum` style — preferred) or in DST memory
(`fmla` into reloaded `ld1w`, KleidiAI style).

---

## 4. Implementation recipe for `src/matmul_sme2.cpp`

**Testing-first (per project rule):** every milestone gates on the existing harness oracle
`cq_reference_gemv_f32` (test_matmul.cpp:185) / a scalar triple loop, MSE ≤ 0.1, before any perf claim.
Register each variant in the harness variant registry (Part 1b).

### M0 — FP32 FMOPA smoke (ZA/streaming plumbing)
- Copy `working-examples/fp32_fmopa_matmul.cpp` structure into the TU. Kernel
  `__arm_new("za") __arm_locally_streaming`, `svcntw()` tiling, `svmopa_za32_f32_m`, `svst1_hor_za32`.
- Gate: 32×32×32 vs scalar, max_abs_err < 1e-3. Proves smstart/ZA/readout/march wiring. **Already
  proven in working-examples — port + wire into harness, don't re-derive.**

### M2 — INT8 SMOPA standalone
- Port `working-examples/int8_smopa_matmul.cpp`. **Byte predicates `svptrue_b8()` on the MOPA**, b32
  only on readout; tail = zero-pad panels. Verified contract `ZA[i][j]+=Σ_c zA[4i+c]·zB[4j+c]`.
- Gate: 32×32×32 s8×s8→s32 vs scalar int32, max_abs_err == 0. (Already proven; wire into harness.)
- Then grow to the §2 macro-tile: mr=16, nr=64 across 4 ZA tiles, K-block=32 (8 smopa/tile).

### M3 — wire SMOPA into CQ `expanded`, M=1 (decode/GEMV)
- Reuse Cactus steps (1)-(3) (Hadamard, int8 act quant, codebook int8) **in non-streaming NEON**.
- Streaming kernel does ONLY the int8 dot-accumulate over re-tiled weights (§3). M=1 ⇒ one A row,
  broadcast into the LHS lanes; only 16-of-16 M-rows are tail-padded.
- Drain int32 → NEON rescale by `act_scale·n_f32` → fp16 (the simple drain). Compare to
  `cq_reference_gemv_f32`; gate MSE ≤ 0.1. Capture GFLOPS vs NEON baseline (cq4 model GEMV = 301).

### M3b — CQ GEMM, M>1
- Interleave up to `mr=16` activation rows into `[k-quad][m-row]` panels. Loop M in strips of 16, N in
  strips of 64. Switch to in-kernel fp32 drain (`svread`+`svcvt`+`svmla`) to amortize readout across
  the larger tile. Gate vs reference for several M; compare GFLOPS vs NEON cq4 (1024³=1813, 2048³=2280).
- **Dispatch rule:** enable in `cactus_quant_matmul` only if MSE ≤ 0.1 AND beats NEON on this Mac;
  else log "perf-deferred", do not dispatch.

### M4 (stretch) — keep weights int4 + `luti4`/ZT0
- If weight memory traffic dominates, adopt KleidiAI's int4 RHS pack
  (`kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon`) + `luti4 {z4-z5}, zt0, z2[0]` widening
  (LUT `{-8..7}`), folding `cb_scale` into the fp16 RHS scale. Only after M3b ships.

---

## 5. Top 5 open risks / unknowns + fallback

1. **SMOPA re-tile byte-pack correctness at nr=64.** The 4-N-row→64-N-col transpose (§3) is the only
   genuinely new code; the verified single-tile contract must hold at full tile width.
   *Fallback:* the proven `int8_smopa_matmul.cpp` on-the-fly `aPanel/bPanel[64]` packing already passes
   exact at 16×16; build nr=64 by tiling that exact pack 4× before writing a fused packer.

2. **Perf vs NEON cq4 may not win** (NEON cq4 already 1813–2280 GFLOPS; one shared SME unit per
   cluster, no thread scaling). M=1 GEMV especially may be memory-bound, not MOPA-bound.
   *Fallback:* correctness-first; if SME loses, log "perf-deferred" and keep NEON dispatched — SME is
   not required to ship, and M>1 prefill is the likelier win than M=1 decode.

3. **In-kernel fp32 rescale inside streaming mode.** SVE `svcvt_f32_s32_x`/`svmul`/`ld1rw` are
   streaming-legal, but the exact `svread`→cvt→`svmla` sequence is Unverified on HW.
   *Fallback:* the **memory-drain** path (store raw int32 via `svst1_hor_za32`, rescale in NEON) is
   fully verified and side-steps all in-streaming float work. Use it for M3.

4. **gs>32 block accumulation & scale alignment.** Cactus `gs` ∈ [32,256]; per-32-block scale must be
   applied before summing blocks. Register-accumulate vs DST-accumulate choice unverified for SME tile.
   *Fallback:* mirror KleidiAI's per-32-block `fmla`-into-DST (proven structure) if register pressure
   on the 16×64 tile is too high; otherwise the register `running_sum` style Cactus already uses.

5. **P-core placement / E-core 4–5× slowdown.** No hard affinity on macOS; QoS is only a hint.
   *Fallback:* run the streaming kernel on a high-QoS (`USER_INTERACTIVE`) thread; accept it's
   advisory. Benchmark with the existing P-core affinity hooks; report numbers as P-cluster-targeted.

---

### Single most important thing from KleidiAI's actual source
Cactus's existing `expanded` int8 weights + per-group int8 acts are a **near-exact structural match to
KleidiAI's symmetric `qsi8d32p × qsi4c32p` SME2 mopa kernel** — same symmetric-int8, per-block-fp(16)-scale,
4-deep-K, `nr=4·SVLs / mr=1·SVLs` shape — and because Cactus **pre-expands to int8**, the SMOPA path
**drops the `luti4`/ZT0 int4→int8 widening stage entirely**. The only real porting work is the N-row
re-tile (4 → 4·SVLs) and choosing where the (already-`cb_scale`-folded) `norm·act_scale` rescale lands:
after the integer mopa, before the fp16 store. No zero-point/row-sum correction is needed (that is the
*asymmetric* qai8dxp variant; Cactus is symmetric like qsi8d32p).
