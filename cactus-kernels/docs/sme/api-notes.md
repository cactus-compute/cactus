# API Notes — verified facts

> Append-only-ish; correct entries in place if proven wrong (note the correction). Every claim should
> carry evidence: `[compiled]`, `[asm-checked]`, `[ran-correct]`, or `[source: <url/file>]`.

## ⚑ VERIFIED `-march` for Apple M4 (CRITICAL — corrects earlier guess)
- **USE `-march=armv8.2-a+fp16+simd+dotprod+i8mm+sme2`** (or `armv8-a+sme`). `[ran-correct]` M0 PASS, max_err=0.
- **DO NOT use `-march=armv9-a+sme2`** — it COMPILES but the binary **SIGILLs at runtime (exit 132)** on M4.
  Root cause: `armv9-a` implies the full non-streaming SVE2 ISA, which Apple Silicon does NOT implement
  (Apple has only SME's streaming-SVE subset). clang then emits non-streaming SVE2 instructions that trap.
  `armv8(.2)-a+sme2` advertises SME2 WITHOUT the bad SVE2 baseline → only streaming-legal code is emitted.
- This means the earlier design-phase claim ("armv9-a+sme2 verified") was an ASM-only check, not a run.
  Lesson: always RUN, never trust asm-compiles-clean for SME on Apple.

## VERIFIED signatures from clang 17 arm_sme.h (resource-dir header, authoritative on this box)
`[source: $(clang -print-resource-dir)/include/arm_sme.h]`
- `void svmopa_za32_f32_m(uint64_t tile, svbool_t pn, svbool_t pm, svfloat32_t zn, svfloat32_t zm);`
- `void svmopa_za32_f16_m(uint64_t, svbool_t, svbool_t, svfloat16_t, svfloat16_t);`  (FP16→FP32 widening — EXISTS here)
- `void svmopa_za32_bf16_m(uint64_t, svbool_t, svbool_t, svbfloat16_t, svbfloat16_t);`
- `void svmopa_za32_s8_m(uint64_t, svbool_t, svbool_t, svint8_t, svint8_t);`  (INT8 4-way → za32)
- `void svmopa_za32_u8_m(...)`, `svmopa_za32_s16_m/u16_m(...)` also present.
- **ZA store (note arg order!):** `void svst1_hor_za32(uint64_t tile, uint32_t slice, svbool_t pg, void* ptr);`
  Also `_ver_` and za8/16/64/128 variants. `slice` = horizontal slice index 0..SVLs-1 (=0..15) within `tile` (0..3).
- `void svzero_za(void);`  `void svzero_mask_za(uint64_t mask);`
- `svuint32_t svread_hor_za32_u32_m(svuint32_t inactive, svbool_t pg, uint64_t tile, uint32_t slice);`
- `void svaddva_za32_s32_m(uint64_t tile, svbool_t pn, svbool_t pm, svint32_t);` (vertical add-accumulate — useful for GEMV reductions)
- `svcntw()` (arm_sve.h) returns streaming-VL word count inside a streaming fn (=16 on M4). Predicates `svptrue_b32()`.
- **Leaf-kernel attribute combo (callable from normal NEON code):** `__arm_locally_streaming __arm_new("za")`
  — function self-transitions to streaming mode + owns private ZA (auto smstart/smstop). Use this for the
  Cactus SME leaf kernel so the non-streaming caller needs no streaming context.

## SME ACLE essentials (Apple clang 17, `-march=armv9-a+sme2`)
- Header: `#include <arm_sme.h>` (pulls in `arm_sve.h`). Guard with `#if defined(__ARM_FEATURE_SME2)`.
  `[compiled]` on Apple clang 17 with `-march=armv9-a+sme2`.
- A function that owns ZA: mark `__arm_new("za")` (auto `smstart za` on entry / `smstop za` on exit)
  and `__arm_streaming` (runs in streaming-SVE mode, PSTATE.SM=1). `[asm-checked]` emits `smstart za`.
- Streaming vector length: `svcntw()` = words (FP32 lanes) in streaming mode = **16 on M4** (SVL 64B).
  Tiling MUST be SVL-parametric (use `svcntw()/svcntb()/svcnth()`), never hard-code 16B like NEON.
- Outer-product accumulate intrinsics (accumulate into FP32 `za32`, tile index 0..3):
  - FP32: `svmopa_za32_f32_m(uint64_t tile, svbool_t pn, svbool_t pm, svfloat32_t zn, svfloat32_t zm)`
    → `fmopa za<tile>.s, p/m, p/m, zn.s, zm.s`. `[asm-checked]`
  - BF16 (widening 2-way): `svmopa_za32_bf16_m(...)` → `bfmopa`. (`FEAT_BF16=1`.)
  - INT8 (widening 4-way): `svmopa_za32_s8_m(uint64_t tile, pn, pm, svint8_t, svint8_t)` → `smopa
    za<tile>.s, p/m, p/m, zn.b, zm.b`. `[asm-checked]` Accumulates 4 INT8 products per 32-bit cell.
  - FP16→FP32 (widening 2-way): `svmopa_za32_f16_m(...)` → `fmopa` (since `FEAT_SME_F16F16=0`, must
    use the za32 widening form, not za16). TO CONFIRM during M1.
- ZA read-out: `svread_hor_za32_*` / `svst1_hor_za32(pg, ptr, tile, slice)` writes a horizontal slice
  of a za32 tile to memory. Also `svaddha_za32` (add horizontally-accumulated). Verify exact names M0.
- Load helpers in streaming mode are the SVE `svld1_f32(svptrue_b32(), ptr)` etc. (predicated).
- Data path tip `[source: Hello SME! arxiv 2409.18779]`: load to Z regs via SVE then feed MOPA;
  direct memory→ZA loads (`ldr za`) are ~2.6× slower than Z-reg staging.

## INT8 SMOPA tiling shape (the M2/M3 target)
- SMOPA computes `ZA[i][j] += sum_{c=0..3} A[i][4k+c] * B[j][4k+c]` over a streaming tile: zn holds an
  SVL-byte column-ish vector (i index along lanes, 4 contiguous K per 32-bit group), zm likewise (j).
- For an `M×N` tile of size `SVL/4 (=16) × SVL/4` accumulating over K in steps of 4. Confirm exact
  operand semantics against scalable-analyses/sme INT8 kernel + KleidiAI `..._sme2_mopa` before coding.

## Cactus kernel layout (from matmul.cpp / test_matmul.cpp, read 2026-06-09)
- Public dispatch: `cactus_quant_matmul(const CactusQuantMatrix* W, const __fp16* A, uint32_t M,
  __fp16* C)` at `src/matmul.cpp:1479`. This is what the test harness calls; SME variants mirror it.
- `CactusQuantMatrix` (cactus_kernels.h:206): bits, K, N, group_size, num_groups, flags, codebook(fp16
  2^bits), input_scale/recip(fp16,K), norms(fp16, N*num_groups), packed_indices(u8), left/right_signs
  (i8, group_size), permutation(u32, group_size), rotation, **expanded(int8)**, **norm_f32(float)**.
- CQ pipeline per group: (1) activation transform `z = x/input_scale * left_sign`, FWHT, `* right_sign`
  (`cactus_quant_transform_hadamard_activations`, matmul.cpp:282); (2) INT8-quantize transformed acts
  (`tq_quantize_group_i8`, 1247); (3) codebook→INT8 (`tq_quantize_codebook_i8`, 1233); (4) dot via
  SDOT (`cactus_quant_sdot_gemv_int8`, 651) over `expanded` weights; (5) rescale by `norm_f32` (which
  already folds the codebook INT8 scale `cb_sc`). Reference oracle: `cq_reference_gemv_f32`
  (test_matmul.cpp:185).
- `expanded` weight layout (`tq_preexpand_weights`, 1440; mirrored in test `preexpand()`): grouped by
  N-blocks of 4 rows; for each (nb, g) a `group_size*4` int8 panel, 4 N-rows interleaved at 32-bit
  granularity (built via `vzip` of 4 dequantized int8 rows). `norm_f32` is `N_blocks*num_groups*4`,
  `nd[ni] = norms[(n_start+ni)*num_groups+g] * cb_sc`. **This layout is tuned for `vdotq_laneq_s32`,
  NOT for MOPA** — M3 must re-tile (on-the-fly first, dedicated `expanded_sme` later).
- **Strategy:** keep steps (1)-(3) + rescale in NON-streaming NEON code; the `__arm_streaming` SME
  kernel does ONLY the INT8 dot-accumulate (step 4) over re-tiled weights. No NEON intrinsics inside
  streaming functions.

## Build wiring
- New TUs `src/matmul_sme2.cpp` (`-march=armv8.2-a+fp16+simd+dotprod+i8mm+sme2` — NOT armv9-a+sme2,
  see SIGILL note above) / `src/matmul_sve2.cpp` (`+sve2`) via
  `set_source_files_properties(... COMPILE_OPTIONS ...)` in `cactus-kernels/CMakeLists.txt`.
  Library-wide flag (line 34) stays `-march=armv8.2-a+fp16+simd+dotprod+i8mm`.
- Runtime detect: add `cpu_has_sve2/sme/sme2` beside `cpu_has_i8mm` (threading.h:50). Apple keys
  `hw.optional.arm.FEAT_SME` / `FEAT_SME2`; Android HWCAP2 `SME=1<<23`, `SME2=1<<37`, `SVE2=1<<1`.

## ⚑ EMPIRICAL VERIFICATION PASS 2026-06-09 (probes in /tmp/sme-probe, working-examples/ updated)
Probed on Apple M4 Pro, Apple clang 17.0.0. SVL=512b. FEAT_SME_F16F16=0.

### -march that works
- `-march=armv8.2-a+fp16+simd+dotprod+i8mm+sme2` `[ran-correct]` — compiles, runs, both matmuls PASS.
- `-march=armv8-a+sme` `[ran-correct]` — compiles + runs (svcntw probe PASS, exit 0).
- `-mcpu=apple-m4` `[ran-correct]` — compiles + runs (svcntw probe PASS, exit 0).
- `-march=armv9-a+sme2` `[asm-checked + RAN]` — compiles, emits IDENTICAL SME asm, but the BINARY
  SIGILLs (exit 132). CONFIRMED by running probe p6 under all four strings. Use it for `-S` asm
  inspection only; never to produce a runnable binary on M4.
- For `-S` ASM inspection ALL FOUR strings emit the same fmopa/smopa/bfmopa/zero/st1w. So asm-check
  is march-insensitive; only RUN distinguishes them.

### Per-probe results (all asm-checked; p6 ran)
- P1 `svmopa_za32_f32_m(0,pn,pm,zn,zm)` `[asm-checked]` -> `fmopa za0.s, p0/m, p1/m, z0.s, z1.s`. ✓
- P2 `svmopa_za32_s8_m(0,pn,pm,zn,zm)` `[asm-checked]` -> `smopa za0.s, p0/m, p1/m, z0.b, z1.b`. ✓
  (4-way widening: .b operands accumulate into .s tile.)
- P3 `svzero_za()` AND `svzero_mask_za(255)` `[asm-checked]` -> both emit `zero {za}`. ✓
- P4 readout, three forms all compile+emit a ZA store/read `[asm-checked]`:
    * `svst1_hor_za32(0, slice, pg, ptr)` -> `st1w {za0h.s[wN, 0]}, p0, [ptr]`  (USE THIS for matmul out)
    * `svstr_za(slice, ptr)`              -> `str za[wN, 0], [ptr]`             (whole-ZA spill)
    * `svread_hor_za32_f32_m(inactive,pg,0,slice)` -> `mov z0.s, p0/m, za0h.s[wN, 0]` (ZA->Z, then rescale+st)
- P5 `svmopa_za32_f16_m(0,pn,pm,zn,zm)` `[asm-checked]` EXISTS + compiles here ->
    `fmopa za0.s, p0/m, p1/m, z0.h, z1.h` (FP16->FP32 widening; uses za32 widening form since
    FEAT_SME_F16F16=0, no za16 f16 path). ✓
- P6 `svcntw()` `[ran-correct]` inside __arm_locally_streaming returns **16** on M4.
    Also ran: `svcntb()=64`, `svcntsw()=16`, `svcntsb()=64`. SVL=512b confirmed by run, not docs.
- P7 `svmopa_za32_bf16_m(0,pn,pm,zn,zm)` `[asm-checked]` -> `bfmopa za0.s, p0/m, p1/m, z0.h, z1.h`. ✓

### Full runnable matmuls (saved to working-examples/, both PASS exact)
- `working-examples/fp32_fmopa_matmul.cpp` `[ran-correct]` 32x32x32, max_abs_err=0.000e+00.
- `working-examples/int8_smopa_matmul.cpp` `[ran-correct]` 32x32x32, max_abs_err=0 (all 1024 cells).

### SMOPA widening contract — VERIFIED BY SINGLE-HOT PROBE (not docs)
- `ZA32[i][j] += sum_{c=0..3} zA[4*i + c] * zB[4*j + c]`. Proven: a[ai]=1,b[bj]=1 lights
  `ZA[ai/4][bj/4]` ONLY when `ai%4 == bj%4` (matching inner-K c); all-ones -> cell value 4.
  Pack: `aPanel[4*i+c]=A[(row+i)*K + 4g+c]`, `bPanel[4*j+c]=B[(4g+c)*N + (col+j)]`.

### CORRECTION to research/acle-sme-intrinsics.md §0 item 2 (re: za32 s16/u16 MOPA)
- research doc claims "There is NO svmopa_za32_s16_m / _u16_m". **WRONG on this box.** Apple clang 17
  `arm_sme.h` DOES declare `svmopa_za32_s16_m` and `svmopa_za32_u16_m`. `[asm-checked]`
  `svmopa_za32_s16_m(0,pn,pm, svint16, svint16)` compiles and emits `smopa za0.s, ..., z0.h, z1.h`
  (a 2-way-widening S16×S16 -> S32-in-za32; an SME2.1 form Apple exposes). So both exist:
  za32-from-s16 (2-way) AND za64-from-s16 (`svmopa_za64_s16_m`, needs FEAT_SME_I16I64, 4-way).
  Header confirmed to define: f32, f16, bf16, s8, u8, s16, u16 -> za32.
