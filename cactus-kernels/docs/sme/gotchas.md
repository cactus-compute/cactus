# Gotchas & Constraints

> Hard-won traps. Read before writing any SME/SVE code.

## Streaming mode (PSTATE.SM)
- Inside an `__arm_streaming` function the vector length is the **streaming VL (SVL)**, NOT NEON's
  128-bit. NEON Advanced-SIMD intrinsics are largely **illegal/UB in streaming mode** — do not call
  `vdotq_*`, `vfmaq_*`, etc. inside a streaming function.
- Entering/leaving streaming mode (`smstart`/`smstop`) **zeroes the SVE Z/P registers** and is not
  free. Batch a whole N-panel of work per streaming call; never one column at a time.
- Keep the Hadamard transform, INT8 quantization, codebook handling, and final rescale in the
  **non-streaming** NEON caller; the streaming kernel does ONLY the MOPA accumulate.

## ZA state
- `__arm_new("za")` makes the function own ZA (auto smstart/smstop). Keep ZA lifetime within one
  function — don't pass live ZA across calls unless using `__arm_in/out/inout("za")` deliberately.
- ZA tile count is inverse to element size: 32-bit → 4 tiles (za0.s..za3.s); 8-bit → 1 tile worth of
  slices. Plan tiling around `za32` (FP32 accumulators) because `FEAT_SME_F16F16=0` here.
- macOS xnu auto-saves/restores ZA across context switches; no entitlement needed. **Do NOT use SME in
  signal handlers** (corrupts interrupted thread's ZA).

## Apple Silicon specifics
- Apple exposes **SME/SME2 but not SVE/SVE2** to userspace → the SVE2 kernel path can't be natively
  run/tested here; validate it under QEMU / on Android. SME2 is the locally-testable path.
- `-march` string is contested between sources: probe says `-march=armv9-a+sme2` compiles on Apple
  clang 17; some sources use `-march=armv8-a+sme`. RESOLUTION: pin whatever compiles AND emits the
  expected `smstart`/`mopa` asm; keep per-source in CMake. (Verify in M0.)
- E-cores are ~5× slower at MOPA than P-cores → pin SME work to P-cores (existing affinity hooks).

## Toolchain / build
- Confine SME to its own TU compiled with `+sme2`; do NOT add `+sme2` to the library-wide flag, or the
  optimizer may emit streaming/SME code into NEON paths and the base lib stops running on non-SME HW.
- Double-guard each SME TU: `#if defined(__ARM_FEATURE_SME2)` real kernel / `#else` NEON-fallback stub,
  so older toolchains still build and link.
- ACLE attribute syntax drifted: keyword form `__arm_streaming` / `__arm_new("za")` (post-2024 Q1) vs
  older `__attribute__((arm_streaming))`. Use the keyword form on clang 17.

## Numerics
- All MOPA accumulates into FP32 here (no F16F16) — this is *good* for the MSE-0.1 gate; matches the
  existing NEON CQ path which also accumulates INT8 dots in FP32. Watch FP16 input rounding only.

## NEW TRAPS — empirical verification 2026-06-09 (Apple M4 Pro, clang 17)

### Attribute POSITION matters: streaming/ZA-io attrs are TYPE attrs (go AFTER the param list)
- `__arm_streaming`, `__arm_streaming_compatible`, `__arm_in/out/inout/preserves("za")` are
  **type attributes** — they must appear in the TYPE position, AFTER the parameter list:
  `void f(args) __arm_streaming __arm_inout("za") { ... }`.
- `__arm_new("za")` and `__arm_locally_streaming` are **declaration attributes** — they go in the
  PREFIX position, BEFORE the function: `__arm_new("za") __arm_locally_streaming void f(args) {...}`.
- Putting `__arm_streaming` in the prefix position is a HARD ERROR on clang 17:
  `error: '__arm_streaming' cannot be applied to a declaration`.  `[compiled-fail then fixed]`
- A self-contained leaf kernel callable from NEON code uses BOTH:
  `__arm_new("za") __arm_locally_streaming void kernel(args) { ... }`  (no type attr needed because
  locally-streaming + new-za auto-bracket smstart/smstop). Both matmul examples use exactly this.

### SMOPA (and any 8-bit MOPA) needs BYTE predicates — b32 silently zeros 3/4 of the K reduction
- `svmopa_za32_s8_m(tile, pn, pm, zA, zB)` operands are BYTES. The governing predicates `pn`/`pm`
  MUST be **byte** predicates (`svptrue_b8()` / `svwhilelt_b8`). They predicate the BYTE lanes, and
  the 4-way inner-K reduction lives inside the 4 bytes of each 32-bit group.
- Passing a **b32** predicate (the kind you use for the f32 FMOPA tile or for the `svst1_hor_za32`
  readout) marks only 1 byte in 4 as active → the SMOPA drops 3 of every 4 inner-K products →
  output is 100% wrong (NOT a tail/edge bug; ALL cells wrong). This exact mistake made the first
  INT8 matmul give max_abs_err=90351 on all 1024 cells; switching pn/pm to `svptrue_b8()` made it
  exact (err=0).  `[ran-correct after fix]`
- Mnemonic: predicate element width == OPERAND element width (b8 for s8/u8 MOPA, b16 for f16/bf16,
  b32 for f32). The 32-bit predicate is ONLY for the za32 readout store, never for the s8 MOPA.
- Do M/N tail handling for INT8 by ZERO-PADDING the packed Z panels, not by narrowing the byte
  predicate (an off-tile byte predicate would also kill inner-K lanes).

### SMOPA widening contract (so the byte-pack is right) — verified by single-hot probe
- `ZA32[i][j] += sum_{c=0..3} zA[4*i + c] * zB[4*j + c]`. a[ai]=1 & b[bj]=1 lights ZA[ai/4][bj/4]
  ONLY when ai%4==bj%4. So 4 contiguous K for one output index live in ONE 32-bit lane group:
  `aPanel[4*i+c]=A[(row+i)*K + 4g+c]`, `bPanel[4*j+c]=B[(4g+c)*N + (col+j)]`.

### THE central performance model: Apple SME = per-cluster IN-ORDER command queue (AMX heritage)
- All streaming ops (loads, luti4, dots, ZA ops) from all cores of a cluster funnel into ONE shared
  in-order queue per cluster (M4 Pro: 2 P-clusters + 1 E-cluster ≈ 2.2 effective units).
- **ZA→Z→core reads (`svread_*`) are queue DRAINS + core round-trips**: with per-group reads the
  LUTI4 GEMV pinned at ~137 GMAC/s aggregate vs 406 GMAC/s for the same dot stream without reads —
  and NO amount of chain interleaving recovers it (in-order queue; ILP can't cross a read barrier).
- **ZA→MEMORY stores (`svst1_hor_za32`) are fire-and-forget** — replacing per-group reads+in-stream
  fp rescale with partial-int32 stores + NEON rescale after `smstop` recovered the throughput.
  This matches why the M3b GEMM (which always stored partials) beat NEON while GEMV lagged.
- `svzero_za` is also ordered behind everything; amortize (once per 4 groups via slice rotation).
- Z-register svdot (no ZA) is NOT a workaround: z-dot issue rate is the bottleneck then (0.55x).
- Per-worker in-situ rate ~13-16 GMAC/s (DRAM-latency stalls in-order); microbench L1 rate 95-190.
  More workers per cluster ≈ saturate the unit; k_sme≈5-6 of 14 pool workers is the sweet spot.

### vg1 ZA-array vector -> svst1 slice mapping (probed)
- vg1 za32 vector v lives at `svst1_hor_za32(tile=0, slice=(v&3)*4 + (v>>2))` (SVL=512). All 16
  vg1 vectors map into "tile 0"'s 16 slices in interleaved order — probe before assuming geometry.

### LUTI4/ZT0 (probed, 512 hot-nibble cases, -O0+-O1)
- `svluti4_lane_zt_s8_x2(0, zn, imm)`: IDENTITY mapping — nibble j of zn -> output byte j
  (out[0]=nibbles 0..63, out[1]=64..127), low nibble first (= Cactus packing). imm is IRRELEVANT
  for the .b x2 form. Table entry i = ZT0 byte 4*i (low byte of word i). Function needs
  `__arm_new("zt0")`; load via `svldr_zt(0, ptr64B)`.
- LUTI4 in-engine expansion halves the weight stream (packed nibbles) at no correctness cost:
  table = cb_i8 reproduces the NEON vqtbl expansion exactly.

### clang 17 -O2/-O3 MIS-COMPILES vg1x4 ZA-array dots (runtime SIGILL)
- `svdot_lane_za32_s8_vg1x4` / `svdot_za32_s8_vg1x4` loops compile at -O2 but SIGILL at runtime;
  correct + fast at -O0/-O1. Quarantine all vg1x4 kernels in a dedicated -O1 TU (matmul_sme2_gemv.cpp).
- `svreadz_*` (read-and-zero) needs SME2p1 — absent on M4.
- `svld1_s8_x4` (svcount multi-vector load) works at -O1; also implicated in -O2 SIGILLs.

### Thermal asymmetry: NEON fades, SME holds
- NEON-pure GEMV swings 277-365 GFLOPS with chip temperature (clock fade under sustained load);
  the SME/hybrid path holds ~310-315 steady. Cold-start benchmarks flatter NEON; sustained decode
  (the real LLM workload) favors the hybrid. Always compare via in-run alternation (best-of-5x30).

### -march=armv9-a+sme2 is an ASM-ONLY string on Apple Silicon (re-confirmed by RUN)
- Re-verified: `armv9-a+sme2` compiles and emits byte-identical fmopa/smopa/bfmopa/zero/st1w asm to
  the runnable strings, but the BINARY SIGILLs at runtime (exit 132). All four candidate -march
  strings produce the same `-S` output, so an asm-check CANNOT distinguish runnable from trapping —
  you MUST run. Runnable: `armv8.2-a+fp16+simd+dotprod+i8mm+sme2`, `armv8-a+sme`, `-mcpu=apple-m4`.

## parallel_for static splitter dumps the remainder on the LAST worker
`CactusThreading::parallel_for` computes `work_per_thread = total/num_threads` (FLOOR) and gives
thread N-1 everything left: 64 items / 14 workers = 13x4 + 1x12 (3x straggler). Invisible for
fine-grained items (1000s), catastrophic for coarse heavy items (attention q-tiles: measured ~60%
pool idle, 1.39ms -> 0.81ms after fix). Fix pattern: dispatch `n_slots = min(total, num_workers())`
pseudo-items with ParallelConfig{1,1} and walk `for (i = slot; i < total; i += n_slots)`.

## Per-quant-group ZA readout makes SMOPA kernels store-dominated
If output scales differ per K-segment (e.g. per-32-dim quant groups), reading ZA out per group
costs 2 stores per SMOPA (16 rows x 4 tiles per group vs 32 SMOPAs) and multiplies partial-buffer
round-trip traffic by n_groups. Profile signature: issuing thread stalls INSIDE the leaf (430/754
busy samples). Fix: FLATTEN scales so ZA accumulates the whole K extent — requantize the
per-group operand to one flat scale (ratios folded into int8, <=1 bit loss), readout once.
Attention QK: 8x less readout, leaf 430 -> 14 samples, same accuracy within 0.1%.

## USMOPA u8 x s8 (svusmopa_za32_u8_m) is exact on M4 and free accuracy for non-negative operands
Softmax probabilities (and any >=0 operand) waste the sign bit under s8 SMOPA. USMOPA
(unsigned zn x signed zm) doubles resolution at identical cost. Probe-verified max_err=0 with
full 0..255 range (working-examples/usmopa_probe.cpp).

## Portability: SVL-512 pinning + Apple-tuned policies (Samsung S26-class devices)
ALL shipped SME layouts (esme cache [16ch][4K] panels, 64-ch super-blocks, attention 16x64 tiles,
every +64/+128/+192 offset and svptrue_b8 assumption) are pinned to SVL = 64 bytes. Non-Apple
SME2 implementations may use other vector lengths -> would be SILENTLY WRONG, not slow.
GATED 2026-06-09: cpu_has_sme2() now also requires cactus_sme2_svl_bytes() == 64 (streaming
svcntb probe in matmul_sme2.cpp); other SVLs fall back to NEON for correctness. SVL-parametric
variants are future work.
Separately, the PERF model (per-cluster in-order queue, k_sme=4/6 splits, 3MB/6MB GEMV policy,
cluster-spread at nt>=6) is measured on Apple M4 Pro only — on Qualcomm/Samsung SME2 these are
heuristics to re-tune, not laws. Feature detection (Apple sysctl / Android HWCAP2 1UL<<37) and
the CACTUS_HAS_SME2 compile-check + stubs already degrade gracefully on non-SME toolchains.

## pool.wait_all() cv-sleep + idle main = 20-30% tax on sub-100us parallel kernels
Every enqueue_n_threads + wait_all pays: N task allocations under one mutex, a cv sleep for main,
and a cv wake by the last worker (~5-15us wall), while the main thread contributes nothing. For
GEMV-sized calls (~30-90us) this is 20-30% of the call. Fix pattern (gemv_fused/orth drivers):
enqueue nt-1 workers, MAIN runs worker 0 inline (also the earliest SME issuer — zero wake
latency), spin-join an own atomic done counter. Kernel-level: o_proj 149->194 GF, gate_up
588->707 GF, q_proj 145->192 GF — same integer compute, 21/21 green.
