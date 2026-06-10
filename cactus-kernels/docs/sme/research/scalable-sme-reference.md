# scalable-analyses/sme — Known-Good SME Matmul Reference (M0 FP32 FMOPA, M2 INT8 SMOPA)

Source: https://github.com/scalable-analyses/sme  ("Hello SME!", SC24 workshop, DOI 10.1109/SCW63240.2024.00185; arXiv 2409.18779).
Local clone: `/tmp/sme-refs/scalable-sme` @ commit `c9de42a561c9c3bf9e926748fb5113a075d1f096` (2025-03-31).
Target M4 device: 2024 11-inch iPad Pro, Apple M4 SoC — first publicly available SME silicon. **SVL = 512 bits** (svcntw = 16 FP32 lanes, svcntb = 64 INT8 lanes).

These hand-written `.s` kernels are validated against Apple Accelerate `cblas_sgemm` in `gemm.cpp` (max abs/rel error printed; README shows 0.0 error vs reference for the libxsmm path). They are the templates we adapt in
`cactus-kernels/src/matmul_sme2.cpp`.

---

## 0. The ZA / SVL geometry on M4 (the facts everything below depends on)

- **SVL = 512 bits.** A ZA *tile group* (`ZA`) is `SVL x SVL` bits = `64 x 64` bytes.
- For **`.s` (FP32) element size**, there are **4 ZA tiles** `za0.s..za3.s`, each holding a **16 x 16 FP32** accumulator (16 rows x 16 cols x 4 bytes; 16*4 = 64 bytes per row = SVL).
- `fmopa za<t>.s, p0/m, p1/m, zN.s, zM.s` computes a **16x16 rank-1 (outer-product) update**: `ZA[i][j] += zN[i] * zM[j]` over the 16 active FP32 lanes. (Note source convention in these kernels: written `fmopa za, pred, pred, zB, zA` so the first Z is the column/B operand and the second Z is the row/A operand — see the `// c  b  a` comments in the loops.)
- **INT8 SMOPA** (`smopa za<t>.s, p0/m, p1/m, zN.b, zM.b`) accumulates into the **same `.s` (32-bit int) ZA tiles**, but each Z holds **64 INT8 lanes** and the instruction does a **K=4 inner reduction**: it computes a 16x16 INT32 update where each of the 64 byte lanes is interpreted as a [16 outer] x [4 inner-K] layout, i.e. one `smopa` does the work of 4 FP32 `fmopa`s worth of K. So INT8 has **4 ZA tiles of 16x16 INT32**, each fed by 64-byte Z registers, K-step = 4 per instruction.
- ZA read-out uses either `str za[wIdx, #0], [ptr]` (store one ZA row-slice to memory) or `mov { z.. }, za<t>h.s[wIdx, 0:3]` (move 4 horizontal ZA slices into Z regs, then `st1w`). ZA load-in uses `ldr za[..]` or `mov za<t>h.s[..], { z.. }` after a Z-register `ld1w`.

---

## 1. M0 reference — simplest standalone FP32 FMOPA (single 16x16 outer product + ZA→mem)

This is the minimal, self-contained correctness kernel. It is exercised by `examples.c::showcase_fmopa_fp32_fp32_fp32()` with `a[i]=b[i]=i+1`, producing the rank-1 outer product `C[i][j] = (i+1)*(j+1)`.

File: `MicrobenchmarkApp/kernels.s` lines 957–983.

```asm
    .global _example_sme_fmopa_fp32_fp32_fp32
    .align 4
_example_sme_fmopa_fp32_fp32_fp32:
    smstart                                  // enter streaming-SVE + enable ZA

    ptrue p0.b                               // all-true governing predicate

    ldr z0, [x0]                             // stage A column (16 FP32) into Z0
    ldr z1, [x1]                             // stage B row    (16 FP32) into Z1

    fmopa za0.s, p0/m, p0/m, z0.s, z1.s      // ZA0[i][j] += z0[i]*z1[j]  (16x16)

    mov w12, #0
    mov x3, #16                              // 16 ZA row-slices to store

loop_example_sme_fmopa_fp32_fp32_fp32:
    str za[w12, #0], [x2]                    // store ZA0 slice w12 -> C row
    add w12, w12, #4                         // (slice index stride: tile in low 2 bits)
    add x2, x2,   #16*4                       // advance C by one 16-FP32 row
    sub x3, x3,   #1
    cbnz x3, loop_example_sme_fmopa_fp32_fp32_fp32

    smstop                                   // exit streaming, disable ZA
    ret
```

C ABI (from `examples.c`): `void example_sme_fmopa_fp32_fp32_fp32(float* a, float* b, float* c)` — `a`,`b` are 16 floats, `c` is 16*16 floats (row-major, leading dim 16). **This is our M0 unit-test oracle.**

Key M0 takeaways for `matmul_sme2.cpp`:
1. `smstart` / `smstop` bracket all ZA/streaming code; do PCS save/restore of `d8–d15` (and `x19–x30` if clobbered) — see the full GEMM kernels.
2. Stage operands into **Z registers first** (`ldr z` / `ld1w`) then `fmopa`. Do **not** try to feed memory directly.
3. ZA read-out via `str za[wN, #0]` walks the 16 row-slices; `wN += 4` per row because the low 2 bits of the slice index select the ZA tile (here tile 0).

---

## 2. M0 reference — full FP32 GEMM with K-loop, 4 ZA tiles, Z-staging (32x32x32)

This is the realistic M0 template: `C += A * B^T`, MxNxK = 32x32x32, using **all 4 ZA tiles** as a 2x2 grid of 16x16 accumulators (so 32x32 output). It loads C into ZA, runs the K-loop of `fmopa`, then reads ZA back to C.

File: `MicrobenchmarkApp/kernels_gemm.s` lines 113–391 (`_gemm_micro_32_32_32`). The load-C and store-C phases are long (ZA<->Z<->mem via `ld1w/st1w` + `mov za..h`); the **load of the actual A/B tiles and the FMOPA accumulation** is the part to adapt:

```asm
    // ---- prologue: smstart, ptrue p0.b/p1.b/pn8.b, then load C(32x32) into za0..za3
    //      via ld1w {z,z},pn8/z + `mov zaNh.s[w,0:3],{z..}` (lines 132-243) ----

    mov x6, #32                              // K loop count (K=32)
loop_32_32_k:
    sub x6, x6, #1

    // load A: two 16-FP32 columns of A (32 rows) -> z0, z1
    ldr z0, [x0]
    add x0, x0, #64                          // +16 floats
    ldr z1, [x0]
    add x0, x0, #64

    // load B: two 16-FP32 rows of B^T (32 cols) -> z2, z3
    ldr z2, [x1]
    add x1, x1, #64
    ldr z3, [x1]
    add x1, x1, #64

    //       c(ZA)            b      a      -> 2x2 grid of 16x16 outer products
    fmopa za0.s, p0/m, p1/m, z2.s, z0.s      // C[0:16 ,0:16]
    fmopa za1.s, p0/m, p1/m, z2.s, z1.s      // C[16:32,0:16]
    fmopa za2.s, p0/m, p1/m, z3.s, z0.s      // C[0:16 ,16:32]
    fmopa za3.s, p0/m, p1/m, z3.s, z1.s      // C[16:32,16:32]

    cbnz x6, loop_32_32_k

    // ---- epilogue: read za0..za3 back to C via `mov {z..},zaNh.s[w,0:3]` + st1w
    //      (lines 271-381) ----
    smstop
    ret
```

C ABI: `void gemm_micro_32_32_32(float const* a, float const* b, float* c)` (`kernels_gemm.h`). A is 32x32 col-major-ish packed, B is the B^T operand, C is 32x32. **Validated against `cblas_sgemm(..., CblasTrans, ...)` in `gemm.cpp` (max error printed).**

There is also `_gemm_micro_64_16_2` (M=64,N=16,K=2, lines 1–110) which is the **cleanest** illustration of "load A column into z0..z3, load B row into z30, fire 4 fmopa into za0..za3" with a simple `str za[w,#0]` read-out loop — good as a second M0 unit.

---

## 3. M2 reference — INT8 SMOPA

The repo contains **no full INT8 GEMM kernel**, only a peak-throughput probe (no memory I/O, all 4 ZA tiles, reused Z regs). It is still the authoritative reference for *which instruction form + ZA tiling INT8 uses on M4*.

File: `MicrobenchmarkApp/kernels.s` lines 1630–1696 (`_peak_sme_smopa_i8_i8_i32`). One loop body issues 32 `smopa` across the 4 ZA tiles:

```asm
    .global _peak_sme_smopa_i8_i8_i32
    .align 4
_peak_sme_smopa_i8_i8_i32:
    // PCS save d8-d15 ...
    smstart
    ptrue p0.b
    ptrue p1.b
loop_peak_sme_smopa_i8_i8_i32:
    sub x0, x0, #1
    // 64-INT8-lane Z operands; each smopa = 16x16 INT32 update w/ K=4 inner reduction
    smopa za0.s, p0/m, p1/m, z0.b, z1.b
    smopa za1.s, p0/m, p1/m, z2.b, z3.b
    smopa za2.s, p0/m, p1/m, z4.b, z5.b
    smopa za3.s, p0/m, p1/m, z6.b, z7.b
    smopa za0.s, p0/m, p1/m, z8.b, z9.b
    smopa za1.s, p0/m, p1/m, z10.b, z11.b
    smopa za2.s, p0/m, p1/m, z12.b, z13.b
    smopa za3.s, p0/m, p1/m, z14.b, z15.b
    smopa za0.s, p0/m, p1/m, z16.b, z17.b
    smopa za1.s, p0/m, p1/m, z18.b, z19.b
    smopa za2.s, p0/m, p1/m, z20.b, z21.b
    smopa za3.s, p0/m, p1/m, z22.b, z23.b
    smopa za0.s, p0/m, p1/m, z24.b, z25.b
    smopa za1.s, p0/m, p1/m, z26.b, z27.b
    smopa za2.s, p0/m, p1/m, z28.b, z29.b
    smopa za3.s, p0/m, p1/m, z30.b, z31.b
    // (16 more smopa, same pattern, to fill the pipeline) ...
    cbnz x0, loop_peak_sme_smopa_i8_i8_i32
    smstop
    // PCS restore ...
    mov x0, 32*2048                          // 32 smopa * (16*16*4 MACs) per iter = FLOP/iter count
    ret
```

To turn this into a real M2 GEMM we adapt the **M0 32x32x32 structure** and substitute:
- `ldr z` → `ld1b`/`ldr z` loading **64 INT8 lanes** per Z (K packed by 4 in the byte layout),
- `fmopa za.s,...,z.s,z.s` → `smopa za.s,...,z.b,z.b`,
- **K-step in the loop becomes 4** (one `smopa` consumes 4 K elements per output), so a K=32 reduction is 8 smopa iterations instead of 32,
- ZA read-out stores **INT32** (still `.s` tiles, `str za[w,#0]` / `st1w`), to be requantized downstream.

The FLOP-count constant `mov x0, 32*2048` encodes `32 smopa * 2048` where `2048 = 16*16*4*2` (16x16 outputs, K=4 inner, MAC=2 ops) → confirms the **K=4 inner-reduction** semantics of INT8 SMOPA on M4.

INT16 variant `_peak_sme_smopa_i16_i16_i32` (lines 1698+) uses `.h` operands and `mov x0, 32*1024` (K=2 inner), for reference.

---

## 4. SVL=512b (M4) tiling choice + the Z-staging-vs-direct-ZA-load perf insight

### Tiling actually used (from the real GEMM kernels)
- **Microkernel tile (register-blocked):** **M=32, N=32** output = a **2x2 grid of 16x16 ZA tiles** = **all 4 `.s` ZA tiles** (`za0..za3`) used at once. This is the unit in `_gemm_micro_32_32_32`, `_gemm_micro_32_no_trans`, and the inner block of `_gemm_128_128_128`.
- **K-step:** **2 per loop iteration** for FP32 (two `ldr z` per operand, K reduced by 2 → in `gemm_micro_31/32` the loop runs `K` times each doing one fmopa per tile; in `gemm_128_128_128` and `gemm_micro_32_no_trans` it's `ld1w {z,z}` pairs, i.e. K consumed in steps via the 4-fmopa block). For INT8 SMOPA the effective K-step is **4** (inner reduction).
- **ZA tiles at once:** **4** (the full ZA register on M4). All four are kept live across the entire K-loop; C is loaded into ZA before the loop and written out after.
- **Outer blocking (`_gemm_128_128_128`, lines 1304–1608):** 128x128x128 is done as a **4x4 grid of 32x32 microkernels** (`loop_3x128_n` x `loop_3x128_m`, each 4 iters) with a **K-loop of 128** inside, reusing the same 4-ZA-tile microkernel. Address strides: C tile-1 offset `#8192` (= 2048 floats = 32 rows * 64 floats... the 2nd ZA tile column), n-step `#16384`. This is the template for tiling a large GEMM.

### Z-register-staging vs direct-ZA-load — the documented perf insight
The microbenchmarks in `kernels.s` are explicitly built to expose this. The relevant probe pairs:

1. **`fmopa` with ZA-tile reuse vs ZA-tile rotation:**
   - `_peak_sme_fmopa_1_fp32_fp32_fp32` (lines 550–615): **every** fmopa targets `za0` (single accumulator) → serializes on the ZA write-back, lower throughput.
   - `_peak_sme_fmopa_2_fp32` / `_peak_sme_fmopa_4_fp32` (lines 618–751): round-robin across **2 / 4 ZA tiles** → hides ZA accumulate latency. **Using all 4 ZA tiles is required to reach peak FMOPA throughput** (~1.7–1.8 TFLOPS FP32 per the README libxsmm numbers).
   - `_peak_sme_fmopa_4_reorder_fp32` (lines 754–819): same 4 tiles but grouped (4 consecutive fmopa to za0, then za1...) — a scheduling sensitivity probe.

2. **`smstart`/`smstop` cost:** `_peak_sme_fmopa_smstart_smstop_{8,16,32,...}` (lines 1117+) measure amortizing the streaming-mode entry/exit — **keep `smstart`/`smstop` outside the K-loop and ideally outside the whole GEMM**, because the enable/disable is expensive relative to a single fmopa block.

3. **Direct-ZA-load (`ldr za` / `str za`) vs Z-stage-then-`mov`:** The real GEMM kernels deliberately **stage C through Z registers** (`ld1w {z,z}` then `mov zaNh.s[w,0:3],{z..}`) for the contiguous-but-strided C load/store, while using **direct `ldr za[w,#0]` only for the B-transpose path** (`_gemm_micro_32_no_trans`, lines 841–860 load B straight into ZA, then read it out *vertically* `za0v.s` to transpose). The takeaway encoded in the code: **operands for FMOPA must be staged in Z registers (you cannot fmopa from memory); the `ld1w`→`mov za` two-step is the standard ZA load path, and direct `ldr/str za` is reserved for whole-tile moves and the in-register transpose trick (load horizontal, read vertical).**

### Compile / build configuration (this is the "compile command")
These are assembled by Apple Clang as part of an Xcode iOS target. The required arch/feature flags (from `MicrobenchmarkApp/README.md` step 10) are:

```
-march=v9-a+sme2+sme2p1+sme-f16f16+b16b16+sme-f64f64
```

Equivalent standalone clang invocation for the `.s` + driver (deployment-targeted at the M4 / iOS arm64):
```bash
clang -O3 -target arm64-apple-ios17.5 \
      -march=armv9-a+sme2+sme2p1+sme-f16f16+b16b16+sme-f64f64 \
      kernels_gemm.s gemm.cpp -framework Accelerate -o gemm_bench
```
(The GemmApp libxsmm path additionally uses `iOS.cmake` with `CMAKE_IOS_SDK_ROOT=.../iPhoneOS17.5.sdk`, `CMAKE_OSX_ARCHITECTURES=arm64`, `make -j BLAS=0`, plus `patchIos` which disables `pthread_jit_write_protect_np` so JIT-emitted SME code is executable on Apple Silicon — relevant if we ever JIT instead of hand-write.)

> For our cactus build we only need the `-march=armv9-a+sme2...` feature string on the file containing `matmul_sme2.cpp` (or the `.s`), plus a runtime SME check (`_sme_support`/`_sme2_support` in `kernels.s` lines 20–37 show the probe pattern: `smstart; fmopa/bfdot; smstop`).

---

## 5. File-path index (everything quoted above)
- FP32 single-FMOPA M0 oracle: `MicrobenchmarkApp/kernels.s:957` (`_example_sme_fmopa_fp32_fp32_fp32`); driver `MicrobenchmarkApp/examples.c:59`.
- FP32 full GEMM (4-ZA-tile, K-loop, Z-staging): `MicrobenchmarkApp/kernels_gemm.s:113` (`_gemm_micro_32_32_32`); simplest `:1` (`_gemm_micro_64_16_2`); 128-blocked `:1304` (`_gemm_128_128_128`); transpose-via-ZA `:809` (`_gemm_micro_32_no_trans`). Decls: `MicrobenchmarkApp/kernels_gemm.h`.
- INT8 SMOPA probe (M2 instruction reference): `MicrobenchmarkApp/kernels.s:1630` (`_peak_sme_smopa_i8_i8_i32`); INT16 `:1698`.
- FMOPA tiling / ZA-reuse perf probes: `MicrobenchmarkApp/kernels.s:550,618,686,754`; smstart/smstop cost `:1117+`.
- Validation harness (vs Accelerate `cblas_sgemm`): `MicrobenchmarkApp/gemm.cpp:35` (`rep_kernel`), `:52` (`bench_gemm`, error check `:180`), `:267` (`run_gemm` shapes).
- Build flags: `MicrobenchmarkApp/README.md:17`; libxsmm iOS: `GemmApp/README.md`, `GemmApp/iOS.cmake:20`, `GemmApp/patchIos`.
