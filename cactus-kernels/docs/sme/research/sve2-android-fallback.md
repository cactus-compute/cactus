# SVE2 (non-SME) INT8 matmul — Android/Linux fallback path

> Raw research notes. Apple Silicon does **not** expose SVE/SVE2 to userspace, so this path is
> for the **Android/Linux mobile target only** (validated here via cross-compile + QEMU/Docker, not
> native). All compile/macro/asm facts below were verified on this Mac with **Apple clang 17.0.0**
> cross-targeting `aarch64-linux-gnu` (`--target=aarch64-linux-gnu -march=...`). Date: 2026-06-09.

## TL;DR for Cactus
- The SVE2 INT8 matmul **high-throughput** primitive is `svmmla_s32` (signed) / `svusmmla_s32`
  (unsigned×signed). Each does a **2×8 · 8×2 → 2×2** widening INT8 matmul-accumulate per 128-bit
  segment (8 INT8 MACs per 32-bit output cell). The **GEMV-friendly** primitive is `svdot_s32`
  (4-way widening dot, no transpose/interleave needed) — direct analog of the NEON `vdotq_*` the CQ
  GEMV path already uses (`cactus_quant_sdot_gemv_int8`, matmul.cpp:651).
- **Guard:** the mmla intrinsics need `__ARM_FEATURE_SVE_MATMUL_INT8`, which is only defined when you
  pass **`+i8mm`** in addition to `+sve2`. `+sve2` alone is NOT enough (verified: `svmmla_s32` →
  `error: needs target feature sve,i8mm`). `svdot_s32` needs only `__ARM_FEATURE_SVE` (`+sve`/`+sve2`).
- **march for NDK:** `-march=armv9-a+sve2+i8mm` (or `armv8-a+sve2+i8mm`). Confine to its own TU exactly
  like the planned `src/matmul_sve2.cpp` (api-notes.md "Build wiring"). Never on the library-wide flag.
- **Runtime detect (Android):** `getauxval(AT_HWCAP2)` with `HWCAP2_SVE2=(1<<1)`,
  `HWCAP2_SVEI8MM=(1<<9)`, `HWCAP2_I8MM=(1<<13)`. SVE *mmla* needs **both** SVE2 and SVEI8MM bits.
- **SoC reality:** Most shipping Android flagships are still **NEON-i8mm only, no SVE2**. SVE2 arrives
  with Armv9 cores: Cortex-X4/X925/A720/A725 (Dimensity 9300/9400, some Exynos) have **SVE2** but
  **no SME**. Qualcomm Oryon (Snapdragon 8 Elite Gen 1/Gen 2 = "8 Gen 4/5") was **Armv8.7, no SVE** in
  Gen 1; **SVE2 + SME1** is reported for the Gen-2/Gen-3 Oryon. **No shipping phone has SME2.** So SVE2
  is the realistic mobile matmul-accel fallback when SME2 is absent.

---

## 1. The SVE2 INT8 matmul intrinsics

### 1.1 `svmmla_s32` / `svmmla_u32` — widening 2×2 INT8 matrix-multiply-accumulate
ACLE source: SVE ACLE §7.3.1 "MMLA: Accumulating widening multiplication of integer matrices",
guarded by `__ARM_FEATURE_SVE_MATMUL_INT8`. `[source: acle_sve_100987_0000_06_en.pdf §7.3]`

```c
// #if defined(__ARM_FEATURE_SVE_MATMUL_INT8)   // maps to SMMLA / UMMLA
svint32_t  svmmla_s32(svint32_t  op1, svint8_t  op2, svint8_t  op3);   // SMMLA
svuint32_t svmmla_u32(svuint32_t op1, svuint8_t op2, svuint8_t op3);   // UMMLA
```

**What it computes (ACLE verbatim):** "They partition the inputs into 128-bit quadwords, with the
first input containing a row-by-row **2×2 matrix of 32-bit integers**, the second input containing a
row-by-row **2×8 matrix of 8-bit integers**, and the third input containing a column-by-column
**8×2 matrix of 8-bit integers**. For each quadword, they multiply the second input matrix by the
third input matrix using natural arithmetic and then add the result to the first input using modular
arithmetic." → Each of the 4 INT32 cells in a 128-bit segment = `op1[cell] + Σ_{k=0..7} op2[row,k] *
op3[k,col]`, i.e. **8 INT8 MACs per output cell, 32 INT8 MACs per 128-bit segment**, repeated across
every 128-bit segment of the scalable vector (so 2× that at VL=256, 4× at VL=512). "The result is
well-defined for all inputs; there is no undefined behavior for signed overflow."

### 1.2 `svusmmla_s32` — mixed signedness (unsigned activations × signed weights)
ACLE §7.3.2, maps to **USMMLA**. Same 2×8·8×2→2×2 semantics, but `op2` (the 2×8) is **unsigned**,
`op3` (the 8×2) is **signed**. `[source: acle §7.3.2]`

```c
svint32_t svusmmla_s32(svint32_t op1, svuint8_t op2, svint8_t op3);    // USMMLA
```

This is the natural shape for asymmetric/uint8 activations against int8 weights (the usual quant LLM
layout). There is intentionally **no `svsumla`** (signed×unsigned) — if your A is signed and B
unsigned, swap operands / transpose, or bias-correct.

### 1.3 `svdot_s32` — 4-way widening dot (the GEMV primitive)
ACLE §6.7.13 "DOT: Integer addition of dot product", needs only `__ARM_FEATURE_SVE` (no i8mm).
`[source: acle §6.7.13]`

```c
svint32_t  svdot_s32 (svint32_t  op1, svint8_t  op2, svint8_t  op3);   // SDOT
svuint32_t svdot_u32 (svuint32_t op1, svuint8_t op2, svuint8_t op3);   // UDOT
svint32_t  svdot_n_s32(svint32_t op1, svint8_t  op2, int8_t   op3);    // SDOT, splat scalar
svint32_t  svdot_lane_s32(svint32_t op1, svint8_t op2, svint8_t op3, uint64_t imm_index); // SDOT (indexed)
// also svusdot_s32 (USDOT, unsigned×signed) under __ARM_FEATURE_SVE_MATMUL_INT8 (§7.3.3)
```

ACLE verbatim: "They partition the second and third vector inputs into **groups of four elements**.
They calculate the dot product of each group (without loss of precision) and then add each result to
the overlapping element of the first vector input." So `svdot_s32` does `acc[j] += Σ_{c=0..3}
a[4j+c]*b[4j+c]` — **4 INT8 MACs per 32-bit lane**, accumulating in place. The `_lane` form replicates
one 4-element group from each 128-bit quadword of `op3` (`imm_index ∈ [0, 128/N)`), the SVE analog of
NEON `vdotq_laneq_s32` — directly mirrors `CACTUS_DOTQ_LANE` (matmul.cpp:23).

### 1.4 mmla vs dot — throughput
- `svmmla` packs **2× the INT8 MACs per instruction** of `svdot` (32 vs 16 per 128-bit segment at the
  same VL) but requires the data **interleaved into 2×8 / 8×2 tiles** (the `trn1/trn2` deinterleave you
  see in real kernels). Worth it for **GEMM (M>1)**.
- `svdot` needs **no transpose**, accumulates straight down K — ideal for **GEMV (M=1)**, which is the
  Cactus decode/`expanded` path. For M=1 there is no second A-row to fill the mmla 2×8, so dot is the
  right tool; mmla pays off in prefill / M-batched GEMM.

### 1.5 Verified on this Mac (Apple clang 17, cross-target)
```
$ clang --target=aarch64-linux-gnu -march=armv9-a+sve2+i8mm -O2 -S sve2_probe.c
  smmla   z0.s, z1.b, z2.b      # from svmmla_s32
  usmmla  z0.s, z1.b, z2.b      # from svusmmla_s32
  sdot    z0.s, z1.b, z2.b      # from svdot_s32
$ clang --target=aarch64-linux-gnu -march=armv9-a+sve2+i8mm -dM -E -x c /dev/null | grep MATMUL
  #define __ARM_FEATURE_MATMUL_INT8 1        # NEON i8mm (vmmla/vusmmla)
  #define __ARM_FEATURE_SVE_MATMUL_INT8 1    # SVE  mmla (svmmla/svusmmla)
```
**Guard gotcha (verified):** with `-march=armv9-a+sve2` (no `+i8mm`) → `__ARM_FEATURE_SVE_MATMUL_INT8`
is **undefined** and `svmmla_s32` fails to compile (`'svmmla_s32' needs target feature sve,i8mm`).
`svdot_s32` still works (only needs SVE). So:
- `#if defined(__ARM_FEATURE_SVE2)` → gate the dot-based GEMV.
- `#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)` → gate the mmla-based GEMM (implies you compiled `+i8mm`).

---

## 2. Representative SVE2 INT8 GEMM tile loop (open-source references)

### 2.1 `svdot` GEMV inner loop — llama.cpp / ggml (closest to Cactus's CQ GEMV)
`ggml/src/ggml-cpu/arch/arm/quants.c`, `ggml_vec_dot_q8_0_q8_0`, guarded `#if defined(__ARM_FEATURE_SVE)`.
The 128-bit-VL branch (each block = 32 INT8 quants):
```c
const svint8_t qx0_0 = svld1_s8(ph16, x0->qs);
const svint8_t qx0_1 = svld1_s8(ph16, x0->qs+16);
const svint8_t qy0_0 = svld1_s8(ph16, y0->qs);
const svint8_t qy0_1 = svld1_s8(ph16, y0->qs+16);

sumv0 = svmla_n_f32_x(pl16, sumv0,
          svcvt_f32_s32_x(pl16,
            svadd_x(pl16,
              svdot_s32(svdup_n_s32(0), qx0_0, qy0_0),
              svdot_s32(svdup_n_s32(0), qx0_1, qy0_1))),
          GGML_CPU_FP16_TO_FP32(x0->d)*GGML_CPU_FP16_TO_FP32(y0->d));
// ...
sumf = svaddv_f32(pl16, svadd_f32_x(pl16, sumv0, sumv1));  // horizontal reduce to scalar
```
Pattern = **VLA tiling** (`svcntb()`/predicates pick the branch), `svdot_s32` accumulates INT8 K, then
dequant-scale into FP32 and `svaddv_f32` reduces. This is the structure Cactus's `expanded`/SDOT GEMV
maps onto: replace the per-block `vdotq_s32` chain with `svdot_s32` over a `svcntb()`-sized K stride,
keep the FP32 rescale (by `norm_f32`) in the non-SVE caller. ggml's own VLA convention:
`#define VECTOR_REGISTERS (svcntb()*8/32)` and `svcntb()`-strided K loops. `[source: ggml-org/llama.cpp]`

### 2.2 `svmmla` GEMM inner loop — Arm Compute Library `arm_gemm` (the 2×2-tile reference)
`src/core/NEON/kernels/arm_gemm/kernels/sve_hybrid_s8s32_mmla_6x4VL/generic.cpp` — a 6(M)×4VL(N) INT8
hybrid GEMM, K blocked in groups of 8. Core mmla accumulation (emitted as `.inst` SMMLA encodings; the
intrinsic equivalent is `svmmla_s32`):
```asm
"whilelt p0.b, XZR, x27\n"                  // K-tail predicate over bytes
"ld1b { z7.b }, p5/Z, [x11]\n"              // load B panel (8x2 tiles, pre-packed)
"ld1b { z6.b }, p5/Z, [x11, #1, MUL VL]\n"
"subs  x27, x27, #0x8\n"                    // K -= 8 per iteration
"ld1rqb { z1.b }, p0/Z, [x26]\n"            // broadcast-load 16B of A row 0 (ld1rqb = quad replicate)
"ld1rqb { z2.b }, p0/Z, [x25]\n"            //               A row 1
"trn1  z0.d, z1.d, z2.d\n"                  // interleave A rows -> 2x8 tile halves
"trn2  z1.d, z1.d, z2.d\n"
".inst 0x45079808  // smmla z8.s,  z0.b, z7.b\n"   // == svmmla_s32(z8, z0, z7)
".inst 0x4506980c  // smmla z12.s, z0.b, z6.b\n"
"ld1b { z7.b }, p5/Z, [x11, #2, MUL VL]\n"
".inst 0x45079809  // smmla z9.s,  z0.b, z7.b\n"
".inst 0x4506980d  // smmla z13.s, z0.b, z6.b\n"
"... (z10..z15 across the 4VL N-panel) ...\n"
"addvl x11, x11, #8\n"                      // advance B by 8 vectors
```
Key takeaways for re-tiling Cactus weights for mmla:
- **A** (activations) is loaded with `ld1rqb` (quad-word replicate) then **`trn1/trn2`-deinterleaved**
  into the 2×8 layout mmla expects — two M-rows per mmla. For Cactus M=1 GEMV this degenerates (only
  one A row), confirming **dot, not mmla, for decode**.
- **B** (weights) must be **pre-packed** into the 8×2 column-major tile layout (`ld1b` straight in, no
  transpose at runtime). Cactus's current `expanded` layout (N-blocks of 4 rows, interleaved for
  `vdotq_laneq_s32`, api-notes.md) is the **wrong** packing for mmla — a dedicated `expanded_sve2`
  pack (8×2 K-major panels) is needed, same conclusion as the SME MOPA re-tile note (api-notes.md:49).
- K is consumed in **steps of 8** (mmla's contraction depth), N across `4*VL` columns, M in 6-row
  strips, with multiple `z*.s` accumulators to hide the ~mmla latency. `[source: ARM-software/ComputeLibrary]`

ACL also ships `sve_hybrid_u8u32_dot_*` (svdot GEMM), `sve_hybrid_u8s8s32_mmla_*` / `_usmmla` (the
mixed-sign USMMLA variant for uint8 activations × int8 weights) — those are the templates for a
`svusmmla_s32` Cactus path. KleidiAI (already the SME2 reference, sources.md) has matching SVE2
`...sve2_...` ukernels (`kai_matmul_clamp_*_qai8dxp_qsi4c32p_..._dot`) for the CQ4 shape.

---

## 3. Android runtime feature detection (`getauxval(AT_HWCAP2)`)

Authoritative bit values from the Linux kernel UAPI header
`arch/arm64/include/uapi/asm/hwcap.h` (verbatim, current master). `[source: torvalds/linux .../hwcap.h]`
```c
#define HWCAP_SVE         (1 << 22)   // base SVE   (AT_HWCAP, not HWCAP2)
#define HWCAP2_SVE2       (1 << 1)
#define HWCAP2_SVEI8MM    (1 << 9)    // SVE  int8 matmul (svmmla/svusmmla)  <-- needed for mmla
#define HWCAP2_I8MM       (1 << 13)   // NEON int8 matmul (vmmla/vusmmla)    <-- Cactus already uses this
#define HWCAP2_SME        (1UL << 23)
#define HWCAP2_SME2       (1UL << 37) // note: needs UL — bit 37 overflows a 32-bit int
#define HWCAP2_SME2P1     (1UL << 38)
```
All SVE2/SME bits live in **AT_HWCAP2** (base `HWCAP_SVE` is in AT_HWCAP bit 22). `AT_HWCAP3` exists in
newer kernels but only carries MTE/LS64 bits — irrelevant here.

Drop-in detectors beside `cpu_has_i8mm()` (threading.h:50). Note `getauxval` returns `unsigned long`
(64-bit on arm64), so bits 37/38 are reachable; **do not** assign HWCAP2 to a 32-bit `int`.
```c
// Android only (Apple uses sysctl hw.optional.arm.FEAT_SME / FEAT_SME2; Apple has NO SVE2)
inline bool cpu_has_sve2() {        // svdot GEMV path
#if defined(__ANDROID__) && defined(__aarch64__)
    unsigned long h2 = getauxval(AT_HWCAP2);
    #ifndef HWCAP2_SVE2
    #define HWCAP2_SVE2 (1UL << 1)
    #endif
    return (h2 & HWCAP2_SVE2) != 0;
#else
    return false;
#endif
}
inline bool cpu_has_sve_i8mm() {    // svmmla / svusmmla GEMM path (requires SVE2 + SVEI8MM)
#if defined(__ANDROID__) && defined(__aarch64__)
    unsigned long h2 = getauxval(AT_HWCAP2);
    #ifndef HWCAP2_SVE2
    #define HWCAP2_SVE2 (1UL << 1)
    #endif
    #ifndef HWCAP2_SVEI8MM
    #define HWCAP2_SVEI8MM (1UL << 9)
    #endif
    return (h2 & HWCAP2_SVE2) && (h2 & HWCAP2_SVEI8MM);
#else
    return false;
#endif
}
```
Caveats: (1) `getauxval(AT_HWCAP2)` since Android API 18; older toolchains may lack the `HWCAP2_*`
macros → self-`#define` as above. (2) Google's `cpu_features` lib exists for SoC-quirk workarounds, but
HWCAP for SVE2/i8mm is reliable on Armv9 (the old div-by-zero-claim quirk was Armv7-era). (3) Always
**also** gate at compile time (`#if defined(__ARM_FEATURE_SVE2)`) so the base lib still links on older
NDKs and non-SVE devices.

### Which Android SoCs actually have SVE2 vs SME (as of mid-2026)
- **SVE2, no SME (the realistic SVE2 target):** Armv9 Cortex cores — **Cortex-A510/A520, A710/A715/
  A720/A725, X2/X3/X4/X925**. SoCs: MediaTek **Dimensity 9300/9400** (X4/X925 + A720/A725),
  Exynos 2400, Snapdragon 8 Gen 1/2/3 (those used Armv9 Cortex, SVE2). These have **no SME/SME2**.
- **Qualcomm Oryon (custom):** Snapdragon 8 Elite **Gen 1** (a.k.a. "8 Gen 4", 2024) Oryon was
  **Armv8.7, NO SVE/SVE2**. Reporting indicates **Gen-2/Gen-3 Oryon** ("8 Elite Gen 2/Gen 5") add
  **SVE2 + SME1**. So newest Snapdragon may be the first phones with SME — but **SME1, not SME2**.
- **No shipping Android phone has SME2.** SME2 is Apple-M4-class / future. ⇒ On mobile, the dispatch
  ladder is: **SME2 (rare/none) → SVE2+i8mm (Armv9 Cortex, newest Oryon) → NEON i8mm (everything else,
  Cactus's current baseline) → NEON dotprod**.
`[source: ARM Cortex-X925 / Dimensity 9400 specs; Snapdragon 8 Elite deep-dives; chips-and-cheese Oryon]`

---

## 4. Build flags + testing under QEMU on this Mac

### 4.1 NDK / clang `-march`
- SVE2 GEMV (dot): `-march=armv9-a+sve2`  (or `armv8-a+sve2`).
- SVE2 INT8 GEMM (mmla): **`-march=armv9-a+sve2+i8mm`** — `+i8mm` is **required** for
  `__ARM_FEATURE_SVE_MATMUL_INT8` (verified §1.5). It also (re)defines `__ARM_FEATURE_MATMUL_INT8`
  for the NEON path, which is harmless. `armv9-a` implies SVE2; on `armv8-a` you must spell `+sve2`.
- **Per-TU only.** Mirror api-notes.md "Build wiring": new `src/matmul_sve2.cpp` via
  `set_source_files_properties(src/matmul_sve2.cpp PROPERTIES COMPILE_OPTIONS
  "-march=armv9-a+sve2+i8mm")` in CMakeLists.txt; keep the library-wide flag
  `-march=armv8.2-a+fp16+simd+dotprod+i8mm` (line 34) unchanged. Double-guard the TU
  (`#if defined(__ARM_FEATURE_SVE2)` real / `#else` stub). **Never** put `+sve2` on the global flag —
  the optimizer would emit SVE into NEON TUs and the base lib stops running on non-SVE phones.
- NDK clang accepts these as-is (the same Apple clang 17 frontend; LLVM ≥ 11 has the SVE matmul
  builtins). For LTO, the per-source march is preserved by the function multi-versioning attrs; if you
  hit LTO march-clobber, gate with `__attribute__((target("arch=armv9-a+sve2+i8mm")))` per-function.

### 4.2 Testing on this Mac — important limitation
Apple Silicon does **not** run SVE2 in userspace (PSTATE has no SVE), so you cannot run the SVE2 TU
natively. Two emulation options:
- **`qemu-aarch64` user-mode (preferred, fast iteration) — NOT installed here.** Verified:
  `brew --prefix qemu`/bin ships **system-mode only** (`qemu-system-aarch64` 10.2.2 present; **no
  `qemu-aarch64` / `qemu-arm` user binaries** — macOS Homebrew QEMU is built without linux-user).
  To get it: build QEMU from source with `--target-list=aarch64-linux-user`, or run inside a Linux
  shell. Once present:
  ```sh
  # cross-build a Linux aarch64 test binary, then:
  qemu-aarch64 -cpu max,sve=on,sve2=on,i8mm=on,sme=off ./test_matmul_sve2
  # -cpu max enables all SVE sub-features incl SVE2 + SVEI8MM; force sme=off to exercise the SVE2 fallback,
  #  not an SME path. (qemu also forwards correct HWCAP2 SVE2/SVEI8MM bits to getauxval.)
  ```
- **Docker (works today on this Mac — `docker` is installed):** Docker Desktop's Linux VM has
  `qemu-user-static` + binfmt registered, so an `--platform linux/arm64` container transparently runs
  aarch64 ELF through QEMU TCG. To get SVE2 you must point binfmt's QEMU at `-cpu max`; simplest is to
  run the cross-built binary under an explicit `qemu-aarch64-static -cpu max,sve2=on` **inside** an
  arm64 Linux container (the container's qemu-user-static *does* ship user-mode). Sketch:
  ```sh
  docker run --rm --platform linux/arm64 -v "$PWD:/w" -w /w debian:trixie \
    bash -c 'apt-get update && apt-get install -y qemu-user && \
             qemu-aarch64 -cpu max,sve2=on,i8mm=on ./test_matmul_sve2'
  ```
  (Slow — TCG, no SVE2 host accel — fine for **correctness/MSE gate**, useless for the perf gate.
   Project policy: don't run long live sweeps. Correctness only here; perf must be measured on a
   real Armv9 Android device.)
- **`-cpu max` knobs:** `sve=on` auto-enables all supported SVE sub-features incl `sve2`; add `i8mm=on`
  for SVEI8MM (mmla). You can pin a VL with `sve512=on`/`vl=…` to test VL-agnosticism at 128/256/512.

### 4.3 Dispatch gate (consistent with status-board rule)
A SVE2 variant is enabled in `cactus_quant_matmul` only if it (a) passes MSE ≤ 0.1 (QEMU-verifiable
here) and (b) beats the NEON i8mm baseline **on real Armv9 hardware** (NOT on QEMU). Until a device
benchmark exists, ship it `runtime-detect → dispatch` but flag "perf-unverified-on-device".

---

## Sources
- ACLE for SVE (Arm 100987_0000_06) §6.7.13 DOT, §7.3 INT8 matmul — intrinsic prototypes + 2×8·8×2→2×2
  semantics (extracted from PDF, verbatim above).
- Linux kernel `arch/arm64/include/uapi/asm/hwcap.h` (torvalds/linux master) — HWCAP2 bit values.
- LLVM/clang `mmla` builtins commit (cfe-commits e2cc12e) — `svmmla_s32/_u32/svusmmla_s32` → smmla/
  ummla/usmmla, `__ARM_FEATURE_SVE_MATMUL_INT8` guard, test signatures.
- ggml/llama.cpp `ggml-cpu/arch/arm/quants.c` — `svdot_s32` VLA GEMV inner loop.
- Arm Compute Library `arm_gemm/.../sve_hybrid_s8s32_mmla_6x4VL/generic.cpp` — smmla GEMM tile loop
  (ld1rqb + trn1/trn2 deinterleave + 6×4VL accumulators).
- Local verification: Apple clang 17 `--target=aarch64-linux-gnu -march=armv9-a+sve2+i8mm` emits
  smmla/usmmla/sdot; `+sve2` alone leaves `__ARM_FEATURE_SVE_MATMUL_INT8` undefined and rejects
  `svmmla_s32`. `qemu-system-aarch64` 10.2.2 present; no user-mode `qemu-aarch64` in brew; `docker` present.
- SoC landscape: Cortex-X925 / Dimensity 9400 specs, Snapdragon 8 Elite Oryon deep-dives (SVE2/SME
  generational support).
