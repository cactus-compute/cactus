# Sources (annotated)

> Rating: ★★★ primary / ★★ useful / ★ context. Date-checked 2026-06-09.

## Primary references (implementation-load-bearing)
- ★★★ **KleidiAI** — github.com/ARM-software/kleidiai (mirror of gitlab.arm.com/kleidi/kleidiai).
  Production SME2 INT4/INT8 matmul micro-kernels: `kai_matmul_clamp_f32_qsi8d32p..._qsi4c32p..._sme2_mopa`,
  `qai8dxp` LHS dynamic-quant+pack, `qsi4c32p`/`qsi4cxp` RHS 4-bit pack. Nearly Cactus's CQ4 use case.
  Read `kai/ukernels/matmul/**` + `pack/README.md`. THE primary tiling/packing reference.
- ★★★ **scalable-analyses/sme** + "Hello SME!" (arxiv.org/abs/2409.18779). Runnable FP32/FP16/INT8 SME
  matmul kernels benchmarked on M4; Z-reg→ZA staging is 2.6× faster than direct ZA load. M0/M2 oracle.
- ★★★ **Arm ACLE** — arm-software.github.io/acle/main/acle.html. `arm_sme.h` intrinsics + attribute
  semantics (`__arm_streaming`, `__arm_new("za")`, `__arm_streaming_compatible`, `__arm_in/out/inout`).
- ★★★ **LLVM AArch64SME** — llvm.org/docs/AArch64SME.html. ACLE→IR mapping, streaming restrictions,
  SMEABIPass, calling conventions.
- ★★★ **Arm Learning Path "Multiplying matrices with SME2"** —
  learn.arm.com/learning-paths/cross-platform/multiplying-matrices-with-sme2/ (streaming mode, ZA,
  FMOPA, C intrinsics pages). Canonical tutorial with working code.

## Apple-Silicon practicalities
- ★★ tzakharko/m4-sme-exploration — reverse-engineered M4 SME per-instruction microbenchmarks (P/E core).
- ★★ Zenn "Trying Out SME" (mod_poppo) — real M4 + clang/gcc gotchas, ACLE-spec-vs-compiler drift.
- ★★ apple-oss-distributions/xnu doc/arm/sme.md — macOS ZA context-switch/ABI.
- ★ Go `internal/cpu/cpu_arm64_darwin.go` — confirms sysctl key usage pattern for Apple feature detect.

## Tiling / quantized GEMM technique
- ★★ MpGEMM "Demystifying ARM SME" (arxiv.org/abs/2512.21473) — modern SME GEMM cache/tiling strategy,
  mixed INT4/INT8/FP16, benchmarked vs ACL/KleidiAI/OpenBLAS.
- ★★ Arm Compute Library — github.com/ARM-software/ComputeLibrary, `src/core/NEON/kernels/arm_gemm/`
  (SME/SVE2 GEMM asm kernels).
- ★ ggml/llama.cpp SME PRs + Arm KleidiAI llama.cpp patch — real inference-engine integration patterns.

## SVE2 (Android/Linux fallback)
- ★★ Arm intrinsics guide — `svmmla_s32`/`svusmmla_s32`/`svdot` (INT8 matmul, `__ARM_FEATURE_SVE_MATMUL_INT8`).
- ★ Linux kernel docs/arch/arm64/sme.md + sve.md — HWCAP2 bits, context switch.
- ★★★ ACLE for SVE (Arm 100987_0000_06) §6.7.13 DOT, §7.3 INT8 matmul — verbatim svmmla/svusmmla/svdot
  prototypes + 2×8·8×2→2×2 semantics. (2026-06-09) → see research/sve2-android-fallback.md.
- ★★★ Linux kernel `arch/arm64/include/uapi/asm/hwcap.h` (torvalds master) — exact HWCAP2 bit values:
  SVE2=1<<1, SVEI8MM=1<<9, I8MM=1<<13, SME=1<<23, SME2=1UL<<37. (2026-06-09)
- ★★ ARM-software/ComputeLibrary `arm_gemm/.../sve_hybrid_s8s32_mmla_6x4VL/generic.cpp` — smmla INT8
  GEMM tile loop (ld1rqb + trn1/trn2 deinterleave, 6×4VL accumulators). (2026-06-09)
- ★★ ggml/llama.cpp `ggml-cpu/arch/arm/quants.c` — `svdot_s32` VLA GEMV inner loop (Cactus CQ-GEMV analog).
- ★ Consolidated raw research: **research/sve2-android-fallback.md** (full SVE2 Android-fallback writeup,
  cross-compile + QEMU/Docker testing notes, SoC SVE2-vs-SME landscape). (2026-06-09)

> Append new finds here with a one-line "what it answers" + date.
