# SME2 / SVE2 Integration — Knowledge Base

This directory is the **living knowledge base** for the Cactus SME2/SVE2 matmul effort. Everyone
working on this code should follow the protocol below so knowledge compounds instead of being
re-derived.

## Protocol (read-before / write-after)
1. **Read-before:** read this `README.md` (status board) → `gotchas.md` → the relevant `api-notes.md`
   entry → the tail of `debug-log.md` for the active milestone. Do **not** re-derive a fact already
   recorded here.
2. **During:** when you discover a new intrinsic fact or trap, append it to `api-notes.md` /
   `gotchas.md` immediately, with compile/asm/run evidence.
3. **Write-after:** append a dated entry to `debug-log.md` (hypothesis → experiment → result →
   decision). Promote a snippet to `working-examples/` only after its correctness/asm is verified.
   Update the status board row below.

## Layout
- `README.md` — this file: index + status board.
- `api-notes.md` — verified ACLE/intrinsic facts + Cactus kernel-layout notes.
- `gotchas.md` — traps and constraints.
- `sources.md` — annotated external references.
- `working-examples/` — verified, compilable SME/SVE snippets (trusted corpus).
- `debug-log.md` — append-only chronological experiment log.
- `research/` — raw deep-dive research notes.
  **START HERE: `research/SYNTHESIS.md`** — the consolidated, actionable implementation plan
  (verified `-march`/intrinsics, SMOPA tiling, Cactus re-tile, M0→M3b recipe, top-5 risks). It rolls
  up `research/{acle-sme-intrinsics, apple-macos-runtime, kleidiai-int4-sme2, scalable-sme-reference,
  sve2-android-fallback}.md` + `api-notes.md` + `gotchas.md` + `working-examples/*`.

## Hardware / toolchain (verified this session)
- **Apple M4 Pro**, macOS Darwin 25.3, **Apple clang 17.0.0**.
- `FEAT_SME=1`, `FEAT_SME2=1`, `FEAT_BF16=1`, `FEAT_I8MM=1`, `FEAT_SME_I16I64=1`, `FEAT_SME_F64F64=1`.
- `FEAT_SME_F16F16=0`, `FEAT_SME_B16B16=0`, `FEAT_SME2p1=0` → **MOPA accumulates into FP32 `za32`**.
- **SVL = 64 bytes = 512 bits = 16 FP32 / 32 BF16 / 64 INT8 lanes** (`hw.optional.arm.sme_max_svl_b=64`).
- Apple Silicon does **not** expose SVE/SVE2 to userspace → SVE2 path is Android/Linux only (QEMU-tested here).

## Status board
| Milestone | Description | State | Best vs NEON | Log |
|---|---|---|---|---|
| Part 1 | Matmul test infra green + baseline | **done** | baseline captured | see debug-log 2026-06-09 |
| Part 1b | Variant registry in harness (NEON vs SME2, CQ1-4 + GEMM) | **done** | — | test_matmul.cpp: cactus_quant_set_backend + 11 tests |
| Research | Consolidated synthesis (`research/SYNTHESIS.md`) | **done** | — | march+intrinsics+tiling verified; standalone matmuls run-correct |
| M0 | FP32 FMOPA smoke (ZA/streaming plumbing) | **done** | — | working-examples/fp32_fmopa_matmul.cpp PASS (max_err=0) |
| M1 | FP16/BF16 MOPA → FP32 accumulate | deferred (not needed for CQ4) | — | svmopa_za32_f16/bf16 verified to compile+run; skipped per CQ4 focus |
| M2 | INT8 SMOPA (operand layout reverse-engineered) | **done** | — | working-examples/int8_smopa_matmul.cpp PASS (err=0) |
| M3 | Wire SMOPA into CQ4 `expanded` path, M=1 GEMV | **done ✅** | 0.26–0.56× (perf-deferred) | matmul_cq4[sme2] PASS, MSE≤0.1 via real cactus_quant_matmul |
| M3b | CQ4 GEMM, M>1 (incl. tail) | **done ✅** | 0.49× @256×1024×1024 | matmul_cq4_M5/M20[sme2] PASS, MSE≤0.1 |
| M4/CQ4 | Optimize CQ4 GEMM (full-tile nr=16 gather) | **done ✅** | **1.08–1.18× @ M≥128** | beats tuned NEON SDOT; auto-dispatched for M≥128 |
| M5/GEMV | Hybrid NEON+SME LUTI4 GEMV, FUSED preamble (one dispatch) | **done ✅✅** | **1.41–1.96× @ model shape (6/6 runs)**; <6MB → NEON by policy | ~489–535 GFLOPS vs NEON 253–380; thermal-stable; see debug-log Phase1 DONE |
| M6/GEMM | Fused LUTI4 SMOPA GEMM ((M-tile×SB) stealing, no gather) | **done ✅✅** | **2.4–3.6× at EVERY M** (M=32: 3.6×; M=256: 2.5×, ~3.0 TFLOPS) | auto-dispatched for all M>1 with cache; prod: 1024³=3603, 2048³=5479 GFLOPS |
| M7/ATTN | SME prefill attention (cached-int8 segment: flat-scale QK seg + u8 USMOPA AV per-block) | **done ✅✅** | **4.6× global / 2.7× sliding** @ gemma shape (3.76→0.81ms / 2.23→0.83ms) | attention_hybrid.cpp `cactus_attention_sme_prefill`; gate hd%64==0, qg=32, pos≥cache; differential+oracle test in test_attention.cpp; see debug-log 2026-06-09 attn entries |
| SVE2 | Android/Linux fallback (svmmla/svusmmla) | designed (research/sve2-android-fallback.md) | — | not built — needs Android/QEMU to validate; out of CQ4-on-M4 scope |

**GOAL 6 COMPLETE (2026-06-09): decode +34.0% / TTFT −34.1% — both targets MET** (6-cycle protocol
c2–c6; worst decode cycle +28.6%). Three levers: SME prefill attention; batched-parallel orthogonal
embedding un-rotation (was serial scalar K² per token per layer — found by WALL profiling after
busy-sample shares misled); wait_all→spin-join + main-as-worker-0 in hybrid GEMV (kernel: o_proj
3.18×, down 2.78×, gate_up 1.62×, q_proj 1.41× now dispatched at ≥3M elements). SVL-64 runtime
gate added for non-Apple SME2 devices.

**GOAL 6 first phase (2026-06-09, SME prefill attention):** cached-int8 attention segment via SMOPA (flat-scale
QK seg leaf + u8 USMOPA AV with per-block P scales), incumbent keeps the fp16 new segment from the
merged flash state. Kernel 4.6×/2.7× (global/sliding) at Gemma shapes. **E2E clean (c2,c3,c6):
prefill +30.9%, decode +20.0%, TTFT −23.6% (floor −20% MET; −25% target 1.4pt short).** Remaining
levers: transpiler tail-sized prefill graphs (~−31% TTFT projected), ffn redesign for decode +25%.

**GOAL 5 (2026-06-09, "multiply the E2E win"):** Phase-0 profiling found the ORTHOGONAL lm_head
(fp16-codebook dot) was ~45% of decode; shipped cactus_quant_orth_sme_gemv (virtual 12x128-group
remap onto the hybrid). **E2E: decode +20.3% (gate +15% ✓ exceeded), TTFT -17.9% (floor -20%,
~88ms short), prefill +21.7%.** Remaining TTFT levers documented in debug-log: M=44 unpadded tail
GEMM, act-pack NEON-ization, prefill attention. Decode now 41.6 vs 34.6 tok/s.

**GOAL 4 COMPLETE (2026-06-09, production formats):** interleaved-4row CQ4 re-gate done — fixture
+ registry cover both layouts (bit-exact vs the production kernel; cache layout-invariant); hybrid
restored with the interleaved NEON co-worker; policy: K-heavy>=3MB or >=6MB. Kernel: down/o_proj
2.0-2.3x, ffn ~1.08, lm_head ~1.10 (DRAM-capped, documented). **E2E Gemma 4 E2B: decode +5.7%,
TTFT -13.5%, prefill +15.5%.**

**GOAL 3 COMPLETE (2026-06-09):** SME2 CQ4 decisively beats NEON — GEMV 1.4–2.0× steady-state at
model shapes (NEON below 6MB by honest policy), GEMM 2.4–3.6× across every M. Production auto-mode:
cq4 1024³ **3603 GFLOPS** / 2048³ **5479 GFLOPS** vs session-start NEON baseline 1813 / 2280.
Key design rules that got there (all verified, see gotchas): count pool DISPATCHES not preamble µs
(one fused dispatch per call); never read ZA mid-stream (stores only); maximize MACs/queue-op
(SMOPA 4-tile > vg1x4 > svdot); stream packed nibbles + LUTI4 (half the bytes); steal fine-grained
(M-tile × SB) pairs so all 14 workers stay busy at every M.

### SME2 vs NEON SDOT — measured (M4 Pro, cq4, K=N=1024, forced backends)
| M (batch) | NEON GFLOPS | SME2 GFLOPS | ratio | dispatch |
|---|---|---|---|---|
| 1 (GEMV) | ~300–320 (model) | ~80 | 0.25–0.48× | NEON (perf-deferred) |
| 16 | 249 | 216 | 0.87× | NEON |
| 32 | 431 | 323 | 0.75× | NEON |
| 64 | 652 | 631 | 0.97× | NEON |
| 128 | 991 | 1073 | 1.08× | **SME2 (auto)** |
| 256 | 1343 | 1475 | 1.10× | **SME2 (auto)** |

Auto policy (`cactus_quant_use_sme_gemm`, matmul.cpp): backend 0=auto (SME2 iff M≥128 & `cpu_has_sme2()`),
1=force NEON, 2=force SME2. GEMV always NEON (outer-product engine underutilized at M=1). Per-call
weight gather is not yet cached — caching `expanded_sme` at model-load would widen the SME2 win to
smaller M (documented follow-up).

**DEFINITION OF DONE MET (2026-06-09):** SME2 CQ4 path correct for M=1 GEMV + M>1 GEMM through the real
`cactus_quant_matmul`, validated against `cq_reference_gemv_f32` at MSE≤0.1, behind `cpu_has_sme2()`
dispatch (force via `cactus_quant_set_backend(2)`), NEON auto-default intact, base lib builds on the
unchanged `-march=armv8.2-a+...+i8mm` baseline (SME isolated in `matmul_sme2.cpp` + per-source flag).
Perf is currently below NEON → SME stays perf-deferred (not auto-dispatched); M4 optimization ongoing.

## Baseline (NEON, M4 Pro, this session)
| kernel | 1024³ | 2048³ | model GEMV 1×2304×9216 | MSE(CQ4) |
|---|---|---|---|---|
| f16 | 1939 GFLOPS | 2657 GFLOPS | 33 | — |
| cq4 | 1813 GFLOPS | 2280 GFLOPS | 301 | 3.3e-5 |

Dispatch rule: a variant is enabled in `cactus_quant_matmul` only if it (a) passes MSE ≤ 0.1 and
(b) beats NEON on this Mac. Correct-but-slower variants are logged "perf-deferred", not dispatched.
