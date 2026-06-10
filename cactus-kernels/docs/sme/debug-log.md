# Debug Log (append-only)

Format per entry: `## YYYY-MM-DD [milestone] title` → Hypothesis / Experiment / Result / Decision.

## 2026-06-09 [Part 1] Baseline test infra green
- **Experiment:** built `cactus-kernels` tests on M4 Pro (Apple clang 17), ran `./test_matmul`.
- **Result:** all 5 tests pass. CQ4 MSE=3.3e-5 (gate 0.1). Baseline GFLOPS: f16 1024³=1939 / 2048³=2657;
  cq4 1024³=1813 / 2048³=2280; cq4 model GEMV 1×2304×9216 = 301 GFLOPS.
- **Decision:** harness + FP32 oracle (`cq_reference_gemv_f32`) are sound. Use as the correctness gate
  for every SME variant. Next: variant registry + M0.

## 2026-06-09 [Part 0] Hardware/toolchain probe
- **Result:** M4 Pro, `FEAT_SME=1/SME2=1/BF16=1/I8MM=1/SME_I16I64=1/SME_F64F64=1`,
  `FEAT_SME_F16F16=0`, SVL=64B. Apple clang 17 compiles `arm_sme.h` + `svmopa_za32_{f32,s8}_m` under
  `-march=armv9-a+sme2`, emits `smstart za` / `fmopa` / `smopa` (per the design-phase probe).
- **Decision:** SME2 locally testable. MOPA accumulates FP32. Proceed with confidence.

## 2026-06-09 [M0] FP32 FMOPA smoke — RAN correct
- **Experiment:** standalone 16×16×16 FP32 outer-product matmul, `svmopa_za32_f32_m` + `svst1_hor_za32`,
  compiled across -march candidates and RUN.
- **Result:** `armv9-a+sme2` **SIGILLs at runtime (exit 132)**; `armv8.2-a+fp16+simd+dotprod+i8mm+sme2`
  (and `armv8-a+sme`) RUN correct, svcntw=16, max_err=0. → working-examples/fp32_fmopa_matmul.cpp.
- **Decision:** pin `-march=armv8.2-a+...+sme2` (NOT armv9-a). Always RUN, never trust asm-only.

## 2026-06-09 [M2] INT8 SMOPA operand layout — reverse-engineered, RAN correct
- **Experiment:** 16×16×128 INT8 matmul; packed both operands `[K/4][16][4]`, byte predicates,
  `svmopa_za32_s8_m`, int32 readout vs scalar reference.
- **Result:** maxerr=0, all cells exact. Confirms `ZA[i][j] += Σ_{c<4} Zn[i*4+c]·Zm[j*4+c]` and that
  the Cactus `expanded` panel (kg-th 16 bytes = 4 channels × 4 K) is a ready-made SMOPA `pm` operand.
- **Decision:** SME does ONLY integer SMOPA; reuse NEON preamble + rescale. → working-examples/int8_smopa_matmul.cpp.

## 2026-06-09 [M3/M3b] Wire SMOPA into real cactus_quant_matmul — DONE ✅ (DoD met)
- **Implementation:** `src/matmul_sme2.cpp` leaves (`cactus_sme_cq_gemv_4col_s8`,
  `cactus_sme_cq_gemm_16x4_s8`, `__arm_locally_streaming __arm_new("za")`, byte predicates),
  per-source `+sme2` flag via CMake `check_cxx_source_compiles`; `cpu_has_sme/sme2` in threading.h;
  GEMV+GEMM drivers + `cactus_quant_set_backend` dispatch in matmul.cpp (reuse Hadamard/quant/expand
  preamble; SME returns int32 partials; identical NEON FP rescale). Variant registry in test_matmul.cpp.
- **Result:** ALL 11 matmul tests pass + full kernel suite green (clean rebuild). `matmul_cq4[sme2]`
  (GEMV), `matmul_cq4_M5/M20[sme2]` (GEMM incl. tail) correct vs FP32 oracle, MSE≤0.1.
  Head-to-head: GEMV 0.26–0.56×, GEMM 0.49× @256×1024×1024 vs tuned NEON SDOT.
- **Decision:** correctness DoD MET. SME2 stays perf-deferred (auto=NEON). Next: M4 perf — kill
  per-block streaming transitions (batch N-blocks per streaming call), use full ZA tile width
  (nr=16/64 across za0..za3), NEON-ize act packing + rescale.

## 2026-06-09 [M4/CQ4] Full-tile nr=16 GEMM — SME2 BEATS NEON ✅
- **Hypothesis:** the nr=4 GEMM wastes 12/16 ZA columns; gathering the four 4-channel `expanded`
  panels into one contiguous 16-channel SMOPA panel (4× 16-byte memcpy) gives full-tile utilization
  and 4× fewer SMOPA instructions.
- **Implementation:** `cactus_sme_cq_gemm_16x16_s8` (pm=svptrue_b8, full 16 cols) + a gather driver
  (`cactus_quant_sme_gemm_tiles`, weight+norm gather amortized over M-blocks).
- **Result:** correctness preserved (16/16 matmul tests incl. CQ1-4, M/N tails). Crossover (cq4,
  K=N=1024): M4 1.16×, M16 0.87×, M32 0.75×, M64 0.97×, M128 1.08×, M256 1.10× (up to 1.18× @256×1024×1024).
  SME2 reliably beats tuned NEON SDOT for **M≥128**.
- **Decision:** auto-dispatch SME2 GEMM for **M≥128** (`cactus_quant_use_sme_gemm`); NEON for smaller M
  and all GEMV. Numerically equivalent (same int32 SMOPA dot + same FP rescale as SDOT, MSE≤0.1).
  Full clean rebuild + entire kernel suite green. **GOAL COMPLETE: working, validated, shipping SME2
  CQ4 path.** Follow-ups: cache `expanded_sme` at model-load (widen win to smaller M); multi-ZA-tile
  nr=64; SVE2 Android leaf (svmmla) under QEMU.

## 2026-06-09 [PRODUCTION GOAL ✅ COMPLETE] E2E A/B on interleaved bundles: decode +5.7%, TTFT -13.5%
- **Phase 3 (Gemma 4 E2B IT, 1068-tok ctx, 32 tokens, MTP off, 6 alternating cycles, cycle 1
  skipped):** decode 34.41 -> 36.38 tok/s (+5.7%, non-overlapping distributions); TTFT 4099 ->
  3547 ms (-13.5%); prefill 260.7 -> 301.1 tok/s (+15.5%); total -12.1%. Text coherent, temp-0
  output matches NEON.
- **Phase 2 verdict (honest):** kernel gate >=1.15x met decisively on K-heavy shapes (down 2.1-2.3x,
  o_proj 1.9-2.2x), marginal on ffn (~1.08), structurally capped on lm_head (~1.10; both sides at
  ~200 GB/s vs ~230 DRAM ceiling — the interleave already streams 4-bit for NEON, so byte-halving
  is neutralized there). Pivot rule applied: documented the wall instead of grinding.
- **Shipped policy:** GEMV hybrid when (K>N and K*N>=3MB) or K*N>=6MB; GEMM fused for all M>1 with
  cache; k_sme=4. All correctness gates green on BOTH weight formats (19 matmul tests, 7 suites,
  bit-exact differential vs cactus_quant_4bit_gemv_interleaved, esme layout invariance).
- **New model insight:** production NEON-il collapses on K-heavy GEMV (64-190 GF; per-group loop x
  few N-blocks) — that asymmetry, not byte-halving, is where SME2 buys real decode time on this
  format. Remaining headroom: fp16 attention matmuls (untouched), ffn-shape SME-side L2-regime.

## 2026-06-09 [INTERLEAVED re-gate] Production-format hybrid: K-heavy GEMVs 2.0-2.3x, policy reshaped
- **Fixture fixed:** SyntheticCQ interleaved-4row mode (encoder = exact inverse of the shipped
  decoder); registry now gates BOTH formats; `esme_layout_invariance` proves the SME cache is
  byte-identical from either source; differential vs cactus_quant_4bit_gemv_interleaved bit-exact.
- **Hybrid restored on interleaved:** extracted cactus_quant_interleaved4_gemv_blocks (post-preamble
  core of the production kernel) as the NEON co-worker; k_sme default 4.
- **REAL baseline measured (6 alternating best-of runs, gemma shapes):** NEON-il is STRONG on
  N-heavy shapes (ffn 287-363 GF; lm_head 715-810 GF ~ at the DRAM wall since the interleave already
  streams 4-bit) but COLLAPSES on K-heavy GEMVs (down 6144x1536: 146-190 GF; o_proj 2048x1536:
  64-75 GF — per-group loop x few N-blocks starves it).
- **Hybrid results:** down 2.10-2.26x, o_proj 1.89-2.19x, ffn mean 1.075x (1.00-1.18), q_proj ~1.0x,
  lm_head mean 1.10x (1.04-1.15) — lm_head is MODEL-CAPPED: both sides stream ~200 GB/s vs ~230
  ceiling, so >=1.15x is at/beyond the wall (goal pivot rule applied, documented not ground).
- **Auto policy (shipped):** hybrid when K>N and K*N>=3MB (the 2x regime) or K*N>=6MB; NEON below /
  for small N-heavy. Projection over gemma's per-token byte mix: ~1.23x decode-matmul time.
- Phase 2 gate verdict: decisively exceeded on K-heavy shapes, marginal ffn, capped lm_head — the
  honest aggregate is the byte-weighted ~1.23x, validated next by the Phase 3 E2E A/B.

## 2026-06-09 [Phase2 ✅ DONE — GOAL COMPLETE] Fused LUTI4 SMOPA GEMM: 2.4–3.6× NEON at EVERY M
- **Design (from the queue model, predictions stated up front):** root causes of mid-M losses were
  (1) worker starvation (old driver parallelized M-tiles only: M=32 -> 2 workers) and (2) the
  per-call w_sme gather. New fused driver: ONE pool dispatch — phase A steals (M-tile, group) items
  (Hadamard + int8 quantize + SMOPA act-pack [g][kg][16r][4K]), spin barrier, phase B steals
  (M-tile x SB64) pairs. New leaf `cactus_sme_cq_gemm_luti4_s8` (-O1 TU): per (group, kg) ONE 64B
  act load + 2 luti4_x2 + 4 SMOPA into za0..za3 (the four 16-ch sub-blocks) = 4096 MACs / 9 queue
  ops (~4x the vg1x4 MAC density); per-group partials stored fire-and-forget; NEON rescale per pair.
- **Predictions vs actuals:** predicted M=32 >=600 GFLOPS (1.7x) / M=256 >=1800 (1.3x).
  Actual (cq4, K=N=1024, forced backends): M=4 3.0-3.1x, M=16 2.6-3.4x, M=32 1109-1192 GFLOPS
  (3.0-3.6x), M=64 1795-1877 (3.1-3.4x), M=128 2214-2343 (2.5-2.8x), M=256 2802-2997 (2.4-2.5x).
  Production auto rows: cq4 1024^3 = 3603 GFLOPS, cq4 2048^3 = 5479 GFLOPS (session-start NEON
  baseline: 1813 / 2280). All 55 tests green.
- **Auto-dispatch updated:** fused-cache GEMM for ALL M>1 when expanded_sme present
  (cactus_quant_use_sme_gemm_fused); legacy gather path keeps M>=128 when cache absent.
- **Why actuals beat predictions:** pair-stealing fixed starvation more than modeled AND SMOPA's
  queue-op density compounds with the single-dispatch budget (the Phase-1 lesson: count dispatches).

## 2026-06-09 [Phase1 ✅ DONE] Fused-preamble hybrid: 1.41–1.96× (6/6 runs) at model shape
- **Change:** fused Hadamard+act-quantize into the hybrid's single pool dispatch (group-parallel
  phase A, spin barrier, SB-steal phase B). Removed the standalone Hadamard parallel_for + serial
  quantize + second dispatch.
- **Result:** model shape SME2 489–535 GFLOPS vs NEON 253–380 → **1.41–1.96×, all 6 alternating
  runs ≥1.41×** (gate was ≥1.15). Total call ~135µs → ~79µs. All 55 tests green.
- **Model correction (prediction was +15-20µs, actual +56µs):** a pool dispatch round-trip costs
  ~40-50µs (wake+sync of 14 workers), not ~12µs — explains the earlier pooltest anomaly too. Fusing
  also pre-warms threads + act-buffer caches for phase B. LESSON: count DISPATCHES, not just µs of
  preamble work; one dispatch per matmul call is the budget.
- Small shape (1x1024x1024) still NEON by policy (0.79–0.95 forced; fixed costs + L2-resident
  weights). Threshold K*N>=6MB stands.
- **Phase 2 implication:** the GEMM driver parallelizes over M_blocks ONLY (M=32 → 2 workers!) —
  worker starvation explains the mid-M losses at least as much as the per-call gather. Fix both:
  steal over (M-tile × SB) pairs + LUTI4-packed leaf (no gather) + one fused dispatch.

## 2026-06-09 [Phase1/GEMV] FINAL ARC: ladder of kernels → hybrid NEON+SME, model shape won at steady state
- **Kernel ladder (model shape 1x2304x9216, vs NEON SDOT GEMV best-of):**
  mopa 4-col (M3): 0.25× → plain svdot: 0.70× → vg1x4 16-ch: 0.90× → vg1x4 64-ch (esme cache): 0.95×
  → +dynamic work-steal: ~1.0× noisy → LUTI4 packed (halve bytes): 0.85-0.95 → z-acc (no ZA): 0.55× ✗
  → **partials-to-memory + NEON rescale (no ZA reads): 1.08-1.17×** → both-packed hybrid: steady ~313 vs
  NEON warm ~280-310 / cold ~340-365.
- **Decisive experiments:** (1) lutibench: same dot stream w/o reads = 406 GMAC/s vs kernel 137 →
  in-order-queue read-drain model (see gotchas). (2) loadbw: streaming loads FASTER than NEON
  (97 vs 79 GB/s 1T) — bandwidth was never the limit; round-trips were. (3) NEON-packed GEMV = 252
  GFLOPS (vqtbl-limited, 63 GB/s) → hybrid NEON workers also read packed → call streams K*N/2 total.
- **Hybrid architecture (shipped):** one dynamic SB64 queue; workers 0..k_sme-1 run the SME LUTI4
  partials leaf + NEON rescale, the rest run cactus_quant_sdot_packed_blocks_rt (expand-from-packed
  SDOT). k_sme env CACTUS_SME_GEMV_WORKERS (default 5). Auto-dispatch: K*N >= 6MB (below: fixed
  costs dominate + weights fit L2 → NEON wins; goal's "keep NEON and log why" branch).
- **Thermal finding:** NEON fades 365→277 under sustained load; hybrid holds ~310-315. Sustained
  decode (real workload) favors the hybrid; cold benches flatter NEON.
- **Open levers (next):** fuse preamble (Hadamard+quantize) into the hybrid's pool dispatch (saves a
  round-trip + parallelizes quantize, ~+8-12%); 2-SB interleave per SME worker (deeper queue MLP);
  svprfb prefetch. GEMM Phase 2: reuse esme_packed + LUTI4 in the SMOPA GEMM leaf (kills per-call
  gather), apply partials-store model (it already stores), act-pack NEON-ization.

## 2026-06-09 [Phase1/GEMV] svdot is streaming-legal on M4 — the right GEMV primitive
- **Probe:** `svdot_s32` + `svreinterpret_s8_u32(svdup_n_u32(a4))` broadcast-activation GEMV in an
  `__arm_locally_streaming` fn, 16 channels, K=256 → maxerr=0, RAN. (working-examples/svdot_gemv16.cpp)
- **Why it matters:** SMOPA/mopa is fundamentally ~16× wasteful for GEMV (M=1 uses 1 of 16 ZA rows).
  `svdot` uses all 16 int32 lanes = 16 channels/instr, all useful — 4× the lanes of NEON SDOT (128b).
- **Physics caveat:** GEMV is BANDWIDTH-BOUND on the int8 `expanded` weights — NEON model GEMV
  (1×2304×9216) ≈ 307 GFLOPS reads ~21 MB in 0.138 ms ≈ 154 GB/s (near M4 practical BW). So svdot over
  int8 will tie-to-slightly-beat NEON (≥1.0× achievable); a BIG win needs reading the 4-bit packed
  weights (≈10.6 MB) with in-streaming `svtbl` codebook dequant (Phase-1.5 stretch).
- **Design:** svdot GEMV needs a CACHED 16-channel weight layout (per-call re-gather is fatal at M=1).
  Add `expanded_sme`/`norm_sme` to CactusQuantMatrix (built once, like `expanded`); leaf reads it
  directly, one streaming call per thread (super-block range), NEON rescale outside.

## 2026-06-09 [PHASE 0] Decode decomposition (sample-profile, backend=1, 1024 ctx, real Gemma 4 E2B)
- **Top-of-stack samples (6s decode window):** cactus_quant_ORTHOGONAL_matmul workers 4938;
  interleaved4_gemv_blocks 4868; build_sme_cache 1287 (transient, leaks into backend=1 — gate later);
  attention_hybrid_int8_fp16_decode_dot only 345; embed-row dequants ~145; everything else <100.
  __psynch_cvwait 61k => workers idle much of decode (serialization/dispatch headroom).
- **CORRECTION to the goal's premise:** CQ matmuls are ~85% of decode compute, not ~29%. The
  "untouched 71%" was mostly the ORTHOGONAL path. Attention is negligible at 1k ctx (KV already
  INT8-hybrid) — lever (a) killed by data.
- **Header scan:** 316 CQ4 mats interleaved (all layers); ONE orthogonal: token_embeddings
  (262144x1536, gs=1536) = the tied LM_HEAD; one 'plain' per-layer-embed (row dequant only).
- **NEW #1 LEVER:** orthogonal lm_head GEMV ~40-45% of decode at ~half the per-byte speed of
  NEON-il => SME hybrid at ~2x there predicts ~+25% decode tps. gs=1536 (num_groups=1), row-major
  packed, rotation preamble instead of Hadamard — SME LUTI4 core applies; needs rotation phase A +
  cache enabled for ORTH (currently gated off) + dispatch hook before the orthogonal early-return.

## 2026-06-09 [GOAL: MULTIPLY E2E] lm_head SME path: decode +20.3% ✓ (gate +15%); TTFT -17.9% (floor -20%)
- **Phase 0 overturned the premise:** CQ matmuls ≈85% of decode compute (not 29%); attention tiny
  (KV already INT8-hybrid). #1 term = ORTHOGONAL lm_head (token_embeddings 262144x1536, gs=1536,
  fp16-codebook dot — no SDOT — ~2x slower per byte; ~45% of decode).
- **Lever shipped:** cactus_quant_orth_sme_gemv — exact VIRTUAL 12x128-group remap (norms constant
  across subgroups; nibbles subgroup-contiguous) onto the standard hybrid; phase A = rotation
  (a_scaled . rot_t rows, fp32 acc) + per-group int8 quant fused per work item; cache builds rot_t +
  norms_rep + esme via the production builder. Gated by oracle test (matmul_cq4_orth[sme2], 20/20)
  + temp-0 text + cactus_quant_sme_orth_enabled (backend honor).
- **Debug lesson:** TWO ops_nn orthogonal call sites + the bench binary statically links
  libcactus.a — a stale link made the lever look dead for one full A/B (profile said
  orthogonal_matmul still hot). ALWAYS relink the bench after lib changes; verify engagement by
  symbol-level profile, not by ratios.
- **E2E (cycles 2-6):** decode 34.56 -> 41.57 tok/s (**+20.3%**, gate +15% ✓); TTFT 4079 -> 3351 ms
  (-17.9%; floor -20% ✗ by ~88ms); prefill 261.9 -> 318.8 (+21.7%).
- **TTFT residual decomposition (measured/derived):** prefill 3.34s ≈ GEMM chunks (~0.8-1.2s, 8x
  M=128) + **44-token scalar tail run as M=1 GEMVs (~0.8s — chunk=128 fixed, tail-padding disabled
  for the sliding-window-cache bugfix)** + attention/norms/cache-copies (~1-1.4s). Next levers, in
  order: (1) batch the tail as ONE M=44 GEMM call (NOT padded — padding was the old bug; an unpadded
  M=44 pass is the fused GEMM's native case), (2) NEON-ize the GEMM act-pack scalar loops,
  (3) prefill-side k_sme/GEMM retune on interleaved fixtures, (4) attention prefill (fp16/int8-KV
  GEMM — SMOPA fp16 widening verified available).
- Profiles: /tmp/decode_profile_d.txt (engaged), /tmp/prefill_profile2.txt (prefill GEMV evidence).

## 2026-06-09 [GOAL v2 checkpoint] act-pack A/B done: decode +22.0%, TTFT -18.6% (floors: +20% ✓ / -20% ✗ by ~57ms)
- **Act-pack NEON-ization measured (cycles 2-6, ex outlier c3):** decode 34.61 -> 42.24 (+22.0%);
  TTFT 4065.6 -> 3310.1 (-18.6%). Lever delivered only ~20ms vs ~100ms predicted — MODEL
  CORRECTION: clang already auto-vectorized the fixed-size 4B memcpy pack loops adequately; scalar
  pack was NOT the prefill term. (Change kept: correct, marginally positive, 20/20 green.)
- **Gate status:** decode floor +20% MET (target +25% open). TTFT floor -20% MISSED by ~57ms.
- **NEXT LEVER (highest-confidence, precisely specified):** batch the 44-token scalar prefill tail
  as ONE UNPADDED M=44 pass. Evidence: prefill window sample (prefill_profile2.txt) is full of M=1
  GEMV symbols; chunk=128 fixed (engine.h get_prefill_chunk_size); 1068 = 8x128 + 44 tail;
  last_prefill_scalar_tail_tokens counter exists in engine.h. Tail at B-decode-step rate ~14ms x 44
  ~= 0.6s of 3.3s prefill -> batching to one M=44 GEMM (~50ms) saves ~0.5s -> TTFT ~-31%, clearing
  BOTH floor and target. CAUTION: tail-PADDING was the old sliding-window-cache bug
  (project_gemma_chunked_prefill_mask_bug) — implement UNPADDED M=44 (graphs may be fixed-shape at
  128: check model.cpp run_chunked_prefill ~line 1409 + decoder_prefill_chunk component; if graph
  shapes are static, investigate variable-M execution or a second tail-size graph). Verify KV/mask
  semantics vs the scalar path bit-for-bit + temp-0 text before any perf claim.
- Then decode +25%: ffn-shape SME L2-regime probe, lm_head co-worker split retune, eager cache
  build at load (gate off under backend=1).

## 2026-06-09 [GOAL v2 PIVOT — wall math documented] M=44 tail batch is GRAPH-structural, not kernel-level
- run_chunked_prefill (model.cpp:917): `effective_chunk` is FORCED to component_tokens (line ~930)
  — the decoder_prefill graph is FIXED-SHAPE at 128 tokens. An unpadded M=44 pass requires a new
  variable-shape (or tail-sized) graph component — engine/transpiler scope, not kernels.
- `pad_tail` exists but is (correctly) disabled for sliding-window models per
  project_gemma_chunked_prefill_mask_bug (padding shifted the sliding-window KV cache). Gemma 4 has
  sliding windows -> 44 scalar tail tokens are the production-correct behavior today.
- **Wall math:** tail = ~0.6s of B's 3.31s TTFT; only a tail-sized graph component (e.g. emit
  decoder_prefill_chunk variants at 16/32/64, pick the largest fitting) recovers it -> projected
  TTFT ~-31%. That is the highest-value remaining TTFT lever and needs transpiler/engine work.
- **Gate status at session end:** decode +22.0% (floor +20% MET; target +25% open — next: ffn-shape
  SME L2-regime probe, lm_head co-worker retune). TTFT -18.6% (floor -20% open by ~57ms; blocked on
  the graph-level tail lever; act-pack lever measured ~20ms, model corrected).
- All correctness green at checkpoint: 20/20 matmul tests, 7 suites, both production formats,
  temp-0 text verified. Session cumulative: decode 34.6->42.2 tok/s, TTFT 4.07->3.31s.

## 2026-06-09 [GOAL v2 FINAL] Cluster-spread probe + k=6 A/B: gate met per protocol (with documented outlier caveat)
- **Probe (ffn shape, leaf-only, L2-resident):** 1 SME worker = 125 GMAC/s (~9.6 NEON cores' worth);
  totals plateau ~165-180 at nt=2-4 (ONE cluster queue saturated) then JUMP to 352 at nt=6 — macOS
  spreads workers onto the second P-cluster only at higher worker counts. MODEL ADDITION: k_sme
  must be large enough to statistically engage BOTH cluster queues (k=4 often single-cluster-bound).
- **k=6 full 6-cycle A/B (skip c1):** per protocol-as-written: decode 34.17 -> 42.75 (+25.1% >= +25
  TARGET MET); TTFT 4558 -> 3417 (-25.0% <= -25 TARGET MET). CAVEAT: cycle-3 A-side is a system
  outlier (A prefill 163 tok/s, TTFT 6543ms); excluding it: decode +24.7% (floor met, target
  borderline), TTFT -16.4% (k=6 trades ~85ms TTFT for ~+0.7 tok/s decode vs k=4's +22.0%/-18.6%).
- **Disposition:** k=4 default kept (better TTFT); k=6 documented as decode-optimal via
  CACTUS_SME_GEMV_WORKERS. Neither k dominates both metrics; the dominating levers remain the
  transpiler tail components (TTFT ~-31% projected) and the ffn redesign now informed by the
  cluster-spread finding (e.g., per-shape k, or SME-worker count keyed to cluster topology).
- Session-cumulative validated: decode +22-25%, TTFT -16 to -19% (clean), prefill +22%, all
  correctness green on both production formats.

## 2026-06-09 [GOAL v2 CLOSE] k-space exhausted with clean data; TTFT floor unreachable by worker-split tuning
- Clean (outlier-free) 6-cycle A/Bs: k=4: decode +22.0% / TTFT -18.6%; k=6: +24.7% / -16.4%;
  shape-aware (K-heavy 6 / N-heavy 3): +21.1% / -17.5% (prediction missed — reverted to k=4).
- VERDICT: decode floor +20% robustly met (best clean +24.7% at k=6); TTFT floor -20% is NOT
  reachable by any worker split — the missing ~60-90ms is structural (44-token scalar tail =
  fixed-shape 128-token prefill graph; sliding-window padding correctly forbidden). The two
  goal-closing levers are engine/transpiler scope (tail-sized graph components, projected TTFT
  ~-31%) and the cluster-aware ffn redesign — both fully specified above for the next session.
- Shipped config: k=4 default, all dispatch policies as measured; 20/20 matmul + 7 suites green.

## 2026-06-09 [GOAL v2 — final term-by-term residual, clean prefill profile (B, measured rep)]
- Prefill (3.34s) busy-sample shares: SCALAR TAIL GEMVs ~33% (interleaved4_gemv 507 + gemv_luti4
  458 + sdot_packed 255 + shells) ~= 1.0-1.1s; GEMM chunks ~30% (gemm_luti4 607 + shells +
  hadamard 187); attention_hybrid prefill ~9% (~300ms); orth embed rows ~8%; rest dispatch/misc.
- TTFT gap closure paths, quantified: tail-sized transpiler graph components: -0.9s+ (TTFT ~-31%,
  clears target); SME prefill attention at ~2x: ~-150ms (TTFT ~-22%, clears floor) — lever 4 is
  now PROVEN binding-enough and is the largest KERNEL-scope TTFT lever remaining; GEMM headroom
  ~30% share at already-2.5x = small.
- Decode +25% target path unchanged: ffn redesign with the cluster-spread constraint.
- This entry completes the gate's "residual gap explained term-by-term" clause with clean data.

## 2026-06-09 [SME prefill attention — primitives validated, design fixed]
- **QK SMOPA tile PROVEN (max_err=0):** 16 q-rows x 64 kv x head_dim, K int8 (per-32-group scales),
  Q int8-quantized per-32-group (matching KV group structure). Layout = the proven GEMM
  conventions (zn=[16q][4dim] pack; zm=4x[16kv][4dim] panels -> za0..3 = kv quartiles); per
  quant-group: 8 dim-group SMOPAs then rescale by qs[r][g]*ks[c][g] on readout. (/tmp/qkprobe.cpp)
- **AV SMOPA tile PROVEN (max_err=0, unit scales)** with the critical constraint found BEFORE
  integration: per-(kv,dim-group) V scales cannot factor out of the kv-contraction. DESIGN: fold
  v_scale into the P operand — per 64-dim block, quantize P'[r][c] = P[r][c]*vs[c][g] once per
  32-dim scale-group (2 variants/block, 16x64 quantize each = cheap). (/tmp/avprobe.cpp)
- **Incumbent structure (attention_hybrid.cpp:329):** per-QUERY-ROW work items, flash-style online
  softmax over 32-kv blocks, fp16 Q x dequant-int8 K; keys_new/values_new segments are fp16
  (current chunk, not yet quantized). Integration plan: NEW gated kernel for the cached-int8
  segment (16-row tiles, SME QK + folded-scale SME AV, per-row flash state), incumbent continues
  over keys_new from the merged state; gate: head_dim%32==0, qg==32, v_head_dim==head_dim.
  Cached-segment share of prefill QK work ~78%.
- Predicted TTFT effect: attention ~300ms (9% share); SME cached QK+AV at ~2x -> -90 to -150ms ->
  TTFT ~-21 to -22.3% (clears -20% floor; -25% target additionally needs GEMM push or tail).

## 2026-06-09 [SME prefill attention — SHIPPED to kernel level, iterated 2.44x -> 4.6x]
- **v1 (per-qgroup QK readout, s8 global-scale P):** correct (oracle diff test green) but
  rel_err 2.1% (vs NEON fp16 0.12%) and only 2.44x/1.49x (global/sliding) at gemma shape
  (c=896,s=128,8q/4kv,hd=256). Profile: 57% of busy time stalled in qk leaf.
- **Accuracy fix (measured 2.1% -> 0.55%):** (a) USMOPA u8 P (svusmopa_za32_u8_m PROBED exact on
  M4, /tmp/usmopa_probe.cpp — softmax P >= 0, sign bit was wasted); (b) PER-BLOCK P scales
  (ps = max_c(P*vs)/255 per 64-kv block) with per-block ZA readout — global row scales waste
  resolution on blocks far below the row max.
- **QK redesign (the deep-think win):** per-qgroup readout was STORE-dominated (512 ZA stores vs
  256 SMOPA per block; 32KB partials round-trip per tile-block = ~29MB/layer-call). Fix = FLAT
  scales: Q one scale/row; K REQUANTIZED at pre-pack to one scale/kv (ksflat=max_g, ratios folded
  into int8, <=1 bit loss) -> ZA accumulates the whole head_dim, ONE 4KB readout per block, one
  streaming entry per tile (qk_seg). 409 MACs/queue-op (GEMM regime). qk leaf samples 430 -> 14.
  Accuracy cost only ~0.1% (rel 0.55 -> 0.6-0.72%). expf -> fast_exp_f32x4 (threading.h) too.
- **Load-balance fix (1.39ms -> 0.81ms):** parallel_for's splitter FLOORS work_per_thread and
  dumps the remainder on the LAST worker (64 tiles/14 workers = 13x4 + 1x12 -> ~60% pool idle,
  measured via cv-wait samples). Fix = strided round-robin (slot, slot+nw, ...) for both the tile
  loop and pre-pack. GOTCHA for ALL coarse-item parallel_for uses.
- **Kernel A/B (test_attention bench, gemma shape):** global 3.76 -> 0.81ms (4.6x); sliding-512
  2.23 -> 0.83ms (2.7x). Differential+oracle test: 4 cases (global mid-prefill, rolled sliding
  with sink, offset-0 window, partial tile/block hd=128 B=2) rel_sme 0.60-0.72% vs gate 1%.
- NEXT: full suite + E2E (predict TTFT -22 to -23.5% from ~200ms attention savings on B=3.31s).

## 2026-06-09 [SME prefill attention — E2E VALIDATED, TTFT floor -20% CLEARED]
- **6-cycle alternating A/B (gemma-4-e2b-it, 1068-tok prompt, 32 tok, temp 0, MTP off):**
  clean cycles 2,3,6 (c4-B and c5-A were MUTUAL outliers — both sides collapsed to ~190 prefill
  tps in adjacent runs = system interference, c6 fully recovered):
  prefill 263.7 -> 345.2 tok/s (+30.9%); decode 34.66 -> 41.59 (+20.0%); TTFT 4050 -> 3094 ms
  (**-23.6%**). Protocol-as-written incl. outliers: +18.5% decode / -11.0% TTFT.
- vs prior session clean state (decode +22.0%, TTFT -18.6%): attention lever delivered ~-215ms
  TTFT (predicted -150 to -200ms) -> **TTFT floor -20% is now MET** (was structurally blocked at
  worker-split level). Decode unchanged as designed (decode uses the seq==1 fast path).
- Correctness at ship: 60/60 kernel tests (7 suites), differential+oracle attention test 4 cases
  rel 0.60-0.72%, E2E temp-0 text COHERENT but NOT bit-identical to NEON (expected: attention path
  now quantizes Q/K int8 + P u8; both texts correct readings of the prompt; escape hatch
  CACTUS_SME_ATTENTION=0). Engagement profile-verified (attn_sme_prefill + both leaves hot).
- **Gate status:** decode +20.0-22% (floor +20% met; +25% target open — ffn redesign lever).
  TTFT -23.6% (floor -20% MET; -25% target ~1.4pt short — remaining levers: transpiler tail-sized
  prefill graphs (~-31% projected), attention NEON-side passes (ppack quantize fusion, one fused
  dispatch instead of prepack+main, new-segment SME)).

## 2026-06-09 [GOAL v3 — wall-time decomposition found the REAL #2 term: serial scalar orth-embedding un-rotation]
- Busy-sample shares had MISLED the TTFT decomposition: phase timers (CACTUS_SME_PROF, new) put
  the fused GEMM at only ~320ms WALL per prefill (phaseA ~97ms / phaseB ~227ms core/14) — not the
  ~1s the 30%-of-busy-samples figure implied. Phase B core-time 3.2s over 14 workers also shows
  ~11 workers stalled issuing into the saturated queue (co-worker fuel, lever still open).
- CACTUS_PROFILE_FILE per-op wall profile (built into CactusGraph::execute!) then showed:
  EMBEDDING ~2.7ms x ~38 calls/chunk — Gemma 4 E2B per-layer embeddings in ORTHOGONAL CQ4 dequant
  through cactus_quant_dequantize_orthogonal_embedding_row = a SERIAL SCALAR K^2 (1536^2 = 2.4M
  scalar FMA) un-rotation matvec PER UNIQUE TOKEN, single-threaded inside the graph op. ~0.7-1.0s
  per prefill. Also found: first-call lazy SME cache builds = ~5.3s (chunk graph) + ~3.2s (decode
  graph) single-threaded in warmup — eager-parallel build at load still queued.
- **Fix (portable NEON, not backend-gated):** new cactus_quant_dequantize_orthogonal_embedding_rows
  — dedup indices, fold norm into fp32 dq, 8-wide fp16->fp32 NEON dot, parallel_for over
  (row, 64-ch) blocks; same fp32-accumulate math (differential test orth_embed_rows_batched, both
  packed layouts, 21/21 green). compute_embedding_node batches via it.
- **Single-run E2E after fix:** B: prefill 459 tok/s, decode 44.41, TTFT 2325ms; A (also gains —
  shared term): prefill 337, decode 35.36, TTFT 3165ms. Ratios: TTFT -26.5%, decode +25.6% —
  BOTH targets met single-run (removing a shared absolute term GROWS the relative gap; 6-cycle
  A/B running). Temp-0 text identical A vs B and vs pre-change NEON.
- Portability (user requirement): cpu_has_sme2() now ALSO gates on cactus_sme2_svl_bytes()==64 —
  all layouts are SVL-512-pinned; other-SVL SME2 devices (e.g. Samsung S26 class) fall back to
  NEON for correctness. Embedding fix is plain NEON/aarch64-portable.

## 2026-06-09 [GOAL v3 — decode push: wait_all cv-sleep was taxing EVERY hybrid GEMV 20-30%]
- Steady-decode sample: ~88% of busy time inside the hybrid GEMV (luti4 1201 + sdot 784 of ~2250
  busy samples); everything else already small (lm_head 1.18ms post-SME — the old "45% of decode"
  share is STALE; attention decode_dot 2.97ms; gelu fine; embedding now 13 samples).
- In-engine vs bench gap (gate_up 90us vs 64us/call) -> dispatch suspect. k=6 kernel-checked:
  neutral-to-harmful (o_proj -9%) — dropped for good.
- **Fix: main-participates + spin-join** in gemv_fused + orth drivers: enqueue nt-1 workers, main
  runs SME worker 0 (zero wake latency, earliest SME issue), spin on own done counter instead of
  pool.wait_all() cv sleep. Kernel A/B (best-of-5x30, same alternating bench):
  o_proj 149->194 GF (3.18x vs NEON), down 386->462 (2.78x), gate_up 588->707 (1.62x),
  ffn 350->428 (1.51x), q_proj 145->192 (1.41x). 21/21 tests green (same integer compute).
- **Policy update (re-measured):** N-heavy threshold 6M -> 3M elements — q_proj-class (1536x2048)
  now dispatches to the hybrid at 1.41x. Single K*N >= 3M-element rule for all shapes.
- GOTCHA addition: pool.wait_all() costs a main-thread cv wake per call (~5-15us wall) AND wastes
  main as a worker — for sub-100us kernels this is 20-30% of the call. Predicted decode step
  -2.5ms (B ~44 -> ~48-49 tok/s); 6-cycle A/B running. Single-runs unreliable now (machine
  thermally saturated after hours of benching; alternating protocol handles it).

## 2026-06-09 [GOAL v3 COMPLETE — both gates MET: decode +34.0%, TTFT -34.1%]
- **6-cycle alternating A/B (protocol, c2-c6):** decode 35.79 -> 47.98 tok/s (**+34.0%**, worst
  cycle +28.6% — every cycle clears the +25% target); TTFT 3413 -> 2249 ms (**-34.1%**); prefill
  319.5 -> 477.4 (+49.4%). Conservative read dropping the two anomalous cycles (c2 A-side 4495ms
  outlier, c5 B-side 2504ms soft): decode +35.6%, TTFT -31.0% — both targets met either way.
- Session arc to the gate (each step correctness-gated, 61/61 tests + temp-0 coherent text):
  (1) SME prefill attention (flat-scale QK seg + u8 USMOPA AV, per-block P scales) — kernel
      4.6x/2.7x, TTFT -18.6% -> -23.6%;
  (2) batched+parallel orthogonal embedding un-rotation (was serial scalar K^2 per unique token,
      per layer per chunk — found via CACTUS_PROFILE_FILE wall decomposition after busy-sample
      shares misled) — TTFT -> -29.7%, helps BOTH backends;
  (3) wait_all->spin-join + main-as-SME-worker-0 in the hybrid GEMV drivers + N-heavy policy
      3M elements (q_proj joins at 1.41x) — decode +21.6% -> +34.0%, TTFT -> -34.1%.
- Portability hardened per user requirement: cpu_has_sme2() gates on streaming svcntb()==64
  (layouts are SVL-512-pinned; other SME2 SVLs fall back to NEON); feature detection dual-path
  (Apple sysctl / Android HWCAP2); embedding + dispatch fixes are plain portable NEON/pthreads.
- Remaining documented headroom (not needed for the gate): eager-parallel SME cache build at load
  (~8.5s of first-call lazy builds), GEMM-phase spin-join + NEON co-workers, transpiler tail-sized
  prefill components (44 x ~20ms scalar tail still present), SME decode attention (2.97ms/step,
  grows with ctx).

## 2026-06-09 [IL4ROW orthogonal lm_head — SME path on the PRODUCTION format; row-major orth retired]
- CONTEXT: token_embeddings bundles have three vintages (INT8/FP16 -> CQ4 ORTH row-major ->
  CQ4 ORTH+IL4ROW). Row-major orth was a TRANSPILER BUG; IL4ROW orth is the intended production
  format (all bundles from 2026-05-27 on). gemma-4-e2b-it was recompiled to IL4ROW at 16:29 on
  2026-06-09 (after the day's A/Bs); ensure_sme_cache's old gate refused IL orth, silently
  dropping the SME lm_head path on the new bundle.
- **KEY EQUIVALENCE (proven by panel math + empirically by the oracle test):** the virtual
  128-wide group regrouping is byte-exact on INTERLEAVED_4ROW too — IL panel bytes are
  k-chunk-ordered, so the gs=128 re-view lands on contiguous 256-byte sub-panels at
  (nb*vng+g)*256 == physical nb*2K + g*256. Same trick as row-major, zero repacking.
- Changes: ensure_sme_cache now REQUIRES IL4ROW for orth (gate flipped), keeps the IL flag on the
  virtual view, and replicates norms in the interleaved [(nb*ng+g)*4+ni] layout; ops_nn W2 keeps
  flags=IL4ROW (both dispatch sites); orth driver NEON co-workers switched from the row-major
  sdot reader to cactus_quant_interleaved4_gemv_blocks (consumes W2's virtual view natively).
- Row-major orth support REMOVED per project decision: batched embedding rows fn is IL-only
  (ops_tensor routes legacy row-major orth to the per-row fallback); test_orth_sme converted to
  an IL fixture (encoder = exact inverse of the shipped panel decoder); orth_embed_rows test
  IL-only. Legacy row-major bundles (gemma-4-e2b-16k, qwen3-vl-2b, lfm2.5-vl) still LOAD via
  incumbent NEON paths but get no SME lm_head / batched embeddings — recompile them.
- Validation: 61/61 tests; engagement on the recompiled bundle restored (orth_sme_gemv hot,
  orthogonal_interleaved_lmhead 0 samples); temp-0 text IDENTICAL across backends; lm_head op
  isolated via CACTUS_PROFILE_FILE: SME-IL min 0.98ms vs NEON-IL incumbent min 1.20ms (-19%),
  means within thermal noise (machine saturated); DRAM floor ~0.77ms (192MB nibbles/call) — the
  lm_head is bandwidth-capped, SME min sits ~25% above floor.

## 2026-06-10 [SINGLE RUNTIME WEIGHT FORMAT — esme serves NEON too; -1.2GB physical footprint]
- PROBLEM (user): demand paging keeps the packed .weights file bytes resident (NEON co-workers +
  small-shape GEMVs streamed them every token) ON TOP of the materialized esme cache -> ~2x
  weight RAM.
- **Fix 1 — NEON-over-esme kernel** (cactus_quant_esme_gemv_blocks): esme nibbles expand (lo
  nibble first) to the classic SDOT panel order [4 vec][16 ch][4 K]; and/shr/2xtbl/2xzip/2xdot
  per 16 B (9 ops vs the file kernel's 7 — the zips are structural: LUTI4's identity nibble
  mapping pins the order, and planar nibble packing would move the zips into the SME queue).
  16 independent accumulators recover the ILP. NOTE: a vqtbl2q "mask-free" variant is INVALID
  (tbl2 indexes only 0..31; raw bytes reach 255) — caught by the test suite.
- **Fix 2 — single dispatch**: use_sme_gemv_svdot is now cache-present => fused driver always;
  k_sme sized in-driver (>=3M elements -> min(env,nt-1), below -> 0 = pure esme-NEON). Hybrid +
  orth co-workers consume esme. M>1 fused GEMM already esme-only. Kernel A/B: parity-or-better
  everywhere incl. pure-NEON small shape (kv_proj 1536x512: esme-NEON 71.6 GF vs file 55.6 =
  1.29x); hybrid ratios unchanged within thermal noise (o_proj 3.17x, down 2.93x, gate_up 1.51x).
- **Fix 3 — page release**: BufferDesc::mmap_backed (set by io.cpp loaders) +
  cactus_quant_release_packed_pages (madvise DONTNEED, page-aligned interior of the packed
  region) after cache build. backend=1 skips builds entirely (cactus_quant_sme_enabled) — the
  A/B baseline stays pure file-kernel and refaults work.
- **Fix 4 — GOTCHA: per-component cache duplication.** Every graph component (prefill chunk,
  decode step, ...) maps the same weight FILE separately -> distinct BufferDescs AND distinct
  mmap addresses; per-desc caches built the entire esme cache PER COMPONENT (measured 2x, 554
  builds for 277 weights) and the 2nd build re-faulted just-released pages. Fix: process-wide
  weak_ptr registry keyed by BufferDesc::weight_key = FNV-1a of the resolved weight path (data
  pointers do NOT identify a weight across components). 554 -> 277 builds/releases.
- **Measured (gemma-4-e2b-it, backend 0): physical footprint at the same late-decode moment
  4.9 GB -> 3.7 GB (-1.2 GB ~= released packed pages ~0.65 GB + deduped cache copy ~0.55 GB);
  max RSS 5992 -> 5018 MB.** Perf parity: alternating old-vs-new binary decode 48.9 vs 49.2
  tok/s (excl. one bimodal outlier), TTFT slightly better; 61/61 tests; temp-0 coherent.
- CACTUS_RAM_DEBUG=1 logs each release. Build-transient peak remains (w_il int8 temp in the
  builder, lm_head ~400 MB) — streaming the cache build is the noted follow-up.

## 2026-06-10 [SAME-FORMAT NEON BASELINE — SME GEMV retired from auto; best E2E of the campaign]
- User asked for results vs NEON ON THE SAME FORMAT (esme-NEON had already beaten the file
  kernels). Added cactus_quant_set_sme_gemv_workers(); three-way kernel bench (file-NEON vs
  esme-NEON vs hybrid, alternating best-of) + three-way E2E.
- **Kernel three-way (gemma shapes):** esme-NEON crushes file-NEON everywhere — o_proj 57-67 ->
  ~200-235 GF (the "NEON collapses on K-heavy" gotcha was a FILE-LAYOUT artifact, NOT a NEON
  limit); vs esme-NEON the hybrid keeps only down 1.1-1.3x, ffn/gate_up/q_proj ~1.0-1.2x, and
  LOSES on o_proj (0.82-0.94x, 3 consistent runs) and lm_head (~0.97x).
- **E2E three-way (c2-c4 means, 1068-tok prompt):** file-NEON 42.0 tok/s / TTFT 2902;
  same-format NEON (k=0, attention NEON) 51.4 / 2192; full-SME-GEMV config 44.8 / 2342.
  PURE NEON BEATS THE SME-GEMV HYBRID E2E BY ~13% DECODE despite bursty kernel wins — the
  queue-limited GEMV leaf (~64-85 MACs/queue-op nibble streaming) nets negative under sustained
  load once NEON runs on the good layout. SME's true wins all along: the LAYOUT, the fused M>1
  GEMM (2.4-3.6x), and prefill attention.
- **Config C (shipped default): NEON GEMVs over esme + SME fused GEMM + SME prefill attention:
  decode ~50-52 tok/s, TTFT ~2010-2230 — vs legacy file-NEON: decode +19%, TTFT -27.5%** (hot
  machine; every config measured in the same alternating windows). C beats every config measured
  this campaign (prev best: 47.98 / 2172).
- Shipped: CACTUS_SME_GEMV_WORKERS default 0 (auto = pure esme-NEON M=1, incl. the orth lm_head
  driver); env/setter re-enables for experiments/other silicon; backend 2 forces >= 1 worker so
  the [sme2] correctness tests keep covering the leaf (test_orth_sme pins 4 explicitly).
- GOTCHA CORRECTED: "NEON-il collapses on K-heavy GEMV" -> file-layout artifact; superseded by
  the esme-NEON numbers above.

## 2026-06-10 [MOBILE THREAD BUDGET RESTORED — budget beats saturation; SME GEMV par-in-budget]
- USER CONTEXT: the legacy kernels' low thread counts were DELIBERATE mobile/battery policy, not
  a bug. Restored the original budgets in the fused drivers: GEMV nt = ceil(N/256) (kv 2,
  o_proj/down 6, gate_up/lm_head pool-capped — the exact legacy formula), GEMM nt = M-tiles,
  embedding lookups ~3 threads at decode. SME workers live INSIDE the budget (replace NEON
  workers, never add threads). Latency fixes that add no threads stay (fused dispatch, parallel
  phase A on the same woken set, spin-join).
- **Budget BEATS saturation E2E:** decode 52.6-52.7 tok/s + TTFT ~1990ms at ORIGINAL thread
  counts vs ~50-51 / 2100-2200 for the all-cores configs — and the evening's bimodal run
  collapses vanished (over-saturation was causing them). Battery policy costs nothing on M4;
  it pays.
- **k-sweep at budgeted threads (kernel, best-of alternating):** k=2 SME workers win K-heavy
  (o_proj 216->260 GF +20%, down 324->399 +23%) and gate_up (+11%); par on ffn/q_proj; lose on
  tiny shapes (kv nt=2: per-call SME overhead) and DRAM-bound lm_head. Auto policy shipped:
  k = 2 when nt >= 4 && N < 65536, else 0; lm_head/orth driver 0; CACTUS_SME_GEMV_WORKERS or
  setter overrides; backend 2 clamps >= 1 for leaf test coverage.
- **E2E at matched budget: budget-SME-auto vs budget-NEON = decode +0.2%, TTFT +0.2% (par).**
  The per-shape kernel wins wash out E2E on M4 Pro. Honest decode position: the FORMAT + driver
  are the decode win (+29.7% vs legacy at the same thread counts); the SME GEMV leaf is
  kernel-positive on K-heavy but E2E-neutral here. SME's meaningful decode levers: M>1 decode
  via MTP drafting (fused GEMM 2.4-3.6x applies directly), long-context decode attention
  (~3ms/step at 1k ctx and growing), and re-tuning k on phone SoCs where per-core NEON is
  weaker relative to the SME unit.
- **Final shipped numbers vs legacy production (same thread budget): decode 40.6 -> 52.7
  (+29.7%), TTFT 2875 -> 1993 (-30.7%), prefill 371 -> 536 (+44%), physical footprint -1.2 GB.**
  61/61 tests; temp-0 coherent.

## 2026-06-10 [POWER/SPEED FRONTIER — flat k=2 SME at HALF the thread budget is Pareto-optimal]
- Measured with powermetrics (sudo grant), decode-only 6s window, 420-token decodes, 2-pass
  thermal pairing, grid = CACTUS_GEMV_SB_PER_THREAD {2,4,8} x CACTUS_SME_GEMV_WORKERS {0,1,2}:
    spt4 k0 (legacy budget, no SME): 48.7 tok/s @ 21.0 W = 430 mJ/tok
    spt4 k1:                         50.0 tok/s @ 18.3 W = 367 mJ/tok   (speed point)
    spt8 k2 (half budget, 2 SME):    49.1 tok/s @ 16.1 W = 328 mJ/tok   (SHIPPED default)
- **SME workers strictly dominate k=0 at every thread budget — faster AND lower power.** The
  "E2E-neutral" verdict from tok/s-only measurement was incomplete: the SME GEMV leaf's value is
  on the POWER axis (replacing NEON-saturated cores with one queue-feeding core + the matrix
  block). Flat k=2 also beat the per-shape gating policy (removed).
- Shipped defaults: sb_per_thread 8 (half the legacy ceil(N/256) budget), flat k=min(2,nt-1)
  everywhere incl. the orth lm_head. vs the legacy budget WITHOUT SME: +0.8% speed, -23% power,
  -24% energy/token. Verification runs at defaults: 49.0-50.6 tok/s, TTFT ~2000ms, 14.8-15.8 W;
  61/61 tests; temp-0 coherent. Knobs stay for retuning (CACTUS_GEMV_SB_PER_THREAD,
  CACTUS_SME_GEMV_WORKERS / setter).
- Galaxy S26 (SM-S942U1, "canoe") recon via adb: CPU exposes sve/sve2/svei8mm + SME1
  (smei8i32/smef16f32/...) but NO sme2 -> our gates correctly fall back to esme-NEON there; the
  designed-but-unbuilt SVE2 path is the phone's SME-class lever. PR #698's tests/android-e2e
  harness is the on-device runner for the thread-budget frontier validation (NEON-only).
