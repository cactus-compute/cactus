# Bonsai 1-bit Lossless Repack — Worklog

Plan: lossless CQ1 repack of QAT-binary checkpoints (PrismML Bonsai-27B), no-rotation
flag through quantizer → serialization → loaders → kernels, gated by exact oracles.
Full plan: ~/.claude/plans/serialized-humming-sutton.md

## Gate log

### Kickoff (2026-07-16)
- Branch `bonsai-repack` off main (df4482ac).
- Commit 9d07a3bc: `_qwen35_rotary_embeddings` fix (qwen3_5 --low-memory-load capture crash).
- Deleted broken `weights/bonsai-27b-unpacked-cq1` (rotated 1-bit PTQ, word salad — repro of PTQ collapse).
- Kept: HF cache (54.7 GB), `weights/bonsai-27b-unpacked-cq4` (14 GB, E-gate comparator).

### GATE A — in progress
- Truncated 4-layer checkpoint built: `weights/bonsai-truncated-4l-src` (56 tensors, 8.13 GB,
  layer_types [lin,lin,lin,full]). CQ4 calibration bundle: `weights/bonsai-trunc4l-cq4` (2.1 GB,
  converts in ~4 min — fast-iteration loop confirmed).
- Oracle gotchas found: FFI response has NO token ids (text only) → metric switched to
  greedy text-prefix agreement; `auto_handoff: False` REQUIRED in options or hybrid mode
  silently answers from cloud; transformers 5.x apply_chat_template needs return_dict=False.
- Risk noted: decapitated-model outputs are high-entropy; if CQ4 text agreement is ~0,
  switch metric to cactus_score_window logprob deltas on fixed windows.
- **GATE A result**: CQ4 truncated bundle vs transformers = 0.0142 mean text-prefix
  agreement — tie-chaos as predicted, floor is weak. `cactus_score_window` is a STUB
  (model.cpp:4392 returns 0.0 — flag for Karen). Gates restructured: numerical exactness
  proven at kernel+graph level (Phase C tests); Gate D = smoke/determinism/≥floor on
  truncated; integration proof = Gate E exact-answer battery on full 27B (low-entropy
  answers are robust to fp16 tie noise — CQ4-27B demonstrated this).

### GATE B — in progress
- FLAG CORRECTION vs plan: file-side bit 3 is taken (qdq.py FLAG_INTERLEAVED for INT8);
  using FLAG_NO_ROTATION = 1<<5 file-side, 1u<<4 kernel-side.
- Interleaved layout EXCLUDED for repack tensors in v1: the cb_factor=1/127 int8 rescale
  makes norms non-fp16-exact. Linear LSB layout keeps bit-exactness; interleaved+pow2
  cb_factor is a later speed optimization.
- quantize_binary_repack + write path + qdq reader landed; synthetic round-trip test
  bit-exact (test_cq.py 10/10 green, incl. all-zero-group and reject-non-binary cases).
- **GATE B: PASS** — `gate_b.py weights/bonsai-trunc4l-cq1 weights/bonsai-truncated-4l-src
  --sample-27b`: 31/31 bundle CQ1 tensors dequantize bit-exactly vs source; 5/5 sampled
  full-27B tensors (incl. embed_tokens + lm_head 248320x5120) EXACT. Truncated cq1 bundle
  1.59 GB vs 2.1 GB cq4. Lesson: cli.py exception handler silently FP16-falls-back — a
  missing import cost one convert cycle; gate caught it via counts_by_precision.
### GATE C — PASS (2026-07-16)
- Flag plumbing: cq.py/qdq.py FLAG_NO_ROTATION=1<<5 → io.cpp parse_header + 3 loaders
  (null tail, kernel flag 1u<<4) → matmul.cpp bypass in shared activation transform +
  embedding-row dequant (new `flags` param, both callers updated).
- NOTE: file-flag bit 0 is FLAG_HAS_SCALES (test_nn.cpp) — bits 0-4 all taken; 1<<5 confirmed free.
- Kernel suite 8/8 incl. NEW `matmul_cq1_norot` (MSE < 1e-3 vs fp32 reference — 100x
  tighter than rotated tests; residual = activation int8/fp16 only, weights exact).
- Graph suite 15/15 incl. NEW "MatMul CQ (no rotation)" (file→loader→mmap→matmul vs
  direct kernel, 1e-3 elementwise).
- Python 82/82 (convert incl. repack round-trip + policy + transpile_weight_compat).
- Canary: gemma-4-e2b-it-cq4 on rebuilt engine → "391" correct. No shipped-format regression.

### GATE D — PASS (2026-07-16)
- Truncated CQ1-repack bundle (1.59 GB) runs in engine: 64 tokens/prompt, no crash/NaN,
  **bit-identical across two runs** (determinism), text-prefix agreement vs transformers
  0.0151 ≥ CQ4 floor 0.0142. Metric is tie-chaos-bound on the decapitated model (as
  documented under GATE A); exactness burden carried by GATE B + GATE C tests.
  Integration quality proof deferred to GATE E full-27B battery by design.
### GATE E — PASS (2026-07-16)
- Full 27B repack (`weights/bonsai-27b-repack-cq1`, 4.9 GB, 606 CQ1 + 2 CQ4 embed/head):
  **GSM8K slice 17/20 = 85%** (floor 70; Prism full-set reference 92.8) + factual 4/4
  ("391", "Canberra", "1969", "12"), greedy, auto_handoff off. Mean decode 3.7 tok/s
  (CQ1 linear layout; CQ4 interleaved gets 11.8 — pow2-cb_factor interleaved repack is
  the known speed follow-up). Battery wall time ~35 min on the M4 Pro.
- Prefill-chunk DESCOPED from this gate: not a capture bug — the qwen causal-lm builder
  only emits decoder+decoder_step for the whole qwen family (model_adapters.py
  `_build_qwen_causal_lm_component_specs`); chunked prefill for hybrid DeltaNet needs
  chunk-wise recurrent-state I/O — family-wide feature, affects CQ4 identically.
### GATE F — PASS (2026-07-16)
- Binary embeddings + lm_head land: full 27B bundle `weights/bonsai-27b-repack-cq1-f`
  = **3.88 GB, 608 CQ1 / 0 CQ4** — matches Prism's 3.9 GB deployed footprint with
  bit-identical weights. **GSM8K slice 18/20 = 90%**, factual 4/4, greedy local.
  Truncated Gate B re-passed (33/33 exact incl. embed/head); engine embedding-lookup +
  lm_head verified through the no-rotation dequant.
- CQ4 comparator battery (identical harness):

| bundle | size | GSM8K-20 | factual | mean decode |
|---|---|---|---|---|
| rotated CQ1 (pre-repack) | 4.9 GB | word salad | — | — |
| **repack CQ1, binary embeds** | **3.88 GB** | **90%** | 4/4 | 3.7 tok/s |
| repack CQ1, CQ4 embeds | 4.9 GB | 85% | 4/4 | 3.7 tok/s |
| CQ4 | 14 GB | 85% | 4/4 | 12.1 tok/s |

Lossless 1-bit repack delivers CQ4-class quality at 3.6x smaller. Speed gap is the
linear-layout CQ1 GEMV; follow-up = interleaved repack with power-of-two cb_factor.
### GATE G — PASS, compile half (2026-07-16)
- `cactus build --android` green: android/libcactus_engine.{so,a} built with the
  no-rotation CQ1 path. On-device run deferred (no device available) — when a phone
  with USB debugging appears: push `weights/bonsai-27b-repack-cq1-f` (3.88 GB) and run
  `cactus test --android` / the gate_e battery. Device needs ~6 GB app-available RAM.

## Follow-ups (ordered)
1. Interleaved repack layout with power-of-two cb_factor → close the 3.7 vs 12.1 tok/s
   gap while keeping bit-exactness.
2. decoder_prefill_chunk for the qwen family (chunked DeltaNet state I/O) — token-by-token
   prefill is the UX bottleneck for long prompts, affects CQ4 too.
3. Android on-device validation (first 27B on Android moment).
4. cactus_score_window is a stub — implement or remove from FFI (Karen).
5. Vision tower under --low-memory-load (weight-dependent constants baked at adapter init).
6. Ternary-Bonsai-27B: same repack at 2-bit (CQ2 codebook {-1,0,+1} — needs detection
   variant for 3-value groups; quality-oriented 5.9 GB operating point).

## Compat note (for review)
New file flag FLAG_NO_ROTATION=1<<3: an OLD engine loading a NEW no-rotation .weights
file would misparse (assumes hadamard tail). Bundles ship with their engine, so accepted
here — flagging in case a format version bump is preferred.
