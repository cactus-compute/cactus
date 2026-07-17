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
### GATE E — pending
### GATE F — pending
### GATE G — pending

## Compat note (for review)
New file flag FLAG_NO_ROTATION=1<<3: an OLD engine loading a NEW no-rotation .weights
file would misparse (assumes hadamard tail). Bundles ship with their engine, so accepted
here — flagging in case a format version bump is preferred.
