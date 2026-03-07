# Gemma 3n Accuracy Investigation

## Summary (Current State: 2026-03-07)

Gemma 3n accuracy regressions were caused by multiple issues across config extraction, graph math, and debug instrumentation. The original conclusion ("attention is catastrophically wrong in Cactus FP16") is now **superseded**.

Current validated state:

1. **Config bug fixed**: missing `num_kv_shared_layers` / `layer_types` caused invalid shared-KV behavior.
2. **Axis bug mitigated in Gemma path**: `mean(-1)` / `variance(-1)` semantics can reduce globally; Gemma now uses explicit positive axis (`1`) for per-row stats.
3. **HF debug hook bug fixed**: patched attention path missed causal/sliding mask when `attention_mask=None`, creating false divergence at `Attention*V`.
4. **Attention math is now matching closely** in corrected prefill comparisons (cos usually `0.995-0.9999`).
5. **Remaining real divergence is not in attention sub-ops**; it starts in post-attention block composition (residual/laurel/ffn/AltUp-PLI path), first strongly visible at layer 4.

---

## Timeline of Confirmed Issues

### 1) Missing Config Parameters (Fixed)

The converter originally missed key text config fields:

| Parameter | Required | Broken default | Effect |
|---|---:|---:|---|
| `num_kv_shared_layers` | 10 | 0 | Shared-KV layers (20-29) used wrong K/V behavior |
| `layer_types` | sliding/full pattern | heuristic fallback | Wrong layer typing risk |
| `sliding_window` | 512 | implicit | Needed explicit consistency |

Fix was applied in `python/src/config_utils.py` and reconverted configs now include:

- `num_kv_shared_layers=10`
- `sliding_window=512`
- correct `layer_types` extraction path

---

### 2) `mean(-1)` / `variance(-1)` Axis Semantics (Identified, Mitigated in Gemma)

`CactusGraph::mean(input, -1)` and `variance(input, -1)` were resolving to global reduction instead of last-axis reduction. This broke `build_magnitude_normalize` (AltUp streams 1-3), and that error propagated into the transformer stack.

Current mitigation in Gemma:

- `model_gemma3n.cpp` now uses explicit positive axis `1` + reshape in row-wise stats paths (`build_magnitude_normalize`, `build_gaussian_topk`).
- Temporary graph-builder negative-axis remapping was later rolled back to keep graph semantics unchanged outside Gemma.

---

### 3) HF Patched Hook Missing Mask (Fixed)

Patched HF attention captures in debug scripts did not apply fallback causal/sliding mask when `attention_mask` was not passed explicitly. This produced false signatures where Q/K/V looked good but `Attention*V` and `o_proj` looked broken.

Fix: build and apply fallback causal/sliding mask in patched forward paths.

---

## Superseded Conclusion (Important)

These older conclusions are no longer valid:

- "L0 attention output cos ~0.79 in correct setup"
- "Attention is the primary error source in every layer"
- "Main remaining gap is inherent FP16-vs-bf16 inside attention softmax path"

Those observations came from combinations of:

- stale config,
- mean/variance bug,
- misaligned token comparisons,
- patched HF attention masking bug,
- and some shared-KV instrumentation mismatches.

---

## Latest Validated Results (2026-03-07)

All key comparisons below use:

- identical chat prompt tokens,
- prefill focus where stated,
- corrected HF mask behavior,
- and explicit BOS/token alignment checks.

### A) Direct output quality sanity check (HF vs Cactus)

Prompt checks showed:

- HF bf16: generally coherent on tested prompts.
- Cactus FP16: mostly coherent but still shows occasional artifacts (example: duplicated words / rare odd character insertion in identity prompt).

This confirms remaining degradation is real, but far smaller than the earlier "attention collapse" narrative.

---

### B) Attention sub-op comparisons (corrected)

#### Layer 0 prefill (corrected)

| Sub-op | Cosine |
|---|---:|
| Input LayerNorm | 0.999988 |
| Q Projection | 0.999992 |
| Q RMSNorm | 0.999991 |
| Q RoPE | 0.999989 |
| K Projection | 0.999996 |
| K RMSNorm | 0.999995 |
| K RoPE | 0.999993 |
| V Projection | 0.999992 |
| V RMSNorm | 0.999990 |
| Attention*V (pre o_proj) | 0.999974 |
| Output Projection | 0.999973 |

#### Layer 4 prefill (first full-attention layer)

| Sub-op | Cosine |
|---|---:|
| Input LayerNorm | 0.999935 |
| Q Projection | 0.999961 |
| Q RMSNorm | 0.999969 |
| Q RoPE | 0.999968 |
| K Projection | 0.999973 |
| K RMSNorm | 0.999975 |
| K RoPE | 0.999975 |
| V Projection | 0.999956 |
| V RMSNorm | 0.999941 |
| Attention*V (pre o_proj) | 0.999907 |
| Output Projection | 0.999949 |

#### Layer 10 prefill

Attention path still close:

- input norm: `0.991559`
- Q/K/V projections and norms: `~0.993-0.998`
- attention pre-o_proj: `0.995967`
- o_proj: `0.996669`

#### Layer 15 prefill

Attention path remains high:

- input norm: `0.997304`
- Q/K/V projections and norms: `~0.995-0.999`
- attention pre-o_proj: `0.995915`
- o_proj: `0.996678`

Conclusion: **attention internals are not the dominant remaining error source**.

---

### C) Where divergence now actually starts (Layer 4 decomposition)

With corrected alignment/hooking, layer 4 breakdown:

- `L4_block_normed` vs HF input LN: **0.999935**
- `L4_block_attn_raw` vs HF self-attn output: **0.999968**
- `L4_block_post_attn` (after post-attn norm + residual/laurel combine): **0.905710**
- `L4_block_output` (after FFN path): **0.806164**
- final `L4_post_pli_stream0` vs HF layer output stream0: **0.913320**

This localizes the first major true divergence to **post-attention block composition**, not attention sub-ops.

Likely hotspot path in Cactus:

- `build_transformer_block` in `cactus/models/model_gemma3n.cpp`
  - `post_attention_layernorm`
  - `combined = hidden + attn + laurel`
  - `residual = combined * RSQRT2`
  - `MLP` + `post_feedforward_layernorm`
  - plus subsequent AltUp correction and PLI injection in `build_layer`

---

### D) Full-layer hidden-state trend (prefill, aligned, `use_cache=True`)

Layer output cosine (Cactus stream0 vs HF stream0):

- L0 `0.999983`
- L1 `0.999975`
- L2 `0.999966`
- L3 `0.999967`
- L4 `0.913320`  <-- first large drop
- L5 `0.937576`
- L6 `0.994606`
- ...
- L20 `0.996628`
- ...
- L27 `0.985395`
- L28 `0.977718`
- L29 `0.976103`

Key point: no catastrophic collapse at shared-KV boundary when compared correctly. Remaining decay is moderate after the L4 event.

---

### E) Rollback Control Experiments (2026-03-07 addendum)

#### Attention-kernel FP32 rollback (RMSNorm/RoPE vector path)

After rolling back FP32 vector-path edits in `kernel_attention.cpp`:

- L0 attention remained near-perfect (`attn_output_pre_oproj=0.999974`, `o_proj=0.999974`)
- L4 attention remained near-perfect (`attn_output_pre_oproj=0.999907`, `o_proj=0.999949`)
- first major layer drop stayed at `L3->L4` (`0.086666`)
- `L29` stayed ~`0.9763`

Result: no divergence shift back into attention.

#### Matmul-kernel FP32 rollback (`cactus_matmul_f16_worker`)

After rolling back FP32 accumulator edits in `kernel_matmul.cpp` worker path:

- attention sub-op cosines at L0/L4 remained unchanged in practice
- first major drop still occurred at `L4` with the same magnitude
- `L29` cosine stayed ~`0.9763`

Result: no shift in divergence location. Likely explanation is that prefill shapes here dispatch mostly through Accelerate/SME2 paths, so this worker rollback is not the dominant compute path for the repro.

#### Graph-builder rollback + model-scoped explicit axis

- Graph-builder negative-axis remap changes were reverted.
- Gemma retained explicit positive axis usage in model code.
- Build remained clean after rollback.

Result: Gemma keeps correct per-row stats behavior without requiring graph-wide negative-axis remapping.

---

## Instrumentation / Comparison Caveats

### Shared-KV layer capture caveat

Some compare scripts can misreport layers `>=20` if HF patched forward does not preserve shared-KV bookkeeping exactly like native HF flow. Symptoms included:

- zeros for shared-layer K/V-derived tensors,
- `N/A-C` / `0.0` cosine artifacts,
- occasional `KeyError` for shared layer indices in partial patching setups.

When shared-KV comparisons looked catastrophic, re-running with corrected cache mode and hook path removed that false break.

### Cache mode mismatch caveat

Comparing Cactus prefill vs HF with inconsistent `use_cache` behavior can fabricate large late-layer divergence signatures. Use consistent cache mode and BOS alignment for fair comparisons.

---

## What Was Tried (and Current Interpretation)

Previously tested:

- FP32 matmul accumulators for FP16 path
- improved exp approximation
- FP16 KV cache
- INT8 vs FP16 weight runs
- FP32-style RMSNorm/RoPE kernel changes
- rollback of FP32 attention-kernel vector changes (no meaningful change in divergence location)
- rollback of FP32 matmul worker accumulator path (no meaningful change in divergence location)
- rollback of graph-builder axis remap after moving Gemma to explicit positive axis in-model

These changes do not explain the old `attn cos ~0.79` claim because that claim itself was instrumentation-bug dominated. They can still affect robustness/perf, but they are not the root cause of the corrected remaining gap.

---

## Remaining Problem Statement

After fixes, Cactus is mostly coherent but still not fully equal to HF bf16. Remaining mismatch is concentrated in:

- post-attention block composition and normalization path,
- and downstream AltUp/PLI interactions.

This can still flip tight token margins in decode, especially under FP16 multi-thread reduction non-associativity.

---

## Non-Determinism (Still True)

At `temperature=0`, Cactus can produce run-to-run differences due to parallel FP16 reduction order in kernels. HF bf16 reference remains deterministic in these tests.

---

## Recommended Next Debug Focus

1. Add per-layer debug nodes for **post-attn and FFN internals** (not just attention):
   - `post_attention_layernorm` output
   - `laurel` output
   - `hidden + attn + laurel` pre-RSQRT2
   - post-RSQRT2 residual
   - `pre_ffn_norm`, raw MLP output, post-FFN norm, block output
   - post-correct stream0 and post-PLI stream0

2. Mirror those exact captures in HF hooks and compare at L4 first, then L10/L15.

3. Validate whether mismatch is:
   - formula/order mismatch in block composition,
   - shape/broadcast subtlety,
   - or precision storage ordering in these non-attention ops.

4. Keep shared-KV hook logic faithful when probing layers `>=20`.

---

## Reference Directories

Two reference source directories are checked into the repo root for use during debugging and comparison:

- **`gemma/`** — Clone of Google Deepmind's official Gemma repository (includes `gemma/gemma/` subdirectory with model code, plus docs/examples/colabs).
- **`hf_gemma3n/`** — HuggingFace Transformers Gemma 3n module files (`modeling_gemma3n.py`, `configuration_gemma3n.py`, etc.). Used as the ground-truth reference implementation for prefill comparisons and hook-based debug captures.

---

## Files / Areas Changed During Investigation

- `python/src/config_utils.py` (layer_types/config extraction)
- `cactus/graph/graph_builder.cpp` (temporary negative-axis remap experiment, later reverted)
- `cactus/models/model_gemma3n.cpp` (debug node instrumentation + explicit positive-axis row-wise reductions)
- `cactus/kernel/kernel_attention.cpp` (precision and exp-path experiments)
- `cactus/kernel/kernel_matmul.cpp` (FP16 matmul accumulator experiments)
- `tests/compare_prefill.py`
- `tests/compare_attention_ops.py`
- `tests/generate_profiles.py`
- additional debug scripts under `tests/`

---

## Final Current Conclusion

The major "attention divergence" diagnosis was a false lead after instrumentation bugs were corrected.  
**Current root divergence is downstream of attention, starting in layer block composition (post-attention residual/FFN/AltUp-PLI path), with first major drop at L4.**
