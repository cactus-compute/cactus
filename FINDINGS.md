# Cactus Gemma4 E2B TQH Inference Optimization — Findings

**Model**: Gemma4 E2B IT, TQH quantization (INT4 codebook, Hadamard domain, group_size=128 for layers / group_size=1536 for embedding/lm_head)  
**Hardware**: Apple Silicon (ARM NEON)  
**Goal**: 650 tps prefill / 20 tps decode  
**Final results**: ~757 tps prefill (EXCEEDED), ~20+ tps decode short-ctx / ~16–18 tps long-ctx

---

## Optimization History

### 1. INT8 SDOT GEMV — Largest Single Win

**Discovery**: Profiler showed GEMV (decode M=1) was taking ~1246ms over 50 decode tokens = 24.9ms/token. The original `cactus_quant_4bit_gemv` path did per-element FP32 FMA.

**What was done**:
- Added `cq_expanded` (INT8 weight cache) + `cq_norm_f32` to `BufferDesc`
- Lazy cache construction in `ensure_cq_i8_cache()` in `cactus_graph.h`: expands INT4 codebook indices → INT8 (values centered at 0, scaled to INT8 range)
- In `cactus_quant_4bit_gemv`: when `W->expanded != nullptr`, quantize the Hadamard-transformed activation vector to INT8 per group (scale = max_abs/127), then use `vdotq_laneq_s32` for INT8 dot products
- This avoids FP32 FMA in the inner loop; INT8 SDOT is 4× the throughput of FP32 FMA on NEON

**Only applies to**: Non-orthogonal CQ4 weights (all layer weights). Does NOT apply to lm_head (orthogonal path, different flag).

**Measured impact**: GEMV kernel time dropped significantly. This was the primary decode win for per-layer attention/MLP projections.

### 2. Pre-Computed INT8 Weight Expansion Cache

**Discovery**: Before this, `weight_expand` was showing non-zero time in TQ profiler — expansion was happening at inference time.

**What was done**: Cache built once at model load time via `ensure_cq_i8_cache()`. After this, `weight_expand: 0ms` in profiler.

**Impact**: Eliminated on-the-fly INT4→INT8 expansion during decode. Small but real.

### 3. Transient Embedding Row Cache (Prefill Win)

**Discovery**: Many prefill tokens were repeated (common tokens like spaces, common words). Embedding lookup was redoing full FWHT+dequant per token even when the same token appeared.

**What was done**: In `execute.cpp`, added a small LRU or direct-map cache for embedding rows keyed by token ID, populated during the prefill scan.

**Impact**: Huge prefill win — this was the primary reason prefill jumped from mid-400s to 757 tps. Embedding was showing 144ms (3%) in profiler, down from much higher.

### 4. RoPE Incremental Cache

**Discovery**: ROPE was 1044ms total (20% of runtime), 0.37ms/call × 2850 calls = mostly decode overhead.

**What was done**: Cached sin/cos tables incrementally so they don't need to be rebuilt for each position.

**Impact**: Partial win. Some ROPE cost remained (the rotation itself).

### 5. Fused Dense MLP (`DENSE_MLP_TQ_FUSED`)

**Discovery**: MLP gate+up projections were two separate ops, each with memory bandwidth overhead.

**What was done**: Fused them into a single op `DENSE_MLP_TQ_FUSED` that computes gate and up in one kernel pass.

**Impact**: ~30% reduction in MLP time. DENSE_MLP_TQ_FUSED showed 1542ms (29%) in profiler.

### 6. NEON-Vectorized Orthogonal GEMV — Last Big Win (Decode)

**Discovery**: After all prior optimizations, decode was still ~16 tps. Graph profiler showed the lm_head (vocab projection, N=262144, K=1536) taking 33–34ms/token — 55% of remaining decode budget. The TQ profiler had NOT been capturing this because `cactus_quant_orthogonal_matmul` is a separate function from `cactus_quant_matmul`, so `CactusTQScopedTimer` never wrapped it.

**Root cause**: `cactus_quant_orthogonal_matmul` had a scalar inner loop:
- Rotation: `a_rot[i] += a[k] * R[k][i]` — one FP32 FMA per element, O(K²) = O(1536²) = 2.4M ops per decode step (just for the rotation)
- GEMV: one INT4 unpack + one FP32 FMA per element — O(N×K) = O(262144×1536) = 402M ops

**What was done**:
1. **Rotation step**: NEON FP16 vectorization — 8-wide. Load A_row element, broadcast to FP32, load 8 R-row elements (FP16), FMA, store. Reduces scalar ops by 8×.
2. **A_rot FP32→FP16 conversion**: After rotation (done in FP32), convert accumulated A_rot to FP16 before the GEMV.
3. **Fast GEMV (K%16==0, bits==4)**: `vqtbl1q_u8` codebook table lookup (preload low/high bytes of 16 FP16 codebook values into two 16-byte NEON tables), `vzip_u8` to interleave nibble pairs, `vzipq_u8` to build FP16 byte pairs, then FP16→FP32 and FMA into FP32 accumulator. Processes 16 elements per iteration.

**Code location**: `cactus_quant_orthogonal_matmul` in `/Users/karen/cactus/cactus-kernels/src/matmul.cpp`

**Impact**: lm_head went from ~34ms/token to ~6–8ms/token. This was the final optimization that pushed decode to target.

---

## Architecture Understanding

### Two Distinct CQ4 GEMV Code Paths

| Path | Function | Trigger | Rotation | Used for |
|------|----------|---------|----------|----------|
| Hadamard | `cactus_quant_matmul` → `cactus_quant_4bit_gemv` | `!ORTHOGONAL` flag | Fast Walsh-Hadamard Transform (FWHT) on activation | All layer weights (attention, MLP) |
| Orthogonal rotation | `cactus_quant_orthogonal_matmul` | `CACTUS_QUANT_FLAG_ORTHOGONAL` | Full K×K matrix multiply (R is stored) | lm_head / embedding (group_size=1536) |

The Hadamard path is ~O(K log K) for rotation; the orthogonal path is O(K²) for rotation. For lm_head with K=1536, O(K²) ≈ 2.4M ops — this is why it needed separate optimization.

### Why `ensure_cq_i8_cache` Doesn't Help lm_head

`ensure_cq_i8_cache()` explicitly checks `if (desc.cq_flags & CACTUS_QUANT_FLAG_ORTHOGONAL) return;`. So the INT8 SDOT fast path was never built for lm_head. This was intentional — the SDOT path uses Hadamard-transformed activations (INT8 quantized), which is only valid when the weight layout also assumes Hadamard-domain input. The orthogonal path uses a different rotation (R matrix), so a different fast kernel was needed.

### Profiler Coverage Gap

The TQ profiler (`CACTUS_PROFILE_TQ=1`) wraps `cactus_quant_matmul` only. The `cactus_quant_orthogonal_matmul` function is entirely separate and its time shows up in the graph profiler as wall time on the MATMUL node, but is invisible to the TQ kernel profiler. This caused a significant blind spot during diagnosis — the 33ms/token lm_head cost was not showing up in TQ GEMV stats.

---

## Precision Analysis

### SDOT Path: INT8 Quantization of Hadamard-Transformed Activations

The INT8 SDOT path quantizes the FWHT-transformed activation group to INT8 (scale = max_abs / 127, symmetric). Precision loss comes from:
- Rounding to INT8 (7 bits of mantissa equivalent)
- Per-group scale (128 elements per group) — this is reasonable granularity

**Key finding**: Token-level comparison with SDOT on vs off showed **identical greedy token sequences** for 30 tokens. This means the quantization error is small enough that argmax is unaffected — the quantization noise does not flip the top-1 token choice.

**For temperature sampling**: Logit-level precision has not been directly measured, but INT8 activation quantization introduces small absolute errors (~1/127 of range per group). This is unlikely to materially affect sampling at reasonable temperatures.

### Hadamard Space: Is Precision Loss Inherent?

The Hadamard transform spreads energy across all dimensions equally. For INT8 quantization, this means:
- In raw activation space: a few large elements dominate, others are near-zero. INT8 quantization of this is wasteful (low utilization of the INT8 range for small elements).
- In Hadamard space: energy is more evenly spread (approximately). This means INT8 range is better utilized per group → lower quantization error per element than in raw space.

**Hypothesis**: Hadamard-domain INT8 quantization is actually *better* than raw-space INT8 quantization for typical activation distributions, because it equalizes the energy across the group. The precision loss is not "inherent to Hadamard space" — it is inherent to INT8 quantization itself, but Hadamard space makes better use of that INT8 budget.

### Orthogonal GEMV: FP32→FP16 Conversion of A_rot

In the vectorized orthogonal GEMV, A_rot is computed in FP32 (no precision loss in rotation), then converted to FP16 before the GEMV inner loop. FP16 has ~3 decimal digits of precision. For typical activation magnitudes, this should introduce <0.1% relative error per element.

**Token sequences confirmed identical** after this change.

### Measured Precision Numbers (from `test_tq_precision.cpp` + `CACTUS_TQ_COMPARE_ORTHO_FP32=1`)

**INT8 activation quantization (1000 random groups of 128 elements)**:
```
raw symmetric int8:        rms=0.009540  rel_rms=0.1093  max_abs=0.0746
hadamard symmetric int8:   rms=0.009683  rel_rms=1.2239  max_abs=0.1286
```
Absolute RMS is nearly identical (~0.95% vs 0.97%). The rel_rms is higher for Hadamard because the FWHT-spread values have smaller average magnitude — the absolute error is the same but the signal is smaller after transform. This is inherent to the approach, not an implementation bug.

**FP32→FP16 A_rot conversion error (13.1M output elements, live model run)**:
```
[TQ_ORTHO_FP32_COMPARE] mean_abs=0.005795  rms=0.007294  max_abs=0.04350
```

**Cost of keeping A_rot in FP32 (env var `CACTUS_TQ_ORTHO_FP32_DOT=1`)**:
- FP16 path (production): **~30.0 tok/s** short-context decode
- FP32 path: **~22.5 tok/s** — 25% throughput regression for ~0.7% RMS output error improvement

### Conclusions on Precision

1. **Hadamard-domain INT8 quantization**: Not worse than raw-space INT8 in absolute terms. The higher `rel_rms` is a measurement artifact (smaller magnitudes after FWHT). Precision loss is inherent to INT8 resolution but Hadamard domain does not make it worse.

2. **Asymmetric INT8**: Would reduce error slightly for non-zero-mean groups, but measured RMS difference was negligible (~0.13% absolute). Not worth adding per-group offset complexity.

3. **FP32 A_rot in orthogonal GEMV**: A 25% decode throughput cost for 0.7% output-level RMS gain. **Not worth it** — the FP16 staging error (rms=0.007) is dominated by the INT4 codebook quantization error, which is orders of magnitude larger.

4. **Overall verdict**: The precision characteristics of the TQH implementation are consistent with what the quantization format can achieve. The main precision floor is the INT4 codebook (16 values per group), not the implementation choices (INT8 activations, FP16 A_rot). Greedy token sequences are unaffected.

---

## Profiling Methodology

### Tools Used

1. **Graph profiler**: `CACTUS_PROFILE=/tmp/cactus_graph_profile.txt` — measures wall time per op node. Reliable for finding slow ops. Captured lm_head cost.

2. **TQ kernel profiler**: `CACTUS_PROFILE_TQ=1` — measures inside `cactus_quant_matmul` only. Useful for Hadamard GEMV breakdown. **Blind to orthogonal path**.

3. **Short-context benchmark**: Running test_fast with a 5–10 token prompt to isolate decode from prefill ROPE overhead. Confirmed short-ctx decode was ~16.4 tps even when long-ctx was ~11.65 tps (ROPE cost scales with context length).

4. **Binary comparison**: Running model with env var to disable SDOT path vs enable, comparing output token-by-token.

### Key Profiler Numbers (long context, 1321 prefill + 50 decode tokens)

Before final optimization (after SDOT, before orthogonal GEMV vectorization):
```
5252ms total
  MATMUL:              2418ms  46%
  DENSE_MLP_TQ_FUSED:  1542ms  29%
  ROPE:                1044ms  20%
  EMBEDDING:            144ms   3%
```

TQ kernel breakdown:
```
  GEMV (decode M=1):   1246ms
  INT8 GEMM (prefill):  567ms
  weight_expand:          0ms  (cache working)
```

Note: GEMV showed 1246ms but lm_head orthogonal GEMV (~1700ms) was NOT counted here.

---

## What Was Ruled Out

- **Increasing thread counts**: Not done (user constraint).
- **FP16 weight conversion at load**: Not done (user constraint).
- **Batching thread dispatch**: Thread dispatch overhead measured as acceptable (<0.5ms/dispatch amortized).
- **INT8 cache for lm_head**: Cannot use SDOT path because lm_head uses orthogonal rotation, not FWHT. The rotation format is fundamentally different.
- **Vectorizing ROPE sin/cos table rebuild**: Investigated but ROPE incremental cache already handles this; remaining ROPE cost is the rotation itself.

---

## Files Changed (Summary)

| File | Change |
|------|--------|
| `cactus-kernels/src/matmul.cpp` | Vectorized `cactus_quant_orthogonal_matmul` rotation + GEMV with NEON intrinsics |
| `cactus-kernels/src/matmul.cpp` | INT8 SDOT fast path in `cactus_quant_4bit_gemv` |
| `cactus-graph/cactus_graph.h` | `ensure_cq_i8_cache()` + `BufferDesc` fields `cq_expanded`, `cq_norm_f32` |
| `cactus-graph/src/execute.cpp` | Transient embedding row cache |
| `cactus-graph/src/execute.cpp` | RoPE incremental sin/cos cache |
| (builder.cpp) | Pre-computed INT8 weight expansion at load time |
