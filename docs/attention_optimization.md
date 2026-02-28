# Attention Kernel Optimization

Living document tracking optimization work on the cactus attention kernels.

## Goal

Maximize throughput of the attention kernels on ARM (Apple Silicon + Android) for both **decode** (seq_len=1, autoregressive generation) and **prefill** (seq_len>=64, prompt processing). The hybrid INT8/FP16 path is the primary target since all production models (Qwen, Gemma, LFM2, LFM2-MoE) use INT8-quantized KV caches.

## Constraints

- **No accuracy regression.** Max output diff vs. reference must stay below 0.05 (INT8 quantization noise floor). Ideally < 0.01 for FP16-only paths.
- **No new dependencies.** Apple Accelerate is already linked; no additional libraries.
- **ARM NEON baseline.** All paths must have a non-Accelerate fallback. Build flags: `-march=armv8.2-a+fp16+simd+dotprod+i8mm`.
- **Thread safety.** Kernels are called from `CactusThreading::parallel_for`. No shared mutable state across work items.
- **GQA support.** Must handle `num_q_heads != num_kv_heads` (grouped query attention). All models use GQA.
- **Head dimensions.** Must work for any `head_dim` divisible by 8. Common values: 64 (older), 128 (Qwen3, Gemma).
- **Quantization groups.** KV cache uses per-group INT8 quantization with `KV_QUANT_GROUP_SIZE = 32`. Scales are per `(seq_pos, kv_head, group)`.

## Current Architecture

```
cactus_attention_f16()                         # Public API: all-FP16 attention
  |-- head_dim%8, no mask, no window?
  |     YES --> cactus_attention_f16_fast()     # NEON scalar, BLOCK_SIZE=32
  |               |-- [Apple] seq_len >= 64?
  |               |     YES --> cactus_attention_f16_accelerate()  # cblas_sgemm, BLOCK=64
  |               |     NO  --> NEON inner loop
  |     NO  --> generic path (mask/window support)

cactus_attention_hybrid_int8_fp16()            # Public API: INT8 cached KV + FP16 new
  |-- [Apple] seq_len>=64, head_dim%8, no window?
  |     YES --> cactus_attention_hybrid_int8_fp16_accelerate()  # cblas_sgemm + dequant
  |     NO  --> NEON inner loop with per-element INT8/FP16 branching
```

**Threading:** `parallel_for` over `batch * num_q_heads * seq_len` (NEON paths) or `batch * num_q_heads` (Accelerate paths). Apple threshold: min_work=32, work_per_thread=16 (runtime-configurable via `get_attention_config()`).

**Online softmax:** All paths use block-wise online softmax with running max/sum. NEON paths use scalar `expf()`. Accelerate paths use a vectorized polynomial exp approximation.

## Tests

### Correctness

- **test_kernel.cpp** — `test_neon_attention_fp16_correctness()`: sanity check (batch=1, seq=2, heads=1, head_dim=8)
- **test_graph.cpp** — `test_attention()`: graph-level integration (batch=1, seq=2, heads=1, head_dim=4)

### Performance Benchmarks

**Primary: `tests/run_benchmark.sh --attention`** (uses `tests/bench/attn_bench.cpp`)

The bench framework uses the backend system in `tests/bench/backend_cactus.cpp`:
- **Prefill**: calls `cactus_attention_f16` (all FP16, dispatches to Accelerate at seq_len>=64)
- **Decode**: calls `cactus_attention_hybrid_int8_fp16` (INT8 cached KV + FP16 new token)
- Accuracy verified against an fp32 reference implementation

**Cold-cache methodology**: For decode, the benchmark creates N distinct KV cache states (each with unique random data) and cycles through them during timed iterations (`states[iter % N]`). N is computed dynamically so total KV data across all states is ~64MB, ensuring every iteration forces a cold load from DRAM rather than hitting L2/SLC cache. This matches real inference where each transformer layer has unique KV cache data.

```bash
# Sweep decode across cache_len={8,16,32,64,128,256,512} with cold-cache cycling
bash tests/run_benchmark.sh --attention --backends cactus --sweep

# Single cache_len
bash tests/run_benchmark.sh --attention --backends cactus --cache_len 1024
```

Config: head_dim=128, q_heads=32, kv_heads=8 (Llama-3 style). Adjustable via `--head_dim`, `--q_heads`, `--kv_heads`.

## Experiment Tracker

### Baseline

Measured on Apple M3 Pro. Config: head_dim=128, q_heads=32, kv_heads=8, causal=true.

Cold-cache decode (64MB state cycling, `--sweep`):

| cache_len | states | us/call | GFLOPS |
|---|---|---|---|
| 8 | 512 | 14.3 | 10.4 |
| 16 | 512 | 20.2 | 13.9 |
| 32 | 512 | 28.6 | 19.1 |
| 64 | 431 | 39.4 | 27.3 |
| 128 | 221 | 65.6 | 32.5 |
| 256 | 112 | 114.5 | 37.1 |
| 512 | 56 | 214.9 | 39.5 |

Prefill (seq_len=1024): 42023us (412.8 GFLOPS)

### Exp 1+2: Q Pre-convert to f32 + Vectorized Softmax Exp

Pre-convert Q fp16->f32 once per query (stack array), use vectorized polynomial exp for softmax (4 scores/iteration), stack arrays instead of std::vector for accumulators.

Cold-cache results (64MB cycling, `--sweep`):

| cache_len | Baseline us | Exp 1+2 us | Change |
|---|---|---|---|
| 8 | 14.6 | 14.5 | ~same |
| 16 | 20.4 | 15.5 | **24% faster** |
| 32 | 24.4 | 22.9 | 6% faster |
| 64 | 40.6 | 50.0 | 23% slower |
| 128 | 68.9 | 80.0 | 16% slower |
| 256 | 124.0 | 113.5 | 8% faster |
| 512 | 217.4 | 218.1 | ~same |

**Conclusion:** Noisy, inconsistent results. Some sizes faster, some slower — within measurement variance. No clear directional improvement. **Reverted.**

### Exp 3: SDOT INT8 Q*K Scoring

Dynamically quantize Q to int8 once, use `vdotq_s32` for 4x arithmetic throughput on Q*K scoring against INT8 cached keys.

Cold-cache results (64MB cycling, `--sweep`):

| cache_len | Baseline us | +SDOT us | Change |
|---|---|---|---|
| 8 | 14.6 | 17.4 | 19% slower |
| 16 | 20.4 | 20.4 | ~same |
| 32 | 24.4 | 27.7 | 14% slower |
| 64 | 40.6 | 37.5 | 8% faster |
| 128 | 68.9 | 67.7 | ~same |
| 256 | 124.0 | 118.0 | 5% faster |
| 512 | 217.4 | 220.1 | ~same |

**Conclusion:** Again noisy. The Q quantization overhead hurts at small cache_len. At larger sizes, within noise. SDOT saves compute but the kernel is bandwidth-bound — the savings are hidden by memory stalls. **Reverted.**

### Exp 4: GQA-Grouped KV Loading

Not re-run with cold-cache. The hot-cache result showed a 10% regression at long contexts from halved parallelism (16→8 work items). Cold-cache would only worsen this since the larger per-thread stack footprint (~20KB for G=2) increases cache pressure. **Not pursued.**

### Key Finding

All experiments confirm: **the hybrid INT8/FP16 decode path is memory-bandwidth-bound on Apple Silicon.** No arithmetic optimization (Q pre-conversion, vectorized exp, SDOT, GQA grouping) produces consistent improvement under cold-cache conditions. Future optimizations must reduce the amount of data loaded, not the compute per element.

### Exp 5: Transposed KV Layout

**What:** Store cached KV data in `[kv_heads, seq_len, head_dim]` instead of `[seq_len, kv_heads, head_dim]`. This makes consecutive positions for the same head contiguous in memory (stride = `head_dim` = 128 bytes) instead of strided (stride = `kv_heads * head_dim` = 1024 bytes).

**How:** New kernel `cactus_attention_hybrid_int8_fp16_transposed()` with changed stride calculations:

```cpp
// Original: K_cached_base + kv_pos * (num_kv_heads * head_dim) + kv_head * head_dim
// Transposed: K_head_base + kv_pos * head_dim  (where K_head_base = K + kv_head * cache_len * head_dim)
```

Scales follow the same transposition. Benchmark generates data in head-first layout via straight fp32→fp16 conversion (no transpose needed since input is already `[head, seq, dim]`).

Cold-cache results (64MB cycling, fine-grained sweep):

| cache_len | Baseline us | Transposed us | Change |
|---|---|---|---|
| 8 | 18.1 | 17.5 | 3% faster |
| 16 | 22.8 | 20.6 | **10% faster** |
| 32 | 25.1 | 23.6 | **6% faster** |
| 48 | 33.3 | 30.5 | **8% faster** |
| 64 | 40.8 | 34.8 | **15% faster** |
| 80 | 48.7 | 41.3 | **15% faster** |
| 96 | 55.8 | 47.4 | **15% faster** |
| 128 | 63.8 | 59.7 | **7% faster** |
| 256 | 113.6 | 109.0 | **4% faster** |
| 512 | 211.0 | 206.0 | 2% faster |

**Conclusion:** Consistent improvement across all sizes, 2-15%. Best gains at small-to-medium cache_len (16-96) where the hardware prefetcher benefits most from the reduced stride. At long contexts the prefetcher warms up regardless, so the layout advantage shrinks. **Not yet integrated into production** — requires KV cache engine changes to store data in head-first layout.

**Caveats:** The benchmark generates data directly in `[head, seq, dim]` layout, skipping the transpose cost. In production, the graph outputs K/V in `[seq_len, kv_heads, head_dim]`, so the quantization step would need to scatter-write into `kv_heads` separate memory regions — an extra cost the benchmark doesn't capture. Beyond the kernel, adopting this layout touches several engine subsystems:

- **Cache writes become strided.** Appending a new token currently writes one contiguous block of `kv_heads * head_dim` bytes. With transposed layout, each token's data must be written to `kv_heads` separate locations (one per head's contiguous region), turning a single memcpy into `kv_heads` scattered writes.
- **Sliding window shifts multiply.** The current layout shifts tokens with one memmove across the entire cache. Transposed layout requires `kv_heads` independent memmoves (one per head), adding code complexity and potentially more cache line conflicts during the shift.
- **Gains are smallest where they matter most.** The 15% wins are at cache_len 64-96, where decode is already fast (~40-55us). At cache_len 512 where decode actually bottlenecks, the gain is only 2%. The total bytes loaded are identical — the optimization only helps the prefetcher at short strides, and the prefetcher adapts to the strided pattern at longer contexts regardless.
- **Unattributed source of improvement.** The conclusion attributes the gains to "the hardware prefetcher benefits most from the reduced stride," but reducing stride from 1024 to 128 bytes simultaneously improves multiple things: (a) prefetcher stream detection, (b) TLB efficiency (8x fewer page crossings per head), and (c) spatial locality (consecutive cache lines are all useful data for this head, vs. the baseline where 7/8 of the surrounding cache lines belong to other heads). The benchmark doesn't isolate which factor dominates. This matters because the factors respond differently to production conditions — e.g., TLB pressure from concurrent layers could make (b) more significant than the benchmark suggests.
- **Isolated prefetcher state.** The benchmark runs the attention kernel in a tight loop — even with cold-cache cycling, the prefetcher sees the same stride pattern repeatedly and stays warmed for that pattern. In production, attention is interleaved with MLP/matmul/norm layers whose different access patterns disrupt the prefetcher state between calls. The relative benefit of a simpler stride could differ when the prefetcher must re-learn the pattern each layer.

The refactor blast radius (cache allocation, quantization, sliding window, sink tokens) is significant relative to the 2-4% gain at the context lengths that matter for user-perceived latency.

### Exp 6: Split-K Decode Parallelism

**What:** Split the KV range across 4 threads per query head. Each thread computes a partial (max, sum, weighted_acc) over its chunk, then a merge phase combines the partial online-softmax results.

**Why:** The baseline parallelizes decode over `batch * num_q_heads * seq_len`. For decode (seq_len=1), that's `batch * num_q_heads` work items — 32 for a typical Llama-3 config. Each work item walks the entire KV range sequentially, issuing one stream of memory loads. With 12 cores on M3 Pro, that's ~2.7 heads per core, and each core can only have a limited number of outstanding memory requests. The memory controller is underutilized because there aren't enough independent load streams to saturate it.

Split-K multiplies the work items by the split count: `batch * num_q_heads * 4` = 128 work items. Each work item loads from a different region of the KV cache, creating 4x more independent load streams. The memory controller can now service more requests in parallel, better saturating DRAM bandwidth. The total bytes loaded are the same, but the effective bandwidth utilization is higher.

**CactusThreading reference** (see `kernel/kernel_utils.h`):

- **ThreadPool:** Singleton pool initialized to `hardware_concurrency()` threads (capped at 16). Persistent worker threads pull tasks from a shared deque protected by a mutex + condition variable. No work stealing — each thread gets a fixed range of work items via static partitioning.
- **`parallel_for(total_work, config, work_func)`:** Computes thread count from config, divides work evenly (`total_work / num_threads`, last thread gets remainder), enqueues one task per thread, blocks until all complete (by default). Each task calls `work_func(start_idx, end_idx)`. Synchronous — acts as an implicit barrier.
- **`ParallelConfig{min_work_gate, work_per_thread}`:** Two-parameter threshold. If `total_work < min_work_gate`, run single-threaded. Otherwise, `num_threads = ceil(total_work / work_per_thread)`, capped at pool size.
- **`Thresholds::ATTENTION`:** Apple = `{32, 16}`, Android = `{64, 32}`. For baseline decode on Apple (32 work items): 32 >= 32 so not gated, `ceil(32/16) = 2` threads. For split-K decode (128 work items): `ceil(128/16) = 8` threads. On Android, 32 < 64 gates to single-threaded; split-K's 128 work items would give `ceil(128/32) = 4` threads.
- **No async overlap:** `parallel_for` blocks the caller until all threads finish. Two consecutive `parallel_for` calls (as in split-K's compute + merge phases) execute sequentially with a full barrier between them.

**How:** Two-phase `parallel_for`:
- Phase 1: `parallel_for(batch * num_q_heads * 4)` — each work item processes `kv_range / 4` positions, producing partial online-softmax state (running_max, running_sum, weighted V accumulator). Writes these to a shared buffer indexed by `(q_item, split_idx)`.
- Phase 2: `parallel_for(batch * num_q_heads)` — each work item merges 4 partial results using the log-sum-exp correction: `global_max = max(split_max_i)`, then `global_sum = Σ(split_sum_i * exp(split_max_i - global_max))`, `output = Σ(split_acc_i * exp(split_max_i - global_max)) / global_sum`.

Critical implementation detail: the block loop must start at the exact `split_start` position, not rounded down to a block boundary. Rounding down causes positions to be double-counted across splits, producing incorrect softmax weights.

Falls back to the baseline single-thread path when `kv_seq_len < 64` (too little work to justify split overhead).

Cold-cache results (64MB cycling, fine-grained sweep):

| cache_len | Baseline us | Split-K us | Change |
|---|---|---|---|
| 8 | 18.1 | 18.0 | ~same (fallback) |
| 16 | 22.8 | 18.8 | **15% faster** |
| 32 | 25.1 | 26.5 | ~same |
| 48 | 33.3 | 32.3 | 3% faster |
| 64 | 40.8 | 48.6 | 19% slower |
| 80 | 48.7 | 52.9 | 9% slower |
| 96 | 55.8 | 55.3 | ~same |
| 128 | 63.8 | 55.8 | **13% faster** |
| 192 | 88.1 | 64.1 | **27% faster** |
| 256 | 113.6 | 75.2 | **34% faster** |
| 384 | 170.9 | 82.4 | **52% faster** |
| 512 | 211.0 | 98.0 | **54% faster** |

**Conclusion:** Massive wins at long contexts (192+: 27-54% faster). The 4-way split better saturates the memory controller's outstanding request queue. Regression at 64-80 where the split creates only ~16-20 positions per thread — not enough work to amortize the overhead. Crossover point is around cache_len=96-112.

**Caveats:** These results isolate the attention kernel. In production, split-K introduces overhead that the benchmark doesn't capture:

- **Double barrier.** Split-K runs two `parallel_for` calls (compute + merge) per invocation vs one in the baseline. Each is a thread barrier. In a 30-layer model where attention is interleaved with matmul/MLP/norm, these extra barriers add pipeline stalls that compound across layers.
- **Per-call allocation.** The partials buffer (~65KB for the Llama-3 config) is heap-allocated and zeroed on every decode step. This competes with KV cache data for L2 space and adds allocator pressure absent from the benchmark's tight loop.
- **Thread threshold mismatch.** The baseline's ATTENTION threshold (`work_per_thread=16`) was tuned for the non-split case and results in only 2 threads for 32 decode work items on Apple. Split-K's 128 work items jump to 8 threads. The threshold wasn't designed for this — the original 2-thread choice may reflect a deliberate tradeoff against synchronization overhead. On Android (heterogeneous cores), the baseline intentionally runs single-threaded to avoid stalling on slow little cores; split-K would force work onto them.
- **Confounded threading vs. access pattern.** ~~The benchmark doesn't isolate whether the speedup comes from the split access pattern or simply from having more active threads.~~ **Resolved** — see isolation experiment below.

End-to-end token generation latency should be measured before shipping. The partials buffer should be pre-allocated rather than per-call, and the ATTENTION thresholds may need re-tuning for the split-K work distribution.

#### Exp 6b: Threading Isolation (Highthread Baseline)

**What:** Isolate whether split-K's gains come from the split access pattern (independent load streams hitting different KV regions) or simply from more threads. Added a `cactus_decode_highthread` benchmark variant that gives the baseline NEON path the same 8-thread count as split-K, without changing the access pattern.

**How:** Runtime-overridable `ParallelConfig` for attention threading (`set_attention_config(1, 4)` → `ceil(32/4) = 8` threads from 32 work items) plus runtime-overridable split-K auto-dispatch threshold (`set_splitk_auto_threshold(SIZE_MAX)` → prevents redirect to split-K). Both reset after each call. The baseline `cactus_decode` was also fixed to disable split-K auto-dispatch so it always runs the NEON path with default 2 threads, ensuring a clean comparison.

| Variant | Threads | Access Pattern |
|---|---|---|
| `decode` (baseline) | 2 | Each thread walks full KV range |
| `highthread` | 8 | Each thread walks full KV range |
| `splitk` | 8 | Each thread walks 1/4 of KV range |

Cold-cache results (64MB cycling, `--sweep`):

| cache_len | Baseline us | Highthread us | Split-K us | Highthread vs base | Split-K vs base |
|---|---|---|---|---|---|
| 8 | 16.1 | 36.5 | 17.6 | +127% (worse) | +9% |
| 16 | 17.6 | 35.4 | 17.3 | +101% (worse) | -2% |
| 32 | 25.0 | 32.2 | 27.0 | +29% (worse) | +8% |
| 64 | 42.5 | 44.9 | 48.7 | +6% | +15% |
| 128 | 66.7 | 54.0 | 55.5 | **-19%** | **-17%** |
| 256 | 112.2 | 63.7 | 95.4 | **-43%** | **-15%** |
| 512 | 212.6 | 97.0 | 94.7 | **-54%** | **-55%** |

**Conclusion:** The speedup is almost entirely from higher concurrency, not the split access pattern. At every cache length where split-K shows gains, highthread matches or beats it using the same NEON code path with more threads. At cache_len=256, highthread (-43%) significantly outperforms split-K (-15%) — the split-K reduce phase is pure overhead when concurrency is equal. They converge at 512 where there's enough work to amortize the merge cost.

**Implication:** Rather than the split-K auto-dispatch (which adds a two-phase compute+merge, per-call partials allocation, and double barrier), the same or better speedup can be achieved by simply lowering the attention `ParallelConfig` threshold from `{32, 16}` to something like `{1, 4}` to give the standard NEON path more threads at longer cache lengths. This eliminates the split-K code path complexity entirely.

### Conclusions

Seven experiments across arithmetic optimizations (Exp 1-3), data layout (Exp 4-5), and parallelism (Exp 6) establish that the hybrid INT8/FP16 decode path is **memory-bandwidth-bound** on Apple Silicon. No arithmetic optimization — Q pre-conversion, vectorized exp, SDOT scoring, GQA grouping — produces consistent improvement under cold-cache conditions. The compute hides entirely behind memory stalls, ruling out the entire class of "do less math per element" optimizations.

Two levers produce real gains:

**1. Concurrency (40-55% at long cache).** Exp 6b's threading isolation proved that split-K's gains come entirely from having more active threads (8 vs 2), not from the split access pattern. Simply lowering the attention `ParallelConfig` threshold from `{32, 16}` to `{1, 4}` — giving the unmodified NEON path 8 threads — matches or exceeds split-K at every cache length, without the two-phase overhead, per-call allocation, or double barrier. At cache_len=256, the highthread baseline (-43%) significantly outperforms split-K (-15%). This is the highest-impact, lowest-effort optimization available.

**2. Access pattern / spatial locality (2-15%).** The transposed KV layout (Exp 5) reduces the stride between consecutive KV positions for the same head from 1024 bytes (`kv_heads * head_dim`) to 128 bytes (`head_dim`), producing consistent gains across all cache lengths. The improvement comes from better prefetcher stream detection, TLB efficiency (8x fewer page crossings per head), and spatial locality (consecutive cache lines are all useful data). Best gains are at short-to-medium cache lengths (16-96: 6-15%) where the prefetcher benefits most from the simpler stride; at long contexts (256-512) the gain is 2-4% as the prefetcher adapts to strided patterns regardless.

These two levers are orthogonal — concurrency increases the number of independent memory streams, while access pattern optimization makes each stream more efficient. They should compound, though this combination hasn't been benchmarked with the higher thread count (only with split-K in Exp 7, which confounded the two).

Further access pattern optimizations remain unexplored — see Exp 8 below for the most promising candidate.

**Recommended production strategy:**

- **Increase attention thread count.** Lower `Thresholds::ATTENTION` from `{32, 16}` to `{1, 4}` (or make it adaptive based on `kv_seq_len`). One-line change, no API or engine impact, 40-55% speedup at long cache lengths.
- **Transposed layout when engine supports it.** The kernel implementations are ready (`_transposed` variants). Worth integrating when the KV cache engine is refactored for other reasons (e.g., packed KV), since the 5-15% gains at medium cache lengths are real and the layout change would be part of a larger cache format update anyway.

### Exp 7: Packed KV Cache (Fused K/V Streaming)

**What:** Store K and V data adjacent per position instead of in separate buffers. This turns two independent memory streams into one, allowing V data to arrive in cache "for free" while loading K data.

**Why:** The current layout stores K and V in separate buffers:

```
K_cached: [pos0_h0_K][pos0_h1_K]...[pos1_h0_K]...   ← stream 1
V_cached: [pos0_h0_V][pos0_h1_V]...[pos1_h0_V]...   ← stream 2 (megabytes away)
```

The decode inner loop loads K data, computes Q*K scores, applies softmax, then switches to V data for weighted accumulation. The K and V loads are to completely different memory regions — potentially megabytes apart. This creates two problems:

1. **Two prefetcher streams per head.** The hardware prefetcher must track both K and V as separate sequential streams. When the kernel switches from K scoring to V accumulation, the prefetcher needs to start fetching from a completely different region. The V data isn't prefetched during K scoring.
2. **Double TLB pressure.** K and V occupy separate page ranges, so the TLB must map pages from both regions concurrently. With GQA (32 q_heads / 8 kv_heads = 4 heads processed per thread), that's 8 active TLB regions (K + V for each of 4 heads).

Apple Silicon uses **128-byte cache lines**. With head_dim=128 and INT8 data, K data per position per head is exactly 128 bytes = 1 cache line, and V data is another 128 bytes = 1 cache line.

**Packed layout** places K and V adjacent per position per head:

```
KV_cached: [pos0_h0_K(128B)][pos0_h0_V(128B)][pos0_h1_K(128B)][pos0_h1_V(128B)]...
```

Now K and V for the same position are consecutive cache lines. When the kernel loads K at position `pos`, the hardware prefetcher pulls V into L2 (or L1) before the kernel even needs it. By the time Q*K scoring and softmax are done, V data is already warm. Single memory stream, half the TLB pressure, V loads are effectively free.

The total bytes loaded are identical — the optimization is purely about turning two streams into one and exploiting spatial locality between K and V. This is the same principle that made Exp 5 (transposed layout) work: better locality without changing total bandwidth.

**How:** Three components:

1. **Packed quantization function** — new `cactus_quantize_kv_packed_fp16_to_int8()` that interleaves K and V into a single buffer with stride `2 * head_dim` per position per head. Scales can remain separate (they're small and accessed per-group, not per-element) or be interleaved similarly.
2. **Packed attention kernel** — new `cactus_attention_hybrid_int8_fp16_packed()` where the inner loop loads K and V from the same buffer at offsets `pos * 2 * head_dim` and `pos * 2 * head_dim + head_dim`. The online softmax block loop loads K for a block, computes scores, then loads V from adjacent memory (already in cache) for accumulation.
3. **Benchmark backend** — new `cactus_decode_packed` in `backend_cactus.cpp` with a prepare function that packs K+V and a run function that calls the packed kernel.

**Expected outcome:** The gain should be largest at medium cache lengths (64-256) where the working set exceeds L1 but fits in L2/SLC — this is where stream-switching overhead is highest. At very short cache lengths (8-32) both K and V likely fit in L1 regardless, so packing won't help. At very long cache lengths (512+) the kernel is deeply bandwidth-bound and the prefetcher has ample time to warm both streams, so the gain may be smaller.

If Exp 5's transposed layout (which improved spatial locality along the sequence dimension) gave 2-15%, packed layout (which improves spatial locality between K and V) could give a similar or larger improvement since it eliminates an entire memory stream rather than just reducing stride.

**Scales layout:** K and V scales are much smaller than the data (`4 floats = 16 bytes per position per head` vs `128 bytes`). Three options:
- Keep scales separate (simplest, scales are small enough that the extra streams don't matter much)
- Pack scales with data: `[K_data(128B)][V_data(128B)][K_scales(16B)][V_scales(16B)]` = 288 bytes per position (wastes 96 bytes in the third 128B cache line)
- Pack scales at the end of each KV pair's data: `[K_data(128B)][K_scales(16B)][V_data(128B)][V_scales(16B)]` = 288 bytes (same waste, but scales are adjacent to their data)

Start with separate scales for simplicity. If the baseline packed result is promising, try interleaved scales as a follow-up.

### Exp 7: Packed KV Cache (Fused K/V Streaming)

Two variants tested to isolate the effect of K/V interleaving vs. the head-first (transposed) layout:

| Variant | Layout | What it tests |
|---|---|---|
| `cactus_decode_stdpacked` | `[seq_pos, kv_heads, {K, V}]` | K/V interleaving only (standard stride, same as baseline but K+V adjacent) |
| `cactus_decode_packed` | `[kv_heads, seq_pos, {K, V}]` | K/V interleaving + head-first (transposed) stride combined |

**How:** Shared `cactus_quantize_kv_packed_fp16_to_int8()` packs K and V into a single buffer with stride `2 * head_dim` per position per head. Scales remain separate. Two attention kernels:
- `cactus_attention_hybrid_int8_fp16_stdpacked()` — based on baseline, standard `[seq_pos, kv_heads, 2*head_dim]` layout. Stride between positions for same head: `kv_heads * 2 * head_dim = 2048 bytes` (vs baseline's 1024).
- `cactus_attention_hybrid_int8_fp16_packed()` — based on transposed variant, head-first `[kv_heads, seq_pos, 2*head_dim]` layout. Stride between positions: `2 * head_dim = 256 bytes`.

#### Exp 7a: Standard-Layout Packed (K/V Interleaving Only)

Cold-cache results (64MB cycling, fine-grained sweep):

| cache_len | Baseline us | Stdpacked us | Change |
|---|---|---|---|
| 8 | 18.2 | 17.8 | ~same |
| 16 | 17.6 | 17.5 | ~same |
| 32 | 25.8 | 25.1 | 3% faster |
| 48 | 31.9 | 30.6 | 4% faster |
| 64 | 38.4 | 38.9 | ~same |
| 80 | 48.1 | 45.5 | 5% faster |
| 96 | 52.0 | 50.2 | 3% faster |
| 128 | 62.4 | 60.6 | 3% faster |
| 192 | 90.0 | 88.2 | 2% faster |
| 256 | 110.9 | 114.4 | 3% slower |
| 384 | 168.2 | 163.2 | 3% faster |
| 512 | 210.6 | 213.9 | ~same |

**Conclusion:** K/V interleaving alone provides minimal benefit — 2-5% at some cache lengths, within noise at others, and occasionally slightly slower. The doubled stride (2048 vs 1024 bytes between positions) offsets whatever K/V adjacency gains there are. The prefetcher hypothesis from the design — that V data would arrive "for free" in the same cache line as K — doesn't hold in practice because the block loop processes all K scores for a block first, then all V accumulations. By the time V is needed, the prefetcher has moved on and the V cache lines from K-loading are likely evicted.

#### Exp 7b: Transposed + Packed (Combined)

Cold-cache results (64MB cycling, fine-grained sweep):

| cache_len | Baseline us | Transposed us | Packed us | Trans. vs base | Packed vs base |
|---|---|---|---|---|---|
| 8 | 18.2 | 17.7 | 17.8 | 3% faster | ~same |
| 16 | 17.6 | 16.8 | 16.8 | 5% faster | 5% faster |
| 32 | 25.8 | 23.8 | 24.0 | 8% faster | 7% faster |
| 48 | 31.9 | 28.8 | 28.9 | **10% faster** | **9% faster** |
| 64 | 38.4 | 34.8 | 36.6 | **9% faster** | 5% faster |
| 80 | 48.1 | 41.7 | 41.3 | **13% faster** | **14% faster** |
| 96 | 52.0 | 47.3 | 48.1 | **9% faster** | **7% faster** |
| 128 | 62.4 | 57.9 | 60.2 | **7% faster** | 4% faster |
| 192 | 90.0 | 84.7 | 83.6 | 6% faster | **7% faster** |
| 256 | 110.9 | 107.4 | 107.4 | 3% faster | 3% faster |
| 384 | 168.2 | 157.4 | 156.9 | 6% faster | **7% faster** |
| 512 | 210.6 | 203.5 | 204.7 | 3% faster | 3% faster |

**Conclusion:** Transposed+packed tracks the transposed-only variant (Exp 5) almost exactly. Adding K/V interleaving on top of the head-first layout provides no additional measurable benefit — the gains come entirely from the reduced stride (head-first layout), not from K/V adjacency. This makes sense: with head-first layout, the stride between positions is already `2 * head_dim = 256 bytes` (packed) vs `head_dim = 128 bytes` (transposed-only). The doubled stride from interleaving slightly hurts the position-to-position locality that the transposed layout optimized, offsetting whatever K/V adjacency benefit exists.

**Overall Exp 7 conclusion:** K/V interleaving does not produce meaningful gains in isolation or in combination with the transposed layout. The hypothesis that V data would prefetch "for free" alongside K data is not validated. The block loop structure — which computes all K scores for a block before touching V — creates enough temporal distance between K and V loads that the V data doesn't benefit from K/V adjacency at the cache line level. The transposed layout (Exp 5) remains the better access pattern optimization, with its gains coming from reduced position-to-position stride rather than K/V co-location.

### Exp 8: Streaming Stores for Prefill

**What:** Use non-temporal stores (`stnp`) for the final output write in the prefill NEON kernel. Normal stores allocate a cache line (read-for-ownership) even though the output is write-once and never re-read by the kernel. Streaming stores bypass the cache, freeing cache lines for K/V input data that other query positions still need.

**Why prefill specifically:** The output is large relative to the input. For seq_len=1024: output = `1024 * 32 * 128 * 2B = 8MB`, K+V input = `1024 * 8 * 128 * 2B = 4MB`. Every output cache line evicts K/V data that other query positions are reusing. In decode the output is tiny (8KB vs 1MB+ KV cache), so the effect would be negligible.

**How:** Two new functions:
- `cactus_attention_f16_neon()` — the existing NEON fast path exposed as a public API, bypassing the Accelerate dispatch (to enable apples-to-apples comparison on Apple Silicon)
- `cactus_attention_f16_neon_streaming()` — identical but replaces the final `vst1q_f16` output loop with paired `stnp` (Store Non-temporal Pair) via inline assembly, writing two 128-bit Q registers (32 bytes) per instruction

Benchmark variants `cactus_prefill_neon` and `cactus_prefill_streaming` both call the NEON path directly, skipping the Accelerate dispatch at seq_len >= 64.

Results (seq_len sweep, head_dim=128, q_heads=32, kv_heads=8):

| seq_len | NEON us | Streaming us | Change |
|---|---|---|---|
| 64 | 222.6 | 228.8 | 3% slower |
| 128 | 777.1 | 873.4 | 12% slower |
| 256 | 2874.4 | 2899.1 | ~same |
| 512 | 11344.9 | 11884.7 | 5% slower |
| 1024 | 48981.4 | 48017.2 | 2% faster |
| 2048 | 188847.9 | 194305.8 | 3% slower |

**Conclusion:** Streaming stores provide no benefit and are consistently slightly slower. The hypothesis that output writes evict useful K/V cache lines is not validated at the NEON kernel level.

Several factors explain this:

- **Parallelization dilutes the effect.** The NEON path parallelizes over `batch * q_heads * seq_len` = 32,768 work items for seq_len=1024. Each work item writes only `head_dim * 2B = 256 bytes` of output — exactly 2 cache lines on Apple Silicon. The cache pressure from output stores per thread is negligible regardless of store type.
- **`stnp` overhead.** Non-temporal stores use a small write-combining buffer and bypass cache entirely, including L1. On Apple Silicon's deep memory hierarchy, this may increase effective write latency compared to normal stores that hit L1/L2. The cache pollution saved doesn't offset the slower write path.
- **K/V reuse is per-query, not cross-query.** Each work item reads K/V for one query position. With `batch * q_heads * seq_len` work items divided across threads, different threads are reading different portions of K/V. The output writes from one thread are unlikely to evict K/V data that *the same thread* needs, because the output and K/V are in completely different address ranges.
- **Prefill is compute-bound, not bandwidth-bound.** The Accelerate path achieves 2000+ GFLOPS via cblas_sgemm (which uses AMX), while the NEON path maxes out at ~400 GFLOPS. The NEON path is limited by arithmetic throughput on the NEON units, not by memory bandwidth. Optimizing the store path doesn't help when the bottleneck is compute.

**Takeaway:** Streaming stores are not beneficial for attention on Apple Silicon. The kernel is either compute-bound (prefill via NEON) or bandwidth-bound on *loads* (decode), and in neither case are output stores the bottleneck. The Accelerate/AMX path for prefill is ~5x faster than NEON and is the correct production path on Apple.

### Exp 9: Stack-Allocated Accumulators

**What:** Replace heap-allocated `std::vector<float32x4_t>` accumulators with fixed-size stack arrays in the hybrid INT8/FP16 NEON decode path.

**Why:** Lines 1249-1251 allocate `output_accum_low`, `output_accum_high`, and `block_scores` as `std::vector`:

```cpp
std::vector<float> block_scores(BLOCK_SIZE);                           // 32 floats = 128 bytes
std::vector<float32x4_t> output_accum_low(head_dim_aligned / 8 * 2);  // heap
std::vector<float32x4_t> output_accum_high(head_dim_aligned / 8 * 2); // heap
```

Every V accumulation FMA (lines 1404-1405) loads from and stores to the heap vector:
```cpp
output_accum_low[idx] = vfmaq_f32(output_accum_low[idx], v_low, weight_vec);
output_accum_high[idx] = vfmaq_f32(output_accum_high[idx], v_high, weight_vec);
```

For head_dim=128, that's 32 load+store pairs per KV position × 32 positions per block = 1024 round-trips to memory for the accumulators alone. With stack arrays, the compiler can keep the accumulators in registers. AArch64 has 32 NEON registers — head_dim=128 needs 16 pairs (low+high) = exactly 32 registers. The compiler may not keep all of them in registers, but stack arrays in a known-size frame are far more likely to stay in L1 than heap allocations.

The softmax correction loop (lines 1360-1363) also walks the heap accumulators:
```cpp
for (size_t i = 0; i < output_accum_low.size() / 2; ++i) {
    output_accum_low[i] = vmulq_n_f32(output_accum_low[i], scale_correction);
    output_accum_high[i] = vmulq_n_f32(output_accum_high[i], scale_correction);
}
```

**How:** Replace with fixed-size stack arrays sized for the maximum supported head_dim (128):

```cpp
static constexpr size_t MAX_HEAD_DIM = 128;
static constexpr size_t MAX_ACCUM_SLOTS = MAX_HEAD_DIM / VECTOR_WIDTH;  // 16

float block_scores[BLOCK_SIZE];
float32x4_t output_accum_low[MAX_ACCUM_SLOTS];
float32x4_t output_accum_high[MAX_ACCUM_SLOTS];
```

Total stack footprint: `32*4 + 16*16 + 16*16 = 640 bytes` — fits in a single L1 cache line group and is trivially within stack limits.

**Expected impact:** This changes the V accumulation from heap load+FMA+heap store to register/L1 FMA. On a bandwidth-bound kernel, the impact depends on whether the heap vector was already hot in L1. If yes (likely for short cache_len where the loop is tight), the gain may be small. If the heap allocations compete with KV cache data for L1 space (more likely at medium cache_len), the gain could be 5-15%. The zeroing and softmax correction loops also benefit from predictable stack access.

**How to benchmark:** Modify the existing kernel in-place (or create `cactus_attention_hybrid_int8_fp16_stackaccum()`) and compare against baseline with cold-cache sweep.

Cold-cache results (64MB cycling, `--sweep`, 2048 iterations):

| cache_len | Baseline us | Stackalloc us | Change |
|---|---|---|---|
| 8 | 17.5 | 12.8 | **27% faster** |
| 16 | 17.5 | 15.8 | **10% faster** |
| 32 | 26.1 | 25.6 | 2% faster |
| 64 | 38.5 | 37.6 | 2% faster |
| 128 | 62.5 | 61.3 | 2% faster |
| 256 | 111.4 | 110.6 | ~same |
| 512 | 211.3 | 209.4 | ~same |

**Conclusion:** Consistent small improvements at short cache lengths (8-16: 10-27% faster) where the tighter stack frame helps register allocation and avoids heap overhead. At longer cache lengths (64+) the improvement converges to ~2% or noise — the kernel is bandwidth-bound and the accumulator location doesn't matter when memory stalls dominate. The 27% gain at cache_len=8 is real: with only 8 KV positions, the inner loop is short enough that heap allocation and indirection overhead is a measurable fraction of total time. **Marginal at production-relevant cache lengths but validates the stack-array approach used by later experiments.**

### Exp 10: Per-Element Online Softmax

**What:** Eliminate the block loop and `block_scores` array. Process one KV position at a time (like ggml CPU flash attention). For each `kv_pos`: compute Q*K score, update running max/sum + scale accumulators, immediately accumulate `V * exp(score - max)`. Uses stack arrays from Exp 9.

**Why:** The block loop processes BLOCK_SIZE=32 K positions, stores all scores, computes softmax, then processes V for the same 32 positions. This means V data for position 0 isn't loaded until all 32 K positions are scored — by which time position 0's K data (and the cache lines around it) may be evicted. Per-element processing loads K and V for the same position back-to-back, maximizing temporal locality between K and V data for the same position. The tradeoff is more `expf()` calls (one per position vs one per block for the max-correction) and more accumulator scaling operations.

**Kernel:** `cactus_attention_hybrid_int8_fp16_perelement` / **Bench:** `cactus_decode_perelement`

Cold-cache results (64MB cycling, `--sweep`, 2048 iterations):

| cache_len | Baseline us | Perelement us | Change |
|---|---|---|---|
| 8 | 17.5 | 15.2 | 13% faster |
| 16 | 17.5 | 21.6 | 23% slower |
| 32 | 26.1 | 35.5 | 36% slower |
| 64 | 38.5 | 58.5 | 52% slower |
| 128 | 62.5 | 108.3 | 73% slower |
| 256 | 111.4 | 208.3 | 87% slower |
| 512 | 211.3 | 405.2 | 92% slower |

**Conclusion:** Catastrophically slower at all cache lengths beyond 8, scaling to nearly 2x slower at cache_len=512. The per-element online softmax pays a massive penalty from calling `expf()` and rescaling all accumulators at every KV position instead of once per block of 32. With 512 positions, that's 512 `expf()` + 512 accumulator-scaling passes vs 16 in the blocked version. The tiny gain at cache_len=8 (13%) is from reduced loop overhead with only 8 positions, but this advantage is immediately lost as cache length grows. The blocked approach amortizes the expensive `expf()` and accumulator rescaling over BLOCK_SIZE=32 positions, which is critical for performance. **Reverted — blocking is essential.**

### Exp 11: Deferred Scale in Dequant-Dot

**What:** In the Q*K scoring loop for INT8 K, accumulate raw int-to-float dot product per quantization group without multiplying by the dequantization scale, then apply scale once per group via `score += k_scale * horizontal_sum(raw_accum)`. Saves 32 `vmulq_f32` operations per KV position (replaced by 4 scalar multiplies). Uses stack arrays from Exp 9.

**Why:** The baseline dequantizes each 8-element INT8 vector by broadcasting the group scale and multiplying before the FMA. With 4 quant groups of 32 elements each (4 inner iterations of 8 elements), that's `4 groups × 4 iters × 2 vmulq_f32 = 32` vector multiplies just for scale application. By deferring the scale to after the horizontal reduction, we replace these with `4 groups × 1 scalar mul = 4` scalar operations. The V accumulation path is unchanged since it needs per-element dequantized values.

**Kernel:** `cactus_attention_hybrid_int8_fp16_deferscale` / **Bench:** `cactus_decode_deferscale`

Cold-cache results (64MB cycling, `--sweep`, 2048 iterations):

| cache_len | Baseline us | Deferscale us | Change |
|---|---|---|---|
| 8 | 17.5 | 12.5 | **29% faster** |
| 16 | 17.5 | 16.5 | 6% faster |
| 32 | 26.1 | 23.6 | **10% faster** |
| 64 | 38.5 | 35.0 | **9% faster** |
| 128 | 62.5 | 57.7 | **8% faster** |
| 256 | 111.4 | 104.2 | **6% faster** |
| 512 | 211.3 | 200.1 | **5% faster** |

**Conclusion:** Consistent 5-10% improvement across all cache lengths, with a larger 29% gain at cache_len=8 (where stack allocation also helps — deferscale includes stack arrays). The deferred scale saves 32 `vmulq_f32` per KV position in the Q*K scoring path, replacing them with 4 scalar multiplies + 4 horizontal sums. At short cache lengths the compute savings are directly measurable. At longer cache lengths (128-512) the improvement persists at 5-8%, suggesting that reducing instruction pressure in the Q*K loop frees NEON execution resources for the memory subsystem. Unlike Exp 1-3's arithmetic optimizations which showed noisy/inconsistent results, deferscale's gains are directionally consistent across all sizes. **Worth keeping — the simplest arithmetic optimization that actually produces consistent improvement on a bandwidth-bound kernel.**

### Exp 12: GQA-Aware KV Loading

**What:** Restructure parallelism from `batch * num_q_heads` (32 items) to `batch * num_kv_heads` (8 items). Each work item processes all 4 Q heads sharing one KV head. Per block: load K once, score all 4 Q heads, per-head softmax, load V once, accumulate for all 4 Q heads. Uses `set_attention_config(1, 1)` to force 8 threads.

**Why:** The baseline loads K and V data once per Q head. With GQA (4 Q heads per KV head), the same K/V data is loaded 4 times by 4 different work items. By grouping all Q heads for the same KV head into one work item, K and V are loaded only once and reused for all 4 Q heads' scoring and accumulation. This reduces total KV memory traffic by ~4x in theory, though the larger per-thread working set (~2.5KB for 4 heads' worth of accumulators and block_scores) may increase cache pressure.

**Kernel:** `cactus_attention_hybrid_int8_fp16_gqa` / **Bench:** `cactus_decode_gqa`

Cold-cache results (64MB cycling, `--sweep`, 2048 iterations):

| cache_len | Baseline us | GQA us | Change |
|---|---|---|---|
| 8 | 17.5 | 27.9 | 59% slower |
| 16 | 17.5 | 27.0 | 54% slower |
| 32 | 26.1 | 31.5 | 21% slower |
| 64 | 38.5 | 35.5 | **8% faster** |
| 128 | 62.5 | 45.7 | **27% faster** |
| 256 | 111.4 | 62.5 | **44% faster** |
| 512 | 211.3 | 111.6 | **47% faster** |

**Conclusion:** Sharp crossover at cache_len ~48-64. At short cache lengths (8-32), the reduced parallelism (8 work items vs 32) and larger per-thread working set (~2.5KB for 4 heads of accumulators + block_scores) dominates, causing 21-59% slowdowns. At cache_len=64 the crossover begins (8% faster), and at long cache lengths (128+) the 4x reduction in KV memory traffic produces massive speedups (27-47%).

The crossover behavior mirrors Exp 6b (highthread): at cache_len=128, GQA (27% faster) matches highthread (27% faster with 2048 iters). At cache_len=256, GQA (44%) underperforms highthread (46%), and at cache_len=512 GQA (47%) underperforms highthread (60%). The mechanisms are different — GQA reduces total bytes loaded while highthread increases memory-level parallelism — but both produce similar magnitude improvements.

**Key insight:** GQA-aware loading is the first optimization that reduces the *amount* of data loaded, not just the compute or access pattern. At cache_len=512, the baseline loads `512 * 128 bytes * 32 Q heads = 2MB` of K data (each Q head re-loads its KV head's data). GQA loads `512 * 128 bytes * 8 KV heads = 0.5MB` — a 4x reduction. This is why GQA produces the largest gains at long contexts where the kernel is most deeply bandwidth-bound.

**Combining with highthread:** GQA + higher thread count is untested but promising. GQA's 8 work items with `config(1,1)` give 8 threads. A split-K variant within GQA (splitting each KV head's KV range across multiple threads) could combine the 4x KV reuse with higher memory-level parallelism, potentially compounding both advantages.

### Exp 13: Software Prefetch + Deferscale

**What:** Add explicit `__builtin_prefetch` for K and V data 4 positions ahead in the block loop, combined with deferred scale from Exp 11. The prefetch issues a non-blocking DRAM request during compute for the current position, so future positions' data may arrive before it's needed.

**Why:** In cold-cache conditions, each KV position requires loading 128 bytes of K data from DRAM (~200 cycle latency) followed by ~35 cycles of compute. The hardware prefetcher should detect the stride pattern (1024 bytes between positions), but in the cold-cache benchmark each invocation uses unique data, so the prefetcher may not warm up within a single call. Explicit software prefetch starts the DRAM request during the current position's compute, hiding latency.

**Kernel:** `cactus_attention_hybrid_int8_fp16_prefetch` / **Bench:** `cactus_decode_prefetch`

Cold-cache results (64MB cycling, `--sweep`, 2048 iterations):

| cache_len | Baseline us | Prefetch us | Change |
|---|---|---|---|
| 8 | 16.9 | 11.7 | **31% faster** |
| 16 | 16.7 | 14.3 | **14% faster** |
| 32 | 24.8 | 23.8 | 4% faster |
| 64 | 39.7 | 35.3 | **11% faster** |
| 128 | 63.6 | 58.9 | **7% faster** |
| 256 | 112.9 | 106.2 | **6% faster** |
| 512 | 211.8 | 198.9 | **6% faster** |

**Conclusion:** Consistent 6-14% improvement at medium-to-long cache lengths, similar magnitude to deferscale alone (which prefetch includes). The prefetch instructions provide a small additional benefit on top of deferscale at short cache lengths (31% vs 29% at cache_len=8), but the gains converge at longer lengths — suggesting the hardware prefetcher warms up and handles the stride pattern after a few iterations regardless of explicit hints. The explicit prefetch mainly helps the first few positions per block where the hardware prefetcher hasn't yet detected the pattern.

### Exp 14: 2-Position Interleaved Scoring

**What:** Process 2 KV positions per inner loop iteration with interleaved loads and FMAs. Load K data for both positions at the top of the iteration, then alternate FMA instructions between them. This doubles outstanding memory requests per iteration, giving the load-store unit more requests to pipeline. Combined with deferscale.

**Why:** The baseline's inner loop issues loads for one KV position, waits for data, computes, then moves to the next. The out-of-order engine can overlap some loads with compute, but with a single position's loads completing before the next is issued, the memory pipeline may be underutilized. By loading two positions' data simultaneously, the load-store unit has twice as many outstanding requests, potentially better saturating the memory controller.

**Kernel:** `cactus_attention_hybrid_int8_fp16_interleaved` / **Bench:** `cactus_decode_interleaved`

Cold-cache results (64MB cycling, `--sweep`, 2048 iterations):

| cache_len | Baseline us | Interleaved us | Change |
|---|---|---|---|
| 8 | 16.9 | 11.7 | **31% faster** |
| 16 | 16.7 | 15.5 | 7% faster |
| 32 | 24.8 | 23.1 | 7% faster |
| 64 | 39.7 | 35.4 | **11% faster** |
| 128 | 63.6 | 57.7 | **9% faster** |
| 256 | 112.9 | 104.1 | **8% faster** |
| 512 | 211.8 | 194.8 | **8% faster** |

**Conclusion:** The best single-kernel result at equal threading — consistent 7-9% improvement at medium-to-long cache lengths, slightly outperforming both deferscale (5-6%) and prefetch (6-7%) at 256-512. The interleaving creates more instruction-level parallelism in the inner loop, allowing the OoO engine to better overlap memory loads with computation. The 2-position processing doubles the working set of live registers but head_dim=128 fits within AArch64's 32 NEON registers. The improvement is modest because Apple Silicon's OoO engine is already good at reordering loads — the explicit interleaving just helps at the margin.

### Exp 15: Small Block Size (BLOCK_SIZE=8)

**What:** Reduce BLOCK_SIZE from 32 to 8. Combined with deferscale. This reduces the temporal gap between K scoring and V accumulation for the same positions — with 8 positions per block, V data is loaded ~8 scoring iterations after K, vs ~32 iterations with the baseline. The tradeoff is 4x more online softmax max-correction operations (one per block).

**Why:** The block loop scores all K positions in a block before loading any V data. With BLOCK_SIZE=32 and 1024-byte stride, 32 K loads span ~32KB of address space. By the time V accumulation starts, the K cache lines and any V lines the hardware prefetcher may have speculatively loaded could be evicted from L1/L2. With BLOCK_SIZE=8, only 8 K positions (~8KB) are scored before V accumulation, keeping V data temporally closer to the corresponding K data. The cost is 4x more `expf()` calls for the softmax max-correction (once per 8-position block vs once per 32).

**Kernel:** `cactus_attention_hybrid_int8_fp16_smallblock` / **Bench:** `cactus_decode_smallblock`

Cold-cache results (64MB cycling, `--sweep`, 2048 iterations):

| cache_len | Baseline us | Smallblock us | Change |
|---|---|---|---|
| 8 | 16.9 | 13.3 | **21% faster** |
| 16 | 16.7 | 15.9 | 5% faster |
| 32 | 24.8 | 23.5 | 5% faster |
| 64 | 39.7 | 36.2 | **9% faster** |
| 128 | 63.6 | 59.4 | **7% faster** |
| 256 | 112.9 | 107.3 | 5% faster |
| 512 | 211.8 | 201.2 | 5% faster |

**Conclusion:** Similar to deferscale alone (which smallblock includes). The reduced block size doesn't provide measurable K/V locality benefit on top of the deferscale gains. At 256-512, smallblock (5%) slightly underperforms deferscale (5-6%), likely because the 4x more softmax overhead (expf + accumulator scaling per 8 positions instead of per 32) offsets any V locality improvement. The cold-cache benchmark cycles through 64MB of state data, so V cache lines from K-loading time are evicted regardless of block size — the working set far exceeds L2.

### Exp 16: Wider INT8 Loads (vld1q_s8)

**What:** Use 128-bit `vld1q_s8` (16 INT8 elements per load) instead of 64-bit `vld1_s8` (8 elements) in both K scoring and V accumulation inner loops. Combined with deferscale. This halves the number of K load instructions and reduces the inner loop from 4 iterations to 2 per quant group.

**Why:** The baseline issues two `vld1_s8` loads per 16 K elements. A single `vld1q_s8` replaces both, reducing load instruction count and loop overhead (fewer branches, fewer index calculations). The total bytes loaded from DRAM are identical — this optimizes instruction-side overhead, not bandwidth.

**Kernel:** `cactus_attention_hybrid_int8_fp16_wide` / **Bench:** `cactus_decode_wide`

Comparison vs deferscale (10k iterations, cache_len=32 re-run in isolation to correct sweep anomaly):

| cache_len | Deferscale us | Wide us | Wide vs DS |
|---|---|---|---|
| 8 | 11.9 | 12.0 | ~same |
| 16 | 15.7 | 16.7 | 6% slower |
| 32 | 22.7 | 23.1 | ~same |
| 64 | 41.9 | 39.4 | 6% faster |
| 128 | 58.9 | 59.6 | ~same |
| 256 | 109.4 | 107.6 | 2% faster |
| 512 | 201.5 | 200.7 | ~same |

**Conclusion:** Within noise everywhere. Wider loads don't add anything measurable on top of deferscale. The 128-bit `vld1q_s8` loads the same bytes from DRAM as two 64-bit `vld1_s8` — the memory controller doesn't care about instruction granularity, only total bytes requested. The halved loop iterations save a few branch/index instructions per quant group, but this is negligible on a bandwidth-bound kernel. **Not worth the added complexity.**

### Exp 13-16 Summary

All four microarchitectural optimizations (prefetch, interleaving, small blocks, wide loads) produce improvements in the same 5-9% range vs baseline at production cache lengths. They all include deferscale, so the comparison against baseline includes that ~5% gain. The incremental benefit of each technique on top of deferscale is 0-3%, confirming that Apple Silicon's OoO engine and hardware prefetcher already do a good job at hiding latency for the stride pattern in the inner loop.

**Interleaved (Exp 14) is marginally the best** at 9% vs 6% (deferscale) at cache_len=512 with 10k iterations, likely because it creates genuinely independent load streams — the OoO engine sees two positions' loads simultaneously rather than sequentially. This is the only technique that provides a consistent ~3% gain on top of deferscale.

### Accelerate BLAS Disassembly Analysis

To understand why Apple's Accelerate framework achieves 5x higher throughput than our NEON kernels on prefill, we disassembled `cblas_sgemm` from the Accelerate framework's vecLib (`libBLAS.dylib` in the dyld shared cache).

**Finding 1: AMX hardware on the main path.** The primary GEMM codepath uses Apple's AMX (Apple Matrix eXtension) instructions — proprietary undocumented opcodes that the disassembler renders as `.long` directives (e.g., `.long 0xe1206200` at offset 0x6c710 in the library). AMX is a separate coprocessor from the NEON units, with dedicated matrix multiply hardware that achieves ~16x NEON throughput for FP32 GEMM. This is not accessible via public APIs or compiler intrinsics — Accelerate has exclusive access through undocumented system calls.

**Finding 2: NEON fallback uses 8x6 outer-product microkernel.** For cases where AMX isn't used (small matrices, specific shapes), the library falls back to a NEON kernel with a carefully tuned **outer product** inner loop:

```asm
// Inner loop: 8x6 register-blocked outer product
ld1.4s  { v12, v13 }, [A], stride    // Load 8 floats of A column (2 regs)
ld1.s   { v14 }[0], [B], #4          // Load 6 floats of B row (scattered)
ld1.s   { v14 }[1], [B], stride      //   using lane insertion with stride
ld1.s   { v14 }[2], [B], #4
ld1.s   { v14 }[3], [B], stride
ld1.s   { v15 }[0], [B], #4
ld1.s   { v15 }[1], [B], stride

fmla.4s v16, v12, v14[0]   // C[0:4, 0] += A[0:4] * B[0]  ← lane broadcast
fmla.4s v17, v13, v14[0]   // C[4:8, 0] += A[4:8] * B[0]
fmla.4s v18, v12, v14[1]   // C[0:4, 1] += A[0:4] * B[1]
fmla.4s v19, v13, v14[1]   // C[4:8, 1] += A[4:8] * B[1]
fmla.4s v20, v12, v14[2]   // C[0:4, 2] += A[0:4] * B[2]
fmla.4s v21, v13, v14[2]   // C[4:8, 2] += A[4:8] * B[2]
fmla.4s v22, v12, v14[3]   // C[0:4, 3] += A[0:4] * B[3]
fmla.4s v23, v13, v14[3]   // C[4:8, 3] += A[4:8] * B[3]
fmla.4s v24, v12, v15[0]   // C[0:4, 4] += A[0:4] * B[4]
fmla.4s v25, v13, v15[0]   // C[4:8, 4] += A[4:8] * B[4]
fmla.4s v26, v12, v15[1]   // C[0:4, 5] += A[0:4] * B[5]
fmla.4s v27, v13, v15[1]   // C[4:8, 5] += A[4:8] * B[5]

subs    x9, x9, #1         // K loop counter
b.ne    loop
```

The key technique is **lane-broadcast outer product**: `fmla.4s v16, v12, v14[0]` multiplies all 4 elements of v12 by a single scalar (lane 0 of v14). Each K-loop iteration:
- **Loads 14 elements** (8 from A + 6 from B)
- **Computes 48 multiply-adds** (12 FMLA instructions × 4 lanes each)
- **3.4 FMAs per loaded value** — very high arithmetic intensity

Register allocation: 2 registers for A tile, 2 for B tile, 12 for C accumulator (v16-v27) = 16 of 32 NEON registers. The accumulators stay in registers across the entire K reduction, eliminating all accumulator memory traffic.

**Why these techniques don't transfer to decode attention:**

The fundamental difference is **outer product vs inner product**:

| | GEMM (Accelerate) | Decode Attention (our kernel) |
|---|---|---|
| **Operation** | C[M,N] += A[M,K] × B[K,N] | score = Q[D] · K[D] |
| **Output per K-step** | M×N elements (2D tile) | 1 scalar |
| **Data reuse** | Each A element reused N times, each B element M times | K data used once, discarded |
| **Arithmetic intensity** | O(min(M,N)) FMAs per load | O(1) FMAs per load |
| **Bottleneck** | Compute-bound | Memory-bandwidth-bound |

GEMM achieves high throughput by computing a 2D output tile — each loaded element contributes to multiple output values. Decode attention computes a 1D dot product — each K element produces one partial sum toward a single scalar score. No restructuring of the inner loop can change this fundamental arithmetic intensity gap. The 8x6 register blocking and lane broadcasting are specifically optimized for the 2D reuse structure of matrix multiplication.

The only way to increase arithmetic intensity in decode attention is to share loaded data across multiple outputs — which is exactly what GQA (Exp 12) does by scoring 4 Q heads per K load. But at equal thread count, GQA's 4x compute increase per work item outweighed its 4x bandwidth reduction (Exp 12 re-benchmark without thread override).

**Conclusion:** Accelerate's decode-path advantage comes entirely from AMX hardware (unavailable to custom kernels) and the algorithmic structure of GEMM (which doesn't apply to dot-product attention). The ~8% we achieved from deferscale + interleaving represents the genuine optimization ceiling for NEON decode attention at 2 threads. Further gains require either more threads, less data (INT4 KV cache), or access to AMX/matrix hardware.

### Upcoming Experiments

| # | Optimization | Impact | Effort | Rationale |
|---|---|---|---|---|
| 1 | INT4 KV cache | High | Medium | Halves data loaded per token → direct bandwidth reduction. INT4 infra exists for weights but not KV cache yet. Compounds with interleaved gains. |
| 2 | Increase attention thread count | High | Low | One-line config change `{1, 4}` gives 40-60% at long cache (Exp 6b). Needs end-to-end validation for MLP/matmul contention. |
| 3 | Focus on Android | Medium | Low | Smaller L2 (256-512KB), heterogeneous cores, lower bandwidth — optimizations may have a different impact profile. |
| 4 | End-to-end validation | Medium | Low | Measure full token generation latency (30-layer model) to confirm kernel-level gains translate to production. |
