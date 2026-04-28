# K=96 Grouped-Experts MoE for Gemma-4 E2B

This directory ships the converter and runtime support for the post-trained Gemma-4 E2B "K=96 grouped-experts" MoE checkpoint hosted at `Cactus-Compute/gemma4-e2b-grouped-k96`. Each layer's MLP neurons are partitioned into 96 contiguous clusters; per token only `K_active=48` clusters fire (50% MLP density). The model was distilled from a dense Gemma-4 E2B in `matryoshka-distil`.

## Final performance (MacBook Pro, M-series, 8 P-cores, steady-state decode)

| Variant | tok/s | Speedup |
|---|---|---|
| Dense INT8 | 33 | 1.00× |
| K96 packed INT8 | 41 | 1.23× |
| Dense INT4 | 37 | 1.00× |
| **K96 packed INT4** | **49** | **1.32–1.36×** |

INT4 K96-packed is the headline path. Steady-state was measured as the average of 7 consecutive `chat ... --prompt "Tell me a 200-word story about a robot."` runs after dropping the first (cold-cache) iteration with 15s cooldowns between.

## Files

- `convert_k96.py` — converts the upstream K96 checkpoint into the cactus packed weight format. Merges base + LoRA, applies the offline cluster permutation, rounds cluster offsets to multiples of 32, and writes per-layer `[up_c | up_c_scales | down_c | down_c_scales]` slabs.
- `verify_converted.py` — sanity-checks header magic, cluster count, sums of cluster widths.
- `make_fake_offsets.py` — emits synthetic cluster offsets for kernel-only benches.
- `run_e2e.sh` — convenience driver: download + convert + run a sample prompt.

## Runtime path

When `k96.meta` includes `k96_packed=1`, `cactus/models/gemma4/model_gemma4.cpp` loads the per-layer packed file and routes the FFN through `OpType::GROUPED_MLP_INT8` (which dispatches INT4 vs INT8 internally). The graph op is `compute_grouped_mlp_int8_packed` in `cactus/graph/graph_ops_nn.cpp`; per token it:

1. Quantizes the activation row to INT8.
2. Runs `gate_proj` once (full d_ffn — needed for routing AND the gate term).
3. Computes per-cluster max-abs of `GeLU(gate_proj)` and selects the top-K_active clusters.
4. Per-active-cluster, on a thread-pool worker (atomic work-stealing on `next_cluster`):
   - `up_proj` GEMV against the cluster's `up_c` slab → `up_part` (size `N_c`).
   - `h = gate * up_part` with per-cluster max-abs in fp16 NEON.
   - Quantize `h` to INT8 with the cluster's own scale.
   - `down_proj` GEMV against the cluster's `down_c` slab → fp16 partial.
   - Accumulate fp16 partial into per-worker fp32 `y_acc`.
5. Reduce per-worker `y_acc` into the layer output.

## What was tried (and what worked)

The journey from the initial 1.07× speedup to today's 1.32–1.36× is documented because most of the lessons are non-obvious.

### Wins (kept)

| Change | Effect | Location |
|---|---|---|
| Per-cluster contiguous packed weights `[up_c \| down_c]` | 1.07× → 1.20× INT8, 1.10× → 1.27× INT4 | `convert_k96.py` + `model_gemma4.cpp` loader + new graph op |
| Atomic work-stealing across clusters | Replaced static partitioning that left workers idle on big clusters | `compute_grouped_mlp_int8_packed` |
| Fused vectorized h-compute + quantize (NEON fp16 → int8) | Collapsed ~9% scalar pass to ~0.6% | `compute_grouped_mlp_int8_packed` |
| Per-block (not per-run) thread parallelism in selective non-packed kernels | Removed `min(threads, num_runs)` cap that left cores idle | `cactus_gemv_int{4,8}_active_block_runs` |

### Regressions (kept as opt-in env vars for future hardware)

These all *seemed* like wins on paper and all empirically lost on M-series. Each is preserved behind an env var.

| Idea | Why it lost | Env var |
|---|---|---|
| **N-block-outer / cluster-inner** down GEMV (concat-and-flatten) | Per N-block we touched 48 disjoint weight regions ~50KB apart. Hardware prefetcher pulls 256-byte cache lines but consumed only 64-128 B per cluster before jumping → ~50% wasted bandwidth. **0.94× speedup vs dense.** | `K96_MULTI_KSEG=1` |
| **Two-phase split-down** (cluster-parallel up+h+q, barrier, N-block-parallel down) | Extra thread-pool dispatch costs 50–100 µs per layer. Tail recovery saved less than that. **1.23×.** | `K96_SPLIT_DOWN=1` |
| **Sub-cluster split** (split big clusters into 32-aligned 64-row sub-tasks) | Down-GEMV's per-call setup (n-block iteration, scale loads) doesn't shrink with k_count → smaller K-slices have worse arithmetic intensity. **~1.24-1.29× (neutral or slight regression).** | `K96_SUB_SPLIT=1` |
| **Fused gate_proj + GeLU + cluster-max in one sweep** | Saving was only the ~5 µs/layer post-pass. Gate_proj itself is bandwidth-bound; adding GeLU+max into the same threads doesn't reduce traffic. **1.25-1.28× (within noise, slight regression).** | `K96_FUSED_GATE_ROUTE=1` |
| **Reduced K_active at decode** | K_active=44 stays coherent for ~1 tok/s gain. K_active≤42 frequently produces 1024-token gibberish runaway. Marginal at best. | `K96_K_ACTIVE_OVERRIDE=44` |
| **Approximate routing on K-subset of input dims** | Partial gate values broke GeLU dynamic range → gibberish. Recomputing full gate per active cluster ate the savings. Validated dead-end. | (not exposed) |
| **Cluster-sort-by-size for LPT scheduling** | Worse cache behavior (more total worker-time spent), no wall-time win. Reordering hurt TLB locality. | (not exposed) |

## Lessons learned

1. **Per-cluster sequential weight streaming is the locality superpower.** Each worker reads one cluster's ~49 KB of `down_w` linearly through L2; that pattern is exactly what hardware prefetchers expect. Any restructure that crosses cluster boundaries within a worker pays a ~50% bandwidth penalty.
2. **Down-GEMV's per-call cost is N-axis bound, not K-axis bound.** The kernel walks all 384 N-blocks (`hidden_dim/4`) per call regardless of how many K-groups it iterates. Splitting clusters in K direction makes per-byte arithmetic intensity worse.
3. **Thread-pool dispatch overhead is real.** ~50–100 µs/layer per dispatch on a small pool. With 35 layers that's 1.75–3.5 ms/token, ~5–10% of decode time. Anything that doubles dispatch counts has to save more than this — and "tail recovery" alone doesn't.
4. **The cluster-tail problem is smaller than estimated.** Active cluster sizes are typically more uniform than the worst-case histogram suggests; 24% idle was an upper bound, not the median.
5. **gate_proj is the unbreakable wall.** It's 33% of MLP weight bytes and we read every byte every token because routing is computed *from* it. No way around that without a separate cheap router (model retraining required).

## Why we're at ~1.36× and not 1.95×

The 1.95× ceiling presumed full 50% MLP byte savings. Real savings are 33% because `gate_proj` cannot be skipped — its output IS the routing signal:

```
MLP bytes after K96 = (100% gate + 50% up + 50% down) / 3 = 67% of dense
```

With MLP at ~78% of decode bandwidth, the bandwidth ceiling is `1 / (1 − 0.78 × 0.33) ≈ 1.346×`. We are at the ceiling. To break it, the model would need a separate router head (e.g., 96 × hidden_dim, ~150 KB) trained alongside the K96 split, so `gate_proj` could be skipped on inactive clusters. Theoretical ceiling with a cheap router would be ~1.64×. That's a `matryoshka-distil` change, not a cactus change.

## Conversion

```bash
python -m python.src.k96.convert_k96 \
    --src weights/k96_raw \
    --base-cactus weights/gemma-4-e2b-it \
    --out weights/gemma-4-e2b-it-k96-packed \
    --precision INT4
```

Then run with `chat weights/gemma-4-e2b-it-k96-packed --prompt "..."`.

## Env-var quick reference

All env vars below default OFF; set to `1` to opt in. They exist for benchmarking on different CPUs.

- `K96_PROF=1` — print mean per-op timings every 600 calls.
- `K96_PROF_PCL=1` — finer per-cluster phase timings (with `K96_PROF`).
- `K96_SUB_SPLIT=1` — split big clusters into 32-aligned sub-tasks.
- `K96_FUSED_GATE_ROUTE=1` — fuse gate_proj GeLU + per-cluster max-abs into one sweep.
- `K96_SPLIT_DOWN=1` — two-phase up/down split with barrier.
- `K96_MULTI_KSEG=1` — concat-flatten multi-segment down GEMV.
- `K96_K_ACTIVE_OVERRIDE=N` — override K_active at decode (default 48).
- `K96_SUB_SPLIT_THRESH=N`, `K96_SUB_SPLIT_TARGET=N` — tune sub-split chunking.
