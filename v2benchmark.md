# V2Bench — End-to-End Decode-TPS Benchmark

End-to-end generation benchmark from the `int4-benchmark` branch
(`tests/bench/e2e_bench.cpp` + `e2e_run_all.sh`). Measures real model
prefill/decode throughput, comparing **cactus** against llama.cpp, ONNX Runtime,
ExecuTorch, and LiteRT.

- **Cactus precision:** CQ4 (Cactus CQ4 weights).
- **Models:** `qwen3-0.6b`, `gemma3-270m`.
- **Decode budget:** `--max-tokens 512` (greedy: temp 0, top_k 1).
- **Prompt:** "Write out the first 1k tokens of Romeo and Juliet's 1st chapter:".
- **Rounds:** 10 measured rounds per backend (1 warmup discarded), interleaved
  round-robin scheduling.
- All values are means across the 10 rounds. `dec_tps` is the headline metric.

## Mac (Apple M4 Pro — 10P+4E, 24 GB)

Source: local `tests/bench/e2e_results.csv` (CQ4 run).

### qwen3-0.6b

| Backend     | Prefill TPS | Decode TPS | TTFT (ms) |
|-------------|------------:|-----------:|----------:|
| cactus      |       655.7 |       73.7 |      39.7 |
| llama.cpp   |       581.7 |      143.2 |      31.0 |
| onnxrt      |       912.9 |      122.6 |      20.6 |
| executorch  |        45.5 |       45.5 |     396.3 |
| litert      |       107.8 |       29.7 |     285.0 |

### gemma3-270m

| Backend     | Prefill TPS | Decode TPS | TTFT (ms) |
|-------------|------------:|-----------:|----------:|
| cactus      |      1519.7 |      158.5 |      18.5 |
| llama.cpp   |      1180.6 |      319.3 |      17.0 |
| onnxrt      |      1628.0 |      279.9 |      12.4 |
| executorch  |       103.7 |      103.7 |     192.9 |
| litert      |      1142.7 |      114.9 |      31.0 |

> Notes: ExecuTorch and LiteRT are driven via their Python harnesses
> (`e2e_bench_executorch.py`, `e2e_bench_litert.py`); ExecuTorch reports a single
> throughput value (shown for both prefill and decode). Decode token counts vary
> per backend because non-cactus backends were not all stopped at exactly 512.

## Android

Cactus-only, CPU, 512-token decode budget (`tests/android-e2e`, branch
`android-test`). Engine: **cactus-v2 @ `01e6a704`** (= current `main`; includes
chunked prefill from #686), built from a clean worktree and linked statically
into `e2e_bench`. Cloud handoff and telemetry disabled
(`CACTUS_DISABLE_CLOUD_HANDOFF=1 CACTUS_NO_CLOUD_TELE=1`) — both make network
calls mid-round and pollute timings. Means across 3 measured rounds (1 warmup).

> **Engine format note:** these CQ4 transpiled bundles (header precision 6) are
> only readable by the cactus-v2 engine. The old in-repo engine mis-reads them
> as int8 and segfaults at load — that, not device throttling, caused all
> earlier crashes/slow runs. The earlier "Pixel adb-shell frequency cap" note
> was wrong: with a clean engine link the Pixel runs at full clocks under adb.

### Canonical prompt (same as Mac run; ~30 tokens, padded to one 128-token prefill chunk)

| Device | SoC | Model | Prefill TPS* | Decode TPS | TTFT (ms) |
|--------|-----|-------|-------------:|-----------:|----------:|
| Samsung SM-S942U1 | Snapdragon 8 Elite | qwen3-0.6b     | 50.3 | 37.0 |  597 |
| Samsung SM-S942U1 | Snapdragon 8 Elite | gemma-4-e2b-it | 18.9 | 16.9 | 1498 |
| Pixel 10a         | Tensor G4          | qwen3-0.6b     | 20.8 | 23.7 | 1447 |
| Pixel 10a         | Tensor G4          | gemma-4-e2b-it |  7.2 |  6.7 | 3916 |

*\*Prefill TPS on this prompt is padding-taxed: the transpiled
`decoder_prefill_chunk` graph is fixed at 128 tokens, so a ~30-token prompt
pays a full chunk's compute (≈4.3× understatement). Applies equally to the Mac
numbers above.*

### Long prompt (~548 tokens = 5 prefill chunks; true chunked-prefill throughput)

| Device | Model | Prefill TPS | Decode TPS | TTFT (ms) |
|--------|-------|------------:|-----------:|----------:|
| Samsung | qwen3-0.6b     | 127.6 | 29.7 |  4297 |
| Samsung | gemma-4-e2b-it | 107.4 | 13.8 |  5092 |
| Pixel   | qwen3-0.6b     |  52.2 | 18.3 | 10489 |
| Pixel   | gemma-4-e2b-it |  43.3 |  6.1 | 12594 |

> **Run-state sensitivity:** device state dominates phone numbers. The Samsung
> rows were measured cool and on power (the representative state); back-to-back
> hot runs read 20–30% lower (e.g. qwen3 decode 28.3 hot vs 37.0 cool). An
> interleaved A/B confirmed the May 26 and current-main engines decode within
> noise of each other at matched state (27.6 vs 27.5 tps, Samsung qwen3) — no
> engine regression. Cool-state gemma decode (16.9) matches the best May-era
> scheduler-study recordings (14.6–16.5 @ 100-token decode).
>
> **Threadpool spin fix — measured with/without on both devices, and
> rejected.** The adaptive spin-then-block worker patch (commit 88679112,
> ported to the cactus-v2 pool; spin active only when ≤4 perf-core workers)
> was measured at the canonical spec, both models, both devices
> (decode tps, 512×3):
>
> | Device / model | With fix | Without fix | Δ |
> |---|---:|---:|---|
> | Pixel qwen3-0.6b | 11.4 | 23.7 | −52% |
> | Pixel gemma-4-e2b | 5.1 | 6.7 | −24% |
> | Samsung qwen3-0.6b | 35.7 | 37.0 | −3% (spin gated off) |
> | Samsung gemma-4-e2b | 16.7 | 16.9 | −1% (spin gated off) |
>
> Plus an interleaved A/B (Pixel gemma @545ctx/100dec): 5.4/4.8 with vs 6.6
> without. Spinning splits the power budget on the power-constrained 1+3-core
> Tensor and slows the prime core; the smaller model is hit hardest (more
> parallel sections/s → more spin windows). The Samsung rows demonstrate the
> adaptive gate: at 8 perf-core workers spin disables itself and nothing
> regresses. The earlier "3.8→14 tps" validation was an artifact of an
> ODR-corrupted binary (old-repo kernel objects overriding v2 pool symbols
> with a different class layout). Final configuration: NO spin — the main
> tables above are the no-spin numbers. The kernel_utils.h segfault attributed
> to this patch was root-caused to the engine/bundle format mismatch above,
> not the patch.
>
> **Pixel caveat (unit runs ~35–45% below older recordings):** this unit
> benches uniformly below older Pixel 10a recordings (gemma @512: ~8.7
> single-thread, ~12 multi-core load-aware). Eliminated by experiment, each
> with measurements: engine vintage (May 26 vs current main A/B'd equal),
> threadpool spin (made it worse), scheduling policy (present and active),
> charging (fixed a "Charge connected device" source-role drain; ~5% recovery),
> battery saver (off), thermal (Status 0, cooled), OS update (March build
> throughout), CPU clocks (cpu7 verified 2.94 GHz under load), worker count and
> core selection (sweep: 1 worker on cpu7 5.6 ≈ 1 worker on cpu4 5.7 ≈ 4
> workers 6.9 — core-invariant ⇒ memory-bandwidth-bound ceiling), pool
> overhead, reboot (no change), and screen-on boost (no change). Conclusion:
> the unit's memory subsystem currently delivers ~2/3 of the older recordings'
> bandwidth; confirming why needs a userdebug build (MIF devfreq is unreadable
> unrooted). The older reference numbers are from a different recording
> session/unit state, not reproducible here today. Samsung, same harness and
> binaries, reproduces and exceeds its May recordings.
>
> Side-finding for cactus-v2: pool workers pin to `performance_cores[i % n]`
> in CPU-index order (prime core last), so reduced worker counts select the
> slowest perf cores — 2 workers on the two mid cores scored 2.3 tps decode
> vs 6.9 with 4. Order performance cores by capacity descending.
>
> **Prime-core-only measurement (Pixel, taskset cpu7 + 1 worker):** qwen3
> decode 32.4 canonical / 27.3 @512-seq — **+37–49% over the 4-core default**
> (23.7 / 18.3); prefill no worse (55.4 vs 52.2 @512-seq). gemma: 5.8 / 5.5 —
> −10–13% vs 4-core (6.7 / 6.1). Small-model decode fires hundreds of tiny
> barrier-terminated parallel sections per second; parked mid cores act as
> stragglers, so one fast core wins outright. Suggests size-gated worker
> participation in cactus-v2: small parallel sections → prime worker only,
> large GEMM chunks → all cores.


## Pixel exact-spec fixture results (512-token prefill + 32 decode)

`cactus_benchmark_tokens` fixture (raw token IDs, zero padding,
`cactus-v2/tests/android/cactus_llm_bench.cpp`), gemma-4-e2b-it, Pixel 10a at
optimal device state (100% battery, AC, cool). Engine: cactus-v2 main.

| Config | Prefill TPS | Decode TPS |
|--------|------------:|-----------:|
| default (4 workers, chunk×16)    | **74.5–78.2** | **7.1–7.5** |
| native 1 thread (inline on cpu7) | 68–71 | 5.9 |
| May 26 engine, default           | 72.8–74.7 | 7.1–8.0 |
| chunk×4 / ×2 / ×1                | 56 / 44 / 24 | 6.5 / 5.6 / 3.2 |

Findings, each measured:
- **Prefill parity achieved**: 74.5+ vs the ~70 May multi-core recording;
  single-thread prefill (69) is ~2× the old single-thread recording (35).
- **Decode ceiling ~7.5** on this unit in every configuration; the old 8.7
  (1-thread) / 12.1 (multi) recordings are unreachable today. Both engine
  vintages identical → not software.
- Compute-bound prefill at/above parity while bandwidth-bound decode is
  uniformly ~35% below the old recordings → cores healthy, effective DRAM
  throughput is the limiter. Streaming-read microbench: Pixel cpu7 23–29 GB/s
  vs Samsung prime 69–73 GB/s — the 2.4× ratio matches the cross-device gemma
  decode ratio (16.9 vs 6.7), confirming decode is single-core-bandwidth-bound.
- Pool duty sampling: ~1.3 of 5 threads running on average — the 16× dynamic
  chunking concentrates work on the hot prime; reducing the multiplier is
  monotonically worse (slow mids become section stragglers), so the current
  distribution is optimal given parked mids.
- `cactus_complete` path costs ~4% decode vs the raw-token fixture; its
  prefill readings additionally suffer the padding tax (50.7 apparent vs 74.5
  fixture at the same true workload).

### Pixel decode-ceiling investigation (512-prefill + 32-decode spec)

Stable at spec (cool, charged, standard state): **decode 7.2–7.4, prefill
73–78**. Best observed: **decode 8.84/8.83 with prefill 82–90** in a
platform boost state that appeared in two consecutive runs but resisted
three replication attempts (30s mid-core warm, single 3-min warmer run,
10-min load soak — the soak instead degraded runs to 6.6 via thermals).
Mechanism ledger, all measured:

- **Single-core kernel ceiling:** the interleaved CQ4 GEMV runs ~8.8 GB/s
  effective on cpu7 (vs 26–29 GB/s streaming memcpy on the same core) —
  X4 SIMD issue rate on the vqtbl+vdot LUT-expansion mix is the wall.
  Same kernel on Samsung's Oryon: 2.4× — matches the device decode ratio.
  Single-core ≥9 tps is arithmetically impossible on this silicon today;
  the old 8.7 single recording implies the favorable platform state.
- **Kernel patch (+4%, committed to cactus-v2 `g4-gemv-tuning` f9b47cbe):**
  4-block stream widening + boundary prefetch, 7.01→7.25 multi / 5.76→6.06
  single, test_matmul passes.
- **Scheduling levers all measured dead:** spin (every budget 512–16384,
  monotonic harm), static chunk splits (mult 1/2/4 all worse than 16),
  worker no-pin (EAS sends workers to LITTLE cores, 2.8 tps), self-uclamp
  (EPERM), min_freq writes (denied). Pool duty ~1.3/5 threads running —
  16× dynamic chunking is the optimal response to parked mids, not the bug.
- **Next real lever:** restructure the GEMV LUT expansion to cut tbl ops
  per byte (kernel project), or control the platform state (root/userdebug).
