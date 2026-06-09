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
> **Threadpool spin fix — measured with/without, and rejected:** the adaptive
> spin-then-block worker patch (commit 88679112, ported to the cactus-v2 pool)
> was A/B'd on the Pixel, gemma @545ctx/100dec, interleaved spin→no-spin→spin:
> **with fix 5.4 / 4.8 decode tps; without 6.6** — spinning costs ~20% on this
> power-constrained SoC (forcing 4 cores busy splits the power budget and slows
> the prime core). The earlier "3.8→14 tps" validation was an artifact of an
> ODR-corrupted binary (old-repo kernel objects overriding v2 pool symbols with
> a different class layout). The fix is therefore NOT applied; these tables are
> the no-spin configuration. Samsung was never eligible (8 perf cores → spin
> auto-disabled). The kernel_utils.h segfault attributed to this patch was
> root-caused to the engine/bundle format mismatch above, not the patch.
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

