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

Cactus-only run at 512-token decode budget. _Pending — to be filled by the
on-device run (see `Android test` branch)._

| Device | Model | Prefill TPS | Decode TPS | TTFT (ms) |
|--------|-------|------------:|-----------:|----------:|
| _TBD_  | qwen3-0.6b  | — | — | — |
| _TBD_  | gemma3-270m | — | — | — |
