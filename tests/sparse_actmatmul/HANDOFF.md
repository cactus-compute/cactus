# Sparse-Activation INT4 GEMV Handoff

Date: 2026-04-20
Machine: Apple M4 Pro, 14 hardware threads. `sysctl` reports 16 MB
P-cluster L2, 4 MB E-cluster L2, 128 B cache line.

## Status

Phase A is signed off. The original `HANDOFF.md` was missing when the
second-agent audit started, so this file is a reconstruction from
`ITERATION_LOG.md`, the source tree, and the reproduced benches.

Correctness passed before optimization and passed again after the new
candidate was added and after the killed Round 7 branch was removed.
The correctness harness now checks sparse candidates against the dense
masked-A kernel with `1e-3` absolute and `1e-3` relative tolerances;
the observed candidate mismatches were zero.

The prior cold-cache kernel timings reproduced on this machine. Winner
times were within +/-10% on all 48 rows versus `round5_blocky.csv`.
Dense baseline drift exceeded +/-10% on 6/48 rows, so compare both
prior and second-agent columns rather than using a single speedup column
as an absolute truth.

The Phase B targets were not met. No roofline stop is claimed: the best
large-row sparse stream bandwidth measured in Round 6 was 117.6 GB/s,
while the dense stream accounting reached 135.3 GB/s in the same run.
That is close enough to show the kernel is memory-pressure dominated,
but not within the requested 10% ceiling proof.

## Commands

Correctness gate:

```sh
source ./venv/bin/activate && cactus build && cmake --build tests/build --target test_sparse_actmatmul_correctness && ./tests/build/test_sparse_actmatmul_correctness
```

Kernel-only cold-cache bench:

```sh
source ./venv/bin/activate && cactus build && cmake --build tests/build --target test_bench_sparse_actmatmul && ./tests/build/test_bench_sparse_actmatmul
```

The benchmark output writes `sparse_actmatmul_bench.csv` at repo root.
Archived CSVs are in this directory.

## Audit Outcome

- `tests/test_bench_sparse_actmatmul.cpp` is a kernel-only benchmark. It
  precomputes `A_masked`, bitmasks, and live lists before timing, so it
  does not measure the full `(A, S, B) -> C` per-token path.
- The weight pool is large enough for the measured shapes. The trash
  buffer is 64 MB, which is enough for visible L2 but only about 2.7x
  the prior assumed 24 MB Apple SLC.
- Timed iteration order is deterministic round-robin. The largest
  shapes use only 8 matrix/mask patterns, so branch predictor effects
  can leak into tiny deltas.
- No hidden `S`-dependent `B` preprocessing was found. K-major repacks
  depend only on `B`, scales, `K`, `N`, and `group_size`.
- Threading is sane for the winner family. Workers write disjoint `C`
  ranges; existing scratch buffers are `thread_local`.
- Public surface is still experimental: several runners-up are exposed
  in `kernel.h` because the benchmark builds them all.

Full audit: `AUDIT_round2.md`.

## Numbers

Prior is `round5_blocky.csv`. Second-agent current source-aligned bench
is `round6_blocky.csv`, which adds `KMI4c`. `round7_blocky.csv` is
archived only for the killed 80-byte tile experiment.

Target pass counts, Round 6 versus prior best-of-family:

| sparsity | required gain | rows passing |
|---:|---:|---:|
| 80% | 1.30x | 0 / 16 |
| 70% | 1.20x | 0 / 16 |
| 50% | 1.15x | 0 / 16 |

Representative prior-vs-new rows:

| KxN | sp | skip | prior dense us | prior best | new dense us | new best | new/prior |
|---|---:|---:|---:|---|---:|---|---:|
| 8192x16384 | 80% | 78.6% | 561.8 | 3.27x KMI4 | 572.9 | 3.67x KMI2 | 1.12x |
| 8192x16384 | 70% | 67.5% | 558.8 | 2.50x KMI2 | 568.4 | 2.44x KMI2 | 0.97x |
| 8192x16384 | 50% | 47.5% | 531.3 | 1.65x KMI | 558.0 | 1.73x KMI | 1.05x |
| 8192x8192 | 80% | 78.6% | 317.3 | 2.75x KMI4v2 | 321.2 | 2.58x KMI4c | 0.94x |
| 8192x8192 | 70% | 67.5% | 325.1 | 2.25x KMI2 | 325.1 | 2.20x KMI2 | 0.98x |
| 8192x8192 | 50% | 47.5% | 320.7 | 1.53x KMI2 | 339.0 | 1.62x KMI4 | 1.06x |
| 4096x16384 | 80% | 76.4% | 302.4 | 2.47x KMI | 341.5 | 2.68x KMI4 | 1.08x |
| 4096x16384 | 70% | 67.2% | 308.1 | 2.10x KMI2 | 316.7 | 2.16x KMI4c | 1.03x |
| 4096x16384 | 50% | 47.1% | 292.2 | 1.49x KMI2 | 341.6 | 1.69x KMI4 | 1.14x |
| 4096x8192 | 80% | 77.0% | 184.5 | 2.23x KMI4 | 191.0 | 2.28x KMI4c | 1.02x |
| 4096x8192 | 70% | 67.9% | 185.0 | 1.95x KMI4 | 182.0 | 1.84x KMI4c | 0.94x |
| 4096x8192 | 50% | 46.8% | 175.8 | 1.31x KMI2 | 183.4 | 1.37x KMI4 | 1.05x |
| 2048x2048 | 80% | 74.5% | 69.9 | 1.16x KMI2 | 70.6 | 1.13x KMI4v2 | 0.98x |

Top Round 6 improvements over prior best-of-family:

| KxN | sp | prior best | Round 6 best | ratio |
|---|---:|---|---|---:|
| 8192x2048 | 80% | 1.86x KMI2 | 2.29x KMI4c | 1.23x |
| 4096x16384 | 50% | 1.49x KMI2 | 1.69x KMI4 | 1.14x |
| 2048x8192 | 70% | 1.54x KMI2 | 1.73x KMI4f | 1.12x |
| 8192x16384 | 80% | 3.27x KMI4 | 3.67x KMI2 | 1.12x |

## Current Candidate

No replacement winner was found. The prior best-of-family remains the
right scoreboard framing.

The only surviving new kernel is
`cactus_gemv_int4_actsparse_kmi4_chain`. It uses the same 72-byte
K-major inline tile as KMI4, batches four live groups, but updates each
output accumulator immediately after each group's dot and scale load.
This reduces live temporaries compared with KMI4. It won 15/48 Round 6
rows and beat the no-chain family by >5% on 2/48 rows, but it did not
move the target rows enough.

## Dead Ends

- 80-byte aligned inline tile (`KMI4a`) was correctness-clean but killed.
  It won 2/48 rows and beat the 72-byte family by >5% on 0/48 rows. The
  implementation was removed; `round7_blocky.csv` keeps the data.
- Another same-layout KMI batch variant is unlikely to meet the target.
  Round 3 through Round 7 moved winners around but did not produce a
  stable large-shape gain near the requested 1.2x-1.3x range.
- SME/SME2 is not a current quick win. The local macOS build flags only
  exposed `__ARM_FEATURE_DOTPROD`; no SME/SME2 macro appeared in the
  compile-predefine check.
- Partial half-group skipping is not promising for the current blocky
  bench because live groups usually come from whole 128-score blocks;
  boundary groups are rare. It may need revisiting for real Gemma 4 E2B
  sparsity traces.
- Mask-derivation fusion cannot be scored with the current kernel-only
  benchmark. Another agent is updating the benchmark toward Gemma 4 E2B
  deployment behavior; use that before spending more effort here.

## Hot Spots

The KMI family streams one 72-byte tile per live group per four output
rows: 64 packed INT4 bytes plus four FP16 scales. The hot loop is the
packed-B stream, INT4 unpack/dot, FP16 scale conversion, and FP32
accumulator update. `A_masked` loads are small and reused across each
worker's `N` slice. Mask construction/top-k is outside the timed region.

Round 6 peak accounted bandwidth:

| path | max accounted bandwidth |
|---|---:|
| dense | 135.3 GB/s |
| KMI2 | 117.3 GB/s |
| KMI4c | 117.6 GB/s |
| KMI4f | 114.1 GB/s |

This supports memory pressure as the main limiter, but it is not a
formal roofline proof within 10% of the device ceiling.

## Known Weaknesses

- The current sparse benchmark is not an end-to-end deployment metric.
  It excludes mask derivation and uses synthetic blocky/iid score
  patterns rather than Gemma 4 E2B traces.
- The cold-cache harness should move to a fixed shuffled iteration order
  and derive or raise the trash buffer size before tiny deltas are
  trusted.
- Small `N` rows are dominated by thread dispatch and scratch setup.
- Public experimental API surface should be trimmed once a production
  kernel is chosen.
- The sparse kernel-only benchmark and Gemma 4 E2B deployment benchmark
  work need merge coordination because both touch sparse benchmarking
  assumptions.
