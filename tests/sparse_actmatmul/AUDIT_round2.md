# Sparse-Activation INT4 GEMV Audit - Round 2

Date: 2026-04-20
Machine: Apple M4 Pro, 14 hardware threads. `sysctl` reports P-cluster L2
16 MB, E-cluster L2 4 MB, cacheline 128 B. Apple SLC is not exposed by
`sysctl`; the prior log assumes about 24 MB.

## Inputs Read

- `tests/sparse_actmatmul/ITERATION_LOG.md`: read end to end.
- `tests/sparse_actmatmul/HANDOFF.md`: missing from the working tree. `find . -name HANDOFF.md` found no copy.
- Baseline: `cactus/kernel/kernel_matmul.cpp`, `cactus_gemv_int4`.
- Current candidate source: `cactus/kernel/kernel_matmul_actsparse.cpp` and declarations in `cactus/kernel/kernel.h`.
- Correctness harness: `tests/test_sparse_actmatmul_correctness.cpp`.
- Cold-cache harness: `tests/test_bench_sparse_actmatmul.cpp`.
- Prior cold-cache reference: `sparse-int4:tests/test_bench_sparse_formats.cpp`.

## Reproduction

Required build sequence was used before running Cactus tests:

```sh
source ./venv/bin/activate && cactus build
```

Correctness command:

```sh
source ./venv/bin/activate && cactus build && cmake --build tests/build --target test_sparse_actmatmul_correctness && ./tests/build/test_sparse_actmatmul_correctness
```

Result: PASS. All sparse candidates matched `cactus_gemv_int4(A_masked, ...)` with zero candidate-vs-dense mismatches in the printed cases.

Cold-cache bench command:

```sh
source ./venv/bin/activate && cactus build && cmake --build tests/build --target test_bench_sparse_actmatmul && ./tests/build/test_bench_sparse_actmatmul
```

Result CSV: `sparse_actmatmul_bench.csv`.

Comparison against `tests/sparse_actmatmul/round5_blocky.csv`:

- 48/48 rows reproduced with the same shape, sparsity, and group-skip percentages.
- Winner time reproduced within +/-10% on all 48 rows.
- Best speedup was outside +/-10% on 2/48 rows, both driven by dense-baseline timing drift rather than winner timing drift.
- Dense baseline time was outside +/-10% on 6/48 rows.

Key prior-vs-local rows:

| shape | sp | prior dense us | prior best | local dense us | local best |
|---|---:|---:|---:|---:|---:|
| 8192x16384 | 80% | 561.8 | 3.27x KMI4 | 560.0 | 3.25x KMI4f |
| 8192x16384 | 70% | 558.8 | 2.50x KMI2 | 560.6 | 2.37x KMI4f |
| 8192x16384 | 50% | 531.3 | 1.64x KMI4v2 | 550.6 | 1.68x KM |
| 8192x8192 | 80% | 317.3 | 2.75x KMI4v2 | 329.9 | 2.81x KMI4f |
| 8192x8192 | 70% | 325.1 | 2.25x KMI2 | 329.3 | 2.14x KMI4 |
| 8192x8192 | 50% | 320.7 | 1.53x KMI2 | 326.2 | 1.58x KMI2 |
| 4096x16384 | 80% | 302.4 | 2.44x KMI2 | 308.8 | 2.45x KMI2 |
| 4096x16384 | 70% | 308.1 | 2.10x KMI2 | 317.3 | 2.14x KMI4 |
| 4096x16384 | 50% | 292.2 | 1.49x KMI2 | 311.8 | 1.56x KMI2 |
| 4096x8192 | 80% | 184.5 | 2.23x KMI4 | 185.2 | 2.19x KMI4f |
| 4096x8192 | 70% | 185.0 | 1.95x KMI4 | 187.2 | 1.92x KMI4f |
| 4096x8192 | 50% | 175.8 | 1.31x KMI2 | 187.5 | 1.42x KMI2 |
| 2048x2048 | 80% | 69.9 | 1.16x KMI2 | 70.0 | 1.11x KMI4v2 |

Conclusion: the prior winner timings are comparable on this machine. For strict speedup comparisons, publish prior and local columns because dense timing drift exceeds +/-10% on a few rows.

## Audit Findings

1. Benchmark scope excludes mask derivation from timed sparse calls.

   `tests/test_bench_sparse_actmatmul.cpp` builds `A_masked`, `bitmasks`, and `livelists` before the timed loops, then times only the dense or sparse GEMV function. This means the reported sparse speedups are kernel-only speedups after top-k/mask materialization, not per-token speedups from the prompt's full input contract `(A, S, B) -> C`. It also means the listed Phase B avenue "fusing mask derivation" cannot be measured by the current scoreboard. This is not a kernel correctness bug, but it is a benchmark-scope issue that should be fixed or explicitly labeled before using the numbers for end-to-end claims.

2. Timed iteration order is deterministic round-robin.

   The bench uses `idx = i % matrices` for every kernel. The pool is large enough for cold weights, but the largest shapes have only 8 matrix/mask patterns and repeat them in the same order. That leaves room for branch predictor and loop-trip-count memorization of the sparsity pattern. Use a fixed shuffled index sequence, shared across kernels for fairness, before trusting tiny deltas.

3. Weight pool sizing is sound; trash buffer sizing is only partially sound.

   `POOL_TARGET_BYTES = 256 MB` and `count >= 8` keep every shape well beyond the 16 MB P-cluster L2 and the prior assumed 24 MB SLC. `CACHE_TRASH_BYTES = 64 MB` is 4x P-cluster L2 and 16x E-cluster L2, but only about 2.7x the prior assumed 24 MB SLC. The pool rotation is the main cold-cache mechanism, so this does not invalidate the bench, but the trash buffer should be derived from detected/assumed LLC or raised to at least 128 MB.

4. Correctness checks are stronger in practice than in policy.

   The candidate outputs matched dense-masked output exactly in the reproduction run, but the harness allows `1e-2` absolute and `2e-3` relative for most candidates. The task requires `1e-3` relative in final FP16 output. Tighten the test threshold; the current kernels should still pass.

   Follow-up: the candidate-vs-dense checks were tightened to `1e-3` absolute and `1e-3` relative after Round 7 cleanup, and the correctness gate still passed with zero candidate mismatches.

5. No hidden `S`-dependent B preprocessing found.

   `cactus_repack_int4_kmajor` and `cactus_repack_int4_kmajor_inline` depend only on `B`, `B_scales`, `K`, `N`, and `group_size`. The per-token `S` path builds `A_masked`, a group bitmask, and a live-group list only.

6. Threading is sane for the winner family.

   KMI/KMI2/KMI4 split disjoint `N_blocks` ranges, and each worker writes a disjoint slice of `C`. `kmi4_fast` and `kmi4_v2` use `thread_local` scratch, not shared scratch. The small-N thread-pool overhead remains visible in results, but it is not a race.

7. Runners-up are built, but public surface and comments are too noisy.

   The tree keeps azero, bitmask, livelist, mask2nb, livepf, KM, KMI, KMI2, KMI4, KMI4_fast, and KMI4_v2. They are built and named by round, so they are not dead in the linker sense, but the public header exposes every experimental runner-up and contains narrative comments. This violates the repo's preference for narrow public surface and minimal comments. Before handoff, either keep only the winner plus clearly needed fallbacks or label the rest as benchmark-only experimental APIs.

   Follow-up: the header comments were trimmed to short experimental grouping labels. The public surface is still broad because the benchmark currently builds all runners-up.

## Phase A Signoff

No sparse-kernel correctness bug was found. The prior winner's kernel timings reproduce closely enough to compare on this machine, but the benchmark should be treated as a kernel-only benchmark until mask derivation is timed or explicitly separated.
