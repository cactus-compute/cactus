# Goal Doc: CQ4 Small-M Kernel Optimization

## Goal

Explore and implement optimized CQ4 kernel paths for small decode verifier widths `M=2`, `M=3`, and `M=4`.

The purpose is to make assistant/speculative decode target verification as fast as possible while preserving Cactus's core advantages: default threading behavior, low RAM use, packed quantized weights, and portable performance across supported platforms.

Performance targets are measured against the non-MTP baseline main-model decode forward pass:

- `M=2`: target main-model forward-pass decode time per generated token is less than 15% slower than baseline.
- `M=3`: target main-model forward-pass decode time per generated token is less than 25% slower than baseline.
- `M=4`: target main-model forward-pass decode time per generated token is less than 35% slower than baseline.

Lower is better. Passing the threshold is not enough if further general, low-risk kernel improvements are available.

## Background

Cactus uses compact quantized weights to keep local model RAM small. CQ4 is especially important because it gives strong model-size reduction while still supporting fast decode on-device. Assistant/speculative decoding changes the target model workload: instead of verifying one next-token position at a time, the target may verify a small candidate block. For this goal, the important block sizes are `M=2`, `M=3`, and `M=4`.

The current generic `decode_tps` and end-to-end completion timings are not precise enough for this work because they include assistant draft generation, sampling, callbacks, response construction, logging, and other completion overhead. The optimization target must be the target main model's forward-pass decode work only.

The main risk is chasing misleading wins. Increasing threads, changing core allocation, unpacking whole weights, or tuning for a single Mac can make a local benchmark look better while violating Cactus's product constraints. This work must improve the raw CQ4 small-M algorithm while keeping the runtime memory profile and default execution model intact.

Existing areas likely involved:

- CQ quantized matmul and GEMV/GEMM dispatch in `cactus-kernels/src/matmul.cpp`.
- Kernel APIs in `cactus-kernels/cactus_kernels.h`.
- CQ correctness and benchmark coverage in `cactus-kernels/tests/test_matmul.cpp`.
- Speculative decode integration and timing surfaces in `cactus-engine/src/model.cpp` and `cactus-engine/src/complete.cpp`.
- CLI smoke/benchmark entry points around `cactus-engine/tests/chat.cpp`.

## Definitions

- `baseline`: normal non-MTP decode with the same model, prompt, generation options, build type, and platform. This is the reference for main-model forward-pass time per generated token.
- `M`: the number of target decode positions evaluated together by the CQ4 verifier path for a speculative decode round. If the implementation uses a different internal name for this dimension, document the mapping before changing code.
- `target forward time`: wall-clock time spent inside the main model's forward execution during decode verification. This excludes assistant draft time, tokenization, sampling, accept/reject bookkeeping, callbacks, JSON construction, logging, and other miscellaneous completion overhead.
- `avg target forward ms/token`: total target forward time during measured decode divided by emitted/generated token count for the measured decode window.
- `slowdown`: `(candidate_avg_target_forward_ms_per_token / baseline_avg_target_forward_ms_per_token) - 1`.

## Guardrails

- Do not change default Cactus threading behavior.
- Do not change thread allocation, thread counts, core affinity, worker-pool sizing, per-kernel work partitioning policy, or command-line defaults for `M=2`, `M=3`, or `M=4`.
- Do not make performance wins come from giving these sizes different threading treatment.
- Do not de-pack or cache the full CQ4 weight matrix in unpacked form.
- Do not introduce any representation that uses memory on the order of the model's unquantized or fully unpacked weight size.
- Keep weights packed in their existing CQ4 representation. Transient register, stack, or small per-thread tile scratch is acceptable only when it is bounded by tile/group size, not model size.
- Do not overfit to one Mac. Optimizations should be algorithmic and portable across supported Apple, Android, and other ARM targets.
- Do not regress non-MTP baseline decode performance.
- Do not regress CQ1, CQ2, CQ3, FP16, or embedding behavior while optimizing CQ4.
- Do not explain misses as quantization problems without a focused experiment that localizes the issue.
- Do not make broad style-only refactors while working on this goal.
- Keep diffs focused and reviewable.
- Prefer extending existing CQ kernel structure over adding parallel one-off implementations.
- Preserve existing fallback paths for unsupported shapes.

## General Plan

1. Read the recent CQ4 and MTP timing-related code and commit history before editing.
2. Establish or add benchmark instrumentation that reports target main-model forward-pass decode time separately from assistant and miscellaneous overhead.
3. Capture baseline numbers before changing kernels.
4. Profile `M=2`, `M=3`, and `M=4` CQ4 verification to identify the real bottleneck.
5. Implement the smallest targeted CQ4 small-M optimization that addresses the measured bottleneck.
6. Add or extend focused CQ4 correctness tests for the affected shapes.
7. Run the smallest relevant correctness and benchmark commands after each meaningful change.
8. Compare candidate medians against baseline using the benchmark contract.
9. Iterate only on algorithmic kernel improvements that respect the guardrails.
10. Finish with a concise note showing measurements, memory behavior, threading behavior, and remaining limitations.

## Benchmark Contract

The benchmark must report target main-model forward-pass decode timing separately from assistant and miscellaneous time.

For each run, capture:

- model id/path and quantization mode
- build type, commit, platform, CPU, and OS
- prompt text or prompt fixture id
- prompt token count
- generated token count
- `mtp_enabled`
- `mtp_draft_tokens`
- effective verifier width `M`
- default Cactus thread count/allocation evidence
- baseline `avg_target_forward_ms_per_token`
- candidate `avg_target_forward_ms_per_token`
- slowdown percentage
- assistant draft time, reported separately
- miscellaneous completion time, reported separately
- accepted draft tokens, rejected draft tokens, and acceptance rate
- peak resident memory or Cactus-reported RAM usage

The acceptance metric is:

```text
slowdown_pct =
  100 * (candidate_avg_target_forward_ms_per_token / baseline_avg_target_forward_ms_per_token - 1)
```

Do not use end-to-end `decode_tps` alone as the acceptance metric. `decode_tps` is useful context, but it mixes target forward time with assistant, sampling, callback, JSON, and other overhead.

## Required Benchmark Procedure

Before every Cactus command:

```bash
source ./venv/bin/activate
cactus build
```

Rebuild after C++/FFI/kernel changes. If running multiple benchmark commands in the same activated terminal session after one build, rebuilding is not required unless the code changes again.

Run the benchmark in this order:

1. Warm up the exact model and prompt once, and discard the result.
2. Run the baseline non-MTP decode at least 5 times.
3. Run `M=2` at least 5 times.
4. Run `M=3` at least 5 times.
5. Run `M=4` at least 5 times.
6. Repeat the full sequence after any meaningful kernel change.

Use the same:

- model
- quantization
- prompt
- max generated tokens
- temperature/sampling settings
- build configuration
- default thread settings
- thermal/power conditions as much as practical

Prefer deterministic decode settings for kernel timing:

```json
{
  "temperature": 0.0,
  "top_k": 1,
  "auto_handoff": false,
  "confidence_threshold": -1.0
}
```

Use enough generated tokens to reduce noise. The default benchmark window should be at least 128 decode tokens after warmup unless a smaller fixture is needed for rapid iteration.

## Benchmark Output Shape

The final benchmark should produce a table like:

| mode | M | baseline target ms/token | candidate target ms/token | slowdown | accepted | rejected | assistant ms/token | misc ms/token | peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1 | 0.000 | 0.000 | 0.0% | n/a | n/a | n/a | 0.000 | 0 MB |
| mtp | 2 | 0.000 | 0.000 | 0.0% | 0 | 0 | 0.000 | 0.000 | 0 MB |
| mtp | 3 | 0.000 | 0.000 | 0.0% | 0 | 0 | 0.000 | 0.000 | 0 MB |
| mtp | 4 | 0.000 | 0.000 | 0.0% | 0 | 0 | 0.000 | 0.000 | 0 MB |

Use median as the headline number. Also record min, max, and standard deviation in raw output so regressions are visible.

## Investigation Plan

### 1. Establish Measurement First

Add or identify instrumentation that can time only target main-model forward work during decode.

Acceptance criteria:

- Baseline and MTP runs expose comparable target forward timing.
- Assistant timing is reported separately.
- Miscellaneous overhead is reported separately.
- The benchmark can be repeated without changing default threading settings.

### 2. Identify CQ4 Small-M Hot Paths

Profile the CQ4 decode verifier path for `M=2`, `M=3`, and `M=4`.

Look specifically at:

- CQ4 activation transform cost
- activation quantization cost
- codebook expansion cost
- packed-index decode cost
- dot-product accumulation structure
- output store layout
- small-M GEMM dispatch overhead
- reuse opportunities across rows of `M`
- avoidable temporary allocation or cache pressure

Acceptance criteria:

- The measured hotspot is identified before implementation.
- The proposed optimization states whether it targets transform, quantize, unpack, dot, reduction, store, or dispatch overhead.

### 3. Optimize Algorithmically

Prioritize improvements that reduce repeated work for `M=2..4` without changing thread behavior.

Candidate directions:

- Share CQ4 codebook quantization across the small-M group where valid.
- Reuse packed weight decode work across adjacent `M` rows without materializing full unpacked weights.
- Fuse per-group loops so small-M rows consume the same packed weight stream while it is hot.
- Reduce repeated activation transform or activation quantization overhead across `M`.
- Specialize compact `M=2`, `M=3`, and `M=4` microkernels only where the dispatch remains clean and portable.
- Improve register blocking and accumulator layout for small-M without increasing large persistent memory.
- Remove avoidable per-call allocation from CQ4 small-M paths.

Avoid:

- platform-specific hacks that only help one CPU model
- changing scheduling or thread counts
- broad rewrites of unrelated matmul paths
- adding a parallel implementation that duplicates most of the existing CQ kernel stack

### 4. Preserve Packed Memory Behavior

Any new scratch space must be explicitly bounded.

Allowed examples:

- small stack/register temporaries for one tile
- per-thread buffers sized by `M * group_size`, `num_groups`, or a small output tile
- reusable scratch already consistent with existing kernel patterns

Disallowed examples:

- full unpacked CQ4 matrix cache
- full int8 copy of model weights
- full fp16/fp32 dequantized copy of model weights
- hidden global caches that scale with total parameter count

Acceptance criteria:

- Peak RAM does not grow materially.
- Code review can verify scratch bounds from dimensions.
- Packed weight storage remains the runtime source of truth.

### 5. Validate Correctness

For every optimized path, add focused tests or extend existing kernel tests.

Required coverage:

- CQ4 `M=2`
- CQ4 `M=3`
- CQ4 `M=4`
- odd tails for `N` where applicable
- representative group sizes used by converted models
- comparison against existing CQ4 output within the current tolerance
- fallback behavior for unsupported shapes

Do not weaken tolerances to hide a bug. If numeric differences change, localize and explain the source.

### 6. Validate Performance

Run the benchmark after each meaningful optimization.

A change is acceptable only if:

- `M=2` slowdown is less than 15%
- `M=3` slowdown is less than 25%
- `M=4` slowdown is less than 35%
- baseline non-MTP target forward timing does not regress
- RAM stays within the intended small-memory profile
- no default threading behavior changed

If a target cannot be met, document:

- the best measured slowdown
- the remaining hotspot
- experiments already tried
- why the result is blocked
- the next plausible algorithmic direction

## Code Areas To Inspect

Start with:

- `cactus-kernels/src/matmul.cpp`
- `cactus-kernels/cactus_kernels.h`
- `cactus-kernels/tests/test_matmul.cpp`
- `cactus-engine/src/model.cpp`
- `cactus-engine/src/complete.cpp`
- `cactus-engine/tests/chat.cpp`

Also inspect recent commit history for CQ4 and MTP timing changes before editing kernel code.

## Final Acceptance Criteria

The work is complete only when all of the following are true:

- A repeatable benchmark reports target main-model forward decode time per generated token.
- The benchmark excludes assistant time and miscellaneous completion overhead from the acceptance metric.
- `M=2` is less than 15% slower than baseline by the benchmark contract.
- `M=3` is less than 25% slower than baseline by the benchmark contract.
- `M=4` is less than 35% slower than baseline by the benchmark contract.
- Default Cactus threading behavior and thread allocation are unchanged.
- No full unpacked/dequantized weight cache is introduced.
- Peak RAM remains consistent with Cactus's small-memory design.
- Kernel changes are general enough for supported platforms, not tuned only to this Mac.
- Focused CQ4 correctness tests pass.
- Relevant Cactus decode/MTP tests pass.
- `git diff --check` passes.

## Final Deliverables

- Focused kernel changes for CQ4 small-M paths.
- Benchmark output with raw runs and median summaries.
- A short implementation note explaining:
  - which hotspot was optimized
  - why the optimization helps `M=2..4`
  - how scratch memory is bounded
  - proof that threading behavior was not changed
  - remaining limitations or follow-up opportunities
