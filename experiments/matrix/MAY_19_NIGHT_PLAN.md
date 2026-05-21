# May 19 Night Plan

This plan now has an explicit execution order. The current decode investigation and implementation must be finished before any broad trajectory rerun. The full-core prefill investigation starts only after the decode implementation is complete and the single-thread trajectories have been regenerated.

- Keep prefill and decode separate.
- Do not run the full matrix while decode productionization is still active.
- Single-thread decode and prefill are validation tracks after H66 is integrated.
- Full-core prefill is the next research track, not the immediate task.
- Before each Cactus command: `source ./venv/bin/activate && cactus build`.
- After C++/CMake/FFI/model-runner changes, rebuild and redeploy Android runners before trusting Android results.
- Do not change model files, weight formats/layout, threading, or per-kernel thread choices unless that change is the explicit H66 productionization path or a ledger-promoted full-core prefill hypothesis.
- Treat probe wins as diagnostic until real model throughput moves by predicted Amdahl amount.

## Current Execution Order

1. Finish decode productionization from H65/H66.
   - H65 proved the LM-head/output-head mechanism but failed the production gate because runtime folding destroyed TTFT/prefill.
   - H66 is the current production direction: stored folded rowwise-int8 LM-head sidecar plus main-branch-style I8MM layout reached 9.06 tok/s on strict Pixel Gemma 512 CPU7.
   - The remaining decode work is guardrails plus integration: converter/model-format path, no runtime folding, correctness/quality, Samsung, second model/shape, RAM, TTFT, and prefill side effects. Guardrails can run against the current H66 sidecar path before full converter integration, but broad trajectory regeneration remains blocked until the final integrated path is accepted or precisely blocked.
2. Regenerate single-thread trajectories through context 4096 for all Cactus LLMs.
   - Status: completed for Gemma, LFM, and Qwen with optimized output-head sidecars active; the primary CSV/docs update must be committed and pushed before longer follow-up work.
   - Explicit scope: Gemma 4 E2B, LFM 2.5 VL 1.6B, and Qwen3 VL 2B.
   - Do not carry forward existing LFM/Qwen rows as completion evidence. First adapt the Gemma optimized LM-head sidecar/I8MM path, or an explicitly equivalent model-specific output-head optimization, to LFM and Qwen; then rerun their trajectories with the optimized artifacts active.
   - Gemma is only the first optimized trajectory because it is the current validation target, not the end of the rebenchmarking plan.
   - Include single-thread decode and prefill.
   - Optimized LFM/Qwen numbers were manually patched into the primary Pixel single-thread CSV `experiments/matrix/results/matrix_pixel_10a_single_thread.csv` immediately after H72 completed; commit and push the primary CSV/docs update before continuing longer investigations.
   - Commit and push results immediately after validation.
   - Commit only updates to the primary single-thread CSVs, not scratch/intermediate/new one-off CSVs generated during the run.
3. Start the full-core prefill-only investigation.
   - Status: mechanism found and benchmark fix validated. Pixel full-core Cactus prefill needs max-performance-core-only worker pinning plus benchmark/control-thread CPU7 pinning; naive all-performance-core use on Pixel is slower because equal work lands on mid cores.
   - Treat the old decode kernel-copying work as historical mechanism evidence, not the next active research path.
   - Begin with fresh full-core prefill baselines and profiles before kernel changes.
4. After full-core prefill is fixed or precisely blocked, rerun full-core prefill benchmarks through context 4096.
   - Status: completed for Cactus rows. Gemma, LFM, and Qwen were rerun through 4096 and patched into `experiments/matrix/results/matrix_pixel_10a_full_core.csv`; Qwen was rerun after cooldown and only the 4096 row remains thermal-status 1.
   - Explicit scope: Gemma 4 E2B, LFM 2.5 VL 1.6B, and Qwen3 VL 2B.
   - Commit and push results immediately after validation.
   - Commit only updates to the primary full-core CSVs, not scratch/intermediate/new one-off CSVs generated during the run.

## Decode Implementation Gate

- Observed result: H66 sidecar + I8MM improved strict Pixel Gemma 512 CPU7 from adjacent no-env 5.55 tok/s to 9.06 tok/s, with TTFT 13.53 s, prefill 37.85 tok/s, one 213-215 ms sidecar load, and no runtime folded-cache builds.
- Remaining mechanisms to settle: sidecar/model-format integration vs ad hoc env path, full correctness/quality vs sampled hidden-vector evidence, memory lifetime/mmap behavior, Samsung and second-model/shape behavior.
- Chosen path/share: continue H66 as decode productionization. Do not restart generic CQ4 kernel exploration unless a guardrail failure proves H66 cannot be productized.
- Predicted signatures:
  - Integrated H66 path should keep decode near the H66 9.06 tok/s result.
  - TTFT should not include H65's 130 s runtime-folding penalty.
  - Prefill should not collapse; H66 improved the Gemma 512 scout prefill over adjacent no-env.
  - Guardrails should identify any model-specific or device-specific risk before trajectory reruns.
- Falsifier: H66 cannot be represented in the converter/model format without unacceptable compatibility, correctness, RAM, or guardrail regressions.
- Cheapest decisive next tests: run Gemma 512 Pixel repeat, Samsung guardrail, and one second model/shape on the current H66 sidecar path, then finish the smallest integrated H66 path before launching trajectory regeneration.

## Full-Core Prefill Tier Plan

Start this only after decode productionization and single-thread trajectory regeneration are complete.

1. Tier 0: define harness and acceptance.
   - Models, context lengths through 4096, thread counts, core masks, devices, thermal checks, output schema, and target comparison.
2. Tier 1: fresh no-change full-core prefill baselines.
   - Pixel first, Samsung comparison where required, with thermals/RAM/TTFT/prefill only.
3. Tier 2: coarse full-core prefill profile.
   - Attribute time to threadpool overhead, matmul kernels, attention/KV, memory movement, quant/dequant, synchronization, allocator, and model I/O.
4. Tier 3: scaling and scheduling split.
   - Thread-count/core-mask sweep to distinguish wrong-core placement, poor parallel scaling, contention, memory bandwidth, and kernel inefficiency.
5. Tier 4: focused diagnostics.
   - Only code experiments with predicted Amdahl movement and falsifiable signatures from Tier 2/3.
6. Tier 5: real-model candidate fix.
   - Adjacent baseline, Pixel repeat, Samsung/second-model guardrails, and no promotion from probe-only wins.
7. Tier 6: full-core prefill benchmark regeneration.
   - Contexts through 4096 for Gemma 4 E2B, LFM 2.5 VL 1.6B, and Qwen3 VL 2B, update only the primary full-core CSVs, then commit and push results immediately.

## Historical H49 Probe Direction

This section is retained only as decode-history context. It is not the active next experiment unless H66 guardrails fail and the ledger explicitly reopens generic CQ4 decode work.

The extraction step supports a focused probe:

- Cactus hot path: CQ4 interleaved group-size 128, with codebook/TBL expansion and per-group norms.
- LiteRT hot path: delegated TFLite `FULLY_CONNECTED`, mostly `INT8 x INT4 -> INT8`, with per-output-channel weight scales.
- Probe target: a standalone Pixel hot-shape kernel in `tests/android/cq4_kernel_probe.cpp` or adjacent test code that uses Cactus-relevant shapes but replaces CQ codebook lookup with a LiteRT-like per-output-channel int4/int8 packed dot path.
- Required prediction before real-model hook: because H45 gives current interleaved generic GEMV about 40.34% self time, reaching 8 tok/s from about 6 tok/s needs roughly 25% total time reduction. A generic-only path therefore needs about 60% reduction of that bucket before it is worth promoting.
