# May 19 Night Status

## Current Run State

- Last completed command: H72 optimized LFM/Qwen single-thread trajectory regeneration on Pixel 10a:
  `source ./venv/bin/activate && cactus build && python experiments/matrix/run_matrix.py --config experiments/matrix/matrix.yaml --device pixel_10a --runtime cactus --model lfm_2_5_vl_1_6b --model qwen3_vl_2b --operation prefill --operation decode --seqlen 256 --seqlen 512 --seqlen 1024 --seqlen 2048 --seqlen 4096 --warmup-runs 1 --measurement-runs 3 --out experiments/matrix/results/matrix_pixel_10a_cactus_lfm_qwen_optimized.csv`.
- Immediate next action: commit and push the updated full-core CSV/docs, then decide whether Qwen needs a cooled acceptance repeat. The current target is competitive full-core prefill numbers without losing the H66/H68 decode win.
- Optimized single-thread trajectory regeneration is complete for Gemma, LFM, and Qwen. The H72 LFM/Qwen rows were manually patched into `experiments/matrix/results/matrix_pixel_10a_single_thread.csv` immediately after the CSV landed.
- Full-core prefill root cause is now narrowed: Pixel's old no-affinity full-core path sent equal work to heterogeneous mid/big cores and left the benchmark/control thread off CPU7. The validated benchmark policy is max-performance-core worker pinning plus main-thread CPU7 pinning.
- H66 stored LM-head sidecar completed in `experiments/matrix/results/h66_stored_lmhead_i8mm_sidecar_20260521/`: strict Pixel Gemma 512 CPU7 sidecar + I8MM reached decode 9.06 tok/s, prefill 37.85 tok/s, TTFT 13.53 s, peak RAM 2447.54 MB; adjacent no-env was decode 5.55 tok/s, prefill 27.97 tok/s, TTFT 18.30 s.
- H67 H66 guardrails completed in `experiments/matrix/results/h67_h66_guardrails_20260521/`: Samsung Gemma 512 sidecar improved 13.86 -> 20.46 tok/s with no regression; Pixel Gemma 1024 sidecar improved 5.56 -> 8.64 tok/s; all sidecar runs loaded once, no runtime cache builds, thermal status 0.
- H68 integrated model-local sidecar discovery completed in `experiments/matrix/results/h68_integrated_lmhead_sidecar_20260521/`: with no model-local sidecar, Pixel Gemma 512 stayed at decode 5.57 tok/s and prefill 28.13 tok/s; with `token_embeddings.lmhead_qd8_qc8.cache` next to the weight file and no env vars, Pixel Gemma 512 reached decode 8.96 tok/s and prefill 37.52 tok/s, one 216.993 ms sidecar load, no runtime cache build, thermal status 0.
- Baseline/device-state check completed in `experiments/matrix/results/current_nochange_pixel_tier12_20260521_post_h66/`: Tier 1 decode 5.56 tok/s, Tier 2 decode 5.55 tok/s, prefill 28.21/27.69 tok/s, thermal status 0 before/after, RAM 2452.35-2452.90 MB.
- H62 qd8 activation gate completed in `experiments/matrix/results/h62_lmhead_qd8_activation_error_20260521/`: decode-last100 affine qd8 plus rowwise-int8 folded LM-head sampled output relative L2 mean 0.01917, p95 0.02962, sampled-vocab top1 agreement 100% over 2048 sampled rows.
- H61 normal XNNPACK build gate completed: normal `android/build` is `android-21`, side XNNPACK build is `android-26`, and the normal build still fails linking `xnnpack_qc4_probe` on KleidiAI `stdout`/`stderr` references.
- H63 API-26 Cactus bench gate completed in `experiments/matrix/results/h63_api26_cactus_bench_gate_20260521/`: API-26 side `cactus_llm_bench` built and ran strict Gemma 512 CPU7 no-env at decode 5.61 tok/s, prefill 28.49 tok/s, peak RAM 2452.79 MB.
- H64 native dense qd8/qc8 output-head speed gate completed: `orthogonal_qd8_qc8_dot4` repeated at 18.11 ms for `K=1536,N=262144`, predicting about 8.54 tok/s from the H59 measured LM-head share if it transfers to the real hook.
- H65 env-gated real-model LM-head hook completed in `experiments/matrix/results/h65_lmhead_qd8_qc8_hook_20260521/`: decode improved from adjacent no-env 5.46 tok/s to 8.75 and 8.64 tok/s repeats, but TTFT rose to 146-150 s because the folded int8 cache builds twice on device.
- H59 LM-head share/real-activation diagnostic completed in `experiments/matrix/results/h59_lmhead_share_real_activation_20260521/`: repeat decode 5.56 tok/s, peak RAM 2455.97 MB, 170 LM-head calls averaging 80.226 ms, decode last-100 LM-head share 45.04%, real-hidden rowwise-int8 sampled output relative L2 mean 0.006338.
- Baseline/device-state check completed in `experiments/matrix/results/current_nochange_pixel_tier12_20260521_0114/`: Tier 1 decode 5.62 tok/s, Tier 2 decode 5.56 tok/s, thermal status 0 before/after, RAM 2456.70-2456.75 MB.
- Baseline/device-state check completed in `experiments/matrix/results/current_nochange_pixel_tier12_20260520_2154/`: Tier 1 decode 5.56 tok/s, Tier 2 decode 5.43 tok/s, thermal status 0, RAM near 2451-2455 MB.
- H39 combined diagnostic completed in `experiments/matrix/results/generic_ortho_combined_diag_20260520_2154/`: decode 6.14 and 6.10 tok/s, peak RAM near 2452 MB, thermal status 0.
- The status/ledger files were missing in this checkout and have been recreated from current artifacts and command outputs.
- Before every Cactus command: `source ./venv/bin/activate && cactus build`.
- Equivalent strict Gemma 512 command:
  `taskset 80 /data/local/tmp/cactus_matrix/bin/cactus_llm_bench /data/local/tmp/cactus_matrix/models/gemma-4-e2b-it-sharedkv /data/local/tmp/cactus_matrix/inputs/seqlen_512_default_token2.csv 100`

## Active Work Queue

1. Finish H66 productionization and guardrails.
   - Integrate the stored LM-head int8/I8MM artifact path into the converter/model format or document the precise blocker.
   - Validate decode speed, TTFT, prefill side effects, RAM, correctness/quality, Samsung, and a second model/shape. The immediate H67 guardrails can run on the current H66 sidecar path, but trajectory regeneration remains blocked until the final integrated path is accepted or precisely blocked.
2. Regenerate single-thread trajectories through context 4096 for all Cactus LLMs.
   - Explicit scope: Gemma 4 E2B, LFM 2.5 VL 1.6B, and Qwen3 VL 2B.
   - Do not reuse existing LFM/Qwen rows as completion evidence. Apply the same Gemma optimization direction, or a model-specific equivalent if their LM-head structure differs, then rerun LFM and Qwen with the optimized path active.
   - Gemma is the first optimized trajectory, not the whole rebenchmarking plan.
   - Include decode and prefill.
   - Critical handoff requirement: the instant optimized LFM/Qwen CSV rows are available, manually patch them into `experiments/matrix/results/matrix_pixel_10a_single_thread.csv`, record that this was done here, then commit and push the primary CSV/docs update before continuing slower follow-up work.
   - Commit and push results immediately after validation.
   - Commit only updates to the primary single-thread CSVs, not scratch/intermediate/new one-off CSVs generated during the run.
3. Investigate full-core prefill using the tiered plan in `MAY_19_NIGHT_PLAN.md`.
   - Status: mechanism found and benchmark fix validated. The fixed full-core Cactus rows have been patched into `experiments/matrix/results/matrix_pixel_10a_full_core.csv`.
4. Rerun full-core prefill benchmarks through context 4096 for Gemma 4 E2B, LFM 2.5 VL 1.6B, and Qwen3 VL 2B after the fix/blocker, then commit and push only updates to the primary full-core CSVs.
   - Status: completed for Cactus rows; Qwen rows were collected under thermal status 1 and may need cooled acceptance repeats if exact Qwen medians are critical.

The H50-H66 records below are chronological history. Any `Next Experiment Record` heading inside that history is superseded unless it is restated in the Current Run State, Active Work Queue, or Planned H67 section above.

## Planned H67 Record

- Observed gap/target: H66 reaches the strict Pixel Gemma 512 decode target at 9.06 tok/s from a fresh no-change baseline of 5.55-5.56 tok/s. Promotion still requires guardrails: no meaningful Samsung regression and at least one second model/shape before trajectory regeneration.
- Mechanisms separated: Pixel-specific benefit only vs portable stored-LM-head/I8MM path; context-shape dependence vs per-token LM-head independence; sidecar load/RAM behavior on a comparison device.
- Chosen path/share: run the H66 sidecar path as a guardrail, not a new optimization. First use Samsung strict Gemma 512 no-env and H66 sidecar runs from the same rebuilt binary. Then run one Pixel second-shape scout at Gemma 1024 if Samsung does not expose a regression.
- Predicted signatures:
  - Samsung sidecar run should not materially regress versus same-binary Samsung no-env; if it improves, that is acceptable but not the Pixel root-cause target.
  - Pixel Gemma 1024 sidecar should remain near the 8-9 tok/s class because LM-head cost per decode token is context-independent, while attention/KV cost may reduce the exact TPS.
  - Sidecar load should remain one short load per process with no runtime `cache_built` lines.
- Falsifier: Samsung sidecar regresses by more than noise versus no-env, Pixel second-shape decode falls back near baseline, sidecar loads repeatedly, or RAM/TTFT becomes unacceptable.
- Cheapest decisive test: push the current runner and sidecar to Samsung, run strict Gemma 512 no-env and sidecar, then run Pixel Gemma 1024 sidecar/no-env if the Samsung guardrail passes.

## Completed H67 Record

- Evidence:
  - Results: `experiments/matrix/results/h67_h66_guardrails_20260521/`.
  - Samsung strict Gemma 512 same-binary no-env: decode 13.86 tok/s, prefill 64.88 tok/s, TTFT 7890.94 ms, peak RAM 2467.78 MB.
  - Samsung strict Gemma 512 sidecar: decode 20.46 tok/s, prefill 74.66 tok/s, TTFT 6857.52 ms, peak RAM 2460.55 MB.
  - Samsung sidecar load: one load, 269.958 ms; output-head calls settle around 8.0 ms; no `cache_built` lines.
  - Pixel strict Gemma 1024 same-binary no-env: decode 5.56 tok/s, prefill 36.98 tok/s, TTFT 27687.46 ms, peak RAM 2453.02 MB.
  - Pixel strict Gemma 1024 sidecar: decode 8.64 tok/s, prefill 43.51 tok/s, TTFT 23536.50 ms, peak RAM 2446.48 MB.
  - Pixel 1024 sidecar load: one load, 235.205 ms; output-head calls settle around 17.7-18.0 ms; no `cache_built` lines.
  - Thermal status after both guardrails: 0.
- Decision: H66 passes the comparison-device and second-shape guardrails. It remains gated by production integration and broader correctness/quality, not by speed, TTFT, RAM, or obvious Samsung regression.

## Planned H68 Record

- Observed gap/target: H66/H67 establish the speed path, but current execution still requires `CACTUS_LMHEAD_QD8_QC8_DIAG=1` and `CACTUS_LMHEAD_QD8_QC8_CACHE_PATH=...`. Production needs model-local discovery/loading with existing models unchanged unless an explicit sidecar exists.
- Mechanisms separated: env-path diagnostic vs model artifact sidecar discovery; kernel-owned global sidecar cache vs loader-attached weight metadata; read-into-vector memory lifetime vs mmap-backed sidecar lifetime.
- Chosen path/share: inspect graph weight loading and converter paths to find the smallest compatible integration. Preferred first productionization step is optional model-local sidecar discovery for `token_embeddings.weights`, preserving current CQ4 fallback when the sidecar is absent.
- Predicted signatures:
  - With sidecar present in the model directory, H66 path should activate without `CACTUS_LMHEAD_QD8_QC8_CACHE_PATH`.
  - With sidecar absent, no-env baseline remains unchanged.
  - The sidecar should load once, with no runtime folding.
- Falsifier: runtime kernels do not have enough path/name context for safe sidecar discovery without broad graph/format changes, or loader integration would require incompatible model format changes.
- Cheapest decisive test: inspect `cactus-graph/src/io.cpp`, `cactus_graph.h`, and converter weight-writing code, then either implement optional sidecar discovery or document the exact integration blocker.

## Completed H68 Record

- Code:
  - `cactus-kernels/cactus_kernels.h` and `cactus-graph/cactus_graph.h`: optional `lmhead_qd8_qc8_path` metadata on CQ matrices.
  - `cactus-graph/src/io.cpp`: optional sidecar discovery for structural Gemma LM-head weights, looking for `token_embeddings.lmhead_qd8_qc8.cache` next to `token_embeddings.weights`.
  - `cactus-kernels/src/matmul.cpp`: H66 path activates automatically when the CQ matrix has a sidecar path, while env vars remain supported for diagnostics.
- Evidence:
  - Results: `experiments/matrix/results/h68_integrated_lmhead_sidecar_20260521/`.
  - Sidecar absent: decode 5.57 tok/s, prefill 28.13 tok/s, TTFT 18201.96 ms, peak RAM 2455.43 MB, no sidecar logs.
  - Sidecar present, no env vars: decode 8.96 tok/s, prefill 37.52 tok/s, TTFT 13645.59 ms, peak RAM 2449.69 MB.
  - Sidecar load appeared once from the model-local path and took 216.993 ms. No runtime `cache_built` lines appeared.
- Decision: integrated model-local discovery is validated enough to move to benchmark regeneration and prefill work. Remaining packaging polish is converter emission of the sidecar, but the immediate production concern is now competitive prefill numbers rather than env-path removal.

## Planned H69 Record

- Observed gap/target: H68 preserves the decode win and improves single-thread Gemma 512 prefill from 28.13 to 37.52 tok/s, but the docs now define the next production focus as regenerating optimized Gemma trajectories and then fixing full-core prefill numbers.
- Mechanisms separated: single-thread optimized decode/prefill validation vs full-core prefill scheduling/kernel/memory bottleneck vs benchmark artifact bookkeeping.
- Chosen path/share: regenerate the primary optimized Gemma single-thread trajectory first, using the model-local H68 sidecar path. Do not start full-core kernel experiments until those validation rows are recorded.
- Predicted signatures:
  - Gemma single-thread decode trajectory should stay near the H66/H67/H68 class where LM-head dominates.
  - Single-thread prefill should be reasonable and not collapse under the integrated sidecar path.
  - The run should update only the intended primary CSVs, not scratch outputs.
- Falsifier: integrated sidecar fails outside 512/1024, prefill regresses versus no-sidecar, or the benchmark harness does not pick up the optimized artifact.
- Cheapest decisive test: inspect the matrix commands/primary CSV targets, then run the smallest optimized Gemma single-thread trajectory regeneration that covers decode and prefill through context 4096.

## H69 Result

- Command completed: `source ./venv/bin/activate && cactus build && python experiments/matrix/run_matrix.py --config experiments/matrix/matrix.yaml --device pixel_10a --runtime cactus --model gemma_4_e2b --operation prefill --operation decode --seqlen 256 --seqlen 512 --seqlen 1024 --seqlen 2048 --seqlen 4096 --warmup-runs 1 --measurement-runs 3 --out experiments/matrix/results/matrix_pixel_10a_cactus_gemma_fast.csv`.
- Output: `experiments/matrix/results/matrix_pixel_10a_cactus_gemma_fast.csv`.
- Pixel strict CPU7 Gemma optimized trajectory:
  - 256: prefill 26.12 tok/s, decode 9.03 tok/s.
  - 512: prefill 34.78 tok/s, decode 8.66 tok/s.
  - 1024: prefill 38.18 tok/s, decode 7.78 tok/s.
  - 2048: prefill 36.71 tok/s, decode 6.87 tok/s.
  - 4096: prefill 38.58 tok/s.
- Logs confirm the integrated model-local sidecar loaded from `/data/local/tmp/cactus_matrix/models/gemma-4-e2b-it-sharedkv/weights/gemma-4-e2b-it/token_embeddings.lmhead_qd8_qc8.cache`; no env path was required.
- Device note: Android thermal status stayed 0, but post-run `thermal-cpufreq-2` cooling was nonzero. Treat this as valid regeneration evidence for the integrated path, but rerun cooled if these rows become acceptance-critical.
- Decision: Gemma optimized single-thread regeneration is complete. The broader regeneration objective still requires adapting the optimized output-head path to LFM 2.5 VL 1.6B and Qwen3 VL 2B, then rerunning them; existing LFM/Qwen rows are not sufficient.

## Planned H70 Record

- Observed gap/target: existing Pixel full-core prefill rows are much slower than Pixel strict single-thread rows, not merely slower than Samsung. At 512, Pixel strict single-thread prefill is Gemma 34.78 tok/s, LFM 43.13 tok/s, Qwen 29.08 tok/s; existing Pixel full-core prefill is about Gemma 10.03 tok/s, LFM 10.61 tok/s, Qwen 6.12 tok/s. Samsung full-core 512 rows are about Gemma 119.26 tok/s, LFM 58.86 tok/s, Qwen 52.09 tok/s.
- Mechanisms separated: full-core harness does not actually enable Cactus intra-op threading; no-affinity scheduling may place the single active Cactus work on slow cores; prefill-only `prepare_decode=false` may use a slower path than paired strict prefill; true multi-core prefill kernel scaling may still be absent after the harness issue is separated.
- Chosen path/share: first test scheduler/harness behavior using direct Pixel Gemma 512 prefill-only runs with the same `cactus_llm_bench` binary and input, comparing no affinity against CPU7 pinning. This covers the full measured prefill row and costs one short scout pair.
- Predicted signatures:
  - If scheduler/no-affinity is primary, CPU7-pinned prefill-only should jump near the H69 strict single-thread prefill band while no-affinity stays near the old full-core row.
  - If `prepare_decode=false` is primary, both no-affinity and CPU7-pinned prefill-only stay low.
  - If real multi-threading is absent but scheduler is not primary, both runs are similar and close to single-thread.
- Falsifier: no-affinity and CPU7-pinned prefill-only are both near the current H69 prefill band, which would make the existing full-core CSV stale or caused by another run-matrix artifact.
- Cheapest decisive test: after required `source ./venv/bin/activate && cactus build`, run direct `cactus_llm_bench` Gemma 512 prefill-only once with no taskset and once with `taskset 80`, capturing throughput, CPU allowance, and thermal state.

## H71 Result

- Code/artifact changes:
  - Generalized `experiments/matrix/build_lmhead_qd8_qc8_sidecar.py` from Gemma-only shape to any CQ4 one-group orthogonal LM head with `N % 4 == 0` and `K % 16 == 0`.
  - Generalized model-local sidecar discovery and runtime hook shape gates from Gemma `(N=262144,K=1536)` to the same one-group orthogonal sidecar contract.
  - Generated and pushed LFM sidecar `token_embeddings.lmhead_qd8_qc8.cache` for `(N=65536,K=2048)`, 129 MB.
  - Generated and pushed Qwen sidecar `token_embeddings.lmhead_qd8_qc8.cache` for `(N=151936,K=2048)`, 298 MB.
- Scout command completed: `source ./venv/bin/activate && cactus build && python experiments/matrix/run_matrix.py --config experiments/matrix/matrix.yaml --device pixel_10a --runtime cactus --model lfm_2_5_vl_1_6b --model qwen3_vl_2b --operation prefill --operation decode --seqlen 512 --warmup-runs 1 --measurement-runs 1 --out experiments/matrix/results/h71_lfm_qwen_lmhead_sidecar_scout_20260521.csv`.
- Output: `experiments/matrix/results/h71_lfm_qwen_lmhead_sidecar_scout_20260521.csv`.
- Pixel strict CPU7 512 scout:
  - LFM: prefill 33.80 tok/s, decode 17.49 tok/s.
  - Qwen: prefill 21.99 tok/s, decode 10.91 tok/s.
- Logs confirm sidecar loads for `K=2048,N=65536` and `K=2048,N=151936`.
- Decision: the Gemma optimization direction transfers to LFM and Qwen. Next run the full optimized LFM/Qwen single-thread trajectories through context 4096; do not reuse prior LFM/Qwen rows.

## H72 Result

- Command completed: `source ./venv/bin/activate && cactus build && python experiments/matrix/run_matrix.py --config experiments/matrix/matrix.yaml --device pixel_10a --runtime cactus --model lfm_2_5_vl_1_6b --model qwen3_vl_2b --operation prefill --operation decode --seqlen 256 --seqlen 512 --seqlen 1024 --seqlen 2048 --seqlen 4096 --warmup-runs 1 --measurement-runs 3 --out experiments/matrix/results/matrix_pixel_10a_cactus_lfm_qwen_optimized.csv`.
- Output: `experiments/matrix/results/matrix_pixel_10a_cactus_lfm_qwen_optimized.csv`.
- Pixel strict CPU7 LFM optimized trajectory:
  - 256: prefill 24.73 tok/s, decode 16.28 tok/s.
  - 512: prefill 24.53 tok/s, decode 11.81 tok/s.
  - 1024: prefill 28.56 tok/s, decode 13.63 tok/s.
  - 2048: prefill 29.27 tok/s, decode 14.11 tok/s.
  - 4096: prefill 27.92 tok/s.
- Pixel strict CPU7 Qwen optimized trajectory:
  - 256: prefill 12.21 tok/s, decode 8.28 tok/s.
  - 512: prefill 16.33 tok/s, decode 7.85 tok/s.
  - 1024: prefill 18.14 tok/s, decode 6.88 tok/s.
  - 2048: prefill 18.26 tok/s, decode 7.12 tok/s.
  - 4096: prefill 19.48 tok/s.
- Manual primary CSV patch completed immediately after the optimized CSV landed: 18 matching LFM/Qwen rows were replaced in `experiments/matrix/results/matrix_pixel_10a_single_thread.csv`.
- Device note: Qwen rows from 512 onward recorded `thermal_status=1`; post-run CPU7 frequency was observed at 700 MHz. Treat these as the urgent patched rows, but rerun cooled if final acceptance depends on exact Qwen medians.
- Decision: optimized single-thread regeneration is complete for Gemma, LFM, and Qwen. Next work is full-core prefill investigation and regeneration.

## H73-H76 Full-Core Prefill Mechanism

- Direct Gemma 512 prefill-only scheduler split:
  - no affinity, default worker policy: 8.74 tok/s.
  - CPU7 taskset: 63.67 tok/s.
  - CPUs 4-7 taskset (`f0`): 15.02 tok/s.
  - CPUs 4-7 plus main-thread CPU7 pin: 15.12 tok/s.
  - CPUs 4-7 plus round-robin worker pinning and main-thread CPU7 pin: 17.43 tok/s.
  - max-performance-core-only workers plus main-thread CPU7 pin: 67.05 tok/s.
  - no taskset plus max-performance-core-only workers and main-thread CPU7 pin: 67.71 tok/s.
- Interpretation: the old Pixel full-core prefill rows were not slow because prefill inherently needed all cores or because sidecar output-head was broken. They were slow because equal work was allowed onto Pixel's heterogeneous mid cores and the control thread was not anchored on CPU7. On Pixel, naive use of CPUs 4-7 is worse than using only the max-capacity core for this workload.
- Harness validation: `experiments/matrix/results/h76_pixel_gemma512_full_core_maxperf_smoke_20260521.csv` produced Gemma 512 prefill 65.10 tok/s through `run_matrix.py` with no taskset, and notes record `android_threadpool_pin_max_perf_only=true` plus `android_bench_pin_main_max_perf=true`.
- All-model 512 validation: `experiments/matrix/results/h77_pixel_llm512_full_core_maxperf_validation_20260521.csv` produced Gemma 63.26 tok/s, LFM 47.61 tok/s, and Qwen 32.16 tok/s at thermal status 0.
- Implementation validation: after moving main-thread pinning into the tracked `cactus_benchmark_tokens` path, `experiments/matrix/results/h73_full_core_prefill_scheduler_20260521/gemma512_prefill_engine_main_pin_smoke.log` produced Gemma 512 prefill 62.74 tok/s with the same env policy.

## H78 Full-Core Prefill Regeneration

- Command completed: `source ./venv/bin/activate && cactus build && python experiments/matrix/run_matrix.py --config experiments/matrix/matrix.yaml --device pixel_10a --runtime cactus --model gemma_4_e2b --model lfm_2_5_vl_1_6b --model qwen3_vl_2b --operation prefill --seqlen 256 --seqlen 512 --seqlen 1024 --seqlen 2048 --seqlen 4096 --benchmark-mode full_core_prefill --warmup-runs 1 --measurement-runs 3 --out experiments/matrix/results/matrix_pixel_10a_cactus_full_core_maxperf.csv`.
- Pixel Cactus full-core-prefill rows with max-performance-core worker policy and main-thread CPU7 pin:
  - Gemma: 256 69.03, 512 61.35, 1024 52.57, 2048 41.35, 4096 37.67 tok/s.
  - LFM: 256 43.00, 512 41.43, 1024 38.19, 2048 34.35, 4096 28.28 tok/s.
  - Qwen: 256 29.31, 512 27.68, 1024 23.65, 2048 20.93, 4096 19.78 tok/s.
- Manual primary CSV patch completed: 15 matching Cactus prefill rows were replaced in `experiments/matrix/results/matrix_pixel_10a_full_core.csv`. Local helper CSVs `matrix_pixel_10a_full_core_prefill_4096.csv` and `matrix_pixel_10a_full_core_prefill_upto2048.csv` were also patched.
- Device note: Gemma and LFM rows recorded thermal status 0. Qwen rows recorded thermal status 1 and post-run CPU7 frequency was observed at 700 MHz, so cooled Qwen repeats are the remaining acceptance risk.
- Decision: full-core prefill is now competitive/reasonable versus the prior Pixel Cactus rows. The remaining production choice is whether to make the max-performance-core/main-thread policy default on Tensor-class Android devices or keep it benchmark-gated.

## Completed H50 Record

- Observed gap/target: LiteRT-LM cache-disabled still reaches 9.55 tok/s decode on strict Pixel CPU7, while Cactus local-source orthogonal-only is about 5.9-6.0 tok/s. Target is at least 8 tok/s.
- Mechanisms separated: Cactus CQ4 codebook/TBL decode plus per-group norms vs LiteRT/XNNPACK per-output-channel int4/int8 `FULLY_CONNECTED` execution vs residual non-MATMUL/KV overhead.
- Chosen path/share: focus on the current interleaved generic GEMV path, which H45 shows at 40.34% self time after the valid orthogonal diagnostic.
- Predicted signatures:
  - If LiteRT's FC contract is a plausible explanation, a per-output-channel int4/int8 hot-shape probe should beat `packed_tbl_dot_norm` by enough to imply a material Amdahl move.
  - If CQ codebook/TBL and per-group scaling are not the main difference, the LiteRT-like probe will be within noise or slower.
  - If this is just a probe artifact, it will not hold across both `K=1536,N=12288` and the reverse/down-projection shape.
- Falsifier: the LiteRT-like probe does not reduce hot-shape time by a large margin, or only wins on a shape that does not dominate Gemma decode.
- Cheapest decisive test: add a standalone probe to `tests/android/cq4_kernel_probe.cpp`, build/deploy only the probe, then run Pixel CPU7 hot shapes before any real-model hook.
- Evidence:
  - Results: `experiments/matrix/results/h50_litert_like_i4pc_probe_20260521/`.
  - 1000-iteration CPU7 repeat: `gemma_ffn_up` current 0.401211 ms vs LiteRT-like 0.435359 ms; `gemma_ffn_down` current 0.406898 ms vs LiteRT-like 0.384061 ms; `gemma_attn_q` current 0.099081 ms vs LiteRT-like 0.093238 ms.
  - Decision: rejected as primary explanation. This is not enough movement for a path to 8 tok/s; a simple per-output-channel int4/int8 rowwise FC contract is not materially better than Cactus's current hot CQ4 path.

## Completed H51 Record

- Observed gap/target: Cactus remains about 5.9-6.0 tok/s under the valid orthogonal diagnostic, while LiteRT-LM remains about 9.55-10.04 tok/s on CPU7. Target is at least 8 tok/s.
- Mechanisms separated: real-model per-shape timing and dispatch overhead vs standalone microkernel timing vs LiteRT/XNNPACK compiled FC microkernel selection/scheduling.
- Chosen path/share: measure actual Cactus decode linear time by `(K,N,weight_format,path)` before more code changes, then profile LiteRT enough to identify its delegated FC top symbols/counter signatures.
- Predicted signatures:
  - If probe shape averages hide the real blocker, actual Gemma decode will show a small number of shape/path buckets dominating the residual.
  - If XNNPACK's microkernel is the missing piece, LiteRT simpleperf should expose specific quantized FC/GEMV symbols with better cycles/MAC or lower backend-stall behavior than Cactus.
  - If graph/runtime overhead is meaningful, Cactus aggregated linear/attention/orthogonal timing will leave a large residual that LiteRT does not show.
- Falsifier: Cactus actual per-shape timing matches standalone probe proportions and LiteRT profile shows no materially different FC or non-FC behavior.
- Cheapest decisive test: add one env-gated real-model timing aggregation in the Cactus matmul/linear path and run strict Gemma 512 CPU7 once; separately run a short LiteRT-LM CPU7 simpleperf profile.
- Evidence:
  - Direct XNNPACK operator probe: `tests/android/xnnpack_qc4_probe.cpp`.
  - Results: `experiments/matrix/results/h51_xnnpack_qc4_probe_20260521/`.
  - Same-session 1000-iteration CPU7 comparison:
    - `gemma_ffn_up`: Cactus 0.400904 ms, XNNPACK `qd8_f32_qc4w` 0.261327 ms, 1.53x faster.
    - `gemma_ffn_down`: Cactus 0.428915 ms, XNNPACK 0.260334 ms, 1.65x faster.
    - `gemma_attn_q`: Cactus 0.099505 ms, XNNPACK 0.046913 ms, 2.12x faster.
  - Simpleperf on XNNPACK FFN-up: `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm` is 91.62% self time.
  - Decision: XNNPACK's real I8MM QC4 kernel is materially better than Cactus's current generic CQ4 GEMV. H50's naive rowwise int4 loop was not a valid proxy. Generic-only transfer predicts roughly 7 tok/s, but applying the same class of win to the broader roughly 80% matmul-like profile share predicts about 8.2-8.8 tok/s. Coverage of the orthogonal/output-head bucket is now the key question.

## Completed H52 Record

- Observed gap/target: Cactus is still about 5.9-6.0 tok/s under the valid orthogonal diagnostic, while LiteRT-LM remains about 9.55-10.04 tok/s on CPU7. Target is at least 8 tok/s.
- Mechanisms separated: XNNPACK I8MM QC4 microkernel/layout vs Cactus CQ4 codebook semantics vs orthogonal/output-head layout vs real-model conversion/init/RAM overhead vs residual non-matmul decode work.
- Chosen path/share: H51 identifies the relevant XNNPACK operator/ukernel. Next, test whether that kernel advantage transfers into actual Cactus decode by routing selected generic CQ4 hot weights through an env-gated XNNPACK/QC4 diagnostic cache, then determine whether the orthogonal/output-head bucket can use the same class of kernel.
- Predicted signatures:
  - If transfer works only for generic CQ4, strict Gemma 512 decode should move by roughly 15-20% from the orthogonal-enabled baseline and likely land near 7 tok/s.
  - If transfer works for the broader matmul-like share, strict Gemma 512 decode should move into the 8-9 tok/s range.
  - If conversion/layout overhead dominates or CQ4 semantics do not map cleanly, real-model movement will be much smaller than the standalone operator result.
- Falsifier: an env-gated XNNPACK/QC4 route covers the hot shapes but gives less than about half the predicted decode movement or causes unacceptable RAM/init growth.
- Cheapest decisive test: implement the smallest diagnostic cache for selected Gemma generic weights and run strict 512 decode on CPU7 with RAM/init logging.
- Evidence:
  - Results: `experiments/matrix/results/h52_xnnpack_shape_suite_20260521/shape_suite_cpu7_clean_stdout.txt`.
  - Weighted by Gemma decode shape counts across the six major non-output-head linear families, Cactus current toy total is 39.901738 ms and XNNPACK QC4 toy total is 23.990912 ms, for 1.663x.
  - From a 6.0 tok/s baseline, applying 1.663x to only the 40.34% generic GEMV bucket predicts about 7.15 tok/s.
  - Applying 1.663x to a broader 80% matmul-like share predicts about 8.81 tok/s.
  - Output-head-sized toy: XNNPACK QC4 `K=1536,N=262144` is 10.828792 ms; current orthogonal `K=1536,N=262144` is 61.980803 ms, a 5.72x kernel-level gap. This is promising but not yet a proven semantic drop-in for the orthogonal output-head path.
  - Decision: 9 tok/s is theoretically defensible if XNNPACK-style coverage extends beyond generic FFN GEMV into most linear/matmul-like decode work. Generic-only is not enough.

## Completed H53 Record

- Observed gap/target: toy matrix evidence can justify about 8.8 tok/s from broad 80% coverage and much higher upper-bound estimates if output-head-sized work maps to XNNPACK QC4. The next blocker is semantic and integration feasibility, not standalone kernel speed.
- Mechanisms separated: toy matrix speed vs real CQ4 codebook/normalization semantics vs model-load conversion/cache cost vs actual decode dispatch overhead.
- Chosen path/share: inspect Cactus CQ4 weight representation and the output-head orthogonal representation to determine whether an env-gated XNNPACK operator cache can be built without changing model files.
- Predicted signatures:
  - If generic CQ4 weights can be materialized as XNNPACK signed/unsigned QC4 plus per-output scales, a diagnostic route should reproduce the toy speed class on real FFN/attention weights.
  - If orthogonal/output-head can be represented as an XNNPACK QC4 operator after applying or folding its transform/scales, the 8-9 tok/s target becomes realistic.
  - If output-head semantics cannot be folded, generic-only routing should still improve but should likely stop around 7 tok/s.
- Falsifier: Cactus CQ4 codebook/norm semantics require per-group/per-token math that cannot be represented in XNNPACK's per-output-channel QC4 contract without recreating the slow path around it.
- Cheapest decisive test: write a minimal real-weight conversion/readout diagnostic for one FFN-up weight and the output-head weight that reports whether scales/codebook/norms can map to XNNPACK QC4 operator inputs with bounded error and bounded memory.
- Evidence:
  - Results: `experiments/matrix/results/h53_real_weight_xnnpack_mapping_20260521/mapping_stdout.txt`.
  - Real generic Gemma CQ4 weights do not directly match XNNPACK's one-scale-per-output-row QC4 contract: FFN-up has 12 norm groups per row, FFN-down has 48, and attention Q has 12.
  - Collapsing sampled reconstructed generic rows to one signed-int4 scale per row gives nontrivial relative L2 error: FFN-up mean 0.1705, FFN-down mean 0.1999, attention Q mean 0.1601.
  - The tied output-head/token-embedding matrix is CQ4 orthogonal with one group per row and is structurally compatible after folding, but it needs a large packed cache: about 192 MiB plus scales, with 1536 MiB if naively materialized as dense float32.
  - Codebook affine check: `experiments/matrix/results/h53_real_weight_xnnpack_mapping_20260521/codebook_affine_stdout.txt`; the real generic CQ4 codebook has affine-fit relative L2 0.1101, so replacing the TBL codebook with plain signed-int4 MMLA is a diagnostic with measurable semantic risk, not an exact fix.
  - Decision: direct XNNPACK QC4 routing is not the right product direction for generic Cactus CQ4. The new direction is to learn from XNNPACK's fast Pixel QC4 kernel and adapt its tiling, packing, and I8MM schedule to Cactus CQ4 while preserving existing codebook/per-group semantics.

## Planned H54 Record

- Observed gap/target: Cactus remains around 5.9-6.0 tok/s with the valid orthogonal diagnostic; LiteRT-LM remains about 9.6-10.0 tok/s. XNNPACK's QC4 kernel speed class can theoretically justify 8-9 tok/s if the design transfers to most decode linear work, but H53 rejects a direct generic weight-contract swap.
- Mechanisms separated: XNNPACK I8MM ukernel design vs Cactus CQ4 codebook/per-group math vs Cactus packing/layout vs real-model dispatch/Amdahl movement.
- Chosen path/share: compare XNNPACK's `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm` and packer against Cactus's current interleaved CQ4 hot kernel, then build the smallest Cactus-native probe that borrows the XNNPACK schedule without changing CQ4 semantics.
- Predicted signatures:
  - The probe binary should contain I8MM instructions on Pixel, not just SDOT.
  - Hot-shape probes should close much of the 1.5-2.2x standalone gap before model integration.
  - Real Gemma 512 decode should move by the Amdahl prediction for the covered profile share.
  - Outputs must match the current CQ4 path before performance is credited.
- Falsifier: the codebook/per-group CQ4 semantics cannot be arranged into an I8MM-friendly packed block without losing the speed advantage, or a hot-shape win fails to transfer to real decode.
- Cheapest decisive test: source-level pack/ukernel comparison, followed by one Cactus CQ4 hot-shape probe variant with XNNPACK-like output tiling and I8MM-friendly blocks.

## Completed H54 Record

- Observed gap/target: current-session strict Pixel Gemma 512 no-env refresh is 5.62/5.56 tok/s; LiteRT-LM remains about 9.6-10.0 tok/s. H51/H52 show XNNPACK QC4 kernels are fast enough, while H53 rejects a direct one-scale generic CQ4 mapping.
- Mechanisms separated: XNNPACK-style 16-output I8MM schedule vs Cactus SDOT plus TBL codebook expansion vs memory cost of pre-expanded codebook-preserving int8 weights.
- Chosen path/share: target current interleaved generic CQ4 first, because H45 puts it at 40.34% self time after the working orthogonal diagnostic; generic-only needs roughly 60% bucket reduction to be promotable toward 8 tok/s.
- Predicted signatures:
  - Probe binary contains MMLA/I8MM instructions.
  - Expanded-int8 I8MM hot-shape timings approach XNNPACK QC4 timings if scheduling is the transferable mechanism.
  - If true nibble-packed signed-int4 is the key, the expanded-int8 path will remain close to current SDOT/TBL timing.
  - Memory cost is reported separately; a 2x expanded-weight path is diagnostic until bounded.
- Falsifier: probe is within noise/slower than current `packed_tbl_dot_norm`, lacks MMLA instructions, or wins only on non-dominant shapes.
- Cheapest decisive test: add one standalone `vmmlaq_s32` expanded-int8 tile to `tests/android/cq4_kernel_probe.cpp`, build/deploy `cq4_kernel_probe`, run Pixel CPU7 hot shapes, and grep/disassemble for MMLA.
- Evidence:
  - Results: `experiments/matrix/results/h54_i8mm_expanded_cq4_probe_20260521/probe_cpu7_stdout.txt`.
  - Disassembly: `experiments/matrix/results/h54_i8mm_expanded_cq4_probe_20260521/disasm_dot_mmla.txt` contains `smmla`, so the probe used I8MM instructions.
  - `gemma_ffn_up`: current 0.402044 ms vs expanded I8MM 0.567226 ms, 41.1% slower.
  - `gemma_ffn_down`: current 0.421505 ms vs expanded I8MM 0.554969 ms, 31.7% slower.
  - `gemma_attn_q`: current 0.083297 ms vs expanded I8MM 0.071041 ms, and repeat 0.082414 ms vs 0.070033 ms, a repeated 14-15% attention-shape win.
  - Decision: rejected as a broad generic path. Preserving arbitrary CQ4 codebook values as expanded int8 and then using an XNNPACK-like I8MM tile does not explain the dominant FFN gap. XNNPACK's large FFN win likely depends on compact nibble-packed affine int4 weights and lower weight bandwidth, not just I8MM scheduling.

## Planned H55 Record

- Observed gap/target: Pixel Gemma 512 no-env baseline is 5.62/5.56 tok/s; target remains 8-9 tok/s. H54 rejected expanded-int8 I8MM for dominant FFN shapes, while H51/H52 still show exact XNNPACK QC4 is fast enough.
- Mechanisms separated: semantic cost of approximating Cactus CQ4 codebook with an affine signed-int4 lattice vs performance benefit of true nibble-packed XNNPACK-style MMLA vs orthogonal LM-head folding.
- Chosen path/share: do not promote expanded-int8 I8MM. The next cheapest split is a dot/output-error diagnostic for an affine-codebook nibble-packed path on real generic weights/activations, plus an independent orthogonal LM-head folded-cache path if generic-only remains capped near 7 tok/s.
- Predicted signatures:
  - If affine-codebook signed-int4 is semantically acceptable at dot/output level, it unlocks the true XNNPACK nibble-packed kernel class for generic CQ4 with per-group norms.
  - If output error is localized to specific layers/shapes, that identifies where exact CQ4 semantics must be preserved.
  - If error is broad or large, generic Cactus CQ4 cannot use the exact XNNPACK nibble contract without re-quantization/fine-tuning and the path should pivot to orthogonal/output-head or graph-level work.
- Falsifier: real sampled dot/output error from affine-codebook replacement is too large or unstable across FFN-up/down/attention shapes, or a nibble-packed probe still fails to approach XNNPACK timings.
- Cheapest decisive test: offline real-weight sampled dot diagnostic comparing exact CQ4 dequant rows against affine-codebook signed-int4 rows for FFN-up, FFN-down, and attention; only if it passes, add a nibble-packed MMLA probe preserving Cactus per-group norms.

## Completed H55 Record

- Observed gap/target: H54 rejected expanded arbitrary-int8 I8MM for dominant FFN shapes, but true XNNPACK nibble-packed QC4 remains fast. The semantic gate is whether Cactus CQ4's codebook can be approximated by an affine signed-int4 lattice closely enough to use the true nibble-packed kernel class.
- Mechanisms separated: codebook approximation drift at dot/output level vs performance benefit from compact nibble-packed MMLA vs shape-localized sensitivity.
- Chosen path/share: sample real Gemma FFN-up, FFN-down, and attention-Q weights; compare exact CQ4 codebook dot products against best affine signed-int4 codebook dot products over random transformed activation vectors.
- Predicted signatures:
  - Low and stable dot error would justify a nibble-packed MMLA probe.
  - Broad 10%+ dot error across shapes would reject this as an unqualified replacement.
- Falsifier: sampled dot/output error is negligible across the three generic families.
- Cheapest decisive test: `experiments/matrix/results/h55_affine_codebook_dot_error_20260521/dot_error_stdout.txt`.
- Evidence:
  - `ffn_up`: dot relative L2 mean 0.1247, p95 0.1325, max 0.1369.
  - `ffn_down`: dot relative L2 mean 0.1252, p95 0.1341, max 0.1397.
  - `attn_q`: dot relative L2 mean 0.1254, p95 0.1320, max 0.1398.
  - Decision: rejected as an unqualified generic replacement. This is measured dot-level semantic drift, not a placeholder explanation. A true nibble-packed XNNPACK-style generic kernel would need a correctness/fine-tuning story before it could replace Cactus CQ4.

## Next Experiment Record

- Observed gap/target: current Pixel Gemma 512 baseline is 5.62/5.56 tok/s; generic-only exact-CQ4 routes now have two blockers: expanded exact-codebook I8MM is too slow on FFN, and affine nibble-packed replacement has about 12-14% dot-level drift. Target remains 8-9 tok/s.
- Mechanisms separated: XNNPACK's actual QC4 packing/order/schedule vs Cactus exact CQ4 codebook/per-group semantics vs orthogonal/output-head folded-cache feasibility vs LiteRT graph/runtime overhead.
- Chosen path/share: make XNNPACK QC4 the reference implementation and replicate or learn from it before pivoting to output-head folding. Directly using XNNPACK's one-scale-per-output-row contract remains rejected for generic Cactus CQ4 unless a separate semantic gate passes; the active path is a Cactus-native exact-CQ4 probe that copies transferable mechanics from XNNPACK/KleidiAI source and disassembly.
- Predicted signatures:
  - Source/disassembly comparison should identify concrete differences in packing, tile order, compensation, prefetching, unroll, or activation quantization that can be copied into Cactus without changing weights.
  - A copied XNNPACK-style exact-CQ4 probe should beat current `packed_tbl_dot_norm` on at least one dominant FFN hot shape by enough to predict real-model movement.
  - If the speed only appears after changing Cactus CQ4 into affine signed-int4 nibbles, the path remains diagnostic because H55 measured 12-14% dot-level drift.
  - Output equivalence against the current CQ4 path is required before any performance result counts.
- Falsifier: copied XNNPACK mechanics do not improve dominant FFN exact-CQ4 timing, or the only fast route is the H55-rejected semantic replacement.
- Cheapest decisive test: inspect `tests/build/xnnpack-src` generated QC4 ukernel/packer source and the existing XNNPACK disassembly, compare against `tests/android/cq4_kernel_probe.cpp` and `cactus-kernels/src/matmul.cpp`, then add one standalone `cq4_kernel_probe` variant with the most transferable XNNPACK mechanic.

## Completed H56 Record

- Observed gap/target: XNNPACK QC4 remains fast enough for the 8-9 tok/s target if its speed class can cover broad decode linear work, but H54/H55 showed that neither expanded exact-codebook I8MM nor unqualified affine-codebook replacement is viable for generic Cactus CQ4.
- Mechanisms separated: XNNPACK 16-output activation-reuse tile vs affine signed-int4 compact nibble/I8MM execution vs Cactus exact arbitrary-codebook TBL lookup and per-group norms.
- Chosen path/share: copy the most transferable XNNPACK mechanic first while preserving exact Cactus CQ4: a 16-output compact-nibble tile in `tests/android/cq4_kernel_probe.cpp` named `packed_tbl_x16_dot_norm`.
- Evidence:
  - XNNPACK source inspected: `tests/build/xnnpack-src/src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-neoni8mm.c`, `tests/build/xnnpack-src/src/reference/packing.cc`, and `tests/build/xnnpack-src/src/configs/gemm-config.c`.
  - Pixel CPU7 results in `experiments/matrix/results/h56_xnnpack_exact_cq4_tile_probe_20260521/`.
  - `gemma_ffn_up`: current 0.405678 ms; exact x16 tile 0.415352 ms, 2.4% slower.
  - `gemma_ffn_down`: current 0.425430 ms; exact x16 tile 0.382052 ms, 10.2% faster.
  - `gemma_attn_q`: current 0.083991 ms; exact x16 tile 0.076756 ms, 8.6% faster.
  - Repeat vs pair tile: x16 was 0.413877 ms on FFN-up, 0.381591 ms on FFN-down, and 0.076304 ms on attention-Q.
- Decision: output tiling/activation reuse alone does not replicate XNNPACK's speed class. The evidence now points to XNNPACK's compact affine signed-int4 representation feeding I8MM, plus row compensation/scaling, as the core mechanism. Exact arbitrary-codebook CQ4 cannot use that path directly without either expanded-weight bandwidth loss or a semantic replacement.

## Next Experiment Record

- Observed gap/target: H56 makes exact-codebook tiling a small win at best, while XNNPACK true QC4 still has the kernel speed needed for 8-9 tok/s if a correctness path exists.
- Mechanisms separated: exact arbitrary-codebook preservation vs affine signed-int4 fast path plus residual correction/validation vs layer-output tolerance of a semantic replacement.
- Chosen path/share: do not write another large exact-CQ4 kernel until the remaining semantic bridge is quantified. The next scout should test whether Cactus CQ4 can be decomposed into an XNNPACK-fast affine signed-int4 component plus a small residual correction or layer-limited fallback, with predicted error and cost.
- Predicted signatures:
  - If codebook residual correction is cheap or localized, affine-fast plus residual/fallback should reduce most FFN cost while keeping sampled output error much lower than H55's uncorrected 12-14%.
  - If residual correction costs about the same as current TBL CQ4, the XNNPACK path is not transferable to exact generic Cactus CQ4.
  - If only some layer families tolerate the affine path, the Amdahl estimate must be recomputed from covered profile share before any model hook.
- Falsifier: residual correction either restores current-kernel cost or sampled layer-output error remains broad after the correction/fallback strategy.
- Cheapest decisive test: extend the offline real-weight H55 diagnostic to report affine component share, residual-codebook sparsity/top-k structure, and dot-error/cost estimates for affine-fast plus residual correction on FFN-up, FFN-down, and attention-Q.

## Completed H57 Record

- Observed gap/target: H55's uncorrected affine-codebook replacement is too inaccurate, but H56 says exact-codebook tiling does not reach XNNPACK speed. A plausible generic bridge would need XNNPACK's affine-int4 fast path plus a cheap residual correction.
- Mechanisms separated: affine signed-int4 component vs residual codebook correction vs correction coverage/cost.
- Chosen path/share: reuse the H55 real-weight sampled dot setup and add back only the largest residual codebook entries by absolute residual magnitude.
- Evidence:
  - Results: `experiments/matrix/results/h57_affine_residual_bridge_20260521/residual_bridge_stdout.txt`.
  - Fitted affine codebook matches H55: slope 14.8500517, intercept 7.42478171, codebook relative L2 0.110078799.
  - Two residual codes cover 53.1% residual energy but still leave about 11.7% mean dot relative L2.
  - Four residual codes cover 66.7% energy but still leave about 10.1%.
  - Eight residual codes cover 88.4% energy and leave about 5.45%.
  - Twelve of sixteen residual codes are needed to reach about 1.3% mean dot relative L2 across FFN-up, FFN-down, and attention-Q.
- Decision: negative for a cheap generic bridge. Residual correction would need most of the codebook entries back, which likely gives back the TBL/codebook work that XNNPACK avoids. Generic CQ4 still has no proven exact route to XNNPACK's 1.5-2x FFN speed class.

## Next Experiment Record

- Observed gap/target: generic CQ4 routes have now failed three ways: expanded exact I8MM is slower on FFN, exact compact x16 tiling is only a small win, and affine-fast residual correction needs most of the codebook back. Target remains 8-9 tok/s.
- Mechanisms separated: generic CQ4 kernel limit vs output-head orthogonal folded-cache opportunity vs LiteRT graph/runtime differences.
- Chosen path/share: pivot away from generic CQ4 kernel copying unless a new exact representation appears. The remaining high-leverage candidate is the orthogonal/output-head bucket because H52 showed a 5.72x output-head-sized kernel gap and H53 showed LM head has one norm group per row.
- Predicted signatures:
  - If LM-head folding/repacking is viable, it should produce a large standalone output-head speedup with bounded memory and a measured semantic/error profile.
  - If folded LM-head requires dense FP16/FP32 materialization or large error, the remaining gap is likely LiteRT's model contract/weight format rather than a Cactus-side kernel tweak.
- Falsifier: folded output-head cache is too large, too slow to build, inaccurate, or fails to move the output-head-sized timing enough by Amdahl.
- Cheapest decisive test: run the offline/standalone LM-head folded-cache feasibility diagnostic using real `token_embeddings.weights` rows, including memory estimate, materialization cost estimate, and dot-error for packed int4/int8/fp16 candidates.

## Completed H58 Record

- Observed gap/target: latest strict Pixel Gemma 512 baseline is 5.62/5.56 tok/s; target remains 8-9 tok/s. H52 showed output-head-sized XNNPACK QC4 at 10.828792 ms vs current orthogonal probe at 61.980803 ms, but this was not a semantic drop-in.
- Mechanisms separated: exact orthogonal CQ4 folding vs rowwise int8/int4 folded cache drift vs cache memory/materialization overhead vs output-head Amdahl share.
- Chosen path/share: target only the tied LM-head/token-embedding orthogonal CQ4 matrix first because generic CQ4 kernel copying has been narrowed/rejected by H54-H57 and H53 showed LM head has one norm group per row.
- Predicted signatures:
  - FP16 dense fold should be near-exact but memory-heavy.
  - Rowwise int8/int4 folded caches trade memory and XNNPACK-like speed against measured output drift.
  - The full-cache size and materialization time estimate must fit a plausible Pixel deployment envelope before a model hook is worth writing.
- Falsifier: all bounded-memory folded candidates have high dot/output error, or exact candidates require impractical memory/init cost.
- Cheapest decisive test: offline real-weight sampled fold diagnostic over `token_embeddings.weights`, reporting row error, sampled output-vector error, full-cache bytes, and materialization extrapolation.
- Evidence:
  - Offline fold/error diagnostic: `experiments/matrix/results/h58_lmhead_folded_cache_feasibility_20260521/folded_cache_stdout.txt`.
  - LM-head structure: `N=262144,K=1536`, CQ4 orthogonal, one group per row, current payload 192.00 MiB plus 5.01 MiB metadata.
  - FP16 dense folded cache: 768.00 MiB additional data, 965.01 MiB if retained with original payload; sampled output relative L2 mean 0.000211.
  - Rowwise int8 folded cache: 384.00 MiB data plus 1.00 MiB scales, 582.01 MiB if retained with original payload; sampled output relative L2 mean 0.009846.
  - Rowwise int4 folded cache: 192.00 MiB data plus 1.00 MiB scales, 390.01 MiB if retained with original payload; sampled output relative L2 mean 0.178467.
  - Pixel timing proxy: `experiments/matrix/results/h58_lmhead_folded_cache_feasibility_20260521/xnnpack_qc8_output_head_cpu7_stdout.txt`.
  - The normal Android build failed on the known XNNPACK/KleidiAI static-link issue (`stdout`/`stderr` unresolved), so the existing API-26 side build `android/build_xnnpack_probe` was used.
  - Same Pixel CPU7 output-head shape, 100 iterations: XNNPACK QC4 11.824582 ms; XNNPACK QC8 14.470288 ms.
  - QC8 vs current orthogonal output-head-sized probe gives 61.980803 / 14.470288 = 4.28x. Amdahl from 5.56 tok/s predicts about 7.55 tok/s if the covered bucket is H45's 34.37% orthogonal self-time share, and about 8.02 tok/s only if the covered share is 40%.
- Decision: rowwise int4 is rejected for LM head due high sampled output error. FP16 is near-exact but memory-heavy and still needs a Pixel speed proxy. Rowwise int8 is a plausible diagnostic candidate but not a fix: it adds about 385 MiB, has about 1% sampled output drift, and likely reaches 8 tok/s only if the real output-head bucket is at least about 40% of decode time or combines with another accepted win.

## Next Experiment Record

- Observed gap/target: H58 rowwise int8 folded LM-head is plausible but not sufficient by itself unless the true covered share is about 40% or more; H45's orthogonal self-time was 34.37% in a simpleperf profile. Target remains 8-9 tok/s.
- Mechanisms separated: actual real-model output-head/orthogonal share vs folded-int8 candidate speed/error vs remaining generic CQ4 residual.
- Chosen path/share: before writing a real-model folded-cache hook, measure the actual Gemma 512 decode time attributable to the token-embedding/output-head orthogonal matmul and gather representative hidden/logit samples for a stronger correctness gate.
- Predicted signatures:
  - If output-head time is at least about 40% of strict decode and rowwise-int8 logit error is tolerable on real hidden activations, an env-gated real-model hook is justified.
  - If output-head share is closer to 34% or less, LM-head-only int8 folding is a partial diagnostic and cannot hit 8 tok/s without another accepted mechanism.
  - If real hidden activation logit error is much worse than the random-vector scout, the int8 path is rejected or must be limited to a non-production diagnostic.
- Falsifier: measured output-head share below the Amdahl threshold, or real-activation sampled logits show unacceptable drift.
- Cheapest decisive test: add low-overhead env-gated timing/sample instrumentation around the orthogonal LM-head call in Cactus Gemma 512 decode, run strict CPU7 once, and use the captured hidden vectors to replay exact vs folded-int8 logits offline.

## Completed H59 Record

- Observed gap/target: H58 rowwise-int8 folded LM-head needed at least about 40% real decode share to plausibly reach 8 tok/s. Target remains 8-9 tok/s.
- Mechanisms separated: real output-head decode share vs folded-int8 speed proxy vs real-hidden activation drift.
- Chosen path/share: instrument only the structural LM-head orthogonal matrix (`bits=4,K=1536,N=262144,group_size=1536,num_groups=1`) with env-gated timing and hidden-vector capture.
- Evidence:
  - Code: `cactus-kernels/src/matmul.cpp`, env vars `CACTUS_LMHEAD_SHARE_DIAG`, `CACTUS_LMHEAD_SAMPLE_DIAG`, `CACTUS_LMHEAD_SAMPLE_COUNT`, and `CACTUS_LMHEAD_SAMPLE_PATH`.
  - Results: `experiments/matrix/results/h59_lmhead_share_real_activation_20260521/`.
  - First run: decode 5.58 tok/s, peak RAM 2456.97 MB, 170 LM-head calls, average 80.04 ms, total LM-head time 13.606 s of 35.750 s.
  - Repeat with all hidden samples: decode 5.56 tok/s, peak RAM 2455.97 MB, 170 LM-head calls, average 80.226 ms, total LM-head time 13.638 s of 35.995 s.
  - Decode-specific last 100 calls: 8.016 s of 17.796 s decode time, giving 45.04% decode share.
  - Amdahl from 5.56 tok/s at 45.04% share: H58 QC8-vs-current-toy 4.28x predicts 8.49 tok/s; real measured current LM-head 80.226 ms vs XNNPACK QC8 14.470 ms predicts 8.81 tok/s.
  - Real hidden replay over 2048 sampled vocab rows:
    - Decode last 100 rowwise-int8 sampled output relative L2 mean 0.006338, p95 0.008001, max 0.008382, cosine mean 0.999979.
    - Decode last 100 rowwise-int4 sampled output relative L2 mean 0.112377, p95 0.140734, max 0.151297.
- Decision: supported for an env-gated real-model hook. Rowwise-int8 folded LM-head is now large enough by Amdahl and low enough sampled real-activation drift to test in actual decode. It is not promoted as a fix until a hook moves real Pixel throughput by prediction, correctness is checked, and RAM/Samsung/second-shape guardrails are addressed.

## Next Experiment Record

- Observed gap/target: H59 shows the LM-head is large enough to explain a path to 8.5-8.8 tok/s if it can execute in an XNNPACK-like speed class, but it does not explain generic CQ4 and it changes the effective LM-head weights. The broader target remains making Cactus CQ4 competitive with LiteRT/XNNPACK on Pixel, around 8-9 tok/s decode.
- Mechanisms separated: XNNPACK's true QC4 compact affine-int4/I8MM contract vs exact Cactus CQ4 codebook/per-group semantics vs output-head folded-cache special case vs graph/runtime overhead.
- Chosen path/share: make the next direction explicitly "replicate or learn from XNNPACK CQ4." LM-head rowwise-int8 remains a high-leverage diagnostic candidate, but the immediate mechanism work should compare/copy XNNPACK CQ4 mechanics into Cactus probes before promoting the LM-head hook. Direct generic use of XNNPACK's one-scale-per-output-row contract remains rejected unless a new semantic gate passes.
- Predicted signatures:
  - If the transferable mechanism is packing/tile/schedule, a Cactus exact-CQ4 probe that copies the missing XNNPACK piece should beat current FFN hot-shape timing by enough to predict real-model movement while matching current outputs.
  - If XNNPACK's win depends on affine signed-int4 nibbles plus row compensation, exact arbitrary-codebook CQ4 will not reach the XNNPACK speed class; any fast route must either change the effective weights with measured layer/model tolerance or be limited to structurally compatible weights such as the LM head.
  - If direct XNNPACK operator integration is the only path that reproduces the speed, the normal Android link issue and operator cache overhead become the next concrete integration blockers to solve.
- Falsifier: copied XNNPACK mechanics still leave exact Cactus CQ4 near current FFN timings, or any fast result only appears after the already-rejected unqualified affine-codebook semantic replacement.
- Cheapest decisive test: inspect the XNNPACK/KleidiAI generated QC4 ukernel and packer against `tests/android/cq4_kernel_probe.cpp`, then add one focused probe variant for an untested transferable mechanic such as XNNPACK's activation qd8 contract, row compensation/scaling placement, packed K-block order, or prefetch/unroll structure. Record timing, output equivalence, and Amdahl impact before any real-model hook.

## Completed H60 Record

- Observed gap/target: H56's exact x16 tile copied output tiling/activation reuse but still used Cactus's strided 4-output block layout. XNNPACK packs each 16-output tile so K-block bytes are contiguous. Target remains 8-9 tok/s decode, but the immediate test was whether packed order is a large part of the XNNPACK gap.
- Mechanisms separated: XNNPACK-style tile-major K-block locality vs exact Cactus CQ4 codebook/per-group math vs true affine signed-int4/I8MM execution.
- Chosen path/share: added `packed_tbl_x16_tilepack_dot_norm` in `tests/android/cq4_kernel_probe.cpp`. It repacks compact CQ4 bytes into a 16-output tile-major order for the probe while preserving the same codebook and norm math as the prior exact x16 path.
- Evidence:
  - Results: `experiments/matrix/results/h60_xnnpack_tilepack_exact_cq4_20260521/`.
  - First run: `gemma_ffn_up` current 0.406939 ms, prior x16 0.417663 ms, tile-packed x16 0.369040 ms; `gemma_ffn_down` current 0.406820 ms, prior x16 0.385316 ms, tile-packed 0.372796 ms; `gemma_attn_q` current 0.082400 ms, tile-packed 0.076673 ms.
  - Repeat/extra major-shape weighted result: current 38.723 ms vs tile-packed exact CQ4 35.159 ms across the known Gemma major linear counts, a 1.101x weighted kernel speedup.
  - Amdahl using H45's 40.34% current interleaved generic GEMV share predicts only about 5.77 tok/s from the 5.56 tok/s baseline if this generic win transferred perfectly into the model.
- Decision: supported as a small exact-CQ4 improvement but rejected as the main path to XNNPACK performance. Packed order/locality matters, and FFN-up now improves instead of regressing, but the result is still far from XNNPACK's 1.5-2x hot-shape speed class. This strengthens the conclusion that the core XNNPACK advantage is not just tiling or packing order; it is the compact affine signed-int4/I8MM execution contract plus compensation/scaling, or a broader LiteRT graph/runtime contract.

## Next Experiment Record

- Observed gap/target: H60 narrows exact generic CQ4 copying further: exact tile-major packing gives only about 1.10x weighted kernel speedup, while XNNPACK QC4 remains 1.5-2.1x faster on the same class of hot shapes. Target remains 8-9 tok/s decode.
- Mechanisms separated: exact arbitrary-codebook CQ4 limit vs direct XNNPACK operator integration/weight-contract limit vs LM-head structurally compatible folded/int8 special case.
- Chosen path/share: stop spending effort on more minor exact-CQ4 packing tweaks unless a new mechanism has a predicted Amdahl path. The next high-signal split should either integrate/copy the true XNNPACK affine signed-int4 path behind a measured semantic gate, or use the H59 LM-head path as a special-case diagnostic while explicitly keeping generic CQ4 unresolved.
- Predicted signatures:
  - If a direct XNNPACK operator diagnostic on a structurally compatible weight reproduces the speed in Cactus context, the remaining work is integration and correctness, not microkernel theory.
  - If generic affine semantic gates remain around H55/H57 error levels, generic Cactus CQ4 cannot become XNNPACK-fast without changing the weight function or model artifact.
  - If the LM-head hook moves decode near the H59 Amdahl estimate, the Pixel slowdown is partly a removable Cactus output-head implementation issue, but generic CQ4 still needs a separate answer.
- Falsifier: direct XNNPACK integration cannot reproduce the standalone speed inside the Cactus build, or the LM-head hook fails to move decode by the predicted amount despite its 45.04% measured share.
- Cheapest decisive test: either fix the Android XNNPACK/KleidiAI link path enough for an env-gated Cactus-side operator diagnostic, or implement the H59 LM-head hook as a scoped special-case experiment. Do not run a full matrix until one of those moves strict Gemma 512 CPU7 decode materially.

## Planned H61 Record

- Observed gap/target: H60 shows exact Cactus CQ4 packing changes are too small, while H51/H52 show direct XNNPACK QC4/QC8 operators have the speed class needed for 8-9 tok/s if a compatible weight path exists. The current blocker is whether the direct XNNPACK operator path can be used from the normal Cactus Android build/runtime, not only from the API-26 side probe.
- Mechanisms separated: normal Cactus Android API/build/link compatibility vs XNNPACK/KleidiAI static archive requirements vs real runtime operator overhead.
- Chosen path/share: first test the smallest build-only gate: build the existing `xnnpack_qc4_probe` target from `android/build` after the required Cactus build. This does not change model behavior and only determines whether the direct XNNPACK diagnostic path is locally integrable.
- Predicted signatures:
  - If the normal build links, we can run the existing operator probe from the normal build and then consider a scoped Cactus-side diagnostic.
  - If it fails with the known `stdout`/`stderr` or API-level symbol issue, the immediate blocker is dependency/build compatibility, and the next action is either fix the Android API/dependency path or keep using the side build for standalone evidence only.
  - If it links but standalone timings regress from the side build, the integration/build flags changed the selected XNNPACK path and must be profiled before any hook.
- Falsifier: normal `android/build` cannot link `xnnpack_qc4_probe`, or the linked probe does not reproduce the side-build XNNPACK speed class on Pixel CPU7.
- Cheapest decisive test: `source ./venv/bin/activate && cactus build && cmake --build android/build --target xnnpack_qc4_probe -j $(sysctl -n hw.ncpu)`, then run the normal-built probe only if linking succeeds.

## Completed H61 Record

- Evidence:
  - Build log: `experiments/matrix/results/h61_normal_xnnpack_build_gate_20260521_build.log`.
  - Normal Cactus Android build cache: `ANDROID_PLATFORM=android-21`.
  - Working side XNNPACK probe build cache: `ANDROID_PLATFORM=android-26`.
  - Normal `android/build` fails linking `xnnpack_qc4_probe` from the ExecuTorch XNNPACK/KleidiAI static archives:
    - `undefined symbol: stdout`
    - `undefined symbol: stderr`
    - references originate in `libkleidiai.a`, for example `kai_lhs_quant_pack_qai8dxp_f32.c.o`.
- Decision: H61 is a precise integration blocker for direct XNNPACK use in the normal Cactus Android build. The XNNPACK speed evidence remains valid from the API-26 side probe, but an in-model Cactus hook cannot use these static archives from the current API-21 build without fixing the dependency/API path or accepting an API-26 diagnostic build.

## Completed H62 Record

- Observed gap/target: H59's rowwise-int8 folded LM-head looked plausible from weight-only sampled output error, but XNNPACK QC8 timing assumes dynamic qd8 activation quantization. Before writing a hook, the missing semantic gate was activation quantization error on real hidden vectors.
- Mechanisms separated: folded rowwise-int8 weight error vs dynamic qd8 activation error vs sampled-rank/top-logit stability.
- Chosen path/share: replay the 170 H59 captured hidden vectors against 2048 sampled folded LM-head rows, comparing exact folded logits to rowwise-int8 weight-only and rowwise-int8 plus dynamic qd8 activation approximations.
- Evidence:
  - Results: `experiments/matrix/results/h62_lmhead_qd8_activation_error_20260521/qd8_activation_error_stdout.txt` and `qd8_sampled_rank_stdout.txt`.
  - Decode last 100 hidden vectors:
    - Weight-only rowwise int8 relative L2 mean 0.006428, p95 0.008177, max 0.008561.
    - Symmetric qd8 activation plus rowwise int8 weights relative L2 mean 0.023144, p95 0.035536, max 0.042504.
    - Affine qd8 activation plus rowwise int8 weights relative L2 mean 0.019167, p95 0.029623, max 0.032170.
    - Sampled-vocab rank over 2048 rows: weight-only top1 match 100%; affine qd8 top1 match 100%; affine qd8 top-logit absolute error mean 0.059902, p95 0.159615, with exact sampled top margin mean 1.789592 and p05 0.439672.
- Decision: qd8 activation quantization adds measurable drift but does not reject the LM-head diagnostic. The relative L2 rises from about 0.64% to about 1.9% on decode hidden vectors, while sampled-rank stability remains strong. This is enough to justify a scoped LM-head real-model hook as a diagnostic, but not enough for production correctness acceptance without full-vocab/model-output checks.

## Next Experiment Record

- Observed gap/target: H61 blocks normal API-21 direct XNNPACK integration, while H62 keeps the XNNPACK-compatible rowwise-int8/qd8 LM-head path alive as a diagnostic. H59 predicts 8.5-8.8 tok/s if the LM-head bucket executes near XNNPACK QC8 speed.
- Mechanisms separated: API-26 diagnostic build viability vs native Cactus int8 execution speed vs direct XNNPACK QC8 operator integration.
- Chosen path/share: first verify whether the existing API-26 side build can also build the full Cactus `cactus_llm_bench` target, not just `xnnpack_qc4_probe`. If yes, the next implementation can use an API-26 diagnostic runner for XNNPACK-backed LM-head experiments without disturbing the normal API-21 build.
- Predicted signatures:
  - If API-26 `cactus_llm_bench` builds and runs the no-env Gemma scout at the same baseline, it is a valid diagnostic harness for an XNNPACK hook.
  - If API-26 bench build or run fails, direct XNNPACK model hooks are blocked harder, and the next diagnostic should be a native Cactus rowwise-int8 LM-head hook.
  - If API-26 no-env baseline differs materially, any hook result must be compared against adjacent API-26 baseline rather than the normal API-21 baseline.
- Falsifier: API-26 side build cannot build or run `cactus_llm_bench`, or the no-env baseline is too different to use for Amdahl comparison.
- Cheapest decisive test: build `cactus_llm_bench` in `android/build_xnnpack_probe`, deploy it under a distinct name, and run one strict Gemma 512 CPU7 no-env scout only if the build succeeds.

## Completed H63 Record

- Evidence:
  - Results: `experiments/matrix/results/h63_api26_cactus_bench_gate_20260521/`.
  - API-26 side build successfully built `cactus_llm_bench`.
  - Deployed as `/data/local/tmp/cactus_matrix/bin/cactus_llm_bench_api26`.
  - Strict Gemma 512 CPU7 no-env run: decode 5.61 tok/s, prefill 28.49 tok/s, TTFT 17970.53 ms, total 35610.30 ms, peak RAM 2452.79 MB.
- Decision: API-26 side build is a valid diagnostic harness for future XNNPACK-backed model experiments. Its no-env baseline matches the current normal-build baseline closely enough that hook results can use an adjacent API-26 baseline.

## Planned H64 Record

- Observed gap/target: H59/H62 keep the rowwise-int8 qd8 LM-head path alive semantically, but H58's speed estimate uses XNNPACK QC8 at 14.47 ms for the output-head shape. Before writing a full model hook, we need to know whether a native Cactus dense-int8 implementation can get close to that speed class or whether the hook must be XNNPACK-backed.
- Mechanisms separated: native rowwise-int8/q8 dense GEMV speed vs XNNPACK QC8 operator speed vs current orthogonal CQ4 output-head speed.
- Chosen path/share: add a standalone `cq4_kernel_probe` output-head benchmark that uses dense rowwise-int8 weights and qd8 activations for `K=1536,N=262144`, with SDOT over four output rows. This is only a speed scout; H62 carries the semantic gate.
- Predicted signatures:
  - If native SDOT dense int8 is near 14-20 ms, a native LM-head hook may be enough to approach 8+ tok/s.
  - If native SDOT is much slower, especially above about 35-40 ms, the real-model hook needs XNNPACK/KleidiAI or another optimized packed kernel to hit the H59 Amdahl estimate.
  - If it is slower than current orthogonal output-head timing, the rowwise-int8 hook should not be implemented natively.
- Falsifier: native dense-int8 output-head probe is far below the required speed class.
- Cheapest decisive test: extend `tests/android/cq4_kernel_probe.cpp`, rebuild/deploy only `cq4_kernel_probe`, and run `gemma_orthogonal orthogonal_qd8_qc8_dot4` on Pixel CPU7.

## Completed H64 Record

- Evidence:
  - Results: `experiments/matrix/results/h64_native_dense_qd8_qc8_output_head_20260521_stdout.txt` and `h64_native_dense_qd8_qc8_output_head_20260521_repeat100_stdout.txt`.
  - First run, 20 iterations: `orthogonal_qd8_qc8_dot4_ms=17.522681`.
  - Repeat, 100 iterations: `orthogonal_qd8_qc8_dot4_ms=18.112995`.
  - Compared with H59 real LM-head average 80.226 ms, the repeated native dense-int8 probe gives a 4.43x bucket speedup. At H59's 45.04% decode share, Amdahl predicts about 8.54 tok/s from the 5.56 tok/s baseline.
  - Compared with H58 XNNPACK QC8 14.470 ms, native SDOT dense int8 is about 25% slower but still in the right speed class.
- Decision: supported for a real-model env-gated hook. Unlike the earlier exact-CQ4 generic probes, this native dense-int8 LM-head path has enough measured speed and Amdahl headroom to plausibly reach the 8 tok/s target. It remains a diagnostic because it changes effective LM-head math and adds about 385 MiB cache.

## Planned H65 Record

- Observed gap/target: H64 predicts about 8.54 tok/s if the native dense rowwise-int8/qd8 output-head probe transfers to the real LM-head call. Target remains 8-9 tok/s decode.
- Mechanisms separated: standalone dense-int8 probe speed vs real cache-build/init cost vs real-model dispatch/output overhead vs qd8/folded-weight correctness drift.
- Chosen path/share: implement the smallest env-gated native LM-head hook for the structural Gemma LM-head only (`bits=4,K=1536,N=262144,group_size=1536,num_groups=1,orthogonal`). Build a folded rowwise-int8 cache once, dynamically quantize each hidden vector, and run a 4-row SDOT dense-int8 output head.
- Predicted signatures:
  - Strict Gemma 512 CPU7 decode should move from about 5.56 tok/s toward 8.3-8.6 tok/s if real execution matches H64.
  - Peak RAM should rise by roughly 385 MiB plus allocator overhead before any source-page dropping.
  - TTFT/init may worsen from cache materialization; decode and TTFT must be reported separately.
  - If real decode remains far below Amdahl, cache layout/build overhead or hook dispatch differs from the standalone probe.
- Falsifier: real-model decode movement is far below Amdahl, RAM/init is unacceptable even for diagnostic, or correctness/sampled-output gates fail.
- Cheapest decisive test: implement `CACTUS_LMHEAD_QD8_QC8_DIAG=1`, rebuild/deploy `cactus_llm_bench`, and run one strict Gemma 512 CPU7 scout with adjacent no-env baseline if needed.

## Completed H65 Record

- Evidence:
  - Code: `cactus-kernels/src/matmul.cpp`, env var `CACTUS_LMHEAD_QD8_QC8_DIAG=1`.
  - Results: `experiments/matrix/results/h65_lmhead_qd8_qc8_hook_20260521/`.
  - First hook run: decode 8.75 tok/s, prefill 3.49 tok/s, TTFT 146708.86 ms, total 158020.50 ms, peak RAM 2839.86 MB.
  - Repeat hook run: decode 8.64 tok/s, prefill 3.42 tok/s, TTFT 149922.51 ms, total 161375.28 ms, peak RAM 2835.93 MB.
  - Adjacent no-env run from the same rebuilt binary: decode 5.46 tok/s, prefill 27.81 tok/s, TTFT 18413.48 ms, total 36538.82 ms, peak RAM 2457.09 MB.
  - Hook stderr shows two cache builds per process:
    - first run build times 65592.922 ms and 67453.914 ms.
    - repeat build times 67287.112 ms and 68750.481 ms.
  - Hook per-call output-head run times settle around 21-22 ms after the first call, matching the H64 speed class closely enough to explain the decode movement.
- Decision: validated as a real-model decode diagnostic partial fix. The Pixel M=1 decode gap is substantially explained by the LM-head/output-head implementation: replacing that one bucket with native rowwise-int8/qd8 moves strict Gemma 512 CPU7 decode to the target range. This is not a production fix or goal completion because it changes LM-head math, has only sampled correctness evidence, doubles cache build during init, destroys prefill/TTFT, adds about 379-383 MB peak RAM, and lacks Samsung/second-shape guardrails.

## Next Experiment Record

- Observed gap/target: H65 reaches the decode target but fails the broader production gate on init/prefill. The largest immediate defect is not per-token decode speed; it is on-device cache materialization, especially two 65-69 s builds per process.
- Mechanisms separated: on-device folding cost vs offline/stored folded-int8 LM-head artifact vs row-major SDOT kernel quality vs main-branch 4-row interleaved I8MM kernel quality.
- Chosen path/share: store the folded rowwise-int8 LM-head as a sidecar artifact and load it under `CACTUS_LMHEAD_QD8_QC8_CACHE_PATH`. Use the main-branch int8 kernel layout as the reference: 4-row interleaved weights with `vmmlaq_s32`, while retaining H65's affine activation zero-point correction.
- Predicted signatures:
  - Sidecar load should replace 133-136 s of on-device builds with a much smaller file read, and should happen once per process.
  - Decode should stay in or above the H65 8.6-8.8 tok/s band if row-major SDOT was not hiding a kernel quality issue.
  - If main-style I8MM is better for this shape, per-call output-head time should fall below H65's settled 21-22 ms and move decode closer to LiteRT-LM.
  - Peak RAM should remain in the same broad class as H65 unless mmap/lifetime is changed later.
- Falsifier: sidecar load remains TTFT-dominant, decode regresses materially from H65, or output-head logs show the I8MM path is slower than the H64/H65 SDOT class.
- Cheapest decisive test: generate the sidecar from `token_embeddings.weights`, push it to the Pixel, rebuild/redeploy `cactus_llm_bench`, and run one strict Gemma 512 CPU7 scout with `CACTUS_LMHEAD_QD8_QC8_DIAG=1 CACTUS_LMHEAD_QD8_QC8_CACHE_PATH=...`.

## Planned H66 Record

- Observed gap/target: H65 decode reaches 8.64-8.75 tok/s, but production needs the folded LM head stored rather than built on device. The target is to preserve decode while restoring TTFT/prefill toward the adjacent no-env run.
- Mechanisms separated: stored artifact read/copy cost vs H65 fold cost; main-branch I8MM layout vs H65 row-major SDOT layout; affine qd8 correction overhead vs dense int8 dot cost.
- Chosen path/share: H66 implements a diagnostic sidecar format (`CLM8` v1) containing float32 row scales, int32 row sums, and 4-row interleaved int8 weights. Runtime loading is env-gated, shape-gated to the Gemma LM-head, and leaves existing CQ4 models unchanged by default.
- Predicted signatures:
  - `sidecar_loaded` appears once, with no `cache_built` lines.
  - TTFT loses the 130 s build penalty; any remaining overhead is attributable to sidecar read/copy and ordinary model setup.
  - Per-token decode remains at least near H65, and ideally improves if main's I8MM kernel is superior on Pixel.
- Falsifier: load fails/falls back to CQ4, sidecar loaded more than once, TTFT remains near H65, or decode drops toward baseline.
- Cheapest decisive test: build sidecar locally, push once, run a strict 512 decode scout on Pixel CPU7, then compare against H65 and adjacent no-env.

## Completed H66 Record

- Code:
  - `cactus-kernels/src/matmul.cpp`: `CACTUS_LMHEAD_QD8_QC8_CACHE_PATH` sidecar loader plus main-branch-style 4-row interleaved I8MM LM-head kernel.
  - `experiments/matrix/build_lmhead_qd8_qc8_sidecar.py`: offline generator for `CLM8` v1 sidecars from orthogonal CQ4 LM-head weights.
- Results: `experiments/matrix/results/h66_stored_lmhead_i8mm_sidecar_20260521/`.
- Sidecar artifact:
  - Local file `lmhead_qd8_qc8_i8mm.cache`: 404,750,380 bytes.
  - Device path `/data/local/tmp/cactus_matrix/cache/lmhead_qd8_qc8_i8mm.cache`.
- Evidence:
  - 1-token scout: TTFT 13424.93 ms, prefill 38.14 tok/s, peak RAM 2449.28 MB.
  - 100-token scout: TTFT 13527.39 ms, total 24450.23 ms, prefill 37.85 tok/s, decode 9.06 tok/s, peak RAM 2447.54 MB.
  - Adjacent no-env same rebuilt binary: TTFT 18302.07 ms, total 36138.08 ms, prefill 27.97 tok/s, decode 5.55 tok/s, peak RAM 2452.79 MB.
  - Sidecar load appeared once per process and took about 213-215 ms. No `cache_built` lines appeared.
  - Main-style I8MM per-call output-head times were about 15-17 ms, improving over H65's settled row-major SDOT 21-22 ms.
  - Disassembly contains `smmla`, confirming I8MM codegen for the new path.
  - Sidecar numeric check on 8 real H59 hidden samples and 2048 sampled vocab rows: weight rel L2 mean 0.00970, output rel L2 mean 0.00571, max 0.00618, cosine mean 0.999985, sampled top-1 match 8/8.
  - Pixel thermal status after the runs: 0.
- Decision: validated as the first production-shaped Pixel decode result. Storing the folded LM-head as int8 and using a main-branch-style I8MM layout removes the H65 TTFT failure and reaches the requested 9 tok/s target on strict Pixel Gemma 512 CPU7. This is still not final production because the path is env-gated, Gemma-shape-specific, uses a sidecar outside the converter/model format, and still needs stronger correctness/quality gates plus Samsung/second-shape guardrails.

## Latest Evidence

- H34 full pair-repack diagnostic: `experiments/matrix/results/generic_actualshape_repack_model_diag_20260520_2121/`, 5.56 -> 5.94 tok/s, peak RAM about 2455 -> 3046 MB.
- H36 page-rounded source drop: `experiments/matrix/results/generic_pair_repack_madvise_diag_20260520_2132/`, RAM near baseline but only partial speed.
- H37 stats: `experiments/matrix/results/generic_pair_repack_stats_20260520_2144/`, 64 cache builds, 622,854,144 bytes; madvise time only 10.637 ms.
- H38 interior-only source drop: `experiments/matrix/results/generic_pair_repack_inner_madvise_20260520_2144/`, baseline 5.55, interior 5.66, interior repeat 5.64 tok/s with RAM near baseline.
- H39 combined diagnostic: `experiments/matrix/results/generic_ortho_combined_diag_20260520_2154/`, current-session baseline avg 5.50 tok/s, combined avg 6.12 tok/s, RAM unchanged near 2452 MB.
- H40 residual profile: `experiments/matrix/results/h39_combined_residual_profile_20260520_2210/`, profiled decode 6.19 tok/s, 131,389 samples, no lost samples, flat self time orthogonal 35.86%, pair-repack generic 31.15%, quant matmul 8.37%, default generic 6.20%.
- H42 anonymous same-layout copy: `experiments/matrix/results/generic_anon_copy_diag_fixed_20260520_2225/`, baseline 5.55 tok/s, copy-only 5.52 tok/s, RAM 2450.51 -> 4019.41 MB; rejects mmap residency as primary for generic decode.
- Local-source H39 parity attempt: `experiments/matrix/results/h39_local_source_parity_20260520_2215/`, baseline 5.56 tok/s, reconstructed combined diagnostic 5.40 tok/s, RAM 2461 MB; did not reproduce stale-binary H39.
- Local-source H39 isolation: `experiments/matrix/results/h39_local_source_isolation_20260520_2215/`, orthogonal-only 5.93 tok/s, pair-only 5.16 tok/s, pair-inner 5.17 tok/s.
- H45 local orthogonal residual profile: `experiments/matrix/results/local_ortho_residual_profile_20260520_2215/`, profiled decode 5.76 tok/s, 137,748 samples, no lost samples, flat self time current interleaved generic GEMV 40.34%, orthogonal parallel task 34.37%, generic `cactus_quant_matmul` task 7.98%, attention kernels 6.22%.
- H46 current generic stats: `experiments/matrix/results/h46_current_generic_stat_20260520_2240/`, orthogonal-enabled decode 6.02/5.87 tok/s under stat, backend stalls about 33% of cycles, frontend stalls about 0.7%, L1D miss rate 1.48%, dTLB miss rate 0.16%.
- H46 split-panels diagnostic: `experiments/matrix/results/h46_split_panels_model_diag_20260520_2240/`, orthogonal-only 6.04 tok/s, orthogonal+split 5.75 tok/s. Rejected.
- H47 LiteRT-LM same-context comparison: `experiments/matrix/results/matrix_pixel_10a_litert_lm_gemma_4_e2b_512.csv`, Pixel strict CPU7 one-thread Gemma 512 prefill 46.23 tok/s, decode 9.88 tok/s, peak RAM 3382.18 MB.
- H48 LiteRT-LM cache separation: `experiments/matrix/results/h48_litert_cache_decode_20260521/`, cache-enabled decode 10.04 tok/s and prefill 47.56 tok/s; `--disable_cache=true` decode 9.55 tok/s and prefill 41.85 tok/s, while executor init rose 580.90 ms -> 9969.63 ms and peak system RAM 3382 MB -> 4794.6 MB. Persistent XNNPACK cache is not the primary steady-state decode source.

## LiteRT-LM Artifact Findings

- Pixel has `/data/local/tmp/cactus_matrix/litert_lm/models/gemma-4-e2b-it-int4.litertlm` plus a 752 MB `.xnnpack_cache`.
- Local `.litertlm` inspection found 12 sections. The `tf_lite_prefill_decode` section is 818,275,184 bytes and exposes `decode`, `prefill_1024`, `prefill_128`, and `verify` signatures.
- The decode subgraph has 2068 ops: 277 `FULLY_CONNECTED`, 276 `DEQUANTIZE`, 211 `QUANTIZE`, 327 `STABLEHLO_COMPOSITE`, and mixed `INT4`/`INT8`/`FLOAT32` tensors.
- Direct Pixel logs show XNNPACK delegation over most of the relevant graph. This points to compiled delegated int4/per-output-channel operator execution and graph contract as the next comparison target, not cache-file reuse alone.

## Cactus-vs-LiteRT Linear Contract Findings

- Cactus Gemma decode IR has 276 `linear` nodes. The hot language projection files are CQ4 interleaved with group size 128.
- Cactus shape counts include 40 `(12288,1536)`, 30 `(6144,1536)`, 28 `(2048,1536)`, 28 `(1536,2048)`, 20 `(1536,12288)`, and 15 `(1536,6144)`.
- LiteRT `decode` has 277 `FULLY_CONNECTED` ops. Most are quantized: 145 `INT8 x INT4 -> INT8`, 71 `INT8 x INT8 -> INT8`, and 60 `INT8 x type19 -> INT8`.
- LiteRT FC weight quantization metadata is per output row/channel, for example `(2048,1536)` weights have 2048 scales and `(12288,1536)` weights have 12288 scales.
- H49 next probe candidate: approximate the LiteRT per-output-channel int4/int8 FC contract on Cactus hot shapes and compare against current CQ4 interleaved GEMV before attempting a real-model hook.
