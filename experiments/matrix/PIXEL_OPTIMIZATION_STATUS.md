# Pixel Optimization Status

Active goal: finish the primary single-thread CSV/docs commit, then root-cause and fix the remaining full-core prefill slowdown on Pixel 10a.

## Current State

- Decode is no longer the open mechanism scout target. The remaining decode work is productionization and guardrails.
- H66 is the first production-shaped real-model diagnostic to reach the requested decode target: stored folded rowwise-int8 LM-head sidecar plus a main-branch-style 4-row interleaved I8MM kernel gives 9.06 tok/s on strict Pixel Gemma 512 CPU7, with TTFT 13.53 s and prefill 37.85 tok/s. Adjacent no-env from the same rebuilt binary is 5.55 tok/s decode, TTFT 18.30 s, prefill 27.97 tok/s.
- H67 guardrails passed: Samsung strict Gemma 512 sidecar improved 13.86 -> 20.46 tok/s, and Pixel strict Gemma 1024 sidecar improved 5.56 -> 8.64 tok/s. No repeated sidecar loads, no runtime cache builds, thermal status 0.
- H68 integrated model-local sidecar discovery passed: with no model-local sidecar, Pixel Gemma 512 stayed at decode 5.57 tok/s and prefill 28.13 tok/s; with `token_embeddings.lmhead_qd8_qc8.cache` next to the weight file and no env vars, Pixel Gemma 512 reached decode 8.96 tok/s and prefill 37.52 tok/s.
- Optimized single-thread trajectory regeneration is complete for Gemma, LFM, and Qwen. The H72 LFM/Qwen rows were manually patched into `experiments/matrix/results/matrix_pixel_10a_single_thread.csv` immediately after `experiments/matrix/results/matrix_pixel_10a_cactus_lfm_qwen_optimized.csv` landed.
- Fresh H66-post no-change Pixel Tier 1/Tier 2 baselines are stable: 5.56 and 5.55 tok/s decode, 28.21 and 27.69 tok/s prefill, thermal status 0, RAM 2452.35-2452.90 MB.
- H65 first proved the LM-head mechanism: env-gated native rowwise-int8/qd8 LM-head replacement gave 8.75 and 8.64 tok/s repeats, but it was not production-shaped because TTFT rose to 146-150 s from two on-device cache builds, prefill collapsed to 3.4-3.5 tok/s, RAM rose to about 2.84 GB, and correctness was only sampled.
- Single-core prefill wrong-core placement is considered solved, pending integration validation.
- Do not rerun the single-thread trajectory matrix again unless acceptance needs cooled repeats. The primary single-thread CSV/docs update is ready to commit and push.
- The next research target after that regeneration is full-core prefill. Decode history remains evidence, but new full-core prefill work must start from full-core prefill baselines/profiles rather than inherited decode assumptions.
- Same-device LiteRT-LM CPU comparison at Gemma 512 is now 9.88 tok/s decode and 46.23 tok/s prefill under strict Pixel CPU7 one-thread settings, so the target is realistic on Pixel CPU with the right artifact/runtime contract.
- Historical decode findings that led to H66:
  - Orthogonal path: Pixel is slow on FP16-to-FP32 `fcvtl/fcvtl2`; the locally rebuilt FP32-activation diagnostic moved Pixel decode 5.56 -> 5.93 tok/s.
  - Generic interleaved path: after the working local orthogonal diagnostic, current-source generic GEMV is the top residual self-time bucket at 40.34%.
- Rejected current explanations:
  - CPU7 pinning alone does not fix Pixel decode.
  - Same-layout anonymous copies do not improve decode, so mmap/file-backed residency is not the primary mechanism.
  - The reconstructed current-source pair-layout path regresses Gemma 512 decode and should not be used for current implementation decisions.
  - i8mm alone did not fix generic CQ4: the H54 expanded-int8 generic probe regressed dominant FFN shapes. It did help the stored-int8 LM-head sidecar once the weight layout matched the main-branch 4-row interleaved kernel contract.

## Latest H66 Result

- Code:
  - `cactus-kernels/src/matmul.cpp`: `CACTUS_LMHEAD_QD8_QC8_CACHE_PATH` sidecar loader and I8MM LM-head path.
  - `experiments/matrix/build_lmhead_qd8_qc8_sidecar.py`: offline sidecar generator.
- Results: `experiments/matrix/results/h66_stored_lmhead_i8mm_sidecar_20260521/`.
- Sidecar: 404,750,380-byte `CLM8` artifact, containing float32 row scales, int32 row sums, and 4-row interleaved int8 weights.
- Strict Pixel CPU7 Gemma 512:
  - sidecar + I8MM: decode 9.06 tok/s, prefill 37.85 tok/s, TTFT 13527.39 ms, total 24450.23 ms.
  - adjacent no-env: decode 5.55 tok/s, prefill 27.97 tok/s, TTFT 18302.07 ms, total 36138.08 ms.
  - sidecar load: one load, about 213 ms; no on-device folded-cache build.
  - output-head call times: about 15-17 ms, better than H65 row-major SDOT 21-22 ms.
  - sampled numeric check: 8 real hidden samples x 2048 vocab rows, output rel L2 mean 0.00571, max 0.00618, cosine mean 0.999985, top-1 match 8/8.
- Decision: validated as a production direction, not yet final production. Next gates are converter/model-format integration, correctness/quality validation beyond sampled hidden-vector errors, Samsung/second-shape guardrails, and deciding whether to mmap the sidecar rather than read it into vectors.

## Latest H67 Guardrails

- Results: `experiments/matrix/results/h67_h66_guardrails_20260521/`.
- Samsung strict Gemma 512:
  - no-env: decode 13.86 tok/s, prefill 64.88 tok/s, TTFT 7890.94 ms.
  - sidecar: decode 20.46 tok/s, prefill 74.66 tok/s, TTFT 6857.52 ms.
- Pixel strict Gemma 1024:
  - no-env: decode 5.56 tok/s, prefill 36.98 tok/s, TTFT 27687.46 ms.
  - sidecar: decode 8.64 tok/s, prefill 43.51 tok/s, TTFT 23536.50 ms.
- Decision: speed/TTFT/RAM guardrails pass; production integration remains the active blocker.

## Latest H68 Integration

- Results: `experiments/matrix/results/h68_integrated_lmhead_sidecar_20260521/`.
- Code path:
  - loader discovers `token_embeddings.lmhead_qd8_qc8.cache` next to `token_embeddings.weights`.
  - `CactusQuantMatrix` carries the optional sidecar path.
  - the H66 I8MM LM-head path activates without env vars only when the sidecar exists.
- Pixel strict Gemma 512:
  - sidecar absent: decode 5.57 tok/s, prefill 28.13 tok/s, TTFT 18201.96 ms.
  - sidecar present, no env vars: decode 8.96 tok/s, prefill 37.52 tok/s, TTFT 13645.59 ms.
- Decision: integrated discovery works; the immediate production focus is now benchmark regeneration and full-core prefill competitiveness.

## Active Work Queue

1. Finish H66/H68 productionization and guardrails.
   - Integrate the stored LM-head int8/I8MM artifact path into the converter/model format or document the precise blocker.
   - Validate decode speed, TTFT, prefill side effects, RAM, correctness/quality, Samsung, and a second model/shape. H67/H68 cover speed, TTFT, RAM, Samsung, and second shape; converter emission is remaining packaging polish, while prefill competitiveness is now the practical production focus.
2. Regenerate single-thread trajectories through context 4096 for all Cactus LLMs.
   - Status: completed for Gemma, LFM, and Qwen with optimized output-head sidecars active.
   - Explicit scope: Gemma 4 E2B, LFM 2.5 VL 1.6B, and Qwen3 VL 2B.
   - Existing LFM/Qwen rows must not be carried forward as optimized results. The Gemma output-head sidecar/I8MM optimization, or an equivalent optimized output-head path if their artifacts differ, must be adapted to LFM and Qwen before rerunning those rows.
   - Gemma is first because it validates the optimized path, not because LFM/Qwen are out of scope.
   - Include single-thread decode and prefill.
   - Boss-critical handoff: when optimized LFM/Qwen CSV rows are produced, immediately manually patch those exact rows into `experiments/matrix/results/matrix_pixel_10a_single_thread.csv`, document the patch, then commit and push the primary CSV/docs update before continuing long-running work.
   - Commit and push results immediately after validation.
   - Commit only updates to the primary single-thread CSVs, not scratch/intermediate/new one-off CSVs generated during the run.
3. Start the full-core prefill-only investigation.
   - Use fresh full-core prefill baselines/profiles and the tier ladder in `MAY_19_NIGHT_PLAN.md`.
4. After full-core prefill is fixed or precisely blocked, rerun full-core prefill benchmarks through context 4096 for Gemma 4 E2B, LFM 2.5 VL 1.6B, and Qwen3 VL 2B, then commit and push only updates to the primary full-core CSVs.

## Next Experiment Record

- Observed gap/target: H68 preserves the decode win and improves single-thread Gemma 512 prefill, but the next production concern is competitive prefill numbers and benchmark regeneration before full-core prefill investigation.
- Mechanisms separated: single-thread optimized decode/prefill validation vs full-core prefill scheduling/kernel/memory bottleneck vs benchmark artifact bookkeeping.
- Chosen path/share: H69 will regenerate the primary optimized Gemma single-thread trajectory first, using the model-local H68 sidecar path. Full-core kernel experiments wait until this validation is recorded.
- Predicted signatures:
  - Gemma single-thread decode trajectory should stay near the H66/H67/H68 class where LM-head dominates.
  - Single-thread prefill should be reasonable and not collapse under the integrated sidecar path.
  - The run should update only intended primary CSVs, not scratch outputs.
- Falsifier: integrated sidecar fails outside 512/1024, prefill regresses versus no-sidecar, or the benchmark harness does not pick up the optimized artifact.
- Cheapest decisive test: inspect the matrix commands/primary CSV targets, then run the smallest optimized Gemma single-thread trajectory regeneration covering decode and prefill through context 4096.
- Result: `experiments/matrix/results/matrix_pixel_10a_cactus_gemma_fast.csv` was regenerated on Pixel strict CPU7 with the integrated model-local sidecar path active. Gemma rows are 256 prefill/decode 26.12/9.03 tok/s, 512 34.78/8.66 tok/s, 1024 38.18/7.78 tok/s, 2048 36.71/6.87 tok/s, and 4096 prefill 38.58 tok/s. This validates the integrated optimized Gemma path for trajectory regeneration, but the full single-thread regeneration still requires adapting and rerunning LFM 2.5 VL 1.6B and Qwen3 VL 2B under equivalent optimized output-head paths.
- H71 transfer result: LFM and Qwen token embeddings are also CQ4 one-group orthogonal LM heads. The sidecar builder, loader discovery, and runtime hook were generalized beyond Gemma's shape. Generated sidecars are LFM `(N=65536,K=2048)`, 129 MB, and Qwen `(N=151936,K=2048)`, 298 MB. Pixel strict CPU7 512 scouts with sidecars active reached LFM prefill/decode 33.80/17.49 tok/s and Qwen 21.99/10.91 tok/s, proving the optimization direction transfers. Full optimized LFM/Qwen trajectories are still required.
- H72 result: full optimized LFM/Qwen trajectories completed and were manually patched into `experiments/matrix/results/matrix_pixel_10a_single_thread.csv`. LFM rows are 256 prefill/decode 24.73/16.28 tok/s, 512 24.53/11.81 tok/s, 1024 28.56/13.63 tok/s, 2048 29.27/14.11 tok/s, and 4096 prefill 27.92 tok/s. Qwen rows are 256 prefill/decode 12.21/8.28 tok/s, 512 16.33/7.85 tok/s, 1024 18.14/6.88 tok/s, 2048 18.26/7.12 tok/s, and 4096 prefill 19.48 tok/s. Qwen 512+ rows recorded `thermal_status=1`, so cooled repeats may be needed for final acceptance, but the requested primary CSV patch is complete.

## Active Ledger

The ledger below is chronological. Older entries may contain `Next Experiment Record` language from prior phases; those are historical unless the Active Work Queue or the current `Next Experiment Record` above explicitly re-promotes them.

### H34: pair-adjacent cached layout helps actual Gemma generic GEMV shapes

- Status: `production_supported_diagnostic`
- Evidence:
  - `experiments/matrix/results/generic_actualshape_repack_probe_20260520_2124/`: Pixel `K=1536,N=12288` probe 0.998 -> 0.709 ms and memory-bound stalls 17.0% -> 2.3%; Samsung regressed 0.403 -> 0.439 ms.
  - `experiments/matrix/results/generic_actualshape_repack_model_diag_20260520_2121/`: Pixel baseline 5.56 tok/s, full pair-repack diagnostic 5.94 and 5.94 tok/s repeats, but peak RAM rose about 2455 -> 3046 MB.
- Decision: real-model diagnostic, not production fix because of RAM cost and Samsung regression in the probe.

### H35: bounded pair-repack cache can keep most of the win with less RAM

- Status: `inconclusive_device_state_drift`
- Evidence:
  - `experiments/matrix/results/generic_actualshape_repack_model_diag_20260520_2121/cache_limit_curve/`: limits changed RAM but throughput drift made the curve uninterpretable.
  - `experiments/matrix/results/generic_pair_repack_paired_curve_20260520_2132/`: baseline drifted 5.55 -> 5.36 -> 5.30 tok/s; limit16 had no useful movement.
- Decision: limit16 rejected as useful. Full-cache memory work should use memory-specific diagnostics.

### H36: source mmap pages can be dropped after runtime pair-repack

- Status: `memory_mechanism_supported`
- Evidence: `experiments/matrix/results/generic_pair_repack_madvise_diag_20260520_2132/`
  - baseline: 5.48 tok/s, 2451 MB.
  - full cache plus page-rounded source drop: 5.58 tok/s, 2459 MB.
  - immediate no-madvise full cache: 5.66 tok/s, 3046 MB.
- Decision: source-page dropping almost eliminates the RAM cost while preserving correctness, but speed was not fully preserved.

### H37: H36 throughput loss is first-token repack/drop overhead

- Status: `rejected_as_primary`
- Evidence: `experiments/matrix/results/generic_pair_repack_stats_20260520_2144/`
  - no-madvise: 64 cache builds, 622,854,144 bytes, 235.307 ms copy time, 0.009 ms madvise time.
  - source-drop: 64 cache builds, 622,854,144 bytes, 243.700 ms copy time, 10.637 ms madvise time.
- Decision: madvise syscall/drop time is too small to explain the throughput gap.

### H38: H36 throughput loss comes from dropping edge pages with neighboring tensor data

- Status: `memory_safe_diagnostic_supported`
- Evidence: `experiments/matrix/results/generic_pair_repack_inner_madvise_20260520_2144/`
  - baseline: 5.55 tok/s, 2451 MB.
  - interior-only source drop: 5.66 tok/s, 2452 MB.
  - no-madvise full cache later: 5.58 tok/s, 3046 MB.
  - interior-only repeat: 5.64 tok/s, 2456 MB.
- Decision: interior-only source drop preserves the RAM benefit and avoids obvious throughput loss, but movement is still below production-fix threshold in this device state.

### H39: orthogonal FCVT and generic pair-layout mechanisms are additive

- Status: `partial_fix_diagnostic_supported`
- Observed gap/target: H25 orthogonal FP32 activation diagnostic moved Pixel average decode from 5.58 -> 6.08 tok/s; H38 provides a RAM-safe generic pair-layout diagnostic around 5.64-5.66 tok/s in current state.
- Mechanisms separated: independent orthogonal FCVT mechanism vs generic packed-layout mechanism vs device-state drift/noise.
- Chosen path/share: combine `CACTUS_ORTHO_F32_AR_DIAG=1` with `CACTUS_GENERIC_PAIR_REPACK_DIAG=1 CACTUS_GENERIC_PAIR_REPACK_MADVISE_INNER=1`.
- Predicted signatures:
  - Combined decode should exceed adjacent baseline and either single diagnostic by more than noise.
  - Peak RAM should stay close to baseline due interior source-page drop.
- Falsifier: combined run is not materially above adjacent baseline/single-diagnostic evidence or RAM rises near no-madvise full-cache path.
- Cheapest decisive test: run same-binary Gemma 512 baseline and combined env-gated diagnostic on Pixel.
- Evidence:
  - `experiments/matrix/results/current_nochange_pixel_tier12_20260520_2154/`: current-session no-change baseline decode 5.56 and 5.43 tok/s, peak RAM 2451-2455 MB, thermal status 0.
  - `experiments/matrix/results/generic_ortho_combined_diag_20260520_2154/`: combined diagnostic decode 6.14 and 6.10 tok/s, peak RAM 2452 MB, thermal status 0.
- Decision: supported as a repeated RAM-safe diagnostic partial win. Average decode improved 5.50 -> 6.12 tok/s (+11.4%), leaving about 1.97x residual gap versus tracked Samsung 12.06 tok/s. Not a production fix until residual profile, Samsung guardrail, second model/shape, and a non-env diagnostic implementation path are addressed.

### H40: residual combined decode is still dominated by known MATMUL paths

- Status: `supported`
- Observed gap/target: H39 improves Pixel Gemma 512 decode to 6.10-6.14 tok/s, but residual gap remains about 1.97x versus Samsung 12.06 tok/s.
- Mechanisms separated: residual orthogonal FCVT cost vs residual generic CQ4 GEMV memory/layout cost vs another fallback/dispatcher/kernel path becoming dominant after H39.
- Chosen path/share: profile Gemma 512 decode with H39 combined env enabled, using the same 100-token strict CPU7 scout shape as H39 so the profiled run can be compared against the measured throughput band.
- Predicted signatures:
  - If orthogonal is still dominant, `matmul_q4_0_4x8_q4_0_4x8_orthogonal` remains the top self-time symbol despite the FP32 activation diagnostic.
  - If generic is still dominant, the pair-repack branch still owns a large fraction of self time and needs a lower-overhead/layout-compatible mechanism.
  - If another path dominates, H39 has exposed the next bottleneck and the ledger should pivot before coding.
- Falsifier: profile samples are too few/noisy, symbols are unresolved, or run performance is outside the H39 band.
- Cheapest decisive test: simpleperf profile of a strict Gemma 512 combined-env 100-token decode scout on Pixel.
- Evidence:
  - `experiments/matrix/results/h39_combined_residual_profile_20260520_2210/`: profiled run decode 6.19 tok/s, peak RAM 2456.24 MB, 131,389 samples, 0 lost.
  - Flat self time: orthogonal MATMUL 35.86%, pair-repack generic branch 31.15%, `cactus_quant_matmul` 8.37%, default generic branch 6.20%, attention kernels 6.57%.
- Decision: supported. H39 exposes two remaining large MATMUL buckets rather than a new non-MATMUL bottleneck. Generic/MMAP/layout work can at most attack about 37% directly; orthogonal remains an independent about 36% target.

### H41: i8mm/MMLA can close the remaining generic decode gap

- Status: `available_but_not_currently_used_low_leverage`
- Observed gap/target: H40 profiled H39 at 6.19 tok/s; target is 8-9 tok/s. Reaching 8 tok/s requires about 22.6% total time reduction versus H40.
- Mechanisms separated: current SDOT issue limits vs MMLA higher int8 throughput for multi-output tiles vs non-dot overhead from CQ unpack/layout, scaling, and orthogonal FP conversion.
- Chosen path/share: only the generic/int8 residual is directly relevant, about 37.35% self time from pair-repack and default generic branches. Orthogonal 35.86% cannot be helped by i8mm.
- Predicted signatures:
  - If i8mm is a realistic path, an MMLA-shaped generic probe must cut Pixel generic hot-shape time by at least 60% without large Samsung regression.
  - Binary should contain `smmla`, `ummla`, or `usmmla` in the tested path.
  - Real-model H39-like decode must move by the Amdahl expectation; probe-only wins are insufficient.
- Falsifier: current binary lacks MMLA in hot paths; an MMLA probe fails to beat SDOT by enough on Pixel, or improves only a small slice whose Amdahl impact cannot reach 8 tok/s.
- Current evidence:
  - Pixel `/proc/cpuinfo` advertises `i8mm` and `svei8mm`.
  - `cactus-kernels` and `cactus-engine` compile with `-march=armv8.2-a+fp16+simd+dotprod+i8mm`.
  - Source hot path uses `vdotq_laneq_s32`/SDOT.
  - Local disassembly of `android/build/bin/cactus_llm_bench` shows 302 `sdot` instructions and no `smmla/ummla/usmmla` matches.
- Decision: not tried as an MMLA real-model diagnostic. It is not a plausible sole fix because it cannot touch orthogonal residual time and the current binary is already built with i8mm flags but emits SDOT in the hot path. It is worth only a focused generic hot-shape probe if it can be implemented without changing model layout or acceptance semantics, and if the probe predicts enough real-model Amdahl movement.

### H42: generic slowdown is caused by mmap/file-backed source residency rather than packed layout itself

- Status: `rejected_as_primary`
- Observed gap/target: H40 leaves about 37% direct generic self time under H39. H34/H38 showed pair-adjacent runtime layout helps Pixel and source-page dropping controls RAM, but did not separate mmap residency from packed-layout access order.
- Mechanisms separated: mmap/file-backed page residency/TLB behavior vs packed-layout cache-line access pattern vs runtime pair-repack overhead.
- Chosen path/share: compare the same original packed layout when sourced from mmap-backed model memory versus an anonymous copied buffer for the hot generic matrices. This targets the generic residual only and should not affect orthogonal.
- Predicted signatures:
  - If mmap residency is causal, anonymous same-layout should improve generic probe/model time without pair-adjacent layout.
  - If layout is causal, anonymous same-layout should not materially improve, while pair-adjacent layout still does.
  - If page faults are causal, fault/RSS counters should differ before first decode and then converge after pre-touch.
- Falsifier: anonymous same-layout is within noise on Pixel, or only pair-adjacent layout improves.
- Cheapest decisive test: env-gated anonymous-copy diagnostic for the actual hot generic matrices, with same-layout copy first; run Gemma 512 baseline vs copy-only vs pair-repack.
- Evidence:
  - `experiments/matrix/results/generic_anon_copy_diag_20260520_2225/`: invalid first attempt; cache key used transient `CactusQuantMatrix*` and rebuilt copies repeatedly, so the slowdown was not used for mechanism judgment.
  - `experiments/matrix/results/generic_anon_copy_diag_fixed_20260520_2225/`: corrected cache keyed by source pointers and shape. Baseline decode 5.55 tok/s, anonymous same-layout copy 5.52 tok/s, peak RAM 2450.51 -> 4019.41 MB, thermal status 0.
  - Corrected copy log built 416 unique generic buffers and copied 1,637,627,904 bytes.
- Decision: rejected as primary. Same-layout anonymous memory does not improve real-model decode, so mmap/file-backed residency alone does not explain Pixel’s remaining generic slowdown. The pair-repack wins are more consistent with layout/cache access order than mmap itself.

### H43: remaining path to 8 tok/s requires combined orthogonal and generic layout fixes

- Status: `candidate_stale_binary_not_reproduced_yet`
- Observed gap/target: current local-source baseline is about 5.55 tok/s; stale-binary H39 combined diagnostic reached 6.10-6.19 tok/s; target is 8-9 tok/s.
- Mechanisms separated: orthogonal FP16->FP32 conversion residual vs generic packed-layout/cache residual vs low-leverage instruction substitution such as i8mm-only.
- Chosen path/share: use H40 residual shares as the guide: orthogonal about 35.86%, generic/pair/default about 37.35%, `cactus_quant_matmul` about 8.37%.
- Predicted signatures:
  - A realistic 8 tok/s path must reduce both orthogonal and generic residuals, or reduce one of them by an extremely large amount.
  - Generic-only work needs roughly 60% generic-time reduction from H40 to reach 8 tok/s.
  - Orthogonal-only work needs a similarly large reduction and cannot be solved by i8mm.
- Falsifier: a proposed change targets less than about 20% model self time or lacks a predicted real-model Amdahl movement.
- Cheapest decisive next test: restore local source parity for the orthogonal FP32 activation diagnostic and pair-layout diagnostic, then profile one combined run from the rebuilt local source before adding new kernel work.
- Evidence:
  - `experiments/matrix/results/h39_local_source_parity_20260520_2215/`: rebuilt local-source baseline 5.56 tok/s, reconstructed combined diagnostic 5.40 tok/s, peak RAM 2455 -> 2461 MB, thermal status 0.
- Decision: the reconstructed local-source hooks do not yet reproduce stale-binary H39. Isolate orthogonal-only and pair-only before using these hooks for any new conclusion.

### H44: reconstructed H39 hook failure comes from one bad diagnostic implementation

- Status: `supported_pair_reconstruction_bad`
- Observed gap/target: local-source reconstructed combined diagnostic regressed 5.56 -> 5.40 tok/s instead of reproducing 6.1 tok/s.
- Mechanisms separated: orthogonal FP32 activation reconstruction is harmful vs pair-layout reconstruction is harmful vs interaction between the two.
- Chosen path/share: run individual Gemma 512 scout cases for orthogonal-only, pair-only no source drop, and pair-only with interior source drop.
- Predicted signatures:
  - Orthogonal-only improves if it matches the previous FP32 activation diagnostic.
  - Pair-only improves if the pair-adjacent layout reconstruction matches H34/H38.
  - Combined failure with individual wins would point to interaction or cache/RAM side effect.
- Falsifier: no individual reconstructed diagnostic improves above adjacent baseline.
- Cheapest decisive test: same rebuilt runner, three env-gated Pixel Gemma 512 runs.
- Evidence:
  - `experiments/matrix/results/h39_local_source_isolation_20260520_2215/`: orthogonal-only decode 5.93 tok/s, pair-only 5.16 tok/s, pair-only plus interior source drop 5.17 tok/s.
  - Adjacent baseline from `experiments/matrix/results/h39_local_source_parity_20260520_2215/`: 5.56 tok/s.
- Decision: supported. The orthogonal FP32 activation reconstruction is useful locally (+6.7% decode, baseline RAM). The reconstructed pair-layout path is harmful in current source and should not be used as evidence for a current implementation.

### H45: after local orthogonal diagnostic, residual current-source decode is still MATMUL-dominated

- Status: `supported`
- Observed gap/target: local orthogonal-only diagnostic improves Gemma 512 decode 5.56 -> 5.93 tok/s, still far from 8-9 tok/s.
- Mechanisms separated: residual orthogonal cost vs current interleaved generic path vs non-MATMUL path exposed after orthogonal improvement.
- Chosen path/share: simpleperf profile of local-source orthogonal-only Gemma 512 decode, because this is the only reconstructed diagnostic that currently improves real model throughput.
- Predicted signatures:
  - If residual is still orthogonal-heavy, further orthogonal mechanism work remains viable.
  - If generic interleaved now dominates, design a current-source generic experiment rather than using the rejected pair reconstruction.
  - If non-MATMUL dominates, pivot away from kernel changes.
- Falsifier: profiled run falls outside the local orthogonal-only throughput band or symbols are unresolved/noisy.
- Cheapest decisive test: one Pixel simpleperf run with `CACTUS_ORTHO_F32_AR_DIAG=1` and Gemma 512 decode 100.
- Evidence:
  - `experiments/matrix/results/local_ortho_residual_profile_20260520_2215/`: profiled decode 5.76 tok/s, prefill 29.26 tok/s, peak RAM 2451.02 MB, thermal status 0, 137,748 samples, 0 lost.
  - Flat self time: current interleaved generic GEMV 40.34%, orthogonal parallel task 34.37%, generic `cactus_quant_matmul` task 7.98%, attention kernels 6.22%.
- Decision: supported. The working local orthogonal diagnostic leaves decode dominated by current interleaved generic GEMV plus residual orthogonal work. Non-MATMUL is not the dominant Pixel issue. The next generic experiment must target the current interleaved path directly, not the rejected pair reconstruction.

### H46: current interleaved generic GEMV is limited by backend execution/latency, but simple split-panels is rejected

- Status: `supported_split_panels_rejected`
- Observed gap/target: local orthogonal-only decode is 5.76-5.93 tok/s, still far from 8-9 tok/s. H45 shows 40.34% self time in current interleaved generic GEMV.
- Mechanisms separated: SDOT issue limits vs load/use latency from stack-expanded panels vs frontend/compiler scheduling vs L1/cache behavior vs missing MMLA instruction selection.
- Chosen path/share: profile and counter-test the current interleaved generic path before adding another kernel rewrite.
- Predicted signatures:
  - Load/use or cache bottleneck: backend memory/L1 stall counters are high, source/panel load PCs dominate, and prefetch/scheduling diagnostics should move probe time before real-model promotion.
  - SDOT issue bottleneck: retired instruction mix and low memory stalls point at dot throughput; an MMLA probe must show a large hot-shape reduction and contain `smmla`, `ummla`, or `usmmla`.
  - Frontend/compiler scheduling bottleneck: high frontend stalls or PC concentration around macro-expanded load/dot sequences without corresponding cache misses.
- Falsifier: counters are noisy/unavailable, generic probe does not match H45 hot path, or real-model Amdahl movement is below the measured share.
- Cheapest decisive test: run simpleperf stat/record on the current interleaved generic hot path and compare against a targeted diagnostic only if the counters predict a clear signature.
- Evidence:
  - `experiments/matrix/results/h46_current_generic_stat_20260520_2240/stat_user.stdout.txt`: orthogonal-enabled decode 6.02 tok/s, 82.12B cycles, 271.43B instructions, IPC 3.31, backend stalls 27.08B cycles, frontend stalls 0.57B cycles, L1D load miss rate 1.48%, dTLB load miss rate 0.16%.
  - `experiments/matrix/results/h46_current_generic_stat_20260520_2240/stall_user.stdout.txt`: backend stalls about 33.1% of cycles, CPU-bound backend about 21.8%, memory-bound backend about 11.2%, L1D about 1.7%, TLB about 0.9%, rename about 4.7%, frontend about 0.7%.
  - Old deployed `cq4_kernel_probe` sidecar for `K=1536,N=12288`: split-panels repeated about 0.697-0.700 ms vs streaming about 0.800 ms at 200 iterations, but this probe is stale relative to current source.
  - `experiments/matrix/results/h46_split_panels_model_diag_20260520_2240/`: orthogonal-only decode 6.04 tok/s; orthogonal plus `CACTUS_GENERIC_SPLIT_PANELS_DIAG=1` decode 5.75 tok/s.
- Decision: supported as a mechanism narrowing result, with the split-panels implementation rejected. The counters argue against mmap/page-fault or frontend primary mechanisms and point toward backend execution/latency with both CPU-bound and some memory-bound pressure. The old probe is not sufficient for promotion; real-model movement remains the gate.

### H47: LiteRT-LM speed comes from a different artifact/runtime contract, not a small kernel tweak

- Status: `supported_initial`
- Observed gap/target: Cactus local-source orthogonal-only Gemma 512 decode is about 5.9-6.0 tok/s; LiteRT-LM Gemma 512 decode on the same Pixel CPU7 strict setting is 9.88 tok/s.
- Mechanisms separated: LiteRT-LM artifact/runtime contract vs Cactus current CQ4 model format and hand-written M=1 kernels.
- Evidence:
  - `experiments/matrix/results/matrix_pixel_10a_litert_lm_gemma_4_e2b_512.csv`: Pixel strict CPU7 one-thread LiteRT-LM Gemma 512 prefill 46.23 tok/s, decode 9.88 tok/s, peak RAM 3382.18 MB.
  - LiteRT-LM runtime uses a `.litertlm` end-to-end artifact with `int4_per_output_channel`, paired prefill/decode signatures, and a LiteRT `CompiledModel` execution path.
  - CPU backend enables XNNPACK threads, XNNPACK weight cache, latest XNNPACK operators, and compressed quantization zero points in `third_party/litert-lm/runtime/executor/llm_litert_compiled_model_executor.cc`.
  - CPU KV cache uses a single-buffer input/output path to improve performance and memory usage.
- Decision: supported initial conclusion. LiteRT-LM is the best benchmark target for what Pixel CPU should be able to do, but its performance likely comes from the combination of converted artifact layout, XNNPACK/LiteRT compilation, cached packed weights, and runtime KV/signature handling. We should compare artifacts and execution contracts before assuming one missing Cactus kernel trick can close the gap.

### H48: persistent XNNPACK cache is not the main LiteRT-LM steady-decode advantage

- Status: `supported_cache_not_primary`
- Observed gap/target: Cactus local-source orthogonal-only Gemma 512 decode is about 5.9-6.0 tok/s; LiteRT-LM strict Pixel CPU7 reaches 9.88-10.04 tok/s.
- Mechanisms separated: persistent XNNPACK weight-cache reuse vs XNNPACK delegated operator implementation and packed steady-state execution vs LiteRT-LM graph/signature/KV contract.
- Chosen path/share: run the same Pixel LiteRT-LM Gemma 512/100 benchmark with the staged cache and with `--disable_cache=true`.
- Predicted signatures:
  - If the persistent cache file is the main steady-decode mechanism, disabling it should cause a large decode regression, not only init/RAM movement.
  - If the cache mainly avoids compile/packing cost, init and RAM should worsen while decode stays near the cached result.
  - If XNNPACK compiled operators are the real steady-state mechanism, both runs should remain far above Cactus decode as long as delegation still happens.
- Falsifier: disabled-cache decode falls near Cactus throughput or the runner stops delegating the decode/prefill graph to XNNPACK.
- Evidence:
  - `experiments/matrix/results/h48_litert_cache_decode_20260521/`: cache-enabled direct Pixel run decode 10.04 tok/s, prefill 47.56 tok/s, executor init 580.90 ms, peak system RAM 3382 MB.
  - Same command with `--disable_cache=true`: decode 9.55 tok/s, prefill 41.85 tok/s, executor init 9969.63 ms, peak system RAM 4794.6 MB, peak private footprint 5651.08 MB.
  - Disabled-cache stderr explicitly reports `Can't use cache: INVALID_ARGUMENT: Cache is explicitly disabled.`
  - LiteRT-LM artifact inspection: `.litertlm` has 12 sections; the `tf_lite_prefill_decode` TFLite section is 818,275,184 bytes with signatures `decode`, `prefill_1024`, `prefill_128`, and `verify`.
  - The decode subgraph has 2068 ops, including 277 `FULLY_CONNECTED`, 276 `DEQUANTIZE`, 211 `QUANTIZE`, and 327 `STABLEHLO_COMPOSITE`; its tensors include `INT4`, `INT8`, and `FLOAT32`. XNNPACK delegates most of the graph in the benchmark logs.
- Decision: persistent cache reuse is a large init/RAM factor but not the primary steady-state decode explanation. The useful Cactus comparison is now the compiled XNNPACK int4/per-output-channel `FULLY_CONNECTED` path and LiteRT graph/KV contract, not mmap/cache-file behavior alone.

### H49: LiteRT-LM uses delegated quantized FC where Cactus uses CQ4 interleaved codebook GEMV

- Status: `supported_next_probe_candidate`
- Observed gap/target: LiteRT-LM cache-disabled still decodes at 9.55 tok/s, while Cactus local-source orthogonal-only is about 5.9-6.0 tok/s.
- Mechanisms separated: Cactus CQ4 codebook/TBL decode plus per-group scaling vs LiteRT/XNNPACK int4/int8 `FULLY_CONNECTED` delegated execution vs non-MATMUL KV/signature overhead.
- Chosen path/share: inspect Gemma decode IR and weight headers, then compare against the LiteRT `decode` TFLite subgraph.
- Predicted signatures:
  - If the main gap is Cactus CQ4 codebook overhead/layout, the Cactus hot linear weights should be CQ4 interleaved and the LiteRT decode graph should use quantized FC with static per-output-channel quantization.
  - If the main gap is outside linear kernels, the artifact comparison should show similar linear contracts or the profile should be dominated by non-linear work.
  - If a Cactus probe mimicking per-output-channel int4/int8 FC does not beat current CQ4 GEMV by a large hot-shape margin, this hypothesis should not be promoted.
- Falsifier: TFLite decode is not mostly delegated quantized FC, or Cactus hot linear weights are already equivalent to LiteRT's FC contract, or a focused probe has no material Pixel movement.
- Evidence:
  - Cactus `optimized_ir_decoder_step.json` has 276 `linear` nodes. Hot language projections are CQ4 interleaved group-size 128, including q/o/gate/up/down and per-layer gate/projection files.
  - Cactus key shapes include 40 `(12288,1536)`, 30 `(6144,1536)`, 28 `(2048,1536)`, 28 `(1536,2048)`, 20 `(1536,12288)`, 15 `(1536,6144)`, plus smaller attention/per-layer projections.
  - LiteRT `decode` subgraph has 277 `FULLY_CONNECTED` ops. Type triplets are 145 `INT8 x INT4 -> INT8`, 71 `INT8 x INT8 -> INT8`, 60 `INT8 x type19 -> INT8`, and 1 `FLOAT32 x type19 -> FLOAT32`.
  - LiteRT FC weights show per-output-channel quant metadata: examples have scale/zero-point counts matching output rows, such as `(2048,1536)` with 2048 scales and `(12288,1536)` with 12288 scales.
- Decision: next Cactus diagnostic should not be another mmap/cache tweak. The most direct falsifiable probe is a current-source hot-shape kernel/probe that removes the CQ codebook/TBL path and approximates LiteRT's per-output-channel int4/int8 FC execution, with Amdahl prediction against the 40% current generic GEMV share.

### H50: per-output-channel int4/int8 FC hot-shape probe can explain the generic GEMV residual

- Status: `rejected_as_primary_explanation`
- Observed gap/target: Cactus valid local orthogonal diagnostic is about 5.9-6.0 tok/s at Gemma 512 decode; LiteRT-LM CPU is 9.55-10.04 tok/s. Reaching 8 tok/s from 6 tok/s requires about 25% total time reduction.
- Mechanisms separated: Cactus CQ4 codebook/TBL plus per-group norm accumulation vs LiteRT-like per-output-channel int4/int8 dot path vs shape-specific memory locality.
- Chosen path/share: H45 gives current interleaved generic GEMV 40.34% self time after orthogonal improvement. A generic-only fix needs roughly 60% reduction of that bucket to be promotable toward 8 tok/s.
- Predicted signatures:
  - A LiteRT-like per-output-channel int4/int8 probe should beat current `packed_tbl_dot_norm` by a large margin on Gemma hot shapes such as `K=1536,N=12288`.
  - The win should hold on at least one reverse/down-projection shape, not only the large FFN up/gate shape.
  - If the result is strong enough, a real-model hook must still move Gemma decode by the Amdahl-predicted amount before promotion.
- Falsifier: the LiteRT-like probe is within roughly 5% of the CQ4 path, slower, or only helps a non-dominant shape.
- Cheapest decisive test: extend `tests/android/cq4_kernel_probe.cpp` with a standalone per-output-channel int4/int8 benchmark and run it on Pixel CPU7.
- Evidence:
  - Probe target added in `tests/android/cq4_kernel_probe.cpp` and built as Android target `cq4_kernel_probe`.
  - Pixel CPU7 focused run stored in `experiments/matrix/results/h50_litert_like_i4pc_probe_20260521/`.
  - 200-iteration first pass:
    - `gemma_ffn_up` current `packed_tbl_dot_norm` 0.415609 ms vs LiteRT-like `litert_i4pc_dot4_f32` 0.422017 ms; LiteRT-like was 1.5% slower.
    - `gemma_ffn_down` current 0.402045 ms vs LiteRT-like 0.387985 ms; LiteRT-like was 3.5% faster.
    - `gemma_attn_q` current 0.135290 ms vs LiteRT-like 0.123773 ms; LiteRT-like was 8.5% faster.
  - 1000-iteration repeat:
    - `gemma_ffn_up` current 0.401211 ms vs LiteRT-like 0.435359 ms; LiteRT-like was 8.5% slower.
    - `gemma_ffn_down` current 0.406898 ms vs LiteRT-like 0.384061 ms; LiteRT-like was 5.6% faster.
    - `gemma_attn_q` current 0.099081 ms vs LiteRT-like 0.093238 ms; LiteRT-like was 5.9% faster.
- Decision: a simple per-output-channel int4/int8 rowwise FC contract does not provide the large hot-shape win required by the Amdahl target. This rejects "Cactus is slow mainly because CQ4 has per-group norms/codebook lookup instead of per-output-channel int4" as a standalone explanation. It does not reject a more specific XNNPACK/LiteRT execution advantage such as different tiling, code generation, scheduling, or graph-level fusion.

### H51: real-model per-shape timing and LiteRT symbol/counter comparison

- Status: `supported_kernel_mechanism_not_full_solution`
- Observed gap/target: Cactus is still about 5.9-6.0 tok/s with the valid orthogonal diagnostic, while LiteRT-LM remains about 9.55-10.04 tok/s on CPU7. H50 shows a same-MAC standalone rowwise int4 contract is not enough to explain this.
- Mechanisms separated: real-model shape mix and dispatch overhead vs standalone microkernel timing vs LiteRT/XNNPACK compiled microkernel selection and scheduling.
- Chosen path/share: first add or use low-overhead Cactus decode diagnostics that aggregate elapsed time by linear shape/path for actual Gemma 512 decode, then profile LiteRT decode symbols/counters enough to identify whether XNNPACK is using materially different int4 microkernels or spending less time outside FC.
- Predicted signatures:
  - If the probe is misleading because the real model spends time in a small set of shape/path combinations, per-shape decode timing will show a narrower dominant bucket than op counts suggest.
  - If LiteRT's advantage is XNNPACK microkernel implementation, simpleperf symbols or disassembly should show a specific quantized FC/GEMV ukernel with better cycles/MAC or lower backend-stall signatures than Cactus's current generic GEMV.
  - If the gap is graph/runtime overhead, Cactus per-shape linear totals plus attention/orthogonal totals will leave a large residual outside the probed GEMV work, and LiteRT profile should show less non-FC overhead.
- Falsifier: real-model per-shape timing closely matches the standalone probe proportions and leaves no unexplained residual, while LiteRT profiles show similar FC cycles/MAC and similar non-FC overhead.
- Cheapest decisive test: add one env-gated real-model timing aggregation for decode linear calls by `(K,N,weight_format,path)` and collect one strict Gemma 512 CPU7 run; separately run a short LiteRT-LM CPU7 simpleperf profile to identify delegated FC symbols and top buckets.
- Evidence:
  - Added `tests/android/xnnpack_qc4_probe.cpp`, which directly calls XNNPACK's Android `xnn_create_fully_connected_nc_qd8_f32_qc4w` operator on Gemma hot shapes. This uses XNNPACK's own operator packing and selected run ukernel rather than the H50 hand-written approximation.
  - Built in a separate API-26 Android build directory because the existing ExecuTorch/XNNPACK/KleidiAI static archives were built for Android API 26 and do not link cleanly in the repo's normal API-21 Android build.
  - Results stored in `experiments/matrix/results/h51_xnnpack_qc4_probe_20260521/`.
  - Same-session 1000-iteration Pixel CPU7 comparison:
    - `gemma_ffn_up` (`K=1536,N=12288`): Cactus `packed_tbl_dot_norm` 0.400904 ms; XNNPACK `qd8_f32_qc4w` 0.261327 ms. XNNPACK is 1.53x faster.
    - `gemma_ffn_down` (`K=12288,N=1536`): Cactus 0.428915 ms; XNNPACK 0.260334 ms. XNNPACK is 1.65x faster.
    - `gemma_attn_q` (`K=1536,N=2048`): Cactus 0.099505 ms; XNNPACK 0.046913 ms. XNNPACK is 2.12x faster.
  - XNNPACK FFN-up simpleperf profile: `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm` accounts for 91.62% self time; `xnn_pack_qs8_qc4w_gemm_goi_w` is 1.79%. This confirms the relevant Pixel path is an I8MM QC4 GEMM ukernel, not a generic C++ rowwise loop.
- Decision: user's objection is supported. H50's hand-written per-output-channel int4 loop was not a valid proxy for LiteRT/XNNPACK kernel quality. XNNPACK's real QC4 operator is materially faster than Cactus's current CQ4 GEMV on hot shapes, especially attention-size projections. Coverage is now the key question. Applying a 1.5-1.65x speedup only to the 40.34% current interleaved generic GEMV bucket predicts roughly 7 tok/s from a 6 tok/s base. Applying the same class of speedup to the broader roughly 80% matmul-like profile share predicts roughly 8.2-8.8 tok/s, which is the actual target range. The next experiment should test whether XNNPACK/KleidiAI-style kernels can cover both generic interleaved CQ4 and the orthogonal/output-head bucket in real Cactus decode.

### H52: adapt XNNPACK-style I8MM QC4 microkernel contract into Cactus generic decode

- Status: `theoretically_supports_8_9_tps_if_coverage_transfers`
- Observed gap/target: XNNPACK direct QC4 operator is 1.53-1.65x faster than Cactus current CQ4 on FFN hot shapes and 2.12x faster on the probed attention shape. If this can cover the broader matmul-like profile share, the math can plausibly reach 8-9 tok/s.
- Mechanisms separated: XNNPACK I8MM microkernel and packed-weight layout vs Cactus CQ4 codebook semantics/weight format vs orthogonal/output-head layout vs real-model conversion cost and memory overhead.
- Chosen path/share: use H51's identified ukernel family, `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm`, as the reference for a diagnostic Cactus-side path. Start by proving coverage: generic interleaved CQ4 is the first target, then orthogonal/output-head if it can be expressed as an XNNPACK-compatible QC4 operator.
- Predicted signatures:
  - If the kernel path transfers only to generic interleaved CQ4, an env-gated Cactus run should improve real Gemma 512 decode by roughly 15-20% from the current orthogonal-enabled baseline.
  - If the same class of win also transfers to the orthogonal/output-head bucket, end-to-end decode should move into the 8-9 tok/s range.
  - If CQ4 codebook semantics or conversion overhead prevent transfer, the real-model win will be far below the standalone operator speedup or memory/init cost will become unacceptable.
  - If remaining non-generic work is the blocker, Cactus will improve toward about 7 tok/s and still miss 8-9, matching the Amdahl bound.
- Falsifier: direct XNNPACK/QC4 routing inside Cactus covers hot generic shapes but improves decode by less than about half the Amdahl-predicted amount, or cannot represent the relevant CQ4 weights without a large semantic/layout conversion that dominates memory or init.
- Cheapest decisive test: implement an env-gated diagnostic cache for a small selected set of generic Gemma CQ4 interleaved weights, materialize equivalent XNNPACK QC4 operator weights once, and route M=1 decode through `xnn_run_operator`; measure strict Gemma 512 CPU7 decode plus RAM/init deltas.
- Evidence:
  - Shape-suite results: `experiments/matrix/results/h52_xnnpack_shape_suite_20260521/shape_suite_cpu7_clean_stdout.txt`.
  - Same-session Pixel CPU7 toy comparison, Cactus current `packed_tbl_dot_norm` vs XNNPACK `xnn_create_fully_connected_nc_qd8_f32_qc4w`:
    - `gemma_ffn_up` (`K=1536,N=12288`, count 40): 0.404224 ms vs 0.264781 ms, 1.53x.
    - `gemma_ffn_down` (`K=12288,N=1536`, count 20): 0.405988 ms vs 0.256284 ms, 1.58x.
    - `gemma_ffn_mid_up` (`K=1536,N=6144`, count 30): 0.225826 ms vs 0.127623 ms, 1.77x.
    - `gemma_ffn_mid_down` (`K=6144,N=1536`, count 15): 0.221350 ms vs 0.127058 ms, 1.74x.
    - `gemma_attn_q` (`K=1536,N=2048`, count 28): 0.099200 ms vs 0.046165 ms, 2.15x.
    - `gemma_attn_in` (`K=2048,N=1536`, count 28): 0.097871 ms vs 0.044529 ms, 2.20x.
  - Weighted by those Gemma decode shape counts, Cactus toy total is 39.901738 ms and XNNPACK toy total is 23.990912 ms, for a 1.663x weighted speedup across the major non-output-head linear shapes.
  - Output-head-sized toy:
    - XNNPACK `gemma_output_head` (`K=1536,N=262144`): 10.828792 ms.
    - Current orthogonal probe `orthogonal_dot_full_ar32` (`K=1536,N=262144`): 61.980803 ms.
    - Kernel-level output-head-sized speedup is 5.72x, but this is not yet a semantic drop-in because the current path includes orthogonal/CQ codebook behavior.
  - Amdahl estimates from a 6.0 tok/s orthogonal-enabled baseline:
    - If only the 40.34% generic GEMV bucket gets the 1.663x weighted speedup: about 7.15 tok/s.
    - If the broader roughly 80% matmul-like share gets the 1.663x weighted speedup: about 8.81 tok/s.
    - If generic and other quant matmul get 1.663x, output-head/orthogonal gets the 5.72x toy speedup, attention stays unchanged, and the residual stays unchanged: about 11.46 tok/s. This is an upper-bound plausibility calculation, not a prediction.
- Decision: a 9 tok/s target is theoretically defensible from the toy matrix evidence, but only if XNNPACK-style kernels cover more than the current generic CQ4 FFN bucket. Generic-only replacement likely lands near 7 tok/s. Broad linear coverage, especially the output-head/orthogonal bucket, is the difference between "nice speedup" and "competitive with LiteRT".

### H53: directly mapping real Cactus CQ4 weights to XNNPACK QC4 is not the right product path

- Status: `diagnostic_reframes_direction`
- Observed gap/target: H52 shows XNNPACK QC4 kernels are fast enough to make 8-9 tok/s plausible only with broad decode linear coverage. The next question was whether Cactus can directly route real CQ4 weights into that exact XNNPACK per-output-channel QC4 contract.
- Mechanisms separated: Cactus CQ4 codebook plus per-group norms vs XNNPACK signed-int4 weights plus one scale per output row vs orthogonal output-head folding/materialization cost.
- Chosen path/share: inspect real Gemma weights and sample-reconstruct one FFN-up, one FFN-down, one attention projection, and the tied output-head/token-embedding matrix. Requantize sampled reconstructed rows to signed int4 with one scale per row to measure the semantic cost of that contract.
- Predicted signatures:
  - If generic Cactus CQ4 can map directly, real FFN/attention weights should have one scale/norm group per output row or negligible error when collapsed to one per-row scale.
  - If output-head orthogonal CQ4 can map structurally, it should have one group per row after folding the orthogonal transform, with memory/init cost becoming the main risk.
  - If direct mapping is the wrong product path, generic weights will have multiple norm groups per row and require either semantic re-quantization or a Cactus-native kernel that preserves CQ4 semantics.
- Falsifier: real generic weights already use a single norm group per row or sampled one-scale signed-int4 reconstruction has trivial error across the hot families.
- Cheapest decisive test: `experiments/matrix/results/h53_real_weight_xnnpack_mapping_20260521/mapping_stdout.txt`.
- Evidence:
  - `layer_0_ffn_up.weights`: CQ4 interleaved, `N=6144,K=1536`, group size 128, 12 norm groups per output row. One-scale signed-int4 reconstruction on sampled rows has mean relative L2 0.1705 and p95 0.2387.
  - `layer_0_ffn_down.weights`: CQ4 interleaved, `N=1536,K=6144`, group size 128, 48 norm groups per output row. One-scale signed-int4 reconstruction mean relative L2 0.1999 and p95 0.2960.
  - `layer_0_attn_q.weights`: CQ4 interleaved, `N=2048,K=1536`, group size 128, 12 norm groups per output row. One-scale signed-int4 reconstruction mean relative L2 0.1601 and p95 0.1953.
  - `token_embeddings.weights` / LM head: CQ4 orthogonal, `N=262144,K=1536`, group size 1536, one norm group per row. It is structurally compatible with a folded per-row-scale representation, but requires materializing a 192 MiB packed-i4 cache plus scales; naive dense materialization would be 1536 MiB.
  - Codebook affine check: `experiments/matrix/results/h53_real_weight_xnnpack_mapping_20260521/codebook_affine_stdout.txt`. The real generic CQ4 codebook is not exactly signed-int4-linear; affine fit relative L2 is 0.1101 with max absolute codebook error 15.62 on the int8-quantized generic table.
- Decision: direct use of XNNPACK's exact QC4 weight contract is not a clean product path for generic Cactus CQ4 because it would collapse 12-48 per-row groups into one scale and introduce measurable semantic drift. The better direction is to learn from XNNPACK's fast Pixel QC4 kernel design and adapt the tiling, packing, and I8MM schedule to Cactus CQ4 while preserving the existing codebook/per-group semantics. The orthogonal LM head remains a separate high-leverage candidate for a folded packed cache, but it should be treated as its own experiment.

## Next Experiment Record

- Observed gap/target: current working local orthogonal diagnostic leaves Pixel Gemma 512 decode around 5.9-6.0 tok/s, while LiteRT-LM reaches about 9.6-10.0 tok/s on the same CPU7 setting. H51/H52 show XNNPACK's Pixel QC4 path is 1.5-2.2x faster on hot shapes, but H53 shows direct generic weight-contract mapping is not semantically clean.
- Mechanisms separated: XNNPACK's I8MM ukernel/packing/scheduling advantage vs Cactus CQ4 codebook and per-group norms vs output-head orthogonal folding vs graph/runtime overhead.
- Chosen path/share: compare XNNPACK's `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm` and pack format against Cactus's current interleaved CQ4 GEMV, then build the smallest Cactus-native diagnostic that borrows the XNNPACK schedule without collapsing CQ4 semantics.
- Predicted signatures:
  - The diagnostic kernel should emit `smmla`/`ummla`/`usmmla` or equivalent I8MM instructions on Pixel, not only SDOT.
  - Hot-shape probes should approach the XNNPACK speed class before any real-model promotion.
  - Real Gemma 512 decode should move by at least half of the Amdahl-predicted improvement for the covered share; otherwise the mechanism is incomplete.
  - Output equivalence should be checked against the current CQ4 path before performance is credited.
- Falsifier: preserving Cactus CQ4 codebook/per-group semantics prevents an I8MM-style packed kernel from beating the current SDOT/TBL path on hot shapes, or the probe wins but the real-model decode movement is far below the profile-share prediction.
- Cheapest decisive test: source-level comparison of XNNPACK pack/ukernel vs Cactus CQ4 hot kernel, followed by a standalone Cactus CQ4 probe variant with XNNPACK-like output tiling and I8MM-friendly packed blocks for one actual Gemma FFN/attention shape.

## Baseline Refreshes

- `experiments/matrix/results/current_nochange_pixel_tier12_20260521_0114/`: rebuilt and redeployed `cactus_llm_bench`, then ran the strict Gemma 512 no-env CPU7 command twice. Tier 1 decode 5.62 tok/s, Tier 2 decode 5.56 tok/s, prefill 27.31/27.97 tok/s, peak RAM 2456.70/2456.75 MB, thermal status 0 before and after. This is device-state calibration only, not improvement evidence.

### H54: XNNPACK's Pixel win comes from I8MM tile scheduling after weights are already expanded/packed

- Status: `rejected_as_broad_generic_path`
- Observed gap/target: latest strict Pixel Gemma 512 no-env baseline is 5.62/5.56 tok/s; LiteRT-LM is about 9.6-10.0 tok/s. H51/H52 show XNNPACK QC4 hot-shape kernels are 1.5-2.2x faster, but H53 shows direct one-scale QC4 mapping is not semantically clean for generic Cactus CQ4.
- Mechanisms separated: XNNPACK-style 16-output I8MM scheduling vs Cactus SDOT plus TBL codebook expansion vs unavoidable cost of preserving arbitrary CQ4 codebook values and per-group norms.
- Chosen path/share: generic interleaved CQ4 remains the first target because H45 shows it at 40.34% self time after the working orthogonal diagnostic. A generic-only path needs about a 60% reduction of that bucket to move 6 tok/s toward 8 tok/s.
- Predicted signatures:
  - A standalone probe that pre-expands CQ4 codebook values into int8 blocks and uses an XNNPACK-like I8MM 16-output tile should contain `smmla`/`ummla`/`usmmla`.
  - If I8MM scheduling is the main transferable mechanism, this codebook-preserving expanded-int8 probe should approach the XNNPACK QC4 speed class on Gemma hot shapes.
  - If XNNPACK's advantage depends on true signed-int4 nibble packing rather than I8MM scheduling alone, preserving arbitrary CQ4 codebook values as expanded int8 will be much closer to current Cactus SDOT time.
  - If the probe wins only by using a 2x larger expanded-weight cache, the result remains a diagnostic until a bounded-memory real-model route is designed.
- Falsifier: the I8MM expanded-int8 probe is within noise of current `packed_tbl_dot_norm`, slower, lacks MMLA instructions in the binary, or only wins on shapes too small to move Gemma decode by Amdahl.
- Cheapest decisive test: extend `tests/android/cq4_kernel_probe.cpp` with a pre-expanded arbitrary-int8 XNNPACK-like `vmmlaq_s32` tile, build/deploy only `cq4_kernel_probe`, run Pixel CPU7 hot shapes, and disassemble/grep the probe binary for MMLA instructions.
- Evidence:
  - Probe implementation: `tests/android/cq4_kernel_probe.cpp`, benchmark `expanded_i8mm_x16_dot_norm`.
  - Results: `experiments/matrix/results/h54_i8mm_expanded_cq4_probe_20260521/probe_cpu7_stdout.txt`.
  - Disassembly signature: `experiments/matrix/results/h54_i8mm_expanded_cq4_probe_20260521/disasm_dot_mmla.txt` contains `smmla`, so the test did exercise an I8MM path.
  - `gemma_ffn_up`: current `packed_tbl_dot_norm` 0.402044 ms; preexpanded SDOT 0.515062 ms; expanded I8MM 0.567226 ms; LiteRT-like rowwise 0.427881 ms. Expanded I8MM is 41.1% slower than current.
  - `gemma_ffn_down`: current 0.421505 ms; preexpanded SDOT 0.559135 ms; expanded I8MM 0.554969 ms; LiteRT-like rowwise 0.386888 ms. Expanded I8MM is 31.7% slower than current.
  - `gemma_attn_q`: current 0.083297 ms; expanded I8MM 0.071041 ms; repeat current 0.082414 ms vs expanded I8MM 0.070033 ms. Attention-size shape sees a repeated 14-15% win.
- Decision: rejected as the broad generic route. Simply preserving CQ4 codebook values by expanding them to int8 and then using an XNNPACK-like I8MM tile does not transfer the XNNPACK speed class to the dominant FFN shapes. The XNNPACK advantage appears tied to compact nibble-packed weights/per-output affine scale and reduced weight bandwidth, not just MMLA scheduling. The attention-size win is real but too small a profile slice to justify promotion alone.

### H55: affine-codebook signed-int4 replacement is semantically risky on real generic CQ4 weights

- Status: `rejected_as_unqualified_generic_replacement`
- Observed gap/target: H54 says expanded arbitrary-int8 I8MM is not viable for dominant FFN shapes, while H51/H52 show true XNNPACK nibble-packed QC4 is fast. The remaining generic path would need Cactus CQ4 codebook values to collapse into an affine signed-int4 lattice without destabilizing outputs.
- Mechanisms separated: codebook approximation error at dot/output level vs performance benefit from true nibble-packed MMLA vs layer/shape localization of semantic drift.
- Chosen path/share: test real generic Gemma CQ4 weights before writing a nibble-packed MMLA kernel. Sample rows from FFN-up, FFN-down, and attention-Q, compare exact CQ4 codebook dot products against best affine signed-int4 codebook dot products over random transformed activation vectors.
- Predicted signatures:
  - If affine signed-int4 is a plausible exact-kernel substitute, sampled dot/output relative error should be very small and stable across FFN/attention shapes.
  - If the codebook carries meaningful non-affine structure, dot/output relative error should be broad and repeat across shapes.
- Falsifier: low dot/output error across sampled FFN-up/down/attention, making a nibble-packed MMLA probe worth implementing.
- Cheapest decisive test: `experiments/matrix/results/h55_affine_codebook_dot_error_20260521/dot_error_stdout.txt`.
- Evidence:
  - `ffn_up`: affine codebook relative L2 0.1101; sampled dot relative L2 mean 0.1247, p95 0.1325, max 0.1369; cosine mean 0.99562.
  - `ffn_down`: sampled dot relative L2 mean 0.1252, p95 0.1341, max 0.1397; cosine mean 0.99567.
  - `attn_q`: sampled dot relative L2 mean 0.1254, p95 0.1320, max 0.1398; cosine mean 0.99555.
- Decision: rejected as an unqualified generic replacement. The error is not a vague quantization excuse; it is a measured, repeatable dot-level semantic drift around 12-14% relative L2 across three real weight families. A true XNNPACK-style nibble-packed generic kernel would require changing the effective weight function, so it cannot be promoted without a correctness/fine-tuning story. Generic decode work should either preserve the exact CQ4 codebook and accept that H54 did not produce the needed FFN speed, or pivot to the orthogonal/output-head folded-cache path and graph/runtime-level differences.

### H56: replicate or learn from XNNPACK QC4 to make exact Cactus CQ4 fast on Pixel

- Status: `tile_only_rejected_as_sufficient_mechanism`
- Observed gap/target: latest strict Pixel Gemma 512 no-env baseline is 5.62/5.56 tok/s; LiteRT-LM is about 9.6-10.0 tok/s. XNNPACK's real QC4 operator is already measured 1.53x faster on FFN-up, 1.65x faster on FFN-down, 2.12x faster on attention-Q, and 5.72x faster on an output-head-sized toy shape. The performance target remains 8-9 tok/s decode.
- Mechanisms separated: XNNPACK's actual QC4 packing order, nibble decode path, 1x16c8 I8MM microkernel schedule, precomputed row/zero-point compensation, prefetch/unroll structure, and operator setup overhead vs Cactus's current exact CQ4 codebook/TBL path with 12-48 per-row norm groups.
- Chosen path/share: use XNNPACK/KleidiAI source and disassembly as the reference implementation, but do not collapse Cactus weights into XNNPACK's one-scale-per-output-row contract unless a separate semantic gate passes. The immediate goal is a Cactus-native exact-CQ4 probe that copies the transferable mechanics: output tiling, packed K-block order, activation quantization contract, accumulation/reduction schedule, and compensation handling.
- Predicted signatures:
  - Source/disassembly comparison should identify a small number of concrete differences that plausibly explain XNNPACK's lower cycles, not just "it uses I8MM".
  - A copied XNNPACK-style probe that preserves exact Cactus CQ4 semantics should beat current `packed_tbl_dot_norm` on at least one FFN hot shape by enough to predict real-model movement; attention-only wins are not sufficient.
  - If the required XNNPACK speed depends on true affine signed-int4 nibbles and one scale per row, exact Cactus CQ4 will not approach the XNNPACK timing without changing the effective weight function.
  - Any probe win must include an output-equivalence check against current CQ4 before it is counted.
- Falsifier: after copying the reusable packing/tile/schedule mechanics, exact Cactus CQ4 still stays near current FFN timings, or the only fast path requires the H55-rejected affine-codebook semantic change.
- Cheapest decisive test: inspect XNNPACK generated source and packers under `tests/build/xnnpack-src`, compare them to `tests/android/cq4_kernel_probe.cpp` and `cactus-kernels/src/matmul.cpp`, then add one standalone `cq4_kernel_probe` benchmark variant that implements the most transferable XNNPACK mechanic with exact Cactus CQ4 outputs.
- Evidence:
  - XNNPACK source inspected: `tests/build/xnnpack-src/src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-neoni8mm.c`, `tests/build/xnnpack-src/src/reference/packing.cc`, and `tests/build/xnnpack-src/src/configs/gemm-config.c`.
  - Pixel path confirmed from source: ARM64 I8MM selects `xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm`, `nr=16`, `log2_kr=3`, `planes=2`. The kernel keeps weights compact, loads 16 output columns, shifts/masks nibbles into high-nibble signed int4 lanes, uses `vmmlaq_s32`, then applies input scale and per-output filter scale.
  - Cactus exact-CQ4 cannot use that same nibble-to-I8MM path directly because the 4-bit value indexes an arbitrary codebook plus per-group norms. H54's expanded-int8 I8MM test preserved codebook values but doubled effective weight bytes and was slower on FFN.
  - Added `packed_tbl_x16_dot_norm` in `tests/android/cq4_kernel_probe.cpp` as the first exact-CQ4 source-copy diagnostic: it keeps compact Cactus nibbles/TBL/codebook/per-group norms and copies XNNPACK's 16-output tile/activation-reuse shape.
  - Results: `experiments/matrix/results/h56_xnnpack_exact_cq4_tile_probe_20260521/probe_cpu7_stdout.txt` and `repeat_pair_x16_cpu7_stdout.txt`.
  - Pixel CPU7 first run:
    - `gemma_ffn_up`: current `packed_tbl_dot_norm` 0.405678 ms; exact x16 tile 0.415352 ms, 2.4% slower.
    - `gemma_ffn_down`: current 0.425430 ms; exact x16 tile 0.382052 ms, 10.2% faster.
    - `gemma_attn_q`: current 0.083991 ms; exact x16 tile 0.076756 ms, 8.6% faster.
  - Repeat against the prior pair tile:
    - `gemma_ffn_up`: pair 0.456103 ms; x16 0.413877 ms.
    - `gemma_ffn_down`: pair 0.387157 ms; x16 0.381591 ms.
    - `gemma_attn_q`: pair 0.079529 ms; x16 0.076304 ms.
- Decision: copying XNNPACK's output tiling/activation reuse while preserving exact Cactus CQ4 semantics is not enough. It gives a useful small win on FFN-down and attention but misses the 1.5-2x XNNPACK speed class and regresses FFN-up. This narrows the mechanism: XNNPACK's Pixel advantage depends mainly on its affine signed-int4 nibble representation feeding I8MM at compact weight bandwidth, plus its row compensation/scaling contract. Exact arbitrary-codebook CQ4 cannot simply copy that path without either expanded-weight bandwidth loss or a semantic replacement.

### H57: affine-fast plus residual correction is not obviously cheap enough

- Status: `diagnostic_negative_for_generic_bridge`
- Observed gap/target: H56 says exact compact tiling does not reach XNNPACK speed, while H55 says uncorrected affine-codebook replacement causes about 12-14% dot-level drift. The remaining generic bridge would need an XNNPACK-fast affine component plus a cheap residual correction.
- Mechanisms separated: affine signed-int4 codebook component vs residual codebook entries vs correction coverage/cost.
- Chosen path/share: extend the H55 real-weight sampled dot diagnostic offline. Keep the same `index - 8` affine lattice as H55, then add back the largest residual codebook entries by absolute residual magnitude and measure dot error for FFN-up, FFN-down, and attention-Q.
- Predicted signatures:
  - If residual correction is plausible, a small number of residual codes should cover most residual energy and sharply reduce dot error.
  - If many residual codes are needed, correction likely costs too much and recreates the current TBL/codebook path.
- Falsifier: top-k residual correction with a small k, ideally 2-4 codes, drops sampled dot error close to the level needed for a model hook.
- Cheapest decisive test: `experiments/matrix/results/h57_affine_residual_bridge_20260521/residual_bridge_stdout.txt`.
- Evidence:
  - The fitted affine codebook matches H55: slope 14.8500517, intercept 7.42478171, codebook relative L2 0.110078799.
  - Residual energy is not cheaply localized enough. The two largest residual codes cover 53.1% residual energy but leave dot relative L2 around 11.7%; four codes cover 66.7% and leave about 10.1%; eight codes cover 88.4% and leave about 5.45%.
  - Twelve of sixteen residual codes are needed to reach about 1.3% dot relative L2 across FFN-up, FFN-down, and attention-Q.
  - Representative means:
    - `ffn_up`: top-0 0.125271, top-4 0.101343, top-8 0.054451, top-12 0.013254.
    - `ffn_down`: top-0 0.125478, top-4 0.101594, top-8 0.054905, top-12 0.013385.
    - `attn_q`: top-0 0.124494, top-4 0.101236, top-8 0.054521, top-12 0.013250.
- Decision: this does not look like a cheap generic bridge to XNNPACK speed. To preserve correctness, residual correction appears to need most of the codebook back; that likely forfeits the compact affine-I8MM advantage. Generic Cactus CQ4 still has no proven route to XNNPACK's kernel class without changing effective weights.

### H58: orthogonal LM-head folded cache can cover the output-head bucket

- Status: `partial_candidate_needs_share_and_correctness_gate`
- Observed gap/target: latest strict Pixel Gemma 512 no-env baseline is 5.62/5.56 tok/s, and the working orthogonal diagnostic leaves decode around 5.9-6.0 tok/s. XNNPACK output-head-sized toy timing is 10.828792 ms while the current orthogonal output-head-sized probe is 61.980803 ms, a 5.72x kernel-level gap. Target remains 8-9 tok/s decode.
- Mechanisms separated: exact orthogonal CQ4 fold to dense FP16/FP32 vs rowwise int8/int4 folded caches with measured output drift vs cache materialization/RAM cost vs real output-head Amdahl share.
- Chosen path/share: target only the tied LM-head/token-embedding orthogonal CQ4 matrix first. H53 showed it has one norm group per row and is structurally cleaner than generic interleaved CQ4; H57 makes further generic CQ4 kernel copying low-priority unless a new exact representation appears.
- Predicted signatures:
  - FP16 dense fold should have very small row/dot error but high memory cost.
  - Rowwise int8 fold may have acceptable error with lower memory than FP16; rowwise int4/XNNPACK-style fold should be fastest/smallest but may repeat H53 semantic drift.
  - Materialization time and peak cache size must be bounded enough to justify a real-model hook.
  - If the folded cache is viable, standalone output-head timing should be large enough to move decode by Amdahl, not just a toy speedup.
- Falsifier: folded candidates either require dense memory well outside the LiteRT RAM envelope, have unacceptable sampled dot/output error, or cannot plausibly reduce the output-head bucket enough after materialization overhead.
- Cheapest decisive test: offline real-weight diagnostic on `weights/gemma-4-e2b-it/token_embeddings.weights` that samples folded rows, measures FP16/int8/int4 row and dot error, estimates full cache bytes, and extrapolates materialization cost before any C++ hook.
- Evidence:
  - Offline diagnostic: `experiments/matrix/results/h58_lmhead_folded_cache_feasibility_20260521/folded_cache_stdout.txt`.
  - Real LM-head header: `N=262144,K=1536`, CQ4 orthogonal, one group per row, current payload 192.00 MiB plus 5.01 MiB metadata.
  - FP16 dense folded cache: 768.00 MiB additional data, 965.01 MiB if retained with original payload; sampled row relative L2 mean 0.000207 and sampled output relative L2 mean 0.000211.
  - Rowwise int8 folded cache: 384.00 MiB data plus 1.00 MiB scales, 582.01 MiB if retained with original payload; sampled row relative L2 mean 0.00937 and sampled output relative L2 mean 0.00985.
  - Rowwise int4 folded cache: 192.00 MiB data plus 1.00 MiB scales, 390.01 MiB if retained with original payload; sampled row relative L2 mean 0.16963 and sampled output relative L2 mean 0.17847.
  - Pixel timing proxy: `experiments/matrix/results/h58_lmhead_folded_cache_feasibility_20260521/xnnpack_qc8_output_head_cpu7_stdout.txt`. The normal Android build hit the known XNNPACK/KleidiAI static-link issue, so the existing API-26 `android/build_xnnpack_probe` side build was used.
  - Same Pixel CPU7 output-head shape, 100 iterations: XNNPACK QC4 11.824582 ms; XNNPACK QC8 14.470288 ms. Against the current orthogonal output-head-sized probe 61.980803 ms, QC8 is a 4.28x timing proxy.
  - Amdahl from the 5.56 tok/s no-env baseline using 4.28x on only the output/orthogonal bucket predicts about 7.22 tok/s at 30% share, 7.55 tok/s at the H45 34.37% orthogonal self-time share, 8.02 tok/s at 40% share, and 8.49 tok/s at 45% share.
- Decision: rowwise int4 is rejected for LM head because sampled output error is too high. FP16 is near-exact but memory-heavy and lacks a Pixel speed proxy. Rowwise int8 is the only plausible folded-cache candidate from this scout: it has about 1% sampled output drift, about 385 MiB cache size, and a Pixel timing proxy fast enough to matter. It is not yet a fix: output-head share must be measured precisely in real Cactus decode, and rowwise-int8 correctness needs a real-activation/logit gate before any model hook.

### H59: real Gemma decode output-head share and activations justify a rowwise-int8 LM-head hook

- Status: `diagnostic_supported_for_env_hook`
- Observed gap/target: H58 rowwise-int8 folded LM-head has a 4.28x Pixel XNNPACK timing proxy and about 1% random-vector sampled output drift, but it only reaches 8 tok/s if the real output-head bucket is about 40% or larger. Target remains 8-9 tok/s.
- Mechanisms separated: real decode output-head share vs folded-int8 candidate speed vs real-hidden-activation logit drift.
- Chosen path/share: add env-gated instrumentation only for the structural LM-head shape (`bits=4,K=1536,N=262144,group_size=1536,num_groups=1,orthogonal`) to time each call and capture hidden vectors for offline exact-vs-int8 replay.
- Predicted signatures:
  - If LM-head share is at least about 40% of decode, the H58 timing proxy can predict an 8+ tok/s path.
  - If real hidden activations have materially worse sampled logit drift than random vectors, rowwise-int8 folding is rejected as more than a speed toy.
  - If instrumentation perturbs throughput heavily, the timing share should not be trusted.
- Falsifier: decode LM-head share below Amdahl threshold, severe real-activation logit drift, or diagnostic throughput far outside current baseline.
- Cheapest decisive test: env-gated timing/sample run on strict Gemma 512 CPU7, then replay captured hidden vectors over sampled folded LM-head rows offline.
- Evidence:
  - Instrumentation code: `cactus-kernels/src/matmul.cpp`, env vars `CACTUS_LMHEAD_SHARE_DIAG`, `CACTUS_LMHEAD_SAMPLE_DIAG`, `CACTUS_LMHEAD_SAMPLE_COUNT`, and `CACTUS_LMHEAD_SAMPLE_PATH`.
  - Results: `experiments/matrix/results/h59_lmhead_share_real_activation_20260521/`.
  - First run with 8 samples: decode 5.58 tok/s, peak RAM 2456.97 MB, 170 LM-head calls, average 80.04 ms, total LM-head time 13.606 s of 35.750 s.
  - Repeat with 170 captured samples: decode 5.56 tok/s, peak RAM 2455.97 MB, 170 LM-head calls, average 80.226 ms, total LM-head time 13.638 s of 35.995 s.
  - Decode-specific last 100 calls: 8.016 s of 17.796 s decode time, so LM-head decode share is 45.04%.
  - Amdahl from 5.56 tok/s at 45.04% share: using H58 QC8-vs-current-toy 4.28x predicts 8.49 tok/s; using real measured current LM-head 80.226 ms vs XNNPACK QC8 14.470 ms predicts 8.81 tok/s.
  - Real hidden activation replay over 2048 sampled vocab rows:
    - Decode last 100 rowwise-int8 sampled output relative L2 mean 0.006338, p95 0.008001, max 0.008382, cosine mean 0.999979.
    - Decode last 100 rowwise-int4 sampled output relative L2 mean 0.112377, p95 0.140734, max 0.151297, cosine mean 0.993508.
- Decision: supported as a diagnostic path large enough for an env-gated real-model hook. Rowwise int8 folded LM-head now has enough share, timing, memory, and real-activation error evidence to test in real Cactus decode. It is still not a validated fix because it changes the effective LM-head weights, adds about 385 MiB cache, and needs actual real-model throughput/correctness measurement plus Samsung/second-shape guardrails before promotion.

### H60: XNNPACK-style 16-output tile packing helps exact CQ4 but is not enough

- Status: `supported_small_win_insufficient_for_target`
- Observed gap/target: Pixel strict Gemma 512 decode remains about 5.56 tok/s in the current no-env baseline, while LiteRT-LM/XNNPACK reaches about 9.55-10.04 tok/s. H59 shows an LM-head-only route could theoretically reach 8.5-8.8 tok/s, but it is a special-case rowwise-int8 semantic change. The broader question is still whether Cactus CQ4 can learn enough from XNNPACK CQ4 to close the Pixel gap without silently changing generic weights.
- Mechanisms separated: XNNPACK compact affine-int4 nibble packing, qd8 activation contract, I8MM 1x16c8 tile schedule, row/zero-point compensation, scaling placement, prefetch/unroll structure, and operator cache setup vs Cactus exact arbitrary-codebook CQ4 with per-group norms.
- Chosen path/share: copy one more XNNPACK mechanic into the exact Cactus CQ4 probe: 16-output tile-major packing with contiguous K-block bytes. This preserves the existing codebook/per-group norm math and only changes cache/pointer locality for the standalone probe.
- Predicted signatures:
  - If the missing mechanism is transferable while preserving CQ4 semantics, the next probe should improve dominant FFN shapes, not only attention, and should predict a measurable real-model Amdahl move.
  - If XNNPACK's speed is inseparable from affine signed-int4 nibbles and one scale per output row, exact Cactus CQ4 will stay below the XNNPACK speed class and the remaining options are semantic conversion with measured tolerance, structurally compatible special cases, or LiteRT-style artifact/runtime changes.
  - If direct XNNPACK integration is required to reproduce the numbers, the next blocker is the Android XNNPACK/KleidiAI link/operator-cache path rather than another handwritten microkernel.
- Falsifier: copied XNNPACK mechanics do not improve exact-CQ4 FFN timing materially, or the only fast path is the H55-rejected unqualified affine-codebook replacement with 12-14% dot-level drift.
- Cheapest decisive test: add `packed_tbl_x16_tilepack_dot_norm` in `tests/android/cq4_kernel_probe.cpp`, rebuild/deploy only `cq4_kernel_probe`, rerun Pixel CPU7 hot-shape probes, and compute the Amdahl implication before touching the full model.
- Evidence:
  - Results: `experiments/matrix/results/h60_xnnpack_tilepack_exact_cq4_20260521/`.
  - First Pixel CPU7 run, 1000 iterations:
    - `gemma_ffn_up`: current 0.406939 ms; prior x16 0.417663 ms; tile-packed x16 0.369040 ms.
    - `gemma_ffn_down`: current 0.406820 ms; prior x16 0.385316 ms; tile-packed x16 0.372796 ms.
    - `gemma_attn_q`: current 0.082400 ms; prior x16 0.076821 ms; tile-packed x16 0.076673 ms.
  - Repeat plus extra major-shape run:
    - `gemma_ffn_up`: 0.407334 -> 0.365638, 1.114x.
    - `gemma_ffn_down`: 0.406018 -> 0.368956, 1.100x.
    - `gemma_ffn_mid_up`: 0.213866 -> 0.189614, 1.128x.
    - `gemma_ffn_mid_down`: 0.214522 -> 0.197232, 1.088x.
    - `gemma_attn_q`: 0.083267 -> 0.077303, 1.077x.
    - `gemma_attn_in`: 0.084304 -> 0.077209, 1.092x.
  - Weighted by the known major Gemma decode linear counts, current total is 38.723 ms and tile-packed exact CQ4 total is 35.159 ms, a 1.101x weighted kernel speedup and 9.2% reduction in that bucket.
  - If applied to only H45's 40.34% current interleaved generic GEMV bucket, Amdahl predicts about 5.77 tok/s from a 5.56 tok/s baseline.
- Decision: XNNPACK-style tile-major packing is real and should be kept as a candidate implementation detail, but it is not the missing XNNPACK mechanism. Exact arbitrary-codebook CQ4 still does not approach the 1.5-2x XNNPACK QC4 speed class. The remaining gap is most likely the compact affine signed-int4/I8MM contract plus compensation/scaling, or a higher-level LiteRT artifact/runtime contract, not only packed-order locality.

### H61: normal Cactus Android build must prove direct XNNPACK operator integration is viable

- Status: `blocked_on_android_api_link_compat`
- Observed gap/target: H60 leaves exact generic CQ4 far below XNNPACK's speed class. H51/H52 prove XNNPACK QC4/QC8 operators are fast enough in a side build, but a real Cactus diagnostic needs the operator path to build and run in the normal Android environment before a model hook is justified.
- Mechanisms separated: XNNPACK operator speed vs Cactus Android link/API compatibility vs runtime operator setup/cache overhead.
- Chosen path/share: test the existing `xnnpack_qc4_probe` target in the normal `android/build` tree. This is a build/integration gate, not an optimization, and it has no model-time share until it links and runs.
- Predicted signatures:
  - Normal build links: run Pixel CPU7 and compare timings to H51/H58 side-build results.
  - Normal build fails with API/static archive symbols: direct XNNPACK integration is blocked on dependency/build compatibility before any Cactus hook.
  - Normal build links but timings differ materially: profile selected symbols and flags before trusting XNNPACK as an embedded path.
- Falsifier: the normal build cannot produce a working XNNPACK probe, or the working probe loses the XNNPACK speed class.
- Cheapest decisive test: required Cactus build followed by `cmake --build android/build --target xnnpack_qc4_probe`.
- Evidence:
  - Build log: `experiments/matrix/results/h61_normal_xnnpack_build_gate_20260521_build.log`.
  - Normal build cache has `ANDROID_PLATFORM=android-21`; the working side probe build has `ANDROID_PLATFORM=android-26`.
  - Normal build fails linking `xnnpack_qc4_probe` because `libkleidiai.a` references `stdout` and `stderr`.
- Decision: direct XNNPACK operator integration is not currently viable in the normal Cactus Android build. The side-build XNNPACK timings remain useful as speed evidence, but model-level integration requires either an API/dependency fix or an API-26 diagnostic runner.

### H62: qd8 activation quantization does not reject the folded LM-head diagnostic

- Status: `diagnostic_supported_for_real_model_hook`
- Observed gap/target: H59 predicts a path to 8.5-8.8 tok/s if the 45.04% LM-head decode bucket can run near the XNNPACK QC8 timing proxy. H59 measured rowwise-int8 folded-weight error, but XNNPACK QC8 also dynamically quantizes activations.
- Mechanisms separated: rowwise-int8 folded weight drift vs dynamic qd8 activation drift vs sampled-rank/logit stability.
- Chosen path/share: replay the H59 captured hidden vectors over 2048 sampled folded LM-head rows and compare exact folded logits against weight-only int8 and qd8-activation plus int8-weight approximations.
- Predicted signatures:
  - If qd8 activation dominates error, relative L2 should rise sharply and sampled top-rank agreement should degrade.
  - If qd8 is tolerable, error should remain low enough for a diagnostic hook, with sampled top-rank stable.
  - If rank stability only holds for weight-only int8, an XNNPACK QC8 hook needs a stronger correctness guard.
- Falsifier: qd8 activation causes broad sampled-rank changes or much larger logit error than H59.
- Cheapest decisive test: offline replay using `repeat_lmhead_hidden_samples_f16.bin` and `token_embeddings.weights`.
- Evidence:
  - Results: `experiments/matrix/results/h62_lmhead_qd8_activation_error_20260521/`.
  - Decode last100 relative L2:
    - weight-only rowwise int8 mean 0.006428, p95 0.008177.
    - symmetric qd8 plus rowwise int8 mean 0.023144, p95 0.035536.
    - affine qd8 plus rowwise int8 mean 0.019167, p95 0.029623.
  - Sampled-vocab rank over 2048 rows for decode last100: weight-only top1 match 100%; affine qd8 top1 match 100%; affine qd8 top-logit absolute error mean 0.059902, p95 0.159615.
- Decision: activation qd8 adds real but bounded drift in this diagnostic. It does not prove production correctness, but it clears the next gate for a scoped real-model LM-head hook or XNNPACK-backed API-26 diagnostic.

### H63: API-26 Cactus bench is usable as an XNNPACK diagnostic harness

- Status: `supported_diagnostic_harness`
- Observed gap/target: H61 blocks normal API-21 XNNPACK integration, but the side XNNPACK probe works at API-26. A real-model XNNPACK diagnostic needs an adjacent Cactus baseline in an API-26 runner.
- Mechanisms separated: API/build compatibility vs model-runner baseline drift vs XNNPACK hook feasibility.
- Chosen path/share: build and run `cactus_llm_bench` from `android/build_xnnpack_probe` and compare no-env strict Gemma 512 CPU7 to the normal current baseline.
- Evidence:
  - Results: `experiments/matrix/results/h63_api26_cactus_bench_gate_20260521/`.
  - API-26 `cactus_llm_bench` no-env strict Gemma 512 CPU7: decode 5.61 tok/s, prefill 28.49 tok/s, peak RAM 2452.79 MB.
- Decision: API-26 side build is valid for diagnostic model-level XNNPACK experiments. It does not fix the normal API-21 integration blocker.

### H64: native dense qd8/qc8 output-head speed gate

- Status: `supported_for_real_model_hook`
- Observed gap/target: XNNPACK QC8 output-head proxy is 14.47 ms, current orthogonal output-head-sized probe is 61.98 ms, and H59 requires near-XNNPACK LM-head speed to reach about 8.5-8.8 tok/s.
- Mechanisms separated: native SDOT dense-int8 execution vs XNNPACK/KleidiAI QC8 operator execution vs current CQ4 orthogonal path.
- Chosen path/share: add a standalone dense rowwise-int8/qd8 output-head probe to `cq4_kernel_probe` before writing a real-model hook.
- Predicted signatures:
  - Near-XNNPACK native timing would justify a native LM-head hook.
  - Much slower native timing would require an XNNPACK-backed hook or reject LM-head as a practical implementation path.
- Falsifier: native dense-int8 probe does not approach the XNNPACK QC8 speed class.
- Cheapest decisive test: Pixel CPU7 `gemma_orthogonal orthogonal_qd8_qc8_dot4` probe.
- Evidence:
  - Results: `experiments/matrix/results/h64_native_dense_qd8_qc8_output_head_20260521_stdout.txt` and `h64_native_dense_qd8_qc8_output_head_20260521_repeat100_stdout.txt`.
  - Native dense qd8/qc8 output-head probe: 17.522681 ms first run, 18.112995 ms repeat.
  - H59 real LM-head average was 80.226 ms, so the repeated probe implies a 4.43x covered-bucket speedup and about 8.54 tok/s from the 5.56 tok/s baseline at 45.04% LM-head decode share.
- Decision: native dense rowwise-int8/qd8 is fast enough to justify a real-model LM-head hook. This is the first candidate in this branch with both semantic gate evidence and Amdahl headroom near the requested target.

### H65: env-gated native LM-head rowwise-int8/qd8 hook

- Status: `real_model_decode_partial_fix_not_production`
- Observed gap/target: H64 predicts about 8.54 tok/s if the native dense-int8 output-head probe transfers to real Gemma decode. Target remains 8-9 tok/s decode.
- Mechanisms separated: standalone probe speed vs real folded-cache materialization/layout vs per-token dynamic activation quantization overhead vs real-model correctness/RAM impact.
- Chosen path/share: implement only an env-gated hook for the structural LM-head shape. Do not change model files or generic CQ4.
- Predicted signatures:
  - Decode moves from about 5.56 tok/s toward 8.3-8.6 tok/s.
  - Peak RAM rises by about 385 MiB plus overhead.
  - TTFT may rise due one-time cache folding; decode must be evaluated separately.
- Falsifier: decode movement is far below Amdahl or the cache/correctness cost is too high even for a diagnostic.
- Cheapest decisive test: `CACTUS_LMHEAD_QD8_QC8_DIAG=1` strict Gemma 512 CPU7 scout after rebuild/deploy.
- Evidence:
  - Code: `cactus-kernels/src/matmul.cpp`, env var `CACTUS_LMHEAD_QD8_QC8_DIAG=1`.
  - Results: `experiments/matrix/results/h65_lmhead_qd8_qc8_hook_20260521/`.
  - Hook run 1: decode 8.75 tok/s, prefill 3.49 tok/s, TTFT 146708.86 ms, peak RAM 2839.86 MB.
  - Hook repeat: decode 8.64 tok/s, prefill 3.42 tok/s, TTFT 149922.51 ms, peak RAM 2835.93 MB.
  - Adjacent no-env: decode 5.46 tok/s, prefill 27.81 tok/s, TTFT 18413.48 ms, peak RAM 2457.09 MB.
  - Cache builds twice per process, taking 65.6-68.8 s each; per-call output-head run time settles around 21-22 ms.
- Decision: H65 validates the root-cause direction for decode. The Pixel decode gap is largely in the LM-head/output-head path once generic exact-CQ4 routes are ruled out. The candidate reaches 8+ tok/s, but remains a diagnostic partial fix until cache build/TTFT, correctness, and guardrails are solved.

### H66: stored folded-int8 LM-head sidecar with main-style I8MM

- Status: `production_direction_supported`
- Observed gap/target: H65 reaches decode target but fails production because folded-cache construction costs about 133-136 s per process. The target is to store the folded LM-head as int8 and preserve the decode gain without the on-device build.
- Mechanisms separated: stored artifact load/copy cost vs on-device fold cost; main-branch 4-row interleaved I8MM kernel layout vs H65 row-major SDOT layout; affine qd8 correction cost vs dense int8 dot cost.
- Chosen path/share: generate a `CLM8` sidecar containing float32 row scales, int32 row sums, and 4-row interleaved int8 weights. Load it under `CACTUS_LMHEAD_QD8_QC8_CACHE_PATH` and execute the LM-head with a main-branch-style `vmmlaq_s32` kernel.
- Predicted signatures:
  - `sidecar_loaded` appears once, with no `cache_built`.
  - TTFT returns near normal because sidecar load is much cheaper than folding.
  - Per-call output-head time improves over H65 if the main-style I8MM layout is better.
  - Strict Gemma 512 decode reaches the 8-9 tok/s target.
- Falsifier: sidecar load remains TTFT-dominant, decode regresses toward baseline, or the I8MM path is slower than H65 SDOT.
- Cheapest decisive test: build and push the sidecar, run strict Pixel CPU7 Gemma 512 decode 100, and compare against adjacent no-env.
- Evidence:
  - Results: `experiments/matrix/results/h66_stored_lmhead_i8mm_sidecar_20260521/`.
  - Sidecar + I8MM: decode 9.06 tok/s, prefill 37.85 tok/s, TTFT 13527.39 ms, total 24450.23 ms, peak RAM 2447.54 MB.
  - Adjacent no-env same binary: decode 5.55 tok/s, prefill 27.97 tok/s, TTFT 18302.07 ms, total 36138.08 ms, peak RAM 2452.79 MB.
  - Sidecar load appeared once and took 212.928 ms; no on-device cache builds occurred.
  - Per-call output-head time was about 15-17 ms, versus H65 row-major SDOT 21-22 ms.
  - Disassembly contains `smmla`.
  - Sampled numeric check on 8 real hidden samples x 2048 vocab rows: output rel L2 mean 0.00571, max 0.00618, cosine mean 0.999985, top-1 match 8/8.
- Decision: the production direction is supported. The remaining work is to move this from env-gated sidecar diagnostic into the converter/model-loading contract, add broader correctness/quality validation, and run Samsung/second-shape guardrails.

### P0: full-core prefill slowdown

- Status: `pending_after_decode_and_single_thread_regen`
- Observed gap/target: fresh full-core Cactus LFM/Qwen c128 prefill was `5.17x-9.14x` slower on Pixel than Samsung, while other backends were usually `1.1x-1.7x` Samsung/Pixel.
- Scope: full-core prefill only. Do not import decode assumptions unless full-core prefill profiling proves shared code matters.
- Next action after gate: start Tier 0/Tier 1 from `MAY_19_NIGHT_PLAN.md`, define harness/acceptance, then collect fresh full-core prefill baselines before profiling or coding.
- H70 starting hypothesis: existing Pixel full-core prefill is slower than Pixel CPU7 single-thread prefill, so first separate harness/scheduler behavior from kernel scaling. `run_matrix.py` full-core mode exports thread env vars, but `tests/android/cactus_llm_bench.cpp` does not consume `CACTUS_MATRIX_BENCH_THREADS`; it only removes CPU affinity. Cheapest test is direct Gemma 512 prefill-only no-affinity versus CPU7-pinned runs with the same binary/input.
