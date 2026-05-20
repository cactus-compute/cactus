# May 19 Night Status

## Purpose

Live execution tracker for tonight's work. Use this for handoffs, active runs, completed runs, blockers, and next-agent instructions.

The main plan stays in `MAY_19_NIGHT_PLAN.md`. Pixel investigation details stay in `PIXEL_OPTIMIZATION_STATUS.md`.

## Current State

- Main plan has three top-level sections:
  1. Correct full-core prefill rows
  2. Pixel slowdown investigation
  3. ASR performance notes
- Parakeet full-core rows for Cactus and ExecuTorch across all devices are now included under section 1.
- Pixel optimization details have a separate status doc.

## Active Runs

- Subgoal: corrected full-core Cactus LFM/Qwen c128 prefill rerun
  Device: pixel_10a (`5C111JEA320125`)
  Command: `source ./venv/bin/activate` then `cactus build` then `python experiments/matrix/run_matrix.py --config experiments/matrix/matrix.yaml --device pixel_10a --runtime cactus --model lfm_2_5_vl_1_6b --model qwen3_vl_2b --benchmark-mode full_core_prefill --full-core-threads 8 --warmup-runs 1 --measurement-runs 3 --out experiments/matrix/results/matrix_pixel_10a_cactus_lfm_qwen_full_core_c128_fresh_runner_escalated_20260520.csv`
  Log path: `experiments/matrix/results/night_logs/pixel_lfm_qwen_full_core_c128_fresh_runner_escalated_20260520.log`
  Output path: `experiments/matrix/results/matrix_pixel_10a_cactus_lfm_qwen_full_core_c128_fresh_runner_escalated_20260520.csv`
  Started: 2026-05-20 09:25:11 PDT
  Next poll no earlier than: 2026-05-20 10:50:28 PDT
  Notes: The previously recorded 09:08 and 09:19 active entries had no matching process and no output/log file as of 2026-05-20 09:25:11 PDT, so the run was restarted. The sandboxed 09:17 rerun wrote `experiments/matrix/results/matrix_pixel_10a_cactus_lfm_qwen_full_core_c128_fresh_runner_20260520.csv` with 10 `unsupported` rows because adb could not connect to its daemon; that artifact is invalid for final CSV patching. `android/build/cactus_llm_bench` was refreshed at 2026-05-20 09:06 after a clean `cactus_llm_bench` target build; `adb devices -l` showed Pixel serial `5C111JEA320125` attached before restart, and run_matrix is expected to redeploy that runner. Polled at 2026-05-20 09:35:35 PDT: session `81878` was still running after completing the local `cactus build` sequence; no final output validation yet. Polled at 2026-05-20 09:46:01 PDT: `run_matrix.py` still active, adb was in `measure_lfm_2_5_vl_1_6b_prefill_2048_1`, output CSV not written yet. Polled at 2026-05-20 09:56:36 PDT: `run_matrix.py` still active, adb was in `measure_lfm_2_5_vl_1_6b_prefill_4096_0`, output CSV not written yet. Polled at 2026-05-20 10:07:09 PDT: `run_matrix.py` still active, adb was in `measure_lfm_2_5_vl_1_6b_prefill_4096_1`, output CSV not written yet. Polled at 2026-05-20 10:18:38 PDT: `run_matrix.py` still active, adb was in `warmup_qwen3_vl_2b_prefill_512_0`, output CSV not written yet. Polled at 2026-05-20 10:29:15 PDT: `run_matrix.py` still active, adb was in `measure_qwen3_vl_2b_prefill_1024_1`, output CSV not written yet. Polled at 2026-05-20 10:40:28 PDT: `run_matrix.py` still active, adb was in `measure_qwen3_vl_2b_prefill_2048_0`, output CSV not written yet.

When a long-running task starts, record:

- command
- device
- output CSV or artifact path
- log path
- start time
- expected next poll time

Poll long-running tasks at most once every 10 minutes unless there is a concrete reason to expect completion or failure sooner.

Active run template:

```text
- Subgoal:
  Device:
  Command:
  Log path:
  Output path:
  Started:
  Next poll no earlier than:
  Notes:
```

Completed result template:

```text
- Subgoal:
  Device:
  Output path:
  Source raw path:
  Command:
  Git SHA / build identifier:
  Runner:
  Device serial:
  Model artifact:
  Benchmark mode:
  Chunk size:
  Validation:
  Decision:
  Next action:
```

## Completed Work

- Subgoal: corrected full-core Cactus LFM/Qwen c128 prefill rerun
  Device: galaxy_s24_or_s25
  Output path: `experiments/matrix/results/matrix_galaxy_s24_or_s25_cactus_lfm_qwen_full_core_c128_fresh_runner_20260520.csv`
  Source raw path: `experiments/matrix/results/matrix_galaxy_s24_or_s25_cactus_lfm_qwen_full_core_c128_fresh_runner_20260520.csv`
  Log path: `experiments/matrix/results/night_logs/samsung_lfm_qwen_full_core_c128_fresh_runner_20260520.log`
  Command: `source ./venv/bin/activate` then `cactus build` then `python experiments/matrix/run_matrix.py --config experiments/matrix/matrix.yaml --device galaxy_s24_or_s25 --runtime cactus --model lfm_2_5_vl_1_6b --model qwen3_vl_2b --benchmark-mode full_core_prefill --full-core-threads 8 --warmup-runs 1 --measurement-runs 3 --out experiments/matrix/results/matrix_galaxy_s24_or_s25_cactus_lfm_qwen_full_core_c128_fresh_runner_20260520.csv`
  Git SHA / build identifier: `40dcd664706d17f754893b2a7760652310642f64` with dirty worktree; `android/build/cactus_llm_bench` mtime `2026-05-20 09:06:00 PDT`; `cactus build` completed immediately before the run.
  Runner: `android_cactus_llm_bench`
  Device serial: `RFGL42B1VLW`
  Model artifact: LFM `weights/lfm2.5-vl-1.6b-retranspile-4096`; Qwen `weights/qwen3-vl-2b-instruct-reconvert`
  Benchmark mode: `full_core_prefill`
  Chunk size: `c128`
  Validation: 10/10 rows `status=ok`; rows are prefill-only with `decode_tokens=0`; notes include `benchmark_mode=full_core_prefill`, `cactus_chunk_size=c128`, corrected artifact paths, `runner=android_cactus_llm_bench`, `serial=RFGL42B1VLW`, `threads=8`, `thread_count=8`, `thread_mode=full_core_prefill`, `affinity=not_set`, and `taskset_mask=none`. Throughput rows: LFM `78.55,78.37,76.97,74.77,57.57`; Qwen `52.04,55.92,50.27,42.98,39.86`.
  Decision: patched `experiments/matrix/results/matrix_galaxy_s24_or_s25_full_core.csv` with these 10 rows; older May 19 Samsung rows are superseded for this goal.
  Next action: wait for the active Pixel LFM/Qwen fresh-runner rerun and validate before patching `matrix_pixel_10a_full_core.csv`.
- Subgoal: corrected full-core Cactus LFM/Qwen c128 prefill rerun
  Device: mac_m4pro
  Output path: `experiments/matrix/results/matrix_mac_m4pro_cactus_lfm_qwen_full_core_c128_rerun_20260520.csv`
  Source raw path: `experiments/matrix/results/matrix_mac_m4pro_cactus_lfm_qwen_full_core_c128_rerun_20260520.csv`; backing raw JSON files are the 40 `experiments/matrix/results/.run_json/cactus_matrix_*.json` files modified from 2026-05-20 01:54:53 PDT through 2026-05-20 01:58:46 PDT.
  Log path: `experiments/matrix/results/night_logs/mac_lfm_qwen_full_core_c128_20260520.log`
  Command: `source ./venv/bin/activate` then `cactus build` then `python experiments/matrix/run_matrix.py --config experiments/matrix/matrix.yaml --device mac_m4pro --runtime cactus --model lfm_2_5_vl_1_6b --model qwen3_vl_2b --benchmark-mode full_core_prefill --full-core-threads 8 --warmup-runs 1 --measurement-runs 3 --out experiments/matrix/results/matrix_mac_m4pro_cactus_lfm_qwen_full_core_c128_rerun_20260520.csv`
  Git SHA / build identifier: `40dcd664706d17f754893b2a7760652310642f64` with dirty worktree; `cactus build` completed immediately before the run.
  Runner: `run_matrix.py` invoking local `cactus run`
  Device serial: n/a (`mac_m4pro` local Mac)
  Model artifact: LFM `weights/lfm2.5-vl-1.6b-retranspile-4096`; Qwen `weights/qwen3-vl-2b-instruct-reconvert`
  Benchmark mode: `full_core_prefill`
  Chunk size: `c128`
  Validation: 10/10 rows `status=ok`; rows are prefill-only with `decode_tokens=0`; notes include `benchmark_mode=full_core_prefill`, `cactus_chunk_size=c128`, `prefill_only=true`, `threads=8`, `thread_count=8`, `thread_mode=full_core_prefill`, `affinity=not_set`, and `taskset_mask=none`; corrected artifact paths are present for both models.
  Decision: patched `experiments/matrix/results/matrix_mac_m4pro_full_core.csv` with these 10 rows.
  Next action: run Android full-core Cactus LFM/Qwen reruns after a fresh Android runner rebuild/redeploy.
- Corrected single-thread CSVs were staged earlier:
  - `experiments/matrix/results/matrix_mac_m4pro_single_thread.csv`
  - `experiments/matrix/results/matrix_pixel_10a_single_thread.csv`
  - `experiments/matrix/results/matrix_galaxy_s24_or_s25_single_thread.csv`
- Created and refined `experiments/matrix/MAY_19_NIGHT_PLAN.md`.
- Added Pixel slowdown investigation to the plan.
- Added Parakeet full-core rows to section 1.
- Ran CQ4 profiling/probe experiments for Pixel diagnosis.
- Pixel full-core Cactus LFM/Qwen rerun attempt notes:
  - The first sandboxed attempt returned only `unsupported` rows because `run_matrix.py` could not connect to the adb daemon.
  - The 2026-05-20 02:13 PDT rerun produced 10 `ok` rows at `experiments/matrix/results/matrix_pixel_10a_cactus_lfm_qwen_full_core_c128_rerun_20260520.csv`, but it is invalid for final CSV patching because `android/build.sh` did not refresh `android/build/cactus_llm_bench`; the stale runner had mtime 2026-05-20 00:42 while `android/CMakeLists.txt` was newer.
  - A clean target build refreshed `android/build/cactus_llm_bench` at 2026-05-20 09:06. The fresh-runner rerun command was interrupted before it started; as of 2026-05-20 09:07:37 PDT there were no active `run_matrix.py`, `cactus_llm_bench`, or device adb benchmark processes, and no fresh-runner output/log file existed.

## Pending Runs

Full-core Cactus prefill reruns:

- Pixel LFM and Qwen
- Samsung LFM and Qwen

Parakeet full-core confirmation:

- Cactus and ExecuTorch on Mac
- Cactus and ExecuTorch on Pixel
- Cactus and ExecuTorch on Samsung

Pixel confirmation:

- Rerun official Pixel/Samsung Cactus rows with freshly rebuilt Android runners.

## Blockers / Caveats

- Old Pixel Cactus numbers may include stale-runner effects.
- Do not patch final full-core CSVs until corrected rows are validated.
- Experimental Pixel kernel probes are diagnostic only unless integrated and rerun in the real model path.
- After C++/CMake/FFI/model-runner changes, Android results require rebuild, redeploy, and runner/version verification before they can be trusted.
- For final CSV edits, keep provenance attached: raw output, command, git SHA or build identifier, runner, serial, model artifact, benchmark mode, and chunk size.
- For Pixel ASR, do not assume the CQ4 LLM diagnosis applies. The rebuilt ASR recheck did not hit CQ4.
- For Pixel kernel work, do not change model files, weight formats, runtime weight layout, threading implementation, or thread choices.

## Subagent Handoff Template

```text
- Agent:
  Hypothesis:
  Worktree / file scope:
  Falsification test:
  Devices tested:
  Real model result:
  Probe result:
  Regressions:
  Conclusion:
  Next action:
```

## Next Agent Instructions

1. Read `MAY_19_NIGHT_PLAN.md` first.
2. Read `MAY_19_NIGHT_STATUS.md` for current execution state.
3. Read `PIXEL_OPTIMIZATION_STATUS.md` before doing Pixel optimization work.
4. Work iteratively: pick one subgoal, finish it, validate it, update this status doc, then move to the next subgoal.
5. Before Cactus commands, follow repo setup:
   - `source ./venv/bin/activate`
   - `cactus build`
6. For kernel changes, test all available target devices: Mac, Pixel, and Samsung. Treat small deltas cautiously because these runs have high variance.
7. Keep final CSV patching separate from diagnostic/probe changes.
8. For Pixel optimization, do not change model files, weight formats, runtime weight layout, threading implementation, or thread choices for a given kernel.
9. Do not let subagents work open-endedly. Give each one a single hypothesis, a scoped worktree or file set, and a concrete falsification test.
