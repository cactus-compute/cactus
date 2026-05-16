# CQ4 Small-M Kernel Optimization Results

## Implementation

- Added a CQ4-only `M=2..4` SDOT path in `cactus-kernels/src/matmul.cpp`.
- The path keeps weights packed and streams one 16-output tile at a time from `packed_indices`.
- Scratch is bounded by `M * K` activation transform/quantization buffers, `M * num_groups` scales, and four `256 * 4` int8 panel stacks per worker.
- Dispatch still uses existing `CactusThreading::parallel_gemm_tiles`; thread counts, worker pool sizing, affinity, command defaults, and scheduling policy are unchanged.
- Unsupported CQ4 shapes continue to fall back through the existing GEMM dispatch.

## Measurement Instrumentation

`cactus_complete` JSON can now report:

- `target_forward_time_ms`
- `avg_target_forward_ms_per_token`
- `target_context_forward_time_ms`
- `avg_target_context_forward_ms_per_token`
- `assistant_forward_time_ms`
- `avg_assistant_forward_ms_per_token`
- `misc_completion_time_ms`
- `avg_misc_completion_ms_per_token`
- `mtp_verifier_width`

In MTP runs, `target_forward_time_ms` is the target verification pass only. The separate `target_context_forward_time_ms` field captures the target context/prep pass used to seed assistant hidden/KV state.

The `cactus-engine/tests/chat.cpp` harness now accepts deterministic benchmark flags:

```bash
--temperature 0 --top-k 1 --confidence-threshold -1 --json
```

For this MTP implementation, effective verifier width maps to `draft.tokens.size() + 1`, reported as `mtp_verifier_width`.

The benchmark helper also supports `--profile-check`, which runs one profiled sample per `M` and checks the target verifier graph's `MATMUL` output row dimensions. Add `--require-small-m` to make that check a hard acceptance gate. This prevents treating a passing timing table as proof that the real path exercised the small-M kernel.

## Kernel Benchmark

Local model-shaped packed CQ4 matmul benchmark: `K=2304`, `N=9216`, `group_size=128`.

Raw five-run output from `./cactus-kernels/build/test_matmul`:

| mode | M | total ms raw | per-token ms raw |
| --- | ---: | --- | --- |
| cq4_packed | 1 | 0.213, 0.194, 0.178, 0.170, 0.176 | 0.2131, 0.1938, 0.1779, 0.1699, 0.1757 |
| cq4_packed | 2 | 0.205, 0.182, 0.187, 0.207, 0.192 | 0.1026, 0.0912, 0.0936, 0.1033, 0.0962 |
| cq4_packed | 3 | 0.200, 0.206, 0.198, 0.216, 0.207 | 0.0667, 0.0687, 0.0659, 0.0721, 0.0690 |
| cq4_packed | 4 | 0.213, 0.208, 0.224, 0.204, 0.204 | 0.0532, 0.0520, 0.0559, 0.0511, 0.0510 |

Median summary:

| mode | M | median ms/token | min | max | stddev | slowdown vs M=1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cq4_packed | 1 | 0.1779 | 0.1699 | 0.2131 | 0.0156 | 0.0% |
| cq4_packed | 2 | 0.0962 | 0.0912 | 0.1033 | 0.0048 | -45.9% |
| cq4_packed | 3 | 0.0687 | 0.0659 | 0.0721 | 0.0022 | -61.4% |
| cq4_packed | 4 | 0.0520 | 0.0510 | 0.0559 | 0.0018 | -70.8% |

This synthetic kernel benchmark isolates the CQ4 target matmul shape and shows the optimized path amortizes packed decode and transform overhead across `M=2..4`.

## Real Target-Forward Benchmark Status

No usable transpiled MTP CQ4 bundle was initially present. The first local benchmark bundle only contained `decoder`, `target_embedding`, and `assistant`; profile checks showed the verifier still executed full static-context rows (`[150]`). After adding cached target component generation and C++ runtime wiring, a fresh bundle was generated with:

```bash
source ./venv/bin/activate
cactus build

cactus convert google/gemma-4-E2B-it /private/tmp/cactus_gemma4_e2b_it_mtp_cached_runtime \
  --bits 4 \
  --device cpu \
  --task causal_lm_logits \
  --torch-dtype bfloat16 \
  --assistant-model google/gemma-4-E2B-it-assistant \
  --local-files-only \
  --max-new-tokens 128 \
  --prompt "Continue this sequence with the next eight words: red blue green yellow"
```

Two packaging fixes were required before the benchmark could run:

- tied Gemma4 `lm_head.weight` now resolves to the packed `token_embeddings.weights` binding instead of materializing a 1.5 GB FP32 saved constant
- assistant packaging reuses the target bundle's captured `prompt_input_ids`, so target and assistant static context lengths match

The resulting target and assistant component captures both used `input_input_ids_shape=[1, 150]`. The target manifest now includes `decoder_prefill_chunk`, `decoder_step`, `decoder_verify_m2`, `decoder_verify_m3`, and `decoder_verify_m4`, with non-empty cache-state metadata for the cached target components.

The stable repeated benchmark fixture was:

```bash
source ./venv/bin/activate
cactus build

python3 cactus-engine/tests/benchmark_mtp_forward.py /private/tmp/cactus_gemma4_e2b_it_mtp_cached_runtime \
  --prompt "Write a short sentence about apples." \
  --max-tokens 8 \
  --warmup 1 \
  --repeats 5 \
  --temperature 0 \
  --top-k 1 \
  --profile-check \
  --require-small-m
```

The helper runs warmup plus five baseline, `M=2`, `M=3`, and `M=4` measured runs. It uses verifier-only `avg_target_forward_ms_per_token` as the acceptance metric and prints median/min/max/stddev raw summaries. Target context/prep, assistant, and miscellaneous timings are reported separately.
It also prints model path, chat binary, commit, CMake build type when available, platform/CPU metadata, prompt text, generation settings, prompt/decode token counts, verifier width, acceptance rate, peak reported RAM, and thread-related environment variables. It does not pass any thread flags to `chat`.

Median summary:

| mode | M | prompt tokens | generated | baseline target ms/token | candidate target ms/token | slowdown | accepted | rejected | acceptance | target context ms/token | assistant ms/token | misc ms/token | peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1 | 16 | 7 | 1278.166857 | 1278.166857 | 0.0% | 0 | 0 | 0.0% | 0.000000 | 0.000000 | 0.024571 | 3625.81 MB |
| mtp | 2 | 16 | 8 | 1278.166857 | 56.408250 | -95.6% | 0 | 8 | 0.0% | 1625.937125 | 4.061250 | 3.308875 | 3772.52 MB |
| mtp | 3 | 16 | 8 | 1278.166857 | 69.448500 | -94.6% | 0 | 8 | 0.0% | 1615.620000 | 7.463625 | 4.834125 | 3885.20 MB |
| mtp | 4 | 16 | 8 | 1278.166857 | 83.406750 | -93.5% | 0 | 8 | 0.0% | 1623.535625 | 10.270875 | 6.121250 | 4019.40 MB |

Raw target-forward summaries:

| M | median | min | max | stddev |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1278.166857 | 1235.094571 | 1312.087571 | 28.105166 |
| 2 | 56.408250 | 55.939500 | 56.882500 | 0.316004 |
| 3 | 69.448500 | 69.085625 | 70.318875 | 0.419692 |
| 4 | 83.406750 | 82.921375 | 84.344250 | 0.505555 |

By the verifier-only benchmark contract, the repeated run meets the numeric slowdown thresholds with large margin: `M=2` is 95.6% faster than baseline, `M=3` is 94.6% faster, and `M=4` is 93.5% faster.

The profile-check helper now scans the whole profiled decode instead of only the final graph execution. This matters because later speculative rounds can have fewer remaining output slots and therefore legitimately execute a smaller verifier width.

Profile-check output:

| M | reported verifier width | graph executions | verifier MATMUL rows | small-M verified |
| ---: | ---: | ---: | --- | --- |
| 2 | 2 | 41 | `[1, 2, 128, 150, 256]` | true |
| 3 | 3 | 67 | `[1, 2, 3, 128, 150, 256]` | true |
| 4 | 4 | 96 | `[1, 2, 3, 4, 128, 150, 256]` | true |

The same command with `--require-small-m` exits with status `0` on `/private/tmp/cactus_gemma4_e2b_it_mtp_cached_runtime`.

## Runtime Notes

The original causal Gemma4 MTP runtime could not exercise true `M=2..4` verifier rows by only changing the C++ verifier loop. The old `decoder` component had one fixed-shape token input captured at conversion time, and `CactusGraph::set_input` copies the graph input buffer's full `byte_size`; a shorter verification token vector still executed the same static decoder graph shape.

The runtime now loads optional cached target sidecar components when present:

- `decoder_prefill_chunk`
- `decoder_step`
- `decoder_verify_m2`
- `decoder_verify_m3`
- `decoder_verify_m4`

The cached path is used only if the component exposes `input_ids`, `position_ids`, `verifier_logits`, and cache-state node metadata. Otherwise it falls back to the previous full-context decoder verifier. Prefix cache preparation resets/copies component cache states with the new graph buffer APIs, runs `decoder_step` for exact prefix remainders, and executes the verifier component over `[base_context.back()] + draft.tokens` with absolute positions.

The remaining performance limitation is no longer the verifier pass. `target_context_forward_time_ms` is still high because each speculative round builds a target prefix cache in addition to the full target context pass needed to seed the assistant hidden/KV inputs. That cost is reported separately and is outside the target-forward verifier acceptance metric, but it remains the next end-to-end MTP bottleneck.

## Completion Audit

| requirement | evidence | status |
| --- | --- | --- |
| report target main-model verifier timing separately from assistant and miscellaneous timing | JSON fields and benchmark table include target verifier, target context/prep, assistant, and misc timings | complete |
| preserve default threading behavior and scheduling | kernel dispatch uses existing `CactusThreading::parallel_gemm_tiles`; no thread flags or defaults changed | complete |
| keep CQ4 weights packed and avoid full unpacked/dequantized caches | small-M kernel streams from packed indices with tile/group-bounded scratch | complete |
| add correctness coverage for CQ4 `M=2`, `M=3`, `M=4`, tails, group sizes, and fallback | `./cactus-kernels/build/test_matmul` reports all 7 tests passed | complete |
| capture focused packed CQ4 small-M kernel performance | kernel microbench shows lower per-token medians for `M=2..4` versus `M=1` | complete |
| run documented target-forward-only benchmark and compare against thresholds | repeated benchmark reports verifier-only medians 93.5-95.6% faster than baseline for `M=2..4` | complete |
| provide C++ primitives needed for cached verifier rollback/copy/reset | graph output buffer snapshot/restore/copy/reset APIs pass KV-cache rollback, graph-to-graph copy, and empty reset tests | enabling primitive complete |
| provide explicit causal cached target verifier component specs | `decoder_prefill_chunk`, `decoder_step`, and `decoder_verify_m2/m3/m4` specs expose `input_ids`, `position_ids`, `verifier_logits`, and internal KV-cache metadata when requested | enabling artifact complete |
| make new assistant-model conversions include cached verifier artifacts | `cmd_convert` requests `decoder,target_embedding,decoder_prefill_chunk,decoder_step,decoder_verify_m2,decoder_verify_m3,decoder_verify_m4` for assistant target bundles | conversion default complete |
| wire cached verifier components into the C++ causal MTP runtime | runtime loads optional cached target components, copies/resets cache states, executes verifier components, and falls back when unavailable | complete |
| prove the real production verifier graph exercises `M=2`, `M=3`, and `M=4` rows | regenerated bundle passes `--profile-check --require-small-m`; profile rows include `2`, `3`, and `4` | complete |

The target-forward-only goal is achieved. End-to-end speculative decode still has a separate prefix cache-preparation bottleneck, visible in `target_context_forward_time_ms`.

## Validation

- `cactus build`: passed.
- `./cactus-kernels/build/test_matmul`: all 7 tests passed.
- `./cactus-engine/tests/build/test_mtp_decode`: all 5 tests passed.
- `./cactus-engine/tests/build/test_model_loading`: all 15 tests passed, including cached verifier branch coverage and assertions for `mtp_verifier_width`, target verifier, target context/prep, assistant, and misc timing fields.
- `./cactus-graph/build/test_cache`: all 16 tests passed, including KV cache snapshot/restore, graph-to-graph copy, and empty reset coverage.
- `python3 -m py_compile cactus-engine/tests/benchmark_mtp_forward.py`: passed.
- `pytest -q python/tests/test_benchmark_mtp_forward.py`: 3 passed.
- `python3 cactus-engine/tests/benchmark_mtp_forward.py /private/tmp/cactus_gemma4_e2b_it_mtp_cached_runtime --prompt "Write a short sentence about apples." --max-tokens 1 --warmup 0 --repeats 1 --temperature 0 --top-k 1 --profile-check --require-small-m`: exited with status `0` and reported `small_m_verified=true` for `M=2..4`.
- `python3 cactus-engine/tests/benchmark_mtp_forward.py /private/tmp/cactus_gemma4_e2b_it_mtp_cached_runtime --prompt "Write a short sentence about apples." --max-tokens 8 --warmup 1 --repeats 5 --temperature 0 --top-k 1 --profile-check --require-small-m`: exited with status `0`; verifier-only medians were `M=2` 56.408250 ms/token, `M=3` 69.448500 ms/token, `M=4` 83.406750 ms/token.
- `python3 -m py_compile python/cactus/transpile/weight_binding.py python/cactus/cli/assistant_bundle.py`: passed.
- `python3 -m py_compile python/cactus/cli/convert.py python/cactus/transpile/model_adapters.py python/cactus/transpile/lower.py`: passed.
- `pytest -q python/tests/test_convert_assistant_components.py python/tests/test_gemma4_cached_verifier_specs.py python/tests/test_benchmark_mtp_forward.py python/tests/test_transpile_weight_binding.py python/tests/test_assistant_bundle_prompt_ids.py`: 7 passed.
- `python3 cactus-engine/tests/benchmark_mtp_forward.py --help`: passed.
- `git diff --check`: passed.
