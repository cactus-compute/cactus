# Fix Log

Current status: V2 has recorded fixes in the worktree. Do not use the old `No V2 fixes yet` state.

Rules:
- A code change must cite a supported hypothesis and experiment ID.
- A failed fix must be reverted or explicitly documented as intentionally retained.
- Each fix record must include before/after quality and nearest regression result.

Historical fixes from before V2:
- Audio incremental decode changed to feed only the last token.
- Audio encoder mask changed to mark valid frames only.
- Fresh Gemma artifact reconverted as `gemma-4-e2b-it-fresh-mm`.
- Simple media-step fallback was tested and reverted.

## Local investigation tooling

2026-05-20:
- Added local-only E002 probe commands to `local_integration_tests/native_runner.cpp`:
  - `e002-tokens`
  - `e002-preprocess`
- Updated `local_integration_tests/CMakeLists.txt` so the local runner can include internal engine headers.
- Extended `scripts/run_probe.py` with `--case e002-boundary`.
- These are investigation artifacts only, not runtime fixes.

## F1: Gemma4 image preprocessing parity

Supported by:
- `artifacts/E003/preprocess_operation_compare.json`
- `artifacts/E002_after_E003_pillow_resize/image_preprocess_compare.json`

Change:
- `cactus-engine/src/engine_image.cpp` now has a Pillow-compatible uint8 bilinear resize path used by `preprocess_gemma4_image`.

Before:
- `artifacts/E002/image_preprocess_compare.json` recorded `pixel_values` max abs diff `0.1298471341896057` and mean abs diff `0.002023158910580417`.
- C++ pixel range extended outside `[0, 1]`.

After:
- `artifacts/E002_after_E003_pillow_resize/image_preprocess_compare.json` records `status: pass`.
- `pixel_values` max abs diff is `4.987010955859184e-07`.
- `pixel_values` mean abs diff is `2.2151478535820782e-07`.

## F2: Vision graph/runtime/transpile fixes recorded by later artifacts

Supported by:
- `artifacts/E006_gateC_after_fp16_embedding_indices/vision_encoder_compare.json`
- `artifacts/E014_gateC_after_unscaled_fp16_projection/vision_encoder_compare.json`
- `artifacts/E015_gemma_image_after_vision_gate/cactus_gemma_image.json`

Changes visible in the worktree:
- `cactus-graph/src/ops_tensor.cpp` handles FP16 embedding indices in `compute_embedding_node`.
- `cactus-graph/src/ops_math.cpp` adds FP32 binary and reduction support used by scalar/math lowering paths.
- `python/cactus/transpile/lower.py` preserves tensor dtype in `_ensure_scalar_math_tensor`.
- `python/cactus/transpile/runtime_compat.py` preserves tensor dtype for scalar ops, `abs`, and `pow`.
- `python/cactus/transpile/model_adapters.py` updates Gemma4 native-like vision pooling/projection behavior.

Evidence:
- `artifacts/E014_gateC_after_unscaled_fp16_projection/vision_encoder_compare.json` records FP16-close `pooler_output` agreement: cosine `0.9999955754245847`, mean abs diff `0.0015996834738693023`.
- `artifacts/E015_gemma_image_after_vision_gate/cactus_gemma_image.json` records image-only output: `The animal in the image is a golden monkey.`

Limit:
- This does not prove audio-only or mixed image+audio acceptance.

## F3: Gemma4 vision projection conversion policy and active artifact adjustment

Supported by:
- `artifacts/E013_gateC_after_fp16_projection_graph/vision_encoder_compare.json`
- `artifacts/E014_gateC_after_unscaled_fp16_projection/vision_encoder_compare.json`

Changes visible in the worktree:
- `python/cactus/convert/model_adapters/policy.py` makes `model.embed_vision.embedding_projection.weight` an FP16 fallback with reason `vision embedding projection scale-sensitive`.
- `python/cactus/convert/model_adapters/adapters.py` no longer overrides that policy back to CQ.
- `python/cactus/convert/model_adapters/naming.py` and `python/cactus/convert/cactus_adapters/tensor_io.py` remove `embed_vision_proj` from Gemma4 divide-scale handling.
- `python/cactus/convert/tests/test_policy.py` now expects FP16 for the Gemma4 vision projection policy.

Evidence:
- The latest recorded vision comparison improves to FP16-close agreement in `artifacts/E014_gateC_after_unscaled_fp16_projection/vision_encoder_compare.json`.

## F4: Audio full-tensor diagnostic instrumentation

Supported by:
- `artifacts/E021_audio_full_compare/cactus_audio_summary.json`
- `artifacts/E021_audio_full_compare/audio_encoder_compare.json`

Change:
- `local_integration_tests/native_runner.cpp` now emits full `audio_features.values` for `e002-audio-summary`.

Evidence:
- `artifacts/E021_audio_full_compare/cactus_audio_summary.json` contains full emitted `audio_features.values`.
- `artifacts/E021_audio_full_compare/audio_encoder_compare.json` compares those values against Python native-like audio features.

Limit:
- This is diagnostic instrumentation only.
- It proves the emitted audio feature boundary mismatches, but not which internal audio encoder operation or weight is the first cause.

## F5: Gemma4 audio bias FP16 policy and artifact adjustment

Supported by:
- `artifacts/E022_audio_node_localization/`
- `artifacts/E023_audio_after_fp16_bias/audio_encoder_compare.json`
- `artifacts/E024_retranspile_audio_fp16_bias/`

Changes:
- `python/cactus/convert/model_adapters/policy.py` keeps Gemma4 audio component bias tensors in FP16.
- `python/cactus/convert/tests/test_policy.py` expects FP16 for Gemma4 audio bias policy.
- The active fresh Gemma artifact has an FP16 `audio_output_proj.bias` and regenerated `audio_encoder` graph artifacts.

Before:
- E021 first 64 emitted audio rows had cosine `0.3695799112` and mean abs diff `0.16708998` versus Python.

After:
- E023 first 64 emitted audio rows have cosine `0.9978210926` and mean abs diff `0.00645508` versus Python.

## F6: Dynamic Gemma4 audio/media merge for chunked prefill

Supported by:
- `artifacts/E027_first_token_after_audio_encoder_fix/`
- `artifacts/E029_first_token_after_dynamic_merge/`
- `artifacts/E030_gemma_audio_after_dynamic_merge/cactus_gemma_audio.json`
- `artifacts/E031_gemma_mixed_after_dynamic_merge/cactus_gemma_mixed.json`

Changes:
- `cactus-engine/src/model.cpp` now builds Gemma4 LM encoder outputs dynamically for prompts containing audio.
- The dynamic path walks actual prompt tokens, uses `lm_encoder_step` for text, uses `lm_encoder_media_step` for image/audio token rows, and feeds the assembled outputs into `decoder_prefill_chunk`.
- `cactus-engine/src/engine.h` declares the dynamic helper.
- The existing static `lm_encoder` merge path remains for non-audio cases.

Before:
- E025/E026 still returned empty audio-only and mixed responses after the audio encoder fix.
- E027 first-token probes selected `<turn|>` for both audio-only and mixed prompts on the static chunk path.

After:
- E029 first-token probes produce audio `Please` and mixed `This`.
- E030 audio-only produces coherent text responding to the spoken image question.
- E031 mixed image+audio produces `This image is of a golden monkey.`
- E032/E033/E034/E035 and smoke/policy tests record no local regression.

## F7: Cleanup of generated local outputs

Supported by:
- `artifacts/E036_cleanup_gemma_image/cactus_gemma_image.json`
- `artifacts/E037_cleanup_gemma_audio/cactus_gemma_audio.json`
- `artifacts/E038_cleanup_gemma_mixed/cactus_gemma_mixed.json`
- `artifacts/E039_cleanup_qwen_image/cactus_qwen_image.json`
- `artifacts/E040_cleanup_parakeet/`

Changes:
- `.gitignore` now ignores local integration `.tmp` output, local Python bytecode, and local investigation artifact directories.
- Removed rebuildable local `.tmp` and `__pycache__` directories.
- Benchmark source/results under `benchmarks/` were not changed.

Evidence:
- Policy tests passed after cleanup: `16 passed`.
- Smoke tests passed after cleanup: `7 passed, 2 deselected`.
- E036-E040 quality probes passed after cleanup.
