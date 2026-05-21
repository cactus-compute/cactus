# Experiments

## Experiment format

Each experiment must include:
- ID
- Question
- Setup
- Command
- Raw result path
- Interpretation
- Next decision

## E001: Baseline ledger

Question:
- Can the harness reproduce the known HF/Cactus/control outputs and store them in a comparable format?

Setup:
- Fresh Gemma artifact: `../cactus/weights/gemma-4-e2b-it-fresh-mm`
- Image: `cactus-engine/tests/assets/test_monkey.png`
- Audio: converted `what_is_in_this_image.wav`

Command:

```bash
python local_investigations/gemma4_multimodal_v2/scripts/run_probe.py --case all-baseline --experiment E001
```

Raw result path:
- `artifacts/E001/`

Interpretation:
- Completed.
- HF Gemma normal prompt: `The animal in this image is an orangutan.`
- HF Gemma Cactus-native expanded prompt/tensors: `The animal in this image is an orangutan.`
- Cactus fresh Gemma native: `The image shows a close-up of a human nose.`
- Cactus Qwen control identifies the proboscis monkey.
- Parakeet file and PCM controls both transcribe: `What is in this image?`

Next decision:
- Move to E002 component-boundary comparison.
- The next probe must localize the first bad boundary after HF-native prompt/preprocessing, not attempt another runtime fix.

## E002: Component-boundary comparison

Question:
- At which component boundary does fresh Cactus native first diverge from the HF/Python reference?

Setup:
- Same artifact, prompt, and image as E001.
- Compare in this order: input IDs/media spans, image tensors, vision encoder output, LM merge output, decoder first-token logits.

Command:
```bash
source ../cactus/venv/bin/activate
./cactus/build.sh
python local_investigations/gemma4_multimodal_v2/scripts/run_probe.py --case e002-boundary --experiment E002
```

Raw result path:
- `artifacts/E002/`
- `artifacts/E002/token_spans.json`
- `artifacts/E002/image_preprocess_compare.json`
- `artifacts/E002/hf_native_inputs.json`
- `artifacts/E002/cactus_token_probe.json`
- `artifacts/E002/cactus_preprocess.json`

Interpretation:
- Completed 2026-05-20.
- Gate A passed: HF/Python native and C++ Cactus native `input_ids` match exactly.
- Gate A image span also matches exactly: token count `282`, assistant generation start `279`, image token id `258880`, image start `5`, image token count `256`.
- Gate B failed at `pixel_values`.
- `pixel_values` shape matches: `[1, 2520, 768]`.
- `pixel_position_ids` shape and values match, with valid position count `2304`.
- `pixel_values` differ immediately at element `0`; max absolute difference is `0.1298471341896057`, mean absolute difference is `0.002023158910580417`.
- HF/Python pixel range is `[0.0, 1.0]`; C++ pixel range is `[-0.0213023126, 1.00900519]`.
- The first divergent boundary is Gemma4 native image preprocessing, before the vision encoder.

Next decision:
- Stop E002 at Gate B for now, then continue with E003. Do not run vision encoder, LM merge, or decoder comparison until preprocessing parity is fixed or a stricter preprocessing reference proves the current C++ tensor is acceptable.
- The next fix should be scoped to `preprocess_gemma4_image` parity with `_prepare_gemma4_native_image_tensors`, including resize/interpolation and numeric range behavior.
- This is not final acceptance; the mission continues until Gemma image-only, audio-only, and mixed image+audio pass end to end.

Superseded by later artifacts below:
- E003 / `E002_after_E003_pillow_resize` records preprocessing parity as passing.
- E015 records image-only generation as passing.
- Audio-only and mixed image+audio are now the active failures.

## Current status correction: later artifacts after E002

This section records the newer artifacts currently present in the worktree. It does not erase E001/E002 history; it updates the active boundary for future agents.

## E003 / E002_after_E003_pillow_resize: Image preprocessing parity

Question:
- Does the C++ Gemma4 image preprocessing output match the Python/HF native reference after the Pillow-compatible resize path?

Raw result paths:
- `artifacts/E003/preprocess_operation_compare.json`
- `artifacts/E002_after_E003_pillow_resize/image_preprocess_compare.json`

Interpretation:
- Completed in current artifacts.
- `artifacts/E002_after_E003_pillow_resize/image_preprocess_compare.json` records `status: pass`.
- `pixel_values` max abs diff is `4.987010955859184e-07`.
- `pixel_values` mean abs diff is `2.2151478535820782e-07`.
- HF/Python range is `[0.0, 1.0]`; C++ range is `[0, 1]`.

Next decision:
- Treat image preprocessing as fixed unless a fresh rerun contradicts this artifact.
- Continue to the vision encoder boundary.

## E014: Latest recorded vision encoder comparison

Question:
- Does the Cactus `vision_encoder` output match the Python/HF reference closely enough after the recorded graph/runtime/conversion fixes?

Raw result path:
- `artifacts/E014_gateC_after_unscaled_fp16_projection/vision_encoder_compare.json`

Interpretation:
- The latest recorded `pooler_output` comparison has matching shape `[256, 1536]`.
- Cosine similarity is `0.9999955754245847`.
- Max abs diff is `0.03515437499999985`.
- Mean abs diff is `0.0015996834738693023`.
- This is recorded as FP16-close. The artifact does not by itself prove every later LM merge/logit boundary, so do not overstate it as full multimodal acceptance.

Next decision:
- Use image-only generation quality as the next acceptance check for the vision path.

## E015: Gemma image-only quality check after vision fixes

Question:
- Does fresh Cactus Gemma image-only generation identify `test_monkey.png` as a monkey-class animal?

Raw result path:
- `artifacts/E015_gemma_image_after_vision_gate/cactus_gemma_image.json`

Interpretation:
- Cactus returned success.
- Response: `The animal in the image is a golden monkey.`
- `prefill_tokens: 282`.
- `decode_tokens: 10`.
- This satisfies the image-only semantic acceptance class recorded in `07_ACCEPTANCE.md`.

Next decision:
- Treat Gemma image-only as passing in the current artifact set.
- Continue to Gemma audio-only and mixed image+audio.

## E016 / E020: Gemma audio-only quality checks

Question:
- Does fresh Cactus Gemma audio-only generation produce coherent text using the paired audio prompt?

Raw result paths:
- `artifacts/E016_gemma_audio_after_vision_gate/cactus_gemma_audio.json`
- `artifacts/E020_gemma_audio_after_audio_retranspile/cactus_gemma_audio.json`

Interpretation:
- Both recorded Cactus runs returned success but no generated text.
- E020 response is empty: `response: ""`.
- E020 has `prefill_tokens: 83` and `decode_tokens: 0`.
- Retranspiling the audio encoder in the recorded E019/E020 sequence did not resolve audio-only generation.

Next decision:
- Audio-only remains an active failure.
- Continue with audio boundary comparison before changing prompts or decoder settings.

## E017: Gemma mixed image+audio quality check

Question:
- Does fresh Cactus Gemma mixed image+audio generation answer the spoken question about `test_monkey.png`?

Raw result path:
- `artifacts/E017_gemma_mixed_after_vision_gate/cactus_gemma_mixed.json`

Interpretation:
- Cactus returned success but no generated text.
- Response is empty: `response: ""`.
- `prefill_tokens: 349`.
- `decode_tokens: 0`.

Next decision:
- Mixed image+audio remains an active failure.
- Do not treat image-only success as full acceptance.

## E018 / E019: Audio summary before and after audio retranspile

Question:
- Do the recorded Cactus audio preprocessing and emitted audio feature summaries change after retranspiling the audio encoder?

Raw result paths:
- `artifacts/E018_audio_summary/audio_summary.json`
- `artifacts/E019_audio_summary_after_retranspile/audio_summary.json`

Interpretation:
- Both artifacts record `num_frames: 255`, `num_soft_tokens: 64`, and `feature_count: 32640`.
- Both artifacts record emitted `audio_features` shape `[1, 753, 1536]` with precision `FP16`.
- The first emitted values are unchanged across E018 and E019.
- These summaries are not enough to prove audio preprocessing parity because they only include prefixes and aggregate statistics.

Next decision:
- Use full-tensor comparison before selecting an audio fix.

## E021: Full Cactus-vs-Python audio feature comparison

Question:
- Do the Cactus emitted `audio_features` match the Python native-like audio feature reference?

Raw result paths:
- `artifacts/E021_audio_full_compare/cactus_audio_summary.json`
- `artifacts/E021_audio_full_compare/audio_encoder_compare.json`

Interpretation:
- `cactus_audio_summary.json` now includes full `audio_features.values`.
- Cactus and Python output shapes match: `[1, 753, 1536]`.
- Python prepared feature shape is `[1, 3012, 128]`; Python input feature mask sum is `255`.
- For the first 64 rows, cosine similarity is `0.36957991123199463`, max abs diff is `2.796870708465576`, and mean abs diff is `0.16708998382091522`.
- For all 753 rows, cosine similarity is `0.7203282713890076`, max abs diff is `2.796870708465576`, and mean abs diff is `0.10658295452594757`.
- The `python_audio_token_count` field in this artifact is not used for conclusions because the comparison script counted a token id that is not validated here.

Next decision:
- The current best proven active boundary is the audio encoder output / emitted audio feature boundary.
- Next work should localize the first divergent internal audio encoder operation or weight. Do not infer a specific internal cause from E021 alone.

## E022 / E023 / E024: Audio encoder localization and FP16 bias fix

Question:
- What caused the emitted Gemma4 `audio_features` mismatch from E021?

Raw result paths:
- `artifacts/E022_audio_node_localization/`
- `artifacts/E023_audio_after_fp16_bias/`
- `artifacts/E024_retranspile_audio_fp16_bias/`

Interpretation:
- E022 localized the first important divergence to the audio output projection bias handling.
- The active artifact had stored `audio_output_proj.bias` as an INT8 fallback, which made Cactus add raw values like `[1, -2, 0, 70]` instead of HF FP16-like bias values.
- After changing Gemma4 audio bias policy to FP16 and updating the active artifact, E023 records first 64 audio rows as FP16-close to Python: cosine `0.9978210926`, mean abs diff `0.00645508`.

Next decision:
- Keep Gemma4 audio bias tensors as FP16.
- Continue to generation and first-token probes because audio-only and mixed still returned empty strings after E023.

## E025 / E026 / E027: Post-audio-encoder generation and first-token failure

Question:
- After fixing the audio encoder boundary, why do audio-only and mixed generation still fail?

Raw result paths:
- `artifacts/E025_gemma_audio_after_fp16_bias/cactus_gemma_audio.json`
- `artifacts/E026_gemma_mixed_after_fp16_bias/cactus_gemma_mixed.json`
- `artifacts/E027_first_token_after_audio_encoder_fix/`

Interpretation:
- Audio-only and mixed still returned empty responses after the audio encoder fix.
- First-token probes showed the static chunk path selected `<turn|>` for both audio-only and mixed prompts, unlike HF references (`The` / `This` starts).
- Forcing the old media-step path improved audio-only first token to `Please`, but mixed produced whitespace, so wholesale fallback was not acceptable.
- The static `lm_encoder` graph was captured with stale media span assumptions and could insert audio at the wrong positions/counts for the current prompt.

Next decision:
- Use a Gemma4-audio-only dynamic media merge inside the chunked prefill path instead of relying on stale static `lm_encoder` merge captures.

## E029: First-token probes after dynamic merge

Question:
- Does the dynamic Gemma4 audio merge fix first-token behavior for audio-only and mixed prompts?

Raw result path:
- `artifacts/E029_first_token_after_dynamic_merge/`

Interpretation:
- Audio-only first token is `Please`, with normal text alternatives in top logits.
- Mixed image+audio first token is `This`, matching the HF-style first-token direction.
- Neither probe selects `<turn|>`, `<eos>`, or whitespace-only as the top token.

Next decision:
- Continue to full generation quality checks.

## E030: Gemma audio-only after dynamic merge

Question:
- Does audio-only Gemma use the paired spoken prompt coherently?

Raw result path:
- `artifacts/E030_gemma_audio_after_dynamic_merge/cactus_gemma_audio.json`

Interpretation:
- Cactus generated: `Please provide the image you are referring to. I need an image to tell you what is in it.`
- This is coherent with the audio prompt saying `what is in this image` and no image being supplied.
- `decode_tokens: 21`.

Next decision:
- Audio-only quality gate passes. Continue to mixed image+audio.

## E031: Gemma mixed image+audio after dynamic merge

Question:
- Does mixed Gemma answer the spoken question about `test_monkey.png`?

Raw result path:
- `artifacts/E031_gemma_mixed_after_dynamic_merge/cactus_gemma_mixed.json`

Interpretation:
- Cactus generated: `This image is of a golden monkey.`
- This satisfies the mixed acceptance class.
- `decode_tokens: 8`.

Next decision:
- Mixed gate passes. Run regressions.

## E032 / E033 / E034 / E035: Regression checks after dynamic merge

Question:
- Did the dynamic merge fix regress image-only Gemma, Qwen, Parakeet, or the audio encoder boundary?

Raw result paths:
- `artifacts/E032_gemma_image_regression_after_dynamic_merge/cactus_gemma_image.json`
- `artifacts/E033_qwen_image_regression_after_dynamic_merge/cactus_qwen_image.json`
- `artifacts/E034_parakeet_regression_after_dynamic_merge/`
- `artifacts/E035_audio_encoder_regression_after_dynamic_merge/audio_encoder_regression_compare.json`

Interpretation:
- Gemma image-only still generated: `The animal in the image is a golden monkey.`
- Qwen image control generated a proboscis-monkey answer.
- Parakeet file and PCM controls both transcribed: `What is in this image?`
- E035 shows the emitted Cactus audio tensor is unchanged from E023, with full-tensor max diff `0`.
- Policy tests passed: `16 passed`.
- Smoke tests passed: `7 passed, 2 deselected`.

Next decision:
- The planned local acceptance gates pass. Keep the working tree uncommitted for user audit.

## E036 / E037 / E038 / E039 / E040: Cleanup verification probes

Question:
- After ignoring/removing generated local outputs, do the required quality probes still regenerate needed files and pass?

Raw result paths:
- `artifacts/E036_cleanup_gemma_image/cactus_gemma_image.json`
- `artifacts/E037_cleanup_gemma_audio/cactus_gemma_audio.json`
- `artifacts/E038_cleanup_gemma_mixed/cactus_gemma_mixed.json`
- `artifacts/E039_cleanup_qwen_image/cactus_qwen_image.json`
- `artifacts/E040_cleanup_parakeet/`

Interpretation:
- Gemma image-only still generated: `The animal in the image is a golden monkey.`
- Gemma audio-only still generated: `Please provide the image you are referring to. I need an image to tell you what is in it.`
- Gemma mixed still generated: `This image is of a golden monkey.`
- Qwen image still generated a proboscis-monkey answer.
- Parakeet file and PCM still transcribed: `What is in this image?`
- The ignored `.tmp` directory was recreated successfully by the harness.

Next decision:
- Cleanup did not regress the acceptance probes or remove benchmark/test harness source.
