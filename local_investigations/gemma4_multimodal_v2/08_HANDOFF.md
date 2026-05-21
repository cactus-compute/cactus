# Handoff

Status: local acceptance probes pass; leave uncommitted for user audit.

Current proven state:
- Gemma4 image preprocessing parity is fixed.
- Gemma4 vision/image-only generation passes.
- Gemma4 audio encoder output is FP16-close to Python after the audio bias FP16 fix.
- Gemma4 audio-only generation is coherent.
- Gemma4 mixed image+audio answers the spoken question about `test_monkey.png` correctly.
- Qwen image, Parakeet file/PCM, policy tests, and smoke tests pass.

## Key artifacts

- Image preprocessing: `artifacts/E002_after_E003_pillow_resize/image_preprocess_compare.json`
- Vision encoder: `artifacts/E014_gateC_after_unscaled_fp16_projection/vision_encoder_compare.json`
- Audio encoder after bias fix: `artifacts/E023_audio_after_fp16_bias/audio_encoder_compare.json`
- First-token dynamic merge: `artifacts/E029_first_token_after_dynamic_merge/`
- Gemma audio-only: `artifacts/E030_gemma_audio_after_dynamic_merge/cactus_gemma_audio.json`
- Gemma mixed: `artifacts/E031_gemma_mixed_after_dynamic_merge/cactus_gemma_mixed.json`
- Gemma image regression: `artifacts/E032_gemma_image_regression_after_dynamic_merge/cactus_gemma_image.json`
- Qwen regression: `artifacts/E033_qwen_image_regression_after_dynamic_merge/cactus_qwen_image.json`
- Parakeet regression: `artifacts/E034_parakeet_regression_after_dynamic_merge/`
- Audio encoder unchanged after dynamic merge: `artifacts/E035_audio_encoder_regression_after_dynamic_merge/audio_encoder_regression_compare.json`

## Latest outputs

- Gemma image-only: `The animal in the image is a golden monkey.`
- Gemma audio-only: `Please provide the image you are referring to. I need an image to tell you what is in it.`
- Gemma mixed: `This image is of a golden monkey.`
- Qwen image: identifies a proboscis monkey.
- Parakeet file and PCM: `What is in this image?`

## Tests

- Build completed with `source ../cactus/venv/bin/activate && ./cactus/build.sh`.
- `PYTHONPATH=python ../cactus/venv/bin/python -m pytest python/cactus/convert/tests/test_policy.py -q` -> `16 passed`.
- `PYTHONPATH=python ../cactus/venv/bin/python -m pytest local_integration_tests -m smoke -q -s` -> `7 passed, 2 deselected`.

## Review focus

- Runtime: dynamic Gemma4 audio merge in `cactus-engine/src/model.cpp`.
- Policy: Gemma4 audio bias FP16 handling in `python/cactus/convert/model_adapters/policy.py`.
- Artifact changes: active fresh Gemma audio encoder graph/bias files were adjusted during the investigation.
