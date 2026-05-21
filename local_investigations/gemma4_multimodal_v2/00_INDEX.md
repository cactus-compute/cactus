# Gemma4 Multimodal V2 Index

Status: Gemma4 image-only, audio-only, and mixed image+audio now pass local end-to-end probes after the dynamic audio merge fix; Qwen, Parakeet, policy, and smoke controls pass

Active phase: post-fix review / audit. Do not commit or push until the user reviews.

Default artifact under investigation:
- `../cactus/weights/gemma-4-e2b-it-fresh-mm`

Key controls:
- HF reference snapshot: `905e84b50c4d2a365ebde34e685027578e6728db`
- Qwen image control: `../cactus/weights/qwen3-vl-2b-instruct-reconvert`
- Parakeet audio control: `../cactus/weights/parakeet-tdt-0.6b-v3-transpiled`

Read order:
1. `09_STATUS_FOR_NEXT_AGENT.md`
2. `10_CONTINUATION_PLAN.md`
3. `01_BACKGROUND.md`
4. `02_QUESTIONS.md`
5. `03_HYPOTHESES.md`
6. `04_EXPERIMENTS.md`
7. `05_DECISIONS.md`
8. `06_FIX_LOG.md`
9. `07_ACCEPTANCE.md`
10. `08_HANDOFF.md`

Latest proven artifacts:

- `artifacts/E002_after_E003_pillow_resize/image_preprocess_compare.json`: image preprocessing parity recorded as `status: pass`, max abs diff `4.987010955859184e-07`.
- `artifacts/E014_gateC_after_unscaled_fp16_projection/vision_encoder_compare.json`: latest recorded vision encoder output is FP16-close to the Python/HF reference for `pooler_output`, cosine `0.9999955754245847`, mean abs diff `0.0015996834738693023`.
- `artifacts/E015_gemma_image_after_vision_gate/cactus_gemma_image.json`: image-only Cactus Gemma answered `The animal in the image is a golden monkey.`
- `artifacts/E023_audio_after_fp16_bias/audio_encoder_compare.json`: after the Gemma4 audio bias FP16 fix, first 64 audio rows are FP16-close to Python, cosine `0.9978210926`, mean abs diff `0.00645508`.
- `artifacts/E029_first_token_after_dynamic_merge/`: dynamic merge first-token probes now produce normal text starts, audio `Please`, mixed `This`.
- `artifacts/E030_gemma_audio_after_dynamic_merge/cactus_gemma_audio.json`: audio-only Cactus Gemma responds coherently: `Please provide the image you are referring to. I need an image to tell you what is in it.`
- `artifacts/E031_gemma_mixed_after_dynamic_merge/cactus_gemma_mixed.json`: mixed Cactus Gemma answers: `This image is of a golden monkey.`
- `artifacts/E032_gemma_image_regression_after_dynamic_merge/cactus_gemma_image.json`: image-only still answers: `The animal in the image is a golden monkey.`
- `artifacts/E033_qwen_image_regression_after_dynamic_merge/cactus_qwen_image.json`: Qwen image control identifies a proboscis monkey.
- `artifacts/E034_parakeet_regression_after_dynamic_merge/`: Parakeet file and PCM controls both transcribe `What is in this image?`
- `artifacts/E035_audio_encoder_regression_after_dynamic_merge/audio_encoder_regression_compare.json`: full emitted audio tensor matches the E023 post-bias tensor exactly, max diff `0`.

Latest test commands:
- `source ../cactus/venv/bin/activate && ./cactus/build.sh`
- `PYTHONPATH=python ../cactus/venv/bin/python -m pytest python/cactus/convert/tests/test_policy.py -q` -> `16 passed`
- `PYTHONPATH=python ../cactus/venv/bin/python -m pytest local_integration_tests -m smoke -q -s` -> `7 passed, 2 deselected`

Continuation contract:
- Follow `10_CONTINUATION_PLAN.md`.
- The acceptance probes now pass locally. Next work is review, cleanup, and any additional audit the user requests.
