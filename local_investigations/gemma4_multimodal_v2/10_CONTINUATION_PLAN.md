# Continuation Plan

## Goal

Keep fresh Gemma4 Cactus native working end to end for image-only, audio-only, and mixed image+audio, while preserving Qwen, Parakeet, and smoke controls.

The local acceptance plan now passes. This document is for review and any follow-up hardening, not for continuing a known active failure.

## Current local result

- Gemma image-only: passes, identifies `test_monkey.png` as a golden monkey.
- Gemma audio-only: passes, responds coherently to the spoken `what is in this image` prompt when no image is supplied.
- Gemma mixed image+audio: passes, answers `This image is of a golden monkey.`
- Qwen image control: passes, identifies a proboscis monkey.
- Parakeet file and PCM controls: pass, both transcribe `What is in this image?`
- Policy tests: pass, `16 passed`.
- Smoke tests: pass, `7 passed, 2 deselected`.

## Important implementation decisions

- Keep Gemma4 audio bias tensors FP16.
- For Gemma4 prompts with audio, use the dynamic media merge path rather than the static full `lm_encoder` merge graph.
- Do not carry additional vision behavior from `justin/v2-cpp-fix`; that branch is already in the ancestry and vision already passes.
- Keep non-audio/image-only behavior on the existing static chunk path.

## Verification artifacts

- Audio encoder boundary: `artifacts/E023_audio_after_fp16_bias/audio_encoder_compare.json`
- First-token dynamic merge: `artifacts/E029_first_token_after_dynamic_merge/`
- Gemma audio-only: `artifacts/E030_gemma_audio_after_dynamic_merge/cactus_gemma_audio.json`
- Gemma mixed: `artifacts/E031_gemma_mixed_after_dynamic_merge/cactus_gemma_mixed.json`
- Gemma image regression: `artifacts/E032_gemma_image_regression_after_dynamic_merge/cactus_gemma_image.json`
- Qwen regression: `artifacts/E033_qwen_image_regression_after_dynamic_merge/cactus_qwen_image.json`
- Parakeet regression: `artifacts/E034_parakeet_regression_after_dynamic_merge/`
- Audio encoder unchanged after dynamic merge: `artifacts/E035_audio_encoder_regression_after_dynamic_merge/audio_encoder_regression_compare.json`

## If further testing is requested

Run the same sequence under the required build rule:

```bash
source ../cactus/venv/bin/activate
./cactus/build.sh
```

Then rerun:

```bash
PYTHONPATH=python ../cactus/venv/bin/python local_investigations/gemma4_multimodal_v2/scripts/run_probe.py --case cactus-gemma-audio --experiment <new-id>
PYTHONPATH=python ../cactus/venv/bin/python local_investigations/gemma4_multimodal_v2/scripts/run_probe.py --case cactus-gemma-mixed --experiment <new-id>
PYTHONPATH=python ../cactus/venv/bin/python local_investigations/gemma4_multimodal_v2/scripts/run_probe.py --case cactus-gemma-image --experiment <new-id>
PYTHONPATH=python ../cactus/venv/bin/python local_investigations/gemma4_multimodal_v2/scripts/run_probe.py --case cactus-qwen-image --experiment <new-id>
PYTHONPATH=python ../cactus/venv/bin/python local_investigations/gemma4_multimodal_v2/scripts/run_probe.py --case parakeet-audio --experiment <new-id>
PYTHONPATH=python ../cactus/venv/bin/python -m pytest python/cactus/convert/tests/test_policy.py -q
PYTHONPATH=python ../cactus/venv/bin/python -m pytest local_integration_tests -m smoke -q -s
```

Do not commit or push until the user audits the changes.
