# Background

## Stable facts copied from V1

The older local Gemma multimodal bundles are:
- `../cactus/weights/gemma-4-e2b-it`
- `../cactus/weights/gemma-4-e2b-it-alt-fixed-rowtuned-mm`

Both older bundles reference missing HF snapshot:
- `b324173c7d5721c2baba7f3b17b3b9b3d34ab1e9`

The fresh investigation bundle is:
- `../cactus/weights/gemma-4-e2b-it-fresh-mm`

The fresh bundle references current cached HF snapshot:
- `905e84b50c4d2a365ebde34e685027578e6728db`

The shared-KV Gemma artifacts are text-only for this investigation because they do not include `vision_encoder` or `audio_encoder`.

## Confirmed prior observations

- HF reference identifies `test_monkey.png` as an orangutan/monkey-class animal.
- HF still works when using Cactus-native expanded image prompt/tensors with `pixel_position_ids` passed as HF `image_position_ids`.
- Fresh Cactus native Gemma originally failed image quality in E001, describing the monkey image as a human/fawn nose.
- Later current artifacts record image preprocessing parity and image-only generation success:
  - `artifacts/E002_after_E003_pillow_resize/image_preprocess_compare.json`
  - `artifacts/E015_gemma_image_after_vision_gate/cactus_gemma_image.json`
- Later artifacts record audio-only and mixed image+audio passing after the audio bias FP16 fix and dynamic Gemma4 audio merge:
  - `artifacts/E030_gemma_audio_after_dynamic_merge/cactus_gemma_audio.json`
  - `artifacts/E031_gemma_mixed_after_dynamic_merge/cactus_gemma_mixed.json`
- Qwen3-VL identifies the same image as a proboscis monkey.
- Parakeet transcribes the paired audio as: `What is in this image?`
- Direct MP3 input to native completion fails with `Not RIFF`; the paired audio must be converted to WAV.

## Local paths

Image:
- `cactus-engine/tests/assets/test_monkey.png`

Audio:
- Source MP3: `../cactus/what_is_in_this_image.mp3`
- Converted WAV when needed: `local_integration_tests/.tmp/what_is_in_this_image.wav`

Harness:
- `local_investigations/gemma4_multimodal_v2/scripts/run_probe.py`

Artifacts:
- `local_investigations/gemma4_multimodal_v2/artifacts/`
