# Hypotheses

## H1: Artifact staleness caused the original failure

Status: supported but not sufficient; not the active boundary

Evidence:
- Older multimodal artifacts reference a missing HF snapshot.
- Fresh artifact was reconverted from current cached snapshot.
- Fresh artifact originally failed native Gemma vision quality in E001.
- Later artifacts show image-only now passes after targeted fixes.

Decision:
- Use the fresh artifact for V2 unless a stronger artifact is produced for a specific experiment.

## H2: Prompt style or native image preprocessing is invalid

Status: refuted for current image-only path

Evidence:
- HF works with the normal processor prompt.
- HF also works with Cactus-native expanded prompt/tensors.
- E001 stores both records under `artifacts/E001/`.
- `artifacts/E002_after_E003_pillow_resize/image_preprocess_compare.json` records image preprocessing parity as `status: pass`.
- `artifacts/E015_gemma_image_after_vision_gate/cactus_gemma_image.json` records image-only output `The animal in the image is a golden monkey.`

Resolution:
- Audio and mixed image+audio now pass in E030/E031; do not reopen image preprocessing without contradictory fresh evidence.

## H3: The transpiled/C++ component path diverges after preprocessing

Status: historically true for vision and audio; currently fixed in local artifacts

Evidence:
- Vision had later graph/runtime/conversion fixes recorded in `06_FIX_LOG.md`.
- `artifacts/E014_gateC_after_unscaled_fp16_projection/vision_encoder_compare.json` records FP16-close vision output agreement.
- `artifacts/E021_audio_full_compare/audio_encoder_compare.json` records a substantial emitted `audio_features` mismatch against the Python native-like reference.
- `artifacts/E023_audio_after_fp16_bias/audio_encoder_compare.json` records audio features as FP16-close after the audio bias fix.

Next test:
- Rerun E030/E031 if the artifact or runtime changes.

## H4: The C++ media merge/cache path is using the wrong component contract

Status: supported and fixed for Gemma4 prompts containing audio

Evidence:
- Simple `lm_encoder_media_step` fallback produced gibberish.
- Fresh artifact includes `lm_encoder_text_chunk`, `lm_encoder_media_step`, and `lm_encoder_media_chunk`, but C++ currently uses the monolithic `lm_encoder` chunk path for media.
- E025/E026 previously returned empty responses after the audio encoder fix, which made media merge/cache the next boundary.
- After the audio encoder fix, E027 showed the static `lm_encoder` chunk path selected `<turn|>` for audio/mixed first tokens.
- E029/E030/E031 show dynamic prompt-position media merge fixes first-token and generation behavior.

Next test:
- Keep dynamic merge active for Gemma4 prompts with audio and rerun regressions after runtime changes.

## H5: Audio should remain blocked until vision is explained

Status: retired

Evidence:
- Vision-only has a recorded passing artifact in `artifacts/E015_gemma_image_after_vision_gate/cactus_gemma_image.json`.
- Audio-only and mixed image+audio pass in E030/E031.

Next test:
- None unless a fresh rerun contradicts the passing artifacts.
