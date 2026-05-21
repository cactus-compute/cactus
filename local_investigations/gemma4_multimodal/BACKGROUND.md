# Background

## Local artifact inventory

Multimodal Gemma candidates:
- `gemma-4-e2b-it`
- `gemma-4-e2b-it-alt-fixed-rowtuned-mm`
- `gemma-4-e2b-it-fresh-mm`

Text-only/shared-KV Gemma candidates:
- `gemma-4-e2b-it-sharedkv`
- `gemma-4-e2b-it-sharedkv-c128`
- `gemma-4-e2b-it-sharedkv-c256`
- `gemma-4-e2b-it-sharedkv-c512`

The shared-KV bundles do not include `vision_encoder` or `audio_encoder`, so they are not valid vision/audio quality candidates.

## Multimodal component shape facts

Both local multimodal Gemma E2B bundles point at `google/gemma-4-E2B-it` and use the same local HF snapshot.

The two older multimodal bundles point at missing HF snapshot `b324173c7d5721c2baba7f3b17b3b9b3d34ab1e9`.

The fresh bundle points at current cached HF snapshot `905e84b50c4d2a365ebde34e685027578e6728db`.

Component order:
- `audio_encoder`
- `vision_encoder`
- `lm_encoder`
- `decoder_prefill_chunk`
- `lm_encoder_step`
- `lm_encoder_media_step`
- `decoder_step`

Important logical inputs:
- `vision_encoder`: `pixel_values`, `pixel_position_ids`
- `audio_encoder`: `input_features`, `input_features_mask`
- `lm_encoder`: `input_ids`, `image_features`, `audio_features`
- `lm_encoder_media_step`: `inputs_embeds`, `input_ids`, `position_ids`

## Controls

The paired audio created from `../cactus/what_is_in_this_image.mp3` was converted to RIFF/WAV because `cactus_complete` currently rejects MP3 with `Not RIFF`.

Parakeet transcribes the converted WAV as:

```text
What is in this image?
```

Qwen3-VL identifies `test_monkey.png` as a proboscis monkey, so the asset itself is a good vision control.

HF reference on the current cached snapshot identifies the same image with the same general prompt as an orangutan. HF also works when using the Cactus-native expanded image prompt/tensors after renaming `pixel_position_ids` to HF's `image_position_ids`.
