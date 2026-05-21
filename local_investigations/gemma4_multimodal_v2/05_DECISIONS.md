# Decisions

## D1: Do not use older multimodal Gemma artifacts as the default

Decision:
- Use `gemma-4-e2b-it-fresh-mm` for V2 baseline work.

Reason:
- Older artifacts reference a missing HF snapshot.

## D2: Do not continue the simple media-step fallback

Decision:
- Do not disable Gemma chunk prefill wholesale to force `lm_encoder_media_step`.

Reason:
- It produced gibberish on old and fresh artifacts.

## D3: Treat quality probes as acceptance, not root-cause tools

Decision:
- Use quality probes to confirm behavior, but debug via component boundary comparison.

Reason:
- Human text output identified the issue but did not localize the first bad boundary.

## D4: First bad boundary is Gemma4 image preprocessing

Decision:
- Treat E002 Gate B as the first proven divergence.
- Do not debug vision encoder, LM merge, or decoder logits until Gemma4 native preprocessing matches the HF/Python reference or a stronger reference refutes the HF/Python preprocessing tensor as the correct target.

Reason:
- E002 Gate A proves prompt `input_ids` and image token spans match exactly between HF/Python native and C++ Cactus native.
- E002 Gate B proves `pixel_position_ids` match, but `pixel_values` differ immediately and outside tiny tolerance.
- The mismatch exists before any graph runtime, media merge, cache, or decoder behavior is involved.

## D5: Localization is not acceptance

Decision:
- Boundary localization is now treated only as a checkpoint.
- The investigation remains active until Gemma4 native works end to end across image-only, audio-only, and mixed image+audio, with controls and smoke still passing.

Reason:
- The user explicitly raised the acceptance bar from identifying the cause to making the model fully work end to end.

## D6: Image preprocessing is no longer the active boundary

Decision:
- Treat Gemma4 image preprocessing parity as fixed in the current artifact set.

Reason:
- `artifacts/E002_after_E003_pillow_resize/image_preprocess_compare.json` records `status: pass`.
- The recorded `pixel_values` max abs diff is `4.987010955859184e-07`, and mean abs diff is `2.2151478535820782e-07`.

## D7: Image-only generation is currently accepted, but full multimodal is not

Decision:
- Treat Gemma image-only as passing in the current artifact set.
- Do not treat the investigation as complete.

Reason:
- `artifacts/E015_gemma_image_after_vision_gate/cactus_gemma_image.json` records the response `The animal in the image is a golden monkey.`
- `07_ACCEPTANCE.md` allows monkey/orangutan/proboscis monkey class answers for image-only.
- Audio-only and mixed image+audio still return empty responses in the latest recorded artifacts.

## D8: Superseded historical audio / mixed failure

Decision:
- This was the correct decision at E021, but it is superseded by D10 and D11 after E023/E029/E030/E031.
- Do not return to prompt tweaks or decoder parameters unless a fresh rerun contradicts the passing artifacts.

Reason:
- `artifacts/E020_gemma_audio_after_audio_retranspile/cactus_gemma_audio.json` records `response: ""` and `decode_tokens: 0`.
- `artifacts/E017_gemma_mixed_after_vision_gate/cactus_gemma_mixed.json` records `response: ""` and `decode_tokens: 0`.
- `artifacts/E021_audio_full_compare/audio_encoder_compare.json` records matching Cactus/Python audio feature output shapes but poor first-64-row agreement: cosine `0.36957991123199463`, mean abs diff `0.16708998382091522`.

## D9: E021 localizes a boundary, not an internal cause

Decision:
- Use E021 only to justify investigating inside the audio encoder path.
- Do not claim a specific audio op, weight, runtime kernel, or conversion policy is wrong until a smaller comparison identifies it.

Reason:
- E021 compares full emitted audio features against a Python native-like reference and proves a mismatch at that boundary.
- It does not identify the first divergent node inside the audio encoder.

## D10: Gemma4 audio bias tensors must stay FP16

Decision:
- Keep Gemma4 audio component bias tensors as FP16 fallback tensors.

Reason:
- E022 localized the audio encoder divergence to the output projection bias handling.
- E023 shows the FP16 bias policy brings the emitted audio features close to the Python reference.
- Storing these biases as INT8 caused Cactus to add raw integer-like values instead of the expected FP16 bias values.

## D11: Use dynamic Gemma4 merge for prompts with audio

Decision:
- For Gemma4 prompts with audio, do not use the static full `lm_encoder` merge graph.
- Dynamically walk the actual prompt tokens, route text through `lm_encoder_step`, route image/audio token rows through `lm_encoder_media_step`, then feed the assembled tensors to `decoder_prefill_chunk`.

Reason:
- E027 showed the static chunk path selected `<turn|>` for audio-only and mixed prompts after the audio encoder boundary was fixed.
- The static `lm_encoder` graph can encode stale media span assumptions from capture time.
- E029/E030/E031 show the dynamic merge produces normal first tokens and passing end-to-end audio/mixed outputs.

## D12: Do not carry vision changes from `justin/v2-cpp-fix`

Decision:
- Treat `origin/justin/v2-cpp-fix` as already absorbed baseline infrastructure.
- Carry over only the later dynamic Gemma4 audio merge idea from `77f808ed`, adapted to the current branch structure.

Reason:
- The latest `justin/v2-cpp-fix` commit is already in this branch line.
- Vision already has passing local artifacts.
- The relevant audio fix pattern is the dynamic media walk, not a vision-path change.
