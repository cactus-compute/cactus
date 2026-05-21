# Hypotheses

## H1: Current multimodal artifact is stale or malformed

Status: supported but not sufficient

Evidence so far:
- Both local multimodal Gemma E2B bundles have full multimodal components.
- Both reference the same HF snapshot.
- Shared-KV artifacts are text-only and should not be used for vision/audio quality.
- The older multimodal bundles reference a missing local HF snapshot.
- A fresh bundle was reconverted from the current cached HF snapshot.
- The fresh bundle improves the failure mode but still does not match HF vision quality.

Next evidence needed:
- Keep using `gemma-4-e2b-it-fresh-mm` as the best Cactus artifact for further debugging.
- Do not assume reconversion alone fixes the native runtime path.

## H2: Gemma native image preprocessing diverges from Python/HF preprocessing

Status: open

Evidence so far:
- C++ and Python both appear to use Gemma4-style resizing, patching, and position IDs.
- No tensor-level diff has been run yet.

Next evidence needed:
- Compare `pixel_values` and `pixel_position_ids` between C++ and Python for the same image.

## H3: Gemma multimodal chunk prefill uses a static traced merge plan that is invalid for arbitrary prompts

Status: partially supported, simple fallback refuted

Evidence so far:
- The `lm_encoder` component was traced with a fixed processor prompt embedded in the graph metadata.
- The Python adapter builds a native merge plan during transpile from the example `input_ids`.
- C++ currently sends arbitrary prompt tokens through this `lm_encoder` for media prefill.
- Gemma image-only fails even though Qwen identifies the same image.

Experiment:
- Route Gemma media prefill through `lm_encoder_media_step` instead of the chunk `lm_encoder` path, then re-run image-only quality probes.

Result:
- The simple dynamic fallback produced obvious gibberish for image-only monkey prompts.
- The same result reproduced with the fresh artifact.
- This suggests `lm_encoder_media_step` is not a drop-in prefill replacement as currently wired, or it needs additional exact contract handling.
- The static-merge concern remains plausible, but the fix is not simply disabling chunk prefill for all Gemma media.

## H5: HF/native prompt and preprocessing are sound; the divergence is in transpiled/C++ execution

Status: supported

Evidence so far:
- HF with the normal processor chat template identifies the monkey image as an orangutan.
- HF with Cactus-native expanded prompt/tensors also identifies the image as an orangutan when `pixel_position_ids` are passed as HF `image_position_ids`.
- Fresh Cactus native with the same image says the image is a human/fawn nose.

Next evidence needed:
- Compare component-level outputs or implement/test the intended dynamic Gemma media chunk path using `lm_encoder_text_chunk` and `lm_encoder_media_chunk`.

## H4: Audio frontend or audio mask is wrong

Status: open

Evidence so far:
- The completion decode loop previously replayed the whole prompt for audio generation; this was fixed.
- The audio encoder mask was changed to mark only valid frames.
- Audio-only still does not show evidence of understanding the spoken content.

Next evidence needed:
- Work on audio only after vision-only is explained or fixed.
