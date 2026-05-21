# Experiments

## E1: Paired audio validation

Purpose:
- Verify that the new audio fixture says what the user intended before using it for Gemma mixed testing.

Inputs:
- `../cactus/what_is_in_this_image.mp3`
- Converted WAV: `local_integration_tests/.tmp/what_is_in_this_image.wav`
- Parakeet model: `../cactus/weights/parakeet-tdt-0.6b-v3-transpiled`

Observed:

```text
What is in this image?
```

Interpretation:
- The audio fixture is good.
- The native completion path currently requires WAV input, not MP3.

## E2: Gemma and Qwen image controls

Purpose:
- Confirm whether the monkey image is a valid vision test and whether the Gemma failure is model/path-specific.

Observed:
- Qwen3-VL identifies the image as a proboscis monkey.
- Gemma multimodal artifacts fail to identify the monkey in image-only prompts.

Interpretation:
- The asset is a valid vision control.
- The failure is specific to Gemma artifact/code path behavior, not the image itself.

## E2b: HF reference with Cactus-native prompt style

Purpose:
- Check whether Cactus-native expanded media prompts and native image tensors are themselves valid.

Observed:
- HF current snapshot with normal processor chat template answered: `The animal in this image is an orangutan.`
- HF current snapshot with Cactus-native expanded prompt/tensors also answered: `The animal in this image is an orangutan.`

Interpretation:
- Prompt style and native image tensor preparation are not the high-level cause.
- The remaining fault is in the transpiled/C++ component execution path or artifact lowering/runtime behavior.

## E3: Force Gemma media through media-step path

Purpose:
- Test whether Gemma image-only quality fails because arbitrary prompts are incorrectly routed through the static `lm_encoder` chunk path.

Change tested:
- Temporarily disabled chunk media prefill for Gemma so `prefill_with_media` used the `lm_encoder_media_step` loop.

Observed:
- Gemma image-only monkey prompt produced malformed/gibberish output instead of identifying the image.
- Repeating the same temporary routing change with `gemma-4-e2b-it-fresh-mm` also produced malformed/gibberish output.

Interpretation:
- The simple routing fallback is not valid.
- The investigation should compare actual component contracts and traced prompt assumptions before changing routing further.

## E4: Fresh reconvert artifact

Purpose:
- Determine whether stale artifacts caused the Gemma vision failure.

Artifact:
- `../cactus/weights/gemma-4-e2b-it-fresh-mm`

Observed:
- Fresh artifact points at current cached snapshot `905e84b50c4d2a365ebde34e685027578e6728db`.
- Image-only monkey prompt produced: `The image shows a close-up of a human nose.`
- Similar prompt produced: `The image shows a close-up of a fawn's nose.`

Interpretation:
- Reconvert was necessary to remove stale-artifact uncertainty.
- Reconvert alone does not solve Gemma native vision quality.
