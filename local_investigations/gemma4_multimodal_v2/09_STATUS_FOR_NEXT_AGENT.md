# Status For Next Agent

## One-line status

Gemma4 image-only, audio-only, and mixed image+audio now pass local end-to-end probes after the audio bias FP16 fix and dynamic Gemma4 audio merge. Qwen, Parakeet, policy, and smoke controls also pass.

## Current working state

Repo:
- `/Users/noahcylich/Documents/Desert/cactus-v2-chunked-prefill`

Branch:
- `v2-chunked-prefill`

Do not commit or push unless the user explicitly asks. The worktree contains unrelated and related tracked changes plus local investigation artifacts.

Default Gemma artifact:
- `../cactus/weights/gemma-4-e2b-it-fresh-mm`

HF reference snapshot:
- `/Users/noahcylich/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/905e84b50c4d2a365ebde34e685027578e6728db`

Assets:
- Image: `cactus-engine/tests/assets/test_monkey.png`
- Audio WAV: `local_integration_tests/.tmp/what_is_in_this_image.wav`
- Source MP3: `../cactus/what_is_in_this_image.mp3`

## Fixes applied

- Gemma4 audio component bias tensors are kept FP16. E023 shows the audio encoder boundary became FP16-close to Python after this.
- Gemma4 prompts containing audio now bypass the stale static full `lm_encoder` merge graph. The runtime dynamically walks the actual token sequence, routes text through `lm_encoder_step`, routes image/audio token rows through `lm_encoder_media_step`, assembles the LM encoder outputs, and feeds `decoder_prefill_chunk`.
- This keeps the existing static path for image-only and non-audio cases.

## Latest proof artifacts

- `artifacts/E023_audio_after_fp16_bias/audio_encoder_compare.json`: first 64 audio rows cosine `0.9978210926`, mean abs diff `0.00645508` versus Python.
- `artifacts/E029_first_token_after_dynamic_merge/`: audio first token `Please`, mixed first token `This`; no `<turn|>` or immediate stop.
- `artifacts/E030_gemma_audio_after_dynamic_merge/cactus_gemma_audio.json`: audio-only response is coherent with the spoken prompt: `Please provide the image you are referring to. I need an image to tell you what is in it.`
- `artifacts/E031_gemma_mixed_after_dynamic_merge/cactus_gemma_mixed.json`: mixed response: `This image is of a golden monkey.`
- `artifacts/E032_gemma_image_regression_after_dynamic_merge/cactus_gemma_image.json`: image-only response: `The animal in the image is a golden monkey.`
- `artifacts/E033_qwen_image_regression_after_dynamic_merge/cactus_qwen_image.json`: Qwen identifies a proboscis monkey.
- `artifacts/E034_parakeet_regression_after_dynamic_merge/`: Parakeet file and PCM both transcribe `What is in this image?`
- `artifacts/E035_audio_encoder_regression_after_dynamic_merge/audio_encoder_regression_compare.json`: full emitted audio tensor matches E023 exactly, max diff `0`.

## Latest commands run

Required build:

```bash
source ../cactus/venv/bin/activate
./cactus/build.sh
```

Policy:

```bash
PYTHONPATH=python ../cactus/venv/bin/python -m pytest python/cactus/convert/tests/test_policy.py -q
```

Result:
- `16 passed`

Smoke:

```bash
PYTHONPATH=python ../cactus/venv/bin/python -m pytest local_integration_tests -m smoke -q -s
```

Result:
- `7 passed, 2 deselected`

## Remaining work

No known local acceptance failure remains from the planned test matrix. Next step is user audit/review of the uncommitted changes.
