# Gemma4 multimodal investigation

Local, uncommitted workspace for investigating Gemma4 native vision/audio quality.

Current status:
- Vision-only Gemma native does not reliably identify `cactus-engine/tests/assets/test_monkey.png`.
- Audio-only Gemma native does not yet show evidence that it is using the audio content.
- Mixed image+audio with `what_is_in_this_image.wav` currently fails because the underlying unimodal paths are not known-good.
- Qwen3-VL and Parakeet are useful controls: Qwen identifies the monkey, and Parakeet transcribes the paired audio.

Primary artifacts:
- `../cactus/weights/gemma-4-e2b-it`
- `../cactus/weights/gemma-4-e2b-it-alt-fixed-rowtuned-mm`
- `../cactus/weights/qwen3-vl-2b-instruct-reconvert`
- `../cactus/weights/parakeet-tdt-0.6b-v3-transpiled`

Read in order:
1. `BACKGROUND.md`
2. `PLAN.md`
3. `HYPOTHESES.md`
4. `EXPERIMENTS.md`
5. `FIXES.md`
6. `SUMMARY.md`
