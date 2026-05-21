# Acceptance

## End-to-end gate

Gemma4 native multimodal work is accepted only when the full user path works:
- Image-only Gemma correctly identifies `test_monkey.png` as a monkey/orangutan/proboscis monkey class animal.
- Audio-only Gemma shows it uses the paired spoken prompt coherently.
- Mixed image+audio Gemma answers the spoken question about `test_monkey.png` correctly.
- Any component boundary divergence found during debugging has a before/after artifact showing the fix.
- Qwen image control still works.
- Parakeet file and PCM controls still transcribe coherently.
- Smoke tests pass.

Localization is not acceptance. It is a checkpoint that determines the next fix.

## Vision gate

Gemma native vision is accepted only when:
- HF and Cactus use the same image and semantically equivalent prompt.
- The first bad component boundary has been identified and fixed or ruled out.
- Cactus output identifies the monkey image as a monkey/orangutan/proboscis monkey class animal.
- Qwen image control still works.

## Audio gate

Gemma audio work starts after vision has a known-good path. A documented vision boundary failure is not enough to accept the mission.

Audio is accepted only when:
- The paired WAV is verified by Parakeet.
- Gemma audio preprocessing and audio encoder behavior are compared against the Python/HF reference.
- Cactus output shows evidence of using the spoken content.

## Mixed gate

Mixed image+audio is accepted only after vision and audio have each passed their gates and the combined path works.

Mixed image+audio must use the actual paired test assets:
- Image: `cactus-engine/tests/assets/test_monkey.png`
- Audio prompt: `what is in this image`
- Expected answer: a short coherent answer identifying the monkey/orangutan/proboscis monkey class.

## Regression gate

After any code fix:

```bash
source ../cactus/venv/bin/activate
./cactus/build.sh
PYTHONPATH=python ../cactus/venv/bin/python -m pytest local_integration_tests -m smoke -q -s
```
