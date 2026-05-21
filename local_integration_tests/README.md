# Local integration tests

These tests are intentionally local and uncommitted until the suite is ready to become part of the repository.

## Requirements

- Run from the repository root.
- Use existing converted model artifacts only.
- Set `CACTUS_INTEGRATION_WEIGHTS` to override model discovery.
- Default discovery checks `../cactus/weights`, then `./weights`.

## Commands

```bash
PYTHONPATH=python ./venv/bin/python -m pytest local_integration_tests -m smoke
PYTHONPATH=python ./venv/bin/python -m pytest local_integration_tests -m full
PYTHONPATH=python ./venv/bin/python -m pytest local_integration_tests -k parakeet
```

The pytest session builds `./cactus/build.sh` once and compiles `local_integration_tests/.tmp/build/integration_runner`.

## Coverage

- Gemma4 E2B: text and multiturn reuse with the matrix Cactus artifact, `gemma-4-e2b-it-sharedkv`.
- Gemma4 E2B: image, audio, and image+audio combined with the latest local multimodal artifact, preferring `gemma-4-e2b-it-alt-fixed-rowtuned-mm`.
- Qwen3-VL: text and long-prompt chunked prefill smoke.
- Qwen3-VL: image with the matrix Cactus artifact, `qwen3-vl-2b-instruct-reconvert`.
- LFM2-VL: text, image, long-prompt chunked prefill smoke.
- Parakeet TDT: file and PCM transcription.
- Whisper small: full-mode file and PCM transcription when a component-backed local bundle is available.

## Current local result

```bash
PYTHONPATH=python ../cactus/venv/bin/python -m pytest local_integration_tests -m smoke -q -s
# 7 passed, 2 deselected

PYTHONPATH=python ../cactus/venv/bin/python -m pytest local_integration_tests -k parakeet -q -s
# 1 passed, 8 deselected

PYTHONPATH=python ../cactus/venv/bin/python -m pytest local_integration_tests -m full -q -s
# 2 skipped, 7 deselected
```
