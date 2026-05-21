# Fixes

## F1: Audio incremental decode prompt replay

Status: applied before this investigation workspace was created

Rationale:
- `cactus_complete` passed the entire processed prompt into `decode_with_audio` on each generated token.
- `decode_with_audio` currently delegates to incremental `decode`, so this replayed the prompt into KV cache repeatedly.

Observed effect:
- Mixed-media generation speed improved.
- Gemma audio semantics remained poor, so this was not the full root cause.

## F2: Audio encoder mask valid-frame handling

Status: applied before this investigation workspace was created

Rationale:
- The audio frontend pads feature buffers to graph capacity.
- The mask should mark only valid frames, not all padded frames.

Observed effect:
- Build and smoke tests still pass.
- Gemma audio semantics remained poor, so this was not the full root cause.

## F3: Force Gemma media-step fallback

Status: reverted

Rationale:
- Tested whether the static chunk `lm_encoder` path was the immediate cause of Gemma image-only failure.

Observed effect:
- Output became gibberish.
- The change was reverted.
- The same experiment was repeated with the fresh artifact and was still gibberish.

## F4: Fresh Gemma4 multimodal reconvert

Status: created for investigation

Artifact:
- `../cactus/weights/gemma-4-e2b-it-fresh-mm`

Rationale:
- Existing multimodal manifests referenced a missing older HF snapshot.
- The current cached HF snapshot produces a valid HF reference answer.

Observed effect:
- Fresh artifact no longer depends on the missing snapshot.
- Native Cactus vision quality is still wrong, so further runtime/component investigation is required.
