# Questions

## Q1: Where is the first bad boundary?

Status: answered for vision and audio in the current artifact set

Candidate boundaries:
- Prompt/token IDs and media spans.
- Image preprocessing tensors.
- Vision encoder output.
- LM media merge output.
- Decoder first-token logits.
- Autoregressive decode loop.

Resolution:
- For vision, E002/E003/E014/E015 have current artifacts and image-only is recorded as passing.
- For audio, E022/E023 localized and fixed the audio encoder bias issue, and E027/E029 localized and fixed the static media merge issue.

## Q2: Is the fresh Gemma artifact good enough to be the default investigation artifact?

Status: yes for current investigation

Known evidence:
- It points at the current cached HF snapshot.
- E001 confirmed HF on the same snapshot worked while Cactus image-only originally did not.
- Later artifacts record Cactus image-only success on the fresh artifact.
- E030/E031 record audio-only and mixed image+audio passing on the fresh artifact.

Next falsifier:
- If a fresh rerun contradicts E030/E031, inspect the changed artifact/runtime state before changing prompts.

## Q3: Is the previous media-step fallback worth pursuing?

Status: no as a wholesale fallback; dynamic media-step use inside chunked prefill is accepted.

Known evidence:
- Forcing all Gemma media through the old step fallback produced gibberish on mixed prompts.
- The accepted fix uses `lm_encoder_media_step` only to assemble real prompt-position outputs before chunked decoder prefill.

Next falsifier:
- Only revisit after proving the correct `lm_encoder_media_chunk`/step contract and how cache copies should work.
