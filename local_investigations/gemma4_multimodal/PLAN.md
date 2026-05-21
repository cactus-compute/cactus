# Plan

## Current phase

Isolate Gemma vision-only correctness before treating mixed image+audio as a product behavior.

## Method

Use a hypothesis-driven loop:
1. Record a hypothesis in `HYPOTHESES.md`.
2. Run the smallest experiment that can support or refute it.
3. Record exact output and interpretation in `EXPERIMENTS.md`.
4. If supported, make the smallest fix and document it in `FIXES.md`.
5. Re-run the focused quality probe and the nearest regression suite.

## Layer order

1. Artifact validity and freshness.
2. Prompt/token construction.
3. Image preprocessing.
4. Vision encoder output.
5. LM media merge behavior.
6. Decoder/cache behavior.
7. Audio preprocessing and encoder behavior.
8. Mixed image+audio behavior.

## Current decision

Investigate whether Gemma multimodal prefill is incorrectly using the static chunk `lm_encoder` path for arbitrary prompts instead of the dynamic `lm_encoder_media_step` path.
