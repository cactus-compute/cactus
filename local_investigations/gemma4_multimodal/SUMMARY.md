# Summary

Investigation is in progress.

Current leading hypothesis:
- Gemma media prefill has a component-contract mismatch or lowering/runtime mismatch after HF-native prompt/preprocessing, likely around transpiled media encoder/LM merge execution.

Current next step:
- Compare component-level outputs or implement/test the intended dynamic Gemma media chunk path using `lm_encoder_text_chunk` and `lm_encoder_media_chunk`.
