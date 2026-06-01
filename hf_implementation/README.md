# DeepSeek V4 Flash HF Implementation Snapshot

Fetched for local architecture study before implementing DeepSeek V4 Flash in cactus.

Sources:
- `model_repo/`: `deepseek-ai/DeepSeek-V4-Flash` on Hugging Face Hub, main branch.
- `transformers/`: Hugging Face `transformers` main branch, `src/transformers/models/deepseek_v4/`.

Notes:
- The model repo provides reference inference code under `model_repo/inference/` and tokenizer encoding code under `model_repo/encoding/`.
- Transformers currently provides `configuration_deepseek_v4.py` and `modeling_deepseek_v4.py`; there was no `tokenization_deepseek_v4.py` at the fetched path.
- Large weight/index/tokenizer JSON artifacts were intentionally not copied here.
