# Loading and Running the TQH Packed Model

## What's in the archive

```
packed_shipping/
  bf16_baseline/        held-constant tensors (norms, vision/audio towers, tokenizer, config)
  pli_embed/            PLI + embedding weights, 2–3 bit packed
  tqh_u4/               transformer linears, 4-bit TQH  (~0.96 GB)   ← highest quality
  tqh_u3/               transformer linears, 3-bit TQH  (~0.74 GB)
  p3/                   transformer linears, 3-bit non-Hadamard
  p6/                   transformer linears, 6-bit
  tqh_runtime.py        self-contained loader (copy this into your project)
```

Pick one transformer config (`tqh_u4`, `tqh_u3`, `p3`, or `p6`). The instructions below use `tqh_u4`.

## Prerequisites

```bash
pip install torch safetensors scipy transformers numpy
```

## Extract the archive

```bash
tar xf packed_shipping-2.tar
```

## Assemble and run

```python
import torch
from transformers import AutoProcessor
from packed_shipping.tqh_runtime import assemble_model_from_slim

# Dehydrate weights and patch them into the model architecture.
# This is CPU-only; peaks at ~14 GB RAM during PLI dehydration.
model = assemble_model_from_slim(
    bf16_baseline_dir    = "packed_shipping/bf16_baseline",
    pli_embed_packed_dir = "packed_shipping/pli_embed",
    transformer_packed_dir = "packed_shipping/tqh_u4",   # swap for tqh_u3, p3, p6
)
model.eval()

processor = AutoProcessor.from_pretrained("packed_shipping/bf16_baseline")

messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", tokenize=True,
    return_dict=True,
)

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=200)

response = processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

## Optional: save the assembled model to disk

Pass `out_path` to avoid re-dehydrating on every run:

```python
model = assemble_model_from_slim(
    bf16_baseline_dir      = "packed_shipping/bf16_baseline",
    pli_embed_packed_dir   = "packed_shipping/pli_embed",
    transformer_packed_dir = "packed_shipping/tqh_u4",
    out_path               = "assembled_tqh_u4",   # saved as safetensors
)
```

Then load instantly next time:

```python
from transformers import Gemma4ForConditionalGeneration
model = Gemma4ForConditionalGeneration.from_pretrained(
    "assembled_tqh_u4", torch_dtype=torch.bfloat16, local_files_only=True,
)
```

## Approximate disk and RAM requirements

| Config   | Archive size | Peak RAM (assembly) | Model RAM |
|----------|-------------|---------------------|-----------|
| tqh_u4   | ~2.7 GB     | ~14 GB              | ~4.5 GB   |
| tqh_u3   | ~2.3 GB     | ~14 GB              | ~4.2 GB   |
| p3 / p6  | similar     | ~14 GB              | ~4.2–5 GB |

The ~14 GB peak during assembly is from the PLI/embed dehydration step; it drops back down once that completes.
