# Gemma 3n Accuracy Debugging — Agent Prompt (Updated 2026-03-07)

## Context

You are debugging remaining accuracy issues in Cactus's Gemma 3n implementation.

Cactus is a C++ mobile inference engine (FP16 runtime path). HuggingFace reference runs in bf16.  
The old hypothesis ("attention is where it breaks") is no longer correct after recent fixes.

**Read `inaccuracy_findings.md` first.** It contains the validated timeline and latest measured results.

Do not repeat already-closed investigations unless you are verifying a regression.

---

## Current Ground Truth

### What is fixed

1. Config extraction bug fixed (`num_kv_shared_layers`, `layer_types`, `sliding_window`).
2. Graph reduction bug fixed (`mean(-1)` / `variance(-1)` axis handling).
3. HF patched attention hook bug fixed (fallback causal/sliding mask when `attention_mask=None`).

### What now matches

Attention sub-ops are close to HF when compared correctly:

- L0 attention path: cos ~`0.99997-0.99999`.
- L4 (first full-attention layer) attention path: cos ~`0.9999+`.
- L10/L15 attention path: typically `~0.995-0.998`.

### Where divergence actually starts

At layer 4, after attention:

- `L4_block_normed` vs HF input LN: `~0.9999`
- `L4_block_attn_raw` vs HF self-attn output: `~0.99997`
- `L4_block_post_attn`: `~0.9057`
- `L4_block_output`: `~0.8062`
- final layer stream output: `~0.9133`

So the first major true divergence is in **post-attention block composition**, not in Q/K/V/RoPE/softmax.

Likely area:

- `cactus/models/model_gemma3n.cpp`
  - `build_transformer_block(...)`
  - `build_layer(...)` (AltUp correction + PLI path)

---

## Your Task

Find and fix the remaining root cause in post-attention block math/order/shape handling.

### 1) Add detailed debug nodes in Cactus block path

In `model_gemma3n.cpp`, add debug captures for at least:

- `post_attention_layernorm` output
- `laurel` output
- `hidden + attn + laurel` (pre-`RSQRT2`)
- post-`RSQRT2` residual
- `pre_feedforward_layernorm` output
- raw MLP output (before post-FFN norm)
- post-FFN norm output
- block output before AltUp correction
- stream0 after AltUp correction
- stream0 after PLI addition

Use consistent naming:

- `L{layer}_...` via `gb->register_debug_node(...)`.

### 2) Add corresponding HF captures

Extend `tests/generate_profiles.py` (or add a focused compare script) to capture HF tensors for the same points, from the same layer, with matching token alignment.

### 3) Compare and localize first drop

Start with **layer 4** only.  
Find the first sub-op where cosine drops substantially from ~1.0.

Then confirm the same pattern at one or two deeper layers (for example L10/L15).

### 4) Determine bug vs precision

For the first divergent sub-op, determine whether this is:

- implementation mismatch (formula/order/broadcast/shape),
- or unavoidable precision effects.

To isolate:

- run the same operation in FP32 (where feasible),
- and compare Cactus op semantics line-by-line to HF.

### 5) Implement fix and validate

After fixing, provide:

- before/after cosine table for the affected sub-ops at L4,
- updated layer-output trend (at least L0, L4, L10, L15, L29),
- one short qualitative HF vs Cactus output check (`who are you`).

---

## Key Files

| File | Purpose |
|---|---|
| `inaccuracy_findings.md` | Full investigation history and latest corrected conclusions |
| `cactus/models/model_gemma3n.cpp` | Core block math (`build_transformer_block`, `build_layer`) |
| `cactus/graph/graph_builder.cpp` | Graph op construction and axis behavior |
| `cactus/kernel/kernel_attention.cpp` | Attention kernels (already mostly validated) |
| `cactus/kernel/kernel_matmul.cpp` | FP16/INT8 matmul kernels |
| `cactus/graph/graph_ops_nn.cpp` | Attention/matmul dispatch |
| `hf_gemma3n/modeling_gemma3n.py` | HF reference implementation |
| `tests/generate_profiles.py` | Profile/hook generation |
| `tests/compare_prefill.py` | Existing prefill comparison utility |

---

## Required Run Sequence

Before running any Cactus command in a terminal session:

1. `source ./venv/bin/activate`
2. `cactus build`
3. Run your command

Rebuild after C++/FFI/model changes.

---

## How to Run (Baseline)

```bash
source ./venv/bin/activate
cactus build

# Generate profiles for a layer (prefill context)
python tests/generate_profiles.py --weights weights/gemma-3n-e2b-it-fp16 --layer 4 --prompt "who are you"

# Run Cactus output check
echo "who are you" | tests/build/chat weights/gemma-3n-e2b-it-fp16

# Capture Cactus debug node values
CACTUS_CAPTURE_DIR=/tmp/cactus_attn_caps \
CACTUS_CAPTURE_PREFILL_ONLY=1 \
CACTUS_DISABLE_SPLIT_PREFILL=1 \
CACTUS_CAPTURE_STDOUT=1 \
CACTUS_THREADS=1 \
echo "who are you" | tests/build/chat weights/gemma-3n-e2b-it-fp16
```

HF reference (bf16 only):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("google/gemma-3n-E2B-it", dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
ids = tokenizer.encode("<start_of_turn>user\nwho are you<end_of_turn>\n<start_of_turn>model\n")
out = model.generate(torch.tensor([ids]), max_new_tokens=50, do_sample=False)
print(tokenizer.decode(out[0]))
```

---

## Important Notes

- Use `weights/gemma-3n-e2b-it-fp16` for debugging.
- HF fp32 likely OOMs; use bf16.
- Keep BOS/token alignment explicit when comparing.
- Shared-KV layers (`>=20`) require careful hook/cache handling; mismatched cache mode can create fake divergence.
- Non-determinism at `temperature=0` can still occur from parallel FP16 reductions.

---

## Deliverable

Produce a concise report with:

1. First true divergent block sub-op (exact node names and cosine values).
2. Root cause classification (bug vs precision) with code references.
3. Patch summary.
4. Before/after metrics and one output-quality sanity check.
