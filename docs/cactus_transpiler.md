---
title: "Cactus Transpiler"
description: "Convert any PyTorch model to run on-device with Cactus. Step-by-step guide from a simple PyTorch model to on-device inference."
keywords: ["transpiler", "PyTorch", "torch.export", "on-device AI", "model conversion", "custom models"]
---

# Cactus Transpiler

The Cactus Transpiler takes a PyTorch model and turns it into a Cactus runtime graph
that runs on ARM devices. You write normal PyTorch, and the transpiler handles the rest:
capturing the computation graph, fusing ops, quantizing weights, and producing a bundle
that loads with zero-copy memory mapping.

## From PyTorch to On-Device in 2 Steps

### Step 1: Write a PyTorch Model

Any `torch.nn.Module` works. Here's a small feedforward classifier:

```python
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size=32000, embed_dim=256, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)       # (batch, seq, embed_dim)
        x = x.mean(dim=1)                   # (batch, embed_dim)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = TextClassifier()
torch.save(model.state_dict(), "classifier.pt")
```

### Step 2: Convert and Run

`cactus convert` quantizes the weights to Cactus format (writing a
`weights_manifest.json`) and then captures the graph, optimizes it, and saves a
self-contained bundle — all in one step:

```bash
cactus convert ./classifier.pt ./classifier-cactus --bits 4

# Run the converted bundle
cactus run ./classifier-cactus
```

That's it. The output is a self-contained bundle with memory-mapped weights that
loads in milliseconds on any ARM device.

---

## How It Works

The transpiler is a six-stage compiler pipeline:

```
PyTorch nn.Module
      |
      v
+------------------+
| 1. Canonicalize  |  Wrap the model in a task-specific adapter
|    model         |  (causal LM, encoder, multimodal, etc.)
+--------+---------+
         |
         v
+------------------+
| 2. torch.export  |  Capture the forward pass as an FX graph
|                  |  (no Python control flow, tensor-only)
+--------+---------+
         |
         v
+------------------+
| 3. Import to IR  |  FX nodes -> canonical IRGraph
|                  |  Bind converted weight files from manifest
+--------+---------+
         |
         v
+------------------+
| 4. Canonicalize  |  Normalize ops, constant-fold, remove no-ops,
|    IR            |  insert precision casts, dead code elimination
+--------+---------+
         |
         v
+------------------+
| 5. Fuse patterns |  Collapse subgraphs into single ops:
|                  |  RMSNorm, RoPE, attention, MLP, LSTM, etc.
+--------+---------+
         |
         v
+------------------+
| 6. Lower to      |  IRNodes -> CactusGraph ops
|    CactusGraph   |  Constants -> mmap_weights() or inline tensors
+------------------+
         |
         v
   Saved bundle: graph.cactus + manifest.json + weights
```

### Stage 1: Canonicalize the Model

The transpiler doesn't try to handle every model's forward signature as-is. It wraps
the model in a thin adapter that exposes a stable, task-specific interface:

| Task | Input | Output |
|------|-------|--------|
| `causal_lm_logits` | `input_ids` | next-token logits |
| `multimodal_causal_lm_logits` | `input_ids`, image/audio features | next-token logits |
| `encoder_hidden_states` | `input_features` | hidden states |
| `ctc_logits` | `input_features` | CTC logits |
| `seq2seq_transcription` | `input_features` | transcription token IDs |
| `tdt_transcription` | `input_features` | TDT token IDs |

This is where model-family-specific knowledge lives (Gemma4, Qwen, LFM2, Parakeet, etc.).
The adapter also injects `use_cache=False` and `return_dict=False` so the export
boundary stays tensor-only.

### Stage 2: Capture with `torch.export`

The adapted model is captured using `torch.export`, producing an FX graph where every
operation is a canonical ATen function call (e.g. `aten.linear.default`, `aten.silu.default`).

This is the generality boundary. After capture, the transpiler works with ATen op names,
not layer names. It doesn't care whether your model calls its attention layer `self_attn`
or `mha` — it recognizes the computation pattern.

### Stage 3: Import to IR

The FX graph is converted into Cactus's intermediate representation (`IRGraph`), a simple
dataflow graph of `IRNode`s connected by `IRValue`s.

During import, constants that correspond to converted weight files are resolved through
`weights_manifest.json`. This metadata follows the value through the pipeline so that
lowering can use `mmap_weights()` instead of embedding the tensor into the graph.

### Stage 4: Canonicalize the IR

Multi-pass cleanup:

- Rename ops to canonical spellings
- Rewrite `transpose`/`movedim` into `permute`
- Constant-fold trivial expressions
- Materialize `ones`, `zeros`, `arange` constants
- Remove no-op casts, views, slices
- Insert FP16 precision casts where the runtime requires them
- Dead code elimination

### Stage 5: Fuse Patterns

The optimizer recognizes computational subgraphs and collapses them into single
high-level ops:

| Pattern | Fused Op |
|---------|----------|
| `variance -> rsqrt -> mul -> mul` | `rms_norm` |
| `cos/sin interleave on position freqs` | `rope` |
| `Q @ K^T -> scale -> mask -> softmax -> @ V` | `attention` |
| `gate_proj * up_proj -> activation -> down_proj` | `dense_mlp_tq_fused` |
| `conv -> batchnorm -> activation` | `conv_module` |
| `input/forget/cell/output gates` | `lstm_cell` |

These fused ops map directly to optimized Cactus kernels.

### Stage 6: Lower to CactusGraph

Each `IRNode` becomes one or more `CactusGraph` operations. Constants become:

- `graph.mmap_weights(path)` — zero-copy memory-mapped from disk (most weights)
- Inline scalar values
- Materialized tensor nodes (small constants baked into the graph)

The output is a `TranspiledGraph` containing the runtime graph, input/output handles,
and weight binding metadata.

---

## Converting HuggingFace Models

The most common use case is converting a model from HuggingFace. `cactus convert`
quantizes the weights and builds the runtime graph in one step:

```bash
# 1. Convert (CQ weights + runtime bundle)
cactus convert google/gemma-4-E2B-it ./gemma4-weights --bits 4

# 2. Run
cactus run ./gemma4-weights --prompt "Hello!"
```

### Text Models

```bash
cactus convert Qwen/Qwen3-0.6B ./qwen3-weights --bits 4 \
  --prompt "The capital of France is"
```

### Multimodal Models (Gemma4)

Multimodal models are automatically split into components (`vision_encoder`,
`audio_encoder`, `lm_encoder`, `decoder`) that are each independently captured
and optimized:

```bash
cactus convert google/gemma-4-E2B-it ./gemma4-weights --bits 4 \
  --task multimodal_causal_lm_logits \
  --image-file photo.jpg \
  --audio-file speech.wav \
  --prompt "Describe what you see and hear."
```

### Speech Models (Parakeet TDT)

```bash
cactus convert nvidia/parakeet-tdt-0.6b-v3 ./parakeet-weights --bits 4 \
  --audio-file recording.wav \
  --task tdt_transcription
```

---

## Running Saved Bundles

Transpiled bundles are saved to `./weights/<model>/` by default (alongside their CQ
weights). Run them later without re-transpiling:

```bash
# Text
cactus run ./weights/qwen3-0.6b --prompt "Write a haiku"

# Multimodal
cactus run ./weights/gemma-4-e2b-it \
  --image photo.jpg \
  --audio speech.wav \
  --prompt "What do you see?"

# Audio
cactus run ./weights/parakeet-tdt-0.6b-v3 \
  --audio meeting.wav
```

`cactus run` accepts either a HuggingFace model id (downloads the bundle) or a
local bundle directory path (detected by the presence of
`components/manifest.json`). The standalone `run-transpiled` subcommand was
removed — `cactus run` handles both cases.

---

## Using the Python API Directly

You can also drive the transpiler from Python:

```python
from cactus.transpile.capture_pytorch import capture_model
from cactus.transpile.lower import transpile_ir
from cactus.bindings.cactus import Graph
import torch
import numpy as np

# Define your model
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32)

    def forward(self, x):
        return torch.relu(self.linear(x))

model = MyModel().eval()
example_input = torch.randn(1, 64)

# Capture and transpile
captured = capture_model(model, (example_input,))
transpiled = transpile_ir(captured.ir_graph)

# Run
transpiled.set_input(0, np.random.randn(1, 64).astype(np.float16))
outputs = transpiled.execute()
result = outputs[0].numpy()
print("Output shape:", result.shape)  # (1, 32)
```

### Generic JAX User Graphs

JAX and Flax models can be bundled by providing params and one or more graph
entrypoints. The caller owns tokenization, masks, sampling, and any prefill/decode
loop; Cactus captures the supplied functions, writes FP16 mmap weights, and saves
component graphs.

```python
import jax.numpy as jnp
import numpy as np

from cactus.transpile.capture_jax import JaxGraphSpec
from cactus.transpile.jax_user_graph_bundle import build_jax_user_graph_bundle
from cactus.transpile.jax_user_graph_bundle import load_jax_user_graph_bundle


def model(params, x):
    return jnp.maximum(x @ params["w"] + params["b"], 0)


params = {"w": jnp.ones((4, 3), jnp.float16), "b": jnp.zeros((3,), jnp.float16)}
example_x = jnp.ones((2, 4), jnp.float16)

result = build_jax_user_graph_bundle(
    params=params,
    output_dir="/tmp/my-jax-bundle",
    model_id="my-jax-model",
    specs=(JaxGraphSpec(name="forward", fn=model, example_args=(example_x,)),),
)

loaded = load_jax_user_graph_bundle(result.output_dir)
y = loaded.execute("forward", np.ones((2, 4), np.float16))[0].numpy()
```

For encoder-decoder models, provide each graph boundary explicitly:

```python
params, model, tokenizer, config = load_needle_model()
src = tokenizer.encode("What time is it in Tokyo?")
tgt = tokenizer.encode("<bos>")
src_mask = make_padding_mask(src, config.pad_token_id)
tgt_mask = make_causal_mask(tgt.shape[1])
encoder_out, enc_mask = model.apply({"params": params}, src, src_mask, method=model.encode_text)

result = build_jax_user_graph_bundle(
    params=params,
    output_dir="weights/needle",
    model_id="needle",
    task="encoder-decoder",
    specs=(
        JaxGraphSpec(
            name="encoder",
            fn=lambda params, src, src_mask: model.apply(
                {"params": params}, src, src_mask, method=model.encode_text
            ),
            example_args=(src, src_mask),
            output_names=("encoder_out", "encoder_mask"),
        ),
        JaxGraphSpec(
            name="decoder_prefill",
            fn=lambda params, tgt, encoder_out, tgt_mask, cross_mask: model.apply(
                {"params": params}, tgt, encoder_out, tgt_mask, cross_mask, method=model.decode
            ),
            example_args=(tgt, encoder_out, tgt_mask, enc_mask),
            output_names=("logits",),
        ),
    ),
)

loaded = load_jax_user_graph_bundle(result.output_dir)
encoder_out, enc_mask = loaded.execute("encoder", src, src_mask)
logits = loaded.execute("decoder_prefill", tgt, encoder_out, tgt_mask, enc_mask)[0].numpy()
next_token = int(np.argmax(logits[0, -1]))
```

Each component is saved under `components/<name>/` with `graph.cactus`,
`raw_ir.json`, and `optimized_ir.json`. For decode, expose a separate
`decoder_step` graph. The generic JAX path does not create or manage internal
KV cache yet; component output `Tensor`s can be passed directly into another
component. Use `loaded.reset()` between independent generations.

---

## Artifact Layout

A transpiled bundle looks like this:

```
weights/<model>/
  raw_ir.json              # IR before optimization (debugging)
  optimized_ir.json        # IR after fusion passes (debugging)
  graph.cactus             # serialized runtime graph
  graph_bindings.json      # weight binding metadata
  result.json              # execution results (if --execute-after-transpile)
  components/              # for multimodal models
    manifest.json          # component order, input/output names
    vision_encoder/
      graph.cactus
      bound_constants/
    audio_encoder/
      graph.cactus
    lm_encoder/
      graph.cactus
    decoder/
      graph.cactus
```

The `manifest.json` tells the runtime loader how to chain the components and where
to find the memory-mapped weight files. The raw/optimized IR JSONs are also valid
reload targets — the runtime can re-lower from saved IR as a fallback.

---

## CLI Reference

### `cactus convert`

```bash
cactus convert <model-id-or-path> [output-dir] [options]
```

Quantizes the weights to CQ and builds the runtime graph. Pass `--weights-only` to
stop after the CQ weights. The graph-build options below are the same ones the
transpiler accepts.

The decoder graph is emitted with a dynamic batch axis and a single baked KV-cache
slot, so single-stream decode stays lean while fixed-batch decode (`decode_batch` /
`generate_batch`) can size the KV-cache slot pool at runtime via
`Model::set_decode_slots(N)`. No separate conversion flag is needed.

| Option | Description |
|--------|-------------|
| `--bits 1\|2\|3\|4` | CQ quantization bits (default: 4) |
| `--weights-only` | Stop after CQ quantization; skip the runtime graph |
| `--weights-dir <path>` | Path to converted CQ weights (default: `weights/<model_name>`) |
| `--task <name>` | Force task type (default: `auto` — inferred from model config). Choices: `causal_lm_logits`, `multimodal_causal_lm_logits`, `ctc_logits`, `encoder_hidden_states`, `seq2seq_transcription`, `tdt_transcription` |
| `--prompt <text>` | Representative prompt for shape capture |
| `--image-file <path>` | Representative image (repeatable) |
| `--audio-file <path>` | Representative audio file |
| `--max-new-tokens <N>` | Generation room to preallocate for causal decode graphs |
| `--artifact-dir <path>` | Output directory (default: `weights/<model>`) |
| `--execute-after-transpile` | Run the graph after saving |
| `--skip-reference-compare` | Skip PyTorch vs transpiled comparison |
| `--component-pipeline auto\|on\|off` | Use split component graph transpilation when supported |
| `--components <list>` | Comma-separated component subset for component-pipeline models |
| `--trust-remote-code` | Allow HF remote code during the build |
| `--local-files-only` | Require HF files to already be local |
| `--allow-unconverted-weights` | Build against an unconverted source checkpoint |
| `--no-fuse-rms-norm` | Disable RMSNorm fusion |
| `--no-fuse-rope` | Disable RoPE fusion |
| `--no-fuse-attention` | Disable attention fusion |

NPU emission (CoreML `.mlpackage`s for Apple Silicon audio and vision encoders)
is available through the `--npu`, `--npu-quantize`, `--npu-audio-quantize`, and
`--npu-vision-quantize` flags on `cactus convert`.

### `cactus run`

```bash
cactus run <model-id-or-bundle-path> [options]
```

`model_id` may be a HuggingFace model id (downloads the matching bundle from
huggingface.co/Cactus-Compute) or a local path to a bundle directory.

| Option | Description |
|--------|-------------|
| `--bits 1\|2\|3\|4` | CQ quantization bits when downloading (default: 4) |
| `--weights general\|apple` | Weights bundle variant (default: general — portable on every platform) |
| `--token <token>` | HuggingFace token (gated models) |
| `--prompt <text>` | Input prompt |
| `--input-ids <ids>` | Comma-separated token IDs |
| `--image <path>` | Image file for multimodal bundles |
| `--audio <path>` | Audio file for speech / audio bundles |
| `--system <prompt>` | System prompt |
| `--thinking` | Enable thinking / reasoning mode |
| `--max-new-tokens <N>` | Max tokens to generate |
| `--result-json <path>` | Save bundle result payload to JSON |
| `--no-cloud-handoff` | Disable automatic cloud handoff |
| `--confidence-threshold <f>` | Confidence threshold below which to hand off to cloud |
| `--cloud-timeout-ms <N>` | Max wait time for cloud handoff |
| `--no-cloud-tele` | Disable cloud telemetry (write to cache only) |
| `--reconvert` | Force local rebuild from source |

---

## Supported Model Families

The transpiler works with any model that `torch.export` can capture, but has
tested adapters for:

| Family | Tasks | Component Split |
|--------|-------|-----------------|
| Gemma 3/4 | text, multimodal | yes (vision + audio + LM + decoder) |
| Qwen 3/3.5 | text, vision | no |
| LFM2/2.5 | text, vision | no |
| Whisper | encoder | no |
| Parakeet CTC | encoder | no |
| Parakeet TDT | transcription | yes (encoder + decoder) |

Adding a new model family means writing a small adapter in `model_adapters.py` that
wraps the HF forward into the canonical task interface. The rest of the pipeline
(capture, optimize, lower) is model-agnostic.

---

## Design Decisions

**ATen ops, not layer names.** After `torch.export`, the transpiler dispatches on
canonical ATen operation names (`aten.linear.default` -> `linear`). It never reads
HuggingFace module paths like `model.layers.0.self_attn`. This means the fusion
passes work on any model that produces the same computational pattern, regardless
of how the code is organized.

**Weight binding through manifests.** The transpiler resolves weights exclusively
through `weights_manifest.json` written by `cactus convert`. It does not guess
filenames from layer names. This is intentional — it keeps the contract between
conversion and transpilation explicit and auditable.

**Component splitting for multimodal.** Complex models like Gemma4 are split into
independent components (vision encoder, audio encoder, LM encoder, decoder) before
capture. Each component goes through the full optimize/lower pipeline independently.
This keeps each graph small enough for `torch.export` to handle and lets the runtime
load/unload components independently.

**Saved IR as fallback.** Bundles include both the serialized `.cactus` graph and
the raw/optimized IR JSON. If the binary graph format changes between versions, the
runtime can re-lower from saved IR instead of requiring a full re-transpile.

## See Also

- [Cactus Engine API](/docs/cactus_engine.md) — C API for inference
- [Cactus Graph API](/docs/cactus_graph.md) — Computation graph reference
- [Fine-tuning Guide](/docs/finetuning.md) — Train LoRA fine-tunes and deploy to mobile
- [Transpiler source](https://github.com/cactus-compute/cactus/tree/main/python/cactus/transpile) — Full implementation
