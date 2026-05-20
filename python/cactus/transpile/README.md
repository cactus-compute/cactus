# Cactus Transpiler

This package turns Hugging Face/PyTorch models into Cactus component bundles.
The public workflow is:

```bash
cactus convert <hf-model-id> [weights/output-dir] --bits 4
cactus run <weights/output-dir>
```

`cactus convert` does two things in one folder:

1. Converts Hugging Face weights into Cactus CQ tensor files.
2. Captures the PyTorch graph, lowers it into Cactus `.cactus` component graphs,
   and writes `components/manifest.json`.

After conversion, `cactus run` auto-detects `components/manifest.json` and runs
the transpiled graph bundle. Users should not need a separate model-specific C++
implementation for supported transpiled bundles.

## Design Contract

- The graph compiler maps exported PyTorch/ATen operations and tensor flow into
  Cactus graph ops. It should not depend on Hugging Face layer names to decide how
  compute ops lower.
- Converted CQ weights are the source of truth for learned parameters. Runtime
  execution should use `.cactus` graphs plus Cactus `.weights` files, not NumPy
  sidecars.
- Small deterministic constants can be embedded into `.cactus` graphs or repaired
  from config at load time. Learned weights remain external CQ/Cactus tensors.
- Model-family profiles are allowed for prompt style, component boundaries,
  supported input combinations, weight aliases, and performance fusions.
- Changes to `cactus-engine`, `cactus-graph`, and `cactus-kernels` should stay
  minimal and general.

## Quick Start

From the repo root:

```bash
source ./setup
cactus build
```

The examples below use `--local-files-only` to keep tests RAM/network predictable
once the Hugging Face snapshot is already cached. Remove that flag the first time
you download a model, or pass `--token` for gated/private models.

Convert and run Gemma4 multimodal:

```bash
cactus convert google/gemma-4-E2B-it weights/gemma-4 \
  --bits 4 \
  --local-files-only \
  --trust-remote-code \
  --max-new-tokens 512

cactus run weights/gemma-4
```

Run with media:

```bash
cactus run weights/gemma-4 \
  --image assets/app.png \
  --prompt "Describe this image in one sentence."

cactus run weights/gemma-4 \
  --audio assets/test.wav \
  --prompt "Transcribe this audio."
```

Convert and run Parakeet TDT:

```bash
cactus convert nvidia/parakeet-tdt-0.6b-v3 weights/parakeet-tdt \
  --bits 4 \
  --local-files-only

cactus run weights/parakeet-tdt --audio assets/test.wav
```

Convert and run Whisper:

```bash
cactus convert openai/whisper-small weights/whisper-small \
  --bits 4 \
  --local-files-only

cactus run weights/whisper-small --audio assets/test.wav
```

Convert and run LFM2-VL:

```bash
cactus convert LiquidAI/LFM2-VL-450M weights/lfm2-vl-450m \
  --bits 4 \
  --local-files-only \
  --trust-remote-code

cactus run weights/lfm2-vl-450m --prompt "Hello"
cactus run weights/lfm2-vl-450m \
  --image assets/app.png \
  --prompt "Describe this image in one sentence."
```

Convert and run Qwen:

```bash
cactus convert Qwen/Qwen3.5-0.8B weights/qwen3.5-0.8b \
  --bits 4 \
  --local-files-only \
  --trust-remote-code

cactus run weights/qwen3.5-0.8b --prompt "Hello"
cactus run weights/qwen3.5-0.8b \
  --image assets/app.png \
  --prompt "Describe this image in one sentence."
```

## Supported Profiles

The current maintained profiles live in `model_profiles.py`.

| Family | Example model | Inputs | Default task | Key components |
| --- | --- | --- | --- | --- |
| Gemma4 | `google/gemma-4-E2B-it` | text, image, audio | `multimodal_causal_lm_logits` | `vision_encoder`, `audio_encoder`, `lm_encoder`, cached decoder |
| LFM2-VL | `LiquidAI/LFM2-VL-450M` | text, image | `multimodal_causal_lm_logits` | `vision_encoder`, `lm_encoder`, cached decoder |
| Qwen | `Qwen/Qwen3.5-0.8B` | text, image | `multimodal_causal_lm_logits` | `vision_encoder`, `lm_encoder`, cached decoder |
| Whisper | `openai/whisper-small` | audio | `seq2seq_transcription` | `audio_encoder`, cached decoder |
| Parakeet TDT | `nvidia/parakeet-tdt-0.6b-v3` | audio | `tdt_transcription` | `audio_encoder`, TDT decoder |

The generic planner in `component_plan.py` also detects common causal LM, CTC,
Whisper, TDT, and multimodal config shapes. New models should first try the
generic path. Add a profile only when the model needs prompt formatting,
component-boundary, input-combination, weight-alias, or decode-cache details that
cannot be inferred safely.

## What `cactus convert` Writes

A converted/transpiled folder looks like this:

```text
weights/<model>/
  config.json
  config.txt
  tokenizer files...
  weights_manifest.json
  *.weights
  raw_ir.json
  optimized_ir.json
  raw_ir_<component>.json
  optimized_ir_<component>.json
  components/
    manifest.json
    vision_encoder/
      graph.cactus
      raw_ir.json
      optimized_ir.json
    audio_encoder/
      graph.cactus
      raw_ir.json
      optimized_ir.json
    lm_encoder/
      graph.cactus
      raw_ir.json
      optimized_ir.json
    decoder_step/
      graph.cactus
      raw_ir.json
      optimized_ir.json
```

Not every model has every component. `components/manifest.json` is the runtime
contract. It records component order, logical input/output names, runtime node
ids, cache-state node ids, and external weight bindings.

## Runtime Behavior

`cactus run <folder>` resolves the folder as a transpiled bundle when it contains
`components/manifest.json`. It then delegates to `component_bundle_runtime.py`.

The runtime:

1. Builds or locates the Python FFI shared runtime.
2. Loads each needed `.cactus` component graph.
3. Rebinds external CQ/Cactus weight tensors from `weights_manifest.json`.
4. Preprocesses text, images, and audio for the active input combination.
5. Executes encoder components, then cached decode components when available.

Interactive `cactus run` supports:

```text
/image <path> [prompt]
/audio <path> [prompt]
/clear
reset
exit
```

Text-only turns, image turns, audio turns, and image+audio turns should use the
same component bundle when the model profile supports those combinations. We do
not create separate entrypoint folders for each media combination.

## Media Shape Policy

Transpiled graphs are static-shape graphs, so conversion needs representative
media shapes. The CLI supplies bundled tiny assets when a profile requires media
and the user did not pass representative files.

Runtime media is normalized to bounded static shapes:

- Images are padded/resized to `256x256` by default.
- Audio is capped at `30` seconds by default.

Override these only when rebuilding a bundle with compatible shapes:

```bash
CACTUS_TRANSPILER_IMAGE_SIZE=384 cactus convert ...
CACTUS_TRANSPILER_MAX_AUDIO_SECONDS=15 cactus convert ...
```

If a prompt or media turn exceeds the bundle's static token capacity, rerun
`cactus convert` with a larger representative prompt and `--max-new-tokens`.

## Compiler Pipeline

The compiler path is:

```text
HF/PyTorch model
  -> model adapter / component specs
  -> torch.export
  -> exported ATen graph
  -> IRGraph import
  -> canonical cleanup
  -> topology and op-pattern fusions
  -> Cactus Graph lowering
  -> .cactus component bundle
```

Important files:

| File | Role |
| --- | --- |
| `component_plan.py` | Infers task and component plan from converted config. |
| `model_profiles.py` | Small family profiles for aliases, prompt style, components, and cached decode routes. |
| `hf_model.py` | Main transpile entrypoint used by `cactus convert`. |
| `capture_pytorch.py` | Owns the `torch.export` boundary. |
| `aten_ops.py` | Normalizes exported ATen op names. |
| `import_ir.py` | Builds `IRGraph` from the exported FX graph. |
| `importers.py` | Maps canonical ATen ops into Cactus IR nodes. |
| `import_semantics.py` | Early semantic rewrites for attention, linear, and RoPE patterns. |
| `canonicalize/cleanup.py` | Simplifies, folds constants, legalizes precision, and removes dead code. |
| `optimize_graph.py` | Runs fusions such as RMSNorm, RoPE, attention, MLP, convolution, and DeltaNet. |
| `lower.py` | Lowers optimized IR into Cactus `Graph` ops and binds CQ weights. |
| `component_bundle_runtime.py` | Loads and executes saved component bundles. |
| `weight_binding.py` | Resolves IR constants to converted Cactus weight files. |
| `weight_compat.py` | Builds runtime-compatible companion tensors when the current kernels need them. |
| `audio_preprocess.py` | Shared audio loading, duration limiting, and feature extraction helpers. |
| `multimodal_runtime.py` | Shared prompt/media preparation for multimodal bundles. |
| `tdt_runtime.py` | Parakeet TDT audio features and greedy decode loop. |
| `runtime_compat.py` | Python-side compatibility shims for available Cactus graph symbols. |

## Component Splitting

Component splitting is a runtime and memory tool, not a separate model
implementation. A profile or generic plan decides the component names, then each
component is still captured and lowered from PyTorch ops.

Common component names:

| Component | Meaning |
| --- | --- |
| `vision_encoder` | Converts image tensors into visual token embeddings/features. |
| `audio_encoder` | Converts audio features into audio token embeddings/features or encoder states. |
| `lm_encoder` | Merges text/media inputs into decoder-ready embeddings for prefill. |
| `lm_encoder_step` | One-token embedding path for cached autoregressive decode. |
| `decoder_prefill_chunk` | Chunked prompt priming for KV cache. |
| `decoder_step` | One-token cached autoregressive decoder. |
| `decoder_media_step` | Media-aware one-token decoder variant for Qwen-style bundles. |
| `decoder` | Full-context decoder fallback or task-specific decoder. |

The decoder is the autoregressive text stack. `lm_encoder` is the prefill/merge
stage that prepares decoder inputs from tokens and optional media features.

## Weight Binding

Converted CQ weights are recorded in `weights_manifest.json`. During import,
captured constants are annotated with source names. During lowering, matching
constants become mmap-backed Cactus weight tensors instead of embedded payloads.

Expected behavior:

- Learned parameters come from Cactus `.weights` files.
- CQ quantization stays in the Cactus weight/kernel path.
- No `.npy` files are required for normal execution.
- Missing deterministic constants are repaired from config only when they are not
  learned parameters, such as RoPE inverse-frequency buffers.

If a model has equivalent weights under unusual names, add the smallest possible
alias in `model_profiles.py`. Do not add model-specific compute lowering for a
standard PyTorch op.

## Debugging

For normal users, prefer:

```bash
cactus convert <model> weights/<name> --bits 4
cactus run weights/<name>
```

For compiler debugging, call the transpiler module directly:

```bash
PYTHONPATH=python ./venv/bin/python -m cactus.transpile.hf_model \
  --model-id <hf-model-id> \
  --weights-dir weights/<name> \
  --artifact-dir /tmp/<bundle> \
  --task causal_lm_logits \
  --prompt "Hello" \
  --skip-execute
```

Useful environment variables:

| Variable | Purpose |
| --- | --- |
| `CACTUS_TRANSPILER_IMAGE_SIZE` | Static image resize/pad size, default `256`. |
| `CACTUS_TRANSPILER_MAX_AUDIO_SECONDS` | Static audio duration cap, default `30`. |
| `CACTUS_TRANSPILER_DISABLE_CACHED_STEP_DECODE=1` | Force full-context decode fallback for debugging. |
| `CACTUS_TRANSPILER_DISABLE_CHUNK_PREFILL=1` | Disable chunked KV prefill. |
| `CACTUS_TRANSPILER_PREFER_SAVED_GRAPH=0` | Rebuild components from saved IR instead of loading `.cactus`. |
| `CACTUS_TRANSPILER_DEBUG_MMAP=1` | Print constant/weight binding details during lowering. |
| `CACTUS_KV_CACHE_FP16=1` | Use FP16 KV cache where supported. |

## Adding A New Model

1. Run `cactus convert <model> weights/<name> --bits 4`.
2. Check `weights/<name>/components/manifest.json`.
3. Run `cactus run weights/<name>` with a short prompt or small media sample.
4. If task/component detection is wrong, update `component_plan.py` or add a
   compact profile in `model_profiles.py`.
5. If op lowering fails, add or fix the ATen op importer/lowering/fusion.
6. If output is slow but correct, add a general fusion or cached-step component.
7. If weights do not bind, add aliases in `model_profiles.py` or fix converter
   naming so the manifest records the expected source name.

Keep model-specific code small and data-shaped. Prefer profile entries and
general fusions over new per-model runtime files.

## Known Practical Notes

- Gemma4 multimodal has a heavier first-token path because media towers and cache
  priming run before decode. Once primed, cached decode should be close to native
  autoregressive speed.
- Parakeet and Whisper are audio-first bundles. Pass audio with `--audio` through
  `cactus run`.
- LFM2-VL and Qwen support text-only and image+text from the same bundle.
- For RAM-limited machines, use `--local-files-only` after the model is cached,
  avoid FP32 experiments, and test one model at a time.
