# NPU Transpiler Pipeline

Parallel to the graph transpiler: emits CoreML `.mlpackage`s so the runtime
engine can dispatch through the Apple Neural Engine. Triggered by
`cactus convert --npu`. Runtime side: `cactus-engine/src/model_npu.cpp`.

Three emit paths, matching what `origin/main` historically NPU-accelerated:

| Entry point | Emits | manifest key | Runtime dispatch |
|---|---|---|---|
| `run_prefill_pipeline` (`prefill.py`) | text-decoder prefill | `npu_prefill` | `prefill_via_npu` |
| `run_encoder_pipeline` → `audio.py` | audio encoder | `npu_audio_encoder` | `audio_encode_via_npu` |
| `run_encoder_pipeline` → `vision.py` | vision encoder | `npu_vision_encoder` | `vision_encode_via_npu` |

`run_encoder_pipeline` reuses the exact `ComponentModuleSpec` adapter
modules + example inputs the graph transpiler captures, so the NPU encoder
stays in lockstep with the CPU component graph. Auxiliary adapter inputs
(masks, position ids) are baked into the exported model as constants —
the ANE requires a static single-input signature.

## Status

| Target | Works | Notes |
|---|---|---|
| Parakeet TDT audio encoder | **yes — verified end-to-end** | `cactus transcribe` via NPU, ~7.5x faster warm than CPU with `CACTUS_ANE_AUDIO_COMPUTE_UNITS=ALL`. |
| `run_encoder_pipeline` emit mechanics | **yes — verified** | Synthetic 2-input (audio) and 3-input (vision) encoders export + convert + load + predict. |
| Gemma 4 audio + vision encoder emit | wired, **untested** | Can't convert Gemma 4 locally (OOM). Coworker must run `cactus convert --npu`; see handoff note below. |
| `coremltools.optimize.linear_quantize_weights` | works | `--npu-quantize 4` / `--npu-quantize 8` post-conversion. int8 recommended for encoders. |

## Gemma 4 coworker handoff

Gemma 4 conversion OOMs on a 16 GB host, so the actual emit must run on a
bigger machine. To produce + test the Gemma 4 NPU bundle:

1. `cactus convert --npu --npu-quantize 8 google/gemma-4-E2B-it` — emits
   `components/model.mlpackage` (prefill), `components/audio_encoder.mlpackage`,
   `components/vision_encoder.mlpackage`, and writes `npu_prefill` /
   `npu_audio_encoder` / `npu_vision_encoder` keys into `components/manifest.json`.
2. The runtime auto-loads any present mlpackage at init and dispatches
   automatically (`run_audio_encoder` / `run_vision_encoder` / prefill).
3. If `coremltools.convert` fails on an unsupported op, add a patch to
   `coremltools_patches.py` (see the layer_norm / `new_ones` / `__and__`
   patches already there for the pattern) and re-run.
4. Test: `cactus transcribe` for audio, a multimodal prompt for vision.
   Try `CACTUS_ANE_AUDIO_COMPUTE_UNITS=ALL` — ANE beat CPU+GPU for Parakeet.

## Known coremltools 9.0 gaps when converting HF Gemma 4 via `torch.export`

Each requires a custom `@register_torch_op` in this module:

1. `new_ones` — `tensor.new_ones(shape)`. Fix: `mb.fill(shape=size, value=1.0)`.
2. `__and__` — bitwise AND on bool tensors. Fix: `mb.logical_and(...)`.
3. Probably more once those two are fixed.

`torch.jit.trace` is not viable on current HF transformers — fails inside
`masking_utils.sdpa_mask` because `q_length` is an int but the code
indexes it as a tensor. `torch.export` is the correct path.

## Where the NPU perf gains actually come from

Surveyed `origin/main` for inspiration. Main has **CPU-side** kernel
fusion (e.g., `dense_mlp_int4_fused` from PR #617 in `cactus/models/
gemma4/model_gemma4.cpp` — gate+up+down INT4 matmul in one kernel call).
There is **no NPU-side fusion** code in main: the `.mlpackage` is
expected pre-built externally, and `cactus/npu/npu_ane.mm` is a thin
CoreML MLModel wrapper. So fused-op wins for NPU need to be added on
the convert path here.

The four NPU-fusion levers, ranked by expected impact:

1. **Weight quantization (int4 / int8)** — biggest single perf knob on
   ANE. Smaller weights ⇒ less memory bandwidth ⇒ faster matmuls.
   coremltools' `linear_quantize_weights` does post-conversion
   per-channel quantization. **Wired**: pass `--npu-quantize 4` (or 8)
   on `cactus convert`. Recommend int4 for big LMs.

2. **SDPA preservation** — `torch.nn.functional.scaled_dot_product_attention`
   maps to a single MIL op when coremltools sees it. If the model
   decomposes attention to Q·Kᵀ + softmax + ·V manually, coremltools
   has to recompose it, often losing the ANE fast path. HF defaults to
   `attn_implementation="sdpa"` for compatible models — we benefit
   automatically, but worth checking with `model.config._attn_implementation`
   after load.

3. **Stateful KV cache** — coremltools 9 supports `mb.state` for models
   that hold K/V on ANE between calls. Avoids transferring the entire
   cache through host memory per chunk. Requires:
   - CoreML model authored with `mb.state` ops (the prefill wrapper
     would need rewriting in MIL directly, not via PyTorch trace)
   - `npu_ane.mm` would need to use the iOS 18 `MLState` API instead of
     plain `MLPredictionOptions`
   Significant work. **Not done.**

4. **Pre-export module replacement** — swap HF's `LlamaAttention`,
   `LlamaMLP`, etc. for NPU-friendly versions before `torch.export`.
   E.g., fuse Q/K/V projections into a single matmul with stacked
   weights (saves 2 matmuls per layer). Per-architecture work. **Not done.**

## Architectural notes

- Wrapper takes `input_ids` (not `inputs_embeds`) because Gemma 4
  computes auxiliary per-layer embeddings from the original token IDs.
  Also more efficient (no host-side embedding lookup).
- Output naming matches `cactus-engine/src/npu_ane.mm` (`hidden`,
  `k_cache_0`, `v_cache_0`, ...) — change in lockstep with that file.
- Text decoder located via `_find_text_decoder`: tries
  `model.model.language_model` (multimodal HF), then `model.model`,
  then `model`.
- Dims via `_extract_dims` — tries `config.text_config` first
  (multimodal) then `config`.
