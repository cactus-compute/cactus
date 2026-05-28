# GPU Transpiler Pipeline

Emits per-model GPU bundles for Apple Metal. Triggered by `cactus convert --gpu`.
Runtime side: `cactus-engine/src/gpu/` (TBD as of this commit) +
`cactus-kernels-gpu/` (Metal kernel library).

**Scope: text decoder only.** Audio/vision encoders are on NPU (ANE) — see
`../npu/`. Tokenization stays on CPU.

## Status

| Component | Status |
|---|---|
| `pipeline.py` — top-level entry | ✅ skeleton |
| `plan.py` — emit `gpu_plan.json` from HF config | ✅ M1 (dims + per-layer op enumeration) |
| `weight_pack.py` — lay out GPU binary bundle | 🟡 M1 (offsets correct, zero-filled placeholders) |
| Metal kernels (`../../../cactus-kernels-gpu/`) | 🟡 M1 (matmul int4 mat-vec, RMSNorm work numerically; flash_attn / RoPE / SwiGLU compile) |
| C++ runtime GPU model (`cactus-engine/src/gpu/`) | ❌ M2 |
| End-to-end GPU prefill+decode | ❌ M3 |
| Benchmarks vs CPU / NPU / llama.cpp | ❌ M4 |

## Why a transpiler at all (vs. a runtime graph executor)?

llama.cpp's Metal backend is hand-written kernels + a dispatch table; cactus
already has a transpiler-emit-bundle pattern (graph transpiler, NPU transpiler).
We follow that pattern: convert-time emits a **per-model dispatch plan**
that the runtime just consumes. No graph interpretation at inference time.

This trades up-front compilation overhead (a few seconds at convert) for:

- Function constants baked per layer (head_dim, GQA factor)
- No per-token JSON walking or graph reconstruction
- Weights pre-packed in the exact layout the kernel reads
- Per-(K,N) kernel pipeline cached at load time

See `../../../docs/gpu/DESIGN.md` for the full architecture.

## Bundle layout

When `cactus convert --gpu` runs successfully, the bundle gains:

```
<bundle>/components/
  ├── manifest.json   (keys added: gpu_plan, gpu_weights, gpu_scales, gpu_embedding)
  └── gpu/
      ├── gpu_plan.json   — dispatch plan
      ├── weights.bin     — packed int4 weights + fp16 norms + fp16 LM head
      ├── scales.bin      — per-group fp16 scales for int4 layers
      └── embedding.bin   — fp16 input embedding table
```

## What this commit lands

A working scaffold:

1. **Branch + design doc** (`docs/gpu/DESIGN.md`): full architecture decisions
   informed by surveys of llama.cpp's Metal backend, MLX, and LiteRT.
2. **Metal kernel library** (`cactus-kernels-gpu/`): builds a `.metallib` and
   a dispatch static lib. Two kernels numerically verified against CPU
   reference (RMSNorm and INT4 matrix-vector, both with mean error < 5e-4).
   Other kernels compile but aren't yet end-to-end tested.
3. **Python transpiler skeleton** (this directory): emits a `gpu_plan.json`
   matching the HF model's architecture; weight binaries are correctly
   sized zero-filled placeholders.

What's deliberately deferred to M2+:

- Actual CQ4 quantization of HF weights → real `weights.bin` / `scales.bin`
  contents (not just byte counts).
- The C++ `GPUModel` runtime class that consumes the bundle.
- Tiled `mul_mm_int4_fp16` for prefill (the M1 placeholder is a per-row mat-vec).
- Per-shape kernel auto-tuning at load time.
- Flash attention prefill variant (M1 only has the decode path).
