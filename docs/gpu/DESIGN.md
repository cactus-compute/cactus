# Cactus GPU Transpiler & Runtime — Design

**Status:** draft, work in progress on `justin/v2-gpu` branch
**Author:** initial draft generated 2026-05-27 from llama.cpp + MLX + LiteRT survey
**Target:** Apple Silicon only (M-series Macs, A-series iPhones/iPads), Metal + MPSGraph
**Goal:** match or beat llama.cpp Metal backend on decode + prefill TPS, for INT4-quantized cactus LLMs
**Constraints:** **no regard for RAM usage or code complexity** — favor specialization, pre-computation, pre-dequant, and code generation over generality

---

## 1. What lives on GPU vs. CPU vs. NPU

| Workload | Device | Why |
|---|---|---|
| Tokenization (BPE / SentencePiece) | **CPU** | Sequential string ops, branchy. GPU launch overhead exceeds work. |
| Embedding lookup | **GPU** | Avoids the sync; embedding table is already on GPU. |
| Transformer block forward (matmul + attn + norm + RoPE + MLP) | **GPU** | This *is* the LLM. Bandwidth-bound; Apple Silicon has unified memory and `simdgroup_matrix` HW intrinsics. |
| KV cache R/W | **GPU resident, never copy back** | Copy-back is the #1 perf killer in naive GPU LLM impls. |
| Final LM-head matmul | **GPU** | Largest matmul in the network — needs the bandwidth. |
| Sampling (top-k / top-p / temperature) | **GPU** | Keeps the loop on-device; only the sampled token id crosses to CPU. |
| Audio / vision encoder | **NPU (ANE)** | Already done on `justin/v2-npu`. ANE is faster than GPU for encoders. |
| Mel-spectrogram / image preprocessing | **CPU** | Small ops, not bottleneck, not worth the sync. |

Net: **GPU owns the entire decoder hot loop. CPU owns I/O and control flow. NPU owns encoders.**

---

## 2. Architecture: transpile → bundle → load → dispatch

This mirrors the existing `npu/` pipeline:

```
HuggingFace model  ──┐
                     │  python/cactus/transpile/gpu/
                     ▼  (build per-model GPU bundle)
       weights/<model>/components/gpu/
            ├── decoder.gpu_plan.json   ← per-layer dispatch plan + function constants
            ├── weights.bin             ← packed weights in GPU-ready layout
            ├── scales.bin              ← per-group scales (fp16)
            └── decoder.metallib        ← optional pre-compiled kernel pack (else built at load)

                     │
                     ▼  cactus-engine/src/gpu/
                       GPUModel loads bundle, builds MTL pipelines,
                       sets function constants, dispatches per token
```

**Why this shape:**
- The transpiler can *see* the model topology (head_dim, num_kv_heads, layer count, MLP type). It bakes those into the dispatch plan at convert time, so the runtime never branches on them.
- Weights are pre-packed into the exact memory layout the kernel expects. No re-layout at load time.
- The `.gpu_plan.json` is small (~10 KB per model) and tells the runtime: "for layer 0, use `mul_mv_int4_d4096_k128` with these buffer offsets, then `flash_attn_dk128_dv128_h32_g8`, then ...".
- Kernels themselves are model-agnostic (just specialized by head_dim, MLP variant, etc.) and live in `cactus-kernels-gpu/`. The bundle picks which to call.

This is **the llama.cpp approach** (hand-written kernels + per-token dispatch plan), not the LiteRT approach (graph interpreter + JIT codegen). LiteRT's design carries TF Lite legacy overhead we don't need.

---

## 3. Memory model

### 3.1 Weights

- `MTLResourceStorageModeShared` everywhere. Apple Silicon has unified memory; this gives us zero-copy from mmapped weight files to GPU.
- **Pre-pack at convert time** into the exact layout the kernel wants. No runtime layout fixup.
- For INT4 weights using cactus's CQ4 (group size 128, hadamard or orthogonal rotation):
  - Packed nibbles laid out as `uint16[K/4 × N]` (4 nibbles per uint16, matches MLX/llama.cpp pattern for the in-kernel bitwise dequant).
  - Scales `fp16[K/128 × N]` in a separate buffer (group-by-group).
  - The rotation matrix is *not* stored — it's baked into the quantization at convert time and the kernel just sees the pre-rotated values.
- For fp16 weights (norms, embeddings, possibly LM head if memory allows): stored as-is.

### 3.2 KV cache

- One `MTLBuffer` per layer per direction (K, V), allocated at `init()` time at full context size.
- Storage mode: `Private` (GPU-only, never visible to CPU) — there's no reason to ever read these back.
- Layout: `[num_kv_heads, max_seq_len, head_dim]` (head-major). Lets the attention kernel walk the K/V cache with linear strides.
- Optional INT8 quantization of the KV cache (cactus already has CPU-side INT8 KV cache code) — for the GPU path we'll start fp16 and add INT8 once the fp16 path is solid.

### 3.3 Activations & scratch

- One persistent scratch arena per stream (Apple's `MTLHeap` or just a large `MTLBuffer` we sub-allocate into).
- Per-token activations live in this arena; no per-token allocator churn.
- Following MLX: `MTLResourceHazardTrackingModeUntracked` (we track dependencies ourselves via explicit `MTLBarrier`).

---

## 4. The kernel set

Drawn from the survey. All Metal kernels in `cactus-kernels-gpu/src/kernels/*.metal`, dispatch wrappers in `cactus-kernels-gpu/src/dispatch/*.mm`.

### 4.1 Matmul

Four variants, all 32-thread SIMD-group based:

| Kernel | Use case | Algorithm |
|---|---|---|
| `mul_mm_f16_f16` | prefill (M ≥ 4), fp16 weights | Steel-style: `simdgroup_matrix<half, 8, 8>` MMA, BM=64 BN=32 BK=32, threadgroup tiles, MLX-style padding. |
| `mul_mm_int4_f16` | prefill, INT4 CQ4 weights | Same MMA pattern, but loads + dequants INT4 nibbles into the simdgroup matrix fragment before MMA. |
| `mul_mv_f16_f16` | decode (M = 1), fp16 weights | Single-row vector-matrix, 1 thread per N row, simdgroup reduce across K. |
| `mul_mv_int4_f16` | decode, INT4 weights | llama.cpp-style: 4 register accumulators with pre-shifted x inputs, bitwise nibble unpack, `simd_sum` reduce. **The hot kernel.** |

Specialization: per-K, per-N alignment becomes function constants; tile sizes baked at template instantiation time; specific head_dims compiled as separate kernel names where needed.

### 4.2 Attention

Single Metal kernel, multiple instantiations:

```
flash_attn_dk{32,64,128,256}_dv{32,64,128,256}_kv{1,2,4,8}
```

(`kv` = num_kv_heads grouping factor, i.e. how many query heads share one KV head.)

- Single-pass online softmax (max/sum updated as we walk K-tile by K-tile).
- BQ = 64 (per Q tile), BK = 32 or 64 (per K tile), BD = head_dim.
- K, V cache laid out so each K-tile is a contiguous read.
- Function constants: `has_mask`, `do_causal`, `has_sinks`, `softcap_enabled`.
- For decode (Q tile = 1 token), a separate vectorized variant `flash_attn_vec_*` (matching llama.cpp's split).

### 4.3 Small fused ops

| Kernel | Notes |
|---|---|
| `rms_norm_fwd` | MLX-style: `simd_sum` for warp reduce, `threadgroup` only for final reduce, fused weight scaling. |
| `rope_apply_inplace` | Applies RoPE to Q and K in place; per-token. Function constants for `is_neox`, `is_imrope`, `theta_base`. |
| `swiglu_fwd` | Fused gate × silu(up). |
| `softmax_logits` | For sampling; computes softmax + cumulative for top-p. |
| `sample_argmax`, `sample_top_k_top_p` | Final sampling on GPU. |
| `embed_lookup` | Token id → embedding row, on GPU. |
| `cache_append_kv` | Writes new K/V into the cache at the right offset. |

### 4.4 Apple Silicon tricks we steal

From the survey:

1. **`simdgroup_matrix<T, 8, 8>` MMA** (MLX Steel): direct path to peak FMA throughput. Used in all prefill matmuls and the FA inner product.
2. **`simd_sum()` for warp reductions** (MLX RMS norm): 2 barriers instead of 4 for any softmax / norm / dot reduction.
3. **Threadgroup memory padding** (`pad = 16/sizeof(T)`): avoids bank conflicts on Apple's 32-bank SIMDs.
4. **Bit-packed INT4 dequant** (llama.cpp + MLX): the `(ws & 0x000f) * x[0] + (ws & 0x00f0) * x[1] + ...` pattern with pre-shifted inputs. **The single highest-leverage trick.**
5. **Persistent command queue** (MLX): one `MTLCommandQueue` per inference stream, reused. No command-buffer-per-eval allocator churn.

### 4.5 What we steal from LiteRT (and what we don't)

**Steal:**
- **Per-kernel auto-tuning at init time.** For each kernel × layer shape, enumerate 4–8 candidate threadgroup sizes at load time, time each on a warmup buffer, cache the winner per `(kernel_name, M, N, K)` in the `gpu_plan.json` so subsequent runs skip re-tuning. ~50 ms one-time cost, captures device-specific cliffs.
- **Macro-based precision codegen.** Define `FLT`, `FLT4`, `simd_sum_FLT()` as preprocessor macros that the Metal compiler substitutes per precision target. One `.metal` source, both fp16 and fp32 paths.

**Don't steal:**
- LiteRT's quantization-by-dequantize. We keep weights INT4 *in GPU memory* and dequant inline in the matmul — that's the entire bandwidth win.
- LiteRT's lack of a fused attention. We build flash attention from day one.
- LiteRT's generic `OperationDef` / `TensorDescriptor` layer designed to be portable across OpenGL/Vulkan/OpenCL/Metal. We're Metal-only and direct.
- FlatBuffers serialization. The cactus convert pipeline already writes JSON manifests; `gpu_plan.json` follows that pattern.

---

## 5. Dispatch model

### 5.1 At load time (`GPUModel::init()`)

1. Read `gpu_plan.json` from the bundle.
2. Compile `decoder.metallib` (or build the library from `.metal` source if not pre-compiled).
3. For each kernel referenced in the plan:
   - Set function constants per the plan.
   - Create `MTLComputePipelineState`.
   - Cache by `(kernel_name, function_const_hash)`.
4. Mmap weights, wrap as `MTLBuffer` with `newBufferWithBytesNoCopy:length:options:deallocator:`. (Shared mode, no copy.)
5. Allocate KV cache buffers (private mode) at full context size.
6. Allocate persistent activation scratch buffer (~50 MB at fp16, sized for the largest layer).

### 5.2 Per token (`GPUModel::decode_one()`)

1. Take a `MTLCommandBuffer` from a pool of 4 (pipelined like llama.cpp).
2. For each layer in the plan:
   - Encode the kernels listed (RMSNorm → QKV → RoPE → cache_append → FlashAttn → out_proj → residual_add → RMSNorm → MLP_up/gate → SwiGLU → MLP_down → residual_add).
   - Insert `MTLBarrier::Buffers` between any kernel pair where the output of one feeds the input of another.
3. Encode LM head matmul + sampling kernel.
4. `commit` the command buffer. Don't `waitUntilCompleted` here — start encoding the next token immediately.
5. Read back only the **sampled token id** (single int32) from the previous in-flight buffer when it completes.

### 5.3 Prefill (`GPUModel::prefill()`)

- Same plan, but with batch dim N > 1 (multiple tokens at once).
- Matmul variant switches from `mul_mv_*` (decode) to `mul_mm_*` (batched).
- Attention runs with Q tile size = min(prefill_chunk, 64).
- Chunked at the same 128-token chunks as the CPU path, to stay compatible with cactus's chunked prefill semantics.

---

## 6. Quantization plan

Stage 1 (this branch's MVP): support **CQ4 (hadamard rotation, INT4, group 128)** — the most common cactus weight format. fp16 weights work as-is.

Stage 2: add **CQ4 (orthogonal rotation, INT4, group 128)** — same kernel, the rotation is baked at convert time.

Stage 3: optional **fp16 LM head** when memory allows (LM head matmul is 30%+ of decode time; fp16 path is meaningfully faster than INT4 here because head_dim × vocab is huge).

Stage 4: **INT8 KV cache** (port the CPU path).

CQ4's hadamard/orthogonal rotation is a no-op for the kernel — the rotation is applied at quantization and inverted at the input X to the matmul. From the kernel's view, it just sees `int4_weight × float_input`. (This matches MLX's `bias` parameter pattern.)

---

## 7. Build & integration

### 7.1 CMake

New target: `cactus-kernels-gpu` (static lib). Conditional on `APPLE`. Links `Metal.framework`, `MetalKit.framework`, `MetalPerformanceShaders.framework` (for MPSGraph fallback paths if we need them).

`cactus-engine` gains a new compile-time flag `CACTUS_HAS_METAL` (default ON on Apple). When ON, links `cactus-kernels-gpu` and includes `cactus-engine/src/gpu/*`.

### 7.2 Python transpiler

New module: `python/cactus/transpile/gpu/`:

```
pipeline.py            — top-level entry: build_gpu_bundle(component_specs, artifact_dir)
weight_pack.py         — pack INT4 weights into Metal-ready layout
plan.py                — build the per-layer dispatch plan
kernel_select.py       — pick the right kernel variant per layer
manifest.py            — write components/manifest.json keys: gpu_plan, gpu_weights, gpu_metallib
README.md              — public-facing doc
```

CLI: `cactus convert --gpu` (parallel to `--npu`). `--gpu-quantize 4` for CQ4, etc.

---

## 8. Risk / what could go wrong

1. **CQ4 hadamard rotation vs. matmul.** The cactus matmul kernels (CPU side) apply the inverse rotation to the *input* X before matmul. The GPU kernel needs the same. If we get the rotation direction wrong, outputs look like noise. Mitigation: numerical check against the CPU path on every layer during early dev.

2. **simdgroup_matrix is iOS 14+ / Metal 2.3+.** Should be fine for all supported devices, but worth confirming the minimum iOS we target in the runtime spec.

3. **Per-kernel function constant explosion.** If we naively specialize on every (K, N, head_dim, num_kv_heads, mlp_type) combo we'll have hundreds of compiled kernel binaries. Strategy: function constants for what changes per layer within a model, kernel-name variants only for what's fixed per architecture (head_dim, num_kv_heads).

4. **MPSGraph as a fallback.** Some ops (e.g., specialized softmax with sinks) might be easier to express in MPSGraph than raw Metal. We allow MPSGraph as an escape hatch for ops where the Metal kernel isn't worth writing.

5. **Validation against llama.cpp.** We benchmark on the *same model* at the *same quant* — convert Gemma 4 to llama.cpp's q4_0 in parallel and run side by side. The cactus number has to beat (or come within 10% of) llama.cpp's Metal number, or the design is wrong.

---

## 9. Milestones

| Milestone | Deliverable |
|---|---|
| **M0** | Branch, dirs, design doc, kernel skeleton stubs that compile. |
| **M1** | INT4 matmul Metal kernel + dispatch + Python pack. Outputs match CPU within 1e-2 on a single linear layer. |
| **M2** | Add RMSNorm + RoPE + flash attention (single head_dim). Full single-layer forward matches CPU within 1e-2. |
| **M3** | Multi-layer decoder. End-to-end Gemma 4 E2B prefill+decode running on GPU. Match HF logits within 1e-2 at token 0; reasonable text by token 32. |
| **M4** | Benchmark vs. CPU + NPU + llama.cpp Metal. Reference sheet update. |

This commit sequence starts at M0 (skeleton + design doc), then M1 (INT4 matmul) on top, etc.

---

## 10. Open decisions (TODO)

- **Pre-dequant for prefill?** Decode is bandwidth-bound (INT4 wins). Prefill is more compute-bound — does it pay to pre-dequant fp16 once at prefill start so the matmul kernel is denser? Decide after M2 numbers.
- **MPSGraph for the LM head?** Apple's MPSGraph has a heavily-tuned matmul. For the very large LM head matmul (head_dim × vocab) it might beat our hand-written kernel. Bench-decide.
- **Multi-stream**: should we use multiple `MTLCommandQueue`s for parallel sequence decoding? Probably yes for batch>1, no for batch=1.
- **iPad Pro M-series**: same ANE/GPU split as Mac? Or different (e.g., smaller GPU tiles for thermal)?
