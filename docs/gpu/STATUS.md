# GPU Branch (`justin/v2-gpu`) — Status & Remaining Work

This file describes the state of the GPU transpiler + runtime on this branch.
For the architectural rationale, see `DESIGN.md` in this directory.

## What landed (M0 → M1)

| Area | Status | Notes |
|---|---|---|
| Branch off `justin/v2-npu` | ✅ | |
| Research surveys (llama.cpp, MLX, LiteRT) | ✅ | See `DESIGN.md` for the synthesis. Repos cloned under `/Users/justinl/Desktop/GitRepos/{llama.cpp,mlx,LiteRT}` (sibling to cactus). |
| Design doc | ✅ | `docs/gpu/DESIGN.md` |
| `cactus-kernels-gpu/` Metal library | ✅ | Builds a `.metallib` + a dispatch static lib. Compiles cleanly. |
| Metal kernels (compile) | ✅ | `matmul_int4`, `rms_norm`, `rope`, `swiglu`, `embed_lookup`, `kv_cache_append`, `sample_argmax`, `flash_attn` (per-head-dim 64/128/256) |
| Numerical correctness vs CPU | ✅ for RMSNorm + mat-vec INT4 | mean error ≈ 3.5e-4, max ≈ 2e-3 — within fp16+int4 tolerance |
| Python `cactus.transpile.gpu` | ✅ skeleton | Emits `gpu_plan.json` with per-layer op enumeration; weight binaries are correctly-sized zero placeholders. End-to-end emit verified on real Gemma 4 E2B config. |
| `cactus-engine` C++ `GPUModel` | ✅ skeleton | Plan parse, mmap+wrap, buffer alloc. Forward pass stubs return BOS token with a "not implemented (M3)" log. |
| CMake integration | ✅ | `cactus-engine` includes `cactus-kernels-gpu` on Apple. Top-level `libcactus.a` bundles the GPU dispatch symbols. |
| CLI plumbing (`--gpu`, `--gpu-quantize`) | ✅ | Added to the convert flow alongside `--npu`. |
| Existing tests (Parakeet ASR, LFM-VL chat) | ✅ still pass | Verified no regression after GPU integration. |

## What landed in the M2/M3 push (this session)

| Area | Status |
|---|---|
| `weight_pack.py` real q4_0 quantization (group=128 symmetric) | ✅ |
| Bundle emit on real Gemma 4 → 1.7 GB weights + 28.7 MB scales + 805 MB embedding | ✅ |
| q_proj layer 0 round-trip dequant cos_sim vs HF | **0.993** ✅ |
| Embedding round-trip vs HF | **bit-exact** ✅ |
| Plan parser in C++ (picojson) | ✅ |
| Pipeline cache + per-shape Metal pipelines built lazily on first use | ✅ |
| `decode_one` plan walker — dispatches embed → (rms_norm → q/k/v → rope → kv_append → flash_attn → out_proj → residual → rms_norm → gate/up → swiglu → down → residual)×N → final_norm → lm_head → argmax | ✅ |
| `residual_add` + `mul_mv_fp16` Metal kernels | ✅ |
| `command_buffer_destroy` public API | ✅ |
| End-to-end smoke test (`test_gpu_decode`): loads Gemma 4 bundle, runs `decode_one`×2 without crashing, KV cache advances | ✅ |
| Numerical correctness vs HF on a real model | ❌ (see "Known gap" below) |

### Forward pass produces real output (M3 fully wired)

After fixing three subtle bugs:

1. **Buffer lifetime** — `command_buffer_begin` uses `commandBufferWithUnretainedReferences` (MLX optimization, skips Metal's hazard tracking). This requires every bound `MTLBuffer` to stay alive until the command buffer completes. The walker was calling `buffer_destroy` mid-encode for weight wrappers + small param buffers, so the GPU was reading freed memory by the time it executed. Fix: collect all ephemerals into a `std::vector<Buffer*>` and free after `command_buffer_wait`. **This was the killer bug.**
2. **RoPE binding** — kernel reads `num_q_heads` / `num_kv_heads` via `[[buffer(3)]]` / `[[buffer(4)]]`; dispatch was only binding 3 buffers. Fixed by allocating two 4-byte SHARED buffers per call.
3. **Empirical attention_kind detection** — Gemma 4's `cfg.layer_types` is misleading. Some "full_attention"-tagged layers (19, 24, 29, 34) also lack their own k_proj. Switched plan generator to ignore the config tag and just check `module.self_attn.k_proj.weight` empirically.

**Result on Gemma 4 E2B**, `decode_one(BOS=2)`:
- residual / norm_out / logits all contain real fp16 values (not zeros)
- Argmax returns token id **8227** (` Miss`); HF returns **236761** (`.`)
- Output is incorrect but the entire 35-layer × 14-op-per-layer dispatch path executes cleanly

### Followup pass: Gemma 4 sliding-attention KV sharing (partial)

Encoded `attention_kind` ("full"/"sliding") and `kv_source_layer` per layer in the plan. `plan.py` now skips k_proj/v_proj op emission for sliding layers; the C++ walker routes `flash_attn` to the source layer's K/V cache. After re-emit, the "missing tags" warning dropped from 20 each → 4 each (the first 4 sliding layers preceding the first full-attention layer at index 4 — these are an unhandled edge case; Gemma 4 may apply special init or actually have them with full KV).

Re-tested: `decode_one(2)` still returns 0. Root causes still in play, in order of likelihood:

1. **`mul_mv_fp16` (LM head kernel) is untested.** It's the simplest of the kernels but has zero numerical validation against a CPU reference. If it produces garbage, the sampler gets uninitialized memory and `simd_max` over all-zeros returns 0.
2. **RoPE dispatch binding is incomplete.** I bind 3 buffers (q, k, position_ids) but the kernel signature also references `num_q_heads` / `num_kv_heads` via additional buffers — those aren't bound in the dispatch wrapper. They'd read garbage.
3. **First 4 layers of Gemma 4** have no `full_attention` predecessor; my plan generator emits q-only (sliding) for them which means their attention has no K/V cache to read from. Should likely fall back to "full" if the layer module has k_proj.
4. **q_proj on sliding layers reads from norm_out** which was computed correctly, BUT the output `q_buf` gets reused next layer without being cleared — but the kernel writes every element so this should be fine.

### Known gap: numerical correctness needs kernel-level debugging

Gemma 4's `Gemma4TextDecoderLayer` interleaves two attention flavors:

- 7 `full_attention` layers (every 5th: 4, 9, 14, 19, 24, 29, 34) — these have q, k, v, o projections.
- 28 `sliding_attention` layers — these only have q + o; they **reuse K/V from the nearest preceding `full_attention` layer.**

Our plan generator emits q/k/v/o ops for *every* layer, but `weight_pack.py` correctly skips the missing k_proj/v_proj on sliding layers (with a console warning). The C++ `decode_one` walker then **also** skips those ops at runtime — but because the K/V buffers (`k_buf`, `v_buf`) hold stale values from the previous layer, the `kv_cache_append` + `flash_attn` for sliding layers read garbage.

**This is why `decode_one(2)` currently returns token 0** on Gemma 4 — partially-correct residual stream propagates through, but with garbage attention from sliding layers, the final logits collapse.

Fix lives in:
- `plan.py`: encode `attention_kind: "full"|"sliding"` per layer + `kv_source_layer: int` for sliding ones.
- `gpu_model.mm`: when an op is on a sliding layer, route `kv_cache_append` + `flash_attn` to the K/V buffers of `kv_source_layer` instead of computing fresh.

Estimated work: ~100 lines in plan.py, ~50 in gpu_model.mm.

## What's left (M3 polish → M4)

| Area | Owner | Notes |
|---|---|---|
| **M2: CQ4 quantization in `weight_pack.py`** | Python | Hook into `cactus.convert.quantization.cq` to actually quantize HF Linear weights into the kernel's nibble + scale layout. Today: zero placeholders. |
| **M2: per-shape kernel auto-tuning at load** | C++ | Per LiteRT pattern: try 4-8 threadgroup sizes per `(K, N)`, time each, cache winner in `gpu_plan.json`. |
| **M2: tiled `mul_mm_int4_fp16` for prefill** | Metal | Current implementation falls back to per-row mat-vec. Need `simdgroup_matrix<half,8,8>` MMA tiles (MLX Steel pattern). |
| **M3: `GPUModel::prefill_and_sample` / `decode_one`** | C++ | Walk the parsed plan, encode kernel dispatches into command buffers, pipeline ~4 deep, sample on GPU. |
| **M3: flash attention prefill variant** | Metal | The current `flash_attn_dk*_dv*` instantiations are decode-only (Q tile = 1). Need the Q-tile-of-64 prefill version. |
| **M3: routing in `cactus-engine/src/model.cpp`** | C++ | When the bundle has `gpu_plan` in manifest, prefer `GPUModel` over the CPU graph executor. Today the GPU model exists but no dispatch site calls it. |
| **M4: benchmarks** | — | Decode TPS + prefill TPS on Gemma 4 E2B / LFM2 / Qwen at M-series. Side-by-side with `llama.cpp -m gemma-4-e2b.gguf` running its Metal backend on the same hardware. |

## How to verify M1 locally

```bash
# Build the Metal kernel library + dispatch lib
cd cactus-kernels-gpu
mkdir -p build && cd build
cmake -DCACTUS_KERNELS_GPU_TESTS=ON ..
cmake --build . -j

# Run the kernel correctness tests
./tests/test_rms_norm        cactus-kernels-gpu/cactus_kernels.metallib
./tests/test_matmul_int4     cactus-kernels-gpu/cactus_kernels.metallib
./tests/test_matmul_int4_simple cactus-kernels-gpu/cactus_kernels.metallib
```

Expected: RMSNorm + matmul both report `mean_err ≈ 3-4e-4`, exit 0.

```bash
# Build the integrated cactus library (with the GPU module linked in)
cd cactus
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
# → builds libcactus.a + libcactus.dylib with cactus-kernels-gpu symbols included.
```

```bash
# Exercise the Python transpiler skeleton against a real HF config
cd cactus
python - <<'EOF'
import sys
sys.path.insert(0, 'python')
from pathlib import Path
import tempfile, json
from transformers import AutoConfig

class _Mock:
    def __init__(self, cfg): self.config = cfg
    @property
    def __class__(self): return type('Gemma4ForCausalLM', (), {'__name__': 'Gemma4ForCausalLM'})

m = _Mock(AutoConfig.from_pretrained('google/gemma-4-E2B-it'))
from cactus.transpile.gpu.pipeline import run_gpu_pipeline
with tempfile.TemporaryDirectory() as td:
    print(run_gpu_pipeline(m, Path(td), enabled=True, quantize_bits=4))
EOF
```

Expected: emit a `gpu_plan.json` with 35 layers × 15 ops/layer, plus correctly-sized
zero-filled `weights.bin` (~1.4 GB), `scales.bin` (~19 MB), `embedding.bin` (~805 MB).

## Files added (33 total)

```
docs/gpu/
  DESIGN.md
  STATUS.md                                   ← this file

cactus-kernels-gpu/
  CMakeLists.txt
  include/cactus_gpu.h                        ← public C++ API
  src/dispatch/
    context.mm                                ← MTLDevice / MTLCommandQueue
    buffer.mm                                 ← MTLBuffer wrap/create
    pipeline.mm                               ← MTLComputePipelineState build
    pipeline_factory.mm                       ← per-op factories
    command_buffer.mm                         ← dispatch encoding
    internal.h                                ← cross-TU accessors
  src/kernels/
    cactus_kernels.metal                      ← include-aggregator master
    common.metal                              ← simd_sum reductions, CQ4 nibble dot
    matmul_int4.metal                         ← mat-vec + (placeholder) mat-mat
    rms_norm.metal
    rope.metal
    swiglu.metal
    embed_and_kv.metal
    sample.metal
    flash_attn.metal                          ← decode-only, dk{64,128,256}
  tests/
    CMakeLists.txt
    test_rms_norm.mm                          ← passes, mean_err 3.5e-4
    test_matmul_int4.mm                       ← passes, mean_err 3.5e-4
    test_matmul_int4_simple.mm                ← passes, y = [128,128,128,128]

cactus-engine/src/gpu/
  gpu_model.h                                 ← public class header
  gpu_model.mm                                ← M2/M3 skeleton

python/cactus/transpile/gpu/
  __init__.py
  pipeline.py                                 ← top-level run_gpu_pipeline()
  plan.py                                     ← GPUPlan from HF config
  weight_pack.py                              ← byte layout (M1: zero placeholders)
  README.md
```

## Files modified (6 total)

```
cactus/CMakeLists.txt                         ← bundle cactus-kernels-gpu into libcactus.a
cactus-engine/CMakeLists.txt                  ← add_subdirectory + gpu_model.mm
python/cactus/cli/__init__.py                 ← --gpu / --gpu-quantize flags
python/cactus/cli/convert.py                  ← pass gpu / gpu_quantize through
python/cactus/cli/model.py                    ← TranspileOptions + run_transpile args
python/cactus/transpile/hf_model.py           ← invoke run_gpu_pipeline + manifest merge
```

## Surveys

Cloned (kept as sibling repos, NOT in cactus repo):
- `~/Desktop/GitRepos/llama.cpp` (184 MB, shallow)
- `~/Desktop/GitRepos/mlx` (16 MB, shallow)
- `~/Desktop/GitRepos/LiteRT` (10 MB, sparse-checkout `tflite/delegates/gpu/`)

Each was surveyed by a dedicated agent; findings synthesized in `DESIGN.md`.
