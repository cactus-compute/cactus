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

## What's left for end-to-end (M2 → M3 → M4)

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
