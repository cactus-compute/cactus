---
title: "Cactus Quants (CQ)"
description: "Rotation-and-codebook post-training quantization for on-device multimodal models. One recipe for every weight tensor — linears, embeddings, vision towers, audio towers — from 4-bit down to 1-bit."
keywords: ["quantization", "PTQ", "CQ", "Cactus Quants", "on-device AI", "mobile inference", "Hadamard", "codebook", "GPTQ", "AWQ", "HQQ"]
---

# Cactus Quants

Cactus Quants (CQ) is the post-training quantization system used by Cactus to compress
models for on-device deployment. It applies a single rotation-and-codebook recipe to
every weight tensor in a multimodal model — transformer linears, vision encoders, audio
encoders, cross-modal bridges, per-layer embedding tables, and the shared token embedding
— at bit widths from 4-bit down to 1-bit.

## Why Another Quantization Method?

A 2B-parameter model in FP16 is roughly 4 GB on disk. At 4-bit it's 1 GB. At 2-bit
it's 500 MB. Edge deployment is a storage and memory problem.

Existing PTQ methods (GPTQ, AWQ, HQQ) work well at 4 bits on transformer linear layers.
But they were not designed for the full multimodal setting:

- **They only quantize linears.** Per-layer embedding tables, modality towers, and
  vision-language bridges are left at FP16. In models like Gemma 4 E2B, per-layer
  embeddings alone are 60% of the model — more than half the storage is untouched.

- **They break below 4 bits.** At 3-bit, accuracy drops 15-30 points on hard benchmarks.
  At 2-bit, every baseline collapses to near-random on generation tasks.

- **They don't handle embeddings.** Embedding tables are gathered by token ID, not
  multiplied against activations. Methods that depend on an activation Hessian
  (GPTQ, AWQ) have no signal to work with.

CQ solves all three problems with one unified recipe.

## Results

We evaluated CQ against GPTQ, AWQ, and HQQ on three sub-2B multimodal model families:
**Gemma 4 E2B**, **Qwen3-VL-2B**, and **LFM-2.5-VL-1.6B**. All baselines use group
size 128 where applicable. Results averaged over 3 seeds.

### At 4 Bits: Competitive with the Best Baseline

At 4-bit, the field is closely contested. CQ leads or ties on most metrics across
Gemma 4 and Qwen3-VL-2B:

| | CQ-4bit | GPTQ-4bit | AWQ-4bit | HQQ-4bit | FP16 |
|---|---|---|---|---|---|
| MMLU (Gemma 4) | 59.45 | 59.48 | **59.66** | 57.23 | 62.33 |
| GSM8K (Gemma 4) | **71.20** | 68.33 | 70.53 | 58.87 | 73.67 |
| HumanEval (Gemma 4) | **57.11** | 50.61 | 52.64 | 43.29 | 54.88 |
| ARC-E (Gemma 4) | **73.73** | 72.00 | 73.00 | 72.00 | 73.80 |

On LFM-2.5-VL-1.6B, HQQ leads at 4-bit on several metrics — CQ's advantage doesn't
appear until bits get tighter.

### At 3 Bits: CQ Dominates

This is where the gap opens. At 3-bit, CQ sweeps all 8 text metrics on Gemma 4, with
margins of up to **+32 points** on the hardest benchmarks:

| | CQ-3bit | GPTQ-3bit | AWQ-3bit | HQQ-3bit |
|---|---|---|---|---|
| GSM8K (Gemma 4) | **52.67** | 16.40 | 21.87 | 1.13 |
| HumanEval (Gemma 4) | **45.12** | 8.54 | 13.01 | 4.88 |
| MMLU (Gemma 4) | **54.56** | 49.26 | 49.22 | 36.02 |
| BFCL Parallel (Gemma 4) | **81.17** | 41.00 | 58.00 | 0.50 |

The same pattern holds on Qwen3-VL-2B (CQ leads 5 of 7 text metrics, up to +30 points
on GSM8K) and LFM-2.5-VL-1.6B (CQ leads 5 of 8 metrics).

### At 2 Bits: CQ Is the Only Survivor

At 2-bit, every baseline collapses. CQ is the only method retaining usable accuracy:

| | CQ-2bit | GPTQ-2bit | AWQ-2bit | HQQ-2bit |
|---|---|---|---|---|
| ARC-E (Gemma 4) | **50.80** | 24.20 | 27.40 | 23.00 |
| MMLU (Gemma 4) | **33.18** | 23.40 | 24.28 | 25.06 |
| BFCL Simple (Gemma 4) | **18.75** | 0.00 | 0.00 | 0.00 |
| BFCL Multi (Gemma 4) | **13.67** | 0.00 | 0.00 | 0.00 |

On Qwen3-VL-2B at 2-bit, CQ retains 50.5 ARC-E and 29.4 MMLU while baselines are
at near-random. On vision, CQ holds 27.9 MMMU and 31.2 AI2D where the next-best
method scores below 7.

### Function Calling

Tool calling requires longer-horizon coherence than multiple-choice reasoning, making
it the hardest task to preserve under quantization:

| | CQ | GPTQ | AWQ | HQQ |
|---|---|---|---|---|
| 4-bit BFCL Parallel-Multi (Gemma 4) | **83.33** | 80.33 | 80.50 | 81.00 |
| 3-bit BFCL Parallel-Multi (Gemma 4) | **79.67** | 38.83 | 31.33 | 0.00 |
| 2-bit BFCL Parallel-Multi (Gemma 4) | **1.33** | 0.00 | 0.00 | 0.00 |

At 3-bit, CQ retains near-FP16 function calling while baselines lose 40-80 points.

### Multimodal: Vision and Audio

CQ quantizes modality towers with the same recipe. Under a joint (M, L) schedule where
M is the modality-tower bit width and L is the LLM bit width:

| Setting | CQ ChartQA | AWQ ChartQA | HQQ ChartQA |
|---|---|---|---|
| M4 + L4 | **51.00** | 49.00 | 49.33 |
| M2 + L4 | **48.44** | 14.22 | 14.33 |

At 2-bit modality tower with 4-bit LLM, AWQ and HQQ collapse on ChartQA from ~49 to
~14 while CQ retains 48.44.

### Transcription (Whisper and Parakeet)

On speech models, CQ is the best method on Parakeet at every bit width. The standout:
2-bit Parakeet-1.1B where CQ holds 0.147 WER while every baseline collapses to ≥ 1.0.

| | CQ | GPTQ | AWQ | HQQ |
|---|---|---|---|---|
| Parakeet-1.1B 4-bit WER | **0.115** | 0.125 | 0.116 | 0.125 |
| Parakeet-1.1B 3-bit WER | **0.115** | 0.298 | 0.157 | 0.298 |
| Parakeet-1.1B 2-bit WER | **0.147** | 1.043 | 1.084 | 1.043 |

On Whisper, CQ is competitive but doesn't consistently lead — it's within 1% of the
best method at most settings.

### Mixed-Precision Allocation

CQ supports sensitivity-driven mixed precision: rank tensors by their impact on
downstream accuracy, keep high-sensitivity tensors at higher bits, push low-sensitivity
ones lower.

The sweet spot on Gemma 4: allocate 4-bit to 68 high-sensitivity LLM units and 3-bit
to the remaining 207 (3.26-bit average). For CQ, this costs ~2 points vs uniform 4-bit.
For GPTQ/AWQ/HQQ, the same allocation costs 7-15 points — mixed precision is
approximately free for CQ but expensive for baselines.

A systematic sweep shows that promoting just 25% of LLM tensors from 2-bit to 3-bit
(2.50 average bits) recovers ~75% of the CQ2-to-CQ3 accuracy gap. The marginal value
of a bit is sharply non-uniform across tensors.

### Production Bundles

Real deployment uses different bit widths for different tensor types. The production
bundle for Gemma 4 E2B:

| Component | Bits | Rationale |
|---|---|---|
| LLM linears | 4 | Accuracy-sensitive |
| Per-layer embeddings (PLI) | 2 | 60% of model, compresses 7.5x |
| Shared token embeddings | 4 | Small, accuracy-sensitive |
| Vision tower | 2 | Compresses gracefully |
| Audio tower | 2 | Holds understanding, some WER cost |

**Result: 4.79 GB (FP16) → ~0.9 GB on disk. 5.3x reduction.**

At ~3.0 bits-per-weight, this production bundle:

- Matches or exceeds Unsloth UD-Q2_K_XL (~3.4 bpw) on 10 of 12 text/tool metrics
  despite a tighter bit budget
- Retains MMLU 59.14 (FP16: 62.33), GSM8K 74.00 (FP16: 73.67)
- Retains BFCL Parallel-Multi 78.00 (FP16: 78.00)
- Retains MMAU 58.67 (FP16: 56.00)
- Largest concession: LibriSpeech WER degrades from ~5.9 to 10.42 at 2-bit audio tower

The mixed-precision bundle at ~2.7 bpw is the deployment sweet spot: half the bit budget
of Unsloth UD-Q4_K_XL while matching it on most metrics.

## Where CQ Is Weakest

Three patterns are visible:

1. **Audio transcription at 2-bit modality tower.** LibriSpeech WER degrades from ~5.9
   to ~10.4. Audio understanding (MMAU) is preserved — the degradation is concentrated
   in token-level acoustic decoding, not higher-level reasoning.

2. **Tool calling at 2-bit.** BFCL Parallel-Multi is near zero at 2-bit for every method
   including CQ. Function calling requires longer-horizon coherence that 2-bit noise breaks.

3. **LFM at 4-bit.** On LFM-2.5-VL-1.6B, HQQ wins several metrics at 4-bit. CQ's
   advantage only appears at 3-bit and below on this family, suggesting that at sufficient
   bit budget the choice of quantizer matters less than per-model factors.

## Using CQ Weights

`cactus convert <model>` builds a runnable bundle locally; `cactus run` runs a
bundle path or a model id.

```bash
cactus convert google/gemma-4-E2B-it ./gemma4-bundle
cactus convert google/gemma-4-E2B-it ./gemma4-bundle --bits 2
cactus run ./gemma4-bundle
```

For models on huggingface.co/Cactus-Compute, `cactus run <model-id>` (or `cactus
download <model-id>`) fetches the pre-built bundle, building locally if none is
published.

The CQ matmul kernels live in `cactus-kernels/src/matmul.cpp`. At inference,
the kernel applies a Walsh-Hadamard transform to the FP16 activation (not the weight),
then performs a fused codebook-lookup + norm-scale + matmul in a single NEON pass.

## See Also

- [Cactus Kernels](/docs/cactus_kernels.md) — CQ GEMV/GEMM kernel implementations
- [TurboQuant-H](/blog/turboquant-h.md) — The 2-bit embedding quantization technique for AltUp models
- [Cactus Engine API](/docs/cactus_engine.md) — Run quantized models via the C API
- [Fine-tuning Guide](/docs/finetuning.md) — Convert LoRA fine-tunes to CQ format
