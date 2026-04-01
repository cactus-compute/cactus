---
title: "LFM-2.5-350m on Cactus: 140 tok/sec, Single Core, 355 MB"
description: "Benchmarking Liquid's LFM-2.5-350m across seven devices with Cactus. INT8 quantization, single-core CPU decode, zero-copy loading, and why this configuration makes on-device inference practical."
keywords: ["LFM-2.5", "on-device inference", "INT8 quantization", "ARM CPU", "mobile AI", "edge inference", "Cactus"]
author: "Henry Ndubuaku"
date: 2026-03-31
tags: ["LFM", "INT8", "edge AI", "mobile inference", "ARM", "benchmarks"]
---

# LFM-2.5-350m on Cactus: 140 tok/sec, Single Core, 355 MB

*By Henry Ndubuaku*

We ran Liquid's LFM-2.5-350m through the Cactus inference engine across seven devices, from a Vision Pro to a Raspberry Pi 5. 
The configuration: 1024-token prefill, 100-token decode, INT8 quantization, CPU-only, single-core decode. 
The model file is 355 MB on Cactus Compressed format. 

## Device Benchmarks

### Apple
| Device | Prefill (tok/s) | Decode (tok/s) | RAM |
|---|---|---|---|
| Vision Pro | 2,067 | 140 | 85 MB |
| iPhone 17 Pro | 797 | 140 | 75 MB |
| iPhone 13 Mini | 496 | 88 | 56 MB |

### Android
| Device | Prefill (tok/s) | Decode (tok/s) | RAM |
|---|---|---|---|
| Galaxy S25 Ultra | 660 | 100 | 322 MB |
| Google Pixel 6a | 208 | 42 | 328 MB |
| Galaxy A56 | 110 | 24 | 330 MB |

### Other
| Device | Prefill (tok/s) | Decode (tok/s) | RAM |
|---|---|---|---|
| Raspberry Pi 5 | 200 | 30 | 300 MB |

Two things stand out in this data. First, the decode speeds: 140 tok/sec on a single core is fast, 
most people read at roughly 4-5 words per second, so even the Raspberry Pi's 30 tok/sec is well ahead of human consumption. 
Second, the RAM gap between Apple and Android: 56-85 MB versus 300-330 MB for the same 355 MB model.
That gap is not a bug. It's a direct consequence of how the two platforms handle memory-mapped files, which we'll get into in the Cactus format section below.

## Why CPU, Not GPU or NPU

It is tempting to reach for the GPU. On paper, mobile GPUs have 10-50x the FLOPS of a single CPU core. 
But inference on a 350M-parameter model is not a FLOPS problem, it's a memory bandwidth problem. 
During decode, you generate one token at a time, which means you read the entire weight matrix for each token but only perform a single matrix-vector multiply. 
The arithmetic intensity (FLOPS per byte loaded) is extremely low. Without batched decode, GPU compute units spend most of their time stalled on memory fetches. 
GPU excels when arithmetic intensity is high, large batch sizes, long matrix-matrix multiplies. 
For single-token decode, the GPU's compute units sit mostly idle while waiting for memory.

There's also the practical problem: mobile GPU inference requires Metal on iOS and Vulkan or OpenCL on Android. 
That's two separate shader codepaths to maintain, with different memory models, different synchronization primitives, and different performance characteristics per vendor. 
And GPU inference holds a persistent GPU context, which competes with the UI rendering pipeline and drains battery even when the inference workload is light.
GPUs are also energy-inefficient. On Macs and PCs this barely matters since they're plugged into a power source most of the time, and GPU inference works well there. But mobile devices run on battery, and every watt spent on GPU inference cuts directly into the user's runtime. The GPU path that works fine on a MacBook becomes a battery killer on a phone.

NPU is a more interesting target. Apple's Neural Engine and Qualcomm's Hexagon DSP both offer high-throughput inference at good power efficiency. 
The problem is access. Apple's Neural Engine requires CoreML model conversion, which imposes its own quantization and graph constraints.
You don't control the execution schedule, the memory layout, or the precision semantics. 
The quantisation techniques available through CoreML severely lag the state-of-the-art methods used by GGML and Cactus.
Qualcomm's QNN SDK has similar restrictions (limited quantisation scheme support) and not all Android devices ship Qualcomm chips. 
Both lock you into a vendor-specific format that may not support your model architecture at all, and neither works on budget devices that lack the NPU entirely.

CPU is the universal target. Every ARM device, flagship, budget, wearable, Raspberry Pi, and most recent ones support DOTPROD or I8MM extensions. 
A single well-optimized CPU codepath works everywhere, the main challenge is energy-inefficiency compared to NPU.

ARM is also closing this gap. The progression from NEON to DOTPROD to I8MM to SME2 reflects a deliberate push to bring matrix-processing efficiency onto the CPU itself. Each generation narrows the energy and throughput gap with dedicated NPUs. SME2, available on Apple's M4 and upcoming ARMv9.2 cores, adds streaming matrix operations that approach NPU-class efficiency for INT8 workloads while remaining fully programmable. Cactus already compiles SME2 codepaths where available. The bet is that general-purpose CPU silicon will continue absorbing the capabilities that once justified a separate accelerator.

## Why Single Core for Decode

Decode is memory-bandwidth-bound. For a 350M INT8 model, each decode step reads roughly 355 MB of weights to produce a single token.
On an iPhone 17 Pro with ~60 GB/s memory bandwidth, the theoretical maximum decode speed is around 169 tok/sec,
our measured 140 tok/sec is 83% of that ceiling, the haircut comes from our choice to stream weights from storage for memory efficiency, reducing OS chances of throttling background inference. 
Adding more cores doesn't give you more memory bandwidth; it just adds synchronization overhead.

Prefill is different. With 1024 tokens in the input, each weight load amortizes across 1024 multiply-accumulate operations. 
The arithmetic intensity is 1024x higher than decode, making prefill genuinely compute-bound. 
That's why we use multi-threaded prefill, the Vision Pro's 2,067 tok/sec prefill speed reflects all available cores working in parallel.

But there's a deeper reason single-core decode matters for mobile: background execution. 
On both iOS and Android, the OS aggressively throttles multi-core workloads that run in the background. 
iOS will suspend or terminate background tasks that consume excessive CPU across multiple cores. 
Android's thermal management will cap frequency on the performance cores. 
We found that single-core decode workload, by contrast, looks like a lightweight background task to the OS scheduler. 
It doesn't trigger thermal throttling, doesn't compete with foreground apps, and doesn't get killed. 
This is the difference between inference as a demo and inference as a product feature, the model needs to run while the user is doing something else.

## Why INT8

At 350M parameters, quantization precision matters more than at larger scales. 
Quantization works by approximating continuous weight values with discrete integers, and the approximation error is relative to the dynamic range of each weight group. 
Smaller models have fewer parameters absorbing this error, so each parameter carries more of the model's capacity. 
INT4 quantization at 350M parameters produces measurable accuracy degradation, our internal testing shows 2-4 point drops on MMLU compared to INT8, which is near-lossless at this scale.

Cactus uses grouped affine quantization: weights are divided into groups of 32 elements, and each group gets its own scale factor. 
For each group, the scale is computed as `max(|w|) / 127`, and each weight is rounded to the nearest integer in [-128, 127]. 
During inference, the INT8 values are multiplied by their group's scale to reconstruct the approximate FP32 value. 
Per-group scaling is critical, a single outlier weight in a per-tensor scheme would compress the effective precision of every other weight in that tensor.

The performance benefit is concrete. On ARM CPUs with I8MM support (ARMv8.2-a and later), the `SMMLA` instruction computes a 2x8 by 8x2 INT8 matrix multiply in a single instruction, accumulating into INT32. 
This is a native hardware operation, not a software emulation of low-precision arithmetic. 
On devices without I8MM (like the Pixel 6a's Tensor G1), Cactus falls back to DOTPROD instructions, which compute 4-element INT8 dot products. 
Both paths are substantially faster than FP16 arithmetic because they process more elements per instruction and use less memory bandwidth per parameter.

## The Cactus Format and Zero-Copy Loading

The RAM numbers in our benchmarks tell a story about memory architecture. 
The LFM-2.5-350m model file is 355 MB. On Apple devices, runtime RAM usage is 56-85 MB. On Android, it's 300-330 MB. 
The model is the same. The quantization is the same. The difference is in how the operating system handles memory-mapped files. 

The Cactus format (identified by the magic number `0x54434143`, "CACT") is a binary container designed around `mmap()`. 
Rather than loading the entire model into a heap allocation, Cactus maps the file directly into virtual address space with `mmap(PROT_READ, MAP_SHARED)`. 
The weights stay on disk. When the inference engine accesses a weight tensor, the OS kernel page-faults it into physical memory on demand. 
When memory pressure rises, the OS can evict those pages, they're backed by the file, so no writeback is needed.

On Apple Silicon, this works exceptionally well. 
The unified memory architecture means there's no distinction between CPU memory and file cache, 
a memory-mapped page that's been paged in *is* the physical memory the CPU reads from. 
The OS can manage a working set that's a fraction of the total model size because not all layers are active simultaneously during a single decode step.

On Android, the behavior varies by kernel version and vendor. 
Many Android devices aggressively copy memory-mapped pages into anonymous memory due to SELinux policies and vendor-specific memory management. 
The result is that `mmap()` often degrades to something closer to `malloc() + read()`, explaining the 300+ MB footprint.

The format itself is designed for SIMD-friendly access. 
Weight data is aligned to 32-byte boundaries (matching ARM NEON register width), 
and an optional block-interleaved layout groups 4 columns of weights together so that a single vector load fetches data for 4 output neurons simultaneously. 
This layout eliminates the need for runtime transposition and ensures every memory access is a sequential, aligned read, optimal for both cache prefetching and NEON/I8MM instruction operands.

## Model Quality

A fast small model is only useful if it's accurate enough. Here's how the LFM-2.5 family compares to similarly-sized open models:

| Benchmark | LFM2-350M | LFM2-700M | LFM2-1.2B | Qwen3-0.6B | Qwen3-1.7B | Llama-3.2-1B | gemma-3-1b-it |
|---|---|---|---|---|---|---|---|
| MMLU | 43.43 | 49.90 | **55.23** | 44.93 | **59.11** | 46.60 | 40.08 |
| GPQA | **27.46** | **28.48** | **31.47** | 22.14 | 27.72 | 28.84 | 21.07 |
| IFEval | **65.12** | **72.23** | **74.89** | 64.24 | 73.98 | 52.39 | 62.90 |
| IFBench | 16.41 | **20.56** | **20.70** | 19.75 | **21.27** | 16.86 | 17.72 |
| GSM8K | 30.10 | 46.40 | 58.30 | 36.47 | 51.40 | 35.71 | **59.59** |
| MGSM | 29.52 | 45.36 | 55.04 | 41.28 | **66.56** | 29.12 | 43.60 |
| MMMLU | **37.99** | **43.28** | **46.73** | 30.84 | 46.51 | 38.15 | 34.43 |

The numbers worth examining closely:

**IFEval (instruction following):** LFM2-350M scores 65.12, versus 52.39 for Llama-3.2-1B (a model 3x its size) and 64.24 for Qwen3-0.6B (nearly 2x its size). For on-device applications,
where the model is typically following structured instructions from an app rather than engaging in open-ended conversation, 
instruction following is arguably the most important benchmark. The LFM2-1.2B's 74.89 is within a point of Qwen3-1.7B's 73.98, at 70% of the parameters.

**GPQA (graduate-level reasoning):** LFM2-350M at 27.46 beats both Qwen3-0.6B (22.14) and gemma-3-1b-it (21.07). 
This is notable because GPQA tests reasoning depth, not just pattern matching, it suggests the LFM architecture is efficient at encoding reasoning capability per parameter.

**MMMLU (multilingual knowledge):** LFM2-350M scores 37.99 versus Qwen3-0.6B's 30.84. 
A 7-point lead over a model nearly twice the size suggests strong multilingual data efficiency. The full LFM2-1.2B hits 46.73, matching Qwen3-1.7B's 46.51.

**Where LFM2-350M trails:** GSM8K (30.10) and MGSM (29.52) math benchmarks. gemma-3-1b-it leads GSM8K at 59.59, and Qwen3-1.7B leads MGSM at 66.56. 
Math is where the parameter count ceiling hits hardest, since multi-step arithmetic requires retaining intermediate state that larger models can distribute across more capacity. 
For pure math tasks at this scale, a larger model is the right answer.

## What This Adds Up To

The LFM-2.5-350m at INT8 on Cactus is a 355 MB model that decodes at 88 tok/sec on a three-year-old iPhone 13 Mini using 56 MB of RAM on a single CPU core. 
It follows instructions better than models 3x its size. It runs as a background process without triggering OS throttling.
These are not theoretical numbers on a development board with a fan. They're measured on production consumer devices, under the constraints that real mobile apps face; 
memory pressure from other apps, thermal limits, background execution policies, 4G download speeds for model delivery.
The question for on-device inference has always been whether the quality-speed-size tradeoff lands in a useful region. 
At 350M parameters, LFM-2.5 on Cactus suggests it does: accurate enough for structured on-device tasks, fast enough to be invisible to the user, and small enough to ship inside an app.

## Where is Cactus headed?

On-device inference on a Mac or PC is a solved problem. You have gigabytes of RAM, a fast SSD, a multi-core CPU with wide SIMD, and a stable OS that doesn't kill your process for using too much battery. 
Any reasonably written inference engine will run well on a MacBook and GPUs are a great idea, given these devices are plugged in anyway. 

Mobile devices, wearables, and custom hardware are a different story entirely. The ecosystem is incredibly fragmented. 
You're dealing with hundreds of ARM chip variants across Apple, Qualcomm, MediaTek, Samsung, and Broadcom, each with different ISA extensions, different memory bandwidth, different thermal envelopes. 
Android alone ships on devices ranging from 3 GB of RAM to 16 GB, with kernel versions spanning half a decade. 
Wearables have even tighter constraints. Custom embedded hardware has its own. 
A model that runs fine on a Galaxy S25 Ultra may crash on a Galaxy A56, not because the code is wrong, but because the memory management behavior differs at the kernel level.

This fragmentation is what makes on-device AI difficult in production. It's a research and engineering problem that compounds at scale: every new device, every OS update, 
every vendor-specific memory policy is another edge case that can break inference for some slice of your user base. 
For enterprises shipping AI features to millions of devices, this is a hard pill. 
The choice often comes down to small models that fit everywhere but are too weak to be useful, or capable models with large files that fail on half the device population.

Cactus exists because we took on this specific slice of the problem; I spent the last few years building foundation models and inference for tiny devices. 
The team's approach is systematic: profile every major ARM variant, write dedicated kernel paths for each ISA extension tier (NEON, DOTPROD, I8MM, SME2), 
design the model format around the worst-case memory behavior (Android's mmap limitations), and test on real production devices, not emulators. 
This it not-trivial; battling with chip manufactures with different brand visions, model developers chasing benchmarks over useability and more. 

The result is that businesses shipping with Cactus don't need to worry about inference failing silently on a budget Samsung, or a model that's accurate on benchmarks but too large for half their install base. 
The kernel, the format, and the quantization strategy are all designed so that the same model file works correctly across the full device spectrum, with predictable RAM, predictable speed, and predictable quality.
On-device market is quite noisy, with hobbyists driving decisions, but production use is largely unsolved, Cactus is a bit unconventional but rightfully so. 

LFM-2.5-350m is a step in the right direction, hence this dedicated post from me personally. 
Not just that LFM-2.5-350m is fast on one device, but on Cactus, it brings frontier intelligence to very different devices, 
from a Vision Pro to a Raspberry Pi 5, with no special casing and no device-specific builds. 

## See Also

- [Cactus Engine API Reference](/docs/cactus_engine.md) - Full C API docs for completion, tool calling, and cloud handoff
- [LFM2-24B-A2B](/blog/lfm2_24b_a2b.md) - Running LFM2 24B MoE locally on Mac for coding use cases
- [Hybrid Transcription](/blog/hybrid_transcription.md) - On-device/cloud hybrid speech transcription with Cactus
- [Ridiculously Fast Transcription](/blog/parakeet.md) - 6M tok/sec decode speed with Parakeet CTC 1.1B