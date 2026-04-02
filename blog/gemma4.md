---
title: "Gemma 4 on Cactus: The first model you can talk to, show things, and trust to know when it needs help"
description: "Gemma 4 runs natively on your device with real-time voice, vision, and audio, and routes hard problems to the cloud when it should."
keywords: ["gemma4", "multimodal", "hybrid", "realtime inference", "on-device inference", "Google", "DeepMind"]
author: "Henry Ndubuaku & The Cactus Jacks"
date: 2026-04-01
tags: ["gemma4", "cactus", "on-device", "hybrid inference", "multimodal", "voice AI"]
---

Gemma 4 is the first on-device model that genuinely works across text, vision, and audio in a single architecture, and does it well enough that you stop thinking about the model and start thinking about what to build with it.

We shipped day-one support in Cactus. Here's what that means in practice and why we think this changes the trajectory of on-device AI.

## What's actually new here

Most "multimodal" models bolt on vision or audio as an afterthought. Gemma 4 is different. The vision encoder, audio encoder, and language model were trained together from the start. The audio conformer isn't a whisper-style bolt-on. It's a 300M-parameter encoder that feeds directly into the transformer's residual stream. Same for vision. The model doesn't "transcribe then think." It reasons over the raw modality.

This matters because latency compounds. A pipeline that transcribes audio, then feeds text to an LLM, then generates a response has three serialized steps. Gemma 4 collapses that into one forward pass. On a modern ARM device, that means 0.3 seconds from the end of a 30-second voice clip to the first token of a response. Not 0.3 seconds for the transcription step. 0.3 seconds total.

## Performance

Cactus targets ARM across platforms: Apple Silicon Macs, iPhones, iPads, Vision Pro, and Android devices with ARM64 chipsets (Snapdragon, Dimensity, Tensor). On Apple hardware we additionally leverage the Neural Engine for vision and audio encoders, while on Android and Linux the runtime uses NEON, i8mm, and dot-product intrinsics.

| Metric (M5 Mac/iPad/Vision Pro) | E2B |
|---|---|
| 4096-token prefill | 660 tok/s |
| 1024-token decode | 40 tok/s |
| 30s audio end-to-end | 0.3s |
| Image encode (ANE) | 0.7s |

40 tok/s decode is faster than most people read. You're not waiting for the model. The model is waiting for you.

## The benchmarks are hard to believe

We normally don't lead with benchmarks because they rarely reflect real-world use. But these numbers demand attention.

### LLM

| Benchmark | E4B | E2B | Gemma 3 27B (no think) |
|---|---|---|---|
| MMLU Pro | 69.4% | 60.0% | 67.6% |
| AIME 2026 (no tools) | 42.5% | 37.5% | 20.8% |
| LiveCodeBench v6 | 52.0% | 44.0% | 29.1% |
| Codeforces ELO | 940 | 633 | 110 |
| GPQA Diamond | 58.6% | 43.4% | 42.4% |
| Tau2 (avg over 3) | 42.2% | 24.5% | 16.2% |
| BigBench Extra Hard | 33.1% | 21.9% | 19.3% |
| MMMLU | 76.6% | 67.4% | 70.7% |
| MRCR v2 8-needle 128k (avg) | 25.4% | 19.1% | 13.5% |

E4B, a 4.5B-effective-parameter model running on your phone, outperforms Gemma 3 27B on nearly every benchmark. E2B, the smaller variant at 2.3B effective parameters, matches or beats it on half of them. The AIME and LiveCodeBench numbers are particularly striking: these are hard reasoning tasks where you'd expect scale to dominate.

### Vision

| Benchmark | E4B | E2B | Gemma 3 27B (no think) |
|---|---|---|---|
| MMMU Pro | 52.6% | 44.2% | 49.7% |
| OmniDocBench 1.5 (edit dist, lower=better) | 0.181 | 0.290 | 0.365 |
| MATH-Vision | 59.5% | 52.4% | 46.0% |
| MedXPertQA MM | 28.7% | 23.5% | - |

E4B beats Gemma 3 27B on vision tasks across the board. Document understanding (OmniDocBench) is 2x better. This isn't a toy. It's genuinely useful for reading receipts, parsing handwritten notes, understanding diagrams.

### Audio

| Benchmark | E4B | E2B |
|---|---|---|
| CoVoST | 35.54 | 33.47 |
| FLEURS (lower=better) | 0.08 | 0.09 |

## Model Architecture

| Property | E2B | E4B |
|---|---|---|
| Total Parameters | 2.3B effective (5.1B w/ embeddings) | 4.5B effective (8B w/ embeddings) |
| Layers | 35 | 42 |
| Sliding Window | 512 tokens | 512 tokens |
| Context Length | 128K tokens | 128K tokens |
| Vocabulary Size | 262K | 262K |
| Supported Modalities | Text, Image, Audio | Text, Image, Audio |
| Vision Encoder | ~150M params | ~150M params |
| Audio Encoder | ~300M params | ~300M params |

The architecture uses per-layer embeddings with AltUp, a technique that keeps most of the vocabulary knowledge in a shared embedding table while giving each layer a small specialized projection. This is how they fit 262K vocabulary tokens into a model that runs on a phone. The sliding window attention at 512 tokens with global attention every 5 layers gives you 128K context without the quadratic memory cost.

## What this unlocks

### Voice control that actually works

We've all used voice assistants that feel like speaking into a form field. The voice gets transcribed, the text gets processed, and you get a response that could just as easily have been typed. Gemma 4 doesn't work that way. Because it reasons directly over the audio signal, it picks up on tone, hesitation, emphasis. It knows the difference between "delete that" spoken confidently and "delete that?" spoken as a question.

On headsets running Cactus (Vision Pro, Quest, or any AR/VR platform with ARM compute), this means spatial voice control that responds to how you speak, not just what you say. Ask it to "move that over there" while looking at a 3D object and gesturing, and the model has the audio understanding to disambiguate "that" from context. The 0.3s latency means the interaction feels physical. You speak, things happen.

### Voice agents that run locally

If you're building a voice agent (a customer service bot, an in-car assistant, a medical triage system) you currently have two bad options. You can run everything in the cloud and deal with latency, cost, and privacy issues. Or you can cobble together a local pipeline of ASR + LLM + TTS and deal with error propagation and integration pain.

Gemma 4 on Cactus gives you a third option. The model handles voice understanding natively. You pipe audio in, you get structured responses out. Tool calling works. The model can decide to call functions, hit APIs, or route to a cloud model when the task exceeds its capability. And because it runs locally, there's no per-request cost and no audio leaving the device.

For healthcare, legal, and finance applications where audio data is sensitive, this isn't a nice-to-have. It's a compliance requirement.

### Hybrid inference: knowing when to ask for help

This is the part we're most excited about. Gemma 4 is good, but it's not omniscient. A 2B-parameter model isn't going to write a production database migration or debug a complex distributed system. What it can do is recognize when a task is beyond its capability and route it to a frontier cloud model.

Cactus implements this as cloud handoff. The on-device model evaluates the complexity of the request, and if it determines it can't handle it confidently, it signals for handoff. The request goes to a cloud model (Claude, GPT-4, Gemini, your choice), the response comes back, and the user sees a seamless interaction. The on-device model handles the 80% of requests that are straightforward. The cloud handles the 20% that need heavy lifting.

This means you can build apps that feel like they have frontier-model intelligence while keeping costs at a fraction of full cloud inference. And the user's casual conversations, voice notes, and image queries never leave their device.

### Building on any platform

Cactus runs Gemma 4 on macOS, iOS, Android, and Linux.

## Try it

```bash
brew install cactus-compute/cactus/cactus
cactus run google/gemma-4-E2B-it
```

The weights download automatically. INT4 quantized, optimized for ARM. Both E2B and E4B are available.

## Build with it

Cactus supports React Native, Flutter, Swift, Kotlin, Python, Rust, and C++. Pick the SDK that fits your stack:

| Platform | SDK | Install |
|---|---|---|
| React Native | [cactus-react-native](https://github.com/cactus-compute/cactus-react-native) | `npm install cactus-react-native` |
| Flutter | [cactus-flutter](https://cactus.dev/flutter/) | Dart FFI bindings |
| Swift (iOS/macOS) | [cactus-apple](https://cactus.dev/apple/) | XCFramework with NPU support |
| Kotlin (Android) | [cactus-android](https://cactus.dev/android/) | JNI + Kotlin Multiplatform |
| Python | [cactus-ai](https://cactus.dev/python/) | `pip install cactus-ai` |
| Rust | [cactus-sys](https://cactus.dev/rust/) | `cargo add cactus-sys` |
| C++ | [cactus.h](https://cactus.dev/docs/cactus_engine/) | Single header, link `libcactus.a` |

Full quickstart with code examples for every SDK: [cactus.dev/docs/quickstart](https://cactus.dev/docs/quickstart/)

Pre-quantized weights for Gemma 4 and 30+ other models: [huggingface.co/cactus-compute](https://huggingface.co/cactus-compute)

If you're building something with on-device multimodal AI (voice agents, VR interfaces, local-first apps) we want to hear about it. Open an issue on [GitHub](https://github.com/cactus-compute/cactus), or just start building.
