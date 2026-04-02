---
title: "Gemma4 on Cactus: On-device AI you can voice-prompt and trust to route complex tasks to frontier cloud models"
description: "You can naturally talk to Gemma 4 in real-time, show it images, use for transcription, and trust to route complex tasks to frontier cloud models."
keywords: ["gemma4", "multimodal", "hybrid", "realtime inference", "Mac inference", "Google", "DeepMind"]
author: "Henry Ndubuaku & The Cactus Jacks"
date: 2026-04-01
tags: ["gemma4", "cactus", "on-device", "hybrid inference", "multimodal", "voice AI"]
---

## Performance

| Device | 4096-prefill | 100-decode (single CPU core) | 30s-audio-NPU latency | image-NPU latency |
|---|---|---|---|---|
| Mac/iPad/Vision M5 | | | | |
| Mac/iPad M4 Pro | | | | |
| Mac/iPad M4 | | | | |
| Mac/iPad M3 | | | | |
| iPhone 17 Pro | | | | |
| iPhone 13 Mini | | | | |

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

## LLM Benchmarks

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

## Vision Benchmarks

| Benchmark | E4B | E2B | Gemma 3 27B (no think) |
|---|---|---|---|
| MMMU Pro | 52.6% | 44.2% | 49.7% |
| OmniDocBench 1.5 (edit dist, lower=better) | 0.181 | 0.290 | 0.365 |
| MATH-Vision | 59.5% | 52.4% | 46.0% |
| MedXPertQA MM | 28.7% | 23.5% | - |

## Audio Benchmarks

| Benchmark | E4B | E2B |
|---|---|---|
| CoVoST | 35.54 | 33.47 |
| FLEURS (lower=better) | 0.08 | 0.09 |
