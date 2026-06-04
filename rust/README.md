---
title: "Cactus Rust SDK"
description: "Rust FFI bindings to the Cactus C API for on-device AI inference. Auto-generated via bindgen with CMake build integration."
keywords: ["Rust SDK", "FFI bindings", "bindgen", "on-device AI", "Cactus inference"]
---

# Cactus Rust Bindings

Raw FFI bindings to the Cactus C API. Auto-generated via `bindgen`.

> **Model weights:** Pre-converted weights for all supported models at [huggingface.co/Cactus-Compute](https://huggingface.co/Cactus-Compute).

## Installation

<!-- --8<-- [start:install] -->
Add to your `Cargo.toml`:

```toml
[dependencies]
cactus-sys = { path = "rust/cactus-sys" }
```

Build requirements: CMake, C++20 compiler, and platform tools (Xcode CLI on macOS, `build-essential` + `libcurl4-openssl-dev` + `libclang-dev` on Linux).
<!-- --8<-- [end:install] -->

## Usage

All functions mirror the C API documented in `docs/cactus_engine.md`. For vision models (LFM2-VL, LFM2.5-VL, Gemma4, Qwen3.5), add `"images": ["path/to/image.png"]` to any message. For audio models (Gemma4), add `"audio": ["path/to/audio.wav"]`.

For usage examples, see:
- Test files: `rust/cactus-sys/tests/`
- C API docs: `docs/cactus_engine.md`
- Other SDKs: `python/README.md`, `apple/README.md`

## Testing

```bash
export CACTUS_MODEL_PATH=/path/to/model
export CACTUS_STT_MODEL_PATH=/path/to/whisper-model
export CACTUS_STT_AUDIO_PATH=/path/to/audio.wav
cargo test --manifest-path rust/Cargo.toml -- --nocapture
```

## See Also

- [Cactus Engine API](/docs/cactus_engine.md) — Full C API reference that the Rust bindings wrap
- [Python SDK](/python/) — Python bindings with higher-level wrappers
- [Swift SDK](/apple/) — Swift bindings for Apple platforms
- [Kotlin/Android SDK](/android/) — Kotlin bindings for Android
