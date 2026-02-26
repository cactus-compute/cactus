---
title: "Cactus Rust SDK"
description: "Rust FFI bindings to the Cactus C API for on-device AI inference. Auto-generated via bindgen with CMake build integration."
keywords: ["Rust SDK", "FFI bindings", "bindgen", "on-device AI", "Cactus inference"]
---

# Cactus Rust Bindings

Raw FFI bindings to the Cactus C API. Auto-generated via `bindgen`.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
cactus-sys = { path = "rust/cactus-sys" }
```

Build requirements:
- CMake
- C++20 compiler
- On macOS: Xcode command line tools
- On Linux: `build-essential`, `libcurl4-openssl-dev`, `libclang-dev`

## Usage

All functions mirror the C API documented in `docs/cactus_engine.md`.

For usage examples, see:
- Test files: `rust/cactus-sys/tests/`
- C API docs: `docs/cactus_engine.md`
- Other SDKs: `python/README.md`, `apple/README.md`

## Testing

```bash
export CACTUS_MODEL_PATH=/path/to/model
export CACTUS_STT_MODEL_PATH=/path/to/whisper-model
cargo test --manifest-path rust/Cargo.toml -- --nocapture
```

## See Also

- [Cactus Engine API](/docs/cactus_engine.md) — Full C API reference that the Rust bindings wrap
- [Python SDK](/python/) — Python bindings with higher-level wrappers
- [Swift SDK](/apple/) — Swift bindings for Apple platforms
- [Kotlin/Android SDK](/android/) — Kotlin bindings for Android
