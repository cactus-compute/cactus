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
