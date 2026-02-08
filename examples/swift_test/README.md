# Cactus Swift Integration Test

Command-line test for the Cactus Swift bindings on macOS.

## Prerequisites

- macOS 13.0+ with Xcode command line tools
- `cactus.framework` built at `apple/build-macos/Release/`
- Model weights (e.g., `weights/lfm2-350m`)

## Build & Run
```bash
# 1. Build the framework (if not already built)
cactus build --apple

# 2. Build the test
cd examples/swift_test
./build.sh

# 3. Run (auto-detects model at ~/cactus/weights/lfm2-350m)
./swift_test

# Or specify a model path
./swift_test /path/to/weights/lfm2-350m
```

## Tests

| # | Test | What it verifies |
|---|------|-----------------|
| 1 | Model Init | Load model, error handling on bad path |
| 2 | Basic Completion | Single prompt → response text + metrics |
| 3 | Chat Messages | System + user messages, name recall |
| 4 | Completion Options | Temperature, max tokens, stop sequences |
| 5 | Streaming Tokens | Token callback fires, text accumulates |
| 6 | Confidence & Handoff | Confidence score populated, cloud_handoff flag |
| 7 | Performance Metrics | Prefill/decode tok/s, TTFT, total time |
| 8 | Reset | KV cache clear between independent completions |
| 9 | Tokenization | Tokenizer returns valid token IDs |

## Bugs Found

See the parent PR for bugs found and fixed in `Cactus.swift`.
