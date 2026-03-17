---
title: "Cactus Swift Multiplatform SDK"
description: "Swift API for running AI models on-device on iOS, macOS, and Android. Supports transcription, embeddings, RAG, and tool calling."
keywords: ["Swift SDK", "iOS", "macOS", "XCFramework", "on-device AI", "Apple Silicon", "NPU"]
---

# Cactus for Swift Multiplatform

Run AI models on-device with a simple Swift API on iOS, macOS, and Android.

> **Model weights:** Pre-converted weights for all supported models at [huggingface.co/Cactus-Compute](https://huggingface.co/Cactus-Compute).

## Building

<!-- --8<-- [start:install] -->
```bash
git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup
cactus build --apple
```

Build outputs (in `apple/`):

| File | Description |
|------|-------------|
| `cactus-ios.xcframework/` | iOS framework (device + simulator) |
| `cactus-macos.xcframework/` | macOS framework |
| `libcactus-device.a` | Static library for iOS device |
| `libcactus-simulator.a` | Static library for iOS simulator |
<!-- --8<-- [end:install] -->

See the main [README.md](../README.md) for how to use CLI & download weights

For Android, build `libcactus.so` from the `android/` directory.

### Vendored libcurl (iOS + macOS)

To bundle libcurl from this repo instead of relying on system curl, place artifacts under:

- `libs/curl/include/curl/*.h`
- `libs/curl/ios/device/libcurl.a`
- `libs/curl/ios/simulator/libcurl.a`
- `libs/curl/macos/libcurl.a`

Build scripts auto-detect `libs/curl`. Override with:

```bash
CACTUS_CURL_ROOT=/absolute/path/to/curl cactus build --apple
```

## Integration

<!-- --8<-- [start:integration] -->
### iOS/macOS: XCFramework (Recommended)

1. Drag `cactus-ios.xcframework` (or `cactus-macos.xcframework`) into your Xcode project
2. Ensure "Embed & Sign" is selected in "Frameworks, Libraries, and Embedded Content"
3. Copy `Cactus.swift` into your project

### iOS/macOS: Static Library

1. Add `libcactus-device.a` (or `libcactus-simulator.a`) to "Link Binary With Libraries"
2. Create a folder with `cactus_ffi.h` and `module.modulemap`, add to Build Settings:
   - "Header Search Paths" → path to folder
   - "Import Paths" (Swift) → path to folder
3. Copy `Cactus.swift` into your project
<!-- --8<-- [end:integration] -->

### Android (Swift SDK)

Requires [Swift SDK for Android](https://www.swift.org/documentation/articles/swift-sdk-for-android-getting-started.html).

1. Copy files to your Swift project:
   - `libcactus.so` → your library path
   - `cactus_ffi.h` → your include path
   - `module.android.modulemap` → rename to `module.modulemap` in include path
   - `Cactus.swift` → your sources

2. Build with Swift SDK for Android:
```bash
swift build --swift-sdk aarch64-unknown-linux-android28 \
    -Xswiftc -I/path/to/include \
    -Xlinker -L/path/to/lib \
    -Xlinker -lcactus
```

3. Bundle `libcactus.so` with your APK in `jniLibs/arm64-v8a/`

## Usage

Handles are typed as `CactusModelT`, `CactusIndexT`, and `CactusStreamTranscribeT` (all `UnsafeMutableRawPointer` aliases).

### Basic Completion

<!-- --8<-- [start:example] -->
```swift
import Foundation

let model = try cactusInit("/path/to/model", nil, false)
defer { cactusDestroy(model) }

let messages = #"[{"role":"user","content":"What is the capital of France?"}]"#
let resultJson = try cactusComplete(model, messages, nil, nil, nil)
print(resultJson)
```
<!-- --8<-- [end:example] -->

For vision models (LFM2-VL, LFM2.5-VL), add `"images": ["path/to/image.png"]` to any message. See [Engine API](/docs/cactus_engine.md) for details.

### Completion with Options and Streaming

```swift
let options = #"{"max_tokens":256,"temperature":0.7}"#

let resultJson = try cactusComplete(model, messages, options, nil as String?) { token, _ in
    print(token, terminator: "")
}
print(resultJson)
```

### Prefill

Pre-processes input text and populates the KV cache without generating output tokens. This reduces latency for subsequent calls to `cactusComplete`.

```swift
func cactusPrefill(
    _ model: CactusModelT,
    _ messagesJson: String,
    _ optionsJson: String?,
    _ toolsJson: String?
) throws -> String
```

```swift
let tools = #"[
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, State, Country"}
                },
                "required": ["location"]
            }
        }
    }
]"#

let messages = #"[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": "<|tool_call_start|>get_weather(location=\"Paris\")<|tool_call_end|>"},
    {"role": "tool", "content": "{\"name\": \"get_weather\", \"content\": \"Sunny, 72°F\"}"},
    {"role": "assistant", "content": "It's sunny and 72°F in Paris!"}
]"#

let resultJson = try cactusPrefill(model, messages, nil, tools)

let completionMessages = #"[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": "<|tool_call_start|>get_weather(location=\"Paris\")<|tool_call_end|>"},
    {"role": "tool", "content": "{\"name\": \"get_weather\", \"content\": \"Sunny, 72°F\"}"},
    {"role": "assistant", "content": "It's sunny and 72°F in Paris!"},
    {"role": "user", "content": "What about SF?"}
]"#
let completion = try cactusComplete(model, completionMessages, nil, tools, nil)
```

**Response format:**
```json
{
    "success": true,
    "error": null,
    "prefill_tokens": 25,
    "prefill_tps": 166.1,
    "total_time_ms": 150.5,
    "ram_usage_mb": 245.67
}
```

### Audio Transcription

```swift
// From file
let resultJson = try cactusTranscribe(model, "/path/to/audio.wav", nil, nil, nil as ((String, UInt32) -> Void)?, nil as Data?)
print(resultJson)

// From PCM data (16 kHz mono)
let pcmData: Data = ...
let resultJson2 = try cactusTranscribe(model, nil, nil, nil, nil as ((String, UInt32) -> Void)?, pcmData)
print(resultJson2)
```

`segments` contains timestamps (seconds): phrase-level for Whisper, word-level for Parakeet TDT, one segment per transcription window for Parakeet CTC and Moonshine (consecutive VAD speech regions up to 30s).

```swift
if let data = resultJson.data(using: .utf8),
   let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
   let segments = obj["segments"] as? [[String: Any]] {
    for seg in segments {
        let start = seg["start"] as? Double ?? 0
        let end = seg["end"] as? Double ?? 0
        let text = seg["text"] as? String ?? ""
        print(String(format: "[%.3fs - %.3fs] %@", start, end, text))
    }
}
```

### Streaming Transcription

```swift
let stream = try cactusStreamTranscribeStart(model, nil as String?)

let partialJson = try cactusStreamTranscribeProcess(stream, audioChunk)
print(partialJson)

let finalJson = try cactusStreamTranscribeStop(stream)
print(finalJson)
```

### Embeddings

```swift
let embedding      = try cactusEmbed(model, "Hello, world!", true)
let imageEmbedding = try cactusImageEmbed(model, "/path/to/image.jpg")
let audioEmbedding = try cactusAudioEmbed(model, "/path/to/audio.wav")
```

### Tokenization

```swift
let tokens = try cactusTokenize(model, "Hello, world!")
let scoresJson = try cactusScoreWindow(model, tokens, 0, tokens.count, min(tokens.count, 512))
print(scoresJson)
```

### Detect Language

```swift
let langJson = try cactusDetectLanguage(model, "/path/to/audio.wav", nil, nil)
print(langJson)
```

### VAD

```swift
let vadJson = try cactusVad(model, "/path/to/audio.wav", nil as String?, nil as Data?)
print(vadJson)
```

### RAG

```swift
let ragJson = try cactusRagQuery(model, "What is machine learning?", 5)
print(ragJson)
```

### Vector Index

```swift
let index = try cactusIndexInit("/path/to/index", 384)
defer { cactusIndexDestroy(index) }

try cactusIndexAdd(index, [Int32(1), Int32(2)], ["doc1", "doc2"],
                   [[Float(0.1), Float(0.2), ...], [Float(0.3), Float(0.4), ...]], nil)

let results = try cactusIndexQuery(index, [Float(0.1), Float(0.2), ...], nil)

try cactusIndexDelete(index, [Int32(2)])
try cactusIndexCompact(index)
```

## API Reference

All functions are top-level and mirror the C FFI directly.

### Types

```swift
public typealias CactusModelT           = UnsafeMutableRawPointer
public typealias CactusIndexT           = UnsafeMutableRawPointer
public typealias CactusStreamTranscribeT = UnsafeMutableRawPointer
```

All `throws` functions throw `NSError` (domain `"cactus"`) on failure.

### Init / Lifecycle

```swift
func cactusInit(_ modelPath: String, _ corpusDir: String?, _ cacheIndex: Bool) throws -> CactusModelT
func cactusDestroy(_ model: CactusModelT)
func cactusReset(_ model: CactusModelT)
func cactusStop(_ model: CactusModelT)
func cactusGetLastError() -> String
```

### Prefill

```swift
func cactusPrefill(
    _ model: CactusModelT,
    _ messagesJson: String,
    _ optionsJson: String?,
    _ toolsJson: String?
) throws -> String
```

### Completion

```swift
func cactusComplete(
    _ model: CactusModelT,
    _ messagesJson: String,
    _ optionsJson: String?,
    _ toolsJson: String?,
    _ callback: ((String, UInt32) -> Void)?
) throws -> String
```

### Transcription

```swift
func cactusTranscribe(
    _ model: CactusModelT,
    _ audioPath: String?,
    _ prompt: String?,
    _ optionsJson: String?,
    _ callback: ((String, UInt32) -> Void)?,
    _ pcmData: Data?
) throws -> String

func cactusStreamTranscribeStart(_ model: CactusModelT, _ optionsJson: String?) throws -> CactusStreamTranscribeT
func cactusStreamTranscribeProcess(_ stream: CactusStreamTranscribeT, _ pcmData: Data) throws -> String
func cactusStreamTranscribeStop(_ stream: CactusStreamTranscribeT) throws -> String
```

### Embeddings

```swift
func cactusEmbed(_ model: CactusModelT, _ text: String, _ normalize: Bool) throws -> [Float]
func cactusImageEmbed(_ model: CactusModelT, _ imagePath: String) throws -> [Float]
func cactusAudioEmbed(_ model: CactusModelT, _ audioPath: String) throws -> [Float]
```

### Tokenization / Scoring

```swift
func cactusTokenize(_ model: CactusModelT, _ text: String) throws -> [UInt32]
func cactusScoreWindow(_ model: CactusModelT, _ tokens: [UInt32], _ start: Int, _ end: Int, _ context: Int) throws -> String
```

### Detect Language

```swift
func cactusDetectLanguage(_ model: CactusModelT, _ audioPath: String?, _ optionsJson: String?, _ pcmData: Data?) throws -> String
```

### VAD / RAG

```swift
func cactusVad(_ model: CactusModelT, _ audioPath: String?, _ optionsJson: String?, _ pcmData: Data?) throws -> String
func cactusRagQuery(_ model: CactusModelT, _ query: String, _ topK: Int) throws -> String
```

### Vector Index

```swift
func cactusIndexInit(_ indexDir: String, _ embeddingDim: Int) throws -> CactusIndexT
func cactusIndexDestroy(_ index: CactusIndexT)
func cactusIndexAdd(_ index: CactusIndexT, _ ids: [Int32], _ documents: [String], _ embeddings: [[Float]], _ metadatas: [String]?) throws
func cactusIndexDelete(_ index: CactusIndexT, _ ids: [Int32]) throws
func cactusIndexGet(_ index: CactusIndexT, _ ids: [Int32]) throws -> String
func cactusIndexQuery(_ index: CactusIndexT, _ embedding: [Float], _ optionsJson: String?) throws -> String
func cactusIndexCompact(_ index: CactusIndexT) throws
```

### Logging

```swift
func cactusLogSetLevel(_ level: Int32)  // 0=DEBUG, 1=INFO, 2=WARN (default), 3=ERROR, 4=NONE
func cactusLogSetCallback(_ callback: ((Int32, String, String) -> Void)?)
```

### Telemetry

```swift
func cactusSetTelemetryEnvironment(_ path: String)
func cactusSetAppId(_ appId: String)
func cactusTelemetryFlush()
func cactusTelemetryShutdown()
```

## Requirements

**Apple Platforms:**
- iOS 13.0+ / macOS 13.0+
- Xcode 14.0+
- Swift 5.7+

**Android:**
- Swift 6.0+ with [Swift SDK for Android](https://www.swift.org/documentation/articles/swift-sdk-for-android-getting-started.html)
- Android NDK 27d+
- Android API 28+ / arm64-v8a

## See Also

- [Cactus Engine API](/docs/cactus_engine.md) — Full C API reference underlying the Swift bindings
- [Cactus Index API](/docs/cactus_index.md) — Vector database API for RAG applications
- [Fine-tuning Guide](/docs/finetuning.md) — Deploy custom fine-tunes to iOS/macOS
- [Kotlin/Android SDK](/android/) — Kotlin alternative for Android
- [Flutter SDK](/flutter/) — Cross-platform alternative using Dart
