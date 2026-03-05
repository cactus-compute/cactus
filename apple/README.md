---
title: "Cactus Swift Multiplatform SDK"
description: "Swift API for running AI models on-device on iOS, macOS, tvOS, watchOS, and Android. Supports transcription, embeddings, RAG, and tool calling."
keywords: ["Swift SDK", "iOS", "macOS", "XCFramework", "on-device AI", "Apple Silicon", "NPU"]
---

# Cactus for Swift Multiplatform

Run AI models on-device with a simple Swift API on iOS, macOS, and Android.

## Building

```bash
cactus build --apple
```

Build outputs (in `apple/`):

see the main [README.md](../README.md) for how to use CLI & download weight

| File | Description |
|------|-------------|
| `cactus-ios.xcframework/` | iOS framework (device + simulator) |
| `cactus-macos.xcframework/` | macOS framework |
| `libcactus-device.a` | Static library for iOS device |
| `libcactus-simulator.a` | Static library for iOS simulator |

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

```swift
import Foundation

let model = try cactusInit("/path/to/model", nil, false)
defer { cactusDestroy(model) }

let messages = #"[{"role":"user","content":"What is the capital of France?"}]"#
let resultJson = try cactusComplete(model, messages, nil, nil, nil)
// resultJson is a JSON string: {"response":"Paris","success":true,...}
if let data = resultJson.data(using: .utf8),
   let result = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
    print(result["response"] as? String ?? "")
}
```

### Completion with Options and Streaming

```swift
let options = #"{"max_tokens":256,"temperature":0.7}"#

let resultJson = try cactusComplete(model, messages, options, nil) { token, _ in
    print(token, terminator: "")
}
```

### Audio Transcription

```swift
// From file
let result = try cactusTranscribe(model, "/path/to/audio.wav", "", nil, nil as ((String, UInt32) -> Void)?, nil as Data?)

// From PCM data (16 kHz mono)
let pcmData: Data = ...
let result = try cactusTranscribe(model, nil, nil, nil, nil as ((String, UInt32) -> Void)?, pcmData)
```

### Streaming Transcription

```swift
let stream = try cactusStreamTranscribeStart(model, nil as String?)
let partial = try cactusStreamTranscribeProcess(stream, audioChunk)
let final_  = try cactusStreamTranscribeStop(stream)
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
let scores = try cactusScoreWindow(model, tokens, 0, tokens.count, min(tokens.count, 512))
```

### VAD

```swift
let result = try cactusVad(model, "/path/to/audio.wav", nil as String?, nil as Data?)
```

### RAG

```swift
let result = try cactusRagQuery(model, "What is machine learning?", 5)
```

### Vector Index

```swift
let index = try cactusIndexInit("/path/to/index", 384)
defer { cactusIndexDestroy(index) }

try cactusIndexAdd(index, [Int32(1), Int32(2)], ["doc1", "doc2"],
                   [[0.1, 0.2, ...], [0.3, 0.4, ...]], nil)

let results = try cactusIndexQuery(index, [0.1, 0.2, ...], nil)
// results is a JSON string: {"results":[{"id":1,"score":0.99,...},...]}

try cactusIndexDelete(index, [2])
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

### Telemetry

```swift
func cactusSetTelemetryEnvironment(_ cacheDir: String)
func cactusSetAppId(_ appId: String)
func cactusTelemetryFlush()
func cactusTelemetryShutdown()
```

## Requirements

**Apple Platforms:**
- iOS 14.0+ / macOS 13.0+ / tvOS 14.0+ / watchOS 7.0+
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
