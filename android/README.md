---
title: "Cactus Android & Kotlin Multiplatform SDK"
description: "Kotlin API for running AI models on-device on Android and iOS via Kotlin Multiplatform. Supports completion, transcription, embeddings, RAG, and tool calling."
keywords: ["Android SDK", "Kotlin Multiplatform", "on-device AI", "mobile inference", "JNI", "KMP"]
---

# Cactus for Android & Kotlin Multiplatform

Run AI models on-device with a simple Kotlin API.

## Building

```bash
cactus build --android
```

Build output: `android/build/lib/libcactus.so`

see the main [README.md](../README.md) for how to use CLI & download weight

### Vendored libcurl (device builds)

To bundle libcurl locally for Android device testing, place artifacts using:

`libs/curl/android/arm64-v8a/libcurl.a` and `libs/curl/include/curl/*.h`

The build auto-detects `libs/curl`. You can override with:

```bash
CACTUS_CURL_ROOT=/absolute/path/to/curl cactus build --android
```

## Integration

### Android-only

1. Copy `libcactus.so` to `app/src/main/jniLibs/arm64-v8a/`
2. Copy `Cactus.kt` to `app/src/main/java/com/cactus/`

### Kotlin Multiplatform

Source files:

| File | Copy to |
|------|---------|
| `Cactus.common.kt` | `shared/src/commonMain/kotlin/com/cactus/` |
| `Cactus.android.kt` | `shared/src/androidMain/kotlin/com/cactus/` |
| `Cactus.ios.kt` | `shared/src/iosMain/kotlin/com/cactus/` |
| `cactus.def` | `shared/src/nativeInterop/cinterop/` |

Binary files:

| Platform | Location |
|----------|----------|
| Android | `libcactus.so` → `app/src/main/jniLibs/arm64-v8a/` |
| iOS | `libcactus-device.a` → link via cinterop |

build.gradle.kts:

```kotlin
kotlin {
    androidTarget()

    listOf(iosArm64(), iosSimulatorArm64()).forEach {
        it.compilations.getByName("main") {
            cinterops {
                create("cactus") {
                    defFile("src/nativeInterop/cinterop/cactus.def")
                    includeDirs("/path/to/cactus/ffi")
                }
            }
        }
        it.binaries.framework {
            linkerOpts("-L/path/to/apple", "-lcactus-device")
        }
    }

    sourceSets {
        commonMain.dependencies {
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.0")
        }
    }
}
```

## Usage

Handles are plain `Long` values (C pointers). All functions are top-level.

### Basic Completion

```kotlin
import com.cactus.*
import org.json.JSONObject

val model = cactusInit("/path/to/model", null, false)
val messages = """[{"role":"user","content":"What is the capital of France?"}]"""
val resultJson = cactusComplete(model, messages, null, null, null)
val result = JSONObject(resultJson)
println(result.getString("response"))
cactusDestroy(model)
```

### Completion with Options and Streaming

```kotlin
val options = """{"max_tokens":256,"temperature":0.7}"""
val tokens = mutableListOf<String>()

val resultJson = cactusComplete(model, messages, options, null) { token, _ ->
    tokens.add(token)
    print(token)
}
```

### Audio Transcription

```kotlin
// From file
val result = cactusTranscribe(model, "/path/to/audio.wav", "", null, null, null)

// From PCM data (16 kHz mono)
val pcmData: ByteArray = ...
val result = cactusTranscribe(model, null, null, null, null, pcmData)
```

### Streaming Transcription

```kotlin
val stream  = cactusStreamTranscribeStart(model, null)
val partial = cactusStreamTranscribeProcess(stream, audioChunk)
val final_  = cactusStreamTranscribeStop(stream)
```

### Embeddings

```kotlin
val embedding      = cactusEmbed(model, "Hello, world!", true)   // FloatArray
val imageEmbedding = cactusImageEmbed(model, "/path/to/image.jpg")
val audioEmbedding = cactusAudioEmbed(model, "/path/to/audio.wav")
```

### Tokenization

```kotlin
val tokens = cactusTokenize(model, "Hello, world!")  // IntArray
val scores = cactusScoreWindow(model, tokens, 0, tokens.size, 512)
```

### VAD

```kotlin
val result = cactusVad(model, "/path/to/audio.wav", null, null)
```

### RAG

```kotlin
val result = cactusRagQuery(model, "What is machine learning?", 5)
```

### Vector Index

```kotlin
val index = cactusIndexInit("/path/to/index", 384)

cactusIndexAdd(
    index,
    intArrayOf(1, 2),
    arrayOf("Document 1", "Document 2"),
    arrayOf(floatArrayOf(0.1f, 0.2f), floatArrayOf(0.3f, 0.4f)),
    null
)

val resultsJson = cactusIndexQuery(index, floatArrayOf(0.1f, 0.2f), null)
// JSON: {"results":[{"id":1,"score":0.99,...},...]}

cactusIndexDelete(index, intArrayOf(2))
cactusIndexCompact(index)
cactusIndexDestroy(index)
```

## API Reference

All functions are top-level and mirror the C FFI directly. Handles are `Long` values.

### Init / Lifecycle

```kotlin
fun cactusInit(modelPath: String, corpusDir: String?, cacheIndex: Boolean): Long  // throws RuntimeException
fun cactusDestroy(model: Long)
fun cactusReset(model: Long)
fun cactusStop(model: Long)
fun cactusGetLastError(): String
```

### Completion

```kotlin
fun cactusComplete(
    model: Long,
    messagesJson: String,
    optionsJson: String?,
    toolsJson: String?,
    callback: CactusTokenCallback?
): String
```

### Transcription

```kotlin
fun cactusTranscribe(
    model: Long,
    audioPath: String?,
    prompt: String?,
    optionsJson: String?,
    callback: CactusTokenCallback?,
    pcmData: ByteArray?
): String

fun cactusStreamTranscribeStart(model: Long, optionsJson: String?): Long  // throws RuntimeException
fun cactusStreamTranscribeProcess(stream: Long, pcmData: ByteArray): String
fun cactusStreamTranscribeStop(stream: Long): String
```

### Embeddings

```kotlin
fun cactusEmbed(model: Long, text: String, normalize: Boolean): FloatArray
fun cactusImageEmbed(model: Long, imagePath: String): FloatArray
fun cactusAudioEmbed(model: Long, audioPath: String): FloatArray
```

### Tokenization / Scoring

```kotlin
fun cactusTokenize(model: Long, text: String): IntArray
fun cactusScoreWindow(model: Long, tokens: IntArray, start: Int, end: Int, context: Int): String
```

### VAD / RAG

```kotlin
fun cactusVad(model: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?): String
fun cactusRagQuery(model: Long, query: String, topK: Int): String
```

### Vector Index

```kotlin
fun cactusIndexInit(indexDir: String, embeddingDim: Int): Long  // throws RuntimeException
fun cactusIndexDestroy(index: Long)
fun cactusIndexAdd(index: Long, ids: IntArray, documents: Array<String>, embeddings: Array<FloatArray>, metadatas: Array<String>?): Int
fun cactusIndexDelete(index: Long, ids: IntArray): Int
fun cactusIndexGet(index: Long, ids: IntArray): String
fun cactusIndexQuery(index: Long, embedding: FloatArray, optionsJson: String?): String
fun cactusIndexCompact(index: Long): Int
```

### Telemetry

```kotlin
fun cactusSetTelemetryEnvironment(cacheDir: String)
fun cactusSetAppId(appId: String)
fun cactusTelemetryFlush()
fun cactusTelemetryShutdown()
```

### Types

```kotlin
fun interface CactusTokenCallback {
    fun onToken(token: String, tokenId: Int)
}
```

## Requirements

- Android API 24+ / arm64-v8a
- iOS 14+ / arm64 (KMP only)

## See Also

- [Cactus Engine API](/docs/cactus_engine.md) — Full C API reference underlying the Kotlin bindings
- [Cactus Index API](/docs/cactus_index.md) — Vector database API for RAG applications
- [Fine-tuning Guide](/docs/finetuning.md) — Deploy custom fine-tunes to Android
- [Swift SDK](/apple/) — Swift alternative for Apple platforms
- [Flutter SDK](/flutter/) — Cross-platform alternative using Dart
