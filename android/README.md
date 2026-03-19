---
title: "Cactus Android & Kotlin Multiplatform SDK"
description: "Kotlin API for running AI models on-device on Android and iOS via Kotlin Multiplatform. Supports completion, transcription, embeddings, RAG, and tool calling."
keywords: ["Android SDK", "Kotlin Multiplatform", "on-device AI", "mobile inference", "JNI", "KMP"]
---

# Cactus for Android & Kotlin Multiplatform

Run AI models on-device with a simple Kotlin API.

> **Model weights:** Pre-converted weights for all supported models at [huggingface.co/Cactus-Compute](https://huggingface.co/Cactus-Compute).

## Building

<!-- --8<-- [start:install] -->
```bash
git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup
cactus build --android
```

Build output: `android/libcactus.so` (and `android/libcactus.a`)
<!-- --8<-- [end:install] -->

see the main [README.md](../README.md) for how to use CLI & download weight

### Vendored libcurl (device builds)

To bundle libcurl locally for Android device testing, place artifacts using:

`libs/curl/android/arm64-v8a/libcurl.a` and `libs/curl/include/curl/*.h`

The build auto-detects `libs/curl`. You can override with:

```bash
CACTUS_CURL_ROOT=/absolute/path/to/curl cactus build --android
```

## Integration

<!-- --8<-- [start:integration] -->
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
<!-- --8<-- [end:integration] -->

## Usage

Handles are plain `Long` values (C pointers). All functions are top-level.

### Basic Completion

<!-- --8<-- [start:example] -->
```kotlin
import com.cactus.*

val model = cactusInit("/path/to/model", null, false)
val messages = """[{"role":"user","content":"What is the capital of France?"}]"""
val resultJson = cactusComplete(model, messages, null, null, null)
println(resultJson)
cactusDestroy(model)
```
<!-- --8<-- [end:example] -->

For vision models (LFM2-VL, LFM2.5-VL), add `"images": ["path/to/image.png"]` to any message. See [Engine API](/docs/cactus_engine.md) for details.

### Completion with Options and Streaming

```kotlin
import com.cactus.*

val options = """{"max_tokens":256,"temperature":0.7}"""

val resultJson = cactusComplete(model, messages, options, null) { token, _ ->
    print(token)
}
println(resultJson)
```

### Prefill

Pre-processes input text and populates the KV cache without generating output tokens. This reduces latency for subsequent calls to `cactusComplete`.

```kotlin
fun cactusPrefill(
    model: Long,
    messagesJson: String,
    optionsJson: String?,
    toolsJson: String?
): String
```

```kotlin
val tools = """[
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
]"""

val messages = """[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": "<|tool_call_start|>get_weather(location=\"Paris\")<|tool_call_end|>"},
    {"role": "tool", "content": "{\"name\": \"get_weather\", \"content\": \"Sunny, 72°F\"}"},
    {"role": "assistant", "content": "It's sunny and 72°F in Paris!"}
]"""

val resultJson = cactusPrefill(model, messages, null, tools)

val completionMessages = """[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": "<|tool_call_start|>get_weather(location=\"Paris\")<|tool_call_end|>"},
    {"role": "tool", "content": "{\"name\": \"get_weather\", \"content\": \"Sunny, 72°F\"}"},
    {"role": "assistant", "content": "It's sunny and 72°F in Paris!"},
    {"role": "user", "content": "What about SF?"}
]"""

val completion = cactusComplete(model, completionMessages, null, tools, null)
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

```kotlin
import com.cactus.*

// From file
val resultJson = cactusTranscribe(model, "/path/to/audio.wav", null, null, null, null)
println(resultJson)

// From PCM data (16 kHz mono)
val pcmData: ByteArray = ...
val resultJson2 = cactusTranscribe(model, null, null, null, null, pcmData)
println(resultJson2)
```

`segments` contains timestamps (seconds): phrase-level for Whisper, word-level for Parakeet TDT, one segment per transcription window for Parakeet CTC and Moonshine (consecutive VAD speech regions up to 30s).

```kotlin
import org.json.JSONObject

val result = JSONObject(resultJson)
val segments = result.getJSONArray("segments")
for (i in 0 until segments.length()) {
    val seg = segments.getJSONObject(i)
    println("[${seg.getDouble("start")}s - ${seg.getDouble("end")}s] ${seg.getString("text")}")
}
```

**Custom vocabulary** biases the decoder toward domain-specific words (supported for Whisper and Moonshine models). Pass `custom_vocabulary` and `vocabulary_boost` in the options JSON:

```kotlin
val options = """{"custom_vocabulary": ["Omeprazole", "HIPAA", "Cactus"], "vocabulary_boost": 3.0}"""
val result = cactusTranscribe(model, "/path/to/audio.wav", "", options, null, null)
```

### Streaming Transcription

```kotlin
val stream = cactusStreamTranscribeStart(model, null)
val partial = cactusStreamTranscribeProcess(stream, audioChunk)
val final_  = cactusStreamTranscribeStop(stream)
```

Streaming also accepts `custom_vocabulary` in the options passed to `cactusStreamTranscribeStart`. The bias is applied for the lifetime of the stream session.

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
val index = cactusIndexInit("/path/to/index", 3)

cactusIndexAdd(
    index,
    intArrayOf(1, 2),
    arrayOf("Document 1", "Document 2"),
    arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f), floatArrayOf(0.4f, 0.5f, 0.6f)),
    null
)

val resultsJson = cactusIndexQuery(index, floatArrayOf(0.1f, 0.2f, 0.3f), null)
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

### Prefill

```kotlin
fun cactusPrefill(
    model: Long,
    messagesJson: String,
    optionsJson: String?,
    toolsJson: String?
): String
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

### Detect Language

```kotlin
fun cactusDetectLanguage(model: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?): String
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

### Logging

```kotlin
fun cactusLogSetLevel(level: Int)  // 0=DEBUG 1=INFO 2=WARN 3=ERROR 4=NONE
fun cactusLogSetCallback(callback: CactusLogCallback?)
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

fun interface CactusLogCallback {
    fun onLog(level: Int, component: String, message: String)
}
```

## Requirements

- Android API 21+ / arm64-v8a
- iOS 13+ / arm64 (KMP only)

## See Also

- [Cactus Engine API](/docs/cactus_engine.md) — Full C API reference underlying the Kotlin bindings
- [Cactus Index API](/docs/cactus_index.md) — Vector database API for RAG applications
- [Fine-tuning Guide](/docs/finetuning.md) — Deploy custom fine-tunes to Android
- [Swift SDK](/apple/) — Swift alternative for Apple platforms
- [Flutter SDK](/flutter/) — Cross-platform alternative using Dart
