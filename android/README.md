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

### Basic Completion

```kotlin
import com.cactus.*

val model = Cactus.create("/path/to/model")
val result = model.complete("What is the capital of France?")
model.close()
```

### Chat Messages

```kotlin
Cactus.create(modelPath).use { model ->
    val result = model.complete(
        messages = listOf(
            Message.system("You are a helpful assistant."),
            Message.user("What is 2 + 2?")
        )
    )
    println(result.text)
}
```

### Completion Options

```kotlin
val options = CompletionOptions(
    temperature = 0.7f,
    topP = 0.9f,
    topK = 40,
    maxTokens = 256,
    stopSequences = listOf("\n\n")
)

val result = model.complete("Write a haiku:", options)
```

### Streaming Tokens

```kotlin
val result = model.complete(
    messages = listOf(Message.user("Tell me a story")),
    callback = TokenCallback { token, tokenId ->
        print(token)
    }
)
```

### Audio Transcription

```kotlin
val result = model.transcribe("/path/to/audio.wav")

val pcmData: ByteArray = ... // 16kHz mono PCM
val result = model.transcribe(pcmData)
```

### Embeddings

```kotlin
val embedding = model.embed("Hello, world!")
val imageEmbedding = model.imageEmbed("/path/to/image.jpg")
val audioEmbedding = model.audioEmbed("/path/to/audio.wav")
```

### Tokenization

```kotlin
val tokens = model.tokenize("Hello, world!")
val scores = model.scoreWindow(tokens, start = 0, end = tokens.size, context = 512)
```

### Streaming Transcription

```kotlin
model.createStreamTranscriber().use { stream ->
    stream.insert(audioChunk1)
    stream.insert(audioChunk2)
    val partial = stream.process()
    println("Partial: ${partial.text}")
    val final = stream.finalize()
    println("Final: ${final.text}")
}
```

### RAG

```kotlin
val model = Cactus.create(
    modelPath = "/path/to/model",
    corpusDir = "/path/to/documents"
)
val result = model.complete("What does the documentation say about X?")
```

### Vector Index

```kotlin
CactusIndex.create("/path/to/index", embeddingDim = 384).use { index ->
    val embeddings = arrayOf(model.embed("doc1"), model.embed("doc2"))
    index.add(
        ids = intArrayOf(1, 2),
        documents = arrayOf("Document 1", "Document 2"),
        embeddings = embeddings
    )
    val results = index.query(model.embed("search query"), topK = 5)
    results.forEach { println("ID: ${it.id}, Score: ${it.score}") }
}
```

## API Reference

### Cactus

```kotlin
object Cactus {
    fun create(modelPath: String, corpusDir: String? = null): Cactus
}

fun complete(prompt: String, options: CompletionOptions = CompletionOptions()): CompletionResult
fun complete(messages: List<Message>, options: CompletionOptions = CompletionOptions(), tools: List<Map<String, Any>>? = null, callback: TokenCallback? = null): CompletionResult
fun transcribe(audioPath: String, prompt: String? = null, language: String? = null, translate: Boolean = false): TranscriptionResult
fun transcribe(pcmData: ByteArray, prompt: String? = null, language: String? = null, translate: Boolean = false): TranscriptionResult
fun embed(text: String, normalize: Boolean = true): FloatArray
fun imageEmbed(imagePath: String): FloatArray
fun audioEmbed(audioPath: String): FloatArray
fun ragQuery(query: String, topK: Int = 5): String
fun tokenize(text: String): IntArray
fun scoreWindow(tokens: IntArray, start: Int, end: Int, context: Int): String
fun createStreamTranscriber(): StreamTranscriber
fun reset()
fun stop()
fun close()
```

### Message

```kotlin
data class Message(val role: String, val content: String) {
    companion object {
        fun system(content: String): Message
        fun user(content: String): Message
        fun assistant(content: String): Message
    }
}
```

### CompletionOptions

```kotlin
data class CompletionOptions(
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Int = 40,
    val maxTokens: Int = 512,
    val stopSequences: List<String> = emptyList(),
    val confidenceThreshold: Float = 0f
)
```

### CompletionResult

```kotlin
data class CompletionResult(
    val text: String,
    val functionCalls: List<Map<String, Any>>?,
    val promptTokens: Int,
    val completionTokens: Int,
    val timeToFirstToken: Double,
    val totalTime: Double,
    val prefillTokensPerSecond: Double,
    val decodeTokensPerSecond: Double,
    val confidence: Double,
    val needsCloudHandoff: Boolean
)
```

### TranscriptionResult

```kotlin
data class TranscriptionResult(
    val text: String,
    val segments: List<Map<String, Any>>?,
    val totalTime: Double
)
```

### TokenCallback

```kotlin
fun interface TokenCallback {
    fun onToken(token: String, tokenId: Int)
}
```

### StreamTranscriber

```kotlin
class StreamTranscriber : Closeable {
    fun insert(pcmData: ByteArray)
    fun process(language: String? = null): TranscriptionResult
    fun finalize(): TranscriptionResult
    fun close()
}
```

### CactusIndex

```kotlin
class CactusIndex : Closeable {
    companion object {
        fun create(indexDir: String, embeddingDim: Int): CactusIndex
    }

    fun add(ids: IntArray, documents: Array<String>, embeddings: Array<FloatArray>, metadatas: Array<String>? = null)
    fun delete(ids: IntArray)
    fun query(embedding: FloatArray, topK: Int = 5): List<IndexResult>
    fun compact()
    fun close()
}

data class IndexResult(val id: Int, val score: Float)
```

## Requirements

- Android API 24+ / arm64-v8a
- iOS 14+ / arm64 (KMP only)
