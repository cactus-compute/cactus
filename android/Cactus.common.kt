package com.cactus

expect class Cactus : AutoCloseable {
    companion object {
        fun create(modelPath: String, corpusDir: String? = null): Cactus
    }

    fun complete(prompt: String, options: CompletionOptions = CompletionOptions()): CompletionResult
    fun complete(
        messages: List<Message>,
        options: CompletionOptions = CompletionOptions(),
        tools: List<Map<String, Any>>? = null,
        callback: TokenCallback? = null
    ): CompletionResult

    fun transcribe(
        audioPath: String,
        prompt: String? = null,
        language: String? = null,
        translate: Boolean = false
    ): TranscriptionResult

    fun transcribe(
        pcmData: ByteArray,
        prompt: String? = null,
        language: String? = null,
        translate: Boolean = false
    ): TranscriptionResult

    fun embed(text: String, normalize: Boolean = true): FloatArray
    fun imageEmbed(imagePath: String): FloatArray
    fun audioEmbed(audioPath: String): FloatArray
    fun ragQuery(query: String, topK: Int = 5): String
    fun tokenize(text: String): IntArray
    fun scoreWindow(tokens: IntArray, start: Int, end: Int, context: Int): String
    fun vad(audioPath: String, options: VADOptions = VADOptions()): VADResult
    fun vad(pcmData: ByteArray, options: VADOptions = VADOptions()): VADResult
    fun createStreamTranscriber(): StreamTranscriber
    fun reset()
    fun stop()
    override fun close()
}

expect class StreamTranscriber : AutoCloseable {
    fun insert(pcmData: ByteArray)
    fun process(language: String? = null): TranscriptionResult
    fun finalize(): TranscriptionResult
    override fun close()
}

expect class CactusIndex : AutoCloseable {
    companion object {
        fun create(indexDir: String, embeddingDim: Int): CactusIndex
    }

    fun add(ids: IntArray, documents: Array<String>, embeddings: Array<FloatArray>, metadatas: Array<String>? = null)
    fun delete(ids: IntArray)
    fun query(embedding: FloatArray, topK: Int = 5): List<IndexResult>
    fun compact()
    override fun close()
}

data class Message(val role: String, val content: String) {
    companion object {
        fun system(content: String) = Message("system", content)
        fun user(content: String) = Message("user", content)
        fun assistant(content: String) = Message("assistant", content)
    }
}

data class CompletionOptions(
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Int = 40,
    val maxTokens: Int = 512,
    val stopSequences: List<String> = emptyList(),
    val confidenceThreshold: Float = 0f
)

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

data class TranscriptionResult(
    val text: String,
    val segments: List<Map<String, Any>>?,
    val totalTime: Double
)

data class VADSegment(
    val start: Int,
    val end: Int
)

data class VADResult(
    val segments: List<VADSegment>,
    val totalTime: Double,
    val ramUsage: Double
)

data class VADOptions(
    val threshold: Float? = null,
    val negThreshold: Float? = null,
    val minSpeechDurationMs: Int? = null,
    val maxSpeechDurationS: Float? = null,
    val minSilenceDurationMs: Int? = null,
    val speechPadMs: Int? = null,
    val windowSizeSamples: Int? = null,
    val samplingRate: Int? = null
)

fun interface TokenCallback {
    fun onToken(token: String, tokenId: Int)
}

data class IndexResult(val id: Int, val score: Float)

class CactusException(message: String) : Exception(message)
