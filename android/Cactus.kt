package com.cactus

import org.json.JSONArray
import org.json.JSONObject
import java.io.Closeable

class Cactus private constructor(private var handle: Long) : Closeable {

    companion object {
        init {
            System.loadLibrary("cactus")
        }

        @JvmStatic
        fun create(modelPath: String, corpusDir: String? = null): Cactus {
            val handle = nativeInit(modelPath, corpusDir)
            if (handle == 0L) {
                throw CactusException(nativeGetLastError().ifEmpty { "Failed to initialize model" })
            }
            return Cactus(handle)
        }

        @JvmStatic
        private external fun nativeInit(modelPath: String, corpusDir: String?): Long
        @JvmStatic
        private external fun nativeGetLastError(): String
    }

    fun complete(prompt: String, options: CompletionOptions = CompletionOptions()): CompletionResult {
        return complete(listOf(Message.user(prompt)), options)
    }

    fun complete(
        messages: List<Message>,
        options: CompletionOptions = CompletionOptions(),
        tools: List<Map<String, Any>>? = null,
        callback: TokenCallback? = null
    ): CompletionResult {
        checkHandle()
        val messagesJson = JSONArray(messages.map { it.toJson() }).toString()
        val toolsJson = tools?.let { JSONArray(it.map { t -> JSONObject(t) }).toString() }

        val responseJson = nativeComplete(handle, messagesJson, options.toJson(), toolsJson, callback)
        val json = JSONObject(responseJson)

        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }

        return json.toCompletionResult()
    }

    fun transcribe(
        audioPath: String,
        prompt: String? = null,
        language: String? = null,
        translate: Boolean = false
    ): TranscriptionResult {
        checkHandle()
        val optionsJson = JSONObject().apply {
            language?.let { put("language", it) }
            put("translate", translate)
        }.toString()

        val responseJson = nativeTranscribe(handle, audioPath, prompt, optionsJson, null)
        val json = JSONObject(responseJson)

        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }

        return json.toTranscriptionResult()
    }

    fun transcribe(
        pcmData: ByteArray,
        prompt: String? = null,
        language: String? = null,
        translate: Boolean = false
    ): TranscriptionResult {
        checkHandle()
        val optionsJson = JSONObject().apply {
            language?.let { put("language", it) }
            put("translate", translate)
        }.toString()

        val responseJson = nativeTranscribe(handle, null, prompt, optionsJson, pcmData)
        val json = JSONObject(responseJson)

        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }

        return json.toTranscriptionResult()
    }

    fun embed(text: String, normalize: Boolean = true): FloatArray {
        checkHandle()
        return nativeEmbed(handle, text, normalize)
            ?: throw CactusException(nativeGetLastError().ifEmpty { "Failed to generate embedding" })
    }

    fun ragQuery(query: String, topK: Int = 5): String {
        checkHandle()
        val responseJson = nativeRagQuery(handle, query, topK)
        val json = JSONObject(responseJson)

        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }

        return responseJson
    }

    fun reset() {
        checkHandle()
        nativeReset(handle)
    }

    fun stop() {
        checkHandle()
        nativeStop(handle)
    }

    fun tokenize(text: String): IntArray {
        checkHandle()
        return nativeTokenize(handle, text)
            ?: throw CactusException(nativeGetLastError().ifEmpty { "Failed to tokenize" })
    }

    fun scoreWindow(tokens: IntArray, start: Int, end: Int, context: Int): String {
        checkHandle()
        val responseJson = nativeScoreWindow(handle, tokens, start, end, context)
        val json = JSONObject(responseJson)
        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }
        return responseJson
    }

    fun imageEmbed(imagePath: String): FloatArray {
        checkHandle()
        return nativeImageEmbed(handle, imagePath)
            ?: throw CactusException(nativeGetLastError().ifEmpty { "Failed to generate image embedding" })
    }

    fun audioEmbed(audioPath: String): FloatArray {
        checkHandle()
        return nativeAudioEmbed(handle, audioPath)
            ?: throw CactusException(nativeGetLastError().ifEmpty { "Failed to generate audio embedding" })
    }

    fun vad(audioPath: String, options: VADOptions = VADOptions()): VADResult {
        checkHandle()
        val optionsJson = options.toJson()
        val responseJson = nativeVad(handle, audioPath, optionsJson, null)
        val json = JSONObject(responseJson)

        if (json.has("error") && !json.isNull("error")) {
            throw CactusException(json.getString("error"))
        }

        return json.toVADResult()
    }

    fun vad(pcmData: ByteArray, options: VADOptions = VADOptions()): VADResult {
        checkHandle()
        val optionsJson = options.toJson()
        val responseJson = nativeVad(handle, null, optionsJson, pcmData)
        val json = JSONObject(responseJson)

        if (json.has("error") && !json.isNull("error")) {
            throw CactusException(json.getString("error"))
        }

        return json.toVADResult()
    }

    fun createStreamTranscriber(): StreamTranscriber {
        checkHandle()
        val streamHandle = nativeStreamTranscribeInit(handle)
        if (streamHandle == 0L) {
            throw CactusException(nativeGetLastError().ifEmpty { "Failed to create stream transcriber" })
        }
        return StreamTranscriber(streamHandle)
    }

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private fun checkHandle() {
        if (handle == 0L) {
            throw CactusException("Model has been closed")
        }
    }

    private external fun nativeDestroy(handle: Long)
    private external fun nativeReset(handle: Long)
    private external fun nativeStop(handle: Long)
    private external fun nativeComplete(handle: Long, messagesJson: String, optionsJson: String?, toolsJson: String?, callback: TokenCallback?): String
    private external fun nativeTranscribe(handle: Long, audioPath: String?, prompt: String?, optionsJson: String?, pcmData: ByteArray?): String
    private external fun nativeEmbed(handle: Long, text: String, normalize: Boolean): FloatArray?
    private external fun nativeRagQuery(handle: Long, query: String, topK: Int): String
    private external fun nativeTokenize(handle: Long, text: String): IntArray?
    private external fun nativeScoreWindow(handle: Long, tokens: IntArray, start: Int, end: Int, context: Int): String
    private external fun nativeImageEmbed(handle: Long, imagePath: String): FloatArray?
    private external fun nativeAudioEmbed(handle: Long, audioPath: String): FloatArray?
    private external fun nativeVad(handle: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?): String
    private external fun nativeStreamTranscribeInit(handle: Long): Long
}

class StreamTranscriber internal constructor(private var handle: Long) : Closeable {

    /**
     * Process a chunk of PCM audio data and get intermediate transcription results.
     * @param pcmData Raw PCM audio data (16-bit signed, mono, 16kHz)
     * @return Intermediate transcription result
     */
    fun process(pcmData: ByteArray): TranscriptionResult {
        checkHandle()
        val responseJson = nativeStreamTranscribeProcess(handle, pcmData)
        val json = JSONObject(responseJson)
        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }
        return json.toTranscriptionResult()
    }

    /**
     * Stop the streaming transcription and get the final result.
     * This also releases the stream resources.
     * @return Final transcription result
     */
    fun stop(): TranscriptionResult {
        checkHandle()
        val responseJson = nativeStreamTranscribeStop(handle)
        handle = 0L  // Stream is now closed
        val json = JSONObject(responseJson)
        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }
        return json.toTranscriptionResult()
    }

    override fun close() {
        if (handle != 0L) {
            // Stop also closes the stream
            try {
                nativeStreamTranscribeStop(handle)
            } catch (_: Exception) {
                // Ignore errors during close
            }
            handle = 0L
        }
    }

    private fun checkHandle() {
        if (handle == 0L) {
            throw CactusException("Stream transcriber has been closed")
        }
    }

    private external fun nativeStreamTranscribeProcess(handle: Long, pcmData: ByteArray): String
    private external fun nativeStreamTranscribeStop(handle: Long): String
}

class CactusIndex private constructor(private var handle: Long) : Closeable {

    companion object {
        init {
            System.loadLibrary("cactus")
        }

        @JvmStatic
        fun create(indexDir: String, embeddingDim: Int): CactusIndex {
            val handle = nativeIndexInit(indexDir, embeddingDim)
            if (handle == 0L) {
                throw CactusException("Failed to initialize index")
            }
            return CactusIndex(handle)
        }

        @JvmStatic
        private external fun nativeIndexInit(indexDir: String, embeddingDim: Int): Long
    }

    fun add(
        ids: IntArray,
        documents: Array<String>,
        embeddings: Array<FloatArray>,
        metadatas: Array<String>? = null
    ) {
        checkHandle()
        val result = nativeIndexAdd(handle, ids, documents, metadatas, embeddings, embeddings[0].size)
        if (result < 0) {
            throw CactusException("Failed to add documents to index")
        }
    }

    fun delete(ids: IntArray) {
        checkHandle()
        val result = nativeIndexDelete(handle, ids)
        if (result < 0) {
            throw CactusException("Failed to delete documents from index")
        }
    }

    fun query(embedding: FloatArray, topK: Int = 5): List<IndexResult> {
        checkHandle()
        val responseJson = nativeIndexQuery(handle, embedding, topK, null)
        val json = JSONObject(responseJson)
        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }
        val results = json.getJSONArray("results")
        return (0 until results.length()).map { i ->
            val obj = results.getJSONObject(i)
            IndexResult(obj.getInt("id"), obj.getDouble("score").toFloat())
        }
    }

    fun compact() {
        checkHandle()
        val result = nativeIndexCompact(handle)
        if (result < 0) {
            throw CactusException("Failed to compact index")
        }
    }

    override fun close() {
        if (handle != 0L) {
            nativeIndexDestroy(handle)
            handle = 0L
        }
    }

    private fun checkHandle() {
        if (handle == 0L) {
            throw CactusException("Index has been closed")
        }
    }

    private external fun nativeIndexAdd(handle: Long, ids: IntArray, documents: Array<String>, metadatas: Array<String>?, embeddings: Array<FloatArray>, embeddingDim: Int): Int
    private external fun nativeIndexDelete(handle: Long, ids: IntArray): Int
    private external fun nativeIndexQuery(handle: Long, embedding: FloatArray, topK: Int, optionsJson: String?): String
    private external fun nativeIndexCompact(handle: Long): Int
    private external fun nativeIndexDestroy(handle: Long)
}

data class IndexResult(val id: Int, val score: Float)

data class Message(val role: String, val content: String) {
    companion object {
        fun system(content: String) = Message("system", content)
        fun user(content: String) = Message("user", content)
        fun assistant(content: String) = Message("assistant", content)
    }

    fun toJson(): JSONObject = JSONObject().apply {
        put("role", role)
        put("content", content)
    }
}

data class CompletionOptions(
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Int = 40,
    val maxTokens: Int = 512,
    val stopSequences: List<String> = emptyList(),
    val confidenceThreshold: Float = 0f
) {
    fun toJson(): String = JSONObject().apply {
        put("temperature", temperature)
        put("top_p", topP)
        put("top_k", topK)
        put("max_tokens", maxTokens)
        put("stop", JSONArray(stopSequences))
        put("confidence_threshold", confidenceThreshold)
    }.toString()
}

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
) {
    fun toJson(): String? {
        val options = JSONObject()
        threshold?.let { options.put("threshold", it) }
        negThreshold?.let { options.put("neg_threshold", it) }
        minSpeechDurationMs?.let { options.put("min_speech_duration_ms", it) }
        maxSpeechDurationS?.let { options.put("max_speech_duration_s", it) }
        minSilenceDurationMs?.let { options.put("min_silence_duration_ms", it) }
        speechPadMs?.let { options.put("speech_pad_ms", it) }
        windowSizeSamples?.let { options.put("window_size_samples", it) }
        samplingRate?.let { options.put("sampling_rate", it) }
        return if (options.length() > 0) options.toString() else null
    }
}

fun interface TokenCallback {
    fun onToken(token: String, tokenId: Int)
}

class CactusException(message: String) : Exception(message)

private fun JSONObject.toCompletionResult(): CompletionResult {
    val functionCalls = optJSONArray("function_calls")?.let { arr ->
        (0 until arr.length()).map { arr.getJSONObject(it).toMap() }
    }
    return CompletionResult(
        text = optString("text", ""),
        functionCalls = functionCalls,
        promptTokens = optInt("prompt_tokens", 0),
        completionTokens = optInt("completion_tokens", 0),
        timeToFirstToken = optDouble("time_to_first_token_ms", 0.0),
        totalTime = optDouble("total_time_ms", 0.0),
        prefillTokensPerSecond = optDouble("prefill_tokens_per_second", 0.0),
        decodeTokensPerSecond = optDouble("decode_tokens_per_second", 0.0),
        confidence = optDouble("confidence", 1.0),
        needsCloudHandoff = optBoolean("cloud_handoff", false)
    )
}

private fun JSONObject.toTranscriptionResult(): TranscriptionResult {
    val segments = optJSONArray("segments")?.let { arr ->
        (0 until arr.length()).map { arr.getJSONObject(it).toMap() }
    }
    return TranscriptionResult(
        text = optString("text", ""),
        segments = segments,
        totalTime = optDouble("total_time_ms", 0.0)
    )
}

private fun JSONObject.toVADResult(): VADResult {
    val segments = getJSONArray("segments").let { arr ->
        (0 until arr.length()).map { i ->
            val obj = arr.getJSONObject(i)
            VADSegment(
                start = obj.getInt("start"),
                end = obj.getInt("end")
            )
        }
    }
    return VADResult(
        segments = segments,
        totalTime = optDouble("total_time_ms", 0.0),
        ramUsage = optDouble("ram_usage_mb", 0.0)
    )
}

private fun JSONObject.toMap(): Map<String, Any> {
    val map = mutableMapOf<String, Any>()
    keys().forEach { key ->
        map[key] = when (val value = get(key)) {
            is JSONObject -> value.toMap()
            is JSONArray -> value.toList()
            JSONObject.NULL -> Unit
            else -> value
        }
    }
    return map
}

private fun JSONArray.toList(): List<Any> {
    return (0 until length()).map { i ->
        when (val value = get(i)) {
            is JSONObject -> value.toMap()
            is JSONArray -> value.toList()
            JSONObject.NULL -> Unit
            else -> value
        }
    }
}
