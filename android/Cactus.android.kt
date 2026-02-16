package com.cactus

import org.json.JSONArray
import org.json.JSONObject

actual class Cactus private constructor(private var handle: Long) : AutoCloseable {

    actual companion object {
        init {
            System.loadLibrary("cactus")
        }

        actual fun create(modelPath: String, corpusDir: String?): Cactus {
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

    actual fun complete(prompt: String, options: CompletionOptions): CompletionResult {
        return complete(listOf(Message.user(prompt)), options, null, null)
    }

    actual fun complete(
        messages: List<Message>,
        options: CompletionOptions,
        tools: List<Map<String, Any>>?,
        callback: TokenCallback?
    ): CompletionResult {
        checkHandle()
        val messagesJson = JSONArray(messages.map { it.toJson() }).toString()
        val optionsJson = options.toJson()
        val toolsJson = tools?.let { JSONArray(it.map { t -> JSONObject(t) }).toString() }

        val responseJson = nativeComplete(handle, messagesJson, optionsJson, toolsJson, callback)
        val json = JSONObject(responseJson)

        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }

        return json.toCompletionResult()
    }

    actual fun transcribe(
        audioPath: String,
        prompt: String?,
        language: String?,
        translate: Boolean
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

    actual fun transcribe(
        pcmData: ByteArray,
        prompt: String?,
        language: String?,
        translate: Boolean
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

    actual fun embed(text: String, normalize: Boolean): FloatArray {
        checkHandle()
        return nativeEmbed(handle, text, normalize)
            ?: throw CactusException(nativeGetLastError().ifEmpty { "Failed to generate embedding" })
    }

    actual fun ragQuery(query: String, topK: Int): String {
        checkHandle()
        val responseJson = nativeRagQuery(handle, query, topK)
        val json = JSONObject(responseJson)

        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }

        return responseJson
    }

    actual fun tokenize(text: String): IntArray {
        checkHandle()
        return nativeTokenize(handle, text)
            ?: throw CactusException(nativeGetLastError().ifEmpty { "Failed to tokenize" })
    }

    actual fun scoreWindow(tokens: IntArray, start: Int, end: Int, context: Int): String {
        checkHandle()
        val responseJson = nativeScoreWindow(handle, tokens, start, end, context)
        val json = JSONObject(responseJson)
        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }
        return responseJson
    }

    actual fun imageEmbed(imagePath: String): FloatArray {
        checkHandle()
        return nativeImageEmbed(handle, imagePath)
            ?: throw CactusException(nativeGetLastError().ifEmpty { "Failed to generate image embedding" })
    }

    actual fun audioEmbed(audioPath: String): FloatArray {
        checkHandle()
        return nativeAudioEmbed(handle, audioPath)
            ?: throw CactusException(nativeGetLastError().ifEmpty { "Failed to generate audio embedding" })
    }

    actual fun vad(audioPath: String, options: VADOptions): VADResult {
        checkHandle()
        val optionsJson = options.toJson()
        val responseJson = nativeVad(handle, audioPath, optionsJson, null)
        val json = JSONObject(responseJson)

        if (json.has("error") && !json.isNull("error")) {
            throw CactusException(json.getString("error"))
        }

        return json.toVADResult()
    }

    actual fun vad(pcmData: ByteArray, options: VADOptions): VADResult {
        checkHandle()
        val optionsJson = options.toJson()
        val responseJson = nativeVad(handle, null, optionsJson, pcmData)
        val json = JSONObject(responseJson)

        if (json.has("error") && !json.isNull("error")) {
            throw CactusException(json.getString("error"))
        }

        return json.toVADResult()
    }

    actual fun createStreamTranscriber(): StreamTranscriber {
        checkHandle()
        val streamHandle = nativeStreamTranscribeInit(handle)
        if (streamHandle == 0L) {
            throw CactusException(nativeGetLastError().ifEmpty { "Failed to create stream transcriber" })
        }
        return StreamTranscriber(streamHandle)
    }

    actual fun reset() {
        checkHandle()
        nativeReset(handle)
    }

    actual fun stop() {
        checkHandle()
        nativeStop(handle)
    }

    actual override fun close() {
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

actual class StreamTranscriber internal constructor(private var handle: Long) : AutoCloseable {

    actual fun insert(pcmData: ByteArray) {
        checkHandle()
        val result = nativeStreamTranscribeInsert(handle, pcmData)
        if (result < 0) {
            throw CactusException("Failed to insert audio data")
        }
    }

    actual fun process(language: String?): TranscriptionResult {
        checkHandle()
        val optionsJson = language?.let { JSONObject().put("language", it).toString() }
        val responseJson = nativeStreamTranscribeProcess(handle, optionsJson)
        val json = JSONObject(responseJson)
        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }
        return json.toTranscriptionResult()
    }

    actual fun finalize(): TranscriptionResult {
        checkHandle()
        val responseJson = nativeStreamTranscribeFinalize(handle)
        val json = JSONObject(responseJson)
        if (json.has("error")) {
            throw CactusException(json.getString("error"))
        }
        return json.toTranscriptionResult()
    }

    actual override fun close() {
        if (handle != 0L) {
            nativeStreamTranscribeDestroy(handle)
            handle = 0L
        }
    }

    private fun checkHandle() {
        if (handle == 0L) {
            throw CactusException("Stream transcriber has been closed")
        }
    }

    private external fun nativeStreamTranscribeInsert(handle: Long, pcmData: ByteArray): Int
    private external fun nativeStreamTranscribeProcess(handle: Long, optionsJson: String?): String
    private external fun nativeStreamTranscribeFinalize(handle: Long): String
    private external fun nativeStreamTranscribeDestroy(handle: Long)
}

actual class CactusIndex private constructor(private var handle: Long) : AutoCloseable {

    actual companion object {
        init {
            System.loadLibrary("cactus")
        }

        actual fun create(indexDir: String, embeddingDim: Int): CactusIndex {
            val handle = nativeIndexInit(indexDir, embeddingDim)
            if (handle == 0L) {
                throw CactusException("Failed to initialize index")
            }
            return CactusIndex(handle)
        }

        @JvmStatic
        private external fun nativeIndexInit(indexDir: String, embeddingDim: Int): Long
    }

    actual fun add(ids: IntArray, documents: Array<String>, embeddings: Array<FloatArray>, metadatas: Array<String>?) {
        checkHandle()
        val result = nativeIndexAdd(handle, ids, documents, metadatas, embeddings, embeddings[0].size)
        if (result < 0) {
            throw CactusException("Failed to add documents to index")
        }
    }

    actual fun delete(ids: IntArray) {
        checkHandle()
        val result = nativeIndexDelete(handle, ids)
        if (result < 0) {
            throw CactusException("Failed to delete documents from index")
        }
    }

    actual fun query(embedding: FloatArray, topK: Int): List<IndexResult> {
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

    actual fun compact() {
        checkHandle()
        val result = nativeIndexCompact(handle)
        if (result < 0) {
            throw CactusException("Failed to compact index")
        }
    }

    actual override fun close() {
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

private fun Message.toJson(): JSONObject = JSONObject().apply {
    put("role", role)
    put("content", content)
}

private fun CompletionOptions.toJson(): String = JSONObject().apply {
    put("temperature", temperature)
    put("top_p", topP)
    put("top_k", topK)
    put("max_tokens", maxTokens)
    put("stop", JSONArray(stopSequences))
    put("confidence_threshold", confidenceThreshold)
}.toString()

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

private fun VADOptions.toJson(): String? {
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
