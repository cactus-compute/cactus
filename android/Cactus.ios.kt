package com.cactus

import cactus.*
import kotlinx.cinterop.*
import kotlinx.serialization.json.*
import platform.Foundation.NSLog

@OptIn(ExperimentalForeignApi::class)
actual class Cactus private constructor(private var handle: COpaquePointer?) : AutoCloseable {

    actual companion object {
        actual fun create(modelPath: String, corpusDir: String?): Cactus {
            val handle = cactus_init(modelPath, corpusDir)
            if (handle == null) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }
            return Cactus(handle)
        }
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
        memScoped {
            val buffer = allocArray<ByteVar>(65536)
            val messagesJson = serializeMessages(messages)
            val optionsJson = serializeOptions(options)
            val toolsJson = tools?.let { serializeTools(it) }

            val result = cactus_complete(
                handle,
                messagesJson,
                buffer,
                65536u,
                optionsJson,
                toolsJson,
                null,
                null
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            val response = buffer.toKString()
            return parseCompletionResult(response)
        }
    }

    actual fun transcribe(
        audioPath: String,
        prompt: String?,
        language: String?,
        translate: Boolean
    ): TranscriptionResult {
        checkHandle()
        memScoped {
            val buffer = allocArray<ByteVar>(65536)
            val optionsJson = buildJsonObject {
                language?.let { put("language", it) }
                put("translate", translate)
            }.toString()

            val result = cactus_transcribe(
                handle,
                audioPath,
                prompt,
                buffer,
                65536u,
                optionsJson,
                null,
                null,
                null,
                0u
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            return parseTranscriptionResult(buffer.toKString())
        }
    }

    actual fun transcribe(
        pcmData: ByteArray,
        prompt: String?,
        language: String?,
        translate: Boolean
    ): TranscriptionResult {
        checkHandle()
        memScoped {
            val buffer = allocArray<ByteVar>(65536)
            val optionsJson = buildJsonObject {
                language?.let { put("language", it) }
                put("translate", translate)
            }.toString()

            val pcmPtr = pcmData.refTo(0).getPointer(this)

            val result = cactus_transcribe(
                handle,
                null,
                prompt,
                buffer,
                65536u,
                optionsJson,
                null,
                null,
                pcmPtr.reinterpret(),
                pcmData.size.toULong()
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            return parseTranscriptionResult(buffer.toKString())
        }
    }

    actual fun embed(text: String, normalize: Boolean): FloatArray {
        checkHandle()
        memScoped {
            val buffer = allocArray<FloatVar>(4096)
            val dimPtr = alloc<ULongVar>()

            val result = cactus_embed(
                handle,
                text,
                buffer,
                4096u,
                dimPtr.ptr,
                normalize
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            val dim = dimPtr.value.toInt()
            return FloatArray(dim) { buffer[it] }
        }
    }

    actual fun ragQuery(query: String, topK: Int): String {
        checkHandle()
        memScoped {
            val buffer = allocArray<ByteVar>(65536)

            val result = cactus_rag_query(
                handle,
                query,
                buffer,
                65536u,
                topK.toULong()
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            return buffer.toKString()
        }
    }

    actual fun tokenize(text: String): IntArray {
        checkHandle()
        memScoped {
            val buffer = allocArray<UIntVar>(8192)
            val tokenLen = alloc<ULongVar>()

            val result = cactus_tokenize(
                handle,
                text,
                buffer,
                8192u,
                tokenLen.ptr
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            return IntArray(tokenLen.value.toInt()) { buffer[it].toInt() }
        }
    }

    actual fun scoreWindow(tokens: IntArray, start: Int, end: Int, context: Int): String {
        checkHandle()
        memScoped {
            val buffer = allocArray<ByteVar>(65536)
            val tokenBuffer = allocArray<UIntVar>(tokens.size)
            tokens.forEachIndexed { i, v -> tokenBuffer[i] = v.toUInt() }

            val result = cactus_score_window(
                handle,
                tokenBuffer,
                tokens.size.toULong(),
                start.toULong(),
                end.toULong(),
                context.toULong(),
                buffer,
                65536u
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            return buffer.toKString()
        }
    }

    actual fun imageEmbed(imagePath: String): FloatArray {
        checkHandle()
        memScoped {
            val buffer = allocArray<FloatVar>(4096)
            val dimPtr = alloc<ULongVar>()

            val result = cactus_image_embed(
                handle,
                imagePath,
                buffer,
                4096u,
                dimPtr.ptr
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            return FloatArray(dimPtr.value.toInt()) { buffer[it] }
        }
    }

    actual fun audioEmbed(audioPath: String): FloatArray {
        checkHandle()
        memScoped {
            val buffer = allocArray<FloatVar>(4096)
            val dimPtr = alloc<ULongVar>()

            val result = cactus_audio_embed(
                handle,
                audioPath,
                buffer,
                4096u,
                dimPtr.ptr
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            return FloatArray(dimPtr.value.toInt()) { buffer[it] }
        }
    }

    actual fun vad(audioPath: String, options: VADOptions): VADResult {
        checkHandle()
        memScoped {
            val buffer = allocArray<ByteVar>(65536)
            val optionsJson = serializeVADOptions(options)

            val result = cactus_vad(
                handle,
                audioPath,
                buffer,
                65536u,
                optionsJson,
                null,
                0u
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            return parseVADResult(buffer.toKString())
        }
    }

    actual fun vad(pcmData: ByteArray, options: VADOptions): VADResult {
        checkHandle()
        memScoped {
            val buffer = allocArray<ByteVar>(65536)
            val optionsJson = serializeVADOptions(options)
            val pcmPtr = pcmData.refTo(0).getPointer(this)

            val result = cactus_vad(
                handle,
                null,
                buffer,
                65536u,
                optionsJson,
                pcmPtr.reinterpret(),
                pcmData.size.toULong()
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            return parseVADResult(buffer.toKString())
        }
    }

    actual fun createStreamTranscriber(): StreamTranscriber {
        checkHandle()
        val streamHandle = cactus_stream_transcribe_init(handle)
        if (streamHandle == null) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw CactusException(error)
        }
        return StreamTranscriber(streamHandle)
    }

    actual fun reset() {
        checkHandle()
        cactus_reset(handle)
    }

    actual fun stop() {
        checkHandle()
        cactus_stop(handle)
    }

    actual override fun close() {
        handle?.let { cactus_destroy(it) }
        handle = null
    }

    private fun checkHandle() {
        if (handle == null) throw CactusException("Model has been closed")
    }

    private fun serializeMessages(messages: List<Message>): String {
        return buildJsonArray {
            messages.forEach { msg ->
                addJsonObject {
                    put("role", msg.role)
                    put("content", msg.content)
                }
            }
        }.toString()
    }

    private fun serializeOptions(options: CompletionOptions): String {
        return buildJsonObject {
            put("temperature", options.temperature)
            put("top_p", options.topP)
            put("top_k", options.topK)
            put("max_tokens", options.maxTokens)
            putJsonArray("stop") { options.stopSequences.forEach { add(it) } }
            put("confidence_threshold", options.confidenceThreshold)
        }.toString()
    }

    private fun serializeTools(tools: List<Map<String, Any>>): String {
        return Json.encodeToString(tools)
    }

    private fun parseCompletionResult(json: String): CompletionResult {
        val obj = Json.parseToJsonElement(json).jsonObject
        return CompletionResult(
            text = obj["text"]?.jsonPrimitive?.contentOrNull ?: "",
            functionCalls = obj["function_calls"]?.jsonArray?.map { it.jsonObject.toMap() },
            promptTokens = obj["prompt_tokens"]?.jsonPrimitive?.intOrNull ?: 0,
            completionTokens = obj["completion_tokens"]?.jsonPrimitive?.intOrNull ?: 0,
            timeToFirstToken = obj["time_to_first_token_ms"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
            totalTime = obj["total_time_ms"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
            prefillTokensPerSecond = obj["prefill_tokens_per_second"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
            decodeTokensPerSecond = obj["decode_tokens_per_second"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
            confidence = obj["confidence"]?.jsonPrimitive?.doubleOrNull ?: 1.0,
            needsCloudHandoff = obj["cloud_handoff"]?.jsonPrimitive?.booleanOrNull ?: false
        )
    }

    private fun parseTranscriptionResult(json: String): TranscriptionResult {
        val obj = Json.parseToJsonElement(json).jsonObject
        return TranscriptionResult(
            text = obj["text"]?.jsonPrimitive?.contentOrNull ?: "",
            segments = obj["segments"]?.jsonArray?.map { it.jsonObject.toMap() },
            totalTime = obj["total_time_ms"]?.jsonPrimitive?.doubleOrNull ?: 0.0
        )
    }

    private fun serializeVADOptions(options: VADOptions): String? {
        val obj = buildJsonObject {
            options.threshold?.let { put("threshold", it) }
            options.negThreshold?.let { put("neg_threshold", it) }
            options.minSpeechDurationMs?.let { put("min_speech_duration_ms", it) }
            options.maxSpeechDurationS?.let { put("max_speech_duration_s", it) }
            options.minSilenceDurationMs?.let { put("min_silence_duration_ms", it) }
            options.speechPadMs?.let { put("speech_pad_ms", it) }
            options.windowSizeSamples?.let { put("window_size_samples", it) }
            options.samplingRate?.let { put("sampling_rate", it) }
        }
        return if (obj.isEmpty()) null else obj.toString()
    }

    private fun parseVADResult(json: String): VADResult {
        val obj = Json.parseToJsonElement(json).jsonObject
        val segments = obj["segments"]?.jsonArray?.map { segObj ->
            val seg = segObj.jsonObject
            VADSegment(
                start = seg["start"]?.jsonPrimitive?.intOrNull ?: 0,
                end = seg["end"]?.jsonPrimitive?.intOrNull ?: 0
            )
        } ?: emptyList()
        return VADResult(
            segments = segments,
            totalTime = obj["total_time_ms"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
            ramUsage = obj["ram_usage_mb"]?.jsonPrimitive?.doubleOrNull ?: 0.0
        )
    }

    private fun JsonObject.toMap(): Map<String, Any> {
        return entries.associate { (k, v) ->
            k to when (v) {
                is JsonPrimitive -> v.contentOrNull ?: v.toString()
                is JsonObject -> v.toMap()
                is JsonArray -> v.map { it.toString() }
                else -> v.toString()
            }
        }
    }
}

@OptIn(ExperimentalForeignApi::class)
actual class StreamTranscriber internal constructor(private var handle: COpaquePointer?) : AutoCloseable {

    actual fun insert(pcmData: ByteArray) {
        checkHandle()
        memScoped {
            val pcmPtr = pcmData.refTo(0).getPointer(this)
            val result = cactus_stream_transcribe_insert(
                handle,
                pcmPtr.reinterpret(),
                pcmData.size.toULong()
            )
            if (result < 0) {
                throw CactusException("Failed to insert audio data")
            }
        }
    }

    actual fun process(language: String?): TranscriptionResult {
        checkHandle()
        memScoped {
            val buffer = allocArray<ByteVar>(65536)
            val optionsJson = language?.let {
                buildJsonObject { put("language", it) }.toString()
            }

            val result = cactus_stream_transcribe_process(
                handle,
                buffer,
                65536u,
                optionsJson
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            return parseTranscriptionResult(buffer.toKString())
        }
    }

    actual fun finalize(): TranscriptionResult {
        checkHandle()
        memScoped {
            val buffer = allocArray<ByteVar>(65536)

            val result = cactus_stream_transcribe_finalize(
                handle,
                buffer,
                65536u
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            return parseTranscriptionResult(buffer.toKString())
        }
    }

    actual override fun close() {
        handle?.let { cactus_stream_transcribe_destroy(it) }
        handle = null
    }

    private fun checkHandle() {
        if (handle == null) throw CactusException("Stream transcriber has been closed")
    }

    private fun parseTranscriptionResult(json: String): TranscriptionResult {
        val obj = Json.parseToJsonElement(json).jsonObject
        return TranscriptionResult(
            text = obj["text"]?.jsonPrimitive?.contentOrNull ?: "",
            segments = obj["segments"]?.jsonArray?.map { it.jsonObject.toMap() },
            totalTime = obj["total_time_ms"]?.jsonPrimitive?.doubleOrNull ?: 0.0
        )
    }

    private fun JsonObject.toMap(): Map<String, Any> {
        return entries.associate { (k, v) ->
            k to when (v) {
                is JsonPrimitive -> v.contentOrNull ?: v.toString()
                is JsonObject -> v.toMap()
                is JsonArray -> v.map { it.toString() }
                else -> v.toString()
            }
        }
    }
}

@OptIn(ExperimentalForeignApi::class)
actual class CactusIndex private constructor(private var handle: COpaquePointer?) : AutoCloseable {

    actual companion object {
        actual fun create(indexDir: String, embeddingDim: Int): CactusIndex {
            val handle = cactus_index_init(indexDir, embeddingDim.toULong())
            if (handle == null) {
                throw CactusException("Failed to initialize index")
            }
            return CactusIndex(handle)
        }
    }

    actual fun add(ids: IntArray, documents: Array<String>, embeddings: Array<FloatArray>, metadatas: Array<String>?) {
        checkHandle()
        memScoped {
            val idPtr = allocArray<IntVar>(ids.size)
            ids.forEachIndexed { i, v -> idPtr[i] = v }

            val docPtrs = allocArray<CPointerVar<ByteVar>>(documents.size)
            documents.forEachIndexed { i, doc -> docPtrs[i] = doc.cstr.ptr }

            val metaPtrs = metadatas?.let {
                val ptrs = allocArray<CPointerVar<ByteVar>>(it.size)
                it.forEachIndexed { i, meta -> ptrs[i] = meta.cstr.ptr }
                ptrs
            }

            val embPtrs = allocArray<CPointerVar<FloatVar>>(embeddings.size)
            embeddings.forEachIndexed { i, emb ->
                val embArr = allocArray<FloatVar>(emb.size)
                emb.forEachIndexed { j, v -> embArr[j] = v }
                embPtrs[i] = embArr
            }

            val result = cactus_index_add(
                handle,
                idPtr,
                docPtrs,
                metaPtrs,
                embPtrs,
                ids.size.toULong(),
                embeddings[0].size.toULong()
            )

            if (result < 0) {
                throw CactusException("Failed to add documents to index")
            }
        }
    }

    actual fun delete(ids: IntArray) {
        checkHandle()
        memScoped {
            val idPtr = allocArray<IntVar>(ids.size)
            ids.forEachIndexed { i, v -> idPtr[i] = v }

            val result = cactus_index_delete(handle, idPtr, ids.size.toULong())
            if (result < 0) {
                throw CactusException("Failed to delete documents from index")
            }
        }
    }

    actual fun query(embedding: FloatArray, topK: Int): List<IndexResult> {
        checkHandle()
        memScoped {
            val embArr = allocArray<FloatVar>(embedding.size)
            embedding.forEachIndexed { i, v -> embArr[i] = v }
            val embPtr = alloc<CPointerVar<FloatVar>>()
            embPtr.value = embArr

            val idBuffer = allocArray<IntVar>(topK)
            val scoreBuffer = allocArray<FloatVar>(topK)
            val idBufferSize = alloc<ULongVar>()
            val scoreBufferSize = alloc<ULongVar>()
            idBufferSize.value = topK.toULong()
            scoreBufferSize.value = topK.toULong()

            val idPtrPtr = alloc<CPointerVar<IntVar>>()
            idPtrPtr.value = idBuffer
            val scorePtrPtr = alloc<CPointerVar<FloatVar>>()
            scorePtrPtr.value = scoreBuffer

            val result = cactus_index_query(
                handle,
                embPtr.ptr,
                1u,
                embedding.size.toULong(),
                null,
                idPtrPtr.ptr,
                idBufferSize.ptr,
                scorePtrPtr.ptr,
                scoreBufferSize.ptr
            )

            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw CactusException(error)
            }

            return (0 until idBufferSize.value.toInt()).map { i ->
                IndexResult(idBuffer[i], scoreBuffer[i])
            }
        }
    }

    actual fun compact() {
        checkHandle()
        val result = cactus_index_compact(handle)
        if (result < 0) {
            throw CactusException("Failed to compact index")
        }
    }

    actual override fun close() {
        handle?.let { cactus_index_destroy(it) }
        handle = null
    }

    private fun checkHandle() {
        if (handle == null) throw CactusException("Index has been closed")
    }
}
