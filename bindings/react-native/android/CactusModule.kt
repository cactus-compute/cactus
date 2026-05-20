package com.cactus.reactnative

import android.util.Base64
import com.cactus.CactusJNI
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableArray
import org.json.JSONArray
import org.json.JSONObject

class CactusModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {
    companion object {
        private const val DEFAULT_BUFFER_SIZE = 65536
        private const val LARGE_BUFFER_SIZE = 1 shl 20
        private const val DEFAULT_EMBEDDING_BUFFER_SIZE = 4096
        private const val DEFAULT_TOKEN_BUFFER_SIZE = 8192
        private const val DEFAULT_INDEX_RESULT_CAPACITY = 1000
        private const val DEFAULT_INDEX_DOC_BUFFER_SIZE = 4096
        private const val DEFAULT_INDEX_EMBED_BUFFER_SIZE = 4096
    }

    override fun getName(): String = "Cactus"

    private fun parseHandle(handle: String, promise: Promise): Long? {
        val parsed = handle.toLongOrNull()
        if (parsed == null) {
            promise.reject("CACTUS_ERROR", "Invalid native handle")
        }
        return parsed
    }

    private fun fail(promise: Promise, defaultMessage: String) {
        val message = CactusJNI.nativeGetLastError().ifEmpty { defaultMessage }
        promise.reject("CACTUS_ERROR", message)
    }

    private fun decodeNullTerminatedUtf8(buffer: ByteArray): String {
        val end = buffer.indexOf(0).let { if (it >= 0) it else buffer.size }
        return buffer.copyOf(end).toString(Charsets.UTF_8)
    }

    private fun decodeBase64OrNull(value: String?): ByteArray? =
        if (value == null) null else Base64.decode(value, Base64.DEFAULT)

    private fun readableArrayToIntArray(values: ReadableArray): IntArray =
        IntArray(values.size()) { index -> values.getDouble(index).toInt() }

    private fun readableArrayToFloatArray(values: ReadableArray): FloatArray =
        FloatArray(values.size()) { index -> values.getDouble(index).toFloat() }

    private fun readableArrayToStringArray(values: ReadableArray): Array<String> =
        Array(values.size()) { index -> values.getString(index) ?: "" }

    private fun readableNestedFloatArrays(values: ReadableArray): Array<FloatArray> =
        Array(values.size()) { index ->
            readableArrayToFloatArray(values.getArray(index) ?: throw IllegalArgumentException("embeddings[$index] is null"))
        }

    @ReactMethod
    fun init(modelPath: String, corpusDir: String?, cacheIndex: Boolean, promise: Promise) {
        val handle = CactusJNI.nativeInit(modelPath, corpusDir, cacheIndex)
        if (handle == 0L) {
            fail(promise, "Failed to initialize model")
            return
        }
        promise.resolve(handle.toString())
    }

    @ReactMethod
    fun destroy(handle: String, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        CactusJNI.nativeDestroy(nativeHandle)
        promise.resolve(null)
    }

    @ReactMethod
    fun reset(handle: String, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        CactusJNI.nativeReset(nativeHandle)
        promise.resolve(null)
    }

    @ReactMethod
    fun stop(handle: String, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        CactusJNI.nativeStop(nativeHandle)
        promise.resolve(null)
    }

    @ReactMethod
    fun prefill(
        handle: String,
        messagesJson: String,
        optionsJson: String?,
        toolsJson: String?,
        pcmDataBase64: String?,
        promise: Promise,
    ) {
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        val nativeHandle = parseHandle(handle, promise) ?: return
        val rc = CactusJNI.nativePrefill(
            nativeHandle,
            messagesJson,
            buffer,
            optionsJson,
            toolsJson,
            decodeBase64OrNull(pcmDataBase64),
        )
        if (rc < 0) {
            fail(promise, "Prefill failed")
            return
        }
        promise.resolve(decodeNullTerminatedUtf8(buffer))
    }

    @ReactMethod
    fun complete(
        handle: String,
        messagesJson: String,
        optionsJson: String?,
        toolsJson: String?,
        pcmDataBase64: String?,
        promise: Promise,
    ) {
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        val nativeHandle = parseHandle(handle, promise) ?: return
        val rc = CactusJNI.nativeComplete(
            nativeHandle,
            messagesJson,
            buffer,
            optionsJson,
            toolsJson,
            null,
            decodeBase64OrNull(pcmDataBase64),
        )
        if (rc < 0) {
            fail(promise, "Completion failed")
            return
        }
        promise.resolve(decodeNullTerminatedUtf8(buffer))
    }

    @ReactMethod
    fun tokenize(handle: String, text: String, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        val buffer = IntArray(DEFAULT_TOKEN_BUFFER_SIZE)
        val outLen = LongArray(1)
        val rc = CactusJNI.nativeTokenize(nativeHandle, text, buffer, outLen)
        if (rc < 0) {
            fail(promise, "Tokenization failed")
            return
        }
        val result = Arguments.createArray()
        for (i in 0 until outLen[0].toInt()) {
            result.pushInt(buffer[i])
        }
        promise.resolve(result)
    }

    @ReactMethod
    fun scoreWindow(handle: String, tokens: ReadableArray, start: Double, end: Double, context: Double, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        val rc = CactusJNI.nativeScoreWindow(
            nativeHandle,
            readableArrayToIntArray(tokens),
            start.toLong(),
            end.toLong(),
            context.toLong(),
            buffer,
        )
        if (rc < 0) {
            fail(promise, "Score window failed")
            return
        }
        promise.resolve(decodeNullTerminatedUtf8(buffer))
    }

    @ReactMethod
    fun transcribe(
        handle: String,
        audioPath: String?,
        prompt: String?,
        optionsJson: String?,
        pcmDataBase64: String?,
        promise: Promise,
    ) {
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        val nativeHandle = parseHandle(handle, promise) ?: return
        val rc = CactusJNI.nativeTranscribe(
            nativeHandle,
            audioPath,
            prompt ?: "",
            buffer,
            optionsJson,
            null,
            decodeBase64OrNull(pcmDataBase64),
        )
        if (rc < 0) {
            fail(promise, "Transcription failed")
            return
        }
        promise.resolve(decodeNullTerminatedUtf8(buffer))
    }

    @ReactMethod
    fun embed(handle: String, text: String, normalize: Boolean, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        val buffer = FloatArray(DEFAULT_EMBEDDING_BUFFER_SIZE)
        val outDim = LongArray(1)
        val rc = CactusJNI.nativeEmbed(nativeHandle, text, buffer, outDim, normalize)
        if (rc < 0) {
            fail(promise, "Embedding failed")
            return
        }
        val result = Arguments.createArray()
        for (i in 0 until outDim[0].toInt()) {
            result.pushDouble(buffer[i].toDouble())
        }
        promise.resolve(result)
    }

    @ReactMethod
    fun imageEmbed(handle: String, imagePath: String, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        val buffer = FloatArray(DEFAULT_EMBEDDING_BUFFER_SIZE)
        val outDim = LongArray(1)
        val rc = CactusJNI.nativeImageEmbed(nativeHandle, imagePath, buffer, outDim)
        if (rc < 0) {
            fail(promise, "Image embedding failed")
            return
        }
        val result = Arguments.createArray()
        for (i in 0 until outDim[0].toInt()) {
            result.pushDouble(buffer[i].toDouble())
        }
        promise.resolve(result)
    }

    @ReactMethod
    fun audioEmbed(handle: String, audioPath: String, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        val buffer = FloatArray(DEFAULT_EMBEDDING_BUFFER_SIZE)
        val outDim = LongArray(1)
        val rc = CactusJNI.nativeAudioEmbed(nativeHandle, audioPath, buffer, outDim)
        if (rc < 0) {
            fail(promise, "Audio embedding failed")
            return
        }
        val result = Arguments.createArray()
        for (i in 0 until outDim[0].toInt()) {
            result.pushDouble(buffer[i].toDouble())
        }
        promise.resolve(result)
    }

    @ReactMethod
    fun vad(handle: String, audioPath: String?, optionsJson: String?, pcmDataBase64: String?, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        val rc = CactusJNI.nativeVad(nativeHandle, audioPath, buffer, optionsJson, decodeBase64OrNull(pcmDataBase64))
        if (rc < 0) {
            fail(promise, "VAD failed")
            return
        }
        promise.resolve(decodeNullTerminatedUtf8(buffer))
    }

    @ReactMethod
    fun diarize(handle: String, audioPath: String?, optionsJson: String?, pcmDataBase64: String?, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        val buffer = ByteArray(LARGE_BUFFER_SIZE)
        val rc = CactusJNI.nativeDiarize(nativeHandle, audioPath, buffer, optionsJson, decodeBase64OrNull(pcmDataBase64))
        if (rc < 0) {
            fail(promise, "Diarize failed")
            return
        }
        promise.resolve(decodeNullTerminatedUtf8(buffer))
    }

    @ReactMethod
    fun embedSpeaker(
        handle: String,
        audioPath: String?,
        optionsJson: String?,
        pcmDataBase64: String?,
        maskWeights: ReadableArray?,
        promise: Promise,
    ) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        val mask = maskWeights?.let { readableArrayToFloatArray(it) }
        val rc = CactusJNI.nativeEmbedSpeaker(
            nativeHandle,
            audioPath,
            buffer,
            optionsJson,
            decodeBase64OrNull(pcmDataBase64),
            mask,
            mask?.size?.toLong() ?: 0L,
        )
        if (rc < 0) {
            fail(promise, "EmbedSpeaker failed")
            return
        }
        promise.resolve(decodeNullTerminatedUtf8(buffer))
    }

    @ReactMethod
    fun ragQuery(handle: String, query: String, topK: Double, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        val rc = CactusJNI.nativeRagQuery(nativeHandle, query, buffer, topK.toLong())
        if (rc < 0) {
            fail(promise, "RAG query failed")
            return
        }
        promise.resolve(decodeNullTerminatedUtf8(buffer))
    }

    @ReactMethod
    fun indexInit(indexDir: String, embeddingDim: Double, promise: Promise) {
        val handle = CactusJNI.nativeIndexInit(indexDir, embeddingDim.toLong())
        if (handle == 0L) {
            fail(promise, "Failed to initialize index")
            return
        }
        promise.resolve(handle.toString())
    }

    @ReactMethod
    fun indexAdd(
        handle: String,
        ids: ReadableArray,
        documents: ReadableArray,
        embeddings: ReadableArray,
        metadatas: ReadableArray?,
        promise: Promise,
    ) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        val idsArray = readableArrayToIntArray(ids)
        val docsArray = readableArrayToStringArray(documents)
        val embArray = readableNestedFloatArrays(embeddings)
        val metaArray = metadatas?.let { readableArrayToStringArray(it) }
        val embeddingDim = if (embArray.isNotEmpty()) embArray[0].size.toLong() else 0L
        val rc = CactusJNI.nativeIndexAdd(nativeHandle, idsArray, docsArray, metaArray, embArray, embeddingDim)
        if (rc < 0) {
            fail(promise, "Failed to add to index")
            return
        }
        promise.resolve(null)
    }

    @ReactMethod
    fun indexDelete(handle: String, ids: ReadableArray, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        val rc = CactusJNI.nativeIndexDelete(nativeHandle, readableArrayToIntArray(ids))
        if (rc < 0) {
            fail(promise, "Failed to delete from index")
            return
        }
        promise.resolve(null)
    }

    @ReactMethod
    fun indexGet(handle: String, ids: ReadableArray, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        val idArray = readableArrayToIntArray(ids)
        val count = idArray.size
        val docBuffers = Array(count) { ByteArray(DEFAULT_INDEX_DOC_BUFFER_SIZE) }
        val docSizes = LongArray(count) { DEFAULT_INDEX_DOC_BUFFER_SIZE.toLong() }
        val metaBuffers = Array(count) { ByteArray(DEFAULT_INDEX_DOC_BUFFER_SIZE) }
        val metaSizes = LongArray(count) { DEFAULT_INDEX_DOC_BUFFER_SIZE.toLong() }
        val embBuffers = Array(count) { FloatArray(DEFAULT_INDEX_EMBED_BUFFER_SIZE) }
        val embSizes = LongArray(count) { DEFAULT_INDEX_EMBED_BUFFER_SIZE.toLong() }
        val rc = CactusJNI.nativeIndexGet(
            nativeHandle,
            idArray,
            docBuffers,
            docSizes,
            metaBuffers,
            metaSizes,
            embBuffers,
            embSizes,
        )
        if (rc < 0) {
            fail(promise, "Failed to get from index")
            return
        }
        val results = JSONArray()
        for (i in 0 until count) {
            val item = JSONObject()
            val document = decodeNullTerminatedUtf8(docBuffers[i])
            val metadataRaw = decodeNullTerminatedUtf8(metaBuffers[i])
            val embDim = embSizes[i].toInt()
            val embeddingJson = JSONArray()
            for (value in embBuffers[i].copyOf(embDim)) {
                embeddingJson.put(value.toDouble())
            }
            item.put("document", document)
            if (metadataRaw.isEmpty()) {
                item.put("metadata", JSONObject.NULL)
            } else {
                item.put("metadata", metadataRaw)
            }
            item.put("embedding", embeddingJson)
            results.put(item)
        }
        promise.resolve(JSONObject().put("results", results).toString())
    }

    @ReactMethod
    fun indexQuery(handle: String, embedding: ReadableArray, optionsJson: String?, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        val emb = readableArrayToFloatArray(embedding)
        val idBuffers = arrayOf(IntArray(DEFAULT_INDEX_RESULT_CAPACITY))
        val idSizes = LongArray(1) { DEFAULT_INDEX_RESULT_CAPACITY.toLong() }
        val scoreBuffers = arrayOf(FloatArray(DEFAULT_INDEX_RESULT_CAPACITY))
        val scoreSizes = LongArray(1) { DEFAULT_INDEX_RESULT_CAPACITY.toLong() }
        val rc = CactusJNI.nativeIndexQuery(
            nativeHandle,
            arrayOf(emb),
            emb.size.toLong(),
            optionsJson,
            idBuffers,
            idSizes,
            scoreBuffers,
            scoreSizes,
        )
        if (rc < 0) {
            fail(promise, "Index query failed")
            return
        }
        val results = JSONArray()
        for (i in 0 until idSizes[0].toInt()) {
            results.put(
                JSONObject()
                    .put("id", idBuffers[0][i])
                    .put("score", scoreBuffers[0][i].toDouble()),
            )
        }
        promise.resolve(JSONObject().put("results", results).toString())
    }

    @ReactMethod
    fun indexCompact(handle: String, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        val rc = CactusJNI.nativeIndexCompact(nativeHandle)
        if (rc < 0) {
            fail(promise, "Failed to compact index")
            return
        }
        promise.resolve(null)
    }

    @ReactMethod
    fun indexDestroy(handle: String, promise: Promise) {
        val nativeHandle = parseHandle(handle, promise) ?: return
        CactusJNI.nativeIndexDestroy(nativeHandle)
        promise.resolve(null)
    }

    @ReactMethod
    fun logSetLevel(level: Double, promise: Promise) {
        CactusJNI.nativeLogSetLevel(level.toInt())
        promise.resolve(null)
    }

    @ReactMethod
    fun setTelemetryEnvironment(framework: String?, cacheLocation: String?, version: String?, promise: Promise) {
        CactusJNI.nativeSetTelemetryEnvironment(framework, cacheLocation, version)
        promise.resolve(null)
    }

    @ReactMethod
    fun setAppId(appId: String, promise: Promise) {
        CactusJNI.nativeSetAppId(appId)
        promise.resolve(null)
    }

    @ReactMethod
    fun telemetryFlush(promise: Promise) {
        CactusJNI.nativeTelemetryFlush()
        promise.resolve(null)
    }

    @ReactMethod
    fun telemetryShutdown(promise: Promise) {
        CactusJNI.nativeTelemetryShutdown()
        promise.resolve(null)
    }

    @ReactMethod
    fun getLastError(promise: Promise) {
        promise.resolve(CactusJNI.nativeGetLastError())
    }
}
