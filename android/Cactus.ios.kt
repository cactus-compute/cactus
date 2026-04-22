package com.cactus

import cactus.*
import kotlinx.cinterop.*

@OptIn(ExperimentalForeignApi::class)
actual fun cactusInit(modelPath: String, corpusDir: String?, cacheIndex: Boolean): Long {
    val ptr = cactus_init(modelPath, corpusDir, cacheIndex)
        ?: throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Failed to initialize model")
    return ptr.rawValue.toLong()
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusDestroy(model: Long) {
    cactus_destroy(model.toCPointer())
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusReset(model: Long) { cactus_reset(model.toCPointer()) }

@OptIn(ExperimentalForeignApi::class)
actual fun cactusStop(model: Long) { cactus_stop(model.toCPointer()) }

actual fun cactusGetLastError(): String = cactus_get_last_error()?.toKString() ?: ""

actual fun cactusSetTelemetryEnvironment(cacheDir: String) {
    cactus_set_telemetry_environment(null, cacheDir, null)
}

actual fun cactusSetAppId(appId: String) { cactus_set_app_id(appId) }

actual fun cactusTelemetryFlush() { cactus_telemetry_flush() }

actual fun cactusTelemetryShutdown() { cactus_telemetry_shutdown() }

@OptIn(ExperimentalForeignApi::class)
actual fun cactusComplete(model: Long, messagesJson: String, optionsJson: String?, toolsJson: String?, callback: CactusTokenCallback?, pcmData: ByteArray?): String {
    memScoped {
        val buffer = allocArray<ByteVar>(65536)
        val callbackRef = callback?.let { StableRef.create(it) }
        val pcmPtr = pcmData?.refTo(0)?.getPointer(this)
        try {
            val result = cactus_complete(
                model.toCPointer(),
                messagesJson,
                buffer,
                65536u,
                optionsJson,
                toolsJson,
                callbackRef?.let {
                    staticCFunction<CPointer<ByteVar>?, UInt, COpaquePointer?, Unit> { token, tokenId, userData ->
                        if (token != null && userData != null) {
                            userData.asStableRef<CactusTokenCallback>().get().onToken(token.toKString(), tokenId.toInt())
                        }
                    }
                },
                callbackRef?.asCPointer(),
                pcmPtr?.reinterpret(),
                pcmData?.size?.toULong() ?: 0u
            )
            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw RuntimeException(error)
            }
            return buffer.toKString()
        } finally {
            callbackRef?.dispose()
        }
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusPrefill(model: Long, messagesJson: String, optionsJson: String?, toolsJson: String?, pcmData: ByteArray?): String {
    memScoped {
        val buffer = allocArray<ByteVar>(65536)
        val pcmPtr = pcmData?.refTo(0)?.getPointer(this)
        val result = cactus_prefill(
            model.toCPointer(),
            messagesJson,
            buffer,
            65536u,
            optionsJson,
            toolsJson,
            pcmPtr?.reinterpret(),
            pcmData?.size?.toULong() ?: 0u
        )
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        return buffer.toKString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusTranscribe(model: Long, audioPath: String?, prompt: String?, optionsJson: String?, callback: CactusTokenCallback?, pcmData: ByteArray?): String {
    memScoped {
        val buffer = allocArray<ByteVar>(65536)
        val callbackRef = callback?.let { StableRef.create(it) }
        val pcmPtr = pcmData?.refTo(0)?.getPointer(this)
        try {
            val result = cactus_transcribe(
                model.toCPointer(),
                audioPath,
                prompt,
                buffer,
                65536u,
                optionsJson,
                callbackRef?.let {
                    staticCFunction<CPointer<ByteVar>?, UInt, COpaquePointer?, Unit> { token, tokenId, userData ->
                        if (token != null && userData != null) {
                            userData.asStableRef<CactusTokenCallback>().get().onToken(token.toKString(), tokenId.toInt())
                        }
                    }
                },
                callbackRef?.asCPointer(),
                pcmPtr?.reinterpret(),
                pcmData?.size?.toULong() ?: 0u
            )
            if (result < 0) {
                val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
                throw RuntimeException(error)
            }
            return buffer.toKString()
        } finally {
            callbackRef?.dispose()
        }
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusEmbed(model: Long, text: String, normalize: Boolean): FloatArray {
    memScoped {
        val buffer = allocArray<FloatVar>(4096)
        val dimPtr = alloc<ULongVar>()
        val result = cactus_embed(model.toCPointer(), text, buffer, 4096u, dimPtr.ptr, normalize)
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        return FloatArray(dimPtr.value.toInt()) { buffer[it] }
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusImageEmbed(model: Long, imagePath: String): FloatArray {
    memScoped {
        val buffer = allocArray<FloatVar>(4096)
        val dimPtr = alloc<ULongVar>()
        val result = cactus_image_embed(model.toCPointer(), imagePath, buffer, 4096u, dimPtr.ptr)
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        return FloatArray(dimPtr.value.toInt()) { buffer[it] }
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusAudioEmbed(model: Long, audioPath: String): FloatArray {
    memScoped {
        val buffer = allocArray<FloatVar>(4096)
        val dimPtr = alloc<ULongVar>()
        val result = cactus_audio_embed(model.toCPointer(), audioPath, buffer, 4096u, dimPtr.ptr)
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        return FloatArray(dimPtr.value.toInt()) { buffer[it] }
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusVad(model: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?): String {
    memScoped {
        val buffer = allocArray<ByteVar>(65536)
        val pcmPtr = pcmData?.refTo(0)?.getPointer(this)
        val result = cactus_vad(
            model.toCPointer(), audioPath, buffer, 65536u, optionsJson,
            pcmPtr?.reinterpret(), pcmData?.size?.toULong() ?: 0u
        )
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        return buffer.toKString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusRagQuery(model: Long, query: String, topK: Int): String {
    memScoped {
        val buffer = allocArray<ByteVar>(65536)
        val result = cactus_rag_query(model.toCPointer(), query, buffer, 65536u, topK.toULong())
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        return buffer.toKString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusTokenize(model: Long, text: String): IntArray {
    memScoped {
        val buffer = allocArray<UIntVar>(8192)
        val tokenLen = alloc<ULongVar>()
        val result = cactus_tokenize(model.toCPointer(), text, buffer, 8192u, tokenLen.ptr)
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        return IntArray(tokenLen.value.toInt()) { buffer[it].toInt() }
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusScoreWindow(model: Long, tokens: IntArray, start: Int, end: Int, context: Int): String {
    memScoped {
        val buffer = allocArray<ByteVar>(65536)
        val tokenBuffer = allocArray<UIntVar>(tokens.size)
        tokens.forEachIndexed { i, v -> tokenBuffer[i] = v.toUInt() }
        val result = cactus_score_window(
            model.toCPointer(), tokenBuffer, tokens.size.toULong(),
            start.toULong(), end.toULong(), context.toULong(),
            buffer, 65536u
        )
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        return buffer.toKString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusStreamTranscribeStart(model: Long, optionsJson: String?): Long {
    val ptr = cactus_stream_transcribe_start(model.toCPointer(), optionsJson)
        ?: throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Failed to create stream transcriber")
    return ptr.rawValue.toLong()
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusStreamTranscribeProcess(stream: Long, pcmData: ByteArray): String {
    memScoped {
        val buffer = allocArray<ByteVar>(65536)
        val pcmPtr = pcmData.refTo(0).getPointer(this)
        val result = cactus_stream_transcribe_process(
            stream.toCPointer(), pcmPtr.reinterpret(), pcmData.size.toULong(), buffer, 65536u
        )
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        return buffer.toKString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusStreamTranscribeStop(stream: Long): String {
    memScoped {
        val buffer = allocArray<ByteVar>(65536)
        val result = cactus_stream_transcribe_stop(stream.toCPointer(), buffer, 65536u)
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        return buffer.toKString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusIndexInit(indexDir: String, embeddingDim: Int): Long {
    val ptr = cactus_index_init(indexDir, embeddingDim.toULong())
        ?: throw RuntimeException("Failed to initialize index")
    return ptr.rawValue.toLong()
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusIndexAdd(index: Long, ids: IntArray, documents: Array<String>, embeddings: Array<FloatArray>, metadatas: Array<String>?): Int {
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
            index.toCPointer(), idPtr, docPtrs, metaPtrs, embPtrs,
            ids.size.toULong(), embeddings[0].size.toULong()
        )
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        return result
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusIndexDelete(index: Long, ids: IntArray): Int {
    memScoped {
        val idPtr = allocArray<IntVar>(ids.size)
        ids.forEachIndexed { i, v -> idPtr[i] = v }
        val result = cactus_index_delete(index.toCPointer(), idPtr, ids.size.toULong())
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        return result
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusIndexGet(index: Long, ids: IntArray): String {
    memScoped {
        val count = ids.size
        val idPtr = allocArray<IntVar>(count)
        ids.forEachIndexed { i, v -> idPtr[i] = v }

        val docBufSize = 4096
        val embBufSize = 4096

        val docRaw = Array(count) { allocArray<ByteVar>(docBufSize) }
        val metaRaw = Array(count) { allocArray<ByteVar>(docBufSize) }
        val embRaw = Array(count) { allocArray<FloatVar>(embBufSize) }

        val docBuffers = allocArray<CPointerVar<ByteVar>>(count)
        val docBufferSizes = allocArray<ULongVar>(count)
        val metaBuffers = allocArray<CPointerVar<ByteVar>>(count)
        val metaBufferSizes = allocArray<ULongVar>(count)
        val embBuffers = allocArray<CPointerVar<FloatVar>>(count)
        val embBufferSizes = allocArray<ULongVar>(count)

        for (i in 0 until count) {
            docBuffers[i] = docRaw[i]
            docBufferSizes[i] = docBufSize.toULong()
            metaBuffers[i] = metaRaw[i]
            metaBufferSizes[i] = docBufSize.toULong()
            embBuffers[i] = embRaw[i]
            embBufferSizes[i] = embBufSize.toULong()
        }

        val result = cactus_index_get(
            index.toCPointer(), idPtr, count.toULong(),
            docBuffers, docBufferSizes, metaBuffers, metaBufferSizes,
            embBuffers, embBufferSizes
        )
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        val sb = StringBuilder("{\"results\":[")
        for (i in 0 until count) {
            if (i > 0) sb.append(",")
            sb.append("{\"document\":\"${docRaw[i].toKString()}\"")
            val meta = metaRaw[i].toKString()
            if (meta.isNotEmpty()) sb.append(",\"metadata\":\"$meta\"") else sb.append(",\"metadata\":null")
            sb.append(",\"embedding\":[")
            val embDim = embBufferSizes[i].toInt()
            for (j in 0 until embDim) {
                if (j > 0) sb.append(",")
                sb.append(embRaw[i][j])
            }
            sb.append("]}")
        }
        sb.append("]}")
        return sb.toString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusIndexQuery(index: Long, embedding: FloatArray, optionsJson: String?): String {
    val topK = 1000L
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
            index.toCPointer(), embPtr.ptr, 1u, embedding.size.toULong(), optionsJson,
            idPtrPtr.ptr, idBufferSize.ptr, scorePtrPtr.ptr, scoreBufferSize.ptr
        )
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        val sb = StringBuilder("{\"results\":[")
        for (i in 0 until idBufferSize.value.toInt()) {
            if (i > 0) sb.append(",")
            sb.append("{\"id\":${idBuffer[i]},\"score\":${scoreBuffer[i]}}")
        }
        sb.append("]}")
        return sb.toString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusIndexCompact(index: Long): Int {
    val result = cactus_index_compact(index.toCPointer())
    if (result < 0) {
        val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
        throw RuntimeException(error)
    }
    return result
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusIndexDestroy(index: Long) {
    cactus_index_destroy(index.toCPointer())
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusDetectLanguage(model: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?): String {
    memScoped {
        val buffer = allocArray<ByteVar>(65536)
        val pcmPtr = pcmData?.refTo(0)?.getPointer(this)
        val result = cactus_detect_language(
            model.toCPointer(), audioPath, buffer, 65536u, optionsJson,
            pcmPtr?.reinterpret(), pcmData?.size?.toULong() ?: 0u
        )
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        return buffer.toKString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusDiarize(model: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?): String {
    memScoped {
        val buffer = allocArray<ByteVar>(65536)
        val pcmPtr = pcmData?.refTo(0)?.getPointer(this)
        val result = cactus_diarize(
            model.toCPointer(), audioPath, buffer, 65536u, optionsJson,
            pcmPtr?.reinterpret(), pcmData?.size?.toULong() ?: 0u
        )
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        return buffer.toKString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusEmbedSpeaker(model: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?, maskWeights: FloatArray?): String {
    memScoped {
        val buffer = allocArray<ByteVar>(65536)
        val pcmPtr = pcmData?.refTo(0)?.getPointer(this)
        val maskPtr = maskWeights?.let {
            val arr = allocArray<FloatVar>(it.size)
            it.forEachIndexed { i, v -> arr[i] = v }
            arr
        }
        val result = cactus_embed_speaker(
            model.toCPointer(), audioPath, buffer, 65536u, optionsJson,
            pcmPtr?.reinterpret(), pcmData?.size?.toULong() ?: 0u,
            maskPtr, maskWeights?.size?.toULong() ?: 0u
        )
        if (result < 0) {
            val error = cactus_get_last_error()?.toKString() ?: "Unknown error"
            throw RuntimeException(error)
        }
        return buffer.toKString()
    }
}

actual fun cactusLogSetLevel(level: Int) {
    cactus_log_set_level(level)
}

private var _logCallbackRef: StableRef<CactusLogCallback>? = null

@OptIn(ExperimentalForeignApi::class)
actual fun cactusLogSetCallback(callback: CactusLogCallback?) {
    _logCallbackRef?.dispose()
    _logCallbackRef = null
    if (callback == null) {
        cactus_log_set_callback(null, null)
    } else {
        val callbackRef = StableRef.create(callback)
        _logCallbackRef = callbackRef
        cactus_log_set_callback(
            staticCFunction<Int, CPointer<ByteVar>?, CPointer<ByteVar>?, COpaquePointer?, Unit> { level, component, message, userData ->
                if (userData != null) {
                    userData.asStableRef<CactusLogCallback>().get().onLog(
                        level,
                        component?.toKString() ?: "",
                        message?.toKString() ?: ""
                    )
                }
            },
            callbackRef.asCPointer()
        )
    }
}
