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
actual fun cactusDestroy(handle: Long) {
    cactus_destroy(handle.toCPointer())
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusReset(handle: Long) { cactus_reset(handle.toCPointer()) }

@OptIn(ExperimentalForeignApi::class)
actual fun cactusStop(handle: Long) { cactus_stop(handle.toCPointer()) }

@OptIn(ExperimentalForeignApi::class)
actual fun cactusComplete(handle: Long, messagesJson: String, optionsJson: String?, toolsJson: String?, callback: CactusTokenCallback?, pcmData: ByteArray?): String {
    memScoped {
        val buffer = allocArray<ByteVar>(1048576)
        val callbackRef = callback?.let { StableRef.create(it) }
        val pcmPtr = pcmData?.refTo(0)?.getPointer(this)
        try {
            val result = cactus_complete(
                handle.toCPointer(),
                messagesJson,
                buffer,
                1048576u,
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
            if (result < 0) throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Unknown error")
            return buffer.toKString()
        } finally {
            callbackRef?.dispose()
        }
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusPrefill(handle: Long, messagesJson: String, optionsJson: String?, toolsJson: String?, pcmData: ByteArray?): String {
    memScoped {
        val buffer = allocArray<ByteVar>(1048576)
        val pcmPtr = pcmData?.refTo(0)?.getPointer(this)
        val result = cactus_prefill(
            handle.toCPointer(),
            messagesJson,
            buffer,
            1048576u,
            optionsJson,
            toolsJson,
            pcmPtr?.reinterpret(),
            pcmData?.size?.toULong() ?: 0u
        )
        if (result < 0) throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Unknown error")
        return buffer.toKString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusTokenize(handle: Long, text: String): IntArray {
    memScoped {
        val buffer = allocArray<UIntVar>(8192)
        val tokenLen = alloc<ULongVar>()
        val result = cactus_tokenize(handle.toCPointer(), text, buffer, 8192u, tokenLen.ptr)
        if (result < 0) throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Unknown error")
        return IntArray(tokenLen.value.toInt()) { buffer[it].toInt() }
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusScoreWindow(handle: Long, tokens: IntArray, start: Long, end: Long, context: Long): String {
    memScoped {
        val buffer = allocArray<ByteVar>(1048576)
        val tokenBuffer = allocArray<UIntVar>(tokens.size)
        tokens.forEachIndexed { i, v -> tokenBuffer[i] = v.toUInt() }
        val result = cactus_score_window(
            handle.toCPointer(), tokenBuffer, tokens.size.toULong(),
            start.toULong(), end.toULong(), context.toULong(),
            buffer, 1048576u
        )
        if (result < 0) throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Unknown error")
        return buffer.toKString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusTranscribe(handle: Long, audioPath: String?, prompt: String, optionsJson: String?, callback: CactusTokenCallback?, pcmData: ByteArray?): String {
    memScoped {
        val buffer = allocArray<ByteVar>(1048576)
        val callbackRef = callback?.let { StableRef.create(it) }
        val pcmPtr = pcmData?.refTo(0)?.getPointer(this)
        try {
            val result = cactus_transcribe(
                handle.toCPointer(),
                audioPath,
                prompt,
                buffer,
                1048576u,
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
            if (result < 0) throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Unknown error")
            return buffer.toKString()
        } finally {
            callbackRef?.dispose()
        }
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusEmbed(handle: Long, text: String, normalize: Boolean): FloatArray {
    memScoped {
        val buffer = allocArray<FloatVar>(4096)
        val dimPtr = alloc<ULongVar>()
        val result = cactus_embed(handle.toCPointer(), text, buffer, 4096u, dimPtr.ptr, normalize)
        if (result < 0) throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Unknown error")
        return FloatArray(dimPtr.value.toInt()) { buffer[it] }
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusImageEmbed(handle: Long, imagePath: String): FloatArray {
    memScoped {
        val buffer = allocArray<FloatVar>(4096)
        val dimPtr = alloc<ULongVar>()
        val result = cactus_image_embed(handle.toCPointer(), imagePath, buffer, 4096u, dimPtr.ptr)
        if (result < 0) throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Unknown error")
        return FloatArray(dimPtr.value.toInt()) { buffer[it] }
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusAudioEmbed(handle: Long, audioPath: String): FloatArray {
    memScoped {
        val buffer = allocArray<FloatVar>(4096)
        val dimPtr = alloc<ULongVar>()
        val result = cactus_audio_embed(handle.toCPointer(), audioPath, buffer, 4096u, dimPtr.ptr)
        if (result < 0) throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Unknown error")
        return FloatArray(dimPtr.value.toInt()) { buffer[it] }
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusVad(handle: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?): String {
    memScoped {
        val buffer = allocArray<ByteVar>(1048576)
        val pcmPtr = pcmData?.refTo(0)?.getPointer(this)
        val result = cactus_vad(
            handle.toCPointer(), audioPath, buffer, 1048576u, optionsJson,
            pcmPtr?.reinterpret(), pcmData?.size?.toULong() ?: 0u
        )
        if (result < 0) throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Unknown error")
        return buffer.toKString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusDiarize(handle: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?): String {
    memScoped {
        val buffer = allocArray<ByteVar>(1048576)
        val pcmPtr = pcmData?.refTo(0)?.getPointer(this)
        val result = cactus_diarize(
            handle.toCPointer(), audioPath, buffer, 1048576u, optionsJson,
            pcmPtr?.reinterpret(), pcmData?.size?.toULong() ?: 0u
        )
        if (result < 0) throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Unknown error")
        return buffer.toKString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusEmbedSpeaker(handle: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?, maskWeights: FloatArray?, maskNumFrames: Long): String {
    memScoped {
        val buffer = allocArray<ByteVar>(1048576)
        val pcmPtr = pcmData?.refTo(0)?.getPointer(this)
        val maskPtr = maskWeights?.let {
            val arr = allocArray<FloatVar>(it.size)
            it.forEachIndexed { i, v -> arr[i] = v }
            arr
        }
        val result = cactus_embed_speaker(
            handle.toCPointer(), audioPath, buffer, 1048576u, optionsJson,
            pcmPtr?.reinterpret(), pcmData?.size?.toULong() ?: 0u,
            maskPtr, maskNumFrames.toULong()
        )
        if (result < 0) throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Unknown error")
        return buffer.toKString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusRagQuery(handle: Long, query: String, topK: Long): String {
    memScoped {
        val buffer = allocArray<ByteVar>(1048576)
        val result = cactus_rag_query(handle.toCPointer(), query, buffer, 1048576u, topK.toULong())
        if (result < 0) throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Unknown error")
        return buffer.toKString()
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusIndexInit(indexDir: String, embeddingDim: Long): Long {
    val ptr = cactus_index_init(indexDir, embeddingDim.toULong())
        ?: throw RuntimeException("Failed to initialize index")
    return ptr.rawValue.toLong()
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusIndexAdd(handle: Long, ids: IntArray, documents: Array<String>, metadatas: Array<String>?, embeddings: Array<FloatArray>, embeddingDim: Long): Int {
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
            handle.toCPointer(), idPtr, docPtrs, metaPtrs, embPtrs,
            ids.size.toULong(), embeddingDim.toULong()
        )
        if (result < 0) throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Unknown error")
        return result
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusIndexDelete(handle: Long, ids: IntArray): Int {
    memScoped {
        val idPtr = allocArray<IntVar>(ids.size)
        ids.forEachIndexed { i, v -> idPtr[i] = v }
        val result = cactus_index_delete(handle.toCPointer(), idPtr, ids.size.toULong())
        if (result < 0) throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Unknown error")
        return result
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusIndexGet(handle: Long, ids: IntArray): String {
    memScoped {
        val count = ids.size
        val idPtr = allocArray<IntVar>(count)
        ids.forEachIndexed { i, v -> idPtr[i] = v }
        val docBufs = allocArray<CPointerVar<ByteVar>>(count)
        val docSizes = allocArray<size_tVar>(count)
        val metaBufs = allocArray<CPointerVar<ByteVar>>(count)
        val metaSizes = allocArray<size_tVar>(count)
        val embBufs = allocArray<CPointerVar<FloatVar>>(count)
        val embSizes = allocArray<size_tVar>(count)
        val docAllocs = (0 until count).map { allocArray<ByteVar>(4096) }
        val metaAllocs = (0 until count).map { allocArray<ByteVar>(4096) }
        val embAllocs = (0 until count).map { allocArray<FloatVar>(4096) }
        for (i in 0 until count) {
            docBufs[i] = docAllocs[i]; docSizes[i] = 4096u
            metaBufs[i] = metaAllocs[i]; metaSizes[i] = 4096u
            embBufs[i] = embAllocs[i]; embSizes[i] = 4096u
        }
        cactus_index_get(handle.toCPointer(), idPtr, count.toULong(), docBufs, docSizes, metaBufs, metaSizes, embBufs, embSizes)
        val results = (0 until count).map { i ->
            val doc = docAllocs[i].toKString()
            val meta = metaAllocs[i].toKString()
            """{"document":"$doc","metadata":"$meta"}"""
        }
        return """{"results":[${results.joinToString(",")}]}"""
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusIndexQuery(handle: Long, embedding: FloatArray, optionsJson: String?): String {
    memScoped {
        val embArr = allocArray<FloatVar>(embedding.size)
        embedding.forEachIndexed { i, v -> embArr[i] = v }
        val embPtrs = allocArray<CPointerVar<FloatVar>>(1)
        embPtrs[0] = embArr
        val idBuf = allocArray<IntVar>(1000)
        val idPtrs = allocArray<CPointerVar<IntVar>>(1)
        idPtrs[0] = idBuf
        val idSizes = allocArray<size_tVar>(1); idSizes[0] = 1000u
        val scoreBuf = allocArray<FloatVar>(1000)
        val scorePtrs = allocArray<CPointerVar<FloatVar>>(1)
        scorePtrs[0] = scoreBuf
        val scoreSizes = allocArray<size_tVar>(1); scoreSizes[0] = 1000u
        cactus_index_query(handle.toCPointer(), embPtrs, 1u, embedding.size.toULong(), optionsJson, idPtrs, idSizes, scorePtrs, scoreSizes)
        val results = (0 until idSizes[0].toInt()).map { i ->
            """{"id":${idBuf[i]},"score":${scoreBuf[i]}}"""
        }
        return """{"results":[${results.joinToString(",")}]}"""
    }
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusIndexCompact(handle: Long): Int {
    val result = cactus_index_compact(handle.toCPointer())
    if (result < 0) throw RuntimeException(cactus_get_last_error()?.toKString() ?: "Unknown error")
    return result
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusIndexDestroy(handle: Long) {
    cactus_index_destroy(handle.toCPointer())
}

actual fun cactusGetLastError(): String = cactus_get_last_error()?.toKString() ?: ""

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
        return
    }
    val ref = StableRef.create(callback)
    _logCallbackRef = ref
    cactus_log_set_callback(staticCFunction { level, component, message, userData ->
        val cb = userData!!.asStableRef<CactusLogCallback>().get()
        cb.onLog(level, component?.toKString() ?: "", message?.toKString() ?: "")
    }, ref.asCPointer())
}

@OptIn(ExperimentalForeignApi::class)
actual fun cactusSetTelemetryEnvironment(framework: String?, cacheLocation: String?, version: String?) {
    cactus_set_telemetry_environment(framework, cacheLocation, version)
}

actual fun cactusSetAppId(appId: String) { cactus_set_app_id(appId) }

actual fun cactusTelemetryFlush() { cactus_telemetry_flush() }

actual fun cactusTelemetryShutdown() { cactus_telemetry_shutdown() }
