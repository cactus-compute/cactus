package com.cactus

private fun check(rc: Int) {
    if (rc < 0) throw RuntimeException(CactusJNI.nativeGetLastError().ifEmpty { "Unknown error" })
}

actual fun cactusInit(modelPath: String, corpusDir: String?, cacheIndex: Boolean): Long {
    val handle = CactusJNI.nativeInit(modelPath, corpusDir, cacheIndex)
    if (handle == 0L) throw RuntimeException(CactusJNI.nativeGetLastError().ifEmpty { "Failed to initialize model" })
    return handle
}

actual fun cactusDestroy(handle: Long) = CactusJNI.nativeDestroy(handle)
actual fun cactusReset(handle: Long) = CactusJNI.nativeReset(handle)
actual fun cactusStop(handle: Long) = CactusJNI.nativeStop(handle)

actual fun cactusComplete(handle: Long, messagesJson: String, optionsJson: String?, toolsJson: String?, callback: CactusTokenCallback?, pcmData: ByteArray?): String {
    val buffer = ByteArray(1024 * 1024)
    check(CactusJNI.nativeComplete(handle, messagesJson, buffer, optionsJson, toolsJson, callback, pcmData))
    return buffer.decodeToString().trimEnd('\u0000')
}

actual fun cactusPrefill(handle: Long, messagesJson: String, optionsJson: String?, toolsJson: String?, pcmData: ByteArray?): String {
    val buffer = ByteArray(1024 * 1024)
    check(CactusJNI.nativePrefill(handle, messagesJson, buffer, optionsJson, toolsJson, pcmData))
    return buffer.decodeToString().trimEnd('\u0000')
}

actual fun cactusTokenize(handle: Long, text: String): IntArray {
    val tokenBuffer = IntArray(8192)
    val outLen = LongArray(1)
    check(CactusJNI.nativeTokenize(handle, text, tokenBuffer, outLen))
    return tokenBuffer.copyOf(outLen[0].toInt())
}

actual fun cactusScoreWindow(handle: Long, tokens: IntArray, start: Long, end: Long, context: Long): String {
    val buffer = ByteArray(1024 * 1024)
    check(CactusJNI.nativeScoreWindow(handle, tokens, start, end, context, buffer))
    return buffer.decodeToString().trimEnd('\u0000')
}

actual fun cactusTranscribe(handle: Long, audioPath: String?, prompt: String, optionsJson: String?, callback: CactusTokenCallback?, pcmData: ByteArray?): String {
    val buffer = ByteArray(1024 * 1024)
    check(CactusJNI.nativeTranscribe(handle, audioPath, prompt, buffer, optionsJson, callback, pcmData))
    return buffer.decodeToString().trimEnd('\u0000')
}

actual fun cactusEmbed(handle: Long, text: String, normalize: Boolean): FloatArray {
    val buffer = FloatArray(4096)
    val outDim = LongArray(1)
    check(CactusJNI.nativeEmbed(handle, text, buffer, outDim, normalize))
    return buffer.copyOf(outDim[0].toInt())
}

actual fun cactusImageEmbed(handle: Long, imagePath: String): FloatArray {
    val buffer = FloatArray(4096)
    val outDim = LongArray(1)
    check(CactusJNI.nativeImageEmbed(handle, imagePath, buffer, outDim))
    return buffer.copyOf(outDim[0].toInt())
}

actual fun cactusAudioEmbed(handle: Long, audioPath: String): FloatArray {
    val buffer = FloatArray(4096)
    val outDim = LongArray(1)
    check(CactusJNI.nativeAudioEmbed(handle, audioPath, buffer, outDim))
    return buffer.copyOf(outDim[0].toInt())
}

actual fun cactusRagQuery(handle: Long, query: String, topK: Long): String {
    val buffer = ByteArray(1024 * 1024)
    check(CactusJNI.nativeRagQuery(handle, query, buffer, topK))
    return buffer.decodeToString().trimEnd('\u0000')
}

actual fun cactusIndexInit(indexDir: String, embeddingDim: Long): Long {
    val handle = CactusJNI.nativeIndexInit(indexDir, embeddingDim)
    if (handle == 0L) throw RuntimeException(CactusJNI.nativeGetLastError().ifEmpty { "Failed to initialize index" })
    return handle
}

actual fun cactusIndexAdd(handle: Long, ids: IntArray, documents: Array<String>, metadatas: Array<String>?, embeddings: Array<FloatArray>, embeddingDim: Long): Int {
    val rc = CactusJNI.nativeIndexAdd(handle, ids, documents, metadatas, embeddings, embeddingDim)
    check(rc)
    return rc
}

actual fun cactusIndexDelete(handle: Long, ids: IntArray): Int {
    val rc = CactusJNI.nativeIndexDelete(handle, ids)
    check(rc)
    return rc
}

actual fun cactusIndexGet(handle: Long, ids: IntArray): String {
    val count = ids.size
    val docBuffers = Array(count) { ByteArray(4096) }
    val docSizes = LongArray(count) { 4096L }
    val metaBuffers = Array(count) { ByteArray(4096) }
    val metaSizes = LongArray(count) { 4096L }
    val embBuffers = Array(count) { FloatArray(4096) }
    val embSizes = LongArray(count) { 4096L }
    check(CactusJNI.nativeIndexGet(handle, ids, docBuffers, docSizes, metaBuffers, metaSizes, embBuffers, embSizes))
    val results = ids.indices.map { i ->
        val doc = docBuffers[i].decodeToString().trimEnd('\u0000')
        val meta = metaBuffers[i].decodeToString().trimEnd('\u0000')
        """{"document":"$doc","metadata":"$meta"}"""
    }
    return """{"results":[${results.joinToString(",")}]}"""
}

actual fun cactusIndexQuery(handle: Long, embedding: FloatArray, optionsJson: String?): String {
    val idBuffers = arrayOf(IntArray(1000))
    val idSizes = longArrayOf(1000L)
    val scoreBuffers = arrayOf(FloatArray(1000))
    val scoreSizes = longArrayOf(1000L)
    check(CactusJNI.nativeIndexQuery(handle, arrayOf(embedding), embedding.size.toLong(), optionsJson, idBuffers, idSizes, scoreBuffers, scoreSizes))
    val results = (0 until idSizes[0].toInt()).map { i ->
        """{"id":${idBuffers[0][i]},"score":${scoreBuffers[0][i]}}"""
    }
    return """{"results":[${results.joinToString(",")}]}"""
}

actual fun cactusIndexCompact(handle: Long): Int {
    val rc = CactusJNI.nativeIndexCompact(handle)
    check(rc)
    return rc
}

actual fun cactusIndexDestroy(handle: Long) = CactusJNI.nativeIndexDestroy(handle)
actual fun cactusGetLastError(): String = CactusJNI.nativeGetLastError()
actual fun cactusLogSetLevel(level: Int) = CactusJNI.nativeLogSetLevel(level)
actual fun cactusLogSetCallback(callback: CactusLogCallback?) = CactusJNI.nativeLogSetCallback(callback)

actual fun cactusSetTelemetryEnvironment(framework: String?, cacheLocation: String?, version: String?) =
    CactusJNI.nativeSetTelemetryEnvironment(framework, cacheLocation, version)

actual fun cactusSetAppId(appId: String) = CactusJNI.nativeSetAppId(appId)
actual fun cactusTelemetryFlush() = CactusJNI.nativeTelemetryFlush()
actual fun cactusTelemetryShutdown() = CactusJNI.nativeTelemetryShutdown()
