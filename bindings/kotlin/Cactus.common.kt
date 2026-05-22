package com.cactus

fun interface CactusTokenCallback {
    fun onToken(token: String, tokenId: Int)
}

fun interface CactusLogCallback {
    fun onLog(level: Int, component: String, message: String)
}

expect fun cactusInit(modelPath: String, corpusDir: String?, cacheIndex: Boolean): Long
expect fun cactusDestroy(handle: Long)
expect fun cactusReset(handle: Long)
expect fun cactusStop(handle: Long)
expect fun cactusComplete(handle: Long, messagesJson: String, optionsJson: String?, toolsJson: String?, callback: CactusTokenCallback?, pcmData: ByteArray?): String
expect fun cactusPrefill(handle: Long, messagesJson: String, optionsJson: String?, toolsJson: String?, pcmData: ByteArray?): String
expect fun cactusTokenize(handle: Long, text: String): IntArray
expect fun cactusScoreWindow(handle: Long, tokens: IntArray, start: Long, end: Long, context: Long): String
expect fun cactusTranscribe(handle: Long, audioPath: String?, prompt: String, optionsJson: String?, callback: CactusTokenCallback?, pcmData: ByteArray?): String
expect fun cactusEmbed(handle: Long, text: String, normalize: Boolean): FloatArray
expect fun cactusImageEmbed(handle: Long, imagePath: String): FloatArray
expect fun cactusAudioEmbed(handle: Long, audioPath: String): FloatArray
expect fun cactusRagQuery(handle: Long, query: String, topK: Long): String
expect fun cactusIndexInit(indexDir: String, embeddingDim: Long): Long
expect fun cactusIndexAdd(handle: Long, ids: IntArray, documents: Array<String>, metadatas: Array<String>?, embeddings: Array<FloatArray>, embeddingDim: Long): Int
expect fun cactusIndexDelete(handle: Long, ids: IntArray): Int
expect fun cactusIndexGet(handle: Long, ids: IntArray): String
expect fun cactusIndexQuery(handle: Long, embedding: FloatArray, optionsJson: String?): String
expect fun cactusIndexCompact(handle: Long): Int
expect fun cactusIndexDestroy(handle: Long)
expect fun cactusGetLastError(): String
expect fun cactusLogSetLevel(level: Int)
expect fun cactusLogSetCallback(callback: CactusLogCallback?)
expect fun cactusSetTelemetryEnvironment(framework: String?, cacheLocation: String?, version: String?)
expect fun cactusSetAppId(appId: String)
expect fun cactusTelemetryFlush()
expect fun cactusTelemetryShutdown()
