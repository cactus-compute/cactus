package com.cactus

fun interface CactusTokenCallback {
    fun onToken(token: String, tokenId: Int)
}

expect fun cactusInit(modelPath: String, corpusDir: String?, cacheIndex: Boolean): Long
expect fun cactusDestroy(model: Long)
expect fun cactusReset(model: Long)
expect fun cactusStop(model: Long)
expect fun cactusGetLastError(): String
expect fun cactusSetTelemetryEnvironment(cacheDir: String)
expect fun cactusSetAppId(appId: String)
expect fun cactusTelemetryFlush()
expect fun cactusTelemetryShutdown()
expect fun cactusComplete(model: Long, messagesJson: String, optionsJson: String?, toolsJson: String?, callback: CactusTokenCallback?): String
expect fun cactusTranscribe(model: Long, audioPath: String?, prompt: String?, optionsJson: String?, callback: CactusTokenCallback?, pcmData: ByteArray?): String
expect fun cactusEmbed(model: Long, text: String, normalize: Boolean): FloatArray
expect fun cactusImageEmbed(model: Long, imagePath: String): FloatArray
expect fun cactusAudioEmbed(model: Long, audioPath: String): FloatArray
expect fun cactusVad(model: Long, audioPath: String?, optionsJson: String?, pcmData: ByteArray?): String
expect fun cactusRagQuery(model: Long, query: String, topK: Int): String
expect fun cactusTokenize(model: Long, text: String): IntArray
expect fun cactusScoreWindow(model: Long, tokens: IntArray, start: Int, end: Int, context: Int): String
expect fun cactusStreamTranscribeStart(model: Long, optionsJson: String?): Long
expect fun cactusStreamTranscribeProcess(stream: Long, pcmData: ByteArray): String
expect fun cactusStreamTranscribeStop(stream: Long): String
expect fun cactusIndexInit(indexDir: String, embeddingDim: Int): Long
expect fun cactusIndexAdd(index: Long, ids: IntArray, documents: Array<String>, embeddings: Array<FloatArray>, metadatas: Array<String>?): Int
expect fun cactusIndexDelete(index: Long, ids: IntArray): Int
expect fun cactusIndexGet(index: Long, ids: IntArray): String
expect fun cactusIndexQuery(index: Long, embedding: FloatArray, optionsJson: String?): String
expect fun cactusIndexCompact(index: Long): Int
expect fun cactusIndexDestroy(index: Long)
