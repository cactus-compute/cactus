package com.cactus

object CactusJNI {
    init {
        System.loadLibrary("cactus_engine")
    }

    @JvmStatic external fun nativeInit(modelPath: String, corpusDir: String?, cacheIndex: Boolean): Long
    @JvmStatic external fun nativeDestroy(handle: Long)
    @JvmStatic external fun nativeReset(handle: Long)
    @JvmStatic external fun nativeStop(handle: Long)
    @JvmStatic external fun nativeComplete(handle: Long, messagesJson: String, responseBuffer: ByteArray, optionsJson: String?, toolsJson: String?, callback: CactusTokenCallback?, pcmData: ByteArray?): Int
    @JvmStatic external fun nativePrefill(handle: Long, messagesJson: String, responseBuffer: ByteArray, optionsJson: String?, toolsJson: String?, pcmData: ByteArray?): Int
    @JvmStatic external fun nativeTokenize(handle: Long, text: String, tokenBuffer: IntArray, outTokenLen: LongArray): Int
    @JvmStatic external fun nativeScoreWindow(handle: Long, tokens: IntArray, start: Long, end: Long, context: Long, responseBuffer: ByteArray): Int
    @JvmStatic external fun nativeTranscribe(handle: Long, audioPath: String?, prompt: String, responseBuffer: ByteArray, optionsJson: String?, callback: CactusTokenCallback?, pcmData: ByteArray?): Int
    @JvmStatic external fun nativeStreamTranscribeStart(handle: Long, optionsJson: String?): Long
    @JvmStatic external fun nativeStreamTranscribeProcess(stream: Long, pcmData: ByteArray?, responseBuffer: ByteArray): Int
    @JvmStatic external fun nativeStreamTranscribeStop(stream: Long, responseBuffer: ByteArray): Int
    @JvmStatic external fun nativeEmbed(handle: Long, text: String, embeddingsBuffer: FloatArray, outEmbeddingDim: LongArray, normalize: Boolean): Int
    @JvmStatic external fun nativeImageEmbed(handle: Long, imagePath: String, embeddingsBuffer: FloatArray, outEmbeddingDim: LongArray): Int
    @JvmStatic external fun nativeAudioEmbed(handle: Long, audioPath: String, embeddingsBuffer: FloatArray, outEmbeddingDim: LongArray): Int
    @JvmStatic external fun nativeRagQuery(handle: Long, query: String, responseBuffer: ByteArray, topK: Long): Int
    @JvmStatic external fun nativeIndexInit(indexDir: String, embeddingDim: Long): Long
    @JvmStatic external fun nativeIndexAdd(handle: Long, ids: IntArray, documents: Array<String>, metadatas: Array<String>?, embeddings: Array<FloatArray>, embeddingDim: Long): Int
    @JvmStatic external fun nativeIndexDelete(handle: Long, ids: IntArray): Int
    @JvmStatic external fun nativeIndexGet(handle: Long, ids: IntArray, documentBuffers: Array<ByteArray>, documentBufferSizes: LongArray, metadataBuffers: Array<ByteArray>, metadataBufferSizes: LongArray, embeddingBuffers: Array<FloatArray>, embeddingBufferSizes: LongArray): Int
    @JvmStatic external fun nativeIndexQuery(handle: Long, embeddings: Array<FloatArray>, embeddingDim: Long, optionsJson: String?, idBuffers: Array<IntArray>, idBufferSizes: LongArray, scoreBuffers: Array<FloatArray>, scoreBufferSizes: LongArray): Int
    @JvmStatic external fun nativeIndexCompact(handle: Long): Int
    @JvmStatic external fun nativeIndexDestroy(handle: Long)
    @JvmStatic external fun nativeGetLastError(): String
    @JvmStatic external fun nativeLogSetLevel(level: Int)
    @JvmStatic external fun nativeLogSetCallback(callback: CactusLogCallback?)
    @JvmStatic external fun nativeSetTelemetryEnvironment(framework: String?, cacheLocation: String?, version: String?)
    @JvmStatic external fun nativeSetAppId(appId: String)
    @JvmStatic external fun nativeTelemetryFlush()
    @JvmStatic external fun nativeTelemetryShutdown()
}
