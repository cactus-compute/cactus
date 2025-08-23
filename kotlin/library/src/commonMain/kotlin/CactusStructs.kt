package com.cactus

data class CactusInitParams(
    val modelPath: String,
    val chatTemplate: String? = null,
    val nCtx: Int = 2048,
    val nBatch: Int = 512,
    val nUbatch: Int = 512,
    val nGpuLayers: Int = 0,
    val nThreads: Int = 4,
    val useMmap: Boolean = true,
    val useMlock: Boolean = false,
    val embedding: Boolean = false,
    val poolingType: Int = 0,
    val embdNormalize: Int = -1,
    val flashAttn: Boolean = false,
    val cacheTypeK: String? = null,
    val cacheTypeV: String? = null
)

data class CactusCompletionParams(
    val prompt: String,
    val nPredict: Int = 128,
    val nThreads: Int = 4,
    val seed: Int = -1,
    val temperature: Double = 0.8,
    val topK: Int = 40,
    val topP: Double = 0.95,
    val minP: Double = 0.05,
    val typicalP: Double = 1.0,
    val penaltyLastN: Int = 64,
    val penaltyRepeat: Double = 1.0,
    val penaltyFreq: Double = 0.0,
    val penaltyPresent: Double = 0.0,
    val mirostat: Int = 0,
    val mirostatTau: Double = 5.0,
    val mirostatEta: Double = 0.1,
    val ignoreEos: Boolean = false,
    val nProbs: Int = 0,
    val stopSequences: List<String> = emptyList(),
    val grammar: String? = null,
    val onNewToken: ((String) -> Boolean)? = null
)

data class CactusCompletionResult(
    val text: String,
    val tokensPredicted: Int,
    val tokensEvaluated: Int,
    val truncated: Boolean,
    val stoppedEos: Boolean,
    val stoppedWord: Boolean,
    val stoppedLimit: Boolean,
    val stoppingWord: String?
)

data class CactusTokenArray(
    val tokens: IntArray,
    val count: Int
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is CactusTokenArray) return false
        if (!tokens.contentEquals(other.tokens)) return false
        if (count != other.count) return false
        return true
    }

    override fun hashCode(): Int {
        var result = tokens.contentHashCode()
        result = 31 * result + count
        return result
    }
}

data class CactusFloatArray(
    val values: FloatArray,
    val count: Int
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is CactusFloatArray) return false
        if (!values.contentEquals(other.values)) return false
        if (count != other.count) return false
        return true
    }

    override fun hashCode(): Int {
        var result = values.contentHashCode()
        result = 31 * result + count
        return result
    }
}

data class CactusTokenizeResult(
    val tokens: CactusTokenArray,
    val hasMedia: Boolean,
    val bitmapHashes: List<String>,
    val chunkPositions: List<Long>,
    val chunkPositionsMedia: List<Long>
)

data class CactusLoraAdapter(
    val path: String,
    val scale: Float
)

data class CactusBenchResult(
    val modelName: String,
    val modelSize: Long,
    val modelParams: Long,
    val ppAvg: Double,
    val ppStd: Double,
    val tgAvg: Double,
    val tgStd: Double
)

data class CactusChatResult(
    val prompt: String,
    val jsonSchema: String?,
    val tools: String?,
    val toolChoice: String?,
    val parallelToolCalls: Boolean
) 

data class SpeechRecognitionParams(
    val language: String = "en-US",
    val continuous: Boolean = false,
    val maxDuration: Int = 30,
    val enablePartialResults: Boolean = true,
    val enableProfanityFilter: Boolean = true,
    val enablePunctuation: Boolean = true
)

data class SpeechRecognitionResult(
    val text: String,
    val confidence: Float,
    val isPartial: Boolean = false,
    val alternatives: List<String> = emptyList()
)

sealed class SpeechError : Exception() {
    object PermissionDenied : SpeechError()
    object NotAvailable : SpeechError()
    object NetworkError : SpeechError()
    object AudioUnavailable : SpeechError()
    object NoSpeechDetected : SpeechError()
    object RecognitionUnavailable : SpeechError()
    data class UnknownError(override val message: String) : SpeechError()
}

sealed class SpeechState {
    object Idle : SpeechState()
    object Listening : SpeechState()
    object Processing : SpeechState()
    data class Result(val result: SpeechRecognitionResult) : SpeechState()
    data class Error(val error: SpeechError) : SpeechState()
} 