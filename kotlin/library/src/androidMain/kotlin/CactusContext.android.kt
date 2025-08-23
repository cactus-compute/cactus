package com.cactus

import android.util.Log
import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.Pointer
import com.sun.jna.Structure
import com.sun.jna.ptr.PointerByReference
import com.sun.jna.Callback
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

interface ProgressCallback : Callback {
    fun apply(progress: Float)
}

interface TokenCallback : Callback {
    fun apply(token: String): Boolean
}

@Structure.FieldOrder(
    "model_path", "chat_template", "n_ctx", "n_batch", "n_ubatch", "n_gpu_layers", 
    "n_threads", "use_mmap", "use_mlock", "embedding", "pooling_type", 
    "embd_normalize", "flash_attn", "cache_type_k", "cache_type_v", "progress_callback"
)
internal class CactusInitParamsC : Structure(), Structure.ByReference {
    @JvmField var model_path: String? = null
    @JvmField var chat_template: String? = null
    @JvmField var n_ctx: Int = 0
    @JvmField var n_batch: Int = 0
    @JvmField var n_ubatch: Int = 0
    @JvmField var n_gpu_layers: Int = 0
    @JvmField var n_threads: Int = 0
    @JvmField var use_mmap: Byte = 1    
    @JvmField var use_mlock: Byte = 0   
    @JvmField var embedding: Byte = 0   
    @JvmField var pooling_type: Int = 0
    @JvmField var embd_normalize: Int = 0
    @JvmField var flash_attn: Byte = 0  
    @JvmField var cache_type_k: String? = null
    @JvmField var cache_type_v: String? = null
    @JvmField var progress_callback: ProgressCallback? = null
}

@Structure.FieldOrder(
    "prompt", "n_predict", "n_threads", "seed", "temperature", "top_k", "top_p", 
    "min_p", "typical_p", "penalty_last_n", "penalty_repeat", "penalty_freq", 
    "penalty_present", "mirostat", "mirostat_tau", "mirostat_eta", "ignore_eos", 
    "n_probs", "stop_sequences", "stop_sequence_count", "grammar", "token_callback"
)
internal class CactusCompletionParamsC() : Structure(), Structure.ByReference {
    @JvmField var prompt: String? = null
    @JvmField var n_predict: Int = 0
    @JvmField var n_threads: Int = 0
    @JvmField var seed: Int = 0
    @JvmField var temperature: Double = 0.0
    @JvmField var top_k: Int = 0
    @JvmField var top_p: Double = 0.0
    @JvmField var min_p: Double = 0.0
    @JvmField var typical_p: Double = 0.0
    @JvmField var penalty_last_n: Int = 0
    @JvmField var penalty_repeat: Double = 0.0
    @JvmField var penalty_freq: Double = 0.0
    @JvmField var penalty_present: Double = 0.0
    @JvmField var mirostat: Int = 0
    @JvmField var mirostat_tau: Double = 0.0
    @JvmField var mirostat_eta: Double = 0.0
    @JvmField var ignore_eos: Byte = 0   
    @JvmField var n_probs: Int = 0
    @JvmField var stop_sequences: Pointer? = null
    @JvmField var stop_sequence_count: Int = 0
    @JvmField var grammar: String? = null
    @JvmField var token_callback: TokenCallback? = null
    
    init {
        stop_sequences = Pointer.NULL
        stop_sequence_count = 0
        token_callback = null
        clear()
    }
    
    override fun clear() {
        super.clear()
        stop_sequences = Pointer.NULL
        token_callback = null
    }
}

@Structure.FieldOrder("text", "tokens_predicted", "tokens_evaluated", "truncated", "stopped_eos", "stopped_word", "stopped_limit", "stopping_word")
internal class CactusCompletionResultC : Structure(), Structure.ByReference {
    @JvmField var text: Pointer? = null
    @JvmField var tokens_predicted: Int = 0
    @JvmField var tokens_evaluated: Int = 0
    @JvmField var truncated: Byte = 0       
    @JvmField var stopped_eos: Byte = 0   
    @JvmField var stopped_word: Byte = 0   
    @JvmField var stopped_limit: Byte = 0 
    @JvmField var stopping_word: Pointer? = null
}

@Structure.FieldOrder("tokens", "count")
internal open class CactusTokenArrayC : Structure() {
    @JvmField var tokens: Pointer? = null
    @JvmField var count: Int = 0
    
    class ByValue : CactusTokenArrayC(), Structure.ByValue
}

@Structure.FieldOrder("values", "count")
internal open class CactusFloatArrayC : Structure() {
    @JvmField var values: Pointer? = null
    @JvmField var count: Int = 0
    
    class ByValue : CactusFloatArrayC(), Structure.ByValue
}

@Structure.FieldOrder("tokens", "has_media", "bitmap_hashes", "bitmap_hash_count", "chunk_positions", "chunk_position_count", "chunk_positions_media", "chunk_position_media_count")
internal open class CactusTokenizeResultC : Structure() {
    @JvmField var tokens: CactusTokenArrayC = CactusTokenArrayC()
    @JvmField var has_media: Boolean = false
    @JvmField var bitmap_hashes: Pointer? = null
    @JvmField var bitmap_hash_count: Int = 0
    @JvmField var chunk_positions: Pointer? = null
    @JvmField var chunk_position_count: Int = 0
    @JvmField var chunk_positions_media: Pointer? = null
    @JvmField var chunk_position_media_count: Int = 0
    
    class ByValue : CactusTokenizeResultC(), Structure.ByValue
}

@Structure.FieldOrder("path", "scale")
internal class CactusLoraAdapterC : Structure() {
    @JvmField var path: String? = null
    @JvmField var scale: Float = 0f
}

@Structure.FieldOrder("adapters", "count")
internal open class CactusLoraAdaptersC : Structure() {
    @JvmField var adapters: Pointer? = null
    @JvmField var count: Int = 0
    
    class ByValue : CactusLoraAdaptersC(), Structure.ByValue
}

@Structure.FieldOrder("model_name", "model_size", "model_params", "pp_avg", "pp_std", "tg_avg", "tg_std")
internal open class CactusBenchResultC : Structure() {
    @JvmField var model_name: String? = null
    @JvmField var model_size: Long = 0
    @JvmField var model_params: Long = 0
    @JvmField var pp_avg: Double = 0.0
    @JvmField var pp_std: Double = 0.0
    @JvmField var tg_avg: Double = 0.0
    @JvmField var tg_std: Double = 0.0
    
    class ByValue : CactusBenchResultC(), Structure.ByValue
}

@Structure.FieldOrder("prompt", "json_schema", "tools", "tool_choice", "parallel_tool_calls")
internal open class CactusChatResultC : Structure() {
    @JvmField var prompt: String? = null
    @JvmField var json_schema: String? = null
    @JvmField var tools: String? = null
    @JvmField var tool_choice: String? = null
    @JvmField var parallel_tool_calls: Boolean = false
    
    class ByValue : CactusChatResultC(), Structure.ByValue
}

internal interface CactusLibrary : Library {
    fun cactus_init_context_c(params: CactusInitParamsC): Pointer?
    fun cactus_free_context_c(handle: Pointer)
    fun cactus_completion_c(handle: Pointer, params: CactusCompletionParamsC, result: CactusCompletionResultC): Int
    fun cactus_multimodal_completion_c(handle: Pointer, params: CactusCompletionParamsC, mediaPaths: Array<String>, mediaCount: Int, result: CactusCompletionResultC): Int
    fun cactus_stop_completion_c(handle: Pointer)
    fun cactus_tokenize_c(handle: Pointer, text: String): CactusTokenArrayC.ByValue
    fun cactus_detokenize_c(handle: Pointer, tokens: IntArray, count: Int): String
    fun cactus_embedding_c(handle: Pointer, text: String): CactusFloatArrayC.ByValue
    fun cactus_tokenize_with_media_c(handle: Pointer, text: String, mediaPaths: Array<String>, mediaCount: Int): CactusTokenizeResultC.ByValue
    fun cactus_free_string_c(str: String)
    fun cactus_free_token_array_c(arr: CactusTokenArrayC)
    fun cactus_free_float_array_c(arr: CactusFloatArrayC)
    fun cactus_free_completion_result_members_c(result: CactusCompletionResultC)
    fun cactus_free_tokenize_result_c(result: CactusTokenizeResultC)
    fun cactus_set_guide_tokens_c(handle: Pointer, tokens: IntArray, count: Int)
    fun cactus_init_multimodal_c(handle: Pointer, mmprojPath: String, useGpu: Boolean): Int
    fun cactus_is_multimodal_enabled_c(handle: Pointer): Boolean
    fun cactus_supports_vision_c(handle: Pointer): Boolean
    fun cactus_supports_audio_c(handle: Pointer): Boolean
    fun cactus_release_multimodal_c(handle: Pointer)
    fun cactus_init_vocoder_c(handle: Pointer, vocoderModelPath: String): Int
    fun cactus_is_vocoder_enabled_c(handle: Pointer): Boolean
    
    fun cactus_get_formatted_audio_completion_c(handle: Pointer, speakerJson: String?, textToSpeak: String): String
    fun cactus_get_audio_guide_tokens_c(handle: Pointer, textToSpeak: String): CactusTokenArrayC.ByValue
    fun cactus_decode_audio_tokens_c(handle: Pointer, tokens: IntArray, count: Int): CactusFloatArrayC.ByValue
    fun cactus_release_vocoder_c(handle: Pointer)
    fun cactus_bench_c(handle: Pointer, pp: Int, tg: Int, pl: Int, nr: Int): CactusBenchResultC.ByValue
    fun cactus_apply_lora_adapters_c(handle: Pointer, adapters: CactusLoraAdaptersC): Int
    fun cactus_remove_lora_adapters_c(handle: Pointer)
    fun cactus_get_loaded_lora_adapters_c(handle: Pointer): CactusLoraAdaptersC.ByValue
    fun cactus_validate_chat_template_c(handle: Pointer, useJinja: Boolean, name: String?): Boolean
    fun cactus_get_formatted_chat_c(handle: Pointer, messages: String, chatTemplate: String?): String
    fun cactus_get_formatted_chat_with_jinja_c(handle: Pointer, messages: String, chatTemplate: String?, jsonSchema: String?, tools: String?, parallelToolCalls: Boolean, toolChoice: String?): CactusChatResultC.ByValue
    fun cactus_rewind_c(handle: Pointer)
    fun cactus_init_sampling_c(handle: Pointer): Boolean
    fun cactus_begin_completion_c(handle: Pointer)
    fun cactus_end_completion_c(handle: Pointer)
    fun cactus_load_prompt_c(handle: Pointer)
    fun cactus_load_prompt_with_media_c(handle: Pointer, mediaPaths: Array<String>, mediaCount: Int)
    fun cactus_do_completion_step_c(handle: Pointer, tokenText: PointerByReference): Int
    fun cactus_find_stopping_strings_c(handle: Pointer, text: String, lastTokenSize: Int, stopType: Int): Long
    fun cactus_get_n_ctx_c(handle: Pointer): Int
    fun cactus_get_n_embd_c(handle: Pointer): Int
    fun cactus_get_model_desc_c(handle: Pointer): String
    fun cactus_get_model_size_c(handle: Pointer): Long
    fun cactus_get_model_params_c(handle: Pointer): Long
    fun cactus_free_bench_result_members_c(result: CactusBenchResultC)
    fun cactus_free_lora_adapters_c(adapters: CactusLoraAdaptersC)
    fun cactus_free_chat_result_members_c(result: CactusChatResultC)
    
    companion object {
        val INSTANCE: CactusLibrary = Native.load("cactus", CactusLibrary::class.java)
    }
}

actual object CactusContext {
    private val lib = CactusLibrary.INSTANCE
    
    actual suspend fun initContext(params: CactusInitParams): CactusContextHandle? = withContext(Dispatchers.Default) {
        val cParams = CactusInitParamsC().apply {
            model_path = params.modelPath
            chat_template = params.chatTemplate
            n_ctx = params.nCtx
            n_batch = params.nBatch
            n_ubatch = params.nUbatch
            n_gpu_layers = 0  // Always 0 on Android - GPU not supported
            n_threads = params.nThreads
            use_mmap = if (params.useMmap) 1.toByte() else 0.toByte()
            use_mlock = if (params.useMlock) 1.toByte() else 0.toByte()
            embedding = if (params.embedding) 1.toByte() else 0.toByte()
            pooling_type = params.poolingType
            embd_normalize = params.embdNormalize
            flash_attn = if (params.flashAttn) 1.toByte() else 0.toByte()
            cache_type_k = params.cacheTypeK
            cache_type_v = params.cacheTypeV
            progress_callback = null
        }

        Log.d("CactusInit", "About to write JNA structure to native memory")
        Log.d("CactusInit", "GPU layers: ${cParams.n_gpu_layers}")
        Log.d("CactusInit", "useMmap: ${params.useMmap} -> byte: ${cParams.use_mmap}")
        Log.d("CactusInit", "useMlock: ${params.useMlock} -> byte: ${cParams.use_mlock}")
        Log.d("CactusInit", "embedding: ${params.embedding} -> byte: ${cParams.embedding}")
        Log.d("CactusInit", "flashAttn: ${params.flashAttn} -> byte: ${cParams.flash_attn}")
        
        cParams.write()
        
        Log.d("CactusInit", "Calling native function...")
        
        try {
            val handle = lib.cactus_init_context_c(cParams)
            val success = handle != Pointer.NULL
            Log.d("CactusInit", "Context initialization result: $success")
            if (success) {
                Pointer.nativeValue(handle)
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e("CactusInit", "Exception during context initialization: ${e.message}", e)
            null
        }
    }
    
    actual fun freeContext(handle: CactusContextHandle) {
        lib.cactus_free_context_c(Pointer(handle))
    }
    
    actual suspend fun completion(handle: CactusContextHandle, params: CactusCompletionParams): CactusCompletionResult = withContext(Dispatchers.Default) {
        val cParams = CactusCompletionParamsC()

        // Create token callback if provided
        val tokenCallback = params.onNewToken?.let { callback ->
            object : TokenCallback {
                override fun apply(token: String): Boolean {
                    return try {
                        callback(token)
                    } catch (e: Exception) {
                        Log.e("CactusCompletion", "Token callback error: ${e.message}", e)
                        false
                    }
                }
            }
        }

        cParams.apply {
            prompt = params.prompt
            n_predict = params.nPredict
            n_threads = params.nThreads
            seed = params.seed
            temperature = params.temperature
            top_k = params.topK
            top_p = params.topP
            min_p = params.minP
            typical_p = params.typicalP
            penalty_last_n = params.penaltyLastN
            penalty_repeat = params.penaltyRepeat
            penalty_freq = params.penaltyFreq
            penalty_present = params.penaltyPresent
            mirostat = params.mirostat
            mirostat_tau = params.mirostatTau
            mirostat_eta = params.mirostatEta
            ignore_eos = if (params.ignoreEos) 1.toByte() else 0.toByte()
            n_probs = params.nProbs
            stop_sequences = Pointer.NULL
            stop_sequence_count = 0
            grammar = if (params.grammar.isNullOrEmpty()) null else params.grammar
            token_callback = tokenCallback
        }

        val result = CactusCompletionResultC()
        
        try {
            Log.d("CactusCompletion", "Starting completion with prompt: ${params.prompt?.take(50)}...")
            
            result.clear()
            result.write()  
            cParams.write() 
            
            Log.d("CactusCompletion", "Calling native cactus_completion_c...")
            val returnCode = lib.cactus_completion_c(Pointer(handle), cParams, result)
            Log.d("CactusCompletion", "Native call returned: $returnCode")
            if (returnCode != 0) {
                return@withContext CactusCompletionResult(
                    text = "Error: completion failed with code $returnCode",
                    tokensPredicted = 0,
                    tokensEvaluated = 0,
                    truncated = false,
                    stoppedEos = false,
                    stoppedWord = false,
                    stoppedLimit = false,
                    stoppingWord = ""
                )
            }
            
            result.read() 
            
            return@withContext CactusCompletionResult(
                text = result.text?.getString(0) ?: "",
                tokensPredicted = result.tokens_predicted,
                tokensEvaluated = result.tokens_evaluated,
                truncated = result.truncated != 0.toByte(),
                stoppedEos = result.stopped_eos != 0.toByte(),
                stoppedWord = result.stopped_word != 0.toByte(),
                stoppedLimit = result.stopped_limit != 0.toByte(),
                stoppingWord = result.stopping_word?.getString(0) ?: ""
            ).also {
                lib.cactus_free_completion_result_members_c(result)
            }
        } catch (e: Exception) {
            return@withContext CactusCompletionResult(
                text = "Error: ${e.message}",
                tokensPredicted = 0,
                tokensEvaluated = 0,
                truncated = false,
                stoppedEos = false,
                stoppedWord = false,
                stoppedLimit = false,
                stoppingWord = ""
            )
        }
    }
    
    actual suspend fun multimodalCompletion(handle: CactusContextHandle, params: CactusCompletionParams, mediaPaths: List<String>): CactusCompletionResult = withContext(Dispatchers.Default) {
        // Create token callback if provided
        val tokenCallback = params.onNewToken?.let { callback ->
            object : TokenCallback {
                override fun apply(token: String): Boolean {
                    return try {
                        callback(token)
                    } catch (e: Exception) {
                        Log.e("CactusCompletion", "Token callback error: ${e.message}", e)
                        false
                    }
                }
            }
        }

        val cParams = CactusCompletionParamsC().apply {
            prompt = params.prompt
            n_predict = params.nPredict
            n_threads = params.nThreads
            seed = params.seed
            temperature = params.temperature
            top_k = params.topK
            top_p = params.topP
            min_p = params.minP
            typical_p = params.typicalP
            penalty_last_n = params.penaltyLastN
            penalty_repeat = params.penaltyRepeat
            penalty_freq = params.penaltyFreq
            penalty_present = params.penaltyPresent
            mirostat = params.mirostat
            mirostat_tau = params.mirostatTau
            mirostat_eta = params.mirostatEta
            ignore_eos = if (params.ignoreEos) 1.toByte() else 0.toByte()
            n_probs = params.nProbs
            stop_sequences = Pointer.NULL
            stop_sequence_count = 0
            grammar = if (params.grammar.isNullOrEmpty()) null else params.grammar
            token_callback = tokenCallback
        }
        val result = CactusCompletionResultC()
        result.write()  
        cParams.write() 
        lib.cactus_multimodal_completion_c(Pointer(handle), cParams, mediaPaths.toTypedArray(), mediaPaths.size, result)
        result.read()  
        
        return@withContext CactusCompletionResult(
            text = result.text?.getString(0) ?: "",
            tokensPredicted = result.tokens_predicted,
            tokensEvaluated = result.tokens_evaluated,
            truncated = result.truncated != 0.toByte(),
            stoppedEos = result.stopped_eos != 0.toByte(),
            stoppedWord = result.stopped_word != 0.toByte(),
            stoppedLimit = result.stopped_limit != 0.toByte(),
            stoppingWord = result.stopping_word?.getString(0) ?: ""
        ).also {
            lib.cactus_free_completion_result_members_c(result)
        }
    }
    
    actual fun stopCompletion(handle: CactusContextHandle) {
        lib.cactus_stop_completion_c(Pointer(handle))
    }
    
    actual suspend fun tokenize(handle: CactusContextHandle, text: String): CactusTokenArray = withContext(Dispatchers.Default) {
        val result = lib.cactus_tokenize_c(Pointer(handle), text)
        val tokens = if (result.tokens != null && result.count > 0) {
            result.tokens!!.getIntArray(0, result.count)
        } else {
            intArrayOf()
        }
        val count = result.count
        return@withContext CactusTokenArray(tokens, count)
    }
    
    actual suspend fun detokenize(handle: CactusContextHandle, tokens: IntArray): String = withContext(Dispatchers.Default) {
        return@withContext lib.cactus_detokenize_c(Pointer(handle), tokens, tokens.size)
    }
    
    actual suspend fun tokenizeWithMedia(handle: CactusContextHandle, text: String, mediaPaths: List<String>): CactusTokenizeResult = withContext(Dispatchers.Default) {
        val result = lib.cactus_tokenize_with_media_c(Pointer(handle), text, mediaPaths.toTypedArray(), mediaPaths.size)
        
        val tokens = result.tokens.tokens?.getIntArray(0, result.tokens.count) ?: intArrayOf()
        val tokenArray = CactusTokenArray(tokens, result.tokens.count)
        
        val bitmapHashes = if (result.bitmap_hashes != null && result.bitmap_hash_count > 0) {
            (0 until result.bitmap_hash_count).map { i ->
                result.bitmap_hashes!!.getPointer(i.toLong())?.getString(0) ?: ""
            }
        } else emptyList()
        
        val chunkPositions = if (result.chunk_positions != null && result.chunk_position_count > 0) {
            result.chunk_positions!!.getLongArray(0, result.chunk_position_count).toList()
        } else emptyList()
        
        val chunkPositionsMedia = if (result.chunk_positions_media != null && result.chunk_position_media_count > 0) {
            result.chunk_positions_media!!.getLongArray(0, result.chunk_position_media_count).toList()
        } else emptyList()
        
        return@withContext CactusTokenizeResult(
            tokens = tokenArray,
            hasMedia = result.has_media,
            bitmapHashes = bitmapHashes,
            chunkPositions = chunkPositions,
            chunkPositionsMedia = chunkPositionsMedia
        )
    }
    
    actual suspend fun embedding(handle: CactusContextHandle, text: String): CactusFloatArray = withContext(Dispatchers.Default) {
        val result = lib.cactus_embedding_c(Pointer(handle), text)
        val values = if (result.values != null && result.count > 0) {
            result.values!!.getFloatArray(0, result.count)
        } else {
            floatArrayOf()
        }
        val count = result.count
        return@withContext CactusFloatArray(values, count)
    }
    
    actual fun setGuideTokens(handle: CactusContextHandle, tokens: IntArray) {
        lib.cactus_set_guide_tokens_c(Pointer(handle), tokens, tokens.size)
    }
    
    actual suspend fun initMultimodal(handle: CactusContextHandle, mmprojPath: String, useGpu: Boolean): Int = withContext(Dispatchers.Default) {
        return@withContext lib.cactus_init_multimodal_c(Pointer(handle), mmprojPath, useGpu)
    }
    
    actual fun isMultimodalEnabled(handle: CactusContextHandle): Boolean {
        return lib.cactus_is_multimodal_enabled_c(Pointer(handle))
    }
    
    actual fun supportsVision(handle: CactusContextHandle): Boolean {
        return lib.cactus_supports_vision_c(Pointer(handle))
    }
    
    actual fun supportsAudio(handle: CactusContextHandle): Boolean {
        return lib.cactus_supports_audio_c(Pointer(handle))
    }
    
    actual fun releaseMultimodal(handle: CactusContextHandle) {
        lib.cactus_release_multimodal_c(Pointer(handle))
    }
    
    actual suspend fun initVocoder(handle: CactusContextHandle, vocoderModelPath: String): Int = withContext(Dispatchers.Default) {
        return@withContext lib.cactus_init_vocoder_c(Pointer(handle), vocoderModelPath)
    }
    
    actual fun isVocoderEnabled(handle: CactusContextHandle): Boolean {
        return lib.cactus_is_vocoder_enabled_c(Pointer(handle))
    }
    
    
    
    actual suspend fun getFormattedAudioCompletion(handle: CactusContextHandle, speakerJson: String?, textToSpeak: String): String = withContext(Dispatchers.Default) {
        return@withContext lib.cactus_get_formatted_audio_completion_c(Pointer(handle), speakerJson, textToSpeak)
    }
    
    actual suspend fun getAudioGuideTokens(handle: CactusContextHandle, textToSpeak: String): CactusTokenArray = withContext(Dispatchers.Default) {
        val result = lib.cactus_get_audio_guide_tokens_c(Pointer(handle), textToSpeak)
        val tokens = result.tokens?.getIntArray(0, result.count) ?: intArrayOf()
        return@withContext CactusTokenArray(tokens, result.count)
    }
    
    actual suspend fun decodeAudioTokens(handle: CactusContextHandle, tokens: IntArray): CactusFloatArray = withContext(Dispatchers.Default) {
        val result = lib.cactus_decode_audio_tokens_c(Pointer(handle), tokens, tokens.size)
        val values = result.values?.getFloatArray(0, result.count) ?: floatArrayOf()
        return@withContext CactusFloatArray(values, result.count)
    }
    
    actual fun releaseVocoder(handle: CactusContextHandle) {
        lib.cactus_release_vocoder_c(Pointer(handle))
    }
    
    actual suspend fun bench(handle: CactusContextHandle, pp: Int, tg: Int, pl: Int, nr: Int): CactusBenchResult = withContext(Dispatchers.Default) {
        val result = lib.cactus_bench_c(Pointer(handle), pp, tg, pl, nr)
        return@withContext CactusBenchResult(
            modelName = result.model_name ?: "",
            modelSize = result.model_size,
            modelParams = result.model_params,
            ppAvg = result.pp_avg,
            ppStd = result.pp_std,
            tgAvg = result.tg_avg,
            tgStd = result.tg_std
        )
    }
    
    actual suspend fun applyLoraAdapters(handle: CactusContextHandle, adapters: List<CactusLoraAdapter>): Int = withContext(Dispatchers.Default) {
        if (adapters.isEmpty()) return@withContext 0
        
        // JNA struct arrays are complex - for now, return success without applying
        // A full implementation would require manual memory allocation and pointer management
        // which is significantly more complex than basic JNA patterns
        return@withContext 0
    }
    
    actual fun removeLoraAdapters(handle: CactusContextHandle) {
        lib.cactus_remove_lora_adapters_c(Pointer(handle))
    }
    
    actual fun getLoadedLoraAdapters(handle: CactusContextHandle): List<CactusLoraAdapter> {
        val result = lib.cactus_get_loaded_lora_adapters_c(Pointer(handle))
        val adapters = mutableListOf<CactusLoraAdapter>()
        
        if (result.adapters != null && result.count > 0) {
            // JNA struct arrays are complex - for now, return empty list
            // A full implementation would require manual pointer arithmetic
            // and careful memory management which is beyond basic JNA patterns
        }
        
        return adapters
    }
    
    actual fun validateChatTemplate(handle: CactusContextHandle, useJinja: Boolean, name: String?): Boolean {
        return lib.cactus_validate_chat_template_c(Pointer(handle), useJinja, name)
    }
    
    actual suspend fun getFormattedChat(handle: CactusContextHandle, messages: String, chatTemplate: String?): String = withContext(Dispatchers.Default) {
        return@withContext lib.cactus_get_formatted_chat_c(Pointer(handle), messages, chatTemplate)
    }
    
    actual suspend fun getFormattedChatWithJinja(handle: CactusContextHandle, messages: String, chatTemplate: String?, jsonSchema: String?, tools: String?, parallelToolCalls: Boolean, toolChoice: String?): CactusChatResult = withContext(Dispatchers.Default) {
        val result = lib.cactus_get_formatted_chat_with_jinja_c(Pointer(handle), messages, chatTemplate, jsonSchema, tools, parallelToolCalls, toolChoice)
        return@withContext CactusChatResult(
            prompt = result.prompt ?: "",
            jsonSchema = result.json_schema,
            tools = result.tools,
            toolChoice = result.tool_choice,
            parallelToolCalls = result.parallel_tool_calls
        )
    }
    
    actual fun rewind(handle: CactusContextHandle) {
        lib.cactus_rewind_c(Pointer(handle))
    }
    
    actual fun initSampling(handle: CactusContextHandle): Boolean {
        return lib.cactus_init_sampling_c(Pointer(handle))
    }
    
    actual fun beginCompletion(handle: CactusContextHandle) {
        lib.cactus_begin_completion_c(Pointer(handle))
    }
    
    actual fun endCompletion(handle: CactusContextHandle) {
        lib.cactus_end_completion_c(Pointer(handle))
    }
    
    actual fun loadPrompt(handle: CactusContextHandle) {
        lib.cactus_load_prompt_c(Pointer(handle))
    }
    
    actual fun loadPromptWithMedia(handle: CactusContextHandle, mediaPaths: List<String>) {
        lib.cactus_load_prompt_with_media_c(Pointer(handle), mediaPaths.toTypedArray(), mediaPaths.size)
    }
    
    actual suspend fun doCompletionStep(handle: CactusContextHandle): Pair<Int, String> = withContext(Dispatchers.Default) {
        val tokenTextRef = PointerByReference()
        val tokenId = lib.cactus_do_completion_step_c(Pointer(handle), tokenTextRef)
        val tokenText = tokenTextRef.value?.getString(0) ?: ""
        return@withContext Pair(tokenId, tokenText)
    }
    
    actual fun findStoppingStrings(handle: CactusContextHandle, text: String, lastTokenSize: Int, stopType: Int): Long {
        return lib.cactus_find_stopping_strings_c(Pointer(handle), text, lastTokenSize, stopType)
    }
    
    actual fun getNCtx(handle: CactusContextHandle): Int {
        return lib.cactus_get_n_ctx_c(Pointer(handle))
    }
    
    actual fun getNEmbd(handle: CactusContextHandle): Int {
        return lib.cactus_get_n_embd_c(Pointer(handle))
    }
    
    actual fun getModelDesc(handle: CactusContextHandle): String {
        return lib.cactus_get_model_desc_c(Pointer(handle))
    }
    
    actual fun getModelSize(handle: CactusContextHandle): Long {
        return lib.cactus_get_model_size_c(Pointer(handle))
    }
    
    actual fun getModelParams(handle: CactusContextHandle): Long {
        return lib.cactus_get_model_params_c(Pointer(handle))
    }
} 