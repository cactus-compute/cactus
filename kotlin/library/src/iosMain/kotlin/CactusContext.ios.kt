@file:OptIn(ExperimentalForeignApi::class, kotlin.experimental.ExperimentalNativeApi::class)
package com.cactus

import com.cactus.native.*
import kotlinx.cinterop.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

actual object CactusContext {
    actual suspend fun initContext(params: CactusInitParams): CactusContextHandle? = withContext(Dispatchers.Default) {
        return@withContext memScoped {
            val cParams = cValue<cactus_init_params_c_t> {
                model_path = params.modelPath.cstr.ptr
                chat_template = params.chatTemplate?.cstr?.ptr
                n_ctx = params.nCtx
                n_batch = params.nBatch
                n_ubatch = params.nUbatch
                n_gpu_layers = params.nGpuLayers
                n_threads = params.nThreads
                use_mmap = params.useMmap
                use_mlock = params.useMlock
                embedding = params.embedding
                pooling_type = params.poolingType
                embd_normalize = params.embdNormalize
                flash_attn = params.flashAttn
                cache_type_k = params.cacheTypeK?.cstr?.ptr
                cache_type_v = params.cacheTypeV?.cstr?.ptr
                progress_callback = null
            }
            cactus_init_context_c(cParams.ptr)?.rawValue?.toLong()
        }
    }

    actual fun freeContext(handle: CactusContextHandle) {
        cactus_free_context_c(handle.toCPointer())
    }

    actual suspend fun completion(handle: CactusContextHandle, params: CactusCompletionParams): CactusCompletionResult = withContext(Dispatchers.Default) {
        return@withContext memScoped {
            // Note: Token callback temporarily disabled for staticCFunction issue
            // TODO: Implement proper non-capturing callback for streaming
            val tokenCallback: CPointer<CFunction<(CPointer<ByteVar>?) -> Boolean>>? = null

            val cParams = cValue<cactus_completion_params_c_t> {
                prompt = params.prompt.cstr.ptr
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
                ignore_eos = params.ignoreEos
                n_probs = params.nProbs
                stop_sequences = null
                stop_sequence_count = 0
                grammar = params.grammar?.cstr?.ptr
                token_callback = tokenCallback
            }

            val cResult = alloc<cactus_completion_result_c_t>()
            cactus_completion_c(handle.toCPointer(), cParams.ptr, cResult.ptr)

            val result = CactusCompletionResult(
                text = cResult.text?.toKString() ?: "",
                tokensPredicted = cResult.tokens_predicted,
                tokensEvaluated = cResult.tokens_evaluated,
                truncated = cResult.truncated,
                stoppedEos = cResult.stopped_eos,
                stoppedWord = cResult.stopped_word,
                stoppedLimit = cResult.stopped_limit,
                stoppingWord = cResult.stopping_word?.toKString()
            )
            cactus_free_completion_result_members_c(cResult.ptr)
            result
        }
    }

    actual suspend fun multimodalCompletion(handle: CactusContextHandle, params: CactusCompletionParams, mediaPaths: List<String>): CactusCompletionResult = withContext(Dispatchers.Default) {
        return@withContext memScoped {
            // Note: Token callback temporarily disabled for staticCFunction issue
            // TODO: Implement proper non-capturing callback for streaming
            val tokenCallback: CPointer<CFunction<(CPointer<ByteVar>?) -> Boolean>>? = null

            val cParams = cValue<cactus_completion_params_c_t> {
                prompt = params.prompt.cstr.ptr
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
                ignore_eos = params.ignoreEos
                n_probs = params.nProbs
                stop_sequences = null
                stop_sequence_count = 0
                grammar = params.grammar?.cstr?.ptr
                token_callback = tokenCallback
            }

            val cMediaPaths = mediaPaths.map { it.cstr.ptr }.toCValues()
            val cResult = alloc<cactus_completion_result_c_t>()

            cactus_multimodal_completion_c(
                handle.toCPointer(),
                cParams.ptr,
                cMediaPaths,
                mediaPaths.size,
                cResult.ptr
            )

            val result = CactusCompletionResult(
                text = cResult.text?.toKString() ?: "",
                tokensPredicted = cResult.tokens_predicted,
                tokensEvaluated = cResult.tokens_evaluated,
                truncated = cResult.truncated,
                stoppedEos = cResult.stopped_eos,
                stoppedWord = cResult.stopped_word,
                stoppedLimit = cResult.stopped_limit,
                stoppingWord = cResult.stopping_word?.toKString()
            )
            cactus_free_completion_result_members_c(cResult.ptr)
            result
        }
    }

    actual fun stopCompletion(handle: CactusContextHandle) {
        cactus_stop_completion_c(handle.toCPointer())
    }

    actual suspend fun tokenize(handle: CactusContextHandle, text: String): CactusTokenArray = withContext(Dispatchers.Default) {
        val cResult = cactus_tokenize_c(handle.toCPointer(), text)
        val tokens = cResult.useContents {
            tokens?.readBytes(count * sizeOf<IntVar>().toInt())?.let { bytes ->
                IntArray(count) { i ->
                    bytes.getIntAt(i * sizeOf<IntVar>().toInt())
                }
            } ?: IntArray(0)
        }
        val count = cResult.useContents { count }
        cactus_free_token_array_c(cResult)
        return@withContext CactusTokenArray(tokens, count)
    }

    actual suspend fun detokenize(handle: CactusContextHandle, tokens: IntArray): String = withContext(Dispatchers.Default) {
        return@withContext memScoped {
            val cTokens = tokens.toCValues()
            val result = cactus_detokenize_c(handle.toCPointer(), cTokens, tokens.size)
            result?.toKString() ?: ""
        }
    }

    actual suspend fun tokenizeWithMedia(handle: CactusContextHandle, text: String, mediaPaths: List<String>): CactusTokenizeResult = withContext(Dispatchers.Default) {
        return@withContext memScoped {
            val cMediaPaths = mediaPaths.map { it.cstr.ptr }.toCValues()
            val cResult = cactus_tokenize_with_media_c(handle.toCPointer(), text, cMediaPaths, mediaPaths.size)
            
            val result = cResult.useContents {
                val tokenArray = CactusTokenArray(
                    tokens.tokens?.readBytes(tokens.count * sizeOf<IntVar>().toInt())?.let { bytes ->
                        IntArray(tokens.count) { i -> bytes.getIntAt(i * sizeOf<IntVar>().toInt()) }
                    } ?: IntArray(0),
                    tokens.count
                )
                
                val bitmapHashList = if (bitmap_hashes != null && bitmap_hash_count > 0) {
                    List(bitmap_hash_count) { i ->
                        bitmap_hashes!![i]?.toKString() ?: ""
                    }
                } else emptyList()
                
                val chunkPositionList = if (chunk_positions != null && chunk_position_count > 0) {
                    chunk_positions!!.readBytes(chunk_position_count * sizeOf<ULongVar>().toInt()).let { bytes ->
                        LongArray(chunk_position_count) { i -> 
                            bytes.getLongAt(i * sizeOf<ULongVar>().toInt())
                        }.toList()
                    }
                } else emptyList()
                
                val chunkPositionMediaList = if (chunk_positions_media != null && chunk_position_media_count > 0) {
                    chunk_positions_media!!.readBytes(chunk_position_media_count * sizeOf<ULongVar>().toInt()).let { bytes ->
                        LongArray(chunk_position_media_count) { i -> 
                            bytes.getLongAt(i * sizeOf<ULongVar>().toInt())
                        }.toList()
                    }
                } else emptyList()
                
                CactusTokenizeResult(
                    tokens = tokenArray,
                    hasMedia = has_media,
                    bitmapHashes = bitmapHashList,
                    chunkPositions = chunkPositionList,
                    chunkPositionsMedia = chunkPositionMediaList
                )
            }
            cactus_free_tokenize_result_c(cResult.ptr)
            result
        }
    }

    actual suspend fun embedding(handle: CactusContextHandle, text: String): CactusFloatArray = withContext(Dispatchers.Default) {
        val cResult = cactus_embedding_c(handle.toCPointer(), text)
        val values = cResult.useContents {
            values?.readBytes(count * sizeOf<FloatVar>().toInt())?.let { bytes ->
                FloatArray(count) { i ->
                    bytes.getFloatAt(i * sizeOf<FloatVar>().toInt())
                }
            } ?: FloatArray(0)
        }
        val count = cResult.useContents { count }
        cactus_free_float_array_c(cResult)
        CactusFloatArray(values, count)
    }

    actual fun setGuideTokens(handle: CactusContextHandle, tokens: IntArray) {
        memScoped {
            val cTokens = tokens.toCValues()
            cactus_set_guide_tokens_c(handle.toCPointer(), cTokens, tokens.size)
        }
    }

    actual suspend fun initMultimodal(handle: CactusContextHandle, mmprojPath: String, useGpu: Boolean): Int = withContext(Dispatchers.Default) {
        return@withContext cactus_init_multimodal_c(handle.toCPointer(), mmprojPath, useGpu)
    }

    actual fun isMultimodalEnabled(handle: CactusContextHandle): Boolean {
        return cactus_is_multimodal_enabled_c(handle.toCPointer())
    }

    actual fun supportsVision(handle: CactusContextHandle): Boolean {
        return cactus_supports_vision_c(handle.toCPointer())
    }

    actual fun supportsAudio(handle: CactusContextHandle): Boolean {
        return cactus_supports_audio_c(handle.toCPointer())
    }

    actual fun releaseMultimodal(handle: CactusContextHandle) {
        cactus_release_multimodal_c(handle.toCPointer())
    }

    actual suspend fun initVocoder(handle: CactusContextHandle, vocoderModelPath: String): Int = withContext(Dispatchers.Default) {
        return@withContext cactus_init_vocoder_c(handle.toCPointer(), vocoderModelPath)
    }

    actual fun isVocoderEnabled(handle: CactusContextHandle): Boolean {
        return cactus_is_vocoder_enabled_c(handle.toCPointer())
    }

    

    actual suspend fun getFormattedAudioCompletion(handle: CactusContextHandle, speakerJson: String?, textToSpeak: String): String = withContext(Dispatchers.Default) {
        val result = cactus_get_formatted_audio_completion_c(handle.toCPointer(), speakerJson, textToSpeak)
        return@withContext result?.toKString()?.also { cactus_free_string_c(result) } ?: ""
    }

    actual suspend fun getAudioGuideTokens(handle: CactusContextHandle, textToSpeak: String): CactusTokenArray = withContext(Dispatchers.Default) {
        val cResult = cactus_get_audio_guide_tokens_c(handle.toCPointer(), textToSpeak)
        val tokens = cResult.useContents {
            tokens?.readBytes(count * sizeOf<IntVar>().toInt())?.let { bytes ->
                IntArray(count) { i -> bytes.getIntAt(i * sizeOf<IntVar>().toInt()) }
            } ?: IntArray(0)
        }
        val count = cResult.useContents { count }
        cactus_free_token_array_c(cResult)
        return@withContext CactusTokenArray(tokens, count)
    }

    actual suspend fun decodeAudioTokens(handle: CactusContextHandle, tokens: IntArray): CactusFloatArray = withContext(Dispatchers.Default) {
        memScoped {
            val cTokens = tokens.toCValues()
            val cResult = cactus_decode_audio_tokens_c(handle.toCPointer(), cTokens, tokens.size)
            val values = cResult.useContents {
                values?.readBytes(count * sizeOf<FloatVar>().toInt())?.let { bytes ->
                    FloatArray(count) { i -> bytes.getFloatAt(i * sizeOf<FloatVar>().toInt()) }
                } ?: FloatArray(0)
            }
            val count = cResult.useContents { count }
            cactus_free_float_array_c(cResult)
            CactusFloatArray(values, count)
        }
    }

    actual fun releaseVocoder(handle: CactusContextHandle) {
        cactus_release_vocoder_c(handle.toCPointer())
    }

    actual suspend fun bench(handle: CactusContextHandle, pp: Int, tg: Int, pl: Int, nr: Int): CactusBenchResult = withContext(Dispatchers.Default) {
        return@withContext memScoped {
            val cResult = cactus_bench_c(handle.toCPointer(), pp, tg, pl, nr)
            val result = cResult.useContents {
                CactusBenchResult(
                    modelName = model_name?.toKString() ?: "",
                    modelSize = model_size,
                    modelParams = model_params,
                    ppAvg = pp_avg,
                    ppStd = pp_std,
                    tgAvg = tg_avg,
                    tgStd = tg_std
                )
            }
            val tempResult = cResult.placeTo(memScope)
            cactus_free_bench_result_members_c(tempResult)
            result
        }
    }

    actual suspend fun applyLoraAdapters(handle: CactusContextHandle, adapters: List<CactusLoraAdapter>): Int = withContext(Dispatchers.Default) {
        return@withContext memScoped {
            val cAdapters = allocArray<cactus_lora_adapter_c_t>(adapters.size)
            adapters.forEachIndexed { index, adapter ->
                cAdapters[index].path = adapter.path.cstr.ptr
                cAdapters[index].scale = adapter.scale
            }
            val adaptersC = cValue<cactus_lora_adapters_c_t> {
                this.adapters = cAdapters
                count = adapters.size
            }
            memScoped {
                cactus_apply_lora_adapters_c(handle.toCPointer(), adaptersC.ptr)
            }
        }
    }

    actual fun removeLoraAdapters(handle: CactusContextHandle) {
        cactus_remove_lora_adapters_c(handle.toCPointer())
    }

    actual fun getLoadedLoraAdapters(handle: CactusContextHandle): List<CactusLoraAdapter> {
        return memScoped {
            val cAdapters = cactus_get_loaded_lora_adapters_c(handle.toCPointer())
            val adapterList = cAdapters.useContents {
                List(count) { i ->
                    val cAdapter = adapters!![i]
                    CactusLoraAdapter(
                        path = cAdapter.path?.toKString() ?: "",
                        scale = cAdapter.scale
                    )
                }
            }
            val tempAdapters = cAdapters.placeTo(memScope)
            cactus_free_lora_adapters_c(tempAdapters)
            adapterList
        }
    }

    actual fun validateChatTemplate(handle: CactusContextHandle, useJinja: Boolean, name: String?): Boolean {
        return cactus_validate_chat_template_c(handle.toCPointer(), useJinja, name)
    }

    actual suspend fun getFormattedChat(handle: CactusContextHandle, messages: String, chatTemplate: String?): String = withContext(Dispatchers.Default) {
        val result = cactus_get_formatted_chat_c(handle.toCPointer(), messages, chatTemplate)
        return@withContext result?.toKString()?.also { cactus_free_string_c(result) } ?: ""
    }

    actual suspend fun getFormattedChatWithJinja(handle: CactusContextHandle, messages: String, chatTemplate: String?, jsonSchema: String?, tools: String?, parallelToolCalls: Boolean, toolChoice: String?): CactusChatResult = withContext(Dispatchers.Default) {
        return@withContext memScoped {
            val cResult = cactus_get_formatted_chat_with_jinja_c(
                handle.toCPointer(),
                messages,
                chatTemplate,
                jsonSchema,
                tools,
                parallelToolCalls,
                toolChoice
            )
            val result = cResult.useContents {
                CactusChatResult(
                    prompt = prompt?.toKString() ?: "",
                    jsonSchema = json_schema?.toKString(),
                    tools = this.tools?.toKString(),
                    toolChoice = tool_choice?.toKString(),
                    parallelToolCalls = parallel_tool_calls
                )
            }
            val tempResult = cResult.placeTo(memScope)
            cactus_free_chat_result_members_c(tempResult)
            result
        }
    }

    actual fun rewind(handle: CactusContextHandle) {
        cactus_rewind_c(handle.toCPointer())
    }

    actual fun initSampling(handle: CactusContextHandle): Boolean {
        return cactus_init_sampling_c(handle.toCPointer())
    }

    actual fun beginCompletion(handle: CactusContextHandle) {
        cactus_begin_completion_c(handle.toCPointer())
    }

    actual fun endCompletion(handle: CactusContextHandle) {
        cactus_end_completion_c(handle.toCPointer())
    }

    actual fun loadPrompt(handle: CactusContextHandle) {
        cactus_load_prompt_c(handle.toCPointer())
    }

    actual fun loadPromptWithMedia(handle: CactusContextHandle, mediaPaths: List<String>) {
        memScoped {
            val cMediaPaths = mediaPaths.map { it.cstr.ptr }.toCValues()
            cactus_load_prompt_with_media_c(handle.toCPointer(), cMediaPaths, mediaPaths.size)
        }
    }

    actual suspend fun doCompletionStep(handle: CactusContextHandle): Pair<Int, String> = withContext(Dispatchers.Default) {
        return@withContext memScoped {
            val tokenTextRef = alloc<CPointerVar<ByteVar>>()
            val tokenId = cactus_do_completion_step_c(handle.toCPointer(), tokenTextRef.ptr)
            val tokenText = tokenTextRef.value?.toKString() ?: ""
            Pair(tokenId, tokenText)
        }
    }

    actual fun findStoppingStrings(handle: CactusContextHandle, text: String, lastTokenSize: Int, stopType: Int): Long {
        return cactus_find_stopping_strings_c(handle.toCPointer(), text, lastTokenSize.convert(), stopType).toLong()
    }

    actual fun getNCtx(handle: CactusContextHandle): Int {
        return cactus_get_n_ctx_c(handle.toCPointer())
    }

    actual fun getNEmbd(handle: CactusContextHandle): Int {
        return cactus_get_n_embd_c(handle.toCPointer())
    }

    actual fun getModelDesc(handle: CactusContextHandle): String {
        val result = cactus_get_model_desc_c(handle.toCPointer())
        return result?.toKString()?.also { cactus_free_string_c(result) } ?: ""
    }

    actual fun getModelSize(handle: CactusContextHandle): Long {
        return cactus_get_model_size_c(handle.toCPointer())
    }

    actual fun getModelParams(handle: CactusContextHandle): Long {
        return cactus_get_model_params_c(handle.toCPointer())
    }
} 