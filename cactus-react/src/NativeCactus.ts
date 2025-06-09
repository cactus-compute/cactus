import type { TurboModule } from 'react-native'
import { TurboModuleRegistry } from 'react-native'

export type NativeEmbeddingParams = {
  embd_normalize?: number
}

export type NativeContextParams = {
  model: string
  /**
   * Chat template to override the default one from the model.
   */
  chat_template?: string

  reasoning_format?: string

  is_model_asset?: boolean
  use_progress_callback?: boolean

  n_ctx?: number
  n_batch?: number
  n_ubatch?: number

  n_threads?: number

  /**
   * Number of layers to store in VRAM (Currently only for iOS)
   */
  n_gpu_layers?: number
  /**
   * Skip GPU devices (iOS only)
   */
  no_gpu_devices?: boolean

  /**
   * Enable flash attention, only recommended in GPU device (Experimental in llama.cpp)
   */
  flash_attn?: boolean

  /**
   * KV cache data type for the K (Experimental in llama.cpp)
   */
  cache_type_k?: string
  /**
   * KV cache data type for the V (Experimental in llama.cpp)
   */
  cache_type_v?: string

  use_mlock?: boolean
  use_mmap?: boolean
  vocab_only?: boolean

  /**
   * Single LoRA adapter path
   */
  lora?: string
  /**
   * Single LoRA adapter scale
   */
  lora_scaled?: number
  /**
   * LoRA adapter list
   */
  lora_list?: Array<{ path: string; scaled?: number }>

  rope_freq_base?: number
  rope_freq_scale?: number

  pooling_type?: number

  // Embedding params
  embedding?: boolean
  embd_normalize?: number
}

export type NativeCompletionParams = {
  prompt: string
  n_threads?: number
  /**
   * JSON schema for convert to grammar for structured JSON output.
   * It will be override by grammar if both are set.
   */
  json_schema?: string
  /**
   * Set grammar for grammar-based sampling.  Default: no grammar
   */
  grammar?: string
  /**
   * Lazy grammar sampling, trigger by grammar_triggers. Default: false
   */
  grammar_lazy?: boolean
  /**
   * Lazy grammar triggers. Default: []
   */
  grammar_triggers?: Array<{
    type: number
    value: string
    token: number
  }>
  preserved_tokens?: Array<string>
  chat_format?: number
  /**
   * Specify a JSON array of stopping strings.
   * These words will not be included in the completion, so make sure to add them to the prompt for the next iteration. Default: `[]`
   */
  stop?: Array<string>
  /**
   * Set the maximum number of tokens to predict when generating text.
   * **Note:** May exceed the set limit slightly if the last token is a partial multibyte character.
   * When 0,no tokens will be generated but the prompt is evaluated into the cache. Default: `-1`, where `-1` is infinity.
   */
  n_predict?: number
  /**
   * If greater than 0, the response also contains the probabilities of top N tokens for each generated token given the sampling settings.
   * Note that for temperature < 0 the tokens are sampled greedily but token probabilities are still being calculated via a simple softmax of the logits without considering any other sampler settings.
   * Default: `0`
   */
  n_probs?: number
  /**
   * Limit the next token selection to the K most probable tokens.  Default: `40`
   */
  top_k?: number
  /**
   * Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. Default: `0.95`
   */
  top_p?: number
  /**
   * The minimum probability for a token to be considered, relative to the probability of the most likely token. Default: `0.05`
   */
  min_p?: number
  /**
   * Set the chance for token removal via XTC sampler. Default: `0.0`, which is disabled.
   */
  xtc_probability?: number
  /**
   * Set a minimum probability threshold for tokens to be removed via XTC sampler. Default: `0.1` (> `0.5` disables XTC)
   */
  xtc_threshold?: number
  /**
   * Enable locally typical sampling with parameter p. Default: `1.0`, which is disabled.
   */
  typical_p?: number
  /**
   * Adjust the randomness of the generated text. Default: `0.8`
   */
  temperature?: number
  /**
   * Last n tokens to consider for penalizing repetition. Default: `64`, where `0` is disabled and `-1` is ctx-size.
   */
  penalty_last_n?: number
  /**
   * Control the repetition of token sequences in the generated text. Default: `1.0`
   */
  penalty_repeat?: number
  /**
   * Repeat alpha frequency penalty. Default: `0.0`, which is disabled.
   */
  penalty_freq?: number
  /**
   * Repeat alpha presence penalty. Default: `0.0`, which is disabled.
   */
  penalty_present?: number
  /**
   * Enable Mirostat sampling, controlling perplexity during text generation. Default: `0`, where `0` is disabled, `1` is Mirostat, and `2` is Mirostat 2.0.
   */
  mirostat?: number
  /**
   * Set the Mirostat target entropy, parameter tau. Default: `5.0`
   */
  mirostat_tau?: number
  /**
   * Set the Mirostat learning rate, parameter eta. Default: `0.1`
   */
  mirostat_eta?: number
  /**
   * Set the DRY (Don't Repeat Yourself) repetition penalty multiplier. Default: `0.0`, which is disabled.
   */
  dry_multiplier?: number
  /**
   * Set the DRY repetition penalty base value. Default: `1.75`
   */
  dry_base?: number
  /**
   * Tokens that extend repetition beyond this receive exponentially increasing penalty: multiplier * base ^ (length of repeating sequence before token - allowed length). Default: `2`
   */
  dry_allowed_length?: number
  /**
   * How many tokens to scan for repetitions. Default: `-1`, where `0` is disabled and `-1` is context size.
   */
  dry_penalty_last_n?: number
  /**
   * Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted. Default: `['\n', ':', '"', '*']`
   */
  dry_sequence_breakers?: Array<string>
  /**
   * Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641. Default: `-1.0` (Disabled)
   */
  top_n_sigma?: number

  /**
   * Ignore end of stream token and continue generating. Default: `false`
   */
  ignore_eos?: boolean
  /**
   * Modify the likelihood of a token appearing in the generated text completion.
   * For example, use `"logit_bias": [[15043,1.0]]` to increase the likelihood of the token 'Hello', or `"logit_bias": [[15043,-1.0]]` to decrease its likelihood.
   * Setting the value to false, `"logit_bias": [[15043,false]]` ensures that the token `Hello` is never produced. The tokens can also be represented as strings,
   * e.g.`[["Hello, World!",-0.5]]` will reduce the likelihood of all the individual tokens that represent the string `Hello, World!`, just like the `presence_penalty` does.
   * Default: `[]`
   */
  logit_bias?: Array<Array<number>>
  /**
   * Set the random number generator (RNG) seed. Default: `-1`, which is a random seed.
   */
  seed?: number

  emit_partial_completion: boolean
}

export type NativeCompletionTokenProbItem = {
  tok_str: string
  prob: number
}

export type NativeCompletionTokenProb = {
  content: string
  probs: Array<NativeCompletionTokenProbItem>
}

export type NativeCompletionResultTimings = {
  prompt_n: number
  prompt_ms: number
  prompt_per_token_ms: number
  prompt_per_second: number
  predicted_n: number
  predicted_ms: number
  predicted_per_token_ms: number
  predicted_per_second: number
}

export type NativeCompletionResult = {
  /**
   * Original text (Ignored reasoning_content / tool_calls)
   */
  text: string
  /**
   * Reasoning content (parsed for reasoning model)
   */
  reasoning_content: string
  /**
   * Tool calls
   */
  tool_calls: Array<{
    type: 'function'
    function: {
      name: string
      arguments: string
    }
    id?: string
  }>
  /**
   * Content text (Filtered text by reasoning_content / tool_calls)
   */
  content: string

  tokens_predicted: number
  tokens_evaluated: number
  truncated: boolean
  stopped_eos: boolean
  stopped_word: string
  stopped_limit: number
  stopping_word: string
  tokens_cached: number
  timings: NativeCompletionResultTimings

  completion_probabilities?: Array<NativeCompletionTokenProb>
}

export type NativeTokenizeResult = {
  tokens: Array<number>
  // New multimodal support
  has_media?: boolean
  bitmap_hashes?: Array<string>
  chunk_pos?: Array<number>
  chunk_pos_media?: Array<number>
}

export type NativeEmbeddingResult = {
  embedding: Array<number>
}

// New TTS/Audio types
export type NativeTTSType = {
  type: number // TTS_UNKNOWN = -1, TTS_OUTETTS_V0_2 = 1, TTS_OUTETTS_V0_3 = 2
}

export type NativeAudioCompletionResult = {
  formatted_prompt: string
}

export type NativeAudioTokensResult = {
  tokens: Array<number>
}

export type NativeAudioDecodeResult = {
  audio_data: Array<number> // Float array of audio samples
}

export type NativeLlamaContext = {
  contextId: number
  model: {
    desc: string
    size: number
    nEmbd: number
    nParams: number
    chatTemplates: {
      llamaChat: boolean // Chat template in llama-chat.cpp
      minja: {
        // Chat template supported by minja.hpp
        default: boolean
        defaultCaps: {
          tools: boolean
          toolCalls: boolean
          toolResponses: boolean
          systemRole: boolean
          parallelToolCalls: boolean
          toolCallId: boolean
        }
        toolUse: boolean
        toolUseCaps: {
          tools: boolean
          toolCalls: boolean
          toolResponses: boolean
          systemRole: boolean
          parallelToolCalls: boolean
          toolCallId: boolean
        }
      }
    }
    metadata: Object
    isChatTemplateSupported: boolean // Deprecated
  }
  /**
   * Loaded library name for Android
   */
  androidLib?: string
  gpu: boolean
  reasonNoGPU: string
}

export type NativeSessionLoadResult = {
  tokens_loaded: number
  prompt: string
}

export type NativeLlamaChatMessage = {
  role: string
  content: string
}

export type JinjaFormattedChatResult = {
  prompt: string
  chat_format?: number
  grammar?: string
  grammar_lazy?: boolean
  grammar_triggers?: Array<{
    type: number
    value: string
    token: number
  }>
  preserved_tokens?: Array<string>
  additional_stops?: Array<string>
}

export interface Spec extends TurboModule {
  toggleNativeLog(enabled: boolean): Promise<void>
  setContextLimit(limit: number): Promise<void>

  modelInfo(path: string, skip?: string[]): Promise<Object>
  initContext(
    contextId: number,
    params: NativeContextParams,
  ): Promise<NativeLlamaContext>

  getFormattedChat(
    contextId: number,
    messages: string,
    chatTemplate?: string,
    params?: {
      jinja?: boolean
      json_schema?: string
      tools?: string
      parallel_tool_calls?: string
      tool_choice?: string
    },
  ): Promise<JinjaFormattedChatResult | string>
  loadSession(
    contextId: number,
    filepath: string,
  ): Promise<NativeSessionLoadResult>
  saveSession(
    contextId: number,
    filepath: string,
    size: number,
  ): Promise<number>
  completion(
    contextId: number,
    params: NativeCompletionParams,
  ): Promise<NativeCompletionResult>
  multimodalCompletion(
    contextId: number,
    prompt: string,
    mediaPaths: string[],
    params: NativeCompletionParams,
  ): Promise<NativeCompletionResult>
  stopCompletion(contextId: number): Promise<void>
  tokenize(contextId: number, text: string, mediaPaths?: string[]): Promise<NativeTokenizeResult>
  detokenize(contextId: number, tokens: number[]): Promise<string>
  embedding(
    contextId: number,
    text: string,
    params: NativeEmbeddingParams,
  ): Promise<NativeEmbeddingResult>
  bench(
    contextId: number,
    pp: number,
    tg: number,
    pl: number,
    nr: number,
  ): Promise<string>

  applyLoraAdapters(
    contextId: number,
    loraAdapters: Array<{ path: string; scaled?: number }>,
  ): Promise<void>
  removeLoraAdapters(contextId: number): Promise<void>
  getLoadedLoraAdapters(
    contextId: number,
  ): Promise<Array<{ path: string; scaled?: number }>>

  // New Multimodal Methods
  initMultimodal(
    contextId: number,
    mmprojPath: string,
    useGpu?: boolean,
  ): Promise<boolean>
  isMultimodalEnabled(contextId: number): Promise<boolean>
  isMultimodalSupportVision(contextId: number): Promise<boolean>
  isMultimodalSupportAudio(contextId: number): Promise<boolean>
  releaseMultimodal(contextId: number): Promise<void>

  // New TTS/Vocoder Methods
  initVocoder(
    contextId: number,
    vocoderModelPath: string,
  ): Promise<boolean>
  isVocoderEnabled(contextId: number): Promise<boolean>
  getTTSType(contextId: number): Promise<NativeTTSType>
  getFormattedAudioCompletion(
    contextId: number,
    speakerJsonStr: string,
    textToSpeak: string,
  ): Promise<NativeAudioCompletionResult>
  getAudioCompletionGuideTokens(
    contextId: number,
    textToSpeak: string,
  ): Promise<NativeAudioTokensResult>
  decodeAudioTokens(
    contextId: number,
    tokens: number[],
  ): Promise<NativeAudioDecodeResult>
  releaseVocoder(contextId: number): Promise<void>

  releaseContext(contextId: number): Promise<void>

  releaseAllContexts(): Promise<void>
}

export default TurboModuleRegistry.get<Spec>('Cactus') as Spec
