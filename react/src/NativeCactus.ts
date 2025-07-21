import type { TurboModule } from 'react-native'
import { TurboModuleRegistry } from 'react-native'

export type NativeEmbeddingParams = {
  embd_normalize?: number
}

export type NativeContextParams = {
  model: string
  chat_template?: string
  reasoning_format?: string
  is_model_asset?: boolean
  use_progress_callback?: boolean
  n_ctx?: number
  n_batch?: number
  n_ubatch?: number
  n_threads?: number
  n_gpu_layers?: number
  no_gpu_devices?: boolean
  flash_attn?: boolean
  cache_type_k?: string
  cache_type_v?: string
  use_mlock?: boolean
  use_mmap?: boolean
  vocab_only?: boolean
  lora?: string
  lora_scaled?: number
  lora_list?: Array<{ path: string; scaled?: number }>
  rope_freq_base?: number
  rope_freq_scale?: number
  pooling_type?: number
  embedding?: boolean
  embd_normalize?: number
}

export type NativeCompletionParams = {
  prompt: string
  n_threads?: number
  json_schema?: string
  grammar?: string
  grammar_lazy?: boolean
  grammar_triggers?: Array<{
    type: number
    value: string
    token: number
  }>
  preserved_tokens?: Array<string>
  chat_format?: number
  stop?: Array<string>
  n_predict?: number
  n_probs?: number
  top_k?: number
  top_p?: number
  min_p?: number
  xtc_probability?: number
  xtc_threshold?: number
  typical_p?: number
  temperature?: number
  penalty_last_n?: number
  penalty_repeat?: number
  penalty_freq?: number
  penalty_present?: number
  mirostat?: number
  mirostat_tau?: number
  mirostat_eta?: number
  dry_multiplier?: number
  dry_base?: number
  dry_allowed_length?: number
  dry_penalty_last_n?: number
  dry_sequence_breakers?: Array<string>
  top_n_sigma?: number
  ignore_eos?: boolean
  logit_bias?: Array<Array<number>>
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
  text: string
  reasoning_content: string
  tool_calls: Array<{
    type: 'function'
    function: {
      name: string
      arguments: string
    }
    id?: string
  }>
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
  has_media?: boolean
  bitmap_hashes?: Array<string>
  chunk_pos?: Array<number>
  chunk_pos_media?: Array<number>
}

export type NativeEmbeddingResult = {
  embedding: Array<number>
}

export type NativeTTSType = {
  type: number
}

export type NativeAudioCompletionResult = {
  formatted_prompt: string
}

export type NativeAudioTokensResult = {
  tokens: Array<number>
}

export type NativeAudioDecodeResult = {
  audio_data: Array<number>
}

export type NativeDeviceInfo = {
  deviceId: string
  model: string
  make: string
  os: string
}

export type NativeLlamaContext = {
  contextId: number
  model: {
    desc: string
    size: number
    nEmbd: number
    nParams: number
    chatTemplates: {
      llamaChat: boolean
      minja: {
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
    isChatTemplateSupported: boolean
  }
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
  initMultimodal(
    contextId: number,
    mmprojPath: string,
    useGpu?: boolean,
  ): Promise<boolean>
  isMultimodalEnabled(contextId: number): Promise<boolean>
  isMultimodalSupportVision(contextId: number): Promise<boolean>
  isMultimodalSupportAudio(contextId: number): Promise<boolean>
  releaseMultimodal(contextId: number): Promise<void>
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
  getDeviceInfo(contextId: number): Promise<NativeDeviceInfo>
  releaseVocoder(contextId: number): Promise<void>
  rewind(contextId: number): Promise<void>
  releaseContext(contextId: number): Promise<void>
  releaseAllContexts(): Promise<void>
}

export default TurboModuleRegistry.get<Spec>('Cactus') as Spec
