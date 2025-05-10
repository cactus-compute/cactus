import { NativeEventEmitter, DeviceEventEmitter, Platform } from 'react-native'
import type { DeviceEventEmitterStatic } from 'react-native'
import Cactus from './NativeCactus'
import type {
  NativeContextParams,
  NativeLlamaContext,
  NativeCompletionParams,
  NativeCompletionTokenProb,
  NativeCompletionResult,
  NativeTokenizeResult,
  NativeEmbeddingResult,
  NativeSessionLoadResult,
  NativeEmbeddingParams,
  NativeCompletionTokenProbItem,
  NativeCompletionResultTimings,
  JinjaFormattedChatResult,
} from './NativeCactus'
import type {
  SchemaGrammarConverterPropOrder,
  SchemaGrammarConverterBuiltinRule,
} from './grammar'
import { SchemaGrammarConverter, convertJsonSchemaToGrammar } from './grammar'
import type { CactusMessagePart, CactusOAICompatibleMessage } from './chat'
import { ModelDownloader } from './modelDownloader'
import { formatChat } from './chat'

export type {
  NativeContextParams,
  NativeLlamaContext,
  NativeCompletionParams,
  NativeCompletionTokenProb,
  NativeCompletionResult,
  NativeTokenizeResult,
  NativeEmbeddingResult,
  NativeSessionLoadResult,
  NativeEmbeddingParams,
  NativeCompletionTokenProbItem,
  NativeCompletionResultTimings,
  CactusMessagePart,
  CactusOAICompatibleMessage,
  JinjaFormattedChatResult,

  // Deprecated
  SchemaGrammarConverterPropOrder,
  SchemaGrammarConverterBuiltinRule,
}

export { SchemaGrammarConverter, convertJsonSchemaToGrammar }

const EVENT_ON_INIT_CONTEXT_PROGRESS = '@Cactus_onInitContextProgress'
const EVENT_ON_TOKEN = '@Cactus_onToken'
const EVENT_ON_NATIVE_LOG = '@Cactus_onNativeLog'

let EventEmitter: NativeEventEmitter | DeviceEventEmitterStatic
if (Platform.OS === 'ios') {
  // @ts-ignore
  EventEmitter = new NativeEventEmitter(Cactus)
}
if (Platform.OS === 'android') {
  EventEmitter = DeviceEventEmitter
}

const logListeners: Array<(level: string, text: string) => void> = []

// @ts-ignore
if (EventEmitter) {
  EventEmitter.addListener(
    EVENT_ON_NATIVE_LOG,
    (evt: { level: string; text: string }) => {
      logListeners.forEach((listener) => listener(evt.level, evt.text))
    },
  )
  // Trigger unset to use default log callback
  Cactus?.toggleNativeLog?.(false)?.catch?.(() => {})
}

export type TokenData = {
  token: string
  completion_probabilities?: Array<NativeCompletionTokenProb>
}

type TokenNativeEvent = {
  contextId: number
  tokenResult: TokenData
}

export type ContextParams = Omit<
  NativeContextParams,
  'cache_type_k' | 'cache_type_v' | 'pooling_type'
> & {
  cache_type_k?:
    | 'f16'
    | 'f32'
    | 'q8_0'
    | 'q4_0'
    | 'q4_1'
    | 'iq4_nl'
    | 'q5_0'
    | 'q5_1'
  cache_type_v?:
    | 'f16'
    | 'f32'
    | 'q8_0'
    | 'q4_0'
    | 'q4_1'
    | 'iq4_nl'
    | 'q5_0'
    | 'q5_1'
  pooling_type?: 'none' | 'mean' | 'cls' | 'last' | 'rank'
}

export type EmbeddingParams = NativeEmbeddingParams

export type CompletionResponseFormat = {
  type: 'text' | 'json_object' | 'json_schema'
  json_schema?: {
    strict?: boolean
    schema: object
  }
  schema?: object // for json_object type
}

export type CompletionBaseParams = {
  prompt?: string
  messages?: CactusOAICompatibleMessage[]
  chatTemplate?: string // deprecated
  chat_template?: string
  jinja?: boolean
  tools?: object
  parallel_tool_calls?: object
  tool_choice?: string
  response_format?: CompletionResponseFormat
}
export type CompletionParams = Omit<
  NativeCompletionParams,
  'emit_partial_completion' | 'prompt'
> &
  CompletionBaseParams

export type BenchResult = {
  modelDesc: string
  modelSize: number
  modelNParams: number
  ppAvg: number
  ppStd: number
  tgAvg: number
  tgStd: number
}

const getJsonSchema = (responseFormat?: CompletionResponseFormat) => {
  if (responseFormat?.type === 'json_schema') {
    return responseFormat.json_schema?.schema
  }
  if (responseFormat?.type === 'json_object') {
    return responseFormat.schema || {}
  }
  return null
}

export class LlamaContext {
  id: number

  gpu: boolean = false

  reasonNoGPU: string = ''

  model: NativeLlamaContext['model']

  constructor({ contextId, gpu, reasonNoGPU, model }: NativeLlamaContext) {
    this.id = contextId
    this.gpu = gpu
    this.reasonNoGPU = reasonNoGPU
    this.model = model
  }

  /**
   * Load cached prompt & completion state from a file.
   */
  async loadSession(filepath: string): Promise<NativeSessionLoadResult> {
    let path = filepath
    if (path.startsWith('file://')) path = path.slice(7)
    return Cactus.loadSession(this.id, path)
  }

  /**
   * Save current cached prompt & completion state to a file.
   */
  async saveSession(
    filepath: string,
    options?: { tokenSize: number },
  ): Promise<number> {
    return Cactus.saveSession(this.id, filepath, options?.tokenSize || -1)
  }

  isLlamaChatSupported(): boolean {
    return !!this.model.chatTemplates.llamaChat
  }

  isJinjaSupported(): boolean {
    const { minja } = this.model.chatTemplates
    return !!minja?.toolUse || !!minja?.default
  }

  async getFormattedChat(
    messages: CactusOAICompatibleMessage[],
    template?: string | null,
    params?: {
      jinja?: boolean
      response_format?: CompletionResponseFormat
      tools?: object
      parallel_tool_calls?: object
      tool_choice?: string
    },
  ): Promise<JinjaFormattedChatResult | string> {
    const chat = formatChat(messages)
    const useJinja = this.isJinjaSupported() && params?.jinja
    let tmpl = this.isLlamaChatSupported() || useJinja ? undefined : 'chatml'
    if (template) tmpl = template // Force replace if provided
    const jsonSchema = getJsonSchema(params?.response_format)
    return Cactus.getFormattedChat(this.id, JSON.stringify(chat), tmpl, {
      jinja: useJinja,
      json_schema: jsonSchema ? JSON.stringify(jsonSchema) : undefined,
      tools: params?.tools ? JSON.stringify(params.tools) : undefined,
      parallel_tool_calls: params?.parallel_tool_calls
        ? JSON.stringify(params.parallel_tool_calls)
        : undefined,
      tool_choice: params?.tool_choice,
    })
  }

  async completion(
    params: CompletionParams,
    callback?: (data: TokenData) => void,
  ): Promise<NativeCompletionResult> {
    const nativeParams = {
      ...params,
      prompt: params.prompt || '',
      emit_partial_completion: !!callback,
    }
    if (params.messages) {
      // messages always win
      const formattedResult = await this.getFormattedChat(
        params.messages,
        params.chat_template || params.chatTemplate,
        {
          jinja: params.jinja,
          tools: params.tools,
          parallel_tool_calls: params.parallel_tool_calls,
          tool_choice: params.tool_choice,
        },
      )
      if (typeof formattedResult === 'string') {
        nativeParams.prompt = formattedResult || ''
      } else {
        nativeParams.prompt = formattedResult.prompt || ''
        if (typeof formattedResult.chat_format === 'number')
          nativeParams.chat_format = formattedResult.chat_format
        if (formattedResult.grammar)
          nativeParams.grammar = formattedResult.grammar
        if (typeof formattedResult.grammar_lazy === 'boolean')
          nativeParams.grammar_lazy = formattedResult.grammar_lazy
        if (formattedResult.grammar_triggers)
          nativeParams.grammar_triggers = formattedResult.grammar_triggers
        if (formattedResult.preserved_tokens)
          nativeParams.preserved_tokens = formattedResult.preserved_tokens
        if (formattedResult.additional_stops) {
          if (!nativeParams.stop) nativeParams.stop = []
          nativeParams.stop.push(...formattedResult.additional_stops)
        }
      }
    } else {
      nativeParams.prompt = params.prompt || ''
    }

    if (nativeParams.response_format && !nativeParams.grammar) {
      const jsonSchema = getJsonSchema(params.response_format)
      if (jsonSchema) nativeParams.json_schema = JSON.stringify(jsonSchema)
    }

    let tokenListener: any =
      callback &&
      EventEmitter.addListener(EVENT_ON_TOKEN, (evt: TokenNativeEvent) => {
        const { contextId, tokenResult } = evt
        if (contextId !== this.id) return
        callback(tokenResult)
      })

    if (!nativeParams.prompt) throw new Error('Prompt is required')

    const promise = Cactus.completion(this.id, nativeParams)
    return promise
      .then((completionResult) => {
        tokenListener?.remove()
        tokenListener = null
        return completionResult
      })
      .catch((err: any) => {
        tokenListener?.remove()
        tokenListener = null
        throw err
      })
  }

  stopCompletion(): Promise<void> {
    return Cactus.stopCompletion(this.id)
  }

  tokenize(text: string): Promise<NativeTokenizeResult> {
    return Cactus.tokenize(this.id, text)
  }

  detokenize(tokens: number[]): Promise<string> {
    return Cactus.detokenize(this.id, tokens)
  }

  embedding(
    text: string,
    params?: EmbeddingParams,
  ): Promise<NativeEmbeddingResult> {
    return Cactus.embedding(this.id, text, params || {})
  }

  async bench(
    pp: number,
    tg: number,
    pl: number,
    nr: number,
  ): Promise<BenchResult> {
    const result = await Cactus.bench(this.id, pp, tg, pl, nr)
    const [modelDesc, modelSize, modelNParams, ppAvg, ppStd, tgAvg, tgStd] =
      JSON.parse(result)
    return {
      modelDesc,
      modelSize,
      modelNParams,
      ppAvg,
      ppStd,
      tgAvg,
      tgStd,
    }
  }

  async applyLoraAdapters(
    loraList: Array<{ path: string; scaled?: number }>,
  ): Promise<void> {
    let loraAdapters: Array<{ path: string; scaled?: number }> = []
    if (loraList)
      loraAdapters = loraList.map((l) => ({
        path: l.path.replace(/file:\/\//, ''),
        scaled: l.scaled,
      }))
    return Cactus.applyLoraAdapters(this.id, loraAdapters)
  }

  async removeLoraAdapters(): Promise<void> {
    return Cactus.removeLoraAdapters(this.id)
  }

  async getLoadedLoraAdapters(): Promise<
    Array<{ path: string; scaled?: number }>
  > {
    return Cactus.getLoadedLoraAdapters(this.id)
  }

  async release(): Promise<void> {
    return Cactus.releaseContext(this.id)
  }
}

export async function toggleNativeLog(enabled: boolean): Promise<void> {
  return Cactus.toggleNativeLog(enabled)
}

export function addNativeLogListener(
  listener: (level: string, text: string) => void,
): { remove: () => void } {
  logListeners.push(listener)
  return {
    remove: () => {
      logListeners.splice(logListeners.indexOf(listener), 1)
    },
  }
}

export async function setContextLimit(limit: number): Promise<void> {
  return Cactus.setContextLimit(limit)
}

let contextIdCounter = 0
const contextIdRandom = () =>
  process.env.NODE_ENV === 'test' ? 0 : Math.floor(Math.random() * 100000)

const modelInfoSkip = [
  // Large fields
  'tokenizer.ggml.tokens',
  'tokenizer.ggml.token_type',
  'tokenizer.ggml.merges',
  'tokenizer.ggml.scores'
]
export async function loadLlamaModelInfo(model: string): Promise<Object> {
  let path = model
  if (path.startsWith('file://')) path = path.slice(7)
  return Cactus.modelInfo(path, modelInfoSkip)
}

const poolTypeMap = {
  // -1 is unspecified as undefined
  none: 0,
  mean: 1,
  cls: 2,
  last: 3,
  rank: 4,
}

export async function initLlama(
  {
    model,
    is_model_asset: isModelAsset,
    pooling_type: poolingType,
    lora,
    lora_list: loraList,
    ...rest
  }: ContextParams,
  onProgress?: (progress: number) => void,
): Promise<LlamaContext> {
  let path = model
  if (path.startsWith('file://')) path = path.slice(7)

  let loraPath = lora
  if (loraPath?.startsWith('file://')) loraPath = loraPath.slice(7)

  let loraAdapters: Array<{ path: string; scaled?: number }> = []
  if (loraList)
    loraAdapters = loraList.map((l) => ({
      path: l.path.replace(/file:\/\//, ''),
      scaled: l.scaled,
    }))

  const contextId = contextIdCounter + contextIdRandom()
  contextIdCounter += 1

  let removeProgressListener: any = null
  if (onProgress) {
    removeProgressListener = EventEmitter.addListener(
      EVENT_ON_INIT_CONTEXT_PROGRESS,
      (evt: { contextId: number; progress: number }) => {
        if (evt.contextId !== contextId) return
        onProgress(evt.progress)
      },
    )
  }

  const poolType = poolTypeMap[poolingType as keyof typeof poolTypeMap]
  const {
    gpu,
    reasonNoGPU,
    model: modelDetails,
    androidLib,
  } = await Cactus.initContext(contextId, {
    model: path,
    is_model_asset: !!isModelAsset,
    use_progress_callback: !!onProgress,
    pooling_type: poolType,
    lora: loraPath,
    lora_list: loraAdapters,
    ...rest,
  }).catch((err: any) => {
    removeProgressListener?.remove()
    throw err
  })
  removeProgressListener?.remove()
  return new LlamaContext({
    contextId,
    gpu,
    reasonNoGPU,
    model: modelDetails,
    androidLib,
  })
}

export async function releaseAllLlama(): Promise<void> {
  return Cactus.releaseAllContexts()
}


type DownloadOptions = {
  modelUrl?: string;
  modelFolderName?: string;
  onProgress?: (progress: number) => void;
  onSuccess?: (modelPath: string) => void;
};

/**
 * Download a model from a given URL and save it to a given folder name.
 * @param options - The options for the download.
 * @param options.modelUrl - The URL of the model to download. If not provided, the default model URL will be used.
 * @param options.modelFolderName - The folder name to save the model to. If not provided, the default model folder name will be used.
 * @param options.onProgress - A callback function that is called with the progress of the download. Returns a percentage of the download as an integer. Useful for displaying a progress bar.
 * @param options.onSuccess - A callback function that is called when the download is successful.
 * @returns A promise that resolves to the full model path.
 */
export function downloadModelIfNotExists(
  options: DownloadOptions
): Promise<string> {
  const modelDownloader = new ModelDownloader(options.modelUrl , options.modelFolderName)
  return modelDownloader.downloadModelIfNotExists(options.onProgress, options.onSuccess)
}