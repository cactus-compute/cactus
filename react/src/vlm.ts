import {
  initLlama,
  initMultimodal,
  multimodalCompletion,
  LlamaContext,
  type ContextParams,
  type CompletionParams,
  type CactusOAICompatibleMessage,
  type NativeCompletionResult,
} from './index'

import { Telemetry } from './telemetry'
import { setCactusToken, getTextCompletion, getVisionCompletion } from './remote'
import { ConversationHistoryManager } from './chat'

interface CactusVLMReturn {
  vlm: CactusVLM | null
  error: Error | null
}

export type VLMContextParams = ContextParams & {
  mmproj: string
}

export type VLMCompletionParams = Omit<CompletionParams, 'prompt'> & {
  images?: string[]
  mode?: string
}

export class CactusVLM {
  private context: LlamaContext
  protected conversationHistoryManager: ConversationHistoryManager

  // see CactusLM for detailed docs
  private static _initCache: Map<string, Promise<CactusVLMReturn>> = new Map();

  private static getCacheKey(params: VLMContextParams, cactusToken?: string, retryOptions?: { maxRetries?: number; delayMs?: number }): string {
    return JSON.stringify({ params, cactusToken, retryOptions });
  }

  private constructor(context: LlamaContext) {
    this.context = context
    this.conversationHistoryManager = new ConversationHistoryManager()
  }

  static async init(
    params: VLMContextParams,
    onProgress?: (progress: number) => void,
    cactusToken?: string,
    retryOptions?: { maxRetries?: number; delayMs?: number },
  ): Promise<CactusVLMReturn> {
    if (cactusToken) {
      setCactusToken(cactusToken);
    }

    const key = CactusVLM.getCacheKey(params, cactusToken, retryOptions);
    if (CactusVLM._initCache.has(key)) {
      return CactusVLM._initCache.get(key)!;
    }

    const initPromise = (async () => {
      const maxRetries = retryOptions?.maxRetries ?? 3;
      const delayMs = retryOptions?.delayMs ?? 1000;

      const configs = [
        params,
        { ...params, n_gpu_layers: 0 } 
      ];

      const sleep = (ms: number): Promise<void> => {
        return new Promise(resolve => {
          const start = Date.now();
          const wait = () => {
            if (Date.now() - start >= ms) {
              resolve();
            } else {
              Promise.resolve().then(wait);
            }
          };
          wait();
        });
      };

      for (const config of configs) {
        let lastError: Error | null = null;

        for (let attempt = 1; attempt <= maxRetries; attempt++) {
          try {
            const context = await initLlama(config, onProgress)
            await initMultimodal(context.id, params.mmproj, false)
            return {vlm: new CactusVLM(context), error: null}
          } catch (e) {
            lastError = e as Error;
            const isLastConfig = configs.indexOf(config) === configs.length - 1;
            const isLastAttempt = attempt === maxRetries;

            Telemetry.error(e as Error, {
              n_gpu_layers: config.n_gpu_layers ?? null,
              n_ctx: config.n_ctx ?? null,
              model: config.model ?? null,
            });

            if (!isLastAttempt) {
              const delay = delayMs * Math.pow(2, attempt - 1);
              await sleep(delay);
            } else if (!isLastConfig) {
              break;
            }
          }
        }

        if (configs.indexOf(config) === configs.length - 1 && lastError) {
          return {vlm: null, error: lastError}
        }
      }

      return {vlm: null, error: new Error('Failed to initialize CactusVLM')}
    })();

    CactusVLM._initCache.set(key, initPromise);

    const result = await initPromise;
    if (result.error) {
      CactusVLM._initCache.delete(key); 
    }
    return result;
  }

  async completion(
    messages: CactusOAICompatibleMessage[],
    params: VLMCompletionParams = {},
    callback?: (data: any) => void,
  ): Promise<NativeCompletionResult> {
    const mode = params.mode || 'local';

    let result: NativeCompletionResult;
    let lastError: Error | null = null;

    if (mode === 'remote') {
      result = await this._handleRemoteCompletion(messages, params, callback);
    } else if (mode === 'local') {
      result = await this._handleLocalCompletion(messages, params, callback);
    } else if (mode === 'localfirst') {
      try {
        result = await this._handleLocalCompletion(messages, params, callback);
      } catch (e) {
        lastError = e as Error;
        try {
          result = await this._handleRemoteCompletion(messages, params, callback);
        } catch (remoteError) {
          throw lastError;
        }
      }
    } else if (mode === 'remotefirst') {
      try {
        result = await this._handleRemoteCompletion(messages, params, callback);
      } catch (e) {
        lastError = e as Error;
        try {
          result = await this._handleLocalCompletion(messages, params, callback);
        } catch (localError) {
          throw lastError;
        }
      }
    } else {
      throw new Error('Invalid mode: ' + mode + '. Must be "local", "remote", "localfirst", or "remotefirst"');
    }

    return result;
  }

  private _handleLocalCompletion = async(
    messages: CactusOAICompatibleMessage[],
    params: VLMCompletionParams,
    callback?: (data: any) => void,
  ): Promise<NativeCompletionResult> => {
    const { newMessages, requiresReset } =
      this.conversationHistoryManager.processNewMessages(messages);

    if (requiresReset) {
      this.context?.rewind();
      this.conversationHistoryManager.reset();
    }

    if (newMessages.length === 0) {
      console.warn('No messages to complete!');
    }

    let result: NativeCompletionResult;

    if (params.images && params.images.length > 0) {
      const formattedPrompt = await this.context.getFormattedChat(newMessages)
      const prompt =
        typeof formattedPrompt === 'string'
          ? formattedPrompt
          : formattedPrompt.prompt
      result = await multimodalCompletion(
        this.context.id,
        prompt,
        params.images,
        { ...params, prompt, emit_partial_completion: !!callback },
      )
    } else {
      result = await this.context.completion({ messages: newMessages, ...params }, callback)
    }

    this.conversationHistoryManager.update(newMessages, {
      role: 'assistant',
      content: result.content || result.text,
    });

    return result;
  }

  private async _handleRemoteCompletion(
    messages: CactusOAICompatibleMessage[],
    params: VLMCompletionParams,
    callback?: (data: any) => void,
  ): Promise<NativeCompletionResult> {
    const prompt = messages.map((m) => `${m.role}: ${m.content}`).join('\n');
    const imagePath = params.images && params.images.length > 0 ? params.images[0] : '';
    
    let responseText: string;
    if (imagePath) {
      responseText = await getVisionCompletion(prompt, imagePath);
    } else {
      responseText = await getTextCompletion(prompt);
    }
    
    if (callback) {
      for (let i = 0; i < responseText.length; i++) {
        callback({ token: responseText[i] });
      }
    }
    
    return {
      text: responseText,
      reasoning_content: '',
      tool_calls: [],
      content: responseText,
      tokens_predicted: responseText.split(' ').length,
      tokens_evaluated: prompt.split(' ').length,
      truncated: false,
      stopped_eos: true,
      stopped_word: '',
      stopped_limit: 0,
      stopping_word: '',
      tokens_cached: 0,
      timings: {
        prompt_n: prompt.split(' ').length,
        prompt_ms: 0,
        prompt_per_token_ms: 0,
        prompt_per_second: 0,
        predicted_n: responseText.split(' ').length,
        predicted_ms: 0,
        predicted_per_token_ms: 0,
        predicted_per_second: 0,
      },
    };
  }

  async rewind(): Promise<void> {
    return this.context?.rewind()
  }

  async release(): Promise<void> {
    return this.context.release()
  }

  async stopCompletion(): Promise<void> {
    return await this.context.stopCompletion()
  }
} 