import { initLlama, LlamaContext } from './index'
// @ts-ignore
import { Platform } from 'react-native'
import type {
  ContextParams,
  CompletionParams,
  CactusOAICompatibleMessage,
  NativeCompletionResult,
  EmbeddingParams,
  NativeEmbeddingResult,
} from './index'

import { Telemetry } from './telemetry'
import { setCactusToken, getVertexAIEmbedding, getTextCompletion } from './remote'
import { ConversationHistoryManager } from './chat'

interface CactusLMReturn {
  lm: CactusLM | null
  error: Error | null
}

export class CactusLM {
  protected context: LlamaContext
  protected conversationHistoryManager: ConversationHistoryManager
  private initParams: ContextParams
  private static _initCache: Map<string, Promise<CactusLMReturn>> = new Map();

  private static getCacheKey(params: ContextParams, cactusToken?: string, retryOptions?: { maxRetries?: number; delayMs?: number }): string {
    return JSON.stringify({ params, cactusToken, retryOptions });
  }

  protected constructor(context: LlamaContext, initParams: ContextParams) {
    this.context = context
    this.initParams = initParams
    this.conversationHistoryManager = new ConversationHistoryManager()
  }

  private static isContextNotFoundError(e: unknown): boolean {
    const message = String((e as any)?.message ?? e ?? '')
    return /context not found/i.test(message)
  }

  private async reinit(): Promise<void> {
    const newContext = await initLlama(this.initParams)
    this.context = newContext
    this.conversationHistoryManager.reset()
  }

  private async run<T>(op: () => Promise<T>): Promise<T> {
    try {
      return await op()
    } catch (e) {
      if (!CactusLM.isContextNotFoundError(e)) throw e
      await this.reinit()
      return await op()
    }
  }

  static async init(
    params: ContextParams,
    onProgress?: (progress: number) => void,
    cactusToken?: string,
    retryOptions?: { maxRetries?: number; delayMs?: number },
  ): Promise<CactusLMReturn> {

    if (cactusToken) {
      setCactusToken(cactusToken);
    }

    const key = CactusLM.getCacheKey(params, cactusToken, retryOptions);
    if (CactusLM._initCache.has(key)) {
      return CactusLM._initCache.get(key)!;
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
            const context = await initLlama(config, onProgress);
            return { lm: new CactusLM(context, config), error: null };
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
          return { lm: null, error: lastError };
        }
      }
      return { lm: null, error: new Error('Failed to initialize CactusLM after all retries') };
    })();

    CactusLM._initCache.set(key, initPromise);

    const result = await initPromise;
    // Cache only while in-flight; never cache resolved instances
    CactusLM._initCache.delete(key);
    return result;
  }

  completion = async (
    messages: CactusOAICompatibleMessage[],
    params: CompletionParams & { mode?: string } = {},
    callback?: (data: any) => void,
  ): Promise<NativeCompletionResult> => {
    const mode = params.mode || 'local';

    let result: NativeCompletionResult;
    let lastError: Error | null = null;

    if (mode === 'remote') {
      result = await this._handleRemoteCompletion(messages, callback);
    } else if (mode === 'local') {
      result = await this._handleLocalCompletion(messages, params, callback);
    } else if (mode === 'localfirst') {
      try {
        result = await this._handleLocalCompletion(messages, params, callback);
      } catch (e) {
        lastError = e as Error;
        try {
          result = await this._handleRemoteCompletion(messages, callback);
        } catch (remoteError) {
          throw lastError;
        }
      }
    } else if (mode === 'remotefirst') {
      try {
        result = await this._handleRemoteCompletion(messages, callback);
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
    params: CompletionParams,
    callback?: (data: any) => void,
  ): Promise<NativeCompletionResult> => {
    const { newMessages, requiresReset } =
      this.conversationHistoryManager.processNewMessages(messages);

    if (requiresReset) {
      await this.run(() => this.context.rewind())
      this.conversationHistoryManager.reset();
    }

    if (newMessages.length === 0) {
      console.warn('No messages to complete!');
    }

    const result = await this.run(() =>
      this.context.completion({ messages: newMessages, ...params }, callback),
    )

    this.conversationHistoryManager.update(newMessages, {
      role: 'assistant',
      content: result.content,
    });

    return result;
  }

  private async _handleRemoteCompletion(
    messages: CactusOAICompatibleMessage[],
    callback?: (data: any) => void,
  ): Promise<NativeCompletionResult> {
    const prompt = messages.map((m) => `${m.role}: ${m.content}`).join('\n');
    
    const responseText = await getTextCompletion(messages);
    
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

  async embedding(
    text: string,
    params?: EmbeddingParams,
    mode: string = 'local',
  ): Promise<NativeEmbeddingResult> {
    let result: NativeEmbeddingResult;
    let lastError: Error | null = null;

    if (mode === 'remote') {
      result = await this._handleRemoteEmbedding(text);
    } else if (mode === 'local') {
      result = await this._handleLocalEmbedding(text, params);
    } else if (mode === 'localfirst') {
      try {
        result = await this._handleLocalEmbedding(text, params);
      } catch (e) {
        lastError = e as Error;
        try {
          result = await this._handleRemoteEmbedding(text);
        } catch (remoteError) {
          throw lastError;
        }
      }
    } else if (mode === 'remotefirst') {
      try {
        result = await this._handleRemoteEmbedding(text);
      } catch (e) {
        lastError = e as Error;
        try {
          result = await this._handleLocalEmbedding(text, params);
        } catch (localError) {
          throw lastError;
        }
      }
    } else {
      throw new Error('Invalid mode: ' + mode + '. Must be "local", "remote", "localfirst", or "remotefirst"');
    }
    return result;
  }

  protected async _handleLocalEmbedding(text: string, params?: EmbeddingParams): Promise<NativeEmbeddingResult> {
    return this.run(() => this.context.embedding(text, params))
  }

  protected async _handleRemoteEmbedding(text: string): Promise<NativeEmbeddingResult> {
    const embeddingValues = await getVertexAIEmbedding(text);
    return {
      embedding: embeddingValues,
    };
  }

  rewind = async (): Promise<void> => {
    return this.run(() => this.context.rewind())
  }

  async release(): Promise<void> {
    try {
      return await this.context.release()
    } catch (e) {
      // Treat missing context as already released
      if (CactusLM.isContextNotFoundError(e)) return
      throw e
    }
  }

  async stopCompletion(): Promise<void> {
    return await this.run(() => this.context.stopCompletion())
  }

  isJinjaSupported(): boolean {
    return this.context.isJinjaSupported();
  }
} 