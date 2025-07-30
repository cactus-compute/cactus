import { CactusLM } from './lm'

import type {
  ContextParams,
  CompletionParams,
  CactusOAICompatibleMessage,
  NativeCompletionResult,
  TokenData,
} from './index'

import { Tools } from './tools'

interface Parameter {
  type: string
  description: string
  required?: boolean
}

interface CactusAgentReturn {
  agent: CactusAgent | null
  error: Error | null
  lm: CactusAgent | null
}

export interface AgentCompletionParams extends CompletionParams {
  tools?: Tools
}

export class CactusAgent extends CactusLM {
  private tools: Tools

  protected constructor(context: any) {
    super(context)
    this.tools = new Tools()
  }

  static async init(
    params: ContextParams,
    onProgress?: (progress: number) => void,
    cactusToken?: string,
    retryOptions?: { maxRetries?: number; delayMs?: number },
  ): Promise<CactusAgentReturn> {
    const result = await CactusLM.init(params, onProgress, cactusToken, retryOptions)
    
    if (result.error || !result.lm) {
      return { agent: null, error: result.error, lm: null }
    }

    const agent = Object.setPrototypeOf(result.lm, CactusAgent.prototype) as CactusAgent
    agent.tools = new Tools()
    return { agent, error: null, lm: agent }
  }

  addTool(
    func: Function,
    description: string,
    parameters: { [key: string]: Parameter }
  ): Function {
    return this.tools.add(func, description, parameters)
  }

  getTools(): Tools {
    return this.tools
  }

  async completionWithTools(
    messages: CactusOAICompatibleMessage[],
    params: AgentCompletionParams = {},
    callback?: (data: TokenData) => void,
    recursionCount: number = 0,
    recursionLimit: number = 3
  ): Promise<NativeCompletionResult> {
    const tools = params.tools || this.tools
    
    if (!messages?.length) {
      return this.completion([], params, callback)
    }
    if (!tools || tools.getSchemas().length === 0) {
      return this.completion(messages, params, callback)
    }
    if (recursionCount >= recursionLimit) {
      return this.completion(messages, {
        ...params,
        jinja: true,
        tools: tools.getSchemas()
      }, callback)
    }

    const { newMessages, requiresReset } =
      this.conversationHistoryManager.processNewMessages(messages)

    if (requiresReset) {
      this.context?.rewind()
      this.conversationHistoryManager.reset()
    }

    const result = await this.context.completionWithTools({
      ...params,
      messages: newMessages.length > 0 ? newMessages : messages,
      tools: tools
    }, callback, recursionCount, recursionLimit)

    this.conversationHistoryManager.update(
      newMessages.length > 0 ? newMessages : messages,
      {
        role: 'assistant',
        content: result.content,
        tool_calls: result.tool_calls
      }
    )

    return result
  }
}