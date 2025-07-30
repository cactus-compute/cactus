import { initLlama, LlamaContext } from './index'
import type {
  ContextParams,
  CompletionParams,
  CactusOAICompatibleMessage,
  NativeCompletionResult,
  TokenData,
} from './index'
import { Tools, parseAndExecuteTool } from './tools'
import { Telemetry } from './telemetry'

interface AgentReturn {
  agent: Agent | null
  error: Error | null
}

interface AgentCompletionParams extends Omit<CompletionParams, 'messages'> {
  recursionLimit?: number
}

export class Agent {
  private context: LlamaContext
  private tools: Tools
  private conversationHistory: CactusOAICompatibleMessage[] = []
  private systemPromptCached: boolean = false

  private constructor(context: LlamaContext, tools: Tools, systemPrompt?: string) {
    this.context = context
    this.tools = tools
    
    if (systemPrompt) {
      this.conversationHistory.push({
        role: 'system',
        content: String(systemPrompt).trim() || 'You are a helpful assistant.'
      })
    }
  }

  static async init(
    params: ContextParams,
    tools: Tools,
    onProgress?: (progress: number) => void,
    systemPrompt?: string,
  ): Promise<AgentReturn> {
    const configs = [
      params,
      { ...params, n_gpu_layers: 0 }
    ]

    for (const config of configs) {
      try {
        const context = await initLlama(config, onProgress)
        const agent = new Agent(context, tools, systemPrompt)
        
        if (systemPrompt) {
          await agent._cacheSystemPrompt()
        }
        
        return { agent, error: null }
      } catch (e) {
        Telemetry.error(e as Error, {
          n_gpu_layers: config.n_gpu_layers ?? null,
          n_ctx: config.n_ctx ?? null,
          model: config.model ?? null,
        })
        if (configs.indexOf(config) === configs.length - 1) {
          return { agent: null, error: e as Error }
        }
      }
    }
    return { agent: null, error: new Error('Failed to initialize Agent') }
  }

  private async _cacheSystemPrompt(): Promise<void> {
    if (this.systemPromptCached) {
      return
    }

    try {
      Telemetry.info('Starting system prompt caching...')
      const systemMessage = this.conversationHistory.find(msg => msg.role === 'system')
      if (!systemMessage) {
        Telemetry.warn('No system message found for caching')
        return
      }
      
      const cachePromise = this.context.completion({
        messages: [systemMessage],
        jinja: true,
        tools: this.tools.getSchemas(),
        n_predict: 1,
      })
      
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error('System prompt caching timeout')), 30000)
      })
      
      await Promise.race([cachePromise, timeoutPromise])
      
      this.systemPromptCached = true
      Telemetry.info('System prompt cached successfully')
    } catch (error) {
      console.error('Failed to cache system prompt:', error)
      Telemetry.info('Continuing without system prompt cache - performance may be slower')
      this.systemPromptCached = true
    }
  }

  async completionWithTools(
    userMessage: string,
    params: AgentCompletionParams = {},
    callback?: (data: TokenData) => void
  ): Promise<NativeCompletionResult> {
    const recursionLimit = params.recursionLimit ?? 3
    
    const newUserMessage: CactusOAICompatibleMessage = {
      role: 'user',
      content: String(userMessage || '').trim() || 'Hello'
    }
    this.conversationHistory.push(newUserMessage)

    return await this._executeCompletionLoop(params, callback, 0, recursionLimit)
  }

  private async _executeCompletionLoop(
    params: AgentCompletionParams,
    callback?: (data: TokenData) => void,
    recursionCount: number = 0,
    recursionLimit: number = 3
  ): Promise<NativeCompletionResult> {
    if (recursionCount >= recursionLimit) {
      Telemetry.log(`Recursion limit reached (${recursionCount}/${recursionLimit}), returning final completion`, 'info')
      
      const contextMessages = this._buildContextMessages()
      return await this.context.completion({
        ...params,
        messages: contextMessages,
        jinja: true,
        tools: this.tools.getSchemas()
      }, callback)
    }

    const contextMessages = this._buildContextMessages()
    
    console.log('Calling completion with context messages:', contextMessages.length)
    
    const result = await this.context.completion({
      ...params,
      messages: contextMessages,
      jinja: true,
      tools: this.tools.getSchemas()
    }, callback)
    
    console.log('Completion result:', result)

    const { toolCalled, toolName, toolInput, toolOutput } = 
      await this._parseAndExecuteToolEnhanced(result, this.tools)

    if (toolCalled && toolName && toolInput) {
      // Use proper tool calling pattern like existing completionWithTools
      const assistantMessage = {
        role: 'assistant',
        content: result.content,
        tool_calls: result.tool_calls
      } as CactusOAICompatibleMessage
      this.conversationHistory.push(assistantMessage)

      const toolCallId = result.tool_calls?.[0]?.id
      const toolMessage = {
        role: 'tool',
        content: JSON.stringify(toolOutput),
        tool_call_id: toolCallId
      } as CactusOAICompatibleMessage
      this.conversationHistory.push(toolMessage)

      console.log('Tool executed, continuing conversation loop')
      
      return await this._executeCompletionLoop(
        params,
        callback,
        recursionCount + 1,
        recursionLimit
      )
    }

    const content = String(result.content || (result as any).text || '').trim() || 'I understand.'
    const assistantMessage: CactusOAICompatibleMessage = {
      role: 'assistant',
      content: content
    }
    this.conversationHistory.push(assistantMessage)

    return result
  }

  private _buildContextMessages(): CactusOAICompatibleMessage[] {
    console.log('Raw conversation history:', this.conversationHistory.map(m => ({ role: m.role, content: m.content?.substring(0, 30) + '...' })))
    
    // Basic context window management - keep last 20 messages plus system
    const maxMessages = 20
    let messages = [...this.conversationHistory]
    
    const systemMessage = messages.find(msg => msg.role === 'system')
    const nonSystemMessages = messages.filter(msg => msg.role !== 'system')
    
    // Keep recent messages if we exceed limit
    if (nonSystemMessages.length > maxMessages) {
      const recentMessages = nonSystemMessages.slice(-maxMessages)
      messages = systemMessage ? [systemMessage, ...recentMessages] : recentMessages
    }
    
    const result: CactusOAICompatibleMessage[] = []
    
    // Add system message first
    if (systemMessage) {
      result.push({
        ...systemMessage,
        content: String(systemMessage.content || '').trim() || 'You are a helpful assistant.'
      })
    }
    
    // Add user/assistant messages only (filter out tool messages to maintain alternation)
    const userAssistantMessages = messages.filter(msg => 
      msg.role === 'user' || msg.role === 'assistant'
    )
    
    for (const message of userAssistantMessages) {
      const content = String(message.content || '').trim()
      const msgAny = message as any
      if (content || msgAny.tool_calls) {
        result.push({ ...message, content })
      }
    }
    
    console.log('Built context messages:', result.map(m => ({ role: m.role, content: m.content?.substring(0, 30) + '...' })))
    
    return result
  }

  private async _parseAndExecuteToolEnhanced(result: NativeCompletionResult, tools: Tools): Promise<{toolCalled: boolean, toolName?: string, toolInput?: any, toolOutput?: any}> {
    const standardResult = await parseAndExecuteTool(result, tools)
    if (standardResult.toolCalled) {
      return standardResult
    }
    
    if (result.content || (result as any).text) {
      const textContent = result.content || (result as any).text
      try {
        const cleanText = textContent
          .replace(/<end_of_turn>/g, '')
          .replace(/<eos>/g, '')
          .replace(/\n/g, ' ')
          .trim()
        
        console.log('Cleaned text for parsing:', cleanText)
        
        const toolCallMatch = cleanText.match(/\{[\s\S]*?"tool_call"[\s\S]*?\}(?=\s*$|<)/)
        if (toolCallMatch) {
          console.log('Found tool call match:', toolCallMatch[0])
          const toolCallJson = JSON.parse(toolCallMatch[0])
          const toolName = toolCallJson.tool_call.name
          const toolInput = toolCallJson.tool_call.arguments
          
          console.log('Calling tool (text parsed):', toolName, toolInput)
          const toolOutput = await tools.execute(toolName, toolInput)
          console.log('Tool called result:', toolOutput)
          
          return {
            toolCalled: true,
            toolName,
            toolInput,
            toolOutput
          }
        }
        
        try {
          const directJson = JSON.parse(cleanText)
          if (directJson.tool_call) {
            const toolName = directJson.tool_call.name
            const toolInput = directJson.tool_call.arguments
            
            console.log('Calling tool (direct JSON):', toolName, toolInput)
            const toolOutput = await tools.execute(toolName, toolInput)
            console.log('Tool called result:', toolOutput)
            
            return {
              toolCalled: true,
              toolName,
              toolInput,
              toolOutput
            }
          }
        } catch (directParseError) {
          console.log('Direct JSON parse failed, trying to fix incomplete JSON')
          
          try {
            let fixedText = cleanText
            
            const openBraces = (fixedText.match(/\{/g) || []).length
            const closeBraces = (fixedText.match(/\}/g) || []).length
            const missingBraces = openBraces - closeBraces
            
            for (let i = 0; i < missingBraces; i++) {
              fixedText += '}'
            }
            
            if (fixedText.includes('"description": "') && !fixedText.includes('"}')) {
              fixedText = fixedText.replace(/"description": "([^"]*?)$/, '"description": "$1"')
            }
            
            console.log('Attempting to parse fixed JSON:', fixedText)
            const fixedJson = JSON.parse(fixedText)
            
            if (fixedJson.tool_call) {
              const toolName = fixedJson.tool_call.name
              const toolInput = fixedJson.tool_call.arguments
              
              console.log('Calling tool (fixed JSON):', toolName, toolInput)
              const toolOutput = await tools.execute(toolName, toolInput)
              console.log('Tool called result:', toolOutput)
              
              return {
                toolCalled: true,
                toolName,
                toolInput,
                toolOutput
              }
            }
          } catch (fixError) {
            console.log('Failed to fix incomplete JSON:', fixError)
          }
        }
        
      } catch (error) {
        console.error('Error parsing text-based tool call:', error)
      }
    }
    
    console.log('No tool calls found in any format')
    return {toolCalled: false}
  }

  getConversationHistory(): CactusOAICompatibleMessage[] {
    return [...this.conversationHistory]
  }

  clearHistory(): void {
    const systemMessages = this.conversationHistory.filter(msg => msg.role === 'system')
    this.conversationHistory = systemMessages
  }

  addTool(
    func: Function,
    description: string,
    parameters: { [name: string]: any }
  ): void {
    this.tools.add(func, description, parameters)
  }

  async rewind(): Promise<void> {
    // @ts-ignore
    return this.context?.rewind()
  }

  async release(): Promise<void> {
    return this.context.release()
  }
}

export type { AgentReturn, AgentCompletionParams } 