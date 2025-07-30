import type { NativeLlamaChatMessage } from './NativeCactus'

export type CactusMessagePart = {
  text?: string
}

export type CactusOAICompatibleMessage = {
  role: string
  content?: string | CactusMessagePart[] | any
  tool_calls?: any[]
  tool_call_id?: string
}

export function formatChat(
  messages: CactusOAICompatibleMessage[],
): NativeLlamaChatMessage[] {
  const chat: NativeLlamaChatMessage[] = []

  messages.forEach((currMsg) => {
    const role: string = currMsg.role || ''

    let content: string = ''
    if ('content' in currMsg) {
      if (typeof currMsg.content === 'string') {
        ;({ content } = currMsg)
      } else if (Array.isArray(currMsg.content)) {
        currMsg.content.forEach((part) => {
          if ('text' in part) {
            content += `${content ? '\n' : ''}${part.text}`
          }
        })
      } else {
        throw new TypeError(
          "Invalid 'content' type (ref: https://github.com/ggerganov/llama.cpp/issues/8367)",
        )
      }
    } else {
      throw new Error(
        "Missing 'content' (ref: https://github.com/ggerganov/llama.cpp/issues/8367)",
      )
    }

    chat.push({ role, content })
  })
  return chat
}

export interface ProcessedMessages {
  newMessages: CactusOAICompatibleMessage[];
  requiresReset: boolean;
}

export class ConversationHistoryManager {
  private history: CactusOAICompatibleMessage[] = [];

  public processNewMessages(
    fullMessageHistory: CactusOAICompatibleMessage[]
  ): ProcessedMessages {
    let divergent = fullMessageHistory.length < this.history.length;
    if (!divergent) {
      for (let i = 0; i < this.history.length; i++) {
        if (JSON.stringify(this.history[i]) !== JSON.stringify(fullMessageHistory[i])) {
          divergent = true;
          break;
        }
      }
    }

    if (divergent) {
      return { newMessages: fullMessageHistory, requiresReset: true };
    }

    const newMessages = fullMessageHistory.slice(this.history.length);
    return { newMessages, requiresReset: false };
  }

  public update(
    newMessages: CactusOAICompatibleMessage[],
    assistantResponse: CactusOAICompatibleMessage
  ) {
    this.history.push(...newMessages, assistantResponse);
  }

  public reset() {
    this.history = [];
  }

  public getMessages(): CactusOAICompatibleMessage[] {
    return this.history;
  }
}
