import type { CactusOAICompatibleMessage } from './chat';

export interface ProcessedMessages { // result of processing a new message history
  newMessages: CactusOAICompatibleMessage[]; // new messages that should be sent to the model
  requiresReset: boolean; // flag indicating if the underlying model's state needs to be rewound
}

export class ConversationHistoryManager {
  private history: CactusOAICompatibleMessage[] = [];

  /**
   * Compares a new, full message history with the internal history to determine
   * what messages are new and if a state reset is required.
   */
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
      // If diverged, the caller must reset the model and send the full history.
      return { newMessages: fullMessageHistory, requiresReset: true };
    }

    // If not diverged, only return the new messages
    const newMessages = fullMessageHistory.slice(this.history.length);
    return { newMessages, requiresReset: false };
  }

  /**
   * Updates the history with the new messages sent to the model and its response.
   */
  public update(
    newMessages: CactusOAICompatibleMessage[],
    assistantResponse: CactusOAICompatibleMessage
  ) {
    this.history.push(...newMessages, assistantResponse);
  }

  /**
   * Resets the internal history.
   */
  public reset() {
    this.history = [];
  }
}