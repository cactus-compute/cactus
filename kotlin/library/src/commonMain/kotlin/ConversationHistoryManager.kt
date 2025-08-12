package com.cactus

data class ChatMessage(
    val content: String,
    val role: String,
    val timestamp: Long? = null
)

data class ProcessedMessages(
    val newMessages: List<ChatMessage>,
    val requiresReset: Boolean
)

class ConversationHistoryManager {
    private val _history = mutableListOf<ChatMessage>()

    fun processNewMessages(fullMessageHistory: List<ChatMessage>): ProcessedMessages {
        var divergent = fullMessageHistory.size < _history.size
        if (!divergent) {
            for (i in _history.indices) {
                if (_history[i] != fullMessageHistory[i]) {
                    divergent = true
                    break
                }
            }
        }

        if (divergent) {
            return ProcessedMessages(newMessages = fullMessageHistory, requiresReset = true)
        }

        val newMessages = fullMessageHistory.drop(_history.size)
        return ProcessedMessages(newMessages = newMessages, requiresReset = false)
    }

    fun update(newMessages: List<ChatMessage>, assistantResponse: ChatMessage) {
        _history.addAll(newMessages)
        _history.add(assistantResponse)
    }

    fun reset() {
        _history.clear()
    }
}