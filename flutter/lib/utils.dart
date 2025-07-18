import './types.dart';

class ProcessedMessages {
  final List<ChatMessage> newMessages;
  final bool requiresReset;

  ProcessedMessages({required this.newMessages, required this.requiresReset});
}

class ConversationHistoryManager {
  List<ChatMessage> _history = [];

  ProcessedMessages processNewMessages(List<ChatMessage> fullMessageHistory) {
    bool divergent = fullMessageHistory.length < _history.length;
    if (!divergent) {
      for (int i = 0; i < _history.length; i++) {
        if (_history[i] != fullMessageHistory[i]) { // Assumes ChatMessage has == override; add if needed
          divergent = true;
          break;
        }
      }
    }

    if (divergent) {
      return ProcessedMessages(newMessages: fullMessageHistory, requiresReset: true);
    }

    final newMessages = fullMessageHistory.sublist(_history.length);
    return ProcessedMessages(newMessages: newMessages, requiresReset: false);
  }

  void update(List<ChatMessage> newMessages, ChatMessage assistantResponse) {
    _history.addAll([...newMessages, assistantResponse]);
  }

  void reset() {
    _history = [];
  }
}