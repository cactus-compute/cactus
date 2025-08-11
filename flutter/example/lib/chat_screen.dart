import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import '../cactus_service.dart';
import 'package:cactus/cactus.dart';
import 'message_bubble.dart';

class ChatScreen extends StatefulWidget {
  final CactusService cactusService;

  const ChatScreen({super.key, required this.cactusService});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final urlController = TextEditingController();
  bool _hasImage = false;

  @override
  void initState() {
    super.initState();
    widget.cactusService.messages.addListener(_scrollToBottom);
  }

  @override
  void dispose() {
    widget.cactusService.messages.removeListener(_scrollToBottom);
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _sendMessage() {
    final text = _controller.text.trim();
    if (text.isEmpty) return;

    widget.cactusService.sendMessage(text);
    _controller.clear();
    _hasImage = false;
  }

  void _addImage() async {
    await widget.cactusService.addImage();
    setState(() => _hasImage = true);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cactus Demo'),
        actions: [
          ValueListenableBuilder<List<ChatMessage>>(
            valueListenable: widget.cactusService.messages,
            builder: (context, messages, _) {
              return IconButton(
                icon: const Icon(Icons.clear_all),
                onPressed:
                    messages.isEmpty
                        ? null
                        : () {
                          widget.cactusService.clearConversation();
                          setState(() => _hasImage = false);
                        },
              );
            },
          ),
        ],
      ),
      body: ValueListenableBuilder<String?>(
        valueListenable: widget.cactusService.error,
        builder: (context, error, _) {
          if (error != null) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Text(
                  'Error: $error',
                  style: const TextStyle(color: Colors.red),
                  textAlign: TextAlign.center,
                ),
              ),
            );
          }

          return ValueListenableBuilder<bool>(
            valueListenable: widget.cactusService.isLoading,
            builder: (context, isLoading, _) {
              return ValueListenableBuilder<List<ChatMessage>>(
                valueListenable: widget.cactusService.messages,
                builder: (context, messages, _) {
                  if (isLoading && messages.isEmpty) {
                    return ValueListenableBuilder<String>(
                      valueListenable: widget.cactusService.status,
                      builder: (context, status, _) {
                        return Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              const CircularProgressIndicator(),
                              const SizedBox(height: 16),
                              Text(status),
                            ],
                          ),
                        );
                      },
                    );
                  }

                  return Column(
                    children: [
                      Expanded(
                        child: ListView.builder(
                          controller: _scrollController,
                          padding: const EdgeInsets.all(8),
                          itemCount: messages.length,
                          itemBuilder: (context, index) {
                            return MessageBubble(message: messages[index]);
                          },
                        ),
                      ),
                      _buildInputArea(isLoading),
                    ],
                  );
                },
              );
            },
          );
        },
      ),
    );
  }

  Widget _buildInputArea(bool isLoading) {
    return Padding(
      padding: const EdgeInsets.all(8),
      child: Column(
        children: [
          if (_hasImage)
            Container(
              padding: const EdgeInsets.all(8),
              margin: const EdgeInsets.only(bottom: 8),
              decoration: BoxDecoration(
                color: Colors.blue.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                children: [
                  const Icon(Icons.image, color: Colors.blue),
                  const SizedBox(width: 8),
                  const Text('Image attached'),
                  const Spacer(),
                  IconButton(
                    icon: const Icon(Icons.clear),
                    onPressed: () => setState(() => _hasImage = false),
                  ),
                ],
              ),
            ),
          Row(
            children: [
              IconButton(
                icon: Icon(Icons.image, color: _hasImage ? Colors.blue : null),
                onPressed: isLoading ? null : _addImage,
              ),
              Expanded(
                child: TextField(
                  controller: _controller,
                  decoration: const InputDecoration(
                    hintText: 'Type a message...',
                    border: OutlineInputBorder(),
                  ),
                  onSubmitted: (_) => isLoading ? null : _sendMessage(),
                  enabled: !isLoading,
                ),
              ),
              IconButton(
                icon:
                    isLoading
                        ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                        : const Icon(Icons.send),
                onPressed: isLoading ? null : _sendMessage,
              ),
            ],
          ),
        ],
      ),
    );
  }
}
