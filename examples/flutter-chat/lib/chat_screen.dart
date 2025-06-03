import 'package:flutter/material.dart';
import '../cactus_service.dart';
import 'package:cactus/cactus.dart'; // For ChatMessage, BenchResult
import 'widgets/message_bubble.dart';
import 'widgets/benchmark_view.dart';
import 'widgets/loading_indicator.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final CactusService _cactusService = CactusService();
  final TextEditingController _promptController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _cactusService.initialize();
    // Listen to ValueNotifiers to rebuild UI when they change
    _cactusService.chatMessages.addListener(_scrollToBottom);
    _cactusService.transcribedText.addListener(_onTranscribedTextChanged);
    _cactusService.sttError.addListener(_onSttError);
  }

  @override
  void dispose() {
    _cactusService.chatMessages.removeListener(_scrollToBottom);
    _cactusService.transcribedText.removeListener(_onTranscribedTextChanged);
    _cactusService.sttError.removeListener(_onSttError);
    _cactusService.dispose();
    _promptController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _sendMessage() {
    final userInput = _promptController.text.trim();
    if (userInput.isEmpty && _cactusService.imagePathForNextMessage.value == null) return;

    _cactusService.sendMessage(userInput);
    _promptController.clear();
    _scrollToBottom(); 
  }

  void _scrollToBottom() {
    // Schedule scroll after the frame has been built
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

  void _pickAndStageImage() {
    // Using a predefined asset for simplicity in this refactor
    // A real app would use image_picker or similar
    const String assetPath = 'assets/image.jpg'; 
    const String tempFilename = 'temp_chat_image.jpg';
    _cactusService.stageImageFromAsset(assetPath, tempFilename);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cactus Flutter Chat'),
        actions: [
          ValueListenableBuilder<bool>(
            valueListenable: _cactusService.isBenchmarking,
            builder: (context, isBenchmarking, child) {
              return IconButton(
                icon: const Icon(Icons.memory), // Benchmark icon
                onPressed: isBenchmarking || _cactusService.isLoading.value ? null : () => _cactusService.runBenchmark(),
                tooltip: 'Run Benchmark',
              );
            }
          ),
        ],
      ),
      body: ValueListenableBuilder<String?>(
        valueListenable: _cactusService.initError,
        builder: (context, initError, _) {
          if (initError != null) {
            return Padding(
              padding: const EdgeInsets.all(20.0),
              child: Center(
                child: Text(
                  initError,
                  style: const TextStyle(color: Colors.red, fontSize: 16),
                  textAlign: TextAlign.center,
                ),
              ),
            );
          }

          // Use multiple ValueListenableBuilders to react to specific state changes
          return ValueListenableBuilder<bool>(
            valueListenable: _cactusService.isLoading,
            builder: (context, isLoading, _) {
              return ValueListenableBuilder<bool>(
                valueListenable: _cactusService.isBenchmarking,
                builder: (context, isBenchmarking, _) {
                  return ValueListenableBuilder<List<ChatMessage>>(
                    valueListenable: _cactusService.chatMessages,
                    builder: (context, chatMessages, _ ) {
                       // Show initial loading only if no messages and not benchmarking and no error
                        bool showInitialLoading = isLoading && chatMessages.isEmpty && !isBenchmarking && initError == null;

                        return Column(
                        children: [
                          if (showInitialLoading || (isBenchmarking && chatMessages.isEmpty) ) // Show loading or benchmark progress if chat is empty
                            ValueListenableBuilder<double?>(
                              valueListenable: _cactusService.downloadProgress,
                              builder: (context, downloadProgress, _) {
                                return ValueListenableBuilder<String>(
                                  valueListenable: _cactusService.statusMessage,
                                  builder: (context, statusMessage, _) {
                                    return LoadingIndicator(
                                      isLoading: isLoading, 
                                      isBenchmarking: isBenchmarking,
                                      downloadProgress: downloadProgress,
                                      statusMessage: statusMessage,
                                    );
                                  }
                                );
                              }
                            ),
                          ValueListenableBuilder<BenchResult?>(
                            valueListenable: _cactusService.benchResult,
                            builder: (context, benchResult, _) {
                              return BenchmarkView(benchResult: benchResult);
                            }
                          ),
                          Expanded(
                            child: ListView.builder(
                              controller: _scrollController,
                              padding: const EdgeInsets.all(8.0),
                              itemCount: chatMessages.length,
                              itemBuilder: (context, index) {
                                final message = chatMessages[index];
                                return MessageBubble(message: message);
                              },
                            ),
                          ),
                          if (!showInitialLoading) // Hide input if initially loading
                            _buildChatInputArea(context, isLoading),
                        ],
                      );
                    }
                  );
                }
              );
            }
          );
        }
      ),
    );
  }

  Widget _buildChatInputArea(BuildContext context, bool currentIsLoading) {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Column(
        children: [
          ValueListenableBuilder<String?>(
            valueListenable: _cactusService.stagedAssetPath, // Listen to the staged asset path
            builder: (context, stagedAssetPathValue, _) {
              if (stagedAssetPathValue != null) {
                return Padding(
                  padding: const EdgeInsets.only(bottom: 8.0),
                  child: Row(
                    children: [
                      Image.asset( // Display from asset path for consistency
                        stagedAssetPathValue,
                        width: 50,
                        height: 50,
                        fit: BoxFit.cover,
                      ),
                      const SizedBox(width: 8),
                      const Text("Image staged"),
                      IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () => _cactusService.clearStagedImage(),
                      )
                    ],
                  ),
                );
              }
              return const SizedBox.shrink();
            }
          ),
          Row(
            children: [
              ValueListenableBuilder<String?>(
                valueListenable: _cactusService.stagedAssetPath,
                builder: (context, stagedAssetPathValue, _) {
                  return IconButton(
                    icon: Icon(Icons.image, color: stagedAssetPathValue != null ? Theme.of(context).primaryColor : null),
                    onPressed: currentIsLoading ? null : () {
                      if (stagedAssetPathValue == null) {
                        _pickAndStageImage();
                      } else {
                        _cactusService.clearStagedImage();
                      }
                    },
                  );
                }
              ),
              Expanded(
                child: TextField(
                  controller: _promptController,
                  decoration: const InputDecoration(
                    hintText: 'Type your message...',
                    border: OutlineInputBorder(),
                  ),
                  onSubmitted: (_) => currentIsLoading ? null : _sendMessage(),
                  minLines: 1,
                  maxLines: 3,
                  enabled: !currentIsLoading,
                ),
              ),
              ValueListenableBuilder<bool>(
                valueListenable: _cactusService.isLoading, // Specifically listen to overall isLoading for send button
                builder: (context, isLoadingForSendButton, _) {
                  return IconButton(
                    icon: isLoadingForSendButton && !(_cactusService.chatMessages.value.isEmpty && isLoadingForSendButton)
                        ? const SizedBox(width:24, height:24, child:CircularProgressIndicator(strokeWidth: 2,))
                        : const Icon(Icons.send),
                    onPressed: isLoadingForSendButton ? null : _sendMessage,
                  );
                }
              ),
              // Microphone Button
              ValueListenableBuilder<bool>(
                valueListenable: _cactusService.isRecording,
                builder: (context, isRecording, child) {
                  return IconButton(
                    icon: Icon(isRecording ? Icons.mic_off : Icons.mic, color: isRecording ? Colors.red : null),
                    onPressed: currentIsLoading ? null : _toggleRecording,
                    tooltip: isRecording ? 'Stop Recording' : 'Start Recording',
                  );
                }
              ),
            ],
          ),
          // Display STT Error if any
          ValueListenableBuilder<String?>(
            valueListenable: _cactusService.sttError,
            builder: (context, error, child) {
              if (error != null && error.isNotEmpty) {
                return Padding(
                  padding: const EdgeInsets.only(top: 8.0),
                  child: Text(error, style: const TextStyle(color: Colors.red)),
                );
              }
              return const SizedBox.shrink();
            }
          ),
        ],
      ),
    );
  }

  void _onTranscribedTextChanged() {
    final newText = _cactusService.transcribedText.value;
    if (newText != null && newText.isNotEmpty) {
      _promptController.text = newText;
      // Optionally send message directly after transcription:
      // _sendMessage();
      // _cactusService.transcribedText.value = null; // Clear after use
    }
  }

  void _onSttError() {
    final error = _cactusService.sttError.value;
    if (error != null && error.isNotEmpty) {
      // Optionally show a dialog or a more prominent error message
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(error, style: const TextStyle(color: Colors.white)), backgroundColor: Colors.red)
      );
    }
  }

  Future<void> _toggleRecording() async {
    // Optionally call setUserVocabulary before starting
    // await _cactusService.setSttUserVocabulary(["custom word", "Cactus AI"]);

    if (!_cactusService.isRecording.value) {
      bool granted = await _cactusService.requestMicrophonePermissions();
      if (granted) {
        _cactusService.startVoiceCapture();
      } else {
        _cactusService.sttError.value = "Microphone permission denied.";
      }
    } else {
      _cactusService.stopVoiceCapture();
    }
  }
} 