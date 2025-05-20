import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:convert'; 
import 'dart:io'; 
import 'package:path_provider/path_provider.dart'; 

import 'package:cactus/cactus.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  CactusContext? _cactusContext;
  final TextEditingController _promptController = TextEditingController();
  List<ChatMessage> _chatMessages = []; 
  bool _isLoading = true;
  String _statusMessage = 'Initializing...'; 
  String _initError = '';
  double? _downloadProgress; 

  final ScrollController _scrollController = ScrollController(); 

  @override
  void initState() {
    super.initState();
    _initializeCactus();
  }

  Future<void> _initializeCactus() async {
    setState(() {
      _isLoading = true;
      _initError = '';
      _statusMessage = 'Initializing plugin...';
      _downloadProgress = null; 
    });

    try {
      final params = CactusInitParams(
        modelPath: 'assets/smollm2-360m-instruct-q8_0.gguf',
        onInitProgress: (progress, status, isError) {
          setState(() {
            _downloadProgress = progress;
            _statusMessage = status;
            if (isError) {
              _initError = status;
              _isLoading = false;
            }
          });
        },
      );

      _cactusContext = await CactusContext.init(params);
      
      setState(() {
        _isLoading = false;
        _statusMessage = 'Cactus initialized successfully!';
      });

    } on CactusModelPathException catch (e) {
      if (mounted) {
        setState(() {
          _initError = "Model Error: ${e.message}";
          _statusMessage = 'Failed to load model.';
          _isLoading = false;
        });
      }
      print("Cactus Model Path Exception: ${e.toString()}");
    } on CactusInitializationException catch (e) {
      if (mounted) {
        setState(() {
          _initError = "Initialization Error: ${e.message}";
          _statusMessage = 'Failed to initialize native context.';
          _isLoading = false;
        });
      }
      print("Cactus Initialization Exception: ${e.toString()}");
    } catch (e) {
      if (mounted) {
        setState(() {
          if (_initError.isEmpty) {
             _initError = "An unexpected error occurred during initialization: ${e.toString()}";
          }
          _statusMessage = 'Initialization failed.';
          _isLoading = false;
        });
      }
      print("Generic Exception during Cactus Init: ${e.toString()}");
    } 
  }

  @override
  void dispose() {
    _cactusContext?.free();
    _promptController.dispose();
    _scrollController.dispose(); 
    super.dispose();
  }

  Future<void> _sendMessage() async {
    final userInput = _promptController.text.trim();
    if (userInput.isEmpty) return;

    if (_cactusContext == null) {
      setState(() {
        _chatMessages.add(ChatMessage(role: 'system', content: 'Error: CactusContext not initialized. Please restart the app.'));
      });
      return;
    }

    String currentAssistantResponse = "";

    setState(() {
      _chatMessages.add(ChatMessage(role: 'user', content: userInput));
      _chatMessages.add(ChatMessage(role: 'assistant', content: currentAssistantResponse)); 
      _isLoading = true;
    });
    _promptController.clear();
    _scrollToBottom();

    try {
      List<ChatMessage> currentChatHistoryForCompletion = List.from(_chatMessages);
      if (currentChatHistoryForCompletion.isNotEmpty && 
          currentChatHistoryForCompletion.last.role == 'assistant' && 
          currentChatHistoryForCompletion.last.content.isEmpty) {
        currentChatHistoryForCompletion.removeLast();
      }
      
      final completionParams = CactusCompletionParams(
        messages: currentChatHistoryForCompletion, 
        stopSequences: ['<|im_end|>'], 
        temperature: 0.7,
        topK: 10,
        topP: 0.9,
        onNewToken: (String token) {
          if (!mounted || !_isLoading) {
            return false; 
          }

          if (token == '<|im_end|>') {
            return false;
          }
          
          if (token.isNotEmpty) {
            currentAssistantResponse += token;
            setState(() {
              if (_chatMessages.isNotEmpty && _chatMessages.last.role == 'assistant') {
                _chatMessages[_chatMessages.length - 1] = ChatMessage(
                  role: 'assistant', 
                  content: currentAssistantResponse,
                );
              }
            });
            _scrollToBottom(); 
          }
          return true;
        },
      );

      final result = await _cactusContext!.completion(completionParams);

      String finalCleanText = result.text;
      if (finalCleanText.trim().isEmpty && currentAssistantResponse.trim().isNotEmpty) {
        finalCleanText = currentAssistantResponse.trim();
      } else {
        if (finalCleanText.endsWith('<|im_end|>')) {
          finalCleanText = finalCleanText.substring(0, finalCleanText.length - '<|im_end|>'.length).trim();
        } else {
          finalCleanText = finalCleanText.trim();
        }
      }
      
      setState(() {
        if (_chatMessages.isNotEmpty && _chatMessages.last.role == 'assistant') {
           _chatMessages[_chatMessages.length - 1] = ChatMessage(
            role: 'assistant',
            content: finalCleanText.isNotEmpty ? finalCleanText : "(No further response)",
          );
        }
      });

    } on CactusCompletionException catch (e) {
      setState(() {
        String errorMessage = "Completion Error: ${e.message}";
        if (_chatMessages.isNotEmpty && _chatMessages.last.role == 'assistant') {
           _chatMessages[_chatMessages.length - 1] = ChatMessage(
             role: 'assistant',
             content: errorMessage,
           );
        } else {
           _chatMessages.add(ChatMessage(role: 'system', content: errorMessage));
        }
      });
      print("Cactus Completion Exception: ${e.toString()}");
    } catch (e) {
      setState(() {
        String errorMessage = "An unexpected error occurred during completion: ${e.toString()}";
        if (_chatMessages.isNotEmpty && _chatMessages.last.role == 'assistant') {
           _chatMessages[_chatMessages.length - 1] = ChatMessage(
             role: 'assistant',
             content: errorMessage,
           );
        } else {
           _chatMessages.add(ChatMessage(role: 'system', content: errorMessage));
        }
      });
      print("Generic Exception during completion: ${e.toString()}");
    } finally {
      setState(() {
        _isLoading = false;
      });
      _scrollToBottom();
    }
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

  @override
  Widget build(BuildContext context) {
    const spacerSmall = SizedBox(height: 10);

    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Cactus Flutter Chat'), 
        ),
        body: Column( 
          children: [
            if (_isLoading && _chatMessages.isEmpty && _initError.isEmpty) 
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      if (_downloadProgress != null && _downloadProgress! < 1.0) 
                        LinearProgressIndicator(
                          value: _downloadProgress,
                          minHeight: 10,
                        )
                      else if (_downloadProgress == null || _downloadProgress! >= 1.0)
                        const CircularProgressIndicator(),
                      const SizedBox(height: 20),
                      Text(
                        _statusMessage,
                        textAlign: TextAlign.center,
                        style: TextStyle(fontSize: 16, color: Colors.grey[700]),
                      ),
                    ],
                  ),
                ),
              )
            else if (_initError.isNotEmpty)
              Padding(
                padding: const EdgeInsets.all(20.0),
                child: Center(
                  child: Text(
                    _initError,
                    style: const TextStyle(color: Colors.red, fontSize: 16),
                    textAlign: TextAlign.center,
                  ),
                )
              ),
            
            Expanded(
              child: ListView.builder(
                controller: _scrollController,
                padding: const EdgeInsets.all(8.0),
                itemCount: _chatMessages.length,
                itemBuilder: (context, index) {
                  final message = _chatMessages[index];
                  bool isUser = message.role == 'user';
                  bool isSystem = message.role == 'system';
                  return Align(
                    alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                    child: Container(
                      margin: const EdgeInsets.symmetric(vertical: 4.0),
                      padding: const EdgeInsets.symmetric(horizontal: 12.0, vertical: 8.0),
                      decoration: BoxDecoration(
                        color: isSystem ? Colors.red[100] : (isUser ? Colors.blue[100] : Colors.green[100]),
                        borderRadius: BorderRadius.circular(12.0),
                      ),
                      child: Text(message.content, style: TextStyle(color: isSystem ? Colors.red[900] : Colors.black)),
                    ),
                  );
                },
              ),
            ),

            if (_cactusContext != null && _initError.isEmpty)
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _promptController,
                        decoration: const InputDecoration(
                          hintText: 'Type your message...',
                          border: OutlineInputBorder(),
                        ),
                        onSubmitted: (_) => _isLoading ? null : _sendMessage(),
                        minLines: 1,
                        maxLines: 3,
                        enabled: !_isLoading,
                      ),
                    ),
                    IconButton(
                      icon: _isLoading && !(_chatMessages.isEmpty && _isLoading)
                          ? const SizedBox(width:24, height:24, child:CircularProgressIndicator(strokeWidth: 2,))
                          : const Icon(Icons.send),
                      onPressed: _isLoading ? null : _sendMessage, 
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}
