import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart'; // For ValueNotifier
import 'package:flutter/services.dart'; // For rootBundle
import 'package:path_provider/path_provider.dart';
import 'package:cactus/cactus.dart';

class CactusService {
  CactusContext? _cactusContext;

  // ValueNotifiers for UI updates
  final ValueNotifier<List<ChatMessage>> chatMessages = ValueNotifier([]);
  final ValueNotifier<bool> isLoading = ValueNotifier(true);
  final ValueNotifier<bool> isBenchmarking = ValueNotifier(false);
  final ValueNotifier<String> statusMessage = ValueNotifier('Initializing...');
  final ValueNotifier<String?> initError = ValueNotifier(null);
  final ValueNotifier<double?> downloadProgress = ValueNotifier(null);
  final ValueNotifier<BenchResult?> benchResult = ValueNotifier(null);
  final ValueNotifier<String?> imagePathForNextMessage = ValueNotifier(null);
  final ValueNotifier<String?> stagedAssetPath = ValueNotifier(null); // For image picker display

  // Model and MMProj URLs and filenames (consider making these configurable)
  static const String _modelUrl = 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf';
  static const String _modelFilename = 'SmolVLM-256M-Instruct-Q8_0.gguf';
  static const String _mmprojUrl = 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf';
  static const String _mmprojFilename = "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf";

  Future<void> initialize() async {
    isLoading.value = true;
    initError.value = null;
    statusMessage.value = 'Initializing plugin...';
    downloadProgress.value = null;

    try {
      final Directory appDocDir = await getApplicationDocumentsDirectory();
      final String effectiveModelPath = '${appDocDir.path}/$_modelFilename';
      final String effectiveMmprojPath = '${appDocDir.path}/$_mmprojFilename';

      // Download main model if needed
      await _downloadFileIfNeeded(_modelUrl, effectiveModelPath, "Main Model");
      // Download multimodal projector if needed
      await _downloadFileIfNeeded(_mmprojUrl, effectiveMmprojPath, "MM Projector");

      statusMessage.value = 'Initializing native context...';
      downloadProgress.value = null;

      final params = CactusInitParams(
        modelPath: effectiveModelPath,
        mmprojPath: effectiveMmprojPath,
        onInitProgress: (progress, status, isError) {
          statusMessage.value = "Native Init: $status";
          if (isError) {
            initError.value = status;
            isLoading.value = false;
          }
        },
      );

      _cactusContext = await CactusContext.init(params);
      isLoading.value = false;
      statusMessage.value = 'Cactus initialized successfully!';

      await runBenchmark(); // Run benchmark after successful initialization
    } on CactusModelPathException catch (e) {
      initError.value = "Model Error: ${e.message}";
      statusMessage.value = 'Failed to load model.';
      isLoading.value = false;
      debugPrint("Cactus Model Path Exception: ${e.toString()}");
    } on CactusInitializationException catch (e) {
      initError.value = "Initialization Error: ${e.message}";
      statusMessage.value = 'Failed to initialize native context.';
      isLoading.value = false;
      debugPrint("Cactus Initialization Exception: ${e.toString()}");
    } catch (e) {
      if (initError.value == null) {
        initError.value = "An unexpected error occurred during initialization: ${e.toString()}";
      }
      statusMessage.value = 'Initialization failed.';
      isLoading.value = false;
      debugPrint("Generic Exception during Cactus Init: ${e.toString()}");
    }
  }

  Future<void> _downloadFileIfNeeded(String url, String filePath, String fileDescription) async {
    final file = File(filePath);
    if (!await file.exists()) {
      statusMessage.value = 'Downloading $fileDescription...';
      downloadProgress.value = 0.0;
      await downloadModel(
        url,
        filePath,
        onProgress: (progress, status) {
          downloadProgress.value = progress;
          statusMessage.value = "$fileDescription: $status";
        },
      );
    } else {
      statusMessage.value = '$fileDescription found locally.';
    }
  }

  Future<void> sendMessage(String userInput) async {
    if (_cactusContext == null) {
      chatMessages.value = [...chatMessages.value, ChatMessage(role: 'system', content: 'Error: CactusContext not initialized.')];
      return;
    }

    String currentAssistantResponse = "";
    final userMessageContent = imagePathForNextMessage.value != null
        ? "<__image__>\n$userInput"
        : userInput;

    final userMessage = ChatMessage(
      role: 'user',
      content: userMessageContent,
    );

    final List<ChatMessage> updatedMessages = List.from(chatMessages.value);
    updatedMessages.add(userMessage);
    updatedMessages.add(ChatMessage(role: 'assistant', content: currentAssistantResponse));
    chatMessages.value = updatedMessages;
    isLoading.value = true; // Use general loading for message processing

    final String? imagePathToSend = imagePathForNextMessage.value;
    imagePathForNextMessage.value = null; // Clear after sending
    stagedAssetPath.value = null;

    try {
      List<ChatMessage> currentChatHistoryForCompletion = List.from(chatMessages.value);
      if (currentChatHistoryForCompletion.isNotEmpty &&
          currentChatHistoryForCompletion.last.role == 'assistant' &&
          currentChatHistoryForCompletion.last.content.isEmpty) {
        currentChatHistoryForCompletion.removeLast();
      }

      final completionParams = CactusCompletionParams(
        messages: currentChatHistoryForCompletion,
        imagePath: imagePathToSend,
        stopSequences: ['<|im_end|>'],
        temperature: 0.7,
        topK: 10,
        topP: 0.9,
        onNewToken: (String token) {
          if (!isLoading.value) return false; // Check isLoading, not mounted (service has no mount status)

          if (token == '<|im_end|>') return false;

          if (token.isNotEmpty) {
            currentAssistantResponse += token;
            final List<ChatMessage> streamingMessages = List.from(chatMessages.value);
            if (streamingMessages.isNotEmpty && streamingMessages.last.role == 'assistant') {
              streamingMessages[streamingMessages.length - 1] = ChatMessage(
                role: 'assistant',
                content: currentAssistantResponse,
              );
              chatMessages.value = streamingMessages;
            }
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

      final List<ChatMessage> finalMessages = List.from(chatMessages.value);
      if (finalMessages.isNotEmpty && finalMessages.last.role == 'assistant') {
        finalMessages[finalMessages.length - 1] = ChatMessage(
          role: 'assistant',
          content: finalCleanText.isNotEmpty ? finalCleanText : "(No further response)",
        );
        chatMessages.value = finalMessages;
      }
    } on CactusCompletionException catch (e) {
      _addErrorMessageToChat("Completion Error: ${e.message}");
      debugPrint("Cactus Completion Exception: ${e.toString()}");
    } catch (e) {
      _addErrorMessageToChat("An unexpected error occurred during completion: ${e.toString()}");
      debugPrint("Generic Exception during completion: ${e.toString()}");
    } finally {
      isLoading.value = false;
    }
  }

  void _addErrorMessageToChat(String errorMessage) {
      final List<ChatMessage> errorMessages = List.from(chatMessages.value);
      if (errorMessages.isNotEmpty && errorMessages.last.role == 'assistant') {
        errorMessages[errorMessages.length - 1] = ChatMessage(
          role: 'assistant',
          content: errorMessage,
        );
      } else {
        errorMessages.add(ChatMessage(role: 'system', content: errorMessage));
      }
      chatMessages.value = errorMessages;
  }

  Future<void> runBenchmark() async {
    if (_cactusContext == null) return;
    isBenchmarking.value = true;
    statusMessage.value = 'Running benchmark...';
    try {
      final result = _cactusContext!.bench();
      benchResult.value = result;
      statusMessage.value = 'Benchmark complete.';
    } catch (e) {
      debugPrint("Benchmark Error: ${e.toString()}");
      statusMessage.value = 'Benchmark failed: ${e.toString()}';
    } finally {
      isBenchmarking.value = false;
    }
  }

  void stageImageFromAsset(String assetPath, String tempFilename) async {
      try {
        final ByteData assetData = await rootBundle.load(assetPath); // Requires services.dart
        final Directory tempDir = await getTemporaryDirectory();
        final String tempFilePath = '${tempDir.path}/$tempFilename'; 
        final File tempFile = File(tempFilePath);
        await tempFile.writeAsBytes(assetData.buffer.asUint8List(), flush: true);
        imagePathForNextMessage.value = tempFilePath;
        stagedAssetPath.value = assetPath;
      } catch (e) {
        debugPrint("Error staging image from asset: $e");
        statusMessage.value = "Error staging image: $e";
      }
  }

  void clearStagedImage() {
    imagePathForNextMessage.value = null;
    stagedAssetPath.value = null;
  }

  void dispose() {
    _cactusContext?.free();
    chatMessages.dispose();
    isLoading.dispose();
    isBenchmarking.dispose();
    statusMessage.dispose();
    initError.dispose();
    downloadProgress.dispose();
    benchResult.dispose();
    imagePathForNextMessage.dispose();
    stagedAssetPath.dispose();
  }
} 