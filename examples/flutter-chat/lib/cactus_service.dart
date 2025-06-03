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

  // STT Related ValueNotifiers
  /// Notifies listeners with the latest transcribed text from STT.
  /// Null if no transcription is available or if an error occurred.
  final ValueNotifier<String?> transcribedText = ValueNotifier(null);
  /// Notifies listeners about the current recording state (true if recording, false otherwise).
  final ValueNotifier<bool> isRecording = ValueNotifier(false);
  /// Notifies listeners of any errors that occur during STT operations.
  /// Null if no error has occurred.
  final ValueNotifier<String?> sttError = ValueNotifier(null);


 Future<void> initialize() async {
    isLoading.value = true;
    initError.value = null;
    statusMessage.value = 'Initializing plugin...';
    downloadProgress.value = null; 

    // Cactus usage 
    try {
      final params = CactusInitParams(
        modelUrl: 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf', 
        mmprojUrl: 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf',

        onInitProgress: (progress, status, isError) {
          statusMessage.value = status; 
          downloadProgress.value = progress; 

          if (isError) {
            initError.value = status;
            isLoading.value = false;
          }
        },
      );

      _cactusContext = await CactusContext.init(params);
      
      if (initError.value == null) { 
        isLoading.value = false;
        statusMessage.value = 'Cactus initialized successfully!';
        await runBenchmark();
      } else {
        statusMessage.value = 'Initialization failed: ${initError.value}';
      }

    } on CactusModelPathException catch (e) {
      initError.value = "Model Error: ${e.message}";
      statusMessage.value = 'Failed to load model: ${e.message}';
      isLoading.value = false;
      debugPrint("Cactus Model Path Exception: ${e.toString()}");
    } on CactusInitializationException catch (e) {
      initError.value = "Initialization Error: ${e.message}";
      statusMessage.value = 'Failed to initialize context: ${e.message}';
      isLoading.value = false;
      debugPrint("Cactus Initialization Exception: ${e.toString()}");
    } catch (e) {
      initError.value ??= "An unexpected error occurred: ${e.toString()}";
      statusMessage.value = 'Initialization failed: ${initError.value}';
      isLoading.value = false;
      debugPrint("Generic Exception during Cactus Init: ${e.toString()}");
    }
  }

  Future<void> sendMessage(String userInput) async {
    if (_cactusContext == null) {
      chatMessages.value = [...chatMessages.value, ChatMessage(role: 'system', content: 'Error: CactusContext not initialized.')];
      return;
    }

    String currentAssistantResponse = "";
    final userMessageContent = userInput; 

    final userMessage = ChatMessage(
      role: 'user',
      content: userMessageContent,
    );

    final List<ChatMessage> updatedMessages = List.from(chatMessages.value);
    updatedMessages.add(userMessage);
    updatedMessages.add(ChatMessage(role: 'assistant', content: currentAssistantResponse));
    chatMessages.value = updatedMessages;
    isLoading.value = true;

    final String? imagePathToSend = imagePathForNextMessage.value;
    imagePathForNextMessage.value = null;
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
        stopSequences: ['<|im_end|>', '<end_of_utterance>'],
        temperature: 0.7,
        topK: 10,
        topP: 0.9,
        onNewToken: (String token) {
          if (!isLoading.value) return false;

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
      String finalCleanText = result.text.trim(); 
      
      if (finalCleanText.isEmpty && currentAssistantResponse.trim().isNotEmpty) {
        finalCleanText = currentAssistantResponse.trim();
      }

      final List<ChatMessage> finalMessages = List.from(chatMessages.value);
      if (finalMessages.isNotEmpty && finalMessages.last.role == 'assistant') {
        finalMessages[finalMessages.length - 1] = ChatMessage(
          role: 'assistant',
          content: finalCleanText.isNotEmpty ? finalCleanText : "(No further response)",
          tokensPerSecond: result.tokensPerSecond,
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
    // Dispose STT ValueNotifiers
    transcribedText.dispose();
    isRecording.dispose();
    sttError.dispose();
  }

  // --- STT Methods ---

  /// Placeholder method to request microphone permissions.
  ///
  /// In a real application, this should use a permission handling plugin
  /// like `permission_handler` to manage platform-specific permission requests.
  /// This current implementation is a placeholder and assumes permission is granted.
  ///
  /// Returns `true` as a placeholder, indicating permission is assumed to be granted.
  Future<bool> requestMicrophonePermissions() async {
    // Directly call a permission handler.
    // For simplicity, assuming cactus_flutter might expose a permission handler or
    // one would use a package like `permission_handler`.
    // This is a conceptual placeholder for where permission request logic would go.
    // In a real app, integrate with `permission_handler` package for robust permissions.
    // For now, let's assume the plugin or OS handles it implicitly on first use,
    // or this method would use a specific permission plugin.
    // Returning true for now as a placeholder.
    debugPrint("[CactusService] Requesting microphone permissions (Placeholder - assuming granted or handled by OS/plugin).");
    // Example using a hypothetical permission method on _cactusContext if it existed:
    // return await _cactusContext?.requestMicrophonePermission() ?? false;
    return true; // Placeholder
  }

  /// Starts voice capture for STT.
  ///
  /// This is a placeholder implementation that simulates starting a recording.
  /// It sets [isRecording] to true and updates [statusMessage].
  /// Actual STT functionality (model initialization, audio capture, processing)
  /// would be handled by methods on `_cactusContext` if available.
  ///
  /// If `_cactusContext` is null or if already recording, an error is set on [sttError].
  Future<void> startVoiceCapture() async {
    if (_cactusContext == null) {
      sttError.value = 'STT Error: CactusContext not initialized.';
      return;
    }
    if (isRecording.value) {
      sttError.value = 'STT Error: Already recording.';
      return;
    }

    sttError.value = null;
    transcribedText.value = null; // Clear previous transcription

    // Optional: Call setUserVocabulary if needed
    // await setSttUserVocabulary(["example", "custom word"]);

    // Assuming CactusContext has startVoiceCapture which handles STT model init, audio recording,
    // and provides transcription via a callback or stream that updates transcribedText.
    // For this example, we'll simulate it.
    // In a real scenario, _cactusContext.startVoiceCapture might take a handler.
    // e.g. await _cactusContext.startVoiceCapture(
    //        modelPath: "path/to/stt_model.gguf", // This might be part of CactusContext.init too
    //        onTranscription: (text) { transcribedText.value = text; },
    //        onError: (error) { sttError.value = error.toString(); isRecording.value = false; }
    //      );

    // Placeholder implementation:
    isRecording.value = true;
    statusMessage.value = "Voice recording started...";
    debugPrint("[CactusService] Voice capture started (Simulated).");

    // Simulate receiving transcription after some time
    // In a real app, this would come from the STT engine.
    // For now, let's assume stopVoiceCapture will "finalize" a dummy transcription.
  }

  /// Stops the current voice capture.
  ///
  /// This is a placeholder implementation that simulates stopping a recording
  /// and producing a dummy transcription. It sets [isRecording] to false,
  /// updates [statusMessage], and sets a placeholder value for [transcribedText].
  ///
  /// If not currently recording, this method does nothing.
  Future<void> stopVoiceCapture() async {
    if (!isRecording.value) {
      // sttError.value = 'STT Error: Not recording.'; // Or just ignore
      return;
    }
    // In a real scenario, this would signal the native layer to stop recording
    // and finalize transcription.
    // e.g. await _cactusContext.stopVoiceCapture();

    // Placeholder implementation:
    isRecording.value = false;
    statusMessage.value = "Voice recording stopped. Processing...";
    debugPrint("[CactusService] Voice capture stopped (Simulated).");

    // Simulate a delay for "processing" and then set a dummy transcription
    await Future.delayed(const Duration(seconds: 1));
    if (sttError.value == null) { // Only set transcription if no error occurred during "recording"
        transcribedText.value = "Hello world, this is a dummy transcription."; // Placeholder
        statusMessage.value = "Transcription received.";
    } else {
        statusMessage.value = "Transcription failed: ${sttError.value}";
    }
  }

  /// Sets user-specific vocabulary for STT (Placeholder).
  ///
  /// This method calls the placeholder `setUserVocabulary` on the `_cactusContext`.
  /// The underlying feature is not yet implemented in the core C++ library.
  ///
  /// - Parameter vocabulary: A list of words or phrases to suggest for STT biasing.
  Future<void> setSttUserVocabulary(List<String> vocabulary) async {
    if (_cactusContext == null) {
      sttError.value = 'STT Error: CactusContext not initialized for setUserVocabulary.';
      return;
    }
    try {
      await _cactusContext!.setUserVocabulary(vocabulary); // Calling the placeholder
      debugPrint("[CactusService] setUserVocabulary called (placeholder).");
    } catch (e) {
      sttError.value = "Error calling setUserVocabulary: ${e.toString()}";
      debugPrint("[CactusService] Error in setUserVocabulary: ${e.toString()}");
    }
  }
  // --- End STT Methods ---
} 