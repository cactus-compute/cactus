import 'dart:async';

import './types.dart';
import './context.dart';
import './telemetry.dart';
import './remote.dart';
import './chat.dart';

class CactusVLM {
  CactusContext? _context;
  final ConversationHistoryManager _historyManager = ConversationHistoryManager();
  
  CactusVLM._();

  static Future<CactusVLM> init({
    required String modelUrl,
    required String mmprojUrl,
    String? modelFilename,
    String? mmprojFilename,
    String? chatTemplate,
    int contextSize = 2048,
    int gpuLayers = 0,
    int threads = 4,
    CactusProgressCallback? onProgress,
    String? cactusToken,
  }) async {
    final vlm = CactusVLM._();
    
    if (cactusToken != null) {
      setCactusToken(cactusToken);
    }
    
    final initParams = CactusInitParams(
      modelUrl: modelUrl,
      modelFilename: modelFilename,
      mmprojUrl: mmprojUrl,
      mmprojFilename: mmprojFilename,
      chatTemplate: chatTemplate,
      contextSize: contextSize,
      gpuLayers: gpuLayers,
      threads: threads,
      onInitProgress: onProgress,
    );
    
    try {
      vlm._context = await CactusContext.init(initParams);
    } catch (e) {
      CactusTelemetry.error(e, initParams);
      rethrow;
    }
    
    return vlm;
  }

  Future<CactusCompletionResult> completion(
    List<ChatMessage> messages, {
    List<String> imagePaths = const [],
    int maxTokens = 256,
    double? temperature,
    int? topK,
    double? topP,
    List<String>? stopSequences,
    CactusTokenCallback? onToken,
    String mode = "local",
  }) async {

    CactusCompletionResult? result;
    Exception? lastError;

    if (mode == "remote") {
      result = await _handleRemoteCompletion(messages, imagePaths, maxTokens, temperature, topK, topP, stopSequences, onToken);
    } else if (mode == "local") {
      result = await _handleLocalCompletion(messages, imagePaths, maxTokens, temperature, topK, topP, stopSequences, onToken);
    } else if (mode == "localfirst") {
      try {
        result = await _handleLocalCompletion(messages, imagePaths, maxTokens, temperature, topK, topP, stopSequences, onToken);
      } catch (e) {
        lastError = e is Exception ? e : Exception(e.toString());
        try {
          result = await _handleRemoteCompletion(messages, imagePaths, maxTokens, temperature, topK, topP, stopSequences, onToken);
        } catch (remoteError) {
          throw lastError;
        }
      }
    } else if (mode == "remotefirst") {
      try {
        result = await _handleRemoteCompletion(messages, imagePaths, maxTokens, temperature, topK, topP, stopSequences, onToken);
      } catch (e) {
        lastError = e is Exception ? e : Exception(e.toString());
        try {
          result = await _handleLocalCompletion(messages, imagePaths, maxTokens, temperature, topK, topP, stopSequences, onToken);
        } catch (localError) {
          throw lastError;
        }
      }
    } else {
      throw ArgumentError('Invalid mode: $mode. Must be "local", "remote", "localfirst", or "remotefirst"');
    }
    
    return result;
  }

  Future<CactusCompletionResult> _handleLocalCompletion(
    List<ChatMessage> messages,
    List<String> imagePaths,
    int maxTokens,
    double? temperature,
    int? topK,
    double? topP,
    List<String>? stopSequences,
    CactusTokenCallback? onToken,
  ) async {
    if (_context == null) throw CactusException('CactusVLM not initialized');

    final processed = _historyManager.processNewMessages(messages);
    if (processed.requiresReset) {
      _context!.rewind();
      _historyManager.reset();
    }
    
    final result = await _context!.completion(
      CactusCompletionParams(
        messages: processed.newMessages,
        maxPredictedTokens: maxTokens,
        temperature: temperature,
        topK: topK,
        topP: topP,
        stopSequences: stopSequences,
        onNewToken: onToken,
      ),
      mediaPaths: imagePaths,
    );

    _historyManager.update(processed.newMessages, ChatMessage(role: 'assistant', content: result.text));

    return result;
  }

  Future<CactusCompletionResult> _handleRemoteCompletion(
    List<ChatMessage> messages,
    List<String> imagePaths,
    int maxTokens,
    double? temperature,
    int? topK,
    double? topP,
    List<String>? stopSequences,
    CactusTokenCallback? onToken,
  ) async {
    final prompt = messages.map((m) => '${m.role}: ${m.content}').join('\n');
    final String imagePath = imagePaths.isNotEmpty ? imagePaths.first : '';
    
    String responseText;
    if (imagePath.isNotEmpty) {
      responseText = await getVisionCompletion(prompt, imagePath);
    } else {
      responseText = await getTextCompletion(prompt);
    }
    
    if (onToken != null) {
      for (int i = 0; i < responseText.length; i++) {
        if (!onToken(responseText[i])) break;
      }
    }
    
    return CactusCompletionResult(
      text: responseText,
      tokensPredicted: responseText.split(' ').length,
      tokensEvaluated: prompt.split(' ').length,
      truncated: false,
      stoppedEos: true,
      stoppedWord: false,
      stoppedLimit: false,
      stoppingWord: '',
    );
  }

  Future<bool> get supportsVision async {
    if (_context == null) return false;
    return await _context!.supportsVision();
  }

  Future<bool> get supportsAudio async {
    if (_context == null) return false;
    return await _context!.supportsAudio();
  }

  Future<bool> get isMultimodalEnabled async {
    if (_context == null) return false;
    return await _context!.isMultimodalEnabled();
  }

  Future<List<int>> tokenize(String text) async {
    if (_context == null) throw CactusException('CactusVLM not initialized');
    return await _context!.tokenize(text);
  }

  Future<String> detokenize(List<int> tokens) async {
    if (_context == null) throw CactusException('CactusVLM not initialized');
    return await _context!.detokenize(tokens);
  }

  Future<void> applyLoraAdapters(List<LoraAdapterInfo> adapters) async {
    if (_context == null) throw CactusException('CactusVLM not initialized');
    await _context!.applyLoraAdapters(adapters);
  }

  Future<void> removeLoraAdapters() async {
    if (_context == null) throw CactusException('CactusVLM not initialized');
    await _context!.removeLoraAdapters();
  }

  Future<List<LoraAdapterInfo>> getLoadedLoraAdapters() async {
    if (_context == null) throw CactusException('CactusVLM not initialized');
    return await _context!.getLoadedLoraAdapters();
  }

  Future<void> rewind() async {
    if (_context == null) throw CactusException('CactusVLM not initialized');
    _context!.rewind();
  }

  Future<void> stopCompletion() async {
    if (_context == null) throw CactusException('CactusVLM not initialized');
    await _context!.stopCompletion();
  }

  void dispose() {
    _context?.release();
    _context = null;
  }
} 