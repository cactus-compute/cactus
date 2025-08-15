import 'dart:convert';
import 'dart:io';
import 'package:cactus/chat.dart';
import 'package:cactus/remote.dart';
import 'package:cactus/telemetry.dart';
import 'package:path_provider/path_provider.dart';

import './context.dart';
import './tools.dart';
import './types.dart';

class CompletionResult {
  final String? result;
  final List<String>? toolCalls;

  CompletionResult({
    this.result,
    this.toolCalls,
  });
}

class CactusAgent {
  final int threads;
  final int contextSize;
  final int batchSize;
  final int gpuLayers;

  CactusContext? _context;
  final Tools _tools = Tools();
  String? _chatTemplate;
  String? _lastDownloadedFilename;
  final ConversationHistoryManager _historyManager = ConversationHistoryManager();


  CactusAgent({
    this.threads = 4,
    this.contextSize = 2048,
    this.batchSize = 512,
    this.gpuLayers = 0,
  });

  Future<bool> download({
    required String modelUrl,
    String? modelFilename,
    CactusProgressCallback? onProgress,
  }) async {
    try {
      final actualFilename = modelFilename ?? modelUrl.split('/').last;
      if (actualFilename.isEmpty) {
        throw ArgumentError('Cannot determine filename from URL and no filename provided');
      }

      final success = await CactusContext.downloadModels(
        modelUrl: modelUrl,
        modelFilename: modelFilename,
        onProgress: onProgress,
      );

      if (success) {
        _lastDownloadedFilename = actualFilename;
      }
      return success;
    } catch (e) {
      if (onProgress != null) {
        onProgress(null, "Download failed: ${e.toString()}", true);
      }
      return false;
    }
  }

  Future<bool> init({
    String? modelFilename,
    String? chatTemplate,
    int contextSize = 2048,
    int gpuLayers = 0,
    int threads = 4,
    bool generateEmbeddings = false,
    CactusProgressCallback? onProgress,
    String? cactusToken,
  }) async {
    if (cactusToken != null) {
      setCactusToken(cactusToken);
    }

    final filename = modelFilename ?? _lastDownloadedFilename;
    if (filename == null) {
      throw ArgumentError('No model filename provided and no model was previously downloaded');
    }

    final appDocDir = await getApplicationDocumentsDirectory();
    final modelPath = '${appDocDir.path}/$filename';
    
    final file = File(modelPath);
    if (!await file.exists()) {
      throw ArgumentError('Model file does not exist at path: $modelPath');
    }

    final initParams = CactusInitParams(
      modelPath: modelPath,
      chatTemplate: chatTemplate,
      contextSize: contextSize,
      gpuLayers: gpuLayers,
      threads: threads,
      generateEmbeddings: generateEmbeddings,
      onInitProgress: onProgress,
    );
    
    try {
      if (onProgress != null) {
        onProgress(null, "Initializing...", false);
      }
      _context = await CactusContext.init(initParams);
      return true;
    } catch (e) {
      CactusTelemetry.error(e, initParams);
      if (onProgress != null) {
        onProgress(null, "Initialization failed: ${e.toString()}", true);
      }
      return false;
    }
  }

  Future<CompletionResult> completionWithTools(
    List<ChatMessage> messages, {
    int maxTokens = 256,
    double? temperature,
    int? topK,
    double? topP,
    List<String>? stopSequences,
    CactusTokenCallback? onToken,
  }) async {
    if (_context == null) throw CactusException('CactusLM not initialized');
    
    if (_tools.isEmpty()) {
      final response = await completion(
        messages,
        maxTokens: maxTokens,
        temperature: temperature,
        topP: topP,
      );
      return CompletionResult(
        result: response.text,
        toolCalls: null,
      );
    }

    if (!isLoaded()) {
      return CompletionResult(
        result: "Model is not loaded",
        toolCalls: null,
      );
    }

    final toolSchemas = _tools.getSchemas();
    final toolsJson = jsonEncode(toolSchemas.map((e) => e.toJson()).toList());

    final userMessage = ChatMessage(role: 'user', content: messages.last.content);
    final formattedResult = await _context!.formatChatWithTools([userMessage], toolsJson);
    
    final params = CactusCompletionParams(
      messages: [ChatMessage(role: 'user', content: formattedResult.prompt)],
      maxPredictedTokens: maxTokens,
      temperature: temperature,
      topP: topP,
      chatTemplate: _chatTemplate,
      grammar: formattedResult.grammar,
    );

    final modelResponse = await _context!.completion(params);

    final toolResult = await parseAndExecuteTool(modelResponse.text, _tools);

    return CompletionResult(
      result: toolResult.toolCalled ? toolResult.toolOutput : modelResponse.text,
      toolCalls: toolResult.toolCalled ? [toolResult.toolName!] : null,
    );
  }

  Future<CactusCompletionResult> completion(
    List<ChatMessage> messages, {
    int maxTokens = 256,
    double? temperature,
    int? topK,
    double? topP,
    List<String>? stopSequences,
    CactusTokenCallback? onToken,
  }) async {
    if (_context == null) throw CactusException('CactusLM not initialized');

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
    );
    
    _historyManager.update(processed.newMessages, ChatMessage(role: 'assistant', content: result.text));

    return result;
  }

  bool isLoaded() => _context != null;

  void addTool(
    String name,
    ToolExecutor function,
    String description,
    Map<String, Parameter> parameters,
  ) {
    _tools.add(name, function, description, parameters);
  }

  void dispose() {
    _context?.release();
    _context = null;
  }
}
