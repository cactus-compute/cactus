import 'dart:async';

import './types.dart';
import './context.dart';

class CactusTTS {
  CactusContext? _context;
  
  CactusTTS._();

  static Future<CactusTTS> init({
    required String modelUrl,
    String? modelFilename,
    int contextSize = 2048,
    int gpuLayers = 0,
    int threads = 4,
    CactusProgressCallback? onProgress,
  }) async {
    final tts = CactusTTS._();
    
    tts._context = await CactusContext.init(CactusInitParams(
      modelFilename: modelFilename,
      contextSize: contextSize,
      gpuLayers: gpuLayers,
      threads: threads,
      onInitProgress: onProgress,
    ));
    
    return tts;
  }

  Future<CactusCompletionResult> generate(
    String text, {
    int maxTokens = 256,
    double? temperature,
    int? topK,
    double? topP,
    List<String>? stopSequences,
    CactusTokenCallback? onToken,
  }) async {
    if (_context == null) throw CactusException('CactusTTS not initialized');
    
    final messages = [
      ChatMessage(role: 'user', content: text),
    ];
    
    return await _context!.completion(
      CactusCompletionParams(
        messages: messages,
        maxPredictedTokens: maxTokens,
        temperature: temperature,
        topK: topK,
        topP: topP,
        stopSequences: stopSequences,
        onNewToken: onToken,
      ),
    );
  }

  Future<bool> get supportsAudio async {
    if (_context == null) return false;
    return await _context!.supportsAudio();
  }

  Future<List<int>> tokenize(String text) async {
    if (_context == null) throw CactusException('CactusTTS not initialized');
    return await _context!.tokenize(text);
  }

  Future<String> detokenize(List<int> tokens) async {
    if (_context == null) throw CactusException('CactusTTS not initialized');
    return await _context!.detokenize(tokens);
  }

  Future<void> applyLoraAdapters(List<LoraAdapterInfo> adapters) async {
    if (_context == null) throw CactusException('CactusTTS not initialized');
    await _context!.applyLoraAdapters(adapters);
  }

  Future<void> removeLoraAdapters() async {
    if (_context == null) throw CactusException('CactusTTS not initialized');
    await _context!.removeLoraAdapters();
  }

  Future<List<LoraAdapterInfo>> getLoadedLoraAdapters() async {
    if (_context == null) throw CactusException('CactusTTS not initialized');
    return await _context!.getLoadedLoraAdapters();
  }

  Future<void> rewind() async {
    if (_context == null) throw CactusException('CactusTTS not initialized');
    await _context!.rewind();
  }

  Future<void> stopCompletion() async {
    if (_context == null) throw CactusException('CactusTTS not initialized');
    await _context!.stopCompletion();
  }

  void dispose() {
    _context?.release();
    _context = null;
  }
} 