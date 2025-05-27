import 'package:cactus/cactus.dart';
import 'dart:io';
import 'dart:async';
import '../cactus_initializer.dart';

/// Repository for interacting with the Cactus ML framework
class CactusRepository {
  CactusContext? _cactusContext;
  
  // Getters
  CactusContext? get context => _cactusContext;
  bool get isInitialized => _cactusContext != null;
  
  /// Initialize the Cactus framework
  Future<CactusContext?> initialize({
    required Function(String message, {double? progress}) onStatus,
  }) async {
    try {
      print('Initializing Cactus repository');
      _cactusContext = await CactusInit.init(onStatus: onStatus);
      return _cactusContext;
    } catch (e) {
      print('Error initializing Cactus repository: $e');
      return null;
    }
  }
  
  /// Run inference on an image
  Future<String> processImage(
    File imageFile, {
    String prompt = '<__image__>\nProvide a short description of the image',
    double temperature = 0.7,
    int maxPredictedTokens = 20,
    List<String> stopSequences = const ['<end_of_utterance>', '<|im_end|>'],
  }) async {
    if (_cactusContext == null) {
      throw Exception('Cactus not initialized');
    }
    
    try {
      final result = await _cactusContext!.completion(CactusCompletionParams(
        messages: [ChatMessage(role: 'user', content: prompt)],
        imagePath: imageFile.path,
        temperature: temperature,
        maxPredictedTokens: maxPredictedTokens,
        stopSequences: stopSequences,
      ));
      
      return result.text;
    } catch (e) {
      print('Error during Cactus inference: $e');
      rethrow;
    }
  }
  
  /// Clean up resources
  Future<void> dispose() async {
    if (_cactusContext != null) {
      _cactusContext!.free();
      _cactusContext = null;
      print('Cactus repository disposed');
    }
  }
}

/// Access to the CactusInit functionality 
/// (assumes this class exists in your project)
class CactusInit {
  static Future<CactusContext?> init({
    required Function(String message, {double? progress}) onStatus,
  }) async {
    // This is a placeholder - you should replace this with your actual CactusInit implementation
    // or import it from your existing cactus_initializer.dart file
    
    // For now, we'll import and use your existing implementation
    throw UnimplementedError('Please import and use your existing CactusInit class');
  }
} 