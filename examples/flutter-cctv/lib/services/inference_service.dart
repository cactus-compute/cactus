import 'package:cactus/cactus.dart';
import 'package:camera/camera.dart';
import '../cactus_initializer.dart';
import '../utils/image_converter.dart';

/// Service to manage Cactus inference
class InferenceService {
  CactusContext? _cactusContext;
  bool _isRunning = false;
  String _lastResult = '';

  // Getters
  bool get isInitialized => _cactusContext != null;
  bool get isRunning => _isRunning;
  String get lastResult => _lastResult;

  /// Initialize the Cactus framework
  Future<bool> initialize({
    required Function(String message, {double? progress}) onStatus,
  }) async {
    try {
      _cactusContext = await CactusInit.init(onStatus: onStatus);
      return _cactusContext != null;
    } catch (e) {
      print('Error initializing Cactus: $e');
      return false;
    }
  }

  /// Process a camera image frame
  Future<String> processFrame(
    CameraImage image,
    String tempFilePath, {
    String prompt = '<__image__>\nProvide a short description',
    int maxTokens = 20,
  }) async {
    if (_cactusContext == null) {
      throw Exception('Cactus not initialized');
    }

    if (_isRunning) {
      return _lastResult.isEmpty ? 'Inference already running' : _lastResult;
    }

    _isRunning = true;

    try {
      // Convert camera image to file
      final imageFile = await ImageConverter.convertImageToFile(
        image,
        tempFilePath,
      );

      // Run inference
      final result = await _cactusContext!.completion(
        CactusCompletionParams(
          messages: [ChatMessage(role: 'user', content: prompt)],
          imagePath: imageFile.path,
          temperature: 0.7,
          maxPredictedTokens: maxTokens,
          stopSequences: ['<end_of_utterance>', '<|im_end|>'],
        ),
      );

      // Cleanup temp file
      try {
        await imageFile.delete();
      } catch (e) {
        print('Error deleting temp file: $e');
      }

      _lastResult = result.text;
      return result.text;
    } catch (e) {
      print('Error during inference: $e');
      return 'Error: $e';
    } finally {
      _isRunning = false;
    }
  }

  /// Clean up resources
  void dispose() {
    if (_cactusContext != null) {
      _cactusContext!.free();
      _cactusContext = null;
    }
  }
}
