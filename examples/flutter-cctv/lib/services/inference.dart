import 'package:cactus/cactus.dart';
import 'package:camera/camera.dart';
import '../utils/timer.dart';
import 'cactus_initializer.dart';
import 'image_converter.dart';
import 'dart:io';
import 'dart:math';
import 'package:path_provider/path_provider.dart';

class InferenceService {
  // Singleton implementation
  static final InferenceService _instance = InferenceService._internal();
  factory InferenceService() => _instance;
  InferenceService._internal();

  bool _isProcessing = false;
  String _resultText = 'No results';
  final int _framesToSkip = 30;
  int _frameCount = 0;
  CactusContext? _cactusContext;
  File? _lastProcessedImageFile;
  
  // Array of random user inputs
  final List<String> _userInputs = [
    "What's the weather like today?",
    "How do I bake chocolate chip cookies?",
    "Tell me about the Eiffel Tower.",
    "What's the capital of Japan?",
    "How tall is Mount Everest?",
    "What's the best movie of 2023?",
    "How do I learn programming?",
    "What's the meaning of life?",
    "Who invented the telephone?",
    "How far is the Moon from Earth?",
    "What are some good books to read?",
    "How do I grow tomatoes?",
    "What's the largest animal on Earth?",
    "How many countries are in Africa?",
    "What causes rainbows?",
    "How do I make pasta from scratch?",
    "What's the fastest car in the world?",
    "How do I learn a new language?",
    "What's the population of New York City?",
    "How do I fix a leaky faucet?",
    "What are black holes?",
    "How do airplanes fly?",
    "What's the oldest civilization?",
    "How many planets are in our solar system?",
    "What's the deepest part of the ocean?",
    "How do I make bread?",
    "What's the tallest building in the world?",
    "How do computers work?",
    "What's the biggest desert?",
    "How do I improve my memory?",
    "What are some healthy breakfast ideas?",
    "How does photosynthesis work?",
    "What's the Great Wall of China?",
    "How do I start meditating?",
    "What causes earthquakes?",
    "How do I train my dog?",
    "What's the most spoken language?",
    "How do vaccines work?",
    "What's the largest country by area?",
    "How do I get better sleep?",
    "What are renewable energy sources?",
    "How do I make pancakes?",
    "What's the theory of relativity?",
    "How does the internet work?",
    "What's the biggest animal that ever lived?",
    "How do I tie a tie?",
    "What causes thunder and lightning?",
    "How do I plant a tree?",
    "What's the smallest country in the world?",
    "How do I start exercising?",
    "What's the most popular sport in the world?",

  ];

  get isProcessing => _isProcessing;
  File? getLastProcessedImage() => _lastProcessedImageFile;

  /// Initialize the Cactus framework
  Future<bool> initialize() async {
    try {
      _cactusContext = await CactusInit.init(onStatus: timer.log);
      return _cactusContext != null;
    } catch (e) {
      print('Error initializing Cactus: $e');
      return false;
    }
  }

  Future<String> mockAnalyzeFrame(CameraImage image) async {
    // Pick a random question from the array
    timer.log('Starting mock inference');
    final random = Random();
    final randomQuestion = _userInputs[random.nextInt(_userInputs.length)];

    final result = await _cactusContext!.completion(
      CactusCompletionParams(
        messages: [
          ChatMessage(role: 'system', content: 'You are a helpful assistant.'),
          ChatMessage(role: 'user', content: randomQuestion)
        ],
      ),
    );
    timer.log('Inference complete: ${result.text}');
    _resultText = result.text;
    return result.text;
  }

  Future<String> analyzeFrame(CameraImage image) async {
    // Pick a random question from the array
    timer.log('Starting inference');

    final tempDir = await getTemporaryDirectory();
    final imagePath = ImageConverter.getUniqueImagePath(tempDir.path);
    final imageFile = await ImageConverter.convertImageToFile(image, imagePath);
    _lastProcessedImageFile = imageFile;
    timer.log('Image converted and saved (${imageFile.lengthSync() / 1024 / 1024} Mb)');

    final result = await _cactusContext!.completion(
      CactusCompletionParams(
        messages: [
          ChatMessage(role: 'system', content: 'You are an image analyst. Your task is to provide short, concise descriptions of the scene in the frame.'),
          ChatMessage(role: 'user', content: '<__image__>Describe the frame.')
        ],
        imagePath: imageFile.path,
        maxPredictedTokens: 20,
        stopSequences: ['<end_of_utterance>'],
      ),
    );
    timer.log('Inference complete: ${result.text}');
    return result.text;
  }

  Future<String?> inferenceCallback(CameraImage image) async {
    if (_isProcessing) {
      return null;
    }
    
    if (_frameCount < _framesToSkip) {
      _frameCount++;
      return null;
    }
    
    _isProcessing = true;
    _frameCount = 0;
    
    try {
      return await analyzeFrame(image);
    } finally {
      _isProcessing = false;
    }
  }
}