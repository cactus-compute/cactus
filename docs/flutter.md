# Cactus Flutter

A powerful Flutter plugin for running Large Language Models (LLMs) and Vision Language Models (VLMs) directly on mobile devices, with full support for chat completions, multimodal inputs, embeddings, text-to-speech and advanced features.

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  cactus: ^0.1.0
```

Then run:
```bash
flutter pub get
```

**Platform Requirements:**
- **iOS**: iOS 12.0+, Xcode 14+
- **Android**: API level 24+, NDK support
- **Flutter**: 3.3.0+, Dart 3.0+

## Quick Start

### Basic Text Completion

```dart
import 'package:cactus/cactus.dart';

Future<String> basicCompletion() async {
  // Initialize context
  final context = await CactusContext.init(CactusInitParams(
    modelPath: '/path/to/model.gguf',
    contextSize: 2048,
    threads: 4,
  ));

  // Generate response
  final result = await context.completion(CactusCompletionParams(
    messages: [
      ChatMessage(role: 'user', content: 'Hello, how are you?')
    ],
    maxPredictedTokens: 100,
    temperature: 0.7,
  ));

  context.free();
  return result.text;
}
```

### Complete Chat App Example

```dart
import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';

void main() => runApp(ChatApp());

class ChatApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cactus Chat',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: ChatScreen(),
    );
  }
}

class ChatScreen extends StatefulWidget {
  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  CactusContext? _context;
  List<ChatMessage> _messages = [];
  TextEditingController _controller = TextEditingController();
  bool _isLoading = true;
  bool _isGenerating = false;

  @override
  void initState() {
    super.initState();
    _initializeModel();
  }

  @override
  void dispose() {
    _context?.free();
    super.dispose();
  }

  Future<void> _initializeModel() async {
    try {
      // Initialize context using a model URL.
      // The model will be downloaded and cached automatically.
      _context = await CactusContext.init(CactusInitParams(
        modelUrl: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
        contextSize: 4096,
        threads: 4,
        gpuLayers: 99,
        onInitProgress: (progress, status, isError) {
          print('Init: $status ${progress != null ? '${(progress * 100).toInt()}%' : ''}');
          // You can update UI here, e.g., using setState or a ValueNotifier
        },
      ));

      setState(() => _isLoading = false);
    } catch (e) {
      print('Failed to initialize: $e');
      setState(() => _isLoading = false);
    }
  }

  Future<void> _sendMessage() async {
    if (_context == null || _controller.text.trim().isEmpty) return;

    final userMessage = ChatMessage(role: 'user', content: _controller.text.trim());
    setState(() {
      _messages.add(userMessage);
      _messages.add(ChatMessage(role: 'assistant', content: ''));
      _isGenerating = true;
    });

    _controller.clear();
    String currentResponse = '';

    try {
      final result = await _context!.completion(CactusCompletionParams(
        messages: List.from(_messages)..removeLast(), // Remove empty assistant message
        maxPredictedTokens: 256,
        temperature: 0.7,
        stopSequences: ['<|end|>', '</s>'],
        onNewToken: (token) {
          currentResponse += token;
          setState(() {
            _messages.last = ChatMessage(role: 'assistant', content: currentResponse);
          });
          return true; // Continue generation
        },
      ));

      // Update with final response
      setState(() {
        _messages.last = ChatMessage(role: 'assistant', content: result.text.trim());
      });
    } catch (e) {
      setState(() {
        _messages.last = ChatMessage(role: 'assistant', content: 'Error: $e');
      });
    } finally {
      setState(() => _isGenerating = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text('Loading model...'),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: Text('Cactus Chat'),
        actions: [
          IconButton(
            icon: Icon(Icons.clear),
            onPressed: () => setState(() => _messages.clear()),
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              padding: EdgeInsets.all(16),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final message = _messages[index];
                final isUser = message.role == 'user';
                
                return Align(
                  alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                  child: Container(
                    margin: EdgeInsets.symmetric(vertical: 4),
                    padding: EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: isUser ? Colors.blue : Colors.grey[300],
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      message.content,
                      style: TextStyle(
                        color: isUser ? Colors.white : Colors.black,
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
          Padding(
            padding: EdgeInsets.all(16),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: InputDecoration(
                      hintText: 'Type a message...',
                      border: OutlineInputBorder(),
                    ),
                    onSubmitted: (_) => _sendMessage(),
                  ),
                ),
                SizedBox(width: 8),
                IconButton(
                  icon: _isGenerating 
                    ? SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
                    : Icon(Icons.send),
                  onPressed: _isGenerating ? null : _sendMessage,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

## Text Completion

### Basic Generation

```dart
Future<String> generateText(CactusContext context, String prompt) async {
  final result = await context.completion(CactusCompletionParams(
    messages: [ChatMessage(role: 'user', content: prompt)],
    maxPredictedTokens: 200,
    temperature: 0.8,
    topK: 40,
    topP: 0.9,
  ));
  
  return result.text;
}

// Usage
final response = await generateText(context, 'Write a short poem about Flutter development');
print(response);
```

### Streaming Generation

```dart
class StreamingChatWidget extends StatefulWidget {
  final CactusContext context;
  
  StreamingChatWidget({required this.context});
  
  @override
  _StreamingChatWidgetState createState() => _StreamingChatWidgetState();
}

class _StreamingChatWidgetState extends State<StreamingChatWidget> {
  String _currentResponse = '';
  bool _isGenerating = false;

  Future<void> _generateStreaming(String prompt) async {
    setState(() {
      _currentResponse = '';
      _isGenerating = true;
    });

    try {
      await widget.context.completion(CactusCompletionParams(
        messages: [ChatMessage(role: 'user', content: prompt)],
        maxPredictedTokens: 500,
        temperature: 0.7,
        onNewToken: (token) {
          setState(() {
            _currentResponse += token;
          });
          return true; // Continue generation
        },
      ));
    } finally {
      setState(() => _isGenerating = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          padding: EdgeInsets.all(16),
          child: Text(
            _currentResponse.isEmpty ? 'No response yet' : _currentResponse,
            style: TextStyle(fontSize: 16),
          ),
        ),
        if (_isGenerating)
          LinearProgressIndicator(),
        ElevatedButton(
          onPressed: _isGenerating ? null : () => _generateStreaming('Tell me a story'),
          child: Text(_isGenerating ? 'Generating...' : 'Generate Story'),
        ),
      ],
    );
  }
}
```

### Advanced Parameter Control

```dart
class AdvancedCompletionService {
  final CactusContext context;
  
  AdvancedCompletionService(this.context);

  Future<CactusCompletionResult> creativeCompletion(List<ChatMessage> messages) async {
    return await context.completion(CactusCompletionParams(
      messages: messages,
      maxPredictedTokens: 512,
      
      // Creative settings
      temperature: 0.9,          // High randomness
      topK: 60,                  // Broader sampling
      topP: 0.95,               // More diverse outputs
      
      // Repetition control
      penaltyRepeat: 1.15,      // Strong repetition penalty
      penaltyFreq: 0.1,         // Frequency penalty
      penaltyPresent: 0.1,      // Presence penalty
      
      stopSequences: ['</story>', '<|end|>'],
    ));
  }

  Future<CactusCompletionResult> factualCompletion(List<ChatMessage> messages) async {
    return await context.completion(CactusCompletionParams(
      messages: messages,
      maxPredictedTokens: 256,
      
      // Focused settings
      temperature: 0.3,          // Low randomness
      topK: 20,                  // Focused sampling
      topP: 0.8,                // Conservative
      
      // Minimal repetition penalty
      penaltyRepeat: 1.05,
      
      stopSequences: ['\n\n', '<|end|>'],
    ));
  }

  Future<CactusCompletionResult> codeCompletion(List<ChatMessage> messages) async {
    return await context.completion(CactusCompletionParams(
      messages: messages,
      maxPredictedTokens: 1024,
      
      // Code-optimized settings
      temperature: 0.1,          // Very focused
      topK: 10,                  // Narrow sampling
      topP: 0.7,                // Deterministic
      
      stopSequences: ['```', '\n\n\n'],
    ));
  }
}
```

### Conversation Management

```dart
class ConversationManager {
  final CactusContext context;
  final List<ChatMessage> _history = [];
  
  ConversationManager(this.context);

  void addSystemMessage(String content) {
    _history.insert(0, ChatMessage(role: 'system', content: content));
  }

  Future<String> sendMessage(String userMessage) async {
    // Add user message
    _history.add(ChatMessage(role: 'user', content: userMessage));
    
    // Generate response
    final result = await context.completion(CactusCompletionParams(
      messages: _history,
      maxPredictedTokens: 256,
      temperature: 0.7,
      stopSequences: ['<|end|>', '</s>'],
    ));
    
    // Add assistant response
    _history.add(ChatMessage(role: 'assistant', content: result.text));
    
    return result.text;
  }

  void clearHistory() => _history.clear();
  
  List<ChatMessage> get history => List.unmodifiable(_history);
  
  int get messageCount => _history.length;
  
  // Keep only recent messages to manage context size
  void trimHistory({int maxMessages = 20}) {
    if (_history.length > maxMessages) {
      // Keep system message (if first) and recent messages
      final systemMessages = _history.where((m) => m.role == 'system').take(1).toList();
      final recentMessages = _history.where((m) => m.role != 'system').toList();
      
      if (recentMessages.length > maxMessages - systemMessages.length) {
        recentMessages.removeRange(0, recentMessages.length - (maxMessages - systemMessages.length));
      }
      
      _history.clear();
      _history.addAll(systemMessages);
      _history.addAll(recentMessages);
    }
  }
}

// Usage
final conversation = ConversationManager(context);
conversation.addSystemMessage('You are a helpful coding assistant.');

final response1 = await conversation.sendMessage('How do I create a ListView in Flutter?');
final response2 = await conversation.sendMessage('Can you show me a code example?');

print('Conversation has ${conversation.messageCount} messages');
conversation.trimHistory(maxMessages: 10);
```

## Core Concepts

### Context Management

The `CactusContext` is the main interface for model operations:

```dart
final context = await CactusContext.init(CactusInitParams(
  modelPath: '/path/to/model.gguf',     // Model file path
  contextSize: 4096,                    // Context window size
  batchSize: 512,                       // Batch size for processing
  threads: 4,                           // CPU threads
  gpuLayers: 99,                        // GPU layers (0 = CPU only)
  chatTemplate: 'chatml',               // Chat template
  onInitProgress: (progress, status, isError) {
    if (isError) {
      print('Error: $status');
    } else {
      print('$status ${progress != null ? '${(progress * 100).toInt()}%' : ''}');
    }
  },
));
```

### Message Format

Cactus uses a simple message format compatible with most chat models:

```dart
class ChatMessage {
  final String role;    // 'system', 'user', or 'assistant'
  final String content; // Message content
  
  ChatMessage({required this.role, required this.content});
}

final messages = [
  ChatMessage(role: 'system', content: 'You are a helpful assistant.'),
  ChatMessage(role: 'user', content: 'What is the capital of France?'),
  ChatMessage(role: 'assistant', content: 'The capital of France is Paris.'),
  ChatMessage(role: 'user', content: 'What about Germany?'),
];
```

### Generation Workflow

The standard generation process:

```dart
// 1. Initialize context
final context = await CactusContext.init(params);

// 2. Configure completion parameters
final completionParams = CactusCompletionParams(
  messages: messages,
  maxPredictedTokens: 256,
  temperature: 0.7,
);

// 3. Generate response
final result = await context.completion(completionParams);

// 4. Use the result
print('Response: ${result.text}');
print('Tokens predicted: ${result.tokensPredicted}');

// 5. Clean up when done
context.free();
```

## Multimodal (Vision)

### Basic Image Analysis

```dart
class VisionAnalyzer {
  final CactusContext context;
  
  VisionAnalyzer(this.context);

  Future<String> analyzeImage(String imagePath, String question) async {
    // Ensure multimodal is enabled
    if (!context.isMultimodalEnabled()) {
      throw Exception('Multimodal support not initialized');
    }

    final result = await context.multimodalCompletion(
      CactusCompletionParams(
        messages: [
          ChatMessage(role: 'user', content: question),
        ],
        maxPredictedTokens: 300,
        temperature: 0.3, // Lower temperature for more accurate descriptions
      ),
      [imagePath], // List of image paths
    );

    return result.text;
  }

  Future<String> describeImage(String imagePath) async {
    return await analyzeImage(
      imagePath,
      'Describe this image in detail. What do you see?'
    );
  }

  Future<String> answerImageQuestion(String imagePath, String question) async {
    return await analyzeImage(imagePath, question);
  }
}

// Usage
final analyzer = VisionAnalyzer(context);
final description = await analyzer.describeImage('/path/to/image.jpg');
final answer = await analyzer.answerImageQuestion(
  '/path/to/chart.png', 
  'What is the trend shown in this chart?'
);
```

### Complete Vision Chat App

```dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import 'package:path_provider/path_provider.dart';
import 'package:image_picker/image_picker.dart';

class VisionChatScreen extends StatefulWidget {
  @override
  _VisionChatScreenState createState() => _VisionChatScreenState();
}

class _VisionChatScreenState extends State<VisionChatScreen> {
  CactusContext? _context;
  List<ChatMessage> _messages = [];
  String? _selectedImagePath;
  bool _isLoading = true;
  bool _lastWasMultimodal = false;

  @override
  void initState() {
    super.initState();
    _initializeModel();
  }

  @override
  void dispose() {
    _context?.free();
    super.dispose();
  }

  Future<void> _initializeModel() async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      
      _context = await CactusContext.init(CactusInitParams(
        modelUrl: 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf',
        mmprojUrl: 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf',
        contextSize: 4096,
        onInitProgress: (progress, status, isError) {
          print('$status ${progress != null ? '${(progress * 100).toInt()}%' : ''}');
        },
      ));

      setState(() => _isLoading = false);
    } catch (e) {
      print('Failed to initialize vision model: $e');
      setState(() => _isLoading = false);
    }
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.gallery);
    
    if (image != null) {
      setState(() {
        _selectedImagePath = image.path;
      });
    }
  }

  Future<void> _sendMessage(String text) async {
    if (_context == null) return;

    final isMultimodal = _selectedImagePath != null;
    
    // Clear history if switching modes (multimodal ↔ text)
    if (_lastWasMultimodal != isMultimodal) {
      setState(() {
        _messages.clear();
      });
    }
    _lastWasMultimodal = isMultimodal;

    // Add user message
    final userMessage = ChatMessage(role: 'user', content: text);
    setState(() {
      _messages.add(userMessage);
      _messages.add(ChatMessage(role: 'assistant', content: ''));
    });

    String currentResponse = '';

    try {
      final params = CactusCompletionParams(
        messages: List.from(_messages)..removeLast(),
        maxPredictedTokens: 400,
        temperature: isMultimodal ? 0.3 : 0.7, // Lower temp for vision
        onNewToken: (token) {
          currentResponse += token;
          setState(() {
            _messages.last = ChatMessage(role: 'assistant', content: currentResponse);
          });
          return true;
        },
      );

      CactusCompletionResult result;
      if (isMultimodal && _selectedImagePath != null) {
        result = await _context!.multimodalCompletion(params, [_selectedImagePath!]);
      } else {
        result = await _context!.completion(params);
      }

      setState(() {
        _messages.last = ChatMessage(role: 'assistant', content: result.text.trim());
        _selectedImagePath = null; // Clear image after use
      });

    } catch (e) {
      setState(() {
        _messages.last = ChatMessage(role: 'assistant', content: 'Error: $e');
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: Text('Vision Chat'),
        actions: [
          IconButton(
            icon: Icon(Icons.image),
            onPressed: _pickImage,
          ),
          IconButton(
            icon: Icon(Icons.clear),
            onPressed: () => setState(() => _messages.clear()),
          ),
        ],
      ),
      body: Column(
        children: [
          // Selected image preview
          if (_selectedImagePath != null)
            Container(
              height: 100,
              margin: EdgeInsets.all(8),
              child: Row(
                children: [
                  Image.file(
                    File(_selectedImagePath!),
                    height: 80,
                    width: 80,
                    fit: BoxFit.cover,
                  ),
                  SizedBox(width: 8),
                  Expanded(
                    child: Text('Image selected for next message'),
                  ),
                  IconButton(
                    icon: Icon(Icons.close),
                    onPressed: () => setState(() => _selectedImagePath = null),
                  ),
                ],
              ),
            ),
          
          // Messages
          Expanded(
            child: ListView.builder(
              padding: EdgeInsets.all(16),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final message = _messages[index];
                final isUser = message.role == 'user';
                
                return Align(
                  alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                  child: Container(
                    margin: EdgeInsets.symmetric(vertical: 4),
                    padding: EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: isUser ? Colors.blue : Colors.grey[300],
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      message.content,
                      style: TextStyle(
                        color: isUser ? Colors.white : Colors.black,
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
          
          // Input field
          _MessageInput(
            onSendMessage: _sendMessage,
            hasImage: _selectedImagePath != null,
          ),
        ],
      ),
    );
  }
}

class _MessageInput extends StatefulWidget {
  final Function(String) onSendMessage;
  final bool hasImage;

  _MessageInput({required this.onSendMessage, required this.hasImage});

  @override
  _MessageInputState createState() => _MessageInputState();
}

class _MessageInputState extends State<_MessageInput> {
  final _controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(16),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _controller,
              decoration: InputDecoration(
                hintText: widget.hasImage 
                  ? 'Ask about the image...'
                  : 'Type a message...',
                border: OutlineInputBorder(),
                prefixIcon: widget.hasImage ? Icon(Icons.image, color: Colors.blue) : null,
              ),
              onSubmitted: (text) {
                if (text.trim().isNotEmpty) {
                  widget.onSendMessage(text.trim());
                  _controller.clear();
                }
              },
            ),
          ),
          SizedBox(width: 8),
          IconButton(
            icon: Icon(Icons.send),
            onPressed: () {
              final text = _controller.text.trim();
              if (text.isNotEmpty) {
                widget.onSendMessage(text);
                _controller.clear();
              }
            },
          ),
        ],
      ),
    );
  }
}

### Image Processing Utilities

> **Note**: The following example uses the [image](https://pub.dev/packages/image) package for image processing. You'll need to add it to your `pubspec.yaml` to run this code.

```dart
import 'dart:io';
import 'dart:convert';
import 'package:cactus/cactus.dart';
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart' as img;
import 'package:path/path.dart' as path;

class ImageProcessor {
  static Future<String> downloadAndProcessImage(String imageUrl) async {
    // Download image from URL
    final tempDir = await getTemporaryDirectory();
    final fileName = 'temp_image_${DateTime.now().millisecondsSinceEpoch}.jpg';
    final filePath = '${tempDir.path}/$fileName';
    
    await downloadImage(imageUrl, filePath: filePath);
    return filePath;
  }

  static Future<String> resizeImageIfNeeded(String imagePath) async {
    final file = File(imagePath);
    final image = img.decodeImage(await file.readAsBytes());
    
    if (image == null) throw Exception('Invalid image format');
    
    // Resize if too large (optional optimization)
    if (image.width > 1024 || image.height > 1024) {
      final resized = img.copyResize(
        image,
        width: image.width > image.height ? 1024 : null,
        height: image.height > image.width ? 1024 : null,
      );
      
      final tempDir = await getTemporaryDirectory();
      final resizedPath = '${tempDir.path}/resized_${DateTime.now().millisecondsSinceEpoch}.jpg';
      await File(resizedPath).writeAsBytes(img.encodeJpg(resized));
      
      return resizedPath;
    }
    
    return imagePath;
  }

  static Future<List<String>> extractTextFromImage(
    CactusContext context, 
    String imagePath
  ) async {
    final result = await context.multimodalCompletion(
      CactusCompletionParams(
        messages: [
          ChatMessage(
            role: 'user',
            content: 'Extract all text visible in this image. List each text element on a new line.',
          ),
        ],
        maxPredictedTokens: 500,
        temperature: 0.1,
      ),
      [imagePath],
    );

    return result.text
        .split('\n')
        .map((line) => line.trim())
        .where((line) => line.isNotEmpty)
        .toList();
  }

  static Future<Map<String, dynamic>> analyzeImageStructure(
    CactusContext context,
    String imagePath
  ) async {
    final result = await context.multimodalCompletion(
      CactusCompletionParams(
        messages: [
          ChatMessage(
            role: 'user',
            content: '''Analyze this image and provide a structured description in JSON format:
{
  "main_objects": ["object1", "object2"],
  "colors": ["color1", "color2"],
  "setting": "description of setting",
  "people_count": number,
  "text_present": true/false,
  "mood_or_atmosphere": "description"
}''',
          ),
        ],
        maxPredictedTokens: 400,
        temperature: 0.2,
      ),
      [imagePath],
    );

    try {
      return jsonDecode(result.text);
    } catch (e) {
      return {'error': 'Failed to parse JSON', 'raw_response': result.text};
    }
  }
}
```

## Embeddings

### Text Embeddings

```dart
class EmbeddingService {
  final CactusContext context;
  
  EmbeddingService(this.context);

  Future<List<double>> getEmbedding(String text) async {
    return await context.embedding(text);
  }

  Future<double> cosineSimilarity(List<double> a, List<double> b) async {
    if (a.length != b.length) throw ArgumentError('Vectors must have same length');
    
    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    
    for (int i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    return dotProduct / (math.sqrt(normA) * math.sqrt(normB));
  }

  Future<List<DocumentMatch>> findSimilarDocuments(
    String query,
    List<String> documents,
    {double threshold = 0.7}
  ) async {
    final queryEmbedding = await getEmbedding(query);
    final matches = <DocumentMatch>[];
    
    for (int i = 0; i < documents.length; i++) {
      final docEmbedding = await getEmbedding(documents[i]);
      final similarity = await cosineSimilarity(queryEmbedding, docEmbedding);
      
      if (similarity >= threshold) {
        matches.add(DocumentMatch(
          document: documents[i],
          similarity: similarity,
          index: i,
        ));
      }
    }
    
    matches.sort((a, b) => b.similarity.compareTo(a.similarity));
    return matches;
  }
}

class DocumentMatch {
  final String document;
  final double similarity;
  final int index;
  
  DocumentMatch({
    required this.document,
    required this.similarity,
    required this.index,
  });
}

// Usage
final embeddingService = EmbeddingService(context);
final matches = await embeddingService.findSimilarDocuments(
  'machine learning algorithms',
  [
    'Deep learning neural networks',
    'Cooking recipes and food preparation',
    'Supervised learning techniques',
    'Weather forecast data',
  ],
  threshold: 0.6,
);

for (final match in matches) {
  print('${match.similarity.toStringAsFixed(3)}: ${match.document}');
}
```

## Advanced Features

### Structured Output with JSON Schema

```dart
class StructuredOutputService {
  final CactusContext context;
  
  StructuredOutputService(this.context);

  Future<Map<String, dynamic>> generateStructuredResponse(
    String prompt,
    Map<String, dynamic> jsonSchema,
  ) async {
    final result = await context.completionSmart(
      CactusCompletionParams(
        messages: [ChatMessage(role: 'user', content: prompt)],
        maxPredictedTokens: 400,
        temperature: 0.3,
        responseFormat: ResponseFormat.jsonSchema(schema: jsonSchema),
      ),
    );

    return jsonDecode(result.text);
  }

  Future<Map<String, dynamic>> extractPersonInfo(String text) async {
    final schema = {
      'type': 'object',
      'properties': {
        'name': {'type': 'string'},
        'age': {'type': 'integer'},
        'occupation': {'type': 'string'},
        'location': {'type': 'string'},
        'skills': {
          'type': 'array',
          'items': {'type': 'string'}
        }
      },
      'required': ['name']
    };

    return await generateStructuredResponse(
      'Extract person information from this text: $text',
      schema,
    );
  }
}

// Usage
final service = StructuredOutputService(context);
final personInfo = await service.extractPersonInfo(
  'John Smith is a 32-year-old software engineer from San Francisco. He specializes in Flutter development and machine learning.'
);
print(personInfo); // {'name': 'John Smith', 'age': 32, ...}
```

### Tool Calling

```dart
class WeatherService {
  Future<Map<String, dynamic>> getCurrentWeather(String location) async {
    // Simulate API call
    await Future.delayed(Duration(seconds: 1));
    return {
      'location': location,
      'temperature': 22,
      'condition': 'sunny',
      'humidity': 65
    };
  }
}

class ToolCallingExample {
  final CactusContext context;
  final WeatherService weatherService = WeatherService();
  
  ToolCallingExample(this.context);

  Future<String> chatWithTools(String userMessage) async {
    // Setup tools
    final tools = Tools();
    
    tools.addSimpleTool(
      name: 'get_weather',
      description: 'Get current weather for a location',
      parameters: {
        'type': 'object',
        'properties': {
          'location': {
            'type': 'string',
            'description': 'The city and state/country'
          }
        },
        'required': ['location']
      },
      function: (args) async {
        final location = args['location'] as String;
        return await weatherService.getCurrentWeather(location);
      },
    );

    // Use smart completion with tools
    final result = await context.completionSmart(
      CactusCompletionParams(
        messages: [
          ChatMessage(role: 'system', content: 'You are a helpful assistant.'),
          ChatMessage(role: 'user', content: userMessage),
        ],
        maxPredictedTokens: 300,
        temperature: 0.7,
      ),
      tools: tools,
    );

    return result.text;
  }
}

// Usage
final toolExample = ToolCallingExample(context);
final response = await toolExample.chatWithTools(
  'What\'s the weather like in New York?'
);
print(response);
```

### Performance Optimization

```dart
class OptimizedChatService {
  final CactusContext context;
  final int maxContextTokens;
  List<ChatMessage> _conversationHistory = [];
  
  OptimizedChatService(this.context) : maxContextTokens = context.getNCtx() ?? 4096;

  Future<String> sendMessage(String userMessage) async {
    // Add user message
    _conversationHistory.add(ChatMessage(role: 'user', content: userMessage));
    
    // Optimize context window usage
    final optimizedHistory = await _optimizeHistory();
    
    // Generate response with optimized parameters
    final result = await context.completion(CactusCompletionParams(
      messages: optimizedHistory,
      maxPredictedTokens: _calculateOptimalTokens(),
      temperature: 0.7,
      
      // Performance optimizations
      threads: 4, // Use multiple threads
      topK: 40,   // Limit sampling space
      topP: 0.9,  // Nucleus sampling
      
      // Reduce repetition
      penaltyRepeat: 1.1,
      penaltyLastN: 64,
    ));
    
    // Add response to history
    _conversationHistory.add(ChatMessage(role: 'assistant', content: result.text));
    
    return result.text;
  }

  Future<List<ChatMessage>> _optimizeHistory() async {
    // Keep system message + recent messages within context limit
    final systemMessages = _conversationHistory.where((m) => m.role == 'system').toList();
    final otherMessages = _conversationHistory.where((m) => m.role != 'system').toList();
    
    // Estimate tokens (rough approximation)
    int totalTokens = 0;
    List<ChatMessage> optimized = [...systemMessages];
    
    // Add messages from most recent backwards
    for (int i = otherMessages.length - 1; i >= 0; i--) {
      final message = otherMessages[i];
      final estimatedTokens = message.content.length ~/ 4; // Rough estimate
      
      if (totalTokens + estimatedTokens < maxContextTokens * 0.8) { // Leave 20% for response
        optimized.insert(systemMessages.length, message);
        totalTokens += estimatedTokens;
      } else {
        break;
      }
    }
    
    return optimized;
  }

  int _calculateOptimalTokens() {
    final usedTokens = _conversationHistory
        .map((m) => m.content.length ~/ 4)
        .fold(0, (sum, tokens) => sum + tokens);
    
    return math.min(512, maxContextTokens - usedTokens - 100); // Leave buffer
  }

  void clearHistory() => _conversationHistory.clear();
}
```

## Best Practices

### Memory Management

```dart
class CactusManager {
  CactusContext? _context;
  
  Future<void> initialize() async {
    _context = await CactusContext.init(/* params */);
  }
  
  Future<void> dispose() async {
    _context?.free(); // Always free the context
    _context = null;
  }
  
  bool get isInitialized => _context != null;
}

// In your app
class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> with WidgetsBindingObserver {
  final CactusManager _cactusManager = CactusManager();
  
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _cactusManager.initialize();
  }
  
  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cactusManager.dispose();
    super.dispose();
  }
  
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused) {
      // Optionally pause/save state
    } else if (state == AppLifecycleState.resumed) {
      // Resume operations
    }
  }
}
```

### Error Handling

```dart
class RobustChatService {
  final CactusContext context;
  
  RobustChatService(this.context);

  Future<String> generateWithRetry(
    String prompt, {
    int maxRetries = 3,
    Duration retryDelay = const Duration(seconds: 1),
  }) async {
    for (int attempt = 0; attempt < maxRetries; attempt++) {
      try {
        final result = await context.completion(CactusCompletionParams(
          messages: [ChatMessage(role: 'user', content: prompt)],
          maxPredictedTokens: 256,
        ));
        
        if (result.text.trim().isNotEmpty) {
          return result.text;
        }
        
        throw CactusCompletionException('Empty response generated');
        
      } on CactusCompletionException catch (e) {
        if (attempt == maxRetries - 1) rethrow;
        print('Attempt ${attempt + 1} failed: ${e.message}');
        await Future.delayed(retryDelay);
        
      } catch (e) {
        if (attempt == maxRetries - 1) rethrow;
        print('Unexpected error on attempt ${attempt + 1}: $e');
        await Future.delayed(retryDelay);
      }
    }
    
    throw Exception('All retry attempts failed');
  }
}
```

### Monitoring & Analytics

```dart
class CactusAnalytics {
  static void trackCompletion({
    required int tokensPredicted,
    required int tokensEvaluated,
    required Duration duration,
    required String modelName,
  }) {
    final tokensPerSecond = tokensPredicted / duration.inMilliseconds * 1000;
    
    // Log performance metrics
    print('Performance: ${tokensPerSecond.toStringAsFixed(1)} tokens/sec');
    print('Efficiency: ${tokensEvaluated}/${tokensPredicted} eval/pred ratio');
    
    // Send to your analytics service
    // FirebaseAnalytics.instance.logEvent(name: 'cactus_completion', parameters: {...});
  }

  static void trackError(String error, String context) {
    print('Cactus Error in $context: $error');
    // Report to crash analytics
    // FirebaseCrashlytics.instance.recordError(error, null);
  }
}

// Usage in completion
final stopwatch = Stopwatch()..start();
try {
  final result = await context.completion(params);
  stopwatch.stop();
  
  CactusAnalytics.trackCompletion(
    tokensPredicted: result.tokensPredicted,
    tokensEvaluated: result.tokensEvaluated,
    duration: stopwatch.elapsed,
    modelName: 'your-model-name',
  );
  
  return result.text;
} catch (e) {
  CactusAnalytics.trackError(e.toString(), 'text_completion');
  rethrow;
}
```

## Troubleshooting

### Common Issues

**Model Loading Fails**
```dart
// Check model file exists and is readable
Future<bool> validateModelFile(String modelPath) async {
  final file = File(modelPath);
  if (!await file.exists()) {
    print('Model file not found: $modelPath');
    return false;
  }
  
  final size = await file.length();
  if (size < 1024 * 1024) { // Less than 1MB is suspicious
    print('Model file seems too small: ${size} bytes');
    return false;
  }
  
  return true;
}
```

**Out of Memory Errors**
```dart
// Reduce context size and batch size
final params = CactusInitParams(
  modelPath: modelPath,
  contextSize: 2048,     // Reduce from 4096
  batchSize: 256,        // Reduce from 512
  gpuLayers: 0,          // Use CPU only if GPU memory limited
);
```

**Slow Generation**
```dart
// Optimize for speed
final params = CactusCompletionParams(
  messages: messages,
  maxPredictedTokens: 100,  // Reduce output length
  temperature: 0.3,         // Lower temperature = faster
  topK: 20,                // Reduce sampling space
  threads: 4,              // Use multiple CPU threads
);
```

**Multimodal Issues**
```dart
// Check multimodal initialization
if (!context.isMultimodalEnabled()) {
  throw Exception('Multimodal projector not loaded');
}

if (!context.supportsVision()) {
  throw Exception('Model does not support vision');
}

// Validate image file
Future<bool> validateImageFile(String imagePath) async {
  final file = File(imagePath);
  if (!await file.exists()) return false;
  
  // Check file extension
  final extension = path.extension(imagePath).toLowerCase();
  if (!['.jpg', '.jpeg', '.png', '.bmp'].contains(extension)) {
    return false;
  }
  
  return true;
}
```

### Debug Information

```dart
class CactusDebugger {
  static void printModelInfo(CactusContext context) {
    print('Model Info:');
    print('  Context Size: ${context.getNCtx()}');
    print('  Embedding Dims: ${context.getNEmbd()}');
    print('  Model Desc: ${context.getModelDesc()}');
    print('  Model Size: ${context.getModelSize()} bytes');
    print('  Model Params: ${context.getModelParams()}');
    print('  Multimodal: ${context.isMultimodalEnabled()}');
    print('  Vision Support: ${context.supportsVision()}');
    print('  Audio Support: ${context.supportsAudio()}');
  }

  static void printCompletionResult(CactusCompletionResult result) {
    print('Completion Result:');
    print('  Text Length: ${result.text.length}');
    print('  Tokens Predicted: ${result.tokensPredicted}');
    print('  Tokens Evaluated: ${result.tokensEvaluated}');
    print('  Truncated: ${result.truncated}');
    print('  Stopped EOS: ${result.stoppedEos}');
    print('  Stopped Word: ${result.stoppedWord}');
    print('  Stopped Limit: ${result.stoppedLimit}');
    if (result.stoppingWord.isNotEmpty) {
      print('  Stopping Word: "${result.stoppingWord}"');
    }
  }
}
```

### Performance Monitoring

```dart
class PerformanceMonitor {
  static const int _historySize = 10;
  static final List<Duration> _completionTimes = [];
  static final List<double> _tokensPerSecond = [];

  static void recordCompletion(Duration time, int tokens) {
    _completionTimes.add(time);
    _tokensPerSecond.add(tokens / time.inMilliseconds * 1000);
    
    if (_completionTimes.length > _historySize) {
      _completionTimes.removeAt(0);
      _tokensPerSecond.removeAt(0);
    }
  }

  static Map<String, double> getAverageMetrics() {
    if (_completionTimes.isEmpty) return {};
    
    final avgTime = _completionTimes
        .map((d) => d.inMilliseconds)
        .reduce((a, b) => a + b) / _completionTimes.length;
    
    final avgTokensPerSec = _tokensPerSecond
        .reduce((a, b) => a + b) / _tokensPerSecond.length;
    
    return {
      'avgCompletionTime': avgTime,
      'avgTokensPerSecond': avgTokensPerSec,
    };
  }
}
```

For more examples and advanced usage, check out the [example app](examples/flutter/) in the repository.

## Text-to-Speech (TTS)

Cactus provides powerful text-to-speech capabilities, allowing you to generate high-quality audio directly from text using a vocoder model.

> **Note**: To play the generated audio, you'll need an audio player package like [just_audio](https://pub.dev/packages/just_audio) or [audioplayers](https://pub.dev/packages/audioplayers).

### Setup TTS Model

```dart
Future<CactusContext?> setupTTS(String modelPath, String vocoderPath) async {
  try {
    // Initialize the base model (required for TTS prompt formatting)
    final context = await CactusContext.init(CactusInitParams(
      modelPath: modelPath,
      contextSize: 2048,
    ));

    // Initialize the vocoder
    await context.initVocoder(vocoderPath);
    
    if (context.isVocoderEnabled()) {
      print('TTS initialized successfully. Type: ${context.getTTSType()}');
      return context;
    } else {
      print('Failed to enable vocoder.');
      context.free();
      return null;
    }
  } catch (e) {
    print('Error setting up TTS: $e');
    return null;
  }
}

// Usage
// final ttsContext = await setupTTS('path/to/text-model.gguf', 'path/to/vocoder.gguf');
```

### Basic Speech Generation

```dart
import 'dart:typed_data';
import 'package:audioplayers/audioplayers.dart';

class SpeechGenerator {
  final CactusContext context;
  final AudioPlayer audioPlayer = AudioPlayer();

  SpeechGenerator(this.context);

  Future<void> speak(String text) async {
    if (!context.isVocoderEnabled()) {
      print('TTS not enabled.');
      return;
    }

    try {
      // 1. Format the text for audio completion
      final ttsPrompt = context.getFormattedAudioCompletion('{}', text);

      // 2. Get guide tokens to direct the generation
      final guideTokens = context.getAudioGuideTokens(text);
      
      // 3. Generate the audio tokens
      final result = await context.completion(CactusCompletionParams(
        messages: [ChatMessage(role: 'user', content: ttsPrompt)],
        maxPredictedTokens: 1024, // Adjust as needed for longer text
        temperature: 0.7,
        // Guide tokens can be set here if the model requires it
      ));
      
      // The raw audio tokens are part of the output but often need decoding
      // For many models, you may need to implement custom token filtering
      // and then use decodeAudioTokens.

      // A more direct approach for some models might be available.
      // The following is a placeholder for a complete audio generation flow.
      // This part of the API is illustrative.
      // For a complete example, refer to the native implementations.
      
      // Assuming the model directly outputs playable audio tokens
      // final audioData = context.decodeAudioTokens(audioTokens);
      // await playAudioData(audioData);

      print("Speech generation for '$text' requested. Decoding and playback logic depends on the specific TTS model integration.");

    } catch (e) {
      print('Error generating speech: $e');
    }
  }

  // Placeholder for playing raw audio data (e.g., PCM)
  Future<void> playAudioData(List<double> pcmData) async {
    // Conversion from List<double> to Uint8List for playback
    // This is highly dependent on the audio format and player library
    // final bytes = convertPcmToWav(pcmData);
    // await audioPlayer.play(BytesSource(bytes));
  }
}
```

### Complete TTS Example Widget

```dart
import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import 'package:audioplayers/audioplayers.dart';

class TTSDemoWidget extends StatefulWidget {
  final CactusContext context;
  
  TTSDemoWidget({required this.context});

  @override
  _TTSDemoWidgetState createState() => _TTSDemoWidgetState();
}

class _TTSDemoWidgetState extends State<TTSDemoWidget> {
  final TextEditingController _controller = TextEditingController(
    text: 'Hello, this is a test of the text-to-speech system in Flutter.'
  );
  final AudioPlayer _audioPlayer = AudioPlayer();
  bool _isGenerating = false;

  Future<void> _generateAndPlay() async {
    if (_controller.text.trim().isEmpty) return;

    setState(() => _isGenerating = true);

    try {
      // This is a simplified flow. Real implementation depends on the model.
      // Refer to native examples for precise token handling.
      final ttsPrompt = widget.context.getFormattedAudioCompletion('{}', _controller.text.trim());
      
      // For this example, we assume a direct-to-audio method is not yet exposed
      // and we cannot complete the full audio generation and playback loop
      // without more details on the specific TTS model's output format.
      
      // The API provides the building blocks:
      // 1. getFormattedAudioCompletion
      // 2. getAudioCompletionGuideTokens
      // 3. completion (to get audio tokens)
      // 4. decodeAudioTokens (to get raw PCM data)
      // 5. A player (like audioplayers) to play the raw data

      print("Audio generation flow initiated. Playback requires model-specific handling.");
      
      // In a real app, you would get audio data and play it:
      // final audioData = await getAudioData(...);
      // await _audioPlayer.play(BytesSource(audioData));

    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error generating speech: $e'))
      );
    } finally {
      setState(() => _isGenerating = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text('Text-to-Speech', style: Theme.of(context).textTheme.headlineSmall),
          SizedBox(height: 16),
          TextField(
            controller: _controller,
            decoration: InputDecoration(
              border: OutlineInputBorder(),
              labelText: 'Enter text to speak',
            ),
            maxLines: 3,
          ),
          SizedBox(height: 16),
          ElevatedButton.icon(
            onPressed: _isGenerating ? null : _generateAndPlay,
            icon: _isGenerating 
              ? SizedBox(height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
              : Icon(Icons.volume_up),
            label: Text(_isGenerating ? 'Generating...' : 'Speak'),
          ),
        ],
      ),
    );
  }
}
```

This documentation now includes the Text-to-Speech capabilities of the Cactus Flutter plugin. For more examples, check the [example app](examples/flutter/) in the repository.
