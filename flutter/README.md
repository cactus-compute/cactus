# Cactus Flutter

Run LLMs, VLMs, and TTS models directly on mobile devices.

## Installation

```yaml
dependencies:
  cactus: ^0.2.4
```

**Requirements:** iOS 12.0+, Android API 24+, Flutter 3.3.0+

## Quick Start

### Text Completion

```dart
import 'package:cactus/cactus.dart';

  final lm = await CactusLM.init(
  modelUrl: 'https://huggingface.co/model.gguf',
    contextSize: 2048,
  );

  final result = await lm.completion([
  ChatMessage(role: 'user', content: 'Hello!')
  ], maxTokens: 100, temperature: 0.7);

print(result.text);
  lm.dispose();
```

### Streaming Chat

```dart
import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';

class ChatScreen extends StatefulWidget {
  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  CactusLM? _lm;
  List<ChatMessage> _messages = [];
  final _controller = TextEditingController();
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _initModel();
  }

  Future<void> _initModel() async {
      _lm = await CactusLM.init(
      modelUrl: 'https://huggingface.co/model.gguf',
      contextSize: 2048,
        onProgress: (progress, status, isError) {
        print('$status ${progress != null ? '${(progress * 100).toInt()}%' : ''}');
        },
      );
      setState(() => _isLoading = false);
  }

  Future<void> _sendMessage() async {
    if (_lm == null || _controller.text.trim().isEmpty) return;

    final userMsg = ChatMessage(role: 'user', content: _controller.text.trim());
    setState(() {
      _messages.add(userMsg);
      _messages.add(ChatMessage(role: 'assistant', content: ''));
    });
    _controller.clear();

    String response = '';
    await _lm!.completion(
      _messages.where((m) => m.content.isNotEmpty).toList(),
      maxTokens: 200,
        temperature: 0.7,
        onToken: (token) {
        response += token;
          setState(() {
          _messages.last = ChatMessage(role: 'assistant', content: response);
        });
        return true;
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) return Scaffold(body: Center(child: CircularProgressIndicator()));

    return Scaffold(
      appBar: AppBar(title: Text('Chat')),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final msg = _messages[index];
                return ListTile(
                  title: Text(msg.content),
                  subtitle: Text(msg.role),
                );
              },
            ),
          ),
          Padding(
            padding: EdgeInsets.all(8),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: InputDecoration(hintText: 'Message...'),
                    onSubmitted: (_) => _sendMessage(),
                  ),
                ),
                IconButton(onPressed: _sendMessage, icon: Icon(Icons.send)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _lm?.dispose();
    super.dispose();
  }
}

## Core APIs

### CactusLM (Language Model)

```dart
final lm = await CactusLM.init(
  modelUrl: 'https://huggingface.co/model.gguf',
  contextSize: 2048,
  gpuLayers: 0, // GPU layers (0 = CPU only)
  generateEmbeddings: true, // Enable embeddings
);

final result = await lm.completion([
  ChatMessage(role: 'system', content: 'You are helpful.'),
  ChatMessage(role: 'user', content: 'What is AI?'),
], maxTokens: 200, temperature: 0.7);

final embedding = await lm.embedding('Your text here');
lm.dispose();
```

### CactusVLM (Vision Language Model)

```dart
final vlm = await CactusVLM.init(
  modelUrl: 'https://huggingface.co/model.gguf',
  mmprojUrl: 'https://huggingface.co/mmproj.gguf',
  contextSize: 2048,
);

final result = await vlm.completion([
  ChatMessage(role: 'user', content: 'Describe this image')
], imagePaths: ['/path/to/image.jpg'], maxTokens: 200);

vlm.dispose();
```

### CactusTTS (Text-to-Speech)

```dart
final tts = await CactusTTS.init(
  modelUrl: 'https://huggingface.co/tts-model.gguf',
  contextSize: 1024,
);

final result = await tts.generate(
  'Hello world!',
  maxTokens: 256,
  temperature: 0.7,
);

tts.dispose();
```

### CactusAgent (Agentic Tool Calling)

```dart
final agent = await CactusAgent.init(
  modelUrl: 'https://huggingface.co/model.gguf',
  contextSize: 2048,
  gpuLayers: 0, // GPU layers (0 = CPU only)
  generateEmbeddings: true, // Enable embeddings
);

agent!.addTool(
    'get_weather',
    WeatherTool(),
    'Get current weather information for a location',
    {
      'location': Parameter(
        type: 'string',
        description: 'The location to get weather for',
        required: true,
      ),
    },
  );

final result = await agent.completionWithTools([
  ChatMessage(role: 'system', content: 'You are helpful.'),
  ChatMessage(role: 'user', content: 'What is AI?'),
], maxTokens: 200, temperature: 0.7);

agent.dispose();
```

### Tools
```
class WeatherTool extends ToolExecutor {
  @override
  Future<dynamic> execute(Map<String, dynamic> args) async {
    final location = args['location'] as String? ?? 'unknown';
    return 'The weather in $location is sunny, 72°F';
  }
}
```

## Advanced Usage

### Embeddings & Similarity

```dart
final lm = await CactusLM.init(
  modelUrl: 'https://huggingface.co/model.gguf',
  generateEmbeddings: true,
);

final embedding1 = await lm.embedding('machine learning');
final embedding2 = await lm.embedding('artificial intelligence');

double cosineSimilarity(List<double> a, List<double> b) {
  double dot = 0, normA = 0, normB = 0;
  for (int i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (sqrt(normA) * sqrt(normB));
}

final similarity = cosineSimilarity(embedding1, embedding2);
print('Similarity: $similarity');
```

### Cloud Fallback

```dart
final lm = await CactusLM.init(
  modelUrl: 'https://huggingface.co/model.gguf',
  cactusToken: 'your_enterprise_token',
);

// Try local first, fallback to cloud if local fails
final embedding = await lm.embedding('text', mode: 'localfirst');

// Vision models also support cloud fallback
final vlm = await CactusVLM.init(
  modelUrl: 'https://huggingface.co/model.gguf',
  mmprojUrl: 'https://huggingface.co/mmproj.gguf',
  cactusToken: 'your_enterprise_token',
);

final result = await vlm.completion([
  ChatMessage(role: 'user', content: 'Describe image')
], imagePaths: ['/path/to/image.jpg'], mode: 'localfirst');
```

### Memory Management

```dart
class ModelManager {
  CactusLM? _lm;
  
  Future<void> initialize() async {
    _lm = await CactusLM.init(modelUrl: 'https://huggingface.co/model.gguf');
  }
  
  Future<String> complete(String prompt) async {
    final result = await _lm!.completion([
      ChatMessage(role: 'user', content: prompt)
    ], maxTokens: 100);
    return result.text;
  }

  void clearContext() => _lm?.rewind();
  void dispose() => _lm?.dispose();
}
```

## Vision (Multimodal)

### Image Analysis

```dart
final vlm = await CactusVLM.init(
  modelUrl: 'https://huggingface.co/model.gguf',
  mmprojUrl: 'https://huggingface.co/mmproj.gguf',
);

    final result = await vlm.completion([
  ChatMessage(role: 'user', content: 'What do you see?')
], imagePaths: ['/path/to/image.jpg'], maxTokens: 200);

print(result.text);
vlm.dispose();
```

### Vision Chat with Image Picker

```dart
import 'package:image_picker/image_picker.dart';

class VisionChat extends StatefulWidget {
  @override
  State<VisionChat> createState() => _VisionChatState();
}

class _VisionChatState extends State<VisionChat> {
  CactusVLM? _vlm;
  String? _imagePath;
  String _response = '';

  @override
  void initState() {
    super.initState();
    _initVLM();
  }

  Future<void> _initVLM() async {
      _vlm = await CactusVLM.init(
      modelUrl: 'https://huggingface.co/model.gguf',
      mmprojUrl: 'https://huggingface.co/mmproj.gguf',
    );
  }

  Future<void> _pickImage() async {
    final image = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (image != null) setState(() => _imagePath = image.path);
  }

  Future<void> _analyzeImage() async {
    if (_vlm == null || _imagePath == null) return;

    final result = await _vlm!.completion([
      ChatMessage(role: 'user', content: 'Describe this image')
    ], imagePaths: [_imagePath!], maxTokens: 200);

    setState(() => _response = result.text);
  }

  @override
  Widget build(BuildContext context) {
      return Scaffold(
      appBar: AppBar(title: Text('Vision Chat')),
      body: Column(
        children: [
          if (_imagePath != null) 
            Image.file(File(_imagePath!), height: 200),
          ElevatedButton(onPressed: _pickImage, child: Text('Pick Image')),
          ElevatedButton(onPressed: _analyzeImage, child: Text('Analyze')),
          Expanded(child: Text(_response)),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _vlm?.dispose();
    super.dispose();
  }
}
```

## Error Handling & Performance

### Error Handling

```dart
try {
  final lm = await CactusLM.init(
    modelUrl: 'https://huggingface.co/model.gguf',
    onProgress: (progress, status, isError) {
      print('Status: $status');
      if (isError) print('Error: $status');
    },
  );
  
  final result = await lm.completion([
    ChatMessage(role: 'user', content: 'Hello')
  ], maxTokens: 100);
  
} on CactusException catch (e) {
  print('Cactus error: ${e.message}');
} catch (e) {
  print('General error: $e');
}
```

### Performance Optimization

```dart
// For better performance
final lm = await CactusLM.init(
  modelUrl: 'https://huggingface.co/model.gguf',
  contextSize: 2048,     // Smaller context = faster
  gpuLayers: 20,         // Use GPU acceleration
  threads: 4,            // Optimize for your device
);

// Reduce output length for speed
final result = await lm.completion(messages, 
  maxTokens: 100,        // Shorter responses
  temperature: 0.3,      // Lower temp = faster
);
```



## API Reference

### CactusLM

#### `CactusLM.init()`

Initialize a language model for text completion and embeddings.

```dart
static Future<CactusLM> init({
  required String modelUrl,        // HuggingFace model URL or local path
  String? modelFilename,           // Custom filename for cached model
  String? chatTemplate,            // Custom chat template (Jinja2 format)
  int contextSize = 2048,          // Context window size in tokens
  int gpuLayers = 0,              // Number of layers to run on GPU (0 = CPU only)
  int threads = 4,                // Number of CPU threads to use
  bool generateEmbeddings = false, // Enable embedding generation
  CactusProgressCallback? onProgress, // Download/init progress callback
  String? cactusToken,            // Enterprise token for cloud features
})
```

**Parameters:**
- `modelUrl`: URL to GGUF model on HuggingFace or local file path
- `modelFilename`: Override the cached filename (useful for versioning)
- `chatTemplate`: Custom Jinja2 chat template for formatting conversations
- `contextSize`: Maximum tokens the model can process (affects memory usage)
- `gpuLayers`: Higher values = faster inference but more GPU memory usage
- `threads`: Optimize based on your device's CPU cores
- `generateEmbeddings`: Must be `true` to use `embedding()` method
- `onProgress`: Callback for download progress and initialization status
- `cactusToken`: Required for cloud fallback features

#### `completion()`

Generate text completions with optional streaming.

```dart
Future<CactusCompletionResult> completion(
  List<ChatMessage> messages, {
  int maxTokens = 256,            // Maximum tokens to generate
  double? temperature,            // Randomness (0.0-2.0, default model-specific)
  int? topK,                     // Limit vocabulary to top K tokens
  double? topP,                  // Nucleus sampling threshold
  List<String>? stopSequences,   // Stop generation at these strings
  CactusTokenCallback? onToken,  // Stream tokens as they're generated
})
```

**Parameters:**
- `messages`: Conversation history with roles ('system', 'user', 'assistant')
- `maxTokens`: Higher values = longer responses but slower generation
- `temperature`: 0.1 = focused, 0.7 = balanced, 1.5+ = creative
- `topK`: Lower values (10-50) = more focused, higher (100+) = more diverse
- `topP`: 0.8-0.95 typical range, lower = more focused
- `stopSequences`: Model stops when it generates any of these strings
- `onToken`: Called for each generated token, return `false` to stop

#### `embedding()`

Generate text embeddings for semantic similarity.

```dart
Future<List<double>> embedding(
  String text, {
  String mode = "local",         // "local", "remote", "localfirst", "remotefirst"
})
```

**Parameters:**
- `text`: Input text to embed
- `mode`: 
  - `"local"`: Only use device model
  - `"remote"`: Only use cloud API  
  - `"localfirst"`: Try local, fallback to cloud if it fails
  - `"remotefirst"`: Try cloud, fallback to local if it fails

#### Other Methods

```dart
Future<List<int>> tokenize(String text)           // Convert text to token IDs
Future<String> detokenize(List<int> tokens)       // Convert token IDs to text
Future<void> rewind()                             // Clear conversation context
Future<void> stopCompletion()                     // Cancel ongoing generation
void dispose()                                    // Clean up resources
```

### CactusVLM

#### `CactusVLM.init()`

Initialize a vision-language model for image and text processing.

```dart
static Future<CactusVLM> init({
  required String modelUrl,       // Main model URL
  required String mmprojUrl,      // Vision projection model URL
  String? modelFilename,          // Custom filename for main model
  String? mmprojFilename,         // Custom filename for vision model
  String? chatTemplate,           // Custom chat template
  int contextSize = 2048,         // Context window size
  int gpuLayers = 0,             // GPU layers for acceleration
  int threads = 4,               // CPU threads
  CactusProgressCallback? onProgress, // Progress callback
  String? cactusToken,           // Enterprise token
})
```

**Parameters:**
- `modelUrl`: URL to main vision-language model (GGUF format)
- `mmprojUrl`: URL to corresponding vision projection model
- All other parameters same as `CactusLM.init()`

#### `completion()`

Generate responses from text and/or images.

```dart
Future<CactusCompletionResult> completion(
  List<ChatMessage> messages, {
  List<String> imagePaths = const [], // Local image file paths
  int maxTokens = 256,                // Maximum tokens to generate
  double? temperature,                // Generation randomness
  int? topK,                         // Vocabulary limiting
  double? topP,                      // Nucleus sampling
  List<String>? stopSequences,       // Stop strings
  CactusTokenCallback? onToken,      // Token streaming
  String mode = "local",             // Processing mode
})
```

**Parameters:**
- `imagePaths`: List of local image file paths to include in the conversation
- `mode`: Same options as `CactusLM.embedding()` for cloud fallback
- All other parameters same as `CactusLM.completion()`

#### Vision-Specific Methods

```dart
Future<bool> get supportsVision        // Check if vision is enabled
Future<bool> get supportsAudio         // Check if audio is supported  
Future<bool> get isMultimodalEnabled   // Check if multimodal features work
```

### CactusTTS

#### `CactusTTS.init()`

Initialize a text-to-speech model.

```dart
static Future<CactusTTS> init({
  required String modelUrl,       // TTS model URL
  String? modelFilename,          // Custom cached filename
  int contextSize = 2048,         // Context size
  int gpuLayers = 0,             // GPU acceleration
  int threads = 4,               // CPU threads
  CactusProgressCallback? onProgress, // Progress callback
})
```

#### `generate()`

Generate speech from text.

```dart
Future<CactusCompletionResult> generate(
  String text, {
  int maxTokens = 256,            // Maximum tokens for speech generation
  double? temperature,            // Generation randomness
  int? topK,                     // Vocabulary limiting
  double? topP,                  // Nucleus sampling
  List<String>? stopSequences,   // Stop sequences
  CactusTokenCallback? onToken,  // Token streaming
})
```

### Types

#### `ChatMessage`

```dart
class ChatMessage {
  final String role;        // 'system', 'user', or 'assistant'
  final String content;     // Message text
}
```

#### `CactusCompletionResult`

```dart
class CactusCompletionResult {
  final String text;           // Generated text
  final int tokensPredicted;   // Number of tokens generated
  final int tokensEvaluated;   // Number of input tokens processed
  final bool truncated;        // Whether output was truncated
  final bool stoppedEos;       // Stopped at end-of-sequence token
  final bool stoppedWord;      // Stopped at stop sequence
  final bool stoppedLimit;     // Stopped at token limit
  final String stoppingWord;   // Which stop sequence triggered (if any)
}
```

#### `LoraAdapterInfo`

```dart
class LoraAdapterInfo {
  final String path;        // Path to LoRA adapter file
  final double scale;       // Adapter strength (0.0-1.0)
}
```

#### Callbacks

```dart
typedef CactusTokenCallback = bool Function(String token);
typedef CactusProgressCallback = void Function(double? progress, String statusMessage, bool isError);
```
