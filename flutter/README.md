---
title: "Cactus Flutter SDK"
description: "Flutter bindings for Cactus on-device AI inference. Run LLMs, vision models, and speech models on iOS, macOS, and Android with dart:ffi."
keywords: ["Flutter SDK", "dart FFI", "on-device AI", "mobile inference", "iOS", "Android", "macOS"]
---

# Cactus for Flutter

Run AI models on-device with dart:ffi direct bindings for iOS, macOS, and Android.

## Building

```bash
cactus build --flutter
```

Build output:

| File | Platform |
|------|----------|
| `libcactus.so` | Android (arm64-v8a) |
| `cactus-ios.xcframework` | iOS |
| `cactus-macos.xcframework` | macOS |

see the main [README.md](../README.md) for how to use CLI & download weight

## Integration

### Android

1. Copy `libcactus.so` to `android/app/src/main/jniLibs/arm64-v8a/`
2. Copy `cactus.dart` to your `lib/` folder

### iOS

1. Copy `cactus-ios.xcframework` to your `ios/` folder
2. Open `ios/Runner.xcworkspace` in Xcode
3. Drag the xcframework into the project
4. In Runner target > General > "Frameworks, Libraries, and Embedded Content", set to "Embed & Sign"
5. Copy `cactus.dart` to your `lib/` folder

### macOS

1. Copy `cactus-macos.xcframework` to your `macos/` folder
2. Open `macos/Runner.xcworkspace` in Xcode
3. Drag the xcframework into the project
4. In Runner target > General > "Frameworks, Libraries, and Embedded Content", set to "Embed & Sign"
5. Copy `cactus.dart` to your `lib/` folder

## Usage

### Basic Completion

```dart
import 'cactus.dart';

final model = Cactus.create('/path/to/model.gguf');
final result = model.complete('What is the capital of France?');
print(result.text);
model.dispose();
```

### Chat Messages

```dart
final model = Cactus.create(modelPath);
final result = model.completeMessages([
  Message.system('You are a helpful assistant.'),
  Message.user('What is 2 + 2?'),
]);
print(result.text);
model.dispose();
```

### Completion Options

```dart
final options = CompletionOptions(
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  maxTokens: 256,
  stopSequences: ['\n\n'],
);

final result = model.complete('Write a haiku:', options: options);
```

### Audio Transcription

```dart
// From file
final result = model.transcribe('/path/to/audio.wav');

// From PCM data (16kHz mono)
final pcmData = Uint8List.fromList([...]); // Your PCM bytes
final result = model.transcribePcm(pcmData);
```

### Streaming Transcription

```dart
final stream = model.createStreamTranscriber();
stream.insert(audioChunk1);
stream.insert(audioChunk2);

final partial = stream.process();
print('Partial: ${partial.text}');

final finalResult = stream.finalize();
print('Final: ${finalResult.text}');

stream.dispose();
```

### Embeddings

```dart
final embedding = model.embed('Hello, world!');
final imageEmbedding = model.imageEmbed('/path/to/image.jpg');
final audioEmbedding = model.audioEmbed('/path/to/audio.wav');
```

### Tokenization

```dart
final tokens = model.tokenize('Hello, world!');
final scores = model.scoreWindow(tokens, 0, tokens.length, 512);
```

### RAG (Retrieval-Augmented Generation)

```dart
final model = Cactus.create(
  '/path/to/model.gguf',
  corpusDir: '/path/to/documents',
);

final result = model.complete('What does the documentation say about X?');
```

### Vector Index

```dart
final index = CactusIndex.create('/path/to/index', embeddingDim: 384);

index.add(
  ids: [1, 2],
  documents: ['Document 1', 'Document 2'],
  embeddings: [
    model.embed('Document 1'),
    model.embed('Document 2'),
  ],
);

final results = index.query(model.embed('search query'), topK: 5);
for (final r in results) {
  print('ID: ${r.id}, Score: ${r.score}');
}

index.dispose();
```

## API Reference

### Cactus

```dart
class Cactus {
  static Cactus create(String modelPath, {String? corpusDir});

  CompletionResult complete(String prompt, {CompletionOptions options, void Function(String, int)? onToken});
  CompletionResult completeMessages(List<Message> messages, {CompletionOptions options, List<Map<String, dynamic>>? tools, void Function(String, int)? onToken});

  TranscriptionResult transcribe(String audioPath, {String? prompt, TranscriptionOptions options});
  TranscriptionResult transcribePcm(Uint8List pcmData, {String? prompt, TranscriptionOptions options});

  List<double> embed(String text, {bool normalize = true});
  List<double> imageEmbed(String imagePath);
  List<double> audioEmbed(String audioPath);
  String ragQuery(String query, {int topK = 5});

  List<int> tokenize(String text);
  String scoreWindow(List<int> tokens, int start, int end, int context);
  StreamTranscriber createStreamTranscriber();

  void reset();
  void stop();
  void dispose();

  static String getLastError();
}
```

### Message

```dart
class Message {
  static Message system(String content);
  static Message user(String content);
  static Message assistant(String content);
}
```

### CompletionOptions

```dart
class CompletionOptions {
  final double temperature; 
  final double topP;  
  final int topK;   
  final int maxTokens;       
  final List<String> stopSequences;
  final double confidenceThreshold;

  static const defaultOptions;
}
```

### CompletionResult

```dart
class CompletionResult {
  final String text;
  final List<Map<String, dynamic>>? functionCalls;
  final int promptTokens;
  final int completionTokens;
  final double timeToFirstToken;
  final double totalTime;
  final double prefillTokensPerSecond;
  final double decodeTokensPerSecond;
  final double confidence;
  final bool needsCloudHandoff;
}
```

### TranscriptionResult

```dart
class TranscriptionResult {
  final String text;
  final List<Map<String, dynamic>>? segments;
  final double totalTime;
}
```

### StreamTranscriber

```dart
class StreamTranscriber {
  void insert(Uint8List pcmData);
  TranscriptionResult process({String? language});
  TranscriptionResult finalize();
  void dispose();
}
```

### CactusIndex

```dart
class CactusIndex {
  static CactusIndex create(String indexDir, {required int embeddingDim});

  void add({required List<int> ids, required List<String> documents, required List<List<double>> embeddings, List<String>? metadatas});
  void delete(List<int> ids);
  List<IndexResult> query(List<double> embedding, {int topK = 5});
  void compact();
  void dispose();
}

class IndexResult {
  final int id;
  final double score;
}
```

## Bundling Model Weights

Models must be accessible via file path at runtime.

### Android

Copy from assets to internal storage on first launch:

```dart
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

Future<String> getModelPath() async {
  final dir = await getApplicationDocumentsDirectory();
  final modelFile = File('${dir.path}/model.gguf');

  if (!await modelFile.exists()) {
    final data = await rootBundle.load('assets/model.gguf');
    await modelFile.writeAsBytes(data.buffer.asUint8List());
  }

  return modelFile.path;
}
```

### iOS/macOS

Add model to bundle and access via path:

```dart
import 'package:path_provider/path_provider.dart';

final path = '${Directory.current.path}/model.gguf';
```

## Requirements

- Flutter 3.0+
- Dart 2.17+
- iOS 14.0+ / macOS 13.0+
- Android API 24+ / arm64-v8a

## See Also

- [Cactus Engine API](/docs/cactus_engine.md) — Full C API reference underlying the Flutter bindings
- [Cactus Index API](/docs/cactus_index.md) — Vector database API for RAG applications
- [Fine-tuning Guide](/docs/finetuning.md) — Deploy custom fine-tunes to mobile
- [Swift SDK](/apple/) — Native Swift alternative for Apple platforms
- [Kotlin/Android SDK](/android/) — Native Kotlin alternative for Android
