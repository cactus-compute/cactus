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

Handles are typed as `CactusModelT`, `CactusIndexT`, and `CactusStreamTranscribeT` (all `Pointer<Void>` aliases). All functions are top-level.

### Basic Completion

```dart
import 'cactus.dart';
import 'dart:convert';

final model = cactusInit('/path/to/model', null, false);
final messages = jsonEncode([{'role': 'user', 'content': 'What is the capital of France?'}]);
final resultJson = cactusComplete(model, messages, null, null, null);
final result = jsonDecode(resultJson);
print(result['response']);
cactusDestroy(model);
```

### Completion with Options and Streaming

```dart
final options = jsonEncode({'max_tokens': 256, 'temperature': 0.7});
final tokens = <String>[];

final resultJson = cactusComplete(model, messages, options, null, (token, _) {
  tokens.add(token);
  stdout.write(token);
});
```

### Audio Transcription

```dart
// From file
final result = cactusTranscribe(model, '/path/to/audio.wav', '', null, null, null);

// From PCM data (16 kHz mono)
final pcmData = Uint8List.fromList([...]);
final result = cactusTranscribe(model, null, null, null, null, pcmData);
```

### Streaming Transcription

```dart
final stream  = cactusStreamTranscribeStart(model, null);
final partial = cactusStreamTranscribeProcess(stream, audioChunk);
final final_  = cactusStreamTranscribeStop(stream);
```

### Embeddings

```dart
final embedding      = cactusEmbed(model, 'Hello, world!', true);   // Float32List
final imageEmbedding = cactusImageEmbed(model, '/path/to/image.jpg');
final audioEmbedding = cactusAudioEmbed(model, '/path/to/audio.wav');
```

### Tokenization

```dart
final tokens = cactusTokenize(model, 'Hello, world!');  // List<int>
final scores = cactusScoreWindow(model, tokens, 0, tokens.length, 512);
```

### VAD

```dart
final result = cactusVad(model, '/path/to/audio.wav', null, null);
```

### RAG

```dart
final result = cactusRagQuery(model, 'What is machine learning?', 5);
```

### Vector Index

```dart
final index = cactusIndexInit('/path/to/index', 384);

cactusIndexAdd(
  index,
  [1, 2],
  ['Document 1', 'Document 2'],
  [[0.1, 0.2], [0.3, 0.4]],
  null,
);

final resultsJson = cactusIndexQuery(index, [0.1, 0.2], null);
// JSON: {"results":[{"id":1,"score":0.99,...},...]}

cactusIndexDelete(index, [2]);
cactusIndexCompact(index);
cactusIndexDestroy(index);
```

## API Reference

All functions are top-level and mirror the C FFI directly. All functions throw `Exception` on failure.

### Types

```dart
typedef CactusModelT            = Pointer<Void>;
typedef CactusIndexT            = Pointer<Void>;
typedef CactusStreamTranscribeT = Pointer<Void>;
```

### Init / Lifecycle

```dart
CactusModelT cactusInit(String modelPath, String? corpusDir, bool cacheIndex)
void cactusDestroy(CactusModelT model)
void cactusReset(CactusModelT model)
void cactusStop(CactusModelT model)
String cactusGetLastError()
```

### Completion

```dart
String cactusComplete(
  CactusModelT model,
  String messagesJson,
  String? optionsJson,
  String? toolsJson,
  void Function(String token, int tokenId)? callback,
)
```

### Transcription

```dart
String cactusTranscribe(
  CactusModelT model,
  String? audioPath,
  String? prompt,
  String? optionsJson,
  void Function(String, int)? callback,
  Uint8List? pcmData,
)

CactusStreamTranscribeT cactusStreamTranscribeStart(CactusModelT model, String? optionsJson)
String cactusStreamTranscribeProcess(CactusStreamTranscribeT stream, Uint8List pcmData)
String cactusStreamTranscribeStop(CactusStreamTranscribeT stream)
```

### Embeddings

```dart
Float32List cactusEmbed(CactusModelT model, String text, bool normalize)
Float32List cactusImageEmbed(CactusModelT model, String imagePath)
Float32List cactusAudioEmbed(CactusModelT model, String audioPath)
```

### Tokenization / Scoring

```dart
List<int> cactusTokenize(CactusModelT model, String text)
String cactusScoreWindow(CactusModelT model, List<int> tokens, int start, int end, int context)
```

### VAD / RAG

```dart
String cactusVad(CactusModelT model, String? audioPath, String? optionsJson, Uint8List? pcmData)
String cactusRagQuery(CactusModelT model, String query, int topK)
```

### Vector Index

```dart
CactusIndexT cactusIndexInit(String indexDir, int embeddingDim)
void cactusIndexDestroy(CactusIndexT index)
int cactusIndexAdd(CactusIndexT index, List<int> ids, List<String> documents, List<List<double>> embeddings, List<String>? metadatas)
int cactusIndexDelete(CactusIndexT index, List<int> ids)
String cactusIndexGet(CactusIndexT index, List<int> ids)
String cactusIndexQuery(CactusIndexT index, List<double> embedding, String? optionsJson)
int cactusIndexCompact(CactusIndexT index)
```

### Telemetry

```dart
void cactusSetTelemetryEnvironment(String cacheDir)
void cactusSetAppId(String appId)
void cactusTelemetryFlush()
void cactusTelemetryShutdown()
```

All functions throw a standard `Exception` on failure.

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
  final modelFile = File('${dir.path}/model');

  if (!await modelFile.exists()) {
    final data = await rootBundle.load('assets/model');
    await modelFile.writeAsBytes(data.buffer.asUint8List());
  }

  return modelFile.path;
}
```

### iOS/macOS

Add model to bundle and access via path:

```dart
import 'package:path_provider/path_provider.dart';

final path = '${Directory.current.path}/model';
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
