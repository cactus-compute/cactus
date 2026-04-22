---
title: "Cactus Flutter SDK"
description: "Flutter bindings for Cactus on-device AI inference. Run LLMs, vision models, and speech models on iOS, macOS, and Android with dart:ffi."
keywords: ["Flutter SDK", "dart FFI", "on-device AI", "mobile inference", "iOS", "Android", "macOS"]
---

# Cactus for Flutter

Run AI models on-device with dart:ffi direct bindings for iOS, macOS, and Android.

> **Model weights:** Pre-converted weights for all supported models at [huggingface.co/Cactus-Compute](https://huggingface.co/Cactus-Compute).

## Building

<!-- --8<-- [start:install] -->
```bash
git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup
cactus build --flutter
```

Build output:

| File | Platform |
|------|----------|
| `libcactus.so` | Android (arm64-v8a) |
| `cactus-ios.xcframework` | iOS |
| `cactus-macos.xcframework` | macOS |
<!-- --8<-- [end:install] -->

See the main [README.md](../README.md) for how to use CLI & download weights

## Integration

<!-- --8<-- [start:integration] -->
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
<!-- --8<-- [end:integration] -->

## Usage

Handles are typed as `CactusModelT`, `CactusIndexT`, and `CactusStreamTranscribeT` (all `Pointer<Void>` aliases). All functions are top-level.

<!-- --8<-- [start:example] -->
### Basic Completion

```dart
import 'cactus.dart';

final model = cactusInit('/path/to/model', null, false);
final messages = '[{"role":"user","content":"What is the capital of France?"}]';
final resultJson = cactusComplete(model, messages, null, null, null);
print(resultJson);
cactusDestroy(model);
```
<!-- --8<-- [end:example] -->

For vision models (LFM2-VL, LFM2.5-VL, Gemma4, Qwen3.5), add `"images": ["path/to/image.png"]` to any message. For audio models (Gemma4), add `"audio": ["path/to/audio.wav"]`. See [Engine API](/docs/cactus_engine.md) for details.

### Completion with Options and Streaming

```dart
import 'cactus.dart';
import 'dart:io';

final options = '{"max_tokens":256,"temperature":0.7}';

final resultJson = cactusComplete(model, messages, options, null, (token, tokenId) {
  stdout.write(token);
});
print(resultJson);
```

### Prefill

Pre-processes input text and populates the KV cache without generating output tokens. This reduces latency for subsequent calls to `cactusComplete`.

```dart
String cactusPrefill(
  CactusModelT model,
  String messagesJson,
  String? optionsJson,
  String? toolsJson,
)
```

```dart
final tools = '[{"type":"function","function":{"name":"get_weather","description":"Get weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]';

final messages = '[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"What is the weather in Paris?"}]';

final resultJson = cactusPrefill(model, messages, null, tools);

final completionMessages = '[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"What is the weather in Paris?"},{"role":"user","content":"What about SF?"}]';

final completion = cactusComplete(model, completionMessages, null, tools, null);
```

**Response format:**
```json
{
    "success": true,
    "error": null,
    "prefill_tokens": 25,
    "prefill_tps": 166.1,
    "total_time_ms": 150.5,
    "ram_usage_mb": 245.67
}
```

### Audio Transcription

```dart
import 'cactus.dart';
import 'dart:typed_data';

// From file
final resultJson = cactusTranscribe(model, '/path/to/audio.wav', null, null, null, null);
print(resultJson);

// From PCM data (16 kHz mono)
final pcmData = Uint8List.fromList([...]);
final resultJson2 = cactusTranscribe(model, null, null, null, null, pcmData);
print(resultJson2);
```

`segments` contains timestamps (seconds): phrase-level for Whisper, word-level for Parakeet TDT, one segment per transcription window for Parakeet CTC and Moonshine (consecutive VAD speech regions up to 30s).

```dart
import 'dart:convert';

final result = jsonDecode(resultJson) as Map<String, dynamic>;
for (final seg in result['segments'] as List) {
  print('[${seg['start']}s - ${seg['end']}s] ${seg['text']}');
}
```

**Custom vocabulary** biases the decoder toward domain-specific words (supported for Whisper and Moonshine models). Pass `custom_vocabulary` and `vocabulary_boost` in the options JSON:

```dart
final options = '{"custom_vocabulary": ["Omeprazole", "HIPAA", "Cactus"], "vocabulary_boost": 3.0}';
final result = cactusTranscribe(model, '/path/to/audio.wav', '', options, null, null);
```

### Streaming Transcription

```dart
import 'cactus.dart';
import 'dart:typed_data';

final stream = cactusStreamTranscribeStart(model, null);

final Uint8List audioChunk = ...;
final partialJson = cactusStreamTranscribeProcess(stream, audioChunk);
print(partialJson);

final finalJson = cactusStreamTranscribeStop(stream);
print(finalJson);
```

Streaming also accepts `custom_vocabulary` in the options passed to `cactusStreamTranscribeStart`. The bias is applied for the lifetime of the stream session.

### Embeddings

```dart
import 'cactus.dart';
import 'dart:typed_data';

final Float32List embedding      = cactusEmbed(model, 'Hello, world!', true);
final Float32List imageEmbedding = cactusImageEmbed(model, '/path/to/image.jpg');
final Float32List audioEmbedding = cactusAudioEmbed(model, '/path/to/audio.wav');
```

### Tokenization

```dart
import 'cactus.dart';

final List<int> tokens = cactusTokenize(model, 'Hello, world!');
final String scores = cactusScoreWindow(model, tokens, 0, tokens.length, 512);
```

### Language Detection

```dart
import 'cactus.dart';
import 'dart:typed_data';

// From file
final resultJson = cactusDetectLanguage(model, '/path/to/audio.wav', null, null);
print(resultJson);

// From PCM data (16 kHz mono)
final Uint8List pcmData = ...;
final resultJson2 = cactusDetectLanguage(model, null, null, pcmData);
print(resultJson2);
```

### VAD

```dart
import 'cactus.dart';

final String vadJson = cactusVad(model, '/path/to/audio.wav', null, null);
print(vadJson);
```

### Diarize

```dart
import 'cactus.dart';

final String diarizeJson = cactusDiarize(model, '/path/to/audio.wav', null, null);
print(diarizeJson);
```

Options (all optional):
- `step_ms` (int, default 1000) — sliding window stride in milliseconds
- `threshold` (float) — zero out per-speaker scores below this value
- `num_speakers` (int) — keep only the N most active speakers
- `min_speakers` / `max_speakers` (int) — speaker count bounds
- `raw_powerset` (bool, default false) — return raw 7-class powerset scores instead of 3-speaker probabilities

### Embed Speaker

```dart
import 'cactus.dart';

final String embedJson = cactusEmbedSpeaker(model, '/path/to/audio.wav', null, null);
print(embedJson);

// With diarization mask for speaker-specific embedding
final String embedJson = cactusEmbedSpeaker(model, '/path/to/audio.wav', null, null, maskWeights);
```

Returns a 256-dimensional speaker embedding. When `maskWeights` (a per-frame weight array from diarization) is provided, the embedding is extracted using weighted stats pooling for speaker-specific embeddings.

### RAG

```dart
import 'cactus.dart';

final String result = cactusRagQuery(model, 'What is machine learning?', 5);
print(result);
```

### Vector Index

```dart
import 'cactus.dart';

final embDim = 4;
final index = cactusIndexInit('/path/to/index', embDim);

cactusIndexAdd(
  index,
  [1, 2],
  ['Document 1', 'Document 2'],
  [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
  null,
);

final resultsJson = cactusIndexQuery(index, [0.1, 0.2, 0.3, 0.4], null);
final getJson = cactusIndexGet(index, [1, 2]);

cactusIndexDelete(index, [2]);
cactusIndexCompact(index);
cactusIndexDestroy(index);
```

## API Reference

All functions are top-level and mirror the C FFI directly. Functions that return a value throw `Exception` on failure;

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

### Prefill

```dart
String cactusPrefill(
  CactusModelT model,
  String messagesJson,
  String? optionsJson,
  String? toolsJson,
  {Uint8List? pcmData}
)
```

### Completion

```dart
String cactusComplete(
  CactusModelT model,
  String messagesJson,
  String? optionsJson,
  String? toolsJson,
  void Function(String token, int tokenId)? callback,
  {Uint8List? pcmData}
)
```

### Transcription

```dart
String cactusTranscribe(
  CactusModelT model,
  String? audioPath,
  String? prompt,
  String? optionsJson,
  void Function(String token, int tokenId)? callback,
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

### Detect Language

```dart
String cactusDetectLanguage(CactusModelT model, String? audioPath, String? optionsJson, Uint8List? pcmData)
```

### VAD

```dart
String cactusVad(CactusModelT model, String? audioPath, String? optionsJson, Uint8List? pcmData)
```

### Diarize

```dart
String cactusDiarize(CactusModelT model, String? audioPath, String? optionsJson, Uint8List? pcmData)
```

### Embed Speaker

```dart
String cactusEmbedSpeaker(CactusModelT model, String? audioPath, String? optionsJson, Uint8List? pcmData, [Float32List? maskWeights])
```

### RAG

```dart
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

### Logging

```dart
void cactusLogSetLevel(int level)  // 0=DEBUG 1=INFO 2=WARN 3=ERROR 4=NONE
void cactusLogSetCallback(void Function(int level, String component, String message)? onLog)
```

### Telemetry

```dart
void cactusSetTelemetryEnvironment(String cacheLocation)
void cactusSetAppId(String appId)
void cactusTelemetryFlush()
void cactusTelemetryShutdown()
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
import 'dart:io';

final path = '${Directory.current.path}/model';
```

## Requirements

- Flutter 3.0+
- Dart 2.17+
- iOS 13.0+ / macOS 13.0+
- Android API 21+ / arm64-v8a

## See Also

- [Cactus Engine API](/docs/cactus_engine.md) — Full C API reference underlying the Flutter bindings
- [Cactus Index API](/docs/cactus_index.md) — Vector database API for RAG applications
- [Fine-tuning Guide](/docs/finetuning.md) — Deploy custom fine-tunes to mobile
- [Swift SDK](/apple/) — Native Swift alternative for Apple platforms
- [Kotlin/Android SDK](/android/) — Native Kotlin alternative for Android
