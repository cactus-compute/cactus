import 'dart:ffi';
import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

typedef CactusModelT = Pointer<Void>;
typedef CactusIndexT = Pointer<Void>;
typedef CactusStreamTranscribeT = Pointer<Void>;

typedef TokenCallbackNative = Void Function(
    Pointer<Utf8> token, Uint32 tokenId, Pointer<Void> userData);
typedef TokenCallbackDart = void Function(
    Pointer<Utf8> token, int tokenId, Pointer<Void> userData);

typedef CactusInitNative = CactusModelT Function(
    Pointer<Utf8> modelPath, Pointer<Utf8> corpusDir);
typedef CactusDestroyNative = Void Function(CactusModelT model);
typedef CactusResetNative = Void Function(CactusModelT model);
typedef CactusStopNative = Void Function(CactusModelT model);

typedef CactusCompleteNative = Int32 Function(
    CactusModelT model,
    Pointer<Utf8> messagesJson,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<Utf8> toolsJson,
    Pointer<NativeFunction<TokenCallbackNative>> callback,
    Pointer<Void> userData);

typedef CactusTokenizeNative = Int32 Function(
    CactusModelT model,
    Pointer<Utf8> text,
    Pointer<Uint32> tokenBuffer,
    IntPtr tokenBufferLen,
    Pointer<IntPtr> outTokenLen);

typedef CactusScoreWindowNative = Int32 Function(
    CactusModelT model,
    Pointer<Uint32> tokens,
    IntPtr tokenLen,
    IntPtr start,
    IntPtr end,
    IntPtr context,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize);

typedef CactusTranscribeNative = Int32 Function(
    CactusModelT model,
    Pointer<Utf8> audioFilePath,
    Pointer<Utf8> prompt,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<NativeFunction<TokenCallbackNative>> callback,
    Pointer<Void> userData,
    Pointer<Uint8> pcmBuffer,
    IntPtr pcmBufferSize);

typedef CactusStreamTranscribeInitNative = CactusStreamTranscribeT Function(
    CactusModelT model);
typedef CactusStreamTranscribeInsertNative = Int32 Function(
    CactusStreamTranscribeT stream, Pointer<Uint8> pcmBuffer, IntPtr pcmBufferSize);
typedef CactusStreamTranscribeProcessNative = Int32 Function(
    CactusStreamTranscribeT stream,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize,
    Pointer<Utf8> optionsJson);
typedef CactusStreamTranscribeFinalizeNative = Int32 Function(
    CactusStreamTranscribeT stream, Pointer<Utf8> responseBuffer, IntPtr bufferSize);
typedef CactusStreamTranscribeDestroyNative = Void Function(
    CactusStreamTranscribeT stream);

typedef CactusEmbedNative = Int32 Function(
    CactusModelT model,
    Pointer<Utf8> text,
    Pointer<Float> embeddingsBuffer,
    IntPtr bufferSize,
    Pointer<IntPtr> embeddingDim,
    Bool normalize);

typedef CactusImageEmbedNative = Int32 Function(
    CactusModelT model,
    Pointer<Utf8> imagePath,
    Pointer<Float> embeddingsBuffer,
    IntPtr bufferSize,
    Pointer<IntPtr> embeddingDim);

typedef CactusAudioEmbedNative = Int32 Function(
    CactusModelT model,
    Pointer<Utf8> audioPath,
    Pointer<Float> embeddingsBuffer,
    IntPtr bufferSize,
    Pointer<IntPtr> embeddingDim);

typedef CactusVadNative = Int32 Function(
    CactusModelT model,
    Pointer<Utf8> audioFilePath,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<Uint8> pcmBuffer,
    IntPtr pcmBufferSize);

typedef CactusRagQueryNative = Int32 Function(
    CactusModelT model,
    Pointer<Utf8> query,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize,
    IntPtr topK);

typedef CactusIndexInitNative = CactusIndexT Function(
    Pointer<Utf8> indexDir, IntPtr embeddingDim);
typedef CactusIndexAddNative = Int32 Function(
    CactusIndexT index,
    Pointer<Int32> ids,
    Pointer<Pointer<Utf8>> documents,
    Pointer<Pointer<Utf8>> metadatas,
    Pointer<Pointer<Float>> embeddings,
    IntPtr count,
    IntPtr embeddingDim);
typedef CactusIndexDeleteNative = Int32 Function(
    CactusIndexT index, Pointer<Int32> ids, IntPtr idsCount);
typedef CactusIndexQueryNative = Int32 Function(
    CactusIndexT index,
    Pointer<Pointer<Float>> embeddings,
    IntPtr embeddingsCount,
    IntPtr embeddingDim,
    Pointer<Utf8> optionsJson,
    Pointer<Pointer<Int32>> idBuffers,
    Pointer<IntPtr> idBufferSizes,
    Pointer<Pointer<Float>> scoreBuffers,
    Pointer<IntPtr> scoreBufferSizes);
typedef CactusIndexCompactNative = Int32 Function(CactusIndexT index);
typedef CactusIndexDestroyNative = Void Function(CactusIndexT index);

typedef CactusGetLastErrorNative = Pointer<Utf8> Function();


typedef CactusInitDart = CactusModelT Function(
    Pointer<Utf8> modelPath, Pointer<Utf8> corpusDir);
typedef CactusDestroyDart = void Function(CactusModelT model);
typedef CactusResetDart = void Function(CactusModelT model);
typedef CactusStopDart = void Function(CactusModelT model);

typedef CactusCompleteDart = int Function(
    CactusModelT model,
    Pointer<Utf8> messagesJson,
    Pointer<Utf8> responseBuffer,
    int bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<Utf8> toolsJson,
    Pointer<NativeFunction<TokenCallbackNative>> callback,
    Pointer<Void> userData);

typedef CactusTokenizeDart = int Function(
    CactusModelT model,
    Pointer<Utf8> text,
    Pointer<Uint32> tokenBuffer,
    int tokenBufferLen,
    Pointer<IntPtr> outTokenLen);

typedef CactusScoreWindowDart = int Function(
    CactusModelT model,
    Pointer<Uint32> tokens,
    int tokenLen,
    int start,
    int end,
    int context,
    Pointer<Utf8> responseBuffer,
    int bufferSize);

typedef CactusTranscribeDart = int Function(
    CactusModelT model,
    Pointer<Utf8> audioFilePath,
    Pointer<Utf8> prompt,
    Pointer<Utf8> responseBuffer,
    int bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<NativeFunction<TokenCallbackNative>> callback,
    Pointer<Void> userData,
    Pointer<Uint8> pcmBuffer,
    int pcmBufferSize);

typedef CactusStreamTranscribeInitDart = CactusStreamTranscribeT Function(
    CactusModelT model);
typedef CactusStreamTranscribeInsertDart = int Function(
    CactusStreamTranscribeT stream, Pointer<Uint8> pcmBuffer, int pcmBufferSize);
typedef CactusStreamTranscribeProcessDart = int Function(
    CactusStreamTranscribeT stream,
    Pointer<Utf8> responseBuffer,
    int bufferSize,
    Pointer<Utf8> optionsJson);
typedef CactusStreamTranscribeFinalizeDart = int Function(
    CactusStreamTranscribeT stream, Pointer<Utf8> responseBuffer, int bufferSize);
typedef CactusStreamTranscribeDestroyDart = void Function(
    CactusStreamTranscribeT stream);

typedef CactusEmbedDart = int Function(
    CactusModelT model,
    Pointer<Utf8> text,
    Pointer<Float> embeddingsBuffer,
    int bufferSize,
    Pointer<IntPtr> embeddingDim,
    bool normalize);

typedef CactusImageEmbedDart = int Function(
    CactusModelT model,
    Pointer<Utf8> imagePath,
    Pointer<Float> embeddingsBuffer,
    int bufferSize,
    Pointer<IntPtr> embeddingDim);

typedef CactusAudioEmbedDart = int Function(
    CactusModelT model,
    Pointer<Utf8> audioPath,
    Pointer<Float> embeddingsBuffer,
    int bufferSize,
    Pointer<IntPtr> embeddingDim);

typedef CactusVadDart = int Function(
    CactusModelT model,
    Pointer<Utf8> audioFilePath,
    Pointer<Utf8> responseBuffer,
    int bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<Uint8> pcmBuffer,
    int pcmBufferSize);

typedef CactusRagQueryDart = int Function(
    CactusModelT model,
    Pointer<Utf8> query,
    Pointer<Utf8> responseBuffer,
    int bufferSize,
    int topK);

typedef CactusIndexInitDart = CactusIndexT Function(
    Pointer<Utf8> indexDir, int embeddingDim);
typedef CactusIndexAddDart = int Function(
    CactusIndexT index,
    Pointer<Int32> ids,
    Pointer<Pointer<Utf8>> documents,
    Pointer<Pointer<Utf8>> metadatas,
    Pointer<Pointer<Float>> embeddings,
    int count,
    int embeddingDim);
typedef CactusIndexDeleteDart = int Function(
    CactusIndexT index, Pointer<Int32> ids, int idsCount);
typedef CactusIndexQueryDart = int Function(
    CactusIndexT index,
    Pointer<Pointer<Float>> embeddings,
    int embeddingsCount,
    int embeddingDim,
    Pointer<Utf8> optionsJson,
    Pointer<Pointer<Int32>> idBuffers,
    Pointer<IntPtr> idBufferSizes,
    Pointer<Pointer<Float>> scoreBuffers,
    Pointer<IntPtr> scoreBufferSizes);
typedef CactusIndexCompactDart = int Function(CactusIndexT index);
typedef CactusIndexDestroyDart = void Function(CactusIndexT index);

typedef CactusGetLastErrorDart = Pointer<Utf8> Function();


DynamicLibrary _loadLibrary() {
  if (Platform.isAndroid) {
    return DynamicLibrary.open('libcactus.so');
  } else if (Platform.isIOS) {
    return DynamicLibrary.process();
  } else if (Platform.isMacOS) {
    return DynamicLibrary.process();
  } else {
    throw UnsupportedError('Platform not supported: ${Platform.operatingSystem}');
  }
}

final _lib = _loadLibrary();


final _cactusInit =
    _lib.lookupFunction<CactusInitNative, CactusInitDart>('cactus_init');
final _cactusDestroy =
    _lib.lookupFunction<CactusDestroyNative, CactusDestroyDart>('cactus_destroy');
final _cactusReset =
    _lib.lookupFunction<CactusResetNative, CactusResetDart>('cactus_reset');
final _cactusStop =
    _lib.lookupFunction<CactusStopNative, CactusStopDart>('cactus_stop');
final _cactusComplete =
    _lib.lookupFunction<CactusCompleteNative, CactusCompleteDart>('cactus_complete');
final _cactusTokenize =
    _lib.lookupFunction<CactusTokenizeNative, CactusTokenizeDart>('cactus_tokenize');
final _cactusScoreWindow = _lib
    .lookupFunction<CactusScoreWindowNative, CactusScoreWindowDart>('cactus_score_window');
final _cactusTranscribe =
    _lib.lookupFunction<CactusTranscribeNative, CactusTranscribeDart>('cactus_transcribe');
final _cactusStreamTranscribeInit = _lib.lookupFunction<
    CactusStreamTranscribeInitNative,
    CactusStreamTranscribeInitDart>('cactus_stream_transcribe_init');
final _cactusStreamTranscribeInsert = _lib.lookupFunction<
    CactusStreamTranscribeInsertNative,
    CactusStreamTranscribeInsertDart>('cactus_stream_transcribe_insert');
final _cactusStreamTranscribeProcess = _lib.lookupFunction<
    CactusStreamTranscribeProcessNative,
    CactusStreamTranscribeProcessDart>('cactus_stream_transcribe_process');
final _cactusStreamTranscribeFinalize = _lib.lookupFunction<
    CactusStreamTranscribeFinalizeNative,
    CactusStreamTranscribeFinalizeDart>('cactus_stream_transcribe_finalize');
final _cactusStreamTranscribeDestroy = _lib.lookupFunction<
    CactusStreamTranscribeDestroyNative,
    CactusStreamTranscribeDestroyDart>('cactus_stream_transcribe_destroy');
final _cactusEmbed =
    _lib.lookupFunction<CactusEmbedNative, CactusEmbedDart>('cactus_embed');
final _cactusImageEmbed =
    _lib.lookupFunction<CactusImageEmbedNative, CactusImageEmbedDart>('cactus_image_embed');
final _cactusAudioEmbed =
    _lib.lookupFunction<CactusAudioEmbedNative, CactusAudioEmbedDart>('cactus_audio_embed');
final _cactusVad =
    _lib.lookupFunction<CactusVadNative, CactusVadDart>('cactus_vad');
final _cactusRagQuery =
    _lib.lookupFunction<CactusRagQueryNative, CactusRagQueryDart>('cactus_rag_query');
final _cactusIndexInit =
    _lib.lookupFunction<CactusIndexInitNative, CactusIndexInitDart>('cactus_index_init');
final _cactusIndexAdd =
    _lib.lookupFunction<CactusIndexAddNative, CactusIndexAddDart>('cactus_index_add');
final _cactusIndexDelete =
    _lib.lookupFunction<CactusIndexDeleteNative, CactusIndexDeleteDart>('cactus_index_delete');
final _cactusIndexQuery =
    _lib.lookupFunction<CactusIndexQueryNative, CactusIndexQueryDart>('cactus_index_query');
final _cactusIndexCompact =
    _lib.lookupFunction<CactusIndexCompactNative, CactusIndexCompactDart>('cactus_index_compact');
final _cactusIndexDestroy =
    _lib.lookupFunction<CactusIndexDestroyNative, CactusIndexDestroyDart>('cactus_index_destroy');
final _cactusGetLastError = _lib
    .lookupFunction<CactusGetLastErrorNative, CactusGetLastErrorDart>('cactus_get_last_error');

// ----------------------------------------------------------------------------
// Helper Extensions
// ----------------------------------------------------------------------------

extension Utf8Pointer on String {
  Pointer<Utf8> toNativeUtf8({Allocator allocator = malloc}) {
    final units = utf8.encode(this);
    final ptr = allocator<Uint8>(units.length + 1);
    final list = ptr.asTypedList(units.length + 1);
    list.setAll(0, units);
    list[units.length] = 0;
    return ptr.cast();
  }
}

extension Utf8PointerExtension on Pointer<Utf8> {
  String toDartString() {
    if (this == nullptr) return '';
    final codeUnits = <int>[];
    var i = 0;
    while (true) {
      final byte = cast<Uint8>().elementAt(i).value;
      if (byte == 0) break;
      codeUnits.add(byte);
      i++;
    }
    return utf8.decode(codeUnits);
  }
}

final malloc = _MallocAllocator();

class _MallocAllocator implements Allocator {
  @override
  Pointer<T> allocate<T extends NativeType>(int byteCount, {int? alignment}) {
    return calloc.allocate<T>(byteCount, alignment: alignment);
  }

  @override
  void free(Pointer pointer) {
    calloc.free(pointer);
  }
}

final calloc = _Calloc();

class _Calloc implements Allocator {
  static final _callocPtr = _lib.lookup<NativeFunction<Pointer<Void> Function(IntPtr, IntPtr)>>('calloc');
  static final _freePtr = _lib.lookup<NativeFunction<Void Function(Pointer<Void>)>>('free');
  static final _calloc = _callocPtr.asFunction<Pointer<Void> Function(int, int)>();
  static final _free = _freePtr.asFunction<void Function(Pointer<Void>)>();

  @override
  Pointer<T> allocate<T extends NativeType>(int byteCount, {int? alignment}) {
    return _calloc(byteCount, 1).cast<T>();
  }

  @override
  void free(Pointer pointer) {
    _free(pointer.cast());
  }
}

class Message {
  final String role;
  final String content;

  Message._(this.role, this.content);

  factory Message.system(String content) => Message._('system', content);
  factory Message.user(String content) => Message._('user', content);
  factory Message.assistant(String content) => Message._('assistant', content);

  Map<String, String> toJson() => {'role': role, 'content': content};
}

class CompletionOptions {
  final double temperature;
  final double topP;
  final int topK;
  final int maxTokens;
  final List<String> stopSequences;
  final double confidenceThreshold;

  const CompletionOptions({
    this.temperature = 0.7,
    this.topP = 0.9,
    this.topK = 40,
    this.maxTokens = 512,
    this.stopSequences = const [],
    this.confidenceThreshold = 0.0,
  });

  static const defaultOptions = CompletionOptions();

  Map<String, dynamic> toJson() => {
        'temperature': temperature,
        'top_p': topP,
        'top_k': topK,
        'max_tokens': maxTokens,
        'stop_sequences': stopSequences,
        'confidence_threshold': confidenceThreshold,
      };
}

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

  CompletionResult({
    required this.text,
    this.functionCalls,
    this.promptTokens = 0,
    this.completionTokens = 0,
    this.timeToFirstToken = 0.0,
    this.totalTime = 0.0,
    this.prefillTokensPerSecond = 0.0,
    this.decodeTokensPerSecond = 0.0,
    this.confidence = 1.0,
    this.needsCloudHandoff = false,
  });

  factory CompletionResult.fromJson(Map<String, dynamic> json) {
    return CompletionResult(
      text: json['text'] ?? '',
      functionCalls: json['function_calls'] != null
          ? List<Map<String, dynamic>>.from(json['function_calls'])
          : null,
      promptTokens: json['prompt_tokens'] ?? 0,
      completionTokens: json['completion_tokens'] ?? 0,
      timeToFirstToken: (json['time_to_first_token'] ?? 0.0).toDouble(),
      totalTime: (json['total_time'] ?? 0.0).toDouble(),
      prefillTokensPerSecond: (json['prefill_tokens_per_second'] ?? 0.0).toDouble(),
      decodeTokensPerSecond: (json['decode_tokens_per_second'] ?? 0.0).toDouble(),
      confidence: (json['confidence'] ?? 1.0).toDouble(),
      needsCloudHandoff: json['needs_cloud_handoff'] ?? false,
    );
  }
}

class TranscriptionOptions {
  final String? language;
  final bool translate;

  const TranscriptionOptions({
    this.language,
    this.translate = false,
  });

  static const defaultOptions = TranscriptionOptions();

  Map<String, dynamic> toJson() => {
        if (language != null) 'language': language,
        'translate': translate,
      };
}

class TranscriptionResult {
  final String text;
  final List<Map<String, dynamic>>? segments;
  final double totalTime;

  TranscriptionResult({
    required this.text,
    this.segments,
    this.totalTime = 0.0,
  });

  factory TranscriptionResult.fromJson(Map<String, dynamic> json) {
    return TranscriptionResult(
      text: json['text'] ?? '',
      segments: json['segments'] != null
          ? List<Map<String, dynamic>>.from(json['segments'])
          : null,
      totalTime: (json['total_time'] ?? 0.0).toDouble(),
    );
  }
}

class VADSegment {
  final int start;
  final int end;

  VADSegment({required this.start, required this.end});

  factory VADSegment.fromJson(Map<String, dynamic> json) {
    return VADSegment(
      start: json['start'] ?? 0,
      end: json['end'] ?? 0,
    );
  }
}

class VADResult {
  final List<VADSegment> segments;
  final double totalTime;
  final double ramUsage;

  VADResult({
    required this.segments,
    this.totalTime = 0.0,
    this.ramUsage = 0.0,
  });

  factory VADResult.fromJson(Map<String, dynamic> json) {
    final segmentsList = json['segments'] as List<dynamic>? ?? [];
    return VADResult(
      segments: segmentsList
          .map((s) => VADSegment.fromJson(s as Map<String, dynamic>))
          .toList(),
      totalTime: (json['total_time_ms'] ?? 0.0).toDouble(),
      ramUsage: (json['ram_usage_mb'] ?? 0.0).toDouble(),
    );
  }
}

class VADOptions {
  final double? threshold;
  final double? negThreshold;
  final int? minSpeechDurationMs;
  final double? maxSpeechDurationS;
  final int? minSilenceDurationMs;
  final int? speechPadMs;
  final int? windowSizeSamples;
  final int? samplingRate;

  const VADOptions({
    this.threshold,
    this.negThreshold,
    this.minSpeechDurationMs,
    this.maxSpeechDurationS,
    this.minSilenceDurationMs,
    this.speechPadMs,
    this.windowSizeSamples,
    this.samplingRate,
  });

  static const defaultOptions = VADOptions();

  Map<String, dynamic> toJson() {
    final map = <String, dynamic>{};
    if (threshold != null) map['threshold'] = threshold;
    if (negThreshold != null) map['neg_threshold'] = negThreshold;
    if (minSpeechDurationMs != null) map['min_speech_duration_ms'] = minSpeechDurationMs;
    if (maxSpeechDurationS != null) map['max_speech_duration_s'] = maxSpeechDurationS;
    if (minSilenceDurationMs != null) map['min_silence_duration_ms'] = minSilenceDurationMs;
    if (speechPadMs != null) map['speech_pad_ms'] = speechPadMs;
    if (windowSizeSamples != null) map['window_size_samples'] = windowSizeSamples;
    if (samplingRate != null) map['sampling_rate'] = samplingRate;
    return map;
  }
}

class IndexResult {
  final int id;
  final double score;

  IndexResult({required this.id, required this.score});
}

class Cactus {
  final CactusModelT _handle;
  bool _disposed = false;

  Cactus._(this._handle);

  static Cactus create(String modelPath, {String? corpusDir}) {
    final modelPathPtr = modelPath.toNativeUtf8();
    final corpusDirPtr = corpusDir?.toNativeUtf8() ?? nullptr;

    try {
      final handle = _cactusInit(modelPathPtr, corpusDirPtr);
      if (handle == nullptr) {
        throw CactusException('Failed to initialize model: ${getLastError()}');
      }
      return Cactus._(handle);
    } finally {
      calloc.free(modelPathPtr);
      if (corpusDirPtr != nullptr) calloc.free(corpusDirPtr);
    }
  }

  void _checkNotDisposed() {
    if (_disposed) {
      throw StateError('Cactus instance has been disposed');
    }
  }

  CompletionResult complete(
    String prompt, {
    CompletionOptions options = CompletionOptions.defaultOptions,
    void Function(String token, int tokenId)? onToken,
  }) {
    return completeMessages(
      [Message.user(prompt)],
      options: options,
      onToken: onToken,
    );
  }

  CompletionResult completeMessages(
    List<Message> messages, {
    CompletionOptions options = CompletionOptions.defaultOptions,
    List<Map<String, dynamic>>? tools,
    void Function(String token, int tokenId)? onToken,
  }) {
    _checkNotDisposed();

    final messagesJson = jsonEncode(messages.map((m) => m.toJson()).toList());
    final optionsJson = jsonEncode(options.toJson());
    final toolsJson = tools != null ? jsonEncode(tools) : null;

    const bufferSize = 1024 * 1024; // 1MB buffer
    final responseBuffer = calloc<Uint8>(bufferSize);
    final messagesPtr = messagesJson.toNativeUtf8();
    final optionsPtr = optionsJson.toNativeUtf8();
    final toolsPtr = toolsJson?.toNativeUtf8() ?? nullptr;

    Pointer<NativeFunction<TokenCallbackNative>> callbackPtr = nullptr;
    if (onToken != null) {
    }

    try {
      final result = _cactusComplete(
        _handle,
        messagesPtr,
        responseBuffer.cast(),
        bufferSize,
        optionsPtr,
        toolsPtr,
        callbackPtr,
        nullptr,
      );

      if (result != 0) {
        throw CactusException('Completion failed: ${getLastError()}');
      }

      final responseStr = responseBuffer.cast<Utf8>().toDartString();
      final responseJson = jsonDecode(responseStr) as Map<String, dynamic>;
      return CompletionResult.fromJson(responseJson);
    } finally {
      calloc.free(responseBuffer);
      calloc.free(messagesPtr);
      calloc.free(optionsPtr);
      if (toolsPtr != nullptr) calloc.free(toolsPtr);
    }
  }

  List<int> tokenize(String text) {
    _checkNotDisposed();

    const maxTokens = 8192;
    final tokenBuffer = calloc<Uint32>(maxTokens);
    final outTokenLen = calloc<IntPtr>(1);
    final textPtr = text.toNativeUtf8();

    try {
      final result = _cactusTokenize(
        _handle,
        textPtr,
        tokenBuffer,
        maxTokens,
        outTokenLen,
      );

      if (result != 0) {
        throw CactusException('Tokenization failed: ${getLastError()}');
      }

      final tokenCount = outTokenLen.value;
      return List<int>.generate(tokenCount, (i) => tokenBuffer[i]);
    } finally {
      calloc.free(tokenBuffer);
      calloc.free(outTokenLen);
      calloc.free(textPtr);
    }
  }

  String scoreWindow(List<int> tokens, int start, int end, int context) {
    _checkNotDisposed();

    final tokenBuffer = calloc<Uint32>(tokens.length);
    for (var i = 0; i < tokens.length; i++) {
      tokenBuffer[i] = tokens[i];
    }

    const bufferSize = 65536;
    final responseBuffer = calloc<Uint8>(bufferSize);

    try {
      final result = _cactusScoreWindow(
        _handle,
        tokenBuffer,
        tokens.length,
        start,
        end,
        context,
        responseBuffer.cast(),
        bufferSize,
      );

      if (result != 0) {
        throw CactusException('Score window failed: ${getLastError()}');
      }

      return responseBuffer.cast<Utf8>().toDartString();
    } finally {
      calloc.free(tokenBuffer);
      calloc.free(responseBuffer);
    }
  }

  TranscriptionResult transcribe(
    String audioPath, {
    String? prompt,
    TranscriptionOptions options = TranscriptionOptions.defaultOptions,
  }) {
    _checkNotDisposed();

    const bufferSize = 1024 * 1024;
    final responseBuffer = calloc<Uint8>(bufferSize);
    final audioPathPtr = audioPath.toNativeUtf8();
    final promptPtr = prompt?.toNativeUtf8() ?? nullptr;
    final optionsJson = jsonEncode(options.toJson());
    final optionsPtr = optionsJson.toNativeUtf8();

    try {
      final result = _cactusTranscribe(
        _handle,
        audioPathPtr,
        promptPtr,
        responseBuffer.cast(),
        bufferSize,
        optionsPtr,
        nullptr,
        nullptr,
        nullptr,
        0,
      );

      if (result != 0) {
        throw CactusException('Transcription failed: ${getLastError()}');
      }

      final responseStr = responseBuffer.cast<Utf8>().toDartString();
      final responseJson = jsonDecode(responseStr) as Map<String, dynamic>;
      return TranscriptionResult.fromJson(responseJson);
    } finally {
      calloc.free(responseBuffer);
      calloc.free(audioPathPtr);
      if (promptPtr != nullptr) calloc.free(promptPtr);
      calloc.free(optionsPtr);
    }
  }

  TranscriptionResult transcribePcm(
    Uint8List pcmData, {
    String? prompt,
    TranscriptionOptions options = TranscriptionOptions.defaultOptions,
  }) {
    _checkNotDisposed();

    const bufferSize = 1024 * 1024;
    final responseBuffer = calloc<Uint8>(bufferSize);
    final promptPtr = prompt?.toNativeUtf8() ?? nullptr;
    final optionsJson = jsonEncode(options.toJson());
    final optionsPtr = optionsJson.toNativeUtf8();
    final pcmBuffer = calloc<Uint8>(pcmData.length);
    pcmBuffer.asTypedList(pcmData.length).setAll(0, pcmData);

    try {
      final result = _cactusTranscribe(
        _handle,
        nullptr,
        promptPtr,
        responseBuffer.cast(),
        bufferSize,
        optionsPtr,
        nullptr,
        nullptr,
        pcmBuffer,
        pcmData.length,
      );

      if (result != 0) {
        throw CactusException('Transcription failed: ${getLastError()}');
      }

      final responseStr = responseBuffer.cast<Utf8>().toDartString();
      final responseJson = jsonDecode(responseStr) as Map<String, dynamic>;
      return TranscriptionResult.fromJson(responseJson);
    } finally {
      calloc.free(responseBuffer);
      if (promptPtr != nullptr) calloc.free(promptPtr);
      calloc.free(optionsPtr);
      calloc.free(pcmBuffer);
    }
  }

  StreamTranscriber createStreamTranscriber() {
    _checkNotDisposed();
    final handle = _cactusStreamTranscribeInit(_handle);
    if (handle == nullptr) {
      throw CactusException('Failed to create stream transcriber: ${getLastError()}');
    }
    return StreamTranscriber._(handle);
  }

  List<double> embed(String text, {bool normalize = true}) {
    _checkNotDisposed();

    const maxDim = 8192;
    final embeddingsBuffer = calloc<Float>(maxDim);
    final embeddingDim = calloc<IntPtr>(1);
    final textPtr = text.toNativeUtf8();

    try {
      final result = _cactusEmbed(
        _handle,
        textPtr,
        embeddingsBuffer,
        maxDim,
        embeddingDim,
        normalize,
      );

      if (result != 0) {
        throw CactusException('Embedding failed: ${getLastError()}');
      }

      final dim = embeddingDim.value;
      return List<double>.generate(dim, (i) => embeddingsBuffer[i]);
    } finally {
      calloc.free(embeddingsBuffer);
      calloc.free(embeddingDim);
      calloc.free(textPtr);
    }
  }

  List<double> imageEmbed(String imagePath) {
    _checkNotDisposed();

    const maxDim = 8192;
    final embeddingsBuffer = calloc<Float>(maxDim);
    final embeddingDim = calloc<IntPtr>(1);
    final imagePathPtr = imagePath.toNativeUtf8();

    try {
      final result = _cactusImageEmbed(
        _handle,
        imagePathPtr,
        embeddingsBuffer,
        maxDim,
        embeddingDim,
      );

      if (result != 0) {
        throw CactusException('Image embedding failed: ${getLastError()}');
      }

      final dim = embeddingDim.value;
      return List<double>.generate(dim, (i) => embeddingsBuffer[i]);
    } finally {
      calloc.free(embeddingsBuffer);
      calloc.free(embeddingDim);
      calloc.free(imagePathPtr);
    }
  }

  List<double> audioEmbed(String audioPath) {
    _checkNotDisposed();

    const maxDim = 8192;
    final embeddingsBuffer = calloc<Float>(maxDim);
    final embeddingDim = calloc<IntPtr>(1);
    final audioPathPtr = audioPath.toNativeUtf8();

    try {
      final result = _cactusAudioEmbed(
        _handle,
        audioPathPtr,
        embeddingsBuffer,
        maxDim,
        embeddingDim,
      );

      if (result != 0) {
        throw CactusException('Audio embedding failed: ${getLastError()}');
      }

      final dim = embeddingDim.value;
      return List<double>.generate(dim, (i) => embeddingsBuffer[i]);
    } finally {
      calloc.free(embeddingsBuffer);
      calloc.free(embeddingDim);
      calloc.free(audioPathPtr);
    }
  }

  VADResult vad(
    String audioPath, {
    VADOptions options = VADOptions.defaultOptions,
  }) {
    _checkNotDisposed();

    const bufferSize = 1024 * 1024;
    final responseBuffer = calloc<Uint8>(bufferSize);
    final audioPathPtr = audioPath.toNativeUtf8();
    final optionsMap = options.toJson();
    final optionsJson = optionsMap.isNotEmpty ? jsonEncode(optionsMap) : null;
    final optionsPtr = optionsJson?.toNativeUtf8() ?? nullptr;

    try {
      final result = _cactusVad(
        _handle,
        audioPathPtr,
        responseBuffer.cast(),
        bufferSize,
        optionsPtr,
        nullptr,
        0,
      );

      if (result < 0) {
        throw CactusException('VAD failed: ${getLastError()}');
      }

      final responseStr = responseBuffer.cast<Utf8>().toDartString();
      final responseJson = jsonDecode(responseStr) as Map<String, dynamic>;

      if (responseJson['error'] != null) {
        throw CactusException('VAD error: ${responseJson['error']}');
      }

      return VADResult.fromJson(responseJson);
    } finally {
      calloc.free(responseBuffer);
      calloc.free(audioPathPtr);
      if (optionsPtr != nullptr) calloc.free(optionsPtr);
    }
  }

  VADResult vadPcm(
    Uint8List pcmData, {
    VADOptions options = VADOptions.defaultOptions,
  }) {
    _checkNotDisposed();

    const bufferSize = 1024 * 1024;
    final responseBuffer = calloc<Uint8>(bufferSize);
    final optionsMap = options.toJson();
    final optionsJson = optionsMap.isNotEmpty ? jsonEncode(optionsMap) : null;
    final optionsPtr = optionsJson?.toNativeUtf8() ?? nullptr;
    final pcmBuffer = calloc<Uint8>(pcmData.length);
    pcmBuffer.asTypedList(pcmData.length).setAll(0, pcmData);

    try {
      final result = _cactusVad(
        _handle,
        nullptr,
        responseBuffer.cast(),
        bufferSize,
        optionsPtr,
        pcmBuffer,
        pcmData.length,
      );

      if (result < 0) {
        throw CactusException('VAD failed: ${getLastError()}');
      }

      final responseStr = responseBuffer.cast<Utf8>().toDartString();
      final responseJson = jsonDecode(responseStr) as Map<String, dynamic>;

      if (responseJson['error'] != null) {
        throw CactusException('VAD error: ${responseJson['error']}');
      }

      return VADResult.fromJson(responseJson);
    } finally {
      calloc.free(responseBuffer);
      if (optionsPtr != nullptr) calloc.free(optionsPtr);
      calloc.free(pcmBuffer);
    }
  }

  String ragQuery(String query, {int topK = 5}) {
    _checkNotDisposed();

    const bufferSize = 1024 * 1024;
    final responseBuffer = calloc<Uint8>(bufferSize);
    final queryPtr = query.toNativeUtf8();

    try {
      final result = _cactusRagQuery(
        _handle,
        queryPtr,
        responseBuffer.cast(),
        bufferSize,
        topK,
      );

      if (result != 0) {
        throw CactusException('RAG query failed: ${getLastError()}');
      }

      return responseBuffer.cast<Utf8>().toDartString();
    } finally {
      calloc.free(responseBuffer);
      calloc.free(queryPtr);
    }
  }

  void reset() {
    _checkNotDisposed();
    _cactusReset(_handle);
  }

  void stop() {
    _checkNotDisposed();
    _cactusStop(_handle);
  }

  void dispose() {
    if (!_disposed) {
      _cactusDestroy(_handle);
      _disposed = true;
    }
  }

  static String getLastError() {
    return _cactusGetLastError().toDartString();
  }
}

class StreamTranscriber {
  final CactusStreamTranscribeT _handle;
  bool _disposed = false;

  StreamTranscriber._(this._handle);

  void _checkNotDisposed() {
    if (_disposed) {
      throw StateError('StreamTranscriber has been disposed');
    }
  }

  void insert(Uint8List pcmData) {
    _checkNotDisposed();

    final pcmBuffer = calloc<Uint8>(pcmData.length);
    pcmBuffer.asTypedList(pcmData.length).setAll(0, pcmData);

    try {
      final result = _cactusStreamTranscribeInsert(_handle, pcmBuffer, pcmData.length);
      if (result != 0) {
        throw CactusException('Stream insert failed: ${Cactus.getLastError()}');
      }
    } finally {
      calloc.free(pcmBuffer);
    }
  }

  TranscriptionResult process({String? language}) {
    _checkNotDisposed();

    const bufferSize = 1024 * 1024;
    final responseBuffer = calloc<Uint8>(bufferSize);
    final optionsJson = language != null ? jsonEncode({'language': language}) : null;
    final optionsPtr = optionsJson?.toNativeUtf8() ?? nullptr;

    try {
      final result = _cactusStreamTranscribeProcess(
        _handle,
        responseBuffer.cast(),
        bufferSize,
        optionsPtr,
      );

      if (result != 0) {
        throw CactusException('Stream process failed: ${Cactus.getLastError()}');
      }

      final responseStr = responseBuffer.cast<Utf8>().toDartString();
      final responseJson = jsonDecode(responseStr) as Map<String, dynamic>;
      return TranscriptionResult.fromJson(responseJson);
    } finally {
      calloc.free(responseBuffer);
      if (optionsPtr != nullptr) calloc.free(optionsPtr);
    }
  }

  TranscriptionResult finalize() {
    _checkNotDisposed();

    const bufferSize = 1024 * 1024;
    final responseBuffer = calloc<Uint8>(bufferSize);

    try {
      final result = _cactusStreamTranscribeFinalize(
        _handle,
        responseBuffer.cast(),
        bufferSize,
      );

      if (result != 0) {
        throw CactusException('Stream finalize failed: ${Cactus.getLastError()}');
      }

      final responseStr = responseBuffer.cast<Utf8>().toDartString();
      final responseJson = jsonDecode(responseStr) as Map<String, dynamic>;
      return TranscriptionResult.fromJson(responseJson);
    } finally {
      calloc.free(responseBuffer);
    }
  }

  void dispose() {
    if (!_disposed) {
      _cactusStreamTranscribeDestroy(_handle);
      _disposed = true;
    }
  }
}

class CactusIndex {
  final CactusIndexT _handle;
  final int embeddingDim;
  bool _disposed = false;

  CactusIndex._(this._handle, this.embeddingDim);

  static CactusIndex create(String indexDir, {required int embeddingDim}) {
    final indexDirPtr = indexDir.toNativeUtf8();

    try {
      final handle = _cactusIndexInit(indexDirPtr, embeddingDim);
      if (handle == nullptr) {
        throw CactusException('Failed to create index: ${Cactus.getLastError()}');
      }
      return CactusIndex._(handle, embeddingDim);
    } finally {
      calloc.free(indexDirPtr);
    }
  }

  void _checkNotDisposed() {
    if (_disposed) {
      throw StateError('CactusIndex has been disposed');
    }
  }

  void add({
    required List<int> ids,
    required List<String> documents,
    required List<List<double>> embeddings,
    List<String>? metadatas,
  }) {
    _checkNotDisposed();

    if (ids.length != documents.length || ids.length != embeddings.length) {
      throw ArgumentError('ids, documents, and embeddings must have the same length');
    }

    final count = ids.length;
    final idsPtr = calloc<Int32>(count);
    for (var i = 0; i < count; i++) {
      idsPtr[i] = ids[i];
    }

    final documentsPtr = calloc<Pointer<Utf8>>(count);
    for (var i = 0; i < count; i++) {
      documentsPtr[i] = documents[i].toNativeUtf8();
    }

    final metadatasPtr = metadatas != null ? calloc<Pointer<Utf8>>(count) : nullptr;
    if (metadatas != null) {
      for (var i = 0; i < count; i++) {
        metadatasPtr[i] = metadatas[i].toNativeUtf8();
      }
    }

    final embeddingsPtr = calloc<Pointer<Float>>(count);
    for (var i = 0; i < count; i++) {
      final embedding = embeddings[i];
      final embPtr = calloc<Float>(embedding.length);
      for (var j = 0; j < embedding.length; j++) {
        embPtr[j] = embedding[j];
      }
      embeddingsPtr[i] = embPtr;
    }

    try {
      final result = _cactusIndexAdd(
        _handle,
        idsPtr,
        documentsPtr,
        metadatasPtr,
        embeddingsPtr,
        count,
        embeddingDim,
      );

      if (result != 0) {
        throw CactusException('Index add failed: ${Cactus.getLastError()}');
      }
    } finally {
      calloc.free(idsPtr);
      for (var i = 0; i < count; i++) {
        calloc.free(documentsPtr[i]);
        calloc.free(embeddingsPtr[i]);
        if (metadatasPtr != nullptr) calloc.free(metadatasPtr[i]);
      }
      calloc.free(documentsPtr);
      calloc.free(embeddingsPtr);
      if (metadatasPtr != nullptr) calloc.free(metadatasPtr);
    }
  }

  void delete(List<int> ids) {
    _checkNotDisposed();

    final idsPtr = calloc<Int32>(ids.length);
    for (var i = 0; i < ids.length; i++) {
      idsPtr[i] = ids[i];
    }

    try {
      final result = _cactusIndexDelete(_handle, idsPtr, ids.length);
      if (result != 0) {
        throw CactusException('Index delete failed: ${Cactus.getLastError()}');
      }
    } finally {
      calloc.free(idsPtr);
    }
  }

  List<IndexResult> query(List<double> embedding, {int topK = 5}) {
    _checkNotDisposed();

    final embPtr = calloc<Float>(embedding.length);
    for (var i = 0; i < embedding.length; i++) {
      embPtr[i] = embedding[i];
    }

    final embPtrPtr = calloc<Pointer<Float>>(1);
    embPtrPtr[0] = embPtr;

    final optionsJson = jsonEncode({'top_k': topK});
    final optionsPtr = optionsJson.toNativeUtf8();

    final idBuffers = calloc<Pointer<Int32>>(1);
    final idBufferSizes = calloc<IntPtr>(1);
    final scoreBuffers = calloc<Pointer<Float>>(1);
    final scoreBufferSizes = calloc<IntPtr>(1);

    try {
      final result = _cactusIndexQuery(
        _handle,
        embPtrPtr,
        1,
        embedding.length,
        optionsPtr,
        idBuffers,
        idBufferSizes,
        scoreBuffers,
        scoreBufferSizes,
      );

      if (result != 0) {
        throw CactusException('Index query failed: ${Cactus.getLastError()}');
      }

      final resultCount = idBufferSizes[0];
      final results = <IndexResult>[];
      for (var i = 0; i < resultCount; i++) {
        results.add(IndexResult(
          id: idBuffers[0][i],
          score: scoreBuffers[0][i],
        ));
      }

      if (idBuffers[0] != nullptr) calloc.free(idBuffers[0]);
      if (scoreBuffers[0] != nullptr) calloc.free(scoreBuffers[0]);

      return results;
    } finally {
      calloc.free(embPtr);
      calloc.free(embPtrPtr);
      calloc.free(optionsPtr);
      calloc.free(idBuffers);
      calloc.free(idBufferSizes);
      calloc.free(scoreBuffers);
      calloc.free(scoreBufferSizes);
    }
  }

  void compact() {
    _checkNotDisposed();
    final result = _cactusIndexCompact(_handle);
    if (result != 0) {
      throw CactusException('Index compact failed: ${Cactus.getLastError()}');
    }
  }

  void dispose() {
    if (!_disposed) {
      _cactusIndexDestroy(_handle);
      _disposed = true;
    }
  }
}

class CactusException implements Exception {
  final String message;
  CactusException(this.message);

  @override
  String toString() => 'CactusException: $message';
}
