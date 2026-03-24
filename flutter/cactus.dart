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
    Pointer<Utf8> modelPath, Pointer<Utf8> corpusDir, Bool cacheIndex);
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

typedef CactusPrefillNative = Int32 Function(
    CactusModelT model,
    Pointer<Utf8> messagesJson,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<Utf8> toolsJson);

typedef CactusCompleteDart = Int32 Function(
    CactusModelT model,
    Pointer<Utf8> messagesJson,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<Utf8> toolsJson,
    Pointer<NativeFunction<TokenCallbackNative>> callback,
    Pointer<Void> userData);

typedef CactusPrefillDart = Int32 Function(
    CactusModelT model,
    Pointer<Utf8> messagesJson,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<Utf8> toolsJson);

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

typedef CactusDetectLanguageNative = Int32 Function(
    CactusModelT model,
    Pointer<Utf8> audioFilePath,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<Uint8> pcmBuffer,
    IntPtr pcmBufferSize);

typedef CactusStreamTranscribeStartNative = CactusStreamTranscribeT Function(
    CactusModelT model, Pointer<Utf8> optionsJson);
typedef CactusStreamTranscribeProcessNative = Int32 Function(
    CactusStreamTranscribeT stream,
    Pointer<Uint8> pcmBuffer,
    IntPtr pcmBufferSize,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize);
typedef CactusStreamTranscribeStopNative = Int32 Function(
    CactusStreamTranscribeT stream, Pointer<Utf8> responseBuffer, IntPtr bufferSize);

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

typedef CactusSetTelemetryEnvironmentNative = Void Function(
    Pointer<Utf8> framework, Pointer<Utf8> cacheLocation, Pointer<Utf8> version);

typedef CactusInitDart = CactusModelT Function(
    Pointer<Utf8> modelPath, Pointer<Utf8> corpusDir, bool cacheIndex);
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

typedef CactusDetectLanguageDart = int Function(
    CactusModelT model,
    Pointer<Utf8> audioFilePath,
    Pointer<Utf8> responseBuffer,
    int bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<Uint8> pcmBuffer,
    int pcmBufferSize);

typedef CactusStreamTranscribeStartDart = CactusStreamTranscribeT Function(
    CactusModelT model, Pointer<Utf8> optionsJson);
typedef CactusStreamTranscribeProcessDart = int Function(
    CactusStreamTranscribeT stream,
    Pointer<Uint8> pcmBuffer,
    int pcmBufferSize,
    Pointer<Utf8> responseBuffer,
    int bufferSize);
typedef CactusStreamTranscribeStopDart = int Function(
    CactusStreamTranscribeT stream, Pointer<Utf8> responseBuffer, int bufferSize);

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

typedef CactusSetTelemetryEnvironmentDart = void Function(
    Pointer<Utf8> framework, Pointer<Utf8> cacheLocation, Pointer<Utf8> version);

typedef CactusSetAppIdNative = Void Function(Pointer<Utf8> appId);
typedef CactusSetAppIdDart = void Function(Pointer<Utf8> appId);

typedef CactusTelemetryFlushNative = Void Function();
typedef CactusTelemetryFlushDart = void Function();

typedef CactusTelemetryShutdownNative = Void Function();
typedef CactusTelemetryShutdownDart = void Function();

typedef LogCallbackNative = Void Function(
    Int32 level, Pointer<Utf8> component, Pointer<Utf8> message, Pointer<Void> userData);

typedef CactusLogSetLevelNative = Void Function(Int32 level);
typedef CactusLogSetLevelDart = void Function(int level);

typedef CactusLogSetCallbackNative = Void Function(
    Pointer<NativeFunction<LogCallbackNative>> callback, Pointer<Void> userData);
typedef CactusLogSetCallbackDart = void Function(
    Pointer<NativeFunction<LogCallbackNative>> callback, Pointer<Void> userData);

typedef CactusIndexGetNative = Int32 Function(
    CactusIndexT index,
    Pointer<Int32> ids,
    IntPtr idsCount,
    Pointer<Pointer<Utf8>> documentBuffers,
    Pointer<IntPtr> documentBufferSizes,
    Pointer<Pointer<Utf8>> metadataBuffers,
    Pointer<IntPtr> metadataBufferSizes,
    Pointer<Pointer<Float>> embeddingBuffers,
    Pointer<IntPtr> embeddingBufferSizes);
typedef CactusIndexGetDart = int Function(
    CactusIndexT index,
    Pointer<Int32> ids,
    int idsCount,
    Pointer<Pointer<Utf8>> documentBuffers,
    Pointer<IntPtr> documentBufferSizes,
    Pointer<Pointer<Utf8>> metadataBuffers,
    Pointer<IntPtr> metadataBufferSizes,
    Pointer<Pointer<Float>> embeddingBuffers,
    Pointer<IntPtr> embeddingBufferSizes);


DynamicLibrary _loadLibrary() {
  if (Platform.isAndroid) {
    return DynamicLibrary.open('libcactus.so');
  } else if (Platform.isIOS) {
    return DynamicLibrary.process();
  } else if (Platform.isMacOS) {
    final explicit = Platform.environment['CACTUS_DYLIB_PATH'];
    if (explicit != null) return DynamicLibrary.open(explicit);
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
final _cactusPrefill =
    _lib.lookupFunction<CactusPrefillNative, CactusPrefillDart>('cactus_prefill');
final _cactusTokenize =
    _lib.lookupFunction<CactusTokenizeNative, CactusTokenizeDart>('cactus_tokenize');
final _cactusScoreWindow = _lib
    .lookupFunction<CactusScoreWindowNative, CactusScoreWindowDart>('cactus_score_window');
final _cactusTranscribe =
    _lib.lookupFunction<CactusTranscribeNative, CactusTranscribeDart>('cactus_transcribe');
final _cactusStreamTranscribeStart = _lib.lookupFunction<
    CactusStreamTranscribeStartNative,
    CactusStreamTranscribeStartDart>('cactus_stream_transcribe_start');
final _cactusStreamTranscribeProcess = _lib.lookupFunction<
    CactusStreamTranscribeProcessNative,
    CactusStreamTranscribeProcessDart>('cactus_stream_transcribe_process');
final _cactusStreamTranscribeStop = _lib.lookupFunction<
    CactusStreamTranscribeStopNative,
    CactusStreamTranscribeStopDart>('cactus_stream_transcribe_stop');
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
final _cactusSetTelemetryEnvironment = _lib.lookupFunction<
    CactusSetTelemetryEnvironmentNative,
    CactusSetTelemetryEnvironmentDart>('cactus_set_telemetry_environment');
final _cactusTelemetryFlush = _lib.lookupFunction<
    CactusTelemetryFlushNative,
    CactusTelemetryFlushDart>('cactus_telemetry_flush');
final _cactusTelemetryShutdown = _lib.lookupFunction<
    CactusTelemetryShutdownNative,
    CactusTelemetryShutdownDart>('cactus_telemetry_shutdown');
final _cactusSetAppId = _lib.lookupFunction<
    CactusSetAppIdNative,
    CactusSetAppIdDart>('cactus_set_app_id');
final _cactusIndexGet = _lib.lookupFunction<
    CactusIndexGetNative,
    CactusIndexGetDart>('cactus_index_get');
final _cactusDetectLanguage = _lib.lookupFunction<
    CactusDetectLanguageNative,
    CactusDetectLanguageDart>('cactus_detect_language');
final _cactusLogSetLevel = _lib.lookupFunction<
    CactusLogSetLevelNative,
    CactusLogSetLevelDart>('cactus_log_set_level');
final _cactusLogSetCallback = _lib.lookupFunction<
    CactusLogSetCallbackNative,
    CactusLogSetCallbackDart>('cactus_log_set_callback');

final class Utf8 extends Opaque {}

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

const malloc = _MallocAllocator();

class _MallocAllocator implements Allocator {
  const _MallocAllocator();

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

bool _frameworkSet = false;

void _ensureFramework() {
  if (!_frameworkSet) {
    final frameworkPtr = 'flutter'.toNativeUtf8();
    _cactusSetTelemetryEnvironment(frameworkPtr, nullptr, nullptr);
    calloc.free(frameworkPtr);
    _frameworkSet = true;
  }
}

/// Returns the last error message.
String cactusGetLastError() {
  return _cactusGetLastError().toDartString();
}

/// Sets the telemetry cache directory.
void cactusSetTelemetryEnvironment(String cacheLocation) {
  _ensureFramework();
  final pathPtr = cacheLocation.toNativeUtf8();
  _cactusSetTelemetryEnvironment(nullptr, pathPtr, nullptr);
  calloc.free(pathPtr);
}

/// Sets the application identifier for telemetry.
void cactusSetAppId(String appId) {
  final appIdPtr = appId.toNativeUtf8();
  _cactusSetAppId(appIdPtr);
  calloc.free(appIdPtr);
}

/// Flushes pending telemetry events.
void cactusTelemetryFlush() {
  _cactusTelemetryFlush();
}

/// Flushes and shuts down the telemetry subsystem.
void cactusTelemetryShutdown() {
  _cactusTelemetryShutdown();
}

/// Initializes a model from the given path.
CactusModelT cactusInit(String modelPath, String? corpusDir, bool cacheIndex) {
  _ensureFramework();
  final modelPathPtr = modelPath.toNativeUtf8();
  final corpusDirPtr = corpusDir?.toNativeUtf8() ?? nullptr;
  try {
    final handle = _cactusInit(modelPathPtr, corpusDirPtr, cacheIndex);
    if (handle == nullptr) {
      throw Exception('Failed to initialize model: ${cactusGetLastError()}');
    }
    return handle;
  } finally {
    calloc.free(modelPathPtr);
    if (corpusDirPtr != nullptr) calloc.free(corpusDirPtr);
  }
}

/// Frees all model resources.
void cactusDestroy(CactusModelT model) {
  _cactusDestroy(model);
}

/// Clears the KV cache.
void cactusReset(CactusModelT model) {
  _cactusReset(model);
}

/// Signals the current generation to stop.
void cactusStop(CactusModelT model) {
  _cactusStop(model);
}

/// Runs chat completion. Returns the assistant response.
String cactusComplete(
  CactusModelT model,
  String messagesJson,
  String? optionsJson,
  String? toolsJson,
  void Function(String token, int tokenId)? onToken,
) {
  const bufferSize = 65536;
  final responseBuffer = calloc<Uint8>(bufferSize);
  final messagesPtr = messagesJson.toNativeUtf8();
  final optionsPtr = optionsJson?.toNativeUtf8() ?? nullptr;
  final toolsPtr = toolsJson?.toNativeUtf8() ?? nullptr;

  Pointer<NativeFunction<TokenCallbackNative>> callbackPtr = nullptr;
  NativeCallable<TokenCallbackNative>? nativeCallable;
  if (onToken != null) {
    nativeCallable = NativeCallable<TokenCallbackNative>.isolateLocal(
      (Pointer<Utf8> token, int tokenId, Pointer<Void> _) {
        onToken(token.toDartString(), tokenId);
      },
    );
    callbackPtr = nativeCallable.nativeFunction;
  }

  try {
    final result = _cactusComplete(
      model, messagesPtr, responseBuffer.cast(), bufferSize,
      optionsPtr, toolsPtr, callbackPtr, nullptr,
    );
    if (result < 0) {
      throw Exception('Completion failed: ${cactusGetLastError()}');
    }
    return responseBuffer.cast<Utf8>().toDartString();
  } finally {
    calloc.free(responseBuffer);
    calloc.free(messagesPtr);
    if (optionsPtr != nullptr) calloc.free(optionsPtr);
    if (toolsPtr != nullptr) calloc.free(toolsPtr);
    nativeCallable?.close();
  }
}

/// Prefills the KV cache with messages.
String cactusPrefill(
  CactusModelT model,
  String messagesJson,
  String? optionsJson,
  String? toolsJson,
) {
  const bufferSize = 65536;
  final responseBuffer = calloc<Uint8>(bufferSize);
  final messagesPtr = messagesJson.toNativeUtf8();
  final optionsPtr = optionsJson?.toNativeUtf8() ?? nullptr;
  final toolsPtr = toolsJson?.toNativeUtf8() ?? nullptr;

  try {
    final result = _cactusPrefill(
      model, messagesPtr, responseBuffer.cast(), bufferSize,
      optionsPtr, toolsPtr,
    );
    if (result < 0) {
      throw Exception('Prefill failed: ${cactusGetLastError()}');
    }
    return responseBuffer.cast<Utf8>().toDartString();
  } finally {
    calloc.free(responseBuffer);
    calloc.free(messagesPtr);
    if (optionsPtr != nullptr) calloc.free(optionsPtr);
    if (toolsPtr != nullptr) calloc.free(toolsPtr);
  }
}

/// Transcribes audio to text.
String cactusTranscribe(
  CactusModelT model,
  String? audioPath,
  String? prompt,
  String? optionsJson,
  void Function(String token, int tokenId)? onToken,
  Uint8List? pcmData,
) {
  const bufferSize = 65536;
  final responseBuffer = calloc<Uint8>(bufferSize);
  final audioPathPtr = audioPath?.toNativeUtf8() ?? nullptr;
  final promptPtr = prompt?.toNativeUtf8() ?? nullptr;
  final optionsPtr = optionsJson?.toNativeUtf8() ?? nullptr;

  Pointer<Uint8> pcmBuffer = nullptr;
  int pcmSize = 0;
  if (pcmData != null) {
    pcmBuffer = calloc<Uint8>(pcmData.length);
    pcmBuffer.asTypedList(pcmData.length).setAll(0, pcmData);
    pcmSize = pcmData.length;
  }

  Pointer<NativeFunction<TokenCallbackNative>> callbackPtr = nullptr;
  NativeCallable<TokenCallbackNative>? nativeCallable;
  if (onToken != null) {
    nativeCallable = NativeCallable<TokenCallbackNative>.isolateLocal(
      (Pointer<Utf8> token, int tokenId, Pointer<Void> _) {
        onToken(token.toDartString(), tokenId);
      },
    );
    callbackPtr = nativeCallable.nativeFunction;
  }

  try {
    final result = _cactusTranscribe(
      model, audioPathPtr, promptPtr,
      responseBuffer.cast(), bufferSize,
      optionsPtr, callbackPtr, nullptr,
      pcmBuffer, pcmSize,
    );
    if (result < 0) {
      throw Exception('Transcription failed: ${cactusGetLastError()}');
    }
    return responseBuffer.cast<Utf8>().toDartString();
  } finally {
    calloc.free(responseBuffer);
    if (audioPathPtr != nullptr) calloc.free(audioPathPtr);
    if (promptPtr != nullptr) calloc.free(promptPtr);
    if (optionsPtr != nullptr) calloc.free(optionsPtr);
    if (pcmBuffer != nullptr) calloc.free(pcmBuffer);
    nativeCallable?.close();
  }
}

/// Generates a text embedding.
Float32List cactusEmbed(CactusModelT model, String text, bool normalize) {
  const maxDim = 4096;
  final embeddingsBuffer = calloc<Float>(maxDim);
  final embeddingDim = calloc<IntPtr>(1);
  final textPtr = text.toNativeUtf8();

  try {
    final result = _cactusEmbed(model, textPtr, embeddingsBuffer, maxDim, embeddingDim, normalize);
    if (result < 0) {
      throw Exception('Embedding failed: ${cactusGetLastError()}');
    }
    final dim = embeddingDim.value;
    return Float32List.fromList(embeddingsBuffer.asTypedList(dim));
  } finally {
    calloc.free(embeddingsBuffer);
    calloc.free(embeddingDim);
    calloc.free(textPtr);
  }
}

/// Generates an image embedding.
Float32List cactusImageEmbed(CactusModelT model, String imagePath) {
  const maxDim = 4096;
  final embeddingsBuffer = calloc<Float>(maxDim);
  final embeddingDim = calloc<IntPtr>(1);
  final imagePathPtr = imagePath.toNativeUtf8();

  try {
    final result = _cactusImageEmbed(model, imagePathPtr, embeddingsBuffer, maxDim, embeddingDim);
    if (result < 0) {
      throw Exception('Image embedding failed: ${cactusGetLastError()}');
    }
    final dim = embeddingDim.value;
    return Float32List.fromList(embeddingsBuffer.asTypedList(dim));
  } finally {
    calloc.free(embeddingsBuffer);
    calloc.free(embeddingDim);
    calloc.free(imagePathPtr);
  }
}

/// Generates an audio embedding.
Float32List cactusAudioEmbed(CactusModelT model, String audioPath) {
  const maxDim = 4096;
  final embeddingsBuffer = calloc<Float>(maxDim);
  final embeddingDim = calloc<IntPtr>(1);
  final audioPathPtr = audioPath.toNativeUtf8();

  try {
    final result = _cactusAudioEmbed(model, audioPathPtr, embeddingsBuffer, maxDim, embeddingDim);
    if (result < 0) {
      throw Exception('Audio embedding failed: ${cactusGetLastError()}');
    }
    final dim = embeddingDim.value;
    return Float32List.fromList(embeddingsBuffer.asTypedList(dim));
  } finally {
    calloc.free(embeddingsBuffer);
    calloc.free(embeddingDim);
    calloc.free(audioPathPtr);
  }
}

/// Runs voice activity detection. Returns JSON.
String cactusVad(
  CactusModelT model,
  String? audioPath,
  String? optionsJson,
  Uint8List? pcmData,
) {
  const bufferSize = 65536;
  final responseBuffer = calloc<Uint8>(bufferSize);
  final audioPathPtr = audioPath?.toNativeUtf8() ?? nullptr;
  final optionsPtr = optionsJson?.toNativeUtf8() ?? nullptr;

  Pointer<Uint8> pcmBuffer = nullptr;
  int pcmSize = 0;
  if (pcmData != null) {
    pcmBuffer = calloc<Uint8>(pcmData.length);
    pcmBuffer.asTypedList(pcmData.length).setAll(0, pcmData);
    pcmSize = pcmData.length;
  }

  try {
    final result = _cactusVad(
      model, audioPathPtr, responseBuffer.cast(), bufferSize, optionsPtr, pcmBuffer, pcmSize,
    );
    if (result < 0) {
      throw Exception('VAD failed: ${cactusGetLastError()}');
    }
    return responseBuffer.cast<Utf8>().toDartString();
  } finally {
    calloc.free(responseBuffer);
    if (audioPathPtr != nullptr) calloc.free(audioPathPtr);
    if (optionsPtr != nullptr) calloc.free(optionsPtr);
    if (pcmBuffer != nullptr) calloc.free(pcmBuffer);
  }
}

/// Queries the RAG corpus. Returns JSON.
String cactusRagQuery(CactusModelT model, String query, int topK) {
  const bufferSize = 65536;
  final responseBuffer = calloc<Uint8>(bufferSize);
  final queryPtr = query.toNativeUtf8();

  try {
    final result = _cactusRagQuery(model, queryPtr, responseBuffer.cast(), bufferSize, topK);
    if (result < 0) {
      throw Exception('RAG query failed: ${cactusGetLastError()}');
    }
    return responseBuffer.cast<Utf8>().toDartString();
  } finally {
    calloc.free(responseBuffer);
    calloc.free(queryPtr);
  }
}

/// Tokenizes text into token IDs.
List<int> cactusTokenize(CactusModelT model, String text) {
  const maxTokens = 8192;
  final tokenBuffer = calloc<Uint32>(maxTokens);
  final outTokenLen = calloc<IntPtr>(1);
  final textPtr = text.toNativeUtf8();

  try {
    final result = _cactusTokenize(model, textPtr, tokenBuffer, maxTokens, outTokenLen);
    if (result < 0) {
      throw Exception('Tokenization failed: ${cactusGetLastError()}');
    }
    final tokenCount = outTokenLen.value;
    return List<int>.generate(tokenCount, (i) => tokenBuffer[i]);
  } finally {
    calloc.free(tokenBuffer);
    calloc.free(outTokenLen);
    calloc.free(textPtr);
  }
}

/// Scores a window of tokens. Returns JSON.
String cactusScoreWindow(CactusModelT model, List<int> tokens, int start, int end, int context) {
  final tokenBuffer = calloc<Uint32>(tokens.length);
  for (var i = 0; i < tokens.length; i++) {
    tokenBuffer[i] = tokens[i];
  }
  const bufferSize = 65536;
  final responseBuffer = calloc<Uint8>(bufferSize);

  try {
    final result = _cactusScoreWindow(
      model, tokenBuffer, tokens.length, start, end, context, responseBuffer.cast(), bufferSize,
    );
    if (result < 0) {
      throw Exception('Score window failed: ${cactusGetLastError()}');
    }
    return responseBuffer.cast<Utf8>().toDartString();
  } finally {
    calloc.free(tokenBuffer);
    calloc.free(responseBuffer);
  }
}

/// Creates a streaming transcription session.
CactusStreamTranscribeT cactusStreamTranscribeStart(CactusModelT model, String? optionsJson) {
  final optionsPtr = optionsJson?.toNativeUtf8() ?? nullptr;
  try {
    final handle = _cactusStreamTranscribeStart(model, optionsPtr);
    if (handle == nullptr) {
      throw Exception('Failed to create stream transcriber: ${cactusGetLastError()}');
    }
    return handle;
  } finally {
    if (optionsPtr != nullptr) calloc.free(optionsPtr);
  }
}

/// Processes a chunk of PCM audio. Returns partial text.
String cactusStreamTranscribeProcess(CactusStreamTranscribeT stream, Uint8List pcmData) {
  const bufferSize = 65536;
  final pcmBuffer = calloc<Uint8>(pcmData.length);
  pcmBuffer.asTypedList(pcmData.length).setAll(0, pcmData);
  final responseBuffer = calloc<Uint8>(bufferSize);

  try {
    final result = _cactusStreamTranscribeProcess(
      stream, pcmBuffer, pcmData.length, responseBuffer.cast(), bufferSize,
    );
    if (result < 0) {
      throw Exception('Stream process failed: ${cactusGetLastError()}');
    }
    return responseBuffer.cast<Utf8>().toDartString();
  } finally {
    calloc.free(pcmBuffer);
    calloc.free(responseBuffer);
  }
}

/// Finalizes transcription and returns the result.
String cactusStreamTranscribeStop(CactusStreamTranscribeT stream) {
  const bufferSize = 65536;
  final responseBuffer = calloc<Uint8>(bufferSize);

  try {
    final result = _cactusStreamTranscribeStop(stream, responseBuffer.cast(), bufferSize);
    if (result < 0) {
      throw Exception('Stream stop failed: ${cactusGetLastError()}');
    }
    return responseBuffer.cast<Utf8>().toDartString();
  } finally {
    calloc.free(responseBuffer);
  }
}

/// Initializes a vector index in the given directory.
CactusIndexT cactusIndexInit(String indexDir, int embeddingDim) {
  final indexDirPtr = indexDir.toNativeUtf8();
  try {
    final handle = _cactusIndexInit(indexDirPtr, embeddingDim);
    if (handle == nullptr) {
      throw Exception('Failed to create index: ${cactusGetLastError()}');
    }
    return handle;
  } finally {
    calloc.free(indexDirPtr);
  }
}

/// Frees all index resources.
void cactusIndexDestroy(CactusIndexT index) {
  _cactusIndexDestroy(index);
}

/// Adds documents with embeddings to the index.
int cactusIndexAdd(
  CactusIndexT index,
  List<int> ids,
  List<String> documents,
  List<List<double>> embeddings,
  List<String>? metadatas,
) {
  final count = ids.length;
  final embDim = embeddings[0].length;

  final idsPtr = calloc<Int32>(count);
  for (var i = 0; i < count; i++) idsPtr[i] = ids[i];

  final documentsPtr = calloc<Pointer<Utf8>>(count);
  for (var i = 0; i < count; i++) documentsPtr[i] = documents[i].toNativeUtf8();

  final metadatasPtr = metadatas != null ? calloc<Pointer<Utf8>>(count) : nullptr;
  if (metadatas != null) {
    for (var i = 0; i < count; i++) metadatasPtr[i] = metadatas[i].toNativeUtf8();
  }

  final embeddingsPtr = calloc<Pointer<Float>>(count);
  for (var i = 0; i < count; i++) {
    final embPtr = calloc<Float>(embeddings[i].length);
    for (var j = 0; j < embeddings[i].length; j++) embPtr[j] = embeddings[i][j];
    embeddingsPtr[i] = embPtr;
  }

  try {
    final result = _cactusIndexAdd(index, idsPtr, documentsPtr, metadatasPtr, embeddingsPtr, count, embDim);
    if (result < 0) {
      throw Exception('Index add failed: ${cactusGetLastError()}');
    }
    return result;
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

/// Removes documents by ID.
int cactusIndexDelete(CactusIndexT index, List<int> ids) {
  final idsPtr = calloc<Int32>(ids.length);
  for (var i = 0; i < ids.length; i++) idsPtr[i] = ids[i];

  try {
    final result = _cactusIndexDelete(index, idsPtr, ids.length);
    if (result < 0) {
      throw Exception('Index delete failed: ${cactusGetLastError()}');
    }
    return result;
  } finally {
    calloc.free(idsPtr);
  }
}

/// Retrieves documents by ID. Returns JSON.
String cactusIndexGet(CactusIndexT index, List<int> ids) {
  final count = ids.length;
  if (count == 0) return '{"results":[]}';

  final idsPtr = calloc<Int32>(count);
  for (var i = 0; i < count; i++) idsPtr[i] = ids[i];

  const docBufSize = 4096;
  const embBufSize = 4096;

  final docRaw = List.generate(count, (_) => calloc<Uint8>(docBufSize));
  final metaRaw = List.generate(count, (_) => calloc<Uint8>(docBufSize));
  final embRaw = List.generate(count, (_) => calloc<Float>(embBufSize));

  final docBuffers = calloc<Pointer<Utf8>>(count);
  final docBufferSizes = calloc<IntPtr>(count);
  final metaBuffers = calloc<Pointer<Utf8>>(count);
  final metaBufferSizes = calloc<IntPtr>(count);
  final embBuffers = calloc<Pointer<Float>>(count);
  final embBufferSizes = calloc<IntPtr>(count);

  for (var i = 0; i < count; i++) {
    docBuffers[i] = docRaw[i].cast<Utf8>();
    docBufferSizes[i] = docBufSize;
    metaBuffers[i] = metaRaw[i].cast<Utf8>();
    metaBufferSizes[i] = docBufSize;
    embBuffers[i] = embRaw[i];
    embBufferSizes[i] = embBufSize;
  }

  try {
    final result = _cactusIndexGet(
      index, idsPtr, count,
      docBuffers, docBufferSizes,
      metaBuffers, metaBufferSizes,
      embBuffers, embBufferSizes,
    );
    if (result < 0) {
      throw Exception('Index get failed: ${cactusGetLastError()}');
    }

    final sb = StringBuffer('{"results":[');
    for (var i = 0; i < count; i++) {
      if (i > 0) sb.write(',');
      final doc = docBuffers[i].toDartString();
      final metaStr = metaBuffers[i].toDartString();
      final meta = metaStr.isNotEmpty ? metaStr : null;
      final embDim = embBufferSizes[i];
      sb.write('{"document":"$doc"');
      if (meta != null) sb.write(',"metadata":"$meta"'); else sb.write(',"metadata":null');
      sb.write(',"embedding":[');
      for (var j = 0; j < embDim; j++) {
        if (j > 0) sb.write(',');
        sb.write(embBuffers[i][j]);
      }
      sb.write(']}');
    }
    sb.write(']}');
    return sb.toString();
  } finally {
    calloc.free(idsPtr);
    for (var i = 0; i < count; i++) {
      calloc.free(docRaw[i]);
      calloc.free(metaRaw[i]);
      calloc.free(embRaw[i]);
    }
    calloc.free(docBuffers);
    calloc.free(docBufferSizes);
    calloc.free(metaBuffers);
    calloc.free(metaBufferSizes);
    calloc.free(embBuffers);
    calloc.free(embBufferSizes);
  }
}

/// Searches the index by embedding. Returns JSON.
String cactusIndexQuery(CactusIndexT index, List<double> embedding, String? optionsJson) {
  const resultCapacity = 1000;
  final embPtr = calloc<Float>(embedding.length);
  for (var i = 0; i < embedding.length; i++) embPtr[i] = embedding[i];

  final embPtrPtr = calloc<Pointer<Float>>(1);
  embPtrPtr[0] = embPtr;

  final optionsPtr = optionsJson?.toNativeUtf8() ?? nullptr;

  final idBuf = calloc<Int32>(resultCapacity);
  final scoreBuf = calloc<Float>(resultCapacity);
  final idBuffers = calloc<Pointer<Int32>>(1);
  final idBufferSizes = calloc<IntPtr>(1);
  final scoreBuffers = calloc<Pointer<Float>>(1);
  final scoreBufferSizes = calloc<IntPtr>(1);

  idBuffers[0] = idBuf;
  idBufferSizes[0] = resultCapacity;
  scoreBuffers[0] = scoreBuf;
  scoreBufferSizes[0] = resultCapacity;

  try {
    final result = _cactusIndexQuery(
      index, embPtrPtr, 1, embedding.length, optionsPtr,
      idBuffers, idBufferSizes, scoreBuffers, scoreBufferSizes,
    );
    if (result < 0) {
      throw Exception('Index query failed: ${cactusGetLastError()}');
    }

    final resultCount = idBufferSizes[0];
    final sb = StringBuffer('{"results":[');
    for (var i = 0; i < resultCount; i++) {
      if (i > 0) sb.write(',');
      sb.write('{"id":${idBuffers[0][i]},"score":${scoreBuffers[0][i]}}');
    }
    sb.write(']}');
    return sb.toString();
  } finally {
    calloc.free(embPtr);
    calloc.free(embPtrPtr);
    if (optionsPtr != nullptr) calloc.free(optionsPtr);
    calloc.free(idBuf);
    calloc.free(scoreBuf);
    calloc.free(idBuffers);
    calloc.free(idBufferSizes);
    calloc.free(scoreBuffers);
    calloc.free(scoreBufferSizes);
  }
}

/// Compacts the index storage.
int cactusIndexCompact(CactusIndexT index) {
  final result = _cactusIndexCompact(index);
  if (result < 0) {
    throw Exception('Index compact failed: ${cactusGetLastError()}');
  }
  return result;
}

/// Detects the spoken language in an audio file. Returns JSON.
String cactusDetectLanguage(
  CactusModelT model,
  String? audioPath,
  String? optionsJson,
  Uint8List? pcmData,
) {
  const bufferSize = 65536;
  final responseBuffer = calloc<Uint8>(bufferSize);
  final audioPathPtr = audioPath?.toNativeUtf8() ?? nullptr;
  final optionsPtr = optionsJson?.toNativeUtf8() ?? nullptr;

  Pointer<Uint8> pcmBuffer = nullptr;
  int pcmSize = 0;
  if (pcmData != null) {
    pcmBuffer = calloc<Uint8>(pcmData.length);
    pcmBuffer.asTypedList(pcmData.length).setAll(0, pcmData);
    pcmSize = pcmData.length;
  }

  try {
    final result = _cactusDetectLanguage(
      model, audioPathPtr, responseBuffer.cast(), bufferSize, optionsPtr, pcmBuffer, pcmSize,
    );
    if (result < 0) {
      throw Exception('Detect language failed: ${cactusGetLastError()}');
    }
    return responseBuffer.cast<Utf8>().toDartString();
  } finally {
    calloc.free(responseBuffer);
    if (audioPathPtr != nullptr) calloc.free(audioPathPtr);
    if (optionsPtr != nullptr) calloc.free(optionsPtr);
    if (pcmBuffer != nullptr) calloc.free(pcmBuffer);
  }
}

/// Sets the log level. 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR, 4=NONE.
void cactusLogSetLevel(int level) {
  _cactusLogSetLevel(level);
}

NativeCallable<LogCallbackNative>? _logCallable;

/// Sets a log callback. Pass null to clear.
void cactusLogSetCallback(void Function(int level, String component, String message)? onLog) {
  _logCallable?.close();
  _logCallable = null;
  if (onLog == null) {
    _cactusLogSetCallback(nullptr, nullptr);
    return;
  }
  _logCallable = NativeCallable<LogCallbackNative>.isolateLocal(
    (int level, Pointer<Utf8> component, Pointer<Utf8> message, Pointer<Void> _) {
      onLog(level, component.toDartString(), message.toDartString());
    },
  );
  _cactusLogSetCallback(_logCallable!.nativeFunction, nullptr);
}

