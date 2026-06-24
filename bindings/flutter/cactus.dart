import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

final DynamicLibrary _lib = (Platform.isAndroid || Platform.isLinux)
    ? DynamicLibrary.open('libcactus_engine.so')
    : DynamicLibrary.process();

typedef CactusModelT = Pointer<Void>;
typedef CactusIndexT = Pointer<Void>;
typedef CactusStreamTranscribeT = Pointer<Void>;

typedef TokenCallbackNative = Void Function(
    Pointer<Utf8> token, Uint32 tokenId, Pointer<Void> userData);
typedef LogCallbackNative = Void Function(
    Int32 level, Pointer<Utf8> component, Pointer<Utf8> message,
    Pointer<Void> userData);

typedef _CactusInitN = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Pointer<Utf8> corpusDir, Bool cacheIndex);
typedef _CactusInitD = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Pointer<Utf8> corpusDir, bool cacheIndex);
final cactusInit = _lib.lookupFunction<_CactusInitN, _CactusInitD>('cactus_init');

typedef _CactusDestroyN = Void Function(Pointer<Void> model);
typedef _CactusDestroyD = void Function(Pointer<Void> model);
final cactusDestroy =
    _lib.lookupFunction<_CactusDestroyN, _CactusDestroyD>('cactus_destroy');

typedef _CactusResetN = Void Function(Pointer<Void> model);
typedef _CactusResetD = void Function(Pointer<Void> model);
final cactusReset =
    _lib.lookupFunction<_CactusResetN, _CactusResetD>('cactus_reset');

typedef _CactusStopN = Void Function(Pointer<Void> model);
typedef _CactusStopD = void Function(Pointer<Void> model);
final cactusStop =
    _lib.lookupFunction<_CactusStopN, _CactusStopD>('cactus_stop');

typedef _CactusCompleteN = Int32 Function(
    Pointer<Void> model,
    Pointer<Utf8> messagesJson,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<Utf8> toolsJson,
    Pointer<NativeFunction<TokenCallbackNative>> callback,
    Pointer<Void> userData,
    Pointer<Uint8> pcmBuffer,
    IntPtr pcmBufferSize);
typedef _CactusCompleteD = int Function(
    Pointer<Void> model,
    Pointer<Utf8> messagesJson,
    Pointer<Utf8> responseBuffer,
    int bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<Utf8> toolsJson,
    Pointer<NativeFunction<TokenCallbackNative>> callback,
    Pointer<Void> userData,
    Pointer<Uint8> pcmBuffer,
    int pcmBufferSize);
final cactusComplete =
    _lib.lookupFunction<_CactusCompleteN, _CactusCompleteD>('cactus_complete');

typedef _CactusPrefillN = Int32 Function(
    Pointer<Void> model,
    Pointer<Utf8> messagesJson,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<Utf8> toolsJson,
    Pointer<Uint8> pcmBuffer,
    IntPtr pcmBufferSize);
typedef _CactusPrefillD = int Function(
    Pointer<Void> model,
    Pointer<Utf8> messagesJson,
    Pointer<Utf8> responseBuffer,
    int bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<Utf8> toolsJson,
    Pointer<Uint8> pcmBuffer,
    int pcmBufferSize);
final cactusPrefill =
    _lib.lookupFunction<_CactusPrefillN, _CactusPrefillD>('cactus_prefill');

typedef _CactusTokenizeN = Int32 Function(
    Pointer<Void> model,
    Pointer<Utf8> text,
    Pointer<Uint32> tokenBuffer,
    IntPtr tokenBufferLen,
    Pointer<IntPtr> outTokenLen);
typedef _CactusTokenizeD = int Function(
    Pointer<Void> model,
    Pointer<Utf8> text,
    Pointer<Uint32> tokenBuffer,
    int tokenBufferLen,
    Pointer<IntPtr> outTokenLen);
final cactusTokenize =
    _lib.lookupFunction<_CactusTokenizeN, _CactusTokenizeD>('cactus_tokenize');

typedef _CactusScoreWindowN = Int32 Function(
    Pointer<Void> model,
    Pointer<Uint32> tokens,
    IntPtr tokenLen,
    IntPtr start,
    IntPtr end,
    IntPtr context,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize);
typedef _CactusScoreWindowD = int Function(
    Pointer<Void> model,
    Pointer<Uint32> tokens,
    int tokenLen,
    int start,
    int end,
    int context,
    Pointer<Utf8> responseBuffer,
    int bufferSize);
final cactusScoreWindow =
    _lib.lookupFunction<_CactusScoreWindowN, _CactusScoreWindowD>(
        'cactus_score_window');

typedef _CactusTranscribeN = Int32 Function(
    Pointer<Void> model,
    Pointer<Utf8> audioFilePath,
    Pointer<Utf8> prompt,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<NativeFunction<TokenCallbackNative>> callback,
    Pointer<Void> userData,
    Pointer<Uint8> pcmBuffer,
    IntPtr pcmBufferSize);
typedef _CactusTranscribeD = int Function(
    Pointer<Void> model,
    Pointer<Utf8> audioFilePath,
    Pointer<Utf8> prompt,
    Pointer<Utf8> responseBuffer,
    int bufferSize,
    Pointer<Utf8> optionsJson,
    Pointer<NativeFunction<TokenCallbackNative>> callback,
    Pointer<Void> userData,
    Pointer<Uint8> pcmBuffer,
    int pcmBufferSize);
final cactusTranscribe =
    _lib.lookupFunction<_CactusTranscribeN, _CactusTranscribeD>(
        'cactus_transcribe');

typedef _CactusStreamTranscribeStartN = Pointer<Void> Function(
    Pointer<Void> model, Pointer<Utf8> optionsJson);
typedef _CactusStreamTranscribeStartD = Pointer<Void> Function(
    Pointer<Void> model, Pointer<Utf8> optionsJson);
final cactusStreamTranscribeStart = _lib.lookupFunction<
    _CactusStreamTranscribeStartN,
    _CactusStreamTranscribeStartD>('cactus_stream_transcribe_start');

typedef _CactusStreamTranscribeProcessN = Int32 Function(
    Pointer<Void> stream,
    Pointer<Uint8> pcmBuffer,
    IntPtr pcmBufferSize,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize);
typedef _CactusStreamTranscribeProcessD = int Function(
    Pointer<Void> stream,
    Pointer<Uint8> pcmBuffer,
    int pcmBufferSize,
    Pointer<Utf8> responseBuffer,
    int bufferSize);
final cactusStreamTranscribeProcess = _lib.lookupFunction<
    _CactusStreamTranscribeProcessN,
    _CactusStreamTranscribeProcessD>('cactus_stream_transcribe_process');

typedef _CactusStreamTranscribeStopN = Int32 Function(
    Pointer<Void> stream, Pointer<Utf8> responseBuffer, IntPtr bufferSize);
typedef _CactusStreamTranscribeStopD = int Function(
    Pointer<Void> stream, Pointer<Utf8> responseBuffer, int bufferSize);
final cactusStreamTranscribeStop = _lib.lookupFunction<
    _CactusStreamTranscribeStopN,
    _CactusStreamTranscribeStopD>('cactus_stream_transcribe_stop');

typedef _CactusEmbedN = Int32 Function(
    Pointer<Void> model,
    Pointer<Utf8> text,
    Pointer<Float> embeddingsBuffer,
    IntPtr bufferSize,
    Pointer<IntPtr> embeddingDim,
    Bool normalize);
typedef _CactusEmbedD = int Function(
    Pointer<Void> model,
    Pointer<Utf8> text,
    Pointer<Float> embeddingsBuffer,
    int bufferSize,
    Pointer<IntPtr> embeddingDim,
    bool normalize);
final cactusEmbed =
    _lib.lookupFunction<_CactusEmbedN, _CactusEmbedD>('cactus_embed');

typedef _CactusImageEmbedN = Int32 Function(
    Pointer<Void> model,
    Pointer<Utf8> imagePath,
    Pointer<Float> embeddingsBuffer,
    IntPtr bufferSize,
    Pointer<IntPtr> embeddingDim);
typedef _CactusImageEmbedD = int Function(
    Pointer<Void> model,
    Pointer<Utf8> imagePath,
    Pointer<Float> embeddingsBuffer,
    int bufferSize,
    Pointer<IntPtr> embeddingDim);
final cactusImageEmbed =
    _lib.lookupFunction<_CactusImageEmbedN, _CactusImageEmbedD>(
        'cactus_image_embed');

typedef _CactusAudioEmbedN = Int32 Function(
    Pointer<Void> model,
    Pointer<Utf8> audioPath,
    Pointer<Float> embeddingsBuffer,
    IntPtr bufferSize,
    Pointer<IntPtr> embeddingDim);
typedef _CactusAudioEmbedD = int Function(
    Pointer<Void> model,
    Pointer<Utf8> audioPath,
    Pointer<Float> embeddingsBuffer,
    int bufferSize,
    Pointer<IntPtr> embeddingDim);
final cactusAudioEmbed =
    _lib.lookupFunction<_CactusAudioEmbedN, _CactusAudioEmbedD>(
        'cactus_audio_embed');

typedef _CactusRagQueryN = Int32 Function(
    Pointer<Void> model,
    Pointer<Utf8> query,
    Pointer<Utf8> responseBuffer,
    IntPtr bufferSize,
    IntPtr topK);
typedef _CactusRagQueryD = int Function(
    Pointer<Void> model,
    Pointer<Utf8> query,
    Pointer<Utf8> responseBuffer,
    int bufferSize,
    int topK);
final cactusRagQuery =
    _lib.lookupFunction<_CactusRagQueryN, _CactusRagQueryD>(
        'cactus_rag_query');

typedef _CactusIndexInitN = Pointer<Void> Function(
    Pointer<Utf8> indexDir, IntPtr embeddingDim);
typedef _CactusIndexInitD = Pointer<Void> Function(
    Pointer<Utf8> indexDir, int embeddingDim);
final cactusIndexInit =
    _lib.lookupFunction<_CactusIndexInitN, _CactusIndexInitD>(
        'cactus_index_init');

typedef _CactusIndexAddN = Int32 Function(
    Pointer<Void> index,
    Pointer<Int32> ids,
    Pointer<Pointer<Utf8>> documents,
    Pointer<Pointer<Utf8>> metadatas,
    Pointer<Pointer<Float>> embeddings,
    IntPtr count,
    IntPtr embeddingDim);
typedef _CactusIndexAddD = int Function(
    Pointer<Void> index,
    Pointer<Int32> ids,
    Pointer<Pointer<Utf8>> documents,
    Pointer<Pointer<Utf8>> metadatas,
    Pointer<Pointer<Float>> embeddings,
    int count,
    int embeddingDim);
final cactusIndexAdd =
    _lib.lookupFunction<_CactusIndexAddN, _CactusIndexAddD>(
        'cactus_index_add');

typedef _CactusIndexDeleteN = Int32 Function(
    Pointer<Void> index, Pointer<Int32> ids, IntPtr idsCount);
typedef _CactusIndexDeleteD = int Function(
    Pointer<Void> index, Pointer<Int32> ids, int idsCount);
final cactusIndexDelete =
    _lib.lookupFunction<_CactusIndexDeleteN, _CactusIndexDeleteD>(
        'cactus_index_delete');

typedef _CactusIndexGetN = Int32 Function(
    Pointer<Void> index,
    Pointer<Int32> ids,
    IntPtr idsCount,
    Pointer<Pointer<Utf8>> documentBuffers,
    Pointer<IntPtr> documentBufferSizes,
    Pointer<Pointer<Utf8>> metadataBuffers,
    Pointer<IntPtr> metadataBufferSizes,
    Pointer<Pointer<Float>> embeddingBuffers,
    Pointer<IntPtr> embeddingBufferSizes);
typedef _CactusIndexGetD = int Function(
    Pointer<Void> index,
    Pointer<Int32> ids,
    int idsCount,
    Pointer<Pointer<Utf8>> documentBuffers,
    Pointer<IntPtr> documentBufferSizes,
    Pointer<Pointer<Utf8>> metadataBuffers,
    Pointer<IntPtr> metadataBufferSizes,
    Pointer<Pointer<Float>> embeddingBuffers,
    Pointer<IntPtr> embeddingBufferSizes);
final cactusIndexGet =
    _lib.lookupFunction<_CactusIndexGetN, _CactusIndexGetD>(
        'cactus_index_get');

typedef _CactusIndexQueryN = Int32 Function(
    Pointer<Void> index,
    Pointer<Pointer<Float>> embeddings,
    IntPtr embeddingsCount,
    IntPtr embeddingDim,
    Pointer<Utf8> optionsJson,
    Pointer<Pointer<Int32>> idBuffers,
    Pointer<IntPtr> idBufferSizes,
    Pointer<Pointer<Float>> scoreBuffers,
    Pointer<IntPtr> scoreBufferSizes);
typedef _CactusIndexQueryD = int Function(
    Pointer<Void> index,
    Pointer<Pointer<Float>> embeddings,
    int embeddingsCount,
    int embeddingDim,
    Pointer<Utf8> optionsJson,
    Pointer<Pointer<Int32>> idBuffers,
    Pointer<IntPtr> idBufferSizes,
    Pointer<Pointer<Float>> scoreBuffers,
    Pointer<IntPtr> scoreBufferSizes);
final cactusIndexQuery =
    _lib.lookupFunction<_CactusIndexQueryN, _CactusIndexQueryD>(
        'cactus_index_query');

typedef _CactusIndexCompactN = Int32 Function(Pointer<Void> index);
typedef _CactusIndexCompactD = int Function(Pointer<Void> index);
final cactusIndexCompact =
    _lib.lookupFunction<_CactusIndexCompactN, _CactusIndexCompactD>(
        'cactus_index_compact');

typedef _CactusIndexDestroyN = Void Function(Pointer<Void> index);
typedef _CactusIndexDestroyD = void Function(Pointer<Void> index);
final cactusIndexDestroy =
    _lib.lookupFunction<_CactusIndexDestroyN, _CactusIndexDestroyD>(
        'cactus_index_destroy');

typedef _CactusGetLastErrorN = Pointer<Utf8> Function();
typedef _CactusGetLastErrorD = Pointer<Utf8> Function();
final cactusGetLastError =
    _lib.lookupFunction<_CactusGetLastErrorN, _CactusGetLastErrorD>(
        'cactus_get_last_error');

typedef _CactusLogSetLevelN = Void Function(Int32 level);
typedef _CactusLogSetLevelD = void Function(int level);
final cactusLogSetLevel =
    _lib.lookupFunction<_CactusLogSetLevelN, _CactusLogSetLevelD>(
        'cactus_log_set_level');

typedef _CactusLogSetCallbackN = Void Function(
    Pointer<NativeFunction<LogCallbackNative>> callback,
    Pointer<Void> userData);
typedef _CactusLogSetCallbackD = void Function(
    Pointer<NativeFunction<LogCallbackNative>> callback,
    Pointer<Void> userData);
final cactusLogSetCallback =
    _lib.lookupFunction<_CactusLogSetCallbackN, _CactusLogSetCallbackD>(
        'cactus_log_set_callback');

typedef _CactusSetTelemetryEnvironmentN = Void Function(
    Pointer<Utf8> framework, Pointer<Utf8> cacheLocation,
    Pointer<Utf8> version);
typedef _CactusSetTelemetryEnvironmentD = void Function(
    Pointer<Utf8> framework, Pointer<Utf8> cacheLocation,
    Pointer<Utf8> version);
final cactusSetTelemetryEnvironment = _lib.lookupFunction<
    _CactusSetTelemetryEnvironmentN,
    _CactusSetTelemetryEnvironmentD>('cactus_set_telemetry_environment');

typedef _CactusSetAppIdN = Void Function(Pointer<Utf8> appId);
typedef _CactusSetAppIdD = void Function(Pointer<Utf8> appId);
final cactusSetAppId =
    _lib.lookupFunction<_CactusSetAppIdN, _CactusSetAppIdD>(
        'cactus_set_app_id');

typedef _CactusTelemetryFlushN = Void Function();
typedef _CactusTelemetryFlushD = void Function();
final cactusTelemetryFlush =
    _lib.lookupFunction<_CactusTelemetryFlushN, _CactusTelemetryFlushD>(
        'cactus_telemetry_flush');

typedef _CactusTelemetryShutdownN = Void Function();
typedef _CactusTelemetryShutdownD = void Function();
final cactusTelemetryShutdown =
    _lib.lookupFunction<_CactusTelemetryShutdownN, _CactusTelemetryShutdownD>(
        'cactus_telemetry_shutdown');
