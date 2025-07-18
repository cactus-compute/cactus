import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

import './bindings.dart' as bindings;
import './types.dart';
import './telemetry.dart';

CactusTokenCallback? _currentOnNewTokenCallback;

@pragma('vm:entry-point')
bool _staticTokenCallbackDispatcher(Pointer<Utf8> tokenC) {
  try {
    return _currentOnNewTokenCallback?.call(tokenC.toDartString()) ?? true;
  } catch (e) {
    debugPrint('Token callback error: $e');
    return false;
  }
}

class CactusContext {
  SendPort? _isolateSendPort;
  bool _disposed = false;
  late final CactusInitParams initParams;

  CactusContext._();

  static Future<CactusContext> init(CactusInitParams params) async {
    debugPrint('Starting CactusContext.init...');
    final context = CactusContext._();
    context.initParams = params;
    final resolvedParams = await _resolveParams(params);
    
    debugPrint('Spawning isolate...');
    final mainReceivePort = ReceivePort();
    await Isolate.spawn(_isolateEntry, mainReceivePort.sendPort);
    context._isolateSendPort = await mainReceivePort.first as SendPort;
    mainReceivePort.close();
    debugPrint('Isolate spawned, sending init command...');
    
    final result = await context._sendCommand(['init', _SendableInitParams.fromOriginal(resolvedParams)]);
    debugPrint('Init command result: $result');
    if (result is Exception) throw result;
    return context;
  }
  
  static Future<CactusInitParams> _resolveParams(CactusInitParams params) async {
    final appDocDir = await getApplicationDocumentsDirectory();
    
    String? modelPath = params.modelPath;
    String? mmprojPath = params.mmprojPath;
    
    if (params.modelUrl?.isNotEmpty == true) {
      final filename = params.modelFilename ?? (params.modelUrl!.split('/').last.isEmpty ? "downloaded_model.gguf" : params.modelUrl!.split('/').last);
      modelPath = '${appDocDir.path}/$filename';
      if (!await File(modelPath).exists()) {
        params.onInitProgress?.call(0.0, "Downloading model...", false);
        await _downloadModel(params.modelUrl!, modelPath, onProgress: (p, s) => params.onInitProgress?.call(p, "Model: $s", false));
        params.onInitProgress?.call(1.0, "Model download complete.", false);
      }
    }
    
    if (params.mmprojUrl?.isNotEmpty == true) {
      final filename = params.mmprojFilename ?? (params.mmprojUrl!.split('/').last.isEmpty ? "downloaded_mmproj.gguf" : params.mmprojUrl!.split('/').last);
      mmprojPath = '${appDocDir.path}/$filename';
      if (!await File(mmprojPath).exists()) {
        params.onInitProgress?.call(0.0, "Downloading mmproj...", false);
        await _downloadModel(params.mmprojUrl!, mmprojPath, onProgress: (p, s) => params.onInitProgress?.call(p, "MMProj: $s", false));
        params.onInitProgress?.call(1.0, "MMProj download complete.", false);
      }
    }
    
    if (modelPath?.isEmpty != false) throw ArgumentError('No modelPath or modelUrl provided');
    params.onInitProgress?.call(null, "Initializing...", false);
    
    return CactusInitParams(
      modelPath: modelPath, mmprojPath: mmprojPath, chatTemplate: params.chatTemplate,
      contextSize: params.contextSize, batchSize: params.batchSize, ubatchSize: params.ubatchSize,
      gpuLayers: params.gpuLayers, threads: params.threads, useMmap: params.useMmap, useMlock: params.useMlock,
      generateEmbeddings: params.generateEmbeddings, poolingType: params.poolingType, normalizeEmbeddings: params.normalizeEmbeddings,
      useFlashAttention: params.useFlashAttention, cacheTypeK: params.cacheTypeK, cacheTypeV: params.cacheTypeV,
      onInitProgress: params.onInitProgress,
    );
  }

  Future<CactusCompletionResult> completion(CactusCompletionParams params, {List<String> mediaPaths = const []}) async {
    _checkDisposed();
    final replyPort = ReceivePort();
    final completer = Completer<CactusCompletionResult>();

    final startTime = DateTime.now();
    DateTime? firstTokenTime;
    
    replyPort.listen((message) {
      if (message is String && params.onNewToken != null) {
        firstTokenTime ??= DateTime.now();
        params.onNewToken!(message);
      } else if (message is Map) {
        message['error'] != null ? completer.completeError(message['error']) : completer.complete(message['result']);
        replyPort.close();
      }
    });
    
    _isolateSendPort!.send(['completion', _SendableCompletionParams.fromOriginal(params), mediaPaths, replyPort.sendPort]);
    try{
      final result = await completer.future;
      final endTime = DateTime.now();
      final totalTime = endTime.difference(startTime).inMilliseconds;
      final tokPerSec = totalTime > 0 ? (result.tokensPredicted * 1000.0) / totalTime : null;
      final ttft = firstTokenTime?.difference(startTime).inMilliseconds;
      CactusTelemetry.track({
        'event': 'completion',
        'tok_per_sec': tokPerSec,
        'toks_generated': result.tokensPredicted,
        'ttft': ttft,
        'num_images': mediaPaths.length,
      }, initParams);
      return result;
    } catch (e) {
      CactusTelemetry.error(e, initParams);
      rethrow;
    }
  }

  Future<List<int>> tokenize(String text) => _sendCommand(['tokenize', text]);
  Future<String> detokenize(List<int> tokens) => _sendCommand(['detokenize', tokens]);
  Future<List<double>> embedding(String text) => _sendCommand(['embedding', text]);
  Future<BenchResult> bench({int pp = 512, int tg = 128, int pl = 1, int nr = 1}) => _sendCommand(['bench', pp, tg, pl, nr]);
  Future<void> initMultimodal(String mmprojPath, {bool useGpu = true}) => _sendCommand(['initMultimodal', mmprojPath, useGpu]);
  Future<bool> isMultimodalEnabled() => _sendCommand(['isMultimodalEnabled']);
  Future<bool> supportsVision() => _sendCommand(['supportsVision']);
  Future<bool> supportsAudio() => _sendCommand(['supportsAudio']);
  Future<void> releaseMultimodal() => _sendCommand(['releaseMultimodal']);
  Future<void> applyLoraAdapters(List<LoraAdapterInfo> adapters) => _sendCommand(['applyLoraAdapters', adapters]);
  Future<void> removeLoraAdapters() => _sendCommand(['removeLoraAdapters']);
  Future<List<LoraAdapterInfo>> getLoadedLoraAdapters() => _sendCommand(['getLoadedLoraAdapters']);
  Future<void> stopCompletion() => _sendCommand(['stopCompletion']);
  Future<void> rewind() => _sendCommand(['rewind']);

  Future<T> _sendCommand<T>(dynamic command) async {
    final replyPort = ReceivePort();
    if (command is List) {
      _isolateSendPort!.send([...command, replyPort.sendPort]);
    } else {
      _isolateSendPort!.send([command, replyPort.sendPort]);
    }
    final result = await replyPort.first;
    replyPort.close();
    if (result is Exception) throw result;
    return result as T;
  }

  void _checkDisposed() {
    if (_disposed) throw CactusException('CactusContext has been disposed');
  }

  void release() {
    if (!_disposed) {
      _isolateSendPort?.send(['dispose']);
      _isolateSendPort = null;
      _disposed = true;
    }
  }

  static Future<void> _isolateEntry(SendPort mainSendPort) async {
    final isolateReceivePort = ReceivePort();
    mainSendPort.send(isolateReceivePort.sendPort);
    _CactusIsolateWorker? worker;
    
    await for (final message in isolateReceivePort) {
      try {
        debugPrint('Isolate received message: ${message.runtimeType} - ${message is List ? message[0] : 'not a list'}');
        if (message is List) {
          final cmd = message[0];
          if (cmd == 'init') {
            debugPrint('Initializing worker...');
            worker = await _CactusIsolateWorker.init(message[1]);
            debugPrint('Worker initialized successfully');
            (message.last as SendPort).send(true);
          } else if (cmd == 'completion' && worker != null) {
            await worker.handleCompletion(message[1], message[2], message[3]);
          } else if (cmd == 'dispose') {
            worker?.dispose();
            break;
          } else if (worker != null) {
            final result = await worker.handleCommand(message);
            (message.last as SendPort).send(result);
          }
        }
      } catch (e) {
        debugPrint('Isolate error: $e');
        (message.last as SendPort).send(e);
      }
    }
    isolateReceivePort.close();
  }
}

class _CactusIsolateWorker {
  final bindings.CactusContextHandle _handle;
  
  _CactusIsolateWorker._(this._handle);
  
  static Future<_CactusIsolateWorker> init(_SendableInitParams params) async {
    if (params.modelPath?.isEmpty != false) throw ArgumentError('No modelPath provided');
    
    final cParams = calloc<bindings.CactusInitParamsC>();
    final modelPathC = params.modelPath!.toNativeUtf8(allocator: calloc);
    final chatTemplateC = params.chatTemplate?.toNativeUtf8(allocator: calloc);
    final cacheTypeKC = params.cacheTypeK?.toNativeUtf8(allocator: calloc);
    final cacheTypeVC = params.cacheTypeV?.toNativeUtf8(allocator: calloc);

    try {
      cParams.ref
        ..model_path = modelPathC
        ..chat_template = chatTemplateC ?? nullptr
        ..n_ctx = params.contextSize
        ..n_batch = params.batchSize
        ..n_ubatch = params.ubatchSize
        ..n_gpu_layers = params.gpuLayers
        ..n_threads = params.threads
        ..use_mmap = params.useMmap
        ..use_mlock = params.useMlock
        ..embedding = params.generateEmbeddings
        ..pooling_type = params.poolingType
        ..embd_normalize = params.normalizeEmbeddings
        ..flash_attn = params.useFlashAttention
        ..cache_type_k = cacheTypeKC ?? nullptr
        ..cache_type_v = cacheTypeVC ?? nullptr
        ..progress_callback = nullptr;

      final handle = bindings.initContext(cParams);
      if (handle == nullptr) throw CactusException('Failed to initialize native context');
      
      final worker = _CactusIsolateWorker._(handle);
      if (params.mmprojPath?.isNotEmpty == true) {
        await worker._initMultimodal(params.mmprojPath!, useGpu: params.gpuLayers != 0);
      }
      return worker;
    } finally {
      calloc.free(modelPathC);
      if (chatTemplateC != null) calloc.free(chatTemplateC);
      if (cacheTypeKC != null) calloc.free(cacheTypeKC);
      if (cacheTypeVC != null) calloc.free(cacheTypeVC);
      calloc.free(cParams);
    }
  }

  Future<void> handleCompletion(_SendableCompletionParams params, List<String> mediaPaths, SendPort replyPort) async {
    try {
      final cParams = params.toCactusCompletionParams();
      final promptString = await _getPromptString(cParams, mediaPaths);
      final paramsWithCallback = params.hasTokenCallback 
          ? cParams.copyWith(onNewToken: (token) { replyPort.send(token); return true; })
          : cParams;
      
      final result = await _performCompletion(promptString, paramsWithCallback, mediaPaths, replyPort);
      replyPort.send({'result': result});
    } catch (e) {
      replyPort.send({'error': e});
    }
  }

  Future<String> _getPromptString(CactusCompletionParams params, List<String> mediaPaths) async {
    if (params.responseFormat != null || params.jinja == true) {
      final result = await _getFormattedChatAdvanced(params);
      return result.prompt;
    } else {
      return await _getFormattedChat(params.messages, params.chatTemplate);
    }
  }

  Future<dynamic> handleCommand(List message) async {
    final cmd = message[0] as String;
    switch (cmd) {
      case 'tokenize': return _tokenize(message[1]);
      case 'detokenize': return _detokenize(message[1]);
      case 'embedding': return _embedding(message[1]);
      case 'bench': return _bench(pp: message[1], tg: message[2], pl: message[3], nr: message[4]);
      case 'initMultimodal': return _initMultimodal(message[1], useGpu: message[2]);
      case 'isMultimodalEnabled': return _isMultimodalEnabled();
      case 'supportsVision': return _supportsVision();
      case 'supportsAudio': return _supportsAudio();
      case 'releaseMultimodal': return _releaseMultimodal();
      case 'applyLoraAdapters': return _applyLoraAdapters(message[1]);
      case 'removeLoraAdapters': return _removeLoraAdapters();
      case 'getLoadedLoraAdapters': return _getLoadedLoraAdapters();
      case 'stopCompletion': return _stopCompletion();
      case 'rewind': return _rewind();
      default: throw ArgumentError('Unknown command: $cmd');
    }
  }

  Future<CactusCompletionResult> _performCompletion(String promptString, CactusCompletionParams params, List<String> mediaPaths, SendPort replyPort) async {
    final cCompParams = calloc<bindings.CactusCompletionParamsC>();
    final cResult = calloc<bindings.CactusCompletionResultC>();
    final promptC = promptString.toNativeUtf8(allocator: calloc);
    final grammarC = params.grammar?.toNativeUtf8(allocator: calloc);
    
    Pointer<Pointer<Utf8>> stopSequencesC = nullptr;
    Pointer<Pointer<Utf8>> mediaPathsC = nullptr;

    try {
      if (params.stopSequences?.isNotEmpty == true) {
        stopSequencesC = calloc<Pointer<Utf8>>(params.stopSequences!.length);
        for (int i = 0; i < params.stopSequences!.length; i++) {
          stopSequencesC[i] = params.stopSequences![i].toNativeUtf8(allocator: calloc);
        }
      }

      if (mediaPaths.isNotEmpty) {
        mediaPathsC = calloc<Pointer<Utf8>>(mediaPaths.length);
        for (int i = 0; i < mediaPaths.length; i++) {
          mediaPathsC[i] = mediaPaths[i].toNativeUtf8(allocator: calloc);
        }
      }

      _currentOnNewTokenCallback = params.onNewToken != null 
          ? (token) { replyPort.send(token); return true; }
          : null;

      cCompParams.ref
        ..prompt = promptC
        ..n_predict = params.maxPredictedTokens
        ..n_threads = params.threads ?? 4
        ..seed = params.seed ?? -1
        ..temperature = params.temperature ?? 0.7
        ..top_k = params.topK ?? 40
        ..top_p = params.topP ?? 0.9
        ..min_p = params.minP ?? 0.05
        ..typical_p = params.typicalP ?? 1.0
        ..penalty_last_n = params.penaltyLastN ?? 64
        ..penalty_repeat = params.penaltyRepeat ?? 1.1
        ..penalty_freq = params.penaltyFreq ?? 0.0
        ..penalty_present = params.penaltyPresent ?? 0.0
        ..mirostat = params.mirostat ?? 0
        ..mirostat_tau = params.mirostatTau ?? 5.0
        ..mirostat_eta = params.mirostatEta ?? 0.1
        ..ignore_eos = params.ignoreEos ?? false
        ..n_probs = params.nProbs ?? 0
        ..stop_sequences = stopSequencesC
        ..stop_sequence_count = params.stopSequences?.length ?? 0
        ..grammar = grammarC ?? nullptr
        ..token_callback = _currentOnNewTokenCallback != null 
            ? Pointer.fromFunction<Bool Function(Pointer<Utf8>)>(_staticTokenCallbackDispatcher, false)
            : nullptr;
      
      final status = mediaPaths.isEmpty 
          ? bindings.completion(_handle, cCompParams, cResult)
          : bindings.multimodalCompletion(_handle, cCompParams, mediaPathsC, mediaPaths.length, cResult);

      if (status != 0) throw CactusException('Native completion failed with status: $status');

      return CactusCompletionResult(
        text: cResult.ref.text.toDartString(),
        tokensPredicted: cResult.ref.tokens_predicted,
        tokensEvaluated: cResult.ref.tokens_evaluated,
        truncated: cResult.ref.truncated,
        stoppedEos: cResult.ref.stopped_eos,
        stoppedWord: cResult.ref.stopped_word,
        stoppedLimit: cResult.ref.stopped_limit,
        stoppingWord: cResult.ref.stopping_word.toDartString(),
      );
    } finally {
      _currentOnNewTokenCallback = null;
      calloc.free(promptC);
      if (grammarC != null) calloc.free(grammarC);
      
      if (mediaPathsC != nullptr) {
        for (int i = 0; i < mediaPaths.length; i++) {
          if (mediaPathsC[i] != nullptr) calloc.free(mediaPathsC[i]);
        }
        calloc.free(mediaPathsC);
      }
      
      if (stopSequencesC != nullptr) {
        for (int i = 0; i < (params.stopSequences?.length ?? 0); i++) {
          if (stopSequencesC[i] != nullptr) calloc.free(stopSequencesC[i]);
        }
        calloc.free(stopSequencesC);
      }
      
      if (cResult != nullptr) {
        bindings.freeCompletionResultMembers(cResult);
        calloc.free(cResult);
      }
      calloc.free(cCompParams);
    }
  }

  void _stopCompletion() => bindings.stopCompletion(_handle);
  void _rewind() => bindings.rewind(_handle);

  List<int> _tokenize(String text) {
    if (text.isEmpty) return [];
    final textC = text.toNativeUtf8(allocator: calloc);
    try {
      final cTokenArray = bindings.tokenize(_handle, textC);
      if (cTokenArray.tokens == nullptr || cTokenArray.count == 0) {
        bindings.freeTokenArray(cTokenArray);
        return [];
      }
      final tokens = List<int>.generate(cTokenArray.count, (i) => cTokenArray.tokens[i]);
      bindings.freeTokenArray(cTokenArray);
      return tokens;
    } finally {
      calloc.free(textC);
    }
  }

  String _detokenize(List<int> tokens) {
    if (tokens.isEmpty) return '';
    final tokensC = calloc<Int32>(tokens.length);
    try {
      for (int i = 0; i < tokens.length; i++) tokensC[i] = tokens[i];
      final result = bindings.detokenize(_handle, tokensC, tokens.length);
      final text = result.toDartString();
      bindings.freeString(result);
      return text;
    } finally {
      calloc.free(tokensC);
    }
  }

  List<double> _embedding(String text) {
    final textC = text.toNativeUtf8(allocator: calloc);
    try {
      final cEmbedding = bindings.embedding(_handle, textC);
      if (cEmbedding.values == nullptr || cEmbedding.count == 0) {
        bindings.freeFloatArray(cEmbedding);
        return [];
      }
      final embedding = List<double>.generate(cEmbedding.count, (i) => cEmbedding.values[i]);
      bindings.freeFloatArray(cEmbedding);
      return embedding;
    } finally {
      calloc.free(textC);
    }
  }

  BenchResult _bench({int pp = 512, int tg = 128, int pl = 1, int nr = 1}) {
    final cResult = bindings.bench(_handle, pp, tg, pl, nr);
    final result = BenchResult(
      modelDesc: cResult.model_name.toDartString(),
      modelSize: cResult.model_size,
      modelNParams: cResult.model_params,
      ppAvg: cResult.pp_avg, ppStd: cResult.pp_std,
      tgAvg: cResult.tg_avg, tgStd: cResult.tg_std,
    );
    bindings.freeString(cResult.model_name);
    return result;
  }

  Future<void> _initMultimodal(String mmprojPath, {bool useGpu = true}) async {
    final mmprojPathC = mmprojPath.toNativeUtf8(allocator: calloc);
    try {
      final status = bindings.initMultimodal(_handle, mmprojPathC, useGpu);
      if (status != 0) throw CactusException("Failed to initialize multimodal with status: $status");
    } finally {
      calloc.free(mmprojPathC);
    }
  }

  bool _isMultimodalEnabled() => bindings.isMultimodalEnabled(_handle);
  bool _supportsVision() => bindings.supportsVision(_handle);
  bool _supportsAudio() => bindings.supportsAudio(_handle);
  void _releaseMultimodal() => bindings.releaseMultimodal(_handle);

  void _applyLoraAdapters(List<LoraAdapterInfo> adapters) {
    final cAdaptersStruct = calloc<bindings.CactusLoraAdaptersC>();
    final cAdapters = calloc<bindings.CactusLoraAdapterC>(adapters.length);
    final pathPointers = <Pointer<Utf8>>[];
    
    try {
      for (int i = 0; i < adapters.length; i++) {
        final pathC = adapters[i].path.toNativeUtf8(allocator: calloc);
        pathPointers.add(pathC);
        cAdapters[i].path = pathC;
        cAdapters[i].scale = adapters[i].scale;
      }
      cAdaptersStruct.ref.adapters = cAdapters;
      cAdaptersStruct.ref.count = adapters.length;
      bindings.applyLoraAdapters(_handle, cAdaptersStruct);
    } finally {
      for (var p in pathPointers) {
        calloc.free(p);
      }
      calloc.free(cAdapters);
      calloc.free(cAdaptersStruct);
    }
  }

  void _removeLoraAdapters() => bindings.removeLoraAdapters(_handle);

  List<LoraAdapterInfo> _getLoadedLoraAdapters() {
    final cAdapters = bindings.getLoadedLoraAdapters(_handle);
    final adapters = List<LoraAdapterInfo>.generate(cAdapters.count, (i) => 
      LoraAdapterInfo(path: cAdapters.adapters[i].path.toDartString(), scale: cAdapters.adapters[i].scale));
    final cAdaptersPtr = calloc<bindings.CactusLoraAdaptersC>()..ref = cAdapters;
    bindings.freeLoraAdapters(cAdaptersPtr);
    calloc.free(cAdaptersPtr);
    return adapters;
  }

  void dispose() => bindings.freeContext(_handle);

  Future<String> _getFormattedChat(List<ChatMessage> messages, String? chatTemplate) async {
    final messagesJsonString = jsonEncode(messages.map((m) => m.toJson()).toList());
    final messagesJsonC = messagesJsonString.toNativeUtf8(allocator: calloc);
    final chatTemplateC = chatTemplate?.toNativeUtf8(allocator: calloc) ?? nullptr;
    final formattedPromptC = bindings.getFormattedChat(_handle, messagesJsonC, chatTemplateC);
    if (formattedPromptC == nullptr) throw CactusException("Native chat formatting returned null.");
    final prompt = formattedPromptC.toDartString();
    bindings.freeString(formattedPromptC);
    calloc.free(messagesJsonC);
    if (chatTemplateC != nullptr) calloc.free(chatTemplateC);
    return prompt;
  }

  Future<({String prompt, String? grammar})> _getFormattedChatAdvanced(CactusCompletionParams params) async {
    final finalTemplate = params.chatTemplate ?? 'chatml';
    final messagesC = jsonEncode(params.messages.map((m) => m.toJson()).toList()).toNativeUtf8(allocator: calloc);
    final finalTemplateC = finalTemplate.toNativeUtf8(allocator: calloc);
    final jsonSchemaC = (params.responseFormat?.schema != null) 
        ? jsonEncode(params.responseFormat!.schema).toNativeUtf8(allocator: calloc) 
        : nullptr;

    final resultC = bindings.getFormattedChatWithJinja(_handle, messagesC, finalTemplateC, jsonSchemaC, nullptr, false, nullptr);
    final resultCPtr = calloc<bindings.CactusChatResultC>()..ref = resultC;
    try {
      final promptString = resultC.prompt.toDartString();
      final grammar = resultC.json_schema.toDartString();
      return (prompt: promptString, grammar: grammar.isEmpty ? null : grammar);
    } finally {
      bindings.freeChatResultMembers(resultCPtr);
      calloc.free(resultCPtr);
      calloc.free(messagesC);
      calloc.free(finalTemplateC);
      if (jsonSchemaC != null) calloc.free(jsonSchemaC);
    }
  }
}

class _SendableInitParams {
  final String? modelPath;
  final String? modelUrl;
  final String? modelFilename;
  final String? mmprojPath;
  final String? mmprojUrl;
  final String? mmprojFilename;
  final String? chatTemplate;
  final int contextSize;
  final int batchSize;
  final int ubatchSize;
  final int gpuLayers;
  final int threads;
  final bool useMmap;
  final bool useMlock;
  final bool generateEmbeddings;
  final int poolingType;
  final int normalizeEmbeddings;
  final bool useFlashAttention;
  final String? cacheTypeK;
  final String? cacheTypeV;

  _SendableInitParams({
    this.modelPath,
    this.modelUrl,
    this.modelFilename,
    this.mmprojPath,
    this.mmprojUrl,
    this.mmprojFilename,
    this.chatTemplate,
    this.contextSize = 2048,
    this.batchSize = 512,
    this.ubatchSize = 512,
    this.gpuLayers = 0,
    this.threads = 4,
    this.useMmap = true,
    this.useMlock = false,
    this.generateEmbeddings = false,
    this.poolingType = 0,
    this.normalizeEmbeddings = 2,
    this.useFlashAttention = false,
    this.cacheTypeK,
    this.cacheTypeV,
  });

  factory _SendableInitParams.fromOriginal(CactusInitParams original) {
    return _SendableInitParams(
      modelPath: original.modelPath,
      modelUrl: original.modelUrl,
      modelFilename: original.modelFilename,
      mmprojPath: original.mmprojPath,
      mmprojUrl: original.mmprojUrl,
      mmprojFilename: original.mmprojFilename,
      chatTemplate: original.chatTemplate,
      contextSize: original.contextSize,
      batchSize: original.batchSize,
      ubatchSize: original.ubatchSize,
      gpuLayers: original.gpuLayers,
      threads: original.threads,
      useMmap: original.useMmap,
      useMlock: original.useMlock,
      generateEmbeddings: original.generateEmbeddings,
      poolingType: original.poolingType,
      normalizeEmbeddings: original.normalizeEmbeddings,
      useFlashAttention: original.useFlashAttention,
      cacheTypeK: original.cacheTypeK,
      cacheTypeV: original.cacheTypeV,
    );
  }
}

class _SendableCompletionParams {
  final List<ChatMessage> messages;
  final int maxPredictedTokens;
  final int? threads;
  final int? seed;
  final double? temperature;
  final int? topK;
  final double? topP;
  final double? minP;
  final double? typicalP;
  final int? penaltyLastN;
  final double? penaltyRepeat;
  final double? penaltyFreq;
  final double? penaltyPresent;
  final int? mirostat;
  final double? mirostatTau;
  final double? mirostatEta;
  final bool? ignoreEos;
  final int? nProbs;
  final List<String>? stopSequences;
  final String? grammar;
  final String? chatTemplate;
  final ResponseFormat? responseFormat;
  final bool? jinja;
  final bool hasTokenCallback;

  _SendableCompletionParams({
    required this.messages,
    this.maxPredictedTokens = 256,
    this.threads,
    this.seed,
    this.temperature,
    this.topK,
    this.topP,
    this.minP,
    this.typicalP,
    this.penaltyLastN,
    this.penaltyRepeat,
    this.penaltyFreq,
    this.penaltyPresent,
    this.mirostat,
    this.mirostatTau,
    this.mirostatEta,
    this.ignoreEos,
    this.nProbs,
    this.stopSequences,
    this.grammar,
    this.chatTemplate,
    this.responseFormat,
    this.jinja,
    this.hasTokenCallback = false,
  });

  factory _SendableCompletionParams.fromOriginal(CactusCompletionParams original) {
    return _SendableCompletionParams(
      messages: original.messages,
      maxPredictedTokens: original.maxPredictedTokens,
      threads: original.threads,
      seed: original.seed,
      temperature: original.temperature,
      topK: original.topK,
      topP: original.topP,
      minP: original.minP,
      typicalP: original.typicalP,
      penaltyLastN: original.penaltyLastN,
      penaltyRepeat: original.penaltyRepeat,
      penaltyFreq: original.penaltyFreq,
      penaltyPresent: original.penaltyPresent,
      mirostat: original.mirostat,
      mirostatTau: original.mirostatTau,
      mirostatEta: original.mirostatEta,
      ignoreEos: original.ignoreEos,
      nProbs: original.nProbs,
      stopSequences: original.stopSequences,
      grammar: original.grammar,
      chatTemplate: original.chatTemplate,
      responseFormat: original.responseFormat,
      jinja: original.jinja,
      hasTokenCallback: original.onNewToken != null,
    );
  }

  CactusCompletionParams toCactusCompletionParams() {
    return CactusCompletionParams(
      messages: messages,
      maxPredictedTokens: maxPredictedTokens,
      threads: threads,
      seed: seed,
      temperature: temperature,
      topK: topK,
      topP: topP,
      minP: minP,
      typicalP: typicalP,
      penaltyLastN: penaltyLastN,
      penaltyRepeat: penaltyRepeat,
      penaltyFreq: penaltyFreq,
      penaltyPresent: penaltyPresent,
      mirostat: mirostat,
      mirostatTau: mirostatTau,
      mirostatEta: mirostatEta,
      ignoreEos: ignoreEos,
      nProbs: nProbs,
      stopSequences: stopSequences,
      grammar: grammar,
      chatTemplate: chatTemplate,
      responseFormat: responseFormat,
      jinja: jinja,
      // onNewToken will be set separately in the isolate
    );
  }
}

Future<void> _downloadModel(String url, String filePath, {Function(double, String)? onProgress}) async {
  final client = HttpClient();
  try {
    final request = await client.getUrl(Uri.parse(url));
    final response = await request.close();

    if (response.statusCode != 200) {
      throw Exception('Failed to download model: ${response.statusCode}');
    }

    final file = File(filePath);
    final sink = file.openWrite();
    
    final contentLength = response.contentLength;
    int downloaded = 0;

    await for (final chunk in response) {
      sink.add(chunk);
      downloaded += chunk.length;
      
      if (contentLength > 0 && onProgress != null) {
        final progress = downloaded / contentLength;
        onProgress(progress, '${(progress * 100).toStringAsFixed(1)}%');
      }
    }
    
    await sink.close();
  } finally {
    client.close();
  }
}