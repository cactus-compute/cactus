import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:path_provider/path_provider.dart';

import './ffi_bindings.dart' as bindings;
import './init_params.dart';
import './completion.dart';
import './model_downloader.dart';
import './chat.dart';
import './structs.dart';
import './tools.dart';

class CactusException implements Exception {
  final String message;
  final dynamic underlyingError;

  CactusException(this.message, [this.underlyingError]);

  @override
  String toString() {
    if (underlyingError != null) {
      return 'CactusException: $message (Caused by: $underlyingError)';
    }
    return 'CactusException: $message';
  }
}

class CactusInitializationException extends CactusException {
  CactusInitializationException(String message, [dynamic underlyingError])
      : super('Initialization failed: $message', underlyingError);
}

class CactusCompletionException extends CactusException {
  CactusCompletionException(String message, [dynamic underlyingError])
      : super('Completion failed: $message', underlyingError);
}

class CactusOperationException extends CactusException {
  CactusOperationException(String operation, String message, [dynamic underlyingError])
      : super('$operation failed: $message', underlyingError);
}

class CactusTTSException extends CactusException {
  CactusTTSException(String operation, String message, [dynamic underlyingError])
      : super('TTS $operation failed: $message', underlyingError);
}

CactusTokenCallback? _currentOnNewTokenCallback;

@pragma('vm:entry-point')
bool _staticTokenCallbackDispatcher(Pointer<Utf8> tokenC) {
  if (_currentOnNewTokenCallback != null) {
    try {
      final token = tokenC.toDartString();
      return _currentOnNewTokenCallback!(token);
    } catch (e) {
      // ignore: avoid_print
      print('Error in token callback: $e');
      return false;
    }
  }
  return true;
}

class CactusContext {
  final bindings.CactusContextHandle _handle;

  CactusContext._(this._handle);

  static Future<CactusContext> init(CactusInitParams params) async {
    String? effectiveModelPath = params.modelPath;
    String? effectiveMmprojPath = params.mmprojPath;

    try {
      if (params.modelUrl != null && params.modelUrl!.isNotEmpty) {
        final Directory appDocDir = await getApplicationDocumentsDirectory();
        String modelFilename = params.modelFilename ?? params.modelUrl!.split('/').last;
        if (modelFilename.isEmpty) modelFilename = "downloaded_model.gguf";
        effectiveModelPath = '${appDocDir.path}/$modelFilename';
        final modelFile = File(effectiveModelPath);

        if (!await modelFile.exists()) {
          params.onInitProgress?.call(0.0, "Downloading model from ${params.modelUrl}...", false);
          await downloadModel(
            params.modelUrl!,
            effectiveModelPath,
            onProgress: (progress, status) {
              params.onInitProgress?.call(progress, "Model: $status", false);
            },
          );
          params.onInitProgress?.call(1.0, "Model download complete.", false);
        } else {
          params.onInitProgress?.call(null, "Model found locally at $effectiveModelPath", false);
        }
      } else if (effectiveModelPath == null || effectiveModelPath.isEmpty) {
        throw ArgumentError('No modelPath or modelUrl provided in CactusInitParams.');
      }

      if (params.mmprojUrl != null && params.mmprojUrl!.isNotEmpty) {
        final Directory appDocDir = await getApplicationDocumentsDirectory();
        String mmprojFilename = params.mmprojFilename ?? params.mmprojUrl!.split('/').last;
        if (mmprojFilename.isEmpty) mmprojFilename = "downloaded_mmproj.gguf";
        effectiveMmprojPath = '${appDocDir.path}/$mmprojFilename';
        final mmprojFile = File(effectiveMmprojPath);

        if (!await mmprojFile.exists()) {
          params.onInitProgress?.call(0.0, "Downloading mmproj from ${params.mmprojUrl}...", false);
          await downloadModel(
            params.mmprojUrl!,
            effectiveMmprojPath,
            onProgress: (progress, status) {
              params.onInitProgress?.call(progress, "MMProj: $status", false);
            },
          );
          params.onInitProgress?.call(1.0, "MMProj download complete.", false);
        } else {
          params.onInitProgress?.call(null, "MMProj found locally at $effectiveMmprojPath", false);
        }
      }

      params.onInitProgress?.call(null, "Initializing native context...", false);

      final cParams = calloc<bindings.CactusInitParamsC>();
      final modelPathC = effectiveModelPath.toNativeUtf8(allocator: calloc);
      final chatTemplateForC = params.chatTemplate?.toNativeUtf8(allocator: calloc);
      final cacheTypeKC = params.cacheTypeK?.toNativeUtf8(allocator: calloc);
      final cacheTypeVC = params.cacheTypeV?.toNativeUtf8(allocator: calloc);

      try {
        cParams.ref.model_path = modelPathC;
        cParams.ref.chat_template = chatTemplateForC ?? nullptr;
        cParams.ref.n_ctx = params.contextSize;
        cParams.ref.n_batch = params.batchSize;
        cParams.ref.n_ubatch = params.ubatchSize;
        cParams.ref.n_gpu_layers = params.gpuLayers ?? 0;
        cParams.ref.n_threads = params.threads;
        cParams.ref.use_mmap = params.useMmap;
        cParams.ref.use_mlock = params.useMlock;
        cParams.ref.embedding = params.generateEmbeddings;
        cParams.ref.pooling_type = params.poolingType;
        cParams.ref.embd_normalize = params.normalizeEmbeddings;
        cParams.ref.flash_attn = params.useFlashAttention;
        cParams.ref.cache_type_k = cacheTypeKC ?? nullptr;
        cParams.ref.cache_type_v = cacheTypeVC ?? nullptr;
        cParams.ref.progress_callback = nullptr;

        final bindings.CactusContextHandle handle = bindings.initContext(cParams);

        if (handle == nullptr) {
          const msg = 'Failed to initialize native cactus context. Handle was null. Check native logs for details.';
          params.onInitProgress?.call(null, msg, true);
          throw CactusInitializationException(msg);
        }
        
        final context = CactusContext._(handle);

        if(effectiveMmprojPath != null && effectiveMmprojPath.isNotEmpty){
          await context.initMultimodal(effectiveMmprojPath, useGpu: params.gpuLayers != 0);
        }

        params.onInitProgress?.call(1.0, 'CactusContext initialized successfully.', false);
        return context;
      } catch (e) {
        final msg = 'Error during native context initialization: $e';
        params.onInitProgress?.call(null, msg, true);
        if (e is CactusException) rethrow;
        throw CactusInitializationException(msg, e);
      } finally {
        calloc.free(modelPathC);
        if (chatTemplateForC != null) calloc.free(chatTemplateForC);
        if (cacheTypeKC != null) calloc.free(cacheTypeKC);
        if (cacheTypeVC != null) calloc.free(cacheTypeVC);
        calloc.free(cParams);
      }
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusInitializationException("Error during initialization: ${e.toString()}", e);
    }
  }

  void free() {
    bindings.freeContext(_handle);
  }

  List<int> tokenize(String text) {
    if (text.isEmpty) return [];

    Pointer<Utf8> textC = nullptr;
    try {
      textC = text.toNativeUtf8(allocator: calloc);
      final cTokenArray = bindings.tokenize(_handle, textC);
      
      if (cTokenArray.tokens == nullptr || cTokenArray.count == 0) {
        bindings.freeTokenArray(cTokenArray);
        return [];
      }
      final dartTokens = List<int>.generate(cTokenArray.count, (i) => cTokenArray.tokens[i]);
      bindings.freeTokenArray(cTokenArray);
      return dartTokens;
    } catch (e) {
        throw CactusOperationException("Tokenization", "Native error during tokenization.", e);
    }
    finally {
      if (textC != nullptr) calloc.free(textC);
    }
  }

  String detokenize(List<int> tokens) {
    if (tokens.isEmpty) return "";

    Pointer<Int32> tokensCPtr = nullptr;
    Pointer<Utf8> resultCPtr = nullptr;
    try {
      tokensCPtr = calloc<Int32>(tokens.length);
      for (int i = 0; i < tokens.length; i++) {
        tokensCPtr[i] = tokens[i];
      }

      resultCPtr = bindings.detokenize(_handle, tokensCPtr, tokens.length);
      if (resultCPtr == nullptr) {
        return "";
      }
      final resultString = resultCPtr.toDartString();
      bindings.freeString(resultCPtr);
      resultCPtr = nullptr;
      return resultString;
    } catch (e) {
        throw CactusOperationException("Detokenization", "Native error during detokenization.", e);
    }
    finally {
      if (tokensCPtr != nullptr) calloc.free(tokensCPtr);
    }
  }

  List<double> embedding(String text) {
    if (text.isEmpty) return [];

    Pointer<Utf8> textC = nullptr;
    try {
      textC = text.toNativeUtf8(allocator: calloc);
      final cFloatArray = bindings.embedding(_handle, textC);

      if (cFloatArray.values == nullptr || cFloatArray.count == 0) {
        bindings.freeFloatArray(cFloatArray);
        return [];
      }
      final dartEmbeddings = List<double>.generate(cFloatArray.count, (i) => cFloatArray.values[i]);
      bindings.freeFloatArray(cFloatArray);
      return dartEmbeddings;
    } catch (e) {
        throw CactusOperationException("Embedding generation", "Native error during embedding generation.", e);
    }
    finally {
      if (textC != nullptr) calloc.free(textC);
    }
  }

  Future<CactusCompletionResult> completion(CactusCompletionParams params) async {
    Pointer<bindings.CactusCompletionParamsC> cCompParams = nullptr;
    Pointer<bindings.CactusCompletionResultC> cResult = nullptr;
    Pointer<Utf8> promptC = nullptr;
    Pointer<Utf8> grammarC = nullptr;
    Pointer<Pointer<Utf8>> stopSequencesC = nullptr;

    try {
      final String formattedPromptString = await _getFormattedChat(params.messages, params.chatTemplate);
      
      cCompParams = calloc<bindings.CactusCompletionParamsC>();
      cResult = calloc<bindings.CactusCompletionResultC>();
      promptC = formattedPromptString.toNativeUtf8(allocator: calloc);
      grammarC = params.grammar?.toNativeUtf8(allocator: calloc) ?? nullptr;

      if (params.stopSequences != null && params.stopSequences!.isNotEmpty) {
        stopSequencesC = calloc<Pointer<Utf8>>(params.stopSequences!.length);
        for (int i = 0; i < params.stopSequences!.length; i++) {
          stopSequencesC[i] = params.stopSequences![i].toNativeUtf8(allocator: calloc);
        }
      }

      _currentOnNewTokenCallback = params.onNewToken as bool Function(String)?;
      final tokenCallbackC = _currentOnNewTokenCallback != null
          ? Pointer.fromFunction<Bool Function(Pointer<Utf8>)>(_staticTokenCallbackDispatcher, false)
          : nullptr;

      cCompParams.ref.prompt = promptC;
      cCompParams.ref.n_predict = params.maxPredictedTokens;
      cCompParams.ref.n_threads = params.threads ?? 4;
      cCompParams.ref.seed = params.seed ?? -1;
      cCompParams.ref.temperature = params.temperature ?? 0.7;
      cCompParams.ref.top_k = params.topK ?? 40;
      cCompParams.ref.top_p = params.topP ?? 0.9;
      cCompParams.ref.min_p = params.minP ?? 0.05;
      cCompParams.ref.typical_p = params.typicalP ?? 1.0;
      cCompParams.ref.penalty_last_n = params.penaltyLastN ?? 64;
      cCompParams.ref.penalty_repeat = params.penaltyRepeat ?? 1.1;
      cCompParams.ref.penalty_freq = params.penaltyFreq ?? 0.0;
      cCompParams.ref.penalty_present = params.penaltyPresent ?? 0.0;
      cCompParams.ref.mirostat = params.mirostat ?? 0;
      cCompParams.ref.mirostat_tau = params.mirostatTau ?? 5.0;
      cCompParams.ref.mirostat_eta = params.mirostatEta ?? 0.1;
      cCompParams.ref.ignore_eos = params.ignoreEos ?? false;
      cCompParams.ref.n_probs = params.nProbs ?? 0;
      cCompParams.ref.stop_sequences = stopSequencesC;
      cCompParams.ref.stop_sequence_count = params.stopSequences?.length ?? 0;
      cCompParams.ref.grammar = grammarC;
      cCompParams.ref.token_callback = tokenCallbackC;
      
      final status = bindings.completion(_handle, cCompParams, cResult);

      if (status != 0) {
        throw CactusCompletionException('Native completion call failed with status: $status. Check native logs.');
      }

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
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusCompletionException("Error during completion setup or execution.", e);
    }
    finally {
      _currentOnNewTokenCallback = null; 
      if (promptC != nullptr) calloc.free(promptC);
      if (grammarC != nullptr) calloc.free(grammarC);
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
      if (cCompParams != nullptr) calloc.free(cCompParams);
    }
  }

  Future<CactusCompletionResult> multimodalCompletion(
    CactusCompletionParams params, 
    List<String> mediaPaths
  ) async {
    Pointer<bindings.CactusCompletionParamsC> cCompParams = nullptr;
    Pointer<bindings.CactusCompletionResultC> cResult = nullptr;
    Pointer<Utf8> promptC = nullptr;
    Pointer<Utf8> grammarC = nullptr;
    Pointer<Pointer<Utf8>> stopSequencesC = nullptr;
    Pointer<Pointer<Utf8>> mediaPathsC = nullptr;

    try {
      final String formattedPromptString = await _getFormattedChat(params.messages, params.chatTemplate);
      
      cCompParams = calloc<bindings.CactusCompletionParamsC>();
      cResult = calloc<bindings.CactusCompletionResultC>();
      promptC = formattedPromptString.toNativeUtf8(allocator: calloc);
      grammarC = params.grammar?.toNativeUtf8(allocator: calloc) ?? nullptr;

      if (mediaPaths.isNotEmpty) {
        mediaPathsC = calloc<Pointer<Utf8>>(mediaPaths.length);
        for (int i = 0; i < mediaPaths.length; i++) {
          mediaPathsC[i] = mediaPaths[i].toNativeUtf8(allocator: calloc);
        }
      }

      if (params.stopSequences != null && params.stopSequences!.isNotEmpty) {
        stopSequencesC = calloc<Pointer<Utf8>>(params.stopSequences!.length);
        for (int i = 0; i < params.stopSequences!.length; i++) {
          stopSequencesC[i] = params.stopSequences![i].toNativeUtf8(allocator: calloc);
        }
      }

      _currentOnNewTokenCallback = params.onNewToken as bool Function(String)?;
      final tokenCallbackC = _currentOnNewTokenCallback != null
          ? Pointer.fromFunction<Bool Function(Pointer<Utf8>)>(_staticTokenCallbackDispatcher, false)
          : nullptr;

      cCompParams.ref.prompt = promptC;
      cCompParams.ref.n_predict = params.maxPredictedTokens;
      cCompParams.ref.n_threads = params.threads ?? 4;
      cCompParams.ref.seed = params.seed ?? -1;
      cCompParams.ref.temperature = params.temperature ?? 0.7;
      cCompParams.ref.top_k = params.topK ?? 40;
      cCompParams.ref.top_p = params.topP ?? 0.9;
      cCompParams.ref.min_p = params.minP ?? 0.05;
      cCompParams.ref.typical_p = params.typicalP ?? 1.0;
      cCompParams.ref.penalty_last_n = params.penaltyLastN ?? 64;
      cCompParams.ref.penalty_repeat = params.penaltyRepeat ?? 1.1;
      cCompParams.ref.penalty_freq = params.penaltyFreq ?? 0.0;
      cCompParams.ref.penalty_present = params.penaltyPresent ?? 0.0;
      cCompParams.ref.mirostat = params.mirostat ?? 0;
      cCompParams.ref.mirostat_tau = params.mirostatTau ?? 5.0;
      cCompParams.ref.mirostat_eta = params.mirostatEta ?? 0.1;
      cCompParams.ref.ignore_eos = params.ignoreEos ?? false;
      cCompParams.ref.n_probs = params.nProbs ?? 0;
      cCompParams.ref.stop_sequences = stopSequencesC;
      cCompParams.ref.stop_sequence_count = params.stopSequences?.length ?? 0;
      cCompParams.ref.grammar = grammarC;
      cCompParams.ref.token_callback = tokenCallbackC;
      
      final status = bindings.multimodalCompletion(
        _handle, 
        cCompParams, 
        mediaPathsC, 
        mediaPaths.length, 
        cResult
      );

      if (status != 0) {
        throw CactusCompletionException('Native multimodal completion call failed with status: $status. Check native logs.');
      }

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
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusCompletionException("Error during multimodal completion setup or execution.", e);
    }
    finally {
      _currentOnNewTokenCallback = null; 
      if (promptC != nullptr) calloc.free(promptC);
      if (grammarC != nullptr) calloc.free(grammarC);
      
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
      if (cCompParams != nullptr) calloc.free(cCompParams);
    }
  }

  void stopCompletion() {
    bindings.stopCompletion(_handle);
  }

  Future<String> _getFormattedChat(List<ChatMessage> messages, String? overrideChatTemplate) async {
    Pointer<Utf8> messagesJsonC = nullptr;
    Pointer<Utf8> overrideChatTemplateC = nullptr;
    Pointer<Utf8> formattedPromptC = nullptr;
    try {
      final messagesJsonString = jsonEncode(messages.map((m) => m.toJson()).toList());
      messagesJsonC = messagesJsonString.toNativeUtf8(allocator: calloc);

      if (overrideChatTemplate != null && overrideChatTemplate.isNotEmpty) {
        overrideChatTemplateC = overrideChatTemplate.toNativeUtf8(allocator: calloc);
      }

      formattedPromptC = bindings.getFormattedChat(
        _handle, 
        messagesJsonC, 
        overrideChatTemplateC,
      );

      if (formattedPromptC == nullptr) {
        throw CactusOperationException("ChatFormatting", "Native chat formatting returned null.");
      }
      return formattedPromptC.toDartString();
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusOperationException("ChatFormatting", "Error during _getFormattedChat: ${e.toString()}", e);
    } finally {
      if (messagesJsonC != nullptr) calloc.free(messagesJsonC);
      if (overrideChatTemplateC != nullptr) calloc.free(overrideChatTemplateC);
      if (formattedPromptC != nullptr) bindings.freeString(formattedPromptC);
    }
  }

  Future<void> initMultimodal(String mmprojPath, {bool useGpu = true}) async {
    Pointer<Utf8> mmprojPathC = nullptr;
    
    try {
      mmprojPathC = mmprojPath.toNativeUtf8(allocator: calloc);
      final status = bindings.initMultimodal(_handle, mmprojPathC, useGpu);
      
      if (status != 0) {
        throw CactusOperationException("InitMultimodal", "Failed to initialize multimodal with status: $status");
      }
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusOperationException("InitMultimodal", "Error during multimodal initialization: ${e.toString()}", e);
    } finally {
      if (mmprojPathC != nullptr) calloc.free(mmprojPathC);
    }
  }

  bool isMultimodalEnabled() {
    return bindings.isMultimodalEnabled(_handle);
  }

  bool supportsVision() {
    return bindings.supportsVision(_handle);
  }

  bool supportsAudio() {
    return bindings.supportsAudio(_handle);
  }

  void releaseMultimodal() {
    bindings.releaseMultimodal(_handle);
  }

  Future<void> initVocoder(String vocoderModelPath) async {
    Pointer<Utf8> vocoderPathC = nullptr;
    
    try {
      vocoderPathC = vocoderModelPath.toNativeUtf8(allocator: calloc);
      final status = bindings.initVocoder(_handle, vocoderPathC);
      
      if (status != 0) {
        throw CactusTTSException("InitVocoder", "Failed to initialize vocoder with status: $status");
      }
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusTTSException("InitVocoder", "Error during vocoder initialization: ${e.toString()}", e);
    } finally {
      if (vocoderPathC != nullptr) calloc.free(vocoderPathC);
    }
  }

  bool isVocoderEnabled() {
    return bindings.isVocoderEnabled(_handle);
  }

  int getTTSType() {
    return bindings.getTTSType(_handle);
  }

  String getFormattedAudioCompletion(String speakerJsonStr, String textToSpeak) {
    Pointer<Utf8> speakerJsonC = nullptr;
    Pointer<Utf8> textC = nullptr;
    Pointer<Utf8> resultC = nullptr;
    
    try {
      speakerJsonC = speakerJsonStr.toNativeUtf8(allocator: calloc);
      textC = textToSpeak.toNativeUtf8(allocator: calloc);
      
      resultC = bindings.getFormattedAudioCompletion(_handle, speakerJsonC, textC);
      
      if (resultC == nullptr) {
        throw CactusTTSException("GetFormattedAudioCompletion", "Native call returned null");
      }
      
      return resultC.toDartString();
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusTTSException("GetFormattedAudioCompletion", "Error: ${e.toString()}", e);
    } finally {
      if (speakerJsonC != nullptr) calloc.free(speakerJsonC);
      if (textC != nullptr) calloc.free(textC);
      if (resultC != nullptr) bindings.freeString(resultC);
    }
  }

  List<int> getAudioGuideTokens(String textToSpeak) {
    Pointer<Utf8> textC = nullptr;
    
    try {
      textC = textToSpeak.toNativeUtf8(allocator: calloc);
      final cTokenArray = bindings.getAudioCompletionGuideTokens(_handle, textC);
      
      if (cTokenArray.tokens == nullptr || cTokenArray.count == 0) {
        bindings.freeTokenArray(cTokenArray);
        return [];
      }
      
      final dartTokens = List<int>.generate(cTokenArray.count, (i) => cTokenArray.tokens[i]);
      bindings.freeTokenArray(cTokenArray);
      return dartTokens;
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusTTSException("GetAudioGuideTokens", "Error: ${e.toString()}", e);
    } finally {
      if (textC != nullptr) calloc.free(textC);
    }
  }

  List<double> decodeAudioTokens(List<int> tokens) {
    if (tokens.isEmpty) return [];
    
    Pointer<Int32> tokensCPtr = nullptr;
    
    try {
      tokensCPtr = calloc<Int32>(tokens.length);
      for (int i = 0; i < tokens.length; i++) {
        tokensCPtr[i] = tokens[i];
      }
      
      final cFloatArray = bindings.decodeAudioTokens(_handle, tokensCPtr, tokens.length);
      
      if (cFloatArray.values == nullptr || cFloatArray.count == 0) {
        bindings.freeFloatArray(cFloatArray);
        return [];
      }
      
      final dartAudio = List<double>.generate(cFloatArray.count, (i) => cFloatArray.values[i]);
      bindings.freeFloatArray(cFloatArray);
      return dartAudio;
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusTTSException("DecodeAudioTokens", "Error: ${e.toString()}", e);
    } finally {
      if (tokensCPtr != nullptr) calloc.free(tokensCPtr);
    }
  }

  void releaseVocoder() {
    bindings.releaseVocoder(_handle);
  }

  void applyLoraAdapters(List<LoraAdapterInfo> adapters) {
    if (adapters.isEmpty) return;

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

      final status = bindings.applyLoraAdapters(_handle, cAdaptersStruct);

      if (status != 0) {
        throw CactusOperationException("ApplyLoraAdapters", "Failed to apply LoRA adapters with status: $status");
      }
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusOperationException("ApplyLoraAdapters", "Error: ${e.toString()}", e);
    } finally {
      calloc.free(cAdapters);
      for (var p in pathPointers) {
        calloc.free(p);
      }
      calloc.free(cAdaptersStruct);
    }
  }

  BenchResult bench({int pp = 512, int tg = 128, int pl = 1, int nr = 1}) {
    try {
      final cResult = bindings.bench(_handle, pp, tg, pl, nr);
      
      final result = BenchResult(
        modelDesc: cResult.model_name.toDartString(),
        modelSize: cResult.model_size,
        modelNParams: cResult.model_params,
        ppAvg: cResult.pp_avg,
        ppStd: cResult.pp_std,
        tgAvg: cResult.tg_avg,
        tgStd: cResult.tg_std,
      );

      bindings.freeString(cResult.model_name);
      return result;
    } catch (e) {
      throw CactusOperationException("Benchmarking", "Error during benchmarking: ${e.toString()}", e);
    }
  }

  void removeLoraAdapters() {
    bindings.removeLoraAdapters(_handle);
  }

  List<LoraAdapterInfo> getLoadedLoraAdapters() {
    final cAdapters = bindings.getLoadedLoraAdapters(_handle);
    final cAdaptersStructPtr = calloc<bindings.CactusLoraAdaptersC>()..ref = cAdapters;
    try {
      final adapters = <LoraAdapterInfo>[];
      for (int i = 0; i < cAdapters.count; i++) {
        adapters.add(LoraAdapterInfo(
          path: cAdapters.adapters[i].path.toDartString(),
          scale: cAdapters.adapters[i].scale,
        ));
      }
      return adapters;
    } finally {
      bindings.freeLoraAdapters(cAdaptersStructPtr);
      calloc.free(cAdaptersStructPtr);
    }
  }

  bool validateChatTemplate(String template, {bool useJinja = false}) {
    Pointer<Utf8> templateC = nullptr;
    try {
      templateC = template.toNativeUtf8(allocator: calloc);
      return bindings.validateChatTemplate(_handle, useJinja, templateC);
    } finally {
      if (templateC != nullptr) calloc.free(templateC);
    }
  }

  Future<AdvancedChatResult> getFormattedChatAdvanced(CactusCompletionParams params) async {
    final finalTemplate = params.chatTemplate ?? 'chatml';
    
    Pointer<Utf8> messagesC = nullptr;
    Pointer<Utf8> finalTemplateC = nullptr;
    Pointer<Utf8> jsonSchemaC = nullptr;
    Pointer<Utf8> toolsC = nullptr;
    Pointer<Utf8> toolChoiceC = nullptr;

    try {
      messagesC = jsonEncode(params.messages.map((m) => m.toJson()).toList()).toNativeUtf8(allocator: calloc);
      finalTemplateC = finalTemplate.toNativeUtf8(allocator: calloc);
      jsonSchemaC = (params.responseFormat?.schema != null) 
          ? jsonEncode(params.responseFormat!.schema).toNativeUtf8(allocator: calloc) 
          : nullptr;
      toolsC = (params.tools != null) ? params.tools!.toNativeUtf8(allocator: calloc) : nullptr;
      toolChoiceC = (params.toolChoice != null) ? params.toolChoice!.toNativeUtf8(allocator: calloc) : nullptr;

      final resultC = bindings.getFormattedChatWithJinja(
        _handle,
        messagesC,
        finalTemplateC,
        jsonSchemaC,
        toolsC,
        params.parallelToolCalls ?? false,
        toolChoiceC,
      );
      final resultCPtr = calloc<bindings.CactusChatResultC>()..ref = resultC;

      try {
        final dartResult = AdvancedChatResult(
          prompt: resultC.prompt.toDartString(),
          grammar: resultC.json_schema.toDartString(),
        );
        return dartResult;
      } finally {
        bindings.freeChatResultMembers(resultCPtr);
        calloc.free(resultCPtr);
      }
    } catch (e) {
      final prompt = await _getFormattedChat(params.messages, 'chatml');
      return AdvancedChatResult(prompt: prompt);
    } finally {
      if (messagesC != nullptr) calloc.free(messagesC);
      if (finalTemplateC != nullptr) calloc.free(finalTemplateC);
      if (jsonSchemaC != nullptr) calloc.free(jsonSchemaC);
      if (toolsC != nullptr) calloc.free(toolsC);
      if (toolChoiceC != nullptr) calloc.free(toolChoiceC);
    }
  }

  Future<CactusCompletionResult> completionAdvanced(CactusCompletionParams params) async {
    final chatResult = await getFormattedChatAdvanced(params);
    final newParams = params.copyWith(
      messages: [ChatMessage(role: 'user', content: chatResult.prompt)],
      grammar: chatResult.grammar,
    );
    return completion(newParams);
  }

  Future<CactusCompletionResult> completionSmart(
    CactusCompletionParams params, {
    List<String> mediaPaths = const [],
    Tools? tools,
    int recursionLimit = 3,
  }) async {
    if (tools != null) {
      return _completionWithTools(params, tools: tools, recursionLimit: recursionLimit);
    } else if (params.responseFormat != null || params.jinja == true) {
      return completionAdvanced(params);
    } else if (mediaPaths.isNotEmpty) {
      return multimodalCompletion(params, mediaPaths);
    }
    return completion(params);
  }

  Future<CactusCompletionResult> _completionWithTools(
    CactusCompletionParams params, {
    required Tools tools,
    int recursionLimit = 3,
    int recursionCount = 0,
  }) async {
    if (recursionCount >= recursionLimit) {
      return completion(params);
    }

    final currentMessages = ToolCalling.injectToolsIntoMessages(params.messages, tools);
    final newParams = params.copyWith(
      messages: currentMessages,
      tools: tools.getDefinitionsJson(),
    );
    
    final result = await completionAdvanced(newParams);
    final toolCalls = ToolCalling.parseToolCalls(result.text);

    if (toolCalls.isEmpty) {
      return result;
    }

    final toolExecutionResults = await Future.wait(toolCalls.map((call) => tools.executeTool(call.name, call.arguments)));
    
    List<ChatMessage> updatedMessages = [...currentMessages];
    for (int i = 0; i < toolCalls.length; i++) {
      final toolCall = toolCalls[i];
      final executionResult = toolExecutionResults[i];
      updatedMessages = ToolCalling.updateMessagesWithToolCall(updatedMessages, toolCall.name, toolCall.arguments, executionResult.toolOutput ?? executionResult.error);
    }

    return _completionWithTools(
      params.copyWith(messages: updatedMessages),
      tools: tools,
      recursionLimit: recursionLimit,
      recursionCount: recursionCount + 1,
    );
  }
}