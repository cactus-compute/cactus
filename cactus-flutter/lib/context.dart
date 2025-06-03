import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io'; 
import 'dart:math'; // For min() in FormattedChatResult.toString, will be moved

import 'package:ffi/ffi.dart';
import 'package:path_provider/path_provider.dart';

import './ffi_bindings.dart' as bindings;
import './init_params.dart';
import './completion.dart';
import './model_downloader.dart';
import './chat.dart';
import './tts_params.dart';
import './structs.dart'; // Added import for structs

// --- Custom Exception Classes ---

/// Base exception for Cactus-related errors.
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

/// Exception thrown during [CactusContext] initialization.
class CactusInitializationException extends CactusException {
  CactusInitializationException(String message, [dynamic underlyingError])
      : super('Initialization failed: $message', underlyingError);
}

/// Exception thrown if a provided model path is invalid or the model cannot be accessed.
class CactusModelPathException extends CactusInitializationException {
  CactusModelPathException(String message, [dynamic underlyingError])
      : super('Model path error: $message', underlyingError);
}

/// Exception thrown during a completion operation.
class CactusCompletionException extends CactusException {
  CactusCompletionException(String message, [dynamic underlyingError])
      : super('Completion failed: $message', underlyingError);
}

/// Exception thrown for general errors during Cactus operations like tokenization or embedding.
class CactusOperationException extends CactusException {
  CactusOperationException(String operation, String message, [dynamic underlyingError])
      : super('$operation failed: $message', underlyingError);
}

/// Exception thrown for errors during TTS operations.
class CactusTTSException extends CactusException {
  CactusTTSException(String operation, String message, [dynamic underlyingError])
      : super('TTS $operation failed: $message', underlyingError);
}

// --- End Custom Exception Classes ---

/// Internal callback management, not part of public API.
/// Stores the currently active token callback for a completion operation.
CactusTokenCallback? _currentOnNewTokenCallback;

/// Internal FFI dispatcher for token callbacks, not part of public API.
///
/// This static function is invoked by the native C code for each new token.
/// It then calls the Dart callback stored in [_currentOnNewTokenCallback].
///
/// [tokenC] is a C string (Pointer<Utf8>) containing the new token.
/// Returns `true` to continue generation, `false` to stop.
@pragma('vm:entry-point')
bool _staticTokenCallbackDispatcher(Pointer<Utf8> tokenC) {
  if (_currentOnNewTokenCallback != null) {
    try {
      final token = tokenC.toDartString();
      return _currentOnNewTokenCallback!(token);
    } catch (e) {
      // Stop generation if the Dart callback throws an error.
      print('Error in token callback: $e');
      return false;
    }
  }
  // Continue generation if no Dart callback is set (should not happen if streaming).
  return true;
}

/// Manages a loaded AI model instance and provides methods for interaction
/// with the underlying `cactus` native library.
///
/// Use [CactusContext.init] to create and initialize a context with model parameters.
/// This involves loading the model (from a local path or URL) and setting up
/// the native inference engine.
///
/// Once initialized, the context can be used for:
/// - [tokenize]: Converting text to model-specific tokens.
/// - [detokenize]: Converting tokens back to text.
/// - [embedding]: Generating numerical vector representations of text.
/// - [completion]: Performing text or chat completion.
///
/// It is crucial to call [free] when the context is no longer needed to release
/// the native resources (model, KV cache, etc.) and avoid memory leaks.
class CactusContext {
  /// Internal handle to the native `cactus_context_opaque`. Not for public use.
  final bindings.CactusContextHandle _handle;

  // Private constructor. Users should use the static `CactusContext.init()` method.
  CactusContext._(this._handle);

  /// Initializes a new [CactusContext] with the given [params].
  ///
  /// This is an asynchronous operation that involves:
  /// 1. Resolving the model path:
  ///    - If [CactusInitParams.modelUrl] is provided, the model is downloaded
  ///      to the application's documents directory (if not already present).
  ///      The [CactusInitParams.modelFilename] can be used to specify the local filename.
  ///    - If [CactusInitParams.modelPath] is provided, it's used directly.
  /// 2. Initializing the native `cactus` context with the model and other parameters
  ///    from [params] (e.g., context size, GPU layers, chat template).
  ///
  /// This can be a long-running operation, especially if a model needs to be
  /// downloaded or if the model is large. Use [CactusInitParams.onInitProgress]
  /// to monitor progress updates (download percentage, status messages).
  ///
  /// Throws:
  /// - [CactusModelPathException] if the model path is invalid, the file doesn't exist,
  ///   or download fails.
  /// - [CactusInitializationException] if native context initialization fails for other reasons.
  /// - [ArgumentError] if [params] are invalid (e.g., neither modelPath nor modelUrl provided).
  static Future<CactusContext> init(CactusInitParams params) async {
    String? effectiveModelPath = params.modelPath;
    String? effectiveMmprojPath = params.mmprojPath;

    try {
      // Handle main model download if URL is provided
      if (params.modelUrl != null && params.modelUrl!.isNotEmpty) {
        final Directory appDocDir = await getApplicationDocumentsDirectory();
        String modelFilename = params.modelFilename ?? params.modelUrl!.split('/').last;
        if (modelFilename.isEmpty) modelFilename = "downloaded_model.gguf"; // Fallback filename
        effectiveModelPath = '${appDocDir.path}/$modelFilename';
        final modelFile = File(effectiveModelPath!);

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

      // Handle multimodal projector download if URL is provided
      if (params.mmprojUrl != null && params.mmprojUrl!.isNotEmpty) {
        final Directory appDocDir = await getApplicationDocumentsDirectory();
        String mmprojFilename = params.mmprojFilename ?? params.mmprojUrl!.split('/').last;
        if (mmprojFilename.isEmpty) mmprojFilename = "downloaded_mmproj.gguf"; // Fallback filename
        effectiveMmprojPath = '${appDocDir.path}/$mmprojFilename';
        final mmprojFile = File(effectiveMmprojPath!);

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
      } // If mmprojUrl is null, effectiveMmprojPath (from params.mmprojPath) will be used as is, or null if not provided.

      params.onInitProgress?.call(null, "Initializing native context...", false);

      final cParams = calloc<bindings.CactusInitParamsC>();
      Pointer<Utf8> modelPathC = nullptr;
      Pointer<Utf8> mmprojPathC = nullptr;
      Pointer<Utf8> chatTemplateForC = nullptr;
      Pointer<Utf8> cacheTypeKC = nullptr;
      Pointer<Utf8> cacheTypeVC = nullptr;
      Pointer<NativeFunction<Void Function(Float)>> progressCallbackC = nullptr;

      try {
        modelPathC = effectiveModelPath.toNativeUtf8(allocator: calloc);
        
        if (effectiveMmprojPath != null && effectiveMmprojPath.isNotEmpty) {
          mmprojPathC = effectiveMmprojPath.toNativeUtf8(allocator: calloc);
        }
        
        // Now, pass the user-provided template, or nullptr to let native side use model's default.
        if (params.chatTemplate != null && params.chatTemplate!.isNotEmpty) {
          chatTemplateForC = params.chatTemplate!.toNativeUtf8(allocator: calloc);
        } else {
          chatTemplateForC = nullptr; // Let native side use its default if no template is supplied by user for init
        }
        
        cacheTypeKC = params.cacheTypeK?.toNativeUtf8(allocator: calloc) ?? nullptr;
        cacheTypeVC = params.cacheTypeV?.toNativeUtf8(allocator: calloc) ?? nullptr;

        cParams.ref.model_path = modelPathC;
        cParams.ref.mmproj_path = mmprojPathC ?? nullptr;
        cParams.ref.chat_template = chatTemplateForC;
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
        cParams.ref.cache_type_k = cacheTypeKC;
        cParams.ref.cache_type_v = cacheTypeVC;
        cParams.ref.progress_callback = progressCallbackC;
        cParams.ref.warmup = params.warmup;
        cParams.ref.mmproj_use_gpu = params.mmprojUseGpu;
        cParams.ref.main_gpu = params.mainGpu;

        final bindings.CactusContextHandle handle = bindings.initContext(cParams);

        if (handle == nullptr) {
          const msg = 'Failed to initialize native cactus context. Handle was null. Check native logs for details.';
          params.onInitProgress?.call(null, msg, true);
          throw CactusInitializationException(msg);
        }
        
        final context = CactusContext._(handle);
        params.onInitProgress?.call(1.0, 'CactusContext initialized successfully.', false);
        return context;
      } catch (e) {
        final msg = 'Error during native context initialization: $e';
        params.onInitProgress?.call(null, msg, true);
        if (e is CactusException) rethrow;
        throw CactusInitializationException(msg, e);
      } finally {
        if (modelPathC != nullptr) calloc.free(modelPathC);
        if (mmprojPathC != nullptr) calloc.free(mmprojPathC);
        if (chatTemplateForC != nullptr) calloc.free(chatTemplateForC);
        if (cacheTypeKC != nullptr) calloc.free(cacheTypeKC);
        if (cacheTypeVC != nullptr) calloc.free(cacheTypeVC);
        calloc.free(cParams);
      }
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusInitializationException("Error during initialization: ${e.toString()}", e);
    }
  }

  /// Releases the native resources associated with this context.
  ///
  /// This method **must** be called when the context is no longer needed to free
  /// the memory occupied by the model and other native structures. Failure to do so
  /// will result in memory leaks.
  void free() {
    bindings.freeContext(_handle);
  }

  /// Converts the given [text] into a list of tokens according to the loaded model's tokenizer.
  ///
  /// Tokens are numerical representations used by the model.
  ///
  /// [text] The input string to tokenize.
  /// Returns a list of integer token IDs.
  /// Returns an empty list if the input text is empty or tokenization fails.
  /// Throws [CactusOperationException] if a native error occurs.
  List<int> tokenize(String text) {
    if (text.isEmpty) return [];

    Pointer<Utf8> textC = nullptr;
    try {
      textC = text.toNativeUtf8(allocator: calloc);
      final cTokenArray = bindings.tokenize(_handle, textC);
      
      if (cTokenArray.tokens == nullptr || cTokenArray.count == 0) {
        bindings.freeTokenArray(cTokenArray); // Still need to free the struct itself
        return [];
      }
      final dartTokens = List<int>.generate(cTokenArray.count, (i) => cTokenArray.tokens[i]);
      bindings.freeTokenArray(cTokenArray); // Frees cTokenArray.tokens and the struct
      return dartTokens;
    } catch (e) {
        throw CactusOperationException("Tokenization", "Native error during tokenization.", e);
    }
    finally {
      if (textC != nullptr) calloc.free(textC);
    }
  }

  /// Converts a list of [tokens] back into a string using the model's vocabulary.
  ///
  /// [tokens] A list of integer token IDs.
  /// Returns the detokenized string.
  /// Returns an empty string if the input list is empty or detokenization fails.
  /// Throws [CactusOperationException] if a native error occurs.
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
      bindings.freeString(resultCPtr); // Important: free the C string
      resultCPtr = nullptr; // Avoid double free in finally
      return resultString;
    } catch (e) {
        throw CactusOperationException("Detokenization", "Native error during detokenization.", e);
    }
    finally {
      if (tokensCPtr != nullptr) calloc.free(tokensCPtr);
      // resultCPtr is freed within try if not null, or was null.
    }
  }

  /// Generates an embedding (a list of float values representing the semantic meaning)
  /// for the given [text].
  ///
  /// The model must have been initialized with [CactusInitParams.generateEmbeddings] set to true.
  /// The nature of the embedding (e.g., pooling strategy, normalization) is determined by
  /// [CactusInitParams.poolingType] and [CactusInitParams.normalizeEmbeddings].
  ///
  /// [text] The input string to embed.
  /// Returns a list of double-precision floating-point values representing the embedding.
  /// Returns an empty list if the input text is empty or embedding generation fails.
  /// Throws [CactusOperationException] if a native error occurs or if embedding mode was not enabled.
  List<double> embedding(String text) {
    if (text.isEmpty) return [];

    Pointer<Utf8> textC = nullptr;
    try {
      textC = text.toNativeUtf8(allocator: calloc);
      final cFloatArray = bindings.embedding(_handle, textC);

      if (cFloatArray.values == nullptr || cFloatArray.count == 0) {
        bindings.freeFloatArray(cFloatArray); // Free the struct
        // Check if context was initialized for embeddings. This is a best guess.
        // A more robust check would be if the native layer could return a specific error code.
        // For now, assume empty result might mean not initialized for embeddings or actual empty result.
        // Consider logging a warning or having params.generateEmbeddings available on the context.
        return [];
      }
      final dartEmbeddings = List<double>.generate(cFloatArray.count, (i) => cFloatArray.values[i]);
      bindings.freeFloatArray(cFloatArray); // Frees cFloatArray.values and the struct
      return dartEmbeddings;
    } catch (e) {
        throw CactusOperationException("Embedding generation", "Native error during embedding generation.", e);
    }
    finally {
      if (textC != nullptr) calloc.free(textC);
    }
  }

  /// Performs text or chat completion based on the provided [params].
  ///
  /// This is an asynchronous operation.
  ///
  /// It constructs a prompt from [CactusCompletionParams.messages] using a
  /// default ChatML-like format if a custom `chat_template` was not provided
  /// during [CactusContext.init]. If a `chat_template` was provided at init,
  /// the native layer handles prompt formatting.
  ///
  /// For streaming results (receiving tokens as they are generated), provide an
  /// [CactusCompletionParams.onNewToken] callback.
  ///
  /// [params] Parameters controlling the completion, including messages, sampling settings,
  /// grammar, and streaming callback.
  ///
  /// Returns a [Future] that completes with a [CactusCompletionResult] containing
  /// the generated text and other metadata.
  ///
  /// Throws [CactusCompletionException] if the native completion call fails or if
  /// an error occurs during parameter setup.
  Future<CactusCompletionResult> completion(CactusCompletionParams params) async {
    Pointer<bindings.CactusCompletionParamsC> cCompParams = nullptr;
    Pointer<bindings.CactusCompletionResultC> cResult = nullptr;
    Pointer<Utf8> promptC = nullptr;
    Pointer<Utf8> imagePathForCompletionParamsC = nullptr;
    Pointer<Utf8> grammarC = nullptr;
    Pointer<Pointer<Utf8>> stopSequencesC = nullptr;

    try {
      // Pass params.imagePath to _getFormattedChat
      final String formattedPromptString = await _getFormattedChat(params.messages, params.chatTemplate, params.imagePath);
      
      cCompParams = calloc<bindings.CactusCompletionParamsC>();
      cResult = calloc<bindings.CactusCompletionResultC>();
      promptC = formattedPromptString.toNativeUtf8(allocator: calloc);
      grammarC = params.grammar?.toNativeUtf8(allocator: calloc) ?? nullptr;

      // image_path in cactus_completion_params_c is still needed for the native completion function
      // if it does further processing with the image beyond just templating (e.g., loading pixel data).
      // If all image handling is done by cactus_get_formatted_chat_c and the image is embedded
      // as a token/placeholder, then params.imagePath might not be strictly needed here for the C FFI call to cactus_completion_c.
      // However, the current FFI struct `cactus_completion_params_c` includes `image_path`.
      // So we continue to pass it. The C++ `cactus_completion_c` should know whether to use it
      // if the prompt already contains image info from `cactus_get_formatted_chat_c`.
      if (params.imagePath != null && params.imagePath!.isNotEmpty) {
        imagePathForCompletionParamsC = params.imagePath!.toNativeUtf8(allocator: calloc);
      }

      if (params.stopSequences != null && params.stopSequences!.isNotEmpty) {
        stopSequencesC = calloc<Pointer<Utf8>>(params.stopSequences!.length);
        for (int i = 0; i < params.stopSequences!.length; i++) {
          stopSequencesC[i] = params.stopSequences![i].toNativeUtf8(allocator: calloc);
        }
      }

      Pointer<NativeFunction<Bool Function(Pointer<Utf8>)>> tokenCallbackC = nullptr;
      _currentOnNewTokenCallback = params.onNewToken; 
      if (params.onNewToken != null) {
        tokenCallbackC = Pointer.fromFunction<Bool Function(Pointer<Utf8>)>(_staticTokenCallbackDispatcher, false);
      }

      cCompParams.ref.prompt = promptC;
      cCompParams.ref.image_path = imagePathForCompletionParamsC ?? nullptr;
      cCompParams.ref.n_predict = params.maxPredictedTokens;
      cCompParams.ref.n_threads = params.threads;
      cCompParams.ref.seed = params.seed;
      cCompParams.ref.temperature = params.temperature;
      cCompParams.ref.top_k = params.topK;
      cCompParams.ref.top_p = params.topP;
      cCompParams.ref.min_p = params.minP;
      cCompParams.ref.typical_p = params.typicalP;
      cCompParams.ref.penalty_last_n = params.penaltyLastN;
      cCompParams.ref.penalty_repeat = params.penaltyRepeat;
      cCompParams.ref.penalty_freq = params.penaltyFreq;
      cCompParams.ref.penalty_present = params.penaltyPresent;
      cCompParams.ref.mirostat = params.mirostat;
      cCompParams.ref.mirostat_tau = params.mirostatTau;
      cCompParams.ref.mirostat_eta = params.mirostatEta;
      cCompParams.ref.ignore_eos = params.ignoreEos;
      cCompParams.ref.n_probs = params.nProbs;
      cCompParams.ref.stop_sequences = stopSequencesC ?? nullptr;
      cCompParams.ref.stop_sequence_count = params.stopSequences?.length ?? 0;
      cCompParams.ref.grammar = grammarC;
      cCompParams.ref.token_callback = tokenCallbackC ?? nullptr;
      
      final status = bindings.completion(_handle, cCompParams, cResult);

      if (status != 0) {
        final msg = 'Native completion call failed with status: $status. Check native logs.';
        throw CactusCompletionException(msg);
      }

      final result = CactusCompletionResult(
        text: cResult.ref.text.toDartString(),
        tokensPredicted: cResult.ref.tokens_predicted,
        tokensEvaluated: cResult.ref.tokens_evaluated,
        truncated: cResult.ref.truncated,
        stoppedEos: cResult.ref.stopped_eos,
        stoppedWord: cResult.ref.stopped_word,
        stoppedLimit: cResult.ref.stopped_limit,
        stoppingWord: cResult.ref.stopping_word.toDartString(),
        generationTimeUs: cResult.ref.generation_time_us,
      );

      return result;
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusCompletionException("Error during completion setup or execution.", e);
    }
    finally {
      _currentOnNewTokenCallback = null; 

      if (promptC != nullptr) calloc.free(promptC);
      if (imagePathForCompletionParamsC != nullptr) calloc.free(imagePathForCompletionParamsC);
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

  /// Asynchronously requests the current completion operation to stop.
  ///
  /// This provides a way to interrupt a long-running generation.
  /// The actual stopping is handled by the native side and may not be immediate,
  /// as the native code needs to reach a point where it checks for the interrupt flag.
  ///
  /// This method is non-blocking.
  void stopCompletion() {
    bindings.stopCompletion(_handle);
  }

  // +++ New Helper Method +++
  Future<String> _getFormattedChat(List<ChatMessage> messages, String? overrideChatTemplate, String? imagePath) async {
    Pointer<Utf8> messagesJsonC = nullptr;
    Pointer<Utf8> overrideChatTemplateC = nullptr;
    Pointer<Utf8> imagePathC = nullptr;
    Pointer<Utf8> formattedPromptC = nullptr;
    try {
      final messagesJsonString = jsonEncode(messages.map((m) => m.toJson()).toList());
      messagesJsonC = messagesJsonString.toNativeUtf8(allocator: calloc);

      if (overrideChatTemplate != null && overrideChatTemplate.isNotEmpty) {
        overrideChatTemplateC = overrideChatTemplate.toNativeUtf8(allocator: calloc);
      } else {
        overrideChatTemplateC = nullptr;
      }

      if (imagePath != null && imagePath.isNotEmpty) {
        imagePathC = imagePath.toNativeUtf8(allocator: calloc);
      } else {
        imagePathC = nullptr;
      }

      formattedPromptC = bindings.getFormattedChat(
        _handle, 
        messagesJsonC, 
        overrideChatTemplateC ?? nullptr,
        imagePathC ?? nullptr
      );

      if (formattedPromptC == nullptr) {
        throw CactusOperationException("ChatFormatting", "Native chat formatting returned null. Ensure 'cactus_get_formatted_chat_c' is implemented correctly in native code and handles parameters.");
      }
      final formattedPrompt = formattedPromptC.toDartString();
      return formattedPrompt;
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusOperationException("ChatFormatting", "Error during _getFormattedChat: ${e.toString()}", e);
    } finally {
      if (messagesJsonC != nullptr) calloc.free(messagesJsonC);
      if (overrideChatTemplateC != nullptr) calloc.free(overrideChatTemplateC);
      if (imagePathC != nullptr) calloc.free(imagePathC);
      if (formattedPromptC != nullptr) bindings.freeString(formattedPromptC);
    }
  }
  // +++ End New Helper Method +++

  // --- New TTS Methods ---

  /// Loads the vocoder model required for Text-to-Speech operations.
  ///
  /// This should be called after [CactusContext.init] if TTS is needed.
  /// The main TTS model (if separate) should be loaded via [CactusContext.init].
  ///
  /// [vocoderParams] Parameters for loading the vocoder model.
  /// Throws [CactusTTSException] if loading fails.
  Future<void> loadVocoder(VocoderLoadParams vocoderParams) async {
    Pointer<bindings.CactusVocoderLoadParamsC> cLoadParams = nullptr;
    Pointer<Utf8> vocoderModelPathC = nullptr;
    Pointer<Utf8> speakerFileC = nullptr;

    try {
      cLoadParams = calloc<bindings.CactusVocoderLoadParamsC>();
      
      // VocoderModelParams is nested, so we need to allocate its path separately
      vocoderModelPathC = vocoderParams.modelParams.path.toNativeUtf8(allocator: calloc);
      cLoadParams.ref.model_params.path = vocoderModelPathC;

      if (vocoderParams.speakerFile != null && vocoderParams.speakerFile!.isNotEmpty) {
        speakerFileC = vocoderParams.speakerFile!.toNativeUtf8(allocator: calloc);
      }
      cLoadParams.ref.speaker_file = speakerFileC ?? nullptr;
      cLoadParams.ref.use_guide_tokens = vocoderParams.useGuideTokens;

      final status = bindings.loadVocoder(_handle, cLoadParams);
      if (status != 0) {
        throw CactusTTSException("LoadVocoder", "Native vocoder loading failed with status: $status. Check native logs.");
      }
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusTTSException("LoadVocoder", "Error during vocoder loading: ${e.toString()}", e);
    } finally {
      if (vocoderModelPathC != nullptr) calloc.free(vocoderModelPathC);
      if (speakerFileC != nullptr) calloc.free(speakerFileC);
      if (cLoadParams != nullptr) calloc.free(cLoadParams);
    }
  }

  /// Synthesizes speech from the given text and saves it to a WAV file.
  ///
  /// Both the main TTS model (via [CactusContext.init]) and the vocoder model
  /// (via [loadVocoder]) must be loaded before calling this.
  ///
  /// [ttsParams] Parameters for synthesis, including input text and output path.
  /// Throws [CactusTTSException] if synthesis fails.
  Future<void> synthesizeSpeech(SynthesizeSpeechParams ttsParams) async {
    Pointer<bindings.CactusSynthesizeSpeechParamsC> cSynthParams = nullptr;
    Pointer<Utf8> textInputC = nullptr;
    Pointer<Utf8> outputWavPathC = nullptr;
    Pointer<Utf8> speakerIdC = nullptr;

    try {
      cSynthParams = calloc<bindings.CactusSynthesizeSpeechParamsC>();

      textInputC = ttsParams.textInput.toNativeUtf8(allocator: calloc);
      outputWavPathC = ttsParams.outputWavPath.toNativeUtf8(allocator: calloc);
      if (ttsParams.speakerId != null && ttsParams.speakerId!.isNotEmpty) {
        speakerIdC = ttsParams.speakerId!.toNativeUtf8(allocator: calloc);
      }

      cSynthParams.ref.text_input = textInputC;
      cSynthParams.ref.output_wav_path = outputWavPathC;
      cSynthParams.ref.speaker_id = speakerIdC ?? nullptr;

      final status = bindings.synthesizeSpeech(_handle, cSynthParams);
      if (status != 0) {
        throw CactusTTSException("SynthesizeSpeech", "Native speech synthesis failed with status: $status. Check native logs.");
      }
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusTTSException("SynthesizeSpeech", "Error during speech synthesis: ${e.toString()}", e);
    } finally {
      if (textInputC != nullptr) calloc.free(textInputC);
      if (outputWavPathC != nullptr) calloc.free(outputWavPathC);
      if (speakerIdC != nullptr) calloc.free(speakerIdC);
      if (cSynthParams != nullptr) calloc.free(cSynthParams);
    }
  }
  // --- End New TTS Methods ---

  // +++ LoRA Adapter Management +++

  /// Applies a list of LoRA adapters to the model.
  /// Throws [CactusOperationException] on failure.
  void applyLoraAdapters(List<LoraAdapterInfo> adapters) {
    if (adapters.isEmpty) return;

    final Pointer<bindings.CactusLoraAdapterInfoC> cAdapters = calloc<bindings.CactusLoraAdapterInfoC>(adapters.length);
    List<Pointer<Utf8>> pathPointers = [];

    try {
      for (int i = 0; i < adapters.length; i++) {
        final pathC = adapters[i].path.toNativeUtf8(allocator: calloc);
        pathPointers.add(pathC);
        cAdapters[i].path = pathC;
        cAdapters[i].scale = adapters[i].scale;
      }

      final status = bindings.applyLoraAdapters(_handle, cAdapters, adapters.length);
      if (status != 0) {
        throw CactusOperationException("ApplyLoRA", "Native LoRA application failed with status: $status");
      }
    } finally {
      for (final ptr in pathPointers) {
        calloc.free(ptr);
      }
      calloc.free(cAdapters);
    }
  }

  /// Removes all currently applied LoRA adapters from the model.
  void removeLoraAdapters() {
    bindings.removeLoraAdapters(_handle);
  }

  /// Retrieves information about currently loaded LoRA adapters.
  /// Returns an empty list if no adapters are loaded or on error.
  List<LoraAdapterInfo> getLoadedLoraAdapters() {
    Pointer<Int32> countPtr = calloc<Int32>();
    Pointer<bindings.CactusLoraAdapterInfoC> cAdapters = nullptr;
    List<LoraAdapterInfo> resultAdapters = [];

    try {
      cAdapters = bindings.getLoadedLoraAdapters(_handle, countPtr);
      final count = countPtr.value;

      if (cAdapters != nullptr && count > 0) {
        for (int i = 0; i < count; i++) {
          resultAdapters.add(LoraAdapterInfo(
            path: cAdapters[i].path.toDartString(),
            scale: cAdapters[i].scale,
          ));
        }
      }
      return resultAdapters;
    } finally {
      if (cAdapters != nullptr) {
        bindings.freeLoraAdaptersArray(cAdapters, countPtr.value); // Assumes paths are freed by C
      }
      calloc.free(countPtr);
    }
  }
  // --- End LoRA Adapter Management ---

  // +++ Benchmarking +++
  /// Runs a benchmark on the loaded model.
  /// Throws [CactusOperationException] on failure.
  BenchResult bench({int pp = 512, int tg = 128, int pl = 1, int nr = 1}) {
    Pointer<Utf8> resultJsonC = nullptr;
    try {
      resultJsonC = bindings.bench(_handle, pp, tg, pl, nr);
      if (resultJsonC == nullptr || resultJsonC.toDartString().isEmpty || resultJsonC.toDartString() == "[]") {
        throw CactusOperationException("Benchmarking", "Native benchmarking returned null or empty result.");
      }
      final resultString = resultJsonC.toDartString();
      final List<dynamic> decodedJson = jsonDecode(resultString);
      return BenchResult.fromJson(decodedJson);
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusOperationException("Benchmarking", "Error during benchmarking: ${e.toString()}", e);
    } finally {
      if (resultJsonC != nullptr) bindings.freeString(resultJsonC);
    }
  }
  // --- End Benchmarking ---

  // +++ Advanced Chat Formatting +++
  /// Formats a list of chat messages using Jinja templating, potentially with tools and schema.
  /// Throws [CactusOperationException] on failure.
  Future<FormattedChatResult> getFormattedChatJinja({
    required List<ChatMessage> messages,
    String? chatTemplateOverride,
    String? jsonSchema,
    String? toolsJson,
    bool parallelToolCalls = false,
    String? toolChoice,
  }) async {
    Pointer<Utf8> messagesJsonC = nullptr;
    Pointer<Utf8> templateOverrideC = nullptr;
    Pointer<Utf8> schemaC = nullptr;
    Pointer<Utf8> toolsC = nullptr;
    Pointer<Utf8> choiceC = nullptr;
    Pointer<bindings.CactusFormattedChatResultC> resultC = calloc<bindings.CactusFormattedChatResultC>();

    try {
      final messagesJsonString = jsonEncode(messages.map((m) => m.toJson()).toList());
      messagesJsonC = messagesJsonString.toNativeUtf8(allocator: calloc);
      if (chatTemplateOverride != null) templateOverrideC = chatTemplateOverride.toNativeUtf8(allocator: calloc);
      if (jsonSchema != null) schemaC = jsonSchema.toNativeUtf8(allocator: calloc);
      if (toolsJson != null) toolsC = toolsJson.toNativeUtf8(allocator: calloc);
      if (toolChoice != null) choiceC = toolChoice.toNativeUtf8(allocator: calloc);

      final status = bindings.getFormattedChatJinja(
        _handle,
        messagesJsonC,
        templateOverrideC ?? nullptr,
        schemaC ?? nullptr,
        toolsC ?? nullptr,
        parallelToolCalls,
        choiceC ?? nullptr,
        resultC,
      );

      if (status != 0) {
        throw CactusOperationException("GetFormattedChatJinja", "Native call failed with status $status");
      }

      final prompt = resultC.ref.prompt.toDartString();
      final grammar = resultC.ref.grammar == nullptr ? null : resultC.ref.grammar.toDartString();
      
      return FormattedChatResult(prompt: prompt, grammar: grammar);
    } catch (e) {
      if (e is CactusException) rethrow;
      throw CactusOperationException("GetFormattedChatJinja", "Error: ${e.toString()}", e);
    } finally {
      if (messagesJsonC != nullptr) calloc.free(messagesJsonC);
      if (templateOverrideC != nullptr) calloc.free(templateOverrideC);
      if (schemaC != nullptr) calloc.free(schemaC);
      if (toolsC != nullptr) calloc.free(toolsC);
      if (choiceC != nullptr) calloc.free(choiceC);
      bindings.freeFormattedChatResultMembers(resultC); // Frees internal strings
      calloc.free(resultC); // Frees the struct itself
    }
  }
  // --- End Advanced Chat Formatting ---

  // +++ Model Information and Validation +++
  /// Validates if a chat template for the loaded model is valid.
  bool validateModelChatTemplate({bool useJinja = false, String? name}) {
    Pointer<Utf8> nameC = nullptr;
    try {
      if (name != null) nameC = name.toNativeUtf8(allocator: calloc);
      return bindings.validateModelChatTemplate(_handle, useJinja, nameC ?? nullptr);
    } finally {
      if (nameC != nullptr) calloc.free(nameC);
    }
  }

  /// Retrieves metadata about the loaded model.
  /// Throws [CactusOperationException] if model not loaded.
  ModelMeta getModelMeta() {
    Pointer<bindings.CactusModelMetaC> metaC = calloc<bindings.CactusModelMetaC>();
    try {
      bindings.getModelMeta(_handle, metaC);
      // Check if desc is null, which might happen if native side couldn't get it.
      // The native side should ideally return an error or set a flag if meta couldn't be obtained.
      // For now, we rely on native side populating correctly or desc being nullable.
      final description = metaC.ref.desc == nullptr ? "" : metaC.ref.desc.toDartString();
      
      final meta = ModelMeta(
        description: description,
        size: metaC.ref.size,
        nParams: metaC.ref.n_params,
      );
      return meta;
    } finally {
      bindings.freeModelMetaMembers(metaC); // Frees internal strings like desc
      calloc.free(metaC);
    }
  }
  // --- End Model Information and Validation ---

  // +++ User-Specific Adaptation (Placeholder) +++
  /// Sets user-specific vocabulary to potentially bias STT.
  ///
  /// NOTE: This feature is currently a placeholder and does not affect STT results
  /// as the core C++ implementation for vocabulary adaptation is not yet available.
  ///
  /// [vocabulary] A list of words or phrases to bias towards.
  Future<void> setUserVocabulary(List<String> vocabulary) async {
    // TODO: Implement once core C++ functionality is available.
    // This would involve:
    // 1. Converting the List<String> to a JSON string.
    // 2. Calling an FFI function like `cactus_stt_set_vocabulary_ffi(_handle, vocabularyJsonC)`.
    // For now, just log that it's a placeholder.
    print('[CactusContext] setUserVocabulary called with ${vocabulary.length} items. This feature is a placeholder and not yet implemented at the core C++ level.');
    // Example of how it might look:
    // final String vocabularyJson = jsonEncode(vocabulary);
    // Pointer<Utf8> vocabularyJsonC = nullptr;
    // try {
    //   vocabularyJsonC = vocabularyJson.toNativeUtf8(allocator: calloc);
    //   // Assuming an FFI function like:
    //   // bindings.sttSetUserVocabulary(_handle, vocabularyJsonC);
    //   // Check status if the FFI function returns one.
    // } finally {
    //   if (vocabularyJsonC != nullptr) calloc.free(vocabularyJsonC);
    // }
    return Future.value(); // Immediately return as it's a placeholder
  }
  // --- End User-Specific Adaptation ---
} 