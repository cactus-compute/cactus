import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'dart:io' show Platform; // For platform-specific library names

// Assuming the generated bindings will be in this path, based on ffigen.yaml
import '../cactus_bindings_generated.dart';

/// Manages Speech-to-Text (STT) operations using the Cactus native library.
class CactusSTTService {
  /// Holds the native bindings, loaded from the dynamic library.
  late final CactusBindings _bindings;

  /// Pointer to the native STT context. Null if not initialized.
  Pointer<cactus_stt_context_t> _sttContext = nullptr;

  /// Flag to indicate if the service is initialized and context is valid.
  bool get isInitialized => _sttContext != nullptr;

  /// Loads the native library and initializes the bindings.
  ///
  /// Throws an [Exception] if the library cannot be loaded.
  CactusSTTService() {
    _bindings = CactusBindings(_loadLibrary());
  }

  DynamicLibrary _loadLibrary() {
    if (Platform.isMacOS || Platform.isIOS) {
      // On iOS/macOS, use DynamicLibrary.process() to find symbols in the main executable,
      // assuming the static library is linked. For a dynamic framework, use DynamicLibrary.open().
      // For simplicity with Flutter, where native code is often bundled, process() is a common start.
      // If it's a separate .dylib or .framework, 'path/to/libcactus_core.dylib' would be used.
      return DynamicLibrary.process();
    } else if (Platform.isAndroid) {
      return DynamicLibrary.open('libcactus_core.so');
    } else if (Platform.isLinux) {
      return DynamicLibrary.open('libcactus_core.so');
    } else if (Platform.isWindows) {
      return DynamicLibrary.open('cactus_core.dll');
    }
    throw Exception('Unsupported platform for loading native library.');
  }

  /// Initializes the STT engine with the specified model.
  ///
  /// [modelPath]: Path to the ggml Whisper model file.
  /// [language]: Language code (e.g., "en").
  /// Returns `true` if initialization is successful, `false` otherwise.
  Future<bool> initialize(String modelPath, String language) async {
    if (isInitialized) {
      print('STT Service already initialized. Please free the existing instance first.');
      return true; // Or false, depending on desired behavior for re-initialization
    }

    Pointer<Utf8> modelPathPtr = modelPath.toNativeUtf8();
    Pointer<Utf8> languagePtr = language.toNativeUtf8();

    try {
      _sttContext = _bindings.cactus_stt_init(modelPathPtr, languagePtr);
      if (_sttContext == nullptr) {
        print('Failed to initialize STT context: cactus_stt_init returned nullptr.');
        return false;
      }
      return true;
    } catch (e) {
      print('Exception during STT initialization: $e');
      _sttContext = nullptr; // Ensure context is null on error
      return false;
    } finally {
      malloc.free(modelPathPtr);
      malloc.free(languagePtr);
    }
  }

  /// Processes a chunk of audio data for transcription.
  ///
  /// [audioSamples]: A list of float audio samples (PCM 32-bit, 16kHz, mono).
  /// Returns `true` if processing is successful, `false` otherwise.
  Future<bool> processAudioChunk(List<double> audioSamples) async {
    if (!isInitialized) {
      print('STT Service not initialized.');
      return false;
    }
    if (audioSamples.isEmpty) {
      print('Audio samples list is empty.');
      return false; // Or true, if empty list is not an error
    }

    // Allocate memory for the audio samples and copy the data.
    // Note: `double` in Dart is typically 64-bit, STT expects float (32-bit).
    // The FFI layer for processAudio takes `const float*`, so we need Pointer<Float>.
    final Pointer<Float> samplesPtr = malloc.allocate<Float>(audioSamples.length);
    for (int i = 0; i < audioSamples.length; i++) {
      samplesPtr[i] = audioSamples[i]; // Dart double is implicitly converted to float here if needed by store operation.
    }

    try {
      final success = _bindings.cactus_stt_process_audio(
        _sttContext,
        samplesPtr,
        audioSamples.length,
      );
      return success;
    } catch (e) {
      print('Exception during audio processing: $e');
      return false;
    } finally {
      malloc.free(samplesPtr);
    }
  }

  /// Retrieves the full transcription result from the processed audio.
  ///
  /// Returns the transcribed text as a [String], or `null` if no transcription
  /// is available or an error occurs.
  Future<String?> getTranscription() async {
    if (!isInitialized) {
      print('STT Service not initialized.');
      return null;
    }

    Pointer<Utf8> transcriptionPtr = nullptr;
    try {
      transcriptionPtr = _bindings.cactus_stt_get_transcription(_sttContext);

      if (transcriptionPtr == nullptr) {
        // This can mean no transcription is ready yet, or an error occurred.
        // The C++ layer might log more details if it's an error.
        print('Failed to get transcription: cactus_stt_get_transcription returned nullptr.');
        return null;
      }

      final String transcription = transcriptionPtr.toDartString();
      return transcription;
    } catch (e) {
      print('Exception during transcription retrieval: $e');
      return null;
    } finally {
      // Free the string allocated by C using the provided C FFI function.
      if (transcriptionPtr != nullptr) {
        _bindings.cactus_free_string_c(transcriptionPtr.cast<Char>()); // Cast Pointer<Utf8> to Pointer<Char>
      }
    }
  }

  /// Frees the STT context and associated native resources.
  ///
  /// It's important to call this when the STT service is no longer needed
  /// to prevent memory leaks.
  Future<void> free() async {
    if (!isInitialized) {
      print('STT Service not initialized or already freed.');
      return;
    }

    try {
      _bindings.cactus_stt_free(_sttContext);
      _sttContext = nullptr; // Mark as freed
    } catch (e) {
      print('Exception during STT free: $e');
      // Even on exception, mark as freed to prevent reuse of potentially invalid context.
      _sttContext = nullptr;
    }
  }
}

// Example of how ffigen might generate the Opaque type if not explicitly defined.
// This is just for illustrative purposes; the actual type comes from the generated bindings.
// class cactus_stt_context_t extends Opaque {}
// If cactus_stt_context_t is defined as `typedef struct cactus_stt_context cactus_stt_context_t;`
// and `struct cactus_stt_context` is not exposed, ffigen treats it as opaque.
// If `cactus_ffi.h` had `typedef void* cactus_stt_context_t;`, it would also be `Pointer<Void>` then `Pointer<Opaque>`.
// The key is that `cactus_stt_context_t` in Dart will be a `Pointer<cactus_stt_context_t>` where
// `cactus_stt_context_t` itself is a Dart type representing the native struct (often Opaque).
