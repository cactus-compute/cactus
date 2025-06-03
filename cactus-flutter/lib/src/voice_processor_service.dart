import 'dart:async';

import 'stt_service.dart';
import 'audio_input_service.dart';

/// Represents the state of the VoiceProcessorService.
enum VoiceProcessorState {
  idle,
  initializingSTT,
  sttInitialized, // STT is ready, but not actively listening
  listening, // Actively capturing and processing audio
  processing, // Audio capture stopped, but STT might be finalizing
  error,
}

/// Represents an error originating from the voice processing workflow.
class VoiceProcessorError {
  final String message;
  final Object? underlyingException; // Optional: store the original exception

  VoiceProcessorError(this.message, {this.underlyingException});

  @override
  String toString() {
    return "VoiceProcessorError: $message" +
        (underlyingException != null ? " (Cause: $underlyingException)" : "");
  }
}

/// Orchestrates audio input and speech-to-text (STT) services
/// to provide a complete voice processing workflow.
class VoiceProcessorService {
  final CactusSTTService _sttService;
  final AudioInputService _audioInputService;

  bool _isSTTInitialized = false;
  bool _isListening = false;
  Timer? _transcriptionTimer;
  StreamSubscription<List<double>>? _audioDataSubscription;

  // Stream controllers for exposing state, transcriptions, and errors.
  // Using broadcast controllers to allow multiple listeners if needed.
  final _stateController =
      StreamController<VoiceProcessorState>.broadcast();
  final _transcriptionController = StreamController<String>.broadcast();
  final _errorController = StreamController<VoiceProcessorError>.broadcast();

  /// Current state of the voice processor.
  Stream<VoiceProcessorState> get stateStream => _stateController.stream;

  /// Stream of transcription results (can be partial or final).
  Stream<String> get transcriptionStream => _transcriptionController.stream;

  /// Stream of errors encountered during voice processing.
  Stream<VoiceProcessorError> get errorStream => _errorController.stream;

  /// Public getter for STT initialization status.
  bool get isSTTInitialized => _isSTTInitialized;
  /// Public getter for listening status.
  bool get isListening => _isListening;


  VoiceProcessorService({
    required CactusSTTService sttService,
    required AudioInputService audioInputService,
  })  : _sttService = sttService,
        _audioInputService = audioInputService {
    _stateController.add(VoiceProcessorState.idle);
  }

  /// Initializes the underlying STT engine.
  ///
  /// [modelPath]: Path to the STT model file.
  /// [language]: Language code for STT (e.g., "en").
  Future<void> initialize(
      {required String modelPath, required String language}) async {
    if (_isSTTInitialized) {
      print("VoiceProcessorService: STT already initialized.");
      _stateController.add(VoiceProcessorState.sttInitialized);
      return;
    }
    _stateController.add(VoiceProcessorState.initializingSTT);
    try {
      final success =
          await _sttService.initialize(modelPath, language);
      if (success) {
        _isSTTInitialized = true;
        _stateController.add(VoiceProcessorState.sttInitialized);
        print("VoiceProcessorService: STT initialized successfully.");
      } else {
        _isSTTInitialized = false;
        _stateController.add(VoiceProcessorState.error);
        _errorController.add(VoiceProcessorError(
            "STT initialization failed: initialize returned false."));
      }
    } catch (e, s) {
      _isSTTInitialized = false;
      _stateController.add(VoiceProcessorState.error);
      _errorController.add(VoiceProcessorError(
          "Exception during STT initialization.", underlyingException: e));
      print("VoiceProcessorService: STT Initialization Exception: $e\n$s");
    }
  }

  /// Starts capturing audio and processing it for speech-to-text.
  ///
  /// This involves:
  /// 1. Checking STT initialization and microphone permissions.
  /// 2. Subscribing to the audio data stream from [AudioInputService].
  /// 3. Starting native audio recording.
  /// 4. Periodically fetching transcriptions from [CactusSTTService].
  Future<void> startProcessing() async {
    if (!_isSTTInitialized) {
      _errorController.add(VoiceProcessorError(
          "Cannot start processing: STT not initialized."));
      _stateController.add(VoiceProcessorState.error);
      return;
    }
    if (_isListening) {
      print("VoiceProcessorService: Already listening.");
      return;
    }

    // Check and request permission
    bool hasPerm = await _audioInputService.hasPermission();
    if (!hasPerm) {
      print("VoiceProcessorService: Requesting microphone permission...");
      hasPerm = await _audioInputService.requestPermission();
    }

    if (!hasPerm) {
      _errorController.add(
          VoiceProcessorError("Microphone permission not granted."));
      _stateController.add(VoiceProcessorState.error);
      return;
    }

    _stateController.add(VoiceProcessorState.listening);
    _isListening = true;

    // Subscribe to audio data stream
    _audioDataSubscription?.cancel(); // Cancel previous one if any
    _audioDataSubscription =
        _audioInputService.audioDataStream.listen((audioChunk) async {
      if (!_isListening || !_sttService.isInitialized) return;
      try {
        // Asynchronously process the audio chunk without awaiting,
        // to prevent blocking the audio stream.
        _sttService.processAudioChunk(audioChunk).then((success) {
          if (!success) {
            print("VoiceProcessorService: Failed to process audio chunk.");
            // Optionally emit a non-fatal error
          }
        }).catchError((e,s){
            print("VoiceProcessorService: Error processing audio chunk: $e\n$s");
            _errorController.add(VoiceProcessorError("Error during STT processAudioChunk", underlyingException: e));
        });
      } catch (e,s) {
        print("VoiceProcessorService: Synchronous error in audio data listener: $e\n$s");
         _errorController.add(VoiceProcessorError("Error in audio data listener", underlyingException: e));
      }
    }, onError: (error) {
      print("VoiceProcessorService: Error from AudioInputService stream: $error");
      _errorController.add(VoiceProcessorError(
          "Audio input stream error.", underlyingException: error));
      // Consider stopping processing or attempting to recover.
      // For now, just emit error and update state.
      _stateController.add(VoiceProcessorState.error);
      _isListening = false; // Assume listening stopped due to audio error
      _transcriptionTimer?.cancel();
    }, onDone: () {
        print("VoiceProcessorService: AudioInputService stream closed.");
        if(_isListening) { // If we were listening and stream closed unexpectedly
            _isListening = false;
            _transcriptionTimer?.cancel();
            _stateController.add(VoiceProcessorState.idle); // Or an error state
        }
    });

    // Start native recording
    final bool recordingStarted = await _audioInputService.startRecording();
    if (!recordingStarted) {
      _errorController.add(VoiceProcessorError(
          "Failed to start audio recording via AudioInputService."));
      _stateController.add(VoiceProcessorState.error);
      _isListening = false;
      _audioDataSubscription?.cancel();
      return;
    }

    // Periodically fetch transcription
    _transcriptionTimer?.cancel();
    _transcriptionTimer = Timer.periodic(const Duration(milliseconds: 750), // Adjust interval as needed
        (_) async {
      if (!_isListening || !_sttService.isInitialized) return;
      try {
        final String? transcription = await _sttService.getTranscription();
        if (transcription != null && transcription.isNotEmpty) {
          _transcriptionController.add(transcription);
        }
      } catch (e,s) {
        print("VoiceProcessorService: Error fetching transcription: $e\n$s");
        _errorController.add(VoiceProcessorError(
            "Error fetching transcription.", underlyingException: e));
      }
    });
    print("VoiceProcessorService: Started processing.");
  }

  /// Stops capturing audio and finalizes any ongoing STT processing.
  Future<void> stopProcessing() async {
    if (!_isListening) {
      // print("VoiceProcessorService: Not currently listening or already stopped.");
      // Ensure state is correct even if called multiple times or when not listening.
      if (_isListening) _isListening = false; // Should not happen if check above is true
      _transcriptionTimer?.cancel();
      _transcriptionTimer = null;
      _audioDataSubscription?.cancel();
      _audioDataSubscription = null;
      // If STT is initialized, it's good to go to sttInitialized state.
      // Otherwise, idle is more appropriate.
      _stateController.add(_isSTTInitialized ? VoiceProcessorState.sttInitialized : VoiceProcessorState.idle);
      return;
    }

    print("VoiceProcessorService: Stopping processing...");
    _isListening = false;
    _stateController.add(VoiceProcessorState.processing); // Indicate processing final transcription

    _transcriptionTimer?.cancel();
    _transcriptionTimer = null;

    await _audioInputService.stopRecording(); // Stop native audio capture

    // It's important that the subscription is cancelled after audioInputService.stopRecording(),
    // because stopRecording might close the stream which would trigger onDone.
    // If we cancel before, we might miss some final chunks if stopRecording is not immediate.
    // However, audioInputService.stopRecording() should ideally ensure its stream is flushed/closed
    // before its future completes. For robustness, cancel here.
    await _audioDataSubscription?.cancel();
    _audioDataSubscription = null;


    // Perform a final call to get transcription
    if (_isSTTInitialized) {
      try {
        final String? transcription = await _sttService.getTranscription();
        if (transcription != null && transcription.isNotEmpty) {
          _transcriptionController.add(transcription);
        }
      } catch (e,s) {
        print("VoiceProcessorService: Error fetching final transcription: $e\n$s");
        _errorController.add(VoiceProcessorError(
            "Error fetching final transcription.", underlyingException: e));
      }
    }

    _stateController.add(_isSTTInitialized ? VoiceProcessorState.sttInitialized : VoiceProcessorState.idle);
    print("VoiceProcessorService: Stopped processing.");
  }

  /// Disposes of all resources used by the service.
  ///
  /// This includes stopping any active processing, freeing the STT engine,
  /// and closing all stream controllers.
  Future<void> dispose() async {
    print("VoiceProcessorService: Disposing...");
    await stopProcessing(); // Ensure everything is stopped

    if (_isSTTInitialized) {
      await _sttService.free();
      _isSTTInitialized = false;
    }

    await _audioInputService.dispose();

    await _stateController.close();
    await _transcriptionController.close();
    await _errorController.close();
    print("VoiceProcessorService: Disposed.");
  }
}
