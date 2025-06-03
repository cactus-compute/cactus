import 'dart:async';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart' show visibleForTesting; // For @visibleForTesting

/// Service to manage native audio input capabilities.
///
/// It uses a [MethodChannel] to invoke native methods for controlling audio recording
/// (permissions, start, stop) and an [EventChannel] to stream audio data from native to Dart.
class AudioInputService {
  // Define channel names consistently with the native side implementations.
  @visibleForTesting
  static const MethodChannel methodChannel =
      MethodChannel('com.cactus.sdk/audio_input_methods');
  @visibleForTesting
  static const EventChannel eventChannel =
      EventChannel('com.cactus.sdk/audio_input_events');

  // Stream controller to broadcast received audio data.
  StreamController<List<double>>? _audioDataController;

  /// Public stream of audio data.
  ///
  /// Consumers can listen to this stream to receive chunks of audio samples (List<double>)
  /// as they are captured by the native audio input module.
  Stream<List<double>> get audioDataStream {
    _audioDataController ??= StreamController<List<double>>.broadcast();
    return _audioDataController!.stream;
  }

  StreamSubscription? _eventChannelSubscription;
  bool _isRecordingDart = false;

  /// Checks if microphone permission has been granted.
  ///
  /// Returns `true` if permission is granted, `false` otherwise.
  Future<bool> hasPermission() async {
    try {
      final bool? granted = await methodChannel.invokeMethod<bool>('hasPermission');
      return granted ?? false;
    } on PlatformException catch (e) {
      print("Error checking microphone permission: ${e.message}");
      return false;
    }
  }

  /// Requests microphone permission from the user.
  ///
  /// Returns `true` if permission is granted, `false` otherwise.
  /// This might show a system dialog to the user.
  Future<bool> requestPermission() async {
    try {
      final bool? granted = await methodChannel.invokeMethod<bool>('requestPermission');
      return granted ?? false;
    } on PlatformException catch (e) {
      print("Error requesting microphone permission: ${e.message}");
      return false;
    }
  }

  /// Starts the audio recording process on the native side.
  ///
  /// Sets up a listener on the [EventChannel] to receive audio data.
  /// Returns `true` if the recording successfully starts (native method call succeeds),
  /// `false` if an error occurs (e.g., permission denied, native error).
  Future<bool> startRecording() async {
    if (_isRecordingDart) {
      print("AudioInputService: Already recording.");
      return true; // Or false if this should be an error condition
    }

    // Ensure any previous stream controller is closed and a new one is ready.
    await _audioDataController?.close();
    _audioDataController = StreamController<List<double>>.broadcast();

    _eventChannelSubscription?.cancel(); // Cancel any existing subscription
    _eventChannelSubscription = eventChannel
        .receiveBroadcastStream()
        .listen((dynamic event) {
      if (event is List) {
        // Assuming the list from native is List<double> or can be cast.
        // If native sends Float32List, it needs conversion: event.buffer.asFloat32List().toList();
        try {
          final List<double> audioChunk = event.cast<double>();
          _audioDataController?.sink.add(audioChunk);
        } catch (e) {
           print("Error processing audio event from native: $e. Event was: $event");
           _audioDataController?.sink.addError(FormatException("Invalid audio data format: $e"));
        }
      } else if (event is Map) {
        if (event.containsKey('error')) {
          final String errorMessage = event['error'] as String? ?? "Unknown native error";
          print("Error event from native audio stream: $errorMessage");
          _audioDataController?.sink.addError(PlatformException(code: "NativeAudioError", message: errorMessage));
          // Optionally, stop recording on error or let the native side handle full stop.
          // For now, we just propagate the error and assume native side might stop.
          // Consider calling stopRecording() here if errors are fatal.
        } else {
          print("Unknown map event from native audio stream: $event");
        }
      } else if (event == null) {
        // Stream might send null on close, or it might be an actual error.
        print("Null event received from native audio stream. Assuming stream closed by native.");
        // If the native side closes the stream, we might want to reflect that recording has stopped.
        // However, explicit stopRecording call is preferred for cleanup.
        if (_isRecordingDart) { // Only close if we thought we were recording
            _handleStreamClosedByNative();
        }
      } else {
        print("Unknown event type from native audio stream: $event");
        _audioDataController?.sink.addError(FormatException("Unknown event type: ${event.runtimeType}"));
      }
    }, onError: (dynamic error) {
      print("Error on audio event channel: $error");
      _audioDataController?.sink.addError(error);
      _isRecordingDart = false; // Assume recording stopped on error
      // Consider further cleanup like calling stopRecording() or parts of it.
    }, onDone: () {
      print("Native audio event channel closed.");
      // This means the native side has closed the event channel.
      // We should update our recording state and clean up.
      _handleStreamClosedByNative();
    });

    try {
      await methodChannel.invokeMethod<void>('startRecording');
      _isRecordingDart = true;
      return true;
    } on PlatformException catch (e) {
      print("Failed to start recording: ${e.message}");
      await _eventChannelSubscription?.cancel();
      _eventChannelSubscription = null;
      await _audioDataController?.close();
      _audioDataController = null;
      _isRecordingDart = false;
      return false;
    }
  }

  /// Stops the audio recording process on the native side.
  ///
  /// Cancels the event channel subscription and closes the audio data stream.
  /// Returns `true` if stopping is successful (native method call succeeds),
  /// `false` otherwise.
  Future<bool> stopRecording() async {
    if (!_isRecordingDart && _eventChannelSubscription == null && (_audioDataController?.isClosed ?? true)) {
      print("AudioInputService: Not recording or already stopped/disposed.");
      // Ensure flag is false if somehow out of sync
      _isRecordingDart = false;
      return true;
    }

    _isRecordingDart = false; // Set flag immediately

    try {
      await methodChannel.invokeMethod<void>('stopRecording');
      // Native side should also close its event sink, which will trigger `onDone` for our subscription.
      // However, we also clean up proactively.
      await _eventChannelSubscription?.cancel();
      _eventChannelSubscription = null;
      if (!(_audioDataController?.isClosed ?? true)) {
        await _audioDataController?.close();
      }
      _audioDataController = null; // Allow it to be recreated on next start
      return true;
    } on PlatformException catch (e) {
      print("Failed to stop recording: ${e.message}");
      // Even if native call fails, ensure Dart side resources are cleaned up.
      await _eventChannelSubscription?.cancel();
      _eventChannelSubscription = null;
      if (!(_audioDataController?.isClosed ?? true)) {
         await _audioDataController?.close();
      }
      _audioDataController = null;
      return false;
    }
  }

  /// Returns the current recording state maintained on the Dart side.
  bool isRecording() {
    return _isRecordingDart;
  }

  /// Handles cleanup when the event stream is closed by the native side.
  void _handleStreamClosedByNative() {
      if (_isRecordingDart) { // If we thought we were recording
          print("Audio event stream closed by native, cleaning up Dart side.");
          _isRecordingDart = false;
          // Subscription is already implicitly cancelled by onDone.
          _eventChannelSubscription = null;
          if (!(_audioDataController?.isClosed ?? true)) {
              _audioDataController?.close();
          }
          _audioDataController = null;
      }
  }

  /// Disposes of the service, cleaning up resources.
  ///
  /// Calls [stopRecording] to ensure native resources are released and
  /// cancels any active stream subscriptions.
  Future<void> dispose() async {
    if (_isRecordingDart) {
      await stopRecording(); // This will also handle controller and subscription cleanup.
    } else {
      // Ensure cleanup even if not "recording" by Dart's flag, in case of partial setup.
      await _eventChannelSubscription?.cancel();
      _eventChannelSubscription = null;
      if (!(_audioDataController?.isClosed ?? true)) {
        await _audioDataController?.close();
      }
       _audioDataController = null;
    }
    print("AudioInputService disposed.");
  }
}
