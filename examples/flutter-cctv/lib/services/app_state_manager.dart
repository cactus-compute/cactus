import 'dart:io';

/// Manager for application state
class AppStateManager {
  // Status and loading state
  String _status = 'Starting...';
  bool _isLoading = true;
  double? _progress;
  String? _error;

  // Camera state
  bool _isCameraInitialized = false;
  File? _lastCapturedImage;
  bool _isCapturing = false;
  bool _isInferenceRunning = false;

  // Getters
  String get status => _status;
  bool get isLoading => _isLoading;
  double? get progress => _progress;
  String? get error => _error;
  bool get isCameraInitialized => _isCameraInitialized;
  File? get lastCapturedImage => _lastCapturedImage;
  bool get isCapturing => _isCapturing;
  bool get isInferenceRunning => _isInferenceRunning;

  // Status update methods
  void setLoading(bool loading, {String? message}) {
    _isLoading = loading;
    if (message != null) {
      _status = message;
    }
  }

  void updateStatus(String message) {
    _status = message;
  }

  void updateProgress(double? value) {
    _progress = value;
  }

  void setError(String? errorMessage) {
    _error = errorMessage;
  }

  // Camera state update methods
  void setCameraInitialized(bool initialized) {
    _isCameraInitialized = initialized;
  }

  void setCapturing(bool capturing) {
    _isCapturing = capturing;
  }

  void setInferenceRunning(bool running) {
    _isInferenceRunning = running;
  }

  void setLastCapturedImage(File? image) {
    _lastCapturedImage = image;
  }

  /// Reset application state
  void reset() {
    _status = 'Starting...';
    _isLoading = true;
    _progress = null;
    _error = null;
    _isCameraInitialized = false;
    _lastCapturedImage = null;
    _isCapturing = false;
    _isInferenceRunning = false;
  }
}
