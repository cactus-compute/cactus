import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'dart:async';

/// Service to manage camera operations and state
class CameraService {
  // Camera state
  CameraController? _controller;
  List<CameraDescription> _cameras = [];
  bool _isInitialized = false;
  bool _isStreaming = false;
  int _currentCameraIndex = 0;

  // Getters
  CameraController? get controller => _controller;
  bool get isInitialized => _isInitialized;
  bool get isStreaming => _isStreaming;

  /// Initialize the camera service
  Future<void> initialize() async {
    try {
      _cameras = await availableCameras();

      if (_cameras.isEmpty) {
        print('No cameras available');
        return;
      }

      print('Available cameras: ${_cameras.length}');
      for (var i = 0; i < _cameras.length; i++) {
        print('Camera $i: ${_cameras[i].name} (${_cameras[i].lensDirection})');
      }

      await _initCameraController();
    } catch (e) {
      print('Error initializing camera service: $e');
      _isInitialized = false;
    }
  }

  /// Initialize the camera controller
  Future<void> _initCameraController() async {
    if (_controller != null) {
      await _controller!.dispose();
    }

    try {
      final cameraController = CameraController(
        _cameras[_currentCameraIndex],
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420
      );

      await cameraController.initialize();

      _controller = cameraController;
      _isInitialized = true;
      print(
        'Camera initialized successfully: ${_cameras[_currentCameraIndex].name}',
      );
    } catch (e) {
      print('Error initializing camera controller: $e');
      _isInitialized = false;
    }
  }

  /// Start streaming camera frames
  /// Returns true if streaming started successfully
  Future<bool> startStreaming(
    void Function(CameraImage) onFrame, {
    int frameSkipCount = 30,
  }) async {
    if (!_isInitialized || _controller == null) {
      print('Cannot start streaming: Camera not initialized');
      return false;
    }

    if (_isStreaming) {
      print('Streaming already active');
      return true;
    }

    try {
      int frameCounter = 0;

      await _controller!.startImageStream((image) {
        // Skip frames to reduce processing load
        frameCounter++;
        if (frameCounter < frameSkipCount) return;
        frameCounter = 0;

        // Forward the frame to the callback
        onFrame(image);
      });

      _isStreaming = true;
      print('Camera streaming started');
      return true;
    } catch (e) {
      print('Error starting image stream: $e');
      return false;
    }
  }

  /// Stop streaming camera frames
  Future<void> stopStreaming() async {
    if (!_isStreaming || _controller == null) return;

    try {
      await _controller!.stopImageStream();
      _isStreaming = false;
      print('Camera streaming stopped');
    } catch (e) {
      print('Error stopping image stream: $e');
    }
  }

  /// Switch to the next available camera
  Future<bool> flipCamera() async {
    if (_cameras.length <= 1) {
      print('Only one camera available, cannot flip');
      return false;
    }

    final wasStreaming = _isStreaming;

    // Stop streaming if active
    if (wasStreaming) {
      await stopStreaming();
    }

    // Calculate next camera index
    _currentCameraIndex = (_currentCameraIndex + 1) % _cameras.length;

    // Initialize the new camera
    await _initCameraController();

    // Resume streaming if it was active
    if (wasStreaming && _isInitialized) {
      await startStreaming((image) {});
    }

    return _isInitialized;
  }

  /// Clean up resources
  Future<void> dispose() async {
    await stopStreaming();
    await _controller?.dispose();
    _controller = null;
    _isInitialized = false;
    print('Camera service disposed');
  }
}
