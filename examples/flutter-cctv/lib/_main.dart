import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import 'package:camera/camera.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'dart:async';
import 'dart:math' show min;

// Import services
import 'services/camera_service.dart';
import 'services/inference_service.dart';
import 'services/app_state_manager.dart';
import 'utils/image_converter.dart';
import 'utils/file_manager.dart';

// Import widgets
import 'widgets/loading_screen.dart';
import 'widgets/error_screen.dart';
import 'widgets/camera_screen.dart';
import 'widgets/camera_initializing.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> with WidgetsBindingObserver {
  // Services
  final CameraService _cameraService = CameraService();
  final InferenceService _inferenceService = InferenceService();
  final AppStateManager _stateManager = AppStateManager();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeApp();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopCapturing();
    _cameraService.dispose();
    _inferenceService.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive) {
      _stopCapturing();
      _cameraService.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  Future<void> _initializeApp() async {
    _updateState(() {
      _stateManager.setLoading(true, message: 'Initializing Cactus...');
    });

    // Initialize Cactus
    final success = await _inferenceService.initialize(
      onStatus: (message, {progress}) {
        print('Cactus status: $message, progress: $progress');
        _updateState(() {
          _stateManager.updateStatus(message);
          _stateManager.updateProgress(progress);
          _stateManager.setLoading(
            progress != null &&
                progress < 1.0 &&
                !message.contains('success') &&
                !message.contains('Error'),
          );
        });
      },
    );

    if (!success) {
      _updateState(() {
        _stateManager.setLoading(false);
        _stateManager.setError('Failed to initialize Cactus');
      });
      return;
    }

    // Initialize camera
    await _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    _updateState(() {
      _stateManager.updateStatus('Initializing camera...');
    });

    await _cameraService.initialize();

    _updateState(() {
      _stateManager.setCameraInitialized(_cameraService.isInitialized);
      _stateManager.setLoading(false);
      _stateManager.updateStatus(
        _cameraService.isInitialized
            ? 'Camera ready'
            : 'Failed to initialize camera',
      );
    });
  }

  void _toggleCameraCapture() {
    if (_stateManager.isCapturing) {
      _stopCapturing();
    } else {
      _startCapturing();
    }
  }

  void _startCapturing() {
    if (!_cameraService.isInitialized) {
      _updateState(() {
        _stateManager.updateStatus('Camera not ready');
      });
      return;
    }

    _updateState(() {
      _stateManager.setCapturing(true);
      _stateManager.updateStatus('Started streaming - waiting for frames');
    });

    // Start streaming frames
    _cameraService.startStreaming(_processFrame);
  }

  void _stopCapturing() {
    _cameraService.stopStreaming();

    _updateState(() {
      _stateManager.setCapturing(false);
      _stateManager.updateStatus('Stopped capturing');
    });
  }

  Future<void> _processFrame(CameraImage image) async {
    if (_stateManager.isInferenceRunning || !_inferenceService.isInitialized) {
      return;
    }
    
    _updateState(() {
      _stateManager.setInferenceRunning(true);
    });
    
    try {
      // Create temporary file path
      final tempDir = await getTemporaryDirectory();
      final tempFilePath = ImageConverter.getUniqueImagePath(tempDir.path);
      
      // Save frame to file for display
      final imageFile = await ImageConverter.convertImageToFile(
        image,
        tempFilePath,
      );
      
      _updateState(() {
        _stateManager.setLastCapturedImage(imageFile);
        if (!_stateManager.status.endsWith('...')) {
          _stateManager.updateStatus('${_stateManager.status} (analyzing...)');
        }
      });
      
      // Process with Cactus
      final result = await _inferenceService.processFrame(
        image, 
        tempFilePath,
        maxTokens: 20,
      );
      
      // Format and clean up the inference result
      String displayResult = result;
      if (displayResult.contains('<end_of_utterance>')) {
        displayResult = displayResult.replaceAll('<end_of_utterance>', '');
      }
      if (displayResult.contains('<|im_end|>')) {
        displayResult = displayResult.replaceAll('<|im_end|>', '');
      }
      displayResult = displayResult.trim();
      
      _updateState(() {
        _stateManager.updateStatus(
          displayResult.substring(0, min(100, displayResult.length)),
        );
      });
    } catch (e) {
      print('Error processing frame: $e');
      _updateState(() {
        _stateManager.updateStatus('Error: $e');
      });
    } finally {
      _updateState(() {
        _stateManager.setInferenceRunning(false);
      });
    }
  }

  void _flipCamera() async {
    final success = await _cameraService.flipCamera();

    if (success) {
      _updateState(() {
        _stateManager.updateStatus('Camera switched');
      });

      // Restart capturing if it was active
      if (_stateManager.isCapturing) {
        _cameraService.startStreaming(_processFrame);
      }
    }
  }

  void _updateState(VoidCallback fn) {
    if (mounted) {
      setState(fn);
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cactus CCTV',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: Scaffold(
        appBar: AppBar(
          backgroundColor: Theme.of(context).colorScheme.inversePrimary,
          title: const Text('Cactus CCTV'),
          actions: [
            IconButton(
              icon: const Icon(Icons.info_outline),
              onPressed: () {
                _updateState(() {
                  _stateManager.updateStatus(
                    'Debug Info:\nCactus: ${_inferenceService.isInitialized}\n'
                    'Camera: ${_stateManager.isCameraInitialized}\n'
                    'Processing: ${_stateManager.isInferenceRunning}',
                  );
                });
              },
            ),
          ],
        ),
        body: Builder(
          builder: (context) {
            if (_stateManager.isLoading) {
              return LoadingScreen(
                status: _stateManager.status,
                progress: _stateManager.progress,
              );
            }

            if (_stateManager.isCameraInitialized &&
                _cameraService.controller != null &&
                _cameraService.controller!.value.isInitialized) {
              return CameraScreen(
                cameraController: _cameraService.controller!,
                status: _stateManager.status,
                lastCapturedImage: _stateManager.lastCapturedImage,
                isCapturing: _stateManager.isCapturing,
                isInferenceRunning: _stateManager.isInferenceRunning,
                onToggleCapture: _toggleCameraCapture,
                onFlipCamera: _flipCamera,
              );
            }

            if (!_inferenceService.isInitialized) {
              return ErrorScreen(
                errorMessage: _stateManager.status,
                onRetry: _initializeApp,
              );
            }

            return CameraInitializing(
              status: _stateManager.status,
              onInitializeCamera: _initializeCamera,
            );
          },
        ),
      ),
    );
  }
}
