import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/camera_service.dart';
import '../utils/timer.dart';
import '../services/inference.dart';
import '../widgets/debug_overlay.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _isCapturing = false;
  bool _showDebug = false;
  String _analysisResult = '';
  final CameraService _cameraService = CameraService();
  final InferenceService _inferenceService = InferenceService();

  @override
  void initState() {
    super.initState();
    timer.log('Starting app initialization');
    _initializeServices();
    timer.log('App initialization complete');
  }

  Future<void> _initializeServices() async {
    await _cameraService.initialize();
    timer.log('Camera initialized');
    await _inferenceService.initialize();
    timer.log('Inference initialized');
  }

  @override
  void dispose() {
    _cameraService.dispose();
    timer.log('Camera disposed');
    super.dispose();
  }

  Future<void> _onCaptureStartCallback(CameraImage image) async {
    final result = await _inferenceService.inferenceCallback(image);
    if (result != null) {
      timer.log('Inference result: $result (in _onCaptureStartCallback!)');
      setState(() => _analysisResult = result);
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
        body: Stack(
          children: [
            Column(
          children: [
            Expanded(
              flex: 75,
              child: _cameraService.controller?.value.isInitialized == true 
                ? FittedBox(
                    fit: BoxFit.cover,
                    child: SizedBox(
                      width: _cameraService.controller!.value.previewSize!.height,
                      height: _cameraService.controller!.value.previewSize!.width,
                      child: CameraPreview(_cameraService.controller!),
                    ),
                  )
                : const Center(child: Text('No camera')),
            ),
            
            Expanded(
              flex: 25, // 25% of available height
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                color: Colors.black87,
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            'Camera Analysis',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          SizedBox(height: 4),
                          Text(
                            _analysisResult,
                            style: TextStyle(color: Colors.white70),
                          ),
                        ],
                      ),
                    ),
                        Row(
                          children: [
                    _isCapturing
                    ? FloatingActionButton.small(
                      onPressed: () async {
                        await _cameraService.endCapture();
                        setState(() => _isCapturing = !_isCapturing);
                      },
                      backgroundColor: Colors.red,
                      child: Icon(Icons.stop),
                    )
                    : FloatingActionButton.small(
                      onPressed: () async {
                        await _cameraService.beginCapture(_onCaptureStartCallback);
                        setState(() => _isCapturing = !_isCapturing);
                      },
                      backgroundColor: Colors.green,
                      child: Icon(Icons.play_arrow),
                            ),
                            SizedBox(width: 8),
                            FloatingActionButton.small(
                              onPressed: () => setState(() => _showDebug = !_showDebug),
                              backgroundColor: _showDebug ? Colors.blue : Colors.grey,
                              child: Icon(Icons.bug_report),
                            ),
                          ],
                    ),
                  ],
                ),
                  ),
                ),
              ],
            ),
            if (_showDebug && _inferenceService.getLastProcessedImage() != null)
              Positioned(
                top: 20,
                right: 20,
                child: DebugOverlay(
                  imageFile: _inferenceService.getLastProcessedImage(),
                  stats: "Frame size: ${_inferenceService.getLastProcessedImage()!.lengthSync() ~/ 1024}KB",
              ),
            ),
          ],
        ),
      ),
    );
  }
} 