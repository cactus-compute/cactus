import 'package:camera/camera.dart';

class CameraService {
  late CameraController _controller;
  List<CameraDescription> _cameras = [];
  bool _ready = false;
  
  CameraController? get controller => _ready ? _controller : null;
  
  Future<void> initialize() async {
    _cameras = await availableCameras();
    if (_cameras.isEmpty) return;
    
    _controller = CameraController(
      _cameras[0], 
      ResolutionPreset.low,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.bgra8888
    );
    await _controller.initialize();
    _ready = true;
  }

  Future<void> beginCapture(Function(CameraImage image) onImageCaptured) async {
    await _controller.startImageStream((image) {
      onImageCaptured(image);
    });
  }

  Future<void> endCapture() async {
    await _controller.stopImageStream();
  }
  
  void dispose() {
    _controller.dispose();
  }
} 