import 'package:flutter/material.dart';

class CameraControls extends StatelessWidget {
  final bool isCapturing;
  final VoidCallback onToggleCapture;
  final VoidCallback onFlipCamera;

  const CameraControls({
    super.key,
    required this.isCapturing,
    required this.onToggleCapture,
    required this.onFlipCamera,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          ElevatedButton.icon(
            onPressed: onToggleCapture,
            icon: Icon(isCapturing ? Icons.stop : Icons.play_arrow),
            label: Text(isCapturing ? 'Stop' : 'Start'),
            style: ElevatedButton.styleFrom(
              backgroundColor: isCapturing ? Colors.red : Colors.green,
              foregroundColor: Colors.white,
            ),
          ),
          ElevatedButton.icon(
            onPressed: onFlipCamera,
            icon: const Icon(Icons.flip_camera_ios),
            label: const Text('Flip'),
          ),
        ],
      ),
    );
  }
}
