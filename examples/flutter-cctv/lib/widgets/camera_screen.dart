import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:io';
import 'camera_controls.dart';
import 'status_overlay.dart';

class CameraScreen extends StatelessWidget {
  final CameraController cameraController;
  final String status;
  final File? lastCapturedImage;
  final bool isCapturing;
  final bool isInferenceRunning;
  final VoidCallback onToggleCapture;
  final VoidCallback onFlipCamera;

  const CameraScreen({
    super.key,
    required this.cameraController,
    required this.status,
    this.lastCapturedImage,
    required this.isCapturing,
    required this.isInferenceRunning,
    required this.onToggleCapture,
    required this.onFlipCamera,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: Container(
            width: double.infinity,
            child: Stack(
              fit: StackFit.expand,
              children: [
                // Camera preview
                ClipRect(
                  child: Center(
                    child: AspectRatio(
                      aspectRatio: cameraController.value.aspectRatio,
                      child: CameraPreview(cameraController),
                    ),
                  ),
                ),

                // Last captured frame overlay
                if (lastCapturedImage != null && isCapturing)
                  Positioned(
                    top: 10,
                    right: 10,
                    width: 120,
                    height: 120,
                    child: Container(
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.white, width: 2),
                      ),
                      child: Image.file(lastCapturedImage!, fit: BoxFit.cover),
                    ),
                  ),

                // Status overlay
                StatusOverlay(status: status, isProcessing: isInferenceRunning),

                // Processing indicator
                if (isInferenceRunning)
                  Positioned(
                    top: 10,
                    left: 10,
                    child: Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Colors.black54,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
        CameraControls(
          isCapturing: isCapturing,
          onToggleCapture: onToggleCapture,
          onFlipCamera: onFlipCamera,
        ),
      ],
    );
  }
}
