import 'package:flutter/material.dart';

class CameraInitializing extends StatelessWidget {
  final String status;
  final VoidCallback onInitializeCamera;

  const CameraInitializing({
    super.key,
    required this.status,
    required this.onInitializeCamera,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const CircularProgressIndicator(),
          const SizedBox(height: 16),
          Text(status),
          const SizedBox(height: 24),
          ElevatedButton(
            onPressed: onInitializeCamera,
            child: const Text('Initialize Camera'),
          ),
        ],
      ),
    );
  }
}
