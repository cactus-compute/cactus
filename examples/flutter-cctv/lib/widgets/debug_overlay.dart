import 'package:flutter/material.dart';
import 'dart:io';

class DebugOverlay extends StatelessWidget {
  final File? imageFile;
  final String stats;
  
  const DebugOverlay({super.key, this.imageFile, required this.stats});
  
  @override
  Widget build(BuildContext context) {
    if (imageFile == null) return const SizedBox.shrink();
    
    return Container(
      color: Colors.black.withOpacity(0.8),
      padding: const EdgeInsets.all(8),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Image.file(imageFile!, height: 200),
          const SizedBox(height: 8),
          Text(stats, style: const TextStyle(color: Colors.white)),
        ],
      ),
    );
  }
} 