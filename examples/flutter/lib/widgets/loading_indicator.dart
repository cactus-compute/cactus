import 'package:flutter/material.dart';

class LoadingIndicator extends StatelessWidget {
  final bool isLoading;
  final bool isBenchmarking; // Added to differentiate benchmark loading
  final double? downloadProgress;
  final String statusMessage;

  const LoadingIndicator({
    super.key,
    required this.isLoading,
    required this.isBenchmarking,
    this.downloadProgress,
    required this.statusMessage,
  });

  @override
  Widget build(BuildContext context) {
    if (!isLoading && !isBenchmarking) {
      return const SizedBox.shrink(); // Nothing to show if not loading
    }

    return Expanded(
      child: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (isBenchmarking) // Specific indicator for benchmarking
              const CircularProgressIndicator()
            else if (downloadProgress != null && downloadProgress! < 1.0)
              LinearProgressIndicator(
                value: downloadProgress,
                minHeight: 10,
              )
            else // General loading or post-download initialization
              const CircularProgressIndicator(),
            const SizedBox(height: 20),
            Text(
              statusMessage,
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16, color: Colors.grey[700]),
            ),
          ],
        ),
      ),
    );
  }
} 