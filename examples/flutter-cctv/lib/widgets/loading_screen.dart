import 'package:flutter/material.dart';

class LoadingScreen extends StatelessWidget {
  final String status;
  final double? progress;

  const LoadingScreen({super.key, required this.status, this.progress});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          if (progress != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 40),
              child: LinearProgressIndicator(value: progress),
            )
          else
            const CircularProgressIndicator(),
          const SizedBox(height: 16),
          Text(status),
        ],
      ),
    );
  }
}
