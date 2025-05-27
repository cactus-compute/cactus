import 'package:flutter/material.dart';

class StatusOverlay extends StatelessWidget {
  final String status;
  final bool isProcessing;

  const StatusOverlay({
    super.key,
    required this.status,
    this.isProcessing = false,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      bottom: 10,
      left: 10,
      right: 10,
      child: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: Colors.black54,
          border: isProcessing ? Border.all(color: Colors.greenAccent, width: 1.5) : null,
        ),
        child: Row(
          children: [
            if (isProcessing)
              const Padding(
                padding: EdgeInsets.only(right: 8),
                child: SizedBox(
                  width: 12,
                  height: 12,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: Colors.white,
                  ),
                ),
              ),
            Expanded(
              child: Text(
                status,
                style: const TextStyle(color: Colors.white),
                textAlign: TextAlign.center,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
