import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:cactus/cactus.dart';

/// Minimal utility for Cactus initialization with model download
class CactusInit {
  static Future<CactusContext?> init({
    required Function(String message, {double? progress}) onStatus,
  }) async {
    try {
      // Paths
      final docDir = await getApplicationDocumentsDirectory();
      final modelPath = '${docDir.path}/SmolVLM-256M-Instruct-Q8_0.gguf';
      final mmprojPath =
          '${docDir.path}/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf';

      // Download model if needed
      if (!await File(modelPath).exists()) {
        onStatus('Downloading model...', progress: 0.0);
        try {
          await _download(
            'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf',
            modelPath,
            (p) => onStatus(
              'Downloading model: ${(p * 100).toInt()}%',
              progress: p,
            ),
          );
        } catch (e) {
          onStatus('Error downloading model: $e', progress: 1.0);
          return null;
        }
      }

      // Download mmproj if needed
      if (!await File(mmprojPath).exists()) {
        onStatus('Downloading mmproj...', progress: 0.0);
        try {
          await _download(
            'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf',
            mmprojPath,
            (p) => onStatus(
              'Downloading mmproj: ${(p * 100).toInt()}%',
              progress: p,
            ),
          );
        } catch (e) {
          onStatus('Error downloading mmproj: $e', progress: 1.0);
          return null;
        }
      }

      // Initialize
      onStatus('Initializing Cactus...', progress: 0.5);

      bool hasError = false;
      final context = await CactusContext.init(
        CactusInitParams(
          modelPath: modelPath,
          mmprojPath: mmprojPath,
          gpuLayers: 0,
          onInitProgress: (_, status, isError) {
            if (isError) {
              hasError = true;
              onStatus('Error: $status', progress: 1.0);
            } else {
              onStatus('Initializing: $status', progress: 0.75);
            }
          },
        ),
      );

      if (hasError || context == null) {
        return null;
      }

      onStatus('Cactus initialized successfully!', progress: 1.0);
      return context;
    } catch (e) {
      onStatus('Error: $e', progress: 1.0);
      return null;
    }
  }

  // Simple download with progress
  static Future<void> _download(
    String url,
    String path,
    Function(double progress) onProgress,
  ) async {
    final client = HttpClient();
    try {
      final request = await client.getUrl(Uri.parse(url));
      final response = await request.close();

      if (response.statusCode != 200) {
        throw Exception('HTTP error ${response.statusCode}');
      }

      final output = File(path).openWrite();

      int total = response.contentLength > 0 ? response.contentLength : 0;
      int received = 0;

      onProgress(0.0); // Start progress

      await for (final data in response) {
        received += data.length;
        output.add(data);
        if (total > 0) onProgress(received / total);
      }

      await output.close();
      onProgress(1.0); // Complete progress
    } finally {
      client.close();
    }
  }
}
