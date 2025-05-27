import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:cactus/cactus.dart';

/// Minimal utility for Cactus initialization with model download
class CactusInit {
  static Future<CactusContext?> init({
    required Function(String message) onStatus,
  }) async {
    try {
      // Paths
      final docDir = await getApplicationDocumentsDirectory();
      final modelPath = '${docDir.path}/SmolVLM-500M-Instruct-Q8_0.gguf';
      final mmprojPath =
          '${docDir.path}/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf';

      // Download model if needed
      if (!await File(modelPath).exists()) {
        onStatus('Downloading model...');
        try {
          await _download(
            'https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf',
            modelPath,
            (p) => onStatus(
              'Downloading model: ${(p * 100).toInt()}%',
            ),
          );
        } catch (e) {
          onStatus('Error downloading model: $e');
          return null;
        }
      }

      // Download mmproj if needed
      if (!await File(mmprojPath).exists()) {
        onStatus('Downloading mmproj...');
        try {
          await _download(
            'https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf',
            mmprojPath,
            (p) => onStatus(
              'Downloading mmproj: ${(p * 100).toInt()}%',
            ),
          );
        } catch (e) {
          onStatus('Error downloading mmproj: $e');
          return null;
        }
      }

      // Initialize
      onStatus('Initializing Cactus...');

      bool hasError = false;
      final context = await CactusContext.init(
        CactusInitParams(
          modelPath: modelPath,
          mmprojPath: mmprojPath,
          gpuLayers: 0,
          onInitProgress: (_, status, isError) {
            if (isError) {
              hasError = true;
              onStatus('Error: $status');
            } else {
              onStatus('[CactusContext.init]: $status');
            }
          },
        ),
      );

      if (hasError || context == null) {
        return null;
      }

      return context;
    } catch (e) {
      onStatus('Error: $e');
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
