import 'dart:async';

import 'package:universal_io/io.dart';

/// Downloads a model file from the given [url] to the specified [filePath].
///
/// This function is used internally by [CactusContext.init] if a `modelUrl` is provided,
/// but can also be used directly if manual download management is needed.
///
/// - [url]: The URL from which to download the model.
/// - [filePath]: The local path where the downloaded model file will be saved.
/// - [onProgress]: An optional callback to monitor download progress.
///   It receives the download progress (0.0 to 1.0, or null if indeterminate)
///   and a status message.
Future<void> downloadModel(String url, String filePath,
    {void Function(double? progress, String statusMessage)? onProgress}) async {
  onProgress?.call(null, 'Starting download for: ${filePath.split('/').last}');
  final File modelFile = File(filePath);

  try {
    final httpClient = HttpClient();
    final request = await httpClient.getUrl(Uri.parse(url));
    final response = await request.close();

    if (response.statusCode == 200) {
      final IOSink fileSink = modelFile.openWrite();
      final totalBytes = response.contentLength;
      int receivedBytes = 0;

      onProgress?.call(0.0, 'Connected. Receiving data...');

      await for (var chunk in response) {
        fileSink.add(chunk);
        receivedBytes += chunk.length;
        if (totalBytes != -1 && totalBytes != 0) {
          final progress = receivedBytes / totalBytes;
          onProgress?.call(
              progress,
              'Downloading: ${(progress * 100).toStringAsFixed(1)}% '
              '(${(receivedBytes / (1024 * 1024)).toStringAsFixed(2)}MB / ${(totalBytes / (1024 * 1024)).toStringAsFixed(2)}MB)');
        } else {
          onProgress?.call(null,
              'Downloading: ${(receivedBytes / (1024 * 1024)).toStringAsFixed(2)}MB received');
        }
      }
      await fileSink.flush();
      await fileSink.close();

      onProgress?.call(1.0, 'Download complete. Saving file...');
      onProgress?.call(1.0, 'Model saved successfully to $filePath');
    } else {
      throw Exception(
          'Failed to download model. Status code: ${response.statusCode}');
    }
    httpClient.close();
  } catch (e) {
    onProgress?.call(null, 'Error during download: $e');
    rethrow;
  }
}
