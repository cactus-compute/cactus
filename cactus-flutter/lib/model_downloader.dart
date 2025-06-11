import 'dart:async';
import 'dart:io';
import 'dart:convert';

typedef ModelDownloadProgressCallback = void Function(double? progress, String statusMessage);

Future<void> downloadModel(
  String url,
  String filePath,
  {ModelDownloadProgressCallback? onProgress}
) async {
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
            '(${(receivedBytes / (1024 * 1024)).toStringAsFixed(2)}MB / ${(totalBytes / (1024 * 1024)).toStringAsFixed(2)}MB)'
          );
        } else {
          onProgress?.call(
            null, 
            'Downloading: ${(receivedBytes / (1024 * 1024)).toStringAsFixed(2)}MB received'
          );
        }
      }
      await fileSink.flush();
      await fileSink.close();
      
      onProgress?.call(1.0, 'Download complete. Saving file...');
      onProgress?.call(1.0, 'Model saved successfully to $filePath');
    } else {
      String responseBody = await response.transform(utf8.decoder).join();
      if (responseBody.length > 200) responseBody = "${responseBody.substring(0,200)}...";
      throw Exception(
          'Failed to download model. Status code: ${response.statusCode}. Response: $responseBody');
    }
    httpClient.close();
  } catch (e) {
    onProgress?.call(null, 'Error during download: $e');
    rethrow;
  }
}