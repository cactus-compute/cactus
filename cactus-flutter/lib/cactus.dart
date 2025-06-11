library cactus;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:path_provider/path_provider.dart';

export 'chat.dart';
export 'init_params.dart';
export 'completion.dart';
export './context.dart';
export './model_downloader.dart';
export './structs.dart';

export 'grammar.dart';   
export 'tools.dart'; 



typedef ImageDownloadProgressCallback = void Function(double? progress, String statusMessage);

Future<String> downloadImage(
  String url,
  {String? filePath,
  ImageDownloadProgressCallback? onProgress}
) async {
  onProgress?.call(null, 'Starting image download from: $url');
  
  String effectiveFilePath;
  if (filePath != null && filePath.isNotEmpty) {
    effectiveFilePath = filePath;
    final dir = Directory(effectiveFilePath.substring(0, effectiveFilePath.lastIndexOf('/')));
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
  } else {
    final Directory tempDir = await getTemporaryDirectory();
    String filename = url.split('/').last;
    if (filename.contains('?')) filename = filename.split('?').first;
    if (filename.isEmpty || !filename.contains('.')) {
      final uuid = Uuid();
      filename = '${uuid.v4()}.jpg';
    }
    effectiveFilePath = '${tempDir.path}/$filename';
  }

  final File imageFile = File(effectiveFilePath);

  try {
    final httpClient = HttpClient();
    final request = await httpClient.getUrl(Uri.parse(url));
    final response = await request.close();

    if (response.statusCode >= 200 && response.statusCode < 300) {
      final IOSink fileSink = imageFile.openWrite();
      final totalBytes = response.contentLength;
      int receivedBytes = 0;

      onProgress?.call(0.0, 'Connected. Receiving image data...');

      await for (var chunk in response) {
        fileSink.add(chunk);
        receivedBytes += chunk.length;
        if (totalBytes != -1 && totalBytes != 0) {
          final progress = receivedBytes / totalBytes;
          onProgress?.call(
            progress,
            'Downloading image: ${(progress * 100).toStringAsFixed(1)}% ' 
            '(${(receivedBytes / 1024).toStringAsFixed(1)}KB / ${(totalBytes / 1024).toStringAsFixed(1)}KB)'
          );
        } else {
          onProgress?.call(
            null, 
            'Downloading image: ${(receivedBytes / 1024).toStringAsFixed(1)}KB received'
          );
        }
      }
      await fileSink.flush();
      await fileSink.close();
      
      onProgress?.call(1.0, 'Image download complete. Saved to $effectiveFilePath');
      return effectiveFilePath;
    } else {
      String responseBody = await response.transform(utf8.decoder).join();
      if (responseBody.length > 200) responseBody = "${responseBody.substring(0,200)}...";
      throw Exception(
          'Failed to download image. Status: ${response.statusCode}. Response: $responseBody');
    }
  } catch (e) {
    onProgress?.call(null, 'Error during image download: $e');
    if (await imageFile.exists()) {
      await imageFile.delete().catchError((_) => imageFile);
    }
    rethrow;
  } finally {
  }
}

class Uuid {
  final _random = Random.secure();
  String v4() {
    final bytes = List<int>.generate(16, (i) => _random.nextInt(256));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    return bytes.map((b) => b.toRadixString(16).padLeft(2, '0')).join();
  }
}