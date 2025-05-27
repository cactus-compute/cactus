import 'package:camera/camera.dart';
import 'dart:io';
import 'package:image/image.dart' as img;
import '../utils/timer.dart';

/// Utility for converting camera images to formats usable by Cactus
class ImageConverter {

  /// Convert a CameraImage to a proper JPEG File - simple version
  static Future<File> convertImageToFile(
    CameraImage cameraImage,
    String outputPath,
  ) async {

    if (cameraImage.format.group != ImageFormatGroup.bgra8888) {
      timer.log('Error: Expected BGRA8888 format, but got ${cameraImage.format.group}');
      throw Exception('Expected BGRA8888 format, but got ${cameraImage.format.group}');
    }

    if (cameraImage.planes.isEmpty) {
      timer.log('Error: Image has no planes!');
      throw Exception('Image has no planes!');
    }

    final file = File(outputPath);
    final plane = cameraImage.planes[0];

    try {
    final image = img.Image.fromBytes(
      width: cameraImage.width,
      height: cameraImage.height,
      bytes: plane.bytes.buffer, // Pass the ByteBuffer
      rowStride: plane.bytesPerRow,
      order: img.ChannelOrder.bgra,
      // alpha: img.Alpha.last,
    );

    final jpegBytes = img.encodeJpg(image, quality: 85);

    await file.writeAsBytes(jpegBytes);

    return file;

  } catch (e) {
    print("Error converting BGRA to JPEG: $e");
    throw Exception('Error converting BGRA to JPEG: $e');
  }
      
  }
  
  /// Get a unique filename for an image
  static String getUniqueImagePath(String directory) {
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    return '$directory/frame_$timestamp.jpg';
  }
}


