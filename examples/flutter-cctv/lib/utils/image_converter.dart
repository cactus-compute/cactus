import 'package:camera/camera.dart';
import 'dart:io';
import 'package:image/image.dart' as img;

/// Utility for converting camera images to formats usable by Cactus
class ImageConverter {

  /// Convert a CameraImage to a proper JPEG File - simple version
  static Future<File> convertImageToFile(
    CameraImage image,
    String outputPath,
  ) async {
    // Create a simple grayscale image from Y plane
    final width = image.width;
    final height = image.height;
    final yPlane = image.planes[0].bytes;
    
    // Create image object
    final imgLib = img.Image(width: width, height: height);
    
    // Fill with grayscale data (just using Y channel)
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixelIndex = y * width + x;
        final yValue = yPlane[pixelIndex];
        imgLib.setPixelRgb(x, y, yValue, yValue, yValue); // grayscale (R=G=B)
      }
    }
    
    // Encode as JPG and write to file
    final jpgData = img.encodeJpg(imgLib, quality: 80);
    final file = File(outputPath);
    await file.writeAsBytes(jpgData);
    return file;
  }
  
  /// Get a unique filename for an image
  static String getUniqueImagePath(String directory) {
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    return '$directory/frame_$timestamp.jpg';
  }
}


