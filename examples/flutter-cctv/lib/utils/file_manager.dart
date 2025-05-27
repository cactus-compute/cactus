import 'dart:io';
import 'package:path_provider/path_provider.dart';

/// Utility for managing temporary files used during inference
class FileManager {
  /// Get a directory for temporary image files
  static Future<Directory> getTemporaryDirectory() async {
    return await getTemporaryDirectory();
  }

  /// Create a temporary file with a unique name
  static Future<File> createTemporaryFile({
    String? prefix,
    String? extension,
  }) async {
    final tempDir = await getTemporaryDirectory();
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final fileName = '${prefix ?? 'temp'}_$timestamp.${extension ?? 'jpg'}';
    return File('${tempDir.path}/$fileName');
  }

  /// Delete a file safely
  static Future<bool> deleteFile(File file) async {
    try {
      if (await file.exists()) {
        await file.delete();
        return true;
      }
      return false;
    } catch (e) {
      print('Error deleting file ${file.path}: $e');
      return false;
    }
  }

  /// Delete multiple files safely
  static Future<int> deleteFiles(List<File> files) async {
    int deletedCount = 0;
    for (final file in files) {
      if (await deleteFile(file)) {
        deletedCount++;
      }
    }
    return deletedCount;
  }

  /// Clean all temporary files with a specific prefix
  static Future<int> cleanTemporaryFiles({String prefix = 'frame_'}) async {
    try {
      final tempDir = await getTemporaryDirectory();
      final files = tempDir.listSync().whereType<File>().where(
        (file) => file.path.contains(prefix),
      );

      int deletedCount = 0;
      for (final file in files) {
        try {
          await file.delete();
          deletedCount++;
        } catch (e) {
          print('Failed to delete ${file.path}: $e');
        }
      }

      return deletedCount;
    } catch (e) {
      print('Error cleaning temporary files: $e');
      return 0;
    }
  }
}
