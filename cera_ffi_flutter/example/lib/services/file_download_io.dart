import 'dart:io';
import 'dart:typed_data';

/// Saves the given bytes to the local filesystem (e.g. current directory or Downloads).
Future<void> downloadFileBytes(
  Uint8List bytes, {
  required String filename,
  String mimeType = 'application/octet-stream',
}) async {
  Directory targetDir;
  try {
    final current = Directory.current;
    if (current.path != '/' &&
        (Platform.isMacOS || Platform.isLinux || Platform.isWindows)) {
      targetDir = current;
    } else {
      targetDir = Directory.systemTemp;
    }
  } catch (_) {
    targetDir = Directory.systemTemp;
  }
  try {
    final file = File('${targetDir.path}/$filename');
    await file.writeAsBytes(bytes);
  } catch (_) {
    final fallback = File('${Directory.systemTemp.path}/$filename');
    await fallback.writeAsBytes(bytes);
  }
}
