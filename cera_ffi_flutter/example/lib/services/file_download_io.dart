import 'dart:io';
import 'dart:typed_data';

/// Saves the given bytes to the local filesystem (e.g. current directory or Downloads).
Future<void> downloadFileBytes(
  Uint8List bytes, {
  required String filename,
  String mimeType = 'application/octet-stream',
}) async {
  final dir = Directory.current.path;
  final file = File('$dir/$filename');
  await file.writeAsBytes(bytes);
}
