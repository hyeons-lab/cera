import 'dart:js_interop';
import 'dart:typed_data';

@JS('ceraDownloadFile')
external void _ceraDownloadFile(
  JSUint8Array bytes,
  JSString filename,
  JSString mimeType,
);

/// Triggers a browser file download for the given bytes.
Future<void> downloadFileBytes(
  Uint8List bytes, {
  required String filename,
  String mimeType = 'application/octet-stream',
}) async {
  _ceraDownloadFile(bytes.toJS, filename.toJS, mimeType.toJS);
}
