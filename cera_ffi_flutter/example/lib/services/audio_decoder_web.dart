import 'dart:async';
import 'dart:js_interop';
import 'dart:typed_data';

@JS('window.ceraDecodeAudioBlob')
external JSPromise<JSFloat32Array>? _ceraDecodeAudioBlob(
  JSString blobUrl,
  JSNumber targetSampleRate,
);

/// Decodes an audio blob URL into single-channel Float32 PCM samples via the browser's Web Audio API.
Future<Float32List?> decodeAudioBlob(String blobUrl, int sampleRate) async {
  try {
    final promise = _ceraDecodeAudioBlob(blobUrl.toJS, sampleRate.toJS);
    if (promise == null) return null;
    final jsFloats = await promise.toDart;
    return jsFloats.toDart;
  } catch (err) {
    return null;
  }
}
