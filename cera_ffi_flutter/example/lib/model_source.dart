// Picking a model, and the platform rules that go with it.
//
// Opening a GGUF the user chose takes three rules that differ per platform:
// which form to ask the picker for, how to tell which form came back, and
// whether a load consumes the bytes it is given.
//
// The middle rule is the one worth centralizing. Testing `file.path` for null
// reads as the obvious way to ask whether a path is available, and it is wrong
// on the web, where the picker manufactures a `blob:` URL and puts it in that
// field: every browser load then goes to `Cera.openPath`, which the web does
// not implement. A rule written once can still be wrong, but it cannot be
// wrong in one place and right in another.

import 'dart:typed_data';

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show visibleForTesting;

/// A model that can be opened on different backends.
abstract class LoadedModel {
  /// The model's display name.
  String get name;

  /// Opens an engine instance for this model with the given [options].
  Future<Cera> open({CeraOptions options = const CeraOptions()});
}

/// A model the user picked, in whichever form this platform can open.
///
/// Which form that is follows [Cera.supportsPaths]: a path off the web, the
/// bytes themselves in a browser. Callers do not need to know which, and get
/// [open] rather than the two fields, so the choice stays in one place.
class ModelSource implements LoadedModel {
  ModelSource._({required this.name, String? path, Uint8List? bytes})
    : _path = path,
      _bytes = bytes,
      assert(
        (path == null) != (bytes == null),
        'a source is a path or bytes, never both and never neither',
      );

  @visibleForTesting
  ModelSource.forTesting({required String name, String? path, Uint8List? bytes})
    : this._(name: name, path: path, bytes: bytes);

  @override
  final String name;

  /// Non-null only where the engine can open a path, i.e. off the web.
  final String? _path;

  /// The whole model, held only when there is no [_path] to open instead.
  final Uint8List? _bytes;

  /// Opens this model.
  ///
  /// On the web a load may **transfer** the buffer it is handed into the worker,
  /// which neuters the caller's view of it, so `openBytes` receives a fresh
  /// sublist copy to ensure this source can be reused across chat turns and
  /// benchmark runs.
  @override
  Future<Cera> open({CeraOptions options = const CeraOptions()}) {
    final path = _path;
    if (path != null) return Cera.openPath(path, options: options);
    final bytes = _bytes!;
    // `sublist(0)` rather than `Uint8List.fromList`: both copy, and this one
    // states the intent, which is a fresh whole buffer.
    return Cera.openBytes(bytes.sublist(0), options: options);
  }
}

/// A published model bundle downloaded from the remote catalog.
class BundleModelSource implements LoadedModel {
  const BundleModelSource({
    required this.name,
    required this.bundleName,
    required this.quant,
    this.storeDir,
    this.getStoreDir,
  });

  @override
  final String name;

  /// The catalog bundle name (e.g. "LFM2-700M").
  final String bundleName;

  /// The quantization variant (e.g. "Q4_0").
  final String quant;

  /// Where bundle downloads are cached, if known synchronously.
  final String? storeDir;

  /// Async resolver for cache directory on platforms requiring it (Android/iOS).
  final Future<String?> Function()? getStoreDir;

  @override
  Future<Cera> open({CeraOptions options = const CeraOptions()}) async {
    final dir = storeDir ?? await getStoreDir?.call();
    return Cera.openBundle(bundleName, quant, options: options, storeDir: dir);
  }
}

/// A file was chosen but cannot be opened.
///
/// Its own type rather than a `StateError`, because both pages show what this
/// stringifies to. `StateError` prepends "Bad state: ", which is accurate about
/// the program and useless to someone who just picked a file.
class UnreadableModel implements Exception {
  const UnreadableModel(this.name);

  /// The file's display name.
  final String name;

  @override
  String toString() => 'could not read $name';
}

/// Shows the file picker and returns what the user chose.
///
/// Returns null when the dialog was cancelled, which is not a failure and is
/// worth telling apart from one. Throws when a file was chosen but cannot be
/// read, and lets the picker's own failures (a denied permission, a platform
/// channel that threw) propagate, so a caller has one place to report both.
Future<ModelSource?> pickModelSource({required String dialogTitle}) async {
  // `withData` matters on the web, where there is no path to open and the
  // picker has to hand over the bytes themselves. On native it would mean
  // reading a multi-gigabyte file into the Dart heap when the engine could have
  // mapped it, so ask for it only where it is the only option.
  final result = await FilePicker.platform.pickFiles(
    dialogTitle: dialogTitle,
    type: FileType.any,
    withData: !Cera.supportsPaths,
  );
  final file = result?.files.single;
  if (file == null) return null;

  // `Cera.supportsPaths`, not `file.path != null`. On the web the picker
  // manufactures a `blob:` URL for the bytes it just read and puts it in `path`
  // (`file_picker`'s `addPickedFile`: `path: path ?? blobUrl`), so that field is
  // non-null in a browser, and testing it would send every web load into
  // `Cera.openPath`, which is the one call the web does not have. Ask the API
  // which mode it supports instead, the same question `withData` just asked.
  final path = Cera.supportsPaths ? file.path : null;
  final bytes = file.bytes;
  if (path == null && bytes == null) {
    // Neither form came back. No picker backend is known to do this off the
    // web, since they all set a path, but the alternative to checking is a
    // bang on `bytes` that would crash instead of explaining.
    throw UnreadableModel(file.name);
  }
  return ModelSource._(
    name: file.name,
    path: path,
    // Dropped where a path is available: the engine maps the file, so keeping a
    // second copy of the weights in the heap would be paying twice for one
    // model. `withData` above already avoids reading them, this is the case
    // where the picker returned them anyway.
    bytes: path == null ? bytes : null,
  );
}
