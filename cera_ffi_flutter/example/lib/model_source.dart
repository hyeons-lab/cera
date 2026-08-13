// Picking a model, and the platform rules that go with it.
//
// Both pages of this example open a GGUF the user chose, and doing that right
// takes three rules that differ per platform: which form to ask the picker for,
// how to tell which form came back, and whether a load consumes the bytes it is
// given.
//
// The middle rule is the one worth centralizing. Testing `file.path` for null
// reads as the obvious way to ask whether a path is available, and it is wrong
// on the web, where the picker manufactures a `blob:` URL and puts it in that
// field: every browser load then goes to `Cera.openPath`, which the web does
// not implement. The chat page shipped that bug. A rule written once can still
// be wrong, but it cannot be wrong in one page and right in the other.

import 'dart:typed_data';

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
import 'package:file_picker/file_picker.dart';

/// A model the user picked, in whichever form this platform can open.
///
/// Which form that is follows [Cera.supportsPaths]: a path off the web, the
/// bytes themselves in a browser. Callers do not need to know which, and get
/// [open] rather than the two fields, so the choice stays in one place.
class ModelSource {
  ModelSource._({required this.name, String? path, Uint8List? bytes})
    : _path = path,
      _bytes = bytes,
      assert(
        (path == null) != (bytes == null),
        'a source is a path or bytes, never both and never neither',
      );

  /// The file's display name, for status lines.
  final String name;

  /// Non-null only where the engine can open a path, i.e. off the web.
  final String? _path;

  /// The whole model, held only when there is no [_path] to open instead.
  final Uint8List? _bytes;

  /// Opens this model.
  ///
  /// Set `reusable` when this source has to survive the call. On the web a load
  /// may **transfer** the buffer it is handed into the worker, which neuters
  /// the caller's view of it, so a second open of the same source would be
  /// handed an empty list. (Whether it transfers or copies depends on the
  /// list's shape and the compiler, and [Cera.openBytes] declines to promise
  /// either, which is reason enough not to rely on the buffer surviving.)
  /// Copying costs a second full-size allocation, so it is off by default: a
  /// page that opens a model once and keeps it, like the chat page, should not
  /// pay for a copy it will never read.
  Future<Cera> open({
    CeraOptions options = const CeraOptions(),
    bool reusable = false,
  }) {
    final path = _path;
    if (path != null) return Cera.openPath(path, options: options);
    final bytes = _bytes!;
    // `sublist(0)` rather than `Uint8List.fromList`: both copy, and this one
    // states the intent, which is a fresh whole buffer.
    return Cera.openBytes(
      reusable ? bytes.sublist(0) : bytes,
      options: options,
    );
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
