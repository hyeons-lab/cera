# Plain-Dart examples

These are **not** part of the Flutter example app. They are standalone
`dart:ffi` scripts showing that `cera_ffi_flutter` runs from plain Dart: a CLI,
a server, a test. Nothing under `lib/` imports `package:flutter`, and no
Flutter machinery is involved at run time.

One caveat, because it is easy to read the above as more than it says.
Dependency *resolution* does need Flutter installed: pub refuses to publish a
package declaring `flutter.plugin.platforms` without a Flutter SDK constraint,
and declaring one makes `dart pub get` reject the package ("requires the
Flutter SDK, version solving failed"). So run `flutter pub get` once; `dart
run` and `dart test` work normally afterwards.

Because there is no Flutter app to bundle the native library, point them at a
locally built one:

```sh
# from the repo root
just dart-libs

# from cera_ffi_flutter/
flutter pub get

CERA_FFI_LIB=../target/debug/libcera_ffi.dylib \
  dart run example/dart/cera_chat.dart /path/to/model.gguf "Why is the sky blue?"
```

| Script | Shows |
|---|---|
| `cera_chat.dart` | Chat template, tokenize, generate, decode back to text |
| `cera_generate.dart` | Minimal synchronous generate, token IDs only |
| `cera_async.dart` | `generateAsync` + `generateStreamingAsync` (recommended streaming path) |
| `cera_stream.dart` | Synchronous `generateStreaming`, and why you must drain the event loop |
| `cera_progress.dart` | `BundleRepo.withProgress` download progress callbacks |

They are excluded from the example app's analyzer (see
`../analysis_options.yaml`): they print to stdout by design, which
`avoid_print` rejects for app code.
