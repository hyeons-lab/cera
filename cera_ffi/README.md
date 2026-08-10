# cera_ffi

Dart bindings for the [Cera](https://github.com/hyeons-lab/cera) on-device
inference engine. Runs GGUF language models locally through `dart:ffi`.

**Building a Flutter app? Use
[`cera_ffi_flutter`](https://pub.dev/packages/cera_ffi_flutter) instead.** It
depends on this package and re-exports it, so the API is identical, and it adds
the per-platform build wiring that ships the native library with your app. This
package on its own does not carry a native library.

This one exists for Dart without Flutter: a CLI, a server, a test. It is also
where the bindings live so that they *can* be resolved without Flutter, since a
package declaring `flutter.plugin.platforms` is one `dart pub get` refuses.

## Install

```sh
dart pub add cera_ffi
```

Then point it at a `cera-ffi` shared library, since nothing bundles one for you:

```sh
# built from the repo: cargo build -p cera-ffi --release --features ffi-buffer
export CERA_FFI_LIB=/path/to/libcera_ffi.dylib
```

Without `CERA_FFI_LIB` the loader falls back to the platform's normal search
path (`libcera_ffi.so`, `libcera_ffi.dylib`, `cera_ffi.dll`).

The `ffi-buffer` cargo feature is not optional. UniFFI emits the
`uniffi_ffibuffer_*` trampolines the Dart bindings call only under that feature,
and a library built without it loads cleanly and then fails at the first call
with a `dlsym` error naming a symbol you never wrote.

## Use

```dart
import 'package:cera_ffi/cera_ffi.dart';

void main() {
  final engine = CeraEngine(
    modelPath: '/path/to/model.gguf',
    config: const EngineConfig(),
  );

  final out = engine.generate(
    prompt: 'Why is the sky blue?',
    opts: const GenerateOpts(maxTokens: 128),
  );

  print(engine.decodeTokens(tokens: out.tokens));
}
```

`generate` returns token IDs plus a summary; `decodeTokens` turns them back into
text. For streaming and the async variants that keep the isolate responsive, see
`example/`.

## Platform support

Everything `dart:ffi` supports: Android, iOS, macOS, Linux, Windows.

The web compiles but runs nothing. `dart:ffi` does not exist there, and importing
it anywhere on the graph fails the entire build rather than one branch, so the
bindings are exported conditionally and the web branch is a **generated stub**
with the same API. It is produced by the same `just dart-bindings` run, from the
same interface, and CI compiles a web app against it, so it cannot drift.

Data types are real on the web: `EngineConfig`, `GenerateOpts`, the error
hierarchy and the enums construct and compare normally, so code that only builds
a request stays platform-agnostic. Engine entry points throw `UnsupportedError`
naming themselves. Use `cera-wasm` for inference in a browser.

## Bindings

`lib/src/generated/cera_ffi.dart` is generated from the compiled `cera-ffi`
cdylib by a vendored `uniffi-bindgen-dart`, then run through a deterministic
patch tool. The same run emits `lib/src/generated/cera_ffi_web.dart`, the
no-FFI stub `lib/cera_ffi.dart` falls back to on the web. Both are committed, so
the package works out of the box. Regenerate from the repo root after any change
to the Rust FFI surface:

```sh
just dart-bindings         # regenerate + patch
just dart-bindings-check   # verify nothing drifted, then analyze
```

Drift is not cosmetic: UniFFI checksums every method at construction, so
bindings that lag the Rust side make the engine throw before the first call.
The stub can drift too, in its own way, so the generator has tests asserting the
two files expose the same members and CI compiles a web app against the stub.

## License

Apache-2.0 OR MIT, matching the rest of the workspace.
