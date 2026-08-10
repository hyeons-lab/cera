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

Not the web. The generated bindings import `dart:ffi` unconditionally, so an app
targeting web that depends on this package fails to *compile* rather than
failing at run time. Use `cera-wasm` in browsers.

## Bindings

`lib/src/generated/cera_ffi.dart` is generated from the compiled `cera-ffi`
cdylib by a vendored `uniffi-bindgen-dart`, then run through a deterministic
patch tool. It is committed, so the package works out of the box. Regenerate
from the repo root after any change to the Rust FFI surface:

```sh
just dart-bindings         # regenerate + patch
just dart-bindings-check   # verify nothing drifted, then analyze
```

Drift is not cosmetic: UniFFI checksums every method at construction, so
bindings that lag the Rust side make the engine throw before the first call.

## License

Apache-2.0 OR MIT, matching the rest of the workspace.
