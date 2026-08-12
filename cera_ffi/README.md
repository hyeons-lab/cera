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

`Cera` is the portable API: one asynchronous surface that runs on every target,
web included.

```dart
import 'dart:io';

import 'package:cera_ffi/cera_ffi.dart';

Future<void> main() async {
  final cera = await Cera.openPath('/path/to/model.gguf');

  final prompt = await cera.applyChatTemplate(
    [const CeraMessage.user('Why is the sky blue?')],
  );
  await for (final piece in cera.generate(prompt, maxTokens: 128)) {
    stdout.write(piece);   // fragments, not tokens: just append them
  }

  await cera.close();
  exit(0);   // see "Platform support": a NativeCallable keeps the isolate alive
}
```

### Downloading a published model

`Cera.listBundles()` returns the models published on `LiquidAI/LeapBundles` as
`<name, quants>` pairs, which is what a picker needs, and `Cera.openBundle`
downloads one and opens it. Both work on every target, and the same catalog and
parser back `cera list-bundles`, so a menu shows what the CLI shows.

```dart
final bundles = await Cera.listBundles();
for (final b in bundles) {
  print('${b.displayName}: ${b.quants.join(" ")}');
}

final cera = await Cera.openBundle(
  bundles.first.name,     // the full id, not displayName
  bundles.first.quants.first,
  onProgress: (p) => print('${p.url} ${p.fraction ?? "?"}'),
);
```

Prefer this over fetching a `.gguf` yourself and calling `openBytes`. A bundle's
manifest names every file it needs and states its modality, so a vision or audio
model arrives complete rather than being guessed at from the arguments. It is
also the cheaper path in memory, and on the web dramatically so: the weights go
from the cache straight into the engine, one copy of the model instead of two,
which is also what keeps it clear of the roughly 2 GiB ceiling on a single
contiguous JavaScript allocation that `openBytes` runs into.

Downloads are cached, so re-opening the same pair is offline and instant.
`storeDir` chooses where, and is **required on Android and iOS**, where an app
may write only inside its own container: pass a path from `path_provider`. On
desktop it defaults to `$HOME/.cache/cera`, which the `cera` CLI also uses, so
the two share downloads.

The generated bindings underneath cover the whole engine (LoRA, vision, audio,
embeddings, grammars, tool calling, KV compression). Reach for them when you
need something `Cera` does not expose, and accept that they are native-only:
`dart:ffi` is what they are, and the web does not have it. Most of their
surface is synchronous, though `fromPathAsync`, `fromBytesAsync` and
`generateStreamingAsync` are what `Cera` is built on.

```dart
import 'package:cera_ffi/cera_ffi.dart';

void main() {
  final engine = CeraEngine.fromPath(
    '/path/to/model.gguf',
    const EngineConfig(
      contextSize: 2048,
      backend: BackendPreference.auto,
      bundleRepo: null,
    ),
  );

  final session = engine.newSession(const SessionConfig());
  session.appendTokens(engine.encodeText('Why is the sky blue?'));
  final out = session.generate(const GenerateOpts(maxTokens: 128));

  print(engine.decodeTokens(out.tokens));
}
```

`generate` returns token IDs plus a summary; `decodeTokens` turns them back into
text. For streaming and the async variants that keep the isolate responsive, see
`example/`.

## Platform support

`Cera` runs everywhere: Android, iOS, macOS, Linux, Windows, and the web.

The generated bindings run everywhere `dart:ffi` does, which is everywhere
except the web.

That split is not a gap someone forgot to close. A browser runs the engine in a
Web Worker; `postMessage` is asynchronous and a worker offers no synchronous
escape hatch, so a synchronous `engine.generate(...)` is not implementable there
at any price. `Cera` is the same engine behind an API whose shape a browser can
satisfy: a Rust async runtime on native, a Web Worker on web.

### On the web

Inference runs on **WebGPU** when the browser has it and falls back to a wasm
CPU build when it does not: ~58 tok/s against ~1.4 tok/s, measured on the same
machine and model. Install the runtime into your app's `web/` directory once
with `dart run cera_ffi:install_web`, then use `Cera.openBundle` (preferred, see
above) or `Cera.openBytes` for a model the user supplies; there is no filesystem
for `openPath`. Downloads are cached in the origin's private filesystem (OPFS),
so a second visit reopens the same model without the network. No COOP/COEP
headers are needed; nothing here uses threads. See the [`cera_ffi_flutter`
README](https://pub.dev/packages/cera_ffi_flutter) for the full setup and for
what is narrower there.

The **generated bindings** are still a stub on the web, and have to be: an
unconditional `dart:ffi` import fails the entire build rather than one branch,
so they are exported conditionally with a generated no-FFI branch. It is
produced by the same `just dart-bindings` run, from the same interface, and CI
compiles a web app against it, so it cannot drift. Data types are real there
(`EngineConfig`, `GenerateOpts`, the error hierarchy and the enums construct
and compare normally) while engine entry points throw `UnsupportedError` naming
themselves.

### Exiting a Dart script

`Cera.generate` registers a callback interface, and the vtable behind it holds
static `NativeCallable`s for the process's lifetime. A live `NativeCallable`
keeps its isolate alive, so a plain Dart CLI has to call `exit()` even after
`close()`. Flutter apps are running anyway and never notice.

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
