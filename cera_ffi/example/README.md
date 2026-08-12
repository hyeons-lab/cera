# Examples

Standalone `dart:ffi` scripts: a CLI, a server, a test. No Flutter involved,
which is the point of this package existing apart from `cera_ffi_flutter`.

Nothing here bundles a native library, so point the loader at one:

```sh
# from the repo root
just dart-libs      # cargo build -p cera-ffi --features ffi-buffer

# from cera_ffi/
dart pub get

CERA_FFI_LIB=../target/debug/libcera_ffi.dylib \
  dart run example/cera_chat.dart /path/to/model.gguf "Why is the sky blue?"
```

| Script | Shows |
|---|---|
| `cera_chat.dart` | Chat template, tokenize, generate, decode back to text |
| `cera_generate.dart` | Minimal synchronous generate, token IDs only |
| `cera_async.dart` | `generateAsync` + `generateStreamingAsync` (recommended streaming path) |
| `cera_stream.dart` | Synchronous `generateStreaming`, and why you must drain the event loop |
| `cera_progress.dart` | `BundleRepo.withProgress` download progress callbacks, via `fromBundleIdAsync` (downloads a full bundle; it cannot be aborted) |

They print to stdout by design; `analysis_options.yaml` disables `avoid_print`
for that reason rather than excluding the directory from analysis, so these
still get type-checked.
