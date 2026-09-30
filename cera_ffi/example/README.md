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
| `chat.dart` | Native `ChatSession` with phase-checked continuation, retained KV state, and reclamation after token-limit interruption |
| `explicit_loading.dart` | Explicit model loading with `ModelLoader` and raw prompt completion |
| `cera_chat.dart` | Chat template, tokenize, generate, decode back to text |
| `cera_generate.dart` | Minimal synchronous generate, token IDs only |
| `cera_async.dart` | `generateAsync` + `generateStreamingAsync` (recommended streaming path) |
| `cera_stream.dart` | Synchronous `generateStreaming`, and why you must drain the event loop |
| `cera_progress.dart` | `BundleRepo.withProgress` download progress callbacks, via `fromBundleIdAsync` (downloads a full bundle; it cannot be aborted) |
| `gpu_ownership_probe.dart` | Native GPU context reservation, Busy error handling, and session release |

They print to stdout by design; `analysis_options.yaml` disables `avoid_print`
for that reason rather than excluding the directory from analysis, so these
still get type-checked.

`chat.dart` continues only after `SessionPhase.turnComplete`. If its token budget
leaves the turn `interrupted`, it reclaims Session and exits rather than ingesting
another user message into an incomplete turn. Reset or replace messages to start
again. See the [API guide](../../docs/API_0_6.md) for lifecycle, streaming,
schema and checkpoint limits. These native examples require a matching library
built with `ffi-buffer`; their generated bindings do not run on the web.
