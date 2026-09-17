# Native ingestion recovery checks

These consumers use production Swift/Kotlin bindings and real GGUF execution.
Each language runs nine cases: None/F16/TurboQuant KV and direct, combined
send/generate, or streaming whole-message ingestion. They check unchanged
prevalidation failures, restored/reset cancellation, retained cancellation,
report lifetime, owned snapshots, fresh-model continuation and `Busy` in a token callback invoked under the lock.
The CPU run also compiles and executes both public recovery examples in standard
and compressed modes.

Build the native library and regenerate bindings with the repository's existing
`just bindings` and `just dart-bindings` recipes. For device runs, build the library
with `cera-ffi` features `gpu,metal,ffi-buffer`. The runner takes an explicit dylib
and a two-token `a`/`b` hybrid LFM2 fixture with at least 32 tokens of context (the
existing GPU ownership fixture's `conversation.gguf`). This synthetic fixture
proves execution and recovery contracts, not language quality.

```sh
python3 tests/api_recovery/run.py \
  --library /absolute/path/libcera_ffi.dylib \
  --model /absolute/path/conversation.gguf \
  --output /private/tmp/cera-recovery-results
# Repeat with --backend metal and --backend wgpu on supported hardware.
```

The runner uses Java 21, the pinned JNA/coroutines jars from
`tests/leap_compat/artifacts.json` (default directory
`/private/tmp/cera-leap-api-baseline`; override with `--dependencies`), and the
checked-in bindings. It stages and hashes the library, verifies Swift's actual
loaded library, and pins Kotlin's library override. Each run records commands,
source/artifact hashes and results in a fresh directory. No device fallback or
successful skip is accepted. Run device consumers with GPU access.

Rust FFI unit tests additionally inject a failed reset, preserving original
cancellation alongside a typed 64-bit allocation failure, and check same-thread
reentrancy, concurrent lock contention, poison and wide rewind payloads:

```sh
cargo test -p cera-ffi --lib recovery::tests
```

[Contract and limitations](../../docs/internals/API_RESHAPE_RECOVERY.md#native-recovery-status).
Raw append semantics, new chat phases, browser async recovery, and warm-KV budgets
remain separate gates. Python/Dart wrappers are regenerated; Swift/Kotlin are the
consumer runtime targets of this probe.
