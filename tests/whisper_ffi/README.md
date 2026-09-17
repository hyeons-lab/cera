# Native Whisper binding probes

Run from the worktree on macOS arm64 with Xcode tools, `kotlinc` and JDK 21:

```bash
export JAVA_HOME=/path/to/jdk-21
python3 tests/whisper_ffi/run.py --artifacts /path/to/cached-jars
```

The artifact directory must contain the JNA and coroutines jars pinned in
[`artifacts.json`](../leap_compat/artifacts.json). The harness verifies their
SHA256 values and does not download dependencies. Cargo dependencies must
already be cached. `CARGO_TARGET_DIR` is honored; the library is selected from
Cargo's build output and copied into the run directory before execution.

The runner generates synthetic Whisper GGUF/PCM inputs, compiles the actual
generated Swift/Kotlin bindings, and checks 11 cases per language with exact
`aaa`/`aa`/`bb` results. It checks file/byte ownership, defaults, language
metadata, sync and concurrent async calls, empty input and typed loading errors.
Each run writes `report.json`, command logs and artifact hashes under a new
temporary directory. Failures return nonzero; no fixture test is optional.

See the [application examples and limitations](../../docs/internals/API_RESHAPE_WHISPER_EXAMPLES.md).
These CPU fixtures prove API execution, not speech quality or Android/iOS
device behavior. Regenerate wrappers with `just bindings` and `just dart-bindings`
after changing the Rust exports.
