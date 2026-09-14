# Leap export probes

Export fixtures and optional CPU native boundary probes for C0. Nothing here is
a distributable compatibility SDK. Production dependencies are unchanged. See
the [protocol and native evidence](../../docs/internals/API_RESHAPE_LEAP_BRIDGE.md)
and the [earlier export experiment](../../docs/internals/API_RESHAPE_LEAP_EXPORTS.md).

From the implementation worktree, on macOS arm64:

```sh
export PATH="/opt/homebrew/bin:$PATH"
export JAVA_HOME="/Users/dberrios/.sdkman/candidates/java/21.0.9-zulu"
python3 tests/leap_compat/run_probes.py --fetch --with-kmp --with-native
```

The first run downloads the public, SHA256-pinned artifacts in
[artifacts.json](artifacts.json) and resolves Kotlin/SKIE build dependencies.
Subsequent runs can omit `--fetch`: artifact downloads are then disabled and
the KMP build uses Gradle's offline mode. The optional native lane also needs
the pinned JNA JAR and existing offline Cargo dependencies. A cached artifact with a mismatched
checksum always fails; even `--fetch` does not replace it silently. Use
`--cache /private/tmp/cera-leap-api-baseline` to reuse the initial verified cache.
Downloaded binaries, generated frameworks and evidence logs stay in ignored
`build/` directories or the explicit cache directory.

The runner requires Python 3.10+, `curl` for downloads, Xcode for Swift, and
`kotlinc` on PATH (or `--kotlinc /path/to/kotlinc`). Tested with Swift 6.3.3 in
Swift 5 language mode targeting macOS 15, Kotlin/JVM compiler 2.4.0 and JDK 21.
The KMP candidate pins Kotlin 2.3.20, SKIE 0.10.11 and coroutines 1.10.2; it
reuses the repository's Gradle 9.5.1 wrapper. SKIE analytics are disabled.

`--platform swift` or `--platform kotlin` runs just that reference lane and
does not require the other compiler. `--with-kmp` requires both platforms and
builds stable and extended framework/JAR profiles before checking consumers.
Both profiles compile unchanged core, conversation, parser-subclass, payload,
stream, enum, and custom-runner fixtures. Both profiles share experimental
LoRA/hidden-state data types; extended adds the newer ModelRunner methods and
consumer coverage. Its Swift custom runners require those newer methods. This is an export experiment,
not a decision to ship two packages. The newer Kotlin positive public baseline
is still missing; its candidate check alone cannot establish compatibility.
Custom Swift runner fixtures cover both imported async and completion-handler
conformance forms; neither can be dropped from the later replacement matrix.

`--with-native` requires `--with-kmp`. It builds the existing `cera-ffi` release
library with Cargo offline, compiles the checked-in Swift/Kotlin bindings, and
runs separate processes against a generated 30,400-byte GGUF. No model or Leap
runtime is downloaded for these executions. `--cera-target /path/to/target`
can reuse an existing Cargo target directory; the default is this worktree's
`target/`. Rust and its dependencies must already be installed/cached. Omitting
`--with-native` keeps the harness compile-only.

Each run prints a unique `build/run-*/results.json` location. The report records
fixture hashes, original artifact pins, tool versions, compiler commands,
outcomes, and per-profile KMP output hashes when enabled. Native evidence also
records model/library/binding hashes and the cross-language result. Adjacent logs preserve diagnostics.
Temporary extraction/compiler directories are removed; rerun the harness to
recreate them. An expected rejection counts only when the compiler exits with
status 1 and reports every specified diagnostic. An unrelated failure or crash
fails the run. A successful report means the matrix matched its expectations,
including known incompatibilities; it does not mean C0 or runtime validation is
complete. `native_boundary.validated` covers only the documented raw CPU checks;
the full-SDK `runtime_validated` and `c0_complete` flags remain false.

The Swift stream fixtures are unchanged across the published artifacts and the
candidate modules. Kotlin consumers are unchanged between Maven and the matching
candidate. The two handwritten Swift candidates deliberately lack parts of
SKIE's contract and must be rejected. The KMP candidate contains abstract
protocols, selected data types and convenience methods. It has no ModelRunner
implementation; loading options, parsers, media and serialization remain
incomplete. Never put these candidate artifacts in an app.

Local checks for this directory:

```sh
python3 -m unittest discover -s tests/leap_compat -p 'test_*.py'
ruff check tests/leap_compat/*.py
ruff format --check tests/leap_compat/*.py
xcrun swift-format lint --strict --recursive tests/leap_compat/swift tests/leap_compat/native/NativeProbe.swift tests/leap_compat/candidates/*.swift tests/leap_compat/candidates/kmp/src/commonMain/swift
env -u JAVA_TOOL_OPTIONS ktlint 'tests/leap_compat/kotlin/*.kt' 'tests/leap_compat/native/*.kt' 'tests/leap_compat/candidates/kmp/src/**/*.kt' 'tests/leap_compat/candidates/kmp/*.kts'
```
