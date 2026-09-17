# Swift/Kotlin GPU session ownership

Run from the worktree on macOS arm64 with working Metal and wgpu devices,
Xcode tools, `kotlinc`, JDK 21 and cached Cargo dependencies:

```bash
export PATH="/opt/homebrew/bin:$PATH"
export JAVA_HOME=/path/to/jdk-21
python3 tests/gpu_session_ffi/run.py --artifacts /path/to/cached-jars
```

The jar directory must contain the JNA and coroutines artifacts pinned in
[artifacts.json](../leap_compat/artifacts.json). The runner verifies their hashes
and downloads nothing. `CARGO_TARGET_DIR` is honored. It builds both GPU backends,
exports the core's existing tiny convolution/attention GGUF, and compiles the
current generated bindings plus [Swift](SessionProbe.swift) and
[Kotlin](SessionProbe.kt) consumers. Swift linkage/runtime checks and Kotlin's
absolute library override select the staged, hashed native library.

Each language must report all 51 cases. Metal and wgpu each exercise uncompressed
and TurboQuant KV: typed Busy (including conflicting configuration), extraction,
cancel/reset/error retention, release, failed construction, a successor matching
an independent model, and generation after releasing the parent engine. CPU
allows two live sessions sharing one engine. The `generateConversation` helper
is a runnable example of successive independent conversations on one GPU engine.
Keep a Session for ordinary continuation within the same conversation.

Models load through the existing `fromBytesAsync` constructor, which runs
initialization on native blocking workers. This is the UI-safe loading path.
The synchronous debug wgpu loader can overflow the small stack of a Swift
cooperative worker during Naga shader compilation; this suite does not establish
that synchronous path on small-stack foreign threads.

The async probe pauses inside a text callback with a bounded synchronization
barrier. It cancels generation, releases the caller's Session, and requires Busy
while the native call is pending. After releasing the callback and awaiting the
cancelled completion, a successor must succeed. It uses neither timing sleeps
nor probabilistic polling; a missing callback or hung call fails the run.

Every run creates a directory containing `report.json`, commands, logs and input,
fixture, dependency and artifact hashes. Inputs include both crates' sources,
embedded shaders, shader templates, build scripts and build support; adding,
removing or changing them during execution fails the source audit. The report
also snapshots inherited Cargo configuration, including ancestor and Cargo-home
files and absent paths; adding, removing or changing them fails the run. Results must
contain a list of all case names; missing cases, duplicates, maps of case names,
unavailable GPUs, compilation errors or incorrect results fail. The harness
controls run with:

```bash
python3 -m unittest discover -s tests/gpu_session_ffi -p 'test_*.py'
```

A separate negative control builds the native library with Session ownership
acquisition bypassed. Both unchanged consumers must fail at their first Metal
`Busy` assertion; the source and normal build are then restored, and both
consumers must pass the complete 51-case matrix again. The
[implementation handoff](../../docs/internals/API_RESHAPE_HANDOFF.md) records
the control script, exact loaded libraries and results.

This proves bounded macOS native behavior through the real bindings. It does not
establish Android/iOS device behavior, browser ownership, full API compatibility,
model quality or performance budgets. See the
[GPU API examples](../../docs/internals/API_RESHAPE_GPU_SESSION_EXAMPLES.md).
