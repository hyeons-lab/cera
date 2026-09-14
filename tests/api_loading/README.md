# Generated loading API checks

This harness runs the public Rust loading API through production UniFFI
Swift/Kotlin and wasm-bindgen Node bindings. Native legacy Probe controls remain
alongside the production loader; Node uses the production CPU WASM loader and
Session. No package release is claimed.
See the [binding audit](../../docs/internals/API_RESHAPE_P0_BINDINGS.md) for
contracts and limits.

For concrete typed loading, session continuation, vision/draft, Swift and Kotlin
examples, see the [executable API guide](../../docs/internals/API_RESHAPE_EXAMPLES.md).
The [standalone Node example](../../cera-wasm/examples/explicit_loading.cjs)
also runs against a production-only generated module; see the [build instructions](../../cera-wasm/README.md#explicit-cpu-model-loading).
The [Rust walkthrough](consumer/src/bin/walkthrough.rs) runs separately inside
the prepared mirror and checks two generations at positions 5 and 9. It is an
additional example; the 21-command probe report does not include that run.
The [audio walkthrough](consumer/src/bin/audio_walkthrough.rs) also runs separately; its [fixture export and run instructions](../../docs/internals/API_RESHAPE_AUDIO_EXAMPLE.md) cover real CPU encoder/vocoder execution with synthetic weights.
The [remote companion examples](../../docs/internals/API_RESHAPE_REMOTE_EXAMPLES.md) run as library tests with a local HTTP server; they do not require a generated consumer workspace.
The [HF revision examples](../../docs/internals/API_RESHAPE_HF_EXAMPLES.md) load changed GGUF revisions into separate cache entries, verify defaults and keep a CPU session live across another revision load. They also exercise failed metadata resolution and explicit commit checks.
The [SafeTensors conversion examples](../../docs/internals/API_RESHAPE_CONVERSION_EXAMPLES.md) exercise real conversion, corruption repair and option-sensitive cached reloads. The examples verify checkpoint bytes, pin inputs to one upstream commit, and refresh changed revisions. Retained models execute after replacement, and sessions continue after their parent model is released.
The [persistent cache examples](../../docs/internals/API_RESHAPE_CACHE_EXAMPLES.md) demonstrate same-path weight replacement, unchanged-weight disk reuse and retained live sessions.

Run on macOS arm64 with Rust plus the wasm32-unknown-unknown target, Swift,
Kotlin 2.4.0, Zulu JDK 21.0.9, Node 24 and wasm-bindgen 0.2.117:

```bash
export PATH="/opt/homebrew/bin:$PATH"
unset JAVA_TOOL_OPTIONS
python3 tests/api_loading/test_harness.py
python3 tests/api_loading/run.py --target /Users/dberrios/development/cera/target/api-loading
```

`--target` selects a parent directory and defaults to `tests/api_loading/build/target`.
Each run creates an exclusive `run-*` child as its actual Cargo target directory;
the report's `target` field records that path. This prevents concurrent probe
runs from replacing one another's native outputs. Keep it separate from production
build targets, and do not direct external builds into an active run's directory.
Artifacts are hashed immediately after each build or generation step and checked
again after execution. `--wasm-bindgen` selects the exact 0.2.117 CLI; its
default is the local wasm-pack cache. `--jna` selects JNA 5.16.0; its default
is `/private/tmp/cera-leap-api-baseline/jna-5.16.0.jar`. The runner validates the
[public dependency pin](../leap_compat/artifacts.json). Download that public
artifact separately if missing; build dependencies stay offline. Native runtime
probes download synthetic fixtures only from a loopback server. Cargo uses
offline metadata to resolve local package changes, then locked offline builds;
every registry dependency must already occur in the repository lock.

`--prepare-only` writes the source mirror without claiming a passing build.
`--build-only` skips runtime consumers. Each run creates an ignored `build/run-*`
directory with the mirror, bindings, fixture, stdout/stderr per command and
`results.json`. A full pass requires 21 command expectations (two deliberate
Rust compiler rejections), complete consumer case sets and matching generation.
The harness rejects unexpected diagnostics, failures and final input/artifact
drift. Do not edit probe or Cera inputs during a run.

The mirror uses the public loader without changing visibility or implementation.
It appends a core session-config observation method and stages the actual
`cera-ffi` and `cera-wasm` source trees,
adding test accessors. Shared engine storage is production code in both paths. The
production config records, converters, sessions and generation methods are reused.
A harness test reverses all four observation-only file adaptations and requires byte equality with
the originals. Original, adapted, mirror, probe, dependency and generated artifact
hashes are recorded; added or removed binding source files also count as drift.
Native checks explicitly target `aarch64-apple-darwin`. The runner selects native
and WASM outputs from Cargo's artifact records for the expected source files,
so an inherited target override cannot select an older `target/debug` output.
Ancestor and Cargo-home configuration files (including their absence) are
snapshotted and checked again; selected inherited Rust/Cargo flags are recorded.
Relative `--wasm-bindgen` paths resolve from the invocation directory.

Ten harness tests include independent build directories for a shared target
parent, artifact provenance and configuration drift controls,
plus a timeout regression where a descendant ignores SIGTERM after its parent
exits. Cleanup sends SIGKILL to remaining process-group members in that case.
Probe commands remove inherited `DYLD_*`, `LD_LIBRARY_PATH`, `LD_PRELOAD`,
JVM/Kotlin option variables and Node preload/module-path overrides so consumers
cannot substitute an older library for the selected Cargo artifact. The report
records removed variable names. The Swift mutation case allocates independent
storage and asserts that zeroing changes the originally passed allocation.

Each native consumer requires 63 cases; Node requires 26. Path/file loading and
native configuration consumers run in Swift/Kotlin; Node also runs the native
conversion helper at 32-bit pointer width. The architecture-only `kws.gguf` fixture must produce
`KindMismatch(expected: "Generative", actual: "Hotword", architecture: "kws")`
through both `build` and `buildGenerative` in all three languages.
These checks request unavailable Metal, so the kind error must precede backend
construction. A second attempt must report `Consumed` after either failure.
The full examples run in [Swift](consumers/LoadingProbe.swift),
[Kotlin](consumers/LoadingProbe.kt) and [Node](consumers/loading_probe.cjs).
This validates error dispatch; the header fixture contains no hotword weights
and does not execute a detector or expose a typed hotword loader.

The result parser requires a list of case-name strings and rejects missing,
duplicate or unexpected cases, including missing `kind-kws`. A dictionary whose
keys name every case and whose values are all false also fails. These malformed
record controls run with `python3 tests/api_loading/test_harness.py` above.

For scoped Rust checks, use the report's mirror directory as the working directory
and its exact `target` field as `CARGO_TARGET_DIR`:

```bash
cargo clippy -p loading-native --target aarch64-apple-darwin --all-targets --locked --offline -- -D warnings
cargo clippy -p loading-web --target wasm32-unknown-unknown --lib --locked --offline -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p cera -p loading-native --target aarch64-apple-darwin --lib --no-deps --locked --offline
RUSTDOCFLAGS="-D warnings" cargo doc -p loading-web --target wasm32-unknown-unknown --lib --no-deps --locked --offline
```

Run format/lint on authored fixtures, excluding generated bindings:

```bash
rustfmt --edition 2024 --check tests/api_loading/shared.rs tests/api_loading/defaults.rs tests/api_loading/native/src/lib.rs tests/api_loading/native/src/bin/generate.rs tests/api_loading/wasm/src/lib.rs tests/api_loading/consumer/src/bin/*.rs
uvx --offline ruff check tests/api_loading/*.py
uvx --offline ruff format --check tests/api_loading/*.py
xcrun swift-format lint --strict tests/api_loading/consumers/*.swift
ktlint tests/api_loading/consumers/*.kt
node --check tests/api_loading/consumers/loading_probe.cjs
node --check tests/api_loading/consumers/production_probe.cjs
```

The exhaustive Rust fixture intentionally does not compile in the normal mirror;
do not use a whole-mirror `cargo check --all-targets` as a positive gate. The
runner verifies its E0004 failure, removes only `#[non_exhaustive]` and checks
success, restores the attribute in `finally`, then requires the failure again.

## Production session access

The production `GenerativeModel.createSession` method returns the existing `Session`.
Its `engine()` method shares the loaded core engine. The generated tests verify
engine identity against a separately loaded negative control, release parent
handles, and then continue generation on independent sessions. Native coverage
also routes all six source variants through production `EngineConfig` conversion;
remote sources reuse the loopback fixtures' populated caches and `BundleRepo`.
Generated backend converters round-trip every production variant and reject
malformed wire tags. Both builders reject unavailable GPU/Metal selections with
structured assembly errors, then reject reuse of the consumed loader.

The complete [Swift](consumers/ProductionProbe.swift) and
[Kotlin](consumers/ProductionProbe.kt) examples exercise default and explicit
`SessionConfig`, F16/TurboQuant settings, streaming cancellation, reset, context
overflow and the original `FfiError` payloads. Swift compiles both generated
components into the probe's `loading_native` module; Kotlin imports production
types from `uniffi.cera_ffi`. Kotlin requires the pinned coroutines 1.10.2 jar
alongside JNA, both checked against [the artifact pins](../leap_compat/artifacts.json).

The [Node example](consumers/production_probe.cjs) uses the existing CPU WASM
`SessionConfig` and `GenerateOpts` classes. With the generated module and the
harness's two-token GGUF fixture:

```javascript
const loader = new api.ModelLoader(api.ModelSource.bytes(bytes), new api.LoadConfig(24, 'cpu'));
const model = loader.buildGenerative();
const config = new api.SessionConfig();
config.maxSeqLen = 8;
config.seed = 42n;
const session = model.createSession(config);
config.free();
model.free();
loader.free();
session.appendTokens(new Uint32Array([0, 1]));
const opts = new api.GenerateOpts();
opts.maxTokens = 1;
opts.temperature = 0.7;
opts.ignoreEos = true;
const summary = session.generate(opts, tokens => console.log(Array.from(tokens)));
console.log(summary.tokensGenerated, session.position); // 1, 3
summary.free();
opts.free();
session.free();
```

Token IDs `[0, 1]` are specific to the synthetic fixture. Real applications use
the production tokenizer or text input methods. Seeded sampling can continue in
a subsequent generation call without an append. The core currently clears its
cached logits after a greedy decoded token, so repeated greedy generation needs
another append; resolving that behavior remains part of chat/recovery work.
These functional tests do not establish a KV performance budget.

CPU WASM retains its existing `LoadConfig(u32, String)` and session properties;
its production binding has no F16 or `gpuDepthformer` session setter. It reports
session failures as `Error` messages. WASM cancellation is tested between calls:
the production JS borrow check rejects reentrant session methods during a token
callback. Browser factories and WebGPU APIs retain their existing paths.

## Native multipart file example

The complete Swift and Kotlin probes also construct `Source.files` / `Source.Files`
with all eight `ProbeModelFiles` fields. For example, using the generated Swift module:

```swift
let files = ProbeModelFiles(
  model: modelPath, multimodalProjector: nil, audioDecoder: nil,
  audioTokenizer: nil, draftModel: nil, extras: [:],
  inferenceType: "llama.cpp/text-to-text", chatTemplate: nil)
let loader = ProbeModelLoader(source: .files(files: files), config: config())
let model = try loader.buildGenerative()
let session = try model.createSession()
try session.append(tokens: [0, 1])
let tokens = try session.generate()
```

`modelPath` is the absolute path to the runner's `model.gguf`; `config()` is the
helper in the [complete Swift probe](consumers/LoadingProbe.swift). The
[complete Kotlin probe](consumers/LoadingProbe.kt) uses
`ProbeModelLoader(Source.Files(files), config()).buildGenerative()`.
Run the command above to generate the module, compile both examples and execute
them against the synthetic model. These imports exist only in the probe mirror.

The file cases exercise both builders with explicit and inferred text. They
check normalized relative auxiliary paths, unchanged absolute paths, named
extras and separately retained template metadata, then generate `[0, 1, 0]` at
position 5 after the loader and all model handles are released. Missing optional
auxiliaries remain nonfatal. Missing primary, Hotword kind mismatch and explicit
unsupported inference each reject both builder calls and leave the loader
consumed. The unsupported inference error precedes opening a missing primary;
kind classification precedes unavailable Metal. Native `files()` observes the
resolved core manifest for these assertions; it is a probe accessor, not a new
production API. No optional auxiliary weights execute in these cases.

See the [loading exit audit](../../docs/internals/API_RESHAPE_LOADING_EXIT.md)
for actual promotion blockers and later chat, platform and release checks.

## Native context configuration example

The generated Swift configuration can omit context size and optional settings:

```swift
let defaults = LoadConfig(backend: "cpu") // contextSize: 4096
let full = LoadConfig(contextSize: 0, backend: "cpu") // use model capacity
let loader = ProbeModelLoader(source: .bytes(bytes: bytes), config: full)
let model = try loader.buildGenerative()
let info = model.info()
```

In Kotlin, the equivalent configuration is
`LoadConfig(contextSize = 0uL, backend = "cpu")`; omitting `contextSize` selects
4096. `bytes` is the runner's `model.gguf` loaded as `Data` / `ByteArray`, as in
the complete probes linked above. Run the same full probe command to compile
and execute these configurations through both builders.

Native context and its observation use `UInt64` / `ULong`. Conversion to the
platform's `usize` is checked; zero selects its maximum sentinel, then the core
caps allocation at the model's context. The fixture declares 64 tokens. Omitted
context reports requested 4096 and capacity64; zero reports64 and capacity64.
Values64,65 and4294967320 preserve their requested value and cap allocation at64.
On a 64-bit native target, UInt64.max/ULong.MAX_VALUE shares the same sentinel and
reports64, matching the existing native accessor. `requestedContext` therefore
reports the model limit for a sentinel, not the literal original zero/MAX input.
A context of1 rejects a second appended token with position still1.
Other profiles generate [0,1,0] at position5 after all parents are released.

WASM `LoadConfig.context_size` and `ModelInfo.requested_context` remain u32 JS
Numbers, with their existing behavior. A separate probe-only
`native_context_size_for_probe(BigInt)` calls the same native conversion helper
on wasm32: zero and0xffffffff succeed, while0x100000000 andu64::MAX reject.
This is real 32-bit helper execution, not an Android native loader or a change
to WASM loading semantics. The helper now returns InvalidConfig with field context_size, reason out_of_range
and an exact decimal-string value. Its display text is diagnostic only.

## Structured loading error example

The complete generated consumers identify failures by category and fields:

```swift
let bad = ProbeModelLoader(source: .bytes(bytes: bytes), config: config("invalid-probe"))
do {
  _ = try bad.buildGenerative()
} catch let ProbeLoadError.InvalidConfig(field, value, reason, detail) {
  precondition(field == "backend" && value == "invalid-probe")
  precondition(reason == "unknown_backend" && !detail.isEmpty)
}
consumed(bad)
```

`bytes`, `config` and `consumed` are defined in the complete Swift probe linked
above. Its Kotlin equivalent catches `ProbeLoadException.InvalidConfig` and reads
`field`, `value`, `reason` and `detail`. Both compiled examples also require
`ProbeLoadException.Source.sourceKind` / Swift `ProbeLoadError.Source` for malformed bytes
and missing files, `Assembly.backend` for initialization failures, and
`UnsupportedInferenceType.inferenceType` for an explicit unsupported type.
All failures consume the loader, through both builder methods.

| Category | Stable payload | Meaning |
|---|---|---|
| `Source` | `source_kind` | Parsing or resolution of bytes, reader, parts, path, files, bundle or hf source failed. Rust retains the original CeraError; foreign callers receive diagnostic detail. |
| `Assembly` | `backend` | Model/tokenizer/weight/backend initialization failed after primary parsing and kind checking. The field is the requested preference (`Auto`, `Cpu`, `Gpu`, `Metal`), not proof of which backend actually ran. |
| `UnsupportedInferenceType` | `inference_type` | Explicit or resolved inference type is unsupported. It takes precedence over the generic phase category. |
| `InvalidConfig` | `field`, `value`, `reason` | Current fields are backend/context_size; reasons unknown_backend/out_of_range. Values are exact strings, including u64::MAX, to avoid precision loss. |
| `KindMismatch` | `expected`, `actual`, `architecture` | Known non-generative input cannot satisfy generative loading. |
| `UnsupportedArchitecture` | `architecture` | Architecture is not classified as a supported kind. |
| `Consumed` | None | An attempt already used this loader. |

Node receives a JS Error with `code` equal to the category and the snake_case
properties in the table. For example:

```javascript
assert.throws(() => bad.buildGenerative(), error =>
  error.code === 'InvalidConfig' && error.field === 'backend' &&
  error.value === 'invalid-probe' && error.reason === 'unknown_backend');
```

The complete Node probe constructs `bad` inside its error-case loop and runs
both methods. It also checks exact decimal values for wasm32 overflow without
parsing display text. Error details remain useful diagnostics; their wording
is not a category or a stable payload. `Assembly` does not distinguish missing
weights from unavailable backend features internally. Rust legacy cause types
and text remain available, and existing production error shapes are unchanged.
The legacy native `ProbeSession` methods still forward generic Engine errors;
production sessions return native `FfiError` or CPU WASM `Error`. Engine
also remains a fallback for unknown future
core load variants; this is not a future-version ABI guarantee.

## Native remote loading example

The full command above builds and runs the [Swift remote consumer](consumers/RemoteProbe.swift)
and [Kotlin remote consumer](consumers/RemoteProbe.kt), with a fresh local HTTP
server and cache directory for each language. It requires permission to bind a
loopback port. No public model download or production binding change is involved.

This Swift fragment uses the fixture arguments passed by the runner:

```swift
let root = URL(fileURLWithPath: CommandLine.arguments[2])
let repo = ProbeBundleRepo(storeDir: root.appendingPathComponent("example-store").path)
let loader = ProbeModelLoader(
  source: .huggingFace(spec: "fixture/text:Q4_K_M@release", quant: "Q8_0", strategy: "hqq"),
  config: LoadConfig(contextSize: 24, backend: "cpu", bundleRepo: repo))
let model = try loader.buildGenerative()
let session = try model.createSession()
try session.append(tokens: [0, 1])
let tokens = try session.generate() // [0, 1, 0], position 5
```

The Kotlin equivalent, in the runner's process with its loopback HF endpoint:

```kotlin
val root = File(args[1])
ProbeBundleRepo(File(root, "example-store").absolutePath).use { repo ->
    ProbeModelLoader(
        Source.HuggingFace("fixture/text:Q4_K_M@release", "Q8_0", "hqq"),
        LoadConfig(contextSize = 24uL, backend = "cpu", bundleRepo = repo),
    ).use { loader ->
        loader.buildGenerative().use { model ->
            model.createSession().use { session ->
                session.append(listOf(0u, 1u))
                check(session.generate() == listOf(0u, 1u, 0u))
            }
        }
    }
}
```

Use `ProbeBundleRepo.withProgress(storeDir:progress:)` in Swift or
`ProbeBundleRepo.withProgress(storeDir, progress)` in Kotlin to attach an implementation
of `ProbeDownloadProgressSink`. Its `onProgress` receives URL, downloaded bytes and
optional total length. The callback runs on the thread performing synchronous
loading; dispatch UI changes to the UI thread. The complete consumers record
real events across the 256 KiB reporting threshold, require the exact 600 KiB
final count, and verify that cached reloads emit no new events.

The loaders retain the supplied repository after the caller releases its handle.
A model's probe-only `repositoryForProbe` and repository `resolveForProbe` methods
observe the actual retained core repository and trigger another local download
after model/loader release. They prove callback retention; they are not proposed
public API additions. Swift also observes callback release after all repositories
are gone. Kotlin uses deterministic native-handle closure; JVM garbage collection
timing is not asserted. Sessions still generate after all parent handles close.

Both builders cover HF revision/explicit-quant selection, cached bundle ID/quant,
manifest-file and directory loading, missing repository, remote source failure,
Hotword mismatch before backend construction, assembly failure and invalid bundle
quant. The fixture's alternate quant is a Hotword header, so losing the explicit
Q8_0 selection fails. `strategy` is forwarded, but GGUF loading does not execute a
SafeTensors quantizer; conversion strategies retain their separate core evidence.
Cached bundle manifests are seeded at their real cache paths; public HTTPS HEAD
probes are rejected by a local proxy. This does not establish live catalog/CDN
behavior, cancellation, asynchronous loading, Android/iOS packages or browser
remote loading. Web remains bytes/parts; production BundleRepo cache management
and download APIs remain supported in their existing binding home.

## Multipart generation defaults

The production bindings represent the complete core defaults enum. Legacy Swift and
Kotlin use `ProbeGenerationDefaults.Text`, `Audio` and `Other` (Swift cases are lower
camel case). `ProbeSamplingDefaults` still carries the five text sampling fields;
Audio adds optional decoding-thread count, audio temperature and audio top-k.
Other accepts JSON text, which becomes the core JSON value rather than an opaque
string. WASM uses `GenerationDefaults.text`, `.audio` and `.other` factories.

These fragments belong inside the linked complete consumers, with their loaded
`bytes` fixture and `config()` helper:

```swift
let sampling = ProbeSamplingDefaults(
  temperature: 0.375, topP: 0.75, topK: 7, minP: 0.125, repetitionPenalty: 1.25)
let parts = ProbeModelParts(
  model: bytes, multimodalProjector: nil, audioDecoder: nil, audioTokenizer: nil,
  draftModel: nil, inferenceType: nil, chatTemplate: nil,
  generationDefaults: .audio(
    sampling: sampling, numberOfDecodingThreads: 3, audioTemperature: 0.625, audioTopK: 11))
let loader = ProbeModelLoader(source: .parts(parts: parts), config: config())
let model = try loader.buildGenerative()
// Test observation of the loaded manifest; this helper is not a promoted API.
let retained = model.generationDefaultsForProbe()
```

```kotlin
val sampling = ProbeSamplingDefaults(0.375f, 0.75f, 7u, 0.125f, 1.25f)
val parts = ProbeModelParts(
    bytes, null, null, null, null, null, null,
    ProbeGenerationDefaults.Audio(sampling, 3u, 0.625f, 11u),
)
ProbeModelLoader(Source.Parts(parts), config()).use { loader ->
    loader.buildGenerative().use { model ->
        check(model.generationDefaultsForProbe() == parts.generationDefaults)
    }
}
```

`runDefaults` in the [Swift](consumers/LoadingProbe.swift) and
[Kotlin](consumers/LoadingProbe.kt) consumers, and the
[Node consumer](consumers/loading_probe.cjs), exercise both builders. They inspect
the retained core manifest, then release loader/handle/model parents and execute
a retained session. Cases cover absent defaults, present empty Text/Audio,
all audio and sampling fields, zero/u32 boundaries, and Other objects, arrays,
scalars and null. Other JSON is preserved as a JSON value; insignificant spacing
and object-key ordering are not retained by the core representation.

Malformed JSON produces `InvalidConfig` with field
`generation_defaults.raw_json`, the rejected value and reason `invalid_json`;
the failed builder is consumed. This tests defaults transport on a tiny text
model, not audio decoding, accuracy, device packages or public API promotion.
Plan31 runtime validation and review status are in the
[current handoff](../../docs/internals/API_RESHAPE_HANDOFF.md).
