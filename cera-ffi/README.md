# cera-ffi

UniFFI bindings for [`cera`](../cera/): exposes the core inference
engine to Kotlin, Swift, Python, and every other language
[`uniffi-rs`](https://mozilla.github.io/uniffi-rs/) supports.

> **Note:** Version 0.6.0 introduces consolidated session lifecycle management, transactional multi-turn chat coordination (`ChatSession`), language-native reactive streaming across Swift, Kotlin, Python, and Dart, native JSON Schema compilation, first-class tool calling, session checkpointing, and a unified audio pipeline. See [Releases](https://github.com/hyeons-lab/cera/releases/tag/v0.6.0).

Concrete [Swift/Kotlin GPU lifetime examples](../docs/internals/API_RESHAPE_GPU_SESSION_EXAMPLES.md#swift-and-kotlin-conversation-lifetimes)
and an [executable native ownership probe](../tests/gpu_session_ffi/README.md)
cover session release, Busy errors and pending async work.

`ChatSession.recoveryStatus()` and `Session.recoveryStatus()` report what happened
after a failed ingestion or whole-message append: context was unchanged, restored, reset,
or left unusable. The original call still throws its existing error; a failed reset is
retained separately as `lastIngestRecovery.resetError`. Read the status after the operation
returns. It throws `Busy` while an operation holds the session lock. Terminal callbacks
and final buffered text flushes can run after that lock is released, so status
may already be available inside those callbacks.

For conversational chat, use the dedicated `ChatSession` coordinator via `session.intoChat()`
or `engine.newChatSession(config)`. The complete [Swift](examples/Chat.swift),
[Kotlin](examples/Chat.kt), and [Python](examples/chat.py) examples demonstrate multi-turn
chat with delta-only prefill, bit-exact KV cache retention, and session reclamation.
The legacy [Swift](examples/IngestionRecovery.swift) and [Kotlin](examples/IngestionRecovery.kt)
examples demonstrate raw append recovery.
See the [recovery contract and target limits](../docs/internals/API_RESHAPE_RECOVERY.md)
and [executable native checks](../tests/api_recovery/README.md).

## Status

**Cache management.** Through PRs 2–13 `cera-ffi` built up a typed
foreign-language surface, mobile cross-compile pipelines, remote
model loading + download progress callbacks, and tokenizer +
chat-template access. PR 14 closes the operational gap mobile apps
need to ship: `BundleRepo::cache_size()` to drive a "Storage: X MB"
UI line, and `BundleRepo::clear_cache()` to wipe downloaded models
when the user runs out of disk. No more "shell out to delete the
filesystem tree manually" workaround.

| PR | Scope |
|---|---|
| 1 | Crate shell + UniFFI scaffolding + smoke-test export |
| 2 | `CeraEngine::from_path`, `EngineConfig`, `ModelMetadata`, `ModalityCapabilities` |
| 3 | `Session`, `SessionConfig`, `GenerateOpts`, `GenerateSummary`, sync `generate` |
| 4 | `ModalitySink` as UniFFI foreign-trait callback + streaming `generate` |
| 5 | `async` `generate_async` + `generate_streaming_async` via `#[uniffi::export(async_runtime = "tokio")]` |
| 6 | Kotlin + Swift binding generation + vendored outputs + CI drift check; UniFFI 0.28 → 0.31.1 |
| 7 | Typed `FfiError` variants mirroring `cera::CeraError` |
| 8 | Android ABI cross-compile + CI matrix + per-ABI artifact upload |
| 9 | Apple-platform XCFramework: arm64-only iOS device + iOS Simulator + native macOS slices, CI artifact |
| 10 | `BundleRepo` remote loading (`remote` feature) + `CeraEngine::from_bundle_id` |
| 11 | `CeraEngine::from_bundle_id_async` via `spawn_blocking` + `AbortOnDrop` |
| 12 | `DownloadProgressSink` foreign-trait callback + `BundleRepo::with_progress` |
| 13 | Tokenizer + chat-template surface on `CeraEngine` (encode/decode, `ChatMessage`, `apply_chat_template`) |
| 14 | `BundleRepo::cache_size` + `clear_cache` for mobile cache mgmt |
| 15 | Parity harness (`cera-parity` Kotlin/Swift legs + perf gate) |
| 16+ | Session-API expansion: `Session::append_audio` placeholder, `Session::clear_cancel`, `CeraEngine::is_special_token`, `CeraEngine::context_size` resolved getter |
| 17+ | Hidden-states extraction: `Session::hidden_states_for_tokens` / `_for_text` (LE-f32 `Data`/`ByteArray`), `hidden_states_mean_pooled` (`[Float]`), `hidden_size` |
| 18+ | LoRA adapters: `LoraAdapters` object (`from_gguf` / `from_safetensors`), `Session::attach_lora` / `remove_lora` / `has_lora`, `FfiError::LoraParse`, `FfiError::LoraUnsupportedByBackend` |
| 19+ | Maven Central (`com.hyeons-lab:cera-ffi-{jvm,android}`) + SwiftPM remote publishing (`.package(url:)` against a prebuilt `CeraFFI.xcframework`); both shipped |
| 20+ | Native Keyword Spotting (KWS): `FfiHotwordConfig`, `FfiHotwordScore`, `FfiHotwordEvent`, `FfiHotwordDetector`, and `FfiHotwordIterator` (`process_chunk`, `reset`) |
| 21+ | OpenAI Whisper ASR: `FfiWhisperModel`, `FfiWhisperTranscribeOpts`, `whisper_default_transcribe_opts` with synchronous/asynchronous transcription and cooperative cancellation on Rust future drop |
| 22+ | Conversational Chat: `ChatSession`, `Message`, `Role`, `SessionPhase`, `TurnResult`, `Session::into_chat`, `CeraEngine::new_chat_session`, wait-free cancellation, and streaming decode |
| 23+ | Reactive Streaming: `AsyncThrowingStream` (Swift), `Flow` (Kotlin), `Iterator` generator (Python), and `Stream` (Dart) |
| 24+ | Structured Outputs: JSON Schema compilation to GBNF, `GenerateOpts.withJsonSchema`, and `completeJson` |
| 25+ | First-Class Tool Calling: `ChatSession.setTools`, `ingestToolResponse`, and automatic grammar triggers |
| 26+ | Session Checkpointing: binary snapshot export and import, atomic file persistence, and multi-turn state resumption |
| 27+ | Unified Audio Pipeline: `FfiAudioPipeline` uniting Silero VAD v5, Keyword Spotting, and Whisper ASR |

Don't add FFI exposure to `cera` directly. The `cera` crate keeps its
idiomatic Rust surface, and everything UniFFI-specific lives here.

## Explicit model loading

This checkout exposes `ModelSource`, `ModelLoader`, `ModelHandle` and
`GenerativeModel` in the generated native bindings.
A loader is single-use, including after a failed build. Loading is synchronous;
run it on a worker thread in a UI application. Existing engine constructors remain
available.

Swift, with `import Cera` and a local generative GGUF:

```swift
let loader = ModelLoader(
  source: .path(path: modelPath), config: EngineConfig(backend: .cpu))
let model = try loader.buildGenerative()
let engine = model.engine()
let session = try model.createSession(config: SessionConfig(seed: 42))
try session.appendTokens(tokens: engine.encodeText(text: "The capital of France is"))
let output = try session.generate(opts: GenerateOpts(maxTokens: 32, temperature: 0.7))
print(engine.decodeTokens(tokens: output.tokens))
```

Kotlin, with the types imported from `uniffi.cera_ffi`:

```kotlin
ModelLoader(ModelSource.Path(modelPath), EngineConfig(backend = BackendPreference.CPU)).use { loader ->
    loader.buildGenerative().use { model ->
        model.engine().use { engine ->
            model.createSession(SessionConfig(seed = 42uL)).use { session ->
                session.appendTokens(engine.encodeText("The capital of France is"))
                val output = session.generate(GenerateOpts(maxTokens = 32u, temperature = 0.7f))
                println(engine.decodeTokens(output.tokens))
            }
        }
    }
}
```

These examples request a raw completion; they do not apply a chat template.
Complete command-line sources with imports and arguments are in
[ExplicitLoading.swift](examples/ExplicitLoading.swift) and
[ExplicitLoading.kt](examples/ExplicitLoading.kt).
`Bytes`, `Parts`, `Files`, `HuggingFace` and `BundleId` provide other explicit
sources. Remote resolution uses `EngineConfig.bundleRepo`. `Parts` preserves
companion bytes, inference type, chat template and Text/Audio/Other generation
defaults; `Files` preserves companion paths and extras. A build reports structured
`LoadError` in Swift or `LoadException` in Kotlin. Existing session methods retain
`FfiError`/`FfiException`.

`model.engine()` shares the loaded core engine, including its tokenizer and cache.
A session keeps its resources when the model and engine wrappers are released.
Reuse that session to continue with its live KV, subject to the backend ownership
rules below. This API does not establish new KV performance budgets.

## Sharing a loaded GPU model

Metal and wgpu permit one live `Session` per loaded model. A second session
returns `Busy` until the first is dropped; reset and cancellation keep ownership.
CPU models continue to support shared weights across concurrent sessions. Load
separate GPU models for simultaneous conversations. The
[session ownership walkthrough](../docs/internals/API_RESHAPE_GPU_SESSION_EXAMPLES.md) shows the public API, cleanup behavior
and executable CPU/Metal/wgpu checks.

LFM2-Audio transcription during a live GPU conversation uses a cached secondary
model built from retained weights. This preserves conversation KV and costs
additional model memory and first-use setup; the walkthrough includes a native
Dart consumer for transcription and seeded generation.

## Whisper in Swift and Kotlin

`FfiWhisperModel` exposes file/byte loading, synchronous and asynchronous
transcription, `languages()` and `isMultilingual()` in the generated wrappers.
Use a Whisper GGUF and decoded 16 kHz mono float PCM. For example, within a Swift
async function using `import Cera`, after loading `model` off the UI thread:

```swift
var opts = whisperDefaultTranscribeOpts()
opts.language = "en"
let text = try await model.transcribeAsync(pcm: pcm16kMono, opts: opts)
```

The Kotlin equivalent, inside a suspend function with a loaded model:

```kotlin
val opts = whisperDefaultTranscribeOpts().copy(language = "en")
val text = model.transcribeAsync(pcm16kMono, opts)
```

The [complete recording helpers and executable probes](../docs/internals/API_RESHAPE_WHISPER_EXAMPLES.md)
show imports, background loading, byte ownership and Kotlin `use` cleanup. They
also document the 30-second input limit, option defaults and cancellation:
dropping an async transcription future aborts queued work and signals a running
decoder to stop at its next cooperative cancellation check.
Kotlin coroutine cancellation frees that future. The pinned Swift wrapper has
no task cancellation handler, so `Task.cancel()` alone leaves transcription
running; the guide records this remaining binding limitation.
This standalone API is distinct from `CeraEngine.transcribe` (LFM2-Audio).

## Crate types

```toml
[lib]
crate-type = ["lib", "cdylib", "staticlib"]
```

- **`cdylib`**: Android / Linux / macOS dynamic loading (`System.loadLibrary` in Kotlin).
- **`staticlib`**: iOS XCFramework archive (Swift Package Manager).
- **`lib`**: other Rust crates in this workspace (future parity harness, examples).

## Build

```bash
cargo build -p cera-ffi
# Produces (on macOS):
# - target/debug/libcera_ffi.dylib   (cdylib)
# - target/debug/libcera_ffi.a       (staticlib)
# - target/debug/libcera_ffi.rlib    (lib)
```

## Bindings

Kotlin + Swift bindings are generated by the `uniffi-bindgen` binary
target in this crate (`src/bin/uniffi-bindgen.rs` → `uniffi::uniffi_bindgen_main()`)
and committed under `cera-ffi/bindings/`:

- `bindings/kotlin/uniffi/cera_ffi/cera_ffi.kt`: ktlint-formatted.
- `bindings/swift/cera_ffi.swift`, `CeraFFI.h`, `CeraFFI.modulemap`.

### Regenerating

```bash
# Regenerate in-place after changing any #[uniffi::*] export.
just bindings

# Then commit the diff:
git add cera-ffi/bindings && git commit
```

Manual invocation (without `just`): note `--features bindgen` to
turn on `uniffi/cli`; the binary target's `required-features`
enforces it:

```bash
cargo build -p cera-ffi
cargo run -p cera-ffi --bin uniffi-bindgen --features bindgen -- \
    generate --library target/debug/libcera_ffi.dylib \
    --language kotlin --out-dir cera-ffi/bindings/kotlin
cargo run -p cera-ffi --bin uniffi-bindgen --features bindgen -- \
    generate --library target/debug/libcera_ffi.dylib \
    --language swift --out-dir cera-ffi/bindings/swift
```

Mobile library consumers (Android `cdylib`, iOS `staticlib`) build
without the `bindgen` feature so `uniffi/cli` (clap + friends) stays
out of their binaries.

`just bindings` requires `ktlint` on `PATH` (macOS: `brew install
ktlint`; Linux: `curl` the standalone binary from the ktlint releases
page). Without ktlint the generator will warn and emit unformatted
Kotlin that diverges from the committed output; the CI drift job will
fail the PR in that case.

### CI drift check

The `ffi-bindings-drift` CI job regenerates the bindings on every PR
and fails if the resulting files differ from what's committed. If
you see that job fail, run `just bindings` locally and commit the
diff; it means a Rust-side `#[uniffi::*]` export changed without
the vendored bindings being regenerated.

### Why vendor?

- **Reviewability.** Reviewers see the foreign-language API diff
  alongside the Rust-side change. Catches accidental breakage
  (renamed method, changed return type, added required trait method)
  at PR time rather than at consumer-side build time.
- **Consumption simplicity.** Foreign-language consumers can pull the
  generated file from a tag without running Rust tooling.
- **Determinism.** The committed output is the source of truth; CI
  verifies it.

## Android

`cera-ffi` cross-compiles to every Android ABI via the Android NDK.
The `android-abis` CI job builds each target in parallel on every PR
and uploads the release `.so` as a per-ABI artifact, so consumer apps
can grab them without running the NDK toolchain themselves. Local
workflow mirrors the CI setup.

### Local setup

```bash
# One-time (pin cargo-ndk to the v4.x series; CI uses the same
# major; earlier cargo-ndk had a different flag shape and would
# fail against the `just android-*` recipes + CI job below):
cargo install cargo-ndk --version '^4' --locked
rustup target add \
    aarch64-linux-android armv7-linux-androideabi \
    x86_64-linux-android i686-linux-android

# Then (ANDROID_NDK_HOME must point at the NDK root,
# typically `~/Library/Android/sdk/ndk/<version>/` if you installed
# via Android Studio, or whatever sdkmanager placed it):
export ANDROID_NDK_HOME=...
just android-all            # all four ABIs, release
just android-arm64          # just arm64-v8a, release (fast iteration)
```

Outputs land at `target/<triple>/release/libcera_ffi.so` per ABI
(~2.5 MB release, ~75 MB debug with embedded debuginfo).

### JNI layout for consumer apps

Drop the release `.so` into your Android module's source set:

```
src/main/jniLibs/
├── arm64-v8a/libcera_ffi.so
├── armeabi-v7a/libcera_ffi.so
├── x86_64/libcera_ffi.so
└── x86/libcera_ffi.so
```

Gradle / AGP will bundle the correct ABI into the APK / AAB at
install time based on the target device. Pair with the vendored
Kotlin binding from `cera-ffi/bindings/kotlin/` for the generated
API surface.

### CI artifacts

The `android-abis` matrix job publishes four artifacts per run:
`cera-ffi-android-arm64-v8a`, `cera-ffi-android-armeabi-v7a`,
`cera-ffi-android-x86_64`, `cera-ffi-android-x86`. Each contains the
single `libcera_ffi.so` for that ABI. 7-day retention; copy to your
app's `jniLibs/` as needed.

### NDK version

CI pins NDK **r27c**, a stable release the workspace is validated
against. The workflow installs it through `nttld/setup-ndk@v1` by
version string (no checksum; the action fetches from Google's CDN
which serves signed artifacts). Bumping is a one-line change: update
the `ndk-version:` value in `.github/workflows/ci.yml`'s
`android-abis` job and re-run CI to confirm every ABI still builds.
The `cargo ndk` flag shape is compatible across recent NDK majors so
the pin is mostly about toolchain + sysroot stability across runs,
not a hard constraint; later NDKs that keep the `armv7-linux-androideabi`
and `i686-linux-android` sysroots should drop in cleanly.

## Apple platforms

`cera-ffi` cross-compiles to a Swift Package Manager-ready
`CeraFFI.xcframework` via Xcode's `xcodebuild`. The
`apple-xcframework` CI job builds the framework on an Apple Silicon
`macos-15` runner and uploads it as a CI artifact every PR; consumer
iOS and native Apple Silicon Mac apps can drop the artifact straight
into their SPM dependency graph.

The framework ships three single-arch slices (**Apple Silicon
only**):

- `ios-arm64`: real iPhones / iPads (`aarch64-apple-ios`).
- `ios-arm64-simulator`: iOS Simulator on Apple Silicon Macs (`aarch64-apple-ios-sim`).
- `macos-arm64`: native Apple Silicon Macs (`aarch64-apple-darwin`).

x86_64 slices are deliberately omitted: Apple stopped selling Intel
Macs in 2023 and modern consumer apps don't need to ship for them.
Dropping the fat-binary `lipo` step keeps the pipeline simple and
the framework smaller (~125 MB total instead of ~211 MB with x86_64
fat slices).

Other Apple platforms (Mac Catalyst, watchOS, tvOS, visionOS) aren't
included yet; adding them is structurally identical (more rustup
targets + more `-library` flags on `xcodebuild -create-xcframework`).

The vendored Swift bindings under `cera-ffi/bindings/swift/` provide
the C header + module map that the framework wraps.

### Local setup

```bash
# One-time:
rustup target add \
    aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin

# Then (Xcode + Command Line Tools must be installed for xcodebuild;
# macOS only):
just apple-xcframework
just ios-arm64           # smoke test: device target only, no XCFramework
```

Output lands at `target/xcframework-build/CeraFFI.xcframework`
(~125 MB on disk: 42 MB per slice × 3 slices + small headers).
Consumer apps embed exactly one slice per build configuration, so
the per-target cost added to a shipped `.ipa` / `.app` is ~42 MB.
Cargo's `release` profile in this workspace runs
`strip = "symbols"` on the staticlibs; further size trimming would
need feature-gating out tokio / rustfft / similar heavyweight deps.

`just apple-xcframework` runs `RUSTFLAGS=""` across all three cross-
compiles for shape-consistency. It's strictly required only for the
`aarch64-apple-darwin` slice; the workspace's `.cargo/config.toml`
sets `target-cpu=native` for that triple (workstation dev
convenience), and a build host's specific microarch isn't a portable
shipped-binary baseline. Applying the override to the iOS builds
too is a no-op (iOS targets have no native flags in the config) but
keeps the recipe shape uniform and forestalls an externally-set
`RUSTFLAGS` contaminating any slice.

### XCFramework structure

```
CeraFFI.xcframework/
├── Info.plist
├── ios-arm64/
│   ├── libcera_ffi.a       # iOS device staticlib (aarch64-apple-ios)
│   └── Headers/{cera_ffiFFI.h, module.modulemap}
├── ios-arm64-simulator/
│   ├── libcera_ffi.a       # iOS Simulator staticlib (aarch64-apple-ios-sim)
│   └── Headers/{cera_ffiFFI.h, module.modulemap}
└── macos-arm64/
    ├── libcera_ffi.a       # native macOS staticlib (aarch64-apple-darwin)
    └── Headers/{cera_ffiFFI.h, module.modulemap}
```

### Swift Package Manager consumption

The repo root ships a consumable **`Cera`** SwiftPM package. Add it to
any iOS / macOS app the usual way:

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/hyeons-lab/cera", from: "0.4.0"),
],
targets: [
    .target(
        name: "MyApp",
        dependencies: [.product(name: "Cera", package: "cera")]
    ),
]
```

or in Xcode via **File → Add Package Dependencies…** with the same
URL. Then `import Cera`.

**Metal GPU.** The shipped `CeraFFI.xcframework` is built **with** the
`metal` feature, so inference prefers the native Metal backend (Auto
probes Metal → CPU) on device, the Simulator, and native macOS,
falling back to the CPU (Accelerate / NEON) when Metal is unavailable.
The iOS Metal path is validated on the iOS **Simulator** (byte-identical
to CPU output); real-device validation on a physical iPhone/iPad is
recommended before relying on it in production.
Because the slices are Metal-enabled *static* libraries, the `Cera`
SwiftPM target links `Metal.framework` + `Foundation` explicitly
(`linkerSettings` in `Package.swift`); a `.binaryTarget` static lib
does not auto-link the system frameworks its symbols reference, so
without those a consumer would hit undefined-symbol link errors. The
framework carries three arm64-only slices (`ios-arm64`,
`ios-arm64-simulator`, `macos-arm64`), so the package targets
`.iOS(.v15)` / `.macOS(.v12)` (no x86_64).

How it resolves:

- The root `Package.swift` declares a remote
  `.binaryTarget(name: "CeraFFI", url: …/releases/download/v<version>/CeraFFI.xcframework.zip, checksum: …)`.
  SPM downloads the prebuilt XCFramework from the matching GitHub
  release, so consumers never compile Rust.
- A thin `Cera` Swift target holds the UniFFI-generated wrapper
  (`cera-ffi/apple/Sources/Cera/cera_ffi.swift`, a committed copy of
  the vendored `cera-ffi/bindings/swift/cera_ffi.swift`) and depends on
  `CeraFFI` so `import cera_ffiFFI` resolves against the XCFramework's
  clang module.

The `url` + `checksum` are checked in with the literal placeholders
`RELEASE_VERSION` / `RELEASE_CHECKSUM`; the `release` job in
`.github/workflows/publish.yml` rewrites them per release in a commit it
points the `v<version>` tag at, without pushing to the default branch
(its ruleset forbids it), so the default branch keeps the placeholders
and `.package(url:, from:)` resolves the tag, which carries a valid
checksum.

**Releasing / validating locally.** `just spm-xcframework-zip` builds,
zips, and prints the `swift package compute-checksum` of the
XCFramework (the manual counterpart to the workflow's `build-spm`
job). To validate the package end-to-end, build the framework and
temporarily point the binary target at the local path:

```bash
just apple-xcframework
# in Package.swift, swap the remote .binaryTarget(url:checksum:) for:
#   .binaryTarget(name: "CeraFFI",
#                 path: "target/xcframework-build/CeraFFI.xcframework")
swift build   # compiles cera_ffi.swift against the local macOS slice
```

Revert to the url/placeholder form before committing. The package's
`cera_ffi.swift` copy is written by `just bindings` alongside the
vendored one and diffed by `just bindings-check`, so it stays
byte-identical without a separate step.

For a **local, vendored** XCFramework instead of the remote release
(e.g. an app that bundles its own build), drop `CeraFFI.xcframework`
into your package and reference it by path:

```swift
.binaryTarget(name: "CeraFFI", path: "Frameworks/CeraFFI.xcframework")
```

### CI artifact

The `apple-xcframework` CI job uploads `cera-ffi-apple-xcframework`
(7-day retention). Each PR publishes a fresh XCFramework on the
action run page; Mac-side consumers can grab the zip without
installing Xcode or running the cross-compile themselves.

### Runner + Xcode pin

CI pins `runs-on: macos-15` (Apple Silicon) rather than
`macos-latest`. Every artifact the job produces is arm64-only, and
`swiftc` on an Intel host would default to x86_64 and silently build
a Swift binary that fails to link against the aarch64 staticlib. The
pin makes the host architecture deterministic; a `uname -m` check at
the start of the job turns a future runner-image tier change into a
loud early failure instead of a mislinked binary. Xcode version
floats within the image; the job logs `xcodebuild -version` so
silent runner-image bumps are visible in the build output. iOS Rust
targets are pinned to whatever nightly the workspace uses; bumping
requires a corresponding
toolchain re-validation.

## Remote model loading

`BundleRepo` + `CeraEngine::from_bundle_id` let foreign callers load
LeapBundles models by HF ID at runtime. The model's manifest and
GGUF files are downloaded into a persistent cache directory and
reused on subsequent calls; apps don't have to bundle the GGUF
in-app, which matters because a 1B-parameter model is ~500 MB–1 GB
and consumer app stores cap at ~200 MB.

### Discovering what to load

`listLeapBundles()` returns the published catalog as
`LeapBundleEntry` records (`name` plus its `quants`), so a picker can
offer `<name>, <quant>` pairs instead of making the user type a bundle
id. The two strings feed straight into `fromBundleId`.

It needs no `BundleRepo`: the catalog is one small JSON response,
deliberately uncached, so a picker opened twice in a session reflects
newly published bundles. Both fields are sorted ascending, so a menu
built from it is stable across runs even if upstream reorders its
response.

One blocking HTTP GET with a 30 s timeout and no retry. Use
`listLeapBundlesAsync()` anywhere a UI thread is involved; the
blocking twin stalls the calling thread for the whole round trip.

The `config` in both snippets below is the repo-bearing `EngineConfig`
built in the Kotlin and Swift sections that follow: listing needs no
`BundleRepo`, but loading the chosen bundle does.

```kotlin
// Off the main thread, or use listLeapBundlesAsync() from a coroutine.
val bundles = listLeapBundles()
bundles.forEach { entry ->
    println("${entry.name}: ${entry.quants.joinToString(" ")}")
}

// Feed a chosen pair straight to from_bundle_id.
val choice = bundles.first()
val engine = CeraEngine.fromBundleId(choice.name, choice.quants.first(), config)
```

```swift
// `listLeapBundles()` blocks; prefer the async twin off a UI thread.
let bundles = try await listLeapBundlesAsync()
for entry in bundles {
    print("\(entry.name): \(entry.quants.joined(separator: " "))")
}

// The async twin here too, and it matters more: listing is a few KB,
// but this downloads the model.
if let choice = bundles.first, let quant = choice.quants.first {
    let engine = try await CeraEngine.fromBundleIdAsync(
        bundleId: choice.name, quant: quant, config: config)
}
```

### Kotlin

```kotlin
import uniffi.cera_ffi.*

// Construct the repo once per app (typical lifecycle: Application
// onCreate). `filesDir` is the Android-recommended persistent path,
// NOT `cacheDir`, which the OS can purge under storage pressure.
val repo = BundleRepo(storeDir = context.filesDir.absolutePath + "/cera-bundles")

val config = EngineConfig(
    contextSize = 0UL,                           // use model default
    backend = BackendPreference.AUTO,
    bundleRepo = repo,                           // enables from_bundle_id
)

val engine = CeraEngine.fromBundleId(
    bundleId = "LFM2-1.2B-GGUF",
    quant = "Q4_0",
    config = config,
)
```

### Swift

```swift
import Foundation
// `CeraFFI` here is the XCFramework module from PR 9.

let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
let repo = BundleRepo(storeDir: appSupport.appendingPathComponent("cera-bundles").path)

let config = EngineConfig(
    contextSize: 0,
    backend: .auto,
    bundleRepo: repo,
)

let engine = try CeraEngine.fromBundleId(
    bundleId: "LFM2-1.2B-GGUF",
    quant: "Q4_0",
    config: config
)
```

### Cache layout + idempotency

Bundles land under `<store_dir>/huggingface.co/<full URL path>`,
mirroring the URL structure. Concrete example for
`from_bundle_id("LFM2-1.2B-GGUF", "Q4_0", _)`:

```
<store_dir>/huggingface.co/
└── LiquidAI/LeapBundles/resolve/main/LFM2-1.2B-GGUF/Q4_0.json
<store_dir>/huggingface.co/
└── LiquidAI/LFM2-1.2B-GGUF/resolve/main/LFM2-1.2B-Q4_0.gguf
```

Second and subsequent calls with the same `bundle_id` + `quant`
resolve entirely from the cache; no network I/O. Integrity
verification: caller-supplied SHA-256 when available (manifest per-
file hashes once they land upstream), otherwise HF's `X-Linked-Etag`
or a `Content-Length` size check.

### Blocking + async variants

`fromBundleId` is **synchronous**; the first call blocks on the
download over the network (potentially minutes for a multi-GB
model). Use it from sync code paths (CLI tools, Rust-side
consumers, simple Mac apps).

`fromBundleIdAsync` (PR 11) is the async twin: same args, returns
a future. Internally `spawn_blocking`s the sync logic onto tokio's
blocking pool so the caller's async context isn't stalled. Pair
with `generateAsync` / `generateStreamingAsync` for an end-to-end
async workflow:

```kotlin
// Kotlin coroutine
val engine = CeraEngine.fromBundleIdAsync("LFM2-1.2B-GGUF", "Q4_0", config)
val session = engine.newSession(SessionConfig())
session.appendText("hello")
val out = session.generateAsync(GenerateOpts())
```

```swift
// Swift async
let engine = try await CeraEngine.fromBundleIdAsync(
    bundleId: "LFM2-1.2B-GGUF", quant: "Q4_0", config: config
)
let session = try engine.newSession(config: SessionConfig())
try session.appendText(text: "hello")
let out = try await session.generateAsync(opts: GenerateOpts())
```

Cancellation: dropping the `fromBundleIdAsync` future drops an
internal `AbortOnDrop` guard which calls `AbortHandle::abort` on
the spawned blocking task. That cancels the task if it hasn't
started yet (queued on the blocking pool); if it has started,
abort is a no-op and the download / engine construction runs to
completion. The downloaded bundle is cached, so the caller's next
attempt resolves from the cache; bandwidth isn't wasted, just
shifted. (`generateAsync`'s `Session::cancel` in-flight
cancellation has no equivalent here because cera's download path
uses `reqwest::blocking` without a cooperative cancel point.)

### Download progress

Mobile apps that show a progress bar during model download attach a
`DownloadProgressSink` at `BundleRepo` construction time. The sink
fires periodically during cache-miss downloads (every ~256 KB
written + once at end-of-stream). Cache-hit resolves never fire.

The same sink receives events for every URL the repo downloads;
distinguish per-file UI (manifest first, then GGUF) by branching on
the `url` argument inside the callback.

**Kotlin:**

```kotlin
import uniffi.cera_ffi.*

class ProgressTracker : DownloadProgressSink {
    override fun onProgress(url: String, bytesDownloaded: ULong, totalBytes: ULong?) {
        val pct = totalBytes?.let { (bytesDownloaded * 100u) / it }
        // Marshal to UI thread; the callback fires from the
        // download thread (caller's thread for `fromBundleId`,
        // a tokio blocking worker for `fromBundleIdAsync`).
        runOnUiThread {
            progressBar.progress = pct?.toInt() ?: 0
            statusText.text = "Downloading ${url.substringAfterLast('/')}"
        }
    }
}

val repo = BundleRepo.withProgress(
    storeDir = context.filesDir.absolutePath + "/cera-bundles",
    progress = ProgressTracker(),
)
val config = EngineConfig(contextSize = 0UL, backend = BackendPreference.AUTO, bundleRepo = repo)
val engine = CeraEngine.fromBundleIdAsync("LFM2-1.2B-GGUF", "Q4_0", config)
```

**Swift:**

```swift
import Foundation

final class ProgressTracker: DownloadProgressSink {
    func onProgress(url: String, bytesDownloaded: UInt64, totalBytes: UInt64?) {
        let pct = totalBytes.map { Double(bytesDownloaded) / Double($0) * 100 }
        DispatchQueue.main.async {
            // update UI
            print("[\(url.split(separator: "/").last ?? "?")] \(pct ?? 0)%")
        }
    }
}

let repo = BundleRepo.withProgress(
    storeDir: appSupport.appendingPathComponent("cera-bundles").path,
    progress: ProgressTracker()
)
let config = EngineConfig(contextSize: 0, backend: .auto, bundleRepo: repo)
let engine = try await CeraEngine.fromBundleIdAsync(
    bundleId: "LFM2-1.2B-GGUF", quant: "Q4_0", config: config
)
```

**Throttling.** cera-core caps the callback rate to one per ~256 KB
written + one final at end-of-stream. At 10 MB/s download speed
that's ~25 callbacks/second, comfortable for a 30 Hz UI repaint
without overhead. Implementers don't need to dedupe.

**`total_bytes` may be `None`**: the server didn't surface a
`Content-Length` (chunked transfer, or HEAD didn't probe). UIs
should display indeterminate progress in that case rather than
divide by zero.

**Cache hits don't fire.** A `from_bundle_id` against a fully
cached bundle returns instantly without any sink invocation. UIs
should handle the "nothing to download" case (call returned in
under a second, sink wasn't called) gracefully.

### Per-platform cache-path recommendations

| Platform | Recommended `store_dir` |
|---|---|
| Android | `Context.getFilesDir()` + subdir: persistent; survives app restarts |
| iOS | `FileManager.applicationSupportDirectory` + subdir: persistent, not iCloud-synced |
| macOS (native) | `~/Library/Application Support/<app bundle id>/cera-bundles` |
| Linux (CLI/server) | `$XDG_CACHE_HOME/cera-bundles` or `~/.cache/cera-bundles` |

Do not use `Context.getCacheDir()` on Android or `tmp` on any
platform; the OS can purge those under storage pressure, forcing
a full re-download every time pressure gets reset.

### Cache management

Two operational methods on `BundleRepo` for mobile apps that need
to surface "Storage: X MB used" in settings or wipe downloaded
models on user request:

| Method | Returns | Behavior |
|---|---|---|
| `repo.cacheSize()` | `u64` (bytes) | Recursive walk of `store_dir`. `0` if the dir doesn't exist yet (no downloads). |
| `repo.clearCache()` | `void` (throws on I/O error) | Removes everything under `store_dir`, recreates `store_dir` empty. Idempotent. |

**Threading.** Both walk / mutate the filesystem; for a multi-GB
cache `cacheSize()` is a real walk (not a constant-time query;
the OS doesn't track per-directory totals). Run off the main UI
thread:

```kotlin
// Kotlin coroutine
val sizeMb = withContext(Dispatchers.IO) { repo.cacheSize() } / 1_048_576
```

```swift
// Swift async
let sizeMb = try await Task.detached { try repo.cacheSize() }.value / 1_048_576
```

**Concurrency note.** `clearCache()` removes files that an
in-flight `fromBundleId*` call might be writing to (manifest +
GGUF). The caller is responsible for serializing a clear against
any active downloads; typically trivial since the action is
user-driven (settings tap), and apps usually don't run a download
in the background while showing a "clear cache" button.

### Known breaking change: Swift `EngineConfig` equality

Adding `bundleRepo: BundleRepo?` to `EngineConfig` drops the
auto-synthesized `Equatable` + `Hashable` conformance from the
generated Swift struct; `BundleRepo` is a UniFFI Object (reference
type) which has no structural equality, so Swift can't derive
`Equatable` on a struct containing one. Swift callers that were
comparing `EngineConfig` values with `==` or using them as
`Dictionary` keys / `Set` elements will need to compare the scalar
fields (`contextSize`, `backend`) directly, or provide a local
extension:

```swift
extension EngineConfig: Equatable {
    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.contextSize == rhs.contextSize
            && lhs.backend == rhs.backend
            && (lhs.bundleRepo === rhs.bundleRepo)  // identity, not structural
    }
}
```

This would go in the consumer app, not in the vendored binding;
`ffi-bindings-drift` overwrites the generated file on every surface
change, and UniFFI can't currently be told to re-derive the
conformance across reference-type fields.

Kotlin isn't affected (`data class` equality compiles against any
field type, using reference-equality for object fields automatically).

## Tokenizer + chat templates

`CeraEngine` exposes the model's BPE tokenizer + chat-template
renderer so foreign callers can tokenize / detokenize / format
messages without going through `Session::append_text`. Useful for:

- Pre-counting prompt tokens before deciding whether to start a
  session (context budgeting).
- Manual prompt construction with explicit special tokens.
- Decoding token IDs streamed back from `generate` when driving an
  incremental UI (`generateStreaming` already returns text chunks
  via `ModalitySink::on_text_chunk`, but consumers building a
  custom token-level UI can encode/decode IDs directly).
- Rendering a chat template against a list of `ChatMessage`s (for low-level
  inspection or manual prompt construction; for multi-turn conversations prefer
  `ChatSession`).

### Surface

| Method | Purpose |
|---|---|
| `engine.encodeText(text)` → `Vec<u32>` | Tokenize a string. |
| `engine.decodeTokens(tokens)` → `String` | Detokenize. |
| `engine.vocabSize()` → `u32` | Total vocab size. |
| `engine.bosToken()` / `engine.eosToken()` → `u32?` | Common special tokens (typo-safe getters for the two everyone needs). |
| `engine.specialTokenId(name)` → `u32?` | Lookup by literal vocab name (e.g. <code>&lt;|im_start|&gt;</code>). Only tokens with `tokenizer.ggml.token_type` 3 (control) or 4 (user-defined) are reachable. |
| `engine.isSpecialToken(id)` → `Bool` | Inverse of `specialTokenId`; useful for filtering control tokens out of streamed output before rendering. |
| `engine.hasChatTemplate()` → `Bool` | Check before render. |
| `engine.applyChatTemplate(messages, addGenerationPrompt)` → `String` | Render template. |
| `engine.applyChatTemplateWithTools(messages, tools, addGenerationPrompt)` → `String` | Render the template with a `tools` array so a tool-trained model emits its tool-definition block. Empty `tools` == `applyChatTemplate`. |
| `engine.toolFormat()` → `ToolFormat?` | Tool-call format auto-detected from the model's architecture (`lfm2Pythonic` / `hermes`), or `null` if unknown. |
| `engine.toolCallStartToken(format)` → `u32?` | Vocab id of `format`'s start marker (e.g. <code>&lt;|tool_call_start|&gt;</code>), for `GenerateOpts.grammarTriggerTokens` (lazy constrained tool calls). `null` if absent. |
| `detectToolFormat(architecture)` → `ToolFormat?` | Free function: format for a GGUF architecture string. |
| `toolGrammar(tools, format)` → `String` | Free function: GBNF that constrains output to a valid call for `tools`. Pair with `grammarTriggerTokens` for lazy constraint. |
| `parseToolCalls(text, format)` → `[ToolCall]` | Free function: parse tool calls out of generated text. `ToolCall` = `{ name, argumentsJson }`. |
| `engine.contextSize()` → `u64` | **Requested** engine `context_size` after applying the `0` → model `max_seq_len` default (so callers never see the internal `usize::MAX` sentinel). This is the engine-level config readback, **not** the per-session ceiling; cera clamps the model's `max_seq_len` to `min(requested, gguf_max)` at load time, so `engine.metadata().maxSeqLen` is the effective cap. |

### Kotlin example

```kotlin
val engine = CeraEngine.fromPath("...", config)

// Pre-count prompt tokens.
val prompt = "Hello, world!"
val tokens = engine.encodeText(prompt)
println("prompt is ${tokens.size} tokens")

// Render a chat template.
if (engine.hasChatTemplate()) {
    val rendered = engine.applyChatTemplate(
        messages = listOf(
            ChatMessage(role = "system", content = "You are helpful."),
            ChatMessage(role = "user", content = "Hi!"),
        ),
        addGenerationPrompt = true,
    )
    engine.newSession(SessionConfig()).use { session ->
        session.appendText(rendered)
        val out = session.generate(GenerateOpts())
        val replyText = engine.decodeTokens(out.tokens)
        println("Assistant: $replyText")
    }
}

// Tool calling: render tools into the prompt, then parse calls from the reply.
val tools = listOf(
    ToolDef(
        name = "get_weather",
        description = "Get the current weather for a city",
        parametersJson = """{"type":"object",
            "properties":{"city":{"type":"string"}},"required":["city"]}""",
    )
)
val format = engine.toolFormat() ?: ToolFormat.LFM2_PYTHONIC
val toolsPrompt = engine.applyChatTemplateWithTools(
    messages = listOf(ChatMessage(role = "user", content = "Weather in Paris?")),
    tools = tools,
    addGenerationPrompt = true,
)
engine.newSession(SessionConfig()).use { session ->
    session.appendText(toolsPrompt)
    // Optional: constrain to a valid call via grammar + lazy trigger.
    val opts = GenerateOpts()
    engine.toolCallStartToken(format)?.let { trigger ->
        opts.grammar = toolGrammar(tools, format)      // GBNF string
        opts.grammarTriggerTokens = listOf(trigger)
    }
    val reply = engine.decodeTokens(session.generate(opts).tokens)
    for (call in parseToolCalls(reply, format)) {
        println("${call.name}(${call.argumentsJson})")
    }
}
```

These are two separate conversations. Each `use` block closes its session,
releasing GPU ownership before another session is created. Keep the same session
open when continuing an existing conversation.

### Swift example

```swift
let engine = try CeraEngine.fromPath(path: "...", config: config)

let tokens = engine.encodeText(text: "Hello, world!")
print("prompt is \(tokens.count) tokens")

if engine.hasChatTemplate() {
    let rendered = try engine.applyChatTemplate(
        messages: [
            ChatMessage(role: "system", content: "You are helpful."),
            ChatMessage(role: "user", content: "Hi!"),
        ],
        addGenerationPrompt: true
    )
    let session = try engine.newSession(config: SessionConfig())
    try session.appendText(text: rendered)
    let out = try session.generate(opts: GenerateOpts())
    let replyText = engine.decodeTokens(tokens: out.tokens)
    print("Assistant: \(replyText)")
}

// Tool calling: render tools into the prompt, then parse calls from the reply.
let tools = [
    ToolDef(
        name: "get_weather",
        description: "Get the current weather for a city",
        parametersJson: #"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#
    )
]
let format = engine.toolFormat() ?? .lfm2Pythonic
let toolsPrompt = try engine.applyChatTemplateWithTools(
    messages: [ChatMessage(role: "user", content: "Weather in Paris?")],
    tools: tools,
    addGenerationPrompt: true
)
let toolSession = try engine.newSession(config: SessionConfig())
try toolSession.appendText(text: toolsPrompt)
// Optional: constrain to a valid call via grammar + lazy trigger.
var opts = GenerateOpts()
if let trigger = engine.toolCallStartToken(format: format) {
    opts.grammar = try toolGrammar(tools: tools, format: format)   // GBNF string
    opts.grammarTriggerTokens = [trigger]
}
let reply = engine.decodeTokens(tokens: try toolSession.generate(opts: opts).tokens)
for call in try parseToolCalls(text: reply, format: format) {
    print("\(call.name)(\(call.argumentsJson))")
}
```

### Notes

- Tokenizer methods are read-only; safe to call concurrently with
  `generate*` on a `Session` opened from the same engine.
- Empty input to `encodeText` returns an empty vec.
- Out-of-vocab token IDs in `decodeTokens` are silently skipped
  (omitted from the decoded output); `BpeTokenizer::decode` only
  appends bytes for IDs it has in its vocab. No substitution glyph,
  no error. Validate against `vocabSize()` first if you need to
  detect invalid IDs.
- `applyChatTemplate` returns `FfiError::Backend` if the model has
  no chat template (check `hasChatTemplate()` first) or if the
  template's Jinja2 render fails against the supplied messages.
- The `ChatMessage` `role` set depends on the model's template;
  typically `"system"`, `"user"`, `"assistant"`, occasionally
  `"tool"` for function-calling. cera-ffi doesn't validate the role
  string; whatever you pass flows directly into the Jinja template.
  Whether an unknown role errors or silently no-ops is up to the
  template's own logic; many templates have an explicit error
  path for unrecognized roles, but it's template-dependent rather
  than enforced by `applyChatTemplate`.

## Conversational Chat Coordinator (ChatSession)

`ChatSession` is the high-level coordinator for multi-turn conversational chat.
It manages Jinja2 template formatting, tracks conversation phases, enforces role alternation,
evaluates only new tokens on continuation turns (delta-only prefill), retains KV cache state
across turns without full history replay, and supports wait-free cancellation.

Obtain a `ChatSession` by calling `session.intoChat()` on an existing `Session`, or instantiate
one directly with `engine.newChatSession(config)`.

### Surface

| Method | Signature | Notes |
|---|---|---|
| `engine.newChatSession(config)` | `(SessionConfig) -> Result<Arc<ChatSession>, FfiError>` | Instantiate a chat coordinator directly from an engine. |
| `session.intoChat()` | `() -> Result<Arc<ChatSession>, FfiError>` | Transition a raw session into a chat coordinator. Moves ownership out of Session. |
| `chat.ingest(message)` | `(Message) -> Result<IngestSummary, FfiError>` | Ingest a single message (typically User) into conversation state. |
| `chat.ingestMessages(messages)` | `(Vec<Message>) -> Result<IngestSummary, FfiError>` | Ingest a sequence of messages (for example, System prompt followed by initial User turn). |
| `chat.replaceMessages(messages)` | `(Vec<Message>) -> Result<IngestSummary, FfiError>` | Replace conversation history and restart framing without full engine re-allocation. |
| `chat.complete(opts)` | `(GenerateOpts) -> Result<TurnResult, FfiError>` | Complete the current turn synchronously (delta prefill + decode). |
| `chat.generateStreaming(opts, sink)` | `(GenerateOpts, Arc<dyn ModalitySink>) -> Result<TurnResult, FfiError>` | Stream turn generation to a `ModalitySink` callback. |
| `chat.phase()` | `() -> SessionPhase` | Non-blocking query of the current coordinator phase (`idle`, `promptReady`, `turnComplete`, `turnRefused`, `cancelled`, `rawContext`). |
| `chat.position()` | `() -> u32` | Lock-free query of current KV tokens. |
| `chat.cancel()` | `() -> ()` | Wait-free cancellation atomic flip. Safe to call from any thread or callback. |
| `chat.clearCancel()` | `() -> ()` | Clear cancellation flag while preserving KV cache and conversation position. |
| `chat.reset()` | `() -> Result<(), FfiError>` | Reset conversation history and clear KV cache. |
| `chat.recoveryStatus()` | `() -> Result<RecoveryOutcome, FfiError>` | Non-blocking diagnostic query returning outcome of failed ingestion or reset. |
| `chat.intoSession()` | `() -> Result<Arc<Session>, FfiError>` | Non-destructively reclaim the underlying raw Session. |

### Message constructors

Foreign bindings provide convenience functions to construct `Message` records:
- `chatMessageUser(content: String)`
- `chatMessageSystem(content: String)`
- `chatMessageAssistant(content: String)`
- `chatMessageTool(callId: String, content: String)`

### Swift example

```swift
import Cera

let engine = try CeraEngine.fromPath(path: "model.gguf", config: config)
let session = try engine.newSession(config: SessionConfig())
let chat = try session.intoChat()

// Ingest system prompt and first user message
try chat.ingestMessages(messages: [
    chatMessageSystem(content: "You are a concise, helpful assistant."),
    chatMessageUser(content: "What is the capital of France?"),
])

// Generate assistant reply
var opts = GenerateOpts()
opts.maxTokens = 64
let turn1 = try chat.complete(opts: opts)
print("Assistant: \(turn1.text)")

// Continuation turn: only the new message is prefilled into KV
try chat.ingest(message: chatMessageUser(content: "What is its population?"))
let turn2 = try chat.complete(opts: opts)
print("Assistant: \(turn2.text)")

// Reclaim raw session if needed
let reclaimedSession = try chat.intoSession()
```

### Kotlin example

```kotlin
import uniffi.cera_ffi.*

val engine = CeraEngine.fromPath("model.gguf", config)
engine.newChatSession(SessionConfig()).use { chat ->
    // Ingest system prompt and first turn
    chat.ingestMessages(listOf(
        chatMessageSystem("You are a concise, helpful assistant."),
        chatMessageUser("What is the capital of France?"),
    ))

    val opts = GenerateOpts(maxTokens = 64u)
    val turn1 = chat.complete(opts)
    println("Assistant: ${turn1.text}")

    // Continuation turn (delta-only prefill, live KV retention)
    chat.ingest(chatMessageUser("What is its population?"))
    val turn2 = chat.complete(opts)
    println("Assistant: ${turn2.text}")
}
```

See runnable multi-language examples in [`examples/Chat.swift`](examples/Chat.swift),
[`examples/Chat.kt`](examples/Chat.kt), and [`examples/chat.py`](examples/chat.py).

## Session API

`engine.newSession(config)` produces an `Arc<Session>` that retains the
engine's model and tokenizer, plus its own sampler, cancel atomic, and
`Mutex` over the inner `cera::Session`. CPU sessions own their live KV state
and can run concurrently against the same engine. Metal/wgpu models own one
live GPU context: a second session on the same loaded model returns
`FfiError::Busy` until the first session is released. Resetting or cancelling
keeps that reservation. Load separate models for simultaneous GPU conversations;
see [Sharing a loaded GPU model](#sharing-a-loaded-gpu-model) for foreign lifetimes.

### Surface

| Method | Signature | Notes |
|---|---|---|
| `engine.newSession(config)` | `(SessionConfig) -> Result<Arc<Session>, FfiError>` | Per-session knobs (`seed`, `nKeep`, `ubatchSize`, `maxSeqLen`, `kvCompression`). Returns `Busy` if another session owns the model's GPU context, or `OutOfMemory` when the KV cache can't be allocated. |
| `session.intoChat()` | `() -> Result<Arc<ChatSession>, FfiError>` | Transition the raw session into a transactional chat coordinator. Moves ownership out of Session. |
| `session.appendText(text)` | `(String) -> Result<(), FfiError>` | Tokenize + push into KV. Convenience over `appendTokens(encodeText(text))`. |
| `session.appendTokens(tokens)` | `(Vec<u32>) -> Result<(), FfiError>` | Push pre-tokenized IDs. Use when you need explicit BOS/EOS framing. |
| `session.sendMessage(message)` | `(UserMessage) -> Result<(), FfiError>` | **Deprecated**: use `ChatSession` via `intoChat()` instead. Append a multimodal envelope (`UserMessage` with optional `text`, `images`, `audio`) enforcing model-canonical ordering and automatic 16 kHz resampling. |
| `session.sendMessageAndGenerate(message, opts)` | `(UserMessage, GenerateOpts) -> Result<GenerateOutput, FfiError>` | **Deprecated**: use `ChatSession` via `intoChat()` instead. |
| `session.sendMessageStreaming(message, opts, sink)` | `(UserMessage, GenerateOpts, Arc<dyn ModalitySink>) -> Result<GenerateSummary, FfiError>` | **Deprecated**: use `ChatSession` via `intoChat()` instead. |
| `session.generate(opts)` | `(GenerateOpts) -> Result<GenerateOutput, FfiError>` | Sync decode; returns the full text + token list + summary in one shot. |
| `session.generateStreaming(opts, sink)` | `(GenerateOpts, Arc<dyn ModalitySink>) -> Result<GenerateSummary, FfiError>` | Sync decode with a foreign-trait callback per flush boundary (text chunks or audio frames per the model's modality). Returns the summary only; text chunks flow through the sink. |
| `session.generateAsync(opts)` | `async (GenerateOpts) -> Result<GenerateOutput, FfiError>` | `spawn_blocking`-backed async twin of `generate`. Cancel by dropping the future. |
| `session.generateStreamingAsync(opts, sink)` | `async (GenerateOpts, Arc<dyn ModalitySink>) -> Result<GenerateSummary, FfiError>` | Async + streaming. Cancel by dropping the future (also fires `Session::cancel` via the internal `AbortOnDrop` guard). |
| `session.position()` | `() -> u32` | Tokens currently in the KV cache. Atomic-backed (no mutex), safe to poll from any thread. |
| `session.cancel()` | `() -> ()` | Flip the cancel atomic. Safe from any thread. Decode loop checks it at every flush boundary. |
| `session.clearCancel()` | `() -> ()` | Clear the cancel flag without dropping any session state. |
| `session.reset()` | `() -> Result<(), FfiError>` | Reset KV + position + last logits + re-seed sampler from `SessionConfig.seed`. Retains GPU context ownership. |
| `session.capabilities()` | `() -> ModalityCapabilities` | The same flags `engine.capabilities()` reports; exposed on `Session` too so a caller holding only the session handle can probe. |

### Lifecycle (Kotlin)

```kotlin
val engine = CeraEngine.fromPath("...", config)

// Per-session knobs. SessionConfig() picks the cera
// defaults (random sampler seed, no nKeep pin, ubatchSize 512).
val session = engine.newSession(SessionConfig())

session.appendText("Hello, what is the capital of France?")
val out = session.generate(GenerateOpts())
println("decoded ${out.tokens.size} tokens, finish=${out.summary.finishReason}")
println(out.text)
```

### Streaming (Swift)

```swift
class StreamSink: ModalitySink {
    func onThoughtChunk(text: String) {
        // Handle streaming internal reasoning / thinking tokens
        DispatchQueue.main.async { self.updateThinkingUI(text) }
    }
    func onTextChunk(text: String) {
        // Marshal off-thread; the callback fires on the decode thread.
        DispatchQueue.main.async { self.updateUI(text) }
    }
    func onAudioFrames(pcm: [Float], sampleRate: UInt32) { /* LFM2-Audio */ }
    func onDone(reason: FinishReason) { print("done: \(reason)") }
    func updateUI(_ text: String) { /* render */ }
    func updateThinkingUI(_ text: String) { /* render thinking */ }
}

let sink = StreamSink()
let summary = try session.generateStreaming(opts: opts, sink: sink)
```

### Resuming after cancel vs starting over

After a cancellation lands (`FfiError::Cancelled` from
`appendText` / `appendTokens` / `appendAudio` mid-prefill, or
`FinishReason::Cancelled` on the `GenerateOutput` from `generate`),
two primitives let callers continue with the same `Session`
without paying `engine.newSession(...)` setup cost again:

| API | KV cache | `position` | Sampler | When to use |
|---|---|---|---|---|
| `session.clearCancel()` | preserved | preserved | preserved | "interrupted but continuing": keep the conversation context, append more tokens, generate again |
| `session.reset()` | dropped | reset to 0 | re-seeded from `cfg.seed` | "clear conversation" UI button: start fresh on the same model + tokenizer |

```kotlin
val tokensBefore = session.position().toInt()       // snapshot before the call
try {
    session.appendTokens(longPrompt)                // may throw FfiException.Cancelled mid-prefill
} catch (e: FfiException.Cancelled) {
    val consumed = session.position().toInt() - tokensBefore
    session.clearCancel()                           // resume without losing KV
    session.appendTokens(longPrompt.drop(consumed))
}
```

```swift
let tokensBefore = Int(session.position())          // snapshot before the call
do {
    try session.appendTokens(tokens: longPrompt)
} catch FfiError.Cancelled {
    let consumed = Int(session.position()) - tokensBefore
    session.clearCancel()
    try session.appendTokens(tokens: Array(longPrompt[consumed...]))
}
```

### Notes

- **Threading.** `appendText` / `appendTokens` / `appendAudio` /
  `generate*` / `reset` mutate session state and acquire an
  internal `Mutex`. Concurrent calls on the same `Session` block;
  cross-session calls don't. `cancel` / `clearCancel` / `position`
  / `capabilities` are atomic-only and safe to call concurrently
  with anything (including from inside a `ModalitySink` callback
  on a different thread).
- **Cancel + drop semantics.** `generate*` calls held by an
  async task that gets dropped also fire the cancel atomic via
  the internal `AbortOnDrop` guard; no need to manually
  `session.cancel()` before letting a Swift `Task` go out of scope.
- **Streaming sink errors aren't recoverable mid-decode.** A
  `ModalitySink` implementation that throws an exception will
  unwind the decode loop. The `GenerateOutput` returned will
  carry whatever tokens decoded before the throw, but the session
  is left in a partial state; call `reset()` (or `clearCancel()`
  if you want to keep the partial KV) before the next `generate`.
- **`appendImage(bytes, maxLongSize)`** appends an encoded image
  (PNG / JPEG) to the context for VL bundles, mirroring
  `appendAudio`. `CeraEngine.newSession` auto-attaches the vision
  mmproj encoder, so no separate load call is needed. The optional
  per-call `maxLongSize` caps the longest side of the *encoded* image
  (aspect-preserving): smaller = fewer image tokens, faster, less
  detail. It only shrinks (never upscales) and takes precedence over
  the model's minimum-resolution floor; `null` applies no cap for that
  call. Returns `UnsupportedModality` on a non-VL model and `Backend`
  on a decode / encoder mismatch. The ViT encode runs on the GPU
  (native Metal or wgpu, per the engine's backend) with a CPU fallback.
- **`setImageMaxLongSize(maxLongSize)`** sets a session-default cap
  honored by *every* image-append path (including chat-template flows),
  so a host can configure the image-encode budget once instead of per
  call. `appendImage`'s explicit argument overrides it for that call.
- **`engine.transcribe(pcm, sampleRate)`** → `Result<String, FfiError>` is a
  one-shot ASR convenience on `CeraEngine` (not `Session`): it runs a full prefill +
  greedy decode over mono `f32` PCM using the model's trained
  `"Perform ASR."` chat mode and returns the transcript. Requires an
  audio-capable bundle (`UnsupportedModality` otherwise) and `sampleRate`
  must match the encoder's expected rate. Blocking; wrap it in
  `spawn_blocking` / `Task.detached` from an async context.

## Keyword Spotting & Whisper ASR

### Keyword Spotting (Wake Word Detection)

`cera-ffi` exposes the native streaming wake word engine (`FfiHotwordDetector`, `FfiHotwordIterator`, `FfiHotwordConfig`, `FfiHotwordEvent`):

- **`FfiHotwordDetector.fromFile(path)`** / **`fromBytes(bytes)`**: Loads a self-describing GGUF keyword spotting model (`kws.keywords`, window/hop dimensions, thresholds) with reusable inference scratch buffers. Foreign argument/result conversion and event creation can still allocate.
- **`FfiHotwordIterator.fromFiles(detectorPath, vadPath, config)`**: Creates a streaming state machine with integrated circular ring buffering, 30.0x AGC peak normalization, Silero VAD gating, and post-detection lockout debounce.
- **`iterator.processChunk(chunk)`**: Ingests mono PCM float samples at the model's sample rate (commonly 16 kHz). It processes the whole chunk and returns its first triggered `FfiHotwordEvent` (keyword, confidence, timestamp, audio sample offset), or no event. Keep one iterator per stream and call it serially on a background audio worker; it does not resample input.
- **`iterator.reset()`**: Clears ring buffers and debounces after command execution.

Event offsets identify the detection window's exclusive end. `commandStartSample`
subtracts the configured pre-roll from that offset, saturating at zero; these
values provide a stream reference rather than acoustic word alignment.

### Whisper Speech Recognition (ASR)

`cera-ffi` exposes pure-Rust OpenAI Whisper transcription (`FfiWhisperModel`, `FfiWhisperTranscribeOpts`):

- **`FfiWhisperModel.fromFile(path)`** / **`fromBytes(bytes)`**: Instantiates the Whisper model from standard GGUF weights.
- **`model.transcribe(pcm, opts)`**: Synchronous transcription returning recognized text.
- **`model.transcribeAsync(pcm, opts)`**: Asynchronous transcription on a Tokio blocking worker. Dropping its Rust future signals cooperative cancellation; Kotlin coroutine cancellation propagates this, while the pinned Swift wrapper does not propagate `Task.cancel()`. See the [Whisper examples](../docs/internals/API_RESHAPE_WHISPER_EXAMPLES.md) for input and cancellation limits.

## Design notes

- **Proc-macro path** chosen over UDL for smaller surface ergonomics.
  Annotations live next to the Rust types they describe; no separate
  grammar to maintain. Can migrate to UDL if the annotation density
  ever becomes unmanageable.
- **Async runtime** is `tokio` (via UniFFI's `tokio` feature flag +
  `#[uniffi::export(async_runtime = "tokio")]`). `tokio` is a `cera-ffi`
  dep only, never `cera` itself; keeps the core crate runtime-agnostic.
  Sync decode work runs on `tokio::task::spawn_blocking` so the async
  worker pool stays free to poll other futures while a generate is in
  flight.
- **Send + Sync** is already guaranteed on every `cera` type we plan
  to expose (landed in PR #42). UniFFI requires it for every
  `#[uniffi::Object]`.
- **UniFFI 0.31.x**: upgraded from 0.28 in PR 6. All proc-macro
  annotations are stable across the version jump; only the bindgen
  CLI shape changed, which is isolated to the `uniffi-bindgen` binary
  target and the `cli` feature on the `uniffi` dep.
