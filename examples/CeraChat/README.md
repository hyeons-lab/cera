# CeraChat: cera iOS example

A small SwiftUI app that runs on-device LLM inference with
[cera](https://github.com/hyeons-lab/cera) via its published Swift Package. It
doubles as a real-device **Metal validation** harness: on device the engine's
`Auto` backend prefers the native Metal GPU path (falling back to CPU).

Three screens:

1. **Load**: download `LFM2.5-1.2B-Instruct / Q4_0` from the LeapBundles
   registry (progress bar), or import a local `.gguf`. Shows model metadata and
   the requested backend.
2. **Chat**: streaming, multi-turn chat. Each send renders the conversation
   through the model's chat template, prefills it, and streams the reply
   token-by-token into the assistant bubble.
3. **Embed / LoRA**: mean-pooled hidden-state embeddings (dimension, first 8
   values, L2 norm), plus optional LoRA adapter attach/detach.

## Add the package to your own app

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/hyeons-lab/cera", from: "0.7.0"),
],
targets: [
    .target(name: "YourApp", dependencies: [
        .product(name: "Cera", package: "cera"),
    ]),
]
```

In Xcode: **File → Add Package Dependencies…**, paste the URL, and add the
`Cera` product to your app target. The package pulls a prebuilt,
Metal-enabled `CeraFFI.xcframework` (arm64 device + arm64 simulator + arm64
macOS), so you never compile Rust.

> The XCFramework slices are Metal-enabled **dynamic** frameworks that embed their own load commands for `Metal.framework` and `Foundation`, which dyld resolves automatically.

## Minimal load + generate

```swift
import Cera

// 1. Load a model (downloads + caches on first run).
let repo = BundleRepo.withProgress(storeDir: cacheDir.path, progress: mySink)
let config = EngineConfig(contextSize: 4096, backend: .auto, bundleRepo: repo)
let engine = try await CeraEngine.fromBundleIdAsync(
    bundleId: "LFM2.5-1.2B-Instruct-GGUF", quant: "Q4_0", config: config)

// 2. Open a session and enter conversational chat.
let session = try engine.newSession(config: SessionConfig(
    maxSeqLen: nil, kvCompression: .none, nKeep: 0, seed: nil, ubatchSize: 0))
let chat = try session.intoChat()

// 3. Ingest message and stream reply using AsyncThrowingStream.
try chat.ingest(message: Message(role: .user, content: "Hello!"))
let opts = GenerateOpts(maxTokens: 256, temperature: 0.7)

for try await chunk in chat.stream(opts: opts) {
    print(chunk, terminator: "")
}
```

`chat.stream(opts:)` returns an `AsyncThrowingStream<String, Error>` yielding decoded
text chunks asynchronously, handling multi-byte UTF-8 and BPE merge reassembly automatically.
Consume it directly in an `async` task on `@MainActor` to update SwiftUI state.

## Backend note

The app requests `backend: .auto`, which probes **Metal → CPU** at load time.
There is no FFI to read back which backend actually won, so the UI only ever
advertises what was *requested* ("Auto (Metal → CPU)"); it does **not** claim
to have detected Metal.

## Build & run

Requires XcodeGen (`brew install xcodegen`). The `.xcodeproj` is generated from
`project.yml` and git-ignored, so regenerate it first:

```bash
cd examples/CeraChat
xcodegen generate
open CeraChat.xcodeproj      # then pick a simulator / device and Run
```

Or from the command line:

```bash
xcodebuild -project CeraChat.xcodeproj -scheme CeraChat \
    -destination 'platform=iOS Simulator,name=iPhone 16 Pro' build
```

> **If `xcodebuild` reports "iOS <sdk> is not installed" / empty destinations:**
> this happens when the Xcode SDK is newer than any installed Simulator
> *runtime* (the scheme's run-destination resolution then finds none). Either
> install the matching runtime (Xcode → Settings → Components), or build the
> target directly against the Simulator SDK, which sidesteps destination
> resolution:
>
> ```bash
> xcodebuild -project CeraChat.xcodeproj -target CeraChat \
>     -sdk iphonesimulator ARCHS=arm64 ONLY_ACTIVE_ARCH=YES build
> ```

On device, tap **Download** on the Load tab (needs network the first run). On
the Simulator you can either download, import a local `.gguf`, or side-load one
by setting `CERA_MODEL_PATH` in the scheme's Run environment to a local file;
the app auto-loads it on launch.

### Headless smoke check (CI / simulator)

Setting `CERA_SMOKE_PROMPT` (alongside `CERA_MODEL_PATH`) makes the app run one
streamed generation after load and print the reply to the console, useful for
validating the full inference path without driving the UI:

```bash
xcrun simctl install <udid> CeraChat.app
SIMCTL_CHILD_CERA_MODEL_PATH=/path/to/model.gguf \
SIMCTL_CHILD_CERA_SMOKE_PROMPT="What is a large language model?" \
    xcrun simctl launch --console-pty <udid> com.hyeonslab.cerachat
# → [CeraChat smoke] reply: A large language model is …
```
