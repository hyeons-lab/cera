# Executable API reshape examples

The Rust loading and chat coordinator APIs are public in this checkout through `cera` imports;
native Swift/Kotlin, Python, and Dart loaders and `ChatSession` coordinators are also implemented.
Released package availability is not claimed. Existing `CeraEngine` constructors
remain available. These examples exercise both the loading/session path and the
high-level transactional `ingest`/`complete` multi-turn chat coordinator.

The [native loading examples](../../cera-ffi/README.md#explicit-model-loading)
show the production Swift/Kotlin types, raw prompt completion and resource cleanup.
The [Node completion example](../../cera-wasm/examples/explicit_loading.cjs)
uses the production CPU WASM module, with [build instructions](../../cera-wasm/README.md#explicit-cpu-model-loading).
The [multi-turn chat examples](#multi-turn-conversational-chat-with-live-kv-retention)
demonstrate warm multi-turn conversation with delta-only prompt evaluation across Rust, Swift, Kotlin, Python, and Dart.

## Load once and continue the same Rust session

The complete [walkthrough](../../tests/api_loading/consumer/src/bin/walkthrough.rs)
loads the probe's two-token GGUF, releases the model handle, generates three
tokens, then appends one new token and generates three more on the same session.
Its position assertions are 5 and 9. It uses CPU inference and keeps live KV
between the two calls. The token IDs are specific to the fixture; this is raw
token continuation, not a chat-template or incremental-chat guarantee.

```rust
use cera::{ModelLoader, ModelSource};
use cera::{BackendPreference, EngineConfig, GenerateOpts, SessionConfig};

let model = ModelLoader::new(ModelSource::path(model_path))
    .config(EngineConfig {
        backend: BackendPreference::Cpu,
        context_size: 24,
        ..EngineConfig::default()
    })
    .build_generative()?;
let mut session = model.create_session(SessionConfig::default())?;
drop(model);

let opts = GenerateOpts {
    temperature: 0.0,
    max_tokens: 3,
    ignore_eos: true,
    ..GenerateOpts::default()
};
session.append_tokens(&[0, 1])?;
session.generate(&opts, &mut first_sink)?;
assert_eq!(session.position(), 5);
session.append_tokens(&[1])?;
session.generate(&opts, &mut next_sink)?;
assert_eq!(session.position(), 9);
```

`model_path` and the two `ModalitySink` implementations are supplied by the linked
complete program. Run the [probe setup](../../tests/api_loading/README.md) first.
Set `LOADING_RUN` to the absolute `build/run-*` directory printed by that run.
The following command reads the exact target from its report and runs the
example against the same fixture:

```bash
export LOADING_RUN=/absolute/path/to/tests/api_loading/build/run-...
export CERA_GIT_SHA=loading-probe
export CARGO_TARGET_DIR="$(python3 -c 'import json,os; print(json.load(open(os.environ["LOADING_RUN"]+"/results.json"))["target"])')"
cargo run --manifest-path "$LOADING_RUN/workspace/Cargo.toml" \
  -p loading-consumer --bin walkthrough --features cera/mmap \
  --locked --offline -- "$LOADING_RUN/model.gguf"
```

With the probe's fixture the first line reports three tokens at position 5 and
the second reports three tokens at position 9. The program asserts counts and
positions; it does not establish a latency budget or device performance.

## Load and execute audio companions

The [audio walkthrough](API_RESHAPE_AUDIO_EXAMPLE.md) loads an encoder and vocoder,
ingests mono PCM after releasing the model handle, and captures 24 kHz output.
It includes complete source, local fixture export and runnable commands, plus
the byte/file fallback rules. Its synthetic weights prove execution mechanics.

## Choose a source or discover the model kind

These are alternatives to the source in the Rust example:

```rust
ModelSource::bytes(std::sync::Arc::<[u8]>::from(gguf_bytes))
ModelSource::reader(std::io::Cursor::new(gguf_bytes))
ModelSource::files(model_files) // requires mmap
ModelSource::parts(model_parts)
```

A loader is consumed by either `build_generative()` or `build()`. For dynamic
loading, call `build()`, inspect `handle.kind()`, then call
`handle.as_generative()` to obtain a shared model handle. Keep a wildcard when
matching the non-exhaustive handle enum, as shown by the compiled
[wildcard consumer](../../tests/api_loading/consumer/src/bin/wildcard.rs).
Known encoder, Whisper and VAD inputs return a kind mismatch from the current
generative builder; the prototype does not yet provide their typed facades.

## Attach real vision or draft weights

For a paired LFM2 vision primary and projector, use explicit image mode when
working with either bytes or files:

```rust
use cera::{ModelBytes, SessionConfig};
use cera::manifest::InferenceType;

let mut parts = ModelBytes::text(primary_bytes);
parts.multimodal_projector = Some(projector_bytes);
parts.inference_type = Some(InferenceType::LlamaCppImageToText);
let model = ModelLoader::new(ModelSource::parts(parts))
    .config(load_config)
    .build_generative()?;
let mut session = model.create_session(SessionConfig::default())?;
drop(model);
session.append_image(&png_bytes)?; // requires vl-preprocess
session.append_tokens(&text_token_ids)?;
session.generate(&options, &mut sink)?;
```

The [vision fixture and execution tests](../../cera/src/engine/loading_prototype/tests/auxiliary/vision.rs)
provide all inputs: a complete one-block LFM2 primary, a one-block ViT/projector
and a PNG built in memory. They compare image ingestion with independently loaded
encoder output and text continuation, including sessions after model release
and, on Unix, deletion of path inputs. Real application prompts also need the model's
image markers and chat format; this fixture tests loading and execution.

To add a paired DSpark sidecar, set `parts.draft_model = Some(draft_bytes)` before
loading. Successful draft bytes win over `EngineConfig::draft_model`. If bytes
fail to load, the configured draft path is tried when mmap is enabled. For file
or manifest sources, a configured path takes precedence over the source's draft
path; a failed configured path does not fall through to the manifest's path.
Optional failure leaves ordinary generation available. The
[draft tests](../../cera/src/engine/loading_prototype/tests/auxiliary/draft.rs)
load two valid companions that draft different tokens, observe actual session
draft calls and compare final output with ordinary greedy decoding.

Run these examples without a downloaded model:

```bash
cargo test -p cera --lib engine::loading_prototype::tests::auxiliary --locked --offline
cargo test -p cera --no-default-features --lib engine::loading_prototype::tests::auxiliary --locked --offline
```

A manifest capability, a parsed projector GGUF and usable typed vision weights
are distinct. Explicit text bytes ignore the projector; file loads can attach
it while `image_in` remains false. A valid GGUF header without required tensors
can leave `image_in` true but no usable encoder. See the
[loading audit](API_RESHAPE_P0_LOADING.md) for the preserved contracts.

## Swift and Kotlin loading

These snippets use the generated **probe** API. `generate()` in that facade is
fixed to three greedy tokens for the fixture. It is not the proposed full
sampling/streaming API, and the module is not a published Leap replacement.
The complete consumers below include imports, executable entry points and
lifetime/error assertions; the probe command compiles and runs both.

Swift, with `bytes` read from the run's `model.gguf`:

```swift
import Foundation
import loading_native

let config = LoadConfig(
  contextSize: 24, backend: "cpu", draftModel: nil, gpuDepthformer: false)
let loader = ProbeModelLoader(source: .bytes(bytes: bytes), config: config)
let model = try loader.buildGenerative()
let session = try model.createSession()
try session.append(tokens: [0, 1])
let tokens = try session.generate()
precondition(tokens.count == 3 && session.position() == 5)
```

Kotlin, with `bytes` read from the same fixture:

```kotlin
import uniffi.loading_native.LoadConfig
import uniffi.loading_native.ProbeModelLoader
import uniffi.loading_native.Source

ProbeModelLoader(Source.Bytes(bytes), LoadConfig(contextSize = 24uL, backend = "cpu")).use { loader ->
    loader.buildGenerative().use { model ->
        model.createSession().use { session ->
            session.append(listOf(0u, 1u))
            val tokens = session.generate()
            check(tokens.size == 3 && session.position() == 5u)
        }
    }
}
```

The production native API takes `EngineConfig` and returns the existing engine
and full session API. Swift applications import `Cera`; Kotlin applications import
`uniffi.cera_ffi`. The test runner additionally compiles the legacy Probe controls;
its Swift module combines both components under `loading_native`. Build the native
library and generated wrappers together from this checkout.

```swift
import Cera

let typedLoader = ModelLoader(
  source: .bytes(bytes: bytes),
  config: EngineConfig(backend: .cpu))
let typedModel = try typedLoader.buildGenerative()
let sharedEngine = typedModel.engine()
let configured = try typedModel.createSession(
  config: SessionConfig(maxSeqLen: 8, kvCompression: .f16, seed: 42))
try configured.appendTokens(tokens: [0, 1])
let output = try configured.generate(
  opts: GenerateOpts(maxTokens: 1, temperature: 0, ignoreEos: true))
precondition(output.tokens.count == 1 && configured.position() == 3)
```

```kotlin
import uniffi.cera_ffi.BackendPreference
import uniffi.cera_ffi.EngineConfig
import uniffi.cera_ffi.GenerateOpts
import uniffi.cera_ffi.KvCompression
import uniffi.cera_ffi.SessionConfig
import uniffi.cera_ffi.ModelSource
import uniffi.cera_ffi.ModelLoader

ModelLoader(
  ModelSource.Bytes(bytes), EngineConfig(backend = BackendPreference.CPU)
).use { loader ->
  loader.buildGenerative().use { model ->
    model.createSession(
      SessionConfig(maxSeqLen = 8u, kvCompression = KvCompression.F16, seed = 42uL)
    ).use { session ->
      session.appendTokens(listOf(0u, 1u))
      val output = session.generate(
        GenerateOpts(maxTokens = 1u, temperature = 0f, ignoreEos = true))
      check(output.tokens.size == 1 && session.position() == 3u)
    }
  }
}
```

The complete production-session fixtures are
[Swift](../../tests/api_loading/consumers/ProductionProbe.swift) and
[Kotlin](../../tests/api_loading/consumers/ProductionProbe.kt). They check caller
configuration, shared ownership, parent release, context overflow, streaming
cancellation and continuation. Consult the handoff for the current validation
result; compiling the wrappers alone does not establish runtime correctness.

Complete sources: [Swift](../../tests/api_loading/consumers/LoadingProbe.swift),
[Kotlin](../../tests/api_loading/consumers/LoadingProbe.kt),
[Node](../../tests/api_loading/consumers/loading_probe.cjs).
See [Leap compatibility status](API_RESHAPE_LEAP_COMPAT.md) for the separate
Swift/Kotlin drop-in workstream. Its actual runner, history, cancellation,
LoRA/embeddings and replacement packages remain gated.

## Multi-turn conversational chat with live KV retention

The high-level chat coordinator manages conversational state machine lifecycles
(`SessionPhase`), turn delimiter framing, and transactional ingestion over an underlying
inference `Session`. In-capacity consecutive turns evaluate only newly ingested user
messages and continuation tokens (delta-only prefill), without replaying history or
rebuilding the KV cache.

Complete runnable examples:
- Rust: [`cera/examples/chat.rs`](../../cera/examples/chat.rs)
- Swift: [`cera-ffi/examples/Chat.swift`](../../cera-ffi/examples/Chat.swift)
- Kotlin: [`cera-ffi/examples/Chat.kt`](../../cera-ffi/examples/Chat.kt)
- Python: [`cera-ffi/examples/chat.py`](../../cera-ffi/examples/chat.py)
- Dart: [`cera_ffi/example/chat.dart`](../../cera_ffi/example/chat.dart)

### Rust Multi-Turn Workflow

```rust
use cera::{GenerateOpts, Message, ModelLoader, ModelSource, SessionConfig, SessionPhase};

let model = ModelLoader::new(ModelSource::path(path)).build_generative()?;
let session = model.create_session(SessionConfig::default())?;
let mut chat = session.into_chat().map_err(|(_, err)| err)?;

// Turn 1: Ingest system and user messages together
let turn1_messages = vec![
    Message::system("You are a helpful assistant."),
    Message::user("What is a KV cache?"),
];
chat.ingest_messages(&turn1_messages)?;
let turn1 = chat.complete(&GenerateOpts::default())?;
println!("Assistant: {}", turn1.text);

// Turn 2: Warm continuation. The previous KV context remains resident.
chat.ingest(&Message::user("When should it be discarded?"))?;
let turn2 = chat.complete(&GenerateOpts::default())?;
println!("Assistant: {}", turn2.text);

// Reclaim the raw Session when done
let raw_session = chat.into_session();
```

### Swift Multi-Turn Workflow

```swift
import Cera

let loader = ModelLoader(source: .path(path: path), config: EngineConfig(backend: .cpu))
let model = try loader.buildGenerative()
let session = try model.createSession(config: SessionConfig())
let chat = try session.intoChat()

// Turn 1
try chat.ingestMessages(messages: [
    chatMessageSystem(content: "You are a helpful assistant."),
    chatMessageUser(content: "What is a KV cache?"),
])
let turn1 = try chat.complete(opts: GenerateOpts(maxTokens: 64, temperature: 0.7))
print("Assistant: \(turn1.text)")

// Turn 2: Delta-only prompt evaluation
try chat.ingest(message: chatMessageUser(content: "When should it be discarded?"))
let turn2 = try chat.complete(opts: GenerateOpts(maxTokens: 64, temperature: 0.7))
print("Assistant: \(turn2.text)")

// Reclaim raw session
let rawSession = try chat.intoSession()
```

### Kotlin Multi-Turn Workflow

```kotlin
import uniffi.cera_ffi.*

ModelLoader(ModelSource.Path(modelPath), EngineConfig(backend = BackendPreference.CPU)).use { loader ->
    loader.buildGenerative().use { model ->
        model.createSession(SessionConfig(seed = 42uL)).use { session ->
            session.intoChat().use { chat ->
                val opts = GenerateOpts(maxTokens = 64u, temperature = 0.7f)

                // Turn 1
                chat.ingestMessages(listOf(
                    chatMessageSystem("You are a helpful assistant."),
                    chatMessageUser("What is a KV cache?")
                ))
                val turn1 = chat.complete(opts)
                println("Assistant: ${turn1.text.trim()}")

                // Turn 2: Delta-only prompt evaluation
                chat.ingest(chatMessageUser("When should it be discarded?"))
                val turn2 = chat.complete(opts)
                println("Assistant: ${turn2.text.trim()}")

                // Reclaim raw session
                chat.intoSession().use { reclaimedSession ->
                    println("Reclaimed session position: ${reclaimedSession.position()}")
                }
            }
        }
    }
}
```

### Python Multi-Turn Workflow

```python
import cera_ffi

loader = cera_ffi.ModelLoader(
    cera_ffi.ModelSource.Path(model_path),
    cera_ffi.EngineConfig(backend=cera_ffi.BackendPreference.CPU),
)
model = loader.build_generative()
session = model.create_session(cera_ffi.SessionConfig(seed=42))
chat = session.into_chat()

opts = cera_ffi.GenerateOpts(max_tokens=64, temperature=0.7)

# Turn 1
chat.ingest_messages([
    cera_ffi.chat_message_system("You are a helpful assistant."),
    cera_ffi.chat_message_user("What is a KV cache?"),
])
turn1 = chat.complete(opts)
print(f"Assistant: {turn1.text.strip()}")

# Turn 2: Delta-only prompt evaluation
chat.ingest(cera_ffi.chat_message_user("When should it be discarded?"))
turn2 = chat.complete(opts)
print(f"Assistant: {turn2.text.strip()}")

# Reclaim raw session
reclaimed_session = chat.into_session()
```

### Dart Multi-Turn Workflow

```dart
import 'package:cera_ffi/cera_ffi.dart';

final loader = ModelLoader.create(
  ModelSourcePath(path: modelPath),
  const EngineConfig(backend: BackendPreference.cpu),
);
final model = loader.buildGenerative();
final session = model.createSession(const SessionConfig(seed: 42));
final chat = session.intoChat();

const opts = GenerateOpts(maxTokens: 64, temperature: 0.7);

// Turn 1
final turn1Messages = [
  chatMessageSystem('You are a helpful assistant.'),
  chatMessageUser('What is a KV cache?'),
];
chat.ingestMessages(turn1Messages);
final turn1 = chat.complete(opts);
print('Assistant: ${turn1.text.trim()}');

// Turn 2: Delta-only prompt evaluation
chat.ingest(chatMessageUser('When should it be discarded?'));
final turn2 = chat.complete(opts);
print('Assistant: ${turn2.text.trim()}');

// Reclaim raw session
final reclaimedSession = chat.intoSession();
```

## Cache behavior and documentation follow-through

The new model/session split retains live session KV. CPU LFM2 and the LFM2
wgpu/Metal cache implementations keep warm prefix caching for anonymous
byte/reader/parts models but ignore persistent `cache_dir`. Path or explicit raw
IDs retain their existing cold-cache namespace behavior; path identity is not
content-validated. The CPU Llama walkthrough has no reusable model prefix cache:
it demonstrates the live KV retained by one session. No speed measurement is
implied by either result.

On public export, move the examples to the public imports and package names,
compile them against the shipped Rust/Swift/Kotlin surfaces, and add the reviewed
chat, recovery, streaming and Leap migration examples as those features land.
Keep the root, crate and binding READMEs linked to the executable examples.

## Remote companion examples

Run the [remote companion examples](API_RESHAPE_REMOTE_EXAMPLES.md) to exercise
HF selection, cache repair, remote manifest assets and retained vision/audio/draft
execution. The guide links complete runnable tests and explains their fixture
inputs, concrete outputs and CPU/public-API limits.

## SafeTensors conversion

[Run the conversion examples](API_RESHAPE_CONVERSION_EXAMPLES.md) to convert
single or sharded weights, retain a session, reuse the converted cache and resume
an interrupted conversion through the loading API.

## Persistent cache identities

[Run the cache examples](API_RESHAPE_CACHE_EXAMPLES.md) to atomically replace model
weights at one path, keep the original session alive, and verify that a new model
does not restore the old weights' persisted state. The unchanged-weight control
measures tokens actually computed and confirms that disk prefix reuse still works.
