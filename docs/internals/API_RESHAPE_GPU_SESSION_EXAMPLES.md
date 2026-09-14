# GPU session ownership

A loaded Metal or wgpu model owns one live KV/convolution context. Creating a
second `Session` from that model now returns `CeraError::Busy`. Dropping the first
session releases ownership. Resetting, cancelling, or finishing generation keeps
the session alive and therefore keeps its reservation.

CPU models keep live state in each session and continue to support multiple
sessions sharing one loaded model. For simultaneous GPU conversations, load
separate model instances. Cloning an engine or the private `GenerativeModel`
handle shares the same model and does not create a second context.

## Use the public API

This is the lifecycle exercised by the
[device tests](../../cera/src/engine/loading_prototype/tests/ownership/gpu_sessions.rs).
Use a local LFM2 GGUF compatible with the selected GPU backend:

```rust,no_run
use cera::{BackendPreference, CeraEngine, CeraError, EngineConfig, SessionConfig};

fn successive_sessions(path: &str) -> Result<(), CeraError> {
    let engine = CeraEngine::from_path(path, EngineConfig {
        backend: BackendPreference::Metal, // Gpu selects wgpu; Cpu permits sharing.
        context_size: 64,
        ..EngineConfig::default()
    })?;
    let mut first = engine.new_session(SessionConfig::default())?;
    first.append_text("Hello")?;
    assert!(matches!(engine.new_session(SessionConfig::default()), Err(CeraError::Busy)));

    first.reset()?;
    assert!(matches!(engine.new_session(SessionConfig::default()), Err(CeraError::Busy)));
    drop(first);

    let mut next = engine.new_session(SessionConfig::default())?;
    assert_eq!(next.position(), 0);
    next.append_text("A new conversation")?;
    Ok(())
}
```

`Session::new` applies the same rule when given the engine's `model_arc()` and
`tokenizer_arc()` directly. A session retains its model after the engine is
dropped. Separate cancellation/position handles do not retain the session or
its reservation. A failed constructor releases its reservation; it does not
roll back the backend's existing compression configuration. GPU compression
mode and seed remain fixed per loaded model, so changing them still requires
a separately loaded model.

The existing Swift/Kotlin error mapping is unchanged: `newSession` reports
Swift `FfiError.Busy` or Kotlin `FfiException.Busy`. Release every Swift reference
to the session (ARC), or close the Kotlin session (`use`/`close`). An in-flight
operation can retain the underlying session until it completes. No foreign
methods or records are added; generated bindings remain unchanged.

## Transcription during a conversation

`CeraEngine::transcribe` uses the model's LFM2-Audio "Perform ASR." workflow;
it is separate from standalone Whisper transcription. If a conversation owns
the primary GPU context, the engine lazily loads a second model from the retained
primary GGUF backing. Only engines with an attached audio encoder retain this
backing for transcription; text/VL GPU byte loads release their staging bytes.
It follows the same backend preference and reuses the
loaded auxiliary weights. It does not reopen source paths, download weights,
reset the conversation or copy its live KV.

This adds first-use setup and memory for another model/context. Successful helper
loads remain cached until the engine is dropped. The private helper disables
prefix caching, so it retains no warm snapshots or disk entries outside the
engine's public cache controls; its active decode KV is still available normally.
Calls using that helper are
serialized through Session destruction; a failed load leaves the slot retryable,
and an input/decode error releases its Session. Other constructor errors retain
their existing behavior. Ordinary session creation still returns `Busy` while
another session owns that same model.

The native Dart `Cera` adapter releases an empty session before applying a new
seed. Failed replacement leaves an empty slot that a later operation can recreate;
cancel and close do not create sessions. Continuing conversations retain their
session. Dart transcription uses the independent core path above.

## Reproduce the checks

From the worktree, with Cargo dependencies already cached:

```bash
# Device-independent constructor, failure, race and drop-order coverage.
cargo test -p cera --lib session_ownership_tests --locked --offline

# Existing real CPU sharing/continuation controls, using tiny GGUF weights.
cargo test -p cera --lib cpu_interleaving_reset_cancel_and_extraction --locked --offline
cargo test -p cera --lib cpu_parallel_sessions_match --locked --offline

# macOS host with native Metal and wgpu device access. Tests fail if unavailable.
cargo test -p cera --features gpu,metal --lib gpu_sessions --locked --offline -- \
  --ignored --nocapture --test-threads=1

# ASR helper reuse, errors, concurrent calls and unchanged live conversation KV.
cargo test -p cera --features gpu,metal --lib transcription_preserves_live_conversation \
  --locked --offline -- --ignored --nocapture --test-threads=1
```

The five device-independent tests cover a provided `Model` hook with no
reservation, `Busy` before configuration, ownership across reset/cancel,
constructor errors and unwinding, eight racing constructors, and resource
destruction before the reservation is released.

The two device tests each run uncompressed and TurboQuant KV. They use a complete
synthetic GGUF with both convolution and attention blocks, explicitly request
the backend, and leave warm prefix caching enabled. They reject second sessions
through the prototype, public engine and direct constructor. A conflicting
compression request must also return `Busy` while a session is active. The
tests also observe that the input byte allocation is released after GPU upload. The
first session continues after parent handles are dropped, and its logits and
generated tokens match a separately loaded control. Reset and successor-session
outputs are checked against newly loaded models. A compression-conflict error
after drop must not prevent the next valid session from being created.

Two additional device tests compare ASR against independently loaded controls,
with a live uncompressed or TurboQuant conversation. They inject a failed helper
load, verify no helper warm entries/bytes before and after public cache operations,
reject empty PCM, reuse the same helper through concurrent calls, compare
live logits and generation after transcription, and load the helper after its
original primary/encoder files have been deleted on Unix.

The [Dart consumer](../../cera_ffi/example/gpu_ownership_probe.dart) runs six
checks through the actual portable API on a Metal host. Export its small synthetic
fixtures with this command, then pass the printed directory to the consumer:

```bash
cargo test -p cera --features metal --lib export_gpu_ownership_fixtures \
  --locked --offline -- --ignored --nocapture
cargo build -p cera-ffi --lib --features metal,ffi-buffer --locked --offline
# From cera_ffi/, using the actual absolute library and printed fixture paths:
CERA_FFI_LIB=/absolute/target/debug/libcera_ffi.dylib \
  dart run example/gpu_ownership_probe.dart /printed/fixture/directory
```

It covers seeded generation, transcription with empty and retained conversations,
error recovery, continuation and reseeding after reset. The standalone probe
explicitly exits after closing its model handles because generated callback
vtables retain process-global Dart listeners. It does not prove automatic VM
shutdown or speech quality.

This is native correctness evidence. It does not establish numeric throughput
budgets, Android/iOS device behavior, or browser WebGPU session ownership.
The browser has a separate async execution path and remains an explicit gate.

## Swift and Kotlin conversation lifetimes

The [native ownership runner](../../tests/gpu_session_ffi/README.md) compiles the
current generated bindings and executes the same lifecycle matrix on Metal and
wgpu. Load with `try await CeraEngine.fromBytesAsync(bytes:config:)` in Swift or
`CeraEngine.fromBytesAsync(bytes, config)` in a Kotlin coroutine; the native
blocking worker handles model initialization. Synchronous debug wgpu loading on
a small-stack Swift cooperative worker can overflow during shader compilation,
and is outside this ownership proof. It also checks pending async calls and CPU sharing. The snippets below are
helpers called by those actual consumers; `tokens` are IDs from the loaded model's
tokenizer. These helpers create separate conversations. To continue one
conversation, keep its Session and append only new context.

```swift
import Cera

func generateConversation(engine: CeraEngine, tokens: [UInt32]) throws -> [UInt32] {
    let session = try engine.newSession(config: SessionConfig(seed: 42))
    try session.appendTokens(tokens: tokens)
    return try session.generate(opts: options()).tokens
}

func options() -> GenerateOpts {
    GenerateOpts(maxTokens: 3, temperature: 0, ignoreEos: true, flushEveryTokens: 1)
}
```

ARC releases the local Swift Session after the helper returns. Keep every
reference in mind: a pending async call retains its Session until completion.

```kotlin
import uniffi.cera_ffi.*

fun generateConversation(engine: CeraEngine, tokens: List<UInt>): List<UInt> =
    engine.newSession(SessionConfig(seed = 42uL)).use { session ->
        session.appendTokens(tokens)
        session.generate(options()).tokens
    }

fun options() = GenerateOpts(maxTokens = 3u, temperature = 0f, ignoreEos = true, flushEveryTokens = 1u)
```

Kotlin `use` closes the Session deterministically. Closing its wrapper while an
async native call is pending does not stop that work or release GPU ownership
immediately. Request cancellation and await completion before expecting a new
Session to succeed. Calls in these small synchronous examples should run on an
application worker thread when used in a UI.

The native ownership check also has a failure control: bypassing Session
ownership acquisition makes both unchanged language consumers fail at their
first Metal `Busy` assertion. Restoring the source and normal build restores all
51 passing cases, including both GPU backends and CPU sharing. See the
[handoff evidence](API_RESHAPE_HANDOFF.md) for reproduction details.

## Raw model callers

The per-call GPU mutex still protects scratch and command bookkeeping. It is
insufficient for interleaved conversation histories. The new reservation is
acquired once at session construction and dropped once at destruction; no
per-token lease checks, KV copies, or cache clears were added.

Direct `Model` forward/configuration/cache calls remain caller-managed and must
not modify a context owned by a live session. Custom stateful `Model`
implementations can retain a `ModelSessionGate` and override `acquire_session`
to return its non-cloneable lease. Implementations that delegate to another
stateful model must also delegate its reservation policy. The default hook is
appropriate when live inference state belongs entirely to the caller.
