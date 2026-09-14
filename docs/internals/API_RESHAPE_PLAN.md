# Cera Public API Reshape

Plan31 completed 2026-09-12T08:18-0700: complete foreign multipart defaults first, with fresh
native/WASM builds and generated consumers passing39/39/21 cases. The text-only
control is rejected in all three languages; two max-effort review rounds with
three reviewers end clean. 28 bounded increments complete (26 core, two Leap),
no major phase closed. Native production config reuse and
shared engine/full Session access remain subsequent proofs; P0-L stays open.
See the [current checklist](API_RESHAPE_HANDOFF.md#plan31--foreign-multipart-defaults).
Next unused32.

Plan30 completed 2026-09-12T06:11-0700: [per-target API retention](API_RESHAPE_TARGET_RETENTION.md)
and executable declaration checks. The audit selects existing native EngineConfig
and shared engine/full Session access for promotion. Full Text/Audio/Other
multipart defaults also need foreign representation and execution. These binding
proofs remain Plan31. Ten declaration controls and scoped checks pass; three
max-effort review rounds end clean. Twenty-seven bounded increments are complete
(25 core, two Leap); no major phase closed. See the
[handoff](API_RESHAPE_HANDOFF.md#plan30--target-api-retention). Next unused31.
The old native/WASM build artifacts became unavailable during Plan30; surviving
sources, generated artifacts and result records are verified separately. Rebuild
before another runtime claim; prior passing runs remain historical evidence.

Plan29 completed 2026-09-12T04:16-0700: native remote sources, repository sharing and
progress ownership pass 33 Swift, 33 Kotlin and 15 Node cases. Guarded consumer
replay verifies unchanged native/WASM builds with fresh local stores; scoped
checks and two max-effort rounds with three reviewers end clean. Twenty-six
bounded increments complete (twenty-four core, two Leap); no major phase closed.
See the [completion record](API_RESHAPE_HANDOFF.md#completed-plan29--native-remote-loading).
Next unused sequence 30: per-target binding retention and additive loading
signatures. Earlier dated checkpoints below are historical.

Plan28 completed 2026-09-11T20:31-0700: structured loading categories and native/WASM
payload consumers pass 24/24/15 cases. Core loading tests pass 25/36/70 cases;
scoped lint/Clippy/Rustdoc and artifact/document audits pass. Two max-effort rounds
with three reviewers end clean after strengthening exact remote error assertions.
Twenty-five bounded increments complete (twenty-three core, two Leap); no major
phase closed. See the [completion record](API_RESHAPE_HANDOFF.md#completed-plan28--structured-loading-errors).
Next unused sequence 29: native remote sources and repository/progress ownership.
Reclaim disposable build caches first; earlier dated checkpoints below are historical.

Plan27 completed 2026-09-11T18:42-0700: native context width/default/zero and capacity
contracts pass actual Swift/Kotlin consumers; the checked conversion also rejects
overflow on wasm32. The full matrix passes22/22/13 cases, scoped lint/Clippy/Rustdoc
and artifact checks. One max-effort round with three reviewers ends clean.
Twenty-four bounded increments complete (twenty-two core, two Leap); no major
phase closed. Next unused sequence28: structured loading error categories.
See the [completion record](API_RESHAPE_HANDOFF.md#completed-plan27--native-loading-configuration)
and [loading exit audit](API_RESHAPE_LOADING_EXIT.md). Earlier dated checkpoints
below preserve their original state.

Plan26 completed 2026-09-11T18:04-0700: native multipart file consumers preserve all eight
fields, both builders, errors and retained sessions. The full generated run passes
17 cases each in Swift/Kotlin and 12 in Node. Scoped checks and two max-effort
review rounds pass; all final reviews are clean. The [loading exit audit](API_RESHAPE_LOADING_EXIT.md)
records concrete configuration, source, error and binding prerequisites separately
from later chat/release gates. Twenty-three bounded increments complete (twenty-one
core, two Leap); no major phase closed. Next unused sequence27. See the
[completion record](API_RESHAPE_HANDOFF.md#completed-plan26--loading-exit-audit-and-native-files).
Earlier dated checkpoints below preserve their original state.

Plan25 completed 2026-09-11T15:30-0700: the isolated loading matrix passes 13 cases each
in Swift/Kotlin and 12 in Node, including Hotword/kws mismatch and consumed-loader
checks. All three consumers reject wrong classification and pass after source
restoration; strict case parsing rejects false-status maps. Scoped checks and
one max-effort round with three reviewers pass; all reviews are clean.
Twenty-two bounded increments are complete (twenty core, two Leap); no major
phase is closed. Next unused sequence 26. See the
[completion record](API_RESHAPE_HANDOFF.md#completed-plan25--foreign-hotword-loading)
and [remaining phase checklist](API_RESHAPE_HANDOFF.md#phase-checklist).
Earlier dated checkpoints below preserve their completion state.

Plan23 completed 2026-09-11T10:42-0700: real Swift/Kotlin consumers each pass 51 native
GPU ownership cases, plus 11 Whisper regression cases. Acquisition-bypass
controls fail at the first Metal Busy assertion; restored source/build passes
the full matrix again. Three max-effort rounds with three reviewers each are
complete; the final round is clean after fixing result validation and fingerprints for shaders, build scripts
and inherited Cargo configuration. Runnable examples, lint and
handoff records are current. Twenty-one bounded increments are complete
(nineteen core, two Leap); no major phase is closed. See the
[current completion record](API_RESHAPE_HANDOFF.md#completed-plan23--native-gpu-binding-ownership)
and [remaining phase checklist](API_RESHAPE_HANDOFF.md#phase-checklist). Next unused
plan sequence is 25; earlier dated checkpoints below are historical.

Plan24 completed 2026-09-10T18:38-0700: rebased onto `origin/main` at `2477a68`
(0.5.6), preserving hotword/Whisper and async multipart APIs, prior GPU/Dart
ownership changes and all working files. Bindings, scoped checks, native
consumers and three max-effort review rounds pass; all final reviews are clean.
The [handoff](API_RESHAPE_HANDOFF.md#completed-plan24--rebase-and-incoming-api-integration)
records the incoming API inventory, corrected hotword event/cancellation contracts
and remaining checklist. Plan23 negative controls and final review are next.
Twenty earlier feature increments remain complete; no major phase is closed.

Plan22 completed 2026-09-10T13:20-0700: native GPU session ownership and its Dart/ASR
callers pass the final scoped validation, executable examples and five max-effort
review rounds; all three final reviews are clean. Twenty bounded increments are
complete, eighteen core and two Leap experiments. No major phase is complete.
See the [current completion and remaining checklist](API_RESHAPE_HANDOFF.md#completed-plan22--gpu-session-ownership)
and [runnable GPU examples](API_RESHAPE_GPU_SESSION_EXAMPLES.md). The dated
checkpoints below preserve earlier progress; the handoff is the current record.

Plan20 completed 2026-09-10T07:28-0700: all eleven scoped checks and three max-effort
review rounds pass after external-draft and prefixed-mirror fixes. Eighteen
bounded increments are complete, sixteen core and two Leap experiments.
[Executable HF snapshot examples](API_RESHAPE_HF_EXAMPLES.md) cover direct
GGUF revision pinning, cache/default freshness, failed resolution and live CPU
session continuation. Existing companion fixtures now require commit URLs. This
is a bounded P0-L increment; no major phase or public binding rollout is complete.


Plan19 completed 2026-09-09T12:16-0700: conversion inputs, checkpoints and completed caches
use a resolved upstream commit. Eleven scoped gates and two max-effort rounds
pass; the final three reviewers are clean after correcting README retention
claims. See the [current handoff](API_RESHAPE_HANDOFF.md#completed-plan-19-upstream-conversion-revisions).
Seventeen bounded increments are complete; all major phases remain open.

Plan18 completed 2026-09-09T04:22-0700: partial-checkpoint integrity, repeated recovery,
runnable examples and eleven scoped gates pass. Three fresh max-effort reviewers
completed one round; the only documentation wording nitpicks are fixed. See the
[Plan18 handoff](API_RESHAPE_HANDOFF.md#completed-plan-18-checkpoint-prefix-integrity).
Sixteen bounded increments are complete; all major phase gates remain open.

Original reviewed baseline: `60fc11c25a51`, 2026-09-06. Plan24 rebases the
implementation onto `2477a6830dc5` (`origin/main`, 0.5.6). Its integration and
validation status are recorded in the [current handoff](API_RESHAPE_HANDOFF.md).

This revision incorporates the review of the earlier API reshape proposal and
supersedes `high_level_api_audit_and_redesign.md`. Rust and binding callers have
equal priority. Proposed names and examples below describe the intended API;
they are not available in the reviewed baseline. P0 must turn them into compiled
prototypes before their signatures are committed as public API.

Plan17 completes converted-cache integrity and request-aware checkpoint reuse.
The [conversion examples](API_RESHAPE_CONVERSION_EXAMPLES.md) demonstrate corruption
repair, sidecar recovery, override changes and retained execution. Eleven scoped
gates and two max-effort rounds pass; the final three reviewers are clean after
correcting a manifest-fixture key. Plan18 adds partial-checkpoint integrity; immutable
source identity and full converted-load hashing budgets remain open.

Plan16 completes loaded-byte identities for named persistent KV caches, lazy CPU
hashing outside cache locks, complete DSpark sources and exact Metal mapping
ownership. [Runnable cache examples](API_RESHAPE_CACHE_EXAMPLES.md), scoped checks,
fresh generated consumers and two max-effort review rounds pass; the final three
reviews have no open findings after two mapping comments were corrected. Measured
one-time hashing cost does not close performance gates.

Plan 15 adds [executable conversion examples](API_RESHAPE_CONVERSION_EXAMPLES.md)
for single/sharded SafeTensors, actual output weights, retained sessions, cached
reloads and interrupted conversion recovery. Scoped validation and two max-effort
review rounds with three reviewers each are complete; the final round is clean
after strengthening quantized-weight and checkpoint-truncation assertions.
This completes another bounded P0-L increment.

Plan 14 joins remote resolution with executable CPU companions; its
[runnable examples](API_RESHAPE_REMOTE_EXAMPLES.md), scoped feature checks and
one max-effort round with three clean reviewers are complete. This remains a
bounded P0-L increment.

Implementation progress and resume instructions are maintained in
[API_RESHAPE_HANDOFF.md](API_RESHAPE_HANDOFF.md). The loading audit and scoped validation are in
[API_RESHAPE_P0_LOADING.md](API_RESHAPE_P0_LOADING.md). Memory, local filesystem and loopback remote
prototypes cover bounded source/error/configuration contracts. Plan 08 adds a
[retained-operation and ownership audit](API_RESHAPE_P0_OWNERSHIP.md), private
forwards and CPU isolation/cache tests; its scoped validation and two max-effort
review rounds pass. Plan 09 adds [generated foreign loading consumers](API_RESHAPE_P0_BINDINGS.md)
for Swift/Kotlin and Node WASM; runtime, harness, Clippy and Rustdoc checks pass.
Its fifth max-effort review round is clean across all three reviewers, and the
bounded increment is complete. Plan 10 reuses parsed primary metadata across
classification, source resolution and assembly; parse-count/error-order controls,
feature tests and fresh generated consumers pass. Two max-effort review rounds
completed with three fresh reviewers each; the final round is clean after one
Linux-only assertion fix. Plan 10 is complete within that scope; Linux execution
of that regression remains unverified locally. Details are in the loading audit
and handoff. Plan 11 applies a shared warm-only cache policy to anonymous models
and removes Metal's incomplete sampled identity; numerical contamination controls
and feature-aware lint/doc checks pass. One max-effort round is clean across
three reviewers; fresh generated consumers, scoped lint/doc, source/artifact
hashes and documentation checks pass. Plan 11 is complete within that scope. Plan16 supplies named loaded-GGUF identities; numeric performance budgets
remain separate gates. Plan 12 adds executable synthetic CPU vision/DSpark companions,
loading precedence and retained-session controls; scoped checks and three
max-effort review rounds pass, with all final reviewers clean. Plan 12 is
complete within that scope.
[Runnable Rust, Swift and Kotlin examples](API_RESHAPE_EXAMPLES.md) now accompany
the root/crate/probe READMEs. Plan 13 adds CPU audio companion loading, source
precedence and retained execution with a [runnable audio walkthrough](API_RESHAPE_AUDIO_EXAMPLE.md);
scoped validation and three max-effort review rounds pass, all final reviewers
are clean, and Plan 13 is complete within that scope. Live catalog/CDN behavior, broader conversion coverage,
device sharing rules and remaining source coverage still gate public loading APIs. Chat and warm-KV gates
remain independent.

Plan21 is complete within its scope: it adds the hotword PR's standalone Whisper UniFFI surface and
[executable Swift/Kotlin recording examples](API_RESHAPE_WHISPER_EXAMPLES.md).
It preserves the existing foreign API names and leaves F5's unified loader
separate. The FFI options expose language, translation, timestamps, token limit
and temperature. Plan24 preserves upstream cooperative cancellation when async
foreign futures are dropped; the options still omit an explicit cancellation
handle. Kotlin propagates coroutine cancellation; the pinned Swift wrapper does
not propagate `Task.cancel()`, which remains a foreign migration gate. Plan22
enforces GPU session lifetimes around model-owned live KV.
The Whisper increment passes actual Swift/Kotlin execution, scoped Rust/FFI
checks, reproducible binding generation and two max-effort review rounds, with
all three final reviewers clean. Mobile devices and speech quality remain untested.

Plan22 enforces one live Session per built-in Metal/wgpu model with the existing
Busy error, retaining ownership through reset/cancel and releasing it on drop.
CPU sharing remains unchanged. The [GPU session walkthrough](API_RESHAPE_GPU_SESSION_EXAMPLES.md)
links executable lifecycle and real-device continuation tests. This addresses
native model-owned live state; raw calls, browser WebGPU, numeric performance
budgets and broader platform coverage remain separate gates.
Native Dart reseeding releases an empty Session before replacement. LFM2-Audio
transcription during a live conversation uses a cached secondary model loaded
from retained weights, preserving conversation KV at the cost of another model
allocation and first-use setup. The same walkthrough includes actual Dart
consumer and native ASR regression controls. Source backing is retained only with
an audio encoder, and the private helper disables prefix caching to avoid hidden
snapshots outside the engine cache controls.

Plan24 also retains the new native hotword detector/iterator APIs and async
multipart engine loading. P0-L classifies `kws` before generative assembly and
inventories its mutable stream ownership; P2 must preserve all hotword records,
constructors, processing/reset methods and `fromPartsAsync` through migration.
F5 includes eventual typed hotword loading alongside Whisper/VAD. Existing
standalone APIs stay supported while those facades are designed and validated.

## 0. Goal and scope

Make loading, chat, streaming, and multimodal input easy to discover, with clear
ownership and failure behavior. Keep raw inference operations available for
callers that already manage tokenization, embeddings, or templates.

**Live KV reuse is a release requirement.** For the initial supported chat
profiles, initialize context once and retain it across successful turns. Prefill
only new input and the explicitly required continuation/boundary tokens. A
normal turn must not reset and rebuild history to make the new API convenient.
Measure both time to first visible output and decode throughput; unchanged
decode kernels alone do not establish unchanged conversation performance.

**Part I — API reshape:** additive types and adapters over existing loading,
rendering, prefill, and generation paths. Preserve prompt tokens, generated
tokens, defaults, source selection, and streaming behavior for equivalent calls.
An adapter may compose existing operations; it must not quietly introduce a new
rendering algorithm, backend behavior, or recovery guarantee.

**Prerequisite R0 — ingestion recovery:** separately reviewed correctness work
on existing failure paths. It may change erroneous failure behavior and must
document that change. It is required before the new ingestion API promises the
recovery contract in §4.3. Successful-call parity remains required. R0 is not a
claim that the existing append methods are already transactional.

**Prerequisite R1 — initial warm-chat support:** P0 freezes the initial named
model/template/backend profiles and proves their turn-boundary and continuation
behavior. Reuse existing paths where they satisfy that contract. If a required
profile needs new rendering or execution bookkeeping, implement and review that
work separately before the corresponding Part I chat surface ships. Record its
behavior changes and reference fixtures. The initial fast path cannot be
deferred while full replacement is presented as the completed chat API.
R1 may intentionally correct malformed warm-path prompt envelopes, including
those reached through existing FFI helpers. Those named fixes are exceptions to
legacy prompt/output parity: preserve before/after regression fixtures, release-
note affected callers, and compare Part I wrappers to the corrected R1 reference.
This exception does not authorize unrelated changes to raw generation or defaults.

**Part II — capabilities:** independently gated additions such as pull
streaming, text stop sequences, and persistent sessions. These do not block
Part I. Additional incremental profiles and general rendering mechanisms beyond
R1 belong here. New rendering behavior must be explicitly assigned to R1 or F8,
rather than hidden inside a compatibility adapter.

**Part III — documentation portal:** site infrastructure can ship independently.
API reference, compiled examples, and migration documentation ship with Part I;
they are required deliverables, not deferred portal work.

**Leap SDK migration compatibility:** provide Swift and Kotlin replacement
facades as the Leap library is deprecated, including LoRA adapters and per-token
embeddings. Target source compatibility after dependency replacement/rebuild;
the replacement runtime is Cera. This workstream adds C0/C1/C2 gates alongside
Part I and has its own precise compatibility matrix (§4.7). It is not satisfied
by retaining the old SDK as a runtime dependency or by shipping text-only stubs.

## 1. Intended caller workflows

The common path is: choose a source, load a typed model, create a session,
provide context, and generate. Dynamic model discovery is optional. The caller
owns the transcript; the session owns execution state and retains its live KV
between turns. The examples assume a profile verified for warm chat in R1.

### 1.1 Initialize once and generate a first response

Proposed Rust sketch; `model_path` points to a text-generative GGUF:

```rust
let model = ModelLoader::new(ModelSource::path(model_path))
    .build_generative()?;
let mut session = model.new_session(SessionConfig::default())?;
let options = model.default_generate_config();

let mut messages = vec![
    Message::system("Answer briefly."),
    Message::user("What is a KV cache?"),
];
session.replace_messages(&messages)?;
let turn = session.complete(&options)?;
messages.push(Message::assistant(turn.text.clone()));
```

This first `replace_messages` initializes an empty session with the system and
user messages together. Subsequent ordinary turns use `ingest`, as below.

`replace_messages` explicitly replaces execution context using the supplied
full history. It is for initialization, deliberate history edits, recovery, or
an explicit fallback for a template without incremental support. It renders
and validates what can be checked before resetting,
then delegates to the existing reset and full-history prefill paths. It is not
an atomic context swap; failures after reset follow §4.3. Initial Part I support
is text and the existing full-history image-template path. Full-history audio
is not promised without an existing equivalent or a separate capability design.

`complete` generates from already ingested context. It never ingests the same
message again or silently falls back to the model's built-in sampling defaults.

### 1.2 Multi-turn chat

Proposed continuation of the preceding sketch:

```rust
let next = Message::user("When does that cache get discarded?");
session.ingest(next.clone())?;
messages.push(next);
let turn = session.complete(&options)?;
messages.push(Message::assistant(turn.text.clone()));
```

The previous prompt and generated context stay resident. `ingest` adds only the
new message plus the profile's required continuation and boundary tokens; it
does not replay the transcript or re-ingest the collected assistant text. The
caller keeps `messages` for display, edits, or explicit recovery. Part I
high-level chat requires `n_keep = 0`. At capacity the caller selects a smaller
message window and explicitly calls `replace_messages`; replacement with the
unchanged overflowing history is not a recovery strategy.

Warm continuation preserves the sampler's RNG progression while keeping the
existing per-generation repetition-history reset and option synchronization.
It is a different execution policy from the CLI's current reset-and-render loop,
which re-seeds at every reset and may use different prefill batching. Do not
claim token-for-token equality
between those policies. Compare new wrappers against an equivalent retained-
state reference, and preserve the legacy replacement workflow during migration.
These consecutive-turn sketches assume a profile-recognized end of the assistant
turn (`SessionPhase::TurnComplete`). A token limit, cancellation, or arbitrary
stop may leave `Interrupted`, even when generation returns `Ok`. Inspect the
phase before the next `ingest`; Part I then requires explicit replacement with
the transcript the caller chooses to keep. See §4.2.1 for no-op exceptions.
For multiple input messages before one answer, use `ingest_messages(&batch)`;
it adds one assistant generation prefix after the entire supported batch.

### 1.3 Streaming and cancellation

Proposed Rust call shape after context has been prepared:

```rust
let cancel = session.cancel_handle();
// Give a clone of `cancel` to the UI or control thread before starting decode.
let result = session.generate_into(&options, &mut sink);
// The control thread requests cancellation with:
// cancel.store(true, std::sync::atomic::Ordering::Relaxed);
```

`sink` implements the existing push `ModalitySink`. Ordinary and speculative
generation retain their current batching and terminal-callback behavior. A
cancel request must be possible without borrowing or locking the active session.
The operation result is authoritative; partial output already delivered to the
sink remains delivered. Cancellation and retry rules are specified in §4.4.
No pull iterator or new async runtime is required for this workflow.

### 1.4 Multimodal input and bindings

For images, load the primary model and projector through a multipart source,
then use a message whose ordered parts include image bytes and text. Raw media
belongs to that message; a successfully completed ingestion retains no message
or media payload. A caller retaining a transcript may still retain those bytes.
Audio input carries PCM samples and sample rate and delegates to the existing
audio-ingestion path; do not imply that images and audio can be combined when
the current model path rejects that combination.

Proposed Swift usage sketch for a text model, mirroring §1.1:

```swift
let model = try ModelLoader(source: .path(modelPath)).buildGenerative()
let session = try model.newSession(config: SessionConfig())
let options = model.defaultGenerateConfig()
try session.replaceMessages(messages: [
    Message.system("Answer briefly."),
    Message.user("What is a KV cache?")
])
let turn = try session.complete(options: options)

// Keep the same session for the next turn.
try session.ingest(message: Message.user("When is it discarded?"))
let followUp = try session.complete(options: options)
```

These Swift spellings are design intent, not generated-binding evidence. P0
must generate and compile the actual Swift and Kotlin equivalents and document
which calls run on a worker. Existing async native wrappers and browser async
loading/WebGPU execution remain available. A synchronous Rust core does not
justify blocking a UI thread or removing those entry points.

## 2. Verified baseline

Source references are repository-relative symbols at the baseline commit, not
line-number promises. Any future "Before" example must come from a real caller.

| Existing behavior | Source |
|---|---|
| Eager, cancellable prefill | [Session::append_tokens](../../cera/src/session.rs) |
| Borrow-free cancellation and position observation | `Session::cancel_handle`, `position_handle` in the same file |
| Existing warm/incremental message ingestion and combined generation, with one FFI session lock | [send_message, send_message_and_generate, send_message_streaming](../../cera-ffi/src/lib.rs) |
| One-call LFM2-Audio ASR | [CeraEngine::transcribe](../../cera/src/engine.rs) |
| Rust Whisper transcription options, including cancellation and max_tokens | [WhisperTranscribeOpts](../../cera/src/model/whisper.rs) |
| Standalone Swift/Kotlin Whisper ASR: file/bytes, sync/async, per-call cooperative cancellation on async future drop | [FfiWhisperModel](../../cera-ffi/src/lib.rs) |
| Hotword detector and stateful iterator, optional VAD, scores/events/configuration, foreign file/bytes loading and stream reset | [HotwordDetector/HotwordIterator](../../cera/src/hotword.rs), [FfiHotwordDetector/FfiHotwordIterator](../../cera-ffi/src/lib.rs) |
| Async multipart loading for native foreign callers, including Dart text/multimodal loading | [CeraEngine::from_parts_async](../../cera-ffi/src/lib.rs), [Dart adapter](../../cera_ffi/lib/src/async/cera_io.dart) |
| LoRA hot-swap affecting future forwards; adapter survives reset | `Session::attach_lora_adapters`, `remove_lora_adapters`, `reset` |
| Push text/audio sink and foreign callback adaptation | `ModalitySink` in `cera/src/session.rs` and `cera-ffi/src/lib.rs` |
| Ordinary and speculative generation paths | `Session::generate`, `generate_greedy_spec` |
| Separate audio loop and budget state machine | [generate_audio, AudioGenerateConfig](../../cera/src/audio_engine.rs) |
| Full-history CLI rendering after reset | [chat command](../../cera-cli/src/main.rs), [chat TUI](../../cera-cli/src/chat_tui.rs) |
| Backend snapshots and serialized KV schema | [Model](../../cera/src/model/mod.rs), [InferenceState](../../cera/src/kv_cache.rs), [schema bindings](../../cera/src/generated/kv_cache_generated.rs) |
| BERT/ModernBERT loading and hidden-state operations | [model loader](../../cera/src/model/mod.rs), [BertModel](../../cera/src/model/bert.rs) |

The migration reaches CLI, FFI, WASM, parity tools, examples, tests, and language
consumers. Recount callers in P0 instead of relying on the earlier approximate
295-call-site estimate. Coordinate with active branches, including encoder,
Apple audio, and website work, before deprecations or removals.

P0 records the execution policy per entry point, not just per language:

| Surface | Warm today vs. reset today | Migration consequence |
|---|---|---|
| Core raw append/generate | Retains state unless caller resets | Preserve raw token and chaining semantics |
| Core `append_user_message`; FFI `send_message*` | Already incremental; no automatic per-turn reset | R1 envelope fixes can change output even for unchanged callers |
| CLI chat and TUI chat loops | Reset and render full transcript each turn | Choosing warm chat changes execution policy explicitly |
| Generated Swift/Kotlin/Python/Dart message helpers | Expose the FFI incremental operations | Validate each consumer and release-note inherited R1 fixes |
| Native async wrappers, WASM/WebGPU, browser/Dart clients, examples, parity tools | Caller-specific; inventory reset/render/append sequences in P0 | Do not infer a workflow from a binding language or async signature |

## 3. Constraints

**I1 — Rewind is a backend capability.** The default `Model::truncate_kv`
delegates to `InferenceState::truncate_to`, which currently asserts on compressed
caches and out-of-range targets and can silently clear to zero when convolution
history is unavailable. The LFM2 GPU/Metal overrides update sequence counters;
they do not restore the device convolution rolling buffers. Counter rewind alone
cannot establish hybrid-model restoration. R0 needs a fallible, backend-aware
recovery path that reports unsupported rewind without partially mutating state.
A CPU-state check is insufficient to promise restoration of a backend's
attention, convolution, and compressed-cache state.

**I2 — Recovery covers more than KV.** `Session` retains logits, draft token
history, position and its atomic mirror, prefill statistics, sampler state, and
an optional drafter. CPU convolution history has a 64-entry ring. Check actual
checkpoint availability and advancement since the checkpoint, not absolute
conversation depth. A context shift removes middle rows that tail truncation
cannot recover. Omitting `Vec<Message>` does not remove these duties.

**I3 — Push generation has multiple implementations.** Preserve ordinary and
speculative dispatch, `flush_every_tokens`, and `flush_every_ms`. A pull iterator
requires a measured mechanism and a separate design, including WASM constraints.

**I4 — Templates may depend on history.** `apply_chat_template_with_tools`
evaluates the model's Jinja template over messages and tools. A full-history
render is not necessarily a previous render plus one independent turn. Text
token offsets also do not describe KV positions after media expansion. Never
derive a multimodal KV append offset by retokenizing the full text history.

**I5 — Audio and encoder operations remain explicit.** Preserve
`AudioGenerateConfig`, Whisper options, VAD processing, hidden states, pooling,
and classification access. A generative facade must not imply BERT can decode.
The reviewed generic model loader includes `lfm2`, `lfm2moe`, `llama`, `qwen2`,
`qwen3`, `granite`, `bert`, and `modernbert`; specialized audio/vision paths need
their own inventory. Current ViT integration is LFM2-VL. Do not invent support
for Qwen-VL or LLaVA by adding marker enum variants.

**I6 — Source compatibility includes types.** Existing `CeraError` is exhaustive.
New variants, renamed config fields, changed struct layouts, removed feature
combinations, and regenerated foreign APIs can break consumers independently of
method-name shims.

## 4. Part I — proposed surface and contracts

### 4.1 Loading: explicit sources and typed results

Provide `ModelSource::path`, `bytes`, `files`, `parts`, `reader`, `bundle`, and
`hf` forms backed by the corresponding constructors. Exact source type
signatures and feature gates are a P0 deliverable. `From<&Path>`/`From<PathBuf>`
are acceptable; there is no `From<&str>` inference between a local path and an
HF repository. Network resolution is explicit and preserves repository/cache,
quantization, strategy, URL, and download-progress behavior.

`ModelLoader::build_generative()` is the ordinary generative path.
`ModelLoader::build()` returns the Rust-side dynamic handle (D6):

```rust
#[non_exhaustive]
pub enum ModelHandle {
    Generative(GenerativeModel),
    // Additional typed variants ship only with their implemented facades.
}
```

Typed and dynamic loading share source resolution, classification, and assembly.
Check the requested model kind before expensive incompatible backend/weight
construction when possible. `build_generative()` must reject encoder-only,
Whisper, VAD, and hotword (`kws`) inputs with a typed mismatch or unsupported-kind error. Do not
suggest `build_vad()` as a remedy until that method actually ships.

`GenerativeModel` wraps existing generative engine functionality. Preserve
metadata, tokenizer access, capabilities, generation defaults, session creation,
LFM2-Audio transcription, and cache management. D9 gates encoder migration:
either introduce an encoder facade over existing operations or keep their
current loader and engine entry points supported. A Generative-only handle is
not a reason to remove existing encoder access. Whisper/VAD/hotword unification is F5.

The pinned `wasm-bindgen` 0.2.117 rejects enums with associated data. Use a WASM
object exposing `kind()` and typed accessors instead of exporting this Rust enum.
P0 must verify UniFFI object-payload support through actual generation; use the
same facade if it cannot express the desired contract. Define whether accessors
borrow, share ownership, or consume the handle, including mismatch behavior.
Rust `#[non_exhaustive]` does not prove foreign-language extensibility; test each
generated representation's unknown-kind behavior before adding variants.

**Constructor payload inventory:** preserve each source's supported fields and
ownership. Do not imply that every source supports every payload.

| Existing input | New home / preservation rule |
|---|---|
| Primary GGUF path, bytes, reader, manifest path | Explicit source variant; preserve mmap, buffering, and feature requirements |
| `ModelFiles` / `ModelBytes` model | Multipart files / parts source |
| Both types' `multimodal_projector` | Same multipart source; path or shared bytes |
| Both types' `audio_decoder` | Same multipart source |
| Both types' `audio_tokenizer` | Same multipart source |
| Both types' `draft_model` | Same multipart source; preserve precedence against load config |
| Both types' `inference_type` | Explicit override; preserve constructor-specific detection behavior |
| Both types' `chat_template` | Retain override metadata separately; baseline rendering uses the embedded GGUF template. Activating the override requires a declared rendering correction with prompt fixtures |
| `ModelFiles::extras` | Preserve named auxiliary paths |
| `ModelBytes::generation_defaults` | Preserve manifest-derived advisory defaults |
| Bundle id and quantization | Explicit bundle source; configured repository required as today |
| HF spec/URL, quantization, strategy | Explicit HF source; preserve resolver selection and progress |

No new projector filename guessing. Delegate to the existing manifest/HF
resolver rules and document their exact selection policy in P0. Explicit
multipart inputs retain current auxiliary parse/fallback behavior; stricter
validation must be a separately declared behavior change. Keep `Arc<[u8]>`
sharing and account for unavoidable copies at foreign-language boundaries.
P0.1 must explicitly scope any correction that activates manifest template
overrides; the loader extraction preserves existing rendered prompts.

### 4.2 Messages, rendering, and advanced prefill

**D7 stays locked:** `Session` does not retain application message history or
offer `retain_history`. A `Message` contains a `Role` and ordered `ContentPart`s;
text, image bytes, and PCM-plus-sample-rate data are self-contained. Define media
order explicitly per existing model path; canonical reordering must match the
legacy helper being adapted. No `ConsumedMedia` placeholder is needed.

The caller owns any transcript required for a later full render. `SessionPhase`,
profile identity, and bounded pending boundary-token state are execution
bookkeeping, compatible with D7. Include them in reset, recovery, and persistence;
they must not grow into a retained transcript or duplicate token history.

| Proposed operation | Meaning and existing basis |
|---|---|
| `ingest(Message)` | Normal subsequent-turn path for an R1-supported profile; add only new input to resident KV, without generation |
| `ingest_messages(&[Message])` | Nonempty supported batch of new messages; one assistant generation prefix after the batch, one ingestion recovery outcome |
| `replace_messages(&[Message])` | Explicit initialization, history edit, recovery, or unsupported-template fallback; full render + reset + prefill, with rebuilding cost documented |
| `complete(&GenerateConfig)` | Collect one existing generation call into `TurnResult` |
| `generate_into(&GenerateConfig, &mut ModalitySink)` | Existing push generation returning `GenerateSummary` |
| Advanced raw prefill | Preserve `append_text`, `append_tokens`, and `append_embeddings` semantics, with no implicit chat envelope |

The raw operations can retain their current names. Media preprocessing helpers,
custom token routing, and full-template APIs stay available where they provide
distinct functionality. Do not deprecate them merely to reduce method count.

P0 must specify a rendering support matrix for each template/profile, covering
first-turn BOS, system/developer messages, repeated turns, assistant prefixes,
assistant continuation, tools, and media boundaries. Freeze a nonempty set of
initial warm-chat profiles and target backends; those are mandatory R1/P1 gates,
not future F8 promises. `ingest` rejects unsupported combinations before mutation
and points callers to explicit replacement where supported. Expose profile
support before a caller starts its chat loop; exact discovery API is fixed in
P0. Never switch from append to reset automatically on the successful path.
Additional profiles may follow in F8. Arbitrary Jinja templates are not assumed
to support incremental rendering.

**Known R1 regression:** the ordinary text decode loop stops on a sampled EOS or
explicit stop token before emitting, counting, recording, or forwarding it.
Successful ordinary text iterations do forward their emitted tokens; the missing
terminal token is itself not emitted. The speculative path also excludes stop
tokens from the retained sequence, including rewinding a verified stop token.
The next single-message `append_user_message` render does not supply the previous
assistant message's closing envelope. Add a two-turn fixture reproducing the
missing terminator for a profile whose template requires it.

Do not generalize this into "exactly one missing token" for every exit/model:
grammar, arbitrary stop IDs, cancellation, audio transitions, and speculative
rewind need their own evidence. Track emitted, committed, and pending boundary
tokens separately. R1 reconciles the profile's closing envelope and next prompt
exactly once, without re-tokenizing collected assistant text or replaying KV.
It must also screen repeated BOS/default-system preambles from isolated-message
renders, assistant prefixes, whitespace/BPE merges across append boundaries, and
media expansion. Require differential full-render/incremental token fixtures;
do not prescribe hardcoded delimiter IDs as the implementation. Any new renderer
belongs explicitly to R1, with model/tokenizer/template identity checks.

**Context capacity:** Part I high-level chat requires `n_keep = 0` (the existing
default); reject chat preparation with nonzero `n_keep` before mutation. Raw and
legacy operations retain their existing sliding behavior. The caller prunes
whole messages and explicitly replaces context after overflow; account for
media-expanded positions and space for output and turn boundaries. Do not try to
map arbitrary evicted KV spans to message indices. Future F8 sliding support must
specify boundary mapping, logits, drafter/history updates, and cursor validity.
After raw mutation invalidates chat bookkeeping, require replacement before
resuming high-level ingestion; do not silently reconstruct history.

For `cera-client` conversions, preserve role, tool calls and IDs, participant
name, refusal, and reasoning metadata, or return a conversion error. Use
`TryFrom` where the new message cannot represent all fields; never silently drop
them. Lossless helpers may use `From`. Keep integration optional so basic local
inference does not gain a mandatory HTTP dependency. Trait conversions must
live in a crate owning one of the types; a separate bridge crate uses conversion
functions or its own wrapper types to respect Rust's orphan rules.
Representing tool data does not imply implementing the F3 tool lifecycle.

### 4.2.1 Turn phases and batch ingestion

P0.1 prototypes `SessionPhase` and an observable phase query. The following
contract applies to chat-prepared execution; raw execution remains explicit.
Calls are serialized by Rust borrowing or the binding's operation lock.

| Phase | Meaning / permitted next action |
|---|---|
| `Idle` | Fresh or successfully reset; initialize with replacement or a profile-supported first batch |
| `PromptReady` | One complete input batch and one assistant prefix prepared; call `complete` or `generate_into`; reject a second ingestion before mutation |
| `TurnComplete` | A profile-recognized assistant end was observed; any required uncommitted closing tokens are recorded; ingest the next batch exactly once |
| `Interrupted` | Output ended without a proven turn boundary; reject new ingestion and implicit repeated completion; Part I requires explicit replacement |
| `RawContext` | Raw mutation invalidated chat bookkeeping; preserve advanced raw generation/chaining, but require replacement before chat ingestion |
| `Unusable` | Recovery could not establish sound execution state; only successful reset or recreation enables inference again |

`ingest` is a single-message convenience over the same batch contract. Validate
the entire batch's roles, media, template support, and capacity before mutation
where determinable; execution failures use one §4.3 outcome for the whole batch.
No assistant prefix is injected between messages. This permits only combinations
in the profile matrix; it does not implement F3's tool lifecycle. Replacement
validates a supplied history and prepares exactly one next assistant prefix,
entering `PromptReady` on success; invalid history is rejected before reset.

For chat, `complete`/`generate_into` require `PromptReady`. After a recognized
end they enter `TurnComplete`; budget exhaustion, context-full, nonterminal stop,
grammar dead end, or cancellation after progress enters `Interrupted`. A proven
no-progress early exit (such as `max_tokens = 0` or cancellation before decode)
keeps `PromptReady`; a validation failure preserves the prior phase. Errors
without established state validity mark `Unusable`. P0/R1 must cover all ordinary,
speculative, grammar, and advertised audio exits without inferring validity from
an `Ok` result or zero emitted tokens alone. Phase changes do not rewrite finish
reasons, counters, or callback semantics.

Double completion after a finished/interrupted chat turn is rejected consistently
across sampling modes. Today ordinary greedy decode clears logits after emitting
tokens, while stochastic decode retains them; the new chat phase contract must
not select between `EmptyInput` and implicit continuation based on temperature.
Raw `generate` and raw-context collection adapters keep existing chaining rules.
Legacy/raw mutation of a chat-prepared session invalidates its chat cursor unless
R1 proves and implements the same bookkeeping transition. Explicit close/continue
operations for interrupted chat require a separate validated contract; clearing
cancellation alone does not close a turn.

### 4.3 Ingestion failure and recovery: prerequisite R0

Proposed new ingestion methods return `Result<IngestSummary, IngestError>`.
`IngestSummary` reports consumed input and before/after positions; Part I chat
does not evict context. Any future sliding extension must add explicit eviction
reporting. `IngestError` preserves the primary cause, recovery outcome,
and any secondary recovery failure. Its errors belong to the new API taxonomy;
do not extend the exhaustive legacy enum as a compatibility shortcut.

| Recovery outcome | Caller-visible guarantee |
|---|---|
| `Unchanged` | Rejected before execution-state mutation; prior context remains usable subject to its pre-existing state and cancellation flag |
| `Restored` | All execution state returns to the pre-call checkpoint, including backend state and session metadata; no successful reset is being disguised as restoration |
| `Reset` | Restoration was unavailable and a verified execution-state reset succeeded; position is zero and prior context is gone; caller must provide context again and explicitly clear any pending prefill cancellation |
| `Unusable` | Neither restoration nor a complete reset was established; reject ingestion/generation until a successful explicit reset or session recreation |

The default recovery order is validation before mutation, then complete
restoration when supported, then a full reset when that backend can establish
it. Otherwise mark unusable. A reset failure never reports `Reset`. Preserve
the original error alongside a reset error. Unsupported input rejected before
mutation must not poison a compressed session.

An automatic recovery reset clears execution state without clearing the external
cancellation latch, including a new request arriving during recovery. R0 must
separate this operation from caller-requested `Session::reset`, which keeps its
existing flag-clearing behavior. A read/reset/write sequence that can lose a
concurrent cancel request is not sufficient. This is an intentional R0 failure-
path correction, not a change to successful decode cleanup.

Restoration requires a backend-aware checkpoint and an account of every mutable
component: attention and convolution state, device counters, current position
and its observer mirror, last logits, draft history/drafter state, prefill
statistics, and any changed sampler or rendering state, including phase, profile
identity, and pending boundary tokens. `Restored` restores that bookkeeping;
`Reset` clears it and enters `Idle`; `Unusable` is enforced on both surfaces.
Do not overwrite an external cancellation request while restoring internal
execution state.
Recording the old position alone is not a checkpoint.

Tail rewind is allowed only when the backend supports it, its target is still
valid, required convolution snapshots exist, and no destructive context shift
or replacement has invalidated the prefix. A failed `replace_messages` after
reset cannot report restoration of the previous context without a full
checkpoint. Part I does not require expensive full-cache copying to provide
atomic replacement; reporting `Reset` or `Unusable` honestly is sufficient.

Add an additive fallible rewind entry point and a pre-mutation backend capability
check, covering compression, device state, target bounds, and conv checkpoints.
Revalidate target/checkpoint availability when recovery actually runs; an initial
capability check cannot promise that a ring-buffer checkpoint survives prefill.
Check all components before touching any of them, under the applicable operation
lock. Unsupported rewind returns an error rather than asserting or silently
clearing to zero. Keep existing public `truncate_to`/`Model::truncate_kv`
signatures during the compatibility window; recovery uses the checked path.
Inventory speculative and other rewind callers so legacy signatures cannot
bypass necessary correctness guards or accidentally claim successful recovery.

Without proven device-convolution rewind, affected hybrid GPU/Metal configurations
must use verified `Reset` or `Unusable` after mutation. Compressed caches likewise
cannot use unsupported tail rewind. Do not label all GPU rewinds impossible:
capability depends on the actual backend/layer/cache/checkpoint combination.
Document these common reset cases and re-supply context via `replace_messages`;
test backend reset itself rather than inferring it from a zero session counter.

Recovery must not impose a full KV snapshot, device-to-host cache readback, or
copy of accumulated token/message history before each successful warm ingest.
Use bounded rollback metadata and backend-supported rewind where sufficient.
If complete restoration would require context-sized copying, retain the explicit
`Reset`/`Unusable` failure outcomes instead of adding that copying to the default
successful path. Any future opt-in transactional snapshot mode has a separate
cost contract. Include checkpoint overhead in §8.1's memory and latency gates.

R0 must fix the existing unconditional rollback path in `append_user_message`
and keep legacy error variants/signatures during migration. Legacy callers still
need a documented reset/recreate rule on failure, and unusable state must be
enforced across both API surfaces. The new methods expose richer recovery
information. R0's corrected failure semantics and tests are recorded separately
from Part I's successful-call parity evidence.

### 4.4 Completion, streaming, cancellation, and retry

`TurnResult` contains collected text, emitted token IDs, and the same
`GenerateSummary` returned by `generate_into`. It has no second competing set
of token/timing counters. Text collection must use the existing tokenizer and
detokenization behavior and preserve special-token handling. Audio frames remain
available through the sink; a text collector must not silently discard audio
output. P0 must choose an explicit rejection or a defined audio-bearing result
before enabling collection on an audio-output session.

`complete` and `generate_into` each invoke the current generation implementation
once. They preserve ordinary/speculative selection, flush thresholds, grammar
behavior, stop tokens, `ignore_eos`, finish reasons, and summary accounting.
Keep the existing sink callback signature; do not change `on_done(FinishReason)`
into a new summary event as part of this reshape.
Distinguish core `cera::ModalitySink` token-ID batches from the foreign sink's
UTF-8 text/thought chunks and audio frames. Preserve the adapter's decoding,
buffering, and exactly-once foreign terminal notification behavior.

Cancellation after partial decode is not transaction rollback. Preserve emitted
output and the existing Result/finish-reason behavior, and document whether the
returned state can continue. R0's ingestion guarantees do not establish decode
continuation. P0 must verify continuation state for ordinary, speculative, and
audio paths; where it is not established, require reset and replay. Clearing a
cancellation flag does not itself certify resumability. Preserve the current
distinction: `Session::generate` installs `CancelGuard`, which clears the flag on
exit, including the speculative path; caller-requested `Session::reset` clears
it too. Cancelled prefill has no corresponding decode guard. After ingestion
recovery preserves that pending request, the caller must explicitly clear it or
request a reset before retrying. Do not impose a new manual-clear requirement
after a decode call whose existing guard already cleared the flag.

Never automatically re-ingest after a decode error: that duplicates the prompt.
After an ingestion error, `Unchanged`/`Restored` allow retry subject to cancellation
and other preconditions; `Reset` requires supplying context again; `Unusable`
requires successful reset or recreation. After partial generation, the caller
decides whether to keep the partial assistant answer and replace context, or use
a documented continuation path. No blind retry helper in Part I.

Combined message-plus-generation helpers may wrap these operations, following
the existing FFI methods. Bindings must hold the operation lock continuously
across prefill and decode, preserve nonblocking cancellation, and keep terminal
notification count/order consistent even on validation or ingestion errors.

The existing FFI session already caches cancellation and position atomics outside
its mutex; no new foreign cancellation object is required. Preserve their **handle
identity** for the lifetime of the session: reset and every recovery path store
through the same shared atomics rather than replacing them. Handles obtained
before reset must still cancel active inference and observe position afterward.
Keep `cancel`, `clear_cancel`, and `position` nonblocking during inference and
callbacks; do not route them through the session lock. Calling lock-taking
methods from text/thought/audio callbacks delivered under the operation lock
can deadlock. Preserve the foreign terminal callback exception: both FFI
streaming helpers release the session lock before calling foreign `on_done`,
including ingestion/validation errors. Publish the final phase before unlocking
and preserve reentrant terminal calls; core sink callbacks and foreign terminal
notification have different lock contexts.

### 4.5 Configuration inventory

These are initial migration homes, with no fields removed. Preserve existing
types, defaults, validation, and precedence until P0 verifies each mapping.
Keep old config types and convert at the boundary during the shim window.
`GenerateConfig` is a complete options object, not an implicit partial patch;
callers can start from model/manifest defaults as in §1.1.

| Current field | Proposed home |
|---|---|
| `EngineConfig::context_size` | `LoadConfig::context_size` — allocation capacity, distinct from session cap |
| `EngineConfig::backend` | `LoadConfig::backend` |
| `EngineConfig::draft_model` | `LoadConfig::draft_model` |
| `EngineConfig::gpu_depthformer` | `LoadConfig::gpu_depthformer` — preserve existing inheritance |
| `EngineConfig::bundle_repo` | `LoadConfig::bundle_repo` — retains `remote` gate |
| `SessionConfig::max_seq_len` | Same session field and cap semantics |
| `SessionConfig::kv_compression` | Same session field and backend configuration restrictions |
| `SessionConfig::n_keep` | Same session field and context-shift behavior |
| `SessionConfig::seed` | Same session field and reset behavior |
| `SessionConfig::ubatch_size` | Same session field; zero retains its existing meaning |
| `SessionConfig::gpu_depthformer` | Same session field |
| `GenerateOpts::max_tokens` | `GenerateConfig::max_tokens` |
| `GenerateOpts::temperature` | `GenerateConfig::temperature` |
| `GenerateOpts::top_p` | `GenerateConfig::top_p` |
| `GenerateOpts::top_k` | `GenerateConfig::top_k` |
| `GenerateOpts::min_p` | `GenerateConfig::min_p` |
| `GenerateOpts::repetition_penalty` | `GenerateConfig::repetition_penalty` |
| `GenerateOpts::stop_tokens` | `GenerateConfig::stop_tokens` |
| `GenerateOpts::ignore_eos` | `GenerateConfig::ignore_eos` — includes grammar exception |
| `GenerateOpts::grammar` | `GenerateConfig::grammar` |
| `GenerateOpts::grammar_trigger_tokens` | `GenerateConfig::grammar_trigger_tokens` |
| `GenerateOpts::flush_every_tokens` | `GenerateConfig::flush_every_tokens` |
| `GenerateOpts::flush_every_ms` | `GenerateConfig::flush_every_ms` |
| `GenerateOpts::spec` | `GenerateConfig::spec`; preserve `SpecDecode::{ngram, k}` |
| `AudioGenerateConfig::max_tokens` | Unchanged audio config field |
| `AudioGenerateConfig::sampler` | Unchanged `SamplerConfig`, including temperature, top_k, top_p, min_p, repetition_penalty, seed |
| `AudioGenerateConfig::audio_temperature` | Unchanged audio config field |
| `AudioGenerateConfig::audio_top_k` | Unchanged audio config field |
| `AudioGenerateConfig::mode` | Unchanged audio mode and interleaved budget semantics |
| `AudioGenerateConfig::gpu_depthformer` | Unchanged audio config field |
| `WhisperTranscribeOpts::language` | Unchanged Whisper option |
| `WhisperTranscribeOpts::translate` | Unchanged Whisper option |
| `WhisperTranscribeOpts::timestamps` | Unchanged Whisper option |
| `WhisperTranscribeOpts::max_tokens` | Unchanged Whisper option, including cap |
| `WhisperTranscribeOpts::temperature` | Unchanged Whisper option |
| `WhisperTranscribeOpts::cancel` | Unchanged cancellation handle semantics |

P0 adds the source/default/mapping evidence beside each row, including manifest
advisory defaults and environment overrides. For example, current depthformer
enablement is inherited across engine/session/environment; an explicit `false`
is not necessarily an override. Do not silently turn it into one. Preserve
constructor-specific metadata precedence and the distinction between load-time
capacity and session limits. Count FFI/parity struct literals and generated
language mirrors; also inventory settings exposed through methods, including
image preprocessing, adapter/drafter attachment, and session generation defaults.

### 4.6 Error taxonomy and compatibility

Introduce an extensible error contract on the new API, with actionable loading,
kind mismatch, input/template, modality, backend, cancellation, and recovery
information. Exact names and foreign payload representations are fixed in P0.
Do not expose Rust-only source chains as the sole way a binding caller can
identify a failure; keep machine-readable categories and useful context.

Keep the legacy `CeraError` and FFI error shape intact during the compatibility
window. Legacy methods use existing variants, with documented loss of detail
where the richer new error cannot be represented. New enum variants or changed
legacy semantics beyond the declared R0/R1 fixes require an explicit compatibility
decision. Adding `#[non_exhaustive]` to an already exhaustive legacy enum is
itself a source break.

Deprecate only genuinely superseded operations, after an equivalent replacement
exists in every supported binding. Preserve old `from_*` constructors, config
structs, return types, and applicable message helpers for at least one released
version after the replacement ships. Adapters may require more than a one-line
delegation; do not budget them from method count alone. Raw operations remain
supported rather than becoming chat-message shims. Remove only in a declared
breaking release after P3's gates pass.

LoRA behavior stays unchanged: attach/swap/remove affects subsequent forwards,
does not recompute old KV rows, and the selected adapter survives reset. A
watermark cannot retroactively adapt cached rows. Any ban, automatic replay, or
new consistency policy belongs in separate engine work.

### 4.7 Leap SDK compatibility

The [compatibility workstream](API_RESHAPE_LEAP_COMPAT.md) defines the Swift and
Kotlin migration target, required LoRA/embedding coverage, package/import and
Flow/SKIE export probes, semantic gaps, and release gates. Use public docs and
published release metadata for references. Stable 0.10.9 is a regression
baseline; newer required capabilities need their own verified artifact/consumer
fixtures and are not silently excluded by that baseline.

The [first C0 export results](API_RESHAPE_LEAP_EXPORTS.md) and
[protocol/native evidence](API_RESHAPE_LEAP_BRIDGE.md) validate selected unchanged
consumers, separate stable/extended KMP/SKIE profiles and the raw Cera CPU bridge
in Swift/JVM. Stable Swift custom conformers require a deliberate version/product
policy; extension defaults do not fulfill newer Objective-C requirements.
Complete package, runtime, platform and warm-KV gates remain open.

The facade owns Leap Conversation history above the execution-only Session,
preserving D7. Normal completed warm turns retain live KV under R1. It translates
nullable Leap options into complete Cera configuration and preserves lifecycle,
stream cancellation, error, and event contracts. LoRA adapter lists/scales,
per-call RNG, embedding adapter isolation, and optional SDK products require
explicit proofs; method-name aliases are insufficient.

C0's unchanged-consumer export prototype can proceed with P0. C1's loading
adapter can proceed after the relevant loading/binding gate; its chat adapter
requires R0/R1 and the relevant P1 surface. Any missing runtime capabilities
are separately reviewed C1 prerequisites. C2 adds dependency-only migrations
and performance checks through the facade to P2's release gates. A dependency
on an F1–F8 capability blocks the corresponding compatibility promise, not
unrelated Cera operations. P3 does not automatically remove the Leap bridge.

## 5. Part II — independently gated capabilities

- **F1 — Pull `GenerationStream`.** Spike the mechanism and measure against the
  push sink before fixing its API. Cover ordinary/speculative paths, batching,
  cancellation, backpressure, early drop, thread ownership, and browser targets.
  A synchronous Rust iterator introduces no mandatory async runtime. The spike
  is not a Part I prerequisite.
- **F2 — Text stop sequences.** Withhold suffixes that could complete a stop
  sequence before sending text to the consumer. Define overlapping matches,
  token/UTF-8 boundaries, end flush, cancellation, and latency cost. Stripping
  already emitted text is impossible. Decide whether hidden stop tokens remain
  in context and how continuation works.
- **F3 — Tool lifecycle.** Build on `crate::tools` and lazy grammar triggers.
  Specify parsing, tool-call IDs, results, multiple calls, and continuation.
  Preserve existing tool APIs during the reshape.
- **F4 — Session persistence.** Implement save/load above existing KV schema
  and snapshot primitives. CPU state uses `InferenceState::snapshot/restore`;
  model-level snapshot methods are backend-specific and can be unimplemented.
  Define compatible model/tokenizer/adapter identities, cache format and
  precision, positions, convolution state, logits, sampler/drafter policy, and
  chat phase/profile/pending-boundary identity.
  Either restore continuation state or explicitly require re-prefill; a KV file
  alone is not a complete resumable session. No transcript is stored by default.
- **F5 — Whisper/VAD/hotword unified loader facades.** Wrap existing implementations
  and options; preserve recurrent VAD state, hotword detector scratch, iterator
  buffers/sample timeline/cooldown and chunk/reset APIs. Define independent stream
  ownership, preserve hotword offsets as detection-window ends with saturating
  pre-roll, and retain Whisper's per-call async cancellation. Add typed builders
  and dynamic handle variants only when their native and WASM representations
  are ready. Preserve one-call LFM2-Audio ASR. This is the first candidate after
  Part I, but does not block it.
- **F6 — Media marker generalization.** Defer named architecture variants until
  a second VL architecture loads. Preserve today's custom raw routing. A new
  `Custom { token ids }` abstraction also needs an explicit mapping/design and
  does not ship automatically as part of renaming methods.
- **F7 — Audio unification.** Keep the separate audio loop/config in Part I.
  Any future merge must cover sequential/interleaved budgets, all sampler fields,
  output collection, GPU depthformer behavior, and cancellation.
- **F8 — Additional incremental template support.** Add profiles or a general
  renderer only with prompt-token, media-position, boundary, and multi-turn
  evidence. R1 supplies the initial mandatory profiles; F8 extends that set.
  Keep unsupported templates on explicit full replacement and document its cost.

## 6. Decisions and gates

| ID | Decision | Status / dependency |
|---|---|---|
| D1 | Failure contract and complete recovery | Contract in §4.3; R0 backend proof and implementation block new ingestion guarantees |
| D2 | Pull mechanism | Open; blocks F1 only |
| D3 | LoRA warm attach | Preserve existing future-forward behavior in Part I |
| D4 | Compatibility window | At least one released version with equivalent replacements; explicit breaking release for removal |
| D5 | Audio unification | Keep separate in Part I; F7 requires its own design |
| D6 | Dynamic handle | Keep Rust `#[non_exhaustive] ModelHandle`; WASM uses object/accessor facade; UniFFI prototype and accessor ownership remain P0 gates |
| D7 | Session ownership | Locked: execution cursor, no retained message history or retain_history option |
| D8 | Rendering compatibility | P0 freezes initial warm-chat profiles; R1 proves them before Part I chat ships; explicit replacement retained; additional profiles are F8 |
| D9 | Existing encoder/classifier API coverage | Inventory and preserve in P0; blocks deprecation/removal of those entry points until an equivalent facade exists |
| D10 | Collected audio result | Open; blocks enabling complete on audio-output sessions, not text collection or existing audio streaming |
| D11 | Client message conversion | Lossless From or fallible TryFrom; optional dependency placement fixed in P0 |
| D12 | Warm-session performance | Live KV reuse is mandatory for initial profiles; P0 records numeric per-target budgets; §8.1 gates P1 and P2 |
| D13 | Turn phases and context window | §4.2.1 gates chat calls; Part I requires n_keep = 0 and caller-managed replacement at capacity; raw chaining remains supported |

The core keeps the current synchronous push API. Existing foreign async APIs
remain supported. Unresolved Part II decisions do not become Part I exit gates.

## 7. Sequencing

**P0 — Scoped design gates, with an independent loading strand.** Resolve only
decisions needed by the operations being shipped; F8's additional profiles and
D2 do not block Part I. The following gates produce executable evidence, not
just more design documents.

**P0-L — Loading contracts and prototypes.** Complete source/constructor payload,
config/default/feature, model-kind, and retained-capability inventories. Compile
typed/dynamic loading in Rust and the affected bindings, including WASM accessors
and UniFFI ownership. Record backend execution-state ownership and existing
restrictions on sharing a model across sessions; do not promise session isolation
merely because the API returns distinct objects.
*Exit:* loading signatures, ownership, mismatch behavior, and compatibility have
executable consumers. Additive loading may enter P1 after this gate, independently
of chat's P0.1/P0.2, R0/R1, and benchmark budgets. Its own binding checks still apply.

**P0.1 — Core chat contract.** Compile Rust prototypes for §1, batch ingestion,
phase transitions, and new errors. Inventory raw and chat methods and classify
each first-party entry point as warm, replacement, or caller-managed. Specify R0's
checked recovery matrix and R1's phase, boundary, `n_keep`, and cancellation rules.
Capture legacy prompt/token fixtures and operation/callback traces. Freeze a
nonempty named model/tokenizer/template/backend matrix for initial warm chat,
including the advertised native and browser targets; prioritize a small set of
proven profiles rather than claiming arbitrary templates.
*Exit:* core ownership and transitions compile; every existing chat/raw capability
has a migration or retention row; known R0/R1 fixes have explicit regression cases.

**P0.2 — Chat bindings and performance baseline.** Compile Swift/Kotlin chat
prototypes and verify the actual WASM/UniFFI representations, phase/error
observation, batch input, callbacks, and nonblocking cancellation. Capture
retained-state and replacement baselines using §8.1's controlled protocol; freeze
numeric per-target budgets, repetitions, and statistics before evaluating new
implementation performance. If a corrected R1 reference is needed, freeze the
comparison protocol against that independently validated reference before measuring
the Part I wrappers; keep the legacy defect evidence separately.
*Exit:* §1's advertised binding workflows compile and the frozen chat matrix has
concrete performance fixtures and acceptance budgets. A missing number or a
target removed after observing a regression does not pass. R0/R1 implementation
may overlap this work once P0.1 fixes the relevant contract, but chat release
requires both P0.1 and P0.2 as well as R0/R1.

**R0 — Correct existing ingestion failures.** Implement §4.3's backend-aware
recovery, fallible rewind/capability checks, and unusable-state enforcement,
including legacy entry points. Preserve legacy error shapes and record intentional
failure-behavior changes. Add fault
injection tests independently of API renaming.
*Exit:* recovery outcomes are proven for the supported backend/cache matrix;
unsupported restoration reports reset/unusable honestly; successful-call
fixtures remain unchanged. Capture this corrected baseline for Part I parity.

**R1 — Prove initial warm-chat support.** Verify or implement the initial
profiles using retained context, correct prompt envelopes, pending-token and
stop-boundary handling, continuous RNG progression with the existing per-call
repetition-history reset, and defined drafter continuity. New rendering
or execution behavior is reviewed here, separately from wrapper changes.
Record the missing-turn-terminator fixture, phase transitions, batch-prefix
behavior, and declared corrections to existing warm FFI envelopes.
*Exit:* §1's multi-turn workflows work for the frozen initial matrix without
normal-turn reset, history replay, or full-cache checkpoint copying. Establish
an independently validated retained-state reference for wrapper parity where
the legacy high-level helper cannot provide one. Pass the differential prompt
fixtures and at least ten successive completed turns without boundary drift.
Existing supported raw paths remain available; preserve their reference
performance. R1 and R0 may proceed
independently where their state contracts permit, and both gate new warm-chat
ingestion.

**P1 — Additive core and binding prototypes.** Add the selected sources, loader,
message adapters, collection wrapper, and new errors alongside the old surface.
Warm-chat ingestion requires R0 and R1 first; unrelated additive loading work
may proceed after P0-L independently of the chat gates. Keep old configs and APIs
available. Surface-diff and compile-test foreign consumers now, before migrating
or deprecating them.
*Exit:* all applicable CI gates pass; new workflows are exercised; existing
oracle assertions remain unchanged in intent; downstream compatibility fixtures
and required feature/target builds pass. Initial warm-chat profiles also pass
§8.1's deterministic reuse checks and numeric performance budgets; a replacement-
only chat surface does not satisfy this phase.

**P2 — First-party migration.** Migrate CLI, FFI, WASM, parity tools, examples,
and integration callers in bounded units. Include Swift, Kotlin, Python, Dart,
and TypeScript consumers. Deprecate only redundant operations with equivalent
replacements. Keep original oracle evidence; changing imports or adapter setup
must not weaken assertions.
Treat a caller's switch from repeated replacement to warm continuation as an
explicit workflow migration, documenting RNG continuity, per-generation
repetition-history reset, and any output
differences from changed execution policy. Retain the existing replacement path
during the compatibility window. Declare R1's named fixes to existing warm methods
in release notes, with before/after fixtures; no unrelated legacy behavior changes
are authorized by migration. Existing FFI users can be affected without opting
into a new method name.
*Exit:* no first-party use of APIs selected for deprecation; all retained raw,
encoder, audio, and tool paths still work; equivalent workflows demonstrate
token parity against the corrected reference and pass binding smoke tests.
The migrated Rust and binding workflows pass §8.1, including multi-turn latency
and allocation checks through the actual public entry points.

**P3 — Removal gate.** Wait at least one released version after replacements and
deprecations ship. Recheck downstream bindings, examples, active branches, and
release notes. Remove only superseded APIs in a declared breaking release.
Unmigrated encoder or other distinct capabilities block their own removal.

**Part II delivery:** each item on its own branch with its own design and tests.
It can proceed independently once its actual prerequisites are satisfied;
there is no blanket wait for P3.

**Leap compatibility delivery:** C0 runs alongside P0, C1 implements the adapters
and their named prerequisites, and C2 validates replacement apps alongside P2.
Track these in the [handoff](API_RESHAPE_HANDOFF.md) and
[compatibility matrix](API_RESHAPE_LEAP_COMPAT.md). Do not mark the full SDK
replacement complete while required Swift/Kotlin LoRA or embeddings are missing.

## 8. Verification

- **Equivalent-call parity:** compare prompt token IDs, media expansion/KV
  positions, generated IDs, finish reasons, and counter semantics under the same
  model, backend, precision, seed, prefill chunking, speculative configuration,
  and effective defaults. Measure timings; do not require elapsed milliseconds
  to be equal. Speculative-vs-sequential or CPU-vs-GPU equality is not the claim.
  Include at least one model per supported family and dedicated image/audio
  workflows; encoder paths need hidden-state/classification parity instead of
  generated-token tests. Use the same retained-state policy, continuous RNG
  progression, and per-generation repetition-history reset for warm-wrapper
  parity, including a non-unit repetition penalty to detect accidental history
  leakage across calls. Legacy reset-per-turn versus warm
  execution is a separate workflow comparison, with differences documented.
  Named R1 envelope corrections use the corrected retained-state oracle; keep
  original malformed-prompt fixtures as regressions, not as required new output.
- **Multi-turn rendering:** system-plus-user initialization, repeated turns,
  assistant continuation, history-dependent templates, tool metadata, image
  markers, and model-canonical media order. Verify unsupported incremental
  profiles reject before mutation. Compare full-history and incremental paths
  only where the support matrix promises equivalence.
  For each initial profile, compare a monolithic full Jinja render/tokenization
  with incremental construction at every input/assistant boundary using fixed
  transcript fixtures. Include missing end markers, repeated BOS/default-system
  preambles, whitespace/BPE joins, empty answers, multiple messages in one batch,
  and at least ten turns. Any mismatch in a promised-equivalent fixture fails R1.
  Use controlled assistant token fixtures for this comparison: arbitrary emitted
  token IDs need not equal re-encoding their decoded text. Separately prove exact
  retention of actual generated IDs and correct appended boundary tokens against
  R1's token-level reference, without retokenizing the response to force parity.
- **Turn phases:** test double ingestion/completion, all finish reasons, zero-
  budget and pre-decode cancellation no-ops, phase observation after errors,
  partial batch failure, invalid/nonzero-n_keep chat preparation, raw/chat mixing,
  and pruned-history replacement. Cover greedy, stochastic, speculative, grammar,
  and advertised audio paths. Restore/reset phase and pending boundaries with
  execution state; an `Ok` result alone must not imply a complete assistant turn.
- **Recovery fault injection:** failures before mutation, after partial prefill,
  at convolution checkpoint boundaries (advances around 63/64/65 with actual
  history availability checked), after context eviction, after explicit
  replacement reset, and on compressed caches. Cover CPU and supported GPU
  backends, reset failure, position-zero checkpoints, missing logits, and
  repeated recovery attempts. Assert state contents/continuation where promised,
  observer positions, counters, draft state, and enforced unusability as well as
  the reported outcome. Verify pending and concurrent cancellation survives
  automatic recovery reset, while explicit reset and decode-guard cleanup retain
  their documented behavior. Conversation depth alone is not a checkpoint test.
  Hold cancellation/position handles across explicit reset and each recovery
  outcome, then verify they still control and observe the same session. Rewind
  refusal must leave state unchanged; unsupported device conv restoration cannot
  pass by checking only sequence counters.
- **Streaming and cancellation:** token/audio chunk order, flush thresholds,
  terminal callback count/order on success and each failure stage, partial
  output, and cancellation from another thread/foreign caller. Continuation
  requires evidence for the active decode path; otherwise reset/replay is the
  documented recovery. Compare timing-triggered batches with a controlled clock
  or threshold contract, not identical uncontrolled wall-clock traces.
  Exercise reentrant foreign `on_done` after success and validation/ingestion/
  decode failure, verifying final phase publication and lock release before
  notification without moving in-generation callbacks outside their contract.
- **Existing oracles:** preserve TurboQuant CPU/GPU/Metal and decode/prefill
  correctness assertions. R0 adds regression coverage without changing expected
  successful inference results. API setup migration is allowed; weakening an
  oracle to accept changed output is not. Add separate R1 before/after prompt
  correction fixtures; they do not replace backend numerical oracles.
- **Extensibility test:** an external-crate compile-fail case exhaustively
  matches every current `ModelHandle` variant without a wildcard. It must fail
  specifically because the enum is non-exhaustive, and compile if that
  attribute is removed. Also keep a positive wildcard example. A passing
  wildcard-only doctest cannot detect attribute removal.
- **Compatibility consumers:** compile representative old Rust constructors,
  config literals, and exhaustive legacy error matches against the new library.
  Generate/diff bindings in P1 and P2 and run native/browser smoke tests covering
  loading, multipart bytes, defaults, cancellation, callbacks, and errors.
  Check async and ownership behavior, not just generated method names.
- **Feature and CI gates:** refresh [.github/workflows/ci.yml](../../.github/workflows/ci.yml)
  for each implementation phase and mirror its applicable format, lint, rustdoc,
  target-build, binding, and test commands locally before pushing. Include
  no-default-features and actual wasm32/WebGPU targets: an all-features host
  check does not cover them. Add all-features Clippy on a compatible host;
  native GPU/Metal coverage is incomplete in the baseline lint job. Treat
  unavailable device or environment gates as unverified, not passing.

### 8.1 Warm-session performance gates

Run the following before P1 and again through migrated first-party and binding
entry points in P2. These are required release checks, not optional benchmarks.

**Reference runs:** compare the new warm API to the equivalent existing retained-
session/raw-prefill path, with the same valid turn inputs and execution settings.
If initial profile support required R1, use its independently validated retained-
state reference and report that provenance. Measure repeated full replacement
separately as the fallback comparison. Do not use the slower replacement path
as the sole baseline for a claim of no API overhead.

**Trial isolation and environment:** use a fresh model instance for each
independent trial/arm so model-owned GPU KV, convolution, and prefix-cache state
cannot leak across comparisons. Within a warm multi-turn trial retain the same
model and session throughout; never reload between measured turns. Prepare
matched resident context for each arm and exclude model loading from turn-latency
timing, reporting load time separately when relevant. Deliberately warmed prefix
caches are set up independently per trial under identical policies.

Interleave sequential A/B and B/A trials, report paired ratios as well as absolute
medians/tails, and release each arm's resources before the next to avoid memory
contention. Record host load, competing work, device/OS/backend, power mode,
temperature/throttling, and memory pressure. On shared Android devices verify
thermal readings are current and distinguish cached from live output rather than
using the first reported block blindly. Set exclusion/cooldown rules before runs
and retain excluded-run evidence. The reviews' specific historical slowdown
figures are not baseline measurements for this API; establish current variability.

**Deterministic work checks:** for successful, in-capacity turns, retain existing
KV and backend state. Instrument actual backend forwards to verify that prefill
processes only new message/media input and the profile's specifically identified
uncommitted continuation/boundary tokens. No replay of already committed history,
normal-turn reset, full-cache snapshot/restore, cache readback to host for
checkpointing, or duplication of accumulated history for rollback/rendering is
allowed. Measure ordinary amortized buffer growth separately against the retained-
state reference. This forbids copying
for API/recovery convenience; ordinary model attention still reads existing KV.
Measure interrupted-turn and capacity recovery separately. Part I chat never
silently shifts context; raw/legacy shift comparisons remain a separate workload.

**Metrics and budgets:** record physical prefill positions processed, reused KV
positions, end-to-end time to first visible text token or audio frame, decode
throughput, and host/device peak memory plus per-turn allocation/copy volume.
Latency starts before message rendering, media preprocessing, and eager ingest;
measuring only the later `generate` call hides the work this gate is intended to
catch. Logical input counts such as `prompt_eval_tokens` are not evidence of
physical prefill work when prefix-cache hits are possible. Use instrumentation
or backend traces to distinguish actual computation from restored cache state.

For latency and throughput, P0 records per-target medians, tail-latency criteria,
repeat counts, and numeric regression budgets derived from baseline variability.
Freeze them before comparing the implementation. For memory, budget wrapper and
recovery overhead separately from necessary KV growth for new tokens: there must
be no additional full-context-sized checkpoint or transcript copy on a normal
turn. No gate passes on "looks unchanged" or on an unmeasured hardware target.
Absolute latency may grow with context length because attention reads more KV;
compare matched context lengths, not constant-time claims.

**Fixture matrix:** include first-turn cold prefill and repeated warm turns with
short and long resident histories, constant-size new inputs, and controlled
output lengths. Include at least ten completed turns and interrupted-turn cases
separately. Test within capacity and near/beyond capacity, including Part I chat
rejection/pruned replacement and separate raw/legacy context shifts;
include long initial system prompts, cancellation/recovery followed by retry,
ordinary/speculative decode, each claimed KV precision/compression mode, and
LoRA where supported. For advertised multimodal warm profiles, prior images or
audio must not be encoded again on an unrelated new text turn. Include native
CPU, supported native GPU/Metal, and the browser/WebGPU paths actually claimed
by the initial matrix; unsupported combinations must remain explicitly marked.

**Prefix-cache isolation:** a live warm session must pass reuse checks without
depending on an engine prefix-cache hit. Reset-based fallbacks may benefit from
prefix caching, but report cache-enabled/disabled and warm/cold-cache runs
separately and preserve identical settings within each comparison. Account for
lookup/restore/snapshot costs and existing adapter/cache compatibility gates.
A prefix-cache hit does not establish that the session retained its live KV.

## 9. Documentation delivery

Ship the operation semantics, template support matrix, recovery table, source
and config precedence, advanced raw-input guide, and old-to-new migration table
with each API phase. Lead chat examples with initialization once followed by
incremental ingestion and phase checks before the next turn. Document batch
input, interrupted-turn replacement, caller-managed message windows at capacity,
handle identity and callback reentrancy, initial warm-profile support, continuous RNG
progression and per-generation repetition-history reset, explicit replacement
and its rebuilding cost, and published performance
evidence for each claimed target. Generate multi-language examples from compiled
consumers
once the prototypes become implementations; do not publish the sketches above
as working snippets before that happens.

Release notes must identify the existing FFI/wrapper entry points affected by R1's
warm-envelope corrections, the concrete before/after token fixture, and resulting
possible output changes for unchanged callers. Distinguish this correctness fix
from a CLI user's explicit switch from reset-per-turn to warm execution.

The portal remains separate. Inspect `feat/cera-website` for prior work before
bootstrapping or selecting dependencies. Starlight, Pagefind, and Expressive
Code remain candidates from the earlier proposal; portal tooling does not gate
the API. Review existing documentation placement and link conventions in P0.
