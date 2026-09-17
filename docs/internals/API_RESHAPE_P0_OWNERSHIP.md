# P0-L retained operations and execution ownership

Updated: 2026-09-08T05:39-0700. Plan 08 is complete as a bounded private
prototype increment. Plan 11 is complete with the anonymous-model cache policy below;
scoped Rust/generated tests, lint/doc and one max-effort review round pass.
Plan 12 companion execution/examples and three clean final reviews are complete
within the scope recorded below. Baseline: `60fc11c25a51`.
This advances the Cera API refactor. It does not implement the Leap runtime
facade or promote a public API. See the [loading audit](API_RESHAPE_P0_LOADING.md)
and [current handoff](API_RESHAPE_HANDOFF.md).

## Retained homes

This inventory covers every public constructor and operation on
[CeraEngine](../../cera/src/engine.rs), including its hidden GGUF accessor and
associated marker constant. Existing public entry points remain supported.
The typed entries below are private test-only forwards, not shipped signatures.
Leaving an operation on the supported legacy surface follows D9 in the
[main plan](API_RESHAPE_PLAN.md#6-decisions-and-gates); it is not approval to
remove that operation later without an equivalent replacement.

| Existing entry point | Typed prototype or retained home | Contract |
|---|---|---|
| `from_path` | `ModelSource::Path` + loader | Local file, manifest and directory resolution; mmap gate |
| `from_bytes` | `ModelSource::Bytes` + loader | Shared byte backing |
| `from_reader` | `ModelSource::Reader` + loader | Consumed reader, one input buffer |
| `from_files` | `ModelSource::Files` + loader | Full multipart path record; mmap gate |
| `from_parts` | `ModelSource::Parts` + loader | Full multipart byte record |
| `from_bundle_id` | `ModelSource::BundleId` + loader | Repository and quantization retained; remote+mmap gate |
| `from_hf`, `from_hf_with_strategy`, `from_hf_url` | `ModelSource::HuggingFace` + loader | Explicit spec/URL, strategy and quantization; remote+mmap gate |
| `new_session` | `GenerativeModel::create_session` | Delegates attachment of encoders, vocoder, drafter and generation defaults |
| `model`, `model_arc` | Same methods on `GenerativeModel` | Borrow lifetime or shared Arc; preserves the raw `Model` trait |
| `tokenizer`, `tokenizer_arc` | Same methods on `GenerativeModel` | Borrow lifetime or shared Arc |
| `manifest`, `metadata`, `config` | Same methods on `GenerativeModel` | Borrow existing records; do not copy or reinterpret them |
| `capabilities` | Same method on `GenerativeModel` | Manifest-derived modality declaration, not an auxiliary-load result |
| `default_generate_opts` | Same method on `GenerativeModel` | Preserve advisory sampling defaults and standard fallback |
| `configure_cache`, `clear_warm_cache`, `clear_cache` | Same methods on `GenerativeModel` | Forward to the backend's prefix cache; do not reset live sessions |
| `transcribe` | Same method on `GenerativeModel` | Preserve the one-call LFM2-Audio path and existing errors |
| `AUDIO_MARKER_CANDIDATES`, `split_tokens_at_marker` | Retain on `CeraEngine` | Existing marker priority and token-split helper stay available |
| `audio_encoder`, `vision_encoder` | Retain on `CeraEngine` | Borrow optional shared typed weights for raw auxiliary computations |
| `has_gpu_vision_encoder`, `has_gpu_audio_encoder`, `has_gpu_audio_decoder` | Retain on `CeraEngine` | Report successfully attached GPU components |
| `vision_encoder_gguf` | Retain on `CeraEngine`, still doc-hidden | Raw GGUF access can succeed while typed projector parsing fails |
| `detect_pii`, `detect_pii_with_lora` | Retain on `CeraEngine` | Fresh caller state for classifier execution; no invented encoder facade |

Associated helpers in the same module remain: `BackendPreference::parse_str`,
`ModelFiles::text` and `ModelBytes::text`. The public free initializer
`engine::init_dspark_drafter` retains its raw draft-loading home and optional
result; this prototype does not replace it. `LoadConfig` aliases the existing
`EngineConfig` in the prototype. The source/configuration inventory remains in
the loading audit. Known encoder, Whisper, VAD and hotword kinds are rejected by
`build_generative`; the existing non-generative loaders remain available.
`ModelHandle` currently holds only the generative prototype variant. Foreign
accessors and future variants still require P0-L evidence.

The [retained facade](../../cera/src/engine/loading_prototype/retained.rs)
adds 13 forwards alongside the existing `create_session`. Its clone shares the
engine, and model/tokenizer Arcs can outlive all typed handles. A raw
`Session::new` made with those two Arcs does not automatically recover the
engine's auxiliary attachments or advisory defaults; callers needing those
must use `create_session` or retain the explicit attachment path.

[Session](../../cera/src/session.rs) remains the home for raw token/embedding
ingestion, generation, cancellation, reset, LoRA attachment/removal, and
per-token hidden-state or pooled embedding extraction. `Model` remains the raw
backend escape hatch. Its defaults can be no-ops or unsupported panics:
preserving access does not make every operation supported on every backend.
In particular, CPU snapshots use `InferenceState::snapshot`; argument-less
`Model::snapshot_state` is implemented by the device backends, not CPU LFM2.

## Execution ownership

The private transcription helper retains primary GGUF backing only when an audio
encoder is attached. Text/VL GPU byte loads release their staging allocation.
Helper prefix caching is disabled, so it cannot retain warm snapshots or disk
entries outside the engine cache controls; its live decode KV is unaffected.

These rows describe inspected implementation, with bounded CPU evidence below.
They are not a device support or throughput matrix. State ownership must be
validated through the actual public and foreign entry points before promotion.

| Backend or component | Mutable execution state | Sharing boundary and remaining evidence |
|---|---|---|
| CPU `LlamaModel`: llama/qwen2/qwen3/granite | Live attention KV and scratch in caller `InferenceState` | Shared weights; actual tiny Llama sessions pass interleaved and parallel controls. Other families and real models remain untested here. Prefix-cache methods and model KV configuration inherit trait no-ops; None/F16 sessions can coexist. |
| CPU `Lfm2Model`: lfm2/lfm2moe | Live attention KV and convolution state in caller `InferenceState`; model owns mutex-protected prefix cache and compression namespace | Tiny dense LFM2 with both block types passes CPU controls. First effective KV tag wins, including TurboQuant seed; conflicting modes require separate model instances. MoE and real-model coverage remain open. |
| CPU `BertModel`: bert/modernbert | Per-call output buffers plus reused scratch in caller `InferenceState`; noncausal encoding | Shared weights do not imply generation support. Keep classifier/raw encoder paths. Raw hidden-state calls truncate overlong input; the Session wrapper instead validates length before dispatch. No encoder execution added by plan 08. |
| `MetalLfm2Model` | Model-owned live Metal KV/conv buffers, shared scratch and per-call `infer_lock` | One live Session per model is enforced by a lifetime lease; another constructor returns Busy before configuration. Drop releases it; reset retains it. Use separate models for simultaneous conversations. Plan22 executes native controls. Native uncompressed KV is f16; None/F16 requests resolve to the same mode. Effective mode is fixed on first configuration; unsupported TurboQuant shapes fall back, supported mode/seed conflicts reject. |
| `GpuLfm2Model` (wgpu) | Model-owned live GPU KV/conv buffers, positions, scratch and per-call `infer_lock` | Same enforced session-lifetime restriction; Plan22 executes native wgpu controls. Native uncompressed KV is f32; None/F16 requests resolve to the same mode. Effective compressed mode is fixed on first configuration. The async WebGPU path has its own binding gates. |
| Session hidden states / LoRA | Session owns reusable prompt-sized extraction scratch and an adapter Arc; CPU forwards use caller state | CPU extraction starts fresh without changing generation state. Invalid attachment leaves the existing adapter installed; reset preserves it; removal changes subsequent extraction. This proves one core adapter, not Leap lists/scales/composition. |
| Metal/wgpu hidden states / LoRA | Model-owned dedicated `HsScratch` KV/conv, guarded routing and adapter activation under `infer_lock` | The inspected native paths reset scratch conv state; wgpu also saves/restores generation position. This supports an isolation design but is not device execution evidence. It does not solve two ongoing generation sessions sharing live GPU KV. |
| CPU vision/audio encoders | Shared weights; encoding allocates per-call output/intermediate buffers | Engine attaches Arcs to sessions. These auxiliary computations are separate from the primary model's live KV. Plan 12 executes synthetic CPU vision weights, PNG ingestion and independent session continuation after parent release and Unix source deletion. Plan 13 adds CPU audio encoding/resampling and retained sessions; see its evidence below. |
| GPU vision/audio encoders | Shared uploaded weights and GPU ops; per-call activation buffers | Separate from the text model's KV. Verify device queue/readback behavior and fallback via real image/PCM consumers before claiming concurrency. |
| CPU audio output | Shared decoder/detokenizer weights; `AudioOutputDecoder` owns depthformer, detokenizer and ISTFT state for a generation call | Plan 13 executes loaded CPU vocoders after parent release and checks per-session resets. State remains scoped to a generation call; continuous audio across separate calls is not established. |
| Metal/wgpu audio output | Shared decoder-owned KV/conv, scratch and counters | `AudioOutputDecoder::new` acquires an exclusive GPU session lease, resets the decoder and releases on drop; a busy decoder falls back to CPU. Metal also locks calls. Direct raw decoder methods do not establish an independent context; keep the lease path. Real-device execution remains open. |
| DSpark | `DSparkSessionDrafter` owns inference state, sync position/tokens and scratch | Engine attachment calls `clone_drafter`; the built-in DSpark implementation creates fresh session state with shared weights. Plan 12 executes synthetic DSpark weights through loaded sessions and concrete session-state controls. This does not establish arbitrary custom Drafter isolation. |
| Whisper | Separate `WhisperModel`/weights; transcription builds call-local encoder output, cross-attention cache and decode state | Plan24 retains upstream per-call cooperative cancellation when an async FFI future is dropped. The FFI options have no explicit cancellation handle; synchronous foreign calls cannot be cancelled. Shared/distinct model consumers and options remain supported. A unified typed loader remains F5; see the [Whisper examples](API_RESHAPE_WHISPER_EXAMPLES.md). |
| Silero VAD | `SileroVad` owns recurrent h/c and trailing audio context; `VadIterator` owns stream segmentation state | Preserve mutable chunk APIs and explicit reset when switching streams. F5 must not treat this as an immutable generative model. |
| Hotword | `HotwordDetector` owns copied weights and mutable log-mel/convolution/window scratch. `HotwordIterator` consumes a detector plus optional VAD and owns buffered audio, sample position, hop/cooldown and peak state | Keep standalone file/bytes/GGUF detector loading, foreign file/bytes detector constructors and foreign iterator `fromFiles`, `processChunk`, `reset`. Foreign mutexes serialize calls; independent audio streams require independent iterators or an explicit reset. The generative prototype classifies `kws` as `Hotword` and rejects it. F5 must preserve stream ownership. Event offsets identify the detecting window's exclusive end; pre-roll subtracts from that offset with saturation, without locating a spoken-word boundary. |

Sources: [CPU Llama](../../cera/src/model/llama.rs),
[CPU LFM2](../../cera/src/model/lfm2.rs), [BERT](../../cera/src/model/bert.rs),
[Model defaults](../../cera/src/model/mod.rs),
[Metal model](../../cera/src/model/metal_lfm2.rs),
[wgpu model](../../cera/src/model/gpu_lfm2.rs),
[CPU vision](../../cera/src/model/vision_encoder.rs),
[CPU audio](../../cera/src/model/audio_encoder.rs),
[GPU vision](../../cera/src/model/vision_encoder_gpu.rs),
[GPU audio encoder](../../cera/src/model/audio_encoder_gpu.rs),
[audio output state and lease](../../cera/src/audio_engine.rs),
[CPU audio decoder](../../cera/src/model/audio_decoder.rs),
[Metal audio decoder](../../cera/src/model/metal_audio_decoder.rs),
[wgpu audio decoder](../../cera/src/model/wgpu_audio_decoder.rs),
[DSpark](../../cera/src/model/dspark.rs),
[Whisper](../../cera/src/model/whisper.rs), [VAD](../../cera/src/vad.rs),
[hotword](../../cera/src/hotword.rs).

The prefix cache is distinct from live conversation KV. CPU LFM2 cache lookup
and insertion apply only to fresh, adapter-free prefills. Exact/full hits are
skipped; a strict prefix can restore caller state before computing the suffix.
Cache clearing/reconfiguration does not release the first-format restriction.
Plan 11 restricts models without a path or explicit raw model ID to warm caching.
The built-in CPU LFM2, wgpu and Metal cache constructors and rebuilds discard
`cache_dir` for an empty raw ID before applying the backend/compression namespace.
This blocks cold reads, writes and clearing; existing anonymous files remain
untouched. Metal no longer substitutes an incomplete embedding-sample hash.
Bytes, readers and byte parts supply no identity. Named path loads and explicit
raw IDs retained their existing namespaces in Plan11. Plan16 now binds named
built-in caches to all loaded backing bytes, including DSpark base and draft;
see the [cache examples](API_RESHAPE_CACHE_EXAMPLES.md). It prevents different
weights at the same path or caller ID from restoring one another's cold state.
The low-level cache and custom GPU sources that do not opt into the new provided
identity hook retain their caller-managed contracts.

Plan22 also preserves independent LFM2-Audio transcription while a conversation
owns the primary GPU context. The engine retains the primary GGUF backing and
lazily loads a private helper model with the same backend preference when its
ordinary Session constructor returns Busy. Existing auxiliary weights are shared;
the helper has its own live KV and no path-derived cold-cache identity. Successful
loads remain cached until engine drop, and helper calls serialize through Session
destruction. This adds model/context memory and first-use setup without reopening
source files or clearing, replaying or copying conversation KV. Native Metal/wgpu
tests and the Dart consumer cover this path; see the
[GPU ownership guide](API_RESHAPE_GPU_SESSION_EXAMPLES.md).

`capabilities()` derives all five modality flags from the manifest inference
type. Optional auxiliary failures can leave those declarations ahead of loaded
components. Preserve the distinction between declared modality flags, successful
encoder accessors, and `Model::supports_hidden_states`/other backend probes.
The new transcription test checks error parity for a tokenizer without the
audio marker; successful ASR remains a separate gate.

## Plan 08 evidence and limits

The [ownership tests](../../cera/src/engine/loading_prototype/tests/ownership.rs)
use deterministic tiny F32 Llama and LFM2 models from a
[new fixture module](../../cera/src/engine/loading_prototype/tests/ownership/fixture.rs).
LFM2 has one convolution and one attention block. Distinct prompts/adapters must
change numerical output by more than 1e-3; parity checks use finite logits/hidden
states with tolerance 1e-5 times one plus the reference magnitude.
Generation controls never run extraction; separate fresh sessions supply
expected hidden states, including separately attached adapter controls. This
prevents identical extraction-induced KV mutation on both sides from passing
the later continuation comparisons.
The same final token with and without prior history must also differ, so loss
of retained state cannot pass solely because token identities match. The older
loading fixture remains unchanged; its repeating weights were too insensitive
for this new negative control.

Five CPU tests cover retained borrows/Arcs/defaults/errors, interleaving with
reset/cancel/extraction, parallel sessions with separate-model controls,
single-adapter isolation and backend-specific KV mode conflicts. Interleaving
covers None/F16 modes for both CPU fixtures. Parallel workers synchronize their
start; no scheduling overlap or throughput is asserted. Greedy generation can
discard last logits, so the helper appends a fixed token after nonzero output to
inspect numerical continuation. Cancellation checks leave zero-token output
untouched. TurboQuant cases freeze effective-mode/seed rejection and continuation
parity; they do not provide comprehensive quantization accuracy evidence.

The sixth historical [disk-cache test](../../cera/src/engine/loading_prototype/tests/ownership/cache.rs)
requires `disk-cache` and runs None/F16 LFM2. It observes actual nonempty cold
files, unchanged cold bytes after warm clearing, numerical strict-prefix
continuation parity, no new cold entries for adapted prefill/extraction,
namespace-scoped deletion, preserved live continuation and a new cache root
after reconfiguration. It does not measure cache-hit counts, warm-tier residency
or speed; those need instrumentation and performance fixtures. Plan 11 changes
this fixture to load its cached model from a real path; it now requires both
`disk-cache` and `mmap`. The table below records the historical Plan 08 run.

Run from the worktree with Homebrew PATH and
`CARGO_TARGET_DIR=/Users/dberrios/development/cera/target`.

| Command | Result |
|---|---|
| `cargo test -p cera --lib --locked --offline --quiet` | 631 passed, five existing ignored |
| `cargo test -p cera --lib engine::loading_prototype::tests::ownership --locked --offline --quiet` | Six passed |
| `cargo test -p cera --no-default-features --lib loading_ --locked --offline --quiet` | 14 passed |
| `cargo test -p cera --no-default-features --features disk-cache --lib engine::loading_prototype::tests::ownership --locked --offline --quiet` | Six passed; disk test needs neither mmap nor parallel |
| `cargo clippy -p cera --features remote --all-targets --locked --offline -- -D warnings` | Passed |
| `cargo clippy -p cera --no-default-features --lib --tests --locked --offline -- -D warnings` | Passed |
| `RUSTDOCFLAGS='-D warnings' cargo doc -p cera --features remote --no-deps --lib --locked --offline` | Passed |

Formatting and all 86 local document links pass. The first max-effort round
found that extraction also ran on the future generation controls, which could
mask identical state corruption, and that the public DSpark initializer was
missing from the helper map. Generation controls now remain untouched, separate
fresh sessions supply hidden-state references, and the initializer has a retained
home. All test and Clippy rows above pass again after the fixes; Rustdoc remains
valid because production code did not change. Two max-effort review rounds
completed with three fresh reviewers each. All three final reviewers returned
NO FINDINGS. No actionable findings remain, and none were skipped.
Full workspace CI, real-model/backend family coverage, auxiliary/draft execution,
GPU/device isolation, generated foreign loaders and performance budgets remain
open. Production generation, Session, KV implementation and public exports did
not change in plan 08.

## Plan 11 anonymous-model persistent-cache policy

Root plan: `devlog/plans/000341-11-anonymous-cache-policy.md` at the repository
root. Increment baseline: `/private/tmp/cera-api-plan11-_4jpa91k`.
No public signatures, inference kernels or live KV representation change.
The cache policy is production behavior shared by the supported legacy API and
the private typed loading prototype.

- [x] Reproduce cross-model cold-cache contamination before the fix.
- [x] Apply one private policy at all nine CPU/wgpu/Metal cache setup/rebuild sites.
- [x] Cover legacy and typed byte, reader and byte-part loaders with CPU fixtures.
- [x] Preserve actual cache-layer warm hits and named cold-cache compatibility.
- [x] Pass default, no-default, disk-only and backend feature lint/doc checks.
- [x] Pass remote library tests and complete one clean max-effort review round.
- [x] Refresh generated consumers and finish documentation audit.

The negative control used models with identical metadata and different F32
weights. Loading a strict prefix cached by the first model changed the second
model's logits (`1.6862283` versus independent `1.6881917`) beyond the numerical
parity tolerance. It failed before the guard and passes after it. The expanded
[anonymous-loader tests](../../cera/src/engine/loading_prototype/tests/ownership/anonymous.rs)
exercise six constructors in None/F16 modes. A second test writes an actual old
anonymous CPU snapshot and confirms it can be read with the public low-level
cache constructor. Every tested anonymous loader then ignores that snapshot,
preserves its bytes through warm/full clearing, creates no new cold directory
after reconfiguration, and retains live numerical continuation after releasing
the parent model. Independent model outputs must be distinct.

The shared [cache-policy tests](../../cera/src/kv_cache.rs) observe actual warm
strict-prefix hits, eviction, clearing and preservation of configured limits.
They also verify named cold files remain readable across constructor versions,
other model namespaces miss, and clearing deletes only the selected namespace.
These tests use CPU/wgpu/Metal namespace strings with None/F16/TurboQuant tags;
they do not execute GPU inference or establish TurboQuant accuracy. The real
path-backed LFM2 cache fixture still checks cold side effects, clearing,
reconfiguration, LoRA bypass and retained continuation in None/F16 modes.

Commands use Homebrew PATH and repository-root `CARGO_TARGET_DIR` as above.

| Command | Result |
|---|---|
| `cargo test -p cera --lib --locked --offline --quiet` | 640 passed, five existing ignored |
| `cargo test -p cera --features remote --lib --locked --offline --quiet` | 674 passed, five existing ignored; loopback listener escalation required |
| `cargo test -p cera --no-default-features --lib --locked --offline --quiet` | 563 passed |
| `cargo test -p cera --no-default-features --features disk-cache --lib --locked --offline --quiet` | 569 passed; anonymous and cache-policy tests need no mmap or parallel |
| `cargo clippy -p cera --features remote,gpu,metal --all-targets --locked --offline -- -D warnings` | Passed; all changed backend paths compile |
| `cargo clippy -p cera --no-default-features --lib --tests --locked --offline -- -D warnings` | Passed |
| `RUSTDOCFLAGS='-D warnings' cargo doc -p cera --features remote,gpu,metal --no-deps --lib --locked --offline` | Passed |
| `cargo fmt --all --check`; `git diff --check` | Passed |

The dependency `block 0.1.6` emits Cargo's existing future-incompatibility notice;
the scoped warning-denied Clippy and Rustdoc gates pass. Device execution, full
workspace CI and numeric latency/throughput budgets remain open. Warm hit
observations prove reuse at the cache layer, not a measured speedup. Persistent
caching for anonymous weights would require a later stable identity design;
there is no eager whole-model hashing pass in this increment.

One max-effort review round completed with three fresh reviewers, all NO FINDINGS.
No fixes or skipped findings; no actionable findings remain. The remote suite
initially failed nine fixture tests because the sandbox denied local listeners;
the escalated offline rerun passes. Fresh generated consumer validation passes
in `tests/api_loading/build/run-d4jn9b_b/results.json`: 21 command expectations,
35 cases (Swift/Kotlin 12 each, Node 11), matching `[0, 1, 0]` generation at
position 5 after parent release. Ten harness tests pass. All 373 current source
hashes, 11 generated/native/WASM artifact hashes and four consumer artifact
hashes match, including after native/WASM warning-denied Clippy and Rustdoc.
The consumer fixture is Llama: these are core loading regression checks;
LFM2 persistent-cache behavior is proved by the Rust fixtures above. Earlier
reports remain evidence for their own snapshots. Exact mirror commands are in
the [binding audit](API_RESHAPE_P0_BINDINGS.md#plan-11-cache-policy-core-refresh).
Plan 11 is complete within this scope; all broader API and Leap gates stay open.

## Plan 12 companion execution

The [loading audit](API_RESHAPE_P0_LOADING.md#plan-12-executable-vision-and-dspark-companions)
now supplements the original source inventory with complete synthetic vision
and DSpark weights. The tests execute PNG ingestion, independent encoder output
and text continuation through an LFM2 session; two sessions interleave and reset
without changing the other's numerical continuation. Draft tests distinguish
loaded sidecars by actual proposals, observe automatic session drafting and
compare target output with ordinary greedy decoding. Concrete DSpark session
instances sharing weights have different hidden states for equal-length prompts
ending in the same token, and match independent state controls.

Byte/file/manifest/directory runtime cases retain sessions after dropping all
model handles. Unix cases also delete the sources first; other platforms retain
a directory guard until all mappings drop. Validation is on macOS CPU, with
feature-aware backend compilation. Audio companions, GPU execution/ownership,
custom drafters, real-model quality and performance budgets remain open.


## Plan 13 audio ownership evidence

Complete with scoped validation and three clean final max-effort reviews.

The [audio loading tests](../../cera/src/engine/loading_prototype/tests/auxiliary/audio.rs)
use complete synthetic CPU weights, with six executable tests and one manual
fixture-export helper. Three memory and nine filesystem engines cover legacy,
typed and dynamic loading. Filesystem runtime cases unlink their sources first
on Unix; other platforms retain the source directory until mappings drop.

Input sessions encode real PCM after parent release. Independent primary sessions
consume embeddings from separately loaded encoder weights; token continuation
logits and positions match. The input fixture distinguishes same-length silence,
and the 8 kHz case is compared with explicit resampling to 16 kHz. Resetting one
session leaves the other's live continuation intact. Output sessions capture
nonempty finite PCM, compare independent explicit-attachment controls and retain
weights after model release. Separate detokenizer states distinguish prior-frame
histories when current codes and history lengths match, and reset independently.

A one-code decoder vocabulary makes the session's existing audio sampler
deterministic. This proves loading/attachment and CPU execution, not realistic
speech, code-sampling quality or streaming continuity between generate calls.
The output decoder, detokenizer and ISTFT state are recreated per generation
call. GPU leases/fallback and foreign audio consumers still need execution.
See the [loading audit](API_RESHAPE_P0_LOADING.md#plan-13-executable-audio-companions)
for feature checks, negative controls and final review status.

## Plan 14 remote ownership checks

The [remote examples](API_RESHAPE_REMOTE_EXAMPLES.md) create sessions from
downloaded companions, release their model parents and execute vision, audio
and draft work. Independent CPU controls check selected weights and continuation.
Cache files remain present during these tests; local source-unlink ownership
evidence remains separately scoped to Plans 12/13. Device sharing and performance
remain unverified. Plan 14 validation and its three-reviewer max-effort round
pass; exact scope and commands are tracked in the handoff.

## Plan 16 named cache identity and backing ownership

Four executable CPU contracts cover late-FFN-only same-path replacement, unchanged
cold reuse, old path-only files and concurrent session creation during the first
hash. Work counters prove actual suffix reuse and no hashing on append/generate.
CPU hashes before taking cache/tag locks and then observes the current compression
tag. Anonymous loads remain warm-only.

GPU sources enumerate complete ordered backing files through the provided
`GpuWeightSource::cache_identity_sources()` hook. Metal retains the exact parsed
Arc mapping rather than reopening its path. A direct Metal execution test matches
parsed weights A after that path is replaced with B. This closes the tested backing
ownership gap; shared GPU conversation state and the full device matrix remain open.

The [runnable guide](API_RESHAPE_CACHE_EXAMPLES.md) contains exact commands and
cost scope. Completed validation/review and source provenance are in the
[current handoff](API_RESHAPE_HANDOFF.md#completed-plan-16-named-cache-identities).

## Plan21 standalone Whisper evidence

Historical evidence below predates the Plan24 rebase. Upstream 0.5.5 adds
cooperative cancellation for running async FFI transcription, superseding the
queued-work-only limitation in this completion record.

Completed 2026-09-10T10:12-0700. The [Whisper guide](API_RESHAPE_WHISPER_EXAMPLES.md) records
the public native surface and runnable consumers. Four mandatory Rust FFI tests
exercise retained file/byte weights, nonempty decoding, defaults/errors and
shared/distinct async calls. Swift/Kotlin each execute 11 equivalent cases with
independent a/b outputs and verified staged native-library identity. Each call
allocates its own Whisper encoder output, cross-attention cache and decode state.
No Session KV or GPU execution-context change is included.

Queued cancellation uses the existing abort guard; already-running work still
continues. Kotlin construction/use stays within one IO context for cleanup during
cancellation. The synthetic fixture deliberately ignores acoustic content and
is not a quality or performance benchmark. F5 unified loading stays open.
Two max-effort rounds are complete, final three reviewers clean. Current evidence
is `/private/tmp/cera-api-plan21-_6ef6msg`, consumer run `cera-whisper-d1t2ojx9`.
