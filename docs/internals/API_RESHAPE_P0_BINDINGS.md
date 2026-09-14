# P0-L generated foreign loading contracts

Plan 09 adds isolated binding evidence for the private loading prototype.
The [runner](../../tests/api_loading/run.py) uses a visibility-only Cera source
mirror; no public production exports or released packages change.
See the [loading audit](API_RESHAPE_P0_LOADING.md),
[ownership audit](API_RESHAPE_P0_OWNERSHIP.md) and
[handoff](API_RESHAPE_HANDOFF.md) for the remaining P0-L gates.

## Candidate contract

Plan23 exercises the production Swift/Kotlin Session bindings on native Metal
and wgpu: each consumer checks 51 cases across uncompressed/TurboQuant KV,
ownership through reset/cancel/error/extraction, constructor failure, parent
release, pending async work and CPU sharing. Both consumers reject a build with
Session ownership acquisition bypassed, then pass the full matrix with the
normal build restored. The [runnable examples](API_RESHAPE_GPU_SESSION_EXAMPLES.md)
and [native runner](../../tests/gpu_session_ffi/README.md) document the scope.
This supplies native foreign ownership evidence; Android/iOS, browser ownership
and the candidate loading/migration gates below remain separate.

Plan24 rebases the production bindings onto upstream 0.5.6. Preserve
`FfiHotwordConfig/Score/Event`, detector file/bytes loading and iterator
`fromFiles/processChunk/reset`, standalone Whisper plus cooperative async
cancellation, and the fourth engine async constructor `fromPartsAsync`.
Kotlin propagates coroutine cancellation by freeing the Rust future. The pinned
Swift wrapper lacks a task cancellation handler, so `Task.cancel()` does not
trigger this behavior; foreign cancellation parity remains an open gate.
Regenerate Swift (including SPM), Kotlin, Python, C and Dart together from that
union. These are existing production surfaces; the isolated candidate below
still does not export a public typed hotword/Whisper/VAD loader. Plan25 adds
`kws` to the foreign wrong-kind corpus with the expected
Generative/Hotword/kws payload, matching the core memory/filesystem/remote
classification tests. Its validation status is recorded in the
[handoff](API_RESHAPE_HANDOFF.md#completed-plan25--foreign-hotword-loading).

| Surface | Native Swift/Kotlin | Node WASM | Verified scope |
|---|---|---|---|
| Owned sources | UniFFI `Source` enum: bytes, parts, path, files | `ModelSource.bytes/parts` | Copy caller bytes before mutation; native direct GGUF path |
| Multipart record | All eight `ModelBytes` fields | Same fields | Primary bytes, explicit text inference, stored template and five sampling overrides; auxiliary fields represented but real execution remains open |
| Load configuration | context size, backend, draft path, GPU depthformer | Same fields | Requested 24 and actual capacity 24; CPU backend; draft path and boolean retained; no draft/GPU execution claim |
| Results | Typed generative or dynamic handle | Same | `kind` plus nullable `as_generative`; multiple accessors share core ownership |
| Builder lifetime | One attempt, protected by a mutex | One attempt via mutable loader | Both methods reject reuse after success or failure with `Consumed` |
| Session lifetime | Swift weak release witnesses; Kotlin `close` | Explicit `free` | Append/generate after loader, dynamic handle and both typed parents release |
| Error payloads | Associated errors / `LoadException` subclasses | `Error` with code and fields | Expected/actual kind and architecture preserved; unknown architecture distinct from engine errors |
| Unknown future kind | Synthetic handle, kind string and absent accessor | Same | Same-build representation control only |

The native object stores the source until the first attempt because the core
builder can contain a non-Send Rust reader. Readers remain Rust-only. Both
native and WASM remove the source before parsing backend configuration or
building, so failure also consumes it. This is an explicit candidate foreign
contract, not an automatic consequence of Rust ownership.

The native error's text field is `detail`. Naming it `message` produced a
UniFFI 0.31.2 Kotlin compilation failure because its generated exception also
overrides `Throwable.message`. Swift and Kotlin consumers compile against
unmodified generated files after that field rename. JS uses native `Error.message`
and adds `code`, `expected`, `actual`, and `architecture` when applicable.

The multipart fixture sets `llama.cpp/text-to-text`, includes ignored/invalid
optional auxiliary payloads, retains `probe-template`, and checks temperature
0.37, top-p 0.71, top-k 7, min-p 0.13 and repetition penalty 1.23. It does not
activate template rendering or validate auxiliary loading or draft precedence.
The config draft path is intentionally missing. WASM is compiled without mmap,
so retaining that path does not establish path-based loading on WASM.

## Plan25 hotword contract refresh

The current corpus has 13 cases in each native consumer and 12 in Node. Both
`build` and `buildGenerative` (`build_generative` in JavaScript) must reject an
architecture-only `kws` model with expected kind Generative, actual kind Hotword
and architecture kws. Each request uses unavailable Metal, proving this error
precedes backend construction; either attempt also consumes its loader.
[Swift](../../tests/api_loading/consumers/LoadingProbe.swift),
[Kotlin](../../tests/api_loading/consumers/LoadingProbe.kt) and
[Node](../../tests/api_loading/consumers/loading_probe.cjs) contain runnable examples.
Use the [runner instructions](../../tests/api_loading/README.md) to generate the
isolated bindings, construct all fixtures and execute the consumers together.

This is dispatch evidence using a header with no tensors. Typed hotword loading,
real detector inference, mobile/browser execution and numeric performance remain
open. Full run `tests/api_loading/build/run-6bvxs_7w/results.json` passes all
21 command expectations with matching [0,1,0] generation at position5.
`/private/tmp/cera-api-plan25-3dhx3t3j/control.json` records a temporary isolated
`kws`-to-Vad classification: all three consumers fail specifically at the first
dynamic Hotword payload assertion, then pass the complete matrix after
restoration. Both typed and dynamic paths are covered by the positive matrix.
Native loaded-library traces match the selected rebuilt artifact.

Rebuilding restored source changes native linker output hashes, so the control
records current native/generator hashes before restored execution and verifies
them afterward; source/mirror/probe/configuration/fixture checks also pass.
The initial full-run native hashes describe the pre-control build. Ten harness
tests include false-status maps, non-string case lists and missing `kind-kws`;
`parser-control.json` also reproduces the old parser's false-map acceptance.
Native/WASM Clippy and Rustdoc pass with warnings denied. One max-effort round
with three reviewers ends clean; one tool interruption was resumed. Plan25 is
complete within this scope. Older evidence sections
below retain the counts and hashes of their original snapshots.

## Executable evidence

The final plan 09 run is `tests/api_loading/build/run-n2nckicv/results.json`:
21 command expectations passed, including two specific expected E0004
rejections. Swift and Kotlin each execute 12 named cases; Node executes 11
(native path excluded). Both typed and dynamic entry points are checked for
each error and subsequent reuse. All three consumers append `[0, 1]`, observe
position 2, generate `[0, 1, 0]`, and observe position 5 after releasing the
parent handles. The Rust helper requires exactly one terminal callback.

Wrong-kind fixtures cover BERT, ModernBERT, Whisper and Silero VAD with an
explicit unavailable Metal backend, demonstrating classification before backend
construction. Separate fixtures cover unknown architecture, malformed bytes,
unavailable Metal on valid data and an invalid backend name. A missing or
duplicate case, unexpected failure, malformed result, invalid tokens/position,
or cross-language mismatch fails the runner.

External Rust controls require a wildcard match to compile and an exhaustive
match to fail with exactly one primary E0004 error in the intended source file.
Removing only `#[non_exhaustive]` makes the same exhaustive consumer compile;
restoring it makes the rejection return. This proves Rust attribute sensitivity;
it supplies no cross-version foreign ABI guarantee.

Ten Python harness tests cover visibility-only transformation/reversal and
anchor drift, missing/duplicate/invalid result evidence, strict compiler
diagnostics, input additions/removal/modification, and process launch/failure/
timeout records, surviving descendant cleanup, Cargo artifact selection and
ancestor/Cargo-home configuration drift, removal of runtime library overrides,
and exclusive target children for repeated runs with the same parent. Builds use the repository's pinned UniFFI 0.31.2 and
wasm-bindgen 0.2.117 dependencies, with matching CLI and hashed JNA 5.16.0.
Reports record source/mirror/probe/lock/artifact hashes and tool versions, and
verify inputs and generated artifacts again after consumer execution.

Ten harness tests and authored formatting/lint pass. The descendant regression
also rejects the original timeout cleanup. The full runtime run supplied a
conflicting Cargo target, older-library macOS/JVM overrides and a relative CLI
path. Native checks use an explicit target; Cargo artifact records select the
outputs; config files, selected flags and removed runtime variable names are
recorded. Swift mutates independent storage and asserts its address is unchanged,
so copy-on-write cannot preserve an unmodified caller allocation in that case.

Direct controls in the run's `library-selection.json` establish sensitivity:
Swift's trace loads the old library with the override and current artifact bytes
with the controlled environment. Kotlin can bypass a deliberately missing
explicit library through inherited JVM options; the controlled environment
rejects the missing path. The ordinary positive run uses the current library.

Five max-effort rounds completed with three reviewers per round. All three
final reviewers returned NO FINDINGS. Eight distinct findings are fixed; none
were skipped or remain open. Every run uses an exclusive child of its target
parent, and artifact hashes are captured at build/generation boundaries. Final
native/WASM Clippy and Rustdoc pass with warnings denied on that isolated
mirror. Local documentation links and status are audited. Plan 09 is complete
as a bounded generated-loading-contract increment.

## Plan 10 core refresh

Updated: 2026-09-07T21:39-0700. The primary metadata reuse change requires fresh
core hashes; plan 09's report remains historical evidence for its snapshot.
The historical Plan 10 complete runtime is
`tests/api_loading/build/run-56svuhvn/results.json`, status `passed`. The
unmodified runner and consumers pass the same 21 command expectations and
35 cases (Swift 12, Kotlin 12, Node 11), with tokens `[0, 1, 0]` at position 5
after parent release. Ten harness tests pass. The resolved lock SHA256 remains
`823b1c1db5389c1cc9fde2ed32fc056d785e81253310e33c51f71f617a7eda81`.
The report captures Plan 10 source, mirror, input and artifact hashes. The
older direct override/library-selection controls above belong to run-n2nckicv;
they were not repeated in this core-only increment.

Final scoped checks pass with warnings denied. From
`tests/api_loading/build/run-56svuhvn/workspace`, use Homebrew on PATH,
`CERA_GIT_SHA=loading-probe` and
`CARGO_TARGET_DIR=/Users/dberrios/development/cera/target/api-loading/run-56svuhvn`:

```sh
cargo clippy -p loading-native --target aarch64-apple-darwin --all-targets --locked --offline -- -D warnings
cargo clippy -p loading-web --target wasm32-unknown-unknown --lib --locked --offline -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc -p cera -p loading-native --target aarch64-apple-darwin --lib --no-deps --locked --offline
RUSTDOCFLAGS='-D warnings' cargo doc -p loading-web --target wasm32-unknown-unknown --lib --no-deps --locked --offline
```

Plan 10 is complete after two max-effort rounds with three fresh reviewers per
round; all three final reviewers returned NO FINDINGS. The Linux-only assertion
fix was the sole distinct finding; none were skipped or remain open. Fresh run
`run-56svuhvn` passed after the fix and all 372 recorded source hashes matched the
Plan 10 tree. All 11 recorded artifact hashes still matched after scoped lint/doc
builds. Run-weeqz9_a predates that correction. Ten harness tests pass again.
Scope and runtime limitations below
still apply; primary parse counts are Rust CPU evidence in the loading audit,
not a foreign-runtime performance measurement.

## Plan 11 cache-policy core refresh

Updated: 2026-09-07T22:07-0700. Current report:
`tests/api_loading/build/run-d4jn9b_b/results.json`, status `passed`.
The unchanged consumers pass 21 command expectations (two deliberate E0004
rejections), Swift/Kotlin 12 cases each and Node 11, with matching tokens
`[0, 1, 0]` at position 5 after parent release. Ten harness tests pass.
All 373 current core source hashes, 11 generated/native/WASM artifact hashes
and four consumer artifact hashes match after scoped lint/doc builds.
The resolved lock SHA256 is unchanged:
`823b1c1db5389c1cc9fde2ed32fc056d785e81253310e33c51f71f617a7eda81`.

From `tests/api_loading/build/run-d4jn9b_b/workspace`, use Homebrew on PATH,
`CERA_GIT_SHA=loading-probe` and
`CARGO_TARGET_DIR=/Users/dberrios/development/cera/target/api-loading/run-d4jn9b_b`:

```sh
cargo clippy -p loading-native --target aarch64-apple-darwin --all-targets --locked --offline -- -D warnings
cargo clippy -p loading-web --target wasm32-unknown-unknown --lib --locked --offline -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc -p cera -p loading-native --target aarch64-apple-darwin --lib --no-deps --locked --offline
RUSTDOCFLAGS='-D warnings' cargo doc -p loading-web --target wasm32-unknown-unknown --lib --no-deps --locked --offline
```

All four checks pass. Plan 11's source increment completed one max-effort round
with three fresh reviewers, all NO FINDINGS; none were skipped or remain open.
The [ownership audit](API_RESHAPE_P0_OWNERSHIP.md#plan-11-anonymous-model-persistent-cache-policy)
contains the CPU LFM2 cold-cache negative control and anonymous-model policy.
These foreign consumers use a Llama fixture and do not execute LFM2 disk-cache
controls or prove KV speed. Device runtimes and the Leap facade remain open.

## Plan26 native multipart files

The native candidate now represents all eight `ModelFiles` fields and delegates
to the private Rust files source. Complete Swift/Kotlin consumers check resolved
paths, extras, inference/template metadata, both builders, consumed state and
post-release sessions. See the [runnable example](../../tests/api_loading/README.md#native-multipart-file-example)
and [loading exit audit](API_RESHAPE_LOADING_EXIT.md) for scope and remaining
configuration/source prerequisites. Run-mvz7tlqs passes all 21 command expectations,
17 cases in each native consumer and 12 in Node. Harness, lint, Clippy, Rustdoc,
source/artifact and documentation audits pass. Two max-effort rounds end clean
with three reviewers each; the one fixed finding adds the structured loading-error
gate to the exit audit. Completion evidence is in the current handoff. Node retains
its bytes/parts-only source representation.

## Plan27 native context boundaries

The candidate native LoadConfig context and requested-context observation now
use u64. Generated defaults preserve4096/None/false; zero and usize::MAX report
the model limit, while ordinary requests remain visible even when allocation
is capped. Both native builders exercise defaults, zero,64/65,4294967320,u64::MAX
and context1 with continued session ownership after parent release.
The [runnable example](../../tests/api_loading/README.md#native-context-configuration-example)
records exact behavior. A probe-only wasm32 function exercises the same checked
conversion at actual32-bit pointer width; existing web config remains u32 Number.
Run45e5w89q passes21 command expectations and22/22/13 consumer cases. Ten harness
tests, scoped lint, native/WASM Clippy/Rustdoc and artifact/documentation audits
pass. One max-effort round with three reviewers is clean, with one resumed tool
interruption and no findings. Completion records are in the current handoff.
Repository/progress and structured loading errors remain open; no production
binding changes are made.

## Plan28 structured loading errors

The candidate now carries source kind, requested assembly backend, unsupported
inference type and invalid config field/value/reason through generated errors.
Swift/Kotlin/Node check those variants and payloads through both builders,
including a parsable primary header with no model weights and explicit unsupported
multipart inference. Native file errors and wasm32 config overflow use structured
payloads too. Details are diagnostic; no display-text parsing chooses a category.
See the [runnable error example](../../tests/api_loading/README.md#structured-loading-error-example)
and current handoff for validation/review status. Legacy production errors are
unchanged. Generic Session errors and future-core fallback stay scoped separately;
this is not a future-version ABI guarantee or an internal assembly-cause taxonomy.

Run `run-bijrgzj4` passes 21 command expectations and 24/24/15 consumer cases.
Core loading tests pass 25 minimal, 36 mmap and 70 remote cases, each with one
existing ignored test. Scoped checks, core/native/WASM Clippy, native/WASM Rustdoc
and input/artifact/document audits pass. Two max-effort rounds with three reviewers
each end clean after exact remote source-label/phase assertions were strengthened;
no skips, interruptions or open findings. The [completion record](API_RESHAPE_HANDOFF.md#completed-plan28--structured-loading-errors)
records the current evidence. Native remote source/repository/progress and target
retention remain loading prerequisites; no production binding changes are made.

## Plan29 native remote sources and repository ownership

The native candidate now includes BundleId/HuggingFace source payloads and an
optional BundleRepo in LoadConfig. The foreign progress trait forwards to the
same core callback; the loader clones the actual core repository. The
[complete remote consumers](../../tests/api_loading/README.md#native-remote-loading-example)
exercise both builders, source/error payloads, cold-download progress and silent
cache reuse, with session and callback retention after caller handles close.
Manifest-file/directory cases pass. The guarded consumer retry
`consumer-retry-rdgqz5ec` passes 33/33/15 cases using verified unchanged native/WASM
artifacts from `run-gxemqvwz` and fresh remote stores. Scoped checks, native/WASM
Clippy/Rustdoc, artifact/document audits and two max-effort three-reviewer rounds
pass. The [completion record](API_RESHAPE_HANDOFF.md#completed-plan29--native-remote-loading)
retains failed-run history and exact evidence. The scoped probe reuses production resolver/repository code and
does not change production bindings or expose probe observations as public APIs.

## Plan30 target retention and selected production signatures

The [target retention map](API_RESHAPE_TARGET_RETENTION.md) covers Rust,
UniFFI Swift/Kotlin/Python/Dart, CPU WASM, WebGPU and Dart's worker-backed API.
The [declaration check](../../tests/api_contracts/README.md) freezes reviewed
existing methods and payloads while source/ABI/platform compilation stays separate.

Three candidate contracts remain: use production native EngineConfig with its typed
backend and existing BundleRepo; expose shared production CeraEngine/Session
access with the full SessionConfig; and carry full Text/Audio/Other multipart
generation defaults. The probe's string LoadConfig, observations, fixed session
wrapper and text-only SamplingDefaults do not prove those contracts. Plan31 should
execute these changes without reloading weights, reconstructing live state or
removing any legacy/async operation. Current review/check status is in the handoff.
Plan29's results are historical: three original build artifacts became absent
during Plan30. The source/result and 12 surviving artifact checks are separate
from a full runtime replay; rebuild before the next binding execution.

## Remaining gates

No whole major phase is complete. Native execution is macOS arm64 CPU; web
execution is Node CPU WASM. Android/iOS devices, browser/WebGPU/Metal, real
auxiliary and draft models through these bindings, live remote service coverage,
per-target retention and future-version ABI compatibility
remain open. The fixture is a tiny synthetic Llama and does not establish
production numerical accuracy, warm-chat behavior or KV performance budgets.

These are Cera loading API prototypes. The actual Leap `ModelRunner` facade,
conversation streams/history/cancellation, LoRA/embedding composition, public
package compatibility profiles and replacement applications remain C0/C1/C2.

## Plan 12 executable examples and companion test refresh

Updated 2026-09-07T22:48-0700. Current generated report:
`tests/api_loading/build/run-s3z9znv1/results.json`, status passed. It contains
21 command expectations (19 successes and two intended E0004 rejections),
12 Swift cases, 12 Kotlin cases and 11 Node cases. All three report tokens
`[0, 1, 0]` at position 5 after model/loader handles are released. Ten harness
tests pass. The final source snapshot includes the Plan 12 test and README
corrections; earlier run-h8w8z40l and run-ai0cpnwm are intermediate reports.

The [example guide](API_RESHAPE_EXAMPLES.md) and root/crate/probe README links
show actual typed loading, retained sessions, generation and companion inputs.
The complete [Rust walkthrough](../../tests/api_loading/consumer/src/bin/walkthrough.rs)
runs separately from the 21-command probe, using the same fixture and exact
Cargo target. It generates `[0, 1, 0]`, appends one token to the same session,
then generates `[0, 1, 0]` again; positions advance from 5 to 9. Its execution
record and binary/source hashes are recorded in `walkthrough.json` next to the
probe report after the final mirror checks. This confirms raw CPU Llama live-KV
continuation, not reusable model prefix caching or incremental chat.

Exact mirror: `tests/api_loading/build/run-s3z9znv1/workspace`.
Exact target: `/Users/dberrios/development/cera/target/api-loading/run-s3z9znv1`.
Resolved lock SHA256: `823b1c1db5389c1cc9fde2ed32fc056d785e81253310e33c51f71f617a7eda81`.
All four final mirror native/WASM Clippy/Rustdoc checks pass with warnings
denied. The walkthrough runs and passes its Clippy check. All 377 source, 11
generated/native/WASM and four consumer artifact hashes match after these
builds; walkthrough.json additionally records its source/binary/log hashes.
Use the existing four native/WASM lint/doc commands against this exact mirror
and target, with `CERA_GIT_SHA=loading-probe`. The walkthrough additionally uses:

```bash
cargo run -p loading-consumer --bin walkthrough --features cera/mmap --locked --offline -- /absolute/path/to/run-s3z9znv1/model.gguf
cargo clippy -p loading-consumer --bin walkthrough --features cera/mmap --locked --offline -- -D warnings
```

The language consumers still use the small Llama fixture. Real synthetic CPU
vision and DSpark execution is established by the Rust companion tests in the
[loading audit](API_RESHAPE_P0_LOADING.md#plan-12-executable-vision-and-dspark-companions).
No actual Leap runtime, audio companions, device execution or performance budget
is claimed by these generated bindings. P0-L and C1 remain open.

Plan 12 completed 2026-09-07T22:54-0700; all three final max-effort reviewers are clean
after three rounds. The walkthrough and example guide are included in that review.


## Plan 13 audio example and generated consumer refresh

Validated 2026-09-08T05:24-0700. Current report:
`tests/api_loading/build/run-b1_7do81/results.json`, status passed. The standard
probe retains 21 expectations, including two deliberate E0004 rejections, and
35 foreign cases: Swift 12, Kotlin 12, Node 11. Each produces `[0, 1, 0]` at
position 5 after parent release. Ten harness tests pass. All 381 source hashes,
11 generated/native/WASM artifact hashes and four consumer hashes match after
all exact mirror checks. Lock SHA256:
`823b1c1db5389c1cc9fde2ed32fc056d785e81253310e33c51f71f617a7eda81`.

Exact mirror: `tests/api_loading/build/run-b1_7do81/workspace`.
Exact Cargo target: `/Users/dberrios/development/cera/target/api-loading/run-b1_7do81`.
Native/WASM Clippy and Rustdoc pass with warnings denied. Both external Rust
walkthroughs also compile, execute and pass Clippy in that mirror.

The [audio walkthrough](API_RESHAPE_AUDIO_EXAMPLE.md) separately records:

```text
audio input: 2 positions; text: [1, 1, 1, 1, 1, 1]; output: 22560 PCM samples at 24000 Hz; final position: 22
```

`audio-walkthrough.json` records the exact command, source, binary, fixture and
log hashes. `walkthrough.json` retains the text example's `[0, 1, 0]` outputs
at positions 5 and 9. These supplementary example executions are outside the
standard 21-command report. The synthetic Rust audio example executes Conformer,
depthformer and detokenizer computation after parent release. Swift/Kotlin/Node
consumers still use the text-only Llama fixture; foreign audio and the Leap
facade are not validated by this result.

The initial `run-mohzbp8g` standard probe passed, but its separate audio example
failed E0283 because an extra `.into()` left a generic argument ambiguous. The
fix removes that conversion. This earlier run predates the explicit same-length
PCM sensitivity assertion too. Its generated Cargo target was removed to recover
disk space after a later build hit ENOSPC; reports, logs and source mirror remain.
The final target above is retained. Review status is in the
[handoff](API_RESHAPE_HANDOFF.md#completed-plan-13-audio-loading).

## Plan 14 evidence scope

Plan 14 adds remote companion library tests and documentation. The generated
`run-b1_7do81` report above belongs to Plan 13's recorded snapshot; Plan 14 does
not rerun generated Swift/Kotlin/Node consumers. Production loading/inference
and consumer sources remain unchanged, but test-module source hashes differ.
The [remote examples](API_RESHAPE_REMOTE_EXAMPLES.md) establish CPU Rust runtime
behavior only. Foreign media and Leap runtime compatibility remain open.

## Plan 16 production dependency refresh

Named cache identities change production source and enable SHA256 under the
default disk-cache feature, including ARM acceleration on supported aarch64
targets. The earlier Plan13 and pre-fix Plan16 `run-srsy5tzn` reports are historical
snapshots. Final `run-83j85_o3` passes 21 expectations and 35 runtime cases; generated
Clippy/Rustdoc and both Rust walkthroughs pass. The 390-file source snapshot has two documented comment-only differences; all
executable source, dependency values, 21 probe inputs, two workspace inputs and
generated/consumer artifact hashes match; see the [handoff](API_RESHAPE_HANDOFF.md#completed-plan-16-named-cache-identities).
The private loading facade and generated cases are unchanged. This does not
implement the Leap runtime or publish a replacement package.

Plan30 completed 2026-09-12T06:11-0700: ten declaration controls, lint/format and document
checks pass; three max-effort rounds with three reviewers end clean. The handoff
records all three fixed findings and the historical-runtime artifact limitation.

## Plan31 full multipart defaults

The candidate now carries Text/Audio/Other defaults through native variants and
CPU WASM factories. The [runnable examples](../../tests/api_loading/README.md#multipart-generation-defaults)
cover field absence, audio-specific values and JSON, including malformed input.
Generated runtime passes39 Swift/39 Kotlin/21 Node cases; the text-only negative
control is rejected by all three consumers. Two max-effort rounds with three
reviewers end clean; Plan31 is complete. Config reuse and shared engine/session
access remain separate next proofs; this increment does not close P0-L.
