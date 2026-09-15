# API reshape implementation handoff

Updated: 2026-09-15T18:17-0400. This is the current implementation record; the
review worktree preserves the earlier design review and is not the active branch.

## Plan47 complete (2026-09-15T18:17-0400)

Lowering of the public Chat coordinator, session phases, and turn execution contracts
to Swift and Kotlin via UniFFI in `cera-ffi`, with non-destructive session transitions,
wait-free cancellation, non-blocking recovery diagnostics, and terminal stream event guarantees.

- [x] Canonical UniFFI wrappers: defined `ChatSession`, `Role`, `Message`, `SessionPhase`,
      `ValidationError`, `IngestSummary`, and `TurnResult` in `cera-ffi/src/chat.rs`.
      Implemented infallible and fallible conversions (`From`, `TryFrom`) with core types,
      including `TryFrom<&cera::session::chat::Message>` with single-part text extraction
      and `FfiError::UnsupportedModality` error mapping. Added convenience constructors
      (`chat_message_user`, `chat_message_system`, `chat_message_assistant`, `chat_message_tool`).
- [x] Wait-free cancellation & non-blocking recovery: `ChatSession::cancel` performs a
      wait-free atomic store on `self.cancel: Arc<AtomicBool>`, eliminating re-entrant
      deadlocks when streaming callbacks cancel decode from within foreign sinks and
      preventing UI thread stalls. `ChatSession::position` and `ChatSession::clear_cancel`
      are lock-free via `position: Arc<AtomicU32>` and fail-close via `moved: AtomicBool`.
      `ChatSession::recovery_status` queries inner state non-blockingly with `try_lock()`,
      returning `FfiError::Busy` on contention.
- [x] Terminal streaming event guarantees: `ChatSession::generate_streaming` synthesizes
      terminal `on_done` callbacks (`FinishReason::Error` or `FinishReason::Cancelled`)
      on all early exit paths (option conversion failure, lock poisoning, moved session,
      and decode errors), preventing foreign stream listeners and coroutines from hanging.
- [x] Non-destructive Session transitions: `Session.inner` in `cera-ffi/src/lib.rs` evolved
      to `Mutex<Option<cera::Session>>` with `SessionGuard` implementing `Deref` and `DerefMut`.
      `Session::into_chat` transfers the inner session into a `ChatSession`, and
      `ChatSession::into_session` provides symmetric return. Added `CeraEngine::new_chat_session(config)`.
      Added `FfiError::ChatValidation { error: ValidationError }`.
- [x] Contract & binding parity: `python3 tests/api_contracts/check.py` matches 69/69 surfaces.
      Foreign language bindings regenerated via `just bindings` and `just dart-bindings`
      for Swift, Kotlin, Python, and Dart.
- [x] Comprehensive test coverage: 12 unit tests in `cera-ffi/src/chat/tests.rs` covering
      multi-turn chat lifecycle, sliding context refusal, session transfer and reclamation,
      wait-free re-entrant cancellation during streaming decode, non-blocking recovery status,
      terminal callbacks on early validation failure, and bidirectional message conversions.
- [x] Verification gates clean: all 56 FFI unit tests, 15 contract tests, 43 core tests,
      6 runner tests, Dart analysis, formatting, clippy, and rustdoc pass cleanly.

Evidence: `devlog/plans/000341-47-uniffi-chat-lowering.md`.
Forty-four increments are complete (42 core, two Leap). Next unused sequence is 48.

## Plan46 complete (2026-09-15T00:10-0400)

Promotion of the core chat contract, transactional coordinator, and Session adapter
into the public API of the `cera` library crate, preparing for P0.2 foreign language lowering.

- [x] Public chat module: exposed `pub mod chat;` in `cera/src/session.rs` and canonical re-exports
      in `cera/src/lib.rs` (`Chat`, `SessionChat`, `Message`, `Role`, `ContentPart`, `SessionPhase`,
      `TurnResult`, `DecodeReport`, `DecodeState`, `Execution`, `Profile`, and error types).
- [x] Canonical implementation in crate: relocated `Chat`, `Execution`, `Profile`, `Message`, and
      contract types directly into `cera/src/session/chat.rs`, removing dependencies on `tests/`
      from production crate sources. `cera/tests/api_chat/contract.rs` re-exports from `cera::session::chat`.
- [x] Public Session entry point: added `Session::into_chat(self) -> Result<SessionChat, (Session, ValidationError)>`
      in `cera/src/session.rs`, symmetric with `Chat<CoreExecution>::into_session(self) -> Session`.
- [x] Default generic parameter: defined `pub struct Chat<E = CoreExecution>` allowing concise un-parameterized
      usage in downstream code while preserving genericity for custom/mock execution backends.
- [x] Test harness parity: `cera/tests/api_chat/tests.rs` uses extension trait `ChatTraceExt` to provide
      `set_failure` and `tokens` helpers on `Chat<TraceExecution>`, conforming to Rust orphan rules.
- [x] Documentation & gates: full rustdoc comments on public types with zero em dashes; `cargo doc -p cera --no-deps --lib`
      passes with warnings denied; all 43 core tests, 15 contract tests, 6 runner tests, 69 contract surfaces,
      and nightly clippy pass cleanly.

Evidence: `devlog/plans/000341-46-public-core-chat-promotion.md`.
Forty-three increments are complete (41 core, two Leap). Next unused sequence is 47.

## Plan45 complete (2026-09-14T17:08-0400)

Resolution of open chat design items prior to foreign lowering: cancellation handle and clear,
non-destructive session return on refusal and teardown, and fallback reset for backends
without device-level checked KV reset.

- [x] Cancellation management: Added `cancel_handle()` (`Option<Arc<AtomicBool>>`), `cancel()`,
      and `clear_cancel()` to `Execution` trait, `Chat<E>`, and `CoreExecution`. Delegated directly
      to `Session::cancel_handle()`, `Session::cancel()`, and `Session::clear_cancel()`.
      Clearing cancel is non-destructive, leaving phase and cursor intact.
- [x] Non-destructive Session return: `Chat::new` returns `Result<Self, (E, ValidationError)>` so
      callers retain their `Session` on validation refusal (`SlidingContext`, `AudioOutput`,
      `UnsupportedProfile`). Added `Chat::into_inner(self) -> E`, `CoreExecution::into_session(self) -> Session`,
      and `Chat<CoreExecution>::into_session(self) -> Session`. Implemented `Debug` for `Session`,
      `CoreExecution`, `DecodeReport`, and `Chat<E>` to maintain ergonomic `.unwrap()` ergonomics.
- [x] Fallback reset for models without `try_reset_kv`: `CoreExecution::reset` and `Session::reset`
      attempt checked KV reset first (`reset_execution_checked()`), falling back to state re-allocation
      (`reset_realloc_state()`) if the backend returns `Backend("checked KV reset is not supported by this backend")`.
      This prevents generic architectures and CPU backends from permanent unusable lockout across explicit
      or replacement resets.
- [x] Pinned test suites: updated `ISOLATED_CASES` to 15 and `CORE_CASES` to 43 in `tests/api_chat/run.py`.
- [x] All 43 core transaction tests and 15 isolated contract tests pass; static analysis
      (`cargo +nightly clippy -p cera --all-targets --all-features -- -D warnings`), formatting
      (`cargo +stable fmt --check`), API contract checks (`check.py`, 69 surfaces), and runner
      tests (`test_runner.py`, 6 tests) pass cleanly.

Evidence: `devlog/plans/000341-45-chat-open-design-items.md`.
Forty-two increments are complete (40 core, two Leap). Next unused sequence is 46.

## Plan44 complete (2026-09-14T07:03-0400)

LFM2.5-350M (`models/LFM2.5-350M-Q4_0.gguf`) chat contract profile discovery and
numerical warm-turn verification completed across core and isolated contract suites.
Both LFM2 and LFM2.5 model generations are recognized, verified, and passing.

- [x] Extended `Profile::discover` in `cera/tests/api_chat/contract.rs` to recognize
      `LFM2_5_TEMPLATE` alongside `TEMPLATE`. Verified identical ChatML turn boundary
      framing across both model generations.
- [x] Executed 10 consecutive warm conversational turns on `LFM2.5-350M-Q4_0.gguf`
      through `Chat<CoreExecution>`, proving physical KV cache retention across turns
      and delta-only prompt evaluation.
- [x] Verified cold reference prefill parity at turn 0 on `LFM2.5-350M-Q4_0.gguf`.
- [x] Bounded repetition grammar (`root ::= [a-zA-Z]{1,16} "\n"`) in warm-turn test,
      ensuring reliable `FinishReason::Stop` on `self.profile.eos` across model generations.
- [x] Verified stochastic RNG determinism (identical seeds) and divergence (different seeds)
      using creative story prompt with system message across both LFM2 and LFM2.5.
- [x] Verified interrupted turn handling and replacement recovery on `LFM2.5-350M-Q4_0.gguf`.
- [x] Configured multi-profile support in `tests/api_chat/profile.json` and `tests/api_chat/run.py`
      to match model hashes against candidate profile pins.
- [x] All 35 core transaction tests and 13 isolated contract tests pass for both
      `LFM2-350M-Q4_0.gguf` and `LFM2.5-350M-Q4_0.gguf` in default and minimal
      (`--no-default-features`) configurations.
- [x] Verified formatting (`cargo +stable fmt --check`), clippy (`-D warnings`),
      contract suites (69 surfaces, 8 classes, 5 mutations), and test runner (6 tests).

Evidence: `devlog/artifacts/000341-plan44/` (`lfm25-core`, `lfm25-core-minimal`,
`lfm25-contract`, `lfm2-core`, `lfm2-contract`). Plan `devlog/plans/000341-44-lfm25-chat-and-warm-turns.md`.
Forty-one increments are complete (39 core, two Leap). Next unused sequence is 45.

## Plan43 complete (2026-09-13T18:05-0400)

R1 numerical warm turns, bitwise KV cache retention, and deterministic sampling
proven on physical model weights using the pinned public fixture
`devlog/artifacts/000341-models/LFM2-350M-Q4_0.gguf` through `Chat<CoreExecution>`.
No public chat facade is published; all validation executes through the private
unit-test adapter and the pinned artifact runner.

- [x] Executed 10 consecutive warm conversational turns through `Chat<CoreExecution>`.
      Verified exact physical KV cache retention across turns: for all attention
      layers, key and value tensors for prior positions match with bitwise equality
      (`diff == 0.0`).
- [x] Verified delta-only prompt evaluation: each turn evaluates only the delta
      tokens plus the pending assistant boundary (`summary.position_before == pos_before`,
      `chat.position() == pos_before + summary.input_tokens`) without recomputing
      earlier context or copying the KV cache.
- [x] Verified cold reference prefill parity: warm prefill logits for turn 0 match
      a cold standalone prefill of the full prompt with bitwise equality.
- [x] Verified deterministic generation reproducibility: consecutive independent
      sessions with fixed seeds yield identical token sequences across all 10 turns.
- [x] Proved stochastic RNG determinism and divergence on real weights: identical
      seeds produce identical token streams, while different seeds diverge.
- [x] Proved interrupted turn and replacement recovery on real weights: hitting
      `FinishReason::MaxTokens` sets `SessionPhase::Interrupted`, blocking subsequent
      ingestion, and `replace_messages` rewinds and resets cleanly to resume identical
      generation.
- [x] GBNF grammar constraint (`root ::= [a-zA-Z]+ "\n"`) configured in `opts.grammar`
      to enforce valid word generation ending in newline, allowing the model to cleanly
      trigger `FinishReason::Stop` on `self.profile.eos` (token 7: `<|im_end|>`) and
      advance the session phase to `TurnComplete`.
- [x] Pinned runner test count updated from 31 to 34 (`CORE_CASES = 34`). All 34
      cases pass in both default and minimal native configurations (`--core-transactions`
      and `--core-transactions --no-default-features`), alongside all 12 isolated
      contract cases.
- [x] Passed full verification: 50 session regressions, 8 spec regressions, 13
      grammar regressions, 52 FFI regressions, 4 runner controls, native all-feature
      Clippy and Rustdoc, WASM core/binding Clippy, WASM lib-test compilation, 69
      declarations, and formatting.
- [x] Four-subagent code review across correctness, conventions/portability,
      concurrency/lifecycles, and simplification/regressions.

Evidence: `devlog/artifacts/000341-plan43/` (`core-default`, `core-minimal`,
`contract-default`, `contract-minimal`, `checks.py`). Build target
`devlog/artifacts/000341-build/native`, pinned model
`devlog/artifacts/000341-models/LFM2-350M-Q4_0.gguf`.

Forty increments are complete (38 core, two Leap). Plan 44 next steps: P0.2
per-target latency and memory budgets, native/browser async lowering, and public
API promotion preparations. Plan `devlog/plans/000341-43-r1-numerical-warm-turns.md`.
Next unused sequence is 44. No commits or pushes.

## Plan42 complete (2026-09-13T15:39-0400)

The private core adapter `cera/src/session/chat.rs` (unit-test build only)
executes the shared chat contract against actual Session whole-batch append,
checked replacement/reset and observed decode. No public chat facade, numerical
model or performance claim is added. The increment was wrapped across an agent
switch: implementation and the first check run happened before it, the review
loop and final audit after it.

- [x] Whole-batch append through `with_ingest_recovery`, owned primary/typed
      rewind/secondary reset errors, checked replacement preserving external
      cancellation, decode failure guard, profile identity checks.
- [x] Explicit reset never consults the profile: identity drift through `raw()`
      (swapped Session, classifier adapter) is undone through `raw()` again
      instead of stranding a healthy Session inside an Unusable chat.
- [x] `NoProgress` observations never disable raw execution; a no-progress error
      leaves the cursor in RawContext (replacement/raw open), a successful
      audio-path exit enters Interrupted, only unproven/error/unwind outcomes
      disable execution.
- [x] Identity re-check at prepare/decode includes `n_keep`, a model vocabulary
      smaller than the tokenizer's, classifier LoRA (canonical `Session::lora`)
      and text capabilities; stale legacy diagnostics cannot be mistaken for the
      adapter's own.
- [x] 31 actual-Session cases (29 offline plus two pinned-tokenizer) and 12
      isolated contract cases pass in default and minimal builds; the runner pins
      both counts exactly and requires the named public fixtures.
- [x] Pass 50 Session, 8 speculative, 13 grammar and 52 FFI regressions, 4 runner
      controls, native all-feature Clippy/Rustdoc, WASM core/binding Clippy and
      WASM lib-test compilation, 69 declarations and formatting.
- [x] Four review rounds at max effort, four lenses each (correctness,
      conventions/portability, lifecycles/cancellation, simplification/
      regressions), findings fixed between rounds. Round four resolved the
      contradictory NoProgress generated-token state mapping, redundant Arc
      cloning, model config query ordering, and cross-platform runner paths.
- [x] Final audit: 36 docs/437 local targets/75 anchors clean; 344 source
      inputs, four exact executables and the pinned model verified.

Evidence: repository-root `devlog/artifacts/000341-plan42` (`checks.json`,
`checks-round0.json`, `checks-round2.json`, `reviews.json`, `doc-audit.json`,
`artifact-audit.json`, `audit-artifacts.py`, latest `run-*` under the four
runner directories). Build target `devlog/artifacts/000341-build`, pinned model
`devlog/artifacts/000341-models/LFM2-350M-Q4_0.gguf`. The predecessor baseline
in `before/` is the reconstructed one described under the environment-recovery
note in the plan; it is not claimed byte-identical to the lost original.

Review findings worth remembering: round one found that the reset path re-ran a
pre-mutation validation after the phase guard had flipped (three reviewers
independently), that the model-identity arm was untested because the swap test
minted two tokenizer Arcs, and that the README claimed prefill-unwind coverage
that existed only for decode. Round two found the runner's case floor already
below the running count, an `n_keep` re-check wrongly removed in round one, and
that the `NoProgress`-error phase choice refused the very access it claimed to
preserve. Round three found that the round-two `n_keep` edit had never landed
(an edit batch aborted before writing) and pinned it with a raw-swap test.

Open design items recorded in `docs/internals/API_RESHAPE_CHAT.md` for the public
facade: no chat cancellation handle or non-destructive clear; replacement and
explicit reset always take the checked `try_reset_kv` path, which is terminal
Unusable on backends without it; construction consumes the Session on refusal.

Thirty-nine increments are complete (37 core, two Leap). R1 still needs numerical
warm KV/RNG/drafter proof on real model weights; P0.2 per-target budgets, foreign
and browser lowering, public promotion/migration and Leap runtime/package gates
remain. Plan `devlog/plans/000341-42-session-chat-transactions.md`. Next unused
sequence is 43. No commits or pushes.

## Plan41 complete (2026-09-13T06:43-0700)

Core generation now records call-local facts for chat: exact pending stop token,
proven no-progress, audio path, interruption or unproven state. Existing public
Result/summary/callbacks remain. Speculative verification checks backend eligibility
after forward immediately before each rewind and validates the resulting position.
Any unproven rewind stays latched, even if legacy generation returns Stop. An
unwind returns no positive outcome. No public chat facade is published yet.

- [x] Implement bounded actual ordinary/speculative/audio observations without retained stale diagnostics.
- [x] Pass15 decode controls in default/minimal builds, including RNG discrimination, grammar, audio, cancellation and callback/forward unwind.
- [x] Cover expired real64-entry convolution history at both speculative rewind sites, counter-only backend state and validity latching.
- [x] Pass50 default/42 minimal Session,8 speculative,13 grammar,52 FFI and11 public-tokenizer chat tests (counts overlap).
- [x] Pass native all-target/all-feature Clippy/Rustdoc, WASM core/binding Clippy and actual WASM lib-test compilation, formatting and69 retained declaration groups.
- [x] Verify three normalized legacy bodies,342 source inputs and exact default/minimal/chat/WASM artifacts; audit36 docs and431 local targets.
- [x] Complete two max review rounds; three final reviews clean with no open/skipped findings.

Evidence `/private/tmp/cera-api-plan41-decode-observations`: `final-pass.json`,
`reviews.json`, `artifact-audit.json`, `checks.json`, `wasm-lib-tests-check.json`,
`unchanged-legacy-core.json`. First review found a positive stop observation after
unproven legacy speculative rewind; fixed and re-reviewed. One review protocol
failure was retried, never counted as clean. Initial lint/audit failures and older
11-case inputs are retained as historical evidence. The supplementary WASM
all-tests Clippy command fails in five unchanged native mmap-dependent integration
targets; focused WASM lib-test compilation passes. No WASM runtime or numerical
GPU/audio inference claim is made by these controls.

Thirty-eight increments are complete (36 core,two Leap). R1 still needs actual
transactional chat integration and numerical warm KV/RNG/drafter proof. P0.2
per-target budgets, bindings, public promotion/migration and Leap runtime/package
gates remain. These observations preserve legacy unsafe rewind behavior; they
prevent a future chat adapter from certifying it as usable state.

Plan42 is prepared: `devlog/plans/000341-42-session-chat-transactions.md`.
Next: connect the shared chat contract to actual whole-batch Session recovery,
checked replacement/reset, phase failure guards and these decode observations.
Next unused43. No commits/pushes. Last disk check3.7GiB free; reuse existing targets
with incremental disabled. Preserve all evidence; no cleanup performed.

## Plan40 complete (2026-09-13T05:44-0700)

The isolated core chat contract is in `cera/tests/api_chat`; its runner and public
profile pin are in `tests/api_chat`. Rust test modules travel with packaged crate
sources. It defines owned ordered messages, batch/replacement operations, six
phases, bounded pending-boundary bookkeeping and collection over one decode call.
The application owns its transcript. No production chat facade is published yet.

- [x] Pin the public LFM2-350M Q4_0 GGUF revision/hash and full tokenizer.
- [x] Reproduce missing resident EOS and repeated BOS through actual Session decode in greedy/stochastic modes.
- [x] Compare ten structural turn prefixes with full rendering using Unicode and nonempty scripted answers.
- [x] Cover phase, validation, capacity, recovery, replacement, collection, raw capability drift and unwind cases.
- [x] Pass all11 Rust tests in native/default and minimal builds, plus3 harness controls; no skipped public fixture.
- [x] Defeat inherited Cargo runner=/usr/bin/true by executing the exact built test binary and requiring positive fixture evidence.
- [x] Pass native all-target/all-feature Clippy/Rustdoc, WASM prototype Clippy, formatting/version and69 retained declarations.
- [x] Verify packaged test modules,36 doc files/427 local targets and338 source inputs plus both executable/model hashes.
- [x] Complete three review rounds; final three reviews clean, no open/skipped findings.

Evidence `/private/tmp/cera-api-plan40-chat-contract`: `final-pass.json`,
`reviews.json`, `artifact-audit.json`, `package-check.json`; final runs
`runner-control/run-0hdjwm3o/results.json` and `validation/run-ya4ol9mx/results.json`.
The first review found capability drift, a runner false-success and incomplete
source provenance. After the second clean round, packaging inspection found
external test modules missing from the crate package; moved them intact and ran
another clean round. Seven review tool-protocol failures were retried, never
counted as clean reviews. Initial compile/lint failures and earlier evidence remain.

Thirty-seven increments are complete (35 core,two Leap). Plan40's scripted/tokenizer
fixtures do not execute numerical model weights or prove R1 KV/RNG/drafter performance.
P0.1/P0.2/R1 and the production facade/backend/binding gates remain open.
`docs/internals/API_RESHAPE_CHAT.md` records the initial matrix, caller inventory
and actual decode exit sites requiring observations.

Plan41 is prepared: `devlog/plans/000341-41-core-decode-observations.md`;
evidence `/private/tmp/cera-api-plan41-decode-observations`. Next: actual bounded
core decode observations, then transactional chat integration. Next unused plan42.
No commits/pushes. Reuse existing targets and retain evidence; no cleanup performed.

## Plan39 complete (2026-09-13T04:51-0700)

Native `Session.recoveryStatus()` now returns an owned, coherent snapshot of
usability, position and last whole-message recovery, with typed rewind and
secondary reset errors. Original operation errors and existing signatures remain.
Observation uses `try_lock`: Busy applies while the lock is held; terminal
callbacks and final buffered flushes may run after unlock and can read status.

- [x] Add native diagnostics, typed64-bit payloads, lock/poison and report-lifetime proof.
- [x] Regenerate Swift/Kotlin/Python/Dart and SwiftPM; exact generator parity passes.
- [x] Pass52 Rust FFI tests and27 actual cases per Swift/Kotlin consumer across CPU/Metal/wgpu.
- [x] Require a reversed same-length prefix to change output in all54 consumer cases.
- [x] Execute both public examples in standard/compressed modes; all four runs pass.
- [x] Pass Dart analysis and28 tests (three existing skips), native lint/docs,
      69 retained declarations,format/version and409 local doc targets.
- [x] Complete two max review rounds, with three clean final reviews; verify source and runtime artifact hashes.

Plan `devlog/plans/000341-39-native-recovery-diagnostics.md`; evidence
`/private/tmp/cera-api-plan39-native-recovery`. Final consumer reports:
`runtime/cpu-91rjvuus/results.json`, `runtime/gpu-0qvp7odq/results.json`.
`final-pass.json`, `artifact-audit.json`, `regeneration-equality.json` and
`reviews.json` record completion. Initial compiler/telemetry/no-device failures
and weaker-oracle runs are historical and retained. Review corrected callback
wording and selected seed1361/temperature1 with an explicit discriminator; two
search-task protocol failures and one final-review protocol failure were retried.
No open or skipped findings. No commits/pushes.

Thirty-six increments are complete (34 core,two Leap); R0 and P1 remain open.
Next unused plan40: core chat contract, phases, caller inventory and initial
profile/boundary fixtures. CPU WASM has no whole-message append yet; its richer
diagnostics accompany the future chat facade. Browser async recovery, numerical
KV budgets, migrations and Leap runtime/package gates remain. Latest disk28GiB
free without cleanup; reuse existing build targets.

## Plan38 complete (2026-09-13T03:53-0700)

- [x] Implement locked Metal/native wgpu checked reset and strict queue completion.
- [x] Compile actual device tests for hybrid/dense, standard/f16/TurboQuant,
      adapters, immediate snapshots, embeddings, cancellation, replay and ownership.
- [x] Pass4 actual-device tests:12 model/cache cases and2 destroyed-device controls; both GPU examples pass.
- [x] Pass native/FFI/WASM lint/docs,69 declarations, formatting and400 local doc targets.
- [x] Complete two max review rounds with three clean final reviews; verify seven source and two runtime artifact hashes.

Plan `devlog/plans/000341-38-device-execution-reset.md`; evidence
`/private/tmp/cera-api-plan38-device-reset`. Initial test compile omitted the
AttentionF16 snapshot variant; fixed. Sandbox runtime found no Metal/wgpu devices
and all3 tests failed rather than skipping; logs retained in
`sandbox-device-unavailable/`. The same offline tests were rerun with GPU access.
Final runtime confirms original cancellation plus secondary reset error and unusable
state after device loss. Review corrected the performance wording: recovery adds
no full-cache copy/readback to successful ingestion; existing fresh-prefill prefix
cache insertion can still snapshot device state. One round2 protocol failure was
retried and did not count as a completed review. No open or skipped findings.
`final-pass.json` and `reviews.json` record completion. Thirty-five increments
are complete (33 core, two Leap); R0 and P1 remain open. Next unused plan39
will expose recovery diagnostics to foreign callers. Disk refreshed to8.4GiB
free without cleanup; preserve existing evidence and reuse targets. No commits/pushes.

## Plan37 complete (2026-09-12T21:37-0700)

User-message failures now use bounded mutation-aware recovery. Checked CPU reset
rebuilds all KV/convolution state, including compression; unsafe rewind falls back
to reset, then unusable state on failure. Additive Rust diagnostics preserve the
original error and secondary recovery failure. Automatic reset never writes the
external cancellation flag. Existing raw partial-prefill behavior is retained.

- [x] Replace unconditional user-message rollback; preserve complete Session metadata on restoration.
- [x] Handle shifts, zero reported progress, expired convolution checkpoints and reset/unwind failures honestly.
- [x] Enforce unusable state through every current Session execution entry; require checked reset or recreation.
- [x] Pass15 recovery tests in both default/minimal builds,35 Session,10 checked-rewind and8 spec tests (counts overlap).
- [x] Prove6 causal model/precision continuation cases, three compressed modes with adapters,
      two expired-convolution cases, noncausal reset, RNG/drafter metadata and concurrent cancellation.
- [x] Run both public recovery example modes: restored position2 and reset position0; both retry to6.
- [x] Pass core/FFI all-feature Clippy/Rustdoc, no-default check, WASM wgpu lint/docs,
      69 retained declaration groups, formatting and400 local documentation targets.
- [x] Complete one max review round with three clean reviews; one protocol failure retried.
- [x] Verify source and runtime artifact hashes; preserve initial failed evidence.

Plan `devlog/plans/000341-37-session-ingestion-recovery.md`; evidence
`/private/tmp/cera-api-plan37-session-recovery`. Final command logs in `validation/`;
`final-pass.json`, `validated-inputs.json`, `artifact-sha256.json` and `reviews.json`
record completion. Initial sink compile failure and shared32-token fixture-cap
failures are retained with their fixes. No open findings or active jobs.

Thirty-four increments are complete (32 core, two Leap). R0 and P1 remain open.
Next Plan38: prove Metal/native wgpu reset, including device convolution, queue
completion and device errors. Unknown/device reset currently refuses and requires
recreation after a mutated failed append. Rich foreign outcomes, raw/chat phase
inventory, P0.1/P0.2/R1 performance/migration and Leap release gates remain.
No commits or pushes. About3.3GiB free; reuse targets and disable incremental.

## Plan36 complete (2026-09-12T21:04-0700)

Checked CPU KV rewind validates all layers before mutation. Model methods default
to unsupported; causal CPU Llama/LFM2 explicitly opt in. LFM2 bidirectional mode
and classifier LoRA return `NonCausal`. Legacy truncation bodies and callers are
unchanged, verified by normalized Rust comparison. The [recovery matrix](API_RESHAPE_RECOVERY.md)
records every existing rewind caller and the remaining backend/session work.

- [x] Add checked bounds/compression/layout/convolution validation and CPU Model opt-in.
- [x] Pass10 checked tests, including6 causal model/precision combinations and
      noncausal/classifier refusal with unchanged state.
- [x] Demonstrate unchecked bidirectional slicing differs from fresh-prefix KV
      on a real Q8 two-attention model; both controls execute without fallback skip.
- [x] Pass32 cache and8 speculative tests; execute the public raw CPU rewind example.
- [x] Pass native all-target/all-feature Clippy, Rustdoc/no-default and WASM wgpu lint/docs.
- [x] Pass69 retained declaration groups, formatting and the33-file local link audit.
- [x] Complete two max review rounds, ending with three clean reviews.
- [x] Record final source/artifact hashes and preserve failed fixture/review evidence.

Evidence: `/private/tmp/cera-api-plan36-checked-rewind`; final gates in `final/`.
Root plan `devlog/plans/000341-36-checked-rewind.md`. Review found that
bidirectional attention invalidates the prefix assumption. The first F32 fixture
selected per-token fallback, so its defect comparison failed; an asymmetric
suffix alone was insufficient. Preserved both failures and switched the fixture
to Q8 to exercise batched attention. Two tool-protocol review failures were retried.
Approved cleanup removed only listed static Rust archives/metadata; tested
native/WASM binaries and fixtures remain intact. No open findings or active jobs.

Thirty-three increments are complete (31 core, two Leap). R0 and P1 remain open:
`append_user_message` still uses the old failure path, and no chat guarantee is
added by this primitive. Plan37 must integrate complete Session recovery,
cancellation-preserving reset and unusable enforcement. Device recovery,
P0.1/P0.2/R1 chat and performance, migrations and Leap release gates remain.
No commits or pushes.

## Plan35 complete (2026-09-12T20:29-0700)

Production CPU WASM exposes the bytes/parts loading API, complete defaults and
structured errors. `GenerativeModel` shares its engine and creates the existing
Session. Existing constructors, browser async loading and WebGPU operations remain
available. Node consumers exercise production objects; all four test mirror
adaptations only append observations.

- [x] Publish CPU WASM loading and a runnable production Node completion example.
- [x] Pass 63 Swift/63 Kotlin/26 Node cases in `run-wv6wkfou`.
- [x] Pass final Node26 on `node-final-wfnq_7gh` after the final source changes.
- [x] Execute the production-only example in `public-ej9qot5l`, matching native output.
- [x] Freeze eight generated loading classes; pass five declaration mutation controls.
- [x] Pass CPU/wgpu Clippy, wgpu Rustdoc and probe Clippy/Rustdoc.
- [x] Pass 69 retained declaration groups, ten declaration controls, ten harness controls,
      scoped Ruff/format/Node checks and the 32-file local link audit.
- [x] Complete two max review rounds, ending with three clean reviews.
- [x] Verify native/final Node sources, mirror files, artifacts and build environment.

Evidence: `/private/tmp/cera-api-plan35-wasm-loading`; root plan
`devlog/plans/000341-35-wasm-loading.md`. The original full run's native inputs
are unchanged. Final Node evidence covers the EngineConfig default-update and
probe cleanup. `artifact-audit.json` records those differences and verifies all
hashes. A temporary mirror outside the worktree missed the pinned toolchain and
SIMD config; that failed lint report is preserved. Corrected-environment Clippy
found one unused probe import, removed before the final runtime and lint passes.
Approved cleanup removed only twelve recorded disposable incremental caches.
The review's initial supported-build claim for an unsupported feature combination
was withdrawn; the config default-update remains for constructor consistency.
No open findings, commits or pushes.

Thirty-two increments are complete (30 core, two Leap). P1 remains open. Next is
Plan36: checked rewind primitives and the backend recovery matrix, followed by
session-level recovery and the remaining P0.1/P0.2/R1 chat gates. These loading
checks do not establish warm-chat performance budgets or release packaging.

## Plan34 complete (2026-09-12T22:23-0400)

Production native `ModelSource`, `ModelLoader`, `ModelHandle`, `GenerativeModel`
and structured defaults/errors are exported alongside the retained constructors.
`GenerativeModel.engine()` shares the loaded core engine; `createSession` returns
the existing Session with its full config and error contracts. Swift/Kotlin/Python/
Dart wrappers and the SwiftPM copy are regenerated. Package releases remain future.

- [x] Publish native loading, full multipart defaults and shared engine ownership.
- [x] Pass 63 Swift/63 Kotlin/26 Node cases (`run-3h5sr8ph`).
- [x] Run standalone production completion examples in Swift, Kotlin and Dart.
- [x] Pass 51 CPU/Metal/wgpu ownership cases per Swift/Kotlin consumer.
- [x] Pass 48 full-feature FFI tests, Clippy/Rustdoc and scoped declarations/lint/docs.
- [x] Pass Dart analysis and 28 tests; three existing missing-fixture tests skipped.
- [x] Verify fresh Swift/Kotlin/Python/Dart generation equals the binding files.
- [x] Complete two max review rounds, ending with three clean reviews.
- [x] Audit production sources, mirror files, generated artifacts and runtime evidence.

Evidence: `/private/tmp/cera-api-plan34-native-loading`. Its `artifact-audit.json`
verifies runtime inputs/artifacts, GPU ownership and example binaries. The two
post-runtime harness edits have identical Python ASTs and reconstruct to their
captured hashes; the other probe change is documentation. Generated Swift
whitespace is retained to match the generator. No findings are skipped or open.

The initial `run-drqj17ip` metadata failure is preserved: object-only UniFFI name
overrides did not rename constructor owners. Actual Probe-prefixed Rust object
names fix that; shared record/error metadata aliases leave WASM names intact.
Two protocol failures were retried. Sandbox-blocked Ruff/Dart checks passed on
approved retries. The first regeneration report preserves its blocked Dart patch;
`regeneration-equality.json` records successful comparison after the retry.

Thirty-one bounded increments are complete (29 core, two Leap). P1 remains open;
CPU WASM production publication is next. Plan35 is the next unused plan. Broader
chat/recovery, KV performance, migration and Leap release gates remain below.
No commits or pushes.

## Plan33 complete (2026-09-12T17:44-0700)

The selected Rust loading types are public through `cera` and `cera::engine` in
this checkout. Existing constructors remain. The loader implementation and its
retained forwards preserve the prior Rust tokens except public visibility,
comments and formatting (`promotion-body-check.json`). Generated candidate
bindings now consume the public root imports; preparation no longer exposes
private loader declarations. The three remaining mirror adaptations are binding
shared engine storage and session-config observations.

- [x] Export and document the public loading types and retained operations.
- [x] Switch external consumers to public imports and remove visibility staging.
- [x] Add and execute the public Rust completion example and session walkthrough.
- [x] Pass 47 Swift/47 Kotlin/26 Node cases (`run-pg2241hc`).
- [x] Pass 70 core loading regressions; one existing platform test remains ignored.
- [x] Pass the no-default-features check, scoped lint/docs/declaration/harness gates.
- [x] Complete two max review rounds, ending with three clean reviews.
- [x] Complete core all-targets Clippy directly in the worktree and finalize evidence.

Evidence: `/private/tmp/cera-api-plan33-public-loading`. The initial mirror
all-targets Clippy attempt could not find `cera-cli/grammars/json.gbnf`, referenced
by a core integration test. The direct-worktree retry passes using a separate
target at `/private/tmp/cera-plan33-core-target`. Native/WASM Clippy and Rustdoc
also pass. Review corrected three stale private-loader descriptions; no findings
were skipped or remain open. Source/artifact audit records the README and one
example doccomment as the only post-runtime input changes. Production foreign
loader publication remains next; P1 as a whole is open. Thirty bounded increments
are complete (28 core, two Leap). Next unused plan34. No commits or pushes.

## Plan32 complete (2026-09-12T17:16-0700)

The full production-reuse run `tests/api_loading/build/run-okqfmma1/results.json`
passes 47 Swift/47 Kotlin/26 Node cases. The final ownership additions also pass
47/47/26 in `/private/tmp/cera-plan32-consumers-yftsma87/results.json`, using
the full run's hash-verified binaries. That consumer-only rerun adds native
tokenizer/cache use after model release and CPU WASM tokenizer use after engine
release. `artifact-audit.json` records the four fixture/README changes since the
full build and verifies all production inputs, mirror files and artifacts.
The earlier `run-oj7fvfvw` report preserves the sandbox loopback failure.

The current change stages production cera-ffi and cera-wasm types, converters,
engine and Session objects. The mirror adapts engine storage to an Arc and adds
test accessors; all six adapted files reverse to their original source bytes in
the harness control. Native config covers all six source variants. Tests observe
actual stored session configuration, shared engine identity, parent release,
independent sessions, KV options, streaming, cancellation recovery and typed
native errors. CPU WASM preserves its existing fields and Error transport.
Evidence: `/private/tmp/cera-api-plan32-repair-ai84j6qw`.

- [x] Repair typed native consumer ownership and complete its runtime rerun.
- [x] Validate production record/conversion and shared engine/session reuse.
- [x] Add production session behavior and identity controls in Swift/Kotlin.
- [x] Complete CPU WASM session access and runtime checks.
- [x] Run scoped lint/docs/harness and declaration gates.
- [x] Complete final runtime revision, ignored-config mutation control and max review loop.

Four repair review rounds ended with three completed clean max-effort reviews.
Fixes cover production backend enum transport, README probe type names and the
WASM tokenizer property's use. Protocol failures and an incomplete third-round
fanout are recorded in `reviews.json`; none counts as a clean review. No findings
were skipped or remain open. All three final consumers reject the ignored-config
mutant at their actual stored-config comparison (`final-negative-results.json`).
The first mutant's temporary-lifetime compile failure is preserved separately.
Native/WASM Clippy and Rustdoc pass on byte-identical staged Rust sources;
ten harness tests, ten declaration controls and all scoped lint/docs gates pass.

P0-L's loading prototype gate is complete. Twenty-nine bounded increments are
complete (27 core, two Leap). P1 production loading is next. Functional KV tests
do not establish performance budgets; the native cache check proves the retained
method is callable and preserves live positions, not populated prefix-cache
reuse on this tiny fixture. Repeated greedy generation still needs another append
in the existing core path. Chat/recovery and device/Leap release gates stay open.

No commits or pushes have been made.

## Workspace

- Worktree: `/Users/dberrios/development/cera/worktrees/api-reshape`
- Branch: `refactor/api-reshape`; base `2477a6830dc5`; no commits or pushes yet.
- Main design: [API_RESHAPE_PLAN.md](API_RESHAPE_PLAN.md).
- Local-only devlog: `/Users/dberrios/development/cera/devlog/000341-refactor-api-reshape.md`.
- Numbered work plans: root `devlog/plans/000341-02-loading-contract-prototype.md`
  and `000341-03-leap-compatibility.md`; the export increment follows
  `000341-04-leap-export-probes.md`. The completed protocol/native increment follows
  `000341-05-leap-protocols-and-bridge.md`. Plans 06 and 07 are complete below;
  the latter is `000341-07-remote-loading-contracts.md`. Plan 08 is complete below;
  plan 09 is complete below. Plan 10 is complete; Plan 11 is complete below.
  Plan 12 is complete below; Plan 13 is complete below. Plan 14 is complete below. Plan 15 is complete below; Plan16 and Plan17 are complete below; Plan18, Plan19 and Plan20 are complete below; Plan21 is complete below; Plan22 is complete below; Plan23 and Plan24 are complete below; Plan25 is complete below; Plan26 is complete below; Plan27 is complete below; Plan28 is complete below; Plan29 is complete below; Plan30 is complete below; Plan31 is complete below; Plan32 through Plan42 are complete above; next unused sequence is 43.

## Current scope

Implementation is authorized. Start with P0-L, then advance the reviewed gates
in bounded changes. Swift and Kotlin Leap SDK compatibility was added during
this work. Use published artifacts rather than assuming the archived docs are
current. Keep Cera's Session free of application history; the Leap adapter owns
its Conversation history.

## Plan31 — foreign multipart defaults

Completed 2026-09-12T08:18-0700. Started 2026-09-12T06:15-0700. Root plan: `devlog/plans/000341-31-foreign-multipart-defaults.md`.
Evidence: `/private/tmp/cera-api-plan31-58bmipsp`.

- [x] Add candidate native enum and CPU WASM factories for Text/Audio/Other.
- [x] Add both-builder consumers for full payloads, errors and parent release.
- [x] Rebuild native/WASM and execute 39 Swift/39 Kotlin/21 Node cases.
- [x] Complete scoped validation and cross-language text-only negative control.
- [x] Complete max three-reviewer loop.
- [x] Record final evidence and next steps.

This bounded increment takes the multipart-defaults gap first. The production
native config and shared engine/full Session access proofs remain subsequent
work; P0-L stays open. No production APIs changed. The concrete
[defaults examples](../../tests/api_loading/README.md#multipart-generation-defaults)
run inside the complete consumers. The probe-only observation reads actual core
manifest defaults, not the supplied input record. Host text-model execution does
not prove audio decoding or device packaging. Earlier runtime reports are
historical; this increment has fresh passing binaries and results. 28 bounded increments
are complete (26 core, two Leap); no major phase closed. Next unused32. No commits or pushes.

Passing report: `tests/api_loading/build/run-wapr_n1d/results.json`, 21 command
expectations including two deliberate E0004 rejections. All 39 Swift/39 Kotlin/
21 Node cases pass. Native/WASM Clippy and Rustdoc pass with warnings denied;
nine scoped checks, ten harness tests, ten declaration controls and Ruff2 pass.
The document audit covers 31 files/370 targets/67 anchors. Two max-effort
rounds with three reviewers each end clean. One stale handoff-next-steps finding
was fixed; three tool interruptions were resumed to completed reviews. No
findings were skipped or remain open. No source reload or live KV mutation was added.
The negative-control workspace initially selected stable Rust outside the repo;
that setup failure is preserved. Retry uses the same verified nightly compiler
and repository Cargo settings, with independent output artifacts. All three
consumers reject the text-only mutant at their retained-defaults comparison;
Swift also verifies the deliberately selected control library path. All 15
passing-run artifacts remain unchanged. Negative evidence is in
`negative-results.json`; the first setup failure remains separately preserved.
`artifact-audit.json` verifies 402 core, 417 mirror and 27 probe files, 24 Cargo
configuration paths, all 15 positive artifacts and the isolated negative
artifacts. `final-pass.json` records final hashes and the four completion-only
document updates after review. No active builds or reviews, commits or pushes.

## Plan30 — target API retention

Completed 2026-09-12T06:11-0700. Started 2026-09-12T04:35-0700. Root plan: `devlog/plans/000341-30-target-api-retention.md`.
Evidence: `/private/tmp/cera-api-plan30-b9v9rb1y`.

- [x] Inventory existing core/native/CPU WASM/WebGPU operation homes and payloads.
- [x] Select additive loading/config/shared-engine/session signatures.
- [x] Add declaration check and mutation controls for reviewed existing APIs.
- [x] Complete lint and document checks; verify surviving historical runtime evidence.
- [x] Complete max three-reviewer loop and durable checkpoint.

The [target retention map](API_RESHAPE_TARGET_RETENTION.md) records 69 declaration
groups: 300 methods/functions/constants and 34 records/enums and three callback traits. The runnable
[declaration checks](../../tests/api_contracts/README.md) pass all ten controls.
This audit finds three concrete candidate differences: native production uses typed
EngineConfig/BackendPreference, while the probe uses a string config; the selected
shared CeraEngine and full SessionConfig/production Session access is not yet
proved through the new foreign model; and multipart defaults currently carry only
Text sampling fields, omitting Audio-specific fields and Other raw JSON. Plan31
must prove all three contracts before P0-L closes.
No production code/signatures change or Rust build runs in this increment.
Twenty-seven bounded increments complete (25 core, two Leap); no major phase
closed. No commits/pushes.

Plan29's runtime audit passed earlier in this increment. At the later
2026-09-12T06:01-0700 checkpoint, its original native library, binding generator
and WASM build artifact were absent from `target/api-loading/run-gxemqvwz`.
No cleanup ran in Plan30. The 12 remaining generated/consumer artifacts, input
fingerprints and 33 Swift/33 Kotlin/15 Node result records remain intact; those
runtime results are historical. The failed full-artifact audit is preserved in
the evidence directory alongside a separate audit of the surviving evidence.
Rebuild before another binding runtime claim. About 44 GiB is now free.

Validation: declaration baseline and ten controls pass; Ruff lint/format, the
five scoped checks and the document audit pass (31 files, 362 local targets,
63 anchors). All five inventoried production source fingerprints are unchanged.
Three max-effort rounds used three reviewers each; all three final reviews are
clean. Three findings were fixed: commented declarations, literal whitespace
normalization and the omitted non-text multipart-default contract. Three tool
interruptions were resumed to completed reviews; none counted as clean errors.
No findings were skipped or remain open. `final-pass.json` records final hashes
and the four documents receiving completion-only updates after review.

## Completed plan29 — native remote loading

Completed 2026-09-12T04:16-0700. Started 2026-09-11T20:46-0700.
Root plan: `devlog/plans/000341-29-native-remote-loading.md`.
Evidence: `/private/tmp/cera-api-plan29-dta0dh9n`.

- [x] Add candidate native BundleId/HuggingFace sources and shared repository config.
- [x] Add foreign progress adapter and actual loopback/cache/ownership consumers.
- [x] Update runnable examples and structured remote failure cases.
- [x] Complete generated Swift/Kotlin/Node runtime and scoped validation.
- [x] Complete source/artifact/document audits and max three-reviewer loop.

See the [remote example](../../tests/api_loading/README.md#native-remote-loading-example).
Both native builders cover explicit HF quant/revision selection, cached bundle
ID/quant, manifest-file/directory loads, missing repository, source/kind/assembly
errors and invalid bundle quant. Progress crosses reporting thresholds, ends at
600 KiB and stays silent on cache hits. The actual core repository and callback
survive caller handle release; a retained repository triggers another download.
Swift observes final callback release, while Kotlin closes native handles without
asserting JVM collection timing. Sessions generate [0,1,0] at position 5 after
all parent handles close. Repository observations remain probe-only; live CDN,
conversion strategy execution, async/device packaging and public promotion are
not established by these host fixtures.

Current passing consumer report:
`tests/api_loading/build/consumer-retry-rdgqz5ec/results.json`.
It combines 15 unchanged passing build/generation expectations from `run-gxemqvwz`
with six fresh consumer commands, for 21 expectations including two deliberate
compiler rejections. Swift/Kotlin each pass 33 cases; Node passes 15. The original
report remains failed at the initial Swift fixture assertion. The guarded retry
verifies unchanged core/mirror/config/native/generated/WASM inputs and artifacts;
only four consumer/fixture inputs changed, and all remote stores are fresh.
This is verified artifact reuse, not a claim that the failed original run passed.

Ten harness tests, six scoped checks, seven Python files under offline Ruff,
workspace formatting and native/WASM Clippy/Rustdoc with warnings denied pass.
The evidence audit covers 402 core, 416 mirror and 26 probe inputs, 24 Cargo
configuration paths, 15 artifacts and six remote input files per language.
All 402 core inputs match completed Plan28; production bindings are unchanged.
Two max-effort rounds with three reviewers each end clean. Fixed two reviewed
fixture defects: cache paths omitted the loopback port, and kind/assembly fixtures
lacked pinned HF metadata. Runtime also caught Kotlin's old single-argument entry
point. Three review tool interruptions were resumed; none counted as clean.
No skipped or open findings. Only completion metadata changed after final review.

Evidence files: `checks.json`, `rust-checks.json`, `ruff-result.json`, `audit.py`,
`artifact-audit.json`, `production-core-audit.json`, `doc-audit.json`, `reviews.json`,
`retry-consumers.py` and `final-pass.json`. Earlier failed consumer retry
`consumer-retry-5ozlfiv9` is retained. Approved cleanup removed only 25 disposable
incremental directories; its exact inventory is in
`/private/tmp/cera-plan29-cache-cleanup/removed.json`. Latest free space is about
708 MiB; another large build requires additional disposable cache space.

Twenty-six bounded increments complete: twenty-four core and two Leap. No major
phase closed, public API promotion, commits or pushes. No active builds/reviews.
Next unused sequence 30: freeze the per-target binding retention map and exact
additive loading signatures, preserving supported legacy and async entry points.

## Completed plan28 — structured loading errors

Completed 2026-09-11T20:31-0700. Started 2026-09-11T18:53-0700.
Root plan: `devlog/plans/000341-28-structured-loading-errors.md`.
Evidence: `/private/tmp/cera-api-plan28-yeed02v3`.

- [x] Attach source/assembly context in the private core while retaining CeraError causes.
- [x] Map native/WASM inference/config/source/assembly categories with stable payloads.
- [x] Replace foreign error-text classification with both-builder payload assertions.
- [x] Complete minimal/mmap/remote core tests and generated native/WASM runtime.
- [x] Complete scoped checks, evidence audits and max three-reviewer loop.

See the [error example](../../tests/api_loading/README.md#structured-loading-error-example)
and [loading exit audit](API_RESHAPE_LOADING_EXIT.md). Assembly reports an
initialization phase and requested backend, not an internal failure reason.
Session probe errors remain Engine for later taxonomy work; original production
errors are unchanged. Twenty-five bounded increments are complete: twenty-three
core and two Leap. No major phase closed or public API promotion. No commits
or pushes.

Core loading tests pass 25 minimal, 36 mmap and 70 remote cases, each with one
existing ignored test. Full run `tests/api_loading/build/run-bijrgzj4/results.json`
passes 21 command expectations, including two deliberate compiler rejections.
Swift/Kotlin each pass 24 cases; Node passes 15. All generate [0,1,0] at position 5.
Ten harness tests, scoped lint/formatting, core and native/WASM Clippy, native/WASM
Rustdoc with warnings denied and artifact/document audits pass. The audit covers
402 core, 415 mirror and 22 probe inputs, 24 Cargo configuration paths and 15
artifacts. The production audit verifies 395 other core inputs unchanged; all
seven changed core files belong to the private test prototype.

Two max-effort rounds with three reviewers each end clean. One verified finding
was fixed: remote fixtures now assert their exact source label and require Source
phase, rejecting Assembly. The remote suite and a fresh full generated run pass
after the fix. No skips, open findings or review interruptions. Earlier runs
`run-s7cr0s75` and `run-iof3cf1e` remain historical. Evidence includes
`core-checks.json`, `checks.json`, `rust-checks.json`, `ruff-result.json`,
`artifact-audit.json`, `production-core-audit.json`, `doc-audit.json`, `reviews.json`
and `final-pass.json`. Only completion metadata changed after final review.

Next unused sequence 29: generated native remote source and repository/progress
ownership. The build volume has about 880 MiB free; reclaim disposable build
caches before another large build, preserving sources, models, reports and runtime
artifacts. No cleanup was performed in this increment. No active builds/reviews.

## Completed plan27 — native loading configuration

Completed 2026-09-11T18:42-0700. Started 2026-09-11T18:13-0700. Root plan: `devlog/plans/000341-27-native-load-config.md`.
Evidence: `/private/tmp/cera-api-plan27-dqucoh11`.

- [x] Preserve native u64 context/defaults/checked conversion and sentinel observation.
- [x] Add Swift/Kotlin boundary consumers through both builders and retained sessions.
- [x] Add wasm32 execution of the native helper, preserving web config representation.
- [x] Complete generated runtime and scoped validation/audits.
- [x] Complete max-effort three-reviewer loop and completion checkpoint.

See the [context example](../../tests/api_loading/README.md#native-context-configuration-example)
and [loading exit audit](API_RESHAPE_LOADING_EXIT.md). The wasm32 helper proves
32-bit conversion, not a complete native Android loader. Next28 should settle
structured loading errors; remote config/source and binding/target retention
remain loading prerequisites. Twenty-four bounded increments are complete:
twenty-two core and two Leap. No major phase closed, public API promotion,
commits or pushes.

Full run `tests/api_loading/build/run-45e5w89q/results.json` passes21 command
expectations, including two deliberate compiler rejections. Swift/Kotlin each
pass22 cases; Node passes13. Both native builders preserve defaults, zero,
capacity64/65,4294967320,u64::MAX and context1. Sentinel observations report64,
ordinary wide requests remain visible, and allocation stays bounded. All larger
profiles generate [0,1,0] at position5 after parent release; context1 rejects its
second token without advancing. The exact native helper passes wasm32 zero/
in-range controls and rejects both2^32 andu64::MAX; this does not establish a
complete32-bit native loader or Android packaging.

Ten harness tests, six scoped checks, offline Ruff, native/WASM Clippy and
Rustdoc with warnings denied pass. The artifact audit covers402core/415mirror/
22probe inputs,24Cargo configuration paths and15artifacts. One max-effort round
with three reviewers ends clean; no findings, skips or open issues. One review
tool interruption resumed and was not counted as a clean review. Evidence files:
`checks.json`, `rust-checks.json`, `ruff-result.json`, `artifact-audit.json`,
`doc-audit.json`, `reviews.json` and `final-pass.json` in the evidence directory.
Only completion metadata changed after review. No active builds/reviews.
Latest free space5.7GiB; no cleanup. Next unused sequence28.

## Completed plan26 — loading exit audit and native files

Completed 2026-09-11T18:04-0700. Started 2026-09-11T17:08-0700. Root plan: `devlog/plans/000341-26-loading-exit-and-files.md`.
Evidence: `/private/tmp/cera-api-plan26-6iwo3o1z`.

- [x] Reconcile loading prerequisites in the [exit audit](API_RESHAPE_LOADING_EXIT.md).
- [x] Add native ModelFiles record/source and resolved-manifest observation.
- [x] Add Swift/Kotlin consumers for all fields, both builders and error/ownership contracts.
- [x] Complete generated native/WASM run and scoped checks.
- [x] Complete max-effort three-reviewer loop and final evidence audit.

The first candidate gap closed by this implementation is native multipart files.
The next concrete gap is native LoadConfig width/default/zero semantics, followed
by remote source/repository/progress representation and the per-target retention
map. Structured loading errors are also a P0-L prerequisite; current generic
Engine.detail assertions prove ordering, not the final error taxonomy. See the
exit audit for later P1/P2 checks and independent chat/Leap gates.
No production exports or generated production bindings change in this increment.
Twenty-three bounded increments are complete: twenty-one core and two Leap.
No major phase is closed. Current full run:
`tests/api_loading/build/run-mvz7tlqs/results.json`, 21 expectations, 17 cases each
in Swift/Kotlin and 12 in Node; all produce [0,1,0] at position5. File consumers
also generate that result after releasing their loader/model parents. All eight
fields, explicit/inferred text, both builders, missing primary, Hotword and
unsupported inference ordering are covered. This does not execute optional
auxiliary weights or claim platform/performance coverage beyond the host probes.

Ten harness tests, six scoped checks, offline Ruff and native/WASM Clippy/Rustdoc
pass. The final audit verifies 402 core, 415 mirror and 22 probe inputs, 24 Cargo
configuration paths and 15 artifacts. Two max-effort rounds with three reviewers
each are complete; all final reviews return NO FINDINGS. Fixed one omission:
structured loading errors must remain an explicit P0-L prerequisite. No skipped
or open findings. Four tool interruptions were resumed, not counted as clean.
The first build92kjidgh failed on a probe accessor using absent Display; the
corrected as_str accessor passes the fresh full run. Both reports are retained.

Evidence records: `artifact-audit.json`, `checks.json`, `rust-checks.json`,
`ruff-result.json`, `doc-audit.json`, `reviews.json` and `final-pass.json`.
Only completion metadata changed after the final review. No active builds/reviews,
commits or pushes. Latest free space7.8GiB; no cleanup. Next unused sequence27.

## Completed plan25 — foreign hotword loading

Completed 2026-09-11T15:30-0700. Started 2026-09-11T15:10-0700. Root plan:
`devlog/plans/000341-25-foreign-hotword-loading.md`. Evidence:
`/private/tmp/cera-api-plan25-3dhx3t3j`. Scope: nine probe/documentation files,
with unchanged production API and generated production bindings.

- [x] Add `kws` fixture and Hotword payload checks in Swift/Kotlin/Node.
- [x] Require a list of case names; reject false-status maps and missing hotword.
- [x] Complete final formatted native/WASM run and negative classification control.
- [x] Verify source restoration and hash rebuilt artifacts before restored execution.
- [x] Ten harness tests and authored Swift/Kotlin/Node/Rust formatting checks.
- [x] Native/WASM Clippy and Rustdoc with warnings denied.
- [x] One max-effort round with three reviewers; all return NO FINDINGS.
- [x] Final artifact/document audit and progress/handoff checkpoint.

The final formatted `tests/api_loading/build/run-6bvxs_7w/results.json` passes
21 command expectations (two deliberate E0004 rejections), 13 cases each in
Swift/Kotlin and 12 in Node. All three produce tokens [0,1,0] at position5 after
parent release. `control.json` records the isolated kws-to-Vad mutation: all
three unchanged consumers reject the first dynamic Hotword payload assertion
(Swift-5, Kotlin1, Node1); restored source/build passes the full matrix again.
Swift/Kotlin DYLD traces verify the intended native library bytes. The initial
run-bdjf8mfc predates Kotlin formatting and remains historical.

The first control completed mutation/restoration, but its final audit incorrectly
assumed identical native hashes after rebuilding unchanged source. The corrected
control captures rebuilt outputs before runtime and verifies those hashes after
execution, separately checking byte-identical mirror restoration and unchanged
production sources. `control.json` contains current artifact hashes; the original
full-run native/generator hashes describe the pre-mutation build. `parser-control.json`
proves the old parser accepts false-status maps and the new one rejects them for
both native/Node case sets. Native/WASM Clippy and Rustdoc pass with warnings denied. All three max-effort
reviewers returned NO FINDINGS in one round. One contract-review tool interruption
was resumed; none counted as clean. No skipped or open findings.
Both loader methods must report Generative/Hotword/kws before unavailable Metal
construction and reject reuse. Header-only fixtures do not execute a hotword
detector. The negative control changes only the isolated mirror's classification.
Twenty-two bounded increments are complete: twenty core and two Leap experiments.
No major phase is closed. The final audit covers 402 core, 415 mirror and 22 probe
inputs, 24 Cargo configuration paths and 15 current artifacts. Current records are
`artifact-audit.json`, `checks.json`, `rust-checks.json`, `ruff-result.json`,
`doc-audit.json` and `reviews.json` in the evidence directory. Only completion
metadata in this handoff, the main plan and the binding audit changed after
review. No builds/reviews remain active; no commits or pushes. Latest free
space 11 GiB; next unused plan number 26.

## Completed plan24 — rebase and incoming API integration

Completed 2026-09-10T18:38-0700. Branch `refactor/api-reshape` is rebased onto fetched
`origin/main` at `2477a6830dc50c05b25fa19fe4681ebbc69f47c2` (0.5.6).
Root plan: `devlog/plans/000341-24-rebase-origin-main.md`. Evidence:
`/private/tmp/cera-api-plan24-pgifa5i4/update-2477a68`.

- [x] Preserve all original tracked/untracked work through both rebases. Recovery
  stashes `aa3f1e28b50724a5e066bdaf8bea2d02e82b0f7c` and
  `f48184098f6ac1d49de8bcf5c2d23d507534f127` remain intact; file/hash backups cover
  the original 162 files and the second pass's 160 working files.
- [x] Retain 0.5.6, publishing fixes, hotword detector/iterator records and methods,
  VAD lag-margin/allocation fixes, async multipart loading and upstream Whisper
  cooperative cancellation. Preserve the earlier GPU/Dart/ASR ownership fixes.
- [x] Classify `kws` as Hotword before generative assembly; memory, filesystem and
  loopback HF kind-mismatch controls pass. Inventory mutable stream ownership and
  add hotword to F5 while keeping standalone APIs supported throughout migration.
- [x] Preserve four mandatory synthetic Whisper tests plus the three upstream
  tests. Regenerate Swift/SPM, Kotlin, Python, C and Dart; all eight outputs match
  independent regeneration, including after the reviewed documentation fix.
- [x] Final-target core tests: default674, minimal591, remote/GPU/Metal838, plus
  four explicitly selected device tests. FFI reports48 passing tests, including
  three optional real-model tests that return early when their models are absent.
  Those three are not evidence of actual hotword/real-Whisper model execution.
- [x] Workspace/GPU/minimal Clippy, all-feature library Rustdoc, Cargo/build-support
  formatting and version drift checks. Dart analysis/format,28tests with3existing
  VAD fixture skips, GPU harness3tests and CI Python tests pass.
- [x] Actual final Swift/Kotlin consumers each pass51GPU ownership cases and11
  Whisper cases. Dart's six GPU/ASR cases pass through the new async multipart
  loader. These are macOS native synthetic correctness checks, not mobile,
  browser, quality or numeric performance results.
- [x] Three max-effort review rounds with three reviewers each; all final reviews
  clean. Fixed one contract finding: hotword offsets are detecting-window ends,
  with saturating pre-roll and a derived timestamp. Five tool interruptions
  resumed; none counted as clean reviews. No skipped or open findings.
- [x] Final source/artifact/restoration audits and current documentation links;
  whole-tree tracked diff is whitespace clean. No commits or pushes.

First target `901b54e` was fully validated before main advanced. Its reports live
in the parent evidence directory and remain historical. The second target's
`final-pass.json` records the full scoped pass. The only later Rust changes are
field documentation; `doc-contract-fix/runtime-equivalence.json` proves both
non-doc bodies unchanged. That directory contains regenerated binding hashes,
refreshed Rustdoc/Dart checks and current native consumer reports:
`cera-gpu-session-21lh2t2j/report.json` and `cera-whisper-k5ises2l/report.json`.
`final-artifacts.json` and `reviews.json` tie the final state to this evidence.

Kotlin coroutine cancellation frees Whisper's Rust future. The pinned Swift
wrapper does not propagate `Task.cancel()`; that parity gap remains a foreign
migration gate. Hotword event positions identify scoring windows, not acoustic
word boundaries. Current READMEs and the plan reflect these limits. Full CI and
remaining P0-L/platform/chat/recovery/warm-KV/Leap release gates remain open.

Approved cleanup removed only disposable Rust incremental caches after checking
no compiler was active; all built artifacts, source, stashes and evidence were
retained. Latest free space is about7GiB. Twenty earlier feature increments remain
complete; rebase integration does not close a major phase. Resume Plan23 next;
next unused plan number is25.

## Completed plan23 — native GPU binding ownership

Completed 2026-09-11T10:42-0700. Resumed 2026-09-10T21:02-0700 after Plan24. Started 2026-09-10T13:51-0700. Root plan: `devlog/plans/000341-23-native-gpu-binding-ownership.md`.
Evidence: `/private/tmp/cera-api-plan23-aai6g6ec`.

- [x] Bounded baseline and numbered plan.
- [x] Shared native runner and actual Swift/Kotlin ownership probes.
- [x] Explicit Metal/wgpu positive controls, CPU sharing and async retention.
- [x] Negative ownership control: both unchanged consumers fail at their first
  Metal Busy assertion; restored source/build passes 51 cases each again.
- [x] Three max-effort rounds, three reviewers per round; all final reviews clean.
- [x] Scoped checks, runnable examples, source/configuration/artifact audit and completion record.

Twenty-one bounded increments are complete: nineteen core and two Leap experiments.
No major phase is complete. Plan23 changes test/evidence infrastructure and
documentation, preserving production API signatures and generated bindings.
Plan24 supplied the current upstream schema; the reports below refresh native
runtime validation after the harness review fixes.
Current evidence: `/private/tmp/cera-api-plan23-aai6g6ec/resume-2477a68/config-fix`.
`cera-gpu-session-kfhqd156/report.json` records a fresh run of 51 cases per language.
`negative.py` and `negative.json` record the temporary Session acquisition bypass,
Swift failure(-5), Kotlin failure(1), exact loaded libraries and full positive
restoration. Round1 review found two harness defects: accepting a map of false
case statuses, and omitting shaders/build scripts from source fingerprints.
Round2 also found inherited Cargo configuration missing from that fingerprint;
the existing configuration snapshot helper now checks ancestor and Cargo-home
files, including absent paths. All three findings are fixed; native source/build
inputs and Cargo configuration match before/after the refreshed negative control.
Five GPU and ten loading-harness tests pass. Ruff lint/format, Rust formatting and
document/whitespace checks pass. `cera-whisper-cij12564/report.json` records a fresh
Whisper run with 11 cases per language after shared-helper formatting. Earlier reports
remain historical evidence. `artifact-audit.json` verifies 300 GPU source inputs,
14 Cargo configuration paths and current artifacts, fixtures and dependencies.
`../reviews.json` records all three review rounds: three actionable harness
findings fixed, no skipped or open findings, four orchestration interruptions
resumed. All three final reviewers returned NO FINDINGS. Only completion metadata
in this handoff and the main plan changed after that review.

This closes the macOS native foreign ownership slice. Async loaders are the
supported tested path; synchronous debug wgpu loading on a small Swift cooperative
stack remains unverified after the observed Naga stack overflow. Swift
`Task.cancel()` forwarding, mobile/browser behavior and numeric performance
budgets remain open. No builds/reviews remain active; no commits/pushes. Latest
free space is 15 GiB; no evidence or recovery stashes were removed. Next unused
sequence is 25.

## Completed plan22 — GPU session ownership

Completed 2026-09-10T13:20-0700. Root plan: `devlog/plans/000341-22-gpu-session-ownership.md`.
Evidence: `/private/tmp/cera-api-plan22-yugzhh7v`; `paths.json` identifies the
21-file scope and `baseline/` preserves pre-increment content. The
[GPU session examples](API_RESHAPE_GPU_SESSION_EXAMPLES.md) include Rust/Kotlin
lifetime guidance, native GPU tests and an actual portable Dart consumer.

- [x] Opaque model gate/lease; Busy before configuration; ownership survives
  reset/cancel and releases after Session resources; CPU sharing unchanged.
- [x] Five lifecycle/failure/race tests, plus real Metal/wgpu uncompressed and
  TurboQuant isolation, continuation, reset and successor-session controls.
- [x] Native Dart empty-session reseeding and failure-safe handle cleanup.
- [x] Independent cached ASR context preserves live conversation KV; failed load,
  invalid PCM recovery, concurrent reuse, independent outputs and deleted sources.
- [x] Text/VL GPU staging buffers released; audio backing retained for helper
  loading; private prefix caching disabled, with zero retained-entry/byte checks.
- [x] Failing-before/passing-after controls for the original Session gate, Dart
  reseeding, ASR helper, source retention and helper cache policy. Sources restored.
- [x] Final Rust checks: default662/minimal579/remote-GPU-Metal826, FFI41, four
  explicitly selected device tests, Clippy, Rustdoc, formatting and version checks.
- [x] Exact Rust guide snippet compiles; six-case Dart Metal consumer runs against
  a freshly staged native library. Kotlin snippet/binding/dependency hashes still
  match its successful compile. Dart analysis/format and 27 tests pass with three
  existing VAD fixture skips; the unchanged Dart sources retain that validation.
- [x] Five max-effort rounds, three reviewers each; all three final reviews clean.
- [x] Source/artifact hashes, document/whitespace audit and remaining phase checklist.

`final-validation.json` records twelve final affected Rust gates. The earlier
`expanded-validation.json` includes the unchanged Dart checks; `final-examples.json`
pins the freshly built native artifact, exact Rust snippet and Dart execution.
The final staged dylib SHA256 is
`693bbd8d113959a715d4c1e1fb35088117b6ff6080d44d82b44461930016f6b3`.
`negative.json`, `caller-controls.json` and `resource-controls.json` preserve the
negative/positive controls. Earlier build hashes remain historical, not evidence
for the final source. `round5.patch` is the reviewed implementation; completion
status updates afterward are limited to this handoff and the main plan.

ASR adds retained source backing for audio engines and first-use setup plus another
model/context allocation, kept until engine drop. It shares auxiliary weights,
never reopens paths or resets/copies conversation KV. Helper prefix caching is
disabled; active decode KV remains available. Ordinary Session construction still
allows one owner per built-in GPU model. No foreign method/record or generated
binding changes were needed. Standalone Whisper remains separate from LFM2-Audio.

The Dart probe explicitly exits after closing model handles because generated
callback listeners keep the VM alive; automatic VM shutdown is not proved.
Synthetic fixtures establish native correctness, not speech quality, throughput
budgets, browser ownership or Android/iOS device behavior. Full CI remains open.

Six distinct findings were fixed: README sharing, Kotlin lifetime, Dart reseeding,
ASR with an existing conversation, avoidable staging retention and hidden helper
cache entries. No findings were skipped or remain open. `reviews.json` records
nine resumed orchestration interruptions across five rounds; errors never counted
as clean reviews. All handwritten files are whitespace clean. Whole-tree
`git diff --check` still reports 66 preexisting generated UniFFI whitespace issues;
those generated files were not changed by this increment.

Twenty bounded increments are complete: eighteen core and two Leap experiments.
No major phase is complete. Next unused sequence23. No active builds/reviews,
commits or pushes remain. Latest disk13GiB free; check capacity before large builds.

## Completed plan21 — standalone Whisper Swift/Kotlin

Completed 2026-09-10T10:12-0700. Root plan: `devlog/plans/000341-21-whisper-swift-kotlin.md`.
Evidence: `/private/tmp/cera-api-plan21-_6ef6msg`. [Runnable Whisper examples](API_RESHAPE_WHISPER_EXAMPLES.md).

- [x] Reuse hotword PR423 commit c98176c's standalone Whisper loaders and UniFFI API.
- [x] Generate Swift, SwiftPM, Kotlin, Python and Dart outputs; eight output files
  match a fresh independent regeneration byte-for-byte.
- [x] Mandatory synthetic GGUF/PCM tests: file/byte retention, defaults/errors,
  nonempty sync decoding and shared/distinct async model calls.
- [x] Compile and run Swift/Kotlin consumers, 11 cases each, with exact `aaa`/`aa`/`bb`
  outputs. Run `cera-whisper-d1t2ojx9` verifies the staged native library actually
  loaded by Swift and pins Kotlin's library override to the same hashed copy.
- [x] Concrete root/crate README examples, recording helpers and a 24-file local
  link/fence audit (260 targets, 31 anchors, zero errors).
- [x] FFI 41 tests, default/minimal core Whisper 22 each, minimal all-target check,
  workspace all-target Clippy, all-feature workspace Rustdoc, Rust formatting,
  build-support formatting and version drift check. A separate no-mmap loader
  probe decodes `aaa`. Dart analysis and 27 tests pass; three existing VAD model
  fixture tests skip. These are scoped local checks, not the full CI matrix.
- [x] Two max-effort review rounds with three fresh reviewers per round; final
  round all NO FINDINGS. All three actionable round 1 findings fixed.

The API exposes `FfiWhisperModel.fromFile/fromBytes`, `transcribe/transcribeAsync`,
`languages/isMultilingual` and `FfiWhisperTranscribeOpts` plus its default helper.
It preserves the hotword PR's public names/record shapes. Reverse max-token
conversion saturates instead of truncating. File/byte constructors retain model
weights; each transcription owns its decode state. This CPU Whisper surface
neither uses nor changes generative Session live KV. F5's unified Whisper/VAD
loader remains future work; `CeraEngine.transcribe` remains LFM2-Audio.

Round 1 fixes reuse the existing abort-on-drop blocking helper for queued work,
keep Kotlin construction/use inside one IO context, and correct the copied
native dylib's install name/signature. The runner checks both link metadata and
Swift's runtime loaded path. The earlier `cera-whisper-xtfkt2t1` report is historical:
its Swift wrapper loaded the Cargo target path rather than the hashed copy;
that artifact fails the new linkage check. Final source/fixture/binary hashes
are verified in the current report. Three review tool orchestration interruptions
were resumed to completion (round 1 contracts; round 2 contracts and reuse).

One call consumes at most 30 seconds of 16 kHz mono float PCM. Synthetic weights
exercise preprocessing/encoding/decoding but deliberately ignore acoustic content;
no speech-quality/device performance claim follows. Cancelling the foreign future
can abort queued work; an already-running decoder continues because the FFI
options omit the core cancellation flag. Loading is synchronous, with off-thread
recording helpers. Kotlin cleanup is deterministic within `use`.

Full `git diff --check` reports 66 standard UniFFI-generated whitespace lines in
Swift/C-header/Python output. Those files remain byte-identical to regeneration
for the CI drift gate; handwritten changes, including new files, are clean.
No Android/iOS device, release/package publication or full-CI result is claimed.

Nineteen bounded increments are complete: seventeen core (including standalone
Whisper) and two Leap experiments. No major phase is complete. Next unused
sequence 22: GPU session-lifetime exclusion for model-owned live KV, preserving
CPU sharing and releasing ownership on Session drop. No GPU code was added by Plan 21.
No active builds/reviews, commits or pushes remain. Latest disk 24 GiB free;
check capacity before the next large build.

## Completed plan 20 direct HF snapshots

Completed 2026-09-10T07:28-0700. Root plan: `devlog/plans/000341-20-hf-gguf-snapshots.md`.
Evidence: `/private/tmp/cera-api-plan20-dyeb42gn`. Guide: [HF revision examples](API_RESHAPE_HF_EXAMPLES.md).

- [x] Baseline snapshots, plan/devlog and three original-production failures.
- [x] Pin direct GGUF/defaults/co-located companions to resolved repository metadata.
- [x] Separately pin known external DSpark repositories; new regression fails round1 code.
- [x] Five snapshot examples plus existing companion CPU execution, cache and error controls.
- [x] Concrete guide/three README updates and document audit.
- [x] Eleven final scoped gates and eight source/five binary hashes verified.
- [x] Three max-effort rounds with three fresh reviewers each; final round clean.

Public metadata structs and signatures remain unchanged. End-to-end HF discovery
uses the private snapshot parser introduced in Plan19. Files/defaults in the
primary repository use its validated full commit. Known external DSpark drafts
resolve their own repository commit before the manifest is returned. Co-located
companions require no extra metadata call. Explicit full commits must match
metadata, and uppercase SHA is normalized in file URLs. Pure metadata/manifest
helpers, explicit manifest loading, fixed catalog URLs and browser behavior retain
their existing semantics.

URL-derived download paths separate commits; old branch entries remain but cannot
satisfy pinned URLs. The primary example serves B at mutable URLs and in an old
branch cache entry while metadata resolves A. Identical-length A/B weights have
different values/defaults. A,B,B,A loads download each GGUF once, reuse verified
files without new progress and fetch each commit's defaults twice. A live CPU
session ingests two tokens before B loads, then three after parent release; its
position/logits match an independent A control. Invalid/null/short/nonhex SHA or404
cannot reuse the remote model or change existing bytes/progress. Explicit nested
file selection, authentication and mismatched commit errors are covered.

Round1's three reviewers found the known external DSpark fallback still used main.
The fix resolves that repository independently. A fourth example holds the primary
commit fixed while draft metadata follows A,B,B,missing,A. Exact bytes/cache paths
and observed proposals distinguish the two draft versions. A retained A session
first runs its drafter after all later loads; it does not claim prior live draft
generation. Each draft and the primary download once; five primary and five draft
metadata calls include the failed attempt. Unchanged-commit reuse emits no new
progress; missing draft SHA fails with no mutable fallback. This proves per-repo
consistency, not arbitrary primary/draft version compatibility.

The fifth example covers repo-ID loads through an HF_ENDPOINT with /hub/mirror
prefix. Co-located and external drafts retain that prefix in metadata, file and
cache paths, with exact bytes and observed execution after parent release. Three
metadata calls and four GGUF GETs stay under the prefix. Round2 code fails it with
HF draft URL lacks a file path (`round2-prefix-regression.log`). Normalization now
strips the configured base from generated draft URLs before parsing repository
information; public full-URL parsing remains unchanged.

Initial production fails all three original snapshot tests (wrong model bytes,
invalid metadata accepted, uppercase URL404); `baseline.log` is the initial sandbox
loopback denial and `baseline-behavior.log` the actual failures. The new external
case fails round1 code on wrong draft bytes in `round1-dspark-regression.log`.
Its initial integer type mismatch is preserved in `round1-dspark-compile.log`.
No other deliberate production mutation controls are claimed.

All eleven scoped gates pass after the review fix: remote712/seven ignored,
minimalremote+mmap25, default657/seven, minimal574/one, remote-without-mmap603/one;
remote,gpu,metal all-target Clippy, no-default remote library Clippy, warnings-denied
Rustdoc, Cargo/build-support format and diff checks. `validation.json` and logs are
final; `*-initial.log`/`validation-initial.json` retain the first pass before the
external-draft fix; `*-round2.log` and `validation-round2.json` retain the second
pass before the mirror fix. `validation-artifacts.json` verifies eight Rust hashes, five
binaries and all logs. The historical minimal-remote all-target vision-test feature
gap remains; both relevant integration files match HEAD, with no failure rerun or
separate HEAD build. Full workspace/device/CI gates are not claimed.

Three max-effort rounds used three fresh reviewers each. Round1 found the external
draft gap; round2 found prefixed mirrors broke during generated draft URL parsing.
Both are fixed with failing-before/passing-after regressions. Contracts review had
two orchestration interruptions in round1, one in round2, and two in round3;
round3 reuse also resumed one interruption. All resumed to completion. Round3 is clean across all lenses, with no open/skipped findings.
`round1.patch`, `round2.patch` and `round3.patch` preserve reviewed snapshots;
`final.patch` adds completion records. No Rust code changed after round3.
Document audit is recorded in `doc-audit.json`.

Every direct load requires primary metadata, including cached/full-commit requests,
and one additional query if it selects an external draft. Conversion still makes
two source metadata queries. Generation defaults remain optional/best effort.
Missing SHA fails discovery before format selection. Local file loading remains
the offline path. Endpoints are trusted to honor commit URLs; no attestation,
cache GC or concurrent-writer coordination is added. Catalog/explicit-manifest/
browser pinning, device sharing, load costs, chat/KV budgets, public promotion and
Leap runtime/packages remain open. No session/KV code or generated consumers
changed; Plan16's generated report is historical. At the Plan20 baseline Whisper was core/CLI only in this worktree; Plan21 above
now ports the hotword PR's standalone Whisper wrappers. The unified loader remains F5.

Eighteen bounded increments are complete: sixteen core and two Leap export
experiments. No major phase is complete. Next unused sequence21. No active builds,
reviews, commits or pushes remain. Initial disk6.2GiB and latest27GiB free; existing artifacts were preserved.
Inspect capacity again before the next large build.

## Completed plan 19 upstream conversion revisions

Completed 2026-09-09T12:16-0700. Root plan: `devlog/plans/000341-19-conversion-upstream-revisions.md`.
Evidence: `/private/tmp/cera-api-plan19-9rch24wl`.

- [x] Public HF protocol references, baseline snapshots and three original-production failures.
- [x] Internal snapshot parser, explicit authentication and revision-specific metadata URL.
- [x] Version-two conversion identity and pinned input/progress URLs before cache/checkpoint reuse.
- [x] Four upstream fixtures and fifteen recovery cases, including legacy source-record migration.
- [x] Runnable conversion guide and three README updates.
- [x] Eleven scoped checks, source/artifact and document audits.
- [x] Two max-effort rounds, three fresh concurrent reviewers each; final round all NO FINDINGS.

The public converter resolves metadata on every call, before completed-cache or
checkpoint reuse. A private flattened snapshot parser preserves public
HfModelInfo fields/signatures, validates a full 40-character hexadecimal commit,
rejects mismatch with an explicitly requested full commit and honors the supplied
conversion authentication option. Shared metadata retry/error handling remains.
Non-main model metadata uses `/api/models/<owner>/<repo>/revision/<revision>`,
matching the official Hub client. Every config/tokenizer/template/defaults/shard
header/tensor download and conversion progress URL uses the resolved commit.

Version-two requests bind requested and resolved revisions plus existing output
options. A changed commit restarts partial output or refreshes completed output;
an unchanged commit preserves verified reuse. Cache paths still use the requested
revision, and legacy receipts/checkpoints without source identity rebuild once.
Missing/invalid SHA or failed resolution returns an error before artifact mutation
and cannot serve stale remote output. Old bytes remain available for local loading
or a later successful resolution. Credentials are not written to records.

The upstream fixture serves A/B weights with identical SafeTensors headers,
shapes and lengths. Mutable main URLs serve B while metadata resolves A; all
actual inputs must use A. Another test stops after five A tensors, resolves B,
restarts all eleven B tensors, reuses B without progress, then refreshes A and
reuses it. The retained B model still executes against independent B weights
after replacement; its session is created afterward and continues after parent
release. This does not prove a live session survived the file replacement event.
The broader API/KV tests remain separate evidence.

Other cases reject null/short/non-hex SHA and HTTP404 without changing existing
bytes or progress, recover valid metadata and reuse A, exercise named release
and explicit full-commit URLs with metadata/file authentication, reject a full
commit mismatch before creating cache output, and migrate legacy records once.
Fifteen checkpoint cases retain all previous local corruption, exact boundary,
repeated interruption and output/execution controls. Exact HTTP/progress counts
verify fresh conversion, partial restart and unchanged-commit reuse.

Original production failed all three initial upstream tests: wrong A weights,
stale-cache acceptance and absent checkpoint source identity. The last failure
is a missing-record assertion, not a separate mixed-weight execution control.
`baseline.log` records the initial sandbox loopback denial;
`baseline-behavior.log` records the escalated failing run. `focused.log` passes
twenty remote-loading tests before final legacy/auth-file additions. Final suites
include those additions. No further deliberate production mutation runs are claimed.

Final scoped results: remote707/seven ignored, default657/seven, no-default574/one,
remote without mmap603/one and minimal remote+mmap fixtures20. Full remote,gpu,metal
all-target Clippy, no-default remote library Clippy, warnings-denied Rustdoc,
Cargo/build-support format and diff checks pass. Initial Clippy rejected one
fixture byte loop; fixed array chunks plus a remainder assertion resolved it,
and affected tests/gates reran. Initial failure logs/JSON are retained.
`validation.json` holds eleven final commands; `validation-artifacts.json` hashes
logs, binaries, ten source files and generated-probe manifests. The historical
minimal-remote all-target vision-test feature gap remains; its two files still
match HEAD, with no separate baseline build or repeated failing gate in Plan19.
Full workspace/device/CI validation is not claimed.

Round1 contracts review found the three READMEs overstated session retention
across model replacement. Their wording now distinguishes retained-model execution
from sessions continuing after parent release. Correctness/reuse were clean.
The guide's cpu_config() helper call was also corrected during the first review.
No Rust code changed between rounds; all ten source hashes match. Three fresh
round2 reviewers return NO FINDINGS; no open or skipped findings remain. One
round1 correctness tool-orchestration failure was resumed and completed.
`round1.patch`/`round2.patch` preserve review snapshots; `final.patch` adds completion
records. Document audit counts/anchors are recorded in `doc-audit.json`.

Loader discovery and conversion each query metadata: two requests even for an
unchanged converted-cache load, or one for direct low-level conversion. Explicit
full-commit requests also require resolution. There is no remote offline fallback;
local GGUF path loading remains available. Consolidating discovery and measuring
large-model/network costs remain open. No append/generate or live-KV code changed.
The endpoint is trusted to honor immutable commit URLs; cryptographic source
attestation, non-conversion GGUF pinning, concurrent conversions/platform sharing,
other formats, live CDN/device behavior and all broader API/chat/Leap gates remain.
Native/WASM consumers do not exercise remote conversion and were not regenerated;
Plan16's report remains historical evidence for its recorded snapshot.

Seventeen bounded increments are complete: fifteen core and two Leap export
experiments. No major phase is complete. Next unused sequence20. No active builds,
reviews, commits or pushes remain. Initial disk6.1GiB and later6.5GiB free; inspect
capacity before another large build. Existing artifacts were preserved.

## Completed plan 18 checkpoint prefix integrity

Completed 2026-09-09T04:22-0700. Root plan: `devlog/plans/000341-18-checkpoint-prefix-integrity.md`.
Evidence: `/private/tmp/cera-api-plan18-_mm_l1hi`.

- [x] Original-production regression: partial payload corruption survives into completed output.
- [x] Incremental output hashing and exact saved header/layout, tensor-boundary and byte verification.
- [x] Fourteen recovery cases with independent weights, retained execution and HTTP/progress counts.
- [x] Repeated interruption at five/ten tensors, verified full-prefix digests and final one-tensor resume.
- [x] Runnable conversion guide and root/crate/probe README updates.
- [x] Eleven scoped validation gates, source/artifact and document audits.
- [x] One max-effort round, three fresh concurrent reviewers; only documentation wording nitpicks, applied.

The existing public converter uses private helpers to hash accepted output bytes,
including GGUF header and alignment padding. Checkpoints save a cloned incremental
SHA256 state after flushing, so they do not reread the growing file every five
tensors. Resume regenerates the header/layout into a hashing sink, checks the
exact boundary implied by completed tensors and hashes the saved prefix with a
64 KiB buffer. Truncation and writing use that same verified file handle. Missing
or malformed integrity fields, mismatched options/layout/boundaries and corrupt
prefix bytes restart conversion. Plan17 completed-cache receipts, public
signatures and dependencies remain unchanged.

The fourteen-case matrix covers valid resume, wrong total count, legacy request,
changed overrides, payload/header damage, short prefix, missing/invalid/wrong
prefix digest, missing/wrong layout digest, wrong completed count and wrong byte
boundary. Each retry restores exact independent F32 weights and executes through
position seven after parent release. The zero-boundary case supplies the matching
empty-prefix digest, proving the boundary check matters. A valid five-tensor
checkpoint downloads only six more tensors; restarts download all eleven.
Both discard an added 128 KiB tail and produce the exact aligned output length.

The repeated example stops at five and ten tensors, independently hashes the
actual temporary bytes at each interruption, preserves the prior prefix and
excludes appended tails. Final loading downloads one tensor and emits two progress
events. Exact HTTP ranges prove all eleven tensors are fetched once across three
attempts, with six shard-header requests. The helper test covers partial writes,
a prefix crossing the 64 KiB buffer, ignored tail and short-read failure.

Original production failed the new payload-corruption case on wrong
`blk.0.ffn_up.weight` bytes (`baseline-behavior.log`). The first sandbox run failed
on loopback permissions (`baseline.log`); the escalated run reached the actual
regression. `initial-focused.log` records six passing conversion tests after the
initial fix; `focused.log` records seven after repeated-resume coverage. No
deliberate mutation-control runs beyond the original-production failing test are
claimed. All fourteen recovery cases pass in the final suite.

Final scoped results: remote 703 passed/seven ignored, default 657/seven,
no-default 574/one, remote without mmap 603/one, minimal remote+mmap fixtures 16.
Remote,gpu,metal all-target Clippy, no-default remote library Clippy,
warnings-denied Rustdoc, Cargo/build-support format and diff checks pass.
`validation.json` records all eleven commands; `validation-artifacts.json` hashes
logs, test binaries, four Rust source files and generated-probe manifests.
The recorded minimal-remote all-target vision-test feature gap from Plan17 remains;
its two integration files still match HEAD exactly. Plan18 neither reran that
historical failure nor built a separate baseline. Full workspace/device/CI gates
are not claimed. Native/WASM loading consumers exclude remote and were not rebuilt;
Plan16 reports remain historical evidence for their recorded snapshot.

One max-effort round used three fresh concurrent reviewers. Correctness and reuse
report NO FINDINGS; contracts identified only wording that confused the public
converter with its private helpers. Both documents are corrected. No source code
changed after the reviewed `round1.patch`; its four source hashes match final
code. `final.patch` adds wording and completion records. No open or skipped
findings remain. The final document audit covers all API reshape Markdown files
and the four related READMEs; exact target/anchor counts are in `doc-audit.json`.

New hashing runs during conversion/checkpoint resume, never append/generate.
This detects accidental local corruption; it does not authenticate files or
establish power-loss durability. Immutable upstream pinning and changed source
weights with identical metadata/layout remain open, as do concurrent conversions,
platform sharing, broader formats/live CDN/device execution and large-model cost.
Chat/recovery/live-KV performance, public API promotion/migration, actual Leap
Swift/Kotlin runtime/packages and PartII capabilities remain separate phases.

Sixteen bounded increments are complete: fourteen core loading/ownership and two
Leap export experiments. No whole major phase is complete. Next unused sequence
is19. No active builds/reviews, commits or pushes remain. Disk capacity was 14 GiB
before this validation; inspect it before another large build.

## Completed plan 17 converted cache integrity

Completed 2026-09-09T03:22-0700. Root plan: `devlog/plans/000341-17-converted-cache-integrity.md`.
Evidence: `/private/tmp/cera-api-plan17-dw124ja8`.

- [x] Completed-cache receipts verify exact GGUF and manifest bytes plus conversion options.
- [x] Same-size corruption, truncation, changed defaults and missing/malformed record repair.
- [x] Verified sidecar repair, interrupted repair, ordered overrides and checkpoint restart.
- [x] Retained original/repaired execution and exact HTTP tensor-download/progress controls.
- [x] Runnable conversion guide and three README updates.
- [x] Eleven scoped validation gates and document/source/artifact audit.
- [x] Two max-effort rounds with three fresh reviewers each; final round all NO FINDINGS.

A private versioned receipt binds the completed model/manifest hashes to owner,
repository, requested revision, quantization, strategy and ordered tensor overrides.
Each converted-cache hit streams a full GGUF hash and hashes the exact manifest.
Missing or damaged receipts force conversion, including a one-time rebuild of
legacy caches. A verified output can repair a missing/stale SHA sidecar without
fetching tensors. Receipt invalidation precedes repair, and a unique temporary
receipt is renamed into place only after model/manifest publication. Cancellation
leaves no completed receipt. Checkpoints require the same request descriptor;
legacy or option-mismatched checkpoints restart, while identical requests still
resume six tensors after the existing five-tensor checkpoint.

Three new integrity tests failed against original production code: wrong tensor
type after override change, no conversion progress after same-size corruption,
and cancelled repair bypassed. Six focused conversion tests then passed. Real
F32 weights/defaults are restored; F16 overrides verify actual types and numerical
values; original mapped and repaired models execute against independent controls
through position seven after parent release. The manifest-damage test now checks
that the parsed default changes to 0.95 before repair restores 0.25.

Round 1 contracts review caught an ignored manifest key in that test. It was fixed
and the affected full-remote/minimal-remote fixture tests, all-target Clippy and
format/diff checks were rerun. Production code did not change after round 1.
Correctness/reuse found no issues; all three fresh round 2 reviewers report NO
FINDINGS. No open or skipped findings remain. One contracts tool-orchestration
error was resumed, and thread capacity required sequential reviewers.

Final scoped checks: remote: 701 passed/seven ignored, default: 657/seven,
no-default: 574/one, no-default remote without mmap: 602/one, and minimal remote+mmap
fixtures: 15. Remote,gpu,metal all-target Clippy, no-default remote library Clippy,
warnings-denied Rustdoc, Cargo/build-support format and diff checks pass.
An extra no-default remote all-target Clippy attempt failed on unchanged vision
integration tests gated only on remote but calling mmap/vl-preprocess APIs.
Those files match HEAD; no separate baseline build was run. Its failure log and
scope adjustment are retained. Full workspace/device/CI gates are not claimed.

`validation.json` records the eleven final commands. `validation-artifacts.json`
hashes logs/test binaries and explains which checks reran for the test-only fix.
`round2.patch` and six source hashes preserve the reviewed increment; `final.patch`
adds completion records. Document audit passes across 20 Markdown files, 230 local
targets and 27 anchors. Native/WASM generated loading probes exclude remote and
were not rebuilt; Plan16's generated report remains historical evidence for its
recorded snapshot. No fresh foreign conversion execution is claimed.

The added full-file read happens on converted load, never append/generate. Large
model/device load-time budgets remain open. These local receipts do not pin a
mutable upstream revision, authenticate an artifact or validate partial checkpoint
contents. Source changes during resume, concurrent same-cache conversions and
platform sharing remain open, along with chat/recovery/warm-KV performance,
public API rollout and actual Leap Swift/Kotlin runtime/packages.

Fifteen bounded increments are complete: thirteen core loading/ownership and two
Leap export experiments. No whole major phase is complete. Next unused sequence
is 18. No active builds/reviews, commits or pushes remain. About 14 GiB free was
available after validation; inspect capacity before another build.

## Completed plan 16 named cache identities

Completed 2026-09-08T19:54-0700. Root plan: `devlog/plans/000341-16-named-cache-identities.md`.
Evidence: `/private/tmp/cera-api-plan16-qjqb6osq`.

- [x] Versioned loaded-byte identities, retaining caller and backend/KV namespaces.
- [x] Lazy once-only CPU cold identity outside cache locks; no append/generate hashing.
- [x] GPU source-time identity, complete DSpark backing and exact Metal mapping ownership.
- [x] Four CPU contracts, two GPU source/ownership checks and direct Metal execution.
- [x] Four deliberate regressions rejected; all production sources restored and verified.
- [x] Runnable cache guide/README links and measured identity cost.
- [x] Fifteen scoped core gates, fresh generated consumers, six mirror/example checks.
- [x] Two max-effort rounds with three fresh reviewers each; final round only wording nitpicks, applied.

Named built-in caches include a SHA256 digest of all ordered GGUF backing buffers,
framed with source count and lengths. CPU hashes once on first cold configuration,
then takes cache/tag locks and reads the current compression tag. Unchanged reload
restores a two-token cold prefix and computes only its suffix; changed late FFN
weights at the same path compute the full prompt and cannot restore stale state.
Old path-only files are ignored and preserved. Anonymous models remain warm-only.

Round1 found three actionable gaps, all fixed: Metal reopened the path after
parsing, DSpark omitted its base weights, and CPU hashing held cache locks. Metal
now clones the exact parsed Arc mapping. The additive provided public
`GpuWeightSource::cache_identity_sources()` hook includes all built-in sources;
DSpark supplies draft and base, while custom sources default to caller-managed IDs.
The private ModelLoader remains unpublished. GPU hashing occurs once before source
release and adds a named load-time scan when disk-cache is enabled.

The four mutation controls reject path-only identity (one token computed versus
three), hashing under locks (session creation blocked), DSpark base omission
(unchanged digest), and Metal reopening (execution matches replacement weights).
`review-controls.json` records commands and restored hashes. Direct Metal execution
passes outside the sandbox; sandbox GPU discovery had returned no device. This
proves tested backing ownership, while shared GPU conversation isolation remains open.

Core validation: remote 698 passed/seven ignored; default 657/seven; no-default
574/one; minimal disk-cache+mmap ownership 12; minimal remote+mmap 12; remote without
mmap loading 24/one; named gpu,metal six/one. Full, minimal and GPU/Metal-without-disk
Clippy, warnings-denied Rustdoc, Cargo/build-support formatting and diff checks pass.
`validation.json` records all 15 commands. The direct Metal test and audio fixture
export are additional successful manual runs. These are scoped gates, not the full
workspace/device/CI matrix.

Supported Apple/Unix aarch64 builds enable SHA256's ARM acceleration. Cargo.lock
adds only sha2-asm 0.6.4, using existing cc. No-default dependency inspection excludes
sha2. The release helper scans 67,108,992 resident bytes five times: median 37,285 us,
minimum 36,501, maximum 44,967. A separate generated build was active. This measures
one-time identity work, not model loading, filesystem I/O or inference latency.

Fresh `tests/api_loading/build/run-83j85_o3/results.json` passes 21 expectations,
including two deliberate E0004 rejections. Swift/Kotlin each pass 12 runtime cases;
Node passes 11. All generate [0,1,0] at position 5. Native/WASM Clippy and Rustdoc
pass. The text walkthrough retains its session through positions 5 and 9; audio
produces 22,560 PCM samples at 24 kHz and final position 22 after model release.
Generated artifacts, mirror, consumers, fixtures, 21 probe inputs and two workspace
inputs verify unchanged. Of 390 Cera source entries, 388 match exactly; the only
two differences are the reviewed mapping comments in cera/Cargo.toml and
metal_lfm2.rs. Their exact replacements are recorded in `wording-fixes.json`;
all executable source and dependency values match the generated run. The report
is retained unchanged, and consumers were not rebuilt for comment-only corrections. Exact additional commands
are in `generated-checks.json`; `validation-artifacts.json` hashes logs and binaries.
Pre-fix run-srsy5tzn and previous increments remain historical snapshots.

The final max-effort review has no open or skipped findings. Correctness and reuse
report NO FINDINGS; contracts reported stale mapping wording in two comments, now
corrected. TOML values and executable Rust are unchanged, and warnings-denied
Rustdoc passes again. Capacity required sequential reviewers. One round1 and two
round2 tool-orchestration errors were resumed; all six reviews completed. A parallel round2 launch was rejected
by the thread limit, so the remaining reviewer ran sequentially. `round2.patch` preserves the reviewed
increment; `round2-code-sha256.json` records the pre-wording snapshot; `wording-fixes.json`
proves the two exact comment corrections. `final-code-sha256.json` records current
source hashes. `final.patch`
adds the completion records. The document audit checks 20 Markdown files, 227 local
targets and 27 anchors, with no errors.

Fourteen bounded increments are complete: twelve core loading/ownership and two
Leap export experiments. No whole major phase is complete. Remaining work includes
converted-cache/checkpoint source identity, broader formats/live CDN, device sharing,
chat/recovery/warm-KV performance, public API promotion/migration and the actual
Leap Swift/Kotlin runtime/packages. No active jobs, commits or pushes remain.
Next unused plan sequence is17; check available disk space before another build.

## Completed plan 15 SafeTensors conversion

Completed 2026-09-08T17:55-0700. Root plan:
`devlog/plans/000341-15-safetensors-conversion-contracts.md`.
Evidence: `/private/tmp/cera-api-plan15-f066p24d`.

- [x] Three executable conversion contracts and HTTP range fixture.
- [x] Independent F32/quantized weights, retained execution, cache/progress checks.
- [x] Checkpoint recovery, malformed input and final-artifact publication checks.
- [x] Runnable conversion guide and three README links.
- [x] Two first-round findings corrected; both deliberate regressions rejected.
- [x] All11 scoped validation commands and document audit pass.
- [x] Two max-effort rounds, three fresh reviewers each; final round all NO FINDINGS.

Seven configurations cover single/sharded SafeTensors, F32/F16/Q8_0/Q4_0,
revision and strategy cache separation, HTTP206 ranges and HTTP200 fallback.
Typed/dynamic/legacy sessions continue after parent release. Cancellation primes
a five-tensor checkpoint; retry resumes six or restarts all eleven for an invalid
checkpoint. Conversion progress measures GGUF output bytes, including alignment.

First-round findings: the absolute Q4_0 tolerance admitted zeroed matrices, and a
short checkpoint tail could be overwritten without truncation. Added relative L2
error below 15%, a 128 KiB tail beyond final output, and exact aligned file length.
The temporary length-assertion error from the pause is corrected: tensor offsets
are already absolute. HTTP response writes tolerate only BrokenPipe/ConnectionReset;
unexpected write errors still fail. Zeroed FFN is rejected at relative error 1;
missing truncation is rejected at 145120 vs 30688 bytes. Both scripts restored
production byte-for-byte; the converter SHA256 remains
`a60369b47153dee523100b8f47dcfa2aa84ea9689619056d40eb748d5a768a6e`.

Final checks: remote 694/six ignored; default 653/six; no-default 574/one;
minimal remote+mmap 12; remote without mmap loading 24/one. Full-feature and minimal
all-target Clippy, Rustdoc with warnings denied, Cargo/build-support format and
diff checks pass. `validation.json` records exact commands; `validation-artifacts.json`
hashes 11 logs and five test binaries. The focused baseline and both regression
logs/JSONs are retained. These are scoped core checks, not full workspace/CI gates.

The two-round max-effort review has no open or skipped findings. Limited capacity
required sequential launches; two first-round orchestration errors were resolved,
one second-round orchestration error was resolved, and a parallel launch was
rejected by the thread limit. `round2.patch`
is the final reviewed increment; `round2-code-sha256.json` verifies unchanged code
after completion records. `final.patch` includes these final status updates.

The document audit covers 19 Markdown files, 214 local targets and 23 heading anchors.
Production/foreign consumers are unchanged from the historical Plan13 snapshot;
seven existing README/test-module source entries differ, and new test modules lie
outside that older report. No fresh generated binding run is claimed.

Thirteen bounded increments are complete: eleven core loading/ownership and two
Leap export experiments. No whole major phase is complete. The converted cache
still trusts local manifest/file existence; checkpoint identity, broader formats,
live CDN, device ownership, performance and public API promotion remain open.
The actual Leap Swift/Kotlin runtime and packages are also open. Next unused
sequence is 16 for stable named persistent-cache identities, preserving anonymous
warm-only policy and live session KV. No active jobs, commits or pushes.

## Completed plan 14 remote companion execution

Root plan: `devlog/plans/000341-14-remote-companion-execution.md`.
Baseline and logs: `/private/tmp/cera-api-plan14-tmuigupg`.

- [x] Shared existing complete fixture builders and draft observation helper.
- [x] Four remote execution contracts and custom isolated HTTP route sets.
- [x] Runnable [remote guide](API_RESHAPE_REMOTE_EXAMPLES.md) and three README links.
- [x] Final default/minimal/remote suites, Clippy, Rustdoc and document audit.
- [x] One max-effort review round with three fresh reviewers; all NO FINDINGS.

Vision covers selection, numerical execution, retained PNG ingestion with
preprocessing enabled, and equal-length cache repair with a missing SHA sidecar.
Audio covers encoder input, dedicated decoder preference, tokenizer detokenizer
fallback, actual PCM and continuation after parent release. Draft covers HF,
remote assets in local JSON and preseeded public bundle manifests, including a
configuration override. Download failures remain fatal; downloaded invalid
optional projectors preserve text execution. No production behavior changed.

Plan 13's `run-b1_7do81` generated report is historical evidence for its recorded
snapshot. Plan 14 changes test module sources and docs; no fresh generated run
is claimed. Production loading/inference and foreign consumer sources are unchanged.
The current test source hashes therefore differ from that older report.

Completed 2026-09-08T16:48-0700. Full remote library: 691 passed/six ignored; default:
653/six; no-default: 574/one. Minimal remote+mmap: nine passed; remote without
mmap loading tests: 24 passed/one ignored. The manual audio exporter accounts
for one ignored test; five others predate these increments. Full-feature and
minimal all-target Clippy, Rustdoc with warnings denied, Cargo/build-support
format checks and diff checks pass. Commands and logs are in `validation.json`.
These are scoped core checks; full workspace/device/CI gates have not run here.

One max-effort round completed with three fresh reviewers; all returned NO
FINDINGS. None were skipped and no actionable findings remain. Thread capacity
required sequential launches. Three tool-orchestration errors required retries; all final reviews
completed. `round1.patch` captures the reviewed
increment; `reviewed-code-sha256.json` verifies unchanged code after status updates.

The documentation audit checks 18 Markdown files, 200 local targets and 21
heading anchors with zero errors. Production/source scope was checked against
Plan 13's 381-file source report, 21 probe inputs and two workspace inputs;
only seven existing README/test-module source files and the probe README differ.
New remote test modules are outside that historical report. No fresh generated
foreign-media execution, live catalog/CDN, conversion or performance is claimed.
About 1.3 GiB remained after validation; inspect capacity before more builds.

Twelve bounded increments are complete: ten core loading/ownership and two Leap
export experiments. No whole major phase is complete. Remaining P0-L gates
include conversion, device ownership, stable named cache identities and public
promotion. Chat/recovery/performance, the public refactor/migrations and the Leap
Swift/Kotlin runtime/packages remain open. No active builds or reviews remain;
changes are uncommitted and unpublished.

## Completed plan 13 audio loading

Started 2026-09-08T04:45-0700; completed 2026-09-08T05:39-0700.
Root plan: `devlog/plans/000341-13-audio-loading-contracts.md`.
Baseline, review snapshots and validation logs: `/private/tmp/cera-api-plan13-sg4a3peg`.
`round3.patch` records the final review snapshot; completion records were updated
afterward. `code-examples-final.patch` preserves the unchanged reviewed code/examples.

- [x] Complete CPU encoder, depthformer and detokenizer GGUF fixtures.
- [x] Six audio tests: precedence, failures, actual computation and retained sessions.
- [x] Same-length PCM/silence and same-code detokenizer history controls.
- [x] Four deliberate production regressions rejected and restored.
- [x] Runnable audio walkthrough, local fixture exporter and three README links.
- [x] Final default/minimal library and focused mmap/remote audio tests.
- [x] Core Clippy/Rustdoc, both walkthroughs and fresh generated consumers.
- [x] Three max-effort rounds with three fresh reviewers each; final all clean.
- [x] Source/artifact verification, document audit and updated remaining gates.

Default library: 653 passed/six ignored (five pre-existing, one manual exporter).
No-default: 574 passed/one ignored. Final minimal mmap and remote audio suites:
six passed/one ignored each. Full-feature and minimal Clippy, Rustdoc, formatting
and diff checks pass. No production loader/inference behavior changed.

The [audio walkthrough](API_RESHAPE_AUDIO_EXAMPLE.md) loads from multipart bytes,
releases the model, ingests PCM, then produces six text tokens and 22,560 PCM
samples at 24 kHz. Input consumes two positions; the final position is 22.
These are synthetic CPU weights, so the output does not establish speech quality.
The [loading audit](API_RESHAPE_P0_LOADING.md#plan-13-executable-audio-companions)
records source-specific fallback and ownership evidence. Explicit text skips
projector parsing; dedicated byte vocoders can still load. Filesystem output
requires audio mode and a dedicated vocoder. Decoder dimension mismatch removes
both decoder and detokenizer. Output state remains scoped to one generate call.

Plan 13 generated report: `tests/api_loading/build/run-b1_7do81/results.json`,
status passed, 21 expectations (two intended E0004 rejections), Swift/Kotlin 12
cases each and Node 11. All return `[0, 1, 0]` at position 5. Ten harness tests,
both separate Rust walkthrough execution/Clippy checks and four exact mirror
native/WASM Clippy/Rustdoc checks pass. All 381 source, 11 generated/native/WASM
and four consumer artifact hashes match, as do probe/mirror/configuration hashes.
`walkthrough.json` and `audio-walkthrough.json` record their separate evidence.
Exact target: `/Users/dberrios/development/cera/target/api-loading/run-b1_7do81`.
The [binding audit](API_RESHAPE_P0_BINDINGS.md#plan-13-audio-example-and-generated-consumer-refresh)
keeps Rust audio execution separate from foreign text consumer evidence.

Three review rounds completed with three fresh reviewers at max effort. All
three final reviewers returned NO FINDINGS. Two distinct findings are fixed:
an ambiguous `.into()` prevented the separate audio example from compiling,
and the fallback table omitted the explicit-text projector opt-out. The PCM
fixture also gained an explicit same-length silence control. None were skipped;
no actionable findings remain. Thread capacity required staggered launches.

Initial `run-mohzbp8g` predates the example fix and PCM assertion. Its generated
Cargo target was removed after approval to recover from ENOSPC; reports, logs
and source mirror remain. The final target is retained. About 6.7 GiB remained
after validation; check disk capacity before another isolated generated build.
The manual exporter creates a new temporary fixture directory and prints its
path; the guide gives the complete regeneration command.

Final document audit: 17 Markdown files, 183 local targets and 21 heading anchors; zero errors. The 11-file, 1071-line code/example delta matches the final reviewed snapshot, and source/artifact hashes match after completion records.

Eleven bounded increments are complete: nine core loading/ownership and two Leap
export experiments. No whole major phase is complete. Remote companion execution,
conversion, device sharing, named cache identity, chat/recovery/performance and
the Leap facade remain open. No builds, tests or reviews are active. Changes
remain uncommitted and unpublished. Next unused plan number is 14.

## Completed plan 12 vision and draft loading

Started 2026-09-07T22:12-0700; completed 2026-09-07T22:54-0700.
Root plan `devlog/plans/000341-12-vision-draft-loading.md`.
Baseline snapshot: `/private/tmp/cera-api-plan12-csjc23vh`; `increment.patch`
contains the final 1238-line reviewed increment. Next unused sequence is 13.

- [x] Build complete one-block vision and DSpark GGUF fixtures.
- [x] Prove byte/file/manifest/directory loading, precedence and optional failures.
- [x] Execute image/draft work through sessions after model release and Unix source deletion.
- [x] Prove numerical history dependence and reject three deliberate regressions.
- [x] Add runnable Rust/Swift/Kotlin examples and update three READMEs.
- [x] Pass feature tests, format, Clippy/Rustdoc and fresh generated consumers.
- [x] Complete three max-effort review rounds; all three final reviewers are clean.
- [x] Record source/artifact hashes, audit documentation and update remaining gates.

The [example guide](API_RESHAPE_EXAMPLES.md) links complete executable sources.
The Rust walkthrough loads once, releases the model, generates `[0, 1, 0]`,
appends a token to the same session and generates `[0, 1, 0]` again. Positions
advance from 5 to 9. The guide identifies private prototype versus released API,
Llama live KV versus LFM2 prefix caching, and the separate Leap workstream.
Maintain these examples as the public API and packages are implemented.

Final default library: 647 passed/five existing ignored. No-default: 568 passed.
Minimal mmap companion suite: six passed; final remote companion suite: seven.
Full remote suite: 681/five after local-listener escalation, before test-only
review fixes; no production behavior changed. Full-feature and minimal Clippy,
Rustdoc, formatting and diff checks pass. The [loading audit](API_RESHAPE_P0_LOADING.md#plan-12-executable-vision-and-dspark-companions)
records commands, numerical controls and source-specific behavior.

Current generated report: `tests/api_loading/build/run-s3z9znv1/results.json`,
status passed. It has 21 expectations (two intended E0004 rejections), Swift and
Kotlin 12 cases each, Node 11, matching `[0, 1, 0]` at position 5. Ten harness
tests, all four exact mirror native/WASM Clippy/Rustdoc checks and the separate
walkthrough execution/Clippy pass. All 377 source, 11 generated/native/WASM and
four consumer artifact hashes match after builds. `walkthrough.json` records its
separate source/binary/log hashes. Exact target:
`/Users/dberrios/development/cera/target/api-loading/run-s3z9znv1`.
The [binding audit](API_RESHAPE_P0_BINDINGS.md#plan-12-executable-examples-and-companion-test-refresh)
keeps the Llama consumer evidence separate from Rust vision/DSpark execution.
Earlier run-h8w8z40l and run-ai0cpnwm reports predate review corrections.

Three distinct findings are fixed: a weak draft-history control, overly broad
prefix-cache wording, and deleting mapped files on Windows. Three fresh reviewers
per round completed; all final results are NO FINDINGS. None were skipped and
no actionable findings remain. Thread-capacity limits delayed some launches;
all required reviews completed. Three deliberate production mutations were
rejected and restored byte-for-byte; production loader/inference behavior is
unchanged by Plan 12. Windows/device execution and performance remain unverified.

Final documentation audit: 16 Markdown files, 162 local targets and 16 heading
anchors, zero errors. The reviewed increment and all final source/artifact hashes
match after the completion records.

Ten bounded increments are complete: eight core loading/ownership and two Leap
export experiments. No whole major phase is complete. Audio encoder/vocoder/
detokenizer precedence, remote companion execution, conversion, device sharing,
stable named cache identity, chat/recovery/performance and the actual Leap facade
remain open. No builds, tests or reviews are active. Uncommitted and unpublished.

## Completed plan 11 anonymous-model cache policy

Started 2026-09-07T21:50-0700; completed 2026-09-07T22:07-0700.
Root plan `devlog/plans/000341-11-anonymous-cache-policy.md`.
Baseline snapshot: `/private/tmp/cera-api-plan11-_4jpa91k`; `increment.patch`
contains the exact source increment for review. No commits or pushes.

- [x] Reproduce numerical contamination with same-metadata/different-weight models.
- [x] Disable the cold tier for anonymous CPU/wgpu/Metal models at all nine sites.
- [x] Cover six memory constructors in None/F16, old-file protection and live state.
- [x] Prove warm cache hits and preserve named cold-tier behavior.
- [x] Pass default 640/five ignored, no-default 563 and disk-only 569 library tests.
- [x] Pass remote+gpu+metal all-target Clippy/Rustdoc and no-default Clippy.
- [x] Pass remote 674/five ignored and one clean max-effort review round.
- [x] Pass 35 generated consumer cases, ten harness tests and exact mirror lint/doc.
- [x] Verify source/artifact hashes and finish the documentation audit.

Anonymous models now ignore `cache_dir`; warm prefix caching and live session KV
remain available. Metal's partial embedding hash is removed. A shared private
constructor enforces this at initial setup, cache reconfiguration and compression
retagging. The prior disk fixture now uses a real model path and requires mmap;
anonymous and cache-policy tests still run with disk-cache alone. See the
[ownership audit](API_RESHAPE_P0_OWNERSHIP.md#plan-11-anonymous-model-persistent-cache-policy)
for commands and numerical controls. Supplied path/raw IDs keep existing cold
namespaces but are not content-validated. No performance or device runtime claim.

One max-effort review round completed with three fresh reviewers, all NO FINDINGS.
No fixes, skipped findings or actionable findings remain. Temporary thread
capacity delayed the third launch; retry succeeded. Remote tests required
escalation after the sandbox denied nine local HTTP fixture listeners; the
rerun passes. Current fresh generated report:
`tests/api_loading/build/run-d4jn9b_b/results.json`, status passed,
21 command expectations (two deliberate E0004 rejections), Swift/Kotlin 12 cases
each and Node 11. Tokens `[0, 1, 0]` and position 5 match after parent release.
Ten harness tests and exact native/WASM Clippy/Rustdoc pass; all 373 source,
11 generated/native/WASM and four consumer artifact hashes match after the checks.
Exact target: `/Users/dberrios/development/cera/target/api-loading/run-d4jn9b_b`.
The [binding audit](API_RESHAPE_P0_BINDINGS.md#plan-11-cache-policy-core-refresh)
records commands and limits: its Llama fixture does not execute LFM2 disk caching.
Prior generated run-56svuhvn belongs to Plan 10's source snapshot.

Nine bounded increments are now complete, seven core loading/ownership and two
Leap export experiments. No whole major phase is complete. No builds, tests or
reviews remain active; changes are uncommitted and unpublished.

## Completed plan 10 single-pass primary loading metadata

Started 2026-09-07T19:29-0700; completed 2026-09-07T21:39-0700.
Root plan `devlog/plans/000341-10-single-pass-loading.md`.
Increment baseline for review: `/private/tmp/cera-api-plan10-91f9lv82`. No commits or pushes.

- [x] Record plan and preserve the pre-increment source snapshot.
- [x] Prove duplicate parsing with per-thread parser-entry controls.
- [x] Share parsed multipart and filesystem primary assembly.
- [x] Verify source/error/auxiliary ordering and path behavior on macOS.
- [x] Run feature, integration, lint/doc and fresh generated-binding checks.
- [x] Complete two max-effort review rounds and record final evidence.

The regression controls rejected the previous code: typed multipart primary
parsing was twice, legacy bare-file parsing twice, and typed bare-file parsing
three times. All now parse once in the CPU fixtures. Memory readers, explicit
files, manifests and directories are also covered through legacy and typed/
dynamic entry points. A test-only per-thread counter observes actual parser
entries. Auxiliary parsing and backend-owned reopens are outside that count.

Shared multipart assembly accepts the classified primary; resolved filesystem
sources retain auto-detection's primary through assembly. A private callback
keeps typed kind errors before inference rejection and auxiliary resolution.
Explicit inference gates still precede primary opening. Original ModelFiles
paths and normalized manifest fields retain their distinct behavior. A Linux-only
invalid-UTF-8 path regression is added but was not run: macOS APFS rejected the
temporary filename even outside the sandbox.

Validation: default library 636 passed/five existing ignored; remote library
670/five; no-default loading 13; minimal mmap loading 23; minimal remote+mmap
loading 29; legacy integrations 16/four existing ignored. Remote all-target and
no-default lib/tests Clippy, remote Rustdoc, format and diff checks pass.
Remote loopback fixtures required sandbox escalation for local listeners.

Historical Plan 10 runtime report: `tests/api_loading/build/run-56svuhvn/results.json`,
status passed, 21 command expectations (two intended E0004 rejections),
Swift/Kotlin 12 cases each and Node 11, matching tokens `[0, 1, 0]` at position 5
after parent release. Ten harness tests pass. Native/WASM Clippy and Rustdoc
pass with warnings denied on this run's mirror. Its actual target is
`/Users/dberrios/development/cera/target/api-loading/run-56svuhvn`.
Plan 09's older reports remain historical evidence for its source snapshot.
See the [loading audit](API_RESHAPE_P0_LOADING.md#plan-10-single-pass-primary-metadata)
for commands and the remaining Linux/device/performance limits.

Two max-effort rounds completed with three fresh reviewers per round; all three
final reviewers returned NO FINDINGS. One distinct finding was fixed: the
Linux-only assertion omitted the `backend:` display prefix and now matches the
Backend payload directly. None were skipped and no actionable findings remain.
Five local parse/order tests, remote all-target Clippy and format/diff checks
pass after the fix. Fresh run-56svuhvn and ten harness tests passed again; all 372
Plan 10 source hashes matched the report and all 11 recorded native/generated/WASM
artifact hashes still match after lint/doc builds. Run-weeqz9_a predates the
test-only correction. Temporary thread limits delayed third-reviewer launches;
retries completed each round. No tests, builds or reviews remain active.

The final documentation audit checks 13 Markdown files, 103 local targets and
four heading anchors. Eight bounded increments are now complete across core
loading/ownership and Leap export experiments; no whole major phase is complete.

This is the core API refactor. No new production exports, inference kernels,
KV representation, Leap facade or model fine-tuning work is included.

## Completed plan 09 generated foreign loading contracts

Completed 2026-09-07T19:19-0700 after the final reviewer resumed and returned NO FINDINGS.
Root plan `devlog/plans/000341-09-foreign-loading-contracts.md` is complete as a
bounded loading-contract increment. Its original next sequence was 10, now complete.
No production code changed in plan 09. Include checklists in progress reports.

- [x] Isolated visibility-only mirror and source/tool/lock/artifact hashes.
- [x] Native UniFFI Swift/Kotlin and WASM Node binding generation.
- [x] External Rust wildcard, exhaustive rejection, attribute-removal pass and restored rejection.
- [x] Complete executable consumer case sets and matching generation after parent release.
- [x] Ten Python harness tests and descendant/library-selection negative controls.
- [x] Authored Rust/Python/Swift/Kotlin formatting/lint, Node syntax, root cargo fmt and git diff check.
- [x] Final native/WASM Clippy and Rustdoc in the current runtime mirror, warnings denied.
- [x] Five max-effort review rounds; three reviewers per round, all final NO FINDINGS.
- [x] Final local-link/status audit and plan 09 completion record.

Historical plan 09 runtime: `tests/api_loading/build/run-n2nckicv/results.json`,
status `passed`, 21 command expectations, including two deliberate E0004
rejections. Swift/Kotlin each execute 12 cases; Node executes 11. All produce
`[0, 1, 0]` at position 5 after parent handles release. Input/mirror/fixture/JNA,
Cargo configuration and generated artifact checks pass. Resolved lock SHA256:
`823b1c1db5389c1cc9fde2ed32fc056d785e81253310e33c51f71f617a7eda81`.
This run explicitly supplied conflicting `CARGO_BUILD_TARGET`,
`DYLD_LIBRARY_PATH` and `_JAVA_OPTIONS`, plus a relative wasm-bindgen path.
The report confirms the runtime overrides were removed and native commands used
`aarch64-apple-darwin`. Earlier reports t9xdn_d0, jt8p8mfg and pbm3sid5 predate
review fixes and are historical evidence only. The later run-wm79dj5d predates
the exclusive-target fix; run-n2nckicv is the final plan 09 result. Plan 10's
current source snapshot is validated separately above.

Final mirror: `tests/api_loading/build/run-n2nckicv/workspace`.
Generated bindings are sibling `native/` and `web/`; the native artifact is under
repository-root `target/api-loading/run-n2nckicv/aarch64-apple-darwin/debug/`.
Use `CARGO_TARGET_DIR=/Users/dberrios/development/cera/target/api-loading/run-n2nckicv`
for checks on this mirror. `--target` now names a parent directory; every run
creates a unique exclusive child as its actual Cargo target. Artifacts are hashed
at each build/generation boundary and checked again after execution. Do not
direct external builds into an active run's target. Set `CERA_GIT_SHA=loading-probe` for mirrored checks.
The runner selects actual native library/generator and WASM paths from Cargo
compiler-artifact records tied to the expected source files. Commands are in the
[probe README](../../tests/api_loading/README.md).

Five max-effort review rounds completed with three reviewers each; all three
final reviewers returned NO FINDINGS. Eight distinct findings are fixed: timeout
descendants, stale hardcoded artifact paths, missing Cargo config/flag provenance,
relative CLI paths, macOS library overrides, JVM overrides, Swift copy-on-write
weakening mutation evidence, and shared-target artifact replacement. None were
skipped, and no actionable findings remain. The interrupted final review resumed
without code changes and completed cleanly. Temporary thread-capacity errors
delayed launches; successful retries supplied the third reviewer in rounds four
and five. No sixth round was needed.

Source: `tests/api_loading/{prepare,commands,run,consumers,test_harness}.py`,
shared Rust records/helpers, native/WASM facade crates, external Rust match
consumers, and Swift/Kotlin/Node executable consumers. Generated/build outputs
are ignored. Ten tests cover visibility reversal to byte-equal Cera source,
strict diagnostic/result evidence, inventory/config/artifact drift, process
failures and descendant cleanup, environment isolation, and distinct target
children for two runs with the same parent. The descendant
regression specifically rejects the original cleanup. Swift allocates an
independent mutable buffer and verifies zeroing preserves the passed address.

Additional direct loading controls are in plan 09's run-n2nckicv
`library-selection.json` and four `swift-*.stderr` / `kotlin-*.stderr`
control traces. With an inherited override, Swift loads the older dylib; with the
controlled environment, its loaded library bytes match the current Cargo
artifact (Cargo's install name loads its identical copy under `debug/deps`).
Kotlin's inherited JVM option can bypass a deliberately invalid explicit library
path; the controlled environment correctly rejects that missing path. The full
runtime run uses the valid current library and passes all Kotlin cases.

Use `export PATH="/opt/homebrew/bin:$PATH"` and unset `JAVA_TOOL_OPTIONS`.
Full run: `python3 tests/api_loading/run.py --target /Users/dberrios/development/cera/target/api-loading`.
Pinned wasm-bindgen 0.2.117 defaults to the local wasm-pack cache; relative
`--wasm-bindgen` paths resolve from the invocation directory. JNA 5.16.0 defaults
to `/private/tmp/cera-leap-api-baseline/jna-5.16.0.jar`, verified against the public
pin in `tests/leap_compat/artifacts.json`. JDK is Zulu21.0.9; Kotlin2.4.0,
Swift6.3.3, Node24.9.0 and Rust nightly1.99. Ruff0.16.5 is cached through uvx;
its cache access requires sandbox escalation. Swift lint uses xcrun swift-format.

Consumers cover byte copy, dynamic clones, release of parent handles, CPU
append/generation, native path, all eight multipart fields represented, five
sampling overrides and all non-remote config fields observed, four wrong kinds,
unknown architecture, malformed data, unavailable Metal, invalid backend, both
consuming entry points after success/failure, and same-build future-kind accessors.
UniFFI0.31.2 error payload `message` conflicts with Kotlin's Throwable.message;
the candidate uses `detail`. Generated bindings are unmodified.
Auxiliary inputs are ignored/invalid fixtures; no real auxiliary/draft, GPU,
future-version ABI, warm-chat or KV performance claim is made. This is the core
API loading track; the actual Leap runtime facade remains C1.

[Binding evidence](API_RESHAPE_P0_BINDINGS.md), loading audit and main plan record
scoped results. No commits or pushes. No whole major phase is complete.

## Completed retained capabilities and execution ownership increment

Plan `devlog/plans/000341-08-retained-capabilities-and-ownership.md` at the
repository root is complete. This is the Cera API refactor track. Include checklists in progress updates and distinguish it from
Leap compatibility. The [operation/ownership audit](API_RESHAPE_P0_OWNERSHIP.md)
contains the complete map, exact source references, tests and remaining gates.

- [x] Create plan 08 and update branch devlog 000341.
- [x] Add 13 private retained forwards alongside create_session.
- [x] Inventory every engine operation and backend/auxiliary ownership boundary.
- [x] Strengthen CPU history and adapter sensitivity; add real cold-cache effects.
- [x] Pass full default library, focused feature checks, scoped Clippy and Rustdoc.
- [x] Complete formatting and all 86 local document links.
- [x] Complete two max-effort review rounds with three fresh reviewers each.
- [x] Record final evidence and remaining gates; plan 08 is complete.

New test-only code: `loading_prototype/retained.rs`,
`loading_prototype/tests/ownership.rs`, and its `ownership/{fixture,cache}.rs`
modules. The parent prototype only registers modules. No production code changed
in plan 08; preserve all previous increments' working changes.

Six ownership tests pass with defaults. Full default library: 631 passed,
five existing ignored. No-default loading filter: 14 passed. Minimal disk-cache
ownership filter: six passed without mmap/parallel. Remote all-target Clippy,
no-default lib/tests Clippy and remote Rustdoc with warnings denied all pass.
Commands and limits are in the ownership audit. No test/build process remains.
First max-effort round completed with three reviewers: two returned NO FINDINGS;
the third found a control-isolation gap and an omitted helper. Both are fixed.
The second fresh three-reviewer round returned all NO FINDINGS. No actionable
findings remain, and none were skipped. No tests, builds or reviews remain active.

The five original tests now use a separate deterministic Llama fixture whose
weights are sensitive to retained history. A continuation and a fresh session
ending in the same token must have distinct logits; the old loading fixture
failed this negative control and remains unchanged for previous evidence.
Generation can discard last_logits, so tests append a fixed token after nonzero
generation to inspect live continuation. The disk-cache-gated sixth test uses
isolated temporary directories and observes actual cold files, scoped deletion,
new-root configuration and retained live execution. It does not count cache hits
or measure speed.

Key retained restrictions: CPU LFM2's first effective compression tag remains
fixed after clears/reconfiguration/session drops; CPU Llama's None/F16 modes can
coexist. GPU text models own live KV/conv on the shared model, so per-call locks
are insufficient for independent conversations. Native GPU extraction has
separate guarded scratch; device execution remains unproved. Audio GPU decoders
have an exclusive generation-call lease with CPU fallback when busy. BERT uses
caller scratch as well as call-local buffers. DSpark clones its per-session
state. Capabilities report manifest declarations; successful auxiliary access
is separate. Pathless LFM2 cache identities can collide across different models
in a shared disk directory and require a declared restriction or correction.

Review fixes: the generation reference sessions no longer perform extraction.
Separate fresh sessions provide expected base/adapted hidden states, preventing
identical extraction-induced KV mutation from passing both sides. The public
`engine::init_dspark_drafter` is now explicitly retained in the helper inventory.
Full default library (631/five ignored), no-default loading (14), minimal disk
ownership (six), both Clippy configurations and document checks pass again.

No runtime Leap facade, real auxiliary/draft execution, device isolation,
performance or public API promotion is claimed. No commits or pushes.

## Completed first increment

- An internal loading prototype is in `cera/src/engine/loading_prototype.rs`,
  included only under `cfg(test)` from `engine.rs`. Bytes, borrowed readers,
  multipart bytes, typed/dynamic ownership, and early kind errors are implemented
  as a sketch. Production exports and session execution are unchanged.
- A deterministic one-block GGUF fixture exercises actual CPU assembly and
  generation. Six tests pass with default features and with no default features;
  all 37 tests matched by the engine filter pass with the remote feature.
- The prototype moved from the originally planned integration-test tree to a
  private engine test module so it can reuse assembly without buffering readers
  twice or publishing a new internal API. Its temporary duplicate multipart
  metadata parsing is removed by plan 10 above.
- Leap stable baseline verified: Maven `leap-sdk` and `leap-sdk-jvm` 0.10.9;
  SPM tag v0.10.9. Latest Swift prerelease: v0.10.13-SNAPSHOT. The docs' 0.10.7
  label is stale. Snapshot compatibility is not established.
- [Loading audit](API_RESHAPE_P0_LOADING.md) inventories all seven source forms,
  both eight-field multipart records, load defaults/precedence and model ownership.
  [Leap workstream](API_RESHAPE_LEAP_COMPAT.md) requires Swift/Kotlin source
  replacement, LoRA lists/scales, isolated embeddings and independent runtime
  packaging. Stable 0.10.9 is a regression baseline, not a ceiling on newer
  required capabilities.
- Three baseline Rustdoc links prevented documentation with warnings denied.
  Reproduced on unchanged baseline, then fixed links in `bundle/hf.rs` and
  `model/transformer.rs`. The checked documentation build now passes.

## Validation and review

Run from the implementation worktree with `/opt/homebrew/bin` prepended to PATH.
These builds used `CARGO_TARGET_DIR=/Users/dberrios/development/cera/target`.

| Command | Result |
|---|---|
| `cargo test -p cera --lib engine::loading_prototype --locked --offline` | 6 passed |
| Same test command with `--no-default-features` | 6 passed |
| `cargo test -p cera --features remote --lib engine:: --locked --offline` | 37 passed, including the prototype and existing engine/audio-engine matches |
| `cargo clippy -p cera --lib --tests --locked --offline -- -D warnings` | Passed |
| Same Clippy command with `--no-default-features` | Passed |
| `RUSTDOCFLAGS='-D warnings' cargo doc -p cera --no-deps --lib --locked --offline` | Passed after baseline link fixes |
| `cargo fmt --all --check`; `git diff --check` | Passed |
| Local Markdown references/fences/whitespace | Passed; 35 local links |

Two max-effort review rounds completed, each with three fresh reviewers and
all NO FINDINGS. The final round includes the documentation-link fixes. This
review result covers the first loading increment. No runtime benchmarks,
real-model matrix or full workspace CI were run. Those remain required at their
respective gates; this is not P0-L or Part I release approval. Reinspect CI before
any eventual push.

## Completed export increment

- [C0 evidence](API_RESHAPE_LEAP_EXPORTS.md) and the
  [probe harness](../../tests/leap_compat/README.md) record exact artifact pins,
  unchanged consumer sources, expected rejections, and reproduction commands.
- Stable Swift and Kotlin/JVM core consumers compile; the snapshot Swift
  LoRA/hidden-state consumer compiles. Stable artifacts reject the newer APIs.
  A stable custom Swift runner needs the snapshot's extra protocol methods.
- Simple standard Swift stream aliases fail the nonthrowing/Objective-C bridge
  checks. The KMP/SKIE candidate passes unchanged Swift and Kotlin stream/enum
  fixtures with no Leap dependency. Its payloads are incomplete; it has no model
  implementation or Cera bridge. Carry it into the next spike without freezing
  the full architecture.
- Plan 04's reference/Swift controls passed in `build/run-9_buonf0/results.json`
  and its smaller KMP checks passed in `build/run-5aqhtjuj/results.json`. These
  are historical results: plan 05 changes the candidate and supersedes those
  input/output hashes. Use the current single report below.
- Harness integrity tests cover changed cache bytes, interrupted downloads and
  unrelated compiler errors/crashes. All three tests, Python lint/format, Swift
  formatting and Kotlin formatting pass. First max-effort review found one
  contract documentation/fixture gap: custom Swift runners also support imported
  async witnesses. Corrected the docs and verified both forms. Two max-effort
  rounds completed with three fresh reviewers each; the final round returned
  all NO FINDINGS. All 44 local document links and whitespace checks pass.

## Completed protocol and native increment

- Plan 05's [protocol/native evidence](API_RESHAPE_LEAP_BRIDGE.md) records the
  larger unchanged consumer matrix and all remaining contract gaps.
- Stable and extended KMP/SKIE profiles pass their matching Swift/Kotlin core,
  conversation, parser-subclass, response-payload and custom-runner consumers.
  Both Swift async and completion-handler forms are checked. Stable runners
  still require the newer methods in the extended profile; Swift extension
  defaults fail Objective-C witness requirements. Product/version policy remains
  undecided; one module supporting both contracts has not been proved.
- Swift and JVM native executables use the existing Cera bindings and candidate
  HiddenStates container. A tiny CPU model verifies typed errors, sessions after
  model release, extraction isolation from subsequent generation, independent
  row copies and identical cross-language token/float results. These probes
  are not ModelRunner implementations. LoRA execution and runtime facade,
  GPU, streaming, cancellation and warm-KV performance remain open.
- Full offline evidence: `tests/leap_compat/build/run-qemg14y5/results.json`,
  34 expected checks passed (11 deliberate rejections). Python integrity
  tests (3), lint/format and Swift/Kotlin formatting pass. Two max-effort review
  rounds completed with three fresh reviewers each; the final round returned
  all NO FINDINGS. Fixes use distinct extraction tokens `[1, 0, 1]` against live
  `[0, 1]` to detect destructive reset/replay, and clarify which exports differ
  by profile. All 37 input hashes, six profile outputs and native library/binding
  hashes match the final report. All 58 local document links and whitespace
  checks pass. No new production API/dependency, commits or pushes.

## Completed filesystem loading increment

Plan `000341-06-filesystem-loading-contracts.md` is complete. It extends P0-L with
private shared local resolution and typed path/multipart-file sources, followed
by differential CPU/error/configuration fixtures. Implementation is complete;
622 library tests and all focused feature configurations pass. Scoped Clippy,
Rustdoc and formatting pass. Existing loader/manifest integration targets pass
16 tests, with four existing ignored model/download tests. Two max-effort review
rounds completed with three fresh reviewers each; the final round returned all
NO FINDINGS. See the loading audit for exact commands and limitations. The
next sequence at completion was 07; the current next unused sequence is 11.

Legacy public loading constructors retain their behavior. The typed automatic
file paths initially used an extra early metadata parse to turn legacy encoder
inference errors into KindMismatch; plan 10 removes this and the multipart
reparse by retaining the parsed primary. The final mapped primary is checked
again. No public signatures,
generation code or warm-chat execution changed. First review also found that
older docs promised manifest template precedence, while actual rendering uses
the tokenizer's embedded template. Docs and the main plan now distinguish
retained metadata from rendering; the new explicit prompt assertion passes.
The review fix also corrects the stale plan sequence in the workspace summary.
Remote all-target Clippy and Rustdoc pass after the fixes. No actionable review
findings remain, and none were skipped.

## Completed remote loading increment

Plan `000341-07-remote-loading-contracts.md` is complete. It adds shared private bundle/HF
resolution and private remote sources. Five subprocess-isolated loopback tests
exercise source selection, downloads, cache/progress, errors, manifest payloads
and cached bundle/DSpark routing. Four policy tests cover companion selection
and streaming-conversion options. The full remote-enabled library suite passes
659 tests with five existing ignored tests; the default suite passes 625 with
five existing ignored. Focused no-default, remote without mmap and minimal
remote+mmap checks pass 9/10/23 tests. Legacy integrations pass 16/four existing
ignored. Scoped Clippy, Rustdoc, formatting and 62 local document links pass.
First max-effort review found a vacuous monotonic-progress assertion. The HF
fixture now crosses callback thresholds and requires an intermediate count;
the loopback server also fixes inherited nonblocking sockets and panic cleanup.
All 23 minimal remote+mmap tests, the full remote suite (659 passed/five existing
ignored) and remote all-target Clippy pass after these fixes. Two max-effort
review rounds completed with three fresh reviewers each; all three final
reviewers returned NO FINDINGS. No actionable findings remain, and none were
skipped. See the loading audit for exact commands and fixture limits.

Public bundle URLs use preseeded files and locally rejected HEAD tunnels;
live catalog/CDN behavior remains open. SafeTensors conversion is unchanged;
only option construction is proved here. Auxiliary/draft fixtures exercise
resolution and fallback, not real execution. No public signatures, generation
code, warm-chat behavior, commits or pushes changed.

## Immediate next steps

1. Plans 40 through 42 delivered the core chat contract, decode observations
   and the private actual-Session bridge. Plan42's review loop completed four
   max-effort rounds clean. Next (Plan43, plan file not yet written): R1
   numerical proof on real model weights (ten completed warm turns with
   KV/RNG/drafter evidence), then P0.2 foreign/browser lowering of the phases and
   the open design items listed under Plan42. Keep existing raw methods and constructors; finish P0.1/P0.2/R1 before promising warm chat.
   Preserve loaded-byte identities, anonymous warm-only cache policy and runnable
   examples. Check disk capacity before further builds.
2. Carry the broader source/conversion, catalog/explicit-manifest/browser identity,
   concurrent/platform/device sharing, live CDN and load/network-cost checks into
   the affected P1/P2 shipping scope. Run them before shipping a changed path or
   claiming a target. Linux Plan10 invalid-path execution remains open. These do
   not make chat/recovery/Leap prerequisites to the independent loading strand.
3. Continue Leap C0 packaging and semantics: resolve stable/newer Swift profiles,
   pin a newer public Kotlin/Android LoRA/embedding baseline, and finish options,
   builders, parser/tool/media/value semantics and protected subclass members.
   An actual runner needs history, stream/error, cancellation, LoRA/embedding
   isolation and warm-KV evidence. Platform packages and replacement apps stay C2.
4. Begin P0.1's core chat prototype/profile and recovery matrices. No public
   warm-chat ingestion guarantee ships before R0/R1 and P0.2 pass.

## Phase checklist

- [x] P0-L: all source/constructor/config/ownership/kind and binding prototype proofs.
- [ ] P0.1: core chat prototypes, all caller inventory, R0 recovery and R1 profile matrix.
- [ ] P0.2: chat foreign prototypes and frozen numeric performance budgets.
- [ ] R0: checked recovery for existing ingestion failures.
- [ ] R1: initial warm-chat boundaries, phases, and retained KV proof.
- [ ] P1: additive production core/bindings, compatibility fixtures and runnable examples using public imports.
- [ ] P2: first-party migration, README/chat/recovery examples and documentation; release replacements/deprecations.
- [ ] P3: removal in a breaking release after at least one compatibility release.
- [ ] C0: Leap artifact/API inventory and unchanged-consumer architecture spike.
- [ ] C1: supported Leap loading/conversation/streaming facade and contract tests.
- [ ] C2: packaging, actual app migrations, performance and compatibility release gates.
- [ ] Part II F1–F8: independent capabilities; do not count them as implemented by a facade.

Each unchecked item remains required even though thirty-nine bounded increments
through Plan42 are complete. See the two linked workstream documents for exit
criteria and dependencies; none of these checkboxes means merely writing a plan.

## Fine-tuning follow-up

The user clarified the goal: improve a MoE model as an agentic code editor.
Public information checked on 2026-09-07T19:22-0700. [LQH](https://lqh.ai/)
and its [public README](https://github.com/Liquid4All/lqh#readme) describe
MoE fine-tuning, supervised and preference training, evaluation and GGUF export.
The exact target checkpoint is not selected. LFM2.5-8B-A1B is a candidate;
its [model card](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) recommends tool
use but identifies heavy programming as a weaker fit.

Suggested pilot: train complete search/read/patch/test/repair sequences in the
editor's actual tool format, initially for localized fixes and API migrations.
Evaluate held-out repository tasks using successful patches, required tests,
regressions and completion time. A repository-execution evaluator must be
integrated; public documentation alone does not establish that it is built in.
Then validate an exported model through Cera, including tool formatting, quality,
latency and retained KV. MoE adapter/export compatibility remains unproved.
This is a feasibility assessment; no training job or numbered implementation
plan has been started for it.

## Resume constraints

Work only in this worktree; leave main on main. Devlogs stay at repository root
because they are gitignored. Read numbered plans and append a new one before
expanding implementation. Do not use the removed code-review-graph integration.
Do not commit or push without a user request. Any eventual push must use an
explicit destination `HEAD:refs/heads/refactor/api-reshape`. No attribution
trailers or tool names in authored project records.
