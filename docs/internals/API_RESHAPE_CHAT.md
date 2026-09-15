# Chat contract, profiles and execution inventory

Plan40 implements an isolated P0.1 prototype in
[cera/tests/api_chat](../../cera/tests/api_chat), with the runner in
[tests/api_chat](../../tests/api_chat/README.md). Production chat publication still
requires P0.1/P0.2/R1. Loading/recovery are independent completed increments; they
do not imply warm-turn correctness or numerical KV performance.

## Initial pinned profile matrix

The first named fixture is `lfm2-350m-gguf-simple-text-v1`. Its public artifact is
[LiquidAI/LFM2-350M-GGUF, Q4_0 at revision 8fdc9d5](https://huggingface.co/LiquidAI/LFM2-350M-GGUF/tree/8fdc9d526b7ed346b19257551b05816c7912ecc2).
The [machine-readable pin](../../tests/api_chat/profile.json) records the full
revision, SHA-256, tokenizer metadata and remaining backends. It uses all 65,536
vocabulary entries and 63,683 BPE merges, `gpt2` tokenizer model with `lfm2`
pretokenization, BOS 1, message-start 6 and EOS 7. The template is the exact embedded
simple role/content template. A template published separately or a similarly
named model is not implicitly included in this profile.

| Profile/backend | Plan40 evidence | Production warm-chat gate |
|---|---|---|
| Named LFM2 profile / native CPU | Full tokenizer + real Session decode with a scripted Model; structural ten-turn prefix comparison | Numerical weights, exact decode observations and KV/RNG/drafter traces remain |
| Same profile / native Metal | Rendering contract only | Actual warm decode, device state and performance remain |
| Same profile / native wgpu | Rendering contract only | Actual warm decode, device state and performance remain |
| Same profile / WASM CPU | Portable prototype compile target | Node runtime, production facade and warm traces remain |
| Same profile / browser WebGPU | Required named async target | Async ingestion/recovery/observation implementation and browser runtime remain |

No backend is advertised as production chat-supported by this prototype. Signature
discovery checks the exact template and special-token encodings; it does not hash
or certify arbitrary model weights or tokenizers. The fixture runner separately
verifies the whole pinned artifact. Production discovery must bind its selected
profile to the loaded model/tokenizer/backend capabilities before inference.

The initial profile accepts an optional first system message followed by alternating
user/assistant text messages, ending with a user message. Later batches start with
a user and also end with a user. Consecutive text parts concatenate in order;
empty text is accepted. Tool roles and image/audio parts are typed but rejected
by this profile. Literal `<|` marker syntax, including split text parts, is rejected
without inventing a model-specific escaping convention. History-dependent Jinja,
tools, thinking transforms and media need separately validated profiles.

## Prototype contract

| Phase | Next chat operation |
|---|---|
| Idle | First supported batch or explicit replacement |
| PromptReady | One completion or streaming generation; reject another ingest |
| TurnComplete | Append the next supported batch, including pending closing boundary once |
| Interrupted | Explicit replacement; no implicit continuation or re-ingestion |
| RawContext | Explicit replacement before chat; raw operations remain caller-controlled |
| Unusable | Successful checked reset or recreation |

`Message` owns an ordered `Vec<ContentPart>`; ingestion borrows the batch for the
call and stores none of it. The prototype uses `usize` for Rust positions; foreign
lowering still needs checked width mappings. `n_keep != 0` and audio-output
execution are rejected at construction; output capability is rechecked before
preparation and decode, including after raw access. Rendering, role/media validation and
known capacity checks precede mutation. Each batch has one final assistant prefix
and one transactional append. Replacement validates before checked reset; a later
append failure can report `Reset` or `Unusable`, never restoration of discarded
context. Reset for replacement preserves the external cancellation latch.

Recovery restores the prior phase and pending boundary only for `Unchanged` or
`Restored`; a verified reset enters Idle and unknown/unusable outcomes fail closed.
The prototype marks execution unusable before a fallible execution boundary so
an unwind cannot leave a prepared cursor. Raw escape invalidates the cursor before
returning mutable access. This does not replace the production Session usability
checks already added by R0.

`generate_into` validates before decode and requires an explicit backend observation:
proven no progress, actual terminal token plus whether it was committed, interrupted,
or unusable. FinishReason::Stop alone is insufficient. Zero emitted tokens also
does not prove no progress: sampling a terminal token can advance RNG without
committing a token. Both collector and sink invoke the same entry once. The text
collector uses the existing tokenizer and returns the existing summary; it does
not invent counters or silently enable audio output.

## Before/after turn boundary

For the pinned template and a scripted answer `a`, the first resident prefix ends:

```text
<|im_start|>assistant
a
```

The sampled `<|im_end|>` has not been forwarded to KV. Legacy isolated next-message
rendering then appends BOS and the next user envelope. The corrected suffix is:

```text
<|im_end|>
<|im_start|>user
World<|im_end|>
<|im_start|>assistant
```

The fixture asserts the legacy result differs from full rendering, counts two BOS
tokens, and checks the corrected token sequence matches full rendering. If a decode
path already committed EOS, only the template newline remains pending. Restoration
must preserve that residency distinction so retries cannot duplicate a boundary.

The ten-turn fixture compares canonical, nonempty scripted answers. Arbitrary
generated token sequences are not guaranteed to round-trip through decoded text
and canonical BPE: byte fallback, whitespace merges and alternate segmentations
can change tokenization. R1 must distinguish resident generated IDs from a caller's
text transcript and test boundary-sensitive cases; it cannot promise universal
token identity by re-encoding arbitrary output text.

## First-party execution policies

| Surface and source | Current policy | Migration requirement |
|---|---|---|
| [CLI chat](../../cera-cli/src/main.rs), chat loop | Caller-owned history; reset and full render each turn; drops whole pairs on overflow | Explicit replacement first; warm ingest only after phase/boundary validation |
| [TUI chat](../../cera-cli/src/chat_tui.rs) | Same reset/full-history pattern, including parallel image attachments | Preserve history/attachment ownership and failure durability |
| [Rust Session](../../cera/src/session.rs), `append_user_message` | Isolated message template render, incremental raw KV; R0 failure recovery | Add bounded chat cursor; raw/legacy mutation invalidates it |
| [Native FFI](../../cera-ffi/src/lib.rs), `send_message*` | Incremental whole-message append; combined calls hold mutex across append/decode | Preserve operation lock, cancellation handles, callback ordering and recovery snapshots |
| Same FFI, `generate*`/`generate_*_async` | Raw collection/streaming; async worker retains same session | Add explicit chat operations without silently changing raw chaining |
| Generated Swift/Kotlin/Python/Dart and SwiftPM | Direct lowerings of native FFI | Regenerate all bindings after production types/operations are ready |
| [Dart async native](../../cera_ffi/lib/src/async/cera_io.dart) | Caller supplies tokens; synchronous prefill then native async streaming; separate image/audio paths and reset | Preserve worker/cancel lifecycle; adapt canonical message batching explicitly |
| [CPU WASM](../../cera-wasm/src/lib.rs), `Session` | Raw text/tokens/media append and core generate/reset; no whole-message export | Introduce chat facade with the same observed phases and typed recovery |
| Same file, `WebGpuSession::generate` | Convenience string prompt; separate GPU state/decode implementation | Needs async-aware profile/boundary observation; not covered by core Session hooks |
| Same file, `generate_tokens` | Incremental raw KV and caller-supplied tokens | Preserve raw path and make chat/raw transitions explicit |
| [Dart web worker](../../cera_ffi/lib/src/web/cera_worker.js) | CPU/GPU dispatch, manual media envelopes, raw prompt/decode and reset/recreation | Retain media ordering and async ownership; migrate as separate backend adapters |
| [Dart chat example](../../cera_ffi/example/cera_chat.dart) | One-shot manual template + `encodeTextSpecial(..., true)` + raw generation | Avoid duplicated BOS when promoting a rendered-prompt example |
| [WebGPU demo](../../cera-wasm/examples/webgpu/index.html), bench | Prompt string/raw token convenience demonstrations | Add concrete warm/replacement examples after production facade exists |
| [Parity consumers](../../cera-parity) and loading/recovery examples | Raw or legacy append/generate for a specific probe | Keep their current semantics; add new chat probes rather than conflating contracts |
| [cera-client messages](../../cera-client/src/types.rs) | Separate wire DTO with developer role, name, reasoning, refusal, tool calls and tool-call IDs | Future `TryFrom` must reject unsupported information; no lossy conversion added here |

Source symbols are authoritative if line numbers shift. This inventory covers the
first-party policy families; P1 still needs an exhaustive call-site migration list
and compatibility/deprecation decisions for each exposed declaration.

## Decode sites identified before Plan41

In [Session::generate](../../cera/src/session.rs), ordinary EOS/custom stop occurs
before forward/emission. Grammar trigger/termination changes whether that stop is
allowed. Audio has separate completion/stop branches. Greedy speculative decode
has both initial and accepted-draft terminal checks, with rewind effects. Greedy
generation clears stale logits after output; stochastic generation keeps them.
The new phase must reject double completion consistently across both modes.

Plan41 below records terminal identity/residency, progress and
state validity at those actual exit sites without changing legacy counters,
callbacks, RNG, drafter selection or cancellation cleanup. The isolated test bridge's typed
observations specify the required information; they are not a production adapter.
No inference path should reconstruct this information from summary counters alone.

## Internal core observation implementation (Plan41)

The candidate [decode observations](../../cera/src/session/decode.rs) are internal.
`Session::generate` invokes `generate_observed` once, optionally traces its outcome,
and returns the same legacy Result. The observation travels with that call's result;
it is not retained as a query that could become stale after raw mutation. Public
finish reasons, summary fields and sink callbacks retain their existing shapes.

| Observation | Actual evidence from the decode path |
|---|---|
| NoProgress | Armed cancellation, zero budget, full context, missing logits or cancellation before the first speculative step, all before inference mutation; existing telemetry consumption and cancellation cleanup still occur |
| TokenStop(token) | Exact sampled EOS/custom stop identity; that occurrence was not forwarded, or was removed by accepted-draft rewind |
| Audio | The call entered the audio branch; its Stop is not a text assistant boundary |
| Interrupted | Successful generation ended without one of those proven boundaries; includes grammar dead ends even with zero emissions |
| Unproven | An error or speculative rewind did not establish usable execution state; a chat adapter must fail closed even if the legacy Result is Ok |

Speculative verification observes both its rejected-tail rewind and the later
accepted-stop rewind. Backend eligibility is checked after the forward, immediately
before each legacy rewind, because a convolution checkpoint can expire during a
large draft. The resulting position must match the target. Any unproven rewind is
latched for the whole call; a later successful check cannot certify an earlier
unsafe round. Legacy CPU rewind can clear an expired convolution ring, and device
counter-only rewind can leave physical state at the tail. Both remain Unproven,
including when the unchanged legacy result reports Stop. The standalone public
verification helper does not invoke these additional checks.

NoProgress describes effects, not readiness: an `EmptyInput` Result is still an
error. A future chat caller must enforce its phase/invariants and keep that error.
It must also install its failure guard before calling the observed entry: a panic
in forward, rewind or a sink returns no observation at all. The observer alone
does not implement production chat recovery or alter legacy Session usability.

The [real-loop tests](../../cera/src/session/decode/tests.rs) exercise ordinary
greedy/stochastic stops, omitted EOS with an RNG-step discriminator, no-op state
preservation, accepted-draft EOS/custom-stop rewind, initial speculative stop,
grammar dead end and lazy triggers, cancellation before the first speculative step,
partial cancellation, capacity/budget limits,
and errors/unwinds. An [audio-end control](../../cera/src/session/decode/tests/audio.rs)
enters production audio dispatch through a scripted audio backend, consumes its
audio transition in KV, and returns Stop with zero text tokens and an Audio
observation. The [rewind controls](../../cera/src/session/decode/tests/rewind.rs)
exercise actual CPU ring expiration at both rewind sites and a scripted counter-only
backend. These controls do not validate numerical GPU/audio model quality.

The ordinary, speculative and shared verification bodies can be compared to their recorded
Plan40 versions after removing only the observation statements and restoring the
equivalent `EmptyInput` expression. That source audit complements the behavioral
regressions; it is not a latency measurement or a replacement for R1 real-model
warm-state evidence. Actual chat preparation/recovery is now exercised by the
private Plan42 bridge below; foreign/browser phase publication still needs
integration.

Remaining gates after Plan41 (see the Plan42 section for the private candidate
that covers the first one): actual core transactional chat integration;
native/CPU-WASM and browser async lowering; at least ten real completed warm
turns with KV/RNG/drafter evidence; numerical budgets and baselines; caller
migration and runnable public Swift/Kotlin/Dart/Rust/browser examples; Leap
runtime/package validation.

## Actual Session transaction bridge (Plan42)

The private [core adapter](../../cera/src/session/chat.rs) executes the shared
contract against a real `Session`; its [transaction tests](../../cera/src/session/chat/tests.rs)
inspect actual attention-cache rows, token history, logits and physical forward
inputs. The adapter is compiled in the unit-test build, with no public test hooks
or duplicated phase/rendering implementation. This is an R0/R1 integration
candidate, not a published chat API or a completed numerical performance gate.

Construction discovers the profile from the Session's own tokenizer and reads
its actual configuration. Construction, preparation and decode all run the same
identity check (unusable execution, missing text capabilities, noncausal/classifier
models, classifier LoRA adapters, a model vocabulary smaller than the tokenizer's,
nonzero `n_keep`, and a changed model/tokenizer identity) followed by the separate
audio-output capability check. `n_keep` is re-checked because a raw swap can
install a Session sharing the same model and tokenizer with a sliding context.
Replacement cannot reset a newly swapped Session under a stale profile. A weak
model identity avoids retaining a replaced model's weights just to detect that
mismatch. An explicit
reset deliberately does not consult the profile: it always performs the checked
KV reset and returns to Idle, so identity drift introduced through `raw()` can be
undone through `raw()` again instead of stranding a healthy Session inside an
Unusable chat.

A complete input batch runs through one existing `with_ingest_recovery` operation.
The new returned error owns the original cause, recovery outcome, typed rewind
failure and original secondary reset error. The optional rewind diagnostic is
boxed on the error path to keep the Result small; no successful-call allocation
is added by that representation. The private bridge moves the diagnostic
out of the legacy user-message slot rather than flattening non-Clone errors into
strings; the legacy `append_user_message` getter and bindings retain their existing
surface. Final public diagnostic ownership/lowering must preserve this information.
Replacement validates first and uses checked complete reset without clearing
cancellation. A failed append after that reset reports Reset or Unusable, never
restoration of the discarded history.

Decode uses one observed generation call and a Session failure guard in addition
to the outer chat phase guard. Unproven outcomes and forward/callback unwinds disable
raw Session execution too. A legacy Ok/Stop result may therefore accompany an
Unusable phase after an unproven speculative rewind; the Result is not rewritten.
Proven zero-progress success keeps PromptReady. A zero-progress error (missing
prefill logits) cannot certify that a prompt is ready, but the observation proves
nothing was mutated, so the chat enters RawContext (replacement and raw access
open, the Session usable) rather than Unusable. A successful audio-path exit is
a proven outcome without a text boundary and enters Interrupted. Only unproven
observations, errors after mutation and unwinds disable raw Session execution.
Validation of execution identity precedes the audio capability check at both
the preparation and decode entry points, so both report the same first error.
Custom nonterminal stops enter Interrupted even
when no text token was emitted. Both sampling modes reject double completion.

Resolved chat design items (Plan 45):
- Cancellation management: `Chat` and `Execution` expose `cancel_handle()` (`Option<Arc<AtomicBool>>`), `cancel()`, and `clear_cancel()`, allowing direct thread-safe cancellation and non-destructive clear without altering the chat phase or cursor.
- Fallback reset for backends without `try_reset_kv`: `CoreExecution::reset` and `Session::reset` attempt checked KV reset first (`reset_execution_checked()`), falling back to state re-allocation (`reset_realloc_state()`) if the backend does not support checked reset (returning `Backend("checked KV reset is not supported by this backend")`). This keeps generic, CPU, and WebGPU backends usable across explicit and replacement resets.
- Non-destructive Session return: `Chat::new` returns `Result<Self, (E, ValidationError)>` so caller ownership of the underlying `Session` is preserved when validation fails (such as unsupported profiles, sliding context, or audio output). Furthermore, `Chat::into_inner(self) -> E` and `CoreExecution::into_session(self) -> Session` allow extracting the session at any time.

The actual ten-turn fixture uses this flow (promoted to public library API in Plan 46):

```rust
let mut chat = session.into_chat()?; // or core_chat(session)?
chat.ingest(&Message::text(Role::User, "first"))?;
let first = chat.complete(&options)?;
assert_eq!(chat.phase(), SessionPhase::TurnComplete);
chat.ingest(&Message::text(Role::User, "next"))?;
let next = chat.complete(&options)?;
```

The tests assert that each new prefill starts at the old position and processes
only the new input plus the pending EOS/newline. They verify no normal-turn reset
or recovery rewind, preserved pending boundaries after restoration, typed secondary
failure, cancellation-handle identity, sampler progression and drafter reset rules.
The full public tokenizer case includes Unicode inputs. Model logits remain
scripted, so these controls do not prove numerical weight execution, GPU cache
contents or measured latency/memory budgets. Those R1/P0.2 gates remain mandatory.
Run both the isolated and actual-Session suites using the [fixture runner](../../tests/api_chat/README.md).
