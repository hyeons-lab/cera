# Third Review: Adjudicating the Two Prior Reviews

**Target documents:** [API_RESHAPE_PLAN.md](./API_RESHAPE_PLAN.md),
[API_RESHAPE_REVIEW.md](./API_RESHAPE_REVIEW.md),
[API_RESHAPE_REVIEW_2.md](./API_RESHAPE_REVIEW_2.md)
**Baseline commit:** `60fc11c25a51` (`origin/main`), 2026-09-06
**Role of this document:** incorporate both prior reviews into one actionable
list. Where the second review corrects the first, this document sides
explicitly and says what to do instead. It proposes no plan edits itself —
the plan file is untouched.

**Method:** I spot-checked the load-bearing code claims against the working
tree at the baseline commit (FFI session handles, `Session::reset`,
`truncate_to`, the decode-break path, the shift path, the single-message
template render). Claims I confirmed are marked [verified]; claims I take on
the reviews' authority keep their cited `file:line`.

## 1. Verdict

The plan stands. The first review's architecture is sound but contains three
partly-wrong code claims; the second review's corrections check out against
the code I read. The consolidated recommendation below keeps the first
review's state-machine, `n_keep`, and differential-test asks (sharpened),
replaces its `CancelHandle` and static-delimiter recommendations, and adopts
the second review's new findings wholesale. Net effect on the plan is six
edit sites (§0, §1.2/§4.2, §3/I1+§4.3, §4.4, §7/P0+§2, §8/§8.1+§9).

## 2. Adjudication of the first review's five findings

### F1 — Turn-boundary state machine: STANDS, sharpened

Adopt `SessionPhase` (`Idle` / `PromptReady` / `TurnComplete` / `Interrupted` /
`Unusable`), `ingest_messages` for multi-message turns, and rejection of
`ingest` on an `Interrupted` turn without explicit close-or-replace.

Sharpening from the second review: the double-`complete()` problem is worse
than the first review states. Greedy decode clears `last_logits` when
`generated > 0` and the second call fails with `EmptyInput` [verified:
`cera/src/session.rs:2584-2588`]; stochastic sampling leaves
`last_logits = Some(..)` so the second call silently continues the assistant
turn [verified: same site, `else` branch]. Same call, two behaviors, selected
by `temperature`. The phase contract must normalize this divergence, not just
the greedy error.

Binding to the plan: the phase enum is exactly the "minimal execution
bookkeeping" §4.2 already permits, so adopt it under that clause — and then
it must appear in the §4.3 checkpoint account (`Restored`/`Reset` restore the
phase or say so) and in F4's persistence identity, or it becomes untracked
state.

### F2 — `n_keep` vs D7 transcript-free sessions: STANDS, sharpened (strongest finding)

Adopt Option A outright for Part I: high-level chat defaults to `n_keep = 0`;
eviction is caller-side message windowing plus `replace_messages`. The plan's
"proven cursor transition" (§4.2) is too abstract to implement against.

Sharpening [verified: `cera/src/session.rs:1360-1370`]: a context shift
invalidates more than the rendering cursor — it clears `last_logits`, resets
the drafter, and rewrites `token_history` heuristically (the comment at the
site concedes nothing pins that history to the KV index-for-index). Any F8
proposal to support `n_keep > 0` under high-level `ingest` must account for
all four, plus an evicted-span report the caller can map back to `Message`
objects.

### F3 — Foreign cancellation object: CORRECTED, do not build it

The second review is right and I confirmed it: `cera-ffi::Session` already
caches the core session's `Arc<AtomicBool>`/`Arc<AtomicU32>` at construction
[verified: struct doc comment and fields at `cera-ffi/src/lib.rs:1955-1975`],
`cancel()`/`position()` never take the mutex, and `Session::reset` stores
into the same `Arc`s rather than replacing them [verified:
`cera/src/session.rs:1016, 1020`]. A standalone `CancelHandle` UniFFI object
buys nothing.

What to codify instead: the **handle-identity invariant** — handles are cloned
once at construction and every reset/recovery path must store through the
existing `Arc`s, never rebuild session internals in a way that strands cached
foreign handles, with no compile-time signal if it does. Add it to §4.4 and
drop the new object from the recommendation list.

Kept from the first review (unrefuted): maintain the core-`ModalitySink`
(token IDs) vs foreign-sink (UTF-8 text/thought chunks) nomenclature in the
spec, and document `clear_cancel()` discipline when retrying after a
cancelled prefill (recovery preserves the latch; decode-guard cleanup is
unchanged).

### F4 — R0 recovery vs backend reality: CORRECTED framing, larger R0 scope

The second review's correction holds: both GPU (`gpu_lfm2.rs:6903`) and Metal
(`metal_lfm2.rs:3853`) `truncate_kv` implementations store the sequence
counter — and that is *all* they do. The on-device LFM2 convolution rolling
buffers are not rewound and there is no equivalent of the CPU's
`has_pos` guard [verified CPU side: `cera/src/kv_cache.rs:1094-1125` —
asserts on compressed/out-of-range targets, forces `safe_len = 0` when the
target left the conv ring]. So a hybrid-model truncate on GPU/Metal leaves
**silently corrupt** conv state, and `append_user_message`'s rollback calls
`truncate_kv` unconditionally today [verified: `cera/src/session.rs:2033`].

Required plan changes (§3/I1 + §4.3): `truncate_to` becomes fallible instead
of asserting; `Model::truncate_kv` gains a pre-mutation capability query so
the recovery ladder is chosen *before* state is touched; GPU/Metal rewind
becomes honest (report unsupported → recovery reports `Reset`) or actually
correct for conv layers. Document that `Reset` is the dominant outcome on GPU
and TurboQuant configurations — callers re-supply context via
`replace_messages` — and keep the no-full-copy rule (§4.3): honest
`Reset`/`Unusable` instead of context-sized checkpoint copying.

### F5 — Incremental tokenization: SPLIT — test stands, mechanism rejected

Adopt the differential test as an R1 exit gate (full Jinja render tokenized
monolithically vs the incremental turn-by-turn sequence; any mismatch fails
the gate). Reject the prescribed mechanism: hardcoding static delimiter
token-ID slices per profile *is* a new rendering algorithm, forbidden inside
a compatibility adapter by §0 and §4.2.

Name the concrete per-profile screen instead: the current per-turn path calls
`apply_chat_template` on a **single** message with `add_generation_prompt:
true` [verified: `cera/src/session.rs:2016-2023`] and `append_text` →
`tokenizer.encode` [verified call chain at `session.rs:1157-1163`; the no-BOS
property itself is cited from the second review, not re-verified here].
Templates that emit a BOS or default system block with no system message
present (Llama-3 family) re-emit that preamble every turn. R1 must screen
each frozen profile for exactly this, restricted to 2–3 well-behaved initial
profiles.

## 3. New findings adopted from the second review

1. **The turn terminator is never committed to KV, and nothing re-adds it
   (the defect R1 exists to fix).** The decode loop breaks on EOS/stop
   *before* `token_history.push` and the committing forward [verified shape:
   `cera/src/session.rs:2358-2362` vs `2408`], so after a completed turn every
   emitted token is in KV and the terminal marker is not; the next turn
   renders only the new user message, so the closing marker never renders.
   Consequences for the plan: (a) restate §4.2's "establish the boundary"
   as a determined fact — emitted ⊇ committed, missing exactly the terminal
   stop token — with a fixture; (b) add the R1 carve-out to §0's parity rule
   (mirroring R0's): fixing the envelope *changes* warm-path prompt tokens,
   recorded as a declared behavior change, with P1 parity measured against
   R1's retained-state reference rather than the defective legacy output.
2. **Warm chat already ships — the migration risk is inverted.** `cera-ffi`
   `send_message` → `generate` → `send_message` is already the
   warm/incremental surface while the CLI renders full history after reset
   (cited: `cera-ffi/src/lib.rs:2365-2381`, `cera-cli/src/main.rs:2809,
   2927` — taken on authority). The CLI migration is a workflow upgrade, but
   existing FFI/Swift/Kotlin callers may observe output changes from R1's
   envelope fix in code they never touched. P0's inventory gains a
   "warm today vs. reset today" column per first-party surface (§2), and §9's
   documentation deliverable carries a release note for the FFI behavior
   change.
3. **P0 must be split and unblocked.** Adopt P0.1 (core Rust API, state
   machine, error types) / P0.2 (foreign prototypes, frozen benchmark
   budgets), and additionally put the loading strand (`ModelSource` /
   `ModelLoader`, constructor inventory — no coupling to the state machine,
   recovery ladder, or benchmark freeze) on its own track so it is not held
   behind Swift/Kotlin prototypes and latency budgets.
4. **§8.1's frozen budgets need the repo's named confounds.** Host loadavg
   swings Mac decode throughput ~50%; the shared Android device carries
   thermal/memory-pressure confounds with a stale-then-live `dumpsys`
   thermal block; GPU models keep KV/conv/prefix-cache state on the
   **model**, not the session, which has already produced false bug reports
   on reused models (all cited from the second review, taken on authority).
   Protocol additions: fresh model instance per measurement; interleaved A/B
   runs reported as ratios, not just repeat counts and medians. Plus the
   first review's 10-turn multi-turn drift test in the §8 matrix.

## 4. Consolidated edit list for the plan (in plan order)

1. **§0** — add the R1 carve-out: warm-path prompt tokens may change;
   declared behavior change with fixtures, exempt from equivalent-call
   parity; P1 warm parity targets R1's retained-state reference.
2. **§2** — add the "warm today vs. reset today" column per first-party
   surface to the migration inventory.
3. **§3/I1 + §4.3** — correct GPU truncate framing (counter only, conv
   buffers stale, no guard); require fallible `truncate_to` and a
   pre-mutation `truncate_kv` capability query; `Reset` documented as the
   standard GPU/TurboQuant outcome; no-full-copy rule retained.
4. **§1.2/§4.2** — adopt `SessionPhase` + `ingest_messages`, bound to the
   existing bookkeeping clause; forbid `ingest` on `Interrupted` without
   explicit close-or-replace; record emitted-vs-committed as determined fact;
   state `n_keep = 0` as the Part I high-level-chat default; carry phase into
   §4.3 checkpoints and F4 persistence identity. R1 screens: single-message
   BOS/preamble re-emit per profile; full-vs-incremental differential test as
   exit gate; no prescribed static-delimiter mechanism.
5. **§4.4** — replace the `CancelHandle` object with the handle-identity
   invariant; add the greedy-vs-stochastic double-`complete()` divergence to
   the phase contract; keep sink nomenclature and `clear_cancel()` retry
   discipline.
6. **§7/P0** — split P0.1/P0.2; put the loading strand on its own track off
   the critical path.
7. **§8/§8.1** — add the differential tokenization test (R1 gate), the
   10-turn drift test, fresh-model-per-measurement, interleaved-A/B-ratio
   reporting, and the named confounds.
8. **§9** — release note for the FFI warm-envelope behavior change.
