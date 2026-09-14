# Second Review: Cera Public API Reshape Plan

**Target documents:** [API_RESHAPE_PLAN.md](./API_RESHAPE_PLAN.md), [API_RESHAPE_REVIEW.md](./API_RESHAPE_REVIEW.md)
**Baseline commit:** `60fc11c25a51` (`origin/main`), 2026-09-06
**Review focus:** verifying both documents' code claims against the baseline, correcting the first review, and naming the defects neither document records.

---

## 1. Verdict

The plan is sound and unusually disciplined. It refuses fake transactionality, refuses to call reset-per-turn "warm", and gates the two hard prerequisites (R0, R1) ahead of the wrappers. The first review is a good read of it, but three of its five findings are partly wrong about the codebase, and both documents miss the single concrete defect that R1 exists to fix — a defect that also breaks one of the plan's own top-level rules.

Every claim below was checked against the baseline; source references are `file:line` at that commit.

---

## 2. Corrections to the first review

### 2.1 Finding 3 (UniFFI cancellation deadlock) is already solved

`cera-ffi::Session` caches the core session's `Arc<AtomicBool>` and `Arc<AtomicU32>` at construction time (`cera-ffi/src/lib.rs:1963-1969`); `cancel()` and `position()` never acquire the mutex, and the struct's doc comment states exactly that. A standalone `CancelHandle` UniFFI object buys nothing.

The requirement worth codifying is the *invariant*, not a new type. Handles are cloned once at construction, and `Session::reset` stores into the same `Arc`s rather than replacing them (`cera/src/session.rs:1016`, `1020`). Any reimplementation that rebuilds session internals on reset or on recovery would silently strand every cached foreign handle, with no compile-time signal.

**Edit:** add the handle-identity invariant to §4.4; drop the new object from the recommendation list.

### 2.2 Finding 4's device-counter claim is wrong, and the truth is worse

The review states that GPU/Metal `truncate_kv` "does not rewind GPU-allocated attention matrices or device-side counters." Both implementations *do* store the counter — and that is all they do:

- `cera/src/model/gpu_lfm2.rs:6903`
- `cera/src/model/metal_lfm2.rs:3853`

Neither rewinds the on-device LFM2 convolution rolling buffers. The CPU path at least detects this case and refuses: `InferenceState::truncate_to` forces `safe_len = 0` when `ConvHistory::has_pos` fails (`cera/src/kv_cache.rs:1112-1125`). The GPU and Metal paths have no equivalent check, so a truncate on a hybrid model leaves **silently corrupt** conv state rather than an honestly unrestored one — and `append_user_message`'s rollback calls `truncate_kv` unconditionally today (`cera/src/session.rs:2033`).

R0's scope is therefore larger than "make `truncate_to` fallible". It must make GPU/Metal `truncate_kv` either honest (report unsupported, so recovery reports `Reset`) or actually correct for conv layers.

**Edit:** correct the framing in §3/I1 and §4.3; require a pre-mutation capability query on `Model::truncate_kv` so the recovery ladder is chosen *before* state is touched, and require `truncate_to` to become fallible rather than asserting.

### 2.3 Finding 1.3 is only half the problem

The review notes that greedy decode clears `last_logits` when `generated > 0` (`cera/src/session.rs:2584-2588`), so a second `complete()` returns `EmptyInput`. It misses the other branch: under stochastic sampling `last_logits = Some(logits)`, so a second `complete()` silently **continues the assistant turn**.

Same call, two behaviors, selected by `temperature`. That divergence — not just the greedy error — is what a `SessionPhase` has to normalize.

### 2.4 Finding 5's recommendation conflicts with the plan

Hardcoding static delimiter token-ID slices per profile *is* a new rendering algorithm, which §0 and §4.2 forbid inside a compatibility adapter. The differential test the finding asks for is the right ask and belongs in R1's exit gate; the static-constants mechanism should not be prescribed.

The concrete per-profile screen worth naming instead: the current per-turn path calls `apply_chat_template` on a **single** message with `add_generation_prompt: true` (`cera/src/session.rs:2016-2023`), then `append_text` → `tokenizer.encode`, which adds no BOS (`cera/src/session.rs:1157-1163`). Templates that emit a BOS or a default system block whenever no system message is present (the Llama-3 family) will re-emit that preamble on every turn. That is a per-profile screen R1 must run, not a general renderer problem.

### 2.5 Finding 2 (`n_keep` vs D7) is the review's strongest point, and should be sharpened

A context shift invalidates more than the message cursor. It also clears `last_logits` (`cera/src/session.rs:1370`), resets the drafter (`1361-1363`), and rewrites `token_history` heuristically (`1360` — whose own comment concedes nothing pins that history to the KV index-for-index).

Option A is the right call for Part I: high-level chat defaults to `n_keep = 0`, and eviction is handled by caller-side message windowing plus `replace_messages`. The plan should state this outright rather than leaving §4.2's "proven cursor transition" abstract.

---

## 3. Findings neither document records

### 3.1 The turn terminator is never committed to KV, and nothing re-adds it

The decode loop breaks on EOS or a stop token **before** `token_history.push(token)` and before the forward that would commit it (`cera/src/session.rs:2358-2362` vs `2408`). So after a completed turn: every emitted token is in KV, and the terminal `<|im_end|>` / EOS is not.

The next turn renders only the new user message through Jinja — the previous assistant message is not in the message list, so its closing marker never renders, and nothing prepends it. `cera-ffi`'s `send_message` → `generate` → `send_message` loop (`cera-ffi/src/lib.rs:2365-2381`) therefore builds an unterminated, off-distribution transcript today.

Two consequences the plan needs to absorb:

1. §4.2's "R1 must establish the boundary between emitted tokens and tokens actually committed to KV" should be restated as a **known defect with a fixture**, not an open investigation. The answer is already determinate: emitted ⊇ committed, missing exactly the terminal stop token.
2. It collides with §0's parity rule. "Preserve prompt tokens … for equivalent calls" cannot hold for the FFI warm path, because fixing the envelope *changes* its prompt tokens. R0 already has a carve-out for precisely this shape of problem ("may change erroneous failure behavior and must document that change"). **R1 needs the identical carve-out for warm-path prompt tokens.** Without it, P1's parity gate and R1's exit gate contradict each other.

### 3.2 The plan under-states that warm chat already ships

§2's table lists "Combined ingestion and generation" without noting that `cera-ffi` is already the warm/incremental surface while the CLI is the reset-per-turn one (`cera-cli/src/main.rs:2809`, `2927` render full history after reset).

This inverts the stated migration risk. The CLI migration is a workflow *upgrade*, but existing FFI/Swift/Kotlin callers may observe **output changes** from R1's envelope fix in code they never touched.

**Edit:** P0's inventory gains a "warm today vs. reset today" column per first-party surface, and §9's documentation deliverable carries a release note for the FFI behavior change.

### 3.3 `SessionPhase` is compatible with D7 — the plan should say so

§4.2 already permits "minimal execution bookkeeping, such as whether an assistant prefix is open … if introduced, include it in reset, recovery, and persistence rules." The first review's phase enum is exactly that, so adopt it — but bind it to that clause. The phase must appear in the `Restored`/`Reset` checkpoint account in §4.3 and in F4's persistence identity, or it becomes a fifth piece of untracked state.

### 3.4 P0 is one gate blocking everything

§7 already lets "unrelated additive loading work" bypass R0 and R1, but not P0. `ModelSource` / `ModelLoader` have no coupling to the state machine, the recovery ladder, or the benchmark freeze.

**Edit:** adopt the first review's P0.1 / P0.2 split, and additionally put the loading strand on its own track so the source and constructor inventory is not held behind Swift/Kotlin prototypes and frozen latency budgets.

### 3.5 §8.1's frozen budgets need named confounds

"Derived from baseline variability" is the right instinct, but this repository has documented measurement traps that will silently invalidate a frozen budget:

- Host loadavg swings Mac decode throughput by roughly 50%.
- The Android device is shared and carries both thermal and memory-pressure confounds; its `dumpsys` thermal output prints a stale cached block before the live one.
- GPU models retain KV, conv, and prefix-cache state on the **model**, not the session. Reusing a model across measurements has already produced two false bug reports in this repo.

**Edit:** §8.1 requires a fresh model instance per measurement, and interleaved A/B runs reported as ratios, not just repeat counts and medians.

---

## 4. Recommended plan edits

1. **§0** — add the R1 carve-out: warm-path prompt tokens may change; record it as a declared behavior change with fixtures, exempt from the equivalent-call parity rule.
2. **§3/I1 + §4.3** — correct the GPU truncate framing (seq_len only, conv buffers stale, no guard); require a pre-mutation capability query on `Model::truncate_kv`; require `truncate_to` to become fallible rather than asserting.
3. **§4.2** — record the emitted-vs-committed boundary as a determined fact; state `n_keep = 0` as the Part I default for high-level chat; adopt `SessionPhase` and `ingest_messages`, bound to the existing bookkeeping clause.
4. **§4.4** — replace the proposed `CancelHandle` object with the handle-identity invariant; add the greedy-vs-stochastic double-`complete()` divergence to the phase contract.
5. **§2 / §7 P0** — add the "warm today vs. reset today" column per first-party surface; split the loading strand off the P0 critical path; adopt P0.1 / P0.2.
6. **§8 / §8.1** — add the full-render-vs-incremental differential tokenization test as an R1 exit gate; add the first review's 10-turn drift test; add fresh-model-per-measurement and interleaved A/B to the benchmark protocol.
