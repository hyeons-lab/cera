# Architectural Review: Cera Public API Reshape Plan

**Target Document:** [API_RESHAPE_PLAN.md](file:///Users/dberrios/development/cera/docs/internals/API_RESHAPE_PLAN.md)  
**Baseline Commit:** `60fc11c25a51` (`origin/main`), 2026-09-06  
**Review Focus:** Architectural coherence, state invariants, failure recovery feasibility, binding concurrency, and execution risks.

---

## 1. Executive Summary & Verdict

The proposed API Reshape Plan represents a mature, technically grounded, and rigorous overhaul of Cera's public surface. It directly addresses the primary performance bottleneck in Cera's existing high-level chat interface: the $O(N^2)$ prefill tax caused by re-rendering and re-prefilling complete conversation history on every turn.

Crucially, the plan resists common anti-patterns:
- It refuses to fake transactional rollbacks by introducing expensive full-cache snapshots on normal paths.
- It rejects papering over context resets by calling them "warm" wrappers.
- It decouples prerequisites: ingestion recovery correctness (R0) and warm-chat profile rendering (R1) are mandatory gates that must be proven before exposing high-level warm-chat promises.

**Verdict: Approved with Critical Recommendations.** The architectural foundation is sound. However, there are subtle state machine ambiguities, foreign binding concurrency constraints, and edge-case recovery gaps that should be formalized in Phase P0 before committing signatures to code.

---

## 2. Key Architectural Strengths

1. **Non-Negotiable Live KV Reuse as a Release Gate (§0, §8.1):**
   Treating retained KV execution as an unconditional release requirement backed by strict mechanical gates (§8.1) prevents regressions where convenient abstractions silently trigger full-history re-prefills or background copies.
2. **Prerequisite Decoupling (R0 and R1):**
   Splitting failure recovery correctness (R0) and initial warm-chat profile validation (R1) away from additive API wrappers (P1) ensures that foundational backend behavior is verified before high-level callers rely on it.
3. **Strict Transcript Ownership Boundary (Decision D7):**
   Locking D7 (`Session` owns only execution cursor and KV; caller owns transcript `Vec<Message>`) avoids bloating the engine with chat state synchronization, message history serialization, or UI state management.
4. **Preservation of Low-Level Inference and Modality Primitives (§2, §3, §4.5):**
   Raw prefill (`append_tokens`, `append_embeddings`), audio engines, Whisper transcription, BERT encoders, and TurboQuant compression are explicitly inventoried and preserved rather than casually discarded.

---

## 3. Critical Technical Findings & Analysis

### Finding 1: Turn Boundary Ingestion and Generation State Machine

#### The Issue
The plan sketches two primary turn operations:
```rust
session.ingest(next_message)?;
let turn = session.complete(&options)?;
```
In modern chat templates (e.g. ChatML, Llama-3, Qwen), a user turn consists of:
`{header}user\n{text}{eot}` followed by an assistant generation prompt `{header}assistant\n`.

If `session.ingest(Message::user(...))` appends both the user block and the assistant prompt to the KV cache:
1. **Multi-message Ingestion:** If a caller needs to ingest multiple messages (e.g. user message followed by a tool response or image part), calling `ingest()` twice will inject an extra assistant generation prompt in the middle of the turn.
2. **Cancelled Generation State:** If `session.complete()` or `session.generate_into()` is cancelled mid-generation, the KV cache contains a partial assistant response without an end-of-turn delimiter (`<|im_end|>`). If the caller subsequently calls `session.ingest(next_user_message)`, the model receives syntactically malformed conversation history.
3. **Double Completion:** What occurs if `session.complete()` is called twice without an intervening `ingest()`? In greedy mode, [Session::generate](file:///Users/dberrios/development/cera/cera/src/session.rs#L2584-L2588) currently sets `self.last_logits = None` when `generated > 0`. A subsequent `complete()` immediately returns `Err(CeraError::EmptyInput)`.

#### Recommendation for P0
Formalize an explicit `SessionPhase` state machine on `Session`:
- `Idle` (fresh or awaiting input)
- `PromptReady` (context ingested, assistant prefix primed, ready for `complete`)
- `TurnComplete` (assistant generation completed with EOS/stop token, ready for `ingest`)
- `Interrupted` (generation cancelled or aborted; requires explicit continuation or `replace_messages`)
- `Unusable` (unrecoverable ingestion/backend failure)

Provide a batch ingestion method:
```rust
pub fn ingest_messages<I>(&mut self, messages: I) -> Result<IngestSummary, IngestError>
where I: IntoIterator<Item = Message>;
```
Ensure that calling `ingest` while in `Interrupted` phase is rejected unless the caller explicitly closes the turn or replaces context.

---

### Finding 2: Tension Between Transcript-Free Sessions (D7) and KV Context Shifts (`n_keep`)

#### The Issue
Under Decision D7, `Session` retains zero message objects or turn markers.
When the KV cache reaches `max_seq_len`:
- If `n_keep == 0`, `append_tokens` fails with `CeraError::ContextOverflow`.
- If `n_keep > 0`, `shift_kv` evicts KV cells `[n_keep .. n_keep + shift_needed)`.

Lines 344:348 of the plan note:
> "After custom raw input, context eviction, or any mutation invalidating the rendering cursor, resume high-level ingestion only with a proven cursor transition; otherwise reject before further mutation and require explicit replacement."

Because `shift_kv` drops raw token slices without knowledge of message boundaries:
1. An eviction can slice across the middle of a user message or subword token sequence.
2. `IngestSummary` can report the number of evicted tokens, but the caller owns `messages: Vec<Message>`. The caller has no direct mechanism to map evicted token indices back to their application `Message` objects.
3. If eviction permanently invalidates the high-level rendering cursor, `n_keep` becomes unusable for high-level multi-turn chat: the very first context shift forces the caller to fall back to `replace_messages(&messages)`, which will immediately overflow again unless the caller manually prunes their `Vec<Message>`.

#### Recommendation for P0
Clarify the operational boundary for `n_keep` in high-level chat:
- **Option A (Recommended for High-Level Chat):** For `ingest` and `complete`, enforce that context eviction is handled at the *message level* by the caller (using `replace_messages` with a pruned message window), keeping `n_keep = 0` as the default for high-level chat sessions.
- **Option B (Engine-level Sliding Window):** If `n_keep > 0` is supported during high-level `ingest`, specify how the incremental chat profile tracks boundary tokens across an eviction, and provide an explicit API on `IngestSummary` detailing the evicted token span so callers can synchronize their transcript.

---

### Finding 3: Foreign Language Concurrency, Cancellation Handles, and Sink Semantics

#### The Issue
1. **UniFFI Locking Contention:**
   In [cera-ffi](file:///Users/dberrios/development/cera/cera-ffi/src/lib.rs), UniFFI exports objects using reference types (`Arc<Session>`). Because `Session` methods require `&mut self` (`ingest`, `complete`, `generate_into`), the FFI layer wraps the session in a mutex.
   If a thread starts a blocking `session.complete(&options)` or `session.generate_streaming(sink)`, `Mutex<Session>` is held continuously.
   If cancellation requires calling a method on `session`, the cancellation call will deadlock or block until generation finishes.
   The plan states:
   > "A cancel request must be possible without borrowing or locking the active session."
2. **Core vs. Foreign Sink Distinction:**
   In core Cera, [ModalitySink](file:///Users/dberrios/development/cera/cera/src/session.rs#L302-L305) receives token IDs (`on_text_tokens(&[u32])`). In foreign bindings, [cera-ffi ForeignSinkAdapter](file:///Users/dberrios/development/cera/cera-ffi/src/lib.rs#L1854) decodes tokens into UTF-8 strings (`on_text_chunk(String)` and `on_thought_chunk(String)`).

#### Recommendation for P0
1. Explicitly codify a standalone `CancelHandle` UniFFI object:
   ```rust
   pub struct CancelHandle(Arc<AtomicBool>);
   impl CancelHandle {
       pub fn cancel(&self) { self.0.store(true, Ordering::Relaxed); }
   }
   ```
   The foreign caller obtains `cancel_handle()` from the session *before* calling `complete`/`generate_into`, ensuring that cancellation never contends with the session lock.
2. In the design specification, maintain clear nomenclature distinguishing the core Rust `ModalitySink` (token IDs) from the UniFFI `ForeignModalitySink` (UTF-8 text chunks and thoughts).

---

### Finding 4: R0 Recovery Guarantees vs. Backend Physical Reality

#### The Issue
Section 4.3 defines four caller-visible recovery outcomes:
- `Unchanged`: Rejected prior to mutation.
- `Restored`: Execution state completely reverted to pre-call checkpoint.
- `Reset`: Prior context cleared; position zero.
- `Unusable`: State corrupted; requires explicit recreation.

In Cera's codebase:
1. **TurboQuant:** [truncate_to](file:///Users/dberrios/development/cera/cera/src/kv_cache.rs#L1100-L1103) explicitly panics on compressed caches:
   ```rust
   assert!(!self.is_compressed(), "truncate_to called on a TurboQuant-compressed state; not supported");
   ```
2. **LFM2 Hybrid Conv Layers:** [truncate_to](file:///Users/dberrios/development/cera/cera/src/kv_cache.rs#L1113-L1125) checks `history.has_pos(safe_len)`. If prefill advanced past the 64-entry ring buffer, it forces `safe_len = 0` (full reset).
3. **GPU / Metal:** `truncate_kv` does not rewind GPU-allocated attention matrices or device-side counters.

As a result, `Restored` is physically impossible for GPU, TurboQuant, and large hybrid-conv prefills without full-cache copies (which the plan rightly forbids). Therefore, any prefill failure in these configurations will collapse directly to `Reset` or `Unusable`.

#### Recommendation for P0
1. Amend [cera::kv_cache::InferenceState::truncate_to](file:///Users/dberrios/development/cera/cera/src/kv_cache.rs#L1094) to return a fallible `Result<(), ()>` rather than asserting, so that R0 never triggers a panic on TurboQuant or out-of-range targets.
2. In the caller contract for `IngestError`, highlight that `Reset` is the dominant recovery outcome on GPU and compressed backends. Callers must design their retry workflows to re-run `replace_messages` with sanitized input when `Reset` occurs.

---

### Finding 5: Template Token Merging at Incremental Turn Boundaries (R1)

#### The Issue
In Byte-Pair Encoding (BPE), tokenization is context-sensitive across whitespace and punctuation boundaries.
For example:
- Full render: `\n<|im_start|>user\n` might tokenize as `[TokenA, TokenB]`.
- Sliced render: Tokenizing `\n` followed separately by `<|im_start|>user\n` might produce `[TokenC, TokenD, TokenE]` due to different merge-rule activations at the prefix boundary.

If incremental rendering tokenizes message fragments independently of the preceding committed tokens, token divergence will occur, violating prompt parity against the batch Jinja renderer.

#### Recommendation for P0 / R1
For each frozen warm-chat profile in R1 (e.g. ChatML, Llama-3, Qwen-2.5):
1. Specify explicit, pre-tokenized delimiter constants for turn boundaries (e.g. static token ID slices for `<|im_end|>\n<|im_start|>user\n`) rather than dynamically re-tokenizing string template fragments.
2. Add a differential test in §8 comparing:
   - Full Jinja render tokenized as a single monolithic string.
   - Incremental sequence generated turn-by-turn across `ingest` and `complete`.
   Any mismatch must fail the R1 gate.

---

## 4. Phasing, Scope, and Execution Analysis

| Phase | Strengths & Deliverables | Latent Risks | Mitigation |
|---|---|---|---|
| **P0** | Prototypes in Rust, Swift, Kotlin; feature/config inventories; benchmark budgets. | Risk of scope creep stalling Part I delivery. | Split P0 into two clear milestones: **P0.1** (Core Rust API, state machine, and error types) and **P0.2** (Foreign language prototypes and benchmark baseline freezing). |
| **R0** | Fixes dangerous [append_user_message](file:///Users/dberrios/development/cera/cera/src/session.rs#L2033-L2043) rollback and unifies cancellation semantics. | Backend-specific state rewind differences (CPU vs. Metal vs. TurboQuant). | Implement fault-injection suite first; verify graceful `Reset` and `Unusable` reporting when rewind is unsupported. |
| **R1** | Establishes token parity and continuation rules for initial warm profiles. | Complex Jinja templates with conditional system prompts or whitespace trimming. | Restrict initial R1 matrix to 2-3 well-behaved models (e.g. Qwen 2.5, Llama 3.2, LFM2) with verified static turn delimiters. |
| **P1** | Additive types (`ModelLoader`, `ModelSource`, `GenerativeModel`, `TurnResult`). | High churn across integration test fixtures. | Keep old config and loader entry points fully functional with deprecation warnings. |
| **P2** | Migration of CLI, FFI, WASM, and language bindings. | Silent divergence between old replacement loop and new warm continuation. | Run parity benchmarks checking token IDs, RNG progression, and repetition penalty behavior. |
| **P3** | Breaking cleanup release after deprecation window. | Premature removal breaking downstream clients. | Strictly enforce minimum one minor version grace period with compile-time deprecations. |

---

## 5. Summary of Recommended Plan Edits

Before moving from plan approval to execution, the following targeted refinements should be incorporated into `API_RESHAPE_PLAN.md`:

1. **Section 1.2 & 4.2 (State Machine & Boundaries):**
   - Add `SessionPhase` to formalize valid transition sequences between `ingest` and `complete`.
   - Add `ingest_messages` for multi-message turn support.
   - Explicitly forbid calling `ingest` on an interrupted turn without replacement.
2. **Section 1.3 & 4.4 (Foreign Cancellation):**
   - Document `SessionCancelHandle` as a lock-free UniFFI object separate from `Session`.
   - Clarify `session.clear_cancel()` usage when retrying after a cancelled prefill.
3. **Section 3 (Constraint I1) & Section 4.3 (Recovery):**
   - Note that `truncate_to` must become non-panicking on compressed caches.
   - Clarify that `Reset` (session cleared to zero) is the standard recovery outcome for GPU and TurboQuant models.
4. **Section 4.2 & 4.5 (Context Shifting vs. D7):**
   - Document whether `n_keep` context shifting is permitted during warm chat or if caller-managed message windowing via `replace_messages` is preferred.
5. **Section 8.1 (Verification):**
   - Add a 10-turn multi-turn drift test to the performance and parity test matrix.
