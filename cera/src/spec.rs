//! Speculative decoding: drafting and greedy accept/reject helpers.
//!
//! Phase 1 targets the memory-bandwidth wall that caps CPU decode-at-depth
//! (mobile especially): a target forward reads all weights once per token, so
//! verifying K drafted tokens in ONE forward amortizes that single weight-read
//! over up to K accepted tokens. This module holds the model-free pieces —
//! prompt-lookup drafting and the greedy accept rule — so they can be unit
//! tested without a GGUF. The orchestration lives in `session`.

/// A drafter proposes speculative draft tokens given sequence history.
pub trait Drafter: Send + Sync {
    /// Create an isolated clone/instance of the drafter for a new session.
    fn clone_drafter(&self) -> Box<dyn Drafter>;
    /// Reset internal state (e.g. when session history is cleared or truncated).
    fn reset(&mut self);
    /// Propose up to `max_k` draft tokens given the sequence history so far.
    fn draft(&mut self, tokens: &[u32], max_k: usize) -> Vec<u32>;
    /// Suggested speculation depth (k) if configured for this drafter.
    fn suggested_k(&self) -> Option<usize> {
        None
    }
}

/// Prompt-lookup (n-gram) drafter implementation.
#[derive(Debug, Clone, Copy)]
pub struct PromptLookupDrafter {
    pub ngram: usize,
}

impl PromptLookupDrafter {
    pub fn new(ngram: usize) -> Self {
        Self { ngram }
    }
}

impl Drafter for PromptLookupDrafter {
    fn clone_drafter(&self) -> Box<dyn Drafter> {
        Box::new(*self)
    }

    fn reset(&mut self) {}

    fn draft(&mut self, tokens: &[u32], max_k: usize) -> Vec<u32> {
        prompt_lookup_draft(tokens, self.ngram, max_k)
    }
}

/// Prompt-lookup (n-gram) draft: guess the next tokens by finding the most
/// recent earlier occurrence of the last `ngram` tokens and returning up to `k`
/// tokens that followed it. No draft model — the guess comes from the sequence
/// itself, which is why this shines on repetitive / long-context generation
/// (code, structured output, quoted text).
///
/// Returns an empty `Vec` when there is no match, when `tokens` is too short, or
/// when `ngram`/`k` is zero. The caller then falls back to a normal decode step,
/// so an empty draft is never wrong — only a missed speedup opportunity.
///
/// Correctness of the overall decode does NOT depend on draft quality: every
/// drafted token is verified against the target's own logits and rejected on
/// mismatch. A poor draft only lowers the acceptance rate.
pub fn prompt_lookup_draft(tokens: &[u32], ngram: usize, k: usize) -> Vec<u32> {
    let n = tokens.len();
    if ngram == 0 || k == 0 || n <= ngram {
        return Vec::new();
    }
    let pattern = &tokens[n - ngram..];
    // Scan earlier start positions, most recent first, for a match of the last
    // `ngram` tokens. The freshest match gives the most relevant continuation.
    for start in (0..n - ngram).rev() {
        if &tokens[start..start + ngram] == pattern {
            let follow = start + ngram;
            let end = (follow + k).min(n);
            return tokens[follow..end].to_vec();
        }
    }
    Vec::new()
}

/// Acceptance statistics for a speculative-decoding run (diagnostics / bench).
#[derive(Debug, Default, Clone, Copy)]
pub struct SpecStats {
    /// Draft tokens proposed across all verification rounds.
    pub drafted: usize,
    /// Draft tokens accepted (matched the target's greedy argmax).
    pub accepted: usize,
    /// Verification rounds run (rounds where a non-empty draft was verified).
    pub rounds: usize,
}

impl SpecStats {
    /// Fraction of drafted tokens that were accepted (0.0 when nothing drafted).
    pub fn acceptance_rate(&self) -> f32 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f32 / self.drafted as f32
        }
    }
}

/// Result of verifying one draft against the target in a single forward.
pub struct VerifyResult {
    /// The accepted draft prefix, in order — the longest run of drafted tokens
    /// whose value equals the target's greedy argmax at that position. May be
    /// empty (first draft already mismatched). Never longer than the draft.
    pub accepted: Vec<u32>,
    /// The target's logits for the position immediately after the guaranteed
    /// token plus `accepted` — the "bonus" token when every draft matched, or
    /// the correcting token at the first mismatch. Length == vocab.
    pub follow_logits: Vec<f32>,
}

/// Verify `[guaranteed, draft...]` in a single target forward and accept the
/// longest greedy-matching draft prefix.
///
/// On entry `state.seq_len` is `old`, the position the guaranteed token will
/// occupy — it is **not** yet in the KV. The batch `[guaranteed, draft...]` is
/// fed at `[old .. old + 1 + draft.len()]`, appending all of them; on return the
/// KV holds exactly the guaranteed token plus the accepted drafts (rejected
/// drafts are truncated away), i.e. `state.seq_len == old + 1 + accepted.len()`.
///
/// This is pure verification: it does **not** consult EOS / stop tokens / token
/// budgets. Callers that must stop partway through the accepted run emit the
/// accepted tokens under their own stop policy and then rewind, via
/// [`crate::model::Model::truncate_kv`], back to the number they kept (and
/// ignore `follow_logits`). Keeping the policy in the caller lets both the
/// standalone [`greedy_generate_spec`] driver and the streaming `Session` path
/// share this exact accept/truncate logic.
pub fn verify_draft(
    model: &dyn crate::model::Model,
    state: &mut crate::kv_cache::InferenceState,
    guaranteed: u32,
    draft: &[u32],
    vocab: usize,
) -> VerifyResult {
    use crate::sampler::argmax;

    let old = state.seq_len;
    let mut batch = Vec::with_capacity(1 + draft.len());
    batch.push(guaranteed);
    batch.extend_from_slice(draft);
    let all = model.forward_prefill_logits_all(&batch, old, state);
    // A real assert, not `debug_assert`: everything below slices `all` by
    // `j * vocab`, so a backend returning the wrong row count would otherwise
    // surface in release as an out-of-range slice panic several lines away, or —
    // worse, if it returned *more* than expected — as silently reading the wrong
    // row and accepting a draft token the target never predicted. This is once
    // per verification round, not per token.
    assert_eq!(
        all.len(),
        batch.len() * vocab,
        "forward_prefill_logits_all returned {} logits for {} positions x vocab {vocab}; \
         the row-major [n x vocab] contract is what lets `verify_draft` index row j",
        all.len(),
        batch.len()
    );

    // Row j predicts the token at position old+j+1, i.e. the token that should
    // follow batch[j]. Accept draft[j] while it equals that argmax.
    let mut accepted = Vec::new();
    for (j, &q) in draft.iter().enumerate() {
        if argmax(&all[j * vocab..(j + 1) * vocab]) != q {
            break;
        }
        accepted.push(q);
    }
    let m = accepted.len();
    // Keep the guaranteed token + m accepted drafts; drop the rejected tail.
    // Through the model, not `state.truncate_to`: see `Model::truncate_kv`.
    model.truncate_kv(state, old + 1 + m);
    // Row m holds the logits for the position after the last kept token.
    let follow_logits = all[m * vocab..(m + 1) * vocab].to_vec();
    VerifyResult {
        accepted,
        follow_logits,
    }
}

/// Greedy speculative decoding with prompt-lookup drafting. Every emitted token
/// is the target's argmax at its position — drafts only shortcut the
/// weight-reads, never the decision — so the result is a valid greedy decode.
///
/// It is **not** guaranteed token-for-token equal to a *sequential* greedy loop.
/// Verification forwards the drafted tokens as a batch (`n > 1` takes the
/// batched-GEMM path in `forward_prefill_logits_all`) where a sequential loop
/// forwards one token at a time through a GEMV. The two reduction orders are
/// not bit-equal, so where the top two logits are near-tied the two paths can
/// pick opposite sides. Oracle tests therefore bound the logit gap rather than
/// asserting argmax equality against a plain greedy run.
///
/// The `state` must be fresh (`seq_len == 0`). `eos` lists stop tokens. `ngram`
/// and `k` configure the drafter. Returns the generated tokens (excluding the
/// prompt) and acceptance stats.
///
/// Models with short-conv layers (such as LFM2) preserve convolution state history
/// via `ConvHistory` ring buffers to rewind cleanly during verification rewinds.
pub fn greedy_generate_spec(
    model: &dyn crate::model::Model,
    state: &mut crate::kv_cache::InferenceState,
    prompt: &[u32],
    max_new: usize,
    eos: &[u32],
    ngram: usize,
    k: usize,
) -> (Vec<u32>, SpecStats) {
    use crate::sampler::argmax;

    let mut out: Vec<u32> = Vec::new();
    let mut stats = SpecStats::default();
    // The three documented preconditions, enforced at the boundary. Each has a
    // failure mode that is much harder to read further in: a stale `state` makes
    // every position off by `seq_len` (wrong output, no panic); a compressed
    // cache reaches the assert inside `truncate_to` (via `truncate_kv`) several
    // frames deep, after the KV has already been written; and a model without
    // all-position logits panics inside `verify_draft`'s row indexing.
    assert!(
        model.supports_all_logits(),
        "greedy_generate_spec requires forward_prefill_logits_all support"
    );
    assert_eq!(
        state.seq_len, 0,
        "greedy_generate_spec requires a fresh InferenceState (seq_len == 0); \
         this one holds {} positions, so the prompt would prefill after them and \
         every position would be offset",
        state.seq_len
    );
    assert!(
        !state.is_compressed(),
        "greedy_generate_spec requires an uncompressed KV cache: verification \
         rewinds the KV, which TurboQuant's packed layout cannot do"
    );
    if prompt.is_empty() || max_new == 0 {
        return (out, stats);
    }

    // Prefill the prompt; `next_logits` predicts the token at position seq_len.
    let mut next_logits = model.forward_prefill(prompt, 0, state);
    let vocab = next_logits.len();
    // Running full token sequence (prompt + emitted) for the prompt-lookup drafter.
    let mut history: Vec<u32> = prompt.to_vec();
    let is_eos = |t: u32| eos.contains(&t);

    loop {
        if out.len() >= max_new {
            break;
        }
        // The next greedy token is always correct. Stop semantics match a plain
        // greedy decode and the `Session` spec path: an EOS token is NOT emitted
        // (checked before the push), while the `max_new` budget cap DOES emit the
        // token and then stops. Keeping these identical to `Session` is what lets
        // `session_spec_matches_standalone_driver` cross-check the two paths.
        let t = argmax(&next_logits);
        if is_eos(t) {
            break;
        }
        out.push(t);
        history.push(t);
        if out.len() >= max_new {
            break;
        }

        let draft = prompt_lookup_draft(&history, ngram, k);
        if draft.is_empty() {
            // No speculation available: a plain greedy step (appends t's K/V).
            next_logits = model.forward(&[t], state.seq_len, state);
            continue;
        }

        // Verify [t, draft...] in one forward and accept the longest matching
        // prefix (KV truncated to keep only the accepted run).
        let old = state.seq_len;
        stats.rounds += 1;
        stats.drafted += draft.len();
        let vr = verify_draft(model, state, t, &draft, vocab);

        // Emit accepted drafts under the greedy stop policy (budget / EOS), each
        // checked BEFORE the emit so a stopped token is neither counted nor kept
        // in the KV. On an early stop, roll the KV back to the tokens actually
        // kept and drop the unused `follow_logits`.
        let mut kept = 0usize;
        let mut stopped = false;
        for &q in &vr.accepted {
            if out.len() >= max_new || is_eos(q) {
                stopped = true;
                break;
            }
            out.push(q);
            history.push(q);
            kept += 1;
            stats.accepted += 1;
        }
        if kept < vr.accepted.len() {
            model.truncate_kv(state, old + 1 + kept);
        }
        if stopped {
            break;
        }
        // Every accepted draft was emitted; `follow_logits` predicts the next
        // position (the "bonus" token, or the correcting token at a mismatch).
        next_logits = vr.follow_logits;
    }

    (out, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_predicts_repeated_continuation() {
        // Last 2 tokens [10,20] occurred at index 0, followed by [30,10,20].
        let toks = [10u32, 20, 30, 10, 20];
        assert_eq!(prompt_lookup_draft(&toks, 2, 3), vec![30, 10, 20]);
        // k bounds the draft length.
        assert_eq!(prompt_lookup_draft(&toks, 2, 1), vec![30]);
    }

    #[test]
    fn draft_uses_most_recent_match() {
        // [7,8] appears at idx 0 (→9) and idx 3 (→11). The pattern is the last
        // two tokens; the most recent EARLIER match at idx 3 is skipped only if
        // it is the pattern itself. Here pattern = tokens[5..7] = [7,8]; earlier
        // matches at idx 0 (→ tokens[2]=9) and idx 3 (→ tokens[5]=7). Most recent
        // earlier start is 3 → continuation tokens[5..] = [7,8][..] guess.
        let toks = [7u32, 8, 9, 7, 8, 7, 8];
        // last [7,8] at 5..7; earlier starts matching [7,8]: 3 and 0. Freshest = 3,
        // follow = tokens[5..] = [7,8]. So draft = [7,8] (k=2).
        assert_eq!(prompt_lookup_draft(&toks, 2, 2), vec![7, 8]);
    }

    #[test]
    fn draft_empty_when_no_match() {
        assert!(prompt_lookup_draft(&[1, 2, 3, 4, 5], 2, 3).is_empty());
    }

    #[test]
    fn draft_empty_on_degenerate_args() {
        assert!(prompt_lookup_draft(&[1, 2, 3], 0, 3).is_empty());
        assert!(prompt_lookup_draft(&[1, 2, 3], 2, 0).is_empty());
        assert!(prompt_lookup_draft(&[1, 2], 2, 3).is_empty()); // len == ngram
        assert!(prompt_lookup_draft(&[], 2, 3).is_empty());
    }

    #[test]
    fn draft_respects_sequence_end() {
        // Match at idx 0 for [1,2], but only one token follows before we reach
        // the pattern region — draft is bounded by the sequence length.
        let toks = [1u32, 2, 9, 5, 1, 2];
        // pattern = last [1,2] at 4..6; earlier [1,2] at 0 → follow tokens[2..] up to k.
        assert_eq!(prompt_lookup_draft(&toks, 2, 5), vec![9, 5, 1, 2]);
    }

    /// Every KV rewind on the spec path must go through
    /// [`crate::model::Model::truncate_kv`], never `state.truncate_to` directly.
    ///
    /// No other test can see the difference: the default implementation forwards
    /// to `truncate_to`, so calling either one produces identical output,
    /// identical `seq_len`, and identical logits on every model in the tree
    /// today. The distinction only becomes observable on a backend that overrides
    /// the method, which is exactly the backend that does not exist yet. Without
    /// this test the indirection can be undone by a plausible-looking edit and
    /// nothing goes red until a GPU model silently decodes from the wrong
    /// positions.
    ///
    /// So: a stub model that records the rewinds it is asked to perform, and
    /// then performs them. If a call site regresses to `state.truncate_to`, the
    /// recording comes back short while everything else still passes.
    mod rewind_goes_through_the_model {
        use super::*;
        use crate::kv_cache::InferenceState;
        use crate::model::{BlockType, Model, ModelConfig};
        use std::sync::Mutex;

        /// Vocabulary size for the stubs, kept tiny so a one-hot row is 8 floats
        /// rather than a real vocabulary.
        ///
        /// The `% VOCAB` wrap in `row` is load-bearing, not defensive: the driver
        /// test prefills a prompt containing token 7, so `row(7)` indexes 8 and
        /// would panic without it.
        const VOCAB: u32 = 8;

        /// A dense (attention-only) config, so the default rewind path applies
        /// and `truncate_to` never meets a conv layer.
        fn dense_config() -> ModelConfig {
            let n_layers = 2;
            ModelConfig {
                architecture: "llama".into(),
                n_layers,
                hidden_size: 8,
                intermediate_size: 16,
                n_heads: 2,
                n_kv_heads: 2,
                head_dim: 4,
                vocab_size: VOCAB as usize,
                max_seq_len: 64,
                rope_theta: 10_000.0,
                rms_norm_eps: 1e-5,
                block_types: vec![BlockType::Attention; n_layers],
                conv_kernel_size: None,
                kv_heads_per_layer: vec![2; n_layers],
                scalars: crate::model::ScalarMultipliers::default(),
                moe: None,
            }
        }

        /// Predicts `(t + 1) % VOCAB` after every token, and records the rewinds
        /// it is asked to perform.
        struct CyclingStub {
            config: ModelConfig,
            /// Lengths passed to `truncate_kv`, in order.
            rewinds: Mutex<Vec<usize>>,
        }

        impl CyclingStub {
            fn new() -> Self {
                Self {
                    config: dense_config(),
                    rewinds: Mutex::new(Vec::new()),
                }
            }

            /// One-hot logits selecting `(t + 1) % VOCAB`.
            fn row(t: u32) -> Vec<f32> {
                let mut v = vec![0.0f32; VOCAB as usize];
                v[((t + 1) % VOCAB) as usize] = 1.0;
                v
            }

            fn recorded(&self) -> Vec<usize> {
                self.rewinds.lock().unwrap().clone()
            }
        }

        impl Model for CyclingStub {
            fn config(&self) -> &ModelConfig {
                &self.config
            }

            fn forward(&self, tokens: &[u32], _pos: usize, state: &mut InferenceState) -> Vec<f32> {
                // The KV caches stay empty; `truncate_to` treats an empty cache
                // as a no-op and only `seq_len` is under test here. The sibling
                // `DefaultStub` is the one that writes KV.
                state.seq_len += tokens.len();
                Self::row(*tokens.last().unwrap())
            }

            fn forward_prefill_logits_all(
                &self,
                tokens: &[u32],
                _start_pos: usize,
                state: &mut InferenceState,
            ) -> Vec<f32> {
                state.seq_len += tokens.len();
                tokens.iter().flat_map(|&t| Self::row(t)).collect()
            }

            fn supports_all_logits(&self) -> bool {
                true
            }

            fn truncate_kv(&self, state: &mut InferenceState, len: usize) {
                self.rewinds.lock().unwrap().push(len);
                state.truncate_to(len);
            }
        }

        /// `verify_draft` rewinds once per round, through the model.
        ///
        /// The driver test below would also catch a regression here, in its first
        /// recorded entry. This one exists to fail first and locally, naming the
        /// call site rather than a driver run that happens to route through it.
        #[test]
        fn verify_draft_rewinds_through_the_model() {
            let model = CyclingStub::new();
            let mut state = InferenceState::from_config(model.config()).unwrap();

            // Put the cache at a non-zero position, so a rewind length computed
            // from the wrong base would not coincidentally match.
            model.forward_prefill(&[0, 1, 2], 0, &mut state);
            let old = state.seq_len;
            assert_eq!(old, 3);

            // After 3 the model predicts 4, then 5, then 6. The third draft
            // token is wrong, so two are accepted.
            let vr = verify_draft(&model, &mut state, 3, &[4, 5, 0], VOCAB as usize);

            assert_eq!(vr.accepted, vec![4, 5]);
            assert_eq!(
                model.recorded(),
                vec![old + 1 + 2],
                "verify_draft must rewind via Model::truncate_kv, once, to the \
                 guaranteed token plus the accepted drafts"
            );
            assert_eq!(state.seq_len, old + 1 + 2);
        }

        /// The two tests either side of this one drive a stub that *overrides*
        /// `truncate_kv`, so they pin the call sites without ever running the
        /// default body. Gutting that default to a no-op passes both, and every
        /// other non-ignored test in the workspace: its only other coverage is
        /// the `#[ignore]`d `greedy_spec_matches_greedy_within_tie_tolerance`, and even that
        /// catches it only on a fixture whose rounds actually reject a draft
        /// (Llama-3.2-1B does; on Qwen3-0.6B every round accepts in full, so every
        /// rewind is the free `len == seq_len` case and a no-op default passes).
        ///
        /// So: the same rewind through a model that does **not** override, with
        /// KV actually written, asserting the caches shrank.
        #[test]
        fn the_default_truncate_kv_really_rewinds() {
            /// Same predictions as [`CyclingStub`], but without the override, and
            /// it writes one f32 per position per layer so `truncate_to` has
            /// something to cut. `kv_dim` of 1 keeps the arithmetic readable; the
            /// real stride is irrelevant to what is under test, since
            /// `truncate_to` recovers it by division.
            struct DefaultStub(ModelConfig);

            impl Model for DefaultStub {
                fn config(&self) -> &ModelConfig {
                    &self.0
                }
                fn forward(&self, _: &[u32], _: usize, _: &mut InferenceState) -> Vec<f32> {
                    // Required by the trait, reached by nothing here: this test
                    // drives `forward_prefill_logits_all` directly and so does
                    // `verify_draft`. Delegating would be wrong rather than
                    // merely unused, since `forward` owes one row and that
                    // function returns `n`.
                    unimplemented!("DefaultStub is driven through forward_prefill_logits_all")
                }
                fn forward_prefill_logits_all(
                    &self,
                    tokens: &[u32],
                    _start_pos: usize,
                    state: &mut InferenceState,
                ) -> Vec<f32> {
                    for layer in &mut state.layers {
                        if let crate::kv_cache::LayerState::Attention {
                            key_cache,
                            value_cache,
                            ..
                        } = layer
                        {
                            for &t in tokens {
                                key_cache.push(t as f32);
                                value_cache.push(t as f32);
                            }
                        }
                    }
                    state.seq_len += tokens.len();
                    tokens.iter().flat_map(|&t| CyclingStub::row(t)).collect()
                }
                fn supports_all_logits(&self) -> bool {
                    true
                }
            }

            let model = DefaultStub(dense_config());
            let mut state = InferenceState::from_config(model.config()).unwrap();

            model.forward_prefill_logits_all(&[0, 1, 2], 0, &mut state);
            assert_eq!(state.seq_len, 3);

            // Predicts 4 then 5; the third draft token is wrong, so two are kept
            // and the batch's last two positions must come back out of the KV.
            let vr = verify_draft(&model, &mut state, 3, &[4, 5, 0], VOCAB as usize);
            assert_eq!(vr.accepted, vec![4, 5]);
            assert_eq!(state.seq_len, 6);

            for layer in &state.layers {
                let crate::kv_cache::LayerState::Attention {
                    key_cache,
                    value_cache,
                    ..
                } = layer
                else {
                    unreachable!("dense config has only attention layers")
                };
                // 3 prefilled + [3, 4, 5] kept; the rejected 0 is gone.
                assert_eq!(
                    key_cache,
                    &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    "the default truncate_kv did not cut the rejected tail out of \
                     the key cache"
                );
                assert_eq!(value_cache.len(), 6);
            }
        }

        /// The driver's *second* rewind, the one that fires when a stop policy
        /// cuts an accepted run short, also goes through the model. It is a
        /// separate call site from the one above and regresses independently.
        #[test]
        fn early_stop_rewind_goes_through_the_model() {
            let model = CyclingStub::new();
            let mut state = InferenceState::from_config(model.config()).unwrap();

            // A full cycle plus two tokens. The prompt predicts `2`, which the
            // driver emits as the guaranteed token; the drafter then works on
            // the history `[0..8, 0, 1, 2]`, whose trailing bigram `[1, 2]` also
            // occurs at index 1, so it drafts that occurrence's continuation
            // `[3, 4, 5]` and the stub predicts exactly those.
            let prompt: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1];
            // `max_new = 3` cuts the accepted run: the round drafts 3 tokens and
            // the model accepts all of them, but the budget stops after 2.
            let (out, stats) = greedy_generate_spec(&model, &mut state, &prompt, 3, &[], 2, 3);

            assert_eq!(out, vec![2, 3, 4], "the stub decodes the cycle");
            assert_eq!(stats.accepted, 2, "budget cut the third accepted draft");

            let old = prompt.len();
            assert_eq!(
                model.recorded(),
                vec![old + 1 + 3, old + 1 + 2],
                "expected the per-round rewind (all 3 drafts accepted) followed \
                 by the early-stop rewind (only 2 kept), both through \
                 Model::truncate_kv"
            );
            assert_eq!(state.seq_len, old + 1 + 2);
        }
    }
}
