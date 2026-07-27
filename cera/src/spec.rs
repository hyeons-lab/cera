//! Speculative decoding: drafting and greedy accept/reject helpers.
//!
//! Phase 1 targets the memory-bandwidth wall that caps CPU decode-at-depth
//! (mobile especially): a target forward reads all weights once per token, so
//! verifying K drafted tokens in ONE forward amortizes that single weight-read
//! over up to K accepted tokens. This module holds the model-free pieces —
//! prompt-lookup drafting and the greedy accept rule — so they can be unit
//! tested without a GGUF. The orchestration lives in `session`.

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
/// This is pure verification — it does **not** consult EOS / stop tokens / token
/// budgets. Callers that must stop partway through the accepted run emit the
/// accepted tokens under their own stop policy and then `state.truncate_to` back
/// to the number they kept (and ignore `follow_logits`). Keeping the policy in
/// the caller lets both the standalone [`greedy_generate_spec`] driver and the
/// streaming `Session` path share this exact accept/truncate logic.
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
    state.truncate_to(old + 1 + m);
    // Row m holds the logits for the position after the last kept token.
    let follow_logits = all[m * vocab..(m + 1) * vocab].to_vec();
    VerifyResult {
        accepted,
        follow_logits,
    }
}

/// Greedy speculative decoding with prompt-lookup drafting. Produces output that
/// is **token-for-token identical** to greedy decoding of `model` (the target's
/// argmax at every position is the ground truth; drafts only shortcut the
/// weight-reads), so it can be verified against a plain greedy loop.
///
/// The `state` must be fresh (`seq_len == 0`). `eos` lists stop tokens. `ngram`
/// and `k` configure the drafter. Returns the generated tokens (excluding the
/// prompt) and acceptance stats.
///
/// Dense (pure-attention) models only in Phase 1 — [`crate::kv_cache::InferenceState::truncate_to`]
/// panics on LFM2 conv layers, which are not position-indexed.
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
    // cache reaches `truncate_to`'s own assert several frames deep, after the KV
    // has already been written; and a model without all-position logits panics
    // inside `verify_draft`'s row indexing.
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
         rewinds with `truncate_to`, which TurboQuant's packed layout cannot do"
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
            state.truncate_to(old + 1 + kept);
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
}
