//! Phase 1a of speculative decoding: `forward_prefill_logits_all` must return,
//! for each position, logits whose argmax matches what per-token `forward`
//! produces at that position — because greedy speculative decoding verifies
//! drafts by comparing the target's per-position argmax to the drafted token.
//! If the batched all-logits argmax ever diverged from the per-token argmax by
//! more than a near-tie, greedy-spec output would differ from greedy output for
//! a real reason rather than a floating-point one.
//!
//! `#[ignore]` because it needs a real dense GGUF — see
//! `common::dense_model_or_skip` for the resolve order and for
//! `CERA_REQUIRE_DENSE_MODEL`, which turns a missing fixture into a hard failure
//! instead of a silent skip.
//!
//! Run: `cargo test -p cera --release --test spec_decode_logits -- --ignored --nocapture`

#![cfg(feature = "mmap")]

mod common;

use common::dense_model_or_skip;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// Streaming sink for the Session spec tests: collects every emitted token in
/// order so the test can compare the token stream.
struct Collect(Vec<u32>);
impl cera::ModalitySink for Collect {
    fn on_text_tokens(&mut self, t: &[u32]) {
        self.0.extend_from_slice(t);
    }
    fn on_done(&mut self, _r: cera::FinishReason) {}
}

/// Build a fresh text-only `Session` over `path` with the config the spec
/// Session tests share: an uncompressed KV (so spec engages) and monolithic
/// prefill (`ubatch_size = 0`, so the Session's prefill logits match the
/// standalone driver's `forward_prefill`). `max_seq_len` bounds the context.
fn build_spec_session(path: &std::path::Path, max_seq_len: Option<u32>) -> cera::Session {
    use std::sync::Arc;

    let gguf = cera::gguf::GgufFile::open(path).unwrap();
    let tokenizer = cera::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let model: Arc<dyn cera::model::Model> =
        Arc::from(cera::model::load_model(gguf, None, 8192).unwrap());
    cera::Session::new(
        model,
        Arc::new(tokenizer),
        cera::ModalityCapabilities::text_only(),
        cera::SessionConfig {
            kv_compression: cera::kv_cache::KvCompression::None,
            seed: None,
            ubatch_size: 0,
            max_seq_len,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
#[ignore = "needs a real dense GGUF; set CERA_DENSE_MODEL"]
fn all_logits_argmax_matches_per_token_forward() {
    let Some(path) = dense_model_or_skip() else {
        return;
    };

    // Batched all-position logits.
    let gguf_a = cera::gguf::GgufFile::open(&path).unwrap();
    let model_a = cera::model::load_model(gguf_a, None, 8192).unwrap();
    assert!(
        model_a.supports_all_logits(),
        "dense model must support forward_prefill_logits_all"
    );
    let cfg = model_a.config();
    let vocab = cfg.vocab_size;
    let tokens: Vec<u32> = vec![1, 15043, 3186, 297, 4223, 29889, 306, 626];
    let n = tokens.len();

    let mut state_a = cera::kv_cache::InferenceState::from_config(cfg).unwrap();
    let all = model_a.forward_prefill_logits_all(&tokens, 0, &mut state_a);
    assert_eq!(all.len(), n * vocab, "all-logits must be [n * vocab]");
    assert_eq!(state_a.seq_len, n, "KV must be appended for all n tokens");

    // Per-token forward reference.
    let gguf_b = cera::gguf::GgufFile::open(&path).unwrap();
    let model_b = cera::model::load_model(gguf_b, None, 8192).unwrap();
    let mut state_b = cera::kv_cache::InferenceState::from_config(cfg).unwrap();

    for (i, &tok) in tokens.iter().enumerate() {
        let per_tok = model_b.forward(&[tok], i, &mut state_b);
        let row = &all[i * vocab..(i + 1) * vocab];
        let a1 = argmax(row);
        let a2 = argmax(&per_tok);
        let cos = cosine(row, &per_tok);
        println!("pos {i}: argmax batched={a1} per-token={a2} cosine={cos:.5}");
        assert_eq!(
            a1, a2,
            "position {i}: batched all-logits argmax ({a1}) != per-token forward argmax ({a2})"
        );
        assert!(
            cos > 0.99,
            "position {i}: cosine {cos} too low (batched vs per-token drift)"
        );
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// `truncate_to(L)` must restore the KV cache to exactly its state after the
/// first L tokens: prefill 0..8 then truncate to 4 then prefill 4..8 must yield
/// the same logits as prefill 0..4 then prefill 4..8. Causal K/V for cell i
/// depends only on tokens 0..=i, so the truncated tail is genuinely restorable.
#[test]
#[ignore = "needs a real dense GGUF; set CERA_DENSE_MODEL"]
fn truncate_to_restores_kv_exactly() {
    let Some(path) = dense_model_or_skip() else {
        return;
    };
    let tokens: Vec<u32> = vec![1, 15043, 3186, 297, 4223, 29889, 306, 626];
    let (l, r) = (4usize, tokens.len());

    // Reference: prefill [0..l], then [l..r].
    let ga = cera::gguf::GgufFile::open(&path).unwrap();
    let ma = cera::model::load_model(ga, None, 8192).unwrap();
    let cfg = ma.config();
    let mut sa = cera::kv_cache::InferenceState::from_config(cfg).unwrap();
    ma.forward_prefill(&tokens[..l], 0, &mut sa);
    let ref_logits = ma.forward_prefill(&tokens[l..], l, &mut sa);

    // Test: prefill all [0..r], truncate to l, then re-prefill [l..r].
    let gb = cera::gguf::GgufFile::open(&path).unwrap();
    let mb = cera::model::load_model(gb, None, 8192).unwrap();
    let mut sb = cera::kv_cache::InferenceState::from_config(cfg).unwrap();
    mb.forward_prefill(&tokens, 0, &mut sb);
    assert_eq!(sb.seq_len, r);
    sb.truncate_to(l);
    assert_eq!(
        sb.seq_len, l,
        "seq_len must be reset to the truncation length"
    );
    let test_logits = mb.forward_prefill(&tokens[l..], l, &mut sb);
    assert_eq!(
        sb.seq_len, r,
        "re-prefill must grow the cache back to full length"
    );

    let d = max_abs_diff(&ref_logits, &test_logits);
    println!(
        "truncate_to: max_abs_diff = {d:.3e}, argmax ref={} test={}",
        argmax(&ref_logits),
        argmax(&test_logits)
    );
    assert_eq!(
        argmax(&ref_logits),
        argmax(&test_logits),
        "truncate+re-prefill changed the argmax"
    );
    assert!(
        d < 1e-3,
        "truncate_to did not restore the KV exactly (max_abs_diff = {d})"
    );
}

/// Plain greedy reference: argmax of the target's logits at every position.
fn greedy_reference(
    model: &dyn cera::model::Model,
    state: &mut cera::kv_cache::InferenceState,
    prompt: &[u32],
    max_new: usize,
) -> Vec<u32> {
    // Use the same argmax (tie-break, token type) the spec driver uses, so any
    // divergence is a real spec-decode bug, not an argmax-implementation mismatch.
    let mut next = model.forward_prefill(prompt, 0, state);
    let mut out: Vec<u32> = Vec::new();
    while out.len() < max_new {
        let t = cera::sampler::argmax(&next);
        out.push(t);
        if out.len() >= max_new {
            break;
        }
        next = model.forward(&[t], state.seq_len, state);
    }
    out
}

/// The oracle: greedy speculative decoding must track plain greedy decoding,
/// while (on repetitive text) actually accepting drafts — proving the verify +
/// accept + KV-truncate loop is correct.
///
/// "Track", not "equal": the batched verify forward and the sequential greedy
/// loop use different reduction orders, so a near-tie can flip. The assertion
/// below is therefore a logit-gap bound, not token equality — a real bug would
/// sit many logits below that gap.
#[test]
#[ignore = "needs a real dense GGUF; set CERA_DENSE_MODEL"]
fn greedy_spec_matches_greedy_within_tie_tolerance() {
    let Some(path) = dense_model_or_skip() else {
        return;
    };
    // A repetitive prompt so prompt-lookup drafts actually hit and the
    // verify/accept/truncate path is exercised (the gap bound below must hold
    // regardless of whether any draft is accepted).
    let prompt: Vec<u32> = vec![
        1, 450, 6635, 3290, 373, 278, 1775, 29889, 450, 6635, 3290, 373, 278,
    ];
    let max_new = 64usize;

    let gr = cera::gguf::GgufFile::open(&path).unwrap();
    let mr = cera::model::load_model(gr, None, 8192).unwrap();
    let cfg = mr.config();
    let mut sr = cera::kv_cache::InferenceState::from_config(cfg).unwrap();
    let reference = greedy_reference(mr.as_ref(), &mut sr, &prompt, max_new);

    let gs = cera::gguf::GgufFile::open(&path).unwrap();
    let ms = cera::model::load_model(gs, None, 8192).unwrap();

    // Isolation check: with drafting effectively disabled (ngram too large to ever
    // match), every step is a plain per-token forward — this MUST equal per-token
    // greedy bit-for-bit, proving the emit/accept/truncate orchestration is correct
    // independent of any batched-vs-sequential numerical difference.
    let mut ss0 = cera::kv_cache::InferenceState::from_config(cfg).unwrap();
    let (nodraft, s0) =
        cera::spec::greedy_generate_spec(ms.as_ref(), &mut ss0, &prompt, max_new, &[], 999, 6);
    assert_eq!(s0.accepted, 0, "ngram=999 should draft nothing");
    assert_eq!(
        reference, nodraft,
        "no-draft spec-decode must equal per-token greedy exactly (orchestration bug)"
    );

    let mut ss = cera::kv_cache::InferenceState::from_config(cfg).unwrap();
    let (spec, stats) =
        cera::spec::greedy_generate_spec(ms.as_ref(), &mut ss, &prompt, max_new, &[], 2, 6);

    println!(
        "spec: {} tokens, {} rounds, {}/{} drafts accepted ({:.0}% acceptance)",
        spec.len(),
        stats.rounds,
        stats.accepted,
        stats.drafted,
        stats.acceptance_rate() * 100.0
    );
    assert_eq!(reference.len(), spec.len(), "length mismatch");
    assert!(
        stats.accepted > 0,
        "expected some drafts to be accepted on a repetitive prompt (else the \
         verify/truncate path is untested)"
    );

    // Every token spec emits is, by construction, an argmax of the target's
    // logits at its position — so spec output is a valid greedy decode. It can
    // still differ from *per-token* greedy at a near-tie, because spec verifies
    // with a batched GEMM while per-token greedy uses the GEMV path (the same
    // batched-vs-sequential difference `blas_parity` documents). If they diverge,
    // prove the divergence is exactly that — a floating-point near-tie — and not
    // a bug in the accept/truncate logic.
    if reference != spec {
        let i = reference
            .iter()
            .zip(&spec)
            .position(|(a, b)| a != b)
            .unwrap();
        println!(
            "diverge at gen index {i}: per-token greedy={} vs spec={}",
            reference[i], spec[i]
        );

        // Re-forward the agreed prefix (prompt ++ reference[..i]) and read the
        // logits predicting position i, to compare spec's token against the
        // reference's on a common footing. NOTE: this is a *different* batch
        // shape than the short mid-sequence verify batch that actually produced
        // spec[i], so at a genuine near-tie the two argmaxes can land on
        // opposite sides — we therefore assert a small logit *gap*, not argmax
        // equality. What a real accept/truncate bug cannot fake is the gap:
        // emitting the wrong token (a rejected draft, a shifted row) would leave
        // spec[i] many logits below the max, not a hair away from it.
        let mut seq = prompt.clone();
        seq.extend_from_slice(&reference[..i]);
        let gv = cera::gguf::GgufFile::open(&path).unwrap();
        let mv = cera::model::load_model(gv, None, 8192).unwrap();
        let mut sv = cera::kv_cache::InferenceState::from_config(cfg).unwrap();
        let all = mv.forward_prefill_logits_all(&seq, 0, &mut sv);
        let vocab = cfg.vocab_size;
        let last = &all[(seq.len() - 1) * vocab..];
        let batched_pred = argmax(last) as u32;
        let g_spec = last[spec[i] as usize];
        let g_ref = last[reference[i] as usize];
        let gap = (g_spec - g_ref).abs();
        println!(
            "batched verifier argmax at pos {i} = {batched_pred}; logit(spec)={g_spec:.4} logit(ref)={g_ref:.4} gap={gap:.4}"
        );
        // 0.05 is ~2 orders of magnitude above the observed tie gaps (~1e-3) yet
        // far below any real bug's deficit (many logits). Both tokens are valid
        // greedy choices within this margin.
        assert!(
            gap < 0.05,
            "spec[{i}]={} sits {gap:.4} below reference[{i}]={} under the batched \
             re-forward — too large for a near-tie flip; accept/truncate likely buggy",
            spec[i],
            reference[i]
        );
    } else {
        println!("greedy-spec matched per-token greedy exactly (no near-tie flips)");
    }
}

/// Session wiring oracle: `Session::generate` with `spec` set must emit exactly
/// the same tokens as the standalone `greedy_generate_spec` driver on the same
/// prompt/config. Both drive the identical forward sequence (monolithic prefill
/// via `ubatch_size = 0`, then the shared `verify_draft` accept/truncate loop),
/// so equality is exact — this pins the `Session` layer (emit/flush, stop
/// policy, `truncate_kv` on partial accept, `current_pos` bookkeeping) against
/// the tested core, catching any drift the core's own oracle can't see.
///
/// Note what it does *not* pin: the `Session`'s partial-accept `truncate_kv`
/// call. That line can be deleted with this test still green. Reaching it means
/// the emit loop broke, so the generate loop breaks a few statements later
/// without anything reading `seq_len` or the KV again, and `current_pos` is
/// assigned from `old + 1 + kept` independently of the rewind. Catching it would
/// take a second `generate` on the same session whose budget landed strictly
/// inside an accepted run; the stub-model guards in `spec::tests` cannot build a
/// `Session` at all. Left uncovered deliberately, and noted at the call site.
#[test]
#[ignore = "needs a real dense GGUF; set CERA_DENSE_MODEL"]
fn session_spec_matches_standalone_driver() {
    use cera::{GenerateOpts, SpecDecode};

    let Some(path) = dense_model_or_skip() else {
        return;
    };
    // Repetitive prompt so drafts actually hit (same rationale as the oracle).
    let prompt: Vec<u32> = vec![
        1, 450, 6635, 3290, 373, 278, 1775, 29889, 450, 6635, 3290, 373, 278,
    ];
    let max_new = 64u32;
    let sd = SpecDecode { ngram: 2, k: 6 };

    // Standalone driver reference.
    let gr = cera::gguf::GgufFile::open(&path).unwrap();
    let mr = cera::model::load_model(gr, None, 8192).unwrap();
    let cfg = mr.config();
    let mut sr = cera::kv_cache::InferenceState::from_config(cfg).unwrap();
    let (driver, dstats) = cera::spec::greedy_generate_spec(
        mr.as_ref(),
        &mut sr,
        &prompt,
        max_new as usize,
        &[], // ignore EOS, matching ignore_eos below
        sd.ngram,
        sd.k,
    );
    assert!(
        dstats.accepted > 0,
        "expected accepted drafts (else the shared path is untested)"
    );

    // Session path: same weights, monolithic prefill + uncompressed KV (see
    // `build_spec_session`). A separate model instance is deterministic.
    let mut session = build_spec_session(&path, None);
    session.append_tokens(&prompt).unwrap();
    let mut sink = Collect(Vec::new());
    let summary = session
        .generate(
            &GenerateOpts {
                max_tokens: max_new,
                temperature: 0.0,
                ignore_eos: true,
                spec: Some(sd),
                ..Default::default()
            },
            &mut sink,
        )
        .unwrap();

    println!(
        "session spec: {} tokens (finish {:?}); driver: {} tokens",
        sink.0.len(),
        summary.finish_reason,
        driver.len()
    );
    assert_eq!(
        summary.tokens_generated as usize,
        driver.len(),
        "session must generate the same count as the driver"
    );
    assert_eq!(
        sink.0, driver,
        "Session spec-decode output must equal the standalone driver token-for-token"
    );
    // Session must land `current_pos` exactly at prompt + generated (KV holds
    // every emitted token — no rejected-draft cells left behind).
    assert_eq!(
        session.position() as usize,
        prompt.len() + driver.len(),
        "current_pos must equal prompt + generated after a spec run"
    );
}

/// Stop-token contract: a spec run that honors stops must behave exactly like a
/// plain greedy decode at a stop — the stop token is NOT streamed, NOT counted,
/// and its KV is NOT appended (so `position()` stays put). We force this on the
/// very first token by setting `stop_tokens` to the model's own first argmax:
/// the guaranteed-token stop path must then emit nothing and leave the session
/// exactly at the prompt. This covers the branch the `ignore_eos` oracles skip.
#[test]
#[ignore = "needs a real dense GGUF; set CERA_DENSE_MODEL"]
fn session_spec_honors_stop_without_emitting_it() {
    use cera::{FinishReason, GenerateOpts, SpecDecode};

    let Some(path) = dense_model_or_skip() else {
        return;
    };
    let prompt: Vec<u32> = vec![
        1, 450, 6635, 3290, 373, 278, 1775, 29889, 450, 6635, 3290, 373, 278,
    ];

    // Learn the model's first greedy token via a monolithic prefill, so we can
    // make it a stop token below. The Session prefills the same way (ubatch=0).
    let gr = cera::gguf::GgufFile::open(&path).unwrap();
    let mr = cera::model::load_model(gr, None, 8192).unwrap();
    let cfg = mr.config();
    let mut sr = cera::kv_cache::InferenceState::from_config(cfg).unwrap();
    let prefill = mr.forward_prefill(&prompt, 0, &mut sr);
    let first = argmax(&prefill) as u32;

    let mut session = build_spec_session(&path, None);
    session.append_tokens(&prompt).unwrap();
    let mut sink = Collect(Vec::new());
    let summary = session
        .generate(
            &GenerateOpts {
                max_tokens: 64,
                temperature: 0.0,
                ignore_eos: false,
                stop_tokens: vec![first],
                spec: Some(SpecDecode { ngram: 2, k: 6 }),
                ..Default::default()
            },
            &mut sink,
        )
        .unwrap();

    assert!(
        matches!(summary.finish_reason, FinishReason::Stop),
        "expected Stop, got {:?}",
        summary.finish_reason
    );
    assert_eq!(
        summary.tokens_generated, 0,
        "stop token must not be counted"
    );
    assert!(sink.0.is_empty(), "stop token must not be streamed");
    assert_eq!(
        session.position() as usize,
        prompt.len(),
        "stop token's KV must not be appended (position stays at the prompt)"
    );
}

/// Context bound: a verify round appends up to `1 + k` tokens, so the drafter is
/// clamped to keep the KV within `max_seq_len`. With a tight `max_seq_len` and a
/// repetitive prompt (drafts fire), the spec run must stop at `ContextFull`
/// without overshooting the bound — `position()` never exceeds `max_seq_len`.
#[test]
#[ignore = "needs a real dense GGUF; set CERA_DENSE_MODEL"]
fn session_spec_respects_max_seq_len() {
    use cera::{FinishReason, GenerateOpts, SpecDecode};

    let Some(path) = dense_model_or_skip() else {
        return;
    };
    let prompt: Vec<u32> = vec![
        1, 450, 6635, 3290, 373, 278, 1775, 29889, 450, 6635, 3290, 373, 278,
    ];
    // Leave only a few token-slots after the prompt — smaller than k, so a full
    // draft would overshoot if unclamped.
    let cap = prompt.len() + 3;

    let mut session = build_spec_session(&path, Some(cap as u32));
    session.append_tokens(&prompt).unwrap();
    let mut sink = Collect(Vec::new());
    let summary = session
        .generate(
            &GenerateOpts {
                max_tokens: 256, // large, so max_seq_len is the binding limit
                temperature: 0.0,
                ignore_eos: true,
                spec: Some(SpecDecode { ngram: 2, k: 6 }),
                ..Default::default()
            },
            &mut sink,
        )
        .unwrap();

    assert!(
        session.position() as usize <= cap,
        "spec decode overshot max_seq_len: position {} > cap {cap}",
        session.position()
    );
    assert!(
        matches!(summary.finish_reason, FinishReason::ContextFull),
        "expected ContextFull at the bound, got {:?}",
        summary.finish_reason
    );
}

/// Coverage guard for the batched LM-head projection, which is a *performance*
/// property no correctness assertion can see.
///
/// `forward_prefill_logits_all` projects all `n` positions in one
/// `[rows x n] = [rows x hs] * [hs x n]` GEMM, so the LM head — the largest
/// tensor in the model — is read once per verification round instead of once
/// per verified position. It falls back to a per-row loop when the head's dtype
/// has no batched kernel. Both paths return correct logits, so a fixture whose
/// head is not GEMM-able would pass every other test in this file while
/// measuring nothing: on Llama-3.2-1B-Q4_0 an `n = 7` round costs ~57 ms on the
/// fallback against ~42 ms on the GEMM.
///
/// Assert the fixture actually reaches the fast path. If this fails, the model
/// is fine — the *benchmark* is lying.
///
/// The cfg must match `project_logits_batched`'s exactly, including
/// `not(feature = "blas")`. Under `blas` the batched projection is compiled out
/// entirely, while `batched_gemm_supports` still answers `true` for the dtypes
/// whose arms short-circuit on `cfg!(feature = "blas")` — so a looser cfg here
/// would assert "the fixture reaches the fast path" on a build that has no fast
/// path, precisely the false pass this test exists to prevent.
#[test]
#[ignore = "needs a real dense GGUF; set CERA_DENSE_MODEL"]
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(has_blas)))]
fn batched_all_logits_reaches_the_lm_head_gemm() {
    let Some(path) = dense_model_or_skip() else {
        return;
    };
    let (_model, detail) = common::dense_gemm_head_fixture(&path);
    println!("reaches the batched projection: {detail}");
}
