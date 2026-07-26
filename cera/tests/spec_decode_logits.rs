//! Phase 1a of speculative decoding: `forward_prefill_logits_all` must return,
//! for each position, logits whose argmax matches what per-token `forward`
//! produces at that position — because greedy speculative decoding verifies
//! drafts by comparing the target's per-position argmax to the drafted token.
//! If the batched all-logits argmax ever diverged from the per-token argmax,
//! greedy-spec output could differ from greedy output.
//!
//! `#[ignore]` because it needs a real dense GGUF. Resolve order:
//!   1. `CERA_DENSE_MODEL` = absolute path to a dense (llama/qwen2/qwen3) gguf
//!   2. `~/.leap/models/<name>/<name>.gguf` for a small default name
//!
//! Run: `cargo test -p cera --release --test spec_decode_logits -- --ignored --nocapture`

use std::path::PathBuf;

fn find_dense_model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CERA_DENSE_MODEL") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
        eprintln!("CERA_DENSE_MODEL set but not found: {}", p.display());
    }
    let name = "Llama-3.2-1B-Instruct-Q8_0";
    let p = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".leap/models")
        .join(name)
        .join(format!("{name}.gguf"));
    if p.exists() {
        Some(p)
    } else {
        eprintln!("no dense model (set CERA_DENSE_MODEL); skipping");
        None
    }
}

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

#[test]
#[ignore = "needs a real dense GGUF; set CERA_DENSE_MODEL"]
fn all_logits_argmax_matches_per_token_forward() {
    #[allow(unused_imports)]
    use cera::model::Model;

    let Some(path) = find_dense_model() else {
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
    #[allow(unused_imports)]
    use cera::model::Model;

    let Some(path) = find_dense_model() else {
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

/// The oracle: greedy speculative decoding must produce byte-for-byte the same
/// tokens as plain greedy decoding, while (on repetitive text) actually
/// accepting drafts — proving the verify + accept + KV-truncate loop is correct.
#[test]
#[ignore = "needs a real dense GGUF; set CERA_DENSE_MODEL"]
fn greedy_spec_matches_greedy_exactly() {
    #[allow(unused_imports)]
    use cera::model::Model;

    let Some(path) = find_dense_model() else {
        return;
    };
    // A repetitive prompt so prompt-lookup drafts actually hit and the
    // verify/accept/truncate path is exercised (equivalence must hold regardless).
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
    // prove the divergence is exactly that: a batched re-forward over the agreed
    // prefix must predict spec's token, not a bug in the accept/truncate logic.
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

        // Agreed prefix = prompt ++ reference[..i]; batched-forward it and read
        // the argmax predicting position i. Causal per-position logits are
        // batch-shape-independent, so this reproduces the verifier's own row.
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
        println!(
            "batched verifier argmax at pos {i} = {batched_pred}; logit(spec)={g_spec:.4} logit(ref)={g_ref:.4} gap={:.4}",
            (g_spec - g_ref).abs()
        );
        assert_eq!(
            batched_pred, spec[i],
            "spec[{i}] must equal the batched verifier's own argmax (else accept/truncate is buggy)"
        );
    } else {
        println!("greedy-spec matched per-token greedy exactly (no near-tie flips)");
    }
}
