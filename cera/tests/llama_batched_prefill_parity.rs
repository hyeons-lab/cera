//! Differential parity test for the dense-transformer (LLaMA-family) CPU
//! batched-GEMM prefill against the sequential per-token path, on real GGUFs.
//!
//! For each model that is present locally, two fresh `InferenceState`s compute:
//!   (a) sequential per-token logits: `forward` per token, last token's logits;
//!   (b) batched `forward_prefill` over the whole token slice.
//! The last-token logits must match: max-abs element diff < 1e-2 AND identical
//! argmax. This exercises every dense arch feature — Llama-3 NORM RoPE with
//! freq_factors, Qwen3 per-head QK-norm + decoupled head_dim (NEOX), Qwen2 QKV
//! bias (NEOX), and Granite's four scalar multipliers.
//!
//! Models are resolved from a small set of candidate roots and SKIPPED when
//! absent (mirrors the other real-model tests) so `cargo test` never fails on a
//! machine without the fixtures. Run with `--nocapture` to see the per-model
//! max-abs diffs.

use std::path::PathBuf;

/// Try a few candidate roots so the test works both from the crate dir and
/// from a git worktree whose fixtures live in the main checkout.
fn find_model(rel: &str) -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    // Crate dir → workspace root (../ from CARGO_MANIFEST_DIR).
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(&manifest);
        if let Some(parent) = p.parent() {
            roots.push(parent.to_path_buf());
        }
    }
    // Current working directory.
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    // Explicit override for fixtures that live outside the (work)tree — e.g. a
    // git worktree whose large model files sit in the main checkout. Point
    // `CERA_MODEL_ROOT` at the dir that contains `target/oracle/models/…`.
    if let Ok(root) = std::env::var("CERA_MODEL_ROOT") {
        roots.push(PathBuf::from(root));
    }

    for root in roots {
        let candidate = root.join(rel);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

/// A fixed ~24-token prompt of small, in-vocab token ids. Every target model's
/// vocab is >= 32000, so these ids are valid across all four archs.
const PROMPT: &[u32] = &[
    1, 415, 2323, 302, 4843, 349, 264, 2818, 297, 272, 2607, 28725, 304, 378, 349, 2278, 298, 776,
    684, 456, 2758, 302, 12707, 28723,
];

fn run_parity(rel: &str) -> Option<(f32, usize, usize)> {
    // Bring `Model` into scope so the boxed trait object's methods resolve.
    #[allow(unused_imports)]
    use cera::model::Model;

    let path = find_model(rel)?;
    eprintln!("[parity] loading {}", path.display());

    // (a) Sequential per-token path.
    let gguf_seq = cera::gguf::GgufFile::open(&path).unwrap();
    let model_seq = cera::model::load_model(gguf_seq, None, 8192).unwrap();
    let mut state_seq = cera::kv_cache::InferenceState::from_config(model_seq.config());
    let mut logits_seq = Vec::new();
    for (i, &tok) in PROMPT.iter().enumerate() {
        logits_seq = model_seq.forward(&[tok], i, &mut state_seq);
    }

    // (b) Batched prefill path.
    let gguf_pre = cera::gguf::GgufFile::open(&path).unwrap();
    let model_pre = cera::model::load_model(gguf_pre, None, 8192).unwrap();
    let mut state_pre = cera::kv_cache::InferenceState::from_config(model_pre.config());
    let logits_pre = model_pre.forward_prefill(PROMPT, 0, &mut state_pre);

    assert_eq!(logits_pre.len(), logits_seq.len(), "logit length mismatch");
    Some((
        max_abs_diff(&logits_pre, &logits_seq),
        argmax(&logits_pre),
        argmax(&logits_seq),
    ))
}

fn check(rel: &str) {
    let Some((max_diff, top_pre, top_seq)) = run_parity(rel) else {
        eprintln!("[parity] SKIP (absent): {rel}");
        return;
    };
    eprintln!("[parity] {rel}: max_abs_diff={max_diff:.6e} argmax pre={top_pre} seq={top_seq}");
    assert_eq!(
        top_pre, top_seq,
        "{rel}: batched-prefill argmax {top_pre} != sequential argmax {top_seq}"
    );
    assert!(
        max_diff < 1e-2,
        "{rel}: batched-prefill vs sequential max-abs logit diff {max_diff} >= 1e-2"
    );
}

#[test]
fn llama_batched_prefill_parity_llama3() {
    // Llama-3.2-1B: arch "llama", NORM RoPE with Llama-3 `rope_freqs` factors.
    // The Q8_0 build is used (fully supported); the repo's `Llama-3.2-1B-Q4_0`
    // GGUF carries Q4_1 ffn_down layers in blocks 0/1, a dtype cera can't
    // dequantize, so neither the batched nor the per-token path can run it.
    check("target/oracle/models/Llama-3.2-1B-Instruct-Q8_0.gguf");
}

#[test]
fn llama_batched_prefill_parity_qwen3() {
    check("target/oracle/models/Qwen3-0.6B-Q8_0.gguf");
}

#[test]
fn llama_batched_prefill_parity_qwen2() {
    check("target/oracle/models/qwen2-0_5b-instruct-q8_0.gguf");
}

#[test]
fn llama_batched_prefill_parity_granite() {
    check("target/oracle/models/granite-3.1-2b-instruct-Q8_0.gguf");
}
