//! GPU-vs-CPU parity for the dense transformer archs on the wgpu backend.
//!
//! cera's CPU `LlamaModel` is llama.cpp-verified (PR #172), and llama.cpp's
//! oracle is CPU-only by design — so the GPU path is verified against the CPU
//! model rather than against a separate GPU oracle. GPU and CPU float
//! accumulation differ, so we compare **greedy argmax tokens** over a fixed
//! continuation, not raw logits: a wrong rope layout / missing bias / dropped
//! scalar perturbs every step and diverges the stream at token 0–1. The streams
//! match exactly until — occasionally — Q8 GPU/CPU accumulation noise tips a
//! near-tie; the gate confirms any such divergence is genuinely a near-tie (the
//! two picks sit within a tiny fraction of the logit scale on BOTH backends)
//! and fails only on a real, non-tie divergence. Mirrors how the oracle treats
//! greedy as a coarse but meaningful gate.
//!
//! Covers all five transformer behaviors:
//!   • Qwen2  — NEOX rope + QKV bias
//!   • Qwen3  — NEOX rope + QK-norm + decoupled head_dim
//!   • Llama  — NORM rope + Llama-3 freq factors
//!   • Granite — NORM rope + embedding/residual/attention/logit scalars
//!
//! Models are the same Q8_0 GGUFs the oracle uses (see scripts/oracle/README).
//! A model whose GGUF is absent is skipped; absence of a GPU is also a skip.
//!
//! Gated behind `CERA_GPU_PARITY=1` and `#[ignore]`. Run:
//!   CERA_GPU_PARITY=1 cargo test -p cera --features gpu --release \
//!     --test gpu_transformer_parity -- --ignored --nocapture

#![cfg(feature = "gpu")]

use std::path::PathBuf;

use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, KvCompression};
use cera::model::{Model, load_model, load_model_gpu};
use cera::tokenizer::BpeTokenizer;

/// Greedy next-token pick — matches `sampler::cpu_argmax` (NaN → -inf,
/// `total_cmp`, last-index-on-ties).
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let a = if a.is_nan() { f32::NEG_INFINITY } else { **a };
            let b = if b.is_nan() { f32::NEG_INFINITY } else { **b };
            a.total_cmp(&b)
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn models_dir() -> PathBuf {
    if let Ok(d) = std::env::var("CERA_ORACLE_MODELS_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/oracle/models")
}

/// One greedy step's record: the chosen token plus the full logit vector it was
/// chosen from. Keeping the logits lets the gate distinguish a real bug (GPU
/// picked a token CPU ranked far below its own pick) from a benign near-tie
/// (the two picks are within Q8 GPU/CPU accumulation noise of each other).
struct Step {
    token: u32,
    logits: Vec<f32>,
}

/// Greedy-decode up to `n_predict` tokens (EOS-terminated), recording each
/// step's token + logits.
fn greedy_decode(
    model: &dyn Model,
    prompt_tokens: &[u32],
    n_predict: usize,
    eos: Option<u32>,
) -> Vec<Step> {
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None);
    let mut logits = model.forward_prefill(prompt_tokens, 0, &mut state);
    let mut out = Vec::with_capacity(n_predict);
    for _ in 0..n_predict {
        let next = argmax(&logits);
        if eos == Some(next) {
            break;
        }
        out.push(Step {
            token: next,
            logits: logits.clone(),
        });
        logits = model.forward(&[next], state.seq_len, &mut state);
    }
    out
}

/// Run CPU-vs-GPU greedy parity for one model file. Returns `None` if the GGUF
/// is absent (skip), else a failure message on mismatch (`Some(Err)`) or
/// `Some(Ok(()))` on agreement.
fn check_parity(model_file: &str, prompt: &str, n_predict: usize) -> Option<Result<(), String>> {
    let mp = models_dir().join(model_file);
    if !mp.exists() {
        eprintln!("skipping {model_file}: not found at {}", mp.display());
        return None;
    }
    eprintln!("=== gpu parity: {model_file} ===");

    let gguf = GgufFile::open(&mp).expect("open gguf");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
    let eos = tokenizer.eos_token();
    // Mirror the oracle's BOS handling: prepend BOS when the vocab uses one.
    let mut tokens = Vec::new();
    if let Some(bos) = tokenizer.bos_token() {
        tokens.push(bos);
    }
    tokens.extend(tokenizer.encode(prompt));

    let cpu = load_model(GgufFile::open(&mp).expect("open gguf"), None, 8192).expect("cpu load");
    let gpu = match load_model_gpu(GgufFile::open(&mp).expect("open gguf"), None, 8192) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping {model_file}: GPU unavailable ({e})");
            return None;
        }
    };

    let cpu_steps = greedy_decode(cpu.as_ref(), &tokens, n_predict, eos);
    let gpu_steps = greedy_decode(gpu.as_ref(), &tokens, n_predict, eos);

    let cpu_toks: Vec<u32> = cpu_steps.iter().map(|s| s.token).collect();
    let gpu_toks: Vec<u32> = gpu_steps.iter().map(|s| s.token).collect();
    eprintln!("  cpu text: {:?}", tokenizer.decode(&cpu_toks));
    eprintln!("  gpu text: {:?}", tokenizer.decode(&gpu_toks));

    // Lowest index where the streams differ on their overlapping prefix. `None`
    // ⇒ no token-level disagreement (one may just be shorter).
    let Some(div) = cpu_toks.iter().zip(&gpu_toks).position(|(a, b)| a != b) else {
        // Identical on the overlap. Equal length = exact match. Unequal length
        // means one backend emitted EOS a step earlier — a near-tie at the EOS
        // boundary; the shorter stream didn't record that step's logits, so
        // there's nothing to classify, and the long shared prefix already proves
        // numerical equivalence. Benign either way (and never indexes *_steps,
        // which would be out of bounds at min(len)).
        if cpu_toks.len() == gpu_toks.len() {
            eprintln!("  exact match across {} tokens", cpu_toks.len());
        } else {
            eprintln!(
                "  match across {} shared tokens; benign EOS-timing length diff ({} vs {})",
                cpu_toks.len().min(gpu_toks.len()),
                cpu_toks.len(),
                gpu_toks.len()
            );
        }
        return Some(Ok(()));
    };

    // `div` is strictly less than the shorter stream's length, so indexing
    // `*_steps[div]` below is in bounds. Classify the divergence: a benign
    // near-tie (Q8 GPU/CPU accumulation noise tipping two nearly-equal logits)
    // has a tiny CPU-pick-vs-GPU-pick logit gap on BOTH backends; a real bug
    // makes one backend pick a token the other ranked far below — a large gap.
    //
    // Also require a meaningful exact-match prefix: a wrong rope/bias/scalar
    // perturbs every step, so a genuine bug diverges at the very start. An early
    // flip is failed even if its gap looks small — that early position is itself
    // the bug signature, so a coincidentally-tied first token must not pass.
    const TIE_REL_TOL: f64 = 0.02;
    const MIN_MATCH_PREFIX: usize = 4;
    let gap_rel = |logits: &[f32], a: u32, b: u32| -> f64 {
        let la = logits[a as usize] as f64;
        let lb = logits[b as usize] as f64;
        let scale = la.abs().max(lb.abs()).max(1.0);
        (la - lb).abs() / scale
    };
    let cpu_pick = cpu_toks[div];
    let gpu_pick = gpu_toks[div];
    let gap_cpu = gap_rel(&cpu_steps[div].logits, cpu_pick, gpu_pick);
    let gap_gpu = gap_rel(&gpu_steps[div].logits, cpu_pick, gpu_pick);
    eprintln!(
        "  diverge@{div}: cpu_pick={cpu_pick} gpu_pick={gpu_pick} \
         gap_cpu={gap_cpu:.4} gap_gpu={gap_gpu:.4} (matched {div} tokens)"
    );

    if div >= MIN_MATCH_PREFIX && gap_cpu < TIE_REL_TOL && gap_gpu < TIE_REL_TOL {
        eprintln!("  near-tie flip at {div} (within Q8 noise) — not a bug");
        Some(Ok(()))
    } else {
        Some(Err(format!(
            "{model_file}: GPU diverges at token {div} and it is NOT a benign near-tie \
             (need div ≥ {MIN_MATCH_PREFIX} and gap < {TIE_REL_TOL}; got \
             gap_cpu={gap_cpu:.4}, gap_gpu={gap_gpu:.4})\n    \
             cpu: {cpu_toks:?}\n    gpu: {gpu_toks:?}"
        )))
    }
}

#[test]
#[ignore] // run with --ignored + CERA_GPU_PARITY=1
fn gpu_transformer_matches_cpu() {
    if std::env::var("CERA_GPU_PARITY").as_deref() != Ok("1") {
        eprintln!("skipping: CERA_GPU_PARITY=1 not set");
        return;
    }

    // (file, prompt, n_predict). Short prompts keep greedy off near-ties.
    let cases = [
        (
            "qwen2-0_5b-instruct-q8_0.gguf",
            "The capital of France is",
            24,
        ),
        ("Qwen3-0.6B-Q8_0.gguf", "The capital of France is", 24),
        (
            "Llama-3.2-1B-Instruct-Q8_0.gguf",
            "The capital of France is",
            24,
        ),
        (
            "granite-3.1-2b-instruct-Q8_0.gguf",
            "The capital of France is",
            24,
        ),
    ];

    let mut failures = Vec::new();
    let mut ran = 0;
    for (file, prompt, n) in cases {
        match check_parity(file, prompt, n) {
            None => {}
            Some(Ok(())) => ran += 1,
            Some(Err(e)) => {
                ran += 1;
                failures.push(e);
            }
        }
    }

    if ran == 0 {
        eprintln!("no models present / no GPU — nothing verified");
        return;
    }
    assert!(
        failures.is_empty(),
        "GPU parity failures:\n{}",
        failures.join("\n")
    );
    eprintln!("GPU parity OK across {ran} model(s)");
}
