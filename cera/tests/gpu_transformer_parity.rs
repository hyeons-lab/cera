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

/// How the prompt is consumed before greedy continuation.
#[derive(Clone, Copy, PartialEq)]
enum Prefill {
    /// `forward_prefill` — the batched path on GPU once the gate is flipped.
    Batched,
    /// Drive `forward` one token at a time — the per-token decode loop, which
    /// is the trusted oracle the batched path must match.
    PerToken,
}

/// Greedy-decode up to `n_predict` tokens (EOS-terminated), recording each
/// step's token + logits. `prefill` selects how the prompt is consumed.
fn greedy_decode(
    model: &dyn Model,
    prompt_tokens: &[u32],
    n_predict: usize,
    eos: Option<u32>,
    prefill: Prefill,
) -> Vec<Step> {
    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None);
    let mut logits = match prefill {
        Prefill::Batched => model.forward_prefill(prompt_tokens, 0, &mut state),
        Prefill::PerToken => {
            // Replicate the per-token prefill fallback: forward each prompt
            // token in sequence; the last call's logits are the prefill output.
            let mut l = Vec::new();
            for (i, &t) in prompt_tokens.iter().enumerate() {
                l = model.forward(&[t], i, &mut state);
            }
            l
        }
    };
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

    let cpu_steps = greedy_decode(cpu.as_ref(), &tokens, n_predict, eos, Prefill::Batched);
    let gpu_steps = greedy_decode(gpu.as_ref(), &tokens, n_predict, eos, Prefill::Batched);

    eprintln!(
        "  cpu text: {:?}",
        tokenizer.decode(&cpu_steps.iter().map(|s| s.token).collect::<Vec<_>>())
    );
    eprintln!(
        "  gpu text: {:?}",
        tokenizer.decode(&gpu_steps.iter().map(|s| s.token).collect::<Vec<_>>())
    );

    Some(classify_streams(
        model_file, "cpu", &cpu_steps, "gpu", &gpu_steps,
    ))
}

/// Classify two greedy streams as equivalent or genuinely divergent.
///
/// `ref`/`test` are interchangeable for the verdict — the gate is symmetric. A
/// benign near-tie (Q8 accumulation noise tipping two nearly-equal logits) has
/// a tiny ref-pick-vs-test-pick logit gap on BOTH streams; a real bug makes one
/// stream pick a token the other ranked far below (a large gap). A meaningful
/// exact-match prefix is also required: a wrong rope/bias/scalar perturbs every
/// step, so a genuine bug diverges at the very start — an early flip is failed
/// even if its gap looks small, because that early position is itself the bug
/// signature.
fn classify_streams(
    label: &str,
    ref_name: &str,
    ref_steps: &[Step],
    test_name: &str,
    test_steps: &[Step],
) -> Result<(), String> {
    let ref_toks: Vec<u32> = ref_steps.iter().map(|s| s.token).collect();
    let test_toks: Vec<u32> = test_steps.iter().map(|s| s.token).collect();

    // Lowest index where the streams differ on their overlapping prefix. `None`
    // ⇒ no token-level disagreement (one may just be shorter).
    let Some(div) = ref_toks.iter().zip(&test_toks).position(|(a, b)| a != b) else {
        // Identical on the overlap. Unequal length means one emitted EOS a step
        // earlier — a near-tie at the EOS boundary; the shared prefix already
        // proves numerical equivalence. Benign either way (and never indexes
        // *_steps out of bounds at min(len)).
        if ref_toks.len() == test_toks.len() {
            eprintln!("  exact match across {} tokens", ref_toks.len());
        } else {
            eprintln!(
                "  match across {} shared tokens; benign EOS-timing length diff ({} vs {})",
                ref_toks.len().min(test_toks.len()),
                ref_toks.len(),
                test_toks.len()
            );
        }
        return Ok(());
    };

    const TIE_REL_TOL: f64 = 0.02;
    const MIN_MATCH_PREFIX: usize = 4;
    let gap_rel = |logits: &[f32], a: u32, b: u32| -> f64 {
        let la = logits[a as usize] as f64;
        let lb = logits[b as usize] as f64;
        let scale = la.abs().max(lb.abs()).max(1.0);
        (la - lb).abs() / scale
    };
    let ref_pick = ref_toks[div];
    let test_pick = test_toks[div];
    let gap_ref = gap_rel(&ref_steps[div].logits, ref_pick, test_pick);
    let gap_test = gap_rel(&test_steps[div].logits, ref_pick, test_pick);
    eprintln!(
        "  diverge@{div}: {ref_name}_pick={ref_pick} {test_name}_pick={test_pick} \
         gap_{ref_name}={gap_ref:.4} gap_{test_name}={gap_test:.4} (matched {div} tokens)"
    );

    if div >= MIN_MATCH_PREFIX && gap_ref < TIE_REL_TOL && gap_test < TIE_REL_TOL {
        eprintln!("  near-tie flip at {div} (within Q8 noise) — not a bug");
        Ok(())
    } else {
        Err(format!(
            "{label}: {test_name} diverges from {ref_name} at token {div} and it is NOT a \
             benign near-tie (need div ≥ {MIN_MATCH_PREFIX} and gap < {TIE_REL_TOL}; got \
             gap_{ref_name}={gap_ref:.4}, gap_{test_name}={gap_test:.4})\n    \
             {ref_name}: {ref_toks:?}\n    {test_name}: {test_toks:?}"
        ))
    }
}

/// GPU-internal differential: batched prefill vs the per-token decode loop on
/// the SAME GPU model. Both run identical GEMM kernels, so they isolate the
/// batched attention/rope/bias/scalar path from CPU↔GPU float noise — a wrong
/// batched rope layout, dropped freq-factor, missing QKV bias, mis-sized
/// decoupled head_dim, or unapplied Granite scalar diverges the streams.
fn check_batched_vs_pertoken(
    model_file: &str,
    prompt: &str,
    n_predict: usize,
) -> Option<Result<(), String>> {
    let mp = models_dir().join(model_file);
    if !mp.exists() {
        eprintln!("skipping {model_file}: not found at {}", mp.display());
        return None;
    }
    eprintln!("=== batched-vs-pertoken: {model_file} ===");

    let gguf = GgufFile::open(&mp).expect("open gguf");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
    let eos = tokenizer.eos_token();
    let mut tokens = Vec::new();
    if let Some(bos) = tokenizer.bos_token() {
        tokens.push(bos);
    }
    tokens.extend(tokenizer.encode(prompt));

    let gpu = match load_model_gpu(GgufFile::open(&mp).expect("open gguf"), None, 8192) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping {model_file}: GPU unavailable ({e})");
            return None;
        }
    };

    // Per-token first so its `forward` decode calls never collide with the
    // batched run's prefix-cache insert (only `forward_prefill` writes it).
    let pt_steps = greedy_decode(gpu.as_ref(), &tokens, n_predict, eos, Prefill::PerToken);
    let batched_steps = greedy_decode(gpu.as_ref(), &tokens, n_predict, eos, Prefill::Batched);

    eprintln!(
        "  pertoken text: {:?}",
        tokenizer.decode(&pt_steps.iter().map(|s| s.token).collect::<Vec<_>>())
    );
    eprintln!(
        "  batched  text: {:?}",
        tokenizer.decode(&batched_steps.iter().map(|s| s.token).collect::<Vec<_>>())
    );

    Some(classify_streams(
        model_file,
        "pertoken",
        &pt_steps,
        "batched",
        &batched_steps,
    ))
}

#[test]
#[ignore] // run with --ignored + CERA_GPU_PARITY=1
fn gpu_transformer_matches_cpu() {
    if std::env::var("CERA_GPU_PARITY").as_deref() != Ok("1") {
        eprintln!("skipping: CERA_GPU_PARITY=1 not set");
        return;
    }

    let mut failures = Vec::new();
    let mut ran = 0;
    for (file, prompt, n) in CASES {
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

/// (file, prompt, n_predict). One model per dense-transformer behavior:
///   • qwen2  — NEOX rope + QKV bias
///   • qwen3  — NEOX rope + QK-norm + decoupled head_dim
///   • llama  — NORM rope + Llama-3 freq factors
///   • granite — NORM rope + embedding/residual/attention/logit scalars
const CASES: [(&str, &str, usize); 4] = [
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

/// GPU batched prefill must match the GPU per-token decode loop for every dense
/// transformer behavior. This is the primary correctness gate for the batched
/// path: it compares two GPU runs (no CPU float noise), so any divergence is a
/// real bug in the batched attention/rope/bias/scalar generalization.
#[test]
#[ignore] // run with --ignored + CERA_GPU_PARITY=1
fn gpu_batched_prefill_matches_pertoken() {
    if std::env::var("CERA_GPU_PARITY").as_deref() != Ok("1") {
        eprintln!("skipping: CERA_GPU_PARITY=1 not set");
        return;
    }

    let mut failures = Vec::new();
    let mut ran = 0;
    for (file, prompt, n) in CASES {
        match check_batched_vs_pertoken(file, prompt, n) {
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
        "batched-vs-pertoken failures:\n{}",
        failures.join("\n")
    );
    eprintln!("batched-vs-pertoken OK across {ran} model(s)");
}
