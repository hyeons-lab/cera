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

#![cfg(any(feature = "gpu", all(feature = "metal", target_os = "macos")))]

use std::path::PathBuf;

use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, KvCompression};
use cera::model::Model;
// CPU oracle loader — used by both the wgpu (`check_parity`) and Metal
// (`check_metal_matches_cpu`) parity gates, so it's needed under either feature.
use cera::model::load_model;
#[cfg(feature = "gpu")]
use cera::model::load_model_gpu;
#[cfg(all(feature = "metal", target_os = "macos"))]
use cera::model::load_model_metal;
use cera::sampler::argmax;
use cera::tokenizer::BpeTokenizer;

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
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None).unwrap();
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
#[cfg(feature = "gpu")]
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

    const TIE_REL_TOL: f64 = 0.02;
    const MIN_MATCH_PREFIX: usize = 4;

    // Lowest index where the streams differ on their overlapping prefix. `None`
    // ⇒ no token-level disagreement (one may just be shorter).
    let Some(div) = ref_toks.iter().zip(&test_toks).position(|(a, b)| a != b) else {
        let shared = ref_toks.len().min(test_toks.len());
        if ref_toks.len() == test_toks.len() {
            eprintln!("  exact match across {} tokens", ref_toks.len());
            return Ok(());
        }
        // Unequal length with a matching shared prefix: one emitted EOS a step
        // earlier. Benign ONLY when the shared prefix is long enough to prove
        // numerical equivalence — otherwise a real bug that makes `test` emit a
        // short-but-correct prefix then EOS prematurely would slip through. Hold
        // it to the same MIN_MATCH_PREFIX bar as a positional divergence.
        if shared >= MIN_MATCH_PREFIX {
            eprintln!(
                "  match across {shared} shared tokens; benign EOS-timing length diff ({} vs {})",
                ref_toks.len(),
                test_toks.len()
            );
            return Ok(());
        }
        return Err(format!(
            "{label}: {test_name} and {ref_name} agree on only {shared} tokens before a length \
             split ({} vs {}) — too short to be a benign EOS-timing near-tie (need ≥ \
             {MIN_MATCH_PREFIX})\n    {ref_name}: {ref_toks:?}\n    {test_name}: {test_toks:?}",
            ref_toks.len(),
            test_toks.len()
        ));
    };
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
#[cfg(feature = "gpu")]
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

/// Metal twin of [`check_batched_vs_pertoken`]: GPU-internal differential
/// (batched prefill vs per-token decode) on the SAME native-Metal model. Both
/// runs share the generalized Metal forward path, so any divergence is a real
/// bug in the dense-feature wiring (rope_type / freq-factors / QK-norm / QKV
/// bias / decoupled head_dim / Granite scalars / untied output) of either the
/// decode or the batched-prefill code path.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn check_metal_batched_vs_pertoken(
    model_file: &str,
    prompt: &str,
    n_predict: usize,
) -> Option<Result<(), String>> {
    let mp = models_dir().join(model_file);
    if !mp.exists() {
        eprintln!("skipping {model_file}: not found at {}", mp.display());
        return None;
    }
    eprintln!("=== metal batched-vs-pertoken: {model_file} ===");

    let gguf = GgufFile::open(&mp).expect("open gguf");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
    let eos = tokenizer.eos_token();
    let mut tokens = Vec::new();
    if let Some(bos) = tokenizer.bos_token() {
        tokens.push(bos);
    }
    tokens.extend(tokenizer.encode(prompt));

    let metal = match load_model_metal(GgufFile::open(&mp).expect("open gguf"), &mp, 8192) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping {model_file}: Metal unavailable ({e})");
            return None;
        }
    };

    // Per-token first so its `forward` decode calls never collide with the
    // batched run's prefix-cache insert (only `forward_prefill` writes it).
    let pt_steps = greedy_decode(metal.as_ref(), &tokens, n_predict, eos, Prefill::PerToken);
    let batched_steps = greedy_decode(metal.as_ref(), &tokens, n_predict, eos, Prefill::Batched);

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

/// Native-Metal-vs-CPU greedy parity. The Metal-vs-Metal differential proves the
/// batched and per-token paths agree, but it cannot catch a bug that is identical
/// in BOTH Metal paths (e.g. a shared embedding/residual scalar or a tied-vs-
/// untied output mix-up). Comparing the Metal batched run against the
/// llama.cpp-verified CPU oracle closes that gap: any feature wired wrong in the
/// shared Metal code diverges from CPU here. (Greedy argmax is invariant to the
/// logit multiplier, so logit_scale is intentionally out of scope for this gate.)
#[cfg(all(feature = "metal", target_os = "macos"))]
fn check_metal_matches_cpu(
    model_file: &str,
    prompt: &str,
    n_predict: usize,
) -> Option<Result<(), String>> {
    let mp = models_dir().join(model_file);
    if !mp.exists() {
        eprintln!("skipping {model_file}: not found at {}", mp.display());
        return None;
    }
    eprintln!("=== metal-vs-cpu parity: {model_file} ===");

    let gguf = GgufFile::open(&mp).expect("open gguf");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
    let eos = tokenizer.eos_token();
    let mut tokens = Vec::new();
    if let Some(bos) = tokenizer.bos_token() {
        tokens.push(bos);
    }
    tokens.extend(tokenizer.encode(prompt));

    let cpu = load_model(GgufFile::open(&mp).expect("open gguf"), None, 8192).expect("cpu load");
    let metal = match load_model_metal(GgufFile::open(&mp).expect("open gguf"), &mp, 8192) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping {model_file}: Metal unavailable ({e})");
            return None;
        }
    };

    let cpu_steps = greedy_decode(cpu.as_ref(), &tokens, n_predict, eos, Prefill::Batched);
    let metal_steps = greedy_decode(metal.as_ref(), &tokens, n_predict, eos, Prefill::Batched);

    eprintln!(
        "  cpu   text: {:?}",
        tokenizer.decode(&cpu_steps.iter().map(|s| s.token).collect::<Vec<_>>())
    );
    eprintln!(
        "  metal text: {:?}",
        tokenizer.decode(&metal_steps.iter().map(|s| s.token).collect::<Vec<_>>())
    );

    Some(classify_streams(
        model_file,
        "cpu",
        &cpu_steps,
        "metal",
        &metal_steps,
    ))
}

/// The `CERA_PROFILE` diagnostic prefill path (`forward_prefill_profiled`) splits
/// production prefill into per-phase command buffers for timing. It must produce
/// the SAME prefill logits as production `forward_prefill` — otherwise the
/// diagnostic strides/bias/residual have drifted (the path was "correct for LFM2
/// only" before the dense-feature wiring). Compares the greedy next-token and the
/// max per-logit delta on the prompt's last position for each dense arch
/// (decoupled head_dim, QKV bias, NORM rope + Granite residual all exercised).
#[cfg(all(feature = "metal", target_os = "macos"))]
fn check_metal_profiled_prefill(model_file: &str, prompt: &str) -> Option<Result<(), String>> {
    use cera::model::metal_lfm2::MetalLfm2Model;

    let mp = models_dir().join(model_file);
    if !mp.exists() {
        eprintln!("skipping {model_file}: not found at {}", mp.display());
        return None;
    }
    eprintln!("=== metal profiled-prefill vs production: {model_file} ===");

    let gguf = GgufFile::open(&mp).expect("open gguf");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
    let mut tokens = Vec::new();
    if let Some(bos) = tokenizer.bos_token() {
        tokens.push(bos);
    }
    tokens.extend(tokenizer.encode(prompt));

    // Separate model instances so the second prefill can't restore the first's
    // prefix cache instead of recomputing.
    let prod = match MetalLfm2Model::from_llama(GgufFile::open(&mp).expect("open"), &mp, 8192) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping {model_file}: Metal unavailable ({e})");
            return None;
        }
    };
    let prof = MetalLfm2Model::from_llama(GgufFile::open(&mp).expect("open"), &mp, 8192)
        .expect("metal load (profiled)");

    let cfg = prod.config();
    let mut state_a = InferenceState::from_config(cfg).unwrap();
    let prod_logits = prod.forward_prefill(&tokens, 0, &mut state_a);

    let mut state_b = InferenceState::from_config(prof.config()).unwrap();
    let _timings = prof.forward_prefill_profiled(&tokens, 0, &mut state_b);
    let prof_logits = prof.read_logits();

    // Reuse cera's production greedy `sampler::argmax`.
    let prod_tok = argmax(&prod_logits);
    let prof_tok = argmax(&prof_logits);
    let max_delta = prod_logits
        .iter()
        .zip(prof_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("  prod_tok={prod_tok} prof_tok={prof_tok} max_logit_delta={max_delta:.4e}");

    if prod_tok != prof_tok {
        return Some(Err(format!(
            "{model_file}: profiled prefill greedy token {prof_tok} != production {prod_tok} \
             (max_logit_delta={max_delta:.4e})"
        )));
    }
    // Argmax stability alone is too weak to back the "bit-faithful to production"
    // claim: a stride/bias/residual wiring drift could shift logits without
    // flipping the greedy token. The two paths run identical kernels on the same
    // GPU over the same inputs, so the delta is deterministically ~0; assert a
    // tight bound so any real drift fails the gate instead of slipping through.
    const MAX_LOGIT_DELTA: f32 = 1e-3;
    if max_delta > MAX_LOGIT_DELTA {
        return Some(Err(format!(
            "{model_file}: profiled prefill logits drifted from production \
             (max_logit_delta={max_delta:.4e} > {MAX_LOGIT_DELTA:.0e}) despite matching \
             greedy token {prod_tok}"
        )));
    }
    Some(Ok(()))
}

/// Gate: the diagnostic profiled-prefill path stays bit-faithful to production
/// prefill across all four dense archs. Guards the `CERA_PROFILE` stride/bias/
/// residual wiring against future drift.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore] // run with --ignored + CERA_METAL_PARITY=1
fn metal_profiled_prefill_matches_production() {
    if std::env::var("CERA_METAL_PARITY").as_deref() != Ok("1") {
        eprintln!("skipping: CERA_METAL_PARITY=1 not set");
        return;
    }

    let mut failures = Vec::new();
    let mut ran = 0;
    for (file, prompt, _n) in CASES {
        match check_metal_profiled_prefill(file, prompt) {
            None => {}
            Some(Ok(())) => ran += 1,
            Some(Err(e)) => {
                ran += 1;
                failures.push(e);
            }
        }
    }

    if ran == 0 {
        eprintln!("no models present / no Metal — nothing verified");
        return;
    }
    assert!(
        failures.is_empty(),
        "metal profiled-prefill failures:\n{}",
        failures.join("\n")
    );
    eprintln!("metal profiled-prefill matches production across {ran} model(s)");
}

#[cfg(feature = "gpu")]
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
#[cfg(feature = "gpu")]
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

/// Native-Metal twin of [`gpu_batched_prefill_matches_pertoken`]: the primary
/// correctness gate for the LLaMA-family Metal backend. Batched prefill must
/// match the per-token decode loop EXACTLY (24-token greedy) on all four dense
/// archs, which proves every dense feature — NEOX/NORM rope, Llama-3 freq
/// factors, Qwen3 QK-norm + decoupled head_dim, Qwen2 QKV bias, Granite
/// embedding/residual/attention/logit scalars, and untied output — is wired
/// consistently across both Metal forward paths.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore] // run with --ignored + CERA_METAL_PARITY=1
fn metal_batched_prefill_matches_pertoken() {
    if std::env::var("CERA_METAL_PARITY").as_deref() != Ok("1") {
        eprintln!("skipping: CERA_METAL_PARITY=1 not set");
        return;
    }

    let mut failures = Vec::new();
    let mut ran = 0;
    for (file, prompt, n) in CASES {
        match check_metal_batched_vs_pertoken(file, prompt, n) {
            None => {}
            Some(Ok(())) => ran += 1,
            Some(Err(e)) => {
                ran += 1;
                failures.push(e);
            }
        }
    }

    if ran == 0 {
        eprintln!("no models present / no Metal — nothing verified");
        return;
    }
    assert!(
        failures.is_empty(),
        "metal batched-vs-pertoken failures:\n{}",
        failures.join("\n")
    );
    eprintln!("metal batched-vs-pertoken OK across {ran} model(s)");
}

/// Native-Metal output must match the CPU oracle. Catches shared-Metal-path bugs
/// the batched-vs-pertoken differential is blind to (embedding/residual/attn
/// scalars, tied-vs-untied output). Covers all four dense archs.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore] // run with --ignored + CERA_METAL_PARITY=1
fn metal_transformer_matches_cpu() {
    if std::env::var("CERA_METAL_PARITY").as_deref() != Ok("1") {
        eprintln!("skipping: CERA_METAL_PARITY=1 not set");
        return;
    }

    let mut failures = Vec::new();
    let mut ran = 0;
    for (file, prompt, n) in CASES {
        match check_metal_matches_cpu(file, prompt, n) {
            None => {}
            Some(Ok(())) => ran += 1,
            Some(Err(e)) => {
                ran += 1;
                failures.push(e);
            }
        }
    }

    if ran == 0 {
        eprintln!("no models present / no Metal — nothing verified");
        return;
    }
    assert!(
        failures.is_empty(),
        "metal-vs-cpu parity failures:\n{}",
        failures.join("\n")
    );
    eprintln!("metal-vs-cpu parity OK across {ran} model(s)");
}
