//! GPU-vs-CPU parity for per-token hidden-states extraction (`Model::hidden_states`).
//!
//! cera's CPU `Lfm2Model::hidden_states` mirrors the llama.cpp `--pooling none`
//! semantics (post-final-RMSNorm last-layer state); the GPU backends (Metal +
//! wgpu) are verified against that CPU path rather than a separate GPU oracle.
//! GPU and CPU float accumulation differ, so we compare per-token **cosine
//! similarity** (+ shape + finiteness), not raw equality — the same methodology
//! the text oracle uses. A wrong norm / dropped layer / stale KV would collapse
//! cosine well below the 0.99 bar on the very first token.
//!
//! Gated behind `CERA_GPU_PARITY=1` and `#[ignore]`. Needs a plain (non-VL) LFM2
//! GGUF passed via `CERA_LFM2_MODEL=/path/to/lfm2.gguf` (any local
//! `models/LFM-450M-Q4_0.gguf` is used only if it's a valid GGUF). Run:
//!   CERA_GPU_PARITY=1 CERA_LFM2_MODEL=... cargo test -p cera --features metal \
//!     --release --test hidden_states_parity -- --ignored --nocapture
//! (swap `--features metal` for `--features gpu` to exercise the wgpu case; wgpu
//! runs on Apple Silicon via its Metal backend.)

#![cfg(any(all(feature = "metal", target_os = "macos"), feature = "gpu"))]

use std::path::{Path, PathBuf};

use cera::gguf::GgufFile;
use cera::kv_cache::InferenceState;
use cera::model::{Model, load_model};
use cera::tokenizer::BpeTokenizer;

/// Locate a plain (non-VL) LFM2 GGUF: the `CERA_LFM2_MODEL` override, else a
/// local `models/LFM-450M-Q4_0.gguf` if present. Returns `None` (→ skip) if none
/// is a *valid* GGUF — validated by actually opening it, so a leftover
/// failed-download stub (a common 29-byte "Invalid username or password." file)
/// is rejected rather than accepted and panicked on later.
fn lfm2_model_path() -> Option<PathBuf> {
    let env_path = std::env::var("CERA_LFM2_MODEL").ok().map(PathBuf::from);
    let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/LFM-450M-Q4_0.gguf");
    env_path
        .into_iter()
        .chain(std::iter::once(local))
        .find(|p| p.exists() && GgufFile::open(p).is_ok())
}

/// The gate common to both backends: `CERA_GPU_PARITY=1` + a valid LFM2 model.
fn gated_model() -> Option<PathBuf> {
    if std::env::var("CERA_GPU_PARITY").as_deref() != Ok("1") {
        eprintln!("skip: set CERA_GPU_PARITY=1 to run");
        return None;
    }
    match lfm2_model_path() {
        Some(p) => Some(p),
        None => {
            eprintln!("skip: no LFM2 model (set CERA_LFM2_MODEL)");
            None
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = na * nb;
    // A zero-norm hidden state is itself a bug; return 0.0 (fails the >0.99 gate
    // cleanly) rather than a NaN that muddies the failure message.
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Load the CPU LFM2 model, run both it and `gpu` over the same prompt, and
/// assert per-token cosine > 0.99 + a reproducibility check on the GPU side.
fn check_gpu_matches_cpu(gpu: &dyn Model, path: &Path, label: &str) {
    let tok =
        BpeTokenizer::from_gguf(&GgufFile::open(path).expect("open gguf")).expect("tokenizer");
    let tokens = tok.encode("The quick brown fox jumps over the lazy dog near the riverbank.");
    assert!(tokens.len() > 3, "need a multi-token prompt");

    let cpu = load_model(GgufFile::open(path).expect("open gguf"), None, 8192).expect("cpu load");
    assert!(
        cpu.supports_hidden_states(),
        "CPU LFM2 must support hidden_states"
    );
    let mut cpu_state = InferenceState::for_prefill(cpu.config(), tokens.len()).unwrap();
    let cpu_hidden = cpu.hidden_states(&tokens, &mut cpu_state);

    assert!(
        gpu.supports_hidden_states(),
        "{label} LFM2 must support hidden_states"
    );
    let mut gpu_state = InferenceState::for_prefill(gpu.config(), tokens.len()).unwrap();
    let gpu_hidden = gpu.hidden_states(&tokens, &mut gpu_state);

    let d = cpu.config().hidden_size;
    assert_eq!(cpu_hidden.len(), tokens.len() * d, "CPU shape");
    assert_eq!(gpu_hidden.len(), tokens.len() * d, "{label} shape");

    let mut min_cos = f32::INFINITY;
    for t in 0..tokens.len() {
        let c = &cpu_hidden[t * d..(t + 1) * d];
        let g = &gpu_hidden[t * d..(t + 1) * d];
        assert!(
            g.iter().all(|x| x.is_finite()),
            "token {t}: non-finite {label} hidden"
        );
        let cos = cosine(c, g);
        min_cos = min_cos.min(cos);
        assert!(
            cos > 0.99,
            "token {t}: {label}-vs-CPU cosine {cos:.5} < 0.99"
        );
    }
    eprintln!(
        "{label}_hidden_states_matches_cpu: {} tokens, D={d}, min per-token cosine {min_cos:.5}",
        tokens.len()
    );

    // A second call must reproduce the first — proves the scratch KV/conv is
    // cleared between runs and no generation KV leaks in. A stale (uncleared)
    // scratch would perturb the conv rolling state and diverge sharply; use a
    // tight tolerance rather than bit-exact equality so ULP-level GPU
    // accumulation noise across two command buffers can't flake the gate.
    let mut s2 = InferenceState::for_prefill(gpu.config(), tokens.len()).unwrap();
    let again = gpu.hidden_states(&tokens, &mut s2);
    assert_eq!(again.len(), gpu_hidden.len(), "reproducibility: shape");
    let max_abs = again
        .iter()
        .zip(&gpu_hidden)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-3,
        "{label} hidden_states not reproducible across calls (max abs diff {max_abs:.2e})"
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "needs an LFM2 GGUF + a Metal GPU; gated on CERA_GPU_PARITY"]
fn metal_hidden_states_matches_cpu() {
    let Some(path) = gated_model() else { return };
    let metal =
        match cera::model::load_model_metal(GgufFile::open(&path).expect("open gguf"), &path, 8192)
        {
            Ok(m) => m,
            Err(e) => {
                eprintln!("skip: no Metal GPU: {e}");
                return;
            }
        };
    check_gpu_matches_cpu(metal.as_ref(), &path, "metal");
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs an LFM2 GGUF + a GPU; gated on CERA_GPU_PARITY"]
fn wgpu_hidden_states_matches_cpu() {
    let Some(path) = gated_model() else { return };
    let gpu = match cera::model::load_model_gpu(
        GgufFile::open(&path).expect("open gguf"),
        Some(&path),
        8192,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skip: no wgpu GPU: {e}");
            return;
        }
    };
    check_gpu_matches_cpu(gpu.as_ref(), &path, "wgpu");
}
