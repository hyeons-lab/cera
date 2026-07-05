//! GPU-vs-CPU parity for per-token hidden-states extraction (`Model::hidden_states`).
//!
//! cera's CPU `Lfm2Model::hidden_states` mirrors the llama.cpp `--pooling none`
//! semantics (post-final-RMSNorm last-layer state); the GPU backends are verified
//! against that CPU path rather than a separate GPU oracle. GPU and CPU float
//! accumulation differ, so we compare per-token **cosine similarity** (+ that the
//! shape and finiteness hold), not raw equality — the same methodology the text
//! oracle uses for its cosine gate. A wrong norm / dropped layer / stale KV would
//! collapse cosine well below the 0.99 bar on the very first token.
//!
//! Gated behind `CERA_GPU_PARITY=1` and `#[ignore]`. Needs a plain (non-VL) LFM2
//! GGUF passed via `CERA_LFM2_MODEL=/path/to/lfm2.gguf` (any local
//! `models/LFM-450M-Q4_0.gguf` is used only if it's a valid GGUF). Run:
//!   CERA_GPU_PARITY=1 CERA_LFM2_MODEL=... cargo test -p cera --features metal \
//!     --release --test hidden_states_parity -- --ignored --nocapture

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::PathBuf;

use cera::gguf::GgufFile;
use cera::kv_cache::InferenceState;
use cera::model::{load_model, load_model_metal};
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

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[test]
#[ignore = "needs an LFM2 GGUF + a Metal GPU; gated on CERA_GPU_PARITY"]
fn metal_hidden_states_matches_cpu() {
    if std::env::var("CERA_GPU_PARITY").as_deref() != Ok("1") {
        eprintln!("skip: set CERA_GPU_PARITY=1 to run");
        return;
    }
    let Some(path) = lfm2_model_path() else {
        eprintln!("skip: no LFM2 model (set CERA_LFM2_MODEL)");
        return;
    };

    let tok =
        BpeTokenizer::from_gguf(&GgufFile::open(&path).expect("open gguf")).expect("tokenizer");
    let tokens = tok.encode("The quick brown fox jumps over the lazy dog near the riverbank.");
    assert!(tokens.len() > 3, "need a multi-token prompt");

    let cpu = load_model(GgufFile::open(&path).expect("open gguf"), None, 8192).expect("cpu load");
    assert!(
        cpu.supports_hidden_states(),
        "CPU LFM2 must support hidden_states"
    );
    let mut cpu_state = InferenceState::for_prefill(cpu.config(), tokens.len());
    let cpu_hidden = cpu.hidden_states(&tokens, &mut cpu_state);

    let metal = match load_model_metal(GgufFile::open(&path).expect("open gguf"), &path, 8192) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skip: no Metal GPU: {e}");
            return;
        }
    };
    assert!(
        metal.supports_hidden_states(),
        "Metal LFM2 must support hidden_states"
    );
    let mut metal_state = InferenceState::for_prefill(metal.config(), tokens.len());
    let metal_hidden = metal.hidden_states(&tokens, &mut metal_state);

    let d = cpu.config().hidden_size;
    assert_eq!(cpu_hidden.len(), tokens.len() * d, "CPU shape");
    assert_eq!(metal_hidden.len(), tokens.len() * d, "Metal shape");

    let mut min_cos = f32::INFINITY;
    for t in 0..tokens.len() {
        let c = &cpu_hidden[t * d..(t + 1) * d];
        let m = &metal_hidden[t * d..(t + 1) * d];
        assert!(
            m.iter().all(|x| x.is_finite()),
            "token {t}: non-finite Metal hidden"
        );
        let cos = cosine(c, m);
        min_cos = min_cos.min(cos);
        assert!(cos > 0.99, "token {t}: Metal-vs-CPU cosine {cos:.5} < 0.99");
    }
    eprintln!(
        "metal_hidden_states_matches_cpu: {} tokens, D={d}, min per-token cosine {min_cos:.5}",
        tokens.len()
    );

    // A second call must reproduce the first — proves the scratch KV/conv is
    // cleared between runs and no generation KV leaks in. A stale (uncleared)
    // scratch would perturb the conv rolling state and diverge sharply; use a
    // tight tolerance rather than bit-exact equality so ULP-level GPU
    // accumulation noise across two command buffers can't flake the gate.
    let mut s2 = InferenceState::for_prefill(metal.config(), tokens.len());
    let again = metal.hidden_states(&tokens, &mut s2);
    assert_eq!(again.len(), metal_hidden.len(), "reproducibility: shape");
    let max_abs = again
        .iter()
        .zip(&metal_hidden)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-3,
        "Metal hidden_states not reproducible across calls (max abs diff {max_abs:.2e})"
    );
}
