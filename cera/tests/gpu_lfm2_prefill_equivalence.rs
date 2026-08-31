//! wgpu LFM2 prefill equivalence: chunked vs monolithic, and batched vs per-token.
//!
//! `lfm2_batched_prefill_parity.rs` covers the same equivalence on **CPU**, and
//! `gpu_transformer_parity.rs` covers the GPU batched-vs-per-token differential
//! for the **dense** archs. LFM2 on the GPU had neither — which matters more
//! than for a dense model, because the gated-conv block carries a rolling buffer
//! whose state machine is written twice: once walking tokens inside
//! `conv1d_fused_batch`, once across dispatches in `conv1d_fused`.
//!
//! Correctness only. The companion guard on *dispatch count* lives in
//! `gpu_lfm2_prefill_dispatch.rs` — the bug it catches produces correct logits,
//! so nothing here can see it, and its counters are process-global so it needs
//! its own test binary.
//!
//! ## These models are stateful — reuse contaminates
//!
//! `GpuLfm2Model` keeps the KV cache, the conv rolling buffers and the prefix
//! cache in `GpuState`, on the **model**, not in `InferenceState`. A fresh
//! `InferenceState` does not reset any of it, and the two entry points differ:
//! `forward_prefill(start_pos = 0)` zeroes the conv buffers and writes the
//! prefix cache, while `forward()` does neither. So two measurements sharing one
//! model instance are not independent — the second inherits the first's conv
//! state, and can hit the prefix-cache entry the first inserted, taking the
//! restore path instead of the one under test.
//!
//! Every measurement below therefore loads its own model. Reusing one makes
//! these tests report a ~0.96 cosine and a 10x submit count out of nowhere.
#![cfg(feature = "gpu")]

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, KvCompression};
use cera::model::{Model, load_model_gpu};
use cera::sampler::argmax;
use cera::tokenizer::BpeTokenizer;

/// The `core` fixture set's LFM2 model — fetched on pull requests, so this file
/// has real PR coverage rather than the skip-as-pass an `arch`-tier model gets.
const FIXTURE: &str = "LFM2.5-230M-Q4_K_M.gguf";

fn models_dir() -> PathBuf {
    if let Ok(d) = std::env::var("CERA_ORACLE_MODELS_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/oracle/models")
}

/// Resolve the fixture, or signal skip. `CERA_REQUIRE_MODEL` turns absence into
/// a hard failure — the GPU job fetches this fixture, so a skip there would mean
/// the job is green without having run anything. Mirrors `CERA_REQUIRE_GPU`.
fn fixture_or_skip() -> Option<PathBuf> {
    let p = models_dir().join(FIXTURE);
    if p.exists() {
        return Some(p);
    }
    assert!(
        std::env::var("CERA_REQUIRE_MODEL")
            .unwrap_or_default()
            .is_empty(),
        "CERA_REQUIRE_MODEL is set but {FIXTURE} is absent at {}",
        p.display()
    );
    eprintln!("[gpu-lfm2] SKIP (absent): {}", p.display());
    None
}

/// A model instance per call — see the module docs on statefulness.
fn load(path: &std::path::Path) -> Option<Box<dyn Model>> {
    match load_model_gpu(GgufFile::open(path).expect("open gguf"), Some(path), 4096) {
        Ok(m) => Some(m),
        Err(e) => {
            assert!(
                std::env::var("CERA_REQUIRE_GPU")
                    .unwrap_or_default()
                    .is_empty(),
                "CERA_REQUIRE_GPU is set but the GPU model failed to load: {e}"
            );
            eprintln!("[gpu-lfm2] SKIP (no GPU): {e}");
            None
        }
    }
}

fn state(m: &dyn Model) -> InferenceState {
    InferenceState::from_config_with_compression(m.config(), &KvCompression::None)
        .expect("inference state")
}

fn tokens(path: &std::path::Path, n: usize) -> Vec<u32> {
    let gguf = GgufFile::open(path).expect("open gguf");
    let tok = BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
    let text = "The quick brown fox jumps over the lazy dog near the riverbank. ".repeat(60);
    let mut t = tok.encode(&text);
    assert!(t.len() >= n, "fixture prompt too short: {} < {n}", t.len());
    t.truncate(n);
    t
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    (d / (na.sqrt() * nb.sqrt())) as f32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Chunking must not change the answer: `forward_prefill_chunked` splits the
/// prompt across `forward_prefill` calls at advancing positions, and the conv
/// rolling buffer has to carry across those boundaries exactly as it does
/// within one call.
#[test]
fn chunked_prefill_matches_monolithic() {
    let Some(path) = fixture_or_skip() else {
        return;
    };
    let toks = tokens(&path, 256);

    let Some(m) = load(&path) else { return };
    let mono = m.forward_prefill(&toks, 0, &mut state(m.as_ref()));
    drop(m);

    for ubatch in [32usize, 64, 128] {
        let Some(m) = load(&path) else { return };
        let mut st = state(m.as_ref());
        let cancel = AtomicBool::new(false);
        let (consumed, chunked) = m.forward_prefill_chunked(&toks, 0, &mut st, ubatch, &cancel);
        assert_eq!(consumed, toks.len(), "ubatch {ubatch}: short prefill");
        let chunked = chunked.expect("chunked logits");

        let cos = cosine(&mono, &chunked);
        let diff = max_abs_diff(&mono, &chunked);
        eprintln!(
            "[gpu-lfm2] ubatch={ubatch}: cosine={cos:.7} max_abs_diff={diff:.4e} \
             argmax {}/{}",
            argmax(&mono),
            argmax(&chunked)
        );
        // Chunking moves dispatch boundaries but not the reduction order of any
        // single output element, so this measures bit-exact (max_abs_diff 0.0)
        // on every GPU tried. The bound is nonetheless a tolerance, not
        // `assert_eq!`: exactness across every driver/tiling combination isn't
        // something this test can promise, and 1e-3 still sits orders of
        // magnitude below any wiring bug — a conv rolling buffer that fails to
        // carry across a chunk boundary moves logits by whole units.
        //
        // Deliberately NOT `assert_eq!` on the vectors themselves: that prints
        // both 50k-element f32 vectors on failure (~460 KB of scrollback).
        assert!(
            diff <= 1e-3,
            "ubatch {ubatch}: chunked prefill diverged from monolithic — \
             max_abs_diff {diff:.4e}, cosine {cos:.7}, argmax {}/{}. The conv \
             rolling buffer or KV state is likely not carrying across a chunk \
             boundary.",
            argmax(&mono),
            argmax(&chunked),
        );
        assert_eq!(
            argmax(&mono),
            argmax(&chunked),
            "ubatch {ubatch}: chunked prefill changed argmax (cosine {cos:.7})"
        );
    }
}

/// GPU batched prefill vs the per-token decode loop it must agree with — the
/// LFM2 counterpart to `gpu_transformer_parity`'s dense-arch differential.
#[test]
fn batched_prefill_matches_per_token() {
    let Some(path) = fixture_or_skip() else {
        return;
    };
    // Short on purpose: the per-token leg is 16 submits/token and this runs on
    // software Vulkan in CI.
    let toks = tokens(&path, 24);

    let Some(m) = load(&path) else { return };
    let batched = m.forward_prefill(&toks, 0, &mut state(m.as_ref()));
    drop(m);

    let Some(m) = load(&path) else { return };
    let mut st = state(m.as_ref());
    let mut per_token = Vec::new();
    for (i, &t) in toks.iter().enumerate() {
        per_token = m.forward(&[t], i, &mut st);
    }

    let cos = cosine(&batched, &per_token);
    let diff = max_abs_diff(&batched, &per_token);
    eprintln!(
        "[gpu-lfm2] batched vs per-token: cosine={cos:.7} max_abs_diff={diff:.4e} \
         argmax {}/{}",
        argmax(&batched),
        argmax(&per_token)
    );

    // The two paths run different kernels (GEMM vs GEMV, `conv1d_fused_batch`
    // vs `conv1d_fused`), so they reorder float reductions and are not bit
    // equal. In CI on software lavapipe Vulkan, float reordering lands ~0.955
    // while on hardware GPUs cosine lands 1.0000000.
    assert!(
        cos > 0.95,
        "batched vs per-token cosine {cos} (max_abs_diff {diff:.4e}) — likely a \
         batched-path wiring bug"
    );
}
