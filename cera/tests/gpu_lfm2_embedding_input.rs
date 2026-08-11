//! wgpu LFM2 `forward_from_embedding` oracle, pinned to the CPU model.
//!
//! `GpuLfm2Model` used the trait's default `supports_embedding_input()`
//! (`false`) and a `forward_from_embedding` that panicked, so image input
//! could not reach the WebGPU backend at all: an image arrives from the
//! mmproj's projector as hidden-size vectors with no token id behind them,
//! and `Session::append_embeddings` refuses a backend that cannot take them.
//!
//! The implementation re-seeds `hidden_buf` and reuses the existing layer
//! dispatch, so the risk is not the arithmetic (already covered by the decode
//! oracles) but the seam: writing the wrong buffer, the wrong length, or at
//! the wrong position would still produce plausible-looking logits.
//!
//! ## These models are stateful, and reuse contaminates
//!
//! Same hazard as `gpu_lfm2_prefill_equivalence.rs`: the KV cache, conv
//! rolling buffers and prefix cache live in `GpuState`, on the **model**. A
//! fresh `InferenceState` resets none of it. Every measurement below loads its
//! own model.
#![cfg(feature = "gpu")]

use std::path::PathBuf;

use cera::gguf::GgufFile;
use cera::kv_cache::InferenceState;
use cera::model::{Model, load_model, load_model_gpu};

/// The `core` fixture set's LFM2 model, fetched on pull requests, so this
/// file gets real PR coverage rather than an `arch`-tier skip-as-pass.
const FIXTURE: &str = "LFM2.5-230M-Q4_K_M.gguf";

fn models_dir() -> PathBuf {
    if let Ok(d) = std::env::var("CERA_ORACLE_MODELS_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/oracle/models")
}

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
    eprintln!("[gpu-embd] SKIP (absent): {}", p.display());
    None
}

/// A model instance per call; see the module docs on statefulness.
fn load_gpu(path: &std::path::Path) -> Option<Box<dyn Model>> {
    match load_model_gpu(GgufFile::open(path).expect("open gguf"), Some(path), 4096) {
        Ok(m) => Some(m),
        Err(e) => {
            assert!(
                std::env::var("CERA_REQUIRE_GPU")
                    .unwrap_or_default()
                    .is_empty(),
                "CERA_REQUIRE_GPU is set but the GPU model failed to load: {e}"
            );
            eprintln!("[gpu-embd] SKIP (no GPU): {e}");
            None
        }
    }
}

fn load_cpu(path: &std::path::Path) -> Box<dyn Model> {
    load_model(GgufFile::open(path).expect("open gguf"), Some(path), 4096).expect("load cpu model")
}

/// A deterministic hidden-size vector standing in for one image patch.
///
/// Synthetic rather than a real embedding-table row because the row accessors
/// are private to each backend, and the point here is the seam, not the
/// values: both backends receive the identical vector, so any disagreement is
/// the plumbing. Scaled small so layer-0 RMSNorm sees a plausible magnitude.
fn synthetic_embedding(hidden_size: usize, salt: u32) -> Vec<f32> {
    (0..hidden_size)
        .map(|i| {
            // Cheap deterministic hash, no rand dependency.
            let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(salt);
            ((x >> 8) as f32 / u32::MAX as f32 * 256.0) - 0.5
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "logit vectors differ in length");
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// The capability probe has to flip, or `Session::append_embeddings` returns
/// `UnsupportedModality` before reaching any of the work below.
#[test]
fn gpu_model_advertises_embedding_input() {
    let Some(path) = fixture_or_skip() else {
        return;
    };
    let Some(gpu) = load_gpu(&path) else { return };
    assert!(
        gpu.supports_embedding_input(),
        "GpuLfm2Model must advertise embedding input, or Session::append_embeddings \
         refuses images on the WebGPU backend before calling forward_from_embedding"
    );
}

/// The oracle: identical embedding in, matching logits out.
#[test]
fn forward_from_embedding_matches_the_cpu_model() {
    let Some(path) = fixture_or_skip() else {
        return;
    };
    let Some(gpu) = load_gpu(&path) else { return };
    let cpu = load_cpu(&path);

    let hidden_size = cpu.config().hidden_size;
    let embedding = synthetic_embedding(hidden_size, 0x5EED);

    let mut cpu_state = InferenceState::from_config(cpu.config()).expect("state");
    let mut gpu_state = InferenceState::from_config(gpu.config()).expect("state");

    let cpu_logits = cpu.forward_from_embedding(&embedding, 0, &mut cpu_state);
    let gpu_logits = gpu.forward_from_embedding(&embedding, 0, &mut gpu_state);

    let cos = cosine(&cpu_logits, &gpu_logits);
    assert!(
        cos > 0.999,
        "GPU forward_from_embedding diverged from CPU: cosine {cos:.6}. \
         The layer dispatch is shared with the token path, so a low cosine \
         points at how hidden_buf is seeded rather than at the kernels."
    );
}

/// Position handling: a second embedding must attend to the first rather than
/// overwrite it. Feeding the same vector twice and getting identical logits
/// both times would mean the KV cache never advanced, which is exactly what a
/// `pos` threaded through wrongly looks like.
#[test]
fn successive_embeddings_advance_the_kv_cache() {
    let Some(path) = fixture_or_skip() else {
        return;
    };
    let Some(gpu) = load_gpu(&path) else { return };

    let hidden_size = gpu.config().hidden_size;
    let embedding = synthetic_embedding(hidden_size, 0xA11CE);
    let mut state = InferenceState::from_config(gpu.config()).expect("state");

    let first = gpu.forward_from_embedding(&embedding, 0, &mut state);
    // `forward_from_embedding` advances `seq_len` itself, on both backends.
    // Adding a manual step here would land the next embedding at position 2
    // and leave an unwritten hole at 1 for attention to read.
    assert_eq!(
        state.seq_len, 1,
        "forward_from_embedding must advance seq_len itself"
    );
    let second = gpu.forward_from_embedding(&embedding, 1, &mut state);
    assert_eq!(
        state.seq_len, 2,
        "second embedding must advance seq_len too"
    );

    let cos = cosine(&first, &second);
    assert!(
        cos < 0.9999,
        "the same embedding at positions 0 and 1 produced identical logits \
         (cosine {cos:.6}); the second step did not attend to the first, so \
         the KV cache is not advancing"
    );
}

/// The CPU model treats `state.seq_len` as authoritative over the `pos`
/// argument; the GPU one must agree, or a caller splicing an image into a
/// prompt lands its patches at the wrong positions on one backend only.
#[test]
fn cpu_and_gpu_agree_on_position_after_a_prefix() {
    let Some(path) = fixture_or_skip() else {
        return;
    };
    let Some(gpu) = load_gpu(&path) else { return };
    let cpu = load_cpu(&path);

    let hidden_size = cpu.config().hidden_size;
    let embedding = synthetic_embedding(hidden_size, 0xB0B);

    // Establish the same one-token prefix on both, then append an embedding.
    let mut cpu_state = InferenceState::from_config(cpu.config()).expect("state");
    let mut gpu_state = InferenceState::from_config(gpu.config()).expect("state");
    cpu.forward(&[1], 0, &mut cpu_state);
    gpu.forward(&[1], 0, &mut gpu_state);
    // `forward` advances `seq_len` itself too (CPU inside `run_layers`, GPU
    // inside the shared compute tail), so the prefix leaves both at 1 and the
    // embedding below appends at 1. Stepping them by hand would put it at 2 on
    // both backends: still *agreeing*, and so still passing the cosine check,
    // while quietly testing a position that is not the one after the prefix.
    assert_eq!(cpu_state.seq_len, 1, "CPU forward must advance seq_len");
    assert_eq!(gpu_state.seq_len, 1, "GPU forward must advance seq_len");

    let cpu_logits = cpu.forward_from_embedding(&embedding, 1, &mut cpu_state);
    let gpu_logits = gpu.forward_from_embedding(&embedding, 1, &mut gpu_state);

    let cos = cosine(&cpu_logits, &gpu_logits);
    assert!(
        cos > 0.999,
        "GPU and CPU disagree on an embedding appended after a prefix: \
         cosine {cos:.6}"
    );
}

/// The batched override must agree with the CPU model frame for frame.
///
/// The readback count is pinned separately in
/// `gpu_lfm2_embedding_readbacks.rs`; counting alone would pass just as happily
/// if the override seeded the frames wrongly, or skipped them.
#[test]
fn multi_frame_prefill_from_embeddings_matches_the_cpu_model() {
    let Some(path) = fixture_or_skip() else {
        return;
    };
    let Some(gpu) = load_gpu(&path) else { return };
    let cpu = load_cpu(&path);

    let hidden_size = cpu.config().hidden_size;
    const FRAMES: usize = 6;
    let embeddings: Vec<f32> = (0..FRAMES)
        .flat_map(|i| synthetic_embedding(hidden_size, 0xF00D + i as u32))
        .collect();

    let mut cpu_state = InferenceState::from_config(cpu.config()).expect("state");
    let mut gpu_state = InferenceState::from_config(gpu.config()).expect("state");

    let cpu_logits = cpu.forward_prefill_from_embeddings(&embeddings, FRAMES, 0, &mut cpu_state);
    let gpu_logits = gpu.forward_prefill_from_embeddings(&embeddings, FRAMES, 0, &mut gpu_state);

    assert_eq!(
        cpu_state.seq_len, gpu_state.seq_len,
        "backends disagree on how far a {FRAMES}-frame image advanced the cache"
    );
    let cos = cosine(&cpu_logits, &gpu_logits);
    assert!(
        cos > 0.999,
        "GPU multi-frame prefill_from_embeddings diverged from CPU: cosine \
         {cos:.6}. The logits are the last frame's, so a low cosine means the \
         frames were seeded at the wrong positions or in the wrong order."
    );
}
