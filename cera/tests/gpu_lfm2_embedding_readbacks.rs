//! An image must cost one GPU readback, not one per patch token.
//!
//! `GpuLfm2Model` used to inherit the default `forward_prefill_from_embeddings`
//! from `model/mod.rs`, which loops `forward_from_embedding` and so ends every
//! frame in a blocking `download_f32` of a full vocab-sized logits vector. All
//! but the last were discarded.
//!
//! Natively that is waste. On wasm it is a hang, which is why this is pinned
//! rather than left to a profiler: `download_f32` blocks in `mpsc::recv` waiting
//! for the buffer-map callback, `poll_wait()` is a no-op there, and the JS event
//! loop that would deliver the callback cannot run while that thread is blocked
//! in it. A browser `appendImage` never returns. No native test can observe the
//! deadlock directly, but it and the waste have the same root, and the readback
//! count is the part a test can hold still.
//!
//! ## Own test binary on purpose
//!
//! `io_stats` are process-global atomics, so a test that resets them races any
//! test sharing its binary. This file therefore holds exactly one test rather
//! than joining `gpu_lfm2_embedding_input.rs`.
//!
//! ## These models are stateful, and reuse contaminates
//!
//! KV cache, conv rolling buffers and prefix cache live in `GpuState`, on the
//! **model**. A fresh `InferenceState` resets none of it, so this loads its own.
#![cfg(feature = "gpu")]

use std::path::PathBuf;

use cera::backend::wgpu::io_stats;
use cera::gguf::GgufFile;
use cera::kv_cache::InferenceState;
use cera::model::{Model, load_model_gpu};

const FIXTURE: &str = "LFM2.5-230M-Q4_K_M.gguf";

/// Enough frames that a per-frame readback is unmistakable against one, and in
/// the range a real image lands in.
const FRAMES: usize = 8;

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
    eprintln!("[gpu-embd-io] SKIP (absent): {}", p.display());
    None
}

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
            eprintln!("[gpu-embd-io] SKIP (no GPU): {e}");
            None
        }
    }
}

/// A deterministic hidden-size vector standing in for one image patch.
fn synthetic_embedding(hidden_size: usize, salt: u32) -> Vec<f32> {
    (0..hidden_size)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(salt);
            ((x >> 8) as f32 / u32::MAX as f32 * 256.0) - 0.5
        })
        .collect()
}

#[test]
fn an_image_costs_one_readback_not_one_per_patch() {
    let Some(path) = fixture_or_skip() else {
        return;
    };
    let Some(gpu) = load_gpu(&path) else { return };

    let hidden_size = gpu.config().hidden_size;
    let embeddings: Vec<f32> = (0..FRAMES)
        .flat_map(|i| synthetic_embedding(hidden_size, 0xC0DE + i as u32))
        .collect();
    let mut state = InferenceState::from_config(gpu.config()).expect("state");

    io_stats::reset();
    let logits = gpu.forward_prefill_from_embeddings(&embeddings, FRAMES, 0, &mut state);
    let stats = io_stats::snapshot();

    assert_eq!(
        stats.readbacks, 1,
        "{FRAMES} embedding frames caused {} readbacks; the override in \
         gpu_lfm2.rs must seed every frame and read logits back once. One per \
         frame means the default impl in model/mod.rs is running again, which \
         also deadlocks appendImage in the browser.",
        stats.readbacks
    );

    // The readback that does happen must still be the whole logits vector, so a
    // future "optimization" cannot pass this by reading back nothing useful.
    assert_eq!(
        logits.len(),
        gpu.config().vocab_size,
        "the single readback must still return full logits"
    );
    assert_eq!(
        state.seq_len, FRAMES,
        "every frame must land in the KV cache"
    );
}
