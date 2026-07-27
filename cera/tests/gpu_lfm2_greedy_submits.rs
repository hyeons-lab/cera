//! Queue-submit budget for wgpu greedy decode.
//!
//! A GPU round trip costs far more than the work it carries, and greedy decode
//! was making two it did not need:
//!
//! - the argmax got its own encoder and its own `submit_and_wait` — a pipeline
//!   drain charged for one single-workgroup dispatch (~1.3 ms against the
//!   kernel's own ~0.13 ms);
//! - the 4-byte result readback then submitted *again* to copy it to staging
//!   (~1.5 ms, of which only ~8 µs was the CPU-side handoff).
//!
//! Both now ride along in the output projection's submission: 17 -> 15 submits
//! per token, decode 60 -> 112 tok/s (LFM2.5-230M-Q4_K_M, M1 Max).
//!
//! Nothing about the *output* changes if this regresses — the tokens are
//! identical either way. Same invisible-regression class as the compute-pass
//! budget next door, so it gets the same treatment.
//!
//! Note this is the *greedy* path (`Model::forward_greedy`), not `forward`:
//! only greedy runs the argmax, so only greedy can regress this way.
//!
//! ## Why this test is alone in its own file
//!
//! `io_stats` are process-global atomics and cargo runs the tests inside one
//! file concurrently, so a sibling test's GPU work lands inside this one's
//! measured interval. Each test *file* gets its own process, which is what makes
//! the count meaningful. Do not add tests here.
#![cfg(feature = "gpu")]

use std::path::PathBuf;

use cera::backend::wgpu::io_stats;
use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, KvCompression};
use cera::model::load_model_gpu;

/// The `core` fixture set's LFM2 model — fetched on pull requests, so this has
/// real PR coverage rather than the skip-as-pass an `arch`-tier model gets.
const FIXTURE: &str = "LFM2.5-230M-Q4_K_M.gguf";

/// Measured 15 on a 14-layer model (17 before the two folds): one submit per
/// layer, plus one carrying the output projection, the argmax, and the copy that
/// stages the result for readback.
///
/// The bound is the measured value, not a loose ceiling: splitting either fold
/// back out is the specific regression this guards, and each is worth exactly
/// one submit — a budget with slack in it would not catch what it exists to
/// catch.
const MAX_SUBMITS_PER_TOKEN: u64 = 15;

fn models_dir() -> PathBuf {
    if let Ok(d) = std::env::var("CERA_ORACLE_MODELS_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/oracle/models")
}

#[test]
fn greedy_decode_stays_within_its_submit_budget() {
    let path = models_dir().join(FIXTURE);
    if !path.exists() {
        assert!(
            std::env::var("CERA_REQUIRE_MODEL")
                .unwrap_or_default()
                .is_empty(),
            "CERA_REQUIRE_MODEL is set but {FIXTURE} is absent at {}",
            path.display()
        );
        eprintln!("[gpu-lfm2] SKIP (absent): {}", path.display());
        return;
    }

    let model = match load_model_gpu(
        GgufFile::open(&path).expect("open gguf"),
        Some(path.as_path()),
        4096,
    ) {
        Ok(m) => m,
        Err(e) => {
            assert!(
                std::env::var("CERA_REQUIRE_GPU")
                    .unwrap_or_default()
                    .is_empty(),
                "CERA_REQUIRE_GPU is set but the GPU model failed to load: {e}"
            );
            eprintln!("[gpu-lfm2] SKIP (no GPU): {e}");
            return;
        }
    };

    let mut state =
        InferenceState::from_config_with_compression(model.config(), &KvCompression::None)
            .expect("inference state");

    // Prefill first so the counted step is a real decode at a non-zero position,
    // not the prefill path.
    let prompt: Vec<u32> = (1..24u32).collect();
    let _ = model.forward_prefill(&prompt, 0, &mut state);

    io_stats::reset();
    let _ = model.forward_greedy(&[5u32], state.seq_len, &mut state);
    let stats = io_stats::snapshot();

    let layers = model.config().n_layers;
    eprintln!(
        "[gpu-lfm2] greedy decode: {} submits, {} passes ({} layers)",
        stats.submits, stats.passes, layers,
    );

    assert!(
        stats.submits > 0,
        "counted zero submits — `GpuContext::submit_encoder` is being bypassed, \
         so this budget is not measuring anything"
    );
    assert!(
        stats.submits <= MAX_SUBMITS_PER_TOKEN,
        "greedy decode issued {} queue submits for one token, over the \
         {MAX_SUBMITS_PER_TOKEN} budget ({layers} layers). A submit costs a GPU \
         round trip regardless of how little work it carries — the argmax's own \
         submit cost ~10x the argmax kernel, and staging its 4-byte result cost \
         another ~1.5 ms. Check whether the argmax, or the copy that stages it \
         for readback, went back to its own encoder instead of riding along in \
         the output projection's encoder.",
        stats.submits,
    );
}
