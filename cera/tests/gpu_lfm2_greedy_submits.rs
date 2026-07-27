//! Queue-submit budget for wgpu greedy decode.
//!
//! A blocking submit costs far more than the work it carries. The greedy argmax
//! used to get its own encoder and its own `submit_and_wait`, and that second
//! stall cost ~1.3 ms/token against the argmax kernel's own ~0.13 ms of GPU
//! time — a pipeline drain charged for one single-workgroup dispatch. Folding it
//! into the output projection's encoder took decode from 59 to 84 tok/s
//! (+39%, LFM2.5-230M-Q4_K_M on an M1 Max).
//!
//! That is 17 -> 16 submits per token, and nothing about the *output* changes if
//! it regresses — the tokens are identical either way. Same invisible-regression
//! class as the compute-pass budget next door, so it gets the same treatment.
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

/// Measured 16 on a 14-layer model after the argmax fold (17 before it). One
/// submit per layer plus one for the output projection + argmax tail.
///
/// The bound is the measured value, not a loose ceiling: re-splitting the tail
/// is the specific regression this guards, and it is worth exactly one submit —
/// so a budget with slack in it would not catch the thing it exists to catch.
const MAX_SUBMITS_PER_TOKEN: u64 = 16;

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
         {MAX_SUBMITS_PER_TOKEN} budget ({layers} layers). A blocking submit \
         drains the pipeline regardless of how little work it carries: the \
         argmax's own submit cost ~10x the argmax kernel. Check whether the \
         argmax (or another tail step) went back to its own encoder instead of \
         riding along in the output projection's.",
        stats.submits,
    );
}
