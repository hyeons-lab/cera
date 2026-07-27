//! Compute-pass budget for wgpu LFM2 decode.
//!
//! Pass count is a real perf lever and an invisible one. Identical dispatches
//! cost **2.65x more** split across N compute passes than batched into one
//! (measured on an M1 Max), and a pass boundary costs GPU time — a pipeline
//! drain — not just CPU encode. Decode issued 58 passes/token until #318 merged
//! the conv block's three into one, worth +17% decode.
//!
//! Nothing about the *output* changes when that regresses. It is the same class
//! of defect as the chunked-prefill bug in #316: correct logits, 100x the
//! dispatches, invisible to every correctness assertion. So it gets a counter
//! assertion, like the dispatch guard next door.
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
use cera::model::{Model, load_model_gpu};

/// The `core` fixture set's LFM2 model — fetched on pull requests, so this has
/// real PR coverage rather than the skip-as-pass an `arch`-tier model gets.
const FIXTURE: &str = "LFM2.5-230M-Q4_K_M.gguf";

/// Measured 43 on a 14-layer model after #318 (58 before it). The bound sits
/// above that with room for a layer-count-proportional kernel addition, and far
/// enough below 58 to fail if the conv block's passes are ever re-split.
///
/// Deliberately absolute rather than per-layer: the fixture is pinned, so an
/// absolute number is checkable by hand and does not silently rescale if the
/// per-layer structure changes.
const MAX_PASSES_PER_TOKEN: u64 = 50;

fn models_dir() -> PathBuf {
    if let Ok(d) = std::env::var("CERA_ORACLE_MODELS_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/oracle/models")
}

#[test]
fn decode_stays_within_its_compute_pass_budget() {
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
    let _ = model.forward(&[5u32], state.seq_len, &mut state);
    let stats = io_stats::snapshot();

    let layers = model.config().n_layers;
    eprintln!(
        "[gpu-lfm2] decode: {} passes, {} submits ({} layers)",
        stats.passes, stats.submits, layers,
    );

    assert!(
        stats.passes > 0,
        "counted zero compute passes — `GpuContext::begin_pass` is being \
         bypassed, so this budget is not measuring anything"
    );
    assert!(
        stats.passes <= MAX_PASSES_PER_TOKEN,
        "decode issued {} compute passes for one token, over the {MAX_PASSES_PER_TOKEN} \
         budget ({layers} layers). A pass boundary is not free — the same \
         dispatches cost ~2.65x more split across passes than batched into one. \
         Check whether a block that used to share a pass was split, or whether a \
         new `encode_copy` landed between dispatches (a copy is an encoder \
         operation and forces the pass to end).",
        stats.passes,
    );
}
