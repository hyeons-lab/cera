//! Dispatch-count guard for the wgpu LFM2 batched prefill.
//!
//! `forward_prefill` used to gate its batched path on `start_pos == 0`, while
//! `forward_prefill_chunked` calls it once per ubatch at an advancing position —
//! so every chunk after the first silently ran token-by-token. The bug produced
//! **correct logits** and was purely a ~100x dispatch regression, so no
//! correctness assertion could catch it. Hence a test on submit count.
//!
//! Measured on `LFM2.5-230M-Q4_K_M`, 256 tokens at ubatch 64 (4 chunks):
//!
//! | path                          | submits |
//! |-------------------------------|---------|
//! | batched (fixed)               | 29      |
//! | per-chunk fallback (the bug)  | 2906    |
//! | pure per-token                | 4096    |
//!
//! The invariant: submits scale with *chunks*, not *tokens*.
//!
//! ## Why this test is alone in its own file
//!
//! `io_stats` are process-global atomics. Cargo runs the tests inside one file
//! concurrently, so a sibling test's GPU work lands inside this one's measured
//! interval — with the correctness tests alongside it, this read 220 submits
//! instead of 29 and would flake against any tight bound. Each test *file* gets
//! its own process, so keeping this one alone is what makes the count
//! meaningful. Do not add tests to this file; put them in
//! `gpu_lfm2_prefill_equivalence.rs`.
#![cfg(feature = "gpu")]

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use cera::backend::wgpu::io_stats;
use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, KvCompression};
use cera::model::load_model_gpu;
use cera::tokenizer::BpeTokenizer;

/// The `core` fixture set's LFM2 model — fetched on pull requests, so this has
/// real PR coverage rather than the skip-as-pass an `arch`-tier model gets.
const FIXTURE: &str = "LFM2.5-230M-Q4_K_M.gguf";
const N: usize = 256;
const UBATCH: usize = 64;

fn models_dir() -> PathBuf {
    if let Ok(d) = std::env::var("CERA_ORACLE_MODELS_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/oracle/models")
}

#[test]
fn chunked_prefill_batches_every_chunk() {
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

    let tokens: Vec<u32> = {
        let gguf = GgufFile::open(&path).expect("open gguf");
        let tok = BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
        let mut t = tok.encode(&"The quick brown fox jumps over the lazy dog. ".repeat(80));
        assert!(t.len() >= N, "fixture prompt too short: {} < {N}", t.len());
        t.truncate(N);
        t
    };

    let mut st = InferenceState::from_config_with_compression(model.config(), &KvCompression::None)
        .expect("inference state");
    let cancel = AtomicBool::new(false);

    io_stats::reset();
    let (consumed, _) = model.forward_prefill_chunked(&tokens, 0, &mut st, UBATCH, &cancel);
    let stats = io_stats::snapshot();
    assert_eq!(consumed, N, "short prefill");

    let chunks = N.div_ceil(UBATCH);
    eprintln!(
        "[gpu-lfm2] {N} tokens / ubatch {UBATCH} ({chunks} chunks): {} submits",
        stats.submits
    );

    // `N` sits ~9x above the batched count (29) and ~11x below the fallback
    // (2906) — loose enough to absorb kernel-count churn, tight enough that
    // losing the batched path on any chunk fails immediately.
    assert!(
        stats.submits < N as u64,
        "chunked prefill issued {} submits for {N} tokens — submits should scale \
         with chunks ({chunks}), not tokens. The batched path is likely falling \
         back to per-token for chunks at start_pos > 0.",
        stats.submits,
    );
}
