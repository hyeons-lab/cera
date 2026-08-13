//! Dispatch-count guard for the wgpu routed-FFN batched prefill.
//!
//! `tests/wgpu_moe_oracle.rs` asserts that `forward_prefill` on an `lfm2moe`
//! model agrees with this backend's own decode and with the CPU. Every one of
//! those assertions holds *vacuously* if the batched path silently bails and
//! prefill runs the per-token decode loop instead: prefill would then literally
//! be decode, so of course it agrees with it. That is not a hypothetical failure
//! mode in this repo. `gpu_lfm2_prefill_dispatch.rs` exists because the batched
//! path once bailed on every chunk after the first with correct logits and a
//! ~100x dispatch regression, and a separate bug dropped a whole dense model
//! onto the per-token path because one tensor in it was an unbatchable dtype.
//!
//! A routed model has a fresh way to trip the same wire: `upload_moe` admits
//! Q4_0 experts only, and `unbatchable_matmul_weight` deliberately reports
//! nothing for a routed layer, so the two checks that decide the path no longer
//! look at the same weights the routed FFN runs on.
//!
//! The invariant is the one that file states: submits scale with *chunks*, not
//! with tokens.
//!
//! Measured on LFM2.5-8B-A1B-Q4_0 (24 layers, 22 of them routed), the same
//! 7-token fixture the oracle uses:
//!
//! | path                       | submits | passes |
//! |----------------------------|---------|--------|
//! | batched (correct)          | 33      | 171    |
//! | per-token arm, same tokens | 182     | 434    |
//!
//! The two are only ~5.5x apart because 7 tokens is a short prompt and the
//! batched path still submits per phase; the point is that the second column
//! scales with tokens and the first does not.
//!
//! ## Why this test is alone in its own file
//!
//! `io_stats` are process-global atomics, and cargo runs the tests inside one
//! file concurrently, so a sibling test's GPU work would land inside this one's
//! measured interval. Each test *file* gets its own process. Do not add tests
//! here; put them in `wgpu_moe_oracle.rs`.
//!
//! Gated on an `lfm2moe` GGUF via `CERA_LFM2MOE_MODEL`. Run:
//!   CERA_LFM2MOE_MODEL=... cargo test -p cera --release --features gpu \
//!     --test wgpu_moe_prefill_dispatch -- --ignored --nocapture

#![cfg(feature = "gpu")]

use std::path::PathBuf;

use cera::backend::wgpu::io_stats;
use cera::gguf::GgufFile;
use cera::kv_cache::InferenceState;
use cera::model::load_model_gpu;

/// Same seven-token fixture the oracle uses, so a failure here and a failure
/// there are talking about the same prefill.
const TOKENS: &[u32] = &[124894, 597, 5205, 302, 3980, 355, 20551];

/// Ceiling on submits for this one chunk: comfortably above the measured 33
/// (where the encoder boundaries land is an implementation detail worth slack,
/// and this test is not a perf budget) and far below the 182 the per-token arm
/// needs for the same tokens. A fallback cannot slip under this: it costs ~26
/// submits per token on this model, so it would clear the cap at three.
const MAX_SUBMITS: u64 = 60;

fn model_path() -> Option<PathBuf> {
    let p = std::env::var("CERA_LFM2MOE_MODEL")
        .ok()
        .map(PathBuf::from)?;
    (p.exists() && GgufFile::open(&p).is_ok()).then_some(p)
}

#[test]
#[ignore = "needs an lfm2moe GGUF via CERA_LFM2MOE_MODEL"]
fn routed_prefill_submits_scale_with_chunks_not_tokens() {
    let Some(path) = model_path() else {
        eprintln!("[wgpu-moe-prefill] SKIP: set CERA_LFM2MOE_MODEL");
        return;
    };
    let model = load_model_gpu(
        GgufFile::open(&path).expect("model_path checked this opens"),
        Some(&path),
        4096,
    )
    .expect("lfm2moe loads on the wgpu backend");
    assert!(
        model.config().moe.is_some(),
        "CERA_LFM2MOE_MODEL is not a mixture-of-experts model; this guard would pass vacuously"
    );
    let mut state = InferenceState::for_prefill(model.config(), TOKENS.len())
        .expect("prefill state for a prompt this short");

    // Reset after the load: weight upload submits too, and it is not what this
    // measures.
    io_stats::reset();
    let logits = model.forward_prefill(TOKENS, 0, &mut state);
    let stats = io_stats::snapshot();

    assert_eq!(logits.len(), model.config().vocab_size);
    eprintln!(
        "[wgpu-moe-prefill] {} tokens in one chunk: {} submits, {} passes",
        TOKENS.len(),
        stats.submits,
        stats.passes,
    );
    // Lower bound first: a cap alone passes trivially if the counter itself
    // stops firing, and `logits.len()` does not prove it ran. Well under the
    // measured 33 so encoder-boundary changes do not trip it, but above the
    // one-or-two a stray non-prefill submit would leave behind.
    const MIN_SUBMITS: u64 = 5;
    assert!(
        stats.submits >= MIN_SUBMITS,
        "routed prefill recorded only {} submits (floor {MIN_SUBMITS}); io_stats is not \
         being fed, so the cap below would pass while measuring nothing",
        stats.submits,
    );
    assert!(
        stats.submits <= MAX_SUBMITS,
        "routed prefill issued {} submits for {} tokens (cap {MAX_SUBMITS}); the batched path \
         has bailed to the per-token decode loop, which would also make every assertion in \
         wgpu_moe_oracle.rs pass vacuously",
        stats.submits,
        TOKENS.len(),
    );
}
