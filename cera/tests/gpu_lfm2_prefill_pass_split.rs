//! Compute-pass grouping guard for the Adreno `mul_mat_*` split.
//!
//! Past 128 prompt tokens `emit_prefill_cmds` splits `mul_mat_*` dispatches
//! into their own passes: a merged pass mixing them with other dispatches
//! loses the device on 2.6B Adreno shapes (empirical kind split, not a
//! SPIR-V/WGSL one; see `is_adreno_split_label`). Nothing about the
//! *output* changes when
//! that grouping regresses — numerics stay identical while the device dies,
//! on-device only — so the grouping gets a counter assertion: a 129-token
//! prefill must issue strictly more passes than a 128-token one on the same
//! model. The split is host-side (`chunk_n > 128`), so the count gap holds
//! on every backend, not just Adreno.
//!
//! ## Why this test is alone in its own file
//!
//! `io_stats` are process-global atomics and cargo runs the tests inside one
//! file concurrently, so a sibling test's GPU work lands inside this one's
//! measured interval. Each test *file* gets its own process, which is what makes
//! the count meaningful. Do not add tests here.
#![cfg(feature = "gpu")]

mod common;

use cera::backend::wgpu::io_stats;
use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, KvCompression};
use cera::model::load_model_gpu;

/// The `core` fixture set's LFM2 model — fetched on pull requests, so this has
/// real PR coverage rather than the skip-as-pass an `arch`-tier model gets.
const FIXTURE: &str = "LFM2.5-230M-Q4_K_M.gguf";

/// A fresh model instance per measurement: the two prompts share a
/// 128-token prefix, so reusing one model would serve the 129-token run
/// from the prefix cache (a 1-token tail prefill) instead of measuring a
/// 129-token prefill.
fn prefill_passes(path: &std::path::Path, toks: &[u32]) -> Option<(u64, u64)> {
    let model = match load_model_gpu(GgufFile::open(path).expect("open gguf"), Some(path), 4096) {
        Ok(m) => m,
        Err(e) => {
            assert!(
                std::env::var("CERA_REQUIRE_GPU")
                    .unwrap_or_default()
                    .is_empty(),
                "CERA_REQUIRE_GPU is set but the GPU model failed to load: {e}"
            );
            eprintln!("[gpu-lfm2] SKIP (no GPU): {e}");
            return None;
        }
    };
    let mut st = InferenceState::from_config_with_compression(model.config(), &KvCompression::None)
        .expect("inference state");
    io_stats::reset();
    let _ = model.forward_prefill(toks, 0, &mut st);
    let snap = io_stats::snapshot();
    Some((snap.passes, snap.submits))
}

#[test]
fn prefill_splits_mul_mat_passes_past_128_tokens() {
    let Some(path) = common::fixture_or_skip(FIXTURE, "gpu-lfm2") else {
        return;
    };
    let toks = common::prompt_tokens(&path, 129);

    let Some((p128, s128)) = prefill_passes(&path, &toks[..128]) else {
        return;
    };
    let Some((p129, s129)) = prefill_passes(&path, &toks[..129]) else {
        return;
    };
    eprintln!("[gpu-lfm2] prefill passes: n=128 -> {p128}, n=129 -> {p129}");

    assert!(
        p128 > 0 && p129 > 0,
        "counted zero compute passes — `GpuContext::begin_pass` is being \
         bypassed, so this grouping check is not measuring anything"
    );
    // Fewer submits than tokens: pins the batched path. Under the per-token
    // fallback each token is its own forward (≥1 submit each), so `p129 >
    // p128` would hold for the wrong reason (token count, not the kind
    // split) — this floor rules that out.
    assert!(
        s128 < 128 && s129 < 129,
        "submits look per-token (s128={s128}, s129={s129}): the batched path is gone"
    );
    assert!(
        p129 > p128,
        "129-token prefill issued {p129} passes, not more than 128-token {p128}: \
         the `mul_mat_*` kind split is gone, and with it the Adreno guard"
    );
}
