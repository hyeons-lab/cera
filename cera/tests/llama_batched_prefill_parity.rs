//! Differential parity test for the dense-transformer (LLaMA-family) CPU
//! batched-GEMM prefill against the sequential per-token path, on real GGUFs.
//!
//! For each model that is present locally, two fresh `InferenceState`s compute:
//!   (a) sequential per-token logits: `forward` per token, last token's logits;
//!   (b) batched `forward_prefill` over the whole token slice.
//! The last-token logits must agree by cosine similarity, plus a top-1 check
//! that is tie-tolerant: a differing argmax passes only when the reference rates
//! the two candidates within [`TIE_FRACTION`] of its logit range (see there for
//! the Granite case that forced it).
//! This exercises every dense arch feature — Llama-3 NORM RoPE with
//! freq_factors, Qwen3 per-head QK-norm + decoupled head_dim (NEOX), Qwen2 QKV
//! bias (NEOX), and Granite's four scalar multipliers.
//!
//! Methodology mirrors `blas_parity.rs`: the aarch64 NEON path shares the same
//! Q8_0-quantize + int8-dot arithmetic as the per-token `forward`, so it is
//! bit-identical (cosine = 1.0). A `--features blas` build runs the projections
//! through Accelerate SGEMM in f32 while decode stays on Q8_0 `gemv_preq`, so a
//! legitimate f32-vs-int reduction difference appears (cosine ~0.996); a flat
//! max-abs bound would spuriously fail there. We therefore assert on cosine
//! (tight on NEON, looser on BLAS) plus top-1 agreement, which catches real
//! layout/dim/transpose bugs while tolerating f32 reordering.
//!
//! Compiled only where the batched path exists (`any(aarch64, x86_64, blas)`) so
//! a non-batched target can't silently compare the per-token path against
//! itself. On x86_64 that capability is a *runtime* property (avx2+fma at
//! minimum), not just a cfg, so `assert_batched_path_is_live` re-checks it
//! before comparing — below that the model falls back to per-token and the
//! comparison would be vacuous.
//! Marked `#[ignore]` like the other real-model tests so the mainline
//! `cargo test --workspace` job (which has no ~GB fixtures) does not report a
//! meaningless green; run explicitly with fixtures present:
//!
//! ```text
//! CERA_MODEL_ROOT=/path/to/checkout \
//!   cargo test -p cera --release --test llama_batched_prefill_parity -- --ignored --nocapture
//! ```

#![cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]

use std::path::PathBuf;

/// Try a few candidate roots so the test works both from the crate dir and
/// from a git worktree whose fixtures live in the main checkout.
fn find_model(rel: &str) -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    // Crate dir → workspace root (../ from CARGO_MANIFEST_DIR).
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(&manifest);
        if let Some(parent) = p.parent() {
            roots.push(parent.to_path_buf());
        }
    }
    // Current working directory.
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    // Explicit override for fixtures that live outside the (work)tree — e.g. a
    // git worktree whose large model files sit in the main checkout. Point
    // `CERA_MODEL_ROOT` at the dir that contains `target/oracle/models/…`.
    if let Ok(root) = std::env::var("CERA_MODEL_ROOT") {
        roots.push(PathBuf::from(root));
    }

    for root in roots {
        let candidate = root.join(rel);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

/// A fixed ~24-token prompt of small, in-vocab token ids. Every target model's
/// vocab is >= 32000, so these ids are valid across all four archs.
const PROMPT: &[u32] = &[
    1, 415, 2323, 302, 4843, 349, 264, 2818, 297, 272, 2607, 28725, 304, 378, 349, 2278, 298, 776,
    684, 456, 2758, 302, 12707, 28723,
];

/// A ≥256-token prompt (the 24-token `PROMPT` tiled) that pushes
/// `forward_prefill` over `FLASH_ATTN_THRESHOLD` so the batched path runs the
/// flash-attention branch instead of the naive per-token loop.
fn flash_prompt() -> Vec<u32> {
    PROMPT.iter().copied().cycle().take(288).collect()
}

/// Fraction of the reference's logit range within which two candidates count as
/// tied, so a differing argmax is float reordering rather than a real defect.
///
/// Granite-3.1-2b at the 24-token prompt is the case that forced this: its top
/// two logits are `299` = 8.2572 and `3265` = 8.2363, a separation of 2.09e-2
/// against a range of 23.72 — 0.09% of the span, and *smaller than the 3.5e-1
/// max-abs difference the two paths legitimately show*. Which one wins is
/// decided by reduction order, so it differs by host: aarch64 + Accelerate picks
/// `299` on both paths, x86_64 + OpenBLAS picks `299` batched and `3265`
/// sequentially. A strict `assert_eq!` on argmax therefore encodes the reduction
/// order of whichever machine happened to run it.
///
/// The window is deliberately anchored to the reference alone, not to the
/// observed diff — a bound derived from the thing under test would widen to
/// accommodate any regression. Cosine stays the discriminating check: a real
/// layout/dim/transpose bug drops it far below the per-tier floor, which is
/// asserted independently and unchanged.
const TIE_FRACTION: f32 = 2e-2;

/// Whether the two paths agree on the top token, tolerating a genuine near-tie.
///
/// `seq` is the reference (sequential) logits; both indices are scored against
/// it so the verdict does not depend on the batched path's own values.
fn top1_agrees(seq: &[f32], top_pre: usize, top_seq: usize) -> bool {
    top_pre == top_seq || (seq[top_seq] - seq[top_pre]).abs() <= tie_eps(seq)
}

/// The tie window for a logit vector: [`TIE_FRACTION`] of its full range.
fn tie_eps(seq: &[f32]) -> f32 {
    let (lo, hi) = seq
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
    TIE_FRACTION * (hi - lo)
}

/// How far the reference separates its own best and second-best tokens.
///
/// This is the fragility signal, and it is reported on every run — including the
/// ones that agree. The gap between the two paths' *picks* is zero whenever they
/// match, so it says nothing until the day it fails; this says "granite is
/// 2.09e-2 from flipping" while the test is still green.
fn ref_top2_gap(seq: &[f32]) -> f32 {
    let (mut best, mut second) = (f32::MIN, f32::MIN);
    for &v in seq {
        if v > best {
            second = best;
            best = v;
        } else if v > second {
            second = v;
        }
    }
    best - second
}

/// Outcome of one batched-vs-sequential comparison.
struct Parity {
    cos: f32,
    max_diff: f32,
    top_pre: usize,
    top_seq: usize,
    /// Top-1 verdict, already tie-adjusted (see [`top1_agrees`]).
    top1_ok: bool,
    /// The reference's own best-vs-runner-up separation (see [`ref_top2_gap`]).
    ref_gap: f32,
    /// How far apart the reference rates the two paths' picks; 0 when they agree.
    flip_gap: f32,
    /// The tie window this comparison was judged against.
    eps: f32,
}

/// Runs both paths and scores them, or `None` when the fixture is absent.
fn run_parity(rel: &str, tokens: &[u32]) -> Option<Parity> {
    // Bring `Model` into scope so the boxed trait object's methods resolve.
    #[allow(unused_imports)]
    use cera::model::Model;

    let path = find_model(rel)?;
    eprintln!("[parity] loading {} ({} tok)", path.display(), tokens.len());

    // (a) Sequential per-token path.
    let gguf_seq = cera::gguf::GgufFile::open(&path).unwrap();
    let model_seq = cera::model::load_model(gguf_seq, None, 8192).unwrap();
    let mut state_seq = cera::kv_cache::InferenceState::from_config(model_seq.config()).unwrap();
    let mut logits_seq = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        logits_seq = model_seq.forward(&[tok], i, &mut state_seq);
    }

    // (b) Batched prefill path.
    let gguf_pre = cera::gguf::GgufFile::open(&path).unwrap();
    let model_pre = cera::model::load_model(gguf_pre, None, 8192).unwrap();
    let mut state_pre = cera::kv_cache::InferenceState::from_config(model_pre.config()).unwrap();
    let logits_pre = model_pre.forward_prefill(tokens, 0, &mut state_pre);

    assert_eq!(logits_pre.len(), logits_seq.len(), "logit length mismatch");
    let (top_pre, top_seq) = (argmax(&logits_pre), argmax(&logits_seq));
    Some(Parity {
        cos: cosine(&logits_pre, &logits_seq),
        max_diff: max_abs_diff(&logits_pre, &logits_seq),
        top_pre,
        top_seq,
        top1_ok: top1_agrees(&logits_seq, top_pre, top_seq),
        ref_gap: ref_top2_gap(&logits_seq),
        flip_gap: (logits_seq[top_seq] - logits_seq[top_pre]).abs(),
        eps: tie_eps(&logits_seq),
    })
}

/// Whether `forward_prefill` will actually take the batched path here.
///
/// On x86_64 without `blas` that is a *runtime* property (avx2+fma at minimum),
/// not a cfg: without it the model gates itself back onto the per-token path
/// and both halves of this comparison become the same code — a guaranteed pass
/// that proves nothing.
///
/// Absent the capability this skips rather than fails, so a Scalar-tier dev box
/// CI runner does not get a red build for hardware it does not have. Set
/// `CERA_REQUIRE_BATCHED=1` to turn that skip into a failure on a host known to
/// have the hardware. CI does *not* currently set it: the `blas` leg compiles
/// this check out entirely (so it would assert nothing), and the native leg runs
/// on runners with no guaranteed int8 support. Mirrors `CERA_REQUIRE_SIMD`
/// in `simd.rs`.
fn batched_path_is_live(rel: &str) -> bool {
    #[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
    if !cera::backend::cpu::int8_gemm_available() {
        let msg = format!(
            "{rel}: x86_64 host has no runtime int8 GEMM (needs avx2+fma), so `forward_prefill` \
             falls back to the per-token path — comparing it against itself would \
             pass vacuously"
        );
        assert!(
            std::env::var("CERA_REQUIRE_BATCHED").as_deref() != Ok("1"),
            "CERA_REQUIRE_BATCHED=1 but {msg}"
        );
        eprintln!("[parity] SKIP (no batched path): {msg}");
        return false;
    }
    let _ = rel;
    true
}

/// `x86_naive_floor` is the cosine bound for this model on the x86 non-blas
/// *naive* path. Non-repacked weights (Q8_0) run the batched GEMM bit-identically
/// to the per-token GEMV → tight `0.9999`. Repacked weights take the 8-row
/// interleave prefill GEMM, a legitimate reduction difference whose size depends
/// on the dtype — Q4_0's deferred `-8·Σa` cancels against a large `Σqa` (~0.995,
/// floor `0.99`), Q4_K's mins-only correction is far tighter (~0.9996, floor
/// `0.999`, which still catches an x86 Q6_K regression since Q6_K is not
/// repacked). NEON and BLAS ignore it (see the threshold).
fn check(rel: &str, tokens: &[u32], x86_naive_floor: f32) {
    if !batched_path_is_live(rel) {
        return;
    }
    let Some(p) = run_parity(rel, tokens) else {
        // Absent fixture normally skips — but a skip that reports PASS is how a
        // gate goes green forever without ever running. `CERA_REQUIRE_MODEL`
        // makes the absence a hard failure, so a CI job that is supposed to have
        // the fixture cannot quietly stop testing. Mirrors the lfm2 twin.
        assert!(
            std::env::var("CERA_REQUIRE_MODEL").is_err(),
            "CERA_REQUIRE_MODEL is set but the fixture is absent: {rel} \
             (set CERA_MODEL_ROOT)"
        );
        eprintln!("[parity] SKIP (absent): {rel}");
        return;
    };
    let is_flash = tokens.len() >= 256;
    let path = if is_flash { "flash" } else { "naive" };
    let Parity {
        cos,
        max_diff,
        top_pre,
        top_seq,
        top1_ok,
        ref_gap,
        flip_gap,
        eps,
    } = p;
    // `ref_gap` and `eps` are printed on every run, not just failures: a model
    // whose top-2 sit inside the tie window is one reduction-order nudge from
    // flipping, and that is worth seeing while the test is still green.
    eprintln!(
        "[parity] {rel} [{path}]: cosine={cos:.6} max_abs_diff={max_diff:.4e} \
         argmax pre={top_pre} seq={top_seq} ref_top2_gap={ref_gap:.4e} tie_eps={eps:.4e}"
    );

    // Threshold by (path, feature):
    //  - naive NEON: shares the per-token path's Q8_0-quantize + int8-dot, so
    //    bit-identical (cosine = 1.0) → tight 0.9999 bound. NEON does not repack,
    //    so `x86_naive_floor` does not apply.
    //  - naive x86: `x86_naive_floor`, passed per model. Non-repacked (Q8_0) is
    //    0.9999 (batched GEMM == GEMV). Repacked weights take the 8-row interleave
    //    prefill GEMM, whose reduction (deferred `-8·Σa` / mins correction, no
    //    per-column hsum) differs from the GEMV's — a legitimate reordering, not
    //    the bit-identical arithmetic the tight bar assumes. Q4_0 (~0.995) uses
    //    0.99; Q4_K (~0.9996) uses 0.999, tight enough to still flag an x86 Q6_K
    //    regression in a Q4_K_M mix since Q6_K is not repacked. The GEMV is
    //    untouched, so decode is unaffected.
    //  - flash (n ≥ 256): online-softmax + tiling reorder the reduction (cosine
    //    ~0.999), so use the 0.99 bound `blas_parity.rs` established for flash.
    //  - BLAS: projections go through f32 SGEMM while decode stays Q8_0, a
    //    legitimate f32-vs-int reordering (~0.996), so 0.99 for both paths.
    // top-1 agreement (asserted below) is the discriminating correctness check;
    // a real layout/dim/transpose bug drops cosine far below these or flips it,
    // and the kernels carry their own tight (1e-5) equivalence unit tests.
    #[cfg(any(feature = "blas", target_arch = "aarch64"))]
    let _ = x86_naive_floor; // only the x86 non-blas bound reads it
    #[cfg(all(not(feature = "blas"), target_arch = "aarch64"))]
    let (min_cos, tier) = (if is_flash { 0.99_f32 } else { 0.9999_f32 }, "NEON");
    // The x86 int8 path — VNNI or the AVX2 emulation — shares the same
    // Q8_0-quantize + int8-dot arithmetic as NEON for non-repacked weights, and
    // (since the AVX2 GEMV landed) decode routes through the very same kernels on
    // both tiers, so those earn the same tight bound — only the label differs.
    #[cfg(all(not(feature = "blas"), target_arch = "x86_64"))]
    let (min_cos, tier) = (
        if is_flash { 0.99_f32 } else { x86_naive_floor },
        "x86 int8 (VNNI or AVX2)",
    );
    #[cfg(feature = "blas")]
    let (min_cos, tier) = (0.99_f32, "BLAS");

    assert!(
        cos > min_cos,
        "{rel} [{path}]: batched-prefill vs sequential cosine = {cos} (< {min_cos} on the {tier} path) — likely a layout/dim/transpose bug"
    );
    assert!(
        top1_ok,
        "{rel} [{path}]: batched-prefill argmax {top_pre} != sequential argmax {top_seq}, \
         and they are not tied — the reference separates them by {flip_gap:.4e}, wider \
         than the tie window {eps:.4e} ({TIE_FRACTION:.0e} of its logit range). A flip \
         this far apart is a real disagreement, not reduction order"
    );
}

/// Check both the naive (24-token) and flash (288-token) batched-prefill paths.
fn check_both(rel: &str, x86_naive_floor: f32) {
    check(rel, PROMPT, x86_naive_floor);
    check(rel, &flash_prompt(), x86_naive_floor);
}

// x86 naive cosine floors by weight class (see `check`). Named so the call sites
// read as intent, and so the rationale lives in one place.
const FLOOR_TIGHT: f32 = 0.9999; // not repacked — batched GEMM == GEMV
const FLOOR_Q4_0: f32 = 0.99; // Q4_0 repack: -8·Σa float cancellation (~0.995)
const FLOOR_Q4_K: f32 = 0.999; // Q4_K repack: mins-only correction (~0.9996)

#[test]
#[ignore]
fn llama_batched_prefill_parity_llama3() {
    // Llama-3.2-1B: arch "llama", NORM RoPE with Llama-3 `rope_freqs` factors.
    // The Q8_0 build is used (fully supported); the repo's `Llama-3.2-1B-Q4_0`
    // GGUF carries Q4_1 ffn_down layers in blocks 0/1, a dtype cera can't
    // dequantize, so neither the batched nor the per-token path can run it.
    check_both(
        "target/oracle/models/Llama-3.2-1B-Instruct-Q8_0.gguf",
        FLOOR_TIGHT,
    );
}

/// The dense-transformer K-quant path. Q4_K_M only produces real K-quants when
/// the row length is divisible by 256 (llama.cpp falls back to a legacy quant
/// otherwise), so this needs a model with a 256-divisible hidden size —
/// Llama-3.2-1B at 2048 is 96 Q4_K + 17 Q6_K throughout. A Qwen2-0.5B Q4_K_M
/// would not do: hidden 896 leaves `896 % 256 = 128`, so its projections are
/// Q5_0, which cera cannot load at all.
///
/// This is the fixture the old allowlist NOTE asked for before widening the
/// dense gate to admit K-quants.
#[test]
#[ignore]
fn llama_batched_prefill_parity_llama32_1b_q4_k_m() {
    check_both(
        "target/oracle/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        FLOOR_Q4_K,
    );
}

#[test]
#[ignore]
fn llama_batched_prefill_parity_qwen3() {
    check_both("target/oracle/models/Qwen3-0.6B-Q8_0.gguf", FLOOR_TIGHT);
}

#[test]
#[ignore]
fn llama_batched_prefill_parity_qwen2() {
    check_both(
        "target/oracle/models/qwen2-0_5b-instruct-q8_0.gguf",
        FLOOR_TIGHT,
    );
}

#[test]
#[ignore]
fn llama_batched_prefill_parity_granite() {
    check_both(
        "target/oracle/models/granite-3.1-2b-instruct-Q8_0.gguf",
        FLOOR_TIGHT,
    );
}

// ── CI-sized fixtures ──────────────────────────────────────────────────────
//
// One per batched-GEMM weight dtype. `gemm_preq` dispatches on dtype, so a
// fixture set covering only one leaves the other kernel untested — SmolLM is
// entirely Q4_0 apart from `token_embd`, which the batched path never scans.
// Fetched by `scripts/fetch_test_models.sh`; absent fixtures skip.

/// Q8_0 projections -> `gemm_q8_0_q8_0`. 4 layers, 256 hidden, GQA (16/8),
/// ctx 2048, vocab 32000 — 21 MB and about a second.
#[test]
#[ignore]
fn llama_batched_prefill_parity_tinystories_20m_q8_0() {
    check_both(
        "target/oracle/models/TinyStories-LLaMA2-20M-GQA.Q8_0.gguf",
        FLOOR_TIGHT,
    );
}

/// Q4_0 projections -> `gemm_q4_0_q8_0`. 30 layers, GQA (9/3), ctx 2048.
/// The multi-GB fixtures above stay for local per-arch coverage (Qwen2 bias,
/// Qwen3 QK-norm, Granite scalars), which no single llama-arch file covers.
#[test]
#[ignore]
fn llama_batched_prefill_parity_smollm_135m_q4_0() {
    check_both("target/oracle/models/SmolLM-135M.Q4_0.gguf", FLOOR_Q4_0);
}

/// Unit tests for the tie rule itself. These need no fixture, so they run in the
/// ordinary `cargo test` job — the fixture-backed tests above are `#[ignore]`d
/// and, on a PR, granite is not even downloaded (the parity job fetches the
/// `core` set there), so without these the rule would have no CI coverage at all
/// until a main-branch push.
#[cfg(test)]
mod tie_rule {
    use super::{TIE_FRACTION, tie_eps, top1_agrees};

    /// The real Granite-3.1-2b logits that motivated the rule, measured on the
    /// 24-token prompt: top-2 are 299 and 3265, separated by 2.09e-2 against a
    /// range of 23.72. Index 0 stands in for the -15.46 floor.
    fn granite_like() -> Vec<f32> {
        let mut v = vec![-15.4587_f32; 8];
        v[1] = 8.257178; // token 299 — sequential argmax on aarch64/Accelerate
        v[2] = 8.236296; // token 3265 — sequential argmax on x86_64/OpenBLAS
        v[3] = 7.619644;
        v
    }

    #[test]
    fn identical_argmax_agrees() {
        let seq = granite_like();
        assert!(top1_agrees(&seq, 1, 1));
    }

    /// The exact CI failure: the two hosts disagree on which of the top-2 wins.
    #[test]
    fn granite_top2_flip_is_a_tie() {
        let seq = granite_like();
        let gap = (seq[1] - seq[2]).abs();
        assert!(
            gap < tie_eps(&seq),
            "measured Granite gap {gap} should sit inside the window {}",
            tie_eps(&seq)
        );
        assert!(
            top1_agrees(&seq, 1, 2),
            "flip between the tied top-2 must pass"
        );
        assert!(top1_agrees(&seq, 2, 1), "and must be symmetric");
    }

    /// A flip to a genuinely lower logit is still a failure — the rule must not
    /// have degenerated into "any argmax is fine".
    #[test]
    fn distant_flip_still_fails() {
        let seq = granite_like();
        assert!(
            !top1_agrees(&seq, 3, 1),
            "0.64 apart is well outside the window and must not be excused"
        );
        assert!(!top1_agrees(&seq, 0, 1), "and neither is the -15.46 floor");
    }

    /// The window scales with the reference's own range rather than being a
    /// fixed logit delta, so a model with a wider spread is not held to a
    /// tighter relative bar.
    #[test]
    fn window_scales_with_range() {
        let narrow = vec![0.0_f32, 1.0];
        let wide = vec![0.0_f32, 70.0];
        assert!(tie_eps(&narrow) < tie_eps(&wide));
        assert_eq!(tie_eps(&wide), TIE_FRACTION * 70.0);
    }
}
