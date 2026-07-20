//! Differential parity test for the dense-transformer (LLaMA-family) CPU
//! batched-GEMM prefill against the sequential per-token path, on real GGUFs.
//!
//! For each model that is present locally, two fresh `InferenceState`s compute:
//!   (a) sequential per-token logits: `forward` per token, last token's logits;
//!   (b) batched `forward_prefill` over the whole token slice.
//! The last-token logits must agree by cosine similarity + identical argmax.
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
//! itself. On x86_64 that capability is a *runtime* property (AVX-512 VNNI), not
//! just a cfg, so `assert_batched_path_is_live` re-checks it before comparing —
//! without VNNI the model falls back to per-token and the comparison would be
//! vacuous.
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

/// Returns `(cosine, max_abs_diff, argmax_batched, argmax_sequential)` for the
/// last-token logits, or `None` when the fixture is absent.
fn run_parity(rel: &str, tokens: &[u32]) -> Option<(f32, f32, usize, usize)> {
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
    Some((
        cosine(&logits_pre, &logits_seq),
        max_abs_diff(&logits_pre, &logits_seq),
        argmax(&logits_pre),
        argmax(&logits_seq),
    ))
}

/// Whether `forward_prefill` will actually take the batched path here.
///
/// On x86_64 without `blas` that is a *runtime* property (AVX-512 VNNI), not a
/// cfg: without it the model gates itself back onto the per-token path and both
/// halves of this comparison become the same code — a guaranteed pass that
/// proves nothing.
///
/// Absent the capability this skips rather than fails, so a non-VNNI dev box or
/// CI runner does not get a red build for hardware it does not have. Set
/// `CERA_REQUIRE_BATCHED=1` to turn that skip into a failure on a host known to
/// have the hardware. CI does *not* currently set it: the `blas` leg compiles
/// this check out entirely (so it would assert nothing), and the native leg runs
/// on runners with no guaranteed VNNI. Mirrors `CERA_REQUIRE_SIMD` in `simd.rs`.
fn batched_path_is_live(rel: &str) -> bool {
    #[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
    if !cera::backend::cpu::int8_gemm_available() {
        let msg = format!(
            "{rel}: x86_64 host has no runtime AVX-512 VNNI, so `forward_prefill` \
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

fn check(rel: &str, tokens: &[u32]) {
    if !batched_path_is_live(rel) {
        return;
    }
    let Some((cos, max_diff, top_pre, top_seq)) = run_parity(rel, tokens) else {
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
    eprintln!(
        "[parity] {rel} [{path}]: cosine={cos:.6} max_abs_diff={max_diff:.4e} argmax pre={top_pre} seq={top_seq}"
    );

    // Threshold by (path, feature):
    //  - naive NEON: shares the per-token path's Q8_0-quantize + int8-dot, so
    //    bit-identical (cosine = 1.0) → tight 0.9999 bound.
    //  - flash (n ≥ 256): online-softmax + tiling reorder the reduction (cosine
    //    ~0.999), so use the 0.99 bound `blas_parity.rs` established for flash.
    //  - BLAS: projections go through f32 SGEMM while decode stays Q8_0, a
    //    legitimate f32-vs-int reordering (~0.996), so 0.99 for both paths.
    // top-1 agreement (asserted below) is the discriminating correctness check;
    // a real layout/dim/transpose bug drops cosine far below these or flips it.
    #[cfg(all(not(feature = "blas"), target_arch = "aarch64"))]
    let (min_cos, tier) = (if is_flash { 0.99_f32 } else { 0.9999_f32 }, "NEON");
    // x86_64 VNNI shares the same Q8_0-quantize + int8-dot arithmetic as NEON,
    // so it earns the same tight bound — only the label differs.
    #[cfg(all(not(feature = "blas"), target_arch = "x86_64"))]
    let (min_cos, tier) = (if is_flash { 0.99_f32 } else { 0.9999_f32 }, "AVX-512 VNNI");
    #[cfg(feature = "blas")]
    let (min_cos, tier) = (0.99_f32, "BLAS");

    assert!(
        cos > min_cos,
        "{rel} [{path}]: batched-prefill vs sequential cosine = {cos} (< {min_cos} on the {tier} path) — likely a layout/dim/transpose bug"
    );
    assert_eq!(
        top_pre, top_seq,
        "{rel} [{path}]: batched-prefill argmax {top_pre} != sequential argmax {top_seq}"
    );
}

/// Check both the naive (24-token) and flash (288-token) batched-prefill paths.
fn check_both(rel: &str) {
    check(rel, PROMPT);
    check(rel, &flash_prompt());
}

#[test]
#[ignore]
fn llama_batched_prefill_parity_llama3() {
    // Llama-3.2-1B: arch "llama", NORM RoPE with Llama-3 `rope_freqs` factors.
    // The Q8_0 build is used (fully supported); the repo's `Llama-3.2-1B-Q4_0`
    // GGUF carries Q4_1 ffn_down layers in blocks 0/1, a dtype cera can't
    // dequantize, so neither the batched nor the per-token path can run it.
    check_both("target/oracle/models/Llama-3.2-1B-Instruct-Q8_0.gguf");
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
    check_both("target/oracle/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf");
}

#[test]
#[ignore]
fn llama_batched_prefill_parity_qwen3() {
    check_both("target/oracle/models/Qwen3-0.6B-Q8_0.gguf");
}

#[test]
#[ignore]
fn llama_batched_prefill_parity_qwen2() {
    check_both("target/oracle/models/qwen2-0_5b-instruct-q8_0.gguf");
}

#[test]
#[ignore]
fn llama_batched_prefill_parity_granite() {
    check_both("target/oracle/models/granite-3.1-2b-instruct-Q8_0.gguf");
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
    check_both("target/oracle/models/TinyStories-LLaMA2-20M-GQA.Q8_0.gguf");
}

/// Q4_0 projections -> `gemm_q4_0_q8_0`. 30 layers, GQA (9/3), ctx 2048.
/// The multi-GB fixtures above stay for local per-arch coverage (Qwen2 bias,
/// Qwen3 QK-norm, Granite scalars), which no single llama-arch file covers.
#[test]
#[ignore]
fn llama_batched_prefill_parity_smollm_135m_q4_0() {
    check_both("target/oracle/models/SmolLM-135M.Q4_0.gguf");
}
