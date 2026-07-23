//! Differential parity test for the LFM2 CPU batched-GEMM prefill against the
//! sequential per-token path, on real GGUFs — the **K-quant** case in particular.
//!
//! The dense-transformer twin of this test (`llama_batched_prefill_parity.rs`)
//! already existed; LFM2 had none, which is why the batched K-quant GEMM (T1) could
//! land with nothing able to catch a wiring bug in it. A bad GEMM here does not
//! crash — it produces *fluent but different* text, which is indistinguishable from
//! "the model is just like that" by eye.
//!
//! Why the bar is tight on NEON: on aarch64 the per-token Q4_K/Q6_K GEMVs
//! quantize the activations to Q8_0 and run the same int8 dot as the batched
//! GEMM, so the two paths do the *same arithmetic* and differ only in float
//! summation order — cosine ~1.0, argmax matches. On x86_64 that held too until
//! the Q4_K projections gained the repacked 8-row-interleave prefill GEMM, whose
//! reduction (deferred mins correction, no per-column hsum) differs from the
//! per-token GEMV's — a legitimate reorder, like flash/BLAS, landing ~0.9996
//! here. So the x86 naive bound is relaxed to match; aarch64 (no repack) keeps
//! the tight bar, and argmax is asserted on every path. (Under `blas` the
//! projections run through f32 SGEMM instead, a legitimate reduction difference,
//! so the bound is looser — mirroring `llama_batched_prefill_parity`.)
//!
//! ```text
//! CERA_MODEL_ROOT=/path/to/models \
//!   cargo test -p cera --release --test lfm2_batched_prefill_parity -- --ignored --nocapture
//! ```

#![cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]

use std::path::PathBuf;

/// Resolve a model by trying a few roots, plus `CERA_LFM2_MODEL` as a direct path.
fn find_model(rel: &str) -> Option<PathBuf> {
    if let Ok(direct) = std::env::var("CERA_LFM2_MODEL") {
        let p = PathBuf::from(direct);
        if p.exists() {
            return Some(p);
        }
    }
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR")
        && let Some(parent) = PathBuf::from(&manifest).parent()
    {
        roots.push(parent.to_path_buf());
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    if let Ok(root) = std::env::var("CERA_MODEL_ROOT") {
        roots.push(PathBuf::from(root));
    }
    roots.into_iter().map(|r| r.join(rel)).find(|c| c.exists())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

/// Small in-vocab token ids; LFM2 vocab is 65536, so these are always valid.
const PROMPT: &[u32] = &[
    1, 415, 2323, 302, 4843, 349, 264, 2818, 297, 272, 2607, 28725, 304, 378, 349, 2278, 298, 776,
    684, 456, 2758, 302, 12707, 28723,
];

/// Long enough to push `forward_prefill` onto the flash-attention branch.
fn flash_prompt() -> Vec<u32> {
    PROMPT.iter().copied().cycle().take(288).collect()
}

/// Fail unless the model really is a K-quant mix — i.e. unless this test is
/// exercising the kernels it claims to.
///
/// Without this the test is trivially vacuous: `CERA_LFM2_MODEL` overrides the path
/// for *any* requested model, so pointing it at a Q8_0 file yields a confident green
/// "LFM2.5-230M-Q4_K_M … cosine=1.000000" having never touched a Q4_K or Q6_K tensor.
/// A test that cannot tell you whether it ran the code under test is not a test.
fn assert_is_k_quant(path: &std::path::Path) {
    let gguf = cera::gguf::GgufFile::open(path).unwrap();
    let mut q4k = 0usize;
    let mut q6k = 0usize;
    for info in gguf.tensors.values() {
        match info.dtype {
            cera::tensor::DType::Q4KM => q4k += 1,
            cera::tensor::DType::Q6K => q6k += 1,
            _ => {}
        }
    }
    eprintln!("[parity]   dtype census: {q4k} Q4_K, {q6k} Q6_K tensors");
    assert!(
        q4k > 0 && q6k > 0,
        "{} is not a Q4_K_M mix ({q4k} Q4_K, {q6k} Q6_K) — this test would pass \
         without ever running the K-quant GEMMs it exists to cover",
        path.display()
    );
}

fn run_parity(rel: &str, tokens: &[u32]) -> Option<(f32, usize, usize)> {
    #[allow(unused_imports)]
    use cera::model::Model;

    let path = find_model(rel)?;
    // Log the RESOLVED path, not `rel`: with CERA_LFM2_MODEL set they can differ, and
    // reporting the one you asked for rather than the one you ran is how a green test
    // lies about what it covered.
    eprintln!("[parity] loading {} ({} tok)", path.display(), tokens.len());
    assert_is_k_quant(&path);

    // (a) Sequential per-token.
    let gguf_seq = cera::gguf::GgufFile::open(&path).unwrap();
    let model_seq = cera::model::load_model(gguf_seq, None, 8192).unwrap();
    let mut state_seq = cera::kv_cache::InferenceState::from_config(model_seq.config()).unwrap();
    let mut logits_seq = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        logits_seq = model_seq.forward(&[tok], i, &mut state_seq);
    }

    // (b) Batched prefill.
    let gguf_pre = cera::gguf::GgufFile::open(&path).unwrap();
    let model_pre = cera::model::load_model(gguf_pre, None, 8192).unwrap();
    let mut state_pre = cera::kv_cache::InferenceState::from_config(model_pre.config()).unwrap();
    let logits_pre = model_pre.forward_prefill(tokens, 0, &mut state_pre);

    assert_eq!(logits_pre.len(), logits_seq.len(), "logit length mismatch");
    Some((
        cosine(&logits_pre, &logits_seq),
        argmax(&logits_pre),
        argmax(&logits_seq),
    ))
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

fn check(rel: &str, tokens: &[u32]) {
    // Before `run_parity`, not inside it: `None` from there means "fixture
    // absent" and is what trips the `CERA_REQUIRE_MODEL` assertion below.
    // Folding the liveness skip into that same `None` made a present-but
    // -unusable fixture report as missing — the wrong reason, and under
    // CERA_REQUIRE_MODEL the wrong failure.
    if !batched_path_is_live(rel) {
        return;
    }
    let Some((cos, top_pre, top_seq)) = run_parity(rel, tokens) else {
        // Absent fixture normally skips — but a skip that reports PASS is how a gate
        // goes green forever without ever running. `CERA_REQUIRE_MODEL` makes the
        // absence a hard failure, so a CI job that is supposed to have the fixture
        // cannot quietly stop testing. Mirrors `CERA_REQUIRE_SIMD` in `simd.rs`.
        assert!(
            std::env::var("CERA_REQUIRE_MODEL").is_err(),
            "CERA_REQUIRE_MODEL is set but the fixture is absent: {rel} \
             (set CERA_LFM2_MODEL or CERA_MODEL_ROOT)"
        );
        eprintln!("[parity] SKIP (absent): {rel}");
        return;
    };
    let is_flash = tokens.len() >= 256;
    let path = if is_flash { "flash" } else { "naive" };
    eprintln!("[parity] {rel} [{path}]: cosine={cos:.6} argmax pre={top_pre} seq={top_seq}");

    // Thresholds, calibrated against a Q4_0 control on *untouched* code:
    //  - naive NEON: cosine 1.000000 exactly — the batched GEMM runs the same int8
    //    arithmetic as the per-token GEMV, so anything below 0.9999 is a real kernel
    //    bug. (This bar caught exactly that: a Q6_K accumulation-order difference
    //    worth 3.4e-4 at k=4608.)
    //  - naive x86: the Q4_K projections take the repacked interleave prefill GEMM,
    //    a legitimate reduction difference from the GEMV (see the module doc),
    //    landing ~0.9996 — so relax to the 0.99 the codebase uses for flash/BLAS.
    //    argmax (asserted below) stays the discriminating check, and the kernels
    //    carry their own tight (1e-4) equivalence unit tests.
    //  - flash NEON: the Q4_0 control scores 0.998816 here on code this PR never
    //    touches — flash attention reorders reductions, so the bar must be looser or
    //    it fails for reasons unrelated to the GEMM.
    //  - BLAS: f32 SGEMM vs int8 dot, a legitimate reduction difference.
    let min_cos = if cfg!(feature = "blas") {
        0.995
    } else if is_flash {
        0.998
    } else if cfg!(target_arch = "x86_64") {
        0.99
    } else {
        0.9999
    };
    assert!(
        cos >= min_cos,
        "{rel} [{path}]: batched prefill diverges from the per-token path \
         (cosine {cos:.6} < {min_cos}) — the batched GEMM is wrong, not merely noisy"
    );
    assert_eq!(
        top_pre, top_seq,
        "{rel} [{path}]: batched prefill picks a different top-1 token than the \
         per-token path ({top_pre} vs {top_seq})"
    );
}

/// The whole point: a Q4_K_M LFM2 file mixes Q4_K and Q6_K, so this exercises
/// both new K-quant GEMM kernels and their gates.
#[test]
#[ignore = "needs a real GGUF; run with --ignored"]
fn lfm2_q4km_batched_prefill_matches_sequential() {
    check("target/oracle/models/LFM2.5-230M-Q4_K_M.gguf", PROMPT);
}

#[test]
#[ignore = "needs a real GGUF; run with --ignored"]
fn lfm2_q4km_batched_prefill_flash_matches_sequential() {
    check(
        "target/oracle/models/LFM2.5-230M-Q4_K_M.gguf",
        &flash_prompt(),
    );
}
