//! Benchmark: cost of `forward_prefill_logits_all` as a function of the
//! verification batch size `n`, which is what speculative decoding pays per
//! round (`n = 1 + k` for a `k`-token draft).
//!
//! The point of the measurement is the LM head. `forward_prefill_logits_all`
//! captures every position's hidden state in ONE batched pass, then projects
//! them to logits. `project_logits_batched` does that in a single GEMM, so the
//! `hidden_size x vocab` output matrix — the largest tensor in the model — is
//! read once per round; the per-row fallback it replaced re-streamed it once per
//! position. Speculative decoding exists to amortize a single weight read across
//! `1 + k` tokens, so a per-position term that scales with the largest weight
//! works directly against it.
//!
//! **How to read the output.** `marginal` is the cost of one extra verified
//! position, fitted across the `n > 1` rows only. The `n = 1` row is printed for
//! scale but excluded from the fit and marked `*`: `forward_prefill_logits_all`
//! only takes the batched branch when `n > 1`, so that row times the sequential
//! per-token `forward` instead — differencing against it would fold the gap
//! between two code paths into a number meant to describe one.
//!
//! This benchmark cannot tell you on its own whether the LM head is amortized;
//! a marginal cost is not zero-referenced. Use it as an **A/B**, which needs no
//! code edit: `CERA_LM_HEAD_NO_GEMM=1` forces the per-row projection the GEMM
//! replaced, so both halves are runnable from the same binary. Run them
//! back-to-back on the same machine in the same thermal state, and check the
//! min/max columns before believing the medians.
//!
//! ```text
//! cargo test -p cera --release --test spec_lm_head_bench -- --ignored --nocapture
//! CERA_LM_HEAD_NO_GEMM=1 cargo test -p cera --release --test spec_lm_head_bench \
//!     -- --ignored --nocapture
//! ```
//!
//! Measured result (Llama-3.2-1B-Q4_0, M1 Max, three interleaved rounds,
//! comparing minima): `n = 7` — the default `k = 6` — 57.2 ms -> 42.3 ms, about
//! 26% off a verification round. Least-squares over those minima puts the
//! marginal at 7.09 -> 5.05 ms per position. The `marginal` this benchmark
//! prints fits the *medians* instead, so it runs a little higher; medians carry
//! the background load the minima exclude.
//!
//! Interleave the two halves rather than running all of one then all of the
//! other, and prefer the min column: a first round on a cold binary and any
//! background load both inflate medians here, and an earlier attempt at this
//! measurement produced a 12%-too-slow "before" number by timing the two halves
//! in separate builds minutes apart.
//!
//! `#[ignore]` because it needs a real dense GGUF. Fixture resolution is shared
//! with `spec_decode_logits.rs` via `common::dense_model_or_skip` — deliberately,
//! so that suite's coverage guard ("this model reaches the batched projection")
//! describes the same model this benchmark times. `CERA_REQUIRE_DENSE_MODEL=1`
//! turns a missing fixture into a hard failure rather than a silent skip.
//!

mod common;

use std::time::Instant;

use common::WarnCapture;
use tracing_subscriber::layer::SubscriberExt;

/// Context depth the verification batch is measured at. Attention cost grows
/// with this, the LM-head cost does not — a realistic but modest depth keeps the
/// measurement dominated by the term under study without being unrepresentative.
const BASE_CTX: usize = 128;
/// Timed repetitions per `n`, after a discarded warmup round.
const REPS: usize = 9;

/// Median of a set of timings, in milliseconds. Median rather than mean because
/// a single scheduler hiccup on a laptop skews the mean and there is no reason
/// to let it: the distribution's centre is what we are comparing.
fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "needs a real dense GGUF; set CERA_DENSE_MODEL"]
fn lm_head_cost_vs_verification_batch() {
    let Some(path) = common::dense_model_or_skip() else {
        return;
    };
    println!("model: {}", path.display());

    // Say, in-process, which projection the table below times. A head whose
    // dtype has no GEMM kernel (Q5_K, F16), or an x86 host below the AVX2 tier,
    // produces a complete and plausible timing table for the *fallback* — the
    // one result this benchmark must never be mistaken for.
    // `dense_gemm_head_fixture` fails the run rather than let that happen; the
    // capture below covers what a pre-flight gate cannot see.
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(feature = "blas")
    ))]
    let (model, detail) = {
        let (model, detail) = common::dense_gemm_head_fixture(&path);
        let detail = if std::env::var("CERA_LM_HEAD_NO_GEMM").as_deref() == Ok("1") {
            format!("{detail} — CERA_LM_HEAD_NO_GEMM=1: timing the PER-ROW fallback")
        } else {
            format!("{detail} — timing the batched GEMM projection")
        };
        (model, detail)
    };
    // The batched *projection* is not compiled into this build, so every `n > 1`
    // row's logits come from the per-row loop. Under `blas` the batched
    // hidden-state prefill still runs — only the projection differs — so this
    // says exactly that rather than claiming the whole path changed.
    #[cfg(not(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(feature = "blas")
    )))]
    let (model, detail) = {
        let model = cera::model::load_model(cera::gguf::GgufFile::open(&path).unwrap(), None, 8192)
            .unwrap();
        assert!(
            model.supports_all_logits(),
            "benchmark target requires forward_prefill_logits_all support"
        );
        (
            model,
            "no batched LM-head projection in this build (blas, or no int8 GEMM \
             target) — the projection is per-row on every row"
                .to_string(),
        )
    };

    // Installed after model load, so load-time warnings unrelated to this
    // measurement (tokenizer quirks, thread-pool notices) cannot fail the run,
    // and before the warmup prefill, because `warn_unbatchable` dedupes globally
    // and would fire there first — capturing only the timed loop would miss it.
    let warns = WarnCapture::default();
    let _guard =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(warns.clone()));

    let cfg = model.config();
    let (vocab, hs) = (cfg.vocab_size, cfg.hidden_size);
    println!("hidden_size={hs} vocab={vocab} ctx_depth={BASE_CTX} reps={REPS}");
    println!("{detail}");

    // A fixed pseudo-random token context. Values only need to be in range —
    // the timing does not depend on which tokens they are, and using a fixed
    // sequence keeps runs comparable.
    let tok = |i: usize| ((i * 2654435761) % vocab.max(1)) as u32;
    let ctx: Vec<u32> = (0..BASE_CTX).map(tok).collect();

    let mut state = cera::kv_cache::InferenceState::from_config(cfg).unwrap();
    let _ = model.forward_prefill(&ctx, 0, &mut state);
    assert_eq!(state.seq_len, BASE_CTX);

    // Min and max alongside the median: the fit below rests on four numbers, and
    // without the spread a reader cannot tell a real A/B delta from run-to-run
    // noise — which is precisely how this repo's GEMV bench misread a 3x
    // position artifact as a kernel difference.
    println!(
        "\n{:>4}  {:>10}  {:>10}  {:>10}",
        "n", "min ms", "median ms", "max ms"
    );
    // `(n, ms)` for the batched rows only, for the marginal fit below.
    let mut batched: Vec<(f64, f64)> = Vec::new();
    for &n in &[1usize, 2, 4, 7, 9] {
        let batch: Vec<u32> = (0..n).map(|i| tok(BASE_CTX + i)).collect();

        // Each measurement runs at the same position: append the batch, time
        // it, then rewind the KV so the next rep starts from an identical
        // state. Without the rewind, later reps would run at a deeper context
        // and the attention term would drift upward across reps.
        let mut times = Vec::with_capacity(REPS);
        for rep in 0..REPS + 1 {
            let t0 = Instant::now();
            let all = model.forward_prefill_logits_all(&batch, BASE_CTX, &mut state);
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            // Consume the result so the projection cannot be optimized away.
            std::hint::black_box(&all);
            assert_eq!(all.len(), n * vocab);
            state.truncate_to(BASE_CTX);
            // Discard the first round: it pages in the mmap for this `n` and
            // warms the allocator, and a first-measured-is-slow artifact has
            // bitten this repo's GPU benches before.
            if rep > 0 {
                times.push(dt);
            }
        }

        let lo = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ms = median_ms(times);
        let label = if n > 1 {
            batched.push((n as f64, ms));
            format!("{n}")
        } else {
            // Marked, and kept out of the fit: this row is the per-token path.
            format!("{n}*")
        };
        println!("{label:>4}  {lo:>10.3}  {ms:>10.3}  {hi:>10.3}");
    }

    // Everything warned since the warmup began.
    let captured = warns.messages();
    // Print all of it, then fail on the subset that means the timings describe
    // something other than the banner. The one this must catch beyond an
    // LM-head decline is `warn_unbatchable`: a single unbatchable layer tensor
    // routes the *whole model* through per-token prefill while the head GEMM
    // still runs, leaving the banner accurate and the fit meaningless.
    for m in &captured {
        println!("note: warning during measurement: {m}");
    }
    let fell_back: Vec<&String> = captured
        .iter()
        .filter(|m| {
            m.contains("fell back")
                || m.contains("LM-head")
                || m.contains("was NOT computed")
                || m.contains("not supported on the batched path")
        })
        .collect();
    assert!(
        fell_back.is_empty(),
        "a fast path declined during the measurement, so this table does not \
         measure what the banner says it does: {fell_back:?}"
    );

    // Least-squares slope over the batched rows — the marginal cost of one more
    // verified position. A slope, not a difference against `n = 1`, because the
    // `n = 1` row is a different code path; and least-squares rather than a
    // two-point difference so one noisy row cannot set the whole number.
    let k = batched.len() as f64;
    let sum_x: f64 = batched.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = batched.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = batched.iter().map(|(x, y)| x * y).sum();
    let sum_xx: f64 = batched.iter().map(|(x, _)| x * x).sum();
    let slope = (k * sum_xy - sum_x * sum_y) / (k * sum_xx - sum_x * sum_x);
    println!("\n* n=1 takes the per-token path, not the batched one — excluded from the fit.");
    println!("marginal cost per verified position (fit over n>1): {slope:.3} ms");
    println!(
        "This number is only meaningful against another run of the same fit on \
         the same machine — see the module docs."
    );
}

/// `WarnCapture`'s level filter keeps WARN and ERROR and drops everything more
/// verbose.
///
/// Worth pinning because the predicate reads backwards. `tracing::Level` orders
/// by *verbosity*, so `TRACE > DEBUG > INFO > WARN > ERROR` (`tracing_core`
/// inverts the comparison in its `impl Ord`), and `level > Level::WARN`
/// therefore selects the chatty end, not the severe one. Inverting it would turn
/// the benchmark's fallback self-check into either a pass that ignores a real
/// decline warning or a failure on any stray `info!`.
///
/// This test needs no model, so unlike the benchmark beside it, it is not
/// `#[ignore]`d and does run in the default `cargo test --workspace`.
#[test]
fn warn_capture_keeps_warn_and_error_only() {
    let warns = WarnCapture::default();
    {
        let _guard =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(warns.clone()));
        tracing::error!("kept-error");
        tracing::warn!("kept-warn");
        tracing::info!("dropped-info");
        tracing::debug!("dropped-debug");
        tracing::trace!("dropped-trace");
    }

    let got = warns.messages();
    assert_eq!(
        got.len(),
        2,
        "expected exactly the ERROR and WARN events, got {got:?}"
    );
    assert!(
        got.iter().any(|m| m.contains("kept-error")),
        "ERROR was dropped, so the level comparison is inverted: {got:?}"
    );
    assert!(
        got.iter().any(|m| m.contains("kept-warn")),
        "WARN was dropped: {got:?}"
    );
}
