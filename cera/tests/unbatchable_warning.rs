//! The per-token prefill fallback must be *audible*.
//!
//! `warn_unbatchable` exists because a silent fallback costs ~4x prefill and is
//! otherwise invisible — the numbers are simply worse, with nothing to point at.
//! It emits through `tracing::warn!`, and `cargo test` installs no subscriber, so
//! the warning that guards against a silent regression was itself silent under
//! test. The LFM2 K-quant parity suite spent a release reporting a vacuous
//! `cosine=1.000000` (comparing the per-token path against itself) while this
//! warning fired into the void on every run.
//!
//! This installs a capturing subscriber and asserts the warning actually fires
//! when the batched path declines, so a future change that makes the fallback
//! silent fails here instead of in someone's throughput numbers.
//!
//! `CERA_CPU_TIER=avx512` is the lever: it caps the tier below `Avx512Vnni`, so
//! `int8_gemm_available()` is false, `batched_gemm_supports` declines every
//! dtype, and prefill must fall back. The tier is cached in a `OnceLock` and read
//! once per process, which is why this is its own test binary with a single test.

#![cfg(all(target_arch = "x86_64", not(feature = "blas")))]

use std::sync::{Arc, Mutex};

use tracing_subscriber::layer::SubscriberExt;

/// Collects the `message` field of every `WARN` event, from any target.
///
/// Debug-formatted rather than fully rendered — the assertion only needs a
/// substring match, and reaching for a real formatter here would pull in
/// machinery the test does not use. No target filter on purpose: the point is
/// to prove *something* warned about the fallback, so narrowing to
/// `cera::model::transformer` would bake this test's expectation into where the
/// warning happens to live.
#[derive(Clone, Default)]
struct WarnCapture(Arc<Mutex<Vec<String>>>);

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for WarnCapture {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if *event.metadata().level() != tracing::Level::WARN {
            return;
        }
        // The message is a field, not something `Event` exposes directly; a
        // visitor is the only way to read it back out.
        struct Msg<'a>(&'a mut String);
        impl tracing::field::Visit for Msg<'_> {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0.push_str(&format!("{value:?}"));
                }
            }
        }
        let mut msg = String::new();
        event.record(&mut Msg(&mut msg));
        if !msg.is_empty() {
            self.0.lock().unwrap_or_else(|p| p.into_inner()).push(msg);
        }
    }
}

/// `#[ignore]` like the other fixture-backed tests: the mainline
/// `cargo test --workspace` job has no GGUFs, and a test that silently skips
/// there would report the same green as one that ran. The parity job fetches
/// fixtures and runs this explicitly with `--ignored`.
#[test]
#[ignore = "needs a GGUF fixture; run with --ignored"]
fn per_token_fallback_emits_a_warning() {
    // SAFETY: single-threaded, first thing in the process to touch the
    // environment, and set before any `cpu_features()` call — the tier is
    // cached, so this binary holds exactly one test.
    unsafe {
        std::env::set_var("CERA_CPU_TIER", "avx512");
    }

    let tier = cera::backend::cpu_features::cpu_features().tier;
    assert!(
        tier < cera::backend::cpu_features::CpuTier::Avx512Vnni,
        "CERA_CPU_TIER=avx512 did not downgrade the tier (got {tier:?}); without \
         the downgrade the batched path would still run and this test would be \
         asserting nothing"
    );
    assert!(
        !cera::backend::cpu::int8_gemm_available(),
        "int8 GEMM still reports available at tier {tier:?} — the fallback this \
         test needs would not trigger"
    );

    let capture = WarnCapture::default();
    let subscriber = tracing_subscriber::registry().with(capture.clone());
    let _guard = tracing::subscriber::set_default(subscriber);

    // Drive a real fallback. `warn_unbatchable` is `pub(crate)`, so this goes
    // through the public surface that a user would hit: a Q4_0 model whose
    // batched path declines for lack of an int8 kernel.
    let Some(path) = find_fixture("target/oracle/models/SmolLM-135M.Q4_0.gguf") else {
        // A skip that reports PASS is how a gate goes green forever without
        // running — the failure mode this whole test exists to prevent. CI sets
        // CERA_REQUIRE_MODEL on the leg that fetches fixtures, so an absent one
        // there is a hard failure rather than a quiet no-op. Mirrors the parity
        // suites.
        assert!(
            std::env::var("CERA_REQUIRE_MODEL").is_err(),
            "CERA_REQUIRE_MODEL is set but the fixture is absent: \
             target/oracle/models/SmolLM-135M.Q4_0.gguf (run \
             scripts/fetch_test_models.sh, or set CERA_MODEL_ROOT)"
        );
        eprintln!("[warn-test] SKIP: fixture absent (scripts/fetch_test_models.sh)");
        return;
    };
    let gguf = cera::gguf::GgufFile::open(&path).expect("open fixture");
    let model = cera::model::load_model(gguf, None, 2048).expect("load fixture");
    let mut state =
        cera::kv_cache::InferenceState::from_config(model.config()).expect("inference state");
    // >1 token, or the batched path is never even considered.
    let _ = model.forward_prefill(&[1, 415, 2323, 302, 4843, 349, 264, 2818], 0, &mut state);

    let warnings = capture.0.lock().unwrap_or_else(|p| p.into_inner()).clone();
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("fell back to the per-token path")),
        "prefill fell back but emitted no warning — a silent ~4x regression. \
         Captured warnings: {warnings:?}"
    );
}

/// Mirrors the parity suites' fixture resolution: crate dir's parent, cwd, then
/// `CERA_MODEL_ROOT`.
fn find_fixture(rel: &str) -> Option<std::path::PathBuf> {
    let mut roots: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        if let Some(parent) = std::path::PathBuf::from(&manifest).parent() {
            roots.push(parent.to_path_buf());
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    if let Ok(root) = std::env::var("CERA_MODEL_ROOT") {
        roots.push(std::path::PathBuf::from(root));
    }
    roots.into_iter().map(|r| r.join(rel)).find(|p| p.exists())
}
