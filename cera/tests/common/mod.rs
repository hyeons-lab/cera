//! Shared helpers for integration tests.
//!
//! Lives under `tests/common/` (not `tests/helpers/`) to follow Cargo's
//! convention for non-test files inside the integration-test directory
//! — Cargo skips the `common/` subdir when picking up test binaries.
//!
//! [`download::ensure_cached`] serves tests that need a real GGUF but don't
//! want to bake multi-hundred-MB fixtures into the repo. The download helper
//! compiles only when the `remote` feature is active; callers are
//! `#[cfg(feature = "remote")]`'d accordingly.
//!
//! [`metal_context`] is the shared skip-vs-fail gate used by every Metal
//! suite that dispatches a kernel.
//!
//! [`dense_model_or_skip`] resolves the dense GGUF the speculative-decoding
//! suite and LM-head benchmark share, and [`dense_gemm_head_fixture`] loads it
//! while asserting it actually reaches the batched LM-head projection.
//!
//! [`hidden_states_with_lora`] runs a prompt with an optional adapter attached,
//! shared by the two LoRA suites (`lora_parity`, `moe_lora_parity`).
//!
//! [`write_lora_gguf`] serializes an adapter fixture, shared by
//! `moe_lora_parity`, `metal_moe_oracle` and `wgpu_moe_oracle`.

#![allow(dead_code)]

#[cfg(feature = "remote")]
pub mod download;

/// Per-token hidden states for `tokens`, with an optional LoRA adapter attached
/// to the inference state.
///
/// Defined here rather than per test binary for the reason `metal_context`
/// gives: a copy per module is how the copies drift. Both LoRA suites
/// (`lora_parity`, `moe_lora_parity`) compare a base run against an adapted
/// one, and the comparison is only meaningful if both arms are built the same
/// way.
pub fn hidden_states_with_lora(
    model: &dyn cera::model::Model,
    tokens: &[u32],
    lora: Option<std::sync::Arc<cera::lora::LoraAdapterWeights>>,
) -> Vec<f32> {
    let mut state =
        cera::kv_cache::InferenceState::for_prefill(model.config(), tokens.len()).unwrap();
    state.lora = lora;
    model.hidden_states(tokens, &mut state)
}

/// Acquire a Metal device, or skip the calling test.
///
/// A host without a Metal device skips silently; with `CERA_REQUIRE_METAL=1` the
/// missing device is a hard failure instead, so the CI leg that targets
/// known-capable hardware proves the kernels actually executed rather than
/// reporting green on an empty test set. Same skip-vs-fail convention as
/// `CERA_REQUIRE_SIMD` (`require_simd_or_skip` in `backend/simd.rs`).
///
/// Defined once here rather than per test binary, for the reason that helper
/// states: a copy per module is how one ends up with a weaker gate than the CI
/// leg targeting it assumes. It cannot come from `simd.rs` itself — an
/// integration test is a separate crate and cannot name a `#[cfg(test)]` item in
/// the library — but `tests/common/` is shared across test binaries.
///
/// Callers: `metal_shaders_parity`, `metal_turboquant_oracle`,
/// `metal_kv_shift_oracle`. `metal_params_layout` deliberately does not use it —
/// it compares struct layouts and needs no device.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub fn metal_context() -> Option<cera::backend::metal::MetalContext> {
    match cera::backend::metal::MetalContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            assert!(
                std::env::var("CERA_REQUIRE_METAL").as_deref() != Ok("1"),
                "CERA_REQUIRE_METAL=1 but no Metal device is available ({e})"
            );
            eprintln!("skipping: no Metal device ({e})");
            None
        }
    }
}

/// Resolve a dense (llama/qwen2/qwen3) GGUF for the speculative-decoding tests
/// and benchmark, or skip the calling test.
///
/// Resolve order:
///   1. `CERA_DENSE_MODEL` — absolute path to a dense gguf
///   2. `~/.leap/models/Llama-3.2-1B-Instruct-Q8_0/…`
///   3. `<repo>/models/Llama-3.2-1B-Q4_0.gguf`
///
/// Step 2 comes first because it was the pre-existing default for this suite,
/// and the order decides which GEMM kernels get exercised — demoting it would
/// silently re-point six existing tests at a different quant.
///
/// **Both defaults miss on a typical checkout**, so in practice pass
/// `CERA_DENSE_MODEL`: step 3 is crate-relative, and `models/` is gitignored and
/// lives only in the main checkout, while this repo's workflow mandates working
/// in a git worktree. A path pinned there that does not exist is a hard error,
/// not a fallback. `CERA_REQUIRE_DENSE_MODEL=1` likewise turns a missing fixture
/// into a failure rather than a skip — same convention as `metal_context` above,
/// for the same reason: a suite that skips reports green on an empty test set.
///
/// Shared rather than copied per binary because the spec suite and the LM-head
/// benchmark must resolve the *same* fixture, or the suite's coverage guard says
/// nothing about what the benchmark timed.
///
/// Callers: `spec_decode_logits`, `spec_lm_head_bench`.
pub fn dense_model_or_skip() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;

    if let Ok(p) = std::env::var("CERA_DENSE_MODEL") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
        // Hard failure, not a fallback: an explicitly pinned path is stronger
        // evidence of intent than an unset variable, so a typo must not resolve
        // silently to a default fixture at a different quant and report a number
        // for a model the operator did not choose.
        panic!(
            "CERA_DENSE_MODEL={} does not exist (unset it to use the defaults)",
            p.display()
        );
    }

    let mut tried: Vec<PathBuf> = Vec::new();

    if let Ok(home) = std::env::var("HOME") {
        let name = "Llama-3.2-1B-Instruct-Q8_0";
        let leap = PathBuf::from(home)
            .join(".leap/models")
            .join(name)
            .join(format!("{name}.gguf"));
        if leap.exists() {
            return Some(leap);
        }
        tried.push(leap);
    }

    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/Llama-3.2-1B-Q4_0.gguf");
    if repo.exists() {
        return Some(repo);
    }
    tried.push(repo);

    let tried = tried
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    assert!(
        std::env::var("CERA_REQUIRE_DENSE_MODEL").as_deref() != Ok("1"),
        "CERA_REQUIRE_DENSE_MODEL=1 but no dense model found (CERA_DENSE_MODEL unset; tried {tried})"
    );
    eprintln!("skipping: no dense model (CERA_DENSE_MODEL unset; tried {tried})");
    None
}

/// Load `path` and assert it is a dense model that actually exercises the
/// batched LM-head projection, returning the model and a description of the
/// head for the caller to print.
///
/// Both the coverage guard and the benchmark need exactly this preamble, and it
/// is the *pairing* — dense model AND GEMM-able head — that makes either of them
/// meaningful: the head check alone would approve an LFM2 GGUF, which has a
/// `token_embd.weight` but never reaches this code, and the dense check alone
/// would approve a Q5_K head that silently takes the per-row path.
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(has_blas)))]
pub fn dense_gemm_head_fixture(path: &std::path::Path) -> (Box<dyn cera::model::Model>, String) {
    let gguf = cera::gguf::GgufFile::open(path).unwrap();
    let model =
        cera::model::load_model(cera::gguf::GgufFile::open(path).unwrap(), None, 8192).unwrap();
    assert!(
        model.supports_all_logits(),
        "fixture is not a dense model with all-position logits, so it does not \
         exercise the batched LM-head projection at all"
    );
    let hs = model.config().hidden_size;
    let (reached, detail) = lm_head_gate(&gguf, hs);
    assert!(
        reached,
        "{detail} — the batched LM-head projection would decline on this \
         fixture, so anything measured against it is the per-row fallback"
    );
    (model, detail)
}

/// Mirror of `LlamaModel::project_logits_batched`'s dtype gate: `(reached,
/// detail)`, where `detail` names the failing condition or the head that will be
/// used.
///
/// Does not mirror the projection's `m >= vocab_size`, `k == hidden_size`, or
/// `hidden_size % 32` checks — those are rejected at model load, so a fixture
/// violating any of them fails loudly rather than quietly measuring the
/// fallback. Nor does it consult `CERA_LM_HEAD_NO_GEMM`: that lever is the
/// operator deliberately selecting the per-row path, and the benchmark reports
/// it separately.
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(has_blas)))]
fn lm_head_gate(gguf: &cera::gguf::GgufFile, hidden_size: usize) -> (bool, String) {
    let Some(head) = gguf
        .tensors
        .get("output.weight")
        .or_else(|| gguf.tensors.get("token_embd.weight"))
    else {
        return (
            false,
            "model has neither output.weight nor token_embd.weight".into(),
        );
    };
    if !cera::model::transformer::batched_gemm_supports(head.dtype, hidden_size) {
        return (
            false,
            format!(
                "LM head `{}` is {:?} (hs={hidden_size}) — no batched GEMM kernel here",
                head.name, head.dtype
            ),
        );
    }
    (true, format!("LM head `{}` ({:?})", head.name, head.dtype))
}

/// Collects the message of every `WARN`/`ERROR` event, from any target.
///
/// `cargo test` installs no subscriber, so warnings a test needs to observe —
/// the fallback notices this codebase emits instead of failing — otherwise go
/// nowhere. Install with:
///
/// ```ignore
/// let warns = WarnCapture::default();
/// let _guard = tracing::subscriber::set_default(
///     tracing_subscriber::registry().with(warns.clone()));
/// ```
///
/// `set_default` is thread-local, so the events must be emitted on the calling
/// thread — true of every fallback warning in `model/` today.
///
/// No target filter on purpose: the point is to prove *something* warned, so
/// narrowing to a module would bake a test's expectation into where the warning
/// happens to live. `tests/unbatchable_warning.rs` predates this and carries its
/// own x86-only copy; new callers should use this one.
///
/// The level filter is pinned by `warn_capture_keeps_warn_and_error_only` in
/// `tests/spec_lm_head_bench.rs`, because `tracing::Level`'s ordering is by
/// verbosity and so reads backwards: `ERROR` is the *smallest* level.
#[derive(Clone, Default)]
pub struct WarnCapture(pub std::sync::Arc<std::sync::Mutex<Vec<String>>>);

impl WarnCapture {
    /// Everything captured so far. A poisoned mutex yields the contents anyway
    /// rather than an empty list — a capture that reads as "nothing warned"
    /// after a panic would turn a self-check into a silent pass.
    pub fn messages(&self) -> Vec<String> {
        match self.0.lock() {
            Ok(g) => g.clone(),
            Err(p) => p.into_inner().clone(),
        }
    }
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for WarnCapture {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if *event.metadata().level() > tracing::Level::WARN {
            return;
        }
        struct Visit(Option<String>);
        impl tracing::field::Visit for Visit {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0 = Some(format!("{value:?}"));
                }
            }
        }
        let mut v = Visit(None);
        event.record(&mut v);
        // Events with no `message` field still matter — record the target so a
        // caller printing the capture does not show a bare empty line.
        let msg =
            v.0.unwrap_or_else(|| format!("<no message> target={}", event.metadata().target()));
        // Recover from poisoning on the write side too: a dropped message
        // would leave the caller's self-check reading as "nothing warned".
        match self.0.lock() {
            Ok(mut g) => g.push(msg),
            Err(p) => p.into_inner().push(msg),
        }
    }
}

/// Append a GGUF length-prefixed string.
fn push_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Serialize a minimal GGUF v3 adapter of F32 tensors. Each entry is
/// `(name, ne, data)` with `ne` fastest-varying first, GGUF's own order.
///
/// Shared rather than per-suite: `moe_lora_parity` builds stacked expert
/// adapters with it, and both GPU oracles build two apiece, a router-only
/// adapter and a router-plus-`attn_output` one. Two copies of a byte-layout
/// writer is how those suites end up disagreeing about the format they are all
/// asserting against.
pub fn write_lora_gguf(tensors: &[(String, Vec<usize>, Vec<f32>)], alpha: f32) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes());
    push_str(&mut out, "adapter.lora.alpha");
    out.extend_from_slice(&6u32.to_le_bytes()); // GGUF_TYPE_FLOAT32
    out.extend_from_slice(&alpha.to_le_bytes());

    let mut offset = 0u64;
    for (name, ne, data) in tensors {
        push_str(&mut out, name);
        out.extend_from_slice(&(ne.len() as u32).to_le_bytes());
        ne.iter()
            .for_each(|&d| out.extend_from_slice(&(d as u64).to_le_bytes()));
        out.extend_from_slice(&0u32.to_le_bytes()); // GGML_TYPE_F32
        out.extend_from_slice(&offset.to_le_bytes());
        offset += (data.len() * 4) as u64;
    }
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    for (_, _, data) in tensors {
        out.extend(data.iter().flat_map(|x| x.to_le_bytes()));
    }
    out
}
