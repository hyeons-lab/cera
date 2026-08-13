//! CPU mixture-of-experts LoRA wiring parity.
//!
//! The loader-side unit tests in `cera::lora` pin how a stacked adapter is
//! *split*; nothing there pins that the forward pass then *consumes* the pieces
//! correctly. This file covers that gap on a real `lfm2moe` model: that each of
//! the four routed targets has a live hook, that the gate and up hooks are not
//! the same lookup written twice, and that the per-expert factors are actually
//! indexed by the routed expert rather than pinned to expert 0.
//!
//! What it deliberately does not claim: these are self-consistency checks, not
//! a llama.cpp golden. Numeric agreement was established separately, by hand,
//! feeding both engines the identical token stream `[124894, 597, 5205, 302,
//! 3980, 355]` with a synthetic rank-4 adapter on the first routed layer: at
//! `alpha = 8` the two produce identical greedy text for 40 tokens, and at
//! `alpha = 64` (where the model emits noise, so the argmax is near-random and
//! agreement cannot be luck) they produce the same 25 tokens before rounding
//! drift separates them. llama.cpp must be pinned to CPU (`-ngl 0`) for that
//! comparison: its Metal backend diverges from its own CPU backend on this
//! model at the first generated token.
//!
//! Gated behind `CERA_MOE_LORA_PARITY=1` + an `lfm2moe` GGUF via
//! `CERA_LFM2MOE_MODEL` (CPU only). Run:
//!   CERA_MOE_LORA_PARITY=1 CERA_LFM2MOE_MODEL=... cargo test -p cera --release \
//!     --test moe_lora_parity -- --ignored --nocapture

use std::path::PathBuf;
use std::sync::Arc;

mod common;

use common::hidden_states_with_lora as run;
use common::write_lora_gguf as write_gguf;

use cera::gguf::GgufFile;
use cera::kv_cache::InferenceState;
use cera::lora::LoraAdapterWeights;
use cera::model::load_model;

fn model_path() -> Option<PathBuf> {
    let p = std::env::var("CERA_LFM2MOE_MODEL")
        .ok()
        .map(PathBuf::from)?;
    (p.exists() && GgufFile::open(&p).is_ok()).then_some(p)
}

const RANK: usize = 4;
const ALPHA: f32 = 8.0;

/// A stacked per-expert adapter on one target of one layer. `b_for_expert`
/// returns the base fill for expert `e`'s `B`, which is how a test makes the
/// experts distinguishable from each other.
///
/// Neither factor is filled with a single constant. `A` varies along its input
/// dimension and `B` along its output dimension, so that a wrong row stride, a
/// transposed factor, or an output width truncated to the dense
/// `intermediate_size` all change the result. A constant fill makes every one of
/// those produce exactly the numbers a correct implementation produces.
fn expert_adapter(
    layer: usize,
    stem: &str,
    n_expert: usize,
    k: usize,
    d: usize,
    a_fill: f32,
    b_for_expert: impl Fn(usize) -> f32,
) -> Arc<LoraAdapterWeights> {
    // A is ne [k, rank, n_expert], so within an expert the layout is
    // `rank` rows of `k`; B is ne [rank, d, n_expert], i.e. `d` rows of `rank`.
    let a: Vec<f32> = (0..n_expert)
        .flat_map(|_| {
            (0..RANK).flat_map(move |r| (0..k).map(move |i| a_fill * (1.0 + (r + i) as f32 * 0.01)))
        })
        .collect();
    let b: Vec<f32> = (0..n_expert)
        .flat_map(|e| {
            let base = b_for_expert(e);
            (0..d).flat_map(move |o| (0..RANK).map(move |r| base * (1.0 + (o + r) as f32 * 0.01)))
        })
        .collect();
    let buf = write_gguf(
        &[
            (
                format!("blk.{layer}.{stem}.weight.lora_a"),
                vec![k, RANK, n_expert],
                a,
            ),
            (
                format!("blk.{layer}.{stem}.weight.lora_b"),
                vec![RANK, d, n_expert],
                b,
            ),
        ],
        ALPHA,
    );
    LoraAdapterWeights::from_gguf_bytes(Arc::from(buf.into_boxed_slice()))
        .expect("load synthetic expert adapter")
}

/// An ordinary 2-D adapter on the router (`ffn_gate_inp`), `hidden -> n_expert`.
fn router_adapter(
    layer: usize,
    k: usize,
    d: usize,
    a_fill: f32,
    b_fill: f32,
) -> Arc<LoraAdapterWeights> {
    let buf = write_gguf(
        &[
            (
                format!("blk.{layer}.ffn_gate_inp.weight.lora_a"),
                vec![k, RANK],
                (0..RANK)
                    .flat_map(|r| (0..k).map(move |i| a_fill * (1.0 + (r + i) as f32 * 0.01)))
                    .collect(),
            ),
            (
                format!("blk.{layer}.ffn_gate_inp.weight.lora_b"),
                vec![RANK, d],
                (0..d)
                    .flat_map(|o| (0..RANK).map(move |r| b_fill * (1.0 + (o + r) as f32 * 0.01)))
                    .collect(),
            ),
        ],
        ALPHA,
    );
    LoraAdapterWeights::from_gguf_bytes(Arc::from(buf.into_boxed_slice()))
        .expect("load synthetic router adapter")
}

#[test]
#[ignore = "needs an lfm2moe GGUF; gated on CERA_MOE_LORA_PARITY"]
fn moe_lora_hooks_are_live_and_distinct() {
    if std::env::var("CERA_MOE_LORA_PARITY").as_deref() != Ok("1") {
        eprintln!("skip: set CERA_MOE_LORA_PARITY=1 to run");
        return;
    }
    let Some(path) = model_path() else {
        eprintln!("skip: no lfm2moe model (set CERA_LFM2MOE_MODEL)");
        return;
    };

    let model = load_model(GgufFile::open(&path).expect("open"), None, 512).expect("cpu load");
    assert!(model.supports_hidden_states());
    let cfg = model.config();
    let moe = cfg.moe.as_ref().expect("model is mixture-of-experts");
    let (hs, ff, n_expert) = (cfg.hidden_size, moe.expert_ff_len, moe.n_expert);
    // The first routed layer. Taken from the config rather than hardcoded: the
    // dense/MoE split is a property of the file, not of the architecture.
    let layer = moe
        .is_moe_layer
        .iter()
        .position(|&m| m)
        .expect("model has a routed layer");

    // Several tokens, so the 4-of-32 routing touches more than one expert.
    let tokens: Vec<u32> = vec![1, 5, 9, 42, 100, 7, 2048, 33];
    let base = run(&*model, &tokens, None);
    assert_eq!(base.len(), tokens.len() * hs);

    // (a) B = 0 makes the delta identically zero, so this pins exactly one
    // thing: that the apply *accumulates* into the projection output rather
    // than overwriting it. It cannot catch a hook wired to the wrong buffer,
    // because adding 0.0 to the wrong buffer is invisible too. (`assert_eq!` on
    // f32 also treats -0.0 and +0.0 as equal, which is the one difference
    // `+= 0.0` could introduce, so "identical" here is up to signed zero.)
    let zero = expert_adapter(layer, "ffn_gate_exps", n_expert, hs, ff, 0.3, |_| 0.0);
    assert_eq!(
        run(&*model, &tokens, Some(zero)),
        base,
        "a zero-delta expert adapter must not change the output: the apply is \
         overwriting the projection instead of accumulating into it"
    );

    // (b) Each target has its own live hook. `up` is the load-bearing one: if
    // the gate and up hooks were the same lookup written twice, an up-only
    // adapter would apply nothing and this would equal `base`.
    let gate = run(
        &*model,
        &tokens,
        Some(expert_adapter(
            layer,
            "ffn_gate_exps",
            n_expert,
            hs,
            ff,
            0.05,
            |_| 0.05,
        )),
    );
    let up = run(
        &*model,
        &tokens,
        Some(expert_adapter(
            layer,
            "ffn_up_exps",
            n_expert,
            hs,
            ff,
            0.05,
            |_| 0.05,
        )),
    );
    let down = run(
        &*model,
        &tokens,
        Some(expert_adapter(
            layer,
            "ffn_down_exps",
            n_expert,
            ff,
            hs,
            0.05,
            |_| 0.05,
        )),
    );
    let router = run(
        &*model,
        &tokens,
        Some(router_adapter(layer, hs, n_expert, 0.05, 0.05)),
    );

    assert_ne!(gate, base, "ffn_gate_exps adapter had no effect");
    assert_ne!(up, base, "ffn_up_exps adapter had no effect");
    assert_ne!(down, base, "ffn_down_exps adapter had no effect");
    assert_ne!(router, base, "ffn_gate_inp (router) adapter had no effect");
    // gate and up feed `silu(gate) * up`, so the same delta on each lands
    // differently. Equality here means both hooks read the same target.
    assert_ne!(gate, up, "gate and up hooks are the same lookup");

    eprintln!("[moe_lora] four hooks live and distinct on layer {layer} ✓");
}

#[test]
#[ignore = "needs an lfm2moe GGUF; gated on CERA_MOE_LORA_PARITY"]
fn expert_factors_are_indexed_by_the_routed_expert() {
    if std::env::var("CERA_MOE_LORA_PARITY").as_deref() != Ok("1") {
        eprintln!("skip: set CERA_MOE_LORA_PARITY=1 to run");
        return;
    }
    let Some(path) = model_path() else {
        eprintln!("skip: no lfm2moe model (set CERA_LFM2MOE_MODEL)");
        return;
    };

    let model = load_model(GgufFile::open(&path).expect("open"), None, 512).expect("cpu load");
    let cfg = model.config();
    let moe = cfg.moe.as_ref().expect("model is mixture-of-experts");
    let (hs, ff, n_expert) = (cfg.hidden_size, moe.expert_ff_len, moe.n_expert);
    let layer = moe
        .is_moe_layer
        .iter()
        .position(|&m| m)
        .expect("model has a routed layer");
    let tokens: Vec<u32> = vec![1, 5, 9, 42, 100, 7, 2048, 33];

    // Two adapters that agree on expert 0 and disagree everywhere else. If the
    // forward pass ignored the routed expert id and always read slice 0, these
    // would produce identical hidden states. They differ only because some
    // token routes to an expert other than 0, which 8 tokens x 4-of-32 makes
    // overwhelmingly likely.
    let varying = expert_adapter(layer, "ffn_gate_exps", n_expert, hs, ff, 0.05, |e| {
        0.05 + e as f32 * 0.01
    });
    let flat = expert_adapter(layer, "ffn_gate_exps", n_expert, hs, ff, 0.05, |_| 0.05);

    let a = run(&*model, &tokens, Some(varying));
    let b = run(&*model, &tokens, Some(flat));
    assert_ne!(
        a, b,
        "per-expert factors are not indexed by the routed expert (expert 0's \
         slice appears to be used for every expert)"
    );

    eprintln!("[moe_lora] per-expert indexing is live ✓");
}

/// Batched prefill is a separate code path from per-token decode, and it is the
/// one every real prompt takes. `prefill_moe_ffn` routes token by token through
/// `forward_moe_ffn`, so the adapter should land identically in both, but that
/// is an assumption about the wiring rather than a consequence of it: the
/// prefill branch could route through the base weights and nothing above would
/// notice, because both other tests drive decode.
#[test]
#[ignore = "needs an lfm2moe GGUF; gated on CERA_MOE_LORA_PARITY"]
fn prefill_applies_the_same_expert_deltas_as_decode() {
    if std::env::var("CERA_MOE_LORA_PARITY").as_deref() != Ok("1") {
        eprintln!("skip: set CERA_MOE_LORA_PARITY=1 to run");
        return;
    }
    let Some(path) = model_path() else {
        eprintln!("skip: no lfm2moe model (set CERA_LFM2MOE_MODEL)");
        return;
    };

    let model = load_model(GgufFile::open(&path).expect("open"), None, 512).expect("cpu load");
    let cfg = model.config();
    let moe = cfg.moe.as_ref().expect("model is mixture-of-experts");
    let (hs, ff, n_expert) = (cfg.hidden_size, moe.expert_ff_len, moe.n_expert);
    let layer = moe
        .is_moe_layer
        .iter()
        .position(|&m| m)
        .expect("model has a routed layer");
    // n > 1, so the batched prefill path is taken rather than the per-token one.
    let tokens: Vec<u32> = vec![1, 5, 9, 42, 100, 7, 3, 88];

    let batched = |lora: Option<Arc<LoraAdapterWeights>>| {
        let mut st = InferenceState::for_prefill(cfg, tokens.len()).unwrap();
        st.lora = lora;
        model.forward_prefill(&tokens, 0, &mut st)
    };
    let per_token = |lora: Option<Arc<LoraAdapterWeights>>| {
        let mut st = InferenceState::for_prefill(cfg, tokens.len()).unwrap();
        st.lora = lora;
        let mut logits = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            logits = model.forward(&[tok], i, &mut st);
        }
        logits
    };
    let cosine = |a: &[f32], b: &[f32]| -> f64 {
        assert_eq!(a.len(), b.len(), "logit length mismatch");
        assert!(
            a.iter().chain(b).all(|x| x.is_finite()),
            "logits must be finite"
        );
        let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
        let na: f64 = a
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = b
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        dot / (na * nb)
    };

    let adapter = expert_adapter(layer, "ffn_gate_exps", n_expert, hs, ff, 0.05, |e| {
        0.05 + e as f32 * 0.01
    });
    adapter
        .validate_dims(cfg)
        .expect("expert adapter validates against the model");

    let base_batched = batched(None);
    let lora_batched = batched(Some(adapter.clone()));

    // The adapter must reach the prefill path at all. Without this the other
    // two assertions would still pass if prefill quietly used the base weights.
    assert_ne!(
        base_batched, lora_batched,
        "expert adapter had no effect on the batched prefill path"
    );

    // Batched and per-token must agree with the adapter as closely as without
    // it. Both arms are held to the same fixed bar rather than the adapted arm
    // being judged against the base one: `apply_prefill` is required to be
    // bit-identical to `apply_decode` per column (see `cera::lora`), so the
    // adapter contributes no divergence of its own and there is no tolerance to
    // derive. `base_cos` is reported alongside only to show which side moved
    // when this does fail.
    let base_cos = cosine(&base_batched, &per_token(None));
    let lora_cos = cosine(&lora_batched, &per_token(Some(adapter)));
    assert!(
        base_cos > 0.9999,
        "base batched vs per-token cosine {base_cos:.8} must exceed 0.9999"
    );
    assert!(
        lora_cos > 0.9999,
        "adapted batched vs per-token cosine {lora_cos:.8} must exceed 0.9999 \
         (base arm was {base_cos:.8}), so prefill applies the deltas differently \
         from decode"
    );

    eprintln!("[moe_lora] prefill: base cos={base_cos:.8} lora cos={lora_cos:.8} ✓");
}
