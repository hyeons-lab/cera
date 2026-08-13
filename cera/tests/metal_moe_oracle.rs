//! Metal mixture-of-experts oracle: the routed FFN pinned against the CPU one.
//!
//! `tests/slang_multitarget_parity.rs` pins the three MoE kernels
//! (`moe_route`, `moe_gemv_q4_0`, `moe_combine`) against CPU references on
//! synthetic data, including exact expert-id agreement. This file covers what
//! that cannot: that the *model* wires them together correctly on a real
//! `lfm2moe` GGUF, with the right stacked-tensor strides, the right per-layer
//! router and bias, and the dense leading blocks still taking the dense path.
//!
//! ## Why this is not a bit-exact oracle
//!
//! The other Metal oracles (`metal_turboquant_oracle`) assert byte equality.
//! This one asserts a cosine floor, and the reason is a property of the
//! architecture rather than a weakness in the port.
//!
//! `lfm2moe` ranks experts by `sigmoid(logit) + exp_probs_b`, where the bias is
//! trained to balance expert load. Balancing pushes the scores *together*, so
//! the gap at the top-k boundary is routinely ~1e-3, the same order as the
//! difference between the CPU and Metal hidden states feeding the router (Metal
//! keeps a f16 KV cache where the CPU path is f32, which alone is the dense
//! model's ~2e-4 divergence). When the boundary gap is smaller than that
//! difference the two backends select a different fourth expert, and from there
//! they are computing genuinely different, equally valid functions. No
//! tolerance on the output can paper that over, so the test states it.
//!
//! Measured on LFM2.5-8B-A1B-Q4_0 with a 7-token prompt: 9 of 154
//! (layer, token) routing decisions differ, always in the lowest-weighted slot,
//! and the final hidden state agrees to cosine 0.99. llama.cpp shows the same
//! effect on this model, where its Metal backend diverges from its own CPU
//! backend at the first generated token.
//!
//! ## The assertion that would catch a real bug
//!
//! The first token is the load-bearing one. With a single position there is no
//! attention history, so the CPU and Metal hidden states entering the first
//! routed layer agree to ~1e-4, no boundary flips, and the routed FFN has to
//! reproduce the CPU result to dense-model accuracy. A wrong expert stride, a
//! transposed activation index, or a router read from the wrong layer all show
//! up there immediately. The k=1 diagnostic that localized this port's
//! behaviour agreed on the selected expert for 17 consecutive layers, which no
//! indexing bug survives.
//!
//! Gated on an `lfm2moe` GGUF via `CERA_LFM2MOE_MODEL`. Run:
//!   CERA_LFM2MOE_MODEL=... cargo test -p cera --release --features metal \
//!     --test metal_moe_oracle -- --ignored --nocapture

#![cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]

use std::path::PathBuf;
use std::sync::Arc;

use cera::gguf::GgufFile;
use cera::kv_cache::InferenceState;
use cera::lora::{LoraAdapterWeights, LoraTarget};
use cera::model::{BlockType, Model, load_model, load_model_metal};
use cera::session::{CeraError, ModalityCapabilities, Session, SessionConfig};
use cera::tokenizer::BpeTokenizer;

mod common;

fn model_path() -> Option<PathBuf> {
    let p = std::env::var("CERA_LFM2MOE_MODEL")
        .ok()
        .map(PathBuf::from)?;
    (p.exists() && GgufFile::open(&p).is_ok()).then_some(p)
}

/// `"<bos>The capital of France is Paris"` under the LFM2.5 vocabulary.
///
/// Hard-coded rather than run through the tokenizer so the fixture cannot
/// change underneath the bounds below: the two cosine floors were measured
/// against exactly this stream, and a retokenization that shifted it would move
/// them without touching this file. Seven tokens is enough for routing boundary
/// flips to appear (they start at token 1) while keeping both backends'
/// full-prompt run under a second.
const TOKENS: &[u32] = &[124894, 597, 5205, 302, 3980, 355, 20551];

fn hidden(model: &dyn Model, tokens: &[u32]) -> Vec<f32> {
    let mut state = InferenceState::for_prefill(model.config(), tokens.len())
        .expect("prefill state for a prompt this short");
    model.hidden_states(tokens, &mut state)
}

/// Next-token logits for the whole prompt, through `forward_prefill`.
///
/// Deliberately not `hidden_states`: on Metal that is a token-at-a-time loop
/// over the decode encoders, so it exercises the routed FFN's decode arm only.
/// The batched-prefill arm is a different call with two conventions the decode
/// one does not have (`accumulate = false`, and `x` aliased to `out` in the same
/// buffer), and it is the arm normal generation runs the prompt through.
fn prefill_logits(model: &dyn Model, tokens: &[u32]) -> Vec<f32> {
    let mut state = InferenceState::for_prefill(model.config(), tokens.len())
        .expect("prefill state for a prompt this short");
    model.forward_prefill(tokens, 0, &mut state)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    let na: f64 = a
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    (dot / (na * nb)) as f32
}

/// The two backends' models plus the hidden size they agree on.
struct Pair {
    cpu: Box<dyn Model>,
    metal: Box<dyn Model>,
    hs: usize,
}

/// Load the CPU and Metal models for one test.
///
/// Not cached across tests: `Model` is not `Clone`, a `OnceLock` would have to
/// hand out `&'static` references, and the GGUF is memory-mapped so the second
/// load is page-cache-warm rather than a re-read. The cost is per-test load
/// time on a multi-GB file, which is why the assertions are grouped into a few
/// tests rather than split one per assertion.
///
/// `None` means only that no model was configured. A load that *fails* panics
/// instead: `model_path` has already proved the file exists and parses, so
/// every remaining failure is a regression in the thing under test (the
/// Q4_0-only expert check, the shape validation, a missing tensor), and
/// returning `None` for those would print "SKIP: set CERA_LFM2MOE_MODEL" and
/// pass the suite green on a broken loader.
fn both() -> Option<Pair> {
    let path = model_path()?;
    let open = || GgufFile::open(&path).expect("model_path checked this opens");
    let cpu = load_model(open(), Some(&path), 4096).expect("lfm2moe loads on the CPU backend");
    let metal = load_model_metal(open(), &path, 4096).expect("lfm2moe loads on the Metal backend");
    let hs = cpu.config().hidden_size;
    assert!(
        cpu.config().moe.is_some(),
        "CERA_LFM2MOE_MODEL is not a mixture-of-experts model; this suite would pass vacuously"
    );
    Some(Pair { cpu, metal, hs })
}

/// The first token, where routing is not yet ambiguous, must match the CPU to
/// the same accuracy a dense model does.
///
/// This is the assertion with teeth. Every structural mistake the routed path
/// can make (an expert stride that lands on a neighbour's weights, the
/// gate/up activation index read per entry instead of per token, the router or
/// bias taken from the wrong layer) changes this token's output far beyond the
/// bound below, because it changes it for *every* layer rather than for the one
/// marginal expert a boundary flip moves.
#[test]
#[ignore = "needs an lfm2moe GGUF via CERA_LFM2MOE_MODEL"]
fn first_token_matches_cpu_to_dense_accuracy() {
    let Some(Pair { cpu, metal, hs }) = both() else {
        eprintln!("[metal-moe-oracle] SKIP: set CERA_LFM2MOE_MODEL");
        return;
    };
    let want = hidden(cpu.as_ref(), &TOKENS[..1]);
    let got = hidden(metal.as_ref(), &TOKENS[..1]);
    assert_eq!(want.len(), hs);
    let cos = cosine(&want, &got);
    assert!(
        cos > 0.9995,
        "first-token hidden state cosine {cos:.6} is below the dense-model bound; \
         this is a wiring bug in the routed FFN, not a routing boundary flip \
         (a single position has no attention history for the two backends to \
         disagree on)"
    );
    eprintln!("[metal-moe-oracle] first token cosine {cos:.6}");
}

/// Across a full prompt the two backends stay close, but not dense-close, and
/// the gap is routing divergence rather than arithmetic error.
///
/// The floor is deliberately loose *and* two-sided: too low and a real
/// regression hides under it, but asserting a dense-model bound here would fail
/// on correct code the first time a boundary flip landed. See the module docs
/// for the measurement this is set from.
#[test]
#[ignore = "needs an lfm2moe GGUF via CERA_LFM2MOE_MODEL"]
fn full_prompt_stays_within_the_routing_divergence_bound() {
    let Some(Pair { cpu, metal, hs }) = both() else {
        eprintln!("[metal-moe-oracle] SKIP: set CERA_LFM2MOE_MODEL");
        return;
    };
    let want = hidden(cpu.as_ref(), TOKENS);
    let got = hidden(metal.as_ref(), TOKENS);
    assert_eq!(want.len(), TOKENS.len() * hs);

    let worst = want
        .chunks_exact(hs)
        .zip(got.chunks_exact(hs))
        .enumerate()
        .map(|(i, (a, b))| (cosine(a, b), i))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("TOKENS is non-empty");
    eprintln!(
        "[metal-moe-oracle] worst per-token cosine {:.6} at token {}",
        worst.0, worst.1
    );
    assert!(
        worst.0 > 0.9,
        "worst per-token cosine {:.6} (token {}) is below the bound; a few \
         lowest-slot expert flips cost ~0.05, so this is more than routing \
         divergence",
        worst.0,
        worst.1,
    );
}

/// The batched-prefill routed FFN agrees with the CPU, and with Metal's own
/// decode path.
///
/// The other two tests reach the routed FFN only through `hidden_states`, which
/// on Metal loops token-at-a-time over the decode encoders. The prefill arm is a
/// separate call with two conventions decode does not share: it writes its
/// combined output rather than accumulating into the residual (the *next*
/// layer's fused `add_rmsnorm_batch` folds that in), and it is handed the same
/// buffer for `x` and `out`. Both are exactly the kind of thing that is correct
/// on one path and wrong on the other, and normal generation runs every prompt
/// through this one.
///
/// Asserted two ways because they fail differently. Against the CPU it is the
/// same loose routing-divergence bound as above, since the same boundary flips
/// apply. Against Metal's own decode of the same prompt the bound is tight: one
/// backend and one set of weights, so a dropped or double-counted residual
/// shows up as a gross mismatch rather than a near miss.
///
/// The tight bound rests on the two paths *expecting* the same routing, not on
/// a guarantee of it. Batched GEMM and flash attention are not bit-identical to
/// the decode GEMV and split attention, so the router input differs by ~1e-6,
/// which is two orders below the ~1e-3 top-k boundary gaps that make the
/// CPU comparison loose. Measured at cosine 1.000000. If this ever flakes, the
/// cause to rule out first is a genuine boundary flip, not the wiring.
#[test]
#[ignore = "needs an lfm2moe GGUF via CERA_LFM2MOE_MODEL"]
fn batched_prefill_agrees_with_decode_and_cpu() {
    let Some(Pair { cpu, metal, .. }) = both() else {
        eprintln!("[metal-moe-oracle] SKIP: set CERA_LFM2MOE_MODEL");
        return;
    };

    let cpu_logits = prefill_logits(cpu.as_ref(), TOKENS);
    let metal_logits = prefill_logits(metal.as_ref(), TOKENS);
    assert_eq!(cpu_logits.len(), metal_logits.len());

    // Metal's own decode of the same prompt: feed the tokens one at a time, so
    // the last call's logits predict the same next token the prefill does.
    //
    // On a *fresh* model, not the one just prefilled. The Metal backend keeps
    // its KV cache and the shortconv rolling buffers on the model rather than
    // in `InferenceState`, so replaying from position 0 on the prefilled
    // instance decodes against a dirty cache. That scored 0.43 against a prefill
    // path the CPU agrees with to 0.9958, i.e. it reads as a kernel bug and is
    // purely an artifact of reusing the instance.
    let decode_logits = {
        let path = model_path().expect("both() already resolved it");
        let fresh = load_model_metal(
            GgufFile::open(&path).expect("model_path checked this opens"),
            &path,
            4096,
        )
        .expect("lfm2moe loads on the Metal backend");
        let mut state = InferenceState::for_prefill(fresh.config(), TOKENS.len())
            .expect("prefill state for a prompt this short");
        TOKENS
            .iter()
            .enumerate()
            .map(|(pos, &tok)| fresh.forward(&[tok], pos, &mut state))
            .last()
            .expect("TOKENS is non-empty")
    };

    let argmax = |v: &[f32]| -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .expect("logits are non-empty")
    };

    let vs_decode = cosine(&metal_logits, &decode_logits);
    let vs_cpu = cosine(&metal_logits, &cpu_logits);
    eprintln!(
        "[metal-moe-oracle] prefill logits cosine {vs_decode:.6} vs metal decode, \
         {vs_cpu:.6} vs cpu prefill"
    );

    assert!(
        vs_decode > 0.99,
        "batched-prefill logits diverge from this backend's own decode \
         (cosine {vs_decode:.6}); one backend and one set of weights, so suspect \
         the prefill FFN wiring before a routing boundary flip"
    );
    assert_eq!(
        argmax(&metal_logits),
        argmax(&decode_logits),
        "batched prefill and decode predict different next tokens"
    );
    assert!(
        vs_cpu > 0.9,
        "batched-prefill logits are below the CPU routing-divergence bound \
         (cosine {vs_cpu:.6})"
    );
}

/// Metal refuses a routed-FFN adapter instead of applying half of it.
///
/// Driven through `Session::attach_lora_adapters`, not through the
/// `supports_moe_lora` flag it reads: asserting the flag would pass even with
/// the rejection in `session.rs` deleted, which is the failure this test exists
/// to catch. The predicate behind it, including that a *router* delta counts as
/// a routed-FFN delta, is pinned separately by
/// `lora::tests::has_moe_deltas_covers_the_router_and_the_experts`.
///
/// A router adapter is the sharp case. Structurally it is an ordinary dense
/// target with ordinary dims, so it uploads without complaint and then never
/// reaches a hook; nothing in the output looks wrong.
#[test]
#[ignore = "needs an lfm2moe GGUF via CERA_LFM2MOE_MODEL"]
fn metal_session_rejects_a_router_lora() {
    let Some(path) = model_path() else {
        eprintln!("[metal-moe-oracle] SKIP: set CERA_LFM2MOE_MODEL");
        return;
    };
    let gguf = GgufFile::open(&path).expect("model_path checked this opens");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("lfm2moe carries a tokenizer");
    let model: Arc<dyn Model> =
        Arc::from(load_model_metal(gguf, &path, 512).expect("lfm2moe loads on Metal"));
    let (hidden_size, n_expert, routed_layer) = {
        let cfg = model.config();
        let moe = cfg.moe.as_ref().expect("lfm2moe has MoE params");
        // The layer has to be a *routed* one. On a dense layer `validate_dims`
        // rejects the adapter first, with its own message that also says
        // "mixture-of-experts", and the test would pass without ever reaching
        // the gate it exists to cover.
        let routed = moe
            .is_moe_layer
            .iter()
            .position(|&b| b)
            .expect("lfm2moe has at least one routed layer");
        (cfg.hidden_size, moe.n_expert, routed)
    };
    let mut session = Session::new(
        model,
        Arc::new(tokenizer),
        ModalityCapabilities::text_only(),
        SessionConfig::default(),
    )
    .expect("session over a freshly loaded model");

    // Rank-2 router delta on the first routed layer. `ffn_gate_inp` is
    // `[hidden, n_expert]`, so A is `[hidden, rank]` and B is `[rank, n_expert]`
    // in GGUF's fastest-varying-first order.
    let rank = 2;
    let bytes = common::write_lora_gguf(
        &[
            (
                format!("blk.{routed_layer}.ffn_gate_inp.weight.lora_a"),
                vec![hidden_size, rank],
                vec![0.01; rank * hidden_size],
            ),
            (
                format!("blk.{routed_layer}.ffn_gate_inp.weight.lora_b"),
                vec![rank, n_expert],
                vec![0.01; n_expert * rank],
            ),
        ],
        8.0,
    );
    let adapter = LoraAdapterWeights::from_gguf_bytes(Arc::from(bytes.into_boxed_slice()))
        .expect("a well-formed router adapter loads");
    assert!(
        adapter.has_moe_deltas(),
        "fixture is wrong: this adapter carries no routed-FFN delta, so the \
         rejection below would be vacuous"
    );

    let err = session
        .attach_lora_adapters(adapter)
        .expect_err("Metal has no routed-FFN hooks, so this adapter must be refused");
    // Match the variant, not the message. The dimension check that runs first
    // also says "mixture-of-experts", so anchoring on the text would let this
    // pass on a rejection the gate never made; `LoraUnsupportedByBackend` is
    // reachable only from the `supports_moe_lora` gate this test covers.
    assert!(
        matches!(err, CeraError::LoraUnsupportedByBackend(_)),
        "rejected, but not by the backend-capability gate this test covers: {err:?}"
    );
    eprintln!("[metal-moe-oracle] router adapter refused: {err}");
}

/// An adapter set straight onto `InferenceState` is dropped whole, not applied
/// in half.
///
/// `metal_session_rejects_a_router_lora` covers the gate a normal caller meets.
/// This covers the one that has no gate: `InferenceState::lora` is a public
/// field and `Model::forward` takes the state directly, so the parity harness,
/// an FFI embedder, or a test can hand this backend an adapter `Session` would
/// have refused.
///
/// The fixture carries **both** a router delta and an `attn_output` delta, which
/// is what makes the assertion mean anything. Dropping the adapter and applying
/// only the half this backend has hooks for are indistinguishable if the
/// adapter's only targets are ones it cannot apply: the attention delta is the
/// half that *would* land, so bit-identical logits are evidence the whole
/// adapter was refused rather than evidence that nothing happened.
///
/// `attn_output` specifically, and on a layer that is both routed and an
/// attention block. Two earlier fixtures passed with the guard deleted: one
/// targeted `attn_q` on the first routed layer, which is a gated-conv block with
/// no such weight, and one targeted `attn_q` on a routed *attention* layer,
/// where a single token at position 0 softmaxes over exactly one key and returns
/// 1.0 whatever the score is, so a Q delta provably cannot reach the output. Do
/// not "simplify" this back to `attn_q`; check any replacement by deleting the
/// guard and watching this fail.
///
/// Two freshly loaded models rather than two calls on one: this backend keeps
/// its KV cache and conv rolling buffers on the model, so a second forward at
/// position 0 would run against a dirty cache and the comparison would be
/// measuring that instead.
#[test]
#[ignore = "needs an lfm2moe GGUF via CERA_LFM2MOE_MODEL"]
fn a_routed_adapter_on_the_state_is_dropped_whole() {
    let Some(path) = model_path() else {
        eprintln!("[metal-moe-oracle] SKIP: set CERA_LFM2MOE_MODEL");
        return;
    };
    let load = || {
        load_model_metal(
            GgufFile::open(&path).expect("model_path checked this opens"),
            &path,
            512,
        )
        .expect("lfm2moe loads on the Metal backend")
    };

    let (hidden_size, q_dim, n_expert, routed_layer) = {
        let model = load();
        let cfg = model.config();
        let moe = cfg.moe.as_ref().expect("lfm2moe has MoE params");
        // Routed *and* an attention block. LFM2 alternates gated-conv and
        // attention layers, and routing is a property of the FFN slot, so the
        // first routed layer is usually a conv one with no `attn_q` at all: an
        // `attn_q` delta aimed there targets a projection the model does not
        // have and is never applied, which makes the assertion below pass
        // whatever the guard does. Found the hard way.
        let routed = cfg
            .block_types
            .iter()
            .enumerate()
            .position(|(i, &bt)| {
                bt == BlockType::Attention && moe.is_moe_layer.get(i).copied().unwrap_or(false)
            })
            .expect("lfm2moe has at least one routed attention layer");
        (
            cfg.hidden_size,
            cfg.n_heads * cfg.head_dim,
            moe.n_expert,
            routed,
        )
    };

    let rank = 2;
    let bytes = common::write_lora_gguf(
        &[
            (
                format!("blk.{routed_layer}.ffn_gate_inp.weight.lora_a"),
                vec![hidden_size, rank],
                vec![0.01; rank * hidden_size],
            ),
            (
                format!("blk.{routed_layer}.ffn_gate_inp.weight.lora_b"),
                vec![rank, n_expert],
                vec![0.01; n_expert * rank],
            ),
            // The half this backend does have hooks for. `attn_output`, not
            // `attn_q`: this test runs a single token at position 0, where
            // attention softmaxes over exactly one key and returns 1.0 whatever
            // the score is, so a Q delta cannot reach the output at all. An
            // `attn_q` fixture here passes with the guard deleted. Measured:
            // this one moves the logits by 17.5.
            (
                format!("blk.{routed_layer}.attn_output.weight.lora_a"),
                vec![q_dim, rank],
                vec![0.05; rank * q_dim],
            ),
            (
                format!("blk.{routed_layer}.attn_output.weight.lora_b"),
                vec![rank, hidden_size],
                vec![0.05; hidden_size * rank],
            ),
        ],
        8.0,
    );
    let adapter = LoraAdapterWeights::from_gguf_bytes(Arc::from(bytes.into_boxed_slice()))
        .expect("a well-formed adapter loads");
    assert!(
        adapter.has_moe_deltas(),
        "fixture is wrong: without a routed-FFN delta this asserts nothing"
    );
    assert!(
        adapter.get(routed_layer, LoraTarget::AttnOutput).is_some(),
        "fixture is wrong: the attention half did not parse, so 'dropped' and 'half applied' \
         would look identical and this test would pass either way"
    );

    let logits = |lora: Option<Arc<LoraAdapterWeights>>| -> Vec<f32> {
        let model = load();
        let mut state = InferenceState::for_prefill(model.config(), TOKENS.len())
            .expect("prefill state for a prompt this short");
        state.lora = lora;
        model.forward(&TOKENS[..1], 0, &mut state)
    };

    let base = logits(None);
    let with_adapter = logits(Some(adapter));
    assert_eq!(
        base, with_adapter,
        "a routed-FFN adapter set directly on the state changed the logits, so its attention \
         half was applied while the router and expert deltas were dropped: the half-applied \
         adapter `supports_moe_lora` exists to prevent"
    );
    eprintln!("[metal-moe-oracle] routed adapter on the state was dropped whole");
}
