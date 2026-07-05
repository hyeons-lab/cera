//! CPU LoRA integration parity: the adapter hooks in the forward path must be
//! (a) a true no-op when the delta is zero (`B = 0` or `scale = 0`) — logits
//! **bit-identical** to the base model — and (b) actually change the output when
//! the delta is non-zero. This pins that the apply is wired in (no-op proves the
//! hook fires without perturbing anything; effect proves it's not dead code)
//! without needing a llama.cpp `--lora` golden.
//!
//! Gated behind `CERA_LORA_PARITY=1` + a plain LFM2 GGUF via `CERA_LFM2_MODEL`
//! (CPU only). Run:
//!   CERA_LORA_PARITY=1 CERA_LFM2_MODEL=... cargo test -p cera --release \
//!     --test lora_parity -- --ignored --nocapture

use std::path::PathBuf;
use std::sync::Arc;

use cera::gguf::GgufFile;
use cera::kv_cache::InferenceState;
use cera::lora::LoraAdapterWeights;
use cera::model::{BlockType, Model, load_model};

fn lfm2_model_path() -> Option<PathBuf> {
    let p = std::env::var("CERA_LFM2_MODEL").ok().map(PathBuf::from)?;
    (p.exists() && GgufFile::open(&p).is_ok()).then_some(p)
}

/// Build a synthetic PEFT-safetensors adapter on layer 0 for the given
/// `(peft_module, out_dim)` targets (input width is always `hs`), at `rank`,
/// filled with `a_fill` / `b_fill`, then load it (`alpha`). A per-module list
/// lets us pin attention (`q_proj`) and FFN (`gate_proj`) hooks *separately* —
/// a both-targets adapter would mask a missing hook in one of them.
fn synth_adapter(
    layer: usize,
    hs: usize,
    targets: &[(&str, usize)],
    rank: usize,
    a_fill: f32,
    b_fill: f32,
    alpha: f32,
) -> Arc<LoraAdapterWeights> {
    let mut data: Vec<u8> = Vec::new();
    let mut header = serde_json::Map::new();
    let push = |header: &mut serde_json::Map<String, serde_json::Value>,
                data: &mut Vec<u8>,
                name: &str,
                rows: usize,
                cols: usize,
                fill: f32| {
        let begin = data.len();
        for _ in 0..rows * cols {
            data.extend_from_slice(&fill.to_le_bytes());
        }
        header.insert(
            name.to_string(),
            serde_json::json!({ "dtype": "F32", "shape": [rows, cols], "data_offsets": [begin, data.len()] }),
        );
    };
    for (module, out_dim) in targets {
        // A = [rank, hs], B = [out_dim, rank].
        let base = format!("base_model.model.model.layers.{layer}.{module}");
        push(
            &mut header,
            &mut data,
            &format!("{base}.lora_A.weight"),
            rank,
            hs,
            a_fill,
        );
        push(
            &mut header,
            &mut data,
            &format!("{base}.lora_B.weight"),
            *out_dim,
            rank,
            b_fill,
        );
    }

    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(&data);
    LoraAdapterWeights::from_safetensors_bytes(&buf, Some(alpha)).expect("load synthetic adapter")
}

/// Like [`synth_adapter`] but with an explicit input width per target
/// `(peft_module, in_dim, out_dim)`, so targets whose input width isn't `hs`
/// (e.g. `down_proj`, input = intermediate_size) can be built too.
#[allow(clippy::too_many_arguments)]
fn synth_adapter_io(
    layer: usize,
    targets: &[(&str, usize, usize)],
    rank: usize,
    a_fill: f32,
    b_fill: f32,
    alpha: f32,
) -> Arc<LoraAdapterWeights> {
    let mut data: Vec<u8> = Vec::new();
    let mut header = serde_json::Map::new();
    let push = |header: &mut serde_json::Map<String, serde_json::Value>,
                data: &mut Vec<u8>,
                name: &str,
                rows: usize,
                cols: usize,
                fill: f32| {
        let begin = data.len();
        for _ in 0..rows * cols {
            data.extend_from_slice(&fill.to_le_bytes());
        }
        header.insert(
            name.to_string(),
            serde_json::json!({ "dtype": "F32", "shape": [rows, cols], "data_offsets": [begin, data.len()] }),
        );
    };
    for (module, in_dim, out_dim) in targets {
        let base = format!("base_model.model.model.layers.{layer}.{module}");
        // A = [rank, in_dim], B = [out_dim, rank].
        push(
            &mut header,
            &mut data,
            &format!("{base}.lora_A.weight"),
            rank,
            *in_dim,
            a_fill,
        );
        push(
            &mut header,
            &mut data,
            &format!("{base}.lora_B.weight"),
            *out_dim,
            rank,
            b_fill,
        );
    }
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(&data);
    LoraAdapterWeights::from_safetensors_bytes(&buf, Some(alpha)).expect("load synthetic adapter")
}

/// Per-token hidden states with an optional adapter set on the state.
fn run(model: &dyn Model, tokens: &[u32], lora: Option<Arc<LoraAdapterWeights>>) -> Vec<f32> {
    let mut state = InferenceState::for_prefill(model.config(), tokens.len());
    state.lora = lora;
    model.hidden_states(tokens, &mut state)
}

#[test]
#[ignore = "needs an LFM2 GGUF; gated on CERA_LORA_PARITY"]
fn lora_noop_and_effect() {
    if std::env::var("CERA_LORA_PARITY").as_deref() != Ok("1") {
        eprintln!("skip: set CERA_LORA_PARITY=1 to run");
        return;
    }
    let Some(path) = lfm2_model_path() else {
        eprintln!("skip: no LFM2 model (set CERA_LFM2_MODEL)");
        return;
    };

    let model = load_model(GgufFile::open(&path).expect("open"), None, 512).expect("cpu load");
    assert!(model.supports_hidden_states());
    let cfg = model.config();
    let (hs, q_dim, is) = (
        cfg.hidden_size,
        cfg.n_heads * cfg.head_dim,
        cfg.intermediate_size,
    );
    let tokens: Vec<u32> = vec![1, 5, 9, 42, 100, 7];
    // LFM2 is hybrid: only *attention* layers have q/k/v/o projections (conv
    // layers don't), so the attention adapter must target a real attention layer
    // — layer 0 is a GatedConv. Every layer has an FFN, so gate_proj works anywhere.
    let attn_layer = cfg
        .block_types
        .iter()
        .position(|b| *b == BlockType::Attention)
        .expect("model has an attention layer");
    let q = ("self_attn.q_proj", q_dim); // attention hook (on attn_layer)
    let g = ("mlp.gate_proj", is); // FFN hook (on attn_layer too)

    let base = run(&*model, &tokens, None);
    assert_eq!(base.len(), tokens.len() * hs);

    // (a) B = 0 → the delta is identically zero → bit-identical to base.
    let zero_b = synth_adapter(attn_layer, hs, &[q, g], 4, 0.3, 0.0, 8.0);
    assert_eq!(
        run(&*model, &tokens, Some(zero_b)),
        base,
        "B=0 adapter must be a bit-identical no-op"
    );

    // (a') scale = 0 (alpha = 0) → also a no-op even with non-zero B.
    let zero_scale = synth_adapter(attn_layer, hs, &[q, g], 4, 0.3, 0.3, 0.0);
    assert_eq!(
        run(&*model, &tokens, Some(zero_scale)),
        base,
        "scale=0 adapter must be a bit-identical no-op"
    );

    // (b) A non-zero adapter must change the output — checked SEPARATELY for the
    // attention (q_proj) and FFN (gate_proj) hooks so a missing hook in either
    // can't be masked by the other.
    for (label, target) in [("attention q_proj", q), ("FFN gate_proj", g)] {
        // A non-trivial magnitude so the delta unambiguously propagates through
        // the (attenuating) attention path as well as the FFN residual.
        let eff = synth_adapter(attn_layer, hs, &[target], 4, 0.1, 1.0, 16.0);
        let out = run(&*model, &tokens, Some(eff));
        assert_ne!(out, base, "{label} LoRA must change the hidden states");
        assert!(out.iter().all(|x| x.is_finite()), "{label}: finite output");
    }

    // (c) validate_dims: a matching adapter is accepted; one with the wrong
    // input width (a different model) is rejected up front.
    assert!(
        synth_adapter(attn_layer, hs, &[q, g], 4, 0.1, 0.1, 8.0)
            .validate_dims(cfg)
            .is_ok(),
        "matching adapter must validate"
    );
    assert!(
        synth_adapter(attn_layer, hs + 8, &[q], 4, 0.1, 0.1, 8.0)
            .validate_dims(cfg)
            .is_err(),
        "wrong-input-width adapter must be rejected"
    );
    eprintln!(
        "lora_noop_and_effect: no-op bit-identical; attention + FFN hooks each change output; \
         validate_dims accepts/rejects ✓"
    );
}

/// The batched-GEMM prefill path must apply a LoRA adapter identically to the
/// per-token path. Runs the same prompt through `forward_prefill` (batched GEMM
/// on aarch64/blas, with the in-batch `apply_prefill` hooks) and through a
/// per-token `forward` loop (the decode `apply_decode` hooks), both with the same
/// adapter, and asserts the last-token logits match. They differ only by f32
/// accumulation order (batched GEMM vs per-column GEMV), so the tolerance is
/// tight but not bit-exact.
#[test]
#[ignore = "needs an LFM2 GGUF; gated on CERA_LORA_PARITY"]
fn lora_batched_matches_per_token() {
    if std::env::var("CERA_LORA_PARITY").as_deref() != Ok("1") {
        eprintln!("skip: set CERA_LORA_PARITY=1 to run");
        return;
    }
    let Some(path) = lfm2_model_path() else {
        eprintln!("skip: no LFM2 model (set CERA_LFM2_MODEL)");
        return;
    };

    let model = load_model(GgufFile::open(&path).expect("open"), None, 512).expect("cpu load");
    let cfg = model.config();
    let hs = cfg.hidden_size;
    let q_dim = cfg.n_heads * cfg.head_dim;
    let is = cfg.intermediate_size;

    // Target a real attention layer so the q/k/v/o hooks fire; every layer has an
    // FFN so gate/up/down fire too. Cover ALL seven targets to exercise every
    // batched hook (input widths: q/k/v/gate/up = hs, o = q_dim, down = is).
    let attn_layer = cfg
        .block_types
        .iter()
        .position(|b| *b == BlockType::Attention)
        .expect("model has an attention layer");
    let kv_heads = cfg
        .kv_heads_per_layer
        .get(attn_layer)
        .copied()
        .filter(|&h| h > 0)
        .unwrap_or(cfg.n_kv_heads);
    let kv_dim = kv_heads * cfg.head_dim;

    let targets = [
        ("self_attn.q_proj", hs, q_dim),
        ("self_attn.k_proj", hs, kv_dim),
        ("self_attn.v_proj", hs, kv_dim),
        ("self_attn.o_proj", q_dim, hs),
        ("mlp.gate_proj", hs, is),
        ("mlp.up_proj", hs, is),
        ("mlp.down_proj", is, hs),
    ];
    let adapter = synth_adapter_io(attn_layer, &targets, 4, 0.08, 0.05, 8.0);
    adapter.validate_dims(cfg).expect("adapter validates");

    // A multi-token prompt (n > 1 so the batched path is taken).
    let tokens: Vec<u32> = vec![1, 5, 9, 42, 100, 7, 3, 88];

    // Batched prefill vs per-token forward loop, for both the base model and the
    // adapter-active model. `forward_prefill` returns last-token logits; the
    // per-token loop's last `forward` returns the same.
    let batched = |lora: Option<Arc<LoraAdapterWeights>>| {
        let mut st = InferenceState::for_prefill(cfg, tokens.len());
        st.lora = lora;
        model.forward_prefill(&tokens, 0, &mut st)
    };
    let per_token = |lora: Option<Arc<LoraAdapterWeights>>| {
        let mut st = InferenceState::for_prefill(cfg, tokens.len());
        st.lora = lora;
        let mut logits = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            logits = model.forward(&[tok], i, &mut st);
        }
        logits
    };

    // cosine similarity + max-abs diff between two logit vectors.
    let compare = |a: &[f32], b: &[f32]| -> (f64, f64) {
        assert_eq!(a.len(), b.len(), "logit length mismatch");
        assert!(
            a.iter().chain(b).all(|x| x.is_finite()),
            "logits must be finite"
        );
        let (mut dot, mut na, mut nb, mut max_abs) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            dot += x as f64 * y as f64;
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
            max_abs = max_abs.max((x as f64 - y as f64).abs());
        }
        (dot / (na.sqrt() * nb.sqrt()), max_abs)
    };

    // Base-model divergence: the inherent batched-GEMM vs per-token-GEMV
    // difference, independent of any adapter. On aarch64 the Q8_0 integer kernels
    // accumulate identically, so this is typically bit-exact (max_abs 0).
    let base_logits = per_token(None);
    let (base_cos, base_max) = compare(&batched(None), &base_logits);
    // Adapter-active divergence: `apply_prefill` (batched) vs `apply_decode`
    // (per-token). Both are f32 low-rank products with different accumulation
    // orders, so they're near-identical, not bit-exact.
    let lora_batched = batched(Some(adapter.clone()));
    let (lora_cos, lora_max) = compare(&lora_batched, &per_token(Some(adapter.clone())));

    // Logit scale, so the max-abs diff can be judged relative to signal magnitude.
    let logit_scale = base_logits
        .iter()
        .fold(0.0f64, |m, &x| m.max((x as f64).abs()))
        .max(1.0);

    eprintln!(
        "lora_batched_matches_per_token: base cos={base_cos:.8} max_abs={base_max:.6e} | \
         lora cos={lora_cos:.8} max_abs={lora_max:.6e} | logit_scale={logit_scale:.4} \
         rel={:.3e}",
        lora_max / logit_scale
    );
    // Primary gate: the two paths must be near-identical in direction.
    assert!(
        lora_cos > 0.9999,
        "adapter-active batched vs per-token cosine {lora_cos:.8} must exceed 0.9999"
    );
    // The adapter must also change the output (else this test proves nothing).
    assert_ne!(lora_batched, base_logits, "adapter must alter the logits");
    // Absolute diff must be tiny relative to the logit magnitude — pure f32
    // accumulation-order noise, not a wiring bug.
    assert!(
        lora_max < 1e-3 * logit_scale,
        "adapter batched vs per-token max_abs {lora_max:.6e} exceeds 1e-3 × logit_scale \
         ({logit_scale:.4}) — the batched LoRA hook likely diverges"
    );
}
