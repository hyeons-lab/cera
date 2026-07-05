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
use cera::model::{Model, load_model};

fn lfm2_model_path() -> Option<PathBuf> {
    let p = std::env::var("CERA_LFM2_MODEL").ok().map(PathBuf::from)?;
    (p.exists() && GgufFile::open(&p).is_ok()).then_some(p)
}

/// Build a synthetic PEFT-safetensors adapter touching layer 0's `q_proj`
/// (`k=hs`, `d=q_dim`) and `gate_proj` (`k=hs`, `d=intermediate`) at `rank`,
/// filled with `a_fill` / `b_fill`, then load it (`alpha`).
fn synth_adapter(
    hs: usize,
    q_dim: usize,
    intermediate: usize,
    rank: usize,
    a_fill: f32,
    b_fill: f32,
    alpha: f32,
) -> Arc<LoraAdapterWeights> {
    let mut data: Vec<u8> = Vec::new();
    let mut header = serde_json::Map::new();
    let mut push = |header: &mut serde_json::Map<String, serde_json::Value>,
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
    // q_proj: A=[rank, hs], B=[q_dim, rank]
    push(
        &mut header,
        &mut data,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        rank,
        hs,
        a_fill,
    );
    push(
        &mut header,
        &mut data,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
        q_dim,
        rank,
        b_fill,
    );
    // gate_proj: A=[rank, hs], B=[intermediate, rank]
    push(
        &mut header,
        &mut data,
        "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight",
        rank,
        hs,
        a_fill,
    );
    push(
        &mut header,
        &mut data,
        "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight",
        intermediate,
        rank,
        b_fill,
    );

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

    let base = run(&*model, &tokens, None);
    assert_eq!(base.len(), tokens.len() * hs);

    // (a) B = 0 → the delta is identically zero → bit-identical to base.
    let zero_b = synth_adapter(hs, q_dim, is, 4, 0.3, 0.0, 8.0);
    assert_eq!(
        run(&*model, &tokens, Some(zero_b)),
        base,
        "B=0 adapter must be a bit-identical no-op"
    );

    // (a') scale = 0 (alpha = 0) → also a no-op even with non-zero B.
    let zero_scale = synth_adapter(hs, q_dim, is, 4, 0.3, 0.3, 0.0);
    assert_eq!(
        run(&*model, &tokens, Some(zero_scale)),
        base,
        "scale=0 adapter must be a bit-identical no-op"
    );

    // (b) B ≠ 0, scale ≠ 0 → the hidden states must change.
    let effect = synth_adapter(hs, q_dim, is, 4, 0.05, 0.05, 8.0);
    let with_effect = run(&*model, &tokens, Some(effect));
    assert_ne!(
        with_effect, base,
        "a non-zero adapter must change the hidden states"
    );
    assert!(with_effect.iter().all(|x| x.is_finite()), "finite output");
    eprintln!("lora_noop_and_effect: base/no-op bit-identical; non-zero adapter changed output ✓");
}
