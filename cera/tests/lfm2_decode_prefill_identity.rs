//! LFM2 batched prefill must leave exactly the state, and exactly the logits,
//! that token-by-token decode does, with and without a LoRA adapter.
//!
//! The models are tiny synthetic LFM2s (`scripts/oracle/create_lfm2_test_model.py`): a Q8_0
//! dense one and, with `--moe`, a routed `lfm2moe` (4 of 8 experts, Q4_0).
//! The dense one is Q8_0:
//! Q8_0 is the batched path whose arithmetic is meant to be identical to decode
//! (repacked Q4_0/Q4_K prefill GEMMs are an accepted reorder, so they are not
//! bit-exact and are not checked here).
//!
//! What this pins, all found by `lora_batched_matches_per_token` failing:
//!   * the fused decode `rmsnorm_and_quantize_q8_0` and the prefill
//!     `rmsnorm_into` + quantize produce the same bytes;
//!   * decode's NEON short-conv accumulation is fused multiply-add like prefill's.
//!     That one is invisible to the base model, because the next Q8_0 quantization
//!     absorbs a 1-ulp difference, and only an f32 consumer of the conv output,
//!     the LoRA `out_proj` hook, exposes it: hence the adapter case;
//!   * the LoRA hooks in batched prefill read token-major buffers;
//!   * the routed-MoE prefill sums a token's experts the way decode does. That is
//!     pinned directly by `moe_prefill_identity_tests` in `model/lfm2.rs`, which
//!     compares the FFN output bits; the end-to-end MoE test here is a coarser
//!     consistency check, because on a tiny model the next quantization absorbs
//!     a last-bit difference. The MoE model is loaded without the CPU repacks,
//!     whose prefill GEMMs reorder sums on purpose.
//!
//! Exactness is asserted on aarch64 without BLAS (the arithmetic this was written
//! against); elsewhere the test only checks the paths agree closely. It skips when
//! python lacks `numpy`/`gguf`, unless `CERA_REQUIRE_ORACLE=1`.

#![cfg(feature = "mmap")]

use std::path::PathBuf;
use std::sync::Arc;

use cera::gguf::GgufFile;
use cera::kv_cache::{InferenceState, LayerState};
use cera::lora::LoraAdapterWeights;
use cera::model::Model;
use cera::model::lfm2::Lfm2Model;

/// The tiny model's layout (see the generator).
const N_EMBD: usize = 64;
const N_FF: usize = 128;
const N_KV_DIM: usize = 32; // 2 KV heads x head_dim 16

/// A generated test GGUF that deletes itself.
struct TempModel(PathBuf);

impl Drop for TempModel {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn synthetic_model(moe: bool) -> Option<TempModel> {
    let strict = std::env::var("CERA_REQUIRE_ORACLE").as_deref() == Ok("1");
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../scripts/oracle/create_lfm2_test_model.py");
    let has_deps = std::process::Command::new("python3")
        .args(["-c", "import numpy, gguf"])
        .output()
        .is_ok_and(|o| o.status.success());
    if !has_deps {
        assert!(
            !strict,
            "python3 lacks numpy/gguf and CERA_REQUIRE_ORACLE=1"
        );
        eprintln!("skipping: python3 lacks numpy/gguf for the synthetic lfm2 model");
        return None;
    }
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let out = std::env::temp_dir().join(format!(
        "test_lfm2_{}_{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let mut cmd = std::process::Command::new("python3");
    cmd.arg(&script).arg(&out);
    if moe {
        cmd.arg("--moe");
    }
    let status = cmd.status().expect("run python3");
    assert!(
        status.success(),
        "generating the synthetic lfm2 model failed"
    );
    Some(TempModel(out))
}

/// A LoRA adapter with constant factors on the given `(layer, module, in, out)`
/// targets, as safetensors bytes. Constant, strong factors keep the effect large
/// next to rounding noise.
fn adapter(targets: &[(usize, &str, usize, usize)]) -> Arc<LoraAdapterWeights> {
    const RANK: usize = 4;
    let mut data: Vec<u8> = Vec::new();
    let mut header = serde_json::Map::new();
    let mut push = |name: String, rows: usize, cols: usize, fill: f32| {
        let begin = data.len();
        (0..rows * cols).for_each(|_| data.extend_from_slice(&fill.to_le_bytes()));
        header.insert(
            name,
            serde_json::json!({ "dtype": "F32", "shape": [rows, cols], "data_offsets": [begin, data.len()] }),
        );
    };
    targets
        .iter()
        .for_each(|&(layer, module, in_dim, out_dim)| {
            let base = format!("base_model.model.model.layers.{layer}.{module}");
            push(format!("{base}.lora_A.weight"), RANK, in_dim, 0.05);
            push(format!("{base}.lora_B.weight"), out_dim, RANK, 0.04);
        });
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).expect("header");
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(&data);
    LoraAdapterWeights::from_safetensors_bytes(&buf, Some(8.0)).expect("synthetic adapter")
}

/// Adapter on every kind of block: conv in/out (layers 0, 1, 3), attention
/// q/k/v/o (layers 2, 4) and the FFN (layer 4).
fn full_adapter() -> Arc<LoraAdapterWeights> {
    let conv = |layer| {
        [
            (layer, "conv.in_proj", N_EMBD, 3 * N_EMBD),
            (layer, "conv.out_proj", N_EMBD, N_EMBD),
        ]
    };
    let attn = |layer| {
        [
            (layer, "self_attn.q_proj", N_EMBD, N_EMBD),
            (layer, "self_attn.k_proj", N_EMBD, N_KV_DIM),
            (layer, "self_attn.v_proj", N_EMBD, N_KV_DIM),
            (layer, "self_attn.o_proj", N_EMBD, N_EMBD),
        ]
    };
    let targets: Vec<_> = [0, 1, 3]
        .into_iter()
        .flat_map(conv)
        .chain([2, 4].into_iter().flat_map(attn))
        .chain([
            (4, "mlp.gate_proj", N_EMBD, N_FF),
            (4, "mlp.up_proj", N_EMBD, N_FF),
            (4, "mlp.down_proj", N_FF, N_EMBD),
        ])
        .collect();
    adapter(&targets)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Run `tokens` through batched prefill and token-by-token decode on fresh state
/// and check they agree: bit for bit where exactness applies, closely elsewhere.
fn assert_prefill_matches_decode(
    model: &Lfm2Model,
    tokens: &[u32],
    lora: Option<Arc<LoraAdapterWeights>>,
    label: &str,
) {
    let cfg = model.config();
    let exact = cfg!(all(target_arch = "aarch64", not(has_blas)));

    let mut decode = InferenceState::for_prefill(cfg, tokens.len()).expect("state");
    decode.lora = lora.clone();
    let decode_logits = tokens
        .iter()
        .map(|&t| model.forward(&[t], decode.seq_len, &mut decode))
        .last()
        .expect("tokens");

    let mut prefill = InferenceState::for_prefill(cfg, tokens.len()).expect("state");
    prefill.lora = lora;
    let prefill_logits = model.forward_prefill(tokens, 0, &mut prefill);

    if !exact {
        let scale = decode_logits.iter().fold(1.0f32, |m, x| m.max(x.abs()));
        assert!(
            max_abs_diff(&prefill_logits, &decode_logits) < 1e-2 * scale,
            "{label}: prefill and decode logits differ"
        );
        return;
    }
    assert!(
        bits(&prefill_logits) == bits(&decode_logits),
        "{label}: logits differ by {}",
        max_abs_diff(&prefill_logits, &decode_logits)
    );
    decode
        .layers
        .iter()
        .zip(&prefill.layers)
        .enumerate()
        .for_each(|(i, pair)| match pair {
            (LayerState::Conv { buffer: d, .. }, LayerState::Conv { buffer: p, .. }) => {
                assert!(bits(p) == bits(d), "{label}: layer {i} conv state");
            }
            (
                LayerState::Attention {
                    key_cache: dk,
                    value_cache: dv,
                    ..
                },
                LayerState::Attention {
                    key_cache: pk,
                    value_cache: pv,
                    ..
                },
            ) => {
                assert!(bits(pk) == bits(dk), "{label}: layer {i} keys");
                assert!(bits(pv) == bits(dv), "{label}: layer {i} values");
            }
            _ => panic!("{label}: layer {i} kinds differ"),
        });
}

/// Below the 16-token flash-attention threshold: flash attention is a legitimate
/// reorder of the per-token softmax and would not be bit-exact.
const TOKENS: [u32; 12] = [76, 105, 112, 112, 115, 123, 115, 118, 112, 104, 53, 54];

#[test]
fn lfm2_batched_prefill_matches_decode() {
    let Some(model_file) = synthetic_model(false) else {
        return;
    };
    let model = Lfm2Model::from_gguf(GgufFile::open(&model_file.0).expect("open gguf"), 512)
        .expect("load synthetic lfm2");
    assert_prefill_matches_decode(&model, &TOKENS, None, "base");
    assert_prefill_matches_decode(&model, &TOKENS, Some(full_adapter()), "adapter");
}

#[test]
fn lfm2moe_batched_prefill_matches_decode() {
    let Some(model_file) = synthetic_model(true) else {
        return;
    };
    // No CPU repacks: see the module docs.
    let model =
        Lfm2Model::from_gguf_no_repack(GgufFile::open(&model_file.0).expect("open gguf"), 512)
            .expect("load synthetic lfm2moe");
    assert!(
        model.config().moe.is_some(),
        "the synthetic model must be routed"
    );
    assert_prefill_matches_decode(&model, &TOKENS, None, "moe base");
}
