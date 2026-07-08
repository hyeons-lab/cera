//! wgpu LoRA apply parity: the WGSL LoRA epilogue on `GpuLfm2Model` must match
//! the CPU `Lfm2Model` LoRA apply, and stay a true no-op when the delta is zero.
//!
//! Two properties are pinned:
//!  1. **Effect parity** — with a non-zero synthetic adapter set on both the CPU
//!     and the wgpu model, per-token `hidden_states` agree at cosine > 0.99
//!     (GPU/CPU float accumulation differs, so cosine, not bit-equality, is the
//!     bar — the same methodology as `hidden_states_parity.rs`). A wrong B
//!     pre-scale, transposed A/B, or a dropped hook collapses cosine well below
//!     0.99 on the first token.
//!  2. **No-op** — a `B = 0` adapter leaves the wgpu output equal to the base
//!     (no-adapter) wgpu output within a tight `1e-3` tolerance (in practice the
//!     delta is exactly `0.0`). The adapter buffers are still uploaded and the
//!     hooks still dispatch; the delta `B·(A·x)` is just identically zero because
//!     `B = 0`. This proves the hook fires without perturbing anything.
//!
//! Gated behind `CERA_LORA_WGPU_PARITY=1` + a plain LFM2 GGUF via
//! `CERA_LFM2_MODEL`, `#[ignore]`, `gpu` feature only. Run:
//!   CERA_LORA_WGPU_PARITY=1 CERA_LFM2_MODEL=... cargo test -p cera --release \
//!     --features gpu --test lora_wgpu_parity -- --ignored --nocapture

#![cfg(feature = "gpu")]

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

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = na * nb;
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Build a synthetic PEFT-safetensors adapter on `layer` targeting the given
/// `(peft_module, out_dim)` pairs (input width is always `hs`), at `rank`,
/// filled with `a_fill` / `b_fill`, then load it with `alpha`. Copied from
/// `lora_metal_parity.rs` so this test carries no cross-file dependency.
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

fn run(model: &dyn Model, tokens: &[u32], lora: Option<Arc<LoraAdapterWeights>>) -> Vec<f32> {
    let mut state = InferenceState::for_prefill(model.config(), tokens.len()).unwrap();
    state.lora = lora;
    model.hidden_states(tokens, &mut state)
}

/// Max absolute per-element difference between two equal-length vectors.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[ignore = "needs an LFM2 GGUF + a GPU; gated on CERA_LORA_WGPU_PARITY"]
fn wgpu_lora_matches_cpu_and_noop() {
    if std::env::var("CERA_LORA_WGPU_PARITY").as_deref() != Ok("1") {
        eprintln!("skip: set CERA_LORA_WGPU_PARITY=1 to run");
        return;
    }
    let Some(path) = lfm2_model_path() else {
        eprintln!("skip: no LFM2 model (set CERA_LFM2_MODEL)");
        return;
    };

    // CPU reference model.
    let cpu = load_model(GgufFile::open(&path).expect("open"), None, 8192).expect("cpu load");
    assert!(cpu.supports_hidden_states());
    let cfg = cpu.config();
    let (hs, q_dim, is) = (
        cfg.hidden_size,
        cfg.n_heads * cfg.head_dim,
        cfg.intermediate_size,
    );

    // wgpu model.
    let gpu = match cera::model::load_model_gpu(
        GgufFile::open(&path).expect("open gguf"),
        Some(&path),
        8192,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skip: no wgpu GPU: {e}");
            return;
        }
    };
    assert!(gpu.supports_hidden_states());

    let tokens: Vec<u32> = vec![1, 5, 9, 42, 100, 7];

    // LFM2 is hybrid: only *attention* layers have q/k/v/o projections (conv
    // layers don't), so the attention adapter must target a real attention
    // layer. Every layer has an FFN, so gate_proj works there too.
    let attn_layer = cfg
        .block_types
        .iter()
        .position(|b| *b == BlockType::Attention)
        .expect("model has an attention layer");
    let q = ("self_attn.q_proj", q_dim); // attention hook
    let g = ("mlp.gate_proj", is); // FFN hook

    // ── (1) effect parity: non-zero adapter, CPU vs wgpu per-token cosine ──
    let eff = synth_adapter(attn_layer, hs, &[q, g], 4, 0.1, 1.0, 16.0);
    let cpu_out = run(cpu.as_ref(), &tokens, Some(eff.clone()));
    let gpu_out = run(gpu.as_ref(), &tokens, Some(eff));
    assert_eq!(cpu_out.len(), tokens.len() * hs, "CPU shape");
    assert_eq!(gpu_out.len(), tokens.len() * hs, "wgpu shape");
    assert!(
        gpu_out.iter().all(|x| x.is_finite()),
        "wgpu LoRA output must be finite"
    );

    let mut min_cos = f32::INFINITY;
    for t in 0..tokens.len() {
        let c = &cpu_out[t * hs..(t + 1) * hs];
        let m = &gpu_out[t * hs..(t + 1) * hs];
        let cos = cosine(c, m);
        min_cos = min_cos.min(cos);
        assert!(
            cos > 0.99,
            "token {t}: wgpu-vs-CPU LoRA cosine {cos:.5} < 0.99"
        );
    }

    // wgpu `hidden_states` may not be bit-reproducible across submissions
    // (ULP-level GPU accumulation noise — see `hidden_states_parity.rs`, which
    // uses a 1e-3 bar). So the no-op below is checked against that same noise
    // floor rather than bit-equality. Measure the floor with two base runs.
    let gpu_base = run(gpu.as_ref(), &tokens, None);
    let gpu_base2 = run(gpu.as_ref(), &tokens, None);
    let base_noise = max_abs_diff(&gpu_base, &gpu_base2);

    // Sanity: the non-zero adapter's effect is far larger than the noise floor
    // (not a dead hook masked by reproducibility slack).
    let eff_delta = max_abs_diff(&gpu_out, &gpu_base);
    assert!(
        eff_delta > base_noise * 10.0 && eff_delta > 1e-2,
        "non-zero LoRA must change the wgpu hidden states well beyond GPU noise \
         (eff_delta {eff_delta:.3e}, base_noise {base_noise:.3e})"
    );

    // ── (2) no-op: a B = 0 adapter must not perturb the output beyond the GPU
    // reproducibility noise floor. The buffers upload and the hooks dispatch, but
    // `B·(A·x)` is identically zero (B = 0), so this is the base pipeline plus
    // only ULP-level cross-run noise. ──
    let zero_b = synth_adapter(attn_layer, hs, &[q, g], 4, 0.3, 0.0, 8.0);
    let gpu_zero = run(gpu.as_ref(), &tokens, Some(zero_b));
    let noop_delta = max_abs_diff(&gpu_zero, &gpu_base);
    assert!(
        noop_delta < 1e-3,
        "B=0 adapter must be a no-op within the GPU noise floor \
         (noop_delta {noop_delta:.3e}, base_noise {base_noise:.3e})"
    );

    eprintln!(
        "wgpu_lora_matches_cpu_and_noop: {} tokens, D={hs}, min CPU-vs-wgpu cosine {min_cos:.5}; \
         non-zero adapter delta {eff_delta:.3e} (base noise {base_noise:.3e}); \
         B=0 no-op delta {noop_delta:.3e} < 1e-3 ✓",
        tokens.len()
    );
}

/// Prefill logits for the last token via the batched `forward_prefill` path.
fn prefill_logits(
    model: &dyn Model,
    tokens: &[u32],
    lora: Option<Arc<LoraAdapterWeights>>,
) -> Vec<f32> {
    let mut state = InferenceState::for_prefill(model.config(), tokens.len()).unwrap();
    state.lora = lora;
    model.forward_prefill(tokens, 0, &mut state)
}

/// Like [`synth_adapter`] but fills A and B with **index-dependent** values (not
/// a single constant), so the LoRA delta VARIES across output channels. A uniform
/// fill makes the delta constant per output channel for the pure-linear hooks
/// (o/gate/up/down), which hides an output-channel/N-dimension GEMM layout bug;
/// this varied fill exposes it. Copied from `lora_metal_parity.rs`:
/// `A[r][j] = a·(1+0.4·((3r+j)%5))`, `B[o][r] = b·(1+0.5·((o+2r)%7)) · (±1 by o parity)`.
fn synth_adapter_io_varied(
    layer: usize,
    targets: &[(&str, usize, usize)],
    rank: usize,
    a: f32,
    b: f32,
    alpha: f32,
) -> Arc<LoraAdapterWeights> {
    let mut data: Vec<u8> = Vec::new();
    let mut header = serde_json::Map::new();
    let push = |header: &mut serde_json::Map<String, serde_json::Value>,
                data: &mut Vec<u8>,
                name: &str,
                rows: usize,
                cols: usize,
                f: &dyn Fn(usize, usize) -> f32| {
        let begin = data.len();
        for r in 0..rows {
            for c in 0..cols {
                data.extend_from_slice(&f(r, c).to_le_bytes());
            }
        }
        header.insert(
            name.to_string(),
            serde_json::json!({ "dtype": "F32", "shape": [rows, cols], "data_offsets": [begin, data.len()] }),
        );
    };
    for (module, in_dim, out_dim) in targets {
        let base = format!("base_model.model.model.layers.{layer}.{module}");
        // A[rank×in_dim]: varies by (r, j).
        let fa = |r: usize, j: usize| a * (1.0 + 0.4 * ((3 * r + j) % 5) as f32);
        push(
            &mut header,
            &mut data,
            &format!("{base}.lora_A.weight"),
            rank,
            *in_dim,
            &fa,
        );
        // B[out_dim×rank]: varies by (o, r), sign flips by output-channel parity —
        // a channel-scramble bug reorders these and collapses the cosine.
        let fb = |o: usize, r: usize| {
            let sign = if o % 2 == 0 { 1.0 } else { -1.0 };
            b * sign * (1.0 + 0.5 * ((o + 2 * r) % 7) as f32)
        };
        push(
            &mut header,
            &mut data,
            &format!("{base}.lora_B.weight"),
            *out_dim,
            rank,
            &fb,
        );
    }
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(&data);
    LoraAdapterWeights::from_safetensors_bytes(&buf, Some(alpha)).expect("load varied adapter")
}

/// Batched-prefill LoRA parity: with an adapter active, `GpuLfm2Model`'s
/// batched-GEMM `forward_prefill` (in-batch LoRA across all N tokens, two NT
/// GEMMs per target) must match the CPU `Lfm2Model`'s batched prefill at last-
/// token-logit cosine above 0.99. A wrong GEMM layout (token- vs channel-major),
/// a transposed/mis-scaled B, a dropped hook, or a conv-layer QKV hook collapses
/// this well below 0.99. Mirrors Metal's `metal_batched_lora_matches_cpu_prefill`.
#[test]
#[ignore = "needs an LFM2 GGUF + a GPU; gated on CERA_LORA_WGPU_PARITY"]
fn wgpu_batched_lora_matches_cpu_prefill() {
    if std::env::var("CERA_LORA_WGPU_PARITY").as_deref() != Ok("1") {
        eprintln!("skip: set CERA_LORA_WGPU_PARITY=1 to run");
        return;
    }
    let Some(path) = lfm2_model_path() else {
        eprintln!("skip: no LFM2 model (set CERA_LFM2_MODEL)");
        return;
    };

    let cpu = load_model(GgufFile::open(&path).expect("open"), None, 8192).expect("cpu load");
    let cfg = cpu.config();
    let hs = cfg.hidden_size;
    let is = cfg.intermediate_size;
    let head_dim = cfg.head_dim;
    let q_dim = cfg.n_heads * head_dim;
    let vocab = cfg.vocab_size;

    let gpu = match cera::model::load_model_gpu(
        GgufFile::open(&path).expect("open gguf"),
        Some(&path),
        8192,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skip: no wgpu GPU: {e}");
            return;
        }
    };

    // Target an attention layer so q/k/v/o adapters land on real projections; FFN
    // targets are valid on any layer, but we use the same (attention) layer here.
    let attn_layer = cfg
        .block_types
        .iter()
        .position(|b| *b == BlockType::Attention)
        .expect("model has an attention layer");
    let kv_dim = cfg.kv_heads_per_layer[attn_layer] * head_dim;

    let attn: &[(&str, usize, usize)] = &[
        ("self_attn.q_proj", hs, q_dim),
        ("self_attn.k_proj", hs, kv_dim),
        ("self_attn.v_proj", hs, kv_dim),
        ("self_attn.o_proj", q_dim, hs),
    ];
    let ffn: &[(&str, usize, usize)] = &[
        ("mlp.gate_proj", hs, is),
        ("mlp.up_proj", hs, is),
        ("mlp.down_proj", is, hs),
    ];
    let all: Vec<(&str, usize, usize)> = attn.iter().chain(ffn).copied().collect();

    // A multi-token prompt (n > 1) forces the batched path.
    let tokens: Vec<u32> = vec![1, 5, 9, 42, 100, 7, 3, 11];
    let gpu_base = prefill_logits(gpu.as_ref(), &tokens, None);

    // Compare wgpu-batched vs CPU-batched last-token logits for an adapter over
    // `targets`, asserting the adapter is live (moves the logits vs base) and the
    // two backends agree at cosine > 0.99.
    let check = |label: &str, targets: &[(&str, usize, usize)], a: f32, b: f32, alpha: f32| {
        // VARIED (per-output-channel) fills: a uniform fill makes the delta
        // constant across output channels for the linear hooks (o/gate/up/down),
        // which would hide an output-channel/N-dimension GEMM layout bug. Varied
        // fills actually pin the channel layout.
        let adapter = synth_adapter_io_varied(attn_layer, targets, 4, a, b, alpha);
        let cpu_logits = prefill_logits(cpu.as_ref(), &tokens, Some(adapter.clone()));
        let gpu_logits = prefill_logits(gpu.as_ref(), &tokens, Some(adapter));
        assert_eq!(cpu_logits.len(), vocab, "{label}: CPU logits shape");
        assert_eq!(gpu_logits.len(), vocab, "{label}: wgpu logits shape");
        assert!(
            gpu_logits.iter().all(|x| x.is_finite()),
            "{label}: wgpu batched-LoRA logits must be finite"
        );
        // Live-hook guard: a dead batched hook would leave the logits at base.
        let eff_delta = max_abs_diff(&gpu_logits, &gpu_base);
        assert!(
            eff_delta > 1e-2,
            "{label}: batched LoRA must change the wgpu logits (eff_delta {eff_delta:.3e})"
        );
        let cos = cosine(&cpu_logits, &gpu_logits);
        assert!(
            cos > 0.99,
            "{label}: batched wgpu-vs-CPU LoRA prefill cosine {cos:.5} < 0.99"
        );
        eprintln!("  {label}: cosine {cos:.5} > 0.99, adapter delta {eff_delta:.3e} ✓");
        cos
    };

    // Per-group + all-7 at a REALISTIC magnitude (scale = alpha/rank = 1): pins
    // every batched hook's GEMM layout, B pre-scale, and buffer wiring (all four
    // attention hooks; all three FFN hooks incl. the residual-fed down_proj). At
    // aggressive scale (≥2) the attention softmax + FFN silu amplify any f16-KV
    // accumulation difference into the logits, so scale-1 is the correctness gate.
    eprintln!(
        "wgpu_batched_lora_matches_cpu_prefill: {} tokens, vocab={vocab}, layer {attn_layer}",
        tokens.len()
    );
    check("attn q/k/v/o", attn, 0.01, 0.01, 4.0);
    check("ffn gate/up/down", ffn, 0.01, 0.01, 4.0);
    check("all 7 composed", &all, 0.01, 0.01, 4.0);
}
