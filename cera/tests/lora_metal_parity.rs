//! Metal LoRA apply parity: the MSL LoRA epilogue on `MetalLfm2Model` must match
//! the CPU `Lfm2Model` LoRA apply, and stay a true no-op when the delta is zero.
//!
//! Two properties are pinned:
//!  1. **Effect parity** — with a non-zero synthetic adapter set on both the CPU
//!     and the Metal model, per-token `hidden_states` agree at cosine > 0.99
//!     (GPU/CPU float accumulation differs, so cosine, not bit-equality, is the
//!     bar — the same methodology as `hidden_states_parity.rs`). A wrong B
//!     pre-scale, transposed A/B, or a dropped hook collapses cosine well below
//!     0.99 on the first token.
//!  2. **No-op** — a `B = 0` adapter leaves the Metal output equal to the base
//!     (no-adapter) Metal output within a tight `1e-3` tolerance (in practice the
//!     delta is exactly `0.0`, since Metal `hidden_states` is run-to-run
//!     deterministic on this model). This proves the hook fires without
//!     perturbing anything (a `+= 0.0` that flipped a sign or reordered accum
//!     would show up here).
//!
//! Gated behind `CERA_LORA_METAL_PARITY=1` + a plain LFM2 GGUF via
//! `CERA_LFM2_MODEL`, `#[ignore]`, macOS + `metal` only. Run:
//!   CERA_LORA_METAL_PARITY=1 CERA_LFM2_MODEL=... cargo test -p cera --release \
//!     --features metal --test lora_metal_parity -- --ignored --nocapture

#![cfg(all(feature = "metal", target_os = "macos"))]

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
/// `lora_parity.rs` so this test carries no cross-file dependency.
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
    let mut state = InferenceState::for_prefill(model.config(), tokens.len());
    state.lora = lora;
    model.hidden_states(tokens, &mut state)
}

#[test]
#[ignore = "needs an LFM2 GGUF + a Metal GPU; gated on CERA_LORA_METAL_PARITY"]
fn metal_lora_matches_cpu_and_noop() {
    if std::env::var("CERA_LORA_METAL_PARITY").as_deref() != Ok("1") {
        eprintln!("skip: set CERA_LORA_METAL_PARITY=1 to run");
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

    // Metal model.
    let metal =
        match cera::model::load_model_metal(GgufFile::open(&path).expect("open gguf"), &path, 8192)
        {
            Ok(m) => m,
            Err(e) => {
                eprintln!("skip: no Metal GPU: {e}");
                return;
            }
        };
    assert!(metal.supports_hidden_states());

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

    // ── (1) effect parity: non-zero adapter, CPU vs Metal per-token cosine ──
    let eff = synth_adapter(attn_layer, hs, &[q, g], 4, 0.1, 1.0, 16.0);
    let cpu_out = run(cpu.as_ref(), &tokens, Some(eff.clone()));
    let metal_out = run(metal.as_ref(), &tokens, Some(eff));
    assert_eq!(cpu_out.len(), tokens.len() * hs, "CPU shape");
    assert_eq!(metal_out.len(), tokens.len() * hs, "Metal shape");
    assert!(
        metal_out.iter().all(|x| x.is_finite()),
        "Metal LoRA output must be finite"
    );

    let mut min_cos = f32::INFINITY;
    for t in 0..tokens.len() {
        let c = &cpu_out[t * hs..(t + 1) * hs];
        let m = &metal_out[t * hs..(t + 1) * hs];
        let cos = cosine(c, m);
        min_cos = min_cos.min(cos);
        assert!(
            cos > 0.99,
            "token {t}: Metal-vs-CPU LoRA cosine {cos:.5} < 0.99"
        );
    }

    // Metal `hidden_states` is not bit-reproducible across command buffers
    // (documented ULP-level GPU accumulation noise — see `hidden_states_parity.rs`,
    // which uses a 1e-3 bar). So the no-op below is checked against that same
    // noise floor rather than bit-equality. Measure the floor with two base runs.
    let metal_base = run(metal.as_ref(), &tokens, None);
    let metal_base2 = run(metal.as_ref(), &tokens, None);
    let base_noise = max_abs_diff(&metal_base, &metal_base2);

    // Sanity: the non-zero adapter's effect is far larger than the noise floor
    // (not a dead hook masked by reproducibility slack).
    let eff_delta = max_abs_diff(&metal_out, &metal_base);
    assert!(
        eff_delta > base_noise * 10.0 && eff_delta > 1e-2,
        "non-zero LoRA must change the Metal hidden states well beyond GPU noise \
         (eff_delta {eff_delta:.3e}, base_noise {base_noise:.3e})"
    );

    // ── (2) no-op: a B = 0 adapter must not perturb the output beyond the GPU
    // reproducibility noise floor. The zero delta is skipped at upload, so this
    // is the base pipeline plus only ULP-level cross-run noise. ──
    let zero_b = synth_adapter(attn_layer, hs, &[q, g], 4, 0.3, 0.0, 8.0);
    let metal_zero = run(metal.as_ref(), &tokens, Some(zero_b));
    let noop_delta = max_abs_diff(&metal_zero, &metal_base);
    assert!(
        noop_delta < 1e-3,
        "B=0 adapter must be a no-op within the GPU noise floor \
         (noop_delta {noop_delta:.3e}, base_noise {base_noise:.3e})"
    );

    eprintln!(
        "metal_lora_matches_cpu_and_noop: {} tokens, D={hs}, min CPU-vs-Metal cosine {min_cos:.5}; \
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
    let mut state = InferenceState::for_prefill(model.config(), tokens.len());
    state.lora = lora;
    model.forward_prefill(tokens, 0, &mut state)
}

/// Batched-prefill LoRA parity: with an adapter active, `MetalLfm2Model`'s
/// batched-GEMM `forward_prefill` (in-batch LoRA, all N tokens) must match the
/// CPU `Lfm2Model`'s batched prefill at cosine > 0.99 on the last-token logits.
/// A wrong GEMM layout (token- vs channel-major), a transposed/mis-scaled B, a
/// dropped hook, or a conv-layer QKV hook collapses this well below 0.99.
#[test]
#[ignore = "needs an LFM2 GGUF + a Metal GPU; gated on CERA_LORA_METAL_PARITY"]
fn metal_batched_lora_matches_cpu_prefill() {
    if std::env::var("CERA_LORA_METAL_PARITY").as_deref() != Ok("1") {
        eprintln!("skip: set CERA_LORA_METAL_PARITY=1 to run");
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

    let metal =
        match cera::model::load_model_metal(GgufFile::open(&path).expect("open gguf"), &path, 8192)
        {
            Ok(m) => m,
            Err(e) => {
                eprintln!("skip: no Metal GPU: {e}");
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
    let metal_base = prefill_logits(metal.as_ref(), &tokens, None);

    // Compare Metal-batched vs CPU-batched last-token logits for an adapter over
    // `targets`, asserting the adapter is live (moves the logits vs base) and the
    // two backends agree at cosine > 0.99.
    let check = |label: &str, targets: &[(&str, usize, usize)], a: f32, b: f32, alpha: f32| {
        // VARIED (per-output-channel) fills: a uniform fill makes the delta
        // constant across output channels for the linear hooks (o/gate/up/down),
        // which would hide an output-channel/N-dimension GEMM layout bug. Varied
        // fills actually pin the channel layout.
        let adapter = synth_adapter_io_varied(attn_layer, targets, 4, a, b, alpha);
        let cpu_logits = prefill_logits(cpu.as_ref(), &tokens, Some(adapter.clone()));
        let metal_logits = prefill_logits(metal.as_ref(), &tokens, Some(adapter));
        assert_eq!(cpu_logits.len(), vocab, "{label}: CPU logits shape");
        assert_eq!(metal_logits.len(), vocab, "{label}: Metal logits shape");
        assert!(
            metal_logits.iter().all(|x| x.is_finite()),
            "{label}: Metal batched-LoRA logits must be finite"
        );
        // Live-hook guard: a dead batched hook would leave the logits at base.
        let eff_delta = max_abs_diff(&metal_logits, &metal_base);
        assert!(
            eff_delta > 1e-2,
            "{label}: batched LoRA must change the Metal logits (eff_delta {eff_delta:.3e})"
        );
        let cos = cosine(&cpu_logits, &metal_logits);
        assert!(
            cos > 0.99,
            "{label}: batched Metal-vs-CPU LoRA prefill cosine {cos:.5} < 0.99"
        );
        eprintln!("  {label}: cosine {cos:.5} > 0.99, adapter delta {eff_delta:.3e} ✓");
        cos
    };

    // Per-group at a strong magnitude: pins every batched hook's GEMM layout,
    // B pre-scale, and buffer wiring (all four attention hooks; all three FFN
    // hooks incl. the residual-fed down_proj).
    eprintln!(
        "metal_batched_lora_matches_cpu_prefill: {} tokens, vocab={vocab}, layer {attn_layer}",
        tokens.len()
    );
    check("attn q/k/v/o", attn, 0.01, 0.01, 4.0);
    check("ffn gate/up/down", ffn, 0.01, 0.01, 4.0);
    // All seven composed. A REALISTIC magnitude (scale = alpha/rank = 1). At
    // aggressive scale (≥2) the attention softmax + FFN silu amplify the Metal
    // f16 (vs CPU f32) accumulation difference into the logits — see
    // `metal_batched_vs_pertoken_isolates_f16` for the magnitude sweep that pins
    // that drop as f16 amplification, not a layout bug.
    check("all 7 composed", &all, 0.01, 0.01, 4.0);
}

/// Like [`synth_adapter_io`] but fills A and B with **index-dependent** values
/// (not a single constant), so the LoRA delta VARIES across output channels. A
/// uniform fill makes the delta constant per output channel for the pure-linear
/// hooks (o/gate/up/down), which hides an output-channel/N-dimension layout bug;
/// this varied fill exposes it. `A[r][j] = a·(1+0.4·((3r+j)%5))`,
/// `B[o][r] = b·(1+0.5·((o+2r)%7)) · (±1 by o parity)`.
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

/// Last-token logits via the PER-TOKEN path: loop `forward` one token at a time,
/// which applies LoRA through the *decode* hooks (`b_scaled`, per-token GEMV).
/// Fresh model instance ⇒ the internal GPU KV starts empty.
fn per_token_logits(
    model: &dyn Model,
    tokens: &[u32],
    lora: Option<Arc<LoraAdapterWeights>>,
) -> Vec<f32> {
    let mut state = InferenceState::for_prefill(model.config(), tokens.len());
    state.lora = lora;
    let mut logits = Vec::new();
    for (j, &tok) in tokens.iter().enumerate() {
        logits = model.forward(&[tok], j, &mut state);
    }
    logits
}

/// DISCRIMINATOR for the strong-all-7 Metal-vs-CPU cosine drop (~0.88): is it a
/// batched-LoRA LOGIC bug, or genuinely f16-KV-vs-f32 amplification? Compare
/// Metal-BATCHED (`forward_prefill`, `b_batched`, in-batch GEMM) against
/// Metal-PER-TOKEN (`forward` loop, `b_scaled`, decode GEMV) — BOTH f16 KV, so
/// the f16-vs-f32 difference cancels and only the batched-vs-per-token LoRA logic
/// remains. Runs the STRONG magnitude (0.02/alpha 8) that dropped Metal-vs-CPU to
/// ~0.88. If the batched logic is correct, Metal-batched≈Metal-per-token even
/// though Metal-vs-CPU drops — proving the drop is f16, not a bug.
#[test]
#[ignore = "needs an LFM2 GGUF + a Metal GPU; gated on CERA_LORA_METAL_PARITY"]
fn metal_batched_vs_pertoken_isolates_f16() {
    if std::env::var("CERA_LORA_METAL_PARITY").as_deref() != Ok("1") {
        eprintln!("skip: set CERA_LORA_METAL_PARITY=1 to run");
        return;
    }
    let Some(path) = lfm2_model_path() else {
        eprintln!("skip: no LFM2 model");
        return;
    };
    let cpu = load_model(GgufFile::open(&path).expect("open"), None, 8192).expect("cpu load");
    let cfg = cpu.config();
    let (hs, is, head_dim) = (cfg.hidden_size, cfg.intermediate_size, cfg.head_dim);
    let q_dim = cfg.n_heads * head_dim;
    // Two fresh Metal instances: one for the batched path, one for per-token
    // (each owns its GPU KV, so neither run pollutes the other).
    let mk = || cera::model::load_model_metal(GgufFile::open(&path).expect("open"), &path, 8192);
    let (metal_b, metal_p) = match (mk(), mk()) {
        (Ok(a), Ok(b)) => (a, b),
        _ => {
            eprintln!("skip: no Metal GPU");
            return;
        }
    };
    let attn_layer = cfg
        .block_types
        .iter()
        .position(|b| *b == BlockType::Attention)
        .expect("attn layer");
    let kv_dim = cfg.kv_heads_per_layer[attn_layer] * head_dim;
    let all: Vec<(&str, usize, usize)> = vec![
        ("self_attn.q_proj", hs, q_dim),
        ("self_attn.k_proj", hs, kv_dim),
        ("self_attn.v_proj", hs, kv_dim),
        ("self_attn.o_proj", q_dim, hs),
        ("mlp.gate_proj", hs, is),
        ("mlp.up_proj", hs, is),
        ("mlp.down_proj", is, hs),
    ];
    let tokens: Vec<u32> = vec![1, 5, 9, 42, 100, 7, 3, 11];

    // Baseline (no adapter): the intrinsic Metal batched-vs-per-token divergence
    // (f16 GEMM vs f16 GEMV accumulation) — the floor cos_mm can reach.
    let base_mm = cosine(
        &prefill_logits(metal_b.as_ref(), &tokens, None),
        &per_token_logits(metal_p.as_ref(), &tokens, None),
    );

    // STRONG all-7 adapter, VARIED fills (per-output-channel delta — exposes any
    // output-channel/N-dimension layout bug that a uniform fill would hide).
    let adapter = synth_adapter_io_varied(attn_layer, &all, 4, 0.02, 0.02, 8.0);
    let mb = prefill_logits(metal_b.as_ref(), &tokens, Some(adapter.clone())); // Metal batched
    let mp = per_token_logits(metal_p.as_ref(), &tokens, Some(adapter.clone())); // Metal per-token
    let cb = prefill_logits(cpu.as_ref(), &tokens, Some(adapter.clone())); // CPU batched (ground truth)
    let cp = per_token_logits(cpu.as_ref(), &tokens, Some(adapter.clone())); // CPU per-token (ground truth)

    // Full pairwise matrix vs the CPU f32 ground truth. CPU-batched≈CPU-per-token
    // is the reference (verified in #217); whichever Metal path drops away from
    // BOTH CPU results is the buggy one.
    let cos_cc = cosine(&cb, &cp); // CPU batched vs CPU per-token (consistency check)
    let cos_mb_cb = cosine(&mb, &cb); // Metal batched vs CPU batched
    let cos_mp_cp = cosine(&mp, &cp); // Metal per-token vs CPU per-token (LoRA-4 decode path)
    let cos_mm = cosine(&mb, &mp); // Metal batched vs Metal per-token

    // Cleanest confirmation via the ESTABLISHED per-token `hidden_states` path
    // (the requester's classifier path) with the varied adapter — CPU vs Metal.
    let cpu_h = run(cpu.as_ref(), &tokens, Some(adapter.clone()));
    let met_h = run(metal_p.as_ref(), &tokens, Some(adapter.clone()));
    let mut min_h = f32::INFINITY;
    for t in 0..tokens.len() {
        min_h = min_h.min(cosine(
            &cpu_h[t * hs..(t + 1) * hs],
            &met_h[t * hs..(t + 1) * hs],
        ));
    }

    eprintln!(
        "VARIED strong all-7: base(no-lora) Mbatch-vs-Mpertok {base_mm:.5}\n  \
         CPUbatch-vs-CPUpertok {cos_cc:.5} (ground-truth consistency)\n  \
         Mbatch-vs-CPUbatch    {cos_mb_cb:.5}\n  \
         Mpertok-vs-CPUpertok  {cos_mp_cp:.5}  <- LoRA-4 decode path\n  \
         Mbatch-vs-Mpertok     {cos_mm:.5}\n  \
         hidden_states CPU-vs-Metal (varied) min per-token cosine {min_h:.5}  <- requester's path"
    );
    // Isolate: FFN-only vs attention-only varied adapters on the decode/hidden
    // path. FFN has NO attention nonlinearity, so if FFN-only decode diverges
    // from CPU it's a genuine kernel/hook bug (not attention f16 accumulation).
    let ffn_only: Vec<(&str, usize, usize)> = all
        .iter()
        .filter(|(m, _, _)| m.starts_with("mlp"))
        .copied()
        .collect();
    let attn_only: Vec<(&str, usize, usize)> = all
        .iter()
        .filter(|(m, _, _)| m.starts_with("self_attn"))
        .copied()
        .collect();
    // Magnitude sweep: a real channel-scramble BUG is magnitude-independent (wrong
    // delta direction at any scale → cosine stays low); f16 AMPLIFICATION recovers
    // as the perturbation shrinks (cosine → ~1 at small magnitude). At the
    // REALISTIC magnitude (scale = alpha/rank = 1) BOTH groups must exceed 0.99 —
    // that is the correctness gate; the aggressive drop is the characterization.
    let mut realistic: std::collections::HashMap<&str, f32> = Default::default();
    for (label, tg) in [("FFN-only", &ffn_only), ("attn-only", &attn_only)] {
        for (a, b, alpha) in [(0.02, 0.02, 8.0), (0.008, 0.008, 4.0), (0.002, 0.002, 4.0)] {
            let ad = synth_adapter_io_varied(attn_layer, tg, 4, a, b, alpha);
            let ch = run(cpu.as_ref(), &tokens, Some(ad.clone()));
            let mh = run(metal_p.as_ref(), &tokens, Some(ad));
            let mut mc = f32::INFINITY;
            for t in 0..tokens.len() {
                mc = mc.min(cosine(&ch[t * hs..(t + 1) * hs], &mh[t * hs..(t + 1) * hs]));
            }
            eprintln!(
                "  {label} varied (a={a} b={b} α={alpha}) hidden CPU-vs-Metal min cosine {mc:.5}"
            );
            if (alpha - 4.0).abs() < 1e-6 && (a - 0.008).abs() < 1e-6 {
                realistic.insert(label, mc);
            }
        }
    }
    eprintln!("cos_mm (Metal batched vs per-token, aggressive) = {cos_mm:.5}");

    // ── Correctness gates (realistic scale-1 varied adapter) ──
    // 1. The batched path (this PR) matches the CPU ground truth at the aggressive
    //    magnitude already (batched GEMM accumulates stably) — cheap strong check.
    assert!(
        cos_mb_cb > 0.99,
        "Metal-batched-vs-CPU varied {cos_mb_cb:.5} < 0.99 — a batched-LoRA layout/scale bug"
    );
    // 2. The decode path recovers to >0.99 at realistic magnitude for BOTH the
    //    FFN and attention groups — proving the aggressive drop is f16
    //    amplification (magnitude-dependent), not a channel bug (magnitude-independent).
    let ffn_r = realistic["FFN-only"];
    let attn_r = realistic["attn-only"];
    assert!(
        ffn_r > 0.99 && attn_r > 0.99,
        "decode varied parity at realistic scale-1: FFN {ffn_r:.5}, attn {attn_r:.5} — \
         a magnitude-INDEPENDENT drop here would be a real channel bug, not f16"
    );
}

/// Max absolute per-element difference between two equal-length vectors.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}
