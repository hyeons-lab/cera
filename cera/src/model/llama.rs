// Plain dense transformer text model. Covers two RoPE families on one code path:
//   - NEOX (split-halves) rope: Qwen2, Qwen3.
//   - NORM (interleaved-pair) rope: LLaMA, Mistral, Granite 3.x.
//
// Per-arch differences are gated on tensor presence / metadata at load time:
//   - Qwen2 carries Q/K/V projection biases (`blk.N.attn_{q,k,v}.bias`) and no
//     QK-norm.
//   - Qwen3 carries per-head Q/K RMSNorm weights (`blk.N.attn_{q,k}_norm.weight`)
//     and no biases.
//   - LLaMA / Mistral carry neither (plain attention) but use NORM rope.
//   - Granite 3.x is a NORM-rope llama variant plus four scalar multipliers
//     (`{arch}.embedding_scale`, `.residual_scale`, `.attention.scale`,
//     `.logit_scale`). All default to identity, so the other archs are unaffected.
//
// GGUF weights for every supported arch are stored un-permuted, matching
// llama.cpp, so the correct rope layout is selected per arch (NEOX vs NORM)
// rather than permuting weights at load.

use anyhow::{Context, Result, bail, ensure};

use crate::backend::cpu;
use crate::backend::cpu::RopeType;
use crate::gguf::GgufFile;
use crate::kv_cache::InferenceState;
#[cfg(any(target_arch = "aarch64", feature = "blas"))]
use crate::kv_cache::LayerState;
use crate::model::transformer::{self, AttnDims, AttnExtras, AttnWeights, FfnWeights, WeightRef};
use crate::model::{BlockType, Model, ModelConfig, ScalarMultipliers};

// ── Per-layer weight references ─────────────────────────────────────────────

/// Pre-resolved quantized weight refs for one transformer layer.
struct LayerWeightRefs {
    attn_q: WeightRef,
    attn_k: WeightRef,
    attn_v: WeightRef,
    attn_output: WeightRef,
    ffn_gate: WeightRef,
    ffn_up: WeightRef,
    ffn_down: WeightRef,
}

// ── LLaMA-family Model ──────────────────────────────────────────────────────

pub struct LlamaModel {
    gguf: GgufFile,
    config: ModelConfig,
    head_dim: usize,
    /// RoPE pair layout: `Neox` for Qwen2/Qwen3, `Norm` for LLaMA/Mistral/Granite.
    rope_type: RopeType,
    /// Llama-3 RoPE frequency-scaling factors (`rope_freqs.weight`, `head_dim/2`),
    /// applied per-pair on the NORM path. `None` for archs without the tensor
    /// (Qwen/Mistral/Granite) ⇒ plain RoPE.
    rope_freqs: Option<Vec<f32>>,
    // Granite 3.x scalar multipliers live on `config.scalars` (identity for
    // every other arch) — see `ScalarMultipliers`.
    // Pre-dequantized small F32 weights.
    output_norm_weight: Vec<f32>,
    attn_norm_weights: Vec<Vec<f32>>,
    ffn_norm_weights: Vec<Vec<f32>>,
    // Qwen3 per-head QK-norm weights (None for Qwen2).
    attn_q_norm_weights: Vec<Option<Vec<f32>>>,
    attn_k_norm_weights: Vec<Option<Vec<f32>>>,
    // Qwen2 Q/K/V projection biases (None for Qwen3).
    attn_q_bias: Vec<Option<Vec<f32>>>,
    attn_k_bias: Vec<Option<Vec<f32>>>,
    attn_v_bias: Vec<Option<Vec<f32>>>,
    // Pre-resolved quantized weight refs.
    embd_ref: WeightRef,
    /// Separate output projection (`output.weight`) when present; `None` means
    /// tied embeddings (`token_embd.weight` reused for the logit projection).
    output_ref: Option<WeightRef>,
    layer_refs: Vec<LayerWeightRefs>,
    #[allow(dead_code)]
    model_id: String,
}

impl LlamaModel {
    /// Construct without a model identifier.
    #[allow(dead_code)]
    pub fn from_gguf(gguf: GgufFile, context_size: usize) -> Result<Self> {
        Self::from_gguf_with_id(gguf, context_size, String::new())
    }

    /// Construct with an explicit model identifier (typically the GGUF path).
    pub fn from_gguf_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        ensure!(context_size > 0, "context_size must be > 0");

        // Metadata prefix is the architecture string itself
        // ("qwen2"/"qwen3"/"llama"/"granite"; classic Mistral ships as "llama").
        let arch = gguf
            .get_str("general.architecture")
            .context("missing general.architecture")?
            .to_string();
        let prefix = arch.as_str();

        // RoPE layout per arch. Qwen GGUFs are NEOX (split-halves); the
        // LLaMA-family (incl. Mistral and Granite) are NORM (interleaved pairs).
        let rope_type = match prefix {
            "qwen2" | "qwen3" => RopeType::Neox,
            // "llama" also covers classic Mistral (it ships as GGUF arch "llama").
            "llama" | "granite" => RopeType::Norm,
            // Keep exhaustive with the `load_model` dispatch allow-list: a new arch
            // routed here without a layout mapping must fail loudly rather than
            // silently default to NORM (wrong for any NEOX-family arch — phi3,
            // stablelm, gemma, starcoder2, … are all NEOX in llama.cpp).
            other => bail!(
                "LlamaModel: no RoPE layout mapping for arch {other:?}; \
                 add it to the rope_type match in llama.rs"
            ),
        };

        // Granite 3.x scalar multipliers (embedding/residual/attention/logit).
        // Absent on every other arch ⇒ identity, so this is a no-op for
        // LLaMA/Mistral/Qwen. Carried on `config.scalars`.
        let scalars = ScalarMultipliers::from_gguf(&gguf, prefix)?;

        let n_layers =
            gguf.get_u32(&format!("{prefix}.block_count"))
                .with_context(|| format!("missing {prefix}.block_count"))? as usize;
        let hidden_size = gguf
            .get_u32(&format!("{prefix}.embedding_length"))
            .with_context(|| format!("missing {prefix}.embedding_length"))?
            as usize;
        let intermediate_size = gguf
            .get_u32(&format!("{prefix}.feed_forward_length"))
            .with_context(|| format!("missing {prefix}.feed_forward_length"))?
            as usize;
        let n_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count"))
            .with_context(|| format!("missing {prefix}.attention.head_count"))?
            as usize;
        // SCALAR head_count_kv (not the per-layer array LFM2 uses).
        let n_kv_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count_kv"))
            .with_context(|| format!("missing {prefix}.attention.head_count_kv"))?
            as usize;
        ensure!(
            n_heads > 0 && n_kv_heads > 0 && n_heads % n_kv_heads == 0,
            "n_heads ({n_heads}) must be a positive multiple of n_kv_heads ({n_kv_heads})"
        );
        // Qwen GGUFs typically omit `{prefix}.vocab_size`; derive it from the
        // embedding tensor's outer dim (row count) when the key is absent.
        let vocab_size = match gguf.get_u32(&format!("{prefix}.vocab_size")) {
            Some(v) => v as usize,
            None => {
                let info = gguf
                    .tensors
                    .get("token_embd.weight")
                    .context("missing token_embd.weight (cannot derive vocab_size)")?;
                ensure!(
                    info.shape.len() >= 2,
                    "token_embd.weight has unexpected shape {:?}",
                    info.shape
                );
                info.shape[1]
            }
        };

        // Cap max_seq_len by the requested context_size (mirrors LFM2).
        let gguf_max_seq_len = gguf
            .get_u32(&format!("{prefix}.context_length"))
            .unwrap_or(128000) as usize;
        let max_seq_len = context_size.min(gguf_max_seq_len);
        let rope_theta = gguf
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(1_000_000.0);
        let rms_norm_eps = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);

        // head_dim: default hidden_size / n_heads, overridden by the optional
        // `{prefix}.attention.key_length` (Qwen3 sets this explicitly).
        let head_dim = gguf
            .get_u32(&format!("{prefix}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(hidden_size / n_heads);
        ensure!(head_dim > 0, "head_dim must be > 0");

        let block_types = vec![BlockType::Attention; n_layers];
        let kv_heads_per_layer = vec![n_kv_heads; n_layers];

        let config = ModelConfig {
            architecture: arch.clone(),
            n_layers,
            hidden_size,
            intermediate_size,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
            block_types,
            conv_kernel_size: None,
            kv_heads_per_layer,
            scalars,
        };

        // Final norm tensor (NOT the LFM2 `token_embd_norm.weight`).
        let output_norm_weight = gguf.get_tensor("output_norm.weight")?.to_f32_vec();

        let mut attn_norm_weights = Vec::with_capacity(n_layers);
        let mut ffn_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_q_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_k_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_q_bias = Vec::with_capacity(n_layers);
        let mut attn_k_bias = Vec::with_capacity(n_layers);
        let mut attn_v_bias = Vec::with_capacity(n_layers);
        let mut layer_refs = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            attn_norm_weights.push(
                gguf.get_tensor(&format!("blk.{i}.attn_norm.weight"))?
                    .to_f32_vec(),
            );
            ffn_norm_weights.push(
                gguf.get_tensor(&format!("blk.{i}.ffn_norm.weight"))?
                    .to_f32_vec(),
            );

            // Qwen3 QK-norm — gate on tensor presence so the same code path
            // serves both archs.
            let q_norm_name = format!("blk.{i}.attn_q_norm.weight");
            let k_norm_name = format!("blk.{i}.attn_k_norm.weight");
            if gguf.tensors.contains_key(&q_norm_name) {
                attn_q_norm_weights.push(Some(gguf.get_tensor(&q_norm_name)?.to_f32_vec()));
                attn_k_norm_weights.push(Some(gguf.get_tensor(&k_norm_name)?.to_f32_vec()));
            } else {
                attn_q_norm_weights.push(None);
                attn_k_norm_weights.push(None);
            }

            // Qwen2 Q/K/V biases — gate on tensor presence.
            let q_bias_name = format!("blk.{i}.attn_q.bias");
            let k_bias_name = format!("blk.{i}.attn_k.bias");
            let v_bias_name = format!("blk.{i}.attn_v.bias");
            if gguf.tensors.contains_key(&q_bias_name) {
                attn_q_bias.push(Some(gguf.get_tensor(&q_bias_name)?.to_f32_vec()));
                attn_k_bias.push(Some(gguf.get_tensor(&k_bias_name)?.to_f32_vec()));
                attn_v_bias.push(Some(gguf.get_tensor(&v_bias_name)?.to_f32_vec()));
            } else {
                attn_q_bias.push(None);
                attn_k_bias.push(None);
                attn_v_bias.push(None);
            }

            layer_refs.push(LayerWeightRefs {
                attn_q: transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_q.weight"))?,
                attn_k: transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_k.weight"))?,
                attn_v: transformer::resolve_weight(&gguf, &format!("blk.{i}.attn_v.weight"))?,
                attn_output: transformer::resolve_weight(
                    &gguf,
                    &format!("blk.{i}.attn_output.weight"),
                )?,
                ffn_gate: transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_gate.weight"))?,
                ffn_up: transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_up.weight"))?,
                ffn_down: transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_down.weight"))?,
            });
        }

        let embd_ref = transformer::resolve_weight(&gguf, "token_embd.weight")?;
        // Separate output projection when present, else tied embeddings.
        let output_ref = if gguf.tensors.contains_key("output.weight") {
            Some(transformer::resolve_weight(&gguf, "output.weight")?)
        } else {
            None
        };

        // Llama-3 RoPE frequency scaling (`rope_scaling: llama3`): per-pair factors
        // that divide each rotation angle, applied by llama.cpp on every rope call.
        // Present on Llama-3.x, absent on Qwen/Mistral/Granite ⇒ None (plain RoPE).
        let rope_freqs = gguf
            .get_tensor("rope_freqs.weight")
            .ok()
            .map(|t| t.to_f32_vec());
        if let Some(rf) = &rope_freqs {
            ensure!(
                rf.len() == head_dim / 2,
                "rope_freqs.weight has {} entries, expected head_dim/2 = {}",
                rf.len(),
                head_dim / 2
            );
        }

        Ok(Self {
            gguf,
            config,
            head_dim,
            rope_type,
            rope_freqs,
            output_norm_weight,
            attn_norm_weights,
            ffn_norm_weights,
            attn_q_norm_weights,
            attn_k_norm_weights,
            attn_q_bias,
            attn_k_bias,
            attn_v_bias,
            embd_ref,
            output_ref,
            layer_refs,
            model_id,
        })
    }

    /// Attention dims for a layer (constant across layers here).
    fn attn_dims(&self) -> AttnDims<'_> {
        AttnDims {
            hidden_size: self.config.hidden_size,
            n_heads: self.config.n_heads,
            n_kv_heads: self.config.n_kv_heads,
            head_dim: self.head_dim,
            rope_theta: self.config.rope_theta,
            rms_norm_eps: self.config.rms_norm_eps,
            rope_type: self.rope_type,
            attn_scale: self.config.scalars.attn,
            rope_freqs: self.rope_freqs.as_deref(),
        }
    }

    /// Run all layers + final RMSNorm on a single-token hidden state.
    fn run_layers(&self, hidden: &mut [f32], pos: usize, state: &mut InferenceState) {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let dims = self.attn_dims();

        // Take scratch out of `state` to avoid borrow conflicts with the
        // helpers that need `&mut state`; restore at the end.
        let mut normed = std::mem::take(&mut state.scratch.normed);
        let mut ffn_input = std::mem::take(&mut state.scratch.ffn_input);
        normed.resize(hs, 0.0);
        ffn_input.resize(hs, 0.0);

        for i in 0..cfg.n_layers {
            // Attention pre-norm.
            normed.copy_from_slice(hidden);
            cpu::rmsnorm(&mut normed, &self.attn_norm_weights[i], cfg.rms_norm_eps);

            #[cfg(target_arch = "aarch64")]
            transformer::quantize_to_scratch(&normed, state);

            let refs = &self.layer_refs[i];
            let weights = AttnWeights {
                attn_q: &refs.attn_q,
                attn_k: &refs.attn_k,
                attn_v: &refs.attn_v,
                attn_output: &refs.attn_output,
            };
            let extras = AttnExtras {
                qkv_bias: match (
                    self.attn_q_bias[i].as_deref(),
                    self.attn_k_bias[i].as_deref(),
                    self.attn_v_bias[i].as_deref(),
                ) {
                    (Some(q), Some(k), Some(v)) => Some((q, k, v)),
                    _ => None,
                },
                qk_norm: match (
                    self.attn_q_norm_weights[i].as_deref(),
                    self.attn_k_norm_weights[i].as_deref(),
                ) {
                    (Some(q), Some(k)) => Some((q, k)),
                    _ => None,
                },
            };
            transformer::forward_attn_block(
                &self.gguf, i, &weights, &extras, dims, &normed, pos, state,
            );

            // Granite scales the block output before the residual add (identity
            // for every other arch).
            if self.config.scalars.residual != 1.0 {
                cpu::scale_inplace(&mut state.scratch.out[..hs], self.config.scalars.residual);
            }
            cpu::add_inplace(hidden, &state.scratch.out[..hs]);

            // FFN pre-norm.
            ffn_input.copy_from_slice(hidden);
            cpu::rmsnorm(&mut ffn_input, &self.ffn_norm_weights[i], cfg.rms_norm_eps);

            #[cfg(target_arch = "aarch64")]
            transformer::quantize_to_scratch(&ffn_input, state);

            let refs = &self.layer_refs[i];
            let ffn_weights = FfnWeights {
                ffn_gate: &refs.ffn_gate,
                ffn_up: &refs.ffn_up,
                ffn_down: &refs.ffn_down,
            };
            transformer::forward_ffn_block(
                &self.gguf,
                &ffn_weights,
                hs,
                cfg.intermediate_size,
                &ffn_input,
                state,
            );

            if self.config.scalars.residual != 1.0 {
                cpu::scale_inplace(&mut state.scratch.out[..hs], self.config.scalars.residual);
            }
            cpu::add_inplace(hidden, &state.scratch.out[..hs]);

            // Oracle gate: residual stream after the full layer (= llama.cpp's
            // `l_out-{i}`). All-position for early layers, last-position for the
            // final layer — the test sums vs. takes-last accordingly. Guarded so
            // the per-token `format!` allocation only happens when dumping.
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record(&format!("l_out-{i}"), hidden);
            }
        }

        cpu::rmsnorm(hidden, &self.output_norm_weight, cfg.rms_norm_eps);
        transformer::oracle_dump::record("result_norm", hidden);
        state.seq_len += 1;

        state.scratch.normed = normed;
        state.scratch.ffn_input = ffn_input;
    }

    /// Project the final hidden state to logits over the vocabulary, using the
    /// separate `output.weight` when present, else the tied embedding table.
    fn project_logits(&self, hidden: &[f32], state: &mut InferenceState) -> Vec<f32> {
        let cfg = &self.config;
        let out_ref = self.output_ref.as_ref().unwrap_or(&self.embd_ref);
        let mut logits = vec![0.0f32; cfg.vocab_size];
        #[cfg(target_arch = "aarch64")]
        {
            transformer::quantize_to_scratch(hidden, state);
            transformer::gemv_preq(
                &self.gguf,
                out_ref,
                hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut logits,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = state;
            transformer::gemv(&self.gguf, out_ref, hidden, &mut logits);
        }
        // Granite divides the logits by `logits_scaling` (identity elsewhere).
        if self.config.scalars.logit != 1.0 {
            cpu::scale_inplace(&mut logits, 1.0 / self.config.scalars.logit);
        }
        transformer::oracle_dump::record("result_output", &logits);
        logits
    }

    /// Batched-GEMM CPU prefill for the dense transformer (mirrors LFM2's CPU
    /// prefill). Reads each weight matrix once for all `n` tokens. Column-major
    /// `hidden[hs × n]` (token `j` of channel `i` at `i*n + j`). Numerically
    /// matches the per-token `forward` path. Only compiled where a batched-GEMM
    /// kernel exists (aarch64 NEON, or any target with the `blas` feature); the
    /// per-token fallback covers every other build.
    #[cfg(any(target_arch = "aarch64", feature = "blas"))]
    /// Batched-GEMM prefill. When `hidden_out` is `Some`, this captures the
    /// per-token post-final-norm hidden states into it (row-major `[n * hs]`),
    /// skips the logit projection, and returns an empty Vec — the hidden-states
    /// path. When `None`, it norms+projects the last token and returns its logits
    /// — the normal prefill path.
    fn forward_prefill_batched(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let is = cfg.intermediate_size;
        let n = tokens.len();
        let head_dim = self.head_dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let group_size = n_heads / n_kv_heads;
        // Granite overrides the softmax scale via `attention.scale`; every other
        // arch uses the default 1/sqrt(head_dim).
        let scale = cfg
            .scalars
            .attn
            .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());

        // The batched-GEMM kernels (NEON `gemm_q{4_0,8_0}_q8_0` and BLAS SGEMM
        // after `dequantize_q{4_0,8_0}_matrix`) only handle Q4_0/Q8_0. If any
        // per-layer projection uses another dtype (e.g. Q4KM/Q6K), fall back to
        // the sequential per-token path so the result stays correct.
        let all_gemmable = self.layer_refs.iter().all(|r| {
            [
                &r.attn_q,
                &r.attn_k,
                &r.attn_v,
                &r.attn_output,
                &r.ffn_gate,
                &r.ffn_up,
                &r.ffn_down,
            ]
            .iter()
            .all(|w| {
                matches!(
                    w.dtype,
                    crate::tensor::DType::Q4_0 | crate::tensor::DType::Q8_0
                )
            })
        });
        if !all_gemmable {
            // No batched kernel for these dtypes: capture per-token if requested,
            // else fall back to the sequential per-token logit path.
            if let Some(out) = hidden_out {
                *out = self.hidden_states_per_token(tokens, state);
                return Vec::new();
            }
            let mut logits = Vec::new();
            for (i, &token) in tokens.iter().enumerate() {
                logits = self.forward(&[token], start_pos + i, state);
            }
            return logits;
        }

        // Embed all tokens → column-major hidden[hs × n] (Granite embedding scale).
        let mut hidden = vec![0.0f32; hs * n];
        let mut emb_buf = vec![0.0f32; hs];
        for (j, &token_id) in tokens.iter().enumerate() {
            let token_id = token_id as usize;
            // Bound on `vocab_size` (not the possibly-padded embedding row count
            // `embd_ref.m`) so an out-of-vocab id is rejected identically to the
            // per-token `forward` path rather than silently reading a pad row.
            assert!(
                token_id < cfg.vocab_size,
                "token_id {token_id} out of range (vocab_size={})",
                cfg.vocab_size
            );
            transformer::dequantize_row_into(&self.gguf, &self.embd_ref, token_id, &mut emb_buf);
            if cfg.scalars.embedding != 1.0 {
                cpu::scale_inplace(&mut emb_buf, cfg.scalars.embedding);
            }
            for i in 0..hs {
                hidden[i * n + j] = emb_buf[i];
            }
        }

        // Per-layer buffers (reused across layers).
        let mut normed = vec![0.0f32; hs * n];
        let mut block_out = vec![0.0f32; hs * n];
        let mut ffn_input = vec![0.0f32; hs * n];
        let mut ffn_out = vec![0.0f32; hs * n];
        let mut norm_col = vec![0.0f32; hs];
        let mut ffn_col = vec![0.0f32; hs];
        let mut q_mat = vec![0.0f32; q_dim * n];
        let mut k_mat = vec![0.0f32; kv_dim * n];
        let mut v_mat = vec![0.0f32; kv_dim * n];
        let mut out_proj_input = vec![0.0f32; q_dim * n];
        let mut gate_mat = vec![0.0f32; is * n];
        let mut up_mat = vec![0.0f32; is * n];

        // NEON-fallback Q8_0 input scratch. One buffer set sized to the largest
        // GEMM k-dim (hs, q_dim, or is) — each quantize call is immediately
        // followed by its paired GEMM with the same k, so reuse is safe.
        #[cfg(not(feature = "blas"))]
        let max_dim = hs.max(q_dim).max(is);
        #[cfg(not(feature = "blas"))]
        let mut col = vec![0.0f32; max_dim];
        #[cfg(not(feature = "blas"))]
        let mut bq_scales = vec![0.0f32; n * (max_dim / 32)];
        #[cfg(not(feature = "blas"))]
        let mut bq_quants = vec![0i8; n * max_dim];

        // Flash attention (tiled + rayon) beats the naive per-token loop only for
        // longer prompts; below the threshold its two-pass online-softmax overhead
        // loses. Mirrors LFM2's measured crossover (~pp256 on Apple Silicon).
        const FLASH_ATTN_THRESHOLD: usize = 256;
        let use_flash = n >= FLASH_ATTN_THRESHOLD;
        // Per-KV-head attention output, [n_kv_heads][group_size * n * head_dim],
        // scattered back into out_proj_input after the flash pass. Reused across
        // layers; empty (unused) below the threshold.
        let mut flash_out = if use_flash {
            vec![0.0f32; n_heads * n * head_dim]
        } else {
            Vec::new()
        };

        for layer in 0..cfg.n_layers {
            let refs = &self.layer_refs[layer];

            // Attention pre-norm: rmsnorm each column.
            for j in 0..n {
                for i in 0..hs {
                    norm_col[i] = hidden[i * n + j];
                }
                cpu::rmsnorm(
                    &mut norm_col,
                    &self.attn_norm_weights[layer],
                    cfg.rms_norm_eps,
                );
                for i in 0..hs {
                    normed[i * n + j] = norm_col[i];
                }
            }

            // Batched Q/K/V projections (weight [m×hs] × normed[hs×n] → [m×n]).
            #[cfg(feature = "blas")]
            {
                transformer::try_blas_prefill_gemm(
                    &self.gguf,
                    &refs.attn_q,
                    &normed,
                    &mut q_mat,
                    q_dim,
                    n,
                    hs,
                    &mut state.scratch.dequant_weight_scratch,
                );
                transformer::try_blas_prefill_gemm(
                    &self.gguf,
                    &refs.attn_k,
                    &normed,
                    &mut k_mat,
                    kv_dim,
                    n,
                    hs,
                    &mut state.scratch.dequant_weight_scratch,
                );
                transformer::try_blas_prefill_gemm(
                    &self.gguf,
                    &refs.attn_v,
                    &normed,
                    &mut v_mat,
                    kv_dim,
                    n,
                    hs,
                    &mut state.scratch.dequant_weight_scratch,
                );
            }
            #[cfg(not(feature = "blas"))]
            {
                transformer::quantize_columns(
                    &normed,
                    hs,
                    n,
                    &mut col,
                    &mut bq_scales,
                    &mut bq_quants,
                );
                transformer::gemm_preq(
                    &self.gguf,
                    &refs.attn_q,
                    &bq_scales,
                    &bq_quants,
                    &mut q_mat,
                    q_dim,
                    n,
                    hs,
                );
                transformer::gemm_preq(
                    &self.gguf,
                    &refs.attn_k,
                    &bq_scales,
                    &bq_quants,
                    &mut k_mat,
                    kv_dim,
                    n,
                    hs,
                );
                transformer::gemm_preq(
                    &self.gguf,
                    &refs.attn_v,
                    &bq_scales,
                    &bq_quants,
                    &mut v_mat,
                    kv_dim,
                    n,
                    hs,
                );
            }

            // Per-arch attention knobs (constant across tokens within a layer).
            let qkv_bias = match (
                self.attn_q_bias[layer].as_deref(),
                self.attn_k_bias[layer].as_deref(),
                self.attn_v_bias[layer].as_deref(),
            ) {
                (Some(q), Some(k), Some(v)) => Some((q, k, v)),
                _ => None,
            };
            let qk_norm = match (
                self.attn_q_norm_weights[layer].as_deref(),
                self.attn_k_norm_weights[layer].as_deref(),
            ) {
                (Some(q), Some(k)) => Some((q, k)),
                _ => None,
            };

            // Pass A: per token, bias → QK-norm → RoPE → stash post-RoPE Q back
            // into q_mat (so the attention pass can read every query) → append
            // K/V to the f32 cache. Destructure the cache once (not per token)
            // and reserve the whole prompt's growth up front (matches lfm2) so
            // the per-token extend_from_slice doesn't repeatedly reallocate.
            let (key_cache, value_cache) = match &mut state.layers[layer] {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    ..
                } => (key_cache, value_cache),
                _ => unreachable!("dense transformer layer is always Attention"),
            };
            key_cache.reserve(n * kv_dim);
            value_cache.reserve(n * kv_dim);
            for j in 0..n {
                let pos = start_pos + j;
                let q = &mut state.scratch.q[..q_dim];
                let k = &mut state.scratch.k[..kv_dim];
                let v = &mut state.scratch.v[..kv_dim];
                for i in 0..q_dim {
                    q[i] = q_mat[i * n + j];
                }
                for i in 0..kv_dim {
                    k[i] = k_mat[i * n + j];
                    v[i] = v_mat[i * n + j];
                }

                // Qwen2 Q/K/V bias.
                if let Some((q_bias, k_bias, v_bias)) = qkv_bias {
                    cpu::add_inplace(q, q_bias);
                    cpu::add_inplace(k, k_bias);
                    cpu::add_inplace(v, v_bias);
                }

                // Qwen3 per-head QK-norm — BEFORE RoPE.
                if let Some((q_norm, k_norm)) = qk_norm {
                    for h in 0..n_heads {
                        cpu::rmsnorm(
                            &mut q[h * head_dim..(h + 1) * head_dim],
                            q_norm,
                            cfg.rms_norm_eps,
                        );
                    }
                    for h in 0..n_kv_heads {
                        cpu::rmsnorm(
                            &mut k[h * head_dim..(h + 1) * head_dim],
                            k_norm,
                            cfg.rms_norm_eps,
                        );
                    }
                }

                // RoPE — layout per arch (NEOX for Qwen, NORM for LLaMA/Granite).
                match self.rope_type {
                    RopeType::Neox => {
                        cpu::rope(q, k, pos, n_heads, n_kv_heads, head_dim, cfg.rope_theta)
                    }
                    RopeType::Norm => cpu::rope_norm(
                        q,
                        k,
                        pos,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        cfg.rope_theta,
                        self.rope_freqs.as_deref(),
                    ),
                }

                // Stash post-RoPE Q back into q_mat for the attention pass.
                for i in 0..q_dim {
                    q_mat[i * n + j] = q[i];
                }

                // Append K, V to the f32 cache (destructured once above the loop).
                key_cache.extend_from_slice(&state.scratch.k[..kv_dim]);
                value_cache.extend_from_slice(&state.scratch.v[..kv_dim]);
            }

            // Pass B: GQA attention over the now-complete KV cache → out_proj_input.
            let (k_cache, v_cache) = match &state.layers[layer] {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    ..
                } => (key_cache.as_slice(), value_cache.as_slice()),
                _ => unreachable!("dense transformer layer is always Attention"),
            };
            if use_flash {
                // Flash attention (tiled + rayon), parallel across KV heads. Each
                // KV head writes a contiguous [group_size * n * head_dim] chunk;
                // par_chunks_mut hands out disjoint &mut so there's no aliasing.
                let chunk_size = group_size * n * head_dim;
                let flash_buf = &mut flash_out[..n_kv_heads * chunk_size];
                let q_ref = &q_mat[..];
                #[cfg_attr(not(feature = "parallel"), allow(unused_imports))]
                use crate::par::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
                flash_buf
                    .par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(kv_h, chunk)| {
                        cpu::flash_attention_gqa_cpu(
                            q_ref,
                            k_cache,
                            v_cache,
                            chunk,
                            kv_h * group_size,
                            group_size,
                            n,
                            n,
                            kv_dim,
                            kv_h * head_dim,
                            head_dim,
                            scale,
                            start_pos,
                        );
                    });
                // Scatter flash_out [n_heads, n, head_dim] → out_proj_input [q_dim,
                // n] (stride-n columns). d-then-j inner order keeps out writes
                // sequential (stride 1) with small-stride reads from flash_buf.
                for kv_h in 0..n_kv_heads {
                    for g in 0..group_size {
                        let h = kv_h * group_size + g;
                        let src_base = kv_h * chunk_size + g * n * head_dim;
                        for d in 0..head_dim {
                            let row_idx = (h * head_dim + d) * n;
                            for j in 0..n {
                                out_proj_input[row_idx + j] =
                                    flash_buf[src_base + j * head_dim + d];
                            }
                        }
                    }
                }
            } else {
                // Naive per-token attention: token j attends over cache[0..pos+1]
                // (causal). Bit-identical to the per-token `forward` path.
                let attn_out = &mut state.scratch.attn_out[..q_dim];
                let q = &mut state.scratch.q[..q_dim];
                let scores = &mut state.scratch.scores;
                for j in 0..n {
                    let seq_len = start_pos + j + 1;
                    for i in 0..q_dim {
                        q[i] = q_mat[i * n + j];
                    }
                    scores.resize(seq_len, 0.0);
                    for h in 0..n_heads {
                        let kv_h = h / group_size;
                        let q_head = &q[h * head_dim..(h + 1) * head_dim];
                        let kv_h_offset = kv_h * head_dim;
                        cpu::attn_scores(
                            q_head,
                            k_cache,
                            scores,
                            kv_dim,
                            kv_h_offset,
                            head_dim,
                            scale,
                            seq_len,
                        );
                        cpu::softmax_inplace(scores);
                        cpu::attn_values(
                            scores,
                            v_cache,
                            &mut attn_out[h * head_dim..(h + 1) * head_dim],
                            kv_dim,
                            kv_h_offset,
                            head_dim,
                            seq_len,
                        );
                    }
                    for i in 0..q_dim {
                        out_proj_input[i * n + j] = attn_out[i];
                    }
                }
            }

            // Batched output projection GEMM → block_out[hs × n] (k = q_dim).
            #[cfg(feature = "blas")]
            {
                transformer::try_blas_prefill_gemm(
                    &self.gguf,
                    &refs.attn_output,
                    &out_proj_input,
                    &mut block_out,
                    hs,
                    n,
                    q_dim,
                    &mut state.scratch.dequant_weight_scratch,
                );
            }
            #[cfg(not(feature = "blas"))]
            {
                transformer::quantize_columns(
                    &out_proj_input,
                    q_dim,
                    n,
                    &mut col,
                    &mut bq_scales,
                    &mut bq_quants,
                );
                transformer::gemm_preq(
                    &self.gguf,
                    &refs.attn_output,
                    &bq_scales,
                    &bq_quants,
                    &mut block_out,
                    hs,
                    n,
                    q_dim,
                );
            }

            // Granite residual scale, then residual add into hidden.
            if cfg.scalars.residual != 1.0 {
                cpu::scale_inplace(&mut block_out, cfg.scalars.residual);
            }
            cpu::add_inplace(&mut hidden, &block_out);

            // FFN pre-norm: rmsnorm each column.
            for j in 0..n {
                for i in 0..hs {
                    ffn_col[i] = hidden[i * n + j];
                }
                cpu::rmsnorm(
                    &mut ffn_col,
                    &self.ffn_norm_weights[layer],
                    cfg.rms_norm_eps,
                );
                for i in 0..hs {
                    ffn_input[i * n + j] = ffn_col[i];
                }
            }

            // FFN gate/up GEMM → silu(gate)⊙up → down GEMM.
            #[cfg(feature = "blas")]
            {
                transformer::try_blas_prefill_gemm(
                    &self.gguf,
                    &refs.ffn_gate,
                    &ffn_input,
                    &mut gate_mat,
                    is,
                    n,
                    hs,
                    &mut state.scratch.dequant_weight_scratch,
                );
                transformer::try_blas_prefill_gemm(
                    &self.gguf,
                    &refs.ffn_up,
                    &ffn_input,
                    &mut up_mat,
                    is,
                    n,
                    hs,
                    &mut state.scratch.dequant_weight_scratch,
                );
            }
            #[cfg(not(feature = "blas"))]
            {
                transformer::quantize_columns(
                    &ffn_input,
                    hs,
                    n,
                    &mut col,
                    &mut bq_scales,
                    &mut bq_quants,
                );
                transformer::gemm_preq(
                    &self.gguf,
                    &refs.ffn_gate,
                    &bq_scales,
                    &bq_quants,
                    &mut gate_mat,
                    is,
                    n,
                    hs,
                );
                transformer::gemm_preq(
                    &self.gguf,
                    &refs.ffn_up,
                    &bq_scales,
                    &bq_quants,
                    &mut up_mat,
                    is,
                    n,
                    hs,
                );
            }

            cpu::silu_mul_inplace(&mut gate_mat[..is * n], &up_mat[..is * n]);

            #[cfg(feature = "blas")]
            {
                transformer::try_blas_prefill_gemm(
                    &self.gguf,
                    &refs.ffn_down,
                    &gate_mat,
                    &mut ffn_out,
                    hs,
                    n,
                    is,
                    &mut state.scratch.dequant_weight_scratch,
                );
            }
            #[cfg(not(feature = "blas"))]
            {
                transformer::quantize_columns(
                    &gate_mat,
                    is,
                    n,
                    &mut col,
                    &mut bq_scales,
                    &mut bq_quants,
                );
                transformer::gemm_preq(
                    &self.gguf,
                    &refs.ffn_down,
                    &bq_scales,
                    &bq_quants,
                    &mut ffn_out,
                    hs,
                    n,
                    is,
                );
            }

            // Granite residual scale, then residual add.
            if cfg.scalars.residual != 1.0 {
                cpu::scale_inplace(&mut ffn_out, cfg.scalars.residual);
            }
            cpu::add_inplace(&mut hidden, &ffn_out);
        }

        // Advance seq_len (the block loops appended KV cells without bumping it).
        state.seq_len = start_pos + n;

        // Hidden-states capture: final-norm EVERY column into a row-major
        // `[n * hs]` buffer (post-final-RMSNorm = llama.cpp `result_norm`),
        // skipping the logit projection. Reuses `norm_col` as per-column scratch.
        if let Some(out) = hidden_out {
            out.clear();
            out.reserve(n * hs);
            for j in 0..n {
                for i in 0..hs {
                    norm_col[i] = hidden[i * n + j];
                }
                cpu::rmsnorm(&mut norm_col, &self.output_norm_weight, cfg.rms_norm_eps);
                out.extend_from_slice(&norm_col);
            }
            return Vec::new();
        }

        // Final norm on the LAST column, then project last-token logits (what the
        // decode loop consumes). Reuse `norm_col` (an hs-length scratch that's
        // dead after the layer loop) rather than allocating. `project_logits`
        // handles the Granite logit scale and the aarch64 pre-quantized GEMV.
        for i in 0..hs {
            norm_col[i] = hidden[i * n + (n - 1)];
        }
        cpu::rmsnorm(&mut norm_col, &self.output_norm_weight, cfg.rms_norm_eps);
        self.project_logits(&norm_col, state)
    }

    /// Per-token hidden-states fallback: embed → `run_layers` (which applies the
    /// final RMSNorm) per token, concatenated row-major `[n * hidden_size]`.
    /// Post-final-norm, matching the batched capture path. Used when there's no
    /// batched-GEMM kernel (`n == 1`, non-gemmable dtypes, or non-aarch64/non-blas).
    /// Assumes `state` starts cleared at position 0.
    fn hidden_states_per_token(&self, tokens: &[u32], state: &mut InferenceState) -> Vec<f32> {
        let hs = self.config.hidden_size;
        let mut out = Vec::with_capacity(tokens.len() * hs);
        // Reuse one embedding buffer across tokens (`dequantize_row_into`) instead
        // of allocating a fresh Vec per token.
        let mut hidden = vec![0.0f32; hs];
        for &token in tokens {
            let token_id = token as usize;
            assert!(
                token_id < self.config.vocab_size,
                "token_id {token_id} out of range (vocab_size={})",
                self.config.vocab_size
            );
            transformer::dequantize_row_into(&self.gguf, &self.embd_ref, token_id, &mut hidden);
            if self.config.scalars.embedding != 1.0 {
                cpu::scale_inplace(&mut hidden, self.config.scalars.embedding);
            }
            // `run_layers` ropes at `pos` and appends one KV cell, bumping
            // seq_len; starting from a cleared state walks positions 0..n.
            let pos = state.seq_len;
            self.run_layers(&mut hidden, pos, state);
            out.extend_from_slice(&hidden);
        }
        out
    }
}

impl Model for LlamaModel {
    fn supports_hidden_states(&self) -> bool {
        true
    }

    fn hidden_states(&self, tokens: &[u32], state: &mut InferenceState) -> Vec<f32> {
        assert!(
            !tokens.is_empty(),
            "hidden_states requires at least one token"
        );
        // Batched-GEMM capture when a batched kernel exists and n > 1; the
        // batched path internally falls back to per-token for non-gemmable dtypes.
        #[cfg(any(target_arch = "aarch64", feature = "blas"))]
        if tokens.len() > 1 {
            let mut out = Vec::new();
            self.forward_prefill_batched(tokens, 0, state, Some(&mut out));
            return out;
        }
        self.hidden_states_per_token(tokens, state)
    }

    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(tokens.len(), 1, "LlamaModel forward expects single token");
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        assert!(
            token_id < cfg.vocab_size,
            "token_id {token_id} out of range (vocab_size={})",
            cfg.vocab_size
        );

        let mut hidden = transformer::dequantize_row(&self.gguf, &self.embd_ref, token_id);
        if self.config.scalars.embedding != 1.0 {
            cpu::scale_inplace(&mut hidden, self.config.scalars.embedding);
        }
        // Record after the embedding scale: llama.cpp fires its "embd" callback
        // post-scale, so the dumped node is GET_ROWS for plain archs (scale=1) and
        // SCALE for Granite. Either way the value matches.
        transformer::oracle_dump::record("embd", &hidden);
        self.run_layers(&mut hidden, pos, state);
        self.project_logits(&hidden, state)
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        assert!(
            !tokens.is_empty(),
            "forward_prefill requires at least one token"
        );
        // Each `forward` appends one K/V cell and advances `seq_len`, so the
        // rope position of token `i` must equal the current cache length. That
        // holds only when `start_pos` lines up with the existing cache — enforce
        // it so a mismatched snapshot/prefix-cache restore fails loudly here
        // rather than drifting into a later KV-shift panic.
        assert_eq!(
            start_pos, state.seq_len,
            "forward_prefill: start_pos ({start_pos}) must equal state.seq_len ({})",
            state.seq_len
        );
        // Batched-GEMM prefill (reads each weight once for all N tokens) on
        // targets that have a batched kernel — aarch64 NEON or any `blas` build.
        // `n == 1` stays on the per-token path to avoid GEMM setup overhead, and
        // every other target has no batched kernel, so it also falls through.
        // When the oracle-dump harness is collecting, fall back to the per-token
        // path too: the batched path bypasses `run_layers` and so emits none of
        // the per-substep `oracle_dump::record` nodes that `tests/oracle_text.rs`
        // validates against llama.cpp.
        #[cfg(any(target_arch = "aarch64", feature = "blas"))]
        if tokens.len() > 1 && !transformer::oracle_dump::is_active() {
            return self.forward_prefill_batched(tokens, start_pos, state, None);
        }

        // Sequential per-token prefill (single-token, or no batched kernel).
        let mut logits = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            logits = self.forward(&[token], start_pos + i, state);
        }
        logits
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn supports_kv_shift(&self) -> bool {
        true
    }

    fn shift_kv(&self, state: &mut InferenceState, n_keep: usize, shift: usize) {
        state.shift_kv_with_rope(
            n_keep,
            shift,
            self.config.rope_theta,
            self.head_dim,
            &self.config.kv_heads_per_layer,
            self.rope_type,
            self.rope_freqs.as_deref(),
        );
    }
}

// ── GPU weight source ───────────────────────────────────────────────────────
//
// Lets the wgpu loader (`gpu_lfm2.rs`) upload a dense transformer the same way
// it uploads LFM2. Every layer is attention (no conv refs); QK-norm / QKV-bias
// / untied-output / Llama-3 freq-factors are surfaced per-arch via the `Option`
// accessors. Granite scalars ride on `config().scalars`.
#[cfg(any(feature = "gpu", all(feature = "metal", target_os = "macos")))]
impl crate::model::gpu_weight_source::GpuWeightSource for LlamaModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    fn output_norm_weight(&self) -> &[f32] {
        &self.output_norm_weight
    }
    fn attn_norm_weight(&self, layer: usize) -> &[f32] {
        &self.attn_norm_weights[layer]
    }
    fn ffn_norm_weight(&self, layer: usize) -> &[f32] {
        &self.ffn_norm_weights[layer]
    }
    fn attn_q_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.attn_q_norm_weights[layer].as_deref()
    }
    fn attn_k_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.attn_k_norm_weights[layer].as_deref()
    }
    fn conv_weight(&self, _layer: usize) -> Option<&[f32]> {
        None
    }
    fn attn_q_bias(&self, layer: usize) -> Option<&[f32]> {
        self.attn_q_bias[layer].as_deref()
    }
    fn attn_k_bias(&self, layer: usize) -> Option<&[f32]> {
        self.attn_k_bias[layer].as_deref()
    }
    fn attn_v_bias(&self, layer: usize) -> Option<&[f32]> {
        self.attn_v_bias[layer].as_deref()
    }
    fn rope_freqs(&self) -> Option<&[f32]> {
        self.rope_freqs.as_deref()
    }

    fn weight_bytes(&self, wref: &WeightRef) -> &[u8] {
        transformer::weight_data(&self.gguf, wref)
    }
    fn dequantize_weight(&self, wref: &WeightRef) -> Vec<f32> {
        transformer::dequantize_weight(&self.gguf, wref)
    }

    fn output_ref(&self) -> Option<&WeightRef> {
        self.output_ref.as_ref()
    }
    fn ffn_gate_ref(&self, layer: usize) -> &WeightRef {
        &self.layer_refs[layer].ffn_gate
    }
    fn ffn_up_ref(&self, layer: usize) -> &WeightRef {
        &self.layer_refs[layer].ffn_up
    }
    fn ffn_down_ref(&self, layer: usize) -> &WeightRef {
        &self.layer_refs[layer].ffn_down
    }
    fn conv_in_proj_ref(&self, _layer: usize) -> Option<&WeightRef> {
        None
    }
    fn conv_out_proj_ref(&self, _layer: usize) -> Option<&WeightRef> {
        None
    }
    fn attn_q_ref(&self, layer: usize) -> Option<&WeightRef> {
        Some(&self.layer_refs[layer].attn_q)
    }
    fn attn_k_ref(&self, layer: usize) -> Option<&WeightRef> {
        Some(&self.layer_refs[layer].attn_k)
    }
    fn attn_v_ref(&self, layer: usize) -> Option<&WeightRef> {
        Some(&self.layer_refs[layer].attn_v)
    }
    fn attn_output_ref(&self, layer: usize) -> Option<&WeightRef> {
        Some(&self.layer_refs[layer].attn_output)
    }

    fn rope_type(&self) -> RopeType {
        self.rope_type
    }
    fn supports_batched_prefill(&self) -> bool {
        // The batched wgpu prefill path now generalizes every dense-transformer
        // feature the per-token decode loop handles: `rope_type` (NEOX/NORM),
        // Llama-3 `freq_factors`, optional QK-norm, Qwen2 QKV bias, Qwen3
        // decoupled head_dim, Granite scalars (embedding/residual/attention/
        // logit), and untied output. Correctness is gated by the GPU-internal
        // differential test (batched vs per-token, all four archs) in
        // `tests/gpu_transformer_parity.rs`.
        true
    }
}
