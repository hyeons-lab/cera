//! Ling 3.0 Tiny (bailingmoe3) hybrid linear, latent attention, and MoE model implementation.
//!
//! Supports:
//! - `bailingmoe3` / `bailingmoe`: BailingMoE 3 (e.g. Ling 3.0 Tiny, Ling 3.0 Flash)
//!   combining Kernel Delta Attention (KDA) recurrent layers, Multi-head Latent
//!   Attention (MLA) full attention layers, and Mixture-of-Experts (MoE) with
//!   parallel shared experts.

use anyhow::{Context, Result, ensure};

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, LayerState};
use crate::model::transformer::{self, WeightRef};
use crate::model::{BlockType, Model, ModelConfig, SsmConfig};

// ── Per-layer weight references ─────────────────────────────────────────────

pub(crate) struct KdaLayerRefs {
    pub(crate) wq: WeightRef,
    pub(crate) wk: WeightRef,
    pub(crate) wv: WeightRef,
    pub(crate) ssm_conv1d_q: Vec<f32>,
    pub(crate) ssm_conv1d_k: Vec<f32>,
    pub(crate) ssm_conv1d_v: Vec<f32>,
    pub(crate) ssm_f_a: WeightRef,
    pub(crate) ssm_dt_bias: Vec<f32>,
    pub(crate) ssm_a: Vec<f32>,
    pub(crate) ssm_beta: WeightRef,
    pub(crate) ssm_g_a: WeightRef,
    pub(crate) ssm_norm: Vec<f32>,
    pub(crate) wo: WeightRef,
}

pub(crate) enum MlaQueryRefs {
    Decomposed {
        wq_a: WeightRef,
        q_a_norm: Vec<f32>,
        wq_b: WeightRef,
    },
    Direct(WeightRef),
}

pub(crate) struct MlaLayerRefs {
    pub(crate) q_proj: MlaQueryRefs,
    pub(crate) wkv_a_mqa: WeightRef,
    pub(crate) kv_a_norm: Vec<f32>,
    pub(crate) wk_b: Vec<f32>,
    pub(crate) wv_b: Vec<f32>,
    pub(crate) wqkv_gate: WeightRef,
    pub(crate) wo: WeightRef,
}

pub(crate) enum LayerKindRefs {
    Kda(KdaLayerRefs),
    Mla(MlaLayerRefs),
}

pub(crate) struct DenseFfnRefs {
    pub(crate) gate: WeightRef,
    pub(crate) up: WeightRef,
    pub(crate) down: WeightRef,
}

pub(crate) struct MoeFfnRefs {
    pub(crate) router: WeightRef,
    pub(crate) exp_probs_b: Vec<f32>,
    pub(crate) gate_exps: Vec<WeightRef>,
    pub(crate) up_exps: Vec<WeightRef>,
    pub(crate) down_exps: Vec<WeightRef>,
    pub(crate) gate_shexp: WeightRef,
    pub(crate) up_shexp: WeightRef,
    pub(crate) down_shexp: WeightRef,
    pub(crate) n_expert: usize,
    pub(crate) n_expert_used: usize,
    pub(crate) routed_scaling_factor: f32,
}

pub(crate) enum FfnKindRefs {
    Dense(DenseFfnRefs),
    Moe(MoeFfnRefs),
}

pub(crate) struct LayerRefs {
    pub(crate) kind: LayerKindRefs,
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) ffn: FfnKindRefs,
    pub(crate) ffn_norm: Vec<f32>,
}

// ── BailingMoE 3 Model ──────────────────────────────────────────────────────

pub struct BailingMoe3Model {
    gguf: GgufFile,
    config: ModelConfig,
    head_dim: usize,
    n_heads: usize,
    kda_d_conv: usize,
    kda_lower_bound: f32,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    kq_scale: f32,
    embd_ref: WeightRef,
    output_norm_weight: Vec<f32>,
    output_ref: Option<WeightRef>,
    layers: Vec<LayerRefs>,
    #[allow(dead_code)]
    model_id: String,
}

impl BailingMoe3Model {
    pub fn from_gguf(gguf: GgufFile, context_size: usize) -> Result<Self> {
        Self::from_gguf_with_id(gguf, context_size, String::new())
    }

    pub fn from_gguf_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        let arch = gguf
            .get_str("general.architecture")
            .context("missing general.architecture in GGUF")?
            .to_string();

        ensure!(
            arch == "bailingmoe3" || arch == "bailingmoe" || arch == "bailingmoe2",
            "expected architecture 'bailingmoe', 'bailingmoe2', or 'bailingmoe3', got '{arch}'"
        );

        let prefix = ["bailingmoe3", "bailingmoe2", "bailingmoe", arch.as_str()]
            .into_iter()
            .find(|p| gguf.get_u32(&format!("{p}.block_count")).is_some())
            .unwrap_or(arch.as_str());

        let n_layers = gguf
            .get_u32(&format!("{prefix}.block_count"))
            .context("missing block_count")? as usize;
        let hidden_size = gguf
            .get_u32(&format!("{prefix}.embedding_length"))
            .context("missing embedding_length")? as usize;
        let intermediate_size = gguf
            .get_u32(&format!("{prefix}.feed_forward_length"))
            .context("missing feed_forward_length")? as usize;
        let n_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count"))
            .context("missing attention.head_count")? as usize;

        ensure!(n_layers > 0, "block_count must be positive");
        ensure!(
            n_layers <= 512,
            "block_count exceeds reasonable limit: {n_layers}"
        );
        ensure!(hidden_size > 0, "embedding_length must be positive");
        ensure!(
            intermediate_size > 0,
            "feed_forward_length must be positive"
        );
        ensure!(n_heads > 0, "attention.head_count must be positive");

        let file_context_size = gguf
            .get_u32(&format!("{prefix}.context_length"))
            .map(|v| v as usize);
        let context_size = match (file_context_size, context_size) {
            (Some(fc), cs) if cs > 0 => fc.min(cs),
            (Some(fc), _) => fc,
            (None, cs) => cs,
        };
        ensure!(context_size > 0, "context_size must be positive");

        let rms_norm_eps = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        ensure!(
            rms_norm_eps.is_finite() && rms_norm_eps > 0.0,
            "layer_norm_rms_epsilon must be positive and finite"
        );

        let kda_head_dim = gguf
            .get_u32(&format!("{prefix}.kda.head_dim"))
            .unwrap_or(128) as usize;
        let kda_d_conv = gguf
            .get_u32(&format!("{prefix}.ssm.conv_kernel"))
            .unwrap_or(4) as usize;
        let kda_lower_bound = gguf
            .get_f32(&format!("{prefix}.kda.gate_lower_bound"))
            .unwrap_or(-5.0);

        ensure!(kda_head_dim > 0, "kda.head_dim must be positive");
        ensure!(kda_d_conv >= 1, "ssm.conv_kernel must be >= 1");
        ensure!(
            kda_lower_bound.is_finite() && kda_lower_bound < 0.0,
            "kda.gate_lower_bound must be negative and finite"
        );

        let kv_lora_rank = gguf
            .get_u32(&format!("{prefix}.attention.kv_lora_rank"))
            .unwrap_or(512) as usize;
        let q_lora_rank = gguf
            .get_u32(&format!("{prefix}.attention.q_lora_rank"))
            .map(|v| v as usize);
        let qk_rope_head_dim = gguf
            .get_u32(&format!("{prefix}.rope.dimension_count"))
            .unwrap_or(64) as usize;
        let qk_head_dim = gguf
            .get_u32(&format!("{prefix}.attention.key_length_mla"))
            .unwrap_or(192) as usize;
        let v_head_dim = gguf
            .get_u32(&format!("{prefix}.attention.value_length_mla"))
            .unwrap_or(128) as usize;
        let rope_theta = gguf
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(6000000.0);

        ensure!(kv_lora_rank > 0, "kv_lora_rank must be positive");
        ensure!(
            qk_rope_head_dim > 0,
            "rope.dimension_count must be positive"
        );
        ensure!(
            qk_rope_head_dim.is_multiple_of(2),
            "rope.dimension_count must be even for rotary pairs"
        );
        ensure!(
            qk_head_dim > qk_rope_head_dim,
            "key_length_mla must be strictly greater than rope.dimension_count"
        );
        let qk_nope_head_dim = qk_head_dim - qk_rope_head_dim;
        ensure!(v_head_dim > 0, "value_length_mla must be positive");
        ensure!(
            rope_theta.is_finite() && rope_theta > 0.0,
            "rope.freq_base must be positive and finite"
        );

        let leading_dense_block_count = gguf
            .get_u32(&format!("{prefix}.leading_dense_block_count"))
            .unwrap_or(1) as usize;
        let n_expert = gguf.get_u32(&format!("{prefix}.expert_count")).unwrap_or(0) as usize;
        let n_expert_used = gguf
            .get_u32(&format!("{prefix}.expert_used_count"))
            .unwrap_or(0) as usize;
        let moe_intermediate_size = gguf
            .get_u32(&format!("{prefix}.expert_feed_forward_length"))
            .unwrap_or(512) as usize;
        let routed_scaling_factor = gguf
            .get_f32(&format!("{prefix}.expert_weights_scale"))
            .unwrap_or(1.0);
        ensure!(
            routed_scaling_factor.is_finite() && routed_scaling_factor > 0.0,
            "expert_weights_scale must be positive and finite"
        );

        if n_layers > leading_dense_block_count {
            ensure!(n_expert > 0, "expert_count must be positive for MoE layers");
            ensure!(
                n_expert_used > 0 && n_expert_used <= n_expert,
                "expert_used_count ({n_expert_used}) must be in 1..={n_expert}"
            );
            ensure!(
                moe_intermediate_size > 0,
                "expert_feed_forward_length must be positive"
            );
        }

        // Determine recurrent vs full-attention layers from head_count_kv
        let head_count_kv: Vec<usize> =
            if let Some(arr) = gguf.get_i32_array(&format!("{prefix}.attention.head_count_kv")) {
                arr.into_iter().map(|v| v.max(0) as usize).collect()
            } else {
                // Default Ling 3.0 schedule: every 4th layer is MLA full-attention
                (0..n_layers)
                    .map(|il| if (il + 1) % 4 == 0 { 1 } else { 0 })
                    .collect()
            };
        ensure!(
            head_count_kv.len() >= n_layers,
            "head_count_kv has {} entries, expected at least {n_layers}",
            head_count_kv.len()
        );

        let mut block_types = Vec::with_capacity(n_layers);
        let mut kv_heads_per_layer = Vec::with_capacity(n_layers);
        for &kv_heads in &head_count_kv[..n_layers] {
            if kv_heads == 0 {
                block_types.push(BlockType::DeltaNet);
                kv_heads_per_layer.push(0);
            } else {
                ensure!(
                    kv_heads == 1,
                    "bailingmoe3 MLA attention requires head_count_kv == 1, got {kv_heads}"
                );
                block_types.push(BlockType::Attention);
                kv_heads_per_layer.push(1);
            }
        }

        let d_inner_kda = n_heads
            .checked_mul(kda_head_dim)
            .context("d_inner_kda arithmetic overflow")?;
        let ssm_config = SsmConfig {
            d_conv: kda_d_conv,
            d_inner: d_inner_kda,
            d_state: kda_head_dim,
            dt_rank: n_heads,
            n_group: n_heads,
        };

        // Cache dimension in ModelConfig: must accommodate max(MLA key_length, KDA head_dim)
        let mla_key_dim = kv_lora_rank
            .checked_add(qk_rope_head_dim)
            .context("mla_key_dim arithmetic overflow")?;
        let cache_head_dim = mla_key_dim.max(kda_head_dim);

        let embd_tensor = gguf
            .get_tensor("token_embd.weight")
            .context("missing token_embd.weight")?;
        let embd_shape = embd_tensor.shape();
        ensure!(
            embd_shape.len() >= 2 && embd_shape[0] == hidden_size,
            "invalid token_embd.weight shape: {embd_shape:?}, expected [{hidden_size}, vocab_size]"
        );
        let vocab_size = embd_shape[1];
        ensure!(
            vocab_size > 0,
            "token_embd.weight vocab_size must be positive"
        );
        let embd_ref = transformer::resolve_weight(&gguf, "token_embd.weight")?;

        let output_norm_weight = gguf
            .get_tensor("output_norm.weight")
            .context("missing output_norm.weight")?
            .try_to_f32_vec()
            .context("failed to dequantize output_norm.weight")?;
        ensure!(
            output_norm_weight.len() == hidden_size,
            "output_norm.weight len ({}) does not match hidden_size ({hidden_size})",
            output_norm_weight.len()
        );

        let output_ref = if gguf.tensors.contains_key("output.weight") {
            let out_tensor = gguf
                .get_tensor("output.weight")
                .context("missing output.weight")?;
            let out_shape = out_tensor.shape();
            ensure!(
                out_shape.len() >= 2 && out_shape[0] == hidden_size && out_shape[1] == vocab_size,
                "invalid output.weight shape: {out_shape:?}, expected [{hidden_size}, {vocab_size}]"
            );
            Some(transformer::resolve_weight(&gguf, "output.weight")?)
        } else {
            None
        };

        // Resolve layers
        let mut layers = Vec::with_capacity(n_layers);
        for (il, &kv_heads) in head_count_kv.iter().enumerate().take(n_layers) {
            let attn_norm = gguf
                .get_tensor(&format!("blk.{il}.attn_norm.weight"))
                .with_context(|| format!("missing blk.{il}.attn_norm.weight"))?
                .try_to_f32_vec()
                .with_context(|| format!("failed to dequantize blk.{il}.attn_norm.weight"))?;
            ensure!(
                attn_norm.len() == hidden_size,
                "layer {il} attn_norm len mismatch"
            );

            let ffn_norm = gguf
                .get_tensor(&format!("blk.{il}.ffn_norm.weight"))
                .with_context(|| format!("missing blk.{il}.ffn_norm.weight"))?
                .try_to_f32_vec()
                .with_context(|| format!("failed to dequantize blk.{il}.ffn_norm.weight"))?;
            ensure!(
                ffn_norm.len() == hidden_size,
                "layer {il} ffn_norm len mismatch"
            );

            let kind = if kv_heads == 0 {
                // KDA Recurrent layer
                let wq = transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_q.weight"))?;
                let wk = transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_k.weight"))?;
                let wv = transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_v.weight"))?;
                ensure!(
                    wq.m == d_inner_kda
                        && wq.k == hidden_size
                        && wk.m == d_inner_kda
                        && wk.k == hidden_size
                        && wv.m == d_inner_kda
                        && wv.k == hidden_size,
                    "layer {il} KDA Q/K/V shape mismatch: expected [{d_inner_kda}, {hidden_size}]"
                );

                let ssm_conv1d_q = gguf
                    .get_tensor(&format!("blk.{il}.ssm_conv1d_q.weight"))
                    .with_context(|| format!("missing blk.{il}.ssm_conv1d_q.weight"))?
                    .try_to_f32_vec()
                    .with_context(|| {
                        format!("failed to dequantize blk.{il}.ssm_conv1d_q.weight")
                    })?;
                let ssm_conv1d_k = gguf
                    .get_tensor(&format!("blk.{il}.ssm_conv1d_k.weight"))
                    .with_context(|| format!("missing blk.{il}.ssm_conv1d_k.weight"))?
                    .try_to_f32_vec()
                    .with_context(|| {
                        format!("failed to dequantize blk.{il}.ssm_conv1d_k.weight")
                    })?;
                let ssm_conv1d_v = gguf
                    .get_tensor(&format!("blk.{il}.ssm_conv1d_v.weight"))
                    .with_context(|| format!("missing blk.{il}.ssm_conv1d_v.weight"))?
                    .try_to_f32_vec()
                    .with_context(|| {
                        format!("failed to dequantize blk.{il}.ssm_conv1d_v.weight")
                    })?;

                let expected_conv_len = kda_d_conv
                    .checked_mul(d_inner_kda)
                    .context("conv weight arithmetic overflow")?;
                ensure!(
                    ssm_conv1d_q.len() == expected_conv_len
                        && ssm_conv1d_k.len() == expected_conv_len
                        && ssm_conv1d_v.len() == expected_conv_len,
                    "layer {il} conv weights len mismatch: expected {expected_conv_len}"
                );

                let ssm_f_a =
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.ssm_f_a.weight"))?;
                ensure!(
                    ssm_f_a.m == d_inner_kda && ssm_f_a.k == hidden_size,
                    "layer {il} ssm_f_a shape mismatch: expected [{d_inner_kda}, {hidden_size}]"
                );

                let ssm_dt_bias = if let Ok(t) = gguf.get_tensor(&format!("blk.{il}.ssm_dt.bias")) {
                    t.try_to_f32_vec().context("dequantize ssm_dt.bias")?
                } else if let Ok(t) = gguf.get_tensor(&format!("blk.{il}.ssm_dt_b.bias")) {
                    t.try_to_f32_vec().context("dequantize ssm_dt_b.bias")?
                } else {
                    gguf.get_tensor(&format!("blk.{il}.ssm_dt.weight"))
                        .with_context(|| format!("missing blk.{il}.ssm_dt bias"))?
                        .try_to_f32_vec()
                        .context("dequantize ssm_dt.weight")?
                };
                ensure!(
                    ssm_dt_bias.len() == d_inner_kda,
                    "layer {il} ssm_dt bias len mismatch: expected {d_inner_kda}"
                );

                let ssm_a = if let Ok(t) = gguf.get_tensor(&format!("blk.{il}.ssm_a")) {
                    t.try_to_f32_vec().context("dequantize ssm_a")?
                } else {
                    gguf.get_tensor(&format!("blk.{il}.ssm_a.weight"))
                        .with_context(|| format!("missing blk.{il}.ssm_a"))?
                        .try_to_f32_vec()
                        .context("dequantize ssm_a.weight")?
                };
                ensure!(
                    ssm_a.len() == n_heads,
                    "layer {il} ssm_a len mismatch: expected {n_heads}"
                );

                let ssm_beta =
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.ssm_beta.weight"))?;
                ensure!(
                    ssm_beta.m == n_heads && ssm_beta.k == hidden_size,
                    "layer {il} ssm_beta shape mismatch: expected [{n_heads}, {hidden_size}]"
                );
                let ssm_g_a =
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.ssm_g_a.weight"))?;
                ensure!(
                    ssm_g_a.m == d_inner_kda && ssm_g_a.k == hidden_size,
                    "layer {il} ssm_g_a shape mismatch: expected [{d_inner_kda}, {hidden_size}]"
                );

                let ssm_norm = if let Ok(t) = gguf.get_tensor(&format!("blk.{il}.ssm_norm.weight"))
                {
                    t.try_to_f32_vec().context("dequantize ssm_norm.weight")?
                } else {
                    gguf.get_tensor(&format!("blk.{il}.ssm_o_norm.weight"))
                        .with_context(|| format!("missing blk.{il}.ssm_norm.weight"))?
                        .try_to_f32_vec()
                        .context("dequantize ssm_o_norm.weight")?
                };
                ensure!(
                    ssm_norm.len() == kda_head_dim,
                    "layer {il} ssm_norm len mismatch: expected {kda_head_dim}"
                );

                let wo = if gguf
                    .tensors
                    .contains_key(&format!("blk.{il}.attn_output.weight"))
                {
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_output.weight"))?
                } else {
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_out.weight"))?
                };
                ensure!(
                    wo.m == hidden_size && wo.k == d_inner_kda,
                    "layer {il} KDA wo shape mismatch: expected [{hidden_size}, {d_inner_kda}]"
                );

                LayerKindRefs::Kda(KdaLayerRefs {
                    wq,
                    wk,
                    wv,
                    ssm_conv1d_q,
                    ssm_conv1d_k,
                    ssm_conv1d_v,
                    ssm_f_a,
                    ssm_dt_bias,
                    ssm_a,
                    ssm_beta,
                    ssm_g_a,
                    ssm_norm,
                    wo,
                })
            } else {
                // MLA Full Attention layer
                let q_proj = if let Some(q_rank) = q_lora_rank {
                    let wq_a =
                        transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_q_a.weight"))?;
                    ensure!(
                        wq_a.m == q_rank && wq_a.k == hidden_size,
                        "layer {il} wq_a shape mismatch: expected [{q_rank}, {hidden_size}], got [{}, {}]",
                        wq_a.m,
                        wq_a.k
                    );
                    let q_a_norm = gguf
                        .get_tensor(&format!("blk.{il}.attn_q_a_norm.weight"))
                        .with_context(|| format!("missing blk.{il}.attn_q_a_norm.weight"))?
                        .try_to_f32_vec()
                        .with_context(|| {
                            format!("failed to dequantize blk.{il}.attn_q_a_norm.weight")
                        })?;
                    ensure!(
                        q_a_norm.len() == q_rank,
                        "layer {il} attn_q_a_norm len mismatch: expected {q_rank}, got {}",
                        q_a_norm.len()
                    );
                    let wq_b =
                        transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_q_b.weight"))?;
                    let expected_wq_b_m = n_heads
                        .checked_mul(qk_head_dim)
                        .context("wq_b rows overflow")?;
                    ensure!(
                        wq_b.m == expected_wq_b_m && wq_b.k == q_rank,
                        "layer {il} wq_b shape mismatch: expected [{expected_wq_b_m}, {q_rank}], got [{}, {}]",
                        wq_b.m,
                        wq_b.k
                    );
                    MlaQueryRefs::Decomposed {
                        wq_a,
                        q_a_norm,
                        wq_b,
                    }
                } else if gguf
                    .tensors
                    .contains_key(&format!("blk.{il}.attn_q_a.weight"))
                    && gguf
                        .tensors
                        .contains_key(&format!("blk.{il}.attn_q_a_norm.weight"))
                    && gguf
                        .tensors
                        .contains_key(&format!("blk.{il}.attn_q_b.weight"))
                {
                    let wq_a =
                        transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_q_a.weight"))?;
                    let q_a_norm = gguf
                        .get_tensor(&format!("blk.{il}.attn_q_a_norm.weight"))
                        .with_context(|| format!("missing blk.{il}.attn_q_a_norm.weight"))?
                        .try_to_f32_vec()
                        .with_context(|| {
                            format!("failed to dequantize blk.{il}.attn_q_a_norm.weight")
                        })?;
                    let q_rank = q_a_norm.len();
                    ensure!(
                        wq_a.m == q_rank && wq_a.k == hidden_size,
                        "layer {il} wq_a shape mismatch: expected [{q_rank}, {hidden_size}], got [{}, {}]",
                        wq_a.m,
                        wq_a.k
                    );
                    let wq_b =
                        transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_q_b.weight"))?;
                    let expected_wq_b_m = n_heads
                        .checked_mul(qk_head_dim)
                        .context("wq_b rows overflow")?;
                    ensure!(
                        wq_b.m == expected_wq_b_m && wq_b.k == q_rank,
                        "layer {il} wq_b shape mismatch: expected [{expected_wq_b_m}, {q_rank}], got [{}, {}]",
                        wq_b.m,
                        wq_b.k
                    );
                    MlaQueryRefs::Decomposed {
                        wq_a,
                        q_a_norm,
                        wq_b,
                    }
                } else {
                    let wq =
                        transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_q.weight"))?;
                    let expected_wq_m = n_heads
                        .checked_mul(qk_head_dim)
                        .context("wq rows overflow")?;
                    ensure!(
                        wq.m == expected_wq_m && wq.k == hidden_size,
                        "layer {il} wq shape mismatch: expected [{expected_wq_m}, {hidden_size}], got [{}, {}]",
                        wq.m,
                        wq.k
                    );
                    MlaQueryRefs::Direct(wq)
                };

                let wkv_a_mqa =
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_kv_a_mqa.weight"))?;
                let kv_proj_len = kv_lora_rank
                    .checked_add(qk_rope_head_dim)
                    .context("kv_proj_len overflow")?;
                ensure!(
                    wkv_a_mqa.m == kv_proj_len && wkv_a_mqa.k == hidden_size,
                    "layer {il} wkv_a_mqa shape mismatch: expected [{kv_proj_len}, {hidden_size}], got [{}, {}]",
                    wkv_a_mqa.m,
                    wkv_a_mqa.k
                );
                let kv_a_norm = gguf
                    .get_tensor(&format!("blk.{il}.attn_kv_a_norm.weight"))
                    .with_context(|| format!("missing blk.{il}.attn_kv_a_norm.weight"))?
                    .try_to_f32_vec()
                    .with_context(|| {
                        format!("failed to dequantize blk.{il}.attn_kv_a_norm.weight")
                    })?;
                ensure!(
                    kv_a_norm.len() == kv_lora_rank,
                    "layer {il} attn_kv_a_norm len mismatch: expected {kv_lora_rank}, got {}",
                    kv_a_norm.len()
                );

                let wk_b = gguf
                    .get_tensor(&format!("blk.{il}.attn_k_b.weight"))
                    .with_context(|| format!("missing blk.{il}.attn_k_b.weight"))?
                    .try_to_f32_vec()
                    .with_context(|| format!("failed to dequantize blk.{il}.attn_k_b.weight"))?;
                let expected_wk_b_len = n_heads
                    .checked_mul(kv_lora_rank)
                    .and_then(|m| m.checked_mul(qk_nope_head_dim))
                    .context("wk_b size arithmetic overflow")?;
                ensure!(
                    wk_b.len() == expected_wk_b_len,
                    "layer {il} attn_k_b len mismatch: expected {expected_wk_b_len}, got {}",
                    wk_b.len()
                );

                let wv_b = gguf
                    .get_tensor(&format!("blk.{il}.attn_v_b.weight"))
                    .with_context(|| format!("missing blk.{il}.attn_v_b.weight"))?
                    .try_to_f32_vec()
                    .with_context(|| format!("failed to dequantize blk.{il}.attn_v_b.weight"))?;
                let expected_wv_b_len = n_heads
                    .checked_mul(v_head_dim)
                    .and_then(|m| m.checked_mul(kv_lora_rank))
                    .context("wv_b size arithmetic overflow")?;
                ensure!(
                    wv_b.len() == expected_wv_b_len,
                    "layer {il} attn_v_b len mismatch: expected {expected_wv_b_len}, got {}",
                    wv_b.len()
                );

                let wqkv_gate = if gguf
                    .tensors
                    .contains_key(&format!("blk.{il}.attn_gate.weight"))
                {
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_gate.weight"))?
                } else {
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_qkv_gate.weight"))?
                };
                ensure!(
                    wqkv_gate.m == n_heads && wqkv_gate.k == hidden_size,
                    "layer {il} wqkv_gate shape mismatch: expected [{n_heads}, {hidden_size}], got [{}, {}]",
                    wqkv_gate.m,
                    wqkv_gate.k
                );

                let wo = if gguf
                    .tensors
                    .contains_key(&format!("blk.{il}.attn_output.weight"))
                {
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_output.weight"))?
                } else {
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_out.weight"))?
                };
                let expected_wo_k = n_heads.checked_mul(v_head_dim).context("wo_k overflow")?;
                ensure!(
                    wo.m == hidden_size && wo.k == expected_wo_k,
                    "layer {il} MLA wo shape mismatch: expected [{hidden_size}, {expected_wo_k}], got [{}, {}]",
                    wo.m,
                    wo.k
                );

                LayerKindRefs::Mla(MlaLayerRefs {
                    q_proj,
                    wkv_a_mqa,
                    kv_a_norm,
                    wk_b,
                    wv_b,
                    wqkv_gate,
                    wo,
                })
            };

            let ffn = if il < leading_dense_block_count {
                let gate =
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.ffn_gate.weight"))?;
                let up = transformer::resolve_weight(&gguf, &format!("blk.{il}.ffn_up.weight"))?;
                let down =
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.ffn_down.weight"))?;
                let ff = intermediate_size;
                ensure!(
                    gate.m == ff
                        && gate.k == hidden_size
                        && up.m == ff
                        && up.k == hidden_size
                        && down.m == hidden_size
                        && down.k == ff,
                    "layer {il} dense FFN shape mismatch: expected gate/up [{ff}, {hidden_size}] and down [{hidden_size}, {ff}], got gate [{}, {}], up [{}, {}], down [{}, {}]",
                    gate.m,
                    gate.k,
                    up.m,
                    up.k,
                    down.m,
                    down.k
                );
                FfnKindRefs::Dense(DenseFfnRefs { gate, up, down })
            } else {
                let router =
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.ffn_gate_inp.weight"))?;
                ensure!(
                    router.m == n_expert && router.k == hidden_size,
                    "layer {il} router shape mismatch: expected [{n_expert}, {hidden_size}], got [{}, {}]",
                    router.m,
                    router.k
                );
                let exp_probs_b = if let Ok(t) =
                    gguf.get_tensor(&format!("blk.{il}.exp_probs_b.bias"))
                {
                    t.try_to_f32_vec().context("dequantize exp_probs_b.bias")?
                } else if let Ok(t) = gguf.get_tensor(&format!("blk.{il}.ffn_exp_probs_b.bias")) {
                    t.try_to_f32_vec()
                        .context("dequantize ffn_exp_probs_b.bias")?
                } else {
                    vec![0.0f32; n_expert]
                };
                ensure!(
                    exp_probs_b.len() == n_expert,
                    "layer {il} exp_probs_b len mismatch: expected {n_expert}, got {}",
                    exp_probs_b.len()
                );

                let mut gate_exps = Vec::with_capacity(n_expert);
                let mut up_exps = Vec::with_capacity(n_expert);
                let mut down_exps = Vec::with_capacity(n_expert);

                for e in 0..n_expert {
                    let g = transformer::resolve_expert_weight(
                        &gguf,
                        &format!("blk.{il}.ffn_gate_exps.weight"),
                        e,
                    )?;
                    let u = transformer::resolve_expert_weight(
                        &gguf,
                        &format!("blk.{il}.ffn_up_exps.weight"),
                        e,
                    )?;
                    let d = transformer::resolve_expert_weight(
                        &gguf,
                        &format!("blk.{il}.ffn_down_exps.weight"),
                        e,
                    )?;
                    ensure!(
                        g.m == moe_intermediate_size
                            && g.k == hidden_size
                            && u.m == moe_intermediate_size
                            && u.k == hidden_size
                            && d.m == hidden_size
                            && d.k == moe_intermediate_size,
                        "layer {il} expert {e} shape mismatch: expected gate/up [{moe_intermediate_size}, {hidden_size}] and down [{hidden_size}, {moe_intermediate_size}], got gate [{}, {}], up [{}, {}], down [{}, {}]",
                        g.m,
                        g.k,
                        u.m,
                        u.k,
                        d.m,
                        d.k
                    );
                    gate_exps.push(g);
                    up_exps.push(u);
                    down_exps.push(d);
                }

                let gate_shexp =
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.ffn_gate_shexp.weight"))?;
                let up_shexp =
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.ffn_up_shexp.weight"))?;
                let down_shexp =
                    transformer::resolve_weight(&gguf, &format!("blk.{il}.ffn_down_shexp.weight"))?;
                ensure!(
                    gate_shexp.k == hidden_size
                        && up_shexp.m == gate_shexp.m
                        && up_shexp.k == hidden_size
                        && down_shexp.m == hidden_size
                        && down_shexp.k == gate_shexp.m,
                    "layer {il} shared expert shape mismatch: gate [{}, {}], up [{}, {}], down [{}, {}]",
                    gate_shexp.m,
                    gate_shexp.k,
                    up_shexp.m,
                    up_shexp.k,
                    down_shexp.m,
                    down_shexp.k
                );

                FfnKindRefs::Moe(MoeFfnRefs {
                    router,
                    exp_probs_b,
                    gate_exps,
                    up_exps,
                    down_exps,
                    gate_shexp,
                    up_shexp,
                    down_shexp,
                    n_expert,
                    n_expert_used,
                    routed_scaling_factor,
                })
            };

            layers.push(LayerRefs {
                kind,
                attn_norm,
                ffn,
                ffn_norm,
            });
        }

        let moe_config = if n_layers > leading_dense_block_count && n_expert > 0 {
            Some(crate::model::MoeConfig {
                n_expert,
                n_expert_used,
                expert_ff_len: moe_intermediate_size,
                is_moe_layer: (0..n_layers)
                    .map(|il| il >= leading_dense_block_count)
                    .collect(),
            })
        } else {
            None
        };

        let config = ModelConfig {
            architecture: arch.clone(),
            n_layers,
            hidden_size,
            intermediate_size,
            n_heads,
            n_kv_heads: 1,
            vocab_size,
            max_seq_len: context_size,
            head_dim: cache_head_dim,
            rms_norm_eps,
            rope_theta,
            block_types,
            conv_kernel_size: Some(kda_d_conv),
            ssm: Some(ssm_config),
            kv_heads_per_layer,
            scalars: crate::model::ScalarMultipliers::default(),
            moe: moe_config,
            is_causal: true,
            class_labels: Vec::new(),
        };

        let kq_scale = 1.0 / (qk_head_dim as f32).sqrt();

        Ok(Self {
            gguf,
            config,
            head_dim: kda_head_dim,
            n_heads,
            kda_d_conv,
            kda_lower_bound,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kq_scale,
            embd_ref,
            output_norm_weight,
            output_ref,
            layers,
            model_id,
        })
    }

    /// Forward pass through one KDA recurrent block.
    fn forward_kda_block(
        &self,
        layer: usize,
        normed: &[f32],
        state: &mut InferenceState,
        refs: &KdaLayerRefs,
        out: &mut [f32],
    ) {
        let hidden_size = self.config.hidden_size;
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let d_inner = n_heads * head_dim;
        let d_conv = self.kda_d_conv;
        let eps = self.config.rms_norm_eps;

        if out.len() < hidden_size || normed.len() < hidden_size {
            return;
        }

        let (conv_state, ssm_state) = match state.layers.get_mut(layer) {
            Some(LayerState::DeltaNet {
                conv_state,
                ssm_state,
            }) => (conv_state.as_mut_slice(), ssm_state.as_mut_slice()),
            _ => return,
        };

        let expected_conv_len = (d_conv.saturating_sub(1)) * 3 * d_inner;
        let expected_ssm_len = n_heads * head_dim * head_dim;
        if conv_state.len() < expected_conv_len || ssm_state.len() < expected_ssm_len {
            return;
        }

        // 1. Projections for Q, K, V
        let mut q_proj = std::mem::take(&mut state.scratch.conv_proj);
        let mut k_proj = std::mem::take(&mut state.scratch.k);
        let mut v_proj = std::mem::take(&mut state.scratch.v);
        let mut q_conv_out = std::mem::take(&mut state.scratch.ssm_in_proj);
        let mut k_conv_out = std::mem::take(&mut state.scratch.ssm_conv_out);
        let mut v_conv_out = std::mem::take(&mut state.scratch.ssm_y);

        q_proj.resize(d_inner, 0.0);
        k_proj.resize(d_inner, 0.0);
        v_proj.resize(d_inner, 0.0);
        q_conv_out.resize(d_inner, 0.0);
        k_conv_out.resize(d_inner, 0.0);
        v_conv_out.resize(d_inner, 0.0);

        transformer::gemv(&self.gguf, &refs.wq, normed, &mut q_proj);
        transformer::gemv(&self.gguf, &refs.wk, normed, &mut k_proj);
        transformer::gemv(&self.gguf, &refs.wv, normed, &mut v_proj);

        // 2. Depthwise causal convolutions for Q, K, V
        let conv_sub_len = (d_conv.saturating_sub(1)) * d_inner;
        let (q_cstate, rest_cstate) = conv_state.split_at_mut(conv_sub_len);
        let (k_cstate, v_cstate) = rest_cstate.split_at_mut(conv_sub_len);
        let v_cstate = &mut v_cstate[..conv_sub_len];

        cpu::mamba2_conv1d_step(
            &q_proj,
            q_cstate,
            &refs.ssm_conv1d_q,
            None,
            d_inner,
            d_conv,
            &mut q_conv_out,
        );
        cpu::mamba2_conv1d_step(
            &k_proj,
            k_cstate,
            &refs.ssm_conv1d_k,
            None,
            d_inner,
            d_conv,
            &mut k_conv_out,
        );
        cpu::mamba2_conv1d_step(
            &v_proj,
            v_cstate,
            &refs.ssm_conv1d_v,
            None,
            d_inner,
            d_conv,
            &mut v_conv_out,
        );

        // 3. Gating computations: ssm_f_a, ssm_dt_bias, ssm_a, ssm_beta
        let mut gate_raw = std::mem::take(&mut state.scratch.gate);
        gate_raw.resize(d_inner, 0.0);
        transformer::gemv(&self.gguf, &refs.ssm_f_a, normed, &mut gate_raw);
        cpu::add_inplace(&mut gate_raw, &refs.ssm_dt_bias);

        let mut beta_raw_buf = [0.0f32; 64];
        let mut beta_raw_heap;
        let beta_raw = if n_heads <= 64 {
            &mut beta_raw_buf[..n_heads]
        } else {
            beta_raw_heap = vec![0.0f32; n_heads];
            beta_raw_heap.as_mut_slice()
        };
        transformer::gemv(&self.gguf, &refs.ssm_beta, normed, beta_raw);
        cpu::sigmoid_inplace(beta_raw);

        // Compute decay per element: gate[h, j] = lower_bound * sigmoid(gate_raw * ssm_a[h])
        for h in 0..n_heads {
            let a_h = refs.ssm_a[h];
            let head_slice = &mut gate_raw[h * head_dim..(h + 1) * head_dim];
            for val in head_slice.iter_mut() {
                let scaled = (*val * a_h).clamp(-80.0, 80.0);
                let sig = 1.0 / (1.0 + (-scaled).exp());
                let g = (self.kda_lower_bound * sig).clamp(-80.0, 0.0);
                *val = g.exp();
            }
        }

        // 4. Per-head L2 norm on Q and K, with Q scaling 1/sqrt(head_dim)
        let q_scale = 1.0 / (head_dim as f32).sqrt();
        for h in 0..n_heads {
            let q_head = &mut q_conv_out[h * head_dim..(h + 1) * head_dim];
            let sum_sq_q = cpu::dot_f32(q_head, q_head);
            let l2_q = sum_sq_q.sqrt().max(eps);
            let factor_q = q_scale / l2_q;
            cpu::scale_inplace(q_head, factor_q);

            let k_head = &mut k_conv_out[h * head_dim..(h + 1) * head_dim];
            let sum_sq_k = cpu::dot_f32(k_head, k_head);
            let l2_k = sum_sq_k.sqrt().max(eps);
            let factor_k = 1.0 / l2_k;
            cpu::scale_inplace(k_head, factor_k);
        }

        // 5. DeltaNet recurrence update
        let mut kda_core_out = std::mem::take(&mut state.scratch.ssm_branch_out);
        kda_core_out.resize(d_inner, 0.0);

        let mut sk_buf = [0.0f32; 256];
        let mut d_buf = [0.0f32; 256];
        let mut sk_heap;
        let mut d_heap;
        let (sk, d) = if head_dim <= 256 {
            (&mut sk_buf[..head_dim], &mut d_buf[..head_dim])
        } else {
            sk_heap = vec![0.0f32; head_dim];
            d_heap = vec![0.0f32; head_dim];
            (sk_heap.as_mut_slice(), d_heap.as_mut_slice())
        };

        for h in 0..n_heads {
            let q = &q_conv_out[h * head_dim..(h + 1) * head_dim];
            let k = &k_conv_out[h * head_dim..(h + 1) * head_dim];
            let v = &v_conv_out[h * head_dim..(h + 1) * head_dim];
            let decay_head = &gate_raw[h * head_dim..(h + 1) * head_dim];
            let b = beta_raw[h];

            let state_offset = h * head_dim * head_dim;
            let s_mat = &mut ssm_state[state_offset..state_offset + head_dim * head_dim];

            // 5a. Row-wise state decay: S[y, :] *= decay_head[y]
            for (y, &dec) in decay_head.iter().enumerate().take(head_dim) {
                let row = &mut s_mat[y * head_dim..(y + 1) * head_dim];
                if !dec.is_finite() || dec <= 0.0 {
                    row.fill(0.0);
                } else if (dec - 1.0).abs() > 1e-7 {
                    cpu::scale_inplace(row, dec);
                }
            }

            // 5b. sk[y] = sum_x S[y, x] * k[x]
            sk.fill(0.0);
            for y in 0..head_dim {
                let row = &s_mat[y * head_dim..(y + 1) * head_dim];
                sk[y] = cpu::dot_f32(row, k);
            }

            // 5c. d[y] = beta * (v[y] - sk[y])
            for ((dj, &vj), &skj) in d.iter_mut().zip(v.iter()).zip(sk.iter()) {
                *dj = b * (vj - skj);
            }

            // 5d & 5e. State update S[y, x] += d[y] * k[x] and output out[y] = sum_x S[y, x] * q[x]
            let o_head = &mut kda_core_out[h * head_dim..(h + 1) * head_dim];
            o_head.fill(0.0);
            for y in 0..head_dim {
                let dy = d[y];
                let row = &mut s_mat[y * head_dim..(y + 1) * head_dim];
                cpu::axpy_inplace(row, k, dy);
                o_head[y] = cpu::dot_f32(row, q);
            }
        }

        // 6. Modulated RMSNorm: per-head norm modulated by output gate
        let mut out_gate = std::mem::take(&mut state.scratch.up);
        out_gate.resize(d_inner, 0.0);
        transformer::gemv(&self.gguf, &refs.ssm_g_a, normed, &mut out_gate);
        cpu::sigmoid_inplace(&mut out_gate);

        for h in 0..n_heads {
            let o_head = &mut kda_core_out[h * head_dim..(h + 1) * head_dim];
            cpu::rmsnorm(o_head, &refs.ssm_norm, eps);
        }
        cpu::mul_inplace(&mut kda_core_out, &out_gate);

        // 7. Output projection
        transformer::gemv(&self.gguf, &refs.wo, &kda_core_out, &mut out[..hidden_size]);

        // Restore scratch buffers
        state.scratch.conv_proj = q_proj;
        state.scratch.k = k_proj;
        state.scratch.v = v_proj;
        state.scratch.ssm_in_proj = q_conv_out;
        state.scratch.ssm_conv_out = k_conv_out;
        state.scratch.ssm_y = v_conv_out;
        state.scratch.ssm_branch_out = kda_core_out;
        state.scratch.gate = gate_raw;
        state.scratch.up = out_gate;
    }

    /// Forward pass through one MLA full attention block.
    fn forward_mla_block(
        &self,
        layer: usize,
        normed: &[f32],
        pos: usize,
        state: &mut InferenceState,
        refs: &MlaLayerRefs,
        out: &mut [f32],
    ) {
        let hidden_size = self.config.hidden_size;
        let n_heads = self.n_heads;
        let kv_lora_rank = self.kv_lora_rank;
        let qk_nope_head_dim = self.qk_nope_head_dim;
        let qk_rope_head_dim = self.qk_rope_head_dim;
        let qk_head_dim = self.qk_head_dim;
        let v_head_dim = self.v_head_dim;
        let eps = self.config.rms_norm_eps;

        if out.len() < hidden_size || normed.len() < hidden_size {
            return;
        }
        if !matches!(state.layers.get(layer), Some(LayerState::Attention { .. })) {
            return;
        }

        // 1. Query projection
        let mut q_all = std::mem::take(&mut state.scratch.conv_proj);
        let q_all_len = n_heads * qk_head_dim;
        q_all.resize(q_all_len, 0.0);

        match &refs.q_proj {
            MlaQueryRefs::Decomposed {
                wq_a,
                q_a_norm,
                wq_b,
            } => {
                let q_rank = q_a_norm.len();
                let mut q_a = std::mem::take(&mut state.scratch.q);
                q_a.resize(q_rank, 0.0);
                transformer::gemv(&self.gguf, wq_a, normed, &mut q_a);
                cpu::rmsnorm(&mut q_a, q_a_norm, eps);
                transformer::gemv(&self.gguf, wq_b, &q_a, &mut q_all);
                state.scratch.q = q_a;
            }
            MlaQueryRefs::Direct(wq) => {
                transformer::gemv(&self.gguf, wq, normed, &mut q_all);
            }
        }

        // 2. Compressed KV projection
        let kv_proj_len = kv_lora_rank + qk_rope_head_dim;
        let mut kv_all = std::mem::take(&mut state.scratch.k);
        kv_all.resize(kv_proj_len, 0.0);
        transformer::gemv(&self.gguf, &refs.wkv_a_mqa, normed, &mut kv_all);

        let (c_kv, k_pe) = kv_all.split_at_mut(kv_lora_rank);
        cpu::rmsnorm(c_kv, &refs.kv_a_norm, eps);

        // 3. Decoupled RoPE: apply to q_rope and k_rope
        let mut q_rope_buf = std::mem::take(&mut state.scratch.v);
        let q_rope_total = n_heads * qk_rope_head_dim;
        q_rope_buf.resize(q_rope_total, 0.0);

        for h in 0..n_heads {
            let src = &q_all[h * qk_head_dim + qk_nope_head_dim..(h + 1) * qk_head_dim];
            q_rope_buf[h * qk_rope_head_dim..(h + 1) * qk_rope_head_dim].copy_from_slice(src);
        }

        cpu::rope(
            &mut q_rope_buf,
            k_pe,
            pos,
            n_heads,
            1,
            qk_rope_head_dim,
            self.config.rope_theta,
        );

        // 4. Form cached key and value vectors and append to cache
        // Key is [c_kv, k_pe] (length kv_lora_rank + qk_rope_head_dim)
        // Value is c_kv (length kv_lora_rank)
        let mut cached_k = std::mem::take(&mut state.scratch.ssm_conv_out);
        cached_k.resize(kv_proj_len, 0.0);
        cached_k[..kv_lora_rank].copy_from_slice(c_kv);
        cached_k[kv_lora_rank..].copy_from_slice(k_pe);

        if state.kv_f16 {
            state.append_kv_f16(layer, &cached_k, c_kv);
        } else {
            state.append_kv(layer, &cached_k, c_kv);
        }

        // 5. Query absorption and scaled dot-product attention
        // For each head h:
        // q'_nope_h = W_k_b,h * q_nope_h (dimension kv_lora_rank)
        // q'_h = [q'_nope_h, q_pe_h] (dimension kv_lora_rank + qk_rope_head_dim)
        let (key_cache_f32, val_cache_f32, key_cache_f16, val_cache_f16) =
            match state.layers.get(layer) {
                Some(LayerState::Attention {
                    key_cache,
                    value_cache,
                    key_cache_f16,
                    value_cache_f16,
                    ..
                }) => (
                    key_cache.as_slice(),
                    value_cache.as_slice(),
                    key_cache_f16.as_slice(),
                    value_cache_f16.as_slice(),
                ),
                _ => {
                    state.scratch.conv_proj = q_all;
                    state.scratch.k = kv_all;
                    state.scratch.v = q_rope_buf;
                    state.scratch.ssm_conv_out = cached_k;
                    return;
                }
            };

        let cached_tokens = if state.kv_f16 {
            key_cache_f16.len() / kv_proj_len
        } else {
            key_cache_f32.len() / kv_proj_len
        };
        let seq_len = (pos + 1).min(cached_tokens);

        let mut scores = std::mem::take(&mut state.scratch.scores);
        scores.resize(seq_len, 0.0);

        let mut mla_heads_out = std::mem::take(&mut state.scratch.attn_out);
        let total_out_dim = n_heads * v_head_dim;
        mla_heads_out.resize(total_out_dim, 0.0);

        let mut q_full_h = std::mem::take(&mut state.scratch.ssm_in_proj);
        q_full_h.resize(kv_proj_len, 0.0);

        let mut attn_ctx = std::mem::take(&mut state.scratch.ssm_y);
        attn_ctx.resize(kv_lora_rank, 0.0);

        let wk_b_slice_len = kv_lora_rank * qk_nope_head_dim;
        let wv_b_slice_len = v_head_dim * kv_lora_rank;

        for h in 0..n_heads {
            let q_nope_h = &q_all[h * qk_head_dim..h * qk_head_dim + qk_nope_head_dim];
            let q_pe_h = &q_rope_buf[h * qk_rope_head_dim..(h + 1) * qk_rope_head_dim];

            // Project q_nope through W_k_b,h
            let wk_b_h = &refs.wk_b[h * wk_b_slice_len..(h + 1) * wk_b_slice_len];
            let q_prime_nope = &mut q_full_h[..kv_lora_rank];
            for r in 0..kv_lora_rank {
                let w_row = &wk_b_h[r * qk_nope_head_dim..(r + 1) * qk_nope_head_dim];
                q_prime_nope[r] = cpu::dot_f32(w_row, q_nope_h);
            }
            q_full_h[kv_lora_rank..].copy_from_slice(q_pe_h);

            // Compute attention scores against all cached tokens
            if state.kv_f16 {
                cpu::attn_scores_f16(
                    &q_full_h,
                    key_cache_f16,
                    &mut scores,
                    kv_proj_len,
                    0,
                    kv_proj_len,
                    self.kq_scale,
                    seq_len,
                );
            } else {
                cpu::attn_scores(
                    &q_full_h,
                    key_cache_f32,
                    &mut scores,
                    kv_proj_len,
                    0,
                    kv_proj_len,
                    self.kq_scale,
                    seq_len,
                );
            }

            cpu::softmax_inplace(&mut scores);

            // Value aggregation
            if state.kv_f16 {
                cpu::attn_values_f16(
                    &scores,
                    val_cache_f16,
                    &mut attn_ctx,
                    kv_lora_rank,
                    0,
                    kv_lora_rank,
                    seq_len,
                );
            } else {
                cpu::attn_values(
                    &scores,
                    val_cache_f32,
                    &mut attn_ctx,
                    kv_lora_rank,
                    0,
                    kv_lora_rank,
                    seq_len,
                );
            }

            // Project attn_ctx through W_v_b,h
            let wv_b_h = &refs.wv_b[h * wv_b_slice_len..(h + 1) * wv_b_slice_len];
            let head_out = &mut mla_heads_out[h * v_head_dim..(h + 1) * v_head_dim];
            for r in 0..v_head_dim {
                let w_row = &wv_b_h[r * kv_lora_rank..(r + 1) * kv_lora_rank];
                head_out[r] = cpu::dot_f32(w_row, &attn_ctx);
            }
        }

        // 6. Head-wise gating
        let mut attn_gate = std::mem::take(&mut state.scratch.gate);
        attn_gate.resize(n_heads, 0.0);
        transformer::gemv(&self.gguf, &refs.wqkv_gate, normed, &mut attn_gate);
        cpu::sigmoid_inplace(&mut attn_gate);

        for h in 0..n_heads {
            let g = attn_gate[h];
            let head_out = &mut mla_heads_out[h * v_head_dim..(h + 1) * v_head_dim];
            cpu::scale_inplace(head_out, g);
        }

        // 7. Output projection
        transformer::gemv(
            &self.gguf,
            &refs.wo,
            &mla_heads_out,
            &mut out[..hidden_size],
        );

        // Restore scratch buffers
        state.scratch.conv_proj = q_all;
        state.scratch.k = kv_all;
        state.scratch.v = q_rope_buf;
        state.scratch.ssm_conv_out = cached_k;
        state.scratch.scores = scores;
        state.scratch.attn_out = mla_heads_out;
        state.scratch.ssm_in_proj = q_full_h;
        state.scratch.ssm_y = attn_ctx;
        state.scratch.gate = attn_gate;
    }

    /// Forward pass through one FFN block (Dense or MoE with shared expert).
    fn forward_ffn_block(
        &self,
        ffn_in: &[f32],
        refs: &FfnKindRefs,
        state: &mut InferenceState,
        out: &mut [f32],
    ) {
        let hidden_size = self.config.hidden_size;
        if out.len() < hidden_size || ffn_in.len() < hidden_size {
            return;
        }

        match refs {
            FfnKindRefs::Dense(dense) => {
                let ff = self.config.intermediate_size;
                let mut gate = std::mem::take(&mut state.scratch.gate);
                let mut up = std::mem::take(&mut state.scratch.up);
                gate.resize(ff, 0.0);
                up.resize(ff, 0.0);

                transformer::gemv(&self.gguf, &dense.gate, ffn_in, &mut gate);
                transformer::gemv(&self.gguf, &dense.up, ffn_in, &mut up);
                cpu::silu_mul_inplace(&mut gate, &up);
                transformer::gemv(&self.gguf, &dense.down, &gate, &mut out[..hidden_size]);

                state.scratch.gate = gate;
                state.scratch.up = up;
            }
            FfnKindRefs::Moe(moe) => {
                let n_expert = moe.n_expert;
                let n_used = moe.n_expert_used;

                // 1. Router logits & probabilities
                state.scratch.moe_probs.resize(n_expert, 0.0);
                transformer::gemv(
                    &self.gguf,
                    &moe.router,
                    ffn_in,
                    &mut state.scratch.moe_probs[..n_expert],
                );
                cpu::sigmoid_inplace(&mut state.scratch.moe_probs[..n_expert]);

                // 2. Select top experts
                let mut selected = std::mem::take(&mut state.scratch.moe_selected);
                select_bailingmoe_experts(
                    &state.scratch.moe_probs[..n_expert],
                    &moe.exp_probs_b,
                    n_used,
                    moe.routed_scaling_factor,
                    &mut selected,
                );

                // 3. Accumulate routed experts
                out[..hidden_size].fill(0.0);
                let mut expert_out = std::mem::take(&mut state.scratch.moe_expert_out);
                expert_out.resize(hidden_size, 0.0);

                if moe.gate_exps.is_empty() {
                    state.scratch.moe_selected = selected;
                    state.scratch.moe_expert_out = expert_out;
                    return;
                }

                let ff_exp = moe.gate_exps[0].m;
                let mut gate = std::mem::take(&mut state.scratch.gate);
                let mut up = std::mem::take(&mut state.scratch.up);
                gate.resize(ff_exp, 0.0);
                up.resize(ff_exp, 0.0);

                for &(exp_idx, weight) in &selected {
                    if weight == 0.0 || exp_idx >= moe.gate_exps.len() {
                        continue;
                    }
                    transformer::gemv(&self.gguf, &moe.gate_exps[exp_idx], ffn_in, &mut gate);
                    transformer::gemv(&self.gguf, &moe.up_exps[exp_idx], ffn_in, &mut up);
                    cpu::silu_mul_inplace(&mut gate, &up);
                    transformer::gemv(&self.gguf, &moe.down_exps[exp_idx], &gate, &mut expert_out);
                    cpu::axpy_inplace(&mut out[..hidden_size], &expert_out[..hidden_size], weight);
                }

                // 4. Parallel shared expert SwiGLU FFN
                let ff_shexp = moe.gate_shexp.m;
                gate.resize(ff_shexp, 0.0);
                up.resize(ff_shexp, 0.0);
                transformer::gemv(&self.gguf, &moe.gate_shexp, ffn_in, &mut gate);
                transformer::gemv(&self.gguf, &moe.up_shexp, ffn_in, &mut up);
                cpu::silu_mul_inplace(&mut gate, &up);
                transformer::gemv(&self.gguf, &moe.down_shexp, &gate, &mut expert_out);
                cpu::add_inplace(&mut out[..hidden_size], &expert_out[..hidden_size]);

                state.scratch.moe_selected = selected;
                state.scratch.moe_expert_out = expert_out;
                state.scratch.gate = gate;
                state.scratch.up = up;
            }
        }
    }

    /// Run single token through all layers.
    fn run_layers(&self, hidden: &mut [f32], pos: usize, state: &mut InferenceState) {
        let hidden_size = self.config.hidden_size;

        let mut normed = std::mem::take(&mut state.scratch.normed);
        let mut layer_out = std::mem::take(&mut state.scratch.out);
        let mut ffn_in = std::mem::take(&mut state.scratch.ffn_input);
        let mut ffn_out = std::mem::take(&mut state.scratch.lora_tmp);

        normed.resize(hidden_size, 0.0);
        layer_out.resize(hidden_size, 0.0);
        ffn_in.resize(hidden_size, 0.0);
        ffn_out.resize(hidden_size, 0.0);

        for (il, layer_ref) in self.layers.iter().enumerate() {
            layer_out.fill(0.0);
            normed.copy_from_slice(hidden);
            cpu::rmsnorm(&mut normed, &layer_ref.attn_norm, self.config.rms_norm_eps);

            match &layer_ref.kind {
                LayerKindRefs::Kda(kda) => {
                    self.forward_kda_block(il, &normed, state, kda, &mut layer_out);
                }
                LayerKindRefs::Mla(mla) => {
                    self.forward_mla_block(il, &normed, pos, state, mla, &mut layer_out);
                }
            }

            // Residual connection: hidden += layer_out
            cpu::add_inplace(hidden, &layer_out);

            // FFN norm
            ffn_in.copy_from_slice(hidden);
            cpu::rmsnorm(&mut ffn_in, &layer_ref.ffn_norm, self.config.rms_norm_eps);

            // FFN block
            ffn_out.fill(0.0);
            self.forward_ffn_block(&ffn_in, &layer_ref.ffn, state, &mut ffn_out);

            // Residual connection: hidden += ffn_out
            cpu::add_inplace(hidden, &ffn_out);

            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record(&format!("l_out-{il}"), hidden);
            }
        }

        state.scratch.normed = normed;
        state.scratch.out = layer_out;
        state.scratch.ffn_input = ffn_in;
        state.scratch.lora_tmp = ffn_out;
    }

    /// Project final hidden state to logits.
    fn project_logits(&self, hidden: &[f32], state: &mut InferenceState) -> Vec<f32> {
        let vocab_size = self.config.vocab_size;
        if state.scratch.logits.len() < vocab_size {
            state.scratch.logits.resize(vocab_size, 0.0);
        }

        let out_ref = self.output_ref.as_ref().unwrap_or(&self.embd_ref);
        transformer::gemv(
            &self.gguf,
            out_ref,
            hidden,
            &mut state.scratch.logits[..vocab_size],
        );

        state.scratch.logits[..vocab_size].to_vec()
    }
}

impl Model for BailingMoe3Model {
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        if tokens.is_empty() {
            tracing::warn!(target: "cera::bailingmoe3", "forward called with empty token slice");
            return Vec::new();
        }
        if tokens.len() > 1 {
            return self.forward_prefill(tokens, pos, state);
        }
        if pos != state.seq_len {
            tracing::error!(
                target: "cera::bailingmoe3",
                pos,
                state_seq_len = state.seq_len,
                "pos does not match state.seq_len"
            );
            return Vec::new();
        }
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        if token_id >= cfg.vocab_size {
            tracing::error!(
                target: "cera::bailingmoe3",
                token_id,
                vocab_size = cfg.vocab_size,
                "token ID out of range"
            );
            return Vec::new();
        }

        let mut hidden_stack = [0.0f32; 4096];
        let mut hidden_heap;
        let hidden = if cfg.hidden_size <= 4096 {
            &mut hidden_stack[..cfg.hidden_size]
        } else {
            hidden_heap = vec![0.0f32; cfg.hidden_size];
            &mut hidden_heap[..]
        };
        transformer::dequantize_row_into(&self.gguf, &self.embd_ref, token_id, hidden);
        if transformer::oracle_dump::is_active() {
            transformer::oracle_dump::record("embd", hidden);
        }
        self.run_layers(hidden, pos, state);
        cpu::rmsnorm(hidden, &self.output_norm_weight, self.config.rms_norm_eps);
        if transformer::oracle_dump::is_active() {
            transformer::oracle_dump::record("result_norm", hidden);
        }
        let logits = self.project_logits(hidden, state);
        if transformer::oracle_dump::is_active() {
            transformer::oracle_dump::record("result_output", &logits);
        }
        state.seq_len = pos + 1;
        logits
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        if tokens.is_empty() {
            tracing::warn!(target: "cera::bailingmoe3", "forward_prefill requires at least one token");
            return Vec::new();
        }
        if start_pos != state.seq_len {
            tracing::error!(
                target: "cera::bailingmoe3",
                start_pos,
                state_seq_len = state.seq_len,
                "start_pos does not match state.seq_len"
            );
            return Vec::new();
        }

        let n = tokens.len();
        let cfg = &self.config;

        for (i, &token) in tokens.iter().enumerate() {
            let token_id = token as usize;
            if token_id >= cfg.vocab_size {
                tracing::error!(
                    target: "cera::bailingmoe3",
                    token_id,
                    pos = start_pos + i,
                    vocab_size = cfg.vocab_size,
                    "token ID out of range during prefill"
                );
                return Vec::new();
            }
        }

        let mut hidden_stack = [0.0f32; 4096];
        let mut hidden_heap;
        let hidden = if cfg.hidden_size <= 4096 {
            &mut hidden_stack[..cfg.hidden_size]
        } else {
            hidden_heap = vec![0.0f32; cfg.hidden_size];
            &mut hidden_heap[..]
        };

        for (i, &token) in tokens[..n - 1].iter().enumerate() {
            let token_id = token as usize;
            transformer::dequantize_row_into(&self.gguf, &self.embd_ref, token_id, hidden);
            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record("embd", hidden);
            }
            self.run_layers(hidden, start_pos + i, state);
            state.seq_len = start_pos + i + 1;
        }

        self.forward(&[tokens[n - 1]], start_pos + n - 1, state)
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn validate_checkpoint_state(
        &self,
        snapshot: &crate::kv_cache::StateSnapshot,
    ) -> Result<(), String> {
        snapshot.validate_for_model_with_head_dims(
            &self.config,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.kv_lora_rank,
        )
    }

    fn f16_kv_supported(&self) -> bool {
        true
    }

    fn supports_kv_shift(&self) -> bool {
        false
    }

    /// No LoRA hooks: the bespoke attention/FFN paths never read `state.lora`
    /// (the `lora` hits in this file are MLA ranks and `lora_tmp` scratch
    /// reuse, both unrelated). Restated (not inherited) so whoever adds the
    /// hooks reads this here.
    fn supports_lora(&self) -> bool {
        false
    }

    fn supports_all_logits(&self) -> bool {
        false
    }

    fn try_reset_kv(
        &self,
        state: &mut InferenceState,
        compression: &crate::kv_cache::KvCompression,
        max_seq_len: usize,
    ) -> Result<(), crate::session::CeraError> {
        super::reset_cpu_kv(self, state, compression, max_seq_len)
    }

    fn check_kv_rewind(
        &self,
        state: &InferenceState,
        len: usize,
    ) -> Result<(), crate::kv_cache::KvRewindError> {
        state.check_truncate_to(len)
    }

    fn try_truncate_kv(
        &self,
        state: &mut InferenceState,
        len: usize,
    ) -> Result<(), crate::kv_cache::KvRewindError> {
        state.try_truncate_to(len)
    }

    fn truncate_kv(&self, state: &mut InferenceState, len: usize) {
        state.truncate_to(len);
    }
}

/// Select top-k experts based on biased scores and weight by normalized unbiased probabilities.
pub fn select_bailingmoe_experts(
    probs: &[f32],
    biases: &[f32],
    n_used: usize,
    routed_scaling_factor: f32,
    selected: &mut Vec<(usize, f32)>,
) {
    selected.clear();
    let n_expert = probs.len().min(biases.len());
    let mut stack_biased = [0.0f32; 512];
    let heap_biased;
    let biased: &[f32] = if n_expert <= 512 {
        for i in 0..n_expert {
            let val = probs[i] + biases[i];
            stack_biased[i] = if val.is_finite() { val } else { -f32::INFINITY };
        }
        &stack_biased[..n_expert]
    } else {
        heap_biased = probs
            .iter()
            .zip(biases)
            .map(|(&p, &b)| {
                let val = p + b;
                if val.is_finite() { val } else { -f32::INFINITY }
            })
            .collect::<Vec<_>>();
        &heap_biased[..]
    };

    for _ in 0..n_used.min(n_expert) {
        let best = (0..n_expert)
            .filter(|e| !selected.iter().any(|(taken, _)| taken == e))
            .max_by(|&a, &b| biased[a].total_cmp(&biased[b]).then(b.cmp(&a)));
        if let Some(e) = best {
            selected.push((e, probs[e].max(0.0)));
        }
    }

    const MIN_POSITIVE_F16: f32 = 1.0 / 16384.0;
    let sum: f32 = selected.iter().map(|(_, w)| *w).sum();
    if !sum.is_finite() || sum <= 0.0 {
        for (_, w) in selected.iter_mut() {
            *w = 0.0;
        }
        return;
    }
    let norm_factor = (1.0 / sum.max(MIN_POSITIVE_F16)) * routed_scaling_factor;
    if !norm_factor.is_finite() {
        for (_, w) in selected.iter_mut() {
            *w = 0.0;
        }
        return;
    }
    for (_, w) in selected.iter_mut() {
        if !w.is_finite() {
            *w = 0.0;
        } else {
            *w *= norm_factor;
        }
    }
}
