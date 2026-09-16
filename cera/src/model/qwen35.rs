//! Qwen 3.5 and Ornith 1.0 (qwen35) hybrid linear and full attention model implementation.
//!
//! Supports:
//! - `qwen35`: Qwen 3.5 9B and Ornith 1.0 9B interleaved hybrid architecture
//!   (Gated Delta Net recurrent linear attention + multi-head full attention).

use anyhow::{Context, Result, bail, ensure};

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, LayerState};
use crate::model::transformer::{self, DecodeAttnDims, KvView, WeightRef};
use crate::model::{BlockType, Model, ModelConfig, ScalarMultipliers, SsmConfig};

/// Weight references for a full attention layer.
pub(crate) struct AttnLayerRefs {
    pub(crate) attn_q: WeightRef,
    pub(crate) attn_k: WeightRef,
    pub(crate) attn_v: WeightRef,
    pub(crate) attn_output: WeightRef,
    pub(crate) attn_q_norm: Vec<f32>,
    pub(crate) attn_k_norm: Vec<f32>,
}

/// Weight references for a Gated Delta Net recurrent layer.
pub(crate) struct DeltaNetLayerRefs {
    pub(crate) wqkv: WeightRef,
    pub(crate) wqkv_gate: WeightRef,
    pub(crate) ssm_conv1d: Vec<f32>,
    pub(crate) ssm_conv1d_bias: Option<Vec<f32>>,
    pub(crate) ssm_dt: Vec<f32>,
    pub(crate) ssm_a: Vec<f32>,
    pub(crate) ssm_beta: WeightRef,
    pub(crate) ssm_alpha: WeightRef,
    pub(crate) ssm_norm: Vec<f32>,
    pub(crate) ssm_out: WeightRef,
}

pub(crate) enum LayerKindRefs {
    Attention(AttnLayerRefs),
    DeltaNet(DeltaNetLayerRefs),
}

pub(crate) struct LayerRefs {
    pub(crate) kind: LayerKindRefs,
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) attn_post_norm: Vec<f32>,
    pub(crate) ffn_gate: WeightRef,
    pub(crate) ffn_up: WeightRef,
    pub(crate) ffn_down: WeightRef,
}

/// Qwen 3.5 / Ornith 1.0 hybrid model.
pub struct Qwen35Model {
    gguf: GgufFile,
    config: ModelConfig,
    head_dim: usize,
    embd_ref: WeightRef,
    output_norm_weight: Vec<f32>,
    output_ref: Option<WeightRef>,
    layers: Vec<LayerRefs>,
    sliding_window: Option<usize>,
    #[allow(dead_code)]
    model_id: String,
}

impl Qwen35Model {
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
            arch == "qwen35" || arch == "qwen3_5" || arch == "qwen3.5",
            "expected architecture 'qwen35', got '{arch}'"
        );

        let prefix = ["qwen35", "qwen3_5", "qwen3.5", arch.as_str()]
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
            .unwrap_or(1) as usize;
        let n_kv_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(n_heads as u32) as usize;

        ensure!(context_size > 0, "context_size must be > 0");
        ensure!(n_layers > 0, "block_count must be > 0");
        ensure!(hidden_size > 0, "embedding_length must be > 0");
        ensure!(intermediate_size > 0, "feed_forward_length must be > 0");
        ensure!(n_heads > 0, "head_count must be > 0");
        ensure!(n_kv_heads > 0, "head_count_kv must be > 0");
        ensure!(
            n_heads.is_multiple_of(n_kv_heads),
            "head_count ({n_heads}) must be a multiple of head_count_kv ({n_kv_heads})"
        );

        let head_dim = if let Some(hd) = gguf.get_u32(&format!("{prefix}.attention.key_length")) {
            hd as usize
        } else {
            ensure!(
                hidden_size.is_multiple_of(n_heads),
                "hidden_size ({hidden_size}) must be a multiple of head_count ({n_heads})"
            );
            hidden_size.checked_div(n_heads).unwrap_or(0)
        };

        let rms_norm_eps = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);

        let rope_theta = gguf
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(1_000_000.0);

        ensure!(head_dim > 0, "head_dim must be > 0");
        ensure!(
            head_dim.is_multiple_of(2),
            "head_dim ({head_dim}) must be an even integer for RoPE rotation"
        );
        ensure!(
            rms_norm_eps > 0.0 && rms_norm_eps.is_finite(),
            "rms_norm_eps must be positive and finite"
        );
        ensure!(
            rope_theta > 0.0 && rope_theta.is_finite(),
            "rope_theta must be positive and finite"
        );

        // SSM / Delta Net parameters
        let ssm_d_conv = gguf
            .get_u32(&format!("{prefix}.ssm.conv_kernel"))
            .unwrap_or(4) as usize;
        ensure!(
            ssm_d_conv >= 1,
            "ssm.conv_kernel must be >= 1, got {ssm_d_conv}"
        );

        let ssm_d_state = gguf
            .get_u32(&format!("{prefix}.ssm.state_size"))
            .unwrap_or(head_dim as u32) as usize;
        ensure!(
            ssm_d_state > 0,
            "ssm.state_size must be > 0, got {ssm_d_state}"
        );
        ensure!(
            ssm_d_state <= 256,
            "ssm.state_size must be <= 256, got {ssm_d_state}"
        );

        let ssm_dt_rank = gguf
            .get_u32(&format!("{prefix}.ssm.time_step_rank"))
            .unwrap_or(n_heads as u32) as usize;
        ensure!(
            ssm_dt_rank > 0,
            "ssm.time_step_rank must be > 0, got {ssm_dt_rank}"
        );
        ensure!(
            ssm_dt_rank <= 128,
            "ssm.time_step_rank must be <= 128, got {ssm_dt_rank}"
        );

        let ssm_n_group = gguf
            .get_u32(&format!("{prefix}.ssm.group_count"))
            .unwrap_or(n_kv_heads as u32) as usize;
        ensure!(
            ssm_n_group > 0,
            "ssm.group_count must be > 0, got {ssm_n_group}"
        );
        ensure!(
            ssm_dt_rank.is_multiple_of(ssm_n_group),
            "ssm.time_step_rank ({ssm_dt_rank}) must be a multiple of ssm.group_count ({ssm_n_group})"
        );

        let ssm_d_inner = gguf
            .get_u32(&format!("{prefix}.ssm.inner_size"))
            .map(|v| v as usize)
            .unwrap_or_else(|| ssm_dt_rank.saturating_mul(ssm_d_state));
        ensure!(
            Some(ssm_d_inner) == ssm_dt_rank.checked_mul(ssm_d_state),
            "ssm.inner_size ({ssm_d_inner}) must equal dt_rank * d_state ({} * {})",
            ssm_dt_rank,
            ssm_d_state
        );

        let full_attention_interval = gguf
            .get_u32(&format!("{prefix}.full_attention_interval"))
            .unwrap_or(4) as usize;
        ensure!(
            full_attention_interval > 0,
            "full_attention_interval must be > 0, got {full_attention_interval}"
        );

        let sliding_window = gguf
            .get_u32(&format!("{prefix}.attention.sliding_window"))
            .or_else(|| gguf.get_u32("attention.sliding_window"))
            .map(|sw| sw as usize);

        // Determine per-layer block types
        let mut block_types = Vec::with_capacity(n_layers);
        let mut kv_heads_per_layer = Vec::with_capacity(n_layers);

        if let Some(recr_layers) =
            gguf.get_bool_array(&format!("{prefix}.attention.recurrent_layers"))
        {
            ensure!(
                recr_layers.len() >= n_layers,
                "{prefix}.attention.recurrent_layers length ({}) must be >= block_count ({n_layers})",
                recr_layers.len()
            );
            for &is_recr in recr_layers.iter().take(n_layers) {
                if is_recr {
                    block_types.push(BlockType::DeltaNet);
                    kv_heads_per_layer.push(0);
                } else {
                    block_types.push(BlockType::Attention);
                    kv_heads_per_layer.push(n_kv_heads);
                }
            }
        } else {
            for il in 0..n_layers {
                let is_recr = (il + 1) % full_attention_interval != 0;
                if is_recr {
                    block_types.push(BlockType::DeltaNet);
                    kv_heads_per_layer.push(0);
                } else {
                    block_types.push(BlockType::Attention);
                    kv_heads_per_layer.push(n_kv_heads);
                }
            }
        }

        let ssm_cfg = SsmConfig {
            d_conv: ssm_d_conv,
            d_inner: ssm_d_inner,
            d_state: ssm_d_state,
            dt_rank: ssm_dt_rank,
            n_group: ssm_n_group,
        };

        let embd_ref = transformer::resolve_weight(&gguf, "token_embd.weight")?;
        let vocab_size = embd_ref.m;
        ensure!(
            embd_ref.k == hidden_size,
            "token_embd.weight hidden dim {} != expected {hidden_size}",
            embd_ref.k
        );

        let output_ref = transformer::resolve_weight(&gguf, "output.weight").ok();
        if let Some(ref out) = output_ref {
            ensure!(
                out.m == vocab_size && out.k == hidden_size,
                "output.weight shape [{}, {}] != expected [{vocab_size}, {hidden_size}]",
                out.m,
                out.k
            );
        }
        let output_norm_weight = gguf.get_tensor("output_norm.weight")?.to_f32_vec();
        ensure!(
            output_norm_weight.len() == hidden_size,
            "output_norm.weight length {} != hidden_size {hidden_size}",
            output_norm_weight.len()
        );

        let ssm_value_dim = ssm_dt_rank
            .checked_mul(ssm_d_state)
            .context("ssm value_dim overflow")?;
        let ssm_key_dim = ssm_n_group
            .checked_mul(ssm_d_state)
            .context("ssm key_dim overflow")?;
        let ssm_conv_dim = ssm_key_dim
            .checked_mul(2)
            .and_then(|k2| k2.checked_add(ssm_value_dim))
            .context("ssm conv_dim overflow")?;
        let ssm_conv_total = ssm_conv_dim
            .checked_mul(ssm_d_conv)
            .context("ssm conv1d total elements overflow")?;

        let mut layers = Vec::with_capacity(n_layers);
        for (il, &block_type) in block_types.iter().enumerate().take(n_layers) {
            let attn_norm = gguf
                .get_tensor(&format!("blk.{il}.attn_norm.weight"))
                .with_context(|| format!("missing blk.{il}.attn_norm.weight"))?
                .to_f32_vec();
            let attn_post_norm = gguf
                .get_tensor(&format!("blk.{il}.attn_post_norm.weight"))
                .or_else(|_| gguf.get_tensor(&format!("blk.{il}.ffn_norm.weight")))
                .with_context(|| {
                    format!("missing blk.{il}.attn_post_norm.weight or ffn_norm.weight")
                })?
                .to_f32_vec();

            let ffn_gate =
                transformer::resolve_weight(&gguf, &format!("blk.{il}.ffn_gate.weight"))?;
            let ffn_up = transformer::resolve_weight(&gguf, &format!("blk.{il}.ffn_up.weight"))?;
            let ffn_down =
                transformer::resolve_weight(&gguf, &format!("blk.{il}.ffn_down.weight"))?;

            let kind = match block_type {
                BlockType::DeltaNet => {
                    let wqkv =
                        transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_qkv.weight"))
                            .or_else(|_| {
                                transformer::resolve_weight(
                                    &gguf,
                                    &format!("blk.{il}.ssm_qkv.weight"),
                                )
                            })?;
                    let wqkv_gate =
                        transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_gate.weight"))
                            .or_else(|_| {
                            transformer::resolve_weight(&gguf, &format!("blk.{il}.ssm_gate.weight"))
                        })?;
                    let ssm_conv1d = gguf
                        .get_tensor(&format!("blk.{il}.ssm_conv1d.weight"))
                        .with_context(|| format!("missing blk.{il}.ssm_conv1d.weight"))?
                        .to_f32_vec();
                    let ssm_conv1d_bias = gguf
                        .get_tensor(&format!("blk.{il}.ssm_conv1d.bias"))
                        .ok()
                        .map(|t| t.to_f32_vec());
                    let ssm_dt = gguf
                        .get_tensor(&format!("blk.{il}.ssm_dt.bias"))
                        .or_else(|_| gguf.get_tensor(&format!("blk.{il}.ssm_dt")))
                        .with_context(|| format!("missing blk.{il}.ssm_dt.bias"))?
                        .to_f32_vec();
                    let ssm_a = gguf
                        .get_tensor(&format!("blk.{il}.ssm_a"))
                        .or_else(|_| gguf.get_tensor(&format!("blk.{il}.ssm_a.weight")))
                        .with_context(|| format!("missing blk.{il}.ssm_a"))?
                        .to_f32_vec();
                    let ssm_beta =
                        transformer::resolve_weight(&gguf, &format!("blk.{il}.ssm_beta.weight"))?;
                    let ssm_alpha =
                        transformer::resolve_weight(&gguf, &format!("blk.{il}.ssm_alpha.weight"))?;
                    let ssm_norm = gguf
                        .get_tensor(&format!("blk.{il}.ssm_norm.weight"))
                        .or_else(|_| gguf.get_tensor(&format!("blk.{il}.ssm_norm")))
                        .with_context(|| format!("missing blk.{il}.ssm_norm.weight"))?
                        .to_f32_vec();
                    let ssm_out =
                        transformer::resolve_weight(&gguf, &format!("blk.{il}.ssm_out.weight"))
                            .or_else(|_| {
                                transformer::resolve_weight(
                                    &gguf,
                                    &format!("blk.{il}.ssm_output.weight"),
                                )
                            })?;

                    ensure!(
                        wqkv.m == ssm_conv_dim && wqkv.k == hidden_size,
                        "blk.{il}.attn_qkv shape [{}, {}] != expected [{ssm_conv_dim}, {hidden_size}]",
                        wqkv.m,
                        wqkv.k
                    );
                    ensure!(
                        wqkv_gate.m == ssm_value_dim && wqkv_gate.k == hidden_size,
                        "blk.{il}.attn_gate shape [{}, {}] != expected [{ssm_value_dim}, {hidden_size}]",
                        wqkv_gate.m,
                        wqkv_gate.k
                    );
                    ensure!(
                        ssm_conv1d.len() == ssm_conv_total,
                        "blk.{il}.ssm_conv1d length {} != expected {}",
                        ssm_conv1d.len(),
                        ssm_conv_total
                    );
                    if let Some(ref bias) = ssm_conv1d_bias {
                        ensure!(
                            bias.len() == ssm_conv_dim,
                            "blk.{il}.ssm_conv1d.bias length {} != expected {}",
                            bias.len(),
                            ssm_conv_dim
                        );
                    }
                    ensure!(
                        ssm_dt.len() == ssm_dt_rank,
                        "blk.{il}.ssm_dt length {} != expected {}",
                        ssm_dt.len(),
                        ssm_dt_rank
                    );
                    ensure!(
                        ssm_a.len() == ssm_dt_rank,
                        "blk.{il}.ssm_a length {} != expected {}",
                        ssm_a.len(),
                        ssm_dt_rank
                    );
                    ensure!(
                        ssm_beta.m == ssm_dt_rank && ssm_beta.k == hidden_size,
                        "blk.{il}.ssm_beta shape [{}, {}] != expected [{ssm_dt_rank}, {hidden_size}]",
                        ssm_beta.m,
                        ssm_beta.k
                    );
                    ensure!(
                        ssm_alpha.m == ssm_dt_rank && ssm_alpha.k == hidden_size,
                        "blk.{il}.ssm_alpha shape [{}, {}] != expected [{ssm_dt_rank}, {hidden_size}]",
                        ssm_alpha.m,
                        ssm_alpha.k
                    );
                    ensure!(
                        ssm_norm.len() == ssm_d_state,
                        "blk.{il}.ssm_norm length {} != expected {}",
                        ssm_norm.len(),
                        ssm_d_state
                    );
                    ensure!(
                        ssm_out.m == hidden_size && ssm_out.k == ssm_value_dim,
                        "blk.{il}.ssm_out shape [{}, {}] != expected [{hidden_size}, {ssm_value_dim}]",
                        ssm_out.m,
                        ssm_out.k
                    );

                    LayerKindRefs::DeltaNet(DeltaNetLayerRefs {
                        wqkv,
                        wqkv_gate,
                        ssm_conv1d,
                        ssm_conv1d_bias,
                        ssm_dt,
                        ssm_a,
                        ssm_beta,
                        ssm_alpha,
                        ssm_norm,
                        ssm_out,
                    })
                }
                BlockType::Attention => {
                    let attn_q =
                        transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_q.weight"))?;
                    let attn_k =
                        transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_k.weight"))?;
                    let attn_v =
                        transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_v.weight"))?;
                    let attn_output =
                        transformer::resolve_weight(&gguf, &format!("blk.{il}.attn_output.weight"))
                            .or_else(|_| {
                                transformer::resolve_weight(
                                    &gguf,
                                    &format!("blk.{il}.attn_out.weight"),
                                )
                            })?;
                    let attn_q_norm = gguf
                        .get_tensor(&format!("blk.{il}.attn_q_norm.weight"))
                        .with_context(|| format!("missing blk.{il}.attn_q_norm.weight"))?
                        .to_f32_vec();
                    let attn_k_norm = gguf
                        .get_tensor(&format!("blk.{il}.attn_k_norm.weight"))
                        .with_context(|| format!("missing blk.{il}.attn_k_norm.weight"))?
                        .to_f32_vec();

                    ensure!(
                        (attn_q.m == n_heads * head_dim || attn_q.m == 2 * n_heads * head_dim)
                            && attn_q.k == hidden_size,
                        "blk.{il}.attn_q shape [{}, {}] unexpected for n_heads={n_heads}, head_dim={head_dim}, hidden_size={hidden_size}",
                        attn_q.m,
                        attn_q.k
                    );
                    ensure!(
                        attn_k.m == n_kv_heads * head_dim && attn_k.k == hidden_size,
                        "blk.{il}.attn_k shape [{}, {}] != expected [{}, {hidden_size}]",
                        attn_k.m,
                        attn_k.k,
                        n_kv_heads * head_dim
                    );
                    ensure!(
                        attn_v.m == n_kv_heads * head_dim && attn_v.k == hidden_size,
                        "blk.{il}.attn_v shape [{}, {}] != expected [{}, {hidden_size}]",
                        attn_v.m,
                        attn_v.k,
                        n_kv_heads * head_dim
                    );
                    ensure!(
                        attn_output.m == hidden_size && attn_output.k == n_heads * head_dim,
                        "blk.{il}.attn_output shape [{}, {}] != expected [{hidden_size}, {}]",
                        attn_output.m,
                        attn_output.k,
                        n_heads * head_dim
                    );
                    ensure!(
                        attn_q_norm.len() == head_dim,
                        "blk.{il}.attn_q_norm length {} != head_dim {}",
                        attn_q_norm.len(),
                        head_dim
                    );
                    ensure!(
                        attn_k_norm.len() == head_dim,
                        "blk.{il}.attn_k_norm length {} != head_dim {}",
                        attn_k_norm.len(),
                        head_dim
                    );

                    LayerKindRefs::Attention(AttnLayerRefs {
                        attn_q,
                        attn_k,
                        attn_v,
                        attn_output,
                        attn_q_norm,
                        attn_k_norm,
                    })
                }
                other => bail!("unsupported layer type {other:?} in Qwen35Model"),
            };

            ensure!(
                attn_norm.len() == hidden_size,
                "blk.{il}.attn_norm length {} != hidden_size {hidden_size}",
                attn_norm.len()
            );
            ensure!(
                attn_post_norm.len() == hidden_size,
                "blk.{il}.attn_post_norm length {} != hidden_size {hidden_size}",
                attn_post_norm.len()
            );
            ensure!(
                ffn_gate.m == intermediate_size && ffn_gate.k == hidden_size,
                "blk.{il}.ffn_gate shape [{}, {}] != expected [{intermediate_size}, {hidden_size}]",
                ffn_gate.m,
                ffn_gate.k
            );
            ensure!(
                ffn_up.m == intermediate_size && ffn_up.k == hidden_size,
                "blk.{il}.ffn_up shape [{}, {}] != expected [{intermediate_size}, {hidden_size}]",
                ffn_up.m,
                ffn_up.k
            );
            ensure!(
                ffn_down.m == hidden_size && ffn_down.k == intermediate_size,
                "blk.{il}.ffn_down shape [{}, {}] != expected [{hidden_size}, {intermediate_size}]",
                ffn_down.m,
                ffn_down.k
            );

            layers.push(LayerRefs {
                kind,
                attn_norm,
                attn_post_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        let max_seq_len = gguf
            .get_u32(&format!("{prefix}.context_length"))
            .unwrap_or(2048) as usize;

        let config = ModelConfig {
            architecture: arch,
            n_layers,
            hidden_size,
            intermediate_size,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            max_seq_len: max_seq_len.min(context_size),
            rope_theta,
            rms_norm_eps,
            block_types,
            conv_kernel_size: None,
            ssm: Some(ssm_cfg),
            kv_heads_per_layer,
            scalars: ScalarMultipliers::default(),
            moe: None,
            is_causal: true,
            class_labels: Vec::new(),
        };

        Ok(Self {
            gguf,
            config,
            head_dim,
            embd_ref,
            output_norm_weight,
            output_ref,
            layers,
            sliding_window,
            model_id,
        })
    }

    /// Process a single token through one Gated Delta Net recurrent block.
    fn forward_deltanet_block(
        &self,
        layer: usize,
        normed: &[f32],
        state: &mut InferenceState,
        refs: &DeltaNetLayerRefs,
        out: &mut [f32],
    ) {
        let Some(ssm) = &self.config.ssm else {
            return;
        };
        let hidden_size = self.config.hidden_size;
        let num_k_heads = ssm.n_group;
        let num_v_heads = ssm.dt_rank;
        let head_k_dim = ssm.d_state;
        let head_v_dim = ssm.d_state;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let d_conv = ssm.d_conv;
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
        let expected_conv_len = conv_dim * (d_conv.saturating_sub(1));
        let expected_ssm_len = num_v_heads * head_v_dim * head_v_dim;
        if conv_state.len() < expected_conv_len || ssm_state.len() < expected_ssm_len {
            return;
        }

        // 1. In-projections
        let mut qkv_mixed = std::mem::take(&mut state.scratch.ssm_in_proj);
        let mut conv_out = std::mem::take(&mut state.scratch.ssm_conv_out);
        let mut z = std::mem::take(&mut state.scratch.ssm_y);
        let mut core_out = std::mem::take(&mut state.scratch.ssm_branch_out);

        qkv_mixed.resize(conv_dim, 0.0);
        conv_out.resize(conv_dim, 0.0);
        z.resize(value_dim, 0.0);
        core_out.resize(value_dim, 0.0);

        let mut beta_raw_buf = [0.0f32; 128];
        let mut alpha_raw_buf = [0.0f32; 128];
        let beta_raw = &mut beta_raw_buf[..num_v_heads];
        let alpha_raw = &mut alpha_raw_buf[..num_v_heads];

        transformer::gemv(&self.gguf, &refs.wqkv, normed, &mut qkv_mixed);
        transformer::gemv(&self.gguf, &refs.wqkv_gate, normed, &mut z);
        transformer::gemv(&self.gguf, &refs.ssm_beta, normed, beta_raw);
        transformer::gemv(&self.gguf, &refs.ssm_alpha, normed, alpha_raw);

        // 2. Activations (beta sigmoid in-place, gate softplus * ssm_a in-place into alpha_raw)
        cpu::sigmoid_inplace(beta_raw);
        for (h, a) in alpha_raw.iter_mut().enumerate().take(num_v_heads) {
            let alpha_biased = *a + refs.ssm_dt[h];
            let alpha_sp = cpu::softplus(alpha_biased);
            let gate_h = (alpha_sp * refs.ssm_a[h]).clamp(-80.0, 0.0);
            *a = gate_h.exp();
        }

        // 3. Causal Depthwise Conv1D with SiLU
        cpu::mamba2_conv1d_step(
            &qkv_mixed,
            conv_state,
            &refs.ssm_conv1d,
            refs.ssm_conv1d_bias.as_deref(),
            conv_dim,
            d_conv,
            &mut conv_out,
        );

        // 4. Split conv_out into Q, K, V
        let (q_part, rest) = conv_out.split_at_mut(key_dim);
        let (k_part, v_part) = rest.split_at_mut(key_dim);

        // 5. Per-head L2 norm on Q and K, with query scaling 1/sqrt(head_k_dim)
        let q_scale = 1.0 / (head_k_dim as f32).sqrt();
        for kh in 0..num_k_heads {
            let q_head = &mut q_part[kh * head_k_dim..(kh + 1) * head_k_dim];
            let sum_sq = cpu::dot_f32(q_head, q_head);
            let l2 = sum_sq.sqrt().max(eps);
            let factor = q_scale / l2;
            cpu::scale_inplace(q_head, factor);

            let k_head = &mut k_part[kh * head_k_dim..(kh + 1) * head_k_dim];
            let sum_sq_k = cpu::dot_f32(k_head, k_head);
            let l2_k = sum_sq_k.sqrt().max(eps);
            let factor_k = 1.0 / l2_k;
            cpu::scale_inplace(k_head, factor_k);
        }

        // 6. Gated Delta Net recurrence update
        let s_dim = head_v_dim;
        let mut sk_buf = [0.0f32; 256];
        let mut d_buf = [0.0f32; 256];
        let sk = &mut sk_buf[..s_dim];
        let d = &mut d_buf[..s_dim];
        let heads_per_group = (num_v_heads / num_k_heads.max(1)).max(1);

        for h in 0..num_v_heads {
            let kh = (h / heads_per_group).min(num_k_heads.saturating_sub(1));
            let q = &q_part[kh * head_k_dim..(kh + 1) * head_k_dim];
            let k = &k_part[kh * head_k_dim..(kh + 1) * head_k_dim];
            let v = &v_part[h * head_v_dim..(h + 1) * head_v_dim];
            let dec = alpha_raw[h];
            let b = beta_raw[h];

            let state_offset = h * s_dim * s_dim;
            let s_mat = &mut ssm_state[state_offset..state_offset + s_dim * s_dim];

            // 6a. Decay state: S <- S * decay
            if !dec.is_finite() || dec <= 0.0 {
                s_mat.fill(0.0);
            } else if (dec - 1.0).abs() > 1e-7 {
                cpu::scale_inplace(s_mat, dec);
            }

            // 6b. Memory read: sk[j] = sum_i S[i * s_dim + j] * k[i]
            sk.fill(0.0);
            for i in 0..s_dim {
                let ki = k[i];
                if ki == 0.0 {
                    continue;
                }
                let row = &s_mat[i * s_dim..(i + 1) * s_dim];
                for (sk_elem, &r) in sk.iter_mut().zip(row.iter()) {
                    *sk_elem += r * ki;
                }
            }

            // 6c. Delta error: d = beta * (v - sk)
            for ((dj, &vj), &skj) in d.iter_mut().zip(v.iter()).zip(sk.iter()) {
                *dj = b * (vj - skj);
            }

            // 6d & 6e. Fused Associative write and Output read
            let o_head = &mut core_out[h * head_v_dim..(h + 1) * head_v_dim];
            o_head.fill(0.0);
            for i in 0..s_dim {
                let ki = k[i];
                let qi = q[i];
                let row_offset = i * s_dim;
                let row = &mut s_mat[row_offset..row_offset + s_dim];
                for j in 0..s_dim {
                    let updated = row[j] + ki * d[j];
                    row[j] = updated;
                    o_head[j] += updated * qi;
                }
            }
        }

        // 7. Gated RMSNorm: RMSNorm(o, ssm_norm) * silu(z) per value head
        cpu::silu_inplace(&mut z);
        for h in 0..num_v_heads {
            let o_head = &mut core_out[h * head_v_dim..(h + 1) * head_v_dim];
            let z_head = &z[h * head_v_dim..(h + 1) * head_v_dim];

            cpu::rmsnorm(o_head, &refs.ssm_norm, eps);
            cpu::mul_inplace(o_head, z_head);
        }

        // 8. Out-projection: ssm_out * core_out -> out
        transformer::gemv(
            &self.gguf,
            &refs.ssm_out,
            &core_out,
            &mut out[..hidden_size],
        );

        state.scratch.ssm_in_proj = qkv_mixed;
        state.scratch.ssm_conv_out = conv_out;
        state.scratch.ssm_y = z;
        state.scratch.ssm_branch_out = core_out;
    }

    /// Process a single token through one full Attention block.
    fn forward_attention_block(
        &self,
        layer: usize,
        normed: &[f32],
        pos: usize,
        state: &mut InferenceState,
        refs: &AttnLayerRefs,
        out: &mut [f32],
    ) {
        let hidden_size = self.config.hidden_size;
        if out.len() < hidden_size || normed.len() < hidden_size {
            return;
        }
        let Some(&n_kv_heads) = self.config.kv_heads_per_layer.get(layer) else {
            return;
        };
        let n_heads = self.config.n_heads;
        let head_dim = self.head_dim;
        let eps = self.config.rms_norm_eps;

        let q_out_dim = refs.attn_q.m;
        let has_gate = q_out_dim == 2 * n_heads * head_dim;

        let mut q_full = std::mem::take(&mut state.scratch.conv_proj);
        let mut q = std::mem::take(&mut state.scratch.q);
        let mut k = std::mem::take(&mut state.scratch.k);
        let mut v = std::mem::take(&mut state.scratch.v);
        let mut attn_out = std::mem::take(&mut state.scratch.attn_out);
        let mut gate = if has_gate {
            let mut g = std::mem::take(&mut state.scratch.conv_scratch);
            g.resize(n_heads * head_dim, 0.0);
            Some(g)
        } else {
            None
        };

        q_full.resize(q_out_dim, 0.0);
        q.resize(n_heads * head_dim, 0.0);
        k.resize(n_kv_heads * head_dim, 0.0);
        v.resize(n_kv_heads * head_dim, 0.0);
        attn_out.resize(n_heads * head_dim, 0.0);

        transformer::gemv(&self.gguf, &refs.attn_q, normed, &mut q_full);

        if let Some(ref mut gate_buf) = gate {
            for h in 0..n_heads {
                let src = &q_full[h * 2 * head_dim..(h + 1) * 2 * head_dim];
                q[h * head_dim..(h + 1) * head_dim].copy_from_slice(&src[..head_dim]);
                gate_buf[h * head_dim..(h + 1) * head_dim].copy_from_slice(&src[head_dim..]);
            }
        } else {
            q.copy_from_slice(&q_full[..n_heads * head_dim]);
        }

        // Apply Q RMSNorm per head
        for h in 0..n_heads {
            let q_head = &mut q[h * head_dim..(h + 1) * head_dim];
            cpu::rmsnorm(q_head, &refs.attn_q_norm, eps);
        }

        // K projection
        transformer::gemv(&self.gguf, &refs.attn_k, normed, &mut k);

        // Apply K RMSNorm per head
        for kh in 0..n_kv_heads {
            let k_head = &mut k[kh * head_dim..(kh + 1) * head_dim];
            cpu::rmsnorm(k_head, &refs.attn_k_norm, eps);
        }

        // V projection
        transformer::gemv(&self.gguf, &refs.attn_v, normed, &mut v);

        // Apply RoPE to Q and K
        cpu::rope(
            &mut q,
            &mut k,
            pos,
            n_heads,
            n_kv_heads,
            head_dim,
            self.config.rope_theta,
        );

        // Append K and V to cache
        if state.kv_f16 {
            state.append_kv_f16(layer, &k, &v);
        } else {
            state.append_kv(layer, &k, &v);
        }

        // Decode attention
        let kv = match &state.layers.get(layer) {
            Some(LayerState::Attention {
                key_cache,
                value_cache,
                key_cache_f16,
                value_cache_f16,
                ..
            }) => {
                if state.kv_f16 {
                    KvView::F16 {
                        k: key_cache_f16,
                        v: value_cache_f16,
                    }
                } else {
                    KvView::F32 {
                        k: key_cache,
                        v: value_cache,
                    }
                }
            }
            _ => {
                state.scratch.conv_proj = q_full;
                state.scratch.q = q;
                state.scratch.k = k;
                state.scratch.v = v;
                state.scratch.attn_out = attn_out;
                if let Some(g) = gate {
                    state.scratch.conv_scratch = g;
                }
                return;
            }
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let d = DecodeAttnDims {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            seq_len: pos + 1,
            attn_logit_softcapping: None,
            sliding_window: self.sliding_window,
        };

        transformer::decode_attention(&q, &kv, &d, &mut attn_out, &mut state.scratch.scores);

        // Multiply by sigmoid(gate) if present
        if let Some(mut g) = gate {
            cpu::sigmoid_inplace(&mut g);
            cpu::mul_inplace(&mut attn_out, &g);
            state.scratch.conv_scratch = g;
        }

        // Out projection
        transformer::gemv(
            &self.gguf,
            &refs.attn_output,
            &attn_out,
            &mut out[..hidden_size],
        );

        state.scratch.conv_proj = q_full;
        state.scratch.q = q;
        state.scratch.k = k;
        state.scratch.v = v;
        state.scratch.attn_out = attn_out;
    }

    /// Run single token through all layers.
    fn run_layers(&self, hidden: &mut [f32], pos: usize, state: &mut InferenceState) {
        let hs = self.config.hidden_size;
        let hidden = &mut hidden[..hs];
        let mut normed = std::mem::take(&mut state.scratch.normed);
        let mut layer_out = std::mem::take(&mut state.scratch.out);
        let mut ffn_in = std::mem::take(&mut state.scratch.ffn_input);
        let mut ffn_gate = std::mem::take(&mut state.scratch.gate);
        let mut ffn_up = std::mem::take(&mut state.scratch.up);
        let mut ffn_out = std::mem::take(&mut state.scratch.lora_tmp);

        normed.resize(hs, 0.0);
        layer_out.resize(hs, 0.0);
        ffn_in.resize(hs, 0.0);
        ffn_gate.resize(self.config.intermediate_size, 0.0);
        ffn_up.resize(self.config.intermediate_size, 0.0);
        ffn_out.resize(hs, 0.0);

        for (il, layer_ref) in self.layers.iter().enumerate() {
            normed.copy_from_slice(hidden);
            cpu::rmsnorm(&mut normed, &layer_ref.attn_norm, self.config.rms_norm_eps);
            layer_out.fill(0.0);

            match &layer_ref.kind {
                LayerKindRefs::DeltaNet(dnet_refs) => {
                    self.forward_deltanet_block(il, &normed, state, dnet_refs, &mut layer_out);
                }
                LayerKindRefs::Attention(attn_refs) => {
                    self.forward_attention_block(
                        il,
                        &normed,
                        pos,
                        state,
                        attn_refs,
                        &mut layer_out,
                    );
                }
            }

            // Residual connection: hidden += layer_out
            cpu::add_inplace(hidden, &layer_out);

            // Post-attention norm
            ffn_in.copy_from_slice(hidden);
            cpu::rmsnorm(
                &mut ffn_in,
                &layer_ref.attn_post_norm,
                self.config.rms_norm_eps,
            );

            // SwiGLU FFN
            transformer::gemv(&self.gguf, &layer_ref.ffn_gate, &ffn_in, &mut ffn_gate);
            transformer::gemv(&self.gguf, &layer_ref.ffn_up, &ffn_in, &mut ffn_up);
            cpu::silu_mul_inplace(&mut ffn_gate, &ffn_up);
            transformer::gemv(&self.gguf, &layer_ref.ffn_down, &ffn_gate, &mut ffn_out);

            // FFN residual connection
            cpu::add_inplace(hidden, &ffn_out);

            if transformer::oracle_dump::is_active() {
                transformer::oracle_dump::record(&format!("l_out-{il}"), hidden);
            }
        }

        state.scratch.normed = normed;
        state.scratch.out = layer_out;
        state.scratch.ffn_input = ffn_in;
        state.scratch.gate = ffn_gate;
        state.scratch.up = ffn_up;
        state.scratch.lora_tmp = ffn_out;
    }

    /// Project final hidden state to logits.
    fn project_logits(&self, hidden: &[f32], state: &mut InferenceState) -> Vec<f32> {
        let vocab_size = self.config.vocab_size;
        let out_ref = self.output_ref.as_ref().unwrap_or(&self.embd_ref);
        if state.scratch.logits.len() < out_ref.m {
            state.scratch.logits.resize(out_ref.m, 0.0);
        }

        transformer::gemv(
            &self.gguf,
            out_ref,
            &hidden[..out_ref.k],
            &mut state.scratch.logits[..out_ref.m],
        );

        state.scratch.logits[..vocab_size].to_vec()
    }
}

impl Model for Qwen35Model {
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(tokens.len(), 1, "forward() expects exactly 1 token");
        assert_eq!(
            pos, state.seq_len,
            "forward: pos ({pos}) must match state.seq_len ({})",
            state.seq_len
        );
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        assert!(
            token_id < cfg.vocab_size,
            "token_id {token_id} out of range (vocab_size={})",
            cfg.vocab_size
        );

        let mut hidden = std::mem::take(&mut state.scratch.hidden_in);
        hidden.resize(cfg.hidden_size, 0.0);
        transformer::dequantize_row_into(
            &self.gguf,
            &self.embd_ref,
            token_id,
            &mut hidden[..cfg.hidden_size],
        );
        if transformer::oracle_dump::is_active() {
            transformer::oracle_dump::record("embd", &hidden[..cfg.hidden_size]);
        }
        self.run_layers(&mut hidden[..cfg.hidden_size], pos, state);
        cpu::rmsnorm(
            &mut hidden[..cfg.hidden_size],
            &self.output_norm_weight,
            self.config.rms_norm_eps,
        );
        if transformer::oracle_dump::is_active() {
            transformer::oracle_dump::record("result_norm", &hidden[..cfg.hidden_size]);
        }
        let logits = self.project_logits(&hidden[..cfg.hidden_size], state);
        if transformer::oracle_dump::is_active() {
            transformer::oracle_dump::record("result_output", &logits);
        }
        state.seq_len = pos + 1;
        state.scratch.hidden_in = hidden;
        logits
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        if tokens.is_empty() {
            return Vec::new();
        }
        assert_eq!(
            start_pos, state.seq_len,
            "start_pos ({start_pos}) must match state.seq_len ({})",
            state.seq_len
        );

        let n = tokens.len();
        let cfg = &self.config;

        if n > 1 {
            let mut hidden = std::mem::take(&mut state.scratch.hidden_in);
            hidden.resize(cfg.hidden_size, 0.0);
            for (i, &token) in tokens[..n - 1].iter().enumerate() {
                let token_id = token as usize;
                assert!(
                    token_id < cfg.vocab_size,
                    "token_id {token_id} out of range (vocab_size={})",
                    cfg.vocab_size
                );
                transformer::dequantize_row_into(
                    &self.gguf,
                    &self.embd_ref,
                    token_id,
                    &mut hidden[..cfg.hidden_size],
                );
                if transformer::oracle_dump::is_active() {
                    transformer::oracle_dump::record("embd", &hidden[..cfg.hidden_size]);
                }
                self.run_layers(&mut hidden[..cfg.hidden_size], start_pos + i, state);
                state.seq_len = start_pos + i + 1;
            }
            state.scratch.hidden_in = hidden;
        }

        self.forward(&[tokens[n - 1]], start_pos + n - 1, state)
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn f16_kv_supported(&self) -> bool {
        true
    }

    fn supports_kv_shift(&self) -> bool {
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
