//! Hybrid Attention + Mamba-2 SSM model implementation.
//!
//! Supports:
//! - `granitehybrid` / `granite-hybrid`: Granite 4.0-h (interleaved Attention and Mamba-2 SSM)
//! - `falcon-h1` / `falcon_h1`: Falcon H1R (parallel Attention and Mamba-2 SSM)
//! - `mamba2`: Pure Mamba-2 SSM recurrent models

use anyhow::{Context, Result, ensure};

use crate::backend::cpu::{self, RopeType};
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, LayerState};
use crate::model::transformer::{self, AttnDims, AttnExtras, AttnWeights, FfnWeights, WeightRef};
use crate::model::{BlockType, Model, ModelConfig, ScalarMultipliers, SsmConfig};

// ── Per-layer weight references ─────────────────────────────────────────────

pub(crate) struct LayerWeightRefs {
    pub(crate) attn_q: Option<WeightRef>,
    pub(crate) attn_k: Option<WeightRef>,
    pub(crate) attn_v: Option<WeightRef>,
    pub(crate) attn_output: Option<WeightRef>,
    pub(crate) ssm_in: Option<WeightRef>,
    pub(crate) ssm_out: Option<WeightRef>,
    pub(crate) ffn_gate: WeightRef,
    pub(crate) ffn_up: WeightRef,
    pub(crate) ffn_down: WeightRef,
}

// ── Hybrid Model ────────────────────────────────────────────────────────────

pub struct HybridModel {
    gguf: GgufFile,
    config: ModelConfig,
    head_dim: usize,
    rope_type: RopeType,
    rope_freqs: Option<Vec<f32>>,
    output_norm_weight: Vec<f32>,
    attn_norm_weights: Vec<Vec<f32>>,
    ffn_norm_weights: Vec<Vec<f32>>,
    // Mamba-2 SSM weights
    ssm_conv1d_weights: Vec<Option<Vec<f32>>>,
    ssm_conv1d_biases: Vec<Option<Vec<f32>>>,
    ssm_dt_biases: Vec<Option<Vec<f32>>>,
    ssm_a_weights: Vec<Option<Vec<f32>>>,
    ssm_d_weights: Vec<Option<Vec<f32>>>,
    ssm_norm_weights: Vec<Option<Vec<f32>>>,
    // Attention biases
    attn_q_bias: Vec<Option<Vec<f32>>>,
    attn_k_bias: Vec<Option<Vec<f32>>>,
    attn_v_bias: Vec<Option<Vec<f32>>>,
    #[allow(dead_code)]
    attn_out_bias: Vec<Option<Vec<f32>>>,
    // FFN biases
    #[allow(dead_code)]
    ffn_gate_bias: Vec<Option<Vec<f32>>>,
    #[allow(dead_code)]
    ffn_up_bias: Vec<Option<Vec<f32>>>,
    #[allow(dead_code)]
    ffn_down_bias: Vec<Option<Vec<f32>>>,
    // Weight references
    embd_ref: WeightRef,
    output_ref: Option<WeightRef>,
    layer_refs: Vec<LayerWeightRefs>,
    #[allow(dead_code)]
    model_id: String,
}

impl HybridModel {
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

        let prefix = &arch;
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

        let kv_heads_per_layer: Vec<usize> = if let Some(arr) =
            gguf.get_i32_array(&format!("{prefix}.attention.head_count_kv"))
        {
            arr.into_iter().map(|x| x.max(0) as usize).collect()
        } else if let Some(kv_heads) = gguf.get_u32(&format!("{prefix}.attention.head_count_kv")) {
            vec![kv_heads as usize; n_layers]
        } else {
            vec![n_heads; n_layers]
        };

        let n_kv_heads = kv_heads_per_layer.iter().copied().max().unwrap_or(n_heads);

        let head_dim = gguf
            .get_u32(&format!("{prefix}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or_else(|| hidden_size.checked_div(n_heads).unwrap_or(hidden_size));
        ensure!(head_dim > 0, "head_dim must be > 0");

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

        let gguf_max_seq_len = gguf
            .get_u32(&format!("{prefix}.context_length"))
            .unwrap_or(2048) as usize;
        let max_seq_len = context_size.min(gguf_max_seq_len);
        let rope_theta = gguf
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(10000.0);
        let rms_norm_eps = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        let scalars = ScalarMultipliers::from_gguf(&gguf, prefix)?;

        // Mamba-2 SSM configuration
        let d_conv = gguf
            .get_u32(&format!("{prefix}.ssm.conv_kernel"))
            .unwrap_or(4) as usize;
        let d_inner = gguf
            .get_u32(&format!("{prefix}.ssm.inner_size"))
            .unwrap_or((2 * hidden_size) as u32) as usize;
        let d_state = gguf
            .get_u32(&format!("{prefix}.ssm.state_size"))
            .unwrap_or(16) as usize;
        let dt_rank = gguf
            .get_u32(&format!("{prefix}.ssm.time_step_rank"))
            .unwrap_or(n_heads as u32) as usize;
        let n_group = gguf
            .get_u32(&format!("{prefix}.ssm.group_count"))
            .unwrap_or(1) as usize;

        ensure!(d_conv > 0, "ssm.conv_kernel must be > 0");
        ensure!(d_state > 0, "ssm.state_size must be > 0");
        ensure!(d_inner > 0, "ssm.inner_size must be > 0");
        ensure!(n_group > 0, "ssm.group_count must be > 0");
        ensure!(dt_rank > 0, "ssm.time_step_rank must be > 0");
        ensure!(
            dt_rank.is_multiple_of(n_group),
            "ssm time_step_rank ({dt_rank}) must be a multiple of group_count ({n_group})"
        );
        ensure!(
            d_inner.is_multiple_of(dt_rank),
            "ssm inner_size ({d_inner}) must be a multiple of time_step_rank ({dt_rank})"
        );

        let ssm_config = SsmConfig {
            d_conv,
            d_inner,
            d_state,
            dt_rank,
            n_group,
        };

        // Classify per-layer block types
        let is_falcon = arch == "falcon-h1" || arch == "falcon_h1";
        let is_granite = arch == "granitehybrid" || arch == "granite-hybrid";

        let mut block_types = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let bt = if is_falcon {
                BlockType::ParallelAttentionMamba2
            } else if is_granite {
                if kv_heads_per_layer.get(i).copied().unwrap_or(0) == 0 {
                    BlockType::Mamba2
                } else {
                    BlockType::Attention
                }
            } else if arch == "mamba2" {
                BlockType::Mamba2
            } else {
                let has_ssm = gguf.tensors.contains_key(&format!("blk.{i}.ssm_in.weight"));
                let has_attn = gguf.tensors.contains_key(&format!("blk.{i}.attn_q.weight"));
                if has_ssm && has_attn {
                    BlockType::ParallelAttentionMamba2
                } else if has_ssm {
                    BlockType::Mamba2
                } else {
                    BlockType::Attention
                }
            };
            block_types.push(bt);
        }

        let rope_type = if is_falcon {
            RopeType::Neox
        } else {
            RopeType::Norm
        };

        let rope_freqs = gguf
            .get_tensor("rope_freqs.weight")
            .ok()
            .map(|t| t.to_f32_vec());

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
            block_types: block_types.clone(),
            conv_kernel_size: None,
            ssm: Some(ssm_config),
            kv_heads_per_layer: kv_heads_per_layer.clone(),
            scalars,
            moe: None,
            is_causal: true,
            class_labels: Vec::new(),
        };

        let output_norm_weight = gguf.get_tensor("output_norm.weight")?.to_f32_vec();

        let mut attn_norm_weights = Vec::with_capacity(n_layers);
        let mut ffn_norm_weights = Vec::with_capacity(n_layers);
        let mut ssm_conv1d_weights = Vec::with_capacity(n_layers);
        let mut ssm_conv1d_biases = Vec::with_capacity(n_layers);
        let mut ssm_dt_biases = Vec::with_capacity(n_layers);
        let mut ssm_a_weights = Vec::with_capacity(n_layers);
        let mut ssm_d_weights = Vec::with_capacity(n_layers);
        let mut ssm_norm_weights = Vec::with_capacity(n_layers);

        let mut attn_q_bias = Vec::with_capacity(n_layers);
        let mut attn_k_bias = Vec::with_capacity(n_layers);
        let mut attn_v_bias = Vec::with_capacity(n_layers);
        let mut attn_out_bias = Vec::with_capacity(n_layers);

        let mut ffn_gate_bias = Vec::with_capacity(n_layers);
        let mut ffn_up_bias = Vec::with_capacity(n_layers);
        let mut ffn_down_bias = Vec::with_capacity(n_layers);

        let mut layer_refs = Vec::with_capacity(n_layers);

        for (i, &bt) in block_types.iter().enumerate().take(n_layers) {
            let has_attn = bt == BlockType::Attention || bt == BlockType::ParallelAttentionMamba2;
            let has_ssm = bt == BlockType::Mamba2 || bt == BlockType::ParallelAttentionMamba2;

            attn_norm_weights.push(
                gguf.get_tensor(&format!("blk.{i}.attn_norm.weight"))?
                    .to_f32_vec(),
            );

            let ffn_norm = gguf
                .get_tensor(&format!("blk.{i}.ffn_norm.weight"))
                .or_else(|_| gguf.get_tensor(&format!("blk.{i}.ffn_norm")))?
                .to_f32_vec();
            ffn_norm_weights.push(ffn_norm);

            // SSM tensors
            if has_ssm {
                let conv_dim = d_inner + 2 * n_group * d_state;
                let conv1d_w = gguf
                    .get_tensor(&format!("blk.{i}.ssm_conv1d.weight"))?
                    .to_f32_vec();
                ensure!(
                    conv1d_w.len() == conv_dim * d_conv,
                    "blk.{i}.ssm_conv1d.weight length {} does not match expected conv_dim * d_conv ({})",
                    conv1d_w.len(),
                    conv_dim * d_conv
                );
                ssm_conv1d_weights.push(Some(conv1d_w));

                let conv1d_b = gguf
                    .get_tensor(&format!("blk.{i}.ssm_conv1d.bias"))
                    .ok()
                    .map(|t| t.to_f32_vec());
                if let Some(ref b) = conv1d_b {
                    ensure!(
                        b.len() == conv_dim,
                        "blk.{i}.ssm_conv1d.bias length {} does not match conv_dim ({conv_dim})",
                        b.len()
                    );
                }
                ssm_conv1d_biases.push(conv1d_b);

                let dt_b = gguf
                    .get_tensor(&format!("blk.{i}.ssm_dt.bias"))?
                    .to_f32_vec();
                ensure!(
                    dt_b.len() == dt_rank,
                    "blk.{i}.ssm_dt.bias length {} does not match dt_rank ({dt_rank})",
                    dt_b.len()
                );
                ssm_dt_biases.push(Some(dt_b));

                let a = gguf
                    .get_tensor(&format!("blk.{i}.ssm_a"))
                    .or_else(|_| gguf.get_tensor(&format!("blk.{i}.ssm_a.weight")))?
                    .to_f32_vec();
                ensure!(
                    a.len() == dt_rank,
                    "blk.{i}.ssm_a length {} does not match dt_rank ({dt_rank})",
                    a.len()
                );
                ssm_a_weights.push(Some(a));

                let d = gguf
                    .get_tensor(&format!("blk.{i}.ssm_d"))
                    .or_else(|_| gguf.get_tensor(&format!("blk.{i}.ssm_d.weight")))?
                    .to_f32_vec();
                ensure!(
                    d.len() == dt_rank,
                    "blk.{i}.ssm_d length {} does not match dt_rank ({dt_rank})",
                    d.len()
                );
                ssm_d_weights.push(Some(d));

                let norm = gguf
                    .get_tensor(&format!("blk.{i}.ssm_norm.weight"))
                    .or_else(|_| gguf.get_tensor(&format!("blk.{i}.ssm_norm")))
                    .ok()
                    .map(|t| t.to_f32_vec());
                if let Some(ref w) = norm {
                    let group_size = d_inner / n_group;
                    ensure!(
                        w.len() == d_inner || w.len() == group_size,
                        "blk.{i}.ssm_norm length {} matches neither d_inner ({d_inner}) nor group_size ({group_size})",
                        w.len()
                    );
                }
                ssm_norm_weights.push(norm);
            } else {
                ssm_conv1d_weights.push(None);
                ssm_conv1d_biases.push(None);
                ssm_dt_biases.push(None);
                ssm_a_weights.push(None);
                ssm_d_weights.push(None);
                ssm_norm_weights.push(None);
            }

            // Attention biases
            if has_attn {
                attn_q_bias.push(
                    gguf.get_tensor(&format!("blk.{i}.attn_q.bias"))
                        .ok()
                        .map(|t| t.to_f32_vec()),
                );
                attn_k_bias.push(
                    gguf.get_tensor(&format!("blk.{i}.attn_k.bias"))
                        .ok()
                        .map(|t| t.to_f32_vec()),
                );
                attn_v_bias.push(
                    gguf.get_tensor(&format!("blk.{i}.attn_v.bias"))
                        .ok()
                        .map(|t| t.to_f32_vec()),
                );
                attn_out_bias.push(
                    gguf.get_tensor(&format!("blk.{i}.attn_output.bias"))
                        .ok()
                        .map(|t| t.to_f32_vec()),
                );
            } else {
                attn_q_bias.push(None);
                attn_k_bias.push(None);
                attn_v_bias.push(None);
                attn_out_bias.push(None);
            }

            // FFN biases
            ffn_gate_bias.push(
                gguf.get_tensor(&format!("blk.{i}.ffn_gate.bias"))
                    .ok()
                    .map(|t| t.to_f32_vec()),
            );
            ffn_up_bias.push(
                gguf.get_tensor(&format!("blk.{i}.ffn_up.bias"))
                    .ok()
                    .map(|t| t.to_f32_vec()),
            );
            ffn_down_bias.push(
                gguf.get_tensor(&format!("blk.{i}.ffn_down.bias"))
                    .ok()
                    .map(|t| t.to_f32_vec()),
            );

            // Layer weight refs
            let (attn_q, attn_k, attn_v, attn_output) = if has_attn {
                (
                    Some(transformer::resolve_weight(
                        &gguf,
                        &format!("blk.{i}.attn_q.weight"),
                    )?),
                    Some(transformer::resolve_weight(
                        &gguf,
                        &format!("blk.{i}.attn_k.weight"),
                    )?),
                    Some(transformer::resolve_weight(
                        &gguf,
                        &format!("blk.{i}.attn_v.weight"),
                    )?),
                    Some(transformer::resolve_weight(
                        &gguf,
                        &format!("blk.{i}.attn_output.weight"),
                    )?),
                )
            } else {
                (None, None, None, None)
            };

            let (ssm_in, ssm_out) = if has_ssm {
                (
                    Some(transformer::resolve_weight(
                        &gguf,
                        &format!("blk.{i}.ssm_in.weight"),
                    )?),
                    Some(transformer::resolve_weight(
                        &gguf,
                        &format!("blk.{i}.ssm_out.weight"),
                    )?),
                )
            } else {
                (None, None)
            };

            let ffn_gate = transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_gate.weight"))?;
            let ffn_up = transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_up.weight"))?;
            let ffn_down = transformer::resolve_weight(&gguf, &format!("blk.{i}.ffn_down.weight"))?;

            layer_refs.push(LayerWeightRefs {
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                ssm_in,
                ssm_out,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        let embd_ref = transformer::resolve_weight(&gguf, "token_embd.weight")?;
        let output_ref = transformer::resolve_weight(&gguf, "output.weight").ok();

        Ok(Self {
            gguf,
            config,
            head_dim,
            rope_type,
            rope_freqs,
            output_norm_weight,
            attn_norm_weights,
            ffn_norm_weights,
            ssm_conv1d_weights,
            ssm_conv1d_biases,
            ssm_dt_biases,
            ssm_a_weights,
            ssm_d_weights,
            ssm_norm_weights,
            attn_q_bias,
            attn_k_bias,
            attn_v_bias,
            attn_out_bias,
            ffn_gate_bias,
            ffn_up_bias,
            ffn_down_bias,
            embd_ref,
            output_ref,
            layer_refs,
            model_id,
        })
    }

    fn attn_dims(&self, layer: usize) -> AttnDims<'_> {
        AttnDims {
            hidden_size: self.config.hidden_size,
            n_heads: self.config.n_heads,
            n_kv_heads: self.config.kv_heads_per_layer[layer],
            head_dim: self.head_dim,
            rope_theta: self.config.rope_theta,
            rms_norm_eps: self.config.rms_norm_eps,
            rope_type: self.rope_type,
            attn_scale: self.config.scalars.attn,
            rope_freqs: self.rope_freqs.as_deref(),
        }
    }

    /// Process a single token through one Mamba-2 SSM block.
    fn forward_mamba2_block(
        &self,
        layer: usize,
        normed: &[f32],
        state: &mut InferenceState,
        out_target: &mut [f32],
    ) {
        let ssm = self.config.ssm.as_ref().expect("ssm config required");
        let d_inner = ssm.d_inner;
        let d_state = ssm.d_state;
        let d_conv = ssm.d_conv;
        let n_head = ssm.dt_rank;
        let n_group = ssm.n_group;
        let conv_dim = d_inner + 2 * n_group * d_state;
        let d_in_proj = 2 * d_inner + 2 * n_group * d_state + n_head;

        let refs = &self.layer_refs[layer];
        let ssm_in = refs.ssm_in.as_ref().expect("ssm_in weight required");
        let ssm_out = refs.ssm_out.as_ref().expect("ssm_out weight required");

        // 1. in_proj: normed -> d_in_proj
        let proj = &mut state.scratch.ssm_in_proj[..d_in_proj];
        #[cfg(target_arch = "aarch64")]
        {
            transformer::gemv_preq(
                &self.gguf,
                ssm_in,
                normed,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                proj,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            transformer::gemv(&self.gguf, ssm_in, normed, proj);
        }

        // Split proj into z, x_bc, dt
        let (z, rest) = proj.split_at(d_inner);
        let (x_bc, dt) = rest.split_at(conv_dim);

        // 2. 1D Causal Convolution
        let conv_out = &mut state.scratch.ssm_conv_out[..conv_dim];
        let conv_weight = self.ssm_conv1d_weights[layer]
            .as_ref()
            .expect("ssm_conv1d weight required");
        let conv_bias = self.ssm_conv1d_biases[layer].as_deref();

        // Borrow state.layers directly so state.scratch remains disjointly borrowable
        let (conv_state, ssm_state) = match &mut state.layers[layer] {
            LayerState::Mamba2 {
                conv_state,
                ssm_state,
            }
            | LayerState::ParallelAttentionMamba2 {
                conv_state,
                ssm_state,
                ..
            } => (conv_state.as_mut_slice(), ssm_state.as_mut_slice()),
            _ => panic!("expected Mamba2 state for layer {layer}"),
        };

        cpu::mamba2_conv1d_step(
            x_bc,
            conv_state,
            conv_weight,
            conv_bias,
            conv_dim,
            d_conv,
            conv_out,
        );

        // Split conv_out into x, b, c
        let (x, rest_bc) = conv_out.split_at(d_inner);
        let (b, c) = rest_bc.split_at(n_group * d_state);

        // 3. Recurrent SSD step
        let y = &mut state.scratch.ssm_y[..d_inner];
        let dt_bias = self.ssm_dt_biases[layer]
            .as_ref()
            .expect("ssm_dt bias required");
        let ssm_a = self.ssm_a_weights[layer].as_ref().expect("ssm_a required");
        let ssm_d = self.ssm_d_weights[layer].as_ref().expect("ssm_d required");
        let ssm_norm = self.ssm_norm_weights[layer].as_deref();

        cpu::mamba2_ssd_step(
            z,
            x,
            b,
            c,
            dt,
            dt_bias,
            ssm_a,
            ssm_d,
            ssm_norm,
            self.config.rms_norm_eps,
            ssm_state,
            d_inner,
            d_state,
            n_head,
            n_group,
            y,
        );

        // 4. out_proj: y -> hidden_size
        #[cfg(target_arch = "aarch64")]
        if d_inner.is_multiple_of(32) {
            transformer::quantize_to_scratch_bufs(
                &state.scratch.ssm_y[..d_inner],
                &mut state.scratch.q8_scales,
                &mut state.scratch.q8_quants,
            );
            transformer::gemv_preq(
                &self.gguf,
                ssm_out,
                &state.scratch.ssm_y[..d_inner],
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                out_target,
            );
        } else {
            transformer::gemv(
                &self.gguf,
                ssm_out,
                &state.scratch.ssm_y[..d_inner],
                out_target,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            transformer::gemv(
                &self.gguf,
                ssm_out,
                &state.scratch.ssm_y[..d_inner],
                out_target,
            );
        }
    }

    /// Process a single token through one Attention block.
    fn forward_attn_block(
        &self,
        layer: usize,
        normed: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) {
        let refs = &self.layer_refs[layer];
        let weights = AttnWeights {
            attn_q: refs.attn_q.as_ref().expect("attn_q required"),
            attn_k: refs.attn_k.as_ref().expect("attn_k required"),
            attn_v: refs.attn_v.as_ref().expect("attn_v required"),
            attn_output: refs.attn_output.as_ref().expect("attn_output required"),
        };
        let extras = AttnExtras {
            qkv_bias: match (
                self.attn_q_bias[layer].as_deref(),
                self.attn_k_bias[layer].as_deref(),
                self.attn_v_bias[layer].as_deref(),
            ) {
                (Some(q), Some(k), Some(v)) => Some((q, k, v)),
                _ => None,
            },
            qk_norm: None,
        };
        let dims = self.attn_dims(layer);
        transformer::forward_attn_block(
            &self.gguf, layer, &weights, &extras, dims, normed, pos, state,
        );
    }

    /// Run all layers + final RMSNorm on a single-token hidden state.
    fn run_layers(&self, hidden: &mut [f32], pos: usize, state: &mut InferenceState) {
        let cfg = &self.config;
        let hs = cfg.hidden_size;

        let mut normed = std::mem::take(&mut state.scratch.normed);
        let mut ffn_input = std::mem::take(&mut state.scratch.ffn_input);
        normed.resize(hs, 0.0);
        ffn_input.resize(hs, 0.0);

        for i in 0..cfg.n_layers {
            normed.copy_from_slice(hidden);
            cpu::rmsnorm(&mut normed, &self.attn_norm_weights[i], cfg.rms_norm_eps);

            #[cfg(target_arch = "aarch64")]
            transformer::quantize_to_scratch(&normed, state);

            let bt = cfg.block_types[i];
            match bt {
                BlockType::Attention => {
                    self.forward_attn_block(i, &normed, pos, state);
                    if self.config.scalars.residual != 1.0 {
                        cpu::scale_inplace(
                            &mut state.scratch.out[..hs],
                            self.config.scalars.residual,
                        );
                    }
                    cpu::add_inplace(hidden, &state.scratch.out[..hs]);
                }
                BlockType::Mamba2 => {
                    let mut out = std::mem::take(&mut state.scratch.out);
                    out.resize(hs, 0.0);
                    self.forward_mamba2_block(i, &normed, state, &mut out[..hs]);
                    if self.config.scalars.residual != 1.0 {
                        cpu::scale_inplace(&mut out[..hs], self.config.scalars.residual);
                    }
                    cpu::add_inplace(hidden, &out[..hs]);
                    state.scratch.out = out;
                }
                BlockType::ParallelAttentionMamba2 => {
                    // Attention writes to state.scratch.out[..hs]
                    self.forward_attn_block(i, &normed, pos, state);

                    // Mamba-2 writes to state.scratch.ssm_branch_out[..hs]
                    let mut ssm_out = std::mem::take(&mut state.scratch.ssm_branch_out);
                    ssm_out.resize(hs, 0.0);
                    self.forward_mamba2_block(i, &normed, state, &mut ssm_out[..hs]);

                    // Sum attention and ssm outputs
                    cpu::add_inplace(&mut state.scratch.out[..hs], &ssm_out[..hs]);
                    if self.config.scalars.residual != 1.0 {
                        cpu::scale_inplace(
                            &mut state.scratch.out[..hs],
                            self.config.scalars.residual,
                        );
                    }
                    cpu::add_inplace(hidden, &state.scratch.out[..hs]);
                    state.scratch.ssm_branch_out = ssm_out;
                }
                _ => panic!("unsupported block type {:?} for hybrid model", bt),
            }

            // FFN pre-norm
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
                i,
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

    /// Project the final hidden state to logits over the vocabulary.
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
        if self.config.scalars.logit != 1.0 {
            cpu::scale_inplace(&mut logits, 1.0 / self.config.scalars.logit);
        }
        transformer::oracle_dump::record("result_output", &logits);
        logits
    }
}

impl Model for HybridModel {
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(tokens.len(), 1, "forward() expects exactly 1 token");
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
        false
    }

    fn supports_all_logits(&self) -> bool {
        false
    }

    fn truncate_kv(&self, state: &mut InferenceState, len: usize) {
        state.truncate_to(len);
    }
}
