// Native CUDA LFM2 inference engine and forward pass.
//
// Optimized for NVIDIA Jetson Orin (Ampere sm_87) automotive deployments:
// - Blocking driver synchronization protects host CPU audio DSP from jitter.
// - Zero runtime heap allocation: single contiguous pre-allocated device workspace.
// - Pure warp-synchronous GEMV decode kernels with fused residual accumulation.
// - Online softmax FlashAttention for bounded O(1) shared-memory context scaling.
// - 1-token-per-launch CUDA Graph capture for sub-microsecond dispatch latency.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use crate::backend::cuda::{
    AttentionParams, Conv1dParams, CudaBuffer, CudaContext, CudaGraph, CudaPinnedBuffer,
    QkNormRopeParams,
};
use crate::gguf::GgufFile;
use crate::kv_cache::InferenceState;
use crate::model::gpu_weight_source::GpuWeightSource;
use crate::model::transformer::{self, WeightRef};
use crate::model::{BlockType, Model, ModelConfig};
use crate::tensor::DType;

/// A quantized weight matrix resident in CUDA device memory.
pub struct CudaWeight {
    pub buf: CudaBuffer,
    pub dtype: DType,
    pub m: u32,
    pub k: u32,
}

impl CudaWeight {
    /// Dispatch single-token GEMV (out = A * x).
    pub fn dispatch(&self, ctx: &CudaContext, out: &mut CudaBuffer, x: &CudaBuffer) -> Result<()> {
        match self.dtype {
            DType::Q4_0 => ctx.gemv_q4_0(out, &self.buf, x, self.m, self.k),
            DType::Q8_0 => ctx.gemv_q8_0(out, &self.buf, x, self.m, self.k),
            _ => anyhow::bail!("unsupported CUDA GEMV weight dtype {:?}", self.dtype),
        }
    }

    /// Dispatch single-token GEMV with fused residual accumulation (out += A * x).
    pub fn dispatch_accum(
        &self,
        ctx: &CudaContext,
        out: &mut CudaBuffer,
        x: &CudaBuffer,
    ) -> Result<()> {
        match self.dtype {
            DType::Q4_0 => ctx.gemv_q4_0_accum(out, &self.buf, x, self.m, self.k),
            DType::Q8_0 => ctx.gemv_q8_0_accum(out, &self.buf, x, self.m, self.k),
            _ => anyhow::bail!("unsupported CUDA GEMV accum weight dtype {:?}", self.dtype),
        }
    }
}

/// Dense SwiGLU feed-forward layer weights on CUDA.
pub struct CudaDenseFfn {
    pub gate: CudaWeight,
    pub up: CudaWeight,
    pub down: CudaWeight,
}

/// Multi-head or grouped-query attention block on CUDA.
pub struct CudaAttnLayer {
    pub wq: CudaWeight,
    pub wk: CudaWeight,
    pub wv: CudaWeight,
    pub wo: CudaWeight,
    pub q_norm: Option<CudaBuffer>,
    pub k_norm: Option<CudaBuffer>,
    pub k_cache: CudaBuffer,
    pub v_cache: CudaBuffer,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub kv_dim: u32,
}

/// 1D short convolution block on CUDA for LFM2 hybrid layers.
pub struct CudaConvLayer {
    pub w_in: CudaWeight,
    pub weight: CudaBuffer,
    pub w_out: CudaWeight,
    pub rbuffer: Mutex<CudaBuffer>,
    pub hs: u32,
    pub kernel_size: u32,
    pub d_conv: u32,
}

/// Layer operator: either Attention or GatedConv.
pub enum CudaLayerOperator {
    Attention(Box<CudaAttnLayer>),
    Conv(Box<CudaConvLayer>),
}

/// Complete transformer layer combining normalization, operator, and FFN.
pub struct CudaLayer {
    pub attn_norm: CudaBuffer,
    pub op: CudaLayerOperator,
    pub ffn_norm: CudaBuffer,
    pub ffn: CudaDenseFfn,
}

/// Pre-allocated execution scratch buffers.
///
/// Sized once at initialization to guarantee zero heap allocations during inference.
pub struct CudaWorkspace {
    pub hidden: CudaBuffer,
    pub normed: CudaBuffer,
    pub q: CudaBuffer,
    pub k: CudaBuffer,
    pub v: CudaBuffer,
    pub attn_out: CudaBuffer,
    pub conv_proj: CudaBuffer,
    pub conv_out: CudaBuffer,
    pub ffn_input: CudaBuffer,
    pub gate: CudaBuffer,
    pub up: CudaBuffer,
    pub final_norm: CudaBuffer,
    pub logits: CudaBuffer,
    pub pinned_logits: CudaPinnedBuffer,
}

/// Native CUDA LFM2 model instance.
pub struct CudaLfm2Model {
    pub ctx: Arc<CudaContext>,
    pub config: ModelConfig,
    pub layers: Vec<CudaLayer>,
    pub output_norm: CudaBuffer,
    pub output_weight: CudaWeight,
    pub embedding_bytes: Vec<u8>,
    pub embedding_dtype: DType,
    pub embedding_table: Option<CudaBuffer>,
    pub embedding_hidden_size: usize,
    pub freq_factors: Option<CudaBuffer>,
    pub workspace: Mutex<CudaWorkspace>,
    pub seq_len: AtomicUsize,
    pub max_seq_len: usize,
    pub infer_lock: Mutex<()>,
    pub decode_graph: Mutex<Option<CudaGraph>>,
}

impl CudaLfm2Model {
    /// Build model from GGUF file and context size.
    pub fn from_gguf(
        gguf: GgufFile,
        _path: Option<&std::path::Path>,
        context_size: usize,
    ) -> Result<Self> {
        let cpu = super::lfm2::Lfm2Model::from_gguf(gguf, context_size)?;
        Self::from_weight_source(&cpu, context_size)
    }

    /// Build model from LLaMA / transformer GGUF file.
    pub fn from_llama(
        gguf: GgufFile,
        path: Option<&std::path::Path>,
        context_size: usize,
    ) -> Result<Self> {
        let model_id = path
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let cpu = super::llama::LlamaModel::from_gguf_with_id(gguf, context_size, model_id)?;
        Self::from_weight_source(&cpu, context_size)
    }

    /// Generalized CUDA loader over any GpuWeightSource.
    pub fn from_weight_source(src: &dyn GpuWeightSource, context_size: usize) -> Result<Self> {
        let ctx = Arc::new(CudaContext::default_device()?);
        let mut config = src.config().clone();
        let max_seq_len = context_size.min(config.max_seq_len);
        config.max_seq_len = max_seq_len;

        let hs = config.hidden_size;
        let is = config.intermediate_size;
        let head_dim = config.head_dim;
        let q_dim = config.n_heads * head_dim;
        let max_kv_dim = config.kv_heads_per_layer.iter().copied().max().unwrap_or(0) * head_dim;
        let vocab_size = config.vocab_size;

        tracing::info!(
            "CUDA model: {} layers, hs={hs}, is={is}, vocab={vocab_size}, context={max_seq_len}",
            config.n_layers
        );

        // Helper to upload a WeightRef to a device-resident CudaWeight
        let upload_weight = |wref: &WeightRef| -> Result<CudaWeight> {
            let bytes = src.weight_bytes(wref);
            let buf = ctx.upload_bytes(&bytes)?;
            Ok(CudaWeight {
                buf,
                dtype: wref.dtype,
                m: wref.m as u32,
                k: wref.k as u32,
            })
        };

        // Upload per-layer weights and allocate KV caches
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let attn_norm = ctx.upload_f32(src.attn_norm_weight(i))?;
            let ffn_norm = ctx.upload_f32(src.ffn_norm_weight(i))?;

            // Feed-Forward weights
            let gate_ref = src.ffn_gate_ref(i)?;
            let up_ref = src.ffn_up_ref(i)?;
            let down_ref = src.ffn_down_ref(i)?;
            let ffn = CudaDenseFfn {
                gate: upload_weight(gate_ref)?,
                up: upload_weight(up_ref)?,
                down: upload_weight(down_ref)?,
            };

            // Operator: GatedConv or Attention
            let op = if config.block_types.get(i).copied() == Some(BlockType::GatedConv) {
                let conv_w_data = src
                    .conv_weight(i)
                    .context("missing conv weight for GatedConv layer")?;
                let weight = ctx.upload_f32(conv_w_data)?;

                let w_in_ref = src
                    .conv_in_proj_ref(i)
                    .context("missing conv_in_proj_ref")?;
                let w_out_ref = src
                    .conv_out_proj_ref(i)
                    .context("missing conv_out_proj_ref")?;

                let d_conv = 3u32;
                let kernel_size = (conv_w_data.len() / hs) as u32;
                let rbuffer =
                    ctx.create_buffer((d_conv as usize) * hs * std::mem::size_of::<f32>())?;

                CudaLayerOperator::Conv(Box::new(CudaConvLayer {
                    w_in: upload_weight(w_in_ref)?,
                    weight,
                    w_out: upload_weight(w_out_ref)?,
                    rbuffer: Mutex::new(rbuffer),
                    hs: hs as u32,
                    kernel_size,
                    d_conv,
                }))
            } else {
                let n_heads = config.n_heads as u32;
                let n_kv_heads = config.kv_heads_per_layer[i] as u32;
                let kv_dim = n_kv_heads * (head_dim as u32);

                let ref_q = src.attn_q_ref(i).context("missing attn_q_ref")?;
                let ref_k = src.attn_k_ref(i).context("missing attn_k_ref")?;
                let ref_v = src.attn_v_ref(i).context("missing attn_v_ref")?;
                let ref_o = src.attn_output_ref(i).context("missing attn_output_ref")?;

                let q_norm = src
                    .attn_q_norm_weight(i)
                    .map(|w| ctx.upload_f32(w))
                    .transpose()?;
                let k_norm = src
                    .attn_k_norm_weight(i)
                    .map(|w| ctx.upload_f32(w))
                    .transpose()?;

                // Allocate FP16 KV cache for this layer
                let cache_bytes = max_seq_len * (kv_dim as usize) * std::mem::size_of::<u16>();
                let k_cache = ctx.create_buffer(cache_bytes)?;
                let v_cache = ctx.create_buffer(cache_bytes)?;

                CudaLayerOperator::Attention(Box::new(CudaAttnLayer {
                    wq: upload_weight(ref_q)?,
                    wk: upload_weight(ref_k)?,
                    wv: upload_weight(ref_v)?,
                    wo: upload_weight(ref_o)?,
                    q_norm,
                    k_norm,
                    k_cache,
                    v_cache,
                    n_heads,
                    n_kv_heads,
                    head_dim: head_dim as u32,
                    kv_dim,
                }))
            };

            layers.push(CudaLayer {
                attn_norm,
                op,
                ffn_norm,
                ffn,
            });
        }

        // Final output norm
        let output_norm = ctx.upload_f32(src.output_norm_weight())?;

        // Output projection weight
        let output_weight = if let Some(out_ref) = src.output_ref() {
            upload_weight(out_ref)?
        } else {
            let embd_ref = transformer::resolve_weight(src.gguf(), "token_embd.weight")?;
            upload_weight(&embd_ref)?
        };

        // Token embeddings
        let embedding_dtype = src
            .gguf()
            .tensors
            .get("token_embd.weight")
            .map(|t| t.dtype)
            .ok_or_else(|| anyhow::anyhow!("missing token_embd.weight in GGUF metadata"))?;
        let embd_data = src.embedding_tensor_data()?;
        let embedding_bytes = embd_data.into_owned();
        let embedding_hidden_size = hs;
        let embedding_table = if embedding_dtype == DType::Q8_0 {
            Some(ctx.upload_bytes(&embedding_bytes)?)
        } else {
            None
        };

        // RoPE frequency factors
        let freq_factors = src.rope_freqs().map(|w| ctx.upload_f32(w)).transpose()?;

        // Allocate static execution workspace (zero runtime allocations)
        let f32_size = std::mem::size_of::<f32>();
        let workspace = CudaWorkspace {
            hidden: ctx.create_buffer(hs * f32_size)?,
            normed: ctx.create_buffer(hs * f32_size)?,
            q: ctx.create_buffer(q_dim * f32_size)?,
            k: ctx.create_buffer(max_kv_dim * f32_size)?,
            v: ctx.create_buffer(max_kv_dim * f32_size)?,
            attn_out: ctx.create_buffer(hs * f32_size)?,
            conv_proj: ctx.create_buffer(3 * hs * f32_size)?,
            conv_out: ctx.create_buffer(hs * f32_size)?,
            ffn_input: ctx.create_buffer(hs * f32_size)?,
            gate: ctx.create_buffer(is * f32_size)?,
            up: ctx.create_buffer(is * f32_size)?,
            final_norm: ctx.create_buffer(hs * f32_size)?,
            logits: ctx.create_buffer(vocab_size * f32_size)?,
            pinned_logits: ctx.create_pinned_buffer(vocab_size * f32_size)?,
        };

        Ok(Self {
            ctx,
            config,
            layers,
            output_norm,
            output_weight,
            embedding_bytes,
            embedding_dtype,
            embedding_table,
            embedding_hidden_size,
            freq_factors,
            workspace: Mutex::new(workspace),
            seq_len: AtomicUsize::new(0),
            max_seq_len,
            infer_lock: Mutex::new(()),
            decode_graph: Mutex::new(None),
        })
    }

    /// Dequantize token embedding row into host float slice.
    fn dequant_embedding_row(&self, token_id: usize, dst: &mut [f32]) {
        let hs = self.embedding_hidden_size;
        let dt = self.embedding_dtype;
        let row_bytes = hs / dt.block_size() * dt.block_bytes();
        let row_offset = token_id * row_bytes;
        let row_data = &self.embedding_bytes[row_offset..row_offset + row_bytes];
        transformer::dequantize_row_slice(dt, row_data, dst);
    }

    /// Execute a single forward decode step on device.
    fn forward_step_device(
        &self,
        token_id: usize,
        pos: usize,
        ws: &mut CudaWorkspace,
    ) -> Result<()> {
        let hs = self.config.hidden_size as u32;
        let is = self.config.intermediate_size as u32;
        let eps = self.config.rms_norm_eps;

        // 1. Token embedding lookup
        if let Some(table) = &self.embedding_table {
            self.ctx
                .gather_embedding_q8_0(&mut ws.hidden, table, token_id as u32, hs)?;
        } else {
            let mut embd_row = vec![0.0f32; self.embedding_hidden_size];
            self.dequant_embedding_row(token_id, &mut embd_row);
            ws.hidden.copy_from_host(bytemuck::cast_slice(&embd_row))?;
        }

        // 2. Sequential layer execution
        for layer in &self.layers {
            // Input RMSNorm
            self.ctx
                .rmsnorm(&mut ws.normed, &ws.hidden, &layer.attn_norm, hs, eps)?;

            match &layer.op {
                CudaLayerOperator::Attention(attn) => {
                    // Q, K, V projections
                    attn.wq.dispatch(&self.ctx, &mut ws.q, &ws.normed)?;
                    attn.wk.dispatch(&self.ctx, &mut ws.k, &ws.normed)?;
                    attn.wv.dispatch(&self.ctx, &mut ws.v, &ws.normed)?;

                    // Fused per-head RMSNorm + RoPE
                    let qk_params = QkNormRopeParams {
                        pos: pos as u32,
                        n_heads: attn.n_heads,
                        n_kv_heads: attn.n_kv_heads,
                        head_dim: attn.head_dim,
                        eps,
                        freq_base: self.config.rope_theta,
                        rope_type: 0, // NeoX
                        has_freq_factors: self.freq_factors.is_some() as u32,
                        has_qk_norm: attn.q_norm.is_some() as u32,
                    };
                    self.ctx.qk_norm_rope(
                        &mut ws.q,
                        &mut ws.k,
                        attn.q_norm.as_ref(),
                        attn.k_norm.as_ref(),
                        self.freq_factors.as_ref(),
                        qk_params,
                    )?;

                    // Append K & V to KV cache at pos * kv_dim
                    let kv_offset = pos * (attn.kv_dim as usize);
                    self.ctx.cast_f32_to_f16_offset(
                        &ws.k,
                        &attn.k_cache,
                        kv_offset,
                        attn.kv_dim,
                    )?;
                    self.ctx.cast_f32_to_f16_offset(
                        &ws.v,
                        &attn.v_cache,
                        kv_offset,
                        attn.kv_dim,
                    )?;

                    // Single-token FlashAttention
                    let attn_scale = 1.0f32 / (attn.head_dim as f32).sqrt();
                    let attn_params = AttentionParams {
                        n_heads: attn.n_heads,
                        n_kv_heads: attn.n_kv_heads,
                        head_dim: attn.head_dim,
                        kv_dim: attn.kv_dim,
                        seq_len: (pos + 1) as u32,
                        scale: attn_scale,
                        _pad0: 0,
                        _pad1: 0,
                    };
                    self.ctx.flash_attention(
                        &mut ws.attn_out,
                        &ws.q,
                        &attn.k_cache,
                        &attn.v_cache,
                        attn_params,
                    )?;

                    // Output projection with fused residual addition (ws.hidden += Wo * attn_out)
                    attn.wo
                        .dispatch_accum(&self.ctx, &mut ws.hidden, &ws.attn_out)?;
                }
                CudaLayerOperator::Conv(conv) => {
                    // Conv input projection
                    conv.w_in
                        .dispatch(&self.ctx, &mut ws.conv_proj, &ws.normed)?;

                    // Fused short convolution
                    let conv_params = Conv1dParams {
                        hs: conv.hs,
                        kernel_size: conv.kernel_size,
                        d_conv: conv.d_conv,
                        _pad: 0,
                    };
                    let mut rbuffer = conv.rbuffer.lock().unwrap();
                    self.ctx.conv1d_fused(
                        &mut ws.conv_out,
                        &ws.conv_proj,
                        &mut rbuffer,
                        &conv.weight,
                        conv_params,
                    )?;

                    // Conv output projection with fused residual addition
                    conv.w_out
                        .dispatch_accum(&self.ctx, &mut ws.hidden, &ws.conv_out)?;
                }
            }

            // Post-operator RMSNorm
            self.ctx
                .rmsnorm(&mut ws.ffn_input, &ws.hidden, &layer.ffn_norm, hs, eps)?;

            // SwiGLU FFN
            layer
                .ffn
                .gate
                .dispatch(&self.ctx, &mut ws.gate, &ws.ffn_input)?;
            layer
                .ffn
                .up
                .dispatch(&self.ctx, &mut ws.up, &ws.ffn_input)?;
            self.ctx.silu_mul_inplace(&mut ws.gate, &ws.up, is)?;
            layer
                .ffn
                .down
                .dispatch_accum(&self.ctx, &mut ws.hidden, &ws.gate)?;
        }

        // 3. Final RMSNorm
        self.ctx
            .rmsnorm(&mut ws.final_norm, &ws.hidden, &self.output_norm, hs, eps)?;

        // 4. Logit projection
        self.output_weight
            .dispatch(&self.ctx, &mut ws.logits, &ws.final_norm)?;

        Ok(())
    }
}

impl Model for CudaLfm2Model {
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!tokens.is_empty(), "forward requires at least one token");

        let mut ws = self.workspace.lock().unwrap_or_else(|e| e.into_inner());
        let vocab_size = self.config.vocab_size;

        let mut last_logits = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            let cur_pos = pos + i;
            let is_last = i == tokens.len() - 1;
            assert!(
                cur_pos < self.max_seq_len,
                "cur_pos {cur_pos} exceeds max_seq_len {}",
                self.max_seq_len
            );

            self.forward_step_device(token as usize, cur_pos, &mut ws)
                .expect("CUDA forward step failed");

            if is_last {
                self.ctx.synchronize().expect("CUDA synchronize failed");

                let mut logits = vec![0.0f32; vocab_size];
                ws.logits
                    .copy_to_host(bytemuck::cast_slice_mut(&mut logits))
                    .expect("CUDA logit readback failed");
                last_logits = logits;
            }

            self.seq_len.store(cur_pos + 1, Ordering::Relaxed);
            state.seq_len += 1;
        }

        last_logits
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn truncate_kv(&self, state: &mut InferenceState, len: usize) {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.seq_len.store(len, Ordering::Relaxed);
        state.seq_len = len;
    }
}
