// Native CUDA LFM2 inference engine and forward pass.
//
// Optimized for NVIDIA Jetson Orin (Ampere sm_87) automotive deployments:
// - Blocking driver synchronization protects host CPU audio DSP from jitter.
// - Zero runtime heap allocation: single contiguous pre-allocated device workspace.
// - Pure warp-synchronous GEMV decode kernels with fused residual accumulation.
// - Online softmax FlashAttention for bounded O(1) shared-memory context scaling.
// - 1-token-per-launch CUDA Graph capture for sub-microsecond dispatch latency.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};

use crate::backend::cuda::{
    AttentionParams, Conv1dParams, CudaBuffer, CudaContext, CudaPinnedBuffer, QkNormRopeParams,
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
            DType::Q4KM => ctx.gemv_q4k(out, &self.buf, x, self.m, self.k),
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
            DType::Q4KM => ctx.gemv_q4k_accum(out, &self.buf, x, self.m, self.k),
            _ => anyhow::bail!("unsupported CUDA GEMV accum weight dtype {:?}", self.dtype),
        }
    }

    /// Dispatch batched GEMM (out = X * A^T).
    pub fn dispatch_gemm(
        &self,
        ctx: &CudaContext,
        out: &mut CudaBuffer,
        x: &CudaBuffer,
        batch_size: u32,
    ) -> Result<()> {
        match self.dtype {
            DType::Q4_0 => ctx.gemm_q4_0(out, &self.buf, x, batch_size, self.m, self.k),
            DType::Q8_0 => ctx.gemm_q8_0(out, &self.buf, x, batch_size, self.m, self.k),
            DType::Q4KM => ctx.gemm_q4k(out, &self.buf, x, batch_size, self.m, self.k),
            _ => anyhow::bail!("unsupported CUDA GEMM weight dtype {:?}", self.dtype),
        }
    }

    /// Dispatch batched GEMM with fused residual accumulation (out += X * A^T).
    pub fn dispatch_gemm_accum(
        &self,
        ctx: &CudaContext,
        out: &mut CudaBuffer,
        x: &CudaBuffer,
        batch_size: u32,
    ) -> Result<()> {
        match self.dtype {
            DType::Q4_0 => ctx.gemm_q4_0_accum(out, &self.buf, x, batch_size, self.m, self.k),
            DType::Q8_0 => ctx.gemm_q8_0_accum(out, &self.buf, x, batch_size, self.m, self.k),
            DType::Q4KM => ctx.gemm_q4k_accum(out, &self.buf, x, batch_size, self.m, self.k),
            _ => anyhow::bail!("unsupported CUDA GEMM accum weight dtype {:?}", self.dtype),
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
    pub argmax_token: CudaBuffer,
    pub pinned_token: CudaPinnedBuffer,
    pub embd_scratch: Vec<f32>,
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
    pub rope_inv_freq: CudaBuffer,
    pub rope_type: u32,
    pub workspace: Mutex<CudaWorkspace>,
    pub seq_len: AtomicUsize,
    pub max_seq_len: usize,
    pub infer_lock: Mutex<()>,
    pub gpu_mem_bytes: u64,
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
        let rope_type = src.rope_type() as u32;
        let max_seq_len = context_size.min(config.max_seq_len);
        config.max_seq_len = max_seq_len;

        let hs = config.hidden_size;
        let is = config.intermediate_size;
        let head_dim = config.head_dim;
        ensure!(
            head_dim <= 128,
            "CUDA FlashAttention kernel supports head_dim <= 128, got {head_dim}"
        );
        ensure!(
            hs > 0 && hs.is_multiple_of(4),
            "hidden_size must be positive and a multiple of 4, got {hs}"
        );
        ensure!(
            head_dim > 0 && head_dim.is_multiple_of(4),
            "head_dim must be positive and a multiple of 4, got {head_dim}"
        );
        ensure!(
            config.n_heads > 0,
            "n_heads must be positive, got {}",
            config.n_heads
        );
        for (i, &n_kv) in config.kv_heads_per_layer.iter().enumerate() {
            ensure!(
                n_kv > 0 && config.n_heads.is_multiple_of(n_kv),
                "layer {i}: n_heads ({}) must be multiple of n_kv ({n_kv})",
                config.n_heads
            );
        }
        let q_dim = config.n_heads * head_dim;
        let max_kv_dim = config.kv_heads_per_layer.iter().copied().max().unwrap_or(0) * head_dim;
        let vocab_size = config.vocab_size;

        tracing::info!(
            "CUDA model: {} layers, hs={hs}, is={is}, vocab={vocab_size}, context={max_seq_len}",
            config.n_layers
        );

        let gpu_mem_bytes = Cell::new(0u64);

        // Helper to upload a WeightRef to a device-resident CudaWeight
        let upload_weight = |wref: &WeightRef| -> Result<CudaWeight> {
            let bytes = src.weight_bytes(wref);
            let buf = ctx.upload_bytes(&bytes)?;
            gpu_mem_bytes.set(gpu_mem_bytes.get() + buf.len() as u64);
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
            gpu_mem_bytes.set(gpu_mem_bytes.get() + attn_norm.len() as u64);
            let ffn_norm = ctx.upload_f32(src.ffn_norm_weight(i))?;
            gpu_mem_bytes.set(gpu_mem_bytes.get() + ffn_norm.len() as u64);

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
                gpu_mem_bytes.set(gpu_mem_bytes.get() + weight.len() as u64);

                let w_in_ref = src
                    .conv_in_proj_ref(i)
                    .context("missing conv_in_proj_ref")?;
                let w_out_ref = src
                    .conv_out_proj_ref(i)
                    .context("missing conv_out_proj_ref")?;

                let kernel_size = (conv_w_data.len() / hs) as u32;
                let d_conv = kernel_size.saturating_sub(1);
                let rbuffer =
                    ctx.create_buffer((d_conv as usize) * hs * std::mem::size_of::<f32>())?;
                gpu_mem_bytes.set(gpu_mem_bytes.get() + rbuffer.len() as u64);

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
                if let Some(ref qn) = q_norm {
                    gpu_mem_bytes.set(gpu_mem_bytes.get() + qn.len() as u64);
                }
                let k_norm = src
                    .attn_k_norm_weight(i)
                    .map(|w| ctx.upload_f32(w))
                    .transpose()?;
                if let Some(ref kn) = k_norm {
                    gpu_mem_bytes.set(gpu_mem_bytes.get() + kn.len() as u64);
                }

                // Allocate FP16 KV cache for this layer
                let cache_bytes = max_seq_len * (kv_dim as usize) * std::mem::size_of::<u16>();
                let k_cache = ctx.create_buffer(cache_bytes)?;
                let v_cache = ctx.create_buffer(cache_bytes)?;
                gpu_mem_bytes.set(gpu_mem_bytes.get() + (k_cache.len() + v_cache.len()) as u64);

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
        gpu_mem_bytes.set(gpu_mem_bytes.get() + output_norm.len() as u64);

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
        let embedding_table = if embedding_dtype == DType::Q8_0
            || embedding_dtype == DType::Q4_0
            || embedding_dtype == DType::Q4KM
        {
            Some(ctx.upload_bytes(&embedding_bytes)?)
        } else {
            None
        };
        if let Some(ref table) = embedding_table {
            gpu_mem_bytes.set(gpu_mem_bytes.get() + table.len() as u64);
        }

        // Precompute RoPE inverse frequencies once on CPU, incorporating optional LLaMA-3 freq factors
        let half_dim = (head_dim / 2).min(64);
        let theta_scale = config.rope_theta.powf(-2.0 / head_dim as f32);
        let mut inv_freqs = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let mut f = theta_scale.powf(i as f32);
            if let Some(factor) = src.rope_freqs().and_then(|facs| facs.get(i).copied()) {
                f /= factor;
            }
            inv_freqs.push(f);
        }
        let rope_inv_freq = ctx.upload_f32(&inv_freqs)?;
        gpu_mem_bytes.set(gpu_mem_bytes.get() + rope_inv_freq.len() as u64);

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
            argmax_token: ctx.create_buffer(std::mem::size_of::<u32>())?,
            pinned_token: ctx.create_pinned_buffer(std::mem::size_of::<u32>())?,
            embd_scratch: vec![0.0f32; hs],
        };
        gpu_mem_bytes.set(
            gpu_mem_bytes.get()
                + (workspace.hidden.len()
                    + workspace.normed.len()
                    + workspace.q.len()
                    + workspace.k.len()
                    + workspace.v.len()
                    + workspace.attn_out.len()
                    + workspace.conv_proj.len()
                    + workspace.conv_out.len()
                    + workspace.ffn_input.len()
                    + workspace.gate.len()
                    + workspace.up.len()
                    + workspace.final_norm.len()
                    + workspace.logits.len()
                    + workspace.pinned_logits.len()
                    + workspace.argmax_token.len()
                    + workspace.pinned_token.len()) as u64,
        );

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
            rope_inv_freq,
            rope_type,
            workspace: Mutex::new(workspace),
            seq_len: AtomicUsize::new(0),
            max_seq_len,
            infer_lock: Mutex::new(()),
            gpu_mem_bytes: gpu_mem_bytes.into_inner(),
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
        compute_logits: bool,
    ) -> Result<()> {
        ensure!(
            token_id < self.config.vocab_size,
            "token_id {token_id} out of range (vocab_size={})",
            self.config.vocab_size
        );

        let hs = self.config.hidden_size as u32;
        let is = self.config.intermediate_size as u32;
        let eps = self.config.rms_norm_eps;

        // 1. Token embedding lookup
        if let Some(table) = &self.embedding_table {
            match self.embedding_dtype {
                DType::Q8_0 => {
                    self.ctx
                        .gather_embedding_q8_0(&mut ws.hidden, table, token_id as u32, hs)?;
                }
                DType::Q4_0 => {
                    self.ctx
                        .gather_embedding_q4_0(&mut ws.hidden, table, token_id as u32, hs)?;
                }
                DType::Q4KM => {
                    self.ctx
                        .gather_embedding_q4k(&mut ws.hidden, table, token_id as u32, hs)?;
                }
                _ => {
                    self.dequant_embedding_row(token_id, &mut ws.embd_scratch);
                    ws.hidden
                        .copy_from_host(bytemuck::cast_slice(&ws.embd_scratch))?;
                }
            }
        } else {
            self.dequant_embedding_row(token_id, &mut ws.embd_scratch);
            ws.hidden
                .copy_from_host(bytemuck::cast_slice(&ws.embd_scratch))?;
        }

        // 2. Sequential layer execution
        for layer in &self.layers {
            // Input RMSNorm
            self.ctx
                .rmsnorm(&mut ws.normed, &ws.hidden, &layer.attn_norm, hs, eps)?;

            match &layer.op {
                CudaLayerOperator::Attention(attn) => {
                    // Q, K, V projections (fused when dtypes match)
                    if attn.wq.dtype == attn.wk.dtype && attn.wq.dtype == attn.wv.dtype {
                        match attn.wq.dtype {
                            DType::Q4_0 => {
                                self.ctx.gemv_q4_0_concat3(
                                    &mut ws.q,
                                    &mut ws.k,
                                    &mut ws.v,
                                    &attn.wq.buf,
                                    &attn.wk.buf,
                                    &attn.wv.buf,
                                    &ws.normed,
                                    attn.wq.m,
                                    attn.wk.m,
                                    attn.wv.m,
                                    attn.wq.k,
                                )?;
                            }
                            DType::Q8_0 => {
                                self.ctx.gemv_q8_0_concat3(
                                    &mut ws.q,
                                    &mut ws.k,
                                    &mut ws.v,
                                    &attn.wq.buf,
                                    &attn.wk.buf,
                                    &attn.wv.buf,
                                    &ws.normed,
                                    attn.wq.m,
                                    attn.wk.m,
                                    attn.wv.m,
                                    attn.wq.k,
                                )?;
                            }
                            DType::Q4KM => {
                                self.ctx.gemv_q4k_concat3(
                                    &mut ws.q,
                                    &mut ws.k,
                                    &mut ws.v,
                                    &attn.wq.buf,
                                    &attn.wk.buf,
                                    &attn.wv.buf,
                                    &ws.normed,
                                    attn.wq.m,
                                    attn.wk.m,
                                    attn.wv.m,
                                    attn.wq.k,
                                )?;
                            }
                            _ => {
                                attn.wq.dispatch(&self.ctx, &mut ws.q, &ws.normed)?;
                                attn.wk.dispatch(&self.ctx, &mut ws.k, &ws.normed)?;
                                attn.wv.dispatch(&self.ctx, &mut ws.v, &ws.normed)?;
                            }
                        }
                    } else {
                        attn.wq.dispatch(&self.ctx, &mut ws.q, &ws.normed)?;
                        attn.wk.dispatch(&self.ctx, &mut ws.k, &ws.normed)?;
                        attn.wv.dispatch(&self.ctx, &mut ws.v, &ws.normed)?;
                    }

                    // Fused per-head RMSNorm + RoPE
                    let qk_params = QkNormRopeParams {
                        pos: pos as u32,
                        n_heads: attn.n_heads,
                        n_kv_heads: attn.n_kv_heads,
                        head_dim: attn.head_dim,
                        eps,
                        freq_base: self.config.rope_theta,
                        rope_type: self.rope_type,
                        has_freq_factors: 0,
                        has_qk_norm: attn.q_norm.is_some() as u32,
                    };
                    self.ctx.qk_norm_rope(
                        &mut ws.q,
                        &mut ws.k,
                        attn.q_norm.as_ref(),
                        attn.k_norm.as_ref(),
                        Some(&self.rope_inv_freq),
                        qk_params,
                    )?;

                    // Append K & V to KV cache at pos * kv_dim in a single fused dispatch
                    let kv_offset = pos * (attn.kv_dim as usize);
                    self.ctx.append_kv_cache_f16(
                        &ws.k,
                        &ws.v,
                        &attn.k_cache,
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
                    let mut rbuffer = conv.rbuffer.lock().unwrap_or_else(|e| e.into_inner());
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

            // SwiGLU FFN (fused gate and up projections with in-register silu_mul when dtypes match)
            if layer.ffn.gate.dtype == layer.ffn.up.dtype {
                match layer.ffn.gate.dtype {
                    DType::Q4_0 => {
                        self.ctx.gemv_q4_0_swiglu(
                            &mut ws.gate,
                            &layer.ffn.gate.buf,
                            &layer.ffn.up.buf,
                            &ws.ffn_input,
                            layer.ffn.gate.m,
                            layer.ffn.gate.k,
                        )?;
                    }
                    DType::Q8_0 => {
                        self.ctx.gemv_q8_0_swiglu(
                            &mut ws.gate,
                            &layer.ffn.gate.buf,
                            &layer.ffn.up.buf,
                            &ws.ffn_input,
                            layer.ffn.gate.m,
                            layer.ffn.gate.k,
                        )?;
                    }
                    DType::Q4KM => {
                        self.ctx.gemv_q4k_swiglu(
                            &mut ws.gate,
                            &layer.ffn.gate.buf,
                            &layer.ffn.up.buf,
                            &ws.ffn_input,
                            layer.ffn.gate.m,
                            layer.ffn.gate.k,
                        )?;
                    }
                    _ => {
                        layer
                            .ffn
                            .gate
                            .dispatch(&self.ctx, &mut ws.gate, &ws.ffn_input)?;
                        layer
                            .ffn
                            .up
                            .dispatch(&self.ctx, &mut ws.up, &ws.ffn_input)?;
                        self.ctx.silu_mul_inplace(&mut ws.gate, &ws.up, is)?;
                    }
                }
            } else {
                layer
                    .ffn
                    .gate
                    .dispatch(&self.ctx, &mut ws.gate, &ws.ffn_input)?;
                layer
                    .ffn
                    .up
                    .dispatch(&self.ctx, &mut ws.up, &ws.ffn_input)?;
                self.ctx.silu_mul_inplace(&mut ws.gate, &ws.up, is)?;
            }
            layer
                .ffn
                .down
                .dispatch_accum(&self.ctx, &mut ws.hidden, &ws.gate)?;
        }

        // 3. Final RMSNorm and logit projection (elided for intermediate prefill tokens)
        if compute_logits {
            self.ctx
                .rmsnorm(&mut ws.final_norm, &ws.hidden, &self.output_norm, hs, eps)?;
            self.output_weight
                .dispatch(&self.ctx, &mut ws.logits, &ws.final_norm)?;
        }

        Ok(())
    }

    /// Zero out all recurrent convolution rolling buffers in device memory.
    pub fn zero_conv_buffers(&self) -> Result<()> {
        for layer in &self.layers {
            if let CudaLayerOperator::Conv(conv) = &layer.op {
                let mut rbuffer = conv.rbuffer.lock().unwrap_or_else(|e| e.into_inner());
                rbuffer.zero()?;
            }
        }
        Ok(())
    }
}

impl Model for CudaLfm2Model {
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        if tokens.is_empty() {
            return vec![0.0f32; self.config.vocab_size];
        }

        if pos == 0 {
            self.seq_len.store(0, Ordering::Relaxed);
            let _ = self.zero_conv_buffers();
        }

        let mut ws = self.workspace.lock().unwrap_or_else(|e| e.into_inner());
        let vocab_size = self.config.vocab_size;

        let mut last_logits = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            let cur_pos = pos + i;
            let is_last = i == tokens.len() - 1;
            if cur_pos >= self.max_seq_len {
                tracing::error!("cur_pos {cur_pos} exceeds max_seq_len {}", self.max_seq_len);
                return vec![0.0f32; vocab_size];
            }

            if let Err(e) = self.forward_step_device(token as usize, cur_pos, &mut ws, is_last) {
                tracing::error!("CUDA forward step failed at pos {cur_pos}: {e:?}");
                return vec![0.0f32; vocab_size];
            }

            if is_last {
                let ws_ref = &mut *ws;
                let pinned_slice = ws_ref.pinned_logits.as_mut_slice();
                if let Err(e) = ws_ref.logits.copy_to_host(pinned_slice) {
                    tracing::error!("CUDA logit readback failed: {e:?}");
                    return vec![0.0f32; vocab_size];
                }
                let pinned_f32: &[f32] = bytemuck::cast_slice(ws_ref.pinned_logits.as_slice());
                last_logits = pinned_f32[..vocab_size].to_vec();
            }

            self.seq_len.store(cur_pos + 1, Ordering::Relaxed);
            state.seq_len += 1;
        }

        last_logits
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        self.forward(tokens, start_pos, state)
    }

    fn forward_greedy(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> u32 {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        if tokens.is_empty() {
            return 0;
        }

        if pos == 0 {
            self.seq_len.store(0, Ordering::Relaxed);
            let _ = self.zero_conv_buffers();
        }

        let mut ws = self.workspace.lock().unwrap_or_else(|e| e.into_inner());
        let vocab_size = self.config.vocab_size as u32;

        for (i, &token) in tokens.iter().enumerate() {
            let cur_pos = pos + i;
            let is_last = i == tokens.len() - 1;
            if cur_pos >= self.max_seq_len {
                tracing::error!("cur_pos {cur_pos} exceeds max_seq_len {}", self.max_seq_len);
                return 0;
            }

            if let Err(e) = self.forward_step_device(token as usize, cur_pos, &mut ws, is_last) {
                tracing::error!("CUDA forward_greedy step failed at pos {cur_pos}: {e:?}");
                return 0;
            }

            self.seq_len.store(cur_pos + 1, Ordering::Relaxed);
            state.seq_len += 1;
        }

        // Execute GPU-resident argmax directly into ws.pinned_token via zero-copy UMA
        let ws_ref = &mut *ws;
        if let Err(e) =
            self.ctx
                .argmax_f32_pinned(&mut ws_ref.pinned_token, &ws_ref.logits, vocab_size)
        {
            tracing::error!("CUDA argmax_f32 failed: {e:?}");
            return 0;
        }

        if let Err(e) = self.ctx.synchronize() {
            tracing::error!("CUDA synchronize failed: {e:?}");
            return 0;
        }

        // Read back ONLY 4 bytes directly from pinned host memory without intermediate copies
        let pinned_slice = ws_ref.pinned_token.as_slice();
        u32::from_ne_bytes(pinned_slice[..4].try_into().unwrap_or([0; 4]))
    }

    fn gpu_memory_bytes(&self) -> u64 {
        self.gpu_mem_bytes
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn truncate_kv(&self, state: &mut InferenceState, len: usize) {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.seq_len.store(len, Ordering::Relaxed);
        state.seq_len = len;
        if len == 0 {
            let _ = self.zero_conv_buffers();
        }
    }
}
