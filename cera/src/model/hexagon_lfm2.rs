//! Native Qualcomm Hexagon NPU model implementation for dense transformer architectures.
//!
//! Provides HTP-accelerated forward execution on Snapdragon mobile and edge platforms
//! using FastRPC shared memory and per-architecture DSP skeleton libraries.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::backend::cpu::RopeType;
use crate::backend::hexagon::{
    HTP_TENSOR_REPACK, HTP_TENSOR_WEIGHT, HexagonArch, HexagonContext, HexagonDevice,
    HexagonOpBatch, HtpDataType, HtpOpCode, HtpOpDesc, HtpTensor, RpcmemBuffer,
    build_mul_mat_kernel_params, build_rms_norm_params, build_rope_params, repack_q4_0_tiled,
    repack_q8_0_tiled, tiled_matrix_size_q4_0, tiled_matrix_size_q8_0,
};
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, KvCompression, KvRewindError};
use crate::model::session_gate::{ModelSessionGate, ModelSessionLease};
use crate::model::{BlockType, Model, ModelConfig, ScalarMultipliers};
use crate::session::CeraError;

/// Repacked weight buffers for a single transformer layer on the DSP.
struct HexagonLayerWeights {
    attn_norm: RpcmemBuffer,
    attn_q: RpcmemBuffer,
    attn_k: RpcmemBuffer,
    attn_v: RpcmemBuffer,
    attn_output: RpcmemBuffer,
    ffn_norm: RpcmemBuffer,
    ffn_gate: RpcmemBuffer,
    ffn_up: RpcmemBuffer,
    ffn_down: RpcmemBuffer,

    // Dimensions and dtypes
    q_dtype: HtpDataType,
    k_dtype: HtpDataType,
    v_dtype: HtpDataType,
    out_dtype: HtpDataType,
    ffn_gate_dtype: HtpDataType,
    ffn_up_dtype: HtpDataType,
    ffn_down_dtype: HtpDataType,
}

/// Hexagon NPU accelerated model instance for dense transformers.
pub struct HexagonLfm2Model {
    #[allow(dead_code)]
    context: Arc<HexagonContext>,
    device: Mutex<HexagonDevice>,
    config: ModelConfig,
    session_gate: ModelSessionGate,

    // Token embedding on CPU (matching GPU model architecture)
    token_embd: Vec<f32>,

    // Per-layer weights
    layers: Vec<HexagonLayerWeights>,
    output_norm: RpcmemBuffer,
    lm_head: RpcmemBuffer,
    lm_head_dtype: HtpDataType,

    // Per-layer KV cache in rpcmem
    k_cache: Vec<RpcmemBuffer>,
    v_cache: Vec<RpcmemBuffer>,

    // Scratch buffers
    activation_buf: RpcmemBuffer,
    normed_buf: RpcmemBuffer,
    q_buf: RpcmemBuffer,
    k_buf: RpcmemBuffer,
    v_buf: RpcmemBuffer,
    attn_out_buf: RpcmemBuffer,
    ffn_gate_buf: RpcmemBuffer,
    ffn_up_buf: RpcmemBuffer,
    ffn_out_buf: RpcmemBuffer,
    logits_buf: RpcmemBuffer,

    // Sequence tracking
    rope_type: RopeType,
    current_seq_len: AtomicUsize,
    seq_counter: AtomicUsize,
}

// Compile-time proof that HexagonLfm2Model is Send + Sync.
unsafe impl Send for HexagonLfm2Model {}
unsafe impl Sync for HexagonLfm2Model {}

impl HexagonLfm2Model {
    /// Load a dense transformer model onto the Hexagon NPU from GGUF.
    pub fn from_gguf(
        gguf: GgufFile,
        _path: Option<&Path>,
        context_size: usize,
    ) -> Result<Self, CeraError> {
        let arch = gguf
            .get_str("general.architecture")
            .unwrap_or("llama")
            .to_string();

        let prefix = arch.clone();
        let n_layers = gguf
            .get_u32(&format!("{prefix}.block_count"))
            .ok_or_else(|| CeraError::Backend("missing block_count".into()))?
            as usize;
        let hidden_size = gguf
            .get_u32(&format!("{prefix}.embedding_length"))
            .ok_or_else(|| CeraError::Backend("missing embedding_length".into()))?
            as usize;
        let intermediate_size = gguf
            .get_u32(&format!("{prefix}.feed_forward_length"))
            .ok_or_else(|| CeraError::Backend("missing feed_forward_length".into()))?
            as usize;
        let n_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count"))
            .ok_or_else(|| CeraError::Backend("missing head_count".into()))?
            as usize;
        let n_kv_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(n_heads as u32) as usize;
        let head_dim = gguf
            .get_u32(&format!("{prefix}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or_else(|| hidden_size / n_heads);

        let vocab_size = gguf
            .get_u32(&format!("{prefix}.vocab_size"))
            .map(|v| v as usize)
            .unwrap_or(32000);

        let gguf_max_seq_len = gguf
            .get_u32(&format!("{prefix}.context_length"))
            .unwrap_or(4096) as usize;
        let max_seq_len = context_size.min(gguf_max_seq_len);

        let rope_theta = gguf
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(10000.0);
        let rms_norm_eps = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        let rope_type = match arch.as_str() {
            "qwen2" | "qwen3" => RopeType::Neox,
            _ => RopeType::Norm,
        };

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
            block_types: vec![BlockType::Attention; n_layers],
            conv_kernel_size: None,
            ssm: None,
            kv_heads_per_layer: vec![n_kv_heads; n_layers],
            scalars: ScalarMultipliers::default(),
            moe: None,
            is_causal: true,
            class_labels: Vec::new(),
        };

        // Initialize FastRPC userspace driver
        let context = HexagonContext::new()?;

        // Probe available Hexagon architectures in descending generation order (V81 -> V79 -> V75 -> V73)
        // or prioritize an explicit architecture override via CERA_HEXAGON_ARCH.
        let arch_override = std::env::var("CERA_HEXAGON_ARCH")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .and_then(HexagonArch::from_u32);

        let probe_archs: Vec<HexagonArch> = if let Some(arch) = arch_override {
            vec![arch]
        } else {
            vec![
                HexagonArch::V81,
                HexagonArch::V79,
                HexagonArch::V75,
                HexagonArch::V73,
            ]
        };

        let mut device_opt = None;
        let mut probed_errors = Vec::new();
        for arch in probe_archs {
            match HexagonDevice::new(Arc::clone(context.driver()), arch) {
                Ok(dev) => {
                    tracing::info!(arch = ?arch, "initialized Hexagon NPU device");
                    device_opt = Some(dev);
                    break;
                }
                Err(e) => {
                    probed_errors.push(format!("{arch:?}: {e}"));
                }
            }
        }
        let device = match device_opt {
            Some(d) => d,
            None => {
                return Err(CeraError::Backend(format!(
                    "no compatible Hexagon skeleton library found (probed {}). Errors: {}",
                    if arch_override.is_some() {
                        "override"
                    } else {
                        "V81, V79, V75, V73"
                    },
                    probed_errors.join("; ")
                )));
            }
        };

        // Repack token embeddings to CPU f32 table
        let token_embd = gguf
            .get_tensor("token_embd.weight")
            .map_err(|e| CeraError::Backend(format!("missing token_embd.weight: {e}")))?
            .to_f32_vec();

        // Scratch buffer allocations
        let driver = context.driver();
        let activation_buf = RpcmemBuffer::alloc(Arc::clone(driver), hidden_size * 4, true)?;
        let normed_buf = RpcmemBuffer::alloc(Arc::clone(driver), hidden_size * 4, true)?;
        let q_buf = RpcmemBuffer::alloc(Arc::clone(driver), n_heads * head_dim * 4, true)?;
        let k_buf = RpcmemBuffer::alloc(Arc::clone(driver), n_kv_heads * head_dim * 4, true)?;
        let v_buf = RpcmemBuffer::alloc(Arc::clone(driver), n_kv_heads * head_dim * 4, true)?;
        let attn_out_buf = RpcmemBuffer::alloc(Arc::clone(driver), hidden_size * 4, true)?;
        let ffn_gate_buf = RpcmemBuffer::alloc(Arc::clone(driver), intermediate_size * 4, true)?;
        let ffn_up_buf = RpcmemBuffer::alloc(Arc::clone(driver), intermediate_size * 4, true)?;
        let ffn_out_buf = RpcmemBuffer::alloc(Arc::clone(driver), hidden_size * 4, true)?;
        let logits_buf = RpcmemBuffer::alloc(Arc::clone(driver), vocab_size * 4, true)?;

        // Per-layer KV caches (max_seq_len * n_kv_heads * head_dim * sizeof(f32))
        let kv_slab_size = max_seq_len * n_kv_heads * head_dim * 4;
        let mut k_cache = Vec::with_capacity(n_layers);
        let mut v_cache = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_cache.push(RpcmemBuffer::alloc(Arc::clone(driver), kv_slab_size, true)?);
            v_cache.push(RpcmemBuffer::alloc(Arc::clone(driver), kv_slab_size, true)?);
        }

        // Helper to load and repack a 2D weight matrix
        let load_weight = |name: &str| -> Result<(RpcmemBuffer, HtpDataType), CeraError> {
            let t = gguf
                .tensors
                .get(name)
                .ok_or_else(|| CeraError::Backend(format!("missing tensor {name}")))?;
            let raw_data = gguf
                .tensor_data(name)
                .map_err(|e| CeraError::Backend(e.to_string()))?;
            if t.shape.len() < 2 {
                return Err(CeraError::Backend(format!(
                    "tensor {name} must have at least 2 dimensions, got {:?}",
                    t.shape
                )));
            }
            let ne0 = t.shape[0];
            let ne1 = t.shape[1];

            match t.dtype {
                crate::tensor::DType::Q4_0 => {
                    let repacked_size = tiled_matrix_size_q4_0(ne0, ne1);
                    let mut buf = RpcmemBuffer::alloc(Arc::clone(driver), repacked_size, true)?;
                    repack_q4_0_tiled(raw_data, ne0, ne1, buf.as_mut_slice())?;
                    buf.flush_cpu_cache(0, repacked_size);
                    Ok((buf, HtpDataType::Q4_0Tiled))
                }
                crate::tensor::DType::Q8_0 => {
                    let repacked_size = tiled_matrix_size_q8_0(ne0, ne1);
                    let mut buf = RpcmemBuffer::alloc(Arc::clone(driver), repacked_size, true)?;
                    repack_q8_0_tiled(raw_data, ne0, ne1, buf.as_mut_slice())?;
                    buf.flush_cpu_cache(0, repacked_size);
                    Ok((buf, HtpDataType::Q8_0Tiled))
                }
                other => Err(CeraError::Backend(format!(
                    "unsupported quant format {other:?} for Hexagon weight {name}"
                ))),
            }
        };

        // Helper to load 1D norm weights (F32)
        let load_norm = |name: &str| -> Result<RpcmemBuffer, CeraError> {
            let t = gguf
                .get_tensor(name)
                .map_err(|e| CeraError::Backend(format!("missing tensor {name}: {e}")))?;
            let f32_vals = t.to_f32_vec();
            let byte_size = f32_vals.len() * std::mem::size_of::<f32>();
            let buf = RpcmemBuffer::alloc(Arc::clone(driver), byte_size, true)?;
            // Copy as bytes to avoid any alignment requirement on raw shared memory pointers
            unsafe {
                std::ptr::copy_nonoverlapping(
                    f32_vals.as_ptr() as *const u8,
                    buf.as_mut_ptr(),
                    byte_size,
                );
            }
            buf.flush_cpu_cache(0, byte_size);
            Ok(buf)
        };

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let attn_norm = load_norm(&format!("blk.{i}.attn_norm.weight"))?;
            let (attn_q, q_dtype) = load_weight(&format!("blk.{i}.attn_q.weight"))?;
            let (attn_k, k_dtype) = load_weight(&format!("blk.{i}.attn_k.weight"))?;
            let (attn_v, v_dtype) = load_weight(&format!("blk.{i}.attn_v.weight"))?;
            let (attn_output, out_dtype) = load_weight(&format!("blk.{i}.attn_output.weight"))?;
            let ffn_norm = load_norm(&format!("blk.{i}.ffn_norm.weight"))?;
            let (ffn_gate, ffn_gate_dtype) = load_weight(&format!("blk.{i}.ffn_gate.weight"))?;
            let (ffn_up, ffn_up_dtype) = load_weight(&format!("blk.{i}.ffn_up.weight"))?;
            let (ffn_down, ffn_down_dtype) = load_weight(&format!("blk.{i}.ffn_down.weight"))?;

            layers.push(HexagonLayerWeights {
                attn_norm,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                q_dtype,
                k_dtype,
                v_dtype,
                out_dtype,
                ffn_gate_dtype,
                ffn_up_dtype,
                ffn_down_dtype,
            });
        }

        let output_norm = load_norm("output_norm.weight")?;
        let (lm_head, lm_head_dtype) = if gguf.tensors.contains_key("output.weight") {
            load_weight("output.weight")?
        } else {
            load_weight("token_embd.weight")?
        };

        Ok(Self {
            context,
            device: Mutex::new(device),
            config,
            session_gate: ModelSessionGate::default(),
            token_embd,
            layers,
            output_norm,
            lm_head,
            lm_head_dtype,
            k_cache,
            v_cache,
            activation_buf,
            normed_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            ffn_gate_buf,
            ffn_up_buf,
            ffn_out_buf,
            logits_buf,
            rope_type,
            current_seq_len: AtomicUsize::new(0),
            seq_counter: AtomicUsize::new(1),
        })
    }
}

impl Model for HexagonLfm2Model {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn acquire_session(&self) -> Result<Option<ModelSessionLease>, CeraError> {
        self.session_gate.try_acquire().map(Some)
    }

    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        if tokens.is_empty() {
            return Vec::new();
        }

        let token = tokens[0] as usize;
        if token >= self.config.vocab_size {
            tracing::warn!(
                token,
                vocab_size = self.config.vocab_size,
                "token ID exceeds model vocab size; returning zero logits"
            );
            return vec![0.0f32; self.config.vocab_size];
        }

        // Acquire device lock BEFORE writing to shared scratch activation buffer to prevent concurrent races
        let mut device = self.device.lock().unwrap_or_else(|e| e.into_inner());

        let hs = self.config.hidden_size;
        let embd_start = token * hs;
        let embd_slice = &self.token_embd[embd_start..embd_start + hs];

        // Copy embedding vector to activation shared buffer as raw bytes under the device lock
        unsafe {
            std::ptr::copy_nonoverlapping(
                embd_slice.as_ptr() as *const u8,
                self.activation_buf.as_mut_ptr(),
                hs * std::mem::size_of::<f32>(),
            );
        }
        self.activation_buf.flush_cpu_cache(0, hs * 4);

        let seq = self.seq_counter.fetch_add(1, Ordering::Relaxed) as u64;

        let mut batch = HexagonOpBatch::new();

        // Register activation buffer
        let act_bi = batch.add_buffer(
            self.activation_buf.base(),
            self.activation_buf.size() as u64,
            0,
            self.activation_buf.fd() as u32,
        );

        let act_ti = batch.add_tensor(HtpTensor {
            data: 0,
            size: (hs * 4) as u32,
            flags: 0,
            dtype: HtpDataType::F32 as u32,
            bi: act_bi,
            ti: 0,
            ne: [hs as u32, 1, 1, 1],
            nb: [4, (hs * 4) as u32, (hs * 4) as u32, (hs * 4) as u32],
        });

        let q_dim = self.config.n_heads * self.config.head_dim;
        let kv_dim = self.config.n_kv_heads * self.config.head_dim;
        let n_threads = device.hw_info().n_threads;

        // Loop over layers and enqueue HTP operators
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Attention RMS norm
            let norm_bi = batch.add_buffer(
                layer.attn_norm.base(),
                layer.attn_norm.size() as u64,
                0,
                layer.attn_norm.fd() as u32,
            );
            let norm_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: (hs * 4) as u32,
                flags: HTP_TENSOR_WEIGHT,
                dtype: HtpDataType::F32 as u32,
                bi: norm_bi,
                ti: 0,
                ne: [hs as u32, 1, 1, 1],
                nb: [4, (hs * 4) as u32, (hs * 4) as u32, (hs * 4) as u32],
            });

            let normed_bi = batch.add_buffer(
                self.normed_buf.base(),
                self.normed_buf.size() as u64,
                0,
                self.normed_buf.fd() as u32,
            );
            let normed_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: (hs * 4) as u32,
                flags: 0,
                dtype: HtpDataType::F32 as u32,
                bi: normed_bi,
                ti: 0,
                ne: [hs as u32, 1, 1, 1],
                nb: [4, (hs * 4) as u32, (hs * 4) as u32, (hs * 4) as u32],
            });

            let mut rms_norm_op = HtpOpDesc::new(HtpOpCode::RmsNorm);
            rms_norm_op.params = build_rms_norm_params(self.config.rms_norm_eps);
            rms_norm_op.src[0] = act_ti;
            rms_norm_op.src[1] = norm_ti;
            rms_norm_op.dst[0] = normed_ti;
            batch.add_op(rms_norm_op);

            // Q projection
            let q_w_bi = batch.add_buffer(
                layer.attn_q.base(),
                layer.attn_q.size() as u64,
                0,
                layer.attn_q.fd() as u32,
            );
            let q_w_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: layer.attn_q.size() as u32,
                flags: HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
                dtype: layer.q_dtype as u32,
                bi: q_w_bi,
                ti: 0,
                ne: [hs as u32, q_dim as u32, 1, 1],
                nb: [
                    4,
                    (hs * 4) as u32,
                    (hs * q_dim * 4) as u32,
                    (hs * q_dim * 4) as u32,
                ],
            });

            let q_out_bi = batch.add_buffer(
                self.q_buf.base(),
                self.q_buf.size() as u64,
                0,
                self.q_buf.fd() as u32,
            );
            let q_out_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: (q_dim * 4) as u32,
                flags: 0,
                dtype: HtpDataType::F32 as u32,
                bi: q_out_bi,
                ti: 0,
                ne: [q_dim as u32, 1, 1, 1],
                nb: [
                    4,
                    (q_dim * 4) as u32,
                    (q_dim * 4) as u32,
                    (q_dim * 4) as u32,
                ],
            });

            let mut q_mat_op = HtpOpDesc::new(HtpOpCode::MulMat);
            q_mat_op.kernel_params = build_mul_mat_kernel_params(1, 1, n_threads);
            q_mat_op.src[0] = q_w_ti;
            q_mat_op.src[1] = normed_ti;
            q_mat_op.dst[0] = q_out_ti;
            batch.add_op(q_mat_op);

            // K projection
            let k_w_bi = batch.add_buffer(
                layer.attn_k.base(),
                layer.attn_k.size() as u64,
                0,
                layer.attn_k.fd() as u32,
            );
            let k_w_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: layer.attn_k.size() as u32,
                flags: HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
                dtype: layer.k_dtype as u32,
                bi: k_w_bi,
                ti: 0,
                ne: [hs as u32, kv_dim as u32, 1, 1],
                nb: [
                    4,
                    (hs * 4) as u32,
                    (hs * kv_dim * 4) as u32,
                    (hs * kv_dim * 4) as u32,
                ],
            });

            let k_out_bi = batch.add_buffer(
                self.k_buf.base(),
                self.k_buf.size() as u64,
                0,
                self.k_buf.fd() as u32,
            );
            let k_out_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: (kv_dim * 4) as u32,
                flags: 0,
                dtype: HtpDataType::F32 as u32,
                bi: k_out_bi,
                ti: 0,
                ne: [kv_dim as u32, 1, 1, 1],
                nb: [
                    4,
                    (kv_dim * 4) as u32,
                    (kv_dim * 4) as u32,
                    (kv_dim * 4) as u32,
                ],
            });

            let mut k_mat_op = HtpOpDesc::new(HtpOpCode::MulMat);
            k_mat_op.kernel_params = build_mul_mat_kernel_params(1, 1, n_threads);
            k_mat_op.src[0] = k_w_ti;
            k_mat_op.src[1] = normed_ti;
            k_mat_op.dst[0] = k_out_ti;
            batch.add_op(k_mat_op);

            // V projection
            let v_w_bi = batch.add_buffer(
                layer.attn_v.base(),
                layer.attn_v.size() as u64,
                0,
                layer.attn_v.fd() as u32,
            );
            let v_w_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: layer.attn_v.size() as u32,
                flags: HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
                dtype: layer.v_dtype as u32,
                bi: v_w_bi,
                ti: 0,
                ne: [hs as u32, kv_dim as u32, 1, 1],
                nb: [
                    4,
                    (hs * 4) as u32,
                    (hs * kv_dim * 4) as u32,
                    (hs * kv_dim * 4) as u32,
                ],
            });

            let v_out_bi = batch.add_buffer(
                self.v_buf.base(),
                self.v_buf.size() as u64,
                0,
                self.v_buf.fd() as u32,
            );
            let v_out_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: (kv_dim * 4) as u32,
                flags: 0,
                dtype: HtpDataType::F32 as u32,
                bi: v_out_bi,
                ti: 0,
                ne: [kv_dim as u32, 1, 1, 1],
                nb: [
                    4,
                    (kv_dim * 4) as u32,
                    (kv_dim * 4) as u32,
                    (kv_dim * 4) as u32,
                ],
            });

            let mut v_mat_op = HtpOpDesc::new(HtpOpCode::MulMat);
            v_mat_op.kernel_params = build_mul_mat_kernel_params(1, 1, n_threads);
            v_mat_op.src[0] = v_w_ti;
            v_mat_op.src[1] = normed_ti;
            v_mat_op.dst[0] = v_out_ti;
            batch.add_op(v_mat_op);

            // RoPE on Q and K
            let rope_mode = match self.rope_type {
                RopeType::Neox => 2u32,
                _ => 0u32,
            };
            let rope_params = build_rope_params(
                pos,
                self.config.head_dim,
                rope_mode,
                self.config.max_seq_len as u32,
                self.config.rope_theta,
                1.0,
            );

            let mut rope_q_op = HtpOpDesc::new(HtpOpCode::Rope);
            rope_q_op.params = rope_params;
            rope_q_op.src[0] = q_out_ti;
            rope_q_op.dst[0] = q_out_ti;
            batch.add_op(rope_q_op);

            let mut rope_k_op = HtpOpDesc::new(HtpOpCode::Rope);
            rope_k_op.params = rope_params;
            rope_k_op.src[0] = k_out_ti;
            rope_k_op.dst[0] = k_out_ti;
            batch.add_op(rope_k_op);

            // KV Cache buffers
            let k_cache_buf = &self.k_cache[layer_idx];
            let v_cache_buf = &self.v_cache[layer_idx];

            let k_cache_bi = batch.add_buffer(
                k_cache_buf.base(),
                k_cache_buf.size() as u64,
                0,
                k_cache_buf.fd() as u32,
            );
            let k_cache_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: k_cache_buf.size() as u32,
                flags: 0,
                dtype: HtpDataType::F32 as u32,
                bi: k_cache_bi,
                ti: 0,
                ne: [
                    self.config.head_dim as u32,
                    self.config.n_kv_heads as u32,
                    self.config.max_seq_len as u32,
                    1,
                ],
                nb: [
                    4,
                    (self.config.head_dim * 4) as u32,
                    (kv_dim * 4) as u32,
                    (kv_dim * self.config.max_seq_len * 4) as u32,
                ],
            });

            let v_cache_bi = batch.add_buffer(
                v_cache_buf.base(),
                v_cache_buf.size() as u64,
                0,
                v_cache_buf.fd() as u32,
            );
            let v_cache_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: v_cache_buf.size() as u32,
                flags: 0,
                dtype: HtpDataType::F32 as u32,
                bi: v_cache_bi,
                ti: 0,
                ne: [
                    self.config.head_dim as u32,
                    self.config.n_kv_heads as u32,
                    self.config.max_seq_len as u32,
                    1,
                ],
                nb: [
                    4,
                    (self.config.head_dim * 4) as u32,
                    (kv_dim * 4) as u32,
                    (kv_dim * self.config.max_seq_len * 4) as u32,
                ],
            });

            // Attention FlashAttnExt
            let attn_out_bi = batch.add_buffer(
                self.attn_out_buf.base(),
                self.attn_out_buf.size() as u64,
                0,
                self.attn_out_buf.fd() as u32,
            );
            let attn_out_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: (hs * 4) as u32,
                flags: 0,
                dtype: HtpDataType::F32 as u32,
                bi: attn_out_bi,
                ti: 0,
                ne: [hs as u32, 1, 1, 1],
                nb: [4, (hs * 4) as u32, (hs * 4) as u32, (hs * 4) as u32],
            });

            // Insert current token K and V projections into KV cache slabs at row index pos
            let mut set_k_op = HtpOpDesc::new(HtpOpCode::SetRows);
            set_k_op.params[0] = pos as i32;
            set_k_op.src[0] = k_out_ti;
            set_k_op.src[1] = k_cache_ti;
            set_k_op.dst[0] = k_cache_ti;
            batch.add_op(set_k_op);

            let mut set_v_op = HtpOpDesc::new(HtpOpCode::SetRows);
            set_v_op.params[0] = pos as i32;
            set_v_op.src[0] = v_out_ti;
            set_v_op.src[1] = v_cache_ti;
            set_v_op.dst[0] = v_cache_ti;
            batch.add_op(set_v_op);

            let mut attn_op = HtpOpDesc::new(HtpOpCode::FlashAttnExt);
            attn_op.src[0] = q_out_ti;
            attn_op.src[1] = k_cache_ti;
            attn_op.src[2] = v_cache_ti;
            attn_op.dst[0] = attn_out_ti;
            batch.add_op(attn_op);

            // Attention output projection
            let attn_proj_bi = batch.add_buffer(
                layer.attn_output.base(),
                layer.attn_output.size() as u64,
                0,
                layer.attn_output.fd() as u32,
            );
            let attn_proj_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: layer.attn_output.size() as u32,
                flags: HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
                dtype: layer.out_dtype as u32,
                bi: attn_proj_bi,
                ti: 0,
                ne: [q_dim as u32, hs as u32, 1, 1],
                nb: [
                    4,
                    (q_dim * 4) as u32,
                    (hs * q_dim * 4) as u32,
                    (hs * q_dim * 4) as u32,
                ],
            });

            let mut attn_proj_op = HtpOpDesc::new(HtpOpCode::MulMat);
            attn_proj_op.kernel_params = build_mul_mat_kernel_params(1, 1, n_threads);
            attn_proj_op.src[0] = attn_proj_ti;
            attn_proj_op.src[1] = attn_out_ti;
            attn_proj_op.dst[0] = normed_ti;
            batch.add_op(attn_proj_op);

            // Residual add: act = act + normed
            let mut res_add_op = HtpOpDesc::new(HtpOpCode::Add);
            res_add_op.src[0] = act_ti;
            res_add_op.src[1] = normed_ti;
            res_add_op.dst[0] = act_ti;
            batch.add_op(res_add_op);

            // FFN RMS norm
            let ffn_norm_bi = batch.add_buffer(
                layer.ffn_norm.base(),
                layer.ffn_norm.size() as u64,
                0,
                layer.ffn_norm.fd() as u32,
            );
            let ffn_norm_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: (hs * 4) as u32,
                flags: HTP_TENSOR_WEIGHT,
                dtype: HtpDataType::F32 as u32,
                bi: ffn_norm_bi,
                ti: 0,
                ne: [hs as u32, 1, 1, 1],
                nb: [4, (hs * 4) as u32, (hs * 4) as u32, (hs * 4) as u32],
            });

            let mut ffn_norm_op = HtpOpDesc::new(HtpOpCode::RmsNorm);
            ffn_norm_op.params = build_rms_norm_params(self.config.rms_norm_eps);
            ffn_norm_op.src[0] = act_ti;
            ffn_norm_op.src[1] = ffn_norm_ti;
            ffn_norm_op.dst[0] = normed_ti;
            batch.add_op(ffn_norm_op);

            // FFN Gate projection
            let ffn_gate_w_bi = batch.add_buffer(
                layer.ffn_gate.base(),
                layer.ffn_gate.size() as u64,
                0,
                layer.ffn_gate.fd() as u32,
            );
            let intermediate_size = self.config.intermediate_size;
            let ffn_gate_w_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: layer.ffn_gate.size() as u32,
                flags: HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
                dtype: layer.ffn_gate_dtype as u32,
                bi: ffn_gate_w_bi,
                ti: 0,
                ne: [hs as u32, intermediate_size as u32, 1, 1],
                nb: [
                    4,
                    (hs * 4) as u32,
                    (hs * intermediate_size * 4) as u32,
                    (hs * intermediate_size * 4) as u32,
                ],
            });

            let ffn_gate_out_bi = batch.add_buffer(
                self.ffn_gate_buf.base(),
                self.ffn_gate_buf.size() as u64,
                0,
                self.ffn_gate_buf.fd() as u32,
            );
            let ffn_gate_out_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: (intermediate_size * 4) as u32,
                flags: 0,
                dtype: HtpDataType::F32 as u32,
                bi: ffn_gate_out_bi,
                ti: 0,
                ne: [intermediate_size as u32, 1, 1, 1],
                nb: [
                    4,
                    (intermediate_size * 4) as u32,
                    (intermediate_size * 4) as u32,
                    (intermediate_size * 4) as u32,
                ],
            });

            let mut ffn_gate_mat_op = HtpOpDesc::new(HtpOpCode::MulMat);
            ffn_gate_mat_op.kernel_params = build_mul_mat_kernel_params(1, 1, n_threads);
            ffn_gate_mat_op.src[0] = ffn_gate_w_ti;
            ffn_gate_mat_op.src[1] = normed_ti;
            ffn_gate_mat_op.dst[0] = ffn_gate_out_ti;
            batch.add_op(ffn_gate_mat_op);

            // FFN Up projection
            let ffn_up_w_bi = batch.add_buffer(
                layer.ffn_up.base(),
                layer.ffn_up.size() as u64,
                0,
                layer.ffn_up.fd() as u32,
            );
            let ffn_up_w_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: layer.ffn_up.size() as u32,
                flags: HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
                dtype: layer.ffn_up_dtype as u32,
                bi: ffn_up_w_bi,
                ti: 0,
                ne: [hs as u32, intermediate_size as u32, 1, 1],
                nb: [
                    4,
                    (hs * 4) as u32,
                    (hs * intermediate_size * 4) as u32,
                    (hs * intermediate_size * 4) as u32,
                ],
            });

            let ffn_up_out_bi = batch.add_buffer(
                self.ffn_up_buf.base(),
                self.ffn_up_buf.size() as u64,
                0,
                self.ffn_up_buf.fd() as u32,
            );
            let ffn_up_out_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: (intermediate_size * 4) as u32,
                flags: 0,
                dtype: HtpDataType::F32 as u32,
                bi: ffn_up_out_bi,
                ti: 0,
                ne: [intermediate_size as u32, 1, 1, 1],
                nb: [
                    4,
                    (intermediate_size * 4) as u32,
                    (intermediate_size * 4) as u32,
                    (intermediate_size * 4) as u32,
                ],
            });

            let mut ffn_up_mat_op = HtpOpDesc::new(HtpOpCode::MulMat);
            ffn_up_mat_op.kernel_params = build_mul_mat_kernel_params(1, 1, n_threads);
            ffn_up_mat_op.src[0] = ffn_up_w_ti;
            ffn_up_mat_op.src[1] = normed_ti;
            ffn_up_mat_op.dst[0] = ffn_up_out_ti;
            batch.add_op(ffn_up_mat_op);

            // SwiGLU activation
            let ffn_out_bi = batch.add_buffer(
                self.ffn_out_buf.base(),
                self.ffn_out_buf.size() as u64,
                0,
                self.ffn_out_buf.fd() as u32,
            );
            let ffn_out_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: (intermediate_size * 4) as u32,
                flags: 0,
                dtype: HtpDataType::F32 as u32,
                bi: ffn_out_bi,
                ti: 0,
                ne: [intermediate_size as u32, 1, 1, 1],
                nb: [
                    4,
                    (intermediate_size * 4) as u32,
                    (intermediate_size * 4) as u32,
                    (intermediate_size * 4) as u32,
                ],
            });

            let mut swiglu_op = HtpOpDesc::new(HtpOpCode::GluSwiglu);
            swiglu_op.src[0] = ffn_gate_out_ti;
            swiglu_op.src[1] = ffn_up_out_ti;
            swiglu_op.dst[0] = ffn_out_ti;
            batch.add_op(swiglu_op);

            // FFN Down projection
            let ffn_down_w_bi = batch.add_buffer(
                layer.ffn_down.base(),
                layer.ffn_down.size() as u64,
                0,
                layer.ffn_down.fd() as u32,
            );
            let ffn_down_w_ti = batch.add_tensor(HtpTensor {
                data: 0,
                size: layer.ffn_down.size() as u32,
                flags: HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
                dtype: layer.ffn_down_dtype as u32,
                bi: ffn_down_w_bi,
                ti: 0,
                ne: [intermediate_size as u32, hs as u32, 1, 1],
                nb: [
                    4,
                    (intermediate_size * 4) as u32,
                    (intermediate_size * hs * 4) as u32,
                    (intermediate_size * hs * 4) as u32,
                ],
            });

            let mut ffn_down_mat_op = HtpOpDesc::new(HtpOpCode::MulMat);
            ffn_down_mat_op.kernel_params = build_mul_mat_kernel_params(1, 1, n_threads);
            ffn_down_mat_op.src[0] = ffn_down_w_ti;
            ffn_down_mat_op.src[1] = ffn_out_ti;
            ffn_down_mat_op.dst[0] = normed_ti;
            batch.add_op(ffn_down_mat_op);

            // Residual add: act = act + normed
            let mut res2_add_op = HtpOpDesc::new(HtpOpCode::Add);
            res2_add_op.src[0] = act_ti;
            res2_add_op.src[1] = normed_ti;
            res2_add_op.dst[0] = act_ti;
            batch.add_op(res2_add_op);
        }

        // Output norm and final LM head projection
        let out_norm_bi = batch.add_buffer(
            self.output_norm.base(),
            self.output_norm.size() as u64,
            0,
            self.output_norm.fd() as u32,
        );
        let out_norm_ti = batch.add_tensor(HtpTensor {
            data: 0,
            size: (hs * 4) as u32,
            flags: HTP_TENSOR_WEIGHT,
            dtype: HtpDataType::F32 as u32,
            bi: out_norm_bi,
            ti: 0,
            ne: [hs as u32, 1, 1, 1],
            nb: [4, (hs * 4) as u32, (hs * 4) as u32, (hs * 4) as u32],
        });

        let mut out_norm_op = HtpOpDesc::new(HtpOpCode::RmsNorm);
        out_norm_op.params = build_rms_norm_params(self.config.rms_norm_eps);
        out_norm_op.src[0] = act_ti;
        out_norm_op.src[1] = out_norm_ti;
        out_norm_op.dst[0] = act_ti;
        batch.add_op(out_norm_op);

        // Final LM head projection
        let vocab_size = self.config.vocab_size;
        let lm_head_bi = batch.add_buffer(
            self.lm_head.base(),
            self.lm_head.size() as u64,
            0,
            self.lm_head.fd() as u32,
        );
        let lm_head_ti = batch.add_tensor(HtpTensor {
            data: 0,
            size: self.lm_head.size() as u32,
            flags: HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
            dtype: self.lm_head_dtype as u32,
            bi: lm_head_bi,
            ti: 0,
            ne: [hs as u32, vocab_size as u32, 1, 1],
            nb: [
                4,
                (hs * 4) as u32,
                (hs * vocab_size * 4) as u32,
                (hs * vocab_size * 4) as u32,
            ],
        });

        let logits_bi = batch.add_buffer(
            self.logits_buf.base(),
            self.logits_buf.size() as u64,
            0,
            self.logits_buf.fd() as u32,
        );
        let logits_ti = batch.add_tensor(HtpTensor {
            data: 0,
            size: (vocab_size * 4) as u32,
            flags: 0,
            dtype: HtpDataType::F32 as u32,
            bi: logits_bi,
            ti: 0,
            ne: [vocab_size as u32, 1, 1, 1],
            nb: [
                4,
                (vocab_size * 4) as u32,
                (vocab_size * 4) as u32,
                (vocab_size * 4) as u32,
            ],
        });

        let mut lm_head_op = HtpOpDesc::new(HtpOpCode::MulMat);
        lm_head_op.kernel_params = build_mul_mat_kernel_params(1, 1, n_threads);
        lm_head_op.src[0] = lm_head_ti;
        lm_head_op.src[1] = act_ti;
        lm_head_op.dst[0] = logits_ti;
        batch.add_op(lm_head_op);

        // Submit to DSP and wait for completion
        if let Err(e) = device.queue_session_mut().submit(&batch, seq) {
            tracing::error!("Hexagon NPU batch submit failed: {e}");
            return vec![0.0f32; self.config.vocab_size];
        }

        if let Err(e) = device.queue_session_mut().wait_completion(seq) {
            tracing::error!("Hexagon NPU execution failed: {e}");
            return vec![0.0f32; self.config.vocab_size];
        }

        self.current_seq_len.store(pos + 1, Ordering::SeqCst);
        state.seq_len = pos + 1;

        // Read back computed logits from shared memory
        self.logits_buf
            .invalidate_cpu_cache(0, self.config.vocab_size * 4);
        let logits_slice = unsafe {
            std::slice::from_raw_parts(
                self.logits_buf.as_ptr() as *const f32,
                self.config.vocab_size,
            )
        };
        logits_slice.to_vec()
    }

    fn supports_all_logits(&self) -> bool {
        false
    }

    fn truncate_kv(&self, state: &mut InferenceState, len: usize) {
        let _guard = self.device.lock().unwrap_or_else(|e| e.into_inner());
        state.truncate_to(len);
        self.current_seq_len.store(len, Ordering::SeqCst);
    }

    fn check_kv_rewind(&self, state: &InferenceState, len: usize) -> Result<(), KvRewindError> {
        let current = self.current_seq_len.load(Ordering::SeqCst);
        if len > current {
            return Err(KvRewindError::OutOfBounds {
                requested: len,
                current,
            });
        }
        state.check_truncate_to(len)
    }

    fn try_truncate_kv(&self, state: &mut InferenceState, len: usize) -> Result<(), KvRewindError> {
        let _guard = self.device.lock().unwrap_or_else(|e| e.into_inner());
        self.check_kv_rewind(state, len)?;
        state.try_truncate_to(len)?;
        self.current_seq_len.store(len, Ordering::SeqCst);
        Ok(())
    }

    fn try_reset_kv(
        &self,
        state: &mut InferenceState,
        compression: &KvCompression,
        max_seq_len: usize,
    ) -> Result<(), CeraError> {
        if !matches!(compression, KvCompression::None) {
            return Err(CeraError::Backend(
                "TurboQuant KV compression is not supported by the Hexagon backend".into(),
            ));
        }
        let _guard = self.device.lock().unwrap_or_else(|e| e.into_inner());
        let mut fresh = InferenceState::from_config_capped(&self.config, compression, max_seq_len)?;
        fresh.lora = state.lora.clone();
        self.current_seq_len.store(0, Ordering::SeqCst);
        *state = fresh;
        Ok(())
    }
}
