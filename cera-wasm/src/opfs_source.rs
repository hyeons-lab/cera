use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;
use wasm_bindgen::JsError;
use web_sys::{FileSystemReadWriteOptions, FileSystemSyncAccessHandle};

use cera::gguf::GgufFile;
use cera::model::ModelConfig;
use cera::model::gpu_weight_source::{GpuWeightSource, RopeType};
use cera::model::lfm2::{LfmModel, MoeFfnRefs};
use cera::model::llama::LlamaModel;
use cera::model::transformer::{FfnActivation, WeightRef};
use cera::model::weights::MmapWeight;
use cera::tensor::Tensor;

/// RAII wrapper around `FileSystemSyncAccessHandle` that guarantees the exclusive
/// browser OPFS file lock is released via `sync.close()` on every exit path,
/// including early error returns before `OpfsGpuWeightSource` is constructed.
pub struct OpfsSyncHandle {
    sync: FileSystemSyncAccessHandle,
    closed: AtomicBool,
}

impl OpfsSyncHandle {
    pub fn new(sync: FileSystemSyncAccessHandle) -> Self {
        Self {
            sync,
            closed: AtomicBool::new(false),
        }
    }

    pub fn as_raw(&self) -> &FileSystemSyncAccessHandle {
        &self.sync
    }

    /// Read `dst.len()` bytes starting at `offset` from OPFS, failing closed if
    /// the read returns fewer bytes than requested.
    pub fn read_exact(&self, offset: u64, dst: &mut [u8]) -> anyhow::Result<()> {
        let opts = FileSystemReadWriteOptions::new();
        opts.set_at(offset as f64);
        let read = self
            .sync
            .read_with_u8_array_and_options(dst, &opts)
            .map_err(|e| anyhow::anyhow!("reading at offset {offset} from OPFS: {e:?}"))?;
        anyhow::ensure!(
            (read as usize) == dst.len(),
            "short read from OPFS at offset {offset}: read {read} of {} bytes",
            dst.len()
        );
        Ok(())
    }

    /// Read the full byte payload for `info` from OPFS, rejecting unmapped GGML
    /// quantization types (`size_bytes == 0`) and short reads.
    pub fn read_tensor_bytes(&self, info: &cera::gguf::TensorInfo) -> anyhow::Result<Vec<u8>> {
        anyhow::ensure!(
            info.size_bytes > 0,
            "tensor {} has unsupported GGML type {} ({})",
            info.name,
            info.ggml_type_id,
            cera::gguf::ggml_type_name(info.ggml_type_id)
        );
        let mut buf = vec![0u8; info.size_bytes];
        self.read_exact(info.offset, &mut buf)
            .with_context(|| format!("reading tensor {}", info.name))?;
        Ok(buf)
    }

    pub fn close(&self) {
        if !self.closed.swap(true, Ordering::Relaxed) {
            self.sync.close();
        }
    }
}

impl Drop for OpfsSyncHandle {
    fn drop(&mut self) {
        self.close();
    }
}

enum OpfsArchSource {
    Lfm2(Box<LfmModel>),
    Llama(Box<LlamaModel>),
}

/// An on-demand weight source that streams tensor data directly from browser OPFS storage
/// (`FileSystemSyncAccessHandle`) into WebGPU buffers.
///
/// Memory footprint: only keeps the lightweight GGUF metadata header and small
/// RMSNorm/conv/bias vectors (< 500 KB) resident in WASM linear memory. Projection matrix
/// bytes are read on demand per tensor during GPU upload (or row-by-row during dequantization),
/// completely avoiding the 32-bit WASM memory limit.
pub struct OpfsGpuWeightSource {
    gguf: Arc<GgufFile>,
    sync: OpfsSyncHandle,
    arch: OpfsArchSource,
}

impl OpfsGpuWeightSource {
    /// Create a new OPFS streaming weight source from an open [`OpfsSyncHandle`]
    /// and parsed header-only `GgufFile`.
    pub fn new(
        sync: OpfsSyncHandle,
        gguf: Arc<GgufFile>,
        context_size: usize,
        model_id: String,
    ) -> Result<Self, JsError> {
        let read_tensor_f32 = |g: &GgufFile, name: &str| -> anyhow::Result<Vec<f32>> {
            let info = g
                .tensors
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("tensor not found: {name}"))?;
            let bytes = sync.read_tensor_bytes(info)?;
            Tensor::new(bytes, info.shape.clone(), info.dtype)
                .try_to_f32_vec()
                .map_err(|e| anyhow::anyhow!("tensor '{name}': {e}"))
        };

        let arch_str = gguf
            .get_str("general.architecture")
            .unwrap_or("lfm2")
            .to_ascii_lowercase();

        let arch = match arch_str.as_str() {
            "lfm2" | "lfm2moe" => {
                let lfm = LfmModel::from_gguf_no_repack_with_loader(
                    (*gguf).clone(),
                    context_size,
                    model_id,
                    read_tensor_f32,
                )
                .map_err(|e| {
                    JsError::new(&format!("loading {arch_str} config from OPFS: {e:#}"))
                })?;
                if let Some(why) = GpuWeightSource::gpu_unsupported_reason(&lfm) {
                    return Err(JsError::new(&why));
                }
                if !GpuWeightSource::config(&lfm).is_causal {
                    return Err(JsError::new(
                        "WebGpuSession runs causal generation only, and this model is bidirectional (CPU only)",
                    ));
                }
                OpfsArchSource::Lfm2(Box::new(lfm))
            }
            "llama" | "qwen2" | "qwen3" | "granite" | "minicpm" | "minicpm5" | "nanbeige"
            | "phi3" | "phi" | "mistral3" | "ministral" | "ministral3" | "internlm2"
            | "internlm" | "baichuan" | "deepseek" | "openelm" => {
                let llama = LlamaModel::from_gguf_no_repack_with_loader(
                    (*gguf).clone(),
                    context_size,
                    model_id,
                    read_tensor_f32,
                )
                .map_err(|e| {
                    JsError::new(&format!("loading {arch_str} config from OPFS: {e:#}"))
                })?;
                if let Some(why) = llama.gpu_unsupported_reason() {
                    return Err(JsError::new(&why));
                }
                if !GpuWeightSource::config(&llama).is_causal {
                    return Err(JsError::new(
                        "WebGpuSession runs causal generation only, and this model is bidirectional (CPU only)",
                    ));
                }
                OpfsArchSource::Llama(Box::new(llama))
            }
            "stablelm" | "starcoder2" | "cohere" | "command-r" => {
                return Err(JsError::new(&format!(
                    "architecture {arch_str} requires LayerNorm/parallel-residual support, which the GPU backends do not implement (CPU only)"
                )));
            }
            other => {
                return Err(JsError::new(&format!(
                    "unsupported architecture for GPU: {other}"
                )));
            }
        };

        Ok(Self { gguf, sync, arch })
    }

    fn inner_source(&self) -> &dyn GpuWeightSource {
        match &self.arch {
            OpfsArchSource::Lfm2(m) => m.as_ref(),
            OpfsArchSource::Llama(m) => m.as_ref(),
        }
    }

    /// Read a 2D weight tensor from OPFS and wrap it in an owned [`MmapWeight`].
    fn read_mmap_weight(&self, name: &str) -> anyhow::Result<MmapWeight> {
        let info = self
            .gguf
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor not found in GGUF: {name}"))?;
        let (rows, cols) = match info.shape.len() {
            2 => (info.shape[1], info.shape[0]),
            1 => (1, info.shape[0]),
            r => anyhow::bail!("MmapWeight: expected 1D or 2D tensor for {name}, got rank {r}"),
        };
        GgufFile::ensure_whole_blocks(name, info.dtype, cols)?;
        let data = self.sync.read_tensor_bytes(info)?;
        Ok(MmapWeight::from_owned_bytes(data, info.dtype, rows, cols))
    }

    /// Close the underlying OPFS sync access handle, releasing its exclusive file lock.
    pub fn close(&self) {
        self.sync.close();
    }
}

impl Drop for OpfsGpuWeightSource {
    fn drop(&mut self) {
        self.close();
    }
}

impl GpuWeightSource for OpfsGpuWeightSource {
    fn config(&self) -> &ModelConfig {
        self.inner_source().config()
    }

    fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    fn output_norm_weight(&self) -> &[f32] {
        self.inner_source().output_norm_weight()
    }

    fn attn_norm_weight(&self, layer: usize) -> &[f32] {
        self.inner_source().attn_norm_weight(layer)
    }

    fn ffn_norm_weight(&self, layer: usize) -> &[f32] {
        self.inner_source().ffn_norm_weight(layer)
    }

    fn attn_q_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.inner_source().attn_q_norm_weight(layer)
    }

    fn attn_k_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.inner_source().attn_k_norm_weight(layer)
    }

    fn conv_weight(&self, layer: usize) -> Option<&[f32]> {
        self.inner_source().conv_weight(layer)
    }

    fn attn_q_bias(&self, layer: usize) -> Option<&[f32]> {
        self.inner_source().attn_q_bias(layer)
    }

    fn attn_k_bias(&self, layer: usize) -> Option<&[f32]> {
        self.inner_source().attn_k_bias(layer)
    }

    fn attn_v_bias(&self, layer: usize) -> Option<&[f32]> {
        self.inner_source().attn_v_bias(layer)
    }

    fn rope_freqs(&self) -> Option<&[f32]> {
        self.inner_source().rope_freqs()
    }

    fn yarn_rope_freq_factors(&self) -> Option<&[f32]> {
        self.inner_source().yarn_rope_freq_factors()
    }

    fn attn_scale_multiplier(&self) -> f32 {
        self.inner_source().attn_scale_multiplier()
    }

    fn attn_post_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.inner_source().attn_post_norm_weight(layer)
    }

    fn ffn_post_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.inner_source().ffn_post_norm_weight(layer)
    }

    fn activation(&self) -> FfnActivation {
        self.inner_source().activation()
    }

    fn attn_logit_softcapping(&self) -> Option<f32> {
        self.inner_source().attn_logit_softcapping()
    }

    fn final_logit_softcapping(&self) -> Option<f32> {
        self.inner_source().final_logit_softcapping()
    }

    fn gpu_unsupported_reason(&self) -> Option<String> {
        self.inner_source().gpu_unsupported_reason()
    }

    fn gpu_context_cap(&self) -> Option<usize> {
        self.inner_source().gpu_context_cap()
    }

    fn embedding_mmap_weight(&self, _gguf_arc: &Arc<GgufFile>) -> anyhow::Result<MmapWeight> {
        self.read_mmap_weight("token_embd.weight")
    }

    fn output_mmap_weight(&self, _gguf_arc: &Arc<GgufFile>) -> anyhow::Result<MmapWeight> {
        self.read_mmap_weight("output.weight")
    }

    fn weight_bytes(&self, wref: &WeightRef) -> anyhow::Result<std::borrow::Cow<'_, [u8]>> {
        let mut buf = vec![0u8; wref.size];
        self.sync.read_exact(wref.start, &mut buf)?;
        Ok(std::borrow::Cow::Owned(buf))
    }

    fn dequantize_weight(&self, wref: &WeightRef) -> anyhow::Result<Vec<f32>> {
        let mut out = vec![0.0f32; wref.m * wref.k];
        let block_size = wref.dtype.block_size().max(1);
        let blocks_per_row = wref.k.div_ceil(block_size);
        let row_bytes = blocks_per_row * wref.dtype.block_bytes();
        let mut row_buf = vec![0u8; row_bytes];
        for row in 0..wref.m {
            let start = row * row_bytes;
            let end = (row + 1) * row_bytes;
            anyhow::ensure!(
                end <= wref.size,
                "weight at offset {} ({} bytes) is shorter than row {row} end {end}",
                wref.start,
                wref.size,
            );
            let abs_start = wref
                .start
                .checked_add(start as u64)
                .ok_or_else(|| anyhow::anyhow!("weight row {row} offset overflow"))?;
            self.sync.read_exact(abs_start, &mut row_buf)?;
            let row_out = &mut out[row * wref.k..(row + 1) * wref.k];
            cera::model::transformer::dequantize_row_slice(wref.dtype, &row_buf, row_out);
        }
        Ok(out)
    }

    fn output_ref(&self) -> Option<&WeightRef> {
        self.inner_source().output_ref()
    }

    fn ffn_gate_ref(&self, layer: usize) -> anyhow::Result<&WeightRef> {
        self.inner_source().ffn_gate_ref(layer)
    }

    fn ffn_up_ref(&self, layer: usize) -> anyhow::Result<&WeightRef> {
        self.inner_source().ffn_up_ref(layer)
    }

    fn ffn_down_ref(&self, layer: usize) -> anyhow::Result<&WeightRef> {
        self.inner_source().ffn_down_ref(layer)
    }

    fn moe_refs(&self, layer: usize) -> Option<&MoeFfnRefs> {
        self.inner_source().moe_refs(layer)
    }

    fn conv_in_proj_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.inner_source().conv_in_proj_ref(layer)
    }

    fn conv_out_proj_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.inner_source().conv_out_proj_ref(layer)
    }

    fn attn_q_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.inner_source().attn_q_ref(layer)
    }

    fn attn_k_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.inner_source().attn_k_ref(layer)
    }

    fn attn_v_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.inner_source().attn_v_ref(layer)
    }

    fn attn_output_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.inner_source().attn_output_ref(layer)
    }

    fn rope_type(&self) -> RopeType {
        self.inner_source().rope_type()
    }

    fn supports_batched_prefill(&self) -> bool {
        self.inner_source().supports_batched_prefill()
    }

    fn loop_norm_interval(&self) -> Option<usize> {
        self.inner_source().loop_norm_interval()
    }
}
