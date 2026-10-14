use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use wasm_bindgen::JsError;
use web_sys::{FileSystemReadWriteOptions, FileSystemSyncAccessHandle};

use cera::gguf::GgufFile;
use cera::model::gpu_weight_source::{GpuWeightSource, RopeType};
use cera::model::lfm2::{LayerWeightRefs, LfmModel, MoeFfnRefs};
use cera::model::llama::LlamaModel;
use cera::model::transformer::{FfnActivation, WeightRef};
use cera::model::weights::MmapWeight;
use cera::model::{BlockType, ModelConfig};
use cera::tensor::{DType, Tensor};

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

#[allow(clippy::large_enum_variant)]
enum OpfsArchSource {
    Lfm2 {
        config: ModelConfig,
        output_norm_weight: Vec<f32>,
        attn_norm_weights: Vec<Vec<f32>>,
        ffn_norm_weights: Vec<Vec<f32>>,
        attn_q_norm_weights: Vec<Option<Vec<f32>>>,
        attn_k_norm_weights: Vec<Option<Vec<f32>>>,
        conv_weights: Vec<Option<Vec<f32>>>,
        layer_refs: Vec<LayerWeightRefs>,
        output_ref: Option<WeightRef>,
    },
    Llama(Box<LlamaModel>),
}

/// An on-demand weight source that streams tensor data directly from browser OPFS storage
/// (`FileSystemSyncAccessHandle`) into WebGPU buffers.
///
/// Memory footprint: only keeps the lightweight GGUF metadata header (~1 MB) and small
/// RMSNorm/conv/bias vectors (< 500 KB) resident in WASM linear memory. Projection matrix
/// bytes are read on-demand into a single reusable chunk buffer directly during GPU upload,
/// completely avoiding the 32-bit WASM memory limit.
pub struct OpfsGpuWeightSource {
    gguf: Arc<GgufFile>,
    sync: OpfsSyncHandle,
    arch: OpfsArchSource,
    chunk_buf: RefCell<Vec<u8>>,
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
        let read_tensor_bytes = |info: &cera::gguf::TensorInfo| -> anyhow::Result<Vec<u8>> {
            let mut buf = vec![0u8; info.size_bytes];
            let opts = FileSystemReadWriteOptions::new();
            opts.set_at(info.offset as f64);
            let read = sync
                .as_raw()
                .read_with_u8_array_and_options(&mut buf, &opts)
                .map_err(|e| {
                    anyhow::anyhow!(
                        "reading tensor {} at offset {} from OPFS: {:?}",
                        info.name,
                        info.offset,
                        e
                    )
                })?;
            anyhow::ensure!(
                (read as usize) == info.size_bytes,
                "short read for tensor {}: read {} of {} bytes",
                info.name,
                read,
                info.size_bytes
            );
            Ok(buf)
        };

        let read_tensor_f32 = |g: &GgufFile, name: &str| -> anyhow::Result<Vec<f32>> {
            let info = g
                .tensors
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("tensor not found: {name}"))?;
            let bytes = read_tensor_bytes(info)?;
            match info.dtype {
                DType::F32 => {
                    anyhow::ensure!(
                        bytes.len().is_multiple_of(4),
                        "tensor {name} length {} is not a multiple of 4 for F32",
                        bytes.len()
                    );
                    Ok(bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
                        .collect())
                }
                DType::F16 => {
                    anyhow::ensure!(
                        bytes.len().is_multiple_of(2),
                        "tensor {name} length {} is not a multiple of 2 for F16",
                        bytes.len()
                    );
                    Ok(bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| cera::quant::f16_to_f32(u16::from_le_bytes(*c)))
                        .collect())
                }
                DType::BF16 => {
                    anyhow::ensure!(
                        bytes.len().is_multiple_of(2),
                        "tensor {name} length {} is not a multiple of 2 for BF16",
                        bytes.len()
                    );
                    Ok(bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| cera::quant::bf16_to_f32(u16::from_le_bytes(*c)))
                        .collect())
                }
                dt => {
                    let numel: usize = info.shape.iter().product();
                    let mut out = vec![0.0f32; numel];
                    cera::model::transformer::dequantize_row_slice(dt, &bytes, &mut out);
                    Ok(out)
                }
            }
        };

        let arch_str = gguf
            .get_str("general.architecture")
            .unwrap_or("lfm2")
            .to_ascii_lowercase();

        let arch = if matches!(arch_str.as_str(), "lfm2" | "lfm2moe") {
            let config = LfmModel::parse_config(&gguf, context_size)
                .map_err(|e| JsError::new(&format!("parsing LFM2 config: {e:#}")))?;
            // No CPU repacks: this source only resolves metadata for GPU upload
            // (see `with_repack_if`), streaming small 1D MoE biases via `read_tensor_f32`.
            let layer_refs = LfmModel::resolve_all_layer_refs_with_loader(
                &gguf,
                &config,
                false,
                read_tensor_f32,
            )
            .map_err(|e| JsError::new(&format!("resolving layer refs: {e:#}")))?;
            let output_ref = cera::model::transformer::resolve_weight(&gguf, "output.weight").ok();

            let output_norm_weight = read_tensor_f32(&gguf, "token_embd_norm.weight")
                .or_else(|_| read_tensor_f32(&gguf, "output_norm.weight"))
                .map_err(|e| JsError::new(&format!("{e:#}")))?;

            let mut attn_norm_weights = Vec::with_capacity(config.n_layers);
            let mut ffn_norm_weights = Vec::with_capacity(config.n_layers);
            let mut attn_q_norm_weights = Vec::with_capacity(config.n_layers);
            let mut attn_k_norm_weights = Vec::with_capacity(config.n_layers);
            let mut conv_weights = Vec::with_capacity(config.n_layers);

            for (i, bt) in config.block_types.iter().enumerate() {
                attn_norm_weights.push(
                    read_tensor_f32(&gguf, &format!("blk.{i}.attn_norm.weight"))
                        .map_err(|e| JsError::new(&format!("{e:#}")))?,
                );
                ffn_norm_weights.push(
                    read_tensor_f32(&gguf, &format!("blk.{i}.ffn_norm.weight"))
                        .map_err(|e| JsError::new(&format!("{e:#}")))?,
                );

                if *bt == BlockType::Attention {
                    let q_norm = if gguf
                        .tensors
                        .contains_key(&format!("blk.{i}.attn_q_norm.weight"))
                    {
                        Some(
                            read_tensor_f32(&gguf, &format!("blk.{i}.attn_q_norm.weight"))
                                .map_err(|e| JsError::new(&format!("{e:#}")))?,
                        )
                    } else {
                        None
                    };
                    let k_norm = if gguf
                        .tensors
                        .contains_key(&format!("blk.{i}.attn_k_norm.weight"))
                    {
                        Some(
                            read_tensor_f32(&gguf, &format!("blk.{i}.attn_k_norm.weight"))
                                .map_err(|e| JsError::new(&format!("{e:#}")))?,
                        )
                    } else {
                        None
                    };
                    attn_q_norm_weights.push(q_norm);
                    attn_k_norm_weights.push(k_norm);
                    conv_weights.push(None);
                } else {
                    attn_q_norm_weights.push(None);
                    attn_k_norm_weights.push(None);
                    let conv_name = format!("blk.{i}.shortconv.conv.weight");
                    if gguf.tensors.contains_key(&conv_name) {
                        let w = read_tensor_f32(&gguf, &conv_name)
                            .map_err(|e| JsError::new(&format!("{e:#}")))?;
                        conv_weights.push(Some(w));
                    } else {
                        conv_weights.push(None);
                    }
                }
            }

            OpfsArchSource::Lfm2 {
                config,
                output_norm_weight,
                attn_norm_weights,
                ffn_norm_weights,
                attn_q_norm_weights,
                attn_k_norm_weights,
                conv_weights,
                layer_refs,
                output_ref,
            }
        } else {
            let llama = LlamaModel::from_gguf_no_repack_with_loader(
                (*gguf).clone(),
                context_size,
                model_id,
                read_tensor_f32,
            )
            .map_err(|e| JsError::new(&format!("loading {arch_str} config from OPFS: {e:#}")))?;
            OpfsArchSource::Llama(Box::new(llama))
        };

        Ok(Self {
            gguf,
            sync,
            arch,
            chunk_buf: RefCell::new(Vec::with_capacity(16 * 1024 * 1024)),
        })
    }

    /// Read raw bytes for `wref` into `self.chunk_buf`
    fn read_into_chunk(&self, wref: &WeightRef) -> Result<(), JsError> {
        let mut chunk = self.chunk_buf.borrow_mut();
        if chunk.len() < wref.size {
            chunk.resize(wref.size, 0);
        }
        let opts = FileSystemReadWriteOptions::new();
        opts.set_at(wref.start as f64);
        let read = self
            .sync
            .as_raw()
            .read_with_u8_array_and_options(&mut chunk[..wref.size], &opts)
            .map_err(|e| {
                JsError::new(&format!(
                    "reading weight at offset {} from OPFS: {:?}",
                    wref.start, e
                ))
            })?;
        if (read as usize) != wref.size {
            return Err(JsError::new(&format!(
                "short read from OPFS: read {} of {} bytes",
                read, wref.size
            )));
        }
        Ok(())
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
        let mut data = vec![0u8; info.size_bytes];
        let opts = FileSystemReadWriteOptions::new();
        opts.set_at(info.offset as f64);
        let read = self
            .sync
            .as_raw()
            .read_with_u8_array_and_options(&mut data, &opts)
            .map_err(|e| {
                anyhow::anyhow!("reading {name} at offset {} from OPFS: {e:?}", info.offset)
            })?;
        anyhow::ensure!(
            (read as usize) == info.size_bytes,
            "short read for {name} from OPFS: read {read} of {} bytes",
            info.size_bytes
        );
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
        match &self.arch {
            OpfsArchSource::Lfm2 { config, .. } => config,
            OpfsArchSource::Llama(m) => GpuWeightSource::config(m.as_ref()),
        }
    }

    fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    fn output_norm_weight(&self) -> &[f32] {
        match &self.arch {
            OpfsArchSource::Lfm2 {
                output_norm_weight, ..
            } => output_norm_weight,
            OpfsArchSource::Llama(m) => GpuWeightSource::output_norm_weight(m.as_ref()),
        }
    }

    fn attn_norm_weight(&self, layer: usize) -> &[f32] {
        match &self.arch {
            OpfsArchSource::Lfm2 {
                attn_norm_weights, ..
            } => &attn_norm_weights[layer],
            OpfsArchSource::Llama(m) => GpuWeightSource::attn_norm_weight(m.as_ref(), layer),
        }
    }

    fn ffn_norm_weight(&self, layer: usize) -> &[f32] {
        match &self.arch {
            OpfsArchSource::Lfm2 {
                ffn_norm_weights, ..
            } => &ffn_norm_weights[layer],
            OpfsArchSource::Llama(m) => GpuWeightSource::ffn_norm_weight(m.as_ref(), layer),
        }
    }

    fn attn_q_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        match &self.arch {
            OpfsArchSource::Lfm2 {
                attn_q_norm_weights,
                ..
            } => attn_q_norm_weights.get(layer).and_then(|o| o.as_deref()),
            OpfsArchSource::Llama(m) => GpuWeightSource::attn_q_norm_weight(m.as_ref(), layer),
        }
    }

    fn attn_k_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        match &self.arch {
            OpfsArchSource::Lfm2 {
                attn_k_norm_weights,
                ..
            } => attn_k_norm_weights.get(layer).and_then(|o| o.as_deref()),
            OpfsArchSource::Llama(m) => GpuWeightSource::attn_k_norm_weight(m.as_ref(), layer),
        }
    }

    fn conv_weight(&self, layer: usize) -> Option<&[f32]> {
        match &self.arch {
            OpfsArchSource::Lfm2 { conv_weights, .. } => {
                conv_weights.get(layer).and_then(|o| o.as_deref())
            }
            OpfsArchSource::Llama(m) => GpuWeightSource::conv_weight(m.as_ref(), layer),
        }
    }

    fn attn_q_bias(&self, layer: usize) -> Option<&[f32]> {
        match &self.arch {
            OpfsArchSource::Lfm2 { .. } => None,
            OpfsArchSource::Llama(m) => GpuWeightSource::attn_q_bias(m.as_ref(), layer),
        }
    }

    fn attn_k_bias(&self, layer: usize) -> Option<&[f32]> {
        match &self.arch {
            OpfsArchSource::Lfm2 { .. } => None,
            OpfsArchSource::Llama(m) => GpuWeightSource::attn_k_bias(m.as_ref(), layer),
        }
    }

    fn attn_v_bias(&self, layer: usize) -> Option<&[f32]> {
        match &self.arch {
            OpfsArchSource::Lfm2 { .. } => None,
            OpfsArchSource::Llama(m) => GpuWeightSource::attn_v_bias(m.as_ref(), layer),
        }
    }

    fn rope_freqs(&self) -> Option<&[f32]> {
        match &self.arch {
            OpfsArchSource::Lfm2 { .. } => None,
            OpfsArchSource::Llama(m) => GpuWeightSource::rope_freqs(m.as_ref()),
        }
    }

    fn yarn_rope_freq_factors(&self) -> Option<&[f32]> {
        match &self.arch {
            OpfsArchSource::Lfm2 { .. } => None,
            OpfsArchSource::Llama(m) => GpuWeightSource::yarn_rope_freq_factors(m.as_ref()),
        }
    }

    fn attn_scale_multiplier(&self) -> f32 {
        match &self.arch {
            OpfsArchSource::Lfm2 { .. } => 1.0,
            OpfsArchSource::Llama(m) => GpuWeightSource::attn_scale_multiplier(m.as_ref()),
        }
    }

    fn attn_post_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        match &self.arch {
            OpfsArchSource::Lfm2 { .. } => None,
            OpfsArchSource::Llama(m) => GpuWeightSource::attn_post_norm_weight(m.as_ref(), layer),
        }
    }

    fn ffn_post_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        match &self.arch {
            OpfsArchSource::Lfm2 { .. } => None,
            OpfsArchSource::Llama(m) => GpuWeightSource::ffn_post_norm_weight(m.as_ref(), layer),
        }
    }

    fn activation(&self) -> FfnActivation {
        match &self.arch {
            OpfsArchSource::Lfm2 { .. } => FfnActivation::Swiglu,
            OpfsArchSource::Llama(m) => GpuWeightSource::activation(m.as_ref()),
        }
    }

    fn attn_logit_softcapping(&self) -> Option<f32> {
        match &self.arch {
            OpfsArchSource::Lfm2 { .. } => None,
            OpfsArchSource::Llama(m) => GpuWeightSource::attn_logit_softcapping(m.as_ref()),
        }
    }

    fn final_logit_softcapping(&self) -> Option<f32> {
        match &self.arch {
            OpfsArchSource::Lfm2 { .. } => None,
            OpfsArchSource::Llama(m) => GpuWeightSource::final_logit_softcapping(m.as_ref()),
        }
    }

    fn embedding_tensor(&self) -> anyhow::Result<Tensor> {
        let info = self
            .gguf
            .tensors
            .get("token_embd.weight")
            .ok_or_else(|| anyhow::anyhow!("token_embd.weight not found in GGUF"))?;
        let mut data = vec![0u8; info.size_bytes];
        let opts = FileSystemReadWriteOptions::new();
        opts.set_at(info.offset as f64);
        let read = self
            .sync
            .as_raw()
            .read_with_u8_array_and_options(&mut data, &opts)
            .map_err(|e| anyhow::anyhow!("reading token_embd.weight: {:?}", e))?;
        anyhow::ensure!(
            (read as usize) == info.size_bytes,
            "short read for token_embd.weight"
        );
        Ok(Tensor::new(data, info.shape.clone(), info.dtype))
    }

    fn embedding_tensor_data(&self) -> anyhow::Result<std::borrow::Cow<'_, [u8]>> {
        let info = self
            .gguf
            .tensors
            .get("token_embd.weight")
            .ok_or_else(|| anyhow::anyhow!("token_embd.weight not found in GGUF"))?;
        let wref = WeightRef::new(info.offset, info.size_bytes, info.dtype, 1, 1);
        self.read_into_chunk(&wref)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let chunk = self.chunk_buf.borrow();
        let len = info.size_bytes.min(chunk.len());
        Ok(std::borrow::Cow::Owned(chunk[..len].to_vec()))
    }

    fn embedding_mmap_weight(&self, _gguf_arc: &Arc<GgufFile>) -> anyhow::Result<MmapWeight> {
        self.read_mmap_weight("token_embd.weight")
    }

    fn output_mmap_weight(&self, _gguf_arc: &Arc<GgufFile>) -> anyhow::Result<MmapWeight> {
        self.read_mmap_weight("output.weight")
    }

    fn weight_bytes(&self, wref: &WeightRef) -> std::borrow::Cow<'_, [u8]> {
        if let Err(e) = self.read_into_chunk(wref) {
            tracing::error!("OPFS weight read failed for {wref:?}: {e:?}");
            return std::borrow::Cow::Borrowed(&[]);
        }
        let chunk = self.chunk_buf.borrow();
        let len = wref.size.min(chunk.len());
        std::borrow::Cow::Owned(chunk[..len].to_vec())
    }

    fn dequantize_weight(&self, wref: &WeightRef) -> Vec<f32> {
        if let Err(e) = self.read_into_chunk(wref) {
            tracing::error!("OPFS weight read failed for dequantize {wref:?}: {e:?}");
            return vec![0.0f32; wref.m * wref.k];
        }
        let chunk = self.chunk_buf.borrow();
        let data = &chunk[..wref.size.min(chunk.len())];
        let mut out = vec![0.0f32; wref.m * wref.k];
        let block_size = wref.dtype.block_size().max(1);
        let blocks_per_row = wref.k.div_ceil(block_size);
        let row_bytes = blocks_per_row * wref.dtype.block_bytes();
        for row in 0..wref.m {
            let start = row * row_bytes;
            let end = (row + 1) * row_bytes;
            if end <= data.len() {
                let row_data = &data[start..end];
                let row_out = &mut out[row * wref.k..(row + 1) * wref.k];
                cera::model::transformer::dequantize_row_slice(wref.dtype, row_data, row_out);
            }
        }
        out
    }

    fn output_ref(&self) -> Option<&WeightRef> {
        match &self.arch {
            OpfsArchSource::Lfm2 { output_ref, .. } => output_ref.as_ref(),
            OpfsArchSource::Llama(m) => GpuWeightSource::output_ref(m.as_ref()),
        }
    }

    fn ffn_gate_ref(&self, layer: usize) -> anyhow::Result<&WeightRef> {
        match &self.arch {
            OpfsArchSource::Lfm2 { layer_refs, .. } => Ok(&layer_refs[layer].ffn.dense()?.gate),
            OpfsArchSource::Llama(m) => GpuWeightSource::ffn_gate_ref(m.as_ref(), layer),
        }
    }

    fn ffn_up_ref(&self, layer: usize) -> anyhow::Result<&WeightRef> {
        match &self.arch {
            OpfsArchSource::Lfm2 { layer_refs, .. } => Ok(&layer_refs[layer].ffn.dense()?.up),
            OpfsArchSource::Llama(m) => GpuWeightSource::ffn_up_ref(m.as_ref(), layer),
        }
    }

    fn ffn_down_ref(&self, layer: usize) -> anyhow::Result<&WeightRef> {
        match &self.arch {
            OpfsArchSource::Lfm2 { layer_refs, .. } => Ok(&layer_refs[layer].ffn.dense()?.down),
            OpfsArchSource::Llama(m) => GpuWeightSource::ffn_down_ref(m.as_ref(), layer),
        }
    }

    fn moe_refs(&self, layer: usize) -> Option<&MoeFfnRefs> {
        match &self.arch {
            OpfsArchSource::Lfm2 { layer_refs, .. } => match &layer_refs[layer].ffn {
                cera::model::lfm2::FfnRefs::Moe(m) => Some(m),
                _ => None,
            },
            OpfsArchSource::Llama(m) => GpuWeightSource::moe_refs(m.as_ref(), layer),
        }
    }

    fn conv_in_proj_ref(&self, layer: usize) -> Option<&WeightRef> {
        match &self.arch {
            OpfsArchSource::Lfm2 { layer_refs, .. } => layer_refs
                .get(layer)
                .and_then(|l| l.shortconv_in_proj.as_ref()),
            OpfsArchSource::Llama(m) => GpuWeightSource::conv_in_proj_ref(m.as_ref(), layer),
        }
    }

    fn conv_out_proj_ref(&self, layer: usize) -> Option<&WeightRef> {
        match &self.arch {
            OpfsArchSource::Lfm2 { layer_refs, .. } => layer_refs
                .get(layer)
                .and_then(|l| l.shortconv_out_proj.as_ref()),
            OpfsArchSource::Llama(m) => GpuWeightSource::conv_out_proj_ref(m.as_ref(), layer),
        }
    }

    fn attn_q_ref(&self, layer: usize) -> Option<&WeightRef> {
        match &self.arch {
            OpfsArchSource::Lfm2 { layer_refs, .. } => {
                layer_refs.get(layer).and_then(|l| l.attn_q.as_ref())
            }
            OpfsArchSource::Llama(m) => GpuWeightSource::attn_q_ref(m.as_ref(), layer),
        }
    }

    fn attn_k_ref(&self, layer: usize) -> Option<&WeightRef> {
        match &self.arch {
            OpfsArchSource::Lfm2 { layer_refs, .. } => {
                layer_refs.get(layer).and_then(|l| l.attn_k.as_ref())
            }
            OpfsArchSource::Llama(m) => GpuWeightSource::attn_k_ref(m.as_ref(), layer),
        }
    }

    fn attn_v_ref(&self, layer: usize) -> Option<&WeightRef> {
        match &self.arch {
            OpfsArchSource::Lfm2 { layer_refs, .. } => {
                layer_refs.get(layer).and_then(|l| l.attn_v.as_ref())
            }
            OpfsArchSource::Llama(m) => GpuWeightSource::attn_v_ref(m.as_ref(), layer),
        }
    }

    fn attn_output_ref(&self, layer: usize) -> Option<&WeightRef> {
        match &self.arch {
            OpfsArchSource::Lfm2 { layer_refs, .. } => {
                layer_refs.get(layer).and_then(|l| l.attn_output.as_ref())
            }
            OpfsArchSource::Llama(m) => GpuWeightSource::attn_output_ref(m.as_ref(), layer),
        }
    }

    fn rope_type(&self) -> RopeType {
        match &self.arch {
            OpfsArchSource::Lfm2 { .. } => RopeType::Neox,
            OpfsArchSource::Llama(m) => GpuWeightSource::rope_type(m.as_ref()),
        }
    }

    fn supports_batched_prefill(&self) -> bool {
        match &self.arch {
            OpfsArchSource::Lfm2 { .. } => true,
            OpfsArchSource::Llama(m) => GpuWeightSource::supports_batched_prefill(m.as_ref()),
        }
    }

    fn loop_norm_interval(&self) -> Option<usize> {
        match &self.arch {
            OpfsArchSource::Lfm2 { .. } => None,
            OpfsArchSource::Llama(m) => GpuWeightSource::loop_norm_interval(m.as_ref()),
        }
    }
}
