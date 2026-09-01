use std::cell::RefCell;
use std::sync::Arc;

use wasm_bindgen::JsError;
use web_sys::{FileSystemReadWriteOptions, FileSystemSyncAccessHandle};

use cera::gguf::GgufFile;
use cera::model::gpu_weight_source::{GpuWeightSource, RopeType};
use cera::model::lfm2::{LayerWeightRefs, Lfm2Model, MoeFfnRefs};
use cera::model::transformer::WeightRef;
use cera::model::{BlockType, ModelConfig};
use cera::tensor::{DType, Tensor};

/// An on-demand weight source that streams tensor data directly from browser OPFS storage
/// (`FileSystemSyncAccessHandle`) into WebGPU buffers.
///
/// Memory footprint: only keeps the lightweight GGUF metadata header (~1 MB) and small
/// RMSNorm/conv vectors (< 500 KB) resident in WASM linear memory. Projection matrix bytes
/// are read on-demand into a single reusable chunk buffer directly during GPU upload,
/// completely avoiding the 32-bit WASM memory limit.
pub struct OpfsGpuWeightSource {
    config: ModelConfig,
    gguf: Arc<GgufFile>,
    sync: FileSystemSyncAccessHandle,
    output_norm_weight: Vec<f32>,
    attn_norm_weights: Vec<Vec<f32>>,
    ffn_norm_weights: Vec<Vec<f32>>,
    attn_q_norm_weights: Vec<Option<Vec<f32>>>,
    attn_k_norm_weights: Vec<Option<Vec<f32>>>,
    conv_weights: Vec<Option<Vec<f32>>>,
    layer_refs: Vec<LayerWeightRefs>,
    output_ref: Option<WeightRef>,
    chunk_buf: RefCell<Vec<u8>>,
    closed: std::sync::atomic::AtomicBool,
}

impl OpfsGpuWeightSource {
    /// Create a new OPFS streaming weight source from an open `FileSystemSyncAccessHandle`
    /// and parsed header-only `GgufFile`.
    pub fn new(
        sync: FileSystemSyncAccessHandle,
        gguf: Arc<GgufFile>,
        context_size: usize,
    ) -> Result<Self, JsError> {
        let config = Lfm2Model::parse_config(&gguf, context_size)
            .map_err(|e| JsError::new(&format!("parsing LFM2 config: {e:#}")))?;
        let layer_refs = Lfm2Model::resolve_all_layer_refs(&gguf, &config)
            .map_err(|e| JsError::new(&format!("resolving layer refs: {e:#}")))?;
        let output_ref = cera::model::transformer::resolve_weight(&gguf, "output.weight").ok();

        // Helper to read a raw tensor into a byte buffer from OPFS
        let read_tensor_bytes = |info: &cera::gguf::TensorInfo| -> Result<Vec<u8>, JsError> {
            let mut buf = vec![0u8; info.size_bytes];
            let opts = FileSystemReadWriteOptions::new();
            opts.set_at(info.offset as f64);
            let read = sync
                .read_with_u8_array_and_options(&mut buf, &opts)
                .map_err(|e| {
                    JsError::new(&format!(
                        "reading tensor {} at offset {} from OPFS: {:?}",
                        info.name, info.offset, e
                    ))
                })?;
            if (read as usize) != info.size_bytes {
                return Err(JsError::new(&format!(
                    "short read for tensor {}: read {} of {} bytes",
                    info.name, read, info.size_bytes
                )));
            }
            Ok(buf)
        };

        // Helper to read and convert an F32/F16/quant tensor to Vec<f32>
        let read_tensor_f32 = |name: &str| -> Result<Vec<f32>, JsError> {
            let info = gguf
                .tensors
                .get(name)
                .ok_or_else(|| JsError::new(&format!("tensor not found: {name}")))?;
            let bytes = read_tensor_bytes(info)?;
            match info.dtype {
                DType::F32 => {
                    if !bytes.len().is_multiple_of(4) {
                        return Err(JsError::new(&format!(
                            "tensor {name} length {} is not a multiple of 4 for F32",
                            bytes.len()
                        )));
                    }
                    Ok(bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
                        .collect())
                }
                DType::F16 => {
                    if !bytes.len().is_multiple_of(2) {
                        return Err(JsError::new(&format!(
                            "tensor {name} length {} is not a multiple of 2 for F16",
                            bytes.len()
                        )));
                    }
                    Ok(bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| cera::quant::f16_to_f32(u16::from_le_bytes(*c)))
                        .collect())
                }
                DType::BF16 => {
                    if !bytes.len().is_multiple_of(2) {
                        return Err(JsError::new(&format!(
                            "tensor {name} length {} is not a multiple of 2 for BF16",
                            bytes.len()
                        )));
                    }
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

        // Extract small norm and conv weights (< 500 KB total)
        let output_norm_weight = read_tensor_f32("token_embd_norm.weight")
            .or_else(|_| read_tensor_f32("output_norm.weight"))?;

        let mut attn_norm_weights = Vec::with_capacity(config.n_layers);
        let mut ffn_norm_weights = Vec::with_capacity(config.n_layers);
        let mut attn_q_norm_weights = Vec::with_capacity(config.n_layers);
        let mut attn_k_norm_weights = Vec::with_capacity(config.n_layers);
        let mut conv_weights = Vec::with_capacity(config.n_layers);

        for (i, bt) in config.block_types.iter().enumerate() {
            attn_norm_weights.push(read_tensor_f32(&format!("blk.{i}.attn_norm.weight"))?);
            ffn_norm_weights.push(read_tensor_f32(&format!("blk.{i}.ffn_norm.weight"))?);

            if *bt == BlockType::Attention {
                let q_norm = if gguf
                    .tensors
                    .contains_key(&format!("blk.{i}.attn_q_norm.weight"))
                {
                    Some(read_tensor_f32(&format!("blk.{i}.attn_q_norm.weight"))?)
                } else {
                    None
                };
                let k_norm = if gguf
                    .tensors
                    .contains_key(&format!("blk.{i}.attn_k_norm.weight"))
                {
                    Some(read_tensor_f32(&format!("blk.{i}.attn_k_norm.weight"))?)
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
                    let w = read_tensor_f32(&conv_name)?;
                    conv_weights.push(Some(w));
                } else {
                    conv_weights.push(None);
                }
            }
        }

        Ok(Self {
            config,
            gguf,
            sync,
            output_norm_weight,
            attn_norm_weights,
            ffn_norm_weights,
            attn_q_norm_weights,
            attn_k_norm_weights,
            conv_weights,
            layer_refs,
            output_ref,
            chunk_buf: RefCell::new(Vec::with_capacity(16 * 1024 * 1024)),
            closed: std::sync::atomic::AtomicBool::new(false),
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

    /// Close the underlying OPFS sync access handle, releasing its exclusive file lock.
    pub fn close(&self) {
        if !self.closed.swap(true, std::sync::atomic::Ordering::Relaxed) {
            self.sync.close();
        }
    }
}

impl Drop for OpfsGpuWeightSource {
    fn drop(&mut self) {
        self.close();
    }
}

impl GpuWeightSource for OpfsGpuWeightSource {
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
        self.attn_q_norm_weights
            .get(layer)
            .and_then(|o| o.as_deref())
    }

    fn attn_k_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.attn_k_norm_weights
            .get(layer)
            .and_then(|o| o.as_deref())
    }

    fn conv_weight(&self, layer: usize) -> Option<&[f32]> {
        self.conv_weights.get(layer).and_then(|o| o.as_deref())
    }

    fn attn_q_bias(&self, _layer: usize) -> Option<&[f32]> {
        None
    }

    fn attn_k_bias(&self, _layer: usize) -> Option<&[f32]> {
        None
    }

    fn attn_v_bias(&self, _layer: usize) -> Option<&[f32]> {
        None
    }

    fn rope_freqs(&self) -> Option<&[f32]> {
        None
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
        let block_size = wref.dtype.block_size();
        let row_bytes = wref.k / block_size * wref.dtype.block_bytes();
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
        self.output_ref.as_ref()
    }

    fn ffn_gate_ref(&self, layer: usize) -> anyhow::Result<&WeightRef> {
        Ok(&self.layer_refs[layer].ffn.dense()?.gate)
    }

    fn ffn_up_ref(&self, layer: usize) -> anyhow::Result<&WeightRef> {
        Ok(&self.layer_refs[layer].ffn.dense()?.up)
    }

    fn ffn_down_ref(&self, layer: usize) -> anyhow::Result<&WeightRef> {
        Ok(&self.layer_refs[layer].ffn.dense()?.down)
    }

    fn moe_refs(&self, layer: usize) -> Option<&MoeFfnRefs> {
        match &self.layer_refs[layer].ffn {
            cera::model::lfm2::FfnRefs::Moe(m) => Some(m),
            _ => None,
        }
    }

    fn conv_in_proj_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs
            .get(layer)
            .and_then(|l| l.shortconv_in_proj.as_ref())
    }

    fn conv_out_proj_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs
            .get(layer)
            .and_then(|l| l.shortconv_out_proj.as_ref())
    }

    fn attn_q_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs.get(layer).and_then(|l| l.attn_q.as_ref())
    }

    fn attn_k_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs.get(layer).and_then(|l| l.attn_k.as_ref())
    }

    fn attn_v_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs.get(layer).and_then(|l| l.attn_v.as_ref())
    }

    fn attn_output_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs
            .get(layer)
            .and_then(|l| l.attn_output.as_ref())
    }

    fn rope_type(&self) -> RopeType {
        RopeType::Neox
    }

    fn supports_batched_prefill(&self) -> bool {
        true
    }
}
