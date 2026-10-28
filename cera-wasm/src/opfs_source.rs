use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;
use wasm_bindgen::JsError;
use web_sys::{FileSystemReadWriteOptions, FileSystemSyncAccessHandle};

use cera::gguf::{GgufFile, GgufValue};
use cera::model::gpu_weight_source::{GpuWeightSource, RopeType};
use cera::model::lfm2::{FfnRefs, LayerWeightRefs, LfmModel, MoeFfnRefs};
use cera::model::llama::LlamaModel;
use cera::model::transformer::{self, FfnActivation, WeightRef};
use cera::model::weights::MmapWeight;
use cera::model::{BlockType, ModelConfig};
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

struct OpfsLfm3Attn {
    gate: WeightRef,
    window: Option<usize>,
    rope_theta: f32,
    has_rope_freqs: bool,
}

struct OpfsLfm3Layer {
    post_operator_norm: Option<Vec<f32>>,
    post_ffn_norm: Option<Vec<f32>>,
    attn: Option<OpfsLfm3Attn>,
}

/// Header-only OPFS weight source for the `lfm3` architecture.
///
/// `LfmModel::from_gguf_no_repack_with_loader` validates `lfm3` 1D tensors and
/// loads `lfm3` extras (`rope_freqs.weight`, `post_attention_norm.weight`,
/// `post_ffw_norm.weight`) through `gguf.get_tensor(...)`, which requires tensor
/// payload bytes to be resident in `gguf.data`. On an OPFS header-only `GgufFile`,
/// this source resolves those small host vectors through `load_f32` instead while
/// enforcing the same shape and metadata invariants.
struct OpfsLfm3Source {
    gguf: GgufFile,
    config: ModelConfig,
    output_norm_weight: Vec<f32>,
    attn_norm_weights: Vec<Vec<f32>>,
    ffn_norm_weights: Vec<Vec<f32>>,
    attn_q_norm_weights: Vec<Option<Vec<f32>>>,
    attn_k_norm_weights: Vec<Option<Vec<f32>>>,
    conv_weights_transposed: Vec<Option<Vec<f32>>>,
    output_ref: Option<WeightRef>,
    layer_refs: Vec<LayerWeightRefs>,
    rope_freq_factors: Option<Vec<f32>>,
    layers: Vec<OpfsLfm3Layer>,
}

fn lfm3_u32(gguf: &GgufFile, key: &str) -> anyhow::Result<Option<u32>> {
    match gguf.metadata.get(key) {
        None => Ok(None),
        Some(GgufValue::U32(v)) => Ok(Some(*v)),
        Some(_) => anyhow::bail!("{key} must be stored as a u32"),
    }
}

fn lfm3_f32(gguf: &GgufFile, key: &str) -> anyhow::Result<Option<f32>> {
    match gguf.metadata.get(key) {
        None => Ok(None),
        Some(GgufValue::F32(v)) => Ok(Some(*v)),
        Some(_) => anyhow::bail!("{key} must be stored as an f32"),
    }
}

impl OpfsLfm3Source {
    fn from_gguf_with_loader(
        gguf: GgufFile,
        context_size: usize,
        mut load_f32: impl FnMut(&GgufFile, &str) -> anyhow::Result<Vec<f32>>,
    ) -> anyhow::Result<Self> {
        let config = LfmModel::parse_config(&gguf, context_size)?;
        let n_layers = config.n_layers;
        let hs = config.hidden_size;
        let ff = config.intermediate_size;
        let head_dim = config.head_dim;

        let matrix = |name: &str, rows: usize, cols: usize| -> anyhow::Result<WeightRef> {
            let w = transformer::resolve_weight(&gguf, name)?;
            anyhow::ensure!(
                w.m == rows && w.k == cols,
                "{name} is {}x{}, expected {rows}x{cols}",
                w.m,
                w.k
            );
            Ok(w)
        };

        let embd_ref = matrix("token_embd.weight", config.vocab_size, hs)?;
        let output_norm_weight = load_f32(&gguf, "output_norm.weight")?;
        anyhow::ensure!(
            output_norm_weight.len() == hs,
            "output_norm.weight has {} entries, expected {hs}",
            output_norm_weight.len()
        );

        let swa_pattern = match gguf.get_bool_array("lfm3.attention.sliding_window_pattern") {
            Some(pattern) => pattern,
            None => {
                anyhow::ensure!(
                    lfm3_u32(&gguf, "lfm3.attention.sliding_window")?.is_none(),
                    "lfm3.attention.sliding_window is set but \
                     lfm3.attention.sliding_window_pattern is missing or not a bool array"
                );
                vec![false; n_layers]
            }
        };
        anyhow::ensure!(
            swa_pattern.len() >= n_layers,
            "lfm3.attention.sliding_window_pattern has {} entries, block_count is {n_layers}",
            swa_pattern.len()
        );
        let window = if swa_pattern.iter().take(n_layers).any(|&swa| swa) {
            let w = lfm3_u32(&gguf, "lfm3.attention.sliding_window")?
                .context("missing lfm3.attention.sliding_window")? as usize;
            anyhow::ensure!(w > 0, "lfm3.attention.sliding_window must be > 0");
            Some(w)
        } else {
            None
        };
        anyhow::ensure!(
            config.rope_theta.is_finite() && config.rope_theta > 1.0,
            "lfm3.rope.freq_base must be finite and > 1"
        );
        let theta_swa = lfm3_f32(&gguf, "lfm3.rope.freq_base_swa")?.unwrap_or(10_000.0);
        anyhow::ensure!(
            theta_swa.is_finite() && theta_swa > 1.0,
            "lfm3.rope.freq_base_swa must be finite and > 1"
        );

        let global_layers =
            (0..n_layers).any(|i| config.block_types[i] == BlockType::Attention && !swa_pattern[i]);
        let rope_freq_factors = if global_layers {
            let ff_vec = load_f32(&gguf, "rope_freqs.weight")
                .context("lfm3 global attention layers need rope_freqs.weight")?;
            anyhow::ensure!(
                ff_vec.len() == head_dim / 2,
                "rope_freqs.weight has {} entries, expected {}",
                ff_vec.len(),
                head_dim / 2
            );
            anyhow::ensure!(
                ff_vec.iter().all(|f| f.is_finite() && *f >= 1e-9),
                "rope_freqs.weight holds a non-finite factor or one below 1e-9 (position / factor \
                 would overflow); an F16 table overflows the 1e30 mask factor to infinity, so \
                 store it as F32"
            );
            Some(ff_vec)
        } else {
            None
        };

        let mut attn_norm_weights = Vec::with_capacity(n_layers);
        let mut ffn_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_q_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_k_norm_weights = Vec::with_capacity(n_layers);
        let mut conv_weights_transposed = Vec::with_capacity(n_layers);
        let mut layers = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            let attn_norm_name = format!("blk.{i}.attn_norm.weight");
            let attn_norm = load_f32(&gguf, &attn_norm_name)?;
            anyhow::ensure!(
                attn_norm.len() == hs,
                "{attn_norm_name} has {} entries, expected {hs}",
                attn_norm.len()
            );
            attn_norm_weights.push(attn_norm);

            let ffn_norm_name = format!("blk.{i}.ffn_norm.weight");
            let ffn_norm = load_f32(&gguf, &ffn_norm_name)?;
            anyhow::ensure!(
                ffn_norm.len() == hs,
                "{ffn_norm_name} has {} entries, expected {hs}",
                ffn_norm.len()
            );
            ffn_norm_weights.push(ffn_norm);

            let post_attn_name = format!("blk.{i}.post_attention_norm.weight");
            let post_ffn_name = format!("blk.{i}.post_ffw_norm.weight");
            let has_post_attn = gguf.tensors.contains_key(&post_attn_name);
            let has_post_ffn = gguf.tensors.contains_key(&post_ffn_name);
            anyhow::ensure!(
                has_post_attn == has_post_ffn,
                "layer {i}: post_attention_norm.weight and post_ffw_norm.weight must both be present or both absent"
            );
            let (post_operator_norm, post_ffn_norm) = if has_post_attn {
                let pa = load_f32(&gguf, &post_attn_name)?;
                anyhow::ensure!(
                    pa.len() == hs,
                    "{post_attn_name} has {} entries, expected {hs}",
                    pa.len()
                );
                let pf = load_f32(&gguf, &post_ffn_name)?;
                anyhow::ensure!(
                    pf.len() == hs,
                    "{post_ffn_name} has {} entries, expected {hs}",
                    pf.len()
                );
                (Some(pa), Some(pf))
            } else {
                (None, None)
            };

            let is_moe = config
                .moe
                .as_ref()
                .and_then(|m| m.is_moe_layer.get(i).copied())
                .unwrap_or(false);
            if is_moe {
                let moe = config.moe.as_ref().unwrap();
                matrix(&format!("blk.{i}.ffn_gate_inp.weight"), moe.n_expert, hs)?;
                for (name, rows, cols) in [
                    ("ffn_gate_exps", moe.expert_ff_len, hs),
                    ("ffn_up_exps", moe.expert_ff_len, hs),
                    ("ffn_down_exps", hs, moe.expert_ff_len),
                ] {
                    let full_name = format!("blk.{i}.{name}.weight");
                    for e in 0..moe.n_expert {
                        let (_, _, m, k, _) = gguf.tensor_meta_expert_u64(&full_name, e)?;
                        anyhow::ensure!(
                            m == rows && k == cols,
                            "{full_name} expert {e} is {m}x{k}, expected {rows}x{cols}"
                        );
                    }
                }
            } else {
                for (name, rows, cols) in [
                    ("ffn_gate", ff, hs),
                    ("ffn_up", ff, hs),
                    ("ffn_down", hs, ff),
                ] {
                    matrix(&format!("blk.{i}.{name}.weight"), rows, cols)?;
                }
            }

            if config.block_types[i] != BlockType::Attention {
                for (name, rows) in [("in_proj", 3 * hs), ("out_proj", hs)] {
                    matrix(&format!("blk.{i}.shortconv.{name}.weight"), rows, hs)?;
                }
                let conv_w = load_f32(&gguf, &format!("blk.{i}.shortconv.conv.weight"))?;
                let kernel_size = config.conv_kernel_size.unwrap_or(3);
                anyhow::ensure!(
                    conv_w.len() == hs * kernel_size,
                    "blk.{i}.shortconv.conv.weight size {} != hidden_size * conv_kernel_size ({hs} * {kernel_size})",
                    conv_w.len()
                );
                attn_q_norm_weights.push(None);
                attn_k_norm_weights.push(None);
                conv_weights_transposed.push(Some(conv_w));
                layers.push(OpfsLfm3Layer {
                    post_operator_norm,
                    post_ffn_norm,
                    attn: None,
                });
            } else {
                let q_norm_name = format!("blk.{i}.attn_q_norm.weight");
                let q_norm = load_f32(&gguf, &q_norm_name)?;
                anyhow::ensure!(
                    q_norm.len() == head_dim,
                    "{q_norm_name} has {} entries, expected {head_dim}",
                    q_norm.len()
                );
                let k_norm_name = format!("blk.{i}.attn_k_norm.weight");
                let k_norm = load_f32(&gguf, &k_norm_name)?;
                anyhow::ensure!(
                    k_norm.len() == head_dim,
                    "{k_norm_name} has {} entries, expected {head_dim}",
                    k_norm.len()
                );
                let q_dim = config.n_heads * head_dim;
                let kv_dim = config.kv_heads_per_layer[i] * head_dim;
                let gate = matrix(&format!("blk.{i}.attn_gate.weight"), config.n_heads, hs)?;
                for (name, rows, cols) in [
                    ("attn_q", q_dim, hs),
                    ("attn_k", kv_dim, hs),
                    ("attn_v", kv_dim, hs),
                    ("attn_output", hs, q_dim),
                ] {
                    matrix(&format!("blk.{i}.{name}.weight"), rows, cols)?;
                }
                let is_swa = swa_pattern[i];
                attn_q_norm_weights.push(Some(q_norm));
                attn_k_norm_weights.push(Some(k_norm));
                conv_weights_transposed.push(None);
                layers.push(OpfsLfm3Layer {
                    post_operator_norm,
                    post_ffn_norm,
                    attn: Some(OpfsLfm3Attn {
                        gate,
                        window: window.filter(|_| is_swa),
                        rope_theta: if is_swa { theta_swa } else { config.rope_theta },
                        has_rope_freqs: !is_swa && rope_freq_factors.is_some(),
                    }),
                });
            }
        }

        let output_ref = if gguf.tensors.contains_key("output.weight") {
            let output = transformer::resolve_weight(&gguf, "output.weight")?;
            anyhow::ensure!(
                output.m == embd_ref.m && output.k == embd_ref.k,
                "output.weight is {}x{} but token_embd.weight is {}x{}",
                output.m,
                output.k,
                embd_ref.m,
                embd_ref.k
            );
            Some(output)
        } else {
            None
        };

        let layer_refs =
            LfmModel::resolve_all_layer_refs_with_loader(&gguf, &config, false, &mut load_f32)?;

        Ok(Self {
            gguf,
            config,
            output_norm_weight,
            attn_norm_weights,
            ffn_norm_weights,
            attn_q_norm_weights,
            attn_k_norm_weights,
            conv_weights_transposed,
            output_ref,
            layer_refs,
            rope_freq_factors,
            layers,
        })
    }
}

impl GpuWeightSource for OpfsLfm3Source {
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

    fn conv_weight(&self, layer: usize) -> Option<&[f32]> {
        self.conv_weights_transposed[layer].as_deref()
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
        self.rope_freq_factors.as_deref()
    }

    fn attn_post_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.layers[layer].post_operator_norm.as_deref()
    }

    fn ffn_post_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.layers[layer].post_ffn_norm.as_deref()
    }

    fn weight_bytes(&self, _wref: &WeightRef) -> anyhow::Result<std::borrow::Cow<'_, [u8]>> {
        anyhow::bail!("OpfsLfm3Source::weight_bytes must be called on OpfsGpuWeightSource")
    }

    fn dequantize_weight(&self, _wref: &WeightRef) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("OpfsLfm3Source::dequantize_weight must be called on OpfsGpuWeightSource")
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
            FfnRefs::Dense(_) => None,
            FfnRefs::Moe(m) => Some(m),
        }
    }

    fn conv_in_proj_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].shortconv_in_proj.as_ref()
    }

    fn conv_out_proj_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].shortconv_out_proj.as_ref()
    }

    fn attn_q_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].attn_q.as_ref()
    }

    fn attn_k_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].attn_k.as_ref()
    }

    fn attn_v_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].attn_v.as_ref()
    }

    fn attn_output_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layer_refs[layer].attn_output.as_ref()
    }

    fn attn_gate_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.layers[layer].attn.as_ref().map(|a| &a.gate)
    }

    fn layer_rope_theta(&self, layer: usize) -> f32 {
        self.layers[layer]
            .attn
            .as_ref()
            .map(|a| a.rope_theta)
            .unwrap_or(self.config.rope_theta)
    }

    fn layer_has_rope_freqs(&self, layer: usize) -> bool {
        self.layers[layer]
            .attn
            .as_ref()
            .is_some_and(|a| a.has_rope_freqs)
    }

    fn layer_sliding_window(&self, layer: usize) -> Option<usize> {
        self.layers[layer].attn.as_ref().and_then(|a| a.window)
    }

    fn layer_is_non_causal(&self, layer: usize) -> bool {
        if self.config.is_causal {
            return false;
        }
        self.layers[layer]
            .attn
            .as_ref()
            .is_some_and(|a| a.window.is_none())
    }

    fn gpu_unsupported_reason(&self) -> Option<String> {
        if !self.config.is_causal && self.config.moe.is_some() {
            return Some(format!(
                "the GPU backends run {} with causal attention only, and this model is bidirectional (CPU only)",
                self.config.architecture
            ));
        }
        if let Some(k) = self.config.conv_kernel_size
            && k > 8
        {
            return Some(format!(
                "the GPU backends support shortconv.l_cache in 2..=8, got {k}"
            ));
        }
        None
    }

    fn rope_type(&self) -> RopeType {
        RopeType::Neox
    }

    fn supports_batched_prefill(&self) -> bool {
        true
    }
}

enum OpfsArchSource {
    Lfm2(Box<LfmModel>),
    Lfm3(Box<OpfsLfm3Source>),
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
    ///
    /// On any failure the exclusive handle is closed before the error is returned, so a
    /// retry can open the file again instead of hitting `NoModificationAllowedError`.
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
            "lfm3" => {
                let lfm3 = OpfsLfm3Source::from_gguf_with_loader(
                    (*gguf).clone(),
                    context_size,
                    read_tensor_f32,
                )
                .map_err(|e| {
                    JsError::new(&format!("loading {arch_str} config from OPFS: {e:#}"))
                })?;
                if let Some(why) = GpuWeightSource::gpu_unsupported_reason(&lfm3) {
                    return Err(JsError::new(&why));
                }
                if !GpuWeightSource::config(&lfm3).is_causal {
                    return Err(JsError::new(
                        "WebGpuSession runs causal generation only, and this model is bidirectional (CPU only)",
                    ));
                }
                OpfsArchSource::Lfm3(Box::new(lfm3))
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
            OpfsArchSource::Lfm3(m) => m.as_ref(),
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

    fn attn_gate_ref(&self, layer: usize) -> Option<&WeightRef> {
        self.inner_source().attn_gate_ref(layer)
    }

    fn layer_rope_theta(&self, layer: usize) -> f32 {
        self.inner_source().layer_rope_theta(layer)
    }

    fn layer_has_rope_freqs(&self, layer: usize) -> bool {
        self.inner_source().layer_has_rope_freqs(layer)
    }

    fn layer_sliding_window(&self, layer: usize) -> Option<usize> {
        self.inner_source().layer_sliding_window(layer)
    }

    fn layer_is_non_causal(&self, layer: usize) -> bool {
        self.inner_source().layer_is_non_causal(layer)
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
