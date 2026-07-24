// LFM2 / LFM2.5 hybrid conv+attention model.

use std::sync::Mutex;

use anyhow::{Context, Result, ensure};

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, KvPrefixCache, LayerState};
use crate::model::transformer::{self, FfnWeights};
use crate::model::{BlockType, Model, ModelConfig, ScalarMultipliers};
// DType's only remaining uses here are the aarch64 per-token GEMV dispatch
// (`forward_conv_block`'s Q4_0/Q8_0 checks). The prefill gates used to name dtypes
// directly too, but they now ask `transformer::batched_gemm_supports`, which takes
// the dtype as a parameter — so on x86_64 + `blas` this import became unused. Gate
// it to where it is actually referenced.
#[cfg(target_arch = "aarch64")]
use crate::tensor::DType;
use crate::turboquant;

// ── Pre-resolved weight reference ───────────────────────────────────────────

// The pre-resolved mmap weight reference is the arch-agnostic one from
// `transformer.rs`; LFM2 shares it (and the weight-plumbing helpers below) so the
// type and kernels have a single definition. `gpu_lfm2.rs` / `metal_lfm2.rs` keep
// referring to `lfm2::WeightRef` via this re-export.
pub(crate) use transformer::WeightRef;

/// Per-layer weight references for quantized tensors.
#[derive(Debug, Clone)]
pub(crate) struct LayerWeightRefs {
    pub ffn_gate: WeightRef,
    pub ffn_up: WeightRef,
    pub ffn_down: WeightRef,
    pub shortconv_in_proj: Option<WeightRef>,
    pub shortconv_out_proj: Option<WeightRef>,
    pub attn_q: Option<WeightRef>,
    pub attn_k: Option<WeightRef>,
    pub attn_v: Option<WeightRef>,
    pub attn_output: Option<WeightRef>,
}

/// Dimensions for a layer's weight matrices (for GPU model construction).
pub struct LayerWeightDims {
    pub ffn_gate_m: usize,
    pub ffn_gate_k: usize,
    pub ffn_down_m: usize,
    pub ffn_down_k: usize,
}

// ── LFM2 Model ─────────────────────────────────────────────────────────────

pub struct Lfm2Model {
    gguf: GgufFile,
    config: ModelConfig,
    // Pre-dequantized small F32 weights
    output_norm_weight: Vec<f32>,
    attn_norm_weights: Vec<Vec<f32>>,
    ffn_norm_weights: Vec<Vec<f32>>,
    attn_q_norm_weights: Vec<Option<Vec<f32>>>,
    attn_k_norm_weights: Vec<Option<Vec<f32>>>,
    conv_weights: Vec<Option<Vec<f32>>>,
    // Pre-resolved quantized weight refs
    embd_ref: WeightRef,
    layer_refs: Vec<LayerWeightRefs>,
    /// Identifier passed into `KvPrefixCache::new`. CPU prefixes the
    /// caller-supplied id with `"cpu:"` so disk-cache files don't
    /// collide with Metal's f16-byte snapshots of the same model file
    /// (model_fingerprint doesn't include element width). Empty string
    /// when constructed via `from_gguf` (path-less case): warm cache
    /// still works, but disk-cache files for different path-less
    /// from_bytes loads of distinct models would namespace-collide —
    /// acceptable since `from_bytes` is documented as "testing".
    model_id: String,
    /// Two-tier prefix cache (warm in-memory + cold on-disk via
    /// FlatBuffers). Replaced wholesale by `Model::configure_cache`.
    /// Defaults to `KvCacheConfig::default()` (warm-only) at
    /// construction time so warm hits work without explicit config.
    prefix_cache: Mutex<KvPrefixCache>,
}

impl Lfm2Model {
    /// Construct without a model identifier. Equivalent to
    /// `from_gguf_with_id(gguf, context_size, "")`. Warm prefix cache
    /// works after `Model::configure_cache`; disk cache (when
    /// configured) would namespace-collide between path-less loads of
    /// different models, which is acceptable for the `from_bytes`
    /// testing use case the doc calls out.
    pub fn from_gguf(gguf: GgufFile, context_size: usize) -> Result<Self> {
        Self::from_gguf_with_id(gguf, context_size, String::new())
    }

    /// Construct with an explicit model identifier (typically the GGUF
    /// file path) used to namespace prefix-cache entries. The id is
    /// prefixed with `"cpu:"` before being fed to `model_fingerprint`
    /// so CPU and Metal can share a `--cache-dir` without their
    /// disk-cache files (different element widths: CPU=f32, Metal=f16)
    /// colliding.
    pub fn from_gguf_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        ensure!(context_size > 0, "context_size must be > 0");
        let prefix = "lfm2";

        let n_layers = gguf
            .get_u32(&format!("{prefix}.block_count"))
            .context("missing lfm2.block_count")? as usize;
        let hidden_size = gguf
            .get_u32(&format!("{prefix}.embedding_length"))
            .context("missing lfm2.embedding_length")? as usize;
        let intermediate_size = gguf
            .get_u32(&format!("{prefix}.feed_forward_length"))
            .context("missing lfm2.feed_forward_length")? as usize;
        let n_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count"))
            .context("missing lfm2.attention.head_count")? as usize;
        let vocab_size = gguf
            .get_u32(&format!("{prefix}.vocab_size"))
            .context("missing lfm2.vocab_size")? as usize;
        // Cap the model's max_seq_len by the user's requested context_size so
        // KV cache pre-allocation in `InferenceState::from_config_with_compression`
        // matches the actual budget. Mirrors the pattern used by metal_lfm2 and
        // gpu_lfm2.
        let gguf_max_seq_len = gguf
            .get_u32(&format!("{prefix}.context_length"))
            .unwrap_or(128000) as usize;
        let max_seq_len = context_size.min(gguf_max_seq_len);
        let rope_theta = gguf
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(1_000_000.0);
        let rms_norm_eps = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let conv_kernel_size = gguf
            .get_u32(&format!("{prefix}.shortconv.l_cache"))
            .map(|v| v as usize);

        // Per-layer KV head counts
        let kv_heads_array = gguf
            .get_i32_array(&format!("{prefix}.attention.head_count_kv"))
            .context("missing lfm2.attention.head_count_kv")?;

        // Validate kv_heads_array length matches n_layers
        anyhow::ensure!(
            kv_heads_array.len() >= n_layers,
            "head_count_kv array length ({}) < block_count ({n_layers})",
            kv_heads_array.len()
        );

        // Detect block types from tensor presence
        let mut block_types = Vec::with_capacity(n_layers);
        let mut kv_heads_per_layer = Vec::with_capacity(n_layers);
        for (i, &kv_heads) in kv_heads_array.iter().enumerate().take(n_layers) {
            let is_attn = gguf.tensors.contains_key(&format!("blk.{i}.attn_q.weight"));
            if is_attn {
                let n_kv = kv_heads as usize;
                anyhow::ensure!(
                    n_kv > 0 && n_heads.is_multiple_of(n_kv),
                    "layer {i}: n_kv_heads ({n_kv}) must be > 0 and divide n_heads ({n_heads})"
                );
                block_types.push(BlockType::Attention);
                kv_heads_per_layer.push(n_kv);
            } else {
                block_types.push(BlockType::GatedConv);
                kv_heads_per_layer.push(0);
            }
        }

        let n_kv_heads = kv_heads_per_layer.iter().copied().max().unwrap_or(0);

        let config = ModelConfig {
            architecture: "lfm2".to_string(),
            n_layers,
            hidden_size,
            intermediate_size,
            n_heads,
            n_kv_heads,
            head_dim: hidden_size / n_heads,
            vocab_size,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
            block_types: block_types.clone(),
            conv_kernel_size,
            kv_heads_per_layer: kv_heads_per_layer.clone(),
            scalars: ScalarMultipliers::default(),
        };

        // Pre-extract small F32 weights
        let output_norm_weight = gguf.get_tensor("token_embd_norm.weight")?.to_f32_vec();

        let mut attn_norm_weights = Vec::with_capacity(n_layers);
        let mut ffn_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_q_norm_weights = Vec::with_capacity(n_layers);
        let mut attn_k_norm_weights = Vec::with_capacity(n_layers);
        let mut conv_weights = Vec::with_capacity(n_layers);

        for (i, bt) in block_types.iter().enumerate() {
            attn_norm_weights.push(
                gguf.get_tensor(&format!("blk.{i}.attn_norm.weight"))?
                    .to_f32_vec(),
            );
            ffn_norm_weights.push(
                gguf.get_tensor(&format!("blk.{i}.ffn_norm.weight"))?
                    .to_f32_vec(),
            );

            if *bt == BlockType::Attention {
                attn_q_norm_weights.push(Some(
                    gguf.get_tensor(&format!("blk.{i}.attn_q_norm.weight"))?
                        .to_f32_vec(),
                ));
                attn_k_norm_weights.push(Some(
                    gguf.get_tensor(&format!("blk.{i}.attn_k_norm.weight"))?
                        .to_f32_vec(),
                ));
                conv_weights.push(None);
            } else {
                attn_q_norm_weights.push(None);
                attn_k_norm_weights.push(None);
                conv_weights.push(Some(
                    gguf.get_tensor(&format!("blk.{i}.shortconv.conv.weight"))?
                        .to_f32_vec(),
                ));
            }
        }

        // Pre-resolve quantized weight references
        let embd_ref = Self::resolve_weight(&gguf, "token_embd.weight")?;

        let mut layer_refs = Vec::with_capacity(n_layers);
        for (i, bt) in block_types.iter().enumerate() {
            // `.with_repack` on every projection weight (all hit the batched
            // prefill GEMM at `n > 1`); token_embd is excluded above.
            let ffn_gate = Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_gate.weight"))?
                .with_repack(&gguf);
            let ffn_up =
                Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_up.weight"))?.with_repack(&gguf);
            let ffn_down = Self::resolve_weight(&gguf, &format!("blk.{i}.ffn_down.weight"))?
                .with_repack(&gguf);

            let (shortconv_in_proj, shortconv_out_proj, attn_q, attn_k, attn_v, attn_output) =
                if *bt == BlockType::GatedConv {
                    (
                        Some(
                            Self::resolve_weight(
                                &gguf,
                                &format!("blk.{i}.shortconv.in_proj.weight"),
                            )?
                            .with_repack(&gguf),
                        ),
                        Some(
                            Self::resolve_weight(
                                &gguf,
                                &format!("blk.{i}.shortconv.out_proj.weight"),
                            )?
                            .with_repack(&gguf),
                        ),
                        None,
                        None,
                        None,
                        None,
                    )
                } else {
                    (
                        None,
                        None,
                        Some(
                            Self::resolve_weight(&gguf, &format!("blk.{i}.attn_q.weight"))?
                                .with_repack(&gguf),
                        ),
                        Some(
                            Self::resolve_weight(&gguf, &format!("blk.{i}.attn_k.weight"))?
                                .with_repack(&gguf),
                        ),
                        Some(
                            Self::resolve_weight(&gguf, &format!("blk.{i}.attn_v.weight"))?
                                .with_repack(&gguf),
                        ),
                        Some(
                            Self::resolve_weight(&gguf, &format!("blk.{i}.attn_output.weight"))?
                                .with_repack(&gguf),
                        ),
                    )
                };

            layer_refs.push(LayerWeightRefs {
                ffn_gate,
                ffn_up,
                ffn_down,
                shortconv_in_proj,
                shortconv_out_proj,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
            });
        }

        let prefix_cache = Mutex::new(KvPrefixCache::new(
            crate::kv_cache::KvCacheConfig::default(),
            &config,
            &format!("cpu:{model_id}"),
        ));

        Ok(Self {
            gguf,
            config,
            output_norm_weight,
            attn_norm_weights,
            ffn_norm_weights,
            attn_q_norm_weights,
            attn_k_norm_weights,
            conv_weights,
            embd_ref,
            layer_refs,
            model_id,
            prefix_cache,
        })
    }

    /// Resolve a tensor name to a pre-computed byte range in the mmap.
    /// Thin wrapper over the shared `transformer::resolve_weight`.
    fn resolve_weight(gguf: &GgufFile, name: &str) -> Result<WeightRef> {
        transformer::resolve_weight(gguf, name)
    }

    // ── Public accessors for GPU model construction ───────────────────────

    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    pub fn output_norm_weight(&self) -> &[f32] {
        &self.output_norm_weight
    }

    pub fn attn_norm_weight(&self, layer: usize) -> &[f32] {
        &self.attn_norm_weights[layer]
    }

    pub fn ffn_norm_weight(&self, layer: usize) -> &[f32] {
        &self.ffn_norm_weights[layer]
    }

    pub fn attn_q_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.attn_q_norm_weights[layer].as_deref()
    }

    pub fn attn_k_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        self.attn_k_norm_weights[layer].as_deref()
    }

    pub fn conv_weight(&self, layer: usize) -> Option<&[f32]> {
        self.conv_weights[layer].as_deref()
    }

    /// Dequantize a token embedding row to f32.
    pub fn dequantize_embedding(&self, token_id: usize) -> Vec<f32> {
        self.dequantize_row(&self.embd_ref, token_id)
    }

    /// Conv in_proj GEMV for a layer.
    pub fn conv_in_proj_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) {
        let wref = self.layer_refs[layer].shortconv_in_proj.as_ref().unwrap();
        self.gemv(wref, x, y);
    }

    /// Conv out_proj GEMV for a layer.
    pub fn conv_out_proj_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) {
        let wref = self.layer_refs[layer].shortconv_out_proj.as_ref().unwrap();
        self.gemv(wref, x, y);
    }

    /// FFN gate GEMV for a layer.
    pub fn ffn_gate_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) {
        self.gemv(&self.layer_refs[layer].ffn_gate, x, y);
    }

    /// FFN up GEMV for a layer.
    pub fn ffn_up_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) {
        self.gemv(&self.layer_refs[layer].ffn_up, x, y);
    }

    /// FFN down GEMV for a layer.
    pub fn ffn_down_gemv(&self, layer: usize, x: &[f32], y: &mut [f32]) {
        self.gemv(&self.layer_refs[layer].ffn_down, x, y);
    }

    /// Returns (ffn_gate_m, ffn_gate_k, ffn_down_m, ffn_down_k) for a layer.
    pub fn layer_weight_info(&self, layer: usize) -> LayerWeightDims {
        let refs = &self.layer_refs[layer];
        LayerWeightDims {
            ffn_gate_m: refs.ffn_gate.m,
            ffn_gate_k: refs.ffn_gate.k,
            ffn_down_m: refs.ffn_down.m,
            ffn_down_k: refs.ffn_down.k,
        }
    }

    /// Get raw weight bytes for a WeightRef (for GPU quantized upload).
    #[allow(dead_code)] // used by metal_lfm2/gpu_lfm2 behind feature gates
    pub(crate) fn weight_bytes(&self, wref: &WeightRef) -> &[u8] {
        self.weight_data(wref)
    }

    // Full-matrix dequant lives in `transformer::dequantize_weight`; the LFM2
    // `GpuWeightSource` impl delegates to it. (The old inherent duplicate +
    // `dequantize_row_into_slice` helper were removed — single implementation.)

    /// Access the per-layer weight refs (for GPU model construction).
    #[allow(dead_code)]
    pub(crate) fn layer_refs(&self) -> &[LayerWeightRefs] {
        &self.layer_refs
    }

    /// Access the embedding weight ref.
    #[allow(dead_code)]
    pub(crate) fn embd_ref(&self) -> &WeightRef {
        &self.embd_ref
    }

    // ── Internal methods ────────────────────────────────────────────────

    /// Get the raw bytes for a pre-resolved weight.
    #[inline]
    fn weight_data(&self, wref: &WeightRef) -> &[u8] {
        transformer::weight_data(&self.gguf, wref)
    }

    /// GEMV dispatch without scratch buffers (shared `transformer::gemv`).
    fn gemv(&self, wref: &WeightRef, x: &[f32], y: &mut [f32]) {
        transformer::gemv(&self.gguf, wref, x, y);
    }

    /// GEMV with pre-quantized Q8_0 input (shared `transformer::gemv_preq`).
    #[cfg(target_arch = "aarch64")]
    fn gemv_preq(&self, wref: &WeightRef, x_f32: &[f32], q8s: &[f32], q8q: &[i8], y: &mut [f32]) {
        transformer::gemv_preq(&self.gguf, wref, x_f32, q8s, q8q, y);
    }

    // The batched-prefill GEMM helpers (`try_blas_prefill_gemm`, `gemm_preq`,
    // `quantize_columns`) are shared with the dense-transformer path and now
    // live as free functions in `transformer` — call sites use
    // `transformer::{try_blas_prefill_gemm, gemm_preq, quantize_columns}`.

    /// Quantize x to Q8_0 into scratch buffers (shared
    /// `transformer::quantize_to_scratch`).
    #[cfg(target_arch = "aarch64")]
    fn quantize_to_scratch(x: &[f32], state: &mut InferenceState) {
        transformer::quantize_to_scratch(x, state);
    }

    /// Dequantize a single row from a quantized matrix (for embedding lookup).
    fn dequantize_row(&self, wref: &WeightRef, row_idx: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; wref.k];
        self.dequantize_row_into(wref, row_idx, &mut out);
        out
    }

    /// Dequantize a single row into `out` (shared
    /// `transformer::dequantize_row_into`).
    fn dequantize_row_into(&self, wref: &WeightRef, row_idx: usize, out: &mut [f32]) {
        transformer::dequantize_row_into(&self.gguf, wref, row_idx, out);
    }

    /// Process a single conv (recurrent) block using pre-allocated scratch buffers.
    fn forward_conv_block(&self, layer: usize, hidden: &[f32], state: &mut InferenceState) {
        let refs = &self.layer_refs[layer];
        let hidden_size = self.config.hidden_size;
        let kernel_size = self.config.conv_kernel_size.unwrap_or(3);
        let d_conv = kernel_size - 1;
        let in_proj = refs.shortconv_in_proj.as_ref().unwrap();
        let out_proj = refs.shortconv_out_proj.as_ref().unwrap();
        let conv_weight = self.conv_weights[layer].as_ref().unwrap();

        // Cloned once (cheap Arc bump) so the adapter can be read while the base
        // scratch buffers stay mutably borrowed — same pattern as
        // `forward_attn_block`.
        let lora = state.lora.clone();

        // in_proj: hidden → 3*hidden (uses pre-quantized Q8_0 data when available)
        let proj = &mut state.scratch.conv_proj[..3 * hidden_size];
        #[cfg(target_arch = "aarch64")]
        if in_proj.dtype == DType::Q4_0 || in_proj.dtype == DType::Q8_0 {
            let data = self.weight_data(in_proj);
            if in_proj.dtype == DType::Q4_0 {
                cpu::gemv_q4_0_with_q8(
                    data,
                    &state.scratch.q8_scales,
                    &state.scratch.q8_quants,
                    proj,
                    in_proj.m,
                    in_proj.k,
                );
            } else {
                unsafe {
                    crate::backend::simd::neon::gemv_q8_0_q8_0_neon(
                        data,
                        &state.scratch.q8_scales,
                        &state.scratch.q8_quants,
                        proj,
                        in_proj.m,
                        in_proj.k,
                    );
                }
            }
        } else {
            self.gemv(in_proj, hidden, proj);
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(in_proj, hidden, proj);

        // LoRA on the conv in_proj — `proj += scale·B·(A·hidden)`, applied to the
        // full 3·hidden output before it is split into the B/C/x gates. Matches
        // llama.cpp applying the adapter to `shortconv.in_proj` on conv layers.
        if let Some(lora) = &lora
            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::ShortconvInProj)
        {
            crate::lora::apply_decode(t, hidden, proj, &mut state.scratch.lora_tmp);
        }

        // Split: b, c, x
        let (b, rest) = proj.split_at(hidden_size);
        let (c, x) = rest.split_at(hidden_size);

        // bx = b ⊙ x (element-wise gate before conv)
        let conv_scratch = &mut state.scratch.conv_scratch[..hidden_size];
        for (out, (bi, xi)) in conv_scratch.iter_mut().zip(b.iter().zip(x.iter())) {
            *out = bi * xi;
        }
        // conv_scratch now holds bx

        // Depthwise conv1d with valid convolution using rolling buffer
        let LayerState::Conv { buffer } = &mut state.layers[layer] else {
            panic!("expected Conv state for layer {layer}");
        };

        let out_buf = &mut state.scratch.out[..hidden_size];
        for ch in 0..hidden_size {
            let mut sum = 0.0f32;
            for k in 0..d_conv {
                sum += buffer[k * hidden_size + ch] * conv_weight[ch * kernel_size + k];
            }
            sum += conv_scratch[ch] * conv_weight[ch * kernel_size + d_conv];
            out_buf[ch] = sum;
        }

        // Update rolling buffer: shift left by one slot, append bx
        if d_conv > 0 {
            if d_conv > 1 {
                buffer.copy_within(hidden_size.., 0);
            }
            let last_slot = (d_conv - 1) * hidden_size;
            buffer[last_slot..last_slot + hidden_size].copy_from_slice(conv_scratch);
        }

        // o = c ⊙ conv_out (second gate), reuse conv_scratch
        for (o, (ci, co)) in conv_scratch.iter_mut().zip(c.iter().zip(out_buf.iter())) {
            *o = ci * co;
        }

        // out_proj: hidden → hidden, write result into out_buf
        self.gemv(out_proj, conv_scratch, out_buf);
        // LoRA on the conv out_proj — `out_buf += scale·B·(A·conv_scratch)`, where
        // conv_scratch is the gated conv output that feeds out_proj.
        if let Some(lora) = &lora
            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::ShortconvOutProj)
        {
            crate::lora::apply_decode(t, conv_scratch, out_buf, &mut state.scratch.lora_tmp);
        }
        // Result is now in state.scratch.out[..hidden_size]
    }

    /// Process a single attention block using pre-allocated scratch buffers.
    fn forward_attn_block(
        &self,
        layer: usize,
        hidden: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) {
        let refs = &self.layer_refs[layer];
        let cfg = &self.config;
        let head_dim = cfg.hidden_size / cfg.n_heads;
        let n_kv_heads = cfg.kv_heads_per_layer[layer];
        let kv_dim = n_kv_heads * head_dim;

        // Cloned once (cheap Arc bump) so the base scratch buffers can stay
        // mutably borrowed while the adapter (a disjoint field) is read.
        let lora = state.lora.clone();

        // Q, K, V projections using pre-quantized hidden state
        let q = &mut state.scratch.q[..cfg.hidden_size];
        let k = &mut state.scratch.k[..kv_dim];
        let v = &mut state.scratch.v[..kv_dim];

        // hidden was pre-quantized at layer level — use integer path
        #[cfg(target_arch = "aarch64")]
        {
            self.gemv_preq(
                refs.attn_q.as_ref().unwrap(),
                hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                q,
            );
            self.gemv_preq(
                refs.attn_k.as_ref().unwrap(),
                hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                k,
            );
            self.gemv_preq(
                refs.attn_v.as_ref().unwrap(),
                hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                v,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.gemv(refs.attn_q.as_ref().unwrap(), hidden, q);
            self.gemv(refs.attn_k.as_ref().unwrap(), hidden, k);
            self.gemv(refs.attn_v.as_ref().unwrap(), hidden, v);
        }

        // LoRA on Q/K/V — input is the normed hidden, applied before QK-norm/RoPE.
        if let Some(lora) = &lora {
            crate::lora::apply_attn_qkv(lora, layer, hidden, q, k, v, &mut state.scratch.lora_tmp);
        }

        // Per-head QK norm (RMSnorm each head slice with shared weights)
        let q_norm = self.attn_q_norm_weights[layer].as_ref().unwrap();
        let k_norm = self.attn_k_norm_weights[layer].as_ref().unwrap();
        for h in 0..cfg.n_heads {
            cpu::rmsnorm(
                &mut q[h * head_dim..(h + 1) * head_dim],
                q_norm,
                cfg.rms_norm_eps,
            );
        }
        for h in 0..n_kv_heads {
            cpu::rmsnorm(
                &mut k[h * head_dim..(h + 1) * head_dim],
                k_norm,
                cfg.rms_norm_eps,
            );
        }

        // RoPE
        cpu::rope(q, k, pos, cfg.n_heads, n_kv_heads, head_dim, cfg.rope_theta);

        // Grab per-model TurboQuant state once (None when disabled)
        // TurboQuant rotation state lives on InferenceState now (since PR #12
        // refactor). A single KvCompression::TurboQuant { seed, ... } config
        // on the state is enough — no separate model-side enable needed.
        let tq_rotation = state.tq_rotations.get(layer).and_then(|r| r.as_ref());
        let tq_config = state.tq_config.as_ref();

        // Append K, V to cache. Keys and values are compressed independently —
        // whichever side has a CompressedKvCache present gets the TurboQuant
        // path; the other side falls through to the f32 cache.
        if let LayerState::Attention {
            key_cache,
            value_cache,
            compressed_keys,
            compressed_values,
            ..
        } = &mut state.layers[layer]
        {
            let tq_ok =
                tq_rotation.is_some() && tq_config.is_some() && state.tq_encode_scratch.is_some();
            match (tq_ok, compressed_keys.as_mut()) {
                (true, Some(k_cache_tq)) => {
                    turboquant::compress_and_append_keys(
                        &state.scratch.k[..kv_dim],
                        n_kv_heads,
                        head_dim,
                        tq_rotation.unwrap(),
                        tq_config.unwrap(),
                        k_cache_tq,
                        state.tq_encode_scratch.as_mut().unwrap(),
                    );
                }
                _ => {
                    key_cache.extend_from_slice(&state.scratch.k[..kv_dim]);
                }
            }
            match (tq_ok, compressed_values.as_mut()) {
                (true, Some(v_cache_tq)) => {
                    turboquant::compress_and_append_values(
                        &state.scratch.v[..kv_dim],
                        n_kv_heads,
                        head_dim,
                        tq_rotation.unwrap(),
                        tq_config.unwrap(),
                        v_cache_tq,
                        state.tq_encode_scratch.as_mut().unwrap(),
                    );
                }
                _ => {
                    value_cache.extend_from_slice(&state.scratch.v[..kv_dim]);
                }
            }
        }

        // GQA: grouped query attention
        let group_size = cfg.n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        {
            // Access layers and scratch as disjoint fields to avoid whole-state borrow
            let (ck, cv, k_cache, v_cache) = match &state.layers[layer] {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    compressed_keys,
                    compressed_values,
                    ..
                } => (
                    compressed_keys.as_ref(),
                    compressed_values.as_ref(),
                    key_cache.as_slice(),
                    value_cache.as_slice(),
                ),
                _ => panic!("expected Attention state for layer {layer}"),
            };

            // Keys and values are compressed independently — determine which
            // side of the attention read path uses TurboQuant.
            let tq_prereq =
                tq_rotation.is_some() && tq_config.is_some() && state.tq_query_scratch.is_some();
            let use_tq_keys = tq_prereq && ck.is_some();
            let use_tq_values = tq_prereq && cv.is_some();

            // seq_len comes from whichever cache is populated. All four
            // combinations agree on seq_len per layer because encode appends
            // to one cache per side per token.
            let seq_len = if use_tq_keys {
                ck.unwrap().seq_len()
            } else if use_tq_values {
                cv.unwrap().seq_len()
            } else {
                k_cache.len() / kv_dim
            };
            let attn_out = &mut state.scratch.attn_out[..cfg.hidden_size];
            let q = &state.scratch.q[..cfg.hidden_size];
            let scores = &mut state.scratch.scores;

            if use_tq_keys || use_tq_values {
                // GQA batched path — one score buffer per group, shared
                // between the key score and value weighted-sum stages.
                let rotation = tq_rotation.unwrap();
                let cfg_tq = tq_config.unwrap();
                let qr_scratch = state.tq_query_scratch.as_mut().unwrap();
                if use_tq_keys {
                    turboquant::rotate_queries(q, cfg.n_heads, head_dim, rotation, qr_scratch);
                }
                scores.resize(seq_len * group_size, 0.0);
                for kv_h in 0..n_kv_heads {
                    let group_start = kv_h * group_size;
                    let kv_h_offset = kv_h * head_dim;

                    // Scores: TurboQuant or f32.
                    if use_tq_keys {
                        turboquant::attn_scores_turboquant_gqa(
                            ck.unwrap(),
                            kv_h,
                            group_start,
                            group_size,
                            scores,
                            head_dim,
                            scale,
                            seq_len,
                            cfg_tq,
                            qr_scratch,
                        );
                    } else {
                        for g in 0..group_size {
                            let h = group_start + g;
                            let q_head = &q[h * head_dim..(h + 1) * head_dim];
                            let head_scores = &mut scores[g * seq_len..(g + 1) * seq_len];
                            cpu::attn_scores(
                                q_head,
                                k_cache,
                                head_scores,
                                kv_dim,
                                kv_h_offset,
                                head_dim,
                                scale,
                                seq_len,
                            );
                        }
                    }

                    // Softmax each head's scores in place.
                    for g in 0..group_size {
                        let head_scores = &mut scores[g * seq_len..(g + 1) * seq_len];
                        cpu::softmax_inplace(head_scores);
                    }

                    // Values: TurboQuant or f32.
                    if use_tq_values {
                        turboquant::attn_values_turboquant_gqa(
                            cv.unwrap(),
                            kv_h,
                            group_start,
                            group_size,
                            scores,
                            attn_out,
                            head_dim,
                            seq_len,
                            rotation,
                            cfg_tq,
                        );
                    } else {
                        for g in 0..group_size {
                            let h = group_start + g;
                            let head_scores = &scores[g * seq_len..(g + 1) * seq_len];
                            cpu::attn_values(
                                head_scores,
                                v_cache,
                                &mut attn_out[h * head_dim..(h + 1) * head_dim],
                                kv_dim,
                                kv_h_offset,
                                head_dim,
                                seq_len,
                            );
                        }
                    }
                }
            } else {
                scores.resize(seq_len, 0.0);
                for h in 0..cfg.n_heads {
                    let kv_h = h / group_size;
                    let q_head = &q[h * head_dim..(h + 1) * head_dim];
                    let kv_h_offset = kv_h * head_dim;
                    cpu::attn_scores(
                        q_head,
                        k_cache,
                        scores,
                        kv_dim,
                        kv_h_offset,
                        head_dim,
                        scale,
                        seq_len,
                    );
                    cpu::softmax_inplace(scores);
                    cpu::attn_values(
                        scores,
                        v_cache,
                        &mut attn_out[h * head_dim..(h + 1) * head_dim],
                        kv_dim,
                        kv_h_offset,
                        head_dim,
                        seq_len,
                    );
                }
            }
        }

        // Output projection
        let out = &mut state.scratch.out[..cfg.hidden_size];
        self.gemv(
            refs.attn_output.as_ref().unwrap(),
            &state.scratch.attn_out[..cfg.hidden_size],
            out,
        );
        // LoRA on the output projection (input = the attention output).
        if let Some(lora) = &lora
            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnOutput)
        {
            crate::lora::apply_decode(
                t,
                &state.scratch.attn_out[..cfg.hidden_size],
                out,
                &mut state.scratch.lora_tmp,
            );
        }
    }
    /// Run all layers + output norm on a hidden state vector. Shared by
    /// forward(), forward_embedding(), and forward_hidden_from_embedding().
    fn run_layers(&self, hidden: &mut [f32], pos: usize, state: &mut InferenceState) {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        // Reuse pre-allocated scratch from InferenceState instead of allocating
        // fresh Vecs on every call. Take them out of `state.scratch` to avoid
        // borrow-checker conflicts with the mutable `state` passed to
        // forward_attn_block / forward_conv_block below; put them back at the end.
        let mut normed = std::mem::take(&mut state.scratch.normed);
        let mut ffn_input = std::mem::take(&mut state.scratch.ffn_input);
        normed.resize(hs, 0.0);
        ffn_input.resize(hs, 0.0);

        for i in 0..cfg.n_layers {
            normed.copy_from_slice(hidden);
            cpu::rmsnorm(&mut normed, &self.attn_norm_weights[i], cfg.rms_norm_eps);

            #[cfg(target_arch = "aarch64")]
            Self::quantize_to_scratch(&normed, state);

            if cfg.block_types[i] == BlockType::GatedConv {
                self.forward_conv_block(i, &normed, state);
            } else {
                self.forward_attn_block(i, &normed, pos, state);
            }

            cpu::add_inplace(hidden, &state.scratch.out[..hs]);

            ffn_input.copy_from_slice(hidden);
            cpu::rmsnorm(&mut ffn_input, &self.ffn_norm_weights[i], cfg.rms_norm_eps);

            // SwiGLU FFN via the shared helper. On aarch64 it consumes the
            // pre-quantized ffn_input, so quantize first (same contract as the
            // llama/qwen per-token path).
            #[cfg(target_arch = "aarch64")]
            Self::quantize_to_scratch(&ffn_input, state);

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

            cpu::add_inplace(hidden, &state.scratch.out[..cfg.hidden_size]);
        }

        cpu::rmsnorm(hidden, &self.output_norm_weight, cfg.rms_norm_eps);
        state.seq_len += 1;

        // Return the scratch buffers for the next call.
        state.scratch.normed = normed;
        state.scratch.ffn_input = ffn_input;
    }

    /// Layer loop + output norm + tied logit projection for batched
    /// prefill. Takes a column-major hidden buffer (`hs × n`, channel
    /// `i` of token `j` at index `i * n + j`) already populated by the
    /// caller, runs the per-layer attn / conv + FFN passes, advances
    /// `state.seq_len` to `start_pos + n`, and projects the last
    /// frame's hidden state to logits over the vocabulary.
    ///
    /// Shared between [`Self::forward_prefill`] (token-id input,
    /// embedding-table lookup) and [`Self::forward_prefill_from_embeddings`]
    /// (raw embedding input, copied + transposed into the column-major
    /// `hidden` layout). The two entry points differ only in how they
    /// fill `hidden`; everything from the first RMSnorm onward is
    /// identical and lives here.
    fn prefill_layers_and_logits(
        &self,
        mut hidden: Vec<f32>,
        n: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;

        // Cloned once (cheap Arc bump) so the adapter can be read after each
        // projection GEMM while the base-weight scratch buffers stay borrowed.
        let lora = state.lora.clone();

        // Per-layer hidden-state diagnostic. Set `CERA_DEBUG_HIDDEN=1`
        // to print the last token's RMS at each layer entry, after
        // attn/conv (post 1st residual), and after FFN (post 2nd
        // residual). Used to find the layer where cera's hidden
        // states diverge from llama.cpp's reference. Off in
        // production: a missing-env-var check is one syscall per
        // call, gated above every loop body to keep the hot path
        // cold. Any non-`"1"` value (including unset, empty, or
        // `"0"`) leaves diagnostics off — this matches the
        // documented `=1` setter and avoids a stray
        // `CERA_DEBUG_HIDDEN=0` accidentally enabling logging.
        let debug_hidden = std::env::var("CERA_DEBUG_HIDDEN").as_deref() == Ok("1");
        let log_rms = |label: &str, hidden: &[f32]| {
            if !debug_hidden {
                return;
            }
            // Last token's hidden vector lives at `hidden[i * n + (n-1)]`
            // for i in 0..hs (column-major). RMS of those `hs` values
            // is what feeds the next layer / output norm.
            let mut sum_sq = 0.0f64;
            let mut max_abs = 0.0f64;
            for i in 0..hs {
                let v = hidden[i * n + (n - 1)] as f64;
                sum_sq += v * v;
                let abs_v = v.abs();
                if abs_v > max_abs {
                    max_abs = abs_v;
                }
            }
            let rms = (sum_sq / hs as f64).sqrt();
            eprintln!("[cera.hidden] {label}: rms={rms:.6e} max_abs={max_abs:.6e}");
        };
        log_rms("input (pre-layer-0)", &hidden);

        // Per-layer loop — pre-allocate all large buffers outside the loop
        let mut normed = vec![0.0f32; hs * n];
        let mut block_out = vec![0.0f32; hs * n];
        let mut ffn_input = vec![0.0f32; hs * n];
        let mut ffn_out = vec![0.0f32; hs * n];
        let mut norm_col = vec![0.0f32; hs];
        let mut ffn_col = vec![0.0f32; hs];
        let mut col = vec![0.0f32; hs];
        let mut gate_col = vec![0.0f32; cfg.intermediate_size];
        let mut up_col = vec![0.0f32; cfg.intermediate_size];
        let mut out_col = vec![0.0f32; hs];
        // Batched projection buffers for conv/attn input projections.
        // Used by the no-`blas` int8 `gemm_preq` path (aarch64 NEON and
        // x86_64 int8, VNNI or AVX2) and the any-arch BLAS path
        // (`try_blas_prefill_gemm`).
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let max_kv_dim =
            cfg.kv_heads_per_layer.iter().copied().max().unwrap_or(0) * (hs / cfg.n_heads);
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let proj_rows = (3 * hs).max(hs + 2 * max_kv_dim);
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut proj_mat = vec![0.0f32; proj_rows * n];
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut out_proj_input = vec![0.0f32; hs * n];
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut q_mat = vec![0.0f32; hs * n];
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut k_mat = vec![0.0f32; max_kv_dim * n];
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut v_mat = vec![0.0f32; max_kv_dim * n];
        // Pre-allocated GEMM buffers (reused across layers)
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let is = cfg.intermediate_size;
        // bq_*/dq_*/inter_col are scratch for the no-`blas` `gemm_preq` path
        // (aarch64 NEON and x86_64 int8, VNNI or AVX2)
        // (they hold the pre-quantized Q8_0 input matrix). With BLAS on, the
        // SGEMM path consumes f32 directly and these buffers are not needed.
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        let nb_hs = hs / 32;
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        let nb_is = is / 32;
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        let mut bq_scales = vec![0.0f32; n * nb_hs];
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        let mut bq_quants = vec![0i8; n * hs];
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut gate_mat = vec![0.0f32; is * n];
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut up_mat = vec![0.0f32; is * n];
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        let mut dq_scales = vec![0.0f32; n * nb_is];
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        let mut dq_quants = vec![0i8; n * is];
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            not(feature = "blas")
        ))]
        let mut inter_col = vec![0.0f32; is];
        // Flash attention scratch: contiguous output buffer reused across
        // layers. Sized for the largest possible attention layer (max
        // n_kv_heads * group_size * n * head_dim = hs * n).
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
        let mut flash_out = vec![0.0f32; hs * n];

        for layer in 0..cfg.n_layers {
            // RMSnorm each column independently
            for j in 0..n {
                for i in 0..hs {
                    norm_col[i] = hidden[i * n + j];
                }
                cpu::rmsnorm(
                    &mut norm_col,
                    &self.attn_norm_weights[layer],
                    cfg.rms_norm_eps,
                );
                for i in 0..hs {
                    normed[i * n + j] = norm_col[i];
                }
            }

            // Operator: conv or attention — batch projections via GEMM, sequential core
            let is_conv = cfg.block_types[layer] == BlockType::GatedConv;

            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
            let used_block_gemm = {
                let refs = &self.layer_refs[layer];
                if is_conv {
                    // --- Conv: batch in_proj + out_proj via GEMM ---
                    let in_proj = refs.shortconv_in_proj.as_ref().unwrap();
                    let out_proj = refs.shortconv_out_proj.as_ref().unwrap();
                    // Require BOTH projections to be batchable: a mixed-dtype conv
                    // block would leave the second matrix silently uncomputed. Any
                    // other combo falls through to the per-token fallback — loudly,
                    // because a quiet fallback here costs ~4x prefill.
                    let blas_ok = [
                        ("shortconv.in_proj", in_proj),
                        ("shortconv.out_proj", out_proj),
                    ]
                    .into_iter()
                    .fold(true, |ok, (name, w)| {
                        if transformer::batched_gemm_supports(w.dtype, w.k) {
                            ok
                        } else {
                            transformer::warn_unbatchable(name, w.dtype);
                            false
                        }
                    });
                    if blas_ok {
                        // Phase 1: Batch in_proj GEMM: normed[hs×n] → proj_mat[3*hs × n]
                        // quantize_columns is only needed for the NEON fallback. With BLAS
                        // on, the SGEMM path consumes f32 directly so this work is skipped.
                        #[cfg(not(feature = "blas"))]
                        transformer::quantize_columns(
                            &normed,
                            hs,
                            n,
                            &mut col,
                            &mut bq_scales,
                            &mut bq_quants,
                        );
                        #[cfg(feature = "blas")]
                        {
                            transformer::try_blas_prefill_gemm(
                                &self.gguf,
                                in_proj,
                                &normed,
                                &mut proj_mat,
                                3 * hs,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                        }
                        // Sliced, like the LoRA call below: `proj_mat` is sized
                        // `max(3*hs, hs + 2*kv_dim) * n` because it is shared
                        // with the attention projection, so it can be longer
                        // than this GEMM's `3*hs*n` output. `gemm_preq` slices
                        // defensively too, but the invariant belongs where the
                        // over-long buffer is created.
                        #[cfg(not(feature = "blas"))]
                        transformer::gemm_preq(
                            &self.gguf,
                            in_proj,
                            &bq_scales,
                            &bq_quants,
                            &mut proj_mat[..3 * hs * n],
                            3 * hs,
                            n,
                            hs,
                        );

                        // LoRA on the conv in_proj — `proj_mat[3hs×n] += scale·B·(A·normed)`,
                        // applied to the full projection before the B/C/x split. Mirrors
                        // the per-token `forward_conv_block` path for the batched prefill.
                        // `proj_mat` is sized `proj_rows = max(3·hs, hs+2·max_kv_dim)`, which
                        // can exceed `3·hs`; slice to the conv's `3·hs` rows so the length
                        // matches `apply_prefill`'s `t.d × n` contract exactly.
                        if let Some(lora) = &lora
                            && let Some(t) =
                                lora.get(layer, crate::lora::LoraTarget::ShortconvInProj)
                        {
                            crate::lora::apply_prefill(
                                t,
                                &normed,
                                &mut proj_mat[..3 * hs * n],
                                n,
                                &mut state.scratch.lora_tmp,
                            );
                        }

                        // Phase 2: Per-token sequential conv using pre-computed projections
                        let kernel_size = cfg.conv_kernel_size.unwrap_or(3);
                        let d_conv = kernel_size - 1;
                        let conv_weight = self.conv_weights[layer].as_ref().unwrap();
                        for j in 0..n {
                            let proj = &mut state.scratch.conv_proj[..3 * hs];
                            for i in 0..hs {
                                proj[i] = proj_mat[i * n + j];
                                proj[hs + i] = proj_mat[(hs + i) * n + j];
                                proj[2 * hs + i] = proj_mat[(2 * hs + i) * n + j];
                            }
                            let (b, rest) = proj.split_at(hs);
                            let (c_slice, x_slice) = rest.split_at(hs);

                            let conv_scratch = &mut state.scratch.conv_scratch[..hs];
                            for i in 0..hs {
                                conv_scratch[i] = b[i] * x_slice[i];
                            }

                            let LayerState::Conv { buffer } = &mut state.layers[layer] else {
                                panic!("expected Conv state for layer {layer}");
                            };
                            let out_buf = &mut state.scratch.out[..hs];
                            for ch in 0..hs {
                                let mut sum = 0.0f32;
                                for k in 0..d_conv {
                                    sum += buffer[k * hs + ch] * conv_weight[ch * kernel_size + k];
                                }
                                sum += conv_scratch[ch] * conv_weight[ch * kernel_size + d_conv];
                                out_buf[ch] = sum;
                            }
                            if d_conv > 0 {
                                if d_conv > 1 {
                                    buffer.copy_within(hs.., 0);
                                }
                                let last_slot = (d_conv - 1) * hs;
                                buffer[last_slot..last_slot + hs].copy_from_slice(conv_scratch);
                            }

                            for i in 0..hs {
                                out_proj_input[i * n + j] = c_slice[i] * out_buf[i];
                            }
                        }

                        // Phase 3: Batch out_proj GEMM
                        #[cfg(not(feature = "blas"))]
                        transformer::quantize_columns(
                            &out_proj_input,
                            hs,
                            n,
                            &mut col,
                            &mut bq_scales,
                            &mut bq_quants,
                        );
                        #[cfg(feature = "blas")]
                        {
                            transformer::try_blas_prefill_gemm(
                                &self.gguf,
                                out_proj,
                                &out_proj_input,
                                &mut block_out,
                                hs,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                        }
                        #[cfg(not(feature = "blas"))]
                        transformer::gemm_preq(
                            &self.gguf,
                            out_proj,
                            &bq_scales,
                            &bq_quants,
                            &mut block_out,
                            hs,
                            n,
                            hs,
                        );

                        // LoRA on the conv out_proj — `block_out[hs×n] += scale·B·(A·in)`,
                        // where `in` is the gated conv output (`out_proj_input`), before
                        // the residual add.
                        if let Some(lora) = &lora
                            && let Some(t) =
                                lora.get(layer, crate::lora::LoraTarget::ShortconvOutProj)
                        {
                            crate::lora::apply_prefill(
                                t,
                                &out_proj_input,
                                &mut block_out,
                                n,
                                &mut state.scratch.lora_tmp,
                            );
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    // --- Attention: batch Q/K/V + output projections via GEMM ---
                    let attn_q_ref = refs.attn_q.as_ref().unwrap();
                    let attn_k_ref = refs.attn_k.as_ref().unwrap();
                    let attn_v_ref = refs.attn_v.as_ref().unwrap();
                    let attn_output_ref = refs.attn_output.as_ref().unwrap();
                    // Require ALL four projections to be batchable — a mixed-dtype
                    // attention block would leave later matrices silently uncomputed
                    // in the batched path and produce wrong outputs.
                    let blas_ok = [
                        ("attn_q", attn_q_ref),
                        ("attn_k", attn_k_ref),
                        ("attn_v", attn_v_ref),
                        ("attn_output", attn_output_ref),
                    ]
                    .into_iter()
                    .fold(true, |ok, (name, w)| {
                        if transformer::batched_gemm_supports(w.dtype, w.k) {
                            ok
                        } else {
                            transformer::warn_unbatchable(name, w.dtype);
                            false
                        }
                    });
                    if blas_ok {
                        let head_dim = hs / cfg.n_heads;
                        let n_kv_heads = cfg.kv_heads_per_layer[layer];
                        let kv_dim = n_kv_heads * head_dim;

                        // Phase 1: Batch Q/K/V GEMM
                        #[cfg(not(feature = "blas"))]
                        transformer::quantize_columns(
                            &normed,
                            hs,
                            n,
                            &mut col,
                            &mut bq_scales,
                            &mut bq_quants,
                        );
                        #[cfg(feature = "blas")]
                        {
                            transformer::try_blas_prefill_gemm(
                                &self.gguf,
                                attn_q_ref,
                                &normed,
                                &mut q_mat,
                                hs,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                            transformer::try_blas_prefill_gemm(
                                &self.gguf,
                                attn_k_ref,
                                &normed,
                                &mut k_mat[..kv_dim * n],
                                kv_dim,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                            transformer::try_blas_prefill_gemm(
                                &self.gguf,
                                attn_v_ref,
                                &normed,
                                &mut v_mat[..kv_dim * n],
                                kv_dim,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                        }
                        #[cfg(not(feature = "blas"))]
                        {
                            transformer::gemm_preq(
                                &self.gguf, attn_q_ref, &bq_scales, &bq_quants, &mut q_mat, hs, n,
                                hs,
                            );
                            transformer::gemm_preq(
                                &self.gguf,
                                attn_k_ref,
                                &bq_scales,
                                &bq_quants,
                                &mut k_mat[..kv_dim * n],
                                kv_dim,
                                n,
                                hs,
                            );
                            transformer::gemm_preq(
                                &self.gguf,
                                attn_v_ref,
                                &bq_scales,
                                &bq_quants,
                                &mut v_mat[..kv_dim * n],
                                kv_dim,
                                n,
                                hs,
                            );
                        }

                        // LoRA on Q/K/V — added to the projection outputs before
                        // QK-norm/RoPE, input is the normed hidden `[hs×n]`.
                        if let Some(lora) = &lora {
                            if let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnQ) {
                                crate::lora::apply_prefill(
                                    t,
                                    &normed,
                                    &mut q_mat,
                                    n,
                                    &mut state.scratch.lora_tmp,
                                );
                            }
                            if let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnK) {
                                crate::lora::apply_prefill(
                                    t,
                                    &normed,
                                    &mut k_mat[..kv_dim * n],
                                    n,
                                    &mut state.scratch.lora_tmp,
                                );
                            }
                            if let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnV) {
                                crate::lora::apply_prefill(
                                    t,
                                    &normed,
                                    &mut v_mat[..kv_dim * n],
                                    n,
                                    &mut state.scratch.lora_tmp,
                                );
                            }
                        }

                        // Phase 2: Per-token attention (QK norm, RoPE, KV cache, scores)
                        // Hoist tq state capture so the reserve block can match the
                        // exact same condition as the actual append path below, and
                        // so the per-token loop can key off pre-computed bools.
                        let tq_rotation = state.tq_rotations.get(layer).and_then(|r| r.as_ref());
                        let tq_config = state.tq_config.as_ref();
                        // Needed to encode keys + values (append path).
                        let will_compress_kv = tq_rotation.is_some()
                            && tq_config.is_some()
                            && state.tq_encode_scratch.is_some();
                        // Needed to read compressed keys/values (attention path).
                        let will_read_compressed_kv = tq_rotation.is_some()
                            && tq_config.is_some()
                            && state.tq_query_scratch.is_some();

                        // Pre-reserve KV cache to avoid repeated reallocations.
                        // Keys and values are handled independently — whichever
                        // side is compressed reserves the packed buffers;
                        // the other side reserves the f32 flat cache.
                        if let LayerState::Attention {
                            key_cache,
                            value_cache,
                            compressed_keys,
                            compressed_values,
                            ..
                        } = &mut state.layers[layer]
                        {
                            match (will_compress_kv, compressed_keys.as_mut()) {
                                (true, Some(c)) => {
                                    for v in c.polar_data.iter_mut() {
                                        v.reserve(n * head_dim / 4);
                                    }
                                    for v in c.jl_data.iter_mut() {
                                        v.reserve(n * head_dim / 8);
                                    }
                                    for v in c.norms.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.residual_norms.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.norms_f32.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.residual_norms_f32.iter_mut() {
                                        v.reserve(n);
                                    }
                                }
                                _ => {
                                    key_cache.reserve(n * kv_dim);
                                }
                            }
                            match (will_compress_kv, compressed_values.as_mut()) {
                                (true, Some(c)) => {
                                    for v in c.polar_data.iter_mut() {
                                        v.reserve(n * head_dim / 4);
                                    }
                                    for v in c.norms.iter_mut() {
                                        v.reserve(n);
                                    }
                                    for v in c.norms_f32.iter_mut() {
                                        v.reserve(n);
                                    }
                                }
                                _ => {
                                    value_cache.reserve(n * kv_dim);
                                }
                            }
                        }
                        let q_norm = self.attn_q_norm_weights[layer].as_ref().unwrap();
                        let k_norm = self.attn_k_norm_weights[layer].as_ref().unwrap();
                        let group_size = cfg.n_heads / n_kv_heads;
                        let scale = 1.0 / (head_dim as f32).sqrt();

                        // ── Pass A: QK-norm + RoPE + KV cache append ──────────
                        // Processes all n tokens sequentially (O(n) per token).
                        // After this loop, q_mat contains post-RoPE Q and the
                        // KV cache is fully populated through start_pos + n - 1.
                        for j in 0..n {
                            let pos = start_pos + j;
                            let q = &mut state.scratch.q[..hs];
                            let k = &mut state.scratch.k[..kv_dim];
                            let v = &mut state.scratch.v[..kv_dim];
                            for i in 0..hs {
                                q[i] = q_mat[i * n + j];
                            }
                            for i in 0..kv_dim {
                                k[i] = k_mat[i * n + j];
                                v[i] = v_mat[i * n + j];
                            }

                            // QK norm
                            for h in 0..cfg.n_heads {
                                cpu::rmsnorm(
                                    &mut q[h * head_dim..(h + 1) * head_dim],
                                    q_norm,
                                    cfg.rms_norm_eps,
                                );
                            }
                            for h in 0..n_kv_heads {
                                cpu::rmsnorm(
                                    &mut k[h * head_dim..(h + 1) * head_dim],
                                    k_norm,
                                    cfg.rms_norm_eps,
                                );
                            }

                            // RoPE
                            cpu::rope(q, k, pos, cfg.n_heads, n_kv_heads, head_dim, cfg.rope_theta);

                            // Write processed Q back to q_mat so flash attention
                            // can read it. K/V go into the cache below.
                            for i in 0..hs {
                                q_mat[i * n + j] = q[i];
                            }

                            // Append K, V to cache (f32 or TurboQuant-compressed).
                            if let LayerState::Attention {
                                key_cache,
                                value_cache,
                                compressed_keys,
                                compressed_values,
                                ..
                            } = &mut state.layers[layer]
                            {
                                match (will_compress_kv, compressed_keys.as_mut()) {
                                    (true, Some(k_cache_tq)) => {
                                        turboquant::compress_and_append_keys(
                                            &state.scratch.k[..kv_dim],
                                            n_kv_heads,
                                            head_dim,
                                            tq_rotation.unwrap(),
                                            tq_config.unwrap(),
                                            k_cache_tq,
                                            state.tq_encode_scratch.as_mut().unwrap(),
                                        );
                                    }
                                    _ => {
                                        key_cache.extend_from_slice(&state.scratch.k[..kv_dim]);
                                    }
                                }
                                match (will_compress_kv, compressed_values.as_mut()) {
                                    (true, Some(v_cache_tq)) => {
                                        turboquant::compress_and_append_values(
                                            &state.scratch.v[..kv_dim],
                                            n_kv_heads,
                                            head_dim,
                                            tq_rotation.unwrap(),
                                            tq_config.unwrap(),
                                            v_cache_tq,
                                            state.tq_encode_scratch.as_mut().unwrap(),
                                        );
                                    }
                                    _ => {
                                        value_cache.extend_from_slice(&state.scratch.v[..kv_dim]);
                                    }
                                }
                            }
                        }

                        // ── Pass B: attention ────────────────────────────────
                        // The KV cache is now fully populated. Branch on
                        // whether TurboQuant compressed KV is active.
                        let use_tq = will_read_compressed_kv
                            && match &state.layers[layer] {
                                LayerState::Attention {
                                    compressed_keys,
                                    compressed_values,
                                    ..
                                } => compressed_keys.is_some() || compressed_values.is_some(),
                                _ => false,
                            };

                        // Flash attention (tiled + rayon) is faster at longer
                        // prompts. Below the threshold the overhead of the
                        // two-pass decomposition + online softmax exceeds the
                        // naive NEON path, so fall back.
                        // Flash attention (tiled + rayon) is faster than the naive
                        // NEON path only for longer prompts. The crossover is around
                        // pp200 on Apple Silicon (measured: naive wins at pp128 by 5%,
                        // flash wins at pp252 by 6%). Use 256 to avoid regressions.
                        const FLASH_ATTN_THRESHOLD: usize = 256;
                        let use_flash = !use_tq && n >= FLASH_ATTN_THRESHOLD;

                        if use_flash {
                            // f32 path: flash attention over the full KV cache,
                            // parallel across KV heads via rayon.
                            //
                            // Each KV head writes to a contiguous chunk of
                            // flash_out [group_size * n * head_dim], split via
                            // par_chunks_mut so there's no aliased &mut.
                            // After the par_iter we scatter-copy back to
                            // out_proj_input in stride-n layout for Phase 3.
                            let (k_cache, v_cache) = match &state.layers[layer] {
                                LayerState::Attention {
                                    key_cache,
                                    value_cache,
                                    ..
                                } => (key_cache.as_slice(), value_cache.as_slice()),
                                _ => unreachable!(),
                            };
                            let chunk_size = group_size * n * head_dim;
                            let flash_len = n_kv_heads * chunk_size;
                            let flash_buf = &mut flash_out[..flash_len];
                            let q_ref = &q_mat[..];

                            #[cfg_attr(not(feature = "parallel"), allow(unused_imports))]
                            use crate::par::{
                                IndexedParallelIterator, ParallelIterator, ParallelSliceMut,
                            };
                            flash_buf.par_chunks_mut(chunk_size).enumerate().for_each(
                                |(kv_h, chunk)| {
                                    cpu::flash_attention_gqa_cpu(
                                        q_ref,
                                        k_cache,
                                        v_cache,
                                        chunk,
                                        kv_h * group_size,
                                        group_size,
                                        n,
                                        n,
                                        kv_dim,
                                        kv_h * head_dim,
                                        head_dim,
                                        scale,
                                        start_pos,
                                    );
                                },
                            );

                            // Scatter-copy: flash_buf [n_heads, n, head_dim]
                            // → out_proj_input [hs, n] stride-n.
                            // Loop order d-then-j gives sequential writes to
                            // out_proj_input (stride 1) and small-stride reads
                            // from flash_buf (stride head_dim).
                            for kv_h in 0..n_kv_heads {
                                for g in 0..group_size {
                                    let h = kv_h * group_size + g;
                                    let src_base = kv_h * chunk_size + g * n * head_dim;
                                    for d in 0..head_dim {
                                        let row_idx = (h * head_dim + d) * n;
                                        for j in 0..n {
                                            out_proj_input[row_idx + j] =
                                                flash_buf[src_base + j * head_dim + d];
                                        }
                                    }
                                }
                            }
                        } else if use_tq {
                            // TurboQuant path: per-token attention using the
                            // compressed KV cache. Re-extract post-RoPE Q from
                            // q_mat for each token.
                            state.scratch.scores.reserve((start_pos + n) * group_size);
                            for j in 0..n {
                                let q = &mut state.scratch.q[..hs];
                                for i in 0..hs {
                                    q[i] = q_mat[i * n + j];
                                }

                                let (ck, cv, k_cache, v_cache) = match &state.layers[layer] {
                                    LayerState::Attention {
                                        key_cache,
                                        value_cache,
                                        compressed_keys,
                                        compressed_values,
                                        ..
                                    } => (
                                        compressed_keys.as_ref(),
                                        compressed_values.as_ref(),
                                        key_cache.as_slice(),
                                        value_cache.as_slice(),
                                    ),
                                    _ => unreachable!(),
                                };

                                let use_tq_keys = will_read_compressed_kv && ck.is_some();
                                let use_tq_values = will_read_compressed_kv && cv.is_some();

                                let seq_len = if use_tq_keys {
                                    ck.unwrap().seq_len()
                                } else if use_tq_values {
                                    cv.unwrap().seq_len()
                                } else {
                                    k_cache.len() / kv_dim
                                };
                                let attn_out = &mut state.scratch.attn_out[..hs];
                                let q = &state.scratch.q[..hs];
                                let scores = &mut state.scratch.scores;

                                let rotation = tq_rotation.unwrap();
                                let cfg_tq = tq_config.unwrap();
                                let qr_scratch = state.tq_query_scratch.as_mut().unwrap();
                                if use_tq_keys {
                                    turboquant::rotate_queries(
                                        q,
                                        cfg.n_heads,
                                        head_dim,
                                        rotation,
                                        qr_scratch,
                                    );
                                }
                                scores.resize(seq_len * group_size, 0.0);
                                for kv_h in 0..n_kv_heads {
                                    let group_start = kv_h * group_size;
                                    let kv_h_offset = kv_h * head_dim;

                                    if use_tq_keys {
                                        turboquant::attn_scores_turboquant_gqa(
                                            ck.unwrap(),
                                            kv_h,
                                            group_start,
                                            group_size,
                                            scores,
                                            head_dim,
                                            scale,
                                            seq_len,
                                            cfg_tq,
                                            qr_scratch,
                                        );
                                    } else {
                                        for g in 0..group_size {
                                            let h = group_start + g;
                                            let q_head = &q[h * head_dim..(h + 1) * head_dim];
                                            let head_scores =
                                                &mut scores[g * seq_len..(g + 1) * seq_len];
                                            cpu::attn_scores(
                                                q_head,
                                                k_cache,
                                                head_scores,
                                                kv_dim,
                                                kv_h_offset,
                                                head_dim,
                                                scale,
                                                seq_len,
                                            );
                                        }
                                    }

                                    for g in 0..group_size {
                                        let head_scores =
                                            &mut scores[g * seq_len..(g + 1) * seq_len];
                                        cpu::softmax_inplace(head_scores);
                                    }

                                    if use_tq_values {
                                        turboquant::attn_values_turboquant_gqa(
                                            cv.unwrap(),
                                            kv_h,
                                            group_start,
                                            group_size,
                                            scores,
                                            attn_out,
                                            head_dim,
                                            seq_len,
                                            rotation,
                                            cfg_tq,
                                        );
                                    } else {
                                        for g in 0..group_size {
                                            let h = group_start + g;
                                            let head_scores =
                                                &scores[g * seq_len..(g + 1) * seq_len];
                                            cpu::attn_values(
                                                head_scores,
                                                v_cache,
                                                &mut attn_out[h * head_dim..(h + 1) * head_dim],
                                                kv_dim,
                                                kv_h_offset,
                                                head_dim,
                                                seq_len,
                                            );
                                        }
                                    }
                                }

                                for i in 0..hs {
                                    out_proj_input[i * n + j] = attn_out[i];
                                }
                            }
                        } else {
                            // Short-prompt f32 fallback: naive per-token
                            // attention (no tiling, no rayon). Faster than
                            // flash attention when n < FLASH_ATTN_THRESHOLD
                            // because the attention portion is trivially small.
                            let (k_cache, v_cache) = match &state.layers[layer] {
                                LayerState::Attention {
                                    key_cache,
                                    value_cache,
                                    ..
                                } => (key_cache.as_slice(), value_cache.as_slice()),
                                _ => unreachable!(),
                            };
                            state.scratch.scores.reserve((start_pos + n) * group_size);
                            for j in 0..n {
                                let seq_len = (start_pos + j + 1).min(k_cache.len() / kv_dim);
                                // Q is already post-RoPE in q_mat from Pass A;
                                // re-extract into scratch for the naive path.
                                for i in 0..hs {
                                    state.scratch.q[i] = q_mat[i * n + j];
                                }
                                let q = &state.scratch.q[..hs];
                                let attn_out = &mut state.scratch.attn_out[..hs];
                                let scores = &mut state.scratch.scores;
                                scores.resize(seq_len, 0.0);
                                for h in 0..cfg.n_heads {
                                    let kv_h = h / group_size;
                                    let q_head = &q[h * head_dim..(h + 1) * head_dim];
                                    let kv_h_offset = kv_h * head_dim;
                                    cpu::attn_scores(
                                        q_head,
                                        k_cache,
                                        scores,
                                        kv_dim,
                                        kv_h_offset,
                                        head_dim,
                                        scale,
                                        seq_len,
                                    );
                                    cpu::softmax_inplace(scores);
                                    cpu::attn_values(
                                        scores,
                                        v_cache,
                                        &mut attn_out[h * head_dim..(h + 1) * head_dim],
                                        kv_dim,
                                        kv_h_offset,
                                        head_dim,
                                        seq_len,
                                    );
                                }

                                for i in 0..hs {
                                    out_proj_input[i * n + j] = attn_out[i];
                                }
                            }
                        }

                        // Phase 3: Batch output projection GEMM
                        #[cfg(not(feature = "blas"))]
                        transformer::quantize_columns(
                            &out_proj_input,
                            hs,
                            n,
                            &mut col,
                            &mut bq_scales,
                            &mut bq_quants,
                        );
                        #[cfg(feature = "blas")]
                        {
                            transformer::try_blas_prefill_gemm(
                                &self.gguf,
                                attn_output_ref,
                                &out_proj_input,
                                &mut block_out,
                                hs,
                                n,
                                hs,
                                &mut state.scratch.dequant_weight_scratch,
                            );
                        }
                        #[cfg(not(feature = "blas"))]
                        transformer::gemm_preq(
                            &self.gguf,
                            attn_output_ref,
                            &bq_scales,
                            &bq_quants,
                            &mut block_out,
                            hs,
                            n,
                            hs,
                        );

                        // LoRA on the output projection — applied to `block_out`
                        // BEFORE the residual add; input is the attention output
                        // `[hs×n]`.
                        if let Some(lora) = &lora
                            && let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnOutput)
                        {
                            crate::lora::apply_prefill(
                                t,
                                &out_proj_input,
                                &mut block_out,
                                n,
                                &mut state.scratch.lora_tmp,
                            );
                        }
                        true
                    } else {
                        false
                    }
                }
            };

            // Fallback: per-token sequential path. Used on x86_64-no-blas
            // (no batched path compiled), and on any target where the
            // batched path saw mixed dtypes and bailed (`used_block_gemm
            // = false`).
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
            let need_block_fallback = !used_block_gemm;
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas")))]
            let need_block_fallback = true;
            if need_block_fallback {
                block_out.fill(0.0);
                for j in 0..n {
                    for i in 0..hs {
                        col[i] = normed[i * n + j];
                    }
                    #[cfg(target_arch = "aarch64")]
                    Self::quantize_to_scratch(&col, state);

                    if is_conv {
                        self.forward_conv_block(layer, &col, state);
                    } else {
                        self.forward_attn_block(layer, &col, start_pos + j, state);
                    }

                    for i in 0..hs {
                        block_out[i * n + j] = state.scratch.out[i];
                    }
                }
            }

            // Log the BLOCK OUTPUT magnitude (pre-residual) so we
            // can see how much each layer contributes vs the prior
            // residual. Together with the post-block log this lets
            // us identify whether the magnitude growth is from a
            // misbehaving block, an oversize residual, or both.
            // The `if debug_hidden` guard keeps the per-layer
            // `format!` allocations off the hot path when the env
            // var isn't set.
            if debug_hidden {
                let block_kind = if cfg.block_types[layer] == BlockType::GatedConv {
                    "conv"
                } else {
                    "attn"
                };
                log_rms(
                    &format!("layer {layer} ({block_kind}) block-out"),
                    &block_out,
                );
            }

            // Residual: hidden += block_out
            for i in 0..hs * n {
                hidden[i] += block_out[i];
            }
            if debug_hidden {
                let block_kind = if cfg.block_types[layer] == BlockType::GatedConv {
                    "conv"
                } else {
                    "attn"
                };
                log_rms(&format!("layer {layer} ({block_kind}) post-block"), &hidden);
            }

            // FFN pre-norm each column
            for j in 0..n {
                for i in 0..hs {
                    ffn_col[i] = hidden[i * n + j];
                }
                cpu::rmsnorm(
                    &mut ffn_col,
                    &self.ffn_norm_weights[layer],
                    cfg.rms_norm_eps,
                );
                for i in 0..hs {
                    ffn_input[i * n + j] = ffn_col[i];
                }
            }

            // FFN: batched GEMM (reads weights once for all n tokens) for the
            // dtypes `batched_gemm_supports` admits. Available on aarch64 (NEON
            // `gemm_preq`), on x86_64 with runtime avx2+fma (int8
            // `gemm_preq`), and on any target with `feature = "blas"` (BLAS
            // SGEMM via `try_blas_prefill_gemm`). Require all three projections
            // (gate/up/down) to be batchable — a mixed-dtype FFN block would
            // leave later matrices silently uncomputed in the batched path and
            // produce wrong outputs.
            let refs = &self.layer_refs[layer];
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
            let used_gemm = if [
                ("ffn_gate", &refs.ffn_gate),
                ("ffn_up", &refs.ffn_up),
                ("ffn_down", &refs.ffn_down),
            ]
            .into_iter()
            .fold(true, |ok, (name, w)| {
                if transformer::batched_gemm_supports(w.dtype, w.k) {
                    ok
                } else {
                    transformer::warn_unbatchable(name, w.dtype);
                    false
                }
            }) {
                // Pre-quantize all n columns to Q8_0 — only needed for the NEON fallback.
                #[cfg(not(feature = "blas"))]
                transformer::quantize_columns(
                    &ffn_input,
                    hs,
                    n,
                    &mut col,
                    &mut bq_scales,
                    &mut bq_quants,
                );

                // Gate + Up via batched GEMM
                #[cfg(feature = "blas")]
                {
                    transformer::try_blas_prefill_gemm(
                        &self.gguf,
                        &refs.ffn_gate,
                        &ffn_input,
                        &mut gate_mat,
                        is,
                        n,
                        hs,
                        &mut state.scratch.dequant_weight_scratch,
                    );
                    transformer::try_blas_prefill_gemm(
                        &self.gguf,
                        &refs.ffn_up,
                        &ffn_input,
                        &mut up_mat,
                        is,
                        n,
                        hs,
                        &mut state.scratch.dequant_weight_scratch,
                    );
                }
                #[cfg(not(feature = "blas"))]
                {
                    transformer::gemm_preq(
                        &self.gguf,
                        &refs.ffn_gate,
                        &bq_scales,
                        &bq_quants,
                        &mut gate_mat,
                        is,
                        n,
                        hs,
                    );
                    transformer::gemm_preq(
                        &self.gguf,
                        &refs.ffn_up,
                        &bq_scales,
                        &bq_quants,
                        &mut up_mat,
                        is,
                        n,
                        hs,
                    );
                }

                // LoRA on gate/up — BEFORE the SiLU+mul, input is the normed FFN
                // input `[hs×n]`.
                if let Some(lora) = &lora {
                    if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnGate) {
                        crate::lora::apply_prefill(
                            t,
                            &ffn_input,
                            &mut gate_mat[..is * n],
                            n,
                            &mut state.scratch.lora_tmp,
                        );
                    }
                    if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnUp) {
                        crate::lora::apply_prefill(
                            t,
                            &ffn_input,
                            &mut up_mat[..is * n],
                            n,
                            &mut state.scratch.lora_tmp,
                        );
                    }
                }

                // Fused SiLU+mul (row-major is×n)
                cpu::silu_mul_inplace(&mut gate_mat[..is * n], &up_mat[..is * n]);

                // Re-quantize gate_mat columns for down projection — only needed for NEON fallback.
                #[cfg(not(feature = "blas"))]
                transformer::quantize_columns(
                    &gate_mat,
                    is,
                    n,
                    &mut inter_col,
                    &mut dq_scales,
                    &mut dq_quants,
                );

                // Down via batched GEMM
                #[cfg(feature = "blas")]
                {
                    transformer::try_blas_prefill_gemm(
                        &self.gguf,
                        &refs.ffn_down,
                        &gate_mat,
                        &mut ffn_out,
                        hs,
                        n,
                        is,
                        &mut state.scratch.dequant_weight_scratch,
                    );
                }
                #[cfg(not(feature = "blas"))]
                transformer::gemm_preq(
                    &self.gguf,
                    &refs.ffn_down,
                    &dq_scales,
                    &dq_quants,
                    &mut ffn_out,
                    hs,
                    n,
                    is,
                );

                // LoRA on the down projection — applied to `ffn_out` BEFORE the
                // residual add; input is the SiLU⊙up product in `gate_mat` `[is×n]`.
                if let Some(lora) = &lora
                    && let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnDown)
                {
                    crate::lora::apply_prefill(
                        t,
                        &gate_mat[..is * n],
                        &mut ffn_out,
                        n,
                        &mut state.scratch.lora_tmp,
                    );
                }
                true
            } else {
                false
            };

            // Fallback: per-token GEMV. Used on x86_64-no-blas (no batched
            // path compiled), and on any target where the FFN weights
            // weren't all batchable (`used_gemm = false`).
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
            let need_fallback = !used_gemm;
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas")))]
            let need_fallback = true;
            if need_fallback {
                ffn_out.fill(0.0);
                for j in 0..n {
                    for i in 0..hs {
                        col[i] = ffn_input[i * n + j];
                    }

                    #[cfg(target_arch = "aarch64")]
                    {
                        Self::quantize_to_scratch(&col, state);
                        self.gemv_preq(
                            &refs.ffn_gate,
                            &col,
                            &state.scratch.q8_scales,
                            &state.scratch.q8_quants,
                            &mut gate_col,
                        );
                        self.gemv_preq(
                            &refs.ffn_up,
                            &col,
                            &state.scratch.q8_scales,
                            &state.scratch.q8_quants,
                            &mut up_col,
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        self.gemv(&refs.ffn_gate, &col, &mut gate_col);
                        self.gemv(&refs.ffn_up, &col, &mut up_col);
                    }

                    // LoRA on gate/up (per-token decode hook) — this fallback loop
                    // doesn't route through `forward_ffn_block`, so apply it here.
                    if let Some(lora) = &lora {
                        if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnGate) {
                            crate::lora::apply_decode(
                                t,
                                &col,
                                &mut gate_col,
                                &mut state.scratch.lora_tmp,
                            );
                        }
                        if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnUp) {
                            crate::lora::apply_decode(
                                t,
                                &col,
                                &mut up_col,
                                &mut state.scratch.lora_tmp,
                            );
                        }
                    }

                    cpu::silu_mul_inplace(&mut gate_col, &up_col);

                    #[cfg(target_arch = "aarch64")]
                    {
                        Self::quantize_to_scratch(&gate_col, state);
                        self.gemv_preq(
                            &refs.ffn_down,
                            &gate_col,
                            &state.scratch.q8_scales,
                            &state.scratch.q8_quants,
                            &mut out_col,
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    self.gemv(&refs.ffn_down, &gate_col, &mut out_col);

                    // LoRA on the down projection (per-token decode hook) — input is
                    // the SiLU⊙up product in `gate_col`.
                    if let Some(lora) = &lora
                        && let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnDown)
                    {
                        crate::lora::apply_decode(
                            t,
                            &gate_col,
                            &mut out_col,
                            &mut state.scratch.lora_tmp,
                        );
                    }

                    for i in 0..hs {
                        ffn_out[i * n + j] = out_col[i];
                    }
                }
            }

            if debug_hidden {
                log_rms(&format!("layer {layer} ffn-out"), &ffn_out);
            }
            // Second residual
            for i in 0..hs * n {
                hidden[i] += ffn_out[i];
            }
            if debug_hidden {
                log_rms(&format!("layer {layer} post-ffn"), &hidden);
            }
        }

        // seq_len tracks total tokens processed. The conv/attn blocks handle
        // per-token KV cache growth internally. We need seq_len = start_pos + n
        // at the end for the decode phase to continue from the right position.
        // Note: seq_len was NOT incremented inside the block functions — only
        // the single-token forward() does that. So set it here:
        state.seq_len = start_pos + n;

        // Extract last token, apply output norm + projection
        let mut last_hidden = vec![0.0f32; hs];
        for i in 0..hs {
            last_hidden[i] = hidden[i * n + (n - 1)];
        }
        cpu::rmsnorm(&mut last_hidden, &self.output_norm_weight, cfg.rms_norm_eps);

        let mut logits = vec![0.0f32; cfg.vocab_size];
        #[cfg(target_arch = "aarch64")]
        {
            Self::quantize_to_scratch(&last_hidden, state);
            self.gemv_preq(
                &self.embd_ref,
                &last_hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut logits,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(&self.embd_ref, &last_hidden, &mut logits);

        logits
    }

    /// Lock-free body of `Model::forward_prefill` — does the actual
    /// embed + layer loop without consulting the prefix cache.
    /// `forward_prefill` wraps this with cache lookup/insert; cache
    /// hits bypass embedding the prefix tokens entirely and re-enter
    /// here with `start_pos = prefix_len` to prefill only the suffix.
    pub(crate) fn forward_prefill_inner(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let n = tokens.len();
        assert!(
            !tokens.is_empty(),
            "forward_prefill_inner requires at least one token"
        );

        // Embed all tokens → hidden[hs × n] with stride n (token j at
        // indices [j, n+j, 2n+j, ...]). Layer loop + output projection
        // is shared with `forward_prefill_from_embeddings` via
        // `prefill_layers_and_logits`.
        let mut hidden = vec![0.0f32; hs * n];
        let mut emb_buf = vec![0.0f32; hs];
        for (j, &token_id) in tokens.iter().enumerate() {
            let token_id = token_id as usize;
            assert!(
                token_id < self.embd_ref.m,
                "token_id {token_id} out of range for vocab size {}",
                self.embd_ref.m
            );
            self.dequantize_row_into(&self.embd_ref, token_id, &mut emb_buf);
            for i in 0..hs {
                hidden[i * n + j] = emb_buf[i];
            }
        }

        self.prefill_layers_and_logits(hidden, n, start_pos, state)
    }
}

impl Model for Lfm2Model {
    fn supports_hidden_states(&self) -> bool {
        true
    }

    /// Per-token post-final-norm hidden states, row-major `[n * hidden_size]`.
    /// Mirrors [`Self::forward`]'s embedding path (dequantize → `run_layers`,
    /// which applies the output norm) minus the logit projection — so the result
    /// is the same post-`output_norm` vector, matching llama.cpp `--pooling none`.
    /// Per-token (not batched): LFM2's batched prefill is entangled with the
    /// prefix cache, and this stateless one-shot path must not touch it; batched
    /// capture is a possible perf follow-up. `state` must start cleared at pos 0.
    fn hidden_states(&self, tokens: &[u32], state: &mut InferenceState) -> Vec<f32> {
        assert!(
            !tokens.is_empty(),
            "hidden_states requires at least one token"
        );
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let mut out = Vec::with_capacity(tokens.len() * hs);
        // Reuse one embedding buffer across tokens instead of a per-token Vec.
        let mut hidden = vec![0.0f32; hs];
        for &token in tokens {
            let token_id = token as usize;
            assert!(
                token_id < cfg.vocab_size,
                "token_id {token_id} out of range (vocab_size={})",
                cfg.vocab_size
            );
            self.dequantize_row_into(&self.embd_ref, token_id, &mut hidden);
            // `run_layers` ropes at `pos == seq_len` and appends one cell,
            // bumping seq_len; a cleared state walks positions 0..n.
            let pos = state.seq_len;
            self.run_layers(&mut hidden, pos, state);
            out.extend_from_slice(&hidden);
        }
        out
    }

    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        assert_eq!(tokens.len(), 1, "LFM2 forward expects single token");
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        assert!(
            token_id < cfg.vocab_size,
            "token_id {token_id} out of range (vocab_size={})",
            cfg.vocab_size
        );

        // 1. Embedding lookup → layers → output norm
        let mut hidden = self.dequantize_row(&self.embd_ref, token_id);
        self.run_layers(&mut hidden, pos, state);

        // 2. Output projection (tied embeddings)
        let mut logits = vec![0.0f32; cfg.vocab_size];
        #[cfg(target_arch = "aarch64")]
        {
            Self::quantize_to_scratch(&hidden, state);
            self.gemv_preq(
                &self.embd_ref,
                &hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut logits,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(&self.embd_ref, &hidden, &mut logits);

        logits
    }

    fn supports_embedding_input(&self) -> bool {
        true
    }

    fn forward_from_embedding(
        &self,
        embedding: &[f32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let mut hidden = embedding.to_vec();
        let pos = state.seq_len;
        self.run_layers(&mut hidden, pos, state);

        // Output projection (tied embeddings)
        let mut logits = vec![0.0f32; cfg.vocab_size];
        #[cfg(target_arch = "aarch64")]
        {
            Self::quantize_to_scratch(&hidden, state);
            self.gemv_preq(
                &self.embd_ref,
                &hidden,
                &state.scratch.q8_scales,
                &state.scratch.q8_quants,
                &mut logits,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.gemv(&self.embd_ref, &hidden, &mut logits);

        logits
    }

    fn forward_embedding(
        &self,
        tokens: &[u32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        assert_eq!(tokens.len(), 1);
        let token_id = tokens[0] as usize;
        let mut hidden = self.dequantize_row(&self.embd_ref, token_id);
        let pos = state.seq_len;
        self.run_layers(&mut hidden, pos, state);
        hidden
    }

    fn forward_hidden_from_embedding(
        &self,
        embedding: &[f32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let mut hidden = embedding.to_vec();
        let pos = state.seq_len;
        self.run_layers(&mut hidden, pos, state);
        hidden
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

        // Cache participation gate: only on a fresh prefill
        // (`start_pos == 0`). Continuation prefills (chunked /
        // mid-sequence) carry KV state from the prior chunk so a
        // cache restore would clobber it. TurboQuant-compressed
        // states are now supported via `LayerSnapshot::AttentionCompressed`
        // (the `!is_compressed()` exclusion was lifted in this PR).
        //
        // An active LoRA also disables the prefix cache: the batched
        // `forward_prefill_inner` still runs and applies the adapter in-batch (via
        // `apply_prefill`), but the resulting KV is adapter-specific, and the cache
        // key doesn't include the adapter — caching it would let a later base-model
        // (or different-adapter) prefill restore adapted KV. Skip both the lookup
        // and the insert while LoRA is attached.
        let cache_eligible = start_pos == 0 && state.lora.is_none();

        if cache_eligible {
            let hit = self
                .prefix_cache
                .lock()
                .expect("prefix_cache mutex poisoned")
                .find_longest_prefix(tokens);
            if let Some((snapshot, prefix_len)) = hit {
                // Compatibility gate: snapshot's compression mode
                // must match the live state's. Cross-mode restores
                // would panic in `InferenceState::restore`
                // (compressed snapshot into `None` slots, or
                // uncompressed snapshot into a TurboQuant-
                // configured state with mismatched scratch / rotation
                // shape). Three live-state modes:
                //
                // - fully uncompressed → match `Attention` snapshots.
                // - fully compressed   → match `AttentionCompressed`.
                // - mixed-mode (one side compressed, the other not):
                //   `snapshot()` returns `None` so the cache never
                //   holds an entry that matches; both branches below
                //   reject.
                //
                // `state.is_compressed()` (any-side-compressed) is
                // too loose for the compressed branch: a mixed-mode
                // state would erroneously match a fully-compressed
                // snapshot and panic in `restore`. Use
                // `is_fully_compressed` for the compressed branch,
                // `!is_compressed` for the uncompressed branch.
                // Today `model_fingerprint` doesn't include the
                // compression flags, so a `--cache-dir` shared
                // between TurboQuant and uncompressed runs of the
                // same model file relies on this gate; v2 could
                // fold compression into the fingerprint.
                let compatible = if snapshot.is_compressed() {
                    state.is_fully_compressed() && state.is_compressed()
                } else {
                    !state.is_compressed()
                };
                if !compatible {
                    // skip; fall through to cold prefill.
                } else if prefix_len < tokens.len() && prefix_len > 0 {
                    // Strict-prefix-only: a `prefix_len == tokens.len()`
                    // hit would force `use_len = tokens.len() - 1`, but
                    // the restored conv rolling buffer reflects "after
                    // all tokens" — re-running the last token would
                    // advance the conv buffer one position past where
                    // it should be (conv layers don't gate on
                    // seq_len). Skip full hits + fall through to cold
                    // prefill. Same fix wgpu got in PR #120; tracked
                    // as 8f on the punch list.
                    let use_len = prefix_len;
                    state.restore(&snapshot);
                    let logits = self.forward_prefill_inner(&tokens[use_len..], use_len, state);
                    if let Some(snap) = state.snapshot() {
                        self.prefix_cache
                            .lock()
                            .expect("prefix_cache mutex poisoned")
                            .insert(tokens, snap);
                    }
                    return logits;
                }
            }
        }

        let logits = self.forward_prefill_inner(tokens, start_pos, state);
        if cache_eligible && let Some(snap) = state.snapshot() {
            self.prefix_cache
                .lock()
                .expect("prefix_cache mutex poisoned")
                .insert(tokens, snap);
        }
        logits
    }

    fn configure_cache(&self, config: crate::kv_cache::KvCacheConfig) {
        *self
            .prefix_cache
            .lock()
            .expect("prefix_cache mutex poisoned") =
            KvPrefixCache::new(config, &self.config, &format!("cpu:{}", self.model_id));
    }

    fn forward_prefill_from_embeddings(
        &self,
        embeddings: &[f32],
        n_tokens: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let n = n_tokens;
        assert!(
            n > 0,
            "forward_prefill_from_embeddings requires at least one frame"
        );
        assert_eq!(
            embeddings.len(),
            n * hs,
            "embeddings.len() ({}) != n_tokens ({}) * hidden_size ({})",
            embeddings.len(),
            n,
            hs
        );

        // An active LoRA is applied in-batch by `prefill_layers_and_logits` (via
        // `apply_prefill` after each projection GEMM), so embedding-input
        // (multimodal) spans get the adapter too — no per-frame fallback needed.

        // Transpose row-major embeddings (frame j at [j*hs..(j+1)*hs])
        // into column-major hidden (token j's channel i at [i*n + j]) —
        // same layout `forward_prefill` builds via the embed-table
        // lookup. After this, the layer loop + output projection in
        // `prefill_layers_and_logits` is identical to the token path.
        let mut hidden = vec![0.0f32; hs * n];
        for j in 0..n {
            let frame = &embeddings[j * hs..(j + 1) * hs];
            for i in 0..hs {
                hidden[i * n + j] = frame[i];
            }
        }

        self.prefill_layers_and_logits(hidden, n, start_pos, state)
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn turboquant_supported(&self) -> bool {
        let head_dim = self.config.hidden_size / self.config.n_heads;
        head_dim.is_power_of_two()
    }

    fn supports_kv_shift(&self) -> bool {
        // CPU LFM2 implements shift with RoPE re-rotation. Metal's
        // override stays at the trait default (false) until its
        // GPU-side shift shader lands.
        true
    }

    fn shift_kv(&self, state: &mut crate::kv_cache::InferenceState, n_keep: usize, shift: usize) {
        let head_dim = self.config.hidden_size / self.config.n_heads;
        state.shift_kv_with_rope(
            n_keep,
            shift,
            self.config.rope_theta,
            head_dim,
            &self.config.kv_heads_per_layer,
            crate::backend::cpu::RopeType::Neox,
            None,
        );
    }
}

// ── GPU weight source ───────────────────────────────────────────────────────
//
// Drives the wgpu loader (`gpu_lfm2.rs`) for LFM2. Conv layers expose
// `conv_*` refs; attention layers expose `attn_*` refs + QK-norm. LFM2 has no
// QKV bias / untied output / Llama-3 freq-factors, uses NEOX RoPE, identity
// scalars, and supports the batched-prefill GPU path.
#[cfg(any(
    feature = "gpu",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
impl crate::model::gpu_weight_source::GpuWeightSource for Lfm2Model {
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
        Lfm2Model::attn_q_norm_weight(self, layer)
    }
    fn attn_k_norm_weight(&self, layer: usize) -> Option<&[f32]> {
        Lfm2Model::attn_k_norm_weight(self, layer)
    }
    fn conv_weight(&self, layer: usize) -> Option<&[f32]> {
        Lfm2Model::conv_weight(self, layer)
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

    fn weight_bytes(&self, wref: &WeightRef) -> &[u8] {
        transformer::weight_data(&self.gguf, wref)
    }
    fn dequantize_weight(&self, wref: &WeightRef) -> Vec<f32> {
        transformer::dequantize_weight(&self.gguf, wref)
    }

    fn output_ref(&self) -> Option<&WeightRef> {
        None
    }
    fn ffn_gate_ref(&self, layer: usize) -> &WeightRef {
        &self.layer_refs[layer].ffn_gate
    }
    fn ffn_up_ref(&self, layer: usize) -> &WeightRef {
        &self.layer_refs[layer].ffn_up
    }
    fn ffn_down_ref(&self, layer: usize) -> &WeightRef {
        &self.layer_refs[layer].ffn_down
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

    fn rope_type(&self) -> crate::backend::cpu::RopeType {
        crate::backend::cpu::RopeType::Neox
    }
    fn supports_batched_prefill(&self) -> bool {
        true
    }
}
