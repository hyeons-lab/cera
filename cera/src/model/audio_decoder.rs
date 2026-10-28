//! Audio decoder for LFM2.5-Audio: vocoder GGUF loading + weight structures.
//!
//! Three sub-models loaded from the vocoder GGUF:
//! 1. DecoderModel — samples 8 audio codes per frame from LLM embedding
//! 2. DepthformerModel — small 6-layer attention-only transformer (backbone of DecoderModel)
//! 3. Detokenizer — codes → spectrogram → PCM (not loaded here yet — Phase 5)

use std::sync::Arc;

use anyhow::{Context, Result};

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::model::weights::MmapWeight;

/// Whether an accelerator's depthformer should run: it must exist
/// (`supports`), and either be trusted by default (`default_on`, see
/// [`AudioAccelerator::depthformer_default_on`]) or be explicitly requested
/// (`SessionConfig::gpu_depthformer` / `CERA_GPU_DF=1`). One definition for the
/// engine and the CLI. (The wasm/WebGPU path requests it unconditionally by
/// design: the browser has no `CERA_GPU_DF`, and it always ran the WebGPU
/// depthformer.)
pub fn accelerated_depthformer_enabled(supports: bool, default_on: bool, requested: bool) -> bool {
    supports && (default_on || requested)
}

/// Hardware-accelerated audio backend (Qualcomm Hexagon NPU, Apple Metal, WebGPU).
/// Implementations provide offloaded execution for the depthformer (code sampling)
/// and detokenizer (spectrum).
pub trait AudioAccelerator: Send + Sync {
    /// Sample 8 audio codes from an LLM embedding using the depthformer.
    fn sample_audio_frame(&self, embedding: &[f32], temperature: f32, top_k: usize) -> [i32; 8];

    /// Async version of [`Self::sample_audio_frame`] for WebGPU / browser wasm.
    fn sample_audio_frame_async<'a>(
        &'a self,
        embedding: &'a [f32],
        temperature: f32,
        top_k: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<[i32; 8]>> + Send + 'a>> {
        Box::pin(async move {
            // Same drain discipline as the sync engine paths: discard a
            // stale record, run the bare call, and fail the future on a
            // fresh fault instead of returning zero/stale output as `Ok`.
            let _ = self.take_audio_error();
            let out = self.sample_audio_frame(embedding, temperature, top_k);
            if let Some(e) = self.take_audio_error() {
                return Err(e.into());
            }
            Ok(out)
        })
    }

    /// Async depthformer sampling using a GPU hidden state buffer directly without CPU roundtrip.
    #[cfg(feature = "gpu")]
    fn sample_audio_frame_from_gpu_hidden_async<'a>(
        &'a self,
        _hidden_buf: &'a wgpu::Buffer,
        _temperature: f32,
        _top_k: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<[i32; 8]>> + Send + 'a>> {
        Box::pin(async move {
            anyhow::bail!("sample_audio_frame_from_gpu_hidden_async not implemented")
        })
    }

    /// Convert 8 audio codes to spectrum [n_frames × n_fft_bins × 2].
    fn detokenize_to_spectrum(&self, cpu_weights: &DetokenizerWeights, codes: &[i32]) -> Vec<f32>;

    /// Async version of [`Self::detokenize_to_spectrum`].
    fn detokenize_to_spectrum_async<'a>(
        &'a self,
        cpu_weights: &'a DetokenizerWeights,
        codes: &'a [i32],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>>> + Send + 'a>> {
        Box::pin(async move {
            let _ = self.take_audio_error();
            let out = self.detokenize_to_spectrum(cpu_weights, codes);
            if let Some(e) = self.take_audio_error() {
                return Err(e.into());
            }
            Ok(out)
        })
    }

    /// Reset depthformer KV caches (called per audio frame).
    fn reset_depthformer(&self);

    /// Reset detokenizer state (conv buffers + KV caches, called per generation).
    fn reset_detokenizer(&self);

    /// Seed the internal depthformer PRNG used for stochastic codebook sampling.
    fn seed_rng(&self, _seed: u64) {}

    /// Whether [`Self::sample_audio_frame`] is actually implemented here.
    fn supports_depthformer(&self) -> bool;

    /// Whether [`Self::detokenize_to_spectrum`] supports `lfm3a_gen` 12-codebook FSQ detokenizers.
    fn supports_lfm3a_gen_detokenizer(&self) -> bool {
        false
    }

    /// Whether the accelerator's depthformer is trusted enough to run without
    /// an explicit opt-in. Defaults to `false` (the GPU depthformers stay
    /// opt-in); a backend overrides this once its codes match the CPU path.
    fn depthformer_default_on(&self) -> bool {
        false
    }

    /// Convert the accumulated spectrum `[n_frames × n_fft_bins × 2]` (log-mag,
    /// angle) to PCM via ISTFT. Defaults to the CPU `istft_to_pcm`; GPU backends
    /// override to run the iDFT-matmul + windowed overlap-add on-device.
    fn istft_to_pcm(&self, spectrum: &[f32], n_fft: usize, hop_length: usize) -> Vec<f32> {
        istft_to_pcm(spectrum, n_fft, hop_length)
    }

    /// Async version of [`Self::istft_to_pcm`].
    fn istft_to_pcm_async<'a>(
        &'a self,
        spectrum: &'a [f32],
        n_fft: usize,
        hop_length: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>>> + Send + 'a>> {
        Box::pin(async move {
            let _ = self.take_audio_error();
            let out = self.istft_to_pcm(spectrum, n_fft, hop_length);
            if let Some(e) = self.take_audio_error() {
                return Err(e.into());
            }
            Ok(out)
        })
    }

    /// Drain a sticky backend fault recorded by the last bare call
    /// ([`Self::sample_audio_frame`], [`Self::detokenize_to_spectrum`],
    /// [`Self::istft_to_pcm`]), first-wins across the decoder's contexts.
    /// The sync engine paths discard-then-drain around every call; the
    /// async defaults below drain before returning `Ok`, so an async
    /// fault surfaces as `Err`, never as zero/stale output. Backends
    /// without fallible hardware (test doubles, CPU-only drivers) keep
    /// the default `None`.
    fn take_audio_error(&self) -> Option<crate::CeraError> {
        None
    }

    /// Attempt to acquire an exclusive session lease for multi-frame generation.
    /// Returns true if acquired, or false if another session is actively generating audio.
    fn try_acquire_session(&self) -> bool {
        true
    }

    /// Release the exclusive session lease acquired by [`Self::try_acquire_session`].
    fn release_session(&self) {}
}

/// Whether a GGUF carries the `lfm3a_gen` audio decoder (12-codebook MaskGIT depthformer).
/// The one definition [`AudioDecoderWeights::from_gguf`] and [`is_lfm3a_gen_gguf`] share.
pub fn has_lfm3a_gen_decoder(gguf: &GgufFile) -> bool {
    gguf.metadata.contains_key("clip.audio_gen.block_count")
        || gguf
            .metadata
            .contains_key("clip.audio_gen.code.block_count")
        || gguf.metadata.contains_key("clip.gen.audio.block_count")
        || gguf.get_str("clip.gen.audio.projector_type") == Some("lfm3a_gen")
        || gguf.tensors.contains_key("a.gen.code.tok_embd.weight")
        || gguf.tensors.contains_key("a.gen.code.embd.weight")
        || gguf.tensors.contains_key("a.gen.code.proj_in.weight")
        || gguf.tensors.contains_key("a.gen.code.in_proj.weight")
        || gguf.tensors.contains_key("a.gen.depth_linear.weight")
}

/// Whether a GGUF carries the `lfm3a_gen` FSQ + ISTFT detokenizer. The one definition
/// [`DetokenizerWeights::from_gguf`] and [`is_lfm3a_gen_gguf`] share.
pub fn has_lfm3a_gen_detokenizer(gguf: &GgufFile) -> bool {
    gguf.tensors.contains_key("a.gen.wav.tfm.in_proj.weight")
        || gguf.tensors.contains_key("a.gen.wav.tfm_in_proj.weight")
        || gguf.tensors.contains_key("a.detok.tfm_in_proj.weight")
        || gguf
            .tensors
            .contains_key("a.gen.wav.quant.first.out_proj.weight")
        || gguf.tensors.contains_key("a.detok.quant_first_out.weight")
        || gguf
            .tensors
            .contains_key("a.gen.wav.quant_first_out.weight")
}

/// Returns `true` when a GGUF file carries `lfm3a_gen` audio generation weights
/// (the decoder and/or the detokenizer).
pub fn is_lfm3a_gen_gguf(gguf: &GgufFile) -> bool {
    has_lfm3a_gen_decoder(gguf) || has_lfm3a_gen_detokenizer(gguf)
}

/// Try to construct an accelerated audio decoder backend for the given vocoder GGUF file.
///
/// For `lfm3a_gen` models, only the Hexagon NPU detokenizer path (`Hexagon`, `Npu`, or `Auto`)
/// is probed; `Metal`, `Gpu`, and `Cpu` return `None`. Returns `None` when the chosen backend's
/// feature is not compiled or when the device/context cannot be created.
pub fn build_audio_accelerator(
    gguf: &Arc<GgufFile>,
    backend: crate::engine::BackendPreference,
) -> Option<Arc<dyn AudioAccelerator>> {
    use crate::engine::BackendPreference as BP;
    if is_lfm3a_gen_gguf(gguf) {
        return match backend {
            BP::Hexagon | BP::Npu | BP::Auto => try_hexagon_audio_decoder(gguf),
            _ => None,
        };
    }
    match backend {
        BP::Cpu => None,
        // Hexagon is the only NPU with an audio accelerator today; `Npu` takes
        // the same path and gains other vendors here as they grow one.
        BP::Hexagon | BP::Npu => try_hexagon_audio_decoder(gguf),
        BP::Metal => try_metal_audio_decoder(gguf),
        BP::Gpu => try_wgpu_audio_decoder(gguf),
        BP::Auto => try_metal_audio_decoder(gguf)
            .or_else(|| try_hexagon_audio_decoder(gguf))
            .or_else(|| try_wgpu_audio_decoder(gguf)),
    }
}

#[cfg(feature = "hexagon")]
fn try_hexagon_audio_decoder(gguf: &Arc<GgufFile>) -> Option<Arc<dyn AudioAccelerator>> {
    crate::model::audio_decoder_hexagon::try_hexagon_audio_decoder(gguf)
        .map(|d| d as Arc<dyn AudioAccelerator>)
}

#[cfg(not(feature = "hexagon"))]
fn try_hexagon_audio_decoder(_gguf: &Arc<GgufFile>) -> Option<Arc<dyn AudioAccelerator>> {
    None
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
fn try_metal_audio_decoder(gguf: &Arc<GgufFile>) -> Option<Arc<dyn AudioAccelerator>> {
    let dummy_path = std::path::Path::new("");
    match crate::model::metal_audio_decoder::MetalAudioDecoder::from_gguf(gguf, dummy_path) {
        Ok(d) => {
            tracing::info!("audio decoder: using native Metal backend");
            Some(Arc::new(d))
        }
        Err(e) => {
            tracing::warn!("audio decoder: Metal backend unavailable: {e:#}");
            None
        }
    }
}

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
fn try_metal_audio_decoder(_gguf: &Arc<GgufFile>) -> Option<Arc<dyn AudioAccelerator>> {
    None
}

#[cfg(feature = "gpu")]
fn try_wgpu_audio_decoder(gguf: &Arc<GgufFile>) -> Option<Arc<dyn AudioAccelerator>> {
    let dummy_path = std::path::Path::new("");
    match crate::model::wgpu_audio_decoder::WgpuAudioDecoder::from_gguf(gguf, dummy_path) {
        Ok(d) => {
            tracing::info!("audio decoder: using WGPU backend");
            Some(Arc::new(d))
        }
        Err(e) => {
            tracing::warn!("audio decoder: WGPU backend unavailable: {e:#}");
            None
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn try_wgpu_audio_decoder(_gguf: &Arc<GgufFile>) -> Option<Arc<dyn AudioAccelerator>> {
    None
}

/// Configuration for the depthformer (small transformer inside the decoder).
#[derive(Debug, Clone)]
pub struct DepthformerConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head: usize,
    pub ffn_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_freq_base: f32,
    pub max_seq_len: usize,
}

/// Configuration for the decoder model (8-codebook LFM2A or 12-codebook LFM3A-Gen sampling).
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub n_codebook: usize,
    pub n_vocab: usize,
    pub n_embd: usize, // LLM embedding dimension (2048)
    pub rms_norm_eps: f32,
}

/// Per-layer weights for one depthformer layer.
/// Uses MmapWeight for GEMV to match ggml's Q4_0 computation path.
pub struct DepthformerLayerWeights {
    pub operator_norm: Vec<f32>,
    pub wqkv: MmapWeight,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub wo: MmapWeight,
    pub ffn_norm: Vec<f32>,
    pub w1: MmapWeight, // gate
    pub w2: MmapWeight, // down
    pub w3: MmapWeight, // up
}

/// Per-codebook embedding layer weights.
pub struct CodebookWeights {
    pub embedding: MmapWeight,
    pub norm: Vec<f32>,
    pub to_logits: MmapWeight,
}

/// All weights for the audio decoder (loaded from vocoder or single-file mmproj GGUF).
pub struct AudioDecoderWeights {
    /// Whether this decoder was loaded from the `lfm3a_gen` schema (the 12-codebook MaskGIT
    /// decoder) rather than the LFM2A depthformer schema. Set by the loader, not inferred from
    /// the codebook count, so a file of the other schema is never routed to the wrong sampler.
    pub lfm3a_gen: bool,
    pub depthformer_config: DepthformerConfig,
    pub decoder_config: DecoderConfig,
    pub depthformer_layers: Vec<DepthformerLayerWeights>,
    pub depth_linear_w: MmapWeight,
    pub depth_linear_b: Vec<f32>,
    pub depth_embeddings: Vec<CodebookWeights>,
    pub audio_embedding: CodebookWeights,
    /// Text/audio alternation this vocoder asks for in interleaved turns.
    pub interleave: crate::audio_engine::InterleaveCadence,
}

impl AudioDecoderWeights {
    /// Returns `true` when this decoder uses the 12-codebook `lfm3a_gen` MaskGIT architecture.
    pub fn is_lfm3a_gen(&self) -> bool {
        self.lfm3a_gen
    }

    /// Load all decoder weights from the vocoder or mmproj GGUF.
    pub fn from_gguf(gguf: &Arc<GgufFile>) -> Result<Self> {
        if has_lfm3a_gen_decoder(gguf) {
            return Self::from_gguf_lfm3a_gen(gguf);
        }
        // a detokenizer-only file would fall into the legacy depthformer loader below and
        // report one of its missing keys, which names the wrong schema
        anyhow::ensure!(
            !has_lfm3a_gen_detokenizer(gguf) || gguf.metadata.contains_key("depthformer_n_layer"),
            "this GGUF carries an lfm3a_gen detokenizer but no audio decoder"
        );

        // Hyperparameters from GGUF metadata.
        let n_layer = gguf
            .get_u32("depthformer_n_layer")
            .context("missing depthformer_n_layer")? as usize;
        let n_embd = gguf
            .get_u32("depthformer_n_embd")
            .context("missing depthformer_n_embd")? as usize;

        // Derive head config from qkv_proj shape.
        // qkv_proj.weight is [n_embd, (n_head + 2*n_head_kv) * head_dim]
        let qkv = MmapWeight::from_gguf(gguf, "depthformer.layers.0.operator.qkv_proj.weight")?;
        let q_norm_w =
            gguf.get_tensor("depthformer.layers.0.operator.attention.q_layernorm.weight")?;
        let n_embd_head = q_norm_w.shape()[0]; // head_dim from q_norm shape
        let qkv_out = qkv.rows; // (n_head + 2*n_head_kv) * head_dim
        let n_head = 32; // hardcoded per decoder.cpp config
        let n_head_kv = 8;
        assert_eq!(
            qkv_out,
            (n_head + 2 * n_head_kv) * n_embd_head,
            "qkv_proj shape mismatch"
        );

        // FFN dim from w1 shape.
        let w1_0 = MmapWeight::from_gguf(gguf, "depthformer.layers.0.feed_forward.w1.weight")?;
        let ffn_dim = w1_0.rows;

        let depthformer_config = DepthformerConfig {
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            n_embd_head,
            ffn_dim,
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            max_seq_len: 8,
        };

        // Load depthformer layers.
        let mut depthformer_layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let prefix = format!("depthformer.layers.{i}");
            depthformer_layers.push(DepthformerLayerWeights {
                operator_norm: gguf
                    .get_tensor(&format!("{prefix}.operator_norm.weight"))?
                    .try_to_f32_vec()?,
                wqkv: MmapWeight::from_gguf(gguf, &format!("{prefix}.operator.qkv_proj.weight"))?,
                q_norm: gguf
                    .get_tensor(&format!("{prefix}.operator.attention.q_layernorm.weight"))?
                    .try_to_f32_vec()?,
                k_norm: gguf
                    .get_tensor(&format!("{prefix}.operator.attention.k_layernorm.weight"))?
                    .try_to_f32_vec()?,
                wo: MmapWeight::from_gguf(gguf, &format!("{prefix}.operator.out_proj.weight"))?,
                ffn_norm: gguf
                    .get_tensor(&format!("{prefix}.ffn_norm.weight"))?
                    .try_to_f32_vec()?,
                w1: MmapWeight::from_gguf(gguf, &format!("{prefix}.feed_forward.w1.weight"))?,
                w2: MmapWeight::from_gguf(gguf, &format!("{prefix}.feed_forward.w2.weight"))?,
                w3: MmapWeight::from_gguf(gguf, &format!("{prefix}.feed_forward.w3.weight"))?,
            });
        }

        // Decoder model weights.
        let depth_linear_w = MmapWeight::from_gguf(gguf, "depth_linear.weight")?;
        let depth_linear_b = gguf.get_tensor("depth_linear.bias")?.try_to_f32_vec()?;

        let n_codebook = 8;
        let mut depth_embeddings = Vec::with_capacity(n_codebook);
        for i in 0..n_codebook {
            let prefix = format!("depth_embeddings.{i}");
            depth_embeddings.push(CodebookWeights {
                embedding: MmapWeight::from_gguf(gguf, &format!("{prefix}.embedding.weight"))?,
                norm: gguf
                    .get_tensor(&format!("{prefix}.embedding_norm.weight"))?
                    .try_to_f32_vec()?,
                to_logits: MmapWeight::from_gguf(gguf, &format!("{prefix}.to_logits.weight"))?,
            });
        }

        let audio_embedding = CodebookWeights {
            embedding: MmapWeight::from_gguf(gguf, "audio_embedding.embedding.weight")?,
            norm: gguf
                .get_tensor("audio_embedding.embedding_norm.weight")?
                .try_to_f32_vec()?,
            to_logits: MmapWeight::from_gguf(gguf, "audio_embedding.to_logits.weight")?,
        };

        // Derive decoder config from weight shapes.
        let n_embd_llm = depth_linear_w.cols; // LLM embedding dim (2048)
        let n_vocab = depth_embeddings[0].to_logits.rows; // 2049

        let decoder_config = DecoderConfig {
            n_codebook,
            n_vocab,
            n_embd: n_embd_llm,
            rms_norm_eps: 1e-5,
        };

        Ok(Self {
            lfm3a_gen: false,
            depthformer_config,
            decoder_config,
            depthformer_layers,
            depth_linear_w,
            depth_linear_b,
            depth_embeddings,
            audio_embedding,
            interleave: crate::audio_engine::InterleaveCadence::from_gguf(gguf),
        })
    }

    /// Load `lfm3a_gen` 12-codebook MaskGIT depthformer weights.
    /// Supports both `a.gen.code.*` (upstream `lfm3.py` / `lfm3a-gen.cpp`) and `a.gen.*` tensor schemas.
    fn from_gguf_lfm3a_gen(gguf: &Arc<GgufFile>) -> Result<Self> {
        let n_layer = gguf
            .get_u32("clip.gen.audio.block_count")
            .or_else(|| gguf.get_u32("clip.audio_gen.code.block_count"))
            .or_else(|| gguf.get_u32("clip.audio_gen.block_count"))
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                let counted = (0..256usize)
                    .take_while(|&i| {
                        gguf.tensors
                            .contains_key(&format!("a.gen.code.blk.{i}.ln1.weight"))
                            || gguf
                                .tensors
                                .contains_key(&format!("a.gen.blk.{i}.operator_norm.weight"))
                            || gguf
                                .tensors
                                .contains_key(&format!("a.gen.blk.{i}.ln1.weight"))
                    })
                    .count();
                if counted > 0 { counted } else { 4 }
            });
        let tok_embd_shape = gguf
            .tensors
            .get("a.gen.code.embd.weight")
            .or_else(|| gguf.tensors.get("a.gen.code.tok_embd.weight"))
            .map(|ti| ti.shape.as_slice());
        let n_codebook = gguf
            .get_u32("clip.audio_gen.code.num_codebooks")
            .or_else(|| gguf.get_u32("clip.audio_gen.n_codebooks"))
            .or_else(|| gguf.get_u32("clip.gen.audio.num_codebooks"))
            .map(|v| v as usize)
            .or_else(|| tok_embd_shape.and_then(|s| if s.len() == 3 { Some(s[2]) } else { None }))
            .or_else(|| {
                gguf.tensors
                    .get("a.gen.code.head_norm.weight")
                    .or_else(|| gguf.tensors.get("a.gen.code.in_norm.weight"))
                    .and_then(|ti| {
                        if ti.shape.len() == 2 {
                            Some(ti.shape[1])
                        } else {
                            None
                        }
                    })
            })
            .unwrap_or(12);
        let n_vocab = gguf
            .get_u32("clip.audio_gen.code.vocab_size")
            .or_else(|| gguf.get_u32("clip.audio_gen.vocab_size"))
            .or_else(|| gguf.get_u32("clip.gen.audio.vocab_size"))
            .map(|v| v as usize)
            .or_else(|| {
                tok_embd_shape.and_then(|s| {
                    if s.len() == 3 {
                        Some(s[1])
                    } else if s.len() == 2 && n_codebook > 0 && s[1].is_multiple_of(n_codebook) {
                        Some(s[1] / n_codebook)
                    } else {
                        None
                    }
                })
            })
            .unwrap_or(257);
        let rms_norm_eps = gguf
            .get_f32("clip.gen.audio.attention.layer_norm_epsilon")
            .or_else(|| gguf.get_f32("clip.audio_gen.code.attention.layer_norm_epsilon"))
            .or_else(|| gguf.get_f32("clip.audio_gen.layer_norm_epsilon"))
            .unwrap_or(1e-5);
        let rope_freq_base = gguf
            .get_f32("clip.gen.audio.rope.freq_base")
            .or_else(|| gguf.get_f32("clip.audio_gen.code.rope.freq_base"))
            .or_else(|| gguf.get_f32("clip.audio_gen.rope.freq_base"))
            .unwrap_or(1_000_000.0);

        anyhow::ensure!(
            (1..=256).contains(&n_layer),
            "lfm3a_gen depthformer n_layer ({n_layer}) must be in 1..=256"
        );
        anyhow::ensure!(
            (1..=64).contains(&n_codebook),
            "lfm3a_gen n_codebook ({n_codebook}) must be in 1..=64"
        );
        anyhow::ensure!(
            (2..=65536).contains(&n_vocab),
            "lfm3a_gen n_vocab ({n_vocab}) must be in 2..=65536"
        );
        anyhow::ensure!(
            rms_norm_eps.is_finite() && rms_norm_eps > 0.0,
            "lfm3a_gen rms_norm_eps ({rms_norm_eps}) must be positive and finite"
        );
        anyhow::ensure!(
            rope_freq_base.is_finite() && rope_freq_base > 0.0,
            "lfm3a_gen rope_freq_base ({rope_freq_base}) must be positive and finite"
        );

        let depth_linear_w = MmapWeight::from_gguf(gguf, "a.gen.code.proj_in.weight")
            .or_else(|_| MmapWeight::from_gguf(gguf, "a.gen.code.in_proj.weight"))
            .or_else(|_| MmapWeight::from_gguf(gguf, "a.gen.depth_linear.weight"))
            .context("loading lfm3a_gen depth_linear weight")?;
        // absent means zeros; present but unreadable is an error, not a zero bias
        let depth_linear_b = crate::model::audio_encoder::load_optional_vec_f32(
            gguf,
            &[
                "a.gen.code.proj_in.bias".to_string(),
                "a.gen.code.in_proj.bias".to_string(),
                "a.gen.depth_linear.bias".to_string(),
            ],
        )?;
        let depth_linear_b = if depth_linear_b.is_empty() {
            vec![0.0; depth_linear_w.rows]
        } else {
            depth_linear_b
        };

        let n_embd = gguf
            .get_u32("clip.gen.audio.embedding_length")
            .or_else(|| gguf.get_u32("clip.audio_gen.code.embedding_length"))
            .or_else(|| gguf.get_u32("clip.audio_gen.embedding_length"))
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                if depth_linear_w.rows.is_multiple_of(n_codebook) && depth_linear_w.rows > 1024 {
                    depth_linear_w.rows / n_codebook
                } else {
                    depth_linear_w.rows
                }
            });

        let mut n_head = gguf
            .get_u32("clip.gen.audio.attention.head_count")
            .or_else(|| gguf.get_u32("clip.audio_gen.code.attention.head_count"))
            .or_else(|| gguf.get_u32("clip.audio_gen.head_count"))
            .unwrap_or(8) as usize;
        let mut n_head_kv = [
            "clip.gen.audio.attention.head_count_kv",
            "clip.audio_gen.code.attention.head_count_kv",
            "clip.audio_gen.head_count_kv",
        ]
        .into_iter()
        .find_map(|k| {
            gguf.get_u32(k).map(|v| v as usize).or_else(|| {
                gguf.get_i32_array(k)
                    .and_then(|arr| arr.into_iter().find(|&v| v > 0))
                    .map(|v| v as usize)
            })
        })
        .unwrap_or(2);
        anyhow::ensure!(n_embd > 0, "lfm3a_gen n_embd ({n_embd}) must be > 0");
        anyhow::ensure!(
            depth_linear_w.cols > 0
                && (depth_linear_w.rows == n_embd || depth_linear_w.rows == n_codebook * n_embd),
            "lfm3a_gen depth_linear_w shape ({}, {}) invalid for n_embd={n_embd}, n_codebook={n_codebook}",
            depth_linear_w.rows,
            depth_linear_w.cols
        );
        anyhow::ensure!(
            depth_linear_b.len() == depth_linear_w.rows,
            "lfm3a_gen depth_linear_b len ({}) != depth_linear_w.rows ({})",
            depth_linear_b.len(),
            depth_linear_w.rows
        );

        let mut n_embd_head = 0usize;
        let mut ffn_dim = gguf
            .get_u32("clip.gen.audio.feed_forward_length")
            .or_else(|| gguf.get_u32("clip.audio_gen.code.feed_forward_length"))
            .or_else(|| gguf.get_u32("clip.audio_gen.feed_forward_length"))
            .unwrap_or(1792) as usize;

        let mut depthformer_layers = Vec::new();
        let mut seen_attn_block = false;
        for i in 0..n_layer {
            let pfx_code = format!("a.gen.code.blk.{i}");
            let pfx_gen = format!("a.gen.blk.{i}");

            let operator_norm = gguf
                .get_tensor(&format!("{pfx_code}.ln1.weight"))
                .or_else(|_| gguf.get_tensor(&format!("{pfx_gen}.operator_norm.weight")))
                .or_else(|_| gguf.get_tensor(&format!("{pfx_gen}.ln1.weight")))
                .with_context(|| format!("loading lfm3a_gen block {i} operator_norm"))?
                .try_to_f32_vec()?;
            anyhow::ensure!(
                operator_norm.len() == n_embd,
                "lfm3a_gen block {i} operator_norm len ({}) != n_embd ({n_embd})",
                operator_norm.len()
            );
            let ffn_norm = gguf
                .get_tensor(&format!("{pfx_code}.ln2.weight"))
                .or_else(|_| gguf.get_tensor(&format!("{pfx_gen}.ffn_norm.weight")))
                .or_else(|_| gguf.get_tensor(&format!("{pfx_gen}.ln2.weight")))
                .with_context(|| format!("loading lfm3a_gen block {i} ffn_norm"))?
                .try_to_f32_vec()?;
            anyhow::ensure!(
                ffn_norm.len() == n_embd,
                "lfm3a_gen block {i} ffn_norm len ({}) != n_embd ({n_embd})",
                ffn_norm.len()
            );

            let w1 = MmapWeight::from_gguf(gguf, &format!("{pfx_code}.ffn_gate.weight"))
                .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_gen}.w1.weight")))
                .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_gen}.ffn_gate.weight")))
                .with_context(|| format!("loading lfm3a_gen block {i} w1"))?;
            let w2 = MmapWeight::from_gguf(gguf, &format!("{pfx_code}.ffn_down.weight"))
                .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_gen}.w2.weight")))
                .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_gen}.ffn_down.weight")))
                .with_context(|| format!("loading lfm3a_gen block {i} w2"))?;
            let w3 = MmapWeight::from_gguf(gguf, &format!("{pfx_code}.ffn_up.weight"))
                .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_gen}.w3.weight")))
                .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_gen}.ffn_up.weight")))
                .with_context(|| format!("loading lfm3a_gen block {i} w3"))?;
            if i == 0 {
                ffn_dim = w1.rows;
            }
            anyhow::ensure!(
                ffn_dim > 0
                    && w1.rows == ffn_dim
                    && w1.cols == n_embd
                    && w3.rows == ffn_dim
                    && w3.cols == n_embd
                    && w2.rows == n_embd
                    && w2.cols == ffn_dim,
                "lfm3a_gen block {i} FFN shapes w1=({}, {}), w2=({}, {}), w3=({}, {}) mismatch for n_embd={n_embd}, ffn_dim={ffn_dim}",
                w1.rows,
                w1.cols,
                w2.rows,
                w2.cols,
                w3.rows,
                w3.cols
            );

            let is_shortconv = gguf
                .tensors
                .contains_key(&format!("{pfx_code}.conv_dw.weight"))
                || gguf
                    .tensors
                    .contains_key(&format!("{pfx_gen}.shortconv.conv.weight"))
                || gguf
                    .tensors
                    .contains_key(&format!("{pfx_gen}.conv_dw.weight"));

            if is_shortconv {
                let dw_t = gguf
                    .get_tensor(&format!("{pfx_code}.conv_dw.weight"))
                    .or_else(|_| gguf.get_tensor(&format!("{pfx_gen}.shortconv.conv.weight")))
                    .or_else(|_| gguf.get_tensor(&format!("{pfx_gen}.conv_dw.weight")))
                    .with_context(|| format!("loading lfm3a_gen block {i} conv_dw"))?;
                let dw_vec = dw_t.try_to_f32_vec()?;
                anyhow::ensure!(
                    !dw_vec.is_empty() && dw_vec.len().is_multiple_of(n_embd),
                    "lfm3a_gen block {i} conv_dw len ({}) must be a positive multiple of n_embd ({n_embd})",
                    dw_vec.len()
                );
                let wqkv = MmapWeight::from_gguf(gguf, &format!("{pfx_code}.conv_pw1.weight"))
                    .or_else(|_| {
                        MmapWeight::from_gguf(gguf, &format!("{pfx_gen}.shortconv.in_proj.weight"))
                    })
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_gen}.conv_pw1.weight")))
                    .with_context(|| format!("loading lfm3a_gen block {i} conv_pw1"))?;
                let wo = MmapWeight::from_gguf(gguf, &format!("{pfx_code}.conv_pw2.weight"))
                    .or_else(|_| {
                        MmapWeight::from_gguf(gguf, &format!("{pfx_gen}.shortconv.out_proj.weight"))
                    })
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_gen}.conv_pw2.weight")))
                    .with_context(|| format!("loading lfm3a_gen block {i} conv_pw2"))?;
                anyhow::ensure!(
                    wqkv.rows == 3 * n_embd
                        && wqkv.cols == n_embd
                        && wo.rows == n_embd
                        && wo.cols == n_embd,
                    "lfm3a_gen block {i} ShortConv shapes in_proj=({}, {}), out_proj=({}, {}) invalid for n_embd={n_embd}",
                    wqkv.rows,
                    wqkv.cols,
                    wo.rows,
                    wo.cols
                );
                depthformer_layers.push(DepthformerLayerWeights {
                    operator_norm,
                    wqkv,
                    q_norm: dw_vec,
                    k_norm: Vec::new(), // Empty k_norm marks a ShortConv block in lfm3a_gen.
                    wo,
                    ffn_norm,
                    w1,
                    w2,
                    w3,
                });
            } else {
                let q_norm = gguf
                    .get_tensor(&format!("{pfx_code}.attn_q_norm.weight"))
                    .or_else(|_| gguf.get_tensor(&format!("{pfx_gen}.q_norm.weight")))
                    .or_else(|_| gguf.get_tensor(&format!("{pfx_gen}.attn_q_norm.weight")))
                    .with_context(|| format!("loading lfm3a_gen block {i} q_norm"))?
                    .try_to_f32_vec()?;
                let k_norm = gguf
                    .get_tensor(&format!("{pfx_code}.attn_k_norm.weight"))
                    .or_else(|_| gguf.get_tensor(&format!("{pfx_gen}.k_norm.weight")))
                    .or_else(|_| gguf.get_tensor(&format!("{pfx_gen}.attn_k_norm.weight")))
                    .with_context(|| format!("loading lfm3a_gen block {i} k_norm"))?
                    .try_to_f32_vec()?;
                anyhow::ensure!(
                    !q_norm.is_empty()
                        && q_norm.len() == k_norm.len()
                        && n_embd.is_multiple_of(q_norm.len()),
                    "lfm3a_gen block {i} q_norm/k_norm lens ({}, {}) invalid for n_embd={n_embd}",
                    q_norm.len(),
                    k_norm.len()
                );
                let blk_head_dim = q_norm.len();

                let (wqkv, blk_n_head, blk_n_head_kv) = if gguf
                    .tensors
                    .contains_key(&format!("{pfx_gen}.wqkv.weight"))
                {
                    let fused = MmapWeight::from_gguf(gguf, &format!("{pfx_gen}.wqkv.weight"))
                        .with_context(|| format!("loading {pfx_gen}.wqkv.weight"))?;
                    anyhow::ensure!(
                        fused.cols == n_embd && fused.rows.is_multiple_of(blk_head_dim),
                        "lfm3a_gen block {i} wqkv shape ({}, {}) invalid for n_embd={n_embd}, head_dim={blk_head_dim}",
                        fused.rows,
                        fused.cols
                    );
                    let total_heads = fused.rows / blk_head_dim;
                    let h = n_embd / blk_head_dim;
                    anyhow::ensure!(
                        total_heads > h && (total_heads - h).is_multiple_of(2),
                        "lfm3a_gen block {i} wqkv total_heads ({total_heads}) invalid for n_head={h}"
                    );
                    let h_kv = (total_heads - h) / 2;
                    anyhow::ensure!(
                        h_kv > 0 && h.is_multiple_of(h_kv),
                        "lfm3a_gen block {i} derived n_head_kv ({h_kv}) does not divide n_head ({h})"
                    );
                    (fused, h, h_kv)
                } else {
                    let wq = gguf
                        .get_tensor(&format!("{pfx_code}.attn_q.weight"))
                        .or_else(|_| gguf.get_tensor(&format!("{pfx_gen}.attn_q.weight")))
                        .with_context(|| format!("loading lfm3a_gen block {i} attn_q"))?;
                    let wk = gguf
                        .get_tensor(&format!("{pfx_code}.attn_k.weight"))
                        .or_else(|_| gguf.get_tensor(&format!("{pfx_gen}.attn_k.weight")))
                        .with_context(|| format!("loading lfm3a_gen block {i} attn_k"))?;
                    let wv = gguf
                        .get_tensor(&format!("{pfx_code}.attn_v.weight"))
                        .or_else(|_| gguf.get_tensor(&format!("{pfx_gen}.attn_v.weight")))
                        .with_context(|| format!("loading lfm3a_gen block {i} attn_v"))?;
                    let q_f32 = wq.try_to_f32_vec()?;
                    let k_f32 = wk.try_to_f32_vec()?;
                    let v_f32 = wv.try_to_f32_vec()?;
                    anyhow::ensure!(
                        q_f32.len() == n_embd * n_embd
                            && k_f32.len().is_multiple_of(n_embd)
                            && v_f32.len() == k_f32.len(),
                        "lfm3a_gen block {i} Q/K/V element counts ({}, {}, {}) invalid for n_embd={n_embd}",
                        q_f32.len(),
                        k_f32.len(),
                        v_f32.len()
                    );
                    let q_rows = q_f32.len() / n_embd;
                    let k_rows = k_f32.len() / n_embd;
                    anyhow::ensure!(
                        q_rows.is_multiple_of(blk_head_dim) && k_rows.is_multiple_of(blk_head_dim),
                        "lfm3a_gen block {i} Q/K rows ({q_rows}, {k_rows}) not divisible by head_dim ({blk_head_dim})"
                    );
                    let h = q_rows / blk_head_dim;
                    let h_kv = k_rows / blk_head_dim;
                    anyhow::ensure!(
                        h > 0 && h_kv > 0 && h.is_multiple_of(h_kv) && n_embd == h * blk_head_dim,
                        "lfm3a_gen block {i} derived n_head ({h}) / n_head_kv ({h_kv}) invalid for n_embd={n_embd}"
                    );
                    let gate_code = format!("{pfx_code}.attn_gate.weight");
                    let gate_gen = format!("{pfx_gen}.attn_gate.weight");
                    let gate_f32 = if gguf.tensors.contains_key(&gate_code)
                        || gguf.tensors.contains_key(&gate_gen)
                    {
                        let gw = gguf
                            .get_tensor(&gate_code)
                            .or_else(|_| gguf.get_tensor(&gate_gen))
                            .with_context(|| format!("loading lfm3a_gen block {i} attn_gate"))?
                            .try_to_f32_vec()?;
                        anyhow::ensure!(
                            gw.len() == h * n_embd || gw.len() == q_f32.len(),
                            "lfm3a_gen block {i} attn_gate len ({}) invalid for n_head={h}, n_embd={n_embd}",
                            gw.len()
                        );
                        gw
                    } else {
                        Vec::new()
                    };
                    let mut cat = Vec::with_capacity(
                        q_f32.len() + k_f32.len() + v_f32.len() + gate_f32.len(),
                    );
                    cat.extend_from_slice(&q_f32);
                    cat.extend_from_slice(&k_f32);
                    cat.extend_from_slice(&v_f32);
                    cat.extend_from_slice(&gate_f32);
                    let total_rows = cat.len() / n_embd;
                    (MmapWeight::from_owned_f32(cat, total_rows, n_embd), h, h_kv)
                };
                if !seen_attn_block {
                    n_embd_head = blk_head_dim;
                    n_head = blk_n_head;
                    n_head_kv = blk_n_head_kv;
                    seen_attn_block = true;
                } else {
                    anyhow::ensure!(
                        blk_head_dim == n_embd_head
                            && blk_n_head == n_head
                            && blk_n_head_kv == n_head_kv,
                        "lfm3a_gen block {i} attention dims (head_dim={blk_head_dim}, n_head={blk_n_head}, n_head_kv={blk_n_head_kv}) mismatch earlier attention block (head_dim={n_embd_head}, n_head={n_head}, n_head_kv={n_head_kv})"
                    );
                }

                let wo = MmapWeight::from_gguf(gguf, &format!("{pfx_code}.attn_out.weight"))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_gen}.wo.weight")))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_gen}.attn_out.weight")))
                    .with_context(|| format!("loading lfm3a_gen block {i} wo"))?;
                anyhow::ensure!(
                    wo.rows == n_embd && wo.cols == n_embd,
                    "lfm3a_gen block {i} wo shape ({}, {}) != ({n_embd}, {n_embd})",
                    wo.rows,
                    wo.cols
                );

                depthformer_layers.push(DepthformerLayerWeights {
                    operator_norm,
                    wqkv,
                    q_norm,
                    k_norm,
                    wo,
                    ffn_norm,
                    w1,
                    w2,
                    w3,
                });
            }
        }

        if !seen_attn_block {
            anyhow::ensure!(
                n_head > 0
                    && n_head_kv > 0
                    && n_embd.is_multiple_of(n_head)
                    && n_head.is_multiple_of(n_head_kv),
                "lfm3a_gen n_embd ({n_embd}) must be divisible by n_head ({n_head}) and n_head divisible by n_head_kv ({n_head_kv})"
            );
            n_embd_head = n_embd / n_head;
        }
        anyhow::ensure!(
            n_head > 0
                && n_head_kv > 0
                && n_embd_head > 0
                && n_embd == n_head * n_embd_head
                && n_head.is_multiple_of(n_head_kv),
            "lfm3a_gen invalid attention dimensions: n_embd={n_embd}, n_head={n_head}, n_head_kv={n_head_kv}, head_dim={n_embd_head}"
        );

        let mut depth_embeddings = Vec::with_capacity(n_codebook);
        let mut shared_output_norm = Vec::new();
        if gguf.tensors.contains_key("a.gen.code.tok_embd.weight")
            || gguf.tensors.contains_key("a.gen.code.embd.weight")
        {
            let tok_embd_t = gguf
                .get_tensor("a.gen.code.tok_embd.weight")
                .or_else(|_| gguf.get_tensor("a.gen.code.embd.weight"))
                .context("loading a.gen.code.tok_embd.weight / a.gen.code.embd.weight")?;
            let tok_all = tok_embd_t.try_to_f32_vec()?;
            let out_all = if gguf.tensors.contains_key("a.gen.code.out_proj.weight") {
                gguf.get_tensor("a.gen.code.out_proj.weight")
                    .context("loading a.gen.code.out_proj.weight")?
                    .try_to_f32_vec()?
            } else {
                tok_all.clone()
            };
            let out_norm_v = gguf
                .get_tensor("a.gen.code.output_norm.weight")
                .context("loading a.gen.code.output_norm.weight")?
                .try_to_f32_vec()?;
            anyhow::ensure!(
                out_norm_v.len() == n_embd,
                "a.gen.code.output_norm.weight len ({}) != n_embd ({n_embd})",
                out_norm_v.len()
            );
            let head_norms_opt = if gguf.tensors.contains_key("a.gen.code.in_norm.weight")
                || gguf.tensors.contains_key("a.gen.code.head_norm.weight")
            {
                let hn = gguf
                    .get_tensor("a.gen.code.in_norm.weight")
                    .or_else(|_| gguf.get_tensor("a.gen.code.head_norm.weight"))?
                    .try_to_f32_vec()?;
                anyhow::ensure!(
                    hn.len() == n_codebook * n_embd,
                    "a.gen.code.in_norm.weight len ({}) != n_codebook * n_embd ({})",
                    hn.len(),
                    n_codebook * n_embd
                );
                shared_output_norm = out_norm_v.clone();
                Some(hn)
            } else {
                None
            };
            let mask_embd_opt = if gguf.tensors.contains_key("a.gen.code.mask_embd") {
                let me = gguf.get_tensor("a.gen.code.mask_embd")?.try_to_f32_vec()?;
                anyhow::ensure!(
                    me.len() == n_embd,
                    "a.gen.code.mask_embd len ({}) != n_embd ({n_embd})",
                    me.len()
                );
                Some(me)
            } else {
                None
            };
            let cb_elems = n_vocab * n_embd;
            anyhow::ensure!(
                tok_all.len() >= n_codebook * cb_elems && out_all.len() >= n_codebook * cb_elems,
                "a.gen.code.tok_embd / out_proj smaller than {n_codebook} * {n_vocab} * {n_embd}"
            );
            for j in 0..n_codebook {
                let mut slice_tok = tok_all[j * cb_elems..(j + 1) * cb_elems].to_vec();
                if let Some(ref me) = mask_embd_opt {
                    let mask_row_start = (n_vocab - 1) * n_embd;
                    slice_tok[mask_row_start..mask_row_start + n_embd].copy_from_slice(me);
                }
                let slice_out = out_all[j * cb_elems..(j + 1) * cb_elems].to_vec();
                let cb_norm = if let Some(ref hn) = head_norms_opt {
                    hn[j * n_embd..(j + 1) * n_embd].to_vec()
                } else {
                    out_norm_v.clone()
                };
                depth_embeddings.push(CodebookWeights {
                    embedding: MmapWeight::from_owned_f32(slice_tok, n_vocab, n_embd),
                    norm: cb_norm,
                    to_logits: MmapWeight::from_owned_f32(slice_out, n_vocab, n_embd),
                });
            }
        } else {
            for j in 0..n_codebook {
                let pfx = format!("a.gen.emb.{j}");
                let embedding = MmapWeight::from_gguf(gguf, &format!("{pfx}.embd.weight"))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx}.embedding.weight")))
                    .with_context(|| format!("loading {pfx}.embd.weight"))?;
                let norm = gguf
                    .get_tensor(&format!("{pfx}.norm.weight"))
                    .or_else(|_| gguf.get_tensor(&format!("{pfx}.embedding_norm.weight")))
                    .with_context(|| format!("loading {pfx}.norm.weight"))?
                    .try_to_f32_vec()?;
                let to_logits = MmapWeight::from_gguf(gguf, &format!("{pfx}.out.weight"))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx}.to_logits.weight")))
                    .with_context(|| format!("loading {pfx}.out.weight"))?;
                anyhow::ensure!(
                    embedding.rows == n_vocab
                        && embedding.cols == n_embd
                        && norm.len() == n_embd
                        && to_logits.rows == n_vocab
                        && to_logits.cols == n_embd,
                    "lfm3a_gen codebook {j} shapes emb=({}, {}), norm={}, to_logits=({}, {}) invalid for n_vocab={n_vocab}, n_embd={n_embd}",
                    embedding.rows,
                    embedding.cols,
                    norm.len(),
                    to_logits.rows,
                    to_logits.cols
                );
                depth_embeddings.push(CodebookWeights {
                    embedding,
                    norm,
                    to_logits,
                });
            }
        }

        let audio_emb_w = MmapWeight::from_gguf(gguf, "a.gen.code.out_embd.weight")
            .or_else(|_| MmapWeight::from_gguf(gguf, "a.gen.audio_embd.weight"))
            .context("loading a.gen.code.out_embd.weight / a.gen.audio_embd.weight")?;
        anyhow::ensure!(
            audio_emb_w.cols == depth_linear_w.cols && audio_emb_w.rows >= n_codebook * n_vocab,
            "lfm3a_gen audio_embedding shape ({}, {}) invalid for n_codebook*n_vocab={}, llm_dim={}",
            audio_emb_w.rows,
            audio_emb_w.cols,
            n_codebook * n_vocab,
            depth_linear_w.cols
        );
        let audio_embedding = CodebookWeights {
            embedding: audio_emb_w,
            norm: shared_output_norm,
            to_logits: MmapWeight::from_owned_f32(Vec::new(), 0, 0),
        };

        let n_embd_llm = depth_linear_w.cols;
        let depthformer_config = DepthformerConfig {
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            n_embd_head,
            ffn_dim,
            rms_norm_eps,
            rope_freq_base,
            max_seq_len: n_codebook,
        };
        let decoder_config = DecoderConfig {
            n_codebook,
            n_vocab,
            n_embd: n_embd_llm,
            rms_norm_eps,
        };

        Ok(Self {
            lfm3a_gen: true,
            depthformer_config,
            decoder_config,
            depthformer_layers,
            depth_linear_w,
            depth_linear_b,
            depth_embeddings,
            audio_embedding,
            interleave: crate::audio_engine::InterleaveCadence::from_gguf(gguf),
        })
    }
}

// ── Depthformer runtime ─────────────────────────────────────────────────

/// Maximum number of recent audio frames retained for the LFM3A-Gen CTRL repetition penalty.
pub const LFM3A_REP_PENALTY_MAX_FRAMES: usize = 62;

/// Default CTRL repetition penalty factor applied across recent frames in LFM3A-Gen.
pub const LFM3A_REP_PENALTY: f32 = 1.2;

/// KV cache for one depthformer layer.
struct LayerKvCache {
    /// [max_seq x n_head_kv x head_dim] for K and V.
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Advance a 64-bit xorshift state and return a uniform float in `[0, 1)`.
#[inline]
fn xorshift64_next_f32_01(state: &mut u64) -> f32 {
    let mut x = *state;
    if x == 0 {
        x = 0x853c_49e6_748f_ea9b;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    ((x >> 40) as f32) * (1.0 / ((1u64 << 24) as f32))
}

/// Runtime state for the depthformer (KV caches reset per audio frame; `code_history`
/// persists across frames within an audio turn for the LFM3A-Gen repetition penalty).
pub struct DepthformerState {
    kv: Vec<LayerKvCache>,
    n_past: usize,
    cur: Vec<f32>,
    residual: Vec<f32>,
    qkv: Vec<f32>,
    attn_out: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    scores: Vec<f32>,
    q8_scales: Vec<f32>,
    q8_quants: Vec<i8>,
    depthformer_in: Vec<f32>,
    emb_row: Vec<f32>,
    normed: Vec<f32>,
    logits: Vec<f32>,
    indices: Vec<usize>,
    /// Ring buffer of recent generated frames (up to `LFM3A_REP_PENALTY_MAX_FRAMES`)
    /// used by the LFM3A-Gen CTRL repetition penalty.
    code_history: std::collections::VecDeque<Vec<i32>>,
    /// Deterministic xorshift64 PRNG state used for stochastic MaskGIT and top-k sampling.
    rng_state: u64,
}

pub(crate) const DEFAULT_AUDIO_RNG_SEED: u64 = 0x853c_49e6_748f_ea9b;

#[inline]
pub(crate) fn non_zero_rng_seed(seed: u64) -> u64 {
    if seed == 0 {
        DEFAULT_AUDIO_RNG_SEED
    } else {
        seed
    }
}

/// The PRNG state for a caller-supplied seed. A small seed fed straight into xorshift64 starts
/// the stream near zero: the first draw is exactly 0.0 for any seed below 1024 and still tiny
/// (under 1e-3) for every seed below about 2^23, so the first sampled codebook of a run seeded
/// with 1, 42 or 12345 would take the first candidate whatever the distribution. A splitmix64
/// round spreads the seed over all 64 bits first. Entropy-drawn seeds are already uniform and
/// need no mixing.
#[inline]
pub(crate) fn seed_to_rng_state(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    non_zero_rng_seed(z ^ (z >> 31))
}

#[inline]
pub(crate) fn next_xorshift64_f32_01(rng_state: &mut u64) -> f32 {
    let mut x = non_zero_rng_seed(*rng_state);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *rng_state = x;
    ((x >> 40) as u32) as f32 / ((1u32 << 24) as f32)
}

#[cfg(any(
    test,
    feature = "gpu",
    feature = "hexagon",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
#[inline]
pub(crate) fn next_atomic_xorshift64_f32_01(rng_state: &std::sync::atomic::AtomicU64) -> f32 {
    let mut cur = rng_state.load(std::sync::atomic::Ordering::Relaxed);
    loop {
        let mut next = cur;
        let val = next_xorshift64_f32_01(&mut next);
        match rng_state.compare_exchange_weak(
            cur,
            next,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        ) {
            Ok(_) => return val,
            Err(actual) => cur = actual,
        }
    }
}

impl DepthformerState {
    pub fn new(cfg: &DepthformerConfig) -> Self {
        let cache_size = cfg.max_seq_len * cfg.n_head_kv * cfg.n_embd_head;
        let kv = (0..cfg.n_layer)
            .map(|_| LayerKvCache {
                k: vec![0.0; cache_size],
                v: vec![0.0; cache_size],
            })
            .collect();
        let n_embd = cfg.n_embd;
        let hd = cfg.n_embd_head;
        let qkv_dim = cfg.n_head * hd + 2 * (cfg.n_head_kv * hd);
        let max_dim = n_embd.max(cfg.ffn_dim).max(2048);
        let nb = max_dim.div_ceil(32);
        Self {
            kv,
            n_past: 0,
            cur: vec![0.0; n_embd],
            residual: vec![0.0; n_embd],
            qkv: vec![0.0; qkv_dim],
            attn_out: vec![0.0; cfg.n_head * hd],
            proj: vec![0.0; n_embd],
            gate: vec![0.0; cfg.ffn_dim],
            up: vec![0.0; cfg.ffn_dim],
            scores: vec![0.0; cfg.max_seq_len],
            q8_scales: vec![0.0; nb],
            q8_quants: vec![0; max_dim],
            depthformer_in: vec![0.0; n_embd],
            emb_row: vec![0.0; n_embd],
            normed: vec![0.0; n_embd],
            logits: vec![0.0; 2049],
            indices: Vec::with_capacity(2049),
            code_history: std::collections::VecDeque::with_capacity(LFM3A_REP_PENALTY_MAX_FRAMES),
            rng_state: non_zero_rng_seed(rand::random::<u64>()),
        }
    }

    /// Seed the internal xorshift64 PRNG used for stochastic sampling.
    pub fn seed_rng(&mut self, seed: u64) {
        self.rng_state = seed_to_rng_state(seed);
    }

    #[inline]
    fn next_f32_01(&mut self) -> f32 {
        next_xorshift64_f32_01(&mut self.rng_state)
    }

    pub fn reset(&mut self) {
        for layer in &mut self.kv {
            layer.k.fill(0.0);
            layer.v.fill(0.0);
        }
        self.n_past = 0;
    }

    /// Clear cross-frame repetition penalty history (called when starting a new audio turn).
    pub fn reset_history(&mut self) {
        self.code_history.clear();
    }

    /// Record a generated frame into the repetition penalty history buffer.
    pub fn push_code_history(&mut self, codes: Vec<i32>) {
        if self.code_history.len() >= LFM3A_REP_PENALTY_MAX_FRAMES {
            self.code_history.pop_front();
        }
        self.code_history.push_back(codes);
    }
}

/// Apply RoPE (interleaved / LLAMA_ROPE_TYPE_NORM) to a single head.
/// Uses iterative theta multiplication to match ggml's `ggml_rope_cache_init`.
fn apply_rope_interleaved(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let theta_scale = freq_base.powf(-2.0 / head_dim as f32);
    let mut theta = pos as f32;
    for i in 0..head_dim / 2 {
        let (sin_t, cos_t) = theta.sin_cos();
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i] = x0 * cos_t - x1 * sin_t;
        x[2 * i + 1] = x0 * sin_t + x1 * cos_t;
        theta *= theta_scale;
    }
}

/// Apply RoPE (NeoX style / LLAMA_ROPE_TYPE_NEOX) to a single head.
/// Uses iterative theta multiplication to match ggml's `ggml_rope_cache_init`.
fn apply_rope_neox(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let theta_scale = freq_base.powf(-2.0 / head_dim as f32);
    let mut theta = pos as f32;
    for i in 0..head_dim / 2 {
        let (sin_t, cos_t) = theta.sin_cos();
        let x0 = x[i];
        let x1 = x[i + head_dim / 2];
        x[i] = x0 * cos_t - x1 * sin_t;
        x[i + head_dim / 2] = x0 * sin_t + x1 * cos_t;
        theta *= theta_scale;
    }
}

/// Run one forward pass through the depthformer (1 token, n_embd input → n_embd output).
pub fn depthformer_forward(
    weights: &AudioDecoderWeights,
    state: &mut DepthformerState,
    input: &[f32],
) -> Vec<f32> {
    depthformer_forward_into(weights, state, input);
    state.cur.clone()
}

/// In-place depthformer forward pass that avoids allocating a returning Vec.
pub fn depthformer_forward_into(
    weights: &AudioDecoderWeights,
    state: &mut DepthformerState,
    input: &[f32],
) {
    state.cur.copy_from_slice(input);
    depthformer_forward_step(weights, state);
}

/// In-place depthformer forward pass directly reading `state.depthformer_in`.
pub fn depthformer_forward_inplace(weights: &AudioDecoderWeights, state: &mut DepthformerState) {
    state.cur.copy_from_slice(&state.depthformer_in);
    depthformer_forward_step(weights, state);
}

fn depthformer_forward_step(weights: &AudioDecoderWeights, state: &mut DepthformerState) {
    let cfg = &weights.depthformer_config;
    let n_embd = cfg.n_embd;
    let n_head = cfg.n_head;
    let n_kv = cfg.n_head_kv;
    let hd = cfg.n_embd_head;
    let pos = state.n_past;
    let kv_dim = n_kv * hd;
    let q_dim = n_head * hd;
    let k_dim = n_kv * hd;
    let group_size = n_head / n_kv;
    let scale = 1.0 / (hd as f32).sqrt();

    for (il, lw) in weights.depthformer_layers.iter().enumerate() {
        state.residual.copy_from_slice(&state.cur);

        // 1. RMSnorm.
        cpu::rmsnorm(&mut state.cur, &lw.operator_norm, cfg.rms_norm_eps);

        // 2. Fused QKV projection → split.
        let q8_scratch = if lw.wqkv.dtype != crate::tensor::DType::F32
            || lw.w1.dtype != crate::tensor::DType::F32
        {
            crate::backend::cpu::quantize_f32_to_q8_0_into(
                &state.cur,
                &mut state.q8_scales,
                &mut state.q8_quants,
            );
            Some((&mut state.q8_scales, &mut state.q8_quants))
        } else {
            None
        };

        state.qkv[..lw.wqkv.rows].iter_mut().for_each(|v| *v = 0.0);
        crate::backend::cpu::gemv_dispatch(
            lw.wqkv.dtype,
            lw.wqkv.data(),
            &state.cur,
            &mut state.qkv[..lw.wqkv.rows],
            lw.wqkv.rows,
            lw.wqkv.cols,
            q8_scratch,
        );

        // 3. Per-head RMSnorm on Q and K (in-place within qkv).
        for h in 0..n_head {
            let s = &mut state.qkv[h * hd..(h + 1) * hd];
            cpu::rmsnorm(s, &lw.q_norm, cfg.rms_norm_eps);
        }
        for h in 0..n_kv {
            let s = &mut state.qkv[q_dim + h * hd..q_dim + (h + 1) * hd];
            cpu::rmsnorm(s, &lw.k_norm, cfg.rms_norm_eps);
        }

        // 4. RoPE on Q and K (depthformer uses interleaved/NORM style).
        for h in 0..n_head {
            apply_rope_interleaved(
                &mut state.qkv[h * hd..(h + 1) * hd],
                pos,
                hd,
                cfg.rope_freq_base,
            );
        }
        for h in 0..n_kv {
            apply_rope_interleaved(
                &mut state.qkv[q_dim + h * hd..q_dim + (h + 1) * hd],
                pos,
                hd,
                cfg.rope_freq_base,
            );
        }

        // 5. Write K, V to cache at position n_past.
        let kv = &mut state.kv[il];
        for h in 0..n_kv {
            let cache_off = pos * kv_dim + h * hd;
            kv.k[cache_off..cache_off + hd]
                .copy_from_slice(&state.qkv[q_dim + h * hd..q_dim + (h + 1) * hd]);
            kv.v[cache_off..cache_off + hd]
                .copy_from_slice(&state.qkv[q_dim + k_dim + h * hd..q_dim + k_dim + (h + 1) * hd]);
        }

        // 6. Attention: Q×K → softmax → ×V (GQA).
        let seq_len = pos + 1;
        state.attn_out.iter_mut().for_each(|v| *v = 0.0);

        for h in 0..n_head {
            let kv_h = h / group_size;
            let q_head = &state.qkv[h * hd..(h + 1) * hd];
            let kv_h_offset = kv_h * hd;

            let sc = &mut state.scores[..seq_len];
            cpu::attn_scores(q_head, &kv.k, sc, kv_dim, kv_h_offset, hd, scale, seq_len);
            cpu::softmax_inplace(sc);

            let out = &mut state.attn_out[h * hd..(h + 1) * hd];
            cpu::attn_values(sc, &kv.v, out, kv_dim, kv_h_offset, hd, seq_len);
        }

        // 7. Out projection + residual (in-place).
        state.proj.iter_mut().for_each(|v| *v = 0.0);
        lw.wo.gemv_scratch(
            &state.attn_out,
            &mut state.proj,
            Some((&mut state.q8_scales, &mut state.q8_quants)),
        );
        for (c, (r, p)) in state
            .cur
            .iter_mut()
            .zip(state.residual.iter().zip(&state.proj))
        {
            *c = r + p;
        }

        // 8. FFN: RMSnorm -> SwiGLU(w1, w3) -> w2 -> residual.
        state.residual.copy_from_slice(&state.cur);
        cpu::rmsnorm(&mut state.cur, &lw.ffn_norm, cfg.rms_norm_eps);

        state.gate.iter_mut().for_each(|v| *v = 0.0);
        state.up.iter_mut().for_each(|v| *v = 0.0);
        lw.w1.gemv_scratch(
            &state.cur,
            &mut state.gate,
            Some((&mut state.q8_scales, &mut state.q8_quants)),
        );
        lw.w3.gemv_scratch(
            &state.cur,
            &mut state.up,
            Some((&mut state.q8_scales, &mut state.q8_quants)),
        );
        cpu::silu_mul_inplace(&mut state.gate, &state.up);

        state.proj.iter_mut().for_each(|v| *v = 0.0);
        lw.w2.gemv_scratch(
            &state.gate,
            &mut state.proj,
            Some((&mut state.q8_scales, &mut state.q8_quants)),
        );
        for (c, (r, d)) in state
            .cur
            .iter_mut()
            .zip(state.residual.iter().zip(&state.proj))
        {
            *c = r + d;
        }
    }

    state.n_past += 1;
}

// ── DecoderModel: 8-codebook LFM2A & 12-codebook LFM3A-Gen sampling ─────

/// Sample one audio frame (8 codes for `lfm2a`, 12 codes for `lfm3a_gen`) from an LLM embedding.
///
/// For `lfm3a_gen`, computes the 3-step cosine-cubed temperature schedule
/// `t_s = temperature * (1 - ((s + 1) / 3)^3)` and runs 3-step MaskGIT iterative parallel
/// decoding with CTRL repetition penalty across recent frames.
pub fn sample_frame(
    weights: &AudioDecoderWeights,
    state: &mut DepthformerState,
    embedding: &[f32],
    temperature: f32,
    top_k: usize,
) -> Vec<i32> {
    if weights.is_lfm3a_gen() {
        let base_t = if temperature.is_finite() && temperature > 0.0 {
            temperature
        } else {
            0.0
        };
        let step_temps = [
            base_t * (1.0 - (1.0f32 / 3.0).powi(3)),
            base_t * (1.0 - (2.0f32 / 3.0).powi(3)),
            0.0,
        ];
        sample_frame_with_temps(weights, state, embedding, step_temps, top_k)
    } else {
        sample_audio_frame(weights, state, embedding, temperature, top_k).to_vec()
    }
}

/// Sample one 12-codebook `lfm3a_gen` audio frame using explicit per-step MaskGIT temperatures
/// `step_temps` (`[step0, step1, step2]`) and the deterministic PRNG inside `state`.
///
/// Implements the 3-step MaskGIT schedule (`[4, 4, 4]` codebooks unmasked per step):
/// - All 12 codebooks start at `mask_token_id = n_vocab - 1` (`256`).
/// - Each step projects the LLM hidden state + current `12` codebook embeddings into a
///   `[12, n_embd]` sequence and runs all `n_layer` bidirectional LFM3 depthformer blocks.
/// - Applies the CTRL repetition penalty (`1.2`) across the last `62` frames in `state.code_history`.
/// - On step 0, codebook 0 can sample `n_vocab - 1` (`256`) as EOS; if it does, terminates early
///   and returns `vec![256; 12]`.
/// - Unmasks the 4 highest-confidence remaining codebooks at each step.
pub fn sample_frame_with_temps(
    weights: &AudioDecoderWeights,
    state: &mut DepthformerState,
    embedding: &[f32],
    step_temps: [f32; 3],
    top_k: usize,
) -> Vec<i32> {
    if !weights.is_lfm3a_gen() {
        return sample_audio_frame(weights, state, embedding, step_temps[0], top_k).to_vec();
    }
    let mut rng_state = state.rng_state;
    let codes =
        sample_frame_with_temps_and_rng(weights, state, embedding, step_temps, top_k, || {
            xorshift64_next_f32_01(&mut rng_state)
        });
    state.rng_state = rng_state;
    codes
}

/// Sample one 12-codebook `lfm3a_gen` audio frame using a caller-supplied uniform `[0, 1)` RNG
/// closure (`rng_f32`) and explicit per-step MaskGIT temperatures `step_temps` (`[step0, step1, step2]`).
pub fn sample_frame_lfm3a<F>(
    weights: &AudioDecoderWeights,
    state: &mut DepthformerState,
    embedding: &[f32],
    step_temps: [f32; 3],
    top_k: usize,
    rng_f32: F,
) -> Vec<i32>
where
    F: FnMut() -> f32,
{
    sample_frame_with_temps_and_rng(weights, state, embedding, step_temps, top_k, rng_f32)
}

/// Sample one 12-codebook `lfm3a_gen` audio frame with explicit per-step MaskGIT temperatures
/// and a caller-supplied uniform `[0, 1)` random float generator `rng_f32`.
pub fn sample_frame_with_temps_and_rng<F>(
    weights: &AudioDecoderWeights,
    state: &mut DepthformerState,
    embedding: &[f32],
    step_temps: [f32; 3],
    top_k: usize,
    mut rng_f32: F,
) -> Vec<i32>
where
    F: FnMut() -> f32,
{
    if !weights.is_lfm3a_gen() {
        return sample_audio_frame(weights, state, embedding, step_temps[0], top_k).to_vec();
    }

    let cfg = &weights.decoder_config;
    let df_cfg = &weights.depthformer_config;
    let n_cb = cfg.n_codebook;
    let n_vocab = cfg.n_vocab;
    let mask_id = (n_vocab.saturating_sub(1)) as i32;
    let n_embd = df_cfg.n_embd;

    let mut codes = vec![mask_id; n_cb];
    let mut unmasked = vec![false; n_cb];

    // Precompute LLM projection: either shared `[n_embd, llm_dim]` broadcast across all 12
    // codebooks, or per-codebook `[12 * n_embd, llm_dim]`.
    let shared_llm_proj = weights.depth_linear_w.rows == n_embd;
    let mut llm_proj_buf = vec![0.0f32; weights.depth_linear_w.rows];
    weights.depth_linear_w.gemv(embedding, &mut llm_proj_buf);
    for (dst, &b) in llm_proj_buf.iter_mut().zip(weights.depth_linear_b.iter()) {
        *dst += b;
    }

    let unmasked_per_step = [4usize, 4, n_cb.saturating_sub(8)];
    let mut seq = vec![0.0f32; n_cb * n_embd];
    let mut tok_row = vec![0.0f32; n_embd];

    for (step, &k_unmask) in unmasked_per_step.iter().enumerate() {
        // 1. Construct `[n_cb, n_embd]` input for this MaskGIT step.
        for j in 0..n_cb {
            let dst = &mut seq[j * n_embd..(j + 1) * n_embd];
            if shared_llm_proj {
                dst.copy_from_slice(&llm_proj_buf);
            } else {
                dst.copy_from_slice(&llm_proj_buf[j * n_embd..(j + 1) * n_embd]);
            }
            let cb_emb = &weights.depth_embeddings[j].embedding;
            let tok_idx = (codes[j].max(0) as usize).min(cb_emb.rows.saturating_sub(1));
            cb_emb.dequantize_row(tok_idx, &mut tok_row);
            for (d, &e) in dst.iter_mut().zip(tok_row.iter()) {
                *d += e;
            }
        }

        // 2. Run bidirectional LFM3 depthformer stack over all `n_cb = 12` tokens simultaneously.
        depthformer_bidirectional_lfm3a(weights, &mut seq, n_cb);

        // 3. Score all remaining masked codebooks.
        let t_step = step_temps[step];
        let mut cand_tokens = vec![0i32; n_cb];
        let mut confidences = vec![f32::NEG_INFINITY; n_cb];
        let mut logits = vec![0.0f32; n_vocab];

        for j in 0..n_cb {
            if unmasked[j] {
                continue;
            }
            let cb = &weights.depth_embeddings[j];
            let mut normed = seq[j * n_embd..(j + 1) * n_embd].to_vec();
            cpu::rmsnorm(&mut normed, &cb.norm, cfg.rms_norm_eps);
            logits.fill(0.0);
            cb.to_logits.gemv(&normed, &mut logits);

            // Apply CTRL repetition penalty across unique codes in the last `62` frames.
            if !state.code_history.is_empty() {
                let mut seen = vec![false; n_vocab];
                for past_frame in &state.code_history {
                    if let Some(&past_code) = past_frame.get(j)
                        && past_code >= 0
                    {
                        let idx = past_code as usize;
                        if idx < n_vocab && !seen[idx] {
                            seen[idx] = true;
                            if logits[idx] > 0.0 {
                                logits[idx] /= LFM3A_REP_PENALTY;
                            } else {
                                logits[idx] *= LFM3A_REP_PENALTY;
                            }
                        }
                    }
                }
            }

            // Only semantic codebook 0 (`j == 0`) can sample `mask_id` (`audio_eos_code`).
            if j > 0 && (mask_id as usize) < logits.len() {
                logits[mask_id as usize] = f32::NEG_INFINITY;
            }

            let max_logit = logits
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(f32::NEG_INFINITY, f32::max);
            let log_z = if max_logit.is_finite() {
                let sum_exp: f64 = logits
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .map(|v| ((v - max_logit) as f64).exp())
                    .sum();
                max_logit + (sum_exp.max(1e-30).ln() as f32)
            } else {
                0.0
            };

            if !t_step.is_finite() || t_step <= 0.0 || top_k == 1 {
                let best = crate::sampler::argmax(&logits);
                cand_tokens[j] = best as i32;
                confidences[j] = logits[best as usize] - log_z;
            } else {
                let raw_logits_snap = logits.clone();
                // Optional top-k filtering before Gumbel-Max.
                if top_k > 0 && top_k < logits.len() {
                    let mut order: Vec<usize> = (0..logits.len()).collect();
                    order.select_nth_unstable_by(top_k - 1, |&a, &b| {
                        logits[b].total_cmp(&logits[a])
                    });
                    for &idx in &order[top_k..] {
                        logits[idx] = f32::NEG_INFINITY;
                    }
                }
                let mut best_idx = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for (v_idx, &logit) in logits.iter().enumerate() {
                    if !logit.is_finite() {
                        continue;
                    }
                    let u = rng_f32().clamp(1e-10, 1.0 - 1e-7);
                    let gumbel = -(-u.ln()).ln();
                    let noisy = logit + t_step * gumbel;
                    if noisy > best_score {
                        best_score = noisy;
                        best_idx = v_idx;
                    }
                }
                let u_conf = rng_f32().clamp(1e-10, 1.0 - 1e-7);
                let gumbel_conf = -(-u_conf.ln()).ln();
                let anneal = 1.0 - ((step + 1) as f32 / unmasked_per_step.len() as f32);
                cand_tokens[j] = best_idx as i32;
                confidences[j] =
                    (raw_logits_snap[best_idx] - log_z) + 0.5 * t_step * gumbel_conf * anneal;
            }

            // Early stop if step 0 samples EOS on codebook 0.
            if step == 0 && j == 0 && cand_tokens[0] == mask_id {
                return vec![mask_id; n_cb];
            }
        }

        // Unmask the `k_unmask` highest-confidence masked codebooks.
        let mut remaining: Vec<usize> = (0..n_cb).filter(|&j| !unmasked[j]).collect();
        remaining.sort_by(|&a, &b| {
            confidences[b]
                .total_cmp(&confidences[a])
                .then_with(|| a.cmp(&b))
        });
        for &j in remaining.iter().take(k_unmask) {
            codes[j] = cand_tokens[j];
            unmasked[j] = true;
        }
        if unmasked.first().copied() == Some(true) && codes.first().copied() == Some(mask_id) {
            return vec![mask_id; n_cb];
        }
    }

    if codes.first().copied() != Some(mask_id) {
        state.push_code_history(codes.clone());
    }
    codes
}

/// Run the bidirectional LFM3 depthformer stack in place over `[n_tokens, n_embd]`.
fn depthformer_bidirectional_lfm3a(
    weights: &AudioDecoderWeights,
    seq: &mut [f32],
    n_tokens: usize,
) {
    let cfg = &weights.depthformer_config;
    let n_embd = cfg.n_embd;
    let n_head = cfg.n_head;
    let n_kv = cfg.n_head_kv;
    let hd = cfg.n_embd_head;
    let eps = cfg.rms_norm_eps;

    let mut norm_mat = vec![0.0f32; n_tokens * n_embd];
    let mut gate_mat = vec![0.0f32; n_tokens * cfg.ffn_dim];
    let mut up_mat = vec![0.0f32; n_tokens * cfg.ffn_dim];
    let mut down_mat = vec![0.0f32; n_tokens * n_embd];

    for lw in &weights.depthformer_layers {
        let is_shortconv = lw.k_norm.is_empty();
        norm_mat.copy_from_slice(&seq[..n_tokens * n_embd]);
        for s in 0..n_tokens {
            cpu::rmsnorm(
                &mut norm_mat[s * n_embd..(s + 1) * n_embd],
                &lw.operator_norm,
                eps,
            );
        }

        if is_shortconv {
            // Bidirectional ShortConv block (`pad_left = (k - 1) / 2`, `pad_right = k - 1 - pad_left`).
            let kernel_size = (lw.q_norm.len() / n_embd).max(1);
            let pad_total = kernel_size - 1;
            let pad_left = pad_total / 2;
            let t_padded = n_tokens + pad_total;
            let mut bcx_mat = vec![0.0f32; n_tokens * 3 * n_embd];
            lw.wqkv.batched_matmul(&norm_mat, &mut bcx_mat, n_tokens);

            let mut bx_padded = vec![0.0f32; n_embd * t_padded];
            let mut c_gate = vec![0.0f32; n_tokens * n_embd];
            for s in 0..n_tokens {
                let bcx = &bcx_mat[s * 3 * n_embd..(s + 1) * 3 * n_embd];
                let (b_part, rest) = bcx.split_at(n_embd);
                let (c_part, x_part) = rest.split_at(n_embd);
                c_gate[s * n_embd..(s + 1) * n_embd].copy_from_slice(c_part);
                for ch in 0..n_embd {
                    bx_padded[ch * t_padded + pad_left + s] = b_part[ch] * x_part[ch];
                }
            }

            let mut conv_out = vec![0.0f32; n_embd * n_tokens];
            cpu::conv1d(
                &bx_padded,
                &lw.q_norm,
                None,
                &mut conv_out,
                n_embd,
                n_embd,
                t_padded,
                kernel_size,
                1,
                0,
                n_embd,
            );

            for s in 0..n_tokens {
                let c_row = &c_gate[s * n_embd..(s + 1) * n_embd];
                let y_in = &mut norm_mat[s * n_embd..(s + 1) * n_embd];
                for ch in 0..n_embd {
                    y_in[ch] = conv_out[ch * n_tokens + s] * c_row[ch];
                }
            }
            lw.wo.batched_matmul(&norm_mat, &mut down_mat, n_tokens);
            cpu::add_inplace(&mut seq[..n_tokens * n_embd], &down_mat);
        } else {
            // Bidirectional Self-Attention block with per-head RMSNorm, Neox RoPE, and optional sigmoid gate.
            let q_dim = n_head * hd;
            let k_dim = n_kv * hd;
            let qkv_dim = q_dim + 2 * k_dim;
            let gate_rows = lw.wqkv.rows.saturating_sub(qkv_dim);
            let group_size = (n_head / n_kv.max(1)).max(1);
            let scale = 1.0f32 / (hd as f32).sqrt();

            let mut qkv_mat = vec![0.0f32; n_tokens * lw.wqkv.rows];
            lw.wqkv.batched_matmul(&norm_mat, &mut qkv_mat, n_tokens);

            let mut all_q = vec![0.0f32; n_tokens * q_dim];
            let mut all_k = vec![0.0f32; n_tokens * k_dim];
            let mut all_v = vec![0.0f32; n_tokens * k_dim];
            let mut all_gate = vec![0.0f32; n_tokens * gate_rows];

            for s in 0..n_tokens {
                let qkv = &qkv_mat[s * lw.wqkv.rows..(s + 1) * lw.wqkv.rows];
                let q_row = &mut all_q[s * q_dim..(s + 1) * q_dim];
                let k_row = &mut all_k[s * k_dim..(s + 1) * k_dim];
                let v_row = &mut all_v[s * k_dim..(s + 1) * k_dim];
                q_row.copy_from_slice(&qkv[..q_dim]);
                k_row.copy_from_slice(&qkv[q_dim..q_dim + k_dim]);
                v_row.copy_from_slice(&qkv[q_dim + k_dim..qkv_dim]);
                if gate_rows > 0 {
                    all_gate[s * gate_rows..(s + 1) * gate_rows].copy_from_slice(&qkv[qkv_dim..]);
                }

                for h in 0..n_head {
                    let qh = &mut q_row[h * hd..(h + 1) * hd];
                    cpu::rmsnorm(qh, &lw.q_norm, eps);
                    apply_rope_neox(qh, s, hd, cfg.rope_freq_base);
                }
                for h_kv in 0..n_kv {
                    let kh = &mut k_row[h_kv * hd..(h_kv + 1) * hd];
                    cpu::rmsnorm(kh, &lw.k_norm, eps);
                    apply_rope_neox(kh, s, hd, cfg.rope_freq_base);
                }
            }

            let mut attn_out_mat = vec![0.0f32; n_tokens * q_dim];
            let mut scores = vec![0.0f32; n_tokens];
            for q_pos in 0..n_tokens {
                let attn_out = &mut attn_out_mat[q_pos * q_dim..(q_pos + 1) * q_dim];
                for h in 0..n_head {
                    let kv_h = h / group_size;
                    let qh = &all_q[q_pos * q_dim + h * hd..q_pos * q_dim + (h + 1) * hd];
                    for k_pos in 0..n_tokens {
                        let kh = &all_k[k_pos * k_dim + kv_h * hd..k_pos * k_dim + (kv_h + 1) * hd];
                        scores[k_pos] = cpu::dot_f32(qh, kh) * scale;
                    }
                    cpu::softmax_inplace(&mut scores);
                    let out_h = &mut attn_out[h * hd..(h + 1) * hd];
                    for k_pos in 0..n_tokens {
                        let vh = &all_v[k_pos * k_dim + kv_h * hd..k_pos * k_dim + (kv_h + 1) * hd];
                        let w = scores[k_pos];
                        for d in 0..hd {
                            out_h[d] += w * vh[d];
                        }
                    }
                }
                if gate_rows > 0 {
                    let g_row = &all_gate[q_pos * gate_rows..(q_pos + 1) * gate_rows];
                    if gate_rows == n_head {
                        for h in 0..n_head {
                            let g = 1.0f32 / (1.0f32 + (-g_row[h]).exp());
                            for v in &mut attn_out[h * hd..(h + 1) * hd] {
                                *v *= g;
                            }
                        }
                    } else {
                        for (v, &g) in attn_out.iter_mut().zip(g_row.iter()) {
                            *v *= 1.0f32 / (1.0f32 + (-g).exp());
                        }
                    }
                }
            }
            lw.wo.batched_matmul(&attn_out_mat, &mut down_mat, n_tokens);
            cpu::add_inplace(&mut seq[..n_tokens * n_embd], &down_mat);
        }

        // SwiGLU FFN on all `n_tokens` positions simultaneously.
        norm_mat.copy_from_slice(&seq[..n_tokens * n_embd]);
        for s in 0..n_tokens {
            cpu::rmsnorm(
                &mut norm_mat[s * n_embd..(s + 1) * n_embd],
                &lw.ffn_norm,
                eps,
            );
        }
        lw.w1.batched_matmul(&norm_mat, &mut gate_mat, n_tokens);
        lw.w3.batched_matmul(&norm_mat, &mut up_mat, n_tokens);
        cpu::silu_mul_inplace(&mut gate_mat, &up_mat);
        lw.w2.batched_matmul(&gate_mat, &mut down_mat, n_tokens);
        cpu::add_inplace(&mut seq[..n_tokens * n_embd], &down_mat);
    }

    if weights.audio_embedding.norm.len() == n_embd {
        for s in 0..n_tokens {
            cpu::rmsnorm(
                &mut seq[s * n_embd..(s + 1) * n_embd],
                &weights.audio_embedding.norm,
                eps,
            );
        }
    }
}

/// Sample one audio frame (8 codes) from an LLM embedding.
///
/// Runs 8 sequential passes through the depthformer, one per codebook.
/// Each pass projects the LLM embedding into the depthformer input,
/// optionally adds the previous codebook's token embedding, runs the
/// depthformer, and samples a token from the codebook's logits.
pub fn sample_audio_frame(
    weights: &AudioDecoderWeights,
    state: &mut DepthformerState,
    embedding: &[f32],
    temperature: f32,
    top_k: usize,
) -> [i32; 8] {
    let cfg = &weights.decoder_config;
    let mut token = [0i32; 8];
    let mut prev_token: i32 = -1;

    state.reset();

    let n_embd_d = weights.depth_embeddings[0].embedding.cols;

    for j in 0..cfg.n_codebook.min(8) {
        let cb = &weights.depth_embeddings[j];
        debug_assert_eq!(
            cb.embedding.cols, n_embd_d,
            "codebook {j} has mismatched depthformer dim"
        );

        // 1. Project LLM embedding -> depthformer input for codebook j.
        state.depthformer_in.fill(0.0);
        let row_start = j * n_embd_d;
        weights.depth_linear_w.gemv_rows_scratch(
            embedding,
            &mut state.depthformer_in,
            row_start,
            n_embd_d,
            Some((&mut state.q8_scales, &mut state.q8_quants)),
        );
        // Add bias.
        for (r, out) in state.depthformer_in.iter_mut().enumerate() {
            *out += weights.depth_linear_b[row_start + r];
        }

        // 2. Add previous codebook's token embedding (if j > 0).
        if j > 0 && prev_token >= 0 {
            let prev_cb = &weights.depth_embeddings[j - 1];
            let tok = prev_token as usize;
            if tok < prev_cb.embedding.rows {
                prev_cb.embedding.dequantize_row(tok, &mut state.emb_row);
                for (d, e) in state.depthformer_in.iter_mut().zip(&state.emb_row) {
                    *d += e;
                }
            }
        }

        // 3. Run depthformer in-place.
        depthformer_forward_inplace(weights, state);

        // 4. RMSnorm -> to_logits -> sample.
        state.normed.copy_from_slice(&state.cur);
        cpu::rmsnorm(&mut state.normed, &cb.norm, cfg.rms_norm_eps);

        state.logits.resize(cfg.n_vocab, 0.0);
        state.logits.fill(0.0);
        cb.to_logits.gemv_scratch(
            &state.normed,
            &mut state.logits,
            Some((&mut state.q8_scales, &mut state.q8_quants)),
        );

        // Sample (greedy if temperature <= 0 or top_k <= 1, otherwise top-k).
        let sampled = if state.logits.is_empty() {
            0
        } else if !temperature.is_finite() || temperature <= 0.0 || top_k <= 1 {
            crate::sampler::argmax(&state.logits) as i32
        } else {
            // Temperature scaling + top-k sampling.
            let inv_temp = 1.0 / temperature;
            for l in &mut state.logits {
                *l *= inv_temp;
            }
            cpu::softmax_inplace(&mut state.logits);
            let n_logits = state.logits.len();
            let k = top_k.min(n_logits).max(1);
            state.indices.clear();
            state.indices.extend(0..n_logits);
            if k < n_logits {
                state.indices.select_nth_unstable_by(k - 1, |&a, &b| {
                    state.logits[b].total_cmp(&state.logits[a])
                });
            }
            let r_u01 = state.next_f32_01();
            let top_indices = &state.indices[..k];
            let sum: f32 = top_indices.iter().map(|&i| state.logits[i]).sum();
            let mut r = r_u01 * sum;
            let mut picked = *top_indices.last().unwrap_or(&0);
            for &i in top_indices {
                r -= state.logits[i];
                if r <= 0.0 {
                    picked = i;
                    break;
                }
            }
            picked as i32
        };

        token[j] = sampled;
        prev_token = sampled;
    }

    token
}

/// Convert audio codes (8 for `lfm2a`, 12 for `lfm3a_gen`) back into a single LLM feedback
/// embedding by summing the per-codebook table lookups at `j * n_vocab + codes[j]`.
pub fn embed_audio_codes(weights: &AudioDecoderWeights, codes: &[i32]) -> Vec<f32> {
    let emb = &weights.audio_embedding.embedding;
    let n_codebook = weights.decoder_config.n_codebook;
    let n_vocab = weights.decoder_config.n_vocab.max(1);
    let emb_dim = emb.cols;
    let mut result = vec![0.0f32; emb_dim];

    let mut row = vec![0.0f32; emb_dim];
    for (j, &code) in codes.iter().take(n_codebook).enumerate() {
        if code >= 0 {
            let offset_idx = j * n_vocab + code as usize;
            if offset_idx < emb.rows {
                emb.dequantize_row(offset_idx, &mut row);
                for (r, e) in result.iter_mut().zip(&row) {
                    *r += e;
                }
            }
        }
    }

    result
}

/// Convert 8 audio codes back into an embedding for feeding to the LLM.
///
/// Each code indexes into `audio_embedding.embedding.weight` (with per-codebook
/// offsets), and the embeddings are summed to produce a single vector.
pub fn embed_audio_token(weights: &AudioDecoderWeights, codes: &[i32; 8]) -> Vec<f32> {
    embed_audio_codes(weights, codes)
}

// ── Detokenizer: codes -> PCM ───────────────────────────────────────────

/// Config for the detokenizer's LFM2/LFM3 backbone.
#[derive(Debug, Clone)]
pub struct DetokenizerConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head: usize,
    pub ffn_dim: usize,
    pub d_conv: usize, // conv kernel - 1
    pub rms_norm_eps: f32,
    pub rope_freq_base: f32,
    pub swa_window_size: usize,
    pub n_codes: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub sample_rate: usize,
    /// Per-layer type: true = conv (recurrent), false = attention.
    pub layer_is_conv: Vec<bool>,
}

/// Weights for one detokenizer LFM2/LFM3 layer.
pub struct DetokLayerWeights {
    pub operator_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub attn_post_norm: Option<Vec<f32>>,
    pub ffn_post_norm: Option<Vec<f32>>,
    pub ffn_w1: MmapWeight, // gate
    pub ffn_w2: MmapWeight, // down
    pub ffn_w3: MmapWeight, // up
    // Conv layers
    pub conv_in_proj: Option<MmapWeight>,
    pub conv_out_proj: Option<MmapWeight>,
    pub conv_weight: Option<Vec<f32>>, // [kernel_size, n_embd]
    // Attention layers
    pub wq: Option<MmapWeight>,
    pub wk: Option<MmapWeight>,
    pub wv: Option<MmapWeight>,
    pub wo: Option<MmapWeight>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub attn_gate: Option<MmapWeight>,
}

/// Typed FSQ + input-projection weights for the 12-codebook `lfm3a_gen` detokenizer.
pub struct Lfm3FsqWeights {
    pub fsq_codebook: Vec<f32>,
    pub fsq_proj_dim: usize,
    pub quant_first_out_w: Vec<f32>,
    pub quant_first_out_b: Vec<f32>,
    pub quant_rest_out_w: Vec<f32>,
    pub quant_rest_out_b: Vec<f32>,
    pub tfm_in_proj_w: MmapWeight,
    pub tfm_in_proj_b: Vec<f32>,
}

/// All detokenizer weights (loaded from vocoder or single-file mmproj GGUF).
pub struct DetokenizerWeights {
    pub config: DetokenizerConfig,
    pub output_norm: Vec<f32>,
    pub emb_weight: MmapWeight, // code embedding [n_codes * n_vocab, emb_dim]
    pub lin_w: MmapWeight,      // linear head
    pub lin_b: Vec<f32>,
    pub layers: Vec<DetokLayerWeights>,
    pub lfm3_fsq: Option<Lfm3FsqWeights>,
}

fn get_detok_tensor(gguf: &GgufFile, name1: &str, name2: &str) -> Result<crate::tensor::Tensor> {
    if gguf.tensors.contains_key(name1) {
        gguf.get_tensor(name1)
            .with_context(|| format!("failed to load tensor `{name1}`"))
    } else if gguf.tensors.contains_key(name2) {
        gguf.get_tensor(name2)
            .with_context(|| format!("failed to load tensor `{name2}`"))
    } else {
        anyhow::bail!("tensor not found: neither `{name1}` nor `{name2}`")
    }
}

fn get_detok_mmap_weight(gguf: &Arc<GgufFile>, name1: &str, name2: &str) -> Result<MmapWeight> {
    if gguf.tensors.contains_key(name1) {
        MmapWeight::from_gguf(gguf, name1)
            .with_context(|| format!("failed to load tensor weight `{name1}`"))
    } else if gguf.tensors.contains_key(name2) {
        MmapWeight::from_gguf(gguf, name2)
            .with_context(|| format!("failed to load tensor weight `{name2}`"))
    } else {
        anyhow::bail!("tensor weight not found: neither `{name1}` nor `{name2}`")
    }
}

/// Build the canonical 256-entry 4D 4-level FSQ codebook in `[-1.0, -1/3, +1/3, +1.0]^4`
/// (`[256, 4]` row-major, dim 0 cycling fastest across the 4 levels).
pub fn build_default_fsq_codebook() -> Vec<f32> {
    const LEVELS: [f32; 4] = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0];
    let mut table = vec![0.0f32; 256 * 4];
    for idx in 0..256 {
        let mut rem = idx;
        for d in 0..4 {
            table[idx * 4 + d] = LEVELS[rem % 4];
            rem /= 4;
        }
    }
    table
}

impl DetokenizerWeights {
    /// Returns `true` when this detokenizer uses the 12-codebook `lfm3a_gen` FSQ + ISTFT architecture.
    pub fn is_lfm3a_gen(&self) -> bool {
        self.lfm3_fsq.is_some() || self.config.n_codes == 12
    }

    pub fn from_gguf(gguf: &Arc<GgufFile>) -> Result<Self> {
        if has_lfm3a_gen_detokenizer(gguf) {
            return Self::from_gguf_lfm3a_gen(gguf);
        }

        // Layer types (hardcoded from decoder.cpp).
        let layer_is_conv = vec![true, true, false, true, false, true, false, true];
        let n_layer = layer_is_conv.len();

        // Derive config from weight shapes.
        let conv_in = get_detok_mmap_weight(
            gguf,
            "lfm.layers.0.conv.in_proj.weight",
            "blk.0.shortconv.in_proj.weight",
        )?;
        let n_embd = conv_in.cols;
        anyhow::ensure!(n_embd > 0, "detokenizer n_embd must be > 0");
        let q_norm_w = get_detok_tensor(
            gguf,
            "lfm.layers.2.self_attn.q_layernorm.weight",
            "blk.2.attn_q_norm.weight",
        )?;
        let n_embd_head = q_norm_w.shape()[0];
        // Guard against a corrupt vocoder GGUF reporting an empty
        // q_layernorm shape; both `n_head` and `n_head_kv` are
        // derived from this dim and would div-by-zero panic
        // otherwise.
        anyhow::ensure!(
            n_embd_head > 0,
            "detokenizer n_embd_head must be > 0 (q_layernorm shape was empty)"
        );
        let q_w = get_detok_mmap_weight(
            gguf,
            "lfm.layers.2.self_attn.q_proj.weight",
            "blk.2.attn_q.weight",
        )?;
        anyhow::ensure!(
            q_w.rows >= n_embd_head && q_w.rows.is_multiple_of(n_embd_head),
            "detokenizer q_proj rows ({}) must be a positive multiple of n_embd_head ({n_embd_head})",
            q_w.rows
        );
        let n_head = q_w.rows / n_embd_head;
        let k_w = get_detok_mmap_weight(
            gguf,
            "lfm.layers.2.self_attn.k_proj.weight",
            "blk.2.attn_k.weight",
        )?;
        anyhow::ensure!(
            k_w.rows >= n_embd_head && k_w.rows.is_multiple_of(n_embd_head),
            "detokenizer k_proj rows ({}) must be a positive multiple of n_embd_head ({n_embd_head})",
            k_w.rows
        );
        let n_head_kv = k_w.rows / n_embd_head;
        anyhow::ensure!(
            n_head > 0 && n_head_kv > 0 && n_head.is_multiple_of(n_head_kv),
            "detokenizer n_head ({n_head}) must be a positive multiple of n_head_kv ({n_head_kv})"
        );
        let ffn_w1_0 = get_detok_mmap_weight(
            gguf,
            "lfm.layers.0.feed_forward.w1.weight",
            "blk.0.ffn_gate.weight",
        )?;
        let ffn_dim = ffn_w1_0.rows;
        anyhow::ensure!(ffn_dim > 0, "detokenizer ffn_dim must be > 0");

        let config = DetokenizerConfig {
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            n_embd_head,
            ffn_dim,
            d_conv: 2, // kernel_size=3, d_conv=2
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            swa_window_size: 30,
            n_codes: 8,
            n_fft: 1280,
            hop_length: 320,
            sample_rate: 24000,
            layer_is_conv,
        };

        let output_norm =
            get_detok_tensor(gguf, "lfm.embedding_norm.weight", "token_embd_norm.weight")?
                .try_to_f32_vec()?;
        anyhow::ensure!(
            output_norm.len() == n_embd,
            "detokenizer output_norm len ({}) mismatch n_embd ({n_embd})",
            output_norm.len()
        );

        let emb_weight = get_detok_mmap_weight(gguf, "emb.emb.weight", "token_embd.weight")?;
        let lin_w = get_detok_mmap_weight(gguf, "lin.weight", "dense_2.weight")?;
        let lin_b = get_detok_tensor(gguf, "lin.bias", "dense_2.bias")?.try_to_f32_vec()?;

        anyhow::ensure!(
            emb_weight.cols == n_embd,
            "detokenizer emb_weight cols ({}) mismatch n_embd ({})",
            emb_weight.cols,
            n_embd
        );
        anyhow::ensure!(
            emb_weight.rows == config.n_codes * 2048,
            "detokenizer emb_weight rows ({}) mismatch expected {} ({} codebooks * 2048)",
            emb_weight.rows,
            config.n_codes * 2048,
            config.n_codes
        );
        anyhow::ensure!(
            lin_w.cols == n_embd,
            "detokenizer lin_w cols ({}) mismatch n_embd ({})",
            lin_w.cols,
            n_embd
        );
        let expected_lin_out = (config.n_fft / 2 + 1) * 2;
        anyhow::ensure!(
            lin_w.rows == expected_lin_out,
            "detokenizer lin_w rows ({}) mismatch expected {expected_lin_out}",
            lin_w.rows
        );
        anyhow::ensure!(
            lin_b.len() == expected_lin_out,
            "detokenizer lin_b len ({}) mismatch expected {expected_lin_out}",
            lin_b.len()
        );

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let prefix_lfm = format!("lfm.layers.{i}");
            let prefix_blk = format!("blk.{i}");
            let is_conv = config.layer_is_conv[i];

            let operator_norm = get_detok_tensor(
                gguf,
                &format!("{prefix_lfm}.operator_norm.weight"),
                &format!("{prefix_blk}.attn_norm.weight"),
            )?
            .try_to_f32_vec()?;
            let ffn_norm = get_detok_tensor(
                gguf,
                &format!("{prefix_lfm}.ffn_norm.weight"),
                &format!("{prefix_blk}.ffn_norm.weight"),
            )?
            .try_to_f32_vec()?;
            anyhow::ensure!(
                operator_norm.len() == n_embd && ffn_norm.len() == n_embd,
                "detokenizer layer {i} norm length ({}, {}) mismatch n_embd ({n_embd})",
                operator_norm.len(),
                ffn_norm.len()
            );

            let ffn_w1 = get_detok_mmap_weight(
                gguf,
                &format!("{prefix_lfm}.feed_forward.w1.weight"),
                &format!("{prefix_blk}.ffn_gate.weight"),
            )?;
            let ffn_w2 = get_detok_mmap_weight(
                gguf,
                &format!("{prefix_lfm}.feed_forward.w2.weight"),
                &format!("{prefix_blk}.ffn_down.weight"),
            )?;
            let ffn_w3 = get_detok_mmap_weight(
                gguf,
                &format!("{prefix_lfm}.feed_forward.w3.weight"),
                &format!("{prefix_blk}.ffn_up.weight"),
            )?;
            anyhow::ensure!(
                ffn_w1.rows == ffn_dim
                    && ffn_w1.cols == n_embd
                    && ffn_w3.rows == ffn_dim
                    && ffn_w3.cols == n_embd
                    && ffn_w2.rows == n_embd
                    && ffn_w2.cols == ffn_dim,
                "detokenizer layer {i} FFN weight shape mismatch"
            );

            let conv_in_proj = if is_conv {
                let w = get_detok_mmap_weight(
                    gguf,
                    &format!("{prefix_lfm}.conv.in_proj.weight"),
                    &format!("{prefix_blk}.shortconv.in_proj.weight"),
                )?;
                anyhow::ensure!(
                    w.rows == 3 * n_embd && w.cols == n_embd,
                    "detokenizer layer {i} conv_in_proj shape [{}, {}] mismatch expected [{}, {n_embd}]",
                    w.rows,
                    w.cols,
                    3 * n_embd
                );
                Some(w)
            } else {
                None
            };
            let conv_out_proj = if is_conv {
                let w = get_detok_mmap_weight(
                    gguf,
                    &format!("{prefix_lfm}.conv.out_proj.weight"),
                    &format!("{prefix_blk}.shortconv.out_proj.weight"),
                )?;
                anyhow::ensure!(
                    w.rows == n_embd && w.cols == n_embd,
                    "detokenizer layer {i} conv_out_proj shape [{}, {}] mismatch expected [{n_embd}, {n_embd}]",
                    w.rows,
                    w.cols
                );
                Some(w)
            } else {
                None
            };
            let conv_weight = if is_conv {
                let w = get_detok_tensor(
                    gguf,
                    &format!("{prefix_lfm}.conv.conv.weight"),
                    &format!("{prefix_blk}.shortconv.conv.weight"),
                )?
                .try_to_f32_vec()?;
                anyhow::ensure!(
                    w.len() == (config.d_conv + 1) * n_embd,
                    "detokenizer layer {i} conv_weight len ({}) mismatch expected {}",
                    w.len(),
                    (config.d_conv + 1) * n_embd
                );
                Some(w)
            } else {
                None
            };

            let wq = if !is_conv {
                let w = get_detok_mmap_weight(
                    gguf,
                    &format!("{prefix_lfm}.self_attn.q_proj.weight"),
                    &format!("{prefix_blk}.attn_q.weight"),
                )?;
                anyhow::ensure!(
                    w.rows == n_head * n_embd_head && w.cols == n_embd,
                    "detokenizer layer {i} wq shape [{}, {}] mismatch expected [{}, {n_embd}]",
                    w.rows,
                    w.cols,
                    n_head * n_embd_head
                );
                Some(w)
            } else {
                None
            };
            let wk = if !is_conv {
                let w = get_detok_mmap_weight(
                    gguf,
                    &format!("{prefix_lfm}.self_attn.k_proj.weight"),
                    &format!("{prefix_blk}.attn_k.weight"),
                )?;
                anyhow::ensure!(
                    w.rows == n_head_kv * n_embd_head && w.cols == n_embd,
                    "detokenizer layer {i} wk shape [{}, {}] mismatch expected [{}, {n_embd}]",
                    w.rows,
                    w.cols,
                    n_head_kv * n_embd_head
                );
                Some(w)
            } else {
                None
            };
            let wv = if !is_conv {
                let w = get_detok_mmap_weight(
                    gguf,
                    &format!("{prefix_lfm}.self_attn.v_proj.weight"),
                    &format!("{prefix_blk}.attn_v.weight"),
                )?;
                anyhow::ensure!(
                    w.rows == n_head_kv * n_embd_head && w.cols == n_embd,
                    "detokenizer layer {i} wv shape [{}, {}] mismatch expected [{}, {n_embd}]",
                    w.rows,
                    w.cols,
                    n_head_kv * n_embd_head
                );
                Some(w)
            } else {
                None
            };
            let wo = if !is_conv {
                let w = get_detok_mmap_weight(
                    gguf,
                    &format!("{prefix_lfm}.self_attn.out_proj.weight"),
                    &format!("{prefix_blk}.attn_output.weight"),
                )?;
                anyhow::ensure!(
                    w.rows == n_embd && w.cols == n_head * n_embd_head,
                    "detokenizer layer {i} wo shape [{}, {}] mismatch expected [{n_embd}, {}]",
                    w.rows,
                    w.cols,
                    n_head * n_embd_head
                );
                Some(w)
            } else {
                None
            };
            let q_norm = if !is_conv {
                let v = get_detok_tensor(
                    gguf,
                    &format!("{prefix_lfm}.self_attn.q_layernorm.weight"),
                    &format!("{prefix_blk}.attn_q_norm.weight"),
                )?
                .try_to_f32_vec()?;
                anyhow::ensure!(
                    v.len() == n_embd_head,
                    "detokenizer layer {i} q_norm len ({}) mismatch n_embd_head ({n_embd_head})",
                    v.len()
                );
                Some(v)
            } else {
                None
            };
            let k_norm = if !is_conv {
                let v = get_detok_tensor(
                    gguf,
                    &format!("{prefix_lfm}.self_attn.k_layernorm.weight"),
                    &format!("{prefix_blk}.attn_k_norm.weight"),
                )?
                .try_to_f32_vec()?;
                anyhow::ensure!(
                    v.len() == n_embd_head,
                    "detokenizer layer {i} k_norm len ({}) mismatch n_embd_head ({n_embd_head})",
                    v.len()
                );
                Some(v)
            } else {
                None
            };

            layers.push(DetokLayerWeights {
                operator_norm,
                ffn_norm,
                attn_post_norm: None,
                ffn_post_norm: None,
                ffn_w1,
                ffn_w2,
                ffn_w3,
                conv_in_proj,
                conv_out_proj,
                conv_weight,
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                attn_gate: None,
            });
        }

        Ok(Self {
            config,
            output_norm,
            emb_weight,
            lin_w,
            lin_b,
            layers,
            lfm3_fsq: None,
        })
    }

    /// Load `lfm3a_gen` FSQ + LFM3/LFM2 + ISTFT detokenizer weights.
    /// Supports `a.gen.wav.tfm.*` / `a.gen.wav.*` (upstream `lfm3.py` / `lfm3a-gen.cpp`) and `a.detok.*` tensor schemas.
    fn from_gguf_lfm3a_gen(gguf: &Arc<GgufFile>) -> Result<Self> {
        let tfm_in_w = MmapWeight::from_gguf(gguf, "a.gen.wav.tfm.in_proj.weight")
            .or_else(|_| MmapWeight::from_gguf(gguf, "a.gen.wav.tfm_in_proj.weight"))
            .or_else(|_| MmapWeight::from_gguf(gguf, "a.detok.tfm_in_proj.weight"))
            .context("loading lfm3a_gen tfm_in_proj.weight")?;
        let n_embd = tfm_in_w.rows;
        let fsq_proj_dim = tfm_in_w.cols; // 512 or 256
        anyhow::ensure!(
            n_embd > 0 && fsq_proj_dim > 0,
            "lfm3a_gen detok tfm_in_proj shape [{n_embd}, {fsq_proj_dim}] must be non-empty"
        );

        let tfm_in_b = if gguf.tensors.contains_key("a.gen.wav.tfm.in_proj.bias")
            || gguf.tensors.contains_key("a.gen.wav.tfm_in_proj.bias")
            || gguf.tensors.contains_key("a.detok.tfm_in_proj.bias")
        {
            gguf.get_tensor("a.gen.wav.tfm.in_proj.bias")
                .or_else(|_| gguf.get_tensor("a.gen.wav.tfm_in_proj.bias"))
                .or_else(|_| gguf.get_tensor("a.detok.tfm_in_proj.bias"))?
                .try_to_f32_vec()?
        } else {
            vec![0.0; n_embd]
        };
        anyhow::ensure!(
            tfm_in_b.len() == n_embd,
            "lfm3a_gen detok tfm_in_proj.bias len ({}) mismatch n_embd ({n_embd})",
            tfm_in_b.len()
        );

        let q_first_w = gguf
            .get_tensor("a.gen.wav.quant.first.out_proj.weight")
            .or_else(|_| gguf.get_tensor("a.gen.wav.quant_first_out.weight"))
            .or_else(|_| gguf.get_tensor("a.detok.quant_first_out.weight"))
            .context("loading lfm3a_gen quant_first_out.weight")?
            .try_to_f32_vec()?;
        let single_fsq_proj = q_first_w.len() == fsq_proj_dim * 48;
        anyhow::ensure!(
            q_first_w.len() == fsq_proj_dim * 16 || single_fsq_proj,
            "lfm3a_gen detok quant_first_out.weight len ({}) mismatch expected {} ({fsq_proj_dim} * 16) or {} ({fsq_proj_dim} * 48)",
            q_first_w.len(),
            fsq_proj_dim * 16,
            fsq_proj_dim * 48
        );
        let q_first_b = if gguf
            .tensors
            .contains_key("a.gen.wav.quant.first.out_proj.bias")
            || gguf.tensors.contains_key("a.gen.wav.quant_first_out.bias")
            || gguf.tensors.contains_key("a.detok.quant_first_out.bias")
        {
            gguf.get_tensor("a.gen.wav.quant.first.out_proj.bias")
                .or_else(|_| gguf.get_tensor("a.gen.wav.quant_first_out.bias"))
                .or_else(|_| gguf.get_tensor("a.detok.quant_first_out.bias"))?
                .try_to_f32_vec()?
        } else {
            vec![0.0; fsq_proj_dim]
        };
        anyhow::ensure!(
            q_first_b.len() == fsq_proj_dim,
            "lfm3a_gen detok quant_first_out.bias len ({}) mismatch fsq_proj_dim ({fsq_proj_dim})",
            q_first_b.len()
        );

        let (q_rest_w, q_rest_b) = if single_fsq_proj {
            (Vec::new(), Vec::new())
        } else {
            let rw = if gguf
                .tensors
                .contains_key("a.gen.wav.quant.rest.out_proj.weight")
                || gguf.tensors.contains_key("a.gen.wav.quant_rest_out.weight")
                || gguf.tensors.contains_key("a.detok.quant_rest_out.weight")
            {
                gguf.get_tensor("a.gen.wav.quant.rest.out_proj.weight")
                    .or_else(|_| gguf.get_tensor("a.gen.wav.quant_rest_out.weight"))
                    .or_else(|_| gguf.get_tensor("a.detok.quant_rest_out.weight"))?
                    .try_to_f32_vec()?
            } else {
                q_first_w.clone()
            };
            anyhow::ensure!(
                rw.len() == fsq_proj_dim * 16,
                "lfm3a_gen detok quant_rest_out.weight len ({}) mismatch expected {} ({fsq_proj_dim} * 16)",
                rw.len(),
                fsq_proj_dim * 16
            );
            let rb = if gguf
                .tensors
                .contains_key("a.gen.wav.quant.rest.out_proj.bias")
                || gguf.tensors.contains_key("a.gen.wav.quant_rest_out.bias")
                || gguf.tensors.contains_key("a.detok.quant_rest_out.bias")
            {
                gguf.get_tensor("a.gen.wav.quant.rest.out_proj.bias")
                    .or_else(|_| gguf.get_tensor("a.gen.wav.quant_rest_out.bias"))
                    .or_else(|_| gguf.get_tensor("a.detok.quant_rest_out.bias"))?
                    .try_to_f32_vec()?
            } else {
                q_first_b.clone()
            };
            anyhow::ensure!(
                rb.len() == fsq_proj_dim,
                "lfm3a_gen detok quant_rest_out.bias len ({}) mismatch fsq_proj_dim ({fsq_proj_dim})",
                rb.len()
            );
            (rw, rb)
        };

        let fsq_table = if gguf
            .tensors
            .contains_key("a.gen.wav.quant.first.codebook.weight")
            || gguf.tensors.contains_key("a.detok.fsq_codebook")
            || gguf.tensors.contains_key("a.gen.wav.fsq_codebook")
        {
            gguf.get_tensor("a.gen.wav.quant.first.codebook.weight")
                .or_else(|_| gguf.get_tensor("a.detok.fsq_codebook"))
                .or_else(|_| gguf.get_tensor("a.gen.wav.fsq_codebook"))?
                .try_to_f32_vec()?
        } else {
            build_default_fsq_codebook()
        };
        anyhow::ensure!(
            fsq_table.len() >= 256 * 4 && fsq_table.len().is_multiple_of(4),
            "lfm3a_gen detok fsq_codebook len ({}) must be a multiple of 4 and >= 1024",
            fsq_table.len()
        );

        let lfm3_fsq = Lfm3FsqWeights {
            fsq_codebook: fsq_table,
            fsq_proj_dim,
            quant_first_out_w: q_first_w,
            quant_first_out_b: q_first_b,
            quant_rest_out_w: q_rest_w,
            quant_rest_out_b: q_rest_b,
            tfm_in_proj_w: tfm_in_w,
            tfm_in_proj_b: tfm_in_b,
        };
        let emb_weight = MmapWeight::from_owned_f32(Vec::new(), 0, n_embd);

        let n_layer = gguf
            .get_u32("clip.audio_gen.wav.block_count")
            .or_else(|| gguf.get_u32("clip.detok.block_count"))
            .or_else(|| gguf.get_u32("clip.audio.detokenizer.block_count"))
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                let counted = (0..256usize)
                    .take_while(|&i| {
                        gguf.tensors
                            .contains_key(&format!("a.gen.wav.tfm.blk.{i}.ln1.weight"))
                            || gguf
                                .tensors
                                .contains_key(&format!("a.gen.wav.blk.{i}.ln1.weight"))
                            || gguf
                                .tensors
                                .contains_key(&format!("a.detok.blk.{i}.operator_norm.weight"))
                            || gguf
                                .tensors
                                .contains_key(&format!("a.detok.blk.{i}.ln1.weight"))
                    })
                    .count();
                if counted > 0 { counted } else { 8 }
            });
        anyhow::ensure!(
            (1..=256).contains(&n_layer),
            "lfm3a_gen detok n_layer ({n_layer}) must be in 1..=256"
        );
        let mut layer_is_conv = Vec::with_capacity(n_layer);
        let mut layers = Vec::with_capacity(n_layer);

        let mut n_head = gguf
            .get_u32("clip.audio_gen.wav.attention.head_count")
            .or_else(|| gguf.get_u32("clip.detok.head_count"))
            .or_else(|| gguf.get_u32("clip.audio.detokenizer.attention.head_count"))
            .unwrap_or(16) as usize;
        anyhow::ensure!(n_head > 0, "lfm3a_gen detok n_head must be > 0");
        let mut n_head_kv = gguf
            .get_u32("clip.audio_gen.wav.attention.head_count_kv")
            .or_else(|| gguf.get_u32("clip.detok.head_count_kv"))
            .or_else(|| gguf.get_u32("clip.audio.detokenizer.attention.head_count_kv"))
            .or_else(|| {
                gguf.get_i32_array("clip.audio_gen.wav.attention.head_count_kv")
                    .or_else(|| gguf.get_i32_array("clip.detok.head_count_kv"))
                    .or_else(|| {
                        gguf.get_i32_array("clip.audio.detokenizer.attention.head_count_kv")
                    })
                    .and_then(|arr| arr.iter().copied().find(|&v| v > 0).map(|v| v as u32))
            })
            .unwrap_or(8) as usize;
        anyhow::ensure!(n_head_kv > 0, "lfm3a_gen detok n_head_kv must be > 0");
        let mut n_embd_head = n_embd / n_head;
        let mut ffn_dim = gguf
            .get_u32("clip.audio_gen.wav.feed_forward_length")
            .or_else(|| gguf.get_u32("clip.detok.feed_forward_length"))
            .or_else(|| gguf.get_u32("clip.audio.detokenizer.feed_forward_length"))
            .unwrap_or(1792) as usize;
        let mut d_conv = 2usize;
        let mut seen_conv_block = false;
        let mut seen_attn_block = false;

        for i in 0..n_layer {
            let pfx_tfm = format!("a.gen.wav.tfm.blk.{i}");
            let pfx_wav = format!("a.gen.wav.blk.{i}");
            let pfx_det = format!("a.detok.blk.{i}");

            let operator_norm = gguf
                .get_tensor(&format!("{pfx_tfm}.ln1.weight"))
                .or_else(|_| gguf.get_tensor(&format!("{pfx_wav}.ln1.weight")))
                .or_else(|_| gguf.get_tensor(&format!("{pfx_det}.operator_norm.weight")))
                .or_else(|_| gguf.get_tensor(&format!("{pfx_det}.ln1.weight")))
                .with_context(|| format!("loading lfm3a_gen detok block {i} operator_norm"))?
                .try_to_f32_vec()?;
            anyhow::ensure!(
                operator_norm.len() == n_embd,
                "lfm3a_gen detok block {i} operator_norm len ({}) mismatch n_embd ({n_embd})",
                operator_norm.len()
            );

            let post_attn_tfm = format!("{pfx_tfm}.attn_post_norm.weight");
            let post_attn_wav = format!("{pfx_wav}.attn_post_norm.weight");
            let post_attn_det = format!("{pfx_det}.attn_post_norm.weight");
            let attn_post_norm = if gguf.tensors.contains_key(&post_attn_tfm)
                || gguf.tensors.contains_key(&post_attn_wav)
                || gguf.tensors.contains_key(&post_attn_det)
            {
                let v = gguf
                    .get_tensor(&post_attn_tfm)
                    .or_else(|_| gguf.get_tensor(&post_attn_wav))
                    .or_else(|_| gguf.get_tensor(&post_attn_det))?
                    .try_to_f32_vec()?;
                anyhow::ensure!(
                    v.len() == n_embd,
                    "lfm3a_gen detok block {i} attn_post_norm len ({}) mismatch n_embd ({n_embd})",
                    v.len()
                );
                Some(v)
            } else {
                None
            };

            let ffn_norm = gguf
                .get_tensor(&format!("{pfx_tfm}.ln2.weight"))
                .or_else(|_| gguf.get_tensor(&format!("{pfx_wav}.ln2.weight")))
                .or_else(|_| gguf.get_tensor(&format!("{pfx_det}.ffn_norm.weight")))
                .or_else(|_| gguf.get_tensor(&format!("{pfx_det}.ln2.weight")))
                .with_context(|| format!("loading lfm3a_gen detok block {i} ffn_norm"))?
                .try_to_f32_vec()?;
            anyhow::ensure!(
                ffn_norm.len() == n_embd,
                "lfm3a_gen detok block {i} ffn_norm len ({}) mismatch n_embd ({n_embd})",
                ffn_norm.len()
            );

            let post_ffn_tfm = format!("{pfx_tfm}.ffn_post_norm.weight");
            let post_ffn_wav = format!("{pfx_wav}.ffn_post_norm.weight");
            let post_ffn_det = format!("{pfx_det}.ffn_post_norm.weight");
            let ffn_post_norm = if gguf.tensors.contains_key(&post_ffn_tfm)
                || gguf.tensors.contains_key(&post_ffn_wav)
                || gguf.tensors.contains_key(&post_ffn_det)
            {
                let v = gguf
                    .get_tensor(&post_ffn_tfm)
                    .or_else(|_| gguf.get_tensor(&post_ffn_wav))
                    .or_else(|_| gguf.get_tensor(&post_ffn_det))?
                    .try_to_f32_vec()?;
                anyhow::ensure!(
                    v.len() == n_embd,
                    "lfm3a_gen detok block {i} ffn_post_norm len ({}) mismatch n_embd ({n_embd})",
                    v.len()
                );
                Some(v)
            } else {
                None
            };

            let ffn_w1 = MmapWeight::from_gguf(gguf, &format!("{pfx_tfm}.ffn_gate.weight"))
                .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_wav}.ffn_gate.weight")))
                .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.w1.weight")))
                .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.ffn_gate.weight")))
                .with_context(|| format!("loading lfm3a_gen detok block {i} ffn_w1"))?;
            let ffn_w2 = MmapWeight::from_gguf(gguf, &format!("{pfx_tfm}.ffn_down.weight"))
                .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_wav}.ffn_down.weight")))
                .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.w2.weight")))
                .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.ffn_down.weight")))
                .with_context(|| format!("loading lfm3a_gen detok block {i} ffn_w2"))?;
            let ffn_w3 = MmapWeight::from_gguf(gguf, &format!("{pfx_tfm}.ffn_up.weight"))
                .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_wav}.ffn_up.weight")))
                .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.w3.weight")))
                .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.ffn_up.weight")))
                .with_context(|| format!("loading lfm3a_gen detok block {i} ffn_w3"))?;
            if i == 0 {
                ffn_dim = ffn_w1.rows;
            }
            anyhow::ensure!(
                ffn_dim > 0
                    && ffn_w1.rows == ffn_dim
                    && ffn_w1.cols == n_embd
                    && ffn_w3.rows == ffn_dim
                    && ffn_w3.cols == n_embd
                    && ffn_w2.rows == n_embd
                    && ffn_w2.cols == ffn_dim,
                "lfm3a_gen detok block {i} FFN weight shape mismatch"
            );

            let conv_dw_tfm = format!("{pfx_tfm}.conv_dw.weight");
            let conv_dw_wav = format!("{pfx_wav}.conv_dw.weight");
            let conv_dw_det1 = format!("{pfx_det}.shortconv.conv.weight");
            let conv_dw_det2 = format!("{pfx_det}.conv_dw.weight");
            let is_conv_block = gguf.tensors.contains_key(&conv_dw_tfm)
                || gguf.tensors.contains_key(&conv_dw_wav)
                || gguf.tensors.contains_key(&conv_dw_det1)
                || gguf.tensors.contains_key(&conv_dw_det2);

            if is_conv_block {
                layer_is_conv.push(true);
                let dw_t = gguf
                    .get_tensor(&conv_dw_tfm)
                    .or_else(|_| gguf.get_tensor(&conv_dw_wav))
                    .or_else(|_| gguf.get_tensor(&conv_dw_det1))
                    .or_else(|_| gguf.get_tensor(&conv_dw_det2))
                    .with_context(|| format!("loading lfm3a_gen detok block {i} conv_dw"))?;
                let conv_w = dw_t.try_to_f32_vec()?;
                anyhow::ensure!(
                    conv_w.len() >= n_embd && conv_w.len().is_multiple_of(n_embd),
                    "lfm3a_gen detok block {i} conv_dw len ({}) must be a positive multiple of n_embd ({n_embd})",
                    conv_w.len()
                );
                let k_size = conv_w.len() / n_embd;
                let blk_d_conv = k_size.saturating_sub(1);
                if !seen_conv_block {
                    d_conv = blk_d_conv;
                    seen_conv_block = true;
                } else {
                    anyhow::ensure!(
                        blk_d_conv == d_conv,
                        "lfm3a_gen detok block {i} ShortConv d_conv ({blk_d_conv}) mismatch earlier ShortConv block ({d_conv})"
                    );
                }
                let in_proj = MmapWeight::from_gguf(gguf, &format!("{pfx_tfm}.conv_pw1.weight"))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_wav}.conv_pw1.weight")))
                    .or_else(|_| {
                        MmapWeight::from_gguf(gguf, &format!("{pfx_det}.shortconv.in_proj.weight"))
                    })
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.conv_pw1.weight")))
                    .with_context(|| format!("loading lfm3a_gen detok block {i} conv_in_proj"))?;
                anyhow::ensure!(
                    in_proj.rows == 3 * n_embd && in_proj.cols == n_embd,
                    "lfm3a_gen detok block {i} conv_in_proj shape [{}, {}] mismatch expected [{}, {n_embd}]",
                    in_proj.rows,
                    in_proj.cols,
                    3 * n_embd
                );
                let out_proj = MmapWeight::from_gguf(gguf, &format!("{pfx_tfm}.conv_pw2.weight"))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_wav}.conv_pw2.weight")))
                    .or_else(|_| {
                        MmapWeight::from_gguf(gguf, &format!("{pfx_det}.shortconv.out_proj.weight"))
                    })
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.conv_pw2.weight")))
                    .with_context(|| format!("loading lfm3a_gen detok block {i} conv_out_proj"))?;
                anyhow::ensure!(
                    out_proj.rows == n_embd && out_proj.cols == n_embd,
                    "lfm3a_gen detok block {i} conv_out_proj shape [{}, {}] mismatch expected [{n_embd}, {n_embd}]",
                    out_proj.rows,
                    out_proj.cols
                );

                layers.push(DetokLayerWeights {
                    operator_norm,
                    ffn_norm,
                    attn_post_norm,
                    ffn_post_norm,
                    ffn_w1,
                    ffn_w2,
                    ffn_w3,
                    conv_in_proj: Some(in_proj),
                    conv_out_proj: Some(out_proj),
                    conv_weight: Some(conv_w),
                    wq: None,
                    wk: None,
                    wv: None,
                    wo: None,
                    q_norm: None,
                    k_norm: None,
                    attn_gate: None,
                });
            } else {
                layer_is_conv.push(false);
                let q_norm = gguf
                    .get_tensor(&format!("{pfx_tfm}.attn_q_norm.weight"))
                    .or_else(|_| gguf.get_tensor(&format!("{pfx_wav}.attn_q_norm.weight")))
                    .or_else(|_| gguf.get_tensor(&format!("{pfx_det}.q_norm.weight")))
                    .or_else(|_| gguf.get_tensor(&format!("{pfx_det}.attn_q_norm.weight")))
                    .with_context(|| format!("loading lfm3a_gen detok block {i} q_norm"))?
                    .try_to_f32_vec()?;
                let k_norm = gguf
                    .get_tensor(&format!("{pfx_tfm}.attn_k_norm.weight"))
                    .or_else(|_| gguf.get_tensor(&format!("{pfx_wav}.attn_k_norm.weight")))
                    .or_else(|_| gguf.get_tensor(&format!("{pfx_det}.k_norm.weight")))
                    .or_else(|_| gguf.get_tensor(&format!("{pfx_det}.attn_k_norm.weight")))
                    .with_context(|| format!("loading lfm3a_gen detok block {i} k_norm"))?
                    .try_to_f32_vec()?;
                anyhow::ensure!(
                    !q_norm.is_empty() && q_norm.len() == k_norm.len(),
                    "lfm3a_gen detok block {i} q_norm/k_norm len ({}, {}) must be non-empty and equal",
                    q_norm.len(),
                    k_norm.len()
                );
                let blk_head_dim = q_norm.len();

                let wq = MmapWeight::from_gguf(gguf, &format!("{pfx_tfm}.attn_q.weight"))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_wav}.attn_q.weight")))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.wq.weight")))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.attn_q.weight")))
                    .with_context(|| format!("loading lfm3a_gen detok block {i} wq"))?;
                let wk = MmapWeight::from_gguf(gguf, &format!("{pfx_tfm}.attn_k.weight"))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_wav}.attn_k.weight")))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.wk.weight")))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.attn_k.weight")))
                    .with_context(|| format!("loading lfm3a_gen detok block {i} wk"))?;
                let wv = MmapWeight::from_gguf(gguf, &format!("{pfx_tfm}.attn_v.weight"))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_wav}.attn_v.weight")))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.wv.weight")))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.attn_v.weight")))
                    .with_context(|| format!("loading lfm3a_gen detok block {i} wv"))?;
                let wo = MmapWeight::from_gguf(gguf, &format!("{pfx_tfm}.attn_out.weight"))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_wav}.attn_out.weight")))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.wo.weight")))
                    .or_else(|_| MmapWeight::from_gguf(gguf, &format!("{pfx_det}.attn_out.weight")))
                    .with_context(|| format!("loading lfm3a_gen detok block {i} wo"))?;
                anyhow::ensure!(
                    wq.cols == n_embd
                        && wq.rows >= blk_head_dim
                        && wq.rows.is_multiple_of(blk_head_dim),
                    "lfm3a_gen detok block {i} wq shape [{}, {}] invalid for n_embd={n_embd}, n_embd_head={blk_head_dim}",
                    wq.rows,
                    wq.cols
                );
                anyhow::ensure!(
                    wk.cols == n_embd
                        && wk.rows >= blk_head_dim
                        && wk.rows.is_multiple_of(blk_head_dim)
                        && wv.cols == n_embd
                        && wv.rows == wk.rows,
                    "lfm3a_gen detok block {i} wk/wv shape [{}, {}] / [{}, {}] invalid for n_embd={n_embd}, n_embd_head={blk_head_dim}",
                    wk.rows,
                    wk.cols,
                    wv.rows,
                    wv.cols
                );
                let blk_n_head = wq.rows / blk_head_dim;
                let blk_n_head_kv = wk.rows / blk_head_dim;
                anyhow::ensure!(
                    blk_n_head.is_multiple_of(blk_n_head_kv),
                    "lfm3a_gen detok block {i} n_head ({blk_n_head}) must be a multiple of n_head_kv ({blk_n_head_kv})"
                );
                anyhow::ensure!(
                    wo.rows == n_embd && wo.cols == wq.rows,
                    "lfm3a_gen detok block {i} wo shape [{}, {}] mismatch expected [{n_embd}, {}]",
                    wo.rows,
                    wo.cols,
                    wq.rows
                );
                if !seen_attn_block {
                    n_embd_head = blk_head_dim;
                    n_head = blk_n_head;
                    n_head_kv = blk_n_head_kv;
                    seen_attn_block = true;
                } else {
                    anyhow::ensure!(
                        blk_head_dim == n_embd_head
                            && blk_n_head == n_head
                            && blk_n_head_kv == n_head_kv,
                        "lfm3a_gen detok block {i} attention dims (head_dim={blk_head_dim}, n_head={blk_n_head}, n_head_kv={blk_n_head_kv}) mismatch earlier attention block (head_dim={n_embd_head}, n_head={n_head}, n_head_kv={n_head_kv})"
                    );
                }

                let gate_tfm = format!("{pfx_tfm}.attn_gate.weight");
                let gate_wav = format!("{pfx_wav}.attn_gate.weight");
                let gate_det = format!("{pfx_det}.attn_gate.weight");
                let attn_gate = if gguf.tensors.contains_key(&gate_tfm)
                    || gguf.tensors.contains_key(&gate_wav)
                    || gguf.tensors.contains_key(&gate_det)
                {
                    let gw = MmapWeight::from_gguf(gguf, &gate_tfm)
                        .or_else(|_| MmapWeight::from_gguf(gguf, &gate_wav))
                        .or_else(|_| MmapWeight::from_gguf(gguf, &gate_det))?;
                    anyhow::ensure!(
                        gw.cols == n_embd && (gw.rows == n_head || gw.rows == n_head * n_embd_head),
                        "lfm3a_gen detok block {i} attn_gate shape [{}, {}] invalid (expected [{n_head}, {n_embd}] or [{}, {n_embd}])",
                        gw.rows,
                        gw.cols,
                        n_head * n_embd_head
                    );
                    Some(gw)
                } else {
                    None
                };

                layers.push(DetokLayerWeights {
                    operator_norm,
                    ffn_norm,
                    attn_post_norm,
                    ffn_post_norm,
                    ffn_w1,
                    ffn_w2,
                    ffn_w3,
                    conv_in_proj: None,
                    conv_out_proj: None,
                    conv_weight: None,
                    wq: Some(wq),
                    wk: Some(wk),
                    wv: Some(wv),
                    wo: Some(wo),
                    q_norm: Some(q_norm),
                    k_norm: Some(k_norm),
                    attn_gate,
                });
            }
        }

        anyhow::ensure!(
            n_head > 0 && n_head_kv > 0 && n_embd_head > 0 && n_head.is_multiple_of(n_head_kv),
            "lfm3a_gen detok head geometry invalid: n_head={n_head}, n_head_kv={n_head_kv}, n_embd_head={n_embd_head}"
        );

        let output_norm = gguf
            .get_tensor("a.gen.wav.tfm.output_norm.weight")
            .or_else(|_| gguf.get_tensor("a.gen.wav.tfm_output_norm.weight"))
            .or_else(|_| gguf.get_tensor("a.detok.output_norm.weight"))
            .or_else(|_| gguf.get_tensor("a.detok.tfm_output_norm.weight"))
            .context("loading lfm3a_gen detok output_norm.weight")?
            .try_to_f32_vec()?;
        anyhow::ensure!(
            output_norm.len() == n_embd,
            "lfm3a_gen detok output_norm len ({}) mismatch n_embd ({n_embd})",
            output_norm.len()
        );
        let lin_w = MmapWeight::from_gguf(gguf, "a.gen.wav.tfm.out_proj.weight")
            .or_else(|_| MmapWeight::from_gguf(gguf, "a.gen.wav.tfm_out_proj.weight"))
            .or_else(|_| MmapWeight::from_gguf(gguf, "a.detok.lin.weight"))
            .or_else(|_| MmapWeight::from_gguf(gguf, "a.detok.tfm_out_proj.weight"))
            .context("loading lfm3a_gen detok lin.weight")?;
        let lin_b = gguf
            .get_tensor("a.gen.wav.tfm.out_proj.bias")
            .or_else(|_| gguf.get_tensor("a.gen.wav.tfm_out_proj.bias"))
            .or_else(|_| gguf.get_tensor("a.detok.lin.bias"))
            .or_else(|_| gguf.get_tensor("a.detok.tfm_out_proj.bias"))
            .context("loading lfm3a_gen detok lin.bias")?
            .try_to_f32_vec()?;

        let n_fft = gguf
            .get_u32("clip.audio_gen.wav.n_fft")
            .or_else(|| gguf.get_u32("clip.detok.n_fft"))
            .or_else(|| gguf.get_u32("clip.audio.detokenizer.n_fft"))
            .unwrap_or(1280) as usize;
        let hop_length = gguf
            .get_u32("clip.audio_gen.wav.hop_length")
            .or_else(|| gguf.get_u32("clip.detok.hop_length"))
            .or_else(|| gguf.get_u32("clip.audio.detokenizer.hop_length"))
            .unwrap_or(320) as usize;
        let swa_window_size = gguf
            .get_u32("clip.audio_gen.wav.window_size")
            .or_else(|| gguf.get_u32("clip.detok.sliding_window"))
            .or_else(|| gguf.get_u32("clip.audio.detokenizer.sliding_window"))
            .unwrap_or(30) as usize;
        anyhow::ensure!(
            n_fft >= 2 && n_fft.is_multiple_of(2) && hop_length > 0 && hop_length <= n_fft,
            "lfm3a_gen detok STFT params invalid: n_fft={n_fft}, hop_length={hop_length}"
        );
        anyhow::ensure!(
            (1..=4096).contains(&swa_window_size),
            "lfm3a_gen detok swa_window_size ({swa_window_size}) must be in 1..=4096"
        );
        let single_frame_bins = (n_fft / 2 + 1) * 2;
        anyhow::ensure!(
            lin_w.cols == n_embd
                && lin_b.len() == lin_w.rows
                && (lin_w.rows == single_frame_bins || lin_w.rows == 6 * single_frame_bins),
            "lfm3a_gen detok lin_w [{}, {}] / lin_b ({}) mismatch expected [{single_frame_bins} or {}, {n_embd}]",
            lin_w.rows,
            lin_w.cols,
            lin_b.len(),
            6 * single_frame_bins
        );
        let rms_norm_eps = gguf
            .get_f32("clip.audio_gen.wav.attention.layer_norm_epsilon")
            .or_else(|| gguf.get_f32("clip.detok.layer_norm_epsilon"))
            .or_else(|| gguf.get_f32("clip.audio.detokenizer.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        anyhow::ensure!(
            rms_norm_eps.is_finite() && rms_norm_eps > 0.0,
            "lfm3a_gen detok rms_norm_eps ({rms_norm_eps}) must be finite and > 0"
        );
        let rope_freq_base = gguf
            .get_f32("clip.audio_gen.wav.rope.freq_base")
            .or_else(|| gguf.get_f32("clip.detok.rope.freq_base"))
            .or_else(|| gguf.get_f32("clip.audio.detokenizer.rope.freq_base"))
            .unwrap_or(1_000_000.0);
        anyhow::ensure!(
            rope_freq_base.is_finite() && rope_freq_base > 0.0,
            "lfm3a_gen detok rope_freq_base ({rope_freq_base}) must be finite and > 0"
        );

        let config = DetokenizerConfig {
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            n_embd_head,
            ffn_dim,
            d_conv,
            rms_norm_eps,
            rope_freq_base,
            swa_window_size,
            n_codes: 12,
            n_fft,
            hop_length,
            sample_rate: 24000,
            layer_is_conv,
        };

        Ok(Self {
            config,
            output_norm,
            emb_weight,
            lin_w,
            lin_b,
            layers,
            lfm3_fsq: Some(lfm3_fsq),
        })
    }
}

// ── Detokenizer forward pass ────────────────────────────────────────────

/// Runtime state for the detokenizer's LFM2/LFM3 backbone and streaming ISTFT.
pub struct DetokenizerState {
    /// Per-layer conv rolling buffer [d_conv, n_embd].
    conv_bufs: Vec<Vec<f32>>,
    /// Per-layer KV cache for attention layers.
    attn_kv: Vec<Option<(Vec<f32>, Vec<f32>)>>,
    pub(crate) n_past: usize,
    /// Stateful overlap-add ISTFT processor for `detokenize_streaming`.
    istft_streamer: IstftStreamer,
}

impl DetokenizerState {
    pub fn new(cfg: &DetokenizerConfig) -> Self {
        let mut conv_bufs = Vec::new();
        let mut attn_kv = Vec::new();
        let kv_size = cfg.swa_window_size * cfg.n_head_kv * cfg.n_embd_head;
        for &is_conv in &cfg.layer_is_conv {
            if is_conv {
                conv_bufs.push(vec![0.0; cfg.d_conv * cfg.n_embd]);
                attn_kv.push(None);
            } else {
                conv_bufs.push(vec![]); // placeholder
                attn_kv.push(Some((vec![0.0; kv_size], vec![0.0; kv_size])));
            }
        }
        let mut istft_streamer = IstftStreamer::new(cfg.n_fft.max(2), cfg.hop_length.max(1));
        // streaming returns whole frames: the start-up crop is the caller's
        istft_streamer.padding_remaining = 0;
        Self {
            conv_bufs,
            attn_kv,
            n_past: 0,
            istft_streamer,
        }
    }

    pub fn reset(&mut self) {
        for buf in &mut self.conv_bufs {
            buf.fill(0.0);
        }
        for (k, v) in self.attn_kv.iter_mut().flatten() {
            k.fill(0.0);
            v.fill(0.0);
        }
        self.n_past = 0;
        self.istft_streamer.overlap_buf.fill(0.0);
        self.istft_streamer.window_sum.fill(0.0);
        self.istft_streamer.padding_remaining = 0;
    }
}

/// Embed audio codes into a single `[n_embd]` vector for the detokenizer.
///
/// For `lfm2a` (8 codes): looks up each code with per-codebook offset and averages across codebooks.
/// For `lfm3a_gen` (12 codes): evaluates the typed 4D 4-level FSQ codebook + `quant_first_out` /
/// `quant_rest_out` + `tfm_in_proj` (or falls back to direct table lookup when `lfm3_fsq` is `None`).
pub fn detok_embed_codes(weights: &DetokenizerWeights, codes: &[i32]) -> Vec<f32> {
    let n_codes = weights.config.n_codes;
    let emb_dim = weights.config.n_embd;

    if let Some(fsq) = &weights.lfm3_fsq {
        let fsq_proj_dim = fsq.fsq_proj_dim;
        let max_code = ((fsq.fsq_codebook.len() / 4).saturating_sub(1)) as i32;
        let mut fsq_coords = [0.0f32; 48];
        for j in 0..12 {
            let c = codes.get(j).copied().unwrap_or(0).clamp(0, max_code) as usize;
            fsq_coords[j * 4..(j + 1) * 4].copy_from_slice(&fsq.fsq_codebook[c * 4..(c + 1) * 4]);
        }

        let mut mean_proj = vec![0.0f32; fsq_proj_dim];
        if fsq.quant_first_out_w.len() == fsq_proj_dim * 48 {
            for r in 0..fsq_proj_dim {
                let row = &fsq.quant_first_out_w[r * 48..(r + 1) * 48];
                mean_proj[r] = row.iter().zip(&fsq_coords).map(|(w, x)| w * x).sum::<f32>()
                    + fsq.quant_first_out_b[r];
            }
        } else {
            for grp in 0..3 {
                let grp_in = &fsq_coords[grp * 16..(grp + 1) * 16];
                let (gw, gb) = if grp == 0 {
                    (
                        fsq.quant_first_out_w.as_slice(),
                        fsq.quant_first_out_b.as_slice(),
                    )
                } else {
                    (
                        fsq.quant_rest_out_w.as_slice(),
                        fsq.quant_rest_out_b.as_slice(),
                    )
                };
                for r in 0..fsq_proj_dim {
                    let row = &gw[r * 16..(r + 1) * 16];
                    let sum: f32 = row.iter().zip(grp_in).map(|(w, x)| w * x).sum::<f32>() + gb[r];
                    mean_proj[r] += sum / 3.0;
                }
            }
        }

        let mut out = vec![0.0f32; emb_dim];
        fsq.tfm_in_proj_w.gemv(&mean_proj, &mut out);
        for (dst, &b) in out.iter_mut().zip(fsq.tfm_in_proj_b.iter()) {
            *dst += b;
        }
        return out;
    }

    let emb = &weights.emb_weight;
    let n_vocab_per_cb = (emb.rows / n_codes.max(1)).max(1);
    let mut result = vec![0.0f32; emb_dim];
    let mut row = vec![0f32; emb_dim];
    for (j, &code) in codes.iter().take(n_codes).enumerate() {
        if code >= 0 {
            let idx = j * n_vocab_per_cb + code as usize;
            if idx < emb.rows {
                emb.dequantize_row(idx, &mut row);
                for (r, e) in result.iter_mut().zip(&row) {
                    *r += e;
                }
            }
        }
    }
    let scale = 1.0 / n_codes.max(1) as f32;
    for r in &mut result {
        *r *= scale;
    }
    result
}

/// Linear interpolation upsample: 1 token to n_up tokens.
pub fn upsample(input: &[f32], n_embd: usize, n_up: usize) -> Vec<f32> {
    if n_embd == 0 || n_up == 0 || input.is_empty() {
        return Vec::new();
    }
    let n_in = input.len() / n_embd;
    if n_in == 0 {
        return Vec::new();
    }
    let n_out = n_in * n_up;
    if n_in == 1 {
        let mut output = vec![0.0; n_out * n_embd];
        let token = &input[..n_embd];
        for chunk in output.chunks_exact_mut(n_embd) {
            chunk.copy_from_slice(token);
        }
        return output;
    }
    let mut output = vec![0.0; n_out * n_embd];
    for i in 0..n_out {
        let src_f = i as f32 / n_up as f32;
        let src_i = src_f as usize;
        let frac = src_f - src_i as f32;
        let i0 = src_i.min(n_in - 1);
        let i1 = (src_i + 1).min(n_in - 1);
        for d in 0..n_embd {
            output[i * n_embd + d] =
                input[i0 * n_embd + d] * (1.0 - frac) + input[i1 * n_embd + d] * frac;
        }
    }
    output
}

/// Run the detokenizer: codes -> spectrogram frames (before ISTFT).
///
/// Returns raw spectrogram data as [n_frames x n_fft_bins x 2] (log_abs, angle).
pub fn detokenize_to_spectrum(
    weights: &DetokenizerWeights,
    _detok_weights: &AudioDecoderWeights,
    state: &mut DetokenizerState,
    codes: &[i32],
) -> Vec<f32> {
    detokenize_to_spectrum_with_state(weights, state, codes)
}

/// Run the detokenizer with state: codes -> spectrogram frames (before ISTFT).
pub fn detokenize_to_spectrum_with_state(
    weights: &DetokenizerWeights,
    state: &mut DetokenizerState,
    codes: &[i32],
) -> Vec<f32> {
    let cfg = &weights.config;
    let n_embd = cfg.n_embd;

    // 1. Embed codes -> single vector.
    let embedding = detok_embed_codes(weights, codes);

    // 2. For `lfm2a` (or when `lin_w.rows == (n_fft/2 + 1)*2`), upsample 1 -> 6 tokens before the
    //    transformer. For `lfm3a_gen` when `tfm_out_proj` (`lin_w`) directly outputs all 6 STFT
    //    frames (`lin_w.rows == 6 * (n_fft/2 + 1)*2`), run the transformer at the 1-token frame rate.
    let single_frame_bins = (cfg.n_fft / 2 + 1) * 2;
    let tokens = if weights.lin_w.rows == 6 * single_frame_bins {
        embedding
    } else {
        upsample(&embedding, n_embd, 6)
    };
    let n_tokens = tokens.len() / n_embd;

    // 3. LFM2/LFM3 backbone: layer-by-layer with all n_tokens.
    let mut hidden = tokens;
    let mut norm_mat = vec![0.0f32; n_tokens * n_embd];
    let mut bcx_mat = vec![0.0f32; n_tokens * 3 * n_embd];
    let mut conv_y_mat = vec![0.0f32; n_tokens * n_embd];
    let mut op_out_mat = vec![0.0f32; n_tokens * n_embd];
    let mut gate_mat = vec![0.0f32; n_tokens * cfg.ffn_dim];
    let mut up_mat = vec![0.0f32; n_tokens * cfg.ffn_dim];
    let mut down_mat = vec![0.0f32; n_tokens * n_embd];

    for (il, lw) in weights.layers.iter().enumerate() {
        let op_norm = &lw.operator_norm[..n_embd];
        let post_op_norm = lw.attn_post_norm.as_deref();
        let ffn_norm = &lw.ffn_norm[..n_embd];
        let post_ffn_norm = lw.ffn_post_norm.as_deref();

        norm_mat.copy_from_slice(&hidden);
        for t in 0..n_tokens {
            cpu::rmsnorm(
                &mut norm_mat[t * n_embd..(t + 1) * n_embd],
                op_norm,
                cfg.rms_norm_eps,
            );
        }

        if cfg.layer_is_conv[il] {
            if let (Some(in_proj), Some(conv_w), Some(out_proj)) =
                (&lw.conv_in_proj, &lw.conv_weight, &lw.conv_out_proj)
            {
                in_proj.batched_matmul(&norm_mat, &mut bcx_mat, n_tokens);
                let conv_buf = &mut state.conv_bufs[il];
                let d_conv = cfg.d_conv;
                let kernel_size = d_conv + 1;
                let mut bx = vec![0.0f32; n_embd];
                for t in 0..n_tokens {
                    let bcx = &bcx_mat[t * 3 * n_embd..(t + 1) * 3 * n_embd];
                    let b = &bcx[..n_embd];
                    let c = &bcx[n_embd..2 * n_embd];
                    let x = &bcx[2 * n_embd..];
                    for ch in 0..n_embd {
                        bx[ch] = b[ch] * x[ch];
                    }
                    let y_row = &mut conv_y_mat[t * n_embd..(t + 1) * n_embd];
                    for ch in 0..n_embd {
                        let mut sum = 0.0f32;
                        for k in 0..d_conv {
                            sum += conv_buf[k * n_embd + ch] * conv_w[ch * kernel_size + k];
                        }
                        sum += bx[ch] * conv_w[ch * kernel_size + d_conv];
                        y_row[ch] = c[ch] * sum;
                    }
                    if d_conv > 1 {
                        for k in 0..d_conv - 1 {
                            let src = (k + 1) * n_embd;
                            let dst = k * n_embd;
                            conv_buf.copy_within(src..src + n_embd, dst);
                        }
                    }
                    if d_conv > 0 {
                        conv_buf[(d_conv - 1) * n_embd..d_conv * n_embd].copy_from_slice(&bx);
                    }
                }
                out_proj.batched_matmul(&conv_y_mat, &mut op_out_mat, n_tokens);
            } else {
                op_out_mat.copy_from_slice(&norm_mat);
            }
            if let Some(pn) = post_op_norm {
                for t in 0..n_tokens {
                    cpu::rmsnorm(
                        &mut op_out_mat[t * n_embd..(t + 1) * n_embd],
                        pn,
                        cfg.rms_norm_eps,
                    );
                }
            }
            cpu::add_inplace(&mut hidden, &op_out_mat);
        } else {
            let (Some(wq), Some(wk), Some(wv), Some(wo), Some(q_norm), Some(k_norm), Some(kv)) = (
                &lw.wq,
                &lw.wk,
                &lw.wv,
                &lw.wo,
                &lw.q_norm,
                &lw.k_norm,
                state.attn_kv.get_mut(il).and_then(|opt| opt.as_mut()),
            ) else {
                continue;
            };
            let n_head = cfg.n_head;
            let n_kv = cfg.n_head_kv;
            let hd = cfg.n_embd_head;
            let q_dim = n_head * hd;
            let kv_stride = n_kv * hd;
            let group_size = n_head / n_kv;
            let scale = 1.0 / (hd as f32).sqrt();
            let mut all_q = vec![0.0f32; n_tokens * q_dim];
            let mut all_k = vec![0.0f32; n_tokens * kv_stride];
            let mut all_v = vec![0.0f32; n_tokens * kv_stride];
            wq.batched_matmul(&norm_mat, &mut all_q, n_tokens);
            wk.batched_matmul(&norm_mat, &mut all_k, n_tokens);
            wv.batched_matmul(&norm_mat, &mut all_v, n_tokens);

            let gate_rows = lw.attn_gate.as_ref().map(|g| g.rows).unwrap_or(0);
            let mut all_gate = if let Some(gate_w) = &lw.attn_gate {
                let mut g = vec![0.0f32; n_tokens * gate_rows];
                gate_w.batched_matmul(&norm_mat, &mut g, n_tokens);
                g
            } else {
                Vec::new()
            };

            for t in 0..n_tokens {
                let pos = state.n_past + t;
                let q = &mut all_q[t * q_dim..(t + 1) * q_dim];
                let k = &mut all_k[t * kv_stride..(t + 1) * kv_stride];
                let v = &all_v[t * kv_stride..(t + 1) * kv_stride];
                for h in 0..n_head {
                    cpu::rmsnorm(&mut q[h * hd..(h + 1) * hd], q_norm, cfg.rms_norm_eps);
                    apply_rope_neox(&mut q[h * hd..(h + 1) * hd], pos, hd, cfg.rope_freq_base);
                }
                for h in 0..n_kv {
                    cpu::rmsnorm(&mut k[h * hd..(h + 1) * hd], k_norm, cfg.rms_norm_eps);
                    apply_rope_neox(&mut k[h * hd..(h + 1) * hd], pos, hd, cfg.rope_freq_base);
                }
                // The non-causal variant attends over the batch's final window, so its whole
                // batch is written first. The causal one writes each position just before it
                // attends (below): with a ring of `swa_window_size` slots, writing a later
                // position first would overwrite a slot an earlier token still reads.
                if !weights.is_lfm3a_gen() {
                    let wp = pos % cfg.swa_window_size;
                    kv.0[wp * kv_stride..(wp + 1) * kv_stride].copy_from_slice(k);
                    kv.1[wp * kv_stride..(wp + 1) * kv_stride].copy_from_slice(v);
                }
            }

            let mut attn_out_mat = vec![0.0f32; n_tokens * q_dim];
            for t in 0..n_tokens {
                if weights.is_lfm3a_gen() {
                    let wp = (state.n_past + t) % cfg.swa_window_size;
                    kv.0[wp * kv_stride..(wp + 1) * kv_stride]
                        .copy_from_slice(&all_k[t * kv_stride..(t + 1) * kv_stride]);
                    kv.1[wp * kv_stride..(wp + 1) * kv_stride]
                        .copy_from_slice(&all_v[t * kv_stride..(t + 1) * kv_stride]);
                }
                let q = &all_q[t * q_dim..(t + 1) * q_dim];
                let kv_end = state.n_past + t + 1;
                let kv_end_batch = state.n_past + n_tokens;
                let eff_end = if weights.is_lfm3a_gen() {
                    kv_end
                } else {
                    kv_end_batch
                };
                let kv_start = eff_end.saturating_sub(cfg.swa_window_size);
                let kv_len = eff_end - kv_start;
                let attn_out = &mut attn_out_mat[t * q_dim..(t + 1) * q_dim];
                let mut scores = vec![0.0f32; kv_len];
                for h in 0..n_head {
                    let kv_h = h / group_size;
                    let qh = &q[h * hd..(h + 1) * hd];
                    for tt in 0..kv_len {
                        let cp = (kv_start + tt) % cfg.swa_window_size;
                        let ko = cp * kv_stride + kv_h * hd;
                        scores[tt] = cpu::dot_f32(qh, &kv.0[ko..ko + hd]) * scale;
                    }
                    cpu::softmax_inplace(&mut scores);
                    let out = &mut attn_out[h * hd..(h + 1) * hd];
                    for tt in 0..kv_len {
                        let cp = (kv_start + tt) % cfg.swa_window_size;
                        let vo = cp * kv_stride + kv_h * hd;
                        let s = scores[tt];
                        for d in 0..hd {
                            out[d] += s * kv.1[vo + d];
                        }
                    }
                }
                if gate_rows > 0 {
                    let g_row = &mut all_gate[t * gate_rows..(t + 1) * gate_rows];
                    if gate_rows == n_head {
                        for h in 0..n_head {
                            let s = 1.0f32 / (1.0f32 + (-g_row[h]).exp());
                            for v in &mut attn_out[h * hd..(h + 1) * hd] {
                                *v *= s;
                            }
                        }
                    } else {
                        for (av, &g) in attn_out.iter_mut().zip(g_row.iter()) {
                            *av *= 1.0f32 / (1.0f32 + (-g).exp());
                        }
                    }
                }
            }
            wo.batched_matmul(&attn_out_mat, &mut op_out_mat, n_tokens);
            if let Some(pn) = post_op_norm {
                for t in 0..n_tokens {
                    cpu::rmsnorm(
                        &mut op_out_mat[t * n_embd..(t + 1) * n_embd],
                        pn,
                        cfg.rms_norm_eps,
                    );
                }
            }
            cpu::add_inplace(&mut hidden, &op_out_mat);
        }

        // Batched SwiGLU FFN across all `n_tokens`.
        norm_mat.copy_from_slice(&hidden);
        for t in 0..n_tokens {
            cpu::rmsnorm(
                &mut norm_mat[t * n_embd..(t + 1) * n_embd],
                ffn_norm,
                cfg.rms_norm_eps,
            );
        }
        lw.ffn_w1.batched_matmul(&norm_mat, &mut gate_mat, n_tokens);
        lw.ffn_w3.batched_matmul(&norm_mat, &mut up_mat, n_tokens);
        cpu::silu_mul_inplace(&mut gate_mat, &up_mat);
        lw.ffn_w2.batched_matmul(&gate_mat, &mut down_mat, n_tokens);
        if let Some(pn) = post_ffn_norm {
            for t in 0..n_tokens {
                cpu::rmsnorm(
                    &mut down_mat[t * n_embd..(t + 1) * n_embd],
                    pn,
                    cfg.rms_norm_eps,
                );
            }
        }
        cpu::add_inplace(&mut hidden, &down_mat);
    }

    state.n_past += n_tokens;

    // Output norm per token.
    for t in 0..n_tokens {
        cpu::rmsnorm(
            &mut hidden[t * n_embd..(t + 1) * n_embd],
            &weights.output_norm,
            cfg.rms_norm_eps,
        );
    }
    // 4. Linear projection -> spectrogram.
    let lin_out_dim = weights.lin_w.rows;
    let mut spectrum = vec![0.0f32; n_tokens * lin_out_dim];
    weights
        .lin_w
        .batched_matmul(&hidden, &mut spectrum, n_tokens);
    for t in 0..n_tokens {
        cpu::add_inplace(
            &mut spectrum[t * lin_out_dim..(t + 1) * lin_out_dim],
            &weights.lin_b,
        );
    }

    spectrum
}

/// Streamingly detokenize one frame of audio codes (`8` codes for `lfm2a`, `12` codes for `lfm3a_gen`)
/// directly to `6 * hop_length` (`1920`) PCM samples at 24 kHz using stateful ShortConv, KV cache,
/// and overlap-add ISTFT buffers inside `state`.
///
/// Every frame returns the full `6 * hop_length` samples, including the first: the
/// `(n_fft - hop_length) / 2` start-up samples that [`detokenize`] crops from the head of a
/// sequence are kept here, so a caller that wants the cropped signal drops that many samples
/// from the front of the first frame (a test pins the two as equal past that head).
pub fn detokenize_streaming(
    weights: &DetokenizerWeights,
    state: &mut DetokenizerState,
    codes: &[i32],
) -> Vec<f32> {
    let spectrum = detokenize_to_spectrum_with_state(weights, state, codes);
    state.istft_streamer.feed_frames(&spectrum)
}

/// Detokenize a full sequence of code frames into 24 kHz mono PCM samples.
pub fn detokenize(weights: &DetokenizerWeights, codes_seq: &[Vec<i32>]) -> Vec<f32> {
    if codes_seq.is_empty() {
        return Vec::new();
    }
    let mut state = DetokenizerState::new(&weights.config);
    let mut all_spectrum = Vec::new();
    for codes in codes_seq {
        let spec = detokenize_to_spectrum_with_state(weights, &mut state, codes);
        all_spectrum.extend_from_slice(&spec);
    }
    istft_to_pcm(
        &all_spectrum,
        weights.config.n_fft,
        weights.config.hop_length,
    )
}

/// Convert spectrogram frames to PCM audio samples via ISTFT.
///
/// Input: [n_frames × n_fft_bins × 2] where each pair is (log_abs, angle).
/// Output: PCM float32 samples at the configured sample rate.
/// Streaming ISTFT matching llama.cpp's `mtmd_audio_streaming_istft`.
///
/// Processes frames one at a time with shift-and-accumulate overlap buffers.
/// Each frame produces exactly `hop_length` output samples.
pub fn istft_to_pcm(spectrum: &[f32], n_fft: usize, hop_length: usize) -> Vec<f32> {
    let n_fft_bins = n_fft / 2 + 1;
    let frame_size = n_fft_bins * 2;
    let n_frames = spectrum.len() / frame_size;
    if n_frames == 0 {
        return vec![];
    }

    // Pre-plan the IFFT (reused for every frame).
    let mut planner = rustfft::FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);

    // Periodic Hann window: w[n] = 0.5 * (1 - cos(2π*n/N)).
    let hann = build_hann(n_fft);

    // Streaming overlap buffers (size = n_fft).
    let mut overlap_buf = vec![0.0f32; n_fft];
    let mut window_sum = vec![0.0f32; n_fft];
    let mut output = Vec::with_capacity(n_frames * hop_length);

    // Pre-allocated scratch (reused across frames to avoid per-frame heap allocs).
    use rustfft::num_complex::Complex32;
    let mut fft_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n_fft];
    let mut time_domain = vec![0.0f32; n_fft];

    for i in 0..n_frames {
        // Convert (log_abs, angle) → complex, write directly into fft_buf.
        for c in fft_buf.iter_mut() {
            *c = Complex32::new(0.0, 0.0);
        }
        for j in 0..n_fft_bins {
            let log_abs = spectrum[i * frame_size + j];
            let angle = spectrum[i * frame_size + n_fft_bins + j];
            let mag = log_abs.exp();
            fft_buf[j] = Complex32::new(mag * angle.cos(), mag * angle.sin());
        }
        // Mirror negative frequencies (conjugate).
        for j in 1..n_fft_bins - 1 {
            fft_buf[n_fft - j] = Complex32::new(fft_buf[j].re, -fft_buf[j].im);
        }

        // IFFT (in-place on fft_buf, results written to time_domain).
        ifft_frame(&mut fft_buf, &mut time_domain, n_fft, ifft.as_ref());

        // Add to overlap buffer with Hann window.
        for j in 0..n_fft {
            overlap_buf[j] += time_domain[j] * hann[j];
            window_sum[j] += hann[j] * hann[j];
        }

        // Extract hop_length samples with normalization.
        for k in 0..hop_length {
            let sample = if window_sum[k] > 1e-8 {
                overlap_buf[k] / window_sum[k]
            } else {
                overlap_buf[k]
            };
            output.push(sample);
        }

        // Shift buffers left by hop_length.
        overlap_buf.copy_within(hop_length..n_fft, 0);
        overlap_buf[n_fft - hop_length..].fill(0.0);
        window_sum.copy_within(hop_length..n_fft, 0);
        window_sum[n_fft - hop_length..].fill(0.0);
    }

    // Strip startup padding: the first (n_fft - hop_length) / 2 samples are
    // overlap-add artifacts. Matches llama.cpp's `padding_to_remove`.
    let padding = n_fft.saturating_sub(hop_length) / 2;
    if output.len() > padding {
        output.drain(..padding);
    }
    output
}

/// Periodic Hann window of length `n_fft`: `w[n] = 0.5·(1 - cos(2π·n/n_fft))`.
///
/// Shared by the CPU `istft_to_pcm` and the GPU ISTFT overlap-add, which uploads
/// this vector once and reads `hann[local]` / `hann[local]²` per output sample.
pub fn build_hann(n_fft: usize) -> Vec<f32> {
    (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos()))
        .collect()
}

/// Build the real inverse-DFT basis for the GPU ISTFT.
///
/// Returns a row-major `[n_fft × (2·n_fft_bins)]` matrix `B` such that a frame's
/// time-domain signal is `time[t] = Σ_j halfspec[j] · B[t][j]`, where
/// `halfspec = [re_0..re_{bins-1}, im_0..im_{bins-1}]` is the interleaved
/// real/imag half-spectrum produced by `exp_polar`. The Hermitian mirror of the
/// negative frequencies and the `1/n_fft` inverse-transform scale are folded into
/// the coefficients, so the GPU matmul consumes the half-spectrum directly (no
/// explicit negative-frequency mirror pass, no radix FFT). This reproduces the
/// CPU `ifft_frame` real output bin-for-bin up to float rounding:
///
/// - DC (`k = 0`): `re` coefficient `1/n_fft`, `im` coefficient `0`.
/// - interior (`1 ≤ k ≤ n_fft/2 - 1`): doubled, `re = (2/n_fft)·cos(2πkt/n_fft)`,
///   `im = -(2/n_fft)·sin(2πkt/n_fft)`.
/// - Nyquist (`k = n_fft/2`): `re = (1/n_fft)·cos(πt)`, `im` coefficient `0`.
///
/// The `im` coefficients at DC and Nyquist are zero because those bins are their
/// own conjugate mirror, exactly as the CPU discards the imaginary part there.
///
/// Assumes an even `n_fft` (the STFT convention: the last of `n_fft/2 + 1` bins
/// is Nyquist), matching the CPU `istft_to_pcm` mirror loop. Every shipping
/// vocoder uses an even `n_fft` (1280 here).
pub fn build_idft_basis(n_fft: usize) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let n = n_fft as f32;
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut basis = vec![0.0f32; n_fft * 2 * bins];
    for t in 0..n_fft {
        let row = t * 2 * bins;
        for k in 0..bins {
            let ang = two_pi * k as f32 * t as f32 / n;
            let (re_c, im_c) = if k == 0 {
                (1.0 / n, 0.0)
            } else if k == bins - 1 {
                // Nyquist: cos(πt), no imaginary contribution.
                (ang.cos() / n, 0.0)
            } else {
                (2.0 * ang.cos() / n, -2.0 * ang.sin() / n)
            };
            basis[row + k] = re_c;
            basis[row + bins + k] = im_c;
        }
    }
    basis
}

/// Inverse FFT of one frame using a pre-planned FFT.
/// Processes `fft_buf` in place and writes real parts (scaled by 1/n_fft) into `out`.
fn ifft_frame(
    fft_buf: &mut [rustfft::num_complex::Complex32],
    out: &mut [f32],
    n_fft: usize,
    ifft: &dyn rustfft::Fft<f32>,
) {
    ifft.process(fft_buf);
    let inv_n = 1.0 / n_fft as f32;
    for (o, c) in out.iter_mut().zip(fft_buf.iter()) {
        *o = c.re * inv_n;
    }
}

/// Stateful streaming inverse STFT (overlap-add) processor.
///
/// Maintains overlap-add state across streaming frame decodes so PCM audio can
/// be streamed progressively with zero boundary discontinuities or fade artifacts.
pub struct IstftStreamer {
    n_fft: usize,
    hop_length: usize,
    n_fft_bins: usize,
    frame_size: usize,
    hann: Vec<f32>,
    overlap_buf: Vec<f32>,
    window_sum: Vec<f32>,
    padding_remaining: usize,
    fft_buf: Vec<rustfft::num_complex::Complex32>,
    time_domain: Vec<f32>,
    ifft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl IstftStreamer {
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        let mut planner = rustfft::FftPlanner::new();
        let ifft = planner.plan_fft_inverse(n_fft);
        let n_fft_bins = n_fft / 2 + 1;
        let frame_size = n_fft_bins * 2;
        let hann = build_hann(n_fft);
        let padding = n_fft.saturating_sub(hop_length) / 2;
        Self {
            n_fft,
            hop_length,
            n_fft_bins,
            frame_size,
            hann,
            overlap_buf: vec![0.0f32; n_fft],
            window_sum: vec![0.0f32; n_fft],
            padding_remaining: padding,
            fft_buf: vec![rustfft::num_complex::Complex32::new(0.0, 0.0); n_fft],
            time_domain: vec![0.0f32; n_fft],
            ifft,
        }
    }

    /// Feed spectrum floats for 1 or more frames and return newly decoded PCM samples.
    pub fn feed_frames(&mut self, spectrum: &[f32]) -> Vec<f32> {
        let n_frames = spectrum.len() / self.frame_size;
        if n_frames == 0 {
            return vec![];
        }
        let mut output = Vec::with_capacity(n_frames * self.hop_length);
        for i in 0..n_frames {
            for c in self.fft_buf.iter_mut() {
                *c = rustfft::num_complex::Complex32::new(0.0, 0.0);
            }
            for j in 0..self.n_fft_bins {
                let log_abs = spectrum[i * self.frame_size + j];
                let angle = spectrum[i * self.frame_size + self.n_fft_bins + j];
                let mag = log_abs.exp();
                self.fft_buf[j] =
                    rustfft::num_complex::Complex32::new(mag * angle.cos(), mag * angle.sin());
            }
            for j in 1..self.n_fft_bins - 1 {
                self.fft_buf[self.n_fft - j] =
                    rustfft::num_complex::Complex32::new(self.fft_buf[j].re, -self.fft_buf[j].im);
            }

            ifft_frame(
                &mut self.fft_buf,
                &mut self.time_domain,
                self.n_fft,
                self.ifft.as_ref(),
            );

            for j in 0..self.n_fft {
                self.overlap_buf[j] += self.time_domain[j] * self.hann[j];
                self.window_sum[j] += self.hann[j] * self.hann[j];
            }

            for k in 0..self.hop_length {
                let sample = if self.window_sum[k] > 1e-8 {
                    self.overlap_buf[k] / self.window_sum[k]
                } else {
                    self.overlap_buf[k]
                };
                if self.padding_remaining > 0 {
                    self.padding_remaining -= 1;
                } else {
                    output.push(sample);
                }
            }

            self.overlap_buf.copy_within(self.hop_length..self.n_fft, 0);
            self.overlap_buf[self.n_fft - self.hop_length..].fill(0.0);
            self.window_sum.copy_within(self.hop_length..self.n_fft, 0);
            self.window_sum[self.n_fft - self.hop_length..].fill(0.0);
        }
        output
    }

    /// Flush any remaining samples buffered in the overlap-add accumulator.
    pub fn flush(&mut self) -> Vec<f32> {
        self.overlap_buf.fill(0.0);
        self.window_sum.fill(0.0);
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GGUF holding just a tensor named `name` (one f32).
    fn gguf_with_tensor(name: &str) -> Arc<GgufFile> {
        let mut w = crate::convert::writer::GgufWriter::new();
        w.add_tensor(name, vec![1], 0, 4);
        let mut bytes = Vec::new();
        w.write_header_and_tensor_info(&mut bytes).unwrap();
        w.write_tensor_data(&mut bytes, &1.0f32.to_le_bytes())
            .unwrap();
        Arc::new(GgufFile::from_bytes(Arc::from(bytes)).unwrap())
    }

    #[test]
    fn lfm3a_gen_detection_has_one_definition_per_half() {
        let detok_only = gguf_with_tensor("a.gen.wav.quant_first_out.weight");
        assert!(has_lfm3a_gen_detokenizer(&detok_only));
        assert!(!has_lfm3a_gen_decoder(&detok_only));
        assert!(
            is_lfm3a_gen_gguf(&detok_only),
            "the wav quantizer key was missing from the union"
        );
        let decoder_only = gguf_with_tensor("a.gen.depth_linear.weight");
        assert!(has_lfm3a_gen_decoder(&decoder_only));
        assert!(is_lfm3a_gen_gguf(&decoder_only));
        // a detokenizer-only file asked for a decoder says so, instead of naming a legacy key
        let err = AudioDecoderWeights::from_gguf(&detok_only)
            .err()
            .expect("loaded");
        assert!(err.to_string().contains("no audio decoder"), "{err}");
    }

    #[test]
    fn accelerated_depthformer_needs_support_and_trust_or_opt_in() {
        // (supports, default_on, requested) -> enabled
        for (s, d, r, want) in [
            (false, true, true, false),
            (true, false, false, false),
            (true, false, true, true),
            (true, true, false, true),
            (true, true, true, true),
        ] {
            assert_eq!(
                accelerated_depthformer_enabled(s, d, r),
                want,
                "{s} {d} {r}"
            );
        }
    }

    #[test]
    fn depthformer_is_opt_in_by_default() {
        let mock = MockAudioAccelerator {
            active: std::sync::atomic::AtomicBool::new(false),
        };
        assert!(!mock.depthformer_default_on());
    }

    #[test]
    fn test_istft_streamer_exact_parity_with_batch() {
        let n_fft = 1280;
        let hop_length = 320;
        let n_fft_bins = n_fft / 2 + 1;
        let frame_size = n_fft_bins * 2;
        let n_frames = 18; // 3 tokens * 6 frames

        let mut spectrum = vec![0.0f32; n_frames * frame_size];
        for i in 0..spectrum.len() {
            spectrum[i] = ((i as f32) * 0.01).sin();
        }

        let batch_pcm = istft_to_pcm(&spectrum, n_fft, hop_length);

        let mut streamer = IstftStreamer::new(n_fft, hop_length);
        let mut streamed_pcm = Vec::new();
        // Feed in 6-frame chunks (matching 1 token at a time)
        for chunk in spectrum.chunks(6 * frame_size) {
            let pcm_chunk = streamer.feed_frames(chunk);
            streamed_pcm.extend_from_slice(&pcm_chunk);
        }

        assert_eq!(batch_pcm.len(), streamed_pcm.len());
        for (idx, (&b, &s)) in batch_pcm.iter().zip(streamed_pcm.iter()).enumerate() {
            assert!(
                (b - s).abs() < 1e-6,
                "Sample mismatch at {idx}: batch={b}, stream={s}"
            );
        }
    }

    #[test]
    fn test_build_audio_accelerator_empty_gguf_graceful_none() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        let bytes: Arc<[u8]> = Arc::from(data.into_boxed_slice());
        let gguf = Arc::new(GgufFile::from_bytes(bytes).expect("parse minimal gguf"));

        assert!(build_audio_accelerator(&gguf, crate::engine::BackendPreference::Cpu).is_none());
        assert!(build_audio_accelerator(&gguf, crate::engine::BackendPreference::Metal).is_none());
        assert!(build_audio_accelerator(&gguf, crate::engine::BackendPreference::Gpu).is_none());
        assert!(build_audio_accelerator(&gguf, crate::engine::BackendPreference::Auto).is_none());
        assert!(
            build_audio_accelerator(&gguf, crate::engine::BackendPreference::Hexagon).is_none()
        );
        assert!(build_audio_accelerator(&gguf, crate::engine::BackendPreference::Npu).is_none());
    }

    struct MockAudioAccelerator {
        active: std::sync::atomic::AtomicBool,
    }

    impl AudioAccelerator for MockAudioAccelerator {
        fn supports_depthformer(&self) -> bool {
            false
        }
        fn sample_audio_frame(
            &self,
            _embedding: &[f32],
            _temperature: f32,
            _top_k: usize,
        ) -> [i32; 8] {
            [0; 8]
        }
        fn detokenize_to_spectrum(
            &self,
            _cpu_weights: &DetokenizerWeights,
            _codes: &[i32],
        ) -> Vec<f32> {
            vec![]
        }
        fn reset_depthformer(&self) {}
        fn reset_detokenizer(&self) {}
        fn try_acquire_session(&self) -> bool {
            self.active
                .compare_exchange(
                    false,
                    true,
                    std::sync::atomic::Ordering::Acquire,
                    std::sync::atomic::Ordering::Relaxed,
                )
                .is_ok()
        }
        fn release_session(&self) {
            self.active
                .store(false, std::sync::atomic::Ordering::Release);
        }
    }

    #[test]
    fn test_audio_accelerator_session_lease_lifecycle() {
        let acc = MockAudioAccelerator {
            active: std::sync::atomic::AtomicBool::new(false),
        };
        assert!(acc.try_acquire_session());
        // Second concurrent acquisition fails while active.
        assert!(!acc.try_acquire_session());
        acc.release_session();
        // Subsequent acquisition succeeds.
        assert!(acc.try_acquire_session());
        acc.release_session();
    }

    #[test]
    fn test_upsample_single_token() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let n_embd = 4;
        let n_up = 6;
        let output = upsample(&input, n_embd, n_up);
        assert_eq!(output.len(), 24);
        for chunk in output.chunks_exact(n_embd) {
            assert_eq!(chunk, input.as_slice());
        }
    }

    #[test]
    fn test_upsample_multiple_tokens() {
        let input = vec![0.0, 10.0, 10.0, 20.0];
        let n_embd = 2;
        let n_up = 2;
        let output = upsample(&input, n_embd, n_up);
        assert_eq!(output.len(), 8);
        assert_eq!(&output[0..2], &[0.0, 10.0]);
        assert_eq!(&output[2..4], &[5.0, 15.0]);
        assert_eq!(&output[4..6], &[10.0, 20.0]);
        assert_eq!(&output[6..8], &[10.0, 20.0]);
    }

    #[test]
    fn test_lfm3a_gen_maskgit_3step_and_ctrl_repetition_penalty() {
        let n_codebook = 12;
        let n_vocab = 257;
        let n_embd = 8;
        let llm_dim = 4;
        let ffn_dim = 16;

        let depthformer_config = DepthformerConfig {
            n_layer: 2,
            n_embd,
            n_head: 2,
            n_head_kv: 1,
            n_embd_head: 4,
            ffn_dim,
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            max_seq_len: n_codebook,
        };
        let decoder_config = DecoderConfig {
            n_codebook,
            n_vocab,
            n_embd: llm_dim,
            rms_norm_eps: 1e-5,
        };

        // Layer 0: ShortConv (k_norm empty). Layer 1: Attention (k_norm non-empty).
        let shortconv_layer = DepthformerLayerWeights {
            operator_norm: vec![1.0; n_embd],
            wqkv: MmapWeight::from_owned_f32(vec![0.05; 3 * n_embd * n_embd], 3 * n_embd, n_embd),
            q_norm: vec![0.1; 3 * n_embd],
            k_norm: Vec::new(),
            wo: MmapWeight::from_owned_f32(vec![0.05; n_embd * n_embd], n_embd, n_embd),
            ffn_norm: vec![1.0; n_embd],
            w1: MmapWeight::from_owned_f32(vec![0.05; ffn_dim * n_embd], ffn_dim, n_embd),
            w2: MmapWeight::from_owned_f32(vec![0.05; n_embd * ffn_dim], n_embd, ffn_dim),
            w3: MmapWeight::from_owned_f32(vec![0.05; ffn_dim * n_embd], ffn_dim, n_embd),
        };
        let qkv_rows = (2 + 2) * 4;
        let attn_layer = DepthformerLayerWeights {
            operator_norm: vec![1.0; n_embd],
            wqkv: MmapWeight::from_owned_f32(vec![0.05; qkv_rows * n_embd], qkv_rows, n_embd),
            q_norm: vec![1.0; 4],
            k_norm: vec![1.0; 4],
            wo: MmapWeight::from_owned_f32(vec![0.05; n_embd * n_embd], n_embd, n_embd),
            ffn_norm: vec![1.0; n_embd],
            w1: MmapWeight::from_owned_f32(vec![0.05; ffn_dim * n_embd], ffn_dim, n_embd),
            w2: MmapWeight::from_owned_f32(vec![0.05; n_embd * ffn_dim], n_embd, ffn_dim),
            w3: MmapWeight::from_owned_f32(vec![0.05; ffn_dim * n_embd], ffn_dim, n_embd),
        };

        let mut depth_embeddings = Vec::with_capacity(n_codebook);
        for j in 0..n_codebook {
            let mut logits_w = vec![0.0f32; n_vocab * n_embd];
            // Make code `j + 5` the clear winner initially, and `j + 20` a close runner-up so
            // CTRL repetition penalty (`1.2`) flips the winner to `j + 20` on the second frame.
            let win_code = j + 5;
            let alt_code = j + 20;
            for d in 0..n_embd {
                logits_w[win_code * n_embd + d] = 1.10;
                logits_w[alt_code * n_embd + d] = 1.00;
            }
            depth_embeddings.push(CodebookWeights {
                embedding: MmapWeight::from_owned_f32(vec![0.1; n_vocab * n_embd], n_vocab, n_embd),
                norm: vec![1.0; n_embd],
                to_logits: MmapWeight::from_owned_f32(logits_w, n_vocab, n_embd),
            });
        }

        let weights = AudioDecoderWeights {
            lfm3a_gen: true,
            depthformer_config: depthformer_config.clone(),
            decoder_config,
            depthformer_layers: vec![shortconv_layer, attn_layer],
            depth_linear_w: MmapWeight::from_owned_f32(
                vec![0.2; n_embd * llm_dim],
                n_embd,
                llm_dim,
            ),
            depth_linear_b: vec![0.0; n_embd],
            depth_embeddings,
            audio_embedding: CodebookWeights {
                embedding: MmapWeight::from_owned_f32(
                    vec![0.05; n_codebook * n_vocab * llm_dim],
                    n_codebook * n_vocab,
                    llm_dim,
                ),
                norm: Vec::new(),
                to_logits: MmapWeight::from_owned_f32(Vec::new(), 0, 0),
            },
            interleave: crate::audio_engine::InterleaveCadence { text: 6, audio: 12 },
        };
        assert!(weights.is_lfm3a_gen());

        let mut state = DepthformerState::new(&depthformer_config);
        let llm_hidden = vec![1.0f32; llm_dim];

        // Frame 1 (greedy): all 12 codebooks unmask to `j + 5` (none equal to mask_id = 256).
        let codes1 = sample_frame_with_temps(&weights, &mut state, &llm_hidden, [0.0, 0.0, 0.0], 1);
        assert_eq!(codes1.len(), 12);
        for (j, &c) in codes1.iter().enumerate() {
            assert_eq!(c, (j + 5) as i32);
        }

        // Frame 2 (greedy): CTRL repetition penalty divides `win_code` logit by 1.2, flipping
        // every codebook to `alt_code = j + 20`.
        let codes2 = sample_frame_with_temps(&weights, &mut state, &llm_hidden, [0.0, 0.0, 0.0], 1);
        assert_eq!(codes2.len(), 12);
        for (j, &c) in codes2.iter().enumerate() {
            assert_eq!(c, (j + 20) as i32);
        }

        // Stochastic Gumbel-Max + top-k sampling with deterministic RNG (`[0.8, 0.4, 0.0]`, `top_k = 2`):
        // every sampled codebook must land in the top-2 candidates (`j + 5` or `j + 20`), and
        // re-running from the same RNG state must reproduce the exact same 12 codes.
        state.reset_history();
        let mut rng_a = 0x1234_5678_9abc_def0u64;
        let codes_stoch_a = sample_frame_lfm3a(
            &weights,
            &mut state,
            &llm_hidden,
            [0.8, 0.4, 0.0],
            2,
            || xorshift64_next_f32_01(&mut rng_a),
        );
        state.reset_history();
        let mut rng_b = 0x1234_5678_9abc_def0u64;
        let codes_stoch_b = sample_frame_lfm3a(
            &weights,
            &mut state,
            &llm_hidden,
            [0.8, 0.4, 0.0],
            2,
            || xorshift64_next_f32_01(&mut rng_b),
        );
        assert_eq!(codes_stoch_a, codes_stoch_b);
        for (j, &c) in codes_stoch_a.iter().enumerate() {
            let win = (j + 5) as i32;
            let alt = (j + 20) as i32;
            assert!(
                c == win || c == alt,
                "codebook {j} sampled {c}, expected top-2 candidate ({win} or {alt})"
            );
        }

        // Feedback embedding sums all 12 codebooks into `[llm_dim]`.
        let fb = embed_audio_codes(&weights, &codes1);
        assert_eq!(fb.len(), llm_dim);
        assert!(fb.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[test]
    fn test_lfm3a_gen_fsq_and_streaming_detokenizer_produces_1920_samples() {
        let fsq = build_default_fsq_codebook();
        assert_eq!(fsq.len(), 256 * 4);
        assert!((fsq[0] - (-1.0)).abs() < 1e-6);
        assert!((fsq[255 * 4 + 3] - 1.0).abs() < 1e-6);

        let n_embd = 8;
        let fsq_proj_dim = 512;
        let n_fft = 1280;
        let hop_length = 320;
        let single_spec_bins = (n_fft / 2 + 1) * 2;
        let out_rows = 6 * single_spec_bins;

        let lfm3_fsq = Lfm3FsqWeights {
            fsq_codebook: fsq,
            fsq_proj_dim,
            quant_first_out_w: vec![0.01f32; fsq_proj_dim * 16],
            quant_first_out_b: vec![0.0f32; fsq_proj_dim],
            quant_rest_out_w: vec![0.01f32; fsq_proj_dim * 16],
            quant_rest_out_b: vec![0.0f32; fsq_proj_dim],
            tfm_in_proj_w: MmapWeight::from_owned_f32(
                vec![0.01f32; n_embd * fsq_proj_dim],
                n_embd,
                fsq_proj_dim,
            ),
            tfm_in_proj_b: vec![0.0f32; n_embd],
        };

        let cfg = DetokenizerConfig {
            n_layer: 2,
            n_embd,
            n_head: 2,
            n_head_kv: 1,
            n_embd_head: 4,
            ffn_dim: 16,
            d_conv: 2,
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            swa_window_size: 30,
            n_codes: 12,
            n_fft,
            hop_length,
            sample_rate: 24000,
            layer_is_conv: vec![true, false],
        };

        let layer0 = DetokLayerWeights {
            operator_norm: vec![1.0; n_embd],
            ffn_norm: vec![1.0; n_embd],
            attn_post_norm: Some(vec![1.0; n_embd]),
            ffn_post_norm: Some(vec![1.0; n_embd]),
            ffn_w1: MmapWeight::from_owned_f32(vec![0.05; 16 * n_embd], 16, n_embd),
            ffn_w2: MmapWeight::from_owned_f32(vec![0.05; n_embd * 16], n_embd, 16),
            ffn_w3: MmapWeight::from_owned_f32(vec![0.05; 16 * n_embd], 16, n_embd),
            conv_in_proj: Some(MmapWeight::from_owned_f32(
                vec![0.05; 3 * n_embd * n_embd],
                3 * n_embd,
                n_embd,
            )),
            conv_out_proj: Some(MmapWeight::from_owned_f32(
                vec![0.05; n_embd * n_embd],
                n_embd,
                n_embd,
            )),
            conv_weight: Some(vec![0.1; 3 * n_embd]),
            wq: None,
            wk: None,
            wv: None,
            wo: None,
            q_norm: None,
            k_norm: None,
            attn_gate: None,
        };

        let layer1 = DetokLayerWeights {
            operator_norm: vec![1.0; n_embd],
            ffn_norm: vec![1.0; n_embd],
            attn_post_norm: Some(vec![1.0; n_embd]),
            ffn_post_norm: Some(vec![1.0; n_embd]),
            ffn_w1: MmapWeight::from_owned_f32(vec![0.05; 16 * n_embd], 16, n_embd),
            ffn_w2: MmapWeight::from_owned_f32(vec![0.05; n_embd * 16], n_embd, 16),
            ffn_w3: MmapWeight::from_owned_f32(vec![0.05; 16 * n_embd], 16, n_embd),
            conv_in_proj: None,
            conv_out_proj: None,
            conv_weight: None,
            wq: Some(MmapWeight::from_owned_f32(
                vec![0.05; 8 * n_embd],
                8,
                n_embd,
            )),
            wk: Some(MmapWeight::from_owned_f32(
                vec![0.05; 4 * n_embd],
                4,
                n_embd,
            )),
            wv: Some(MmapWeight::from_owned_f32(
                vec![0.05; 4 * n_embd],
                4,
                n_embd,
            )),
            wo: Some(MmapWeight::from_owned_f32(
                vec![0.05; n_embd * 8],
                n_embd,
                8,
            )),
            q_norm: Some(vec![1.0; 4]),
            k_norm: Some(vec![1.0; 4]),
            attn_gate: Some(MmapWeight::from_owned_f32(
                vec![0.05; 2 * n_embd],
                2,
                n_embd,
            )),
        };

        let weights = DetokenizerWeights {
            config: cfg.clone(),
            output_norm: vec![1.0; n_embd],
            emb_weight: MmapWeight::from_owned_f32(Vec::new(), 0, n_embd),
            lin_w: MmapWeight::from_owned_f32(vec![0.01; out_rows * n_embd], out_rows, n_embd),
            lin_b: vec![0.0; out_rows],
            layers: vec![layer0, layer1],
            lfm3_fsq: Some(lfm3_fsq),
        };
        assert!(weights.is_lfm3a_gen());

        let mut state = DetokenizerState::new(&cfg);
        let codes = vec![10i32; 12];
        let pcm = detokenize_streaming(&weights, &mut state, &codes);
        assert_eq!(pcm.len(), 6 * hop_length);
        assert_eq!(pcm.len(), 1920);
        assert!(pcm.iter().all(|v| v.is_finite()));

        // The streaming API keeps the ISTFT start-up samples that the whole-sequence path
        // crops: past that crop the two agree, so one is the other plus a known head.
        let pad = cfg.n_fft.saturating_sub(cfg.hop_length) / 2;
        let reference = detokenize(&weights, std::slice::from_ref(&codes));
        assert_eq!(reference.len() + pad, pcm.len(), "reference length");
        for (i, (got, want)) in pcm[pad..].iter().zip(&reference).enumerate() {
            assert!((got - want).abs() < 1e-4, "sample {i}: {got} vs {want}");
        }
    }

    #[test]
    fn unseeded_depthformer_states_draw_distinct_streams_and_seeded_states_match() {
        let cfg = DepthformerConfig {
            n_layer: 1,
            n_embd: 32,
            n_head: 1,
            n_head_kv: 1,
            n_embd_head: 32,
            ffn_dim: 64,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10_000.0,
            max_seq_len: 8,
        };
        let mut s1 = DepthformerState::new(&cfg);
        let mut s2 = DepthformerState::new(&cfg);
        let draw1: Vec<u32> = (0..8).map(|_| s1.next_f32_01().to_bits()).collect();
        let draw2: Vec<u32> = (0..8).map(|_| s2.next_f32_01().to_bits()).collect();
        assert_ne!(
            draw1, draw2,
            "fresh unseeded DepthformerState instances must not share a fixed constant seed"
        );

        s1.seed_rng(12345);
        s2.seed_rng(12345);
        let atomic_rng = std::sync::atomic::AtomicU64::new(seed_to_rng_state(12345));
        let seeded1: Vec<u32> = (0..8).map(|_| s1.next_f32_01().to_bits()).collect();
        let seeded2: Vec<u32> = (0..8).map(|_| s2.next_f32_01().to_bits()).collect();
        let seeded_atomic: Vec<u32> = (0..8)
            .map(|_| next_atomic_xorshift64_f32_01(&atomic_rng).to_bits())
            .collect();
        assert_eq!(seeded1, seeded2);
        assert_eq!(seeded1, seeded_atomic);
    }

    #[test]
    fn a_small_seed_does_not_start_the_stream_at_zero() {
        // raw xorshift64 draws exactly 0.0 first for a seed below 1024 and under 1e-3 for any
        // seed below ~2^23
        for seed in [1u64, 2, 42, 12345, 0xFFFF, 1 << 20] {
            let mut raw = seed;
            assert!(
                next_xorshift64_f32_01(&mut raw) < 1e-3,
                "seed {seed}: the unmixed first draw is the near-zero one this test guards against"
            );
            let mut state = seed_to_rng_state(seed);
            let first = next_xorshift64_f32_01(&mut state);
            assert!(first > 1e-3, "seed {seed}: first draw {first}");
        }
        // nearby seeds give different streams, and seed 0 is still a valid one
        assert_ne!(seed_to_rng_state(1), seed_to_rng_state(2));
        assert_ne!(seed_to_rng_state(0), 0);
    }

    #[test]
    fn public_hf_lfm3a_gen_mmproj_schema_loads_cleanly() {
        // Mirrors the exact metadata keys and tensor names of LiquidAI/LFM3-Audio-2B-GGUF
        // and LiquidAI/LFM3-Omni-2B-GGUF (`clip.gen.audio.*`, 3D `a.gen.code.embd.weight`
        // with shape [n_embd, n_vocab, n_codebook], `a.gen.code.proj_in.*`,
        // `a.gen.wav.quant.first.*`, and `a.gen.wav.tfm.*`).
        let n_embd = 8usize;
        let ffn_dim = 16usize;
        let llm_dim = 12usize;
        let n_cb = 12usize;
        let n_vocab = 2017usize;
        let detok_embd = 8usize;
        let detok_ffn = 16usize;
        let fsq_proj_dim = 4usize;
        let n_fft = 16usize;
        let out_bins = (n_fft / 2 + 1) * 2; // 18

        let bytes = crate::gguf::GgufBuilder::new()
            .kv_str("general.architecture", "clip")
            .kv_bool("clip.has_audio_encoder", true)
            .kv_bool("clip.has_gen_audio_encoder", true)
            .kv_str("clip.gen.audio.projector_type", "lfm3a_gen")
            .kv_u32("clip.gen.audio.block_count", 3)
            .kv_u32("clip.gen.audio.embedding_length", n_embd as u32)
            .kv_u32("clip.gen.audio.feed_forward_length", ffn_dim as u32)
            .kv_u32("clip.gen.audio.attention.head_count", 2)
            .kv_u32("clip.gen.audio.attention.head_count_kv", 1)
            .kv_f32("clip.gen.audio.attention.layer_norm_epsilon", 1e-5)
            .kv_u32("clip.gen.audio.projection_dim", llm_dim as u32)
            .kv_u32("clip.audio_gen.wav.n_fft", n_fft as u32)
            .kv_u32("clip.audio_gen.wav.hop_length", 4)
            .tensor_f32(
                "a.gen.code.proj_in.weight",
                &[llm_dim, n_cb * n_embd],
                &vec![0.01; n_cb * n_embd * llm_dim],
            )
            .tensor_f32(
                "a.gen.code.proj_in.bias",
                &[n_cb * n_embd],
                &vec![0.0; n_cb * n_embd],
            )
            .tensor_f32(
                "a.gen.code.embd.weight",
                &[n_embd, n_vocab, n_cb],
                &vec![0.01; n_cb * n_vocab * n_embd],
            )
            .tensor_f32(
                "a.gen.code.head_norm.weight",
                &[n_embd, n_cb],
                &vec![1.0; n_cb * n_embd],
            )
            .tensor_f32("a.gen.code.mask_embd", &[n_embd], &vec![0.2; n_embd])
            .tensor_f32(
                "a.gen.code.output_norm.weight",
                &[n_embd],
                &vec![1.0; n_embd],
            )
            .tensor_f32(
                "a.gen.code.out_embd.weight",
                &[llm_dim, n_cb * n_vocab],
                &vec![0.01; n_cb * n_vocab * llm_dim],
            )
            // 3 depthformer attention layers
            .tensor_f32("a.gen.code.blk.0.ln1.weight", &[n_embd], &vec![1.0; n_embd])
            .tensor_f32("a.gen.code.blk.0.ln2.weight", &[n_embd], &vec![1.0; n_embd])
            .tensor_f32(
                "a.gen.code.blk.0.attn_q.weight",
                &[n_embd, n_embd],
                &vec![0.01; n_embd * n_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.0.attn_k.weight",
                &[n_embd, 4],
                &vec![0.01; 4 * n_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.0.attn_v.weight",
                &[n_embd, 4],
                &vec![0.01; 4 * n_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.0.attn_out.weight",
                &[n_embd, n_embd],
                &vec![0.01; n_embd * n_embd],
            )
            .tensor_f32("a.gen.code.blk.0.attn_q_norm.weight", &[4], &[1.0; 4])
            .tensor_f32("a.gen.code.blk.0.attn_k_norm.weight", &[4], &[1.0; 4])
            .tensor_f32(
                "a.gen.code.blk.0.attn_gate.weight",
                &[n_embd, 2],
                &vec![0.01; 2 * n_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.0.ffn_gate.weight",
                &[n_embd, ffn_dim],
                &vec![0.01; ffn_dim * n_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.0.ffn_down.weight",
                &[ffn_dim, n_embd],
                &vec![0.01; n_embd * ffn_dim],
            )
            .tensor_f32(
                "a.gen.code.blk.0.ffn_up.weight",
                &[n_embd, ffn_dim],
                &vec![0.01; ffn_dim * n_embd],
            )
            .tensor_f32("a.gen.code.blk.1.ln1.weight", &[n_embd], &vec![1.0; n_embd])
            .tensor_f32("a.gen.code.blk.1.ln2.weight", &[n_embd], &vec![1.0; n_embd])
            .tensor_f32(
                "a.gen.code.blk.1.attn_q.weight",
                &[n_embd, n_embd],
                &vec![0.01; n_embd * n_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.1.attn_k.weight",
                &[n_embd, 4],
                &vec![0.01; 4 * n_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.1.attn_v.weight",
                &[n_embd, 4],
                &vec![0.01; 4 * n_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.1.attn_out.weight",
                &[n_embd, n_embd],
                &vec![0.01; n_embd * n_embd],
            )
            .tensor_f32("a.gen.code.blk.1.attn_q_norm.weight", &[4], &[1.0; 4])
            .tensor_f32("a.gen.code.blk.1.attn_k_norm.weight", &[4], &[1.0; 4])
            .tensor_f32(
                "a.gen.code.blk.1.attn_gate.weight",
                &[n_embd, 2],
                &vec![0.01; 2 * n_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.1.ffn_gate.weight",
                &[n_embd, ffn_dim],
                &vec![0.01; ffn_dim * n_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.1.ffn_down.weight",
                &[ffn_dim, n_embd],
                &vec![0.01; n_embd * ffn_dim],
            )
            .tensor_f32(
                "a.gen.code.blk.1.ffn_up.weight",
                &[n_embd, ffn_dim],
                &vec![0.01; ffn_dim * n_embd],
            )
            .tensor_f32("a.gen.code.blk.2.ln1.weight", &[n_embd], &vec![1.0; n_embd])
            .tensor_f32("a.gen.code.blk.2.ln2.weight", &[n_embd], &vec![1.0; n_embd])
            .tensor_f32(
                "a.gen.code.blk.2.attn_q.weight",
                &[n_embd, n_embd],
                &vec![0.01; n_embd * n_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.2.attn_k.weight",
                &[n_embd, 4],
                &vec![0.01; 4 * n_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.2.attn_v.weight",
                &[n_embd, 4],
                &vec![0.01; 4 * n_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.2.attn_out.weight",
                &[n_embd, n_embd],
                &vec![0.01; n_embd * n_embd],
            )
            .tensor_f32("a.gen.code.blk.2.attn_q_norm.weight", &[4], &[1.0; 4])
            .tensor_f32("a.gen.code.blk.2.attn_k_norm.weight", &[4], &[1.0; 4])
            .tensor_f32(
                "a.gen.code.blk.2.attn_gate.weight",
                &[n_embd, 2],
                &vec![0.01; 2 * n_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.2.ffn_gate.weight",
                &[n_embd, ffn_dim],
                &vec![0.01; ffn_dim * n_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.2.ffn_down.weight",
                &[ffn_dim, n_embd],
                &vec![0.01; n_embd * ffn_dim],
            )
            .tensor_f32(
                "a.gen.code.blk.2.ffn_up.weight",
                &[n_embd, ffn_dim],
                &vec![0.01; ffn_dim * n_embd],
            )
            // Detokenizer (`a.gen.wav.quant.first.*` + `a.gen.wav.tfm.*`, 2 blocks)
            .tensor_f32(
                "a.gen.wav.quant.first.codebook.weight",
                &[4, 2016],
                &vec![0.1; 2016 * 4],
            )
            .tensor_f32(
                "a.gen.wav.quant.first.out_proj.weight",
                &[48, fsq_proj_dim],
                &vec![0.01; fsq_proj_dim * 48],
            )
            .tensor_f32(
                "a.gen.wav.quant.first.out_proj.bias",
                &[fsq_proj_dim],
                &vec![0.0; fsq_proj_dim],
            )
            .tensor_f32(
                "a.gen.wav.tfm.in_proj.weight",
                &[fsq_proj_dim, detok_embd],
                &vec![0.01; detok_embd * fsq_proj_dim],
            )
            .tensor_f32(
                "a.gen.wav.tfm.in_proj.bias",
                &[detok_embd],
                &vec![0.0; detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.0.ln1.weight",
                &[detok_embd],
                &vec![1.0; detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.0.attn_post_norm.weight",
                &[detok_embd],
                &vec![1.0; detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.0.ln2.weight",
                &[detok_embd],
                &vec![1.0; detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.0.ffn_post_norm.weight",
                &[detok_embd],
                &vec![1.0; detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.0.conv_dw.weight",
                &[7, detok_embd],
                &vec![0.05; 7 * detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.0.conv_pw1.weight",
                &[detok_embd, 3 * detok_embd],
                &vec![0.01; 3 * detok_embd * detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.0.conv_pw2.weight",
                &[detok_embd, detok_embd],
                &vec![0.01; detok_embd * detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.0.ffn_gate.weight",
                &[detok_embd, detok_ffn],
                &vec![0.01; detok_ffn * detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.0.ffn_down.weight",
                &[detok_ffn, detok_embd],
                &vec![0.01; detok_embd * detok_ffn],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.0.ffn_up.weight",
                &[detok_embd, detok_ffn],
                &vec![0.01; detok_ffn * detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.1.ln1.weight",
                &[detok_embd],
                &vec![1.0; detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.1.attn_post_norm.weight",
                &[detok_embd],
                &vec![1.0; detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.1.ln2.weight",
                &[detok_embd],
                &vec![1.0; detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.1.ffn_post_norm.weight",
                &[detok_embd],
                &vec![1.0; detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.1.attn_q.weight",
                &[detok_embd, detok_embd],
                &vec![0.01; detok_embd * detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.1.attn_k.weight",
                &[detok_embd, 4],
                &vec![0.01; 4 * detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.1.attn_v.weight",
                &[detok_embd, 4],
                &vec![0.01; 4 * detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.1.attn_out.weight",
                &[detok_embd, detok_embd],
                &vec![0.01; detok_embd * detok_embd],
            )
            .tensor_f32("a.gen.wav.tfm.blk.1.attn_q_norm.weight", &[4], &[1.0; 4])
            .tensor_f32("a.gen.wav.tfm.blk.1.attn_k_norm.weight", &[4], &[1.0; 4])
            .tensor_f32(
                "a.gen.wav.tfm.blk.1.attn_gate.weight",
                &[detok_embd, 2],
                &vec![0.01; 2 * detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.1.ffn_gate.weight",
                &[detok_embd, detok_ffn],
                &vec![0.01; detok_ffn * detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.1.ffn_down.weight",
                &[detok_ffn, detok_embd],
                &vec![0.01; detok_embd * detok_ffn],
            )
            .tensor_f32(
                "a.gen.wav.tfm.blk.1.ffn_up.weight",
                &[detok_embd, detok_ffn],
                &vec![0.01; detok_ffn * detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.output_norm.weight",
                &[detok_embd],
                &vec![1.0; detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.out_proj.weight",
                &[detok_embd, out_bins],
                &vec![0.01; out_bins * detok_embd],
            )
            .tensor_f32(
                "a.gen.wav.tfm.out_proj.bias",
                &[out_bins],
                &vec![0.0; out_bins],
            )
            .build_bytes();

        let gguf = Arc::new(GgufFile::from_bytes(Arc::from(bytes)).unwrap());
        assert!(has_lfm3a_gen_decoder(&gguf));
        assert!(has_lfm3a_gen_detokenizer(&gguf));

        let dec = AudioDecoderWeights::from_gguf(&gguf).expect("load public lfm3a_gen decoder");
        assert!(dec.is_lfm3a_gen());
        assert_eq!(dec.depthformer_config.n_layer, 3);
        assert_eq!(dec.depthformer_config.n_embd, n_embd);
        assert_eq!(dec.decoder_config.n_codebook, 12);
        assert_eq!(dec.decoder_config.n_vocab, 2017);

        let det = DetokenizerWeights::from_gguf(&gguf).expect("load public lfm3a_gen detokenizer");
        assert!(det.is_lfm3a_gen());
        assert_eq!(det.config.n_layer, 2);
        assert_eq!(det.config.n_embd, detok_embd);
        assert_eq!(det.config.d_conv, 6);
        assert_eq!(det.lfm3_fsq.as_ref().unwrap().fsq_codebook.len(), 2016 * 4);

        // Also verify `from_gguf_lfm3a_gen` when `head_count` is omitted and `head_count_kv`
        // is stored as a per-layer I32 array with `n_embd = 4` (not divisible by the default 8):
        // attention dimensions `(n_head = 2, n_head_kv = 1, n_embd_head = 2)` are derived from
        // the attention tensors in the block loop.
        let small_embd = 4usize;
        let small_ffn = 8usize;
        let small_bytes = crate::gguf::GgufBuilder::new()
            .kv_str("general.architecture", "clip")
            .kv_bool("clip.has_audio_encoder", true)
            .kv_bool("clip.has_gen_audio_encoder", true)
            .kv_str("clip.gen.audio.projector_type", "lfm3a_gen")
            .kv_u32("clip.gen.audio.block_count", 1)
            .kv_u32("clip.gen.audio.embedding_length", small_embd as u32)
            .kv_u32("clip.gen.audio.feed_forward_length", small_ffn as u32)
            .kv_i32_array("clip.audio_gen.code.attention.head_count_kv", vec![0, 1])
            .kv_f32("clip.gen.audio.attention.layer_norm_epsilon", 1e-5)
            .kv_u32("clip.gen.audio.projection_dim", llm_dim as u32)
            .tensor_f32(
                "a.gen.code.proj_in.weight",
                &[llm_dim, n_cb * small_embd],
                &vec![0.01; n_cb * small_embd * llm_dim],
            )
            .tensor_f32(
                "a.gen.code.proj_in.bias",
                &[n_cb * small_embd],
                &vec![0.0; n_cb * small_embd],
            )
            .tensor_f32(
                "a.gen.code.embd.weight",
                &[small_embd, n_vocab, n_cb],
                &vec![0.01; n_cb * n_vocab * small_embd],
            )
            .tensor_f32(
                "a.gen.code.head_norm.weight",
                &[small_embd, n_cb],
                &vec![1.0; n_cb * small_embd],
            )
            .tensor_f32(
                "a.gen.code.mask_embd",
                &[small_embd],
                &vec![0.2; small_embd],
            )
            .tensor_f32(
                "a.gen.code.output_norm.weight",
                &[small_embd],
                &vec![1.0; small_embd],
            )
            .tensor_f32(
                "a.gen.code.out_embd.weight",
                &[llm_dim, n_cb * n_vocab],
                &vec![0.01; n_cb * n_vocab * llm_dim],
            )
            .tensor_f32(
                "a.gen.code.blk.0.ln1.weight",
                &[small_embd],
                &vec![1.0; small_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.0.ln2.weight",
                &[small_embd],
                &vec![1.0; small_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.0.attn_q.weight",
                &[small_embd, small_embd],
                &vec![0.01; small_embd * small_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.0.attn_k.weight",
                &[small_embd, 2],
                &vec![0.01; 2 * small_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.0.attn_v.weight",
                &[small_embd, 2],
                &vec![0.01; 2 * small_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.0.attn_out.weight",
                &[small_embd, small_embd],
                &vec![0.01; small_embd * small_embd],
            )
            .tensor_f32("a.gen.code.blk.0.attn_q_norm.weight", &[2], &[1.0; 2])
            .tensor_f32("a.gen.code.blk.0.attn_k_norm.weight", &[2], &[1.0; 2])
            .tensor_f32(
                "a.gen.code.blk.0.attn_gate.weight",
                &[small_embd, 2],
                &vec![0.01; 2 * small_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.0.ffn_gate.weight",
                &[small_embd, small_ffn],
                &vec![0.01; small_ffn * small_embd],
            )
            .tensor_f32(
                "a.gen.code.blk.0.ffn_down.weight",
                &[small_ffn, small_embd],
                &vec![0.01; small_embd * small_ffn],
            )
            .tensor_f32(
                "a.gen.code.blk.0.ffn_up.weight",
                &[small_embd, small_ffn],
                &vec![0.01; small_ffn * small_embd],
            )
            .build_bytes();
        let small_gguf = Arc::new(GgufFile::from_bytes(Arc::from(small_bytes)).unwrap());
        let small_dec = AudioDecoderWeights::from_gguf(&small_gguf)
            .expect("load lfm3a_gen decoder with tensor-derived head count");
        assert_eq!(small_dec.depthformer_config.n_embd, small_embd);
        assert_eq!(small_dec.depthformer_config.n_head, 2);
        assert_eq!(small_dec.depthformer_config.n_head_kv, 1);
        assert_eq!(small_dec.depthformer_config.n_embd_head, 2);
    }
}
