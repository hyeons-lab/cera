//! Streaming Sortformer speaker diarization (NVIDIA `diar_streaming_sortformer_4spk-v2.1`)
//! on the CPU.
//!
//! Sortformer is an end-to-end diarizer: audio in, per-frame speaker-activity probabilities
//! out, for up to four speakers, with no clustering stage. Frames are 80 ms. The pieces:
//!
//! ```text
//! pcm 16 kHz
//!   → log-mel [T × 128]          (NeMo front end; no per-feature normalization)
//!   → conv stem + pre_encode.out [T/8 × 512]       (the cached "pre-encode" embeddings)
//!   → ×sqrt(512)                                    (NeMo's `xscaling`)
//!   → 17 × FastConformer block   (the same blocks as the LFM2-Audio encoder)
//!   → encoder_proj 512 → 192
//!   → 18 × Transformer layer     (post-LN, ReLU FFN, no positional embedding)
//!   → relu → Linear → relu → Linear(→4) → sigmoid
//! ```
//!
//! **Streaming.** Each step pre-encodes only the new chunk of features, concatenates it with
//! the cached pre-encode embeddings of a *speaker cache* and a *FIFO*, runs the whole encoder
//! and head over that concatenation, and keeps the chunk's predictions. The FIFO holds the
//! most recent frames; frames popped from it enter the speaker cache, which is compressed
//! (arrival-order speaker cache, "AOSC") to a fixed length by an importance score that favors
//! confident, non-overlapped speech and keeps every speaker represented. The cache and FIFO
//! store embeddings *before* the x-scale, which is why the cost of a step is the encoder over
//! the whole cache, not over the chunk alone.
//!
//! The update logic is a line-for-line port of NeMo's `SortformerModules.streaming_update`
//! (the synchronous path), `_compress_spkcache`, `_get_log_pred_scores`,
//! `_disable_low_scores`, `_boost_topk_scores`, `_get_topk_indices` and
//! `_get_silence_profile`, and the feature chunker of `streaming_feat_loader`. It is pinned to
//! NeMo's own outputs by `tests/sortformer_parity.rs` against the fixtures under
//! `cera/tests/fixtures/sortformer/` (see `scripts/sortformer/`).
//!
//! **Padding.** NeMo pads the mel features to a multiple of `pad_to` (16) and masks the pad in
//! attention, the conv module and the stem. Here the pad is never computed: only the valid
//! frames run through the model, which is equivalent for every valid frame because the masks
//! zero exactly what truncation removes. Frames past the end of the audio get a zero
//! prediction, like NeMo's masked output. A caller streaming live audio has no padding and
//! passes every frame as valid.
//!
//! The model loads from the GGUF written by `scripts/sortformer/convert_sortformer.py`.

#[cfg(feature = "mmap")]
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::model::audio_encoder::{
    AudioEncoderConfig, ConformerLayerWeights, ConvStemWeights, HOP_LEN, LOG_MEL_EPS, N_FFT,
    PREEMPH, SAMPLE_RATE, WINDOW_LEN, conformer_block_forward, conv_stem_forward,
    load_conformer_block, load_conv_layer, load_vec_f32, relative_pos_emb,
};
use crate::model::audio_preprocessor::{MelFrameComputer, N_FFT_BINS, log_mel_with_tables};
use crate::model::weights::MmapWeight;

/// Largest speaker count the model's output head has; fixed by the checkpoint.
pub const MAX_SPEAKERS: usize = 4;

// ── Streaming parameters ───────────────────────────────────────────────────

/// Streaming hyper-parameters. Lengths are in 80 ms encoder frames.
///
/// The GGUF carries the checkpoint's defaults (a 15 s chunk, no FIFO); the model card's
/// "low latency" configuration is the same weights with different numbers here. Latency is
/// `chunk_len + right_context` frames, and the encoder runs over
/// `left_context + chunk_len + right_context + fifo_len + spkcache_len` frames per step.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingParams {
    /// New frames emitted per step.
    pub chunk_len: usize,
    /// Extra frames of left context fed to the encoder with each chunk.
    pub left_context: usize,
    /// Extra frames of right context (lookahead) fed to the encoder with each chunk.
    pub right_context: usize,
    /// Capacity of the FIFO of recent frames.
    pub fifo_len: usize,
    /// Capacity of the speaker cache.
    pub spkcache_len: usize,
    /// Minimum number of frames moved from the FIFO into the cache per pop.
    pub update_period: usize,
    /// Silence frames reserved per speaker in the compressed cache.
    pub sil_frames_per_spk: usize,
    /// Probabilities are clipped below this before the log scores.
    pub pred_score_threshold: f32,
    /// Bonus added to the scores of frames newer than `spkcache_len`.
    pub scores_boost_latest: f32,
    /// A frame is silence when its speaker probabilities sum below this.
    pub sil_threshold: f32,
    /// Fraction of a speaker's cache share that gets the strong boost.
    pub strong_boost_rate: f32,
    /// Fraction of a speaker's cache share that gets the weak boost.
    pub weak_boost_rate: f32,
    /// A speaker with at least this fraction of positive-score frames drops its non-positive ones.
    pub min_pos_scores_rate: f32,
    /// Placeholder index NeMo uses for disabled cache slots.
    pub max_index: usize,
}

impl StreamingParams {
    /// This configuration with different chunking, keeping the score hyper-parameters.
    pub fn with_chunking(
        &self,
        chunk_len: usize,
        left_context: usize,
        right_context: usize,
        fifo_len: usize,
        spkcache_len: usize,
        update_period: usize,
    ) -> Self {
        Self {
            chunk_len,
            left_context,
            right_context,
            fifo_len,
            spkcache_len,
            update_period,
            ..self.clone()
        }
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.chunk_len > 0, "chunk_len must be > 0");
        ensure!(self.update_period > 0, "update_period must be > 0");
        ensure!(
            self.spkcache_len / MAX_SPEAKERS > self.sil_frames_per_spk,
            "spkcache_len {} leaves no room for speakers beside {} silence frames each",
            self.spkcache_len,
            self.sil_frames_per_spk
        );
        Ok(())
    }
}

// ── Config and weights ─────────────────────────────────────────────────────

/// Architecture constants read from the GGUF.
#[derive(Debug, Clone)]
pub struct SortformerConfig {
    /// FastConformer blocks.
    pub n_layer: usize,
    /// FastConformer width (also the cached embedding width).
    pub n_embd: usize,
    /// FastConformer FFN width (derived from the tensors).
    pub n_ff: usize,
    /// FastConformer attention heads.
    pub n_head: usize,
    /// LayerNorm epsilon of the FastConformer.
    pub eps: f32,
    /// Mel bins.
    pub n_mel_bins: usize,
    /// Transformer layers.
    pub tf_layers: usize,
    /// Transformer width.
    pub tf_d: usize,
    /// Transformer attention heads.
    pub tf_heads: usize,
    /// Transformer FFN width.
    pub tf_inner: usize,
    /// LayerNorm epsilon of the Transformer.
    pub tf_eps: f32,
    /// Speakers the head predicts.
    pub n_spk: usize,
    /// Mel frames per encoder frame.
    pub subsampling: usize,
    /// Multiply the encoder input by `sqrt(n_embd)`.
    pub xscaling: bool,
    /// NeMo pads mel features to a multiple of this.
    pub pad_to: usize,
}

struct TransformerLayer {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    q_w: MmapWeight,
    q_b: Vec<f32>,
    k_w: MmapWeight,
    k_b: Vec<f32>,
    v_w: MmapWeight,
    v_b: Vec<f32>,
    o_w: MmapWeight,
    o_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    up_w: MmapWeight,
    up_b: Vec<f32>,
    down_w: MmapWeight,
    down_b: Vec<f32>,
}

/// Every Sortformer tensor, loaded from a converted GGUF.
pub struct SortformerWeights {
    /// Architecture constants.
    pub config: SortformerConfig,
    /// The checkpoint's streaming defaults.
    pub streaming: StreamingParams,
    enc_cfg: AudioEncoderConfig,
    conv_stem: ConvStemWeights,
    layers: Vec<ConformerLayerWeights>,
    proj_w: MmapWeight,
    proj_b: Vec<f32>,
    tf: Vec<TransformerLayer>,
    head_hidden_w: MmapWeight,
    head_hidden_b: Vec<f32>,
    head_out_w: MmapWeight,
    head_out_b: Vec<f32>,
    /// `N_FFT`-long window with the `WINDOW_LEN` taps centered in it.
    window: Vec<f32>,
    /// `[n_mel_bins × N_FFT_BINS]`.
    mel_fb: Vec<f32>,
}

fn req_u32(g: &GgufFile, key: &str) -> Result<usize> {
    Ok(g.get_u32(key)
        .with_context(|| format!("missing GGUF key `{key}`"))? as usize)
}

fn req_f32(g: &GgufFile, key: &str) -> Result<f32> {
    g.get_f32(key)
        .with_context(|| format!("missing GGUF key `{key}`"))
}

fn expect_eq<T: PartialEq + std::fmt::Debug>(what: &str, got: T, want: T) -> Result<()> {
    ensure!(
        got == want,
        "unsupported Sortformer GGUF: {what} is {got:?}, this build implements {want:?}"
    );
    Ok(())
}

impl SortformerWeights {
    /// Load the model from a GGUF written by `scripts/sortformer/convert_sortformer.py`.
    pub fn from_gguf(g: &Arc<GgufFile>) -> Result<Self> {
        ensure!(
            g.architecture() == Some("sortformer"),
            "not a Sortformer GGUF (general.architecture = {:?})",
            g.architecture()
        );

        // Front end: this build implements exactly NeMo's parameters for the shipped
        // checkpoint, so a GGUF that says otherwise is refused instead of silently run
        // through the wrong mel.
        expect_eq(
            "mel normalize",
            g.get_str("sortformer.mel.normalize"),
            Some("NA"),
        )?;
        expect_eq("mel n_fft", req_u32(g, "sortformer.mel.n_fft")?, N_FFT)?;
        expect_eq(
            "mel win_length",
            req_u32(g, "sortformer.mel.win_length")?,
            WINDOW_LEN,
        )?;
        expect_eq(
            "mel hop_length",
            req_u32(g, "sortformer.mel.hop_length")?,
            HOP_LEN,
        )?;
        expect_eq(
            "sample_rate",
            req_u32(g, "sortformer.sample_rate")?,
            SAMPLE_RATE as usize,
        )?;
        expect_eq(
            "mel preemph",
            req_f32(g, "sortformer.mel.preemph")?,
            PREEMPH,
        )?;
        expect_eq(
            "mel mag_power",
            req_f32(g, "sortformer.mel.mag_power")?,
            2.0,
        )?;
        ensure!(
            (req_f32(g, "sortformer.mel.log_zero_guard")? - LOG_MEL_EPS).abs() < 1e-12,
            "unsupported Sortformer GGUF: mel log_zero_guard differs from {LOG_MEL_EPS}"
        );
        expect_eq(
            "tf_activation",
            g.get_str("sortformer.tf_activation"),
            Some("relu"),
        )?;

        let n_layer = req_u32(g, "clip.audio.block_count")?;
        let n_embd = req_u32(g, "clip.audio.embedding_length")?;
        let n_head = req_u32(g, "clip.audio.attention.head_count")?;
        let n_mel_bins = req_u32(g, "clip.audio.num_mel_bins")?;
        let eps = req_f32(g, "clip.audio.attention.layer_norm_epsilon")?;
        let tf_layers = req_u32(g, "sortformer.tf_layer_count")?;
        let tf_d = req_u32(g, "sortformer.tf_d_model")?;
        let tf_heads = req_u32(g, "sortformer.tf_head_count")?;
        let tf_inner = req_u32(g, "sortformer.tf_inner_size")?;
        let tf_eps = req_f32(g, "sortformer.tf_layer_norm_epsilon")?;
        let n_spk = req_u32(g, "sortformer.max_speakers")?;
        expect_eq("max_speakers", n_spk, MAX_SPEAKERS)?;
        expect_eq("fc_d_model", req_u32(g, "sortformer.fc_d_model")?, n_embd)?;
        ensure!(
            tf_d % tf_heads == 0,
            "tf_d_model {tf_d} not divisible by {tf_heads} heads"
        );
        let subsampling = req_u32(g, "sortformer.subsampling_factor")?;
        expect_eq("subsampling_factor", subsampling, 8)?;
        let xscaling = g
            .get_bool("sortformer.xscaling")
            .context("missing GGUF key `sortformer.xscaling`")?;

        // Encoder: the stem and blocks reuse the audio-encoder loaders.
        let mut stem_layers = Vec::new();
        for idx in [0u32, 2, 3, 5, 6] {
            stem_layers.push(load_conv_layer(g, idx)?);
        }
        let conv_stem = ConvStemWeights {
            layers: stem_layers,
            pre_encode_out_w: MmapWeight::from_gguf(g, "a.pre_encode.out.weight")
                .context("loading a.pre_encode.out.weight")?,
            pre_encode_out_b: load_vec_f32(g, "a.pre_encode.out.bias")?,
        };
        let mut layers = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            layers.push(load_conformer_block(g, il)?);
        }
        ensure!(
            !layers.is_empty(),
            "Sortformer GGUF has no FastConformer blocks"
        );
        let n_ff = layers[0].ffn_up_w.rows;
        for (il, l) in layers.iter().enumerate() {
            ensure!(
                l.ffn_up_w.cols == n_embd
                    && l.ffn_up_w.rows == n_ff
                    && l.ffn_down_w.rows == n_embd
                    && l.ffn_down_w.cols == n_ff
                    && l.ffn_up_1_w.rows == n_ff
                    && l.ffn_down_1_w.cols == n_ff,
                "block {il}: FFN shapes disagree with n_embd {n_embd} / n_ff {n_ff}"
            );
            ensure!(
                l.conv_dw_shape.first().copied()
                    == Some(req_u32(g, "sortformer.conv_kernel_size")?),
                "block {il}: depthwise kernel {:?} disagrees with sortformer.conv_kernel_size",
                l.conv_dw_shape
            );
        }
        let enc_cfg = AudioEncoderConfig {
            n_layer,
            n_embd,
            n_ff,
            n_head,
            eps,
            n_mel_bins,
            // The stem helper only reads n_mel_bins and n_embd; there is no LLM adapter here.
            llm_hidden_size: tf_d,
        };

        let weight = |name: &str| -> Result<MmapWeight> {
            MmapWeight::from_gguf(g, name).with_context(|| format!("loading {name}"))
        };
        let proj_w = weight("sf.enc_proj.weight")?;
        ensure!(
            proj_w.rows == tf_d && proj_w.cols == n_embd,
            "sf.enc_proj.weight is {}x{}, expected {tf_d}x{n_embd}",
            proj_w.rows,
            proj_w.cols
        );
        let mut tf = Vec::with_capacity(tf_layers);
        for n in 0..tf_layers {
            let p = format!("sf.blk.{n}");
            let w = |s: &str| weight(&format!("{p}.{s}.weight"));
            let b = |s: &str| load_vec_f32(g, &format!("{p}.{s}.bias"));
            tf.push(TransformerLayer {
                ln1_w: load_vec_f32(g, &format!("{p}.ln1.weight"))?,
                ln1_b: b("ln1")?,
                q_w: w("attn_q")?,
                q_b: b("attn_q")?,
                k_w: w("attn_k")?,
                k_b: b("attn_k")?,
                v_w: w("attn_v")?,
                v_b: b("attn_v")?,
                o_w: w("attn_out")?,
                o_b: b("attn_out")?,
                ln2_w: load_vec_f32(g, &format!("{p}.ln2.weight"))?,
                ln2_b: b("ln2")?,
                up_w: w("ffn_up")?,
                up_b: b("ffn_up")?,
                down_w: w("ffn_down")?,
                down_b: b("ffn_down")?,
            });
            let l = &tf[n];
            ensure!(
                l.q_w.rows == tf_d
                    && l.q_w.cols == tf_d
                    && l.up_w.rows == tf_inner
                    && l.up_w.cols == tf_d
                    && l.down_w.rows == tf_d
                    && l.down_w.cols == tf_inner,
                "transformer layer {n}: shapes disagree with d {tf_d} / inner {tf_inner}"
            );
        }
        let head_hidden_w = weight("sf.head.hidden.weight")?;
        let head_out_w = weight("sf.head.out.weight")?;
        ensure!(
            head_hidden_w.rows == tf_d
                && head_hidden_w.cols == tf_d
                && head_out_w.rows == n_spk
                && head_out_w.cols == tf_d,
            "speaker head shapes disagree with d {tf_d} / {n_spk} speakers"
        );

        // Mel tables, exactly as the checkpoint ships them.
        let win = g
            .get_tensor("sf.mel.window")
            .context("loading sf.mel.window")?
            .to_f32_vec();
        ensure!(
            win.len() == WINDOW_LEN,
            "sf.mel.window has {} taps",
            win.len()
        );
        let lo = (N_FFT - WINDOW_LEN) / 2;
        let mut window = vec![0.0f32; N_FFT];
        window[lo..lo + WINDOW_LEN].copy_from_slice(&win);
        let mel_fb = g
            .get_tensor("sf.mel.fb")
            .context("loading sf.mel.fb")?
            .to_f32_vec();
        ensure!(
            mel_fb.len() == n_mel_bins * N_FFT_BINS,
            "sf.mel.fb has {} values, expected {n_mel_bins} x {N_FFT_BINS}",
            mel_fb.len()
        );

        let streaming = StreamingParams {
            chunk_len: req_u32(g, "sortformer.stream.chunk_len")?,
            left_context: req_u32(g, "sortformer.stream.chunk_left_context")?,
            right_context: req_u32(g, "sortformer.stream.chunk_right_context")?,
            fifo_len: req_u32(g, "sortformer.stream.fifo_len")?,
            spkcache_len: req_u32(g, "sortformer.stream.spkcache_len")?,
            update_period: req_u32(g, "sortformer.stream.spkcache_update_period")?,
            sil_frames_per_spk: req_u32(g, "sortformer.stream.spkcache_sil_frames_per_spk")?,
            pred_score_threshold: req_f32(g, "sortformer.stream.pred_score_threshold")?,
            scores_boost_latest: req_f32(g, "sortformer.stream.scores_boost_latest")?,
            sil_threshold: req_f32(g, "sortformer.stream.sil_threshold")?,
            strong_boost_rate: req_f32(g, "sortformer.stream.strong_boost_rate")?,
            weak_boost_rate: req_f32(g, "sortformer.stream.weak_boost_rate")?,
            min_pos_scores_rate: req_f32(g, "sortformer.stream.min_pos_scores_rate")?,
            max_index: req_u32(g, "sortformer.stream.max_index")?,
        };
        streaming.validate()?;

        Ok(Self {
            config: SortformerConfig {
                n_layer,
                n_embd,
                n_ff,
                n_head,
                eps,
                n_mel_bins,
                tf_layers,
                tf_d,
                tf_heads,
                tf_inner,
                tf_eps,
                n_spk,
                subsampling,
                xscaling,
                pad_to: req_u32(g, "sortformer.mel.pad_to")?,
            },
            streaming,
            enc_cfg,
            conv_stem,
            layers,
            proj_w,
            proj_b: load_vec_f32(g, "sf.enc_proj.bias")?,
            tf,
            head_hidden_w,
            head_hidden_b: load_vec_f32(g, "sf.head.hidden.bias")?,
            head_out_w,
            head_out_b: load_vec_f32(g, "sf.head.out.bias")?,
            window,
            mel_fb,
        })
    }
}

/// Encoder frames a stem of three stride-2, kernel-3, pad-1 convolutions makes from `n` mel
/// frames (`ceil(n / 8)`).
pub fn stem_frames(n: usize) -> usize {
    let mut t = n;
    for _ in 0..3 {
        if t == 0 {
            return 0;
        }
        t = (t - 1) / 2 + 1;
    }
    t
}

// ── Model ──────────────────────────────────────────────────────────────────

/// The Sortformer diarizer: front end, encoder, head and streaming driver.
#[derive(Clone)]
pub struct SortformerModel {
    w: Arc<SortformerWeights>,
}

impl SortformerModel {
    /// Load from a converted GGUF.
    pub fn from_gguf(g: &Arc<GgufFile>) -> Result<Self> {
        Ok(Self {
            w: Arc::new(SortformerWeights::from_gguf(g)?),
        })
    }

    /// Open and load a converted GGUF file (needs the `mmap` feature; without it, parse the
    /// bytes with [`GgufFile::from_bytes`] and use [`Self::from_gguf`]).
    #[cfg(feature = "mmap")]
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let g = GgufFile::open_arc(path).with_context(|| format!("opening {}", path.display()))?;
        Self::from_gguf(&g)
    }

    /// Architecture constants.
    pub fn config(&self) -> &SortformerConfig {
        &self.w.config
    }

    /// The checkpoint's streaming defaults.
    pub fn default_streaming(&self) -> &StreamingParams {
        &self.w.streaming
    }

    /// Log-mel features of mono 16 kHz PCM, `[frames × 128]` time-major, with NeMo's
    /// parameters (no per-feature normalization). Returns `(features, frames)`.
    ///
    /// The STFT of `n` samples has `n / 160 + 1` frames; NeMo's sequence length is
    /// `n / 160`, and the last STFT frame (the one that reaches into the trailing center
    /// padding) is masked out of everything downstream. This returns only the valid frames.
    pub fn log_mel(&self, pcm: &[f32]) -> (Vec<f32>, usize) {
        let (mut mel, n) = log_mel_with_tables(
            pcm,
            self.w.config.n_mel_bins,
            &self.w.window,
            &self.w.mel_fb,
            false,
        );
        let valid = n.min(pcm.len() / HOP_LEN);
        mel.truncate(valid * self.w.config.n_mel_bins);
        (mel, valid)
    }

    /// Conv stem plus `pre_encode.out`: `[frames × 128]` mel to `[ceil(frames/8) × 512]` embeddings.
    /// These are what the speaker cache and FIFO store.
    pub fn pre_encode(&self, mel: &[f32], n_frames: usize) -> (Vec<f32>, usize) {
        conv_stem_forward(mel, n_frames, &self.w.conv_stem, &self.w.enc_cfg)
    }

    /// Everything after the stem: x-scale, FastConformer, `encoder_proj`, Transformer and the
    /// speaker head, over `t` pre-encode embeddings (`[t × 512]`). Returns the sigmoid
    /// speaker activities `[t × 4]`.
    pub fn predict(&self, emb: &[f32], t: usize) -> Vec<f32> {
        self.predict_with_taps(emb, t, &mut |_, _| {})
    }

    /// [`Self::predict`] that reports intermediates to `tap` as it goes: `"xscaled"`,
    /// `"enc.layer{i}"`, `"enc_proj"`, `"tf.layer{i}"`, each a `[t × width]` slice. Used by
    /// the parity tests to find the first stage that diverges.
    pub fn predict_with_taps(
        &self,
        emb: &[f32],
        t: usize,
        tap: &mut dyn FnMut(&str, &[f32]),
    ) -> Vec<f32> {
        let w = &*self.w;
        let c = &w.config;
        assert_eq!(emb.len(), t * c.n_embd, "predict: emb must be [t x n_embd]");
        if t == 0 {
            return Vec::new();
        }

        let mut x = emb.to_vec();
        if c.xscaling {
            // NeMo: `x * math.sqrt(d_model)`, a Python float applied to an f32 tensor.
            let scale = (c.n_embd as f64).sqrt() as f32;
            for v in x.iter_mut() {
                *v *= scale;
            }
        }
        tap("xscaled", &x);

        let pos_emb = relative_pos_emb(t);
        let mut scratch_pre_norm = vec![0.0f32; c.n_embd];
        let mut scratch_ff = vec![0.0f32; c.n_ff];
        for (il, layer) in w.layers.iter().enumerate() {
            conformer_block_forward(
                &mut x,
                layer,
                &pos_emb,
                c.n_embd,
                c.n_ff,
                c.n_head,
                t,
                c.eps,
                &mut scratch_pre_norm,
                &mut scratch_ff,
            );
            tap(&format!("enc.layer{il}"), &x);
        }

        let mut h = vec![0.0f32; t * c.tf_d];
        for (src, dst) in x.chunks_exact(c.n_embd).zip(h.chunks_exact_mut(c.tf_d)) {
            w.proj_w.gemv(src, dst);
            cpu::add_inplace(dst, &w.proj_b);
        }
        tap("enc_proj", &h);

        for (il, layer) in w.tf.iter().enumerate() {
            transformer_layer_forward(&mut h, layer, t, c);
            tap(&format!("tf.layer{il}"), &h);
        }

        let mut preds = vec![0.0f32; t * c.n_spk];
        let mut r = vec![0.0f32; c.tf_d];
        let mut hid = vec![0.0f32; c.tf_d];
        for (row, out) in h.chunks_exact(c.tf_d).zip(preds.chunks_exact_mut(c.n_spk)) {
            r.copy_from_slice(row);
            cpu::relu_inplace(&mut r);
            w.head_hidden_w.gemv(&r, &mut hid);
            cpu::add_inplace(&mut hid, &w.head_hidden_b);
            cpu::relu_inplace(&mut hid);
            w.head_out_w.gemv(&hid, out);
            cpu::add_inplace(out, &w.head_out_b);
            cpu::sigmoid_inplace(out);
        }
        preds
    }

    /// Offline diarization of a whole clip in one pass (no streaming state, so the whole
    /// clip attends to itself). Returns `[frames × 4]` speaker activities, 80 ms per frame.
    pub fn diarize_offline(&self, pcm: &[f32]) -> Vec<f32> {
        let (mel, n) = self.log_mel(pcm);
        let (emb, t) = self.pre_encode(&mel, n);
        self.predict(&emb, t)
    }

    /// Start a streaming session with `params` (see [`Self::default_streaming`]).
    pub fn new_stream(&self, params: StreamingParams) -> Result<SortformerStream> {
        params.validate()?;
        Ok(SortformerStream {
            w: self.w.clone(),
            params,
            spkcache: Vec::new(),
            spkcache_preds: None,
            fifo: Vec::new(),
            fifo_preds: Vec::new(),
            mean_sil_emb: vec![0.0; self.w.config.n_embd],
            n_sil_frames: 0,
        })
    }

    /// Streaming diarization of a whole clip, chunked exactly as NeMo's feature loader
    /// chunks it. Returns `[frames × 4]`.
    pub fn diarize_streaming(&self, pcm: &[f32], params: StreamingParams) -> Result<Vec<f32>> {
        let (mel, n) = self.log_mel(pcm);
        self.new_stream(params)?.diarize_features(&mel, n)
    }
}

// ── Live audio ─────────────────────────────────────────────────────────────

/// Incremental NeMo log-mel: feed PCM in pieces of any size, get the same mel frames the
/// whole-clip [`SortformerModel::log_mel`] computes, bit for bit.
///
/// A frame needs 512 samples centered on its hop position, so a frame is ready once 256
/// samples past its center have arrived; the clip's last frames need the trailing center
/// padding, which [`Self::finish`] supplies. Pre-emphasis carries the previous raw sample
/// across calls (the offline path leaves the first sample untouched, which is the same thing
/// with a previous sample of 0).
pub struct MelStream {
    computer: MelFrameComputer,
    n_mel_bins: usize,
    /// Pre-emphasized samples in padded coordinates (256 zeros of left padding, then audio);
    /// `buf[0]` is padded index `base`.
    buf: Vec<f32>,
    base: usize,
    n_in: usize,
    prev_raw: f32,
    next_frame: usize,
    finished: bool,
}

impl MelStream {
    fn new(w: &SortformerWeights) -> Self {
        let n_mel_bins = w.config.n_mel_bins;
        Self {
            computer: MelFrameComputer::new(n_mel_bins, &w.window, &w.mel_fb),
            n_mel_bins,
            buf: vec![0.0; N_FFT / 2],
            base: 0,
            n_in: 0,
            prev_raw: 0.0,
            next_frame: 0,
            finished: false,
        }
    }

    /// Samples pushed so far.
    pub fn samples(&self) -> usize {
        self.n_in
    }

    /// Mel frames produced so far.
    pub fn frames(&self) -> usize {
        self.next_frame
    }

    /// Append mono 16 kHz PCM. Returns the newly completed mel frames, `[k x 128]` time-major.
    pub fn push(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        ensure!(!self.finished, "MelStream: push after finish");
        self.buf.reserve(pcm.len());
        for &x in pcm {
            // The first sample passes through unchanged because `prev_raw` starts at 0.
            let y = x - PREEMPH * self.prev_raw;
            self.buf.push(y);
            self.prev_raw = x;
            self.n_in += 1;
        }
        Ok(self.emit())
    }

    /// End of audio: add the trailing center padding and return the remaining frames.
    pub fn finish(&mut self) -> Vec<f32> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        self.buf.resize(self.buf.len() + N_FFT / 2, 0.0);
        self.emit()
    }

    fn emit(&mut self) -> Vec<f32> {
        let mut out = Vec::new();
        let mut row = vec![0.0f32; self.n_mel_bins];
        // NeMo's length is n / hop: the STFT's extra last frame is masked, never produced.
        while self.next_frame < self.n_in / HOP_LEN {
            let start = self.next_frame * HOP_LEN;
            if start + N_FFT > self.base + self.buf.len() {
                break;
            }
            let lo = start - self.base;
            self.computer.frame(&self.buf[lo..lo + N_FFT], &mut row);
            out.extend_from_slice(&row);
            self.next_frame += 1;
        }
        // Frames before `next_frame` are done with.
        let keep_from = self.next_frame * HOP_LEN;
        if keep_from > self.base {
            self.buf.drain(..keep_from - self.base);
            self.base = keep_from;
        }
        out
    }
}

/// Live diarization: push PCM as it arrives, receive each frame's speaker activities once they
/// are final.
///
/// It chunks the incremental mel the way [`SortformerStream::diarize_features_unpadded`] does
/// and gives bit-identical predictions to it. A chunk is computed once `chunk_len` frames plus
/// `right_context` frames of lookahead exist (see [`Self::latency_frames`]); predictions are
/// never revised. [`Self::finish`] flushes the tail.
pub struct SortformerLive {
    stream: SortformerStream,
    mel: MelStream,
    /// Mel frames from `rows_start` on, `[k x 128]`.
    rows: Vec<f32>,
    rows_start: usize,
    total_mel: usize,
    /// First mel frame of the next chunk (NeMo's `stt_feat`).
    stt: usize,
    emitted: usize,
    finished: bool,
}

impl SortformerModel {
    /// Start live diarization with `params` (see [`Self::default_streaming`]).
    pub fn new_live(&self, params: StreamingParams) -> Result<SortformerLive> {
        Ok(SortformerLive {
            stream: self.new_stream(params)?,
            mel: MelStream::new(&self.w),
            rows: Vec::new(),
            rows_start: 0,
            total_mel: 0,
            stt: 0,
            emitted: 0,
            finished: false,
        })
    }

    /// An incremental mel front end on its own.
    pub fn new_mel_stream(&self) -> MelStream {
        MelStream::new(&self.w)
    }
}

impl SortformerLive {
    /// Worst-case delay in frames (80 ms each): a chunk's first frame waits for the rest of
    /// the chunk and its lookahead, `chunk_len + right_context` (NeMo's definition of the
    /// preset's latency), on top of the 160 ms the mel front end needs after a frame's center.
    pub fn latency_frames(&self) -> usize {
        self.stream.params.chunk_len + self.stream.params.right_context
    }

    /// Prediction frames returned so far (80 ms each).
    pub fn frames_emitted(&self) -> usize {
        self.emitted
    }

    /// The underlying streaming state (speaker cache, FIFO, silence profile).
    pub fn stream(&self) -> &SortformerStream {
        &self.stream
    }

    /// Feed mono 16 kHz PCM of any length. Returns the predictions that became final,
    /// `[k x 4]` for the next `k` frames in order (possibly empty).
    pub fn push_audio(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        ensure!(!self.finished, "SortformerLive: push_audio after finish");
        let rows = self.mel.push(pcm)?;
        self.accept(&rows);
        self.drain(false)
    }

    /// End of audio: flush the remaining frames with whatever lookahead exists.
    pub fn finish(&mut self) -> Result<Vec<f32>> {
        if self.finished {
            return Ok(Vec::new());
        }
        let rows = self.mel.finish();
        self.accept(&rows);
        self.finished = true;
        self.drain(true)
    }

    fn accept(&mut self, rows: &[f32]) {
        self.rows.extend_from_slice(rows);
        self.total_mel += rows.len() / self.stream.w.config.n_mel_bins;
    }

    fn drain(&mut self, last: bool) -> Result<Vec<f32>> {
        let (nm, ss) = (
            self.stream.w.config.n_mel_bins,
            self.stream.w.config.subsampling,
        );
        let chunk_feat = self.stream.params.chunk_len * ss;
        let right_feat = self.stream.params.right_context * ss;
        let left_feat = self.stream.params.left_context * ss;
        let mut out = Vec::new();
        loop {
            let left_offset = left_feat.min(self.stt);
            let (end, right_offset) = if last {
                if self.stt >= self.total_mel {
                    break;
                }
                let end = (self.stt + chunk_feat).min(self.total_mel);
                (end, right_feat.min(self.total_mel - end))
            } else {
                // Wait for the full chunk and its whole lookahead, as in the middle of an
                // offline clip.
                if self.total_mel < self.stt + chunk_feat + right_feat {
                    break;
                }
                (self.stt + chunk_feat, right_feat)
            };
            let first = self.stt - left_offset;
            let n_feat = end + right_offset - first;
            let lo = (first - self.rows_start) * nm;
            let chunk = self.rows[lo..lo + n_feat * nm].to_vec();
            let preds = self
                .stream
                .step(&chunk, n_feat, n_feat, left_offset, right_offset)?;
            self.stt = end;
            self.emitted += preds.len() / self.stream.w.config.n_spk;
            out.extend(preds);

            // Mel frames before the next chunk's left context are no longer needed.
            let keep_from = self.stt.saturating_sub(left_feat);
            if keep_from > self.rows_start {
                self.rows.drain(..(keep_from - self.rows_start) * nm);
                self.rows_start = keep_from;
            }
        }
        Ok(out)
    }
}

// ── Transformer layer ──────────────────────────────────────────────────────

/// One post-LN Transformer layer, in place on `[t × d]`:
/// `x = LN1(x + Attn(x)); x = LN2(x + FFN(x))`, full self-attention with no mask (the rows
/// are all valid) and no positional embedding.
fn transformer_layer_forward(x: &mut [f32], l: &TransformerLayer, t: usize, c: &SortformerConfig) {
    let d = c.tf_d;
    let heads = c.tf_heads;
    let dh = d / heads;
    let scale = 1.0 / (dh as f64).sqrt();

    let mut q = vec![0.0f32; t * d];
    let mut k = vec![0.0f32; t * d];
    let mut v = vec![0.0f32; t * d];
    for i in 0..t {
        let row = &x[i * d..(i + 1) * d];
        for (wt, b, out) in [
            (&l.q_w, &l.q_b, &mut q),
            (&l.k_w, &l.k_b, &mut k),
            (&l.v_w, &l.v_b, &mut v),
        ] {
            let dst = &mut out[i * d..(i + 1) * d];
            wt.gemv(row, dst);
            cpu::add_inplace(dst, b);
        }
    }

    let mut ctx = vec![0.0f32; t * d];
    let mut scores = vec![0.0f32; t];
    for h in 0..heads {
        let off = h * dh;
        for i in 0..t {
            let qi = &q[i * d + off..i * d + off + dh];
            for (j, s) in scores.iter_mut().enumerate() {
                let kj = &k[j * d + off..j * d + off + dh];
                let dot: f64 = qi.iter().zip(kj).map(|(&a, &b)| a as f64 * b as f64).sum();
                *s = (dot * scale) as f32;
            }
            cpu::softmax_inplace(&mut scores);
            for dd in 0..dh {
                let acc: f64 = (0..t)
                    .map(|j| scores[j] as f64 * v[j * d + off + dd] as f64)
                    .sum();
                ctx[i * d + off + dd] = acc as f32;
            }
        }
    }

    let mut att = vec![0.0f32; d];
    let mut ff = vec![0.0f32; c.tf_inner];
    let mut ffo = vec![0.0f32; d];
    for i in 0..t {
        let row = &mut x[i * d..(i + 1) * d];
        l.o_w.gemv(&ctx[i * d..(i + 1) * d], &mut att);
        cpu::add_inplace(&mut att, &l.o_b);
        cpu::add_inplace(row, &att);
        cpu::layer_norm_inplace(row, &l.ln1_w, &l.ln1_b, c.tf_eps);

        l.up_w.gemv(row, &mut ff);
        cpu::add_inplace(&mut ff, &l.up_b);
        cpu::relu_inplace(&mut ff);
        l.down_w.gemv(&ff, &mut ffo);
        cpu::add_inplace(&mut ffo, &l.down_b);
        cpu::add_inplace(row, &ffo);
        cpu::layer_norm_inplace(row, &l.ln2_w, &l.ln2_b, c.tf_eps);
    }
}

// ── Streaming ──────────────────────────────────────────────────────────────

/// A streaming diarization session: the speaker cache, the FIFO and the silence profile.
///
/// Drive it with [`Self::step`] for live audio (one call per chunk of mel frames), or hand it
/// a whole clip's features with [`Self::diarize_features`].
pub struct SortformerStream {
    w: Arc<SortformerWeights>,
    params: StreamingParams,
    /// `[n × n_embd]`, speaker-ordered once compressed.
    spkcache: Vec<f32>,
    /// `[n × n_spk]`; `None` until the cache has been compressed once (NeMo's `None`).
    spkcache_preds: Option<Vec<f32>>,
    fifo: Vec<f32>,
    fifo_preds: Vec<f32>,
    mean_sil_emb: Vec<f32>,
    n_sil_frames: usize,
}

impl SortformerStream {
    /// The parameters this session runs with.
    pub fn params(&self) -> &StreamingParams {
        &self.params
    }

    /// Speaker-cache embeddings, `[len × n_embd]`.
    pub fn spkcache(&self) -> &[f32] {
        &self.spkcache
    }

    /// Speaker-cache predictions `[len × n_spk]`, or `None` before the first compression.
    pub fn spkcache_preds(&self) -> Option<&[f32]> {
        self.spkcache_preds.as_deref()
    }

    /// FIFO embeddings, `[len × n_embd]`.
    pub fn fifo(&self) -> &[f32] {
        &self.fifo
    }

    /// FIFO predictions, `[len × n_spk]`.
    pub fn fifo_preds(&self) -> &[f32] {
        &self.fifo_preds
    }

    /// Running mean of the embeddings of frames classified as silence.
    pub fn mean_sil_emb(&self) -> &[f32] {
        &self.mean_sil_emb
    }

    /// How many silence frames fed [`Self::mean_sil_emb`].
    pub fn n_sil_frames(&self) -> usize {
        self.n_sil_frames
    }

    /// Run a whole clip's features (`[n_frames × 128]`, from [`SortformerModel::log_mel`])
    /// through the streaming loop, chunked like NeMo's `streaming_feat_loader` over features
    /// padded to a multiple of `pad_to`. Returns `[frames × 4]`; frames past the audio are zeros.
    pub fn diarize_features(&mut self, mel: &[f32], n_frames: usize) -> Result<Vec<f32>> {
        self.diarize_features_with(mel, n_frames, &mut |_, _, _| {})
    }

    /// Like [`Self::diarize_features`], but chunked as a live stream sees the audio: the
    /// features are not padded to `pad_to`, so the final chunk ends at the last real frame.
    /// [`SortformerLive`] produces exactly these predictions.
    pub fn diarize_features_unpadded(&mut self, mel: &[f32], n_frames: usize) -> Result<Vec<f32>> {
        self.chunk_loop(mel, n_frames, n_frames, &mut |_, _, _| {})
    }

    /// [`Self::diarize_features`] that calls `on_step(index, stream, chunk_preds)` after every
    /// step, with the stream's state as the step left it. The parity tests compare that state
    /// with NeMo's, step by step.
    pub fn diarize_features_with(
        &mut self,
        mel: &[f32],
        n_frames: usize,
        on_step: &mut dyn FnMut(usize, &SortformerStream, &[f32]),
    ) -> Result<Vec<f32>> {
        let pad_to = self.w.config.pad_to.max(1);
        self.chunk_loop(mel, n_frames, n_frames.div_ceil(pad_to) * pad_to, on_step)
    }

    /// NeMo's `streaming_feat_loader` over `feat_len` frames (`>= n_frames`; the tail past
    /// `n_frames` is padding that is never computed).
    fn chunk_loop(
        &mut self,
        mel: &[f32],
        n_frames: usize,
        feat_len: usize,
        on_step: &mut dyn FnMut(usize, &SortformerStream, &[f32]),
    ) -> Result<Vec<f32>> {
        let c = &self.w.config;
        let (nm, ss) = (c.n_mel_bins, c.subsampling);
        ensure!(mel.len() == n_frames * nm, "mel is not [n_frames x {nm}]");
        let chunk_feat = self.params.chunk_len * ss;

        let mut total = Vec::new();
        let (mut stt, mut end, mut idx) = (0usize, 0usize, 0usize);
        while end < feat_len {
            let left_offset = (self.params.left_context * ss).min(stt);
            end = (stt + chunk_feat).min(feat_len);
            let right_offset = (self.params.right_context * ss).min(feat_len - end);
            let first = stt - left_offset;
            let n_feat = end + right_offset - first;
            let valid = n_frames.saturating_sub(first).min(n_feat);
            let mut chunk = vec![0.0f32; n_feat * nm];
            chunk[..valid * nm].copy_from_slice(&mel[first * nm..(first + valid) * nm]);
            stt = end;
            let preds = self.step(&chunk, n_feat, valid, left_offset, right_offset)?;
            on_step(idx, self, &preds);
            total.extend(preds);
            idx += 1;
        }
        Ok(total)
    }

    /// One streaming step.
    ///
    /// `feats` is `[n_feat × 128]` mel: the chunk with `left_offset` frames of left context
    /// before it and `right_offset` frames of lookahead after it (both in mel frames, as in
    /// NeMo's loader). `valid_feat <= n_feat` says how many leading frames are real audio; a live
    /// caller passes `valid_feat == n_feat`. The rest (end-of-clip padding) is not computed.
    ///
    /// Returns the chunk's predictions `[chunk_len × 4]`, where `chunk_len` is the number of
    /// encoder frames between the two contexts. Updates the cache, FIFO and silence profile.
    pub fn step(
        &mut self,
        feats: &[f32],
        n_feat: usize,
        valid_feat: usize,
        left_offset: usize,
        right_offset: usize,
    ) -> Result<Vec<f32>> {
        let w = self.w.clone();
        let c = &w.config;
        let (d, s, ss) = (c.n_embd, c.n_spk, c.subsampling);
        ensure!(
            feats.len() == n_feat * c.n_mel_bins,
            "feats is not [n_feat x {}]",
            c.n_mel_bins
        );
        ensure!(
            valid_feat <= n_feat,
            "valid_feat {valid_feat} > n_feat {n_feat}"
        );

        // Frames NeMo would see (padding included) and the ones that exist here.
        let enc_total = stem_frames(n_feat);
        let lc = left_offset / ss; // exact: left offsets are whole encoder frames
        let rc = right_offset.div_ceil(ss);
        ensure!(
            enc_total >= lc + rc,
            "chunk of {enc_total} encoder frames is shorter than its contexts ({lc} + {rc})"
        );
        let chunk_len = enc_total - lc - rc;

        let (chunk_valid, enc_valid) = if valid_feat == 0 {
            (Vec::new(), 0)
        } else {
            w.conv_stem_for(&feats[..valid_feat * c.n_mel_bins])
        };
        debug_assert_eq!(enc_valid, stem_frames(valid_feat));
        let model = SortformerModel { w: w.clone() };

        let n_sc = self.spkcache.len() / d;
        let n_fifo = self.fifo.len() / d;
        let mut concat = Vec::with_capacity((n_sc + n_fifo + enc_valid) * d);
        concat.extend_from_slice(&self.spkcache);
        concat.extend_from_slice(&self.fifo);
        concat.extend_from_slice(&chunk_valid);
        let valid_rows = n_sc + n_fifo + enc_valid;
        let mut preds = model.predict(&concat, valid_rows);
        // Pad frames get zero predictions and zero embeddings, like NeMo's masked output.
        let total_rows = n_sc + n_fifo + enc_total;
        preds.resize(total_rows * s, 0.0);
        let mut chunk_emb = chunk_valid;
        chunk_emb.resize(enc_total * d, 0.0);

        // ---- streaming_update (synchronous path) ----
        let (fifo_cap, update_period, cache_cap) = (
            self.params.fifo_len,
            self.params.update_period,
            self.params.spkcache_len,
        );
        let fifo_preds_now = preds[n_sc * s..(n_sc + n_fifo) * s].to_vec();
        let chunk_slice = &chunk_emb[lc * d..(lc + chunk_len) * d];
        let base = n_sc + n_fifo + lc;
        let chunk_preds = preds[base * s..(base + chunk_len) * s].to_vec();

        self.fifo.extend_from_slice(chunk_slice);
        self.fifo_preds = fifo_preds_now;
        self.fifo_preds.extend_from_slice(&chunk_preds);

        if n_fifo + chunk_len > fifo_cap {
            let mut pop = update_period;
            pop = pop.max((chunk_len + n_fifo).saturating_sub(fifo_cap));
            pop = pop.min(n_fifo + chunk_len);

            let pop_embs = self.fifo[..pop * d].to_vec();
            let pop_preds = self.fifo_preds[..pop * s].to_vec();
            self.update_silence_profile(&pop_embs, &pop_preds, pop);
            self.fifo.drain(..pop * d);
            self.fifo_preds.drain(..pop * s);

            self.spkcache.extend_from_slice(&pop_embs);
            if let Some(sp) = self.spkcache_preds.as_mut() {
                sp.extend_from_slice(&pop_preds);
            }
            let cache_rows = self.spkcache.len() / d;
            if cache_rows > cache_cap {
                if self.spkcache_preds.is_none() {
                    // First compression: the cache's own predictions come from this step's forward.
                    let mut sp = preds[..n_sc * s].to_vec();
                    sp.extend_from_slice(&pop_preds);
                    self.spkcache_preds = Some(sp);
                }
                let (emb, pr) = compress_spkcache(
                    &self.params,
                    self.w.config.n_spk,
                    self.w.config.n_embd,
                    &self.spkcache,
                    self.spkcache_preds.as_ref().expect("set above"),
                    cache_rows,
                    &self.mean_sil_emb,
                );
                self.spkcache = emb;
                self.spkcache_preds = Some(pr);
            }
        }
        Ok(chunk_preds)
    }

    /// NeMo `_get_silence_profile`: fold the silent frames of `embs` into the running mean.
    fn update_silence_profile(&mut self, embs: &[f32], preds: &[f32], n: usize) {
        let (d, s) = (self.w.config.n_embd, self.w.config.n_spk);
        let mut sum = vec![0.0f32; d];
        let mut count = 0usize;
        for f in 0..n {
            let total: f32 = preds[f * s..(f + 1) * s].iter().sum();
            if total < self.params.sil_threshold {
                count += 1;
                for (a, b) in sum.iter_mut().zip(&embs[f * d..(f + 1) * d]) {
                    *a += *b;
                }
            }
        }
        if count == 0 {
            return;
        }
        let upd = self.n_sil_frames + count;
        let denom = upd.max(1) as f32;
        for (m, add) in self.mean_sil_emb.iter_mut().zip(&sum) {
            *m = (*m * self.n_sil_frames as f32 + add) / denom;
        }
        self.n_sil_frames = upd;
    }
}

/// NeMo `_compress_spkcache` (inference: no speaker permutation, no score noise).
/// Keeps the `spkcache_len` most important of the `n` cached frames, ordered by speaker
/// and then by time, with `sil_frames_per_spk` silence slots closing each speaker's block.
fn compress_spkcache(
    p: &StreamingParams,
    s: usize,
    d: usize,
    emb: &[f32],
    preds: &[f32],
    n: usize,
    mean_sil_emb: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(preds.len(), n * s);

    let per_spk = p.spkcache_len / s - p.sil_frames_per_spk;
    let strong = (per_spk as f64 * p.strong_boost_rate as f64).floor() as usize;
    let weak = (per_spk as f64 * p.weak_boost_rate as f64).floor() as usize;
    let min_pos = (per_spk as f64 * p.min_pos_scores_rate as f64).floor() as usize;

    // _get_log_pred_scores
    let mut scores = vec![0.0f32; n * s];
    for f in 0..n {
        let row = &preds[f * s..(f + 1) * s];
        let log_1p: Vec<f32> = row
            .iter()
            .map(|&x| (1.0 - x).max(p.pred_score_threshold).ln())
            .collect();
        let log_1p_sum: f32 = log_1p.iter().sum();
        for k in 0..s {
            let log_p = row[k].max(p.pred_score_threshold).ln();
            scores[f * s + k] = log_p - log_1p[k] + log_1p_sum - (0.5f64).ln() as f32;
        }
    }

    // _disable_low_scores
    let mut pos_count = vec![0usize; s];
    for f in 0..n {
        for k in 0..s {
            let sc = if preds[f * s + k] > 0.5 {
                scores[f * s + k]
            } else {
                f32::NEG_INFINITY
            };
            scores[f * s + k] = sc;
            if sc > 0.0 {
                pos_count[k] += 1;
            }
        }
    }
    for f in 0..n {
        for k in 0..s {
            let speech = preds[f * s + k] > 0.5;
            let is_pos = scores[f * s + k] > 0.0;
            if !is_pos && speech && pos_count[k] >= min_pos {
                scores[f * s + k] = f32::NEG_INFINITY;
            }
        }
    }

    if p.scores_boost_latest > 0.0 {
        for f in p.spkcache_len.min(n)..n {
            for k in 0..s {
                scores[f * s + k] += p.scores_boost_latest;
            }
        }
    }

    // _boost_topk_scores: strong (x2) then weak (x1), per speaker over time.
    for (n_boost, scale) in [(strong, 2.0f64), (weak, 1.0f64)] {
        let delta = (scale * (0.5f64).ln()) as f32;
        for k in 0..s {
            let col: Vec<f32> = (0..n).map(|f| scores[f * s + k]).collect();
            for f in topk_desc(&col, n_boost.min(n)) {
                scores[f * s + k] -= delta;
            }
        }
    }

    // Silence rows (+inf) close each speaker's block, then _get_topk_indices over the
    // speaker-major flattening.
    let sil = p.sil_frames_per_spk;
    let n_ext = n + sil;
    let mut flat = vec![0.0f32; s * n_ext];
    for k in 0..s {
        for f in 0..n_ext {
            flat[k * n_ext + f] = if f < n {
                scores[f * s + k]
            } else {
                f32::INFINITY
            };
        }
    }
    let mut picked: Vec<usize> = topk_desc(&flat, p.spkcache_len)
        .into_iter()
        .filter(|&i| flat[i] != f32::NEG_INFINITY)
        .collect();
    picked.resize(p.spkcache_len, p.max_index);
    picked.sort_unstable();

    let mut out_emb = vec![0.0f32; p.spkcache_len * d];
    let mut out_preds = vec![0.0f32; p.spkcache_len * s];
    for (slot, &flat_idx) in picked.iter().enumerate() {
        let frame = flat_idx % n_ext;
        let disabled = flat_idx == p.max_index || frame >= n;
        if disabled {
            out_emb[slot * d..(slot + 1) * d].copy_from_slice(mean_sil_emb);
        } else {
            out_emb[slot * d..(slot + 1) * d].copy_from_slice(&emb[frame * d..(frame + 1) * d]);
            out_preds[slot * s..(slot + 1) * s].copy_from_slice(&preds[frame * s..(frame + 1) * s]);
        }
    }
    (out_emb, out_preds)
}

impl SortformerWeights {
    /// The stem on an already-validated slice of mel frames.
    fn conv_stem_for(&self, mel: &[f32]) -> (Vec<f32>, usize) {
        let n = mel.len() / self.config.n_mel_bins;
        conv_stem_forward(mel, n, &self.conv_stem, &self.enc_cfg)
    }
}

/// Indices of the `k` largest values, largest first; ties go to the lower index. NaN sorts last.
/// (`torch.topk(sorted=False)` leaves tie order unspecified; the cache selection only depends on
/// the chosen set, and ties at the cut are the one place that could differ.)
fn topk_desc(vals: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&a, &b| {
        let (x, y) = (vals[a], vals[b]);
        y.partial_cmp(&x)
            .unwrap_or_else(|| x.is_nan().cmp(&y.is_nan()))
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stem_frames_is_ceil_div_8() {
        for n in 0..2000usize {
            assert_eq!(stem_frames(n), n.div_ceil(8), "n = {n}");
        }
        // The golden clip: 1537 mel frames -> 193 encoder frames.
        assert_eq!(stem_frames(1537), 193);
        assert_eq!(stem_frames(1552), 194);
    }

    #[test]
    fn topk_orders_by_value_then_index_and_ranks_nan_last() {
        let v = [1.0, 5.0, 5.0, f32::NAN, f32::NEG_INFINITY, 3.0];
        assert_eq!(topk_desc(&v, 3), vec![1, 2, 5]);
        assert_eq!(topk_desc(&v, 6), vec![1, 2, 5, 0, 4, 3]);
        assert_eq!(topk_desc(&v, 0), Vec::<usize>::new());
    }

    #[test]
    fn streaming_params_validation() {
        let ok = StreamingParams {
            chunk_len: 6,
            left_context: 1,
            right_context: 7,
            fifo_len: 188,
            spkcache_len: 188,
            update_period: 144,
            sil_frames_per_spk: 3,
            pred_score_threshold: 0.25,
            scores_boost_latest: 0.05,
            sil_threshold: 0.2,
            strong_boost_rate: 0.75,
            weak_boost_rate: 1.5,
            min_pos_scores_rate: 0.5,
            max_index: 99_999,
        };
        assert!(ok.validate().is_ok());
        // 4 speakers x (3 silence + 0) frames leaves nothing for speech.
        assert!(ok.with_chunking(6, 1, 7, 188, 12, 144).validate().is_err());
        assert!(ok.with_chunking(0, 1, 7, 188, 188, 144).validate().is_err());
        assert!(ok.with_chunking(6, 1, 7, 188, 188, 0).validate().is_err());
    }

    fn params(spkcache_len: usize) -> StreamingParams {
        StreamingParams {
            chunk_len: 6,
            left_context: 1,
            right_context: 7,
            fifo_len: 188,
            spkcache_len,
            update_period: 144,
            sil_frames_per_spk: 1,
            pred_score_threshold: 0.25,
            scores_boost_latest: 0.05,
            sil_threshold: 0.2,
            strong_boost_rate: 0.75,
            weak_boost_rate: 1.5,
            min_pos_scores_rate: 0.5,
            max_index: 99_999,
        }
    }

    /// Embedding width 1 so each frame's embedding is its own marker (frame f holds f + 1) and
    /// the silence embedding is -1.
    fn compress(preds: &[[f32; 4]], spkcache_len: usize) -> (Vec<f32>, Vec<f32>) {
        let n = preds.len();
        let emb: Vec<f32> = (0..n).map(|f| (f + 1) as f32).collect();
        let flat: Vec<f32> = preds.iter().flatten().copied().collect();
        compress_spkcache(&params(spkcache_len), 4, 1, &emb, &flat, n, &[-1.0])
    }

    #[test]
    fn compression_orders_by_speaker_and_closes_each_block_with_silence() {
        // spkcache_len 24 / 4 speakers - 1 silence = 5 frames per speaker; min_pos = floor(5 * 0.5) = 2.
        let (emb, preds) = compress(
            &[
                [0.9, 0.0, 0.0, 0.0],
                [0.9, 0.0, 0.0, 0.0],
                [0.0, 0.9, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0], // silence: not speech for anyone
            ],
            24,
        );
        // speaker 0: frames 0, 1, then its silence slot; speaker 1: frame 2, silence; speakers 2 and 3
        // have only their silence slot; the 24 - 5 unused slots are disabled and come last.
        let mut want = vec![1.0, 2.0, -1.0, 3.0, -1.0, -1.0, -1.0];
        want.resize(24, -1.0);
        assert_eq!(emb, want);
        // Disabled slots predict nothing; real frames keep their predictions.
        assert_eq!(&preds[..4], &[0.9, 0.0, 0.0, 0.0]);
        assert_eq!(&preds[8..12], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&preds[12..16], &[0.0, 0.9, 0.0, 0.0]);
        assert!(preds[16..].iter().all(|&p| p == 0.0));
    }

    #[test]
    fn overlapped_frames_are_dropped_only_once_a_speaker_has_min_pos_clean_frames() {
        // Frame 2 has speakers 0 and 1 talking at once (a non-positive score for both). The
        // threshold is floor(5 * 0.5) = 2 positive frames per speaker.
        //   speaker 0 has exactly 2 clean frames -> `>=` min_pos: its overlap frame is dropped;
        //   speaker 1 has 1 clean frame          -> below min_pos: its overlap frame stays.
        let (emb, _) = compress(
            &[
                [0.9, 0.0, 0.0, 0.0],
                [0.9, 0.0, 0.0, 0.0],
                [0.9, 0.9, 0.0, 0.0],
                [0.0, 0.9, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            24,
        );
        let mut want = vec![1.0, 2.0, -1.0, 3.0, 4.0, -1.0, -1.0, -1.0];
        want.resize(24, -1.0);
        assert_eq!(
            emb, want,
            "frame 3 must appear once (speaker 1's), not twice"
        );
    }
}
