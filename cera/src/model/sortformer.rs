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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result, ensure};

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::model::audio_encoder::{
    AudioEncoderConfig, ConformerLayerWeights, ConvStemWeights, EncoderParts, HOP_LEN, LOG_MEL_EPS,
    N_FFT, POS_EMB_DIM, PREEMPH, SAMPLE_RATE, WINDOW_LEN, conformer_block_forward,
    conv_stem_forward, load_conformer_block, load_conv_layer, load_vec_f32, relative_pos_emb,
};
use crate::model::audio_preprocessor::{
    MelFrameComputer, N_FFT_BINS, log_mel_with_tables, n_frames_for, padded_preemphasized,
};
use crate::model::weights::MmapWeight;
use crate::tensor::DType;

/// Largest speaker count the model's output head has; fixed by the checkpoint.
pub const MAX_SPEAKERS: usize = 4;

/// Largest value accepted for any streaming length, in 80 ms frames (about 24 hours). The
/// checkpoint's own values are in the hundreds.
const MAX_STREAM_FRAMES: usize = 1 << 20;

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

    /// Return the recommended low-latency preset (0.48 s chunk, 1.04 s latency).
    pub fn low_latency(&self) -> Self {
        self.with_chunking(6, 1, 7, 188, 188, 144)
    }

    /// Encoder frames one step attends over at most: both contexts, the chunk, the FIFO and
    /// the speaker cache. What an accelerator has to be staged for.
    pub fn window_frames(&self) -> usize {
        // Saturating: the fields are public and `with_chunking` skips validation, so a
        // hand-made config could otherwise wrap (release) or panic (debug) here.
        self.left_context
            .saturating_add(self.chunk_len)
            .saturating_add(self.right_context)
            .saturating_add(self.fifo_len)
            .saturating_add(self.spkcache_len)
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.chunk_len > 0, "chunk_len must be > 0");
        ensure!(self.update_period > 0, "update_period must be > 0");
        // Bound every length so the frame arithmetic downstream cannot overflow.
        for (name, v) in [
            ("chunk_len", self.chunk_len),
            ("left_context", self.left_context),
            ("right_context", self.right_context),
            ("fifo_len", self.fifo_len),
            ("spkcache_len", self.spkcache_len),
            ("update_period", self.update_period),
            ("sil_frames_per_spk", self.sil_frames_per_spk),
        ] {
            ensure!(
                v <= MAX_STREAM_FRAMES,
                "{name} {v} exceeds {MAX_STREAM_FRAMES}"
            );
        }
        // One step attends over all of these at once (memory is quadratic in the sum), so cap
        // the sum like the offline pass: a hostile file's defaults must not make a stream do
        // an unbounded single pass.
        let window = self.left_context
            + self.chunk_len
            + self.right_context
            + self.fifo_len
            + self.spkcache_len;
        ensure!(
            window <= MAX_OFFLINE_FRAMES,
            "left + chunk + right + fifo + spkcache = {window} frames exceeds {MAX_OFFLINE_FRAMES}"
        );
        // `max_index` marks disabled cache slots, so it must lie above every real flat index
        // (speaker-major over the frames of a cache about to be compressed, plus its silences).
        let flat_range = MAX_SPEAKERS
            * (self.spkcache_len + self.fifo_len + self.chunk_len + self.sil_frames_per_spk);
        ensure!(
            self.max_index >= flat_range,
            "max_index {} lies inside the cache's flat index range (needs >= {flat_range})",
            self.max_index
        );
        // The scores are `ln(max(p, threshold))`: a threshold of 0 would give `ln(0)`.
        ensure!(
            self.pred_score_threshold > 0.0 && self.pred_score_threshold <= 1.0,
            "pred_score_threshold {} must be in (0, 1]",
            self.pred_score_threshold
        );
        ensure!(
            self.scores_boost_latest.is_finite() && self.scores_boost_latest >= 0.0,
            "scores_boost_latest {} must be a finite non-negative number",
            self.scores_boost_latest
        );
        ensure!(
            self.sil_threshold.is_finite() && self.sil_threshold >= 0.0,
            "sil_threshold {} must be a finite non-negative number",
            self.sil_threshold
        );
        ensure!(
            self.strong_boost_rate.is_finite() && self.strong_boost_rate >= 0.0,
            "strong_boost_rate {} must be a finite non-negative number",
            self.strong_boost_rate
        );
        ensure!(
            self.weak_boost_rate.is_finite() && self.weak_boost_rate >= 0.0,
            "weak_boost_rate {} must be a finite non-negative number",
            self.weak_boost_rate
        );
        ensure!(
            self.min_pos_scores_rate.is_finite() && (0.0..=1.0).contains(&self.min_pos_scores_rate),
            "min_pos_scores_rate {} must be in [0, 1]",
            self.min_pos_scores_rate
        );
        ensure!(
            self.spkcache_len / MAX_SPEAKERS > self.sil_frames_per_spk,
            "spkcache_len {} leaves no room for speakers beside {} silence frames each",
            self.spkcache_len,
            self.sil_frames_per_spk
        );
        Ok(())
    }
}

/// An accelerator (the Hexagon NPU) for the two heavy parts of a diarization step. Set once per
/// model with [`SortformerModel::set_accelerator`]; every stream and live diarizer made from the
/// model then uses it. Each method returns `Ok(None)` to decline an input it cannot take (for
/// example one longer than the window it was staged for), and an `Err` is a failure: both fall
/// back to the CPU, the failure with a one-time warning, so a diarizer never stops because the
/// accelerator did. An output of the wrong length is treated like a failure of that stage (CPU
/// fallback with a one-time warning), never trusted.
pub trait SortformerAccelerator: Send + Sync {
    /// The conv stem and `pre_encode.out` over `n_frames` of `[n_frames x n_mel]` mel:
    /// `[stem_frames(n_frames) x n_embd]`, like [`SortformerModel::pre_encode`].
    fn pre_encode(&self, mel: &[f32], n_frames: usize) -> Result<Option<Vec<f32>>>;

    /// The x-scale, FastConformer, `encoder_proj`, Transformer and speaker head over `t`
    /// pre-encode embeddings: `[t x n_spk]` sigmoid activities, like [`SortformerModel::predict`].
    fn predict(&self, emb: &[f32], t: usize) -> Result<Option<Vec<f32>>>;

    /// Log-mel of `n_frames` frames over pre-emphasised, centre-padded samples: frame `f` is
    /// `samples[f * 160 .. f * 160 + 512]`, windowed with the model's own window, projected with
    /// its own filterbank, and `ln(energy + 2^-24)` of the result. `[n_frames x n_mel]`,
    /// time-major, the values [`SortformerModel::log_mel`] returns. Optional: the default
    /// declines, leaving the mel on the CPU.
    fn log_mel(&self, samples: &[f32], n_frames: usize) -> Result<Option<Vec<f32>>> {
        let _ = (samples, n_frames);
        Ok(None)
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

pub(crate) struct TransformerLayer {
    pub(crate) ln1_w: Vec<f32>,
    pub(crate) ln1_b: Vec<f32>,
    pub(crate) q_w: MmapWeight,
    pub(crate) q_b: Vec<f32>,
    pub(crate) k_w: MmapWeight,
    pub(crate) k_b: Vec<f32>,
    pub(crate) v_w: MmapWeight,
    pub(crate) v_b: Vec<f32>,
    pub(crate) o_w: MmapWeight,
    pub(crate) o_b: Vec<f32>,
    pub(crate) ln2_w: Vec<f32>,
    pub(crate) ln2_b: Vec<f32>,
    pub(crate) up_w: MmapWeight,
    pub(crate) up_b: Vec<f32>,
    pub(crate) down_w: MmapWeight,
    pub(crate) down_b: Vec<f32>,
}

/// Every Sortformer tensor, loaded from a converted GGUF. Held by [`SortformerModel`] and its
/// streams; the crate's callers go through those.
pub(crate) struct SortformerWeights {
    /// Architecture constants.
    pub config: SortformerConfig,
    /// The checkpoint's streaming defaults.
    pub streaming: StreamingParams,
    enc_cfg: AudioEncoderConfig,
    conv_stem: ConvStemWeights,
    layers: Vec<ConformerLayerWeights>,
    pub(crate) proj_w: MmapWeight,
    pub(crate) proj_b: Vec<f32>,
    pub(crate) tf: Vec<TransformerLayer>,
    pub(crate) head_hidden_w: MmapWeight,
    pub(crate) head_hidden_b: Vec<f32>,
    pub(crate) head_out_w: MmapWeight,
    pub(crate) head_out_b: Vec<f32>,
    /// `N_FFT`-long window with the `WINDOW_LEN` taps centered in it.
    pub(crate) window: Vec<f32>,
    /// `[n_mel_bins × N_FFT_BINS]`.
    pub(crate) mel_fb: Vec<f32>,
    /// Set once by [`SortformerModel::set_accelerator`].
    accel: OnceLock<Arc<dyn SortformerAccelerator>>,
    /// Whether an accelerator failure has been logged, once per [`AccelStage`].
    accel_warned: AccelWarned,
}

/// One warn-once latch per [`AccelStage`]. Named fields, not an indexed array, so adding a
/// stage fails the build (in `warned`) instead of panicking with an out-of-bounds index on
/// the first warning.
#[derive(Default)]
struct AccelWarned {
    log_mel: AtomicBool,
    stem: AtomicBool,
    predict: AtomicBool,
}

/// One stage of the accelerated diarization step. Each stage warns once, independently: a
/// transient failure in one must not suppress the first warning of another.
#[derive(Clone, Copy)]
enum AccelStage {
    LogMel,
    Stem,
    Predict,
}

impl AccelStage {
    fn label(self) -> &'static str {
        match self {
            AccelStage::LogMel => "log-mel",
            AccelStage::Stem => "conv stem",
            AccelStage::Predict => "predict",
        }
    }

    /// This stage's warn-once latch. Exhaustive: a new variant fails the build here.
    fn warned(self, latched: &AccelWarned) -> &AtomicBool {
        match self {
            AccelStage::LogMel => &latched.log_mel,
            AccelStage::Stem => &latched.stem,
            AccelStage::Predict => &latched.predict,
        }
    }
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

        // The padded length drives how many steps a clip takes, so a hostile value is a CPU hang.
        let pad_to = req_u32(g, "sortformer.mel.pad_to")?;
        ensure!(
            (1..=MAX_PAD_TO).contains(&pad_to),
            "sortformer.mel.pad_to {pad_to} outside 1..={MAX_PAD_TO}"
        );
        let n_layer = req_u32(g, "clip.audio.block_count")?;
        let n_embd = req_u32(g, "clip.audio.embedding_length")?;
        // The relative position table is a fixed `POS_EMB_DIM` columns wide.
        expect_eq("embedding_length", n_embd, POS_EMB_DIM)?;
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
            tf_d > 0 && tf_heads > 0 && tf_d % tf_heads == 0,
            "sortformer.tf_d_model {tf_d} must be > 0 and divisible by sortformer.tf_head_count {tf_heads}"
        );
        ensure!(
            tf_d / tf_heads <= 64,
            "sortformer head dimension {} ({tf_d} / {tf_heads}) exceeds maximum supported accumulator width 64",
            tf_d / tf_heads
        );
        ensure!(tf_inner > 0, "sortformer.tf_inner_size must be > 0");
        ensure!(
            n_head > 0 && n_embd % n_head == 0,
            "clip.audio.attention.head_count {n_head} must be > 0 and divide \
             clip.audio.embedding_length {n_embd}"
        );
        // Counts come from the file; bound them before they size an allocation.
        ensure!(
            (1..=MAX_LAYERS).contains(&n_layer) && (1..=MAX_LAYERS).contains(&tf_layers),
            "layer counts {n_layer} / {tf_layers} outside 1..={MAX_LAYERS}"
        );
        let subsampling = req_u32(g, "sortformer.subsampling_factor")?;
        expect_eq("subsampling_factor", subsampling, 8)?;
        let xscaling = g
            .get_bool("sortformer.xscaling")
            .context("missing GGUF key `sortformer.xscaling`")?;
        let kernel = req_u32(g, "sortformer.conv_kernel_size")?;

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
        let n_ff = layers[0].ffn_up_w.rows;
        for (il, l) in layers.iter().enumerate() {
            check_encoder_block(il, l, n_embd, n_ff, kernel)?;
        }
        check_stem(&conv_stem, n_embd, n_mel_bins)?;
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
            let w = MmapWeight::from_gguf(g, name).with_context(|| format!("loading {name}"))?;
            check_gemv_dtype(name, &w)?;
            Ok(w)
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
            check_transformer_layer(n, &tf[n], tf_d, tf_inner)?;
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
        let proj_b = load_vec_f32(g, "sf.enc_proj.bias")?;
        let head_hidden_b = load_vec_f32(g, "sf.head.hidden.bias")?;
        let head_out_b = load_vec_f32(g, "sf.head.out.bias")?;
        for (name, v, want) in [
            ("sf.enc_proj.bias", &proj_b, tf_d),
            ("sf.head.hidden.bias", &head_hidden_b, tf_d),
            ("sf.head.out.bias", &head_out_b, n_spk),
        ] {
            ensure!(
                v.len() == want,
                "{name} has {} values, expected {want}",
                v.len()
            );
        }

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
                pad_to,
            },
            streaming,
            enc_cfg,
            conv_stem,
            layers,
            proj_w,
            proj_b,
            tf,
            head_hidden_w,
            head_hidden_b,
            head_out_w,
            head_out_b,
            window,
            mel_fb,
            accel: OnceLock::new(),
            accel_warned: AccelWarned::default(),
        })
    }
}

/// Upper bound on a layer count read from a GGUF (the shipped model has 17 and 18).
const MAX_LAYERS: usize = 256;

/// Longest clip [`SortformerModel::diarize_offline`] takes, and widest window any entry point
/// attends over, in encoder frames (10 minutes). Full self-attention is quadratic: about 1.8 GB
/// of scores per block at the limit, on top of the conv stem's peak of about 3.5 GB (about
/// 58 KB per mel frame, measured).
const MAX_OFFLINE_FRAMES: usize = 7_500;

/// The matmul kernels cover these storage types; anything else would be a silent zero matrix
/// (`gemv_dispatch` only `debug_assert`s on an unsupported type).
fn check_gemv_dtype(name: &str, w: &MmapWeight) -> Result<()> {
    ensure!(
        matches!(
            w.dtype,
            DType::F32
                | DType::F16
                | DType::BF16
                | DType::Q8_0
                | DType::Q4_0
                | DType::Q4_1
                | DType::Q4KM
                | DType::Q5KM
                | DType::Q6K
        ),
        "{name}: storage type {:?} has no matmul kernel",
        w.dtype
    );
    // The kernels read whole blocks per row: a column count that is not a multiple of the
    // block size (a 192-wide matrix typed as a 256-wide K-quant) gives all-zero output.
    ensure!(
        w.cols.is_multiple_of(w.dtype.block_size()),
        "{name}: {} columns is not a multiple of the {:?} block size {}",
        w.cols,
        w.dtype,
        w.dtype.block_size()
    );
    Ok(())
}

/// Upper bound on `sortformer.mel.pad_to` (NeMo's value is 16).
const MAX_PAD_TO: usize = 64;

/// The stem's shapes, which `conv_stem_forward` otherwise only asserts at first audio.
fn check_stem(stem: &ConvStemWeights, n_embd: usize, n_mel_bins: usize) -> Result<()> {
    for (i, l) in stem.layers.iter().enumerate() {
        ensure!(
            l.shape.len() == 4
                && l.bias.len() == l.shape[3]
                && l.weight.len() == l.shape.iter().product::<usize>(),
            "conv stem layer {i} ({}): weight shape {:?}, {} weights and {} biases disagree",
            l.name,
            l.shape,
            l.weight.len(),
            l.bias.len()
        );
    }
    // `conv_stem_forward` hard-wires NeMo's dw_striding stem (kernel, stride and padding per
    // layer), so the kernel and channel dims must be exactly NeMo's too: anything else panics
    // at first audio or, for a bad kernel height, runs a different front end without a sound.
    check_gemv_dtype("a.pre_encode.out.weight", &stem.pre_encode_out_w)?;
    let out_ch = stem.layers.last().map_or(0, |l| l.shape[3]);
    let c = out_ch;
    let want_shapes = [
        [3, 3, 1, c],
        [3, 3, 1, c],
        [1, 1, c, c],
        [3, 3, 1, c],
        [1, 1, c, c],
    ];
    for (i, (l, want)) in stem.layers.iter().zip(want_shapes).enumerate() {
        ensure!(
            l.shape == want,
            "conv stem layer {i} ({}): weight shape {:?}, expected {want:?}",
            l.name,
            l.shape
        );
    }
    let want_cols = out_ch * stem_frames(n_mel_bins);
    ensure!(
        stem.pre_encode_out_w.rows == n_embd && stem.pre_encode_out_w.cols == want_cols,
        "a.pre_encode.out.weight is {}x{}, expected {n_embd}x{want_cols}",
        stem.pre_encode_out_w.rows,
        stem.pre_encode_out_w.cols
    );
    ensure!(
        stem.pre_encode_out_b.len() == n_embd,
        "a.pre_encode.out.bias has {} values, expected {n_embd}",
        stem.pre_encode_out_b.len()
    );
    Ok(())
}

/// Every vector in `named` must have the length it is paired with: a truncated or hand-edited
/// file otherwise runs with a partial bias in release builds (the kernels only `debug_assert`).
fn check_lens(what: &str, named: &[(&str, usize, usize)]) -> Result<()> {
    for &(name, got, want) in named {
        ensure!(
            got == want,
            "{what}: {name} has {got} values, expected {want}"
        );
    }
    Ok(())
}

/// Shapes the shared Conformer block kernel asserts on, checked once at load.
fn check_encoder_block(
    il: usize,
    l: &ConformerLayerWeights,
    n_embd: usize,
    n_ff: usize,
    kernel: usize,
) -> Result<()> {
    let what = format!("FastConformer block {il}");
    for (name, w) in [
        ("ffn_up", &l.ffn_up_w),
        ("ffn_down", &l.ffn_down_w),
        ("attn_q", &l.attn_q_w),
        ("attn_k", &l.attn_k_w),
        ("attn_v", &l.attn_v_w),
        ("attn_out", &l.attn_o_w),
        ("linear_pos", &l.linear_pos_w),
        ("conv_pw1", &l.conv_pw1_w),
        ("conv_pw2", &l.conv_pw2_w),
        ("ffn_up_1", &l.ffn_up_1_w),
        ("ffn_down_1", &l.ffn_down_1_w),
    ] {
        check_gemv_dtype(&format!("a.blk.{il}.{name}.weight"), w)?;
    }
    let square = |w: &MmapWeight| w.rows == n_embd && w.cols == n_embd;
    ensure!(
        l.ffn_up_w.cols == n_embd
            && l.ffn_up_w.rows == n_ff
            && l.ffn_down_w.rows == n_embd
            && l.ffn_down_w.cols == n_ff
            && l.ffn_up_1_w.rows == n_ff
            && l.ffn_up_1_w.cols == n_embd
            && l.ffn_down_1_w.rows == n_embd
            && l.ffn_down_1_w.cols == n_ff,
        "{what}: FFN shapes disagree with n_embd {n_embd} / n_ff {n_ff}"
    );
    ensure!(
        square(&l.attn_q_w)
            && square(&l.attn_k_w)
            && square(&l.attn_v_w)
            && square(&l.attn_o_w)
            && square(&l.linear_pos_w)
            && square(&l.conv_pw2_w)
            && l.conv_pw1_w.rows == 2 * n_embd
            && l.conv_pw1_w.cols == n_embd,
        "{what}: attention or pointwise-conv shapes disagree with n_embd {n_embd}"
    );
    ensure!(
        matches!(l.conv_dw_shape.len(), 2 | 3) && l.conv_dw_shape.first().copied() == Some(kernel),
        "{what}: depthwise kernel weight shape {:?} is not a rank 2 or 3 weight of sortformer.conv_kernel_size {kernel}",
        l.conv_dw_shape
    );
    check_lens(
        &what,
        &[
            ("conv_dw.weight", l.conv_dw_w.len(), kernel * n_embd),
            ("conv_dw.bias", l.conv_dw_b.len(), n_embd),
            ("ffn_norm.weight", l.ffn_norm_w.len(), n_embd),
            ("ffn_norm.bias", l.ffn_norm_b.len(), n_embd),
            ("ffn_up.bias", l.ffn_up_b.len(), n_ff),
            ("ffn_down.bias", l.ffn_down_b.len(), n_embd),
            ("ln1.weight", l.ln1_w.len(), n_embd),
            ("ln1.bias", l.ln1_b.len(), n_embd),
            ("attn_q.bias", l.attn_q_b.len(), n_embd),
            ("attn_k.bias", l.attn_k_b.len(), n_embd),
            ("attn_v.bias", l.attn_v_b.len(), n_embd),
            ("attn_out.bias", l.attn_o_b.len(), n_embd),
            ("pos_bias_u", l.pos_bias_u.len(), n_embd),
            ("pos_bias_v", l.pos_bias_v.len(), n_embd),
            ("norm_conv.weight", l.norm_conv_w.len(), n_embd),
            ("norm_conv.bias", l.norm_conv_b.len(), n_embd),
            ("conv_pw1.bias", l.conv_pw1_b.len(), 2 * n_embd),
            ("conv_norm.weight", l.conv_norm_w.len(), n_embd),
            ("conv_norm.bias", l.conv_norm_b.len(), n_embd),
            ("conv_pw2.bias", l.conv_pw2_b.len(), n_embd),
            ("ffn_norm_1.weight", l.ffn_norm_1_w.len(), n_embd),
            ("ffn_norm_1.bias", l.ffn_norm_1_b.len(), n_embd),
            ("ffn_up_1.bias", l.ffn_up_1_b.len(), n_ff),
            ("ffn_down_1.bias", l.ffn_down_1_b.len(), n_embd),
            ("ln2.weight", l.ln2_w.len(), n_embd),
            ("ln2.bias", l.ln2_b.len(), n_embd),
        ],
    )
}

fn check_transformer_layer(n: usize, l: &TransformerLayer, d: usize, inner: usize) -> Result<()> {
    let what = format!("transformer layer {n}");
    ensure!(
        [&l.q_w, &l.k_w, &l.v_w, &l.o_w]
            .iter()
            .all(|w| w.rows == d && w.cols == d)
            && l.up_w.rows == inner
            && l.up_w.cols == d
            && l.down_w.rows == d
            && l.down_w.cols == inner,
        "{what}: shapes disagree with d {d} / inner {inner}"
    );
    check_lens(
        &what,
        &[
            ("ln1.weight", l.ln1_w.len(), d),
            ("ln1.bias", l.ln1_b.len(), d),
            ("attn_q.bias", l.q_b.len(), d),
            ("attn_k.bias", l.k_b.len(), d),
            ("attn_v.bias", l.v_b.len(), d),
            ("attn_out.bias", l.o_b.len(), d),
            ("ln2.weight", l.ln2_w.len(), d),
            ("ln2.bias", l.ln2_b.len(), d),
            ("ffn_up.bias", l.up_b.len(), inner),
            ("ffn_down.bias", l.down_b.len(), d),
        ],
    )
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

    /// Load a converted Sortformer GGUF from in-memory bytes.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>) -> Result<Self> {
        let g = Arc::new(GgufFile::from_bytes(bytes.into())?);
        Self::from_gguf(&g)
    }

    /// The loaded weights, for backends that stage them (the Hexagon tail).
    #[cfg(feature = "hexagon")]
    pub(crate) fn weights(&self) -> &SortformerWeights {
        &self.w
    }

    /// Architecture constants.
    pub fn config(&self) -> &SortformerConfig {
        &self.w.config
    }

    /// The FastConformer weights an accelerated encoder stages (stem and blocks; there is no
    /// MLP adapter, the diarizer continues with `encoder_proj`).
    pub fn encoder_parts(&self) -> EncoderParts<'_> {
        EncoderParts {
            config: &self.w.enc_cfg,
            conv_stem: &self.w.conv_stem,
            layers: &self.w.layers,
            adapter: None,
        }
    }

    /// The factor applied to the pre-encode embeddings before the first FastConformer block
    /// (`sqrt(d_model)` when the checkpoint trains with NeMo's `xscaling`, else 1).
    pub fn encoder_input_scale(&self) -> f32 {
        if self.w.config.xscaling {
            (self.w.config.n_embd as f64).sqrt() as f32
        } else {
            1.0
        }
    }

    /// The checkpoint's streaming defaults.
    pub fn default_streaming(&self) -> &StreamingParams {
        &self.w.streaming
    }

    /// The checkpoint's recommended low-latency streaming parameters.
    pub fn low_latency_streaming(&self) -> StreamingParams {
        self.default_streaming().low_latency()
    }

    /// Log-mel features of mono 16 kHz PCM, `[frames × 128]` time-major, with NeMo's
    /// parameters (no per-feature normalization). Returns `(features, frames)`.
    ///
    /// The STFT of `n` samples has `n / 160 + 1` frames; NeMo's sequence length is
    /// `n / 160`, and the last STFT frame (the one that reaches into the trailing center
    /// padding) is masked out of everything downstream. This returns only the valid frames.
    pub fn log_mel(&self, pcm: &[f32]) -> (Vec<f32>, usize) {
        if let Some(done) = self.log_mel_accelerated(pcm) {
            return done;
        }
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

    /// [`Self::log_mel`] on the accelerator, or `None` when there is none, it declines, or the
    /// clip has no frame.
    fn log_mel_accelerated(&self, pcm: &[f32]) -> Option<(Vec<f32>, usize)> {
        self.w.accel.get()?;
        let valid = n_frames_for(pcm.len()).min(pcm.len() / HOP_LEN);
        if valid == 0 {
            return None;
        }
        let samples = padded_preemphasized(pcm)?;
        let want = valid * self.w.config.n_mel_bins;
        let mel = self.w.accelerated_checked(AccelStage::LogMel, want, |a| {
            a.log_mel(&samples[..(valid - 1) * HOP_LEN + N_FFT], valid)
        })?;
        Some((mel, valid))
    }

    /// Conv stem plus `pre_encode.out`: `[frames × 128]` mel to `[ceil(frames/8) × 512]` embeddings.
    /// These are what the speaker cache and FIFO store.
    ///
    /// # Panics
    ///
    /// If `mel` is not `[n_frames x 128]`.
    pub fn pre_encode(&self, mel: &[f32], n_frames: usize) -> (Vec<f32>, usize) {
        assert_eq!(
            mel.len(),
            n_frames * self.w.config.n_mel_bins,
            "pre_encode: mel must be [n_frames x n_mel_bins]"
        );
        self.w.conv_stem_for(mel)
    }

    /// Everything after the stem: x-scale, FastConformer, `encoder_proj`, Transformer and the
    /// speaker head, over `t` pre-encode embeddings (`[t × 512]`). Returns the sigmoid
    /// speaker activities `[t × 4]`.
    ///
    /// # Panics
    ///
    /// If `emb` is not `[t x 512]`, or `t` is past the 7500-frame attention window the other
    /// entry points enforce (attention memory is quadratic in `t`).
    pub fn predict(&self, emb: &[f32], t: usize) -> Vec<f32> {
        let c = &self.w.config;
        assert_eq!(emb.len(), t * c.n_embd, "predict: emb must be [t x n_embd]");
        assert!(
            t <= MAX_OFFLINE_FRAMES,
            "predict: {t} frames exceeds the {MAX_OFFLINE_FRAMES}-frame attention window"
        );
        let want = t * c.n_spk;
        if t > 0
            && let Some(preds) = self
                .w
                .accelerated_checked(AccelStage::Predict, want, |a| a.predict(emb, t))
        {
            return preds;
        }
        self.predict_with_taps(emb, t, &mut |_, _| {})
    }

    /// [`Self::predict`] on the CPU whatever accelerator is set: the reference the accelerated
    /// path is compared against.
    pub fn predict_cpu(&self, emb: &[f32], t: usize) -> Vec<f32> {
        self.predict_with_taps(emb, t, &mut |_, _| {})
    }

    /// Run the stem and the prediction on `accel` from here on (one accelerator per model,
    /// shared by every stream and live diarizer made from it, including clones).
    pub fn set_accelerator(&self, accel: Arc<dyn SortformerAccelerator>) -> Result<()> {
        self.w
            .accel
            .set(accel)
            .map_err(|_| anyhow::anyhow!("this Sortformer model already has an accelerator"))
    }

    /// Whether [`Self::set_accelerator`] already installed an accelerator.
    pub fn has_accelerator(&self) -> bool {
        self.w.accel.get().is_some()
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
        assert!(
            t <= MAX_OFFLINE_FRAMES,
            "predict: {t} frames exceeds the {MAX_OFFLINE_FRAMES}-frame attention window"
        );
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
    ///
    /// Attention memory and time grow with the square of the clip length (a 2-minute clip took
    /// about 33 s on an M1 Max, and the 7500-frame limit would take on the order of 14 minutes),
    /// so this is for clips of minutes, not hours; use [`Self::new_live`] for long audio. Peak
    /// memory at the limit is about 3.5 GB (the conv stem, about 58 KB per mel frame) plus 1.8 GB
    /// of attention scores per block.
    ///
    /// Fails if a sample is NaN or infinite, which would otherwise turn every prediction to NaN.
    pub fn diarize_offline(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        ensure_finite_pcm(pcm)?;
        ensure_offline_len(pcm.len())?;
        let (mel, n) = self.log_mel(pcm);
        let (emb, t) = self.pre_encode(&mel, n);
        Ok(self.predict(&emb, t))
    }

    /// Start a streaming session with `params` (see [`Self::default_streaming`]).
    pub fn new_stream(&self, params: StreamingParams) -> Result<SortformerStream> {
        params.validate()?;
        Ok(SortformerStream {
            w: self.w.clone(),
            params,
            state: StreamState::new(self.w.config.n_embd),
        })
    }

    /// Streaming diarization of a whole clip, chunked exactly as NeMo's feature loader
    /// chunks it. Returns `[frames × 4]`.
    pub fn diarize_streaming(&self, pcm: &[f32], params: StreamingParams) -> Result<Vec<f32>> {
        ensure_finite_pcm(pcm)?;
        let (mel, n) = self.log_mel(pcm);
        self.new_stream(params)?.diarize_features(&mel, n)
    }
}

// ── Live audio ─────────────────────────────────────────────────────────────

/// Refuse a clip the offline pass cannot attend over, from its length alone (before any mel
/// or attention memory is allocated).
fn ensure_offline_len(n_samples: usize) -> Result<()> {
    let frames = stem_frames(n_samples / HOP_LEN);
    ensure!(
        frames <= MAX_OFFLINE_FRAMES,
        "{frames} encoder frames is past the {MAX_OFFLINE_FRAMES} (10 minutes) the offline \
         pass can attend over; use new_live or diarize_streaming for long audio"
    );
    Ok(())
}

/// Largest PCM magnitude accepted. Real audio is within +-1 (or +-32768 at 16-bit scale); the
/// mel overflows to infinity only near `f32::MAX`, so this leaves room for any real signal and
/// refuses reinterpreted garbage bytes deterministically.
const MAX_ABS_PCM: f32 = 1e9;

/// Largest |log-mel| accepted by the feature entry points. Real values stay within about
/// [-17, 46] even for PCM at [`MAX_ABS_PCM`]; the stem overflows to NaN from about 1e9.
const MAX_ABS_MEL: f32 = 1e3;

/// Index of the first mel value that is NaN, infinite or beyond [`MAX_ABS_MEL`].
fn first_bad_mel(mel: &[f32]) -> Option<usize> {
    mel.iter()
        .position(|x| !x.is_finite() || x.abs() > MAX_ABS_MEL)
}

/// Index of the first sample that is NaN, infinite or beyond [`MAX_ABS_PCM`].
fn first_bad_sample(pcm: &[f32]) -> Option<usize> {
    pcm.iter()
        .position(|x| !x.is_finite() || x.abs() > MAX_ABS_PCM)
}

/// One NaN, infinite or absurdly large sample reaches every later frame through the FFT and the
/// carried state (the mel overflows to infinity), so every entry point that takes PCM refuses it
/// up front.
fn ensure_finite_pcm(pcm: &[f32]) -> Result<()> {
    if let Some(i) = first_bad_sample(pcm) {
        anyhow::bail!("non-finite or out-of-range PCM sample at index {i}");
    }
    Ok(())
}

/// Incremental NeMo log-mel: feed PCM in pieces of any size, get the same mel frames the
/// whole-clip [`SortformerModel::log_mel`] computes, bit for bit.
///
/// A frame needs 512 samples centered on its hop position, so a frame is ready once 256
/// samples past its center have arrived; the clip's last frames need the trailing center
/// padding, which [`Self::finish`] supplies. Pre-emphasis carries the previous raw sample
/// across calls (the offline path leaves the first sample untouched, which is the same thing
/// with a previous sample of 0).
pub struct MelStream {
    /// The model's weights, for its accelerator (absent for the weight-free test streams).
    weights: Option<Arc<SortformerWeights>>,
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
    fn new(w: &Arc<SortformerWeights>) -> Self {
        let mut stream = Self::from_tables(w.config.n_mel_bins, &w.window, &w.mel_fb);
        stream.weights = Some(Arc::clone(w));
        stream
    }

    /// A front end over the given window (`N_FFT` long) and filterbank; weight-free, so the
    /// hermetic tests can drive it with synthetic tables.
    fn from_tables(n_mel_bins: usize, window: &[f32], mel_fb: &[f32]) -> Self {
        Self {
            weights: None,
            computer: MelFrameComputer::new(n_mel_bins, window, mel_fb),
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

    /// Samples currently held (the part of the signal the next frames still need). It stays
    /// within one FFT window plus one hop however long the stream runs.
    pub fn buffered_samples(&self) -> usize {
        self.buf.len()
    }

    /// Append mono 16 kHz PCM. Returns the newly completed mel frames, `[k x 128]` time-major.
    /// Fails, consuming nothing, if the stream has finished or a sample is NaN or infinite.
    pub fn push(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        self.push_samples(pcm)?;
        Ok(self.emit())
    }

    /// [`Self::push`] without computing any frame: the samples are pre-emphasised and held, and
    /// [`Self::ready_frames`] / [`Self::compute_ready`] take the frames later, in one batch.
    fn push_samples(&mut self, pcm: &[f32]) -> Result<()> {
        ensure!(!self.finished, "MelStream: push after finish");
        // One NaN would ride through the FFT into every later frame's attention, so refuse the
        // whole piece (nothing is consumed) instead of poisoning the stream.
        if let Some(i) = first_bad_sample(pcm) {
            anyhow::bail!(
                "MelStream: non-finite or out-of-range PCM sample at stream offset {}",
                self.n_in + i
            );
        }
        self.buf.reserve(pcm.len());
        for &x in pcm {
            // The first sample passes through unchanged because `prev_raw` starts at 0.
            let y = x - PREEMPH * self.prev_raw;
            self.buf.push(y);
            self.prev_raw = x;
            self.n_in += 1;
        }
        Ok(())
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

    /// Frames that can be computed from the samples held now.
    fn ready_frames(&self) -> usize {
        // NeMo's length is n / hop: the STFT's extra last frame is masked, never produced.
        let by_length = self.n_in / HOP_LEN;
        let held = self.base + self.buf.len();
        let by_samples = if held >= N_FFT {
            (held - N_FFT) / HOP_LEN + 1
        } else {
            0
        };
        by_length.min(by_samples).saturating_sub(self.next_frame)
    }

    fn emit(&mut self) -> Vec<f32> {
        self.compute_ready()
    }

    /// Compute every ready frame, in one accelerator batch when the model has an accelerator
    /// that takes it, else one by one on the CPU.
    fn compute_ready(&mut self) -> Vec<f32> {
        let k = self.ready_frames();
        let mut out = Vec::new();
        if k > 0 {
            let lo = self.next_frame * HOP_LEN - self.base;
            let span = (k - 1) * HOP_LEN + N_FFT;
            let want = k * self.n_mel_bins;
            let logged = self.weights.as_ref().and_then(|w| {
                w.accelerated_checked(AccelStage::LogMel, want, |a| {
                    a.log_mel(&self.buf[lo..lo + span], k)
                })
            });
            match logged {
                Some(e) => out.extend_from_slice(&e),
                None => {
                    // Declined or faulty (a wrong-length `Some` warns inside
                    // `accelerated_checked`): compute the frames on the CPU.
                    let mut row = vec![0.0f32; self.n_mel_bins];
                    for f in 0..k {
                        let at = lo + f * HOP_LEN;
                        self.computer.frame(&self.buf[at..at + N_FFT], &mut row);
                        out.extend_from_slice(&row);
                    }
                }
            }
            self.next_frame += k;
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
    /// preset's latency), on top of the 16 ms (256 samples) the mel front end needs after a
    /// frame's center.
    pub fn latency_frames(&self) -> usize {
        // Saturating: the params may be hand-made (`with_chunking` skips validation).
        self.stream
            .params
            .chunk_len
            .saturating_add(self.stream.params.right_context)
    }

    /// Prediction frames returned so far (80 ms each).
    pub fn frames_emitted(&self) -> usize {
        self.emitted
    }

    /// Mel frames currently held for the next chunks (their left context plus whatever is
    /// waiting for lookahead). Bounded by the streaming parameters, not by the stream's length.
    pub fn buffered_frames(&self) -> usize {
        self.rows.len() / self.stream.w.config.n_mel_bins
    }

    /// The underlying streaming state (speaker cache, FIFO, silence profile).
    pub fn stream(&self) -> &SortformerStream {
        &self.stream
    }

    /// Feed mono 16 kHz PCM of any length. Returns the predictions that became final,
    /// `[k x 4]` for the next `k` frames in order (possibly empty). Fails if the stream has
    /// finished or a sample is NaN, infinite or beyond 1e9 in magnitude (nothing is consumed in
    /// that case).
    ///
    /// When the call completes a chunk it runs the model synchronously, and on the CPU that can
    /// take longer than the audio it covers: with Q4_0 on an M1 Max a 390-frame window (the
    /// card's 0.48 s low-latency preset) costs about 2.7 s per step, while the 15 s default chunk
    /// costs about 2.7 s per 15 s. Cost grows with the square of the window (cache, FIFO, chunk
    /// and contexts together), so size a worker for that, not for the audio rate, and never
    /// call this from an audio callback.
    pub fn push_audio(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        ensure!(!self.finished, "SortformerLive: push_audio after finish");
        if self.stream.w.accel.get().is_some() {
            // With an accelerator the mel is computed a chunk at a time: a batch per chunk
            // instead of one tiny accelerator call per push, which would cost the host more than
            // the FFTs it replaces. The result is the same frames, computed when a chunk needs
            // them.
            self.mel.push_samples(pcm)?;
            let ss = self.stream.w.config.subsampling;
            let needed =
                self.stt + (self.stream.params.chunk_len + self.stream.params.right_context) * ss;
            if self.mel.frames() + self.mel.ready_frames() >= needed {
                let rows = self.mel.compute_ready();
                self.accept(&rows);
            }
        } else {
            let rows = self.mel.push(pcm)?;
            self.accept(&rows);
        }
        self.drain(false)
    }

    /// End of audio: flush the remaining frames with whatever lookahead exists.
    ///
    /// With validated parameters and bounded PCM a model step cannot fail; if one ever does, the
    /// predictions already computed in that call are not returned and the stream should be
    /// discarded.
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
            let preds = self.stream.step(
                &self.rows[lo..lo + n_feat * nm],
                n_feat,
                n_feat,
                left_offset,
                right_offset,
            )?;
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
    debug_assert!(dh <= 64, "head dimension {dh} exceeds accumulator size 64");
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
            let mut acc = [0.0f64; 64];
            for j in 0..t {
                let s = scores[j] as f64;
                let v_row = &v[j * d + off..j * d + off + dh];
                for dd in 0..dh {
                    acc[dd] += s * v_row[dd] as f64;
                }
            }
            for dd in 0..dh {
                ctx[i * d + off + dd] = acc[dd] as f32;
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
    state: StreamState,
}

/// The cross-step state of a stream and NeMo's `streaming_update` over it. It owns no weights,
/// so the cache, FIFO and silence-profile logic can be exercised with synthetic embeddings.
struct StreamState {
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
        &self.state.spkcache
    }

    /// Speaker-cache predictions `[len × n_spk]`, or `None` before the first compression.
    pub fn spkcache_preds(&self) -> Option<&[f32]> {
        self.state.spkcache_preds.as_deref()
    }

    /// FIFO embeddings, `[len × n_embd]`.
    pub fn fifo(&self) -> &[f32] {
        &self.state.fifo
    }

    /// FIFO predictions, `[len × n_spk]`.
    pub fn fifo_preds(&self) -> &[f32] {
        &self.state.fifo_preds
    }

    /// Running mean of the embeddings of frames classified as silence.
    pub fn mean_sil_emb(&self) -> &[f32] {
        &self.state.mean_sil_emb
    }

    /// How many silence frames fed [`Self::mean_sil_emb`].
    pub fn n_sil_frames(&self) -> usize {
        self.state.n_sil_frames
    }

    /// Run a whole clip's features (`[n_frames × 128]`, from [`SortformerModel::log_mel`])
    /// through the streaming loop, chunked like NeMo's `streaming_feat_loader` over features
    /// padded to a multiple of `pad_to`. Returns `[frames × 4]`; frames past the audio are zeros.
    ///
    /// The stream keeps its speaker cache, FIFO and silence profile between calls, so a second
    /// clip continues from the first; start a new stream ([`SortformerModel::new_stream`]) per
    /// clip.
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
        let pad_to = self.w.config.pad_to;
        let feat_len = n_frames.checked_next_multiple_of(pad_to).with_context(|| {
            format!("n_frames {n_frames} cannot be padded to a multiple of {pad_to}")
        })?;
        self.chunk_loop(mel, n_frames, feat_len, on_step)
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
        ensure!(
            n_frames.checked_mul(nm) == Some(mel.len()),
            "mel is not [n_frames x {nm}]"
        );
        // Every chunk is checked again in `step`, but a bad value in a later chunk would only
        // surface after earlier chunks had already changed the stream: refuse the clip up front.
        if let Some(i) = first_bad_mel(mel) {
            anyhow::bail!("non-finite or out-of-range mel value at frame {}", i / nm);
        }
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
            if valid > 0 {
                // With `valid == 0` the chunk starts in the padding past the last real frame.
                chunk[..valid * nm].copy_from_slice(&mel[first * nm..(first + valid) * nm]);
            }
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
        // `checked_mul`: a wrapped product would let an empty `feats` pass for a huge `n_feat`.
        ensure!(
            n_feat.checked_mul(c.n_mel_bins) == Some(feats.len()),
            "feats is not [n_feat x {}]",
            c.n_mel_bins
        );
        ensure!(
            valid_feat <= n_feat,
            "valid_feat {valid_feat} > n_feat {n_feat}"
        );
        // Checked before any state is touched: a NaN in the FIFO or cache would taint every
        // later step.
        if let Some(i) = first_bad_mel(&feats[..valid_feat * c.n_mel_bins]) {
            anyhow::bail!(
                "step: non-finite or out-of-range mel value at frame {}",
                i / c.n_mel_bins
            );
        }

        // Frames NeMo would see (padding included) and the ones that exist here.
        let enc_total = stem_frames(n_feat);
        ensure!(
            left_offset.is_multiple_of(ss),
            "left_offset {left_offset} is not a whole number of encoder frames ({ss} mel frames each)"
        );
        let lc = left_offset / ss;
        let rc = right_offset.div_ceil(ss);
        ensure!(
            enc_total >= lc + rc,
            "chunk of {enc_total} encoder frames is shorter than its contexts ({lc} + {rc})"
        );
        let chunk_len = enc_total - lc - rc;
        // The same window bound `StreamingParams::validate` puts on the configuration, checked
        // against what this call really attends over (the caller chooses `n_feat`).
        let window = (self.state.spkcache.len() + self.state.fifo.len()) / d + enc_total;
        ensure!(
            window <= MAX_OFFLINE_FRAMES,
            "step window of {window} encoder frames exceeds {MAX_OFFLINE_FRAMES}"
        );
        // `validate` related `max_index` to the configured chunk; this call may be longer, and
        // a real flat index at or past `max_index` would read as a disabled cache slot. Only
        // the rows `update` can compress count (cache, FIFO and the chunk itself, not the
        // contexts), the same terms `validate` uses.
        let candidates = (self.state.spkcache.len() + self.state.fifo.len()) / d + chunk_len;
        ensure!(
            self.params.max_index >= MAX_SPEAKERS * (candidates + self.params.sil_frames_per_spk),
            "max_index {} lies inside the flat index range of a {candidates}-frame step",
            self.params.max_index
        );

        let (chunk_valid, enc_valid) = if valid_feat == 0 {
            (Vec::new(), 0)
        } else {
            w.conv_stem_for(&feats[..valid_feat * c.n_mel_bins])
        };
        debug_assert_eq!(enc_valid, stem_frames(valid_feat));
        let model = SortformerModel { w: w.clone() };

        let n_sc = self.state.spkcache.len() / d;
        let n_fifo = self.state.fifo.len() / d;
        let mut concat = Vec::with_capacity((n_sc + n_fifo + enc_valid) * d);
        concat.extend_from_slice(&self.state.spkcache);
        concat.extend_from_slice(&self.state.fifo);
        concat.extend_from_slice(&chunk_valid);
        let valid_rows = n_sc + n_fifo + enc_valid;
        let mut preds = model.predict(&concat, valid_rows);
        // Pad frames get zero predictions and zero embeddings, like NeMo's masked output.
        let total_rows = n_sc + n_fifo + enc_total;
        preds.resize(total_rows * s, 0.0);
        let mut chunk_emb = chunk_valid;
        chunk_emb.resize(enc_total * d, 0.0);

        Ok(self.state.update(
            &self.params,
            (d, s),
            &preds,
            &chunk_emb,
            (n_sc, n_fifo, lc, chunk_len),
        ))
    }
}

impl StreamState {
    fn new(n_embd: usize) -> Self {
        Self {
            spkcache: Vec::new(),
            spkcache_preds: None,
            fifo: Vec::new(),
            fifo_preds: Vec::new(),
            mean_sil_emb: vec![0.0; n_embd],
            n_sil_frames: 0,
        }
    }

    /// NeMo's `streaming_update` (synchronous path) after a forward over
    /// `[spkcache, fifo, chunk]`. `preds` is that forward's `[(n_sc + n_fifo + chunk rows) x s]`
    /// sigmoid output and `chunk_emb` the chunk's pre-encode embeddings; `rows` is
    /// `(n_sc, n_fifo, lc, chunk_len)`, where `lc` is the left-context frames at the front of the
    /// chunk. Appends the chunk to the FIFO, pops into the cache when the FIFO overflows
    /// (folding silent frames into the silence profile) and compresses the cache when it does.
    /// Returns the chunk's own predictions, `[chunk_len x s]`.
    fn update(
        &mut self,
        p: &StreamingParams,
        (d, s): (usize, usize),
        preds: &[f32],
        chunk_emb: &[f32],
        (n_sc, n_fifo, lc, chunk_len): (usize, usize, usize, usize),
    ) -> Vec<f32> {
        let (fifo_cap, update_period, cache_cap) = (p.fifo_len, p.update_period, p.spkcache_len);
        let base = n_sc + n_fifo + lc;
        // The forward covers [spkcache, fifo, chunk]: every slice below lands inside it.
        debug_assert!((base + chunk_len) * s <= preds.len());
        debug_assert!((lc + chunk_len) * d <= chunk_emb.len());
        let chunk_slice = &chunk_emb[lc * d..(lc + chunk_len) * d];
        let chunk_preds = preds[base * s..(base + chunk_len) * s].to_vec();

        self.fifo.extend_from_slice(chunk_slice);
        self.fifo_preds.clear();
        self.fifo_preds
            .extend_from_slice(&preds[n_sc * s..(n_sc + n_fifo) * s]);
        self.fifo_preds.extend_from_slice(&chunk_preds);

        if n_fifo + chunk_len > fifo_cap {
            let pop = update_period
                .max((chunk_len + n_fifo).saturating_sub(fifo_cap))
                .min(n_fifo + chunk_len);

            let pop_embs = &self.fifo[..pop * d];
            let pop_preds = &self.fifo_preds[..pop * s];
            Self::update_silence_profile_fields(
                &mut self.mean_sil_emb,
                &mut self.n_sil_frames,
                p,
                (d, s),
                pop_embs,
                pop_preds,
                pop,
            );

            self.spkcache.extend_from_slice(pop_embs);
            if let Some(sp) = self.spkcache_preds.as_mut() {
                sp.extend_from_slice(pop_preds);
            }
            let cache_rows = self.spkcache.len() / d;
            if cache_rows > cache_cap {
                if self.spkcache_preds.is_none() {
                    // First compression: the cache's own predictions come from this step's forward.
                    let mut sp = preds[..n_sc * s].to_vec();
                    sp.extend_from_slice(pop_preds);
                    self.spkcache_preds = Some(sp);
                }
                // Always `Some` here (set just above when missing, or already present);
                // skip the compression rather than panic.
                if let Some(preds_ref) = self.spkcache_preds.as_ref() {
                    let (emb, pr) = compress_spkcache(
                        p,
                        s,
                        d,
                        &self.spkcache,
                        preds_ref,
                        cache_rows,
                        &self.mean_sil_emb,
                    );
                    self.spkcache = emb;
                    self.spkcache_preds = Some(pr);
                }
            }
            self.fifo.drain(..pop * d);
            self.fifo_preds.drain(..pop * s);
        }
        chunk_preds
    }

    /// NeMo `_get_silence_profile`: fold the silent frames of `embs` into the running mean.
    #[cfg(test)]
    fn update_silence_profile(
        &mut self,
        p: &StreamingParams,
        (d, s): (usize, usize),
        embs: &[f32],
        preds: &[f32],
        n: usize,
    ) {
        Self::update_silence_profile_fields(
            &mut self.mean_sil_emb,
            &mut self.n_sil_frames,
            p,
            (d, s),
            embs,
            preds,
            n,
        );
    }

    fn update_silence_profile_fields(
        mean_sil_emb: &mut [f32],
        n_sil_frames: &mut usize,
        p: &StreamingParams,
        (d, s): (usize, usize),
        embs: &[f32],
        preds: &[f32],
        n: usize,
    ) {
        let mut sum = vec![0.0f32; d];
        let mut count = 0usize;
        for f in 0..n {
            let total: f32 = preds[f * s..(f + 1) * s].iter().sum();
            if total < p.sil_threshold {
                count += 1;
                for (a, b) in sum.iter_mut().zip(&embs[f * d..(f + 1) * d]) {
                    *a += *b;
                }
            }
        }
        if count == 0 {
            return;
        }
        let upd = *n_sil_frames + count;
        let denom = upd as f32;
        for (m, add) in mean_sil_emb.iter_mut().zip(&sum) {
            *m = (*m * *n_sil_frames as f32 + add) / denom;
        }
        *n_sil_frames = upd;
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
    let half_ln = (0.5f64).ln() as f32;
    for f in 0..n {
        let row = &preds[f * s..(f + 1) * s];
        let mut log_1p = [0.0f32; MAX_SPEAKERS];
        let mut log_1p_sum = 0.0f32;
        for k in 0..s {
            let val = (1.0 - row[k]).max(p.pred_score_threshold).ln();
            log_1p[k] = val;
            log_1p_sum += val;
        }
        for k in 0..s {
            let log_p = row[k].max(p.pred_score_threshold).ln();
            scores[f * s + k] = log_p - log_1p[k] + log_1p_sum - half_ln;
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
    /// The stem on an already-validated slice of mel frames: on the accelerator when one is
    /// set and takes it, else on the CPU.
    fn conv_stem_for(&self, mel: &[f32]) -> (Vec<f32>, usize) {
        let n = mel.len() / self.config.n_mel_bins;
        let want = stem_frames(n) * self.config.n_embd;
        if let Some(emb) =
            self.accelerated_checked(AccelStage::Stem, want, |a| a.pre_encode(mel, n))
        {
            return (emb, stem_frames(n));
        }
        conv_stem_forward(mel, n, &self.conv_stem, &self.enc_cfg)
    }

    /// Run `call` on the accelerator if one is set. `None` means "use the CPU": nothing is set,
    /// the accelerator declined, or it failed (logged once per stage, then quiet).
    fn accelerated<T>(
        &self,
        stage: AccelStage,
        call: impl FnOnce(&dyn SortformerAccelerator) -> Result<Option<T>>,
    ) -> Option<T> {
        let accel = self.accel.get()?;
        match call(accel.as_ref()) {
            Ok(out) => out,
            Err(e) => {
                self.warn_once(
                    stage,
                    &format!("failed on the accelerator ({e:#}); using the CPU"),
                );
                None
            }
        }
    }

    /// Run `call` on the accelerator and take its output only at the expected length. A
    /// wrong-length `Some` is a faulty stage, not a decline: warn once, then `None` so the
    /// caller falls back to the CPU. Every `Vec<f32>` stage output goes through here so a
    /// future stage cannot forget the length check and consume a short output as valid.
    fn accelerated_checked(
        &self,
        stage: AccelStage,
        want: usize,
        call: impl FnOnce(&dyn SortformerAccelerator) -> Result<Option<Vec<f32>>>,
    ) -> Option<Vec<f32>> {
        let out = self.accelerated(stage, call)?;
        if out.len() == want {
            return Some(out);
        }
        self.warn_once(
            stage,
            &format!("returned {} values, want {want}; using the CPU", out.len()),
        );
        None
    }

    /// Log a stage's accelerator fault once (later faults of the same stage stay quiet).
    fn warn_once(&self, stage: AccelStage, detail: &str) {
        if !stage
            .warned(&self.accel_warned)
            .swap(true, Ordering::Relaxed)
        {
            tracing::warn!("sortformer: {} {detail}", stage.label());
            // No `tracing` subscriber on the shipping mobile/FFI platforms; without this
            // the warning is invisible exactly where the fallback runs.
            eprintln!("cera-sortformer: {} {detail}", stage.label());
        }
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

    /// Like [`compress`] with the given parameters.
    fn compress_with(p: &StreamingParams, preds: &[[f32; 4]]) -> Vec<f32> {
        let n = preds.len();
        let emb: Vec<f32> = (0..n).map(|f| (f + 1) as f32).collect();
        let flat: Vec<f32> = preds.iter().flatten().copied().collect();
        compress_spkcache(p, 4, 1, &emb, &flat, n, &[-1.0]).0
    }

    #[test]
    fn newest_frames_get_the_latest_boost_once_the_cache_overflows() {
        // 20 identical frames of one speaker, room for 12 speech frames (16 slots, 4 closing
        // silences). Without a boost ties go to the oldest frames; with one, the 4 frames past
        // `spkcache_len` (embeddings 17..=20) displace the middle of the cache.
        let preds = [[0.9, 0.0, 0.0, 0.0]; 20];
        let plain = StreamingParams {
            scores_boost_latest: 0.0,
            ..params(16)
        };
        let boosted = StreamingParams {
            scores_boost_latest: 1.0,
            ..params(16)
        };
        let keep = |p: &StreamingParams| {
            let mut e = compress_with(p, &preds);
            e.retain(|&x| x > 0.0);
            e
        };
        assert_eq!(keep(&plain), (1..=12).map(|x| x as f32).collect::<Vec<_>>());
        assert_eq!(
            keep(&boosted),
            [1., 2., 3., 4., 5., 6., 7., 8., 17., 18., 19., 20.]
        );
    }

    #[test]
    fn the_strong_boost_is_twice_the_weak_one() {
        // 16 slots = 3 speech frames per speaker, so the top 2 scores of each speaker get the
        // strong boost (2 x ln 2) and the top 4 the weak one (1 x ln 2) on top. Frame 4 of
        // speaker 0 (embedding 5) sits just below the cut when the strong boost is 2 x ln 2,
        // and above it when the strong boost is only 1 x ln 2: this layout was found by search
        // against NeMo's scoring and pins the 2:1 ratio.
        let preds = [
            [0.93, 0.76, 0.0, 0.87],
            [0.0, 0.6, 0.0, 0.82],
            [0.58, 0.57, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.75],
            [0.92, 0.72, 0.0, 0.63],
            [0.62, 0.7, 0.61, 0.79],
            [0.92, 0.73, 0.65, 0.68],
            [0.0, 0.0, 0.8, 0.55],
            [0.61, 0.0, 0.0, 0.89],
            [0.8, 0.0, 0.0, 0.9],
            [0.0, 0.0, 0.0, 0.94],
            [0.97, 0.74, 0.0, 0.0],
        ];
        let got = compress_with(&params(16), &preds);
        assert_eq!(
            got,
            [
                3., 9., 10., 12., -1., 2., 3., 5., 12., -1., 6., 8., -1., 4., 11., -1.
            ]
        );
    }

    /// A state with embedding width 1 (frame embeddings are markers) and 4 speakers.
    fn state() -> StreamState {
        StreamState::new(1)
    }

    /// Run `update` for a chunk of `chunk_len` frames with no left context. `rows` is the
    /// predictions of the whole `[cache, fifo, chunk]` forward, one marker value per row.
    fn feed(
        st: &mut StreamState,
        p: &StreamingParams,
        chunk_len: usize,
        first_marker: f32,
        row: [f32; 4],
    ) -> Vec<f32> {
        let (n_sc, n_fifo) = (st.spkcache.len(), st.fifo.len());
        let preds: Vec<f32> = (0..n_sc + n_fifo + chunk_len).flat_map(|_| row).collect();
        let chunk_emb: Vec<f32> = (0..chunk_len).map(|i| first_marker + i as f32).collect();
        st.update(p, (1, 4), &preds, &chunk_emb, (n_sc, n_fifo, 0, chunk_len))
    }

    #[test]
    fn the_fifo_pops_enough_to_fit_and_at_least_one_update_period() {
        let p = StreamingParams {
            fifo_len: 4,
            update_period: 2,
            ..params(40)
        };
        let mut st = state();
        // 6 new frames overflow a FIFO of 4 by 2, which is exactly one update period.
        let out = feed(&mut st, &p, 6, 1.0, [0.9, 0.0, 0.0, 0.0]);
        assert_eq!(out.len(), 6 * 4);
        assert_eq!(st.fifo, [3., 4., 5., 6.]);
        assert_eq!(st.spkcache, [1., 2.]);
        // 4 queued + 6 new overflow by 6, more than an update period: all 6 move.
        feed(&mut st, &p, 6, 7.0, [0.9, 0.0, 0.0, 0.0]);
        assert_eq!(st.fifo.len(), 4);
        assert_eq!(st.spkcache.len(), 2 + 6);
        assert_eq!(st.fifo_preds.len(), 4 * 4);
        // A chunk that fits pops nothing.
        let p = StreamingParams { fifo_len: 100, ..p };
        let mut st = state();
        feed(&mut st, &p, 6, 1.0, [0.9, 0.0, 0.0, 0.0]);
        assert_eq!((st.fifo.len(), st.spkcache.len()), (6, 0));
    }

    #[test]
    fn update_returns_the_chunk_rows_between_its_contexts() {
        // 1 frame of left context and 2 of right: the chunk's own rows are the 3 in the middle.
        let p = params(40);
        let mut st = state();
        let (lc, chunk_len, rc) = (1, 3, 2);
        let rows = lc + chunk_len + rc;
        let preds: Vec<f32> = (0..rows * 4).map(|i| i as f32).collect();
        let chunk_emb: Vec<f32> = (0..rows).map(|i| i as f32).collect();
        let out = st.update(&p, (1, 4), &preds, &chunk_emb, (0, 0, lc, chunk_len));
        assert_eq!(out, (4..16).map(|i| i as f32).collect::<Vec<_>>());
        // The FIFO took the same middle rows, not the contexts.
        assert_eq!(st.fifo, [1., 2., 3.]);
    }

    #[test]
    fn the_silence_profile_averages_only_the_silent_frames() {
        let p = params(24); // sil_threshold 0.2
        let mut st = state();
        let rows = [
            [0.01, 0.0, 0.0, 0.0],  // silent
            [0.9, 0.0, 0.0, 0.0],   // speech
            [0.0, 0.05, 0.05, 0.0], // silent (sums to 0.1)
        ];
        let preds: Vec<f32> = rows.iter().flatten().copied().collect();
        st.update_silence_profile(&p, (1, 4), &[2.0, 100.0, 6.0], &preds, 3);
        assert_eq!((st.n_sil_frames, st.mean_sil_emb.clone()), (2, vec![4.0]));
        // The mean is running: 3 silent frames in all, (4 * 2 + 10) / 3.
        st.update_silence_profile(&p, (1, 4), &[10.0], &[0.0; 4], 1);
        assert_eq!((st.n_sil_frames, st.mean_sil_emb.clone()), (3, vec![6.0]));
        // A frame at exactly the threshold is speech.
        st.update_silence_profile(&p, (1, 4), &[1000.0], &[0.2, 0.0, 0.0, 0.0], 1);
        assert_eq!(st.n_sil_frames, 3);
    }

    #[test]
    fn a_pop_with_no_silent_frame_leaves_the_profile_untouched() {
        // The very first pop has no silence: the running mean must not become 0 / 0.
        let p = params(24);
        let mut st = state();
        let preds = [0.9, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0];
        st.update_silence_profile(&p, (1, 4), &[5.0, 7.0], &preds, 2);
        assert_eq!((st.n_sil_frames, st.mean_sil_emb.clone()), (0, vec![0.0]));
    }

    #[test]
    fn the_first_compression_takes_the_cache_predictions_from_the_forward() {
        // No FIFO: every chunk goes straight to the cache. 20 frames overflow a 16-slot cache.
        let p = StreamingParams {
            fifo_len: 0,
            update_period: 20,
            ..params(16)
        };
        let mut st = state();
        assert!(st.spkcache_preds.is_none());
        feed(&mut st, &p, 20, 1.0, [0.9, 0.0, 0.0, 0.0]);
        assert_eq!(st.spkcache.len(), 16);
        let sp = st.spkcache_preds.as_ref().expect("compressed");
        assert_eq!(sp.len(), 16 * 4);
        // Real frames keep their predictions, the closing silence slots predict nothing.
        assert_eq!(&sp[..4], &[0.9, 0.0, 0.0, 0.0]);
        assert_eq!(&sp[sp.len() - 4..], &[0.0; 4]);
    }

    #[test]
    fn streaming_params_reject_absurd_values() {
        let ok = params(24);
        assert!(ok.validate().is_ok());
        let zero_threshold = StreamingParams {
            pred_score_threshold: 0.0,
            ..ok.clone()
        };
        assert!(zero_threshold.validate().is_err());
        for bad in [-0.1, 1.5, f32::NAN] {
            let p = StreamingParams {
                pred_score_threshold: bad,
                ..ok.clone()
            };
            assert!(p.validate().is_err(), "threshold {bad}");
        }
        // A `max_index` inside the flat index range would alias real cache frames: with 24
        // cache slots, 188 FIFO, 6 chunk and 1 silence it must be at least 4 * 219.
        let aliased = StreamingParams {
            max_index: 5,
            ..ok.clone()
        };
        assert!(aliased.validate().is_err());
        let tight = StreamingParams {
            max_index: 4 * (24 + 188 + 6 + 1),
            ..ok.clone()
        };
        assert!(tight.validate().is_ok());
        let below = StreamingParams {
            max_index: 4 * (24 + 188 + 6 + 1) - 1,
            ..ok.clone()
        };
        assert!(below.validate().is_err());
        let huge = ok.with_chunking(usize::MAX, 1, 1, 1, 24, 1);
        assert!(huge.validate().is_err());
        let huge_context = ok.with_chunking(6, usize::MAX / 2, 1, 1, 24, 1);
        assert!(huge_context.validate().is_err());
    }

    #[test]
    fn the_fifo_pop_size_has_a_floor_a_ceiling_and_an_exact_fit() {
        let hot = [0.9f32, 0.0, 0.0, 0.0];
        // 6 new frames overflow a FIFO of 4 by 2, below the period of 4: the whole period moves.
        let p = StreamingParams {
            fifo_len: 4,
            update_period: 4,
            ..params(40)
        };
        let mut st = state();
        feed(&mut st, &p, 6, 1.0, hot);
        assert_eq!(
            (st.spkcache.clone(), st.fifo.clone()),
            (vec![1., 2., 3., 4.], vec![5., 6.])
        );
        // A period of 20 exceeds the 6 frames queued: everything moves and nothing slices past it.
        let p = StreamingParams {
            fifo_len: 4,
            update_period: 20,
            ..params(40)
        };
        let mut st = state();
        feed(&mut st, &p, 6, 1.0, hot);
        assert_eq!((st.spkcache.len(), st.fifo.len()), (6, 0));
        // A FIFO that is exactly full does not overflow: nothing pops.
        let p = StreamingParams {
            fifo_len: 6,
            update_period: 2,
            ..params(40)
        };
        let mut st = state();
        feed(&mut st, &p, 6, 1.0, hot);
        assert_eq!((st.spkcache.len(), st.fifo.len()), (0, 6));
    }

    #[test]
    fn compression_keeps_the_predictions_of_cache_frames_that_were_already_there() {
        // No FIFO, period 10, 16-slot cache: chunk 1 fills 10 slots (no predictions kept yet),
        // chunk 2 overflows and compresses. Speaker 1 spoke only in chunk 1's frames, so its
        // predictions survive only if the first compression read them from the earlier forward.
        let p = StreamingParams {
            fifo_len: 0,
            update_period: 10,
            ..params(16)
        };
        let mut st = state();
        feed(&mut st, &p, 10, 1.0, [0.9, 0.0, 0.0, 0.0]);
        assert_eq!((st.spkcache.len(), st.spkcache_preds.is_none()), (10, true));
        let preds: Vec<f32> = (0..10)
            .flat_map(|_| [0.0, 0.9, 0.0, 0.0])
            .chain((0..10).flat_map(|_| [0.9, 0.0, 0.0, 0.0]))
            .collect();
        let chunk_emb: Vec<f32> = (11..21).map(|x| x as f32).collect();
        st.update(&p, (1, 4), &preds, &chunk_emb, (10, 0, 0, 10));
        let sp = st.spkcache_preds.as_ref().expect("compressed");
        assert!(
            sp.chunks(4).any(|r| r[1] > 0.5),
            "speaker 1 only spoke in the old cache frames"
        );
        // Later steps extend and recompress the cache: its predictions stay one row per slot.
        for i in 0..3 {
            feed(
                &mut st,
                &p,
                10,
                100.0 + 10.0 * i as f32,
                [0.0, 0.9, 0.0, 0.0],
            );
            assert_eq!(st.spkcache.len(), 16);
            assert_eq!(st.spkcache_preds.as_ref().unwrap().len(), 16 * 4);
        }
    }

    #[test]
    fn every_streaming_length_is_bounded_by_name() {
        type Setter = fn(&mut StreamingParams, usize);
        let setters: [(&str, Setter); 7] = [
            ("chunk_len", |p, v| p.chunk_len = v),
            ("left_context", |p, v| p.left_context = v),
            ("right_context", |p, v| p.right_context = v),
            ("fifo_len", |p, v| p.fifo_len = v),
            ("spkcache_len", |p, v| p.spkcache_len = v),
            ("update_period", |p, v| p.update_period = v),
            ("sil_frames_per_spk", |p, v| p.sil_frames_per_spk = v),
        ];
        for (name, set) in setters {
            let mut p = params(24);
            set(&mut p, MAX_STREAM_FRAMES + 1);
            let err = p.validate().unwrap_err().to_string();
            assert!(
                err.contains(name) && err.contains("exceeds"),
                "{name}: {err}"
            );
        }
    }

    #[test]
    fn a_cache_that_exactly_fits_is_not_compressed() {
        // 16 frames into a 16-slot cache: nothing to drop, so the frames (and the absence of
        // cache predictions) are left alone; one more frame tips it over.
        let p = StreamingParams {
            fifo_len: 0,
            update_period: 16,
            ..params(16)
        };
        let mut st = state();
        feed(&mut st, &p, 16, 1.0, [0.9, 0.0, 0.0, 0.0]);
        assert_eq!(st.spkcache, (1..=16).map(|x| x as f32).collect::<Vec<_>>());
        assert!(st.spkcache_preds.is_none());
        let p = StreamingParams {
            update_period: 17,
            ..p
        };
        let mut st = state();
        feed(&mut st, &p, 17, 1.0, [0.9, 0.0, 0.0, 0.0]);
        assert!(st.spkcache_preds.is_some());
    }

    #[test]
    fn first_bad_sample_flags_nan_infinity_and_garbage_magnitudes() {
        // Real signal, including 16-bit scale and the limit itself, is clean.
        let clean = [0.0, 1.0, -32768.0, MAX_ABS_PCM, -MAX_ABS_PCM];
        assert_eq!(first_bad_sample(&clean), None);
        assert_eq!(first_bad_sample(&[0.0, f32::NAN]), Some(1));
        assert_eq!(first_bad_sample(&[f32::NEG_INFINITY]), Some(0));
        // Finite but reinterpreted-bytes garbage: the mel would overflow to infinity.
        assert_eq!(first_bad_sample(&[0.0, 0.0, f32::MAX]), Some(2));
        assert_eq!(first_bad_sample(&[0.0, 2.0 * MAX_ABS_PCM]), Some(1));
        assert_eq!(first_bad_sample(&[-2.0 * MAX_ABS_PCM]), Some(0));
    }

    /// The shared block kernel asserts a rank 2 or 3 depthwise weight, so a GGUF that declares
    /// another rank must be refused at load, not at first audio.
    #[cfg(feature = "mmap")]
    #[test]
    fn a_depthwise_weight_of_the_wrong_rank_is_refused() {
        let path = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".leap/models/sortformer/sortformer-4spk-v2.1-q4_0.gguf");
        if !path.exists() {
            assert!(
                std::env::var("CERA_REQUIRE_MODEL").as_deref() != Ok("1"),
                "CERA_REQUIRE_MODEL=1 but {} is absent",
                path.display()
            );
            eprintln!("{} not found, skipping", path.display());
            return;
        }
        let g = GgufFile::open_arc(&path).unwrap();
        let mut w = SortformerWeights::from_gguf(&g).unwrap();
        let (n_embd, n_ff) = (w.config.n_embd, w.layers[0].ffn_up_w.rows);
        let kernel = w.layers[0].conv_dw_shape[0];
        check_encoder_block(0, &w.layers[0], n_embd, n_ff, kernel).unwrap();
        w.layers[0].conv_dw_shape = vec![kernel, 1, 1, n_embd];
        let err = check_encoder_block(0, &w.layers[0], n_embd, n_ff, kernel)
            .unwrap_err()
            .to_string();
        assert!(err.contains("rank 2 or 3"), "{err}");
    }

    #[test]
    fn a_window_past_the_offline_bound_is_refused_by_name() {
        let huge = StreamingParams {
            max_index: 1 << 30,
            ..params(188)
        }
        .with_chunking(1 << 20, 0, 0, 0, 188, 144);
        let err = huge.validate().unwrap_err().to_string();
        assert!(err.contains("exceeds") && err.contains("7500"), "{err}");
        // Each term counts: the same total split another way is refused too.
        let split = StreamingParams {
            max_index: 1 << 30,
            ..params(188)
        }
        .with_chunking(2_000, 2_000, 2_000, 2_000, 2_000, 144);
        assert!(split.validate().is_err());
        let ok = StreamingParams {
            max_index: 1 << 30,
            ..params(188)
        }
        .with_chunking(2_000, 2_000, 2_000, 1_000, 500, 144);
        assert!(ok.validate().is_ok(), "7500 is allowed");
    }

    #[test]
    fn a_matrix_whose_columns_do_not_fill_its_blocks_is_refused() {
        // 192 columns are 6 blocks of 32 but 0.75 of a K-quant block: every row would read as
        // zeros. A hermetic matrix with the dtype set stands in for a patched file.
        let mut w = MmapWeight::from_owned_f32(vec![0.0; 4 * 192], 4, 192);
        for ok in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::Q8_0,
            DType::Q4_0,
            DType::Q4_1,
        ] {
            w.dtype = ok;
            assert!(check_gemv_dtype("m", &w).is_ok(), "{ok:?}");
        }
        for bad in [DType::Q4KM, DType::Q5KM, DType::Q6K] {
            w.dtype = bad;
            let err = check_gemv_dtype("m", &w).unwrap_err().to_string();
            assert!(err.contains("block size 256"), "{bad:?}: {err}");
        }
        w.dtype = DType::I32;
        assert!(
            check_gemv_dtype("m", &w)
                .unwrap_err()
                .to_string()
                .contains("no matmul kernel")
        );
    }

    #[test]
    fn first_bad_mel_flags_nan_infinity_and_overflow_magnitudes() {
        assert_eq!(
            first_bad_mel(&[0.0, -17.0, 46.0, MAX_ABS_MEL, -MAX_ABS_MEL]),
            None
        );
        assert_eq!(first_bad_mel(&[0.0, f32::NAN]), Some(1));
        assert_eq!(first_bad_mel(&[f32::INFINITY]), Some(0));
        assert_eq!(first_bad_mel(&[0.0, 0.0, 2.0 * MAX_ABS_MEL]), Some(2));
        assert_eq!(first_bad_mel(&[-1e9]), Some(0));
    }

    #[test]
    fn the_session_types_can_move_to_a_worker_thread() {
        // `push_audio` tells callers to use a worker thread: keep that true.
        fn send<T: Send>() {}
        fn sync<T: Sync>() {}
        send::<SortformerLive>();
        send::<SortformerStream>();
        send::<MelStream>();
        send::<SortformerModel>();
        sync::<SortformerModel>();
    }

    /// The incremental mel front end is the whole-clip mel to the bit for any way of cutting the
    /// audio. Synthetic tables stand in for the checkpoint's, so this runs without a model.
    #[test]
    fn mel_stream_matches_the_whole_clip_mel_bit_for_bit() {
        let nm = 8;
        let window: Vec<f32> = (0..N_FFT)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / N_FFT as f32).cos())
            .collect();
        let fb: Vec<f32> = (0..nm * N_FFT_BINS)
            .map(|i| ((i * 7919) % 101) as f32 / 1000.0)
            .collect();
        let mut seed = 12345u32;
        let pcm: Vec<f32> = (0..3_000)
            .map(|i| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                0.3 * (i as f32 * 0.05).sin() + (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect();
        let (mut want, n) = log_mel_with_tables(&pcm, nm, &window, &fb, false);
        let valid = n.min(pcm.len() / HOP_LEN);
        want.truncate(valid * nm);
        for piece in [1usize, 7, 160, 161, 777, pcm.len()] {
            let mut ms = MelStream::from_tables(nm, &window, &fb);
            let mut got = Vec::new();
            for part in pcm.chunks(piece) {
                got.extend(ms.push(part).unwrap());
                assert!(
                    ms.buffered_samples() <= N_FFT + HOP_LEN + piece,
                    "piece {piece}"
                );
            }
            got.extend(ms.finish());
            assert_eq!(ms.frames(), valid, "piece {piece}");
            assert!(
                got.len() == want.len()
                    && got
                        .iter()
                        .zip(&want)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                "piece {piece}: mel differs from the whole-clip mel"
            );
        }
        let mut ms = MelStream::from_tables(nm, &window, &fb);
        ms.finish();
        let err = ms.push(&[0.0]).unwrap_err().to_string();
        assert!(err.contains("push after finish"), "unexpected error: {err}");
    }

    #[test]
    fn a_threshold_of_one_is_allowed() {
        let p = StreamingParams {
            pred_score_threshold: 1.0,
            ..params(24)
        };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn the_offline_length_limit_is_exact() {
        // 7500 encoder frames = 60_000 mel frames; one more is refused, from the length alone.
        assert!(ensure_offline_len(60_000 * HOP_LEN).is_ok());
        let err = ensure_offline_len(60_001 * HOP_LEN)
            .unwrap_err()
            .to_string();
        assert!(err.contains("7501") && err.contains("offline"), "{err}");
        assert!(ensure_offline_len(0).is_ok());
    }

    #[derive(Clone, Copy)]
    enum TwinMode {
        Delegate,
        Decline,
        Fail,
        Short,
        Long,
    }
    struct TwinDouble(TwinMode);
    impl TwinDouble {
        fn out(&self) -> Result<Option<Vec<f32>>> {
            match self.0 {
                TwinMode::Delegate => Ok(Some(vec![1.0; 4])),
                TwinMode::Decline => Ok(None),
                TwinMode::Fail => anyhow::bail!("the accelerator is gone"),
                TwinMode::Short => Ok(Some(vec![1.0; 3])),
                TwinMode::Long => Ok(Some(vec![1.0; 5])),
            }
        }
    }
    impl SortformerAccelerator for TwinDouble {
        fn pre_encode(&self, _mel: &[f32], _n: usize) -> Result<Option<Vec<f32>>> {
            self.out()
        }
        fn predict(&self, _emb: &[f32], _t: usize) -> Result<Option<Vec<f32>>> {
            self.out()
        }
        fn log_mel(&self, _samples: &[f32], _n: usize) -> Result<Option<Vec<f32>>> {
            self.out()
        }
    }

    /// Weightless `SortformerWeights` with a scripted accelerator: the fallback contract
    /// touches no weights, so twins built here run in CI.
    fn twin_weights(mode: TwinMode) -> SortformerWeights {
        let accel = OnceLock::new();
        assert!(
            accel
                .set(Arc::new(TwinDouble(mode)) as Arc<dyn SortformerAccelerator>)
                .is_ok()
        );
        SortformerWeights {
            config: SortformerConfig {
                n_layer: 0,
                n_embd: 0,
                n_ff: 0,
                n_head: 0,
                eps: 0.0,
                n_mel_bins: 0,
                tf_layers: 0,
                tf_d: 0,
                tf_heads: 0,
                tf_inner: 0,
                tf_eps: 0.0,
                n_spk: 0,
                subsampling: 0,
                xscaling: false,
                pad_to: 0,
            },
            streaming: params(24),
            enc_cfg: AudioEncoderConfig {
                n_layer: 0,
                n_embd: 0,
                n_ff: 0,
                n_head: 0,
                eps: 0.0,
                n_mel_bins: 0,
                llm_hidden_size: 0,
            },
            conv_stem: ConvStemWeights {
                layers: vec![],
                pre_encode_out_w: MmapWeight::from_owned_f32(vec![], 0, 0),
                pre_encode_out_b: vec![],
            },
            layers: vec![],
            proj_w: MmapWeight::from_owned_f32(vec![], 0, 0),
            proj_b: vec![],
            tf: vec![],
            head_hidden_w: MmapWeight::from_owned_f32(vec![], 0, 0),
            head_hidden_b: vec![],
            head_out_w: MmapWeight::from_owned_f32(vec![], 0, 0),
            head_out_b: vec![],
            window: vec![],
            mel_fb: vec![],
            accel,
            accel_warned: AccelWarned::default(),
        }
    }

    /// The fallback contract without a model: a well-formed output is taken, a decline, an
    /// error, or a wrong-length output (short or long) falls back to `None`, and each faulty
    /// stage latches its own warn-once bit. The guard touches no weights, so this twin runs
    /// in CI where the model-gated parity suite skips.
    #[test]
    fn accelerated_checked_routes_decline_fail_and_wrong_lengths_to_none() {
        use std::sync::atomic::Ordering;

        let latched = |w: &SortformerWeights| {
            [
                w.accel_warned.log_mel.load(Ordering::Relaxed),
                w.accel_warned.stem.load(Ordering::Relaxed),
                w.accel_warned.predict.load(Ordering::Relaxed),
            ]
        };

        let w = twin_weights(TwinMode::Delegate);
        assert_eq!(
            w.accelerated_checked(AccelStage::Stem, 4, |a| a.pre_encode(&[], 0)),
            Some(vec![1.0; 4])
        );
        assert_eq!(latched(&w), [false, false, false]);

        let w = twin_weights(TwinMode::Decline);
        assert_eq!(
            w.accelerated_checked(AccelStage::Stem, 4, |a| a.pre_encode(&[], 0)),
            None
        );
        assert_eq!(latched(&w), [false, false, false]);

        for mode in [TwinMode::Fail, TwinMode::Short, TwinMode::Long] {
            let w = twin_weights(mode);
            assert_eq!(
                w.accelerated_checked(AccelStage::LogMel, 4, |a| a.log_mel(&[], 0)),
                None
            );
            assert_eq!(latched(&w), [true, false, false]);
            // A second fault of the same stage stays quiet but still falls back.
            assert_eq!(
                w.accelerated_checked(AccelStage::LogMel, 4, |a| a.log_mel(&[], 0)),
                None
            );
            // ...while each other stage still gets its own first warning.
            assert_eq!(
                w.accelerated_checked(AccelStage::Stem, 4, |a| a.pre_encode(&[], 0)),
                None
            );
            assert_eq!(latched(&w), [true, true, false]);
            assert_eq!(
                w.accelerated_checked(AccelStage::Predict, 4, |a| a.predict(&[], 0)),
                None
            );
            assert_eq!(latched(&w), [true, true, true]);
        }
    }

    /// Child leg of [`warn_once_faults_reach_stderr_without_a_subscriber`]: faults one stage
    /// and returns. Harmless under the normal suite; the parent runs it in a subprocess.
    #[test]
    fn warn_once_stderr_child_emits_one_fault() {
        let w = twin_weights(TwinMode::Fail);
        assert_eq!(
            w.accelerated_checked(AccelStage::Predict, 4, |a| a.predict(&[], 0)),
            None
        );
    }

    /// The warn-once fault line reaches stderr with no subscriber installed: the `eprintln!`
    /// is the only accelerator-fault signal visible on shipping mobile, so deleting it must
    /// fail this test, not pass silently. A subprocess with `--nocapture` carries the proof:
    /// the harness captures `eprintln!` in-process, so only a subprocess's real fd 2 shows
    /// the emission.
    #[test]
    fn warn_once_faults_reach_stderr_without_a_subscriber() {
        let exe = std::env::current_exe().unwrap();
        let out = std::process::Command::new(exe)
            .arg("--exact")
            .arg("model::sortformer::tests::warn_once_stderr_child_emits_one_fault")
            .arg("--nocapture")
            .output()
            .unwrap();
        assert!(out.status.success());
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.lines()
                .any(|l| l.starts_with("cera-sortformer: predict ")
                    && l.contains("the accelerator is gone")),
            "stderr must carry the fault: {err:?}"
        );
    }

    /// Child leg of [`warn_once_length_faults_reach_stderr_without_a_subscriber`]: feeds a
    /// short stage output through the length check and returns. Harmless under the normal
    /// suite; the parent runs it in a subprocess.
    #[test]
    fn warn_once_length_stderr_child_emits_one_fault() {
        let w = twin_weights(TwinMode::Short);
        assert_eq!(
            w.accelerated_checked(AccelStage::Predict, 4, |a| a.predict(&[], 0)),
            None
        );
    }

    /// The length-mismatch fault line reaches stderr too: silently returning `None` from
    /// the wrong-length arm of `accelerated_checked` would pass the fail-path test above,
    /// so this pins the arm's own warning. Same subprocess proof.
    #[test]
    fn warn_once_length_faults_reach_stderr_without_a_subscriber() {
        let exe = std::env::current_exe().unwrap();
        let out = std::process::Command::new(exe)
            .arg("--exact")
            .arg("model::sortformer::tests::warn_once_length_stderr_child_emits_one_fault")
            .arg("--nocapture")
            .output()
            .unwrap();
        assert!(out.status.success());
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.lines()
                .any(|l| l.starts_with("cera-sortformer: predict ")
                    && l.contains("returned 3 values, want 4")),
            "stderr must carry the length fault: {err:?}"
        );
    }
}
