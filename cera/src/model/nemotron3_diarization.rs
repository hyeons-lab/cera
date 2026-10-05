//! Nemotron-3-Diarization speaker diarization (NVIDIA `Nemotron-3-Diarization`) on the CPU.
//!
//! An end-to-end diarizer beside [`crate::model::sortformer`]: audio in, per-frame
//! speaker-activity probabilities out, for up to eight speakers, with no clustering stage.
//! Output frames are 10 ms (the 4spk Sortformer's are 80 ms). The pieces:
//!
//! ```text
//! pcm 16 kHz
//!   → log-mel [T × 128]          (NeMo front end; no per-feature normalization)
//!   → stack 8 + Linear [T/8 × 512]              (cached embeddings; zero-pad the tail)
//!   → input LayerNorm
//!   → 31 × Transformer block    (pre-LN RoPE attention, pre-LN erf-GELU MLP)
//!   → final LayerNorm
//!   → proj 512 → 192
//!   → subpixel upsample ×8       (Conv1d k=3 pad 1 + interleave, lowered to a matmul)
//!   → relu → Linear → relu → Linear(→8) → sigmoid
//! ```
//!
//! **Streaming.** Each step stacks+projects only the new chunk of features, concatenates it
//! with the cached embeddings of a *speaker cache* and a *FIFO*, runs the whole network over
//! that concatenation, and keeps the chunk's predictions (the lookahead's are dropped; it
//! never joins the FIFO). There is no left context: the cache and FIFO are the history.
//! RoPE positions restart at 0 for every step over `[cache, fifo, chunk, lookahead]`.
//!
//! The update logic is a line-for-line port of the Transformers
//! `Nemotron3DiarizationSpeakerCache` (`update`, `_compress`, `_get_frame_scores`,
//! `_boost_scores`, `_pool_probs`, `_num_popped_frames`), pinned to NeMo's own outputs by
//! `tests/nemotron3_parity.rs` against the fixtures under `cera/tests/fixtures/nemotron3/`
//! (see `scripts/nemotron3_diarization/`).
//!
//! **Padding.** NeMo pads the mel features with zeros to a multiple of `pad_to` (16) and
//! runs the encoder over every stacked group, masking the pad groups out of attention *keys*
//! (queries at pad positions still produce outputs). The subpixel convolution after the
//! encoder has no mask, so a valid edge frame's output depends on its pad neighbor: the pad
//! groups must be computed, not truncated. This port therefore pads the clip mel the same
//! way (at most 15 frames, i.e. one encoder group), masks pad keys in attention, and slices
//! the valid sub-frames out at the end. The embedder itself only stacks groups of 8; the
//! `pad_to` padding is the caller's (offline, padded streaming), like NeMo's front end.
//!
//! The model loads from the GGUF written by `scripts/nemotron3_diarization/convert.py`.

#[cfg(feature = "mmap")]
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result, ensure};

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::model::audio_encoder::{
    HOP_LEN, LOG_MEL_EPS, N_FFT, PREEMPH, SAMPLE_RATE, WINDOW_LEN, load_vec_f32,
};
use crate::model::audio_preprocessor::{
    MelFrameComputer, N_FFT_BINS, log_mel_with_tables, n_frames_for, padded_preemphasized,
};
use crate::model::weights::MmapWeight;
use crate::tensor::DType;

/// Largest speaker count the model's output head has; fixed by the checkpoint.
pub const MAX_SPEAKERS: usize = 8;

/// Milliseconds per output frame (the upsampler restores the 10 ms mel rate).
pub const FRAME_MS: f64 = 10.0;

/// Mel frames the encoder compresses into one frame (and the upsampler expands back).
pub const SUBSAMPLING: usize = 8;

/// Milliseconds per encoder frame: the unit `chunk_len`/`right_context` count in, shared
/// with Sortformer's 80 ms encoder frames.
pub const ENC_FRAME_MS: f64 = SUBSAMPLING as f64 * FRAME_MS;

/// Largest value accepted for any streaming length, in 80 ms encoder frames. The
/// checkpoint's own values are in the hundreds.
const MAX_STREAM_FRAMES: usize = 1 << 20;

// ── Streaming parameters ───────────────────────────────────────────────────

/// Streaming hyper-parameters. Lengths are in 80 ms encoder frames.
///
/// The GGUF carries the checkpoint's defaults (a 264-frame chunk, no FIFO); the model
/// card's "low latency" configurations are the same weights with different numbers here.
/// Latency is `chunk_len + right_context` frames, and the encoder runs over
/// `chunk_len + right_context + fifo_len + spkcache_len` frames per step.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingParams {
    /// New frames emitted per step.
    pub chunk_len: usize,
    /// Extra frames of right context (lookahead) fed to the encoder with each chunk.
    /// Lookahead frames are attended to but never scored or cached.
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
        right_context: usize,
        fifo_len: usize,
        spkcache_len: usize,
        update_period: usize,
    ) -> Self {
        Self {
            chunk_len,
            right_context,
            fifo_len,
            spkcache_len,
            update_period,
            ..self.clone()
        }
    }

    /// The model card's low-latency preset (0.72 s chunk, 1.04 s latency).
    pub fn low_latency(&self) -> Self {
        self.with_chunking(9, 4, 264, 264, 222)
    }

    /// The model card's very-low-latency preset (0.48 s chunk, 0.64 s latency).
    pub fn very_low_latency(&self) -> Self {
        self.with_chunking(6, 2, 264, 264, 222)
    }

    /// The model card's ultra-low-latency preset (0.24 s chunk, 0.32 s latency).
    pub fn ultra_low_latency(&self) -> Self {
        self.with_chunking(3, 1, 264, 264, 222)
    }

    /// Encoder frames one step attends over at most: the chunk, its lookahead, the FIFO
    /// and the speaker cache. What an accelerator has to be staged for.
    pub fn window_frames(&self) -> usize {
        // Saturating: the fields are public and `with_chunking` skips validation, so a
        // hand-made config could otherwise wrap (release) or panic (debug) here.
        self.chunk_len
            .saturating_add(self.right_context)
            .saturating_add(self.fifo_len)
            .saturating_add(self.spkcache_len)
    }

    /// Check hand-made parameters (`with_chunking` skips validation): every length is
    /// positive and bounded so the frame arithmetic downstream cannot overflow, and the
    /// score hyper-parameters are finite and in range.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.chunk_len > 0, "chunk_len must be > 0");
        ensure!(self.update_period > 0, "update_period must be > 0");
        // Bound every length so the frame arithmetic downstream cannot overflow.
        for (name, v) in [
            ("chunk_len", self.chunk_len),
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
        let window = self.chunk_len + self.right_context + self.fifo_len + self.spkcache_len;
        ensure!(
            window <= MAX_OFFLINE_FRAMES,
            "chunk + right + fifo + spkcache = {window} frames exceeds {MAX_OFFLINE_FRAMES}"
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
        // A zero per-speaker budget is degenerate but well-defined (all boost counts floor
        // to 0); below zero the subtraction would wrap.
        ensure!(
            self.spkcache_len / MAX_SPEAKERS >= self.sil_frames_per_spk,
            "spkcache_len {} leaves no room for {} silence frames per speaker",
            self.spkcache_len,
            self.sil_frames_per_spk
        );
        Ok(())
    }
}

/// An accelerator (the Hexagon NPU) for the heavy parts of a diarization step. Set once per
/// model with [`Nemotron3Model::set_accelerator`]; every stream and live diarizer made from
/// the model then uses it. Each method returns `Ok(None)` to decline an input it cannot take
/// (for example one longer than the window it was staged for), and an `Err` is a failure:
/// both fall back to the CPU, the failure with a one-time warning, so a diarizer never stops
/// because the accelerator did. An output of the wrong length is treated like a failure of
/// that stage (CPU fallback with a one-time warning), never trusted.
///
/// Declining or failing must leave no observable state: the driver calls again on the next
/// input (falling back to the CPU each time), and calls may come from multiple threads (note
/// `Send + Sync`), so implementations must be thread-safe under sharing.
pub trait Nemotron3Accelerator: Send + Sync {
    /// Stack 8 mel frames per group (zero-padding the tail) and project to the encoder
    /// width over `n_frames` of `[n_frames x n_mel]` mel: `[enc_frames(n_frames) x n_embd]`,
    /// like [`Nemotron3Model::embed`]. The caller pads the clip mel to `pad_to` first when the
    /// reference would (offline and padded streaming); the embedder itself only stacks.
    fn embed(&self, mel: &[f32], n_frames: usize) -> Result<Option<Vec<f32>>>;

    /// The input norm, Transformer blocks, final norm, `proj`, subpixel upsampler and
    /// speaker head over `emb` (`[t x n_embd]`, `t` from its length) with `valid_sub` valid
    /// 10 ms sub-frames (cached groups count 8 each): `[t * 8 x n_spk]` sigmoid activities at
    /// 10 ms per frame with sub-frames past the valid groups zeroed, like [`Nemotron3Model::predict`].
    fn predict(&self, emb: &[f32], valid_sub: usize) -> Result<Option<Vec<f32>>>;

    /// Log-mel of `n_frames` frames over pre-emphasised, centre-padded samples: frame `f` is
    /// `samples[f * 160 .. f * 160 + 512]`, windowed with the model's own window, projected
    /// with its own filterbank, and `ln(energy + 2^-24)` of the result. `[n_frames x n_mel]`,
    /// time-major, the values [`Nemotron3Model::log_mel`] returns. Optional: the default
    /// declines, leaving the mel on the CPU.
    fn log_mel(&self, samples: &[f32], n_frames: usize) -> Result<Option<Vec<f32>>> {
        let _ = (samples, n_frames);
        Ok(None)
    }
}

// ── Config and weights ─────────────────────────────────────────────────────

/// Architecture constants read from the GGUF.
#[derive(Debug, Clone)]
pub struct Nemotron3Config {
    /// Transformer blocks.
    pub n_layer: usize,
    /// Encoder width (also the cached embedding width).
    pub n_embd: usize,
    /// MLP width.
    pub n_ff: usize,
    /// Attention heads.
    pub n_head: usize,
    /// LayerNorm epsilon.
    pub eps: f32,
    /// Mel bins.
    pub n_mel_bins: usize,
    /// Head width after `proj`.
    pub tf_d: usize,
    /// Speakers the head predicts.
    pub n_spk: usize,
    /// Mel frames per encoder frame.
    pub subsampling: usize,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// RoPE training context the checkpoint records (informational: positions are computed
    /// on the fly and restart at 0 for every step).
    pub rope_max_pos: usize,
    /// NeMo pads mel features to a multiple of this.
    pub pad_to: usize,
}

pub(crate) struct EncoderLayer {
    pub(crate) ln1_w: Vec<f32>,
    pub(crate) ln1_b: Vec<f32>,
    pub(crate) q_w: MmapWeight,
    pub(crate) k_w: MmapWeight,
    pub(crate) v_w: MmapWeight,
    pub(crate) o_w: MmapWeight,
    pub(crate) o_b: Vec<f32>,
    pub(crate) ln2_w: Vec<f32>,
    pub(crate) ln2_b: Vec<f32>,
    pub(crate) up_w: MmapWeight,
    pub(crate) up_b: Vec<f32>,
    pub(crate) down_w: MmapWeight,
    pub(crate) down_b: Vec<f32>,
}

/// Every Nemotron-3 tensor, loaded from a converted GGUF. Held by [`Nemotron3Model`] and its
/// streams; the crate's callers go through those.
pub(crate) struct Nemotron3Weights {
    /// Architecture constants.
    pub config: Nemotron3Config,
    /// The checkpoint's streaming defaults.
    pub streaming: StreamingParams,
    pub(crate) embed_w: MmapWeight,
    pub(crate) input_norm_w: Vec<f32>,
    pub(crate) input_norm_b: Vec<f32>,
    pub(crate) final_norm_w: Vec<f32>,
    pub(crate) final_norm_b: Vec<f32>,
    pub(crate) layers: Vec<EncoderLayer>,
    pub(crate) proj_w: MmapWeight,
    pub(crate) proj_b: Vec<f32>,
    /// Subpixel Conv1d `[tf_d * 8, tf_d * 3]` (the `[out, in, k]` taps flattened row-major).
    pub(crate) up_w: MmapWeight,
    pub(crate) up_b: Vec<f32>,
    pub(crate) dense_w: MmapWeight,
    pub(crate) dense_b: Vec<f32>,
    pub(crate) out_w: MmapWeight,
    pub(crate) out_b: Vec<f32>,
    /// Learned silence embedding filling reserved cache slots.
    pub(crate) silence_embeds: Vec<f32>,
    /// `N_FFT`-long window with the `WINDOW_LEN` taps centered in it.
    pub(crate) window: Vec<f32>,
    /// `[n_mel_bins × N_FFT_BINS]`.
    pub(crate) mel_fb: Vec<f32>,
    /// Set once by [`Nemotron3Model::set_accelerator`].
    accel: OnceLock<Arc<dyn Nemotron3Accelerator>>,
    /// Whether an accelerator failure has been logged, once per [`AccelStage`].
    accel_warned: AccelWarned,
}

/// One warn-once latch per [`AccelStage`]. Named fields, not an indexed array, so adding a
/// stage fails the build (in `warned`) instead of panicking with an out-of-bounds index on
/// the first warning.
#[derive(Default)]
struct AccelWarned {
    log_mel: AtomicBool,
    embed: AtomicBool,
    predict: AtomicBool,
}

/// One stage of the accelerated diarization step. Each stage warns once, independently: a
/// transient failure in one must not suppress the first warning of another.
#[derive(Clone, Copy)]
enum AccelStage {
    LogMel,
    Embed,
    Predict,
}

impl AccelStage {
    fn label(self) -> &'static str {
        match self {
            AccelStage::LogMel => "log-mel",
            AccelStage::Embed => "embedder",
            AccelStage::Predict => "predict",
        }
    }

    /// This stage's warn-once latch. Exhaustive: a new variant fails the build here.
    fn warned(self, latched: &AccelWarned) -> &AtomicBool {
        match self {
            AccelStage::LogMel => &latched.log_mel,
            AccelStage::Embed => &latched.embed,
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
        "unsupported Nemotron-3 GGUF: {what} is {got:?}, this build implements {want:?}"
    );
    Ok(())
}

impl Nemotron3Weights {
    /// Load the model from a GGUF written by `scripts/nemotron3_diarization/convert.py`.
    pub fn from_gguf(g: &Arc<GgufFile>) -> Result<Self> {
        ensure!(
            g.architecture() == Some("nemotron3_diarization"),
            "not a Nemotron-3 GGUF (general.architecture = {:?})",
            g.architecture()
        );

        // Front end: this build implements exactly NeMo's parameters for the shipped
        // checkpoint, so a GGUF that says otherwise is refused instead of silently run
        // through the wrong mel.
        expect_eq(
            "mel normalize",
            g.get_str("nemotron3.mel.normalize"),
            Some("NA"),
        )?;
        expect_eq("mel n_fft", req_u32(g, "nemotron3.mel.n_fft")?, N_FFT)?;
        expect_eq(
            "mel win_length",
            req_u32(g, "nemotron3.mel.win_length")?,
            WINDOW_LEN,
        )?;
        expect_eq(
            "mel hop_length",
            req_u32(g, "nemotron3.mel.hop_length")?,
            HOP_LEN,
        )?;
        expect_eq(
            "sample_rate",
            req_u32(g, "nemotron3.sample_rate")?,
            SAMPLE_RATE as usize,
        )?;
        expect_eq("mel preemph", req_f32(g, "nemotron3.mel.preemph")?, 0.97)?;
        expect_eq("mel mag_power", req_f32(g, "nemotron3.mel.mag_power")?, 2.0)?;
        ensure!(
            (req_f32(g, "nemotron3.mel.log_zero_guard")? - LOG_MEL_EPS).abs() < 1e-12,
            "unsupported Nemotron-3 GGUF: mel log_zero_guard differs from {LOG_MEL_EPS}"
        );

        // The padded length drives how many steps a clip takes, so a hostile value is a CPU hang.
        let pad_to = req_u32(g, "nemotron3.mel.pad_to")?;
        ensure!(
            (1..=MAX_PAD_TO).contains(&pad_to),
            "nemotron3.mel.pad_to {pad_to} outside 1..={MAX_PAD_TO}"
        );
        let n_layer = req_u32(g, "nemotron3.block_count")?;
        let n_embd = req_u32(g, "nemotron3.embedding_length")?;
        let n_ff = req_u32(g, "nemotron3.feed_forward_length")?;
        let n_head = req_u32(g, "nemotron3.attention.head_count")?;
        let eps = req_f32(g, "nemotron3.attention.layer_norm_epsilon")?;
        ensure!(
            eps.is_finite() && eps > 0.0,
            "nemotron3.attention.layer_norm_epsilon {eps} must be finite and > 0"
        );
        let n_mel_bins = req_u32(g, "nemotron3.num_mel_bins")?;
        let tf_d = req_u32(g, "nemotron3.head.hidden_size")?;
        let n_spk = req_u32(g, "nemotron3.max_speakers")?;
        expect_eq("max_speakers", n_spk, MAX_SPEAKERS)?;
        ensure!(
            n_head > 0 && n_embd % n_head == 0,
            "nemotron3.attention.head_count {n_head} must be > 0 and divide \
             nemotron3.embedding_length {n_embd}"
        );
        ensure!(
            n_embd / n_head <= 64,
            "nemotron3 head dimension {} ({n_embd} / {n_head}) exceeds maximum supported accumulator width 64",
            n_embd / n_head
        );
        ensure!(n_ff > 0, "nemotron3.feed_forward_length must be > 0");
        ensure!(tf_d > 0, "nemotron3.head.hidden_size must be > 0");
        // Counts come from the file; bound them before they size an allocation.
        ensure!(
            (1..=MAX_LAYERS).contains(&n_layer),
            "block_count {n_layer} outside 1..={MAX_LAYERS}"
        );
        let subsampling = req_u32(g, "nemotron3.subsampling_factor")?;
        expect_eq("subsampling_factor", subsampling, SUBSAMPLING)?;
        let rope_theta = req_f32(g, "nemotron3.rope.theta")?;
        expect_eq("rope.theta", rope_theta, 10_000.0)?;
        let rope_max_pos = req_u32(g, "nemotron3.rope.max_pos")?;

        let streaming = StreamingParams {
            chunk_len: req_u32(g, "nemotron3.stream.chunk_len")?,
            right_context: req_u32(g, "nemotron3.stream.chunk_right_context")?,
            fifo_len: req_u32(g, "nemotron3.stream.fifo_len")?,
            spkcache_len: req_u32(g, "nemotron3.stream.spkcache_len")?,
            update_period: req_u32(g, "nemotron3.stream.spkcache_update_period")?,
            sil_frames_per_spk: req_u32(g, "nemotron3.stream.spkcache_sil_frames_per_spk")?,
            pred_score_threshold: req_f32(g, "nemotron3.stream.pred_score_threshold")?,
            scores_boost_latest: req_f32(g, "nemotron3.stream.scores_boost_latest")?,
            sil_threshold: req_f32(g, "nemotron3.stream.sil_threshold")?,
            strong_boost_rate: req_f32(g, "nemotron3.stream.strong_boost_rate")?,
            weak_boost_rate: req_f32(g, "nemotron3.stream.weak_boost_rate")?,
            min_pos_scores_rate: req_f32(g, "nemotron3.stream.min_pos_scores_rate")?,
            max_index: req_u32(g, "nemotron3.stream.max_index")?,
        };
        streaming.validate()?;

        let weight = |name: &str| -> Result<MmapWeight> {
            let w = MmapWeight::from_gguf(g, name).with_context(|| format!("loading {name}"))?;
            check_gemv_dtype(name, &w)?;
            Ok(w)
        };
        let embed_w = weight("nd.embed.proj.weight")?;
        ensure!(
            embed_w.rows == n_embd && embed_w.cols == n_mel_bins * SUBSAMPLING,
            "nd.embed.proj.weight is {}x{}, expected {n_embd}x{}",
            embed_w.rows,
            embed_w.cols,
            n_mel_bins * SUBSAMPLING,
        );
        let input_norm_w = load_vec_f32(g, "nd.input_norm.weight")?;
        let input_norm_b = load_vec_f32(g, "nd.input_norm.bias")?;
        let final_norm_w = load_vec_f32(g, "nd.final_norm.weight")?;
        let final_norm_b = load_vec_f32(g, "nd.final_norm.bias")?;
        check_lens(
            "norms",
            &[
                ("nd.input_norm.weight", input_norm_w.len(), n_embd),
                ("nd.input_norm.bias", input_norm_b.len(), n_embd),
                ("nd.final_norm.weight", final_norm_w.len(), n_embd),
                ("nd.final_norm.bias", final_norm_b.len(), n_embd),
            ],
        )?;

        let mut layers = Vec::with_capacity(n_layer);
        for n in 0..n_layer {
            let p = format!("nd.blk.{n}");
            let w = |s: &str| weight(&format!("{p}.{s}.weight"));
            let b = |s: &str| load_vec_f32(g, &format!("{p}.{s}.bias"));
            layers.push(EncoderLayer {
                ln1_w: load_vec_f32(g, &format!("{p}.ln1.weight"))?,
                ln1_b: b("ln1")?,
                q_w: w("attn_q")?,
                k_w: w("attn_k")?,
                v_w: w("attn_v")?,
                o_w: w("attn_o")?,
                o_b: b("attn_o")?,
                ln2_w: load_vec_f32(g, &format!("{p}.ln2.weight"))?,
                ln2_b: b("ln2")?,
                up_w: w("mlp_up")?,
                up_b: b("mlp_up")?,
                down_w: w("mlp_down")?,
                down_b: b("mlp_down")?,
            });
            check_encoder_layer(n, &layers[n], n_embd, n_ff)?;
        }

        let proj_w = weight("nd.proj.weight")?;
        let up_w = weight("nd.upsample.weight")?;
        let dense_w = weight("nd.classifier.dense.weight")?;
        let out_w = weight("nd.classifier.out.weight")?;
        ensure!(
            proj_w.rows == tf_d && proj_w.cols == n_embd,
            "nd.proj.weight is {}x{}, expected {tf_d}x{n_embd}",
            proj_w.rows,
            proj_w.cols
        );
        ensure!(
            up_w.rows == tf_d * SUBSAMPLING && up_w.cols == tf_d * UPSAMPLE_KERNEL,
            "nd.upsample.weight is {}x{}, expected {}x{}",
            up_w.rows,
            up_w.cols,
            tf_d * SUBSAMPLING,
            tf_d * UPSAMPLE_KERNEL,
        );
        ensure!(
            dense_w.rows == tf_d
                && dense_w.cols == tf_d
                && out_w.rows == n_spk
                && out_w.cols == tf_d,
            "classifier shapes disagree with d {tf_d} / {n_spk} speakers"
        );
        let proj_b = load_vec_f32(g, "nd.proj.bias")?;
        let up_b = load_vec_f32(g, "nd.upsample.bias")?;
        let dense_b = load_vec_f32(g, "nd.classifier.dense.bias")?;
        let out_b = load_vec_f32(g, "nd.classifier.out.bias")?;
        let silence_embeds = load_vec_f32(g, "nd.silence_embeds")?;
        check_lens(
            "head",
            &[
                ("nd.proj.bias", proj_b.len(), tf_d),
                ("nd.upsample.bias", up_b.len(), tf_d * SUBSAMPLING),
                ("nd.classifier.dense.bias", dense_b.len(), tf_d),
                ("nd.classifier.out.bias", out_b.len(), n_spk),
                ("nd.silence_embeds", silence_embeds.len(), n_embd),
            ],
        )?;

        // Mel tables, exactly as the checkpoint ships them.
        let win = g
            .get_tensor("nd.mel.window")
            .context("loading nd.mel.window")?
            .to_f32_vec();
        ensure!(
            win.len() == WINDOW_LEN,
            "nd.mel.window has {} taps",
            win.len()
        );
        let lo = (N_FFT - WINDOW_LEN) / 2;
        let mut window = vec![0.0f32; N_FFT];
        window[lo..lo + WINDOW_LEN].copy_from_slice(&win);
        let mel_fb = g
            .get_tensor("nd.mel.fb")
            .context("loading nd.mel.fb")?
            .to_f32_vec();
        ensure!(
            mel_fb.len() == n_mel_bins * N_FFT_BINS,
            "nd.mel.fb has {} values, expected {n_mel_bins} x {N_FFT_BINS}",
            mel_fb.len()
        );

        Ok(Self {
            config: Nemotron3Config {
                n_layer,
                n_embd,
                n_ff,
                n_head,
                eps,
                n_mel_bins,
                tf_d,
                n_spk,
                subsampling,
                rope_theta,
                rope_max_pos,
                pad_to,
            },
            streaming,
            embed_w,
            input_norm_w,
            input_norm_b,
            final_norm_w,
            final_norm_b,
            layers,
            proj_w,
            proj_b,
            up_w,
            up_b,
            dense_w,
            dense_b,
            out_w,
            out_b,
            silence_embeds,
            window,
            mel_fb,
            accel: OnceLock::new(),
            accel_warned: AccelWarned::default(),
        })
    }
}

/// Taps of the subpixel Conv1d the upsampler lowers to a matmul (kernel 3, pad 1).
const UPSAMPLE_KERNEL: usize = 3;

/// Upper bound on a layer count read from a GGUF (the shipped model has 31).
const MAX_LAYERS: usize = 256;

/// Longest clip [`Nemotron3Model::diarize_offline`] takes, and widest window any entry point
/// attends over, in encoder frames.
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

/// Upper bound on `nemotron3.mel.pad_to` (NeMo's value is 16).
const MAX_PAD_TO: usize = 64;

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

fn check_encoder_layer(n: usize, l: &EncoderLayer, n_embd: usize, n_ff: usize) -> Result<()> {
    let what = format!("encoder layer {n}");
    ensure!(
        [&l.q_w, &l.k_w, &l.v_w, &l.o_w]
            .iter()
            .all(|w| w.rows == n_embd && w.cols == n_embd)
            && l.up_w.rows == n_ff
            && l.up_w.cols == n_embd
            && l.down_w.rows == n_embd
            && l.down_w.cols == n_ff,
        "{what}: shapes disagree with n_embd {n_embd} / n_ff {n_ff}"
    );
    check_lens(
        &what,
        &[
            ("ln1.weight", l.ln1_w.len(), n_embd),
            ("ln1.bias", l.ln1_b.len(), n_embd),
            ("attn_o.bias", l.o_b.len(), n_embd),
            ("ln2.weight", l.ln2_w.len(), n_embd),
            ("ln2.bias", l.ln2_b.len(), n_embd),
            ("mlp_up.bias", l.up_b.len(), n_ff),
            ("mlp_down.bias", l.down_b.len(), n_embd),
        ],
    )
}

/// Valid encoder frames `n` mel frames stack into (the last group zero-pads, like the
/// reference). These are the attention keys; see [`padded_enc_frames`] for the total.
pub fn enc_frames(n: usize) -> usize {
    n.div_ceil(SUBSAMPLING)
}

/// Encoder frames the network runs over for `n` mel frames: the mel zero-pads to a multiple
/// of `pad_to` first, like NeMo, so the convolution's edge tap reads computed pad groups.
pub fn padded_enc_frames(n: usize, pad_to: usize) -> usize {
    n.div_ceil(pad_to.max(1))
        .saturating_mul(pad_to.max(1))
        .div_ceil(SUBSAMPLING)
}

/// Synthetic weights for the Hexagon staging tests (`nemotron3_diarization_hexagon`):
/// `layers` encoder blocks at small widths, every linear a Q8_0 zero matrix. The NPU
/// planner reads shapes and dtypes only, so zeros stage fine.
#[cfg(all(test, feature = "hexagon"))]
pub(crate) fn synthetic_q8_weights(
    layers: usize,
    d: usize,
    heads: usize,
    inner: usize,
    tf_d: usize,
    n_spk: usize,
) -> Nemotron3Weights {
    let q8 = |rows: usize, cols: usize| {
        MmapWeight::from_owned_bytes(
            vec![0u8; rows * cols.div_ceil(32) * 34],
            DType::Q8_0,
            rows,
            cols,
        )
    };
    let zeros = |n: usize| vec![0.0; n];
    Nemotron3Weights {
        config: Nemotron3Config {
            n_layer: layers,
            n_embd: d,
            n_ff: inner,
            n_head: heads,
            eps: 1e-5,
            n_mel_bins: 4,
            tf_d,
            n_spk,
            subsampling: SUBSAMPLING,
            rope_theta: 10_000.0,
            rope_max_pos: 0,
            pad_to: 16,
        },
        streaming: StreamingParams {
            chunk_len: 9,
            right_context: 4,
            fifo_len: 264,
            spkcache_len: 264,
            update_period: 222,
            sil_frames_per_spk: 1,
            pred_score_threshold: 0.25,
            scores_boost_latest: 0.05,
            sil_threshold: 0.2,
            strong_boost_rate: 0.75,
            weak_boost_rate: 1.5,
            min_pos_scores_rate: 0.5,
            max_index: 99_999,
        },
        embed_w: MmapWeight::from_owned_f32(Vec::new(), 0, 0),
        input_norm_w: zeros(d),
        input_norm_b: zeros(d),
        final_norm_w: zeros(d),
        final_norm_b: zeros(d),
        layers: (0..layers)
            .map(|_| EncoderLayer {
                ln1_w: zeros(d),
                ln1_b: zeros(d),
                q_w: q8(d, d),
                k_w: q8(d, d),
                v_w: q8(d, d),
                o_w: q8(d, d),
                o_b: zeros(d),
                ln2_w: zeros(d),
                ln2_b: zeros(d),
                up_w: q8(inner, d),
                up_b: zeros(inner),
                down_w: q8(d, inner),
                down_b: zeros(d),
            })
            .collect(),
        proj_w: q8(tf_d, d),
        proj_b: zeros(tf_d),
        up_w: q8(tf_d * SUBSAMPLING, tf_d * 3),
        up_b: zeros(tf_d * SUBSAMPLING),
        dense_w: q8(tf_d, tf_d),
        dense_b: zeros(tf_d),
        out_w: q8(n_spk, tf_d),
        out_b: zeros(n_spk),
        silence_embeds: Vec::new(),
        window: Vec::new(),
        mel_fb: Vec::new(),
        accel: OnceLock::new(),
        accel_warned: AccelWarned::default(),
    }
}

// ── Model ──────────────────────────────────────────────────────────────────

/// The Nemotron-3 diarizer: front end, encoder, head and streaming driver.
#[derive(Clone)]
pub struct Nemotron3Model {
    w: Arc<Nemotron3Weights>,
}

impl Nemotron3Model {
    /// Load from a converted GGUF.
    pub fn from_gguf(g: &Arc<GgufFile>) -> Result<Self> {
        Ok(Self {
            w: Arc::new(Nemotron3Weights::from_gguf(g)?),
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

    /// Load a converted Nemotron-3 GGUF from in-memory bytes.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>) -> Result<Self> {
        let g = Arc::new(GgufFile::from_bytes(bytes.into())?);
        Self::from_gguf(&g)
    }

    /// The loaded weights, for backends that stage them (the Hexagon port).
    #[cfg(feature = "hexagon")]
    pub(crate) fn weights(&self) -> &Nemotron3Weights {
        &self.w
    }

    /// Architecture constants.
    pub fn config(&self) -> &Nemotron3Config {
        &self.w.config
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

    /// Stack groups of 8 mel frames (zero-padding the tail) and project to the encoder
    /// width: `[frames × 128]` mel to `[enc_frames(frames) × 512]` embeddings. The caller pads
    /// the clip mel to `pad_to` first when the reference would (offline and padded streaming);
    /// the valid groups are what the speaker cache and FIFO store.
    ///
    /// # Panics
    ///
    /// If `mel` is not `[n_frames x 128]`.
    pub fn embed(&self, mel: &[f32], n_frames: usize) -> (Vec<f32>, usize) {
        assert_eq!(
            mel.len(),
            n_frames * self.w.config.n_mel_bins,
            "embed: mel must be [n_frames x n_mel_bins]"
        );
        self.w.embed_for(mel)
    }

    /// Everything after the embedder: input norm, Transformer blocks, final norm, `proj`,
    /// subpixel upsampler and the speaker head, over `emb` (`[t × 512]`, `t` from its length),
    /// with `valid_sub` valid 10 ms sub-frames (cached groups count 8 each: the first
    /// [`enc_frames`] of them are the attention keys). Returns the sigmoid speaker activities
    /// `[t * 8 × 8]` at 10 ms per frame with sub-frames past the valid groups zeroed, like
    /// NeMo's masked output; the caller slices the valid ones.
    ///
    /// # Panics
    ///
    /// If `emb` is not a whole number of 512-wide frames, `valid_sub` is 0 or past `t * 8`,
    /// or `t` is past the 7500-frame attention window the other entry points enforce
    /// (attention memory is quadratic in `t`).
    pub fn predict(&self, emb: &[f32], valid_sub: usize) -> Vec<f32> {
        let c = &self.w.config;
        assert_eq!(emb.len() % c.n_embd, 0, "predict: emb must be [t x n_embd]");
        let t = emb.len() / c.n_embd;
        assert!(
            t <= MAX_OFFLINE_FRAMES,
            "predict: {t} frames exceeds the {MAX_OFFLINE_FRAMES}-frame attention window"
        );
        assert!(
            valid_sub <= t * SUBSAMPLING,
            "predict: {valid_sub} valid sub-frames for {t} frames"
        );
        if t == 0 {
            return Vec::new();
        }
        assert!(
            valid_sub >= 1,
            "predict: no valid sub-frames for {t} frames"
        );
        let want = t * SUBSAMPLING * c.n_spk;
        if let Some(preds) = self
            .w
            .accelerated_checked(AccelStage::Predict, want, |a| a.predict(emb, valid_sub))
        {
            return preds;
        }
        self.predict_with_taps(emb, valid_sub, &mut |_, _| {})
    }

    /// [`Self::predict`] on the CPU whatever accelerator is set: the reference the accelerated
    /// path is compared against.
    pub fn predict_cpu(&self, emb: &[f32], valid_sub: usize) -> Vec<f32> {
        self.predict_with_taps(emb, valid_sub, &mut |_, _| {})
    }

    /// Run the embedder and the prediction on `accel` from here on (one accelerator per model,
    /// shared by every stream and live diarizer made from it, including clones).
    pub fn set_accelerator(&self, accel: Arc<dyn Nemotron3Accelerator>) -> Result<()> {
        self.w
            .accel
            .set(accel)
            .map_err(|_| anyhow::anyhow!("this Nemotron-3 model already has an accelerator"))
    }

    /// Whether [`Self::set_accelerator`] already installed an accelerator.
    pub fn has_accelerator(&self) -> bool {
        self.w.accel.get().is_some()
    }

    /// [`Self::predict`] that reports intermediates to `tap` as it goes: `"input_norm"`,
    /// `"enc.layer{i}"`, `"final_norm"`, `"enc_proj"`, `"upsampled"`, `"logits"`, each a
    /// `[frames × width]` slice over all `t` frames including pad groups (`"upsampled"` and
    /// `"logits"` at 10 ms, the rest at 80 ms). Used by the parity tests to find the first
    /// stage that diverges.
    pub fn predict_with_taps(
        &self,
        emb: &[f32],
        valid_sub: usize,
        tap: &mut dyn FnMut(&str, &[f32]),
    ) -> Vec<f32> {
        let w = &*self.w;
        let c = &w.config;
        assert_eq!(emb.len() % c.n_embd, 0, "predict: emb must be [t x n_embd]");
        let t = emb.len() / c.n_embd;
        assert!(
            t <= MAX_OFFLINE_FRAMES,
            "predict: {t} frames exceeds the {MAX_OFFLINE_FRAMES}-frame attention window"
        );
        assert!(
            valid_sub <= t * SUBSAMPLING,
            "predict: {valid_sub} valid sub-frames for {t} frames"
        );
        if t == 0 {
            return Vec::new();
        }
        assert!(
            valid_sub >= 1,
            "predict: no valid sub-frames for {t} frames"
        );
        // The partial group is a valid key: NeMo's lengths count ceil(mel / 8) groups.
        let valid = enc_frames(valid_sub);

        let mut x = emb.to_vec();
        for row in x.chunks_exact_mut(c.n_embd) {
            cpu::layer_norm_inplace(row, &w.input_norm_w, &w.input_norm_b, c.eps);
        }
        tap("input_norm", &x);

        // One scratch workspace and one RoPE table for the whole forward: positions restart
        // at 0 every step, so every layer shares them.
        let rope = cpu::rope_table(t, c.n_embd / c.n_head, c.rope_theta);
        let mut scratch = EncoderScratch::new(t, c.n_embd, c.n_ff);
        for (il, layer) in w.layers.iter().enumerate() {
            encoder_layer_forward(&mut scratch, &mut x, layer, t, valid, c, &rope);
            tap(&format!("enc.layer{il}"), &x);
        }
        for row in x.chunks_exact_mut(c.n_embd) {
            cpu::layer_norm_inplace(row, &w.final_norm_w, &w.final_norm_b, c.eps);
        }
        tap("final_norm", &x);

        let mut h = vec![0.0f32; t * c.tf_d];
        for (src, dst) in x.chunks_exact(c.n_embd).zip(h.chunks_exact_mut(c.tf_d)) {
            w.proj_w.gemv(src, dst);
            cpu::add_inplace(dst, &w.proj_b);
        }
        tap("enc_proj", &h);

        // Subpixel upsample: Conv1d(k=3, pad 1) lowered to a matmul over unrolled frames,
        // then de-interleaved to 8 sub-frames per encoder frame.
        let n10 = t * SUBSAMPLING;
        let mut up = vec![0.0f32; n10 * c.tf_d];
        let mut unrolled = vec![0.0f32; c.tf_d * UPSAMPLE_KERNEL];
        let mut conv = vec![0.0f32; c.tf_d * SUBSAMPLING];
        for f in 0..t {
            unrolled.fill(0.0);
            for d in 0..UPSAMPLE_KERNEL {
                // Pad 1: tap `d` reads frame `f + d - 1`, zero outside `0..t`.
                let src = f as isize + d as isize - 1;
                if src < 0 || src >= t as isize {
                    continue;
                }
                let row = &h[src as usize * c.tf_d..(src as usize + 1) * c.tf_d];
                for (ch, &v) in row.iter().enumerate() {
                    unrolled[ch * UPSAMPLE_KERNEL + d] = v;
                }
            }
            w.up_w.gemv(&unrolled, &mut conv);
            cpu::add_inplace(&mut conv, &w.up_b);
            for s in 0..SUBSAMPLING {
                up[(f * SUBSAMPLING + s) * c.tf_d..(f * SUBSAMPLING + s + 1) * c.tf_d]
                    .copy_from_slice(&conv[s * c.tf_d..(s + 1) * c.tf_d]);
            }
        }
        tap("upsampled", &up);

        let mut logits = vec![0.0f32; n10 * c.n_spk];
        let mut r = vec![0.0f32; c.tf_d];
        let mut hid = vec![0.0f32; c.tf_d];
        for (row, out) in up
            .chunks_exact(c.tf_d)
            .zip(logits.chunks_exact_mut(c.n_spk))
        {
            r.copy_from_slice(row);
            cpu::relu_inplace(&mut r);
            w.dense_w.gemv(&r, &mut hid);
            cpu::add_inplace(&mut hid, &w.dense_b);
            cpu::relu_inplace(&mut hid);
            w.out_w.gemv(&hid, out);
            cpu::add_inplace(out, &w.out_b);
        }
        tap("logits", &logits);
        cpu::sigmoid_inplace(&mut logits);
        // NeMo's masked output: the mask covers whole groups, so sub-frames past the valid
        // groups read zero while the partial group's own pad sub-frames stay live.
        for v in &mut logits[valid * SUBSAMPLING * c.n_spk..] {
            *v = 0.0;
        }
        logits
    }

    /// Offline diarization of a whole clip in one pass (no streaming state, so the whole
    /// clip attends to itself). Returns `[frames × 8]` speaker activities, 10 ms per frame.
    ///
    /// Attention memory and time grow with the square of the clip length, so this is for
    /// clips of minutes, not hours; use a live session for long audio.
    ///
    /// Fails if a sample is NaN or infinite, which would otherwise turn every prediction to NaN.
    pub fn diarize_offline(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        ensure_finite_pcm(pcm)?;
        ensure_offline_len(pcm.len(), self.w.config.pad_to)?;
        let (mel, n) = self.log_mel(pcm);
        // NeMo pads the clip mel to `pad_to` before stacking; the pad groups' embeddings feed
        // the convolution's edge tap.
        let feat_len = n.div_ceil(self.w.config.pad_to) * self.w.config.pad_to;
        let mut padded = mel;
        padded.resize(feat_len * self.w.config.n_mel_bins, 0.0);
        let (emb, _) = self.embed(&padded, feat_len);
        let mut preds = self.predict(&emb, n);
        // Only the valid mel frames are real; the pad groups' sub-frames are not returned.
        preds.truncate(n * self.w.config.n_spk);
        Ok(preds)
    }

    /// Streaming diarization of a whole clip, chunked exactly as NeMo's feature loader
    /// chunks it. Returns `[frames × 8]`.
    pub fn diarize_streaming(&self, pcm: &[f32], params: StreamingParams) -> Result<Vec<f32>> {
        ensure_finite_pcm(pcm)?;
        let (mel, n) = self.log_mel(pcm);
        self.new_stream(params)?.diarize_features(&mel, n)
    }
}

fn ensure_offline_len(n_samples: usize, pad_to: usize) -> Result<()> {
    // The padded total is what attention runs over (at most one group past the valid count).
    let frames = padded_enc_frames(n_samples / HOP_LEN, pad_to);
    ensure!(
        frames <= MAX_OFFLINE_FRAMES,
        "{frames} encoder frames is past the {MAX_OFFLINE_FRAMES} (10 minutes) the offline \
         pass can attend over; use a live session or diarize_streaming for long audio"
    );
    Ok(())
}

/// Largest PCM magnitude accepted. Real audio is within +-1 (or +-32768 at 16-bit scale); the
/// mel overflows to infinity only near `f32::MAX`, so this leaves room for any real signal and
/// refuses reinterpreted garbage bytes deterministically.
const MAX_ABS_PCM: f32 = 1e9;

/// Largest |log-mel| accepted by the feature entry points. Real values stay within about
/// [-17, 46] even for PCM at [`MAX_ABS_PCM`]; the embedder overflows to NaN from about 1e9.
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

/// Reusable workspace for [`encoder_layer_forward`]: one set of q/k/v/context/score
/// buffers plus the small row buffers, allocated once per
/// [`Nemotron3Model::predict_with_taps`] call instead of per layer. Every buffer is fully
/// overwritten on each use (q/k/v by `gemv`, context and scores elementwise), so reuse needs
/// no re-zeroing, and the peak transient is unchanged.
struct EncoderScratch {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    ctx: Vec<f32>,
    scores: Vec<f32>,
    normed: Vec<f32>,
    att: Vec<f32>,
    ff: Vec<f32>,
    ffo: Vec<f32>,
}

impl EncoderScratch {
    fn new(t: usize, d: usize, n_ff: usize) -> Self {
        Self {
            q: vec![0.0f32; t * d],
            k: vec![0.0f32; t * d],
            v: vec![0.0f32; t * d],
            ctx: vec![0.0f32; t * d],
            scores: vec![0.0f32; t],
            normed: vec![0.0f32; d],
            att: vec![0.0f32; d],
            ff: vec![0.0f32; n_ff],
            ffo: vec![0.0f32; d],
        }
    }
}

/// One pre-LN Transformer block: `x += Attn(LN1(x))` with RoPE positions `0..t`, then
/// `x += MLP(LN2(x))` with erf GELU. Bidirectional attention over the first `valid` keys;
/// queries past them (pad groups) still produce outputs, like NeMo's padding mask. `rope`
/// is the [`cpu::rope_table`] for `t` positions (built once per forward: positions restart
/// at 0 for every step, so every layer shares it).
fn encoder_layer_forward(
    scratch: &mut EncoderScratch,
    x: &mut [f32],
    l: &EncoderLayer,
    t: usize,
    valid: usize,
    c: &Nemotron3Config,
    rope: &[f32],
) {
    let d = c.n_embd;
    let heads = c.n_head;
    let dh = d / heads;
    debug_assert!(dh <= 64, "head dimension {dh} exceeds accumulator size 64");
    let scale = 1.0 / (dh as f64).sqrt();

    let EncoderScratch {
        q,
        k,
        v,
        ctx,
        scores,
        normed,
        att,
        ff,
        ffo,
    } = scratch;
    for i in 0..t {
        normed.copy_from_slice(&x[i * d..(i + 1) * d]);
        cpu::layer_norm_inplace(normed, &l.ln1_w, &l.ln1_b, c.eps);
        l.q_w.gemv(normed, &mut q[i * d..(i + 1) * d]);
        l.k_w.gemv(normed, &mut k[i * d..(i + 1) * d]);
        l.v_w.gemv(normed, &mut v[i * d..(i + 1) * d]);
        // RoPE positions restart at 0 for every forward over `[cache, fifo, chunk, lookahead]`;
        // offline that window is the whole clip, so position is the frame index.
        for h in 0..heads {
            cpu::apply_rope_from_table(&mut q[i * d + h * dh..i * d + (h + 1) * dh], i, dh, rope);
            cpu::apply_rope_from_table(&mut k[i * d + h * dh..i * d + (h + 1) * dh], i, dh, rope);
        }
    }

    for h in 0..heads {
        let off = h * dh;
        for i in 0..t {
            let qi = &q[i * d + off..i * d + off + dh];
            for (j, s) in scores.iter_mut().enumerate() {
                let kj = &k[j * d + off..j * d + off + dh];
                let dot: f64 = qi.iter().zip(kj).map(|(&a, &b)| a as f64 * b as f64).sum();
                *s = (dot * scale) as f32;
            }
            // Pad keys are masked out of every query, valid or pad, like NeMo's padding mask.
            for s in &mut scores[valid.min(t)..t] {
                *s = f32::NEG_INFINITY;
            }
            cpu::softmax_inplace(scores);
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

    for i in 0..t {
        let row = &mut x[i * d..(i + 1) * d];
        l.o_w.gemv(&ctx[i * d..(i + 1) * d], att);
        cpu::add_inplace(att, &l.o_b);
        cpu::add_inplace(row, att);

        normed.copy_from_slice(row);
        cpu::layer_norm_inplace(normed, &l.ln2_w, &l.ln2_b, c.eps);
        l.up_w.gemv(normed, ff);
        cpu::add_inplace(ff, &l.up_b);
        cpu::gelu_erf_inplace(ff);
        l.down_w.gemv(ff, ffo);
        cpu::add_inplace(ffo, &l.down_b);
        cpu::add_inplace(row, ffo);
    }
}

impl Nemotron3Weights {
    /// Stack groups of 8 mel frames (zero-padding the tail) and project: on the
    /// accelerator when one is set and takes it, else on the CPU.
    fn embed_for(&self, mel: &[f32]) -> (Vec<f32>, usize) {
        let n = mel.len() / self.config.n_mel_bins;
        let t = enc_frames(n);
        let want = t * self.config.n_embd;
        if let Some(emb) = self.accelerated_checked(AccelStage::Embed, want, |a| a.embed(mel, n)) {
            return (emb, t);
        }
        let m = self.config.n_mel_bins;
        let mut stacked = vec![0.0f32; m * SUBSAMPLING];
        let mut emb = vec![0.0f32; want];
        for (f, dst) in emb.chunks_exact_mut(self.config.n_embd).enumerate() {
            stacked.fill(0.0);
            for s in 0..SUBSAMPLING {
                let src = f * SUBSAMPLING + s;
                if src >= n {
                    break;
                }
                stacked[s * m..(s + 1) * m].copy_from_slice(&mel[src * m..(src + 1) * m]);
            }
            self.embed_w.gemv(&stacked, dst);
        }
        (emb, t)
    }

    /// Run `call` on the accelerator if one is set. `None` means "use the CPU": nothing is set,
    /// the accelerator declined, or it failed (logged once per stage, then quiet).
    fn accelerated<T>(
        &self,
        stage: AccelStage,
        call: impl FnOnce(&dyn Nemotron3Accelerator) -> Result<Option<T>>,
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

    /// [`Self::accelerated`], plus the output-length check: a short or long output means the
    /// caller falls back to the CPU. Every `Vec<f32>` stage output goes through here so a
    /// future stage cannot forget the length check and consume a short output as valid.
    fn accelerated_checked(
        &self,
        stage: AccelStage,
        want: usize,
        call: impl FnOnce(&dyn Nemotron3Accelerator) -> Result<Option<Vec<f32>>>,
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
            tracing::warn!("nemotron3: {} {detail}", stage.label());
            // No `tracing` subscriber on the shipping mobile/FFI platforms; without this
            // the warning is invisible exactly where the fallback runs.
            eprintln!("cera-nemotron3: {} {detail}", stage.label());
        }
    }
}

// ── Streaming session ──────────────────────────────────────────────────────

/// The speaker cache, FIFO and compression flag of a [`Nemotron3Stream`].
#[derive(Debug, Clone)]
struct Nemotron3StreamState {
    /// Embeddings of the frames judged most informative, `[len × n_embd]`.
    spkcache: Vec<f32>,
    /// Their pooled speaker activities, `[len × n_spk]`.
    spkcache_preds: Vec<f32>,
    /// Whether the cache has been compressed at least once.
    spkcache_compressed: bool,
    /// Embeddings of the most recent chunks, `[len × n_embd]`.
    fifo: Vec<f32>,
    /// Their pooled speaker activities, `[len × n_spk]`.
    fifo_preds: Vec<f32>,
}

/// One streaming session: chunked diarization with a speaker cache and a FIFO.
///
/// Created per clip with [`Nemotron3Model::new_stream`]; cheap to make, but the model does the
/// work. For audio that arrives over time, [`Nemotron3Live`] wraps this with the incremental
/// mel front end.
pub struct Nemotron3Stream {
    w: Arc<Nemotron3Weights>,
    params: StreamingParams,
    state: Nemotron3StreamState,
}

impl Nemotron3Model {
    /// Start a streaming session with `params` (see [`Self::default_streaming`] and the
    /// latency presets on [`StreamingParams`]).
    pub fn new_stream(&self, params: StreamingParams) -> Result<Nemotron3Stream> {
        params.validate()?;
        Ok(Nemotron3Stream {
            w: Arc::clone(&self.w),
            params,
            state: Nemotron3StreamState::new(),
        })
    }
}

impl Nemotron3Stream {
    /// The session's streaming parameters.
    pub fn params(&self) -> &StreamingParams {
        &self.params
    }

    /// Speaker-cache embeddings, `[len × n_embd]`.
    pub fn spkcache(&self) -> &[f32] {
        &self.state.spkcache
    }

    /// Speaker-cache predictions (pooled activities), `[len × n_spk]`.
    pub fn spkcache_preds(&self) -> &[f32] {
        &self.state.spkcache_preds
    }

    /// Whether the cache has been compressed at least once.
    pub fn spkcache_compressed(&self) -> bool {
        self.state.spkcache_compressed
    }

    /// FIFO embeddings, `[len × n_embd]`.
    pub fn fifo(&self) -> &[f32] {
        &self.state.fifo
    }

    /// FIFO predictions (pooled activities), `[len × n_spk]`.
    pub fn fifo_preds(&self) -> &[f32] {
        &self.state.fifo_preds
    }

    /// Run a whole clip's features (`[n_frames × 128]`, from [`Nemotron3Model::log_mel`])
    /// through the streaming loop, chunked like NeMo's `streaming_feat_loader` over features
    /// padded to a multiple of `pad_to`. Returns `[frames × 8]` 10 ms activities; frames past
    /// the audio are zeros.
    ///
    /// The stream keeps its speaker cache and FIFO between calls, so a second clip continues
    /// from the first; start a new stream ([`Nemotron3Model::new_stream`]) per clip.
    pub fn diarize_features(&mut self, mel: &[f32], n_frames: usize) -> Result<Vec<f32>> {
        self.diarize_features_with(mel, n_frames, &mut |_, _, _| {})
    }

    /// Like [`Self::diarize_features`], but chunked as a live stream sees the audio: the
    /// features are not padded to `pad_to`, so the final chunk ends at the last real frame.
    /// [`Nemotron3Live`] produces exactly these predictions.
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
        on_step: &mut dyn FnMut(usize, &Nemotron3Stream, &[f32]),
    ) -> Result<Vec<f32>> {
        let pad_to = self.w.config.pad_to;
        let feat_len = n_frames.checked_next_multiple_of(pad_to).with_context(|| {
            format!("n_frames {n_frames} cannot be padded to a multiple of {pad_to}")
        })?;
        self.chunk_loop(mel, n_frames, feat_len, on_step)
    }

    /// NeMo's `streaming_feat_loader` over `feat_len` frames (`>= n_frames`; the tail past
    /// `n_frames` is zero padding that the embedder turns into zero groups).
    fn chunk_loop(
        &mut self,
        mel: &[f32],
        n_frames: usize,
        feat_len: usize,
        on_step: &mut dyn FnMut(usize, &Nemotron3Stream, &[f32]),
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

        // About one 8-speaker row per mel frame; the chunk portions partition the clip, so
        // this pre-size is exact up to lookahead rounding at chunk edges.
        let mut total = Vec::with_capacity(feat_len * c.n_spk);
        let mut chunk = Vec::new();
        let (mut stt, mut end, mut idx) = (0usize, 0usize, 0usize);
        while end < feat_len {
            end = (stt + chunk_feat).min(feat_len);
            let right_offset = (self.params.right_context * ss).min(feat_len - end);
            let n_feat = end + right_offset - stt;
            let valid = n_frames.saturating_sub(stt).min(n_feat);
            chunk.clear();
            chunk.resize(n_feat * nm, 0.0);
            if valid > 0 {
                // With `valid == 0` the chunk starts in the padding past the last real frame.
                chunk[..valid * nm].copy_from_slice(&mel[stt * nm..(stt + valid) * nm]);
            }
            stt = end;
            let preds = self.step(&chunk, n_feat, valid, right_offset)?;
            on_step(idx, self, &preds);
            total.extend(preds);
            idx += 1;
        }
        Ok(total)
    }

    /// One streaming step.
    ///
    /// `feats` is `[n_feat × 128]` mel: the chunk with `right_offset` frames of lookahead after
    /// it (in mel frames, as in NeMo's loader). `valid_feat <= n_feat` says how many leading
    /// frames are real audio; a live caller passes `valid_feat == n_feat`. The rest
    /// (end-of-clip padding) embeds to zero groups, which join the FIFO like NeMo's. At least
    /// one frame must be valid overall: `valid_feat` may be 0 only once earlier steps left a
    /// non-empty cache or FIFO (or for the `n_feat == 0` no-op); anything else is refused.
    ///
    /// Returns the chunk's predictions `[chunk groups × 8 sub-frames × 8]`, where the chunk is
    /// the slice's groups minus the lookahead groups. Updates the cache and FIFO.
    pub fn step(
        &mut self,
        feats: &[f32],
        n_feat: usize,
        valid_feat: usize,
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

        // Groups NeMo would see: the slice stacks as-is (the clip mel is already padded
        // to `pad_to` by the chunk loop); the lookahead rounds up to whole groups.
        let groups_total = enc_frames(n_feat);
        let rc = right_offset.div_ceil(ss);
        ensure!(
            groups_total >= rc,
            "chunk of {groups_total} groups is shorter than its {rc}-group lookahead"
        );
        let chunk_len = groups_total - rc;
        // The same window bound `StreamingParams::validate` puts on the configuration, checked
        // against what this call really attends over (the caller chooses `n_feat`).
        let window = (self.state.spkcache.len() + self.state.fifo.len()) / d + groups_total;
        ensure!(
            window <= MAX_OFFLINE_FRAMES,
            "step window of {window} encoder frames exceeds {MAX_OFFLINE_FRAMES}"
        );
        // `validate` related `max_index` to the configured chunk; this call may be longer, and
        // a real flat index at or past `max_index` would read as a disabled cache slot. Only
        // the rows `update` can compress count (cache, FIFO and the chunk itself, not the
        // lookahead), the same terms `validate` uses.
        let candidates = (self.state.spkcache.len() + self.state.fifo.len()) / d + chunk_len;
        ensure!(
            self.params.max_index >= MAX_SPEAKERS * (candidates + self.params.sil_frames_per_spk),
            "max_index {} lies inside the flat index range of a {candidates}-frame step",
            self.params.max_index
        );
        // `predict` asserts at least one valid sub-frame over a non-empty window; an
        // all-padding first step (empty cache and FIFO, `valid_feat == 0`) would trip it,
        // so refuse exactly that case here, before any state is touched. The `n_feat == 0`
        // no-op (`groups_total == 0`) and later all-padding steps (non-empty history) pass.
        ensure!(
            valid_feat > 0
                || groups_total == 0
                || !(self.state.spkcache.is_empty() && self.state.fifo.is_empty()),
            "step: no valid frames for {groups_total} groups \
             (valid_feat is 0 and the cache and FIFO are empty)",
        );

        let (chunk_all, t_all) = if valid_feat == 0 {
            (vec![0.0; groups_total * d], groups_total)
        } else {
            w.embed_for(&feats[..valid_feat * c.n_mel_bins])
        };
        // `embed_for` pads the valid mel on its own; pad the result out to the slice's groups
        // (the lookahead's own padding is zeros either way).
        debug_assert!(t_all <= groups_total);
        let mut chunk_all = chunk_all;
        chunk_all.resize(groups_total * d, 0.0);
        let model = Nemotron3Model { w: w.clone() };

        let n_sc = self.state.spkcache.len() / d;
        let n_fifo = self.state.fifo.len() / d;
        let mut concat = Vec::with_capacity((n_sc + n_fifo + groups_total) * d);
        concat.extend_from_slice(&self.state.spkcache);
        concat.extend_from_slice(&self.state.fifo);
        concat.extend_from_slice(&chunk_all);
        // Cached groups count 8 valid sub-frames each; the slice counts its valid mel frames.
        let valid_sub = (n_sc + n_fifo) * ss + valid_feat;
        let hires = model.predict(&concat, valid_sub);
        // The cache scores pooled (average of 8) probabilities, like NeMo's `downsample_preds`.
        let pooled = pool_hires(&hires, s);

        let base = n_sc + n_fifo;
        let chunk_preds = hires[base * ss * s..(base + chunk_len) * ss * s].to_vec();
        Ok(self.state.update(
            &self.params,
            &w.silence_embeds,
            (d, s),
            &pooled,
            &chunk_all,
            (n_sc, n_fifo, chunk_len),
            chunk_preds,
        ))
    }
}

/// Average each non-overlapping window of 8 sub-frames: `[t * 8 x s]` to `[t x s]`.
fn pool_hires(hires: &[f32], s: usize) -> Vec<f32> {
    assert_eq!(hires.len() % (SUBSAMPLING * s), 0);
    let t = hires.len() / (SUBSAMPLING * s);
    let mut out = vec![0.0f32; t * s];
    for (f, dst) in out.chunks_exact_mut(s).enumerate() {
        for k in 0..SUBSAMPLING {
            let row = &hires[(f * SUBSAMPLING + k) * s..(f * SUBSAMPLING + k + 1) * s];
            for (o, &v) in dst.iter_mut().zip(row) {
                *o += v;
            }
        }
        for o in dst.iter_mut() {
            *o /= SUBSAMPLING as f32;
        }
    }
    out
}

impl Nemotron3StreamState {
    fn new() -> Self {
        Self {
            spkcache: Vec::new(),
            spkcache_preds: Vec::new(),
            spkcache_compressed: false,
            fifo: Vec::new(),
            fifo_preds: Vec::new(),
        }
    }

    /// NeMo's `streaming_update` (synchronous eval path) after a forward over
    /// `[spkcache, fifo, chunk]`. `pooled` is that forward's pooled `[(n_sc + n_fifo + groups)
    /// x s]` sigmoid output and `chunk_emb` the chunk's stacked embeddings; `rows` is
    /// `(n_sc, n_fifo, chunk_len)`. Appends the chunk to the FIFO, pops into the cache when
    /// the FIFO overflows, and compresses the cache when it does. Returns the chunk's own
    /// 10 ms predictions (passed in; the update only moves state).
    ///
    /// There is no silence profile: the checkpoint's silence embedding is learned, so the
    /// returned state never carries one (NeMo keeps the fields at their initial zeros).
    #[allow(clippy::too_many_arguments)]
    fn update(
        &mut self,
        p: &StreamingParams,
        silence: &[f32],
        (d, s): (usize, usize),
        pooled: &[f32],
        chunk_emb: &[f32],
        (n_sc, n_fifo, chunk_len): (usize, usize, usize),
        chunk_preds: Vec<f32>,
    ) -> Vec<f32> {
        let (fifo_cap, update_period, cache_cap) = (p.fifo_len, p.update_period, p.spkcache_len);
        // The forward covers [spkcache, fifo, chunk]: every slice below lands inside it.
        debug_assert!((n_sc + n_fifo) * s <= pooled.len());
        debug_assert!(chunk_len * d <= chunk_emb.len());
        let chunk_slice = &chunk_emb[..chunk_len * d];

        self.fifo.extend_from_slice(chunk_slice);
        self.fifo_preds.clear();
        self.fifo_preds
            .extend_from_slice(&pooled[n_sc * s..(n_sc + n_fifo) * s]);
        self.fifo_preds
            .extend_from_slice(&pooled[(n_sc + n_fifo) * s..(n_sc + n_fifo + chunk_len) * s]);

        if n_fifo + chunk_len > fifo_cap {
            let pop = update_period
                .max((chunk_len + n_fifo).saturating_sub(fifo_cap))
                .min(n_fifo + chunk_len);

            let pop_embs = self.fifo[..pop * d].to_vec();
            let pop_preds = self.fifo_preds[..pop * s].to_vec();
            self.fifo.drain(..pop * d);
            self.fifo_preds.drain(..pop * s);

            self.spkcache.extend_from_slice(&pop_embs);
            if self.spkcache_compressed {
                self.spkcache_preds.extend_from_slice(&pop_preds);
            } else {
                // Before the first compression, the cache's own predictions come from this
                // step's forward.
                self.spkcache_preds.clear();
                self.spkcache_preds.extend_from_slice(&pooled[..n_sc * s]);
                self.spkcache_preds.extend_from_slice(&pop_preds);
            }
            let cache_rows = self.spkcache.len() / d;
            if cache_rows > cache_cap {
                let (emb, pr) =
                    compress_spkcache(p, s, d, &self.spkcache, &self.spkcache_preds, silence);
                self.spkcache = emb;
                self.spkcache_preds = pr;
                self.spkcache_compressed = true;
            }
        }
        chunk_preds
    }
}

/// NeMo's `_compress_spkcache` (eval path): keep the `spkcache_len` most informative of the
/// `n_frames > spkcache_len` rows, ordered back into frame order. Frames whose topk slot is
/// disabled (a `-inf` score, or one of the padded silence frames) become the learned silence
/// embedding with zero predictions.
fn compress_spkcache(
    p: &StreamingParams,
    s: usize,
    d: usize,
    emb: &[f32],
    preds: &[f32],
    silence: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let n_frames = emb.len() / d;
    debug_assert_eq!(preds.len(), n_frames * s);
    let budget = p.spkcache_len / s - p.sil_frames_per_spk;
    let strong_n = ((budget as f32 * p.strong_boost_rate).floor() as usize).min(n_frames);
    let weak_n = ((budget as f32 * p.weak_boost_rate).floor() as usize).min(n_frames);
    let min_pos = (budget as f32 * p.min_pos_scores_rate).floor() as usize;

    let mut scores = log_pred_scores(preds, n_frames, s, p.pred_score_threshold);
    disable_low_scores(&mut scores, preds, n_frames, s, min_pos);
    // NeMo gates on `> 0`; adding 0.0 to `-inf` is still `-inf`, but keep the gate so a
    // future NaN threshold cannot ride in through here.
    if p.scores_boost_latest > 0.0 {
        for row in scores.chunks_exact_mut(s).skip(p.spkcache_len) {
            for v in row.iter_mut() {
                *v += p.scores_boost_latest;
            }
        }
    }
    boost_topk(&mut scores, n_frames, s, strong_n, 2.0);
    boost_topk(&mut scores, n_frames, s, weak_n, 1.0);

    // Silence frames (score +inf for every speaker) join the flat topk; each speaker's copy can
    // win its own slot, which is how every speaker keeps a silence frame.
    let n_sil = p.sil_frames_per_spk;
    let scored = n_frames + n_sil;
    let mut flat = vec![0.0f32; scored * s];
    for (sp, dst) in flat.chunks_exact_mut(scored).enumerate() {
        for (f, o) in dst.iter_mut().enumerate() {
            *o = if f < n_frames {
                scores[f * s + sp]
            } else {
                f32::INFINITY
            };
        }
    }
    // Highest `spkcache_len` flat slots; `-inf` scores become the `max_index` placeholder.
    let mut topk = topk_desc(&flat, p.spkcache_len);
    for i in topk.iter_mut() {
        if flat[*i] == f32::NEG_INFINITY {
            *i = p.max_index;
        }
    }
    topk.sort_unstable();

    let mut out_emb = vec![0.0f32; p.spkcache_len * d];
    let mut out_preds = vec![0.0f32; p.spkcache_len * s];
    for (o, &i) in topk.iter().enumerate() {
        let disabled = i == p.max_index;
        let frame = i % scored;
        // The padded silence frames (and the `-inf` slots) gather the silence embedding and
        // zero predictions; a placeholder index that survives the remainder gathers frame 0
        // first, exactly like NeMo, then is overwritten below.
        let gather = if disabled { 0 } else { frame.min(n_frames - 1) };
        let disabled = disabled || frame >= n_frames;
        if disabled {
            out_emb[o * d..(o + 1) * d].copy_from_slice(silence);
        } else {
            out_emb[o * d..(o + 1) * d].copy_from_slice(&emb[gather * d..(gather + 1) * d]);
            out_preds[o * s..(o + 1) * s].copy_from_slice(&preds[gather * s..(gather + 1) * s]);
        }
    }
    (out_emb, out_preds)
}

/// NeMo's `_get_log_pred_scores`: high for a confident prediction of non-overlapped speech.
fn log_pred_scores(preds: &[f32], n_frames: usize, s: usize, threshold: f32) -> Vec<f32> {
    let mut scores = vec![0.0f32; n_frames * s];
    for f in 0..n_frames {
        let row = &preds[f * s..(f + 1) * s];
        let mut sum = 0.0f32;
        for &p in row {
            sum += (1.0 - p).max(threshold).ln();
        }
        for (sp, o) in scores[f * s..(f + 1) * s].iter_mut().enumerate() {
            let p = row[sp];
            *o = p.max(threshold).ln() - (1.0 - p).max(threshold).ln() + sum - 0.5f32.ln();
        }
    }
    scores
}

/// NeMo's `_disable_low_scores`: non-speech (`p <= 0.5`) always scores `-inf`; non-positive
/// scores (usually overlapped speech) do too once a speaker has at least `min_pos`
/// positive-scored frames.
fn disable_low_scores(
    scores: &mut [f32],
    preds: &[f32],
    n_frames: usize,
    s: usize,
    min_pos: usize,
) {
    assert_eq!(scores.len(), n_frames * s);
    assert_eq!(preds.len(), n_frames * s);
    for sp in 0..s {
        let mut n_pos = 0usize;
        for f in 0..n_frames {
            if scores[f * s + sp] > 0.0 {
                n_pos += 1;
            }
        }
        let gate = n_pos >= min_pos;
        for f in 0..n_frames {
            let i = f * s + sp;
            let speech = preds[i] > 0.5;
            if !speech || (gate && scores[i] <= 0.0) {
                scores[i] = f32::NEG_INFINITY;
            }
        }
    }
}

/// NeMo's `_boost_topk_scores` with `offset` 0.5: add `-scale * ln(0.5)` to the `k` highest
/// scores of every speaker. `-inf` entries keep `-inf` (and may be selected, like torch's
/// topk, when fewer than `k` entries are finite).
fn boost_topk(scores: &mut [f32], n_frames: usize, s: usize, k: usize, scale: f32) {
    if k == 0 {
        return;
    }
    let bonus = -scale * 0.5f32.ln();
    let mut col = vec![0.0f32; n_frames];
    for sp in 0..s {
        for (f, o) in col.iter_mut().enumerate() {
            *o = scores[f * s + sp];
        }
        for &f in &topk_desc(&col, k.min(n_frames)) {
            scores[f * s + sp] += bonus;
        }
    }
}

/// Indices of the `k` largest values, ties broken by ascending index (like the golden's
/// `torch.topk(sorted=False)` trace on this data); NaN ranks last.
fn topk_desc(vals: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&a, &b| {
        let (x, y) = (vals[a], vals[b]);
        y.partial_cmp(&x)
            .unwrap_or_else(|| x.is_nan().cmp(&y.is_nan()))
            .then(a.cmp(&b))
    });
    idx.truncate(k.min(vals.len()));
    idx
}

// ── Incremental mel and live diarization ─────────────────────────────────────

/// Incremental log-mel front end: the frames [`Nemotron3Model::log_mel`] computes, one `push`
/// at a time, bit-identical however the audio is split.
pub struct MelStream {
    /// The model's weights, for its accelerator (absent for the weight-free test streams).
    weights: Option<Arc<Nemotron3Weights>>,
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
    fn new(w: &Arc<Nemotron3Weights>) -> Self {
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
/// It chunks the incremental mel the way [`Nemotron3Stream::diarize_features_unpadded`] does
/// and gives bit-identical predictions to it. A chunk is computed once `chunk_len` frames plus
/// `right_context` frames of lookahead exist (see [`Self::latency_frames`]); predictions are
/// never revised. [`Self::finish`] flushes the tail.
pub struct Nemotron3Live {
    stream: Nemotron3Stream,
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

impl Nemotron3Model {
    /// Start live diarization with `params` (see [`Self::default_streaming`]).
    pub fn new_live(&self, params: StreamingParams) -> Result<Nemotron3Live> {
        Ok(Nemotron3Live {
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

impl Nemotron3Live {
    /// Worst-case delay in encoder frames (80 ms each): a chunk's first frame waits for the
    /// rest of the chunk and its lookahead, `chunk_len + right_context` (NeMo's definition of
    /// the preset's latency), on top of the 16 ms (256 samples) the mel front end needs after a
    /// frame's center.
    pub fn latency_frames(&self) -> usize {
        // Saturating: the params may be hand-made (`with_chunking` skips validation).
        self.stream
            .params
            .chunk_len
            .saturating_add(self.stream.params.right_context)
    }

    /// Prediction frames returned so far (10 ms each).
    pub fn frames_emitted(&self) -> usize {
        self.emitted
    }

    /// Mel frames currently held for the next chunks (whatever is waiting for lookahead).
    /// Bounded by the streaming parameters, not by the stream's length.
    pub fn buffered_frames(&self) -> usize {
        self.rows.len() / self.stream.w.config.n_mel_bins
    }

    /// The underlying streaming state (speaker cache and FIFO).
    pub fn stream(&self) -> &Nemotron3Stream {
        &self.stream
    }

    /// Feed mono 16 kHz PCM of any length. Returns the predictions that became final,
    /// `[k x 8]` for the next `k` 10 ms frames in order (possibly empty). Fails if the stream
    /// has finished or a sample is NaN, infinite or beyond 1e9 in magnitude (nothing is
    /// consumed in that case).
    ///
    /// When the call completes a chunk it runs the model synchronously, and on the CPU that can
    /// take longer than the audio it covers: cost grows with the square of the window (cache,
    /// FIFO, chunk and lookahead together), so size a worker for that, not for the audio rate,
    /// and never call this from an audio callback.
    pub fn push_audio(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        ensure!(!self.finished, "Nemotron3Live: push_audio after finish");
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
        let mut out = Vec::new();
        loop {
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
            let n_feat = end + right_offset - self.stt;
            let lo = (self.stt - self.rows_start) * nm;
            let preds = self.stream.step(
                &self.rows[lo..lo + n_feat * nm],
                n_feat,
                n_feat,
                right_offset,
            )?;
            self.stt = end;
            self.emitted += preds.len() / self.stream.w.config.n_spk;
            out.extend(preds);

            // Mel frames before the next chunk are no longer needed (there is no left context).
            if self.stt > self.rows_start {
                self.rows.drain(..(self.stt - self.rows_start) * nm);
                self.rows_start = self.stt;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_counts_are_ceil_divs() {
        for n in 0..2000usize {
            assert_eq!(enc_frames(n), n.div_ceil(8), "n = {n}");
            assert_eq!(padded_enc_frames(n, 16), n.div_ceil(16) * 2, "n = {n}");
        }
        // The golden clip: 1537 mel frames -> 193 valid groups, 194 with the mel pad.
        assert_eq!(enc_frames(1537), 193);
        assert_eq!(padded_enc_frames(1537, 16), 194);
        assert_eq!(padded_enc_frames(1536, 16), 192);
        assert_eq!(padded_enc_frames(0, 16), 0);
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
            chunk_len: 9,
            right_context: 4,
            fifo_len: 264,
            spkcache_len: 264,
            update_period: 222,
            sil_frames_per_spk: 1,
            pred_score_threshold: 0.25,
            scores_boost_latest: 0.05,
            sil_threshold: 0.2,
            strong_boost_rate: 0.75,
            weak_boost_rate: 1.5,
            min_pos_scores_rate: 0.5,
            max_index: 99_999,
        };
        assert!(ok.validate().is_ok());
        // A cache of 7 cannot even hold 8 speakers x 1 silence frame (a cache of 8 is a
        // degenerate but valid zero speech budget).
        assert!(ok.with_chunking(9, 4, 264, 7, 222).validate().is_err());
        assert!(ok.with_chunking(9, 4, 264, 8, 222).validate().is_ok());
        assert!(ok.with_chunking(0, 4, 264, 264, 222).validate().is_err());
        assert!(ok.with_chunking(9, 4, 264, 264, 0).validate().is_err());
        // The valid side of the positivity guards.
        assert!(ok.with_chunking(1, 4, 264, 264, 222).validate().is_ok());
        // Each length is capped on its own (memory is quadratic downstream).
        assert!(
            ok.with_chunking((1 << 20) + 1, 4, 264, 264, 222)
                .validate()
                .is_err()
        );
        // .. and the attended window is capped as a whole (9 + 4 + 8000 + 264 > 7500).
        assert!(ok.with_chunking(9, 4, 8000, 264, 222).validate().is_err());
        for bad in [0.0, -0.1, 1.5, f32::NAN, f32::INFINITY] {
            let mut p = ok.clone();
            p.pred_score_threshold = bad;
            assert!(p.validate().is_err(), "threshold {bad}");
        }
        type FloatGuard = fn(&mut StreamingParams, f32);
        let float_guards: [(&str, FloatGuard); 5] = [
            ("scores_boost_latest", |p, v| {
                p.scores_boost_latest = v;
            }),
            ("sil_threshold", |p, v| {
                p.sil_threshold = v;
            }),
            ("strong_boost_rate", |p, v| {
                p.strong_boost_rate = v;
            }),
            ("weak_boost_rate", |p, v| {
                p.weak_boost_rate = v;
            }),
            ("min_pos_scores_rate", |p, v| {
                p.min_pos_scores_rate = v;
            }),
        ];
        for (name, set) in float_guards {
            for bad in [-1.0, f32::NAN] {
                let mut p = ok.clone();
                set(&mut p, bad);
                assert!(p.validate().is_err(), "{name} {bad}");
            }
        }
        // `min_pos_scores_rate` is also capped at 1.
        let mut p = ok.clone();
        p.min_pos_scores_rate = 1.5;
        assert!(p.validate().is_err());
        let mut p = ok.clone();
        p.max_index = 8;
        assert!(p.validate().is_err());
        // The ok fixture's flat range is 8 * (264 + 264 + 9 + 1) = 4304: straddle it so a
        // dropped term fails.
        let mut p = ok.clone();
        p.max_index = 4303;
        assert!(p.validate().is_err());
        let mut p = ok.clone();
        p.max_index = 4304;
        assert!(p.validate().is_ok());
    }

    #[test]
    fn window_and_latency_helpers_saturate_on_hostile_lengths() {
        let hostile = StreamingParams {
            chunk_len: usize::MAX,
            right_context: 1,
            fifo_len: usize::MAX,
            spkcache_len: 1,
            update_period: 1,
            sil_frames_per_spk: 0,
            pred_score_threshold: 0.25,
            scores_boost_latest: 0.0,
            sil_threshold: 0.0,
            strong_boost_rate: 0.0,
            weak_boost_rate: 0.0,
            min_pos_scores_rate: 0.0,
            max_index: usize::MAX,
        };
        // Wrapping math would return 1 here (and panic in debug); saturating pins the max.
        assert_eq!(hostile.window_frames(), usize::MAX);
        let w = Arc::new(test_weights());
        let live = Nemotron3Live {
            stream: Nemotron3Stream {
                w: Arc::clone(&w),
                params: hostile,
                state: Nemotron3StreamState::new(),
            },
            mel: MelStream::new(&w),
            rows: Vec::new(),
            rows_start: 0,
            total_mel: 0,
            stt: 0,
            emitted: 0,
            finished: false,
        };
        assert_eq!(live.latency_frames(), usize::MAX);
    }

    fn params(spkcache_len: usize) -> StreamingParams {
        StreamingParams {
            chunk_len: 9,
            right_context: 4,
            fifo_len: 264,
            spkcache_len,
            update_period: 222,
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
    fn compress(preds: &[[f32; 8]], spkcache_len: usize) -> (Vec<f32>, Vec<f32>) {
        let n = preds.len();
        let emb: Vec<f32> = (0..n).map(|f| (f + 1) as f32).collect();
        let flat: Vec<f32> = preds.iter().flatten().copied().collect();
        compress_spkcache(&params(spkcache_len), 8, 1, &emb, &flat, &[-1.0])
    }

    /// Like [`compress`] with the given parameters.
    fn compress_with(p: &StreamingParams, preds: &[[f32; 8]]) -> Vec<f32> {
        let n = preds.len();
        let emb: Vec<f32> = (0..n).map(|f| (f + 1) as f32).collect();
        let flat: Vec<f32> = preds.iter().flatten().copied().collect();
        compress_spkcache(p, 8, 1, &emb, &flat, &[-1.0]).0
    }

    #[test]
    fn compression_orders_by_speaker_and_closes_each_block_with_silence() {
        // Cache 24 / 8 speakers - 1 silence = 2 frames per speaker; min_pos = floor(2 * 0.5) = 1.
        let (emb, preds) = compress(
            &[
                [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // silence: not speech for anyone
            ],
            24,
        );
        // Speaker 0: frames 0, 1, then its silence slot; speaker 1: frame 2, silence; the other
        // six speakers have only their silence slot; the unused slots are disabled and last.
        let mut want = vec![1.0, 2.0, -1.0, 3.0];
        want.resize(24, -1.0);
        assert_eq!(emb, want);
        // Disabled slots predict nothing; real frames keep their predictions.
        assert_eq!(&preds[..8], &[0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&preds[16..24], &[0.0; 8]);
        assert_eq!(&preds[24..32], &[0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(preds[32..].iter().all(|&p| p == 0.0));
    }

    #[test]
    fn overlapped_frames_are_dropped_only_once_a_speaker_has_min_pos_clean_frames() {
        // Frame 2 has speakers 0 and 1 talking at once (a non-positive score for both). The
        // threshold is floor(5 * 0.5) = 2 positive frames per speaker (cache 48, budget 5).
        //   speaker 0 has exactly 2 clean frames -> `>=` min_pos: its overlap frame is dropped;
        //   speaker 1 has 1 clean frame          -> below min_pos: its overlap frame stays.
        let (emb, _) = compress(
            &[
                [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            48,
        );
        let mut want = vec![1.0, 2.0, -1.0, 3.0, 4.0];
        want.resize(48, -1.0);
        assert_eq!(
            emb, want,
            "frame 2 must appear once (speaker 1's), not twice"
        );
    }

    #[test]
    fn newest_frames_get_the_latest_boost_once_the_cache_overflows() {
        // 20 identical frames of one speaker, room for 8 speech frames (16 slots, 8 closing
        // silences). Without a boost ties go to the oldest frames; with one, the 4 frames past
        // `spkcache_len` (embeddings 17..=20) displace the middle of the cache.
        let preds = [[0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; 20];
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
        assert_eq!(keep(&plain), (1..=8).map(|x| x as f32).collect::<Vec<_>>());
        assert_eq!(keep(&boosted), [1., 2., 3., 4., 17., 18., 19., 20.]);
    }

    #[test]
    fn boosts_add_minus_scale_times_ln_half_to_each_speakers_top_frames() {
        // One speaker with distinct scores, one silent: the strong boost (scale 2) must add
        // exactly twice the weak boost (scale 1), `-inf` must stay `-inf`, and ties must go to
        // the lowest frame index.
        let mut scores = vec![
            1.0,
            f32::NEG_INFINITY,
            3.0,
            f32::NEG_INFINITY,
            3.0,
            f32::NEG_INFINITY,
            2.0,
            f32::NEG_INFINITY,
        ];
        boost_topk(&mut scores, 4, 2, 2, 1.0);
        let ln2 = 0.5f32.ln().abs();
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[2], 3.0 + ln2);
        assert_eq!(scores[4], 3.0 + ln2);
        assert_eq!(scores[6], 2.0);
        assert!(scores[1] == f32::NEG_INFINITY && scores[3] == f32::NEG_INFINITY);
        // k = 0 (a zero per-speaker budget) is a no-op, like torch's empty topk.
        let mut scores = vec![1.0, 2.0];
        boost_topk(&mut scores, 1, 2, 0, 2.0);
        assert_eq!(scores, [1.0, 2.0]);
        // Scale 2 adds twice scale 1 (up to float rounding: the two sides round
        // through different operation orders).
        let mut a = vec![1.0, 0.5];
        let mut b = a.clone();
        boost_topk(&mut a, 1, 2, 1, 2.0);
        boost_topk(&mut b, 1, 2, 1, 1.0);
        assert!(((a[0] - 1.0) - 2.0 * (b[0] - 1.0)).abs() < 1e-6);
    }

    #[test]
    fn log_scores_reward_clean_speech_and_clamp_at_the_threshold() {
        // A clean frame scores ln(0.9 / 0.5); an overlapped one goes non-positive.
        let clean = log_pred_scores(&[0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1, 8, 0.25);
        assert!((clean[0] - (0.9f32 / 0.5).ln()).abs() < 1e-6, "{clean:?}");
        let overlap = log_pred_scores(&[0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1, 8, 0.25);
        assert!(overlap[0] < 0.0 && overlap[1] < 0.0, "{overlap:?}");
        // A zero probability clamps to the threshold instead of giving ln(0).
        let silent = log_pred_scores(&[0.0; 8], 1, 8, 0.25);
        assert!(silent.iter().all(|v| v.is_finite()), "{silent:?}");
    }

    /// A state with embedding width 1 (frame embeddings are markers) and 8 speakers.
    fn state() -> Nemotron3StreamState {
        Nemotron3StreamState::new()
    }

    /// Run `update` for a chunk of `chunk_len` frames. `rows` is the pooled predictions of the
    /// whole `[cache, fifo, chunk]` forward with one marker value per row; the chunk's 10 ms
    /// predictions pass through untouched and are returned as given.
    fn feed(
        st: &mut Nemotron3StreamState,
        p: &StreamingParams,
        chunk_len: usize,
        first_marker: f32,
        row: [f32; 8],
    ) -> Vec<f32> {
        let (n_sc, n_fifo) = (st.spkcache.len(), st.fifo.len());
        let pooled: Vec<f32> = (0..n_sc + n_fifo + chunk_len).flat_map(|_| row).collect();
        let chunk_emb: Vec<f32> = (0..chunk_len).map(|i| first_marker + i as f32).collect();
        let chunk_preds: Vec<f32> = (0..chunk_len * 8 * 8).map(|i| 100.0 + i as f32).collect();
        st.update(
            p,
            &[-1.0],
            (1, 8),
            &pooled,
            &chunk_emb,
            (n_sc, n_fifo, chunk_len),
            chunk_preds,
        )
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
        let out = feed(
            &mut st,
            &p,
            6,
            1.0,
            [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(out.len(), 6 * 8 * 8);
        assert_eq!(st.fifo, [3., 4., 5., 6.]);
        assert_eq!(st.spkcache, [1., 2.]);
        // 4 queued + 6 new overflow by 6, more than an update period: all 6 move.
        feed(
            &mut st,
            &p,
            6,
            7.0,
            [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(st.fifo.len(), 4);
        assert_eq!(st.spkcache.len(), 2 + 6);
        assert_eq!(st.fifo_preds.len(), 4 * 8);
        // A chunk that fits pops nothing.
        let p = StreamingParams { fifo_len: 100, ..p };
        let mut st = state();
        feed(
            &mut st,
            &p,
            6,
            1.0,
            [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!((st.fifo.len(), st.spkcache.len()), (6, 0));
    }

    #[test]
    fn update_appends_the_chunk_and_passes_its_predictions_through() {
        let p = params(40);
        let mut st = state();
        let chunk_len = 3;
        let pooled: Vec<f32> = (0..chunk_len * 8).map(|i| i as f32).collect();
        let chunk_emb: Vec<f32> = (0..chunk_len).map(|i| i as f32).collect();
        let chunk_preds: Vec<f32> = (0..chunk_len * 8 * 8).map(|i| 50.0 + i as f32).collect();
        let out = st.update(
            &p,
            &[-1.0],
            (1, 8),
            &pooled,
            &chunk_emb,
            (0, 0, chunk_len),
            chunk_preds.clone(),
        );
        assert_eq!(out, chunk_preds);
        // The FIFO took the chunk rows with their pooled predictions.
        assert_eq!(st.fifo, [0., 1., 2.]);
        assert_eq!(st.fifo_preds, pooled);
    }

    #[test]
    fn the_first_compression_takes_the_cache_predictions_from_the_forward() {
        let p = StreamingParams {
            fifo_len: 4,
            update_period: 2,
            ..params(16)
        };
        let mut st = state();
        for m in [1.0, 7.0, 13.0] {
            feed(&mut st, &p, 6, m, [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
        // Caches 2 + 6 + 6 = 14 rows after three pops; the fourth pop overflows 16.
        assert!(!st.spkcache_compressed);
        feed(
            &mut st,
            &p,
            6,
            19.0,
            [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert!(st.spkcache_compressed);
        assert_eq!(st.spkcache.len(), 16);
        assert_eq!(st.spkcache_preds.len(), 16 * 8);
    }

    /// Minimal weights for the refusal tests below: real config dims, empty tensors. Every
    /// guard under test fires before the forward pass, so the weights are never read (the
    /// mel tables are full-size only because `MelFrameComputer` asserts their shape).
    fn test_weights() -> Nemotron3Weights {
        let empty = || MmapWeight::from_owned_f32(Vec::new(), 0, 0);
        Nemotron3Weights {
            config: Nemotron3Config {
                n_layer: 0,
                n_embd: 8,
                n_ff: 8,
                n_head: 2,
                eps: 1e-5,
                n_mel_bins: 4,
                tf_d: 4,
                n_spk: 2,
                subsampling: SUBSAMPLING,
                rope_theta: 10_000.0,
                rope_max_pos: 0,
                pad_to: 16,
            },
            streaming: params(264),
            embed_w: empty(),
            input_norm_w: Vec::new(),
            input_norm_b: Vec::new(),
            final_norm_w: Vec::new(),
            final_norm_b: Vec::new(),
            layers: Vec::new(),
            proj_w: empty(),
            proj_b: Vec::new(),
            up_w: empty(),
            up_b: Vec::new(),
            dense_w: empty(),
            dense_b: Vec::new(),
            out_w: empty(),
            out_b: Vec::new(),
            silence_embeds: Vec::new(),
            window: vec![0.0; N_FFT],
            mel_fb: vec![0.0; 4 * N_FFT_BINS],
            accel: OnceLock::new(),
            accel_warned: AccelWarned::default(),
        }
    }

    fn test_stream(params: StreamingParams) -> Nemotron3Stream {
        Nemotron3Stream {
            w: Arc::new(test_weights()),
            params,
            state: Nemotron3StreamState::new(),
        }
    }

    #[test]
    fn step_refuses_misshaped_features() {
        let mut st = test_stream(params(264));
        let err = st.step(&[0.0; 31], 8, 8, 0).unwrap_err();
        assert!(
            err.to_string().contains("feats is not [n_feat x 4]"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn step_refuses_valid_past_n_feat() {
        let mut st = test_stream(params(264));
        let err = st.step(&[0.0; 8 * 4], 8, 9, 0).unwrap_err();
        assert!(
            err.to_string().contains("valid_feat 9 > n_feat 8"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn step_refuses_non_finite_mel_before_touching_state() {
        let mut st = test_stream(params(264));
        let mut feats = vec![0.0; 8 * 4];
        feats[5] = f32::NAN;
        let err = st.step(&feats, 8, 8, 0).unwrap_err();
        assert!(
            err.to_string().contains("non-finite or out-of-range mel"),
            "unexpected error: {err:#}"
        );
        assert!(st.fifo().is_empty() && st.spkcache().is_empty());
    }

    #[test]
    fn step_refuses_lookahead_longer_than_the_chunk() {
        let mut st = test_stream(params(264));
        // One group of 8 frames with 9 frames of lookahead (2 groups).
        let err = st.step(&[0.0; 8 * 4], 8, 8, 9).unwrap_err();
        assert!(
            err.to_string()
                .contains("shorter than its 2-group lookahead"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn step_refuses_a_window_past_the_offline_cap() {
        let mut st = test_stream(params(264));
        let n_feat = 7_501 * SUBSAMPLING;
        let err = st
            .step(&vec![0.0; n_feat * 4], n_feat, n_feat, 0)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("step window of 7501 encoder frames exceeds 7500"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn step_refuses_max_index_inside_the_step_range() {
        let mut p = params(264);
        p.max_index = 0;
        let mut st = test_stream(p);
        let err = st.step(&[0.0; 8 * 4], 8, 8, 0).unwrap_err();
        assert!(
            err.to_string()
                .contains("lies inside the flat index range of a 1-frame step"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn step_refuses_an_all_padding_first_step() {
        // Empty cache and FIFO with `valid_feat == 0` used to reach `predict` with
        // `valid_sub == 0` and panic; now it is a clean refusal naming the guard.
        let mut st = test_stream(params(264));
        let err = st.step(&[0.0; 8 * 4], 8, 0, 0).unwrap_err();
        assert!(
            err.to_string()
                .contains("step: no valid frames for 1 groups"),
            "unexpected error: {err:#}"
        );
        assert!(st.fifo().is_empty() && st.spkcache().is_empty());
    }

    #[test]
    fn step_accepts_the_empty_no_op() {
        let mut st = test_stream(params(264));
        assert_eq!(st.step(&[], 0, 0, 0).unwrap(), Vec::<f32>::new());
    }

    struct ErrAccel;
    impl Nemotron3Accelerator for ErrAccel {
        fn embed(&self, _mel: &[f32], _n: usize) -> Result<Option<Vec<f32>>> {
            Err(anyhow::anyhow!("embed boom"))
        }
        fn predict(&self, _emb: &[f32], _valid: usize) -> Result<Option<Vec<f32>>> {
            Err(anyhow::anyhow!("predict boom"))
        }
    }

    struct ShortAccel;
    impl Nemotron3Accelerator for ShortAccel {
        fn embed(&self, _mel: &[f32], _n: usize) -> Result<Option<Vec<f32>>> {
            Ok(Some(vec![0.0; 3]))
        }
        fn predict(&self, _emb: &[f32], _valid: usize) -> Result<Option<Vec<f32>>> {
            Ok(Some(vec![0.0; 3]))
        }
    }

    struct DeclineAccel;
    impl Nemotron3Accelerator for DeclineAccel {
        fn embed(&self, _mel: &[f32], _n: usize) -> Result<Option<Vec<f32>>> {
            Ok(None)
        }
        fn predict(&self, _emb: &[f32], _valid: usize) -> Result<Option<Vec<f32>>> {
            Ok(None)
        }
    }

    #[test]
    fn accelerator_errors_fall_back_to_cpu_and_warn_once_per_stage() {
        let w = test_weights();
        assert!(w.accel.set(Arc::new(ErrAccel)).is_ok());
        let call = |a: &dyn Nemotron3Accelerator| a.embed(&[0.0; 8], 2);
        assert!(w.accelerated_checked(AccelStage::Embed, 4, call).is_none());
        assert!(w.accel_warned.embed.load(Ordering::Relaxed));
        // A repeat failure still falls back; the latch stays set (warned once).
        assert!(w.accelerated_checked(AccelStage::Embed, 4, call).is_none());
        // Stages latch independently: predict has not warned yet.
        assert!(!w.accel_warned.predict.load(Ordering::Relaxed));
        assert!(
            w.accelerated_checked(AccelStage::Predict, 4, |a| a.predict(&[0.0; 8], 1))
                .is_none()
        );
        assert!(w.accel_warned.predict.load(Ordering::Relaxed));
    }

    #[test]
    fn accelerator_wrong_length_output_falls_back_to_cpu() {
        let w = test_weights();
        assert!(w.accel.set(Arc::new(ShortAccel)).is_ok());
        assert!(
            w.accelerated_checked(AccelStage::Embed, 4, |a| a.embed(&[0.0; 8], 2))
                .is_none()
        );
        assert!(w.accel_warned.embed.load(Ordering::Relaxed));
    }

    #[test]
    fn accelerator_decline_is_quiet() {
        let w = test_weights();
        assert!(w.accel.set(Arc::new(DeclineAccel)).is_ok());
        assert!(
            w.accelerated_checked(AccelStage::Embed, 4, |a| a.embed(&[0.0; 8], 2))
                .is_none()
        );
        assert!(!w.accel_warned.embed.load(Ordering::Relaxed));
    }

    /// Valid Nemotron-3 metadata with no tensors: every mutation below fails in `from_gguf`
    /// before the first tensor is read, so these pins run header-only in CI.
    fn mock_kv() -> Vec<(String, crate::gguf::KvValue)> {
        use crate::gguf::KvValue;
        [
            (
                "general.architecture",
                KvValue::Str("nemotron3_diarization".into()),
            ),
            ("nemotron3.mel.normalize", KvValue::Str("NA".into())),
            ("nemotron3.mel.n_fft", KvValue::U32(N_FFT as u32)),
            ("nemotron3.mel.win_length", KvValue::U32(WINDOW_LEN as u32)),
            ("nemotron3.mel.hop_length", KvValue::U32(HOP_LEN as u32)),
            ("nemotron3.sample_rate", KvValue::U32(SAMPLE_RATE)),
            ("nemotron3.mel.preemph", KvValue::F32(0.97)),
            ("nemotron3.mel.mag_power", KvValue::F32(2.0)),
            ("nemotron3.mel.log_zero_guard", KvValue::F32(LOG_MEL_EPS)),
            ("nemotron3.mel.pad_to", KvValue::U32(16)),
            ("nemotron3.block_count", KvValue::U32(1)),
            ("nemotron3.embedding_length", KvValue::U32(8)),
            ("nemotron3.feed_forward_length", KvValue::U32(8)),
            ("nemotron3.attention.head_count", KvValue::U32(2)),
            ("nemotron3.attention.layer_norm_epsilon", KvValue::F32(1e-5)),
            ("nemotron3.num_mel_bins", KvValue::U32(4)),
            ("nemotron3.head.hidden_size", KvValue::U32(4)),
            ("nemotron3.max_speakers", KvValue::U32(8)),
            ("nemotron3.subsampling_factor", KvValue::U32(8)),
            ("nemotron3.rope.theta", KvValue::F32(10_000.0)),
            ("nemotron3.rope.max_pos", KvValue::U32(0)),
            ("nemotron3.stream.chunk_len", KvValue::U32(9)),
            ("nemotron3.stream.chunk_right_context", KvValue::U32(4)),
            ("nemotron3.stream.fifo_len", KvValue::U32(264)),
            ("nemotron3.stream.spkcache_len", KvValue::U32(264)),
            ("nemotron3.stream.spkcache_update_period", KvValue::U32(222)),
            (
                "nemotron3.stream.spkcache_sil_frames_per_spk",
                KvValue::U32(1),
            ),
            ("nemotron3.stream.pred_score_threshold", KvValue::F32(0.25)),
            ("nemotron3.stream.scores_boost_latest", KvValue::F32(0.05)),
            ("nemotron3.stream.sil_threshold", KvValue::F32(0.2)),
            ("nemotron3.stream.strong_boost_rate", KvValue::F32(0.75)),
            ("nemotron3.stream.weak_boost_rate", KvValue::F32(1.5)),
            ("nemotron3.stream.min_pos_scores_rate", KvValue::F32(0.5)),
            ("nemotron3.stream.max_index", KvValue::U32(99_999)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }

    /// Mock header bytes with `patches` applied (each names a key from [`mock_kv`]) and
    /// `drop` omitted.
    fn header_bytes(patches: Vec<(&str, crate::gguf::KvValue)>, drop: Option<&str>) -> Vec<u8> {
        let mut kv = mock_kv();
        for (key, value) in patches {
            let slot = kv
                .iter_mut()
                .find(|(k, _)| k == key)
                .unwrap_or_else(|| panic!("mock header has no key `{key}`"));
            slot.1 = value;
        }
        let mut b = crate::gguf::GgufBuilder::new();
        for (k, v) in kv {
            if Some(k.as_str()) != drop {
                b = b.kv(k, v);
            }
        }
        b.build_bytes()
    }

    fn load_err(bytes: Vec<u8>) -> String {
        let g = Arc::new(GgufFile::from_bytes(Arc::from(bytes.into_boxed_slice())).unwrap());
        match Nemotron3Weights::from_gguf(&g) {
            Ok(_) => panic!("hostile header loaded"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn loader_accepts_valid_metadata_up_to_the_first_tensor() {
        // The mock carries no tensors, so a fully valid header must fail only at the first
        // weight load: every metadata guard passed.
        let err = load_err(header_bytes(Vec::new(), None));
        assert!(
            err.contains("loading nd.embed.proj.weight"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn loader_refuses_hostile_metadata() {
        use crate::gguf::KvValue;
        let cases: Vec<(Vec<(&str, KvValue)>, &str)> = vec![
            (
                vec![("nemotron3.mel.n_fft", KvValue::U32(256))],
                "mel n_fft",
            ),
            (
                vec![("nemotron3.mel.preemph", KvValue::F32(0.0))],
                "mel preemph",
            ),
            (
                vec![("nemotron3.mel.pad_to", KvValue::U32(65))],
                "pad_to 65 outside 1..=64",
            ),
            (
                vec![("nemotron3.mel.pad_to", KvValue::U32(0))],
                "pad_to 0 outside",
            ),
            (
                vec![("nemotron3.block_count", KvValue::U32(0))],
                "block_count 0 outside",
            ),
            (
                vec![("nemotron3.block_count", KvValue::U32(257))],
                "block_count 257 outside",
            ),
            (
                vec![("nemotron3.attention.head_count", KvValue::U32(0))],
                "must be > 0 and divide",
            ),
            (
                vec![("nemotron3.attention.head_count", KvValue::U32(3))],
                "must be > 0 and divide",
            ),
            (
                vec![
                    ("nemotron3.embedding_length", KvValue::U32(128)),
                    ("nemotron3.attention.head_count", KvValue::U32(1)),
                ],
                "exceeds maximum supported accumulator width 64",
            ),
            (
                vec![(
                    "nemotron3.attention.layer_norm_epsilon",
                    KvValue::F32(f32::NAN),
                )],
                "must be finite and > 0",
            ),
            (
                vec![("nemotron3.attention.layer_norm_epsilon", KvValue::F32(0.0))],
                "must be finite and > 0",
            ),
            (
                vec![("nemotron3.feed_forward_length", KvValue::U32(0))],
                "feed_forward_length must be > 0",
            ),
            (
                vec![("nemotron3.head.hidden_size", KvValue::U32(0))],
                "head.hidden_size must be > 0",
            ),
            (
                vec![("nemotron3.max_speakers", KvValue::U32(4))],
                "max_speakers",
            ),
            (
                vec![("nemotron3.subsampling_factor", KvValue::U32(4))],
                "subsampling_factor",
            ),
            (
                vec![("nemotron3.rope.theta", KvValue::F32(5000.0))],
                "rope.theta",
            ),
            (
                vec![("nemotron3.stream.chunk_len", KvValue::U32(0))],
                "chunk_len must be > 0",
            ),
            (
                vec![("nemotron3.stream.max_index", KvValue::U32(8))],
                "lies inside the cache's flat index range",
            ),
        ];
        for (patches, want) in cases {
            let err = load_err(header_bytes(patches, None));
            assert!(err.contains(want), "expected `{want}` in error: {err}");
        }
    }

    #[test]
    fn loader_refuses_a_missing_key() {
        let err = header_bytes(Vec::new(), Some("nemotron3.rope.theta"));
        let err = load_err(err);
        assert!(
            err.contains("missing GGUF key `nemotron3.rope.theta`"),
            "unexpected error: {err}"
        );
    }
}
