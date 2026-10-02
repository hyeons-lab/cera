//! Native Qualcomm Hexagon NPU Conformer encoder for LFM2-Audio.
//!
//! Speech input is the one stage of an LFM2-Audio turn that still runs on the
//! CPU when the backbone, decoder and depthformer are on the NPU: the
//! FastConformer encoder in the `mmproj` GGUF costs about 0.27 CPU-seconds per
//! second of audio (see `examples/audio_encoder_bench.rs`), which is exactly
//! what a background transcription task on a phone wants off the CPU.
//!
//! This module is built in milestones, each checked against the CPU encoder
//! (`audio_encoder`) on the device with `examples/hexagon_conformer_probe.rs`:
//!
//! 1. the macaron feed-forward sub-block (LayerNorm, two Q4_0 linears, SiLU,
//!    half-weight residual), the only part that needs no new kernel support;
//! 2. the convolution module (sigmoid GLU, depthwise conv, per-channel affine);
//! 3. relative-position attention;
//! 4. the whole block stack and the MLP adapter;
//! 5. the conv stem (`audio_stem_hexagon`) and the log-mel front end
//!    (`audio_mel_hexagon`), which were the CPU cost left once the blocks ran
//!    on the DSP.
//!
//! The DSP queue waits for each batch with a blocking read rather than the
//! driver's default spin (see `run_on_queue`): with the blocks on the DSP but
//! the host spinning, the encoder still cost 0.043 CPU-seconds per audio second.
//!
//! All weights and activations live in `rpcmem` buffers, as in the ViT and
//! Whisper paths, and every op is emitted through the shared
//! `backend::hexagon::dispatch` helpers so host tests can pin the emitted op
//! sequence with a recording sink.

use std::sync::{Arc, Mutex};

use crate::backend::hexagon::dispatch::{
    self, ConcatSrc, LayerNormArgs, OpSink, TokenShape, TokenTile, View,
};
use crate::backend::hexagon::{
    FastRpcDriver, HexagonDevice, HexagonQueueSession, HexagonWeightDesc, HexagonWeightFormat,
    LockOrRecover, RpcmemBuffer, align128, repack_q4_0, repack_q8_0, repacked_matrix_size_q4_0,
    repacked_matrix_size_q8_0,
};
use crate::model::audio_encoder::{
    AudioEncoderConfig, AudioEncoderWeights, ConformerLayerWeights, POS_EMB_DIM, relative_pos_emb,
};
use crate::model::audio_encoder_gpu::AudioGpuEncode;
use crate::model::audio_mel_hexagon::{
    MelScratch, MelWeightOffsets, emit_mel, finish as finish_mel, put_mel, stage_samples,
};
use crate::model::audio_preprocessor::{n_frames_for, padded_preemphasized};
use crate::model::audio_stem_hexagon::{
    StemGeom, StemScratch, StemWeightOffsets, emit_stem, put_stem, stage_input, to_channel_major,
};
use crate::model::weights::MmapWeight;
use crate::session::CeraError;
use crate::tensor::DType;

/// Token-axis ops (norms, linears, activations) run in tiles of 64 frames:
/// the quantised matmul and the elementwise kernels overflow the VTCM on a
/// whole 126-frame sequence (`VtcmTooSmall` on the device), as Whisper's
/// encoder found. The attention, convolution and padding ops are not tiled:
/// they take the whole sequence by construction.
const TILE: TokenTile = TokenTile::Tiles(64);

/// An F32 vector in `weights_buf`.
pub(crate) fn plan_vec(cur: &mut usize, len: usize) -> usize {
    let off = *cur;
    *cur += align128(len * 4);
    off
}

/// A repacked Q4_0 or Q8_0 linear weight in `weights_buf`.
pub(crate) fn plan_linear(cur: &mut usize, w: &MmapWeight) -> Result<HexagonWeightDesc, CeraError> {
    plan_matrix(cur, w.dtype, w.rows, w.cols)
}

/// A `rows x cols` matrix of `dtype`, repacked, in `weights_buf`.
fn plan_matrix(
    cur: &mut usize,
    dtype: DType,
    rows: usize,
    cols: usize,
) -> Result<HexagonWeightDesc, CeraError> {
    let (format, size) = match dtype {
        DType::Q8_0 => (
            HexagonWeightFormat::RepackedQ8_0,
            repacked_matrix_size_q8_0(cols, rows)?,
        ),
        DType::Q4_0 => (
            HexagonWeightFormat::RepackedQ4_0,
            repacked_matrix_size_q4_0(cols, rows)?,
        ),
        other => {
            return Err(CeraError::Backend(format!(
                "Hexagon audio encoder linear weight needs Q8_0 or Q4_0, got {other:?}"
            )));
        }
    };
    let offset = *cur;
    *cur += align128(size);
    Ok(HexagonWeightDesc {
        offset,
        size_bytes: size,
        format,
        rows,
        cols,
    })
}

/// Where one macaron feed-forward block's weights sit in `weights_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FfnOffsets {
    pub norm_w: usize,
    pub norm_b: usize,
    pub up_w: HexagonWeightDesc,
    pub up_b: usize,
    pub down_w: HexagonWeightDesc,
    pub down_b: usize,
}

/// Where one block's convolution module sits in `weights_buf`. The first
/// pointwise conv is staged as two matrices, one per GLU half, so the halves
/// land in separate contiguous activations and no op reads a strided view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ConvOffsets {
    pub norm_w: usize,
    pub norm_b: usize,
    /// The GLU value half of `conv_pw1` (output channels `0..n_embd`).
    pub pw1a_w: HexagonWeightDesc,
    pub pw1a_b: usize,
    /// The GLU gate half (output channels `n_embd..2 * n_embd`).
    pub pw1b_w: HexagonWeightDesc,
    pub pw1b_b: usize,
    /// Depthwise taps, `[kernel, n_embd]` as in the GGUF (a channel's taps
    /// contiguous): the layout `SsmConv` takes.
    pub dw_w: usize,
    pub dw_b: usize,
    pub conv_norm_w: usize,
    pub conv_norm_b: usize,
    pub pw2_w: HexagonWeightDesc,
    pub pw2_b: usize,
}

/// Where one block's relative-position attention weights sit in `weights_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttnOffsets {
    pub ln_w: usize,
    pub ln_b: usize,
    pub q_w: HexagonWeightDesc,
    pub q_b: usize,
    pub k_w: HexagonWeightDesc,
    pub k_b: usize,
    pub v_w: HexagonWeightDesc,
    pub v_b: usize,
    pub o_w: HexagonWeightDesc,
    pub o_b: usize,
    /// Per-head content and position biases, `[n_head * d_head]` each.
    pub pos_bias_u: usize,
    pub pos_bias_v: usize,
    /// Projects the sinusoidal relative-position embedding (no bias).
    pub linear_pos_w: HexagonWeightDesc,
}

/// Where one Conformer block's weights sit in `weights_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerOffsets {
    pub ffn1: FfnOffsets,
    pub attn: AttnOffsets,
    pub conv: ConvOffsets,
    pub ffn2: FfnOffsets,
    /// The block's closing LayerNorm.
    pub ln2_w: usize,
    pub ln2_b: usize,
}

/// Where the MLP adapter (LayerNorm, up, GELU, down) sits in `weights_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdapterOffsets {
    pub norm_w: usize,
    pub norm_b: usize,
    pub up_w: HexagonWeightDesc,
    pub up_b: usize,
    pub down_w: HexagonWeightDesc,
    pub down_b: usize,
}

/// The whole weights buffer layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WeightOffsets {
    pub layers: Vec<LayerOffsets>,
    pub adapter: AdapterOffsets,
    pub stem: StemWeightOffsets,
    pub mel: MelWeightOffsets,
    /// A `[n_embd]` vector of 0.5: the macaron residual scale.
    pub half_off: usize,
    /// Zeros, `[kernel - 1, n_embd]` time-inner: the depthwise conv's padding.
    pub zeros_off: usize,
    /// A `[n_embd]` vector of `1 / sqrt(d_head)`: the attention score scale
    /// that has to be applied to the position term by hand.
    pub attn_scale_off: usize,
    /// Depthwise conv kernel size (9 for LFM2-Audio).
    pub kernel: usize,
    pub total_bytes: usize,
}

impl WeightOffsets {
    pub(crate) fn plan(weights: &AudioEncoderWeights) -> Result<Self, CeraError> {
        let mut cur = 0;
        let mut layers = Vec::with_capacity(weights.layers.len());
        for l in &weights.layers {
            let mut ffn = |norm_w: &[f32],
                           norm_b: &[f32],
                           up_w: &MmapWeight,
                           up_b: &[f32],
                           down_w: &MmapWeight,
                           down_b: &[f32]|
             -> Result<FfnOffsets, CeraError> {
                Ok(FfnOffsets {
                    norm_w: plan_vec(&mut cur, norm_w.len()),
                    norm_b: plan_vec(&mut cur, norm_b.len()),
                    up_w: plan_linear(&mut cur, up_w)?,
                    up_b: plan_vec(&mut cur, up_b.len()),
                    down_w: plan_linear(&mut cur, down_w)?,
                    down_b: plan_vec(&mut cur, down_b.len()),
                })
            };
            let ffn1 = ffn(
                &l.ffn_norm_w,
                &l.ffn_norm_b,
                &l.ffn_up_w,
                &l.ffn_up_b,
                &l.ffn_down_w,
                &l.ffn_down_b,
            )?;
            let ffn2 = ffn(
                &l.ffn_norm_1_w,
                &l.ffn_norm_1_b,
                &l.ffn_up_1_w,
                &l.ffn_up_1_b,
                &l.ffn_down_1_w,
                &l.ffn_down_1_b,
            )?;
            let n = weights.config.n_embd;
            let half_rows = l.conv_pw1_w.rows / 2;
            if l.conv_pw1_w.rows != 2 * n || l.conv_dw_w.len() % n != 0 || l.conv_dw_w.is_empty() {
                return Err(CeraError::Backend(format!(
                    "audio encoder conv module shape: pw1 rows {} for n_embd {n}, {} depthwise taps",
                    l.conv_pw1_w.rows,
                    l.conv_dw_w.len()
                )));
            }
            let (pw1_dtype, pw1_cols) = (l.conv_pw1_w.dtype, l.conv_pw1_w.cols);
            let conv = ConvOffsets {
                norm_w: plan_vec(&mut cur, l.norm_conv_w.len()),
                norm_b: plan_vec(&mut cur, l.norm_conv_b.len()),
                pw1a_w: plan_matrix(&mut cur, pw1_dtype, half_rows, pw1_cols)?,
                pw1a_b: plan_vec(&mut cur, half_rows),
                pw1b_w: plan_matrix(&mut cur, pw1_dtype, half_rows, pw1_cols)?,
                pw1b_b: plan_vec(&mut cur, half_rows),
                dw_w: plan_vec(&mut cur, l.conv_dw_w.len()),
                dw_b: plan_vec(&mut cur, l.conv_dw_b.len()),
                conv_norm_w: plan_vec(&mut cur, l.conv_norm_w.len()),
                conv_norm_b: plan_vec(&mut cur, l.conv_norm_b.len()),
                pw2_w: plan_linear(&mut cur, &l.conv_pw2_w)?,
                pw2_b: plan_vec(&mut cur, l.conv_pw2_b.len()),
            };
            let attn = AttnOffsets {
                ln_w: plan_vec(&mut cur, l.ln1_w.len()),
                ln_b: plan_vec(&mut cur, l.ln1_b.len()),
                q_w: plan_linear(&mut cur, &l.attn_q_w)?,
                q_b: plan_vec(&mut cur, l.attn_q_b.len()),
                k_w: plan_linear(&mut cur, &l.attn_k_w)?,
                k_b: plan_vec(&mut cur, l.attn_k_b.len()),
                v_w: plan_linear(&mut cur, &l.attn_v_w)?,
                v_b: plan_vec(&mut cur, l.attn_v_b.len()),
                o_w: plan_linear(&mut cur, &l.attn_o_w)?,
                o_b: plan_vec(&mut cur, l.attn_o_b.len()),
                pos_bias_u: plan_vec(&mut cur, l.pos_bias_u.len()),
                pos_bias_v: plan_vec(&mut cur, l.pos_bias_v.len()),
                linear_pos_w: plan_linear(&mut cur, &l.linear_pos_w)?,
            };
            layers.push(LayerOffsets {
                ffn1,
                attn,
                conv,
                ffn2,
                ln2_w: plan_vec(&mut cur, l.ln2_w.len()),
                ln2_b: plan_vec(&mut cur, l.ln2_b.len()),
            });
        }
        let n = weights.config.n_embd;
        let ad = &weights.mlp_adapter;
        let adapter = AdapterOffsets {
            norm_w: plan_vec(&mut cur, ad.norm_w.len()),
            norm_b: plan_vec(&mut cur, ad.norm_b.len()),
            up_w: plan_linear(&mut cur, &ad.up_w)?,
            up_b: plan_vec(&mut cur, ad.up_b.len()),
            down_w: plan_linear(&mut cur, &ad.down_w)?,
            down_b: plan_vec(&mut cur, ad.down_b.len()),
        };
        let stem = StemWeightOffsets::plan(&mut cur, &weights.conv_stem, &weights.config)?;
        let mel = MelWeightOffsets::plan(&mut cur, weights.config.n_mel_bins);
        let kernel = weights.layers.first().map_or(9, |l| l.conv_dw_w.len() / n);
        let half_off = plan_vec(&mut cur, n);
        let zeros_off = plan_vec(&mut cur, (kernel - 1) * n);
        let attn_scale_off = plan_vec(&mut cur, n);
        Ok(Self {
            layers,
            adapter,
            stem,
            mel,
            half_off,
            zeros_off,
            attn_scale_off,
            kernel,
            total_bytes: cur,
        })
    }
}

/// Activation scratch layout for sequences of up to `max_frames` frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScratchOffsets {
    /// The running `[frames, n_embd]` sequence.
    pub x: usize,
    /// The normalised copy a sub-block reads.
    pub norm: usize,
    /// `[frames, n_ff]`: the feed-forward's wide activation.
    pub ff: usize,
    /// `[frames, n_embd]`: a sub-block's output before its residual add.
    pub out: usize,
    /// Convolution module: the GLU value half, `[frames, n_embd]`.
    pub conv_a: usize,
    /// The GLU gate half.
    pub conv_g: usize,
    /// Left-padded channel-major sequence, `[pad_left + frames, n_embd]`.
    pub conv_tmp: usize,
    /// Fully padded channel-major sequence, `[frames + kernel - 1, n_embd]`.
    pub conv_x: usize,
    /// The depthwise conv's output, time-major `[frames, n_embd]`.
    pub conv_y: usize,
    /// Attention: the relative-position embedding, `[2 * frames - 1, POS_EMB_DIM]`
    /// (written by the host for each sequence length).
    pub attn_pos: usize,
    /// `linear_pos` of it, `[2 * frames - 1, n_embd]`.
    pub attn_p: usize,
    /// Q, K, V and the two biased copies of Q, each `[frames, n_embd]`.
    pub attn_q: usize,
    pub attn_k: usize,
    pub attn_v: usize,
    pub attn_qu: usize,
    pub attn_qv: usize,
    /// Content scores `[keys, queries, heads]`, contiguous: the softmax kernel
    /// addresses its source and destination rows by their length alone.
    pub attn_ac: usize,
    /// The softmax output, contiguous like its input.
    pub attn_smc: usize,
    /// The same weights with rows padded to a multiple of 32 elements; the
    /// padding stays zero. This is what `attn @ V` contracts over.
    pub attn_sm: usize,
    /// Position scores `[2 * frames - 1, queries, heads]`, rows padded to 32.
    pub attn_bd: usize,
    /// V transposed per head, `[keys, d_head, heads]`, rows padded to 32 and
    /// zero there.
    pub attn_vt: usize,
    /// The attention output, `[frames, n_embd]`.
    pub attn_av: usize,
    /// The adapter's output, `[frames, llm_hidden_size]`: the embeddings.
    pub adapter_out: usize,
    pub total_bytes: usize,
}

/// Round up to a multiple of 32 elements (128 bytes of F32): the row padding
/// the attention score tensors use so every row starts on a vector boundary.
pub(crate) fn pad32(n: usize) -> usize {
    n.next_multiple_of(32)
}

impl ScratchOffsets {
    pub(crate) fn new(cfg: &AudioEncoderConfig, max_frames: usize, kernel: usize) -> Self {
        let seq = align128(max_frames * cfg.n_embd * 4);
        let padded = align128((max_frames + kernel) * cfg.n_embd * 4);
        let ff = align128(max_frames * cfg.n_ff * 4);
        let mut cur = 0;
        let mut region = |bytes: usize| {
            let off = cur;
            cur += bytes;
            off
        };
        let x = region(seq);
        let norm = region(seq);
        let ff_off = region(ff);
        let out = region(seq);
        let conv_a = region(seq);
        let conv_g = region(seq);
        let conv_tmp = region(padded);
        let conv_x = region(padded);
        let conv_y = region(seq);
        let n_head = cfg.n_head;
        let d_head = cfg.n_embd / cfg.n_head.max(1);
        let seq_pos = max_frames * 2 - 1;
        let attn_pos = region(align128(seq_pos * POS_EMB_DIM * 4));
        let attn_p = region(align128(seq_pos * cfg.n_embd * 4));
        let attn_q = region(seq);
        let attn_k = region(seq);
        let attn_v = region(seq);
        let attn_qu = region(seq);
        let attn_qv = region(seq);
        let scores = align128(n_head * max_frames * pad32(max_frames) * 4);
        let attn_ac = region(scores);
        let attn_smc = region(scores);
        let attn_sm = region(scores);
        let attn_bd = region(align128(n_head * max_frames * pad32(seq_pos) * 4));
        let attn_vt = region(align128(n_head * d_head * pad32(max_frames) * 4));
        let attn_av = region(seq);
        let adapter_out = region(align128(max_frames * cfg.llm_hidden_size * 4));
        Self {
            x,
            norm,
            ff: ff_off,
            out,
            conv_a,
            conv_g,
            conv_tmp,
            conv_x,
            conv_y,
            attn_pos,
            attn_p,
            attn_q,
            attn_k,
            attn_v,
            attn_qu,
            attn_qv,
            attn_ac,
            attn_smc,
            attn_sm,
            attn_bd,
            attn_vt,
            attn_av,
            adapter_out,
            total_bytes: cur,
        }
    }
}

/// One macaron feed-forward sub-block, `x += 0.5 * down(silu(up(norm(x))))`,
/// emitted for `t` frames (mirrors `audio_encoder::conformer_ffn_forward`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_ffn<S: OpSink>(
    s: &mut S,
    weights: &S::Buf,
    scratch: &S::Buf,
    ffn: &FfnOffsets,
    half_off: usize,
    so: &ScratchOffsets,
    cfg: &AudioEncoderConfig,
    t: usize,
) -> Result<(), CeraError> {
    let embd = TokenShape {
        dim: cfg.n_embd,
        n_tokens: t,
    };
    dispatch::layer_norm(
        s,
        LayerNormArgs {
            src: scratch,
            src_offset: so.x,
            dst: scratch,
            dst_offset: so.norm,
            weights,
            w_offset: ffn.norm_w,
            b_offset: ffn.norm_b,
            shape: embd,
            eps: cfg.eps,
            tile: TILE,
        },
    )?;
    dispatch::linear_m(
        s, scratch, so.norm, weights, ffn.up_w, ffn.up_b, scratch, so.ff, t, TILE,
    )?;
    dispatch::silu(
        s,
        scratch,
        so.ff,
        TokenShape {
            dim: cfg.n_ff,
            n_tokens: t,
        },
        TILE,
    )?;
    dispatch::linear_m(
        s, scratch, so.ff, weights, ffn.down_w, ffn.down_b, scratch, so.out, t, TILE,
    )?;
    // The macaron half step: scale the branch, then add it to the sequence.
    dispatch::mul_row_bcast(s, scratch, so.out, weights, half_off, embd, TILE)?;
    dispatch::add_residual(s, scratch, so.x, scratch, so.out, embd, TILE)
}

/// The convolution sub-block, `x += pw2(silu(affine(dwconv(glu(pw1(norm(x)))))))`,
/// for `t` frames (mirrors `audio_encoder::conformer_conv_module_forward`).
///
/// The reference pads the depthwise conv symmetrically (`(kernel - 1) / 2`
/// zeros each side), so the padded channel-major sequence is built in two
/// `Concat`s (left pad ++ transposed GLU output, then ++ right pad) and
/// `SsmConv` runs a "valid" window over it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_conv_module<S: OpSink>(
    s: &mut S,
    weights: &S::Buf,
    scratch: &S::Buf,
    conv: &ConvOffsets,
    zeros_off: usize,
    kernel: usize,
    so: &ScratchOffsets,
    cfg: &AudioEncoderConfig,
    t: usize,
) -> Result<(), CeraError> {
    let n = cfg.n_embd;
    let embd = TokenShape {
        dim: n,
        n_tokens: t,
    };
    let (pad_left, pad_right) = ((kernel - 1) / 2, kernel - 1 - (kernel - 1) / 2);
    dispatch::layer_norm(
        s,
        LayerNormArgs {
            src: scratch,
            src_offset: so.x,
            dst: scratch,
            dst_offset: so.norm,
            weights,
            w_offset: conv.norm_w,
            b_offset: conv.norm_b,
            shape: embd,
            eps: cfg.eps,
            tile: TILE,
        },
    )?;
    // GLU: a * sigmoid(g), from the two halves of the first pointwise conv.
    dispatch::linear_m(
        s,
        scratch,
        so.norm,
        weights,
        conv.pw1a_w,
        conv.pw1a_b,
        scratch,
        so.conv_a,
        t,
        TILE,
    )?;
    dispatch::linear_m(
        s,
        scratch,
        so.norm,
        weights,
        conv.pw1b_w,
        conv.pw1b_b,
        scratch,
        so.conv_g,
        t,
        TILE,
    )?;
    dispatch::sigmoid(s, scratch, so.conv_g, embd, TILE)?;
    dispatch::mul_inplace(s, scratch, so.conv_a, scratch, so.conv_g, embd, TILE)?;
    // Pad and transpose to channel-major: [pad_left + t, n] then [t + kernel - 1, n].
    dispatch::concat_time_inner(
        s,
        ConcatSrc {
            buf: weights,
            offset: zeros_off,
            rows: pad_left,
            nb0: 4,
            nb1: pad_left * 4,
        },
        ConcatSrc {
            buf: scratch,
            offset: so.conv_a,
            rows: t,
            nb0: n * 4,
            nb1: 4,
        },
        scratch,
        so.conv_tmp,
        n,
    )?;
    dispatch::concat_time_inner(
        s,
        ConcatSrc {
            buf: scratch,
            offset: so.conv_tmp,
            rows: pad_left + t,
            nb0: 4,
            nb1: (pad_left + t) * 4,
        },
        ConcatSrc {
            buf: weights,
            offset: zeros_off,
            rows: pad_right,
            nb0: 4,
            nb1: pad_right * 4,
        },
        scratch,
        so.conv_x,
        n,
    )?;
    dispatch::ssm_conv(
        s,
        weights,
        conv.dw_w,
        scratch,
        so.conv_x,
        scratch,
        so.conv_y,
        kernel,
        n,
        t,
        dispatch::VTCM_BUDGET,
    )?;
    // Depthwise bias, the per-channel affine "conv_norm", then SiLU.
    dispatch::add_row_bcast(s, scratch, so.conv_y, weights, conv.dw_b, embd, TILE)?;
    dispatch::mul_row_bcast(s, scratch, so.conv_y, weights, conv.conv_norm_w, embd, TILE)?;
    dispatch::add_row_bcast(s, scratch, so.conv_y, weights, conv.conv_norm_b, embd, TILE)?;
    dispatch::silu(s, scratch, so.conv_y, embd, TILE)?;
    dispatch::linear_m(
        s, scratch, so.conv_y, weights, conv.pw2_w, conv.pw2_b, scratch, so.out, t, TILE,
    )?;
    dispatch::add_residual(s, scratch, so.x, scratch, so.out, embd, TILE)
}

/// The relative-position self-attention sub-block for `t` frames (mirrors
/// `audio_encoder::conformer_self_attention_forward`):
///
/// ```text
/// scores[h, q, k] = (Q[q,h] + u[h]) . K[k,h]  +  s * (Q[q,h] + v[h]) . P[t-1-q+k, h]
/// attn            = softmax(s * scores_content + position_term)
/// ```
///
/// with `s = 1 / sqrt(d_head)`. The position term's "rel-shift" is a strided
/// view of the `[2t-1, t, heads]` position-score matrix whose row stride is
/// one element short of the real one, so row `q` starts `t - 1 - q` elements
/// further along: no data moves. The softmax takes it as its additive mask.
/// Rows of the score tensors are padded to 32 so each starts on a vector
/// boundary, and the padding of `attn_sm` and `attn_vt` is zero (the host
/// clears it, see [`HexagonAudioEncoder::clear_attention_padding`]), which
/// lets the final `attn @ V` contract over the padded length.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_attention<S: OpSink>(
    s: &mut S,
    weights: &S::Buf,
    scratch: &S::Buf,
    attn: &AttnOffsets,
    scale_off: usize,
    so: &ScratchOffsets,
    cfg: &AudioEncoderConfig,
    t: usize,
) -> Result<(), CeraError> {
    let n = cfg.n_embd;
    let (h, d) = (cfg.n_head, cfg.n_embd / cfg.n_head);
    let seq_pos = 2 * t - 1;
    let (tp, lp) = (pad32(t), pad32(seq_pos));
    let scale = 1.0 / (d as f32).sqrt();
    let embd = TokenShape {
        dim: n,
        n_tokens: t,
    };

    dispatch::layer_norm(
        s,
        LayerNormArgs {
            src: scratch,
            src_offset: so.x,
            dst: scratch,
            dst_offset: so.norm,
            weights,
            w_offset: attn.ln_w,
            b_offset: attn.ln_b,
            shape: embd,
            eps: cfg.eps,
            tile: TILE,
        },
    )?;
    for (w, b, dst) in [
        (attn.q_w, attn.q_b, so.attn_q),
        (attn.k_w, attn.k_b, so.attn_k),
        (attn.v_w, attn.v_b, so.attn_v),
    ] {
        dispatch::linear_m(s, scratch, so.norm, weights, w, b, scratch, dst, t, TILE)?;
    }
    // The position embedding through `linear_pos` (no bias).
    dispatch::linear_m(
        s,
        scratch,
        so.attn_pos,
        weights,
        attn.linear_pos_w,
        None,
        scratch,
        so.attn_p,
        seq_pos,
        TILE,
    )?;
    // Q + u for the content term; (Q + v) * s for the position term, so the
    // softmax's own scale only has to cover the content term.
    let rows = |off| View::new(scratch, off, [n, t, 1], [4, n * 4, t * n * 4]);
    dispatch::add_row_bcast_to(
        s,
        rows(so.attn_q),
        weights,
        attn.pos_bias_u,
        rows(so.attn_qu),
    )?;
    dispatch::add_row_bcast_to(
        s,
        rows(so.attn_q),
        weights,
        attn.pos_bias_v,
        rows(so.attn_qv),
    )?;
    dispatch::mul_row_bcast(s, scratch, so.attn_qv, weights, scale_off, embd, TILE)?;

    // One head's `[d, rows]` slice of a `[rows, n_embd]` activation.
    let heads = |off, rows: usize| View::new(scratch, off, [d, rows, h], [4, n * 4, d * 4]);
    // Content scores: ac[k, q, h] = K[k,h] . (Q+u)[q,h].
    let ac = View::new(scratch, so.attn_ac, [t, t, h], [4, t * 4, t * t * 4]);
    dispatch::matmul_f32(s, heads(so.attn_k, t), heads(so.attn_qu, t), ac)?;
    // Position scores over all 2t-1 relative offsets: bd[j, q, h] = P[j,h] . ((Q+v)*s)[q,h].
    let bd = View::new(
        scratch,
        so.attn_bd,
        [seq_pos, t, h],
        [4, lp * 4, t * lp * 4],
    );
    dispatch::matmul_f32(s, heads(so.attn_p, seq_pos), heads(so.attn_qv, t), bd)?;
    // The rel-shift view: element (k, q) is bd[t-1-q+k, q]: start t-1 elements
    // in and step one element less than a row per query.
    let shifted = View::new(
        scratch,
        so.attn_bd + (t - 1) * 4,
        [t, t, h],
        [4, (lp - 1) * 4, t * lp * 4],
    );
    let smc = View::new(scratch, so.attn_smc, [t, t, h], [4, t * 4, t * t * 4]);
    dispatch::softmax(s, ac, Some(shifted), smc, scale)?;
    // Into the zero-padded layout the last contraction reads.
    let sm = View::new(scratch, so.attn_sm, [t, t, h], [4, tp * 4, t * tp * 4]);
    dispatch::copy_view(s, smc, sm)?;
    // V transposed per head so the contraction (over keys) is contiguous.
    let v_src = View::new(scratch, so.attn_v, [t, d, h], [n * 4, 4, d * 4]);
    let vt = View::new(scratch, so.attn_vt, [t, d, h], [4, tp * 4, d * tp * 4]);
    dispatch::copy_view(s, v_src, vt)?;
    // attn @ V per head, written straight into the head-interleaved layout
    // the output projection reads.
    let vt_full = View::new(scratch, so.attn_vt, [tp, d, h], [4, tp * 4, d * tp * 4]);
    let sm_full = View::new(scratch, so.attn_sm, [tp, t, h], [4, tp * 4, t * tp * 4]);
    dispatch::matmul_f32(s, vt_full, sm_full, heads(so.attn_av, t))?;
    dispatch::linear_m(
        s, scratch, so.attn_av, weights, attn.o_w, attn.o_b, scratch, so.out, t, TILE,
    )?;
    dispatch::add_residual(s, scratch, so.x, scratch, so.out, embd, TILE)
}

/// The MLP adapter, `down(gelu(up(norm(x))))`, into `adapter_out`.
///
/// The reference uses the exact (erf) GELU; the DSP has the tanh form, which
/// differs by under 5e-4 in absolute terms, well below the Q4_0 weight noise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_adapter<S: OpSink>(
    s: &mut S,
    weights: &S::Buf,
    scratch: &S::Buf,
    ad: &AdapterOffsets,
    so: &ScratchOffsets,
    cfg: &AudioEncoderConfig,
    t: usize,
) -> Result<(), CeraError> {
    dispatch::layer_norm(
        s,
        LayerNormArgs {
            src: scratch,
            src_offset: so.x,
            dst: scratch,
            dst_offset: so.norm,
            weights,
            w_offset: ad.norm_w,
            b_offset: ad.norm_b,
            shape: TokenShape {
                dim: cfg.n_embd,
                n_tokens: t,
            },
            eps: cfg.eps,
            tile: TILE,
        },
    )?;
    dispatch::linear_m(
        s, scratch, so.norm, weights, ad.up_w, ad.up_b, scratch, so.ff, t, TILE,
    )?;
    dispatch::gelu(
        s,
        scratch,
        so.ff,
        TokenShape {
            dim: ad.up_w.rows,
            n_tokens: t,
        },
        TILE,
    )?;
    dispatch::linear_m(
        s,
        scratch,
        so.ff,
        weights,
        ad.down_w,
        ad.down_b,
        scratch,
        so.adapter_out,
        t,
        TILE,
    )
}

/// One whole Conformer block, in the reference's order: feed-forward, attention,
/// convolution, feed-forward, then the closing LayerNorm (no residual).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_block<S: OpSink>(
    s: &mut S,
    weights: &S::Buf,
    scratch: &S::Buf,
    layer: &LayerOffsets,
    offsets: &WeightOffsets,
    so: &ScratchOffsets,
    cfg: &AudioEncoderConfig,
    t: usize,
) -> Result<(), CeraError> {
    emit_ffn(
        s,
        weights,
        scratch,
        &layer.ffn1,
        offsets.half_off,
        so,
        cfg,
        t,
    )?;
    emit_attention(
        s,
        weights,
        scratch,
        &layer.attn,
        offsets.attn_scale_off,
        so,
        cfg,
        t,
    )?;
    emit_conv_module(
        s,
        weights,
        scratch,
        &layer.conv,
        offsets.zeros_off,
        offsets.kernel,
        so,
        cfg,
        t,
    )?;
    emit_ffn(
        s,
        weights,
        scratch,
        &layer.ffn2,
        offsets.half_off,
        so,
        cfg,
        t,
    )?;
    // ln2 into `norm`, then back over the sequence.
    let embd = TokenShape {
        dim: cfg.n_embd,
        n_tokens: t,
    };
    dispatch::layer_norm(
        s,
        LayerNormArgs {
            src: scratch,
            src_offset: so.x,
            dst: scratch,
            dst_offset: so.norm,
            weights,
            w_offset: layer.ln2_w,
            b_offset: layer.ln2_b,
            shape: embd,
            eps: cfg.eps,
            tile: TILE,
        },
    )?;
    let rows = |off| {
        View::new(
            scratch,
            off,
            [cfg.n_embd, t, 1],
            [4, cfg.n_embd * 4, t * cfg.n_embd * 4],
        )
    };
    dispatch::copy_view(s, rows(so.norm), rows(so.x))
}

/// Each stage of the NPU conv stem in the CPU stem's layouts.
#[derive(Debug, Clone)]
pub struct StemDump {
    /// Convolution outputs, channel-major `[channels, height, width]`.
    pub l0: Vec<f32>,
    pub dw1: Vec<f32>,
    pub pw2: Vec<f32>,
    pub dw3: Vec<f32>,
    pub pw4: Vec<f32>,
    /// `[t, channels * width]`: what the projection reads.
    pub flat: Vec<f32>,
    /// `[t, n_embd]`: the stem's output.
    pub out: Vec<f32>,
    pub t: usize,
}

/// The attention stage's intermediates, as the NPU left them (see
/// [`HexagonAudioEncoder::debug_attention`]). Per-head planes are
/// `[head][row][col]`: `ac`/`sm` are `[t, t]` (row = query, col = key), `bd`
/// is `[t, 2t-1]`, `vt` is `[head * d_head, t]`.
#[derive(Debug, Clone)]
pub struct AttentionDump {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub p: Vec<f32>,
    pub qu: Vec<f32>,
    pub qv: Vec<f32>,
    pub ac: Vec<f32>,
    pub bd: Vec<f32>,
    pub sm: Vec<f32>,
    pub vt: Vec<f32>,
    pub av: Vec<f32>,
    pub out: Vec<f32>,
}

/// LFM2-Audio's Conformer encoder on the Hexagon NPU. Work in progress: see
/// the module docs for what is implemented.
pub struct HexagonAudioEncoder {
    driver: Arc<FastRpcDriver>,
    device: Arc<Mutex<HexagonDevice>>,
    /// Kept for the CPU conv stem fallback.
    weights: Arc<AudioEncoderWeights>,
    config: AudioEncoderConfig,
    weights_buf: RpcmemBuffer,
    offsets: WeightOffsets,
    scratch: Mutex<RpcmemBuffer>,
    scratch_offsets: ScratchOffsets,
    max_frames: usize,
}

// SAFETY: the rpcmem buffers are only touched while holding `device` and
// `scratch`, as in the ViT and Whisper encoders.
unsafe impl Send for HexagonAudioEncoder {}
unsafe impl Sync for HexagonAudioEncoder {}

/// Copy an F32 vector into the weights buffer.
pub(crate) fn put_vec(dst: &mut [u8], off: usize, src: &[f32]) {
    let bytes: &[u8] = bytemuck::cast_slice(src);
    dst[off..off + bytes.len()].copy_from_slice(bytes);
}

/// Repack a linear weight into its planned slot.
pub(crate) fn put_linear(
    dst: &mut [u8],
    desc: HexagonWeightDesc,
    w: &MmapWeight,
) -> Result<(), CeraError> {
    let slot = &mut dst[desc.offset..desc.offset + desc.size_bytes];
    match desc.format {
        HexagonWeightFormat::RepackedQ8_0 => repack_q8_0(w.data(), w.cols, w.rows, slot),
        HexagonWeightFormat::RepackedQ4_0 => repack_q4_0(w.data(), w.cols, w.rows, slot),
    }
    .map_err(|e| CeraError::Backend(format!("audio encoder repack failed: {e}")))
}

fn put_ffn(
    dst: &mut [u8],
    o: &FfnOffsets,
    norm: (&[f32], &[f32]),
    up: (&MmapWeight, &[f32]),
    down: (&MmapWeight, &[f32]),
) -> Result<(), CeraError> {
    put_vec(dst, o.norm_w, norm.0);
    put_vec(dst, o.norm_b, norm.1);
    put_linear(dst, o.up_w, up.0)?;
    put_vec(dst, o.up_b, up.1);
    put_linear(dst, o.down_w, down.0)?;
    put_vec(dst, o.down_b, down.1);
    Ok(())
}

/// Repack rows `lo..hi` of `w` (a Q4_0 or Q8_0 matrix: a row's blocks are
/// contiguous) into the slot planned for them.
fn put_linear_rows(
    dst: &mut [u8],
    desc: HexagonWeightDesc,
    w: &MmapWeight,
    lo: usize,
    hi: usize,
) -> Result<(), CeraError> {
    let block_bytes = match w.dtype {
        DType::Q4_0 => 18,
        DType::Q8_0 => 34,
        other => {
            return Err(CeraError::Backend(format!(
                "audio encoder row split needs Q4_0 or Q8_0, got {other:?}"
            )));
        }
    };
    let row_bytes = w.cols / 32 * block_bytes;
    let rows = &w.data()[lo * row_bytes..hi * row_bytes];
    let slot = &mut dst[desc.offset..desc.offset + desc.size_bytes];
    match desc.format {
        HexagonWeightFormat::RepackedQ8_0 => repack_q8_0(rows, w.cols, hi - lo, slot),
        HexagonWeightFormat::RepackedQ4_0 => repack_q4_0(rows, w.cols, hi - lo, slot),
    }
    .map_err(|e| CeraError::Backend(format!("audio encoder repack failed: {e}")))
}

fn put_conv(dst: &mut [u8], o: &ConvOffsets, l: &ConformerLayerWeights) -> Result<(), CeraError> {
    let half = l.conv_pw1_w.rows / 2;
    put_vec(dst, o.norm_w, &l.norm_conv_w);
    put_vec(dst, o.norm_b, &l.norm_conv_b);
    put_linear_rows(dst, o.pw1a_w, &l.conv_pw1_w, 0, half)?;
    put_vec(dst, o.pw1a_b, &l.conv_pw1_b[..half]);
    put_linear_rows(dst, o.pw1b_w, &l.conv_pw1_w, half, 2 * half)?;
    put_vec(dst, o.pw1b_b, &l.conv_pw1_b[half..]);
    put_vec(dst, o.dw_w, &l.conv_dw_w);
    put_vec(dst, o.dw_b, &l.conv_dw_b);
    put_vec(dst, o.conv_norm_w, &l.conv_norm_w);
    put_vec(dst, o.conv_norm_b, &l.conv_norm_b);
    put_linear(dst, o.pw2_w, &l.conv_pw2_w)?;
    put_vec(dst, o.pw2_b, &l.conv_pw2_b);
    Ok(())
}

fn put_attn(dst: &mut [u8], o: &AttnOffsets, l: &ConformerLayerWeights) -> Result<(), CeraError> {
    put_vec(dst, o.ln_w, &l.ln1_w);
    put_vec(dst, o.ln_b, &l.ln1_b);
    put_linear(dst, o.q_w, &l.attn_q_w)?;
    put_vec(dst, o.q_b, &l.attn_q_b);
    put_linear(dst, o.k_w, &l.attn_k_w)?;
    put_vec(dst, o.k_b, &l.attn_k_b);
    put_linear(dst, o.v_w, &l.attn_v_w)?;
    put_vec(dst, o.v_b, &l.attn_v_b);
    put_linear(dst, o.o_w, &l.attn_o_w)?;
    put_vec(dst, o.o_b, &l.attn_o_b);
    put_vec(dst, o.pos_bias_u, &l.pos_bias_u);
    put_vec(dst, o.pos_bias_v, &l.pos_bias_v);
    put_linear(dst, o.linear_pos_w, &l.linear_pos_w)
}

fn put_adapter(
    dst: &mut [u8],
    o: &AdapterOffsets,
    ad: &crate::model::audio_encoder::AudioMlpAdapterWeights,
) -> Result<(), CeraError> {
    put_vec(dst, o.norm_w, &ad.norm_w);
    put_vec(dst, o.norm_b, &ad.norm_b);
    put_linear(dst, o.up_w, &ad.up_w)?;
    put_vec(dst, o.up_b, &ad.up_b);
    put_linear(dst, o.down_w, &ad.down_w)?;
    put_vec(dst, o.down_b, &ad.down_b);
    Ok(())
}

fn put_layer(dst: &mut [u8], o: &LayerOffsets, l: &ConformerLayerWeights) -> Result<(), CeraError> {
    put_vec(dst, o.ln2_w, &l.ln2_w);
    put_vec(dst, o.ln2_b, &l.ln2_b);
    put_attn(dst, &o.attn, l)?;
    put_conv(dst, &o.conv, l)?;
    put_ffn(
        dst,
        &o.ffn1,
        (&l.ffn_norm_w, &l.ffn_norm_b),
        (&l.ffn_up_w, &l.ffn_up_b),
        (&l.ffn_down_w, &l.ffn_down_b),
    )?;
    put_ffn(
        dst,
        &o.ffn2,
        (&l.ffn_norm_1_w, &l.ffn_norm_1_b),
        (&l.ffn_up_1_w, &l.ffn_up_1_b),
        (&l.ffn_down_1_w, &l.ffn_down_1_b),
    )
}

/// Run `emit` on the queue and wait for everything it submitted. Each batch
/// here is long and nothing is latency critical, so the wait sleeps in the
/// kernel instead of keeping a core spinning: spinning was most of the
/// encoder's CPU time (143 of 145 ms for the blocks of 10 s of audio on the
/// S25 Ultra).
fn run_on_queue(
    what: &str,
    session: &mut HexagonQueueSession,
    emit: impl FnOnce(&mut HexagonQueueSession) -> Result<(), CeraError>,
) -> Result<(), CeraError> {
    session.drop_pending_batch();
    let polled = session.set_blocking_wait(true);
    let run = emit(session).and_then(|()| session.flush());
    session.set_blocking_wait(polled);
    if let Err(e) = run {
        session.drop_pending_batch();
        return Err(CeraError::Backend(format!("{what}: {e}")));
    }
    Ok(())
}

impl HexagonAudioEncoder {
    /// Stage `weights` on the device for sequences of up to `max_frames`
    /// frames (the length after the conv stem's 8x subsampling).
    pub fn new(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        weights: &Arc<AudioEncoderWeights>,
        max_frames: usize,
    ) -> Result<Self, CeraError> {
        let config = weights.config.clone();
        let offsets = WeightOffsets::plan(weights)?;
        let scratch_offsets = ScratchOffsets::new(&config, max_frames, offsets.kernel);
        let scratch = RpcmemBuffer::alloc(Arc::clone(&driver), scratch_offsets.total_bytes, true)?;
        let mut weights_buf = RpcmemBuffer::alloc(Arc::clone(&driver), offsets.total_bytes, true)?;

        let dst = weights_buf.as_mut_slice();
        for (o, l) in offsets.layers.iter().zip(&weights.layers) {
            put_layer(dst, o, l)?;
        }
        put_adapter(dst, &offsets.adapter, &weights.mlp_adapter)?;
        put_stem(dst, &offsets.stem, &weights.conv_stem)?;
        put_mel(dst, &offsets.mel, config.n_mel_bins);
        put_vec(dst, offsets.half_off, &vec![0.5f32; config.n_embd]);
        let d_head = config.n_embd / config.n_head.max(1);
        put_vec(
            dst,
            offsets.attn_scale_off,
            &vec![1.0 / (d_head as f32).sqrt(); config.n_embd],
        );
        put_vec(
            dst,
            offsets.zeros_off,
            &vec![0.0f32; (offsets.kernel - 1) * config.n_embd],
        );
        weights_buf.flush_cpu_cache(0, offsets.total_bytes);

        Ok(Self {
            driver,
            device,
            weights: Arc::clone(weights),
            config,
            weights_buf,
            offsets,
            scratch: Mutex::new(scratch),
            scratch_offsets,
            max_frames,
        })
    }

    /// Run `emit` over the `t`-frame sequence `x` (`[t, n_embd]`) on the NPU
    /// and return the sequence it leaves in `scratch.x`. The stage-level entry
    /// points below are what the probe checks against the CPU encoder.
    fn run_stage(
        &self,
        what: &str,
        x: &[f32],
        t: usize,
        emit: impl FnOnce(
            &mut HexagonQueueSession,
            &RpcmemBuffer,
            &RpcmemBuffer,
            &ScratchOffsets,
        ) -> Result<(), CeraError>,
    ) -> Result<Vec<f32>, CeraError> {
        let out = (self.scratch_offsets.x, t * self.config.n_embd);
        self.run_stage_with(what, x, t, |_, _| (), emit, out)
    }

    /// [`Self::run_stage`] with `prepare` run on the scratch buffer first (host
    /// writes the stage needs).
    fn run_stage_with(
        &self,
        what: &str,
        x: &[f32],
        t: usize,
        prepare: impl FnOnce(&mut RpcmemBuffer, &ScratchOffsets),
        emit: impl FnOnce(
            &mut HexagonQueueSession,
            &RpcmemBuffer,
            &RpcmemBuffer,
            &ScratchOffsets,
        ) -> Result<(), CeraError>,
        out: (usize, usize),
    ) -> Result<Vec<f32>, CeraError> {
        let cfg = &self.config;
        if t == 0 || t > self.max_frames || x.len() != t * cfg.n_embd {
            return Err(CeraError::Backend(format!(
                "{what}: {} floats for {t} frames of {} (at most {} frames)",
                x.len(),
                cfg.n_embd,
                self.max_frames
            )));
        }
        let bytes = t * cfg.n_embd * 4;
        self.run_with_scratch(
            what,
            |scratch, so| {
                scratch.as_mut_slice()[so.x..so.x + bytes].copy_from_slice(bytemuck::cast_slice(x));
                scratch.flush_cpu_cache(so.x, bytes);
                prepare(scratch, so);
            },
            emit,
            out,
        )
    }

    /// Lock the device and scratch, let `prepare` write what the stage needs
    /// on the host, run `emit` on the queue and return `out` (an offset and a
    /// length in floats) of the scratch buffer afterwards.
    fn run_with_scratch(
        &self,
        what: &str,
        prepare: impl FnOnce(&mut RpcmemBuffer, &ScratchOffsets),
        emit: impl FnOnce(
            &mut HexagonQueueSession,
            &RpcmemBuffer,
            &RpcmemBuffer,
            &ScratchOffsets,
        ) -> Result<(), CeraError>,
        out: (usize, usize),
    ) -> Result<Vec<f32>, CeraError> {
        let so = self.scratch_offsets;
        let mut dev = self.device.lock_or_recover();
        let mut scratch = self.scratch.lock_or_recover();
        prepare(&mut scratch, &so);

        run_on_queue(what, dev.queue_session_mut(), |session| {
            emit(session, &self.weights_buf, &scratch, &so)
        })?;
        let (out_off, out_floats) = out;
        scratch.invalidate_cpu_cache(out_off, out_floats * 4);
        Ok(
            bytemuck::cast_slice::<u8, f32>(&scratch.as_slice()[out_off..out_off + out_floats * 4])
                .to_vec(),
        )
    }

    fn layer(&self, what: &str, layer: usize) -> Result<&LayerOffsets, CeraError> {
        self.offsets
            .layers
            .get(layer)
            .ok_or_else(|| CeraError::Backend(format!("{what}: no layer {layer}")))
    }

    /// Run one feed-forward sub-block of `layer` (`second` picks the one after
    /// the convolution module) and return the updated sequence. Used to check
    /// this stage against `conformer_ffn_forward` on the device.
    pub fn run_ffn(
        &self,
        layer: usize,
        second: bool,
        x: &[f32],
        t: usize,
    ) -> Result<Vec<f32>, CeraError> {
        let lo = self.layer("run_ffn", layer)?;
        let ffn = if second { &lo.ffn2 } else { &lo.ffn1 };
        self.run_stage("run_ffn", x, t, |session, weights, scratch, so| {
            emit_ffn(
                session,
                weights,
                scratch,
                ffn,
                self.offsets.half_off,
                so,
                &self.config,
                t,
            )
        })
    }

    /// Zero the padding the attention's final contraction reads (`attn_sm` and
    /// `attn_vt` rows are written only up to `t`), for sequences of `t` frames.
    /// Done once per sequence: no op touches the padding afterwards.
    fn clear_attention_padding(&self, scratch: &mut RpcmemBuffer, t: usize) {
        let cfg = &self.config;
        let (h, d) = (cfg.n_head, cfg.n_embd / cfg.n_head);
        let (so, tp) = (&self.scratch_offsets, pad32(t));
        let (sm_len, vt_len) = (h * t * tp * 4, h * d * tp * 4);
        scratch.as_mut_slice()[so.attn_sm..so.attn_sm + sm_len].fill(0);
        scratch.as_mut_slice()[so.attn_vt..so.attn_vt + vt_len].fill(0);
        scratch.flush_cpu_cache(so.attn_sm, sm_len);
        scratch.flush_cpu_cache(so.attn_vt, vt_len);
    }

    /// Run the relative-position attention of `layer` and return the updated
    /// sequence. Checked against `conformer_self_attention_forward` on the
    /// device.
    pub fn run_attention(&self, layer: usize, x: &[f32], t: usize) -> Result<Vec<f32>, CeraError> {
        let attn = &self.layer("run_attention", layer)?.attn;
        let pos = relative_pos_emb(t.max(1));
        self.run_stage_with(
            "run_attention",
            x,
            t,
            |scratch, so| {
                self.clear_attention_padding(scratch, t);
                let bytes = pos.len() * 4;
                scratch.as_mut_slice()[so.attn_pos..so.attn_pos + bytes]
                    .copy_from_slice(bytemuck::cast_slice(&pos));
                scratch.flush_cpu_cache(so.attn_pos, bytes);
            },
            |session, weights, scratch, so| {
                emit_attention(
                    session,
                    weights,
                    scratch,
                    attn,
                    self.offsets.attn_scale_off,
                    so,
                    &self.config,
                    t,
                )
            },
            (self.scratch_offsets.x, t * self.config.n_embd),
        )
    }

    /// [`Self::run_attention`], returning the stage's intermediate tensors
    /// (compacted to their logical shapes) as well, so the probe can find the
    /// first one that differs from the CPU's.
    pub fn debug_attention(
        &self,
        layer: usize,
        x: &[f32],
        t: usize,
    ) -> Result<AttentionDump, CeraError> {
        let seq = self.run_attention(layer, x, t)?;
        let cfg = &self.config;
        let (n, h, d) = (cfg.n_embd, cfg.n_head, cfg.n_embd / cfg.n_head);
        let (so, tp, lp, l) = (self.scratch_offsets, pad32(t), pad32(2 * t - 1), 2 * t - 1);
        let scratch = self.scratch.lock_or_recover();
        let f32s = |off: usize, n_floats: usize| -> Vec<f32> {
            scratch.invalidate_cpu_cache(off, n_floats * 4);
            bytemuck::cast_slice::<u8, f32>(&scratch.as_slice()[off..off + n_floats * 4]).to_vec()
        };
        // `[rows, width]` planes per head with a padded row stride, compacted.
        let planes = |off: usize, rows: usize, width: usize, stride: usize| -> Vec<f32> {
            let raw = f32s(off, h * rows * stride);
            let mut out = Vec::with_capacity(h * rows * width);
            for hh in 0..h {
                for r in 0..rows {
                    let start = (hh * rows + r) * stride;
                    out.extend_from_slice(&raw[start..start + width]);
                }
            }
            out
        };
        Ok(AttentionDump {
            q: f32s(so.attn_q, t * n),
            k: f32s(so.attn_k, t * n),
            v: f32s(so.attn_v, t * n),
            p: f32s(so.attn_p, l * n),
            qu: f32s(so.attn_qu, t * n),
            qv: f32s(so.attn_qv, t * n),
            ac: planes(so.attn_ac, t, t, t),
            bd: planes(so.attn_bd, t, l, lp),
            sm: planes(so.attn_smc, t, t, t),
            vt: {
                let raw = f32s(so.attn_vt, h * d * tp);
                let mut out = Vec::with_capacity(h * d * t);
                for row in 0..h * d {
                    out.extend_from_slice(&raw[row * tp..row * tp + t]);
                }
                out
            },
            av: f32s(so.attn_av, t * n),
            out: seq,
        })
    }

    /// Run Conformer blocks `0..n_blocks` over `x` (`[t, n_embd]`), one DSP
    /// batch per block, and return the sequence after the last one.
    pub fn run_blocks(&self, n_blocks: usize, x: &[f32], t: usize) -> Result<Vec<f32>, CeraError> {
        if n_blocks == 0 || n_blocks > self.offsets.layers.len() {
            return Err(CeraError::Backend(format!(
                "run_blocks: {n_blocks} blocks of {}",
                self.offsets.layers.len()
            )));
        }
        let pos = relative_pos_emb(t.max(1));
        let cfg = &self.config;
        let bytes = t * cfg.n_embd * 4;
        self.run_stage_with(
            "run_blocks",
            x,
            t,
            |scratch, so| {
                self.clear_attention_padding(scratch, t);
                let pos_bytes = pos.len() * 4;
                scratch.as_mut_slice()[so.attn_pos..so.attn_pos + pos_bytes]
                    .copy_from_slice(bytemuck::cast_slice(&pos));
                scratch.flush_cpu_cache(so.attn_pos, pos_bytes);
            },
            |session, weights, scratch, so| {
                let _ = bytes;
                for layer in &self.offsets.layers[..n_blocks] {
                    emit_block(session, weights, scratch, layer, &self.offsets, so, cfg, t)?;
                    session.flush()?;
                }
                Ok(())
            },
            (self.scratch_offsets.x, t * self.config.n_embd),
        )
    }

    /// Run every Conformer block and the adapter over the conv stem's output
    /// `x` (`[t, n_embd]`), returning the `[t, llm_hidden_size]` embeddings:
    /// everything after the stem, in one call, one DSP batch per block.
    pub fn encode_stem_output(&self, x: &[f32], t: usize) -> Result<Vec<f32>, CeraError> {
        let pos = relative_pos_emb(t.max(1));
        let cfg = &self.config;
        let out = (self.scratch_offsets.adapter_out, t * cfg.llm_hidden_size);
        self.run_stage_with(
            "encode",
            x,
            t,
            |scratch, so| {
                self.clear_attention_padding(scratch, t);
                let pos_bytes = pos.len() * 4;
                scratch.as_mut_slice()[so.attn_pos..so.attn_pos + pos_bytes]
                    .copy_from_slice(bytemuck::cast_slice(&pos));
                scratch.flush_cpu_cache(so.attn_pos, pos_bytes);
            },
            |session, weights, scratch, so| {
                for layer in &self.offsets.layers {
                    emit_block(session, weights, scratch, layer, &self.offsets, so, cfg, t)?;
                    session.flush()?;
                }
                emit_adapter(session, weights, scratch, &self.offsets.adapter, so, cfg, t)?;
                session.flush()
            },
            out,
        )
    }

    /// The whole encoder for mono 16 kHz PCM, on the NPU from the samples to
    /// the embeddings: log-mel, conv stem, blocks and adapter. Same output as
    /// [`crate::model::audio_encoder::encode_audio_pcm`]. A clip too long for
    /// the NPU encoder is refused before any work is done.
    pub fn encode(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize), CeraError> {
        let n_frames = n_frames_for(pcm.len());
        if n_frames == 0 {
            return Ok((Vec::new(), 0));
        }
        self.stem_geom(n_frames)?;
        let mel = self.log_mel_npu(pcm, n_frames)?;
        self.encode_mel(&mel, n_frames)
    }

    /// Log-mel for `pcm` (`n_frames` frames, [`n_frames_for`]): the DFT and
    /// the filterbank on the DSP, the log and the normalization on the host.
    pub fn log_mel_npu(&self, pcm: &[f32], n_frames: usize) -> Result<Vec<f32>, CeraError> {
        let n_mel = self.config.n_mel_bins;
        let samples = padded_preemphasized(pcm)
            .ok_or_else(|| CeraError::Backend("log-mel: input too long".into()))?;
        let so = MelScratch::new(samples.len(), n_frames, n_mel);
        let mut buf = RpcmemBuffer::alloc(Arc::clone(&self.driver), so.total_bytes, true)?;
        stage_samples(buf.as_mut_slice(), &so, &samples);
        buf.flush_cpu_cache(0, so.total_bytes);
        let run = {
            let mut dev = self.device.lock_or_recover();
            run_on_queue("log-mel", dev.queue_session_mut(), |session| {
                emit_mel(
                    session,
                    &self.weights_buf,
                    &buf,
                    &self.offsets.mel,
                    &so,
                    n_frames,
                    n_mel,
                )
            })
        };
        let energies = run.map(|()| {
            buf.invalidate_cpu_cache(so.mel, n_frames * n_mel * 4);
            let floats: &[f32] = bytemuck::cast_slice(buf.as_slice());
            floats[so.mel / 4..so.mel / 4 + n_frames * n_mel].to_vec()
        });
        self.release_buffer(&buf);
        Ok(finish_mel(&energies?, n_mel, n_frames, pcm.len()))
    }

    /// The stem's geometry for `n_frames` mel frames, refused when the
    /// sequence it produces is longer than the NPU encoder stages.
    fn stem_geom(&self, n_frames: usize) -> Result<StemGeom, CeraError> {
        let ch = self.weights.conv_stem.layers[0].bias.len();
        let g = StemGeom::new(n_frames, self.config.n_mel_bins, ch)
            .ok_or_else(|| CeraError::Backend(format!("conv stem: {n_frames} mel frames")))?;
        if g.t_out() > self.max_frames {
            return Err(CeraError::Backend(format!(
                "{} encoder frames exceed the NPU encoder's {} (about {} s of audio)",
                g.t_out(),
                self.max_frames,
                self.max_frames * 8 * 160 / 16_000
            )));
        }
        Ok(g)
    }

    /// The stem's activation buffer for `g`, with the host-written parts of
    /// the input filled in.
    fn staged_stem_buffer(
        &self,
        g: &StemGeom,
        mel: &[f32],
        keep_stages: bool,
    ) -> Result<(RpcmemBuffer, StemScratch), CeraError> {
        let so = StemScratch::new(g, !keep_stages);
        let mut st = RpcmemBuffer::alloc(Arc::clone(&self.driver), so.total_bytes, true)?;
        stage_input(st.as_mut_slice(), &so, g, mel);
        st.flush_cpu_cache(0, so.total_bytes);
        Ok((st, so))
    }

    /// Log-mel in, embeddings out, with the stem on the NPU. The stem's
    /// activations (tens of MB for a long clip) live in a buffer that exists
    /// only for this call.
    pub fn encode_mel(&self, mel: &[f32], n_frames: usize) -> Result<(Vec<f32>, usize), CeraError> {
        let g = self.stem_geom(n_frames)?;
        let t = g.t_out();
        let cfg = &self.config;
        let (st, st_so) = self.staged_stem_buffer(&g, mel, false)?;
        let pos = relative_pos_emb(t.max(1));
        let out = (self.scratch_offsets.adapter_out, t * cfg.llm_hidden_size);
        let run = self.run_with_scratch(
            "encode",
            |scratch, so| {
                self.clear_attention_padding(scratch, t);
                let pos_bytes = pos.len() * 4;
                scratch.as_mut_slice()[so.attn_pos..so.attn_pos + pos_bytes]
                    .copy_from_slice(bytemuck::cast_slice(&pos));
                scratch.flush_cpu_cache(so.attn_pos, pos_bytes);
            },
            |session, weights, scratch, so| {
                emit_stem(
                    session,
                    weights,
                    &st,
                    scratch,
                    so.x,
                    &self.offsets.stem,
                    &st_so,
                    &g,
                    &mut |s| s.flush(),
                )?;
                for layer in &self.offsets.layers {
                    emit_block(session, weights, scratch, layer, &self.offsets, so, cfg, t)?;
                    session.flush()?;
                }
                emit_adapter(session, weights, scratch, &self.offsets.adapter, so, cfg, t)?;
                session.flush()
            },
            out,
        );
        self.release_buffer(&st);
        Ok((run?, t))
    }

    /// Tell the DSP to let go of a per-call buffer before it is unmapped; an
    /// unmap while the DSP still holds a reference fails and leaks the mapping.
    /// Runs after every call, successful or not.
    fn release_buffer(&self, st: &RpcmemBuffer) {
        self.device
            .lock_or_recover()
            .queue_session_mut()
            .release_dsp_reference(st);
    }

    /// Run only the stem on the NPU and return each stage's output in the CPU
    /// stem's layouts, for the probe to compare against the CPU.
    pub fn debug_stem(&self, mel: &[f32], n_frames: usize) -> Result<StemDump, CeraError> {
        let g = self.stem_geom(n_frames)?;
        let t = g.t_out();
        let (st, so) = self.staged_stem_buffer(&g, mel, true)?;
        let out = (self.scratch_offsets.x, t * self.config.n_embd);
        let run = self.run_with_scratch(
            "debug_stem",
            |_, _| (),
            |session, weights, scratch, sco| {
                emit_stem(
                    session,
                    weights,
                    &st,
                    scratch,
                    sco.x,
                    &self.offsets.stem,
                    &so,
                    &g,
                    &mut |s| s.flush(),
                )?;
                session.flush()
            },
            out,
        );
        // Stage outputs are read below, so the buffer is released only after.
        let x = match run {
            Ok(x) => x,
            Err(e) => {
                self.release_buffer(&st);
                return Err(e);
            }
        };
        st.invalidate_cpu_cache(0, so.total_bytes);
        let all: &[f32] = bytemuck::cast_slice(st.as_slice());
        let at = |off: usize, len: usize| &all[off / 4..off / 4 + len];
        let ch = g.ch;
        let dump = StemDump {
            l0: to_channel_major(
                at(so.b0, (g.h1 + 2) * (g.w1 + 2) * ch),
                (g.h1, g.w1, ch),
                g.w1 + 2,
                g.w1 + 3,
            ),
            dw1: to_channel_major(at(so.b1, g.h2 * g.w2 * ch), (g.h2, g.w2, ch), g.w2, 0),
            pw2: to_channel_major(at(so.pw2, g.h2 * g.w2 * ch), (g.h2, g.w2, ch), g.w2, 0),
            dw3: to_channel_major(at(so.b3, g.h3 * g.w3 * ch), (g.h3, g.w3, ch), g.w3, 0),
            pw4: to_channel_major(at(so.b4, g.h3 * g.w3 * ch), (g.h3, g.w3, ch), g.w3, 0),
            flat: at(so.flat, g.h3 * ch * g.w3).to_vec(),
            out: x,
            t,
        };
        self.release_buffer(&st);
        Ok(dump)
    }

    /// Run the convolution module of `layer` and return the updated sequence.
    /// Checked against `conformer_conv_module_forward` on the device.
    pub fn run_conv(&self, layer: usize, x: &[f32], t: usize) -> Result<Vec<f32>, CeraError> {
        let conv = &self.layer("run_conv", layer)?.conv;
        self.run_stage("run_conv", x, t, |session, weights, scratch, so| {
            emit_conv_module(
                session,
                weights,
                scratch,
                conv,
                self.offsets.zeros_off,
                self.offsets.kernel,
                so,
                &self.config,
                t,
            )
        })
    }
}

impl Drop for HexagonAudioEncoder {
    /// The DSP holds references to the weights and scratch from the batches
    /// that read them; telling it to let go first keeps the unmaps that follow
    /// from failing (and from logging for every encoder dropped).
    fn drop(&mut self) {
        let mut dev = self.device.lock_or_recover();
        let session = dev.queue_session_mut();
        session.release_dsp_reference(&self.weights_buf);
        session.release_dsp_reference(&self.scratch.lock_or_recover());
    }
}

impl AudioGpuEncode for HexagonAudioEncoder {
    fn encode_pcm(&self, pcm: &[f32]) -> anyhow::Result<(Vec<f32>, usize)> {
        Ok(self.encode(pcm)?)
    }
}

/// Longest sequence the NPU encoder is sized for (frames after the stem's 8x
/// subsampling: about 32 s of audio). Longer clips fall back to the CPU.
pub const MAX_FRAMES: usize = 400;

/// Stage the encoder on the NPU, or `None` (CPU encode) when there is no
/// usable NPU or the weights cannot be staged.
pub fn try_hexagon_audio_encoder(
    weights: &Arc<AudioEncoderWeights>,
) -> Option<Arc<dyn AudioGpuEncode>> {
    let context = crate::backend::hexagon::HexagonContext::new()
        .inspect_err(|e| {
            crate::backend::hexagon::log_context_unavailable("HexagonAudioEncoder", e);
        })
        .ok()?;
    let arch_override = std::env::var("CERA_HEXAGON_ARCH")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .and_then(crate::backend::hexagon::HexagonArch::from_u32);
    let dev = match crate::backend::hexagon::probe_device(context.driver(), arch_override) {
        Ok(d) => d,
        Err(e) => {
            tracing::info!("HexagonAudioEncoder: DSP device unavailable ({e}), falling back");
            return None;
        }
    };
    let device = Arc::new(Mutex::new(dev));
    match HexagonAudioEncoder::new(Arc::clone(context.driver()), device, weights, MAX_FRAMES) {
        Ok(encoder) => {
            tracing::info!("audio encoder: using native Hexagon NPU backend");
            Some(Arc::new(encoder))
        }
        Err(e) => {
            crate::backend::hexagon::hexagon_error!("failed to create HexagonAudioEncoder: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::hexagon::HtpOpCode;
    use crate::backend::hexagon::dispatch::testing::RecordingSink;

    fn cfg() -> AudioEncoderConfig {
        AudioEncoderConfig {
            n_layer: 2,
            n_embd: 64,
            n_ff: 128,
            n_head: 4,
            eps: 1e-5,
            n_mel_bins: 16,
            llm_hidden_size: 96,
        }
    }

    fn ffn_offsets() -> FfnOffsets {
        let desc = |offset, rows, cols| HexagonWeightDesc {
            offset,
            size_bytes: 1000,
            format: HexagonWeightFormat::RepackedQ4_0,
            rows,
            cols,
        };
        FfnOffsets {
            norm_w: 0,
            norm_b: 256,
            up_w: desc(1024, 128, 64),
            up_b: 4096,
            down_w: desc(8192, 64, 128),
            down_b: 12288,
        }
    }

    #[test]
    fn scratch_regions_do_not_overlap_and_are_aligned() {
        let so = ScratchOffsets::new(&cfg(), 100, 9);
        let regions = [
            (so.x, 100 * 64 * 4),
            (so.norm, 100 * 64 * 4),
            (so.ff, 100 * 128 * 4),
            (so.out, 100 * 64 * 4),
            (so.conv_a, 100 * 64 * 4),
            (so.conv_g, 100 * 64 * 4),
            (so.conv_tmp, 104 * 64 * 4),
            (so.conv_x, 108 * 64 * 4),
            (so.conv_y, 100 * 64 * 4),
            (so.attn_pos, 199 * POS_EMB_DIM * 4),
            (so.attn_p, 199 * 64 * 4),
            (so.attn_q, 100 * 64 * 4),
            (so.attn_k, 100 * 64 * 4),
            (so.attn_v, 100 * 64 * 4),
            (so.attn_qu, 100 * 64 * 4),
            (so.attn_qv, 100 * 64 * 4),
            (so.attn_ac, 4 * 100 * 100 * 4),
            (so.attn_smc, 4 * 100 * 100 * 4),
            (so.attn_sm, 4 * 100 * 128 * 4),
            (so.attn_bd, 4 * 100 * 224 * 4),
            (so.attn_vt, 4 * 16 * 128 * 4),
            (so.attn_av, 100 * 64 * 4),
        ];
        for (i, (off, len)) in regions.iter().enumerate() {
            assert_eq!(off % 128, 0, "region {i} unaligned");
            let end = off + len;
            assert!(end <= so.total_bytes);
            if let Some((next, _)) = regions.get(i + 1) {
                assert!(end <= *next, "region {i} overlaps the next");
            }
        }
    }

    /// The feed-forward is norm (3 ops), up (matmul + bias), silu, down
    /// (matmul + bias), the half-step scale, and the residual add: the order
    /// and the operands are what make it match `conformer_ffn_forward`.
    #[test]
    fn ffn_emits_the_macaron_sequence() {
        let (cfg, so) = (cfg(), ScratchOffsets::new(&cfg(), 10, 9));
        let mut s = RecordingSink::default();
        emit_ffn(&mut s, &"w", &"s", &ffn_offsets(), 16384, &so, &cfg, 10).unwrap();
        let op = |o: HtpOpCode| o as u32;
        assert_eq!(
            s.opcodes(),
            vec![
                op(HtpOpCode::Norm),
                op(HtpOpCode::Mul),
                op(HtpOpCode::Add),
                op(HtpOpCode::MulMat),
                op(HtpOpCode::Add),
                op(HtpOpCode::UnarySilu),
                op(HtpOpCode::MulMat),
                op(HtpOpCode::Add),
                op(HtpOpCode::Mul),
                op(HtpOpCode::Add),
            ]
        );
        // The residual adds the scaled branch (`out`) into the sequence (`x`).
        let last = s.ops.len() - 1;
        assert_eq!(s.src(last, 0).offset, so.x);
        assert_eq!(s.src(last, 1).offset, so.out);
        // The scale reads the 0.5 vector, not a weight.
        let scale = s.ops.len() - 2;
        assert_eq!(s.src(scale, 1).offset, 16384);
        assert_eq!(s.src(scale, 0).offset, so.out);
        // Norm reads x and writes the scratch the up projection reads.
        assert_eq!(s.src(0, 0).offset, so.x);
        assert_eq!(s.dst(0).offset, so.norm);
        assert_eq!(s.src(3, 1).offset, so.norm);
        assert_eq!(s.dst(3).offset, so.ff);
        // Up widens to n_ff, down narrows back to n_embd.
        assert_eq!(s.dst(3).ne, [128, 10, 1, 1]);
        assert_eq!(s.dst(6).ne, [64, 10, 1, 1]);
        assert_eq!(s.dst(6).offset, so.out);
    }

    fn conv_offsets() -> ConvOffsets {
        let desc = |offset| HexagonWeightDesc {
            offset,
            size_bytes: 1000,
            format: HexagonWeightFormat::RepackedQ4_0,
            rows: 64,
            cols: 64,
        };
        ConvOffsets {
            norm_w: 0,
            norm_b: 256,
            pw1a_w: desc(1024),
            pw1a_b: 4096,
            pw1b_w: desc(5120),
            pw1b_b: 8192,
            dw_w: 9216,
            dw_b: 12288,
            conv_norm_w: 13312,
            conv_norm_b: 14336,
            pw2_w: desc(15360),
            pw2_b: 19456,
        }
    }

    /// The conv module is norm, the two GLU matmuls (value and gate halves),
    /// sigmoid on the gate, the product, the two padding concats, SsmConv, the
    /// depthwise bias and per-channel affine, SiLU, pw2, and the residual.
    #[test]
    fn conv_module_pads_symmetrically_and_runs_a_valid_window() {
        let (cfg, so) = (cfg(), ScratchOffsets::new(&cfg(), 10, 9));
        let mut s = RecordingSink::default();
        emit_conv_module(&mut s, &"w", &"s", &conv_offsets(), 20000, 9, &so, &cfg, 10).unwrap();
        let op = |o: HtpOpCode| o as u32;
        assert_eq!(
            s.opcodes(),
            vec![
                op(HtpOpCode::Norm),
                op(HtpOpCode::Mul),
                op(HtpOpCode::Add),
                op(HtpOpCode::MulMat),
                op(HtpOpCode::Add),
                op(HtpOpCode::MulMat),
                op(HtpOpCode::Add),
                op(HtpOpCode::UnarySigmoid),
                op(HtpOpCode::Mul),
                op(HtpOpCode::Concat),
                op(HtpOpCode::Concat),
                op(HtpOpCode::SsmConv),
                op(HtpOpCode::Add),
                op(HtpOpCode::Mul),
                op(HtpOpCode::Add),
                op(HtpOpCode::UnarySilu),
                op(HtpOpCode::MulMat),
                op(HtpOpCode::Add),
                op(HtpOpCode::Add),
            ]
        );
        // GLU: the value half times the sigmoid of the gate half.
        assert_eq!(s.dst(3).offset, so.conv_a);
        assert_eq!(s.dst(5).offset, so.conv_g);
        assert_eq!(
            (s.src(8, 0).offset, s.src(8, 1).offset),
            (so.conv_a, so.conv_g)
        );
        // First concat: 4 zero rows from the weights, then the transposed GLU
        // output (position step = a row, channel step = one float).
        assert_eq!(s.src(9, 0).offset, 20000);
        assert_eq!(s.src(9, 0).ne, [4, 64, 1, 1]);
        assert_eq!(s.src(9, 1).nb[..2], [64 * 4, 4]);
        assert_eq!(s.dst(9).ne, [14, 64, 1, 1]);
        assert_eq!(s.dst(9).offset, so.conv_tmp);
        // Second concat appends the other 4 zero rows: 4 + 10 + 4 = t + k - 1.
        assert_eq!(s.src(10, 0).ne, [14, 64, 1, 1]);
        assert_eq!(s.src(10, 1).ne, [4, 64, 1, 1]);
        assert_eq!(s.dst(10).ne, [18, 64, 1, 1]);
        assert_eq!(s.dst(10).offset, so.conv_x);
        // SsmConv: 9 taps over the padded input, one output per frame.
        assert_eq!(s.src(11, 0).ne, [18, 64, 1, 1]);
        assert_eq!(s.src(11, 1).ne, [9, 64, 1, 1]);
        assert_eq!(s.dst(11).ne, [64, 10, 1, 1]);
        assert_eq!(s.dst(11).offset, so.conv_y);
        // The residual adds pw2's output back into the sequence.
        let last = s.ops.len() - 1;
        assert_eq!(
            (s.src(last, 0).offset, s.src(last, 1).offset),
            (so.x, so.out)
        );
    }

    fn attn_offsets() -> AttnOffsets {
        let desc = |offset| HexagonWeightDesc {
            offset,
            size_bytes: 1000,
            format: HexagonWeightFormat::RepackedQ4_0,
            rows: 64,
            cols: 64,
        };
        AttnOffsets {
            ln_w: 0,
            ln_b: 256,
            q_w: desc(1024),
            q_b: 3072,
            k_w: desc(4096),
            k_b: 6144,
            v_w: desc(7168),
            v_b: 9216,
            o_w: desc(10240),
            o_b: 12288,
            pos_bias_u: 13312,
            pos_bias_v: 14336,
            linear_pos_w: desc(15360),
        }
    }

    /// The attention reads K, Q+u, P and (Q+v)*s through per-head views and
    /// takes the position term as a shifted view of the position scores: the
    /// geometry here is what makes it equal the reference's explicit rel-shift.
    #[test]
    fn attention_shifts_the_position_scores_with_a_strided_view() {
        let (cfg, t) = (cfg(), 10usize);
        let so = ScratchOffsets::new(&cfg, t, 9);
        let mut s = RecordingSink::default();
        emit_attention(&mut s, &"w", &"s", &attn_offsets(), 20000, &so, &cfg, t).unwrap();
        let op = |o: HtpOpCode| o as u32;
        let ops = s.opcodes();
        let matmuls: Vec<usize> = (0..ops.len())
            .filter(|&i| ops[i] == op(HtpOpCode::MulMat))
            .collect();
        // q, k, v, linear_pos, scores, position scores, attn @ V, out.
        assert_eq!(matmuls.len(), 8);
        let (h, d, l, tp, lp) = (4u32, 16u32, 19u32, 32usize, 32usize);

        let softmax = ops
            .iter()
            .position(|&o| o == op(HtpOpCode::Softmax))
            .unwrap();
        // The scores are the softmax input; the position term is its mask, a
        // view into the position scores that starts t - 1 elements in and
        // steps a row less one element, so row q begins t - 1 - q further on.
        assert_eq!(s.src(softmax, 0).offset, so.attn_ac);
        let mask = s.src(softmax, 1);
        assert_eq!(mask.offset, so.attn_bd + (t - 1) * 4);
        assert_eq!(mask.ne, [10, 10, h, 1]);
        assert_eq!(mask.nb[1] as usize, (lp - 1) * 4);
        assert_eq!(mask.nb[2] as usize, t * lp * 4);
        // The softmax reads and writes contiguous rows (its kernel addresses
        // them by length); the zero-padded copy `attn @ V` reads follows.
        assert_eq!(s.src(softmax, 0).nb[1], 10 * 4);
        assert_eq!(s.dst(softmax).offset, so.attn_smc);
        assert_eq!(s.dst(softmax).nb[1], 10 * 4);
        let pad_copy = softmax + 1;
        assert_eq!(s.opcodes()[pad_copy], op(HtpOpCode::Cpy));
        assert_eq!(s.dst(pad_copy).offset, so.attn_sm);
        assert_eq!(s.dst(pad_copy).nb[1] as usize, tp * 4);
        let bd = matmuls[5];
        assert_eq!(s.dst(bd).offset, so.attn_bd);
        assert_eq!(s.dst(bd).ne, [l, 10, h, 1]);
        // Per-head views: d_head wide, the row stride is a full embedding row.
        let (k_view, qu_view) = (s.src(matmuls[4], 0), s.src(matmuls[4], 1));
        assert_eq!((k_view.offset, qu_view.offset), (so.attn_k, so.attn_qu));
        assert_eq!(k_view.ne, [d, 10, h, 1]);
        assert_eq!(k_view.nb[..3], [4, 64 * 4, d * 4]);
        // The position term uses the scaled copy of Q + v.
        assert_eq!(s.src(bd, 1).offset, so.attn_qv);
        assert_eq!(s.src(bd, 0).offset, so.attn_p);
        // V is transposed by a copy whose source steps over frames.
        let cpy = ops.iter().rposition(|&o| o == op(HtpOpCode::Cpy)).unwrap();
        assert_eq!(s.src(cpy, 0).nb[..2], [64 * 4, 4]);
        assert_eq!(s.dst(cpy).offset, so.attn_vt);
        // attn @ V contracts over the padded key length, so it needs the
        // padding of both operands to be zero.
        let av = matmuls[6];
        assert_eq!(s.src(av, 0).ne, [tp as u32, d, h, 1]);
        assert_eq!(s.src(av, 1).ne, [tp as u32, 10, h, 1]);
        assert_eq!(s.dst(av).offset, so.attn_av);
        // The residual closes the block.
        let last = ops.len() - 1;
        assert_eq!(
            (s.src(last, 0).offset, s.src(last, 1).offset),
            (so.x, so.out)
        );
    }

    /// The adapter is norm, an up projection to the wide activation, GELU, and
    /// a down projection into the LLM's width: the embeddings land in their own
    /// region, not over the sequence.
    #[test]
    fn adapter_projects_into_the_llm_width() {
        let (cfg, so) = (cfg(), ScratchOffsets::new(&cfg(), 10, 9));
        let desc = |offset, rows, cols| HexagonWeightDesc {
            offset,
            size_bytes: 1000,
            format: HexagonWeightFormat::RepackedQ4_0,
            rows,
            cols,
        };
        let ad = AdapterOffsets {
            norm_w: 0,
            norm_b: 256,
            up_w: desc(1024, 128, 64),
            up_b: 4096,
            down_w: desc(8192, 96, 128),
            down_b: 12288,
        };
        let mut s = RecordingSink::default();
        emit_adapter(&mut s, &"w", &"s", &ad, &so, &cfg, 10).unwrap();
        let op = |o: HtpOpCode| o as u32;
        assert_eq!(
            s.opcodes(),
            vec![
                op(HtpOpCode::Norm),
                op(HtpOpCode::Mul),
                op(HtpOpCode::Add),
                op(HtpOpCode::MulMat),
                op(HtpOpCode::Add),
                op(HtpOpCode::UnaryGelu),
                op(HtpOpCode::MulMat),
                op(HtpOpCode::Add),
            ]
        );
        assert_eq!(s.dst(3).ne, [128, 10, 1, 1]);
        assert_eq!(s.dst(6).ne, [96, 10, 1, 1]);
        assert_eq!(s.dst(6).offset, so.adapter_out);
    }

    /// Sequences past what the scratch was sized for are refused before any op
    /// is queued, so the session falls back to the CPU encoder.
    #[test]
    fn scratch_is_sized_for_the_longest_sequence() {
        let so = ScratchOffsets::new(&cfg(), 400, 9);
        assert!(so.total_bytes > 400 * 64 * 4 * 8);
        assert_eq!(so.adapter_out % 128, 0);
        assert!(so.adapter_out + 400 * 96 * 4 <= so.total_bytes);
    }

    #[test]
    fn linear_planning_rejects_dense_weights() {
        let w = MmapWeight::from_owned_f32(vec![0.0; 64 * 64], 64, 64);
        let mut cur = 0;
        let err = plan_linear(&mut cur, &w).unwrap_err().to_string();
        assert!(err.contains("Q8_0 or Q4_0"), "{err}");
    }
}
