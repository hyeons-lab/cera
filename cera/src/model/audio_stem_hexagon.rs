//! The FastConformer conv stem on the Hexagon NPU.
//!
//! The stem turns a `[frames, mel bins]` log-mel spectrogram into the
//! `[frames / 8, n_embd]` sequence the Conformer blocks consume. On the CPU it
//! is the largest cost left in a speech turn once the blocks run on the DSP
//! (about 0.3 CPU-seconds for 10 s of audio), so it moves too:
//!
//! ```text
//! mel [T, F]
//! -> 3x3 conv, stride 2, 1 -> C channels, +bias, ReLU     im2col + F32 matmul
//! -> 3x3 depthwise conv, stride 2, +bias                  nine taps: gather, scale, add
//! -> 1x1 conv, C -> C, +bias, ReLU                        F32 matmul
//! -> 3x3 depthwise conv, stride 2, +bias                  nine taps
//! -> 1x1 conv, C -> C, +bias, ReLU                        F32 matmul
//! -> flatten (channel, freq) per time step, linear to n_embd
//! ```
//!
//! Activations are channel-last (`[position, channel]`, channels contiguous):
//! the pointwise convolutions become plain matmuls over positions, the
//! per-channel taps and biases become row broadcasts, and a tap of a strided
//! convolution is a strided view of the input. The two convolutions that read
//! a 3x3 neighbourhood take their input in a buffer with a one-element zero
//! border, so padding costs no ops.
//!
//! Everything here is emitted through `backend::hexagon::dispatch`, so the
//! host tests record the op sequence without a device.

use crate::backend::hexagon::dispatch::{self, OpSink, TokenShape, TokenTile, View};
use crate::backend::hexagon::{HexagonWeightDesc, align128};
use crate::model::audio_encoder::{AudioEncoderConfig, ConvLayerWeights, ConvStemWeights};
use crate::model::audio_encoder_hexagon::{plan_linear, plan_vec, put_linear, put_vec};
use crate::session::CeraError;

/// The first convolution's im2col row: nine taps, a constant one that carries
/// the bias, and zeros up to the 32-element multiple the F32 matmul needs.
pub(crate) const COL_K: usize = 32;
const TAPS: usize = 9;
/// Index of the constant-one column in an im2col row.
const BIAS_COL: usize = TAPS;

/// Token tile for the final quantised linear. Its 4096-wide input makes each
/// row twice as much VTCM as the blocks' widest linear, so it takes half their
/// tile (64 rows overflowed the VTCM on the device).
const LINEAR_TILE: TokenTile = TokenTile::Tiles(16);

/// Output length of a 3x3, stride 2, pad 1 convolution over `n` elements.
fn down(n: usize) -> usize {
    (n - 1) / 2 + 1
}

/// Shapes of every stem stage for one clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StemGeom {
    /// Mel frames and bins in.
    pub frames: usize,
    pub bins: usize,
    /// Channels of every convolution.
    pub ch: usize,
    /// Height (time) and width (frequency) after each stride-2 convolution.
    pub h1: usize,
    pub w1: usize,
    pub h2: usize,
    pub w2: usize,
    pub h3: usize,
    pub w3: usize,
}

impl StemGeom {
    pub(crate) fn new(frames: usize, bins: usize, ch: usize) -> Option<Self> {
        if frames == 0 || bins == 0 || ch == 0 {
            return None;
        }
        let (h1, w1) = (down(frames), down(bins));
        let (h2, w2) = (down(h1), down(w1));
        let (h3, w3) = (down(h2), down(w2));
        Some(Self {
            frames,
            bins,
            ch,
            h1,
            w1,
            h2,
            w2,
            h3,
            w3,
        })
    }

    /// Frames the Conformer blocks see.
    pub(crate) fn t_out(&self) -> usize {
        self.h3
    }

    /// Width of the vector the final linear projects, per output frame.
    pub(crate) fn flat_dim(&self) -> usize {
        self.ch * self.w3
    }
}

/// Output frames the NPU stem produces per chunk. A chunk's buffers scale with
/// it (about 32 MB at 64, against 197 MB for a whole 32 s clip) and a longer
/// clip just runs more chunks.
pub(crate) const STEM_CHUNK_ROWS: usize = 64;

/// One time slice of the stem: which mel rows to run and which output frames
/// of it to keep.
///
/// Every stage is a stride-2 3x3 convolution, so output frame `t` reads mel
/// rows `8t - 7 ..= 8t + 7`. A slice that starts at a multiple of 8 and ends at
/// a multiple of 8 (or at the clip's end, where zero padding is the true
/// behaviour) computes every frame exactly like the whole clip does, except
/// the first: its receptive field reaches the row before the slice, which the
/// slice pads with zeros instead of the neighbouring audio. So a chunk after
/// the first starts one frame early (8 mel rows) and drops that frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StemChunk {
    /// First output frame this chunk contributes, and how many.
    pub first_row: usize,
    pub rows: usize,
    /// The mel rows to run.
    pub mel_start: usize,
    pub mel_rows: usize,
    /// Leading output frames of the run that are not kept (0 or 1).
    pub skip: usize,
}

/// Split a clip of `n_frames` mel frames into chunks of at most `rows` output
/// frames.
pub(crate) fn plan_chunks(n_frames: usize, bins: usize, rows: usize) -> Vec<StemChunk> {
    let Some(whole) = StemGeom::new(n_frames, bins, 1) else {
        return Vec::new();
    };
    let t_out = whole.t_out();
    let rows = rows.max(1);
    (0..t_out)
        .step_by(rows)
        .map(|first_row| {
            let end = (first_row + rows).min(t_out);
            let skip = usize::from(first_row > 0);
            let mel_start = 8 * (first_row - skip);
            let mel_end = if end == t_out { n_frames } else { 8 * end };
            StemChunk {
                first_row,
                rows: end - first_row,
                mel_start,
                mel_rows: mel_end - mel_start,
                skip,
            }
        })
        .collect()
}

/// Where the stem's weights sit in `weights_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StemWeightOffsets {
    /// `[ch, COL_K]` F32: the first convolution's taps, bias and zero padding.
    pub l0_w: usize,
    /// Depthwise taps, tap-major `[9, ch]` F32, and biases `[ch]`.
    pub dw1_w: usize,
    pub dw1_b: usize,
    pub dw3_w: usize,
    pub dw3_b: usize,
    /// Pointwise weights `[ch, ch]` F32 (output-major) and biases.
    pub pw2_w: usize,
    pub pw2_b: usize,
    pub pw4_w: usize,
    pub pw4_b: usize,
    /// The flatten-and-project linear.
    pub out_w: HexagonWeightDesc,
    pub out_b: usize,
}

impl StemWeightOffsets {
    pub(crate) fn plan(
        cur: &mut usize,
        stem: &ConvStemWeights,
        cfg: &AudioEncoderConfig,
    ) -> Result<Self, CeraError> {
        let bad = |what: &str| {
            CeraError::Backend(format!(
                "Hexagon audio stem: {what}; the CPU stem handles it"
            ))
        };
        if stem.layers.len() != 5 {
            return Err(bad(&format!(
                "{} conv layers, expected 5",
                stem.layers.len()
            )));
        }
        let ch = stem.layers[0].shape.get(3).copied().unwrap_or(0);
        let shape_is = |i: usize, want: [usize; 4]| stem.layers[i].shape == want;
        if ch == 0
            || !(shape_is(0, [3, 3, 1, ch])
                && shape_is(1, [3, 3, 1, ch])
                && shape_is(2, [1, 1, ch, ch])
                && shape_is(3, [3, 3, 1, ch])
                && shape_is(4, [1, 1, ch, ch]))
            || stem.layers.iter().any(|l| l.bias.len() != ch)
        {
            return Err(bad("unexpected conv layer shapes"));
        }
        let out_w = &stem.pre_encode_out_w;
        let flat = StemGeom::new(1, cfg.n_mel_bins, ch).map_or(0, |g| g.flat_dim());
        if out_w.rows != cfg.n_embd
            || out_w.cols != flat
            || stem.pre_encode_out_b.len() != cfg.n_embd
        {
            return Err(bad("unexpected output projection shape"));
        }
        let l0_w = *cur;
        *cur += align128(ch * COL_K * 4);
        let mut vec = |len: usize| plan_vec(cur, len);
        let (dw1_w, dw1_b) = (vec(TAPS * ch), vec(ch));
        let (pw2_w, pw2_b) = (vec(ch * ch), vec(ch));
        let (dw3_w, dw3_b) = (vec(TAPS * ch), vec(ch));
        let (pw4_w, pw4_b) = (vec(ch * ch), vec(ch));
        Ok(Self {
            l0_w,
            dw1_w,
            dw1_b,
            dw3_w,
            dw3_b,
            pw2_w,
            pw2_b,
            pw4_w,
            pw4_b,
            out_w: plan_linear(cur, out_w)?,
            out_b: plan_vec(cur, cfg.n_embd),
        })
    }
}

/// The first convolution as a matmul operand: one row per output channel,
/// `[9 taps | bias | zeros]` up to [`COL_K`] (the GGUF holds `[channel][tap]`
/// with the taps row-major over the 3x3 window, the order the im2col columns
/// use).
fn first_conv_rows(layer: &ConvLayerWeights) -> Vec<f32> {
    let ch = layer.bias.len();
    let mut rows = vec![0.0f32; ch * COL_K];
    for (oc, row) in rows.as_chunks_mut::<COL_K>().0.iter_mut().enumerate() {
        row[..TAPS].copy_from_slice(&layer.weight[oc * TAPS..(oc + 1) * TAPS]);
        row[BIAS_COL] = layer.bias[oc];
    }
    rows
}

/// Depthwise taps from the GGUF's `[channel][tap]` to tap-major `[tap, channel]`,
/// which makes each tap's per-channel weights one contiguous row to broadcast.
fn tap_major(w: &[f32], ch: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; TAPS * ch];
    for c in 0..ch {
        for t in 0..TAPS {
            out[t * ch + c] = w[c * TAPS + t];
        }
    }
    out
}

/// Copy the stem's weights to their planned places in `dst` (`weights_buf`).
pub(crate) fn put_stem(
    dst: &mut [u8],
    o: &StemWeightOffsets,
    stem: &ConvStemWeights,
) -> Result<(), CeraError> {
    let ch = stem.layers[0].bias.len();
    put_vec(dst, o.l0_w, &first_conv_rows(&stem.layers[0]));
    put_vec(dst, o.dw1_w, &tap_major(&stem.layers[1].weight, ch));
    put_vec(dst, o.dw1_b, &stem.layers[1].bias);
    put_vec(dst, o.pw2_w, &stem.layers[2].weight);
    put_vec(dst, o.pw2_b, &stem.layers[2].bias);
    put_vec(dst, o.dw3_w, &tap_major(&stem.layers[3].weight, ch));
    put_vec(dst, o.dw3_b, &stem.layers[3].bias);
    put_vec(dst, o.pw4_w, &stem.layers[4].weight);
    put_vec(dst, o.pw4_b, &stem.layers[4].bias);
    put_linear(dst, o.out_w, &stem.pre_encode_out_w)?;
    put_vec(dst, o.out_b, &stem.pre_encode_out_b);
    Ok(())
}

/// Regions of the stem's own activation buffer for one clip.
///
/// With `alias` the buffers that are dead by the time a later stage needs room
/// share it (the im2col matrix, the pointwise outputs and the flattened
/// features all reuse the large temporary), which is what a normal run wants.
/// Without it every stage keeps its output, which is what the probe reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StemScratch {
    /// Zero-bordered mel, `[frames + 2, bins + 2]`.
    pub mel_pad: usize,
    /// im2col rows `[h1 * w1, COL_K]`.
    pub col: usize,
    /// First conv's output, zero-bordered `[h1 + 2, w1 + 2, ch]`.
    pub b0: usize,
    /// First depthwise conv's accumulator `[h2 * w2, ch]`.
    pub b1: usize,
    /// Gather scratch for the depthwise taps `[h2 * w2, ch]`.
    pub tmp: usize,
    /// First pointwise conv's output `[h2 * w2, ch]`.
    pub pw2: usize,
    /// ...zero-bordered `[h2 + 2, w2 + 2, ch]`, the second depthwise input.
    pub b2: usize,
    /// Second depthwise conv's accumulator `[h3 * w3, ch]`.
    pub b3: usize,
    /// Second pointwise conv's output `[h3 * w3, ch]`.
    pub b4: usize,
    /// `[h3, ch * w3]`, channel-major per time step.
    pub flat: usize,
    pub total_bytes: usize,
}

impl StemScratch {
    pub(crate) fn new(g: &StemGeom, alias: bool) -> Self {
        let f = |elems: usize| align128(elems * 4);
        let mut cur = 0;
        let mut region = |bytes: usize| {
            let off = cur;
            cur += bytes;
            off
        };
        let mel_pad = region(f((g.frames + 2) * (g.bins + 2)));
        let b0 = region(f((g.h1 + 2) * (g.w1 + 2) * g.ch));
        let b1 = region(f(g.h2 * g.w2 * g.ch));
        let tmp = region(f(g.h2 * g.w2 * g.ch).max(f(g.h1 * g.w1 * COL_K)));
        let b2 = region(f((g.h2 + 2) * (g.w2 + 2) * g.ch));
        let b3 = region(f(g.h3 * g.w3 * g.ch));
        let (col, pw2, b4, flat) = if alias {
            (tmp, tmp, tmp, b1)
        } else {
            (
                region(f(g.h1 * g.w1 * COL_K)),
                region(f(g.h2 * g.w2 * g.ch)),
                region(f(g.h3 * g.w3 * g.ch)),
                region(f(g.h3 * g.w3 * g.ch)),
            )
        };
        Self {
            mel_pad,
            col,
            b0,
            b1,
            tmp,
            pw2,
            b2,
            b3,
            b4,
            flat,
            total_bytes: cur,
        }
    }
}

fn zero_border(dst: &mut [f32], hp: usize, wp: usize, ch: usize) {
    let row = wp * ch;
    dst[..row].fill(0.0);
    dst[(hp - 1) * row..hp * row].fill(0.0);
    for r in 1..hp - 1 {
        dst[r * row..r * row + ch].fill(0.0);
        dst[r * row + row - ch..(r + 1) * row].fill(0.0);
    }
}

fn f32_region(st: &mut [u8], off: usize, len: usize) -> &mut [f32] {
    bytemuck::cast_slice_mut(&mut st[off..off + len * 4])
}

/// Fill the host-written parts of the stem buffer `st` (its bytes) for the
/// mel spectrogram `mel` (`[frames, bins]`): the zero-bordered mel, the im2col
/// constants and the zero borders the padded activations rely on. The DSP
/// writes everything else.
pub(crate) fn stage_input(st: &mut [u8], so: &StemScratch, g: &StemGeom, mel: &[f32]) {
    debug_assert_eq!(mel.len(), g.frames * g.bins);
    let wp = g.bins + 2;
    let pad = f32_region(st, so.mel_pad, (g.frames + 2) * wp);
    zero_border(pad, g.frames + 2, wp, 1);
    for (r, row) in mel.chunks_exact(g.bins).enumerate() {
        pad[(r + 1) * wp + 1..(r + 1) * wp + 1 + g.bins].copy_from_slice(row);
    }
    let col = f32_region(st, so.col, g.h1 * g.w1 * COL_K);
    for row in col.as_chunks_mut::<COL_K>().0 {
        row[BIAS_COL] = 1.0;
        row[BIAS_COL + 1..].fill(0.0);
    }
    let b0 = f32_region(st, so.b0, (g.h1 + 2) * (g.w1 + 2) * g.ch);
    zero_border(b0, g.h1 + 2, g.w1 + 2, g.ch);
    let b2 = f32_region(st, so.b2, (g.h2 + 2) * (g.w2 + 2) * g.ch);
    zero_border(b2, g.h2 + 2, g.w2 + 2, g.ch);
}

/// The first convolution: im2col gathers, one matmul that applies the taps and
/// the bias, ReLU over the zero-bordered result.
fn emit_first_conv<S: OpSink>(
    s: &mut S,
    weights: &S::Buf,
    st: &S::Buf,
    wo: &StemWeightOffsets,
    so: &StemScratch,
    g: &StemGeom,
) -> Result<(), CeraError> {
    let ch = g.ch;
    let mel_wp = g.bins + 2;
    for t in 0..TAPS {
        let (i, j) = (t / 3, t % 3);
        // Output (oh, ow) reads padded mel row 2*oh + i, column 2*ow + j.
        let src = View::new(
            st,
            so.mel_pad + (i * mel_wp + j) * 4,
            [g.w1, g.h1, 1],
            [8, 2 * mel_wp * 4, g.h1 * 2 * mel_wp * 4],
        );
        let dst = View::new(
            st,
            so.col + t * 4,
            [g.w1, g.h1, 1],
            [COL_K * 4, g.w1 * COL_K * 4, g.h1 * g.w1 * COL_K * 4],
        );
        dispatch::copy_view(s, src, dst)?;
    }
    let a = View::new(
        weights,
        wo.l0_w,
        [COL_K, ch, 1],
        [4, COL_K * 4, ch * COL_K * 4],
    );
    let x = View::new(
        st,
        so.col,
        [COL_K, g.w1, g.h1],
        [4, COL_K * 4, g.w1 * COL_K * 4],
    );
    // Straight into the interior of the zero-bordered buffer.
    let wp = g.w1 + 2;
    let dst = View::new(
        st,
        so.b0 + (wp + 1) * ch * 4,
        [ch, g.w1, g.h1],
        [4, ch * 4, wp * ch * 4],
    );
    dispatch::matmul_f32(s, a, x, dst)?;
    // The border is zero and ReLU keeps it so: one pass over the whole buffer.
    dispatch::relu(
        s,
        st,
        so.b0,
        TokenShape {
            dim: ch,
            n_tokens: (g.h1 + 2) * wp,
        },
        TokenTile::Whole,
    )
}

/// A 3x3, stride 2 depthwise convolution (+ bias) of the zero-bordered
/// `[h_in + 2, w_in + 2, ch]` buffer at `src`, into `acc` `[h_out * w_out, ch]`.
/// One gather, one per-channel scale and one add per tap: nothing here needs
/// a kernel that understands convolutions.
#[allow(clippy::too_many_arguments)]
fn emit_depthwise<S: OpSink>(
    s: &mut S,
    weights: &S::Buf,
    st: &S::Buf,
    src: usize,
    (taps, bias): (usize, usize),
    (acc, tmp): (usize, usize),
    (h_out, w_out, w_in): (usize, usize, usize),
    ch: usize,
) -> Result<(), CeraError> {
    let wp = w_in + 2;
    let n = h_out * w_out;
    let shape = TokenShape {
        dim: ch,
        n_tokens: n,
    };
    for t in 0..TAPS {
        let (i, j) = (t / 3, t % 3);
        // Output (oh, ow) reads padded row 2*oh + i, column 2*ow + j.
        let from = View::new(
            st,
            src + (i * wp + j) * ch * 4,
            [ch, w_out, h_out],
            [4, 2 * ch * 4, 2 * wp * ch * 4],
        );
        let into = if t == 0 { acc } else { tmp };
        let to = View::new(st, into, [ch, w_out, h_out], [4, ch * 4, w_out * ch * 4]);
        dispatch::copy_view(s, from, to)?;
        dispatch::mul_row_bcast(
            s,
            st,
            into,
            weights,
            taps + t * ch * 4,
            shape,
            TokenTile::Whole,
        )?;
        if t > 0 {
            dispatch::add_residual(s, st, acc, st, tmp, shape, TokenTile::Whole)?;
        }
    }
    dispatch::add_row_bcast(s, st, acc, weights, bias, shape, TokenTile::Whole)
}

/// A 1x1 convolution: `dst = relu(W src + b)` over `m` positions.
#[allow(clippy::too_many_arguments)]
fn emit_pointwise<S: OpSink>(
    s: &mut S,
    weights: &S::Buf,
    st: &S::Buf,
    (w, b): (usize, usize),
    src: usize,
    dst: usize,
    m: usize,
    ch: usize,
) -> Result<(), CeraError> {
    let a = View::new(weights, w, [ch, ch, 1], [4, ch * 4, ch * ch * 4]);
    let x = View::new(st, src, [ch, m, 1], [4, ch * 4, m * ch * 4]);
    let y = View::new(st, dst, [ch, m, 1], [4, ch * 4, m * ch * 4]);
    dispatch::matmul_f32(s, a, x, y)?;
    let shape = TokenShape {
        dim: ch,
        n_tokens: m,
    };
    dispatch::add_row_bcast(s, st, dst, weights, b, shape, TokenTile::Whole)?;
    dispatch::relu(s, st, dst, shape, TokenTile::Whole)
}

/// The stem for the slice `g` describes, ending in `[g.t_out() - skip, n_embd]`
/// at `main[x_off..]`. `flush` is called between stages so a chunk's ops do
/// not pile up in one staging buffer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_stem<S: OpSink>(
    s: &mut S,
    weights: &S::Buf,
    st: &S::Buf,
    main: &S::Buf,
    x_off: usize,
    skip: usize,
    wo: &StemWeightOffsets,
    so: &StemScratch,
    g: &StemGeom,
    flush: &mut dyn FnMut(&mut S) -> Result<(), CeraError>,
) -> Result<(), CeraError> {
    let ch = g.ch;
    emit_first_conv(s, weights, st, wo, so, g)?;
    flush(s)?;

    emit_depthwise(
        s,
        weights,
        st,
        so.b0,
        (wo.dw1_w, wo.dw1_b),
        (so.b1, so.tmp),
        (g.h2, g.w2, g.w1),
        ch,
    )?;
    flush(s)?;

    emit_pointwise(
        s,
        weights,
        st,
        (wo.pw2_w, wo.pw2_b),
        so.b1,
        so.pw2,
        g.h2 * g.w2,
        ch,
    )?;
    let wp2 = g.w2 + 2;
    dispatch::copy_view(
        s,
        View::new(st, so.pw2, [ch, g.w2, g.h2], [4, ch * 4, g.w2 * ch * 4]),
        View::new(
            st,
            so.b2 + (wp2 + 1) * ch * 4,
            [ch, g.w2, g.h2],
            [4, ch * 4, wp2 * ch * 4],
        ),
    )?;
    flush(s)?;

    emit_depthwise(
        s,
        weights,
        st,
        so.b2,
        (wo.dw3_w, wo.dw3_b),
        (so.b3, so.tmp),
        (g.h3, g.w3, g.w2),
        ch,
    )?;
    flush(s)?;

    emit_pointwise(
        s,
        weights,
        st,
        (wo.pw4_w, wo.pw4_b),
        so.b3,
        so.b4,
        g.h3 * g.w3,
        ch,
    )?;
    // The projection's columns are (channel, freq); the activations are
    // (freq, channel), so this copy is the permute.
    dispatch::copy_view(
        s,
        View::new(st, so.b4, [g.w3, ch, g.h3], [ch * 4, 4, g.w3 * ch * 4]),
        View::new(st, so.flat, [g.w3, ch, g.h3], [4, g.w3 * 4, ch * g.w3 * 4]),
    )?;
    flush(s)?;

    // The leading `skip` frames are the chunk's overlap, not output.
    dispatch::linear_m(
        s,
        st,
        so.flat + skip * g.flat_dim() * 4,
        weights,
        wo.out_w,
        wo.out_b,
        main,
        x_off,
        g.h3 - skip,
        LINEAR_TILE,
    )?;
    flush(s)
}

/// Rearrange a channel-last `[h, w, ch]` activation (inside a buffer whose
/// rows are `row_w` positions wide, starting `skip` positions in) into the
/// CPU's channel-major `[ch, h, w]`, to compare layer outputs.
pub(crate) fn to_channel_major(
    src: &[f32],
    (h, w, ch): (usize, usize, usize),
    row_w: usize,
    skip: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; ch * h * w];
    for y in 0..h {
        for x in 0..w {
            let at = (y * row_w + x + skip) * ch;
            for c in 0..ch {
                out[(c * h + y) * w + x] = src[at + c];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::hexagon::dispatch::testing::RecordingSink;

    #[test]
    fn geometry_follows_three_stride_two_convolutions() {
        // 30 s at a 10 ms hop: 3000 mel frames, 128 bins -> 375 frames, 16 wide.
        let g = StemGeom::new(3000, 128, 256).unwrap();
        assert_eq!(
            (g.h1, g.w1, g.h2, g.w2, g.h3, g.w3),
            (1500, 64, 750, 32, 375, 16)
        );
        assert_eq!((g.t_out(), g.flat_dim()), (375, 4096));
        // An odd length rounds up, as the CPU conv does.
        let g = StemGeom::new(1001, 128, 256).unwrap();
        assert_eq!((g.h1, g.h2, g.h3), (501, 251, 126));
        assert!(StemGeom::new(0, 128, 256).is_none());
        // One frame still produces one output frame.
        assert_eq!(StemGeom::new(1, 128, 256).unwrap().t_out(), 1);
    }

    #[test]
    fn scratch_regions_are_disjoint_when_nothing_is_aliased_and_dead_ones_share_otherwise() {
        let g = StemGeom::new(160, 128, 256).unwrap();
        let full = StemScratch::new(&g, false);
        let mut spans = vec![
            (full.mel_pad, (g.frames + 2) * (g.bins + 2) * 4),
            (full.col, g.h1 * g.w1 * COL_K * 4),
            (full.b0, (g.h1 + 2) * (g.w1 + 2) * g.ch * 4),
            (full.b1, g.h2 * g.w2 * g.ch * 4),
            (full.tmp, g.h2 * g.w2 * g.ch * 4),
            (full.pw2, g.h2 * g.w2 * g.ch * 4),
            (full.b2, (g.h2 + 2) * (g.w2 + 2) * g.ch * 4),
            (full.b3, g.h3 * g.w3 * g.ch * 4),
            (full.b4, g.h3 * g.w3 * g.ch * 4),
            (full.flat, g.h3 * g.w3 * g.ch * 4),
        ];
        spans.sort();
        for w in spans.windows(2) {
            assert!(w[0].0 + w[0].1 <= w[1].0, "{w:?} overlap");
        }
        assert!(
            spans
                .iter()
                .all(|&(o, n)| o % 128 == 0 && o + n <= full.total_bytes)
        );
        let packed = StemScratch::new(&g, true);
        assert!(packed.total_bytes < full.total_bytes);
        // The im2col rows, pointwise outputs and flat features fit the shared
        // regions they were given.
        assert_eq!(
            (packed.col, packed.pw2, packed.b4),
            (packed.tmp, packed.tmp, packed.tmp)
        );
        assert_eq!(packed.flat, packed.b1);
    }

    #[test]
    fn border_helper_zeroes_exactly_the_ring() {
        let (hp, wp, ch) = (4, 5, 2);
        let mut v = vec![1.0f32; hp * wp * ch];
        zero_border(&mut v, hp, wp, ch);
        for y in 0..hp {
            for x in 0..wp {
                let ring = y == 0 || y == hp - 1 || x == 0 || x == wp - 1;
                for c in 0..ch {
                    assert_eq!(v[(y * wp + x) * ch + c] == 0.0, ring, "({y},{x})");
                }
            }
        }
    }

    #[test]
    fn channel_major_conversion_skips_the_border() {
        // 2x2 interior of a 4-wide padded row, 3 channels: value = 100*y + 10*x + c.
        let (rows, row_w, ch) = (4, 4, 3);
        let mut src = vec![0.0f32; rows * row_w * ch];
        for y in 0..rows {
            for x in 0..row_w {
                for c in 0..ch {
                    src[(y * row_w + x) * ch + c] = (100 * y + 10 * x + c) as f32;
                }
            }
        }
        let out = to_channel_major(&src[(row_w + 1) * ch..], (2, 2, ch), row_w, 0);
        // Channel 1, row 1, column 1 of the interior is padded (2, 2).
        assert_eq!(out[(2 + 1) * 2 + 1], 221.0);
        assert_eq!(out[0], 110.0);
    }

    fn test_weight_offsets() -> StemWeightOffsets {
        let mut wo_cur = 0;
        let mut v = |n: usize| {
            let o = wo_cur;
            wo_cur += align128(n * 4);
            o
        };
        StemWeightOffsets {
            l0_w: v(256 * COL_K),
            dw1_w: v(9 * 256),
            dw1_b: v(256),
            dw3_w: v(9 * 256),
            dw3_b: v(256),
            pw2_w: v(256 * 256),
            pw2_b: v(256),
            pw4_w: v(256 * 256),
            pw4_b: v(256),
            out_w: HexagonWeightDesc {
                offset: v(1),
                size_bytes: 4096,
                format: crate::backend::hexagon::HexagonWeightFormat::RepackedQ4_0,
                rows: 512,
                cols: 4096,
            },
            out_b: v(512),
        }
    }

    #[test]
    fn stem_emits_every_stage_and_flushes_between_them() {
        let g = StemGeom::new(64, 128, 256).unwrap();
        let so = StemScratch::new(&g, true);
        let wo = test_weight_offsets();
        let mut s = RecordingSink::default();
        let mut flushes = 0;
        emit_stem(&mut s, &"w", &"st", &"x", 0, 0, &wo, &so, &g, &mut |_| {
            flushes += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(flushes, 6, "one flush per stage");
        use crate::backend::hexagon::HtpOpCode::*;
        let ops = s.opcodes();
        let count = |op| ops.iter().filter(|&&o| o == op as u32).count();
        // 9 im2col gathers + 2 * 9 depthwise gathers + the pad copy + the permute.
        assert_eq!(count(Cpy), 9 + 18 + 1 + 1);
        // First conv + both pointwise convs, and the quantised projection.
        assert_eq!(count(MulMat), 3 + 1);
        // Taps scaled, summed and biased in both depthwise convs; biases for the
        // pointwise convs and the projection.
        assert_eq!(count(Mul), 18);
        assert_eq!(count(UnaryRelu), 3);
    }

    /// A chunk after the first drops its overlap frames at the projection: the
    /// activations are read from `skip` rows in and only `h3 - skip` rows are
    /// projected, into the output at `x_off`. The scratch is the largest
    /// chunk's, as in `run_stem_chunk`, so a smaller `g` must not move it.
    #[test]
    fn a_chunk_projects_only_the_rows_past_its_overlap() {
        let big = StemGeom::new(128, 128, 256).unwrap();
        let g = StemGeom::new(64, 128, 256).unwrap();
        let so = StemScratch::new(&big, false);
        let wo = test_weight_offsets();
        let (x_off, skip) = (8192, 1);
        let mut s = RecordingSink::default();
        emit_stem(
            &mut s,
            &"w",
            &"st",
            &"x",
            x_off,
            skip,
            &wo,
            &so,
            &g,
            &mut |_| Ok(()),
        )
        .unwrap();
        let out_ops: Vec<usize> = (0..s.ops.len())
            .filter(|&i| {
                s.ops[i].opcode == crate::backend::hexagon::HtpOpCode::MulMat as u32
                    && s.src(i, 0).offset == wo.out_w.offset
            })
            .collect();
        assert!(!out_ops.is_empty());
        let first = out_ops[0];
        assert_eq!(s.src(first, 1).offset, so.flat + skip * g.flat_dim() * 4);
        assert_eq!(s.dst(first).offset, x_off);
        let rows: u32 = out_ops.iter().map(|&i| s.src(i, 1).ne[1]).sum();
        assert_eq!(rows as usize, g.h3 - skip, "rows projected");
    }

    #[test]
    fn first_conv_rows_carry_taps_then_the_bias_then_zeros() {
        let layer = ConvLayerWeights {
            name: "a.conv1d.0.weight".into(),
            weight: (0..18).map(|v| v as f32).collect(),
            bias: vec![100.0, 200.0],
            shape: vec![3, 3, 1, 2],
        };
        let rows = first_conv_rows(&layer);
        assert_eq!(rows.len(), 2 * COL_K);
        assert_eq!(&rows[..11], &[0., 1., 2., 3., 4., 5., 6., 7., 8., 100., 0.]);
        assert_eq!(
            &rows[COL_K..COL_K + 11],
            &[9., 10., 11., 12., 13., 14., 15., 16., 17., 200., 0.]
        );
        assert!(rows[BIAS_COL + 1..COL_K].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn depthwise_taps_become_tap_major_rows() {
        // Two channels, nine taps: value = 10 * channel + tap.
        let w: Vec<f32> = (0..2)
            .flat_map(|c| (0..9).map(move |t| (10 * c + t) as f32))
            .collect();
        let out = tap_major(&w, 2);
        for t in 0..9 {
            assert_eq!(&out[t * 2..t * 2 + 2], &[t as f32, (10 + t) as f32]);
        }
    }

    #[test]
    fn chunks_tile_the_output_on_eight_row_boundaries() {
        for n_frames in [1usize, 7, 8, 9, 63, 64, 65, 513, 1000, 1001, 3201] {
            for rows in [1usize, 3, 8, 64, 1000] {
                let chunks = plan_chunks(n_frames, 128, rows);
                let t_out = StemGeom::new(n_frames, 128, 1).unwrap().t_out();
                let mut next = 0;
                for (i, c) in chunks.iter().enumerate() {
                    assert_eq!(c.first_row, next, "{n_frames}/{rows}: contiguous");
                    next += c.rows;
                    assert_eq!(c.skip, usize::from(i > 0));
                    assert_eq!(c.mel_start % 8, 0, "{n_frames}/{rows}: slice grid");
                    assert!(c.mel_start + c.mel_rows <= n_frames);
                    let last = i + 1 == chunks.len();
                    if !last {
                        assert_eq!((c.mel_start + c.mel_rows) % 8, 0, "{n_frames}/{rows}");
                    } else {
                        assert_eq!(c.mel_start + c.mel_rows, n_frames);
                    }
                    // The slice produces exactly the kept frames plus the overlap.
                    let local = StemGeom::new(c.mel_rows, 128, 1).unwrap();
                    assert_eq!(local.t_out(), c.rows + c.skip, "{n_frames}/{rows}");
                }
                assert_eq!(next, t_out, "{n_frames}/{rows}: covers every frame");
                assert!(chunks.iter().all(|c| c.rows <= rows));
            }
        }
        assert!(plan_chunks(0, 128, 64).is_empty());
    }

    /// The five convolutions on the CPU, channel-major `[ch, h, w]` out.
    fn cpu_convs(mel: &[f32], frames: usize, bins: usize, ch: usize, w: &[Vec<f32>]) -> Vec<f32> {
        // (depthwise, stride, pad, kernel, relu); w holds weight then bias per layer.
        let modes = [
            (false, 2, 1, 3, true),
            (true, 2, 1, 3, false),
            (false, 1, 0, 1, true),
            (true, 2, 1, 3, false),
            (false, 1, 0, 1, true),
        ];
        let (mut cur, mut c_in, mut h, mut wd) = (mel.to_vec(), 1, frames, bins);
        for (i, &(depthwise, stride, pad, k, relu)) in modes.iter().enumerate() {
            let (nh, nw) = (
                (h + 2 * pad - k) / stride + 1,
                (wd + 2 * pad - k) / stride + 1,
            );
            let mut next = vec![0.0f32; ch * nh * nw];
            let groups = if depthwise { c_in } else { 1 };
            crate::backend::cpu::conv2d(
                &cur,
                &w[2 * i],
                Some(&w[2 * i + 1]),
                &mut next,
                c_in,
                ch,
                h,
                wd,
                k,
                k,
                stride,
                stride,
                pad,
                pad,
                groups,
            );
            if relu {
                crate::backend::cpu::relu_inplace(&mut next);
            }
            (cur, c_in, h, wd) = (next, ch, nh, nw);
        }
        cur
    }

    /// Running the stem slice by slice and dropping each overlap frame gives the
    /// whole clip's result, so the halo arithmetic is right without a device.
    #[test]
    fn chunked_convolutions_equal_the_whole_clip() {
        let (bins, ch) = (32usize, 4usize);
        let lcg = |seed: &mut u64| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Non-negative, so no ReLU can wipe out the input and hide an
            // edge error behind a constant.
            (*seed >> 40) as f32 / (1u64 << 24) as f32
        };
        let mut seed = 7u64;
        // Weight lengths per layer: 3x3 dense from one channel, 3x3 depthwise,
        // 1x1 pointwise, 3x3 depthwise, 1x1 pointwise.
        let mut w: Vec<Vec<f32>> = Vec::new();
        for len in [ch * 9, ch * 9, ch * ch, ch * 9, ch * ch] {
            w.push((0..len).map(|_| lcg(&mut seed)).collect());
            w.push((0..ch).map(|_| lcg(&mut seed)).collect());
        }
        for frames in [40usize, 64, 201, 520] {
            let mel: Vec<f32> = (0..frames * bins).map(|_| lcg(&mut seed)).collect();
            let whole = cpu_convs(&mel, frames, bins, ch, &w);
            let distinct: std::collections::HashSet<u32> =
                whole.iter().map(|v| v.to_bits()).collect();
            assert!(
                distinct.len() > whole.len() / 2,
                "the reference must carry the input"
            );
            let t_out = StemGeom::new(frames, bins, ch).unwrap().t_out();
            let w3 = whole.len() / ch / t_out;
            for rows in [1usize, 2, 5, 8] {
                for c in plan_chunks(frames, bins, rows) {
                    let slice = &mel[c.mel_start * bins..(c.mel_start + c.mel_rows) * bins];
                    let part = cpu_convs(slice, c.mel_rows, bins, ch, &w);
                    let local_rows = c.rows + c.skip;
                    for chn in 0..ch {
                        for r in 0..c.rows {
                            for x in 0..w3 {
                                let got = part[(chn * local_rows + c.skip + r) * w3 + x];
                                let want = whole[(chn * t_out + c.first_row + r) * w3 + x];
                                assert!(
                                    got == want,
                                    "frames {frames} rows {rows} chunk {} ch {chn} row {r}: {got} vs {want}",
                                    c.first_row
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The point of chunking: the stem's buffer is set by the chunk, not the clip.
    #[test]
    fn the_buffer_is_bounded_by_the_chunk_not_the_clip() {
        let bytes = |n_frames: usize| {
            let largest = plan_chunks(n_frames, 128, STEM_CHUNK_ROWS)
                .iter()
                .map(|c| c.mel_rows)
                .max()
                .unwrap();
            StemScratch::new(&StemGeom::new(largest, 128, 256).unwrap(), true).total_bytes
        };
        // 30 s and 32 s of audio need the same buffer, and it is a fraction of
        // the 197 MB a whole 32 s clip took.
        assert_eq!(bytes(3000), bytes(3200));
        let whole = StemScratch::new(&StemGeom::new(3200, 128, 256).unwrap(), true).total_bytes;
        assert!(bytes(3200) < 40 << 20, "{} MiB", bytes(3200) >> 20);
        assert!(
            whole > 5 * bytes(3200),
            "{} vs {} MiB",
            whole >> 20,
            bytes(3200) >> 20
        );
    }
}
