//! Nemotron-3-Diarization on the Hexagon NPU: the embedder, the Transformer encoder and
//! the speaker head.
//!
//! The network maps onto the DSP almost directly: the 31 encoder blocks are pre-norm
//! `x += Attn(LN(x)); x += FFN(LN(x))` with RoPE positions restarting at 0 every step, a
//! tanh GELU feed-forward (what `dispatch::gelu_tanh` computes; the Phase 3 spike showed
//! the erf/tanh swap costs no speaker decision on the clip), and a key mask over the pad
//! groups. The mask is folded into the scores with an in-place row-broadcast add of a
//! host-filled bias row (0 for valid keys, `-1e30` past them) between the QK matmul and the
//! softmax, so the softmax itself runs unmasked like the Sortformer tail's. The head dim is
//! 64, a multiple of the 32-float vector the F32 matmul reads, so Q and K need no lane
//! padding.
//!
//! The subpixel upsampler (`Conv1d(k=3, pad 1)`, lowered to a matmul on the CPU too) runs as
//! one matmul over per-frame unrolled rows: `proj`'s output is copied into a zero-edge-padded
//! `[t+2, tf_d]` buffer (the edges from a staged zero row), one strided copy unrolls every
//! frame's three taps, and one `linear_m` produces the `[t, 8 * tf_d]` conv output, which the
//! head reads as `[t*8, tf_d]` without a copy. The log-mel front end is
//! [`HexagonSortformerMel`] with this model's own window and filterbank (same sample rate,
//! hop and log floor, shared loader).
//!
//! Staging covers steps of up to `max_frames` encoder frames; longer windows are declined
//! (the CPU takes them) and `t * 8` sub-frames past the valid groups are zeroed on the host
//! after readback, like NeMo's masked output.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use crate::backend::cpu::RopeType;
use crate::backend::hexagon::dispatch::{self, LayerNormArgs, OpSink, TokenShape, TokenTile, View};
use crate::backend::hexagon::{
    FastRpcDriver, HTP_TENSOR_COMPUTE, HexagonDevice, HexagonWeightDesc, HtpDataType, HtpOpCode,
    LockOrRecover, RpcmemBuffer, align128, build_rope_kernel_params, build_rope_params,
};
use crate::model::audio_encoder_hexagon::{
    pad32, plan_linear, plan_vec, put_linear, put_vec, run_on_queue, settled,
};
use crate::model::nemotron3_diarization::{
    Nemotron3Accelerator, Nemotron3Model, Nemotron3Weights, SUBSAMPLING, enc_frames,
};
use crate::model::sortformer_hexagon::HexagonSortformerMel;
use crate::model::weights::MmapWeight;
use crate::session::CeraError;

/// Token-axis ops run in tiles of 64 frames, as in the Sortformer tail (the quantized matmul
/// and the elementwise kernels overflow the VTCM on a whole sequence).
const TILE: TokenTile = TokenTile::Tiles(64);

/// The additive key bias past the valid groups: finite, so the DSP never sees an infinity
/// (far below any real score, like Whisper's `GREEDY_SUPPRESSED`).
const KEY_BIAS_SUPPRESSED: f32 = -1e30;

/// Largest DSP-side staging (weights plus scratch) attempted, in bytes. Scratch is
/// quadratic in `max_frames` (three heads·t² score buffers plus linear terms), so an
/// element cap alone admits gigabyte transients: the real 528–608-frame windows stage
/// about 160 MB, and this leaves them headroom while a validate-legal 7500-frame window
/// (about 5.4 GB of scratch alone) is refused before the rpcmem attempt.
const MAX_NPU_STAGING_BYTES: usize = 512 * 1024 * 1024;

/// Refuse a window whose quadratic scratch term alone (three heads·t² buffers of 4-byte
/// floats) exceeds [`MAX_NPU_STAGING_BYTES`]. Runs before the planners' plain arithmetic
/// (u128, so it cannot wrap itself): a hostile window must fail here, not wrap the layout
/// math. Callers then budget the true weights-plus-scratch total after planning.
fn check_staging_window(max_frames: usize, heads: usize) -> Result<(), CeraError> {
    let quad = max_frames as u128 * max_frames as u128 * heads.max(1) as u128 * 12;
    if quad > MAX_NPU_STAGING_BYTES as u128 {
        return Err(CeraError::Backend(format!(
            "nemotron3: {max_frames} frames would stage past the {MAX_NPU_STAGING_BYTES}-byte \
             NPU budget; use a smaller window or the CPU"
        )));
    }
    Ok(())
}

/// A linear layer's weight and optional bias in the weights buffer (the QKV and embedder
/// linears have no bias; the rest do).
#[derive(Debug, Clone, Copy)]
struct LinearOffsets {
    w: HexagonWeightDesc,
    b: Option<usize>,
}

/// One encoder block's weights in the weights buffer.
#[derive(Debug, Clone, Copy)]
struct LayerOffsets {
    q: LinearOffsets,
    k: LinearOffsets,
    v: LinearOffsets,
    o: LinearOffsets,
    ln1_w: usize,
    ln1_b: usize,
    up: LinearOffsets,
    down: LinearOffsets,
    ln2_w: usize,
    ln2_b: usize,
}

#[derive(Debug, Clone)]
struct PredictOffsets {
    input_norm_w: usize,
    input_norm_b: usize,
    layers: Vec<LayerOffsets>,
    final_norm_w: usize,
    final_norm_b: usize,
    proj: LinearOffsets,
    upsample: LinearOffsets,
    dense: LinearOffsets,
    out: LinearOffsets,
    /// `arange(max_frames)` as i32: RoPE positions restart at 0 every step, so the position
    /// tensor is staged once, not uploaded per call.
    rope_pos: usize,
    /// One zero row of `tf_d` floats: the upsampler's edge padding.
    zero_row: usize,
    total_bytes: usize,
}

/// The network's widths, read from the loaded weights.
#[derive(Debug, Clone, Copy)]
struct Dims {
    /// Encoder width.
    d: usize,
    heads: usize,
    /// Per-head width (`d / heads`).
    dh: usize,
    /// Feed-forward inner width.
    inner: usize,
    /// Head width after `proj`.
    tf_d: usize,
    n_spk: usize,
    eps: f32,
    rope_theta: f32,
}

fn linear(
    cur: &mut usize,
    w: &MmapWeight,
    b: Option<&[f32]>,
    what: &str,
) -> Result<LinearOffsets, CeraError> {
    if let Some(b) = b
        && b.len() != w.rows
    {
        return Err(CeraError::Backend(format!(
            "nemotron3: {what} has {} biases for {} rows",
            b.len(),
            w.rows
        )));
    }
    let w = plan_linear(cur, w).map_err(|e| {
        CeraError::Backend(format!(
            "nemotron3: {what}: {e}; the NPU reads Q8_0 or Q4_0 weights, so convert the \
             model with `--tail-outtype q8_0` (scripts/nemotron3_diarization/README.md)"
        ))
    })?;
    let b = b.map(|b| plan_vec(cur, b.len()));
    Ok(LinearOffsets { w, b })
}

fn check_norm(
    cur: &mut usize,
    w: &[f32],
    b: &[f32],
    d: usize,
    what: &str,
) -> Result<(usize, usize), CeraError> {
    if w.len() != d || b.len() != d {
        return Err(CeraError::Backend(format!(
            "nemotron3: {what} has {}/{} values for width {d}",
            w.len(),
            b.len()
        )));
    }
    Ok((plan_vec(cur, d), plan_vec(cur, d)))
}

fn check_mat(m: &MmapWeight, rows: usize, cols: usize, what: &str) -> Result<(), CeraError> {
    if m.rows != rows || m.cols != cols {
        return Err(CeraError::Backend(format!(
            "nemotron3: {what} is {}x{}, expected {rows}x{cols}",
            m.rows, m.cols
        )));
    }
    Ok(())
}

impl PredictOffsets {
    fn plan(w: &Nemotron3Weights, dims: &Dims, max_frames: usize) -> Result<Self, CeraError> {
        let mut cur = 0;
        let (input_norm_w, input_norm_b) = check_norm(
            &mut cur,
            &w.input_norm_w,
            &w.input_norm_b,
            dims.d,
            "input norm",
        )?;
        let mut layers = Vec::with_capacity(w.layers.len());
        for (i, l) in w.layers.iter().enumerate() {
            let what = |n: &str| format!("encoder layer {i} {n}");
            check_mat(&l.q_w, dims.d, dims.d, &what("q"))?;
            check_mat(&l.k_w, dims.d, dims.d, &what("k"))?;
            check_mat(&l.v_w, dims.d, dims.d, &what("v"))?;
            check_mat(&l.o_w, dims.d, dims.d, &what("o"))?;
            if l.o_b.len() != dims.d {
                return Err(CeraError::Backend(format!(
                    "nemotron3: {} has {} biases for width {}",
                    what("o bias"),
                    l.o_b.len(),
                    dims.d
                )));
            }
            check_mat(&l.up_w, dims.inner, dims.d, &what("up"))?;
            check_mat(&l.down_w, dims.d, dims.inner, &what("down"))?;
            if l.up_b.len() != dims.inner || l.down_b.len() != dims.d {
                return Err(CeraError::Backend(format!(
                    "nemotron3: {} has {}/{} biases for {}/{}",
                    what("ffn"),
                    l.up_b.len(),
                    l.down_b.len(),
                    dims.inner,
                    dims.d
                )));
            }
            let (ln1_w, ln1_b) = check_norm(&mut cur, &l.ln1_w, &l.ln1_b, dims.d, &what("ln1"))?;
            let (ln2_w, ln2_b) = check_norm(&mut cur, &l.ln2_w, &l.ln2_b, dims.d, &what("ln2"))?;
            layers.push(LayerOffsets {
                q: linear(&mut cur, &l.q_w, None, &what("q"))?,
                k: linear(&mut cur, &l.k_w, None, &what("k"))?,
                v: linear(&mut cur, &l.v_w, None, &what("v"))?,
                o: linear(&mut cur, &l.o_w, Some(&l.o_b), &what("o"))?,
                ln1_w,
                ln1_b,
                up: linear(&mut cur, &l.up_w, Some(&l.up_b), &what("up"))?,
                down: linear(&mut cur, &l.down_w, Some(&l.down_b), &what("down"))?,
                ln2_w,
                ln2_b,
            });
        }
        let (final_norm_w, final_norm_b) = check_norm(
            &mut cur,
            &w.final_norm_w,
            &w.final_norm_b,
            dims.d,
            "final norm",
        )?;
        check_mat(&w.proj_w, dims.tf_d, dims.d, "proj")?;
        if w.proj_b.len() != dims.tf_d {
            return Err(CeraError::Backend(format!(
                "nemotron3: proj has {} biases for width {}",
                w.proj_b.len(),
                dims.tf_d
            )));
        }
        // The upsample taps: `up_w` is `[tf_d * 8, tf_d * taps]`, 3 taps on the checkpoint.
        // The unroll below hard-codes the tap count, so a checkpoint change fails here.
        let taps = w.up_w.cols / dims.tf_d.max(1);
        if w.up_w.rows != dims.tf_d * SUBSAMPLING || w.up_w.cols != dims.tf_d * taps || taps != 3 {
            return Err(CeraError::Backend(format!(
                "nemotron3: upsample is {}x{}, expected {}x{} (3 taps)",
                w.up_w.rows,
                w.up_w.cols,
                dims.tf_d * SUBSAMPLING,
                dims.tf_d * 3
            )));
        }
        if w.up_b.len() != dims.tf_d * SUBSAMPLING {
            return Err(CeraError::Backend(format!(
                "nemotron3: upsample has {} biases for {}",
                w.up_b.len(),
                dims.tf_d * SUBSAMPLING
            )));
        }
        check_mat(&w.dense_w, dims.tf_d, dims.tf_d, "dense")?;
        if w.dense_b.len() != dims.tf_d {
            return Err(CeraError::Backend(format!(
                "nemotron3: dense has {} biases for width {}",
                w.dense_b.len(),
                dims.tf_d
            )));
        }
        check_mat(&w.out_w, dims.n_spk, dims.tf_d, "out")?;
        if w.out_b.len() != dims.n_spk {
            return Err(CeraError::Backend(format!(
                "nemotron3: out has {} biases for {} speakers",
                w.out_b.len(),
                dims.n_spk
            )));
        }
        let proj = linear(&mut cur, &w.proj_w, Some(&w.proj_b), "proj")?;
        let upsample = linear(&mut cur, &w.up_w, Some(&w.up_b), "upsample")?;
        let dense = linear(&mut cur, &w.dense_w, Some(&w.dense_b), "dense")?;
        let out = linear(&mut cur, &w.out_w, Some(&w.out_b), "out")?;
        let rope_pos = plan_vec(&mut cur, max_frames);
        let zero_row = plan_vec(&mut cur, dims.tf_d);
        Ok(Self {
            input_norm_w,
            input_norm_b,
            layers,
            final_norm_w,
            final_norm_b,
            proj,
            upsample,
            dense,
            out,
            rope_pos,
            zero_row,
            total_bytes: cur,
        })
    }
}

/// Activation scratch for sequences of up to `max_frames` encoder frames.
#[derive(Debug, Clone, Copy)]
struct Scratch {
    /// The embedder output, `[t, d]`: the host's input.
    xin: usize,
    /// The residual stream, `[t, d]` (pre-norm needs no second buffer).
    x: usize,
    /// A normed stream, `[t, d]`.
    norm: usize,
    q: usize,
    k: usize,
    v: usize,
    /// The additive key bias, `[max_frames]`: 0 for valid keys, `KEY_BIAS_SUPPRESSED` past
    /// them. Host-filled each call.
    keybias: usize,
    /// Scores and their softmax, `[t, t, heads]`.
    ac: usize,
    smc: usize,
    /// The softmax in the zero-padded layout the last contraction reads, `[t, tp, heads]`.
    sm: usize,
    /// V transposed per head, `[t, dh, heads]`. Host-zeroed each call.
    vt: usize,
    /// attn @ V, head-interleaved `[t, d]`.
    av: usize,
    /// A sub-block's output before its residual add, `[t, d]`.
    out: usize,
    /// Feed-forward inner activation, `[t, inner]`.
    ff: usize,
    /// GELU working rows.
    gelu_tmp: usize,
    /// `proj` output, `[t, tf_d]`.
    h: usize,
    /// `h` with a zero row above and below, `[t+2, tf_d]`.
    hpadded: usize,
    /// Unrolled upsample rows, `[t, 3 * tf_d]`.
    unrolled: usize,
    /// Upsampled conv output, `[t, 8 * tf_d]`, read as `[t*8, tf_d]` without a copy.
    conv: usize,
    /// Head hidden activation, `[t*8, tf_d]`.
    hid: usize,
    /// Speaker logits/activities, `[t*8, n_spk]`.
    pred: usize,
    total_bytes: usize,
}

impl Scratch {
    fn new(dims: &Dims, max_frames: usize) -> Self {
        let t = max_frames;
        let tp = pad32(t);
        let mut cur = 0;
        let mut region = |bytes: usize| {
            let off = cur;
            cur += align128(bytes);
            off
        };
        let seq = |w: usize| t * w * 4;
        let xin = region(seq(dims.d));
        let x = region(seq(dims.d));
        let norm = region(seq(dims.d));
        let q = region(seq(dims.d));
        let k = region(seq(dims.d));
        let v = region(seq(dims.d));
        let keybias = region(t * 4);
        let ac = region(dims.heads * t * t * 4);
        let smc = region(dims.heads * t * t * 4);
        let sm = region(dims.heads * t * tp * 4);
        let vt = region(dims.heads * dims.dh * tp * 4);
        let av = region(seq(dims.d));
        let out = region(seq(dims.d));
        let ff = region(seq(dims.inner));
        let gelu_tmp = region(dispatch::GELU_TMP_ROWS * dims.inner * 4);
        let h = region(seq(dims.tf_d));
        let hpadded = region((t + 2) * dims.tf_d * 4);
        let unrolled = region(t * 3 * dims.tf_d * 4);
        let conv = region(t * SUBSAMPLING * dims.tf_d * 4);
        let hid = region(t * SUBSAMPLING * dims.tf_d * 4);
        let pred = region(t * SUBSAMPLING * dims.n_spk * 4);
        Self {
            xin,
            x,
            norm,
            q,
            k,
            v,
            keybias,
            ac,
            smc,
            sm,
            vt,
            av,
            out,
            ff,
            gelu_tmp,
            h,
            hpadded,
            unrolled,
            conv,
            hid,
            pred,
            total_bytes: cur,
        }
    }
}

/// How much of the network a call runs. The stage names match the CPU
/// [`predict_with_taps`](Nemotron3Model::predict_with_taps) taps, so the device probe compares
/// stage by stage with the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictStage {
    /// The input norm: `[t, d]`.
    InputNorm,
    /// The input norm and the first `n` encoder blocks: `[t, d]`. Clamped to the staged
    /// blocks, so an overlarge `n` runs all of them.
    Layers(usize),
    /// Through the final norm: `[t, d]`.
    FinalNorm,
    /// Through `proj`: `[t, tf_d]`.
    Proj,
    /// Through the subpixel upsampler: `[t*8, tf_d]`.
    Upsampled,
    /// Through the speaker head, pre-sigmoid: `[t*8, n_spk]`.
    Logits,
    /// Everything: the sigmoid speaker activities, `[t*8, n_spk]`.
    Full,
}

/// RoPE over `[n_tokens, head_dim * n_heads]` activations at `act_offset`, in place, with the
/// positions at `pos_offset` (an i32 row). A generic-OpSink port of
/// `hexagon_lfm2::ops::dispatch_rope_m` (concrete on the queue session there): the activation
/// tensor is `[head_dim, n_heads, n_tokens]`, which is the `[n_tokens, d]` row-major layout
/// the QKV linears write, and `mode` is NeoX.
#[allow(clippy::too_many_arguments)]
fn emit_rope<S: OpSink>(
    s: &mut S,
    act: &S::Buf,
    act_offset: usize,
    pos: &S::Buf,
    pos_offset: usize,
    head_dim: usize,
    n_heads: usize,
    n_tokens: usize,
    max_seq_len: usize,
    theta: f32,
) -> Result<(), CeraError> {
    let q_dim = head_dim * n_heads;
    let total_bytes = q_dim * n_tokens * 4;
    let act_ti = s
        .add_tensor(
            act,
            act_offset,
            total_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, n_heads as u32, n_tokens as u32, 1],
            [
                4,
                (head_dim * 4) as u32,
                (q_dim * 4) as u32,
                total_bytes as u32,
            ],
        )
        .map_err(|e| CeraError::Backend(format!("nemotron3 rope: {e}")))?;
    let pos_ti = s
        .add_tensor(
            pos,
            pos_offset,
            n_tokens * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::I32 as u32,
            [n_tokens as u32, 1, 1, 1],
            [
                4,
                (n_tokens * 4) as u32,
                (n_tokens * 4) as u32,
                (n_tokens * 4) as u32,
            ],
        )
        .map_err(|e| CeraError::Backend(format!("nemotron3 rope: {e}")))?;
    let params = build_rope_params(
        head_dim,
        RopeType::Neox.htp_mode(),
        max_seq_len as u32,
        theta,
        1.0,
    );
    let nrows = n_heads * n_tokens;
    let n_threads = s.dsp_threads().min(nrows as u32).max(1);
    let kparams = build_rope_kernel_params(head_dim, nrows, n_heads, n_tokens, n_threads);
    s.enqueue_op(
        HtpOpCode::Rope as u32,
        &[act_ti, pos_ti],
        &[act_ti],
        params,
        kparams,
    )
    .map_err(|e| CeraError::Backend(format!("nemotron3 rope: {e}")))?;
    s.end_group()
        .map_err(|e| CeraError::Backend(format!("nemotron3 rope: {e}")))
}

/// One pre-norm encoder block over the `[t, d]` stream at `so.x`: norm, QKV, RoPE, masked
/// attention, output proj and residual, then norm, tanh-GELU feed-forward and residual. The
/// result is back in `so.x`.
#[allow(clippy::too_many_arguments)]
fn emit_layer<S: OpSink>(
    s: &mut S,
    weights: &S::Buf,
    scratch: &S::Buf,
    l: &LayerOffsets,
    so: &Scratch,
    dims: &Dims,
    t: usize,
    rope_pos: usize,
    max_frames: usize,
) -> Result<(), CeraError> {
    let (d, h, dh) = (dims.d, dims.heads, dims.dh);
    let tp = pad32(t);
    let shape = TokenShape {
        dim: d,
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
            w_offset: l.ln1_w,
            b_offset: l.ln1_b,
            shape,
            eps: dims.eps,
            tile: TILE,
        },
    )?;
    for (lin, dst) in [(&l.q, so.q), (&l.k, so.k), (&l.v, so.v)] {
        dispatch::linear_m(
            s, scratch, so.norm, weights, lin.w, lin.b, scratch, dst, t, TILE,
        )?;
    }
    for act in [so.q, so.k] {
        emit_rope(
            s,
            scratch,
            act,
            weights,
            rope_pos,
            dh,
            h,
            t,
            max_frames,
            dims.rope_theta,
        )?;
    }
    // Each head's `[dh, rows]` slice of a `[rows, d]` activation (dh is a multiple of 32,
    // so no lane padding).
    let heads = |off, rows: usize| View::new(scratch, off, [dh, rows, h], [4, d * 4, dh * 4]);
    // Scores ac[k, q, h] = K[k,h] . Q[q,h], softmax over keys.
    let ac = View::new(scratch, so.ac, [t, t, h], [4, t * 4, t * t * 4]);
    dispatch::matmul_f32(s, heads(so.k, t), heads(so.q, t), ac)?;
    // The pad-key mask, folded into the scores: every one of the `t * h` key rows reads
    // `bias[key]`, 0 for valid keys and suppressing past them.
    dispatch::add_row_bcast(
        s,
        scratch,
        so.ac,
        scratch,
        so.keybias,
        TokenShape {
            dim: t,
            n_tokens: t * h,
        },
        TILE,
    )?;
    let smc = View::new(scratch, so.smc, [t, t, h], [4, t * 4, t * t * 4]);
    dispatch::softmax(s, ac, None, smc, 1.0 / (dh as f32).sqrt())?;
    // Into the zero-padded layout the last contraction reads.
    let sm = View::new(scratch, so.sm, [t, t, h], [4, tp * 4, t * tp * 4]);
    dispatch::copy_view(s, smc, sm)?;
    // V transposed per head so the contraction (over keys) is contiguous.
    let v_src = View::new(scratch, so.v, [t, dh, h], [d * 4, 4, dh * 4]);
    let vt = View::new(scratch, so.vt, [t, dh, h], [4, tp * 4, dh * tp * 4]);
    dispatch::copy_view(s, v_src, vt)?;
    let vt_full = View::new(scratch, so.vt, [tp, dh, h], [4, tp * 4, dh * tp * 4]);
    let sm_full = View::new(scratch, so.sm, [tp, t, h], [4, tp * 4, t * tp * 4]);
    dispatch::matmul_f32(s, vt_full, sm_full, heads(so.av, t))?;
    dispatch::linear_m(
        s, scratch, so.av, weights, l.o.w, l.o.b, scratch, so.out, t, TILE,
    )?;
    dispatch::add_residual(s, scratch, so.x, scratch, so.out, shape, TILE)?;
    // Feed-forward on the normalised stream in `so.norm`.
    dispatch::layer_norm(
        s,
        LayerNormArgs {
            src: scratch,
            src_offset: so.x,
            dst: scratch,
            dst_offset: so.norm,
            weights,
            w_offset: l.ln2_w,
            b_offset: l.ln2_b,
            shape,
            eps: dims.eps,
            tile: TILE,
        },
    )?;
    dispatch::linear_m(
        s, scratch, so.norm, weights, l.up.w, l.up.b, scratch, so.ff, t, TILE,
    )?;
    dispatch::gelu_tanh(
        s,
        scratch,
        so.ff,
        scratch,
        so.gelu_tmp,
        dispatch::GELU_TMP_ROWS,
        TokenShape {
            dim: dims.inner,
            n_tokens: t,
        },
    )?;
    dispatch::linear_m(
        s, scratch, so.ff, weights, l.down.w, l.down.b, scratch, so.out, t, TILE,
    )?;
    dispatch::add_residual(s, scratch, so.x, scratch, so.out, shape, TILE)?;
    Ok(())
}

/// Emit `stage` over `t` embedder output frames staged at `so.xin`. Returns the scratch
/// offset and float count of the result.
#[allow(clippy::too_many_arguments)]
fn emit_predict<S: OpSink>(
    s: &mut S,
    weights: &S::Buf,
    scratch: &S::Buf,
    o: &PredictOffsets,
    so: &Scratch,
    dims: &Dims,
    t: usize,
    max_frames: usize,
    stage: PredictStage,
    flush: &mut dyn FnMut(&mut S) -> Result<(), CeraError>,
) -> Result<(usize, usize), CeraError> {
    let shape = TokenShape {
        dim: dims.d,
        n_tokens: t,
    };
    dispatch::layer_norm(
        s,
        LayerNormArgs {
            src: scratch,
            src_offset: so.xin,
            dst: scratch,
            dst_offset: so.x,
            weights,
            w_offset: o.input_norm_w,
            b_offset: o.input_norm_b,
            shape,
            eps: dims.eps,
            tile: TILE,
        },
    )?;
    if stage == PredictStage::InputNorm {
        return Ok((so.x, t * dims.d));
    }
    let n_layers = match stage {
        PredictStage::InputNorm => 0,
        PredictStage::Layers(n) => n.min(o.layers.len()),
        PredictStage::FinalNorm
        | PredictStage::Proj
        | PredictStage::Upsampled
        | PredictStage::Logits
        | PredictStage::Full => o.layers.len(),
    };
    for l in &o.layers[..n_layers] {
        emit_layer(s, weights, scratch, l, so, dims, t, o.rope_pos, max_frames)?;
        flush(s)?;
    }
    if matches!(stage, PredictStage::Layers(_)) {
        return Ok((so.x, t * dims.d));
    }
    dispatch::layer_norm(
        s,
        LayerNormArgs {
            src: scratch,
            src_offset: so.x,
            dst: scratch,
            dst_offset: so.norm,
            weights,
            w_offset: o.final_norm_w,
            b_offset: o.final_norm_b,
            shape,
            eps: dims.eps,
            tile: TILE,
        },
    )?;
    if stage == PredictStage::FinalNorm {
        return Ok((so.norm, t * dims.d));
    }
    dispatch::linear_m(
        s, scratch, so.norm, weights, o.proj.w, o.proj.b, scratch, so.h, t, TILE,
    )?;
    if stage == PredictStage::Proj {
        return Ok((so.h, t * dims.tf_d));
    }
    // The subpixel upsample: `h` into the zero-edge-padded buffer, per-frame unrolled rows,
    // one matmul. The edges come from the staged zero row.
    let tf = dims.tf_d;
    let h_row = |off| View::new(scratch, off, [tf, t, 1], [4, tf * 4, t * tf * 4]);
    dispatch::copy_view(s, h_row(so.h), h_row(so.hpadded + tf * 4))?;
    let zero = View::new(weights, o.zero_row, [tf, 1, 1], [4, tf * 4, tf * 4]);
    dispatch::copy_view(
        s,
        zero,
        View::new(scratch, so.hpadded, [tf, 1, 1], [4, tf * 4, tf * 4]),
    )?;
    dispatch::copy_view(
        s,
        zero,
        View::new(
            scratch,
            so.hpadded + (t + 1) * tf * 4,
            [tf, 1, 1],
            [4, tf * 4, tf * 4],
        ),
    )?;
    // One strided copy unrolls every frame: source element `(tap d, channel ch, frame
    // f)` steps a padded row per tap, a float per channel, a padded row per frame, and the
    // destination packs it as `unrolled[f][ch * 3 + d]`, matching the old per-frame gathers
    // element for element (the first axis varies fastest on both sides).
    let src = View::new(scratch, so.hpadded, [3, tf, t], [tf * 4, 4, tf * 4]);
    let dst = View::new(scratch, so.unrolled, [3, tf, t], [4, 12, 3 * tf * 4]);
    dispatch::copy_view(s, src, dst)?;
    dispatch::linear_m(
        s,
        scratch,
        so.unrolled,
        weights,
        o.upsample.w,
        o.upsample.b,
        scratch,
        so.conv,
        t,
        TILE,
    )?;
    if stage == PredictStage::Upsampled {
        return Ok((so.conv, t * SUBSAMPLING * dims.tf_d));
    }
    // The speaker head over the conv output read as `[t*8, tf_d]`: relu, dense, relu, out,
    // and the sigmoid on `Full` (the `Logits` tap reads pre-sigmoid).
    let n10 = t * SUBSAMPLING;
    let tf_shape = TokenShape {
        dim: dims.tf_d,
        n_tokens: n10,
    };
    dispatch::relu(s, scratch, so.conv, tf_shape, TILE)?;
    dispatch::linear_m(
        s, scratch, so.conv, weights, o.dense.w, o.dense.b, scratch, so.hid, n10, TILE,
    )?;
    dispatch::relu(s, scratch, so.hid, tf_shape, TILE)?;
    dispatch::linear_m(
        s, scratch, so.hid, weights, o.out.w, o.out.b, scratch, so.pred, n10, TILE,
    )?;
    if stage == PredictStage::Logits {
        return Ok((so.pred, n10 * dims.n_spk));
    }
    dispatch::sigmoid(
        s,
        scratch,
        so.pred,
        TokenShape {
            dim: dims.n_spk,
            n_tokens: n10,
        },
        TILE,
    )?;
    Ok((so.pred, n10 * dims.n_spk))
}

/// The input norm, encoder blocks, final norm, `proj`, subpixel upsampler and speaker head
/// on the Hexagon NPU.
///
/// The `RpcmemBuffer`s retain the driver, so the struct keeps no `Arc` of its own.
pub struct HexagonNemotron3Predict {
    device: Arc<Mutex<HexagonDevice>>,
    weights_buf: RpcmemBuffer,
    offsets: PredictOffsets,
    scratch: Mutex<RpcmemBuffer>,
    so: Scratch,
    dims: Dims,
    max_frames: usize,
    /// Largest `t` any `run` has staged: only a shrink below it can expose stale sm/vt
    /// lanes (see `run`). Monotonic, so a stale read only ever re-zeroes.
    staged_hi: AtomicUsize,
}

// SAFETY: once shared, the rpcmem buffers are only touched while holding `device` and
// `scratch` (construction writes them under exclusive ownership), as in the encoder.
unsafe impl Send for HexagonNemotron3Predict {}
unsafe impl Sync for HexagonNemotron3Predict {}

impl HexagonNemotron3Predict {
    /// Stage the model's prediction weights on the device for sequences of up to `max_frames`
    /// encoder frames. Needs Q8_0 or Q4_0 weights (the NPU's matmul formats): an F32 or F16
    /// GGUF is refused.
    pub fn new(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        model: &Nemotron3Model,
        max_frames: usize,
    ) -> Result<Self, CeraError> {
        let w = model.weights();
        let c = &w.config;
        if c.n_head == 0 || !c.n_embd.is_multiple_of(c.n_head) {
            return Err(CeraError::Backend(format!(
                "nemotron3: width {} with {} heads is not supported",
                c.n_embd, c.n_head
            )));
        }
        let dh = c.n_embd / c.n_head;
        // The score matmul reads 32 lanes at a time; dh must be a whole number of them (no
        // lane-padding path, unlike the Sortformer tail's dh=24).
        if !c.n_embd.is_multiple_of(32) || !dh.is_multiple_of(32) {
            return Err(CeraError::Backend(format!(
                "nemotron3: width {} head {dh} is not a multiple of 32 lanes",
                c.n_embd
            )));
        }
        let dims = Dims {
            d: c.n_embd,
            heads: c.n_head,
            dh,
            inner: c.n_ff,
            tf_d: c.tf_d,
            n_spk: c.n_spk,
            eps: c.eps,
            rope_theta: c.rope_theta,
        };
        if max_frames == 0 {
            return Err(CeraError::Backend(
                "nemotron3: max_frames must be at least 1".into(),
            ));
        }
        check_staging_window(max_frames, dims.heads)?;
        let offsets = PredictOffsets::plan(w, &dims, max_frames)?;
        let so = Scratch::new(&dims, max_frames);
        // The pre-gate above bounded the quadratic term; budget the true weights-plus-scratch
        // total with checked math before either rpcmem attempt.
        match offsets.total_bytes.checked_add(so.total_bytes) {
            Some(staged) if staged <= MAX_NPU_STAGING_BYTES => {}
            _ => {
                return Err(CeraError::Backend(format!(
                    "nemotron3: staging {max_frames} frames needs past the \
                     {MAX_NPU_STAGING_BYTES}-byte NPU budget; use a smaller window or the CPU"
                )));
            }
        }
        let scratch = RpcmemBuffer::alloc(Arc::clone(&driver), so.total_bytes, true)?;
        let mut weights_buf = RpcmemBuffer::alloc(Arc::clone(&driver), offsets.total_bytes, true)?;

        let dst = weights_buf.as_mut_slice();
        let put = |dst: &mut [u8], o: &LinearOffsets, w: &MmapWeight, b: Option<&[f32]>| {
            put_linear(dst, o.w, w)?;
            if let (Some(off), Some(b)) = (o.b, b) {
                put_vec(dst, off, b);
            }
            Ok::<(), CeraError>(())
        };
        put_vec(dst, offsets.input_norm_w, &w.input_norm_w);
        put_vec(dst, offsets.input_norm_b, &w.input_norm_b);
        for (o, l) in offsets.layers.iter().zip(&w.layers) {
            put(dst, &o.q, &l.q_w, None)?;
            put(dst, &o.k, &l.k_w, None)?;
            put(dst, &o.v, &l.v_w, None)?;
            put(dst, &o.o, &l.o_w, Some(&l.o_b))?;
            put(dst, &o.up, &l.up_w, Some(&l.up_b))?;
            put(dst, &o.down, &l.down_w, Some(&l.down_b))?;
            put_vec(dst, o.ln1_w, &l.ln1_w);
            put_vec(dst, o.ln1_b, &l.ln1_b);
            put_vec(dst, o.ln2_w, &l.ln2_w);
            put_vec(dst, o.ln2_b, &l.ln2_b);
        }
        put_vec(dst, offsets.final_norm_w, &w.final_norm_w);
        put_vec(dst, offsets.final_norm_b, &w.final_norm_b);
        put(dst, &offsets.proj, &w.proj_w, Some(&w.proj_b))?;
        put(dst, &offsets.upsample, &w.up_w, Some(&w.up_b))?;
        put(dst, &offsets.dense, &w.dense_w, Some(&w.dense_b))?;
        put(dst, &offsets.out, &w.out_w, Some(&w.out_b))?;
        let pos: Vec<i32> = (0..max_frames as i32).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&pos);
        dst[offsets.rope_pos..offsets.rope_pos + bytes.len()].copy_from_slice(bytes);
        // The zero row stages as zeros: the weights buffer is settled (zeroed) at alloc.
        weights_buf.flush_cpu_cache(0, offsets.total_bytes);

        Ok(Self {
            device,
            weights_buf,
            offsets,
            scratch: Mutex::new(scratch),
            so,
            dims,
            max_frames,
            staged_hi: AtomicUsize::new(0),
        })
    }

    /// The sigmoid speaker activities `[t * 8, n_spk]` for the embedder output `emb`
    /// (`[t, d]`) with `valid_sub` valid 10 ms sub-frames, sub-frames past the valid groups
    /// zeroed. The same result as the CPU `predict` (up to the tanh GELU).
    pub fn predict(&self, emb: &[f32], valid_sub: usize) -> Result<Vec<f32>, CeraError> {
        let t = emb.len() / self.dims.d.max(1);
        let valid = enc_frames(valid_sub);
        let mut out = self.run(emb, t, valid, PredictStage::Full)?;
        // A short output must fail loudly (the file's total-refusal style), never return
        // unmasked: `run` guarantees `t * 8 * n_spk` values, and `valid <= t` there.
        let want = t * SUBSAMPLING * self.dims.n_spk;
        if out.len() != want {
            return Err(CeraError::Backend(format!(
                "nemotron3: NPU predict returned {} values, want {want}",
                out.len()
            )));
        }
        // NeMo's masked output, on the host: the mask covers whole groups.
        for v in &mut out[valid * SUBSAMPLING * self.dims.n_spk..] {
            *v = 0.0;
        }
        Ok(out)
    }

    /// Run `stage` over `emb` (`[t, d]`); what the device probe compares stage by stage with
    /// the CPU. `valid` is the valid group count for the key mask (`enc_frames(valid_sub)`).
    #[doc(hidden)]
    pub fn run(
        &self,
        emb: &[f32],
        t: usize,
        valid: usize,
        stage: PredictStage,
    ) -> Result<Vec<f32>, CeraError> {
        let dims = &self.dims;
        if t == 0 || t > self.max_frames || emb.len() != t * dims.d {
            return Err(CeraError::Backend(format!(
                "nemotron3: {} floats for {t} frames of {} (at most {} frames)",
                emb.len(),
                dims.d,
                self.max_frames
            )));
        }
        if valid == 0 || valid > t {
            return Err(CeraError::Backend(format!(
                "nemotron3: {valid} valid groups for {t} frames"
            )));
        }
        let so = self.so;
        let tp = pad32(t);
        let mut dev = self.device.lock_or_recover();
        let mut scratch = self.scratch.lock_or_recover();
        settled(dev.queue_session_mut(), || {
            let bytes = t * dims.d * 4;
            let buf = scratch.as_mut_slice();
            buf[so.xin..so.xin + bytes].copy_from_slice(bytemuck::cast_slice(emb));
            // The key bias: 0 for valid keys, suppressing past them.
            let bias: &mut [f32] =
                bytemuck::cast_slice_mut(&mut buf[so.keybias..so.keybias + t * 4]);
            bias.fill(0.0);
            for b in &mut bias[valid.min(t)..t] {
                *b = KEY_BIAS_SUPPRESSED;
            }
            // The padded softmax and V^T layouts must read as zero where the DSP's copies
            // do not write. The DSP only ever writes rows/cols `[0..t)` of them (the
            // per-layer `copy_view`s), so:
            // - the first call zeroes the whole max-frames regions (rpcmem is not zeroed
            //   at alloc, so nothing else establishes it),
            // - a shrink step (`t` below the high-water mark) re-zeroes the `t`-sized
            //   regions, since bigger calls may have written stale lanes,
            // - same-size and growth steps skip both (their unread lanes were zeroed
            //   before and untouched since) and only advance the mark.
            // `keybias` is fully rewritten every call, so it has no staleness.
            let hi = self.staged_hi.load(Ordering::Relaxed);
            let zero_now = hi == 0 || t < hi;
            let (zt, ztp) = if hi == 0 {
                (self.max_frames, pad32(self.max_frames))
            } else {
                (t, tp)
            };
            if zero_now {
                let zero = [
                    (so.sm, dims.heads * zt * ztp * 4),
                    (so.vt, dims.heads * dims.dh * ztp * 4),
                ];
                for (off, len) in zero {
                    buf[off..off + len].fill(0);
                }
            }
            scratch.flush_cpu_cache(so.xin, bytes);
            scratch.flush_cpu_cache(so.keybias, t * 4);
            if zero_now {
                scratch.flush_cpu_cache(so.sm, dims.heads * zt * ztp * 4);
                scratch.flush_cpu_cache(so.vt, dims.heads * dims.dh * ztp * 4);
            }
            if t > hi {
                self.staged_hi.store(t, Ordering::Relaxed);
            }
        })
        .map_err(|e| CeraError::Backend(format!("nemotron3: {e}")))?;

        let mut out = (0, 0);
        let max_frames = self.max_frames;
        run_on_queue("nemotron3 predict", dev.queue_session_mut(), |session| {
            out = emit_predict(
                session,
                &self.weights_buf,
                &scratch,
                &self.offsets,
                &so,
                dims,
                t,
                max_frames,
                stage,
                &mut |sess| sess.flush(),
            )?;
            session.flush()
        })?;
        let (off, floats) = out;
        scratch.invalidate_cpu_cache(off, floats * 4);
        Ok(bytemuck::cast_slice::<u8, f32>(&scratch.as_slice()[off..off + floats * 4]).to_vec())
    }
}

impl Drop for HexagonNemotron3Predict {
    fn drop(&mut self) {
        let mut dev = self.device.lock_or_recover();
        let session: &mut crate::backend::hexagon::HexagonQueueSession = dev.queue_session_mut();
        session.release_dsp_references([&self.weights_buf, &*self.scratch.lock_or_recover()]);
    }
}

/// The embedder (stack 8 mel frames, project to the encoder width) on the NPU. The mel upload
/// is host-padded to whole groups, so the DSP runs one matmul over the buffer read as
/// `[t, 8 * n_mel]` rows.
pub struct HexagonNemotron3Embed {
    device: Arc<Mutex<HexagonDevice>>,
    weights_buf: RpcmemBuffer,
    w: HexagonWeightDesc,
    scratch: Mutex<RpcmemBuffer>,
    mel: usize,
    emb: usize,
    n_mel: usize,
    d: usize,
    max_frames: usize,
}

// SAFETY: once shared, the rpcmem buffers are only touched while holding `device` and
// `scratch` (construction writes them under exclusive ownership), as in the predictor.
unsafe impl Send for HexagonNemotron3Embed {}
unsafe impl Sync for HexagonNemotron3Embed {}

impl HexagonNemotron3Embed {
    /// Stage the embedder on the device for up to `max_frames` encoder frames (`8 *`
    /// mel frames). Needs Q8_0 or Q4_0 weights, like the predictor.
    pub fn new(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        model: &Nemotron3Model,
        max_frames: usize,
    ) -> Result<Self, CeraError> {
        let w = model.weights();
        let (n_mel, d) = (w.config.n_mel_bins, w.config.n_embd);
        check_staging_window(max_frames, w.config.n_head)?;
        check_mat(&w.embed_w, d, n_mel * SUBSAMPLING, "embed")?;
        let mut cur = 0;
        let desc = plan_linear(&mut cur, &w.embed_w).map_err(|e| {
            CeraError::Backend(format!(
                "nemotron3 embed: {e}; the NPU reads Q8_0 or Q4_0 weights, so convert the \
                 model with `--tail-outtype q8_0` (scripts/nemotron3_diarization/README.md)"
            ))
        })?;
        let mut weights_buf = RpcmemBuffer::alloc(Arc::clone(&driver), cur, true)?;
        put_linear(weights_buf.as_mut_slice(), desc, &w.embed_w)?;
        weights_buf.flush_cpu_cache(0, cur);
        let mel = 0;
        let emb = align128(max_frames * SUBSAMPLING * n_mel * 4);
        let total = emb + align128(max_frames * d * 4);
        let scratch = RpcmemBuffer::alloc(Arc::clone(&driver), total, true)?;
        Ok(Self {
            device,
            weights_buf,
            w: desc,
            scratch: Mutex::new(scratch),
            mel,
            emb,
            n_mel,
            d,
            max_frames,
        })
    }

    /// The `[t, d]` embeddings for `n_frames` of `[n_frames, n_mel]` mel, like the CPU
    /// `embed` (the tail group zero-padded).
    pub fn embed(&self, mel: &[f32], n_frames: usize) -> Result<Vec<f32>, CeraError> {
        let t = enc_frames(n_frames);
        if n_frames == 0 || t > self.max_frames || mel.len() != n_frames * self.n_mel {
            return Err(CeraError::Backend(format!(
                "nemotron3 embed: {} floats for {n_frames} mel frames (at most {} groups)",
                mel.len(),
                self.max_frames
            )));
        }
        let mut dev = self.device.lock_or_recover();
        let mut scratch = self.scratch.lock_or_recover();
        settled(dev.queue_session_mut(), || {
            let buf = scratch.as_mut_slice();
            let bytes = n_frames * self.n_mel * 4;
            buf[self.mel..self.mel + bytes].copy_from_slice(bytemuck::cast_slice(mel));
            // Zero-pad the tail group.
            let padded = t * SUBSAMPLING * self.n_mel * 4;
            buf[self.mel + bytes..self.mel + padded].fill(0);
            scratch.flush_cpu_cache(self.mel, padded);
        })
        .map_err(|e| CeraError::Backend(format!("nemotron3 embed: {e}")))?;

        run_on_queue("nemotron3 embed", dev.queue_session_mut(), |session| {
            dispatch::linear_m(
                session,
                &scratch,
                self.mel,
                &self.weights_buf,
                self.w,
                None,
                &scratch,
                self.emb,
                t,
                TILE,
            )?;
            session.flush()
        })?;
        scratch.invalidate_cpu_cache(self.emb, t * self.d * 4);
        Ok(bytemuck::cast_slice::<u8, f32>(
            &scratch.as_slice()[self.emb..self.emb + t * self.d * 4],
        )
        .to_vec())
    }
}

impl Drop for HexagonNemotron3Embed {
    fn drop(&mut self) {
        let mut dev = self.device.lock_or_recover();
        dev.queue_session_mut()
            .release_dsp_references([&self.weights_buf, &*self.scratch.lock_or_recover()]);
    }
}

/// The whole Nemotron-3 network on the Hexagon NPU: the log-mel front end
/// ([`HexagonSortformerMel`]), the embedder and the predictor.
pub struct HexagonNemotron3 {
    mel: HexagonSortformerMel,
    embed: HexagonNemotron3Embed,
    predict: HexagonNemotron3Predict,
    max_frames: usize,
}

impl HexagonNemotron3 {
    /// Stage the model on the device for steps of up to `max_frames` encoder frames (use
    /// [`window_frames`](crate::model::nemotron3_diarization::StreamingParams::window_frames)).
    /// Needs the Q8_0-tail GGUF, see the module docs.
    pub fn new(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        model: &Nemotron3Model,
        max_frames: usize,
    ) -> Result<Self, CeraError> {
        let w = model.weights();
        // The embedder stages before the predictor (which budgets precisely): refuse a
        // hostile window here so its linear layout math cannot wrap first.
        check_staging_window(max_frames, w.config.n_head)?;
        let mel = HexagonSortformerMel::new(
            Arc::clone(&driver),
            Arc::clone(&device),
            &w.window,
            &w.mel_fb,
            w.config.n_mel_bins,
        )
        .map_err(|e| CeraError::Backend(format!("nemotron3 mel: {e}")))?;
        let embed = HexagonNemotron3Embed::new(
            Arc::clone(&driver),
            Arc::clone(&device),
            model,
            max_frames,
        )?;
        let predict = HexagonNemotron3Predict::new(driver, device, model, max_frames)?;
        Ok(Self {
            mel,
            embed,
            predict,
            max_frames,
        })
    }

    /// Encoder frames the staging covers; longer inputs are declined (the CPU takes them).
    pub fn max_frames(&self) -> usize {
        self.max_frames
    }
}

impl Nemotron3Accelerator for HexagonNemotron3 {
    fn embed(&self, mel: &[f32], n_frames: usize) -> anyhow::Result<Option<Vec<f32>>> {
        if n_frames == 0 || enc_frames(n_frames) > self.max_frames {
            return Ok(None);
        }
        Ok(Some(self.embed.embed(mel, n_frames)?))
    }

    fn predict(&self, emb: &[f32], valid_sub: usize) -> anyhow::Result<Option<Vec<f32>>> {
        if emb.is_empty() || emb.len() / self.predict.dims.d.max(1) > self.max_frames {
            return Ok(None);
        }
        Ok(Some(self.predict.predict(emb, valid_sub)?))
    }

    fn log_mel(&self, samples: &[f32], n_frames: usize) -> anyhow::Result<Option<Vec<f32>>> {
        if n_frames == 0 {
            return Ok(None);
        }
        Ok(Some(self.mel.log_mel(samples, n_frames)?))
    }
}

/// Stage the model on the Hexagon NPU for steps of up to `max_frames` encoder frames and
/// install the staging as its accelerator, sharing the loader with Sortformer (see
/// `stage_diarizer_accelerator`).
/// `None` (with a log line) when the model already has an accelerator, the DSP is
/// unavailable, or staging fails: the caller keeps the CPU. Call once during setup;
/// concurrent staging is not supported.
pub fn try_hexagon_nemotron3(
    model: &Nemotron3Model,
    max_frames: usize,
) -> Option<Arc<HexagonNemotron3>> {
    crate::backend::hexagon::stage_diarizer_accelerator(
        "Nemotron3",
        model.has_accelerator(),
        max_frames,
        |driver, device| HexagonNemotron3::new(driver, device, model, max_frames),
        |staged| model.set_accelerator(staged),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::hexagon::dispatch::testing::RecordingSink;
    use crate::backend::hexagon::{HexagonWeightFormat, HtpOpCode};
    use crate::model::nemotron3_diarization::synthetic_q8_weights;

    fn desc(seq: usize, rows: usize, cols: usize) -> HexagonWeightDesc {
        HexagonWeightDesc {
            offset: 0x1000 * seq,
            size_bytes: 12345,
            format: HexagonWeightFormat::RepackedQ8_0,
            rows,
            cols,
        }
    }

    fn lin(seq: usize, rows: usize, cols: usize, bias: bool) -> LinearOffsets {
        LinearOffsets {
            w: desc(seq, rows, cols),
            b: bias.then_some(0),
        }
    }

    /// The checkpoint's real widths, `layers` encoder blocks. Every weight gets a distinct
    /// `0x1000 * seq` offset so a test can pin which weight an op reads (a swapped desc
    /// changes a recorded address).
    fn fixture(layers: usize) -> (PredictOffsets, Dims) {
        let dims = Dims {
            d: 512,
            heads: 8,
            dh: 64,
            inner: 2048,
            tf_d: 192,
            n_spk: 8,
            eps: 1e-5,
            rope_theta: 10000.0,
        };
        let mut seq = 1usize;
        let mut lin_next = |rows: usize, cols: usize, bias: bool| {
            let l = lin(seq, rows, cols, bias);
            seq += 1;
            l
        };
        let mut layer_next = || LayerOffsets {
            q: lin_next(512, 512, false),
            k: lin_next(512, 512, false),
            v: lin_next(512, 512, false),
            o: lin_next(512, 512, true),
            ln1_w: 0,
            ln1_b: 0,
            up: lin_next(2048, 512, true),
            down: lin_next(512, 2048, true),
            ln2_w: 0,
            ln2_b: 0,
        };
        let o = PredictOffsets {
            input_norm_w: 0,
            input_norm_b: 0,
            layers: (0..layers).map(|_| layer_next()).collect(),
            final_norm_w: 0,
            final_norm_b: 0,
            proj: lin_next(192, 512, true),
            upsample: lin_next(192 * 8, 192 * 3, true),
            dense: lin_next(192, 192, true),
            out: lin_next(8, 192, true),
            rope_pos: 0,
            zero_row: 0,
            total_bytes: 0,
        };
        (o, dims)
    }

    fn emit_with(
        stage: PredictStage,
        o: &PredictOffsets,
        dims: &Dims,
        t: usize,
    ) -> (RecordingSink, (usize, usize), Scratch) {
        let so = Scratch::new(dims, 64);
        let mut s = RecordingSink::default();
        let out = emit_predict(
            &mut s,
            &"w",
            &"s",
            o,
            &so,
            dims,
            t,
            64,
            stage,
            &mut |_: &mut RecordingSink| Ok(()),
        )
        .unwrap();
        (s, out, so)
    }

    fn emit(
        stage: PredictStage,
        layers: usize,
        t: usize,
    ) -> (RecordingSink, (usize, usize), Scratch) {
        let (o, dims) = fixture(layers);
        emit_with(stage, &o, &dims, t)
    }

    #[test]
    fn a_weight_the_npu_cannot_read_names_the_fix() {
        let dense = MmapWeight::from_owned_f32(vec![0.0; 4 * 32], 4, 32);
        let mut cur = 0;
        let err = linear(&mut cur, &dense, None, "encoder layer 3 q")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("encoder layer 3 q") && err.contains("--tail-outtype q8_0"),
            "{err}"
        );
    }

    /// The full opcode sequence at `t=8` (every tiled op runs once): pre-norm layers with
    /// bias-free QKV, one RoPE per Q/K, the key-bias add between the score matmul and the
    /// softmax, a tanh GELU (never the DSP's quick `UNARY_GELU`), and the head ending in a
    /// sigmoid. Op counts alone pass a within-layer swap.
    #[test]
    fn a_layer_rotates_masks_and_ends_in_a_sigmoid() {
        use HtpOpCode::*;
        let (s, out, so) = emit(PredictStage::Full, 2, 8);
        let norm = [Norm, Mul, Add];
        let layer = [
            Norm,
            Mul,
            Add, // ln1
            MulMat,
            MulMat,
            MulMat, // qkv, no bias
            Rope,
            Rope, // q, k
            MulMat,
            Add,
            Softmax, // scores, key-bias add, softmax
            Cpy,
            Cpy,
            MulMat, // padded softmax, V^T, contraction
            MulMat,
            Add,
            Add, // o proj, residual
            Norm,
            Mul,
            Add, // ln2
            MulMat,
            Add, // up
            Cpy,
            Mul,
            Scale,
            Mul,
            Scale,
            UnarySigmoid,
            Mul, // tanh GELU
            MulMat,
            Add,
            Add, // down, residual
        ];
        let head = [
            UnaryRelu,
            MulMat,
            Add, // dense
            UnaryRelu,
            MulMat,
            Add, // out
            UnarySigmoid,
        ];
        let matadd = [MulMat, Add];
        let copies = [Cpy; 4]; // padded h, zero edges, one unroll for all frames
        let expect: Vec<u32> = norm
            .iter()
            .chain(&layer)
            .chain(&layer)
            .chain(&norm)
            .chain(&matadd) // proj
            .chain(&copies)
            .chain(&matadd) // upsample
            .chain(&head)
            .map(|o| *o as u32)
            .collect();
        assert_eq!(s.opcodes(), expect);
        assert_eq!(out, (so.pred, 8 * 8 * 8));
    }

    /// The two RoPEs run NeoX (mode 2) over the `[dh, heads, t]` row-major Q/K, and the
    /// score matmul reads full 64-wide heads (no lane padding): the contraction key is the
    /// padded frame count.
    #[test]
    fn rope_is_neox_and_heads_need_no_lane_padding() {
        let (s, _, _) = emit(PredictStage::Layers(1), 1, 8);
        let ropes: Vec<usize> = (0..s.ops.len())
            .filter(|&i| s.ops[i].opcode == HtpOpCode::Rope as u32)
            .collect();
        assert_eq!(ropes.len(), 2, "one RoPE for Q and one for K");
        for r in ropes {
            assert_eq!(s.ops[r].params[2], RopeType::Neox.htp_mode() as i32);
            assert_eq!(s.src(r, 0).ne, [64, 8, 8, 1], "Q/K [dh, heads, t]");
            assert_eq!(
                s.src(r, 0).nb,
                [4, 64 * 4, 512 * 4, 512 * 8 * 4],
                "row-major [t, d]"
            );
        }
        let f32_mm: Vec<usize> = (0..s.ops.len())
            .filter(|&i| {
                s.ops[i].opcode == HtpOpCode::MulMat as u32 && s.ops[i].src.len() == 2 && {
                    let a = s.src(i, 0);
                    a.dtype == crate::backend::hexagon::HtpDataType::F32 as u32
                }
            })
            .collect();
        assert_eq!(f32_mm.len(), 2, "scores and contraction");
        assert_eq!(s.src(f32_mm[0], 0).ne[0], 64, "K head width");
        assert_eq!(s.src(f32_mm[0], 1).ne[0], 64, "Q head width");
        assert_eq!(s.src(f32_mm[1], 0).ne[0], 32, "pad32(8) padded keys");
    }

    /// The key bias reaches the scores as an in-place add of the staged bias row (the
    /// softmax runs unmasked): exactly one `Add` reads `so.keybias`.
    #[test]
    fn the_key_mask_is_a_bias_add_not_a_softmax_mask() {
        let (s, _, so) = emit(PredictStage::Layers(1), 1, 8);
        let adds: Vec<usize> = (0..s.ops.len())
            .filter(|&i| {
                s.ops[i].opcode == HtpOpCode::Add as u32
                    && s.ops[i].src.len() == 2
                    && s.src(i, 1).offset == so.keybias
            })
            .collect();
        assert_eq!(adds.len(), 1, "one key-bias add per layer");
        // In place over the scores: dst is src.
        assert_eq!(s.ops[adds[0]].src[0], s.ops[adds[0]].dst[0]);
        assert_eq!(s.src(adds[0], 1).ne[0], 8, "one bias per key");
        // And the softmax takes no mask tensor.
        let smax: Vec<usize> = (0..s.ops.len())
            .filter(|&i| s.ops[i].opcode == HtpOpCode::Softmax as u32)
            .collect();
        assert_eq!(smax.len(), 1);
        assert_eq!(s.ops[smax[0]].src.len(), 1, "scores only, no mask");
    }

    /// Every stage returns its documented buffer and float count, and `Layers(n)` never runs
    /// past the staged layers.
    #[test]
    fn stages_return_the_documented_shapes() {
        let (_, out, so) = emit(PredictStage::InputNorm, 3, 8);
        assert_eq!(out, (so.x, 8 * 512));
        let (_, out, so) = emit(PredictStage::Layers(2), 3, 8);
        assert_eq!(out, (so.x, 8 * 512));
        let (_, out, so) = emit(PredictStage::FinalNorm, 3, 8);
        assert_eq!(out, (so.norm, 8 * 512));
        let (_, out, so) = emit(PredictStage::Proj, 3, 8);
        assert_eq!(out, (so.h, 8 * 192));
        let (_, out, so) = emit(PredictStage::Upsampled, 3, 8);
        assert_eq!(out, (so.conv, 8 * 8 * 192));
        let (_, out, so) = emit(PredictStage::Logits, 3, 8);
        assert_eq!(out, (so.pred, 8 * 8 * 8));
        let (s_all, _, _) = emit(PredictStage::Layers(99), 3, 8);
        let (s_three, _, _) = emit(PredictStage::Layers(3), 3, 8);
        assert_eq!(s_all.opcodes(), s_three.opcodes());
    }

    /// The upsample copies pad the edges from the staged zero row and unroll every
    /// frame's three taps in one strided copy: 4 copies, and the recorded strides must
    /// gather exactly the old per-frame elements.
    #[test]
    fn the_upsample_copies_pad_edges_and_unroll_frames() {
        let (s, _, so) = emit(PredictStage::Upsampled, 0, 4);
        let copies: Vec<usize> = (0..s.ops.len())
            .filter(|&i| s.ops[i].opcode == HtpOpCode::Cpy as u32)
            .collect();
        // No layers, so these are exactly the upsample copies: h into the padded buffer,
        // two zero edges, one unroll for all frames.
        assert_eq!(copies.len(), 4);
        // The edge copies read the staged zero row in the weights buffer.
        for c in &copies[1..3] {
            assert_eq!(s.src(*c, 0).buf, "w");
            assert_eq!(s.src(*c, 0).ne[0], 192);
        }
        // The unroll gathers `(tap, channel, frame)` with a row step per tap and per frame.
        let u = copies[3];
        assert_eq!(s.src(u, 0).buf, "s");
        assert_eq!(s.src(u, 0).offset, so.hpadded);
        assert_eq!(s.src(u, 0).ne, [3, 192, 4, 1]);
        assert_eq!(s.src(u, 0).nb[..3], [192 * 4, 4, 192 * 4]);
        assert_eq!(s.dst(u).offset, so.unrolled);
        assert_eq!(s.dst(u).ne, [3, 192, 4, 1]);
        assert_eq!(s.dst(u).nb[..3], [4, 12, 3 * 192 * 4]);
        // Every gathered element lands where the per-frame semantics say: source element
        // `(d, ch, f)` is padded frame `f + d`, channel `ch`; the destination packs it as
        // `unrolled[f][ch * 3 + d]`. Strides come from the recording, addresses from the
        // semantics, so neither side can drift with the other.
        let snb = s.src(u, 0).nb;
        let dnb = s.dst(u).nb;
        for f in 0..4usize {
            for ch in 0..192usize {
                for d in 0..3usize {
                    assert_eq!(
                        snb[0] as usize * d + snb[1] as usize * ch + snb[2] as usize * f,
                        (f + d) * 192 * 4 + ch * 4,
                        "src (d={d}, ch={ch}, f={f})"
                    );
                    assert_eq!(
                        dnb[0] as usize * d + dnb[1] as usize * ch + dnb[2] as usize * f,
                        f * 3 * 192 * 4 + (ch * 3 + d) * 4,
                        "dst (d={d}, ch={ch}, f={f})"
                    );
                }
            }
        }
    }

    /// A repeat staging attempt stages nothing and keeps the first accelerator. The early-out
    /// fires before any NPU interaction, so the contract (plus its log line) holds on host.
    #[cfg(feature = "mmap")]
    #[test]
    fn try_hexagon_nemotron3_keeps_the_first_accelerator() {
        use crate::model::nemotron3_diarization::{Nemotron3Accelerator, Nemotron3Model};
        use tracing_subscriber::layer::SubscriberExt;

        struct Decline;
        impl Nemotron3Accelerator for Decline {
            fn embed(&self, _mel: &[f32], _n_frames: usize) -> anyhow::Result<Option<Vec<f32>>> {
                Ok(None)
            }
            fn predict(&self, _emb: &[f32], _valid_sub: usize) -> anyhow::Result<Option<Vec<f32>>> {
                Ok(None)
            }
        }

        let path = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".leap/models/nemotron3-diarization/nemotron3-diarization-q8_0-npu.gguf");
        if !crate::model::transformer::require_model_or_skip(&path) {
            return;
        }
        let model = Nemotron3Model::from_file(&path).unwrap();
        assert!(!model.has_accelerator());
        model.set_accelerator(std::sync::Arc::new(Decline)).unwrap();
        assert!(model.has_accelerator());

        #[derive(Clone, Default)]
        struct InfoCapture(std::sync::Arc<std::sync::Mutex<Vec<String>>>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for InfoCapture {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _ctx: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if *event.metadata().level() != tracing::Level::INFO {
                    return;
                }
                struct Msg<'a>(&'a mut String);
                impl tracing::field::Visit for Msg<'_> {
                    fn record_debug(
                        &mut self,
                        field: &tracing::field::Field,
                        value: &dyn std::fmt::Debug,
                    ) {
                        if field.name() == "message" {
                            self.0.push_str(&format!("{value:?}"));
                        }
                    }
                }
                let mut msg = String::new();
                event.record(&mut Msg(&mut msg));
                if !msg.is_empty() {
                    self.0.lock().unwrap_or_else(|p| p.into_inner()).push(msg);
                }
            }
        }
        let capture = InfoCapture::default();
        let sub = tracing_subscriber::registry().with(capture.clone());
        let out = tracing::subscriber::with_default(sub, || try_hexagon_nemotron3(&model, 64));
        assert!(out.is_none(), "a repeat call stages nothing");
        assert!(
            model.has_accelerator(),
            "the repeat call keeps the first accelerator"
        );
        let fired = capture
            .0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .any(|m| m.contains("already has an accelerator"));
        assert!(fired, "the early-out fires instead of staging");
    }

    /// Every weighted matmul reads its own desc, in network order: with distinct fixture
    /// offsets, swapping two descs in the emitter (or reading the wrong layer's weights)
    /// changes a recorded address and fails this test. The weight operand is src 0
    /// (`linear_m` enqueues `&[w_ti, x_ti]`); the score matmuls read scratch on both sides.
    #[test]
    fn every_weighted_matmul_reads_its_own_desc() {
        let (o, dims) = fixture(2);
        let (s, _, _) = emit_with(PredictStage::Full, &o, &dims, 8);
        let got: Vec<usize> = (0..s.ops.len())
            .filter(|&i| s.ops[i].opcode == HtpOpCode::MulMat as u32 && s.src(i, 0).buf == "w")
            .map(|i| s.src(i, 0).offset)
            .collect();
        let mut want = Vec::new();
        for l in &o.layers {
            want.extend([
                l.q.w.offset,
                l.k.w.offset,
                l.v.w.offset,
                l.o.w.offset,
                l.up.w.offset,
                l.down.w.offset,
            ]);
        }
        want.extend([
            o.proj.w.offset,
            o.upsample.w.offset,
            o.dense.w.offset,
            o.out.w.offset,
        ]);
        assert_eq!(got, want);
    }

    /// Small widths for the planner tests (`plan` reads shapes and dtypes only).
    fn plan_dims() -> Dims {
        Dims {
            d: 64,
            heads: 2,
            dh: 32,
            inner: 128,
            tf_d: 32,
            n_spk: 8,
            eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }

    /// `plan` lays out disjoint, prefix-packed weight regions summing to `total_bytes`:
    /// every recorded offset equals the running hand-sum in plan order, so an overlap or a
    /// dropped region fails here instead of corrupting silently on device. Region sizes come
    /// from `align128` and the repack size fn (pinned by their own unit tests); the order and
    /// the per-weight dims are this test's.
    #[test]
    fn plan_lays_out_disjoint_weight_regions() {
        use crate::backend::hexagon::repacked_matrix_size_q8_0;

        let dims = plan_dims();
        let w = synthetic_q8_weights(1, dims.d, dims.heads, dims.inner, dims.tf_d, dims.n_spk);
        let o = PredictOffsets::plan(&w, &dims, 33).unwrap();
        assert_eq!(o.layers.len(), 1);
        let l = &o.layers[0];

        let mat_bytes = |rows: usize, cols: usize| repacked_matrix_size_q8_0(cols, rows).unwrap();
        let mut regions: Vec<(&str, usize, usize)> = Vec::new();
        let vec_region = |regions: &mut Vec<(&'static str, usize, usize)>,
                          name: &'static str,
                          off: usize,
                          len: usize| {
            regions.push((name, off, align128(len * 4)));
        };
        let mat_region = |regions: &mut Vec<(&'static str, usize, usize)>,
                          name: &'static str,
                          d: &HexagonWeightDesc,
                          rows: usize,
                          cols: usize| {
            assert_eq!(d.size_bytes, mat_bytes(rows, cols), "{name} repacked size");
            assert_eq!((d.rows, d.cols), (rows, cols), "{name} dims");
            regions.push((name, d.offset, align128(d.size_bytes)));
        };
        vec_region(&mut regions, "input_norm_w", o.input_norm_w, dims.d);
        vec_region(&mut regions, "input_norm_b", o.input_norm_b, dims.d);
        vec_region(&mut regions, "ln1_w", l.ln1_w, dims.d);
        vec_region(&mut regions, "ln1_b", l.ln1_b, dims.d);
        vec_region(&mut regions, "ln2_w", l.ln2_w, dims.d);
        vec_region(&mut regions, "ln2_b", l.ln2_b, dims.d);
        mat_region(&mut regions, "q", &l.q.w, dims.d, dims.d);
        mat_region(&mut regions, "k", &l.k.w, dims.d, dims.d);
        mat_region(&mut regions, "v", &l.v.w, dims.d, dims.d);
        mat_region(&mut regions, "o", &l.o.w, dims.d, dims.d);
        vec_region(&mut regions, "o_b", l.o.b.unwrap(), dims.d);
        mat_region(&mut regions, "up", &l.up.w, dims.inner, dims.d);
        vec_region(&mut regions, "up_b", l.up.b.unwrap(), dims.inner);
        mat_region(&mut regions, "down", &l.down.w, dims.d, dims.inner);
        vec_region(&mut regions, "down_b", l.down.b.unwrap(), dims.d);
        vec_region(&mut regions, "final_norm_w", o.final_norm_w, dims.d);
        vec_region(&mut regions, "final_norm_b", o.final_norm_b, dims.d);
        mat_region(&mut regions, "proj", &o.proj.w, dims.tf_d, dims.d);
        vec_region(&mut regions, "proj_b", o.proj.b.unwrap(), dims.tf_d);
        mat_region(
            &mut regions,
            "upsample",
            &o.upsample.w,
            dims.tf_d * SUBSAMPLING,
            dims.tf_d * 3,
        );
        vec_region(
            &mut regions,
            "upsample_b",
            o.upsample.b.unwrap(),
            dims.tf_d * SUBSAMPLING,
        );
        mat_region(&mut regions, "dense", &o.dense.w, dims.tf_d, dims.tf_d);
        vec_region(&mut regions, "dense_b", o.dense.b.unwrap(), dims.tf_d);
        mat_region(&mut regions, "out", &o.out.w, dims.n_spk, dims.tf_d);
        vec_region(&mut regions, "out_b", o.out.b.unwrap(), dims.n_spk);
        vec_region(&mut regions, "rope_pos", o.rope_pos, 33);
        vec_region(&mut regions, "zero_row", o.zero_row, dims.tf_d);

        // Prefix-packed in plan order.
        let mut cur = 0;
        for (name, off, len) in &regions {
            assert_eq!(*off, cur, "{name} offset");
            cur += len;
        }
        assert_eq!(o.total_bytes, cur, "total_bytes hand-sum");

        // Pairwise disjoint and inside the slab (the named regression: overlapping weight
        // regions must fail the suite, not corrupt on device).
        let mut by_off = regions.clone();
        by_off.sort_by_key(|&(_, off, _)| off);
        for w in by_off.windows(2) {
            let (an, a_off, a_len) = w[0];
            let (bn, b_off, _) = w[1];
            assert!(
                a_off + a_len <= b_off,
                "{an} [{}..{}] overlaps {bn} at {}",
                a_off,
                a_off + a_len,
                b_off
            );
        }
        for (name, off, len) in &by_off {
            assert!(off + len <= o.total_bytes, "{name} past total_bytes");
        }
    }

    /// One bad-shape case per planner guard: each mutation must fail naming its weight.
    #[test]
    fn plan_rejects_every_misshaped_weight() {
        let dims = plan_dims();
        let fresh =
            || synthetic_q8_weights(1, dims.d, dims.heads, dims.inner, dims.tf_d, dims.n_spk);
        type Mutate = Box<dyn FnOnce(&mut Nemotron3Weights)>;
        let mut cases: Vec<(&str, Mutate, &str)> = Vec::new();
        cases.push((
            "input norm",
            Box::new(|w: &mut Nemotron3Weights| {
                w.input_norm_w.pop();
            }),
            "input norm",
        ));
        cases.push((
            "q rows",
            Box::new(|w: &mut Nemotron3Weights| {
                w.layers[0].q_w.rows = 32;
            }),
            "encoder layer 0 q",
        ));
        cases.push((
            "k cols",
            Box::new(|w: &mut Nemotron3Weights| {
                w.layers[0].k_w.cols = 32;
            }),
            "encoder layer 0 k",
        ));
        cases.push((
            "v rows",
            Box::new(|w: &mut Nemotron3Weights| {
                w.layers[0].v_w.rows = 32;
            }),
            "encoder layer 0 v",
        ));
        cases.push((
            "o rows",
            Box::new(|w: &mut Nemotron3Weights| {
                w.layers[0].o_w.rows = 32;
            }),
            "encoder layer 0 o",
        ));
        cases.push((
            "o bias",
            Box::new(|w: &mut Nemotron3Weights| {
                w.layers[0].o_b.pop();
            }),
            "encoder layer 0 o bias",
        ));
        cases.push((
            "up rows",
            Box::new(|w: &mut Nemotron3Weights| {
                w.layers[0].up_w.rows = 64;
            }),
            "encoder layer 0 up",
        ));
        cases.push((
            "down cols",
            Box::new(|w: &mut Nemotron3Weights| {
                w.layers[0].down_w.cols = 64;
            }),
            "encoder layer 0 down",
        ));
        cases.push((
            "ffn bias",
            Box::new(|w: &mut Nemotron3Weights| {
                w.layers[0].up_b.pop();
            }),
            "encoder layer 0 ffn",
        ));
        cases.push((
            "ln1",
            Box::new(|w: &mut Nemotron3Weights| {
                w.layers[0].ln1_w.pop();
            }),
            "encoder layer 0 ln1",
        ));
        cases.push((
            "ln2",
            Box::new(|w: &mut Nemotron3Weights| {
                w.layers[0].ln2_b.pop();
            }),
            "encoder layer 0 ln2",
        ));
        cases.push((
            "final norm",
            Box::new(|w: &mut Nemotron3Weights| {
                w.final_norm_w.pop();
            }),
            "final norm",
        ));
        cases.push((
            "proj rows",
            Box::new(|w: &mut Nemotron3Weights| {
                w.proj_w.rows = 16;
            }),
            "proj is",
        ));
        cases.push((
            "proj bias",
            Box::new(|w: &mut Nemotron3Weights| {
                w.proj_b.pop();
            }),
            "proj has",
        ));
        cases.push((
            "upsample rows",
            Box::new(|w: &mut Nemotron3Weights| {
                w.up_w.rows = 100;
            }),
            "upsample is",
        ));
        cases.push((
            "upsample taps",
            Box::new({
                let tf_d = dims.tf_d;
                move |w: &mut Nemotron3Weights| {
                    w.up_w.cols = tf_d * 2;
                }
            }),
            "upsample is",
        ));
        cases.push((
            "upsample bias",
            Box::new(|w: &mut Nemotron3Weights| {
                w.up_b.pop();
            }),
            "upsample has",
        ));
        cases.push((
            "dense rows",
            Box::new(|w: &mut Nemotron3Weights| {
                w.dense_w.rows = 16;
            }),
            "dense is",
        ));
        cases.push((
            "dense bias",
            Box::new(|w: &mut Nemotron3Weights| {
                w.dense_b.pop();
            }),
            "dense has",
        ));
        cases.push((
            "out cols",
            Box::new(|w: &mut Nemotron3Weights| {
                w.out_w.cols = 16;
            }),
            "out is",
        ));
        cases.push((
            "out bias",
            Box::new(|w: &mut Nemotron3Weights| {
                w.out_b.pop();
            }),
            "out has",
        ));
        cases.push((
            "k dtype",
            Box::new(|w: &mut Nemotron3Weights| {
                w.layers[0].k_w = MmapWeight::from_owned_f32(vec![0.0; 64 * 64], 64, 64);
            }),
            "--tail-outtype q8_0",
        ));
        for (name, mutate, want) in cases {
            let mut w = fresh();
            mutate(&mut w);
            let err = PredictOffsets::plan(&w, &dims, 33).unwrap_err().to_string();
            assert!(err.contains(want), "{name}: {err}");
        }
    }

    /// Scratch regions at a non-multiple-of-32 `max_frames`: every region starts 128-aligned
    /// and the sm/vt gaps pin `pad32(33) == 64` with literals (not the function itself).
    #[test]
    fn scratch_regions_stay_aligned_and_pad32_wide() {
        let (_, dims) = fixture(0);
        let so = Scratch::new(&dims, 33);
        for (name, off) in [
            ("xin", so.xin),
            ("x", so.x),
            ("norm", so.norm),
            ("q", so.q),
            ("k", so.k),
            ("v", so.v),
            ("keybias", so.keybias),
            ("ac", so.ac),
            ("smc", so.smc),
            ("sm", so.sm),
            ("vt", so.vt),
            ("av", so.av),
            ("out", so.out),
            ("ff", so.ff),
            ("gelu_tmp", so.gelu_tmp),
            ("h", so.h),
            ("hpadded", so.hpadded),
            ("unrolled", so.unrolled),
            ("conv", so.conv),
            ("hid", so.hid),
            ("pred", so.pred),
        ] {
            assert_eq!(off % 128, 0, "{name}");
        }
        // sm holds heads*t*tp f32 and vt heads*dh*tp with tp = pad32(33) = 64.
        assert_eq!(so.vt - so.sm, align128(8 * 33 * 64 * 4));
        assert_eq!(so.av - so.vt, align128(8 * 64 * 64 * 4));
        assert_eq!(so.total_bytes % 128, 0);
        assert!(so.total_bytes > so.pred, "pred inside the slab");
    }

    /// Without a DSP the staging declines and the model keeps running on the CPU.
    /// Host-only: on-device the NPU may exist. Model-gated like the keeps-first test above.
    #[cfg(all(feature = "mmap", not(target_os = "android")))]
    #[test]
    fn try_hexagon_nemotron3_declines_without_a_dsp() {
        let path = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".leap/models/nemotron3-diarization/nemotron3-diarization-q8_0-npu.gguf");
        if !crate::model::transformer::require_model_or_skip(&path) {
            return;
        }
        let model = Nemotron3Model::from_file(&path).unwrap();
        assert!(try_hexagon_nemotron3(&model, 64).is_none());
        assert!(!model.has_accelerator());
        // The declined model still diarizes on the CPU (1 s of silence).
        let out = model.diarize_offline(&vec![0.0; 16_000]).unwrap();
        assert!(!out.is_empty());
    }
}
