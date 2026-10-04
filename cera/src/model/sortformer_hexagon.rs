//! The part of the Sortformer diarizer that follows the FastConformer, on the Hexagon NPU:
//! `encoder_proj`, the 18 post-LN Transformer layers and the speaker head.
//!
//! Together with `audio_encoder_hexagon` (the conv stem and the 17 FastConformer blocks,
//! staged with `HexagonAudioEncoder::from_parts`) this keeps the whole network on the DSP,
//! which is what an always-on background service wants: Android demotes background CPU work
//! and does not demote the NPU.
//!
//! Each layer is `x = LN1(x + Attn(x)); x = LN2(x + FFN(x))` (full attention, no mask, no
//! positional embedding, ReLU feed-forward). The attention reuses the encoder's kernels, with
//! one difference: the heads are 24 wide (192 / 8), not a multiple of the 32-float vector the
//! F32 matmul reads. Q and K are therefore copied head by head into a layout padded to 32
//! lanes whose padding the host zeroes, so the dot products over the padded length equal the
//! ones over 24. The score tensors pad their rows to 32 as in the encoder.
//!
//! The residual stream alternates between two buffers: a layer norm reads one and writes the
//! other, and the roles swap, so no copy follows it.

use std::sync::{Arc, Mutex};

use crate::backend::hexagon::dispatch::{self, LayerNormArgs, OpSink, TokenShape, TokenTile, View};
use crate::backend::hexagon::{
    FastRpcDriver, HexagonDevice, HexagonQueueSession, HexagonWeightDesc, LockOrRecover,
    RpcmemBuffer, align128,
};
use crate::model::audio_encoder::{HOP_LEN, LOG_MEL_EPS, N_FFT};
use crate::model::audio_encoder_hexagon::{
    HexagonAudioEncoder, alloc_settled, pad32, plan_linear, plan_vec, put_linear, put_vec,
    release_or_leak, run_on_queue, settled,
};
use crate::model::audio_mel_hexagon::{
    MelScratch, MelWeightOffsets, emit_mel, put_mel_tables, stage_samples,
};
use crate::model::audio_stem_hexagon::STEM_CHUNK_ROWS;
use crate::model::sortformer::{
    SortformerAccelerator, SortformerModel, SortformerWeights, stem_frames,
};
use crate::model::weights::MmapWeight;
use crate::session::CeraError;

/// Token-axis ops run in tiles of 64 frames, as in the encoder (the quantized matmul and the
/// elementwise kernels overflow the VTCM on a whole sequence).
const TILE: TokenTile = TokenTile::Tiles(64);

/// Lanes per head after padding: one F32 HVX vector.
const HEAD_LANES: usize = 32;

/// A linear layer's weight and bias in the weights buffer.
#[derive(Debug, Clone, Copy)]
struct LinearOffsets {
    w: HexagonWeightDesc,
    b: usize,
}

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
struct TailOffsets {
    proj: LinearOffsets,
    layers: Vec<LayerOffsets>,
    head_hidden: LinearOffsets,
    head_out: LinearOffsets,
    total_bytes: usize,
}

/// The network's widths, read from the loaded weights.
#[derive(Debug, Clone, Copy)]
struct Dims {
    /// FastConformer output width (the input of `encoder_proj`).
    enc_d: usize,
    /// Transformer width.
    d: usize,
    heads: usize,
    /// Per-head width (`d / heads`), before padding.
    dh: usize,
    /// Feed-forward inner width.
    inner: usize,
    /// Head hidden width.
    hid: usize,
    n_spk: usize,
    eps: f32,
}

fn linear(
    cur: &mut usize,
    w: &MmapWeight,
    b: &[f32],
    what: &str,
) -> Result<LinearOffsets, CeraError> {
    if b.len() != w.rows {
        return Err(CeraError::Backend(format!(
            "sortformer tail: {what} has {} biases for {} rows",
            b.len(),
            w.rows
        )));
    }
    let w = plan_linear(cur, w).map_err(|e| {
        CeraError::Backend(format!(
            "sortformer tail: {what}: {e}; the NPU reads Q8_0 or Q4_0 weights, so convert the \
             model with `--tail-outtype q8_0` (scripts/sortformer/README.md)"
        ))
    })?;
    Ok(LinearOffsets {
        w,
        b: plan_vec(cur, b.len()),
    })
}

impl TailOffsets {
    fn plan(w: &SortformerWeights, dims: &Dims) -> Result<Self, CeraError> {
        let mut cur = 0;
        let proj = linear(&mut cur, &w.proj_w, &w.proj_b, "encoder_proj")?;
        let mut layers = Vec::with_capacity(w.tf.len());
        for (i, l) in w.tf.iter().enumerate() {
            let what = |n: &str| format!("transformer layer {i} {n}");
            let q = linear(&mut cur, &l.q_w, &l.q_b, &what("q"))?;
            let k = linear(&mut cur, &l.k_w, &l.k_b, &what("k"))?;
            let v = linear(&mut cur, &l.v_w, &l.v_b, &what("v"))?;
            let o = linear(&mut cur, &l.o_w, &l.o_b, &what("o"))?;
            let up = linear(&mut cur, &l.up_w, &l.up_b, &what("up"))?;
            let down = linear(&mut cur, &l.down_w, &l.down_b, &what("down"))?;
            for (n, len) in [
                ("ln1 weight", l.ln1_w.len()),
                ("ln1 bias", l.ln1_b.len()),
                ("ln2 weight", l.ln2_w.len()),
                ("ln2 bias", l.ln2_b.len()),
            ] {
                if len != dims.d {
                    return Err(CeraError::Backend(format!(
                        "sortformer tail: {} has {len} values for width {}",
                        what(n),
                        dims.d
                    )));
                }
            }
            for (n, m, rows, cols) in [
                ("q", &l.q_w, dims.d, dims.d),
                ("k", &l.k_w, dims.d, dims.d),
                ("v", &l.v_w, dims.d, dims.d),
                ("o", &l.o_w, dims.d, dims.d),
                ("up", &l.up_w, dims.inner, dims.d),
                ("down", &l.down_w, dims.d, dims.inner),
            ] {
                if m.rows != rows || m.cols != cols {
                    return Err(CeraError::Backend(format!(
                        "sortformer tail: {} is {}x{}, expected {rows}x{cols}",
                        what(n),
                        m.rows,
                        m.cols
                    )));
                }
            }
            layers.push(LayerOffsets {
                q,
                k,
                v,
                o,
                ln1_w: plan_vec(&mut cur, l.ln1_w.len()),
                ln1_b: plan_vec(&mut cur, l.ln1_b.len()),
                up,
                down,
                ln2_w: plan_vec(&mut cur, l.ln2_w.len()),
                ln2_b: plan_vec(&mut cur, l.ln2_b.len()),
            });
        }
        let head_hidden = linear(&mut cur, &w.head_hidden_w, &w.head_hidden_b, "head hidden")?;
        let head_out = linear(&mut cur, &w.head_out_w, &w.head_out_b, "head out")?;
        Ok(Self {
            proj,
            layers,
            head_hidden,
            head_out,
            total_bytes: cur,
        })
    }
}

/// Activation scratch for sequences of up to `max_frames` frames.
#[derive(Debug, Clone, Copy)]
struct Scratch {
    /// FastConformer output, `[t, enc_d]`: the host's input.
    xin: usize,
    /// The two residual-stream buffers, `[t, d]`.
    a: usize,
    b: usize,
    q: usize,
    k: usize,
    v: usize,
    /// Head-padded Q and K, `[HEAD_LANES, t, heads]`. Host-zeroed each call.
    qp: usize,
    kp: usize,
    /// Scores and their softmax, `[t, t, heads]`.
    ac: usize,
    smc: usize,
    /// The softmax in the zero-padded layout the last contraction reads, `[tp, t, heads]`.
    sm: usize,
    /// V transposed per head, `[tp, dh, heads]`. Host-zeroed each call.
    vt: usize,
    /// attn @ V, head-interleaved `[t, d]`.
    av: usize,
    /// A sub-block's output before its residual add, `[t, d]`.
    out: usize,
    /// Feed-forward inner activation, `[t, inner]`.
    ff: usize,
    /// Head hidden activation, `[t, hid]`.
    hid: usize,
    /// Speaker activities, `[t, n_spk]`.
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
        let xin = region(seq(dims.enc_d));
        let a = region(seq(dims.d));
        let b = region(seq(dims.d));
        let q = region(seq(dims.d));
        let k = region(seq(dims.d));
        let v = region(seq(dims.d));
        let qp = region(HEAD_LANES * t * dims.heads * 4);
        let kp = region(HEAD_LANES * t * dims.heads * 4);
        let ac = region(dims.heads * t * t * 4);
        let smc = region(dims.heads * t * t * 4);
        let sm = region(dims.heads * t * tp * 4);
        let vt = region(dims.heads * dims.dh * tp * 4);
        let av = region(seq(dims.d));
        let out = region(seq(dims.d));
        let ff = region(seq(dims.inner));
        let hid = region(seq(dims.hid));
        let pred = region(seq(dims.n_spk));
        Self {
            xin,
            a,
            b,
            q,
            k,
            v,
            qp,
            kp,
            ac,
            smc,
            sm,
            vt,
            av,
            out,
            ff,
            hid,
            pred,
            total_bytes: cur,
        }
    }
}

/// How much of the tail a call runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailStage {
    /// `encoder_proj` only: `[t, d]`.
    Proj,
    /// `encoder_proj` and the first `n` Transformer layers: `[t, d]`. Clamped to the staged
    /// layers, so an overlarge `n` runs all of them.
    Layers(usize),
    /// Everything: the speaker activities, `[t, n_spk]`.
    Full,
}

/// One post-LN Transformer layer over the `[t, d]` stream at `cur` (the other buffer is
/// `other`); returns the buffer that holds the result.
#[allow(clippy::too_many_arguments)]
fn emit_layer<S: OpSink>(
    s: &mut S,
    weights: &S::Buf,
    scratch: &S::Buf,
    l: &LayerOffsets,
    so: &Scratch,
    dims: &Dims,
    t: usize,
    cur: usize,
    other: usize,
) -> Result<(usize, usize), CeraError> {
    let (d, h, dh) = (dims.d, dims.heads, dims.dh);
    let tp = pad32(t);
    let shape = TokenShape {
        dim: d,
        n_tokens: t,
    };
    for (lin, dst) in [(&l.q, so.q), (&l.k, so.k), (&l.v, so.v)] {
        dispatch::linear_m(
            s, scratch, cur, weights, lin.w, lin.b, scratch, dst, t, TILE,
        )?;
    }
    // Each head's `[dh, rows]` slice of a `[rows, d]` activation.
    let heads = |off, rows: usize| View::new(scratch, off, [dh, rows, h], [4, d * 4, dh * 4]);
    // The same slice written into the 32-lane layout (24 real lanes, the rest zero).
    let into_lanes = |off| {
        View::new(
            scratch,
            off,
            [dh, t, h],
            [4, HEAD_LANES * 4, HEAD_LANES * t * 4],
        )
    };
    let lanes = |off| {
        View::new(
            scratch,
            off,
            [HEAD_LANES, t, h],
            [4, HEAD_LANES * 4, HEAD_LANES * t * 4],
        )
    };
    dispatch::copy_view(s, heads(so.q, t), into_lanes(so.qp))?;
    dispatch::copy_view(s, heads(so.k, t), into_lanes(so.kp))?;
    // Scores ac[k, q, h] = K[k,h] . Q[q,h], softmax over keys.
    let ac = View::new(scratch, so.ac, [t, t, h], [4, t * 4, t * t * 4]);
    dispatch::matmul_f32(s, lanes(so.kp), lanes(so.qp), ac)?;
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
    dispatch::add_residual(s, scratch, cur, scratch, so.out, shape, TILE)?;
    dispatch::layer_norm(
        s,
        LayerNormArgs {
            src: scratch,
            src_offset: cur,
            dst: scratch,
            dst_offset: other,
            weights,
            w_offset: l.ln1_w,
            b_offset: l.ln1_b,
            shape,
            eps: dims.eps,
            tile: TILE,
        },
    )?;
    // Feed-forward on the normalised stream, now in `other`.
    dispatch::linear_m(
        s, scratch, other, weights, l.up.w, l.up.b, scratch, so.ff, t, TILE,
    )?;
    dispatch::relu(
        s,
        scratch,
        so.ff,
        TokenShape {
            dim: dims.inner,
            n_tokens: t,
        },
        TILE,
    )?;
    dispatch::linear_m(
        s, scratch, so.ff, weights, l.down.w, l.down.b, scratch, so.out, t, TILE,
    )?;
    dispatch::add_residual(s, scratch, other, scratch, so.out, shape, TILE)?;
    dispatch::layer_norm(
        s,
        LayerNormArgs {
            src: scratch,
            src_offset: other,
            dst: scratch,
            dst_offset: cur,
            weights,
            w_offset: l.ln2_w,
            b_offset: l.ln2_b,
            shape,
            eps: dims.eps,
            tile: TILE,
        },
    )?;
    // The result is back in `cur`, so the roles do not swap across a whole layer.
    Ok((cur, other))
}

/// Emit `stage` over `t` FastConformer output frames staged at `so.xin`. Returns the scratch
/// offset and float count of the result.
#[allow(clippy::too_many_arguments)]
fn emit_tail<S: OpSink>(
    s: &mut S,
    weights: &S::Buf,
    scratch: &S::Buf,
    o: &TailOffsets,
    so: &Scratch,
    dims: &Dims,
    t: usize,
    stage: TailStage,
    flush: &mut dyn FnMut(&mut S) -> Result<(), CeraError>,
) -> Result<(usize, usize), CeraError> {
    dispatch::linear_m(
        s, scratch, so.xin, weights, o.proj.w, o.proj.b, scratch, so.a, t, TILE,
    )?;
    let n_layers = match stage {
        TailStage::Proj => 0,
        TailStage::Layers(n) => n.min(o.layers.len()),
        TailStage::Full => o.layers.len(),
    };
    let (mut cur, mut other) = (so.a, so.b);
    for l in &o.layers[..n_layers] {
        (cur, other) = emit_layer(s, weights, scratch, l, so, dims, t, cur, other)?;
        flush(s)?;
    }
    if stage != TailStage::Full {
        return Ok((cur, t * dims.d));
    }
    // Speaker head: relu, hidden linear, relu, output linear, sigmoid.
    let d_shape = TokenShape {
        dim: dims.d,
        n_tokens: t,
    };
    dispatch::relu(s, scratch, cur, d_shape, TILE)?;
    dispatch::linear_m(
        s,
        scratch,
        cur,
        weights,
        o.head_hidden.w,
        o.head_hidden.b,
        scratch,
        so.hid,
        t,
        TILE,
    )?;
    dispatch::relu(
        s,
        scratch,
        so.hid,
        TokenShape {
            dim: dims.hid,
            n_tokens: t,
        },
        TILE,
    )?;
    dispatch::linear_m(
        s,
        scratch,
        so.hid,
        weights,
        o.head_out.w,
        o.head_out.b,
        scratch,
        so.pred,
        t,
        TILE,
    )?;
    dispatch::sigmoid(
        s,
        scratch,
        so.pred,
        TokenShape {
            dim: dims.n_spk,
            n_tokens: t,
        },
        TILE,
    )?;
    Ok((so.pred, t * dims.n_spk))
}

/// Sortformer's `encoder_proj`, Transformer and speaker head on the Hexagon NPU.
///
/// The `RpcmemBuffer`s retain the driver, so the struct keeps no `Arc` of its own.
pub struct HexagonSortformerTail {
    device: Arc<Mutex<HexagonDevice>>,
    weights_buf: RpcmemBuffer,
    offsets: TailOffsets,
    scratch: Mutex<RpcmemBuffer>,
    so: Scratch,
    dims: Dims,
    max_frames: usize,
}

// SAFETY: the rpcmem buffers are only touched while holding `device` and `scratch`, as in the
// encoder.
unsafe impl Send for HexagonSortformerTail {}
unsafe impl Sync for HexagonSortformerTail {}

impl HexagonSortformerTail {
    /// Stage the model's tail weights on the device for sequences of up to `max_frames`
    /// frames. Needs Q8_0 or Q4_0 weights (the NPU's matmul formats): an F32 or F16 GGUF
    /// is refused.
    pub fn new(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        model: &SortformerModel,
        max_frames: usize,
    ) -> Result<Self, CeraError> {
        let w = model.weights();
        let c = &w.config;
        if c.tf_heads == 0 || !c.tf_d.is_multiple_of(c.tf_heads) || !c.tf_d.is_multiple_of(32) {
            return Err(CeraError::Backend(format!(
                "sortformer tail: width {} with {} heads is not supported",
                c.tf_d, c.tf_heads
            )));
        }
        let dh = c.tf_d / c.tf_heads;
        if dh > HEAD_LANES {
            return Err(CeraError::Backend(format!(
                "sortformer tail: head width {dh} exceeds the {HEAD_LANES} padded lanes"
            )));
        }
        let dims = Dims {
            enc_d: c.n_embd,
            d: c.tf_d,
            heads: c.tf_heads,
            dh,
            inner: c.tf_inner,
            hid: w.head_hidden_w.rows,
            n_spk: c.n_spk,
            eps: c.tf_eps,
        };
        if w.head_hidden_w.cols != dims.d
            || w.head_out_w.cols != dims.hid
            || w.head_out_w.rows != dims.n_spk
            || w.proj_w.rows != dims.d
            || w.proj_w.cols != dims.enc_d
        {
            return Err(CeraError::Backend(
                "sortformer tail: encoder_proj or head shapes do not fit the network".into(),
            ));
        }
        let offsets = TailOffsets::plan(w, &dims)?;
        let so = Scratch::new(&dims, max_frames);
        let scratch = RpcmemBuffer::alloc(Arc::clone(&driver), so.total_bytes, true)?;
        let mut weights_buf = RpcmemBuffer::alloc(Arc::clone(&driver), offsets.total_bytes, true)?;

        let dst = weights_buf.as_mut_slice();
        let put = |dst: &mut [u8], o: &LinearOffsets, w: &MmapWeight, b: &[f32]| {
            put_linear(dst, o.w, w)?;
            put_vec(dst, o.b, b);
            Ok::<(), CeraError>(())
        };
        put(dst, &offsets.proj, &w.proj_w, &w.proj_b)?;
        for (o, l) in offsets.layers.iter().zip(&w.tf) {
            put(dst, &o.q, &l.q_w, &l.q_b)?;
            put(dst, &o.k, &l.k_w, &l.k_b)?;
            put(dst, &o.v, &l.v_w, &l.v_b)?;
            put(dst, &o.o, &l.o_w, &l.o_b)?;
            put(dst, &o.up, &l.up_w, &l.up_b)?;
            put(dst, &o.down, &l.down_w, &l.down_b)?;
            put_vec(dst, o.ln1_w, &l.ln1_w);
            put_vec(dst, o.ln1_b, &l.ln1_b);
            put_vec(dst, o.ln2_w, &l.ln2_w);
            put_vec(dst, o.ln2_b, &l.ln2_b);
        }
        put(
            dst,
            &offsets.head_hidden,
            &w.head_hidden_w,
            &w.head_hidden_b,
        )?;
        put(dst, &offsets.head_out, &w.head_out_w, &w.head_out_b)?;
        weights_buf.flush_cpu_cache(0, offsets.total_bytes);

        Ok(Self {
            device,
            weights_buf,
            offsets,
            scratch: Mutex::new(scratch),
            so,
            dims,
            max_frames,
        })
    }

    /// The speaker activities `[t, n_spk]` for the FastConformer's output `enc`
    /// (`[t, enc_d]`, after the last block). The same result as the CPU
    /// `encoder_proj`, Transformer and head.
    pub fn predict(&self, enc: &[f32], t: usize) -> Result<Vec<f32>, CeraError> {
        self.run(enc, t, TailStage::Full)
    }

    /// Run `stage` over `enc`; what the device probe compares stage by stage with the CPU.
    #[doc(hidden)]
    pub fn run(&self, enc: &[f32], t: usize, stage: TailStage) -> Result<Vec<f32>, CeraError> {
        let dims = &self.dims;
        if t == 0 || t > self.max_frames || enc.len() != t * dims.enc_d {
            return Err(CeraError::Backend(format!(
                "sortformer tail: {} floats for {t} frames of {} (at most {} frames)",
                enc.len(),
                dims.enc_d,
                self.max_frames
            )));
        }
        let so = self.so;
        let tp = pad32(t);
        let mut dev = self.device.lock_or_recover();
        let mut scratch = self.scratch.lock_or_recover();
        settled(dev.queue_session_mut(), || {
            let bytes = t * dims.enc_d * 4;
            let buf = scratch.as_mut_slice();
            buf[so.xin..so.xin + bytes].copy_from_slice(bytemuck::cast_slice(enc));
            // The lane padding and the padded softmax and V^T layouts must read as zero.
            let zero = [
                (so.qp, HEAD_LANES * t * dims.heads * 4),
                (so.kp, HEAD_LANES * t * dims.heads * 4),
                (so.sm, dims.heads * t * tp * 4),
                (so.vt, dims.heads * dims.dh * tp * 4),
            ];
            for (off, len) in zero {
                buf[off..off + len].fill(0);
            }
            scratch.flush_cpu_cache(so.xin, bytes);
            for (off, len) in zero {
                scratch.flush_cpu_cache(off, len);
            }
        })
        .map_err(|e| CeraError::Backend(format!("sortformer tail: {e}")))?;

        let mut out = (0, 0);
        run_on_queue("sortformer tail", dev.queue_session_mut(), |session| {
            out = emit_tail(
                session,
                &self.weights_buf,
                &scratch,
                &self.offsets,
                &so,
                dims,
                t,
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

impl Drop for HexagonSortformerTail {
    fn drop(&mut self) {
        let mut dev = self.device.lock_or_recover();
        let session: &mut HexagonQueueSession = dev.queue_session_mut();
        session.release_dsp_references([&self.weights_buf, &*self.scratch.lock_or_recover()]);
    }
}

/// Frames per DSP call of the log-mel front end: its per-call buffer holds the frames, the two
/// DFT planes and the energies (about 11 KB per frame), so a longer clip goes in batches. The
/// checkpoint's default chunk (188 encoder frames, 1504 mel frames) fits in one.
const MEL_CALL_FRAMES: usize = 1536;

/// Sortformer's log-mel front end on the NPU: the windowed DFT, the power and the filterbank
/// (the model's own window and filterbank, from its GGUF), through the same ops as LFM2-Audio's
/// front end, then `ln(energy + 2^-24)` on the DSP too: the host only copies the result.
pub struct HexagonSortformerMel {
    driver: Arc<FastRpcDriver>,
    device: Arc<Mutex<HexagonDevice>>,
    weights_buf: RpcmemBuffer,
    offsets: MelWeightOffsets,
    n_mel: usize,
}

impl HexagonSortformerMel {
    /// Stage `window` (`N_FFT` long) and `filters` (`[n_mel, N_FFT_BINS]`) on the device.
    pub fn new(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        window: &[f32],
        filters: &[f32],
        n_mel: usize,
    ) -> Result<Self, CeraError> {
        if window.len() != N_FFT || n_mel == 0 || filters.len() != n_mel * (N_FFT / 2 + 1) {
            return Err(CeraError::Backend(format!(
                "sortformer mel: window of {}, filterbank of {} for {n_mel} mel bins",
                window.len(),
                filters.len()
            )));
        }
        let mut cur = 0;
        let offsets = MelWeightOffsets::plan(&mut cur, n_mel);
        let mut weights_buf = RpcmemBuffer::alloc(Arc::clone(&driver), cur, true)?;
        put_mel_tables(weights_buf.as_mut_slice(), &offsets, window, filters, n_mel);
        weights_buf.flush_cpu_cache(0, cur);
        Ok(Self {
            driver,
            device,
            weights_buf,
            offsets,
            n_mel,
        })
    }

    /// Log-mel `[n_frames, n_mel]` (`ln(energy + 2^-24)`) for `n_frames` frames over pre-emphasised,
    /// centre-padded `samples` (frame `f` starts at `f * HOP_LEN`).
    pub fn log_mel(&self, samples: &[f32], n_frames: usize) -> Result<Vec<f32>, CeraError> {
        if n_frames == 0 || samples.len() != (n_frames - 1) * HOP_LEN + N_FFT {
            return Err(CeraError::Backend(format!(
                "sortformer mel: {} samples for {n_frames} frames",
                samples.len()
            )));
        }
        let mut out = Vec::with_capacity(n_frames * self.n_mel);
        let mut done = 0;
        while done < n_frames {
            let k = (n_frames - done).min(MEL_CALL_FRAMES);
            let lo = done * HOP_LEN;
            out.extend(self.batch(&samples[lo..lo + (k - 1) * HOP_LEN + N_FFT], k)?);
            done += k;
        }
        Ok(out)
    }

    fn batch(&self, samples: &[f32], k: usize) -> Result<Vec<f32>, CeraError> {
        use crate::backend::hexagon::dispatch::{log_inplace, scale_offset};
        let so = MelScratch::new(samples.len(), k, self.n_mel);
        let mut dev = self.device.lock_or_recover();
        let mut buf = alloc_settled(dev.queue_session_mut(), &self.driver, so.total_bytes)?;
        stage_samples(buf.as_mut_slice(), &so, samples);
        buf.flush_cpu_cache(0, so.total_bytes);
        let run = run_on_queue("sortformer mel", dev.queue_session_mut(), |session| {
            emit_mel(
                session,
                &self.weights_buf,
                &buf,
                &self.offsets,
                &so,
                k,
                self.n_mel,
            )?;
            // ln(energy + 2^-24), on the DSP like the rest of the front end.
            let shape = TokenShape {
                dim: self.n_mel,
                n_tokens: k,
            };
            scale_offset(
                session,
                &buf,
                so.mel,
                shape,
                TokenTile::Whole,
                (1.0, LOG_MEL_EPS),
            )?;
            log_inplace(session, &buf, so.mel, shape, TokenTile::Whole)
        });
        let energies = run.map(|()| {
            buf.invalidate_cpu_cache(so.mel, k * self.n_mel * 4);
            let floats: &[f32] = bytemuck::cast_slice(buf.as_slice());
            floats[so.mel / 4..so.mel / 4 + k * self.n_mel].to_vec()
        });
        release_or_leak(dev.queue_session_mut(), buf);
        energies
    }
}

impl Drop for HexagonSortformerMel {
    fn drop(&mut self) {
        let mut dev = self.device.lock_or_recover();
        dev.queue_session_mut()
            .release_dsp_references([&self.weights_buf]);
    }
}

/// The whole diarizer network on the Hexagon NPU: the log-mel front end ([`HexagonSortformerMel`]),
/// the conv stem and the FastConformer blocks ([`HexagonAudioEncoder`]) and the tail
/// ([`HexagonSortformerTail`]). Plug it into a model with
/// [`SortformerModel::set_accelerator`], or use [`try_hexagon_sortformer`].
///
/// Only the x-scale between the stem and the blocks runs on the host (one multiply per value).
pub struct HexagonSortformer {
    encoder: HexagonAudioEncoder,
    tail: HexagonSortformerTail,
    mel: HexagonSortformerMel,
    /// Encoder frames the staging covers; longer inputs are declined (the CPU takes them).
    max_frames: usize,
    n_blocks: usize,
    scale: f32,
    n_embd: usize,
}

impl HexagonSortformer {
    /// Stage the model on the device for steps of up to `max_frames` encoder frames (use
    /// [`window_frames`](crate::model::sortformer::StreamingParams::window_frames)). Needs the
    /// Q8_0-tail GGUF, see the module docs.
    pub fn new(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        model: &SortformerModel,
        max_frames: usize,
    ) -> Result<Self, CeraError> {
        let parts = model.encoder_parts();
        let encoder = HexagonAudioEncoder::from_parts(
            Arc::clone(&driver),
            Arc::clone(&device),
            &parts,
            max_frames,
            false,
        )?;
        let tail = HexagonSortformerTail::new(
            Arc::clone(&driver),
            Arc::clone(&device),
            model,
            max_frames,
        )?;
        let w = model.weights();
        let mel =
            HexagonSortformerMel::new(driver, device, &w.window, &w.mel_fb, w.config.n_mel_bins)?;
        Ok(Self {
            encoder,
            tail,
            mel,
            max_frames,
            n_blocks: parts.layers.len(),
            scale: model.encoder_input_scale(),
            n_embd: parts.config.n_embd,
        })
    }
}

impl SortformerAccelerator for HexagonSortformer {
    fn pre_encode(&self, mel: &[f32], n_frames: usize) -> anyhow::Result<Option<Vec<f32>>> {
        if n_frames == 0 || stem_frames(n_frames) > self.max_frames {
            return Ok(None);
        }
        Ok(Some(self.encoder.stem_output(
            mel,
            n_frames,
            STEM_CHUNK_ROWS,
        )?))
    }

    fn predict(&self, emb: &[f32], t: usize) -> anyhow::Result<Option<Vec<f32>>> {
        if t == 0 || t > self.max_frames {
            return Ok(None);
        }
        anyhow::ensure!(
            emb.len() == t * self.n_embd,
            "predict: {} values for {t} frames of {}",
            emb.len(),
            self.n_embd
        );
        let x: Vec<f32> = emb.iter().map(|v| v * self.scale).collect();
        let enc = self.encoder.run_blocks(self.n_blocks, &x, t)?;
        Ok(Some(self.tail.predict(&enc, t)?))
    }

    fn log_mel(&self, samples: &[f32], n_frames: usize) -> anyhow::Result<Option<Vec<f32>>> {
        if n_frames == 0 {
            return Ok(None);
        }
        Ok(Some(self.mel.log_mel(samples, n_frames)?))
    }
}

/// Stage `model` on the NPU for steps of up to `max_frames` encoder frames and make it the
/// model's accelerator. `None` (with the reason logged) when there is no usable NPU, the
/// weights cannot be staged (for example a GGUF whose tail is not Q8_0), or the model already
/// has an accelerator (a repeat call stages nothing and keeps the first); in the first two
/// cases the model keeps running on the CPU.
pub fn try_hexagon_sortformer(
    model: &SortformerModel,
    max_frames: usize,
) -> Option<Arc<HexagonSortformer>> {
    if model.has_accelerator() {
        tracing::info!("HexagonSortformer: the model already has an accelerator; keeping it");
        return None;
    }
    let context = crate::backend::hexagon::HexagonContext::new()
        .inspect_err(|e| {
            crate::backend::hexagon::log_context_unavailable("HexagonSortformer", e);
        })
        .ok()?;
    let arch_override = std::env::var("CERA_HEXAGON_ARCH")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .and_then(crate::backend::hexagon::HexagonArch::from_u32);
    let dev = match crate::backend::hexagon::probe_device(context.driver(), arch_override) {
        Ok(d) => d,
        Err(e) => {
            tracing::info!("HexagonSortformer: DSP device unavailable ({e}), using the CPU");
            return None;
        }
    };
    let device = Arc::new(Mutex::new(dev));
    let staged =
        match HexagonSortformer::new(Arc::clone(context.driver()), device, model, max_frames) {
            Ok(s) => Arc::new(s),
            Err(e) => {
                crate::backend::hexagon::hexagon_error!(
                    "failed to stage Sortformer on the NPU: {e}"
                );
                return None;
            }
        };
    if let Err(e) = model.set_accelerator(staged.clone()) {
        crate::backend::hexagon::hexagon_warn!("HexagonSortformer: {e:#}");
        return None;
    }
    tracing::info!("sortformer: using the Hexagon NPU ({max_frames} encoder frames)");
    Some(staged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::hexagon::dispatch::testing::RecordingSink;
    use crate::backend::hexagon::{HexagonWeightFormat, HtpOpCode};

    fn desc(rows: usize, cols: usize) -> HexagonWeightDesc {
        HexagonWeightDesc {
            offset: 4096,
            size_bytes: 12345,
            format: HexagonWeightFormat::RepackedQ8_0,
            rows,
            cols,
        }
    }

    fn lin(rows: usize, cols: usize) -> LinearOffsets {
        LinearOffsets {
            w: desc(rows, cols),
            b: 128,
        }
    }

    /// A weight the NPU cannot read (here dense F32, as an F16 or F32 GGUF would give) is refused
    /// with the tensor's name and the way to convert the model so it works.
    #[test]
    fn a_tail_weight_the_npu_cannot_read_names_the_fix() {
        let dense = MmapWeight::from_owned_f32(vec![0.0; 4 * 32], 4, 32);
        let mut cur = 0;
        let err = linear(&mut cur, &dense, &[0.0; 4], "transformer layer 3 q")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("transformer layer 3 q") && err.contains("--tail-outtype q8_0"),
            "{err}"
        );
    }

    /// Sortformer's real widths, `layers` Transformer layers.
    fn fixture(layers: usize) -> (TailOffsets, Dims) {
        let dims = Dims {
            enc_d: 512,
            d: 192,
            heads: 8,
            dh: 24,
            inner: 768,
            hid: 192,
            n_spk: 4,
            eps: 1e-5,
        };
        let layer = LayerOffsets {
            q: lin(192, 192),
            k: lin(192, 192),
            v: lin(192, 192),
            o: lin(192, 192),
            ln1_w: 0,
            ln1_b: 0,
            up: lin(768, 192),
            down: lin(192, 768),
            ln2_w: 0,
            ln2_b: 0,
        };
        let o = TailOffsets {
            proj: lin(192, 512),
            layers: vec![layer; layers],
            head_hidden: lin(192, 192),
            head_out: lin(4, 192),
            total_bytes: 0,
        };
        (o, dims)
    }

    fn emit(stage: TailStage, layers: usize, t: usize) -> (RecordingSink, (usize, usize), Scratch) {
        let (o, dims) = fixture(layers);
        let so = Scratch::new(&dims, 64);
        let mut s = RecordingSink::default();
        let out = emit_tail(
            &mut s,
            &"w",
            &"s",
            &o,
            &so,
            &dims,
            t,
            stage,
            &mut |_: &mut RecordingSink| Ok(()),
        )
        .unwrap();
        (s, out, so)
    }

    /// Q and K reach the score matmul in the 32-lane layout (24 real lanes), the softmax runs
    /// once per layer without a mask, and the head ends in a sigmoid over the 4 speakers. The
    /// full opcode sequence is pinned: op counts alone pass a within-layer swap, and at `t=10`
    /// the contraction-key pin collapses onto the lane pins (`pad32(10) == HEAD_LANES`), so
    /// this emits at `t=40` where the 64-wide contraction separates from the 32-wide lanes.
    #[test]
    fn a_layer_pads_the_heads_and_the_head_ends_in_a_sigmoid() {
        use HtpOpCode::*;
        let (s, out, so) = emit(TailStage::Full, 2, 40);
        let proj = [MulMat, Add];
        let qkv = [MulMat, Add, MulMat, Add, MulMat, Add];
        let body = [
            Cpy, Cpy, MulMat, Softmax, Cpy, Cpy, MulMat, MulMat, Add, Add, Norm, Mul, Add, MulMat,
            Add, UnaryRelu, MulMat, Add, Add, Norm, Mul, Add,
        ];
        let head = [UnaryRelu, MulMat, Add, UnaryRelu, MulMat, Add, UnarySigmoid];
        let expect: Vec<u32> = proj
            .iter()
            .chain(&qkv)
            .chain(&body)
            .chain(&qkv)
            .chain(&body)
            .chain(&head)
            .map(|o| *o as u32)
            .collect();
        assert_eq!(s.opcodes(), expect);
        // Every F32 score matmul reads 32-lane operands.
        let f32_mm: Vec<usize> = (0..s.ops.len())
            .filter(|&i| {
                s.ops[i].opcode == HtpOpCode::MulMat as u32 && s.ops[i].src.len() == 2 && {
                    let a = s.src(i, 0);
                    a.dtype == crate::backend::hexagon::HtpDataType::F32 as u32
                }
            })
            .collect();
        assert_eq!(f32_mm.len(), 4, "two F32 matmuls per layer");
        for pair in f32_mm.chunks(2) {
            assert_eq!(s.src(pair[0], 0).ne[0], HEAD_LANES as u32, "K lanes");
            assert_eq!(s.src(pair[0], 1).ne[0], HEAD_LANES as u32, "Q lanes");
            // attn @ V contracts over the padded key count (64 here, not the 32 lanes).
            assert_eq!(s.src(pair[1], 0).ne[0], pad32(40) as u32);
        }
        assert_eq!(out, (so.pred, 40 * 4));
    }

    /// The layer norm's output buffer becomes the next sub-block's input, so a whole layer
    /// ends back in the stream buffer it started in (no copy op follows a norm).
    #[test]
    fn the_residual_stream_ends_each_layer_where_it_started() {
        let (_, out_proj, so) = emit(TailStage::Proj, 3, 7);
        assert_eq!(out_proj, (so.a, 7 * 192));
        let (_, out_l, so) = emit(TailStage::Layers(3), 3, 7);
        assert_eq!(out_l, (so.a, 7 * 192));
        // Layers(n) never runs past the staged layers.
        let (s_all, _, _) = emit(TailStage::Layers(99), 3, 7);
        let (s_three, _, _) = emit(TailStage::Layers(3), 3, 7);
        assert_eq!(s_all.opcodes(), s_three.opcodes());
    }

    /// Collects the `message` field of every `INFO` event. Same shape as the warn capture in
    /// `transformer.rs` tests (a sibling unit test this module cannot import).
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

    /// A repeat staging attempt stages nothing and keeps the first accelerator. The early-out
    /// fires before any NPU interaction, so the contract (plus its log line) holds on host.
    #[cfg(feature = "mmap")]
    #[test]
    fn try_hexagon_sortformer_keeps_the_first_accelerator() {
        use crate::model::sortformer::{SortformerAccelerator, SortformerModel};
        use tracing_subscriber::layer::SubscriberExt;

        struct Decline;
        impl SortformerAccelerator for Decline {
            fn pre_encode(
                &self,
                _mel: &[f32],
                _n_frames: usize,
            ) -> anyhow::Result<Option<Vec<f32>>> {
                Ok(None)
            }
            fn predict(&self, _emb: &[f32], _t: usize) -> anyhow::Result<Option<Vec<f32>>> {
                Ok(None)
            }
        }

        let path = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".leap/models/sortformer/sortformer-4spk-v2.1-q8_0.gguf");
        if !crate::model::transformer::require_model_or_skip(&path) {
            return;
        }
        let model = SortformerModel::from_file(&path).unwrap();
        assert!(!model.has_accelerator());
        model.set_accelerator(std::sync::Arc::new(Decline)).unwrap();
        assert!(model.has_accelerator());

        let capture = InfoCapture::default();
        let sub = tracing_subscriber::registry().with(capture.clone());
        let out = tracing::subscriber::with_default(sub, || try_hexagon_sortformer(&model, 64));
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
}
