//! Silero VAD v5's 16 kHz window on the Hexagon NPU.
//!
//! The VAD runs for every 32 ms of audio, speech or not, so it is the one model an always-on
//! background service keeps busy all day. It is tiny (about 0.7 MFLOP a window by op count) but
//! the CPU still pays on the order of 0.3 ms for each (S25 Ultra, release build; re-measure on
//! your SoC with `cera/examples/hexagon_vad_probe.rs`), and Android demotes background CPU work
//! and not the NPU.
//!
//! A window is the same graph every time (fixed shapes, fixed buffers), so it is built once,
//! serialized, and replayed: per window the host writes the 640 input samples and the LSTM state,
//! submits the batch and sleeps until the DSP answers. Building 49 ops each window would cost the
//! host as much as the CPU forward pass it replaces.
//!
//! ```text
//! 640 padded samples
//! -> 4 frames of 256 (hop 128)      framing copy
//! -> STFT: re, im = basis . frame   two F32 matmuls (129 bins, padded to 160)
//! -> magnitude = sqrt(re^2 + im^2)
//! -> 4 x (conv k=3 + ReLU)          three tap matmuls each but the last (one centre
//!                                    tap: the outer taps read only the zero border),
//!                                    strides 1, 2, 2, 1
//! -> LSTM cell (128)                two matmuls, sigmoid and tanh gates
//! -> ReLU -> Linear(128 -> 1) -> sigmoid
//! ```
//!
//! A convolution over `len` positions with padding 1 is `W0 . X[t-1] + W1 . X[t] + W2 . X[t+1]`:
//! the input sits in a buffer with a zero row on each side, each tap is a matmul over a (strided)
//! view of it, and the three results are added. The last layer sees one position, so only the
//! centre tap is non-zero.

use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};

use crate::backend::hexagon::dispatch::{self, OpSink, TokenShape, TokenTile, View};
use crate::backend::hexagon::{
    FastRpcDriver, HexagonContext, HexagonDevice, LockOrRecover, RpcmemBuffer, StagedBatch,
};
use crate::model::audio_encoder_hexagon::{plan_vec, put_vec};
use crate::session::CeraError;
use crate::vad::{SileroVad, VadAccelerator, VadStep, VadWeights};

/// Samples per STFT frame and the hop between frames.
const FRAME: usize = 256;
const HOP: usize = 128;
/// STFT frames in a 640-sample window.
const FRAMES: usize = 4;
/// Frequency bins, padded to 160: the F32 matmul reads whole 32-float vectors.
const BINS: usize = 129;
const BINS_PAD: usize = 160;
/// LSTM width.
const HID: usize = 128;
/// Output rows of the head matmul (one is real), padded to a vector.
const HEAD_ROWS: usize = 32;

const TILE: TokenTile = TokenTile::Whole;

/// One convolution layer's weights: `out x in_pad` per tap (padded input channels), and the bias.
#[derive(Debug, Clone, Copy)]
struct ConvOffsets {
    taps: [usize; 3],
    bias: usize,
    out: usize,
    in_pad: usize,
}

#[derive(Debug, Clone, Copy)]
struct WeightOffsets {
    basis_re: usize,
    basis_im: usize,
    conv: [ConvOffsets; 4],
    w_ih: usize,
    w_hh: usize,
    lstm_bias: usize,
    head_w: usize,
    head_b: usize,
    total_bytes: usize,
}

/// The conv layers as `(out channels, real in channels, padded in channels)`.
const CONV_SHAPES: [(usize, usize, usize); 4] =
    [(128, 129, 160), (64, 128, 128), (64, 64, 64), (128, 64, 64)];

impl WeightOffsets {
    fn plan() -> Self {
        let mut cur = 0;
        let basis_re = plan_vec(&mut cur, BINS_PAD * FRAME);
        let basis_im = plan_vec(&mut cur, BINS_PAD * FRAME);
        let mut conv = [ConvOffsets {
            taps: [0; 3],
            bias: 0,
            out: 0,
            in_pad: 0,
        }; 4];
        for (c, &(out, _, in_pad)) in conv.iter_mut().zip(&CONV_SHAPES) {
            *c = ConvOffsets {
                taps: [
                    plan_vec(&mut cur, out * in_pad),
                    plan_vec(&mut cur, out * in_pad),
                    plan_vec(&mut cur, out * in_pad),
                ],
                bias: plan_vec(&mut cur, out),
                out,
                in_pad,
            };
        }
        Self {
            basis_re,
            basis_im,
            conv,
            w_ih: plan_vec(&mut cur, 4 * HID * HID),
            w_hh: plan_vec(&mut cur, 4 * HID * HID),
            lstm_bias: plan_vec(&mut cur, 4 * HID),
            head_w: plan_vec(&mut cur, HEAD_ROWS * HID),
            head_b: plan_vec(&mut cur, HEAD_ROWS),
            total_bytes: cur,
        }
    }
}

/// Tap `k` of a `[out, in_real, 3]` convolution weight as an `[out, in_pad]` matrix.
fn conv_tap(w: &[f32], out: usize, in_real: usize, in_pad: usize, k: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; out * in_pad];
    for o in 0..out {
        for i in 0..in_real {
            m[o * in_pad + i] = w[(o * in_real + i) * 3 + k];
        }
    }
    m
}

/// Rows `lo..hi` of the `[2 * 129, 256]` STFT basis as a `[160, 256]` matrix (zero bins padded).
fn basis_rows(basis: &[f32], lo: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; BINS_PAD * FRAME];
    m[..BINS * FRAME].copy_from_slice(&basis[lo * FRAME..(lo + BINS) * FRAME]);
    m
}

/// Write every constant matrix of the 16 kHz network to its planned place.
fn put_weights(dst: &mut [u8], o: &WeightOffsets, w: &VadWeights) {
    let basis = w.stft_16k_basis.as_f32_slice();
    put_vec(dst, o.basis_re, &basis_rows(basis, 0));
    put_vec(dst, o.basis_im, &basis_rows(basis, BINS));
    let layers = [
        (
            w.encoder_16k_0_w.as_f32_slice(),
            w.encoder_16k_0_b.as_f32_slice(),
        ),
        (
            w.encoder_16k_1_w.as_f32_slice(),
            w.encoder_16k_1_b.as_f32_slice(),
        ),
        (
            w.encoder_16k_2_w.as_f32_slice(),
            w.encoder_16k_2_b.as_f32_slice(),
        ),
        (
            w.encoder_16k_3_w.as_f32_slice(),
            w.encoder_16k_3_b.as_f32_slice(),
        ),
    ];
    for ((c, &(out, in_real, in_pad)), (wt, bias)) in o.conv.iter().zip(&CONV_SHAPES).zip(layers) {
        for (k, &off) in c.taps.iter().enumerate() {
            put_vec(dst, off, &conv_tap(wt, out, in_real, in_pad, k));
        }
        put_vec(dst, c.bias, bias);
    }
    put_vec(dst, o.w_ih, w.decoder_16k_rnn_w_ih.as_f32_slice());
    put_vec(dst, o.w_hh, w.decoder_16k_rnn_w_hh.as_f32_slice());
    let bias: Vec<f32> = w
        .decoder_16k_rnn_b_ih
        .as_f32_slice()
        .iter()
        .zip(w.decoder_16k_rnn_b_hh.as_f32_slice())
        .map(|(a, b)| a + b)
        .collect();
    put_vec(dst, o.lstm_bias, &bias);
    let mut head = vec![0.0f32; HEAD_ROWS * HID];
    head[..HID].copy_from_slice(w.decoder_16k_head_w.as_f32_slice());
    put_vec(dst, o.head_w, &head);
    let mut head_b = vec![0.0f32; HEAD_ROWS];
    head_b[0] = w.decoder_16k_head_b.as_f32_slice()[0];
    put_vec(dst, o.head_b, &head_b);
}

/// Activation scratch of one window, in bytes.
#[derive(Debug, Clone, Copy)]
struct Scratch {
    /// The 640 padded input samples.
    samples: usize,
    /// The STFT frames, `[4, 256]`.
    frames: usize,
    /// Zero-bordered layer inputs: the magnitude `[6, 160]`, conv 0's output `[6, 128]` and
    /// conv 1's `[4, 64]`. The border rows are zeroed once and never written.
    pad0: usize,
    pad1: usize,
    pad2: usize,
    /// Per-tap results added into the layer output.
    tmp_a: usize,
    tmp_b: usize,
    enc2: usize,
    enc3: usize,
    /// LSTM state in, written by the host each window.
    h_in: usize,
    c_in: usize,
    /// The gates (`i`, `f`, `g`, `o`), then `c'` in the `f` slot and `h'` in the `o` slot.
    gates: usize,
    gates2: usize,
    t128: usize,
    /// The head's output: the probability is element 0.
    head: usize,
    total_bytes: usize,
}

impl Scratch {
    fn plan() -> Self {
        let mut cur = 0;
        let samples = plan_vec(&mut cur, 640);
        let frames = plan_vec(&mut cur, FRAMES * FRAME);
        let pad0 = plan_vec(&mut cur, 6 * BINS_PAD);
        let pad1 = plan_vec(&mut cur, 6 * 128);
        let pad2 = plan_vec(&mut cur, 4 * 64);
        let tmp_a = plan_vec(&mut cur, FRAMES * BINS_PAD);
        let tmp_b = plan_vec(&mut cur, FRAMES * BINS_PAD);
        let enc2 = plan_vec(&mut cur, 64);
        let enc3 = plan_vec(&mut cur, 128);
        let h_in = plan_vec(&mut cur, HID);
        let c_in = plan_vec(&mut cur, HID);
        let gates = plan_vec(&mut cur, 4 * HID);
        let gates2 = plan_vec(&mut cur, 4 * HID);
        let t128 = plan_vec(&mut cur, HID);
        let head = plan_vec(&mut cur, HEAD_ROWS);
        Self {
            samples,
            frames,
            pad0,
            pad1,
            pad2,
            tmp_a,
            tmp_b,
            enc2,
            enc3,
            h_in,
            c_in,
            gates,
            gates2,
            t128,
            head,
            total_bytes: cur,
        }
    }
}

/// One convolution layer over `n_out` output positions: three tap matmuls over views of `src`
/// (a zero-bordered `[rows, in_pad]` buffer; tap `k` starts `k` rows in and steps `stride` rows),
/// summed into the first tap's destination `dst`, then the bias and ReLU.
#[allow(clippy::too_many_arguments)]
fn emit_conv<S: OpSink>(
    s: &mut S,
    w: &S::Buf,
    b: &S::Buf,
    layer: &ConvOffsets,
    src: usize,
    stride: usize,
    n_out: usize,
    dst: usize,
    so: &Scratch,
) -> Result<(), CeraError> {
    let (out, in_pad) = (layer.out, layer.in_pad);
    let row = in_pad * 4;
    let weights = |off: usize| View::new(w, off, [in_pad, out, 1], [4, row, out * row]);
    let rows_out = |off: usize| View::new(b, off, [out, n_out, 1], [4, out * 4, n_out * out * 4]);
    let tmps = [dst, so.tmp_a, so.tmp_b];
    for (k, &to) in tmps.iter().enumerate() {
        let x = View::new(
            b,
            src + k * row,
            [in_pad, n_out, 1],
            [4, stride * row, n_out * stride * row],
        );
        dispatch::matmul_f32(s, weights(layer.taps[k]), x, rows_out(to))?;
    }
    let shape = TokenShape {
        dim: out,
        n_tokens: n_out,
    };
    dispatch::add_residual(s, b, dst, b, so.tmp_a, shape, TILE)?;
    dispatch::add_residual(s, b, dst, b, so.tmp_b, shape, TILE)?;
    dispatch::add_row_bcast(s, b, dst, w, layer.bias, shape, TILE)?;
    dispatch::relu(s, b, dst, shape, TILE)
}

/// Emit the whole window: from the samples at `so.samples` and the state at `so.h_in` / `so.c_in`
/// to the probability at `so.head` and the new state in `so.gates` (see [`Scratch`]).
fn emit_window<S: OpSink>(
    s: &mut S,
    w: &S::Buf,
    b: &S::Buf,
    wo: &WeightOffsets,
    so: &Scratch,
) -> Result<(), CeraError> {
    let rows = |off: usize, dim: usize, n: usize| {
        View::new(b, off, [dim, n, 1], [4, dim * 4, n * dim * 4])
    };
    let basis = |off: usize| {
        View::new(
            w,
            off,
            [FRAME, BINS_PAD, 1],
            [4, FRAME * 4, BINS_PAD * FRAME * 4],
        )
    };
    let wmat = |off: usize, k: usize, n: usize| View::new(w, off, [k, n, 1], [4, k * 4, n * k * 4]);

    // STFT magnitude into rows 1..=4 of the zero-bordered pad0.
    dispatch::copy_view(
        s,
        View::new(
            b,
            so.samples,
            [FRAME, FRAMES, 1],
            [4, HOP * 4, FRAMES * HOP * 4],
        ),
        rows(so.frames, FRAME, FRAMES),
    )?;
    let mag = so.pad0 + BINS_PAD * 4;
    dispatch::matmul_f32(
        s,
        basis(wo.basis_re),
        rows(so.frames, FRAME, FRAMES),
        rows(mag, BINS_PAD, FRAMES),
    )?;
    dispatch::matmul_f32(
        s,
        basis(wo.basis_im),
        rows(so.frames, FRAME, FRAMES),
        rows(so.tmp_a, BINS_PAD, FRAMES),
    )?;
    let bins = TokenShape {
        dim: BINS_PAD,
        n_tokens: FRAMES,
    };
    dispatch::mul_inplace(s, b, mag, b, mag, bins, TILE)?;
    dispatch::mul_inplace(s, b, so.tmp_a, b, so.tmp_a, bins, TILE)?;
    dispatch::add_residual(s, b, mag, b, so.tmp_a, bins, TILE)?;
    dispatch::sqrt(s, b, mag, bins, TILE)?;

    // Encoder: 4 positions -> 4 -> 2 -> 1 -> 1.
    emit_conv(s, w, b, &wo.conv[0], so.pad0, 1, 4, so.pad1 + 128 * 4, so)?;
    emit_conv(s, w, b, &wo.conv[1], so.pad1, 2, 2, so.pad2 + 64 * 4, so)?;
    emit_conv(s, w, b, &wo.conv[2], so.pad2, 2, 1, so.enc2, so)?;
    // One position left: the outer taps read the zero border, so only the centre tap counts.
    let c3 = &wo.conv[3];
    dispatch::matmul_f32(
        s,
        wmat(c3.taps[1], 64, 128),
        rows(so.enc2, 64, 1),
        rows(so.enc3, 128, 1),
    )?;
    let enc3 = TokenShape {
        dim: 128,
        n_tokens: 1,
    };
    dispatch::add_row_bcast(s, b, so.enc3, w, c3.bias, enc3, TILE)?;
    dispatch::relu(s, b, so.enc3, enc3, TILE)?;

    // LSTM cell: gates = W_ih x + W_hh h + (b_ih + b_hh), laid out i, f, g, o.
    dispatch::matmul_f32(
        s,
        wmat(wo.w_ih, HID, 4 * HID),
        rows(so.enc3, HID, 1),
        rows(so.gates, 4 * HID, 1),
    )?;
    dispatch::matmul_f32(
        s,
        wmat(wo.w_hh, HID, 4 * HID),
        rows(so.h_in, HID, 1),
        rows(so.gates2, 4 * HID, 1),
    )?;
    let all = TokenShape {
        dim: 4 * HID,
        n_tokens: 1,
    };
    let one = TokenShape {
        dim: HID,
        n_tokens: 1,
    };
    dispatch::add_residual(s, b, so.gates, b, so.gates2, all, TILE)?;
    dispatch::add_row_bcast(s, b, so.gates, w, wo.lstm_bias, all, TILE)?;
    let (gi, gf, gg, go) = (
        so.gates,
        so.gates + HID * 4,
        so.gates + 2 * HID * 4,
        so.gates + 3 * HID * 4,
    );
    dispatch::sigmoid(
        s,
        b,
        gi,
        TokenShape {
            dim: 2 * HID,
            n_tokens: 1,
        },
        TILE,
    )?;
    dispatch::tanh(s, b, gg, one, TILE)?;
    dispatch::sigmoid(s, b, go, one, TILE)?;
    // c' = f * c + i * g, left in the f slot; h' = o * tanh(c'), left in the o slot.
    dispatch::mul_inplace(s, b, gf, b, so.c_in, one, TILE)?;
    dispatch::mul_inplace(s, b, gi, b, gg, one, TILE)?;
    dispatch::add_residual(s, b, gf, b, gi, one, TILE)?;
    dispatch::copy_view(s, rows(gf, HID, 1), rows(so.t128, HID, 1))?;
    dispatch::tanh(s, b, so.t128, one, TILE)?;
    dispatch::mul_inplace(s, b, go, b, so.t128, one, TILE)?;

    // Head: relu(h') -> Linear(128 -> 1) -> sigmoid (32 rows, one real).
    dispatch::copy_view(s, rows(go, HID, 1), rows(so.t128, HID, 1))?;
    dispatch::relu(s, b, so.t128, one, TILE)?;
    dispatch::matmul_f32(
        s,
        wmat(wo.head_w, HID, HEAD_ROWS),
        rows(so.t128, HID, 1),
        rows(so.head, HEAD_ROWS, 1),
    )?;
    let head = TokenShape {
        dim: HEAD_ROWS,
        n_tokens: 1,
    };
    dispatch::add_row_bcast(s, b, so.head, w, wo.head_b, head, TILE)?;
    dispatch::sigmoid(s, b, so.head, head, TILE)
}

struct Inner {
    scratch: RpcmemBuffer,
    /// The serialized window batch, built from the first window.
    staged: Option<StagedBatch>,
}

/// The 16 kHz Silero VAD window on the Hexagon NPU. See the module docs.
pub struct HexagonVad {
    device: Arc<Mutex<HexagonDevice>>,
    weights_buf: RpcmemBuffer,
    wo: WeightOffsets,
    so: Scratch,
    inner: Mutex<Inner>,
}

// SAFETY: the rpcmem buffers are only touched while holding `device` and `inner`.
unsafe impl Send for HexagonVad {}
unsafe impl Sync for HexagonVad {}

impl HexagonVad {
    /// Stage the network on `device`, whose queue session then sleeps on DSP waits (pass a device
    /// nothing else shares).
    fn new(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        weights: &VadWeights,
    ) -> Result<Self, CeraError> {
        let wo = WeightOffsets::plan();
        let so = Scratch::plan();
        let mut weights_buf = RpcmemBuffer::alloc(Arc::clone(&driver), wo.total_bytes, true)?;
        put_weights(weights_buf.as_mut_slice(), &wo, weights);
        weights_buf.flush_cpu_cache(0, wo.total_bytes);
        // Fresh rpcmem is not guaranteed zero: the border rows of the layer inputs must be.
        let mut scratch = RpcmemBuffer::alloc(driver, so.total_bytes, true)?;
        scratch.as_mut_slice().fill(0);
        scratch.flush_cpu_cache(0, so.total_bytes);
        device
            .lock_or_recover()
            .queue_session_mut()
            .set_blocking_wait(true);
        Ok(Self {
            device,
            weights_buf,
            wo,
            so,
            inner: Mutex::new(Inner {
                scratch,
                staged: None,
            }),
        })
    }
}

impl VadAccelerator for HexagonVad {
    fn window_16k(
        &self,
        padded: &[f32; 640],
        h: &[f32; 128],
        c: &[f32; 128],
    ) -> Result<Option<VadStep>> {
        let so = &self.so;
        let mut dev = self.device.lock_or_recover();
        let mut inner = self.inner.lock_or_recover();
        let session = dev.queue_session_mut();
        ensure!(
            session.outstanding_batches() == 0,
            "a previous VAD batch is still outstanding"
        );

        let Inner { scratch, staged } = &mut *inner;
        for (off, vals) in [
            (so.samples, &padded[..]),
            (so.h_in, &h[..]),
            (so.c_in, &c[..]),
        ] {
            scratch.as_mut_slice()[off..off + vals.len() * 4]
                .copy_from_slice(bytemuck::cast_slice(vals));
            scratch.flush_cpu_cache(off, vals.len() * 4);
        }

        if session.step_mode() {
            // `CERA_HEXAGON_STEP` flushes after every op group to find the one that hangs the
            // DSP; a replayed batch would skip that.
            session.drop_pending_batch();
            emit_window(session, &self.weights_buf, scratch, &self.wo, so)?;
            session.flush()?;
        } else if let Some(batch) = staged.as_ref() {
            // The whole descriptor block is copied again for every window (a few KB). The
            // resident variant, which skips the copy, hung the DSP from its second replay on:
            // the DSP leaves the descriptors it has run in a state the next run cannot use.
            session.flush_staged(batch)?;
        } else {
            // The first window goes through the ordinary flush, which maps the buffers for the
            // DSP; the batch is serialized first so every later window can replay it.
            session.drop_pending_batch();
            emit_window(session, &self.weights_buf, scratch, &self.wo, so)?;
            *staged = Some(session.export_staged_batch()?);
            session.flush()?;
        }

        scratch.invalidate_cpu_cache(so.gates, 4 * HID * 4);
        scratch.invalidate_cpu_cache(so.head, 4);
        let floats: &[f32] = bytemuck::cast_slice(scratch.as_slice());
        let at = |off: usize, n: usize| &floats[off / 4..off / 4 + n];
        let mut step = VadStep {
            prob: at(so.head, 1)[0],
            h: [0.0; 128],
            c: [0.0; 128],
        };
        step.h.copy_from_slice(at(so.gates + 3 * HID * 4, HID));
        step.c.copy_from_slice(at(so.gates + HID * 4, HID));
        // The boundary in `accelerated_16k` re-checks the whole step for every accelerator;
        // these device-side checks attribute an NPU fault to the readback value or state.
        ensure!(
            step.prob_valid(),
            "the NPU returned a speech probability of {}",
            step.prob
        );
        ensure!(
            step.state_finite(),
            "the NPU returned non-finite LSTM state",
        );
        Ok(Some(step))
    }
}

impl Drop for HexagonVad {
    fn drop(&mut self) {
        let mut dev = self.device.lock_or_recover();
        let inner = self.inner.lock_or_recover();
        dev.queue_session_mut()
            .release_dsp_references([&self.weights_buf, &inner.scratch]);
    }
}

impl SileroVad {
    /// Run this VAD's 16 kHz windows on the Hexagon NPU. Returns whether it did: `false` (with
    /// the reason logged) when there is no usable NPU, in which case it keeps running on the CPU.
    /// The window batch is built and exported lazily on the first window (which maps the buffers
    /// for the DSP), so that window pays the staging cost and a staging failure only drops out
    /// there: run one warm-up window before timing short jobs. Replaces any accelerator set
    /// before (sessions own their VAD, so sharing one accelerator across sessions can only come
    /// from an explicit `set_accelerator`).
    pub fn try_enable_hexagon(&mut self) -> bool {
        let Ok(context) = HexagonContext::new().inspect_err(|e| {
            crate::backend::hexagon::log_context_unavailable("HexagonVad", e);
        }) else {
            return false;
        };
        let arch_override = crate::backend::hexagon::arch_override();
        let device = match crate::backend::hexagon::probe_device(context.driver(), arch_override) {
            Ok(d) => Arc::new(Mutex::new(d)),
            Err(e) => {
                tracing::info!("HexagonVad: DSP device unavailable ({e}), using the CPU");
                return false;
            }
        };
        match HexagonVad::new(Arc::clone(context.driver()), device, &self.weights) {
            Ok(vad) => {
                tracing::info!("vad: using the Hexagon NPU");
                self.set_accelerator(Arc::new(vad));
                true
            }
            Err(e) => {
                crate::backend::hexagon::hexagon_error!("failed to stage the VAD on the NPU: {e}");
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::hexagon::HtpOpCode::*;
    use crate::backend::hexagon::dispatch::testing::RecordingSink;

    /// The window is one fixed graph: its op sequence is pinned so a change shows up here, and so
    /// does anything that would make it depend on the input (it is built once and replayed).
    #[test]
    fn a_window_is_a_fixed_graph_of_forty_nine_ops() {
        let (wo, so) = (WeightOffsets::plan(), Scratch::plan());
        let mut s = RecordingSink::default();
        emit_window(&mut s, &"w", &"b", &wo, &so).unwrap();
        let ops = s.opcodes();
        let count = |op| ops.iter().filter(|&&o| o == op as u32).count();
        // STFT, three taps of three conv layers, the centre tap of the last, two LSTM
        // matmuls and the head: 2 + 9 + 1 + 2 + 1.
        assert_eq!(count(MulMat), 15);
        assert_eq!(count(Sqrt), 1);
        assert_eq!(count(UnaryTanh), 2);
        // The i and f gates in one op, the o gate, and the head.
        assert_eq!(count(UnarySigmoid), 3);
        assert_eq!(ops.last().copied(), Some(UnarySigmoid as u32));
        // Framing (1) + STFT (6) + three emit_conv (3 x 7) + conv3 (3) + LSTM (13) + head (5).
        assert_eq!(ops.len(), 49);
        let conv = [MulMat, MulMat, MulMat, Add, Add, Add, UnaryRelu];
        let tail = [
            MulMat,
            Add,
            UnaryRelu, // conv3: one centre tap plus bias and ReLU
            MulMat,
            MulMat,
            Add,
            Add, // LSTM projections plus state plus bias
            UnarySigmoid,
            UnaryTanh,
            UnarySigmoid, // i+f, g, o gates
            Mul,
            Mul,
            Add,
            Cpy,
            UnaryTanh,
            Mul, // c' then h'
            Cpy,
            UnaryRelu,
            MulMat,
            Add,
            UnarySigmoid, // head
        ];
        let want: Vec<u32> = [Cpy, MulMat, MulMat, Mul, Mul, Add, Sqrt]
            .into_iter()
            .chain(conv.iter().copied().cycle().take(3 * conv.len()))
            .chain(tail)
            .map(|o| o as u32)
            .collect();
        assert_eq!(ops, want);
        // Every region fits its buffer: the last write of each is inside it.
        assert!(so.head + HEAD_ROWS * 4 <= so.total_bytes);
        assert!(wo.head_b + HEAD_ROWS * 4 <= wo.total_bytes);
        // `tmp_a`/`tmp_b` back both the STFT imaginary part and every conv layer's tap outputs:
        // each consumer must fit (the STFT fills it exactly). The capacity is read off the plan,
        // not restated.
        let tmp_floats = (so.tmp_b - so.tmp_a) / 4;
        assert!(
            FRAMES * BINS_PAD <= tmp_floats,
            "STFT imag ({} floats) must fit tmp ({tmp_floats})",
            FRAMES * BINS_PAD
        );
        for (layer, &n_out) in wo.conv.iter().zip(&[4, 2, 1]) {
            assert!(
                layer.out * n_out <= tmp_floats,
                "conv out {} x {n_out} must fit tmp ({tmp_floats})",
                layer.out
            );
        }
    }

    /// Every `emit_conv` layer reads its taps one row further in, stepping `stride` rows per
    /// output position, with tap `k` on tap `k`'s weights; the summed output lands at `dst`.
    #[test]
    fn conv_taps_step_rows_and_start_one_row_further() {
        let (wo, so) = (WeightOffsets::plan(), Scratch::plan());
        // (layer, src, stride, n_out, dst): the exact `emit_window` call sites.
        let cases = [
            (&wo.conv[0], so.pad0, 1, 4, so.pad1 + 128 * 4),
            (&wo.conv[1], so.pad1, 2, 2, so.pad2 + 64 * 4),
            (&wo.conv[2], so.pad2, 2, 1, so.enc2),
        ];
        for (layer, src, stride, n_out, dst) in cases {
            let mut s = RecordingSink::default();
            emit_conv(&mut s, &"w", &"b", layer, src, stride, n_out, dst, &so).unwrap();
            let mulmats: Vec<usize> = (0..s.ops.len())
                .filter(|&i| s.ops[i].opcode == MulMat as u32)
                .collect();
            assert_eq!(mulmats.len(), 3);
            let row = layer.in_pad * 4;
            for (k, &i) in mulmats.iter().enumerate() {
                let x = s.src(i, 1);
                assert_eq!(x.offset, src + k * row, "tap {k} start");
                assert_eq!(x.nb[1], (stride * row) as u32, "tap {k} step");
                assert_eq!(x.ne[1], n_out as u32, "tap {k} positions");
                assert_eq!(
                    s.src(i, 0).offset,
                    layer.taps[k],
                    "tap {k} reads tap {k}'s weights"
                );
            }
            assert_eq!(
                s.dst(mulmats[0]).offset,
                dst,
                "the summed output lands at dst"
            );
        }
    }

    /// The last layer is one hand-rolled centre-tap matmul: no op may read its outer taps.
    #[test]
    fn the_last_layer_reads_only_the_centre_tap() {
        let (wo, so) = (WeightOffsets::plan(), Scratch::plan());
        let mut s = RecordingSink::default();
        emit_window(&mut s, &"w", &"b", &wo, &so).unwrap();
        let c3 = &wo.conv[3];
        let mut centre = Vec::new();
        for (i, op) in s.ops.iter().enumerate() {
            if op.opcode != MulMat as u32 {
                continue;
            }
            let w = s.src(i, 0).offset;
            assert!(
                w != c3.taps[0] && w != c3.taps[2],
                "op {i} reads a zero-border tap of the last layer"
            );
            if w == c3.taps[1] {
                centre.push(i);
            }
        }
        assert_eq!(centre.len(), 1);
        let x = s.src(centre[0], 1);
        assert_eq!((x.offset, x.ne[0], x.ne[1]), (so.enc2, 64, 1));
        let d = s.dst(centre[0]);
        assert_eq!((d.offset, d.ne[0], d.ne[1]), (so.enc3, 128, 1));
    }

    /// The LSTM, head and gate ops read their own operands: each op's (weight, input, dst)
    /// triple is observed in the full `emit_window` recording, so swapping `w_ih` with `w_hh`
    /// (identical shapes), shifting a gate, or repointing the head input fails here rather
    /// than silently changing NPU numerics. Each conv layer's tap-0 input likewise, pinning
    /// the `emit_window` call-site args the unit test above restates.
    #[test]
    fn lstm_head_gates_and_conv_call_sites_read_their_own_operands() {
        let (wo, so) = (WeightOffsets::plan(), Scratch::plan());
        let mut s = RecordingSink::default();
        emit_window(&mut s, &"w", &"b", &wo, &so).unwrap();
        // The op reading a weight is found by offset scan, so the pin survives reordering.
        let mulmat_reading = |weight: usize| {
            let found: Vec<usize> = (0..s.ops.len())
                .filter(|&i| s.ops[i].opcode == MulMat as u32 && s.src(i, 0).offset == weight)
                .collect();
            assert_eq!(found.len(), 1, "one op reads the weight at {weight}");
            found[0]
        };
        // LSTM: W_ih x enc3 -> gates, W_hh x h_in -> gates2.
        let ih = mulmat_reading(wo.w_ih);
        assert_eq!(s.src(ih, 1).offset, so.enc3);
        assert_eq!(s.dst(ih).offset, so.gates);
        let hh = mulmat_reading(wo.w_hh);
        assert_eq!(s.src(hh, 1).offset, so.h_in);
        assert_eq!(s.dst(hh).offset, so.gates2);
        // Head: head_w x t128 -> head.
        let head = mulmat_reading(wo.head_w);
        assert_eq!(s.src(head, 1).offset, so.t128);
        assert_eq!(s.dst(head).offset, so.head);
        // Gates (unary, in place): i+f in one op, then o, then the head sigmoid.
        let (gi, gg, go) = (so.gates, so.gates + 2 * HID * 4, so.gates + 3 * HID * 4);
        let sigmoids: Vec<usize> = (0..s.ops.len())
            .filter(|&i| s.ops[i].opcode == UnarySigmoid as u32)
            .collect();
        assert_eq!(sigmoids.len(), 3);
        assert_eq!(s.dst(sigmoids[0]).offset, gi);
        assert_eq!(s.dst(sigmoids[1]).offset, go);
        assert_eq!(s.dst(sigmoids[2]).offset, so.head);
        let tanhs: Vec<usize> = (0..s.ops.len())
            .filter(|&i| s.ops[i].opcode == UnaryTanh as u32)
            .collect();
        assert_eq!(tanhs.len(), 2);
        assert_eq!(s.dst(tanhs[0]).offset, gg);
        assert_eq!(s.dst(tanhs[1]).offset, so.t128);
        // Conv call sites: tap-0 of layer L reads src_L into dst_L.
        for (layer, src, dst) in [
            (&wo.conv[0], so.pad0, so.pad1 + 128 * 4),
            (&wo.conv[1], so.pad1, so.pad2 + 64 * 4),
            (&wo.conv[2], so.pad2, so.enc2),
        ] {
            let tap0 = mulmat_reading(layer.taps[0]);
            assert_eq!(s.src(tap0, 1).offset, src, "tap-0 input");
            assert_eq!(s.dst(tap0).offset, dst, "tap-0 dst");
        }
    }

    /// Tap `k` of `[out, in, 3]` lands at `[out, in_pad]` with zero padded channels.
    #[test]
    fn conv_taps_are_split_and_padded() {
        let (out, in_real, in_pad) = (2, 3, 4);
        let w: Vec<f32> = (0..out * in_real * 3).map(|i| i as f32).collect();
        for k in 0..3 {
            let t = conv_tap(&w, out, in_real, in_pad, k);
            assert_eq!(t.len(), out * in_pad);
            for o in 0..out {
                for i in 0..in_pad {
                    let want = if i < in_real {
                        w[(o * in_real + i) * 3 + k]
                    } else {
                        0.0
                    };
                    assert_eq!(t[o * in_pad + i], want, "tap {k} out {o} in {i}");
                }
            }
        }
    }

    /// Tap order against the CPU ground truth: composing the three taps over a bordered
    /// input must equal `conv1d_relu` (layout, tap order, bias and ReLU).
    #[test]
    fn conv_taps_match_cpu_conv1d_relu() {
        let (out_c, in_c, in_pad) = (2, 3, 4);
        let (stride, pad, in_len) = (2, 1, 5);
        let input: Vec<f32> = (0..in_c * in_len).map(|i| i as f32 * 0.37 - 2.0).collect();
        let w: Vec<f32> = (0..out_c * in_c * 3)
            .map(|i| i as f32 * 0.13 - 1.0)
            .collect();
        let b: Vec<f32> = (0..out_c).map(|i| i as f32 * 0.5 - 0.25).collect();
        let out_len = (in_len + 2 * pad - 3) / stride + 1;
        let mut want = vec![0.0f32; out_c * out_len];
        crate::vad::conv1d_relu(&input, &mut want, in_c, in_len, out_c, stride, pad, &w, &b);
        let taps: Vec<Vec<f32>> = (0..3)
            .map(|k| conv_tap(&w, out_c, in_c, in_pad, k))
            .collect();
        for o in 0..out_c {
            for t in 0..out_len {
                let mut acc = b[o];
                for (k, tap) in taps.iter().enumerate() {
                    // Bordered row (`emit_conv` reads `src + k` stepping `stride` over a
                    // zero-bordered buffer); row 0 and row `in_len + 1` are the zero border.
                    let row = t * stride + k;
                    for i in 0..in_pad {
                        let x = if row == 0 || row == in_len + 1 || i >= in_c {
                            0.0
                        } else {
                            input[i * in_len + (row - 1)]
                        };
                        acc += tap[o * in_pad + i] * x;
                    }
                }
                let (got, want) = (acc.max(0.0), want[o * out_len + t]);
                assert!(
                    (got - want).abs() < 1e-5,
                    "out {o} pos {t}: tap composition {got} != CPU {want}"
                );
            }
        }
    }

    #[test]
    fn the_basis_is_split_into_real_and_imaginary_rows_padded_to_a_vector() {
        let basis: Vec<f32> = (0..2 * BINS * FRAME).map(|i| i as f32 + 1.0).collect();
        let (re, im) = (basis_rows(&basis, 0), basis_rows(&basis, BINS));
        assert_eq!(re[0], 1.0);
        assert_eq!(im[0], (BINS * FRAME) as f32 + 1.0);
        assert!(re[BINS * FRAME..].iter().all(|&v| v == 0.0));
        assert!(im[BINS * FRAME..].iter().all(|&v| v == 0.0));
    }
}
