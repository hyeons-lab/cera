//! The keyword spotter's backbone on the Hexagon NPU.
//!
//! A wake word is the one model an always-on service must evaluate continuously: the detector
//! re-runs its backbone over the whole 1.2 s window every 80 ms, whether or not anyone is
//! speaking (about 3 to 4 ms of CPU each on a phone, 12.5 times a second), and Android demotes
//! background CPU work and not the NPU. The backbone is a log-mel front end and four stride-2
//! convolutions, which map onto the same F32 matmuls the VAD and the audio front ends use:
//!
//! ```text
//! gain-normalized window (host)
//! -> frames: a strided copy, one row per hop (400 samples, padded to 416)
//! -> windowed DFT: two matmuls against Hann-weighted cos / sin matrices (257 bins -> 288)
//! -> power -> mel energies (matmul, HTK filterbank) -> ln(1 + e)
//! -> 4 x (conv k=3 stride 2 + SiLU)    three tap matmuls each, over zero-bordered rows
//! -> out [l3, embedding]               the host mean-pools it and runs the 64 -> 32 -> K head
//! ```
//!
//! The graph is the same for every window (fixed shapes and buffers), so it is built once,
//! serialized and replayed: per window the host writes the samples, submits, sleeps and reads the
//! small convolution output.

use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};

use crate::backend::hexagon::dispatch::{self, OpSink, TokenShape, TokenTile, View};
use crate::backend::hexagon::{
    FastRpcDriver, HexagonContext, HexagonDevice, LockOrRecover, RpcmemBuffer, StagedBatch,
};
use crate::hotword::{HotwordAccelerator, HotwordDetector};
use crate::model::audio_encoder_hexagon::{plan_vec, put_vec};
use crate::session::CeraError;

const TILE: TokenTile = TokenTile::Whole;
/// Channels of the three inner convolutions.
const CH: usize = 64;

/// The shapes of one detection window, from the model's metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Geometry {
    /// Samples per detection window.
    window: usize,
    /// Mel frame length and hop in samples, and the FFT length.
    frame: usize,
    hop: usize,
    fft: usize,
    mel_bins: usize,
    embedding: usize,
    /// Mel frames per window and the length after each stride-2 convolution.
    frames: usize,
    lens: [usize; 4],
}

impl Geometry {
    fn new(
        window: usize,
        frame: usize,
        hop: usize,
        fft: usize,
        mel_bins: usize,
        embedding: usize,
    ) -> Option<Self> {
        if window < frame || frame == 0 || hop == 0 || frame > fft {
            return None;
        }
        // The matmul reads whole 32-float vectors: the mel bins are the first convolution's input.
        if mel_bins == 0 || !mel_bins.is_multiple_of(32) || embedding == 0 {
            return None;
        }
        let frames = (window - frame) / hop + 1;
        let down = |n: usize| (n - 1) / 2 + 1;
        let l0 = down(frames);
        let l1 = down(l0);
        let l2 = down(l1);
        let l3 = down(l2);
        Some(Self {
            window,
            frame,
            hop,
            fft,
            mel_bins,
            embedding,
            frames,
            lens: [l0, l1, l2, l3],
        })
    }

    fn k_pad(&self) -> usize {
        self.frame.next_multiple_of(32)
    }

    fn bins(&self) -> usize {
        self.fft / 2 + 1
    }

    fn bins_pad(&self) -> usize {
        self.bins().next_multiple_of(32)
    }

    /// `(out channels, real in channels)` of the four convolutions.
    fn conv_shapes(&self) -> [(usize, usize); 4] {
        [
            (CH, self.mel_bins),
            (CH, CH),
            (CH, CH),
            (self.embedding, CH),
        ]
    }
}

/// One convolution layer's weights: `out x in` per tap, and the bias.
#[derive(Debug, Clone, Copy)]
struct ConvOffsets {
    taps: [usize; 3],
    bias: usize,
}

#[derive(Debug, Clone, Copy)]
struct WeightOffsets {
    basis_re: usize,
    basis_im: usize,
    filters: usize,
    conv: [ConvOffsets; 4],
    total_bytes: usize,
}

impl WeightOffsets {
    fn plan(g: &Geometry) -> Self {
        let mut cur = 0;
        let basis_re = plan_vec(&mut cur, g.bins_pad() * g.k_pad());
        let basis_im = plan_vec(&mut cur, g.bins_pad() * g.k_pad());
        let filters = plan_vec(&mut cur, g.mel_bins * g.bins_pad());
        let mut conv = [ConvOffsets {
            taps: [0; 3],
            bias: 0,
        }; 4];
        for (c, (out, inn)) in conv.iter_mut().zip(g.conv_shapes()) {
            *c = ConvOffsets {
                taps: [
                    plan_vec(&mut cur, out * inn),
                    plan_vec(&mut cur, out * inn),
                    plan_vec(&mut cur, out * inn),
                ],
                bias: plan_vec(&mut cur, out),
            };
        }
        Self {
            basis_re,
            basis_im,
            filters,
            conv,
            total_bytes: cur,
        }
    }
}

/// Tap `k` of a `[out, in, 3]` convolution weight as an `[out, in]` matrix.
fn conv_tap(w: &[f32], out: usize, inn: usize, k: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; out * inn];
    for o in 0..out {
        for i in 0..inn {
            m[o * inn + i] = w[(o * inn + i) * 3 + k];
        }
    }
    m
}

/// The Hann-weighted DFT basis (`(cos, sin)`, `[bins_pad, k_pad]`): row `k` is
/// `hann[n] cos(2 pi k n / fft)` for the `frame` taps, zero beyond them and below the last bin.
fn dft_matrices(g: &Geometry, hann: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (kp, bp) = (g.k_pad(), g.bins_pad());
    let mut re = vec![0.0f32; bp * kp];
    let mut im = vec![0.0f32; bp * kp];
    for k in 0..g.bins() {
        for (n, &h) in hann.iter().enumerate().take(g.frame) {
            let angle = std::f64::consts::TAU * ((k * n) % g.fft) as f64 / g.fft as f64;
            re[k * kp + n] = (h as f64 * angle.cos()) as f32;
            im[k * kp + n] = (h as f64 * angle.sin()) as f32;
        }
    }
    (re, im)
}

/// The `[mel_bins, bins]` filterbank with every row padded to `bins_pad` columns.
fn padded_filters(g: &Geometry, fb: &[f32]) -> Vec<f32> {
    let (bins, bp) = (g.bins(), g.bins_pad());
    let mut out = vec![0.0f32; g.mel_bins * bp];
    for m in 0..g.mel_bins {
        out[m * bp..m * bp + bins].copy_from_slice(&fb[m * bins..(m + 1) * bins]);
    }
    out
}

/// Activation scratch of one window, in bytes.
#[derive(Debug, Clone, Copy)]
struct Scratch {
    samples: usize,
    frames: usize,
    re: usize,
    im: usize,
    /// Zero-bordered layer inputs (`len + 2` rows each): the log-mel, then the outputs of the
    /// first three convolutions. The border rows are zeroed once and never written.
    pad: [usize; 4],
    /// Per-tap results summed into a layer's output.
    tmp_a: usize,
    tmp_b: usize,
    /// The last convolution's output, `[l3, embedding]`.
    out: usize,
    total_bytes: usize,
}

impl Scratch {
    fn plan(g: &Geometry) -> Self {
        let mut cur = 0;
        let samples = plan_vec(&mut cur, g.window);
        let frames = plan_vec(&mut cur, g.frames * g.k_pad());
        let re = plan_vec(&mut cur, g.frames * g.bins_pad());
        let im = plan_vec(&mut cur, g.frames * g.bins_pad());
        let pad = [
            plan_vec(&mut cur, (g.frames + 2) * g.mel_bins),
            plan_vec(&mut cur, (g.lens[0] + 2) * CH),
            plan_vec(&mut cur, (g.lens[1] + 2) * CH),
            plan_vec(&mut cur, (g.lens[2] + 2) * CH),
        ];
        // The DFT's imaginary plane doubles as a tap buffer later, so the taps are sized for the
        // widest layer output.
        let widest = (0..4)
            .map(|i| g.lens[i] * g.conv_shapes()[i].0)
            .max()
            .unwrap_or(1);
        let tmp_a = plan_vec(&mut cur, widest.max(g.frames * g.bins_pad()));
        let tmp_b = plan_vec(&mut cur, widest);
        let out = plan_vec(&mut cur, g.lens[3] * g.embedding);
        Self {
            samples,
            frames,
            re,
            im,
            pad,
            tmp_a,
            tmp_b,
            out,
            total_bytes: cur,
        }
    }
}

/// One stride-2 convolution layer over `n_out` positions of the zero-bordered `[rows, in]`
/// buffer at `src`: three tap matmuls (tap `k` starts `k` rows in and steps two rows), summed
/// into the first tap's destination `dst`, then the bias and SiLU.
#[allow(clippy::too_many_arguments)]
fn emit_conv<S: OpSink>(
    s: &mut S,
    w: &S::Buf,
    b: &S::Buf,
    layer: &ConvOffsets,
    (out, inn): (usize, usize),
    src: usize,
    n_out: usize,
    dst: usize,
    so: &Scratch,
) -> Result<(), CeraError> {
    let row = inn * 4;
    let weights = |off: usize| View::new(w, off, [inn, out, 1], [4, row, out * row]);
    let rows_out = |off: usize| View::new(b, off, [out, n_out, 1], [4, out * 4, n_out * out * 4]);
    for (k, &to) in [dst, so.tmp_a, so.tmp_b].iter().enumerate() {
        let x = View::new(
            b,
            src + k * row,
            [inn, n_out, 1],
            [4, 2 * row, n_out * 2 * row],
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
    dispatch::silu(s, b, dst, shape, TILE)
}

/// Emit the whole backbone: from the window at `so.samples` to the convolution output at `so.out`.
fn emit_window<S: OpSink>(
    s: &mut S,
    w: &S::Buf,
    b: &S::Buf,
    g: &Geometry,
    wo: &WeightOffsets,
    so: &Scratch,
) -> Result<(), CeraError> {
    let (kp, bp, nf, mb) = (g.k_pad(), g.bins_pad(), g.frames, g.mel_bins);
    let rows = |off: usize, dim: usize, n: usize| {
        View::new(b, off, [dim, n, 1], [4, dim * 4, n * dim * 4])
    };
    let basis =
        |off: usize, n: usize, k: usize| View::new(w, off, [k, n, 1], [4, k * 4, n * k * 4]);

    // Framing copy (overlapping rows cannot be read in place by the matmul).
    dispatch::copy_view(
        s,
        View::new(
            b,
            so.samples,
            [g.frame, nf, 1],
            [4, g.hop * 4, nf * g.hop * 4],
        ),
        View::new(b, so.frames, [g.frame, nf, 1], [4, kp * 4, nf * kp * 4]),
    )?;
    dispatch::matmul_f32(
        s,
        basis(wo.basis_re, bp, kp),
        rows(so.frames, kp, nf),
        rows(so.re, bp, nf),
    )?;
    dispatch::matmul_f32(
        s,
        basis(wo.basis_im, bp, kp),
        rows(so.frames, kp, nf),
        rows(so.im, bp, nf),
    )?;
    let bins = TokenShape {
        dim: bp,
        n_tokens: nf,
    };
    dispatch::mul_inplace(s, b, so.re, b, so.re, bins, TILE)?;
    dispatch::mul_inplace(s, b, so.im, b, so.im, bins, TILE)?;
    dispatch::add_residual(s, b, so.re, b, so.im, bins, TILE)?;
    // Mel energies straight into rows 1..=nf of the zero-bordered first layer input, then ln(1 + e).
    let mel = so.pad[0] + mb * 4;
    dispatch::matmul_f32(
        s,
        basis(wo.filters, mb, bp),
        rows(so.re, bp, nf),
        rows(mel, mb, nf),
    )?;
    let mel_shape = TokenShape {
        dim: mb,
        n_tokens: nf,
    };
    dispatch::scale_offset(s, b, mel, mel_shape, TILE, (1.0, 1.0))?;
    dispatch::log_inplace(s, b, mel, mel_shape, TILE)?;

    // Backbone: nf -> l0 -> l1 -> l2 -> l3 positions, each layer writing rows 1.. of the next
    // layer's zero-bordered input (the last one into `out`).
    for (i, (layer, shape)) in wo.conv.iter().zip(g.conv_shapes()).enumerate() {
        let dst = if i < 3 {
            so.pad[i + 1] + CH * 4
        } else {
            so.out
        };
        emit_conv(s, w, b, layer, shape, so.pad[i], g.lens[i], dst, so)?;
    }
    Ok(())
}

struct Inner {
    scratch: RpcmemBuffer,
    /// The serialized window batch, built from the first window.
    staged: Option<StagedBatch>,
}

/// The keyword spotter's backbone on the Hexagon NPU. See the module docs.
pub struct HexagonHotword {
    device: Arc<Mutex<HexagonDevice>>,
    weights_buf: RpcmemBuffer,
    g: Geometry,
    wo: WeightOffsets,
    so: Scratch,
    inner: Mutex<Inner>,
}

// SAFETY: the rpcmem buffers are only touched while holding `device` and `inner`.
unsafe impl Send for HexagonHotword {}
unsafe impl Sync for HexagonHotword {}

impl HexagonHotword {
    /// Stage `detector`'s backbone on `device`, whose queue session then sleeps on DSP waits
    /// (pass a device nothing else shares).
    fn new(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        detector: &HotwordDetector,
    ) -> Result<Self, CeraError> {
        let w = &detector.weights;
        let g = Geometry::new(
            w.window_samples,
            w.mel_window_samples,
            w.mel_hop_samples,
            w.fft_size,
            w.mel_bins,
            w.embedding_dim,
        )
        .ok_or_else(|| {
            CeraError::Backend(format!(
                "this keyword spotter's shapes cannot run on the NPU (window {}, frame {}, hop {}, \
                 fft {}, {} mel bins)",
                w.window_samples, w.mel_window_samples, w.mel_hop_samples, w.fft_size, w.mel_bins
            ))
        })?;
        let wo = WeightOffsets::plan(&g);
        let so = Scratch::plan(&g);
        let mut weights_buf = RpcmemBuffer::alloc(Arc::clone(&driver), wo.total_bytes, true)?;
        {
            let dst = weights_buf.as_mut_slice();
            let (re, im) = dft_matrices(&g, &detector.front_end.hann_window);
            put_vec(dst, wo.basis_re, &re);
            put_vec(dst, wo.basis_im, &im);
            put_vec(
                dst,
                wo.filters,
                &padded_filters(&g, &detector.front_end.mel_filterbank),
            );
            let layers = [
                (w.conv0_w.as_f32_slice(), w.conv0_b.as_f32_slice()),
                (w.conv1_w.as_f32_slice(), w.conv1_b.as_f32_slice()),
                (w.conv2_w.as_f32_slice(), w.conv2_b.as_f32_slice()),
                (w.conv3_w.as_f32_slice(), w.conv3_b.as_f32_slice()),
            ];
            for ((c, (out, inn)), (wt, bias)) in wo.conv.iter().zip(g.conv_shapes()).zip(layers) {
                for (k, &off) in c.taps.iter().enumerate() {
                    put_vec(dst, off, &conv_tap(wt, out, inn, k));
                }
                put_vec(dst, c.bias, bias);
            }
        }
        weights_buf.flush_cpu_cache(0, wo.total_bytes);
        // Fresh rpcmem is not guaranteed zero: the border rows and the frames' padding must be.
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
            g,
            wo,
            so,
            inner: Mutex::new(Inner {
                scratch,
                staged: None,
            }),
        })
    }
}

impl HotwordAccelerator for HexagonHotword {
    fn embedding(&self, window: &[f32]) -> Result<Option<Vec<f32>>> {
        let (g, so) = (&self.g, &self.so);
        ensure!(
            window.len() == g.window,
            "expected {} samples, got {}",
            g.window,
            window.len()
        );
        let mut dev = self.device.lock_or_recover();
        let mut inner = self.inner.lock_or_recover();
        let session = dev.queue_session_mut();
        ensure!(
            session.outstanding_batches() == 0,
            "a previous keyword batch is still outstanding"
        );

        let Inner { scratch, staged } = &mut *inner;
        scratch.as_mut_slice()[so.samples..so.samples + window.len() * 4]
            .copy_from_slice(bytemuck::cast_slice(window));
        scratch.flush_cpu_cache(so.samples, window.len() * 4);

        if session.step_mode() {
            // `CERA_HEXAGON_STEP` flushes after every op group to find the one that hangs the DSP.
            session.drop_pending_batch();
            emit_window(session, &self.weights_buf, scratch, g, &self.wo, so)?;
            session.flush()?;
        } else if let Some(batch) = staged.as_ref() {
            // The descriptor block is copied again for every window; replaying it resident
            // (as LFM2's decode does) hung the DSP for the VAD's graph from the second replay on.
            session.flush_staged(batch)?;
        } else {
            // The first window goes through the ordinary flush, which maps the buffers for the
            // DSP; the batch is serialized first so every later window can replay it.
            session.drop_pending_batch();
            emit_window(session, &self.weights_buf, scratch, g, &self.wo, so)?;
            *staged = Some(session.export_staged_batch()?);
            session.flush()?;
        }

        let bytes = g.lens[3] * g.embedding * 4;
        scratch.invalidate_cpu_cache(so.out, bytes);
        let out: &[f32] = bytemuck::cast_slice(&scratch.as_slice()[so.out..so.out + bytes]);
        ensure!(
            out.iter().all(|v| v.is_finite()),
            "the NPU returned a non-finite embedding"
        );
        // Temporal mean pooling, like the CPU path.
        let inv = 1.0 / g.lens[3] as f32;
        let emb = (0..g.embedding)
            .map(|c| out.chunks_exact(g.embedding).map(|row| row[c]).sum::<f32>() * inv)
            .collect();
        Ok(Some(emb))
    }
}

impl Drop for HexagonHotword {
    fn drop(&mut self) {
        let mut dev = self.device.lock_or_recover();
        let inner = self.inner.lock_or_recover();
        dev.queue_session_mut()
            .release_dsp_references([&self.weights_buf, &inner.scratch]);
    }
}

impl HotwordDetector {
    /// Run this detector's backbone on the Hexagon NPU. Returns whether it did: `false` (with the
    /// reason logged) when there is no usable NPU or the model's shapes do not fit, in which case
    /// it keeps running on the CPU.
    pub fn try_enable_hexagon(&mut self) -> bool {
        let Ok(context) = HexagonContext::new().inspect_err(|e| {
            crate::backend::hexagon::log_context_unavailable("HexagonHotword", e);
        }) else {
            return false;
        };
        let arch_override = std::env::var("CERA_HEXAGON_ARCH")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .and_then(crate::backend::hexagon::HexagonArch::from_u32);
        let device = match crate::backend::hexagon::probe_device(context.driver(), arch_override) {
            Ok(d) => Arc::new(Mutex::new(d)),
            Err(e) => {
                tracing::info!("HexagonHotword: DSP device unavailable ({e}), using the CPU");
                return false;
            }
        };
        match HexagonHotword::new(Arc::clone(context.driver()), device, self) {
            Ok(h) => {
                tracing::info!("hotword: using the Hexagon NPU");
                self.set_accelerator(Arc::new(h));
                true
            }
            Err(e) => {
                crate::backend::hexagon::hexagon_error!(
                    "failed to stage the keyword spotter on the NPU: {e}"
                );
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

    /// The shipped model's shapes.
    fn shipped() -> Geometry {
        Geometry::new(19_200, 400, 160, 512, 32, 64).unwrap()
    }

    #[test]
    fn the_shipped_models_shapes() {
        let g = shipped();
        assert_eq!((g.frames, g.lens), (118, [59, 30, 15, 8]));
        assert_eq!((g.k_pad(), g.bins(), g.bins_pad()), (416, 257, 288));
        // Shapes the matmul cannot read are refused rather than mis-run.
        assert!(
            Geometry::new(19_200, 400, 160, 512, 40, 64).is_none(),
            "40 mel bins"
        );
        assert!(
            Geometry::new(300, 400, 160, 512, 32, 64).is_none(),
            "window shorter than a frame"
        );
        assert!(
            Geometry::new(19_200, 600, 160, 512, 32, 64).is_none(),
            "frame longer than the fft"
        );
    }

    /// The window is one fixed graph; its ops are pinned so a change shows up here.
    #[test]
    fn a_window_is_a_fixed_graph() {
        let g = shipped();
        let (wo, so) = (WeightOffsets::plan(&g), Scratch::plan(&g));
        let mut s = RecordingSink::default();
        emit_window(&mut s, &"w", &"b", &g, &wo, &so).unwrap();
        let ops = s.opcodes();
        let count = |op| ops.iter().filter(|&&o| o == op as u32).count();
        // The two DFT matmuls and the filterbank, then three taps in each of four layers.
        assert_eq!(count(MulMat), 3 + 12);
        assert_eq!(count(UnaryLog), 1);
        assert_eq!(count(UnarySilu), 4);
        assert_eq!(ops.last().copied(), Some(UnarySilu as u32));
        // Every stride-2 layer's taps start one row further in and step two rows.
        let row = |i: usize| (if i == 0 { g.mel_bins } else { CH } * 4) as u32;
        let layer_mms: Vec<usize> = (0..s.ops.len())
            .filter(|&i| s.ops[i].opcode == MulMat as u32)
            .skip(3)
            .collect();
        assert_eq!(layer_mms.len(), 12);
        for (l, taps) in layer_mms.chunks(3).enumerate() {
            for (k, &i) in taps.iter().enumerate() {
                let x = s.src(i, 1);
                assert_eq!(
                    x.offset,
                    so.pad[l] + k * row(l) as usize,
                    "layer {l} tap {k}"
                );
                assert_eq!(x.nb[1], 2 * row(l), "layer {l} steps two rows");
                assert_eq!(x.ne[1], g.lens[l] as u32);
            }
        }
        assert!(so.out + g.lens[3] * g.embedding * 4 <= so.total_bytes);
    }

    /// A deterministic synthetic keyword-spotting GGUF (random weights, the shipped shapes): the
    /// math is checked against the CPU backbone, which does not need trained weights.
    fn synthetic_model() -> Vec<u8> {
        use crate::convert::writer::{GGML_TYPE_F32, GgufWriter};
        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5
        };
        let mut w = GgufWriter::new();
        w.add_string("general.architecture", "kws");
        w.add_string("general.name", "synthetic KWS");
        w.add_u32("kws.keyword_count", 1);
        w.add_string_array("kws.keywords", vec!["Hey Synthetic".to_string()]);
        for (k, v) in [
            ("kws.sample_rate", 16_000),
            ("kws.window_samples", 19_200),
            ("kws.hop_samples", 1_280),
            ("kws.mel_bins", 32),
            ("kws.mel_window_samples", 400),
            ("kws.mel_hop_samples", 160),
            ("kws.fft_size", 512),
            ("kws.embedding_dim", 64),
        ] {
            w.add_u32(k, v);
        }
        w.add_f32("kws.default_threshold", 0.75);
        let tensors: [(&str, usize, f32); 12] = [
            ("kws.backbone.conv0.weight", 64 * 32 * 3, 0.08),
            ("kws.backbone.conv0.bias", 64, 0.1),
            ("kws.backbone.conv1.weight", 64 * 64 * 3, 0.06),
            ("kws.backbone.conv1.bias", 64, 0.1),
            ("kws.backbone.conv2.weight", 64 * 64 * 3, 0.06),
            ("kws.backbone.conv2.bias", 64, 0.1),
            ("kws.backbone.conv3.weight", 64 * 64 * 3, 0.06),
            ("kws.backbone.conv3.bias", 64, 0.1),
            ("kws.head.dense1.weight", 32 * 64, 0.2),
            ("kws.head.dense1.bias", 32, 0.1),
            ("kws.head.dense2.weight", 32, 0.3),
            ("kws.head.dense2.bias", 1, 0.1),
        ];
        let data: Vec<Vec<f32>> = tensors
            .iter()
            .map(|&(_, n, scale)| (0..n).map(|_| next() * scale).collect())
            .collect();
        for ((name, n, _), _) in tensors.iter().zip(&data) {
            w.add_tensor(*name, vec![*n as u64], GGML_TYPE_F32, n * 4);
        }
        let mut bytes = Vec::new();
        w.write_header_and_tensor_info(&mut bytes).unwrap();
        for d in &data {
            w.write_tensor_data(&mut bytes, bytemuck::cast_slice(d))
                .unwrap();
        }
        bytes
    }

    /// The DSP's arithmetic done in f64 on the host (DFT basis, filterbank, `ln(1 + e)`, the tap
    /// decomposition with its zero borders and stride-2 reads, mean pooling) reproduces the CPU
    /// backbone: this pins the indexing and the padding of every layer without a device.
    #[test]
    fn the_decomposition_reproduces_the_cpu_backbone() {
        let mut detector = HotwordDetector::from_bytes(synthetic_model()).unwrap();
        let g = shipped();
        let window: Vec<f32> = (0..g.window)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                (t * 330.0 * std::f32::consts::TAU).sin() * 0.4
                    + (t * 1700.0 * std::f32::consts::TAU).sin() * 0.2 * (1.0 + (t * 4.0).sin())
            })
            .collect();
        // A real CPU pass: the gain-normalized window and the pooled embedding it produced.
        detector.process_window(&window).unwrap();
        let (scaled, cpu) = (
            detector.window_scratch.clone(),
            detector.emb_scratch.clone(),
        );
        let by_taps = embedding_by_taps(&detector, &scaled, &g);
        let worst = by_taps
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 2e-3, "embedding max |diff| {worst}");
        assert!(
            cpu.iter().any(|v| v.abs() > 1e-3),
            "a trivially zero embedding proves nothing"
        );
    }

    /// The embedding of `window` through the decomposition (f64 on the host).
    fn embedding_by_taps(detector: &HotwordDetector, window: &[f32], g: &Geometry) -> Vec<f32> {
        let (re, im) = dft_matrices(g, &detector.front_end.hann_window);
        let fb = padded_filters(g, &detector.front_end.mel_filterbank);
        let (kp, bp) = (g.k_pad(), g.bins_pad());
        let mut x = vec![vec![0.0f64; g.mel_bins]; g.frames];
        for (f, row) in x.iter_mut().enumerate() {
            let frame: Vec<f64> = (0..kp)
                .map(|n| {
                    if n < g.frame {
                        window[f * g.hop + n] as f64
                    } else {
                        0.0
                    }
                })
                .collect();
            let power: Vec<f64> = (0..bp)
                .map(|k| {
                    let dot = |m: &[f32]| -> f64 {
                        m[k * kp..(k + 1) * kp]
                            .iter()
                            .zip(&frame)
                            .map(|(&a, &b)| a as f64 * b)
                            .sum()
                    };
                    let (r, i) = (dot(&re), dot(&im));
                    r * r + i * i
                })
                .collect();
            for (m, v) in row.iter_mut().enumerate() {
                let e: f64 = fb[m * bp..(m + 1) * bp]
                    .iter()
                    .zip(&power)
                    .map(|(&w, &p)| w as f64 * p)
                    .sum();
                *v = (1.0 + e).ln();
            }
        }
        let w = &detector.weights;
        let layers = [
            (w.conv0_w.as_f32_slice(), w.conv0_b.as_f32_slice()),
            (w.conv1_w.as_f32_slice(), w.conv1_b.as_f32_slice()),
            (w.conv2_w.as_f32_slice(), w.conv2_b.as_f32_slice()),
            (w.conv3_w.as_f32_slice(), w.conv3_b.as_f32_slice()),
        ];
        for (l, ((out, inn), (wt, bias))) in g.conv_shapes().into_iter().zip(layers).enumerate() {
            let taps: Vec<Vec<f32>> = (0..3).map(|k| conv_tap(wt, out, inn, k)).collect();
            let mut next = vec![vec![0.0f64; out]; g.lens[l]];
            for (t, row) in next.iter_mut().enumerate() {
                for (o, v) in row.iter_mut().enumerate() {
                    let mut acc = bias[o] as f64;
                    for (k, tap) in taps.iter().enumerate() {
                        let r = 2 * t as isize + k as isize - 1;
                        if r >= 0 && (r as usize) < x.len() {
                            acc += (0..inn)
                                .map(|i| tap[o * inn + i] as f64 * x[r as usize][i])
                                .sum::<f64>();
                        }
                    }
                    *v = acc / (1.0 + (-acc).exp());
                }
            }
            x = next;
        }
        (0..g.embedding)
            .map(|c| (x.iter().map(|r| r[c]).sum::<f64>() / g.lens[3] as f64) as f32)
            .collect()
    }

    #[test]
    fn conv_taps_are_split_per_tap() {
        let (out, inn) = (2, 3);
        let w: Vec<f32> = (0..out * inn * 3).map(|i| i as f32).collect();
        for k in 0..3 {
            let t = conv_tap(&w, out, inn, k);
            for o in 0..out {
                for i in 0..inn {
                    assert_eq!(t[o * inn + i], w[(o * inn + i) * 3 + k]);
                }
            }
        }
    }
}
