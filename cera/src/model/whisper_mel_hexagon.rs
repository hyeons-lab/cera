//! Whisper's log-mel front end on the Hexagon NPU.
//!
//! With the encoder and decoder on the DSP, the host's remaining cost per utterance was the
//! log-mel: a 400-point FFT per frame and 80 dot products over the power spectrum, on a core the
//! scheduler has clocked down while it slept waiting for the DSP. The DFT and the filterbank are
//! matmuls, so they run on the DSP, through the same ops as LFM2-Audio's front end:
//!
//! ```text
//! padded samples (host: reflection and zeros, only as far as there is audio)
//! -> frames: a strided copy, one row per 160-sample hop (400 samples, padded to 416)
//! -> windowed DFT: two F32 matmuls against Hann-weighted cos / sin matrices (201 bins -> 224)
//! -> power: re*re + im*im
//! -> mel energies: one matmul against the Slaney filterbank
//! ```
//!
//! Only the frames that can hold audio are computed (see `whisper_preprocessor::active_frames`):
//! the rest of the 30 s window is silent and exactly the floor value. The host takes the `log10`
//! and does Whisper's normalization over the result (`finish_whisper_mel`).

use std::sync::{Arc, Mutex};

use crate::backend::hexagon::dispatch::{self, OpSink, TokenShape, TokenTile, View};
use crate::backend::hexagon::{FastRpcDriver, HexagonQueueSession, LockOrRecover, RpcmemBuffer};
use crate::model::audio_encoder_hexagon::{plan_vec, put_vec};
use crate::model::audio_preprocessor::{build_hann_window, build_mel_filterbank};
use crate::model::whisper_preprocessor::{
    CHUNK_FRAMES, CHUNK_SAMPLES, HOP_LEN, N_FFT, N_FFT_BINS, SAMPLE_RATE, samples_for_frames,
};
use crate::session::CeraError;

/// Frame length padded to a multiple of 32 elements, which the F32 matmul reads whole.
const K_PAD: usize = N_FFT.next_multiple_of(32);
/// DFT bins padded the same way; the padding rows and columns are zero.
const BINS_PAD: usize = N_FFT_BINS.next_multiple_of(32);

#[derive(Debug, Clone, Copy)]
struct WeightOffsets {
    /// `[BINS_PAD, K_PAD]`: `hann[n] * cos(2 pi k n / N_FFT)` per bin `k` (and the sine matrix).
    dft_re: usize,
    dft_im: usize,
    /// `[n_mels, BINS_PAD]`.
    filters: usize,
    total_bytes: usize,
}

impl WeightOffsets {
    fn plan(n_mels: usize) -> Self {
        let mut cur = 0;
        Self {
            dft_re: plan_vec(&mut cur, BINS_PAD * K_PAD),
            dft_im: plan_vec(&mut cur, BINS_PAD * K_PAD),
            filters: plan_vec(&mut cur, n_mels * BINS_PAD),
            total_bytes: cur,
        }
    }
}

/// The Hann-weighted DFT basis (`(cos, sin)`, `[BINS_PAD, K_PAD]`) for Whisper's window.
fn dft_matrices() -> (Vec<f32>, Vec<f32>) {
    let hann = build_hann_window(N_FFT);
    let mut re = vec![0.0f32; BINS_PAD * K_PAD];
    let mut im = vec![0.0f32; BINS_PAD * K_PAD];
    for k in 0..N_FFT_BINS {
        for (n, &h) in hann.iter().enumerate() {
            // Reduce k * n mod N_FFT before the trig call: exact in f64 and symmetric.
            let angle = std::f64::consts::TAU * ((k * n) % N_FFT) as f64 / N_FFT as f64;
            re[k * K_PAD + n] = (h as f64 * angle.cos()) as f32;
            im[k * K_PAD + n] = (h as f64 * angle.sin()) as f32;
        }
    }
    (re, im)
}

/// The mel filterbank with every row padded to `BINS_PAD` columns.
fn padded_filters(n_mels: usize) -> Vec<f32> {
    let fb = build_mel_filterbank(n_mels, N_FFT, SAMPLE_RATE);
    let mut out = vec![0.0f32; n_mels * BINS_PAD];
    for (row, src) in out
        .as_chunks_mut::<BINS_PAD>()
        .0
        .iter_mut()
        .zip(fb.as_chunks::<N_FFT_BINS>().0.iter())
    {
        row[..N_FFT_BINS].copy_from_slice(src);
    }
    out
}

/// Regions of the activation buffer, sized for the whole 30 s window.
#[derive(Debug, Clone, Copy)]
struct Scratch {
    samples: usize,
    frames: usize,
    re: usize,
    im: usize,
    mel: usize,
    total_bytes: usize,
}

impl Scratch {
    fn plan(n_mels: usize) -> Self {
        let mut cur = 0;
        let samples = plan_vec(&mut cur, CHUNK_SAMPLES + N_FFT);
        let frames = plan_vec(&mut cur, CHUNK_FRAMES * K_PAD);
        let re = plan_vec(&mut cur, CHUNK_FRAMES * BINS_PAD);
        let im = plan_vec(&mut cur, CHUNK_FRAMES * BINS_PAD);
        let mel = plan_vec(&mut cur, CHUNK_FRAMES * n_mels);
        Self {
            samples,
            frames,
            re,
            im,
            mel,
            total_bytes: cur,
        }
    }
}

/// Emit the ops that turn `n_frames` frames of the samples at `so.samples` into mel energies at
/// `so.mel` (`[n_frames, n_mels]`, time-major).
fn emit_mel<S: OpSink>(
    s: &mut S,
    w: &S::Buf,
    b: &S::Buf,
    wo: &WeightOffsets,
    so: &Scratch,
    n_frames: usize,
    n_mels: usize,
) -> Result<(), CeraError> {
    // Overlapping frames cannot be read as strided rows by the matmul (it faults the DSP), so
    // they are copied into contiguous rows first; the row padding stays zero.
    dispatch::copy_view(
        s,
        View::new(
            b,
            so.samples,
            [N_FFT, n_frames, 1],
            [4, HOP_LEN * 4, n_frames * HOP_LEN * 4],
        ),
        View::new(
            b,
            so.frames,
            [N_FFT, n_frames, 1],
            [4, K_PAD * 4, n_frames * K_PAD * 4],
        ),
    )?;
    let frames = View::new(
        b,
        so.frames,
        [K_PAD, n_frames, 1],
        [4, K_PAD * 4, n_frames * K_PAD * 4],
    );
    let plane = |off: usize, width: usize| {
        View::new(
            b,
            off,
            [width, n_frames, 1],
            [4, width * 4, n_frames * width * 4],
        )
    };
    let basis = |off: usize, rows: usize, cols: usize| {
        View::new(w, off, [cols, rows, 1], [4, cols * 4, rows * cols * 4])
    };
    dispatch::matmul_f32(
        s,
        basis(wo.dft_re, BINS_PAD, K_PAD),
        View::new(
            b,
            so.frames,
            [K_PAD, n_frames, 1],
            [4, K_PAD * 4, n_frames * K_PAD * 4],
        ),
        plane(so.re, BINS_PAD),
    )?;
    dispatch::matmul_f32(
        s,
        basis(wo.dft_im, BINS_PAD, K_PAD),
        frames,
        plane(so.im, BINS_PAD),
    )?;
    let shape = TokenShape {
        dim: BINS_PAD,
        n_tokens: n_frames,
    };
    // power = re^2 + im^2, in place in the real buffer.
    dispatch::mul_inplace(s, b, so.re, b, so.re, shape, TokenTile::Whole)?;
    dispatch::mul_inplace(s, b, so.im, b, so.im, shape, TokenTile::Whole)?;
    dispatch::add_residual(s, b, so.re, b, so.im, shape, TokenTile::Whole)?;
    dispatch::matmul_f32(
        s,
        basis(wo.filters, n_mels, BINS_PAD),
        plane(so.re, BINS_PAD),
        plane(so.mel, n_mels),
    )
}

/// Whisper's DFT and mel filterbank on the DSP: the weights and an activation buffer sized for
/// the whole 30 s window (about 14 MB), staged once.
pub(crate) struct WhisperMelDsp {
    weights_buf: RpcmemBuffer,
    scratch: Mutex<RpcmemBuffer>,
    wo: WeightOffsets,
    so: Scratch,
    n_mels: usize,
}

impl WhisperMelDsp {
    pub(crate) fn new(driver: Arc<FastRpcDriver>, n_mels: usize) -> Result<Self, CeraError> {
        let wo = WeightOffsets::plan(n_mels);
        let so = Scratch::plan(n_mels);
        let mut weights_buf = RpcmemBuffer::alloc(Arc::clone(&driver), wo.total_bytes, true)?;
        let (re, im) = dft_matrices();
        let dst = weights_buf.as_mut_slice();
        put_vec(dst, wo.dft_re, &re);
        put_vec(dst, wo.dft_im, &im);
        put_vec(dst, wo.filters, &padded_filters(n_mels));
        weights_buf.flush_cpu_cache(0, wo.total_bytes);
        // Fresh rpcmem is not guaranteed zero, and the frames' row padding must be.
        let mut scratch = RpcmemBuffer::alloc(driver, so.total_bytes, true)?;
        scratch.as_mut_slice().fill(0);
        scratch.flush_cpu_cache(0, so.total_bytes);
        Ok(Self {
            weights_buf,
            scratch: Mutex::new(scratch),
            wo,
            so,
            n_mels,
        })
    }

    /// The buffers a device must release before they are unmapped.
    pub(crate) fn buffers(&self) -> (&RpcmemBuffer, std::sync::MutexGuard<'_, RpcmemBuffer>) {
        (&self.weights_buf, self.scratch.lock_or_recover())
    }

    /// Mel energies `[n_frames, n_mels]` (time-major, before the log) for the first `n_frames`
    /// frames of `padded`, the window's samples as far as `samples_for_frames(n_frames)`.
    /// The session must already wait asleep.
    pub(crate) fn energies(
        &self,
        session: &mut HexagonQueueSession,
        padded: &[f32],
        n_frames: usize,
    ) -> Result<Vec<f32>, CeraError> {
        if n_frames == 0 || n_frames > CHUNK_FRAMES || padded.len() != samples_for_frames(n_frames)
        {
            return Err(CeraError::Backend(format!(
                "whisper mel: {} samples for {n_frames} frames",
                padded.len()
            )));
        }
        let (so, n_mels) = (&self.so, self.n_mels);
        let mut scratch = self.scratch.lock_or_recover();
        scratch.as_mut_slice()[so.samples..so.samples + padded.len() * 4]
            .copy_from_slice(bytemuck::cast_slice(padded));
        scratch.flush_cpu_cache(so.samples, padded.len() * 4);

        session.drop_pending_batch();
        emit_mel(
            session,
            &self.weights_buf,
            &scratch,
            &self.wo,
            so,
            n_frames,
            n_mels,
        )?;
        session.flush()?;

        let bytes = n_frames * n_mels * 4;
        scratch.invalidate_cpu_cache(so.mel, bytes);
        let floats: &[f32] = bytemuck::cast_slice(&scratch.as_slice()[so.mel..so.mel + bytes]);
        Ok(floats.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::hexagon::HtpOpCode::*;
    use crate::backend::hexagon::dispatch::testing::RecordingSink;

    /// The graph is the framing copy, two DFT matmuls, the power, and the filterbank matmul;
    /// the matmuls contract over padded, whole-vector lengths.
    #[test]
    fn the_front_end_frames_then_two_dfts_a_power_and_the_filterbank() {
        let (n_frames, n_mels) = (301, 80);
        let (wo, so) = (WeightOffsets::plan(n_mels), Scratch::plan(n_mels));
        let mut s = RecordingSink::default();
        emit_mel(&mut s, &"w", &"b", &wo, &so, n_frames, n_mels).unwrap();
        assert_eq!(
            s.opcodes(),
            [Cpy, MulMat, MulMat, Mul, Mul, Add, MulMat].map(|o| o as u32)
        );
        // The copy reads overlapping rows one hop apart into rows of the padded length.
        let src = s.src(0, 0);
        assert_eq!((src.ne[0], src.ne[1]), (N_FFT as u32, n_frames as u32));
        assert_eq!(src.nb[1], (HOP_LEN * 4) as u32);
        assert_eq!(s.dst(0).nb[1], (K_PAD * 4) as u32);
        // Contraction lengths are multiples of 32; the last matmul writes [n_mels, n_frames].
        assert_eq!(K_PAD % 32, 0);
        assert_eq!(BINS_PAD % 32, 0);
        assert_eq!(s.src(1, 1).ne[0], K_PAD as u32);
        assert_eq!(s.src(6, 0).ne[0], BINS_PAD as u32);
        assert_eq!(s.dst(6).ne[0], n_mels as u32);
        assert_eq!(s.dst(6).ne[1], n_frames as u32);
        // Every region fits the buffer at the full window.
        assert!(so.mel + CHUNK_FRAMES * n_mels * 4 <= so.total_bytes);
    }

    /// The DFT basis and the filterbank reproduce the CPU's spectrum: this is the DSP's
    /// arithmetic done in f64 on the host, so it pins the basis, the padding and the framing
    /// without a device.
    #[test]
    fn the_basis_and_filters_reproduce_the_cpu_mel_energies() {
        use crate::model::whisper_preprocessor::{
            active_frames, extract_whisper_mel, finish_whisper_mel, padded_whisper_audio_upto,
        };
        let n_mels = 80;
        let pcm: Vec<f32> = (0..24_000)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                (t * 440.0 * std::f32::consts::TAU).sin() * 0.3
                    + (t * 1830.0 * std::f32::consts::TAU).sin() * 0.1 * (1.0 + (t * 3.0).sin())
            })
            .collect();
        let n_active = active_frames(pcm.len());
        let padded = padded_whisper_audio_upto(&pcm, samples_for_frames(n_active));
        let (re, im) = dft_matrices();
        let filters = padded_filters(n_mels);
        let mut energies = vec![0.0f32; n_active * n_mels];
        let mut frame = vec![0.0f64; K_PAD];
        let mut power = vec![0.0f64; BINS_PAD];
        for f in 0..n_active {
            for (n, v) in frame.iter_mut().enumerate().take(N_FFT) {
                *v = padded[f * HOP_LEN + n] as f64;
            }
            for (k, p) in power.iter_mut().enumerate() {
                let dot = |m: &[f32]| -> f64 {
                    m[k * K_PAD..(k + 1) * K_PAD]
                        .iter()
                        .zip(&frame)
                        .map(|(&a, &b)| a as f64 * b)
                        .sum()
                };
                let (r, i) = (dot(&re), dot(&im));
                *p = r * r + i * i;
            }
            for m in 0..n_mels {
                energies[f * n_mels + m] = filters[m * BINS_PAD..(m + 1) * BINS_PAD]
                    .iter()
                    .zip(&power)
                    .map(|(&w, &p)| w as f64 * p)
                    .sum::<f64>() as f32;
            }
        }
        let got = finish_whisper_mel(&energies, n_mels, n_active);
        let want = extract_whisper_mel(&pcm, n_mels);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 2e-3, "max |diff| {worst}");
    }

    #[test]
    fn padding_rows_and_columns_are_zero() {
        let (re, im) = dft_matrices();
        for m in [&re, &im] {
            assert!(
                m[N_FFT_BINS * K_PAD..].iter().all(|&v| v == 0.0),
                "padded bins"
            );
            for row in m.as_chunks::<K_PAD>().0 {
                assert!(row[N_FFT..].iter().all(|&v| v == 0.0), "padded taps");
            }
        }
        let fb = padded_filters(128);
        assert!(
            fb.as_chunks::<BINS_PAD>()
                .0
                .iter()
                .all(|row| row[N_FFT_BINS..].iter().all(|&v| v == 0.0))
        );
    }
}
