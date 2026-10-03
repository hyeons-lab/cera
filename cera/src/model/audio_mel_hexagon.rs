//! The log-mel front end on the Hexagon NPU.
//!
//! Log-mel is the last stage of a speech turn that still ran on the CPU once the
//! conv stem and the blocks moved to the DSP (0.017 CPU-seconds for 10 s of
//! audio, almost all of it the filterbank dot product in `f64`). It maps onto
//! the same F32 matmul the conv stem uses:
//!
//! ```text
//! padded, pre-emphasised samples (host, O(n))
//! -> frames: a strided view of the samples, one row per 160-sample hop
//! -> windowed DFT: two matmuls against Hann-weighted cos / sin matrices
//! -> power: re*re + im*im
//! -> mel energies: one matmul against the filterbank
//! ```
//!
//! The DSP returns the linear mel energies. The natural log and the
//! per-feature normalization, which needs statistics over every frame, stay on
//! the host through the helpers the CPU path uses
//! ([`finish_log_mel`]), so both paths normalize identically; that part is
//! `O(frames * mel bins)`, a millisecond for ten seconds of audio.

use crate::backend::hexagon::align128;
use crate::backend::hexagon::dispatch::{self, OpSink, TokenShape, TokenTile, View};
use crate::model::audio_encoder::{HOP_LEN, LOG_MEL_EPS, N_FFT, SAMPLE_RATE};
use crate::model::audio_encoder_hexagon::{plan_vec, put_vec};
use crate::model::audio_preprocessor::{
    N_FFT_BINS, build_mel_filterbank, build_padded_hann_window, finish_log_mel,
};
use crate::session::CeraError;

/// DFT bins padded to a multiple of 32 elements, which the F32 matmul needs.
/// The padding rows of the DFT matrices and columns of the filterbank are zero.
pub(crate) const BINS_PAD: usize = N_FFT_BINS.next_multiple_of(32);

/// Where the front end's constant matrices sit in `weights_buf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MelWeightOffsets {
    /// `[BINS_PAD, N_FFT]` F32: `hann[n] * cos(2 pi k n / N_FFT)` per bin `k`.
    pub dft_re: usize,
    /// The same with `sin`.
    pub dft_im: usize,
    /// `[n_mel, BINS_PAD]` F32: the Slaney filterbank.
    pub filters: usize,
}

impl MelWeightOffsets {
    pub(crate) fn plan(cur: &mut usize, n_mel: usize) -> Self {
        Self {
            dft_re: plan_vec(cur, BINS_PAD * N_FFT),
            dft_im: plan_vec(cur, BINS_PAD * N_FFT),
            filters: plan_vec(cur, n_mel * BINS_PAD),
        }
    }
}

/// The windowed DFT basis: `(cos, sin)` matrices of `[BINS_PAD, N_FFT]`.
/// The window is the padded Hann the CPU path uses, so a frame's DFT here is
/// the FFT of the windowed frame there.
pub(crate) fn dft_matrices() -> (Vec<f32>, Vec<f32>) {
    dft_matrices_for(&build_padded_hann_window())
}

/// [`dft_matrices`] for any `N_FFT`-long window (Sortformer ships its own in its GGUF).
pub(crate) fn dft_matrices_for(window: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(window.len(), N_FFT, "window must be N_FFT long");
    let mut re = vec![0.0f32; BINS_PAD * N_FFT];
    let mut im = vec![0.0f32; BINS_PAD * N_FFT];
    for k in 0..N_FFT_BINS {
        for (n, &h) in window.iter().enumerate() {
            // Reduce k * n mod N_FFT before the trig call: it keeps the angle
            // exact in f64 and the table symmetric.
            let angle = std::f64::consts::TAU * ((k * n) % N_FFT) as f64 / N_FFT as f64;
            re[k * N_FFT + n] = (h as f64 * angle.cos()) as f32;
            im[k * N_FFT + n] = (h as f64 * angle.sin()) as f32;
        }
    }
    (re, im)
}

/// The mel filterbank with each row padded to [`BINS_PAD`] columns.
pub(crate) fn padded_filters(n_mel: usize) -> Vec<f32> {
    padded_filters_from(
        &build_mel_filterbank(n_mel, N_FFT, SAMPLE_RATE as usize),
        n_mel,
    )
}

/// A `[n_mel, N_FFT_BINS]` filterbank with each row padded to [`BINS_PAD`] columns.
pub(crate) fn padded_filters_from(filters: &[f32], n_mel: usize) -> Vec<f32> {
    assert_eq!(filters.len(), n_mel * N_FFT_BINS, "filterbank shape");
    let mut out = vec![0.0f32; n_mel * BINS_PAD];
    for (row, src) in out
        .as_chunks_mut::<BINS_PAD>()
        .0
        .iter_mut()
        .zip(filters.as_chunks::<N_FFT_BINS>().0)
    {
        row[..N_FFT_BINS].copy_from_slice(src);
    }
    out
}

/// Copy the front end's constant matrices to their planned places.
pub(crate) fn put_mel(dst: &mut [u8], o: &MelWeightOffsets, n_mel: usize) {
    let (re, im) = dft_matrices();
    put_vec(dst, o.dft_re, &re);
    put_vec(dst, o.dft_im, &im);
    put_vec(dst, o.filters, &padded_filters(n_mel));
}

/// [`put_mel`] with the caller's window and `[n_mel, N_FFT_BINS]` filterbank.
pub(crate) fn put_mel_tables(
    dst: &mut [u8],
    o: &MelWeightOffsets,
    window: &[f32],
    filters: &[f32],
    n_mel: usize,
) {
    let (re, im) = dft_matrices_for(window);
    put_vec(dst, o.dft_re, &re);
    put_vec(dst, o.dft_im, &im);
    put_vec(dst, o.filters, &padded_filters_from(filters, n_mel));
}

/// Regions of the front end's per-clip buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MelScratch {
    /// The padded, pre-emphasised samples.
    pub samples: usize,
    /// The frames gathered into contiguous rows, `[n_frames, N_FFT]`.
    pub frames: usize,
    /// Real and imaginary DFT parts, `[n_frames, BINS_PAD]` each; the real
    /// buffer then holds the power.
    pub re: usize,
    pub im: usize,
    /// Mel energies, time-major `[n_frames, n_mel]`.
    pub mel: usize,
    pub total_bytes: usize,
}

impl MelScratch {
    pub(crate) fn new(n_samples: usize, n_frames: usize, n_mel: usize) -> Self {
        let mut cur = 0;
        let mut region = |floats: usize| {
            let off = cur;
            cur += align128(floats * 4);
            off
        };
        let samples = region(n_samples);
        let frames = region(n_frames * N_FFT);
        let re = region(n_frames * BINS_PAD);
        let im = region(n_frames * BINS_PAD);
        let mel = region(n_frames * n_mel);
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

/// Write the samples into a freshly allocated front-end buffer.
pub(crate) fn stage_samples(buf: &mut [u8], so: &MelScratch, samples: &[f32]) {
    buf[so.samples..so.samples + samples.len() * 4].copy_from_slice(bytemuck::cast_slice(samples));
}

/// Emit the framing, DFT, power and filterbank ops for `n_frames` frames of the
/// samples in `buf`. Frame `f` is the `N_FFT` samples starting at `f * HOP_LEN`.
/// The frames overlap (a hop is a third of a frame), which the matmul kernel
/// cannot read as strided rows (it faults the DSP), so they are copied into
/// contiguous rows first.
pub(crate) fn emit_mel<S: OpSink>(
    s: &mut S,
    weights: &S::Buf,
    buf: &S::Buf,
    wo: &MelWeightOffsets,
    so: &MelScratch,
    n_frames: usize,
    n_mel: usize,
) -> Result<(), CeraError> {
    dispatch::copy_view(
        s,
        View::new(
            buf,
            so.samples,
            [N_FFT, n_frames, 1],
            [4, HOP_LEN * 4, n_frames * HOP_LEN * 4],
        ),
        View::new(
            buf,
            so.frames,
            [N_FFT, n_frames, 1],
            [4, N_FFT * 4, n_frames * N_FFT * 4],
        ),
    )?;
    let frames = View::new(
        buf,
        so.frames,
        [N_FFT, n_frames, 1],
        [4, N_FFT * 4, n_frames * N_FFT * 4],
    );
    let plane = |off: usize, width: usize| {
        View::new(
            buf,
            off,
            [width, n_frames, 1],
            [4, width * 4, n_frames * width * 4],
        )
    };
    let basis = |off: usize, rows: usize, cols: usize| {
        View::new(
            weights,
            off,
            [cols, rows, 1],
            [4, cols * 4, rows * cols * 4],
        )
    };
    dispatch::matmul_f32(
        s,
        basis(wo.dft_re, BINS_PAD, N_FFT),
        frames,
        plane(so.re, BINS_PAD),
    )?;
    dispatch::matmul_f32(
        s,
        basis(wo.dft_im, BINS_PAD, N_FFT),
        frames,
        plane(so.im, BINS_PAD),
    )?;
    let shape = TokenShape {
        dim: BINS_PAD,
        n_tokens: n_frames,
    };
    // power = re^2 + im^2, in place in the real buffer.
    dispatch::mul_inplace(s, buf, so.re, buf, so.re, shape, TokenTile::Whole)?;
    dispatch::mul_inplace(s, buf, so.im, buf, so.im, shape, TokenTile::Whole)?;
    dispatch::add_residual(s, buf, so.re, buf, so.im, shape, TokenTile::Whole)?;
    dispatch::matmul_f32(
        s,
        basis(wo.filters, n_mel, BINS_PAD),
        plane(so.re, BINS_PAD),
        plane(so.mel, n_mel),
    )
}

/// From the DSP's linear mel energies (time-major `[n_frames, n_mel]`) to the
/// normalized log-mel the conv stem reads: natural log with the CPU path's
/// floor, then the shared per-feature normalization.
pub(crate) fn finish(
    energies: &[f32],
    n_mel: usize,
    n_frames: usize,
    n_samples_in: usize,
) -> Vec<f32> {
    let mut mel_major = vec![0.0f32; n_mel * n_frames];
    for (ti, row) in energies.chunks_exact(n_mel).enumerate() {
        for (mi, &e) in row.iter().enumerate() {
            // f64 like the CPU path; the DSP's sums are never negative.
            mel_major[mi * n_frames + ti] = (e as f64 + LOG_MEL_EPS as f64).ln() as f32;
        }
    }
    finish_log_mel(mel_major, n_mel, n_frames, n_samples_in)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::hexagon::HtpOpCode::*;
    use crate::backend::hexagon::dispatch::testing::RecordingSink;
    use crate::model::audio_preprocessor::{
        log_mel_spectrogram, n_frames_for, padded_preemphasized,
    };

    fn speechlike(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                (t * 440.0 * std::f32::consts::TAU).sin() * 0.3
                    + (t * 1830.0 * std::f32::consts::TAU).sin() * 0.1 * (1.0 + (t * 3.0).sin())
            })
            .collect()
    }

    /// The matrices and the host-side finish reproduce the CPU log-mel: this is
    /// the DSP's arithmetic done in `f64` on the host, so it pins the basis,
    /// the filterbank padding, the framing and the finish without a device.
    #[test]
    fn dft_and_filter_matrices_reproduce_the_cpu_log_mel() {
        let n_mel = 128;
        let pcm = speechlike(16_000 * 2 + 77);
        let samples = padded_preemphasized(&pcm).unwrap();
        let n_frames = n_frames_for(pcm.len());
        let (re, im) = dft_matrices();
        let filters = padded_filters(n_mel);
        let mut energies = vec![0.0f32; n_frames * n_mel];
        let mut power = vec![0.0f64; BINS_PAD];
        for f in 0..n_frames {
            let frame = &samples[f * HOP_LEN..f * HOP_LEN + N_FFT];
            for (k, p) in power.iter_mut().enumerate() {
                let dot = |m: &[f32]| -> f64 {
                    m[k * N_FFT..(k + 1) * N_FFT]
                        .iter()
                        .zip(frame)
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum()
                };
                let (r, i) = (dot(&re), dot(&im));
                *p = r * r + i * i;
            }
            for mi in 0..n_mel {
                energies[f * n_mel + mi] = filters[mi * BINS_PAD..(mi + 1) * BINS_PAD]
                    .iter()
                    .zip(&power)
                    .map(|(&w, &p)| w as f64 * p)
                    .sum::<f64>() as f32;
            }
        }
        let got = finish(&energies, n_mel, n_frames, pcm.len());
        let (want, want_frames) = log_mel_spectrogram(&pcm, n_mel);
        assert_eq!(want_frames, n_frames);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 5e-3, "max |diff| {worst}");
    }

    #[test]
    fn padding_rows_of_the_matrices_are_zero() {
        let (re, im) = dft_matrices();
        assert!(re[N_FFT_BINS * N_FFT..].iter().all(|&v| v == 0.0));
        assert!(im[N_FFT_BINS * N_FFT..].iter().all(|&v| v == 0.0));
        // DC has no imaginary part and Nyquist's is zero too.
        assert!(im[..N_FFT].iter().all(|&v| v == 0.0));
        assert!(
            im[(N_FFT / 2) * N_FFT..(N_FFT / 2 + 1) * N_FFT]
                .iter()
                .all(|&v| v.abs() < 1e-6)
        );
        let filters = padded_filters(128);
        assert!(
            filters
                .as_chunks::<BINS_PAD>()
                .0
                .iter()
                .all(|row| row[N_FFT_BINS..].iter().all(|&v| v == 0.0))
        );
    }

    #[test]
    fn the_front_end_frames_then_emits_two_dfts_a_power_and_the_filterbank() {
        let (n_frames, n_mel) = (101, 128);
        let so = MelScratch::new(16_000 + 512, n_frames, n_mel);
        let wo = MelWeightOffsets {
            dft_re: 0,
            dft_im: 1 << 20,
            filters: 2 << 20,
        };
        let mut s = RecordingSink::default();
        emit_mel(&mut s, &"w", &"b", &wo, &so, n_frames, n_mel).unwrap();
        assert_eq!(
            s.opcodes(),
            [Cpy, MulMat, MulMat, Mul, Mul, Add, MulMat].map(|o| o as u32)
        );
        // The framing copy reads overlapping rows one hop apart; the matmuls
        // read the contiguous result.
        let src = s.src(0, 0);
        assert_eq!((src.ne[0], src.ne[1]), (N_FFT as u32, n_frames as u32));
        assert_eq!(src.nb[1], (HOP_LEN * 4) as u32);
        assert_eq!(s.dst(0).nb[1], (N_FFT * 4) as u32);
        assert_eq!(
            s.src(1, 1).nb[1],
            (N_FFT * 4) as u32,
            "matmul rows do not overlap"
        );
    }
}
