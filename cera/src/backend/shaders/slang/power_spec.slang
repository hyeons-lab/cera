// Power spectrum of every STFT frame, as a direct DFT:
//
//   power[t][k] = |sum_n frames[t][n] * exp(-2*pi*i*k*n/n_fft)|^2
//
// for the `n_bins = n_fft/2 + 1` unique bins of a real input. Replaces the CPU
// path's rustfft call, which has no GPU counterpart.
//
// ## Why a direct DFT and not an FFT
//
// This is O(n_fft) taps per bin where a radix-2 FFT is O(log n_fft): 512 vs 9
// for LFM2A. It is still the right trade here. The Conformer body behind this
// kernel is roughly two orders of magnitude more arithmetic per frame, so the
// asymptotic loss does not show up in the pipeline, and a groupshared butterfly
// with its bit-reversal and inter-stage barriers is a much larger correctness
// surface than a fused-multiply-add loop.
//
// ## Why not the crate's other GPU transform
//
// The audio *de*tokenizer already does an inverse transform on the GPU, and does
// it differently: `audio_decoder::build_idft_basis` precomputes a real
// `[n_fft x 2*n_fft_bins]` matrix and each backend runs it through its existing
// GEMM (`wgpu_audio_decoder.rs`). That form does not transfer here. It works
// there because folding the Hermitian mirror and the `1/n_fft` scale into the
// coefficients turns an awkward non-power-of-two iDFT into one GEMM against a
// matrix small enough to be a weight. Forward, the equivalent basis is ~1 MB of
// f32 against the 4 KB table below, and it would have to be dressed up as an
// `MmapWeight` to reach the `linear` op at all. Same primitive, different
// constraints; this is not an oversight.
//
// ## The twiddle table
//
// The angle depends only on `(k*n) mod n_fft`, so a table of `n_fft` entries
// (cos, sin interleaved) covers every `(k, n)` pair: 4 KB for LFM2A, resident in
// L1. Two things follow. The inner loop's per-tap cost becomes three loads (the
// sample and the two twiddle components) and two FMAs, rather than two
// transcendentals; and the twiddles are computed in f64 on the host, so they are
// more accurate than an in-shader `cos()` of a large argument would be.
//
// `m` tracks `(k*n) mod n_fft` incrementally. The conditional subtract is valid
// because `k <= n_fft/2 < n_fft`, so one subtraction always suffices.
//
//   binding/buffer 0: frames   f32, read  [n_frames][n_fft]
//   binding/buffer 1: twiddle  f32, read  [n_fft][2], cos/sin of -2*pi*m/n_fft
//   binding/buffer 2: power    f32, write [n_frames][n_bins]
//   binding/buffer 3: params   (n_frames, n_fft, n_bins, _)
//
// Dispatch: one thread per output bin, ceil(n_frames*n_bins / 256) workgroups of 256.

[[vk::binding(0)]] StructuredBuffer<float>   frames_buf  : register(t0);
[[vk::binding(1)]] StructuredBuffer<float>   twiddle_buf : register(t1);
[[vk::binding(2)]] RWStructuredBuffer<float> power_buf   : register(u2);
[[vk::binding(3)]] StructuredBuffer<uint4>   par_buf     : register(t3);

[shader("compute")]
[numthreads(256, 1, 1)]
void power_spec(uint3 gid : SV_DispatchThreadID) {
    uint n_frames = par_buf[0].x;
    uint n_fft    = par_buf[0].y;
    uint n_bins   = par_buf[0].z;

    uint total = n_frames * n_bins;
    uint idx = gid.x;
    if (idx >= total) {
        return;
    }

    uint t = idx / n_bins;
    uint k = idx - t * n_bins;
    uint base = t * n_fft;

    float re = 0.0f;
    float im = 0.0f;
    uint m = 0u;
    for (uint n = 0u; n < n_fft; ++n) {
        float x = frames_buf[base + n];
        re += x * twiddle_buf[2u * m];
        im += x * twiddle_buf[2u * m + 1u];
        m += k;
        if (m >= n_fft) {
            m -= n_fft;
        }
    }
    power_buf[idx] = re * re + im * im;
}
