// Framing pass of the LFM2A log-mel front-end: turn a mono PCM chunk into the
// `[n_frames][n_fft]` buffer `power_spec` transforms.
//
//   frames[t][n] = hann[n] * padded_preemph[t * hop + n]
//
// Three steps of `audio_preprocessor::log_mel_spectrogram` fold into that
// gather, and none of them needs a materialized intermediate:
//
// - **Center padding.** The CPU reference materializes a copy of the signal with
//   `center_pad` zeros on each side (librosa `center=True`). Nothing is
//   materialized here, on either side of the dispatch: the caller uploads the
//   raw PCM, and the padding is a bounds test on the un-padded index
//   `g = t * hop + n - center_pad`, zero outside `[0, n_samples)`. A port that
//   pads the PCM before uploading it has missed the point of this kernel.
// - **Pre-emphasis**, `y[i] = x[i] - preemph * x[i-1]`. Stateless, so each
//   thread recomputes its own tap from two PCM reads instead of reading a
//   filtered copy. `i = 0` is left untouched, matching the C++ reference (which
//   writes back over `samples[pad+1 .. pad+n_samples]`) and the CPU path.
//   Note the reference reads the *original* `x[i-1]`, not the filtered one, so
//   there is no recurrence to serialize.
// - **The Hann window**, uploaded already zero-padded from `window_len` up to
//   `n_fft` (`audio_preprocessor::build_padded_hann_window`), so this kernel does
//   not need to know where inside the buffer the window sits.
//
//   binding/buffer 0: pcm      f32, read  [n_samples], un-padded
//   binding/buffer 1: hann     f32, read  [n_fft], zero-padded to full width
//   binding/buffer 2: frames   f32, write [n_frames][n_fft]
//   binding/buffer 3: params   (n_frames, n_fft, hop, center_pad), (n_samples, preemph_bits, _, _)
//
// Dispatch: one thread per output tap, ceil(n_frames*n_fft / 256) workgroups of 256.

[[vk::binding(0)]] StructuredBuffer<float>   pcm_buf    : register(t0);
[[vk::binding(1)]] StructuredBuffer<float>   hann_buf   : register(t1);
[[vk::binding(2)]] RWStructuredBuffer<float> frames_buf : register(u2);
[[vk::binding(3)]] StructuredBuffer<uint4>   par_buf    : register(t3);

[shader("compute")]
[numthreads(256, 1, 1)]
void stft_frame(uint3 gid : SV_DispatchThreadID) {
    uint n_frames   = par_buf[0].x;
    uint n_fft      = par_buf[0].y;
    uint hop        = par_buf[0].z;
    uint center_pad = par_buf[0].w;
    uint n_samples  = par_buf[1].x;
    float preemph   = asfloat(par_buf[1].y);

    uint total = n_frames * n_fft;
    uint idx = gid.x;
    if (idx >= total) {
        return;
    }

    uint t = idx / n_fft;
    uint n = idx - t * n_fft;

    // Index into the un-padded PCM. Signed: the first `center_pad` taps of
    // frame 0 sit before the signal starts.
    int g = int(t * hop + n) - int(center_pad);

    float s = 0.0f;
    if (g >= 0 && g < int(n_samples)) {
        s = pcm_buf[g];
        if (g > 0) {
            s -= preemph * pcm_buf[g - 1];
        }
    }
    frames_buf[idx] = hann_buf[n] * s;
}
