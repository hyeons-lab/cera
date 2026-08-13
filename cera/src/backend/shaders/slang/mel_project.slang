// Mel filterbank projection and log floor:
//
//   mel[m][t] = ln(sum_k power[t][k] * filters[m][k] + eps)
//
// The filterbank is the host-built Slaney-scale triangular matrix
// (`audio_preprocessor::build_mel_filterbank`); this kernel only applies it.
//
// Output is **mel-major** `[n_mel][n_frames]`, transposed relative to the
// `[n_frames][n_bins]` input, for the same reason the CPU path builds it that
// way: the per-feature normalization that follows reduces over the time axis,
// and mel-major makes each of its rows contiguous. `mel_norm` transposes back to
// time-major on its way out, so this layout never escapes the front-end.
//
// The dot product runs in f32 where the CPU accumulates in f64. Every term is a
// product of two non-negative numbers (a power spectrum and a triangular
// weight), so the sum has no cancellation and its relative error stays at the
// `n_bins * eps` level rather than growing with the dynamic range.
//
//   binding/buffer 0: power    f32, read  [n_frames][n_bins]
//   binding/buffer 1: filters  f32, read  [n_mel][n_bins]
//   binding/buffer 2: mel      f32, write [n_mel][n_frames], mel-major
//   binding/buffer 3: params   (n_mel, n_frames, n_bins, eps_bits)
//
// Dispatch: one thread per output element, ceil(n_mel*n_frames / 256) workgroups of 256.

[[vk::binding(0)]] StructuredBuffer<float>   power_buf   : register(t0);
[[vk::binding(1)]] StructuredBuffer<float>   filter_buf  : register(t1);
[[vk::binding(2)]] RWStructuredBuffer<float> mel_buf     : register(u2);
[[vk::binding(3)]] StructuredBuffer<uint4>   par_buf     : register(t3);

[shader("compute")]
[numthreads(256, 1, 1)]
void mel_project(uint3 gid : SV_DispatchThreadID) {
    uint n_mel    = par_buf[0].x;
    uint n_frames = par_buf[0].y;
    uint n_bins   = par_buf[0].z;
    float eps     = asfloat(par_buf[0].w);

    uint total = n_mel * n_frames;
    uint idx = gid.x;
    if (idx >= total) {
        return;
    }

    // `idx` is already the mel-major output index: mi * n_frames + t.
    uint mi = idx / n_frames;
    uint t = idx - mi * n_frames;

    uint p_base = t * n_bins;
    uint f_base = mi * n_bins;
    float sum = 0.0f;
    for (uint k = 0u; k < n_bins; ++k) {
        sum += power_buf[p_base + k] * filter_buf[f_base + k];
    }
    mel_buf[idx] = log(sum + eps);
}
