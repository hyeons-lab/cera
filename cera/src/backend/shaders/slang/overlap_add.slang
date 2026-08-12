// Windowed overlap-add, the final GPU ISTFT stage.
//
// Each of `n_frames` time-domain frames (length `n_fft`, produced by the iDFT
// matmul) is Hann-windowed and laid down at global offset `frame·hop`, and the
// overlaps are summed and normalized by the accumulated window energy. This is
// the parallel, position-indexed form of the CPU `istft_to_pcm` shift-accumulate
// loop: one thread per output sample `g` sums only the frames that cover it.
//
// For output position `g` the covering frames are those with
// `i·hop <= g < i·hop + n_fft`, i.e. `i in [i_lo, i_hi]` with
// `i_hi = min(g / hop, n_frames - 1)` and `i_lo = (g >= n_fft) ? (g - n_fft)/hop
// + 1 : 0`. Over that range `local = g - i·hop` is always in `[0, n_fft)`.
//
// Normalization matches the CPU exactly, including the tail where fewer frames
// overlap: `out[g] = Σ td[i][local]·hann[local] / Σ hann[local]²` (falling back
// to the unnormalized numerator when the window energy is ~0). The startup pad
// of `(n_fft - hop)/2` samples is stripped on the CPU after readback.
//
//   binding 0: time_domain  f32, read  (n_frames · n_fft)
//   binding 1: hann         f32, read  (n_fft)
//   binding 2: out          f32, write (n_frames · hop)
//   binding 3: params       (n_frames, n_fft, hop, _)
//
// Dispatch: one thread per output sample, ceil(n_frames·hop / 256) groups of 256.

[[vk::binding(0)]] StructuredBuffer<float>   td_buf   : register(t0);
[[vk::binding(1)]] StructuredBuffer<float>   hann_buf : register(t1);
[[vk::binding(2)]] RWStructuredBuffer<float> out_buf  : register(u2);
[[vk::binding(3)]] StructuredBuffer<uint4>   par_buf  : register(t3);

[shader("compute")]
[numthreads(256, 1, 1)]
void overlap_add(uint3 gid : SV_DispatchThreadID) {
    uint n_frames = par_buf[0].x;
    uint n_fft = par_buf[0].y;
    uint hop = par_buf[0].z;
    uint total = n_frames * hop;
    uint g = gid.x;
    if (g >= total) {
        return;
    }

    uint i_hi = g / hop;
    if (i_hi >= n_frames) {
        i_hi = n_frames - 1u;
    }
    uint i_lo = 0u;
    if (g >= n_fft) {
        i_lo = (g - n_fft) / hop + 1u;
    }

    float numer = 0.0f;
    float denom = 0.0f;
    for (uint i = i_lo; i <= i_hi; i++) {
        uint local = g - i * hop;
        float w = hann_buf[local];
        numer += td_buf[i * n_fft + local] * w;
        denom += w * w;
    }
    out_buf[g] = (denom > 1e-8f) ? (numer / denom) : numer;
}
