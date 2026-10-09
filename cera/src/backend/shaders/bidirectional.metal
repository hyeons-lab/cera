#include <metal_stdlib>
using namespace metal;

// Kernels for running an LFM2 trunk bidirectionally (the d1 decision models).
//
// The causal prefill keeps a rolling window of the last two inputs per channel, which only
// looks backwards. A bidirectional trunk reads the whole sequence at once, so its gated
// convolution is a centred 3-tap filter over the already projected rows instead.

struct BidirConvParams {
    uint n;           // rows
    uint hs;          // channels
    uint prefix;      // rows below this are a media prefix; 0 for none
    uint proj_stride; // floats between rows of `proj` (3 * hs: B, C, x side by side)
    uint out_stride;  // floats between rows of `out`
};

// out[t] = C[t] * (B*x[t-1] * w0 + B*x[t] * w1 + B*x[t+1] * w2), zero padded at both ends.
//
// The window never crosses the end of a media prefix: the last prefix row does not read the
// first text row (a prefix is a function of the media alone), while the first text row does
// read the last prefix row. One thread per (row, channel); `w` is [hs x 3].
kernel void bidir_conv_centered(
    const device float* proj [[buffer(0)]],
    const device float* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant BidirConvParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.n * p.hs) return;
    const uint t = gid / p.hs;
    const uint ch = gid % p.hs;
    const device float* row = proj + t * p.proj_stride;
    const float cur = row[ch] * row[2u * p.hs + ch];
    float prev = 0.0f;
    if (t > 0u) {
        const device float* before = row - p.proj_stride;
        prev = before[ch] * before[2u * p.hs + ch];
    }
    float next = 0.0f;
    if (t + 1u < p.n && t + 1u != p.prefix) {
        const device float* after = row + p.proj_stride;
        next = after[ch] * after[2u * p.hs + ch];
    }
    const float c = row[p.hs + ch];
    out[t * p.out_stride + ch] = c * (prev * w[ch * 3u] + cur * w[ch * 3u + 1u] + next * w[ch * 3u + 2u]);
}
