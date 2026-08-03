// Single-token fused conv block for the LFM2 short-conv path, one thread per
// channel:
//
//   bx  = x * b
//   sum = sum_k rbuffer[k, ch] * weight[ch, k] + bx * weight[ch, d_conv]
//   rbuffer shifts left one slot and bx is appended
//   output[ch] = c * sum
//
// Collapses three decode-time dispatches (bx mul, conv1d, c mul) into one. One
// Slang source for both GPU backends, ported from the handwritten pair it
// replaces (`shaders/conv1d_fused.wgsl`, `shaders/conv1d_fused.metal`).
//
// The Metal twin used to bind x, b and c as three separate buffers, but every
// call site passed the same packed conv-projection buffer three times at offsets
// 0 / 2*hs / hs, so the split carried no information. That twin now reads one
// `proj` buffer at in-shader offsets, matching the WGSL twin and
// `conv1d_fused_batch`. With the binding contracts consolidated the two twins
// agree on element type, bindings and entry name, and there is no reduction or
// subgroup op, so the whole body is written once with no `__target_switch`.
//
//   binding/buffer 0: proj     f32, read, packed [x | c | b], hs floats each
//   binding/buffer 1: rbuffer  f32, read-write, rolling buffer [d_conv x hs]
//   binding/buffer 2: weight   f32, read, conv weights [hs x kernel_size]
//   binding/buffer 3: output   f32, read-write, gated conv output [hs]
//   binding/buffer 4: params   (hs, kernel_size, d_conv, 0)
//
// No kernel-size guard, on purpose. Every index below is dynamic
// (`rbuf[k * hs + ch]`, `w_buf[ch * ks + k]`) with no fixed-size register array
// to overrun, so this kernel is correct for any d_conv. Both handwritten twins
// used to carry a `ks > 4 || d_conv > 3` early-out (the WGSL one did; the Metal
// one never had it); all such a bound can do here is turn a correct result into
// a silently skipped write that leaves the output buffer stale, so it is gone
// from all three. `conv1d_fused_batch` is the one kernel in this tier that
// genuinely needs the bound, for its fixed-size `w_local` / `rb` arrays.
//
// Dispatch: (ceil(hs / 256), 1, 1) workgroups of 256 threads.

[[vk::binding(0)]] StructuredBuffer<float>   proj_buf : register(t0);
[[vk::binding(1)]] RWStructuredBuffer<float> rbuf     : register(u1);
[[vk::binding(2)]] StructuredBuffer<float>   w_buf    : register(t2);
[[vk::binding(3)]] RWStructuredBuffer<float> out_buf  : register(u3);
[[vk::binding(4)]] StructuredBuffer<uint>    par_buf  : register(t4);

[shader("compute")]
[numthreads(256, 1, 1)]
void conv1d_fused(uint3 gid : SV_DispatchThreadID) {
    uint ch     = gid.x;
    uint hs     = par_buf[0];
    uint ks     = par_buf[1];
    uint d_conv = par_buf[2];

    if (ch >= hs) {
        return;
    }

    // Load all three components up front, as both handwritten twins do.
    // Deferring the `c` read to the output write is the shape that measured
    // 0.64x in `conv1d_fused_batch`; the single-token kernel is too
    // launch-bound for the bench to resolve it here, so match the twins.
    float x_val = proj_buf[ch];
    float c_val = proj_buf[hs + ch];
    float b_val = proj_buf[2u * hs + ch];
    float bx = x_val * b_val;

    // Convolution over the rolling buffer plus the current input.
    float sum = 0.0f;
    for (uint k = 0u; k < d_conv; ++k) {
        sum += rbuf[k * hs + ch] * w_buf[ch * ks + k];
    }
    sum += bx * w_buf[ch * ks + d_conv];

    // Roll the buffer left one slot and append bx.
    if (d_conv > 1u) {
        for (uint k = 0u; k < d_conv - 1u; ++k) {
            rbuf[k * hs + ch] = rbuf[(k + 1u) * hs + ch];
        }
    }
    if (d_conv > 0u) {
        rbuf[(d_conv - 1u) * hs + ch] = bx;
    }

    out_buf[ch] = c_val * sum;
}
