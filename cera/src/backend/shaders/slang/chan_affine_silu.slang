// Per-channel affine followed by SiLU, in place over a channel-major
// `[channels][t]` buffer:
//
//   x[c][i] = silu(x[c][i] * w[c] + b[c])
//
// This is the `conv_norm` step inside the Conformer convolution module. Despite
// the tensor names, `conv_norm_w`/`conv_norm_b` are **not** a LayerNorm: there
// is no mean or variance, just a per-channel scale and shift broadcast along
// time (`cpu::conformer_conv_module_forward` step 5). The existing `bias_add`
// broadcasts along the *last* axis of a row-major buffer, which is the opposite
// axis here, so this cannot reuse it.
//
// Fusing the affine and the activation keeps the conv module at one dispatch for
// the pair, and matches how the CPU walks each channel's contiguous time slice
// once.
//
//   binding/buffer 0: x       f32, read-write, in place [channels][t]
//   binding/buffer 1: w       f32, read  [channels]
//   binding/buffer 2: b       f32, read  [channels]
//   binding/buffer 3: params  (channels, t, _, _)
//
// Dispatch: one thread per element, ceil(channels*t / 256) workgroups of 256.

[[vk::binding(0)]] RWStructuredBuffer<float> x_buf   : register(u0);
[[vk::binding(1)]] StructuredBuffer<float>   w_buf   : register(t1);
[[vk::binding(2)]] StructuredBuffer<float>   b_buf   : register(t2);
[[vk::binding(3)]] StructuredBuffer<uint4>   par_buf : register(t3);

[shader("compute")]
[numthreads(256, 1, 1)]
void chan_affine_silu(uint3 gid : SV_DispatchThreadID) {
    uint channels = par_buf[0].x;
    uint t = par_buf[0].y;

    uint total = channels * t;
    uint idx = gid.x;
    if (idx >= total) {
        return;
    }

    uint c = idx / t;
    float v = clamp(x_buf[idx] * w_buf[c] + b_buf[c], -80.0f, 80.0f);
    x_buf[idx] = v / (1.0f + exp(-v));
}
