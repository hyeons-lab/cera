// slang-entries: conv1d_depthwise
//
// Depthwise Conv1d with a rolling buffer, one thread per channel:
//
//   output[ch] = sum_k rbuffer[k, ch] * weight[ch, k] + input[ch] * weight[ch, d_conv]
//
// then the rolling buffer shifts left one slot and `input[ch]` is appended.
//
// One Slang source for both GPU backends, ported from the handwritten pair it
// replaces (`shaders/conv1d.wgsl`, `shaders/conv1d.metal`), preserving both
// contracts exactly so no call site changes. The two twins were already
// identical modulo syntax (same f32 element type, same five bindings, same entry
// name) and there is no reduction or subgroup op here, so the whole body is
// written once with no `__target_switch`.
//
//   binding/buffer 0: input    f32, read, bx = b * x from in_proj [hidden_size]
//   binding/buffer 1: rbuffer  f32, read-write, rolling buffer [d_conv x hidden_size]
//   binding/buffer 2: weight   f32, read, conv weights [hidden_size x kernel_size]
//   binding/buffer 3: output   f32, read-write, conv output [hidden_size]
//   binding/buffer 4: params   (hidden_size, kernel_size, d_conv, 0)
//
// Dispatch: (ceil(hidden_size / 256), 1, 1) workgroups of 256 threads.
//
// `[[vk::binding(n)]]` drives the WGSL binding index and `register(t/u N)` the
// Metal buffer index; they are numbered independently because the two backends
// count slots from different spaces, and here they agree.

[[vk::binding(0)]] StructuredBuffer<float>   in_buf  : register(t0);
[[vk::binding(1)]] RWStructuredBuffer<float> rbuf    : register(u1);
[[vk::binding(2)]] StructuredBuffer<float>   w_buf   : register(t2);
[[vk::binding(3)]] RWStructuredBuffer<float> out_buf : register(u3);
[[vk::binding(4)]] StructuredBuffer<uint>    par_buf : register(t4);

[shader("compute")]
[numthreads(256, 1, 1)]
void conv1d_depthwise(uint3 gid : SV_DispatchThreadID) {
    uint ch     = gid.x;
    uint hs     = par_buf[0];
    uint ks     = par_buf[1];
    uint d_conv = par_buf[2];

    if (ch >= hs) {
        return;
    }

    // Convolution over the rolling buffer plus the current input.
    float sum = 0.0f;
    for (uint k = 0u; k < d_conv; ++k) {
        sum += rbuf[k * hs + ch] * w_buf[ch * ks + k];
    }
    sum += in_buf[ch] * w_buf[ch * ks + d_conv];
    out_buf[ch] = sum;

    // Roll the buffer left one slot and append the current input. The shift is
    // sequential per channel, but channels are independent so no barrier is
    // needed.
    if (d_conv > 1u) {
        for (uint k = 0u; k < d_conv - 1u; ++k) {
            rbuf[k * hs + ch] = rbuf[(k + 1u) * hs + ch];
        }
    }
    if (d_conv > 0u) {
        rbuf[(d_conv - 1u) * hs + ch] = in_buf[ch];
    }
}
