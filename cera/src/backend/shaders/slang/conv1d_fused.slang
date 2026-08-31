// slang-entries: conv1d_fused
//
// Fast single-token fused conv block for the LFM2 short-conv path, one thread per channel:
//
//   bx  = x * b
//   sum = sum_k rbuffer[k, ch] * weight[ch, k] + bx * weight[ch, d_conv]
//   rbuffer shifts left one slot and bx is appended
//   output[ch] = c * sum
//
//   binding/buffer 0: proj     f32, read, packed [x | c | b], hs floats each
//   binding/buffer 1: rbuffer  f32, read-write, rolling buffer [d_conv x hs]
//   binding/buffer 2: weight   f32, read, conv weights [hs x kernel_size]
//   binding/buffer 3: output   f32, read-write, gated conv output [hs]
//   binding/buffer 4: params   (hs, kernel_size, d_conv, 0)

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

    float x_val = proj_buf[ch];
    float c_val = proj_buf[hs + ch];
    float b_val = proj_buf[2u * hs + ch];
    float bx = x_val * b_val;

    float sum = 0.0f;
    for (uint k = 0u; k < d_conv; ++k) {
        sum += rbuf[k * hs + ch] * w_buf[ch * ks + k];
    }
    sum += bx * w_buf[ch * ks + d_conv];

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
