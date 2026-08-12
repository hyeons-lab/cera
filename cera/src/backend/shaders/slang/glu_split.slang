// Batched GLU split for the Conformer convolution module:
//
//   dst[r][c] = src[r][c] * sigmoid(src[r][n + c])
//
// The first pointwise conv doubles the channel count; the first half is the
// value and the second half is the gate. Mirrors `cpu::glu_split`, which runs a
// single `[2n]` row at a time. Here every row of the `[rows][2n]` batch is one
// dispatch, since the GPU encoder projects the whole time axis at once.
//
// The gate uses the hardware `exp`, not the `ggml_expf` polynomial the CPU path
// uses. That is the same substitution every other ported activation makes (see
// `elementwise.slang`'s `silu_mul_inplace`), and it is the dominant term in the
// GPU-vs-CPU tolerance for the conv module.
//
//   binding/buffer 0: src     f32, read  [rows][2*n]
//   binding/buffer 1: dst     f32, write [rows][n]
//   binding/buffer 2: params  (rows, n, _, _)
//
// Dispatch: one thread per output element, ceil(rows*n / 256) workgroups of 256.

[[vk::binding(0)]] StructuredBuffer<float>   src_buf : register(t0);
[[vk::binding(1)]] RWStructuredBuffer<float> dst_buf : register(u1);
[[vk::binding(2)]] StructuredBuffer<uint4>   par_buf : register(t2);

[shader("compute")]
[numthreads(256, 1, 1)]
void glu_split(uint3 gid : SV_DispatchThreadID) {
    uint rows = par_buf[0].x;
    uint n = par_buf[0].y;

    uint total = rows * n;
    uint idx = gid.x;
    if (idx >= total) {
        return;
    }

    uint r = idx / n;
    uint c = idx - r * n;
    uint base = r * 2u * n;

    float value = src_buf[base + c];
    float gate = src_buf[base + n + c];
    dst_buf[idx] = value / (1.0f + exp(-gate));
}
