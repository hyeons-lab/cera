// Swap the two outer axes of a `[A][B][K]` f32 tensor, keeping the innermost
// `K`-wide block contiguous: `dst[b][a][k] = src[a][b][k]`.
//
// Two call sites in the audio encoder, which is why it is blocked rather than a
// plain 2D transpose:
//
//   - Conv-stem flatten, `K = f_out`: the stem emits `[channel][time][freq]` and
//     `pre_encode_out` consumes `[time][channel*freq]` rows. That is the
//     `permute(0,2,1,3) + reshape` in the C++ reference, and it is exactly this
//     kernel with the freq axis as the inner block.
//   - The Conformer conv module, `K = 1`: time-major `[t][n_embd]` has to become
//     channel-major `[n_embd][t]` for the depthwise conv1d and back again for
//     the second pointwise conv. `K = 1` degenerates to an ordinary 2D
//     transpose, so both directions are this one kernel.
//
//   binding/buffer 0: src     f32, read  [A][B][K]
//   binding/buffer 1: dst     f32, write [B][A][K]
//   binding/buffer 2: params  (A, B, K, _)
//
// Dispatch: one thread per element, ceil(A*B*K / 256) workgroups of 256.

[[vk::binding(0)]] StructuredBuffer<float>   src_buf : register(t0);
[[vk::binding(1)]] RWStructuredBuffer<float> dst_buf : register(u1);
[[vk::binding(2)]] StructuredBuffer<uint4>   par_buf : register(t2);

[shader("compute")]
[numthreads(256, 1, 1)]
void transpose_blocked(uint3 gid : SV_DispatchThreadID) {
    uint a_dim = par_buf[0].x;
    uint b_dim = par_buf[0].y;
    uint k_dim = par_buf[0].z;

    uint bk = b_dim * k_dim;
    uint total = a_dim * bk;
    uint idx = gid.x;
    if (idx >= total) {
        return;
    }

    uint a = idx / bk;
    uint rem = idx - a * bk;
    uint b = rem / k_dim;
    uint k = rem - b * k_dim;

    dst_buf[(b * a_dim + a) * k_dim + k] = src_buf[idx];
}
