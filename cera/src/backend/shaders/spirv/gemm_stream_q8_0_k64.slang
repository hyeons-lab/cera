// Streaming Q8_0 GEMM, k-slice-64: the 8-bit twin of `gemm_stream_q4_0_k64.slang`, for Q8_0 weights
// (the LFM2-VL vision tower's mmproj). Same 1x32 fiber, grid, params, bindings and entry (`main`);
// only the weight decode and the repack layout differ.
//
// Weight layout (see `repack_q8_0_stream`): `src_q[(k/4) * m + row]` holds the row's weights
// k..k+3 as four little-endian int8 bytes (a 64-k slice is 16 words), and
// `src_d[(k/64) * m + row]` holds half(scale of block 2s) | half(scale of block 2s+1) << 16.
// Needs k % 64 == 0. A 32-k block is 8 words, so word p of a slice belongs to block p / 8.
//
// Arithmetic is fp16 with fp16 accumulators, as the Q4_0 kernels (the activations arrive as f16
// from `transpose_cast_f16`).

struct StreamParams {
    uint m;
    uint k;
    uint n_valid;
    uint n_pad;
    uint y_stride;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint> src_q;
[[vk::binding(1, 0)]] StructuredBuffer<uint> src_d;
[[vk::binding(2, 0)]] StructuredBuffer<half> src_b;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> dst;
[[vk::binding(4, 0)]] StructuredBuffer<StreamParams> paramsBuf;

groupshared half4 sh_b[64 * 8];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lid : SV_GroupThreadID) {
    let P = paramsBuf[0];
    uint row = gid.x * 256u + lid;
    uint col_base = gid.y * 32u;
    uint rrow = row < P.m ? row : 0u;
    bool row_valid = row < P.m;
    half4 g0 = half4(0.0h);
    half4 g1 = half4(0.0h);
    half4 g2 = half4(0.0h);
    half4 g3 = half4(0.0h);
    half4 g4 = half4(0.0h);
    half4 g5 = half4(0.0h);
    half4 g6 = half4(0.0h);
    half4 g7 = half4(0.0h);

    uint n_kb = P.k / 64u;
    for (uint kb = 0u; kb < n_kb; kb++) {
        // Stage the B tile [64 k][8 groups]: 512 half4s, 2 per thread.
        for (uint s = 0u; s < 2u; s++) {
            uint e = lid * 2u + s;
            uint kk = e / 8u;
            uint g = e % 8u;
            uint b = (kb * 64u + kk) * P.n_pad + col_base + g * 4u;
            sh_b[kk * 8u + g] =
                half4(src_b[b], src_b[b + 1u], src_b[b + 2u], src_b[b + 3u]);
        }
        GroupMemoryBarrierWithGroupSync();

        uint pd0 = src_d[kb * P.m + rrow];
        half dw0 = half(f16tof32(pd0 & 0xFFFFu));
        half dw1 = half(f16tof32(pd0 >> 16u));

        for (uint p = 0u; p < 16u; p++) {
            uint pq = src_q[(kb * 16u + p) * P.m + rrow];
            half dw = (p < 8u ? dw0 : dw1);
            // byte 0: weight k = kb*64 + p*4 + 0
            {
                half w = half(int(pq << 24u) >> 24) * dw;
                uint b = (p * 4u + 0u) * 8u;
                g0 += sh_b[b + 0u] * w;
                g1 += sh_b[b + 1u] * w;
                g2 += sh_b[b + 2u] * w;
                g3 += sh_b[b + 3u] * w;
                g4 += sh_b[b + 4u] * w;
                g5 += sh_b[b + 5u] * w;
                g6 += sh_b[b + 6u] * w;
                g7 += sh_b[b + 7u] * w;
            }
            // byte 1: weight k = kb*64 + p*4 + 1
            {
                half w = half(int(pq << 16u) >> 24) * dw;
                uint b = (p * 4u + 1u) * 8u;
                g0 += sh_b[b + 0u] * w;
                g1 += sh_b[b + 1u] * w;
                g2 += sh_b[b + 2u] * w;
                g3 += sh_b[b + 3u] * w;
                g4 += sh_b[b + 4u] * w;
                g5 += sh_b[b + 5u] * w;
                g6 += sh_b[b + 6u] * w;
                g7 += sh_b[b + 7u] * w;
            }
            // byte 2: weight k = kb*64 + p*4 + 2
            {
                half w = half(int(pq << 8u) >> 24) * dw;
                uint b = (p * 4u + 2u) * 8u;
                g0 += sh_b[b + 0u] * w;
                g1 += sh_b[b + 1u] * w;
                g2 += sh_b[b + 2u] * w;
                g3 += sh_b[b + 3u] * w;
                g4 += sh_b[b + 4u] * w;
                g5 += sh_b[b + 5u] * w;
                g6 += sh_b[b + 6u] * w;
                g7 += sh_b[b + 7u] * w;
            }
            // byte 3: weight k = kb*64 + p*4 + 3
            {
                half w = half(int(pq << 0u) >> 24) * dw;
                uint b = (p * 4u + 3u) * 8u;
                g0 += sh_b[b + 0u] * w;
                g1 += sh_b[b + 1u] * w;
                g2 += sh_b[b + 2u] * w;
                g3 += sh_b[b + 3u] * w;
                g4 += sh_b[b + 4u] * w;
                g5 += sh_b[b + 5u] * w;
                g6 += sh_b[b + 6u] * w;
                g7 += sh_b[b + 7u] * w;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (!row_valid) {
        return;
    }
    uint c0 = col_base + 0u;
    if (c0 < P.n_valid) {
        float4 f0 = float4(g0);
        dst[c0 * P.y_stride + row] = f0.x;
        if (c0 + 1u < P.n_valid) { dst[(c0 + 1u) * P.y_stride + row] = f0.y; }
        if (c0 + 2u < P.n_valid) { dst[(c0 + 2u) * P.y_stride + row] = f0.z; }
        if (c0 + 3u < P.n_valid) { dst[(c0 + 3u) * P.y_stride + row] = f0.w; }
    }
    uint c1 = col_base + 4u;
    if (c1 < P.n_valid) {
        float4 f1 = float4(g1);
        dst[c1 * P.y_stride + row] = f1.x;
        if (c1 + 1u < P.n_valid) { dst[(c1 + 1u) * P.y_stride + row] = f1.y; }
        if (c1 + 2u < P.n_valid) { dst[(c1 + 2u) * P.y_stride + row] = f1.z; }
        if (c1 + 3u < P.n_valid) { dst[(c1 + 3u) * P.y_stride + row] = f1.w; }
    }
    uint c2 = col_base + 8u;
    if (c2 < P.n_valid) {
        float4 f2 = float4(g2);
        dst[c2 * P.y_stride + row] = f2.x;
        if (c2 + 1u < P.n_valid) { dst[(c2 + 1u) * P.y_stride + row] = f2.y; }
        if (c2 + 2u < P.n_valid) { dst[(c2 + 2u) * P.y_stride + row] = f2.z; }
        if (c2 + 3u < P.n_valid) { dst[(c2 + 3u) * P.y_stride + row] = f2.w; }
    }
    uint c3 = col_base + 12u;
    if (c3 < P.n_valid) {
        float4 f3 = float4(g3);
        dst[c3 * P.y_stride + row] = f3.x;
        if (c3 + 1u < P.n_valid) { dst[(c3 + 1u) * P.y_stride + row] = f3.y; }
        if (c3 + 2u < P.n_valid) { dst[(c3 + 2u) * P.y_stride + row] = f3.z; }
        if (c3 + 3u < P.n_valid) { dst[(c3 + 3u) * P.y_stride + row] = f3.w; }
    }
    uint c4 = col_base + 16u;
    if (c4 < P.n_valid) {
        float4 f4 = float4(g4);
        dst[c4 * P.y_stride + row] = f4.x;
        if (c4 + 1u < P.n_valid) { dst[(c4 + 1u) * P.y_stride + row] = f4.y; }
        if (c4 + 2u < P.n_valid) { dst[(c4 + 2u) * P.y_stride + row] = f4.z; }
        if (c4 + 3u < P.n_valid) { dst[(c4 + 3u) * P.y_stride + row] = f4.w; }
    }
    uint c5 = col_base + 20u;
    if (c5 < P.n_valid) {
        float4 f5 = float4(g5);
        dst[c5 * P.y_stride + row] = f5.x;
        if (c5 + 1u < P.n_valid) { dst[(c5 + 1u) * P.y_stride + row] = f5.y; }
        if (c5 + 2u < P.n_valid) { dst[(c5 + 2u) * P.y_stride + row] = f5.z; }
        if (c5 + 3u < P.n_valid) { dst[(c5 + 3u) * P.y_stride + row] = f5.w; }
    }
    uint c6 = col_base + 24u;
    if (c6 < P.n_valid) {
        float4 f6 = float4(g6);
        dst[c6 * P.y_stride + row] = f6.x;
        if (c6 + 1u < P.n_valid) { dst[(c6 + 1u) * P.y_stride + row] = f6.y; }
        if (c6 + 2u < P.n_valid) { dst[(c6 + 2u) * P.y_stride + row] = f6.z; }
        if (c6 + 3u < P.n_valid) { dst[(c6 + 3u) * P.y_stride + row] = f6.w; }
    }
    uint c7 = col_base + 28u;
    if (c7 < P.n_valid) {
        float4 f7 = float4(g7);
        dst[c7 * P.y_stride + row] = f7.x;
        if (c7 + 1u < P.n_valid) { dst[(c7 + 1u) * P.y_stride + row] = f7.y; }
        if (c7 + 2u < P.n_valid) { dst[(c7 + 2u) * P.y_stride + row] = f7.z; }
        if (c7 + 3u < P.n_valid) { dst[(c7 + 3u) * P.y_stride + row] = f7.w; }
    }
}

