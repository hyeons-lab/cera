// Streaming Q4_K (Q4_K_M) prefill GEMM: dst[n, m] = B[n, k] @ W[m, k]^T.
//
// Streaming fp16 fiber accumulators over resident Q4_K layout (src_q and src_d).
// Each fiber (thread) owns 1 row x 32 columns in 256-thread workgroups.
// The B tile [32 k][8 groups] is cooperatively staged to shmem once per 32-k slice
// and broadcast as vector half4s.
//
// Resident Q4_K layout (host repacked once at upload via repack_q4_k_stream):
// - src_q: 32 nibbles per 32-block packed into 4 u32s, row-consecutive:
//   src_q[(b*4 + p)*m + row]
// - src_d: 1 u32 per 32-block holding (half scale | half min << 16), row-consecutive:
//   src_d[b*m + row]
//
// In the inner loop, each thread loads scale and min as halfs from src_d and
// evaluates `half w = half(nib) * scale - min`, streaming weights coalesced
// directly from global memory.

struct StreamParams {
    uint m;
    uint k;
    uint n_valid;
    uint n_pad;
    uint y_stride;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>         src_q;
[[vk::binding(1, 0)]] StructuredBuffer<uint>         src_d;
[[vk::binding(2, 0)]] StructuredBuffer<half>         src_b;
[[vk::binding(3, 0)]] RWStructuredBuffer<float>      dst;
[[vk::binding(4, 0)]] StructuredBuffer<StreamParams> paramsBuf;

groupshared half4 sh_b[32 * 8];

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

    uint n_kb = P.k / 32u;
    for (uint kb = 0u; kb < n_kb; kb++) {
        // Stage the B tile [32 k][8 groups] to shmem as half4s
        for (uint e = lid; e < 256u; e += 256u) {
            uint kk = e / 8u;
            uint g = e % 8u;
            uint b = (kb * 32u + kk) * P.n_pad + col_base + g * 4u;
            sh_b[kk * 8u + g] =
                half4(src_b[b], src_b[b + 1u], src_b[b + 2u], src_b[b + 3u]);
        }
        GroupMemoryBarrierWithGroupSync();

        // One scale/min pair per 32-block packed into u32 (scale in low 16, min in high 16)
        uint pd = src_d[kb * P.m + rrow];
        half scale = half(f16tof32(pd & 0xFFFFu));
        half minv = half(f16tof32(pd >> 16u));

        // 4 u32s per 32-block, each holding 8 nibbles
        for (uint p = 0u; p < 4u; p++) {
            uint pq = src_q[(kb * 4u + p) * P.m + rrow];
            uint w4_lo = pq & 0xFFFFu;
            uint w4_hi = pq >> 16u;
            uint kl0 = p * 8u;

            // t=0: ushort lo, nibble 0
            {
                uint nib = w4_lo & 0xFu;
                half w = half(nib) * scale - minv;
                uint b = kl0 * 8u;
                g0 += sh_b[b] * w;
                g1 += sh_b[b + 1u] * w;
                g2 += sh_b[b + 2u] * w;
                g3 += sh_b[b + 3u] * w;
                g4 += sh_b[b + 4u] * w;
                g5 += sh_b[b + 5u] * w;
                g6 += sh_b[b + 6u] * w;
                g7 += sh_b[b + 7u] * w;
            }
            // t=1: ushort lo, nibble 1
            {
                uint nib = (w4_lo >> 4u) & 0xFu;
                half w = half(nib) * scale - minv;
                uint b = kl0 * 8u + 8u;
                g0 += sh_b[b] * w;
                g1 += sh_b[b + 1u] * w;
                g2 += sh_b[b + 2u] * w;
                g3 += sh_b[b + 3u] * w;
                g4 += sh_b[b + 4u] * w;
                g5 += sh_b[b + 5u] * w;
                g6 += sh_b[b + 6u] * w;
                g7 += sh_b[b + 7u] * w;
            }
            // t=2: ushort lo, nibble 2
            {
                uint nib = (w4_lo >> 8u) & 0xFu;
                half w = half(nib) * scale - minv;
                uint b = kl0 * 8u + 16u;
                g0 += sh_b[b] * w;
                g1 += sh_b[b + 1u] * w;
                g2 += sh_b[b + 2u] * w;
                g3 += sh_b[b + 3u] * w;
                g4 += sh_b[b + 4u] * w;
                g5 += sh_b[b + 5u] * w;
                g6 += sh_b[b + 6u] * w;
                g7 += sh_b[b + 7u] * w;
            }
            // t=3: ushort lo, nibble 3
            {
                uint nib = (w4_lo >> 12u) & 0xFu;
                half w = half(nib) * scale - minv;
                uint b = kl0 * 8u + 24u;
                g0 += sh_b[b] * w;
                g1 += sh_b[b + 1u] * w;
                g2 += sh_b[b + 2u] * w;
                g3 += sh_b[b + 3u] * w;
                g4 += sh_b[b + 4u] * w;
                g5 += sh_b[b + 5u] * w;
                g6 += sh_b[b + 6u] * w;
                g7 += sh_b[b + 7u] * w;
            }
            // t=4: ushort hi, nibble 0
            {
                uint nib = w4_hi & 0xFu;
                half w = half(nib) * scale - minv;
                uint b = kl0 * 8u + 32u;
                g0 += sh_b[b] * w;
                g1 += sh_b[b + 1u] * w;
                g2 += sh_b[b + 2u] * w;
                g3 += sh_b[b + 3u] * w;
                g4 += sh_b[b + 4u] * w;
                g5 += sh_b[b + 5u] * w;
                g6 += sh_b[b + 6u] * w;
                g7 += sh_b[b + 7u] * w;
            }
            // t=5: ushort hi, nibble 1
            {
                uint nib = (w4_hi >> 4u) & 0xFu;
                half w = half(nib) * scale - minv;
                uint b = kl0 * 8u + 40u;
                g0 += sh_b[b] * w;
                g1 += sh_b[b + 1u] * w;
                g2 += sh_b[b + 2u] * w;
                g3 += sh_b[b + 3u] * w;
                g4 += sh_b[b + 4u] * w;
                g5 += sh_b[b + 5u] * w;
                g6 += sh_b[b + 6u] * w;
                g7 += sh_b[b + 7u] * w;
            }
            // t=6: ushort hi, nibble 2
            {
                uint nib = (w4_hi >> 8u) & 0xFu;
                half w = half(nib) * scale - minv;
                uint b = kl0 * 8u + 48u;
                g0 += sh_b[b] * w;
                g1 += sh_b[b + 1u] * w;
                g2 += sh_b[b + 2u] * w;
                g3 += sh_b[b + 3u] * w;
                g4 += sh_b[b + 4u] * w;
                g5 += sh_b[b + 5u] * w;
                g6 += sh_b[b + 6u] * w;
                g7 += sh_b[b + 7u] * w;
            }
            // t=7: ushort hi, nibble 3
            {
                uint nib = (w4_hi >> 12u) & 0xFu;
                half w = half(nib) * scale - minv;
                uint b = kl0 * 8u + 56u;
                g0 += sh_b[b] * w;
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

    uint c0 = col_base;
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
