// Streaming Q4_0 prefill GEMM, k-slice-64 twin of `gemm_stream_q4_0.slang`:
// same 1x32 fiber, same grid/params/bindings/entry (`main`), but each k-slice
// covers 64 k (8 q-words, 2 scales) instead of 32, halving stage+barrier
// pairs. Dispatched when k % 64 == 0 (all LFM2/Qwen shapes); k % 32 == 0 but
// not % 64 falls back to the k-slice-32 kernel. B shmem is 4 KB.
//
// Measured on Adreno 830 (`cera gemm-bench`): +28-33% over the k-slice-32
// kernel at n >= 64 (2.73 vs 2.05 TFLOP/s at 10752x512x2048), tied at n=32.
// k-slice-128 was tried and loses to 64 (8 KB shmem + occupancy cost beats
// the barrier saving); 1x64 fibers lose 3x (shmem broadcast pressure).
// Bit-exact vs the k-slice-32 kernel on all benched shapes.
//
// The p-loop body is the baseline's t=0..7 unroll with `half dw` selected
// per p (p*8..p*8+7 spans one 32-block, so block = p/4); d word j covers
// blocks 2j,2j+1, so slice kb reads the single word j=kb.

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

        uint pd0 = src_d[(kb * 1u + 0u) * P.m + rrow];
        half dw0 = half(f16tof32(pd0 & 0xFFFFu));
        half dw1 = half(f16tof32(pd0 >> 16u));

        for (uint p = 0u; p < 8u; p++) {
            uint pq = src_q[(kb * 8u + p) * P.m + rrow];
            uint w4_lo = pq & 0xFFFFu;
            uint w4_hi = pq >> 16u;
            uint kl0 = p * 8u;
            half dw = (p < 4u ? dw0 : dw1);
            // t=0: ushort lo, nibble 0
            {
                uint nib = w4_lo & 0xFu;
                half w = (half(nib) - half(8.0h)) * dw;
                uint b = kl0 * 8u + 0u;
                g0 += sh_b[b + 0u] * w;
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
                half w = (half(nib) - half(8.0h)) * dw;
                uint b = kl0 * 8u + 8u;
                g0 += sh_b[b + 0u] * w;
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
                half w = (half(nib) - half(8.0h)) * dw;
                uint b = kl0 * 8u + 16u;
                g0 += sh_b[b + 0u] * w;
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
                half w = (half(nib) - half(8.0h)) * dw;
                uint b = kl0 * 8u + 24u;
                g0 += sh_b[b + 0u] * w;
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
                half w = (half(nib) - half(8.0h)) * dw;
                uint b = kl0 * 8u + 32u;
                g0 += sh_b[b + 0u] * w;
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
                half w = (half(nib) - half(8.0h)) * dw;
                uint b = kl0 * 8u + 40u;
                g0 += sh_b[b + 0u] * w;
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
                half w = (half(nib) - half(8.0h)) * dw;
                uint b = kl0 * 8u + 48u;
                g0 += sh_b[b + 0u] * w;
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
                half w = (half(nib) - half(8.0h)) * dw;
                uint b = kl0 * 8u + 56u;
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

