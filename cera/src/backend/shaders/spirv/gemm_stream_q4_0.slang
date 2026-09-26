// Streaming Q4_0 prefill GEMM: dst[n, m] = B[n, k] @ W[m, k]^T.
//
// Transcribed from llama.cpp's kernel_gemm_noshuffle_q4_0_f32
// (ggml-opencl): fp16 fiber accumulators, pre-transposed weights AND
// activations, weights streamed straight from global memory. Each fiber
// (thread) owns 1 row x 32 columns; the B tile is cooperatively staged to
// shmem once per k-slice (coalesced, since B16 is k-major) and broadcast
// from there.
//
// Why this replaces mul_mat_reg_tile for Q4_0 prefill: the reg-tile kernel
// dequantizes weights to f32 in shmem and does f32 MACs at ~200 GFLOP/s on
// Adreno 830 (~7% of peak, 84% of prefill time). The streaming shape does
// fp16 MACs with no weight staging, matching llama's ~2 TFLOP/s on the
// same part.
//
// Adaptations from the OpenCL source:
// - Activations come from a transposed f16 storage buffer (see
//   transpose_cast_f16.slang, dispatched into the same encoder just before
//   this kernel), staged here to shmem — instead of an image1d_buffer_t
//   (texel buffers are not expressible through wgpu).
// - Weight nibbles/scales are packed into u32 storage (2 ushorts / 2 halfs
//   per word) rather than native 16-bit buffers, so the kernel needs only
//   Float16 arithmetic, not 16-bit storage.
// - 8 scalar half4 accumulators (g0..g7), one per 4-column group.
// - The inner 8 k's are manually unrolled (upstream's j=0..3 blocks are
//   straight-line in source); nested p/s/j loops risk the compiler leaving
//   runtime selects and variable shifts in the hot path.
// - Multi-row fibers (2x32, 2x16, 4x8) and 32-thread workgroups were all
//   tried and lost (3-4x slower); the 1x32 fiber in 64-thread workgroups
//   stands.
//
// Weight repack (host, once at upload; see repack_q4_0_stream in gpu_lfm2.rs):
// - q[(k/8)*m + row] = ushort(weights k+0..k+3) | ushort(weights k+4..k+7)<<16
//   (4 nibbles per ushort, low to high)
// - d[(k/64)*m + row] = half(scale of block k+0..k+31) | half(scale k+32..k+63)<<16
//
// Grid: (ceil(m/256), ceil(n_valid/32)); [numthreads(256, 1, 1)].
// Bigger workgroups win monotonically on Adreno 830 (32: 751ms, 64: 180ms,
// 128: 151ms, 256: 149ms); 256 is the WebGPU-maximum workgroup size and
// stages the B tile with exactly one half4 load per thread.
// Constraints (asserted host-side): k % 32 == 0, n_valid >= 32 (a 32-wide
// fiber idles too much below that; short rows ride the reg-tile kernel).
// n_pad is the 32-rounded N the B transpose wrote (stride of src_b);
// columns [n_valid, n_pad) hold zeros. The shmem tile is half4 [32 k][8
// groups]: staging writes and the MAC loop's broadcast reads are all
// vector half4s (scalar-half loads measured 4x the issue pressure).

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

groupshared half4 sh_b[32 * 8];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lid : SV_GroupThreadID) {
    let P = paramsBuf[0];
    uint row = gid.x * 256u + lid;
    uint col_base = gid.y * 32u;
    // Clamp OOB rows to row 0 for the compute (masked at the store), so no
    // load is ever out of bounds even on ragged tiles.
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
        // Stage the B tile [32 k][8 groups] to shmem as half4s: 256 threads
        // x 1 half4; consecutive threads take consecutive groups (B16 is
        // k-major), so the global reads are one coalesced line per 8 threads.
        // NOTE: staging loads are scalar halves — vector half4
        // global loads measured no better on Adreno 830. Only the shmem
        // broadcast reads in the MAC loop below need to be vector half4s
        // (scalar there measured 4x the load-issue pressure), and all eight
        // MUST read sh_b — global reads with tile indices silently return
        // wrong data at full speed.
        for (uint e = lid; e < 256u; e += 256u) {
            uint kk = e / 8u;
            uint g = e % 8u;
            uint b = (kb * 32u + kk) * P.n_pad + col_base + g * 4u;
            sh_b[kk * 8u + g] =
                half4(src_b[b], src_b[b + 1u], src_b[b + 2u], src_b[b + 3u]);
        }
        GroupMemoryBarrierWithGroupSync();

        // One scale-half per 32-block; two packed per u32, kb selects the lane.
        uint pd = src_d[(kb / 2u) * P.m + rrow];
        uint dbits = (kb & 1u) != 0u ? (pd >> 16u) : (pd & 0xFFFFu);
        half dw = half(f16tof32(dbits));

        // One u32 per 8 k's (2 ushorts x 4 nibbles); the 8 k's are unrolled
        // straight-line with constant shifts, mirroring upstream's j-blocks.
        // kl = p*8 + t; B row = (p*8+t)*8 + group.
        for (uint p = 0u; p < 4u; p++) {
            uint pq = src_q[(kb * 4u + p) * P.m + rrow];
            uint w4_lo = pq & 0xFFFFu;
            uint w4_hi = pq >> 16u;
            uint kl0 = p * 8u;
            // t=0: ushort lo, nibble 0
            {
                uint nib = w4_lo & 0xFu;
                half w = (half(nib) - half(8.0h)) * dw;
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
                half w = (half(nib) - half(8.0h)) * dw;
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
                half w = (half(nib) - half(8.0h)) * dw;
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
                half w = (half(nib) - half(8.0h)) * dw;
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
                half w = (half(nib) - half(8.0h)) * dw;
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
                half w = (half(nib) - half(8.0h)) * dw;
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
                half w = (half(nib) - half(8.0h)) * dw;
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
                half w = (half(nib) - half(8.0h)) * dw;
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
    // Stores are the only divergent point: mask columns past n_valid.
    // Unrolled per group so each live store is a tight scalar sequence
    // (strided by y_stride; the common packed case is consecutive rows
    // across the fiber's... rows are per-thread, so these are scalar).
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
