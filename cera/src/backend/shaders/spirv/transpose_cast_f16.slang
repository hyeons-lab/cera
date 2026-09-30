// Transpose + f32->f16 cast for the streaming Q4_0 GEMM: f32 [N, K]
// token-major activations become f16 [K, N] k-major, so the GEMM's 32-wide
// fiber reads 32 consecutive halfs (one cache line) per k. Mirrors
// llama.cpp's transpose_32_16 + image1d f32->f16 path, with a half storage
// buffer instead of a texel buffer (not expressible through wgpu).
//
// Grid: (ceil(N/32), ceil(K/32)); [numthreads(32, 8, 1)] = 256 threads move
// a 32x32 tile through f32 shmem (stride-33 kills bank conflicts). Both
// phases are coalesced: phase 1 groups 32 consecutive threads on one token
// row (consecutive k), phase 2 groups them on one k row (consecutive n).

struct TransposeParams {
    uint n_valid;
    uint n_pad;
    uint k;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<half> dst;
[[vk::binding(2, 0)]] StructuredBuffer<TransposeParams> paramsBuf;

groupshared float tile[32 * 33];

[numthreads(32, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    let P = paramsBuf[0];
    uint t = lid.y * 32u + lid.x;
    uint g = t / 32u;
    uint l = t % 32u;
    uint col_base = gid.x * 32u; // N (token) tile
    uint row_base = gid.y * 32u; // K tile
    // Phase 1: 4 iters x 256 threads = 1024 f32, consecutive-k per group.
    // Reads only valid columns; pad columns stage zeros.
    for (uint r = 0u; r < 4u; r++) {
        uint n = col_base + g * 4u + r;
        uint kk = row_base + l;
        float v = (n < P.n_valid && kk < P.k) ? src[n * P.k + kk] : 0.0;
        tile[(g * 4u + r) * 33u + l] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    // Phase 2: same thread grid, consecutive-n per group into f16 [K, N_pad],
    // zeros past n_valid so the GEMM's B loads need no guards.
    for (uint r = 0u; r < 4u; r++) {
        uint kk = row_base + g * 4u + r;
        uint n = col_base + l;
        if (kk < P.k && n < P.n_pad) {
            half v = (n < P.n_valid) ? half(tile[l * 33u + (g * 4u + r)]) : half(0.0h);
            dst[kk * P.n_pad + n] = v;
        }
    }
}
