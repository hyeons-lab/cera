// ViT attention, step 1 of 3: scores S^T = K Q^T per head, as an fp16 GEMM in the style of
// `gemm_stream_q4_0_k64.slang` (one operand row per thread, 32 columns in eight half4 accumulators,
// the other operand's tile staged through shared memory).
//
// A thread owns one QUERY and computes its scores against 32 keys; the keys' tile is the shared one.
// That orientation is what makes the stores coalesce: the result is key-major, S^T[h][key][query]
// (f32, `n_pad` queries per row), and neighbouring threads are neighbouring queries, so for each key a
// wave writes one contiguous run. (With a thread per key, each thread wrote 32 contiguous floats and
// neighbouring threads wrote 3 KB apart, so every store landed in its own cache line; that kernel took
// 5.3 ms per layer against 0.76 ms for the equally sized P.V GEMM.) Key-major is also what the softmax
// and the P.V GEMM's B tile want.
//
// Both operands arrive k-major from `transpose_cast_f16`:
//   Qt[(h*64 + d) * n_pad + query] (A, read coalesced across the queries of a workgroup)
//   Kt[(h*64 + d) * n_pad + key]   (B tile)
// and the scores are already scaled by `scale * log2(e)` so the softmax can use exp2. The head
// dimension is fixed at 64, so there is exactly one k-slice.
//
// Grid: (ceil(tokens / 256), ceil(tokens / 32), n_head); [numthreads(256, 1, 1)].
// x indexes queries, y indexes 32-key groups.

struct Params {
    uint tokens;
    uint n_pad;
    uint scale_bits; // f32 bits of scale * log2(e)
};

[[vk::binding(0, 0)]] StructuredBuffer<half> qt;
[[vk::binding(1, 0)]] StructuredBuffer<half> kt;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> st;
[[vk::binding(3, 0)]] StructuredBuffer<Params> paramsBuf;

groupshared half4 sh_b[64 * 8];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lid : SV_GroupThreadID) {
    let P = paramsBuf[0];
    uint query = gid.x * 256u + lid;
    uint key_base = gid.y * 32u;
    uint h = gid.z;
    bool valid = query < P.tokens;
    uint rquery = valid ? query : 0u;

    // Stage the K tile [64 d][8 groups of 4 keys]: 512 half4s, 2 per thread.
    for (uint s = 0u; s < 2u; s++) {
        uint e = lid * 2u + s;
        uint dd = e / 8u;
        uint g = e % 8u;
        uint b = (h * 64u + dd) * P.n_pad + key_base + g * 4u;
        sh_b[dd * 8u + g] = half4(kt[b], kt[b + 1u], kt[b + 2u], kt[b + 3u]);
    }
    GroupMemoryBarrierWithGroupSync();

    half4 g0 = half4(0.0h);
    half4 g1 = half4(0.0h);
    half4 g2 = half4(0.0h);
    half4 g3 = half4(0.0h);
    half4 g4 = half4(0.0h);
    half4 g5 = half4(0.0h);
    half4 g6 = half4(0.0h);
    half4 g7 = half4(0.0h);
    // 8 d per iteration (the Q4_0 kernel's body size): a full 64-way unroll made the driver's code
    // so large that a dispatch took over a second.
    [loop]
    for (uint d0 = 0u; d0 < 64u; d0 += 8u) {
        [unroll]
        for (uint u = 0u; u < 8u; u++) {
            uint d = d0 + u;
            half w = qt[(h * 64u + d) * P.n_pad + rquery];
            uint b = d * 8u;
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
    if (!valid) {
        return;
    }
    float sc = asfloat(P.scale_bits);
    // Key j's score for this query: row (h * tokens + key), column `query`. Each store is one float,
    // contiguous across the threads (queries) of a wave. Keys past `tokens` have no row.
    [unroll]
    for (uint j = 0u; j < 8u; j++) {
        float4 f = float4((j == 0u) ? g0 : (j == 1u) ? g1 : (j == 2u) ? g2 : (j == 3u) ? g3
                        : (j == 4u) ? g4 : (j == 5u) ? g5 : (j == 6u) ? g6 : g7) * sc;
        uint key = key_base + j * 4u;
        if (key < P.tokens) {
            st[(h * P.tokens + key) * P.n_pad + query] = f.x;
        }
        if (key + 1u < P.tokens) {
            st[(h * P.tokens + key + 1u) * P.n_pad + query] = f.y;
        }
        if (key + 2u < P.tokens) {
            st[(h * P.tokens + key + 2u) * P.n_pad + query] = f.z;
        }
        if (key + 3u < P.tokens) {
            st[(h * P.tokens + key + 3u) * P.n_pad + query] = f.w;
        }
    }
}
