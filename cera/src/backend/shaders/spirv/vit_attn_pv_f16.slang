// ViT attention, step 3 of 3: O = (P V) / l per head, as an fp16 GEMM. One thread per output
// channel d of the head (64 per workgroup), 32 query columns in eight half4 accumulators, the
// contraction over keys in 64-key slices.
//
//   V16[key * dim + h*64 + d]     (A, f16 token-major: coalesced across the d of a workgroup)
//   PT[(h*key_pad + key) * n_pad + query]   (B tile, from `vit_attn_softmax_t`)
//   lsum[h * n_pad + query]       (row sums, applied in the epilogue)
// Output `out[query * dim + h*64 + d]`, f32 token-major like the other ViT ops.
//
// Within a slice the sums are fp16; each slice is then added into f32 accumulators so the error
// does not grow with the number of keys. `key_pad` is tokens rounded up to 64, and V16/PT are zero
// beyond `tokens`, so the last slice needs no mask.
//
// Grid: (n_head, ceil(tokens / 32), 1); [numthreads(64, 1, 1)].

struct Params {
    uint tokens;
    uint n_pad;
    uint key_pad;
    uint dim;
};

[[vk::binding(0, 0)]] StructuredBuffer<half> v16;
[[vk::binding(1, 0)]] StructuredBuffer<half> pt;
[[vk::binding(2, 0)]] StructuredBuffer<float> lsum;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outbuf;
[[vk::binding(4, 0)]] StructuredBuffer<Params> paramsBuf;

groupshared half4 sh_b[64 * 8];

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lid : SV_GroupThreadID) {
    let P = paramsBuf[0];
    uint h = gid.x;
    uint col_base = gid.y * 32u;
    uint d = lid;

    float4 f0 = float4(0.0);
    float4 f1 = float4(0.0);
    float4 f2 = float4(0.0);
    float4 f3 = float4(0.0);
    float4 f4 = float4(0.0);
    float4 f5 = float4(0.0);
    float4 f6 = float4(0.0);
    float4 f7 = float4(0.0);

    uint n_kb = P.key_pad / 64u;
    for (uint kb = 0u; kb < n_kb; kb++) {
        // Stage the P tile [64 keys][8 groups]: 512 half4s, 8 per thread.
        for (uint s = 0u; s < 8u; s++) {
            uint e = lid * 8u + s;
            uint kk = e / 8u;
            uint g = e % 8u;
            uint b = (h * P.key_pad + kb * 64u + kk) * P.n_pad + col_base + g * 4u;
            sh_b[kk * 8u + g] = half4(pt[b], pt[b + 1u], pt[b + 2u], pt[b + 3u]);
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
        // 8 keys per iteration: a full 64-way unroll made the driver's code so large that a
        // dispatch took over a second (see `vit_attn_scores_f16.slang`).
        [loop]
        for (uint kk0 = 0u; kk0 < 64u; kk0 += 8u) {
            [unroll]
            for (uint u = 0u; u < 8u; u++) {
                uint kk = kk0 + u;
                half w = v16[(kb * 64u + kk) * P.dim + h * 64u + d];
                uint b = kk * 8u;
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
        f0 += float4(g0);
        f1 += float4(g1);
        f2 += float4(g2);
        f3 += float4(g3);
        f4 += float4(g4);
        f5 += float4(g5);
        f6 += float4(g6);
        f7 += float4(g7);
        GroupMemoryBarrierWithGroupSync();
    }

    uint ch = h * 64u + d;
    [unroll]
    for (uint j = 0u; j < 8u; j++) {
        float4 f = (j == 0u) ? f0 : (j == 1u) ? f1 : (j == 2u) ? f2 : (j == 3u) ? f3
                 : (j == 4u) ? f4 : (j == 5u) ? f5 : (j == 6u) ? f6 : f7;
        uint c = col_base + j * 4u;
        if (c < P.tokens) {
            outbuf[c * P.dim + ch] = f.x / lsum[h * P.n_pad + c];
        }
        if (c + 1u < P.tokens) {
            outbuf[(c + 1u) * P.dim + ch] = f.y / lsum[h * P.n_pad + c + 1u];
        }
        if (c + 2u < P.tokens) {
            outbuf[(c + 2u) * P.dim + ch] = f.z / lsum[h * P.n_pad + c + 2u];
        }
        if (c + 3u < P.tokens) {
            outbuf[(c + 3u) * P.dim + ch] = f.w / lsum[h * P.n_pad + c + 3u];
        }
    }
}
