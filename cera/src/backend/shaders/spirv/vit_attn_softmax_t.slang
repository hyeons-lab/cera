// ViT attention, step 2 of 3: softmax over the keys of S^T[h][key][query], one thread per query so
// every pass reads and writes consecutive queries (coalesced).
//
// Writes the UNNORMALIZED probabilities e = exp2(s - max) as f16 into P^T[h][key][query] (key rows
// padded to `key_pad`, zero beyond `tokens`), and the row sums l[h][query] separately; the PV GEMM
// divides by l in its epilogue, so there is no third pass here.
//
// Grid: (ceil(tokens / 256), n_head); [numthreads(256, 1, 1)].

struct Params {
    uint tokens;
    uint n_pad;
    uint key_pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> st;
[[vk::binding(1, 0)]] RWStructuredBuffer<half> pt;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> lsum;
[[vk::binding(3, 0)]] StructuredBuffer<Params> paramsBuf;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lid : SV_GroupThreadID) {
    let P = paramsBuf[0];
    uint q = gid.x * 256u + lid;
    uint h = gid.y;
    if (q >= P.tokens) {
        return;
    }
    uint st_base = h * P.tokens * P.n_pad + q;
    float m = -3.0e38;
    for (uint key = 0u; key < P.tokens; key++) {
        m = max(m, st[st_base + key * P.n_pad]);
    }
    uint pt_base = h * P.key_pad * P.n_pad + q;
    float l = 0.0;
    for (uint key = 0u; key < P.tokens; key++) {
        float e = exp2(st[st_base + key * P.n_pad] - m);
        l += e;
        pt[pt_base + key * P.n_pad] = half(e);
    }
    lsum[h * P.n_pad + q] = l;
}
