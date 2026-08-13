// slang-entries: moe_combine
//
// Combine the routed experts' down-projection outputs into one hidden state per
// token: `out[t] += sum over the token's slots of weight[s] * z[entry(t,s)]`.
//
// Splitting this out of the down-projection GEMV is what keeps that kernel free
// of cross-slot coordination: every entry writes its own `z` row with no
// contention, and the weighted sum happens here where each output element is
// owned by exactly one thread. The alternative, having the down GEMV accumulate
// straight into `out`, needs either float atomics or a guarantee that no two
// slots of a token land on the same element at the same time, and neither is
// worth it for a `n_used`-deep sum.
//
// `accumulate` picks which of the two FFN output conventions the caller uses,
// because the dense path it replaces does not have one convention either. It is
// a property of the call site, not of decode versus prefill. Decode always
// accumulates into the residual stream, mirroring `gemv_q4_0_accum`. Prefill
// goes both ways: the main batched path leaves the FFN output in a scratch
// buffer whose residual add is fused into the *next* layer's
// `add_rmsnorm_batch`, so it must overwrite (adding there would fold in
// whatever the previous layer left behind), while the profiled path accumulates
// into its own batch buffer as its dense twin does. The caller is responsible
// for `z` being fully written first.
//
// Bindings:
//   t0: z           f32, `[n_entries][hidden]` per-entry expert outputs
//   t1: sel_weight  f32, `[n_entries]` renormalized probabilities from `moe_route`
//   u2: out         f32, `[n_tokens][hidden]` FFN output or residual stream
//   t3: params      (hidden, n_used, n_tokens, accumulate)
//
// Dispatch: (ceil(hidden / 256), n_tokens) threadgroups of 256.

[[vk::binding(0)]] StructuredBuffer<float>   z          : register(t0);
[[vk::binding(1)]] StructuredBuffer<float>   sel_weight : register(t1);
[[vk::binding(2)]] RWStructuredBuffer<float> out_buf    : register(u2);
[[vk::binding(3)]] StructuredBuffer<uint4>   params     : register(t3);

static const uint WG = 256u;

[shader("compute")]
[numthreads(256, 1, 1)]
void moe_combine(uint3 lid : SV_GroupThreadID, uint3 grp : SV_GroupID) {
    uint hidden = params[0].x;
    uint n_used = params[0].y;
    uint n_tokens = params[0].z;
    bool accumulate = params[0].w != 0u;

    uint row = grp.x * WG + lid.x;
    uint tok = grp.y;
    if (row >= hidden || tok >= n_tokens) {
        return;
    }

    float acc = 0.0f;
    for (uint s = 0u; s < n_used; ++s) {
        uint entry = tok * n_used + s;
        acc += sel_weight[entry] * z[entry * hidden + row];
    }
    uint dst = tok * hidden + row;
    out_buf[dst] = accumulate ? (out_buf[dst] + acc) : acc;
}
