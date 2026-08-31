// slang-entries: moe_route
//
// Mixture-of-experts routing for `lfm2moe`: sigmoid the router logits, pick the
// top `n_used` experts by *biased* score, and emit the *unbiased* probabilities
// renormalized over the winners.
//
// This is the DeepSeek-V3 selection rule (`expert_gating_func = 2`), and the
// split between the two score sets is the whole point of it: `exp_probs_b`
// steers *which* experts are chosen without touching *how much* each chosen one
// contributes. Ranking and weighting by the same number would quietly change the
// model's output, and the failure is invisible in aggregate metrics, so this
// kernel's selection is pinned against `lfm2::select_experts` by the oracle
// suites, on a fixture whose biases are scaled to land ties on the top-k
// boundary.
//
// That agreement is for finite logits. `select_experts` ranks with `total_cmp`,
// which sorts a NaN *above* every real score. The forward scan below does not
// reproduce that order: `score > best_score` is false against a NaN, so a NaN
// is only ever taken by the `!have` branch on the first unused expert, and once
// taken it locks that slot, since every later comparison against it is false
// too. Which expert wins therefore depends on scan position rather than on the
// NaN, and the two backends can pick differently.
//
// Left alone deliberately. Matching `total_cmp` would cost two extra
// comparisons per expert per token to agree on which garbage to emit: a NaN
// router logit means the hidden state feeding it is already NaN, so the block's
// output is NaN under either choice. Not worth buying; worth knowing.
//
// Bindings (same set on both targets, so no `__target_switch` here):
//   t0: logits      f32, [n_tokens][n_expert] router output, pre-sigmoid
//   t1: bias        f32, [n_expert]           `blk.N.exp_probs_b.bias`
//   u2: sel_expert  u32, [n_tokens][n_used]   chosen expert ids
//   u3: sel_weight  f32, [n_tokens][n_used]   renormalized unbiased probs
//   t4: params      (n_expert, n_used, n_tokens, _)
//
// Dispatch: `n_tokens` threadgroups of 32.

[[vk::binding(0)]] StructuredBuffer<float>   logits     : register(t0);
[[vk::binding(1)]] StructuredBuffer<float>   bias       : register(t1);
[[vk::binding(2)]] RWStructuredBuffer<uint>  sel_expert : register(u2);
[[vk::binding(3)]] RWStructuredBuffer<float> sel_weight : register(u3);
[[vk::binding(4)]] StructuredBuffer<uint4>   params     : register(t4);

static const uint WG = 32u;

/// Upper bound on `n_expert`, sizing the groupshared probability scratch.
/// `lfm2moe` ships 32; the host rejects anything above this at load time rather
/// than letting the kernel read past the array.
static const uint MAX_EXPERTS = 256u;

/// Upper bound on `n_expert_used`, sizing the per-thread winners array. Slang
/// needs a compile-time size here, and the selection loop's duplicate check
/// scans it linearly, so this stays small on purpose. `lfm2moe` uses 4.
static const uint MAX_USED = 16u;

groupshared float sh_prob[MAX_EXPERTS];
groupshared float sh_score[MAX_EXPERTS];

[shader("compute")]
[numthreads(32, 1, 1)]
void moe_route(uint3 lid : SV_GroupThreadID, uint3 grp : SV_GroupID) {
    uint n_expert = params[0].x;
    uint n_used = min(min(params[0].y, n_expert), MAX_USED);
    uint n_tokens = params[0].z;
    uint tok = grp.x;

    if (tok >= n_tokens) {
        return;
    }

    for (uint e = lid.x; e < n_expert; e += WG) {
        float logit = clamp(logits[tok * n_expert + e], -80.0f, 80.0f);
        float p = 1.0f / (1.0f + exp(-logit));
        sh_prob[e] = p;
        sh_score[e] = p + bias[e];
    }
    GroupMemoryBarrierWithGroupSync();

    // Selection is serial on one thread. It is k passes over n_expert (4 x 32
    // here), which is nothing next to the expert GEMVs it feeds, and a serial
    // scan is the cheapest way to get *exactly* the CPU's tie-break: a strict
    // `>` walking upward keeps the first maximum, which is the lower index,
    // matching `ggml_argsort_top_k`'s stable order and the reversed index
    // comparison in `lfm2::select_experts`. A parallel argmax would have to
    // rebuild that tie-break by hand.
    if (lid.x != 0u) {
        return;
    }

    // Only `chosen[0 .. s)` is ever read, so the fill is not load-bearing; it
    // is here because Slang cannot see that and warns about a conditionally
    // assigned array, and a warning in a generated shader is noise that hides
    // the next real one. `~0u` is outside any valid expert id, so if the
    // reasoning above were ever wrong the duplicate check would fail open
    // rather than silently match expert 0.
    uint chosen[MAX_USED];
    float unnorm_w[MAX_USED];
    for (uint i = 0u; i < MAX_USED; ++i) {
        chosen[i] = ~0u;
        unnorm_w[i] = 0.0f;
    }
    float sum = 0.0f;

    for (uint s = 0u; s < n_used; ++s) {
        uint best = 0u;
        float best_score = 0.0f;
        bool have = false;

        for (uint e = 0u; e < n_expert; ++e) {
            bool used = false;
            for (uint t = 0u; t < s; ++t) {
                if (chosen[t] == e) {
                    used = true;
                }
            }
            if (used) {
                continue;
            }
            float score = sh_score[e];
            if (!have || score > best_score) {
                best = e;
                best_score = score;
                have = true;
            }
        }

        chosen[s] = best;
        float w = sh_prob[best];
        unnorm_w[s] = w;
        sel_expert[tok * n_used + s] = best;
        sum += w;
    }

    // Clamp the divisor to f16's smallest positive normal (2^-14) exactly as
    // llama.cpp does, so an all-but-zero gate cannot divide by zero.
    float inv_denom = 1.0f / max(sum, 1.0f / 16384.0f);
    for (uint s = 0u; s < n_used; ++s) {
        sel_weight[tok * n_used + s] = unnorm_w[s] * inv_denom;
    }
}
