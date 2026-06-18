#include <metal_stdlib>
using namespace metal;

// Bidirectional multi-head self-attention (MSL mirror of vit_attention.wgsl).
// One threadgroup of 256 threads per (query token, head):
//   scores[key] = dot(Q[q,h], K[key,h]) * scale
//   p          = softmax(scores)
//   out[q,h,d] = sum_key p[key] * V[key,h,d]
// No causal mask, no RoPE. Q/K/V/out are [tokens, n_head*head_dim] row-major.
// `scores` is threadgroup-resident, sized MAX_TOKENS=1024 (caller guards).
//
// Dispatch: threadgroups (tokens, n_head, 1), threads (256, 1, 1).

constant uint MAX_TOKENS = 1024u;

struct Params { uint tokens; uint n_head; uint head_dim; uint scale_bits; };

kernel void vit_attention(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint3 tid_v [[thread_position_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]
) {
    uint tid = tid_v.x;
    uint tokens = p.tokens;
    uint n_head = p.n_head;
    uint head_dim = p.head_dim;
    float scale = as_type<float>(p.scale_bits);
    uint q_idx = tg.x;
    uint h = tg.y;
    uint dim = n_head * head_dim;
    uint q_off = q_idx * dim + h * head_dim;

    threadgroup float scores[MAX_TOKENS];
    threadgroup float sg[8];
    uint lane = tid & 31u;
    uint sid = tid >> 5u;

    // Phase A: scores.
    for (uint key = tid; key < tokens; key += 256u) {
        uint k_off = key * dim + h * head_dim;
        float s = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            s += q[q_off + d] * k[k_off + d];
        }
        scores[key] = s * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase B: max (two-stage simd reduction).
    float lmax = -3.402823e+38f;
    for (uint key = tid; key < tokens; key += 256u) {
        lmax = max(lmax, scores[key]);
    }
    float mm = simd_max(lmax);
    if (lane == 0u) sg[sid] = mm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0u) {
        float v0 = lane < 8u ? sg[lane] : -3.402823e+38f;
        float t = simd_max(v0);
        if (lane == 0u) sg[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = sg[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase C: exp + sum.
    float lsum = 0.0f;
    for (uint key = tid; key < tokens; key += 256u) {
        float e = exp(scores[key] - mx);
        scores[key] = e;
        lsum += e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sm = simd_sum(lsum);
    if (lane == 0u) sg[sid] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0u) {
        float v0 = lane < 8u ? sg[lane] : 0.0f;
        float t = simd_sum(v0);
        if (lane == 0u) sg[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / sg[0];

    // Phase D: weighted sum of V.
    for (uint d = tid; d < head_dim; d += 256u) {
        float acc = 0.0f;
        for (uint key = 0; key < tokens; key++) {
            acc += scores[key] * v[key * dim + h * head_dim + d];
        }
        out[q_off + d] = acc * inv_sum;
    }
}
