// RoPE (rotary position embedding), applied in-place to Q and K. One Slang
// source, but unlike gelu/bias_add/elementwise the two backends do NOT share a
// kernel here: they factor RoPE differently, so this is a `__target_switch` port
// like softmax and gemm_q8_0, each branch mirroring its handwritten twin and
// emitting only its own bindings.
//
// - metal branch mirrors `shaders/rope.metal`: NEOX (split-halves) only, 3
//   bindings (q, k, params), 5 params. On Metal the general RoPE case (NEOX +
//   interleaved + Llama-3 freq_factors) lives in the fused `qk_norm_rope.metal`,
//   so plain rope is deliberately the minimal NEOX kernel the LFM2 rope-only
//   path uses.
// - default (wgsl) branch mirrors `shaders/rope.wgsl`: NEOX *and* interleaved
//   (NORM) layouts plus optional `freq_factors`, 4 bindings (q, k, params,
//   freq_factors), 7 params. On wgpu the plain rope kernel carries this
//   generality itself.
//
// `freq_factors` (binding 3) is referenced only in the default branch, so the
// metal branch, being the surviving one on Metal, never uses it and Slang omits
// binding 3 from the emitted MSL. Same mechanism gemm_q8_0 uses to keep its
// `half` staging out of the WGSL emission.
//
// Dispatch: (ceil(max(n_heads, n_kv_heads) * head_dim/2 / 256), 1, 1) x 256.

[[vk::binding(0)]] RWStructuredBuffer<float> q            : register(u0);
[[vk::binding(1)]] RWStructuredBuffer<float> k            : register(u1);
[[vk::binding(2)]] StructuredBuffer<uint>    params       : register(t2);
[[vk::binding(3)]] StructuredBuffer<float>   freq_factors : register(t3);

/// Rotate one (x0, x1) pair by `angle`. Portable, used by both branches.
float2 rotate_pair(float x0, float x1, float angle) {
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    return float2(x0 * cos_a - x1 * sin_a, x0 * sin_a + x1 * cos_a);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rope(uint3 gid : SV_DispatchThreadID) {
    uint idx = gid.x;

    __target_switch {
    case metal:
    {
        // Mirrors rope.metal: NEOX-only, params[0..4], no freq_factors.
        uint pos        = params[0];
        uint n_heads    = params[1];
        uint n_kv_heads = params[2];
        uint head_dim   = params[3];
        float freq_base = asfloat(params[4]);
        uint half_dim   = head_dim / 2u;

        // `1.0 / pow(fb, 2d/hd)` mirrors rope.metal's `1.0f / powr(...)`; Slang
        // has no `powr` spelling, so this emits `pow`, a few-ULP difference the
        // parity test's relative bound absorbs.
        uint q_total = n_heads * half_dim;
        if (idx < q_total) {
            uint head = idx / half_dim;
            uint d    = idx % half_dim;
            float freq  = 1.0f / pow(freq_base, float(2u * d) / float(head_dim));
            float angle = float(pos) * freq;
            uint i0 = head * head_dim + d;
            uint i1 = i0 + half_dim;
            float2 r = rotate_pair(q[i0], q[i1], angle);
            q[i0] = r.x;
            q[i1] = r.y;
        }

        uint k_total = n_kv_heads * half_dim;
        if (idx < k_total) {
            uint head = idx / half_dim;
            uint d    = idx % half_dim;
            float freq  = 1.0f / pow(freq_base, float(2u * d) / float(head_dim));
            float angle = float(pos) * freq;
            uint i0 = head * head_dim + d;
            uint i1 = i0 + half_dim;
            float2 r = rotate_pair(k[i0], k[i1], angle);
            k[i0] = r.x;
            k[i1] = r.y;
        }
        break;
    }
    default:
    {
        // Mirrors rope.wgsl: NEOX + interleaved + optional freq_factors,
        // params[0..6] plus the freq_factors buffer.
        uint pos              = params[0];
        uint n_heads          = params[1];
        uint n_kv_heads       = params[2];
        uint head_dim         = params[3];
        float freq_base       = asfloat(params[4]);
        uint rope_type        = params[5];
        uint has_freq_factors = params[6];
        uint half_dim         = head_dim / 2u;

        uint q_total = n_heads * half_dim;
        if (idx < q_total) {
            uint head = idx / half_dim;
            uint d    = idx % half_dim;
            float angle = float(pos) * pow(freq_base, -2.0f * float(d) / float(head_dim));
            if (has_freq_factors == 1u) {
                angle = angle / freq_factors[d];
            }
            uint i0;
            uint i1;
            if (rope_type == 0u) {
                i0 = head * head_dim + d;
                i1 = head * head_dim + d + half_dim;
            } else {
                i0 = head * head_dim + 2u * d;
                i1 = head * head_dim + 2u * d + 1u;
            }
            float2 r = rotate_pair(q[i0], q[i1], angle);
            q[i0] = r.x;
            q[i1] = r.y;
        }

        uint k_total = n_kv_heads * half_dim;
        if (idx < k_total) {
            uint head = idx / half_dim;
            uint d    = idx % half_dim;
            float angle = float(pos) * pow(freq_base, -2.0f * float(d) / float(head_dim));
            if (has_freq_factors == 1u) {
                angle = angle / freq_factors[d];
            }
            uint i0;
            uint i1;
            if (rope_type == 0u) {
                i0 = head * head_dim + d;
                i1 = head * head_dim + d + half_dim;
            } else {
                i0 = head * head_dim + 2u * d;
                i1 = head * head_dim + 2u * d + 1u;
            }
            float2 r = rotate_pair(k[i0], k[i1], angle);
            k[i0] = r.x;
            k[i1] = r.y;
        }
        break;
    }
    }
}
