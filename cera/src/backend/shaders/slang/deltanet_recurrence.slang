// slang-entries: deltanet_recurrence
//
// Single-token Gated Delta Net recurrence step for Qwen 3.5 / Ornith 1.0 hybrid models.
//
// For each head h in 0..num_v_heads:
//   kh = h / (num_v_heads / num_k_heads)
//   decay = alpha[h]
//   b = beta[h]
//   sk[j] = sum_i S[h, i, j] * k[kh, i]
//   d[j]  = b * (v[h, j] - sk[j])
//   S[h, i, j] = decay * S[h, i, j] + k[kh, i] * d[j]
//   o[h, j]    = sum_i S[h, i, j] * q[kh, i]
//
// Grid dispatch:
//   workgroup size: (128, 1, 1)
//   num workgroups: (num_v_heads, ceil(head_v_dim / 128), 1)
//
// Buffer bindings:
//   0: q        f32, read, [num_k_heads * head_k_dim]
//   1: k        f32, read, [num_k_heads * head_k_dim]
//   2: v        f32, read, [num_v_heads * head_v_dim]
//   3: alpha    f32, read, [num_v_heads]
//   4: beta     f32, read, [num_v_heads]
//   5: state    f32, read-write, [num_v_heads * head_v_dim * head_v_dim]
//   6: out_o    f32, read-write, [num_v_heads * head_v_dim]
//   7: params   u32, read, [num_v_heads, num_k_heads, head_k_dim, head_v_dim]

[[vk::binding(0)]] StructuredBuffer<float>   q_buf     : register(t0);
[[vk::binding(1)]] StructuredBuffer<float>   k_buf     : register(t1);
[[vk::binding(2)]] StructuredBuffer<float>   v_buf     : register(t2);
[[vk::binding(3)]] StructuredBuffer<float>   alpha_buf : register(t3);
[[vk::binding(4)]] StructuredBuffer<float>   beta_buf  : register(t4);
[[vk::binding(5)]] RWStructuredBuffer<float> state_buf : register(u5);
[[vk::binding(6)]] RWStructuredBuffer<float> out_buf   : register(u6);
[[vk::binding(7)]] StructuredBuffer<uint>    par_buf   : register(t7);

[shader("compute")]
[numthreads(128, 1, 1)]
void deltanet_recurrence(
    uint3 gid : SV_GroupID,
    uint3 lid : SV_GroupThreadID
) {
    uint h = gid.x;
    uint j = gid.y * 128u + lid.x;

    uint num_v_heads = par_buf[0];
    uint num_k_heads = par_buf[1];
    uint head_k_dim  = par_buf[2];
    uint head_v_dim  = par_buf[3];

    if (h >= num_v_heads || j >= head_v_dim) {
        return;
    }

    uint heads_per_group = max(1u, num_v_heads / max(1u, num_k_heads));
    uint kh = min(h / heads_per_group, num_k_heads - 1u);

    float dec = alpha_buf[h];
    float b   = beta_buf[h];
    float vj  = v_buf[h * head_v_dim + j];

    uint head_state_base = h * head_k_dim * head_v_dim;
    uint k_base = kh * head_k_dim;
    uint q_base = kh * head_k_dim;

    float s_col[128];

    // 1. Memory read: sk_j = sum_i S[i, j] * k[i]
    float sk_j = 0.0f;
    for (uint i = 0u; i < head_k_dim; ++i) {
        float s_val = state_buf[head_state_base + i * head_v_dim + j];
        if (dec <= 0.0f) {
            s_val = 0.0f;
        } else if (abs(dec - 1.0f) > 1e-7f) {
            s_val *= dec;
        }
        if (i < 128u) {
            s_col[i] = s_val;
        }
        sk_j += s_val * k_buf[k_base + i];
    }

    // 2. Delta error
    float dj = b * (vj - sk_j);

    // 3. Fused Associative State Write and Output Projection Read:
    //    S[i, j] = S_decayed[i, j] + k[i] * d[j]
    //    o[j]    = sum_i S[i, j] * q[i]
    float oj = 0.0f;
    for (uint i = 0u; i < head_k_dim; ++i) {
        uint idx = head_state_base + i * head_v_dim + j;
        float s_val = (i < 128u) ? s_col[i] : (state_buf[idx] * ((dec <= 0.0f) ? 0.0f : dec));
        float updated = s_val + k_buf[k_base + i] * dj;
        state_buf[idx] = updated;
        oj += updated * q_buf[q_base + i];
    }

    out_buf[h * head_v_dim + j] = oj;
}
