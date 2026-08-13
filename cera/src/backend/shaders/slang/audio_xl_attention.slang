// Conformer self-attention with Transformer-XL relative-position bias, single
// source for both GPU backends.
//
// One threadgroup of 256 per (query, head); grid is (tokens, n_head, 1).
//
//   ac[key] = <Q[q,h] + u[h], K[key,h]>
//   bd[key] = <Q[q,h] + v[h], P[pos_idx,h]>,  pos_idx = (t-1) + key - q
//   scores  = (ac + bd) * scale
//   p       = softmax(scores)                 // no causal mask
//   out     = sum_key p[key] * V[key,h]
//
// Mirrors `cpu::conformer_self_attention_forward` steps 4-6. Two details carry
// the meaning:
//
//   - **The rel-shift is computed, not materialized.** The reference
//     implementation of Transformer-XL builds `matrix_bd` over absolute position
//     and then pads/slices it into a `t x t` layout. Here `pos_idx` indexes the
//     projected position embedding directly, which is the same mapping with no
//     shift buffer. `relative_pos_emb` orders row 0 as rel_pos `+(t-1)` and row
//     `2t-2` as `-(t-1)`, so query `q` attending to key `k` wants row
//     `(t-1) - (q - k)`. `q, key < tokens` keeps that inside `[0, 2t-2]`, so the
//     index needs no clamp.
//   - **`u` and `v` are per-head and loop-invariant in `key`**, so they are
//     folded into Q once into groupshared `qu`/`qv` before the score loop rather
//     than re-added per key.
//
// The CPU accumulates `ac`/`bd` in f64; this is f32 throughout, which is the
// dominant term in the encoder's GPU-vs-CPU tolerance.
//
// This is a portable (non-MMA) attention, so unlike `vit_attention` it is one
// Slang source rather than a handwritten pair: there is no `simdgroup_matrix`
// path for the MSL post-pass to preserve, and only the softmax reduction has a
// per-target fast path. See `slang/README.md` on when handwriting is warranted.
//
//   binding/buffer 0: q       f32, read  [tokens][n_head*head_dim]
//   binding/buffer 1: k       f32, read  [tokens][n_head*head_dim]
//   binding/buffer 2: v       f32, read  [tokens][n_head*head_dim]
//   binding/buffer 3: p       f32, read  [2*tokens-1][n_head*head_dim]
//   binding/buffer 4: bias_u  f32, read  [n_head*head_dim]
//   binding/buffer 5: bias_v  f32, read  [n_head*head_dim]
//   binding/buffer 6: out     f32, write [tokens][n_head*head_dim]
//   binding/buffer 7: params  (tokens, n_head, head_dim, scale_bits)
//
// Dispatch: (tokens, n_head, 1) workgroups of 256.

[[vk::binding(0)]] StructuredBuffer<float>   q_buf   : register(t0);
[[vk::binding(1)]] StructuredBuffer<float>   k_buf   : register(t1);
[[vk::binding(2)]] StructuredBuffer<float>   v_buf   : register(t2);
[[vk::binding(3)]] StructuredBuffer<float>   p_buf   : register(t3);
[[vk::binding(4)]] StructuredBuffer<float>   bu_buf  : register(t4);
[[vk::binding(5)]] StructuredBuffer<float>   bv_buf  : register(t5);
[[vk::binding(6)]] RWStructuredBuffer<float> out_buf : register(u6);
[[vk::binding(7)]] StructuredBuffer<uint4>   par_buf : register(t7);

static const uint WG = 256u;

// Largest sequence length the groupshared `scores` array can hold. The host
// guards `t_out` against this and falls back to the CPU encoder above it, so a
// longer utterance degrades instead of writing out of bounds. Kept in lockstep
// with `MAX_AUDIO_TOKENS` in `model/audio_encoder_gpu.rs` by
// `const_sync_tests::max_audio_tokens_matches_attention_shader_scratch`.
static const uint MAX_TOKENS = 1024u;

// Largest `head_dim` the groupshared Q+bias staging arrays can hold. Also
// host-guarded; 64 for LFM2A (n_embd 512 / 8 heads).
static const uint MAX_HEAD_DIM = 128u;

groupshared float scores[MAX_TOKENS];
groupshared float qu[MAX_HEAD_DIM];
groupshared float qv[MAX_HEAD_DIM];
// Reduction scratch. The Metal path needs only 8 slots (one per simdgroup) but
// the portable tree path needs all 256; one declaration keeps the two branches
// from disagreeing about the allocation. Same split as `softmax.slang`.
groupshared float scratch[WG];

/// Max of `v` across the whole workgroup. Result valid on every thread.
float block_max(uint tid, float v) {
    float result;
    __target_switch {
    case metal:
    {
        float sg = WaveActiveMax(v);
        if ((tid & 31u) == 0u) { scratch[tid >> 5u] = sg; }
        GroupMemoryBarrierWithGroupSync();
        float lane = (tid < 8u) ? scratch[tid] : -3.402823466e+38f;
        float total = WaveActiveMax(lane);
        if (tid == 0u) { scratch[0] = total; }
        GroupMemoryBarrierWithGroupSync();
        result = scratch[0];
        break;
    }
    default:
    {
        scratch[tid] = v;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = WG / 2u; s > 0u; s >>= 1) {
            if (tid < s) { scratch[tid] = max(scratch[tid], scratch[tid + s]); }
            GroupMemoryBarrierWithGroupSync();
        }
        result = scratch[0];
        break;
    }
    }
    return result;
}

/// Sum of `v` across the whole workgroup. Result valid on every thread.
float block_sum(uint tid, float v) {
    float result;
    __target_switch {
    case metal:
    {
        float sg = WaveActiveSum(v);
        if ((tid & 31u) == 0u) { scratch[tid >> 5u] = sg; }
        GroupMemoryBarrierWithGroupSync();
        float lane = (tid < 8u) ? scratch[tid] : 0.0f;
        float total = WaveActiveSum(lane);
        if (tid == 0u) { scratch[0] = total; }
        GroupMemoryBarrierWithGroupSync();
        result = scratch[0];
        break;
    }
    default:
    {
        scratch[tid] = v;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = WG / 2u; s > 0u; s >>= 1) {
            if (tid < s) { scratch[tid] += scratch[tid + s]; }
            GroupMemoryBarrierWithGroupSync();
        }
        result = scratch[0];
        break;
    }
    }
    return result;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void audio_xl_attention(uint3 lid : SV_GroupThreadID, uint3 wid : SV_GroupID) {
    uint tid = lid.x;
    uint tokens = par_buf[0].x;
    uint n_head = par_buf[0].y;
    uint head_dim = par_buf[0].z;
    float scale = asfloat(par_buf[0].w);

    uint q_idx = wid.x;
    uint h = wid.y;
    uint dim = n_head * head_dim;
    uint head_base = h * head_dim;
    uint q_off = q_idx * dim + head_base;

    // Fold the per-head u/v biases into Q once.
    for (uint d = tid; d < head_dim; d += WG) {
        float qd = q_buf[q_off + d];
        qu[d] = qd + bu_buf[head_base + d];
        qv[d] = qd + bv_buf[head_base + d];
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase A: fused content-content + content-position scores.
    uint t_minus_1 = tokens - 1u;
    for (uint key = tid; key < tokens; key += WG) {
        uint k_off = key * dim + head_base;
        uint p_off = (t_minus_1 + key - q_idx) * dim + head_base;
        float ac = 0.0f;
        float bd = 0.0f;
        for (uint d = 0u; d < head_dim; d++) {
            ac += qu[d] * k_buf[k_off + d];
            bd += qv[d] * p_buf[p_off + d];
        }
        scores[key] = (ac + bd) * scale;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase B: max, for numerical stability.
    float local_max = -3.402823466e+38f;
    for (uint key = tid; key < tokens; key += WG) {
        local_max = max(local_max, scores[key]);
    }
    float max_val = block_max(tid, local_max);

    // Phase C: exp and sum. The barrier before `block_sum` is unconditional
    // here, unlike `softmax.slang`'s target-switched guard: it closes the same
    // `scratch` write-after-read between the two reductions, and this kernel is
    // compute-bound (head_dim * tokens multiply-adds per thread in phase A), so
    // the barrier that measured as a regression on a bandwidth-bound softmax is
    // noise here and not worth a per-target branch.
    float partial = 0.0f;
    for (uint key = tid; key < tokens; key += WG) {
        float e = exp(scores[key] - max_val);
        scores[key] = e;
        partial += e;
    }
    GroupMemoryBarrierWithGroupSync();
    float inv_sum = 1.0f / block_sum(tid, partial);

    // Phase D: weighted sum of V, normalized by the softmax denominator.
    for (uint d = tid; d < head_dim; d += WG) {
        float acc = 0.0f;
        for (uint key = 0u; key < tokens; key++) {
            acc += scores[key] * v_buf[key * dim + head_base + d];
        }
        out_buf[q_off + d] = acc * inv_sum;
    }
}
