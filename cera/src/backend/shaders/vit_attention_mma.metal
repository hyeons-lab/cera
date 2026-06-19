#include <metal_stdlib>
using namespace metal;

// Bidirectional multi-head self-attention for the ViT encoder, flash-attention
// style with simdgroup matrix-multiply (MMA). Same math as the scalar
// vit_attention.metal (no causal mask, no RoPE) but each threadgroup handles a
// block of Q_PER_TG query tokens, staging K/V chunks in threadgroup memory and
// reusing them across the whole query block via 8x8 MMA. This cuts the scalar
// kernel's O(n^2) re-reads of K and V (it re-loaded all of K/V once per query
// token) and runs the dot products on the matrix units.
//
// Adapted from attention_prefill.metal (Iter 5): the causal mask is removed
// (every query attends to every key) and Q/K/V are f32 device inputs staged to
// half for the MMA, with f32 score/output accumulators — the standard
// flash-attention precision shape. Online softmax over C-sized key chunks keeps
// threadgroup memory bounded (≈176·head_dim + 2144 bytes).
//
// Q/K/V/out are [tokens, n_head*head_dim] row-major. Constraint: head_dim % 8
// == 0 and head_dim <= 256 (the host uses the scalar kernel otherwise).
//
// Dispatch: threadgroups (n_head * ceil(tokens/Q_PER_TG), 1, 1), threads (256).

constant constexpr uint Q_PER_TG = 8;
constant constexpr uint C = 64;
constant constexpr uint NW = 32;
constant constexpr uint NSG = 8;
constant constexpr uint N_THREADS = NSG * NW; // 256

struct VitAttnParams {
    uint tokens;
    uint n_head;
    uint head_dim;
    uint scale_bits;
};

// When HD_CONST > 0 the head_dim is a compile-time constant, so the inner MMA
// loop bounds and simdgroup_load strides fold to constexpr and fully unroll.
// HD_CONST == 0 keeps the runtime path for any other head_dim.
template <uint HD_CONST>
inline void vit_attention_mma_impl(
    const device float* q,
    const device float* k,
    const device float* v,
    device float* out,
    constant VitAttnParams& params,
    threadgroup char* shmem,
    uint tid,
    uint tg_idx
) {
    const uint n_head = params.n_head;
    const uint hd = (HD_CONST > 0) ? HD_CONST : params.head_dim;
    const uint tokens = params.tokens;
    const float scale = as_type<float>(params.scale_bits);
    const uint dim = n_head * hd;

    const uint head = tg_idx % n_head;
    const uint q_group = tg_idx / n_head;
    const uint q_base = q_group * Q_PER_TG;
    if (q_base >= tokens) return;
    const uint n_q = min(Q_PER_TG, tokens - q_base);
    const uint h_off = head * hd;

    // TG layout: q_tg half[Q_PER_TG·hd] | kv_tile half[C·hd] (K then V) |
    //   scores f32[Q_PER_TG·C] | out_tg f32[Q_PER_TG·hd] |
    //   state f32[Q_PER_TG·2] | rescales f32[Q_PER_TG].
    threadgroup half*  q_tg     = (threadgroup half*)(shmem);
    threadgroup half*  kv_tile  = q_tg + Q_PER_TG * hd;
    threadgroup float* scores   = (threadgroup float*)(kv_tile + C * hd);
    threadgroup float* out_tg   = scores + Q_PER_TG * C;
    threadgroup float* state    = out_tg + Q_PER_TG * hd;
    threadgroup float* rescales = state + Q_PER_TG * 2;

    const uint simd_lane = tid & 31u;
    const uint simd_id = tid >> 5u;

    // Load Q (f32 -> half) + zero the full output tile and init per-query state.
    for (uint idx = tid; idx < n_q * hd; idx += N_THREADS) {
        uint qq = idx / hd;
        uint d = idx % hd;
        q_tg[qq * hd + d] = half(q[(q_base + qq) * dim + h_off + d]);
    }
    for (uint idx = tid; idx < Q_PER_TG * hd; idx += N_THREADS) {
        out_tg[idx] = 0.0f;
    }
    if (tid < Q_PER_TG) {
        state[tid * 2 + 0] = -INFINITY;
        state[tid * 2 + 1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c0 = 0; c0 < tokens; c0 += C) {
        const uint c_end = min(c0 + C, tokens);
        const uint c_len = c_end - c0;

        // K tile (f32 -> half); tail rows zeroed so 0 x uninit can't make NaN.
        for (uint idx = tid; idx < c_len * hd; idx += N_THREADS) {
            uint t = idx / hd;
            uint d = idx % hd;
            kv_tile[t * hd + d] = half(k[(c0 + t) * dim + h_off + d]);
        }
        for (uint idx = tid + c_len * hd; idx < C * hd; idx += N_THREADS) {
            kv_tile[idx] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // QK^T MMA -> scores [Q_PER_TG x C]; SG `simd_id` owns key col-tile.
        {
            const uint t_tile = simd_id;
            const uint hd_tiles = hd / 8u;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
            simdgroup_half8x8 q_mat;
            simdgroup_half8x8 k_mat;
            for (uint d_tile = 0u; d_tile < hd_tiles; d_tile++) {
                simdgroup_load(q_mat, q_tg + d_tile * 8u, hd);
                simdgroup_load(k_mat,
                               kv_tile + t_tile * 8u * hd + d_tile * 8u,
                               hd,
                               /*origin*/ ulong2(0, 0),
                               /*transpose*/ true);
                simdgroup_multiply_accumulate(acc, q_mat, k_mat, acc);
            }
            simdgroup_store(acc, scores + t_tile * 8u, C);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scale (bidirectional — no causal mask).
        for (uint idx = tid; idx < n_q * c_len; idx += N_THREADS) {
            uint qq = idx / c_len;
            uint t = idx % c_len;
            scores[qq * C + t] = scores[qq * C + t] * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // V tile (f32 -> half), overwriting kv_tile.
        for (uint idx = tid; idx < c_len * hd; idx += N_THREADS) {
            uint t = idx / hd;
            uint d = idx % hd;
            kv_tile[t * hd + d] = half(v[(c0 + t) * dim + h_off + d]);
        }
        for (uint idx = tid + c_len * hd; idx < C * hd; idx += N_THREADS) {
            kv_tile[idx] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax: one SG per query, lane `l` owns cells l and l+32.
        {
            const uint q = simd_id;
            if (q < n_q) {
                const uint idx0 = simd_lane;
                const uint idx1 = simd_lane + NW;
                float s0 = (idx0 < c_len) ? scores[q * C + idx0] : -INFINITY;
                float s1 = (idx1 < c_len) ? scores[q * C + idx1] : -INFINITY;
                float chunk_max = simd_max(max(s0, s1));
                float prev_max = state[q * 2 + 0];
                float new_max = max(prev_max, chunk_max);
                float rescale = (prev_max > -INFINITY) ? exp(prev_max - new_max) : 0.0f;
                float e0 = 0.0f;
                float e1 = 0.0f;
                if (idx0 < c_len) {
                    e0 = exp(s0 - new_max);
                    scores[q * C + idx0] = e0;
                } else {
                    scores[q * C + idx0] = 0.0f;
                }
                if (idx1 < c_len) {
                    e1 = exp(s1 - new_max);
                    scores[q * C + idx1] = e1;
                } else {
                    scores[q * C + idx1] = 0.0f;
                }
                float chunk_sum = simd_sum(e0 + e1);
                if (simd_lane == 0u) {
                    state[q * 2 + 0] = new_max;
                    state[q * 2 + 1] = state[q * 2 + 1] * rescale + chunk_sum;
                    rescales[q] = rescale;
                }
            } else {
                // Unused query row: zero its scores so 0 x V = 0, and zero its
                // rescale so the out_tg pre-rescale below is also 0.
                scores[q * C + simd_lane] = 0.0f;
                scores[q * C + simd_lane + NW] = 0.0f;
                if (simd_lane == 0u) {
                    rescales[q] = 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pre-rescale the running output by each query's rescale factor.
        for (uint idx = tid; idx < Q_PER_TG * hd; idx += N_THREADS) {
            uint q = idx / hd;
            out_tg[idx] = out_tg[idx] * rescales[q];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // PV MMA: out_tg += scores x V. Each SG owns dim-tile(s).
        const uint dim_tiles = hd / 8u;
        const uint inner_tiles = C / 8u;
        for (uint d_off = simd_id; d_off < dim_tiles; d_off += NSG) {
            simdgroup_float8x8 po;
            simdgroup_load(po, out_tg + d_off * 8u, hd);
            simdgroup_float8x8 s_mat;
            simdgroup_half8x8 v_mat;
            for (uint t_in = 0u; t_in < inner_tiles; t_in++) {
                simdgroup_load(s_mat, scores + t_in * 8u, C);
                simdgroup_load(v_mat, kv_tile + t_in * 8u * hd + d_off * 8u, hd);
                simdgroup_multiply_accumulate(po, s_mat, v_mat, po);
            }
            simdgroup_store(po, out_tg + d_off * 8u, hd);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization + write-out.
    for (uint q = 0; q < n_q; q++) {
        float inv_sum = 1.0f / state[q * 2 + 1];
        for (uint d = tid; d < hd; d += N_THREADS) {
            out[(q_base + q) * dim + h_off + d] = out_tg[q * hd + d] * inv_sum;
        }
    }
}

kernel void vit_attention_mma(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant VitAttnParams& params [[buffer(4)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_idx [[threadgroup_position_in_grid]]
) {
    vit_attention_mma_impl<0>(q, k, v, out, params, shmem, tid, tg_idx);
}

kernel void vit_attention_mma_hd64(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant VitAttnParams& params [[buffer(4)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_idx [[threadgroup_position_in_grid]]
) {
    vit_attention_mma_impl<64>(q, k, v, out, params, shmem, tid, tg_idx);
}
