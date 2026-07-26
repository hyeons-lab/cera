#include <metal_stdlib>
using namespace metal;

// FlashAttention over a TurboQuant-compressed KV cache — MSL port of
// `flash_attention_tq.wgsl`, which is the reference for the estimators and the
// packed layout. One kernel serves both paths: decode dispatches a single query
// row (n_queries = 1, start_pos = pos), chunked prefill dispatches the whole
// chunk.
//
// Scores (keys are never reconstructed; the query arrives pre-rotated from
// `tq_rotate_q`):
//   polar_dot  = norm * sum_d q_rot[d] * centroid[idx(d)]
//   signed_sum = 2 * sum_{d : jl_bit(d)=1} q_jl[d]  -  sum_d q_jl[d]
//   correction = norm * residual_norm * sqrt(pi/2)/head_dim * signed_sum
//   score      = (polar_dot + correction) * scale
//
// Values: the RHT is linear, so the accumulator stays in ROTATED space across the
// whole tiled pass and the inverse rotation runs once in the epilogue. The
// online-softmax rescaling is linear too, so the two commute — that is what makes
// a flash formulation possible over a compressed cache.
//
// Deliberately structured line-for-line with the WGSL rather than adopting the
// SIMD-group reductions and `HD_CONST` templating of `flash_attention.metal`: the
// compressed path is bound by unpacking the 2-bit fields, and one CPU oracle
// validates both backends only while they stay structurally identical. The
// templating in `flash_attention.metal` is the model to follow if profiling later
// puts this on the critical path.
//
// Constraints (asserted host-side): head_dim <= 128, a power of two, and a
// multiple of 32. Keys AND values must both be compressed — no mixed mode.
// GQA: kv_head = head / (n_heads / n_kv_heads).
//
// Bindings:
//   buffer(0) qrot   — const float*, rotated queries: [q_rot | q_jl | sums]
//   buffer(1) k_cache — const uint*, [polar | jl | norms]
//   buffer(2) v_cache — const uint*, [polar | norms]
//   buffer(3) out    — float*, n_queries × out_stride
//   buffer(4) params — constant TqAttnParams&
//   buffer(5) signs  — const float*, all layers' [polar | jl] sign flips
// Grid: (n_heads, n_queries) threadgroups of 256 threads.

constant constexpr uint TQA_TILE = 256;
constant constexpr uint TQA_MAX_HEAD_DIM = 128;
constant constexpr float TQA_NEG_INF = -3.402823e+38f;

// Mirror of `TqAttnParams` in `backend/metal/params.rs`. Keep field-identical.
//
// `max_seq` is the causal clamp (start_pos + n_queries); `cache_cap` is the
// cache's allocated timestep capacity, the per-head stride of every compressed
// region. The two are NOT interchangeable — the f32/f16 kernels get away with one
// value because their KV rows are addressed by `kv_dim` alone.
struct TqAttnParams {
    uint  n_heads;
    uint  n_kv_heads;
    uint  head_dim;
    uint  max_seq;
    uint  start_pos;
    float scale;
    uint  q_cap;
    uint  out_stride;
    float qjl_scale;
    uint  sign_off;
    float c0;
    float c1;
    float c2;
    float c3;
    uint  q_base;
    uint  cache_cap;
};

kernel void flash_attention_tq(
    device const float*    qrot    [[buffer(0)]],
    device const uint*     k_cache [[buffer(1)]],
    device const uint*     v_cache [[buffer(2)]],
    device float*          out     [[buffer(3)]],
    constant TqAttnParams& params  [[buffer(4)]],
    device const float*    signs   [[buffer(5)]],
    // MSL requires every grid-position attribute in one kernel to have the same
    // shape, so both are uint3 even though only gid.xy / tid.x are used.
    uint3 gid3 [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]]
) {
    threadgroup float q_rot_shared[TQA_MAX_HEAD_DIM];
    threadgroup float q_jl_shared[TQA_MAX_HEAD_DIM];
    threadgroup float acc[TQA_MAX_HEAD_DIM];   // rotated-space accumulator
    threadgroup float tile_scores[TQA_TILE];
    // This tile's per-timestep value norms, staged by the thread that scored the
    // timestep — otherwise every accumulator thread re-reads the same norm word
    // for every timestep in the tile.
    threadgroup float tile_vnorm[TQA_TILE];
    threadgroup float red[TQA_TILE];
    // [0]=running max, [1]=running sum, [2]=this tile's new max, [3]=correction.
    threadgroup float st[4];

    const uint head = gid3.x;
    const uint q_idx = gid3.y;
    const uint tid = tid3.x;
    const uint n_heads = params.n_heads;
    const uint n_kv_heads = params.n_kv_heads;
    const uint head_dim = params.head_dim;
    const float centroids[4] = {params.c0, params.c1, params.c2, params.c3};

    const uint q_global = params.q_base + q_idx;
    // Per-query causal window over [0..pos_q], clamped so inconsistent params can
    // only truncate the window, never read out of bounds.
    const uint pos_q = params.start_pos + q_global;
    const uint seq_len = min(pos_q + 1u, params.max_seq);

    const uint group_size = n_heads / n_kv_heads;
    const uint kv_head = head / group_size;
    const uint out_offset = q_global * params.out_stride + head * head_dim;

    const uint q_region = params.q_cap * n_heads * head_dim;
    const uint q_offset = (q_global * n_heads + head) * head_dim;
    const float q_jl_sum = qrot[2u * q_region + q_global * n_heads + head];

    const uint polar_words = head_dim / 16u;
    const uint jl_words = head_dim / 32u;
    const uint vecs = n_kv_heads * params.cache_cap;
    const uint k_jl_off = vecs * polar_words;
    const uint k_norm_off = k_jl_off + vecs * jl_words;
    const uint v_norm_off = vecs * polar_words;
    const uint kv_slot_base = kv_head * params.cache_cap;

    // seq_len == 0 would divide by st[1] == 0 → NaN. `seq_len` depends only on
    // params and the (threadgroup-uniform) grid position, so every thread takes
    // this branch together and no barrier is stranded.
    if (seq_len == 0u) {
        if (tid < head_dim) { out[out_offset + tid] = 0.0f; }
        return;
    }

    if (tid < head_dim) {
        q_rot_shared[tid] = qrot[q_offset + tid];
        q_jl_shared[tid] = qrot[q_region + q_offset + tid];
        acc[tid] = 0.0f;
    }
    if (tid == 0u) {
        st[0] = TQA_NEG_INF;
        st[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = 0u; base < seq_len; base += TQA_TILE) {
        // ── score for timestep t = base + tid (one per thread) ──
        const uint t = base + tid;
        float score = TQA_NEG_INF;
        if (t < seq_len) {
            const uint slot = kv_slot_base + t;
            const uint nw = k_cache[k_norm_off + slot];
            const float norm = float(as_type<half>(ushort(nw & 0xFFFFu)));
            const float residual_norm = float(as_type<half>(ushort(nw >> 16)));

            // PolarQuant: dot the rotated query against the centroid each 2-bit
            // index selects, 16 elements per packed word.
            float polar_dot = 0.0f;
            const uint polar_base = slot * polar_words;
            for (uint w = 0u; w < polar_words; ++w) {
                const uint word = k_cache[polar_base + w];
                const uint d0 = w * 16u;
                for (uint k = 0u; k < 16u; ++k) {
                    polar_dot += q_rot_shared[d0 + k] * centroids[(word >> (2u * k)) & 3u];
                }
            }

            // QJL: sum the JL-projected query over the set sign bits, then turn
            // that positive-only sum into the signed one via the precomputed total.
            float pos_sum = 0.0f;
            const uint jl_base = k_jl_off + slot * jl_words;
            for (uint w = 0u; w < jl_words; ++w) {
                const uint word = k_cache[jl_base + w];
                const uint d0 = w * 32u;
                for (uint k = 0u; k < 32u; ++k) {
                    pos_sum += q_jl_shared[d0 + k] * float((word >> k) & 1u);
                }
            }
            const float signed_sum = 2.0f * pos_sum - q_jl_sum;
            const float correction = norm * residual_norm * params.qjl_scale * signed_sum;
            score = (polar_dot * norm + correction) * params.scale;

            tile_vnorm[tid] = float(as_type<half>(ushort(v_cache[v_norm_off + slot] & 0xFFFFu)));
        }
        tile_scores[tid] = score;

        // ── tile max ──
        red[tid] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TQA_TILE / 2u; s > 0u; s >>= 1u) {
            if (tid < s) { red[tid] = max(red[tid], red[tid + s]); }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float tmax = red[0];

        if (tid == 0u) {
            const float nm = max(st[0], tmax);
            st[2] = nm;
            st[3] = exp(st[0] - nm); // first tile: exp(-inf) = 0
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float nm = st[2];
        const float corr = st[3];

        // p = exp(score - nm); reuse tile_scores to hold the exponentials.
        float p = 0.0f;
        if (t < seq_len) { p = exp(tile_scores[tid] - nm); }
        tile_scores[tid] = p;

        // ── tile sum ──
        red[tid] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TQA_TILE / 2u; s > 0u; s >>= 1u) {
            if (tid < s) { red[tid] += red[tid + s]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float tsum = red[0];

        // Rescale the accumulator and add this tile's values — still rotated.
        // Thread `tid` owns rotated dim `tid`, so it needs one 2-bit field out of
        // each timestep's packed value vector.
        if (tid < head_dim) {
            float a = acc[tid] * corr;
            const uint vw = tid / 16u;
            const uint vshift = (tid % 16u) * 2u;
            for (uint jj = 0u; jj < TQA_TILE; ++jj) {
                const uint tt = base + jj;
                if (tt < seq_len) {
                    const uint word = v_cache[(kv_slot_base + tt) * polar_words + vw];
                    a += tile_scores[jj] * tile_vnorm[jj] * centroids[(word >> vshift) & 3u];
                }
            }
            acc[tid] = a;
        }
        if (tid == 0u) {
            st[1] = st[1] * corr + tsum;
            st[0] = nm;
        }
        // Barrier before the next tile reuses tile_scores/red and reads acc/st.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Epilogue: normalize, then one inverse RHT back to the original basis ──
    // `rht_inverse` is: scale by 1/sqrt(head_dim), Walsh-Hadamard (self-inverse),
    // undo the sign flip. Folding the softmax denominator in here is free.
    const float inv_sqrt_d = 1.0f / sqrt(float(head_dim));
    if (tid < head_dim) {
        acc[tid] = acc[tid] / st[1] * inv_sqrt_d;
    }
    for (uint stride = 1u; stride < head_dim; stride <<= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < head_dim / 2u) {
            const uint i = (tid / stride) * 2u * stride + (tid % stride);
            const float a = acc[i];
            const float b = acc[i + stride];
            acc[i] = a + b;
            acc[i + stride] = a - b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < head_dim) {
        out[out_offset + tid] = acc[tid] * signs[params.sign_off + tid];
    }
}
