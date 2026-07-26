#include <metal_stdlib>
using namespace metal;

// TurboQuant KV compression kernels (arXiv:2504.19874) — MSL port of
// `turboquant.wgsl`, which is the reference for the packed layout and the
// algorithm. Read that file's header first; this one only notes the differences.
//
// Three kernels:
//   tq_encode_keys   — 2-bit PolarQuant + 1-bit QJL residual + f16 norms
//   tq_encode_values — 2-bit PolarQuant + f16 norm (no residual)
//   tq_rotate_q      — pre-rotate query heads (q_rot, q_jl, sum(q_jl))
//
// The packed layout is byte-identical to the WGSL version, deliberately: the
// prefix-cache blobs (`TQK1`/`TQV1`) are shared across CPU, wgpu, and Metal, so
// all three must agree bit for bit. In particular the norm word is
// `(f16 bits of norm) | (f16 bits of residual_norm) << 16`, which is exactly what
// WGSL's `pack2x16float(vec2(norm, residual_norm))` produces on a little-endian
// device, and `half(x)` rounds the same way (round-to-nearest-even).
//
// Unlike the other Metal attention kernels, these are NOT templated on a
// head_dim constant and do not use SIMD-group reductions: the compressed path is
// bounded by the packed loads, not by the transform, and keeping the structure
// line-for-line with the WGSL is what lets one CPU oracle validate both. If
// profiling later shows the encode on the critical path, the `HD_CONST` template
// treatment in `flash_attention.metal` is the model to follow.
//
// The zero-norm short-circuit is written as `select` at the write sites rather
// than an early `return`, matching the WGSL. That is not just for symmetry: a
// subset of threads returning early would leave the remaining threads waiting on
// a `threadgroup_barrier` the returned ones never reach.
//
// Bindings (all three kernels):
//   buffer(0) src    — const float*, n_tokens × src_stride source rows
//   buffer(1) out    — uint*, the packed cache (or, for tq_rotate_q, the rotated
//                      queries bit-cast to uint so one binding type serves all)
//   buffer(2) signs  — const float*, all layers' [polar | jl] sign flips
//   buffer(3) params — constant TqParams&
// Grid: one threadgroup of TQ_WG threads per (row, head).

constant constexpr uint TQ_WG = 128;
constant constexpr uint MAX_HEAD_DIM = 128;
// Below this a vector is treated as zero (matches the CPU's 1e-12 guard).
constant constexpr float TQ_EPS = 1e-12f;

// Mirror of `TqParams` in `backend/metal/params.rs`. Keep field-identical.
struct TqParams {
    uint  n_tokens;
    uint  n_heads;
    uint  head_dim;
    uint  src_stride;
    uint  dst_pos;
    uint  max_seq_len;
    uint  sign_off;
    uint  q_cap;
    float c0;
    float c1;
    float c2;
    float c3;
    float b0;
    float b1;
    float b2;
    uint  _pad;
};

// Nearest of the 4 Lloyd-Max centroids as a 2-bit index. Mirrors
// `turboquant::quantize_scalar`'s boundary comparisons exactly.
static inline uint tq_quantize(float v, constant TqParams& p) {
    if (v < p.b1) {
        return v < p.b0 ? 0u : 1u;
    }
    return v < p.b2 ? 2u : 3u;
}

static inline float tq_centroid(uint idx, constant TqParams& p) {
    float c[4] = {p.c0, p.c1, p.c2, p.c3};
    return c[idx];
}

/// Sum `red[0..TQ_WG]` into `red[0]`. Every lane must have written its slot and
/// the caller must NOT have a barrier pending.
static inline float tq_reduce_sum(threadgroup float* red, uint tid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TQ_WG / 2u; s > 0u; s >>= 1u) {
        if (tid < s) { red[tid] += red[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return red[0];
}

/// In-place Walsh-Hadamard butterfly over `x[0..head_dim]`, unnormalized.
/// `head_dim` is a power of two; threads `tid < head_dim/2` each own one pair per
/// stage. Barriers are at the top of each stage so the previous stage's writes
/// are visible before the next stage's reads.
static inline void tq_wht(threadgroup float* x, uint head_dim, uint tid) {
    for (uint stride = 1u; stride < head_dim; stride <<= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < head_dim / 2u) {
            uint i = (tid / stride) * 2u * stride + (tid % stride);
            float a = x[i];
            float b = x[i + stride];
            x[i] = a + b;
            x[i + stride] = a - b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/// Pack 16 two-bit indices per output word; one thread per word.
static inline void tq_pack_polar(
    threadgroup const uint* idxs,
    device uint* out,
    uint base_word,
    uint polar_words,
    uint tid,
    bool force_zero
) {
    if (tid < polar_words) {
        uint w = 0u;
        for (uint k = 0u; k < 16u; ++k) {
            w |= idxs[tid * 16u + k] << (2u * k);
        }
        out[base_word + tid] = select(w, 0u, force_zero);
    }
}

kernel void tq_encode_keys(
    device const float* src    [[buffer(0)]],
    device uint*        out    [[buffer(1)]],
    device const float* signs  [[buffer(2)]],
    constant TqParams&  params [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float rot[MAX_HEAD_DIM];
    threadgroup uint  idxs[MAX_HEAD_DIM];
    threadgroup float red[TQ_WG];

    const uint head_dim = params.head_dim;
    const uint n_kv_heads = params.n_heads;
    if (gid >= params.n_tokens * n_kv_heads) { return; }
    const uint j = gid / n_kv_heads;
    const uint h = gid % n_kv_heads;
    const uint t = params.dst_pos + j;

    const uint polar_words = head_dim / 16u;
    const uint jl_words = head_dim / 32u;
    const uint vecs = n_kv_heads * params.max_seq_len;
    const uint jl_off = vecs * polar_words;
    const uint norm_off = jl_off + vecs * jl_words;
    const uint slot = h * params.max_seq_len + t;
    const uint src_base = j * params.src_stride + h * head_dim;
    const float inv_sqrt_d = 1.0f / sqrt(float(head_dim));

    // 1. norm = ||k_head||. Every lane writes its `red` slot.
    float x = tid < head_dim ? src[src_base + tid] : 0.0f;
    red[tid] = x * x;
    const float norm = sqrt(tq_reduce_sum(red, tid));
    const bool is_zero = norm < TQ_EPS;
    const float safe_norm = select(norm, 1.0f, is_zero);

    // 2. normalize + PolarQuant RHT.
    if (tid < head_dim) {
        rot[tid] = x / safe_norm * signs[params.sign_off + tid];
    }
    tq_wht(rot, head_dim, tid);
    if (tid < head_dim) { rot[tid] *= inv_sqrt_d; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. quantize, stash the residual in `rot`, reduce its squared norm.
    float r = 0.0f;
    if (tid < head_dim) {
        uint qi = tq_quantize(rot[tid], params);
        idxs[tid] = qi;
        r = rot[tid] - tq_centroid(qi, params);
    }
    red[tid] = r * r;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Safe to overwrite `rot`: every lane has read its own slot, and the residual
    // is only read again after the reduction's barriers.
    if (tid < head_dim) { rot[tid] = r; }
    const float residual_norm = sqrt(tq_reduce_sum(red, tid));
    const bool no_residual = residual_norm < TQ_EPS;
    const float safe_rnorm = select(residual_norm, 1.0f, no_residual);

    tq_pack_polar(idxs, out, slot * polar_words, polar_words, tid, is_zero);

    // 4. QJL: normalize the residual, second RHT, pack sign bits. The
    // 1/sqrt(head_dim) normalization is skipped — a positive scalar, and only the
    // sign of each component survives into the packed bits.
    if (tid < head_dim) {
        rot[tid] = rot[tid] / safe_rnorm * signs[params.sign_off + head_dim + tid];
    }
    tq_wht(rot, head_dim, tid);
    if (tid < jl_words) {
        uint w = 0u;
        for (uint k = 0u; k < 32u; ++k) {
            if (rot[tid * 32u + k] >= 0.0f) { w |= 1u << k; }
        }
        out[jl_off + slot * jl_words + tid] = select(w, 0u, is_zero || no_residual);
    }

    if (tid == 0u) {
        uint packed = uint(as_type<ushort>(half(norm)))
                    | (uint(as_type<ushort>(half(residual_norm))) << 16);
        out[norm_off + slot] = select(packed, 0u, is_zero);
    }
}

kernel void tq_encode_values(
    device const float* src    [[buffer(0)]],
    device uint*        out    [[buffer(1)]],
    device const float* signs  [[buffer(2)]],
    constant TqParams&  params [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float rot[MAX_HEAD_DIM];
    threadgroup uint  idxs[MAX_HEAD_DIM];
    threadgroup float red[TQ_WG];

    const uint head_dim = params.head_dim;
    const uint n_kv_heads = params.n_heads;
    if (gid >= params.n_tokens * n_kv_heads) { return; }
    const uint j = gid / n_kv_heads;
    const uint h = gid % n_kv_heads;
    const uint t = params.dst_pos + j;

    const uint polar_words = head_dim / 16u;
    const uint norm_off = n_kv_heads * params.max_seq_len * polar_words;
    const uint slot = h * params.max_seq_len + t;
    const uint src_base = j * params.src_stride + h * head_dim;
    const float inv_sqrt_d = 1.0f / sqrt(float(head_dim));

    float x = tid < head_dim ? src[src_base + tid] : 0.0f;
    red[tid] = x * x;
    const float norm = sqrt(tq_reduce_sum(red, tid));
    const bool is_zero = norm < TQ_EPS;
    const float safe_norm = select(norm, 1.0f, is_zero);

    if (tid < head_dim) {
        rot[tid] = x / safe_norm * signs[params.sign_off + tid];
    }
    tq_wht(rot, head_dim, tid);
    if (tid < head_dim) {
        idxs[tid] = tq_quantize(rot[tid] * inv_sqrt_d, params);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tq_pack_polar(idxs, out, slot * polar_words, polar_words, tid, is_zero);
    if (tid == 0u) {
        uint packed = uint(as_type<ushort>(half(norm)));
        out[norm_off + slot] = select(packed, 0u, is_zero);
    }
}

// Mirrors `turboquant::rotate_queries`: q_rot = RHT_polar(q), q_jl =
// RHT_jl(q_rot) (the residual lives in rotated space, so JL is applied to the
// already-rotated query), plus the per-head sum of q_jl the score correction
// needs. The query is NOT normalized — the estimator is linear in q.
//
// Output regions in `out` (f32 values bit-cast to uint):
//   q_rot: (j * n_heads + h) * head_dim + d
//   q_jl:  region     + same
//   sums:  2 * region + j * n_heads + h
// where region = q_cap * n_heads * head_dim.
kernel void tq_rotate_q(
    device const float* src    [[buffer(0)]],
    device uint*        out    [[buffer(1)]],
    device const float* signs  [[buffer(2)]],
    constant TqParams&  params [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float rot[MAX_HEAD_DIM];
    threadgroup float red[TQ_WG];

    const uint head_dim = params.head_dim;
    const uint n_heads = params.n_heads;
    if (gid >= params.n_tokens * n_heads) { return; }
    const uint j = gid / n_heads;
    const uint h = gid % n_heads;

    const uint region = params.q_cap * n_heads * head_dim;
    const uint out_base = (j * n_heads + h) * head_dim;
    const float inv_sqrt_d = 1.0f / sqrt(float(head_dim));

    if (tid < head_dim) {
        rot[tid] = src[j * params.src_stride + h * head_dim + tid]
                 * signs[params.sign_off + tid];
    }
    tq_wht(rot, head_dim, tid);
    float q_rot_d = 0.0f;
    if (tid < head_dim) {
        q_rot_d = rot[tid] * inv_sqrt_d;
        out[out_base + tid] = as_type<uint>(q_rot_d);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Second RHT, applied to the rotated query.
    if (tid < head_dim) {
        rot[tid] = q_rot_d * signs[params.sign_off + head_dim + tid];
    }
    tq_wht(rot, head_dim, tid);
    float q_jl_d = 0.0f;
    if (tid < head_dim) {
        q_jl_d = rot[tid] * inv_sqrt_d;
        out[region + out_base + tid] = as_type<uint>(q_jl_d);
    }

    red[tid] = q_jl_d;
    float sum = tq_reduce_sum(red, tid);
    if (tid == 0u) {
        out[2u * region + j * n_heads + h] = as_type<uint>(sum);
    }
}
