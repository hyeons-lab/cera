#include <metal_stdlib>
using namespace metal;

// n_keep context shift on the f16 GPU KV cache. Two kernels here +
// reuse of memcpy_f16:
//
// 1. kv_shift_k_to_scratch: read each retained K cell at its OLD
//    position (n_keep + shift + t_off), apply RoPE delta R(-shift)
//    so the cell's stored angle matches its NEW position
//    (n_keep + t_off), and write to a scratch buffer at compact
//    offset (t_off). Mirror of the per-thread loop in
//    `InferenceState::shift_kv_with_rope` (CPU).
//
// 2. memcpy_f16: generic f16 element copy used to (a) move the
//    rotated K from scratch back into the cache at the new
//    n_keep-aligned offset, (b) move V cells through scratch to
//    the new offset (V isn't RoPE'd, just memmoved). Two-pass
//    via scratch is required because the source range
//    [(n_keep+shift)*kv_dim .. seq_len*kv_dim) and destination
//    [n_keep*kv_dim .. new_seq_len*kv_dim) overlap when
//    `shift < new_seq_len - n_keep`, which is the common case.
//    Metal compute kernels can't synchronize across the entire
//    grid, so an in-place per-thread read+write would race.
//
// RoPE convention matches `qk_norm_rope.metal`'s `head_rope`: each
// dim-pair is rotated by `angle = delta_pos * theta_scale^d` (optionally
// divided by `freq_factors[d]` for Llama-3) with `theta_scale =
// freq_base^(-2/head_dim)` and rotation `(x0, x1) → (x0*c - x1*s,
// x0*s + x1*c)`. For the shift case `delta_pos = -shift` so the stored
// angle is reduced — exactly what's needed for the cell to re-encode its
// new (smaller) position. `rope_type` selects the pair layout so this
// composes with whatever the forward pass applied:
//   0 = NeoX        → pairs at [d, d + half_dim]   (Qwen2/Qwen3/LFM2)
//   1 = NORM/interl → pairs at [2d, 2d + 1]        (LLaMA/Mistral/Granite)
// Using the wrong layout (the old NeoX-only kernel on a NORM model)
// pairs the wrong elements and mis-rotates the retained K cells.

struct KParams {
    uint  n_keep;
    uint  shift;
    uint  new_seq_len;
    uint  n_kv_heads;
    uint  head_dim;
    uint  freq_base_bits;
    int   delta_pos;          // -(shift as i32)
    uint  rope_type;          // 0 = NeoX, 1 = NORM/interleaved
    uint  has_freq_factors;   // 1 ⇒ divide each pair's angle by freq_factors[d]
    uint  _pad;
};

kernel void kv_shift_k_to_scratch(
    device const half*  k_cache       [[buffer(0)]],
    device half*        scratch       [[buffer(1)]],
    constant KParams&   params        [[buffer(2)]],
    device const float* freq_factors  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_dim = params.head_dim / 2u;
    uint retained = params.new_seq_len - params.n_keep;
    uint per_t = params.n_kv_heads * half_dim;
    uint total = retained * per_t;
    if (gid >= total) return;

    uint t_off = gid / per_t;
    uint hd = gid % per_t;
    uint h = hd / half_dim;
    uint d = hd % half_dim;

    uint kv_dim = params.n_kv_heads * params.head_dim;
    uint head_off = h * params.head_dim;

    // Pair element offsets within the head depend on the RoPE layout.
    uint e0, e1;
    if (params.rope_type == 0u) {
        e0 = d;             // NeoX: split-halves
        e1 = d + half_dim;
    } else {
        e0 = 2u * d;        // NORM: adjacent pairs
        e1 = 2u * d + 1u;
    }

    uint t_old = params.n_keep + t_off + params.shift;
    uint src_base = t_old * kv_dim + head_off;
    float x0 = float(k_cache[src_base + e0]);
    float x1 = float(k_cache[src_base + e1]);

    float freq_base = as_type<float>(params.freq_base_bits);
    // Same `powr(theta_scale, d)` form the forward `head_rope` uses, so the
    // delta composes with the cell's existing angle to within the f16-storage
    // round-trip error of the surrounding K cache.
    float theta_scale = powr(freq_base, -2.0f / float(params.head_dim));
    float theta = float(params.delta_pos) * powr(theta_scale, float(d));
    if (params.has_freq_factors != 0u) {
        theta = theta / freq_factors[d];
    }
    float c = cos(theta);
    float s = sin(theta);

    float y0 = x0 * c - x1 * s;
    float y1 = x0 * s + x1 * c;

    uint dst_base = t_off * kv_dim + head_off;
    scratch[dst_base + e0] = half(y0);
    scratch[dst_base + e1] = half(y1);
}

struct CopyParams {
    uint n_elements;
    uint src_offset_elements;
    uint dst_offset_elements;
    uint _pad;
};

// Generic f16 element-wise copy with src/dst element offsets.
// Used by the shift to (a) move rotated K from scratch back into
// the cache, (b) ferry V through scratch (no rotation).
kernel void memcpy_f16_offsets(
    device const half*    src    [[buffer(0)]],
    device half*          dst    [[buffer(1)]],
    constant CopyParams&  params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    dst[params.dst_offset_elements + gid] = src[params.src_offset_elements + gid];
}
