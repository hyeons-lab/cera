// n_keep context shift on the f32 wgpu KV cache. WGSL port of the
// `kv_shift_k_to_scratch` half of `kv_shift.metal`.
//
// This kernel reads each RETAINED K cell at its OLD position
// (n_keep + shift + t_off), applies the RoPE delta R(delta_pos) with
// delta_pos = -shift so the cell's stored angle matches its NEW position
// (n_keep + t_off), and writes the rotated pair to a scratch buffer at the
// compact offset t_off. The two memcpy halves of the Metal shift (move the
// rotated K back into the cache, ferry V through scratch) are done host-side
// with `copy_buffer_to_buffer` — wgpu has a native buffer blit, so unlike
// Metal no companion memcpy kernel is needed.
//
// Rotating into scratch (rather than in place) is required: the source range
// [(n_keep+shift)*kv_dim ..) and destination [n_keep*kv_dim ..) overlap when
// `shift < retained` (the common case), and a compute grid can't synchronize a
// per-thread read+write across that overlap without racing.
//
// RoPE convention matches `rope.wgsl`'s decode path so the delta composes with
// whatever the forward pass applied: each dim-pair is rotated by
// `angle = delta_pos * freq_base^(-2d/head_dim)` (optionally divided by
// `freq_factors[d]` for Llama-3), pairing selected by `rope_type`:
//   0 = NeoX        → pairs at (d, d + head_dim/2)   (Qwen2/Qwen3/LFM2)
//   1 = NORM/interl → pairs at (2d, 2d + 1)          (LLaMA/Mistral/Granite)
// `delta_pos` is signed (it is negative for a shift), so the angle is computed
// inline here rather than via `rope_angle` (which takes an unsigned `pos`);
// `rotate_rope` is shared with `rope.wgsl`.
//
// Bind group 0:
//   @binding(0) k_cache: array<f32>       (read, the live K cache)
//   @binding(1) scratch: array<f32>       (read-write, retained*kv_dim floats)
//   @binding(2) params:  array<u32, 9>    (n_keep, shift, new_seq_len,
//                                           n_kv_heads, head_dim, freq_base_bits,
//                                           delta_pos (i32 bits), rope_type,
//                                           has_freq_factors)
//   @binding(3) freq_factors: array<f32>  (head_dim/2 factors, or 1-elem dummy)
//
// Dispatch: (ceil(retained * n_kv_heads * head_dim/2 / 256), 1, 1)

#include "common_decls.tmpl"

@group(0) @binding(0) var<storage, read> k_cache: array<f32>;
@group(0) @binding(1) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32, 9>;
@group(0) @binding(3) var<storage, read> freq_factors: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn kv_shift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_keep = params[0];
    let shift = params[1];
    let new_seq_len = params[2];
    let n_kv_heads = params[3];
    let head_dim = params[4];
    let freq_base = bitcast<f32>(params[5]);
    let delta_pos = bitcast<i32>(params[6]);
    let rope_type = params[7];
    let has_freq_factors = params[8];

    let half_dim = head_dim / 2u;
    let retained = new_seq_len - n_keep;
    let per_t = n_kv_heads * half_dim;
    let total = retained * per_t;
    let idx = gid.x;
    if idx >= total {
        return;
    }

    let t_off = idx / per_t;
    let hd = idx % per_t;
    let h = hd / half_dim;
    let d = hd % half_dim;

    let kv_dim = n_kv_heads * head_dim;
    let head_off = h * head_dim;

    // Pair element offsets within the head depend on the RoPE layout.
    var e0: u32;
    var e1: u32;
    if rope_type == 0u {
        e0 = d;             // NeoX: split-halves
        e1 = d + half_dim;
    } else {
        e0 = 2u * d;        // NORM: adjacent pairs
        e1 = 2u * d + 1u;
    }

    let t_old = n_keep + t_off + shift;
    let src_base = t_old * kv_dim + head_off;
    let x0 = k_cache[src_base + e0];
    let x1 = k_cache[src_base + e1];

    // Same per-pair schedule as `rope_angle`, but with a signed delta so the
    // stored angle is reduced — exactly what re-encodes the cell's new
    // (smaller) position.
    var angle = f32(delta_pos) * pow(freq_base, -2.0 * f32(d) / f32(head_dim));
    if has_freq_factors == 1u {
        angle = angle / freq_factors[d];
    }
    let res = rotate_rope(x0, x1, angle);

    let dst_base = t_off * kv_dim + head_off;
    scratch[dst_base + e0] = res.x;
    scratch[dst_base + e1] = res.y;
}
