// n_keep context shift on the f32 wgpu KV cache. WGSL port of the
// `kv_shift_k_to_scratch` half of `kv_shift.metal`.
//
// This kernel reads each RETAINED K cell at its OLD position
// (n_keep + shift + t_off), applies the RoPE delta R(-shift) so the cell's
// stored angle matches its NEW position (n_keep + t_off), and writes the
// rotated pair to a scratch buffer at the compact offset t_off. The two memcpy
// halves of the Metal shift (move the rotated K back into the cache, ferry V
// through scratch) are done host-side with `copy_buffer_to_buffer` — wgpu has a
// native buffer blit, so unlike Metal no companion memcpy kernel is needed.
//
// Rotating into scratch (rather than in place) is required: the source range
// [(n_keep+shift)*kv_dim ..) and destination [n_keep*kv_dim ..) overlap when
// `shift < retained` (the common case), and a compute grid can't synchronize a
// per-thread read+write across that overlap without racing.
//
// RoPE convention matches `rope.wgsl`'s decode path so the delta composes with
// whatever the forward pass applied: each dim-pair is rotated by the shared
// `rope_angle(shift, d, ...)` schedule, NEGATED (the delta is -shift) and
// optionally divided by `freq_factors[d]` for Llama-3. Sharing `rope_angle`
// keeps this in lockstep with the forward path — if the frequency schedule ever
// changes there, the shift tracks it automatically. Pairing selected by
// `rope_type`:
//   0 = NeoX        → pairs at (d, d + head_dim/2)   (Qwen2/Qwen3/LFM2)
//   1 = NORM/interl → pairs at (2d, 2d + 1)          (LLaMA/Mistral/Granite)
//
// Bind group 0:
//   @binding(0) k_cache: array<f32>       (read, the live K cache)
//   @binding(1) scratch: array<f32>       (read-write, retained*kv_dim floats)
//   @binding(2) params:  array<u32, 8>    (n_keep, shift, retained, n_kv_heads,
//                                           head_dim, freq_base_bits, rope_type,
//                                           has_freq_factors) — see the Rust
//                                           `KvShiftParams` struct (single
//                                           source of truth for this layout)
//   @binding(3) freq_factors: array<f32>  (head_dim/2 factors, or 1-elem dummy)
//
// Dispatch is 2-D so the total workgroup count can exceed the 65535 per-
// dimension limit (the retained context can be tens of thousands of cells):
// the host passes (wg.min(65535), wg.div_ceil(65535), 1) and the kernel
// recovers the linear workgroup index with `get_wid` — the same pattern the
// GEMV/GEMM kernels use.

#include "common_decls.tmpl"

@group(0) @binding(0) var<storage, read> k_cache: array<f32>;
@group(0) @binding(1) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32, 8>;
@group(0) @binding(3) var<storage, read> freq_factors: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn kv_shift(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n_keep = params[0];
    let shift = params[1];
    let retained = params[2];
    let n_kv_heads = params[3];
    let head_dim = params[4];
    let freq_base = bitcast<f32>(params[5]);
    let rope_type = params[6];
    let has_freq_factors = params[7];

    let half_dim = head_dim / 2u;
    let per_t = n_kv_heads * half_dim;
    let total = retained * per_t;
    // 2-D workgroup grid: recover the flat thread index (see header).
    let idx = get_wid(wid) * 256u + lid.x;
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

    // Same per-pair schedule as the forward path's `rope_angle`, negated so the
    // stored angle is reduced by `shift` — exactly what re-encodes the cell's
    // new (smaller) position. `rope_angle` takes an unsigned position, so the
    // delta is applied as the negation of the positive `shift` rotation.
    var angle = -rope_angle(shift, d, head_dim, freq_base);
    if has_freq_factors == 1u {
        angle = angle / freq_factors[d];
    }
    let res = rotate_rope(x0, x1, angle);

    let dst_base = t_off * kv_dim + head_off;
    scratch[dst_base + e0] = res.x;
    scratch[dst_base + e1] = res.y;
}
