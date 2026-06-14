// RoPE: Rotary Position Embedding applied to Q and K vectors.
//
// Each thread handles one (cos, sin) pair for one dimension pair in one head.
// Applied to both Q (n_heads) and K (n_kv_heads) concatenated in the same buffer:
//   q_and_k[0..n_heads*head_dim] = Q
//   q_and_k[n_heads*head_dim..] = K (only first n_kv_heads*head_dim used)
//
// Two pair layouts, selected by `rope_type` (matches `cpu::RopeType`):
//   0 = NEOX (split-halves): rotates (x[d], x[d + head_dim/2]). Qwen2/Qwen3/LFM2.
//   1 = NORM (interleaved):  rotates (x[2d], x[2d+1]).          LLaMA/Mistral/Granite.
// The per-pair angle schedule is identical; only the element pairing differs.
//
// `freq_factors` (Llama-3 `rope_freqs.weight`, length head_dim/2) optionally
// divides each pair's angle, gated by `has_freq_factors`. Bound as a 1-element
// dummy buffer when unused (NEOX archs never set it).
//
// Bind group 0:
//   @binding(0) q: array<f32>       (read-write, Q vector)
//   @binding(1) k: array<f32>       (read-write, K vector)
//   @binding(2) params: array<u32, 7>  (pos, n_heads, n_kv_heads, head_dim,
//                                        freq_base_bits, rope_type, has_freq_factors)
//   @binding(3) freq_factors: array<f32>  (head_dim/2 factors, or 1-elem dummy)
//
// Dispatch: (ceil(max(n_heads, n_kv_heads) * head_dim/2 / 256), 1, 1)

#include "common_decls.tmpl"

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32, 7>;
@group(0) @binding(3) var<storage, read> freq_factors: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn rope(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = params[0];
    let n_heads = params[1];
    let n_kv_heads = params[2];
    let head_dim = params[3];
    let freq_base = bitcast<f32>(params[4]);
    let rope_type = params[5];
    let has_freq_factors = params[6];

    let half_dim = head_dim / 2u;
    let idx = gid.x;

    // Apply to Q heads
    let q_total = n_heads * half_dim;
    if idx < q_total {
        let head = idx / half_dim;
        let d = idx % half_dim;
        var angle = rope_angle(pos, d, head_dim, freq_base);
        if has_freq_factors == 1u {
            angle = angle / freq_factors[d];
        }

        var i0: u32;
        var i1: u32;
        if rope_type == 0u {
            i0 = head * head_dim + d;
            i1 = head * head_dim + d + half_dim;
        } else {
            i0 = head * head_dim + 2u * d;
            i1 = head * head_dim + 2u * d + 1u;
        }
        let res = rotate_rope(q[i0], q[i1], angle);
        q[i0] = res.x;
        q[i1] = res.y;
    }

    // Apply to K heads
    let k_total = n_kv_heads * half_dim;
    if idx < k_total {
        let head = idx / half_dim;
        let d = idx % half_dim;
        var angle = rope_angle(pos, d, head_dim, freq_base);
        if has_freq_factors == 1u {
            angle = angle / freq_factors[d];
        }

        var i0: u32;
        var i1: u32;
        if rope_type == 0u {
            i0 = head * head_dim + d;
            i1 = head * head_dim + d + half_dim;
        } else {
            i0 = head * head_dim + 2u * d;
            i1 = head * head_dim + 2u * d + 1u;
        }
        let res = rotate_rope(k[i0], k[i1], angle);
        k[i0] = res.x;
        k[i1] = res.y;
    }
}
