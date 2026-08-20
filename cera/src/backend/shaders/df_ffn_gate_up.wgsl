// Fused FFN Gate (W1) + Up (W3) GEMV with inline SiLU multiplication
// y[r] = silu(W1[r, :] · x) * (W3[r, :] · x)
//
// W13 contains [W1 (ffn_dim x k), W3 (ffn_dim x k)] contiguously.
// 8 rows per workgroup, 32 threads.

@group(0) @binding(0) var<storage, read> w13: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec4<u32>;

#include "common_decls.tmpl"

fn load_w13_vec4(idx: u32) -> vec4<f32> {
    return vec4<f32>(w13[idx], w13[idx + 1u], w13[idx + 2u], w13[idx + 3u]);
}

fn silu(v: f32) -> f32 {
    return v / (1.0 + exp(-v));
}

const NR: u32 = 8u;
const WG_SIZE: u32 = 32u;

var<workgroup> partials_gate: array<f32, 256>;
var<workgroup> partials_up: array<f32, 256>;

@compute @workgroup_size(32, 1, 1)
fn df_ffn_gate_up(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let ffn_dim = params.x;
    let k = params.y;
    let tid = lid.x;
    let r0 = get_wid(wid) * NR;
    let w3_offset = ffn_dim * k;

    var sums_gate: array<f32, 8>;
    var sums_up: array<f32, 8>;
    for (var r = 0u; r < NR; r += 1u) {
        sums_gate[r] = 0.0;
        sums_up[r] = 0.0;
    }

    // 4-wide vectorized dot product across x
    let k_vec = k & ~3u;
    var col = tid * 4u;
    while col < k_vec {
        let xv = vec4<f32>(x[col], x[col + 1u], x[col + 2u], x[col + 3u]);
        for (var r = 0u; r < NR; r += 1u) {
            if r0 + r < ffn_dim {
                let row_offset = (r0 + r) * k + col;
                let w1_v = load_w13_vec4(row_offset);
                let w3_v = load_w13_vec4(w3_offset + row_offset);
                sums_gate[r] += dot(w1_v, xv);
                sums_up[r] += dot(w3_v, xv);
            }
        }
        col += 128u;
    }

    // Scalar tail
    col = k_vec + tid;
    while col < k {
        let xv = x[col];
        for (var r = 0u; r < NR; r += 1u) {
            if r0 + r < ffn_dim {
                let row_offset = (r0 + r) * k + col;
                sums_gate[r] += w13[row_offset] * xv;
                sums_up[r] += w13[w3_offset + row_offset] * xv;
            }
        }
        col += 32u;
    }

    // Workgroup reduction
    for (var r = 0u; r < NR; r += 1u) {
        partials_gate[r * WG_SIZE + tid] = sums_gate[r];
        partials_up[r * WG_SIZE + tid] = sums_up[r];
    }
    workgroupBarrier();

    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            for (var r = 0u; r < NR; r += 1u) {
                let idx = r * WG_SIZE + tid;
                partials_gate[idx] += partials_gate[idx + stride];
                partials_up[idx] += partials_up[idx + stride];
            }
        }
        workgroupBarrier();
    }

    if tid == 0u {
        for (var r = 0u; r < NR; r += 1u) {
            if r0 + r < ffn_dim {
                let g = partials_gate[r * WG_SIZE];
                let u = partials_up[r * WG_SIZE];
                y[r0 + r] = silu(g) * u;
            }
        }
    }
}
