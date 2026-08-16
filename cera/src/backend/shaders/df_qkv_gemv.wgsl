// Fused QKV GEMV: splits output rows across out_q, out_k, out_v buffers in a single pass.
//
// QKV weight has (q_dim + 2*kv_dim) rows, k columns.
// Rows 0..q_dim-1                 -> out_q[r]
// Rows q_dim..q_dim+kv_dim-1       -> out_k[r - q_dim]
// Rows q_dim+kv_dim..total_rows-1 -> out_v[r - q_dim - kv_dim]

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_q: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_k: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_v: array<f32>;
@group(0) @binding(5) var<storage, read> params: vec4<u32>;
// params = vec4<u32>(q_dim, kv_dim, k_dim, 0u)

#include "common_decls.tmpl"

fn load_a_vec4(idx: u32) -> vec4<f32> {
    return vec4<f32>(a[idx], a[idx + 1u], a[idx + 2u], a[idx + 3u]);
}

const NR: u32 = 8u;
const WG_SIZE: u32 = 32u;

var<workgroup> partials: array<f32, 256>;

@compute @workgroup_size(32, 1, 1)
fn df_qkv_gemv(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let q_dim = params.x;
    let kv_dim = params.y;
    let k = params.z;
    let total_rows = q_dim + 2u * kv_dim;
    let tid = lid.x;
    let r0 = get_wid(wid) * NR;

    var sums: array<f32, 8>;
    for (var r = 0u; r < NR; r += 1u) {
        sums[r] = 0.0;
    }

    let k_vec = k & ~3u;
    var col = tid * 4u;
    while col < k_vec {
        let xv = vec4<f32>(x[col], x[col + 1u], x[col + 2u], x[col + 3u]);
        for (var r = 0u; r < NR; r += 1u) {
            if r0 + r < total_rows {
                let av = load_a_vec4((r0 + r) * k + col);
                sums[r] += dot(av, xv);
            }
        }
        col += 128u;
    }

    col = k_vec + tid;
    while col < k {
        let xv = x[col];
        for (var r = 0u; r < NR; r += 1u) {
            if r0 + r < total_rows {
                sums[r] += a[(r0 + r) * k + col] * xv;
            }
        }
        col += 32u;
    }

    for (var r = 0u; r < NR; r += 1u) {
        partials[r * WG_SIZE + tid] = sums[r];
    }
    workgroupBarrier();

    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            for (var r = 0u; r < NR; r += 1u) {
                let idx = r * WG_SIZE + tid;
                partials[idx] += partials[idx + stride];
            }
        }
        workgroupBarrier();
    }

    if tid == 0u {
        for (var r = 0u; r < NR; r += 1u) {
            let row = r0 + r;
            if row < q_dim {
                out_q[row] = partials[r * WG_SIZE];
            } else if row < q_dim + kv_dim {
                out_k[row - q_dim] = partials[r * WG_SIZE];
            } else if row < total_rows {
                out_v[row - q_dim - kv_dim] = partials[r * WG_SIZE];
            }
        }
    }
}
