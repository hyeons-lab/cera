// F32 GEMV: y[m] = A[m, k] × x[k]
//
// 8 rows per workgroup, 32 threads. x is loaded once per WG
// and reused across 8 rows — 8× less x bandwidth than 1-row-per-WG.
//
// Dispatch: 2-D grid (ceil(m/8) folded across x/y via `get_wid`) so m > 65535*8
// rows still map to distinct rows.

// Weight A. With `F16_A` defined (the `gemv_f16` pipeline) it is stored as f16,
// two values per u32, so the LM head takes half the VRAM; activations (`x`) and
// accumulation stay f32. Without it (`gemv_f32`) A is plain f32.
#ifdef F16_A
@group(0) @binding(0) var<storage, read> a: array<u32>;
#else
@group(0) @binding(0) var<storage, read> a: array<f32>;
#endif
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec4<u32>;

#include "common_decls.tmpl"

// Load weight element `idx` of the row-major A[m, k] as f32.
fn load_a(idx: u32) -> f32 {
#ifdef F16_A
    let pair = unpack2x16float(a[idx / 2u]);
    return select(pair.y, pair.x, (idx & 1u) == 0u);
#else
    return a[idx];
#endif
}

const NR: u32 = 8u;
const WG_SIZE: u32 = 32u;

var<workgroup> partials: array<f32, 256>;

@compute @workgroup_size(32, 1, 1)
fn gemv_f32(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let k = params.y;
    let row_base = params.z;
    let tid = lid.x;
    let r0 = get_wid(wid) * NR;

    var sums: array<f32, 8>;
    for (var r = 0u; r < NR; r += 1u) {
        sums[r] = 0.0;
    }

    // Each thread strides through k in steps of 32.
    var col = tid;
    while col < k {
        let xv = x[col];
        for (var r = 0u; r < NR; r += 1u) {
            if r0 + r < m {
                sums[r] += load_a((r0 + r) * k + col) * xv;
            }
        }
        col += 32u;
    }

    // Workgroup reduction. This is correct on adapters whose subgroups are
    // narrower than the 32-thread workgroup.
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
            if r0 + r < m {
                y[row_base + r0 + r] = partials[r * WG_SIZE];
            }
        }
    }
}

// F32 GEMV with accumulate: y[row] += dot(A[row, :], x). Same bindings, layout,
// and dispatch as `gemv_f32`; used for the LoRA up-projection epilogue
// (out += B_scaled · tmp), where `scale` is pre-folded into the uploaded B so
// no separate scale pass is needed.
@compute @workgroup_size(32, 1, 1)
fn gemv_f32_accum(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let k = params.y;
    let row_base = params.z;
    let tid = lid.x;
    let r0 = get_wid(wid) * NR;

    var sums: array<f32, 8>;
    for (var r = 0u; r < NR; r += 1u) {
        sums[r] = 0.0;
    }

    var col = tid;
    while col < k {
        let xv = x[col];
        for (var r = 0u; r < NR; r += 1u) {
            if r0 + r < m {
                sums[r] += load_a((r0 + r) * k + col) * xv;
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
            if r0 + r < m {
                y[row_base + r0 + r] += partials[r * WG_SIZE];
            }
        }
    }
}
