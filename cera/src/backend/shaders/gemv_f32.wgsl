// F32 GEMV: y[m] = A[m, k] × x[k]
//
// 8 rows per workgroup, 32 threads. x is loaded once per WG
// and reused across 8 rows — 8× less x bandwidth than 1-row-per-WG.
//
// Dispatch: 2-D grid (ceil(m/8) folded across x/y via `get_wid`) so m > 65535*8
// rows still map to distinct rows.

#ifdef SUBGROUP
#ifdef WEBGPU_SUBGROUPS
enable subgroups;
#endif
#endif

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
#ifdef IMMEDIATES
struct ImmParams {
    v: vec4<u32>,
}
var<immediate> imm: ImmParams;
#else
@group(0) @binding(3) var<storage, read> params: vec4<u32>;
#endif

fn get_params() -> vec4<u32> {
#ifdef IMMEDIATES
    return imm.v;
#else
    return params;
#endif
}

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

// Load weight vector (4 consecutive elements) of the row-major A[m, k] as vec4<f32>.
fn load_a_vec4(idx: u32) -> vec4<f32> {
#ifdef F16_A
    let p0 = unpack2x16float(a[idx / 2u]);
    let p1 = unpack2x16float(a[idx / 2u + 1u]);
    return vec4<f32>(p0.x, p0.y, p1.x, p1.y);
#else
    return vec4<f32>(a[idx], a[idx + 1u], a[idx + 2u], a[idx + 3u]);
#endif
}

const NR: u32 = 8u;

#ifdef SUBGROUP
const WG_SIZE: u32 = 64u;
const MAX_SUBGROUPS: u32 = 64u;

var<workgroup> sg_sums: array<f32, NR * MAX_SUBGROUPS>;

@compute @workgroup_size(64, 1, 1)
fn gemv_f32(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let p = get_params();
    let m = p.x;
    let k = p.y;
    let row_base = p.z;
    let r0 = get_wid(wid) * NR;
    if r0 >= m {
        return;
    }
    let tid = lid.x;

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;
    var sum4: f32 = 0.0;
    var sum5: f32 = 0.0;
    var sum6: f32 = 0.0;
    var sum7: f32 = 0.0;

    let k_vec = k & ~3u;
    var col = tid * 4u;
    while col < k_vec {
        let xv = vec4<f32>(x[col], x[col + 1u], x[col + 2u], x[col + 3u]);
        sum0 += dot(load_a_vec4(r0 * k + col), xv);
        if r0 + 1u < m { sum1 += dot(load_a_vec4((r0 + 1u) * k + col), xv); }
        if r0 + 2u < m { sum2 += dot(load_a_vec4((r0 + 2u) * k + col), xv); }
        if r0 + 3u < m { sum3 += dot(load_a_vec4((r0 + 3u) * k + col), xv); }
        if r0 + 4u < m { sum4 += dot(load_a_vec4((r0 + 4u) * k + col), xv); }
        if r0 + 5u < m { sum5 += dot(load_a_vec4((r0 + 5u) * k + col), xv); }
        if r0 + 6u < m { sum6 += dot(load_a_vec4((r0 + 6u) * k + col), xv); }
        if r0 + 7u < m { sum7 += dot(load_a_vec4((r0 + 7u) * k + col), xv); }
        col += 256u;
    }

    col = k_vec + tid;
    while col < k {
        let xv = x[col];
        sum0 += load_a(r0 * k + col) * xv;
        if r0 + 1u < m { sum1 += load_a((r0 + 1u) * k + col) * xv; }
        if r0 + 2u < m { sum2 += load_a((r0 + 2u) * k + col) * xv; }
        if r0 + 3u < m { sum3 += load_a((r0 + 3u) * k + col) * xv; }
        if r0 + 4u < m { sum4 += load_a((r0 + 4u) * k + col) * xv; }
        if r0 + 5u < m { sum5 += load_a((r0 + 5u) * k + col) * xv; }
        if r0 + 6u < m { sum6 += load_a((r0 + 6u) * k + col) * xv; }
        if r0 + 7u < m { sum7 += load_a((r0 + 7u) * k + col) * xv; }
        col += 64u;
    }

    let s0 = subgroupAdd(sum0);
    let s1 = subgroupAdd(sum1);
    let s2 = subgroupAdd(sum2);
    let s3 = subgroupAdd(sum3);
    let s4 = subgroupAdd(sum4);
    let s5 = subgroupAdd(sum5);
    let s6 = subgroupAdd(sum6);
    let s7 = subgroupAdd(sum7);

    let nsg = (WG_SIZE + sg_size - 1u) / sg_size;
    if nsg == 1u {
        if tid == 0u {
            y[row_base + r0] = s0;
            if r0 + 1u < m { y[row_base + r0 + 1u] = s1; }
            if r0 + 2u < m { y[row_base + r0 + 2u] = s2; }
            if r0 + 3u < m { y[row_base + r0 + 3u] = s3; }
            if r0 + 4u < m { y[row_base + r0 + 4u] = s4; }
            if r0 + 5u < m { y[row_base + r0 + 5u] = s5; }
            if r0 + 6u < m { y[row_base + r0 + 6u] = s6; }
            if r0 + 7u < m { y[row_base + r0 + 7u] = s7; }
        }
        return;
    }

    let sgid = tid / sg_size;
    if sg_lane == 0u {
        sg_sums[0u * MAX_SUBGROUPS + sgid] = s0;
        sg_sums[1u * MAX_SUBGROUPS + sgid] = s1;
        sg_sums[2u * MAX_SUBGROUPS + sgid] = s2;
        sg_sums[3u * MAX_SUBGROUPS + sgid] = s3;
        sg_sums[4u * MAX_SUBGROUPS + sgid] = s4;
        sg_sums[5u * MAX_SUBGROUPS + sgid] = s5;
        sg_sums[6u * MAX_SUBGROUPS + sgid] = s6;
        sg_sums[7u * MAX_SUBGROUPS + sgid] = s7;
    }
    workgroupBarrier();

    if tid == 0u {
        var t0: f32 = 0.0;
        var t1: f32 = 0.0;
        var t2: f32 = 0.0;
        var t3: f32 = 0.0;
        var t4: f32 = 0.0;
        var t5: f32 = 0.0;
        var t6: f32 = 0.0;
        var t7: f32 = 0.0;
        for (var sg = 0u; sg < nsg; sg += 1u) {
            t0 += sg_sums[0u * MAX_SUBGROUPS + sg];
            t1 += sg_sums[1u * MAX_SUBGROUPS + sg];
            t2 += sg_sums[2u * MAX_SUBGROUPS + sg];
            t3 += sg_sums[3u * MAX_SUBGROUPS + sg];
            t4 += sg_sums[4u * MAX_SUBGROUPS + sg];
            t5 += sg_sums[5u * MAX_SUBGROUPS + sg];
            t6 += sg_sums[6u * MAX_SUBGROUPS + sg];
            t7 += sg_sums[7u * MAX_SUBGROUPS + sg];
        }
        y[row_base + r0] = t0;
        if r0 + 1u < m { y[row_base + r0 + 1u] = t1; }
        if r0 + 2u < m { y[row_base + r0 + 2u] = t2; }
        if r0 + 3u < m { y[row_base + r0 + 3u] = t3; }
        if r0 + 4u < m { y[row_base + r0 + 4u] = t4; }
        if r0 + 5u < m { y[row_base + r0 + 5u] = t5; }
        if r0 + 6u < m { y[row_base + r0 + 6u] = t6; }
        if r0 + 7u < m { y[row_base + r0 + 7u] = t7; }
    }
}

@compute @workgroup_size(64, 1, 1)
fn gemv_f32_accum(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let p = get_params();
    let m = p.x;
    let k = p.y;
    let row_base = p.z;
    let r0 = get_wid(wid) * NR;
    if r0 >= m {
        return;
    }
    let tid = lid.x;

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;
    var sum4: f32 = 0.0;
    var sum5: f32 = 0.0;
    var sum6: f32 = 0.0;
    var sum7: f32 = 0.0;

    let k_vec = k & ~3u;
    var col = tid * 4u;
    while col < k_vec {
        let xv = vec4<f32>(x[col], x[col + 1u], x[col + 2u], x[col + 3u]);
        sum0 += dot(load_a_vec4(r0 * k + col), xv);
        if r0 + 1u < m { sum1 += dot(load_a_vec4((r0 + 1u) * k + col), xv); }
        if r0 + 2u < m { sum2 += dot(load_a_vec4((r0 + 2u) * k + col), xv); }
        if r0 + 3u < m { sum3 += dot(load_a_vec4((r0 + 3u) * k + col), xv); }
        if r0 + 4u < m { sum4 += dot(load_a_vec4((r0 + 4u) * k + col), xv); }
        if r0 + 5u < m { sum5 += dot(load_a_vec4((r0 + 5u) * k + col), xv); }
        if r0 + 6u < m { sum6 += dot(load_a_vec4((r0 + 6u) * k + col), xv); }
        if r0 + 7u < m { sum7 += dot(load_a_vec4((r0 + 7u) * k + col), xv); }
        col += 256u;
    }

    col = k_vec + tid;
    while col < k {
        let xv = x[col];
        sum0 += load_a(r0 * k + col) * xv;
        if r0 + 1u < m { sum1 += load_a((r0 + 1u) * k + col) * xv; }
        if r0 + 2u < m { sum2 += load_a((r0 + 2u) * k + col) * xv; }
        if r0 + 3u < m { sum3 += load_a((r0 + 3u) * k + col) * xv; }
        if r0 + 4u < m { sum4 += load_a((r0 + 4u) * k + col) * xv; }
        if r0 + 5u < m { sum5 += load_a((r0 + 5u) * k + col) * xv; }
        if r0 + 6u < m { sum6 += load_a((r0 + 6u) * k + col) * xv; }
        if r0 + 7u < m { sum7 += load_a((r0 + 7u) * k + col) * xv; }
        col += 64u;
    }

    let s0 = subgroupAdd(sum0);
    let s1 = subgroupAdd(sum1);
    let s2 = subgroupAdd(sum2);
    let s3 = subgroupAdd(sum3);
    let s4 = subgroupAdd(sum4);
    let s5 = subgroupAdd(sum5);
    let s6 = subgroupAdd(sum6);
    let s7 = subgroupAdd(sum7);

    let nsg = (WG_SIZE + sg_size - 1u) / sg_size;
    if nsg == 1u {
        if tid == 0u {
            y[row_base + r0] += s0;
            if r0 + 1u < m { y[row_base + r0 + 1u] += s1; }
            if r0 + 2u < m { y[row_base + r0 + 2u] += s2; }
            if r0 + 3u < m { y[row_base + r0 + 3u] += s3; }
            if r0 + 4u < m { y[row_base + r0 + 4u] += s4; }
            if r0 + 5u < m { y[row_base + r0 + 5u] += s5; }
            if r0 + 6u < m { y[row_base + r0 + 6u] += s6; }
            if r0 + 7u < m { y[row_base + r0 + 7u] += s7; }
        }
        return;
    }

    let sgid = tid / sg_size;
    if sg_lane == 0u {
        sg_sums[0u * MAX_SUBGROUPS + sgid] = s0;
        sg_sums[1u * MAX_SUBGROUPS + sgid] = s1;
        sg_sums[2u * MAX_SUBGROUPS + sgid] = s2;
        sg_sums[3u * MAX_SUBGROUPS + sgid] = s3;
        sg_sums[4u * MAX_SUBGROUPS + sgid] = s4;
        sg_sums[5u * MAX_SUBGROUPS + sgid] = s5;
        sg_sums[6u * MAX_SUBGROUPS + sgid] = s6;
        sg_sums[7u * MAX_SUBGROUPS + sgid] = s7;
    }
    workgroupBarrier();

    if tid == 0u {
        var t0: f32 = 0.0;
        var t1: f32 = 0.0;
        var t2: f32 = 0.0;
        var t3: f32 = 0.0;
        var t4: f32 = 0.0;
        var t5: f32 = 0.0;
        var t6: f32 = 0.0;
        var t7: f32 = 0.0;
        for (var sg = 0u; sg < nsg; sg += 1u) {
            t0 += sg_sums[0u * MAX_SUBGROUPS + sg];
            t1 += sg_sums[1u * MAX_SUBGROUPS + sg];
            t2 += sg_sums[2u * MAX_SUBGROUPS + sg];
            t3 += sg_sums[3u * MAX_SUBGROUPS + sg];
            t4 += sg_sums[4u * MAX_SUBGROUPS + sg];
            t5 += sg_sums[5u * MAX_SUBGROUPS + sg];
            t6 += sg_sums[6u * MAX_SUBGROUPS + sg];
            t7 += sg_sums[7u * MAX_SUBGROUPS + sg];
        }
        y[row_base + r0] += t0;
        if r0 + 1u < m { y[row_base + r0 + 1u] += t1; }
        if r0 + 2u < m { y[row_base + r0 + 2u] += t2; }
        if r0 + 3u < m { y[row_base + r0 + 3u] += t3; }
        if r0 + 4u < m { y[row_base + r0 + 4u] += t4; }
        if r0 + 5u < m { y[row_base + r0 + 5u] += t5; }
        if r0 + 6u < m { y[row_base + r0 + 6u] += t6; }
        if r0 + 7u < m { y[row_base + r0 + 7u] += t7; }
    }
}
#else
const WG_SIZE: u32 = 32u;

var<workgroup> partials: array<f32, 256>;

@compute @workgroup_size(32, 1, 1)
fn gemv_f32(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let p = get_params();
    let m = p.x;
    let k = p.y;
    let row_base = p.z;
    let tid = lid.x;
    let r0 = get_wid(wid) * NR;
    if r0 >= m {
        return;
    }

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;
    var sum4: f32 = 0.0;
    var sum5: f32 = 0.0;
    var sum6: f32 = 0.0;
    var sum7: f32 = 0.0;

    let k_vec = k & ~3u;
    var col = tid * 4u;
    while col < k_vec {
        let xv = vec4<f32>(x[col], x[col + 1u], x[col + 2u], x[col + 3u]);
        sum0 += dot(load_a_vec4(r0 * k + col), xv);
        if r0 + 1u < m { sum1 += dot(load_a_vec4((r0 + 1u) * k + col), xv); }
        if r0 + 2u < m { sum2 += dot(load_a_vec4((r0 + 2u) * k + col), xv); }
        if r0 + 3u < m { sum3 += dot(load_a_vec4((r0 + 3u) * k + col), xv); }
        if r0 + 4u < m { sum4 += dot(load_a_vec4((r0 + 4u) * k + col), xv); }
        if r0 + 5u < m { sum5 += dot(load_a_vec4((r0 + 5u) * k + col), xv); }
        if r0 + 6u < m { sum6 += dot(load_a_vec4((r0 + 6u) * k + col), xv); }
        if r0 + 7u < m { sum7 += dot(load_a_vec4((r0 + 7u) * k + col), xv); }
        col += 128u;
    }

    col = k_vec + tid;
    while col < k {
        let xv = x[col];
        sum0 += load_a(r0 * k + col) * xv;
        if r0 + 1u < m { sum1 += load_a((r0 + 1u) * k + col) * xv; }
        if r0 + 2u < m { sum2 += load_a((r0 + 2u) * k + col) * xv; }
        if r0 + 3u < m { sum3 += load_a((r0 + 3u) * k + col) * xv; }
        if r0 + 4u < m { sum4 += load_a((r0 + 4u) * k + col) * xv; }
        if r0 + 5u < m { sum5 += load_a((r0 + 5u) * k + col) * xv; }
        if r0 + 6u < m { sum6 += load_a((r0 + 6u) * k + col) * xv; }
        if r0 + 7u < m { sum7 += load_a((r0 + 7u) * k + col) * xv; }
        col += 32u;
    }

    partials[0u * WG_SIZE + tid] = sum0;
    partials[1u * WG_SIZE + tid] = sum1;
    partials[2u * WG_SIZE + tid] = sum2;
    partials[3u * WG_SIZE + tid] = sum3;
    partials[4u * WG_SIZE + tid] = sum4;
    partials[5u * WG_SIZE + tid] = sum5;
    partials[6u * WG_SIZE + tid] = sum6;
    partials[7u * WG_SIZE + tid] = sum7;
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
        y[row_base + r0] = partials[0u * WG_SIZE];
        if r0 + 1u < m { y[row_base + r0 + 1u] = partials[1u * WG_SIZE]; }
        if r0 + 2u < m { y[row_base + r0 + 2u] = partials[2u * WG_SIZE]; }
        if r0 + 3u < m { y[row_base + r0 + 3u] = partials[3u * WG_SIZE]; }
        if r0 + 4u < m { y[row_base + r0 + 4u] = partials[4u * WG_SIZE]; }
        if r0 + 5u < m { y[row_base + r0 + 5u] = partials[5u * WG_SIZE]; }
        if r0 + 6u < m { y[row_base + r0 + 6u] = partials[6u * WG_SIZE]; }
        if r0 + 7u < m { y[row_base + r0 + 7u] = partials[7u * WG_SIZE]; }
    }
}

// F32 GEMV with accumulate: y[row] += dot(A[row, :], x). Same bindings, layout,
// and dispatch as `gemv_f32`; used for the LoRA up-projection epilogue.
@compute @workgroup_size(32, 1, 1)
fn gemv_f32_accum(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let p = get_params();
    let m = p.x;
    let k = p.y;
    let row_base = p.z;
    let tid = lid.x;
    let r0 = get_wid(wid) * NR;
    if r0 >= m {
        return;
    }

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;
    var sum4: f32 = 0.0;
    var sum5: f32 = 0.0;
    var sum6: f32 = 0.0;
    var sum7: f32 = 0.0;

    let k_vec = k & ~3u;
    var col = tid * 4u;
    while col < k_vec {
        let xv = vec4<f32>(x[col], x[col + 1u], x[col + 2u], x[col + 3u]);
        sum0 += dot(load_a_vec4(r0 * k + col), xv);
        if r0 + 1u < m { sum1 += dot(load_a_vec4((r0 + 1u) * k + col), xv); }
        if r0 + 2u < m { sum2 += dot(load_a_vec4((r0 + 2u) * k + col), xv); }
        if r0 + 3u < m { sum3 += dot(load_a_vec4((r0 + 3u) * k + col), xv); }
        if r0 + 4u < m { sum4 += dot(load_a_vec4((r0 + 4u) * k + col), xv); }
        if r0 + 5u < m { sum5 += dot(load_a_vec4((r0 + 5u) * k + col), xv); }
        if r0 + 6u < m { sum6 += dot(load_a_vec4((r0 + 6u) * k + col), xv); }
        if r0 + 7u < m { sum7 += dot(load_a_vec4((r0 + 7u) * k + col), xv); }
        col += 128u;
    }

    col = k_vec + tid;
    while col < k {
        let xv = x[col];
        sum0 += load_a(r0 * k + col) * xv;
        if r0 + 1u < m { sum1 += load_a((r0 + 1u) * k + col) * xv; }
        if r0 + 2u < m { sum2 += load_a((r0 + 2u) * k + col) * xv; }
        if r0 + 3u < m { sum3 += load_a((r0 + 3u) * k + col) * xv; }
        if r0 + 4u < m { sum4 += load_a((r0 + 4u) * k + col) * xv; }
        if r0 + 5u < m { sum5 += load_a((r0 + 5u) * k + col) * xv; }
        if r0 + 6u < m { sum6 += load_a((r0 + 6u) * k + col) * xv; }
        if r0 + 7u < m { sum7 += load_a((r0 + 7u) * k + col) * xv; }
        col += 32u;
    }

    partials[0u * WG_SIZE + tid] = sum0;
    partials[1u * WG_SIZE + tid] = sum1;
    partials[2u * WG_SIZE + tid] = sum2;
    partials[3u * WG_SIZE + tid] = sum3;
    partials[4u * WG_SIZE + tid] = sum4;
    partials[5u * WG_SIZE + tid] = sum5;
    partials[6u * WG_SIZE + tid] = sum6;
    partials[7u * WG_SIZE + tid] = sum7;
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
        y[row_base + r0] += partials[0u * WG_SIZE];
        if r0 + 1u < m { y[row_base + r0 + 1u] += partials[1u * WG_SIZE]; }
        if r0 + 2u < m { y[row_base + r0 + 2u] += partials[2u * WG_SIZE]; }
        if r0 + 3u < m { y[row_base + r0 + 3u] += partials[3u * WG_SIZE]; }
        if r0 + 4u < m { y[row_base + r0 + 4u] += partials[4u * WG_SIZE]; }
        if r0 + 5u < m { y[row_base + r0 + 5u] += partials[5u * WG_SIZE]; }
        if r0 + 6u < m { y[row_base + r0 + 6u] += partials[6u * WG_SIZE]; }
        if r0 + 7u < m { y[row_base + r0 + 7u] += partials[7u * WG_SIZE]; }
    }
}
#endif
