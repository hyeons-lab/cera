#ifdef SUBGROUP
#ifdef WEBGPU_SUBGROUPS
enable subgroups;
#endif
#endif

#define Q8_0_HELPERS
#include "common_decls.tmpl"

// Q8_0 GEMV: y[m] = dequant(A_q8_0[m, k]) * x[k].
//
// Q8_0 block layout and the dequant math live in common_decls.tmpl
// (Q8_0_HELPERS): get_u32_at + process_block_q8_0.
//
// One workgroup handles 8 rows. With `SUBGROUP`, 64 threads cover 8 Q8_0 blocks
// per K-loop step (8 lanes per 32-element block, 4 elements per lane in registers,
// zero barriers inside the K-loop) and reduce via `subgroupAdd` + 1 barrier.

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

const ROWS_PER_WG: u32 = 8u;
const QK: u32 = 32u;

fn q8_scale(blk_byte: u32) -> f32 {
    let word = a[blk_byte >> 2u];
    let half_bits = select(word & 0xFFFFu, word >> 16u, (blk_byte & 2u) != 0u);
    return unpack2x16float(half_bits).x;
}

fn u32_unaligned(off: u32) -> u32 {
    let w = off >> 2u;
    let sh = (off & 3u) * 8u;
    let hi_sh = (32u - sh) & 31u;
    return (a[w] >> sh) | select(a[w + 1u] << hi_sh, 0u, sh == 0u);
}

fn q8_lane_dot(blk_byte: u32, q_off: u32, xv: vec4<f32>) -> f32 {
    let scale = q8_scale(blk_byte);
    let packed = u32_unaligned(blk_byte + 2u + q_off);
    let sp = bitcast<i32>(packed);
    let qv = vec4<f32>(
        f32(extractBits(sp, 0u, 8u)),
        f32(extractBits(sp, 8u, 8u)),
        f32(extractBits(sp, 16u, 8u)),
        f32(extractBits(sp, 24u, 8u)),
    );
    return dot(qv, xv) * scale;
}

#ifdef SUBGROUP
const WG_SIZE: u32 = 64u;
const BLOCKS_PER_ITER: u32 = 32u;
const MAX_SUBGROUPS: u32 = 64u;

var<workgroup> sg_sums: array<f32, ROWS_PER_WG * MAX_SUBGROUPS>;

fn unpack_i8x4(packed: u32) -> vec4<f32> {
    let sp = bitcast<i32>(packed);
    return vec4<f32>(
        f32(extractBits(sp, 0u, 8u)),
        f32(extractBits(sp, 8u, 8u)),
        f32(extractBits(sp, 16u, 8u)),
        f32(extractBits(sp, 24u, 8u)),
    );
}

fn q8_half_block_dot(
    blk_byte: u32,
    il: u32,
    xv0: vec4<f32>,
    xv1: vec4<f32>,
    xv2: vec4<f32>,
    xv3: vec4<f32>,
) -> f32 {
    let scale = q8_scale(blk_byte);
    let qs_byte = blk_byte + 2u + il;
    let w = qs_byte >> 2u;
    let hi = (qs_byte & 2u) != 0u;
    let r0 = a[w];
    let r1 = a[w + 1u];
    let r2 = a[w + 2u];
    let r3 = a[w + 3u];
    let r4 = a[w + 4u];
    let p0 = select(r0, (r0 >> 16u) | (r1 << 16u), hi);
    let p1 = select(r1, (r1 >> 16u) | (r2 << 16u), hi);
    let p2 = select(r2, (r2 >> 16u) | (r3 << 16u), hi);
    let p3 = select(r3, (r3 >> 16u) | (r4 << 16u), hi);
    let acc = dot(unpack_i8x4(p0), xv0)
        + dot(unpack_i8x4(p1), xv1)
        + dot(unpack_i8x4(p2), xv2)
        + dot(unpack_i8x4(p3), xv3);
    return acc * scale;
}

@compute @workgroup_size(64, 1, 1)
fn gemv_q8_0(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let m = params.x;
    let k = params.y;
    let row_base = get_wid(wid) * ROWS_PER_WG;
    if row_base >= m {
        return;
    }

    let tid = lid.x;
    let ix = tid >> 1u;
    let il = (tid & 1u) * 16u;

    let nb = k / QK;
    let row_bytes = nb * 34u;

    let rb0 = row_base * row_bytes;
    let rb1 = min(row_base + 1u, m - 1u) * row_bytes;
    let rb2 = min(row_base + 2u, m - 1u) * row_bytes;
    let rb3 = min(row_base + 3u, m - 1u) * row_bytes;
    let rb4 = min(row_base + 4u, m - 1u) * row_bytes;
    let rb5 = min(row_base + 5u, m - 1u) * row_bytes;
    let rb6 = min(row_base + 6u, m - 1u) * row_bytes;
    let rb7 = min(row_base + 7u, m - 1u) * row_bytes;

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;
    var sum4: f32 = 0.0;
    var sum5: f32 = 0.0;
    var sum6: f32 = 0.0;
    var sum7: f32 = 0.0;

    var ib = ix;
    while ib < nb {
        let xb = ib * QK + il;
        let xv0 = vec4<f32>(x[xb + 0u], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
        let xv1 = vec4<f32>(x[xb + 4u], x[xb + 5u], x[xb + 6u], x[xb + 7u]);
        let xv2 = vec4<f32>(x[xb + 8u], x[xb + 9u], x[xb + 10u], x[xb + 11u]);
        let xv3 = vec4<f32>(x[xb + 12u], x[xb + 13u], x[xb + 14u], x[xb + 15u]);
        let b_off = ib * 34u;

        sum0 += q8_half_block_dot(rb0 + b_off, il, xv0, xv1, xv2, xv3);
        sum1 += q8_half_block_dot(rb1 + b_off, il, xv0, xv1, xv2, xv3);
        sum2 += q8_half_block_dot(rb2 + b_off, il, xv0, xv1, xv2, xv3);
        sum3 += q8_half_block_dot(rb3 + b_off, il, xv0, xv1, xv2, xv3);
        sum4 += q8_half_block_dot(rb4 + b_off, il, xv0, xv1, xv2, xv3);
        sum5 += q8_half_block_dot(rb5 + b_off, il, xv0, xv1, xv2, xv3);
        sum6 += q8_half_block_dot(rb6 + b_off, il, xv0, xv1, xv2, xv3);
        sum7 += q8_half_block_dot(rb7 + b_off, il, xv0, xv1, xv2, xv3);

        ib += BLOCKS_PER_ITER;
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
            y[row_base] = s0;
            if row_base + 1u < m { y[row_base + 1u] = s1; }
            if row_base + 2u < m { y[row_base + 2u] = s2; }
            if row_base + 3u < m { y[row_base + 3u] = s3; }
            if row_base + 4u < m { y[row_base + 4u] = s4; }
            if row_base + 5u < m { y[row_base + 5u] = s5; }
            if row_base + 6u < m { y[row_base + 6u] = s6; }
            if row_base + 7u < m { y[row_base + 7u] = s7; }
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
        y[row_base] = t0;
        if row_base + 1u < m { y[row_base + 1u] = t1; }
        if row_base + 2u < m { y[row_base + 2u] = t2; }
        if row_base + 3u < m { y[row_base + 3u] = t3; }
        if row_base + 4u < m { y[row_base + 4u] = t4; }
        if row_base + 5u < m { y[row_base + 5u] = t5; }
        if row_base + 6u < m { y[row_base + 6u] = t6; }
        if row_base + 7u < m { y[row_base + 7u] = t7; }
    }
}
#else
const WG_SIZE: u32 = 32u;
const BLOCKS_PER_ITER: u32 = 4u;

var<workgroup> partials: array<f32, 256>;

@compute @workgroup_size(32, 1, 1)
fn gemv_q8_0(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let k = params.y;
    let row_base = get_wid(wid) * ROWS_PER_WG;
    if row_base >= m {
        return;
    }

    let tid = lid.x;
    let lane_in_blk = tid & 7u;
    let blk_slot = tid >> 3u;
    let q_off = lane_in_blk * 4u;

    let nb = k / QK;
    let row_bytes = nb * 34u;

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;
    var sum4: f32 = 0.0;
    var sum5: f32 = 0.0;
    var sum6: f32 = 0.0;
    var sum7: f32 = 0.0;

    var ib = blk_slot;
    while ib < nb {
        let xb = ib * QK + q_off;
        let xv = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
        let b_off = ib * 34u;

        sum0 += q8_lane_dot(row_base * row_bytes + b_off, q_off, xv);
        if row_base + 1u < m { sum1 += q8_lane_dot((row_base + 1u) * row_bytes + b_off, q_off, xv); }
        if row_base + 2u < m { sum2 += q8_lane_dot((row_base + 2u) * row_bytes + b_off, q_off, xv); }
        if row_base + 3u < m { sum3 += q8_lane_dot((row_base + 3u) * row_bytes + b_off, q_off, xv); }
        if row_base + 4u < m { sum4 += q8_lane_dot((row_base + 4u) * row_bytes + b_off, q_off, xv); }
        if row_base + 5u < m { sum5 += q8_lane_dot((row_base + 5u) * row_bytes + b_off, q_off, xv); }
        if row_base + 6u < m { sum6 += q8_lane_dot((row_base + 6u) * row_bytes + b_off, q_off, xv); }
        if row_base + 7u < m { sum7 += q8_lane_dot((row_base + 7u) * row_bytes + b_off, q_off, xv); }

        ib += BLOCKS_PER_ITER;
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
            for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
                let idx = r * WG_SIZE + tid;
                partials[idx] += partials[idx + stride];
            }
        }
        workgroupBarrier();
    }

    if tid == 0u {
        y[row_base] = partials[0u * WG_SIZE];
        if row_base + 1u < m { y[row_base + 1u] = partials[1u * WG_SIZE]; }
        if row_base + 2u < m { y[row_base + 2u] = partials[2u * WG_SIZE]; }
        if row_base + 3u < m { y[row_base + 3u] = partials[3u * WG_SIZE]; }
        if row_base + 4u < m { y[row_base + 4u] = partials[4u * WG_SIZE]; }
        if row_base + 5u < m { y[row_base + 5u] = partials[5u * WG_SIZE]; }
        if row_base + 6u < m { y[row_base + 6u] = partials[6u * WG_SIZE]; }
        if row_base + 7u < m { y[row_base + 7u] = partials[7u * WG_SIZE]; }
    }
}
#endif
