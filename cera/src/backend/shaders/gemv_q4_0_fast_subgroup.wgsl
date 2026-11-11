// WGSL subgroup-reduction variant of `slang/gemv_q4_0_fast.slang` (entry `gemv_q4_0_fast`).
//
// NR=8 output rows per workgroup, 64 threads covering NQ=32 Q4_0 blocks per K-loop step.
// Uses `subgroupAdd` plus a 1-barrier cross-subgroup combine (0 barriers on wave64+)
// instead of the 6-barrier shared-memory tree in the portable fallback.

#ifdef SUBGROUP
#ifdef WEBGPU_SUBGROUPS
enable subgroups;
#endif
#endif

@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

#include "common_decls.tmpl"

const NR: u32 = 8u;
const NQ: u32 = 32u;
const WG_SIZE: u32 = 64u;
const BLOCK_BYTES: u32 = 18u;
const MAX_SUBGROUPS: u32 = 64u;

var<workgroup> sg_sums: array<f32, NR * MAX_SUBGROUPS>;

fn block_scale(blk_byte: u32) -> f32 {
    let word = w[blk_byte >> 2u];
    let scale_bits = select(word & 0xFFFFu, word >> 16u, (blk_byte & 2u) != 0u);
    return unpack2x16float(scale_bits).x;
}

fn u32_unaligned(off: u32) -> u32 {
    let wi = off >> 2u;
    let sh = (off & 3u) * 8u;
    let hi_sh = (32u - sh) & 31u;
    return (w[wi] >> sh) | select(w[wi + 1u] << hi_sh, 0u, sh == 0u);
}

fn q4_row_block_dot(
    blk_byte: u32,
    il: u32,
    y0: f32,
    y1: f32,
    y2: f32,
    y3: f32,
    y4: f32,
    y5: f32,
    y6: f32,
    y7: f32,
    y8: f32,
    y9: f32,
    y10: f32,
    y11: f32,
    y12: f32,
    y13: f32,
    y14: f32,
    y15: f32,
    sumy_total: f32,
) -> f32 {
    let d = block_scale(blk_byte);
    let qs_byte = blk_byte + 2u + il;
    let w0 = u32_unaligned(qs_byte);
    let w1 = u32_unaligned(qs_byte + 4u);

    let q0 = w0 & 0xFFFFu;
    let q1 = w0 >> 16u;
    let q2 = w1 & 0xFFFFu;
    let q3 = w1 >> 16u;

    var acc0 = y0 * f32(q0 & 0x000Fu);
    var acc1 = y1 * f32(q0 & 0x0F00u);
    var acc2 = y8 * f32(q0 & 0x00F0u);
    var acc3 = y9 * f32(q0 & 0xF000u);

    acc0 += y2 * f32(q1 & 0x000Fu);
    acc1 += y3 * f32(q1 & 0x0F00u);
    acc2 += y10 * f32(q1 & 0x00F0u);
    acc3 += y11 * f32(q1 & 0xF000u);

    acc0 += y4 * f32(q2 & 0x000Fu);
    acc1 += y5 * f32(q2 & 0x0F00u);
    acc2 += y12 * f32(q2 & 0x00F0u);
    acc3 += y13 * f32(q2 & 0xF000u);

    acc0 += y6 * f32(q3 & 0x000Fu);
    acc1 += y7 * f32(q3 & 0x0F00u);
    acc2 += y14 * f32(q3 & 0x00F0u);
    acc3 += y15 * f32(q3 & 0xF000u);

    return d * (sumy_total * -8.0 + acc0 + acc1 + acc2 + acc3);
}

@compute @workgroup_size(64, 1, 1)
fn gemv_q4_0_fast(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let m = params.x;
    let k = params.y;
    let r0 = get_wid(wid) * NR;
    if r0 >= m {
        return;
    }

    let tid = lid.x;
    let ix = tid >> 1u;
    let il = (tid & 1u) * 8u;
    let nb = k / 32u;
    let row_bytes = nb * BLOCK_BYTES;

    var sumf0: f32 = 0.0;
    var sumf1: f32 = 0.0;
    var sumf2: f32 = 0.0;
    var sumf3: f32 = 0.0;
    var sumf4: f32 = 0.0;
    var sumf5: f32 = 0.0;
    var sumf6: f32 = 0.0;
    var sumf7: f32 = 0.0;

    var yb_off = ix * 32u + il;
    var ib = ix;
    while ib < nb {
        let a0 = x[yb_off + 0u];
        let a1 = x[yb_off + 1u];
        let a2 = x[yb_off + 2u];
        let a3 = x[yb_off + 3u];
        let a4 = x[yb_off + 4u];
        let a5 = x[yb_off + 5u];
        let a6 = x[yb_off + 6u];
        let a7 = x[yb_off + 7u];
        let a8 = x[yb_off + 16u];
        let a9 = x[yb_off + 17u];
        let a10 = x[yb_off + 18u];
        let a11 = x[yb_off + 19u];
        let a12 = x[yb_off + 20u];
        let a13 = x[yb_off + 21u];
        let a14 = x[yb_off + 22u];
        let a15 = x[yb_off + 23u];

        let y0 = a0;
        let y1 = a1 * 0.00390625;
        let y2 = a2;
        let y3 = a3 * 0.00390625;
        let y4 = a4;
        let y5 = a5 * 0.00390625;
        let y6 = a6;
        let y7 = a7 * 0.00390625;
        let y8 = a8 * 0.0625;
        let y9 = a9 * 0.000244140625;
        let y10 = a10 * 0.0625;
        let y11 = a11 * 0.000244140625;
        let y12 = a12 * 0.0625;
        let y13 = a13 * 0.000244140625;
        let y14 = a14 * 0.0625;
        let y15 = a15 * 0.000244140625;

        let sumy0 = (a0 + a1) + (a2 + a3) + (a4 + a5) + (a6 + a7);
        let sumy1 = (a8 + a9) + (a10 + a11) + (a12 + a13) + (a14 + a15);
        let sumy_total = sumy0 + sumy1;

        let b_off = ib * BLOCK_BYTES;
        sumf0 += q4_row_block_dot(r0 * row_bytes + b_off, il, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, sumy_total);
        if r0 + 1u < m { sumf1 += q4_row_block_dot((r0 + 1u) * row_bytes + b_off, il, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, sumy_total); }
        if r0 + 2u < m { sumf2 += q4_row_block_dot((r0 + 2u) * row_bytes + b_off, il, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, sumy_total); }
        if r0 + 3u < m { sumf3 += q4_row_block_dot((r0 + 3u) * row_bytes + b_off, il, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, sumy_total); }
        if r0 + 4u < m { sumf4 += q4_row_block_dot((r0 + 4u) * row_bytes + b_off, il, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, sumy_total); }
        if r0 + 5u < m { sumf5 += q4_row_block_dot((r0 + 5u) * row_bytes + b_off, il, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, sumy_total); }
        if r0 + 6u < m { sumf6 += q4_row_block_dot((r0 + 6u) * row_bytes + b_off, il, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, sumy_total); }
        if r0 + 7u < m { sumf7 += q4_row_block_dot((r0 + 7u) * row_bytes + b_off, il, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, sumy_total); }

        ib += NQ;
        yb_off += NQ * 32u;
    }

    let s0 = subgroupAdd(sumf0);
    let s1 = subgroupAdd(sumf1);
    let s2 = subgroupAdd(sumf2);
    let s3 = subgroupAdd(sumf3);
    let s4 = subgroupAdd(sumf4);
    let s5 = subgroupAdd(sumf5);
    let s6 = subgroupAdd(sumf6);
    let s7 = subgroupAdd(sumf7);

    let nsg = (WG_SIZE + sg_size - 1u) / sg_size;
    if nsg == 1u {
        if tid == 0u {
            y[r0] = s0;
            if r0 + 1u < m { y[r0 + 1u] = s1; }
            if r0 + 2u < m { y[r0 + 2u] = s2; }
            if r0 + 3u < m { y[r0 + 3u] = s3; }
            if r0 + 4u < m { y[r0 + 4u] = s4; }
            if r0 + 5u < m { y[r0 + 5u] = s5; }
            if r0 + 6u < m { y[r0 + 6u] = s6; }
            if r0 + 7u < m { y[r0 + 7u] = s7; }
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
        y[r0] = t0;
        if r0 + 1u < m { y[r0 + 1u] = t1; }
        if r0 + 2u < m { y[r0 + 2u] = t2; }
        if r0 + 3u < m { y[r0 + 3u] = t3; }
        if r0 + 4u < m { y[r0 + 4u] = t4; }
        if r0 + 5u < m { y[r0 + 5u] = t5; }
        if r0 + 6u < m { y[r0 + 6u] = t6; }
        if r0 + 7u < m { y[r0 + 7u] = t7; }
    }
}
