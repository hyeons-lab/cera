// Fast Q4_0 GEMV ported from Metal/llama.cpp algorithm.
//
// Key optimizations vs gemv_q4_0.wgsl:
//  - 2 threads per block: each processes 16 elements (half block)
//  - Pre-scaled y values eliminate per-element bit shifts
//  - Sumy bias hoisting: delta * (sumy * -8 + acc)
//
// 32 threads, 4 rows per WG. Reduction via workgroup memory.
// Dispatch: ceil(m/4) workgroups.
//
// The staged activations are 16 NAMED scalars rather than an
// `array<f32, 16>` handed to a helper by pointer. That is not a style
// preference: taking `&yl` makes the array's address escape, so naga must give
// it a real `function`-address-space allocation and the backend compiler cannot
// promote it to registers — every one of the 16 reads per half-block becomes a
// memory access. `gemv_f32.wgsl` keeps `sums: array<f32, 8>` and runs fine
// precisely because nothing takes its address, so a loop-indexed local array is
// not by itself the problem; the pointer is. Same class of defect as the
// prefill GEMM's spilled accumulators (PR #311). Keep these as scalars, and do
// not refactor the unrolled body back into a helper that takes a pointer.
//
// The unroll preserves the original accumulation order exactly (acc0..acc3 over
// qi = 0,2,4,6), so output is bit-identical to the pointer version.

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

// `get_wid` flattens the 2-D dispatch grid so m > 65535*NR rows still map to
// distinct rows (gemv_workgroups folds the row overflow into wid.y).
#include "common_decls.tmpl"

const NR: u32 = 8u;
const NQ: u32 = 16u;
const WG_SIZE: u32 = 32u;

var<workgroup> partials: array<f32, 256>;
var<workgroup> x_stage: array<f32, 1024>;

// Read the f16 block scale at `blk_byte`. Q4_0 blocks are 18 B, so the scale is
// only 2-byte aligned and straddles a u32 boundary once every two blocks.
fn block_scale(blk_byte: u32) -> f32 {
    let word_off = blk_byte / 4u;
    let byte_rem = blk_byte % 4u;
    var scale_bits: u32;
    if byte_rem == 0u {
        scale_bits = a[word_off] & 0xFFFFu;
    } else if byte_rem == 1u {
        scale_bits = (a[word_off] >> 8u) & 0xFFFFu;
    } else if byte_rem == 2u {
        scale_bits = (a[word_off] >> 16u) & 0xFFFFu;
    } else {
        scale_bits = ((a[word_off] >> 24u) & 0xFFu) | ((a[word_off + 1u] & 0xFFu) << 8u);
    }
    return unpack2x16float(scale_bits).x;
}

// Read the u16 nibble pair at byte offset `byte_pos`. Operates on `a` only —
// no pointer to a local, so this stays inlinable without forcing a spill.
fn q_pair(byte_pos: u32) -> u32 {
    let w_off = byte_pos / 4u;
    let w_rem = byte_pos % 4u;
    if w_rem <= 2u {
        return (a[w_off] >> (w_rem * 8u)) & 0xFFFFu;
    }
    return ((a[w_off] >> 24u) & 0xFFu) | ((a[w_off + 1u] & 0xFFu) << 8u);
}

@compute @workgroup_size(32, 1, 1)
fn gemv_q4_0_fast(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let k = params.y;
    let nb = k / 32u;
    let row_bytes = nb * 18u;
    let r0 = get_wid(wid) * NR;
    let tid = lid.x;

    let ix = tid / 2u;
    let il = (tid & 1u) * 8u;

    var sumf: array<f32, 8>;
    sumf[0] = 0.0;
    sumf[1] = 0.0;
    sumf[2] = 0.0;
    sumf[3] = 0.0;
    sumf[4] = 0.0;
    sumf[5] = 0.0;
    sumf[6] = 0.0;
    sumf[7] = 0.0;

    let use_shmem = (k <= 1024u);
    if use_shmem {
        for (var i = tid; i < k; i += WG_SIZE) {
            x_stage[i] = x[i];
        }
        workgroupBarrier();
    }

    var yb_off: u32 = ix * 32u + il;

    var ib = ix;
    while ib < nb {
        var a0: f32; var a1: f32; var a2: f32; var a3: f32;
        var a4: f32; var a5: f32; var a6: f32; var a7: f32;
        var a8: f32; var a9: f32; var a10: f32; var a11: f32;
        var a12: f32; var a13: f32; var a14: f32; var a15: f32;
        if use_shmem {
            a0 = x_stage[yb_off + 0u];
            a1 = x_stage[yb_off + 1u];
            a2 = x_stage[yb_off + 2u];
            a3 = x_stage[yb_off + 3u];
            a4 = x_stage[yb_off + 4u];
            a5 = x_stage[yb_off + 5u];
            a6 = x_stage[yb_off + 6u];
            a7 = x_stage[yb_off + 7u];
            a8 = x_stage[yb_off + 16u];
            a9 = x_stage[yb_off + 17u];
            a10 = x_stage[yb_off + 18u];
            a11 = x_stage[yb_off + 19u];
            a12 = x_stage[yb_off + 20u];
            a13 = x_stage[yb_off + 21u];
            a14 = x_stage[yb_off + 22u];
            a15 = x_stage[yb_off + 23u];
        } else {
            a0 = x[yb_off + 0u];
            a1 = x[yb_off + 1u];
            a2 = x[yb_off + 2u];
            a3 = x[yb_off + 3u];
            a4 = x[yb_off + 4u];
            a5 = x[yb_off + 5u];
            a6 = x[yb_off + 6u];
            a7 = x[yb_off + 7u];
            a8 = x[yb_off + 16u];
            a9 = x[yb_off + 17u];
            a10 = x[yb_off + 18u];
            a11 = x[yb_off + 19u];
            a12 = x[yb_off + 20u];
            a13 = x[yb_off + 21u];
            a14 = x[yb_off + 22u];
            a15 = x[yb_off + 23u];
        }

        // Pre-scaled activations, staged in named registers (see header).
        let y0 = a0;
        let y1 = a1 / 256.0;
        let y2 = a2;
        let y3 = a3 / 256.0;
        let y4 = a4;
        let y5 = a5 / 256.0;
        let y6 = a6;
        let y7 = a7 / 256.0;
        let y8 = a8 / 16.0;
        let y9 = a9 / 4096.0;
        let y10 = a10 / 16.0;
        let y11 = a11 / 4096.0;
        let y12 = a12 / 16.0;
        let y13 = a13 / 4096.0;
        let y14 = a14 / 16.0;
        let y15 = a15 / 4096.0;

        // Same summation order as the original `sumy0 + sumy1` accumulation.
        let sumy0 = (a0 + a1) + (a2 + a3) + (a4 + a5) + (a6 + a7);
        let sumy1 = (a8 + a9) + (a10 + a11) + (a12 + a13) + (a14 + a15);
        let sumy_total = sumy0 + sumy1;

        for (var r = 0u; r < NR; r += 1u) {
            if r0 + r >= m { continue; }
            let blk_byte = (r0 + r) * row_bytes + ib * 18u;
            let d = block_scale(blk_byte);
            let qs_byte = blk_byte + 2u + il;

            var acc0: f32 = 0.0;
            var acc1: f32 = 0.0;
            var acc2: f32 = 0.0;
            var acc3: f32 = 0.0;

            let q0 = q_pair(qs_byte + 0u);
            acc0 += y0 * f32(q0 & 0x000Fu);
            acc1 += y1 * f32(q0 & 0x0F00u);
            acc2 += y8 * f32(q0 & 0x00F0u);
            acc3 += y9 * f32(q0 & 0xF000u);

            let q1 = q_pair(qs_byte + 2u);
            acc0 += y2 * f32(q1 & 0x000Fu);
            acc1 += y3 * f32(q1 & 0x0F00u);
            acc2 += y10 * f32(q1 & 0x00F0u);
            acc3 += y11 * f32(q1 & 0xF000u);

            let q2 = q_pair(qs_byte + 4u);
            acc0 += y4 * f32(q2 & 0x000Fu);
            acc1 += y5 * f32(q2 & 0x0F00u);
            acc2 += y12 * f32(q2 & 0x00F0u);
            acc3 += y13 * f32(q2 & 0xF000u);

            let q3 = q_pair(qs_byte + 6u);
            acc0 += y6 * f32(q3 & 0x000Fu);
            acc1 += y7 * f32(q3 & 0x0F00u);
            acc2 += y14 * f32(q3 & 0x00F0u);
            acc3 += y15 * f32(q3 & 0xF000u);

            sumf[r] += d * (sumy_total * -8.0 + acc0 + acc1 + acc2 + acc3);
        }

        yb_off += 32u * NQ;
        ib += NQ;
    }

    for (var r = 0u; r < NR; r += 1u) {
        partials[r * WG_SIZE + tid] = sumf[r];
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
                y[r0 + r] = partials[r * WG_SIZE];
            }
        }
    }
}
