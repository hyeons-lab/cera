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

const NR: u32 = 4u;
const NQ: u32 = 16u;
const WG_SIZE: u32 = 32u;

var<workgroup> partials: array<f32, 128>;

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

    var sumf: array<f32, 4>;
    sumf[0] = 0.0;
    sumf[1] = 0.0;
    sumf[2] = 0.0;
    sumf[3] = 0.0;

    var yb_off: u32 = ix * 32u + il;

    var ib = ix;
    while ib < nb {
        // Stage 16 pre-scaled activations in named registers (see header).
        let y0 = x[yb_off + 0u];
        let y1 = x[yb_off + 1u] / 256.0;
        let y2 = x[yb_off + 2u];
        let y3 = x[yb_off + 3u] / 256.0;
        let y4 = x[yb_off + 4u];
        let y5 = x[yb_off + 5u] / 256.0;
        let y6 = x[yb_off + 6u];
        let y7 = x[yb_off + 7u] / 256.0;
        let y8 = x[yb_off + 16u] / 16.0;
        let y9 = x[yb_off + 17u] / 4096.0;
        let y10 = x[yb_off + 18u] / 16.0;
        let y11 = x[yb_off + 19u] / 4096.0;
        let y12 = x[yb_off + 20u] / 16.0;
        let y13 = x[yb_off + 21u] / 4096.0;
        let y14 = x[yb_off + 22u] / 16.0;
        let y15 = x[yb_off + 23u] / 4096.0;

        // Same summation order as the original `sumy0 + sumy1` accumulation.
        let sumy0 = (x[yb_off + 0u] + x[yb_off + 1u])
            + (x[yb_off + 2u] + x[yb_off + 3u])
            + (x[yb_off + 4u] + x[yb_off + 5u])
            + (x[yb_off + 6u] + x[yb_off + 7u]);
        let sumy1 = (x[yb_off + 16u] + x[yb_off + 17u])
            + (x[yb_off + 18u] + x[yb_off + 19u])
            + (x[yb_off + 20u] + x[yb_off + 21u])
            + (x[yb_off + 22u] + x[yb_off + 23u]);
        let sumy_total = sumy0 + sumy1;

        for (var r = 0u; r < NR; r += 1u) {
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
        if r0 + 0u < m { y[r0 + 0u] = partials[0u * WG_SIZE]; }
        if r0 + 1u < m { y[r0 + 1u] = partials[1u * WG_SIZE]; }
        if r0 + 2u < m { y[r0 + 2u] = partials[2u * WG_SIZE]; }
        if r0 + 3u < m { y[r0 + 3u] = partials[3u * WG_SIZE]; }
    }
}
