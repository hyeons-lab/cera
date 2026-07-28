// Q6_K GEMV — Metal kernel layout with per-byte reads.
// Wired into the wgpu GEMV dispatch (Q6K weights stay quantized in VRAM,
// ~4.9× smaller than dequantizing to f32: 210 B / 256 elems ≈ 0.82 B/elem
// vs 4 B/elem). Compute is bound by per-byte u32 load+shift+mask overhead and
// regressed vs f32 on macOS wgpu in earlier measurement, so the primary win is
// VRAM/bandwidth (matters most on mobile Adreno/Mali); byte-extraction
// throughput is a future optimization.
//
// NR=2 rows per WG, 32 threads. Dispatch: ceil(m/2).

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

// `get_wid` flattens the 2-D dispatch grid so m > 65535*NR rows still map to
// distinct rows (gemv_workgroups folds the row overflow into wid.y).
#include "common_decls.tmpl"

const QK_K: u32 = 256u;
const Q6K_BYTES: u32 = 210u;
const NR: u32 = 2u;
const WG_SIZE: u32 = 32u;

var<workgroup> partials: array<f32, 64>;

fn rb(off: u32) -> u32 {
    return (a[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu;
}

// Four consecutive bytes at an arbitrary byte offset, as one u32.
//
// Q6_K blocks are 210 bytes, so a block's base is not 4-aligned and its byte
// runs straddle word boundaries — which is why this kernel originally read one
// byte at a time. Each `rb` costs a full `a[]` load, so four consecutive bytes
// loaded the same word up to four times over. This funnel-shifts two loads into
// the four bytes instead: half the loads for the same data, and the shift is
// register work.
//
// `select`, not a branch, on purpose. `sh` is NOT uniform across the workgroup:
// `b` starts at `ix = tiisg & 1u` and steps by 2, so adjacent lanes work on
// even/odd blocks, and `Q6K_BYTES % 4 == 2` makes their block bases differ by 2
// mod 4. An `if (sh == 0u)` would therefore split every workgroup 16/16 and
// serialize both halves, to save one load on the aligned half. Measured, because
// the branch looks like the cheaper form: it costs **41%** at the LM-head shape
// (120.8 -> 70.7 GB/s) and about a third everywhere else. Do not "simplify" this
// back to a branch.
//
// The unselected `a[w + 1u]` can read one word past the buffer when `off` is
// aligned and lands on the final word. That is well-defined in WGSL — storage
// accesses are bounds-checked — and the value is discarded by the `select`.
fn rw(off: u32) -> u32 {
    let w = off / 4u;
    let sh = (off & 3u) * 8u;
    // `x << 32u` is undefined in WGSL, so mask the high word out when the offset
    // is already aligned rather than shifting by 32.
    let hi_sh = (32u - sh) & 31u;
    return (a[w] >> sh) | select(a[w + 1u] << hi_sh, 0u, sh == 0u);
}

/// The four 6-bit weights sharing one high-bits byte, in the kernel's
/// `q6_1..q6_4` order (which maps to `sums.x..sums.w`).
fn q6_lane(q1: u32, q2: u32, qhv: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32((q1 & 0x0Fu) | ((qhv & 0x03u) << 4u)) - 32),
        f32(i32((q2 & 0x0Fu) | ((qhv & 0x0Cu) << 2u)) - 32),
        f32(i32((q1 >> 4u) | (qhv & 0x30u)) - 32),
        f32(i32((q2 >> 4u) | ((qhv & 0xC0u) >> 2u)) - 32),
    );
}

fn ri8(off: u32) -> i32 {
    let b = rb(off);
    return i32(b) - select(0, 256, (b & 0x80u) != 0u);
}

fn rf16(off: u32) -> f32 {
    let lo = rb(off);
    let hi = rb(off + 1u);
    return unpack2x16float(lo | (hi << 8u)).x;
}

@compute @workgroup_size(32, 1, 1)
fn gemv_q6_k(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let k = params.y;
    let nb = k / QK_K;
    let row_bytes = nb * Q6K_BYTES;
    let tiisg = lid.x;
    let first_row = get_wid(wid) * NR;

    let tid_l = tiisg / 2u;
    let ix  = tiisg & 1u;
    let ip  = tid_l >> 3u;
    let il  = tid_l & 7u;
    let l0  = 4u * il;
    let is_off = 8u * ip + l0 / 16u;

    let y_offset   = 128u * ip + l0;
    let q_offset_l = 64u * ip + l0;
    let q_offset_h = 32u * ip + l0;

    var sumf0: f32 = 0.0;
    var sumf1: f32 = 0.0;

    var b = ix;
    while b < nb {
        let yb = b * QK_K + y_offset;
        // Four `vec4`s of activations rather than an `array<f32, 16>` indexed by
        // the loop variable. A loop-indexed local array is not reliably promoted
        // to registers — same pathology #316 fixed in gemv_q4_0_fast / gemv_q4_k
        // — and these are read once per row per block in the hot path.
        let ylv0 = vec4<f32>(x[yb + 0u], x[yb + 32u], x[yb + 64u], x[yb + 96u]);
        let ylv1 = vec4<f32>(x[yb + 1u], x[yb + 33u], x[yb + 65u], x[yb + 97u]);
        let ylv2 = vec4<f32>(x[yb + 2u], x[yb + 34u], x[yb + 66u], x[yb + 98u]);
        let ylv3 = vec4<f32>(x[yb + 3u], x[yb + 35u], x[yb + 67u], x[yb + 99u]);

        for (var row = 0u; row < NR; row += 1u) {
            // Skip the weight reads for out-of-range rows: on an odd `m` the tail
            // workgroup's second row (`first_row + 1 == m`) would otherwise index
            // `bb = m * row_bytes + ...` past the end of the weight buffer. The
            // writes below are already guarded; sumf1 stays 0 for the skipped row.
            if first_row + row >= m {
                continue;
            }
            let bb = (first_row + row) * row_bytes + b * Q6K_BYTES;
            let ql1 = bb + q_offset_l;
            let ql2 = ql1 + 32u;
            let qh  = bb + 128u + q_offset_h;
            let sc  = bb + 192u + is_off;
            let d_off = bb + 208u;

            // Three word loads for the twelve bytes the old `l` loop read one
            // at a time. Unrolled in the same l = 0..3 order, so the f32
            // accumulation *order* is unchanged.
            //
            // Not bit-identical everywhere, though: as `vec4` madds this appears
            // to contract into FMA where the scalar form did not, so results can
            // differ in the last bits. Measured against the previous kernel on the
            // LM head (m=65536): max abs 2.0e-6, max rel 2.4e-3, cosine
            // 1.0000000000, same argmax. For reference the CPU-vs-GPU gap on the
            // same logits is 0.27 abs, five orders of magnitude larger.
            let w1 = rw(ql1);
            let w2 = rw(ql2);
            let wh = rw(qh);

            var sums = vec4<f32>(0.0);
            sums += q6_lane(w1 & 0xFFu, w2 & 0xFFu, wh & 0xFFu) * ylv0;
            sums += q6_lane((w1 >> 8u) & 0xFFu, (w2 >> 8u) & 0xFFu, (wh >> 8u) & 0xFFu) * ylv1;
            sums += q6_lane((w1 >> 16u) & 0xFFu, (w2 >> 16u) & 0xFFu, (wh >> 16u) & 0xFFu) * ylv2;
            sums += q6_lane(w1 >> 24u, w2 >> 24u, wh >> 24u) * ylv3;

            let dblk = rf16(d_off);
            let s0 = f32(ri8(sc));
            let s2 = f32(ri8(sc + 2u));
            let s4 = f32(ri8(sc + 4u));
            let s6 = f32(ri8(sc + 6u));
            let row_sum = dblk * (sums[0] * s0 + sums[1] * s2 + sums[2] * s4 + sums[3] * s6);

            if row == 0u { sumf0 += row_sum; }
            else { sumf1 += row_sum; }
        }
        b += 2u;
    }

    partials[0u * WG_SIZE + tiisg] = sumf0;
    partials[1u * WG_SIZE + tiisg] = sumf1;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if tiisg < stride {
            for (var r = 0u; r < NR; r += 1u) {
                let idx = r * WG_SIZE + tiisg;
                partials[idx] += partials[idx + stride];
            }
        }
        workgroupBarrier();
    }

    if tiisg == 0u {
        if first_row < m { y[first_row] = partials[0u * WG_SIZE]; }
        if first_row + 1u < m { y[first_row + 1u] = partials[1u * WG_SIZE]; }
    }
}
