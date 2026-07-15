// Q5_K GEMV: y[row] = Σ dequant(W_q5k[row, i]) × x[i].
// Uses the same dequant as cera's `dequantize_q5_k_block` /
// `vec_dot_q5_k_f32_scalar` (quant.rs); results match up to floating-point
// roundoff from the parallel (workgroup) reduction order.
//
// Q5_K super-block: 256 elements, 176 bytes:
//   d      — f16 super-block scale                 (bytes 0..2)
//   dmin   — f16 super-block min                   (bytes 2..4)
//   scales — 12 bytes, 6-bit packed sub-scales+mins (bytes 4..16)
//   qh     — 32 bytes, the 5th (high) bit of each quant (bytes 16..48)
//   qs     — 128 bytes, the low 4 bits of each quant    (bytes 48..176)
//
// 6-bit scale/min unpack is `get_scale_min_k4` (port of `decode_q4km_scales`,
// shared with Q4_K). Thread `t` (0..31) owns the 8 output elements [t*8, t*8+8)
// of each super-block, decodes that half-sub-block's scale/min, folds in the
// `qh` 5th bit, dots with x, and accumulates across all blocks. The 32 partials
// are reduced in workgroup memory. NR=2 rows per WG. Dispatch: ceil(m/2) × 32.
//
// Weight loads are vectorized to whole `u32` words (see gemv_q4_k.wgsl for the
// rationale — T5b measured the per-byte path ~4× off Adreno's achievable
// bandwidth). PRECONDITION: the Q5_K super-block is 176 bytes (a multiple of
// 16), so every block base and each of `d/dmin`, `scales`, the per-thread `qs`
// span and `qh` span is ≥4-byte aligned. The 2-byte-aligned blocks (Q6_K,
// Q4_0, Q8_0) do not satisfy this and keep the per-byte reads.

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

// `get_wid` flattens the 2-D dispatch grid so m > 65535*NR rows still map to
// distinct rows (gemv_workgroups folds the row overflow into wid.y).
#include "common_decls.tmpl"

const QK_K: u32 = 256u;
const Q5K_BYTES: u32 = 176u;
const NR: u32 = 2u;
const WG_SIZE: u32 = 32u;

var<workgroup> partials: array<f32, 64>; // NR * WG_SIZE

// Extract byte `b` (0..12) of the 12-byte scales array from its three
// preloaded words `s0`/`s1`/`s2` — equals the old `rb(scales_off + b)`.
fn scb(s0: u32, s1: u32, s2: u32, b: u32) -> u32 {
    let w = select(select(s2, s1, b < 8u), s0, b < 4u);
    return (w >> ((b & 3u) * 8u)) & 0xFFu;
}

// 6-bit sub-block scale, sub in 0..8 (`decode_q4km_scales`), from preloaded words.
fn get_sc(s0: u32, s1: u32, s2: u32, sub: u32) -> u32 {
    if sub < 4u {
        return scb(s0, s1, s2, sub) & 63u;
    }
    return (scb(s0, s1, s2, sub + 4u) & 0x0Fu) | ((scb(s0, s1, s2, sub - 4u) >> 6u) << 4u);
}

// 6-bit sub-block min, sub in 0..8.
fn get_mn(s0: u32, s1: u32, s2: u32, sub: u32) -> u32 {
    if sub < 4u {
        return scb(s0, s1, s2, sub + 4u) & 63u;
    }
    return (scb(s0, s1, s2, sub + 4u) >> 4u) | ((scb(s0, s1, s2, sub) >> 6u) << 4u);
}

@compute @workgroup_size(32, 1, 1)
fn gemv_q5_k(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let k = params.y;
    let nb = k / QK_K;
    let row_bytes = nb * Q5K_BYTES;

    let t = lid.x;
    let first_row = get_wid(wid) * NR;

    let e0 = t * 8u;          // this thread's 8 output elements [e0, e0+8)
    let j = e0 / 64u;         // 0..3
    let o = e0 % 64u;         // {0,8,16,24,32,40,48,56}
    let hi = o / 32u;         // 0 = low nibble, 1 = high nibble
    let sub = 2u * j + hi;    // 0..7 sub-block index
    let qbase = 32u * j + (o % 32u); // byte index into qs
    let qhbase = o % 32u;            // byte index into qh
    let hbit = 1u << sub;            // qh bit selector for this sub-block

    for (var row = 0u; row < NR; row += 1u) {
        let rr = first_row + row;
        // Skip the odd-tail row (rr == m when m is not a multiple of NR): its
        // y write is already guarded below, and skipping avoids out-of-range
        // weight-buffer reads. `partials` is zero-initialized, so the reduction
        // treats the skipped row as 0. `partials[0]` (row 0) is always valid —
        // the dispatch count ceil(m/NR) guarantees first_row < m.
        if rr >= m {
            continue;
        }
        var acc: f32 = 0.0;
        for (var ib = 0u; ib < nb; ib += 1u) {
            let blk = rr * row_bytes + ib * Q5K_BYTES;
            // d, dmin are the two f16 halves of the block's word 0.
            let ddm = unpack2x16float(a[blk / 4u]);
            let d = ddm.x;
            let dmin = ddm.y;
            // scales occupy bytes 4..16 → three words at word (blk/4 + 1).
            let sw = blk / 4u + 1u;
            let s0 = a[sw];
            let s1 = a[sw + 1u];
            let s2 = a[sw + 2u];
            let scale = d * f32(get_sc(s0, s1, s2, sub));
            let minv = dmin * f32(get_mn(s0, s1, s2, sub));

            // This thread's 8 low-nibble bytes (qs, base blk+48) and 8 high-bit
            // bytes (qh, base blk+16) are each 8 contiguous bytes; qbase/qhbase
            // are multiples of 8, so each span is exactly two words — load once.
            let qw = (blk + 48u + qbase) / 4u;
            let qw0 = a[qw];
            let qw1 = a[qw + 1u];
            let hw = (blk + 16u + qhbase) / 4u;
            let hw0 = a[hw];
            let hw1 = a[hw + 1u];
            let xb = ib * QK_K + e0;

            for (var i = 0u; i < 8u; i += 1u) {
                let sh = (i & 3u) * 8u;
                let qb = ((select(qw1, qw0, i < 4u)) >> sh) & 0xFFu;
                let nib = select(qb >> 4u, qb & 0x0Fu, hi == 0u);
                let hbyte = ((select(hw1, hw0, i < 4u)) >> sh) & 0xFFu;
                let hib = select(0.0, 16.0, (hbyte & hbit) != 0u);
                let q5 = f32(nib) + hib;
                acc += (scale * q5 - minv) * x[xb + i];
            }
        }
        partials[row * WG_SIZE + t] = acc;
    }

    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if t < stride {
            for (var r = 0u; r < NR; r += 1u) {
                let idx = r * WG_SIZE + t;
                partials[idx] += partials[idx + stride];
            }
        }
        workgroupBarrier();
    }

    if t == 0u {
        if first_row < m { y[first_row] = partials[0u * WG_SIZE]; }
        if first_row + 1u < m { y[first_row + 1u] = partials[1u * WG_SIZE]; }
    }
}
