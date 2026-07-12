// Q4_K (Q4_K_M) GEMV — port of gemv_q4_k.metal to the workgroup-reduction idiom
// used by gemv_q6_k.wgsl (WGSL has no portable simd_sum). Matches cera's
// `dequantize_q4_k_m_block` (quant.rs) bit-for-bit.
//
// Q4_K super-block: 256 elements, 144 bytes:
//   d      — f16 super-block scale (bytes 0..2)
//   dmin   — f16 super-block min   (bytes 2..4)
//   scales — 12 bytes 6-bit packed sub-scales + mins (bytes 4..16)
//   qs     — 128 bytes, 256 4-bit quants (bytes 16..144)
//
// Dequant (sub-block j in 0..4, l in 0..32):
//   out[64j + l]      = d*sc[2j]   * (qs[32j+l] & 0xF) - dmin*mn[2j]
//   out[64j + l + 32] = d*sc[2j+1] * (qs[32j+l] >> 4 ) - dmin*mn[2j+1]
//
// NR=2 rows per workgroup, 32 threads. Thread `t` owns the 8 output elements
// [t*8, t*8+8) of each super-block — all 8 fall in one sub-block/nibble — dots
// them across every block, then a workgroup tree-reduction sums the 32 threads.
// Dispatch: ceil(m/2) workgroups. Win is VRAM/bandwidth (Q4_K stays quantized,
// ~7× smaller than f32); compute is bound by per-byte unpack.

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

const QK_K: u32 = 256u;
const Q4K_BYTES: u32 = 144u;
const NR: u32 = 2u;
const WG_SIZE: u32 = 32u;

var<workgroup> partials: array<f32, 64>;

// Read byte `off` from the byte-addressed `a` buffer.
fn rb(off: u32) -> u32 {
    return (a[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu;
}

// Read an f16 (little-endian) at byte offset `off` and widen to f32.
fn rf16(off: u32) -> f32 {
    let lo = rb(off);
    let hi = rb(off + 1u);
    return unpack2x16float(lo | (hi << 8u)).x;
}

// 6-bit sub-block scale / min unpack — port of `decode_q4km_scales` (quant.rs).
// `sb` is the byte offset of the 12-byte scales array.
fn q4k_sc(sb: u32, sub: u32) -> u32 {
    if sub < 4u {
        return rb(sb + sub) & 63u;
    }
    return (rb(sb + sub + 4u) & 0x0Fu) | ((rb(sb + sub - 4u) >> 6u) << 4u);
}

fn q4k_mn(sb: u32, sub: u32) -> u32 {
    if sub < 4u {
        return rb(sb + sub + 4u) & 63u;
    }
    return (rb(sb + sub + 4u) >> 4u) | ((rb(sb + sub) >> 6u) << 4u);
}

@compute @workgroup_size(32, 1, 1)
fn gemv_q4_k(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let k = params.y;
    let nb = k / QK_K;
    let row_bytes = nb * Q4K_BYTES;
    let tiisg = lid.x;
    let first_row = wid.x * NR;

    let e0 = tiisg * 8u;      // 0,8,...,248 across the 256-element block
    let j = e0 / 64u;         // 0..3
    let o = e0 % 64u;         // {0,8,16,24,32,40,48,56}
    let hi = o / 32u;         // nibble half (0 = low, 1 = high)
    let sub = 2u * j + hi;    // 0..7 sub-block index
    let qbase = 32u * j + (o % 32u);

    var sumf0: f32 = 0.0;
    var sumf1: f32 = 0.0;

    for (var ib = 0u; ib < nb; ib += 1u) {
        let x_off = ib * QK_K + e0;
        var xl: array<f32, 8>;
        for (var i = 0u; i < 8u; i += 1u) {
            xl[i] = x[x_off + i];
        }

        for (var r = 0u; r < NR; r += 1u) {
            let row = first_row + r;
            if row >= m {
                continue;
            }
            let blk = row * row_bytes + ib * Q4K_BYTES;
            let d = rf16(blk);
            let dmin = rf16(blk + 2u);
            let sb = blk + 4u;
            let qs = blk + 16u;

            let scale = d * f32(q4k_sc(sb, sub));
            let minv = dmin * f32(q4k_mn(sb, sub));

            var s = 0.0;
            for (var i = 0u; i < 8u; i += 1u) {
                let qb = rb(qs + qbase + i);
                let nib = select(qb >> 4u, qb & 0x0Fu, hi == 0u);
                s += (scale * f32(nib) - minv) * xl[i];
            }
            if r == 0u { sumf0 += s; } else { sumf1 += s; }
        }
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
