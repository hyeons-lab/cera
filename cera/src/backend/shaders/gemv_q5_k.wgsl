// Q5_K GEMV: y[row] = Σ dequant(W_q5k[row, i]) × x[i].
// Matches cera's `dequantize_q5_k_block` / `vec_dot_q5_k_f32_scalar` (quant.rs)
// bit-for-bit.
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
// Per-byte u32 reads (like gemv_q6_k.wgsl) — correct but unpack-heavy; the win
// is keeping the weight quantized (~5.8× less VRAM/bandwidth than the f32 path).

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

const QK_K: u32 = 256u;
const Q5K_BYTES: u32 = 176u;
const NR: u32 = 2u;
const WG_SIZE: u32 = 32u;

var<workgroup> partials: array<f32, 64>; // NR * WG_SIZE

// Read one byte from the u32-addressed weight buffer.
fn rb(off: u32) -> u32 {
    return (a[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu;
}

// Read a little-endian f16 (two bytes) as f32.
fn rf16(off: u32) -> f32 {
    return unpack2x16float(rb(off) | (rb(off + 1u) << 8u)).x;
}

// 6-bit sub-block scale, sub in 0..8 (`decode_q4km_scales`). `so` = scales byte offset.
fn get_sc(so: u32, sub: u32) -> u32 {
    if sub < 4u {
        return rb(so + sub) & 63u;
    }
    return (rb(so + sub + 4u) & 0x0Fu) | ((rb(so + sub - 4u) >> 6u) << 4u);
}

// 6-bit sub-block min, sub in 0..8.
fn get_mn(so: u32, sub: u32) -> u32 {
    if sub < 4u {
        return rb(so + sub + 4u) & 63u;
    }
    return (rb(so + sub + 4u) >> 4u) | ((rb(so + sub) >> 6u) << 4u);
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
    let first_row = wid.x * NR;

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
        var acc: f32 = 0.0;
        for (var ib = 0u; ib < nb; ib += 1u) {
            let blk = rr * row_bytes + ib * Q5K_BYTES;
            let d = rf16(blk);
            let dmin = rf16(blk + 2u);
            let so = blk + 4u;
            let scale = d * f32(get_sc(so, sub));
            let minv = dmin * f32(get_mn(so, sub));

            let qs_off = blk + 48u + qbase;
            let qh_off = blk + 16u + qhbase;
            let xb = ib * QK_K + e0;

            for (var i = 0u; i < 8u; i += 1u) {
                let qb = rb(qs_off + i);
                let nib = select(qb >> 4u, qb & 0x0Fu, hi == 0u);
                let hib = select(0.0, 16.0, (rb(qh_off + i) & hbit) != 0u);
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
