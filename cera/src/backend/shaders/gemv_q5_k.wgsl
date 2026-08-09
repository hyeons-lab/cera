// Q5_K GEMV: y[row] = Σ dequant(W_q5k[row, i]) × x[i].
// Uses the same dequant as cera's `dequantize_q5_k_block` /
// `vec_dot_q5_k_f32_scalar` (quant.rs); results match up to floating-point
// roundoff from two sources: the parallel (workgroup) reduction order, and the
// per-super-block refactoring of the min term described below.
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
// The eight per-element terms are unrolled into eight `q5v` calls with literal
// byte shifts. An earlier version picked the source word with
// `select(qw1, qw0, i < 4u)` inside an `i` loop: an operand chosen by a loop
// variable is a memory access rather than a register read, a shape that has
// cost this codebase 12-15x in the prefill GEMM and ~1.4x in two other decode
// GEMVs. For the same reason the two rows are unrolled into scalar `acc0`/
// `acc1` instead of an `array<f32, NR>` indexed by a loop variable.
//
// The unroll also lets the `dmin*min` bias be applied once per super-block as
// `minv * xsum` rather than subtracted from every element. That is
// `gemv_q4_k.metal`'s `sumy` term, and `gemv_q4_0_fast.wgsl` hoists Q4_0's
// fixed -8 offset the same way. Staging the activations across rows, by
// contrast, is not new: `gemv_q4_k.wgsl` and `gemv_q6_k.wgsl` both already do
// it. See `q5k_block_dot`.
//
// Weight loads are vectorized to whole `u32` words (see gemv_q4_k.wgsl for the
// rationale — T5b measured the per-byte path ~4× off Adreno's achievable
// bandwidth). PRECONDITION: the Q5_K super-block is 176 bytes (a multiple of
// 16), so every block base and each of `d/dmin`, `scales`, the per-thread `qs`
// span and `qh` span is ≥4-byte aligned, so the loads below need no funnel
// shift. Blocks whose size is only 2-byte aligned have to fund one themselves,
// and the siblings differ on how.

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

// `get_wid` flattens the 2-D dispatch grid so m > 65535*NR rows still map to
// distinct rows (gemv_workgroups folds the row overflow into wid.y).
#include "common_decls.tmpl"

const QK_K: u32 = 256u;
const Q5K_BYTES: u32 = 176u;
// NR is not a knob: the row loop is unrolled by hand, so the accumulators and
// writes below hardcode 2, as do `gpu_lfm2.rs`'s dispatch table,
// `wgpu_gemv_bench`, and `test_gpu_gemv_q5_k`'s `m.div_ceil(2) + 2`, where the
// surplus is deliberate and exists to drive the early return below. Changing NR
// alone compiles and silently drops rows.
const NR: u32 = 2u;
const WG_SIZE: u32 = 32u;

var<workgroup> partials: array<f32, NR * WG_SIZE>;

// Extract byte `b` (0..=11) of the 12-byte scales array from its three
// preloaded words `s0`/`s1`/`s2` — equals the old `rb(scales_off + b)`.
fn scb(s0: u32, s1: u32, s2: u32, b: u32) -> u32 {
    let w = select(select(s2, s1, b < 8u), s0, b < 4u);
    return (w >> ((b & 3u) * 8u)) & 0xFFu;
}

// 6-bit sub-block scale, sub in 0..=7 (`decode_q4km_scales`), from preloaded words.
fn get_sc(s0: u32, s1: u32, s2: u32, sub: u32) -> u32 {
    if sub < 4u {
        return scb(s0, s1, s2, sub) & 63u;
    }
    return (scb(s0, s1, s2, sub + 4u) & 0x0Fu) | ((scb(s0, s1, s2, sub - 4u) >> 6u) << 4u);
}

// 6-bit sub-block min, sub in 0..=7.
fn get_mn(s0: u32, s1: u32, s2: u32, sub: u32) -> u32 {
    if sub < 4u {
        return scb(s0, s1, s2, sub + 4u) & 63u;
    }
    return (scb(s0, s1, s2, sub + 4u) >> 4u) | ((scb(s0, s1, s2, sub) >> 6u) << 4u);
}

// One 5-bit weight: byte `sh` of the low-nibble word `qw`, promoted by the
// matching bit of the same byte of the qh word `hw`. `nsh` is 0 for the low
// nibble and 4 for the high one, and `hbit` is this sub-block's qh selector;
// both are fixed per thread. Callers MUST pass a literal `sh` (see the header).
fn q5v(qw: u32, hw: u32, sh: u32, nsh: u32, hbit: u32) -> f32 {
    let nib = ((qw >> sh) >> nsh) & 0x0Fu;
    // hbit <= 0x80, so it can only select a bit of the byte at `sh`; no mask
    // to 0xFF is needed before the test.
    let hib = select(0.0, 16.0, ((hw >> sh) & hbit) != 0u);
    return f32(nib) + hib;
}

// Dot this thread's eight weights from super-block `blk` of one row against the
// activations already held in registers, then apply the block's scale and min.
// `xsum` is Σx over those eight elements, which turns the per-element
// `- dmin*min` into a single multiply-subtract per super-block.
fn q5k_block_dot(
    blk: u32,
    sub: u32,
    qbase: u32,
    qhbase: u32,
    nsh: u32,
    hbit: u32,
    xl: vec4<f32>,
    xh: vec4<f32>,
    xsum: f32,
) -> f32 {
    // d, dmin are the two f16 halves of the block's word 0.
    let ddm = unpack2x16float(a[blk / 4u]);
    // scales occupy bytes 4..16 → three words at word (blk/4 + 1).
    let sw = blk / 4u + 1u;
    let s0 = a[sw];
    let s1 = a[sw + 1u];
    let s2 = a[sw + 2u];
    let scale = ddm.x * f32(get_sc(s0, s1, s2, sub));
    let minv = ddm.y * f32(get_mn(s0, s1, s2, sub));

    // This thread's 8 low-nibble bytes (qs, base blk+48) and 8 high-bit bytes
    // (qh, base blk+16) are each 8 contiguous bytes; qbase/qhbase are multiples
    // of 8, so each span is exactly two words. Load once.
    let qw = (blk + 48u + qbase) / 4u;
    let qw0 = a[qw];
    let qw1 = a[qw + 1u];
    let hw = (blk + 16u + qhbase) / 4u;
    let hw0 = a[hw];
    let hw1 = a[hw + 1u];

    let qsum =
        q5v(qw0, hw0, 0u, nsh, hbit) * xl.x
        + q5v(qw0, hw0, 8u, nsh, hbit) * xl.y
        + q5v(qw0, hw0, 16u, nsh, hbit) * xl.z
        + q5v(qw0, hw0, 24u, nsh, hbit) * xl.w
        + q5v(qw1, hw1, 0u, nsh, hbit) * xh.x
        + q5v(qw1, hw1, 8u, nsh, hbit) * xh.y
        + q5v(qw1, hw1, 16u, nsh, hbit) * xh.z
        + q5v(qw1, hw1, 24u, nsh, hbit) * xh.w;

    return scale * qsum - minv * xsum;
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
    let nsh = hi * 4u;               // nibble shift: 0 for low, 4 for high

    // Whole-workgroup early out. `gemv_row_workgroups` rounds the grid to
    // (min(count, MAX_WG), ceil(count / MAX_WG)), which overshoots once count
    // exceeds MAX_WG: a 151936-row LM head needs 75968 groups and gets 131070.
    // The surplus groups have first_row >= m. This return is REQUIRED, not a
    // cost optimization: `row0` below is unclamped and `y[first_row]` is written
    // unguarded, both of which are only sound because `first_row < m` holds from
    // here down. A surplus group falling through would read past the weight
    // buffer and write garbage over the last valid row of y. The per-row
    // `rr < m` skip this rewrite dropped is what used to make them free.
    // `first_row` is workgroup-uniform, so the return is uniform and therefore
    // safe ahead of the barriers below.
    if first_row >= m {
        return;
    }

    // Clamp the odd-tail second row (first_row + 1 == m when m is not a
    // multiple of NR) instead of branching on it, so the super-block loop stays
    // branch-free; its result is dropped by the guarded write. Same construct
    // as `gemv_q4_k.metal`'s and `gemv_q6_k.metal`'s `min(first_row + r, m-1)`,
    // and a deliberate divergence from the `rr < m` skip the WGSL siblings use:
    // with the rows unrolled, that skip would put a branch between the staged
    // activations and their second consumer.
    let row0 = first_row * row_bytes;
    let row1 = min(first_row + 1u, m - 1u) * row_bytes;

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;

    for (var ib = 0u; ib < nb; ib += 1u) {
        // Load this thread's eight activations once per super-block and reuse
        // them across both rows, so x is read once per row group rather than
        // once per row.
        let xb = ib * QK_K + e0;
        let xl = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
        let xh = vec4<f32>(x[xb + 4u], x[xb + 5u], x[xb + 6u], x[xb + 7u]);
        let xsum = dot(xl, vec4<f32>(1.0)) + dot(xh, vec4<f32>(1.0));

        let off = ib * Q5K_BYTES;
        acc0 += q5k_block_dot(row0 + off, sub, qbase, qhbase, nsh, hbit, xl, xh, xsum);
        acc1 += q5k_block_dot(row1 + off, sub, qbase, qhbase, nsh, hbit, xl, xh, xsum);
    }

    partials[t] = acc0;
    partials[WG_SIZE + t] = acc1;

    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if t < stride {
            partials[t] += partials[t + stride];
            partials[WG_SIZE + t] += partials[WG_SIZE + t + stride];
        }
        workgroupBarrier();
    }

    if t == 0u {
        // `first_row < m` is guaranteed by the early out; only the second row
        // can be past the end.
        y[first_row] = partials[0u];
        if first_row + 1u < m { y[first_row + 1u] = partials[WG_SIZE]; }
    }
}
