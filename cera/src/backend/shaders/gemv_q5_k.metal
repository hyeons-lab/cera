#include <metal_stdlib>
using namespace metal;

// Q5_K (Q5_K_M) GEMV: y[row] = Σ dequant(W_q5k[row, i]) × x[i].
// Matches cera's `dequantize_q5_k_block` (quant.rs) bit-for-bit.
//
// Q5_K super-block: 256 elements, 176 bytes total:
//   d       — f16 super-block scale        (bytes 0..2)
//   dmin    — f16 super-block min           (bytes 2..4)
//   scales  — 12 bytes packed 6-bit sub-scales + mins (bytes 4..16)
//   qh      — 32 bytes, the 5th (high) bit of all 256 quants (bytes 16..48)
//   qs      — 128 bytes, 256 low-4-bit quants (bytes 48..176)
//
// Q5_K is Q4_K plus the `qh` plane, and shares its 6-bit scale/min packing
// (`decode_q4km_scales`), so this is `gemv_q4_k.metal` with one extra term.
//
// Dequant (per sub-block j in 0..4, l in 0..32):
//   out[64j + l]      = d*sc[2j]   * ((qs[32j+l] & 0xF) + 16*bit(qh[l], 2j))   - dmin*mn[2j]
//   out[64j + l + 32] = d*sc[2j+1] * ((qs[32j+l] >> 4 ) + 16*bit(qh[l], 2j+1)) - dmin*mn[2j+1]
//
// Note `qh` is indexed by `l` alone — the same 32 bytes are reused by all four
// j iterations, which consume bit pairs (2j, 2j+1). That is the `u1`/`u2 <<= 2`
// walk in the CPU reference, flattened here to a direct shift by `2j + hi`.
//
// Layout: NR=2 rows/TG, 32 threads (one simdgroup). Thread `t` owns the 8 output
// elements [t*8, t*8+8) of each super-block — all 8 fall in one sub-block/nibble —
// decodes that sub-block's scale/min, dots with x, then simd_sum reduces the row.
// Dispatch: ceil(m/2) threadgroups × 32 threads.
//
// NR stays at 2 to match `gemv_q4_k`. That is inherited, not tuned here: the
// widening to 4/8 was measured on the *wgpu* Q4_K twin and regressed there
// (#316, 47.8 -> 45.9 -> 45.3 tok/s) — at m≈1024 the larger tile leaves too few
// threadgroups to fill the GPU, and `x` stays cached, so the activation re-reads
// a bigger tile would save never reach DRAM. Neither the backend nor the dtype
// matches this kernel, so treat it as an untested default rather than a result;
// Metal Q5_K has not been swept.

constant constexpr uint QK_K = 256;
constant constexpr uint Q5K_BYTES = 176;
constant constexpr short NR = 2;

struct Params { uint m; uint k; };

// 6-bit sub-block scale / min unpack — port of `decode_q4km_scales` in quant.rs.
// Identical to `gemv_q4_k.metal`: Q5_K reuses Q4_K's scale packing verbatim.
static inline uchar q5k_get_sc(device const uchar* s, uint sub) {
    return sub < 4 ? (s[sub] & 63u)
                   : ((s[sub + 4] & 0x0Fu) | ((s[sub - 4] >> 6) << 4));
}
static inline uchar q5k_get_mn(device const uchar* s, uint sub) {
    return sub < 4 ? (s[sub + 4] & 63u)
                   : ((s[sub + 4] >> 4) | ((s[sub] >> 6) << 4));
}

static inline float gemv_q5_k_row_dot(
    device const uchar* row_ptr,
    device const float* x,
    uint nb,
    uint e0, uint sub, uint qbase, uint hi, uint hbase
) {
    float sumf = 0.0f;
    for (uint ib = 0; ib < nb; ib++) {
        device const uchar* blk = row_ptr + ib * Q5K_BYTES;
        float d    = float(*(device const half*)(blk));
        float dmin = float(*(device const half*)(blk + 2));
        device const uchar* scales = blk + 4;
        device const uchar* qh = blk + 16;
        device const uchar* qs = blk + 48;

        float scale = d * float(q5k_get_sc(scales, sub));
        float minv  = dmin * float(q5k_get_mn(scales, sub));

        device const float* xb = x + ib * QK_K + e0;
        for (uint i = 0; i < 8u; i++) {
            uchar qb = qs[qbase + i];
            uint nib = (hi == 0u) ? uint(qb & 0x0Fu) : uint(qb >> 4);
            uint hbit = uint((qh[hbase + i] >> sub) & 1u);
            sumf += (scale * float(nib + 16u * hbit) - minv) * xb[i];
        }
    }
    return sumf;
}

kernel void gemv_q5_k(
    const device uchar* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    const uint m = params.m;
    const uint nb = params.k / QK_K;
    const uint row_bytes = nb * Q5K_BYTES;
    // Linearize the 2-D dispatch grid (`sz2d(min(groups,65535), ceil(groups/65535))`)
    // so `m > 65535 * NR` still maps every threadgroup to a distinct row.
    const uint tgi = tg_id.x + tg_id.y * 65535u;

    const uint e0 = tiisg * 8u;      // 0,8,...,248 across the 256-element block
    const uint j = e0 / 64u;         // 0..3
    const uint o = e0 % 64u;         // {0,8,16,24,32,40,48,56}
    const uint hi = o / 32u;         // nibble half (0 = low, 1 = high)
    // 0..7 sub-block index. Doubles as the `qh` bit selector: the CPU
    // reference's u1 = 1 << 2j and u2 = 1 << (2j + 1) are exactly 1 << sub.
    const uint sub = 2u * j + hi;
    const uint qbase = 32u * j + (o % 32u);
    const uint hbase = o % 32u;      // qh is indexed by `l` only, not by 32j + l

    for (short r = 0; r < NR; r++) {
        const uint row = tgi * NR + r;
        if (row >= m) continue;
        float sumf = gemv_q5_k_row_dot(a + row * row_bytes, x, nb, e0, sub, qbase, hi, hbase);
        float total = simd_sum(sumf);
        if (tiisg == 0u) {
            y[row] = total;
        }
    }
}

kernel void gemv_q5_k_accum(
    const device uchar* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    const uint m = params.m;
    const uint nb = params.k / QK_K;
    const uint row_bytes = nb * Q5K_BYTES;
    const uint tgi = tg_id.x + tg_id.y * 65535u;

    const uint e0 = tiisg * 8u;
    const uint j = e0 / 64u;
    const uint o = e0 % 64u;
    const uint hi = o / 32u;
    const uint sub = 2u * j + hi;    // also the `qh` bit selector — see above
    const uint qbase = 32u * j + (o % 32u);
    const uint hbase = o % 32u;

    for (short r = 0; r < NR; r++) {
        const uint row = tgi * NR + r;
        if (row >= m) continue;
        float sumf = gemv_q5_k_row_dot(a + row * row_bytes, x, nb, e0, sub, qbase, hi, hbase);
        float total = simd_sum(sumf);
        if (tiisg == 0u) {
            y[row] += total;
        }
    }
}
