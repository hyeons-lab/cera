#include <metal_stdlib>
using namespace metal;

// Q4_K (Q4_K_M) GEMV: y[row] = Σ dequant(W_q4k[row, i]) × x[i].
// Matches cera's `dequantize_q4_k_m_block` (quant.rs) bit-for-bit.
//
// Q4_K super-block: 256 elements, 144 bytes total:
//   d       — f16 super-block scale        (bytes 0..2)
//   dmin    — f16 super-block min           (bytes 2..4)
//   scales  — 12 bytes packed 6-bit sub-scales + mins (bytes 4..16)
//   qs      — 128 bytes, 256 4-bit quants   (bytes 16..144)
//
// Dequant (per sub-block j in 0..4, l in 0..32):
//   out[64j + l]      = d*sc[2j]   * (qs[32j+l] & 0xF) - dmin*mn[2j]
//   out[64j + l + 32] = d*sc[2j+1] * (qs[32j+l] >> 4 ) - dmin*mn[2j+1]
//
// Layout: NR=2 rows/TG, 32 threads (one simdgroup). Thread `t` owns the 8 output
// elements [t*8, t*8+8) of each super-block — all 8 fall in one sub-block/nibble —
// decodes that sub-block's scale/min, dots with x, then simd_sum reduces the row.
// Dispatch: ceil(m/2) threadgroups × 32 threads.

constant constexpr uint QK_K = 256;
constant constexpr uint Q4K_BYTES = 144;
constant constexpr short NR = 2;

struct Params { uint m; uint k; };

// 6-bit sub-block scale / min unpack — port of `decode_q4km_scales` (quant.rs:236).
static inline uchar q4k_get_sc(device const uchar* s, uint sub) {
    return sub < 4 ? (s[sub] & 63u)
                   : ((s[sub + 4] & 0x0Fu) | ((s[sub - 4] >> 6) << 4));
}
static inline uchar q4k_get_mn(device const uchar* s, uint sub) {
    return sub < 4 ? (s[sub + 4] & 63u)
                   : ((s[sub + 4] >> 4) | ((s[sub] >> 6) << 4));
}

static inline float gemv_q4_k_row_dot(
    device const uchar* row_ptr,
    device const float* x,
    uint nb,
    uint e0, uint sub, uint qbase, uint hi
) {
    float sumf = 0.0f;
    for (uint ib = 0; ib < nb; ib++) {
        device const uchar* blk = row_ptr + ib * Q4K_BYTES;
        float d    = float(*(device const half*)(blk));
        float dmin = float(*(device const half*)(blk + 2));
        device const uchar* scales = blk + 4;
        device const uchar* qs = blk + 16;

        float scale = d * float(q4k_get_sc(scales, sub));
        float minv  = dmin * float(q4k_get_mn(scales, sub));

        device const float* xb = x + ib * QK_K + e0;
        for (uint i = 0; i < 8u; i++) {
            uchar qb = qs[qbase + i];
            uint nib = (hi == 0u) ? uint(qb & 0x0Fu) : uint(qb >> 4);
            sumf += (scale * float(nib) - minv) * xb[i];
        }
    }
    return sumf;
}

kernel void gemv_q4_k(
    const device uchar* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    const uint m = params.m;
    const uint nb = params.k / QK_K;
    const uint row_bytes = nb * Q4K_BYTES;
    // Linearize the 2-D dispatch grid (`sz2d(min(groups,65535), ceil(groups/65535))`)
    // so `m > 65535 * NR` still maps every threadgroup to a distinct row.
    const uint tgi = tg_id.x + tg_id.y * 65535u;

    const uint e0 = tiisg * 8u;      // 0,8,...,248 across the 256-element block
    const uint j = e0 / 64u;         // 0..3
    const uint o = e0 % 64u;         // {0,8,16,24,32,40,48,56}
    const uint hi = o / 32u;         // nibble half (0 = low, 1 = high)
    const uint sub = 2u * j + hi;    // 0..7 sub-block index
    const uint qbase = 32u * j + (o % 32u);

    for (short r = 0; r < NR; r++) {
        const uint row = tgi * NR + r;
        if (row >= m) continue;
        float sumf = gemv_q4_k_row_dot(a + row * row_bytes, x, nb, e0, sub, qbase, hi);
        float total = simd_sum(sumf);
        if (tiisg == 0u) {
            y[row] = total;
        }
    }
}

kernel void gemv_q4_k_accum(
    const device uchar* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    const uint m = params.m;
    const uint nb = params.k / QK_K;
    const uint row_bytes = nb * Q4K_BYTES;
    const uint tgi = tg_id.x + tg_id.y * 65535u;

    const uint e0 = tiisg * 8u;
    const uint j = e0 / 64u;
    const uint o = e0 % 64u;
    const uint hi = o / 32u;
    const uint sub = 2u * j + hi;
    const uint qbase = 32u * j + (o % 32u);

    for (short r = 0; r < NR; r++) {
        const uint row = tgi * NR + r;
        if (row >= m) continue;
        float sumf = gemv_q4_k_row_dot(a + row * row_bytes, x, nb, e0, sub, qbase, hi);
        float total = simd_sum(sumf);
        if (tiisg == 0u) {
            y[row] += total;
        }
    }
}
