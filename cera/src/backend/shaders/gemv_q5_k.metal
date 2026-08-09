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
// (`decode_q4km_scales`), so this is `gemv_q4_k.metal`'s kernel with the high-bit
// term added. Ported from llama.cpp's `kernel_mul_mv_q5_K_f32_impl`, the same
// source `gemv_q4_k` was ported from.
//
// What the port buys, relative to the per-thread scalar kernel this replaced:
//   * the `dmin*min` bias hoisted out of the element loop via a running `sumy`
//     of the activations, so it costs one multiply per sub-block instead of one
//     subtraction per element;
//   * the 6-bit scale/min unpack done once per super-block with three `uint16`
//     mask ops, instead of `q5k_get_sc`/`q5k_get_mn` re-deriving it per element;
//   * 16 activations held in registers (`yl`/`yh`) and reused across NR rows,
//     turning a per-row re-read of x into a per-row-group one;
//   * 4 rows per threadgroup (NR=2 per simdgroup × NSG=2), matching `gemv_q4_k`.
//
// Note what it does *not* buy, because it is easy to assume otherwise from the
// Q4_K commit: upstream's Q5_K reads `qs` a **byte** at a time (`q1[l] & 0x0F`),
// not as `uint16`. The high-bit plane is indexed by `l` alone while the nibbles
// are indexed by `32*iq + 8*ir + l`, so the two do not stride together and a
// word load of `qs` would not line up with a word load of `qh`. The win here is
// the loop shape and the bias hoist, not load width.
//
// Dequant (per sub-block j in 0..4, l in 0..32):
//   out[64j + l]      = d*sc[2j]   * ((qs[32j+l] & 0xF) + 16*bit(qh[l], 2j))   - dmin*mn[2j]
//   out[64j + l + 32] = d*sc[2j+1] * ((qs[32j+l] >> 4 ) + 16*bit(qh[l], 2j+1)) - dmin*mn[2j+1]
//
// `qh` is indexed by `l` alone: the same 32 bytes are reused by all four j
// iterations, which consume bit pairs (2j, 2j+1). That is the `u1`/`u2 <<= 2`
// walk in the CPU reference; here it is the four `hm*` masks, fixed per lane.
//
// Dispatch: ceil(m/4) threadgroups × 64 threads. Both call sites in
// `metal_lfm2.rs` must pass that geometry, including the logit head.

constant constexpr uint QK_K = 256;
constant constexpr uint Q5K_BYTES = 176;
constant constexpr uint NR = 2;   // rows per simdgroup
constant constexpr uint NSG = 2;  // simdgroups per TG

struct Params { uint m; uint k; };

// Compute the NR per-row simd-reduced dot products for this simdgroup's rows.
// `first_row` and `totals[NR]` (valid on every lane after `simd_sum`) are shared
// by the plain and accumulate kernels.
static inline void gemv_q5_k_compute(
    const device uchar* a,
    const device float* x,
    constant Params& params,
    uint tiisg, uint sgitg, uint tg_id,
    thread uint& first_row,
    thread float* totals
) {
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const uint nb = params.k / QK_K;
    const uint row_bytes = nb * Q5K_BYTES;
    first_row = (tg_id * NSG + sgitg) * NR;

    // `m == 0` never dispatches a threadgroup, but keep the kernel safe for an
    // empty/malformed dispatch: define both out-params and return.
    if (params.m == 0u) {
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < NR; r++) {
            totals[r] = 0.0f;
        }
        return;
    }

    // Per-row base byte pointers. Clamp out-of-range rows (the last TG's rows
    // when m isn't a multiple of NR*NSG) to the final valid row so the weight
    // reads stay in-bounds; their totals are discarded by the writeback guard.
    device const uchar* row_base[NR];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < NR; r++) {
        uint safe_row = min(first_row + r, params.m - 1u);
        row_base[r] = a + safe_row * row_bytes;
    }

    // Thread layout (llama.cpp's mul_mv_q5_K_f32_impl): 32 lanes split so that
    // `ix` strides super-blocks 4 apart, `iq` picks the 128-element half, and
    // `ir` the 8-element run within it.
    const uint tid = tiisg / 4u;   // 0..7
    const uint ix  = tiisg % 4u;   // 0..3, which super-block (mod 4)
    const uint iq  = tid / 4u;     // 0 or 1
    const uint ir  = tid % 4u;     // 0..3

    const uint l0 = 8u * ir;
    const uint q_offset = 32u * iq + l0;
    const uint y_offset = 64u * iq + l0;

    // The four qh bit selectors for this lane. `iq` picks the sub-block pair,
    // and the `<< 4` pair covers the high nibble's two sub-blocks.
    const uchar hm1 = uchar(1u << (2u * iq));
    const uchar hm2 = uchar(hm1 << 1);
    const uchar hm3 = uchar(hm1 << 4);
    const uchar hm4 = uchar(hm2 << 4);

    float yl[16];
    float yh[16];
    float sumf[NR] = {0.0f, 0.0f};

    uint16_t sc16[4];
    thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

    // This lane's scattered slice of the activation vector, advanced per block.
    const device float* y1 = x + ix * QK_K + y_offset;

    // Each group of 4 lanes (ix=0..3) strides consecutive super-blocks.
    for (uint ib = ix; ib < nb; ib += 4u) {
        const device float* y2 = y1 + 128;
        float4 sumy = float4(0.0f);
        #pragma clang loop unroll(full)
        for (uint l = 0; l < 8u; l++) {
            yl[l + 0u] = y1[l +  0]; sumy[0] += yl[l + 0u];
            yl[l + 8u] = y1[l + 32]; sumy[1] += yl[l + 8u];
            yh[l + 0u] = y2[l +  0]; sumy[2] += yh[l + 0u];
            yh[l + 8u] = y2[l + 32]; sumy[3] += yh[l + 8u];
        }

        #pragma clang loop unroll(full)
        for (uint row = 0; row < NR; row++) {
            device const uchar* blk = row_base[row] + ib * Q5K_BYTES;
            device const half* dh = (device const half*)(blk);
            device const uint16_t* sc = (device const uint16_t*)(blk + 4u) + iq;
            device const uchar* qh = blk + 16u + l0;
            device const uchar* q1 = blk + 48u + q_offset;
            device const uchar* q2 = q1 + 64u;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);
            #pragma clang loop unroll(full)
            for (uint l = 0; l < 8u; l++) {
                uchar h = qh[l];
                acc1[0] += yl[l + 0u] * float(q1[l] & 0x0Fu);
                acc1[1] += yl[l + 8u] * float(q1[l] & 0xF0u);
                acc1[2] += yh[l + 0u] * float(q2[l] & 0x0Fu);
                acc1[3] += yh[l + 8u] * float(q2[l] & 0xF0u);
                acc2[0] += (h & hm1) ? yl[l + 0u] : 0.0f;
                acc2[1] += (h & hm2) ? yl[l + 8u] : 0.0f;
                acc2[2] += (h & hm3) ? yh[l + 0u] : 0.0f;
                acc2[3] += (h & hm4) ? yh[l + 8u] : 0.0f;
            }

            // `acc1[1]`/`acc1[3]` carry the high nibble unshifted (masked 0xF0),
            // hence the 1/16; the `16 *` on acc2 is the 5th bit's weight.
            sumf[row] +=
                float(dh[0]) * (sc8[0] * (acc1[0]          + 16.0f * acc2[0])
                              + sc8[1] * (acc1[1] / 16.0f  + 16.0f * acc2[1])
                              + sc8[4] * (acc1[2]          + 16.0f * acc2[2])
                              + sc8[5] * (acc1[3] / 16.0f  + 16.0f * acc2[3]))
              - float(dh[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3]
                              + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y1 += 4u * QK_K;
    }

    #pragma clang loop unroll(full)
    for (uint row = 0; row < NR; row++) {
        totals[row] = simd_sum(sumf[row]);
    }
}

kernel void gemv_q5_k(
    const device uchar* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint3 tg_id [[threadgroup_position_in_grid]]
) {
    uint first_row;
    float totals[NR];
    // Linearize the 2-D dispatch grid so m > 65535 * NR * NSG still maps cleanly.
    uint tgi = tg_id.x + tg_id.y * 65535u;
    gemv_q5_k_compute(a, x, params, tiisg, sgitg, tgi, first_row, totals);
    #pragma clang loop unroll(full)
    for (uint row = 0; row < NR; row++) {
        if (tiisg == 0u && first_row + row < params.m) {
            y[first_row + row] = totals[row];
        }
    }
}

kernel void gemv_q5_k_accum(
    const device uchar* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint3 tg_id [[threadgroup_position_in_grid]]
) {
    uint first_row;
    float totals[NR];
    uint tgi = tg_id.x + tg_id.y * 65535u;
    gemv_q5_k_compute(a, x, params, tiisg, sgitg, tgi, first_row, totals);
    #pragma clang loop unroll(full)
    for (uint row = 0; row < NR; row++) {
        if (tiisg == 0u && first_row + row < params.m) {
            y[first_row + row] += totals[row];
        }
    }
}
