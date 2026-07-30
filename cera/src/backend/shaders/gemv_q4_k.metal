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
// Ported from llama.cpp's `kernel_mul_mv_q4_K_f32_impl`, the same source the
// `gemv_q6_k` port draws on. The old kernel here was a scalar per-byte version
// (2 rows/TG, 32 threads, one `uchar` load per quant) that sustained only a few
// GB/s on the gate/up projections — the largest single share of LFM2 decode.
// This port (like the `gemv_q6_k` sibling) brings the three wins that make the
// llama.cpp K-quant GEMVs bandwidth-efficient:
//   * uint16 loads of `qs`/`scales` (word, not byte, reads);
//   * the low/high nibble of each uint16 accumulated in-place and corrected by
//     1/256 (high byte) and 1/16 (high nibble) factors, so no per-quant shifts;
//   * the `dmin*min` bias hoisted out via a running `sumy` of the activations,
//     turning 32 subtractions into one per sub-block pair.
//
// Layout: NR=2 rows per simdgroup, NSG=2 simdgroups per TG → 4 rows/TG,
// 64 threads/TG. Dispatch: ceil(m/4) threadgroups. Same geometry as `gemv_q6_k`.

constant constexpr uint QK_K = 256;
constant constexpr uint Q4K_BYTES = 144;
constant constexpr uint NR = 2;   // rows per simdgroup
constant constexpr uint NSG = 2;  // simdgroups per TG

struct Params { uint m; uint k; };

// Compute the NR per-row simd-reduced dot products for this simdgroup's rows.
// `first_row` and `totals[NR]` (valid on every lane after `simd_sum`) are shared
// by the plain and accumulate kernels.
static inline void gemv_q4_k_compute(
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
    const uint row_bytes = nb * Q4K_BYTES;
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

    // Thread layout (llama.cpp's mul_mv_q4_K_f32_impl): a simdgroup's 32 lanes
    // split into 4 blocks of 8; within a block, iq picks the 128-element half and
    // ir the 8-element run.
    const uint ix = tiisg / 8u;   // 0..3  — which super-block (mod 4)
    const uint it = tiisg % 8u;   // 0..7
    const uint iq = it / 4u;      // 0 or 1
    const uint ir = it % 4u;      // 0..3

    float yl[16];
    float yh[16];
    float sumf[NR] = {0.0f, 0.0f};

    // This lane's scattered slice of the activation vector, advanced per block.
    const device float* y4 = x + ix * QK_K + 64u * iq + 8u * ir;

    uint16_t sc16[4];
    thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

    // Each group of 4 lanes (ix=0..3) strides consecutive super-blocks.
    for (uint ib = ix; ib < nb; ib += 4u) {
        float4 sumy = float4(0.0f);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; i++) {
            yl[i + 0u] = y4[i +   0]; sumy[0] += yl[i + 0u];
            yl[i + 8u] = y4[i +  32]; sumy[1] += yl[i + 8u];
            yh[i + 0u] = y4[i + 128]; sumy[2] += yh[i + 0u];
            yh[i + 8u] = y4[i + 160]; sumy[3] += yh[i + 8u];
        }

        #pragma clang loop unroll(full)
        for (uint row = 0; row < NR; row++) {
            device const uchar* blk = row_base[row] + ib * Q4K_BYTES;
            device const uint16_t* sc = (device const uint16_t*)(blk + 4u) + iq;
            device const uint16_t* q1 = (device const uint16_t*)(blk + 16u) + 16u * iq + 4u * ir;
            device const half* dh = (device const half*)(blk);

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t* q2 = q1 + 32u;

            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; i++) {
                acc1[0] += yl[2u * i + 0u] * (q1[i] & 0x000F);
                acc1[1] += yl[2u * i + 1u] * (q1[i] & 0x0F00);
                acc1[2] += yl[2u * i + 8u] * (q1[i] & 0x00F0);
                acc1[3] += yl[2u * i + 9u] * (q1[i] & 0xF000);
                acc2[0] += yh[2u * i + 0u] * (q2[i] & 0x000F);
                acc2[1] += yh[2u * i + 1u] * (q2[i] & 0x0F00);
                acc2[2] += yh[2u * i + 8u] * (q2[i] & 0x00F0);
                acc2[3] += yh[2u * i + 9u] * (q2[i] & 0xF000);
            }

            sumf[row] +=
                float(dh[0]) * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * sc8[0]
                              + (acc1[2] + (1.0f / 256.0f) * acc1[3]) * sc8[1] * (1.0f / 16.0f)
                              + (acc2[0] + (1.0f / 256.0f) * acc2[1]) * sc8[4]
                              + (acc2[2] + (1.0f / 256.0f) * acc2[3]) * sc8[5] * (1.0f / 16.0f))
              - float(dh[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3]
                              + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y4 += 4u * QK_K;
    }

    #pragma clang loop unroll(full)
    for (uint row = 0; row < NR; row++) {
        totals[row] = simd_sum(sumf[row]);
    }
}

kernel void gemv_q4_k(
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
    gemv_q4_k_compute(a, x, params, tiisg, sgitg, tgi, first_row, totals);
    #pragma clang loop unroll(full)
    for (uint row = 0; row < NR; row++) {
        if (tiisg == 0u && first_row + row < params.m) {
            y[first_row + row] = totals[row];
        }
    }
}

kernel void gemv_q4_k_accum(
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
    gemv_q4_k_compute(a, x, params, tiisg, sgitg, tgi, first_row, totals);
    #pragma clang loop unroll(full)
    for (uint row = 0; row < NR; row++) {
        if (tiisg == 0u && first_row + row < params.m) {
            y[first_row + row] += totals[row];
        }
    }
}
