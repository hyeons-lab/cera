// Native CUDA Q4_K_M Matrix-Vector multiplication (GEMV) for single-token decode.
//
// Optimized for NVIDIA Ampere (sm_87 on Jetson Orin) memory subsystem:
// - 256-element super-blocks (144 bytes each): 8 sub-blocks of 32 values.
// - Warp-cooperative scale and minimum decoding via __shfl_sync.
// - 100% coalesced 32-byte quant transactions and 128-byte activation transactions.
// - 2 rows processed per warp: vector x is loaded once and reused across adjacent rows.
// - Fused multiply-add accumulation (fmaf) with hardware SFU intrinsics.
// - Intra-warp reduction via __shfl_down_sync.

#include <stdint.h>

__device__ __forceinline__ float half_to_float(uint16_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

extern "C" {

struct GemvParams {
    uint32_t m; // Number of output rows
    uint32_t k; // Number of input columns (must be multiple of 256)
};

struct Concat3Params {
    uint32_t m1;
    uint32_t m2;
    uint32_t m3;
    uint32_t k;
};

// gemv_q4k: y = A_q4k * x
__global__ void gemv_q4k(
    const uint8_t* __restrict__ a,
    const float* __restrict__ x,
    float* __restrict__ y,
    GemvParams params
) {
    const uint32_t warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    const uint32_t row0 = warp_id * 2;
    const uint32_t row1 = row0 + 1;
    if (row0 >= params.m) {
        return;
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nb = params.k / 256;
    const size_t row_bytes = (size_t)nb * 144;

    const uint8_t* row0_ptr = a + (size_t)row0 * row_bytes;
    const uint8_t* row1_ptr = (row1 < params.m) ? (a + (size_t)row1 * row_bytes) : nullptr;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (uint32_t ib = 0; ib < nb; ib++) {
        const size_t blk_off = (size_t)ib * 144;

        // Decode super-scales for row 0 and row 1
        float d0 = 0.0f;
        float dmin0 = 0.0f;
        if (lane == 0) {
            d0 = half_to_float(*reinterpret_cast<const uint16_t*>(row0_ptr + blk_off));
            dmin0 = half_to_float(*reinterpret_cast<const uint16_t*>(row0_ptr + blk_off + 2));
        }
        d0 = __shfl_sync(0xffffffff, d0, 0);
        dmin0 = __shfl_sync(0xffffffff, dmin0, 0);

        float d1 = 0.0f;
        float dmin1 = 0.0f;
        if (row1_ptr && lane == 8) {
            d1 = half_to_float(*reinterpret_cast<const uint16_t*>(row1_ptr + blk_off));
            dmin1 = half_to_float(*reinterpret_cast<const uint16_t*>(row1_ptr + blk_off + 2));
        }
        d1 = __shfl_sync(0xffffffff, d1, 8);
        dmin1 = __shfl_sync(0xffffffff, dmin1, 8);

        // Lanes 0..7 decode row 0 sub-block scales/mins; lanes 8..15 decode row 1
        uint32_t sc0 = 0, mn0 = 0;
        if (lane < 4) {
            const uint8_t* sc_ptr0 = row0_ptr + blk_off + 4;
            sc0 = sc_ptr0[lane] & 63;
            mn0 = sc_ptr0[lane + 4] & 63;
        } else if (lane < 8) {
            const uint8_t* sc_ptr0 = row0_ptr + blk_off + 4;
            sc0 = (sc_ptr0[lane + 4] & 0x0F) | ((sc_ptr0[lane - 4] >> 6) << 4);
            mn0 = (sc_ptr0[lane + 4] >> 4) | ((sc_ptr0[lane] >> 6) << 4);
        }

        uint32_t sc1 = 0, mn1 = 0;
        if (row1_ptr) {
            if (lane >= 8 && lane < 12) {
                const uint8_t* sc_ptr1 = row1_ptr + blk_off + 4;
                uint32_t l = lane - 8;
                sc1 = sc_ptr1[l] & 63;
                mn1 = sc_ptr1[l + 4] & 63;
            } else if (lane >= 12 && lane < 16) {
                const uint8_t* sc_ptr1 = row1_ptr + blk_off + 4;
                uint32_t l = lane - 8;
                sc1 = (sc_ptr1[l + 4] & 0x0F) | ((sc_ptr1[l - 4] >> 6) << 4);
                mn1 = (sc_ptr1[l + 4] >> 4) | ((sc_ptr1[l] >> 6) << 4);
            }
        }

        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            const float sc0_lo = (float)__shfl_sync(0xffffffff, sc0, 2 * j);
            const float mn0_lo = (float)__shfl_sync(0xffffffff, mn0, 2 * j);
            const float sc0_hi = (float)__shfl_sync(0xffffffff, sc0, 2 * j + 1);
            const float mn0_hi = (float)__shfl_sync(0xffffffff, mn0, 2 * j + 1);

            const float d_sc0_lo = d0 * sc0_lo;
            const float d_mn0_lo = dmin0 * mn0_lo;
            const float d_sc0_hi = d0 * sc0_hi;
            const float d_mn0_hi = dmin0 * mn0_hi;

            const float sc1_lo = (float)__shfl_sync(0xffffffff, sc1, 8 + 2 * j);
            const float mn1_lo = (float)__shfl_sync(0xffffffff, mn1, 8 + 2 * j);
            const float sc1_hi = (float)__shfl_sync(0xffffffff, sc1, 8 + 2 * j + 1);
            const float mn1_hi = (float)__shfl_sync(0xffffffff, mn1, 8 + 2 * j + 1);

            const float d_sc1_lo = d1 * sc1_lo;
            const float d_mn1_lo = dmin1 * mn1_lo;
            const float d_sc1_hi = d1 * sc1_hi;
            const float d_mn1_hi = dmin1 * mn1_hi;

            // Shared activation reads (reused for row 0 and row 1)
            const float x_lo = x[ib * 256 + j * 64 + lane];
            const float x_hi = x[ib * 256 + j * 64 + 32 + lane];

            // Row 0 quants
            const uint8_t byte0 = *(row0_ptr + blk_off + 16 + j * 32 + lane);
            const float q0_lo = (float)(byte0 & 0x0F);
            const float q0_hi = (float)(byte0 >> 4);
            sum0 = fmaf(x_lo, fmaf(d_sc0_lo, q0_lo, -d_mn0_lo), sum0);
            sum0 = fmaf(x_hi, fmaf(d_sc0_hi, q0_hi, -d_mn0_hi), sum0);

            // Row 1 quants
            if (row1_ptr) {
                const uint8_t byte1 = *(row1_ptr + blk_off + 16 + j * 32 + lane);
                const float q1_lo = (float)(byte1 & 0x0F);
                const float q1_hi = (float)(byte1 >> 4);
                sum1 = fmaf(x_lo, fmaf(d_sc1_lo, q1_lo, -d_mn1_lo), sum1);
                sum1 = fmaf(x_hi, fmaf(d_sc1_hi, q1_hi, -d_mn1_hi), sum1);
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, mask);
        if (row1_ptr) sum1 += __shfl_down_sync(0xffffffff, sum1, mask);
    }

    if (lane == 0) {
        y[row0] = sum0;
        if (row1 < params.m) y[row1] = sum1;
    }
}

// gemv_q4k_accum: y += A_q4k * x
__global__ void gemv_q4k_accum(
    const uint8_t* __restrict__ a,
    const float* __restrict__ x,
    float* __restrict__ y,
    GemvParams params
) {
    const uint32_t warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    const uint32_t row0 = warp_id * 2;
    const uint32_t row1 = row0 + 1;
    if (row0 >= params.m) {
        return;
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nb = params.k / 256;
    const size_t row_bytes = (size_t)nb * 144;

    const uint8_t* row0_ptr = a + (size_t)row0 * row_bytes;
    const uint8_t* row1_ptr = (row1 < params.m) ? (a + (size_t)row1 * row_bytes) : nullptr;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (uint32_t ib = 0; ib < nb; ib++) {
        const size_t blk_off = (size_t)ib * 144;

        float d0 = 0.0f;
        float dmin0 = 0.0f;
        if (lane == 0) {
            d0 = half_to_float(*reinterpret_cast<const uint16_t*>(row0_ptr + blk_off));
            dmin0 = half_to_float(*reinterpret_cast<const uint16_t*>(row0_ptr + blk_off + 2));
        }
        d0 = __shfl_sync(0xffffffff, d0, 0);
        dmin0 = __shfl_sync(0xffffffff, dmin0, 0);

        float d1 = 0.0f;
        float dmin1 = 0.0f;
        if (row1_ptr && lane == 8) {
            d1 = half_to_float(*reinterpret_cast<const uint16_t*>(row1_ptr + blk_off));
            dmin1 = half_to_float(*reinterpret_cast<const uint16_t*>(row1_ptr + blk_off + 2));
        }
        d1 = __shfl_sync(0xffffffff, d1, 8);
        dmin1 = __shfl_sync(0xffffffff, dmin1, 8);

        uint32_t sc0 = 0, mn0 = 0;
        if (lane < 4) {
            const uint8_t* sc_ptr0 = row0_ptr + blk_off + 4;
            sc0 = sc_ptr0[lane] & 63;
            mn0 = sc_ptr0[lane + 4] & 63;
        } else if (lane < 8) {
            const uint8_t* sc_ptr0 = row0_ptr + blk_off + 4;
            sc0 = (sc_ptr0[lane + 4] & 0x0F) | ((sc_ptr0[lane - 4] >> 6) << 4);
            mn0 = (sc_ptr0[lane + 4] >> 4) | ((sc_ptr0[lane] >> 6) << 4);
        }

        uint32_t sc1 = 0, mn1 = 0;
        if (row1_ptr) {
            if (lane >= 8 && lane < 12) {
                const uint8_t* sc_ptr1 = row1_ptr + blk_off + 4;
                uint32_t l = lane - 8;
                sc1 = sc_ptr1[l] & 63;
                mn1 = sc_ptr1[l + 4] & 63;
            } else if (lane >= 12 && lane < 16) {
                const uint8_t* sc_ptr1 = row1_ptr + blk_off + 4;
                uint32_t l = lane - 8;
                sc1 = (sc_ptr1[l + 4] & 0x0F) | ((sc_ptr1[l - 4] >> 6) << 4);
                mn1 = (sc_ptr1[l + 4] >> 4) | ((sc_ptr1[l] >> 6) << 4);
            }
        }

        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            const float sc0_lo = (float)__shfl_sync(0xffffffff, sc0, 2 * j);
            const float mn0_lo = (float)__shfl_sync(0xffffffff, mn0, 2 * j);
            const float sc0_hi = (float)__shfl_sync(0xffffffff, sc0, 2 * j + 1);
            const float mn0_hi = (float)__shfl_sync(0xffffffff, mn0, 2 * j + 1);

            const float d_sc0_lo = d0 * sc0_lo;
            const float d_mn0_lo = dmin0 * mn0_lo;
            const float d_sc0_hi = d0 * sc0_hi;
            const float d_mn0_hi = dmin0 * mn0_hi;

            const float sc1_lo = (float)__shfl_sync(0xffffffff, sc1, 8 + 2 * j);
            const float mn1_lo = (float)__shfl_sync(0xffffffff, mn1, 8 + 2 * j);
            const float sc1_hi = (float)__shfl_sync(0xffffffff, sc1, 8 + 2 * j + 1);
            const float mn1_hi = (float)__shfl_sync(0xffffffff, mn1, 8 + 2 * j + 1);

            const float d_sc1_lo = d1 * sc1_lo;
            const float d_mn1_lo = dmin1 * mn1_lo;
            const float d_sc1_hi = d1 * sc1_hi;
            const float d_mn1_hi = dmin1 * mn1_hi;

            const float x_lo = x[ib * 256 + j * 64 + lane];
            const float x_hi = x[ib * 256 + j * 64 + 32 + lane];

            const uint8_t byte0 = *(row0_ptr + blk_off + 16 + j * 32 + lane);
            const float q0_lo = (float)(byte0 & 0x0F);
            const float q0_hi = (float)(byte0 >> 4);
            sum0 = fmaf(x_lo, fmaf(d_sc0_lo, q0_lo, -d_mn0_lo), sum0);
            sum0 = fmaf(x_hi, fmaf(d_sc0_hi, q0_hi, -d_mn0_hi), sum0);

            if (row1_ptr) {
                const uint8_t byte1 = *(row1_ptr + blk_off + 16 + j * 32 + lane);
                const float q1_lo = (float)(byte1 & 0x0F);
                const float q1_hi = (float)(byte1 >> 4);
                sum1 = fmaf(x_lo, fmaf(d_sc1_lo, q1_lo, -d_mn1_lo), sum1);
                sum1 = fmaf(x_hi, fmaf(d_sc1_hi, q1_hi, -d_mn1_hi), sum1);
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, mask);
        if (row1_ptr) sum1 += __shfl_down_sync(0xffffffff, sum1, mask);
    }

    if (lane == 0) {
        y[row0] += sum0;
        if (row1 < params.m) y[row1] += sum1;
    }
}

// gemv_q4k_concat3: unified 3-matrix Q4_K GEMV for Q, K, V projections in a single grid launch
__global__ void gemv_q4k_concat3(
    const uint8_t* __restrict__ a1,
    const uint8_t* __restrict__ a2,
    const uint8_t* __restrict__ a3,
    const float* __restrict__ x,
    float* __restrict__ y1,
    float* __restrict__ y2,
    float* __restrict__ y3,
    Concat3Params params
) {
    const uint32_t warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    const uint32_t warps_m1 = (params.m1 + 1) / 2;
    const uint32_t warps_m2 = (params.m2 + 1) / 2;
    const uint32_t warps_m3 = (params.m3 + 1) / 2;

    const uint8_t* a_ptr = nullptr;
    float* y_ptr = nullptr;
    uint32_t local_row0 = 0;
    uint32_t m_limit = 0;

    if (warp_id < warps_m1) {
        a_ptr = a1;
        y_ptr = y1;
        local_row0 = warp_id * 2;
        m_limit = params.m1;
    } else if (warp_id < warps_m1 + warps_m2) {
        a_ptr = a2;
        y_ptr = y2;
        local_row0 = (warp_id - warps_m1) * 2;
        m_limit = params.m2;
    } else if (warp_id < warps_m1 + warps_m2 + warps_m3) {
        a_ptr = a3;
        y_ptr = y3;
        local_row0 = (warp_id - warps_m1 - warps_m2) * 2;
        m_limit = params.m3;
    } else {
        return;
    }

    const uint32_t row0 = local_row0;
    const uint32_t row1 = row0 + 1;
    if (row0 >= m_limit) {
        return;
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nb = params.k / 256;
    const size_t row_bytes = (size_t)nb * 144;

    const uint8_t* row0_ptr = a_ptr + (size_t)row0 * row_bytes;
    const uint8_t* row1_ptr = (row1 < m_limit) ? (a_ptr + (size_t)row1 * row_bytes) : nullptr;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (uint32_t ib = 0; ib < nb; ib++) {
        const size_t blk_off = (size_t)ib * 144;

        float d0 = 0.0f;
        float dmin0 = 0.0f;
        if (lane == 0) {
            d0 = half_to_float(*reinterpret_cast<const uint16_t*>(row0_ptr + blk_off));
            dmin0 = half_to_float(*reinterpret_cast<const uint16_t*>(row0_ptr + blk_off + 2));
        }
        d0 = __shfl_sync(0xffffffff, d0, 0);
        dmin0 = __shfl_sync(0xffffffff, dmin0, 0);

        float d1 = 0.0f;
        float dmin1 = 0.0f;
        if (row1_ptr && lane == 8) {
            d1 = half_to_float(*reinterpret_cast<const uint16_t*>(row1_ptr + blk_off));
            dmin1 = half_to_float(*reinterpret_cast<const uint16_t*>(row1_ptr + blk_off + 2));
        }
        d1 = __shfl_sync(0xffffffff, d1, 8);
        dmin1 = __shfl_sync(0xffffffff, dmin1, 8);

        uint32_t sc0 = 0, mn0 = 0;
        if (lane < 4) {
            const uint8_t* sc_ptr0 = row0_ptr + blk_off + 4;
            sc0 = sc_ptr0[lane] & 63;
            mn0 = sc_ptr0[lane + 4] & 63;
        } else if (lane < 8) {
            const uint8_t* sc_ptr0 = row0_ptr + blk_off + 4;
            sc0 = (sc_ptr0[lane + 4] & 0x0F) | ((sc_ptr0[lane - 4] >> 6) << 4);
            mn0 = (sc_ptr0[lane + 4] >> 4) | ((sc_ptr0[lane] >> 6) << 4);
        }

        uint32_t sc1 = 0, mn1 = 0;
        if (row1_ptr) {
            if (lane >= 8 && lane < 12) {
                const uint8_t* sc_ptr1 = row1_ptr + blk_off + 4;
                uint32_t l = lane - 8;
                sc1 = sc_ptr1[l] & 63;
                mn1 = sc_ptr1[l + 4] & 63;
            } else if (lane >= 12 && lane < 16) {
                const uint8_t* sc_ptr1 = row1_ptr + blk_off + 4;
                uint32_t l = lane - 8;
                sc1 = (sc_ptr1[l + 4] & 0x0F) | ((sc_ptr1[l - 4] >> 6) << 4);
                mn1 = (sc_ptr1[l + 4] >> 4) | ((sc_ptr1[l] >> 6) << 4);
            }
        }

        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            const float sc0_lo = (float)__shfl_sync(0xffffffff, sc0, 2 * j);
            const float mn0_lo = (float)__shfl_sync(0xffffffff, mn0, 2 * j);
            const float sc0_hi = (float)__shfl_sync(0xffffffff, sc0, 2 * j + 1);
            const float mn0_hi = (float)__shfl_sync(0xffffffff, mn0, 2 * j + 1);

            const float d_sc0_lo = d0 * sc0_lo;
            const float d_mn0_lo = dmin0 * mn0_lo;
            const float d_sc0_hi = d0 * sc0_hi;
            const float d_mn0_hi = dmin0 * mn0_hi;

            const float sc1_lo = (float)__shfl_sync(0xffffffff, sc1, 8 + 2 * j);
            const float mn1_lo = (float)__shfl_sync(0xffffffff, mn1, 8 + 2 * j);
            const float sc1_hi = (float)__shfl_sync(0xffffffff, sc1, 8 + 2 * j + 1);
            const float mn1_hi = (float)__shfl_sync(0xffffffff, mn1, 8 + 2 * j + 1);

            const float d_sc1_lo = d1 * sc1_lo;
            const float d_mn1_lo = dmin1 * mn1_lo;
            const float d_sc1_hi = d1 * sc1_hi;
            const float d_mn1_hi = dmin1 * mn1_hi;

            const float x_lo = x[ib * 256 + j * 64 + lane];
            const float x_hi = x[ib * 256 + j * 64 + 32 + lane];

            const uint8_t byte0 = *(row0_ptr + blk_off + 16 + j * 32 + lane);
            const float q0_lo = (float)(byte0 & 0x0F);
            const float q0_hi = (float)(byte0 >> 4);
            sum0 = fmaf(x_lo, fmaf(d_sc0_lo, q0_lo, -d_mn0_lo), sum0);
            sum0 = fmaf(x_hi, fmaf(d_sc0_hi, q0_hi, -d_mn0_hi), sum0);

            if (row1_ptr) {
                const uint8_t byte1 = *(row1_ptr + blk_off + 16 + j * 32 + lane);
                const float q1_lo = (float)(byte1 & 0x0F);
                const float q1_hi = (float)(byte1 >> 4);
                sum1 = fmaf(x_lo, fmaf(d_sc1_lo, q1_lo, -d_mn1_lo), sum1);
                sum1 = fmaf(x_hi, fmaf(d_sc1_hi, q1_hi, -d_mn1_hi), sum1);
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, mask);
        if (row1_ptr) sum1 += __shfl_down_sync(0xffffffff, sum1, mask);
    }

    if (lane == 0) {
        y_ptr[row0] = sum0;
        if (row1 < m_limit) y_ptr[row1] = sum1;
    }
}

// gemv_q4k_swiglu: out = silu(gate * x) * (up * x) in a single fused pass
__global__ void gemv_q4k_swiglu(
    const uint8_t* __restrict__ gate_a,
    const uint8_t* __restrict__ up_a,
    const float* __restrict__ x,
    float* __restrict__ out,
    GemvParams params
) {
    const uint32_t warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    const uint32_t row = warp_id;
    if (row >= params.m) {
        return;
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nb = params.k / 256;
    const size_t row_bytes = (size_t)nb * 144;

    const uint8_t* gate_ptr = gate_a + (size_t)row * row_bytes;
    const uint8_t* up_ptr = up_a + (size_t)row * row_bytes;

    float sum_g = 0.0f;
    float sum_u = 0.0f;

    for (uint32_t ib = 0; ib < nb; ib++) {
        const size_t blk_off = (size_t)ib * 144;

        // Gate scales
        float d_g = 0.0f, dmin_g = 0.0f;
        if (lane == 0) {
            d_g = half_to_float(*reinterpret_cast<const uint16_t*>(gate_ptr + blk_off));
            dmin_g = half_to_float(*reinterpret_cast<const uint16_t*>(gate_ptr + blk_off + 2));
        }
        d_g = __shfl_sync(0xffffffff, d_g, 0);
        dmin_g = __shfl_sync(0xffffffff, dmin_g, 0);

        // Up scales
        float d_u = 0.0f, dmin_u = 0.0f;
        if (lane == 8) {
            d_u = half_to_float(*reinterpret_cast<const uint16_t*>(up_ptr + blk_off));
            dmin_u = half_to_float(*reinterpret_cast<const uint16_t*>(up_ptr + blk_off + 2));
        }
        d_u = __shfl_sync(0xffffffff, d_u, 8);
        dmin_u = __shfl_sync(0xffffffff, dmin_u, 8);

        // Decode scales: lanes 0..7 gate, lanes 8..15 up
        uint32_t sc_g = 0, mn_g = 0;
        if (lane < 4) {
            const uint8_t* sc_ptr = gate_ptr + blk_off + 4;
            sc_g = sc_ptr[lane] & 63;
            mn_g = sc_ptr[lane + 4] & 63;
        } else if (lane < 8) {
            const uint8_t* sc_ptr = gate_ptr + blk_off + 4;
            sc_g = (sc_ptr[lane + 4] & 0x0F) | ((sc_ptr[lane - 4] >> 6) << 4);
            mn_g = (sc_ptr[lane + 4] >> 4) | ((sc_ptr[lane] >> 6) << 4);
        }

        uint32_t sc_u = 0, mn_u = 0;
        if (lane >= 8 && lane < 12) {
            const uint8_t* sc_ptr = up_ptr + blk_off + 4;
            uint32_t l = lane - 8;
            sc_u = sc_ptr[l] & 63;
            mn_u = sc_ptr[l + 4] & 63;
        } else if (lane >= 12 && lane < 16) {
            const uint8_t* sc_ptr = up_ptr + blk_off + 4;
            uint32_t l = lane - 8;
            sc_u = (sc_ptr[l + 4] & 0x0F) | ((sc_ptr[l - 4] >> 6) << 4);
            mn_u = (sc_ptr[l + 4] >> 4) | ((sc_ptr[l] >> 6) << 4);
        }

        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            const float sc_g_lo = (float)__shfl_sync(0xffffffff, sc_g, 2 * j);
            const float mn_g_lo = (float)__shfl_sync(0xffffffff, mn_g, 2 * j);
            const float sc_g_hi = (float)__shfl_sync(0xffffffff, sc_g, 2 * j + 1);
            const float mn_g_hi = (float)__shfl_sync(0xffffffff, mn_g, 2 * j + 1);

            const float d_sc_g_lo = d_g * sc_g_lo;
            const float d_mn_g_lo = dmin_g * mn_g_lo;
            const float d_sc_g_hi = d_g * sc_g_hi;
            const float d_mn_g_hi = dmin_g * mn_g_hi;

            const float sc_u_lo = (float)__shfl_sync(0xffffffff, sc_u, 8 + 2 * j);
            const float mn_u_lo = (float)__shfl_sync(0xffffffff, mn_u, 8 + 2 * j);
            const float sc_u_hi = (float)__shfl_sync(0xffffffff, sc_u, 8 + 2 * j + 1);
            const float mn_u_hi = (float)__shfl_sync(0xffffffff, mn_u, 8 + 2 * j + 1);

            const float d_sc_u_lo = d_u * sc_u_lo;
            const float d_mn_u_lo = dmin_u * mn_u_lo;
            const float d_sc_u_hi = d_u * sc_u_hi;
            const float d_mn_u_hi = dmin_u * mn_u_hi;

            const float x_lo = x[ib * 256 + j * 64 + lane];
            const float x_hi = x[ib * 256 + j * 64 + 32 + lane];

            const uint8_t byte_g = *(gate_ptr + blk_off + 16 + j * 32 + lane);
            const float q_g_lo = (float)(byte_g & 0x0F);
            const float q_g_hi = (float)(byte_g >> 4);
            sum_g = fmaf(x_lo, fmaf(d_sc_g_lo, q_g_lo, -d_mn_g_lo), sum_g);
            sum_g = fmaf(x_hi, fmaf(d_sc_g_hi, q_g_hi, -d_mn_g_hi), sum_g);

            const uint8_t byte_u = *(up_ptr + blk_off + 16 + j * 32 + lane);
            const float q_u_lo = (float)(byte_u & 0x0F);
            const float q_u_hi = (float)(byte_u >> 4);
            sum_u = fmaf(x_lo, fmaf(d_sc_u_lo, q_u_lo, -d_mn_u_lo), sum_u);
            sum_u = fmaf(x_hi, fmaf(d_sc_u_hi, q_u_hi, -d_mn_u_hi), sum_u);
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_g += __shfl_down_sync(0xffffffff, sum_g, mask);
        sum_u += __shfl_down_sync(0xffffffff, sum_u, mask);
    }

    if (lane == 0) {
        float silu = sum_g / (1.0f + __expf(-sum_g));
        out[row] = silu * sum_u;
    }
}

} // extern "C"
