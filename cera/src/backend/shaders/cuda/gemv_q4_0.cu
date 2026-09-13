// Native CUDA Q4_0 Matrix-Vector multiplication (GEMV) for single-token decode.
//
// Optimized for NVIDIA Ampere (sm_87 on Jetson Orin) memory subsystem:
// - Warp-cooperative block processing: 32 threads cooperatively load 32 contiguous nibbles (16 bytes).
// - 100% coalesced 16-byte weight transactions and 128-byte activation transactions.
// - 4 rows processed per warp: vector x is loaded once and reused across 4 adjacent rows.
// - Intra-warp reduction via __shfl_down_sync.

#include <stdint.h>
#include <string.h>

__device__ __forceinline__ float half_to_float(uint16_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

extern "C" {

struct GemvParams {
    uint32_t m; // Number of output rows
    uint32_t k; // Number of input columns (must be multiple of 32)
};

// gemv_q4_0: y = A_q4_0 * x
__global__ void gemv_q4_0(
    const uint8_t* __restrict__ a,
    const float* __restrict__ x,
    float* __restrict__ y,
    GemvParams params
) {
    const uint32_t warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    const uint32_t row0 = warp_id * 4;
    const uint32_t row1 = row0 + 1;
    const uint32_t row2 = row0 + 2;
    const uint32_t row3 = row0 + 3;
    if (row0 >= params.m) {
        return;
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nb = params.k / 32;
    const size_t row_bytes = (size_t)nb * 18;

    const uint8_t* row0_ptr = a + (size_t)row0 * row_bytes;
    const uint8_t* row1_ptr = (row1 < params.m) ? (a + (size_t)row1 * row_bytes) : nullptr;
    const uint8_t* row2_ptr = (row2 < params.m) ? (a + (size_t)row2 * row_bytes) : nullptr;
    const uint8_t* row3_ptr = (row3 < params.m) ? (a + (size_t)row3 * row_bytes) : nullptr;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    const uint32_t byte_idx = lane & 15;
    const bool is_hi = (lane >= 16);

    for (uint32_t ib = 0; ib < nb; ib++) {
        // 128-byte coalesced read of activation vector x (reused across 4 rows)
        const float x_val = x[ib * 32 + lane];

        // Row 0
        float d0 = 0.0f;
        if (lane == 0) {
            uint16_t d0_h;
            memcpy(&d0_h, row0_ptr + (size_t)ib * 18, sizeof(uint16_t));
            d0 = half_to_float(d0_h);
        }
        d0 = __shfl_sync(0xffffffff, d0, 0);
        const uint8_t byte0 = *(row0_ptr + (size_t)ib * 18 + 2 + byte_idx);
        const float q0 = is_hi ? ((float)(byte0 >> 4) - 8.0f) : ((float)(byte0 & 0x0F) - 8.0f);
        sum0 += (q0 * x_val) * d0;

        // Row 1
        if (row1_ptr) {
            float d1 = 0.0f;
            if (lane == 0) {
                uint16_t d1_h;
                memcpy(&d1_h, row1_ptr + (size_t)ib * 18, sizeof(uint16_t));
                d1 = half_to_float(d1_h);
            }
            d1 = __shfl_sync(0xffffffff, d1, 0);
            const uint8_t byte1 = *(row1_ptr + (size_t)ib * 18 + 2 + byte_idx);
            const float q1 = is_hi ? ((float)(byte1 >> 4) - 8.0f) : ((float)(byte1 & 0x0F) - 8.0f);
            sum1 += (q1 * x_val) * d1;
        }

        // Row 2
        if (row2_ptr) {
            float d2 = 0.0f;
            if (lane == 0) {
                uint16_t d2_h;
                memcpy(&d2_h, row2_ptr + (size_t)ib * 18, sizeof(uint16_t));
                d2 = half_to_float(d2_h);
            }
            d2 = __shfl_sync(0xffffffff, d2, 0);
            const uint8_t byte2 = *(row2_ptr + (size_t)ib * 18 + 2 + byte_idx);
            const float q2 = is_hi ? ((float)(byte2 >> 4) - 8.0f) : ((float)(byte2 & 0x0F) - 8.0f);
            sum2 += (q2 * x_val) * d2;
        }

        // Row 3
        if (row3_ptr) {
            float d3 = 0.0f;
            if (lane == 0) {
                uint16_t d3_h;
                memcpy(&d3_h, row3_ptr + (size_t)ib * 18, sizeof(uint16_t));
                d3 = half_to_float(d3_h);
            }
            d3 = __shfl_sync(0xffffffff, d3, 0);
            const uint8_t byte3 = *(row3_ptr + (size_t)ib * 18 + 2 + byte_idx);
            const float q3 = is_hi ? ((float)(byte3 >> 4) - 8.0f) : ((float)(byte3 & 0x0F) - 8.0f);
            sum3 += (q3 * x_val) * d3;
        }
    }

    // Warp reductions
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, mask);
        sum1 += __shfl_down_sync(0xffffffff, sum1, mask);
        sum2 += __shfl_down_sync(0xffffffff, sum2, mask);
        sum3 += __shfl_down_sync(0xffffffff, sum3, mask);
    }

    if (lane == 0) {
        y[row0] = sum0;
        if (row1 < params.m) y[row1] = sum1;
        if (row2 < params.m) y[row2] = sum2;
        if (row3 < params.m) y[row3] = sum3;
    }
}

// gemv_q4_0_accum: y += A_q4_0 * x (fused residual accumulation)
__global__ void gemv_q4_0_accum(
    const uint8_t* __restrict__ a,
    const float* __restrict__ x,
    float* __restrict__ y,
    GemvParams params
) {
    const uint32_t warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    const uint32_t row0 = warp_id * 4;
    const uint32_t row1 = row0 + 1;
    const uint32_t row2 = row0 + 2;
    const uint32_t row3 = row0 + 3;
    if (row0 >= params.m) {
        return;
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nb = params.k / 32;
    const size_t row_bytes = (size_t)nb * 18;

    const uint8_t* row0_ptr = a + (size_t)row0 * row_bytes;
    const uint8_t* row1_ptr = (row1 < params.m) ? (a + (size_t)row1 * row_bytes) : nullptr;
    const uint8_t* row2_ptr = (row2 < params.m) ? (a + (size_t)row2 * row_bytes) : nullptr;
    const uint8_t* row3_ptr = (row3 < params.m) ? (a + (size_t)row3 * row_bytes) : nullptr;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    const uint32_t byte_idx = lane & 15;
    const bool is_hi = (lane >= 16);

    for (uint32_t ib = 0; ib < nb; ib++) {
        // 128-byte coalesced read of activation vector x (reused across 4 rows)
        const float x_val = x[ib * 32 + lane];

        // Row 0
        float d0 = 0.0f;
        if (lane == 0) {
            uint16_t d0_h;
            memcpy(&d0_h, row0_ptr + (size_t)ib * 18, sizeof(uint16_t));
            d0 = half_to_float(d0_h);
        }
        d0 = __shfl_sync(0xffffffff, d0, 0);
        const uint8_t byte0 = *(row0_ptr + (size_t)ib * 18 + 2 + byte_idx);
        const float q0 = is_hi ? ((float)(byte0 >> 4) - 8.0f) : ((float)(byte0 & 0x0F) - 8.0f);
        sum0 += (q0 * x_val) * d0;

        // Row 1
        if (row1_ptr) {
            float d1 = 0.0f;
            if (lane == 0) {
                uint16_t d1_h;
                memcpy(&d1_h, row1_ptr + (size_t)ib * 18, sizeof(uint16_t));
                d1 = half_to_float(d1_h);
            }
            d1 = __shfl_sync(0xffffffff, d1, 0);
            const uint8_t byte1 = *(row1_ptr + (size_t)ib * 18 + 2 + byte_idx);
            const float q1 = is_hi ? ((float)(byte1 >> 4) - 8.0f) : ((float)(byte1 & 0x0F) - 8.0f);
            sum1 += (q1 * x_val) * d1;
        }

        // Row 2
        if (row2_ptr) {
            float d2 = 0.0f;
            if (lane == 0) {
                uint16_t d2_h;
                memcpy(&d2_h, row2_ptr + (size_t)ib * 18, sizeof(uint16_t));
                d2 = half_to_float(d2_h);
            }
            d2 = __shfl_sync(0xffffffff, d2, 0);
            const uint8_t byte2 = *(row2_ptr + (size_t)ib * 18 + 2 + byte_idx);
            const float q2 = is_hi ? ((float)(byte2 >> 4) - 8.0f) : ((float)(byte2 & 0x0F) - 8.0f);
            sum2 += (q2 * x_val) * d2;
        }

        // Row 3
        if (row3_ptr) {
            float d3 = 0.0f;
            if (lane == 0) {
                uint16_t d3_h;
                memcpy(&d3_h, row3_ptr + (size_t)ib * 18, sizeof(uint16_t));
                d3 = half_to_float(d3_h);
            }
            d3 = __shfl_sync(0xffffffff, d3, 0);
            const uint8_t byte3 = *(row3_ptr + (size_t)ib * 18 + 2 + byte_idx);
            const float q3 = is_hi ? ((float)(byte3 >> 4) - 8.0f) : ((float)(byte3 & 0x0F) - 8.0f);
            sum3 += (q3 * x_val) * d3;
        }
    }

    // Warp reductions
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, mask);
        sum1 += __shfl_down_sync(0xffffffff, sum1, mask);
        sum2 += __shfl_down_sync(0xffffffff, sum2, mask);
        sum3 += __shfl_down_sync(0xffffffff, sum3, mask);
    }

    if (lane == 0) {
        y[row0] += sum0;
        if (row1 < params.m) y[row1] += sum1;
        if (row2 < params.m) y[row2] += sum2;
        if (row3 < params.m) y[row3] += sum3;
    }
}

} // extern "C"
