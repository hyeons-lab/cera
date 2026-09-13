// Native CUDA Q8_0 Matrix-Vector multiplication (GEMV) for single-token decode.
//
// Optimized for NVIDIA Ampere (sm_87 on Jetson Orin) memory subsystem:
// - Warp-cooperative block processing: 32 threads cooperatively load 32 contiguous int8 weights.
// - 100% coalesced 32-byte weight transactions and 128-byte activation transactions.
// - 2 rows processed per warp: vector x is loaded once and reused across adjacent rows.
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

// gemv_q8_0: y = A_q8_0 * x
__global__ void gemv_q8_0(
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
    const uint32_t nb = params.k / 32;
    const size_t row_bytes = (size_t)nb * 34;

    const uint8_t* row0_ptr = a + (size_t)row0 * row_bytes;
    const uint8_t* row1_ptr = (row1 < params.m) ? (a + (size_t)row1 * row_bytes) : nullptr;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (uint32_t ib = 0; ib < nb; ib++) {
        // 128-byte coalesced read of activation vector x
        const float x_val = x[ib * 32 + lane];

        // Row 0: load FP16 scale and int8 weight
        const uint8_t* blk0 = row0_ptr + (size_t)ib * 34;
        uint16_t d0_h;
        memcpy(&d0_h, blk0, sizeof(uint16_t));
        const float d0 = half_to_float(d0_h);
        const int8_t q0 = *(const int8_t*)(blk0 + 2 + lane);

        sum0 += ((float)q0 * x_val) * d0;

        // Row 1: load FP16 scale and int8 weight
        if (row1_ptr) {
            const uint8_t* blk1 = row1_ptr + (size_t)ib * 34;
            uint16_t d1_h;
            memcpy(&d1_h, blk1, sizeof(uint16_t));
            const float d1 = half_to_float(d1_h);
            const int8_t q1 = *(const int8_t*)(blk1 + 2 + lane);

            sum1 += ((float)q1 * x_val) * d1;
        }
    }

    // Intra-warp shuffle reduction
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, mask);
        if (row1_ptr) {
            sum1 += __shfl_down_sync(0xffffffff, sum1, mask);
        }
    }

    if (lane == 0) {
        y[row0] = sum0;
        if (row1 < params.m) {
            y[row1] = sum1;
        }
    }
}

// gemv_q8_0_accum: y += A_q8_0 * x (fused residual accumulation)
__global__ void gemv_q8_0_accum(
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
    const uint32_t nb = params.k / 32;
    const size_t row_bytes = (size_t)nb * 34;

    const uint8_t* row0_ptr = a + (size_t)row0 * row_bytes;
    const uint8_t* row1_ptr = (row1 < params.m) ? (a + (size_t)row1 * row_bytes) : nullptr;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (uint32_t ib = 0; ib < nb; ib++) {
        const float x_val = x[ib * 32 + lane];

        const uint8_t* blk0 = row0_ptr + (size_t)ib * 34;
        uint16_t d0_h;
        memcpy(&d0_h, blk0, sizeof(uint16_t));
        const float d0 = half_to_float(d0_h);
        const int8_t q0 = *(const int8_t*)(blk0 + 2 + lane);

        sum0 += ((float)q0 * x_val) * d0;

        if (row1_ptr) {
            const uint8_t* blk1 = row1_ptr + (size_t)ib * 34;
            uint16_t d1_h;
            memcpy(&d1_h, blk1, sizeof(uint16_t));
            const float d1 = half_to_float(d1_h);
            const int8_t q1 = *(const int8_t*)(blk1 + 2 + lane);

            sum1 += ((float)q1 * x_val) * d1;
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, mask);
        if (row1_ptr) {
            sum1 += __shfl_down_sync(0xffffffff, sum1, mask);
        }
    }

    if (lane == 0) {
        y[row0] += sum0;
        if (row1 < params.m) {
            y[row1] += sum1;
        }
    }
}

} // extern "C"
