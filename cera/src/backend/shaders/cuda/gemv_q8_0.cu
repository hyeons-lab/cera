// Native CUDA Q8_0 Matrix-Vector multiplication (GEMV) for single-token decode.
//
// Optimized for NVIDIA Ampere (sm_87 on Jetson Orin) memory subsystem:
// - 1 warp (32 threads) processes 1 output row.
// - Pure warp-synchronous SIMD: zero shared memory, zero bank conflicts.
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
    if (warp_id >= params.m) {
        return;
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nb = params.k / 32;
    const size_t row_bytes = (size_t)nb * 34;
    const uint8_t* row_ptr = a + (size_t)warp_id * row_bytes;

    float sum = 0.0f;

    for (uint32_t ib = lane; ib < nb; ib += 32) {
        const uint8_t* blk = row_ptr + (size_t)ib * 34;

        uint16_t d_half;
        memcpy(&d_half, blk, sizeof(uint16_t));
        const float d = half_to_float(d_half);

        const int8_t* qs = (const int8_t*)(blk + 2);
        const float* xl = x + ib * 32;

        float block_dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            block_dot += (float)qs[i] * xl[i];
        }
        sum += block_dot * d;
    }

    // Warp reduction
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, mask);
    }

    if (lane == 0) {
        y[warp_id] = sum;
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
    if (warp_id >= params.m) {
        return;
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nb = params.k / 32;
    const size_t row_bytes = (size_t)nb * 34;
    const uint8_t* row_ptr = a + (size_t)warp_id * row_bytes;

    float sum = 0.0f;

    for (uint32_t ib = lane; ib < nb; ib += 32) {
        const uint8_t* blk = row_ptr + (size_t)ib * 34;

        uint16_t d_half;
        memcpy(&d_half, blk, sizeof(uint16_t));
        const float d = half_to_float(d_half);

        const int8_t* qs = (const int8_t*)(blk + 2);
        const float* xl = x + ib * 32;

        float block_dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            block_dot += (float)qs[i] * xl[i];
        }
        sum += block_dot * d;
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, mask);
    }

    if (lane == 0) {
        y[warp_id] += sum;
    }
}

} // extern "C"
