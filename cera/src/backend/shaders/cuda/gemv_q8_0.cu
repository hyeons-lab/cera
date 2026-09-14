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
    const uint32_t row0 = warp_id * 4;
    const uint32_t row1 = row0 + 1;
    const uint32_t row2 = row0 + 2;
    const uint32_t row3 = row0 + 3;
    if (row0 >= params.m) {
        return;
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nb = params.k / 32;
    const size_t row_bytes = (size_t)nb * 34;

    const uint8_t* row0_ptr = a + (size_t)row0 * row_bytes;
    const uint8_t* row1_ptr = (row1 < params.m) ? (a + (size_t)row1 * row_bytes) : nullptr;
    const uint8_t* row2_ptr = (row2 < params.m) ? (a + (size_t)row2 * row_bytes) : nullptr;
    const uint8_t* row3_ptr = (row3 < params.m) ? (a + (size_t)row3 * row_bytes) : nullptr;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (uint32_t ib = 0; ib < nb; ib++) {
        // 128-byte coalesced read of activation vector x (reused across 4 rows)
        const float x_val = x[ib * 32 + lane];

        // Parallel scale loads: lanes 0..3 load FP16 scales for rows 0..3 concurrently
        const size_t block_offset = (size_t)ib * 34;
        float d_lane = 0.0f;
        if (lane == 0) {
            d_lane = half_to_float(*reinterpret_cast<const uint16_t*>(row0_ptr + block_offset));
        } else if (lane == 1 && row1_ptr) {
            d_lane = half_to_float(*reinterpret_cast<const uint16_t*>(row1_ptr + block_offset));
        } else if (lane == 2 && row2_ptr) {
            d_lane = half_to_float(*reinterpret_cast<const uint16_t*>(row2_ptr + block_offset));
        } else if (lane == 3 && row3_ptr) {
            d_lane = half_to_float(*reinterpret_cast<const uint16_t*>(row3_ptr + block_offset));
        }

        const float d0 = __shfl_sync(0xffffffff, d_lane, 0);
        const float d1 = __shfl_sync(0xffffffff, d_lane, 1);
        const float d2 = __shfl_sync(0xffffffff, d_lane, 2);
        const float d3 = __shfl_sync(0xffffffff, d_lane, 3);

        const int8_t q0 = *(const int8_t*)(row0_ptr + block_offset + 2 + lane);
        sum0 = fmaf((float)q0 * d0, x_val, sum0);

        if (row1_ptr) {
            const int8_t q1 = *(const int8_t*)(row1_ptr + block_offset + 2 + lane);
            sum1 = fmaf((float)q1 * d1, x_val, sum1);
        }

        if (row2_ptr) {
            const int8_t q2 = *(const int8_t*)(row2_ptr + block_offset + 2 + lane);
            sum2 = fmaf((float)q2 * d2, x_val, sum2);
        }

        if (row3_ptr) {
            const int8_t q3 = *(const int8_t*)(row3_ptr + block_offset + 2 + lane);
            sum3 = fmaf((float)q3 * d3, x_val, sum3);
        }
    }

    // Intra-warp shuffle reduction
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, mask);
        if (row1_ptr) {
            sum1 += __shfl_down_sync(0xffffffff, sum1, mask);
        }
        if (row2_ptr) {
            sum2 += __shfl_down_sync(0xffffffff, sum2, mask);
        }
        if (row3_ptr) {
            sum3 += __shfl_down_sync(0xffffffff, sum3, mask);
        }
    }

    if (lane == 0) {
        y[row0] = sum0;
        if (row1 < params.m) {
            y[row1] = sum1;
        }
        if (row2 < params.m) {
            y[row2] = sum2;
        }
        if (row3 < params.m) {
            y[row3] = sum3;
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
    const uint32_t row0 = warp_id * 4;
    const uint32_t row1 = row0 + 1;
    const uint32_t row2 = row0 + 2;
    const uint32_t row3 = row0 + 3;
    if (row0 >= params.m) {
        return;
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nb = params.k / 32;
    const size_t row_bytes = (size_t)nb * 34;

    const uint8_t* row0_ptr = a + (size_t)row0 * row_bytes;
    const uint8_t* row1_ptr = (row1 < params.m) ? (a + (size_t)row1 * row_bytes) : nullptr;
    const uint8_t* row2_ptr = (row2 < params.m) ? (a + (size_t)row2 * row_bytes) : nullptr;
    const uint8_t* row3_ptr = (row3 < params.m) ? (a + (size_t)row3 * row_bytes) : nullptr;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (uint32_t ib = 0; ib < nb; ib++) {
        const float x_val = x[ib * 32 + lane];

        // Parallel scale loads: lanes 0..3 load FP16 scales for rows 0..3 concurrently
        const size_t block_offset = (size_t)ib * 34;
        float d_lane = 0.0f;
        if (lane == 0) {
            d_lane = half_to_float(*reinterpret_cast<const uint16_t*>(row0_ptr + block_offset));
        } else if (lane == 1 && row1_ptr) {
            d_lane = half_to_float(*reinterpret_cast<const uint16_t*>(row1_ptr + block_offset));
        } else if (lane == 2 && row2_ptr) {
            d_lane = half_to_float(*reinterpret_cast<const uint16_t*>(row2_ptr + block_offset));
        } else if (lane == 3 && row3_ptr) {
            d_lane = half_to_float(*reinterpret_cast<const uint16_t*>(row3_ptr + block_offset));
        }

        const float d0 = __shfl_sync(0xffffffff, d_lane, 0);
        const float d1 = __shfl_sync(0xffffffff, d_lane, 1);
        const float d2 = __shfl_sync(0xffffffff, d_lane, 2);
        const float d3 = __shfl_sync(0xffffffff, d_lane, 3);

        const int8_t q0 = *(const int8_t*)(row0_ptr + block_offset + 2 + lane);
        sum0 = fmaf((float)q0 * d0, x_val, sum0);

        if (row1_ptr) {
            const int8_t q1 = *(const int8_t*)(row1_ptr + block_offset + 2 + lane);
            sum1 = fmaf((float)q1 * d1, x_val, sum1);
        }

        if (row2_ptr) {
            const int8_t q2 = *(const int8_t*)(row2_ptr + block_offset + 2 + lane);
            sum2 = fmaf((float)q2 * d2, x_val, sum2);
        }

        if (row3_ptr) {
            const int8_t q3 = *(const int8_t*)(row3_ptr + block_offset + 2 + lane);
            sum3 = fmaf((float)q3 * d3, x_val, sum3);
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, mask);
        if (row1_ptr) {
            sum1 += __shfl_down_sync(0xffffffff, sum1, mask);
        }
        if (row2_ptr) {
            sum2 += __shfl_down_sync(0xffffffff, sum2, mask);
        }
        if (row3_ptr) {
            sum3 += __shfl_down_sync(0xffffffff, sum3, mask);
        }
    }

    if (lane == 0) {
        y[row0] += sum0;
        if (row1 < params.m) {
            y[row1] += sum1;
        }
        if (row2 < params.m) {
            y[row2] += sum2;
        }
        if (row3 < params.m) {
            y[row3] += sum3;
        }
    }
}

} // extern "C"
