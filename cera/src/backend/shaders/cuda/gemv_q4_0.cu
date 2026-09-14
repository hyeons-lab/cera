// Native CUDA Q4_0 Matrix-Vector multiplication (GEMV) for single-token decode.
//
// Optimized for NVIDIA Ampere (sm_87 on Jetson Orin) memory subsystem:
// - Warp-cooperative block processing: 32 threads cooperatively process 32 contiguous nibbles (16 bytes).
// - 100% coalesced weight transactions and 128-byte activation transactions.
// - 4 rows processed per warp: vector x is loaded once and reused across 4 adjacent rows.
// - Single-cycle FP16 to FP32 hardware conversion via inline PTX cvt.f32.f16.
// - Fused multiply-add accumulation (fmaf) with scale pre-multiplication.
// - Intra-warp reduction via __shfl_down_sync.

#include <stdint.h>

__device__ __forceinline__ float half_to_float(uint16_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ float silu_scalar(float g) {
    if (g < -80.0f) g = -80.0f;
    else if (g > 80.0f) g = 80.0f;
    return g / (1.0f + __expf(-g));
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
    const uint32_t shift = (lane >> 2) & 4;

    #pragma unroll 2
    for (uint32_t ib = 0; ib < nb; ib++) {
        // 128-byte coalesced read of activation vector x (reused across 4 rows)
        const float x_val = x[ib * 32 + lane];
        const size_t block_offset = (size_t)ib * 18;

        // Parallel scale loads: lanes 0..3 load FP16 scales for rows 0..3 concurrently
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

        const float x0 = x_val * d0;
        const float x1 = x_val * d1;
        const float x2 = x_val * d2;
        const float x3 = x_val * d3;

        // Row 0
        const uint8_t byte0 = *(row0_ptr + block_offset + 2 + byte_idx);
        const float q0 = (float)((byte0 >> shift) & 0x0F) - 8.0f;
        sum0 = fmaf(q0, x0, sum0);

        // Row 1
        if (row1_ptr) {
            const uint8_t byte1 = *(row1_ptr + block_offset + 2 + byte_idx);
            const float q1 = (float)((byte1 >> shift) & 0x0F) - 8.0f;
            sum1 = fmaf(q1, x1, sum1);
        }

        // Row 2
        if (row2_ptr) {
            const uint8_t byte2 = *(row2_ptr + block_offset + 2 + byte_idx);
            const float q2 = (float)((byte2 >> shift) & 0x0F) - 8.0f;
            sum2 = fmaf(q2, x2, sum2);
        }

        // Row 3
        if (row3_ptr) {
            const uint8_t byte3 = *(row3_ptr + block_offset + 2 + byte_idx);
            const float q3 = (float)((byte3 >> shift) & 0x0F) - 8.0f;
            sum3 = fmaf(q3, x3, sum3);
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
    const uint32_t shift = (lane >> 2) & 4;

    #pragma unroll 2
    for (uint32_t ib = 0; ib < nb; ib++) {
        // 128-byte coalesced read of activation vector x (reused across 4 rows)
        const float x_val = x[ib * 32 + lane];
        const size_t block_offset = (size_t)ib * 18;

        // Parallel scale loads: lanes 0..3 load FP16 scales for rows 0..3 concurrently
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

        const float x0 = x_val * d0;
        const float x1 = x_val * d1;
        const float x2 = x_val * d2;
        const float x3 = x_val * d3;

        // Row 0
        const uint8_t byte0 = *(row0_ptr + block_offset + 2 + byte_idx);
        const float q0 = (float)((byte0 >> shift) & 0x0F) - 8.0f;
        sum0 = fmaf(q0, x0, sum0);

        // Row 1
        if (row1_ptr) {
            const uint8_t byte1 = *(row1_ptr + block_offset + 2 + byte_idx);
            const float q1 = (float)((byte1 >> shift) & 0x0F) - 8.0f;
            sum1 = fmaf(q1, x1, sum1);
        }

        // Row 2
        if (row2_ptr) {
            const uint8_t byte2 = *(row2_ptr + block_offset + 2 + byte_idx);
            const float q2 = (float)((byte2 >> shift) & 0x0F) - 8.0f;
            sum2 = fmaf(q2, x2, sum2);
        }

        // Row 3
        if (row3_ptr) {
            const uint8_t byte3 = *(row3_ptr + block_offset + 2 + byte_idx);
            const float q3 = (float)((byte3 >> shift) & 0x0F) - 8.0f;
            sum3 = fmaf(q3, x3, sum3);
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

struct Concat3Params {
    uint32_t m1;
    uint32_t m2;
    uint32_t m3;
    uint32_t k;
};

// gemv_q4_0_concat3: unified 3-matrix GEMV for Q, K, V projections in a single grid launch
__global__ void gemv_q4_0_concat3(
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
    const uint32_t warps_m1 = (params.m1 + 3) / 4;
    const uint32_t warps_m2 = (params.m2 + 3) / 4;
    const uint32_t warps_m3 = (params.m3 + 3) / 4;

    const uint8_t* a_ptr = nullptr;
    float* y_ptr = nullptr;
    uint32_t local_row0 = 0;
    uint32_t m_limit = 0;

    if (warp_id < warps_m1) {
        a_ptr = a1;
        y_ptr = y1;
        local_row0 = warp_id * 4;
        m_limit = params.m1;
    } else if (warp_id < warps_m1 + warps_m2) {
        a_ptr = a2;
        y_ptr = y2;
        local_row0 = (warp_id - warps_m1) * 4;
        m_limit = params.m2;
    } else if (warp_id < warps_m1 + warps_m2 + warps_m3) {
        a_ptr = a3;
        y_ptr = y3;
        local_row0 = (warp_id - warps_m1 - warps_m2) * 4;
        m_limit = params.m3;
    } else {
        return;
    }

    const uint32_t row0 = local_row0;
    const uint32_t row1 = row0 + 1;
    const uint32_t row2 = row0 + 2;
    const uint32_t row3 = row0 + 3;
    if (row0 >= m_limit) {
        return;
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t nb = params.k / 32;
    const size_t row_bytes = (size_t)nb * 18;

    const uint8_t* row0_ptr = a_ptr + (size_t)row0 * row_bytes;
    const uint8_t* row1_ptr = (row1 < m_limit) ? (a_ptr + (size_t)row1 * row_bytes) : nullptr;
    const uint8_t* row2_ptr = (row2 < m_limit) ? (a_ptr + (size_t)row2 * row_bytes) : nullptr;
    const uint8_t* row3_ptr = (row3 < m_limit) ? (a_ptr + (size_t)row3 * row_bytes) : nullptr;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    const uint32_t byte_idx = lane & 15;
    const uint32_t shift = (lane >> 2) & 4;

    #pragma unroll 2
    for (uint32_t ib = 0; ib < nb; ib++) {
        const float x_val = x[ib * 32 + lane];
        const size_t block_offset = (size_t)ib * 18;

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

        const float x0 = x_val * d0;
        const float x1 = x_val * d1;
        const float x2 = x_val * d2;
        const float x3 = x_val * d3;

        const uint8_t byte0 = *(row0_ptr + block_offset + 2 + byte_idx);
        const float q0 = (float)((byte0 >> shift) & 0x0F) - 8.0f;
        sum0 = fmaf(q0, x0, sum0);

        if (row1_ptr) {
            const uint8_t byte1 = *(row1_ptr + block_offset + 2 + byte_idx);
            const float q1 = (float)((byte1 >> shift) & 0x0F) - 8.0f;
            sum1 = fmaf(q1, x1, sum1);
        }
        if (row2_ptr) {
            const uint8_t byte2 = *(row2_ptr + block_offset + 2 + byte_idx);
            const float q2 = (float)((byte2 >> shift) & 0x0F) - 8.0f;
            sum2 = fmaf(q2, x2, sum2);
        }
        if (row3_ptr) {
            const uint8_t byte3 = *(row3_ptr + block_offset + 2 + byte_idx);
            const float q3 = (float)((byte3 >> shift) & 0x0F) - 8.0f;
            sum3 = fmaf(q3, x3, sum3);
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, mask);
        sum1 += __shfl_down_sync(0xffffffff, sum1, mask);
        sum2 += __shfl_down_sync(0xffffffff, sum2, mask);
        sum3 += __shfl_down_sync(0xffffffff, sum3, mask);
    }

    if (lane == 0) {
        y_ptr[row0] = sum0;
        if (row1 < m_limit) y_ptr[row1] = sum1;
        if (row2 < m_limit) y_ptr[row2] = sum2;
        if (row3 < m_limit) y_ptr[row3] = sum3;
    }
}

// gemv_q4_0_swiglu: out = silu(gate * x) * (up * x) evaluated in registers
__global__ void gemv_q4_0_swiglu(
    const uint8_t* __restrict__ gate_a,
    const uint8_t* __restrict__ up_a,
    const float* __restrict__ x,
    float* __restrict__ out,
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
    const size_t row_bytes = (size_t)nb * 18;

    const uint8_t* gate0_ptr = gate_a + (size_t)row0 * row_bytes;
    const uint8_t* gate1_ptr = (row1 < params.m) ? (gate_a + (size_t)row1 * row_bytes) : nullptr;
    const uint8_t* up0_ptr = up_a + (size_t)row0 * row_bytes;
    const uint8_t* up1_ptr = (row1 < params.m) ? (up_a + (size_t)row1 * row_bytes) : nullptr;

    float sum_g0 = 0.0f;
    float sum_g1 = 0.0f;
    float sum_u0 = 0.0f;
    float sum_u1 = 0.0f;

    const uint32_t byte_idx = lane & 15;
    const uint32_t shift = (lane >> 2) & 4;

    #pragma unroll 2
    for (uint32_t ib = 0; ib < nb; ib++) {
        const float x_val = x[ib * 32 + lane];
        const size_t block_offset = (size_t)ib * 18;

        float d_lane = 0.0f;
        if (lane == 0) {
            d_lane = half_to_float(*reinterpret_cast<const uint16_t*>(gate0_ptr + block_offset));
        } else if (lane == 1 && gate1_ptr) {
            d_lane = half_to_float(*reinterpret_cast<const uint16_t*>(gate1_ptr + block_offset));
        } else if (lane == 2) {
            d_lane = half_to_float(*reinterpret_cast<const uint16_t*>(up0_ptr + block_offset));
        } else if (lane == 3 && up1_ptr) {
            d_lane = half_to_float(*reinterpret_cast<const uint16_t*>(up1_ptr + block_offset));
        }

        const float d_g0 = __shfl_sync(0xffffffff, d_lane, 0);
        const float d_g1 = __shfl_sync(0xffffffff, d_lane, 1);
        const float d_u0 = __shfl_sync(0xffffffff, d_lane, 2);
        const float d_u1 = __shfl_sync(0xffffffff, d_lane, 3);

        const float x_g0 = x_val * d_g0;
        const float x_g1 = x_val * d_g1;
        const float x_u0 = x_val * d_u0;
        const float x_u1 = x_val * d_u1;

        const uint8_t b_g0 = *(gate0_ptr + block_offset + 2 + byte_idx);
        const float q_g0 = (float)((b_g0 >> shift) & 0x0F) - 8.0f;
        sum_g0 = fmaf(q_g0, x_g0, sum_g0);

        if (gate1_ptr) {
            const uint8_t b_g1 = *(gate1_ptr + block_offset + 2 + byte_idx);
            const float q_g1 = (float)((b_g1 >> shift) & 0x0F) - 8.0f;
            sum_g1 = fmaf(q_g1, x_g1, sum_g1);
        }

        const uint8_t b_u0 = *(up0_ptr + block_offset + 2 + byte_idx);
        const float q_u0 = (float)((b_u0 >> shift) & 0x0F) - 8.0f;
        sum_u0 = fmaf(q_u0, x_u0, sum_u0);

        if (up1_ptr) {
            const uint8_t b_u1 = *(up1_ptr + block_offset + 2 + byte_idx);
            const float q_u1 = (float)((b_u1 >> shift) & 0x0F) - 8.0f;
            sum_u1 = fmaf(q_u1, x_u1, sum_u1);
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_g0 += __shfl_down_sync(0xffffffff, sum_g0, mask);
        sum_g1 += __shfl_down_sync(0xffffffff, sum_g1, mask);
        sum_u0 += __shfl_down_sync(0xffffffff, sum_u0, mask);
        sum_u1 += __shfl_down_sync(0xffffffff, sum_u1, mask);
    }

    if (lane == 0) {
        float silu0 = silu_scalar(sum_g0);
        out[row0] = silu0 * sum_u0;
        if (row1 < params.m) {
            float silu1 = silu_scalar(sum_g1);
            out[row1] = silu1 * sum_u1;
        }
    }
}

} // extern "C"
