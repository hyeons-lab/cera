// Native CUDA Q8_0 Matrix-Matrix multiplication (GEMM) for batched prompt prefill.
//
// Computes Y[M, N] = X[M, K] * A[N, K]^T:
// - Tiled execution: TILE_M = 16 tokens, TILE_N = 16 output rows (256 threads per block).
// - Shared memory staging for activations X[16, 32]: 16 tokens reuse weights simultaneously.
// - Reduces weight memory bandwidth traffic by 16x compared to sequential GEMV.
// - Supports arbitrary batch sizes M and dimensions N, K.

#include <stdint.h>
#include <string.h>

__device__ __forceinline__ float half_to_float(uint16_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

extern "C" {

struct GemmParams {
    uint32_t m; // Number of batch tokens
    uint32_t n; // Number of output rows
    uint32_t k; // Number of input columns (multiple of 32)
};

#define TILE_M 16
#define TILE_N 16

// gemm_q8_0: Y = X * A^T
__global__ void gemm_q8_0(
    const uint8_t* __restrict__ a,
    const float* __restrict__ x,
    float* __restrict__ y,
    GemmParams params
) {
    __shared__ float s_x[TILE_M][32];

    const uint32_t tx = threadIdx.x; // 0..15 (output row within tile)
    const uint32_t ty = threadIdx.y; // 0..15 (token within tile)
    const uint32_t tid = ty * TILE_N + tx; // 0..255

    const uint32_t m_idx = blockIdx.y * TILE_M + ty;
    const uint32_t n_idx = blockIdx.x * TILE_N + tx;

    const uint32_t nb = params.k / 32;
    const size_t row_bytes = (size_t)nb * 34;
    const uint8_t* row_ptr = (n_idx < params.n) ? (a + (size_t)n_idx * row_bytes) : nullptr;

    float sum = 0.0f;

    for (uint32_t ib = 0; ib < nb; ib++) {
        // Cooperatively load 16x32 activations into shared memory
        // 256 threads load 512 floats (2 floats per thread)
        const uint32_t elem0 = tid * 2;
        const uint32_t elem1 = elem0 + 1;

        const uint32_t tok0 = elem0 / 32;
        const uint32_t col0 = elem0 % 32;
        const uint32_t global_m0 = blockIdx.y * TILE_M + tok0;
        if (global_m0 < params.m) {
            s_x[tok0][col0] = x[(size_t)global_m0 * params.k + (size_t)ib * 32 + col0];
        } else {
            s_x[tok0][col0] = 0.0f;
        }

        const uint32_t tok1 = elem1 / 32;
        const uint32_t col1 = elem1 % 32;
        const uint32_t global_m1 = blockIdx.y * TILE_M + tok1;
        if (global_m1 < params.m) {
            s_x[tok1][col1] = x[(size_t)global_m1 * params.k + (size_t)ib * 32 + col1];
        } else {
            s_x[tok1][col1] = 0.0f;
        }

        __syncthreads();

        // Compute dot product for active (m_idx, n_idx)
        if (row_ptr && m_idx < params.m) {
            const uint8_t* blk = row_ptr + (size_t)ib * 34;
            uint16_t d_h;
            memcpy(&d_h, blk, sizeof(uint16_t));
            const float d = half_to_float(d_h);
            const int8_t* qs = (const int8_t*)(blk + 2);

            float dot = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < 32; i++) {
                dot += (float)qs[i] * s_x[ty][i];
            }
            sum += dot * d;
        }

        __syncthreads();
    }

    if (m_idx < params.m && n_idx < params.n) {
        y[(size_t)m_idx * params.n + n_idx] = sum;
    }
}

// gemm_q8_0_accum: Y += X * A^T (fused residual accumulation)
__global__ void gemm_q8_0_accum(
    const uint8_t* __restrict__ a,
    const float* __restrict__ x,
    float* __restrict__ y,
    GemmParams params
) {
    __shared__ float s_x[TILE_M][32];

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t tid = ty * TILE_N + tx;

    const uint32_t m_idx = blockIdx.y * TILE_M + ty;
    const uint32_t n_idx = blockIdx.x * TILE_N + tx;

    const uint32_t nb = params.k / 32;
    const size_t row_bytes = (size_t)nb * 34;
    const uint8_t* row_ptr = (n_idx < params.n) ? (a + (size_t)n_idx * row_bytes) : nullptr;

    float sum = 0.0f;

    for (uint32_t ib = 0; ib < nb; ib++) {
        const uint32_t elem0 = tid * 2;
        const uint32_t elem1 = elem0 + 1;

        const uint32_t tok0 = elem0 / 32;
        const uint32_t col0 = elem0 % 32;
        const uint32_t global_m0 = blockIdx.y * TILE_M + tok0;
        if (global_m0 < params.m) {
            s_x[tok0][col0] = x[(size_t)global_m0 * params.k + (size_t)ib * 32 + col0];
        } else {
            s_x[tok0][col0] = 0.0f;
        }

        const uint32_t tok1 = elem1 / 32;
        const uint32_t col1 = elem1 % 32;
        const uint32_t global_m1 = blockIdx.y * TILE_M + tok1;
        if (global_m1 < params.m) {
            s_x[tok1][col1] = x[(size_t)global_m1 * params.k + (size_t)ib * 32 + col1];
        } else {
            s_x[tok1][col1] = 0.0f;
        }

        __syncthreads();

        if (row_ptr && m_idx < params.m) {
            const uint8_t* blk = row_ptr + (size_t)ib * 34;
            uint16_t d_h;
            memcpy(&d_h, blk, sizeof(uint16_t));
            const float d = half_to_float(d_h);
            const int8_t* qs = (const int8_t*)(blk + 2);

            float dot = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < 32; i++) {
                dot += (float)qs[i] * s_x[ty][i];
            }
            sum += dot * d;
        }

        __syncthreads();
    }

    if (m_idx < params.m && n_idx < params.n) {
        y[(size_t)m_idx * params.n + n_idx] += sum;
    }
}

} // extern "C"
