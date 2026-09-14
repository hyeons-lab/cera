// Native CUDA Q4_0 Matrix-Matrix multiplication (GEMM) for batched prompt prefill.
//
// Computes Y[M, N] = X[M, K] * A[N, K]^T:
// - Tiled execution: TILE_M = 16 tokens, TILE_N = 16 output rows (256 threads per block).
// - Shared memory staging for activations X[16, 32]: 16 tokens reuse weights simultaneously.
// - Reduces weight memory bandwidth traffic by 16x compared to sequential GEMV.
// - Supports arbitrary batch sizes M and dimensions N, K.

#include <stdint.h>

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

// gemm_q4_0: Y = X * A^T
__global__ void gemm_q4_0(
    const uint8_t* __restrict__ a,
    const float* __restrict__ x,
    float* __restrict__ y,
    GemmParams params
) {
    __shared__ float s_x[TILE_M][32];
    __shared__ int8_t s_w[TILE_N][32];
    __shared__ float s_d[TILE_N];

    const uint32_t tx = threadIdx.x; // 0..15 (output row within tile)
    const uint32_t ty = threadIdx.y; // 0..15 (token within tile)
    const uint32_t tid = ty * TILE_N + tx; // 0..255

    const uint32_t m_idx = blockIdx.y * TILE_M + ty;
    const uint32_t n_idx = blockIdx.x * TILE_N + tx;

    const uint32_t nb = params.k / 32;
    const size_t row_bytes = (size_t)nb * 18;

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

        // Cooperatively load and unpack 16x32 Q4_0 weights into shared memory
        // 256 threads each load 1 byte (2 nibbles), covering 256 bytes = 512 weights
        const uint32_t w_row = tid / 16;
        const uint32_t w_col = tid % 16;
        const uint32_t global_n = blockIdx.x * TILE_N + w_row;

        if (global_n < params.n) {
            const uint8_t* blk = a + (size_t)global_n * row_bytes + (size_t)ib * 18;
            const uint8_t byte = *(blk + 2 + w_col);
            s_w[w_row][w_col] = (int8_t)((int)(byte & 0x0F) - 8);
            s_w[w_row][w_col + 16] = (int8_t)((int)(byte >> 4) - 8);
        } else {
            s_w[w_row][w_col] = 0;
            s_w[w_row][w_col + 16] = 0;
        }

        // First 16 threads load the 16 FP16 scales
        if (tid < TILE_N) {
            const uint32_t d_row = blockIdx.x * TILE_N + tid;
            if (d_row < params.n) {
                const uint8_t* blk = a + (size_t)d_row * row_bytes + (size_t)ib * 18;
                s_d[tid] = half_to_float(*reinterpret_cast<const uint16_t*>(blk));
            } else {
                s_d[tid] = 0.0f;
            }
        }

        __syncthreads();

        // Accumulate block dot product
        float dot = 0.0f;
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            dot = fmaf(s_x[ty][k], (float)s_w[tx][k], dot);
        }
        sum = fmaf(dot, s_d[tx], sum);

        __syncthreads();
    }

    if (m_idx < params.m && n_idx < params.n) {
        y[(size_t)m_idx * params.n + n_idx] = sum;
    }
}

// gemm_q4_0_accum: Y += X * A^T (fused residual accumulation)
__global__ void gemm_q4_0_accum(
    const uint8_t* __restrict__ a,
    const float* __restrict__ x,
    float* __restrict__ y,
    GemmParams params
) {
    __shared__ float s_x[TILE_M][32];
    __shared__ int8_t s_w[TILE_N][32];
    __shared__ float s_d[TILE_N];

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t tid = ty * TILE_N + tx;

    const uint32_t m_idx = blockIdx.y * TILE_M + ty;
    const uint32_t n_idx = blockIdx.x * TILE_N + tx;

    const uint32_t nb = params.k / 32;
    const size_t row_bytes = (size_t)nb * 18;

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

        const uint32_t w_row = tid / 16;
        const uint32_t w_col = tid % 16;
        const uint32_t global_n = blockIdx.x * TILE_N + w_row;

        if (global_n < params.n) {
            const uint8_t* blk = a + (size_t)global_n * row_bytes + (size_t)ib * 18;
            const uint8_t byte = *(blk + 2 + w_col);
            s_w[w_row][w_col] = (int8_t)((int)(byte & 0x0F) - 8);
            s_w[w_row][w_col + 16] = (int8_t)((int)(byte >> 4) - 8);
        } else {
            s_w[w_row][w_col] = 0;
            s_w[w_row][w_col + 16] = 0;
        }

        if (tid < TILE_N) {
            const uint32_t d_row = blockIdx.x * TILE_N + tid;
            if (d_row < params.n) {
                const uint8_t* blk = a + (size_t)d_row * row_bytes + (size_t)ib * 18;
                s_d[tid] = half_to_float(*reinterpret_cast<const uint16_t*>(blk));
            } else {
                s_d[tid] = 0.0f;
            }
        }

        __syncthreads();

        float dot = 0.0f;
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            dot = fmaf(s_x[ty][k], (float)s_w[tx][k], dot);
        }
        sum = fmaf(dot, s_d[tx], sum);

        __syncthreads();
    }

    if (m_idx < params.m && n_idx < params.n) {
        y[(size_t)m_idx * params.n + n_idx] += sum;
    }
}

} // extern "C"
