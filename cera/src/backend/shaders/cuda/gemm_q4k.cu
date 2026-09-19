// Native CUDA Q4_K_M Matrix-Matrix multiplication (GEMM) for batched prompt prefill.
//
// Computes Y[M, N] = X[M, K] * A[N, K]^T:
// - Tiled execution: TILE_M = 16 tokens, TILE_N = 16 output rows (256 threads per block).
// - Shared memory staging for activations X[16, 32] and weights W[16, 32].
// - Supports arbitrary batch sizes M and dimensions N, K (k multiple of 256).

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
    uint32_t k; // Number of input columns (multiple of 256)
    uint32_t _pad;
};

#define TILE_M 16
#define TILE_N 16

// gemm_q4k: Y = X * A^T
__global__ void gemm_q4k(
    const uint8_t* __restrict__ a,
    const float* __restrict__ x,
    float* __restrict__ y,
    GemmParams params
) {
    __shared__ float s_x[TILE_M][33];
    __shared__ float s_w[TILE_N][33];

    const uint32_t tx = threadIdx.x; // 0..15 (output row within tile)
    const uint32_t ty = threadIdx.y; // 0..15 (token within tile)
    const uint32_t tid = ty * TILE_N + tx; // 0..255

    const uint32_t m_idx = blockIdx.y * TILE_M + ty;
    const uint32_t n_idx = blockIdx.x * TILE_N + tx;

    const uint32_t nb = params.k / 256;
    const size_t row_bytes = (size_t)nb * 144;

    float sum = 0.0f;

    for (uint32_t ib = 0; ib < nb; ib++) {
        // Each super-block has 8 sub-blocks of 32 elements (4 pairs of lo/hi)
        for (uint32_t j = 0; j < 4; j++) {
            // Sub-block 2*j (low nibbles)
            {
                // Load 16x32 activations
                const uint32_t elem0 = tid * 2;
                const uint32_t elem1 = elem0 + 1;
                const uint32_t tok0 = elem0 / 32;
                const uint32_t col0 = elem0 % 32;
                const uint32_t gm0 = blockIdx.y * TILE_M + tok0;
                if (gm0 < params.m) {
                    s_x[tok0][col0] = x[(size_t)gm0 * params.k + (size_t)ib * 256 + j * 64 + col0];
                } else {
                    s_x[tok0][col0] = 0.0f;
                }

                const uint32_t tok1 = elem1 / 32;
                const uint32_t col1 = elem1 % 32;
                const uint32_t gm1 = blockIdx.y * TILE_M + tok1;
                if (gm1 < params.m) {
                    s_x[tok1][col1] = x[(size_t)gm1 * params.k + (size_t)ib * 256 + j * 64 + col1];
                } else {
                    s_x[tok1][col1] = 0.0f;
                }

                // Dequantize 16x32 weights for sub-block 2*j into s_w
                // 256 threads handle 16 rows x 16 bytes = 256 threads (1 byte per thread)
                const uint32_t w_row = tid / 16;
                const uint32_t w_col = tid % 16;
                const uint32_t gn = blockIdx.x * TILE_N + w_row;

                if (gn < params.n) {
                    const uint8_t* blk = a + (size_t)gn * row_bytes + (size_t)ib * 144;
                    const float d = half_to_float(*reinterpret_cast<const uint16_t*>(blk));
                    const float dmin = half_to_float(*reinterpret_cast<const uint16_t*>(blk + 2));
                    const uint8_t* sc_ptr = blk + 4;

                    uint32_t sc_val = 0, mn_val = 0;
                    const uint32_t sb = 2 * j;
                    if (sb < 4) {
                        sc_val = sc_ptr[sb] & 63;
                        mn_val = sc_ptr[sb + 4] & 63;
                    } else {
                        sc_val = (sc_ptr[sb + 4] & 0x0F) | ((sc_ptr[sb - 4] >> 6) << 4);
                        mn_val = (sc_ptr[sb + 4] >> 4) | ((sc_ptr[sb] >> 6) << 4);
                    }

                    const float d_sc = d * (float)sc_val;
                    const float d_mn = dmin * (float)mn_val;

                    const uint8_t byte = *(blk + 16 + j * 32 + w_col);
                    s_w[w_row][w_col] = fmaf(d_sc, (float)(byte & 0x0F), -d_mn);

                    const uint8_t byte2 = *(blk + 16 + j * 32 + w_col + 16);
                    s_w[w_row][w_col + 16] = fmaf(d_sc, (float)(byte2 & 0x0F), -d_mn);
                } else {
                    s_w[w_row][w_col] = 0.0f;
                    s_w[w_row][w_col + 16] = 0.0f;
                }

                __syncthreads();

                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    sum = fmaf(s_x[ty][k], s_w[tx][k], sum);
                }

                __syncthreads();
            }

            // Sub-block 2*j + 1 (high nibbles)
            {
                const uint32_t elem0 = tid * 2;
                const uint32_t elem1 = elem0 + 1;
                const uint32_t tok0 = elem0 / 32;
                const uint32_t col0 = elem0 % 32;
                const uint32_t gm0 = blockIdx.y * TILE_M + tok0;
                if (gm0 < params.m) {
                    s_x[tok0][col0] = x[(size_t)gm0 * params.k + (size_t)ib * 256 + j * 64 + 32 + col0];
                } else {
                    s_x[tok0][col0] = 0.0f;
                }

                const uint32_t tok1 = elem1 / 32;
                const uint32_t col1 = elem1 % 32;
                const uint32_t gm1 = blockIdx.y * TILE_M + tok1;
                if (gm1 < params.m) {
                    s_x[tok1][col1] = x[(size_t)gm1 * params.k + (size_t)ib * 256 + j * 64 + 32 + col1];
                } else {
                    s_x[tok1][col1] = 0.0f;
                }

                const uint32_t w_row = tid / 16;
                const uint32_t w_col = tid % 16;
                const uint32_t gn = blockIdx.x * TILE_N + w_row;

                if (gn < params.n) {
                    const uint8_t* blk = a + (size_t)gn * row_bytes + (size_t)ib * 144;
                    const float d = half_to_float(*reinterpret_cast<const uint16_t*>(blk));
                    const float dmin = half_to_float(*reinterpret_cast<const uint16_t*>(blk + 2));
                    const uint8_t* sc_ptr = blk + 4;

                    uint32_t sc_val = 0, mn_val = 0;
                    const uint32_t sb = 2 * j + 1;
                    if (sb < 4) {
                        sc_val = sc_ptr[sb] & 63;
                        mn_val = sc_ptr[sb + 4] & 63;
                    } else {
                        sc_val = (sc_ptr[sb + 4] & 0x0F) | ((sc_ptr[sb - 4] >> 6) << 4);
                        mn_val = (sc_ptr[sb + 4] >> 4) | ((sc_ptr[sb] >> 6) << 4);
                    }

                    const float d_sc = d * (float)sc_val;
                    const float d_mn = dmin * (float)mn_val;

                    const uint8_t byte = *(blk + 16 + j * 32 + w_col);
                    s_w[w_row][w_col] = fmaf(d_sc, (float)(byte >> 4), -d_mn);

                    const uint8_t byte2 = *(blk + 16 + j * 32 + w_col + 16);
                    s_w[w_row][w_col + 16] = fmaf(d_sc, (float)(byte2 >> 4), -d_mn);
                } else {
                    s_w[w_row][w_col] = 0.0f;
                    s_w[w_row][w_col + 16] = 0.0f;
                }

                __syncthreads();

                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    sum = fmaf(s_x[ty][k], s_w[tx][k], sum);
                }

                __syncthreads();
            }
        }
    }

    if (m_idx < params.m && n_idx < params.n) {
        y[(size_t)m_idx * params.n + n_idx] = sum;
    }
}

// gemm_q4k_accum: Y += X * A^T
__global__ void gemm_q4k_accum(
    const uint8_t* __restrict__ a,
    const float* __restrict__ x,
    float* __restrict__ y,
    GemmParams params
) {
    __shared__ float s_x[TILE_M][33];
    __shared__ float s_w[TILE_N][33];

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t tid = ty * TILE_N + tx;

    const uint32_t m_idx = blockIdx.y * TILE_M + ty;
    const uint32_t n_idx = blockIdx.x * TILE_N + tx;

    const uint32_t nb = params.k / 256;
    const size_t row_bytes = (size_t)nb * 144;

    float sum = 0.0f;

    for (uint32_t ib = 0; ib < nb; ib++) {
        for (uint32_t j = 0; j < 4; j++) {
            // Sub-block 2*j (low nibbles)
            {
                const uint32_t elem0 = tid * 2;
                const uint32_t elem1 = elem0 + 1;
                const uint32_t tok0 = elem0 / 32;
                const uint32_t col0 = elem0 % 32;
                const uint32_t gm0 = blockIdx.y * TILE_M + tok0;
                if (gm0 < params.m) {
                    s_x[tok0][col0] = x[(size_t)gm0 * params.k + (size_t)ib * 256 + j * 64 + col0];
                } else {
                    s_x[tok0][col0] = 0.0f;
                }

                const uint32_t tok1 = elem1 / 32;
                const uint32_t col1 = elem1 % 32;
                const uint32_t gm1 = blockIdx.y * TILE_M + tok1;
                if (gm1 < params.m) {
                    s_x[tok1][col1] = x[(size_t)gm1 * params.k + (size_t)ib * 256 + j * 64 + col1];
                } else {
                    s_x[tok1][col1] = 0.0f;
                }

                const uint32_t w_row = tid / 16;
                const uint32_t w_col = tid % 16;
                const uint32_t gn = blockIdx.x * TILE_N + w_row;

                if (gn < params.n) {
                    const uint8_t* blk = a + (size_t)gn * row_bytes + (size_t)ib * 144;
                    const float d = half_to_float(*reinterpret_cast<const uint16_t*>(blk));
                    const float dmin = half_to_float(*reinterpret_cast<const uint16_t*>(blk + 2));
                    const uint8_t* sc_ptr = blk + 4;

                    uint32_t sc_val = 0, mn_val = 0;
                    const uint32_t sb = 2 * j;
                    if (sb < 4) {
                        sc_val = sc_ptr[sb] & 63;
                        mn_val = sc_ptr[sb + 4] & 63;
                    } else {
                        sc_val = (sc_ptr[sb + 4] & 0x0F) | ((sc_ptr[sb - 4] >> 6) << 4);
                        mn_val = (sc_ptr[sb + 4] >> 4) | ((sc_ptr[sb] >> 6) << 4);
                    }

                    const float d_sc = d * (float)sc_val;
                    const float d_mn = dmin * (float)mn_val;

                    const uint8_t byte = *(blk + 16 + j * 32 + w_col);
                    s_w[w_row][w_col] = fmaf(d_sc, (float)(byte & 0x0F), -d_mn);

                    const uint8_t byte2 = *(blk + 16 + j * 32 + w_col + 16);
                    s_w[w_row][w_col + 16] = fmaf(d_sc, (float)(byte2 & 0x0F), -d_mn);
                } else {
                    s_w[w_row][w_col] = 0.0f;
                    s_w[w_row][w_col + 16] = 0.0f;
                }

                __syncthreads();

                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    sum = fmaf(s_x[ty][k], s_w[tx][k], sum);
                }

                __syncthreads();
            }

            // Sub-block 2*j + 1 (high nibbles)
            {
                const uint32_t elem0 = tid * 2;
                const uint32_t elem1 = elem0 + 1;
                const uint32_t tok0 = elem0 / 32;
                const uint32_t col0 = elem0 % 32;
                const uint32_t gm0 = blockIdx.y * TILE_M + tok0;
                if (gm0 < params.m) {
                    s_x[tok0][col0] = x[(size_t)gm0 * params.k + (size_t)ib * 256 + j * 64 + 32 + col0];
                } else {
                    s_x[tok0][col0] = 0.0f;
                }

                const uint32_t tok1 = elem1 / 32;
                const uint32_t col1 = elem1 % 32;
                const uint32_t gm1 = blockIdx.y * TILE_M + tok1;
                if (gm1 < params.m) {
                    s_x[tok1][col1] = x[(size_t)gm1 * params.k + (size_t)ib * 256 + j * 64 + 32 + col1];
                } else {
                    s_x[tok1][col1] = 0.0f;
                }

                const uint32_t w_row = tid / 16;
                const uint32_t w_col = tid % 16;
                const uint32_t gn = blockIdx.x * TILE_N + w_row;

                if (gn < params.n) {
                    const uint8_t* blk = a + (size_t)gn * row_bytes + (size_t)ib * 144;
                    const float d = half_to_float(*reinterpret_cast<const uint16_t*>(blk));
                    const float dmin = half_to_float(*reinterpret_cast<const uint16_t*>(blk + 2));
                    const uint8_t* sc_ptr = blk + 4;

                    uint32_t sc_val = 0, mn_val = 0;
                    const uint32_t sb = 2 * j + 1;
                    if (sb < 4) {
                        sc_val = sc_ptr[sb] & 63;
                        mn_val = sc_ptr[sb + 4] & 63;
                    } else {
                        sc_val = (sc_ptr[sb + 4] & 0x0F) | ((sc_ptr[sb - 4] >> 6) << 4);
                        mn_val = (sc_ptr[sb + 4] >> 4) | ((sc_ptr[sb] >> 6) << 4);
                    }

                    const float d_sc = d * (float)sc_val;
                    const float d_mn = dmin * (float)mn_val;

                    const uint8_t byte = *(blk + 16 + j * 32 + w_col);
                    s_w[w_row][w_col] = fmaf(d_sc, (float)(byte >> 4), -d_mn);

                    const uint8_t byte2 = *(blk + 16 + j * 32 + w_col + 16);
                    s_w[w_row][w_col + 16] = fmaf(d_sc, (float)(byte2 >> 4), -d_mn);
                } else {
                    s_w[w_row][w_col] = 0.0f;
                    s_w[w_row][w_col + 16] = 0.0f;
                }

                __syncthreads();

                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    sum = fmaf(s_x[ty][k], s_w[tx][k], sum);
                }

                __syncthreads();
            }
        }
    }

    if (m_idx < params.m && n_idx < params.n) {
        y[(size_t)m_idx * params.n + n_idx] += sum;
    }
}

} // extern "C"
