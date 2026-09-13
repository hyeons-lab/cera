// Native CUDA Root Mean Square Normalization (RMSNorm).
//
// y_i = (x_i / sqrt(mean(x^2) + eps)) * w_i
// Supports both in-place and out-of-place normalization.

#include <math.h>
#include <stdint.h>

extern "C" {

struct RmsNormParams {
    uint32_t n;    // Hidden dimension size
    float eps;     // Numerical epsilon (e.g. 1e-5 or 1e-6)
};

// rmsnorm: 1 block per row/token
// Block dimension: 256 threads (8 warps)
__global__ void rmsnorm(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    RmsNormParams params
) {
    __shared__ float s_warp_sums[8];
    __shared__ float s_scale;

    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane = tid & 31;
    const uint32_t n = params.n;

    const float* row_x = x + (size_t)blockIdx.x * n;
    float* row_y = y + (size_t)blockIdx.x * n;

    // Phase 1: Sum of squares
    float sum_sq = 0.0f;
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        float val = row_x[i];
        sum_sq += val * val;
    }

    // Reduce within warp
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, mask);
    }

    if (lane == 0) {
        s_warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    // Reduce across warps (8 warps in a 256-thread block)
    if (tid < 8) {
        float block_sum = s_warp_sums[tid];
        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, mask);
        }
        if (tid == 0) {
            s_scale = rsqrtf(block_sum / (float)n + params.eps);
        }
    }
    __syncthreads();

    const float scale = s_scale;

    // Phase 2: Normalize and scale with weights
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        row_y[i] = row_x[i] * scale * w[i];
    }
}

} // extern "C"
