// Native CUDA in-place numerically stable Softmax kernel.
//
// x[i] = exp(x[i] - max(x)) / sum(exp(x - max))
// Uses two-stage warp shuffle reductions for maximum throughput.

#include <math.h>
#include <stdint.h>

extern "C" {

struct SoftmaxParams {
    uint32_t n;
    uint32_t _pad;
};

__device__ inline float block_reduce_max(float val, float* s_warp_max) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, mask));
    }

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane = threadIdx.x & 31;
    if (lane == 0) {
        s_warp_max[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < (blockDim.x / 32)) ? s_warp_max[lane] : -1e30f;
        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, mask));
        }
        if (lane == 0) {
            s_warp_max[0] = v;
        }
    }
    __syncthreads();
    return s_warp_max[0];
}

__device__ inline float block_reduce_sum(float val, float* s_warp_sum) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane = threadIdx.x & 31;
    if (lane == 0) {
        s_warp_sum[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < (blockDim.x / 32)) ? s_warp_sum[lane] : 0.0f;
        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, mask);
        }
        if (lane == 0) {
            s_warp_sum[0] = v;
        }
    }
    __syncthreads();
    return s_warp_sum[0];
}

// softmax: 1 block per row
// Block dimension: 256 threads
__global__ void softmax(
    float* __restrict__ x,
    SoftmaxParams params
) {
    __shared__ float s_scratch[8];

    const uint32_t tid = threadIdx.x;
    const uint32_t n = params.n;
    float* row = x + (size_t)blockIdx.x * n;

    // Step 1: Find row maximum
    float local_max = -1e30f;
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, row[i]);
    }
    float row_max = block_reduce_max(local_max, s_scratch);

    // Step 2: Sum exponentials
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        float exp_val = expf(row[i] - row_max);
        row[i] = exp_val;
        local_sum += exp_val;
    }
    float row_sum = block_reduce_sum(local_sum, s_scratch);
    float inv_sum = 1.0f / fmaxf(row_sum, 1e-12f);

    // Step 3: Normalize
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        row[i] *= inv_sum;
    }
}

} // extern "C"
