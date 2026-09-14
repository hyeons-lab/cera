// GPU-resident argmax kernel for autoregressive greedy decode.
//
// Finds the index of the maximum value in a float32 vector (e.g. logits of size vocab_size).
// Inspired by llama.cpp's argmax.cu:
// - Single threadblock with up to 1024 threads.
// - Grid-stride loop to find thread-local maximum and argmax.
// - Intra-warp reduction via __shfl_xor_sync.
// - Inter-warp reduction across warp leaders via shared memory.
// - Writes a single 4-byte uint32 token ID to device memory.

#include <stdint.h>
#include <float.h>
#include <math.h>

extern "C" {

struct ArgmaxParams {
    uint32_t n;      // Number of elements (e.g. vocab_size)
    uint32_t _pad;
};

__global__ void argmax_f32(
    const float* __restrict__ x,
    uint32_t* __restrict__ dst,
    ArgmaxParams params
) {
    const uint32_t n = params.n;
    float maxval = -INFINITY;
    uint32_t argmax = UINT32_MAX;

    const uint32_t n4 = n / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);

    // Vectorized grid-stride loop with 128-bit float4 loads
    for (uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 val4 = x4[i];
        const uint32_t base_idx = i * 4;
        if (val4.x > maxval) { maxval = val4.x; argmax = base_idx; }
        if (val4.y > maxval) { maxval = val4.y; argmax = base_idx + 1; }
        if (val4.z > maxval) { maxval = val4.z; argmax = base_idx + 2; }
        if (val4.w > maxval) { maxval = val4.w; argmax = base_idx + 3; }
    }

    // Scalar remainder loop
    for (uint32_t i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) {
        const float val = x[i];
        if (val > maxval) {
            maxval = val;
            argmax = i;
        }
    }

    // Intra-warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, 32);
        const uint32_t other_idx = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, 32);
        if (other_val > maxval || (other_val == maxval && other_idx < argmax)) {
            maxval = other_val;
            argmax = other_idx;
        }
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t n_warps = blockDim.x >> 5;

    // Inter-warp reduction
    __shared__ float s_maxval[32];
    __shared__ uint32_t s_argmax[32];

    if (lane == 0) {
        s_maxval[warp_id] = maxval;
        s_argmax[warp_id] = argmax;
    }
    __syncthreads();

    if (warp_id == 0) {
        maxval = (lane < n_warps) ? s_maxval[lane] : -INFINITY;
        argmax = (lane < n_warps) ? s_argmax[lane] : UINT32_MAX;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, 32);
            const uint32_t other_idx = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, 32);
            if (other_val > maxval || (other_val == maxval && other_idx < argmax)) {
                maxval = other_val;
                argmax = other_idx;
            }
        }

        if (lane == 0) {
            *dst = argmax;
        }
    }
}

}
