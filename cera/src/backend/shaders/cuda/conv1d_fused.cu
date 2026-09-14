// Native CUDA fused 1D gated short convolution for LFM2 GatedConv blocks.
//
// bx = x * b
// sum = sum_k rbuffer[k, ch] * weight[ch, k] + bx * weight[ch, d_conv]
// rbuffer shifts left one slot and bx is appended
// output[ch] = c * sum

#include <stdint.h>

extern "C" {

struct Conv1dParams {
    uint32_t hs;
    uint32_t kernel_size;
    uint32_t d_conv;
    uint32_t _pad;
};

__global__ void conv1d_fused(
    const float* __restrict__ proj,
    float* __restrict__ rbuffer,
    const float* __restrict__ weight,
    float* __restrict__ output,
    Conv1dParams params
) {
    const uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t hs = params.hs;
    if (ch >= hs) return;

    const uint32_t ks = params.kernel_size;
    const uint32_t d_conv = params.d_conv;

    const float x_val = proj[ch];
    const float c_val = proj[hs + ch];
    const float b_val = proj[2 * hs + ch];
    const float bx = x_val * b_val;

    float sum = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < d_conv; k++) {
        sum = fmaf(rbuffer[(size_t)k * hs + ch], weight[(size_t)ch * ks + k], sum);
    }
    sum = fmaf(bx, weight[(size_t)ch * ks + d_conv], sum);

    if (d_conv > 1) {
        #pragma unroll
        for (uint32_t k = 0; k < d_conv - 1; k++) {
            rbuffer[(size_t)k * hs + ch] = rbuffer[(size_t)(k + 1) * hs + ch];
        }
    }
    if (d_conv > 0) {
        rbuffer[(size_t)(d_conv - 1) * hs + ch] = bx;
    }

    output[ch] = c_val * sum;
}

} // extern "C"
