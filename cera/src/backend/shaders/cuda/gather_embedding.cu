// Native CUDA Q8_0 and Q4_0 Embedding Gather kernels.
//
// Directly dequantizes token embeddings on device into activation buffers with
// warp-cooperative 100% coalesced 128-byte writes, eliminating host-side
// dequantization, CPU allocations, and DtoH memory copies.

#include <stdint.h>

__device__ __forceinline__ float half_to_float(uint16_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

extern "C" {

struct GatherParams {
    uint32_t token_id;
    uint32_t hidden_size;
};

// gather_embedding_q8_0: warp-cooperative dequantization of 1 token's Q8_0 embedding row
__global__ void gather_embedding_q8_0(
    float* __restrict__ dst,
    const uint8_t* __restrict__ table,
    GatherParams params
) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t num_warps = blockDim.x >> 5;

    const uint32_t nb = params.hidden_size / 32;
    const size_t row_bytes = (size_t)nb * 34;
    const uint8_t* row_ptr = table + (size_t)params.token_id * row_bytes;

    for (uint32_t ib = warp_id; ib < nb; ib += num_warps) {
        const uint8_t* blk = row_ptr + (size_t)ib * 34;
        float d = 0.0f;
        if (lane == 0) {
            d = half_to_float(*reinterpret_cast<const uint16_t*>(blk));
        }
        d = __shfl_sync(0xffffffff, d, 0);

        const int8_t q = *(const int8_t*)(blk + 2 + lane);
        dst[(size_t)ib * 32 + lane] = (float)q * d;
    }
}

// gather_embedding_q4_0: warp-cooperative dequantization of 1 token's Q4_0 embedding row
__global__ void gather_embedding_q4_0(
    float* __restrict__ dst,
    const uint8_t* __restrict__ table,
    GatherParams params
) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t num_warps = blockDim.x >> 5;

    const uint32_t nb = params.hidden_size / 32;
    const size_t row_bytes = (size_t)nb * 18;
    const uint8_t* row_ptr = table + (size_t)params.token_id * row_bytes;

    const uint32_t byte_idx = lane & 15;
    const uint32_t shift = (lane >> 2) & 4;

    for (uint32_t ib = warp_id; ib < nb; ib += num_warps) {
        const uint8_t* blk = row_ptr + (size_t)ib * 18;
        float d = 0.0f;
        if (lane == 0) {
            d = half_to_float(*reinterpret_cast<const uint16_t*>(blk));
        }
        d = __shfl_sync(0xffffffff, d, 0);

        const uint8_t byte = *(blk + 2 + byte_idx);
        const float q = (float)((byte >> shift) & 0x0F) - 8.0f;
        dst[(size_t)ib * 32 + lane] = q * d;
    }
}

} // extern "C"
