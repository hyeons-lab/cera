// Native CUDA Q8_0 Embedding Gather kernel.
//
// Directly dequantizes token embeddings on device into activation buffers,
// eliminating host-side dequantization, CPU allocations, and DtoH memory copies.

#include <stdint.h>
#include <string.h>

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

// gather_embedding_q8_0: dequantizes 1 token's Q8_0 embedding row into float activation buffer
__global__ void gather_embedding_q8_0(
    float* __restrict__ dst,
    const uint8_t* __restrict__ table,
    GatherParams params
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t nb = params.hidden_size / 32;
    const size_t row_bytes = (size_t)nb * 34;
    const uint8_t* row_ptr = table + (size_t)params.token_id * row_bytes;

    for (uint32_t ib = tid; ib < nb; ib += blockDim.x) {
        const uint8_t* blk = row_ptr + (size_t)ib * 34;
        uint16_t d_h;
        memcpy(&d_h, blk, sizeof(uint16_t));
        const float d = half_to_float(d_h);
        const int8_t* qs = (const int8_t*)(blk + 2);

        float* dst_blk = dst + (size_t)ib * 32;
        #pragma unroll 8
        for (int i = 0; i < 32; i++) {
            dst_blk[i] = (float)qs[i] * d;
        }
    }
}

} // extern "C"
