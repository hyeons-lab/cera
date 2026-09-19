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

// gather_embedding_q4k: warp-cooperative dequantization of 1 token's Q4_K embedding row
__global__ void gather_embedding_q4k(
    float* __restrict__ dst,
    const uint8_t* __restrict__ table,
    GatherParams params
) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t num_warps = blockDim.x >> 5;

    const uint32_t nb = params.hidden_size / 256;
    const size_t row_bytes = (size_t)nb * 144;
    const uint8_t* row_ptr = table + (size_t)params.token_id * row_bytes;

    for (uint32_t ib = warp_id; ib < nb; ib += num_warps) {
        const size_t blk_off = (size_t)ib * 144;

        float d = 0.0f;
        float dmin = 0.0f;
        if (lane == 0) {
            d = half_to_float(*reinterpret_cast<const uint16_t*>(row_ptr + blk_off));
            dmin = half_to_float(*reinterpret_cast<const uint16_t*>(row_ptr + blk_off + 2));
        }
        d = __shfl_sync(0xffffffff, d, 0);
        dmin = __shfl_sync(0xffffffff, dmin, 0);

        uint32_t sc = 0, mn = 0;
        if (lane < 4) {
            const uint8_t* sc_ptr = row_ptr + blk_off + 4;
            sc = sc_ptr[lane] & 63;
            mn = sc_ptr[lane + 4] & 63;
        } else if (lane < 8) {
            const uint8_t* sc_ptr = row_ptr + blk_off + 4;
            sc = (sc_ptr[lane + 4] & 0x0F) | ((sc_ptr[lane - 4] >> 6) << 4);
            mn = (sc_ptr[lane + 4] >> 4) | ((sc_ptr[lane] >> 6) << 4);
        }

        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            const float sc_lo = (float)__shfl_sync(0xffffffff, sc, 2 * j);
            const float mn_lo = (float)__shfl_sync(0xffffffff, mn, 2 * j);
            const float sc_hi = (float)__shfl_sync(0xffffffff, sc, 2 * j + 1);
            const float mn_hi = (float)__shfl_sync(0xffffffff, mn, 2 * j + 1);

            const float d_sc_lo = d * sc_lo;
            const float d_mn_lo = dmin * mn_lo;
            const float d_sc_hi = d * sc_hi;
            const float d_mn_hi = dmin * mn_hi;

            const uint8_t byte = *(row_ptr + blk_off + 16 + j * 32 + lane);
            const float q_lo = (float)(byte & 0x0F);
            const float q_hi = (float)(byte >> 4);

            dst[ib * 256 + j * 64 + lane] = fmaf(d_sc_lo, q_lo, -d_mn_lo);
            dst[ib * 256 + j * 64 + 32 + lane] = fmaf(d_sc_hi, q_hi, -d_mn_hi);
        }
    }
}

} // extern "C"
