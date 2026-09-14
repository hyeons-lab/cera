// Native CUDA FlashAttention for single-token autoregressive decode.
//
// Computes multi-head / grouped-query attention with online softmax:
// - 1 block per query head (n_heads thread blocks)
// - 256 threads per block
// - Online softmax rescale: O(1) shared memory regardless of context length
// - Direct FP16 KV cache reading via PTX cvt.f32.f16

#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float half_to_float(uint16_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

extern "C" {

struct AttentionParams {
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t kv_dim;
    uint32_t seq_len;
    float scale;
    uint32_t _pad0;
    uint32_t _pad1;
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

__global__ void flash_attention(
    const float* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    float* __restrict__ out,
    AttentionParams params
) {
    const uint32_t head = blockIdx.x;
    if (head >= params.n_heads) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t head_dim = params.head_dim;
    const uint32_t seq_len = params.seq_len;
    if (seq_len == 0) return;

    const uint32_t group_size = params.n_heads / params.n_kv_heads;
    const uint32_t kv_head = head / group_size;
    const size_t kv_h_offset = (size_t)kv_head * head_dim;
    const size_t kv_dim = params.kv_dim;

    __shared__ float s_q[128];
    __shared__ float s_out[128];
    __shared__ float s_scores[256];
    __shared__ float s_warp_scratch[8];

    // Load Q vector into shared memory
    if (tid < head_dim) {
        s_q[tid] = q[(size_t)head * head_dim + tid];
        s_out[tid] = 0.0f;
    }
    __syncthreads();

    float running_max = -1e30f;
    float running_sum = 0.0f;
    const float scale = params.scale;

    const uint32_t TILE = 256;
    for (uint32_t t_start = 0; t_start < seq_len; t_start += TILE) {
        const uint32_t t = t_start + tid;
        float score = -1e30f;

        if (t < seq_len) {
            const uint16_t* k_ptr = k_cache + (size_t)t * kv_dim + kv_h_offset;
            const uint32_t* k_ptr32 = (const uint32_t*)k_ptr;
            float dot = 0.0f;
            const uint32_t num_pairs = head_dim / 2;
            #pragma unroll 4
            for (uint32_t p = 0; p < num_pairs; p++) {
                const uint32_t pair = k_ptr32[p];
                const uint16_t h0 = (uint16_t)(pair & 0xffff);
                const uint16_t h1 = (uint16_t)(pair >> 16);
                dot = fmaf(s_q[p * 2], half_to_float(h0), dot);
                dot = fmaf(s_q[p * 2 + 1], half_to_float(h1), dot);
            }
            if (head_dim & 1) {
                dot = fmaf(s_q[head_dim - 1], half_to_float(k_ptr[head_dim - 1]), dot);
            }
            score = dot * scale;
        }

        // Compute tile maximum
        float tile_max = block_reduce_max(score, s_warp_scratch);

        // Rescaling terms for online softmax
        float new_max = fmaxf(running_max, tile_max);
        float alpha = expf(running_max - new_max);

        // Rescale output accumulator
        if (tid < head_dim) {
            s_out[tid] *= alpha;
        }

        // Exponentiate scores
        float exp_score = 0.0f;
        if (t < seq_len) {
            exp_score = expf(score - new_max);
        }
        s_scores[tid] = exp_score;
        __syncthreads();

        // Update running sum
        float tile_sum = block_reduce_sum(exp_score, s_warp_scratch);
        running_sum = running_sum * alpha + tile_sum;
        running_max = new_max;

        // Accumulate V projection into s_out:
        // Each thread in 0..head_dim accumulates over the active tile tokens
        const uint32_t tile_count = (seq_len - t_start < TILE) ? (seq_len - t_start) : TILE;
        if (tid < head_dim) {
            float v_acc = 0.0f;
            for (uint32_t it = 0; it < tile_count; it++) {
                const uint32_t tok_idx = t_start + it;
                const uint16_t* v_ptr = v_cache + (size_t)tok_idx * kv_dim + kv_h_offset;
                v_acc = fmaf(s_scores[it], half_to_float(v_ptr[tid]), v_acc);
            }
            s_out[tid] += v_acc;
        }
        __syncthreads();
    }

    // Final normalization
    if (tid < head_dim) {
        float inv_sum = 1.0f / fmaxf(running_sum, 1e-12f);
        out[(size_t)head * head_dim + tid] = s_out[tid] * inv_sum;
    }
}

} // extern "C"
