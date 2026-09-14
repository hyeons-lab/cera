// Native CUDA fused per-head RMSNorm + RoPE for Query and Key tensors.
//
// Fuses Q-norm, K-norm, and Rotary Position Embeddings into a single kernel dispatch.
// Dispatched with max(n_heads, n_kv_heads) blocks.

#include <math.h>
#include <stdint.h>

extern "C" {

struct QkNormRopeParams {
    uint32_t pos;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    float eps;
    float freq_base;
    uint32_t rope_type;        // 0 = NeoX (pairs at [i, i+half]), 1 = interleaved (pairs at [2i, 2i+1])
    uint32_t has_freq_factors; // 1 => divide each pair's angle by freq_factors[d]
    uint32_t has_qk_norm;      // 1 => per-head RMSnorm; 0 => rope-only
};

__device__ inline void head_rmsnorm(
    float* buf,
    const float* w,
    float* scratch,
    uint32_t tid,
    uint32_t head_dim,
    float eps
) {
    float partial = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
        float v = buf[i];
        scratch[i] = v;
        partial += v * v;
    }
    __syncthreads();

    // Warp-level reduction
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, mask);
    }

    uint32_t lane = tid & 31;
    uint32_t warp_id = tid / 32;
    float* warp_sums = scratch + head_dim;
    if (lane == 0) {
        warp_sums[warp_id] = partial;
    }
    __syncthreads();

    if (warp_id == 0) {
        uint32_t num_warps = blockDim.x / 32;
        float v = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, mask);
        }
        if (lane == 0) {
            warp_sums[0] = v;
        }
    }
    __syncthreads();

    float inv_rms = rsqrtf(warp_sums[0] / (float)head_dim + eps);
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
        buf[i] = scratch[i] * inv_rms * w[i];
    }
    __syncthreads();
}

__device__ inline void head_rope(
    float* buf,
    uint32_t tid,
    uint32_t head_dim,
    uint32_t pos,
    uint32_t rope_type,
    const float* s_inv_freq
) {
    const uint32_t half_dim = head_dim / 2;

    for (uint32_t d = tid; d < half_dim; d += blockDim.x) {
        float theta = (float)pos * s_inv_freq[d];

        float sin_a, cos_a;
        sincosf(theta, &sin_a, &cos_a);

        if (rope_type == 0) {
            // NeoX style: pairs at [d, d + half_dim]
            float x0 = buf[d];
            float x1 = buf[d + half_dim];
            buf[d] = x0 * cos_a - x1 * sin_a;
            buf[d + half_dim] = x0 * sin_a + x1 * cos_a;
        } else {
            // Interleaved style: pairs at [2d, 2d + 1]
            float x0 = buf[2 * d];
            float x1 = buf[2 * d + 1];
            buf[2 * d] = x0 * cos_a - x1 * sin_a;
            buf[2 * d + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

// qk_norm_rope:
// Grid: (max(n_heads, n_kv_heads), 1, 1)
// Block: (256, 1, 1)
__global__ void qk_norm_rope(
    float* __restrict__ q,
    float* __restrict__ k_cache,
    const float* __restrict__ q_norm_w,
    const float* __restrict__ k_norm_w,
    const float* __restrict__ rope_inv_freq,
    QkNormRopeParams params
) {
    extern __shared__ float shared_scratch[];
    __shared__ float s_inv_freq[64];

    const uint32_t head = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t head_dim = params.head_dim;
    const uint32_t half_dim = head_dim / 2;

    // Load precomputed RoPE inverse frequencies directly into shared memory,
    // eliminating 24,576 runtime powf calls per token across query and KV heads.
    if (tid < half_dim && tid < 64) {
        if (rope_inv_freq != nullptr) {
            s_inv_freq[tid] = rope_inv_freq[tid];
        } else {
            const float theta_scale = powf(params.freq_base, -2.0f / (float)head_dim);
            s_inv_freq[tid] = powf(theta_scale, (float)tid);
        }
    }
    __syncthreads();

    // Process Q head if within n_heads
    if (head < params.n_heads) {
        float* q_head = q + (size_t)head * head_dim;
        if (params.has_qk_norm != 0 && q_norm_w != nullptr) {
            head_rmsnorm(q_head, q_norm_w, shared_scratch, tid, head_dim, params.eps);
        }
        head_rope(
            q_head,
            tid,
            head_dim,
            params.pos,
            params.rope_type,
            s_inv_freq
        );
    }

    // Process K head if within n_kv_heads
    if (head < params.n_kv_heads) {
        float* k_head = k_cache + (size_t)head * head_dim;
        if (params.has_qk_norm != 0 && k_norm_w != nullptr) {
            head_rmsnorm(k_head, k_norm_w, shared_scratch, tid, head_dim, params.eps);
        }
        head_rope(
            k_head,
            tid,
            head_dim,
            params.pos,
            params.rope_type,
            s_inv_freq
        );
    }
}

} // extern "C"
