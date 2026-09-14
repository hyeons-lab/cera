// Native CUDA element-wise operations on f32 buffers.
//
// Supports:
// - memcpy_f32
// - add_inplace
// - mul_inplace
// - mul_out
// - cast_f32_to_f16
// - silu_mul_inplace (SwiGLU gate * up)
// - scaled_add_inplace (Granite scaled residual add)
// - scale_f32

#include <math.h>
#include <stdint.h>

__device__ __forceinline__ uint16_t float_to_half(float f) {
    uint16_t h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

extern "C" {

struct ElementwiseParams {
    uint32_t n;
    uint32_t _pad;
};

struct ScaleParams {
    uint32_t n;
    float scale;
};

__global__ void memcpy_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    ElementwiseParams params
) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.n) return;
    dst[gid] = src[gid];
}

__global__ void add_inplace(
    float* __restrict__ a,
    const float* __restrict__ b,
    ElementwiseParams params
) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.n) return;
    a[gid] = a[gid] + b[gid];
}

__global__ void mul_inplace(
    float* __restrict__ a,
    const float* __restrict__ b,
    ElementwiseParams params
) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.n) return;
    a[gid] = a[gid] * b[gid];
}

__global__ void mul_out(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ dst,
    ElementwiseParams params
) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.n) return;
    dst[gid] = a[gid] * b[gid];
}

__global__ void cast_f32_to_f16(
    const float* __restrict__ src,
    uint16_t* __restrict__ dst,
    ElementwiseParams params
) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.n) return;
    dst[gid] = float_to_half(src[gid]);
}

// Fused dual KV cache append: converts and appends both K and V vectors in a single kernel dispatch
__global__ void append_kv_cache_f16(
    const float* __restrict__ k_src,
    const float* __restrict__ v_src,
    uint16_t* __restrict__ k_dst,
    uint16_t* __restrict__ v_dst,
    ElementwiseParams params
) {
    const uint32_t num_vec4 = params.n / 4;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < num_vec4) {
        const float4 k_val = reinterpret_cast<const float4*>(k_src)[gid];
        const float4 v_val = reinterpret_cast<const float4*>(v_src)[gid];

        uint16_t k_h[4];
        k_h[0] = float_to_half(k_val.x);
        k_h[1] = float_to_half(k_val.y);
        k_h[2] = float_to_half(k_val.z);
        k_h[3] = float_to_half(k_val.w);

        uint16_t v_h[4];
        v_h[0] = float_to_half(v_val.x);
        v_h[1] = float_to_half(v_val.y);
        v_h[2] = float_to_half(v_val.z);
        v_h[3] = float_to_half(v_val.w);

        *reinterpret_cast<uint64_t*>(k_dst + (size_t)gid * 4) = *reinterpret_cast<const uint64_t*>(k_h);
        *reinterpret_cast<uint64_t*>(v_dst + (size_t)gid * 4) = *reinterpret_cast<const uint64_t*>(v_h);
    }

    const uint32_t rem_start = num_vec4 * 4;
    if (gid == 0) {
        for (uint32_t i = rem_start; i < params.n; i++) {
            k_dst[i] = float_to_half(k_src[i]);
            v_dst[i] = float_to_half(v_src[i]);
        }
    }
}

__device__ __forceinline__ float silu_scalar(float g) {
    if (g < -80.0f) g = -80.0f;
    else if (g > 80.0f) g = 80.0f;
    return g / (1.0f + expf(-g));
}

// silu_mul_inplace: a[i] = (silu(a[i])) * b[i]
// Vectorized with float4 for 4x fewer load/store transactions
__global__ void silu_mul_inplace(
    float* __restrict__ a,
    const float* __restrict__ b,
    ElementwiseParams params
) {
    const uint32_t num_vec4 = params.n / 4;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < num_vec4) {
        float4 a_v = reinterpret_cast<float4*>(a)[gid];
        const float4 b_v = reinterpret_cast<const float4*>(b)[gid];

        a_v.x = silu_scalar(a_v.x) * b_v.x;
        a_v.y = silu_scalar(a_v.y) * b_v.y;
        a_v.z = silu_scalar(a_v.z) * b_v.z;
        a_v.w = silu_scalar(a_v.w) * b_v.w;

        reinterpret_cast<float4*>(a)[gid] = a_v;
    }

    const uint32_t rem_start = num_vec4 * 4;
    if (gid == 0) {
        for (uint32_t i = rem_start; i < params.n; i++) {
            a[i] = silu_scalar(a[i]) * b[i];
        }
    }
}

__global__ void scaled_add_inplace(
    float* __restrict__ a,
    const float* __restrict__ b,
    ScaleParams params
) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.n) return;
    a[gid] = a[gid] + params.scale * b[gid];
}

__global__ void scale_f32(
    float* __restrict__ a,
    ScaleParams params
) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.n) return;
    a[gid] = a[gid] * params.scale;
}

} // extern "C"
