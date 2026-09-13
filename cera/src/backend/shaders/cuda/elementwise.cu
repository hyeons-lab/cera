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

// silu_mul_inplace: a[i] = (silu(a[i])) * b[i]
// Used for SwiGLU FFN activation
__global__ void silu_mul_inplace(
    float* __restrict__ a,
    const float* __restrict__ b,
    ElementwiseParams params
) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.n) return;
    float g = a[gid];
    // Clamp to avoid exp overflow
    if (g < -80.0f) g = -80.0f;
    else if (g > 80.0f) g = 80.0f;
    const float silu_g = g / (1.0f + expf(-g));
    a[gid] = silu_g * b[gid];
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
