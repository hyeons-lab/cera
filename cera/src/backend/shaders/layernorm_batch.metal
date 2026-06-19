#include <metal_stdlib>
using namespace metal;

// Batched affine LayerNorm (MSL mirror of layernorm_batch.wgsl):
//   dst[i] = (src[i] - mean) * inv_std * weight[i] + bias[i]
// per row, with population variance and an explicit bias (distinct from
// rmsnorm.metal). One threadgroup of 256 threads per row; two two-stage simd
// reductions (mean, then variance).
//
// Dispatch: threadgroups (rows, 1, 1), threads (256, 1, 1).

struct Params { uint n; uint eps_bits; uint src_stride; uint dst_stride; };

kernel void layernorm_batch(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    const device float* weight [[buffer(2)]],
    const device float* bias [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    uint n = p.n;
    float eps = as_type<float>(p.eps_bits);
    uint src_off = row * p.src_stride;
    uint dst_off = row * p.dst_stride;

    threadgroup float sg[8];
    uint lane = tid & 31u;
    uint sid = tid >> 5u;

    // Pass 1: mean.
    float partial = 0.0f;
    for (uint i = tid; i < n; i += 256u) {
        partial += src[src_off + i];
    }
    float s = simd_sum(partial);
    if (lane == 0u) sg[sid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0u) {
        float v = lane < 8u ? sg[lane] : 0.0f;
        float t = simd_sum(v);
        if (lane == 0u) sg[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = sg[0] / float(n);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: variance.
    partial = 0.0f;
    for (uint i = tid; i < n; i += 256u) {
        float d = src[src_off + i] - mean;
        partial += d * d;
    }
    s = simd_sum(partial);
    if (lane == 0u) sg[sid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0u) {
        float v = lane < 8u ? sg[lane] : 0.0f;
        float t = simd_sum(v);
        if (lane == 0u) sg[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_std = 1.0f / sqrt(sg[0] / float(n) + eps);

    // Pass 3: affine normalize.
    for (uint i = tid; i < n; i += 256u) {
        dst[dst_off + i] = (src[src_off + i] - mean) * inv_std * weight[i] + bias[i];
    }
}
