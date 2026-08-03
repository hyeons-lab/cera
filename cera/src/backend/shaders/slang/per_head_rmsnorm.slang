// Per-head RMS normalization in-place, single source for both GPU backends.
// Input buffer is [n_heads * head_dim] flat; each workgroup normalizes one head
// x[h*head_dim .. (h+1)*head_dim] by its RMS, then scales by a shared weight.
// Ported from the handwritten pair (`shaders/per_head_rmsnorm.wgsl`,
// `shaders/per_head_rmsnorm.metal`), preserving both contracts so no call site
// changes:
//
//   binding/buffer 0: x       f32, read-write, normalized in place [n_heads*head_dim]
//   binding/buffer 1: weight  f32, read-only [head_dim], shared across heads
//   binding/buffer 2: params  (head_dim, eps_bits, 0, 0)
//
// Dispatch: (n_heads, 1, 1) workgroups of 256, one per head.
//
// ## Why the reduction branches per target
//
// Like softmax.slang, the handwritten kernels deliberately diverge on the
// reduction only: Metal reduces the sum of squares with a two-stage `simd_sum`
// while WGSL walks a shared-memory tree, because cera does not request
// `wgpu::Features::SUBGROUP`. `__target_switch` keeps both (the untaken branch is
// eliminated, so the emitted MSL has no tree and the emitted WGSL no subgroup
// op). Everything else (the per-head offset, the grid-stride loops, the
// normalize pass) is written once.

[[vk::binding(0)]] RWStructuredBuffer<float> x_buf      : register(u0);
[[vk::binding(1)]] StructuredBuffer<float>   weight_buf : register(t1);
[[vk::binding(2)]] StructuredBuffer<uint4>   par_buf    : register(t2);

// Shared scratch. The Metal path needs only 8 slots (one per simdgroup); the
// tree path needs all 256. One declaration keeps the branches from disagreeing
// about the allocation.
groupshared float scratch[256];

static const uint WG = 256u;

/// Sum of `v` across the whole workgroup. Result valid on every thread. Metal
/// reduces with a two-stage `simd_sum` (via `WaveActiveSum`), the portable path
/// with a shared-memory tree; both sum in the same order the handwritten twins
/// do.
float block_sum(uint tid, float v) {
    float result;
    __target_switch {
    case metal:
    {
        float sg = WaveActiveSum(v);
        if ((tid & 31u) == 0u) { scratch[tid >> 5u] = sg; }
        GroupMemoryBarrierWithGroupSync();
        // 256 threads / 32 lanes = 8 partials; lanes past 8 contribute 0.
        float lane = (tid < 8u) ? scratch[tid] : 0.0f;
        float total = WaveActiveSum(lane);
        if (tid == 0u) { scratch[0] = total; }
        GroupMemoryBarrierWithGroupSync();
        result = scratch[0];
        break;
    }
    default:
    {
        scratch[tid] = v;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = WG / 2u; s > 0u; s >>= 1) {
            if (tid < s) { scratch[tid] += scratch[tid + s]; }
            GroupMemoryBarrierWithGroupSync();
        }
        result = scratch[0];
        break;
    }
    }
    return result;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void per_head_rmsnorm(uint3 lid : SV_GroupThreadID, uint3 wid : SV_GroupID) {
    uint tid      = lid.x;
    uint head     = wid.x;
    uint head_dim = par_buf[0].x;
    float eps     = asfloat(par_buf[0].y);
    uint offset   = head * head_dim;

    // Phase 1: partial sum of squares over this head.
    float partial = 0.0f;
    for (uint i = tid; i < head_dim; i += WG) {
        float v = x_buf[offset + i];
        partial += v * v;
    }
    float sum_sq = block_sum(tid, partial);

    float inv_rms = 1.0f / sqrt(sum_sq / float(head_dim) + eps);

    // Phase 2: normalize and scale by the shared weight.
    for (uint i = tid; i < head_dim; i += WG) {
        x_buf[offset + i] = x_buf[offset + i] * inv_rms * weight_buf[i];
    }
}
