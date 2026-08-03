// Batched affine LayerNorm, single source for both GPU backends. Each workgroup
// normalizes one row:
//   dst[i] = (src[i] - mean) * inv_std * weight[i] + bias[i]
// with population variance and an explicit bias (distinct from rmsnorm). Ported
// from the handwritten pair (`shaders/layernorm_batch.wgsl`,
// `shaders/layernorm_batch.metal`), preserving both contracts so no call site
// changes:
//
//   binding/buffer 0: src     f32, read, input rows (row stride = params.z)
//   binding/buffer 1: dst     f32, read-write, output rows (row stride = params.w)
//   binding/buffer 2: weight  f32, read, per-element scale [n]
//   binding/buffer 3: bias    f32, read, per-element shift [n]
//   binding/buffer 4: params  (n, eps_bits, src_stride, dst_stride)
//
// Dispatch: (rows, 1, 1) workgroups of 256.
//
// ## Reduction and the between-passes barrier
//
// The reduction divergence is the same as softmax/per_head_rmsnorm: metal
// reduces with a two-stage `simd_sum` (via `WaveActiveSum`), the portable path
// with a shared-memory tree, kept apart by `__target_switch` in `block_sum`.
// There are two reductions per row (mean, then variance). The barrier between
// them is present on BOTH targets (unlike softmax, where metal drops the
// max->sum barrier): both handwritten kernels keep it, since only the short
// `(src - mean)^2` loop separates the first reduction's `scratch` reads from the
// second's writes.

[[vk::binding(0)]] StructuredBuffer<float>   src_buf    : register(t0);
[[vk::binding(1)]] RWStructuredBuffer<float> dst_buf    : register(u1);
[[vk::binding(2)]] StructuredBuffer<float>   weight_buf : register(t2);
[[vk::binding(3)]] StructuredBuffer<float>   bias_buf   : register(t3);
[[vk::binding(4)]] StructuredBuffer<uint4>   par_buf    : register(t4);

// Shared scratch: metal uses 8 slots (one per simdgroup), the tree uses all 256.
groupshared float scratch[256];

static const uint WG = 256u;

/// Sum of `v` across the whole workgroup. Result valid on every thread. Metal
/// reduces with a two-stage `simd_sum` (via `WaveActiveSum`), the portable path
/// with a shared-memory tree.
float block_sum(uint tid, float v) {
    float result;
    __target_switch {
    case metal:
    {
        float sg = WaveActiveSum(v);
        if ((tid & 31u) == 0u) { scratch[tid >> 5u] = sg; }
        GroupMemoryBarrierWithGroupSync();
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
void layernorm_batch(uint3 lid : SV_GroupThreadID, uint3 wid : SV_GroupID) {
    uint tid     = lid.x;
    uint row     = wid.x;
    uint n       = par_buf[0].x;
    float eps    = asfloat(par_buf[0].y);
    uint src_off = row * par_buf[0].z;
    uint dst_off = row * par_buf[0].w;

    // Pass 1: mean = (1/n) sum(src).
    float partial = 0.0f;
    for (uint i = tid; i < n; i += WG) {
        partial += src_buf[src_off + i];
    }
    float mean = block_sum(tid, partial) / float(n);

    // Separate the two reductions' use of `scratch` (kept on both targets).
    GroupMemoryBarrierWithGroupSync();

    // Pass 2: var = (1/n) sum((src - mean)^2).
    partial = 0.0f;
    for (uint i = tid; i < n; i += WG) {
        float d = src_buf[src_off + i] - mean;
        partial += d * d;
    }
    float inv_std = 1.0f / sqrt(block_sum(tid, partial) / float(n) + eps);

    // Pass 3: affine normalize into dst.
    for (uint i = tid; i < n; i += WG) {
        dst_buf[dst_off + i] =
            (src_buf[src_off + i] - mean) * inv_std * weight_buf[i] + bias_buf[i];
    }
}
