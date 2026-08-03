// slang-entries: rmsnorm_batch add_rmsnorm_batch
//
// Batched RMSnorm, single source for both GPU backends. Each workgroup
// normalizes one row. Two entry points sharing the binding layout, ported from
// the handwritten pair (`shaders/rmsnorm_batch.wgsl`, `.metal`):
//
//   rmsnorm_batch      dst[i] = src[i] * inv_rms(src) * w[i]          (reads src)
//   add_rmsnorm_batch  src[i] += res_scale * residual[i]; then        (writes src)
//                      dst[i] = src[i] * inv_rms(src) * w[i]
//
//   binding/buffer 0: src       f32, read-write (add_ writes back the post-add value)
//   binding/buffer 1: dst       f32, read-write, normalized output
//   binding/buffer 2: w         f32, read, per-element scale [n]
//   binding/buffer 3: params    (n, eps_bits, src_stride, dst_stride, res_scale_bits)
//   binding/buffer 4: residual  f32, read, only used by add_rmsnorm_batch (stride = src_stride)
//
// Dispatch: (rows, 1, 1) workgroups of 256. `residual` (binding 4) is referenced
// only by add_rmsnorm_batch, so the rmsnorm_batch pipeline never binds it (the
// per-entry bind group layout drops it), the same way each handwritten entry
// point does.
//
// Reduction divergence via `__target_switch` in `block_sum`: metal reduces the
// sum of squares with a two-stage `simd_sum` (via `WaveActiveSum`), the portable
// path with a shared-memory tree. Each thread owns its grid-strided indices, so
// no barrier is needed between add_rmsnorm_batch's in-place write and the later
// re-read (matching the handwritten kernels).

[[vk::binding(0)]] RWStructuredBuffer<float> src_buf : register(u0);
[[vk::binding(1)]] RWStructuredBuffer<float> dst_buf : register(u1);
[[vk::binding(2)]] StructuredBuffer<float>   w_buf   : register(t2);
[[vk::binding(3)]] StructuredBuffer<uint>    par_buf : register(t3);
[[vk::binding(4)]] StructuredBuffer<float>   res_buf : register(t4);

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
void rmsnorm_batch(uint3 lid : SV_GroupThreadID, uint3 wid : SV_GroupID) {
    uint tid     = lid.x;
    uint n       = par_buf[0];
    float eps    = asfloat(par_buf[1]);
    uint src_off = wid.x * par_buf[2];
    uint dst_off = wid.x * par_buf[3];

    float partial = 0.0f;
    for (uint i = tid; i < n; i += WG) {
        float v = src_buf[src_off + i];
        partial += v * v;
    }
    float inv_rms = 1.0f / sqrt(block_sum(tid, partial) / float(n) + eps);

    for (uint i = tid; i < n; i += WG) {
        dst_buf[dst_off + i] = src_buf[src_off + i] * inv_rms * w_buf[i];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_rmsnorm_batch(uint3 lid : SV_GroupThreadID, uint3 wid : SV_GroupID) {
    uint tid       = lid.x;
    uint n         = par_buf[0];
    float eps      = asfloat(par_buf[1]);
    uint src_off   = wid.x * par_buf[2];
    uint dst_off   = wid.x * par_buf[3];
    float res_scale = asfloat(par_buf[4]);

    // Phase 1: add the scaled residual in place (residual shares src's stride)
    // and accumulate the post-add sum of squares.
    float partial = 0.0f;
    for (uint i = tid; i < n; i += WG) {
        float v = src_buf[src_off + i] + res_scale * res_buf[src_off + i];
        src_buf[src_off + i] = v;
        partial += v * v;
    }
    float inv_rms = 1.0f / sqrt(block_sum(tid, partial) / float(n) + eps);

    for (uint i = tid; i < n; i += WG) {
        dst_buf[dst_off + i] = src_buf[src_off + i] * inv_rms * w_buf[i];
    }
}
