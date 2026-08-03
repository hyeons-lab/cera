// RMSnorm over a single vector (hidden_size <= 8192): x /= rms(x), then scale by
// weight. Single workgroup of 256, grid-striding over n. One Slang source for
// both GPU backends, but this kernel diverges MORE than the others: the two
// handwritten twins disagree on both the reduction AND the I/O model, so both
// diverge via `__target_switch`.
//
//   wgsl (mirrors `shaders/rmsnorm.wgsl`): IN-PLACE, 3 bindings
//     binding 0: x       f32, read-write, normalized in place
//     binding 1: weight  f32, read
//     binding 2: params  (n, eps_bits, 0, 0)
//
//   metal (mirrors `shaders/rmsnorm.metal`): OUT-OF-PLACE, 4 buffers
//     buffer 0: src      f32, read
//     buffer 1: dst      f32, write
//     buffer 2: w        f32, read
//     buffer 3: params   (n, eps_bits, 0, 0)
//
// Each branch references only its own binding set; Slang drops the other set for
// the target it is not building (the same elimination that keeps rope's
// freq_factors out of the MSL and its `metal_powr` out of the WGSL). So the
// emitted WGSL has exactly 3 bindings and the emitted MSL exactly 4 buffers, each
// matching its handwritten twin with no call site change.
//
// The reduction (sum of squares) is the usual softmax-class split inside
// `block_sum`: metal `WaveActiveSum` -> two-stage simd_sum, wgsl shared-memory
// tree.

// wgsl (in-place) binding set. register() values are for Metal, where these are
// eliminated, so they are irrelevant there.
[[vk::binding(0)]] RWStructuredBuffer<float> x_buf  : register(u0);
[[vk::binding(1)]] StructuredBuffer<float>   w_wgsl : register(t1);
[[vk::binding(2)]] StructuredBuffer<uint4>   p_wgsl : register(t2);

// metal (out-of-place) binding set. vk::binding() values are for WGSL, where
// these are eliminated, so they are irrelevant there; register() sets the Metal
// buffer index (src=0, dst=1, w=2, params=3).
[[vk::binding(3)]] StructuredBuffer<float>   src_buf : register(t0);
[[vk::binding(4)]] RWStructuredBuffer<float> dst_buf : register(u1);
[[vk::binding(5)]] StructuredBuffer<float>   w_metal : register(t2);
[[vk::binding(6)]] StructuredBuffer<uint4>   p_metal : register(t3);

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
void rmsnorm(uint3 lid : SV_GroupThreadID) {
    uint tid = lid.x;

    __target_switch {
    case metal:
    {
        // Out-of-place: read src, write dst.
        uint n    = p_metal[0].x;
        float eps = asfloat(p_metal[0].y);

        float partial = 0.0f;
        for (uint i = tid; i < n; i += WG) {
            float v = src_buf[i];
            partial += v * v;
        }
        float inv_rms = 1.0f / sqrt(block_sum(tid, partial) / float(n) + eps);

        for (uint i = tid; i < n; i += WG) {
            dst_buf[i] = src_buf[i] * inv_rms * w_metal[i];
        }
        break;
    }
    default:
    {
        // In-place: normalize x.
        uint n    = p_wgsl[0].x;
        float eps = asfloat(p_wgsl[0].y);

        float partial = 0.0f;
        for (uint i = tid; i < n; i += WG) {
            float v = x_buf[i];
            partial += v * v;
        }
        float inv_rms = 1.0f / sqrt(block_sum(tid, partial) / float(n) + eps);

        for (uint i = tid; i < n; i += WG) {
            x_buf[i] = x_buf[i] * inv_rms * w_wgsl[i];
        }
        break;
    }
    }
}
