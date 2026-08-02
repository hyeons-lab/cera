// Softmax in-place, single source for both GPU backends.
//
//   x[i] = exp(x[i] - max(x)) / sum(exp(x - max))
//
// Single workgroup of 256, grid-striding over n. Ported from the handwritten
// pair it is meant to replace (`shaders/softmax.wgsl`, `shaders/softmax.metal`),
// preserving both contracts exactly so no call site changes:
//
//   binding/buffer 0: x       f32, read-write, softmax applied in place
//   binding/buffer 1: params  (n, _pad)
//
// `[[vk::binding(n)]]` drives the WGSL binding index and `register(u0/t1)` the
// Metal buffer index; they are set independently because the two backends number
// their slots from different spaces, and here they happen to agree.
//
// ## Why the reductions differ per target
//
// The handwritten kernels deliberately diverged: Metal reduces with a two-stage
// `simd_max`/`simd_sum` while WGSL walks a shared-memory tree, because cera does
// not request `wgpu::Features::SUBGROUP` and so has no portable wave intrinsic.
// `__target_switch` keeps both, and the untaken branch is eliminated rather than
// compiled and skipped: the emitted MSL contains no trace of the tree, and the
// emitted WGSL contains no subgroup op. Consolidating on one reduction instead
// would either cost Metal its simd path or force a device-capability change, so
// the split stays until `Features::SUBGROUP` is decided on its own merits.
//
// Everything that carries the kernel's meaning (layout, grid-stride loops, the
// max-shift, the exp and normalize passes) is written once; only the two
// reduction helpers branch.

[[vk::binding(0)]] RWStructuredBuffer<float> x_buf   : register(u0);
[[vk::binding(1)]] StructuredBuffer<uint2>   par_buf : register(t1);

// Shared scratch. The Metal path needs only 8 slots (one per simdgroup) but the
// tree path needs all 256; a single declaration keeps the two branches from
// disagreeing about the allocation.
groupshared float scratch[256];

static const uint WG = 256u;

/// Max of `v` across the whole workgroup. Result valid on every thread.
float block_max(uint tid, float v) {
    float result;
    __target_switch {
    case metal:
    {
        float sg = WaveActiveMax(v);
        if ((tid & 31u) == 0u) { scratch[tid >> 5u] = sg; }
        GroupMemoryBarrierWithGroupSync();
        // 256 threads / 32 lanes = 8 partials; lanes past 8 contribute -inf.
        float lane = (tid < 8u) ? scratch[tid] : -3.402823466e+38f;
        float total = WaveActiveMax(lane);
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
            if (tid < s) { scratch[tid] = max(scratch[tid], scratch[tid + s]); }
            GroupMemoryBarrierWithGroupSync();
        }
        result = scratch[0];
        break;
    }
    }
    return result;
}

/// Sum of `v` across the whole workgroup. Result valid on every thread.
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
void softmax(uint3 lid : SV_GroupThreadID) {
    uint tid = lid.x;
    uint n = par_buf[0].x;

    // Phase 1: max, for numerical stability.
    float local_max = -3.402823466e+38f;
    for (uint i = tid; i < n; i += WG) {
        local_max = max(local_max, x_buf[i]);
    }
    float max_val = block_max(tid, local_max);

    // `block_sum` reuses `scratch`, so there is a WAR between `block_max`'s
    // `scratch[0]` read and `block_sum`'s later write. Whether that needs a
    // barrier here is target-conditional:
    //
    //   - metal: match the handwritten reference (`shaders/softmax.metal`),
    //     which does this exact phase transition with no barrier here. The same
    //     WAR is present in that shipped kernel; its tight post-reduction read
    //     plus the intervening exp loop make a lapping race practically
    //     impossible, and it has run that way in production. Adding a barrier
    //     here only emits a fifth `threadgroup_barrier` the reference never
    //     pays: at small n the kernel is barrier-latency-bound (a few elements
    //     per thread), so that one extra barrier measured as a ~24% regression
    //     (0.76x at n=1024, closing to ~1.00x by n=16384). Drop it to match.
    //   - default (portable tree path): targets arbitrary GPUs with weaker
    //     scheduling guarantees, so keep the guard.
    __target_switch {
    case metal:
    {
        break;
    }
    default:
    {
        GroupMemoryBarrierWithGroupSync();
        break;
    }
    }

    // Phase 2: exp(x - max), keeping the running sum.
    float partial = 0.0f;
    for (uint i = tid; i < n; i += WG) {
        float e = exp(x_buf[i] - max_val);
        x_buf[i] = e;
        partial += e;
    }
    float inv_sum = 1.0f / block_sum(tid, partial);

    // Phase 3: normalize.
    for (uint i = tid; i < n; i += WG) {
        x_buf[i] = x_buf[i] * inv_sum;
    }
}
