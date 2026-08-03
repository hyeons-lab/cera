// Argmax over an f32 array: writes the index of the maximum element to out[0].
// Single workgroup of 256, grid-striding over n. One Slang source for both GPU
// backends, ported from the handwritten pair (`shaders/argmax_f32.wgsl`,
// `shaders/argmax_f32.metal`), preserving both contracts:
//
//   binding/buffer 0: x       f32, read, logits [n]
//   binding/buffer 1: out     u32, read-write, writes out[0] = argmax index
//   binding/buffer 2: params  (n, 0)
//
// Dispatch: (1, 1, 1) workgroup of 256. n up to a few hundred K (vocab).
//
// ## Why the reduction branches per target
//
// Unlike the sum reductions (rmsnorm/softmax), argmax reduces a (value, index)
// PAIR, so a plain simd_sum/tree of a scalar is not enough. The handwritten
// kernels diverge just like softmax: Metal uses a two-stage `simd_shuffle_down`
// that moves both the float value and the uint index between lanes; WGSL walks a
// shared-memory tree. `__target_switch` keeps both. (The coopmat-style probe in
// devlog 000279-01 confirmed Slang emits `simd_shuffle_down`.)
//
// Tie-break note: on an EXACT value tie the two branches can pick different
// indices, exactly as the two handwritten twins already do. The WGSL tree keeps
// the lower index explicitly; the Metal shuffle-down keeps the lower LANE (strict
// `>`), which is not the same as the lower index across lanes. This is a
// non-issue in practice (real logits do not tie and the argmax is unique), and
// the parity fixture injects a unique peak so the branches must agree.

[[vk::binding(0)]] StructuredBuffer<float>  x_buf   : register(t0);
[[vk::binding(1)]] RWStructuredBuffer<uint> out_buf : register(u1);
[[vk::binding(2)]] StructuredBuffer<uint2>  par_buf : register(t2);

// Shared scratch for the (value, index) pair. Metal uses 8 slots (one per
// simdgroup); the tree uses all 256. One declaration each.
groupshared float scratch_v[256];
groupshared uint  scratch_i[256];

static const uint WG = 256u;

// Metal-only lane exchange: the exact `simd_shuffle_down` the handwritten kernel
// uses, for the float value and the uint index. Referenced only in the metal
// branch, so Slang omits them from the WGSL (which walks the tree instead).
__target_intrinsic(metal, "simd_shuffle_down($0, $1)")
float sg_shuffle_down_f(float v, uint delta);
__target_intrinsic(metal, "simd_shuffle_down($0, $1)")
uint sg_shuffle_down_u(uint v, uint delta);

/// Argmax of the per-thread (value, index) pairs across the whole workgroup.
/// Result (the winning index) is valid on every thread. On an exact value tie
/// the two branches can differ (wgsl keeps the lower index, metal the lower
/// lane); see the header's tie-break note.
uint block_argmax(uint tid, float val, uint idx) {
    uint result_idx;
    __target_switch {
    case metal:
    {
        // Stage 1: reduce within each simdgroup with shuffle-down. Strict `>`
        // keeps the current (lower-lane) index on a tie.
        float best_v = val;
        uint best_i = idx;
        [ForceUnroll]
        for (uint off = 16u; off > 0u; off >>= 1u) {
            float ov = sg_shuffle_down_f(best_v, off);
            uint oi = sg_shuffle_down_u(best_i, off);
            if (ov > best_v) {
                best_v = ov;
                best_i = oi;
            }
        }
        // Each simdgroup's lane 0 publishes its winner.
        uint simd_lane = tid & 31u;
        uint simd_id = tid >> 5u;
        if (simd_lane == 0u) {
            scratch_v[simd_id] = best_v;
            scratch_i[simd_id] = best_i;
        }
        GroupMemoryBarrierWithGroupSync();
        // Stage 2: the first simdgroup reduces the 8 partials.
        if (simd_id == 0u) {
            float v = (simd_lane < 8u) ? scratch_v[simd_lane] : -3.402823466e+38f;
            uint ix = (simd_lane < 8u) ? scratch_i[simd_lane] : 0u;
            [ForceUnroll]
            for (uint off = 4u; off > 0u; off >>= 1u) {
                float ov = sg_shuffle_down_f(v, off);
                uint oi = sg_shuffle_down_u(ix, off);
                if (ov > v) {
                    v = ov;
                    ix = oi;
                }
            }
            if (simd_lane == 0u) { scratch_i[0] = ix; }
        }
        GroupMemoryBarrierWithGroupSync();
        result_idx = scratch_i[0];
        break;
    }
    default:
    {
        // Shared-memory tree with an explicit lower-index tie-break.
        scratch_v[tid] = val;
        scratch_i[tid] = idx;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = WG / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                float ov = scratch_v[tid + stride];
                uint oi = scratch_i[tid + stride];
                if (ov > scratch_v[tid]) {
                    scratch_v[tid] = ov;
                    scratch_i[tid] = oi;
                } else if (ov == scratch_v[tid] && oi < scratch_i[tid]) {
                    scratch_i[tid] = oi;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        result_idx = scratch_i[0];
        break;
    }
    }
    return result_idx;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void argmax_f32(uint3 lid : SV_GroupThreadID) {
    uint tid = lid.x;
    uint n = par_buf[0].x;

    // Phase 1: thread-local argmax with stride WG. Strict `>` favors the lower
    // index (the value already recorded), matching the CPU `argmax`.
    float local_max = -3.402823466e+38f;
    uint local_idx = 0u;
    for (uint i = tid; i < n; i += WG) {
        float v = x_buf[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    uint winner = block_argmax(tid, local_max, local_idx);
    if (tid == 0u) {
        out_buf[0] = winner;
    }
}
