// Fast F32 GEMV: y[row_base + r] = A_f32[r, k] @ x[k].
// SPIR-V passthrough kernel (entry `main`), dispatched on Vulkan when
// subgroups are available; `../gemv_f32.wgsl` stays as the portable fallback.
//
// Same 4-binding interface and NR=8 rows-per-workgroup as `gemv_f32.wgsl`:
//   binding 0: weights A (f32)
//   binding 1: x vector (f32)
//   binding 2: y vector (f32, read-write)
//   binding 3: params (uint4: m, k, row_base, unused)
//
// Uses 64 threads per workgroup (full Adreno/RDNA wave64) and `WaveActiveSum`
// subgroup reduction + 1 shared-memory barrier instead of the 32-thread
// 5-barrier tree reduction in `gemv_f32.wgsl`.

[[vk::binding(0, 0)]] StructuredBuffer<float>   a;
[[vk::binding(1, 0)]] StructuredBuffer<float>   x;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(3, 0)]] StructuredBuffer<uint4>   params;

static const uint NR = 8u;
static const uint WG_SIZE = 64u;
static const uint MAX_SUBGROUPS = 64u;

groupshared float sg_scratch[NR * MAX_SUBGROUPS];

uint get_wid(uint3 wid) {
    return wid.x + wid.y * 65535u;
}

float4 load_a_vec4(uint idx) {
    return float4(a[idx], a[idx + 1u], a[idx + 2u], a[idx + 3u]);
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(
    uint3 lid : SV_GroupThreadID,
    uint3 wid : SV_GroupID
) {
    uint m = params[0].x;
    uint k = params[0].y;
    uint row_base = params[0].z;
    uint tid = lid.x;
    uint r0 = get_wid(wid) * NR;

    float sums[NR];
    [unroll]
    for (uint r = 0u; r < NR; r++) {
        sums[r] = 0.0f;
    }

    // 4-wide vectorized loop: 64 threads process 256 elements per step.
    uint k_vec = k & ~3u;
    for (uint col = tid * 4u; col < k_vec; col += WG_SIZE * 4u) {
        float4 xv = float4(x[col], x[col + 1u], x[col + 2u], x[col + 3u]);
        [unroll]
        for (uint r = 0u; r < NR; r++) {
            if (r0 + r < m) {
                float4 av = load_a_vec4((r0 + r) * k + col);
                sums[r] += dot(av, xv);
            }
        }
    }

    // Scalar tail for any remaining unaligned elements.
    for (uint col = k_vec + tid; col < k; col += WG_SIZE) {
        float xv = x[col];
        [unroll]
        for (uint r = 0u; r < NR; r++) {
            if (r0 + r < m) {
                sums[r] += a[(r0 + r) * k + col] * xv;
            }
        }
    }

    uint lane = WaveGetLaneIndex();
    uint lane_count = WaveGetLaneCount();
    uint sgid = tid / lane_count;
    [unroll]
    for (uint r = 0u; r < NR; r++) {
        float s = WaveActiveSum(sums[r]);
        if (lane == 0u) {
            sg_scratch[r * MAX_SUBGROUPS + sgid] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0u) {
        uint nsg = max(WG_SIZE / lane_count, 1u);
        [unroll]
        for (uint r = 0u; r < NR; r++) {
            float t = 0.0f;
            for (uint i = 0u; i < nsg; i++) {
                t += sg_scratch[r * MAX_SUBGROUPS + i];
            }
            if (r0 + r < m) {
                y[row_base + r0 + r] = t;
            }
        }
    }
}
