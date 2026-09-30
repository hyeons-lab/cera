// Fast Q4_K (Q4_K_M) decode GEMV: y[m] = dequant(W_q4_k[m, k]) @ x[k].
// SPIR-V passthrough kernel (entry `main`), dispatched on Vulkan when
// subgroups are available; `../gemv_q4_k.wgsl` stays as the portable fallback.
//
// Same 4-binding interface as the WGSL kernel:
//   binding 0: weights as u32 words (raw Q4_K super-blocks)
//   binding 1: x vector (f32)
//   binding 2: y vector (f32, read-write)
//   binding 3: params (uint4: m, k, unused, unused)
//
// Key improvements over gemv_q4_k.wgsl:
//   - 64 threads covering NQ=2 super-blocks per iteration.
//   - NR=4 rows per workgroup (WGSL did NR=2), amortizing activation reads
//     across 4 output rows.
//   - Subgroup reduction (`WaveActiveSum` + one shared-mem combine) eliminating
//     the 5-barrier workgroup tree reduction.
//   - Vectorized u32 word loads and single f16tof32 unpacking for (d, dmin).

[[vk::binding(0, 0)]] StructuredBuffer<uint>    w;
[[vk::binding(1, 0)]] StructuredBuffer<float>   x;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(3, 0)]] StructuredBuffer<uint4>   params;

static const uint QK_K = 256u;
static const uint Q4K_BYTES = 144u;
static const uint NR = 4u;
static const uint WG_SIZE = 64u;
static const uint NQ = 2u;
static const uint MAX_SUBGROUPS = 64u;

groupshared float sg_scratch[NR * MAX_SUBGROUPS];

uint get_wid(uint3 wid) {
    return wid.x + wid.y * 65535u;
}

uint scb(uint s0, uint s1, uint s2, uint b) {
    uint val = (b < 4u) ? s0 : ((b < 8u) ? s1 : s2);
    return (val >> ((b & 3u) * 8u)) & 0xFFu;
}

uint q4k_sc(uint s0, uint s1, uint s2, uint sub) {
    if (sub < 4u) {
        return scb(s0, s1, s2, sub) & 63u;
    }
    return (scb(s0, s1, s2, sub + 4u) & 0x0Fu) | ((scb(s0, s1, s2, sub - 4u) >> 6u) << 4u);
}

uint q4k_mn(uint s0, uint s1, uint s2, uint sub) {
    if (sub < 4u) {
        return scb(s0, s1, s2, sub + 4u) & 63u;
    }
    return (scb(s0, s1, s2, sub + 4u) >> 4u) | ((scb(s0, s1, s2, sub) >> 6u) << 4u);
}

float dq(uint val, uint byte_idx, uint hi, float scale, float minv) {
    uint qb = (val >> (byte_idx * 8u)) & 0xFFu;
    uint nib = (hi == 0u) ? (qb & 0x0Fu) : (qb >> 4u);
    return scale * float(nib) - minv;
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(
    uint3 lid : SV_GroupThreadID,
    uint3 wid : SV_GroupID
) {
    uint m = params[0].x;
    uint k = params[0].y;
    uint nb = k / QK_K;
    uint row_bytes = nb * Q4K_BYTES;
    uint r0 = get_wid(wid) * NR;
    uint tid = lid.x;

    uint slot = tid / 32u;
    uint tiisg = tid % 32u;
    uint e0 = tiisg * 8u;
    uint j = e0 / 64u;
    uint o = e0 % 64u;
    uint hi = o / 32u;
    uint sub = 2u * j + hi;
    uint qbase = 32u * j + (o % 32u);

    float sumf[NR];
    [unroll]
    for (uint r = 0u; r < NR; r++) {
        sumf[r] = 0.0f;
    }

    for (uint ib_base = 0u; ib_base < nb; ib_base += NQ) {
        uint ib = ib_base + slot;
        if (ib >= nb) {
            continue;
        }

        uint x_off = ib * QK_K + e0;
        float xl0 = x[x_off + 0u];
        float xl1 = x[x_off + 1u];
        float xl2 = x[x_off + 2u];
        float xl3 = x[x_off + 3u];
        float xl4 = x[x_off + 4u];
        float xl5 = x[x_off + 5u];
        float xl6 = x[x_off + 6u];
        float xl7 = x[x_off + 7u];

        [unroll]
        for (uint r = 0u; r < NR; r++) {
            uint row = r0 + r;
            if (row >= m) {
                continue;
            }

            uint blk = row * row_bytes + ib * Q4K_BYTES;
            uint dword = w[blk / 4u];
            float d = f16tof32(dword & 0xFFFFu);
            float dmin = f16tof32(dword >> 16u);

            uint sw = blk / 4u + 1u;
            uint s0 = w[sw];
            uint s1 = w[sw + 1u];
            uint s2 = w[sw + 2u];

            float scale = d * float(q4k_sc(s0, s1, s2, sub));
            float minv = dmin * float(q4k_mn(s0, s1, s2, sub));

            uint qw = (blk + 16u + qbase) / 4u;
            uint qw0 = w[qw];
            uint qw1 = w[qw + 1u];

            float s = 0.0f;
            s += dq(qw0, 0u, hi, scale, minv) * xl0;
            s += dq(qw0, 1u, hi, scale, minv) * xl1;
            s += dq(qw0, 2u, hi, scale, minv) * xl2;
            s += dq(qw0, 3u, hi, scale, minv) * xl3;
            s += dq(qw1, 0u, hi, scale, minv) * xl4;
            s += dq(qw1, 1u, hi, scale, minv) * xl5;
            s += dq(qw1, 2u, hi, scale, minv) * xl6;
            s += dq(qw1, 3u, hi, scale, minv) * xl7;

            sumf[r] += s;
        }
    }

    uint lane = WaveGetLaneIndex();
    uint lane_count = WaveGetLaneCount();
    uint sgid = tid / lane_count;
    [unroll]
    for (uint r = 0u; r < NR; r++) {
        float s = WaveActiveSum(sumf[r]);
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
                y[r0 + r] = t;
            }
        }
    }
}
