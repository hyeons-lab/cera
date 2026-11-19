// Small-M high-split Q4_0 GEMV over the resident stream layout:
// y[m] = dequant(W[m, k]) @ x[k], tuned for m <= 2048 (e.g. ffn_down,
// out_proj, q/k/v/o/gate projections) where 16 rows/WG under-occupies
// high-CU GPUs (e.g. 40 CUs -> only 128 WGs at m=2048).
//
// 8 rows x 16-way K-split in 128-thread workgroups (4 wave32s per WG),
// 1 row per thread (f = tid % 8, s = tid / 8). At m=2048 this launches
// 256 WGs = 1024 wave32s while each 8-fiber group still reads a full 32B
// aligned sector of consecutive rows from `q` and `d`.
//
// Bindings match `gemv_q4_0_stream.slang`:
//   0 = q (u32), 1 = d (u32), 2 = x (float4), 3 = y (f32, rw), 4 = params (uint4).
// Dispatch: gemv_row_workgroups(ceil(m / 8)). Precondition: k % 64 == 0.

[[vk::binding(0, 0)]] StructuredBuffer<uint>    q;
[[vk::binding(1, 0)]] StructuredBuffer<uint>    d;
[[vk::binding(2, 0)]] StructuredBuffer<float4>  x;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(4, 0)]] StructuredBuffer<uint4>   params;

static const uint ROWS_PER_WG = 8u;
static const uint FIBERS = 8u;
static const uint K_SPLITS = 16u;
static const uint WG_SIZE = 128u;

groupshared float split_scratch[ROWS_PER_WG * K_SPLITS];

uint get_wid(uint3 wid) {
    return wid.x + wid.y * 65535u;
}

[shader("compute")]
[WaveSize(32)]
[numthreads(WG_SIZE, 1, 1)]
void main(
    uint3 lid : SV_GroupThreadID,
    uint3 wid : SV_GroupID
) {
    uint m = params[0].x;
    uint k = params[0].y;
    uint tid = lid.x;
    uint f = tid & (FIBERS - 1u);
    uint s = tid / FIBERS;
    uint wid_flat = get_wid(wid);
    uint row = wid_flat * ROWS_PER_WG + f;

    bool live = row < m;
    uint rowc = live ? row : (m - 1u);

    float acc = 0.0f;

    uint nb = k / 32u;
    uint nb_per_split = (nb + K_SPLITS - 1u) / K_SPLITS;
    uint b0 = s * nb_per_split;
    uint b1 = min(b0 + nb_per_split, nb);

    for (uint b = b0; b < b1; b++) {
        uint xbase4 = b * 8u;
        uint di = (b / 2u) * m;
        uint dshift = (b & 1u) * 16u;
        uint qi = b * 4u * m;
        float ds = live ? float(f16tof32((d[di + rowc] >> dshift) & 0xFFFFu)) : 0.0f;
        float d0 = 0.0f;
        float d1 = 0.0f;
        float d2 = 0.0f;
        float d3 = 0.0f;
        float sumy_b = 0.0f;

        [unroll]
        for (uint p = 0u; p < 4u; p++) {
            uint xb4 = xbase4 + p * 2u;
            float4 a03 = x[xb4];
            float4 a47 = x[xb4 + 1u];
            float a0 = a03.x;        float y0 = a0;
            float a1 = a03.y;        float y1 = a1 * 0.0625f;
            float a2 = a03.z;        float y2 = a2 * 0.00390625f;
            float a3 = a03.w;        float y3 = a3 * 0.000244140625f;
            float a4 = a47.x;        float y4 = a4 * 0.0000152587890625f;
            float a5 = a47.y;        float y5 = a5 * 9.5367431640625e-7f;
            float a6 = a47.z;        float y6 = a6 * 5.960464477539063e-8f;
            float a7 = a47.w;        float y7 = a7 * 3.725290298461914e-9f;
            sumy_b += (a0 + a1) + (a2 + a3) + (a4 + a5) + (a6 + a7);

            uint qw = q[qi + p * m + rowc];
            d0 += y0 * float(qw & 0x0000000Fu) + y4 * float(qw & 0x000F0000u);
            d1 += y1 * float(qw & 0x000000F0u) + y5 * float(qw & 0x00F00000u);
            d2 += y2 * float(qw & 0x00000F00u) + y6 * float(qw & 0x0F000000u);
            d3 += y3 * float(qw & 0x0000F000u) + y7 * float(qw & 0xF0000000u);
        }
        acc += ds * (((d0 + d1) + (d2 + d3)) - 8.0f * sumy_b);
    }

    uint lane = WaveGetLaneIndex();
    uint lane_count = WaveGetLaneCount();
    if (lane_count == 32u) {
        acc += WaveReadLaneAt(acc, lane ^ 8u);
        acc += WaveReadLaneAt(acc, lane ^ 16u);
        if ((s & 3u) == 0u) {
            split_scratch[f * 4u + (s >> 2u)] = live ? acc : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid < ROWS_PER_WG) {
            uint out_row = wid_flat * ROWS_PER_WG + tid;
            if (out_row < m) {
                uint base = tid * 4u;
                y[out_row] = (split_scratch[base + 0u] + split_scratch[base + 1u])
                           + (split_scratch[base + 2u] + split_scratch[base + 3u]);
            }
        }
    } else if (lane_count == 64u) {
        acc += WaveReadLaneAt(acc, lane ^ 8u);
        acc += WaveReadLaneAt(acc, lane ^ 16u);
        acc += WaveReadLaneAt(acc, lane ^ 32u);
        if ((s & 7u) == 0u) {
            split_scratch[f * 2u + (s >> 3u)] = live ? acc : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid < ROWS_PER_WG) {
            uint out_row = wid_flat * ROWS_PER_WG + tid;
            if (out_row < m) {
                uint base = tid * 2u;
                y[out_row] = split_scratch[base + 0u] + split_scratch[base + 1u];
            }
        }
    } else {
        split_scratch[f * K_SPLITS + s] = live ? acc : 0.0f;
        GroupMemoryBarrierWithGroupSync();

        if (tid < ROWS_PER_WG) {
            uint out_row = wid_flat * ROWS_PER_WG + tid;
            if (out_row < m) {
                uint base = tid * K_SPLITS;
                float total = 0.0f;
                [unroll]
                for (uint i = 0u; i < K_SPLITS; i++) {
                    total += split_scratch[base + i];
                }
                y[out_row] = total;
            }
        }
    }
}
