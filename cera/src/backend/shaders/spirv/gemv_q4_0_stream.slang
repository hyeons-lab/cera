// Wave32 Q4_0 decode GEMV over the resident stream layout:
//   y[m] = dequant(W[m, k]) @ x[k].
//
// Each 128-thread WG covers 32 output rows across 32 fibers (f = tid % 32,
// s = tid / 32) so every active lane in a Wave32 reads the exact same `x[xb4]`
// (scalar SGPR broadcast via [WaveSize(32)]) and consecutive fibers read
// contiguous 32-word (128-byte) weight cache lines.
//
// Bindings:
//   0 = q (u32), 1 = d (u32), 2 = x (float4),
//   3 = y (rw f32), 4 = params (uint4: m, k, unused, unused).
// Dispatch: gemv_row_workgroups(ceil(m / 32)).

[[vk::binding(0, 0)]] StructuredBuffer<uint>    q;
[[vk::binding(1, 0)]] StructuredBuffer<uint>    d;
[[vk::binding(2, 0)]] StructuredBuffer<float4>  x;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(4, 0)]] StructuredBuffer<uint4>   params;

static const uint ROWS_PER_WG = 32u;
static const uint FIBERS = 32u;
static const uint K_SPLITS = 4u;
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
    uint f = tid % FIBERS;
    uint s = tid / FIBERS;
    uint row = get_wid(wid) * ROWS_PER_WG + f;

    uint nb = k / 32u;
    uint nb_per_split = (nb + K_SPLITS - 1u) / K_SPLITS;
    uint b0 = s * nb_per_split;
    uint b1 = min(b0 + nb_per_split, nb);

    bool live = row < m;
    uint rowc = live ? row : m - 1u;

    float acc = 0.0f;

    for (uint b = b0; b < b1; b++) {
        uint xbase4 = b * 8u;
        uint di = (b >> 1u) * m;
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

    split_scratch[f * K_SPLITS + s] = acc;
    GroupMemoryBarrierWithGroupSync();

    if (tid < ROWS_PER_WG) {
        uint out_row = get_wid(wid) * ROWS_PER_WG + tid;
        if (out_row < m) {
            uint base = tid * K_SPLITS;
            y[out_row] = (split_scratch[base + 0u] + split_scratch[base + 1u])
                       + (split_scratch[base + 2u] + split_scratch[base + 3u]);
        }
    }
}
