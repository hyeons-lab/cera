// Fused Q4_0 Gate+Up+SwiGLU GEMV over the resident stream layout:
//   y[m] = silu(dequant(W_gate[m, k]) @ x[k]) * (dequant(W_up[m, k]) @ x[k]).
//
// Each 128-thread workgroup covers 32 output rows x a 4-way K-split (32 fibers x
// 4 K-splits = 128 threads, 4 wave32s). Each Wave32 processes the SAME K-block across
// 32 consecutive rows so activation loads/scaling (`a0..a7`, `y0..y7`, `sumy_b`) are
// 100% wave-uniform (scalar-cached) while weight reads coalesce into full 128-byte
// transactions across all 32 lanes.
//
// Bindings:
//   0 = gate_q (u32), 1 = gate_d (u32),
//   2 = up_q   (u32), 3 = up_d   (u32),
//   4 = x (float4), 5 = y (f32, read-write), 6 = params (uint4: m, k, unused, unused).
// Dispatch: gemv_row_workgroups(ceil(m / 32)).

[[vk::binding(0, 0)]] StructuredBuffer<uint>    gate_q;
[[vk::binding(1, 0)]] StructuredBuffer<uint>    gate_d;
[[vk::binding(2, 0)]] StructuredBuffer<uint>    up_q;
[[vk::binding(3, 0)]] StructuredBuffer<uint>    up_d;
[[vk::binding(4, 0)]] StructuredBuffer<float4>  x;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(6, 0)]] StructuredBuffer<uint4>   params;

static const uint ROWS_PER_WG = 32u;
static const uint FIBERS = 32u;
static const uint K_SPLITS = 4u;
static const uint WG_SIZE = 128u;

groupshared float split_scratch_g[ROWS_PER_WG * K_SPLITS];
groupshared float split_scratch_u[ROWS_PER_WG * K_SPLITS];

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
    uint row = get_wid(wid) * ROWS_PER_WG + f;

    uint nb = k / 32u;
    uint nb_per_split = (nb + K_SPLITS - 1u) / K_SPLITS;
    uint b0 = s * nb_per_split;
    uint b1 = min(b0 + nb_per_split, nb);

    bool live = row < m;
    uint rowc = live ? row : m - 1u;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (uint b = b0; b < b1; b++) {
        uint xbase4 = b * 8u;
        uint di = (b >> 1u) * m;
        uint dshift = (b & 1u) * 16u;
        uint qi = b * 4u * m;
        float dsg = live ? float(f16tof32((gate_d[di + rowc] >> dshift) & 0xFFFFu)) : 0.0f;
        float dsu = live ? float(f16tof32((up_d[di + rowc] >> dshift) & 0xFFFFu)) : 0.0f;
        float bg0 = 0.0f; float bg1 = 0.0f; float bg2 = 0.0f; float bg3 = 0.0f;
        float bu0 = 0.0f; float bu1 = 0.0f; float bu2 = 0.0f; float bu3 = 0.0f;
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

            uint qgw = gate_q[qi + p * m + rowc];
            uint quw = up_q[qi + p * m + rowc];
            bg0 += y0 * float(qgw & 0x0000000Fu) + y4 * float(qgw & 0x000F0000u);
            bg1 += y1 * float(qgw & 0x000000F0u) + y5 * float(qgw & 0x00F00000u);
            bg2 += y2 * float(qgw & 0x00000F00u) + y6 * float(qgw & 0x0F000000u);
            bg3 += y3 * float(qgw & 0x0000F000u) + y7 * float(qgw & 0xF0000000u);
            bu0 += y0 * float(quw & 0x0000000Fu) + y4 * float(quw & 0x000F0000u);
            bu1 += y1 * float(quw & 0x000000F0u) + y5 * float(quw & 0x00F00000u);
            bu2 += y2 * float(quw & 0x00000F00u) + y6 * float(quw & 0x0F000000u);
            bu3 += y3 * float(quw & 0x0000F000u) + y7 * float(quw & 0xF0000000u);
        }
        acc_g += dsg * (((bg0 + bg1) + (bg2 + bg3)) - 8.0f * sumy_b);
        acc_u += dsu * (((bu0 + bu1) + (bu2 + bu3)) - 8.0f * sumy_b);
    }

    split_scratch_g[f * K_SPLITS + s] = acc_g;
    split_scratch_u[f * K_SPLITS + s] = acc_u;
    GroupMemoryBarrierWithGroupSync();

    if (tid < ROWS_PER_WG) {
        uint out_row = get_wid(wid) * ROWS_PER_WG + tid;
        if (out_row < m) {
            uint base = tid * K_SPLITS;
            float g = (split_scratch_g[base + 0u] + split_scratch_g[base + 1u])
                    + (split_scratch_g[base + 2u] + split_scratch_g[base + 3u]);
            float u = (split_scratch_u[base + 0u] + split_scratch_u[base + 1u])
                    + (split_scratch_u[base + 2u] + split_scratch_u[base + 3u]);
            float cg = clamp(g, -80.0f, 80.0f);
            y[out_row] = (cg / (1.0f + exp(-cg))) * u;
        }
    }
}
