// Fused Q4_0 Gate+Up+SwiGLU GEMV over the resident stream layout:
//   y[m] = silu(dequant(W_gate[m, k]) @ x[k]) * (dequant(W_up[m, k]) @ x[k]).
//
// Each 64-thread workgroup covers 16 output rows x a 4-way K-split (16 fibers x
// 4 K-splits = 64 threads). Each thread computes 1 gate row and 1 up row for
// the same output index `row`, sharing the activation loads, power-of-two nibble
// scaling (`y0..y7`), and zero-point sum (`sumy_b`) across both projections
// while coalescing weight reads into full 16-word (64-byte) cache lines across
// consecutive fibers `f = tid % 16`.
//
// Bindings:
//   0 = gate_q (u32), 1 = gate_d (u32),
//   2 = up_q   (u32), 3 = up_d   (u32),
//   4 = x (f32), 5 = y (f32, read-write), 6 = params (uint4: m, k, unused, unused).
// Dispatch: gemv_row_workgroups(ceil(m / 16)).

[[vk::binding(0, 0)]] StructuredBuffer<uint>    gate_q;
[[vk::binding(1, 0)]] StructuredBuffer<uint>    gate_d;
[[vk::binding(2, 0)]] StructuredBuffer<uint>    up_q;
[[vk::binding(3, 0)]] StructuredBuffer<uint>    up_d;
[[vk::binding(4, 0)]] StructuredBuffer<float>   x;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(6, 0)]] StructuredBuffer<uint4>   params;

static const uint ROWS_PER_WG = 16u;
static const uint FIBERS = 16u;
static const uint K_SPLITS = 4u;
static const uint WG_SIZE = 64u;

groupshared float split_scratch_g[ROWS_PER_WG * K_SPLITS];
groupshared float split_scratch_u[ROWS_PER_WG * K_SPLITS];

uint get_wid(uint3 wid) {
    return wid.x + wid.y * 65535u;
}

[shader("compute")]
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

    float dg0 = 0.0f;
    float dg1 = 0.0f;
    float dg2 = 0.0f;
    float dg3 = 0.0f;
    float sg = 0.0f;
    float du0 = 0.0f;
    float du1 = 0.0f;
    float du2 = 0.0f;
    float du3 = 0.0f;
    float su = 0.0f;

    for (uint b = b0; b < b1; b++) {
        uint xbase = b * 32u;
        uint di = (b / 2u) * m;
        uint dshift = (b & 1u) * 16u;
        uint qi = b * 4u * m;
        float dsg = live ? float(f16tof32((gate_d[di + rowc] >> dshift) & 0xFFFFu)) : 0.0f;
        float dsu = live ? float(f16tof32((up_d[di + rowc] >> dshift) & 0xFFFFu)) : 0.0f;
        float sumy_b = 0.0f;

        [unroll]
        for (uint p = 0u; p < 4u; p++) {
            uint xb = xbase + p * 8u;
            float a0 = x[xb];        float y0 = a0;
            float a1 = x[xb + 1u];   float y1 = a1 * 0.0625f;
            float a2 = x[xb + 2u];   float y2 = a2 * 0.00390625f;
            float a3 = x[xb + 3u];   float y3 = a3 * 0.000244140625f;
            float a4 = x[xb + 4u];   float y4 = a4 * 0.0000152587890625f;
            float a5 = x[xb + 5u];   float y5 = a5 * 9.5367431640625e-7f;
            float a6 = x[xb + 6u];   float y6 = a6 * 5.960464477539063e-8f;
            float a7 = x[xb + 7u];   float y7 = a7 * 3.725290298461914e-9f;
            sumy_b += (a0 + a1) + (a2 + a3) + (a4 + a5) + (a6 + a7);

            uint qgw = gate_q[qi + p * m + rowc];
            uint quw = up_q[qi + p * m + rowc];
            float dot_g = y0 * float(qgw & 0x0000000Fu) + y1 * float(qgw & 0x000000F0u)
                        + y2 * float(qgw & 0x00000F00u) + y3 * float(qgw & 0x0000F000u)
                        + y4 * float(qgw & 0x000F0000u) + y5 * float(qgw & 0x00F00000u)
                        + y6 * float(qgw & 0x0F000000u) + y7 * float(qgw & 0xF0000000u);
            float dot_u = y0 * float(quw & 0x0000000Fu) + y1 * float(quw & 0x000000F0u)
                        + y2 * float(quw & 0x00000F00u) + y3 * float(quw & 0x0000F000u)
                        + y4 * float(quw & 0x000F0000u) + y5 * float(quw & 0x00F00000u)
                        + y6 * float(quw & 0x0F000000u) + y7 * float(quw & 0xF0000000u);

            if (p == 0u) {
                dg0 += dsg * dot_g;
                du0 += dsu * dot_u;
            } else if (p == 1u) {
                dg1 += dsg * dot_g;
                du1 += dsu * dot_u;
            } else if (p == 2u) {
                dg2 += dsg * dot_g;
                du2 += dsu * dot_u;
            } else {
                dg3 += dsg * dot_g;
                du3 += dsu * dot_u;
            }
        }
        sg += dsg * sumy_b;
        su += dsu * sumy_b;
    }

    split_scratch_g[f * K_SPLITS + s] = (dg0 + dg1) + (dg2 + dg3) - 8.0f * sg;
    split_scratch_u[f * K_SPLITS + s] = (du0 + du1) + (du2 + du3) - 8.0f * su;
    GroupMemoryBarrierWithGroupSync();

    if (tid < ROWS_PER_WG) {
        uint r = tid;
        uint out_row = get_wid(wid) * ROWS_PER_WG + r;
        if (out_row < m) {
            float g = (split_scratch_g[r * K_SPLITS] + split_scratch_g[r * K_SPLITS + 1u])
                    + (split_scratch_g[r * K_SPLITS + 2u] + split_scratch_g[r * K_SPLITS + 3u]);
            float u = (split_scratch_u[r * K_SPLITS] + split_scratch_u[r * K_SPLITS + 1u])
                    + (split_scratch_u[r * K_SPLITS + 2u] + split_scratch_u[r * K_SPLITS + 3u]);
            float cg = clamp(g, -80.0f, 80.0f);
            y[out_row] = (cg / (1.0f + exp(-cg))) * u;
        }
    }
}
