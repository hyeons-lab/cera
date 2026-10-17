// Fused Q8_0 Gate+Up+SwiGLU GEMV over the resident stream layout:
//   y[m] = silu(dequant(W_gate[m, k]) @ x[k]) * (dequant(W_up[m, k]) @ x[k]).
//
// Each 64-thread workgroup covers 32 output rows x a 2-way K-split (32 fibers x
// 2 K-splits = 64 threads). Each thread computes 1 gate row and 1 up row for
// the same output index `row`, sharing the activation loads `x[xb..xb+7]` across
// both projections while keeping 8 independent accumulator chains per thread
// (4 p-lanes for gate + 4 p-lanes for up) and coalescing weight reads into full
// 32-word (128-byte) cache lines across consecutive fibers `f = tid % 32`.
//
// Bindings:
//   0 = gate_q (u32), 1 = gate_d (u32),
//   2 = up_q   (u32), 3 = up_d   (u32),
//   4 = x (f32), 5 = y (f32, read-write), 6 = params (uint4: m, k, unused, unused).
// Dispatch: gemv_row_workgroups(ceil(m / 32)).

[[vk::binding(0, 0)]] StructuredBuffer<uint>    gate_q;
[[vk::binding(1, 0)]] StructuredBuffer<uint>    gate_d;
[[vk::binding(2, 0)]] StructuredBuffer<uint>    up_q;
[[vk::binding(3, 0)]] StructuredBuffer<uint>    up_d;
[[vk::binding(4, 0)]] StructuredBuffer<float>   x;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(6, 0)]] StructuredBuffer<uint4>   params;

static const uint ROWS_PER_WG = 32u;
static const uint FIBERS = 32u;
static const uint K_SPLITS = 2u;
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
    uint pairs = nb / 2u;
    uint pairs_per_split = (pairs + K_SPLITS - 1u) / K_SPLITS;
    uint b0 = (s * pairs_per_split) * 2u;
    uint b1 = min(b0 + pairs_per_split * 2u, nb);

    bool live = row < m;
    uint rowc = live ? row : m - 1u;

    float dg0 = 0.0f;
    float dg1 = 0.0f;
    float dg2 = 0.0f;
    float dg3 = 0.0f;
    float du0 = 0.0f;
    float du1 = 0.0f;
    float du2 = 0.0f;
    float du3 = 0.0f;

    for (uint b = b0; b < b1; b++) {
        uint xbase = b * 32u;
        uint di = (b / 2u) * m;
        uint dshift = (b & 1u) * 16u;
        uint qi = b * 8u * m;
        float dsg = live ? float(f16tof32((gate_d[di + rowc] >> dshift) & 0xFFFFu)) : 0.0f;
        float dsu = live ? float(f16tof32((up_d[di + rowc] >> dshift) & 0xFFFFu)) : 0.0f;

        [unroll]
        for (uint p = 0u; p < 4u; p++) {
            uint xb = xbase + p * 8u;
            float a0 = x[xb];
            float a1 = x[xb + 1u];
            float a2 = x[xb + 2u];
            float a3 = x[xb + 3u];
            float a4 = x[xb + 4u];
            float a5 = x[xb + 5u];
            float a6 = x[xb + 6u];
            float a7 = x[xb + 7u];

            uint qbase_lo = qi + (p * 2u) * m + rowc;
            uint qbase_hi = qi + (p * 2u + 1u) * m + rowc;
            uint qg_lo = gate_q[qbase_lo];
            uint qg_hi = gate_q[qbase_hi];
            uint qu_lo = up_q[qbase_lo];
            uint qu_hi = up_q[qbase_hi];

            float dot_g = a0 * float(int(qg_lo << 24u) >> 24)
                        + a1 * float(int(qg_lo << 16u) >> 24)
                        + a2 * float(int(qg_lo << 8u)  >> 24)
                        + a3 * float(int(qg_lo)        >> 24)
                        + a4 * float(int(qg_hi << 24u) >> 24)
                        + a5 * float(int(qg_hi << 16u) >> 24)
                        + a6 * float(int(qg_hi << 8u)  >> 24)
                        + a7 * float(int(qg_hi)        >> 24);
            float dot_u = a0 * float(int(qu_lo << 24u) >> 24)
                        + a1 * float(int(qu_lo << 16u) >> 24)
                        + a2 * float(int(qu_lo << 8u)  >> 24)
                        + a3 * float(int(qu_lo)        >> 24)
                        + a4 * float(int(qu_hi << 24u) >> 24)
                        + a5 * float(int(qu_hi << 16u) >> 24)
                        + a6 * float(int(qu_hi << 8u)  >> 24)
                        + a7 * float(int(qu_hi)        >> 24);

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
    }

    split_scratch_g[f * K_SPLITS + s] = (dg0 + dg1) + (dg2 + dg3);
    split_scratch_u[f * K_SPLITS + s] = (du0 + du1) + (du2 + du3);
    GroupMemoryBarrierWithGroupSync();

    if (tid < ROWS_PER_WG) {
        uint r = tid;
        uint out_row = get_wid(wid) * ROWS_PER_WG + r;
        if (out_row < m) {
            float g = split_scratch_g[r * K_SPLITS] + split_scratch_g[r * K_SPLITS + 1u];
            float u = split_scratch_u[r * K_SPLITS] + split_scratch_u[r * K_SPLITS + 1u];
            float cg = clamp(g, -80.0f, 80.0f);
            y[out_row] = (cg / (1.0f + exp(-cg))) * u;
        }
    }
}
