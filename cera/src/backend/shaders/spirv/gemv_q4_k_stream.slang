// Q4_K (Q4_K_M) GEMV over the resident stream layout: y[m] = dequant(W[m, k]) @ x[k].
//
// The resident (q, d) repack is row-consecutive: `q[qi*m + row]` and
// `d[b*m + row]` run over consecutive rows. Consecutive threads cover
// consecutive rows: 16 rows x an 8-way K-split in 64-thread workgroups,
// 2 rows per thread (thread tid covers rows 2*(tid%8)+{0,1}, K-eighth
// tid/8). Partials combine through 128 floats of shmem with a single barrier.
//
// The nibble extracts use the reciprocal-mult trick (pre-scaling x by exact
// powers of two 1/16^s) so the row loop applies unshifted masks instead of
// shifts. The sub-block min folds through a per-block activation sum:
// row = ds*dots - mn*sumy_b.
//
// Bindings:
//   0 = q (u32, 4 words per 32-block per row)
//   1 = d (u32, 1 word per 32-block per row, f16 scale | f16 min << 16)
//   2 = x (f32)
//   3 = y (f32, read-write)
//   4 = params (uint4: m, k, unused, unused)
// Dispatch: gemv_row_workgroups(ceil(m / 16)).

[[vk::binding(0, 0)]] StructuredBuffer<uint>    q;
[[vk::binding(1, 0)]] StructuredBuffer<uint>    d;
[[vk::binding(2, 0)]] StructuredBuffer<float>   x;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(4, 0)]] StructuredBuffer<uint4>   params;

static const uint ROWS_PER_WG = 16u;
static const uint ROWS_PER_THREAD = 2u;
static const uint FIBERS = ROWS_PER_WG / ROWS_PER_THREAD;
static const uint K_SPLITS = 8u;
static const uint WG_SIZE = 64u;

groupshared float split_scratch[ROWS_PER_WG * K_SPLITS];

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
    uint row0 = get_wid(wid) * ROWS_PER_WG + f * ROWS_PER_THREAD;
    uint row1 = row0 + 1u;

    uint nb = k / 32u;
    uint nb_per_split = (nb + K_SPLITS - 1u) / K_SPLITS;
    uint b0 = s * nb_per_split;
    uint b1 = min(b0 + nb_per_split, nb);

    bool live0 = row0 < m;
    bool live1 = row1 < m;
    uint row0c = live0 ? row0 : m - 1u;
    uint row1c = live1 ? row1 : m - 1u;

    float d00 = 0.0f;
    float d01 = 0.0f;
    float d02 = 0.0f;
    float d03 = 0.0f;
    float m0 = 0.0f;

    float d10 = 0.0f;
    float d11 = 0.0f;
    float d12 = 0.0f;
    float d13 = 0.0f;
    float m1 = 0.0f;

    for (uint b = b0; b < b1; b++) {
        uint xbase = b * 32u;
        uint qi = b * 4u * m;

        uint pd0 = live0 ? d[b * m + row0c] : 0u;
        float ds0 = f16tof32(pd0 & 0xFFFFu);
        float mn0 = f16tof32(pd0 >> 16u);

        uint pd1 = live1 ? d[b * m + row1c] : 0u;
        float ds1 = f16tof32(pd1 & 0xFFFFu);
        float mn1 = f16tof32(pd1 >> 16u);

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

            uint q0w = q[qi + p * m + row0c];
            uint q1w = q[qi + p * m + row1c];
            float dot0 = y0 * float(q0w & 0x0000000Fu) + y1 * float(q0w & 0x000000F0u)
                       + y2 * float(q0w & 0x00000F00u) + y3 * float(q0w & 0x0000F000u)
                       + y4 * float(q0w & 0x000F0000u) + y5 * float(q0w & 0x00F00000u)
                       + y6 * float(q0w & 0x0F000000u) + y7 * float(q0w & 0xF0000000u);
            float dot1 = y0 * float(q1w & 0x0000000Fu) + y1 * float(q1w & 0x000000F0u)
                       + y2 * float(q1w & 0x00000F00u) + y3 * float(q1w & 0x0000F000u)
                       + y4 * float(q1w & 0x000F0000u) + y5 * float(q1w & 0x00F00000u)
                       + y6 * float(q1w & 0x0F000000u) + y7 * float(q1w & 0xF0000000u);

            if (p == 0u) {
                d00 += ds0 * dot0;
                d10 += ds1 * dot1;
            } else if (p == 1u) {
                d01 += ds0 * dot0;
                d11 += ds1 * dot1;
            } else if (p == 2u) {
                d02 += ds0 * dot0;
                d12 += ds1 * dot1;
            } else {
                d03 += ds0 * dot0;
                d13 += ds1 * dot1;
            }
        }
        m0 += mn0 * sumy_b;
        m1 += mn1 * sumy_b;
    }

    uint r0 = f * ROWS_PER_THREAD;
    split_scratch[r0 * K_SPLITS + s] = (d00 + d01) + (d02 + d03) - m0;
    split_scratch[(r0 + 1u) * K_SPLITS + s] = (d10 + d11) + (d12 + d13) - m1;
    GroupMemoryBarrierWithGroupSync();

    if (tid < ROWS_PER_WG) {
        uint r = tid;
        uint row = get_wid(wid) * ROWS_PER_WG + r;
        if (row < m) {
            float t = (split_scratch[r * K_SPLITS] + split_scratch[r * K_SPLITS + 1u])
                    + (split_scratch[r * K_SPLITS + 2u] + split_scratch[r * K_SPLITS + 3u])
                    + (split_scratch[r * K_SPLITS + 4u] + split_scratch[r * K_SPLITS + 5u])
                    + (split_scratch[r * K_SPLITS + 6u] + split_scratch[r * K_SPLITS + 7u]);
            y[row] = t;
        }
    }
}
