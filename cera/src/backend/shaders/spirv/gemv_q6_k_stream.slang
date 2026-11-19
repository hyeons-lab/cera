// Q6_K GEMV over the resident stream layout: y[m] = dequant(W[m, k]) @ x[k].
//
// The resident (q, d) repack is row-consecutive (see `repack_q6_k_stream`):
// `q[(b*8 + w)*m + row]` holds 4 signed int8 weights (`(q6 - 32)` for
// `k = b*32 + w*4 + 0..3`) in little-endian byte order, and
// `d[b*m + row]` holds two f16 pre-multiplied scales
// (`half(d * sc[2*sub]) | half(d * sc[2*sub + 1]) << 16`).
//
// Consecutive threads cover consecutive rows: 16 rows x a 2-way K-split in
// 32-thread workgroups (one wave32), 1 row per thread (thread tid covers row
// tid%16, K-half tid/16) for full 64-byte coalescing on q/d and barrier-free
// wave reduction when subgroup width >= 32.
//
// Bindings: 0 = q (u32), 1 = d (u32), 2 = x (f32), 3 = y (f32, read-write),
// 4 = params (uint4: m, k, unused, unused). Dispatch:
// gemv_row_workgroups(ceil(m / 16)). Constraints: k % 256 == 0.

[[vk::binding(0, 0)]] StructuredBuffer<uint>    q;
[[vk::binding(1, 0)]] StructuredBuffer<uint>    d;
[[vk::binding(2, 0)]] StructuredBuffer<float4>  x;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(4, 0)]] StructuredBuffer<uint4>   params;

static const uint ROWS_PER_WG = 32u;
static const uint WG_SIZE = 32u;

uint get_wid(uint3 wid) {
    return wid.x + wid.y * 65535u;
}

int sbyte0(uint w) { return int(w << 24u) >> 24; }
int sbyte1(uint w) { return int(w << 16u) >> 24; }
int sbyte2(uint w) { return int(w << 8u)  >> 24; }
int sbyte3(uint w) { return int(w)        >> 24; }

[shader("compute")]
[WaveSize(32)]
[numthreads(WG_SIZE, 1, 1)]
void main(
    uint3 lid : SV_GroupThreadID,
    uint3 wid : SV_GroupID
) {
    uint m = params[0].x;
    uint k = params[0].y;
    uint row = get_wid(wid) * ROWS_PER_WG + lid.x;
    if (row >= m) {
        return;
    }

    uint nb = k / 32u;
    float acc = 0.0f;

    for (uint b = 0u; b < nb; b++) {
        uint xbase4 = b * 8u;
        uint di = b * m;
        uint qi = b * 8u * m;
        uint dp = d[di + row];
        float sc_0 = float(f16tof32(dp & 0xFFFFu));
        float sc_1 = float(f16tof32(dp >> 16u));

        float sub0 = 0.0f;
        float sub1 = 0.0f;

        [unroll]
        for (uint p = 0u; p < 4u; p++) {
            uint xb4 = xbase4 + p * 2u;
            float4 a03 = x[xb4];
            float4 a47 = x[xb4 + 1u];

            uint qbase_lo = qi + (p * 2u) * m;
            uint qbase_hi = qbase_lo + m;
            uint q_lo = q[qbase_lo + row];
            uint q_hi = q[qbase_hi + row];

            float dot0 = a03.x * float(sbyte0(q_lo)) + a03.y * float(sbyte1(q_lo))
                       + a03.z * float(sbyte2(q_lo)) + a03.w * float(sbyte3(q_lo))
                       + a47.x * float(sbyte0(q_hi)) + a47.y * float(sbyte1(q_hi))
                       + a47.z * float(sbyte2(q_hi)) + a47.w * float(sbyte3(q_hi));

            if (p < 2u) {
                sub0 += dot0;
            } else {
                sub1 += dot0;
            }
        }
        acc += sc_0 * sub0 + sc_1 * sub1;
    }

    y[row] = acc;
}
