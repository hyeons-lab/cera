// Q8_0 GEMV over the resident stream layout: y[m] = dequant(W[m, k]) @ x[k].
//
// The resident (q, d) repack is row-consecutive (see `repack_q8_0_stream`):
// `q[(b*8 + w)*m + row]` holds 4 signed int8 weights (`k = b*32 + w*4 + 0..3`)
// in little-endian byte order, and `d[(b/2)*m + row]` holds paired f16 scales
// (`half(scale of block 2p) | half(scale of block 2p+1) << 16`).
//
// Consecutive threads cover consecutive rows: 32 rows x a 4-way K-split in
// 64-thread workgroups, 2 rows per thread (thread tid covers rows
// (tid%16)+{0,16}, K-quarter tid/16). Together the 16 fibers' two row halves
// span 32 consecutive u32 words (128 bytes = one full Adreno DRAM cache line)
// per k-slice. Two rows per thread amortizes activation loads across both rows
// and exposes 8 independent accumulator chains per thread (2 rows x 4 p-lanes).
// Split partials combine through 128 floats of shared memory with a single barrier.
//
// Bindings: 0 = q (u32), 1 = d (u32), 2 = x (f32 as float4), 3 = y (f32, read-write),
// 4 = params (uint4: m, k, unused, unused). Dispatch:
// gemv_row_workgroups(ceil(m / 32)). Constraints (asserted host-side):
// k % 64 == 0 (the resident Q8_0 stream layout precondition).
//
// SPIR-V-only (passthrough-gated like `gemm_stream_q8_0_k64`): the resident
// layout only exists where passthrough does, so there is no WGSL twin.

[[vk::binding(0, 0)]] StructuredBuffer<uint>    q;
[[vk::binding(1, 0)]] StructuredBuffer<uint>    d;
[[vk::binding(2, 0)]] StructuredBuffer<float4>  x;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(4, 0)]] StructuredBuffer<uint4>   params;

static const uint ROWS_PER_WG = 32u;
static const uint ROWS_PER_THREAD = 2u;
static const uint FIBERS = ROWS_PER_WG / ROWS_PER_THREAD;
static const uint K_SPLITS = 4u;
static const uint WG_SIZE = 64u;

groupshared float split_scratch[ROWS_PER_WG * K_SPLITS];

uint get_wid(uint3 wid) {
    return wid.x + wid.y * 65535u;
}

int sbyte0(uint w) { return int(w << 24u) >> 24; }
int sbyte1(uint w) { return int(w << 16u) >> 24; }
int sbyte2(uint w) { return int(w << 8u)  >> 24; }
int sbyte3(uint w) { return int(w)        >> 24; }

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
    uint row0 = get_wid(wid) * ROWS_PER_WG + f;
    uint row1 = row0 + FIBERS;

    uint nb = k / 32u;
    uint pairs = nb / 2u;
    uint pairs_per_split = (pairs + K_SPLITS - 1u) / K_SPLITS;
    uint b0 = (s * pairs_per_split) * 2u;
    uint b1 = min(b0 + pairs_per_split * 2u, nb);

    bool live0 = row0 < m;
    bool live1 = row1 < m;
    uint row0c = live0 ? row0 : m - 1u;
    uint row1c = live1 ? row1 : m - 1u;

    float d00 = 0.0f;
    float d01 = 0.0f;
    float d02 = 0.0f;
    float d03 = 0.0f;
    float d10 = 0.0f;
    float d11 = 0.0f;
    float d12 = 0.0f;
    float d13 = 0.0f;

    for (uint b = b0; b < b1; b++) {
        uint xbase4 = b * 8u;
        uint di = (b / 2u) * m;
        uint dshift = (b & 1u) * 16u;
        uint qi = b * 8u * m;
        float ds0 = live0 ? float(f16tof32((d[di + row0c] >> dshift) & 0xFFFFu)) : 0.0f;
        float ds1 = live1 ? float(f16tof32((d[di + row1c] >> dshift) & 0xFFFFu)) : 0.0f;

        [unroll]
        for (uint p = 0u; p < 4u; p++) {
            uint xb4 = xbase4 + p * 2u;
            float4 a03 = x[xb4];
            float4 a47 = x[xb4 + 1u];
            float a0 = a03.x;
            float a1 = a03.y;
            float a2 = a03.z;
            float a3 = a03.w;
            float a4 = a47.x;
            float a5 = a47.y;
            float a6 = a47.z;
            float a7 = a47.w;

            uint qbase_lo = qi + (p * 2u) * m;
            uint qbase_hi = qbase_lo + m;
            uint q0_lo = q[qbase_lo + row0c];
            uint q0_hi = q[qbase_hi + row0c];
            uint q1_lo = q[qbase_lo + row1c];
            uint q1_hi = q[qbase_hi + row1c];

            float dot0 = a0 * float(sbyte0(q0_lo)) + a1 * float(sbyte1(q0_lo))
                       + a2 * float(sbyte2(q0_lo)) + a3 * float(sbyte3(q0_lo))
                       + a4 * float(sbyte0(q0_hi)) + a5 * float(sbyte1(q0_hi))
                       + a6 * float(sbyte2(q0_hi)) + a7 * float(sbyte3(q0_hi));
            float dot1 = a0 * float(sbyte0(q1_lo)) + a1 * float(sbyte1(q1_lo))
                       + a2 * float(sbyte2(q1_lo)) + a3 * float(sbyte3(q1_lo))
                       + a4 * float(sbyte0(q1_hi)) + a5 * float(sbyte1(q1_hi))
                       + a6 * float(sbyte2(q1_hi)) + a7 * float(sbyte3(q1_hi));

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
    }

    split_scratch[f * K_SPLITS + s] = (d00 + d01) + (d02 + d03);
    split_scratch[(f + FIBERS) * K_SPLITS + s] = (d10 + d11) + (d12 + d13);
    GroupMemoryBarrierWithGroupSync();

    if (tid < ROWS_PER_WG) {
        uint r = tid;
        uint row = get_wid(wid) * ROWS_PER_WG + r;
        if (row < m) {
            float t = (split_scratch[r * K_SPLITS] + split_scratch[r * K_SPLITS + 1u])
                    + (split_scratch[r * K_SPLITS + 2u] + split_scratch[r * K_SPLITS + 3u]);
            y[row] = t;
        }
    }
}
