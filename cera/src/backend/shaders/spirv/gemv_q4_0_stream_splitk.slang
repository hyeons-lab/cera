// Split-K Wave32 Q4_0 decode GEMV over the resident stream layout (m <= 2048).
// Zero LDS, zero barriers: each WG is a single Wave32 owning 32 contiguous rows
// (f = tid = 0..31) and one K slice (wid.y = 0..k_splits-1), writing partials
// to partial_y[wid.y * m + row].
//
// Bindings:
//   0 = q (u32), 1 = d (u32), 2 = x (float4),
//   3 = partial_y (rw f32), 4 = params (uint4: m, k, k_splits, unused).
// Dispatch: (ceil(m / 32), k_splits, 1).

[[vk::binding(0, 0)]] StructuredBuffer<uint>    q;
[[vk::binding(1, 0)]] StructuredBuffer<uint>    d;
[[vk::binding(2, 0)]] StructuredBuffer<float4>  x;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> partial_y;
[[vk::binding(4, 0)]] StructuredBuffer<uint4>   params;

[shader("compute")]
[WaveSize(32)]
[numthreads(32, 1, 1)]
void main(
    uint3 lid : SV_GroupThreadID,
    uint3 wid : SV_GroupID
) {
    uint m = params[0].x;
    uint k = params[0].y;
    uint k_splits = params[0].z;

    uint f = lid.x;
    uint s = wid.y;
    uint row = wid.x * 32u + f;
    uint nb = k / 32u;
    uint pairs = nb / 2u;
    uint pairs_per_split = (pairs + k_splits - 1u) / k_splits;
    uint b0 = (s * pairs_per_split) * 2u;
    uint b1 = min(b0 + pairs_per_split * 2u, nb);

    bool live = row < m;
    uint rowc = live ? row : m - 1u;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    float sumy = 0.0f;

    for (uint b = b0; b < b1; b++) {
        uint xbase4 = b * 8u;
        uint di = (b >> 1u) * m;
        uint dshift = (b & 1u) * 16u;
        uint qi = b * 4u * m;
        float ds = live ? float(f16tof32((d[di + rowc] >> dshift) & 0xFFFFu)) : 0.0f;
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
            float dot0 = y0 * float(qw & 0x0000000Fu) + y1 * float(qw & 0x000000F0u)
                       + y2 * float(qw & 0x00000F00u) + y3 * float(qw & 0x0000F000u)
                       + y4 * float(qw & 0x000F0000u) + y5 * float(qw & 0x00F00000u)
                       + y6 * float(qw & 0x0F000000u) + y7 * float(qw & 0xF0000000u);
            if (p == 0u) {
                d0 += ds * dot0;
            } else if (p == 1u) {
                d1 += ds * dot0;
            } else if (p == 2u) {
                d2 += ds * dot0;
            } else {
                d3 += ds * dot0;
            }
        }
        sumy += ds * sumy_b;
    }

    if (live) {
        partial_y[s * m + row] = (d0 + d1) + (d2 + d3) - 8.0f * sumy;
    }
}
