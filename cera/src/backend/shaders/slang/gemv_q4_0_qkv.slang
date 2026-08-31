// slang-entries: gemv_q4_0_qkv
//
// Fused 3-in-1 Q/K/V Q4_0 Matrix-Vector Multiply (GEMV) in Slang for both Metal and WGSL.
// Computes Q (M_q x K), K (M_kv x K), and V (M_kv x K) projections in a single dispatch pass,
// reusing cooperatively staged activation inputs in workgroup shared memory.
// Tiled across NR=4 output rows per workgroup.

[[vk::binding(0)]] StructuredBuffer<uint>    w_q    : register(t0);
[[vk::binding(1)]] StructuredBuffer<uint>    w_k    : register(t1);
[[vk::binding(2)]] StructuredBuffer<uint>    w_v    : register(t2);
[[vk::binding(3)]] StructuredBuffer<float>   x      : register(t3);
[[vk::binding(4)]] RWStructuredBuffer<float> y_q    : register(u4);
[[vk::binding(5)]] RWStructuredBuffer<float> y_k    : register(u5);
[[vk::binding(6)]] RWStructuredBuffer<float> y_v    : register(u6);
[[vk::binding(7)]] StructuredBuffer<uint4>   params : register(t7);

static const uint NR = 4u;
static const uint NQ = 16u;
static const uint WG_SIZE = 32u;
static const uint BLOCK_BYTES = 18u;
static const uint CHUNK_BLOCKS = 16u;
static const uint CHUNK_FLOATS = CHUNK_BLOCKS * 32u; // 512 floats (2 KB)

groupshared float partials[NR * WG_SIZE];
groupshared float x_stage[CHUNK_FLOATS];

uint get_wid(uint3 wid) {
    return wid.x + wid.y * 65535u;
}

float block_scale_q(uint blk_byte) {
    uint word = w_q[blk_byte >> 2u];
    uint scale_bits = ((blk_byte & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
    return float(f16tof32(scale_bits));
}

float block_scale_k(uint blk_byte) {
    uint word = w_k[blk_byte >> 2u];
    uint scale_bits = ((blk_byte & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
    return float(f16tof32(scale_bits));
}

float block_scale_v(uint blk_byte) {
    uint word = w_v[blk_byte >> 2u];
    uint scale_bits = ((blk_byte & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
    return float(f16tof32(scale_bits));
}

uint q_pair_q(uint qs_byte) {
    uint word = w_q[qs_byte >> 2u];
    return ((qs_byte & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
}

uint q_pair_k(uint qs_byte) {
    uint word = w_k[qs_byte >> 2u];
    return ((qs_byte & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
}

uint q_pair_v(uint qs_byte) {
    uint word = w_v[qs_byte >> 2u];
    return ((qs_byte & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void gemv_q4_0_qkv(
    uint3 lid : SV_GroupThreadID,
    uint3 wid : SV_GroupID
) {
    uint m_q = params[0].x;
    uint m_kv = params[0].y;
    uint k = params[0].z;
    uint nb = k / 32u;
    uint row_bytes = nb * BLOCK_BYTES;
    uint global_r0 = get_wid(wid) * NR;
    uint tid = lid.x;

    uint ix = tid / 2u;
    uint il = (tid & 1u) * 8u;

    uint which_matrix; // 0 = Q, 1 = K, 2 = V
    uint r0;
    uint m_cur;
    if (global_r0 < m_q) {
        which_matrix = 0u;
        r0 = global_r0;
        m_cur = m_q;
    } else if (global_r0 < m_q + m_kv) {
        which_matrix = 1u;
        r0 = global_r0 - m_q;
        m_cur = m_kv;
    } else {
        which_matrix = 2u;
        r0 = global_r0 - m_q - m_kv;
        m_cur = m_kv;
    }

    float sumf[NR];
    [unroll]
    for (uint r = 0u; r < NR; r++) {
        sumf[r] = 0.0f;
    }

    for (uint chunk_b = 0u; chunk_b < nb; chunk_b += CHUNK_BLOCKS) {
        uint chunk_k_start = chunk_b * 32u;
        uint chunk_k_len = min(CHUNK_FLOATS, k - chunk_k_start);
        uint chunk_k_vec4 = chunk_k_len / 4u;
        for (uint i = tid; i < chunk_k_vec4; i += WG_SIZE) {
            uint base_idx = chunk_k_start + i * 4u;
            x_stage[i * 4u + 0u] = x[base_idx + 0u];
            x_stage[i * 4u + 1u] = x[base_idx + 1u];
            x_stage[i * 4u + 2u] = x[base_idx + 2u];
            x_stage[i * 4u + 3u] = x[base_idx + 3u];
        }
        GroupMemoryBarrierWithGroupSync();

        uint ib_local = ix;
        while (ib_local < CHUNK_BLOCKS && (chunk_b + ib_local) < nb) {
            uint ib = chunk_b + ib_local;
            uint yb_stage_off = ib_local * 32u + il;

            float a0  = x_stage[yb_stage_off + 0u];
            float a1  = x_stage[yb_stage_off + 1u];
            float a2  = x_stage[yb_stage_off + 2u];
            float a3  = x_stage[yb_stage_off + 3u];
            float a4  = x_stage[yb_stage_off + 4u];
            float a5  = x_stage[yb_stage_off + 5u];
            float a6  = x_stage[yb_stage_off + 6u];
            float a7  = x_stage[yb_stage_off + 7u];
            float a8  = x_stage[yb_stage_off + 16u];
            float a9  = x_stage[yb_stage_off + 17u];
            float a10 = x_stage[yb_stage_off + 18u];
            float a11 = x_stage[yb_stage_off + 19u];
            float a12 = x_stage[yb_stage_off + 20u];
            float a13 = x_stage[yb_stage_off + 21u];
            float a14 = x_stage[yb_stage_off + 22u];
            float a15 = x_stage[yb_stage_off + 23u];

            float y0 = a0;
            float y1 = a1 / 256.0f;
            float y2 = a2;
            float y3 = a3 / 256.0f;
            float y4 = a4;
            float y5 = a5 / 256.0f;
            float y6 = a6;
            float y7 = a7 / 256.0f;
            float y8 = a8 / 16.0f;
            float y9 = a9 / 4096.0f;
            float y10 = a10 / 16.0f;
            float y11 = a11 / 4096.0f;
            float y12 = a12 / 16.0f;
            float y13 = a13 / 4096.0f;
            float y14 = a14 / 16.0f;
            float y15 = a15 / 4096.0f;

            float sumy0 = (a0 + a1) + (a2 + a3) + (a4 + a5) + (a6 + a7);
            float sumy1 = (a8 + a9) + (a10 + a11) + (a12 + a13) + (a14 + a15);
            float sumy_total = sumy0 + sumy1;

            if (which_matrix == 0u) {
                [unroll]
                for (uint r = 0u; r < NR; r++) {
                    if (r0 + r >= m_cur) { continue; }
                    uint blk_byte = (r0 + r) * row_bytes + ib * BLOCK_BYTES;
                    float d = block_scale_q(blk_byte);
                    uint qs_byte = blk_byte + 2u + il;

                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    float acc2 = 0.0f;
                    float acc3 = 0.0f;

                    uint q0 = q_pair_q(qs_byte + 0u);
                    acc0 += y0 * float(q0 & 0x000Fu);
                    acc1 += y1 * float(q0 & 0x0F00u);
                    acc2 += y8 * float(q0 & 0x00F0u);
                    acc3 += y9 * float(q0 & 0xF000u);

                    uint q1 = q_pair_q(qs_byte + 2u);
                    acc0 += y2 * float(q1 & 0x000Fu);
                    acc1 += y3 * float(q1 & 0x0F00u);
                    acc2 += y10 * float(q1 & 0x00F0u);
                    acc3 += y11 * float(q1 & 0xF000u);

                    uint q2 = q_pair_q(qs_byte + 4u);
                    acc0 += y4 * float(q2 & 0x000Fu);
                    acc1 += y5 * float(q2 & 0x0F00u);
                    acc2 += y12 * float(q2 & 0x00F0u);
                    acc3 += y13 * float(q2 & 0xF000u);

                    uint q3 = q_pair_q(qs_byte + 6u);
                    acc0 += y6 * float(q3 & 0x000Fu);
                    acc1 += y7 * float(q3 & 0x0F00u);
                    acc2 += y14 * float(q3 & 0x00F0u);
                    acc3 += y15 * float(q3 & 0xF000u);

                    sumf[r] += d * (sumy_total * -8.0f + acc0 + acc1 + acc2 + acc3);
                }
            } else if (which_matrix == 1u) {
                [unroll]
                for (uint r = 0u; r < NR; r++) {
                    if (r0 + r >= m_cur) { continue; }
                    uint blk_byte = (r0 + r) * row_bytes + ib * BLOCK_BYTES;
                    float d = block_scale_k(blk_byte);
                    uint qs_byte = blk_byte + 2u + il;

                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    float acc2 = 0.0f;
                    float acc3 = 0.0f;

                    uint q0 = q_pair_k(qs_byte + 0u);
                    acc0 += y0 * float(q0 & 0x000Fu);
                    acc1 += y1 * float(q0 & 0x0F00u);
                    acc2 += y8 * float(q0 & 0x00F0u);
                    acc3 += y9 * float(q0 & 0xF000u);

                    uint q1 = q_pair_k(qs_byte + 2u);
                    acc0 += y2 * float(q1 & 0x000Fu);
                    acc1 += y3 * float(q1 & 0x0F00u);
                    acc2 += y10 * float(q1 & 0x00F0u);
                    acc3 += y11 * float(q1 & 0xF000u);

                    uint q2 = q_pair_k(qs_byte + 4u);
                    acc0 += y4 * float(q2 & 0x000Fu);
                    acc1 += y5 * float(q2 & 0x0F00u);
                    acc2 += y12 * float(q2 & 0x00F0u);
                    acc3 += y13 * float(q2 & 0xF000u);

                    uint q3 = q_pair_k(qs_byte + 6u);
                    acc0 += y6 * float(q3 & 0x000Fu);
                    acc1 += y7 * float(q3 & 0x0F00u);
                    acc2 += y14 * float(q3 & 0x00F0u);
                    acc3 += y15 * float(q3 & 0xF000u);

                    sumf[r] += d * (sumy_total * -8.0f + acc0 + acc1 + acc2 + acc3);
                }
            } else {
                [unroll]
                for (uint r = 0u; r < NR; r++) {
                    if (r0 + r >= m_cur) { continue; }
                    uint blk_byte = (r0 + r) * row_bytes + ib * BLOCK_BYTES;
                    float d = block_scale_v(blk_byte);
                    uint qs_byte = blk_byte + 2u + il;

                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    float acc2 = 0.0f;
                    float acc3 = 0.0f;

                    uint q0 = q_pair_v(qs_byte + 0u);
                    acc0 += y0 * float(q0 & 0x000Fu);
                    acc1 += y1 * float(q0 & 0x0F00u);
                    acc2 += y8 * float(q0 & 0x00F0u);
                    acc3 += y9 * float(q0 & 0xF000u);

                    uint q1 = q_pair_v(qs_byte + 2u);
                    acc0 += y2 * float(q1 & 0x000Fu);
                    acc1 += y3 * float(q1 & 0x0F00u);
                    acc2 += y10 * float(q1 & 0x00F0u);
                    acc3 += y11 * float(q1 & 0xF000u);

                    uint q2 = q_pair_v(qs_byte + 4u);
                    acc0 += y4 * float(q2 & 0x000Fu);
                    acc1 += y5 * float(q2 & 0x0F00u);
                    acc2 += y12 * float(q2 & 0x00F0u);
                    acc3 += y13 * float(q2 & 0xF000u);

                    uint q3 = q_pair_v(qs_byte + 6u);
                    acc0 += y6 * float(q3 & 0x000Fu);
                    acc1 += y7 * float(q3 & 0x0F00u);
                    acc2 += y14 * float(q3 & 0x00F0u);
                    acc3 += y15 * float(q3 & 0xF000u);

                    sumf[r] += d * (sumy_total * -8.0f + acc0 + acc1 + acc2 + acc3);
                }
            }

            ib_local += NQ;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (uint r = 0u; r < NR; r++) {
        partials[r * WG_SIZE + tid] = sumf[r];
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = WG_SIZE / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            [unroll]
            for (uint r = 0u; r < NR; r++) {
                uint idx = r * WG_SIZE + tid;
                partials[idx] += partials[idx + stride];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u) {
        [unroll]
        for (uint r = 0u; r < NR; r++) {
            if (r0 + r < m_cur) {
                if (which_matrix == 0u) {
                    y_q[r0 + r] = partials[r * WG_SIZE];
                } else if (which_matrix == 1u) {
                    y_k[r0 + r] = partials[r * WG_SIZE];
                } else {
                    y_v[r0 + r] = partials[r * WG_SIZE];
                }
            }
        }
    }
}
