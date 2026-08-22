// slang-entries: gemv_q4_0_fast
//
// Fast Q4_0 Matrix-Vector Multiply (GEMV) in Slang for both Metal and WGSL.
// Tiled across NR=8 output rows per workgroup with block-chunked cooperative
// workgroup shared memory preloading of activations x for arbitrary K.

[[vk::binding(0)]] StructuredBuffer<uint>    w      : register(t0);
[[vk::binding(1)]] StructuredBuffer<float>   x      : register(t1);
[[vk::binding(2)]] RWStructuredBuffer<float> y      : register(u2);
[[vk::binding(3)]] StructuredBuffer<uint4>   params : register(t3);

static const uint NR = 8u;
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

float block_scale(uint blk_byte) {
    uint word_off = blk_byte / 4u;
    uint byte_rem = blk_byte % 4u;
    uint scale_bits;
    if (byte_rem == 0u) {
        scale_bits = w[word_off] & 0xFFFFu;
    } else if (byte_rem == 2u) {
        scale_bits = (w[word_off] >> 16u) & 0xFFFFu;
    } else {
        uint w0 = w[word_off];
        uint w1 = w[word_off + 1u];
        scale_bits = ((w0 >> (byte_rem * 8u)) | (w1 << ((4u - byte_rem) * 8u))) & 0xFFFFu;
    }
    return float(f16tof32(scale_bits));
}

uint q_pair(uint qs_byte) {
    uint word_off = qs_byte / 4u;
    uint byte_rem = qs_byte % 4u;
    if (byte_rem == 0u) {
        return w[word_off] & 0xFFFFu;
    } else if (byte_rem == 2u) {
        return (w[word_off] >> 16u) & 0xFFFFu;
    } else {
        uint w0 = w[word_off];
        uint w1 = w[word_off + 1u];
        return ((w0 >> (byte_rem * 8u)) | (w1 << ((4u - byte_rem) * 8u))) & 0xFFFFu;
    }
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void gemv_q4_0_fast(
    uint3 lid : SV_GroupThreadID,
    uint3 wid : SV_GroupID
) {
    uint m = params[0].x;
    uint k = params[0].y;
    uint nb = k / 32u;
    uint row_bytes = nb * BLOCK_BYTES;
    uint r0 = get_wid(wid) * NR;
    uint tid = lid.x;

    uint ix = tid / 2u;
    uint il = (tid & 1u) * 8u;

    float sumf[NR];
    [unroll]
    for (uint r = 0u; r < NR; r++) {
        sumf[r] = 0.0f;
    }

    for (uint chunk_b = 0u; chunk_b < nb; chunk_b += CHUNK_BLOCKS) {
        uint chunk_k_start = chunk_b * 32u;
        uint chunk_k_len = min(CHUNK_FLOATS, k - chunk_k_start);
        for (uint i = tid; i < chunk_k_len; i += WG_SIZE) {
            x_stage[i] = x[chunk_k_start + i];
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

            [unroll]
            for (uint r = 0u; r < NR; r++) {
                if (r0 + r >= m) { continue; }
                uint blk_byte = (r0 + r) * row_bytes + ib * BLOCK_BYTES;
                float d = block_scale(blk_byte);
                uint qs_byte = blk_byte + 2u + il;

                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;

                uint q0 = q_pair(qs_byte + 0u);
                acc0 += y0 * float(q0 & 0x000Fu);
                acc1 += y1 * float(q0 & 0x0F00u);
                acc2 += y8 * float(q0 & 0x00F0u);
                acc3 += y9 * float(q0 & 0xF000u);

                uint q1 = q_pair(qs_byte + 2u);
                acc0 += y2 * float(q1 & 0x000Fu);
                acc1 += y3 * float(q1 & 0x0F00u);
                acc2 += y10 * float(q1 & 0x00F0u);
                acc3 += y11 * float(q1 & 0xF000u);

                uint q2 = q_pair(qs_byte + 4u);
                acc0 += y4 * float(q2 & 0x000Fu);
                acc1 += y5 * float(q2 & 0x0F00u);
                acc2 += y12 * float(q2 & 0x00F0u);
                acc3 += y13 * float(q2 & 0xF000u);

                uint q3 = q_pair(qs_byte + 6u);
                acc0 += y6 * float(q3 & 0x000Fu);
                acc1 += y7 * float(q3 & 0x0F00u);
                acc2 += y14 * float(q3 & 0x00F0u);
                acc3 += y15 * float(q3 & 0xF000u);

                sumf[r] += d * (sumy_total * -8.0f + acc0 + acc1 + acc2 + acc3);
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
            if (r0 + r < m) {
                y[r0 + r] = partials[r * WG_SIZE];
            }
        }
    }
}
