// slang-entries: ffn_swiglu_q4_0
//
// Fast Fused Q4_0 SwiGLU FFN in Slang: y[m] = silu(W_gate[m, k] * x[k]) * (W_up[m, k] * x[k])
//
// Tiled across NR=4 output rows per workgroup with cooperative workgroup shared
// memory staging of activations x. Both Gate and Up projections are evaluated
// simultaneously with fast 16-bit word nibble pairs and SFU exp2 SiLU activation.

[[vk::binding(0)]] StructuredBuffer<uint>    w_gate : register(t0);
[[vk::binding(1)]] StructuredBuffer<uint>    w_up   : register(t1);
[[vk::binding(2)]] StructuredBuffer<float>   x      : register(t2);
[[vk::binding(3)]] RWStructuredBuffer<float> y      : register(u3);
[[vk::binding(4)]] StructuredBuffer<uint4>   params : register(t4);

static const uint NR = 4u;
static const uint NQ = 16u;
static const uint WG_SIZE = 32u;
static const uint BLOCK_BYTES = 18u;
static const uint CHUNK_BLOCKS = 16u;
static const uint CHUNK_FLOATS = CHUNK_BLOCKS * 32u; // 512 floats (2 KB)
static const float LOG2_E = 1.4426950408889634f;

groupshared float partials_gate[NR * WG_SIZE];
groupshared float partials_up[NR * WG_SIZE];
groupshared float x_stage[CHUNK_FLOATS];

uint get_wid(uint3 wid) {
    return wid.x + wid.y * 65535u;
}

float block_scale(StructuredBuffer<uint> w, uint blk_byte) {
    uint word = w[blk_byte >> 2u];
    uint scale_bits = ((blk_byte & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
    return float(f16tof32(scale_bits));
}

uint q_pair(StructuredBuffer<uint> w, uint qs_byte) {
    uint word = w[qs_byte >> 2u];
    return ((qs_byte & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void ffn_swiglu_q4_0(
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

    float sum_gate[NR];
    float sum_up[NR];
    [unroll]
    for (uint r = 0u; r < NR; r++) {
        sum_gate[r] = 0.0f;
        sum_up[r]   = 0.0f;
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

            [unroll]
            for (uint r = 0u; r < NR; r++) {
                if (r0 + r >= m) { continue; }
                uint blk_byte = (r0 + r) * row_bytes + ib * BLOCK_BYTES;
                uint qs_byte = blk_byte + 2u + il;

                // Gate dot product
                float d_gate = block_scale(w_gate, blk_byte);
                uint gq0 = q_pair(w_gate, qs_byte + 0u);
                uint gq1 = q_pair(w_gate, qs_byte + 2u);
                uint gq2 = q_pair(w_gate, qs_byte + 4u);
                uint gq3 = q_pair(w_gate, qs_byte + 6u);

                float g_acc0 = y0 * float(gq0 & 0x000Fu) + y2 * float(gq1 & 0x000Fu) + y4 * float(gq2 & 0x000Fu) + y6 * float(gq3 & 0x000Fu);
                float g_acc1 = y1 * float(gq0 & 0x0F00u) + y3 * float(gq1 & 0x0F00u) + y5 * float(gq2 & 0x0F00u) + y7 * float(gq3 & 0x0F00u);
                float g_acc2 = y8 * float(gq0 & 0x00F0u) + y10 * float(gq1 & 0x00F0u) + y12 * float(gq2 & 0x00F0u) + y14 * float(gq3 & 0x00F0u);
                float g_acc3 = y9 * float(gq0 & 0xF000u) + y11 * float(gq1 & 0xF000u) + y13 * float(gq2 & 0xF000u) + y15 * float(gq3 & 0xF000u);

                sum_gate[r] += d_gate * (sumy_total * -8.0f + g_acc0 + g_acc1 + g_acc2 + g_acc3);

                // Up dot product
                float d_up = block_scale(w_up, blk_byte);
                uint uq0 = q_pair(w_up, qs_byte + 0u);
                uint uq1 = q_pair(w_up, qs_byte + 2u);
                uint uq2 = q_pair(w_up, qs_byte + 4u);
                uint uq3 = q_pair(w_up, qs_byte + 6u);

                float u_acc0 = y0 * float(uq0 & 0x000Fu) + y2 * float(uq1 & 0x000Fu) + y4 * float(uq2 & 0x000Fu) + y6 * float(uq3 & 0x000Fu);
                float u_acc1 = y1 * float(uq0 & 0x0F00u) + y3 * float(uq1 & 0x0F00u) + y5 * float(uq2 & 0x0F00u) + y7 * float(uq3 & 0x0F00u);
                float u_acc2 = y8 * float(uq0 & 0x00F0u) + y10 * float(uq1 & 0x00F0u) + y12 * float(uq2 & 0x00F0u) + y14 * float(uq3 & 0x00F0u);
                float u_acc3 = y9 * float(uq0 & 0xF000u) + y11 * float(uq1 & 0xF000u) + y13 * float(uq2 & 0xF000u) + y15 * float(uq3 & 0xF000u);

                sum_up[r] += d_up * (sumy_total * -8.0f + u_acc0 + u_acc1 + u_acc2 + u_acc3);
            }

            ib_local += NQ;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (uint r = 0u; r < NR; r++) {
        partials_gate[r * WG_SIZE + tid] = sum_gate[r];
        partials_up[r * WG_SIZE + tid]   = sum_up[r];
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = WG_SIZE / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            [unroll]
            for (uint r = 0u; r < NR; r++) {
                uint idx = r * WG_SIZE + tid;
                partials_gate[idx] += partials_gate[idx + stride];
                partials_up[idx]   += partials_up[idx + stride];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u) {
        [unroll]
        for (uint r = 0u; r < NR; r++) {
            if (r0 + r < m) {
                float g = partials_gate[r * WG_SIZE];
                float u = partials_up[r * WG_SIZE];
                float clamped_g = clamp(g, -80.0f, 80.0f);
                float sig_g = 1.0f / (1.0f + exp2(-clamped_g * LOG2_E));
                y[r0 + r] = (g * sig_g) * u;
            }
        }
    }
}
