// slang-entries: ffn_swiglu_q4_0
//
// Fused Q4_0 SwiGLU FFN: y[m] = silu(W_gate[m, k] * x[k]) * (W_up[m, k] * x[k])
//
// Combines two Q4_0 GEMVs (gate and up) and the elementwise SiLU product into
// a single compute pass per FFN layer.
//
// Threadgroup layout: 32 threads per row m.
//  - Threads 0..15: evaluate W_gate * x (stride-16 block traversal)
//  - Threads 16..31: evaluate W_up * x (stride-16 block traversal)

[[vk::binding(0)]] StructuredBuffer<uint>    w_gate : register(t0);
[[vk::binding(1)]] StructuredBuffer<uint>    w_up   : register(t1);
[[vk::binding(2)]] StructuredBuffer<float>   x      : register(t2);
[[vk::binding(3)]] RWStructuredBuffer<float> y      : register(u3);
[[vk::binding(4)]] StructuredBuffer<uint2>   params : register(t4);

static const uint WG = 32u;
static const uint BLOCK_BYTES = 18u;

groupshared float scratch[WG];

uint load_gate_byte(uint byte_off) {
    uint word = w_gate[byte_off >> 2u];
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

float load_gate_f16(uint byte_off) {
    uint word = w_gate[byte_off >> 2u];
    uint h = ((byte_off & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
    return f16tof32(h);
}

uint load_up_byte(uint byte_off) {
    uint word = w_up[byte_off >> 2u];
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

float load_up_f16(uint byte_off) {
    uint word = w_up[byte_off >> 2u];
    uint h = ((byte_off & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
    return f16tof32(h);
}

uint get_wid(uint3 wid) {
    return wid.x + wid.y * 65535u;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void ffn_swiglu_q4_0(uint3 lid : SV_GroupThreadID, uint3 grp : SV_GroupID) {
    uint m = params[0].x;
    uint k = params[0].y;
    uint row = get_wid(grp);

    if (row >= m) {
        return;
    }

    uint nb = k / 32u;
    uint row_bytes = nb * BLOCK_BYTES;
    uint row_base = row * row_bytes;

    bool is_up_thread = (lid.x >= 16u);
    uint sub_tid = lid.x & 15u;

    float sum = 0.0f;

    if (!is_up_thread) {
        // Threads 0..15 compute Gate
        for (uint bi = sub_tid; bi < nb; bi += 16u) {
            uint blk = row_base + bi * BLOCK_BYTES;
            uint col = bi * 32u;
            float acc = 0.0f;
            float sumy = 0.0f;
            for (uint i = 0u; i < 16u; ++i) {
                uint byte = load_gate_byte(blk + 2u + i);
                float x0 = x[col + i];
                float x1 = x[col + i + 16u];
                sumy += x0 + x1;
                acc += float(byte & 0xFu) * x0 + float(byte >> 4u) * x1;
            }
            sum += (acc - 8.0f * sumy) * load_gate_f16(blk);
        }
    } else {
        // Threads 16..31 compute Up
        for (uint bi = sub_tid; bi < nb; bi += 16u) {
            uint blk = row_base + bi * BLOCK_BYTES;
            uint col = bi * 32u;
            float acc = 0.0f;
            float sumy = 0.0f;
            for (uint i = 0u; i < 16u; ++i) {
                uint byte = load_up_byte(blk + 2u + i);
                float x0 = x[col + i];
                float x1 = x[col + i + 16u];
                sumy += x0 + x1;
                acc += float(byte & 0xFu) * x0 + float(byte >> 4u) * x1;
            }
            sum += (acc - 8.0f * sumy) * load_up_f16(blk);
        }
    }

    // Workgroup reduction across 16-lane halves
    scratch[lid.x] = sum;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction within each 16-thread half
    if (sub_tid < 8u) {
        scratch[lid.x] += scratch[lid.x + 8u];
    }
    GroupMemoryBarrierWithGroupSync();

    if (sub_tid < 4u) {
        scratch[lid.x] += scratch[lid.x + 4u];
    }
    GroupMemoryBarrierWithGroupSync();

    if (sub_tid < 2u) {
        scratch[lid.x] += scratch[lid.x + 2u];
    }
    GroupMemoryBarrierWithGroupSync();

    if (sub_tid == 0u) {
        scratch[lid.x] += scratch[lid.x + 1u];
    }
    GroupMemoryBarrierWithGroupSync();

    // Thread 0 computes final SiLU product and stores
    if (lid.x == 0u) {
        float gate_sum = scratch[0];
        float up_sum   = scratch[16];
        // silu(gate) = gate / (1.0 + exp(-gate))
        float silu_gate = gate_sum / (1.0f + exp(-gate_sum));
        y[row] = silu_gate * up_sum;
    }
}
