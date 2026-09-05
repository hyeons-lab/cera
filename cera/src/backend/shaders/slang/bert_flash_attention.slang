// Bidirectional multi-head self-attention with online softmax and sliding-window support.
// Multi-target Slang source emitting both Metal and WGSL.
//
// slang-entries: bert_flash_attention
//
// Computes unmasked attention:
//   out = Softmax(Q * K^T * scale) * V
// with online softmax (FlashAttention-2 style) streaming K and V in K_TILE blocks
// into shared memory to keep VRAM consumption strictly O(N).
//
// ModernBERT local layers set window_size > 0 (e.g. 128 tokens); global layers set window_size = 0.
// When window_size > 0, keys outside [q - window_size/2, q + window_size/2] are masked to -inf.

[[vk::binding(0)]] StructuredBuffer<float>   q_buf   : register(t0);
[[vk::binding(1)]] StructuredBuffer<float>   k_buf   : register(t1);
[[vk::binding(2)]] StructuredBuffer<float>   v_buf   : register(t2);
[[vk::binding(3)]] RWStructuredBuffer<float> out_buf : register(u3);
[[vk::binding(4)]] StructuredBuffer<uint4>   par_buf : register(t4);

static const uint Q_TILE = 64u;
static const uint K_TILE = 32u;
static const uint MAX_HEAD_DIM = 64u;
static const float NEG_INF = -3.402823466e+38f;

groupshared float k_tile[K_TILE * MAX_HEAD_DIM];
groupshared float v_tile[K_TILE * MAX_HEAD_DIM];

[shader("compute")]
[numthreads(Q_TILE, 1, 1)]
void bert_flash_attention(
    uint3 lid : SV_GroupThreadID,
    uint3 wid : SV_GroupID
) {
    uint tid = lid.x;
    uint4 p0 = par_buf[0];
    uint4 p1 = par_buf[1];
    
    uint tokens = p0.x;
    uint n_head = p0.y;
    uint head_dim = p0.z;
    float scale = asfloat(p0.w);
    uint window_size = p1.x; // 0 = global attention, >0 = sliding window size
    uint half_window = window_size / 2u;

    uint h = wid.y;
    uint dim = n_head * head_dim;

    uint q_idx = wid.x * Q_TILE + tid;
    bool valid_q = q_idx < tokens;

    // Load query vector for this thread into local registers
    float q_reg[MAX_HEAD_DIM];
    if (valid_q) {
        uint q_off = q_idx * dim + h * head_dim;
        for (uint d = 0u; d < head_dim; d++) {
            q_reg[d] = q_buf[q_off + d];
        }
    } else {
        for (uint d = 0u; d < head_dim; d++) {
            q_reg[d] = 0.0f;
        }
    }

    // Online softmax state in registers
    float m_prev = NEG_INF;
    float l_prev = 0.0f;
    float acc[MAX_HEAD_DIM];
    for (uint d = 0u; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    // Compute uniform workgroup query boundaries for safe cooperative skipping
    uint wg_q_min = wid.x * Q_TILE;
    uint wg_q_max = min(tokens, wg_q_min + Q_TILE);
    uint wg_k_start = (window_size > 0u && wg_q_min > half_window) ? (wg_q_min - half_window) : 0u;
    uint wg_k_end = (window_size > 0u) ? min(tokens, wg_q_max + half_window) : tokens;

    // Loop over Key/Value blocks
    uint num_k_tiles = (tokens + K_TILE - 1u) / K_TILE;
    for (uint kt = 0u; kt < num_k_tiles; kt++) {
        uint k_base = kt * K_TILE;

        // Uniform workgroup skip: only skip if ALL threads in the workgroup are out of range
        if (window_size > 0u && (k_base + K_TILE <= wg_k_start || k_base >= wg_k_end)) {
            continue;
        }

        // Cooperatively load K and V blocks into shared memory
        // Total elements per tile: K_TILE * head_dim (32 * 64 = 2048)
        // With Q_TILE = 64 threads, each thread loads 2048 / 64 = 32 elements
        uint total_elems = K_TILE * head_dim;
        uint elems_per_thread = total_elems / Q_TILE;
        uint thread_load_base = tid * elems_per_thread;

        for (uint i = 0u; i < elems_per_thread; i++) {
            uint elem_idx = thread_load_base + i;
            uint k_row = elem_idx / head_dim;
            uint k_col = elem_idx % head_dim;
            uint global_k_token = k_base + k_row;

            if (global_k_token < tokens) {
                uint g_off = global_k_token * dim + h * head_dim + k_col;
                k_tile[elem_idx] = k_buf[g_off];
                v_tile[elem_idx] = v_buf[g_off];
            } else {
                k_tile[elem_idx] = 0.0f;
                v_tile[elem_idx] = 0.0f;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // Process this K/V block for query token q_idx
        if (valid_q) {
            uint tile_len = min(K_TILE, tokens - k_base);
            for (uint k_pos = 0u; k_pos < tile_len; k_pos++) {
                uint global_k = k_base + k_pos;

                // Check sliding window distance
                if (window_size > 0u) {
                    uint diff = (q_idx >= global_k) ? (q_idx - global_k) : (global_k - q_idx);
                    if (diff > half_window) {
                        continue;
                    }
                }

                // Dot product Q * K
                float score = 0.0f;
                uint k_off = k_pos * head_dim;
                for (uint d = 0u; d < head_dim; d++) {
                    score += q_reg[d] * k_tile[k_off + d];
                }
                score *= scale;

                // Online softmax update:
                // m_new = max(m_prev, score)
                // l_new = l_prev * exp(m_prev - m_new) + exp(score - m_new)
                // acc = acc * exp(m_prev - m_new) + exp(score - m_new) * V
                float m_new = max(m_prev, score);
                float alpha = exp(m_prev - m_new);
                float beta = exp(score - m_new);

                l_prev = l_prev * alpha + beta;
                m_prev = m_new;

                uint v_off = k_pos * head_dim;
                for (uint d = 0u; d < head_dim; d++) {
                    acc[d] = acc[d] * alpha + beta * v_tile[v_off + d];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Final normalization and write to output buffer
    if (valid_q) {
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        uint out_off = q_idx * dim + h * head_dim;
        for (uint d = 0u; d < head_dim; d++) {
            out_buf[out_off + d] = acc[d] * inv_l;
        }
    }
}
