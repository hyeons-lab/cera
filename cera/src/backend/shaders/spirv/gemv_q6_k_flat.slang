// Flat-planes Q6_K GEMV: y[m] = dequant(W_flat[m, k]) @ x[k]. SPIR-V
// passthrough kernel (entry `main`), dispatched on Vulkan when subgroups
// are available; there is no WGSL twin (like `gemv_q4_0_stream`, the flat
// layout only exists where passthrough does). Same 4-binding interface
// (0 = weights as u32, 1 = x f32, 2 = y f32 read-write, 3 = params
// [m, k, ...]), so host dispatch is unchanged apart from rows-per-group.
//
// The weight buffer holds `repack_q6_k_flat` output, not GGUF bytes: all
// rows' low bytes, then high bytes, scales, super-scales
// (`ql[m*nb*128]`, `qh[m*nb*64]`, `s[m*nb*16]`, `d[m*nb*2]`, nb = k/256).
// The 210-byte interleaved stride misaligns every block (210 % 4 == 2),
// forcing the raw kernel's funnel-shift loads (2 word loads + shifts per
// 4 bytes); the planes are 4-aligned throughout, so quants read with
// single word loads and no shifts — 8 loads per (block, row) instead of
// 12, and ~30 fewer ALU ops. Same lane mapping as the raw kernel (64
// threads, NQ=4 blocks per iteration, piece/slot/ip/il) and the same
// accumulation order, so results match to FMA contraction noise.
// (llama.cpp's `mul_mv_q6_K_f32_flat` layout, single-buffer form;
// N_DST=8 here vs their 16.)
//
// NR=8 rows per workgroup: the raw kernel's NR=8 lost to occupancy, but
// the flat body is ~40% leaner per row (no funnel shifts), so 8 rows fit.
// Two blocks per thread per iteration was tried for ILP and regressed 3.4x
// (register spill); one block per iteration is the sweet spot.

[[vk::binding(0, 0)]] StructuredBuffer<uint>  w;
[[vk::binding(1, 0)]] StructuredBuffer<float> x;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> params;

static const uint QK_K = 256u;
static const uint NR = 8u;
static const uint WG_SIZE = 64u;
static const uint NQ = 4u;
// Subgroup-combine scratch: one float per subgroup per row. Sized for the
// smallest conformant subgroup (1 lane -> 64 subgroups, 2KB at NR=8);
// real mobile hardware is 16..64 lanes, but a size-4 software device must
// not write out of bounds.
static const uint MAX_SUBGROUPS = 64u;

groupshared float sg_scratch[NR * MAX_SUBGROUPS];

uint get_wid(uint3 wid) {
    return wid.x + wid.y * 65535u;
}

uint rb(uint off) {
    return (w[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu;
}

// Block scale: an f16 at an always-even offset, so one word load plus a
// half-select (the Q4_0 `block_scale` trick), not two byte loads. The d
// plane base (m*nb*208 bytes) is 4-aligned and every d sits at a
// (row*nb + b)*2 even byte offset inside it.
float block_d(uint d_off) {
    uint word = w[d_off >> 2u];
    uint bits = ((d_off & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
    return float(f16tof32(bits));
}

// The four 6-bit weights sharing one high-bits byte, in q6_1..q6_4 order.
float4 q6_lane(uint q1, uint q2, uint qhv) {
    return float4(
        float(int((q1 & 0x0Fu) | ((qhv & 0x03u) << 4u)) - 32),
        float(int((q2 & 0x0Fu) | ((qhv & 0x0Cu) << 2u)) - 32),
        float(int((q1 >> 4u) | (qhv & 0x30u)) - 32),
        float(int((q2 >> 4u) | ((qhv & 0xC0u) >> 2u)) - 32)
    );
}

int ri8(uint off) {
    uint b = rb(off);
    return int(b) - (((b & 0x80u) != 0u) ? 256 : 0);
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(
    uint3 lid : SV_GroupThreadID,
    uint3 wid : SV_GroupID
) {
    uint m = params[0].x;
    uint k = params[0].y;
    uint nb = k / QK_K;
    uint tid = lid.x;
    uint first_row = get_wid(wid) * NR;

    // Plane bases. All are multiples of 4 bytes (128/64/16-byte row
    // strides; the d plane base is m*nb*208, 208 = 52*4).
    uint qlp_words = m * nb * 32u;
    uint qhp_words = m * nb * 16u;
    uint sp_bytes = (qlp_words + qhp_words) * 4u;
    uint dp_bytes = sp_bytes + m * nb * 16u;

    uint piece = tid & 15u;
    uint slot = (tid >> 4u) & 3u;
    uint ip = piece >> 3u;
    uint il = piece & 7u;
    uint l0 = 4u * il;
    uint is_off = 8u * ip + l0 / 16u;

    uint y_offset   = 128u * ip + l0;
    uint q_offset_l = 16u * ip + l0 / 4u; // ql words (bytes 64*ip+l0, /4)
    uint q_offset_h = 8u * ip + l0 / 4u;  // qh words (bytes 32*ip+l0, /4)

    float sumf[NR];
    [unroll]
    for (uint row = 0u; row < NR; row++) {
        sumf[row] = 0.0f;
    }

    uint b = slot;
    while (b < nb) {
        uint yb = b * QK_K + y_offset;
        // Four float4s, not an indexed local array: loop-indexed arrays are
        // not reliably promoted to registers.
        float4 ylv0 = float4(x[yb + 0u], x[yb + 32u], x[yb + 64u], x[yb + 96u]);
        float4 ylv1 = float4(x[yb + 1u], x[yb + 33u], x[yb + 65u], x[yb + 97u]);
        float4 ylv2 = float4(x[yb + 2u], x[yb + 34u], x[yb + 66u], x[yb + 98u]);
        float4 ylv3 = float4(x[yb + 3u], x[yb + 35u], x[yb + 67u], x[yb + 99u]);

        [unroll]
        for (uint row = 0u; row < NR; row++) {
            // Skip weight reads for out-of-range rows (m tail).
            if (first_row + row >= m) { continue; }
            uint r = first_row + row;
            // Aligned single-word quant loads — no funnel shifts. (ql2 is
            // 32 bytes past ql1, i.e. 8 words.)
            uint w1 = w[r * nb * 32u + b * 32u + q_offset_l];
            uint w2 = w[r * nb * 32u + b * 32u + q_offset_l + 8u];
            uint wh = w[qlp_words + r * nb * 16u + b * 16u + q_offset_h];
            uint sc  = sp_bytes + r * nb * 16u + b * 16u + is_off;
            uint d_off = dp_bytes + r * nb * 2u + b * 2u;

            float4 sums = float4(0.0f, 0.0f, 0.0f, 0.0f);
            sums += q6_lane(w1 & 0xFFu, w2 & 0xFFu, wh & 0xFFu) * ylv0;
            sums += q6_lane((w1 >> 8u) & 0xFFu, (w2 >> 8u) & 0xFFu, (wh >> 8u) & 0xFFu) * ylv1;
            sums += q6_lane((w1 >> 16u) & 0xFFu, (w2 >> 16u) & 0xFFu, (wh >> 16u) & 0xFFu) * ylv2;
            sums += q6_lane(w1 >> 24u, w2 >> 24u, wh >> 24u) * ylv3;

            float dblk = block_d(d_off);
            float s0 = float(ri8(sc));
            float s2 = float(ri8(sc + 2u));
            float s4 = float(ri8(sc + 4u));
            float s6 = float(ri8(sc + 6u));
            sumf[row] += dblk * (sums[0] * s0 + sums[1] * s2 + sums[2] * s4 + sums[3] * s6);
        }
        b += NQ;
    }

    // Subgroup reduction: one wave intrinsic per row, lane 0 of each
    // subgroup publishes, lane 0 of the workgroup combines and writes.
    // One barrier total.
    uint lane = WaveGetLaneIndex();
    uint lane_count = WaveGetLaneCount();
    uint sgid = tid / lane_count;
    [unroll]
    for (uint r = 0u; r < NR; r++) {
        float s = WaveActiveSum(sumf[r]);
        if (lane == 0u) {
            sg_scratch[r * MAX_SUBGROUPS + sgid] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0u) {
        // A wider-than-workgroup subgroup (128 lanes) still fills slot 0 via
        // its first lane; without the clamp nsg would be 0 and the combine
        // would write zeros.
        uint nsg = max(WG_SIZE / lane_count, 1u);
        [unroll]
        for (uint r = 0u; r < NR; r++) {
            float acc = 0.0f;
            for (uint i = 0u; i < nsg; i++) {
                acc += sg_scratch[r * MAX_SUBGROUPS + i];
            }
            if (first_row + r < m) {
                y[first_row + r] = acc;
            }
        }
    }
}
