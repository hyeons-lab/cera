// Fast Q6_K GEMV: y[m] = dequant(W_q6_k[m, k]) @ x[k]. SPIR-V passthrough
// kernel (entry `main`), dispatched on Vulkan when subgroups are available;
// the handwritten `../gemv_q6_k.wgsl` stays as the portable fallback. Same
// 4-binding interface (0 = weights as u32, 1 = x f32, 2 = y f32 read-write,
// 3 = params [m, k, ...]), so host dispatch is unchanged apart from the
// rows-per-workgroup, which the host gates on the pipeline: 1 here, 2 for
// the WGSL twin.
//
// Transcribed from `gemv_q6_k.wgsl` (which documents the funnel-shift loads
// and the vec4 lane layout — read it first), with WS3/WS4 changes:
//   - 64 threads covering NQ=4 blocks per iteration (the WGSL covers 2 with
//     32 threads). Thread (piece = tid & 15, slot = (tid >> 4) & 3) owns
//     piece `piece` of blocks b = slot (mod 4). Threads with slot >= nb
//     (only when k < 1024) skip the loop with zero accumulators.
//   - Subgroup reduction (`WaveActiveSum` + one shared-mem combine) instead
//     of the 5-barrier tree. Portable across subgroup sizes 8..64.
//   - NR=1 row per workgroup (the WGSL does 2). NR=8 was tried for x
//     amortization and regressed 19% (occupancy loss beats the traffic
//     saving); NR=1 and NR=2 measure identically, so NR=1 for the
//     simpler no-row-loop form. The remaining cost is per-(block,row)
//     dequant ALU + dependent funnel loads, insensitive to NR.
//   - Single-load block scale: `d` sits at an always-even byte offset (bb
//     is a sum of even terms: row_bytes = nb*210, 210*b, +208), so one
//     word load + half-select replaces the two byte loads, Q4_0 style.

[[vk::binding(0, 0)]] StructuredBuffer<uint>  w;
[[vk::binding(1, 0)]] StructuredBuffer<float> x;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> params;

static const uint QK_K = 256u;
static const uint Q6K_BYTES = 210u;
static const uint NR = 1u;
static const uint WG_SIZE = 64u;
static const uint NQ = 4u;
// Subgroup-combine scratch: one float per subgroup per row. Sized for the
// smallest conformant subgroup (1 lane -> 64 subgroups); real mobile
// hardware is 16..64 lanes, but a size-4 software device must not write
// out of bounds.
static const uint MAX_SUBGROUPS = 64u;

groupshared float sg_scratch[NR * MAX_SUBGROUPS];

uint get_wid(uint3 wid) {
    return wid.x + wid.y * 65535u;
}

uint rb(uint off) {
    return (w[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu;
}

// Four consecutive bytes at an arbitrary byte offset, as one u32 —
// funnel-shift of two word loads. `select`, NOT a branch: adjacent lanes
// work on even/odd blocks whose bases differ by 2 mod 4 (Q6K_BYTES % 4 ==
// 2), so a branch would split every subgroup. See the WGSL's note.
uint rw(uint off) {
    uint wi = off / 4u;
    uint sh = (off & 3u) * 8u;
    uint hi_sh = (32u - sh) & 31u;
    return (w[wi] >> sh) | (((sh == 0u) ? 0u : w[wi + 1u]) << hi_sh);
}

// Block scale: an f16 at an always-even offset, so one word load plus a
// half-select (the Q4_0 `block_scale` trick), not two byte loads.
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
    uint row_bytes = nb * Q6K_BYTES;
    uint tid = lid.x;
    uint row = get_wid(wid);

    uint piece = tid & 15u;
    uint slot = (tid >> 4u) & 3u;
    uint ip = piece >> 3u;
    uint il = piece & 7u;
    uint l0 = 4u * il;
    uint is_off = 8u * ip + l0 / 16u;

    uint y_offset   = 128u * ip + l0;
    uint q_offset_l = 64u * ip + l0;
    uint q_offset_h = 32u * ip + l0;

    float sumf = 0.0f;

    // Out-of-range rows (m tail) skip with a zero accumulator and still
    // join the reduction below — correct.
    if (row < m) {
        uint row_base = row * row_bytes;
        uint b = slot;
        while (b < nb) {
            uint yb = b * QK_K + y_offset;
            // Four float4s, not an indexed local array: loop-indexed arrays are
            // not reliably promoted to registers (WGSL notes issue #316).
            float4 ylv0 = float4(x[yb + 0u], x[yb + 32u], x[yb + 64u], x[yb + 96u]);
            float4 ylv1 = float4(x[yb + 1u], x[yb + 33u], x[yb + 65u], x[yb + 97u]);
            float4 ylv2 = float4(x[yb + 2u], x[yb + 34u], x[yb + 66u], x[yb + 98u]);
            float4 ylv3 = float4(x[yb + 3u], x[yb + 35u], x[yb + 67u], x[yb + 99u]);

            uint bb = row_base + b * Q6K_BYTES;
            uint ql1 = bb + q_offset_l;
            uint ql2 = ql1 + 32u;
            uint qh  = bb + 128u + q_offset_h;
            uint sc  = bb + 192u + is_off;
            uint d_off = bb + 208u;

            uint w1 = rw(ql1);
            uint w2 = rw(ql2);
            uint wh = rw(qh);

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
            sumf += dblk * (sums[0] * s0 + sums[1] * s2 + sums[2] * s4 + sums[3] * s6);
            b += NQ;
        }
    }

    // Subgroup reduction (see gemv_q4_0_fast.slang's SPIR-V twin): one wave
    // intrinsic, lane 0 of each subgroup publishes, lane 0 of the workgroup
    // combines and writes. One barrier total.
    uint lane = WaveGetLaneIndex();
    uint lane_count = WaveGetLaneCount();
    uint sgid = tid / lane_count;
    {
        float s = WaveActiveSum(sumf);
        if (lane == 0u) {
            sg_scratch[sgid] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0u) {
        // A wider-than-workgroup subgroup (128 lanes) still fills slot 0 via
        // its first lane; without the clamp nsg would be 0 and the combine
        // would write zeros.
        uint nsg = max(WG_SIZE / lane_count, 1u);
        float t = 0.0f;
        for (uint i = 0u; i < nsg; i++) {
            t += sg_scratch[i];
        }
        if (row < m) {
            y[row] = t;
        }
    }
}
