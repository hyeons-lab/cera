// Fast Q4_0 GEMV: y[m] = dequant(W_q4_0[m, k]) @ x[k]. SPIR-V passthrough
// twin of `../slang/gemv_q4_0_fast.slang` (same K loop, same NR/NQ — keep
// them in sync), differing only in the cross-thread reduction: subgroups
// (`WaveActiveSum` + one shared-mem combine) instead of the 6-barrier tree.
//
// Why a twin instead of one source: naga does not implement WGSL `enable
// subgroups`, so the subgroup reduction cannot ship through the WGSL path on
// any backend. The twin compiles straight to SPIR-V (GroupNonUniform is
// core since Vulkan 1.1) and rides `create_shader_module_passthrough` on
// Vulkan; every other backend dispatches the WGSL tree. Mirrors llama.cpp's
// Adreno Q4_0 kernel (mul_mv_q4_0_f32), whose subgroup reduction has no
// shared memory and no barriers.
//
// Bindings (same as the twin): 0 = weights (u32 view of Q4_0 blocks),
// 1 = activations (f32), 2 = dst (f32, read-write), 3 = params (uint4:
// m, k, unused, unused). NR=8 rows per workgroup, 64 threads, NQ=32
// blocks per iteration. Dispatch: gemv_row_workgroups(ceil(m / 8)).

[[vk::binding(0, 0)]] StructuredBuffer<uint>    w;
[[vk::binding(1, 0)]] StructuredBuffer<float>   x;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(3, 0)]] StructuredBuffer<uint4>   params;

static const uint NR = 8u;
static const uint NQ = 32u;
static const uint WG_SIZE = 64u;
static const uint BLOCK_BYTES = 18u;
// Subgroup-combine scratch: one float per subgroup per row. Sized for the
// smallest conformant subgroup (1 lane -> 64 subgroups, 2KB); real mobile
// hardware is 16..64 lanes, but a size-4 software device must not write out
// of bounds.
static const uint MAX_SUBGROUPS = 64u;

groupshared float sg_scratch[NR * MAX_SUBGROUPS];

uint get_wid(uint3 wid) {
    return wid.x + wid.y * 65535u;
}

float block_scale(uint blk_byte) {
    uint word = w[blk_byte >> 2u];
    uint scale_bits = ((blk_byte & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
    return float(f16tof32(scale_bits));
}

uint q_pair(uint qs_byte) {
    uint word = w[qs_byte >> 2u];
    return ((qs_byte & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
}

// The K loop: each thread accumulates its blocks' partial dots for NR rows
// into `sumf`. Identical to the twin's — only the reduction differs.
void q4_block_loop(
    uint m,
    uint k,
    uint r0,
    uint ix,
    uint il,
    inout float sumf[NR]
) {
    uint nb = k / 32u;
    uint row_bytes = nb * BLOCK_BYTES;

    [unroll]
    for (uint r = 0u; r < NR; r++) {
        sumf[r] = 0.0f;
    }

    uint yb_off = ix * 32u + il;
    uint ib = ix;

    // Threads with ix >= nb (only when k < 1024, i.e. nb < NQ) skip the
    // loop with sumf == 0 and still join the reduction below — correct.
    while (ib < nb) {
        float a0  = x[yb_off + 0u];
        float a1  = x[yb_off + 1u];
        float a2  = x[yb_off + 2u];
        float a3  = x[yb_off + 3u];
        float a4  = x[yb_off + 4u];
        float a5  = x[yb_off + 5u];
        float a6  = x[yb_off + 6u];
        float a7  = x[yb_off + 7u];
        float a8  = x[yb_off + 16u];
        float a9  = x[yb_off + 17u];
        float a10 = x[yb_off + 18u];
        float a11 = x[yb_off + 19u];
        float a12 = x[yb_off + 20u];
        float a13 = x[yb_off + 21u];
        float a14 = x[yb_off + 22u];
        float a15 = x[yb_off + 23u];

        // Reciprocal mults, not divides: the divisors are exact powers of two
        // (1/256, 1/16, 1/4096 are all exactly representable), so the products
        // are bit-identical to the quotients.
        float y0 = a0;
        float y1 = a1 * 0.00390625f;
        float y2 = a2;
        float y3 = a3 * 0.00390625f;
        float y4 = a4;
        float y5 = a5 * 0.00390625f;
        float y6 = a6;
        float y7 = a7 * 0.00390625f;
        float y8 = a8 * 0.0625f;
        float y9 = a9 * 0.000244140625f;
        float y10 = a10 * 0.0625f;
        float y11 = a11 * 0.000244140625f;
        float y12 = a12 * 0.0625f;
        float y13 = a13 * 0.000244140625f;
        float y14 = a14 * 0.0625f;
        float y15 = a15 * 0.000244140625f;

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

        ib += NQ;
        yb_off += NQ * 32u;
    }
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void main(
    uint3 lid : SV_GroupThreadID,
    uint3 wid : SV_GroupID
) {
    uint m = params[0].x;
    uint k = params[0].y;
    uint r0 = get_wid(wid) * NR;
    uint tid = lid.x;

    uint ix = tid / 2u;
    uint il = (tid & 1u) * 8u;

    float sumf[NR];
    q4_block_loop(m, k, r0, ix, il, sumf);

    // Subgroup reduction: each subgroup sums its lanes' partials with one
    // wave intrinsic, lane 0 of each subgroup publishes to scratch, and lane
    // 0 of the workgroup combines (<= MAX_SUBGROUPS values) and writes.
    // `lane_count` is uniform, so `sgid`/`nsg` are well-defined. One barrier
    // total, vs six in the WGSL twin's tree.
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
            float t = 0.0f;
            for (uint i = 0u; i < nsg; i++) {
                t += sg_scratch[r * MAX_SUBGROUPS + i];
            }
            if (r0 + r < m) {
                y[r0 + r] = t;
            }
        }
    }
}
