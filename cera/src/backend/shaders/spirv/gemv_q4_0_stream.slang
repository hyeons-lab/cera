// Q4_0 GEMV over the resident stream layout: y[m] = dequant(W[m, k]) @ x[k].
//
// The resident (q, d) repack is row-consecutive: `q[qi*m + row]` and
// `d[di*m + row]` run over consecutive rows, so consecutive threads cover
// consecutive rows — 16 rows x an 8-way K-split in 64-thread workgroups,
// 2 rows per thread (thread tid covers rows 2*(tid%8)+{0,1}, K-eighth
// tid/8). Two rows per thread amortizes the per-block activation scaling
// (the y-trick, below) over both rows and doubles the accumulator chains
// to 8 per row-pair (2 rows x 4 q-lanes + sumy chains), matching the ILP
// of `gemv_q4_0_fast`'s 8 row-chains. Split partials combine through 128
// floats of shmem with a single barrier.
//
// The nibble extracts use the reciprocal-mult trick (same as
// `gemv_q4_0_fast`): the 32 activations are pre-scaled by exact powers of
// two (1/16^s, all exactly representable) so the row loop applies
// unshifted masks instead of shifts — the products are bit-identical to
// shift-then-multiply, and the scaling hoists out of the row loop. The -8
// zero-point folds through a per-block activation sum accumulated on its
// own chain: row = dots - 8*sumy_acc, where dots and sumy_acc both carry
// the per-block scale ds.
//
// (A K-parallel mapping like `gemv_q4_0_fast`'s 32 block-lanes strides
// 4*m u32s between lanes and measured 3x slower than the raw kernel it
// replaces; the layout transposed the coalescing dimension, so the mapping
// follows it.)
//
// Per thread per block: 32 x-loads + 64 scale/sum ops (shared by both
// rows) + per row (4 q u32s + 1 d u32 + 32 masked FMAs), fp32
// accumulation. d is per-32-block (two packed per u32, lane b&1); each q
// u32 holds 8 sequential weights, low nibble first.
//
// Bindings: 0 = q (u32), 1 = d (u32), 2 = x (f32), 3 = y (f32, read-write),
// 4 = params (uint4: m, k, unused, unused). Dispatch:
// gemv_row_workgroups(ceil(m / 16)). Constraints (asserted host-side):
// k % 32 == 0 (the resident layout's precondition).
//
// SPIR-V-only (passthrough-gated like `gemm_stream_q4_0`): the resident
// layout only exists where passthrough does, so there is no WGSL twin.

[[vk::binding(0, 0)]] StructuredBuffer<uint>    q;
[[vk::binding(1, 0)]] StructuredBuffer<uint>    d;
[[vk::binding(2, 0)]] StructuredBuffer<float>   x;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(4, 0)]] StructuredBuffer<uint4>   params;

static const uint ROWS_PER_WG = 16u;
static const uint ROWS_PER_THREAD = 2u;
static const uint FIBERS = ROWS_PER_WG / ROWS_PER_THREAD;
static const uint K_SPLITS = 8u;
static const uint WG_SIZE = 64u;

groupshared float split_scratch[ROWS_PER_WG * K_SPLITS];

uint get_wid(uint3 wid) {
    return wid.x + wid.y * 65535u;
}

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
    uint row0 = get_wid(wid) * ROWS_PER_WG + f * ROWS_PER_THREAD;
    uint row1 = row0 + 1u;

    uint nb = k / 32u;
    // K-eighths divide the block range; nb % 8 != 0 leaves the tail
    // blocks to no split — pad by rounding the per-split count up and
    // letting overshoot splits idle (their partials stay 0).
    uint nb_per_split = (nb + K_SPLITS - 1u) / K_SPLITS;
    uint b0 = s * nb_per_split;
    uint b1 = min(b0 + nb_per_split, nb);

    bool live0 = row0 < m;
    bool live1 = row1 < m;
    // Clamped rows for addressing: dead rows (ds=0) still execute their
    // q/d reads (ternary sides both evaluate), so clamp into range rather
    // than overhanging into the next qi's rows or past the buffer end.
    // Result-identical (0 * anything), strictly in-bounds.
    uint row0c = live0 ? row0 : m - 1u;
    uint row1c = live1 ? row1 : m - 1u;
    // 10 chains: per row, 4 q-lane dot chains + 1 sumy chain.
    float d00 = 0.0f;
    float d01 = 0.0f;
    float d02 = 0.0f;
    float d03 = 0.0f;
    float s0 = 0.0f;
    float d10 = 0.0f;
    float d11 = 0.0f;
    float d12 = 0.0f;
    float d13 = 0.0f;
    float s1 = 0.0f;

    for (uint b = b0; b < b1; b++) {
        uint xbase = b * 32u;
        uint di = (b / 2u) * m;
        uint dshift = (b & 1u) * 16u;
        uint qi = b * 4u * m;
        float ds0 = live0 ? float(f16tof32((d[di + row0c] >> dshift) & 0xFFFFu)) : 0.0f;
        float ds1 = live1 ? float(f16tof32((d[di + row1c] >> dshift) & 0xFFFFu)) : 0.0f;
        float sumy_b = 0.0f;

        // One q-u32 (8 weights) at a time: 8 x-loads + 8 scaled y + the
        // masked dots for both rows, then the registers recycle. Holding
        // all 32 x/y (plus the 10 accumulators) needs ~80 live regs and
        // craters occupancy; this streaming shape needs ~35.
        [unroll]
        for (uint p = 0u; p < 4u; p++) {
            uint xb = xbase + p * 8u;
            // Reciprocal mults, not divides: exact powers of two (see header).
            float a0 = x[xb];        float y0 = a0;
            float a1 = x[xb + 1u];   float y1 = a1 * 0.0625f;
            float a2 = x[xb + 2u];   float y2 = a2 * 0.00390625f;
            float a3 = x[xb + 3u];   float y3 = a3 * 0.000244140625f;
            float a4 = x[xb + 4u];   float y4 = a4 * 0.0000152587890625f;
            float a5 = x[xb + 5u];   float y5 = a5 * 9.5367431640625e-7f;
            float a6 = x[xb + 6u];   float y6 = a6 * 5.960464477539063e-8f;
            float a7 = x[xb + 7u];   float y7 = a7 * 3.725290298461914e-9f;
            sumy_b += (a0 + a1) + (a2 + a3) + (a4 + a5) + (a6 + a7);

            uint q0w = q[qi + p * m + row0c];
            uint q1w = q[qi + p * m + row1c];
            float dot0 = y0 * float(q0w & 0x0000000Fu) + y1 * float(q0w & 0x000000F0u)
                       + y2 * float(q0w & 0x00000F00u) + y3 * float(q0w & 0x0000F000u)
                       + y4 * float(q0w & 0x000F0000u) + y5 * float(q0w & 0x00F00000u)
                       + y6 * float(q0w & 0x0F000000u) + y7 * float(q0w & 0xF0000000u);
            float dot1 = y0 * float(q1w & 0x0000000Fu) + y1 * float(q1w & 0x000000F0u)
                       + y2 * float(q1w & 0x00000F00u) + y3 * float(q1w & 0x0000F000u)
                       + y4 * float(q1w & 0x000F0000u) + y5 * float(q1w & 0x00F00000u)
                       + y6 * float(q1w & 0x0F000000u) + y7 * float(q1w & 0xF0000000u);
            // Dead rows contribute ds=0 dots (their clamped addresses
            // read valid memory, discarded). The p-ladder keeps one
            // accumulator chain per q-lane; [unroll] above folds it.
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
        s0 += ds0 * sumy_b;
        s1 += ds1 * sumy_b;
    }

    uint r0 = f * ROWS_PER_THREAD;
    split_scratch[r0 * K_SPLITS + s] = (d00 + d01) + (d02 + d03) - 8.0f * s0;
    split_scratch[(r0 + 1u) * K_SPLITS + s] = (d10 + d11) + (d12 + d13) - 8.0f * s1;
    GroupMemoryBarrierWithGroupSync();

    // First 16 threads combine their row's 8 splits and write.
    if (tid < ROWS_PER_WG) {
        uint r = tid;
        uint row = get_wid(wid) * ROWS_PER_WG + r;
        if (row < m) {
            float t = (split_scratch[r * K_SPLITS] + split_scratch[r * K_SPLITS + 1u])
                    + (split_scratch[r * K_SPLITS + 2u] + split_scratch[r * K_SPLITS + 3u])
                    + (split_scratch[r * K_SPLITS + 4u] + split_scratch[r * K_SPLITS + 5u])
                    + (split_scratch[r * K_SPLITS + 6u] + split_scratch[r * K_SPLITS + 7u]);
            y[row] = t;
        }
    }
}
