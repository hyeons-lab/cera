// Register-tiled Q4_0 GEMM over the resident stream layout: the fallthrough
// for prefill GEMMs the streaming kernel declines (n < 32, strided B).
// Same tiling as `mul_mat_reg_tile_q4_0` (16x16 workgroup, TILE_M=TILE_N=4,
// TILE_K=16, k-major staging, 4x vec4 register accumulators) — only the
// `init_shmem_src0` loader differs: it dequantizes from the resident (q, d)
// repack (see `repack_q4_0_stream` in gpu_lfm2.rs) instead of raw 18-byte
// blocks.
//
// Loader shape mirrors the raw kernel's, including its benign overspill:
// each thread stages 8 K values (global_k..global_k+7) but the MAC loop
// only reads the live 16-slice, so rows past tile_k 15 are never consumed.
// Live rows never cross a 32-block boundary (the live slice is
// k_outer..k_outer+15, always inside one block); the spill u32 past a
// block end is gated against the buffer size so the last block cannot
// fault. Small-n/strided prefill is rare and tiny — this kernel favors
// obvious correctness over the raw loader's branchless packing.

struct MulMatParams {
    uint m;
    uint k;
    uint n;
    uint x_stride;
    uint y_stride;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>  src_q;
[[vk::binding(1, 0)]] StructuredBuffer<uint>  src_d;
[[vk::binding(2, 0)]] StructuredBuffer<float> src1;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> dst;
[[vk::binding(4, 0)]] StructuredBuffer<MulMatParams> paramsBuf;

static const uint WG_M = 16;
static const uint WG_N = 16;
static const uint TOTAL_WORKGROUP_SIZE = 256;
static const uint TILE_ROWS = 64; // 16 * 4
static const uint TILE_COLS = 64; // 16 * 4
static const uint SA_STRIDE = 68; // TILE_ROWS + 4
static const uint SB_STRIDE = 68; // TILE_COLS + 4

static const uint Q4_0_BLOCK_SIZE = 32;
static const uint Q4_0_PER_THREAD = 8;

groupshared float sa[16 * SA_STRIDE];
groupshared float sb[16 * SB_STRIDE];

void store_sa(uint tile_m, uint tile_k, float value) {
    sa[tile_k * SA_STRIDE + tile_m] = value;
}

void init_shmem_src0(uint thread_id, uint offset_m, uint k_outer) {
    let P = paramsBuf[0];

    for (uint i = thread_id * Q4_0_PER_THREAD;
         i < TILE_ROWS * 16u;
         i += TOTAL_WORKGROUP_SIZE * Q4_0_PER_THREAD) {
        uint tile_m = i / 16u;
        uint tile_k = i % 16u;
        uint global_m = offset_m + tile_m;
        uint global_k = k_outer + tile_k;

        uint b = global_k / Q4_0_BLOCK_SIZE;
        uint w = global_k % Q4_0_BLOCK_SIZE;
        // Gate every weight read on the row/col being in range: overhang
        // threads on a ragged tile compute indices past the buffers, and
        // under SPIR-V passthrough there is no naga-injected bounds check.
        bool in_bounds = global_m < P.m && global_k < P.k;
        // This block's scale: d[(b/2)*m + row], lane b&1.
        uint du = in_bounds ? src_d[(b / 2u) * P.m + global_m] : 0u;
        uint dbits = ((b & 1u) != 0u) ? (du >> 16u) : (du & 0xFFFFu);
        float ds = f16tof32(dbits);
        // The two q u32s covering weights w..w+7 of this block (each u32
        // holds 8 sequential weights, low nibble first). The second spills
        // into the next block's first u32 when w % 8 != 0 and u0 == 3 —
        // gated, since that is past the buffer on the last block.
        uint nq = P.k / 8u;
        uint u0 = w / 8u;
        uint qi0 = b * 4u + u0;
        uint q0 = in_bounds ? src_q[qi0 * P.m + global_m] : 0u;
        uint qi1 = qi0 + 1u;
        uint q1 = (in_bounds && qi1 < nq) ? src_q[qi1 * P.m + global_m] : 0u;

        for (uint j = 0u; j < Q4_0_PER_THREAD; j++) {
            uint t = w + j;
            uint qu = (t / 8u == u0) ? q0 : q1;
            uint s = t % 8u;
            uint shift = (s % 4u) * 4u + (s / 4u) * 16u;
            uint nib = (qu >> shift) & 0xFu;
            bool live = global_m < P.m && (global_k + j) < P.k;
            store_sa(tile_m, tile_k + j, live ? (float(nib) - 8.0) * ds : 0.0);
        }
    }
}

void init_shmem_src1(uint thread_id, uint offset_n, uint k_outer) {
    let P = paramsBuf[0];
    for (uint i = thread_id; i < TILE_COLS * 16u; i += TOTAL_WORKGROUP_SIZE) {
        uint tile_n = i / 16u;
        uint tile_k = i % 16u;
        uint global_n = offset_n + tile_n;
        uint global_k = k_outer + tile_k;
        sb[tile_k * SB_STRIDE + tile_n] =
            (global_n < P.n && global_k < P.k) ? src1[global_n * P.x_stride + global_k] : 0.0;
    }
}

void store_col(uint col, uint row, float4 v) {
    let P = paramsBuf[0];
    if (col >= P.n) {
        return;
    }
    uint base = col * P.y_stride + row;
    if (row + 3u < P.m) {
        dst[base] = v.x;
        dst[base + 1u] = v.y;
        dst[base + 2u] = v.z;
        dst[base + 3u] = v.w;
    } else {
        if (row < P.m) { dst[base] = v.x; }
        if (row + 1u < P.m) { dst[base + 1u] = v.y; }
        if (row + 2u < P.m) { dst[base + 2u] = v.z; }
        if (row + 3u < P.m) { dst[base + 3u] = v.w; }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    uint3 wg_id : SV_GroupID,
    uint3 local_id : SV_GroupThreadID)
{
    let P = paramsBuf[0];
    uint thread_id = local_id.x;
    uint local_m = thread_id % 16u;
    uint local_n = thread_id / 16u;

    uint offset_m = wg_id.x * TILE_ROWS;
    uint offset_n = wg_id.y * TILE_COLS;

    float4 acc0 = float4(0.0);
    float4 acc1 = float4(0.0);
    float4 acc2 = float4(0.0);
    float4 acc3 = float4(0.0);

    uint m0 = local_m * 4u;
    uint n0 = local_n * 4u;

    for (uint k_outer = 0u; k_outer < P.k; k_outer += 16u) {
        init_shmem_src0(thread_id, offset_m, k_outer);
        init_shmem_src1(thread_id, offset_n, k_outer);

        GroupMemoryBarrierWithGroupSync();

        for (uint k_inner = 0u; k_inner < 16u; k_inner++) {
            uint ai = k_inner * SA_STRIDE + m0;
            uint bi = k_inner * SB_STRIDE + n0;
            float4 a = float4(sa[ai], sa[ai + 1u], sa[ai + 2u], sa[ai + 3u]);
            acc0 += a * sb[bi];
            acc1 += a * sb[bi + 1u];
            acc2 += a * sb[bi + 2u];
            acc3 += a * sb[bi + 3u];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    uint row = offset_m + m0;
    uint col = offset_n + n0;
    store_col(col, row, acc0);
    store_col(col + 1u, row, acc1);
    store_col(col + 2u, row, acc2);
    store_col(col + 3u, row, acc3);
}
