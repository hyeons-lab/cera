// Faithful Slang port of cera's mul_mat_reg_tile.wgsl (Q4_0 loader), for an
// offline SPIR-V codegen comparison against naga-24 / naga-30. Same tiling
// (16x16 workgroup, TILE_M=TILE_N=4, TILE_K=16), same k-major staging, same
// helper decomposition, same 4x vec4 register accumulators.

struct MulMatParams {
    uint m;
    uint k;
    uint n;
    uint x_stride;
    uint y_stride;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>  src0;
[[vk::binding(1, 0)]] StructuredBuffer<float> src1;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> dst;
[[vk::binding(3, 0)]] StructuredBuffer<MulMatParams> paramsBuf;

static const uint WG_M = 16;
static const uint WG_N = 16;
static const uint TOTAL_WORKGROUP_SIZE = 256;
static const uint TILE_ROWS = 64; // 16 * 4
static const uint TILE_COLS = 64; // 16 * 4
static const uint SA_STRIDE = 68; // TILE_ROWS + 4
static const uint SB_STRIDE = 68; // TILE_COLS + 4

static const uint Q4_0_BLOCK_SIZE = 32;
static const uint Q4_0_BLOCK_BYTES = 18;
static const uint Q4_0_PER_THREAD = 8;

groupshared float sa[16 * SA_STRIDE];
groupshared float sb[16 * SB_STRIDE];

uint get_byte(uint value, uint index) {
    return (value >> (index * 8)) & 0xFFu;
}

uint load_src0_u32_at(uint byte_offset) {
    uint word_idx = byte_offset / 4u;
    uint shift = (byte_offset & 3u) * 8u;
    uint lo = src0[word_idx];
    if (shift == 0u) {
        return lo;
    }
    uint hi = src0[word_idx + 1u];
    return (lo >> shift) | (hi << (32u - shift));
}

float load_src0_f32_at(uint byte_offset) {
    uint word = src0[byte_offset / 4u];
    uint h16 = (word >> ((byte_offset & 2u) * 8u)) & 0xFFFFu;
    return f16tof32(h16);
}

void store_sa(uint tile_m, uint tile_k, float value) {
    sa[tile_k * SA_STRIDE + tile_m] = value;
}

void init_shmem_src0(uint thread_id, uint offset_m, uint k_outer) {
    let P = paramsBuf[0];
    uint blocks_k = (P.k + Q4_0_BLOCK_SIZE - 1u) / Q4_0_BLOCK_SIZE;

    for (uint i = thread_id * Q4_0_PER_THREAD;
         i < TILE_ROWS * 16u;
         i += TOTAL_WORKGROUP_SIZE * Q4_0_PER_THREAD) {
        uint tile_m = i / 16u;
        uint tile_k = i % 16u;
        uint global_m = offset_m + tile_m;
        uint global_k = k_outer + tile_k;

        uint w = global_k % Q4_0_BLOCK_SIZE;
        uint base = (global_m * blocks_k + global_k / Q4_0_BLOCK_SIZE) * Q4_0_BLOCK_BYTES;
        // Gate every src0 read on the row/col being in range, not just `d`.
        // Overhang threads on a ragged tile compute a `base` past the buffer, and
        // under SPIR-V passthrough there is no naga-injected bounds check, so an
        // unconditional load would be a real out-of-bounds read (the masked-off
        // result is discarded below, but the load still happens). Matches the
        // Q8_0 kernel, which gates all of its src0 reads the same way.
        bool in_bounds = global_m < P.m && global_k < P.k;
        float d = in_bounds ? load_src0_f32_at(base) : 0.0;
        uint q_lo = in_bounds ? load_src0_u32_at(base + 2u + (w % 16u)) : 0u;
        uint q_hi = in_bounds ? load_src0_u32_at(base + 6u + (w % 16u)) : 0u;

        for (uint j = 0u; j < Q4_0_PER_THREAD; j++) {
            uint byte = (j >= 4u) ? get_byte(q_hi, j & 3u) : get_byte(q_lo, j & 3u);
            uint nib = (w >= 16u) ? (byte >> 4u) : (byte & 0xFu);
            bool live = global_m < P.m && (global_k + j) < P.k;
            store_sa(tile_m, tile_k + j, live ? (float(nib) - 8.0) * d : 0.0);
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

    // cera dispatches this kernel as (wg_m, wg_n, 1), so the WGSL's
    // wg_id.y*num_wg.x+wg_id.x linearization reduces to exactly this:
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
