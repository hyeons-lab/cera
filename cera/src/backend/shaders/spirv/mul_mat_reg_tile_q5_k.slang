// Slang port of cera's mul_mat_reg_tile.wgsl with the Q5_K loader, for the wgpu
// SPIR-V passthrough path (bypasses naga-30's codegen regression on PowerVR).
// Identical tiling/inner-loop to the q4_0/q8_0 kernels; only init_shmem_src0 and
// the 6-bit scale/min unpack differ. Faithful to the WGSL INIT_SRC0_SHMEM_Q5_K
// loader in mul_mat_decls.tmpl.
//
// Q5_K super-block: 256 elems / 176 B: d f16 @0, dmin f16 @2,
// scales[12] (6-bit packed sub-scales + mins, identical to Q4_K) @4,
// qh[32] (the 5th-bit plane) @16, qs[128] (low 4 bits) @48.
//   out[64j + l]      = d*sc[2j]   * ((qs[32j+l] & 0xF) + fifth) - dmin*mn[2j]
//   out[64j + l + 32] = d*sc[2j+1] * ((qs[32j+l] >> 4 ) + fifth) - dmin*mn[2j+1]
// where `fifth` is 16 when the qh bit for that element is set, else 0. The qh
// plane is indexed by `l` (= y%32) alone; the same 32 bytes serve all `j`
// sub-blocks, consuming bit `sub` (= 2j for the low nibble, 2j+1 for the high).
// Inverting per output index y: j = y/64, r = y%64, low nibble and sub-block 2j
// for r < 32, else the high nibble and 2j+1; l = y%32.

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

static const uint TOTAL_WORKGROUP_SIZE = 256;
static const uint TILE_ROWS = 64; // 16 * 4
static const uint TILE_COLS = 64; // 16 * 4
static const uint TILE_K = 16;
static const uint SA_STRIDE = 68; // TILE_ROWS + 4
static const uint SB_STRIDE = 68; // TILE_COLS + 4

static const uint Q5K_BLOCK_SIZE = 256;
static const uint Q5K_BLOCK_BYTES = 176;

groupshared float sa[16 * SA_STRIDE];
groupshared float sb[16 * SB_STRIDE];

float load_src0_f32_at(uint byte_offset) {
    uint word = src0[byte_offset / 4u];
    uint h16 = (word >> ((byte_offset & 2u) * 8u)) & 0xFFFFu;
    return f16tof32(h16);
}

uint load_src0_byte_at(uint byte_offset) {
    uint word = src0[byte_offset / 4u];
    return (word >> ((byte_offset & 3u) * 8u)) & 0xFFu;
}

// 6-bit sub-scale / min unpack, identical to Q4_K's q4k_sc/q4k_mn.
uint q5k_sc(uint sb_base, uint sub) {
    if (sub < 4u) {
        return load_src0_byte_at(sb_base + sub) & 63u;
    }
    return (load_src0_byte_at(sb_base + sub + 4u) & 0x0Fu)
        | ((load_src0_byte_at(sb_base + sub - 4u) >> 6u) << 4u);
}

uint q5k_mn(uint sb_base, uint sub) {
    if (sub < 4u) {
        return load_src0_byte_at(sb_base + sub + 4u) & 63u;
    }
    return (load_src0_byte_at(sb_base + sub + 4u) >> 4u)
        | ((load_src0_byte_at(sb_base + sub) >> 6u) << 4u);
}

void store_sa(uint tile_m, uint tile_k, float value) {
    sa[tile_k * SA_STRIDE + tile_m] = value;
}

void init_shmem_src0(uint thread_id, uint offset_m, uint k_outer) {
    let P = paramsBuf[0];
    uint blocks_k = (P.k + Q5K_BLOCK_SIZE - 1u) / Q5K_BLOCK_SIZE;

    for (uint i = thread_id; i < TILE_ROWS * TILE_K; i += TOTAL_WORKGROUP_SIZE) {
        uint tile_m = i / TILE_K;
        uint tile_k = i % TILE_K;
        uint global_m = offset_m + tile_m;
        uint global_k = k_outer + tile_k;

        float v = 0.0;
        if (global_m < P.m && global_k < P.k) {
            uint base = (global_m * blocks_k + global_k / Q5K_BLOCK_SIZE) * Q5K_BLOCK_BYTES;
            uint y = global_k % Q5K_BLOCK_SIZE;
            uint j = y / 64u;
            uint r = y % 64u;
            bool hi = r >= 32u;
            uint sub = 2u * j + (hi ? 1u : 0u);
            uint l = y % 32u;
            uint sb_base = base + 4u;

            uint qb = load_src0_byte_at(base + 48u + 32u * j + l);
            uint nib = hi ? (qb >> 4u) : (qb & 0x0Fu);
            // 5th bit: qh byte at base+16+l, bit `sub`; contributes +16 when set.
            uint qhb = load_src0_byte_at(base + 16u + l);
            float fifth = float(((qhb >> sub) & 1u) * 16u);

            v = load_src0_f32_at(base) * float(q5k_sc(sb_base, sub)) * (float(nib) + fifth)
                - load_src0_f32_at(base + 2u) * float(q5k_mn(sb_base, sub));
        }
        store_sa(tile_m, tile_k, v);
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
