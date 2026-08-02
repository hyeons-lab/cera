// Slang port of cera's mul_mat_reg_tile.wgsl with the Q6_K loader, for the wgpu
// SPIR-V passthrough path (bypasses naga-30's codegen regression on PowerVR).
// Identical tiling/inner-loop to the q4_0/q8_0 kernels; only init_shmem_src0
// differs. Faithful to the WGSL INIT_SRC0_SHMEM_Q6_K loader in mul_mat_decls.tmpl.
//
// Q6_K super-block: 256 elems / 210 B: ql[128] @0, qh[64] @128,
// scales[16] (i8) @192, d f16 @208. The packing is *not* contiguous per output
// index, so we invert `dequantize_q6_k_block`'s (n, l, j) mapping per element:
//   y = n*128 + j*32 + l   with n in 0..2, j in 0..4, l in 0..32
//   ql byte : n*64 + l  (+32 when j is odd);  low nibble for j<2, high for j>=2
//   qh byte : n*32 + l;  bits [2j, 2j+1]
//   scale   : n*8 + l/16 + 2j   (signed i8)

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

static const uint Q6K_BLOCK_SIZE = 256;
static const uint Q6K_BLOCK_BYTES = 210;

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

void store_sa(uint tile_m, uint tile_k, float value) {
    sa[tile_k * SA_STRIDE + tile_m] = value;
}

void init_shmem_src0(uint thread_id, uint offset_m, uint k_outer) {
    let P = paramsBuf[0];
    uint blocks_k = (P.k + Q6K_BLOCK_SIZE - 1u) / Q6K_BLOCK_SIZE;

    for (uint i = thread_id; i < TILE_ROWS * TILE_K; i += TOTAL_WORKGROUP_SIZE) {
        uint tile_m = i / TILE_K;
        uint tile_k = i % TILE_K;
        uint global_m = offset_m + tile_m;
        uint global_k = k_outer + tile_k;

        float v = 0.0;
        if (global_m < P.m && global_k < P.k) {
            uint base = (global_m * blocks_k + global_k / Q6K_BLOCK_SIZE) * Q6K_BLOCK_BYTES;
            uint y = global_k % Q6K_BLOCK_SIZE;
            uint n = y / 128u;
            uint r = y % 128u;
            uint j = r / 32u;
            uint l = r % 32u;

            uint qlb = load_src0_byte_at(base + n * 64u + l + (((j & 1u) == 1u) ? 32u : 0u));
            uint qhb = load_src0_byte_at(base + 128u + n * 32u + l);
            uint nib = (j >= 2u) ? (qlb >> 4u) : (qlb & 0x0Fu);
            int q = int(nib | (((qhb >> (2u * j)) & 3u) << 4u)) - 32;

            // Sub-scales are signed 8-bit; sign-extend via a 24-bit arithmetic shift.
            uint sc_b = load_src0_byte_at(base + 192u + n * 8u + (l / 16u) + 2u * j);
            int sc = int(sc_b << 24u) >> 24u;

            v = load_src0_f32_at(base + 208u) * float(sc) * float(q);
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
