// Cooperative-matrix (16x16x16 F16->F32) direct-B streaming Q4_0 prefill GEMM:
// 128x64 output tile, BK=32 sub-tiling (15.36 KB staging LDS + 4 KB edge LDS), WG=256.
// Wave64 => 4 subgroups in 4x1 grid (32x64/wave, acc[8]);
// Wave32 => 8 subgroups in 4x2 grid (32x32/wave, acc[4]).
using namespace linalg;

struct StreamParams {
    uint m;
    uint k;
    uint n_valid;
    uint n_pad;
    uint y_stride;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint> src_q;
[[vk::binding(1, 0)]] StructuredBuffer<uint> src_d;
[[vk::binding(2, 0)]] StructuredBuffer<float4> src_bf;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> dst;
[[vk::binding(4, 0)]] StructuredBuffer<StreamParams> paramsBuf;

static const uint TILE_M = 128u;
static const uint TILE_N = 64u;
static const uint SH_STRIDE = 40u;

groupshared half sh_a[TILE_M * SH_STRIDE];
groupshared half sh_b[TILE_N * SH_STRIDE];
groupshared float sh_edge[4u * 256u];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lid : SV_GroupThreadID) {
    let P = paramsBuf[0];
    uint row_base = gid.x * TILE_M;
    uint col_base = gid.y * TILE_N;

    uint sg_size = WaveGetLaneCount();
    uint n_sg = 256u / sg_size;
    uint sg = lid / sg_size;
    uint lane = WaveGetLaneIndex();

    typealias MatA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
    typealias MatB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
    typealias MatC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

    MatC acc[8];
    [ForceUnroll]
    for (uint i = 0u; i < 8u; i++) {
        acc[i] = MatC(0.0f);
    }

    uint n_kb = P.k / 64u;
    uint r_a = lid % 128u;
    uint p_half = lid / 128u; // 0 = p0..1, 1 = p2..3 (each thread decodes 2 p's = 16 nibbles)
    uint row = row_base + r_a;
    uint rrow = row < P.m ? row : 0u;
    uint a_dst_base = r_a * SH_STRIDE + p_half * 16u;

    uint k_v4 = P.k >> 2u;
    uint k_g4 = lid % 8u;       // 8 float4s cover 32 floats
    uint k_g = k_g4 * 4u;
    uint n_idx = lid / 8u;      // 32 rows per pass -> 2 passes for 64 N rows
    uint last_col = P.n_valid - 1u;

    for (uint kb = 0u; kb < n_kb; kb++) {
        uint pd = src_d[kb * P.m + rrow];
        half dw0 = half(f16tof32(pd & 0xFFFFu));
        half dw1 = half(f16tof32(pd >> 16u));

        // Sub-tile 0: k[0..31]
        {
            uint p_base = (kb * 8u + p_half * 2u) * P.m + rrow;
            [ForceUnroll]
            for (uint p = 0u; p < 2u; p++) {
                uint pq = src_q[p_base + p * P.m];
                uint a_idx = a_dst_base + p * 8u;
                [ForceUnroll]
                for (uint t = 0u; t < 8u; t++) {
                    sh_a[a_idx + t] = (half((pq >> (t * 4u)) & 0xFu) - half(8.0h)) * dw0;
                }
            }
            uint b_k4 = kb * 16u + k_g4;
            [ForceUnroll]
            for (uint s = 0u; s < 2u; s++) {
                uint local_n = s * 32u + n_idx;
                uint tcol = min(col_base + local_n, last_col);
                float4 bv = src_bf[tcol * k_v4 + b_k4];
                uint b_dst = local_n * SH_STRIDE + k_g;
                sh_b[b_dst + 0u] = half(bv.x);
                sh_b[b_dst + 1u] = half(bv.y);
                sh_b[b_dst + 2u] = half(bv.z);
                sh_b[b_dst + 3u] = half(bv.w);
            }
            GroupMemoryBarrierWithGroupSync();

            if (n_sg == 4u) {
                uint mt = sg * 32u;
                [ForceUnroll]
                for (uint kk = 0u; kk < 32u; kk += 16u) {
                    MatA ma0 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a, mt * SH_STRIDE + kk, SH_STRIDE);
                    MatA ma1 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a, (mt + 16u) * SH_STRIDE + kk, SH_STRIDE);
                    [ForceUnroll]
                    for (uint nt = 0u; nt < 4u; nt++) {
                        MatB mb = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sh_b, (nt * 16u) * SH_STRIDE + kk, SH_STRIDE);
                        acc[nt * 2u + 0u] = coopMatMulAdd<float, false>(ma0, mb, acc[nt * 2u + 0u]);
                        acc[nt * 2u + 1u] = coopMatMulAdd<float, false>(ma1, mb, acc[nt * 2u + 1u]);
                    }
                }
            } else {
                uint sg_m = (sg & 3u) * 32u;
                uint sg_n = (sg >> 2u) * 32u;
                [ForceUnroll]
                for (uint kk = 0u; kk < 32u; kk += 16u) {
                    MatA ma0 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a, sg_m * SH_STRIDE + kk, SH_STRIDE);
                    MatA ma1 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a, (sg_m + 16u) * SH_STRIDE + kk, SH_STRIDE);
                    MatB mb0 = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sh_b, sg_n * SH_STRIDE + kk, SH_STRIDE);
                    MatB mb1 = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sh_b, (sg_n + 16u) * SH_STRIDE + kk, SH_STRIDE);
                    acc[0] = coopMatMulAdd<float, false>(ma0, mb0, acc[0]);
                    acc[1] = coopMatMulAdd<float, false>(ma0, mb1, acc[1]);
                    acc[2] = coopMatMulAdd<float, false>(ma1, mb0, acc[2]);
                    acc[3] = coopMatMulAdd<float, false>(ma1, mb1, acc[3]);
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // Sub-tile 1: k[32..63]
        {
            uint p_base = (kb * 8u + 4u + p_half * 2u) * P.m + rrow;
            [ForceUnroll]
            for (uint p = 0u; p < 2u; p++) {
                uint pq = src_q[p_base + p * P.m];
                uint a_idx = a_dst_base + p * 8u;
                [ForceUnroll]
                for (uint t = 0u; t < 8u; t++) {
                    sh_a[a_idx + t] = (half((pq >> (t * 4u)) & 0xFu) - half(8.0h)) * dw1;
                }
            }
            uint b_k4 = kb * 16u + 8u + k_g4;
            [ForceUnroll]
            for (uint s = 0u; s < 2u; s++) {
                uint local_n = s * 32u + n_idx;
                uint tcol = min(col_base + local_n, last_col);
                float4 bv = src_bf[tcol * k_v4 + b_k4];
                uint b_dst = local_n * SH_STRIDE + k_g;
                sh_b[b_dst + 0u] = half(bv.x);
                sh_b[b_dst + 1u] = half(bv.y);
                sh_b[b_dst + 2u] = half(bv.z);
                sh_b[b_dst + 3u] = half(bv.w);
            }
            GroupMemoryBarrierWithGroupSync();

            if (n_sg == 4u) {
                uint mt = sg * 32u;
                [ForceUnroll]
                for (uint kk = 0u; kk < 32u; kk += 16u) {
                    MatA ma0 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a, mt * SH_STRIDE + kk, SH_STRIDE);
                    MatA ma1 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a, (mt + 16u) * SH_STRIDE + kk, SH_STRIDE);
                    [ForceUnroll]
                    for (uint nt = 0u; nt < 4u; nt++) {
                        MatB mb = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sh_b, (nt * 16u) * SH_STRIDE + kk, SH_STRIDE);
                        acc[nt * 2u + 0u] = coopMatMulAdd<float, false>(ma0, mb, acc[nt * 2u + 0u]);
                        acc[nt * 2u + 1u] = coopMatMulAdd<float, false>(ma1, mb, acc[nt * 2u + 1u]);
                    }
                }
            } else {
                uint sg_m = (sg & 3u) * 32u;
                uint sg_n = (sg >> 2u) * 32u;
                [ForceUnroll]
                for (uint kk = 0u; kk < 32u; kk += 16u) {
                    MatA ma0 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a, sg_m * SH_STRIDE + kk, SH_STRIDE);
                    MatA ma1 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a, (sg_m + 16u) * SH_STRIDE + kk, SH_STRIDE);
                    MatB mb0 = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sh_b, sg_n * SH_STRIDE + kk, SH_STRIDE);
                    MatB mb1 = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sh_b, (sg_n + 16u) * SH_STRIDE + kk, SH_STRIDE);
                    acc[0] = coopMatMulAdd<float, false>(ma0, mb0, acc[0]);
                    acc[1] = coopMatMulAdd<float, false>(ma0, mb1, acc[1]);
                    acc[2] = coopMatMulAdd<float, false>(ma1, mb0, acc[2]);
                    acc[3] = coopMatMulAdd<float, false>(ma1, mb1, acc[3]);
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    bool full_tile = (row_base + TILE_M <= P.m) && (col_base + TILE_N <= P.n_valid);
    if (full_tile && n_sg == 4u) {
        uint mt = sg * 32u;
        [ForceUnroll]
        for (uint nt = 0u; nt < 4u; nt++) {
            acc[nt * 2u + 0u].Store<CoopMatMatrixLayout.ColumnMajor>(dst, (col_base + nt * 16u) * P.y_stride + (row_base + mt), P.y_stride);
            acc[nt * 2u + 1u].Store<CoopMatMatrixLayout.ColumnMajor>(dst, (col_base + nt * 16u) * P.y_stride + (row_base + mt + 16u), P.y_stride);
        }
    } else if (full_tile && n_sg == 8u) {
        uint sg_m = (sg & 3u) * 32u;
        uint sg_n = (sg >> 2u) * 32u;
        acc[0].Store<CoopMatMatrixLayout.ColumnMajor>(dst, (col_base + sg_n) * P.y_stride + (row_base + sg_m), P.y_stride);
        acc[1].Store<CoopMatMatrixLayout.ColumnMajor>(dst, (col_base + sg_n + 16u) * P.y_stride + (row_base + sg_m), P.y_stride);
        acc[2].Store<CoopMatMatrixLayout.ColumnMajor>(dst, (col_base + sg_n) * P.y_stride + (row_base + sg_m + 16u), P.y_stride);
        acc[3].Store<CoopMatMatrixLayout.ColumnMajor>(dst, (col_base + sg_n + 16u) * P.y_stride + (row_base + sg_m + 16u), P.y_stride);
    } else {
        uint count = (n_sg == 4u) ? 8u : 4u;
        uint sg_m = (n_sg == 4u) ? (sg * 32u) : ((sg & 3u) * 32u);
        uint sg_n = (n_sg == 4u) ? 0u : ((sg >> 2u) * 32u);
        uint phases = (n_sg > 4u) ? 2u : 1u;
        for (uint ph = 0u; ph < phases; ph++) {
            bool my_phase = (sg >> 2u) == ph;
            uint sg_slot = (sg & 3u) * 256u;
            for (uint ti = 0u; ti < count; ti++) {
                uint dm = (n_sg == 4u) ? ((ti & 1u) * 16u) : ((ti >> 1u) * 16u);
                uint dn = (n_sg == 4u) ? ((ti >> 1u) * 16u) : ((ti & 1u) * 16u);
                uint tile_r = row_base + sg_m + dm;
                uint tile_c = col_base + sg_n + dn;
                if (my_phase) {
                    if (tile_r + 16u <= P.m && tile_c + 16u <= P.n_valid) {
                        acc[ti].Store<CoopMatMatrixLayout.ColumnMajor>(dst, tile_c * P.y_stride + tile_r, P.y_stride);
                    } else {
                        acc[ti].Store<CoopMatMatrixLayout.ColumnMajor>(sh_edge, sg_slot, 16u);
                    }
                }
                GroupMemoryBarrierWithGroupSync();
                if (my_phase && !(tile_r + 16u <= P.m && tile_c + 16u <= P.n_valid)) {
                    if (tile_r < P.m && tile_c < P.n_valid) {
                        for (uint idx = lane; idx < 256u; idx += sg_size) {
                            uint lr = idx & 15u;
                            uint lc = idx >> 4u;
                            if (tile_r + lr < P.m && tile_c + lc < P.n_valid) {
                                dst[(tile_c + lc) * P.y_stride + (tile_r + lr)] = sh_edge[sg_slot + idx];
                            }
                        }
                    }
                }
                GroupMemoryBarrierWithGroupSync();
            }
        }
    }
}
