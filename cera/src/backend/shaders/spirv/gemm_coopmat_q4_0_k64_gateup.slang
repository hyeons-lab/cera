// Cooperative-matrix (16x16x16 F16->F32) fused Gate+Up+SiLU streaming Q4_0 prefill GEMM:
// 64x64 output tile, BK=32 sub-tiling, WG=256 (15.36 KB staging LDS + 4 KB edge LDS).
// Loads B once into sh_b for both Gate and Up projections.
using namespace linalg;

struct StreamParams {
    uint m;
    uint k;
    uint n_valid;
    uint n_pad;
    uint y_stride;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint> gate_q;
[[vk::binding(1, 0)]] StructuredBuffer<uint> gate_d;
[[vk::binding(2, 0)]] StructuredBuffer<uint> up_q;
[[vk::binding(3, 0)]] StructuredBuffer<uint> up_d;
[[vk::binding(4, 0)]] StructuredBuffer<float4> src_bf;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> dst;
[[vk::binding(6, 0)]] StructuredBuffer<StreamParams> paramsBuf;

static const uint TILE_M = 64u;
static const uint TILE_N = 64u;
static const uint SH_STRIDE = 40u;

groupshared half sh_a_g[TILE_M * SH_STRIDE];
groupshared half sh_a_u[TILE_M * SH_STRIDE];
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

    MatC acc_g[4];
    MatC acc_u[4];
    [ForceUnroll]
    for (uint i = 0u; i < 4u; i++) {
        acc_g[i] = MatC(0.0f);
        acc_u[i] = MatC(0.0f);
    }

    uint n_kb = P.k / 64u;
    uint r_a = lid % 64u;
    uint p_sub = lid / 64u;
    uint row = row_base + r_a;
    uint rrow = row < P.m ? row : 0u;
    uint a_dst_base = r_a * SH_STRIDE + p_sub * 8u;

    uint k_v4 = P.k >> 2u;
    uint k_g4 = lid % 8u;
    uint k_g = k_g4 * 4u;
    uint n_idx = lid / 8u;
    uint last_col = P.n_valid - 1u;

    for (uint kb = 0u; kb < n_kb; kb++) {
        uint pgd = gate_d[kb * P.m + rrow];
        half dgw0 = half(f16tof32(pgd & 0xFFFFu));
        half dgw1 = half(f16tof32(pgd >> 16u));
        uint pud = up_d[kb * P.m + rrow];
        half duw0 = half(f16tof32(pud & 0xFFFFu));
        half duw1 = half(f16tof32(pud >> 16u));

        // Sub-tile 0: k[0..31]
        {
            uint gq = gate_q[(kb * 8u + p_sub) * P.m + rrow];
            uint uq = up_q[(kb * 8u + p_sub) * P.m + rrow];
            [ForceUnroll]
            for (uint t = 0u; t < 8u; t++) {
                uint sft = t * 4u;
                sh_a_g[a_dst_base + t] = (half((gq >> sft) & 0xFu) - half(8.0h)) * dgw0;
                sh_a_u[a_dst_base + t] = (half((uq >> sft) & 0xFu) - half(8.0h)) * duw0;
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
                uint sg_m = (sg & 1u) * 32u;
                uint sg_n = (sg >> 1u) * 32u;
                [ForceUnroll]
                for (uint kk = 0u; kk < 32u; kk += 16u) {
                    MatB mb0 = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sh_b, sg_n * SH_STRIDE + kk, SH_STRIDE);
                    MatB mb1 = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sh_b, (sg_n + 16u) * SH_STRIDE + kk, SH_STRIDE);
                    MatA mag0 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a_g, sg_m * SH_STRIDE + kk, SH_STRIDE);
                    MatA mag1 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a_g, (sg_m + 16u) * SH_STRIDE + kk, SH_STRIDE);
                    acc_g[0] = coopMatMulAdd<float, false>(mag0, mb0, acc_g[0]);
                    acc_g[1] = coopMatMulAdd<float, false>(mag0, mb1, acc_g[1]);
                    acc_g[2] = coopMatMulAdd<float, false>(mag1, mb0, acc_g[2]);
                    acc_g[3] = coopMatMulAdd<float, false>(mag1, mb1, acc_g[3]);
                    MatA mau0 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a_u, sg_m * SH_STRIDE + kk, SH_STRIDE);
                    MatA mau1 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a_u, (sg_m + 16u) * SH_STRIDE + kk, SH_STRIDE);
                    acc_u[0] = coopMatMulAdd<float, false>(mau0, mb0, acc_u[0]);
                    acc_u[1] = coopMatMulAdd<float, false>(mau0, mb1, acc_u[1]);
                    acc_u[2] = coopMatMulAdd<float, false>(mau1, mb0, acc_u[2]);
                    acc_u[3] = coopMatMulAdd<float, false>(mau1, mb1, acc_u[3]);
                }
            } else {
                uint sg_m = (sg & 3u) * 16u;
                uint sg_n = (sg >> 2u) * 32u;
                [ForceUnroll]
                for (uint kk = 0u; kk < 32u; kk += 16u) {
                    MatB mb0 = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sh_b, sg_n * SH_STRIDE + kk, SH_STRIDE);
                    MatB mb1 = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sh_b, (sg_n + 16u) * SH_STRIDE + kk, SH_STRIDE);
                    MatA mag = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a_g, sg_m * SH_STRIDE + kk, SH_STRIDE);
                    acc_g[0] = coopMatMulAdd<float, false>(mag, mb0, acc_g[0]);
                    acc_g[1] = coopMatMulAdd<float, false>(mag, mb1, acc_g[1]);
                    MatA mau = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a_u, sg_m * SH_STRIDE + kk, SH_STRIDE);
                    acc_u[0] = coopMatMulAdd<float, false>(mau, mb0, acc_u[0]);
                    acc_u[1] = coopMatMulAdd<float, false>(mau, mb1, acc_u[1]);
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // Sub-tile 1: k[32..63]
        {
            uint gq = gate_q[(kb * 8u + 4u + p_sub) * P.m + rrow];
            uint uq = up_q[(kb * 8u + 4u + p_sub) * P.m + rrow];
            [ForceUnroll]
            for (uint t = 0u; t < 8u; t++) {
                uint sft = t * 4u;
                sh_a_g[a_dst_base + t] = (half((gq >> sft) & 0xFu) - half(8.0h)) * dgw1;
                sh_a_u[a_dst_base + t] = (half((uq >> sft) & 0xFu) - half(8.0h)) * duw1;
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
                uint sg_m = (sg & 1u) * 32u;
                uint sg_n = (sg >> 1u) * 32u;
                [ForceUnroll]
                for (uint kk = 0u; kk < 32u; kk += 16u) {
                    MatB mb0 = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sh_b, sg_n * SH_STRIDE + kk, SH_STRIDE);
                    MatB mb1 = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sh_b, (sg_n + 16u) * SH_STRIDE + kk, SH_STRIDE);
                    MatA mag0 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a_g, sg_m * SH_STRIDE + kk, SH_STRIDE);
                    MatA mag1 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a_g, (sg_m + 16u) * SH_STRIDE + kk, SH_STRIDE);
                    acc_g[0] = coopMatMulAdd<float, false>(mag0, mb0, acc_g[0]);
                    acc_g[1] = coopMatMulAdd<float, false>(mag0, mb1, acc_g[1]);
                    acc_g[2] = coopMatMulAdd<float, false>(mag1, mb0, acc_g[2]);
                    acc_g[3] = coopMatMulAdd<float, false>(mag1, mb1, acc_g[3]);
                    MatA mau0 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a_u, sg_m * SH_STRIDE + kk, SH_STRIDE);
                    MatA mau1 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a_u, (sg_m + 16u) * SH_STRIDE + kk, SH_STRIDE);
                    acc_u[0] = coopMatMulAdd<float, false>(mau0, mb0, acc_u[0]);
                    acc_u[1] = coopMatMulAdd<float, false>(mau0, mb1, acc_u[1]);
                    acc_u[2] = coopMatMulAdd<float, false>(mau1, mb0, acc_u[2]);
                    acc_u[3] = coopMatMulAdd<float, false>(mau1, mb1, acc_u[3]);
                }
            } else {
                uint sg_m = (sg & 3u) * 16u;
                uint sg_n = (sg >> 2u) * 32u;
                [ForceUnroll]
                for (uint kk = 0u; kk < 32u; kk += 16u) {
                    MatB mb0 = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sh_b, sg_n * SH_STRIDE + kk, SH_STRIDE);
                    MatB mb1 = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sh_b, (sg_n + 16u) * SH_STRIDE + kk, SH_STRIDE);
                    MatA mag = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a_g, sg_m * SH_STRIDE + kk, SH_STRIDE);
                    acc_g[0] = coopMatMulAdd<float, false>(mag, mb0, acc_g[0]);
                    acc_g[1] = coopMatMulAdd<float, false>(mag, mb1, acc_g[1]);
                    MatA mau = MatA.Load<CoopMatMatrixLayout.RowMajor>(sh_a_u, sg_m * SH_STRIDE + kk, SH_STRIDE);
                    acc_u[0] = coopMatMulAdd<float, false>(mau, mb0, acc_u[0]);
                    acc_u[1] = coopMatMulAdd<float, false>(mau, mb1, acc_u[1]);
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    // Fused SiLU(gate) * up in registers, then store directly for full tiles or via sh_edge on boundaries
    uint count = (n_sg == 4u) ? 4u : 2u;
    uint sg_m = (n_sg == 4u) ? ((sg & 1u) * 32u) : ((sg & 3u) * 16u);
    uint sg_n = (n_sg == 4u) ? ((sg >> 1u) * 32u) : ((sg >> 2u) * 32u);
    bool full_tile = (row_base + TILE_M <= P.m) && (col_base + TILE_N <= P.n_valid);

    for (uint ti = 0u; ti < count; ti++) {
        uint len = acc_g[ti].GetLength();
        for (uint e = 0u; e < len; e++) {
            float g = clamp(acc_g[ti][e], -80.0f, 80.0f);
            acc_g[ti][e] = (g / (1.0f + exp(-g))) * acc_u[ti][e];
        }
    }

    if (full_tile && n_sg == 4u) {
        acc_g[0].Store<CoopMatMatrixLayout.ColumnMajor>(dst, (col_base + sg_n) * P.y_stride + (row_base + sg_m), P.y_stride);
        acc_g[1].Store<CoopMatMatrixLayout.ColumnMajor>(dst, (col_base + sg_n + 16u) * P.y_stride + (row_base + sg_m), P.y_stride);
        acc_g[2].Store<CoopMatMatrixLayout.ColumnMajor>(dst, (col_base + sg_n) * P.y_stride + (row_base + sg_m + 16u), P.y_stride);
        acc_g[3].Store<CoopMatMatrixLayout.ColumnMajor>(dst, (col_base + sg_n + 16u) * P.y_stride + (row_base + sg_m + 16u), P.y_stride);
    } else if (full_tile && n_sg == 8u) {
        acc_g[0].Store<CoopMatMatrixLayout.ColumnMajor>(dst, (col_base + sg_n) * P.y_stride + (row_base + sg_m), P.y_stride);
        acc_g[1].Store<CoopMatMatrixLayout.ColumnMajor>(dst, (col_base + sg_n + 16u) * P.y_stride + (row_base + sg_m), P.y_stride);
    } else {
        uint phases = (n_sg > 4u) ? 2u : 1u;
        for (uint ph = 0u; ph < phases; ph++) {
            bool my_phase = (sg >> 2u) == ph;
            uint sg_slot = (sg & 3u) * 256u;
            for (uint ti = 0u; ti < count; ti++) {
                uint dm = (n_sg == 4u) ? ((ti >> 1u) * 16u) : 0u;
                uint dn = (n_sg == 4u) ? ((ti & 1u) * 16u) : (ti * 16u);
                uint tile_r = row_base + sg_m + dm;
                uint tile_c = col_base + sg_n + dn;
                if (my_phase) {
                    if (tile_r + 16u <= P.m && tile_c + 16u <= P.n_valid) {
                        acc_g[ti].Store<CoopMatMatrixLayout.ColumnMajor>(dst, tile_c * P.y_stride + tile_r, P.y_stride);
                    } else {
                        acc_g[ti].Store<CoopMatMatrixLayout.ColumnMajor>(sh_edge, sg_slot, 16u);
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
