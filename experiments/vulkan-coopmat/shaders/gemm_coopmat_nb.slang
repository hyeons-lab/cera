// int8 coopmat GEMM, N-blocked: one subgroup computes a 64x(16*NB) output block,
// loading each 64x16 MatA fragment ONCE and reusing it across NB MatB tiles. This
// cuts A global traffic NB-fold vs gemm_coopmat.slang (which reloads A per N-tile)
// and gives NB independent MMA chains to hide coopMatLoad latency. Same DXT-48
// SINT8xSINT8->SINT32 config (M64 x N16 x K16), subgroup = 128 lanes.

using namespace linalg;

[vk::binding(0)] StructuredBuffer<int8_t> A;  // M x K row-major
[vk::binding(1)] StructuredBuffer<int8_t> B;  // K x N row-major
[vk::binding(2)] RWStructuredBuffer<int32_t> C; // M x N row-major

struct Params { uint M; uint N; uint K; };
[vk::push_constant] ConstantBuffer<Params> pc;

typealias MatA = CoopMat<int8_t, MemoryScope.Subgroup, 64, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB = CoopMat<int8_t, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias MatC = CoopMat<int32_t, MemoryScope.Subgroup, 64, 16, CoopMatMatrixUse.MatrixAccumulator>;

static const uint NB = 4; // N-tiles per subgroup (each 16 wide) -> 64-wide block

[shader("compute")]
[numthreads(128, 1, 1)]
void computeMain(uint3 gid : SV_GroupID)
{
    uint m0 = gid.y * 64;
    uint n0 = gid.x * (16 * NB);

    MatC acc[NB];
    [ForceUnroll] for (uint j = 0; j < NB; j++)
        acc[j] = MatC(0);

    for (uint k = 0; k < pc.K; k += 16)
    {
        MatA a = MatA.Load<CoopMatMatrixLayout.RowMajor>(A, m0 * pc.K + k, pc.K);
        [ForceUnroll] for (uint j = 0; j < NB; j++)
        {
            MatB b = MatB.Load<CoopMatMatrixLayout.RowMajor>(B, k * pc.N + (n0 + j * 16), pc.N);
            acc[j] = coopMatMulAdd<int32_t, false>(a, b, acc[j]);
        }
    }

    [ForceUnroll] for (uint j = 0; j < NB; j++)
        acc[j].Store<CoopMatMatrixLayout.RowMajor>(C, m0 * pc.N + (n0 + j * 16), pc.N);
}
