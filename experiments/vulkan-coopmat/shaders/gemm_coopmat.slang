// int8 cooperative-matrix GEMM: C[M,N] = A[M,K] * B[K,N], int8×int8 → int32.
// One subgroup per 64×16 output tile (M64 × N16), K stepped in 16 — matching the
// DXT-48 SINT8×SINT8→SINT32 config at M64×N16×K16.

using namespace linalg;

// Row-major storage buffers: A = M*K int8, B = K*N int8, C = M*N int32.
[vk::binding(0)] StructuredBuffer<int8_t> A;
[vk::binding(1)] StructuredBuffer<int8_t> B;
[vk::binding(2)] RWStructuredBuffer<int32_t> C;

struct Params { uint M; uint N; uint K; };
[vk::push_constant] ConstantBuffer<Params> pc;

typealias MatA = CoopMat<int8_t, MemoryScope.Subgroup, 64, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB = CoopMat<int8_t, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias MatC = CoopMat<int32_t, MemoryScope.Subgroup, 64, 16, CoopMatMatrixUse.MatrixAccumulator>;

// One full subgroup per output tile. DXT-48 subgroup size is 128, and cooperative
// matrix spans a whole subgroup, so the workgroup must be 128 lanes.
[shader("compute")]
[numthreads(128, 1, 1)]
void computeMain(uint3 gid : SV_GroupID)
{
    uint m0 = gid.y * 64;
    uint n0 = gid.x * 16;

    MatC acc = MatC(0);
    for (uint k = 0; k < pc.K; k += 16)
    {
        MatA a = MatA.Load<CoopMatMatrixLayout.RowMajor>(A, m0 * pc.K + k, pc.K);
        MatB b = MatB.Load<CoopMatMatrixLayout.RowMajor>(B, k * pc.N + n0, pc.N);
        acc = coopMatMulAdd<int32_t, false>(a, b, acc);
    }
    acc.Store<CoopMatMatrixLayout.RowMajor>(C, m0 * pc.N + n0, pc.N);
}
