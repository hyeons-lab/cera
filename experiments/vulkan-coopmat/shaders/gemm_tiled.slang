// Baseline int8 GEMM WITHOUT cooperative matrix: shared-memory tiled, one thread
// per C element, 16×16 workgroup tiles. Represents a competent hand-written WGSL
// kernel (the class cera's #267 register-tiled GEMM is in) so the coopmat speedup
// is measured against a fair baseline, not a naive one-thread-per-element loop.

[vk::binding(0)] StructuredBuffer<int8_t> A; // M×K row-major
[vk::binding(1)] StructuredBuffer<int8_t> B; // K×N row-major
[vk::binding(2)] RWStructuredBuffer<int32_t> C; // M×N row-major

struct Params { uint M; uint N; uint K; };
[vk::push_constant] ConstantBuffer<Params> pc;

static const uint TILE = 16;
groupshared int32_t tileA[TILE][TILE];
groupshared int32_t tileB[TILE][TILE];

[shader("compute")]
[numthreads(TILE, TILE, 1)]
void computeMain(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    uint row = gid.y * TILE + lid.y;
    uint col = gid.x * TILE + lid.x;

    int32_t acc = 0;
    uint nTiles = (pc.K + TILE - 1) / TILE;
    for (uint t = 0; t < nTiles; t++)
    {
        uint aCol = t * TILE + lid.x;
        uint bRow = t * TILE + lid.y;
        tileA[lid.y][lid.x] = (row < pc.M && aCol < pc.K) ? int32_t(A[row * pc.K + aCol]) : 0;
        tileB[lid.y][lid.x] = (bRow < pc.K && col < pc.N) ? int32_t(B[bRow * pc.N + col]) : 0;
        GroupMemoryBarrierWithGroupSync();

        for (uint k = 0; k < TILE; k++)
            acc += tileA[lid.y][k] * tileB[k][lid.x];
        GroupMemoryBarrierWithGroupSync();
    }

    if (row < pc.M && col < pc.N)
        C[row * pc.N + col] = acc;
}
