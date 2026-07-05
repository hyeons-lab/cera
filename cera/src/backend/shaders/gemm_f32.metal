#include <metal_stdlib>
using namespace metal;

// Batched f32 GEMM for the prefill-path LoRA apply.
//
// The Metal prefill batch buffers are TOKEN-MAJOR (`[n_tokens × dim]`,
// row-major — token `i`'s channels are contiguous at `i*dim`), so the LoRA
// factors — `A` `[rank × k]` and `B` `[d × rank]`, both row-major — are consumed
// *transposed* (NT form): each output element is a dot of an lhs row with an rhs
// row. This mirrors the base projection GEMMs (`out[tok,o] = Σ W[o,i]·X[tok,i]`).
//
//   C[M×N] = Lhs[M×K] · Rhs[N×K]ᵀ   →   c[i*N+j] = Σ_p lhs[i*K+p] · rhs[j*K+p]
//
// One threadgroup (one 32-lane simdgroup) per output element; the K reduction is
// a `simd_sum`. A flat output index is split across a 2-D threadgroup grid
// (x + y*65535) so M*N can exceed the 65535 per-dimension cap. Correct-first-cut
// kernel — the LoRA GEMMs are tiny (rank ≤ 64), so this isn't a hot path.

struct GemmParams { uint m; uint n; uint k; };

kernel void gemm_f32_nt(
    const device float* lhs [[buffer(0)]],   // [M×K] row-major
    const device float* rhs [[buffer(1)]],   // [N×K] row-major (accessed transposed)
    device float* out [[buffer(2)]],         // [M×N] row-major (overwrite)
    constant GemmParams& params [[buffer(3)]],
    uint3 tid_v [[thread_position_in_threadgroup]],
    uint3 tg_id [[threadgroup_position_in_grid]]
) {
    uint tid = tid_v.x;
    uint idx = tg_id.x + tg_id.y * 65535u;
    uint total = params.m * params.n;
    if (idx >= total) return;
    uint i = idx / params.n; // lhs row (token)
    uint j = idx % params.n; // rhs row (output channel)
    uint k = params.k;
    uint lhs_off = i * k;
    uint rhs_off = j * k;
    float partial = 0.0f;
    for (uint c = tid; c < k; c += 32u) {
        partial += lhs[lhs_off + c] * rhs[rhs_off + c];
    }
    float total_sum = simd_sum(partial);
    if (tid == 0u) out[idx] = total_sum;
}

// Accumulate variant: `C += Lhs · Rhsᵀ`. Same layout/dispatch as `gemm_f32_nt`;
// used for the LoRA up-projection epilogue (`Y += B_scaled · Tmp`), where the
// `alpha/rank` scale is pre-folded into the uploaded `B` so there is no separate
// scale pass.
kernel void gemm_f32_nt_accum(
    const device float* lhs [[buffer(0)]],
    const device float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint3 tid_v [[thread_position_in_threadgroup]],
    uint3 tg_id [[threadgroup_position_in_grid]]
) {
    uint tid = tid_v.x;
    uint idx = tg_id.x + tg_id.y * 65535u;
    uint total = params.m * params.n;
    if (idx >= total) return;
    uint i = idx / params.n;
    uint j = idx % params.n;
    uint k = params.k;
    uint lhs_off = i * k;
    uint rhs_off = j * k;
    float partial = 0.0f;
    for (uint c = tid; c < k; c += 32u) {
        partial += lhs[lhs_off + c] * rhs[rhs_off + c];
    }
    float total_sum = simd_sum(partial);
    if (tid == 0u) out[idx] += total_sum;
}
