#include <metal_stdlib>
using namespace metal;

// Batched f32 linear: y[tok, row] = dot(x[tok, :], w[row, :]).
//   w: [m, k] row-major (out_dim × in_dim — the MmapWeight linear layout)
//   x: [n, k] row-major (n tokens × in_dim)
//   y: [n, m] row-major (n tokens × out_dim)
//
// One threadgroup of 32 threads (one simdgroup) per (output feature, token).
// Mirrors gemv_f32.metal but with a token index, so the whole token batch runs
// in a single dispatch. No TILE_K restriction: the inner loop handles any k.
//
// Dispatch: threadgroups (m, n, 1), threads (32, 1, 1).

struct Params { uint m; uint k; uint n; uint _pad; };

kernel void vit_linear(
    const device float* w [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    uint3 tid_v [[thread_position_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]
) {
    uint tid = tid_v.x;
    uint row = tg.x;
    uint tok = tg.y;
    if (row >= p.m || tok >= p.n) return;

    uint w_off = row * p.k;
    uint x_off = tok * p.k;
    float partial = 0.0f;
    for (uint c = tid; c < p.k; c += 32u) {
        partial += w[w_off + c] * x[x_off + c];
    }
    float total = simd_sum(partial);
    if (tid == 0u) {
        y[tok * p.m + row] = total;
    }
}
