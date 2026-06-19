#include <metal_stdlib>
using namespace metal;

// Broadcast bias add, in-place (MSL mirror of bias_add.wgsl):
//   x[t*dim + j] += bias[j]
//
// Dispatch: threadgroups (ceil(total/256), 1, 1), threads (256, 1, 1).

struct Params { uint total; uint dim; };

kernel void bias_add(
    device float* x [[buffer(0)]],
    const device float* bias [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.total) return;
    x[gid] += bias[gid % p.dim];
}
