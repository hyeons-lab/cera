#include <metal_stdlib>
using namespace metal;

// tanh-approximation GELU, in-place (MSL mirror of gelu.wgsl):
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Matches cpu::gelu_inplace (ggml default / CLIP use_gelu), not the erf form.
//
// Dispatch: threadgroups (ceil(n/256), 1, 1), threads (256, 1, 1).

struct Params { uint n; uint _pad; };

kernel void gelu_inplace(
    device float* x [[buffer(0)]],
    constant Params& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.n) return;
    float xv = x[gid];
    // Clamp the tanh argument: tanh saturates to ±1 by |arg|≈15, but a GPU
    // tanh computed as (exp(2a)-1)/(exp(2a)+1) overflows to inf/inf = NaN for
    // large `a` (the cubic term makes `a` ~180 for |x|~17). Clamping is
    // numerically identical to the CPU f32::tanh on the saturated tail.
    float inner = clamp(0.7978845608f * (xv + 0.044715f * xv * xv * xv), -15.0f, 15.0f);
    x[gid] = 0.5f * xv * (1.0f + tanh(inner));
}
