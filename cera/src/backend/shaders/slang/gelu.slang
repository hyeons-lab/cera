// slang-entries: gelu_inplace
//
// tanh-approximation GELU, in-place over an f32 buffer. One Slang source for
// both GPU backends, ported from the handwritten pair it replaces
// (`shaders/gelu.wgsl`, `shaders/gelu.metal`), preserving both contracts exactly
// so no call site changes.
//
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Mirrors `cpu::gelu_inplace` (ggml's default GELU, what CLIP-family ViTs trained
// with `clip.use_gelu = true` expect), the tanh approximation, NOT the erf form.
// No reduction and no subgroup op, so the whole body is one branch with no
// `__target_switch`.
//
//   binding/buffer 0: x       f32, read-write, activated in place (n elements)
//   binding/buffer 1: params  (n, unused)
//
// `[[vk::binding(n)]]` drives the WGSL binding index and `register(u0/t1)` the
// Metal buffer index; they are numbered independently and here they agree.
//
// Dispatch: (ceil(n / 256), 1, 1) workgroups of 256.

[[vk::binding(0)]] RWStructuredBuffer<float> x_buf   : register(u0);
[[vk::binding(1)]] StructuredBuffer<uint2>   par_buf : register(t1);

[shader("compute")]
[numthreads(256, 1, 1)]
void gelu_inplace(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    uint n = par_buf[0].x;
    if (i >= n) {
        return;
    }
    float xv = x_buf[i];
    // Clamp the tanh argument: tanh saturates to +/-1 by |arg| ~ 15, but a GPU
    // tanh computed as (exp(2a)-1)/(exp(2a)+1) overflows to inf/inf = NaN for
    // large `a` (the cubic term makes `a` ~ 180 for |x| ~ 17). Clamping is
    // numerically identical to the CPU f32 tanh on the saturated tail.
    float inner = clamp(0.7978845608f * (xv + 0.044715f * xv * xv * xv), -15.0f, 15.0f);
    x_buf[i] = 0.5f * xv * (1.0f + tanh(inner));
}
