// Broadcast bias add, in-place: `x[t*dim + j] += bias[j]`. One Slang source for
// both GPU backends, ported from the handwritten pair it replaces
// (`shaders/bias_add.wgsl`, `shaders/bias_add.metal`), preserving both contracts
// exactly so no call site changes.
//
// Linear-layer bias is a single [dim] vector added to every one of the token
// rows; a plain elementwise add cannot express the broadcast. This kernel has no
// per-target divergence (no reduction, no subgroup op), so the whole body is
// written once with no `__target_switch`.
//
//   binding/buffer 0: x       f32, read-write, bias added in place (rows*dim)
//   binding/buffer 1: bias    f32, read (dim)
//   binding/buffer 2: params  (total = rows*dim, dim)
//
// `[[vk::binding(n)]]` drives the WGSL binding index and `register(u0/t1/t2)` the
// Metal buffer index; they are numbered independently because the two backends
// count slots from different spaces, and here they agree.

[[vk::binding(0)]] RWStructuredBuffer<float> x_buf    : register(u0);
[[vk::binding(1)]] StructuredBuffer<float>   bias_buf : register(t1);
[[vk::binding(2)]] StructuredBuffer<uint2>   par_buf  : register(t2);

[shader("compute")]
[numthreads(256, 1, 1)]
void bias_add(uint3 gid : SV_DispatchThreadID) {
    uint i     = gid.x;
    uint total = par_buf[0].x;
    uint dim   = par_buf[0].y;
    if (i >= total) {
        return;
    }
    x_buf[i] = x_buf[i] + bias_buf[i % dim];
}
