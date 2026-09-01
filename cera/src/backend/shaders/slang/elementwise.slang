// slang-entries: add_inplace scaled_add_inplace mul_inplace silu_mul_inplace
//
// Element-wise f32 operations, one Slang source for both GPU backends. Only the
// four entry points whose WGSL and MSL twins are identical are migrated here:
// `add_inplace`, `scaled_add_inplace`, `mul_inplace`, `silu_mul_inplace`. They
// share one binding contract (a in-place, b read, params) and have no per-target
// divergence, so the bodies are written once with no `__target_switch`.
//
// The handwritten `elementwise.metal` also carries four Metal-only entry points
// (`memcpy_f32`, `mul_out`, `cast_f32_to_f16`, `scale_f32`) that have no WGSL
// twin; `cast_f32_to_f16` in particular would force `enable f16` into the WGSL
// emission on adapters that never requested it. Those stay handwritten and are
// deliberately NOT part of this single-source port.
//
//   binding/buffer 0: a       f32, read-write, modified in place (n elements)
//   binding/buffer 1: b       f32, read (n elements)
//   binding/buffer 2: params  (n, scale_bits): scale_bits is the raw f32 bits of
//                             the scaled_add multiplier, unused by the others
//
// Dispatch: (ceil(n / 256), 1, 1) workgroups of 256.

[[vk::binding(0)]] RWStructuredBuffer<float> a_buf   : register(u0);
[[vk::binding(1)]] StructuredBuffer<float>   b_buf   : register(t1);
[[vk::binding(2)]] StructuredBuffer<uint2>   par_buf : register(t2);

[shader("compute")]
[numthreads(256, 1, 1)]
void add_inplace(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= par_buf[0].x) {
        return;
    }
    a_buf[i] = a_buf[i] + b_buf[i];
}

// Residual add with a scalar on the addend: a[i] += s * b[i], with `s` the f32
// bits in params.y. Granite 3.x sets `s` to its residual multiplier; every other
// arch passes s = 1.0, making this identical to `add_inplace`.
[shader("compute")]
[numthreads(256, 1, 1)]
void scaled_add_inplace(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= par_buf[0].x) {
        return;
    }
    a_buf[i] = a_buf[i] + asfloat(par_buf[0].y) * b_buf[i];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= par_buf[0].x) {
        return;
    }
    a_buf[i] = a_buf[i] * b_buf[i];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void silu_mul_inplace(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= par_buf[0].x) {
        return;
    }
    float g = clamp(a_buf[i], -80.0f, 80.0f);
    a_buf[i] = (g / (1.0f + exp(-g))) * b_buf[i];
}
