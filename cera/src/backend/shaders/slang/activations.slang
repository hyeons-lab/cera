// slang-entries: relu_inplace silu_inplace gelu_erf_inplace
//
// Unary in-place f32 activations, one Slang source for both GPU backends. These
// three share `gelu.slang`'s binding contract (buffer in place at 0, params at
// 1) but live in their own file so the ViT's tanh-GELU kernel keeps its
// single-purpose source and does not have to be regenerated to add an unrelated
// entry point.
//
// All three are on the LFM2A audio-encoder path:
//   - `relu_inplace`     between the conv subsampling stem's layers,
//   - `silu_inplace`     inside each Conformer macaron FFN,
//   - `gelu_erf_inplace` in the MLP adapter that projects to the LLM's width.
//
// **`gelu_erf_inplace` is the erf form, not the tanh form.** `gelu.slang`
// computes the tanh approximation that CLIP-family ViTs expect; the audio
// adapter was trained against the exact erf GELU, and the two differ by ~1e-3
// relative around |x| ~ 1. Picking the wrong one degrades output measurably even
// though any single call looks fine (see the note on `cpu::gelu_inplace`).
//
// `erf` has no WGSL or MSL builtin, so it is the same Abramowitz & Stegun 7.1.26
// approximation `cpu::erff` uses, with the same f32-truncated constants (max
// abs error ~1.5e-7). The CPU feeds that formula `ggml_expf` where this uses the
// hardware `exp`, which is the only intended divergence.
//
//   binding/buffer 0: x       f32, read-write, activated in place (n elements)
//   binding/buffer 1: params  (n, unused)
//
// Dispatch: (ceil(n / 256), 1, 1) workgroups of 256.

[[vk::binding(0)]] RWStructuredBuffer<float> x_buf   : register(u0);
[[vk::binding(1)]] StructuredBuffer<uint2>   par_buf : register(t1);

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_inplace(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= par_buf[0].x) {
        return;
    }
    x_buf[i] = max(x_buf[i], 0.0f);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void silu_inplace(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= par_buf[0].x) {
        return;
    }
    float v = clamp(x_buf[i], -80.0f, 80.0f);
    x_buf[i] = v / (1.0f + exp(-v));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void gelu_erf_inplace(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= par_buf[0].x) {
        return;
    }
    float v = x_buf[i];

    // erf(v / sqrt(2)), Abramowitz & Stegun 7.1.26.
    float a = abs(v * 0.70710678f);
    float sign = (v < 0.0f) ? -1.0f : 1.0f;
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float poly = ((((1.0614054f * t - 1.453152f) * t) + 1.4214137f) * t - 0.2844967f) * t
               + 0.2548296f;
    float erf = sign * (1.0f - poly * t * exp(-a * a));

    x_buf[i] = 0.5f * v * (1.0f + erf);
}
