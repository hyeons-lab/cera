// Batched affine LayerNorm: process N independent vectors in one dispatch.
// Each workgroup handles one vector (one token row).
//   dst[i] = (src[i] - mean) * inv_std * weight[i] + bias[i]
//   mean   = (1/n) Σ src
//   var    = (1/n) Σ (src - mean)^2          (population variance)
//   inv_std = 1 / sqrt(var + eps)
//
// Mirrors `cpu::layer_norm_inplace` (mean-centered LayerNorm with an explicit
// bias term — distinct from RMSNorm). The CPU reference accumulates mean/var in
// f64; the GPU does two f32 reduction passes (sum, then sum of squared
// deviations), which is numerically close enough for the encoder's tolerance.
//
// Dispatch: (N, 1, 1) workgroups of 256 threads.
//
// Bind group 0:
//   @binding(0) src: array<f32>     (read — input rows, stride = params.z)
//   @binding(1) dst: array<f32>     (read-write — normalized rows, stride = params.w)
//   @binding(2) weight: array<f32>  (read — per-element scale, length n)
//   @binding(3) bias: array<f32>    (read — per-element shift, length n)
//   @binding(4) params: vec4<u32>   (n, eps_bits, src_stride, dst_stride)

#define WG_SUM_REDUCE
#include "common_decls.tmpl"

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> params: vec4<u32>;

var<workgroup> shared_sum: array<f32, 256>;
var<workgroup> shared_mean: f32;

@compute @workgroup_size(256, 1, 1)
fn layernorm_batch(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let n = params.x;
    let eps = bitcast<f32>(params.y);
    let src_off = wid.x * params.z;
    let dst_off = wid.x * params.w;

    // Pass 1: mean = (1/n) Σ src.
    var partial: f32 = 0.0;
    var i = tid;
    while i < n {
        partial += src[src_off + i];
        i += 256u;
    }
    shared_sum[tid] = partial;
    workgroupBarrier();
    workgroup_sum_reduce(tid);
    if tid == 0u {
        shared_mean = shared_sum[0] / f32(n);
    }
    workgroupBarrier();
    let mean = shared_mean;

    // Pass 2: var = (1/n) Σ (src - mean)^2.
    // (No barrier needed here — the one after `shared_mean` is written already
    // separates the mean read from the pass-2 `shared_sum` writes below.)
    partial = 0.0;
    i = tid;
    while i < n {
        let d = src[src_off + i] - mean;
        partial += d * d;
        i += 256u;
    }
    shared_sum[tid] = partial;
    workgroupBarrier();
    workgroup_sum_reduce(tid);
    let inv_std = 1.0 / sqrt(shared_sum[0] / f32(n) + eps);

    // Pass 3: affine normalize.
    i = tid;
    while i < n {
        dst[dst_off + i] = (src[src_off + i] - mean) * inv_std * weight[i] + bias[i];
        i += 256u;
    }
}
