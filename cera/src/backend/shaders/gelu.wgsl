// tanh-approximation GELU, in-place over an f32 buffer.
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//
// Mirrors `cpu::gelu_inplace` (ggml's default GELU), which is what CLIP-family
// ViTs trained with `clip.use_gelu = true` expect. NOT the erf form.
//
// Dispatch: (ceil(n / 256), 1, 1) workgroups.
//
// Bind group 0:
//   @binding(0) x: array<f32>      (read-write, activated in-place)
//   @binding(1) params: vec2<u32>  (n, unused)

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> params: vec2<u32>;

@compute @workgroup_size(256, 1, 1)
fn gelu_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.x;
    if i >= n { return; }
    let xv = x[i];
    let inner = 0.7978845608 * (xv + 0.044715 * xv * xv * xv);
    x[i] = 0.5 * xv * (1.0 + tanh(inner));
}
