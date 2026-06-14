// Scale an f32 buffer in place by a scalar constant: a[i] *= s.
//
// Used by Granite 3.x scalar multipliers (embedding / residual / logit). The
// scalar is passed bit-cast into the params buffer so no per-call buffer upload
// of the value is needed beyond the small params write.
//
// Bind group 0:
//   @binding(0) a: array<f32>       (read-write, scaled in place)
//   @binding(1) params: vec2<u32>   (n, scale_bits)
//
// Dispatch: (ceil(n / 256), 1, 1) workgroups

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> params: vec2<u32>;

@compute @workgroup_size(256, 1, 1)
fn scale_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.x;
    if i >= n { return; }
    a[i] = a[i] * bitcast<f32>(params.y);
}
