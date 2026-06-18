// Broadcast bias add, in-place: x[t*dim + j] += bias[j].
//
// Linear-layer bias is a single [dim] vector added to every one of the `rows`
// token rows. Plain `elementwise.add_inplace` can't express the broadcast.
//
// Dispatch: (ceil(rows*dim / 256), 1, 1) workgroups.
//
// Bind group 0:
//   @binding(0) x: array<f32>      (read-write — rows*dim elements)
//   @binding(1) bias: array<f32>   (read — dim elements)
//   @binding(2) params: vec2<u32>  (total = rows*dim, dim)

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read> params: vec2<u32>;

@compute @workgroup_size(256, 1, 1)
fn bias_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.x;
    let dim = params.y;
    if i >= total { return; }
    x[i] = x[i] + bias[i % dim];
}
