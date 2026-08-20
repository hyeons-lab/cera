@group(0) @binding(0) var<storage, read> src_k: array<f32>;
@group(0) @binding(1) var<storage, read> src_v: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_k: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst_v: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32, 2>;

@compute @workgroup_size(256, 1, 1)
fn df_kv_cache_copy(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let count = params[1];
    if (idx < count) {
        let dst_idx = params[0] + idx;
        dst_k[dst_idx] = src_k[idx];
        dst_v[dst_idx] = src_v[idx];
    }
}
