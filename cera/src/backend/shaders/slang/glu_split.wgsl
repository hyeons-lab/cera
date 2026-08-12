@binding(2) @group(0) var<storage, read> par_buf_0 : array<vec4<u32>>;

@binding(0) @group(0) var<storage, read> src_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read_write> dst_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn glu_split(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var n_0 : u32 = par_buf_0[i32(0)].y;
    var idx_0 : u32 = gid_0.x;
    if(idx_0 >= (par_buf_0[i32(0)].x * n_0))
    {
        return;
    }
    var r_0 : u32 = idx_0 / n_0;
    var c_0 : u32 = idx_0 - r_0 * n_0;
    var base_0 : u32 = r_0 * u32(2) * n_0;
    dst_buf_0[idx_0] = src_buf_0[base_0 + c_0] / (1.0f + exp(- src_buf_0[base_0 + n_0 + c_0]));
    return;
}

