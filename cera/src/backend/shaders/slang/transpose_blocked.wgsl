@binding(2) @group(0) var<storage, read> par_buf_0 : array<vec4<u32>>;

@binding(1) @group(0) var<storage, read_write> dst_buf_0 : array<f32>;

@binding(0) @group(0) var<storage, read> src_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn transpose_blocked(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var a_dim_0 : u32 = par_buf_0[i32(0)].x;
    var k_dim_0 : u32 = par_buf_0[i32(0)].z;
    var bk_0 : u32 = par_buf_0[i32(0)].y * k_dim_0;
    var idx_0 : u32 = gid_0.x;
    if(idx_0 >= (a_dim_0 * bk_0))
    {
        return;
    }
    var a_0 : u32 = idx_0 / bk_0;
    var rem_0 : u32 = idx_0 - a_0 * bk_0;
    var b_0 : u32 = rem_0 / k_dim_0;
    dst_buf_0[(b_0 * a_dim_0 + a_0) * k_dim_0 + (rem_0 - b_0 * k_dim_0)] = src_buf_0[idx_0];
    return;
}

