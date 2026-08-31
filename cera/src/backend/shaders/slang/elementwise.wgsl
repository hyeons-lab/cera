@binding(2) @group(0) var<storage, read> par_buf_0 : array<vec2<u32>>;

@binding(0) @group(0) var<storage, read_write> a_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read> b_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn add_inplace(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var i_0 : u32 = gid_0.x;
    if(i_0 >= (par_buf_0[i32(0)].x))
    {
        return;
    }
    a_buf_0[i_0] = a_buf_0[i_0] + b_buf_0[i_0];
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn scaled_add_inplace(@builtin(global_invocation_id) gid_1 : vec3<u32>)
{
    var i_1 : u32 = gid_1.x;
    if(i_1 >= (par_buf_0[i32(0)].x))
    {
        return;
    }
    a_buf_0[i_1] = a_buf_0[i_1] + (bitcast<f32>((par_buf_0[i32(0)].y))) * b_buf_0[i_1];
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn mul_inplace(@builtin(global_invocation_id) gid_2 : vec3<u32>)
{
    var i_2 : u32 = gid_2.x;
    if(i_2 >= (par_buf_0[i32(0)].x))
    {
        return;
    }
    a_buf_0[i_2] = a_buf_0[i_2] * b_buf_0[i_2];
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn silu_mul_inplace(@builtin(global_invocation_id) gid_3 : vec3<u32>)
{
    var i_3 : u32 = gid_3.x;
    if(i_3 >= (par_buf_0[i32(0)].x))
    {
        return;
    }
    var g_0 : f32 = clamp(a_buf_0[i_3], -80.0f, 80.0f);
    a_buf_0[i_3] = g_0 / (1.0f + exp(- g_0)) * b_buf_0[i_3];
    return;
}

