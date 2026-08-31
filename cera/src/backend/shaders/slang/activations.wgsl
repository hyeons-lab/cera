@binding(1) @group(0) var<storage, read> par_buf_0 : array<vec2<u32>>;

@binding(0) @group(0) var<storage, read_write> x_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn relu_inplace(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var i_0 : u32 = gid_0.x;
    if(i_0 >= (par_buf_0[i32(0)].x))
    {
        return;
    }
    x_buf_0[i_0] = max(x_buf_0[i_0], 0.0f);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn silu_inplace(@builtin(global_invocation_id) gid_1 : vec3<u32>)
{
    var i_1 : u32 = gid_1.x;
    if(i_1 >= (par_buf_0[i32(0)].x))
    {
        return;
    }
    var v_0 : f32 = clamp(x_buf_0[i_1], -80.0f, 80.0f);
    x_buf_0[i_1] = v_0 / (1.0f + exp(- v_0));
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn gelu_erf_inplace(@builtin(global_invocation_id) gid_2 : vec3<u32>)
{
    var i_2 : u32 = gid_2.x;
    if(i_2 >= (par_buf_0[i32(0)].x))
    {
        return;
    }
    var v_1 : f32 = x_buf_0[i_2];
    var a_0 : f32 = abs(x_buf_0[i_2] * 0.70710676908493042f);
    var sign_0 : f32;
    if(x_buf_0[i_2] < 0.0f)
    {
        sign_0 = -1.0f;
    }
    else
    {
        sign_0 = 1.0f;
    }
    var t_0 : f32 = 1.0f / (1.0f + 0.32759109139442444f * a_0);
    x_buf_0[i_2] = 0.5f * v_1 * (1.0f + sign_0 * (1.0f - ((((1.06140542030334473f * t_0 - 1.45315194129943848f) * t_0 + 1.42141366004943848f) * t_0 - 0.28449669480323792f) * t_0 + 0.25482958555221558f) * t_0 * exp(- a_0 * a_0)));
    return;
}

