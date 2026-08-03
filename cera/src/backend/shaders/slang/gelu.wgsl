@binding(1) @group(0) var<storage, read> par_buf_0 : array<vec2<u32>>;

@binding(0) @group(0) var<storage, read_write> x_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn gelu_inplace(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var i_0 : u32 = gid_0.x;
    if(i_0 >= (par_buf_0[i32(0)].x))
    {
        return;
    }
    x_buf_0[i_0] = 0.5f * x_buf_0[i_0] * (1.0f + tanh(clamp(0.79788458347320557f * (x_buf_0[i_0] + 0.04471499845385551f * x_buf_0[i_0] * x_buf_0[i_0] * x_buf_0[i_0]), -15.0f, 15.0f)));
    return;
}

