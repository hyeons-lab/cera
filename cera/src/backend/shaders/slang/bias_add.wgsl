@binding(2) @group(0) var<storage, read> par_buf_0 : array<vec2<u32>>;

@binding(0) @group(0) var<storage, read_write> x_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read> bias_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn bias_add(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var i_0 : u32 = gid_0.x;
    var dim_0 : u32 = par_buf_0[i32(0)].y;
    if(i_0 >= (par_buf_0[i32(0)].x))
    {
        return;
    }
    var _S1 : f32 = x_buf_0[i_0];
    var _S2 : u32 = i_0 % dim_0;
    x_buf_0[i_0] = _S1 + bias_buf_0[_S2];
    return;
}

