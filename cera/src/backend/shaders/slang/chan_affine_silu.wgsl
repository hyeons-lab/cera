@binding(3) @group(0) var<storage, read> par_buf_0 : array<vec4<u32>>;

@binding(0) @group(0) var<storage, read_write> x_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read> w_buf_0 : array<f32>;

@binding(2) @group(0) var<storage, read> b_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn chan_affine_silu(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var t_0 : u32 = par_buf_0[i32(0)].y;
    var idx_0 : u32 = gid_0.x;
    if(idx_0 >= (par_buf_0[i32(0)].x * t_0))
    {
        return;
    }
    var c_0 : u32 = idx_0 / t_0;
    var v_0 : f32 = x_buf_0[idx_0] * w_buf_0[c_0] + b_buf_0[c_0];
    x_buf_0[idx_0] = v_0 / (1.0f + exp(- v_0));
    return;
}

