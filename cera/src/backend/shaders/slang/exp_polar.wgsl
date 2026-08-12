@binding(2) @group(0) var<storage, read> par_buf_0 : array<vec2<u32>>;

@binding(0) @group(0) var<storage, read> spec_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read_write> out_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn exp_polar(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var bins_0 : u32 = par_buf_0[i32(0)].y;
    var i_0 : u32 = gid_0.x;
    if(i_0 >= (par_buf_0[i32(0)].x * bins_0))
    {
        return;
    }
    var frame_0 : u32 = i_0 / bins_0;
    var j_0 : u32 = i_0 % bins_0;
    var base_0 : u32 = frame_0 * u32(2) * bins_0;
    var _S1 : u32 = base_0 + j_0;
    var _S2 : u32 = base_0 + bins_0 + j_0;
    var angle_0 : f32 = spec_buf_0[_S2];
    var mag_0 : f32 = exp(spec_buf_0[_S1]);
    out_buf_0[_S1] = mag_0 * cos(angle_0);
    out_buf_0[_S2] = mag_0 * sin(angle_0);
    return;
}

