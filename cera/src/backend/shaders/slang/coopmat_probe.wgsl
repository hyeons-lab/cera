enable f16;

@binding(0) @group(0) var<storage, read> a_buf_0 : array<f16>;

@binding(1) @group(0) var<storage, read> b_buf_0 : array<f16>;

@binding(2) @group(0) var<storage, read_write> c_buf_0 : array<f32>;

@compute
@workgroup_size(32, 1, 1)
fn coopmat_probe(@builtin(local_invocation_id) tid_0 : vec3<u32>)
{
    var t_0 : u32 = tid_0.x;
    if(t_0 < u32(64))
    {
        var _S1 : u32 = t_0 / u32(8);
        var _S2 : u32 = t_0 % u32(8);
        var i_0 : u32 = u32(0);
        var acc_0 : f32 = 0.0f;
        for(;;)
        {
            if(i_0 < u32(8))
            {
            }
            else
            {
                break;
            }
            var acc_1 : f32 = acc_0 + f32(a_buf_0[_S1 * u32(8) + i_0]) * f32(b_buf_0[i_0 * u32(8) + _S2]);
            i_0 = i_0 + u32(1);
            acc_0 = acc_1;
        }
        c_buf_0[t_0] = acc_0;
    }
    return;
}

