@binding(4) @group(0) var<storage, read> par_buf_0 : array<u32>;

@binding(0) @group(0) var<storage, read> proj_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read_write> rbuf_0 : array<f32>;

@binding(2) @group(0) var<storage, read> w_buf_0 : array<f32>;

@binding(3) @group(0) var<storage, read_write> out_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn conv1d_fused(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var ch_0 : u32 = gid_0.x;
    var hs_0 : u32 = par_buf_0[i32(0)];
    var ks_0 : u32 = par_buf_0[i32(1)];
    var d_conv_0 : u32 = par_buf_0[i32(2)];
    if(ch_0 >= hs_0)
    {
        return;
    }
    var c_val_0 : f32 = proj_buf_0[hs_0 + ch_0];
    var bx_0 : f32 = proj_buf_0[ch_0] * proj_buf_0[u32(2) * hs_0 + ch_0];
    var k_0 : u32 = u32(0);
    var sum_0 : f32 = 0.0f;
    for(;;)
    {
        if(k_0 < d_conv_0)
        {
        }
        else
        {
            break;
        }
        var sum_1 : f32 = sum_0 + rbuf_0[k_0 * hs_0 + ch_0] * w_buf_0[ch_0 * ks_0 + k_0];
        k_0 = k_0 + u32(1);
        sum_0 = sum_1;
    }
    var sum_2 : f32 = sum_0 + bx_0 * w_buf_0[ch_0 * ks_0 + d_conv_0];
    if(d_conv_0 > u32(1))
    {
        k_0 = u32(0);
        for(;;)
        {
            if(k_0 < (d_conv_0 - u32(1)))
            {
            }
            else
            {
                break;
            }
            var _S1 : u32 = k_0 + u32(1);
            rbuf_0[k_0 * hs_0 + ch_0] = rbuf_0[_S1 * hs_0 + ch_0];
            k_0 = _S1;
        }
    }
    if(d_conv_0 > u32(0))
    {
        rbuf_0[(d_conv_0 - u32(1)) * hs_0 + ch_0] = bx_0;
    }
    out_buf_0[ch_0] = c_val_0 * sum_2;
    return;
}

