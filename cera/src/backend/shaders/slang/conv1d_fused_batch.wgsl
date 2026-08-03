@binding(4) @group(0) var<storage, read> par_buf_0 : array<u32>;

@binding(2) @group(0) var<storage, read> w_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read_write> rbuf_0 : array<f32>;

@binding(0) @group(0) var<storage, read> proj_buf_0 : array<f32>;

@binding(3) @group(0) var<storage, read_write> out_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn conv1d_fused_batch(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var ch_0 : u32 = gid_0.x;
    var hs_0 : u32 = par_buf_0[i32(0)];
    var ks_0 : u32 = par_buf_0[i32(1)];
    var d_conv_0 : u32 = par_buf_0[i32(2)];
    var _S1 : u32 = par_buf_0[i32(3)];
    var _S2 : u32 = par_buf_0[i32(4)];
    var _S3 : u32 = par_buf_0[i32(5)];
    if(ch_0 >= hs_0)
    {
        return;
    }
    var _S4 : bool;
    if(ks_0 > u32(4))
    {
        _S4 = true;
    }
    else
    {
        _S4 = d_conv_0 > u32(3);
    }
    if(_S4)
    {
        return;
    }
    var w_local_0 : array<f32, i32(4)>;
    w_local_0[i32(0)] = 0.0f;
    w_local_0[i32(1)] = 0.0f;
    w_local_0[i32(2)] = 0.0f;
    w_local_0[i32(3)] = 0.0f;
    if(u32(0) <= d_conv_0)
    {
        _S4 = u32(0) < ks_0;
    }
    else
    {
        _S4 = false;
    }
    if(_S4)
    {
        w_local_0[u32(0)] = w_buf_0[ch_0 * ks_0];
    }
    if(u32(1) <= d_conv_0)
    {
        _S4 = u32(1) < ks_0;
    }
    else
    {
        _S4 = false;
    }
    if(_S4)
    {
        w_local_0[u32(1)] = w_buf_0[ch_0 * ks_0 + u32(1)];
    }
    if(u32(2) <= d_conv_0)
    {
        _S4 = u32(2) < ks_0;
    }
    else
    {
        _S4 = false;
    }
    if(_S4)
    {
        w_local_0[u32(2)] = w_buf_0[ch_0 * ks_0 + u32(2)];
    }
    if(u32(3) <= d_conv_0)
    {
        _S4 = u32(3) < ks_0;
    }
    else
    {
        _S4 = false;
    }
    if(_S4)
    {
        w_local_0[u32(3)] = w_buf_0[ch_0 * ks_0 + u32(3)];
    }
    var rb_0 : array<f32, i32(3)>;
    rb_0[i32(0)] = 0.0f;
    rb_0[i32(1)] = 0.0f;
    rb_0[i32(2)] = 0.0f;
    var _S5 : bool = u32(0) < d_conv_0;
    if(_S5)
    {
        rb_0[u32(0)] = rbuf_0[ch_0];
    }
    var _S6 : bool = u32(1) < d_conv_0;
    if(_S6)
    {
        rb_0[u32(1)] = rbuf_0[hs_0 + ch_0];
    }
    var _S7 : bool = u32(2) < d_conv_0;
    if(_S7)
    {
        rb_0[u32(2)] = rbuf_0[u32(2) * hs_0 + ch_0];
    }
    var t_0 : u32 = u32(0);
    for(;;)
    {
        if(t_0 < _S1)
        {
        }
        else
        {
            break;
        }
        var base_0 : u32 = t_0 * _S2;
        var c_val_0 : f32 = proj_buf_0[base_0 + hs_0 + ch_0];
        var bx_0 : f32 = proj_buf_0[base_0 + ch_0] * proj_buf_0[base_0 + u32(2) * hs_0 + ch_0];
        var sum_0 : f32;
        if(_S5)
        {
            sum_0 = rb_0[u32(0)] * w_local_0[u32(0)];
        }
        else
        {
            sum_0 = 0.0f;
        }
        var sum_1 : f32;
        if(_S6)
        {
            sum_1 = sum_0 + rb_0[u32(1)] * w_local_0[u32(1)];
        }
        else
        {
            sum_1 = sum_0;
        }
        var sum_2 : f32;
        if(_S7)
        {
            sum_2 = sum_1 + rb_0[u32(2)] * w_local_0[u32(2)];
        }
        else
        {
            sum_2 = sum_1;
        }
        var sum_3 : f32 = sum_2 + bx_0 * w_local_0[d_conv_0];
        if(_S6)
        {
            rb_0[u32(0)] = rb_0[u32(1)];
        }
        if(_S7)
        {
            rb_0[u32(1)] = rb_0[u32(2)];
        }
        if(d_conv_0 > u32(0))
        {
            rb_0[d_conv_0 - u32(1)] = bx_0;
        }
        out_buf_0[t_0 * _S3 + ch_0] = c_val_0 * sum_3;
        t_0 = t_0 + u32(1);
    }
    if(_S5)
    {
        rbuf_0[ch_0] = rb_0[u32(0)];
    }
    if(_S6)
    {
        rbuf_0[hs_0 + ch_0] = rb_0[u32(1)];
    }
    if(_S7)
    {
        rbuf_0[u32(2) * hs_0 + ch_0] = rb_0[u32(2)];
    }
    return;
}

