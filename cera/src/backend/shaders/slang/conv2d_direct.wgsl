@binding(4) @group(0) var<storage, read> par_buf_0 : array<vec4<u32>>;

@binding(2) @group(0) var<storage, read> bias_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read> w_buf_0 : array<f32>;

@binding(0) @group(0) var<storage, read> in_buf_0 : array<f32>;

@binding(3) @group(0) var<storage, read_write> out_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn conv2d_direct(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var in_ch_0 : u32 = par_buf_0[i32(0)].x;
    var out_ch_0 : u32 = par_buf_0[i32(0)].y;
    var _S1 : u32 = par_buf_0[i32(0)].z;
    var _S2 : u32 = par_buf_0[i32(0)].w;
    var _S3 : u32 = par_buf_0[i32(1)].x;
    var _S4 : u32 = par_buf_0[i32(1)].y;
    var str_h_0 : u32 = par_buf_0[i32(1)].z;
    var str_w_0 : u32 = par_buf_0[i32(1)].w;
    var pad_h_0 : u32 = par_buf_0[i32(2)].x;
    var pad_w_0 : u32 = par_buf_0[i32(2)].y;
    var w_out_0 : u32 = par_buf_0[i32(2)].w;
    var groups_0 : u32 = par_buf_0[i32(3)].x;
    var plane_out_0 : u32 = par_buf_0[i32(2)].z * w_out_0;
    var idx_0 : u32 = gid_0.x;
    if(idx_0 >= (out_ch_0 * plane_out_0))
    {
        return;
    }
    var oc_0 : u32 = idx_0 / plane_out_0;
    var rem_0 : u32 = idx_0 - oc_0 * plane_out_0;
    var oh_0 : u32 = rem_0 / w_out_0;
    var ow_0 : u32 = rem_0 - oh_0 * w_out_0;
    var _S5 : u32 = in_ch_0 / groups_0;
    var out_per_group_0 : u32 = out_ch_0 / groups_0;
    var _S6 : u32 = oc_0 / out_per_group_0;
    var _S7 : i32 = i32(oh_0 * str_h_0) - i32(pad_h_0);
    var _S8 : i32 = i32(ow_0 * str_w_0) - i32(pad_w_0);
    var _S9 : f32 = bias_buf_0[oc_0];
    var ic_local_0 : u32 = u32(0);
    var acc_0 : f32 = _S9;
    for(;;)
    {
        if(ic_local_0 < _S5)
        {
        }
        else
        {
            break;
        }
        var _S10 : u32 = (oc_0 * _S5 + ic_local_0) * _S3 * _S4;
        var _S11 : u32 = (_S6 * _S5 + ic_local_0) * _S1 * _S2;
        var ki_0 : u32 = u32(0);
        var acc_1 : f32 = acc_0;
        for(;;)
        {
            if(ki_0 < _S3)
            {
            }
            else
            {
                break;
            }
            var ih_0 : i32 = _S7 + i32(ki_0);
            var _S12 : bool;
            if(ih_0 < i32(0))
            {
                _S12 = true;
            }
            else
            {
                _S12 = ih_0 >= i32(_S1);
            }
            var acc_2 : f32;
            if(_S12)
            {
                acc_2 = acc_1;
                ki_0 = ki_0 + u32(1);
                acc_1 = acc_2;
                continue;
            }
            var _S13 : u32 = _S11 + u32(ih_0) * _S2;
            var kj_0 : u32 = u32(0);
            acc_2 = acc_1;
            for(;;)
            {
                if(kj_0 < _S4)
                {
                }
                else
                {
                    break;
                }
                var iw_0 : i32 = _S8 + i32(kj_0);
                var _S14 : bool;
                if(iw_0 < i32(0))
                {
                    _S14 = true;
                }
                else
                {
                    _S14 = iw_0 >= i32(_S2);
                }
                if(_S14)
                {
                    kj_0 = kj_0 + u32(1);
                    continue;
                }
                acc_2 = acc_2 + w_buf_0[_S10 + ki_0 * _S4 + kj_0] * in_buf_0[_S13 + u32(iw_0)];
                kj_0 = kj_0 + u32(1);
            }
            ki_0 = ki_0 + u32(1);
            acc_1 = acc_2;
        }
        ic_local_0 = ic_local_0 + u32(1);
        acc_0 = acc_1;
    }
    out_buf_0[idx_0] = acc_0;
    return;
}

