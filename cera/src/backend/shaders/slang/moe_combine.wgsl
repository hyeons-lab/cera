@binding(3) @group(0) var<storage, read> params_0 : array<vec4<u32>>;

@binding(1) @group(0) var<storage, read> sel_weight_0 : array<f32>;

@binding(0) @group(0) var<storage, read> z_0 : array<f32>;

@binding(2) @group(0) var<storage, read_write> out_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn moe_combine(@builtin(local_invocation_id) lid_0 : vec3<u32>, @builtin(workgroup_id) grp_0 : vec3<u32>)
{
    var hidden_0 : u32 = params_0[i32(0)].x;
    var _S1 : u32 = params_0[i32(0)].y;
    var n_tokens_0 : u32 = params_0[i32(0)].z;
    var accumulate_0 : bool = (params_0[i32(0)].w) != u32(0);
    var row_0 : u32 = grp_0.x * u32(256) + lid_0.x;
    var tok_0 : u32 = grp_0.y;
    var _S2 : bool;
    if(row_0 >= hidden_0)
    {
        _S2 = true;
    }
    else
    {
        _S2 = tok_0 >= n_tokens_0;
    }
    if(_S2)
    {
        return;
    }
    var s_0 : u32 = u32(0);
    var acc_0 : f32 = 0.0f;
    for(;;)
    {
        if(s_0 < _S1)
        {
        }
        else
        {
            break;
        }
        var entry_0 : u32 = tok_0 * _S1 + s_0;
        var acc_1 : f32 = acc_0 + sel_weight_0[entry_0] * z_0[entry_0 * hidden_0 + row_0];
        s_0 = s_0 + u32(1);
        acc_0 = acc_1;
    }
    var dst_0 : u32 = tok_0 * hidden_0 + row_0;
    if(accumulate_0)
    {
        acc_0 = out_buf_0[dst_0] + acc_0;
    }
    else
    {
    }
    out_buf_0[dst_0] = acc_0;
    return;
}

