@binding(4) @group(0) var<storage, read> params_0 : array<vec4<u32>>;

@binding(3) @group(0) var<storage, read> sel_expert_0 : array<u32>;

@binding(0) @group(0) var<storage, read> w_0 : array<u32>;

@binding(1) @group(0) var<storage, read> x_0 : array<f32>;

@binding(2) @group(0) var<storage, read_write> y_0 : array<f32>;

fn load_byte_0( byte_off_0 : u32) -> u32
{
    return (((w_0[(byte_off_0 >> (u32(2)))] >> ((((byte_off_0 & (u32(3)))) * u32(8))))) & (u32(255)));
}

fn load_f16_0( byte_off_1 : u32) -> f32
{
    var word_0 : u32 = w_0[(byte_off_1 >> (u32(2)))];
    var h_0 : u32;
    if(((byte_off_1 & (u32(2)))) != u32(0))
    {
        h_0 = (word_0 >> (u32(16)));
    }
    else
    {
        h_0 = (word_0 & (u32(65535)));
    }
    return (unpack2x16float((h_0)).x);
}

var<workgroup> scratch_0 : array<f32, i32(32)>;

fn block_sum_0( tid_0 : u32,  v_0 : f32) -> f32
{
    scratch_0[tid_0] = v_0;
    workgroupBarrier();
    var s_0 : u32 = u32(16);
    for(;;)
    {
        if(s_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_0 < s_0)
        {
            scratch_0[tid_0] = scratch_0[tid_0] + scratch_0[tid_0 + s_0];
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    var _S1 : f32 = scratch_0[i32(0)];
    return _S1;
}

@compute
@workgroup_size(32, 1, 1)
fn moe_gemv_q4_0(@builtin(local_invocation_id) lid_0 : vec3<u32>, @builtin(workgroup_id) grp_0 : vec3<u32>)
{
    var m_0 : u32 = params_0[i32(0)].x;
    var k_0 : u32 = params_0[i32(0)].y;
    var _S2 : u32 = max(params_0[i32(0)].z, u32(1));
    var n_entries_0 : u32 = params_0[i32(0)].w;
    var expert_stride_0 : u32 = params_0[i32(1)].x;
    var x_by_entry_0 : bool = (params_0[i32(1)].y) != u32(0);
    var row_0 : u32 = grp_0.x;
    var entry_0 : u32 = grp_0.y;
    var _S3 : bool;
    if(row_0 >= m_0)
    {
        _S3 = true;
    }
    else
    {
        _S3 = entry_0 >= n_entries_0;
    }
    if(_S3)
    {
        return;
    }
    var x_row_0 : u32;
    if(x_by_entry_0)
    {
        x_row_0 = entry_0;
    }
    else
    {
        var _S4 : u32 = entry_0 / _S2;
        x_row_0 = _S4;
    }
    var _S5 : u32 = x_row_0 * k_0;
    var nb_0 : u32 = k_0 / u32(32);
    var _S6 : u32 = sel_expert_0[entry_0] * expert_stride_0 + row_0 * (nb_0 * u32(18));
    var _S7 : u32 = lid_0.x;
    var bi_0 : u32 = _S7;
    var sum_0 : f32 = 0.0f;
    for(;;)
    {
        if(bi_0 < nb_0)
        {
        }
        else
        {
            break;
        }
        var blk_0 : u32 = _S6 + bi_0 * u32(18);
        var _S8 : u32 = _S5 + bi_0 * u32(32);
        var i_0 : u32 = u32(0);
        var acc_0 : f32 = 0.0f;
        for(;;)
        {
            if(i_0 < u32(16))
            {
            }
            else
            {
                break;
            }
            var byte_0 : u32 = load_byte_0(blk_0 + u32(2) + i_0);
            var _S9 : u32 = _S8 + i_0;
            var acc_1 : f32 = acc_0 + (f32((byte_0 & (u32(15)))) - 8.0f) * x_0[_S9] + (f32((byte_0 >> (u32(4)))) - 8.0f) * x_0[_S9 + u32(16)];
            i_0 = i_0 + u32(1);
            acc_0 = acc_1;
        }
        var sum_1 : f32 = sum_0 + acc_0 * load_f16_0(blk_0);
        bi_0 = bi_0 + u32(32);
        sum_0 = sum_1;
    }
    var total_0 : f32 = block_sum_0(_S7, sum_0);
    if(_S7 == u32(0))
    {
        y_0[entry_0 * m_0 + row_0] = total_0;
    }
    return;
}

