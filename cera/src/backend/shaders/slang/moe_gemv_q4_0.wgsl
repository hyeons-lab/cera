@binding(4) @group(0) var<storage, read> params_0 : array<vec4<u32>>;

@binding(3) @group(0) var<storage, read> sel_expert_0 : array<u32>;

@binding(0) @group(0) var<storage, read> w_0 : array<u32>;

@binding(1) @group(0) var<storage, read> x_0 : array<f32>;

@binding(2) @group(0) var<storage, read_write> y_0 : array<f32>;

fn load_u16_0( byte_off_0 : u32) -> u32
{
    var word_0 : u32 = w_0[(byte_off_0 >> (u32(2)))];
    var _S1 : u32;
    if(((byte_off_0 & (u32(2)))) != u32(0))
    {
        _S1 = (word_0 >> (u32(16)));
    }
    else
    {
        _S1 = (word_0 & (u32(65535)));
    }
    return _S1;
}

fn load_f16_0( byte_off_1 : u32) -> f32
{
    return (unpack2x16float((load_u16_0(byte_off_1))).x);
}

var<workgroup> scratch_0 : array<f32, i32(32)>;

fn block_sum_0( tid_0 : u32,  v_0 : f32) -> f32
{
    scratch_0[tid_0] = v_0;
    workgroupBarrier();
    var s_0 : u32 = u32(16);
    loop
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
    var _S2 : f32 = scratch_0[i32(0)];
    return _S2;
}

@compute
@workgroup_size(32, 1, 1)
fn moe_gemv_q4_0(@builtin(local_invocation_id) lid_0 : vec3<u32>, @builtin(workgroup_id) grp_0 : vec3<u32>)
{
    var m_0 : u32 = params_0[i32(0)].x;
    var k_0 : u32 = params_0[i32(0)].y;
    var _S3 : u32 = max(params_0[i32(0)].z, u32(1));
    var n_entries_0 : u32 = params_0[i32(0)].w;
    var expert_stride_0 : u32 = params_0[i32(1)].x;
    var x_by_entry_0 : bool = (params_0[i32(1)].y) != u32(0);
    var row_0 : u32 = grp_0.x;
    var entry_0 : u32 = grp_0.y;
    var _S4 : bool;
    if(row_0 >= m_0)
    {
        _S4 = true;
    }
    else
    {
        _S4 = entry_0 >= n_entries_0;
    }
    if(_S4)
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
        var _S5 : u32 = entry_0 / _S3;
        x_row_0 = _S5;
    }
    var _S6 : u32 = x_row_0 * k_0;
    var nb_0 : u32 = k_0 / u32(32);
    var _S7 : u32 = sel_expert_0[entry_0] * expert_stride_0 + row_0 * (nb_0 * u32(18));
    var _S8 : u32 = lid_0.x;
    var bi_0 : u32 = _S8;
    var sum_0 : f32 = 0.0f;
    loop
    {
        if(bi_0 < nb_0)
        {
        }
        else
        {
            break;
        }
        var blk_0 : u32 = _S7 + bi_0 * u32(18);
        var _S9 : u32 = _S6 + bi_0 * u32(32);
        var i_0 : u32 = u32(0);
        var xsum_0 : f32 = 0.0f;
        var acc_0 : f32 = 0.0f;
        loop
        {
            if(i_0 < u32(8))
            {
            }
            else
            {
                break;
            }
            var _S10 : u32 = i_0 * u32(2);
            var q16_0 : u32 = load_u16_0(blk_0 + u32(2) + _S10);
            var idx_0 : u32 = _S9 + _S10;
            var x0_0 : f32 = x_0[idx_0];
            var x1_0 : f32 = x_0[idx_0 + u32(1)];
            var x16_0 : f32 = x_0[idx_0 + u32(16)];
            var x17_0 : f32 = x_0[idx_0 + u32(17)];
            var xsum_1 : f32 = xsum_0 + (x0_0 + x1_0 + (x16_0 + x17_0));
            var acc_1 : f32 = acc_0 + (f32((q16_0 & (u32(15)))) * x0_0 + f32((((q16_0 >> (u32(8)))) & (u32(15)))) * x1_0 + f32((((q16_0 >> (u32(4)))) & (u32(15)))) * x16_0 + f32((q16_0 >> (u32(12)))) * x17_0);
            i_0 = i_0 + u32(1);
            xsum_0 = xsum_1;
            acc_0 = acc_1;
        }
        var sum_1 : f32 = sum_0 + (acc_0 - 8.0f * xsum_0) * load_f16_0(blk_0);
        bi_0 = bi_0 + u32(32);
        sum_0 = sum_1;
    }
    var total_0 : f32 = block_sum_0(_S8, sum_0);
    if(_S8 == u32(0))
    {
        y_0[entry_0 * m_0 + row_0] = total_0;
    }
    return;
}

