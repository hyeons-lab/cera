@binding(4) @group(0) var<storage, read> params_0 : array<vec2<u32>>;

@binding(0) @group(0) var<storage, read> w_gate_0 : array<u32>;

@binding(2) @group(0) var<storage, read> x_0 : array<f32>;

@binding(1) @group(0) var<storage, read> w_up_0 : array<u32>;

@binding(3) @group(0) var<storage, read_write> y_0 : array<f32>;

fn get_wid_0( wid_0 : vec3<u32>) -> u32
{
    return wid_0.x + wid_0.y * u32(65535);
}

fn load_gate_byte_0( byte_off_0 : u32) -> u32
{
    return (((w_gate_0[(byte_off_0 >> (u32(2)))] >> ((((byte_off_0 & (u32(3)))) * u32(8))))) & (u32(255)));
}

fn load_gate_f16_0( byte_off_1 : u32) -> f32
{
    var word_0 : u32 = w_gate_0[(byte_off_1 >> (u32(2)))];
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

fn load_up_byte_0( byte_off_2 : u32) -> u32
{
    return (((w_up_0[(byte_off_2 >> (u32(2)))] >> ((((byte_off_2 & (u32(3)))) * u32(8))))) & (u32(255)));
}

fn load_up_f16_0( byte_off_3 : u32) -> f32
{
    var word_1 : u32 = w_up_0[(byte_off_3 >> (u32(2)))];
    var h_1 : u32;
    if(((byte_off_3 & (u32(2)))) != u32(0))
    {
        h_1 = (word_1 >> (u32(16)));
    }
    else
    {
        h_1 = (word_1 & (u32(65535)));
    }
    return (unpack2x16float((h_1)).x);
}

var<workgroup> scratch_0 : array<f32, i32(32)>;

@compute
@workgroup_size(32, 1, 1)
fn ffn_swiglu_q4_0(@builtin(local_invocation_id) lid_0 : vec3<u32>, @builtin(workgroup_id) grp_0 : vec3<u32>)
{
    var k_0 : u32 = params_0[i32(0)].y;
    var row_0 : u32 = get_wid_0(grp_0);
    if(row_0 >= (params_0[i32(0)].x))
    {
        return;
    }
    var nb_0 : u32 = k_0 / u32(32);
    var _S1 : u32 = row_0 * (nb_0 * u32(18));
    var _S2 : u32 = lid_0.x;
    var sub_tid_0 : u32 = (_S2 & (u32(15)));
    var bi_0 : u32;
    var i_0 : u32;
    var sum_0 : f32;
    var sumy_0 : f32;
    var acc_0 : f32;
    if(!(_S2 >= u32(16)))
    {
        bi_0 = sub_tid_0;
        sum_0 = 0.0f;
        for(;;)
        {
            if(bi_0 < nb_0)
            {
            }
            else
            {
                break;
            }
            var blk_0 : u32 = _S1 + bi_0 * u32(18);
            var _S3 : u32 = bi_0 * u32(32);
            i_0 = u32(0);
            sumy_0 = 0.0f;
            acc_0 = 0.0f;
            for(;;)
            {
                if(i_0 < u32(16))
                {
                }
                else
                {
                    break;
                }
                var byte_0 : u32 = load_gate_byte_0(blk_0 + u32(2) + i_0);
                var _S4 : u32 = _S3 + i_0;
                var x0_0 : f32 = x_0[_S4];
                var x1_0 : f32 = x_0[_S4 + u32(16)];
                var sumy_1 : f32 = sumy_0 + (x0_0 + x1_0);
                var acc_1 : f32 = acc_0 + (f32((byte_0 & (u32(15)))) * x0_0 + f32((byte_0 >> (u32(4)))) * x1_0);
                i_0 = i_0 + u32(1);
                sumy_0 = sumy_1;
                acc_0 = acc_1;
            }
            var sum_1 : f32 = sum_0 + (acc_0 - 8.0f * sumy_0) * load_gate_f16_0(blk_0);
            bi_0 = bi_0 + u32(16);
            sum_0 = sum_1;
        }
    }
    else
    {
        bi_0 = sub_tid_0;
        sum_0 = 0.0f;
        for(;;)
        {
            if(bi_0 < nb_0)
            {
            }
            else
            {
                break;
            }
            var blk_1 : u32 = _S1 + bi_0 * u32(18);
            var _S5 : u32 = bi_0 * u32(32);
            i_0 = u32(0);
            sumy_0 = 0.0f;
            acc_0 = 0.0f;
            for(;;)
            {
                if(i_0 < u32(16))
                {
                }
                else
                {
                    break;
                }
                var byte_1 : u32 = load_up_byte_0(blk_1 + u32(2) + i_0);
                var _S6 : u32 = _S5 + i_0;
                var x0_1 : f32 = x_0[_S6];
                var x1_1 : f32 = x_0[_S6 + u32(16)];
                var sumy_2 : f32 = sumy_0 + (x0_1 + x1_1);
                var acc_2 : f32 = acc_0 + (f32((byte_1 & (u32(15)))) * x0_1 + f32((byte_1 >> (u32(4)))) * x1_1);
                i_0 = i_0 + u32(1);
                sumy_0 = sumy_2;
                acc_0 = acc_2;
            }
            var sum_2 : f32 = sum_0 + (acc_0 - 8.0f * sumy_0) * load_up_f16_0(blk_1);
            bi_0 = bi_0 + u32(16);
            sum_0 = sum_2;
        }
    }
    scratch_0[_S2] = sum_0;
    workgroupBarrier();
    if(sub_tid_0 < u32(8))
    {
        scratch_0[_S2] = scratch_0[_S2] + scratch_0[_S2 + u32(8)];
    }
    workgroupBarrier();
    if(sub_tid_0 < u32(4))
    {
        scratch_0[_S2] = scratch_0[_S2] + scratch_0[_S2 + u32(4)];
    }
    workgroupBarrier();
    if(sub_tid_0 < u32(2))
    {
        scratch_0[_S2] = scratch_0[_S2] + scratch_0[_S2 + u32(2)];
    }
    workgroupBarrier();
    if(sub_tid_0 == u32(0))
    {
        scratch_0[_S2] = scratch_0[_S2] + scratch_0[_S2 + u32(1)];
    }
    workgroupBarrier();
    if(_S2 == u32(0))
    {
        y_0[row_0] = scratch_0[i32(0)] / (1.0f + exp(- scratch_0[i32(0)])) * scratch_0[i32(16)];
    }
    return;
}

