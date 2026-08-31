@binding(3) @group(0) var<storage, read> params_0 : array<vec4<u32>>;

@binding(1) @group(0) var<storage, read> x_0 : array<f32>;

@binding(0) @group(0) var<storage, read> w_0 : array<u32>;

@binding(2) @group(0) var<storage, read_write> y_0 : array<f32>;

fn get_wid_0( wid_0 : vec3<u32>) -> u32
{
    return wid_0.x + wid_0.y * u32(65535);
}

fn block_scale_0( blk_byte_0 : u32) -> f32
{
    var word_0 : u32 = w_0[(blk_byte_0 >> (u32(2)))];
    var scale_bits_0 : u32;
    if(((blk_byte_0 & (u32(2)))) != u32(0))
    {
        scale_bits_0 = (word_0 >> (u32(16)));
    }
    else
    {
        scale_bits_0 = (word_0 & (u32(65535)));
    }
    return (unpack2x16float((scale_bits_0)).x);
}

fn q_pair_0( qs_byte_0 : u32) -> u32
{
    var word_1 : u32 = w_0[(qs_byte_0 >> (u32(2)))];
    var _S1 : u32;
    if(((qs_byte_0 & (u32(2)))) != u32(0))
    {
        _S1 = (word_1 >> (u32(16)));
    }
    else
    {
        _S1 = (word_1 & (u32(65535)));
    }
    return _S1;
}

var<workgroup> partials_0 : array<f32, i32(256)>;

@compute
@workgroup_size(32, 1, 1)
fn gemv_q4_0_fast(@builtin(local_invocation_id) lid_0 : vec3<u32>, @builtin(workgroup_id) wid_1 : vec3<u32>)
{
    var _S2 : u32 = params_0[i32(0)].x;
    var nb_0 : u32 = params_0[i32(0)].y / u32(32);
    var _S3 : u32 = nb_0 * u32(18);
    var _S4 : u32 = get_wid_0(wid_1) * u32(8);
    var tid_0 : u32 = lid_0.x;
    var ix_0 : u32 = tid_0 / u32(2);
    var il_0 : u32 = ((tid_0 & (u32(1)))) * u32(8);
    var sumf_0 : array<f32, i32(8)>;
    var r_0 : u32 = u32(0);
    for(;;)
    {
        if(r_0 < u32(8))
        {
        }
        else
        {
            break;
        }
        sumf_0[r_0] = 0.0f;
        r_0 = r_0 + u32(1);
    }
    var _S5 : u32 = ix_0 * u32(32) + il_0;
    var ib_0 : u32 = ix_0;
    var yb_off_0 : u32 = _S5;
    for(;;)
    {
        if(ib_0 < nb_0)
        {
        }
        else
        {
            break;
        }
        var a0_0 : f32 = x_0[yb_off_0];
        var a1_0 : f32 = x_0[yb_off_0 + u32(1)];
        var a2_0 : f32 = x_0[yb_off_0 + u32(2)];
        var a3_0 : f32 = x_0[yb_off_0 + u32(3)];
        var a4_0 : f32 = x_0[yb_off_0 + u32(4)];
        var a5_0 : f32 = x_0[yb_off_0 + u32(5)];
        var a6_0 : f32 = x_0[yb_off_0 + u32(6)];
        var a7_0 : f32 = x_0[yb_off_0 + u32(7)];
        var a8_0 : f32 = x_0[yb_off_0 + u32(16)];
        var a9_0 : f32 = x_0[yb_off_0 + u32(17)];
        var a10_0 : f32 = x_0[yb_off_0 + u32(18)];
        var a11_0 : f32 = x_0[yb_off_0 + u32(19)];
        var a12_0 : f32 = x_0[yb_off_0 + u32(20)];
        var a13_0 : f32 = x_0[yb_off_0 + u32(21)];
        var a14_0 : f32 = x_0[yb_off_0 + u32(22)];
        var a15_0 : f32 = x_0[yb_off_0 + u32(23)];
        var _S6 : f32 = a1_0 / 256.0f;
        var _S7 : f32 = a3_0 / 256.0f;
        var _S8 : f32 = a5_0 / 256.0f;
        var _S9 : f32 = a7_0 / 256.0f;
        var _S10 : f32 = a8_0 / 16.0f;
        var _S11 : f32 = a9_0 / 4096.0f;
        var _S12 : f32 = a10_0 / 16.0f;
        var _S13 : f32 = a11_0 / 4096.0f;
        var _S14 : f32 = a12_0 / 16.0f;
        var _S15 : f32 = a13_0 / 4096.0f;
        var _S16 : f32 = a14_0 / 16.0f;
        var _S17 : f32 = a15_0 / 4096.0f;
        var _S18 : f32 = a0_0 + a1_0 + (a2_0 + a3_0) + (a4_0 + a5_0) + (a6_0 + a7_0) + (a8_0 + a9_0 + (a10_0 + a11_0) + (a12_0 + a13_0) + (a14_0 + a15_0));
        r_0 = u32(0);
        for(;;)
        {
            if(r_0 < u32(8))
            {
            }
            else
            {
                break;
            }
            var _S19 : u32 = _S4 + r_0;
            if(_S19 >= _S2)
            {
                r_0 = r_0 + u32(1);
                continue;
            }
            var blk_byte_1 : u32 = _S19 * _S3 + ib_0 * u32(18);
            var qs_byte_1 : u32 = blk_byte_1 + u32(2) + il_0;
            var q0_0 : u32 = q_pair_0(qs_byte_1);
            var q1_0 : u32 = q_pair_0(qs_byte_1 + u32(2));
            var q2_0 : u32 = q_pair_0(qs_byte_1 + u32(4));
            var q3_0 : u32 = q_pair_0(qs_byte_1 + u32(6));
            sumf_0[r_0] = sumf_0[r_0] + block_scale_0(blk_byte_1) * (_S18 * -8.0f + (a0_0 * f32((q0_0 & (u32(15)))) + a2_0 * f32((q1_0 & (u32(15)))) + a4_0 * f32((q2_0 & (u32(15)))) + a6_0 * f32((q3_0 & (u32(15))))) + (_S6 * f32((q0_0 & (u32(3840)))) + _S7 * f32((q1_0 & (u32(3840)))) + _S8 * f32((q2_0 & (u32(3840)))) + _S9 * f32((q3_0 & (u32(3840))))) + (_S10 * f32((q0_0 & (u32(240)))) + _S12 * f32((q1_0 & (u32(240)))) + _S14 * f32((q2_0 & (u32(240)))) + _S16 * f32((q3_0 & (u32(240))))) + (_S11 * f32((q0_0 & (u32(61440)))) + _S13 * f32((q1_0 & (u32(61440)))) + _S15 * f32((q2_0 & (u32(61440)))) + _S17 * f32((q3_0 & (u32(61440))))));
            r_0 = r_0 + u32(1);
        }
        var yb_off_1 : u32 = yb_off_0 + u32(512);
        ib_0 = ib_0 + u32(16);
        yb_off_0 = yb_off_1;
    }
    r_0 = u32(0);
    for(;;)
    {
        if(r_0 < u32(8))
        {
        }
        else
        {
            break;
        }
        partials_0[r_0 * u32(32) + tid_0] = sumf_0[r_0];
        r_0 = r_0 + u32(1);
    }
    workgroupBarrier();
    var stride_0 : u32 = u32(16);
    for(;;)
    {
        if(stride_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_0 < stride_0)
        {
            r_0 = u32(0);
            for(;;)
            {
                if(r_0 < u32(8))
                {
                }
                else
                {
                    break;
                }
                var idx_0 : u32 = r_0 * u32(32) + tid_0;
                partials_0[idx_0] = partials_0[idx_0] + partials_0[idx_0 + stride_0];
                r_0 = r_0 + u32(1);
            }
        }
        workgroupBarrier();
        stride_0 = (stride_0 >> (u32(1)));
    }
    if(tid_0 == u32(0))
    {
        r_0 = u32(0);
        for(;;)
        {
            if(r_0 < u32(8))
            {
            }
            else
            {
                break;
            }
            var _S20 : u32 = _S4 + r_0;
            if(_S20 < _S2)
            {
                y_0[_S20] = partials_0[r_0 * u32(32)];
            }
            r_0 = r_0 + u32(1);
        }
    }
    return;
}

