@binding(3) @group(0) var<storage, read> params_0 : array<vec4<u32>>;

@binding(1) @group(0) var<storage, read> x_0 : array<f32>;

@binding(0) @group(0) var<storage, read> w_0 : array<u32>;

@binding(2) @group(0) var<storage, read_write> y_0 : array<f32>;

fn get_wid_0( wid_0 : vec3<u32>) -> u32
{
    return wid_0.x + wid_0.y * u32(65535);
}

var<workgroup> x_stage_0 : array<f32, i32(512)>;

fn block_scale_0( blk_byte_0 : u32) -> f32
{
    var word_off_0 : u32 = blk_byte_0 / u32(4);
    var byte_rem_0 : u32 = blk_byte_0 % u32(4);
    var scale_bits_0 : u32;
    if(byte_rem_0 == u32(0))
    {
        scale_bits_0 = (w_0[word_off_0] & (u32(65535)));
    }
    else
    {
        if(byte_rem_0 == u32(2))
        {
            scale_bits_0 = (((w_0[word_off_0] >> (u32(16)))) & (u32(65535)));
        }
        else
        {
            scale_bits_0 = (((((w_0[word_off_0] >> ((byte_rem_0 * u32(8))))) | (((w_0[word_off_0 + u32(1)] << (((u32(4) - byte_rem_0) * u32(8)))))))) & (u32(65535)));
        }
    }
    return (unpack2x16float((scale_bits_0)).x);
}

fn q_pair_0( qs_byte_0 : u32) -> u32
{
    var word_off_1 : u32 = qs_byte_0 / u32(4);
    var byte_rem_1 : u32 = qs_byte_0 % u32(4);
    if(byte_rem_1 == u32(0))
    {
        return (w_0[word_off_1] & (u32(65535)));
    }
    else
    {
        if(byte_rem_1 == u32(2))
        {
            return (((w_0[word_off_1] >> (u32(16)))) & (u32(65535)));
        }
        else
        {
            return (((((w_0[word_off_1] >> ((byte_rem_1 * u32(8))))) | (((w_0[word_off_1 + u32(1)] << (((u32(4) - byte_rem_1) * u32(8)))))))) & (u32(65535)));
        }
    }
}

var<workgroup> partials_0 : array<f32, i32(256)>;

@compute
@workgroup_size(32, 1, 1)
fn gemv_q4_0_fast(@builtin(local_invocation_id) lid_0 : vec3<u32>, @builtin(workgroup_id) wid_1 : vec3<u32>)
{
    var i_0 : u32;
    var _S1 : u32 = params_0[i32(0)].x;
    var k_0 : u32 = params_0[i32(0)].y;
    var nb_0 : u32 = k_0 / u32(32);
    var _S2 : u32 = nb_0 * u32(18);
    var _S3 : u32 = get_wid_0(wid_1) * u32(8);
    var tid_0 : u32 = lid_0.x;
    var _S4 : u32 = tid_0 / u32(2);
    var _S5 : u32 = ((tid_0 & (u32(1)))) * u32(8);
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
    var chunk_b_0 : u32 = u32(0);
    for(;;)
    {
        if(chunk_b_0 < nb_0)
        {
        }
        else
        {
            break;
        }
        var chunk_k_start_0 : u32 = chunk_b_0 * u32(32);
        var _S6 : u32 = min(u32(512), k_0 - chunk_k_start_0);
        i_0 = tid_0;
        for(;;)
        {
            if(i_0 < _S6)
            {
            }
            else
            {
                break;
            }
            x_stage_0[i_0] = x_0[chunk_k_start_0 + i_0];
            i_0 = i_0 + u32(32);
        }
        workgroupBarrier();
        var ib_local_0 : u32 = _S4;
        for(;;)
        {
            var _S7 : bool;
            if(ib_local_0 < u32(16))
            {
                _S7 = (chunk_b_0 + ib_local_0) < nb_0;
            }
            else
            {
                _S7 = false;
            }
            if(_S7)
            {
            }
            else
            {
                break;
            }
            var _S8 : u32 = chunk_b_0 + ib_local_0;
            var yb_stage_off_0 : u32 = ib_local_0 * u32(32) + _S5;
            var a0_0 : f32 = x_stage_0[yb_stage_off_0];
            var a2_0 : f32 = x_stage_0[yb_stage_off_0 + u32(2)];
            var a4_0 : f32 = x_stage_0[yb_stage_off_0 + u32(4)];
            var a6_0 : f32 = x_stage_0[yb_stage_off_0 + u32(6)];
            var _S9 : f32 = x_stage_0[yb_stage_off_0 + u32(1)] / 256.0f;
            var _S10 : f32 = x_stage_0[yb_stage_off_0 + u32(3)] / 256.0f;
            var _S11 : f32 = x_stage_0[yb_stage_off_0 + u32(5)] / 256.0f;
            var _S12 : f32 = x_stage_0[yb_stage_off_0 + u32(7)] / 256.0f;
            var _S13 : f32 = x_stage_0[yb_stage_off_0 + u32(16)] / 16.0f;
            var _S14 : f32 = x_stage_0[yb_stage_off_0 + u32(17)] / 4096.0f;
            var _S15 : f32 = x_stage_0[yb_stage_off_0 + u32(18)] / 16.0f;
            var _S16 : f32 = x_stage_0[yb_stage_off_0 + u32(19)] / 4096.0f;
            var _S17 : f32 = x_stage_0[yb_stage_off_0 + u32(20)] / 16.0f;
            var _S18 : f32 = x_stage_0[yb_stage_off_0 + u32(21)] / 4096.0f;
            var _S19 : f32 = x_stage_0[yb_stage_off_0 + u32(22)] / 16.0f;
            var _S20 : f32 = x_stage_0[yb_stage_off_0 + u32(23)] / 4096.0f;
            var _S21 : f32 = x_stage_0[yb_stage_off_0] + x_stage_0[yb_stage_off_0 + u32(1)] + (x_stage_0[yb_stage_off_0 + u32(2)] + x_stage_0[yb_stage_off_0 + u32(3)]) + (x_stage_0[yb_stage_off_0 + u32(4)] + x_stage_0[yb_stage_off_0 + u32(5)]) + (x_stage_0[yb_stage_off_0 + u32(6)] + x_stage_0[yb_stage_off_0 + u32(7)]) + (x_stage_0[yb_stage_off_0 + u32(16)] + x_stage_0[yb_stage_off_0 + u32(17)] + (x_stage_0[yb_stage_off_0 + u32(18)] + x_stage_0[yb_stage_off_0 + u32(19)]) + (x_stage_0[yb_stage_off_0 + u32(20)] + x_stage_0[yb_stage_off_0 + u32(21)]) + (x_stage_0[yb_stage_off_0 + u32(22)] + x_stage_0[yb_stage_off_0 + u32(23)]));
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
                var _S22 : u32 = _S3 + r_0;
                if(_S22 >= _S1)
                {
                    r_0 = r_0 + u32(1);
                    continue;
                }
                var blk_byte_1 : u32 = _S22 * _S2 + _S8 * u32(18);
                var qs_byte_1 : u32 = blk_byte_1 + u32(2) + _S5;
                var q0_0 : u32 = q_pair_0(qs_byte_1);
                var q1_0 : u32 = q_pair_0(qs_byte_1 + u32(2));
                var q2_0 : u32 = q_pair_0(qs_byte_1 + u32(4));
                var q3_0 : u32 = q_pair_0(qs_byte_1 + u32(6));
                sumf_0[r_0] = sumf_0[r_0] + block_scale_0(blk_byte_1) * (_S21 * -8.0f + (a0_0 * f32((q0_0 & (u32(15)))) + a2_0 * f32((q1_0 & (u32(15)))) + a4_0 * f32((q2_0 & (u32(15)))) + a6_0 * f32((q3_0 & (u32(15))))) + (_S9 * f32((q0_0 & (u32(3840)))) + _S10 * f32((q1_0 & (u32(3840)))) + _S11 * f32((q2_0 & (u32(3840)))) + _S12 * f32((q3_0 & (u32(3840))))) + (_S13 * f32((q0_0 & (u32(240)))) + _S15 * f32((q1_0 & (u32(240)))) + _S17 * f32((q2_0 & (u32(240)))) + _S19 * f32((q3_0 & (u32(240))))) + (_S14 * f32((q0_0 & (u32(61440)))) + _S16 * f32((q1_0 & (u32(61440)))) + _S18 * f32((q2_0 & (u32(61440)))) + _S20 * f32((q3_0 & (u32(61440))))));
                r_0 = r_0 + u32(1);
            }
            ib_local_0 = ib_local_0 + u32(16);
        }
        workgroupBarrier();
        chunk_b_0 = chunk_b_0 + u32(16);
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
    i_0 = u32(16);
    for(;;)
    {
        if(i_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_0 < i_0)
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
                partials_0[idx_0] = partials_0[idx_0] + partials_0[idx_0 + i_0];
                r_0 = r_0 + u32(1);
            }
        }
        workgroupBarrier();
        i_0 = (i_0 >> (u32(1)));
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
            var _S23 : u32 = _S3 + r_0;
            if(_S23 < _S1)
            {
                y_0[_S23] = partials_0[r_0 * u32(32)];
            }
            r_0 = r_0 + u32(1);
        }
    }
    return;
}

