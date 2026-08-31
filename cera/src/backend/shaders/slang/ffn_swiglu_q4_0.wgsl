@binding(4) @group(0) var<storage, read> params_0 : array<vec4<u32>>;

@binding(2) @group(0) var<storage, read> x_0 : array<f32>;

@binding(0) @group(0) var<storage, read> w_gate_0 : array<u32>;

@binding(1) @group(0) var<storage, read> w_up_0 : array<u32>;

@binding(3) @group(0) var<storage, read_write> y_0 : array<f32>;

fn get_wid_0( wid_0 : vec3<u32>) -> u32
{
    return wid_0.x + wid_0.y * u32(65535);
}

var<workgroup> x_stage_0 : array<f32, i32(512)>;

var<workgroup> partials_gate_0 : array<f32, i32(128)>;

var<workgroup> partials_up_0 : array<f32, i32(128)>;

fn block_scale_0( _S1 : u32) -> f32
{
    var word_0 : u32 = w_gate_0[(_S1 >> (u32(2)))];
    var scale_bits_0 : u32;
    if(((_S1 & (u32(2)))) != u32(0))
    {
        scale_bits_0 = (word_0 >> (u32(16)));
    }
    else
    {
        scale_bits_0 = (word_0 & (u32(65535)));
    }
    return (unpack2x16float((scale_bits_0)).x);
}

fn q_pair_0( _S2 : u32) -> u32
{
    var word_1 : u32 = w_gate_0[(_S2 >> (u32(2)))];
    var _S3 : u32;
    if(((_S2 & (u32(2)))) != u32(0))
    {
        _S3 = (word_1 >> (u32(16)));
    }
    else
    {
        _S3 = (word_1 & (u32(65535)));
    }
    return _S3;
}

fn block_scale_1( _S4 : u32) -> f32
{
    var word_2 : u32 = w_up_0[(_S4 >> (u32(2)))];
    var scale_bits_1 : u32;
    if(((_S4 & (u32(2)))) != u32(0))
    {
        scale_bits_1 = (word_2 >> (u32(16)));
    }
    else
    {
        scale_bits_1 = (word_2 & (u32(65535)));
    }
    return (unpack2x16float((scale_bits_1)).x);
}

fn q_pair_1( _S5 : u32) -> u32
{
    var word_3 : u32 = w_up_0[(_S5 >> (u32(2)))];
    var _S6 : u32;
    if(((_S5 & (u32(2)))) != u32(0))
    {
        _S6 = (word_3 >> (u32(16)));
    }
    else
    {
        _S6 = (word_3 & (u32(65535)));
    }
    return _S6;
}

@compute
@workgroup_size(32, 1, 1)
fn ffn_swiglu_q4_0(@builtin(local_invocation_id) lid_0 : vec3<u32>, @builtin(workgroup_id) wid_1 : vec3<u32>)
{
    var i_0 : u32;
    var _S7 : u32 = params_0[i32(0)].x;
    var k_0 : u32 = params_0[i32(0)].y;
    var nb_0 : u32 = k_0 / u32(32);
    var _S8 : u32 = nb_0 * u32(18);
    var _S9 : u32 = get_wid_0(wid_1) * u32(4);
    var tid_0 : u32 = lid_0.x;
    var _S10 : u32 = tid_0 / u32(2);
    var _S11 : u32 = ((tid_0 & (u32(1)))) * u32(8);
    var sum_gate_0 : array<f32, i32(4)>;
    var sum_up_0 : array<f32, i32(4)>;
    var r_0 : u32 = u32(0);
    for(;;)
    {
        if(r_0 < u32(4))
        {
        }
        else
        {
            break;
        }
        sum_gate_0[r_0] = 0.0f;
        sum_up_0[r_0] = 0.0f;
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
        var _S12 : u32 = min(u32(512), k_0 - chunk_k_start_0) / u32(4);
        i_0 = tid_0;
        for(;;)
        {
            if(i_0 < _S12)
            {
            }
            else
            {
                break;
            }
            var _S13 : u32 = i_0 * u32(4);
            var base_idx_0 : u32 = chunk_k_start_0 + _S13;
            x_stage_0[_S13] = x_0[base_idx_0];
            x_stage_0[_S13 + u32(1)] = x_0[base_idx_0 + u32(1)];
            x_stage_0[_S13 + u32(2)] = x_0[base_idx_0 + u32(2)];
            x_stage_0[_S13 + u32(3)] = x_0[base_idx_0 + u32(3)];
            i_0 = i_0 + u32(32);
        }
        workgroupBarrier();
        var ib_local_0 : u32 = _S10;
        for(;;)
        {
            var _S14 : bool;
            if(ib_local_0 < u32(16))
            {
                _S14 = (chunk_b_0 + ib_local_0) < nb_0;
            }
            else
            {
                _S14 = false;
            }
            if(_S14)
            {
            }
            else
            {
                break;
            }
            var _S15 : u32 = chunk_b_0 + ib_local_0;
            var yb_stage_off_0 : u32 = ib_local_0 * u32(32) + _S11;
            var a0_0 : f32 = x_stage_0[yb_stage_off_0];
            var a2_0 : f32 = x_stage_0[yb_stage_off_0 + u32(2)];
            var a4_0 : f32 = x_stage_0[yb_stage_off_0 + u32(4)];
            var a6_0 : f32 = x_stage_0[yb_stage_off_0 + u32(6)];
            var _S16 : f32 = x_stage_0[yb_stage_off_0 + u32(1)] / 256.0f;
            var _S17 : f32 = x_stage_0[yb_stage_off_0 + u32(3)] / 256.0f;
            var _S18 : f32 = x_stage_0[yb_stage_off_0 + u32(5)] / 256.0f;
            var _S19 : f32 = x_stage_0[yb_stage_off_0 + u32(7)] / 256.0f;
            var _S20 : f32 = x_stage_0[yb_stage_off_0 + u32(16)] / 16.0f;
            var _S21 : f32 = x_stage_0[yb_stage_off_0 + u32(17)] / 4096.0f;
            var _S22 : f32 = x_stage_0[yb_stage_off_0 + u32(18)] / 16.0f;
            var _S23 : f32 = x_stage_0[yb_stage_off_0 + u32(19)] / 4096.0f;
            var _S24 : f32 = x_stage_0[yb_stage_off_0 + u32(20)] / 16.0f;
            var _S25 : f32 = x_stage_0[yb_stage_off_0 + u32(21)] / 4096.0f;
            var _S26 : f32 = x_stage_0[yb_stage_off_0 + u32(22)] / 16.0f;
            var _S27 : f32 = x_stage_0[yb_stage_off_0 + u32(23)] / 4096.0f;
            var _S28 : f32 = x_stage_0[yb_stage_off_0] + x_stage_0[yb_stage_off_0 + u32(1)] + (x_stage_0[yb_stage_off_0 + u32(2)] + x_stage_0[yb_stage_off_0 + u32(3)]) + (x_stage_0[yb_stage_off_0 + u32(4)] + x_stage_0[yb_stage_off_0 + u32(5)]) + (x_stage_0[yb_stage_off_0 + u32(6)] + x_stage_0[yb_stage_off_0 + u32(7)]) + (x_stage_0[yb_stage_off_0 + u32(16)] + x_stage_0[yb_stage_off_0 + u32(17)] + (x_stage_0[yb_stage_off_0 + u32(18)] + x_stage_0[yb_stage_off_0 + u32(19)]) + (x_stage_0[yb_stage_off_0 + u32(20)] + x_stage_0[yb_stage_off_0 + u32(21)]) + (x_stage_0[yb_stage_off_0 + u32(22)] + x_stage_0[yb_stage_off_0 + u32(23)]));
            r_0 = u32(0);
            for(;;)
            {
                if(r_0 < u32(4))
                {
                }
                else
                {
                    break;
                }
                var _S29 : u32 = _S9 + r_0;
                if(_S29 >= _S7)
                {
                    r_0 = r_0 + u32(1);
                    continue;
                }
                var blk_byte_0 : u32 = _S29 * _S8 + _S15 * u32(18);
                var qs_byte_0 : u32 = blk_byte_0 + u32(2) + _S11;
                var _S30 : u32 = q_pair_0(qs_byte_0);
                var _S31 : u32 = qs_byte_0 + u32(2);
                var _S32 : u32 = q_pair_0(_S31);
                var _S33 : u32 = qs_byte_0 + u32(4);
                var _S34 : u32 = q_pair_0(_S33);
                var _S35 : u32 = qs_byte_0 + u32(6);
                var _S36 : u32 = q_pair_0(_S35);
                var _S37 : f32 = _S28 * -8.0f;
                sum_gate_0[r_0] = sum_gate_0[r_0] + block_scale_0(blk_byte_0) * (_S37 + (a0_0 * f32((_S30 & (u32(15)))) + a2_0 * f32((_S32 & (u32(15)))) + a4_0 * f32((_S34 & (u32(15)))) + a6_0 * f32((_S36 & (u32(15))))) + (_S16 * f32((_S30 & (u32(3840)))) + _S17 * f32((_S32 & (u32(3840)))) + _S18 * f32((_S34 & (u32(3840)))) + _S19 * f32((_S36 & (u32(3840))))) + (_S20 * f32((_S30 & (u32(240)))) + _S22 * f32((_S32 & (u32(240)))) + _S24 * f32((_S34 & (u32(240)))) + _S26 * f32((_S36 & (u32(240))))) + (_S21 * f32((_S30 & (u32(61440)))) + _S23 * f32((_S32 & (u32(61440)))) + _S25 * f32((_S34 & (u32(61440)))) + _S27 * f32((_S36 & (u32(61440))))));
                var _S38 : u32 = q_pair_1(qs_byte_0);
                var _S39 : u32 = q_pair_1(_S31);
                var _S40 : u32 = q_pair_1(_S33);
                var _S41 : u32 = q_pair_1(_S35);
                sum_up_0[r_0] = sum_up_0[r_0] + block_scale_1(blk_byte_0) * (_S37 + (a0_0 * f32((_S38 & (u32(15)))) + a2_0 * f32((_S39 & (u32(15)))) + a4_0 * f32((_S40 & (u32(15)))) + a6_0 * f32((_S41 & (u32(15))))) + (_S16 * f32((_S38 & (u32(3840)))) + _S17 * f32((_S39 & (u32(3840)))) + _S18 * f32((_S40 & (u32(3840)))) + _S19 * f32((_S41 & (u32(3840))))) + (_S20 * f32((_S38 & (u32(240)))) + _S22 * f32((_S39 & (u32(240)))) + _S24 * f32((_S40 & (u32(240)))) + _S26 * f32((_S41 & (u32(240))))) + (_S21 * f32((_S38 & (u32(61440)))) + _S23 * f32((_S39 & (u32(61440)))) + _S25 * f32((_S40 & (u32(61440)))) + _S27 * f32((_S41 & (u32(61440))))));
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
        if(r_0 < u32(4))
        {
        }
        else
        {
            break;
        }
        var _S42 : u32 = r_0 * u32(32) + tid_0;
        partials_gate_0[_S42] = sum_gate_0[r_0];
        partials_up_0[_S42] = sum_up_0[r_0];
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
                if(r_0 < u32(4))
                {
                }
                else
                {
                    break;
                }
                var idx_0 : u32 = r_0 * u32(32) + tid_0;
                var _S43 : u32 = idx_0 + i_0;
                partials_gate_0[idx_0] = partials_gate_0[idx_0] + partials_gate_0[_S43];
                partials_up_0[idx_0] = partials_up_0[idx_0] + partials_up_0[_S43];
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
            if(r_0 < u32(4))
            {
            }
            else
            {
                break;
            }
            var _S44 : u32 = _S9 + r_0;
            if(_S44 < _S7)
            {
                var _S45 : u32 = r_0 * u32(32);
                y_0[_S44] = partials_gate_0[_S45] * (1.0f / (1.0f + exp2(- clamp(partials_gate_0[_S45], -80.0f, 80.0f) * 1.4426950216293335f))) * partials_up_0[_S45];
            }
            r_0 = r_0 + u32(1);
        }
    }
    return;
}

