@binding(7) @group(0) var<storage, read> params_0 : array<vec4<u32>>;

@binding(3) @group(0) var<storage, read> x_0 : array<f32>;

@binding(0) @group(0) var<storage, read> w_q_0 : array<u32>;

@binding(1) @group(0) var<storage, read> w_k_0 : array<u32>;

@binding(2) @group(0) var<storage, read> w_v_0 : array<u32>;

@binding(4) @group(0) var<storage, read_write> y_q_0 : array<f32>;

@binding(5) @group(0) var<storage, read_write> y_k_0 : array<f32>;

@binding(6) @group(0) var<storage, read_write> y_v_0 : array<f32>;

fn get_wid_0( wid_0 : vec3<u32>) -> u32
{
    return wid_0.x + wid_0.y * u32(65535);
}

var<workgroup> x_stage_0 : array<f32, i32(512)>;

fn block_scale_q_0( blk_byte_0 : u32) -> f32
{
    var word_0 : u32 = w_q_0[(blk_byte_0 >> (u32(2)))];
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

fn q_pair_q_0( qs_byte_0 : u32) -> u32
{
    var word_1 : u32 = w_q_0[(qs_byte_0 >> (u32(2)))];
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

fn block_scale_k_0( blk_byte_1 : u32) -> f32
{
    var word_2 : u32 = w_k_0[(blk_byte_1 >> (u32(2)))];
    var scale_bits_1 : u32;
    if(((blk_byte_1 & (u32(2)))) != u32(0))
    {
        scale_bits_1 = (word_2 >> (u32(16)));
    }
    else
    {
        scale_bits_1 = (word_2 & (u32(65535)));
    }
    return (unpack2x16float((scale_bits_1)).x);
}

fn q_pair_k_0( qs_byte_1 : u32) -> u32
{
    var word_3 : u32 = w_k_0[(qs_byte_1 >> (u32(2)))];
    var _S2 : u32;
    if(((qs_byte_1 & (u32(2)))) != u32(0))
    {
        _S2 = (word_3 >> (u32(16)));
    }
    else
    {
        _S2 = (word_3 & (u32(65535)));
    }
    return _S2;
}

fn block_scale_v_0( blk_byte_2 : u32) -> f32
{
    var word_4 : u32 = w_v_0[(blk_byte_2 >> (u32(2)))];
    var scale_bits_2 : u32;
    if(((blk_byte_2 & (u32(2)))) != u32(0))
    {
        scale_bits_2 = (word_4 >> (u32(16)));
    }
    else
    {
        scale_bits_2 = (word_4 & (u32(65535)));
    }
    return (unpack2x16float((scale_bits_2)).x);
}

fn q_pair_v_0( qs_byte_2 : u32) -> u32
{
    var word_5 : u32 = w_v_0[(qs_byte_2 >> (u32(2)))];
    var _S3 : u32;
    if(((qs_byte_2 & (u32(2)))) != u32(0))
    {
        _S3 = (word_5 >> (u32(16)));
    }
    else
    {
        _S3 = (word_5 & (u32(65535)));
    }
    return _S3;
}

var<workgroup> partials_0 : array<f32, i32(128)>;

@compute
@workgroup_size(32, 1, 1)
fn gemv_q4_0_qkv(@builtin(local_invocation_id) lid_0 : vec3<u32>, @builtin(workgroup_id) wid_1 : vec3<u32>)
{
    var i_0 : u32;
    var m_q_0 : u32 = params_0[i32(0)].x;
    var m_kv_0 : u32 = params_0[i32(0)].y;
    var k_0 : u32 = params_0[i32(0)].z;
    var nb_0 : u32 = k_0 / u32(32);
    var _S4 : u32 = nb_0 * u32(18);
    var global_r0_0 : u32 = get_wid_0(wid_1) * u32(4);
    var tid_0 : u32 = lid_0.x;
    var _S5 : u32 = tid_0 / u32(2);
    var _S6 : u32 = ((tid_0 & (u32(1)))) * u32(8);
    var which_matrix_0 : u32;
    var r0_0 : u32;
    var m_cur_0 : u32;
    if(global_r0_0 < m_q_0)
    {
        which_matrix_0 = u32(0);
        r0_0 = global_r0_0;
        m_cur_0 = m_q_0;
    }
    else
    {
        if(global_r0_0 < (m_q_0 + m_kv_0))
        {
            var _S7 : u32 = global_r0_0 - m_q_0;
            which_matrix_0 = u32(1);
            r0_0 = _S7;
        }
        else
        {
            var _S8 : u32 = global_r0_0 - m_q_0 - m_kv_0;
            which_matrix_0 = u32(2);
            r0_0 = _S8;
        }
        m_cur_0 = m_kv_0;
    }
    var sumf_0 : array<f32, i32(4)>;
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
        var _S9 : u32 = min(u32(512), k_0 - chunk_k_start_0) / u32(4);
        i_0 = tid_0;
        for(;;)
        {
            if(i_0 < _S9)
            {
            }
            else
            {
                break;
            }
            var _S10 : u32 = i_0 * u32(4);
            var base_idx_0 : u32 = chunk_k_start_0 + _S10;
            x_stage_0[_S10] = x_0[base_idx_0];
            x_stage_0[_S10 + u32(1)] = x_0[base_idx_0 + u32(1)];
            x_stage_0[_S10 + u32(2)] = x_0[base_idx_0 + u32(2)];
            x_stage_0[_S10 + u32(3)] = x_0[base_idx_0 + u32(3)];
            i_0 = i_0 + u32(32);
        }
        workgroupBarrier();
        var ib_local_0 : u32 = _S5;
        for(;;)
        {
            var _S11 : bool;
            if(ib_local_0 < u32(16))
            {
                _S11 = (chunk_b_0 + ib_local_0) < nb_0;
            }
            else
            {
                _S11 = false;
            }
            if(_S11)
            {
            }
            else
            {
                break;
            }
            var _S12 : u32 = chunk_b_0 + ib_local_0;
            var yb_stage_off_0 : u32 = ib_local_0 * u32(32) + _S6;
            var a0_0 : f32 = x_stage_0[yb_stage_off_0];
            var a2_0 : f32 = x_stage_0[yb_stage_off_0 + u32(2)];
            var a4_0 : f32 = x_stage_0[yb_stage_off_0 + u32(4)];
            var a6_0 : f32 = x_stage_0[yb_stage_off_0 + u32(6)];
            var _S13 : f32 = x_stage_0[yb_stage_off_0 + u32(1)] / 256.0f;
            var _S14 : f32 = x_stage_0[yb_stage_off_0 + u32(3)] / 256.0f;
            var _S15 : f32 = x_stage_0[yb_stage_off_0 + u32(5)] / 256.0f;
            var _S16 : f32 = x_stage_0[yb_stage_off_0 + u32(7)] / 256.0f;
            var _S17 : f32 = x_stage_0[yb_stage_off_0 + u32(16)] / 16.0f;
            var _S18 : f32 = x_stage_0[yb_stage_off_0 + u32(17)] / 4096.0f;
            var _S19 : f32 = x_stage_0[yb_stage_off_0 + u32(18)] / 16.0f;
            var _S20 : f32 = x_stage_0[yb_stage_off_0 + u32(19)] / 4096.0f;
            var _S21 : f32 = x_stage_0[yb_stage_off_0 + u32(20)] / 16.0f;
            var _S22 : f32 = x_stage_0[yb_stage_off_0 + u32(21)] / 4096.0f;
            var _S23 : f32 = x_stage_0[yb_stage_off_0 + u32(22)] / 16.0f;
            var _S24 : f32 = x_stage_0[yb_stage_off_0 + u32(23)] / 4096.0f;
            var _S25 : f32 = x_stage_0[yb_stage_off_0] + x_stage_0[yb_stage_off_0 + u32(1)] + (x_stage_0[yb_stage_off_0 + u32(2)] + x_stage_0[yb_stage_off_0 + u32(3)]) + (x_stage_0[yb_stage_off_0 + u32(4)] + x_stage_0[yb_stage_off_0 + u32(5)]) + (x_stage_0[yb_stage_off_0 + u32(6)] + x_stage_0[yb_stage_off_0 + u32(7)]) + (x_stage_0[yb_stage_off_0 + u32(16)] + x_stage_0[yb_stage_off_0 + u32(17)] + (x_stage_0[yb_stage_off_0 + u32(18)] + x_stage_0[yb_stage_off_0 + u32(19)]) + (x_stage_0[yb_stage_off_0 + u32(20)] + x_stage_0[yb_stage_off_0 + u32(21)]) + (x_stage_0[yb_stage_off_0 + u32(22)] + x_stage_0[yb_stage_off_0 + u32(23)]));
            if(which_matrix_0 == u32(0))
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
                    var _S26 : u32 = r0_0 + r_0;
                    if(_S26 >= m_cur_0)
                    {
                        r_0 = r_0 + u32(1);
                        continue;
                    }
                    var blk_byte_3 : u32 = _S26 * _S4 + _S12 * u32(18);
                    var qs_byte_3 : u32 = blk_byte_3 + u32(2) + _S6;
                    var q0_0 : u32 = q_pair_q_0(qs_byte_3);
                    var q1_0 : u32 = q_pair_q_0(qs_byte_3 + u32(2));
                    var q2_0 : u32 = q_pair_q_0(qs_byte_3 + u32(4));
                    var q3_0 : u32 = q_pair_q_0(qs_byte_3 + u32(6));
                    sumf_0[r_0] = sumf_0[r_0] + block_scale_q_0(blk_byte_3) * (_S25 * -8.0f + (a0_0 * f32((q0_0 & (u32(15)))) + a2_0 * f32((q1_0 & (u32(15)))) + a4_0 * f32((q2_0 & (u32(15)))) + a6_0 * f32((q3_0 & (u32(15))))) + (_S13 * f32((q0_0 & (u32(3840)))) + _S14 * f32((q1_0 & (u32(3840)))) + _S15 * f32((q2_0 & (u32(3840)))) + _S16 * f32((q3_0 & (u32(3840))))) + (_S17 * f32((q0_0 & (u32(240)))) + _S19 * f32((q1_0 & (u32(240)))) + _S21 * f32((q2_0 & (u32(240)))) + _S23 * f32((q3_0 & (u32(240))))) + (_S18 * f32((q0_0 & (u32(61440)))) + _S20 * f32((q1_0 & (u32(61440)))) + _S22 * f32((q2_0 & (u32(61440)))) + _S24 * f32((q3_0 & (u32(61440))))));
                    r_0 = r_0 + u32(1);
                }
            }
            else
            {
                if(which_matrix_0 == u32(1))
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
                        var _S27 : u32 = r0_0 + r_0;
                        if(_S27 >= m_cur_0)
                        {
                            r_0 = r_0 + u32(1);
                            continue;
                        }
                        var blk_byte_4 : u32 = _S27 * _S4 + _S12 * u32(18);
                        var qs_byte_4 : u32 = blk_byte_4 + u32(2) + _S6;
                        var q0_1 : u32 = q_pair_k_0(qs_byte_4);
                        var q1_1 : u32 = q_pair_k_0(qs_byte_4 + u32(2));
                        var q2_1 : u32 = q_pair_k_0(qs_byte_4 + u32(4));
                        var q3_1 : u32 = q_pair_k_0(qs_byte_4 + u32(6));
                        sumf_0[r_0] = sumf_0[r_0] + block_scale_k_0(blk_byte_4) * (_S25 * -8.0f + (a0_0 * f32((q0_1 & (u32(15)))) + a2_0 * f32((q1_1 & (u32(15)))) + a4_0 * f32((q2_1 & (u32(15)))) + a6_0 * f32((q3_1 & (u32(15))))) + (_S13 * f32((q0_1 & (u32(3840)))) + _S14 * f32((q1_1 & (u32(3840)))) + _S15 * f32((q2_1 & (u32(3840)))) + _S16 * f32((q3_1 & (u32(3840))))) + (_S17 * f32((q0_1 & (u32(240)))) + _S19 * f32((q1_1 & (u32(240)))) + _S21 * f32((q2_1 & (u32(240)))) + _S23 * f32((q3_1 & (u32(240))))) + (_S18 * f32((q0_1 & (u32(61440)))) + _S20 * f32((q1_1 & (u32(61440)))) + _S22 * f32((q2_1 & (u32(61440)))) + _S24 * f32((q3_1 & (u32(61440))))));
                        r_0 = r_0 + u32(1);
                    }
                }
                else
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
                        var _S28 : u32 = r0_0 + r_0;
                        if(_S28 >= m_cur_0)
                        {
                            r_0 = r_0 + u32(1);
                            continue;
                        }
                        var blk_byte_5 : u32 = _S28 * _S4 + _S12 * u32(18);
                        var qs_byte_5 : u32 = blk_byte_5 + u32(2) + _S6;
                        var q0_2 : u32 = q_pair_v_0(qs_byte_5);
                        var q1_2 : u32 = q_pair_v_0(qs_byte_5 + u32(2));
                        var q2_2 : u32 = q_pair_v_0(qs_byte_5 + u32(4));
                        var q3_2 : u32 = q_pair_v_0(qs_byte_5 + u32(6));
                        sumf_0[r_0] = sumf_0[r_0] + block_scale_v_0(blk_byte_5) * (_S25 * -8.0f + (a0_0 * f32((q0_2 & (u32(15)))) + a2_0 * f32((q1_2 & (u32(15)))) + a4_0 * f32((q2_2 & (u32(15)))) + a6_0 * f32((q3_2 & (u32(15))))) + (_S13 * f32((q0_2 & (u32(3840)))) + _S14 * f32((q1_2 & (u32(3840)))) + _S15 * f32((q2_2 & (u32(3840)))) + _S16 * f32((q3_2 & (u32(3840))))) + (_S17 * f32((q0_2 & (u32(240)))) + _S19 * f32((q1_2 & (u32(240)))) + _S21 * f32((q2_2 & (u32(240)))) + _S23 * f32((q3_2 & (u32(240))))) + (_S18 * f32((q0_2 & (u32(61440)))) + _S20 * f32((q1_2 & (u32(61440)))) + _S22 * f32((q2_2 & (u32(61440)))) + _S24 * f32((q3_2 & (u32(61440))))));
                        r_0 = r_0 + u32(1);
                    }
                }
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
                if(r_0 < u32(4))
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
            if(r_0 < u32(4))
            {
            }
            else
            {
                break;
            }
            var _S29 : u32 = r0_0 + r_0;
            if(_S29 < m_cur_0)
            {
                if(which_matrix_0 == u32(0))
                {
                    y_q_0[_S29] = partials_0[r_0 * u32(32)];
                }
                else
                {
                    if(which_matrix_0 == u32(1))
                    {
                        y_k_0[_S29] = partials_0[r_0 * u32(32)];
                    }
                    else
                    {
                        y_v_0[_S29] = partials_0[r_0 * u32(32)];
                    }
                }
            }
            r_0 = r_0 + u32(1);
        }
    }
    return;
}

