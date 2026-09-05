@binding(4) @group(0) var<storage, read> par_buf_0 : array<vec4<u32>>;

@binding(0) @group(0) var<storage, read> q_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read> k_buf_0 : array<f32>;

@binding(2) @group(0) var<storage, read> v_buf_0 : array<f32>;

@binding(3) @group(0) var<storage, read_write> out_buf_0 : array<f32>;

var<workgroup> k_tile_0 : array<f32, i32(2048)>;

var<workgroup> v_tile_0 : array<f32, i32(2048)>;

@compute
@workgroup_size(64, 1, 1)
fn bert_flash_attention(@builtin(local_invocation_id) lid_0 : vec3<u32>, @builtin(workgroup_id) wid_0 : vec3<u32>)
{
    var tid_0 : u32 = lid_0.x;
    var p0_0 : vec4<u32> = par_buf_0[i32(0)];
    var tokens_0 : u32 = p0_0.x;
    var head_dim_0 : u32 = p0_0.z;
    var _S1 : f32 = (bitcast<f32>((p0_0.w)));
    var window_size_0 : u32 = par_buf_0[i32(1)].x;
    var half_window_0 : u32 = window_size_0 / u32(2);
    var h_0 : u32 = wid_0.y;
    var dim_0 : u32 = p0_0.y * head_dim_0;
    var _S2 : u32 = wid_0.x * u32(64);
    var q_idx_0 : u32 = _S2 + tid_0;
    var valid_q_0 : bool = q_idx_0 < tokens_0;
    var q_reg_0 : array<f32, i32(64)>;
    var d_0 : u32;
    if(valid_q_0)
    {
        var _S3 : u32 = q_idx_0 * dim_0 + h_0 * head_dim_0;
        d_0 = u32(0);
        for(;;)
        {
            if(d_0 < head_dim_0)
            {
            }
            else
            {
                break;
            }
            q_reg_0[d_0] = q_buf_0[_S3 + d_0];
            d_0 = d_0 + u32(1);
        }
    }
    else
    {
        d_0 = u32(0);
        for(;;)
        {
            if(d_0 < head_dim_0)
            {
            }
            else
            {
                break;
            }
            q_reg_0[d_0] = 0.0f;
            d_0 = d_0 + u32(1);
        }
    }
    var acc_0 : array<f32, i32(64)>;
    d_0 = u32(0);
    for(;;)
    {
        if(d_0 < head_dim_0)
        {
        }
        else
        {
            break;
        }
        acc_0[d_0] = 0.0f;
        d_0 = d_0 + u32(1);
    }
    var _S4 : u32 = min(tokens_0, _S2 + u32(64));
    var _S5 : bool = window_size_0 > u32(0);
    var _S6 : bool;
    if(_S5)
    {
        _S6 = _S2 > half_window_0;
    }
    else
    {
        _S6 = false;
    }
    if(_S6)
    {
        d_0 = _S2 - half_window_0;
    }
    else
    {
        d_0 = u32(0);
    }
    var _S7 : u32;
    if(_S5)
    {
        _S7 = min(tokens_0, _S4 + half_window_0);
    }
    else
    {
        _S7 = tokens_0;
    }
    var _S8 : u32 = (tokens_0 + u32(32) - u32(1)) / u32(32);
    var m_prev_0 : f32 = -3.4028234663852886e+38f;
    var l_prev_0 : f32 = 0.0f;
    var kt_0 : u32 = u32(0);
    for(;;)
    {
        if(kt_0 < _S8)
        {
        }
        else
        {
            break;
        }
        var k_base_0 : u32 = kt_0 * u32(32);
        if(_S5)
        {
            if((k_base_0 + u32(32)) <= d_0)
            {
                _S6 = true;
            }
            else
            {
                _S6 = k_base_0 >= _S7;
            }
        }
        else
        {
            _S6 = false;
        }
        var m_prev_1 : f32;
        var l_prev_1 : f32;
        if(_S6)
        {
            m_prev_1 = m_prev_0;
            l_prev_1 = l_prev_0;
            var _S9 : u32 = kt_0 + u32(1);
            m_prev_0 = m_prev_1;
            l_prev_0 = l_prev_1;
            kt_0 = _S9;
            continue;
        }
        var elems_per_thread_0 : u32 = u32(32) * head_dim_0 / u32(64);
        var _S10 : u32 = tid_0 * elems_per_thread_0;
        var i_0 : u32 = u32(0);
        for(;;)
        {
            if(i_0 < elems_per_thread_0)
            {
            }
            else
            {
                break;
            }
            var elem_idx_0 : u32 = _S10 + i_0;
            var k_row_0 : u32 = elem_idx_0 / head_dim_0;
            var k_col_0 : u32 = elem_idx_0 % head_dim_0;
            var global_k_token_0 : u32 = k_base_0 + k_row_0;
            if(global_k_token_0 < tokens_0)
            {
                var g_off_0 : u32 = global_k_token_0 * dim_0 + h_0 * head_dim_0 + k_col_0;
                k_tile_0[elem_idx_0] = k_buf_0[g_off_0];
                v_tile_0[elem_idx_0] = v_buf_0[g_off_0];
            }
            else
            {
                k_tile_0[elem_idx_0] = 0.0f;
                v_tile_0[elem_idx_0] = 0.0f;
            }
            i_0 = i_0 + u32(1);
        }
        workgroupBarrier();
        if(valid_q_0)
        {
            var _S11 : u32 = min(u32(32), tokens_0 - k_base_0);
            m_prev_1 = m_prev_0;
            l_prev_1 = l_prev_0;
            var k_pos_0 : u32 = u32(0);
            for(;;)
            {
                if(k_pos_0 < _S11)
                {
                }
                else
                {
                    break;
                }
                var global_k_0 : u32 = k_base_0 + k_pos_0;
                var d_1 : u32;
                if(_S5)
                {
                    if(q_idx_0 >= global_k_0)
                    {
                        d_1 = q_idx_0 - global_k_0;
                    }
                    else
                    {
                        d_1 = global_k_0 - q_idx_0;
                    }
                    if(d_1 > half_window_0)
                    {
                        k_pos_0 = k_pos_0 + u32(1);
                        continue;
                    }
                }
                var _S12 : u32 = k_pos_0 * head_dim_0;
                d_1 = u32(0);
                var score_0 : f32 = 0.0f;
                for(;;)
                {
                    if(d_1 < head_dim_0)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var score_1 : f32 = score_0 + q_reg_0[d_1] * k_tile_0[_S12 + d_1];
                    d_1 = d_1 + u32(1);
                    score_0 = score_1;
                }
                var score_2 : f32 = score_0 * _S1;
                var _S13 : f32 = max(m_prev_1, score_2);
                var alpha_0 : f32 = exp(m_prev_1 - _S13);
                var beta_0 : f32 = exp(score_2 - _S13);
                var _S14 : f32 = l_prev_1 * alpha_0 + beta_0;
                var d_2 : u32 = u32(0);
                for(;;)
                {
                    if(d_2 < head_dim_0)
                    {
                    }
                    else
                    {
                        break;
                    }
                    acc_0[d_2] = acc_0[d_2] * alpha_0 + beta_0 * v_tile_0[_S12 + d_2];
                    d_2 = d_2 + u32(1);
                }
                m_prev_1 = _S13;
                l_prev_1 = _S14;
                k_pos_0 = k_pos_0 + u32(1);
            }
        }
        else
        {
            m_prev_1 = m_prev_0;
            l_prev_1 = l_prev_0;
        }
        workgroupBarrier();
        var _S9 : u32 = kt_0 + u32(1);
        m_prev_0 = m_prev_1;
        l_prev_0 = l_prev_1;
        kt_0 = _S9;
    }
    if(valid_q_0)
    {
        if(l_prev_0 > 0.0f)
        {
            m_prev_0 = 1.0f / l_prev_0;
        }
        else
        {
            m_prev_0 = 0.0f;
        }
        var _S15 : u32 = q_idx_0 * dim_0 + h_0 * head_dim_0;
        d_0 = u32(0);
        for(;;)
        {
            if(d_0 < head_dim_0)
            {
            }
            else
            {
                break;
            }
            out_buf_0[_S15 + d_0] = acc_0[d_0] * m_prev_0;
            d_0 = d_0 + u32(1);
        }
    }
    return;
}

