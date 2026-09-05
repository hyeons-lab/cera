#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 105 "cera/src/backend/shaders/slang/bert_flash_attention.slang"
struct KernelContext_0
{
    packed_uint4 device* par_buf_0;
    float device* q_buf_0;
    float device* k_buf_0;
    float device* v_buf_0;
    float device* out_buf_0;
    array<float, int(2048)> threadgroup* k_tile_0;
    array<float, int(2048)> threadgroup* v_tile_0;
};


#line 33
[[kernel]] void bert_flash_attention(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 wid_0 [[threadgroup_position_in_grid]], packed_uint4 device* par_buf_1 [[buffer(4)]], float device* q_buf_1 [[buffer(0)]], float device* k_buf_1 [[buffer(1)]], float device* v_buf_1 [[buffer(2)]], float device* out_buf_1 [[buffer(3)]])
{
    thread KernelContext_0 kernelContext_0;

#line 35
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 35
    (&kernelContext_0)->q_buf_0 = q_buf_1;

#line 35
    (&kernelContext_0)->k_buf_0 = k_buf_1;

#line 35
    (&kernelContext_0)->v_buf_0 = v_buf_1;

#line 35
    (&kernelContext_0)->out_buf_0 = out_buf_1;

#line 35
    threadgroup array<float, int(2048)> k_tile_1;

#line 35
    (&kernelContext_0)->k_tile_0 = &k_tile_1;

#line 35
    threadgroup array<float, int(2048)> v_tile_1;

#line 35
    (&kernelContext_0)->v_tile_0 = &v_tile_1;

    uint tid_0 = lid_0.x;

#line 37
    uint4 _S1 = uint4(*(par_buf_1+int(0))) ;



    uint tokens_0 = _S1.x;

    uint head_dim_0 = _S1.z;
    float _S2 = (as_type<float>((_S1.w)));
    uint window_size_0 = (uint4(*(par_buf_1+int(1))) ).x;
    uint half_window_0 = window_size_0 / 2U;

    uint h_0 = wid_0.y;
    uint dim_0 = _S1.y * head_dim_0;

    uint _S3 = wid_0.x * 64U;

#line 51
    uint q_idx_0 = _S3 + tid_0;
    bool valid_q_0 = q_idx_0 < tokens_0;


    if(head_dim_0 > 64U)
    {

#line 56
        return;
    }


    thread array<float, int(64)> q_reg_0;

#line 60
    uint d_0;
    if(valid_q_0)
    {

#line 62
        uint _S4 = q_idx_0 * dim_0 + h_0 * head_dim_0;

#line 62
        d_0 = 0U;
        for(;;)
        {

#line 63
            if(d_0 < head_dim_0)
            {
            }
            else
            {

#line 63
                break;
            }

#line 64
            q_reg_0[d_0] = (&kernelContext_0)->q_buf_0[_S4 + d_0];

#line 63
            d_0 = d_0 + 1U;

#line 63
        }

#line 61
    }
    else
    {

#line 61
        d_0 = 0U;

#line 67
        for(;;)
        {

#line 67
            if(d_0 < head_dim_0)
            {
            }
            else
            {

#line 67
                break;
            }

#line 68
            q_reg_0[d_0] = 0.0f;

#line 67
            d_0 = d_0 + 1U;

#line 67
        }

#line 61
    }

#line 75
    thread array<float, int(64)> acc_0;

#line 75
    d_0 = 0U;
    for(;;)
    {

#line 76
        if(d_0 < head_dim_0)
        {
        }
        else
        {

#line 76
            break;
        }

#line 77
        acc_0[d_0] = 0.0f;

#line 76
        d_0 = d_0 + 1U;

#line 76
    }

#line 82
    uint _S5 = min(tokens_0, _S3 + 64U);
    bool _S6 = window_size_0 > 0U;

#line 83
    bool _S7;

#line 83
    if(_S6)
    {

#line 83
        _S7 = _S3 > half_window_0;

#line 83
    }
    else
    {

#line 83
        _S7 = false;

#line 83
    }

#line 83
    if(_S7)
    {

#line 83
        d_0 = _S3 - half_window_0;

#line 83
    }
    else
    {

#line 83
        d_0 = 0U;

#line 83
    }

#line 83
    uint _S8;
    if(_S6)
    {

#line 84
        _S8 = min(tokens_0, _S5 + half_window_0);

#line 84
    }
    else
    {

#line 84
        _S8 = tokens_0;

#line 84
    }


    uint _S9 = (tokens_0 + 32U - 1U) / 32U;

#line 87
    float m_prev_0 = -3.4028234663852886e+38f;

#line 87
    float l_prev_0 = 0.0f;

#line 87
    uint kt_0 = 0U;
    for(;;)
    {

#line 88
        if(kt_0 < _S9)
        {
        }
        else
        {

#line 88
            break;
        }

#line 89
        uint k_base_0 = kt_0 * 32U;


        if(_S6)
        {

#line 92
            if((k_base_0 + 32U) <= d_0)
            {

#line 92
                _S7 = true;

#line 92
            }
            else
            {

#line 92
                _S7 = k_base_0 >= _S8;

#line 92
            }

#line 92
        }
        else
        {

#line 92
            _S7 = false;

#line 92
        }

#line 92
        float m_prev_1;

#line 92
        float l_prev_1;

#line 92
        if(_S7)
        {

#line 92
            m_prev_1 = m_prev_0;

#line 92
            l_prev_1 = l_prev_0;

#line 88
            uint _S10 = kt_0 + 1U;

#line 88
            m_prev_0 = m_prev_1;

#line 88
            l_prev_0 = l_prev_1;

#line 88
            kt_0 = _S10;

#line 88
            continue;
        }

#line 97
        uint _S11 = 32U * head_dim_0;

#line 97
        uint elem_idx_0 = tid_0;
        for(;;)
        {

#line 98
            if(elem_idx_0 < _S11)
            {
            }
            else
            {

#line 98
                break;
            }

#line 99
            uint k_row_0 = elem_idx_0 / head_dim_0;
            uint k_col_0 = elem_idx_0 % head_dim_0;
            uint global_k_token_0 = k_base_0 + k_row_0;

            if(global_k_token_0 < tokens_0)
            {

#line 104
                uint g_off_0 = global_k_token_0 * dim_0 + h_0 * head_dim_0 + k_col_0;
                (*(&kernelContext_0)->k_tile_0)[elem_idx_0] = (&kernelContext_0)->k_buf_0[g_off_0];
                (*(&kernelContext_0)->v_tile_0)[elem_idx_0] = (&kernelContext_0)->v_buf_0[g_off_0];

#line 103
            }
            else
            {


                (*(&kernelContext_0)->k_tile_0)[elem_idx_0] = 0.0f;
                (*(&kernelContext_0)->v_tile_0)[elem_idx_0] = 0.0f;

#line 103
            }

#line 98
            elem_idx_0 = elem_idx_0 + 64U;

#line 98
        }

#line 113
        threadgroup_barrier(mem_flags::mem_threadgroup);


        if(valid_q_0)
        {

#line 117
            uint _S12 = min(32U, tokens_0 - k_base_0);

#line 117
            m_prev_1 = m_prev_0;

#line 117
            l_prev_1 = l_prev_0;

#line 117
            uint k_pos_0 = 0U;
            for(;;)
            {

#line 118
                if(k_pos_0 < _S12)
                {
                }
                else
                {

#line 118
                    break;
                }

#line 119
                uint global_k_0 = k_base_0 + k_pos_0;

#line 119
                uint d_1;


                if(_S6)
                {

#line 123
                    if(q_idx_0 >= global_k_0)
                    {

#line 123
                        d_1 = q_idx_0 - global_k_0;

#line 123
                    }
                    else
                    {

#line 123
                        d_1 = global_k_0 - q_idx_0;

#line 123
                    }
                    if(d_1 > half_window_0)
                    {

#line 125
                        k_pos_0 = k_pos_0 + 1U;

#line 118
                        continue;
                    }


                }

#line 131
                uint _S13 = k_pos_0 * head_dim_0;

#line 131
                d_1 = 0U;

#line 131
                float score_0 = 0.0f;
                for(;;)
                {

#line 132
                    if(d_1 < head_dim_0)
                    {
                    }
                    else
                    {

#line 132
                        break;
                    }

#line 133
                    float score_1 = score_0 + q_reg_0[d_1] * (*(&kernelContext_0)->k_tile_0)[_S13 + d_1];

#line 132
                    d_1 = d_1 + 1U;

#line 132
                    score_0 = score_1;

#line 132
                }


                float score_2 = score_0 * _S2;

#line 141
                float _S14 = max(m_prev_1, score_2);

#line 141
                float alpha_0;
                if(m_prev_1 > -1.00000001504746622e+30f)
                {

#line 142
                    alpha_0 = exp(m_prev_1 - _S14);

#line 142
                }
                else
                {

#line 142
                    alpha_0 = 0.0f;

#line 142
                }
                float beta_0 = exp(score_2 - _S14);

                float _S15 = l_prev_1 * alpha_0 + beta_0;

#line 145
                uint d_2 = 0U;



                for(;;)
                {

#line 149
                    if(d_2 < head_dim_0)
                    {
                    }
                    else
                    {

#line 149
                        break;
                    }

#line 150
                    acc_0[d_2] = acc_0[d_2] * alpha_0 + beta_0 * (*(&kernelContext_0)->v_tile_0)[_S13 + d_2];

#line 149
                    d_2 = d_2 + 1U;

#line 149
                }

#line 149
                m_prev_1 = _S14;

#line 149
                l_prev_1 = _S15;

#line 118
                k_pos_0 = k_pos_0 + 1U;

#line 118
            }

#line 116
        }
        else
        {

#line 116
            m_prev_1 = m_prev_0;

#line 116
            l_prev_1 = l_prev_0;

#line 116
        }

#line 155
        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 88
        uint _S10 = kt_0 + 1U;

#line 88
        m_prev_0 = m_prev_1;

#line 88
        l_prev_0 = l_prev_1;

#line 88
        kt_0 = _S10;

#line 88
    }

#line 159
    if(valid_q_0)
    {

#line 160
        if(l_prev_0 > 0.0f)
        {

#line 160
            m_prev_0 = 1.0f / l_prev_0;

#line 160
        }
        else
        {

#line 160
            m_prev_0 = 0.0f;

#line 160
        }
        uint _S16 = q_idx_0 * dim_0 + h_0 * head_dim_0;

#line 161
        d_0 = 0U;
        for(;;)
        {

#line 162
            if(d_0 < head_dim_0)
            {
            }
            else
            {

#line 162
                break;
            }

#line 163
            *((&kernelContext_0)->out_buf_0+(_S16 + d_0)) = acc_0[d_0] * m_prev_0;

#line 162
            d_0 = d_0 + 1U;

#line 162
        }

#line 159
    }

#line 166
    return;
}

