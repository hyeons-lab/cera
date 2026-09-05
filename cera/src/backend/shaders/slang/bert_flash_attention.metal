#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 102 "cera/src/backend/shaders/slang/bert_flash_attention.slang"
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


#line 30
[[kernel]] void bert_flash_attention(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 wid_0 [[threadgroup_position_in_grid]], packed_uint4 device* par_buf_1 [[buffer(4)]], float device* q_buf_1 [[buffer(0)]], float device* k_buf_1 [[buffer(1)]], float device* v_buf_1 [[buffer(2)]], float device* out_buf_1 [[buffer(3)]])
{
    thread KernelContext_0 kernelContext_0;

#line 32
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 32
    (&kernelContext_0)->q_buf_0 = q_buf_1;

#line 32
    (&kernelContext_0)->k_buf_0 = k_buf_1;

#line 32
    (&kernelContext_0)->v_buf_0 = v_buf_1;

#line 32
    (&kernelContext_0)->out_buf_0 = out_buf_1;

#line 32
    threadgroup array<float, int(2048)> k_tile_1;

#line 32
    (&kernelContext_0)->k_tile_0 = &k_tile_1;

#line 32
    threadgroup array<float, int(2048)> v_tile_1;

#line 32
    (&kernelContext_0)->v_tile_0 = &v_tile_1;

    uint tid_0 = lid_0.x;

#line 34
    uint4 _S1 = uint4(*(par_buf_1+int(0))) ;



    uint tokens_0 = _S1.x;

    uint head_dim_0 = _S1.z;
    float _S2 = (as_type<float>((_S1.w)));
    uint window_size_0 = (uint4(*(par_buf_1+int(1))) ).x;
    uint half_window_0 = window_size_0 / 2U;

    uint h_0 = wid_0.y;
    uint dim_0 = _S1.y * head_dim_0;

    uint _S3 = wid_0.x * 64U;

#line 48
    uint q_idx_0 = _S3 + tid_0;
    bool valid_q_0 = q_idx_0 < tokens_0;


    if(head_dim_0 > 64U)
    {

#line 53
        return;
    }


    thread array<float, int(64)> q_reg_0;

#line 57
    uint d_0;
    if(valid_q_0)
    {

#line 59
        uint _S4 = q_idx_0 * dim_0 + h_0 * head_dim_0;

#line 59
        d_0 = 0U;
        for(;;)
        {

#line 60
            if(d_0 < head_dim_0)
            {
            }
            else
            {

#line 60
                break;
            }

#line 61
            q_reg_0[d_0] = (&kernelContext_0)->q_buf_0[_S4 + d_0];

#line 60
            d_0 = d_0 + 1U;

#line 60
        }

#line 58
    }
    else
    {

#line 58
        d_0 = 0U;

#line 64
        for(;;)
        {

#line 64
            if(d_0 < head_dim_0)
            {
            }
            else
            {

#line 64
                break;
            }

#line 65
            q_reg_0[d_0] = 0.0f;

#line 64
            d_0 = d_0 + 1U;

#line 64
        }

#line 58
    }

#line 72
    thread array<float, int(64)> acc_0;

#line 72
    d_0 = 0U;
    for(;;)
    {

#line 73
        if(d_0 < head_dim_0)
        {
        }
        else
        {

#line 73
            break;
        }

#line 74
        acc_0[d_0] = 0.0f;

#line 73
        d_0 = d_0 + 1U;

#line 73
    }

#line 79
    uint _S5 = min(tokens_0, _S3 + 64U);
    bool _S6 = window_size_0 > 0U;

#line 80
    bool _S7;

#line 80
    if(_S6)
    {

#line 80
        _S7 = _S3 > half_window_0;

#line 80
    }
    else
    {

#line 80
        _S7 = false;

#line 80
    }

#line 80
    if(_S7)
    {

#line 80
        d_0 = _S3 - half_window_0;

#line 80
    }
    else
    {

#line 80
        d_0 = 0U;

#line 80
    }

#line 80
    uint _S8;
    if(_S6)
    {

#line 81
        _S8 = min(tokens_0, _S5 + half_window_0);

#line 81
    }
    else
    {

#line 81
        _S8 = tokens_0;

#line 81
    }


    uint _S9 = (tokens_0 + 32U - 1U) / 32U;

#line 84
    float m_prev_0 = -3.4028234663852886e+38f;

#line 84
    float l_prev_0 = 0.0f;

#line 84
    uint kt_0 = 0U;
    for(;;)
    {

#line 85
        if(kt_0 < _S9)
        {
        }
        else
        {

#line 85
            break;
        }

#line 86
        uint k_base_0 = kt_0 * 32U;


        if(_S6)
        {

#line 89
            if((k_base_0 + 32U) <= d_0)
            {

#line 89
                _S7 = true;

#line 89
            }
            else
            {

#line 89
                _S7 = k_base_0 >= _S8;

#line 89
            }

#line 89
        }
        else
        {

#line 89
            _S7 = false;

#line 89
        }

#line 89
        float m_prev_1;

#line 89
        float l_prev_1;

#line 89
        if(_S7)
        {

#line 89
            m_prev_1 = m_prev_0;

#line 89
            l_prev_1 = l_prev_0;

#line 85
            uint _S10 = kt_0 + 1U;

#line 85
            m_prev_0 = m_prev_1;

#line 85
            l_prev_0 = l_prev_1;

#line 85
            kt_0 = _S10;

#line 85
            continue;
        }

#line 94
        uint _S11 = 32U * head_dim_0;

#line 94
        uint elem_idx_0 = tid_0;
        for(;;)
        {

#line 95
            if(elem_idx_0 < _S11)
            {
            }
            else
            {

#line 95
                break;
            }

#line 96
            uint k_row_0 = elem_idx_0 / head_dim_0;
            uint k_col_0 = elem_idx_0 % head_dim_0;
            uint global_k_token_0 = k_base_0 + k_row_0;

            if(global_k_token_0 < tokens_0)
            {

#line 101
                uint g_off_0 = global_k_token_0 * dim_0 + h_0 * head_dim_0 + k_col_0;
                (*(&kernelContext_0)->k_tile_0)[elem_idx_0] = (&kernelContext_0)->k_buf_0[g_off_0];
                (*(&kernelContext_0)->v_tile_0)[elem_idx_0] = (&kernelContext_0)->v_buf_0[g_off_0];

#line 100
            }
            else
            {


                (*(&kernelContext_0)->k_tile_0)[elem_idx_0] = 0.0f;
                (*(&kernelContext_0)->v_tile_0)[elem_idx_0] = 0.0f;

#line 100
            }

#line 95
            elem_idx_0 = elem_idx_0 + 64U;

#line 95
        }

#line 110
        threadgroup_barrier(mem_flags::mem_threadgroup);


        if(valid_q_0)
        {

#line 114
            uint _S12 = min(32U, tokens_0 - k_base_0);

#line 114
            m_prev_1 = m_prev_0;

#line 114
            l_prev_1 = l_prev_0;

#line 114
            uint k_pos_0 = 0U;
            for(;;)
            {

#line 115
                if(k_pos_0 < _S12)
                {
                }
                else
                {

#line 115
                    break;
                }

#line 116
                uint global_k_0 = k_base_0 + k_pos_0;

#line 116
                uint d_1;


                if(_S6)
                {

#line 120
                    if(q_idx_0 >= global_k_0)
                    {

#line 120
                        d_1 = q_idx_0 - global_k_0;

#line 120
                    }
                    else
                    {

#line 120
                        d_1 = global_k_0 - q_idx_0;

#line 120
                    }
                    if(d_1 > half_window_0)
                    {

#line 122
                        k_pos_0 = k_pos_0 + 1U;

#line 115
                        continue;
                    }


                }

#line 128
                uint _S13 = k_pos_0 * head_dim_0;

#line 128
                d_1 = 0U;

#line 128
                float score_0 = 0.0f;
                for(;;)
                {

#line 129
                    if(d_1 < head_dim_0)
                    {
                    }
                    else
                    {

#line 129
                        break;
                    }

#line 130
                    float score_1 = score_0 + q_reg_0[d_1] * (*(&kernelContext_0)->k_tile_0)[_S13 + d_1];

#line 129
                    d_1 = d_1 + 1U;

#line 129
                    score_0 = score_1;

#line 129
                }


                float score_2 = score_0 * _S2;

#line 138
                float _S14 = max(m_prev_1, score_2);

#line 138
                float alpha_0;
                if(m_prev_1 > -1.00000001504746622e+30f)
                {

#line 139
                    alpha_0 = exp(m_prev_1 - _S14);

#line 139
                }
                else
                {

#line 139
                    alpha_0 = 0.0f;

#line 139
                }
                float beta_0 = exp(score_2 - _S14);

                float _S15 = l_prev_1 * alpha_0 + beta_0;

#line 142
                uint d_2 = 0U;



                for(;;)
                {

#line 146
                    if(d_2 < head_dim_0)
                    {
                    }
                    else
                    {

#line 146
                        break;
                    }

#line 147
                    acc_0[d_2] = acc_0[d_2] * alpha_0 + beta_0 * (*(&kernelContext_0)->v_tile_0)[_S13 + d_2];

#line 146
                    d_2 = d_2 + 1U;

#line 146
                }

#line 146
                m_prev_1 = _S14;

#line 146
                l_prev_1 = _S15;

#line 115
                k_pos_0 = k_pos_0 + 1U;

#line 115
            }

#line 113
        }
        else
        {

#line 113
            m_prev_1 = m_prev_0;

#line 113
            l_prev_1 = l_prev_0;

#line 113
        }

#line 152
        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 85
        uint _S10 = kt_0 + 1U;

#line 85
        m_prev_0 = m_prev_1;

#line 85
        l_prev_0 = l_prev_1;

#line 85
        kt_0 = _S10;

#line 85
    }

#line 156
    if(valid_q_0)
    {

#line 157
        if(l_prev_0 > 0.0f)
        {

#line 157
            m_prev_0 = 1.0f / l_prev_0;

#line 157
        }
        else
        {

#line 157
            m_prev_0 = 0.0f;

#line 157
        }
        uint _S16 = q_idx_0 * dim_0 + h_0 * head_dim_0;

#line 158
        d_0 = 0U;
        for(;;)
        {

#line 159
            if(d_0 < head_dim_0)
            {
            }
            else
            {

#line 159
                break;
            }

#line 160
            *((&kernelContext_0)->out_buf_0+(_S16 + d_0)) = acc_0[d_0] * m_prev_0;

#line 159
            d_0 = d_0 + 1U;

#line 159
        }

#line 156
    }

#line 163
    return;
}

