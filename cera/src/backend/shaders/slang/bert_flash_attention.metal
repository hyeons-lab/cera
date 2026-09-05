#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 103 "cera/src/backend/shaders/slang/bert_flash_attention.slang"
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


    thread array<float, int(64)> q_reg_0;

#line 52
    uint d_0;
    if(valid_q_0)
    {

#line 54
        uint _S4 = q_idx_0 * dim_0 + h_0 * head_dim_0;

#line 54
        d_0 = 0U;
        for(;;)
        {

#line 55
            if(d_0 < head_dim_0)
            {
            }
            else
            {

#line 55
                break;
            }

#line 56
            q_reg_0[d_0] = (&kernelContext_0)->q_buf_0[_S4 + d_0];

#line 55
            d_0 = d_0 + 1U;

#line 55
        }

#line 53
    }
    else
    {

#line 53
        d_0 = 0U;

#line 59
        for(;;)
        {

#line 59
            if(d_0 < head_dim_0)
            {
            }
            else
            {

#line 59
                break;
            }

#line 60
            q_reg_0[d_0] = 0.0f;

#line 59
            d_0 = d_0 + 1U;

#line 59
        }

#line 53
    }

#line 67
    thread array<float, int(64)> acc_0;

#line 67
    d_0 = 0U;
    for(;;)
    {

#line 68
        if(d_0 < head_dim_0)
        {
        }
        else
        {

#line 68
            break;
        }

#line 69
        acc_0[d_0] = 0.0f;

#line 68
        d_0 = d_0 + 1U;

#line 68
    }

#line 74
    uint _S5 = min(tokens_0, _S3 + 64U);
    bool _S6 = window_size_0 > 0U;

#line 75
    bool _S7;

#line 75
    if(_S6)
    {

#line 75
        _S7 = _S3 > half_window_0;

#line 75
    }
    else
    {

#line 75
        _S7 = false;

#line 75
    }

#line 75
    if(_S7)
    {

#line 75
        d_0 = _S3 - half_window_0;

#line 75
    }
    else
    {

#line 75
        d_0 = 0U;

#line 75
    }

#line 75
    uint _S8;
    if(_S6)
    {

#line 76
        _S8 = min(tokens_0, _S5 + half_window_0);

#line 76
    }
    else
    {

#line 76
        _S8 = tokens_0;

#line 76
    }


    uint _S9 = (tokens_0 + 32U - 1U) / 32U;

#line 79
    float m_prev_0 = -3.4028234663852886e+38f;

#line 79
    float l_prev_0 = 0.0f;

#line 79
    uint kt_0 = 0U;
    for(;;)
    {

#line 80
        if(kt_0 < _S9)
        {
        }
        else
        {

#line 80
            break;
        }

#line 81
        uint k_base_0 = kt_0 * 32U;


        if(_S6)
        {

#line 84
            if((k_base_0 + 32U) <= d_0)
            {

#line 84
                _S7 = true;

#line 84
            }
            else
            {

#line 84
                _S7 = k_base_0 >= _S8;

#line 84
            }

#line 84
        }
        else
        {

#line 84
            _S7 = false;

#line 84
        }

#line 84
        float m_prev_1;

#line 84
        float l_prev_1;

#line 84
        if(_S7)
        {

#line 84
            m_prev_1 = m_prev_0;

#line 84
            l_prev_1 = l_prev_0;

#line 80
            uint _S10 = kt_0 + 1U;

#line 80
            m_prev_0 = m_prev_1;

#line 80
            l_prev_0 = l_prev_1;

#line 80
            kt_0 = _S10;

#line 80
            continue;
        }

#line 92
        uint elems_per_thread_0 = 32U * head_dim_0 / 64U;
        uint _S11 = tid_0 * elems_per_thread_0;

#line 93
        uint i_0 = 0U;

        for(;;)
        {

#line 95
            if(i_0 < elems_per_thread_0)
            {
            }
            else
            {

#line 95
                break;
            }

#line 96
            uint elem_idx_0 = _S11 + i_0;
            uint k_row_0 = elem_idx_0 / head_dim_0;
            uint k_col_0 = elem_idx_0 % head_dim_0;
            uint global_k_token_0 = k_base_0 + k_row_0;

            if(global_k_token_0 < tokens_0)
            {

#line 102
                uint g_off_0 = global_k_token_0 * dim_0 + h_0 * head_dim_0 + k_col_0;
                (*(&kernelContext_0)->k_tile_0)[elem_idx_0] = (&kernelContext_0)->k_buf_0[g_off_0];
                (*(&kernelContext_0)->v_tile_0)[elem_idx_0] = (&kernelContext_0)->v_buf_0[g_off_0];

#line 101
            }
            else
            {


                (*(&kernelContext_0)->k_tile_0)[elem_idx_0] = 0.0f;
                (*(&kernelContext_0)->v_tile_0)[elem_idx_0] = 0.0f;

#line 101
            }

#line 95
            i_0 = i_0 + 1U;

#line 95
        }

#line 111
        threadgroup_barrier(mem_flags::mem_threadgroup);


        if(valid_q_0)
        {

#line 115
            uint _S12 = min(32U, tokens_0 - k_base_0);

#line 115
            m_prev_1 = m_prev_0;

#line 115
            l_prev_1 = l_prev_0;

#line 115
            uint k_pos_0 = 0U;
            for(;;)
            {

#line 116
                if(k_pos_0 < _S12)
                {
                }
                else
                {

#line 116
                    break;
                }

#line 117
                uint global_k_0 = k_base_0 + k_pos_0;

#line 117
                uint d_1;


                if(_S6)
                {

#line 121
                    if(q_idx_0 >= global_k_0)
                    {

#line 121
                        d_1 = q_idx_0 - global_k_0;

#line 121
                    }
                    else
                    {

#line 121
                        d_1 = global_k_0 - q_idx_0;

#line 121
                    }
                    if(d_1 > half_window_0)
                    {

#line 123
                        k_pos_0 = k_pos_0 + 1U;

#line 116
                        continue;
                    }


                }

#line 129
                uint _S13 = k_pos_0 * head_dim_0;

#line 129
                d_1 = 0U;

#line 129
                float score_0 = 0.0f;
                for(;;)
                {

#line 130
                    if(d_1 < head_dim_0)
                    {
                    }
                    else
                    {

#line 130
                        break;
                    }

#line 131
                    float score_1 = score_0 + q_reg_0[d_1] * (*(&kernelContext_0)->k_tile_0)[_S13 + d_1];

#line 130
                    d_1 = d_1 + 1U;

#line 130
                    score_0 = score_1;

#line 130
                }


                float score_2 = score_0 * _S2;

#line 139
                float _S14 = max(m_prev_1, score_2);
                float alpha_0 = exp(m_prev_1 - _S14);
                float beta_0 = exp(score_2 - _S14);

                float _S15 = l_prev_1 * alpha_0 + beta_0;

#line 143
                uint d_2 = 0U;



                for(;;)
                {

#line 147
                    if(d_2 < head_dim_0)
                    {
                    }
                    else
                    {

#line 147
                        break;
                    }

#line 148
                    acc_0[d_2] = acc_0[d_2] * alpha_0 + beta_0 * (*(&kernelContext_0)->v_tile_0)[_S13 + d_2];

#line 147
                    d_2 = d_2 + 1U;

#line 147
                }

#line 147
                m_prev_1 = _S14;

#line 147
                l_prev_1 = _S15;

#line 116
                k_pos_0 = k_pos_0 + 1U;

#line 116
            }

#line 114
        }
        else
        {

#line 114
            m_prev_1 = m_prev_0;

#line 114
            l_prev_1 = l_prev_0;

#line 114
        }

#line 153
        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 80
        uint _S10 = kt_0 + 1U;

#line 80
        m_prev_0 = m_prev_1;

#line 80
        l_prev_0 = l_prev_1;

#line 80
        kt_0 = _S10;

#line 80
    }

#line 157
    if(valid_q_0)
    {

#line 158
        if(l_prev_0 > 0.0f)
        {

#line 158
            m_prev_0 = 1.0f / l_prev_0;

#line 158
        }
        else
        {

#line 158
            m_prev_0 = 0.0f;

#line 158
        }
        uint _S16 = q_idx_0 * dim_0 + h_0 * head_dim_0;

#line 159
        d_0 = 0U;
        for(;;)
        {

#line 160
            if(d_0 < head_dim_0)
            {
            }
            else
            {

#line 160
                break;
            }

#line 161
            *((&kernelContext_0)->out_buf_0+(_S16 + d_0)) = acc_0[d_0] * m_prev_0;

#line 160
            d_0 = d_0 + 1U;

#line 160
        }

#line 157
    }

#line 164
    return;
}

