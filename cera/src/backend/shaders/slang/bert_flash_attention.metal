#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 107 "cera/src/backend/shaders/slang/bert_flash_attention.slang"
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

    uint q_idx_0 = wid_0.x * 64U + tid_0;
    bool valid_q_0 = q_idx_0 < tokens_0;


    thread array<float, int(64)> q_reg_0;

#line 52
    uint d_0;
    if(valid_q_0)
    {

#line 54
        uint _S3 = q_idx_0 * dim_0 + h_0 * head_dim_0;

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
            q_reg_0[d_0] = (&kernelContext_0)->q_buf_0[_S3 + d_0];

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

#line 75
    bool _S4 = window_size_0 > 0U;

#line 75
    bool _S5;

#line 75
    if(_S4)
    {

#line 75
        _S5 = valid_q_0;

#line 75
    }
    else
    {

#line 75
        _S5 = false;

#line 75
    }

#line 75
    uint k_start_token_0;

#line 75
    uint k_end_token_0;

#line 75
    if(_S5)
    {

#line 76
        if(q_idx_0 > half_window_0)
        {

#line 76
            d_0 = q_idx_0 - half_window_0;

#line 76
        }
        else
        {

#line 76
            d_0 = 0U;

#line 76
        }
        uint _S6 = min(tokens_0, q_idx_0 + half_window_0 + 1U);

#line 77
        k_start_token_0 = d_0;

#line 77
        k_end_token_0 = _S6;

#line 75
    }
    else
    {

#line 75
        k_start_token_0 = 0U;

#line 75
        k_end_token_0 = tokens_0;

#line 75
    }

#line 81
    uint _S7 = (tokens_0 + 32U - 1U) / 32U;

#line 81
    float m_prev_0 = -3.4028234663852886e+38f;

#line 81
    float l_prev_0 = 0.0f;

#line 81
    uint kt_0 = 0U;
    for(;;)
    {

#line 82
        if(kt_0 < _S7)
        {
        }
        else
        {

#line 82
            break;
        }

#line 83
        uint k_base_0 = kt_0 * 32U;

#line 83
        float m_prev_1;

#line 83
        float l_prev_1;


        if(_S4)
        {

#line 87
            if((k_base_0 + 32U) <= k_start_token_0)
            {

#line 87
                _S5 = true;

#line 87
            }
            else
            {

#line 87
                _S5 = k_base_0 >= k_end_token_0;

#line 87
            }

#line 87
            if(_S5)
            {

#line 87
                m_prev_1 = m_prev_0;

#line 87
                l_prev_1 = l_prev_0;

#line 82
                uint _S8 = kt_0 + 1U;

#line 82
                m_prev_0 = m_prev_1;

#line 82
                l_prev_0 = l_prev_1;

#line 82
                kt_0 = _S8;

#line 82
                continue;
            }


        }

#line 96
        uint elems_per_thread_0 = 32U * head_dim_0 / 64U;
        uint _S9 = tid_0 * elems_per_thread_0;

#line 97
        uint i_0 = 0U;

        for(;;)
        {

#line 99
            if(i_0 < elems_per_thread_0)
            {
            }
            else
            {

#line 99
                break;
            }

#line 100
            uint elem_idx_0 = _S9 + i_0;
            uint k_row_0 = elem_idx_0 / head_dim_0;
            uint k_col_0 = elem_idx_0 % head_dim_0;
            uint global_k_token_0 = k_base_0 + k_row_0;

            if(global_k_token_0 < tokens_0)
            {

#line 106
                uint g_off_0 = global_k_token_0 * dim_0 + h_0 * head_dim_0 + k_col_0;
                (*(&kernelContext_0)->k_tile_0)[elem_idx_0] = (&kernelContext_0)->k_buf_0[g_off_0];
                (*(&kernelContext_0)->v_tile_0)[elem_idx_0] = (&kernelContext_0)->v_buf_0[g_off_0];

#line 105
            }
            else
            {


                (*(&kernelContext_0)->k_tile_0)[elem_idx_0] = 0.0f;
                (*(&kernelContext_0)->v_tile_0)[elem_idx_0] = 0.0f;

#line 105
            }

#line 99
            i_0 = i_0 + 1U;

#line 99
        }

#line 115
        threadgroup_barrier(mem_flags::mem_threadgroup);


        if(valid_q_0)
        {

#line 119
            uint _S10 = min(32U, tokens_0 - k_base_0);

#line 119
            m_prev_1 = m_prev_0;

#line 119
            l_prev_1 = l_prev_0;

#line 119
            uint k_pos_0 = 0U;
            for(;;)
            {

#line 120
                if(k_pos_0 < _S10)
                {
                }
                else
                {

#line 120
                    break;
                }

#line 121
                uint global_k_0 = k_base_0 + k_pos_0;

#line 121
                uint d_1;


                if(_S4)
                {

#line 125
                    if(q_idx_0 >= global_k_0)
                    {

#line 125
                        d_1 = q_idx_0 - global_k_0;

#line 125
                    }
                    else
                    {

#line 125
                        d_1 = global_k_0 - q_idx_0;

#line 125
                    }
                    if(d_1 > half_window_0)
                    {

#line 127
                        k_pos_0 = k_pos_0 + 1U;

#line 120
                        continue;
                    }


                }

#line 133
                uint _S11 = k_pos_0 * head_dim_0;

#line 133
                d_0 = 0U;

#line 133
                float score_0 = 0.0f;
                for(;;)
                {

#line 134
                    if(d_0 < head_dim_0)
                    {
                    }
                    else
                    {

#line 134
                        break;
                    }

#line 135
                    float score_1 = score_0 + q_reg_0[d_0] * (*(&kernelContext_0)->k_tile_0)[_S11 + d_0];

#line 134
                    d_0 = d_0 + 1U;

#line 134
                    score_0 = score_1;

#line 134
                }


                float score_2 = score_0 * _S2;

#line 143
                float _S12 = max(m_prev_1, score_2);
                float alpha_0 = exp(m_prev_1 - _S12);
                float beta_0 = exp(score_2 - _S12);

                float _S13 = l_prev_1 * alpha_0 + beta_0;

#line 147
                d_1 = 0U;



                for(;;)
                {

#line 151
                    if(d_1 < head_dim_0)
                    {
                    }
                    else
                    {

#line 151
                        break;
                    }

#line 152
                    acc_0[d_1] = acc_0[d_1] * alpha_0 + beta_0 * (*(&kernelContext_0)->v_tile_0)[_S11 + d_1];

#line 151
                    d_1 = d_1 + 1U;

#line 151
                }

#line 151
                m_prev_1 = _S12;

#line 151
                l_prev_1 = _S13;

#line 120
                k_pos_0 = k_pos_0 + 1U;

#line 120
            }

#line 118
        }
        else
        {

#line 118
            m_prev_1 = m_prev_0;

#line 118
            l_prev_1 = l_prev_0;

#line 118
        }

#line 157
        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 82
        uint _S8 = kt_0 + 1U;

#line 82
        m_prev_0 = m_prev_1;

#line 82
        l_prev_0 = l_prev_1;

#line 82
        kt_0 = _S8;

#line 82
    }

#line 161
    if(valid_q_0)
    {

#line 162
        if(l_prev_0 > 0.0f)
        {

#line 162
            m_prev_0 = 1.0f / l_prev_0;

#line 162
        }
        else
        {

#line 162
            m_prev_0 = 0.0f;

#line 162
        }
        uint _S14 = q_idx_0 * dim_0 + h_0 * head_dim_0;

#line 163
        d_0 = 0U;
        for(;;)
        {

#line 164
            if(d_0 < head_dim_0)
            {
            }
            else
            {

#line 164
                break;
            }

#line 165
            *((&kernelContext_0)->out_buf_0+(_S14 + d_0)) = acc_0[d_0] * m_prev_0;

#line 164
            d_0 = d_0 + 1U;

#line 164
        }

#line 161
    }

#line 168
    return;
}

