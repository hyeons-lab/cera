#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 27 "cera/src/backend/shaders/slang/gemv_q4_0_qkv.slang"
uint get_wid_0(uint3 wid_0)
{

#line 28
    return wid_0.x + wid_0.y * 65535U;
}


#line 286
struct KernelContext_0
{
    packed_uint4 device* params_0;
    float device* x_0;
    uint device* w_q_0;
    uint device* w_k_0;
    uint device* w_v_0;
    float device* y_q_0;
    float device* y_k_0;
    float device* y_v_0;
    array<float, int(512)> threadgroup* x_stage_0;
    array<float, int(128)> threadgroup* partials_0;
};


#line 31
float block_scale_q_0(uint blk_byte_0, KernelContext_0 thread* kernelContext_0)
{

#line 32
    uint word_0 = kernelContext_0->w_q_0[blk_byte_0 >> 2U];

#line 32
    uint scale_bits_0;
    if((blk_byte_0 & 2U) != 0U)
    {

#line 33
        scale_bits_0 = word_0 >> 16U;

#line 33
    }
    else
    {

#line 33
        scale_bits_0 = word_0 & 65535U;

#line 33
    }
    return (as_type<half>((ushort)((scale_bits_0))));
}


#line 49
uint q_pair_q_0(uint qs_byte_0, KernelContext_0 thread* kernelContext_1)
{

#line 50
    uint word_1 = kernelContext_1->w_q_0[qs_byte_0 >> 2U];

#line 50
    uint _S1;
    if((qs_byte_0 & 2U) != 0U)
    {

#line 51
        _S1 = word_1 >> 16U;

#line 51
    }
    else
    {

#line 51
        _S1 = word_1 & 65535U;

#line 51
    }

#line 51
    return _S1;
}


#line 37
float block_scale_k_0(uint blk_byte_1, KernelContext_0 thread* kernelContext_2)
{

#line 38
    uint word_2 = kernelContext_2->w_k_0[blk_byte_1 >> 2U];

#line 38
    uint scale_bits_1;
    if((blk_byte_1 & 2U) != 0U)
    {

#line 39
        scale_bits_1 = word_2 >> 16U;

#line 39
    }
    else
    {

#line 39
        scale_bits_1 = word_2 & 65535U;

#line 39
    }
    return (as_type<half>((ushort)((scale_bits_1))));
}


#line 54
uint q_pair_k_0(uint qs_byte_1, KernelContext_0 thread* kernelContext_3)
{

#line 55
    uint word_3 = kernelContext_3->w_k_0[qs_byte_1 >> 2U];

#line 55
    uint _S2;
    if((qs_byte_1 & 2U) != 0U)
    {

#line 56
        _S2 = word_3 >> 16U;

#line 56
    }
    else
    {

#line 56
        _S2 = word_3 & 65535U;

#line 56
    }

#line 56
    return _S2;
}


#line 43
float block_scale_v_0(uint blk_byte_2, KernelContext_0 thread* kernelContext_4)
{

#line 44
    uint word_4 = kernelContext_4->w_v_0[blk_byte_2 >> 2U];

#line 44
    uint scale_bits_2;
    if((blk_byte_2 & 2U) != 0U)
    {

#line 45
        scale_bits_2 = word_4 >> 16U;

#line 45
    }
    else
    {

#line 45
        scale_bits_2 = word_4 & 65535U;

#line 45
    }
    return (as_type<half>((ushort)((scale_bits_2))));
}


#line 59
uint q_pair_v_0(uint qs_byte_2, KernelContext_0 thread* kernelContext_5)
{

#line 60
    uint word_5 = kernelContext_5->w_v_0[qs_byte_2 >> 2U];

#line 60
    uint _S3;
    if((qs_byte_2 & 2U) != 0U)
    {

#line 61
        _S3 = word_5 >> 16U;

#line 61
    }
    else
    {

#line 61
        _S3 = word_5 & 65535U;

#line 61
    }

#line 61
    return _S3;
}



[[kernel]] void gemv_q4_0_qkv(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 wid_1 [[threadgroup_position_in_grid]], packed_uint4 device* params_1 [[buffer(7)]], float device* x_1 [[buffer(3)]], uint device* w_q_1 [[buffer(0)]], uint device* w_k_1 [[buffer(1)]], uint device* w_v_1 [[buffer(2)]], float device* y_q_1 [[buffer(4)]], float device* y_k_1 [[buffer(5)]], float device* y_v_1 [[buffer(6)]])
{
    uint i_0;

#line 68
    thread KernelContext_0 kernelContext_6;

#line 68
    (&kernelContext_6)->params_0 = params_1;

#line 68
    (&kernelContext_6)->x_0 = x_1;

#line 68
    (&kernelContext_6)->w_q_0 = w_q_1;

#line 68
    (&kernelContext_6)->w_k_0 = w_k_1;

#line 68
    (&kernelContext_6)->w_v_0 = w_v_1;

#line 68
    (&kernelContext_6)->y_q_0 = y_q_1;

#line 68
    (&kernelContext_6)->y_k_0 = y_k_1;

#line 68
    (&kernelContext_6)->y_v_0 = y_v_1;

#line 68
    threadgroup array<float, int(512)> x_stage_1;

#line 68
    (&kernelContext_6)->x_stage_0 = &x_stage_1;

#line 68
    threadgroup array<float, int(128)> partials_1;

#line 68
    (&kernelContext_6)->partials_0 = &partials_1;

    uint m_q_0 = (uint4(*(params_1+int(0))) ).x;
    uint m_kv_0 = (uint4(*(params_1+int(0))) ).y;
    uint k_0 = (uint4(*(params_1+int(0))) ).z;
    uint nb_0 = k_0 / 32U;
    uint _S4 = nb_0 * 18U;
    uint global_r0_0 = get_wid_0(wid_1) * 4U;
    uint tid_0 = lid_0.x;

    uint _S5 = tid_0 / 2U;
    uint _S6 = (tid_0 & 1U) * 8U;

#line 79
    uint which_matrix_0;

#line 79
    uint r0_0;

#line 79
    uint m_cur_0;

#line 84
    if(global_r0_0 < m_q_0)
    {

#line 84
        which_matrix_0 = 0U;

#line 84
        r0_0 = global_r0_0;

#line 84
        m_cur_0 = m_q_0;

#line 84
    }
    else
    {

        if(global_r0_0 < (m_q_0 + m_kv_0))
        {
            uint _S7 = global_r0_0 - m_q_0;

#line 90
            which_matrix_0 = 1U;

#line 90
            r0_0 = _S7;

#line 88
        }
        else
        {



            uint _S8 = global_r0_0 - m_q_0 - m_kv_0;

#line 94
            which_matrix_0 = 2U;

#line 94
            r0_0 = _S8;

#line 88
        }

#line 88
        m_cur_0 = m_kv_0;

#line 84
    }

#line 98
    thread array<float, int(4)> sumf_0;

#line 98
    uint r_0 = 0U;

    for(;;)
    {

#line 100
        if(r_0 < 4U)
        {
        }
        else
        {

#line 100
            break;
        }

#line 101
        sumf_0[r_0] = 0.0f;

#line 100
        r_0 = r_0 + 1U;

#line 100
    }

#line 100
    uint chunk_b_0 = 0U;



    for(;;)
    {

#line 104
        if(chunk_b_0 < nb_0)
        {
        }
        else
        {

#line 104
            break;
        }

#line 105
        uint chunk_k_start_0 = chunk_b_0 * 32U;

        uint _S9 = min(512U, k_0 - chunk_k_start_0) / 4U;

#line 107
        i_0 = tid_0;
        for(;;)
        {

#line 108
            if(i_0 < _S9)
            {
            }
            else
            {

#line 108
                break;
            }

#line 109
            uint _S10 = i_0 * 4U;

#line 109
            uint base_idx_0 = chunk_k_start_0 + _S10;
            (*(&kernelContext_6)->x_stage_0)[_S10] = (&kernelContext_6)->x_0[base_idx_0];
            (*(&kernelContext_6)->x_stage_0)[_S10 + 1U] = (&kernelContext_6)->x_0[base_idx_0 + 1U];
            (*(&kernelContext_6)->x_stage_0)[_S10 + 2U] = (&kernelContext_6)->x_0[base_idx_0 + 2U];
            (*(&kernelContext_6)->x_stage_0)[_S10 + 3U] = (&kernelContext_6)->x_0[base_idx_0 + 3U];

#line 108
            i_0 = i_0 + 32U;

#line 108
        }

#line 115
        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 115
        uint ib_local_0 = _S5;


        for(;;)
        {

#line 118
            bool _S11;

#line 118
            if(ib_local_0 < 16U)
            {

#line 118
                _S11 = (chunk_b_0 + ib_local_0) < nb_0;

#line 118
            }
            else
            {

#line 118
                _S11 = false;

#line 118
            }

#line 118
            if(_S11)
            {
            }
            else
            {

#line 118
                break;
            }

#line 119
            uint _S12 = chunk_b_0 + ib_local_0;
            uint yb_stage_off_0 = ib_local_0 * 32U + _S6;

            float a0_0 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0];

            float a2_0 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 2U];

            float a4_0 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 4U];

            float a6_0 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 6U];

#line 140
            float _S13 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 1U] / 256.0f;

            float _S14 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 3U] / 256.0f;

            float _S15 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 5U] / 256.0f;

            float _S16 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 7U] / 256.0f;
            float _S17 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 16U] / 16.0f;
            float _S18 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 17U] / 4096.0f;
            float _S19 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 18U] / 16.0f;
            float _S20 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 19U] / 4096.0f;
            float _S21 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 20U] / 16.0f;
            float _S22 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 21U] / 4096.0f;
            float _S23 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 22U] / 16.0f;
            float _S24 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 23U] / 4096.0f;



            float _S25 = (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0] + (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 1U] + ((*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 2U] + (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 3U]) + ((*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 4U] + (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 5U]) + ((*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 6U] + (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 7U]) + ((*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 16U] + (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 17U] + ((*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 18U] + (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 19U]) + ((*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 20U] + (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 21U]) + ((*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 22U] + (*(&kernelContext_6)->x_stage_0)[yb_stage_off_0 + 23U]));

            if(which_matrix_0 == 0U)
            {

#line 160
                r_0 = 0U;

                for(;;)
                {

#line 162
                    if(r_0 < 4U)
                    {
                    }
                    else
                    {

#line 162
                        break;
                    }

#line 163
                    uint _S26 = r0_0 + r_0;

#line 163
                    if(_S26 >= m_cur_0)
                    {

#line 163
                        r_0 = r_0 + 1U;

#line 162
                        continue;
                    }
                    uint blk_byte_3 = _S26 * _S4 + _S12 * 18U;

#line 164
                    float _S27 = block_scale_q_0(blk_byte_3, &kernelContext_6);

                    uint qs_byte_3 = blk_byte_3 + 2U + _S6;

#line 166
                    uint _S28 = q_pair_q_0(qs_byte_3, &kernelContext_6);

#line 174
                    float _S29 = a0_0 * float(_S28 & 15U);
                    float _S30 = _S13 * float(_S28 & 3840U);
                    float _S31 = _S17 * float(_S28 & 240U);
                    float _S32 = _S18 * float(_S28 & 61440U);

#line 177
                    uint _S33 = q_pair_q_0(qs_byte_3 + 2U, &kernelContext_6);


                    float acc0_0 = _S29 + a2_0 * float(_S33 & 15U);
                    float acc1_0 = _S30 + _S14 * float(_S33 & 3840U);
                    float acc2_0 = _S31 + _S19 * float(_S33 & 240U);
                    float acc3_0 = _S32 + _S20 * float(_S33 & 61440U);

#line 183
                    uint _S34 = q_pair_q_0(qs_byte_3 + 4U, &kernelContext_6);


                    float acc0_1 = acc0_0 + a4_0 * float(_S34 & 15U);
                    float acc1_1 = acc1_0 + _S15 * float(_S34 & 3840U);
                    float acc2_1 = acc2_0 + _S21 * float(_S34 & 240U);
                    float acc3_1 = acc3_0 + _S22 * float(_S34 & 61440U);

#line 189
                    uint _S35 = q_pair_q_0(qs_byte_3 + 6U, &kernelContext_6);

#line 197
                    sumf_0[r_0] = sumf_0[r_0] + _S27 * (_S25 * -8.0f + (acc0_1 + a6_0 * float(_S35 & 15U)) + (acc1_1 + _S16 * float(_S35 & 3840U)) + (acc2_1 + _S23 * float(_S35 & 240U)) + (acc3_1 + _S24 * float(_S35 & 61440U)));

#line 162
                    r_0 = r_0 + 1U;

#line 162
                }

#line 160
            }
            else
            {

#line 199
                if(which_matrix_0 == 1U)
                {

#line 199
                    r_0 = 0U;

                    for(;;)
                    {

#line 201
                        if(r_0 < 4U)
                        {
                        }
                        else
                        {

#line 201
                            break;
                        }

#line 202
                        uint _S36 = r0_0 + r_0;

#line 202
                        if(_S36 >= m_cur_0)
                        {

#line 202
                            r_0 = r_0 + 1U;

#line 201
                            continue;
                        }
                        uint blk_byte_4 = _S36 * _S4 + _S12 * 18U;

#line 203
                        float _S37 = block_scale_k_0(blk_byte_4, &kernelContext_6);

                        uint qs_byte_4 = blk_byte_4 + 2U + _S6;

#line 205
                        uint _S38 = q_pair_k_0(qs_byte_4, &kernelContext_6);

#line 213
                        float _S39 = a0_0 * float(_S38 & 15U);
                        float _S40 = _S13 * float(_S38 & 3840U);
                        float _S41 = _S17 * float(_S38 & 240U);
                        float _S42 = _S18 * float(_S38 & 61440U);

#line 216
                        uint _S43 = q_pair_k_0(qs_byte_4 + 2U, &kernelContext_6);


                        float acc0_2 = _S39 + a2_0 * float(_S43 & 15U);
                        float acc1_2 = _S40 + _S14 * float(_S43 & 3840U);
                        float acc2_2 = _S41 + _S19 * float(_S43 & 240U);
                        float acc3_2 = _S42 + _S20 * float(_S43 & 61440U);

#line 222
                        uint _S44 = q_pair_k_0(qs_byte_4 + 4U, &kernelContext_6);


                        float acc0_3 = acc0_2 + a4_0 * float(_S44 & 15U);
                        float acc1_3 = acc1_2 + _S15 * float(_S44 & 3840U);
                        float acc2_3 = acc2_2 + _S21 * float(_S44 & 240U);
                        float acc3_3 = acc3_2 + _S22 * float(_S44 & 61440U);

#line 228
                        uint _S45 = q_pair_k_0(qs_byte_4 + 6U, &kernelContext_6);

#line 236
                        sumf_0[r_0] = sumf_0[r_0] + _S37 * (_S25 * -8.0f + (acc0_3 + a6_0 * float(_S45 & 15U)) + (acc1_3 + _S16 * float(_S45 & 3840U)) + (acc2_3 + _S23 * float(_S45 & 240U)) + (acc3_3 + _S24 * float(_S45 & 61440U)));

#line 201
                        r_0 = r_0 + 1U;

#line 201
                    }

#line 199
                }
                else
                {

#line 199
                    r_0 = 0U;

#line 240
                    for(;;)
                    {

#line 240
                        if(r_0 < 4U)
                        {
                        }
                        else
                        {

#line 240
                            break;
                        }

#line 241
                        uint _S46 = r0_0 + r_0;

#line 241
                        if(_S46 >= m_cur_0)
                        {

#line 241
                            r_0 = r_0 + 1U;

#line 240
                            continue;
                        }
                        uint blk_byte_5 = _S46 * _S4 + _S12 * 18U;

#line 242
                        float _S47 = block_scale_v_0(blk_byte_5, &kernelContext_6);

                        uint qs_byte_5 = blk_byte_5 + 2U + _S6;

#line 244
                        uint _S48 = q_pair_v_0(qs_byte_5, &kernelContext_6);

#line 252
                        float _S49 = a0_0 * float(_S48 & 15U);
                        float _S50 = _S13 * float(_S48 & 3840U);
                        float _S51 = _S17 * float(_S48 & 240U);
                        float _S52 = _S18 * float(_S48 & 61440U);

#line 255
                        uint _S53 = q_pair_v_0(qs_byte_5 + 2U, &kernelContext_6);


                        float acc0_4 = _S49 + a2_0 * float(_S53 & 15U);
                        float acc1_4 = _S50 + _S14 * float(_S53 & 3840U);
                        float acc2_4 = _S51 + _S19 * float(_S53 & 240U);
                        float acc3_4 = _S52 + _S20 * float(_S53 & 61440U);

#line 261
                        uint _S54 = q_pair_v_0(qs_byte_5 + 4U, &kernelContext_6);


                        float acc0_5 = acc0_4 + a4_0 * float(_S54 & 15U);
                        float acc1_5 = acc1_4 + _S15 * float(_S54 & 3840U);
                        float acc2_5 = acc2_4 + _S21 * float(_S54 & 240U);
                        float acc3_5 = acc3_4 + _S22 * float(_S54 & 61440U);

#line 267
                        uint _S55 = q_pair_v_0(qs_byte_5 + 6U, &kernelContext_6);

#line 275
                        sumf_0[r_0] = sumf_0[r_0] + _S47 * (_S25 * -8.0f + (acc0_5 + a6_0 * float(_S55 & 15U)) + (acc1_5 + _S16 * float(_S55 & 3840U)) + (acc2_5 + _S23 * float(_S55 & 240U)) + (acc3_5 + _S24 * float(_S55 & 61440U)));

#line 240
                        r_0 = r_0 + 1U;

#line 240
                    }

#line 199
                }

#line 160
            }

#line 160
            ib_local_0 = ib_local_0 + 16U;

#line 118
        }

#line 281
        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 104
        chunk_b_0 = chunk_b_0 + 16U;

#line 104
    }

#line 104
    r_0 = 0U;

#line 285
    for(;;)
    {

#line 285
        if(r_0 < 4U)
        {
        }
        else
        {

#line 285
            break;
        }

#line 286
        (*(&kernelContext_6)->partials_0)[r_0 * 32U + tid_0] = sumf_0[r_0];

#line 285
        r_0 = r_0 + 1U;

#line 285
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 288
    i_0 = 16U;

    for(;;)
    {

#line 290
        if(i_0 > 0U)
        {
        }
        else
        {

#line 290
            break;
        }

#line 291
        if(tid_0 < i_0)
        {

#line 291
            r_0 = 0U;

            for(;;)
            {

#line 293
                if(r_0 < 4U)
                {
                }
                else
                {

#line 293
                    break;
                }

#line 294
                uint idx_0 = r_0 * 32U + tid_0;
                (*(&kernelContext_6)->partials_0)[idx_0] = (*(&kernelContext_6)->partials_0)[idx_0] + (*(&kernelContext_6)->partials_0)[idx_0 + i_0];

#line 293
                r_0 = r_0 + 1U;

#line 293
            }

#line 291
        }

#line 298
        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 290
        i_0 = i_0 >> 1U;

#line 290
    }

#line 301
    if(tid_0 == 0U)
    {

#line 301
        r_0 = 0U;

        for(;;)
        {

#line 303
            if(r_0 < 4U)
            {
            }
            else
            {

#line 303
                break;
            }

#line 304
            uint _S56 = r0_0 + r_0;

#line 304
            if(_S56 < m_cur_0)
            {

#line 305
                if(which_matrix_0 == 0U)
                {

#line 306
                    *((&kernelContext_6)->y_q_0+_S56) = (*(&kernelContext_6)->partials_0)[r_0 * 32U];

#line 305
                }
                else
                {

#line 307
                    if(which_matrix_0 == 1U)
                    {

#line 308
                        *((&kernelContext_6)->y_k_0+_S56) = (*(&kernelContext_6)->partials_0)[r_0 * 32U];

#line 307
                    }
                    else
                    {
                        *((&kernelContext_6)->y_v_0+_S56) = (*(&kernelContext_6)->partials_0)[r_0 * 32U];

#line 307
                    }

#line 305
                }

#line 304
            }

#line 303
            r_0 = r_0 + 1U;

#line 303
        }

#line 301
    }

#line 315
    return;
}

