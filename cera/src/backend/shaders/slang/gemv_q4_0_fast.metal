#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 44 "cera/src/backend/shaders/slang/gemv_q4_0_fast.slang"
uint get_wid_0(uint3 wid_0)
{

#line 45
    return wid_0.x + wid_0.y * 65535U;
}


#line 188
struct KernelContext_0
{
    packed_uint4 device* params_0;
    float device* x_0;
    uint device* w_0;
    float device* y_0;
    array<float, int(512)> threadgroup* partials_0;
};


#line 48
float block_scale_0(uint blk_byte_0, KernelContext_0 thread* kernelContext_0)
{

#line 49
    uint word_0 = kernelContext_0->w_0[blk_byte_0 >> 2U];

#line 49
    uint scale_bits_0;
    if((blk_byte_0 & 2U) != 0U)
    {

#line 50
        scale_bits_0 = word_0 >> 16U;

#line 50
    }
    else
    {

#line 50
        scale_bits_0 = word_0 & 65535U;

#line 50
    }
    return (as_type<half>((ushort)((scale_bits_0))));
}

uint q_pair_0(uint qs_byte_0, KernelContext_0 thread* kernelContext_1)
{

#line 55
    uint word_1 = kernelContext_1->w_0[qs_byte_0 >> 2U];

#line 55
    uint _S1;
    if((qs_byte_0 & 2U) != 0U)
    {

#line 56
        _S1 = word_1 >> 16U;

#line 56
    }
    else
    {

#line 56
        _S1 = word_1 & 65535U;

#line 56
    }

#line 56
    return _S1;
}



void q4_block_loop_0(uint m_0, uint k_0, uint r0_0, uint ix_0, uint il_0, array<float, int(8)> thread* sumf_0, KernelContext_0 thread* kernelContext_2)
{

#line 69
    uint nb_0 = k_0 / 32U;
    uint _S2 = nb_0 * 18U;

#line 70
    uint r_0 = 0U;


    for(;;)
    {

#line 73
        if(r_0 < 8U)
        {
        }
        else
        {

#line 73
            break;
        }

#line 74
        (*sumf_0)[r_0] = 0.0f;

#line 73
        r_0 = r_0 + 1U;

#line 73
    }



    uint _S3 = ix_0 * 32U + il_0;

#line 77
    uint ib_0 = ix_0;

#line 77
    uint yb_off_0 = _S3;

#line 82
    for(;;)
    {

#line 82
        if(ib_0 < nb_0)
        {
        }
        else
        {

#line 82
            break;
        }

#line 83
        float a0_0 = kernelContext_2->x_0[yb_off_0];
        float a1_0 = kernelContext_2->x_0[yb_off_0 + 1U];
        float a2_0 = kernelContext_2->x_0[yb_off_0 + 2U];
        float a3_0 = kernelContext_2->x_0[yb_off_0 + 3U];
        float a4_0 = kernelContext_2->x_0[yb_off_0 + 4U];
        float a5_0 = kernelContext_2->x_0[yb_off_0 + 5U];
        float a6_0 = kernelContext_2->x_0[yb_off_0 + 6U];
        float a7_0 = kernelContext_2->x_0[yb_off_0 + 7U];
        float a8_0 = kernelContext_2->x_0[yb_off_0 + 16U];
        float a9_0 = kernelContext_2->x_0[yb_off_0 + 17U];
        float a10_0 = kernelContext_2->x_0[yb_off_0 + 18U];
        float a11_0 = kernelContext_2->x_0[yb_off_0 + 19U];
        float a12_0 = kernelContext_2->x_0[yb_off_0 + 20U];
        float a13_0 = kernelContext_2->x_0[yb_off_0 + 21U];
        float a14_0 = kernelContext_2->x_0[yb_off_0 + 22U];
        float a15_0 = kernelContext_2->x_0[yb_off_0 + 23U];

#line 105
        float _S4 = a1_0 * 0.00390625f;

        float _S5 = a3_0 * 0.00390625f;

        float _S6 = a5_0 * 0.00390625f;

        float _S7 = a7_0 * 0.00390625f;
        float _S8 = a8_0 * 0.0625f;
        float _S9 = a9_0 * 0.000244140625f;
        float _S10 = a10_0 * 0.0625f;
        float _S11 = a11_0 * 0.000244140625f;
        float _S12 = a12_0 * 0.0625f;
        float _S13 = a13_0 * 0.000244140625f;
        float _S14 = a14_0 * 0.0625f;
        float _S15 = a15_0 * 0.000244140625f;



        float _S16 = a0_0 + a1_0 + (a2_0 + a3_0) + (a4_0 + a5_0) + (a6_0 + a7_0) + (a8_0 + a9_0 + (a10_0 + a11_0) + (a12_0 + a13_0) + (a14_0 + a15_0));

#line 123
        r_0 = 0U;


        for(;;)
        {

#line 126
            if(r_0 < 8U)
            {
            }
            else
            {

#line 126
                break;
            }

#line 127
            uint _S17 = r0_0 + r_0;

#line 127
            if(_S17 >= m_0)
            {

#line 127
                r_0 = r_0 + 1U;

#line 126
                continue;
            }
            uint blk_byte_1 = _S17 * _S2 + ib_0 * 18U;

#line 128
            float _S18 = block_scale_0(blk_byte_1, kernelContext_2);

            uint qs_byte_1 = blk_byte_1 + 2U + il_0;

#line 130
            uint _S19 = q_pair_0(qs_byte_1, kernelContext_2);

#line 138
            float _S20 = a0_0 * float(_S19 & 15U);
            float _S21 = _S4 * float(_S19 & 3840U);
            float _S22 = _S8 * float(_S19 & 240U);
            float _S23 = _S9 * float(_S19 & 61440U);

#line 141
            uint _S24 = q_pair_0(qs_byte_1 + 2U, kernelContext_2);


            float acc0_0 = _S20 + a2_0 * float(_S24 & 15U);
            float acc1_0 = _S21 + _S5 * float(_S24 & 3840U);
            float acc2_0 = _S22 + _S10 * float(_S24 & 240U);
            float acc3_0 = _S23 + _S11 * float(_S24 & 61440U);

#line 147
            uint _S25 = q_pair_0(qs_byte_1 + 4U, kernelContext_2);


            float acc0_1 = acc0_0 + a4_0 * float(_S25 & 15U);
            float acc1_1 = acc1_0 + _S6 * float(_S25 & 3840U);
            float acc2_1 = acc2_0 + _S12 * float(_S25 & 240U);
            float acc3_1 = acc3_0 + _S13 * float(_S25 & 61440U);

#line 153
            uint _S26 = q_pair_0(qs_byte_1 + 6U, kernelContext_2);

#line 161
            (*sumf_0)[r_0] = (*sumf_0)[r_0] + _S18 * (_S16 * -8.0f + (acc0_1 + a6_0 * float(_S26 & 15U)) + (acc1_1 + _S7 * float(_S26 & 3840U)) + (acc2_1 + _S14 * float(_S26 & 240U)) + (acc3_1 + _S15 * float(_S26 & 61440U)));

#line 126
            r_0 = r_0 + 1U;

#line 126
        }

#line 165
        uint yb_off_1 = yb_off_0 + 1024U;

#line 165
        ib_0 = ib_0 + 32U;

#line 165
        yb_off_0 = yb_off_1;

#line 82
    }

#line 167
    return;
}


[[kernel]] void gemv_q4_0_fast(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 wid_1 [[threadgroup_position_in_grid]], packed_uint4 device* params_1 [[buffer(3)]], float device* x_1 [[buffer(1)]], uint device* w_1 [[buffer(0)]], float device* y_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_3;

#line 173
    (&kernelContext_3)->params_0 = params_1;

#line 173
    (&kernelContext_3)->x_0 = x_1;

#line 173
    (&kernelContext_3)->w_0 = w_1;

#line 173
    (&kernelContext_3)->y_0 = y_1;

#line 173
    threadgroup array<float, int(512)> partials_1;

#line 173
    (&kernelContext_3)->partials_0 = &partials_1;

    uint m_1 = (uint4(*(params_1+int(0))) ).x;

    uint r0_1 = get_wid_0(wid_1) * 8U;
    uint tid_0 = lid_0.x;

#line 183
    thread array<float, int(8)> sumf_1;

#line 183
    q4_block_loop_0(m_1, (uint4(*(params_1+int(0))) ).y, r0_1, tid_0 / 2U, (tid_0 & 1U) * 8U, &sumf_1, &kernelContext_3);

#line 183
    uint r_1 = 0U;



    for(;;)
    {

#line 187
        if(r_1 < 8U)
        {
        }
        else
        {

#line 187
            break;
        }

#line 188
        (*(&kernelContext_3)->partials_0)[r_1 * 64U + tid_0] = sumf_1[r_1];

#line 187
        r_1 = r_1 + 1U;

#line 187
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 190
    uint stride_0 = 32U;

    for(;;)
    {

#line 192
        if(stride_0 > 0U)
        {
        }
        else
        {

#line 192
            break;
        }

#line 193
        if(tid_0 < stride_0)
        {

#line 193
            r_1 = 0U;

            for(;;)
            {

#line 195
                if(r_1 < 8U)
                {
                }
                else
                {

#line 195
                    break;
                }

#line 196
                uint idx_0 = r_1 * 64U + tid_0;
                (*(&kernelContext_3)->partials_0)[idx_0] = (*(&kernelContext_3)->partials_0)[idx_0] + (*(&kernelContext_3)->partials_0)[idx_0 + stride_0];

#line 195
                r_1 = r_1 + 1U;

#line 195
            }

#line 193
        }

#line 200
        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 192
        stride_0 = stride_0 >> 1U;

#line 192
    }

#line 203
    if(tid_0 == 0U)
    {

#line 203
        r_1 = 0U;

        for(;;)
        {

#line 205
            if(r_1 < 8U)
            {
            }
            else
            {

#line 205
                break;
            }

#line 206
            uint _S27 = r0_1 + r_1;

#line 206
            if(_S27 < m_1)
            {

#line 207
                *((&kernelContext_3)->y_0+_S27) = (*(&kernelContext_3)->partials_0)[r_1 * 64U];

#line 206
            }

#line 205
            r_1 = r_1 + 1U;

#line 205
        }

#line 203
    }

#line 211
    return;
}

