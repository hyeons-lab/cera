#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 18 "cera/src/backend/shaders/slang/gemv_q4_0_fast.slang"
uint get_wid_0(uint3 wid_0)
{

#line 19
    return wid_0.x + wid_0.y * 65535U;
}


#line 142
struct KernelContext_0
{
    packed_uint4 device* params_0;
    float device* x_0;
    uint device* w_0;
    float device* y_0;
    array<float, int(256)> threadgroup* partials_0;
};


#line 22
float block_scale_0(uint blk_byte_0, KernelContext_0 thread* kernelContext_0)
{

#line 23
    uint word_0 = kernelContext_0->w_0[blk_byte_0 >> 2U];

#line 23
    uint scale_bits_0;
    if((blk_byte_0 & 2U) != 0U)
    {

#line 24
        scale_bits_0 = word_0 >> 16U;

#line 24
    }
    else
    {

#line 24
        scale_bits_0 = word_0 & 65535U;

#line 24
    }
    return (as_type<half>((ushort)((scale_bits_0))));
}

uint q_pair_0(uint qs_byte_0, KernelContext_0 thread* kernelContext_1)
{

#line 29
    uint word_1 = kernelContext_1->w_0[qs_byte_0 >> 2U];

#line 29
    uint _S1;
    if((qs_byte_0 & 2U) != 0U)
    {

#line 30
        _S1 = word_1 >> 16U;

#line 30
    }
    else
    {

#line 30
        _S1 = word_1 & 65535U;

#line 30
    }

#line 30
    return _S1;
}



[[kernel]] void gemv_q4_0_fast(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 wid_1 [[threadgroup_position_in_grid]], packed_uint4 device* params_1 [[buffer(3)]], float device* x_1 [[buffer(1)]], uint device* w_1 [[buffer(0)]], float device* y_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_2;

#line 37
    (&kernelContext_2)->params_0 = params_1;

#line 37
    (&kernelContext_2)->x_0 = x_1;

#line 37
    (&kernelContext_2)->w_0 = w_1;

#line 37
    (&kernelContext_2)->y_0 = y_1;

#line 37
    threadgroup array<float, int(256)> partials_1;

#line 37
    (&kernelContext_2)->partials_0 = &partials_1;

    uint _S2 = (uint4(*(params_1+int(0))) ).x;

    uint nb_0 = (uint4(*(params_1+int(0))) ).y / 32U;
    uint _S3 = nb_0 * 18U;
    uint _S4 = get_wid_0(wid_1) * 8U;
    uint tid_0 = lid_0.x;

    uint ix_0 = tid_0 / 2U;
    uint il_0 = (tid_0 & 1U) * 8U;

    thread array<float, int(8)> sumf_0;

#line 49
    uint r_0 = 0U;

    for(;;)
    {

#line 51
        if(r_0 < 8U)
        {
        }
        else
        {

#line 51
            break;
        }

#line 52
        sumf_0[r_0] = 0.0f;

#line 51
        r_0 = r_0 + 1U;

#line 51
    }



    uint _S5 = ix_0 * 32U + il_0;

#line 55
    uint ib_0 = ix_0;

#line 55
    uint yb_off_0 = _S5;


    for(;;)
    {

#line 58
        if(ib_0 < nb_0)
        {
        }
        else
        {

#line 58
            break;
        }

#line 59
        float a0_0 = (&kernelContext_2)->x_0[yb_off_0];
        float a1_0 = (&kernelContext_2)->x_0[yb_off_0 + 1U];
        float a2_0 = (&kernelContext_2)->x_0[yb_off_0 + 2U];
        float a3_0 = (&kernelContext_2)->x_0[yb_off_0 + 3U];
        float a4_0 = (&kernelContext_2)->x_0[yb_off_0 + 4U];
        float a5_0 = (&kernelContext_2)->x_0[yb_off_0 + 5U];
        float a6_0 = (&kernelContext_2)->x_0[yb_off_0 + 6U];
        float a7_0 = (&kernelContext_2)->x_0[yb_off_0 + 7U];
        float a8_0 = (&kernelContext_2)->x_0[yb_off_0 + 16U];
        float a9_0 = (&kernelContext_2)->x_0[yb_off_0 + 17U];
        float a10_0 = (&kernelContext_2)->x_0[yb_off_0 + 18U];
        float a11_0 = (&kernelContext_2)->x_0[yb_off_0 + 19U];
        float a12_0 = (&kernelContext_2)->x_0[yb_off_0 + 20U];
        float a13_0 = (&kernelContext_2)->x_0[yb_off_0 + 21U];
        float a14_0 = (&kernelContext_2)->x_0[yb_off_0 + 22U];
        float a15_0 = (&kernelContext_2)->x_0[yb_off_0 + 23U];


        float _S6 = a1_0 / 256.0f;

        float _S7 = a3_0 / 256.0f;

        float _S8 = a5_0 / 256.0f;

        float _S9 = a7_0 / 256.0f;
        float _S10 = a8_0 / 16.0f;
        float _S11 = a9_0 / 4096.0f;
        float _S12 = a10_0 / 16.0f;
        float _S13 = a11_0 / 4096.0f;
        float _S14 = a12_0 / 16.0f;
        float _S15 = a13_0 / 4096.0f;
        float _S16 = a14_0 / 16.0f;
        float _S17 = a15_0 / 4096.0f;



        float _S18 = a0_0 + a1_0 + (a2_0 + a3_0) + (a4_0 + a5_0) + (a6_0 + a7_0) + (a8_0 + a9_0 + (a10_0 + a11_0) + (a12_0 + a13_0) + (a14_0 + a15_0));

#line 95
        r_0 = 0U;


        for(;;)
        {

#line 98
            if(r_0 < 8U)
            {
            }
            else
            {

#line 98
                break;
            }

#line 99
            uint _S19 = _S4 + r_0;

#line 99
            if(_S19 >= _S2)
            {

#line 99
                r_0 = r_0 + 1U;

#line 98
                continue;
            }
            uint blk_byte_1 = _S19 * _S3 + ib_0 * 18U;

#line 100
            float _S20 = block_scale_0(blk_byte_1, &kernelContext_2);

            uint qs_byte_1 = blk_byte_1 + 2U + il_0;

#line 102
            uint _S21 = q_pair_0(qs_byte_1, &kernelContext_2);

#line 110
            float _S22 = a0_0 * float(_S21 & 15U);
            float _S23 = _S6 * float(_S21 & 3840U);
            float _S24 = _S10 * float(_S21 & 240U);
            float _S25 = _S11 * float(_S21 & 61440U);

#line 113
            uint _S26 = q_pair_0(qs_byte_1 + 2U, &kernelContext_2);


            float acc0_0 = _S22 + a2_0 * float(_S26 & 15U);
            float acc1_0 = _S23 + _S7 * float(_S26 & 3840U);
            float acc2_0 = _S24 + _S12 * float(_S26 & 240U);
            float acc3_0 = _S25 + _S13 * float(_S26 & 61440U);

#line 119
            uint _S27 = q_pair_0(qs_byte_1 + 4U, &kernelContext_2);


            float acc0_1 = acc0_0 + a4_0 * float(_S27 & 15U);
            float acc1_1 = acc1_0 + _S8 * float(_S27 & 3840U);
            float acc2_1 = acc2_0 + _S14 * float(_S27 & 240U);
            float acc3_1 = acc3_0 + _S15 * float(_S27 & 61440U);

#line 125
            uint _S28 = q_pair_0(qs_byte_1 + 6U, &kernelContext_2);

#line 133
            sumf_0[r_0] = sumf_0[r_0] + _S20 * (_S18 * -8.0f + (acc0_1 + a6_0 * float(_S28 & 15U)) + (acc1_1 + _S9 * float(_S28 & 3840U)) + (acc2_1 + _S16 * float(_S28 & 240U)) + (acc3_1 + _S17 * float(_S28 & 61440U)));

#line 98
            r_0 = r_0 + 1U;

#line 98
        }

#line 137
        uint yb_off_1 = yb_off_0 + 512U;

#line 137
        ib_0 = ib_0 + 16U;

#line 137
        yb_off_0 = yb_off_1;

#line 58
    }

#line 58
    r_0 = 0U;

#line 141
    for(;;)
    {

#line 141
        if(r_0 < 8U)
        {
        }
        else
        {

#line 141
            break;
        }

#line 142
        (*(&kernelContext_2)->partials_0)[r_0 * 32U + tid_0] = sumf_0[r_0];

#line 141
        r_0 = r_0 + 1U;

#line 141
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 144
    uint stride_0 = 16U;

    for(;;)
    {

#line 146
        if(stride_0 > 0U)
        {
        }
        else
        {

#line 146
            break;
        }

#line 147
        if(tid_0 < stride_0)
        {

#line 147
            r_0 = 0U;

            for(;;)
            {

#line 149
                if(r_0 < 8U)
                {
                }
                else
                {

#line 149
                    break;
                }

#line 150
                uint idx_0 = r_0 * 32U + tid_0;
                (*(&kernelContext_2)->partials_0)[idx_0] = (*(&kernelContext_2)->partials_0)[idx_0] + (*(&kernelContext_2)->partials_0)[idx_0 + stride_0];

#line 149
                r_0 = r_0 + 1U;

#line 149
            }

#line 147
        }

#line 154
        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 146
        stride_0 = stride_0 >> 1U;

#line 146
    }

#line 157
    if(tid_0 == 0U)
    {

#line 157
        r_0 = 0U;

        for(;;)
        {

#line 159
            if(r_0 < 8U)
            {
            }
            else
            {

#line 159
                break;
            }

#line 160
            uint _S29 = _S4 + r_0;

#line 160
            if(_S29 < _S2)
            {

#line 161
                *((&kernelContext_2)->y_0+_S29) = (*(&kernelContext_2)->partials_0)[r_0 * 32U];

#line 160
            }

#line 159
            r_0 = r_0 + 1U;

#line 159
        }

#line 157
    }

#line 165
    return;
}

