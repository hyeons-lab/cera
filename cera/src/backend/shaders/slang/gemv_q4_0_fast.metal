#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 22 "cera/src/backend/shaders/slang/gemv_q4_0_fast.slang"
uint get_wid_0(uint3 wid_0)
{

#line 23
    return wid_0.x + wid_0.y * 65535U;
}


#line 175
struct KernelContext_0
{
    packed_uint4 device* params_0;
    float device* x_0;
    uint device* w_0;
    float device* y_0;
    array<float, int(512)> threadgroup* x_stage_0;
    array<float, int(256)> threadgroup* partials_0;
};


#line 26
float block_scale_0(uint blk_byte_0, KernelContext_0 thread* kernelContext_0)
{

#line 27
    uint word_off_0 = blk_byte_0 / 4U;
    uint byte_rem_0 = blk_byte_0 % 4U;

#line 28
    uint scale_bits_0;

    if(byte_rem_0 == 0U)
    {

#line 30
        scale_bits_0 = kernelContext_0->w_0[word_off_0] & 65535U;

#line 30
    }
    else
    {

#line 32
        if(byte_rem_0 == 2U)
        {

#line 32
            scale_bits_0 = (kernelContext_0->w_0[word_off_0] >> 16U) & 65535U;

#line 32
        }
        else
        {

#line 32
            scale_bits_0 = ((kernelContext_0->w_0[word_off_0] >> (byte_rem_0 * 8U)) | (kernelContext_0->w_0[word_off_0 + 1U] << ((4U - byte_rem_0) * 8U))) & 65535U;

#line 32
        }

#line 30
    }

#line 39
    return (as_type<half>((ushort)((scale_bits_0))));
}

uint q_pair_0(uint qs_byte_0, KernelContext_0 thread* kernelContext_1)
{

#line 43
    uint word_off_1 = qs_byte_0 / 4U;
    uint byte_rem_1 = qs_byte_0 % 4U;
    if(byte_rem_1 == 0U)
    {

#line 46
        return kernelContext_1->w_0[word_off_1] & 65535U;
    }
    else
    {

#line 47
        if(byte_rem_1 == 2U)
        {

#line 48
            return (kernelContext_1->w_0[word_off_1] >> 16U) & 65535U;
        }
        else
        {
            return ((kernelContext_1->w_0[word_off_1] >> (byte_rem_1 * 8U)) | (kernelContext_1->w_0[word_off_1 + 1U] << ((4U - byte_rem_1) * 8U))) & 65535U;
        }

#line 52
    }

#line 52
}


#line 58
[[kernel]] void gemv_q4_0_fast(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 wid_1 [[threadgroup_position_in_grid]], packed_uint4 device* params_1 [[buffer(3)]], float device* x_1 [[buffer(1)]], uint device* w_1 [[buffer(0)]], float device* y_1 [[buffer(2)]])
{
    uint i_0;

#line 60
    thread KernelContext_0 kernelContext_2;

#line 60
    (&kernelContext_2)->params_0 = params_1;

#line 60
    (&kernelContext_2)->x_0 = x_1;

#line 60
    (&kernelContext_2)->w_0 = w_1;

#line 60
    (&kernelContext_2)->y_0 = y_1;

#line 60
    threadgroup array<float, int(512)> x_stage_1;

#line 60
    (&kernelContext_2)->x_stage_0 = &x_stage_1;

#line 60
    threadgroup array<float, int(256)> partials_1;

#line 60
    (&kernelContext_2)->partials_0 = &partials_1;

    uint _S1 = (uint4(*(params_1+int(0))) ).x;
    uint k_0 = (uint4(*(params_1+int(0))) ).y;
    uint nb_0 = k_0 / 32U;
    uint _S2 = nb_0 * 18U;
    uint _S3 = get_wid_0(wid_1) * 8U;
    uint tid_0 = lid_0.x;

    uint _S4 = tid_0 / 2U;
    uint _S5 = (tid_0 & 1U) * 8U;

    thread array<float, int(8)> sumf_0;

#line 72
    uint r_0 = 0U;

    for(;;)
    {

#line 74
        if(r_0 < 8U)
        {
        }
        else
        {

#line 74
            break;
        }

#line 75
        sumf_0[r_0] = 0.0f;

#line 74
        r_0 = r_0 + 1U;

#line 74
    }

#line 74
    uint chunk_b_0 = 0U;



    for(;;)
    {

#line 78
        if(chunk_b_0 < nb_0)
        {
        }
        else
        {

#line 78
            break;
        }

#line 79
        uint chunk_k_start_0 = chunk_b_0 * 32U;
        uint _S6 = min(512U, k_0 - chunk_k_start_0);

#line 80
        i_0 = tid_0;
        for(;;)
        {

#line 81
            if(i_0 < _S6)
            {
            }
            else
            {

#line 81
                break;
            }

#line 82
            (*(&kernelContext_2)->x_stage_0)[i_0] = (&kernelContext_2)->x_0[chunk_k_start_0 + i_0];

#line 81
            i_0 = i_0 + 32U;

#line 81
        }


        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 84
        uint ib_local_0 = _S4;


        for(;;)
        {

#line 87
            bool _S7;

#line 87
            if(ib_local_0 < 16U)
            {

#line 87
                _S7 = (chunk_b_0 + ib_local_0) < nb_0;

#line 87
            }
            else
            {

#line 87
                _S7 = false;

#line 87
            }

#line 87
            if(_S7)
            {
            }
            else
            {

#line 87
                break;
            }

#line 88
            uint _S8 = chunk_b_0 + ib_local_0;
            uint yb_stage_off_0 = ib_local_0 * 32U + _S5;

            float a0_0 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0];

            float a2_0 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 2U];

            float a4_0 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 4U];

            float a6_0 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 6U];

#line 109
            float _S9 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 1U] / 256.0f;

            float _S10 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 3U] / 256.0f;

            float _S11 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 5U] / 256.0f;

            float _S12 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 7U] / 256.0f;
            float _S13 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 16U] / 16.0f;
            float _S14 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 17U] / 4096.0f;
            float _S15 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 18U] / 16.0f;
            float _S16 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 19U] / 4096.0f;
            float _S17 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 20U] / 16.0f;
            float _S18 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 21U] / 4096.0f;
            float _S19 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 22U] / 16.0f;
            float _S20 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 23U] / 4096.0f;



            float _S21 = (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0] + (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 1U] + ((*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 2U] + (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 3U]) + ((*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 4U] + (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 5U]) + ((*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 6U] + (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 7U]) + ((*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 16U] + (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 17U] + ((*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 18U] + (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 19U]) + ((*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 20U] + (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 21U]) + ((*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 22U] + (*(&kernelContext_2)->x_stage_0)[yb_stage_off_0 + 23U]));

#line 127
            r_0 = 0U;


            for(;;)
            {

#line 130
                if(r_0 < 8U)
                {
                }
                else
                {

#line 130
                    break;
                }

#line 131
                uint _S22 = _S3 + r_0;

#line 131
                if(_S22 >= _S1)
                {

#line 131
                    r_0 = r_0 + 1U;

#line 130
                    continue;
                }
                uint blk_byte_1 = _S22 * _S2 + _S8 * 18U;

#line 132
                float _S23 = block_scale_0(blk_byte_1, &kernelContext_2);

                uint qs_byte_1 = blk_byte_1 + 2U + _S5;

#line 134
                uint _S24 = q_pair_0(qs_byte_1, &kernelContext_2);

#line 142
                float _S25 = a0_0 * float(_S24 & 15U);
                float _S26 = _S9 * float(_S24 & 3840U);
                float _S27 = _S13 * float(_S24 & 240U);
                float _S28 = _S14 * float(_S24 & 61440U);

#line 145
                uint _S29 = q_pair_0(qs_byte_1 + 2U, &kernelContext_2);


                float acc0_0 = _S25 + a2_0 * float(_S29 & 15U);
                float acc1_0 = _S26 + _S10 * float(_S29 & 3840U);
                float acc2_0 = _S27 + _S15 * float(_S29 & 240U);
                float acc3_0 = _S28 + _S16 * float(_S29 & 61440U);

#line 151
                uint _S30 = q_pair_0(qs_byte_1 + 4U, &kernelContext_2);


                float acc0_1 = acc0_0 + a4_0 * float(_S30 & 15U);
                float acc1_1 = acc1_0 + _S11 * float(_S30 & 3840U);
                float acc2_1 = acc2_0 + _S17 * float(_S30 & 240U);
                float acc3_1 = acc3_0 + _S18 * float(_S30 & 61440U);

#line 157
                uint _S31 = q_pair_0(qs_byte_1 + 6U, &kernelContext_2);

#line 165
                sumf_0[r_0] = sumf_0[r_0] + _S23 * (_S21 * -8.0f + (acc0_1 + a6_0 * float(_S31 & 15U)) + (acc1_1 + _S12 * float(_S31 & 3840U)) + (acc2_1 + _S19 * float(_S31 & 240U)) + (acc3_1 + _S20 * float(_S31 & 61440U)));

#line 130
                r_0 = r_0 + 1U;

#line 130
            }

#line 130
            ib_local_0 = ib_local_0 + 16U;

#line 87
        }

#line 170
        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 78
        chunk_b_0 = chunk_b_0 + 16U;

#line 78
    }

#line 78
    r_0 = 0U;

#line 174
    for(;;)
    {

#line 174
        if(r_0 < 8U)
        {
        }
        else
        {

#line 174
            break;
        }

#line 175
        (*(&kernelContext_2)->partials_0)[r_0 * 32U + tid_0] = sumf_0[r_0];

#line 174
        r_0 = r_0 + 1U;

#line 174
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 177
    i_0 = 16U;

    for(;;)
    {

#line 179
        if(i_0 > 0U)
        {
        }
        else
        {

#line 179
            break;
        }

#line 180
        if(tid_0 < i_0)
        {

#line 180
            r_0 = 0U;

            for(;;)
            {

#line 182
                if(r_0 < 8U)
                {
                }
                else
                {

#line 182
                    break;
                }

#line 183
                uint idx_0 = r_0 * 32U + tid_0;
                (*(&kernelContext_2)->partials_0)[idx_0] = (*(&kernelContext_2)->partials_0)[idx_0] + (*(&kernelContext_2)->partials_0)[idx_0 + i_0];

#line 182
                r_0 = r_0 + 1U;

#line 182
            }

#line 180
        }

#line 187
        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 179
        i_0 = i_0 >> 1U;

#line 179
    }

#line 190
    if(tid_0 == 0U)
    {

#line 190
        r_0 = 0U;

        for(;;)
        {

#line 192
            if(r_0 < 8U)
            {
            }
            else
            {

#line 192
                break;
            }

#line 193
            uint _S32 = _S3 + r_0;

#line 193
            if(_S32 < _S1)
            {

#line 194
                *((&kernelContext_2)->y_0+_S32) = (*(&kernelContext_2)->partials_0)[r_0 * 32U];

#line 193
            }

#line 192
            r_0 = r_0 + 1U;

#line 192
        }

#line 190
    }

#line 198
    return;
}

