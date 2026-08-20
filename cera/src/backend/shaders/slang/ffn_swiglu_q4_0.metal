#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 45 "cera/src/backend/shaders/slang/ffn_swiglu_q4_0.slang"
uint get_wid_0(uint3 wid_0)
{

#line 46
    return wid_0.x + wid_0.y * 65535U;
}


#line 104
struct KernelContext_0
{
    packed_uint2 device* params_0;
    uint device* w_gate_0;
    float device* x_0;
    uint device* w_up_0;
    float device* y_0;
    array<float, int(32)> threadgroup* scratch_0;
};


#line 23
uint load_gate_byte_0(uint byte_off_0, KernelContext_0 thread* kernelContext_0)
{
    return (kernelContext_0->w_gate_0[byte_off_0 >> 2U] >> ((byte_off_0 & 3U) * 8U)) & 255U;
}

float load_gate_f16_0(uint byte_off_1, KernelContext_0 thread* kernelContext_1)
{

#line 29
    uint word_0 = kernelContext_1->w_gate_0[byte_off_1 >> 2U];

#line 29
    uint h_0;
    if((byte_off_1 & 2U) != 0U)
    {

#line 30
        h_0 = word_0 >> 16U;

#line 30
    }
    else
    {

#line 30
        h_0 = word_0 & 65535U;

#line 30
    }
    return (as_type<half>((ushort)((h_0))));
}

uint load_up_byte_0(uint byte_off_2, KernelContext_0 thread* kernelContext_2)
{
    return (kernelContext_2->w_up_0[byte_off_2 >> 2U] >> ((byte_off_2 & 3U) * 8U)) & 255U;
}

float load_up_f16_0(uint byte_off_3, KernelContext_0 thread* kernelContext_3)
{

#line 40
    uint word_1 = kernelContext_3->w_up_0[byte_off_3 >> 2U];

#line 40
    uint h_1;
    if((byte_off_3 & 2U) != 0U)
    {

#line 41
        h_1 = word_1 >> 16U;

#line 41
    }
    else
    {

#line 41
        h_1 = word_1 & 65535U;

#line 41
    }
    return (as_type<half>((ushort)((h_1))));
}


#line 51
[[kernel]] void ffn_swiglu_q4_0(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 grp_0 [[threadgroup_position_in_grid]], packed_uint2 device* params_1 [[buffer(4)]], uint device* w_gate_1 [[buffer(0)]], float device* x_1 [[buffer(2)]], uint device* w_up_1 [[buffer(1)]], float device* y_1 [[buffer(3)]])
{

#line 51
    thread KernelContext_0 kernelContext_4;

#line 51
    (&kernelContext_4)->params_0 = params_1;

#line 51
    (&kernelContext_4)->w_gate_0 = w_gate_1;

#line 51
    (&kernelContext_4)->x_0 = x_1;

#line 51
    (&kernelContext_4)->w_up_0 = w_up_1;

#line 51
    (&kernelContext_4)->y_0 = y_1;

#line 51
    threadgroup array<float, int(32)> scratch_1;

#line 51
    (&kernelContext_4)->scratch_0 = &scratch_1;

    uint k_0 = (uint2(*(params_1+int(0))) ).y;
    uint row_0 = get_wid_0(grp_0);

    if(row_0 >= ((uint2(*(params_1+int(0))) ).x))
    {

#line 57
        return;
    }

    uint nb_0 = k_0 / 32U;

    uint _S1 = row_0 * (nb_0 * 18U);

    uint _S2 = lid_0.x;
    uint sub_tid_0 = _S2 & 15U;

#line 65
    uint bi_0;

#line 65
    uint i_0;

#line 65
    float sum_0;

#line 65
    float sumy_0;

#line 65
    float acc_0;



    if(!(_S2 >= 16U))
    {

#line 69
        bi_0 = sub_tid_0;

#line 69
        sum_0 = 0.0f;

        for(;;)
        {

#line 71
            if(bi_0 < nb_0)
            {
            }
            else
            {

#line 71
                break;
            }

#line 72
            uint blk_0 = _S1 + bi_0 * 18U;
            uint _S3 = bi_0 * 32U;

#line 73
            i_0 = 0U;

#line 73
            sumy_0 = 0.0f;

#line 73
            acc_0 = 0.0f;


            for(;;)
            {

#line 76
                if(i_0 < 16U)
                {
                }
                else
                {

#line 76
                    break;
                }

#line 76
                uint _S4 = load_gate_byte_0(blk_0 + 2U + i_0, &kernelContext_4);

                uint _S5 = _S3 + i_0;

#line 78
                float x0_0 = (&kernelContext_4)->x_0[_S5];
                float x1_0 = (&kernelContext_4)->x_0[_S5 + 16U];
                float sumy_1 = sumy_0 + (x0_0 + x1_0);
                float acc_1 = acc_0 + (float(_S4 & 15U) * x0_0 + float(_S4 >> 4U) * x1_0);

#line 76
                i_0 = i_0 + 1U;

#line 76
                sumy_0 = sumy_1;

#line 76
                acc_0 = acc_1;

#line 76
            }

#line 83
            float _S6 = acc_0 - 8.0f * sumy_0;

#line 83
            float _S7 = load_gate_f16_0(blk_0, &kernelContext_4);

#line 83
            float sum_1 = sum_0 + _S6 * _S7;

#line 71
            bi_0 = bi_0 + 16U;

#line 71
            sum_0 = sum_1;

#line 71
        }

#line 69
    }
    else
    {

#line 69
        bi_0 = sub_tid_0;

#line 69
        sum_0 = 0.0f;

#line 87
        for(;;)
        {

#line 87
            if(bi_0 < nb_0)
            {
            }
            else
            {

#line 87
                break;
            }

#line 88
            uint blk_1 = _S1 + bi_0 * 18U;
            uint _S8 = bi_0 * 32U;

#line 89
            i_0 = 0U;

#line 89
            sumy_0 = 0.0f;

#line 89
            acc_0 = 0.0f;


            for(;;)
            {

#line 92
                if(i_0 < 16U)
                {
                }
                else
                {

#line 92
                    break;
                }

#line 92
                uint _S9 = load_up_byte_0(blk_1 + 2U + i_0, &kernelContext_4);

                uint _S10 = _S8 + i_0;

#line 94
                float x0_1 = (&kernelContext_4)->x_0[_S10];
                float x1_1 = (&kernelContext_4)->x_0[_S10 + 16U];
                float sumy_2 = sumy_0 + (x0_1 + x1_1);
                float acc_2 = acc_0 + (float(_S9 & 15U) * x0_1 + float(_S9 >> 4U) * x1_1);

#line 92
                i_0 = i_0 + 1U;

#line 92
                sumy_0 = sumy_2;

#line 92
                acc_0 = acc_2;

#line 92
            }

#line 99
            float _S11 = acc_0 - 8.0f * sumy_0;

#line 99
            float _S12 = load_up_f16_0(blk_1, &kernelContext_4);

#line 99
            float sum_2 = sum_0 + _S11 * _S12;

#line 87
            bi_0 = bi_0 + 16U;

#line 87
            sum_0 = sum_2;

#line 87
        }

#line 69
    }

#line 104
    (*(&kernelContext_4)->scratch_0)[_S2] = sum_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);


    if(sub_tid_0 < 8U)
    {

#line 109
        (*(&kernelContext_4)->scratch_0)[_S2] = (*(&kernelContext_4)->scratch_0)[_S2] + (*(&kernelContext_4)->scratch_0)[_S2 + 8U];

#line 108
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);

    if(sub_tid_0 < 4U)
    {

#line 114
        (*(&kernelContext_4)->scratch_0)[_S2] = (*(&kernelContext_4)->scratch_0)[_S2] + (*(&kernelContext_4)->scratch_0)[_S2 + 4U];

#line 113
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);

    if(sub_tid_0 < 2U)
    {

#line 119
        (*(&kernelContext_4)->scratch_0)[_S2] = (*(&kernelContext_4)->scratch_0)[_S2] + (*(&kernelContext_4)->scratch_0)[_S2 + 2U];

#line 118
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);

    if(sub_tid_0 == 0U)
    {

#line 124
        (*(&kernelContext_4)->scratch_0)[_S2] = (*(&kernelContext_4)->scratch_0)[_S2] + (*(&kernelContext_4)->scratch_0)[_S2 + 1U];

#line 123
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);


    if(_S2 == 0U)
    {



        *((&kernelContext_4)->y_0+row_0) = (*(&kernelContext_4)->scratch_0)[int(0)] / (1.0f + exp(- (*(&kernelContext_4)->scratch_0)[int(0)])) * (*(&kernelContext_4)->scratch_0)[int(16)];

#line 129
    }

#line 136
    return;
}

