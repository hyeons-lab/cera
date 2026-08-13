#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 152 "cera/src/backend/shaders/slang/moe_gemv_q4_0.slang"
struct KernelContext_0
{
    packed_uint4 device* params_0;
    uint device* sel_expert_0;
    uint device* w_0;
    float device* x_0;
    float device* y_0;
};


#line 94
uint load_byte_0(uint byte_off_0, KernelContext_0 thread* kernelContext_0)
{
    return (kernelContext_0->w_0[byte_off_0 >> 2U] >> ((byte_off_0 & 3U) * 8U)) & 255U;
}




float load_f16_0(uint byte_off_1, KernelContext_0 thread* kernelContext_1)
{

#line 103
    uint word_0 = kernelContext_1->w_0[byte_off_1 >> 2U];

#line 103
    uint h_0;
    if((byte_off_1 & 2U) != 0U)
    {

#line 104
        h_0 = word_0 >> 16U;

#line 104
    }
    else
    {

#line 104
        h_0 = word_0 & 65535U;

#line 104
    }
    return (as_type<half>((ushort)((h_0))));
}


#line 69
float block_sum_0(uint tid_0, float v_0)
{



    float _S1 = simd_sum(v_0);

#line 91
    return _S1;
}


#line 110
[[kernel]] void moe_gemv_q4_0(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 grp_0 [[threadgroup_position_in_grid]], packed_uint4 device* params_1 [[buffer(4)]], uint device* sel_expert_1 [[buffer(3)]], uint device* w_1 [[buffer(0)]], float device* x_1 [[buffer(1)]], float device* y_1 [[buffer(2)]])
{

#line 110
    thread KernelContext_0 kernelContext_2;

#line 110
    (&kernelContext_2)->params_0 = params_1;

#line 110
    (&kernelContext_2)->sel_expert_0 = sel_expert_1;

#line 110
    (&kernelContext_2)->w_0 = w_1;

#line 110
    (&kernelContext_2)->x_0 = x_1;

#line 110
    (&kernelContext_2)->y_0 = y_1;
    uint m_0 = (uint4(*(params_1+int(0))) ).x;
    uint k_0 = (uint4(*(params_1+int(0))) ).y;
    uint _S2 = max((uint4(*(params_1+int(0))) ).z, 1U);
    uint n_entries_0 = (uint4(*(params_1+int(0))) ).w;
    uint expert_stride_0 = (uint4(*(params_1+int(1))) ).x;
    bool x_by_entry_0 = ((uint4(*(params_1+int(1))) ).y) != 0U;

    uint row_0 = grp_0.x;
    uint entry_0 = grp_0.y;

#line 119
    bool _S3;


    if(row_0 >= m_0)
    {

#line 122
        _S3 = true;

#line 122
    }
    else
    {

#line 122
        _S3 = entry_0 >= n_entries_0;

#line 122
    }

#line 122
    if(_S3)
    {

#line 123
        return;
    }

#line 123
    uint x_row_0;


    if(x_by_entry_0)
    {

#line 126
        x_row_0 = entry_0;

#line 126
    }
    else
    {

#line 126
        uint _S4 = entry_0 / _S2;

#line 126
        x_row_0 = _S4;

#line 126
    }
    uint _S5 = x_row_0 * k_0;

    uint nb_0 = k_0 / 32U;

    uint _S6 = (&kernelContext_2)->sel_expert_0[entry_0] * expert_stride_0 + row_0 * (nb_0 * 18U);


    uint _S7 = lid_0.x;

#line 134
    uint bi_0 = _S7;

#line 134
    float sum_0 = 0.0f;

#line 134
    for(;;)
    {

#line 134
        if(bi_0 < nb_0)
        {
        }
        else
        {

#line 134
            break;
        }

#line 135
        uint blk_0 = _S6 + bi_0 * 18U;
        uint _S8 = _S5 + bi_0 * 32U;

#line 136
        uint i_0 = 0U;

#line 136
        float acc_0 = 0.0f;

#line 142
        for(;;)
        {

#line 142
            if(i_0 < 16U)
            {
            }
            else
            {

#line 142
                break;
            }

#line 142
            uint _S9 = load_byte_0(blk_0 + 2U + i_0, &kernelContext_2);

            uint _S10 = _S8 + i_0;
            float acc_1 = acc_0 + (float(_S9 & 15U) - 8.0f) * (&kernelContext_2)->x_0[_S10] + (float(_S9 >> 4U) - 8.0f) * (&kernelContext_2)->x_0[_S10 + 16U];

#line 142
            i_0 = i_0 + 1U;

#line 142
            acc_0 = acc_1;

#line 142
        }

#line 142
        float _S11 = load_f16_0(blk_0, &kernelContext_2);

#line 147
        float sum_1 = sum_0 + acc_0 * _S11;

#line 134
        bi_0 = bi_0 + 32U;

#line 134
        sum_0 = sum_1;

#line 134
    }

#line 150
    float total_0 = block_sum_0(_S7, sum_0);
    if(_S7 == 0U)
    {

#line 152
        *((&kernelContext_2)->y_0+(entry_0 * m_0 + row_0)) = total_0;

#line 151
    }


    return;
}

