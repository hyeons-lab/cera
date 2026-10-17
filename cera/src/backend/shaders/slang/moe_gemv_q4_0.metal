#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 159 "cera/src/backend/shaders/slang/moe_gemv_q4_0.slang"
struct KernelContext_0
{
    packed_uint4 device* params_0;
    uint device* sel_expert_0;
    uint device* w_0;
    float device* x_0;
    float device* y_0;
};


#line 97
uint load_u16_0(uint byte_off_0, KernelContext_0 thread* kernelContext_0)
{

#line 98
    uint word_0 = kernelContext_0->w_0[byte_off_0 >> 2U];

#line 98
    uint _S1;
    if((byte_off_0 & 2U) != 0U)
    {

#line 99
        _S1 = word_0 >> 16U;

#line 99
    }
    else
    {

#line 99
        _S1 = word_0 & 65535U;

#line 99
    }

#line 99
    return _S1;
}


#line 97
uint load_u16_1(uint byte_off_1, KernelContext_0 thread* kernelContext_1)
{

#line 98
    uint word_1 = kernelContext_1->w_0[byte_off_1 >> 2U];

#line 98
    uint _S2;
    if((byte_off_1 & 2U) != 0U)
    {

#line 99
        _S2 = word_1 >> 16U;

#line 99
    }
    else
    {

#line 99
        _S2 = word_1 & 65535U;

#line 99
    }

#line 99
    return _S2;
}

float load_f16_0(uint byte_off_2, KernelContext_0 thread* kernelContext_2)
{

#line 102
    uint _S3 = load_u16_1(byte_off_2, kernelContext_2);
    return (as_type<half>((ushort)((_S3))));
}


#line 69
float block_sum_0(uint tid_0, float v_0)
{



    float _S4 = simd_sum(v_0);

#line 91
    return _S4;
}


#line 108
[[kernel]] void moe_gemv_q4_0(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 grp_0 [[threadgroup_position_in_grid]], packed_uint4 device* params_1 [[buffer(4)]], uint device* sel_expert_1 [[buffer(3)]], uint device* w_1 [[buffer(0)]], float device* x_1 [[buffer(1)]], float device* y_1 [[buffer(2)]])
{

#line 108
    thread KernelContext_0 kernelContext_3;

#line 108
    (&kernelContext_3)->params_0 = params_1;

#line 108
    (&kernelContext_3)->sel_expert_0 = sel_expert_1;

#line 108
    (&kernelContext_3)->w_0 = w_1;

#line 108
    (&kernelContext_3)->x_0 = x_1;

#line 108
    (&kernelContext_3)->y_0 = y_1;
    uint m_0 = (uint4(*(params_1+int(0))) ).x;
    uint k_0 = (uint4(*(params_1+int(0))) ).y;
    uint _S5 = max((uint4(*(params_1+int(0))) ).z, 1U);
    uint n_entries_0 = (uint4(*(params_1+int(0))) ).w;
    uint expert_stride_0 = (uint4(*(params_1+int(1))) ).x;
    bool x_by_entry_0 = ((uint4(*(params_1+int(1))) ).y) != 0U;

    uint row_0 = grp_0.x;
    uint entry_0 = grp_0.y;

#line 117
    bool _S6;


    if(row_0 >= m_0)
    {

#line 120
        _S6 = true;

#line 120
    }
    else
    {

#line 120
        _S6 = entry_0 >= n_entries_0;

#line 120
    }

#line 120
    if(_S6)
    {

#line 121
        return;
    }

#line 121
    uint x_row_0;


    if(x_by_entry_0)
    {

#line 124
        x_row_0 = entry_0;

#line 124
    }
    else
    {

#line 124
        uint _S7 = entry_0 / _S5;

#line 124
        x_row_0 = _S7;

#line 124
    }
    uint _S8 = x_row_0 * k_0;

    uint nb_0 = k_0 / 32U;

    uint _S9 = (&kernelContext_3)->sel_expert_0[entry_0] * expert_stride_0 + row_0 * (nb_0 * 18U);


    uint _S10 = lid_0.x;

#line 132
    uint bi_0 = _S10;

#line 132
    float sum_0 = 0.0f;

#line 132
    for(;;)
    {

#line 132
        if(bi_0 < nb_0)
        {
        }
        else
        {

#line 132
            break;
        }

#line 133
        uint blk_0 = _S9 + bi_0 * 18U;
        uint _S11 = _S8 + bi_0 * 32U;

#line 134
        uint i_0 = 0U;

#line 134
        float xsum_0 = 0.0f;

#line 134
        float acc_0 = 0.0f;

#line 141
        for(;;)
        {

#line 141
            if(i_0 < 8U)
            {
            }
            else
            {

#line 141
                break;
            }

#line 142
            uint _S12 = i_0 * 2U;

#line 142
            uint _S13 = load_u16_0(blk_0 + 2U + _S12, &kernelContext_3);
            uint idx_0 = _S11 + _S12;
            float x0_0 = (&kernelContext_3)->x_0[idx_0];
            float x1_0 = (&kernelContext_3)->x_0[idx_0 + 1U];
            float x16_0 = (&kernelContext_3)->x_0[idx_0 + 16U];
            float x17_0 = (&kernelContext_3)->x_0[idx_0 + 17U];
            float xsum_1 = xsum_0 + (x0_0 + x1_0 + (x16_0 + x17_0));
            float acc_1 = acc_0 + (float(_S13 & 15U) * x0_0 + float((_S13 >> 8U) & 15U) * x1_0 + float((_S13 >> 4U) & 15U) * x16_0 + float(_S13 >> 12U) * x17_0);

#line 141
            i_0 = i_0 + 1U;

#line 141
            xsum_0 = xsum_1;

#line 141
            acc_0 = acc_1;

#line 141
        }

#line 154
        float _S14 = acc_0 - 8.0f * xsum_0;

#line 154
        float _S15 = load_f16_0(blk_0, &kernelContext_3);

#line 154
        float sum_1 = sum_0 + _S14 * _S15;

#line 132
        bi_0 = bi_0 + 32U;

#line 132
        sum_0 = sum_1;

#line 132
    }

#line 157
    float total_0 = block_sum_0(_S10, sum_0);
    if(_S10 == 0U)
    {

#line 159
        *((&kernelContext_3)->y_0+(entry_0 * m_0 + row_0)) = total_0;

#line 158
    }


    return;
}

