#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 83 "cera/src/backend/shaders/slang/audio_xl_attention.slang"
struct KernelContext_0
{
    packed_uint4 device* par_buf_0;
    float device* q_buf_0;
    float device* bu_buf_0;
    float device* bv_buf_0;
    float device* k_buf_0;
    float device* p_buf_0;
    float device* v_buf_0;
    float device* out_buf_0;
    array<float, int(128)> threadgroup* qu_0;
    array<float, int(128)> threadgroup* qv_0;
    array<float, int(1024)> threadgroup* scores_0;
    array<float, int(256)> threadgroup* scratch_0;
};


#line 77
float block_max_0(uint tid_0, float v_0, KernelContext_0 thread* kernelContext_0)
{



    float sg_0 = simd_max(v_0);
    if((tid_0 & 31U) == 0U)
    {

#line 83
        (*kernelContext_0->scratch_0)[tid_0 >> 5U] = sg_0;

#line 83
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 84
    float lane_0;
    if(tid_0 < 8U)
    {

#line 85
        lane_0 = (*kernelContext_0->scratch_0)[tid_0];

#line 85
    }
    else
    {

#line 85
        lane_0 = -3.4028234663852886e+38f;

#line 85
    }
    float total_0 = simd_max(lane_0);
    if(tid_0 == 0U)
    {

#line 87
        (*kernelContext_0->scratch_0)[int(0)] = total_0;

#line 87
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S1 = (*kernelContext_0->scratch_0)[int(0)];

#line 104
    return _S1;
}


float block_sum_0(uint tid_1, float v_1, KernelContext_0 thread* kernelContext_1)
{



    float sg_1 = simd_sum(v_1);
    if((tid_1 & 31U) == 0U)
    {

#line 114
        (*kernelContext_1->scratch_0)[tid_1 >> 5U] = sg_1;

#line 114
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 115
    float lane_1;
    if(tid_1 < 8U)
    {

#line 116
        lane_1 = (*kernelContext_1->scratch_0)[tid_1];

#line 116
    }
    else
    {

#line 116
        lane_1 = 0.0f;

#line 116
    }
    float total_1 = simd_sum(lane_1);
    if(tid_1 == 0U)
    {

#line 118
        (*kernelContext_1->scratch_0)[int(0)] = total_1;

#line 118
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S2 = (*kernelContext_1->scratch_0)[int(0)];

#line 135
    return _S2;
}



[[kernel]] void audio_xl_attention(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 wid_0 [[threadgroup_position_in_grid]], packed_uint4 device* par_buf_1 [[buffer(7)]], float device* q_buf_1 [[buffer(0)]], float device* bu_buf_1 [[buffer(4)]], float device* bv_buf_1 [[buffer(5)]], float device* k_buf_1 [[buffer(1)]], float device* p_buf_1 [[buffer(3)]], float device* v_buf_1 [[buffer(2)]], float device* out_buf_1 [[buffer(6)]])
{

#line 140
    float ac_0;

#line 140
    float bd_0;

#line 140
    thread KernelContext_0 kernelContext_2;

#line 140
    (&kernelContext_2)->par_buf_0 = par_buf_1;

#line 140
    (&kernelContext_2)->q_buf_0 = q_buf_1;

#line 140
    (&kernelContext_2)->bu_buf_0 = bu_buf_1;

#line 140
    (&kernelContext_2)->bv_buf_0 = bv_buf_1;

#line 140
    (&kernelContext_2)->k_buf_0 = k_buf_1;

#line 140
    (&kernelContext_2)->p_buf_0 = p_buf_1;

#line 140
    (&kernelContext_2)->v_buf_0 = v_buf_1;

#line 140
    (&kernelContext_2)->out_buf_0 = out_buf_1;

#line 140
    threadgroup array<float, int(128)> qu_1;

#line 140
    (&kernelContext_2)->qu_0 = &qu_1;

#line 140
    threadgroup array<float, int(128)> qv_1;

#line 140
    (&kernelContext_2)->qv_0 = &qv_1;

#line 140
    threadgroup array<float, int(1024)> scores_1;

#line 140
    (&kernelContext_2)->scores_0 = &scores_1;

#line 140
    threadgroup array<float, int(256)> scratch_1;

#line 140
    (&kernelContext_2)->scratch_0 = &scratch_1;
    uint tid_2 = lid_0.x;
    uint tokens_0 = (uint4(*(par_buf_1+int(0))) ).x;

    uint head_dim_0 = (uint4(*(par_buf_1+int(0))) ).z;
    float _S3 = (as_type<float>(((uint4(*(par_buf_1+int(0))) ).w)));

    uint q_idx_0 = wid_0.x;

    uint dim_0 = (uint4(*(par_buf_1+int(0))) ).y * head_dim_0;
    uint head_base_0 = wid_0.y * head_dim_0;
    uint _S4 = q_idx_0 * dim_0 + head_base_0;

#line 151
    uint d_0 = tid_2;


    for(;;)
    {

#line 154
        if(d_0 < head_dim_0)
        {
        }
        else
        {

#line 154
            break;
        }

#line 155
        float qd_0 = (&kernelContext_2)->q_buf_0[_S4 + d_0];
        uint _S5 = head_base_0 + d_0;

#line 156
        (*(&kernelContext_2)->qu_0)[d_0] = qd_0 + (&kernelContext_2)->bu_buf_0[_S5];
        (*(&kernelContext_2)->qv_0)[d_0] = qd_0 + (&kernelContext_2)->bv_buf_0[_S5];

#line 154
        d_0 = d_0 + 256U;

#line 154
    }

#line 159
    threadgroup_barrier(mem_flags::mem_threadgroup);


    uint _S6 = tokens_0 - 1U;

#line 162
    uint key_0 = tid_2;
    for(;;)
    {

#line 163
        if(key_0 < tokens_0)
        {
        }
        else
        {

#line 163
            break;
        }

#line 164
        uint _S7 = key_0 * dim_0 + head_base_0;
        uint _S8 = (_S6 + key_0 - q_idx_0) * dim_0 + head_base_0;

#line 165
        d_0 = 0U;

#line 165
        ac_0 = 0.0f;

#line 165
        bd_0 = 0.0f;


        for(;;)
        {

#line 168
            if(d_0 < head_dim_0)
            {
            }
            else
            {

#line 168
                break;
            }

#line 169
            float ac_1 = ac_0 + (*(&kernelContext_2)->qu_0)[d_0] * (&kernelContext_2)->k_buf_0[_S7 + d_0];
            float bd_1 = bd_0 + (*(&kernelContext_2)->qv_0)[d_0] * (&kernelContext_2)->p_buf_0[_S8 + d_0];

#line 168
            d_0 = d_0 + 1U;

#line 168
            ac_0 = ac_1;

#line 168
            bd_0 = bd_1;

#line 168
        }



        (*(&kernelContext_2)->scores_0)[key_0] = (ac_0 + bd_0) * _S3;

#line 163
        key_0 = key_0 + 256U;

#line 163
    }

#line 174
    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 174
    ac_0 = -3.4028234663852886e+38f;

#line 174
    key_0 = tid_2;



    for(;;)
    {

#line 178
        if(key_0 < tokens_0)
        {
        }
        else
        {

#line 178
            break;
        }

#line 179
        float _S9 = max(ac_0, (*(&kernelContext_2)->scores_0)[key_0]);

#line 178
        uint key_1 = key_0 + 256U;

#line 178
        ac_0 = _S9;

#line 178
        key_0 = key_1;

#line 178
    }

#line 178
    float _S10 = block_max_0(tid_2, ac_0, &kernelContext_2);

#line 178
    key_0 = tid_2;

#line 178
    bd_0 = 0.0f;

#line 190
    for(;;)
    {

#line 190
        if(key_0 < tokens_0)
        {
        }
        else
        {

#line 190
            break;
        }

#line 191
        float e_0 = exp((*(&kernelContext_2)->scores_0)[key_0] - _S10);
        (*(&kernelContext_2)->scores_0)[key_0] = e_0;
        float partial_0 = bd_0 + e_0;

#line 190
        key_0 = key_0 + 256U;

#line 190
        bd_0 = partial_0;

#line 190
    }

#line 195
    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 195
    float _S11 = block_sum_0(tid_2, bd_0, &kernelContext_2);
    float _S12 = 1.0f / _S11;

#line 196
    d_0 = tid_2;


    for(;;)
    {

#line 199
        if(d_0 < head_dim_0)
        {
        }
        else
        {

#line 199
            break;
        }

#line 199
        key_0 = 0U;

#line 199
        float acc_0 = 0.0f;

        for(;;)
        {

#line 201
            if(key_0 < tokens_0)
            {
            }
            else
            {

#line 201
                break;
            }

#line 202
            float acc_1 = acc_0 + (*(&kernelContext_2)->scores_0)[key_0] * (&kernelContext_2)->v_buf_0[key_0 * dim_0 + head_base_0 + d_0];

#line 201
            key_0 = key_0 + 1U;

#line 201
            acc_0 = acc_1;

#line 201
        }


        *((&kernelContext_2)->out_buf_0+(_S4 + d_0)) = acc_0 * _S12;

#line 199
        d_0 = d_0 + 256U;

#line 199
    }

#line 206
    return;
}

