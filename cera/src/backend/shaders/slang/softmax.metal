#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 48 "softmax.slang"
struct KernelContext_0
{
    packed_uint2 device* par_buf_0;
    float device* x_buf_0;
    array<float, int(256)> threadgroup* scratch_0;
};


#line 42
float block_max_0(uint tid_0, float v_0, KernelContext_0 thread* kernelContext_0)
{



    float sg_0 = simd_max(v_0);
    if((tid_0 & 31U) == 0U)
    {

#line 48
        (*kernelContext_0->scratch_0)[tid_0 >> 5U] = sg_0;

#line 48
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 49
    float lane_0;

    if(tid_0 < 8U)
    {

#line 51
        lane_0 = (*kernelContext_0->scratch_0)[tid_0];

#line 51
    }
    else
    {

#line 51
        lane_0 = -3.4028234663852886e+38f;

#line 51
    }
    float total_0 = simd_max(lane_0);
    if(tid_0 == 0U)
    {

#line 53
        (*kernelContext_0->scratch_0)[int(0)] = total_0;

#line 53
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S1 = (*kernelContext_0->scratch_0)[int(0)];

#line 70
    return _S1;
}


float block_sum_0(uint tid_1, float v_1, KernelContext_0 thread* kernelContext_1)
{



    float sg_1 = simd_sum(v_1);
    if((tid_1 & 31U) == 0U)
    {

#line 80
        (*kernelContext_1->scratch_0)[tid_1 >> 5U] = sg_1;

#line 80
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 81
    float lane_1;
    if(tid_1 < 8U)
    {

#line 82
        lane_1 = (*kernelContext_1->scratch_0)[tid_1];

#line 82
    }
    else
    {

#line 82
        lane_1 = 0.0f;

#line 82
    }
    float total_1 = simd_sum(lane_1);
    if(tid_1 == 0U)
    {

#line 84
        (*kernelContext_1->scratch_0)[int(0)] = total_1;

#line 84
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S2 = (*kernelContext_1->scratch_0)[int(0)];

#line 101
    return _S2;
}



[[kernel]] void softmax(uint3 lid_0 [[thread_position_in_threadgroup]], packed_uint2 device* par_buf_1 [[buffer(1)]], float device* x_buf_1 [[buffer(0)]])
{

#line 106
    thread KernelContext_0 kernelContext_2;

#line 106
    (&kernelContext_2)->par_buf_0 = par_buf_1;

#line 106
    (&kernelContext_2)->x_buf_0 = x_buf_1;

#line 106
    threadgroup array<float, int(256)> scratch_1;

#line 106
    (&kernelContext_2)->scratch_0 = &scratch_1;
    uint tid_2 = lid_0.x;
    uint _S3 = (uint2(*(par_buf_1+int(0))) ).x;

#line 108
    float local_max_0 = -3.4028234663852886e+38f;

#line 108
    uint i_0 = tid_2;



    for(;;)
    {

#line 112
        if(i_0 < _S3)
        {
        }
        else
        {

#line 112
            break;
        }

#line 113
        float _S4 = max(local_max_0, *((&kernelContext_2)->x_buf_0+i_0));

#line 112
        uint i_1 = i_0 + 256U;

#line 112
        local_max_0 = _S4;

#line 112
        i_0 = i_1;

#line 112
    }

#line 112
    float _S5 = block_max_0(tid_2, local_max_0, &kernelContext_2);

#line 120
    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 120
    i_0 = tid_2;

#line 120
    float partial_0 = 0.0f;



    for(;;)
    {

#line 124
        if(i_0 < _S3)
        {
        }
        else
        {

#line 124
            break;
        }

#line 125
        float e_0 = exp(*((&kernelContext_2)->x_buf_0+i_0) - _S5);
        *((&kernelContext_2)->x_buf_0+i_0) = e_0;
        float partial_1 = partial_0 + e_0;

#line 124
        i_0 = i_0 + 256U;

#line 124
        partial_0 = partial_1;

#line 124
    }

#line 124
    float _S6 = block_sum_0(tid_2, partial_0, &kernelContext_2);

#line 129
    float _S7 = 1.0f / _S6;

#line 129
    i_0 = tid_2;


    for(;;)
    {

#line 132
        if(i_0 < _S3)
        {
        }
        else
        {

#line 132
            break;
        }

#line 133
        *((&kernelContext_2)->x_buf_0+i_0) = *((&kernelContext_2)->x_buf_0+i_0) * _S7;

#line 132
        i_0 = i_0 + 256U;

#line 132
    }


    return;
}

