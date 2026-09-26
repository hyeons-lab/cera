#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 65 "cera/src/backend/shaders/slang/rmsnorm.slang"
struct KernelContext_0
{
    packed_uint4 device* p_oop_0;
    float device* src_buf_0;
    float device* dst_buf_0;
    float device* w_oop_0;
    array<float, int(256)> threadgroup* scratch_0;
};


#line 59
float block_sum_0(uint tid_0, float v_0, KernelContext_0 thread* kernelContext_0)
{



    float sg_0 = simd_sum(v_0);
    if((tid_0 & 31U) == 0U)
    {

#line 65
        (*kernelContext_0->scratch_0)[tid_0 >> 5U] = sg_0;

#line 65
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 66
    float lane_0;
    if(tid_0 < 8U)
    {

#line 67
        lane_0 = (*kernelContext_0->scratch_0)[tid_0];

#line 67
    }
    else
    {

#line 67
        lane_0 = 0.0f;

#line 67
    }
    float total_0 = simd_sum(lane_0);
    if(tid_0 == 0U)
    {

#line 69
        (*kernelContext_0->scratch_0)[int(0)] = total_0;

#line 69
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S1 = (*kernelContext_0->scratch_0)[int(0)];

#line 86
    return _S1;
}



[[kernel]] void rmsnorm(uint3 lid_0 [[thread_position_in_threadgroup]], packed_uint4 device* p_oop_1 [[buffer(3)]], float device* src_buf_1 [[buffer(0)]], float device* dst_buf_1 [[buffer(1)]], float device* w_oop_1 [[buffer(2)]])
{

#line 91
    thread KernelContext_0 kernelContext_1;

#line 91
    (&kernelContext_1)->p_oop_0 = p_oop_1;

#line 91
    (&kernelContext_1)->src_buf_0 = src_buf_1;

#line 91
    (&kernelContext_1)->dst_buf_0 = dst_buf_1;

#line 91
    (&kernelContext_1)->w_oop_0 = w_oop_1;

#line 91
    threadgroup array<float, int(256)> scratch_1;

#line 91
    (&kernelContext_1)->scratch_0 = &scratch_1;
    uint tid_1 = lid_0.x;

#line 98
    uint n_0 = (uint4(*(p_oop_1+int(0))) ).x;
    float eps_0 = (as_type<float>(((uint4(*(p_oop_1+int(0))) ).y)));

#line 99
    uint i_0 = tid_1;

#line 99
    float partial_0 = 0.0f;


    for(;;)
    {

#line 102
        if(i_0 < n_0)
        {
        }
        else
        {

#line 102
            break;
        }

#line 103
        float v_1 = (&kernelContext_1)->src_buf_0[i_0];
        float partial_1 = partial_0 + v_1 * v_1;

#line 102
        i_0 = i_0 + 256U;

#line 102
        partial_0 = partial_1;

#line 102
    }

#line 102
    float _S2 = block_sum_0(tid_1, partial_0, &kernelContext_1);



    float _S3 = 1.0f / sqrt(_S2 / float(n_0) + eps_0);

#line 106
    i_0 = tid_1;

    for(;;)
    {

#line 108
        if(i_0 < n_0)
        {
        }
        else
        {

#line 108
            break;
        }

#line 109
        *((&kernelContext_1)->dst_buf_0+i_0) = (&kernelContext_1)->src_buf_0[i_0] * _S3 * (&kernelContext_1)->w_oop_0[i_0];

#line 108
        i_0 = i_0 + 256U;

#line 108
    }

#line 132
    return;
}




[[kernel]] void rmsnorm_out(uint3 lid_1 [[thread_position_in_threadgroup]], packed_uint4 device* p_oop_2 [[buffer(3)]], float device* src_buf_2 [[buffer(0)]], float device* dst_buf_2 [[buffer(1)]], float device* w_oop_2 [[buffer(2)]])
{

#line 138
    thread KernelContext_0 kernelContext_2;

#line 138
    (&kernelContext_2)->p_oop_0 = p_oop_2;

#line 138
    (&kernelContext_2)->src_buf_0 = src_buf_2;

#line 138
    (&kernelContext_2)->dst_buf_0 = dst_buf_2;

#line 138
    (&kernelContext_2)->w_oop_0 = w_oop_2;

#line 138
    threadgroup array<float, int(256)> scratch_2;

#line 138
    (&kernelContext_2)->scratch_0 = &scratch_2;
    uint tid_2 = lid_1.x;
    uint n_1 = (uint4(*(p_oop_2+int(0))) ).x;
    float eps_1 = (as_type<float>(((uint4(*(p_oop_2+int(0))) ).y)));

#line 141
    uint i_1 = tid_2;

#line 141
    float partial_2 = 0.0f;


    for(;;)
    {

#line 144
        if(i_1 < n_1)
        {
        }
        else
        {

#line 144
            break;
        }

#line 145
        float v_2 = (&kernelContext_2)->src_buf_0[i_1];
        float partial_3 = partial_2 + v_2 * v_2;

#line 144
        i_1 = i_1 + 256U;

#line 144
        partial_2 = partial_3;

#line 144
    }

#line 144
    float _S4 = block_sum_0(tid_2, partial_2, &kernelContext_2);



    float _S5 = 1.0f / sqrt(_S4 / float(n_1) + eps_1);

#line 148
    i_1 = tid_2;

    for(;;)
    {

#line 150
        if(i_1 < n_1)
        {
        }
        else
        {

#line 150
            break;
        }

#line 151
        *((&kernelContext_2)->dst_buf_0+i_1) = (&kernelContext_2)->src_buf_0[i_1] * _S5 * (&kernelContext_2)->w_oop_0[i_1];

#line 150
        i_1 = i_1 + 256U;

#line 150
    }


    return;
}

