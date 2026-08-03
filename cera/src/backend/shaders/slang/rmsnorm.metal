#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 56 "cera/src/backend/shaders/slang/rmsnorm.slang"
struct KernelContext_0
{
    packed_uint4 device* p_metal_0;
    float device* src_buf_0;
    float device* dst_buf_0;
    float device* w_metal_0;
    array<float, int(256)> threadgroup* scratch_0;
};


#line 50
float block_sum_0(uint tid_0, float v_0, KernelContext_0 thread* kernelContext_0)
{



    float sg_0 = simd_sum(v_0);
    if((tid_0 & 31U) == 0U)
    {

#line 56
        (*kernelContext_0->scratch_0)[tid_0 >> 5U] = sg_0;

#line 56
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 57
    float lane_0;
    if(tid_0 < 8U)
    {

#line 58
        lane_0 = (*kernelContext_0->scratch_0)[tid_0];

#line 58
    }
    else
    {

#line 58
        lane_0 = 0.0f;

#line 58
    }
    float total_0 = simd_sum(lane_0);
    if(tid_0 == 0U)
    {

#line 60
        (*kernelContext_0->scratch_0)[int(0)] = total_0;

#line 60
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S1 = (*kernelContext_0->scratch_0)[int(0)];

#line 77
    return _S1;
}



[[kernel]] void rmsnorm(uint3 lid_0 [[thread_position_in_threadgroup]], packed_uint4 device* p_metal_1 [[buffer(3)]], float device* src_buf_1 [[buffer(0)]], float device* dst_buf_1 [[buffer(1)]], float device* w_metal_1 [[buffer(2)]])
{

#line 82
    thread KernelContext_0 kernelContext_1;

#line 82
    (&kernelContext_1)->p_metal_0 = p_metal_1;

#line 82
    (&kernelContext_1)->src_buf_0 = src_buf_1;

#line 82
    (&kernelContext_1)->dst_buf_0 = dst_buf_1;

#line 82
    (&kernelContext_1)->w_metal_0 = w_metal_1;

#line 82
    threadgroup array<float, int(256)> scratch_1;

#line 82
    (&kernelContext_1)->scratch_0 = &scratch_1;
    uint tid_1 = lid_0.x;

#line 89
    uint n_0 = (uint4(*(p_metal_1+int(0))) ).x;
    float eps_0 = (as_type<float>(((uint4(*(p_metal_1+int(0))) ).y)));

#line 90
    uint i_0 = tid_1;

#line 90
    float partial_0 = 0.0f;


    for(;;)
    {

#line 93
        if(i_0 < n_0)
        {
        }
        else
        {

#line 93
            break;
        }

#line 94
        float v_1 = (&kernelContext_1)->src_buf_0[i_0];
        float partial_1 = partial_0 + v_1 * v_1;

#line 93
        i_0 = i_0 + 256U;

#line 93
        partial_0 = partial_1;

#line 93
    }

#line 93
    float _S2 = block_sum_0(tid_1, partial_0, &kernelContext_1);



    float _S3 = 1.0f / sqrt(_S2 / float(n_0) + eps_0);

#line 97
    i_0 = tid_1;

    for(;;)
    {

#line 99
        if(i_0 < n_0)
        {
        }
        else
        {

#line 99
            break;
        }

#line 100
        *((&kernelContext_1)->dst_buf_0+i_0) = (&kernelContext_1)->src_buf_0[i_0] * _S3 * (&kernelContext_1)->w_metal_0[i_0];

#line 99
        i_0 = i_0 + 256U;

#line 99
    }

#line 123
    return;
}

