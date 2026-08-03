#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 45 "cera/src/backend/shaders/slang/per_head_rmsnorm.slang"
struct KernelContext_0
{
    packed_uint4 device* par_buf_0;
    float device* x_buf_0;
    float device* weight_buf_0;
    array<float, int(256)> threadgroup* scratch_0;
};


#line 39
float block_sum_0(uint tid_0, float v_0, KernelContext_0 thread* kernelContext_0)
{



    float sg_0 = simd_sum(v_0);
    if((tid_0 & 31U) == 0U)
    {

#line 45
        (*kernelContext_0->scratch_0)[tid_0 >> 5U] = sg_0;

#line 45
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 46
    float lane_0;

    if(tid_0 < 8U)
    {

#line 48
        lane_0 = (*kernelContext_0->scratch_0)[tid_0];

#line 48
    }
    else
    {

#line 48
        lane_0 = 0.0f;

#line 48
    }
    float total_0 = simd_sum(lane_0);
    if(tid_0 == 0U)
    {

#line 50
        (*kernelContext_0->scratch_0)[int(0)] = total_0;

#line 50
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S1 = (*kernelContext_0->scratch_0)[int(0)];

#line 67
    return _S1;
}



[[kernel]] void per_head_rmsnorm(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 wid_0 [[threadgroup_position_in_grid]], packed_uint4 device* par_buf_1 [[buffer(2)]], float device* x_buf_1 [[buffer(0)]], float device* weight_buf_1 [[buffer(1)]])
{

#line 72
    thread KernelContext_0 kernelContext_1;

#line 72
    (&kernelContext_1)->par_buf_0 = par_buf_1;

#line 72
    (&kernelContext_1)->x_buf_0 = x_buf_1;

#line 72
    (&kernelContext_1)->weight_buf_0 = weight_buf_1;

#line 72
    threadgroup array<float, int(256)> scratch_1;

#line 72
    (&kernelContext_1)->scratch_0 = &scratch_1;
    uint tid_1 = lid_0.x;

    uint head_dim_0 = (uint4(*(par_buf_1+int(0))) ).x;
    float eps_0 = (as_type<float>(((uint4(*(par_buf_1+int(0))) ).y)));
    uint _S2 = wid_0.x * head_dim_0;

#line 77
    uint i_0 = tid_1;

#line 77
    float partial_0 = 0.0f;



    for(;;)
    {

#line 81
        if(i_0 < head_dim_0)
        {
        }
        else
        {

#line 81
            break;
        }

#line 82
        float device* _S3 = (&kernelContext_1)->x_buf_0+(_S2 + i_0);
        float partial_1 = partial_0 + *_S3 * *_S3;

#line 81
        i_0 = i_0 + 256U;

#line 81
        partial_0 = partial_1;

#line 81
    }

#line 81
    float _S4 = block_sum_0(tid_1, partial_0, &kernelContext_1);

#line 87
    float _S5 = 1.0f / sqrt(_S4 / float(head_dim_0) + eps_0);

#line 87
    i_0 = tid_1;


    for(;;)
    {

#line 90
        if(i_0 < head_dim_0)
        {
        }
        else
        {

#line 90
            break;
        }

#line 91
        uint _S6 = _S2 + i_0;

#line 91
        *((&kernelContext_1)->x_buf_0+_S6) = *((&kernelContext_1)->x_buf_0+_S6) * _S5 * (&kernelContext_1)->weight_buf_0[i_0];

#line 90
        i_0 = i_0 + 256U;

#line 90
    }


    return;
}

