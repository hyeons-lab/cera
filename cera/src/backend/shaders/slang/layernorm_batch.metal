#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 48 "cera/src/backend/shaders/slang/layernorm_batch.slang"
struct KernelContext_0
{
    packed_uint4 device* par_buf_0;
    float device* src_buf_0;
    float device* dst_buf_0;
    float device* weight_buf_0;
    float device* bias_buf_0;
    array<float, int(256)> threadgroup* scratch_0;
};


#line 42
float block_sum_0(uint tid_0, float v_0, KernelContext_0 thread* kernelContext_0)
{



    float sg_0 = simd_sum(v_0);
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

#line 50
        lane_0 = (*kernelContext_0->scratch_0)[tid_0];

#line 50
    }
    else
    {

#line 50
        lane_0 = 0.0f;

#line 50
    }
    float total_0 = simd_sum(lane_0);
    if(tid_0 == 0U)
    {

#line 52
        (*kernelContext_0->scratch_0)[int(0)] = total_0;

#line 52
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S1 = (*kernelContext_0->scratch_0)[int(0)];

#line 69
    return _S1;
}



[[kernel]] void layernorm_batch(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 wid_0 [[threadgroup_position_in_grid]], packed_uint4 device* par_buf_1 [[buffer(4)]], float device* src_buf_1 [[buffer(0)]], float device* dst_buf_1 [[buffer(1)]], float device* weight_buf_1 [[buffer(2)]], float device* bias_buf_1 [[buffer(3)]])
{

#line 74
    thread KernelContext_0 kernelContext_1;

#line 74
    (&kernelContext_1)->par_buf_0 = par_buf_1;

#line 74
    (&kernelContext_1)->src_buf_0 = src_buf_1;

#line 74
    (&kernelContext_1)->dst_buf_0 = dst_buf_1;

#line 74
    (&kernelContext_1)->weight_buf_0 = weight_buf_1;

#line 74
    (&kernelContext_1)->bias_buf_0 = bias_buf_1;

#line 74
    threadgroup array<float, int(256)> scratch_1;

#line 74
    (&kernelContext_1)->scratch_0 = &scratch_1;
    uint tid_1 = lid_0.x;
    uint row_0 = wid_0.x;
    uint n_0 = (uint4(*(par_buf_1+int(0))) ).x;
    float eps_0 = (as_type<float>(((uint4(*(par_buf_1+int(0))) ).y)));
    uint _S2 = row_0 * (uint4(*(par_buf_1+int(0))) ).z;
    uint _S3 = row_0 * (uint4(*(par_buf_1+int(0))) ).w;

#line 80
    uint i_0 = tid_1;

#line 80
    float partial_0 = 0.0f;



    for(;;)
    {

#line 84
        if(i_0 < n_0)
        {
        }
        else
        {

#line 84
            break;
        }

#line 85
        float partial_1 = partial_0 + (&kernelContext_1)->src_buf_0[_S2 + i_0];

#line 84
        i_0 = i_0 + 256U;

#line 84
        partial_0 = partial_1;

#line 84
    }

#line 84
    float _S4 = block_sum_0(tid_1, partial_0, &kernelContext_1);


    float _S5 = float(n_0);

#line 87
    float _S6 = _S4 / _S5;


    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 90
    i_0 = tid_1;

#line 90
    partial_0 = 0.0f;



    for(;;)
    {

#line 94
        if(i_0 < n_0)
        {
        }
        else
        {

#line 94
            break;
        }

#line 95
        float d_0 = (&kernelContext_1)->src_buf_0[_S2 + i_0] - _S6;
        float partial_2 = partial_0 + d_0 * d_0;

#line 94
        i_0 = i_0 + 256U;

#line 94
        partial_0 = partial_2;

#line 94
    }

#line 94
    float _S7 = block_sum_0(tid_1, partial_0, &kernelContext_1);



    float _S8 = 1.0f / sqrt(_S7 / _S5 + eps_0);

#line 98
    i_0 = tid_1;


    for(;;)
    {

#line 101
        if(i_0 < n_0)
        {
        }
        else
        {

#line 101
            break;
        }

#line 102
        *((&kernelContext_1)->dst_buf_0+(_S3 + i_0)) = ((&kernelContext_1)->src_buf_0[_S2 + i_0] - _S6) * _S8 * (&kernelContext_1)->weight_buf_0[i_0] + (&kernelContext_1)->bias_buf_0[i_0];

#line 101
        i_0 = i_0 + 256U;

#line 101
    }



    return;
}

