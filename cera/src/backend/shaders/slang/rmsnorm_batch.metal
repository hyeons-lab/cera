#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 51 "cera/src/backend/shaders/slang/rmsnorm_batch.slang"
struct KernelContext_0
{
    uint device* par_buf_0;
    float device* src_buf_0;
    float device* dst_buf_0;
    float device* w_buf_0;
    float device* res_buf_0;
    array<float, int(256)> threadgroup* scratch_0;
};


#line 45
float block_sum_0(uint tid_0, float v_0, KernelContext_0 thread* kernelContext_0)
{



    float sg_0 = simd_sum(v_0);
    if((tid_0 & 31U) == 0U)
    {

#line 51
        (*kernelContext_0->scratch_0)[tid_0 >> 5U] = sg_0;

#line 51
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 52
    float lane_0;
    if(tid_0 < 8U)
    {

#line 53
        lane_0 = (*kernelContext_0->scratch_0)[tid_0];

#line 53
    }
    else
    {

#line 53
        lane_0 = 0.0f;

#line 53
    }
    float total_0 = simd_sum(lane_0);
    if(tid_0 == 0U)
    {

#line 55
        (*kernelContext_0->scratch_0)[int(0)] = total_0;

#line 55
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S1 = (*kernelContext_0->scratch_0)[int(0)];

#line 72
    return _S1;
}



[[kernel]] void rmsnorm_batch(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 wid_0 [[threadgroup_position_in_grid]], uint device* par_buf_1 [[buffer(3)]], float device* src_buf_1 [[buffer(0)]], float device* dst_buf_1 [[buffer(1)]], float device* w_buf_1 [[buffer(2)]], float device* res_buf_1 [[buffer(4)]])
{

#line 77
    thread KernelContext_0 kernelContext_1;

#line 77
    (&kernelContext_1)->par_buf_0 = par_buf_1;

#line 77
    (&kernelContext_1)->src_buf_0 = src_buf_1;

#line 77
    (&kernelContext_1)->dst_buf_0 = dst_buf_1;

#line 77
    (&kernelContext_1)->w_buf_0 = w_buf_1;

#line 77
    (&kernelContext_1)->res_buf_0 = res_buf_1;

#line 77
    threadgroup array<float, int(256)> scratch_1;

#line 77
    (&kernelContext_1)->scratch_0 = &scratch_1;
    uint tid_1 = lid_0.x;
    uint n_0 = par_buf_1[int(0)];
    float eps_0 = (as_type<float>((par_buf_1[int(1)])));
    uint _S2 = wid_0.x;

#line 81
    uint _S3 = _S2 * par_buf_1[int(2)];
    uint _S4 = _S2 * par_buf_1[int(3)];

#line 82
    uint i_0 = tid_1;

#line 82
    float partial_0 = 0.0f;


    for(;;)
    {

#line 85
        if(i_0 < n_0)
        {
        }
        else
        {

#line 85
            break;
        }

#line 86
        float device* _S5 = (&kernelContext_1)->src_buf_0+(_S3 + i_0);
        float partial_1 = partial_0 + *_S5 * *_S5;

#line 85
        i_0 = i_0 + 256U;

#line 85
        partial_0 = partial_1;

#line 85
    }

#line 85
    float _S6 = block_sum_0(tid_1, partial_0, &kernelContext_1);



    float _S7 = 1.0f / sqrt(_S6 / float(n_0) + eps_0);

#line 89
    i_0 = tid_1;

    for(;;)
    {

#line 91
        if(i_0 < n_0)
        {
        }
        else
        {

#line 91
            break;
        }

#line 92
        *((&kernelContext_1)->dst_buf_0+(_S4 + i_0)) = *((&kernelContext_1)->src_buf_0+(_S3 + i_0)) * _S7 * (&kernelContext_1)->w_buf_0[i_0];

#line 91
        i_0 = i_0 + 256U;

#line 91
    }


    return;
}


[[kernel]] void add_rmsnorm_batch(uint3 lid_1 [[thread_position_in_threadgroup]], uint3 wid_1 [[threadgroup_position_in_grid]], uint device* par_buf_2 [[buffer(3)]], float device* src_buf_2 [[buffer(0)]], float device* dst_buf_2 [[buffer(1)]], float device* w_buf_2 [[buffer(2)]], float device* res_buf_2 [[buffer(4)]])
{

#line 98
    thread KernelContext_0 kernelContext_2;

#line 98
    (&kernelContext_2)->par_buf_0 = par_buf_2;

#line 98
    (&kernelContext_2)->src_buf_0 = src_buf_2;

#line 98
    (&kernelContext_2)->dst_buf_0 = dst_buf_2;

#line 98
    (&kernelContext_2)->w_buf_0 = w_buf_2;

#line 98
    (&kernelContext_2)->res_buf_0 = res_buf_2;

#line 98
    threadgroup array<float, int(256)> scratch_2;

#line 98
    (&kernelContext_2)->scratch_0 = &scratch_2;
    uint tid_2 = lid_1.x;
    uint n_1 = par_buf_2[int(0)];
    float eps_1 = (as_type<float>((par_buf_2[int(1)])));
    uint _S8 = wid_1.x;

#line 102
    uint _S9 = _S8 * par_buf_2[int(2)];
    uint _S10 = _S8 * par_buf_2[int(3)];
    float _S11 = (as_type<float>((par_buf_2[int(4)])));

#line 104
    uint i_1 = tid_2;

#line 104
    float partial_2 = 0.0f;

#line 109
    for(;;)
    {

#line 109
        if(i_1 < n_1)
        {
        }
        else
        {

#line 109
            break;
        }

#line 110
        uint _S12 = _S9 + i_1;

#line 110
        float v_1 = *((&kernelContext_2)->src_buf_0+_S12) + _S11 * (&kernelContext_2)->res_buf_0[_S12];
        *((&kernelContext_2)->src_buf_0+_S12) = v_1;
        float partial_3 = partial_2 + v_1 * v_1;

#line 109
        i_1 = i_1 + 256U;

#line 109
        partial_2 = partial_3;

#line 109
    }

#line 109
    float _S13 = block_sum_0(tid_2, partial_2, &kernelContext_2);

#line 114
    float _S14 = 1.0f / sqrt(_S13 / float(n_1) + eps_1);

#line 114
    i_1 = tid_2;

    for(;;)
    {

#line 116
        if(i_1 < n_1)
        {
        }
        else
        {

#line 116
            break;
        }

#line 117
        *((&kernelContext_2)->dst_buf_0+(_S10 + i_1)) = *((&kernelContext_2)->src_buf_0+(_S9 + i_1)) * _S14 * (&kernelContext_2)->w_buf_0[i_1];

#line 116
        i_1 = i_1 + 256U;

#line 116
    }


    return;
}

