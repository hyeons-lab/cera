#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 58 "cera/src/backend/shaders/slang/mel_norm.slang"
struct KernelContext_0
{
    packed_uint4 device* par_buf_0;
    float device* dst_buf_0;
    float device* mel_buf_0;
    array<float, int(256)> threadgroup* scratch_0;
};


#line 52
float block_sum_0(uint tid_0, float v_0, KernelContext_0 thread* kernelContext_0)
{



    float sg_0 = simd_sum(v_0);
    if((tid_0 & 31U) == 0U)
    {

#line 58
        (*kernelContext_0->scratch_0)[tid_0 >> 5U] = sg_0;

#line 58
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 59
    float lane_0;
    if(tid_0 < 8U)
    {

#line 60
        lane_0 = (*kernelContext_0->scratch_0)[tid_0];

#line 60
    }
    else
    {

#line 60
        lane_0 = 0.0f;

#line 60
    }
    float total_0 = simd_sum(lane_0);
    if(tid_0 == 0U)
    {

#line 62
        (*kernelContext_0->scratch_0)[int(0)] = total_0;

#line 62
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float _S1 = (*kernelContext_0->scratch_0)[int(0)];

#line 79
    return _S1;
}



[[kernel]] void mel_norm(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 wid_0 [[threadgroup_position_in_grid]], packed_uint4 device* par_buf_1 [[buffer(2)]], float device* dst_buf_1 [[buffer(1)]], float device* mel_buf_1 [[buffer(0)]])
{

#line 84
    thread KernelContext_0 kernelContext_1;

#line 84
    (&kernelContext_1)->par_buf_0 = par_buf_1;

#line 84
    (&kernelContext_1)->dst_buf_0 = dst_buf_1;

#line 84
    (&kernelContext_1)->mel_buf_0 = mel_buf_1;

#line 84
    threadgroup array<float, int(256)> scratch_1;

#line 84
    (&kernelContext_1)->scratch_0 = &scratch_1;
    uint tid_1 = lid_0.x;
    uint mi_0 = wid_0.x;
    uint _S2 = (uint4(*(par_buf_1+int(0))) ).x;
    uint n_frames_0 = (uint4(*(par_buf_1+int(0))) ).y;
    uint eff_0 = (uint4(*(par_buf_1+int(0))) ).z;
    float eps_0 = (as_type<float>(((uint4(*(par_buf_1+int(0))) ).w)));
    uint _S3 = mi_0 * n_frames_0;

#line 91
    uint t_0;



    if(eff_0 <= 1U)
    {

#line 95
        t_0 = tid_1;
        for(;;)
        {

#line 96
            if(t_0 < n_frames_0)
            {
            }
            else
            {

#line 96
                break;
            }

#line 97
            *((&kernelContext_1)->dst_buf_0+(t_0 * _S2 + mi_0)) = 0.0f;

#line 96
            t_0 = t_0 + 256U;

#line 96
        }


        return;
    }

#line 99
    t_0 = tid_1;

#line 99
    float partial_0 = 0.0f;

#line 104
    for(;;)
    {

#line 104
        if(t_0 < eff_0)
        {
        }
        else
        {

#line 104
            break;
        }

#line 105
        float partial_1 = partial_0 + (&kernelContext_1)->mel_buf_0[_S3 + t_0];

#line 104
        t_0 = t_0 + 256U;

#line 104
        partial_0 = partial_1;

#line 104
    }

#line 104
    float _S4 = block_sum_0(tid_1, partial_0, &kernelContext_1);


    float _S5 = _S4 / float(eff_0);


    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 110
    t_0 = tid_1;

#line 110
    partial_0 = 0.0f;



    for(;;)
    {

#line 114
        if(t_0 < eff_0)
        {
        }
        else
        {

#line 114
            break;
        }

#line 115
        float d_0 = (&kernelContext_1)->mel_buf_0[_S3 + t_0] - _S5;
        float partial_2 = partial_0 + d_0 * d_0;

#line 114
        t_0 = t_0 + 256U;

#line 114
        partial_0 = partial_2;

#line 114
    }

#line 114
    float _S6 = block_sum_0(tid_1, partial_0, &kernelContext_1);



    float _S7 = 1.0f / sqrt(_S6 / float(eff_0 - 1U) + eps_0);

#line 118
    t_0 = tid_1;


    for(;;)
    {

#line 121
        if(t_0 < n_frames_0)
        {
        }
        else
        {

#line 121
            break;
        }

#line 121
        float v_1;
        if(t_0 < eff_0)
        {

#line 122
            v_1 = ((&kernelContext_1)->mel_buf_0[_S3 + t_0] - _S5) * _S7;

#line 122
        }
        else
        {

#line 122
            v_1 = 0.0f;

#line 122
        }
        *((&kernelContext_1)->dst_buf_0+(t_0 * _S2 + mi_0)) = v_1;

#line 121
        t_0 = t_0 + 256U;

#line 121
    }



    return;
}

