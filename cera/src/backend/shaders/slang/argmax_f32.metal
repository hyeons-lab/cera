#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 74 "cera/src/backend/shaders/slang/argmax_f32.slang"
struct KernelContext_0
{
    packed_uint2 device* par_buf_0;
    float device* x_buf_0;
    uint device* out_buf_0;
    array<float, int(256)> threadgroup* scratch_v_0;
    array<uint, int(256)> threadgroup* scratch_i_0;
};


#line 51
uint block_argmax_0(uint tid_0, float val_0, uint idx_0, KernelContext_0 thread* kernelContext_0)
{

#line 62
    float ov_0 = (simd_shuffle_down((val_0), (16U)));
    uint oi_0 = (simd_shuffle_down((idx_0), (16U)));

#line 63
    float best_v_0;

#line 63
    uint best_i_0;
    if(ov_0 > val_0)
    {

#line 64
        best_v_0 = ov_0;

#line 64
        best_i_0 = oi_0;

#line 64
    }
    else
    {

#line 64
        best_v_0 = val_0;

#line 64
        best_i_0 = idx_0;

#line 64
    }

#line 62
    float ov_1 = (simd_shuffle_down((best_v_0), (8U)));
    uint oi_1 = (simd_shuffle_down((best_i_0), (8U)));
    if(ov_1 > best_v_0)
    {

#line 64
        best_v_0 = ov_1;

#line 64
        best_i_0 = oi_1;

#line 64
    }

#line 62
    float ov_2 = (simd_shuffle_down((best_v_0), (4U)));
    uint oi_2 = (simd_shuffle_down((best_i_0), (4U)));
    if(ov_2 > best_v_0)
    {

#line 64
        best_v_0 = ov_2;

#line 64
        best_i_0 = oi_2;

#line 64
    }

#line 62
    float ov_3 = (simd_shuffle_down((best_v_0), (2U)));
    uint oi_3 = (simd_shuffle_down((best_i_0), (2U)));
    if(ov_3 > best_v_0)
    {

#line 64
        best_v_0 = ov_3;

#line 64
        best_i_0 = oi_3;

#line 64
    }

#line 62
    float ov_4 = (simd_shuffle_down((best_v_0), (1U)));
    uint oi_4 = (simd_shuffle_down((best_i_0), (1U)));
    if(ov_4 > best_v_0)
    {

#line 64
        best_v_0 = ov_4;

#line 64
        best_i_0 = oi_4;

#line 64
    }

#line 70
    uint simd_lane_0 = tid_0 & 31U;
    uint simd_id_0 = tid_0 >> 5U;
    bool _S1 = simd_lane_0 == 0U;

#line 72
    if(_S1)
    {

#line 73
        (*kernelContext_0->scratch_v_0)[simd_id_0] = best_v_0;
        (*kernelContext_0->scratch_i_0)[simd_id_0] = best_i_0;

#line 72
    }



    threadgroup_barrier(mem_flags::mem_threadgroup);

    if(simd_id_0 == 0U)
    {

#line 79
        bool _S2 = simd_lane_0 < 8U;

#line 79
        if(_S2)
        {

#line 79
            best_v_0 = (*kernelContext_0->scratch_v_0)[simd_lane_0];

#line 79
        }
        else
        {

#line 79
            best_v_0 = -3.4028234663852886e+38f;

#line 79
        }
        if(_S2)
        {

#line 80
            best_i_0 = (*kernelContext_0->scratch_i_0)[simd_lane_0];

#line 80
        }
        else
        {

#line 80
            best_i_0 = 0U;

#line 80
        }


        float ov_5 = (simd_shuffle_down((best_v_0), (4U)));
        uint oi_5 = (simd_shuffle_down((best_i_0), (4U)));

#line 84
        float v_0;

#line 84
        uint ix_0;
        if(ov_5 > best_v_0)
        {

#line 85
            v_0 = ov_5;

#line 85
            ix_0 = oi_5;

#line 85
        }
        else
        {

#line 85
            v_0 = best_v_0;

#line 85
            ix_0 = best_i_0;

#line 85
        }

#line 83
        float ov_6 = (simd_shuffle_down((v_0), (2U)));
        uint oi_6 = (simd_shuffle_down((ix_0), (2U)));
        if(ov_6 > v_0)
        {

#line 85
            v_0 = ov_6;

#line 85
            ix_0 = oi_6;

#line 85
        }

#line 83
        float ov_7 = (simd_shuffle_down((v_0), (1U)));
        uint oi_7 = (simd_shuffle_down((ix_0), (1U)));
        if(ov_7 > v_0)
        {

#line 85
            ix_0 = oi_7;

#line 85
        }

#line 90
        if(_S1)
        {

#line 90
            (*kernelContext_0->scratch_i_0)[int(0)] = ix_0;

#line 90
        }

#line 78
    }

#line 92
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S3 = (*kernelContext_0->scratch_i_0)[int(0)];

#line 119
    return _S3;
}



[[kernel]] void argmax_f32(uint3 lid_0 [[thread_position_in_threadgroup]], packed_uint2 device* par_buf_1 [[buffer(2)]], float device* x_buf_1 [[buffer(0)]], uint device* out_buf_1 [[buffer(1)]])
{

#line 124
    thread KernelContext_0 kernelContext_1;

#line 124
    (&kernelContext_1)->par_buf_0 = par_buf_1;

#line 124
    (&kernelContext_1)->x_buf_0 = x_buf_1;

#line 124
    (&kernelContext_1)->out_buf_0 = out_buf_1;

#line 124
    threadgroup array<float, int(256)> scratch_v_1;

#line 124
    (&kernelContext_1)->scratch_v_0 = &scratch_v_1;

#line 124
    threadgroup array<uint, int(256)> scratch_i_1;

#line 124
    (&kernelContext_1)->scratch_i_0 = &scratch_i_1;
    uint tid_1 = lid_0.x;
    uint _S4 = (uint2(*(par_buf_1+int(0))) ).x;

#line 126
    float local_max_0 = -3.4028234663852886e+38f;

#line 126
    uint local_idx_0 = 0U;

#line 126
    uint i_0 = tid_1;

#line 132
    for(;;)
    {

#line 132
        if(i_0 < _S4)
        {
        }
        else
        {

#line 132
            break;
        }

#line 133
        float v_1 = (&kernelContext_1)->x_buf_0[i_0];
        if(v_1 > local_max_0)
        {

#line 134
            local_max_0 = v_1;

#line 134
            local_idx_0 = i_0;

#line 134
        }

#line 132
        i_0 = i_0 + 256U;

#line 132
    }

#line 132
    uint _S5 = block_argmax_0(tid_1, local_max_0, local_idx_0, &kernelContext_1);

#line 141
    if(tid_1 == 0U)
    {
        *((&kernelContext_1)->out_buf_0+(uint2(*((&kernelContext_1)->par_buf_0+int(0))) ).y) = _S5;

#line 141
    }



    return;
}

