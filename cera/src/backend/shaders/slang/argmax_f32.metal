#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 72 "cera/src/backend/shaders/slang/argmax_f32.slang"
struct KernelContext_0
{
    packed_uint2 device* par_buf_0;
    float device* x_buf_0;
    uint device* out_buf_0;
    array<float, int(256)> threadgroup* scratch_v_0;
    array<uint, int(256)> threadgroup* scratch_i_0;
};


#line 49
uint block_argmax_0(uint tid_0, float val_0, uint idx_0, KernelContext_0 thread* kernelContext_0)
{

#line 60
    float ov_0 = (simd_shuffle_down((val_0), (16U)));
    uint oi_0 = (simd_shuffle_down((idx_0), (16U)));

#line 61
    float best_v_0;

#line 61
    uint best_i_0;
    if(ov_0 > val_0)
    {

#line 62
        best_v_0 = ov_0;

#line 62
        best_i_0 = oi_0;

#line 62
    }
    else
    {

#line 62
        best_v_0 = val_0;

#line 62
        best_i_0 = idx_0;

#line 62
    }

#line 60
    float ov_1 = (simd_shuffle_down((best_v_0), (8U)));
    uint oi_1 = (simd_shuffle_down((best_i_0), (8U)));
    if(ov_1 > best_v_0)
    {

#line 62
        best_v_0 = ov_1;

#line 62
        best_i_0 = oi_1;

#line 62
    }

#line 60
    float ov_2 = (simd_shuffle_down((best_v_0), (4U)));
    uint oi_2 = (simd_shuffle_down((best_i_0), (4U)));
    if(ov_2 > best_v_0)
    {

#line 62
        best_v_0 = ov_2;

#line 62
        best_i_0 = oi_2;

#line 62
    }

#line 60
    float ov_3 = (simd_shuffle_down((best_v_0), (2U)));
    uint oi_3 = (simd_shuffle_down((best_i_0), (2U)));
    if(ov_3 > best_v_0)
    {

#line 62
        best_v_0 = ov_3;

#line 62
        best_i_0 = oi_3;

#line 62
    }

#line 60
    float ov_4 = (simd_shuffle_down((best_v_0), (1U)));
    uint oi_4 = (simd_shuffle_down((best_i_0), (1U)));
    if(ov_4 > best_v_0)
    {

#line 62
        best_v_0 = ov_4;

#line 62
        best_i_0 = oi_4;

#line 62
    }

#line 68
    uint simd_lane_0 = tid_0 & 31U;
    uint simd_id_0 = tid_0 >> 5U;
    bool _S1 = simd_lane_0 == 0U;

#line 70
    if(_S1)
    {

#line 71
        (*kernelContext_0->scratch_v_0)[simd_id_0] = best_v_0;
        (*kernelContext_0->scratch_i_0)[simd_id_0] = best_i_0;

#line 70
    }



    threadgroup_barrier(mem_flags::mem_threadgroup);

    if(simd_id_0 == 0U)
    {

#line 77
        bool _S2 = simd_lane_0 < 8U;

#line 77
        if(_S2)
        {

#line 77
            best_v_0 = (*kernelContext_0->scratch_v_0)[simd_lane_0];

#line 77
        }
        else
        {

#line 77
            best_v_0 = -3.4028234663852886e+38f;

#line 77
        }
        if(_S2)
        {

#line 78
            best_i_0 = (*kernelContext_0->scratch_i_0)[simd_lane_0];

#line 78
        }
        else
        {

#line 78
            best_i_0 = 0U;

#line 78
        }


        float ov_5 = (simd_shuffle_down((best_v_0), (4U)));
        uint oi_5 = (simd_shuffle_down((best_i_0), (4U)));

#line 82
        float v_0;

#line 82
        uint ix_0;
        if(ov_5 > best_v_0)
        {

#line 83
            v_0 = ov_5;

#line 83
            ix_0 = oi_5;

#line 83
        }
        else
        {

#line 83
            v_0 = best_v_0;

#line 83
            ix_0 = best_i_0;

#line 83
        }

#line 81
        float ov_6 = (simd_shuffle_down((v_0), (2U)));
        uint oi_6 = (simd_shuffle_down((ix_0), (2U)));
        if(ov_6 > v_0)
        {

#line 83
            v_0 = ov_6;

#line 83
            ix_0 = oi_6;

#line 83
        }

#line 81
        float ov_7 = (simd_shuffle_down((v_0), (1U)));
        uint oi_7 = (simd_shuffle_down((ix_0), (1U)));
        if(ov_7 > v_0)
        {

#line 83
            ix_0 = oi_7;

#line 83
        }

#line 88
        if(_S1)
        {

#line 88
            (*kernelContext_0->scratch_i_0)[int(0)] = ix_0;

#line 88
        }

#line 76
    }

#line 90
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S3 = (*kernelContext_0->scratch_i_0)[int(0)];

#line 117
    return _S3;
}



[[kernel]] void argmax_f32(uint3 lid_0 [[thread_position_in_threadgroup]], packed_uint2 device* par_buf_1 [[buffer(2)]], float device* x_buf_1 [[buffer(0)]], uint device* out_buf_1 [[buffer(1)]])
{

#line 122
    thread KernelContext_0 kernelContext_1;

#line 122
    (&kernelContext_1)->par_buf_0 = par_buf_1;

#line 122
    (&kernelContext_1)->x_buf_0 = x_buf_1;

#line 122
    (&kernelContext_1)->out_buf_0 = out_buf_1;

#line 122
    threadgroup array<float, int(256)> scratch_v_1;

#line 122
    (&kernelContext_1)->scratch_v_0 = &scratch_v_1;

#line 122
    threadgroup array<uint, int(256)> scratch_i_1;

#line 122
    (&kernelContext_1)->scratch_i_0 = &scratch_i_1;
    uint tid_1 = lid_0.x;
    uint _S4 = (uint2(*(par_buf_1+int(0))) ).x;

#line 124
    float local_max_0 = -3.4028234663852886e+38f;

#line 124
    uint local_idx_0 = 0U;

#line 124
    uint i_0 = tid_1;

#line 130
    for(;;)
    {

#line 130
        if(i_0 < _S4)
        {
        }
        else
        {

#line 130
            break;
        }

#line 131
        float v_1 = (&kernelContext_1)->x_buf_0[i_0];
        if(v_1 > local_max_0)
        {

#line 132
            local_max_0 = v_1;

#line 132
            local_idx_0 = i_0;

#line 132
        }

#line 130
        i_0 = i_0 + 256U;

#line 130
    }

#line 130
    uint _S5 = block_argmax_0(tid_1, local_max_0, local_idx_0, &kernelContext_1);

#line 139
    if(tid_1 == 0U)
    {

#line 140
        *((&kernelContext_1)->out_buf_0+int(0)) = _S5;

#line 139
    }


    return;
}

