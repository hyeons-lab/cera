#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 55 "cera/src/backend/shaders/slang/mel_project.slang"
struct KernelContext_0
{
    packed_uint4 device* par_buf_0;
    float device* power_buf_0;
    float device* filter_buf_0;
    float device* mel_buf_0;
};


#line 33
[[kernel]] void mel_project(uint3 gid_0 [[thread_position_in_grid]], packed_uint4 device* par_buf_1 [[buffer(3)]], float device* power_buf_1 [[buffer(0)]], float device* filter_buf_1 [[buffer(1)]], float device* mel_buf_1 [[buffer(2)]])
{

#line 33
    thread KernelContext_0 kernelContext_0;

#line 33
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 33
    (&kernelContext_0)->power_buf_0 = power_buf_1;

#line 33
    (&kernelContext_0)->filter_buf_0 = filter_buf_1;

#line 33
    (&kernelContext_0)->mel_buf_0 = mel_buf_1;

    uint n_frames_0 = (uint4(*(par_buf_1+int(0))) ).y;
    uint n_bins_0 = (uint4(*(par_buf_1+int(0))) ).z;
    float eps_0 = (as_type<float>(((uint4(*(par_buf_1+int(0))) ).w)));


    uint idx_0 = gid_0.x;
    if(idx_0 >= ((uint4(*(par_buf_1+int(0))) ).x * n_frames_0))
    {

#line 42
        return;
    }


    uint mi_0 = idx_0 / n_frames_0;


    uint _S1 = (idx_0 - mi_0 * n_frames_0) * n_bins_0;
    uint _S2 = mi_0 * n_bins_0;

#line 50
    uint k_0 = 0U;

#line 50
    float sum_0 = 0.0f;

    for(;;)
    {

#line 52
        if(k_0 < n_bins_0)
        {
        }
        else
        {

#line 52
            break;
        }

#line 53
        float sum_1 = sum_0 + (&kernelContext_0)->power_buf_0[_S1 + k_0] * (&kernelContext_0)->filter_buf_0[_S2 + k_0];

#line 52
        k_0 = k_0 + 1U;

#line 52
        sum_0 = sum_1;

#line 52
    }


    *((&kernelContext_0)->mel_buf_0+idx_0) = log(sum_0 + eps_0);
    return;
}

