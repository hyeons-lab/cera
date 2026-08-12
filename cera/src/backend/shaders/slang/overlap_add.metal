#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 60 "cera/src/backend/shaders/slang/overlap_add.slang"
struct KernelContext_0
{
    packed_uint4 device* par_buf_0;
    float device* hann_buf_0;
    float device* td_buf_0;
    float device* out_buf_0;
};


#line 33
[[kernel]] void overlap_add(uint3 gid_0 [[thread_position_in_grid]], packed_uint4 device* par_buf_1 [[buffer(3)]], float device* hann_buf_1 [[buffer(1)]], float device* td_buf_1 [[buffer(0)]], float device* out_buf_1 [[buffer(2)]])
{

#line 33
    thread KernelContext_0 kernelContext_0;

#line 33
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 33
    (&kernelContext_0)->hann_buf_0 = hann_buf_1;

#line 33
    (&kernelContext_0)->td_buf_0 = td_buf_1;

#line 33
    (&kernelContext_0)->out_buf_0 = out_buf_1;
    uint n_frames_0 = (uint4(*(par_buf_1+int(0))) ).x;
    uint n_fft_0 = (uint4(*(par_buf_1+int(0))) ).y;
    uint hop_0 = (uint4(*(par_buf_1+int(0))) ).z;

    uint g_0 = gid_0.x;
    if(g_0 >= (n_frames_0 * hop_0))
    {

#line 40
        return;
    }

    uint i_hi_0 = g_0 / hop_0;

#line 43
    uint i_hi_1;
    if(i_hi_0 >= n_frames_0)
    {

#line 44
        i_hi_1 = n_frames_0 - 1U;

#line 44
    }
    else
    {

#line 44
        i_hi_1 = i_hi_0;

#line 44
    }

#line 44
    uint i_lo_0;



    if(g_0 >= n_fft_0)
    {

#line 49
        uint _S1 = (g_0 - n_fft_0) / hop_0;

#line 49
        i_lo_0 = _S1 + 1U;

#line 48
    }
    else
    {

#line 48
        i_lo_0 = 0U;

#line 48
    }

#line 48
    uint i_0 = i_lo_0;

#line 48
    float numer_0 = 0.0f;

#line 48
    float denom_0 = 0.0f;

#line 54
    for(;;)
    {

#line 54
        if(i_0 <= i_hi_1)
        {
        }
        else
        {

#line 54
            break;
        }

#line 55
        uint local_0 = g_0 - i_0 * hop_0;
        float w_0 = (&kernelContext_0)->hann_buf_0[local_0];
        float numer_1 = numer_0 + (&kernelContext_0)->td_buf_0[i_0 * n_fft_0 + local_0] * w_0;
        float denom_1 = denom_0 + w_0 * w_0;

#line 54
        i_0 = i_0 + 1U;

#line 54
        numer_0 = numer_1;

#line 54
        denom_0 = denom_1;

#line 54
    }

#line 60
    float device* _S2 = (&kernelContext_0)->out_buf_0+g_0;

#line 60
    if(denom_0 > 9.99999993922529029e-09f)
    {

#line 60
        numer_0 = numer_0 / denom_0;

#line 60
    }

#line 60
    *_S2 = numer_0;
    return;
}

