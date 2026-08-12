#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 67 "cera/src/backend/shaders/slang/stft_frame.slang"
struct KernelContext_0
{
    packed_uint4 device* par_buf_0;
    float device* pcm_buf_0;
    float device* frames_buf_0;
    float device* hann_buf_0;
};


#line 39
[[kernel]] void stft_frame(uint3 gid_0 [[thread_position_in_grid]], packed_uint4 device* par_buf_1 [[buffer(3)]], float device* pcm_buf_1 [[buffer(0)]], float device* frames_buf_1 [[buffer(2)]], float device* hann_buf_1 [[buffer(1)]])
{

#line 39
    thread KernelContext_0 kernelContext_0;

#line 39
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 39
    (&kernelContext_0)->pcm_buf_0 = pcm_buf_1;

#line 39
    (&kernelContext_0)->frames_buf_0 = frames_buf_1;

#line 39
    (&kernelContext_0)->hann_buf_0 = hann_buf_1;

    uint n_fft_0 = (uint4(*(par_buf_1+int(0))) ).y;
    uint hop_0 = (uint4(*(par_buf_1+int(0))) ).z;
    uint center_pad_0 = (uint4(*(par_buf_1+int(0))) ).w;
    uint n_samples_0 = (uint4(*(par_buf_1+int(1))) ).x;
    float preemph_0 = (as_type<float>(((uint4(*(par_buf_1+int(1))) ).y)));


    uint idx_0 = gid_0.x;
    if(idx_0 >= ((uint4(*(par_buf_1+int(0))) ).x * n_fft_0))
    {

#line 50
        return;
    }

    uint t_0 = idx_0 / n_fft_0;
    uint n_0 = idx_0 - t_0 * n_fft_0;



    int g_0 = int(t_0 * hop_0 + n_0) - int(center_pad_0);

#line 58
    bool _S1;


    if(g_0 >= int(0))
    {

#line 61
        _S1 = g_0 < int(n_samples_0);

#line 61
    }
    else
    {

#line 61
        _S1 = false;

#line 61
    }

#line 61
    float s_0;

#line 61
    if(_S1)
    {

#line 62
        float s_1 = (&kernelContext_0)->pcm_buf_0[g_0];
        if(g_0 > int(0))
        {

#line 63
            s_0 = s_1 - preemph_0 * (&kernelContext_0)->pcm_buf_0[g_0 - int(1)];

#line 63
        }
        else
        {

#line 63
            s_0 = s_1;

#line 63
        }

#line 61
    }
    else
    {

#line 61
        s_0 = 0.0f;

#line 61
    }

#line 67
    *((&kernelContext_0)->frames_buf_0+idx_0) = (&kernelContext_0)->hann_buf_0[n_0] * s_0;
    return;
}

