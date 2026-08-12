#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 83 "cera/src/backend/shaders/slang/power_spec.slang"
struct KernelContext_0
{
    packed_uint4 device* par_buf_0;
    float device* frames_buf_0;
    float device* twiddle_buf_0;
    float device* power_buf_0;
};


#line 56
[[kernel]] void power_spec(uint3 gid_0 [[thread_position_in_grid]], packed_uint4 device* par_buf_1 [[buffer(3)]], float device* frames_buf_1 [[buffer(0)]], float device* twiddle_buf_1 [[buffer(1)]], float device* power_buf_1 [[buffer(2)]])
{

#line 56
    thread KernelContext_0 kernelContext_0;

#line 56
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 56
    (&kernelContext_0)->frames_buf_0 = frames_buf_1;

#line 56
    (&kernelContext_0)->twiddle_buf_0 = twiddle_buf_1;

#line 56
    (&kernelContext_0)->power_buf_0 = power_buf_1;

    uint n_fft_0 = (uint4(*(par_buf_1+int(0))) ).y;
    uint n_bins_0 = (uint4(*(par_buf_1+int(0))) ).z;


    uint idx_0 = gid_0.x;
    if(idx_0 >= ((uint4(*(par_buf_1+int(0))) ).x * n_bins_0))
    {

#line 64
        return;
    }

    uint t_0 = idx_0 / n_bins_0;
    uint _S1 = idx_0 - t_0 * n_bins_0;
    uint _S2 = t_0 * n_fft_0;

#line 69
    uint n_0 = 0U;

#line 69
    uint m_0 = 0U;

#line 69
    float re_0 = 0.0f;

#line 69
    float im_0 = 0.0f;

#line 74
    for(;;)
    {

#line 74
        if(n_0 < n_fft_0)
        {
        }
        else
        {

#line 74
            break;
        }

#line 75
        float x_0 = (&kernelContext_0)->frames_buf_0[_S2 + n_0];
        uint _S3 = 2U * m_0;

#line 76
        float re_1 = re_0 + x_0 * (&kernelContext_0)->twiddle_buf_0[_S3];
        float im_1 = im_0 + x_0 * (&kernelContext_0)->twiddle_buf_0[_S3 + 1U];
        uint m_1 = m_0 + _S1;
        if(m_1 >= n_fft_0)
        {

#line 79
            m_0 = m_1 - n_fft_0;

#line 79
        }
        else
        {

#line 79
            m_0 = m_1;

#line 79
        }

#line 74
        n_0 = n_0 + 1U;

#line 74
        re_0 = re_1;

#line 74
        im_0 = im_1;

#line 74
    }

#line 83
    *((&kernelContext_0)->power_buf_0+idx_0) = re_0 * re_0 + im_0 * im_0;
    return;
}

