#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 49 "cera/src/backend/shaders/slang/conv1d.slang"
struct KernelContext_0
{
    uint device* par_buf_0;
    float device* rbuf_0;
    float device* w_buf_0;
    float device* in_buf_0;
    float device* out_buf_0;
};


#line 36
[[kernel]] void conv1d_depthwise(uint3 gid_0 [[thread_position_in_grid]], uint device* par_buf_1 [[buffer(4)]], float device* rbuf_1 [[buffer(1)]], float device* w_buf_1 [[buffer(2)]], float device* in_buf_1 [[buffer(0)]], float device* out_buf_1 [[buffer(3)]])
{

#line 36
    thread KernelContext_0 kernelContext_0;

#line 36
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 36
    (&kernelContext_0)->rbuf_0 = rbuf_1;

#line 36
    (&kernelContext_0)->w_buf_0 = w_buf_1;

#line 36
    (&kernelContext_0)->in_buf_0 = in_buf_1;

#line 36
    (&kernelContext_0)->out_buf_0 = out_buf_1;
    uint ch_0 = gid_0.x;
    uint hs_0 = par_buf_1[int(0)];
    uint ks_0 = par_buf_1[int(1)];
    uint d_conv_0 = par_buf_1[int(2)];

    if(ch_0 >= hs_0)
    {

#line 43
        return;
    }

#line 43
    uint k_0 = 0U;

#line 43
    float sum_0 = 0.0f;

#line 48
    for(;;)
    {

#line 48
        if(k_0 < d_conv_0)
        {
        }
        else
        {

#line 48
            break;
        }

#line 49
        float sum_1 = sum_0 + *((&kernelContext_0)->rbuf_0+(k_0 * hs_0 + ch_0)) * (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + k_0];

#line 48
        k_0 = k_0 + 1U;

#line 48
        sum_0 = sum_1;

#line 48
    }



    *((&kernelContext_0)->out_buf_0+ch_0) = sum_0 + (&kernelContext_0)->in_buf_0[ch_0] * (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + d_conv_0];

#line 57
    if(d_conv_0 > 1U)
    {

#line 57
        k_0 = 0U;
        for(;;)
        {

#line 58
            if(k_0 < (d_conv_0 - 1U))
            {
            }
            else
            {

#line 58
                break;
            }

#line 59
            uint _S1 = k_0 + 1U;

#line 59
            *((&kernelContext_0)->rbuf_0+(k_0 * hs_0 + ch_0)) = *((&kernelContext_0)->rbuf_0+(_S1 * hs_0 + ch_0));

#line 58
            k_0 = _S1;

#line 58
        }

#line 57
    }

#line 62
    if(d_conv_0 > 0U)
    {

#line 63
        *((&kernelContext_0)->rbuf_0+((d_conv_0 - 1U) * hs_0 + ch_0)) = (&kernelContext_0)->in_buf_0[ch_0];

#line 62
    }


    return;
}

