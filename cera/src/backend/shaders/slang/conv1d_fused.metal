#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 73 "cera/src/backend/shaders/slang/conv1d_fused.slang"
struct KernelContext_0
{
    uint device* par_buf_0;
    float device* proj_buf_0;
    float device* rbuf_0;
    float device* w_buf_0;
    float device* out_buf_0;
};


#line 51
[[kernel]] void conv1d_fused(uint3 gid_0 [[thread_position_in_grid]], uint device* par_buf_1 [[buffer(4)]], float device* proj_buf_1 [[buffer(0)]], float device* rbuf_1 [[buffer(1)]], float device* w_buf_1 [[buffer(2)]], float device* out_buf_1 [[buffer(3)]])
{

#line 51
    thread KernelContext_0 kernelContext_0;

#line 51
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 51
    (&kernelContext_0)->proj_buf_0 = proj_buf_1;

#line 51
    (&kernelContext_0)->rbuf_0 = rbuf_1;

#line 51
    (&kernelContext_0)->w_buf_0 = w_buf_1;

#line 51
    (&kernelContext_0)->out_buf_0 = out_buf_1;
    uint ch_0 = gid_0.x;
    uint hs_0 = par_buf_1[int(0)];
    uint ks_0 = par_buf_1[int(1)];
    uint d_conv_0 = par_buf_1[int(2)];

    if(ch_0 >= hs_0)
    {

#line 58
        return;
    }

#line 66
    float c_val_0 = (&kernelContext_0)->proj_buf_0[hs_0 + ch_0];

    float bx_0 = (&kernelContext_0)->proj_buf_0[ch_0] * (&kernelContext_0)->proj_buf_0[2U * hs_0 + ch_0];

#line 68
    uint k_0 = 0U;

#line 68
    float sum_0 = 0.0f;



    for(;;)
    {

#line 72
        if(k_0 < d_conv_0)
        {
        }
        else
        {

#line 72
            break;
        }

#line 73
        float sum_1 = sum_0 + *((&kernelContext_0)->rbuf_0+(k_0 * hs_0 + ch_0)) * (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + k_0];

#line 72
        k_0 = k_0 + 1U;

#line 72
        sum_0 = sum_1;

#line 72
    }


    float sum_2 = sum_0 + bx_0 * (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + d_conv_0];


    if(d_conv_0 > 1U)
    {

#line 78
        k_0 = 0U;
        for(;;)
        {

#line 79
            if(k_0 < (d_conv_0 - 1U))
            {
            }
            else
            {

#line 79
                break;
            }

#line 80
            uint _S1 = k_0 + 1U;

#line 80
            *((&kernelContext_0)->rbuf_0+(k_0 * hs_0 + ch_0)) = *((&kernelContext_0)->rbuf_0+(_S1 * hs_0 + ch_0));

#line 79
            k_0 = _S1;

#line 79
        }

#line 78
    }

#line 83
    if(d_conv_0 > 0U)
    {

#line 84
        *((&kernelContext_0)->rbuf_0+((d_conv_0 - 1U) * hs_0 + ch_0)) = bx_0;

#line 83
    }



    *((&kernelContext_0)->out_buf_0+ch_0) = c_val_0 * sum_2;
    return;
}

