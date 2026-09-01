#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 41 "cera/src/backend/shaders/slang/conv1d_fused.slang"
struct KernelContext_0
{
    uint device* par_buf_0;
    float device* proj_buf_0;
    float device* rbuf_0;
    float device* w_buf_0;
    float device* out_buf_0;
};


#line 24
[[kernel]] void conv1d_fused(uint3 gid_0 [[thread_position_in_grid]], uint device* par_buf_1 [[buffer(4)]], float device* proj_buf_1 [[buffer(0)]], float device* rbuf_1 [[buffer(1)]], float device* w_buf_1 [[buffer(2)]], float device* out_buf_1 [[buffer(3)]])
{

#line 24
    thread KernelContext_0 kernelContext_0;

#line 24
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 24
    (&kernelContext_0)->proj_buf_0 = proj_buf_1;

#line 24
    (&kernelContext_0)->rbuf_0 = rbuf_1;

#line 24
    (&kernelContext_0)->w_buf_0 = w_buf_1;

#line 24
    (&kernelContext_0)->out_buf_0 = out_buf_1;
    uint ch_0 = gid_0.x;
    uint hs_0 = par_buf_1[int(0)];
    uint ks_0 = par_buf_1[int(1)];
    uint d_conv_0 = par_buf_1[int(2)];

    if(ch_0 >= hs_0)
    {

#line 31
        return;
    }


    float c_val_0 = (&kernelContext_0)->proj_buf_0[hs_0 + ch_0];

    float bx_0 = (&kernelContext_0)->proj_buf_0[ch_0] * (&kernelContext_0)->proj_buf_0[2U * hs_0 + ch_0];

#line 37
    uint k_0 = 0U;

#line 37
    float sum_0 = 0.0f;


    for(;;)
    {

#line 40
        if(k_0 < d_conv_0)
        {
        }
        else
        {

#line 40
            break;
        }

#line 41
        float sum_1 = sum_0 + *((&kernelContext_0)->rbuf_0+(k_0 * hs_0 + ch_0)) * (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + k_0];

#line 40
        k_0 = k_0 + 1U;

#line 40
        sum_0 = sum_1;

#line 40
    }


    float sum_2 = sum_0 + bx_0 * (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + d_conv_0];

    if(d_conv_0 > 1U)
    {

#line 45
        k_0 = 0U;
        for(;;)
        {

#line 46
            if(k_0 < (d_conv_0 - 1U))
            {
            }
            else
            {

#line 46
                break;
            }

#line 47
            uint _S1 = k_0 + 1U;

#line 47
            *((&kernelContext_0)->rbuf_0+(k_0 * hs_0 + ch_0)) = *((&kernelContext_0)->rbuf_0+(_S1 * hs_0 + ch_0));

#line 46
            k_0 = _S1;

#line 46
        }

#line 45
    }

#line 50
    if(d_conv_0 > 0U)
    {

#line 51
        *((&kernelContext_0)->rbuf_0+((d_conv_0 - 1U) * hs_0 + ch_0)) = bx_0;

#line 50
    }



    *((&kernelContext_0)->out_buf_0+ch_0) = c_val_0 * sum_2;
    return;
}

