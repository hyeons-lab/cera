#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 109 "cera/src/backend/shaders/slang/conv1d_fused_batch.slang"
struct KernelContext_0
{
    uint device* par_buf_0;
    float device* w_buf_0;
    float device* rbuf_0;
    float device* proj_buf_0;
    float device* out_buf_0;
};


#line 67
[[kernel]] void conv1d_fused_batch(uint3 gid_0 [[thread_position_in_grid]], uint device* par_buf_1 [[buffer(4)]], float device* w_buf_1 [[buffer(2)]], float device* rbuf_1 [[buffer(1)]], float device* proj_buf_1 [[buffer(0)]], float device* out_buf_1 [[buffer(3)]])
{

#line 67
    thread KernelContext_0 kernelContext_0;

#line 67
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 67
    (&kernelContext_0)->w_buf_0 = w_buf_1;

#line 67
    (&kernelContext_0)->rbuf_0 = rbuf_1;

#line 67
    (&kernelContext_0)->proj_buf_0 = proj_buf_1;

#line 67
    (&kernelContext_0)->out_buf_0 = out_buf_1;
    uint ch_0 = gid_0.x;
    uint hs_0 = par_buf_1[int(0)];
    uint ks_0 = par_buf_1[int(1)];
    uint d_conv_0 = par_buf_1[int(2)];
    uint _S1 = par_buf_1[int(3)];
    uint _S2 = par_buf_1[int(4)];
    uint _S3 = par_buf_1[int(5)];

    if(ch_0 >= hs_0)
    {

#line 77
        return;
    }

#line 77
    bool _S4;



    if(ks_0 > 4U)
    {

#line 81
        _S4 = true;

#line 81
    }
    else
    {

#line 81
        _S4 = d_conv_0 > 3U;

#line 81
    }

#line 81
    if(_S4)
    {

#line 82
        return;
    }

#line 98
    thread array<float, int(4)> w_local_0;

#line 98
    w_local_0[int(0)] = 0.0f;

#line 98
    w_local_0[int(1)] = 0.0f;

#line 98
    w_local_0[int(2)] = 0.0f;

#line 98
    w_local_0[int(3)] = 0.0f;


    if(0U <= d_conv_0)
    {

#line 101
        _S4 = 0U < ks_0;

#line 101
    }
    else
    {

#line 101
        _S4 = false;

#line 101
    }

#line 101
    if(_S4)
    {

#line 102
        w_local_0[0U] = (&kernelContext_0)->w_buf_0[ch_0 * ks_0];

#line 101
    }

#line 101
    if(1U <= d_conv_0)
    {

#line 101
        _S4 = 1U < ks_0;

#line 101
    }
    else
    {

#line 101
        _S4 = false;

#line 101
    }

#line 101
    if(_S4)
    {

#line 102
        w_local_0[1U] = (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + 1U];

#line 101
    }

#line 101
    if(2U <= d_conv_0)
    {

#line 101
        _S4 = 2U < ks_0;

#line 101
    }
    else
    {

#line 101
        _S4 = false;

#line 101
    }

#line 101
    if(_S4)
    {

#line 102
        w_local_0[2U] = (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + 2U];

#line 101
    }

#line 101
    if(3U <= d_conv_0)
    {

#line 101
        _S4 = 3U < ks_0;

#line 101
    }
    else
    {

#line 101
        _S4 = false;

#line 101
    }

#line 101
    if(_S4)
    {

#line 102
        w_local_0[3U] = (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + 3U];

#line 101
    }



    thread array<float, int(3)> rb_0;

#line 105
    rb_0[int(0)] = 0.0f;

#line 105
    rb_0[int(1)] = 0.0f;

#line 105
    rb_0[int(2)] = 0.0f;


    bool _S5 = 0U < d_conv_0;

#line 108
    if(_S5)
    {

#line 109
        rb_0[0U] = *((&kernelContext_0)->rbuf_0+ch_0);

#line 108
    }

#line 108
    bool _S6 = 1U < d_conv_0;

#line 108
    if(_S6)
    {

#line 109
        rb_0[1U] = *((&kernelContext_0)->rbuf_0+(hs_0 + ch_0));

#line 108
    }

#line 108
    bool _S7 = 2U < d_conv_0;

#line 108
    if(_S7)
    {

#line 109
        rb_0[2U] = *((&kernelContext_0)->rbuf_0+(2U * hs_0 + ch_0));

#line 108
    }

#line 107
    uint t_0 = 0U;

#line 113
    for(;;)
    {

#line 113
        if(t_0 < _S1)
        {
        }
        else
        {

#line 113
            break;
        }

#line 114
        uint base_0 = t_0 * _S2;

#line 120
        float c_val_0 = (&kernelContext_0)->proj_buf_0[base_0 + hs_0 + ch_0];

        float bx_0 = (&kernelContext_0)->proj_buf_0[base_0 + ch_0] * (&kernelContext_0)->proj_buf_0[base_0 + 2U * hs_0 + ch_0];

#line 122
        float sum_0;

#line 127
        if(_S5)
        {

#line 127
            sum_0 = rb_0[0U] * w_local_0[0U];

#line 127
        }
        else
        {

#line 127
            sum_0 = 0.0f;

#line 127
        }

#line 126
        float sum_1;
        if(_S6)
        {

#line 127
            sum_1 = sum_0 + rb_0[1U] * w_local_0[1U];

#line 127
        }
        else
        {

#line 127
            sum_1 = sum_0;

#line 127
        }

#line 126
        float sum_2;
        if(_S7)
        {

#line 127
            sum_2 = sum_1 + rb_0[2U] * w_local_0[2U];

#line 127
        }
        else
        {

#line 127
            sum_2 = sum_1;

#line 127
        }

#line 137
        float sum_3 = sum_2 + bx_0 * w_local_0[d_conv_0];

#line 142
        if(_S6)
        {

#line 143
            rb_0[0U] = rb_0[1U];

#line 142
        }

#line 142
        if(_S7)
        {

#line 143
            rb_0[1U] = rb_0[2U];

#line 142
        }



        if(d_conv_0 > 0U)
        {

#line 147
            rb_0[d_conv_0 - 1U] = bx_0;

#line 146
        }



        *((&kernelContext_0)->out_buf_0+(t_0 * _S3 + ch_0)) = c_val_0 * sum_3;

#line 113
        t_0 = t_0 + 1U;

#line 113
    }

#line 156
    if(_S5)
    {

#line 157
        *((&kernelContext_0)->rbuf_0+ch_0) = rb_0[0U];

#line 156
    }

#line 156
    if(_S6)
    {

#line 157
        *((&kernelContext_0)->rbuf_0+(hs_0 + ch_0)) = rb_0[1U];

#line 156
    }

#line 156
    if(_S7)
    {

#line 157
        *((&kernelContext_0)->rbuf_0+(2U * hs_0 + ch_0)) = rb_0[2U];

#line 156
    }



    return;
}

