#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 65 "cera/src/backend/shaders/slang/conv1d_fused_batch.slang"
struct KernelContext_0
{
    uint device* par_buf_0;
    float device* w_buf_0;
    float device* rbuf_0;
    float device* proj_buf_0;
    float device* out_buf_0;
};


#line 70
[[kernel]] void conv1d_fused_batch(uint3 gid_0 [[thread_position_in_grid]], uint device* par_buf_1 [[buffer(4)]], float device* w_buf_1 [[buffer(2)]], float device* rbuf_1 [[buffer(1)]], float device* proj_buf_1 [[buffer(0)]], float device* out_buf_1 [[buffer(3)]])
{

#line 70
    thread KernelContext_0 kernelContext_0;

#line 70
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 70
    (&kernelContext_0)->w_buf_0 = w_buf_1;

#line 70
    (&kernelContext_0)->rbuf_0 = rbuf_1;

#line 70
    (&kernelContext_0)->proj_buf_0 = proj_buf_1;

#line 70
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

#line 80
        return;
    }

#line 80
    bool _S4;



    if(ks_0 > 8U)
    {

#line 84
        _S4 = true;

#line 84
    }
    else
    {

#line 84
        _S4 = d_conv_0 > 7U;

#line 84
    }

#line 84
    if(_S4)
    {

#line 85
        return;
    }

#line 101
    thread array<float, int(8)> w_local_0;

#line 101
    w_local_0[int(0)] = 0.0f;

#line 101
    w_local_0[int(1)] = 0.0f;

#line 101
    w_local_0[int(2)] = 0.0f;

#line 101
    w_local_0[int(3)] = 0.0f;

#line 101
    w_local_0[int(4)] = 0.0f;

#line 101
    w_local_0[int(5)] = 0.0f;

#line 101
    w_local_0[int(6)] = 0.0f;

#line 101
    w_local_0[int(7)] = 0.0f;


    if(0U <= d_conv_0)
    {

#line 104
        _S4 = 0U < ks_0;

#line 104
    }
    else
    {

#line 104
        _S4 = false;

#line 104
    }

#line 104
    if(_S4)
    {

#line 105
        w_local_0[0U] = (&kernelContext_0)->w_buf_0[ch_0 * ks_0];

#line 104
    }

#line 104
    if(1U <= d_conv_0)
    {

#line 104
        _S4 = 1U < ks_0;

#line 104
    }
    else
    {

#line 104
        _S4 = false;

#line 104
    }

#line 104
    if(_S4)
    {

#line 105
        w_local_0[1U] = (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + 1U];

#line 104
    }

#line 104
    if(2U <= d_conv_0)
    {

#line 104
        _S4 = 2U < ks_0;

#line 104
    }
    else
    {

#line 104
        _S4 = false;

#line 104
    }

#line 104
    if(_S4)
    {

#line 105
        w_local_0[2U] = (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + 2U];

#line 104
    }

#line 104
    if(3U <= d_conv_0)
    {

#line 104
        _S4 = 3U < ks_0;

#line 104
    }
    else
    {

#line 104
        _S4 = false;

#line 104
    }

#line 104
    if(_S4)
    {

#line 105
        w_local_0[3U] = (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + 3U];

#line 104
    }

#line 104
    if(4U <= d_conv_0)
    {

#line 104
        _S4 = 4U < ks_0;

#line 104
    }
    else
    {

#line 104
        _S4 = false;

#line 104
    }

#line 104
    if(_S4)
    {

#line 105
        w_local_0[4U] = (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + 4U];

#line 104
    }

#line 104
    if(5U <= d_conv_0)
    {

#line 104
        _S4 = 5U < ks_0;

#line 104
    }
    else
    {

#line 104
        _S4 = false;

#line 104
    }

#line 104
    if(_S4)
    {

#line 105
        w_local_0[5U] = (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + 5U];

#line 104
    }

#line 104
    if(6U <= d_conv_0)
    {

#line 104
        _S4 = 6U < ks_0;

#line 104
    }
    else
    {

#line 104
        _S4 = false;

#line 104
    }

#line 104
    if(_S4)
    {

#line 105
        w_local_0[6U] = (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + 6U];

#line 104
    }

#line 104
    if(7U <= d_conv_0)
    {

#line 104
        _S4 = 7U < ks_0;

#line 104
    }
    else
    {

#line 104
        _S4 = false;

#line 104
    }

#line 104
    if(_S4)
    {

#line 105
        w_local_0[7U] = (&kernelContext_0)->w_buf_0[ch_0 * ks_0 + 7U];

#line 104
    }



    thread array<float, int(7)> rb_0;

#line 108
    rb_0[int(0)] = 0.0f;

#line 108
    rb_0[int(1)] = 0.0f;

#line 108
    rb_0[int(2)] = 0.0f;

#line 108
    rb_0[int(3)] = 0.0f;

#line 108
    rb_0[int(4)] = 0.0f;

#line 108
    rb_0[int(5)] = 0.0f;

#line 108
    rb_0[int(6)] = 0.0f;


    bool _S5 = 0U < d_conv_0;

#line 111
    if(_S5)
    {

#line 112
        rb_0[0U] = *((&kernelContext_0)->rbuf_0+ch_0);

#line 111
    }

#line 111
    bool _S6 = 1U < d_conv_0;

#line 111
    if(_S6)
    {

#line 112
        rb_0[1U] = *((&kernelContext_0)->rbuf_0+(hs_0 + ch_0));

#line 111
    }

#line 111
    bool _S7 = 2U < d_conv_0;

#line 111
    if(_S7)
    {

#line 112
        rb_0[2U] = *((&kernelContext_0)->rbuf_0+(2U * hs_0 + ch_0));

#line 111
    }

#line 111
    bool _S8 = 3U < d_conv_0;

#line 111
    if(_S8)
    {

#line 112
        rb_0[3U] = *((&kernelContext_0)->rbuf_0+(3U * hs_0 + ch_0));

#line 111
    }

#line 111
    bool _S9 = 4U < d_conv_0;

#line 111
    if(_S9)
    {

#line 112
        rb_0[4U] = *((&kernelContext_0)->rbuf_0+(4U * hs_0 + ch_0));

#line 111
    }

#line 111
    bool _S10 = 5U < d_conv_0;

#line 111
    if(_S10)
    {

#line 112
        rb_0[5U] = *((&kernelContext_0)->rbuf_0+(5U * hs_0 + ch_0));

#line 111
    }

#line 111
    bool _S11 = 6U < d_conv_0;

#line 111
    if(_S11)
    {

#line 112
        rb_0[6U] = *((&kernelContext_0)->rbuf_0+(6U * hs_0 + ch_0));

#line 111
    }

#line 110
    uint t_0 = 0U;

#line 116
    for(;;)
    {

#line 116
        if(t_0 < _S1)
        {
        }
        else
        {

#line 116
            break;
        }

#line 117
        uint base_0 = t_0 * _S2;

#line 123
        float c_val_0 = (&kernelContext_0)->proj_buf_0[base_0 + hs_0 + ch_0];

        float bx_0 = (&kernelContext_0)->proj_buf_0[base_0 + ch_0] * (&kernelContext_0)->proj_buf_0[base_0 + 2U * hs_0 + ch_0];

#line 125
        float sum_0;

#line 130
        if(_S5)
        {

#line 130
            sum_0 = rb_0[0U] * w_local_0[0U];

#line 130
        }
        else
        {

#line 130
            sum_0 = 0.0f;

#line 130
        }

#line 129
        float sum_1;
        if(_S6)
        {

#line 130
            sum_1 = sum_0 + rb_0[1U] * w_local_0[1U];

#line 130
        }
        else
        {

#line 130
            sum_1 = sum_0;

#line 130
        }

#line 129
        float sum_2;
        if(_S7)
        {

#line 130
            sum_2 = sum_1 + rb_0[2U] * w_local_0[2U];

#line 130
        }
        else
        {

#line 130
            sum_2 = sum_1;

#line 130
        }

#line 129
        float sum_3;
        if(_S8)
        {

#line 130
            sum_3 = sum_2 + rb_0[3U] * w_local_0[3U];

#line 130
        }
        else
        {

#line 130
            sum_3 = sum_2;

#line 130
        }

#line 129
        float sum_4;
        if(_S9)
        {

#line 130
            sum_4 = sum_3 + rb_0[4U] * w_local_0[4U];

#line 130
        }
        else
        {

#line 130
            sum_4 = sum_3;

#line 130
        }

#line 129
        float sum_5;
        if(_S10)
        {

#line 130
            sum_5 = sum_4 + rb_0[5U] * w_local_0[5U];

#line 130
        }
        else
        {

#line 130
            sum_5 = sum_4;

#line 130
        }

#line 129
        float sum_6;
        if(_S11)
        {

#line 130
            sum_6 = sum_5 + rb_0[6U] * w_local_0[6U];

#line 130
        }
        else
        {

#line 130
            sum_6 = sum_5;

#line 130
        }

#line 140
        float sum_7 = sum_6 + bx_0 * w_local_0[d_conv_0];

#line 145
        if(_S6)
        {

#line 146
            rb_0[0U] = rb_0[1U];

#line 145
        }

#line 145
        if(_S7)
        {

#line 146
            rb_0[1U] = rb_0[2U];

#line 145
        }

#line 145
        if(_S8)
        {

#line 146
            rb_0[2U] = rb_0[3U];

#line 145
        }

#line 145
        if(_S9)
        {

#line 146
            rb_0[3U] = rb_0[4U];

#line 145
        }

#line 145
        if(_S10)
        {

#line 146
            rb_0[4U] = rb_0[5U];

#line 145
        }

#line 145
        if(_S11)
        {

#line 146
            rb_0[5U] = rb_0[6U];

#line 145
        }



        if(d_conv_0 > 0U)
        {

#line 150
            rb_0[d_conv_0 - 1U] = bx_0;

#line 149
        }



        *((&kernelContext_0)->out_buf_0+(t_0 * _S3 + ch_0)) = c_val_0 * sum_7;

#line 116
        t_0 = t_0 + 1U;

#line 116
    }

#line 159
    if(_S5)
    {

#line 160
        *((&kernelContext_0)->rbuf_0+ch_0) = rb_0[0U];

#line 159
    }

#line 159
    if(_S6)
    {

#line 160
        *((&kernelContext_0)->rbuf_0+(hs_0 + ch_0)) = rb_0[1U];

#line 159
    }

#line 159
    if(_S7)
    {

#line 160
        *((&kernelContext_0)->rbuf_0+(2U * hs_0 + ch_0)) = rb_0[2U];

#line 159
    }

#line 159
    if(_S8)
    {

#line 160
        *((&kernelContext_0)->rbuf_0+(3U * hs_0 + ch_0)) = rb_0[3U];

#line 159
    }

#line 159
    if(_S9)
    {

#line 160
        *((&kernelContext_0)->rbuf_0+(4U * hs_0 + ch_0)) = rb_0[4U];

#line 159
    }

#line 159
    if(_S10)
    {

#line 160
        *((&kernelContext_0)->rbuf_0+(5U * hs_0 + ch_0)) = rb_0[5U];

#line 159
    }

#line 159
    if(_S11)
    {

#line 160
        *((&kernelContext_0)->rbuf_0+(6U * hs_0 + ch_0)) = rb_0[6U];

#line 159
    }



    return;
}

